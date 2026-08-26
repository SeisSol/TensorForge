# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""The wrap-around schedule, checked on the generated text.

The interesting property is not that the pass runs but that the buffer it
wraps still holds the right element.  Four things have to hold together for
that, and each of them was a bug at some point while the pass was written:

* the declaration leaves the loop, or the value written at the tail of
  iteration ``k`` is not there at the head of ``k + 1``;
* a peeled transfer fills the buffer before the loop, addressed from the
  thread's first element;
* the in-loop transfer is addressed from the *lookahead* index;
* the wrapped write lands after the buffer's **last** read, not merely after
  its first -- a buffer read in two slots stays live between them, and
  wrapping to the first read overwrites what the second still wants.

The last one is why the distance is clamped per transfer rather than per body:
``d`` is bounded by ``n - 1 - span``, not by ``n - 1``.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

from tensorforge.common.context import Context, Options
from tensorforge.generators.generator import Generator
from tensorforge.backend.opt.slots import SlotModel, Transfer

CASES = Path(__file__).parent / "cases"

# n = 4 compute slots on both backends, with register transfers on each; the
# shape the ADER derivative chain has.
CHAIN = "chain_five.py"


def _generate(case_file: str, backend: str, arch: str, **opts) -> str:
    path = CASES / case_file
    spec = importlib.util.spec_from_file_location("tf_wrap__" + path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ctx = Context(arch=arch, backend=backend,
                  fp_type=getattr(mod, "DTYPE", None),
                  options=Options(**opts))
    gen = Generator(mod.descr_list(), ctx)
    gen.generate()
    return gen.get_kernel()


def _loop_start(lines) -> int:
    return next(i for i, l in enumerate(lines) if "for (size_t batchId0" in l)


def _wrapped_buffers(kernel: str):
    """Register buffers the pass moved, read back off the generated text."""
    return sorted({m.group(1) for m in
                   re.finditer(r'\bwrap_glb_\w+\[', kernel)} |
                  {m.group(1) for m in
                   re.finditer(r'(\br\d+)\[[^\]]*\]\s*=', kernel)})


# ---------------------------------------------------------------------- #
# the accounting
# ---------------------------------------------------------------------- #

@pytest.mark.parametrize("d,span,n,expected", [
    (1, 0, 4, 1),      # the plain case: one consumer, distance well inside
    (3, 0, 4, 1),      # d = n - 1 is still one copy
    (4, 0, 4, 2),      # d = n needs the second, which is what Pipeline does
    (2, 2, 4, 2),      # ... and so does a two-slot span at half the distance
    (1, 2, 4, 1),      # d + span + 1 == n exactly: still one
])
def test_copies_counts_the_span_not_just_the_distance(d, span, n, expected):
    assert Transfer.copies(d, n, span) == expected


# ---------------------------------------------------------------------- #
# the transform
# ---------------------------------------------------------------------- #

@pytest.mark.parametrize("backend,arch", [("hip", "gfx90a"), ("cuda", "sm_86")])
def test_pass_is_inert_when_disabled(backend, arch):
    off = _generate(CHAIN, backend, arch, enable_wrap_loads=False)
    default = _generate(CHAIN, backend, arch)
    assert off == default


@pytest.mark.parametrize("backend,arch", [("hip", "gfx90a"), ("cuda", "sm_86")])
def test_wrapped_declaration_leaves_the_loop(backend, arch):
    kernel = _generate(CHAIN, backend, arch,
                       enable_wrap_loads=True, wrap_distance=1)
    lines = kernel.splitlines()
    loop = _loop_start(lines)
    wrapped = [m.group(1) for m in re.finditer(r'wrap_glb_(\w+)', kernel)]
    assert wrapped, "expected chain_five to have something to wrap"

    for name in {m.group(1) for m in re.finditer(r'\b(r\d+)\[\d+\]\s*=', kernel)}:
        decls = [i for i, l in enumerate(lines)
                 if re.search(rf'\b(float|double) {name}\[', l)]
        writes_in_loop = [i for i, l in enumerate(lines)
                          if i > loop and re.search(rf'\b{name}\[[^\]]*\]\s*=', l)]
        writes_before = [i for i, l in enumerate(lines)
                         if i < loop and re.search(rf'\b{name}\[[^\]]*\]\s*=', l)]
        if not writes_before:
            continue                      # not a wrapped buffer
        assert decls and decls[0] < loop, (
            f'{name} is filled before the loop but declared inside it')
        assert writes_in_loop, f'{name} is peeled but never refilled'


@pytest.mark.parametrize("backend,arch", [("hip", "gfx90a"), ("cuda", "sm_86")])
def test_peeled_and_wrapped_transfers_use_the_right_element(backend, arch):
    kernel = _generate(CHAIN, backend, arch,
                       enable_wrap_loads=True, wrap_distance=1)
    peeled = re.findall(r'peel_glb_\w+ = &\w+\[([^\]]+)\]', kernel)
    wrapped = re.findall(r'wrap_glb_\w+ = &\w+\[([^\]]+)\]', kernel)
    assert peeled, 'no peeled pointer emitted'
    assert wrapped, 'no lookahead pointer emitted'
    # `batchId1` before the loop is `batchId_start` clamped into range, and it
    # is the clamped one the peel has to name: the peel runs ahead of the size
    # guard, and `batchId_start` is bounded by the launch geometry rather than
    # by the element count.  This asserted `batchId_start` when it was written,
    # which is the defect test_peel_bounds.py now pins.  Inside the loop the
    # same name means `clamp(batchId0 + stride)`, which is what the wrapped
    # transfer wants -- hence the same token on both lines, meaning two
    # different things either side of the loop header.
    assert all('batchId1' in a for a in peeled), peeled
    assert all('batchId1' in a for a in wrapped), wrapped


@pytest.mark.parametrize("d", [1, 2, 4, 8])
@pytest.mark.parametrize("backend,arch", [("hip", "gfx90a"), ("cuda", "sm_86")])
def test_wrapped_write_lands_after_the_last_read(d, backend, arch):
    """The write-after-read the span clamp exists to prevent.

    Without clamping ``d`` to ``n - 1 - span``, a buffer read in two slots gets
    its wrapped write placed between them: the second read then sees the *next*
    element's data.  Nothing raises, the kernel just computes the wrong answer,
    which is why this is asserted on every distance rather than the default.
    """
    kernel = _generate(CHAIN, backend, arch,
                       enable_wrap_loads=True, wrap_distance=d)
    lines = kernel.splitlines()
    loop = _loop_start(lines)

    for name in {m.group(1) for m in re.finditer(r'\b(r\d+)\[', kernel)}:
        writes = [i for i, l in enumerate(lines)
                  if i > loop and re.search(rf'\b{name}\[[^\]]*\]\s*=', l)]
        before = [i for i, l in enumerate(lines)
                  if i < loop and re.search(rf'\b{name}\[[^\]]*\]\s*=', l)]
        if not before or not writes:
            continue                      # not wrapped
        reads = [i for i, l in enumerate(lines)
                 if i > loop and re.search(rf'=\s*[^=]*\b{name}\[', l)]
        if not reads:
            continue
        assert max(reads) < min(writes), (
            f'{name} is read at line {max(reads)} after its wrapped write at '
            f'{min(writes)}: the read sees the next element')


def test_single_iteration_loop_is_left_alone():
    """`SINGLE` has no next element and no back edge to wrap across."""
    kernel = _generate("square_notrans.py", "hip", "gfx90a",
                       enable_wrap_loads=True, wrap_distance=1)
    assert 'wrap_glb_' not in kernel
    assert 'peel_glb_' not in kernel
