# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""The section prologue's copies of the operators into shared memory.

`preload_globals` copies every batch-constant operand into shared memory once
per block, before the batch loop.  On NVIDIA that did not build: each copy was
built into a body of its own, found no source value there, and fell back to
driving a `cuda::pipeline` object as text -- one no kernel declares, so nvcc
refused every kernel with the option.  And nothing waited for the copies: the
barrier after them orders the threads, not the transfers they issued.

Two things were wrong on every target once it built.  The copy counted the
operator's elements, not the scalars it is stored in, so a TF32-split operator
arrived half; and an operator offered its fragment order at emission outgrew
the image sized before it.  `local_flux` checks all of it through the source;
the numbers were checked on sm_120 and gfx1150 with the probe.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

from tensorforge.common.basic_types import Addressing
from tensorforge.common.context import Context, Options
from tensorforge.generators.generator import Generator

CASES = Path(__file__).resolve().parent / "cases"
CASE = "local_flux.py"
OPERATORS = ("glb_m0", "glb_m4", "glb_m6", "glb_m8")


def _descrs(parts: int = 1):
    path = CASES / CASE
    spec = importlib.util.spec_from_file_location("tf_preload__" + path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    descrs = mod.descr_list()
    for descr in descrs:
        for op in descr.ops:
            tensor = getattr(op, 'tensor', None)
            if tensor is not None and tensor.addressing == Addressing.NONE:
                tensor.storage_parts = parts
    return mod, descrs


def _generate(arch, backend="cuda", parts=1, **opts):
    mod, descrs = _descrs(parts)
    ctx = Context(arch=arch, backend=backend, fp_type=mod.DTYPE,
                  options=Options(preload_globals=True, **opts))
    gen = Generator(descrs, ctx)
    gen.generate()
    return gen, descrs


def _images(kernel):
    """Where each preloaded operator's image starts in the block's arena."""
    return {m.group(1): int(m.group(2)) for m in
            re.finditer(r'\b(glb_m\d+) = &totalShrMem\[(\d+)\]', kernel)}


def _prologue(kernel):
    return kernel[:re.search(r'for \(size_t \w*batchId', kernel).start()]


@pytest.mark.parametrize("arch", ["sm_86", "sm_120"])
def test_the_prologue_drives_no_pipeline_object(arch):
    kernel = _generate(arch)[0].get_kernel()
    assert set(_images(kernel)) == set(OPERATORS)
    assert 'producer_acquire' not in kernel
    assert 'cuda::memcpy_async' not in kernel
    assert '__pipeline_memcpy_async' in _prologue(kernel)


@pytest.mark.parametrize("arch", ["sm_86", "sm_120"])
def test_the_copies_are_waited_for_before_the_barrier(arch):
    prologue = _prologue(_generate(arch)[0].get_kernel())
    barrier = prologue.rindex('__syncthreads()')
    last_copy = prologue.rindex('__pipeline_memcpy_async')
    waits = [m.start() for m in re.finditer(r'__pipeline_wait_prior\(0\)',
                                            prologue)]
    assert any(last_copy < w < barrier for w in waits)


def test_a_split_operator_is_copied_whole():
    """Two parts of 56x56 scalars each: 6272 per image, not 3136."""
    gen, descrs = _generate("sm_100", parts=2)
    kernel = gen.get_kernel()
    images = _images(kernel)
    assert sorted(images.values()) == [0, 6272, 12544, 18816]
    assert '+ 6144]' in _prologue(kernel)   # the copy reaches the second part


def test_a_fragment_ordered_operator_is_preloaded_in_its_order(monkeypatch):
    """The order pads 56x56 to 3584 slots, so the images grow to 7168."""
    from tensorforge.backend.instructions.compute.primitives import nvidia
    monkeypatch.setattr(nvidia, "ENABLED", True)
    gen, descrs = _generate("sm_100", parts=2, prepare_operands=True)
    images = _images(gen.get_kernel())
    assert sorted(images.values()) == [0, 7168, 14336, 21504]
    operators = [op.tensor for descr in descrs for op in descr.ops
                 if op.tensor.addressing == Addressing.NONE]
    assert operators and all(t.storage_order is not None for t in operators)


def test_operators_that_leave_no_room_are_read_from_global():
    """Split, the four take 100 kB: under sm_120's 99 kB for a block they
    passed the prologue's check and left nothing for one multiplication."""
    gen, _ = _generate("sm_120", parts=2)
    kernel = gen.get_kernel()
    assert not _images(kernel)
    assert 'glb_m0 = &m0[0]' in kernel


def test_an_rdna_work_group_holds_64_kb():
    for arch in ("gfx1100", "gfx1150", "gfx1201"):
        hw = Context(arch=arch, backend="hip",
                     fp_type=_descrs()[0].DTYPE).get_vm().get_hw_descr()
        assert hw.max_local_mem_size_per_block == 64 * 1024, arch


@pytest.mark.parametrize("preload", [True, False])
def test_no_transfer_drives_a_pipeline_object_without_wide_bodies(preload):
    """One body per instruction: a loader's source is a value of another body,
    so the copy cannot be structured.  It used to drive a `cuda::pipeline` no
    kernel declares; it moves its bytes with plain loads now."""
    mod, descrs = _descrs()
    ctx = Context(arch="sm_120", backend="cuda", fp_type=mod.DTYPE,
                  options=Options(preload_globals=preload, wide_bodies=False))
    gen = Generator(descrs, ctx)
    gen.generate()
    kernel = gen.get_kernel()
    for text in ('producer_acquire', 'producer_commit', 'consumer_wait',
                 'cuda::memcpy_async'):
        assert text not in kernel
