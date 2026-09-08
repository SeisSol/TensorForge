# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""The host oracle for operands that arrive by `cp.async`.

Two defects held each other up here, and each one is why the other did not
show.

`kernel_eval` swallowed `__pipeline_memcpy_async` under a catch-all written
for `cuda::pipeline` objects, which genuinely move nothing.  The transfer
does.  What that cost was not a visible failure: `Slot` fills an unwritten
address from its seed, so a kernel whose operand arrives this way still
produced a full destination of plausible numbers -- 128 of 128 entries
non-zero for `aligned_operands`, and every one of them wrong.  Against a
reference the error was 5e+02.

And because no lane read anything through the staging window, it did not
matter how many lanes ran.  The oracles named a round number -- 32 lanes, 64
tids -- where the kernel's own width is 16, or 4 once the lead is widened.
Surplus lanes are not a wider wave; they are a second block's threads
addressing one block's memory, and a hop loop is only guarded where the
extent fails to divide, so they copy from past the end of the operand.  That
reads exactly like a generator overrun and is not one.

Hence both halves are pinned here: that the copy moves data, and that the
lanes come from the launcher.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import pathlib
import re

import numpy as np
import pytest

import kernel_eval
from tensorforge.common.context import Context
from tensorforge.generators.generator import Generator

CASES = pathlib.Path(__file__).parent / 'cases'

#: The cases whose operands promise an aligned stride, which is what
#: `widths_for` needs before it will pick anything but scalar.  Every other
#: case in the corpus leaves `Tensor.alignment` at 0 -- unknown -- so the
#: staged path they exercise is the narrow one.
STAGED = ['aligned_operands', 'wide_cascade', 'wide_cascade_tail']


def _build(name):
    """The kernel, and the geometry its launcher starts it with."""
    path = next(p for p in CASES.rglob('*.py') if p.stem == name)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(mod)
    gen = Generator(mod.descr_list(),
                    Context(arch='sm_86', backend='cuda', fp_type=mod.DTYPE))
    with contextlib.redirect_stdout(io.StringIO()):
        gen.generate()
    return (mod, gen.get_kernel(),
            kernel_eval.launch_geometry(gen.get_launcher()))


# --------------------------------------------------------------------------- #
# The copy moves data
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('name', STAGED)
def test_a_staged_kernel_computes_the_product(name):
    """Against numpy, over the same seeded inputs.

    Both operands reach the multiply through the staging window; if the
    transfer is a no-op the window holds seed fill and this is off by two
    orders of magnitude, which is what it was.
    """
    mod, src, (lanes, mults) = _build(name)
    assert '__pipeline_memcpy_async' in src, 'this case no longer stages'
    mem = kernel_eval.evaluate_wave(src, lanes, seed=11, globals_only=True,
                                    mults=mults)

    seed = kernel_eval.Slot(11)
    m, n, k = mod.M, mod.N, mod.K
    a = np.array([[seed.read('m1', i + j * m) for j in range(k)]
                  for i in range(m)])
    b = np.array([[seed.read('m2', i + j * k) for j in range(n)]
                  for i in range(k)])
    got = np.array([[mem.get(('m0', i + j * m), 0.0) or 0.0
                     for j in range(n)] for i in range(m)])
    assert np.max(np.abs(a @ b - got)) < 1e-6


def test_the_copy_writes_the_slots_it_names():
    """The transfer on its own, without a kernel around it."""
    mem = kernel_eval.Slot(3)
    env = {'dst': kernel_eval.Ptr(mem, 'dst'),
           'src': kernel_eval.Ptr(mem, 'src')}
    interp = kernel_eval.Interp(mem, env)
    interp.widths['dst'] = 4
    for i in range(4):
        mem.write('src', 10 + i, float(i) + 0.5)
    interp.assign('__pipeline_memcpy_async(&dst[2], &src[10], 16)')
    assert [mem.read('dst', 2 + i) for i in range(4)] == [0.5, 1.5, 2.5, 3.5]


def test_an_unmodelled_copy_refuses_instead_of_vanishing():
    """The four-argument zero-fill form, or anything else new.

    A transfer that quietly does nothing is the defect this branch exists to
    keep from recurring, so an unrecognised spelling is an abort rather than a
    return.
    """
    interp = kernel_eval.Interp(kernel_eval.Slot(0), {})
    with pytest.raises(kernel_eval.Abort):
        interp.assign('__pipeline_memcpy_async(&dst[0], &src[0], 16, 4)')


def test_the_commit_and_the_wait_still_move_nothing():
    """They are what the catch-all was written for, and they stay in it."""
    interp = kernel_eval.Interp(kernel_eval.Slot(0), {})
    interp.assign('__pipeline_commit()')
    interp.assign('__pipeline_wait_prior(0)')


# --------------------------------------------------------------------------- #
# The lanes come from the launcher
# --------------------------------------------------------------------------- #

def test_the_geometry_is_the_launcher_s():
    lanes, mults = kernel_eval.launch_geometry(
        'void launcher() { dim3 block(4, 16, 1); }')
    assert (lanes, mults) == (4, 16)


def test_a_launcher_without_block_dimensions_refuses():
    """Rather than falling back on a round number, which is the bug."""
    with pytest.raises(kernel_eval.Abort):
        kernel_eval.launch_geometry('void launcher() { }')


def test_the_widened_build_is_narrower_than_the_plain_one():
    """Four lanes against sixteen, which is why a fixed count cannot serve.

    Widening the lead does not make the block wider; it makes it narrower,
    because each lane carries four elements instead of one.  A harness that
    names one number for both configurations is over by a factor of four in
    exactly the configuration it was added to test.
    """
    from tensorforge.backend.instructions.memory import vectorize
    old = (vectorize.LEAD_VECTORIZE, vectorize.LEAD_BLOCKING)
    try:
        vectorize.LEAD_VECTORIZE, vectorize.LEAD_BLOCKING = False, 1
        _, _, plain = _build('aligned_operands')
        vectorize.LEAD_VECTORIZE, vectorize.LEAD_BLOCKING = True, 1
        _, _, wide = _build('aligned_operands')
    finally:
        vectorize.LEAD_VECTORIZE, vectorize.LEAD_BLOCKING = old
    assert wide[0] < plain[0]


@pytest.mark.parametrize('name', STAGED)
def test_no_lane_reads_past_its_operand(name):
    """At the launcher's width, the staged reads land inside the operand.

    This is the check that turns an overrun into a measurement.  At 32 lanes
    `aligned_operands` reads `m2` to 191 and at 64 to 239, against an operand
    of 128 -- neither of which is the generator's doing.

    It is also what the cascade cases are for.  A hop plan that overshoots at
    one width and a plan that double-covers at another both produce a kernel
    that runs; the first shows up here and the second in the product above.
    """
    mod, src, (lanes, mults) = _build(name)
    mem = kernel_eval.evaluate_wave(src, lanes, seed=11, globals_only=True,
                                    mults=mults)
    extent = {'m0': mod.M * mod.N, 'm1': mod.M * mod.K, 'm2': mod.K * mod.N}
    for base, size in extent.items():
        touched = [i for b, i in mem if b == base]
        assert touched, f'{base} was never touched'
        assert max(touched) < size, (
            f'{base}: read index {max(touched)} of an operand holding {size}')

# --------------------------------------------------------------------------- #
# The corpus reaches the widths at all
# --------------------------------------------------------------------------- #

def test_the_corpus_exercises_every_copy_width():
    """Stated as a property, because it silently stopped being true once.

    `alignment` defaults to 0 and 0 is *unknown*, which `widths_for` turns
    into scalar.  With `aligned_operands` alone the corpus reached one width
    and one hop count: 128 elements over 16 lanes is two hops of four and
    nothing else, so the cascade below four was unreachable and the code for
    it was as good as absent from every snapshot diff.
    """
    seen = set()
    for name in STAGED:
        _, src, _ = _build(name)
        seen |= {int(w) for w in
                 re.findall(r'__pipeline_memcpy_async\([^;]*?,\s*(\d+)\)', src)}
    assert {4, 8, 16} <= seen, f'widths reached: {sorted(seen)}'


def test_a_cascade_case_drops_lanes_in_its_tail():
    """The width decision and the lane predicate, on one transfer.

    Separately each is simple.  Together they are where an offset counted in
    elements per lane meets a bound counted in elements, and no kernel in the
    corpus had both happen to the same transfer.
    """
    _, src, _ = _build('wide_cascade_tail')
    assert re.search(r'if \(threadIdx\.x < \d+\)', src)
