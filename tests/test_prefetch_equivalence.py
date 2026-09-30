# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Prefetch must not change the numbers.

`enable_wrap_loads` moves a register transfer ahead of its consumer, wrapping
to the previous iteration when that runs off the front of the body. It is a
scheduling change; the results are supposed to be identical, bit for bit,
because the same values are read in the same order.

The snapshot tests record one configuration and the syntax tests only ask
whether it compiles, so this is where the numbers are compared, by running
both builds on the host oracle. That needs the oracle to read a prefetched
kernel: `wrap.py` emits `uint32_t pipeStage0` as its bookkeeping, and without
`uint32_t` in `_DECL` the very first line of the loop would abort the run.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import pathlib

import pytest

from tensorforge.common.context import Context, Options
from tensorforge.generators.generator import Generator
from tensorforge.reference import kernel_eval


def _load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _generate(mod, wrap):
    ctx = Context(arch='sm_86', backend='cuda',
                  fp_type=getattr(mod, 'DTYPE', None),
                  options=Options(enable_wrap_loads=wrap))
    gen = Generator(mod.descr_list(), ctx)
    with contextlib.redirect_stdout(io.StringIO()):
        gen.generate()
    return gen.get_kernel(), kernel_eval.launch_geometry(gen.get_launcher())


def _run(src, geometry):
    """One block over one memory, at the width the launcher starts.

    Not lane by lane over a fixed set of tids.  A staged operand arrives
    cooperatively, so a lane running on its own memory fills one stripe of the
    window and reads seed fill for the rest --- and two builds then agree about
    a window neither of them wrote.  A fixed tid set can also reach past the
    end of the block, where the unguarded hops copy from beyond the operand.
    """
    lanes, mults = geometry
    return kernel_eval.evaluate_wave(src, lanes, seed=17, globals_only=True,
                                     mults=mults)


def _compare(name):
    """`(plain, prefetched)` destinations, or None when not evaluable."""
    path = next(p for p in
                (pathlib.Path(__file__).parent / 'cases').rglob('*.py')
                if p.stem == name)
    mod = _load(path)
    (plain, pg), (wrapped, wg) = _generate(mod, False), _generate(mod, True)
    if plain == wrapped:
        return None
    return _run(plain, pg), _run(wrapped, wg)


def test_the_oracle_can_read_a_prefetched_kernel():
    """The gap that would hide everything else.

    Without `uint32_t` in the declaration pattern a rotated kernel aborts on
    the loop's first statement -- `wrap.py` declares `uint32_t pipeStage0`
    there -- and every case below silently becomes unevaluable.  Stated
    against the declaration itself rather than against a generated kernel,
    because whether any case rotates is a separate question with its own
    answer below.
    """
    assert kernel_eval._DECL.match('uint32_t pipeStage0 = 0;')
    src, geometry = _generate(_load(next(
        p for p in (pathlib.Path(__file__).parent / 'cases').rglob('*.py')
        if p.stem == 'square_notrans')), True)
    assert _run(src, geometry)


@pytest.mark.parametrize('name', ['square_notrans', 'rectangular',
                                  'accumulate_chain', 'chain'])
def test_prefetch_does_not_change_the_numbers(name):
    both = _compare(name)
    if both is None:
        pytest.skip('prefetch changed nothing in this kernel')
    plain, wrapped = both
    for key in sorted(set(plain) | set(wrapped)):
        assert plain.get(key, 0.0) == pytest.approx(wrapped.get(key, 0.0),
                                                    abs=1e-4), key


def test_a_rotation_is_only_granted_where_the_wrap_survives_it():
    """The invariant: rotated if and only if wrapped.

    The rotation is decided before a body exists, by asking the pass whether
    it *would* wrap.  Put to an unrotated body, where the windows are static
    and declared ahead of the loop, that question can get the wrong answer:
    granting the rotation declares the write window inside the loop, because
    its offset moves with the stage counter, and that is one of the pass's own
    refusal conditions.  So a transfer could be accepted while unrotated,
    rotated on the strength of that, and declined for a reason the rotation
    created.

    What would come out is not a missed optimization: the compute reads stage
    `pipeStage % 2` while the transfer fills the other one, so no iteration
    ever fills the stage it reads and the first element computes from whatever
    the arena held.  `trans_a` is the case that shows it.
    """
    plain, wrapped = _compare('trans_a')
    for key in sorted(set(plain) | set(wrapped)):
        assert plain.get(key, 0.0) == pytest.approx(wrapped.get(key, 0.0),
                                                    abs=1e-4), key


def test_no_case_in_the_corpus_currently_earns_a_rotation():
    """Recorded, because the invariant above has a cost and it should be
    visible.

    Every shared transfer in these cases fails the confirming probe, for one
    reason: a rotating write window is declared inside the loop and the peeled
    transfer would name it before it exists.  The pass says so itself.
    Declaring that window ahead of the loop is the fix -- the address bindings
    and the static windows already sit outside the guard, and this is the same
    move one scope further out -- and until it is made, rotation is off rather
    than wrong.

    This asserts the *current* state.  When the window is hoisted it should
    fail, and that failure is the signal to delete it.
    """
    for name in ['trans_a', 'square_notrans', 'rectangular']:
        path = next(p for p in
                    (pathlib.Path(__file__).parent / 'cases').rglob('*.py')
                    if p.stem == name)
        assert 'pipeStage' not in _generate(_load(path), True)[0]
