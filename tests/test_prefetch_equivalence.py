# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Prefetch must not change the numbers.

`enable_wrap_loads` moves a register transfer ahead of its consumer, wrapping
to the previous iteration when that runs off the front of the body. It is a
scheduling change; the results are supposed to be identical, bit for bit,
because the same values are read in the same order.

Nothing checked that. The snapshot tests record one configuration, the syntax
tests only ask whether it compiles, and the host oracle could not read a
prefetched kernel at all: `wrap.py` emits `uint32_t pipeStage0` as its
bookkeeping and `_DECL` had no `uint32_t`, so the very first line of the loop
aborted the run. The one configuration that most needed an oracle was the one
it could not evaluate.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import pathlib

import pytest

import kernel_eval
from tensorforge.common.context import Context, Options
from tensorforge.generators.generator import Generator

TIDS = (0, 1, 7, 15, 16, 31)


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
    return gen.get_kernel()


def _run(src):
    out = {}
    for tid in TIDS:
        out.update(kernel_eval.evaluate(src, tid=tid, seed=17,
                                        globals_only=True))
    return out


def _compare(name):
    """`(plain, prefetched)` destinations, or None when not evaluable."""
    path = next(p for p in
                (pathlib.Path(__file__).parent / 'cases').rglob('*.py')
                if p.stem == name)
    mod = _load(path)
    plain, wrapped = _generate(mod, False), _generate(mod, True)
    if plain == wrapped:
        return None
    return _run(plain), _run(wrapped)


def test_the_oracle_can_read_a_prefetched_kernel():
    """The gap that hid everything else.

    Without `uint32_t` in the declaration pattern a rotated kernel aborts on
    the loop's first statement -- `wrap.py` declares `uint32_t pipeStage0`
    there -- and every case below silently becomes unevaluable.  Stated
    against the declaration itself rather than against a generated kernel,
    because whether any case still rotates is a separate question with its own
    answer below.
    """
    assert kernel_eval._DECL.match('uint32_t pipeStage0 = 0;')
    src = _generate(_load(next(
        p for p in (pathlib.Path(__file__).parent / 'cases').rglob('*.py')
        if p.stem == 'square_notrans')), True)
    assert kernel_eval.evaluate(src, tid=0, seed=17, globals_only=True)


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
    it *would* wrap -- and that question used to be put to an unrotated body,
    where the windows are static and declared ahead of the loop.  Granting the
    rotation then declares the write window inside the loop, because its
    offset moves with the stage counter, and that is one of the pass's own
    refusal conditions.  So a transfer could be accepted while unrotated,
    rotated on the strength of that, and declined for a reason the rotation
    created.

    What came out was not a missed optimisation: the compute reads stage
    `pipeStage % 2` while the transfer fills the other one, so no iteration
    ever fills the stage it reads and the first element computes from whatever
    the arena held.  `trans_a` did exactly that -- 192 of 432 destination
    entries wrong, stable across seeds.
    """
    plain, wrapped = _compare('trans_a')
    for key in sorted(set(plain) | set(wrapped)):
        assert plain.get(key, 0.0) == pytest.approx(wrapped.get(key, 0.0),
                                                    abs=1e-4), key


def test_no_case_in_the_corpus_currently_earns_a_rotation():
    """Recorded, because the fix above has a cost and it should be visible.

    Every shared transfer that qualified before now fails the confirming
    probe, for one reason: a rotating write window is declared inside the
    loop and the peeled transfer would name it before it exists.  The pass
    says so itself.  Declaring that window ahead of the loop is the fix --
    the same move that took the address bindings and the static windows out
    of the guard, one scope further -- and until it is made, rotation is off
    rather than wrong.

    This asserts the *current* state.  When the window is hoisted it should
    fail, and that failure is the signal to delete it.
    """
    for name in ['trans_a', 'square_notrans', 'rectangular']:
        path = next(p for p in
                    (pathlib.Path(__file__).parent / 'cases').rglob('*.py')
                    if p.stem == name)
        assert 'pipeStage' not in _generate(_load(path), True)
