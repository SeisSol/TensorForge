# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Stating a run once and putting it back.

The load-bearing test is the round trip: whatever `roll` does, `unroll` has to
undo exactly.  Everything else here is about what `roll` declines to do, which
matters more than what it does -- a loop that hides a barrier or that tidies
away a repetition nobody asked about is worse than no loop.
"""

import pytest

from tensorforge.analysis.antiunify import skeleton
from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators.descriptions import (ForDescr, GemmDescr,
                                                 GridBarrierDescr,
                                                 GridFenceDescr)
from tensorforge.generators.rolling import roll, unroll

DTYPE = Datatype.F32


def make(alias, shape):
    key = (alias, tuple(shape))
    if key not in make.pool:
        make.pool[key] = Tensor(list(shape), Addressing.STRIDED,
                                BoundingBox([0] * len(shape), list(shape)),
                                alias=alias, datatype=DTYPE)
    return SubTensor(make.pool[key])


make.pool = {}


@pytest.fixture(autouse=True)
def fresh_pool():
    make.pool = {}
    yield
    make.pool = {}


def gemm(a, b, c, add=False):
    return GemmDescr(trans_a=False, trans_b=False, a=a, b=b, c=c,
                     alpha=1.0, beta=1.0 if add else 0.0)


def recursion(steps=6):
    return [gemm(make('kDivM', [56, 56]), make(f'deriv{k}', [56, 9]),
                 make(f'deriv{k + 1}', [56, 9])) for k in range(steps)]


def accumulation(faces=4):
    return [gemm(make(f'fPrT{i}', [56, 56]), make('I', [56, 9]),
                 make('Q', [56, 9]), add=True) for i in range(faces)]


def separate(faces=4):
    return [gemm(make(f'fPrT{i}', [56, 56]), make('I', [56, 9]),
                 make(f'face{i}', [56, 9])) for i in range(faces)]


def same_shape(descrs):
    return skeleton(descrs)[0]


# --- the round trip ---------------------------------------------------------


@pytest.mark.parametrize('build', [recursion, accumulation, separate])
def test_unroll_undoes_roll(build):
    original = build()
    assert same_shape(unroll(roll(original))) == same_shape(original)


def test_unroll_undoes_roll_with_a_prologue_and_an_epilogue():
    head = [gemm(make('W', [56, 56]), make('U', [56, 9]), make('I', [56, 9]))]
    tail = [gemm(make('M', [56, 56]), make('Q', [56, 9]), make('R', [56, 9]))]
    original = head + accumulation() + tail
    rolled = roll(original, max_arity=1)
    assert any(isinstance(d, ForDescr) for d in rolled)
    assert same_shape(unroll(rolled)) == same_shape(original)


def test_unroll_leaves_a_list_without_loops_alone():
    original = separate(2)
    assert unroll(original) is not original
    assert same_shape(unroll(original)) == same_shape(original)


def test_rolling_twice_changes_nothing_more():
    once = roll(recursion())
    twice = roll(once)
    assert same_shape(unroll(twice)) == same_shape(unroll(once))


# --- what ends up in the loop -----------------------------------------------


def test_a_recursion_rolls_into_one_sequential_loop():
    out = roll(recursion())
    assert len(out) == 1
    loop = out[0]
    assert isinstance(loop, ForDescr)
    assert (loop.iterations, loop.arity) == (6, 2)
    assert loop.sequential


def test_an_accumulation_rolls_and_may_be_reordered():
    out = roll(accumulation())
    loop = out[0]
    assert (loop.iterations, loop.arity) == (4, 1)
    assert not loop.sequential
    assert loop.dependence.accumulators == ('Q',)
    assert loop.writes().tensor.alias == 'Q'


def test_separate_destinations_leave_the_loop_free():
    loop = roll(separate())[0]
    assert not loop.sequential
    assert loop.dependence.independent
    assert loop.writes() is None
    assert len(loop.destinations()) == 4


def test_the_loop_carries_what_decides_the_buffers():
    """A recursion whose intermediates are summed afterwards is an array."""
    steps = recursion()
    tail = [gemm(make('w', [56, 56]), make(f'deriv{k}', [56, 9]),
                 make('total', [56, 9]), add=True) for k in range(1, 7)]
    loop = roll(steps + tail, max_arity=2)[0]
    assert loop.periods == (None, None)
    assert loop.escaping == tuple(sorted(f'deriv{k}' for k in range(1, 7)))


def test_a_rotation_shows_a_period_and_nothing_escaping():
    descrs = [gemm(make('A', [56, 56]), make(f'tmp{k % 2}', [56, 9]),
                   make(f'tmp{(k + 1) % 2}', [56, 9])) for k in range(6)]
    out = roll(descrs, max_period=1)
    loop = out[0]
    assert loop.periods == (2, 2)
    assert loop.escaping == ()


def test_a_loop_reports_the_reads_of_every_iteration():
    loop = roll(accumulation())[0]
    aliases = sorted(v.tensor.alias for v in loop.reads())
    assert aliases == ['I'] + sorted(f'fPrT{i}' for i in range(4))


def test_an_accumulated_destination_is_a_write_here_and_a_read_next_door():
    """`reads()` follows the descriptor contract; the dependence does not.

    A descriptor states the operands it was given, so an accumulated
    destination appears under `writes()` and not under `reads()`.  Asking
    whether one chunk depends on another is a different question and has to
    count that same tensor as read, or every accumulation would look like a
    set of chunks that share nothing.  Both readings are used, so the two
    disagreeing is worth pinning rather than reconciling.
    """
    loop = roll(accumulation())[0]
    assert 'Q' not in [v.tensor.alias for v in loop.reads()]
    assert loop.writes().tensor.alias == 'Q'
    assert loop.dependence.accumulators == ('Q',)


# --- what roll declines to do -----------------------------------------------


def test_a_barrier_in_the_body_stops_the_roll():
    """The section split happens on this list; a loop would hide the barrier."""
    descrs = [d for i in range(4)
              for d in accumulation(4)[i:i + 1] + [GridBarrierDescr()]]
    out = roll(descrs)
    assert not any(isinstance(d, ForDescr) for d in out)
    assert same_shape(out) == same_shape(descrs)


def test_a_fence_in_the_body_stops_the_roll():
    descrs = [d for i in range(4)
              for d in accumulation(4)[i:i + 1] + [GridFenceDescr()]]
    assert not any(isinstance(d, ForDescr) for d in roll(descrs))


def test_a_barrier_may_be_rolled_when_asked_for():
    descrs = [d for i in range(4)
              for d in accumulation(4)[i:i + 1] + [GridFenceDescr()]]
    out = roll(descrs, allow_barriers=True)
    assert isinstance(out[0], ForDescr)
    assert out[0].barrier()
    assert same_shape(unroll(out)) == same_shape(descrs)


def test_a_run_that_varies_in_nothing_is_left_alone():
    """The same computation three times is a question, not an opportunity."""
    once = accumulation(1)
    descrs = once + once + once
    out = roll(descrs)
    assert not any(isinstance(d, ForDescr) for d in out)


def test_a_list_with_nothing_repeated_comes_back_unchanged():
    descrs = [gemm(make('A', [56, 56]), make('I', [56, 9]),
                   make('Q', [56, 9])),
              gemm(make('B', [9, 9]), make('J', [9, 4]), make('R', [9, 4]))]
    out = roll(descrs)
    assert len(out) == 2
    assert not any(isinstance(d, ForDescr) for d in out)


def test_an_empty_list_rolls_to_an_empty_list():
    assert roll([]) == []
    assert unroll([]) == []
