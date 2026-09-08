# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Which slices of a descriptor list repeat.

The cases are shaped like the two lists this is aimed at: a recursion written
out once per step, and a face contribution written out once per face.  What a
run has to get right is where it stops -- a run reported one chunk too long
would roll a step into a loop that does not belong in it.
"""

import pytest

from tensorforge.analysis.families import Repeat, find_repeats
from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators.descriptions import (GemmDescr, GridBarrierDescr,
                                                 GridFenceDescr)

DTYPE = Datatype.F32


def tensor(alias, shape):
    return SubTensor(Tensor(list(shape), Addressing.STRIDED,
                            BoundingBox([0] * len(shape), list(shape)),
                            alias=alias, datatype=DTYPE))


def gemm(a, b, c):
    return GemmDescr(trans_a=False, trans_b=False, a=a, b=b, c=c)


def step(k):
    """``deriv[k+1] = kDivM @ deriv[k]`` -- one term of the recursion."""
    return [gemm(tensor('kDivM', [56, 56]),
                 tensor(f'deriv{k}', [56, 9]),
                 tensor(f'deriv{k + 1}', [56, 9]))]


def face(i):
    """``Q += fPrT{i} @ I`` -- one face contribution."""
    return [gemm(tensor(f'fPrT{i}', [56, 56]),
                 tensor('I', [56, 9]),
                 tensor('Q', [56, 9]))]


def flat(chunks):
    return [d for chunk in chunks for d in chunk]


# --- the shapes this is for -------------------------------------------------


def test_a_recursion_written_out_is_one_run():
    descrs = flat([step(k) for k in range(6)])
    runs = find_repeats(descrs)
    assert len(runs) == 1
    run = runs[0]
    assert (run.start, run.period, run.count) == (0, 1, 6)
    # both buffers vary, the operator does not
    assert run.arity == 2
    assert run.general.binding_names()[0] == ('deriv1', 'deriv0')


def test_four_faces_vary_in_one_operand():
    descrs = flat([face(i) for i in range(4)])
    runs = find_repeats(descrs)
    assert len(runs) == 1
    assert (runs[0].start, runs[0].period, runs[0].count) == (0, 1, 4)
    assert runs[0].arity == 1
    assert runs[0].general.binding_names() == tuple((f'fPrT{i}',)
                                                    for i in range(4))


def test_a_run_between_a_prologue_and_an_epilogue():
    head = [gemm(tensor('W', [56, 56]), tensor('U', [56, 9]),
                 tensor('I', [56, 9]))]
    tail = [gemm(tensor('M', [56, 56]), tensor('Q', [56, 9]),
                 tensor('R', [56, 9]))]
    descrs = head + flat([face(i) for i in range(4)]) + tail
    runs = find_repeats(descrs, max_arity=1)
    assert len(runs) == 1
    assert (runs[0].start, runs[0].count) == (1, 4)
    assert runs[0].stop == 5


def test_a_run_stops_where_the_body_changes():
    """A chunk that varies in one more operand ends the run at that bound."""
    odd = [gemm(tensor('other', [56, 56]), tensor('I', [56, 9]),
                tensor('R', [56, 9])),
           gemm(tensor('more', [56, 56]), tensor('I', [56, 9]),
                tensor('R', [56, 9]))]
    descrs = flat([face(i) for i in range(3)]) + odd
    runs = find_repeats(descrs, max_arity=1)
    assert [(r.start, r.count) for r in runs] == [(0, 3), (3, 2)]


def test_equal_shapes_over_unrelated_tensors_are_one_run():
    """Two contractions of the same shapes differ only in what they name.

    Which is what a hole is, so structurally they *are* a repetition -- what
    tells a family from an accident of shape is the price, and the price is
    the arity.  Stated as a test because the alternative reading, that this
    should not be a run at all, would have to be enforced somewhere and there
    is no structural ground to enforce it on.
    """
    head = [gemm(tensor('W', [56, 56]), tensor('U', [56, 9]),
                 tensor('V', [56, 9]))]
    runs = find_repeats(head + flat([face(i) for i in range(3)]))
    assert len(runs) == 1
    assert (runs[0].start, runs[0].count) == (0, 4)
    assert runs[0].arity == 3

    bounded = find_repeats(head + flat([face(i) for i in range(3)]),
                           max_arity=1)
    assert [(r.start, r.count) for r in bounded] == [(1, 3)]


def test_nothing_repeats():
    descrs = [gemm(tensor('A', [56, 56]), tensor('I', [56, 9]),
                   tensor('Q', [56, 9])),
              gemm(tensor('B', [9, 9]), tensor('J', [9, 4]),
                   tensor('R', [9, 4]))]
    assert find_repeats(descrs) == []


def test_a_single_chunk_is_not_a_run():
    assert find_repeats(face(0)) == []


# --- fences take part -------------------------------------------------------


def test_a_fence_inside_the_chunk_is_part_of_the_repetition():
    descrs = flat([face(i) + [GridFenceDescr()] for i in range(4)])
    runs = find_repeats(descrs)
    assert len(runs) == 1
    assert (runs[0].period, runs[0].count) == (2, 4)
    assert runs[0].arity == 1


def test_a_fence_in_a_different_place_breaks_the_run():
    descrs = (face(0) + [GridFenceDescr()] + face(1) + [GridFenceDescr()] +
              face(2) + face(3) + [GridFenceDescr()])
    runs = find_repeats(descrs)
    assert all(r.count < 4 for r in runs)


def test_a_fence_and_a_barrier_are_not_the_same_descriptor():
    descrs = (face(0) + [GridFenceDescr()] + face(1) + [GridBarrierDescr()] +
              face(2) + [GridFenceDescr()] + face(3) + [GridFenceDescr()])
    runs = find_repeats(descrs)
    assert all(r.count < 4 for r in runs)


# --- the ranking ------------------------------------------------------------


def test_runs_do_not_overlap():
    descrs = flat([face(i) for i in range(4)] + [step(k) for k in range(4)])
    runs = find_repeats(descrs)
    spans = [(r.start, r.stop) for r in runs]
    for (a_start, a_stop), (b_start, b_stop) in zip(spans, spans[1:]):
        assert a_stop <= b_start


def test_a_run_with_no_holes_is_still_a_run():
    """The same computation twice repeats; whether to roll it is not asked here."""
    descrs = flat([face(0), face(0), face(0)])
    runs = find_repeats(descrs)
    assert len(runs) == 1
    assert runs[0].count == 3
    assert runs[0].arity == 0


def test_the_answer_does_not_depend_on_the_order_things_were_tried():
    descrs = flat([face(i) for i in range(6)])
    once = find_repeats(descrs)
    twice = find_repeats(descrs)
    assert [(r.start, r.period, r.count) for r in once] == \
           [(r.start, r.period, r.count) for r in twice]


def test_min_count_below_two_is_refused():
    with pytest.raises(ValueError):
        find_repeats(face(0) + face(1), min_count=1)


def test_max_arity_bounds_what_a_run_may_cost():
    descrs = flat([step(k) for k in range(6)])
    assert find_repeats(descrs, max_arity=2)[0].count == 6
    assert find_repeats(descrs, max_arity=1) == []


def test_max_period_bounds_the_search():
    descrs = flat([face(i) + face(i + 10) for i in range(3)])
    assert find_repeats(descrs, max_period=1) != []
    bounded = find_repeats(descrs, max_period=1)
    assert all(r.period == 1 for r in bounded)


def test_a_longer_run_wins_over_a_shorter_one():
    descrs = flat([face(i) for i in range(6)])
    runs = find_repeats(descrs)
    assert len(runs) == 1
    assert runs[0].count == 6
