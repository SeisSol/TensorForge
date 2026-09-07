# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""What a repeated run is allowed to become.

Three shapes with the same skeleton and nearly the same hole count, which have
to come out in three different classes: a recurrence that must stay in order,
an accumulation that may be split at a price, and contributions that share
nothing.  A test that put them in one class would be a test that let the loop
binder reorder a recursion.
"""

import pytest

from tensorforge.analysis.dependence import (Carried, binding_period, carried,
                                             escapes)
from tensorforge.analysis.families import find_repeats
from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators.descriptions import GemmDescr

DTYPE = Datatype.F32


def make(alias, shape):
    """One tensor per alias, so that two chunks naming it name one object."""
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


def chunks_of(descrs, period, count, start=0):
    return [list(descrs[start + period * i:start + period * (i + 1)])
            for i in range(count)]


# --- the three shapes -------------------------------------------------------


def recursion(steps=6):
    """``deriv[k+1] = kDivM @ deriv[k]``."""
    return [gemm(make('kDivM', [56, 56]), make(f'deriv{k}', [56, 9]),
                 make(f'deriv{k + 1}', [56, 9])) for k in range(steps)]


def accumulation(faces=4):
    """``Q += fPrT{i} @ I``."""
    return [gemm(make(f'fPrT{i}', [56, 56]), make('I', [56, 9]),
                 make('Q', [56, 9]), add=True) for i in range(faces)]


def independent(faces=4):
    """``face[i] = fPrT{i} @ I`` -- one destination each."""
    return [gemm(make(f'fPrT{i}', [56, 56]), make('I', [56, 9]),
                 make(f'face{i}', [56, 9])) for i in range(faces)]


def test_a_recursion_is_ordered():
    d = carried(chunks_of(recursion(), 1, 6))
    assert d.ordered
    assert Carried.FLOW in d.kinds
    assert d.distance == 1
    assert d.accumulators == ()


def test_an_accumulation_is_joined_only_at_the_sum():
    d = carried(chunks_of(accumulation(), 1, 4))
    assert d.only_accumulation
    assert not d.ordered
    assert d.accumulators == ('Q',)


def test_independent_contributions_share_nothing():
    d = carried(chunks_of(independent(), 1, 4))
    assert d.independent
    assert d.kinds == frozenset()
    assert d.accumulators == ()


def test_the_three_shapes_land_in_three_classes():
    a = carried(chunks_of(recursion(), 1, 6))
    b = carried(chunks_of(accumulation(), 1, 4))
    c = carried(chunks_of(independent(), 1, 4))
    assert (a.ordered, b.only_accumulation, c.independent) == (True, True, True)
    assert not (a.only_accumulation or a.independent)
    assert not (b.ordered or b.independent)
    assert not (c.ordered or c.only_accumulation)


# --- the accumulator has to be taken out ------------------------------------


def test_an_overwritten_destination_is_not_an_accumulator():
    """Writing the same tensor without reading it is an output dependence."""
    descrs = [gemm(make(f'A{i}', [56, 56]), make('I', [56, 9]),
                   make('Q', [56, 9])) for i in range(3)]
    d = carried(chunks_of(descrs, 1, 3))
    assert Carried.OUTPUT in d.kinds
    assert d.accumulators == ()
    assert d.ordered


def test_an_accumulator_only_some_chunks_read_is_not_one():
    descrs = [gemm(make('A0', [56, 56]), make('I', [56, 9]),
                   make('Q', [56, 9]), add=False),
              gemm(make('A1', [56, 56]), make('I', [56, 9]),
                   make('Q', [56, 9]), add=True)]
    d = carried(chunks_of(descrs, 1, 2))
    assert d.accumulators == ()
    assert d.ordered


def test_a_shared_input_is_not_a_dependence():
    """Every chunk reading ``I`` constrains nothing."""
    d = carried(chunks_of(independent(), 1, 4))
    assert 'I' not in d.tensors


# --- what the run wants for buffers -----------------------------------------


def test_a_chain_of_distinct_buffers_has_no_period():
    run = find_repeats(recursion())[0]
    holes = list(zip(*run.general.bindings))
    assert all(binding_period(h) is None for h in holes)


def test_a_ping_pong_has_period_two():
    """Two buffers alternating: each hole names one again every other chunk."""
    descrs = [gemm(make('A', [56, 56]), make(f'tmp{k % 2}', [56, 9]),
                   make(f'tmp{(k + 1) % 2}', [56, 9])) for k in range(6)]
    run = find_repeats(descrs, max_period=1)[0]
    holes = list(zip(*run.general.bindings))
    assert [binding_period(h) for h in holes] == [2, 2]


def test_a_ping_pong_is_a_period_two_chunk_with_nothing_varying():
    """Left to itself the search states the same fact as a longer chunk.

    Two steps that alternate between two buffers are one two-step chunk that
    names both, so there is no hole at all -- which is the same answer as a
    one-step chunk with two holes of period two, reached without a table.
    """
    descrs = [gemm(make('A', [56, 56]), make(f'tmp{k % 2}', [56, 9]),
                   make(f'tmp{(k + 1) % 2}', [56, 9])) for k in range(6)]
    run = find_repeats(descrs)[0]
    assert (run.period, run.count, run.arity) == (2, 3, 0)


def test_a_single_binding_has_no_period():
    assert binding_period([make('A', [4, 4])]) is None


# --- what the run leaves behind ---------------------------------------------


def test_intermediates_read_afterwards_escape():
    """The reason a recursion is an indexed array and not a rotation."""
    steps = recursion()
    total = make('total', [56, 9])
    tail = [gemm(make('w', [56, 56]), make(f'deriv{k}', [56, 9]), total,
                 add=True) for k in range(1, 7)]
    descrs = steps + tail
    assert escapes(descrs, 0, len(steps)) == tuple(
        sorted(f'deriv{k}' for k in range(1, 7)))


def test_nothing_escapes_a_run_whose_output_is_final():
    descrs = independent()
    assert escapes(descrs, 0, len(descrs)) == ()


def test_only_what_is_read_again_escapes():
    steps = recursion(3)
    tail = [gemm(make('w', [56, 56]), make('deriv3', [56, 9]),
                 make('total', [56, 9]))]
    assert escapes(steps + tail, 0, len(steps)) == ('deriv3',)


# --- edges ------------------------------------------------------------------


def test_one_chunk_carries_nothing():
    assert carried(chunks_of(recursion(1), 1, 1)).independent


def test_the_run_and_its_dependence_agree_on_the_slices():
    descrs = accumulation()
    run = find_repeats(descrs)[0]
    d = carried(chunks_of(descrs, run.period, run.count, run.start))
    assert d.only_accumulation
    assert run.arity == 1
