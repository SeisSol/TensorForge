# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""The section analysis, asked directly.

`SectionPlan` answers three questions about a descriptor list, and until it
became an object of its own the only way to ask any of them was to generate a
kernel and read the answer back out of the emitted source.  That made every
statement about it a statement about a whole pipeline, which is why the corpus
grew a case per geometry.

These are the geometries themselves.  They matter most for the step that comes
next: teaching the analysis to see elementwise and reduction descriptors
changes which boxes it collects, and a snapshot corpus containing no mixed
kernel that generates cannot notice a shift in `written_in_slices`.  Here it
would be one failing assertion.
"""

from __future__ import annotations

import pytest

from tensorforge.backend.scopes import Scopes
from tensorforge.backend.section_plan import SectionPlan
from tensorforge.backend.symbol import Symbol, SymbolType
from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.exceptions import GenerationError
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.common.operation import AddOperator
from tensorforge.generators import elementwise as ew
from tensorforge.generators.descriptions import (GemmDescr,
                                                 MultilinearDescr,
                                                 ReductionDescr)

DTYPE = Datatype.F32
N = 8


def _tensor(alias, shape=(N, N), tmp=False):
    return Tensor(list(shape),
                  Addressing.PTR_BASED if tmp else Addressing.STRIDED,
                  BoundingBox([0] * len(shape), list(shape)),
                  alias=alias, is_tmp=tmp, datatype=DTYPE)


def _slice(tensor, lower, upper, offset=None, sliced=False):
    return SubTensor(tensor, BoundingBox(list(lower), list(upper)),
                     offset, sliced=sliced)


def _scopes(*tensors):
    """A scope in which each tensor already has a symbol.

    Which tensors have one is not incidental: the operand union is keyed by
    symbol name, so a tensor without a symbol contributes to the read union and
    to coverage but not to the staging size.  Passing them explicitly makes the
    tests say which case they are in.
    """
    scopes = Scopes()
    for i, tensor in enumerate(tensors):
        tensor.name = f"m{i}"
        scopes.add_to_global(Symbol(obj=tensor, name=tensor.name,
                                    stype=SymbolType.Batch))
    return scopes


def _box(plan_box):
    return (list(plan_box.lower()), list(plan_box.upper()))


# ----------------------------------------------------------------------
# operand_union: how wide a staging has to be
# ----------------------------------------------------------------------

def test_operand_union_is_the_hull_of_every_read():
    """Two operations reading different columns of one tensor.

    The first to touch it creates the staging and every later one inherits it,
    so it has to be sized for all of them at once.
    """
    a = _tensor("A")
    b, c, d = _tensor("B"), _tensor("C"), _tensor("D")
    scopes = _scopes(a, b, c, d)
    plan = SectionPlan([
        GemmDescr(False, False, a=_slice(a, [0, 0], [N, 4]),
                  b=SubTensor(b, BoundingBox([0, 0], [4, N])),
                  c=SubTensor(c)),
        GemmDescr(False, False, a=_slice(a, [0, 0], [N, 4], [0, 4]),
                  b=SubTensor(b, BoundingBox([0, 0], [4, N])),
                  c=SubTensor(d)),
    ], scopes)

    assert _box(plan.operand_union("m0")) == ([0, 0], [N, N])


def test_operand_union_is_absent_for_a_tensor_with_no_symbol():
    """A temporary has no symbol when the plan runs, so it has no union.

    Consumers fall back to their own box for those, which is what keeps a
    temporary's staging sized by whoever materialises it rather than by an
    entry that could not exist yet.
    """
    a, b = _tensor("A"), _tensor("B")
    tmp = _tensor("TMP", tmp=True)
    out = _tensor("OUT")
    scopes = _scopes(a, b, out)
    plan = SectionPlan([
        GemmDescr(False, False, a=SubTensor(a), b=SubTensor(b),
                  c=SubTensor(tmp)),
        GemmDescr(False, False, a=SubTensor(tmp), b=SubTensor(b),
                  c=SubTensor(out)),
    ], scopes)

    assert plan.operand_union("m0") is not None
    assert all(plan.operand_union(n) is None for n in ("TMP", "t0", "m3"))


# ----------------------------------------------------------------------
# written_in_slices: may the value stay in registers between operations
# ----------------------------------------------------------------------

def test_a_single_covering_writer_is_not_written_in_slices():
    a, b = _tensor("A"), _tensor("B")
    tmp = _tensor("TMP", tmp=True)
    out = _tensor("OUT")
    scopes = _scopes(a, b, out)
    plan = SectionPlan([
        GemmDescr(False, False, a=SubTensor(a), b=SubTensor(b),
                  c=SubTensor(tmp)),
        GemmDescr(False, False, a=SubTensor(tmp), b=SubTensor(b),
                  c=SubTensor(out)),
    ], scopes)

    assert plan.written_in_slices(tmp) is False


def test_an_accumulation_chain_is_not_written_in_slices():
    """Every term covers the whole box, so the last accumulator holds it all.

    This is the case the register residency exists for, and forcing a store per
    term would cost a round trip on each of them.  Several writers are not by
    themselves a reason to materialise.
    """
    a1, b1, a2, b2 = (_tensor(n) for n in ("A1", "B1", "A2", "B2"))
    tmp = _tensor("TMP", tmp=True)
    out = _tensor("OUT")
    scopes = _scopes(a1, b1, a2, b2, out)
    plan = SectionPlan([
        GemmDescr(False, False, a=SubTensor(a1), b=SubTensor(b1),
                  c=SubTensor(tmp)),
        GemmDescr(False, False, a=SubTensor(a2), b=SubTensor(b2),
                  c=SubTensor(tmp), alpha=1.0, beta=1.0),
        GemmDescr(False, False, a=SubTensor(tmp), b=SubTensor(b1),
                  c=SubTensor(out)),
    ], scopes)

    assert plan.written_in_slices(tmp) is False


def test_two_half_writes_are_written_in_slices():
    """Each operation holds only its own half, so both have to reach memory."""
    a = _tensor("A")
    b1 = _tensor("B1", shape=(N, 4))
    b2 = _tensor("B2", shape=(N, 4))
    tmp = _tensor("TMP", tmp=True)
    out = _tensor("OUT")
    scopes = _scopes(a, b1, b2, out)
    plan = SectionPlan([
        GemmDescr(False, False, a=SubTensor(a), b=SubTensor(b1),
                  c=_slice(tmp, [0, 0], [N, 4], [0, 0], sliced=True)),
        GemmDescr(False, False, a=SubTensor(a), b=SubTensor(b2),
                  c=_slice(tmp, [0, 0], [N, 4], [0, 4])),
        GemmDescr(False, False, a=SubTensor(tmp), b=SubTensor(a),
                  c=SubTensor(out)),
    ], scopes)

    assert plan.written_in_slices(tmp) is True


def test_a_writer_narrower_than_the_read_is_written_in_slices():
    """One writer is not enough if it covers less than what is read back.

    The destination here is global, and it has to be.  For a *temporary* this
    geometry never reaches the question: reading `tmp[:, 4:8]` where only
    `tmp[:, 0:4]` was written is an uncovered gap, and the initialisation check
    refuses the section before anything asks `written_in_slices`.  So the
    second half of that predicate -- the declared write union against the
    declared read union -- is reachable only for a tensor the check exempts,
    which is a global destination the caller may have filled.
    """
    a = _tensor("A")
    b = _tensor("B", shape=(N, 4))
    m = _tensor("M")
    out = _tensor("OUT")
    scopes = _scopes(a, b, m, out)
    plan = SectionPlan([
        GemmDescr(False, False, a=SubTensor(a), b=SubTensor(b),
                  c=_slice(m, [0, 0], [N, 4], [0, 0], sliced=True)),
        GemmDescr(False, False, a=SubTensor(m), b=SubTensor(a),
                  c=SubTensor(out)),
    ], scopes)

    assert plan.written_in_slices(m) is True


# ----------------------------------------------------------------------
# the initialisation check
# ----------------------------------------------------------------------

def test_a_temporary_read_where_nothing_writes_is_refused():
    """`tmp[:, 4:8]` is read and only `tmp[:, 0:4]` is written."""
    a = _tensor("A")
    b = _tensor("B", shape=(N, 4))
    tmp = _tensor("TMP", tmp=True)
    out = _tensor("OUT")
    scopes = _scopes(a, b, out)
    with pytest.raises(GenerationError, match="never written by any operation"):
        SectionPlan([
            GemmDescr(False, False, a=SubTensor(a), b=SubTensor(b),
                      c=_slice(tmp, [0, 0], [N, 4], [0, 0], sliced=True)),
            GemmDescr(False, False, a=_slice(tmp, [0, 0], [N, 4], [0, 4]),
                      b=SubTensor(a, BoundingBox([0, 0], [4, N])),
                      c=SubTensor(out)),
        ], scopes)


def test_a_temporary_that_is_never_written_at_all_is_refused():
    a, b = _tensor("A"), _tensor("B")
    tmp = _tensor("TMP", tmp=True)
    out = _tensor("OUT")
    scopes = _scopes(a, b, out)
    with pytest.raises(GenerationError, match="never written"):
        SectionPlan([
            GemmDescr(False, False, a=SubTensor(tmp), b=SubTensor(b),
                      c=SubTensor(out)),
        ], scopes)


def test_a_global_output_may_be_read_without_being_written():
    """Only temporaries are checked: a global may hold what the caller put there."""
    a, b = _tensor("A"), _tensor("B")
    out = _tensor("OUT")
    scopes = _scopes(a, b, out)
    SectionPlan([
        GemmDescr(False, False, a=SubTensor(out), b=SubTensor(b),
                  c=SubTensor(a)),
    ], scopes)


# ----------------------------------------------------------------------
# every descriptor kind states its own geometry
# ----------------------------------------------------------------------

def test_an_elementwise_write_initialises_a_temporary():
    """`tmp = abs(A)` counts as writing `tmp`.

    While the analysis only looked at contractions, a temporary produced
    pointwise looked to it like one nothing ever wrote, and the section was
    refused on a premise that was false.
    """
    a, b = _tensor("A"), _tensor("B")
    tmp = _tensor("TMP", tmp=True)
    out = _tensor("OUT")
    scopes = _scopes(a, b, out)
    plan = SectionPlan([
        ew.abs(SubTensor(tmp), SubTensor(a)),
        GemmDescr(False, False, a=SubTensor(tmp), b=SubTensor(b),
                  c=SubTensor(out)),
    ], scopes)

    assert plan.written_in_slices(tmp) is False


def test_an_elementwise_read_counts_toward_coverage():
    """`C = abs(tmp)` where only half of `tmp` was ever written.

    This is the direction that used to pass silently: the read was invisible,
    so nothing compared it against the writes and the kernel went on to read
    whatever the buffer held.
    """
    a = _tensor("A")
    b = _tensor("B", shape=(N, 4))
    tmp = _tensor("TMP", tmp=True)
    c = _tensor("C")
    scopes = _scopes(a, b, c)
    with pytest.raises(GenerationError, match="never written by any operation"):
        SectionPlan([
            GemmDescr(False, False, a=SubTensor(a), b=SubTensor(b),
                      c=_slice(tmp, [0, 0], [N, 4], [0, 0], sliced=True)),
            ew.abs(SubTensor(c), SubTensor(tmp)),
        ], scopes)


def test_a_reduction_states_its_own_shape():
    """The destination of a reduction has the source's shape minus `dims`."""
    a = _tensor("A")
    tmp = _tensor("TMP", shape=(N,), tmp=True)
    b = _tensor("B")
    out = _tensor("OUT")
    scopes = _scopes(a, b, out)
    plan = SectionPlan([
        ReductionDescr(SubTensor(tmp), SubTensor(a), [1], AddOperator()),
        MultilinearDescr(SubTensor(out), [SubTensor(tmp), SubTensor(b)],
                         [[0], [0, 1]], [[0], [0, 1]]),
    ], scopes)

    assert _box(plan.dest_union(tmp)) == ([0], [N])
    assert plan.written_in_slices(tmp) is False


def test_an_elementwise_scalar_operand_contributes_no_read():
    """A bare number in `srcs` is a value, not a read of anything."""
    a, c = _tensor("A"), _tensor("C")
    descr = ew.mul(SubTensor(c), SubTensor(a), 3.0)

    reads, _ = descr.effective_boxes()
    assert set(reads) == {a}


def test_a_contraction_scalar_is_a_rank_zero_read():
    """`alpha != 1` is modelled as a rank-0 tensor operand, not as a value.

    Worth stating rather than assuming, because the two descriptor kinds do
    genuinely differ here and a reader may expect them not to.  It is harmless:
    a rank-0 box has nothing to cover, so the initialisation check leaves it
    alone, and a scalar is never a temporary.
    """
    a, b, c = _tensor("A"), _tensor("B"), _tensor("C")
    scopes = _scopes(a, b, c)
    descr = GemmDescr(False, False, a=SubTensor(a), b=SubTensor(b),
                      c=SubTensor(c), alpha=2.0)
    SectionPlan([descr], scopes)

    reads, _ = descr.effective_boxes()
    assert {a, b} <= set(reads)
    scalars = [t for t in reads if t not in (a, b)]
    assert len(scalars) == 1 and reads[scalars[0]].rank() == 0


def test_declared_and_effective_coincide_for_a_pointwise_operation():
    """A pointwise operation iterates what it declares, so nothing narrows.

    The two part company only where the range comes from the operands, which
    is the contraction's case; stating that here is what makes the default
    implementation on `OperationDescription` a claim rather than a shortcut.
    """
    a = _tensor("A")
    c = _tensor("C")
    descr = ew.abs(_slice(c, [0, 0], [N, 4], [0, 4]),
                   _slice(a, [0, 0], [N, 4], [0, 4]))

    reads, write = descr.effective_boxes()
    assert _box(reads[a]) == ([0, 4], [N, 8])
    assert _box(write) == ([0, 4], [N, 8])


# ----------------------------------------------------------------------
# effective boxes: what the intersection leaves
# ----------------------------------------------------------------------

def test_coverage_is_judged_after_the_range_intersection():
    """A declared full-box write from a half-box operand writes half.

    `t += Q_face * c` in the elastic ADER kernels has this shape: every term
    declares the whole tensor and each covers only the rows its own face
    touches.  Judged on declared boxes this is one writer covering everything;
    judged on effective boxes it is a gap, and the gap is real.
    """
    a = _tensor("A")
    b = _tensor("B")
    tmp = _tensor("TMP", tmp=True)
    out = _tensor("OUT")
    scopes = _scopes(a, b, out)

    # The contraction index runs over [0, 4) because A supports only that, so
    # the output index that B carries is narrowed to [0, 4) as well.
    narrow = SubTensor(b, BoundingBox([0, 0], [N, 4]))
    with pytest.raises(GenerationError, match="never written by any operation"):
        SectionPlan([
            GemmDescr(False, False, a=SubTensor(a), b=narrow,
                      c=SubTensor(tmp)),
            GemmDescr(False, False, a=SubTensor(tmp), b=SubTensor(a),
                      c=SubTensor(out)),
        ], scopes)
