# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Distribution is total, and it is the only statement about lane-varying.

Three properties, each of which was false before and each of which an
explicitly vectorised emitter depends on:

1. ``uniformity`` and ``layout`` cannot disagree.  They did, on 71443 of the
   93837 values in the corpus that carried a layout, and always in the unsafe
   direction --- ``GRID`` on a value spread across the lanes.

2. *Replicated* and *unknown* are different answers.  Both spell ``float x``
   in SPMD, so nothing noticed; in ESIMD one is ``T`` and the other cannot be
   given a type at all.

3. A replicated operand does not destroy the layout of the one it is scaled
   into.  ``alpha * A`` is spread exactly like ``A``, and SeisSol scales
   almost every operator it generates.
"""

from __future__ import annotations

import pytest

from tensorforge.backend.pir.core import (SCALAR_LAYOUT, IRError, LaneAxis,
                                          RegisterLayout, ScalarType,
                                          Uniformity, Value, join_layout)
from tensorforge.backend.symbol import LeadIndex, VarOffset, layout_of
from tensorforge.common.basic_types import Datatype

F32 = ScalarType(Datatype.F32)
SPREAD = RegisterLayout((LaneAxis(16),))
SPREAD8 = RegisterLayout((LaneAxis(8),))


def val(vid, layout=None, uniformity=Uniformity.GRID):
    return Value(id=vid, type=F32, uniformity=uniformity, layout=layout)


# --------------------------------------------------------------------------
# 1. one fact, one place
# --------------------------------------------------------------------------

def test_a_distributed_value_is_lane_varying_whatever_the_caller_said():
    assert val(1, SPREAD).uniformity is Uniformity.LANE


def test_derivation_only_tightens():
    """An explicit LANE is not an error to be corrected, and MULT is not raised.

    The rule narrows a claim that is too broad; it does not widen one that is
    already narrow.  Otherwise a caller who knows more than the layout does --
    the batch id is ``MULT`` and carries no layout at all -- would be overruled
    by a default.
    """
    assert val(2, SPREAD, Uniformity.LANE).uniformity is Uniformity.LANE
    assert val(3, None, Uniformity.MULT).uniformity is Uniformity.MULT
    assert val(4, SCALAR_LAYOUT, Uniformity.MULT).uniformity is Uniformity.MULT


def test_replication_does_not_make_a_value_lane_varying():
    assert val(5, SCALAR_LAYOUT).uniformity is Uniformity.GRID


# --------------------------------------------------------------------------
# 2. replicated is not unknown
# --------------------------------------------------------------------------

def test_lane_span_distinguishes_replicated_from_untracked():
    assert val(6, SCALAR_LAYOUT).lane_span() == 1
    assert val(7, SPREAD).lane_span() == 16
    # Not 1.  Those are the same number and opposite facts, and an emitter
    # that cannot tell them apart writes a scalar where a vector belongs.
    with pytest.raises(IRError):
        val(8).lane_span()


def test_distributed_predicate():
    assert val(9, SPREAD).distributed
    assert not val(10, SCALAR_LAYOUT).distributed
    assert not val(11).distributed


def test_multi_axis_span_is_the_product():
    lay = RegisterLayout((LaneAxis(4, 1), LaneAxis(4, 4)))
    assert val(12, lay).lane_span() == 16


# --------------------------------------------------------------------------
# 3. a scalar does not veto
# --------------------------------------------------------------------------

def test_scaling_keeps_the_layout_it_is_scaled_into():
    assert join_layout([val(13, SCALAR_LAYOUT), val(14, SPREAD)]) == SPREAD


def test_two_different_distributions_still_give_unknown():
    """Not an error -- a vendor intrinsic may consume two -- but nothing may
    be concluded from it either."""
    assert join_layout([val(15, SPREAD), val(16, SPREAD8)]) is None


def test_untracked_operands_are_ignored_not_fatal():
    assert join_layout([val(17), val(18, SPREAD)]) == SPREAD


def test_no_layout_anywhere_stays_unknown():
    assert join_layout([val(19), val(20)]) is None


def test_all_replicated_agrees():
    assert join_layout([val(21, SCALAR_LAYOUT), val(22, SCALAR_LAYOUT)]) == SCALAR_LAYOUT


# --------------------------------------------------------------------------
# layout_of: the entry point where a distribution enters the IR
# --------------------------------------------------------------------------

def test_thread_independent_index_is_replicated_not_unknown():
    assert layout_of([0, 3]) == SCALAR_LAYOUT
    assert layout_of([]) == SCALAR_LAYOUT


def test_a_lead_index_gives_its_axis():
    assert layout_of([LeadIndex(0, 16, 1)]) == RegisterLayout((LaneAxis(16, 1),))


def test_an_offset_lead_index_is_still_that_axis():
    """A slicing shift does not change *which lane holds what*.

    The shift used to be a `VarOffset` wrapped around the index and is now a
    field on it -- see `LeadIndex._offset`.  Either way the layout is the same
    axis, which is the property this test is about.
    """
    assert (layout_of([LeadIndex(0, 16, 1, offset=32)])
            == RegisterLayout((LaneAxis(16, 1),)))


def test_two_lead_axes_need_the_wave_size_to_be_checkable():
    idx = [LeadIndex(0, 4, 1), LeadIndex(0, 4, 4)]
    assert layout_of(idx, num_threads=None) is None
    assert layout_of(idx, num_threads=16) == RegisterLayout(
        (LaneAxis(4, 1), LaneAxis(4, 4)))


def test_axes_that_do_not_tile_the_wave_stay_unknown():
    """Unknown, not a guess: the axes describe partial replication this
    function has not established, and a wrong layout lets a pass merge two
    values that differ."""
    idx = [LeadIndex(0, 4, 1), LeadIndex(0, 4, 1)]
    assert layout_of(idx, num_threads=16) is None


# --------------------------------------------------------------------------
# a staged image's distribution is recorded by whoever fills it
# --------------------------------------------------------------------------

def test_a_shared_image_can_carry_a_layout():
    """It could not, and the omission was silent.

    `_record_linear_layout` answered only for registers, because a register
    image knows its own lane count and a shared one does not -- `num_threads`
    is None there, a shared buffer not being owned by one multiplication.  The
    loader that writes the run does know, and is the only party that does.

    A later read is `load_linear`, whose address has no lane term at all: it
    reports what the fill recorded and can derive nothing.  So an unrecorded
    claim left every consumer of a staged image failing closed -- invisible
    under SPMD, where unknown costs only precision, and fatal under an
    explicit vector, where a declaration needs a distribution.
    """
    from tensorforge.backend.symbol import Symbol, SymbolType
    sym = Symbol.__new__(Symbol)
    sym.stype = SymbolType.SharedMem
    sym.num_threads = None
    sym.layout = None
    sym._record_linear_layout(0, 1, threads=16)
    assert sym.layout == RegisterLayout((LaneAxis(16, 1),))


def test_without_a_lane_count_it_stays_unknown():
    """`None` means unknown, and a fill that cannot say how wide the run is
    must leave it that way rather than pick a default."""
    from tensorforge.backend.symbol import Symbol, SymbolType
    sym = Symbol.__new__(Symbol)
    sym.stype = SymbolType.SharedMem
    sym.num_threads = None
    sym.layout = None
    sym._record_linear_layout(0, 1)
    assert sym.layout is None


def test_two_fills_disagreeing_leave_it_unknown():
    """A silent overwrite would hand the second fill's claim to consumers of
    the first."""
    from tensorforge.backend.symbol import Symbol, SymbolType
    sym = Symbol.__new__(Symbol)
    sym.stype = SymbolType.SharedMem
    sym.num_threads = None
    sym.layout = None
    sym._record_linear_layout(0, 1, threads=16)
    sym._record_linear_layout(0, 1, threads=32)
    assert sym.layout is None


# --------------------------------------------------------------------------
# a discarded speculation leaves nothing behind
# --------------------------------------------------------------------------

def test_a_discarded_attempt_takes_its_layout_claim_back():
    """The claim belongs to the fill that makes it true.

    A speculative attempt that gets discarded emitted no fill, so the image is
    not distributed that way and the claim is not true.  Left behind, the
    second attempt sees a symbol the first did not -- and the same case
    generates two different kernels.

    `_rollback` restores the body, the value counter, the name counter and the
    scope stack, all of which the builder owns.  This is state it does not,
    which is why the mechanism is a registered undo rather than a snapshot:
    the list of everything mutable would otherwise live in the one place that
    can see none of it.
    """
    from tensorforge.backend.symbol import Symbol, SymbolType
    from tensorforge.backend.pir.build import IRBuilder
    from tensorforge.common.context import Context as _Ctx

    b = IRBuilder(fptype=Datatype.F32,
                  context=_Ctx(arch='pvc', backend='esimd',
                               fp_type=Datatype.F32))
    sym = Symbol.__new__(Symbol)
    sym.stype = SymbolType.SharedMem
    sym.num_threads = None
    sym.layout = None

    with b.speculative() as spec:
        sym._record_linear_layout(0, 1, threads=16, writer=b)
        assert sym.layout is not None, 'the attempt did record it'
        spec.discard()
    assert sym.layout is None, 'and the discard took it back'


def test_a_kept_attempt_keeps_it():
    from tensorforge.backend.symbol import Symbol, SymbolType
    from tensorforge.backend.pir.build import IRBuilder
    from tensorforge.common.context import Context as _Ctx

    b = IRBuilder(fptype=Datatype.F32,
                  context=_Ctx(arch='pvc', backend='esimd',
                               fp_type=Datatype.F32))
    sym = Symbol.__new__(Symbol)
    sym.stype = SymbolType.SharedMem
    sym.num_threads = None
    sym.layout = None

    with b.speculative():
        sym._record_linear_layout(0, 1, threads=16, writer=b)
    assert sym.layout == RegisterLayout((LaneAxis(16, 1),))


def test_a_structured_store_records_what_it_distributes():
    """The third fill path, and the third place the statement was missing.

    A staged image is filled linearly by the loader, in bulk by the transfer,
    or one element at a time by a compute instruction writing out of its
    registers.  The first two recorded how the image ends up distributed; the
    third did not, so an image written that way read back as unknown.

    Derivable here, unlike the linear paths: the index carries a `LeadIndex`,
    which *is* the distribution, so this reports what `layout_of` already
    computes rather than restating it.
    """
    from tensorforge.backend.symbol import Symbol, SymbolType
    sym = Symbol.__new__(Symbol)
    sym.stype = SymbolType.SharedMem
    sym.layout = None
    sym._note_layout(RegisterLayout((LaneAxis(16, 1),)))
    assert sym.layout == RegisterLayout((LaneAxis(16, 1),))


def test_an_underivable_store_leaves_it_alone():
    """`layout_of` answers `None` when it cannot establish the distribution,
    and `None` is not a claim -- recording it would erase a claim an earlier
    fill did establish."""
    from tensorforge.backend.symbol import Symbol, SymbolType
    sym = Symbol.__new__(Symbol)
    sym.stype = SymbolType.SharedMem
    sym.layout = RegisterLayout((LaneAxis(16, 1),))
    sym._note_layout(None)
    assert sym.layout == RegisterLayout((LaneAxis(16, 1),))


def test_two_fill_paths_disagreeing_leave_it_unknown():
    from tensorforge.backend.symbol import Symbol, SymbolType
    sym = Symbol.__new__(Symbol)
    sym.stype = SymbolType.SharedMem
    sym.layout = None
    sym._note_layout(RegisterLayout((LaneAxis(16, 1),)))
    sym._note_layout(RegisterLayout((LaneAxis(8, 1),)))
    assert sym.layout is None


# -- what a register image says about itself ------------------------------- #

def _register(threads, width=1, dims=(0,), axes=None):
    from tensorforge.backend.data_types import RegMemObject
    from tensorforge.backend.symbol import Symbol, SymbolType
    sym = Symbol(name='r', stype=SymbolType.Register, obj=RegMemObject('r', 64))
    sym.num_threads, sym.lead_width = threads, width
    sym.lead_dims, sym.lead_axes = list(dims), axes
    return sym


@pytest.mark.parametrize('threads', [1, 4, 8, 16, 32, 64])
@pytest.mark.parametrize('width', [1, 2, 4])
def test_the_owner_is_the_lane_the_arithmetic_named(threads, width):
    """The answer is derived from the layout now instead of computed here,
    and at rank one it has to be the same answer bit for bit -- every image
    in the tree is rank one, so anything else would be an unreviewed change
    to all of them.

    The width is divided out before the layout is asked, because a packing is
    a property of the register and not of the distribution: a lane holding
    `width` neighbours holds them in the lane the axis already named.
    """
    sym = _register(threads, width)
    for element in range(200):
        assert sym.owning_lane([element]) == (element // width) % threads


def test_a_rank_two_image_has_an_owner_once_its_producer_says_so():
    """What `lead_dims` could not say.  Two positions do not distinguish one
    pair of axes from another, and the pair is the whole content: `LaneAxis(8,
    4)` beside `LaneAxis(4, 1)` puts the row in the high lane bits and the
    column pair in the low ones, which is where a chained matrix product
    leaves its accumulator."""
    from tensorforge.backend.pir.core import LaneAxis
    sym = _register(32, dims=(0, 1), axes=(LaneAxis(8, 4), LaneAxis(4, 1)))
    assert sym.register_layout() is not None
    for row in range(8):
        for col in range(4):
            assert sym.owning_lane([row, col]) == row * 4 + col


def test_axes_that_replicate_have_no_owner():
    """An element held by four lanes has no one owner, and an address naming
    a single lane would name one of them arbitrarily.  Refused as a layout
    rather than at each reader, because whether the same axis replicates is
    settled by the rest of the layout and not by the axis."""
    from tensorforge.backend.pir.core import LaneAxis
    sym = _register(32, dims=(0, 1), axes=(LaneAxis(16, 4), LaneAxis(16, 4)))
    assert sym.register_layout() is None
    assert sym.owning_lane([1, 1]) is None


def test_a_rank_two_image_without_stated_axes_is_still_unknown():
    """Positions alone are not a distribution, and reading them as one is
    what would hand a consumer the wrong element with nothing to notice."""
    sym = _register(32, dims=(0, 1))
    assert sym.register_layout() is None
    assert sym.owning_lane([1, 1]) is None


def test_the_declaration_is_not_the_recorded_layout():
    """Two facts, and the difference matters.  `Symbol.layout` is what a
    filler wrote down afterwards, for a reader that cannot see the write;
    this is what the image is, stated before anyone reads it.  A consumer
    deciding whether it can take an operand needs the second, and at the
    moment it asks the first does not exist yet."""
    sym = _register(32)
    assert sym.layout is None
    assert sym.register_layout() is not None


# -- the number a coordinate is divided by --------------------------------- #

def test_the_block_is_the_wave_for_every_image_in_the_tree():
    """The neutrality claim, stated so it is checked rather than argued.
    Every register image is one axis over the whole wave, so every address
    divides by the number it always did."""
    for threads in (1, 4, 16, 32, 64):
        sym = _register(threads)
        assert sym.lead_block(0) == threads
        # And for a dimension this symbol did not declare, the wave as well:
        # a lead index arriving there has always meant the wave, and reading
        # it as unspread would change an address that has been right since
        # before there were axes to state.
        assert sym.lead_block(1) == threads


def test_the_block_is_the_axis_where_a_producer_stated_one():
    """And this is the whole of the generalisation: eight rows and four
    columns in one round of a 32-lane wave, where dividing either coordinate
    by 32 has no reading at all."""
    from tensorforge.backend.pir.core import LaneAxis
    sym = _register(32, dims=(0, 1), axes=(LaneAxis(8, 4), LaneAxis(4, 1)))
    assert (sym.lead_block(0), sym.lead_block(1)) == (8, 4)


def test_slots_are_counted_in_blocks_and_not_in_waves():
    """What the rename is about.  A sixteen-element dimension is one slot on
    a wave that holds sixteen and two on an axis whose block is eight, and
    the formula never knew which number it was given."""
    from tensorforge.backend.symbol import slots_for
    assert slots_for(0, 16, 16) == 1
    assert slots_for(0, 16, 8) == 2
    assert slots_for(0, 16, 4) == 4
    assert slots_for(0, 16, 8, 2) == 2 * 1


def test_an_unstated_rank_two_image_still_divides_by_the_wave():
    """Because that is what it did.  Positions without axes are not a
    distribution, and changing what they address is not this patch's to
    make -- `register_layout` refuses them and this falls back to what the
    addressing has always done."""
    sym = _register(32, dims=(0, 1))
    assert sym.register_layout() is None
    assert (sym.lead_block(0), sym.lead_block(1)) == (32, 32)


# -- allocation and addressing, sized in the same units -------------------- #

def _temporaries(threads=32):
    from tensorforge.backend.scopes import Scopes
    from tensorforge.backend.temporaries import Temporaries
    from tensorforge.common.basic_types import Datatype
    from tensorforge.common.context import Context
    ctx = Context(arch='sm_80', backend='cuda', fp_type=Datatype.F32)
    return Temporaries(ctx, Scopes(), threads)


def _bbox(*sizes):
    from tensorforge.common.matrix.boundingbox import BoundingBox
    return BoundingBox([0] * len(sizes), list(sizes))


def test_a_single_axis_image_is_sized_as_it_always_was():
    """The neutrality claim.  Every caller passes a bare dimension index, and
    that has to mean the wave, cyclic, with the same register count."""
    temps = _temporaries(32)
    sym, alloc = temps.register_array(_bbox(64, 3), 0)
    assert sym.lead_dims == [0] and sym.lead_axes is None
    assert sym.obj.size == (64 // 32) * 3
    assert sym.lead_block(0) == 32


@pytest.mark.parametrize('blocks', [((0, 8), (1, 4)), ((1, 4), (0, 8))])
def test_the_blocks_it_sized_in_are_the_blocks_the_addressing_divides_by(blocks):
    """The property the whole parameter exists for.  An allocation in units
    the addressing does not divide by aliases the next dimension onto this
    one, and until the blocks could be passed the two agreed only because
    both defaulted to the wave."""
    from tensorforge.backend.symbol import slots_for
    temps = _temporaries(32)
    sym, _ = temps.register_array(_bbox(16, 8), list(blocks))
    for dim, block in blocks:
        assert sym.lead_block(dim) == block
    assert sym.register_layout() is not None
    assert sym.obj.size == (slots_for(0, 16, dict(blocks)[0])
                            * slots_for(0, 8, dict(blocks)[1]))


def test_the_lane_nesting_is_the_order_the_pairs_came_in():
    """Innermost first: the first pair gets stride 1.  A producer that wants
    a row in the high lane bits and a column pair in the low ones lists the
    column first, which is why this is the pair order and not the dimension
    order."""
    from tensorforge.backend.pir.core import LaneAxis
    temps = _temporaries(32)
    sym, _ = temps.register_array(_bbox(16, 8), [(1, 4), (0, 8)])
    assert sym.lead_dims == [1, 0]
    assert sym.lead_axes == (LaneAxis(4, 1), LaneAxis(8, 4))
    # lane t holds column t % 4 and row (t // 4) % 8
    for row in range(8):
        for col in range(4):
            assert sym.owning_lane([row, col]) == row * 4 + col


def test_blocks_that_do_not_tile_the_wave_are_refused_where_they_are_chosen():
    """Rather than at the reader.  They would leave lanes holding copies,
    `register_layout` would refuse the layout, `lead_block` would fall back to
    the wave -- and the addressing would then divide by a number this
    allocation did not size in."""
    from tensorforge.common.exceptions import InternalError
    temps = _temporaries(32)
    with pytest.raises(InternalError, match='do not tile'):
        temps.register_array(_bbox(16, 8), [(0, 8), (1, 8)])
    with pytest.raises(InternalError, match='two lane axes'):
        temps.register_array(_bbox(16, 8), [(0, 8), (0, 4)])


def test_a_packed_image_on_two_axes_has_no_owner():
    """One width and no statement of which axis carries it.  Dividing an
    arbitrary coordinate by it would name a lane confidently and wrongly."""
    from tensorforge.backend.pir.core import LaneAxis
    sym = _register(32, width=2, dims=(0, 1),
                    axes=(LaneAxis(4, 1), LaneAxis(8, 4)))
    assert sym.register_layout() is not None
    assert sym.owning_lane([1, 1]) is None


def test_the_slot_run_is_counted_once_however_many_axes_share_the_wave():
    """How many register entries one slot occupies is a question about the
    lowering, not about the distribution.  Under ESIMD the work-item holds
    the whole wave, so a slot is a run of that many entries -- once, however
    many axes divide the wave between them."""
    from tensorforge.backend.scopes import Scopes
    from tensorforge.backend.symbol import slots_for
    from tensorforge.backend.temporaries import Temporaries
    from tensorforge.common.basic_types import Datatype
    from tensorforge.common.context import Context
    ctx = Context(arch='pvc', backend='esimd', fp_type=Datatype.F32)
    temps = Temporaries(ctx, Scopes(), 16)
    sym, _ = temps.register_array(_bbox(8, 4), [(0, 4), (1, 4)])
    assert sym.obj.size == slots_for(0, 8, 4) * slots_for(0, 4, 4) * 16
