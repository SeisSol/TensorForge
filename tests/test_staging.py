# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""The fallback, and the two properties that make it one.

It has to close every gap -- including the ones the swap family and the
transpose decline, which is the only reason to have it -- and it has to be
worse than either, or preferring them would need an argument rather than a
count.
"""

from __future__ import annotations

import pytest

from tensorforge.backend.instructions.compute import bitlayout, staging
from tensorforge.backend.instructions.compute.bitlayout import (
    Bit, BitLayout, Place, Position)
from tensorforge.backend.instructions.compute.primitives.amd import (
    MATRIX_OPS, layouts, relayout)


def _indices(*extents):
    out = [()]
    for extent in extents:
        out = [tuple(prefix) + (value,)
               for prefix in out for value in range(extent)]
    return out


# -- it closes every gap --------------------------------------------------- #

def test_it_serves_the_gap_the_transpose_declines():
    """A packed operand holds its low bits inside a register, and no exchange
    of lane and slot bits reaches an element index.  That is the case the
    emitter declines today, and the one this exists for."""
    packed = BitLayout((
        (Bit(Place.VECTOR, 1), Bit(Place.VECTOR, 2)),
        tuple(Bit(Place.LANE, 1 << b) for b in range(6)),
    ))
    want = relayout.transposed(4, 64)
    assert relayout.transposes_between(packed, want, 4) is None
    plan = staging.staged(packed, want, _indices(4, 64))
    assert plan is not None
    assert staging.buffer_elements(plan) == 4 * 64


def test_it_serves_the_gap_the_swap_family_declines():
    """Two elements sharing a pair of slots that need different toggles are
    not one region, and no sequence of reads serves them."""
    have = BitLayout(((Bit(Place.LANE, 1), Bit(Place.LANE, 2)),))
    want = BitLayout(((Bit(Place.LANE, 2), Bit(Place.LANE, 1)),))
    indices = _indices(4)
    assert bitlayout.moves(have, want, indices) is None
    assert staging.staged(have, want, indices) is not None


def test_the_only_refusal_is_a_mismatched_index_space():
    """There is no shape of gap this declines; a caller reaching it has
    already been told `None` by something cheaper."""
    one = BitLayout(((Bit(Place.LANE, 1),),))
    two = BitLayout(((Bit(Place.LANE, 1),), (Bit(Place.LANE, 2),)))
    assert staging.staged(one, two, [(0,)]) is None


# -- every element goes through exactly once ------------------------------- #

@pytest.mark.parametrize('name', sorted(op.builtin for op in MATRIX_OPS
                                        if op.lane_batched()))
def test_each_element_is_written_once_and_read_once(name):
    """The property a buffer rests on.  Two elements at one address is the
    quieter failure of the two: the store still lands and only the value is
    wrong."""
    op = next(o for o in MATRIX_OPS if o.builtin == name)
    fragment = layouts.fragment_layout(op, 'A')
    nest = relayout.nest_shared(op.m, op.wave)
    if fragment is None:
        pytest.skip('no tabulated fragment')
    indices = _indices(op.m, op.wave)
    plan = staging.staged(nest, relayout.transposed(op.m, op.wave), indices)
    assert plan is not None
    addresses = [transfer.address for transfer in plan]
    assert len(addresses) == len(set(addresses)) == len(indices)


def test_the_buffer_is_read_from_the_plan():
    """A reservation smaller than the plan is an overrun and one larger is
    shared memory nobody writes, so the two are one number."""
    indices = _indices(4, 16)
    plan = staging.staged(relayout.nest_shared(4, 16),
                          relayout.transposed(4, 16), indices)
    assert staging.buffer_elements(plan) == len(indices)


# -- and it is the expensive one ------------------------------------------- #

def test_it_moves_one_element_per_access_where_a_register_path_moves_a_region():
    """Which is why it is the fallback and not a peer.  The count is what
    lets the cheaper one be preferred for a reason rather than by habit."""
    from tensorforge.backend.instructions.compute.primitives.amd import reorder
    op = next(o for o in MATRIX_OPS
              if o.builtin == 'mfma_f64_16x16x4f64')
    register = reorder.accumulator_cost(op)
    assert register is not None

    indices = _indices(op.m, op.wave)
    plan = staging.staged(relayout.nest_shared(op.m, op.wave),
                          relayout.transposed(op.m, op.wave), indices)
    stores, loads = staging.accesses(plan)
    assert stores == loads == len(indices)
    assert stores + loads > register


def test_a_trip_that_changes_nothing_still_costs_the_trip():
    """No shortcut for an operand that already arrives right -- which is why
    the cheaper questions are asked first and this one last."""
    layout = relayout.transposed(4, 64)
    indices = _indices(4, 64)
    plan = staging.staged(layout, layout, indices)
    assert all(t.source == t.target for t in plan)
    assert staging.accesses(plan) == (len(indices), len(indices))


def test_a_base_on_each_side():
    """The group offset, as everywhere else: which side carries it depends on
    which of the two the wave is divided over."""
    layout = BitLayout(((Bit(Place.LANE, 1),),))
    plan = staging.staged(layout, layout, _indices(2),
                          base_want=Position(lane=8))
    assert [t.target.lane - t.source.lane for t in plan] == [8, 8]


# -- the rung between registers and memory --------------------------------- #

def test_a_transpose_is_swaps_and_merges():
    """No `swap` moves anything between a register and a lane, so none
    transposes one register.  Several registers and a merge do: the elements
    going from register `r` to register `s` all move by the one XOR `r ^ s`,
    and masking the merge to the lanes they land on leaves the rest alone.

    Which makes the runtime's transposes a convenience rather than a
    capability -- the arrangement written once."""
    from tensorforge.backend.instructions.compute.primitives.amd import reorder
    for ext, wave in ((4, 64), (4, 32), (8, 64), (16, 64)):
        indices = _indices(ext, wave)
        plan = bitlayout.moves(relayout.nest_shared(ext, wave),
                               relayout.transposed(ext, wave), indices)
        assert plan is not None and len(plan) == ext * ext, (ext, wave)
        assert all(move.xor == move.source ^ move.target for move in plan)
        assembled = reorder.compose_exchange(
            relayout.nest_shared(ext, wave),
            relayout.transposed(ext, wave), indices, wave)
        assert len(assembled) == ext
        assert all(len(group) == ext for group in assembled)


def test_the_assembled_exchange_is_emitted_now(monkeypatch):
    """`laneMerge` is what unblocked it: the regions are one lane out of every
    `ext`, no `dppUpdate` mask expresses that, and a ternary on the lane id
    does -- one the runtime's own transpose already writes eight times.

    So a width the runtime has no `transpose*` for stays in registers instead
    of falling to the trip."""
    from tensorforge.backend.instructions.compute.primitives.amd import (
        exchange_codegen, reorder)
    from tensorforge.backend.pir.build import IRBuilder
    from tensorforge.backend.pir.core import ScalarType
    from tensorforge.common.basic_types import Datatype
    from tensorforge.common.context import Context

    indices = _indices(8, 64)
    assembled = reorder.compose_exchange(relayout.nest_shared(8, 64),
                                         relayout.transposed(8, 64),
                                         indices, 64)
    assert reorder.emittable(assembled)
    assert not relayout.has_transpose(8)
    route = relayout.reach(relayout.nest_shared(8, 64),
                           relayout.transposed(8, 64), 8, indices, wave=64)
    assert isinstance(route[0], tuple)
    assert reorder.compose_cost(route) < sum(
        staging.accesses(staging.staged(relayout.nest_shared(8, 64),
                                        relayout.transposed(8, 64), indices)))

    ctx = Context(arch='gfx90a', backend='hip', fp_type=Datatype.F32)
    writer = IRBuilder(Datatype.F32, context=ctx)
    ftype = ScalarType(Datatype.F32)
    regs = [writer.declare(ftype, hint='op') for _ in range(8)]
    out = exchange_codegen.apply_exchange(writer, regs, assembled, ftype)
    assert out is not None and len(out) == 8


@pytest.mark.parametrize('ext', [4, 8, 16])
def test_what_is_emitted_stays_under_what_was_planned(ext):
    """And by exactly the ternaries.

    `Move.cost` counts a `cndmask` region as swaps, a merge and a select,
    because setting the mask is a scalar move beside the merge.  `laneMerge`
    takes it as a template constant, so the IR issues one call and the move is
    the compiler's to hoist out of the loop -- both true, at different levels,
    and the plan being the conservative one is the right direction.
    """
    from tensorforge.backend.instructions.compute.primitives.amd import (
        exchange_codegen, reorder)
    from tensorforge.backend.pir.build import IRBuilder
    from tensorforge.backend.pir.core import ScalarType
    from tensorforge.common.basic_types import Datatype
    from tensorforge.common.context import Context

    indices = _indices(ext, 64)
    assembled = reorder.compose_exchange(relayout.nest_shared(ext, 64),
                                         relayout.transposed(ext, 64),
                                         indices, 64)
    ctx = Context(arch='gfx90a', backend='hip', fp_type=Datatype.F32)
    writer = IRBuilder(Datatype.F32, context=ctx)
    ftype = ScalarType(Datatype.F32)
    regs = [writer.declare(ftype, hint='op') for _ in range(ext)]
    before = len(writer._stack[-1].body)
    exchange_codegen.apply_exchange(writer, regs, assembled, ftype)
    emitted = len(writer._stack[-1].body) - before

    ternaries = sum(1 for group in assembled for move in group
                    if not move.select.free)
    assert emitted == reorder.compose_cost(assembled) - ternaries
    assert ternaries == ext * ext


def test_the_builtin_still_wins_where_there_is_one():
    """One call against sixteen regions: assembling is the fallback for a
    width without a function, not a replacement for one with."""
    for ext in (4, 16):
        assert relayout.has_transpose(ext)
        assert relayout.reach(relayout.nest_shared(ext, 64),
                              relayout.transposed(ext, 64), ext,
                              _indices(ext, 64), wave=64) == 1


def test_without_a_wave_the_middle_rung_is_skipped():
    """A merge is masked by lane, so a caller that does not say how wide the
    wave is cannot be given one."""
    indices = _indices(8, 64)
    route = relayout.reach(relayout.nest_shared(8, 64),
                           relayout.transposed(8, 64), 8, indices)
    assert isinstance(route[0], staging.Transfer)


# -- the packed case is not a case -------------------------------------- #

def _packed(ext, threads):
    """A packed *shared matrix*: the column index inside a register.

    Not what `lead_width` produces -- it packs the leading dimension, which
    the shared matrix does not carry.  Kept because the reduction it shows is
    real and is what a fragment wanting those elements in registers gets;
    `_packed_lead` below is the operand packing actually reaches.
    """
    width = (ext - 1).bit_length()
    return BitLayout((
        tuple(Bit(Place.VECTOR, 1 << b) for b in range(width)),
        tuple(Bit(Place.LANE, 1 << b) for b in range((threads - 1).bit_length())),
    ))


@pytest.mark.parametrize('ext,threads', [(4, 64), (4, 32), (16, 64)])
def test_unpacking_a_packed_operand_yields_the_ordinary_one(ext, threads):
    """Which is why it needs no rung of its own: a vector bit is an element of
    a register and reaching one is a subscript, so it closes first and what is
    left is exactly the distribution the nest hands over anyway."""
    plain, freed = bitlayout.unpacked(_packed(ext, threads))
    assert plain == relayout.nest_shared(ext, threads)
    assert freed == (ext - 1).bit_length()


@pytest.mark.parametrize('ext,threads', [(4, 64), (4, 32), (16, 64)])
def test_the_remaining_gap_is_the_one_that_already_works(ext, threads):
    """So a packed operand costs the extracts and then the transpose that
    runs today -- a cost rather than a case."""
    indices = _indices(ext, threads)
    packed = _packed(ext, threads)
    assert relayout.extracts(packed) == (ext - 1).bit_length()
    assert relayout.reach(packed, relayout.transposed(ext, threads), ext,
                          indices, wave=threads) == 1


def test_an_unpacked_operand_costs_no_extracts():
    assert relayout.extracts(relayout.nest_shared(4, 64)) == 0


def test_what_still_refuses_a_packed_operand_is_the_strategy_layer():
    """`is_contraction` declines `lead_width > 1` for every matrix
    arrangement, so nothing packed reaches a relayout question at all.  That
    is the one line between here and a packed kernel taking a matrix path,
    and flipping it changes generated code."""
    from tensorforge.backend.instructions.compute.strategy import is_contraction
    assert is_contraction(operands=2, lead_width=1)
    assert not is_contraction(operands=2, lead_width=4)


def _packed_lead(width, wave):
    """The lead operand as `lead_width` leaves it.

    A 32-element dimension at width 2 becomes 16 lanes each holding a
    `float2`: the low bit of the index is the element inside the register and
    the rest are lanes.
    """
    low = (width - 1).bit_length()
    lanes = (wave // width - 1).bit_length()
    return BitLayout((
        tuple(Bit(Place.VECTOR, 1 << b) for b in range(low))
        + tuple(Bit(Place.LANE, 1 << b) for b in range(lanes)),))


def _flat_lead(wave):
    """What a B fragment wants: the leading dimension one element per lane."""
    return BitLayout((
        tuple(Bit(Place.LANE, 1 << b) for b in range((wave - 1).bit_length())),))


@pytest.mark.parametrize('width', [2, 4])
def test_unpacking_does_not_close_the_lead_operand_s_gap(width):
    """The correction that matters, and it is the operand `lead_width` packs.

    Its low bits sit inside the register and the fragment wants the leading
    dimension across the lanes, so moving them into slots is the wrong
    direction.  What remains after either reading is a permutation between
    lane weights, which no row of `RELAYOUTS` performs.
    """
    packed = _packed_lead(width, 64)
    flat = _flat_lead(64)
    assert bitlayout.is_exchange(bitlayout.displacement(packed, flat)) is None

    plain, extracts = bitlayout.unpacked(packed)
    assert extracts == (width - 1).bit_length()
    assert bitlayout.is_exchange(bitlayout.displacement(plain, flat)) is None


@pytest.mark.parametrize('width', [2, 4])
def test_so_the_trip_is_what_answers_it(width):
    """Which gives the staged rung the case it was built for, and corrects
    the reading that it had none."""
    indices = _indices(64)
    route = relayout.reach(_packed_lead(width, 64), _flat_lead(64), 4,
                           indices, wave=64)
    assert not isinstance(route, int)
    assert isinstance(route[0], staging.Transfer)


def test_an_unpacked_lead_operand_needs_nothing():
    """Width one is the flat distribution already, which is why this is a
    packing question and not a matrix-path one."""
    assert _packed_lead(1, 64) == _flat_lead(64)
    assert relayout.reach(_packed_lead(1, 64), _flat_lead(64), 4,
                          _indices(64), wave=64) == 0


# -- the reservation ------------------------------------------------------- #

def _shape(width=1, threads=64):
    from tensorforge.backend.instructions.compute.strategy import ComputeShape
    from tensorforge.common.basic_types import Datatype
    return ComputeShape(threads=threads, accumulator=Datatype.F32,
                        sparse=False, explicit_simd=False, lead_width=width)


def test_an_unpacked_operand_reserves_nothing():
    """Every arrangement keeps its operands in registers while they arrive
    unpacked, and the relayouts between register layouts are swaps and
    merges."""
    from tensorforge.backend.instructions.compute.primitives import amd
    from tensorforge.backend.instructions.compute.strategy import Strategy
    assert amd.scratch(Strategy.MATRIX, _shape(width=1), None) == 0
    assert amd.scratch(Strategy.GENERIC, _shape(width=4), None) == 0


@pytest.mark.parametrize('width', [2, 4])
@pytest.mark.parametrize('threads', [32, 64])
def test_a_packed_operand_reserves_one_wave(width, threads):
    """The buffer carries one operand register at a time, so it does not grow
    with the problem -- which is what lets a reservation be made before any
    body exists."""
    from tensorforge.backend.instructions.compute.primitives import amd
    from tensorforge.backend.instructions.compute.strategy import Strategy
    assert amd.scratch(Strategy.MATRIX, _shape(width, threads),
                       None) == threads


@pytest.mark.parametrize('width', [2, 4])
def test_the_reservation_is_the_plan_s_own_size(width):
    """Not a number computed beside it: smaller is an overrun and larger is
    memory nobody writes."""
    from tensorforge.backend.instructions.compute.primitives import amd
    from tensorforge.backend.instructions.compute.strategy import Strategy
    plan = staging.staged(amd._packed_lead(width, 64), amd._flat_lead(64),
                          _indices(64))
    assert amd.scratch(Strategy.MATRIX, _shape(width), None) == \
        staging.buffer_elements(plan)


def test_the_shape_is_built_in_one_place():
    """The plan and the reservation read the same one, or a buffer sized for
    one arrangement meets an emission of another."""
    import inspect
    from tensorforge.backend.instructions.compute import multilinear
    source = inspect.getsource(multilinear.MultilinearInstruction)
    assert source.count('ComputeShape(') == 1
