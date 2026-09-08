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


def test_the_assembled_exchange_is_priced_and_not_emittable():
    """Cheaper than the trip and still not offered.

    A transpose's regions are one lane out of every `ext` -- one per bank,
    which neither a `dppUpdate` row mask nor a bank mask expresses -- so the
    merge would need a ternary on the lane id.  `Select` reports that and the
    merge refuses it, which is the module's own policy: the path reads no lane
    id anywhere else and one appearing is a thing to look at.

    So the cost stands as a statement of what a merge primitive would buy, and
    the route offered is the trip."""
    from tensorforge.backend.instructions.compute.primitives.amd import (
        exchange_codegen, reorder)
    indices = _indices(8, 64)
    assembled = reorder.compose_exchange(relayout.nest_shared(8, 64),
                                         relayout.transposed(8, 64),
                                         indices, 64)
    assert not reorder.emittable(assembled)
    assert reorder.compose_cost(assembled) < sum(
        staging.accesses(staging.staged(relayout.nest_shared(8, 64),
                                        relayout.transposed(8, 64), indices)))

    assert not relayout.has_transpose(8)
    route = relayout.reach(relayout.nest_shared(8, 64),
                           relayout.transposed(8, 64), 8, indices, wave=64)
    assert isinstance(route[0], staging.Transfer)


def test_the_emitter_refuses_a_mask_that_does_not_exist():
    """Rather than returning a register with a hole in it."""
    from tensorforge.backend.instructions.compute.primitives.amd import (
        exchange_codegen, reorder)
    from tensorforge.backend.pir.build import IRBuilder
    from tensorforge.backend.pir.core import ScalarType
    from tensorforge.common.basic_types import Datatype
    from tensorforge.common.context import Context

    ctx = Context(arch='gfx90a', backend='hip', fp_type=Datatype.F32)
    assembled = reorder.compose_exchange(relayout.nest_shared(4, 64),
                                         relayout.transposed(4, 64),
                                         _indices(4, 64), 64)
    writer = IRBuilder(Datatype.F32, context=ctx)
    ftype = ScalarType(Datatype.F32)
    regs = [writer.declare(ftype, hint='op') for _ in range(4)]
    assert exchange_codegen.apply_exchange(writer, regs, assembled,
                                           ftype) is None


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
