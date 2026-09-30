# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Whose registers `pressure(in_bytes=True)` counts.

The byte form sums two things: register arrays, which are per lane, and live
SSA values, which `register_bytes` scales by the lane axis only under an
explicit vector.  There that is right -- one work-item holds the whole wave,
so a value really does occupy `lanes` of its registers.  Under SPMD it is not:
the lane axis is threads, each holding one, and multiplying by it would report
the wave's total register file and put the sum in two different units.

That has a direction, which is what makes it worth a test rather than a
comment.  Doubling the lane count halves every array and doubles the
wave-total per-value figure, so the two nearly cancel and the total barely
moves -- while the thing a caller wants, one lane's footprint, halves.  A
search minimizing the wave total would pick 32 lanes on gfx90a precisely where
64 is what relieves the pressure.
"""

from __future__ import annotations

import pytest

from tensorforge.backend.pir.core import (BufferType, MemSpace, RegisterLayout,
                                          LaneAxis, ScalarType, Value)
from tensorforge.backend.pir.passes import pressure, register_bytes
from tensorforge.common.basic_types import Datatype


def _scalar(lanes: int | None, length: int = 1) -> Value:
    return Value(id=1, type=ScalarType(Datatype.F64, length),
                 layout=(RegisterLayout((LaneAxis(lanes, 1),))
                         if lanes is not None else None))


def _array(volume: int) -> Value:
    return Value(id=2, type=BufferType(Datatype.F64, (volume,),
                                       MemSpace.REGISTER))


def test_a_value_costs_one_lane_under_spmd_and_a_wave_under_explicit_simd():
    v = _scalar(lanes=64)
    assert register_bytes(v, explicit_simd=False) == 8
    assert register_bytes(v, explicit_simd=True) == 64 * 8


def test_an_array_is_per_lane_either_way():
    """Its volume is what one lane holds, whichever way the body is lowered.

    Which is exactly why the scalar half has to be asked: two terms of one sum
    cannot be in different units.
    """
    a = _array(74)
    assert register_bytes(a, explicit_simd=False) == 74 * 8
    assert register_bytes(a, explicit_simd=True) == 74 * 8


def test_an_untracked_layout_still_counts_as_one_lane():
    """A floor, not an estimate: it is what SPMD would need either way."""
    v = _scalar(lanes=None)
    assert register_bytes(v, explicit_simd=True) == 8
    assert register_bytes(v, explicit_simd=False) == 8


def test_the_slot_axis_survives_both():
    """`ScalarType.length` is how many elements one lane holds consecutively.

    A different axis from the lane one, and it makes the register bigger on
    both paths.
    """
    v = _scalar(lanes=None, length=4)
    assert register_bytes(v, explicit_simd=False) == 4 * 8


@pytest.mark.parametrize("values_halve", [True, False],
                         ids=["values halve", "values stay"])
def test_the_wave_total_barely_moves_where_the_per_lane_figure_halves(
        values_halve):
    """The property that makes the setting decide an answer, not sharpen one.

    Doubling the lanes halves every register array.  Under the wave-total
    reading each live value also doubles, so that term scales as
    `lanes * count(lanes)`: exactly flat where the count halves with the lanes
    -- an unrolled load per slot -- and doubled where it does not, as in a
    reduction whose live set is the accumulator.

    So the only movement the wave total has left is the array's, and the array
    is the smaller of the two terms.  It understates the effect by their ratio,
    and in the corpus that leaves it useless: measured over the cases where the
    ceiling binds on gfx90a, the median ratio at 64 lanes against 32 is 1.03
    for the wave total against 0.57 per lane -- and 0.57 is what the register
    slots do, which is what the figure is supposed to track.
    """
    slots_32, values_32 = 148, 40
    slots_64 = slots_32 // 2
    values_64 = values_32 // 2 if values_halve else values_32

    def total(lanes, slots, values, explicit_simd):
        return (register_bytes(_array(slots), explicit_simd)
                + values * register_bytes(_scalar(lanes), explicit_simd))

    per_lane = (total(32, slots_32, values_32, False),
                total(64, slots_64, values_64, False))
    wave = (total(32, slots_32, values_32, True),
            total(64, slots_64, values_64, True))

    assert per_lane[1] <= 0.65 * per_lane[0], (
        "per lane, the wider configuration wins by a lot")
    assert wave[1] > 0.9 * wave[0], (
        "as a wave total the same change is noise or a loss, which is the "
        "reading that would have driven a search to the narrower answer")


def test_a_broadcast_the_target_folds_holds_no_register():
    """On Intel a broadcast of one lane's element is a region, not a value.

    `r20.3<0;1,0>` reads one element of a register and spreads it over the
    instruction's lanes, so it occupies nothing of its own -- the order-6
    derivative's simd32 build has 8033 such regions and no message that is not
    a spill.  Counted as values they would be more than half of the bytes a
    lane that the kernel is judged by.
    """
    from tensorforge.backend.pir.build import IRBuilder
    from tensorforge.backend.pir.core import MemSpace

    def built():
        b = IRBuilder(fptype=Datatype.F32)
        g = b.alloc(Datatype.F32, (16,), MemSpace.GLOBAL, hint='g')
        loaded = b.load(g, 0)
        shared = b.rawexpr('broadcast({0})', loaded, type_=ScalarType(Datatype.F32),
                           hint='bc', pure=True, crosslane=True)
        for _ in range(4):
            b.store(g, shared, 1)
        return b.finish()

    body = built()
    counted = pressure(body, in_bytes=True, explicit_simd=False)
    folded = pressure(body, in_bytes=True, explicit_simd=False,
                      folded_crosslane=True)
    assert folded < counted, 'the broadcast is a region there, not a value'
    # its source still holds one: the region reads that register
    assert folded > 0


def test_which_targets_fold_a_broadcast():
    """Intel under SPMD does; an instruction that writes a register does not."""
    from tensorforge.common.context import Context

    def lexic(arch, backend):
        return Context(arch=arch, backend=backend,
                       fp_type=Datatype.F32).get_vm().get_lexic()

    assert lexic('pvc', 'oneapi').folds_broadcast()
    assert not lexic('pvc', 'esimd').folds_broadcast()
    assert not lexic('sm_90', 'cuda').folds_broadcast()
    assert not lexic('gfx942', 'hip').folds_broadcast()
