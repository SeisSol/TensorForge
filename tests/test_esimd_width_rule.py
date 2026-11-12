# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""The vector width under the explicit vector (`tuning.esimd_width_rule`).

The widest width VC builds without spilling, down to one register's worth.
Where every width spills, the one that spills least in double precision, and
the default in single.  The spills below are the ones IGC 2.41.5 reported for
SeisSol's elastic kernels on PVC; the scorer here only replays them.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tensorforge.common.basic_types import Datatype
from tensorforge.common.context import Context
from tensorforge.generators import tuning
from tensorforge.generators.lanes import LaneConfig


class _Replay:
    """`CompiledScore.resources` from a table: lanes -> spilled bytes."""

    def __init__(self, spill):
        self.spill = spill
        self.asked = []

    def resources(self, result):
        lanes = result.candidate.lanes.num_threads
        self.asked.append(lanes)
        return SimpleNamespace(spill_bytes=self.spill[lanes])


def _generate(descrs, context, lanes):
    return SimpleNamespace(lanes=lanes)


def _pick(fp_type, spill, default=32, active=35):
    ctx = Context(arch='pvc', backend='esimd', fp_type=fp_type)
    origin = tuning.Candidate(lanes=LaneConfig(default, active, 1))
    first = tuning.build(lambda: [], ctx, origin, _generate)
    scorer = _Replay(spill)
    pick = tuning.esimd_width_rule(lambda: [], ctx, scorer, origin, first,
                                   generate=_generate)
    return pick.lanes.num_threads, scorer.asked


def test_a_default_that_spills_nothing_costs_one_compile():
    # Order-5 `neighboringFlux` in double precision: nothing spills.
    assert _pick(Datatype.F64, {32: 0, 16: 0, 8: 0}) == (32, [32])


def test_the_widest_that_spills_nothing():
    # Order-5 `volume` in double precision.
    assert _pick(Datatype.F64, {32: 2496, 16: 0, 8: 0}) == (16, [32, 16])


@pytest.mark.parametrize('spill', [
    {32: 13824, 16: 10368, 8: 4288},       # order 5 `derivative`
    {32: 33792, 16: 40384, 8: 22016},      # order 6 `derivative`
    {32: 14016, 16: 12288, 8: 6912},       # order 6 `localFluxAll`
])
def test_where_every_width_spills_double_precision_takes_the_least(spill):
    assert _pick(Datatype.F64, spill) == (8, [32, 16, 8])


def test_where_every_width_spills_single_precision_keeps_the_default():
    # Order-6 `volume` in single precision; eight floats are half a
    # register, so eight lanes are not asked about.
    assert _pick(Datatype.F32, {32: 5824, 16: 3264, 8: 4224}) == (32, [32, 16])


def test_one_register_is_the_floor():
    ctx = Context(arch='pvc', backend='esimd', fp_type=Datatype.F64)
    assert tuning._register_lanes(ctx) == 8
    ctx = Context(arch='pvc', backend='esimd', fp_type=Datatype.F32)
    assert tuning._register_lanes(ctx) == 16
