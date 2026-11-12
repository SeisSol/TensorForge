# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""The two guards of `autotune=lanes` on NVIDIA that the caller may ask for:
the warps a multiprocessor holds (`Options.lanes_occupancy_keep`) and the
spilling that counts as none (`Options.lanes_spill_floor`).

The compiler and the counts are stood in as in `test_lanes_rule`.
"""

from __future__ import annotations

from tensorforge.generators import tuning
from tensorforge.generators.generator import Generator

from test_lanes_rule import (TableScorer, _compiler_asked_throughout,  # noqa: F401
                             _context, _counting, _forget_picks, _key,
                             _local_flux)

COUNTS = {(32, None): 1000, (32, True): 1000, (32, False): 900,
          (16, True): 950, (16, False): 700, (8, True): 800,
          (8, False): 500, (64, True): 1500, (64, False): 300}


def _pick(monkeypatch, scorer, **opts):
    monkeypatch.setattr(tuning, 'executed_instructions', _counting(COUNTS))
    monkeypatch.setattr(tuning.CompiledScore, 'available',
                        lambda self, context: True)
    monkeypatch.setattr(tuning.CompiledScore, 'resources',
                        lambda self, result: scorer.resources(result))
    gen = Generator(_local_flux()(), _context(autotune='lanes', **opts))
    gen.generate()
    return _key(gen.tuned)


def _blocks(eight):
    """Blocks the registers admit: eight for every candidate but the
    eight-lane ones, which hold `eight`."""
    return {key: (eight if key[0] == 8 else 8) for key in COUNTS}


def test_fewer_instructions_for_fewer_warps_are_taken_by_default(monkeypatch):
    assert _pick(monkeypatch, TableScorer({}, waves=_blocks(1))) == (8, False)


def test_a_candidate_holding_too_few_warps_is_left_out(monkeypatch):
    pick = _pick(monkeypatch, TableScorer({}, waves=_blocks(1)),
                 lanes_occupancy_keep=0.75)
    assert pick == (16, False)


def test_a_candidate_holding_as_many_warps_stays_in(monkeypatch):
    pick = _pick(monkeypatch, TableScorer({}, waves=_blocks(8)),
                 lanes_occupancy_keep=0.75)
    assert pick == (8, False)


SPILLS = {(32, None): 100, (32, True): 100, (32, False): 100,
          (8, False): 300, (8, True): 300}


def test_a_default_spilling_a_little_opens_the_wider_counts(monkeypatch):
    assert _pick(monkeypatch, TableScorer(SPILLS)) == (64, False)


def test_below_the_floor_the_default_spills_nothing(monkeypatch):
    """The wider counts stay out, and the narrower ones that spill under the
    floor are in -- priced for what they spill, which keeps sixteen lanes
    ahead of eight here."""
    scorer = TableScorer(SPILLS)
    pick = _pick(monkeypatch, scorer, lanes_spill_floor=512)
    assert pick == (16, False)
    assert (8, False) in scorer.asked


def test_the_guards_keep_their_own_picks(monkeypatch):
    """A walk with a guard is remembered apart from one without."""
    first = _pick(monkeypatch, TableScorer({}, waves=_blocks(1)))
    second = _pick(monkeypatch, TableScorer({}, waves=_blocks(1)),
                   lanes_occupancy_keep=0.75)
    assert (first, second) == ((8, False), (16, False))
