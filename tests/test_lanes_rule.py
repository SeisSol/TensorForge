# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""`autotune=lanes`: the lane count and merging, with the compiler asked
about spilling only.

The rule (`tuning.lanes_rule`) needs a compiler but no GPU, so these tests
stand one in: a scorer whose `resources` reports the spilling a table says,
and the emitter's count taken from a table too where the point is the
ranking rather than the count.
"""

from __future__ import annotations

import pytest

from tensorforge.common.basic_types import Datatype
from tensorforge.common.context import Context, Options
from tensorforge.generators import lanes as lane_config
from tensorforge.generators import tuning
from tensorforge.generators.generator import Generator
from tensorforge.toolchain import Resources

def _local_flux(m=40, n=4, faces=4):
    """`tests/cases/local_flux.py` at a size a test can build eight times:
    `faces` faces of ``R += (A_f @ B) @ C_f`` over `m` rows -- 40, so that
    the deduced 32 lanes have a wider count above them (64) -- and `n`
    columns."""
    from tensorforge.common.basic_types import Addressing
    from tensorforge.common.matrix.boundingbox import BoundingBox
    from tensorforge.common.matrix.tensor import SubTensor, Tensor
    from tensorforge.generators.descriptions import MultilinearDescr

    def tensor(shape, alias, addressing, is_tmp=False):
        return Tensor(list(shape), addressing, BoundingBox([0, 0], list(shape)),
                      alias=alias, datatype=Datatype.F32, is_tmp=is_tmp)

    def descr_list():
        b = tensor([m, n], "B", Addressing.PTR_BASED)
        r = tensor([m, n], "R", Addressing.PTR_BASED)
        x = tensor([m, n], "X", Addressing.STRIDED, is_tmp=True)
        out = []
        for f in range(faces):
            a = tensor([m, m], f"A{f}", Addressing.NONE)
            c = tensor([n, n], f"C{f}", Addressing.PTR_BASED)
            out.append(MultilinearDescr(
                dest=SubTensor(x), ops=[SubTensor(a), SubTensor(b)],
                target=[[0, -1], [-1, 1]], permute=[[0, 1], [0, 1]]))
            out.append(MultilinearDescr(
                dest=SubTensor(r), ops=[SubTensor(x), SubTensor(c)],
                target=[[0, -1], [-1, 1]], permute=[[0, 1], [0, 1]],
                add=(f > 0)))
        return out
    return descr_list


def _context(arch="sm_120", **opts):
    return Context(arch=arch, backend="cuda", fp_type=Datatype.F32,
                   options=Options(**opts))


@pytest.fixture(autouse=True)
def _forget_picks():
    tuning._PICKS.clear()
    tuning._LOADED.clear()
    yield
    tuning._PICKS.clear()
    tuning._LOADED.clear()


def _key(cand):
    lanes = cand.lanes.num_threads if cand.lanes is not None else None
    return lanes, cand.get('merge_variants')


class TableScorer:
    """Reports the spilling `spills` names for a candidate's (lanes, merge),
    and counts the questions."""

    def __init__(self, spills):
        self.spills = spills
        self.asked = []

    def resources(self, result):
        key = _key(result.candidate)
        self.asked.append(key)
        return Resources(registers=255, spill_bytes=self.spills.get(key, 0))


def _counting(table):
    """`executed_instructions` read from `table` by (lanes, merge)."""
    def count(result):
        return table[_key(result.candidate)]
    return count


def _origin(descr_factory, context):
    origin = tuning.start(descr_factory(), context)
    first = tuning.build(descr_factory, context, origin, Generator.trial)
    assert first.ok
    return origin, first


def test_the_candidates_are_the_lane_counts_times_merging():
    """Powers of two from the deduced count down to eight, the wider count up
    to the extent's power of two (40 rows: 64), each merged and not -- the
    four faces repeat."""
    ctx = _context()
    descrs = _local_flux()()
    origin = tuning.start(descrs, ctx)
    cands = tuning.lane_candidates(descrs, ctx, origin)
    assert {_key(c) for c in cands} == {
        (lanes, merge) for lanes in (8, 16, 32, 64) for merge in (False, True)}


def test_a_count_the_caller_fixed_leaves_only_merging():
    ctx = _context()
    descrs = _local_flux()()
    fixed = {'lanes': lane_config.LaneConfig(16, 40, 1)}
    origin = tuning.start(descrs, ctx).set('lanes', fixed['lanes'])
    cands = tuning.lane_candidates(descrs, ctx, origin, fixed)
    assert {_key(c) for c in cands} == {(16, False), (16, True)}


def test_the_fewest_instructions_inside_the_guard_win(monkeypatch):
    """Eight lanes unmerged count the fewest, but spill where the default
    does not: out.  Sixteen unmerged is next and spills nothing: taken."""
    ctx = _context()
    descr_factory = _local_flux()
    origin, first = _origin(descr_factory, ctx)
    counts = {(32, None): 1000, (32, True): 1000, (32, False): 900,
              (16, True): 950, (16, False): 700, (8, True): 800,
              (8, False): 500, (64, True): 1500, (64, False): 1400}
    monkeypatch.setattr(tuning, 'executed_instructions', _counting(counts))
    scorer = TableScorer({(8, False): 300})
    pick = tuning.lanes_rule(descr_factory, ctx, scorer, origin, first,
                             generate=Generator.trial)
    assert _key(pick) == (16, False)


def test_the_compiler_is_asked_no_further_than_the_bound(monkeypatch):
    """By count: the default, then 16 unmerged (taken, nothing spilled at a
    count of 700), and then 32 unmerged at 900 is past that cost -- nothing
    after it is asked about."""
    ctx = _context()
    descr_factory = _local_flux()
    origin, first = _origin(descr_factory, ctx)
    counts = {(32, None): 1000, (32, True): 1000, (32, False): 900,
              (16, True): 950, (16, False): 700, (8, True): 1200,
              (8, False): 1100, (64, True): 1500, (64, False): 1400}
    monkeypatch.setattr(tuning, 'executed_instructions', _counting(counts))
    scorer = TableScorer({})
    tuning.lanes_rule(descr_factory, ctx, scorer, origin, first,
                      generate=Generator.trial)
    assert scorer.asked == [(32, None), (16, False)]


def test_spilling_is_priced_where_the_default_spills(monkeypatch):
    """A default that spills lets candidates spill as much: then the count
    alone would take 16 lanes unmerged, and the price of its spilling puts 64
    unmerged, which spills nothing, ahead -- order-8 `localFluxAll` in double
    precision on A100, 131 % between the two."""
    ctx = _context()
    descr_factory = _local_flux()
    origin, first = _origin(descr_factory, ctx)
    counts = {(32, None): 30000, (32, True): 30000, (32, False): 20000,
              (16, True): 26000, (16, False): 15000, (8, True): 22000,
              (8, False): 14000, (64, True): 38000, (64, False): 22000}
    spills = {(32, None): 4800, (32, True): 4800, (32, False): 2000,
              (16, True): 7000, (16, False): 4000, (8, True): 9000,
              (8, False): 9000, (64, True): 0, (64, False): 0}
    monkeypatch.setattr(tuning, 'executed_instructions', _counting(counts))
    pick = tuning.lanes_rule(descr_factory, ctx, TableScorer(spills), origin,
                             first, generate=Generator.trial)
    assert _key(pick) == (64, False)


def test_a_move_within_the_margin_keeps_the_default(monkeypatch):
    ctx = _context()
    descr_factory = _local_flux()
    origin, first = _origin(descr_factory, ctx)
    counts = {(32, None): 1000, (32, True): 1000, (32, False): 980,
              (16, True): 1100, (16, False): 1100, (8, True): 1200,
              (8, False): 1200, (64, True): 1500, (64, False): 1400}
    monkeypatch.setattr(tuning, 'executed_instructions', _counting(counts))
    pick = tuning.lanes_rule(descr_factory, ctx, TableScorer({}), origin,
                             first, generate=Generator.trial)
    assert pick == origin


def test_the_emitters_count_follows_the_trips():
    """What the rule ranks by is executed, not listed: merged, the four faces
    are one body run four times, and the count says about as much as the
    faces written out -- the listing of the merged body would say a quarter."""
    ctx = _context()
    descr_factory = _local_flux()
    lanes = tuning.start(descr_factory(), ctx).lanes
    merged = tuning.build(descr_factory, ctx, tuning.Candidate(lanes=lanes)
                          .set('merge_variants', True), Generator.trial)
    written = tuning.build(descr_factory, ctx, tuning.Candidate(lanes=lanes)
                           .set('merge_variants', False), Generator.trial)
    m, w = (tuning.executed_instructions(merged),
            tuning.executed_instructions(written))
    assert m and w and 0.5 < m / w < 2.0


def test_without_a_compiler_or_off_nvidia_the_default_stands(monkeypatch):
    monkeypatch.setattr(tuning.CompiledScore, 'available',
                        lambda self, context: False)
    with pytest.warns(UserWarning, match='no compiler'):
        assert tuning.autotune(_local_flux(), _context(autotune='lanes'),
                               mode='lanes', generate=Generator.trial) is None
    amd = Context(arch='gfx942', backend='hip', fp_type=Datatype.F32,
                  options=Options(autotune='lanes'))
    with pytest.warns(UserWarning, match='NVIDIA only'):
        assert tuning.autotune(_local_flux(), amd, mode='lanes',
                               generate=Generator.trial) is None


def test_the_generator_builds_the_pick_and_remembers_it(monkeypatch):
    """Through `Options.autotune`: the kernel is built at the rule's pick, and
    a second generation of the same kernel asks the compiler nothing."""
    counts = {(32, None): 1000, (32, True): 1000, (32, False): 900,
              (16, True): 950, (16, False): 700, (8, True): 800,
              (8, False): 500, (64, True): 1500, (64, False): 1400}
    monkeypatch.setattr(tuning, 'executed_instructions', _counting(counts))
    scorer = TableScorer({(8, False): 300})
    monkeypatch.setattr(tuning.CompiledScore, 'available',
                        lambda self, context: True)
    monkeypatch.setattr(tuning.CompiledScore, 'resources',
                        lambda self, result: scorer.resources(result))
    gen = Generator(_local_flux()(), _context(autotune='lanes'))
    gen.generate()
    assert gen.tuned is not None and _key(gen.tuned) == (16, False)
    assert gen.lanes.num_threads == 16
    asked = len(scorer.asked)
    again = Generator(_local_flux()(), _context(autotune='lanes'))
    again.generate()
    assert _key(again.tuned) == (16, False) and len(scorer.asked) == asked
