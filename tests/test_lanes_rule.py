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
    """Reports the spilling `spills`, the code `code` and the occupancy
    `waves` name for a candidate's (lanes, merge), and counts the
    questions."""

    def __init__(self, spills, code=None, waves=None):
        self.spills = spills
        self.code = code or {}
        self.waves = waves or {}
        self.asked = []

    def resources(self, result):
        key = _key(result.candidate)
        self.asked.append(key)
        return Resources(registers=255, spill_bytes=self.spills.get(key, 0),
                         code_bytes=self.code.get(key),
                         register_blocks=self.waves.get(key))


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


@pytest.mark.parametrize('arch,taken', [('sm_90', False), ('sm_120', True)])
def test_code_past_the_cache_costs_what_the_part_measured(monkeypatch, arch,
                                                          taken):
    """32 lanes unmerged count 30 % fewer instructions and are 256 kB of
    code, half of it past the cache: on GH200, whose probe ran three times
    slower past it, that is no move; on sm_120, at 2.05, it is.  Order-8
    `localFluxAll` in single precision written out is 26 % slower than merged
    on GH200 and 10 % faster on sm_120."""
    ctx = _context(arch)
    descr_factory = _local_flux()
    origin, first = _origin(descr_factory, ctx)
    counts = {(32, None): 1000, (32, True): 1000, (32, False): 700,
              (16, True): 1200, (16, False): 1200, (8, True): 1300,
              (8, False): 1300, (64, True): 1500, (64, False): 1500}
    monkeypatch.setattr(tuning, 'executed_instructions', _counting(counts))
    scorer = TableScorer({}, code={(32, None): 60 * 1024,
                                   (32, False): 256 * 1024})
    pick = tuning.lanes_rule(descr_factory, ctx, scorer, origin, first,
                             generate=Generator.trial)
    assert (_key(pick) == (32, False)) == taken
    assert pick == origin or taken


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


def test_without_a_compiler_or_off_the_measured_lowerings_the_default_stands(
        monkeypatch):
    monkeypatch.setattr(tuning.CompiledScore, 'available',
                        lambda self, context: False)
    with pytest.warns(UserWarning, match='no compiler'):
        assert tuning.autotune(_local_flux(), _context(autotune='lanes'),
                               mode='lanes', generate=Generator.trial) is None
    with pytest.warns(UserWarning, match='no compiler'):
        assert tuning.autotune(_local_flux(), _amd(autotune='lanes'),
                               mode='lanes', generate=Generator.trial) is None
    for backend in ('esimd', 'acpp'):
        intel = Context(arch='pvc', backend=backend, fp_type=Datatype.F32,
                        options=Options(autotune='lanes'))
        with pytest.warns(UserWarning, match='not measured'):
            assert tuning.autotune(_local_flux(), intel, mode='lanes',
                                   generate=Generator.trial) is None


def _amd(arch='gfx942', **opts):
    return Context(arch=arch, backend='hip', fp_type=Datatype.F32,
                   options=Options(**opts))


def _bounds(table):
    """`pipe_bound` read from `table` by (lanes, merge), the spilling added
    as it is: on top."""
    def bound(result, spill_bytes=0.0):
        return table[_key(result.candidate)] + spill_bytes
    return bound


def test_amd_ranks_by_the_pipe_bound(monkeypatch):
    """Eight lanes count the fewest instructions and sixteen merged have the
    lowest bound: on AMD the bound decides.  Order-6 `volume` in single
    precision on MI300A, where eight lanes run 58 % behind sixteen."""
    ctx = _amd()
    descr_factory = _local_flux()
    origin, first = _origin(descr_factory, ctx)
    counts = {(32, None): 1160, (32, True): 1160, (32, False): 1150,
              (16, True): 920, (16, False): 930, (8, True): 740,
              (8, False): 744, (64, True): 1250, (64, False): 1260}
    bounds = {(32, None): 1273, (32, True): 1273, (32, False): 1270,
              (16, True): 1020, (16, False): 1030, (8, True): 1176,
              (8, False): 1180, (64, True): 1443, (64, False): 1450}
    monkeypatch.setattr(tuning, 'executed_instructions', _counting(counts))
    monkeypatch.setattr(tuning, 'pipe_bound', _bounds(bounds))
    pick = tuning.lanes_rule(descr_factory, ctx, TableScorer({}), origin,
                             first, generate=Generator.trial)
    assert _key(pick) == (16, True)
    nvidia = _context()
    origin, first = _origin(descr_factory, nvidia)
    pick = tuning.lanes_rule(descr_factory, nvidia, TableScorer({}), origin,
                             first, generate=Generator.trial)
    assert _key(pick) == (8, True)


def test_on_amd_the_spilling_goes_into_the_bound(monkeypatch):
    """The default spills, so the guard admits a candidate that spills too;
    its spilling raises its bound past one that spills nothing, and the
    compiler is asked in the order of the bounds without it."""
    ctx = _amd()
    descr_factory = _local_flux()
    origin, first = _origin(descr_factory, ctx)
    bounds = {(32, None): 1200, (32, True): 1200, (32, False): 1190,
              (16, True): 950, (16, False): 900, (8, True): 1300,
              (8, False): 1300, (64, True): 1500, (64, False): 1500}
    scorer = TableScorer({(32, None): 100, (16, False): 140})
    monkeypatch.setattr(tuning, 'pipe_bound', _bounds(bounds))
    pick = tuning.lanes_rule(descr_factory, ctx, scorer, origin, first,
                             generate=Generator.trial)
    assert _key(pick) == (16, True)
    assert scorer.asked[:3] == [(32, None), (16, False), (16, True)]


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


def _pvc(**opts):
    return Context(arch='pvc', backend='oneapi', fp_type=Datatype.F32,
                   options=Options(**opts))


def test_under_dpcpp_the_walk_starts_at_the_sixteen_lanes_it_builds():
    """The target's count (`Preferences.lanes_per_mult`) is where the
    generator builds unasked, so it is the default the rule weighs the rest
    against -- and not the deduction, 64 lanes for these 40 rows."""
    descrs = _local_flux()()
    origin = tuning.start(descrs, _pvc())
    assert origin.lanes.num_threads == 16
    assert tuning.start(descrs, _context()).lanes.num_threads == 32
    cands = tuning.lane_candidates(descrs, _pvc(), origin)
    assert {_key(c)[0] for c in cands} == {16, 32, 64}


def test_under_dpcpp_the_rule_ranks_by_the_count(monkeypatch):
    """Sixteen lanes spill where 32 do not, and the price of it moves the
    kernel: order-6 `neighboringFlux` in single precision on the Max 1550,
    11.8 kB of IGC spill at sixteen lanes and none at 32, 8 % faster at 32."""
    ctx = _pvc()
    descr_factory = _local_flux()
    origin, first = _origin(descr_factory, ctx)
    assert _key(origin)[0] == 16
    counts = {(16, None): 3618, (16, True): 3618, (16, False): 3618,
              (32, True): 4945, (32, False): 4945, (64, True): 8475,
              (64, False): 8475}
    monkeypatch.setattr(tuning, 'executed_instructions', _counting(counts))
    scorer = TableScorer({(16, None): 11776, (16, True): 11776,
                          (16, False): 11776})
    pick = tuning.lanes_rule(descr_factory, ctx, scorer, origin, first,
                             generate=Generator.trial)
    assert _key(pick)[0] == 32


def test_on_mi210_fewer_waves_stretch_the_bound(monkeypatch):
    """The same bounds as on MI300A, where sixteen merged lanes win; on MI210
    a build at one wave a SIMD is charged eight times its bound and one at
    two four times, so 64 lanes at two waves win -- order-6 `derivative` in
    single precision runs 33 % faster at 64 lanes there and twice as long at
    sixteen."""
    descr_factory = _local_flux()
    bounds = {(32, None): 4449, (32, True): 4449, (32, False): 4449,
              (16, True): 3488, (16, False): 3488, (8, True): 2898,
              (8, False): 2898, (64, True): 5665, (64, False): 5665}
    waves = {(32, None): 1, (32, True): 1, (32, False): 1, (16, True): 1,
             (16, False): 1, (8, True): 1, (8, False): 1, (64, True): 2,
             (64, False): 2}
    spills = {(8, True): 668, (8, False): 668}
    monkeypatch.setattr(tuning, 'pipe_bound', _bounds(bounds))
    picks = {}
    for arch in ('gfx942', 'gfx90a'):
        ctx = _amd(arch)
        origin, first = _origin(descr_factory, ctx)
        picks[arch] = _key(tuning.lanes_rule(
            descr_factory, ctx, TableScorer(spills, waves=waves), origin,
            first, generate=Generator.trial))[0]
    assert picks == {'gfx942': 16, 'gfx90a': 64}


def test_an_amd_part_nobody_measured_keeps_its_default(monkeypatch):
    monkeypatch.setattr(tuning.CompiledScore, 'available',
                        lambda self, context: True)
    with pytest.warns(UserWarning, match='not measured'):
        assert tuning.autotune(_local_flux(), _amd('gfx1150', autotune='lanes'),
                               mode='lanes', generate=Generator.trial) is None
