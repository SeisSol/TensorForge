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


@pytest.fixture(autouse=True)
def _compiler_asked_throughout(request, monkeypatch):
    """The stand-in compiler answers for every candidate, as the rule asks it
    where the estimate cannot settle the spilling (`tuning.ASK_BAND`) -- but
    in the tests about that band."""
    if 'band' not in request.node.name:
        monkeypatch.setattr(tuning, '_settled', lambda result: None)


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


def test_where_the_default_spills_on_amd_a_clean_build_comes_first(monkeypatch):
    """Order-8 `derivative` in single precision on MI300A: 32 lanes spill
    1340 B, which the bound charges 8 %, sixteen spill past the guard, and 64
    spill nothing at an 18 % higher count -- and run 38 % ahead of 32.
    The compiler is asked past the bound until a clean build turns up.  On
    NVIDIA the count and the price of the spilling decide as before."""
    descr_factory = _local_flux()
    bounds = {(32, None): 10603, (32, True): 10603, (32, False): 10603,
              (16, True): 8909, (16, False): 8909, (8, True): 7328,
              (8, False): 7328, (64, True): 12511, (64, False): 12511}
    spills = {(32, None): 1340, (32, True): 1340, (32, False): 1340,
              (16, True): 3552, (16, False): 3552, (8, True): 7476,
              (8, False): 7476}
    monkeypatch.setattr(tuning, 'pipe_bound', _bounds(bounds))
    ctx = _amd()
    origin, first = _origin(descr_factory, ctx)
    scorer = TableScorer(spills)
    pick = tuning.lanes_rule(descr_factory, ctx, scorer, origin, first,
                             generate=Generator.trial)
    assert _key(pick)[0] == 64
    # Eight lanes in single precision are out on AMD before anything is
    # asked (`AMD_SINGLE_NARROWEST`).
    assert {k[0] for k in scorer.asked} == {16, 32, 64}
    monkeypatch.setattr(tuning, 'CLEAN_FIRST', False)
    pick = tuning.lanes_rule(descr_factory, ctx, TableScorer(spills), origin,
                             first, generate=Generator.trial)
    assert _key(pick)[0] == 32
    # On NVIDIA the price of the spilling decides: 64 lanes counting more
    # than 32 lanes' price stay out, clean or not.
    monkeypatch.setattr(tuning, 'CLEAN_FIRST', True)
    counts = dict(bounds)
    counts[(64, True)] = counts[(64, False)] = 40000
    monkeypatch.setattr(tuning, 'executed_instructions', _counting(counts))
    nvidia = _context()
    origin, first = _origin(descr_factory, nvidia)
    pick = tuning.lanes_rule(descr_factory, nvidia, TableScorer(spills),
                             origin, first, generate=Generator.trial)
    assert _key(pick)[0] == 32


def test_at_one_wave_on_cdna_every_instruction_takes_a_turn(monkeypatch):
    """Viscoelastic order-4 `fluxLocalAll` in double precision on MI300A:
    every build of up to 32 lanes holds one wave a SIMD, and sixteen merged
    lanes are 3 % below the default's bound -- inside the margin -- and 13 %
    below its count, and run 21 % faster.  At one wave the count over the
    waves is a floor of the cost, and it moves the pick; by the bound alone
    the default stands."""
    descr_factory = _local_flux()
    bounds = {(32, None): 1756, (32, True): 1756, (32, False): 1753,
              (16, True): 1703, (16, False): 1678, (8, True): 2333,
              (8, False): 2121, (64, True): 4084, (64, False): 3506}
    counts = {(32, None): 2193, (32, True): 2193, (32, False): 2090,
              (16, True): 1904, (16, False): 1804, (8, True): 2476,
              (8, False): 2207, (64, True): 5092, (64, False): 4180}
    waves = {key: 1 for key in bounds}
    waves[(64, True)], waves[(64, False)] = 3, 2
    spills = {(16, False): 36, (8, True): 284, (8, False): 5032}
    monkeypatch.setattr(tuning, 'pipe_bound', _bounds(bounds))
    monkeypatch.setattr(tuning, 'executed_instructions', _counting(counts))
    ctx = _amd()
    origin, first = _origin(descr_factory, ctx)
    pick = tuning.lanes_rule(descr_factory, ctx,
                             TableScorer(spills, waves=waves), origin, first,
                             generate=Generator.trial)
    assert _key(pick) == (16, True)
    monkeypatch.setattr(tuning, 'ISSUE_PER_WAVE', ())
    pick = tuning.lanes_rule(descr_factory, ctx,
                             TableScorer(spills, waves=waves), origin, first,
                             generate=Generator.trial)
    assert _key(pick) == (32, None)


def test_where_nothing_up_to_the_extent_is_clean_the_wave_is_asked(
        monkeypatch):
    """Viscoelastic order-4 `derivative` in double precision on MI300A
    spreads 20 rows: every candidate up to 32 lanes spills, and 64 -- one
    multiplication a wave, past the extent -- spills nothing and runs 30 %
    ahead of the 32 lanes the walk takes without it.  Where something up to
    the extent is clean, the wave is not asked about."""
    descr_factory = _local_flux(m=20)
    ctx = _amd()
    origin, first = _origin(descr_factory, ctx)
    bounds = {(32, None): 3666, (32, True): 3666, (32, False): 3469,
              (16, True): 3325, (16, False): 2895, (8, True): 3901,
              (8, False): 3433, (64, True): 7331, (64, False): 6937}
    spills = {(32, None): 716, (32, True): 716, (32, False): 952,
              (16, True): 1148, (16, False): 1388, (8, True): 2052,
              (8, False): 6416}
    monkeypatch.setattr(tuning, 'pipe_bound', _bounds(bounds))
    scorer = TableScorer(spills)
    pick = tuning.lanes_rule(descr_factory, ctx, scorer, origin, first,
                             generate=Generator.trial)
    assert _key(pick) == (64, False)
    assert {(64, False), (64, True)} <= set(scorer.asked)
    clean = dict(spills)
    clean[(16, True)] = 0
    scorer = TableScorer(clean)
    pick = tuning.lanes_rule(descr_factory, ctx, scorer, origin, first,
                             generate=Generator.trial)
    assert _key(pick) == (16, True)
    assert not {k for k in scorer.asked if k[0] == 64}
    monkeypatch.setattr(tuning, 'CLEAN_WAVE', False)
    pick = tuning.lanes_rule(descr_factory, ctx, TableScorer(spills), origin,
                             first, generate=Generator.trial)
    assert _key(pick)[0] == 32


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
    two four times, which made 64 lanes at two waves win -- but where the
    default spills nothing, MI210 takes no wider candidate (`WIDE_SPILL`: 25
    such picks lost and won to -0.1 % together over paper-1106, worst
    +371 %).  A narrower one it takes as MI300A does where the code stays in
    CDNA 2's 32 KB of instruction cache, and not where it takes the code
    across (`ICACHE_RAMP`), which MI300A's 64 KB do not see.  Spilling, the
    stretch decides again."""
    descr_factory = _local_flux()
    bounds = {(32, None): 4449, (32, True): 4449, (32, False): 4449,
              (16, True): 3488, (16, False): 3488, (8, True): 2898,
              (8, False): 2898, (64, True): 5665, (64, False): 5665}
    waves = {(32, None): 1, (32, True): 1, (32, False): 1, (16, True): 1,
             (16, False): 1, (8, True): 1, (8, False): 1, (64, True): 2,
             (64, False): 2}
    spills = {(8, True): 668, (8, False): 668}
    fits = {key: 30 * 1024 for key in bounds}
    across = {**fits, (16, True): 42 * 1024, (16, False): 42 * 1024}
    monkeypatch.setattr(tuning, 'pipe_bound', _bounds(bounds))
    picks = {}
    for arch, code in (('gfx942', across), ('gfx90a', fits),
                       ('gfx90a', across)):
        ctx = _amd(arch)
        origin, first = _origin(descr_factory, ctx)
        picks[arch, code is fits] = _key(tuning.lanes_rule(
            descr_factory, ctx, TableScorer(spills, code=code, waves=waves),
            origin, first, generate=Generator.trial))[0]
    assert picks == {('gfx942', False): 16, ('gfx90a', True): 16,
                     ('gfx90a', False): 32}
    spilled = {**spills, (32, None): 600, (32, True): 600, (32, False): 600}
    ctx = _amd('gfx90a')
    origin, first = _origin(descr_factory, ctx)
    assert _key(tuning.lanes_rule(
        descr_factory, ctx, TableScorer(spilled, waves=waves), origin, first,
        generate=Generator.trial))[0] == 64


def test_an_amd_part_nobody_measured_keeps_its_default(monkeypatch):
    monkeypatch.setattr(tuning.CompiledScore, 'available',
                        lambda self, context: True)
    with pytest.warns(UserWarning, match='not measured'):
        assert tuning.autotune(_local_flux(), _amd('gfx1150', autotune='lanes'),
                               mode='lanes', generate=Generator.trial) is None


class PinScorer:
    """Reports the spilling `spills` and the code `code` name for a
    candidate's (lanes, pin) -- (lanes, pin, 'global') where it fences on
    global loads only, which reports as its fully fenced twin unless named --
    and counts the questions."""

    def __init__(self, spills, code=None):
        self.spills = spills
        self.code = code or {}
        self.asked = []

    def resources(self, result):
        cand = result.candidate
        key = (cand.lanes.num_threads, cand.get('pin_loads', 0))
        if cand.get('pin_shared') is False:
            key += ('global',)
        self.asked.append(key)
        return Resources(registers=255,
                         spill_bytes=self.spills.get(key, self.spills.get(
                             key[:2], 0)),
                         code_bytes=self.code.get(key))


def _default_ahead(result):
    """A count that keeps the default ahead of every other candidate, so far
    that the compiler is asked about the default alone: 32 lanes, unmerged
    and merged counting the same as the default's own deduction would
    otherwise be asked about too."""
    cand = result.candidate
    table = {8: 9000, 16: 8000, 32: 1000, 64: 7000}
    return (table[cand.lanes.num_threads]
            + (8000 if cand.get('merge_variants') is not None else 0))


def _pin_tuned(monkeypatch, spills, fixed=None, code=None, scale=1, **opts):
    """`autotune=lanes` with `PinScorer(spills, code)` for the compiler and
    the default ahead (`_default_ahead`, its counts `scale` times)."""
    scorer = PinScorer(spills, code)
    monkeypatch.setattr(tuning.CompiledScore, 'available',
                        lambda self, context: True)
    monkeypatch.setattr(tuning.CompiledScore, 'resources',
                        lambda self, result: scorer.resources(result))
    monkeypatch.setattr(tuning, 'executed_instructions',
                        lambda result: scale * _default_ahead(result))
    pick = tuning.autotune(_local_flux(), _context(autotune='lanes', **opts),
                           mode='lanes', fixed=fixed, generate=Generator.trial)
    return pick, scorer


def test_under_the_lanes_rule_ptxas_decides_the_pin(monkeypatch):
    """`pin_loads='auto'`: the default built unpinned, and pinned too where
    ptxas spills `PIN_SPILL` bytes or more -- viscoelastic order-6
    `derivative` in single precision on GB200, 7308 B stored and loaded,
    148 B pinned, and 71 % faster pinned."""
    pick, scorer = _pin_tuned(monkeypatch, {(32, 0): 7308, (32, 32): 148},
                              pin_loads='auto')
    assert pick.lanes.num_threads == 32 and pick.get('pin_loads') == 32
    assert pick.get('pin_shared') is None, (
        'fenced on every load where its twin costs no less')
    assert scorer.asked == [(32, 0), (32, 32), (32, 32, 'global')]


def test_the_global_twin_is_taken_where_it_costs_less(monkeypatch):
    """SeisSol's order-8 elastic `localFluxAll` in double on GH200: fenced
    on every load 147 kB of code and 424 B spilled, on global loads only 100
    kB and 660 B, and 28 % faster (`PIN_GLOBAL_TWIN`); counted at the
    23 456 instructions an element executes there."""
    pick, scorer = _pin_tuned(
        monkeypatch, {(32, 0): 2000, (32, 32): 424, (32, 32, 'global'): 660},
        code={(32, 32): 300 * 1024, (32, 32, 'global'): 100 * 1024},
        scale=23.456, pin_loads='auto')
    assert pick.get('pin_loads') == 32 and pick.get('pin_shared') is False


def test_the_full_pin_stays_where_the_global_twin_spills(monkeypatch):
    """The viscoelastic order-4 derivative in double: unfenced, the compiler
    hoists the shared loads into 2924 B of spill, fenced on every load it
    spills 192, and runs 0.6 times as long for it."""
    pick, scorer = _pin_tuned(
        monkeypatch, {(32, 0): 6000, (32, 32): 192, (32, 32, 'global'): 2924},
        code={(32, 32): 84 * 1024, (32, 32, 'global'): 55 * 1024},
        scale=8.264, pin_loads='auto')
    assert pick.get('pin_loads') == 32 and pick.get('pin_shared') is None


def test_no_global_twin_where_shared_loads_count_for_nothing(monkeypatch):
    pick, scorer = _pin_tuned(monkeypatch, {(32, 0): 7308, (32, 32): 148},
                              pin_loads='auto', pin_shared=False)
    assert scorer.asked == [(32, 0), (32, 32)]


def test_a_small_spill_is_left_unpinned(monkeypatch):
    """Elastic order-6 `derivative` in single precision on GB200: 148 B
    stored and loaded, 8 B pinned and 90 % slower for it.  With the
    threshold set for the stores alone the rule pinned it on sm_120, and
    the kernel ran 39 % behind the default.  Pinned is not even built."""
    pick, scorer = _pin_tuned(monkeypatch, {(32, 0): 148, (32, 32): 8},
                              pin_loads='auto')
    assert pick.get('pin_loads') == 0 and scorer.asked == [(32, 0)]


def test_a_pin_that_keeps_the_spilling_is_not_taken(monkeypatch):
    pick, scorer = _pin_tuned(monkeypatch, {(32, 0): 800, (32, 32): 600},
                              pin_loads='auto')
    assert pick.get('pin_loads') == 0
    assert scorer.asked[:2] == [(32, 0), (32, 32)]


def test_the_pin_is_the_compilers_only_under_auto_and_whole_warps(
        monkeypatch):
    """Off where 0 is asked (on NVIDIA 'auto' is the default,
    `Preferences.pin_loads`); and sixteen lanes share a warp with another
    multiplication, whose loop can end a trip earlier."""
    pick, scorer = _pin_tuned(monkeypatch, {(32, 0): 3840}, pin_loads=0)
    assert not pick.get('pin_loads')
    assert scorer.asked[0] == (32, 0)
    assert all(pin == 0 for _, pin in scorer.asked)
    sixteen = lane_config.LaneConfig(16, 40, 1)
    pick, scorer = _pin_tuned(monkeypatch, {(16, 0): 3840},
                              fixed={'lanes': sixteen}, pin_loads='auto')
    assert pick.get('pin_loads') == 0
    assert all(pin == 0 for _, pin in scorer.asked)


def test_the_generator_builds_the_pinned_pick(monkeypatch):
    """Through `Options.autotune`: fenced, where the estimate alone
    (`PTXAS_NEED`) leaves this small body as it is -- also after the same
    kernel was tuned without the pin, whose default build has the same
    source and whose pick is remembered."""
    scorer = PinScorer({(32, 0): 3840, (32, 32): 80})
    monkeypatch.setattr(tuning.CompiledScore, 'available',
                        lambda self, context: True)
    monkeypatch.setattr(tuning.CompiledScore, 'resources',
                        lambda self, result: scorer.resources(result))
    monkeypatch.setattr(tuning, 'executed_instructions', _default_ahead)
    unpinned = Generator(_local_flux()(), _context(autotune='lanes',
                                                   pin_loads=0))
    unpinned.generate()
    assert not unpinned.tuned.get('pin_loads')
    gen = Generator(_local_flux()(), _context(autotune='lanes',
                                              pin_loads='auto'))
    gen.generate()
    assert gen.tuned.get('pin_loads') == 32
    plain = Generator(_local_flux()(), _context(pin_loads='auto'))
    plain.generate()
    assert (gen.get_kernel().count('__syncwarp();')
            > plain.get_kernel().count('__syncwarp();'))


def _needs(table):
    """`_estimated_need` read from `table` by (lanes, merge)."""
    def need(result):
        return table[_key(result.candidate)]
    return need


def test_the_band_leaves_the_compiler_out_where_the_estimate_settles(
        monkeypatch):
    """Far below the register file a candidate spills nothing, far above it
    the registers past it, without the compiler (`ASK_BAND`): asked are only
    the candidates within the band the walk reaches -- the default at 240
    and 16 unmerged at 270.  Eight unmerged, the fewest instructions,
    estimated at 330, spills 600 B where the default spills 300 (the
    compiler's word): out; 16 unmerged, clean in the compiler's word, wins.
    No pin: what the compiler would pin is asked about pinned as well."""
    ctx = _context(pin_loads=0, lanes_band=True)
    descr_factory = _local_flux()
    origin, first = _origin(descr_factory, ctx)
    counts = {(32, None): 1000, (32, True): 1000, (32, False): 900,
              (16, True): 950, (16, False): 700, (8, True): 800,
              (8, False): 500, (64, True): 1500, (64, False): 1400}
    needs = {(32, None): 240, (32, True): 240, (32, False): 150,
             (16, True): 150, (16, False): 270, (8, True): 150,
             (8, False): 330, (64, True): 120, (64, False): 120}
    monkeypatch.setattr(tuning, 'executed_instructions', _counting(counts))
    monkeypatch.setattr(tuning, '_estimated_need', _needs(needs))
    scorer = TableScorer({(32, None): 300})
    pick = tuning.lanes_rule(descr_factory, ctx, scorer, origin, first,
                             generate=Generator.trial)
    assert _key(pick) == (16, False)
    assert scorer.asked == [(32, None), (16, False)]


def test_a_rolled_candidate_near_the_band_is_asked(monkeypatch):
    """Where the operations keep a loop -- a merged run, a rolled chain --
    the estimate errs both ways, and the compiler is asked from
    `ASK_LOOPS` below the file up."""
    ctx = _context()
    descr_factory = _local_flux()
    origin, first = _origin(descr_factory, ctx)
    monkeypatch.setattr(tuning, '_estimated_need', lambda result: 200)
    assert tuning._settled(first) is not None
    monkeypatch.setattr(tuning, '_rolled', lambda result: True)
    assert tuning._settled(first) is None


def test_a_candidate_the_band_settles_is_pinned_by_the_estimate(monkeypatch):
    """Under `pin_loads='auto'` the compiler decides the pin of what it was
    asked about (`PIN_SPILL`); a candidate the estimate settled comes out
    with the estimate's pin, `'auto'` (`PIN_MARGIN`)."""
    ctx = _context(pin_loads='auto', lanes_band=True)
    descr_factory = _local_flux()
    origin, first = _origin(descr_factory, ctx)
    counts = {(32, None): 1000, (32, True): 1000, (32, False): 900,
              (16, True): 950, (16, False): 700, (8, True): 800,
              (8, False): 500, (64, True): 1500, (64, False): 1400}
    monkeypatch.setattr(tuning, 'executed_instructions', _counting(counts))
    monkeypatch.setattr(tuning, '_estimated_need', lambda result: 120)
    scorer = TableScorer({})
    pick = tuning.lanes_rule(descr_factory, ctx, scorer, origin, first,
                             generate=Generator.trial)
    assert scorer.asked == []
    assert _key(pick) == (8, False)
    assert pick.get('pin_loads') == 'auto'


@pytest.mark.parametrize('pin', [0, 'auto'])
def test_the_walk_picks_alike_one_at_a_time_and_several_at_once(monkeypatch,
                                                                pin):
    """`Options.autotune_jobs`: the candidates built in forked processes and
    the compiler asked in threads ahead of the walk, which then takes from
    there what it asks -- the pick is the one the walk makes one at a time,
    pins included where the compiler decides them."""
    picks = []
    for jobs in (1, 4):
        tuning._PICKS.clear()
        ctx = _context(pin_loads=pin, autotune_jobs=jobs)
        descr_factory = _local_flux()
        origin, first = _origin(descr_factory, ctx)
        counts = {(32, None): 1000, (32, True): 1000, (32, False): 900,
                  (16, True): 950, (16, False): 700, (8, True): 800,
                  (8, False): 500, (64, True): 1500, (64, False): 1400}
        monkeypatch.setattr(tuning, 'executed_instructions',
                            _counting(counts))
        scorer = TableScorer({(8, False): 300, (32, None): 250})
        picks.append(tuning.lanes_rule(descr_factory, ctx, scorer, origin,
                                       first, generate=Generator.trial))
    assert picks[0] == picks[1]


@pytest.mark.parametrize('merge', [False, True])
def test_a_build_written_again_pinned_is_the_pinned_build(merge):
    """The fence is the emitter's alone, so a build written again with a pin
    (`Generator.emitted_with_pin`) is what a build made with that pin writes
    -- the same body and the same counts -- but for the options line and the
    kernel's name; and a merged build, which hands its kernel to another
    generator, is written again by that one."""
    import re
    descr_factory = _local_flux()

    def trial(pin):
        return Generator.trial(descr_factory(), _context(
            pin_loads=pin, merge_variants=merge, autotune='off'))

    def body(text):
        text = re.sub(r'kernel_\w+\(', 'kernel(', text)
        return [l for l in text.splitlines() if '// options:' not in l]
    unpinned, pinned = trial(0), trial(32)
    text, metrics = unpinned.emitted_with_pin(32)
    assert body(text) == body(pinned.get_kernel())
    assert text != unpinned.get_kernel()
    assert metrics.issue_mix == pinned.metrics.issue_mix
    assert metrics.code_units == pinned.metrics.code_units
    # and the build itself is left as it was
    assert unpinned.get_kernel() == trial(0).get_kernel()


@pytest.mark.parametrize('kib,charge', [
    (None, 0.0), (30, 0.0), (32, 0.0), (40, 1.0), (48, 2.0), (64, 4.0),
    (90, 4.0)])
def test_mi210_charges_the_fetch_on_a_ramp(kib, charge):
    """From nothing at CDNA 2's 32 KB to four times every instruction at
    twice that, and no further (`ICACHE_RAMP`)."""
    hw = _amd('gfx90a').target.hw
    code = None if kib is None else kib * 1024
    assert tuning.icache_charge(code, hw, 0.85) == pytest.approx(charge)


def test_elsewhere_the_charge_is_the_share_past_the_cache():
    """NVIDIA keeps the share of the code past the cache, scaled as
    calibrated; MI300A states no factor and is charged nothing."""
    from tensorforge.common.basic_types import Datatype
    hw = Context(arch='sm_90', backend='cuda',
                 fp_type=Datatype.F32).target.hw
    code = 2 * hw.icache_size
    assert tuning.icache_charge(code, hw, 0.94) == pytest.approx(0.47)
    hw = _amd('gfx942').target.hw
    miss = hw.icache_miss_factor
    refetch = tuning.ICACHE_SHARE * (miss - 1.0) if miss else 0.0
    assert tuning.icache_charge(4 * hw.icache_size, hw, refetch) == 0.0


def test_rolls_join_the_walk_where_the_code_is_past_the_cache():
    """The default rolled by 16 and written out whole, where its code is
    past the instruction cache by the emitter's estimate (`ROLL_UNROLLS`);
    nothing where it fits, or where the caller fixed the unroll."""
    from types import SimpleNamespace
    ctx = _context()
    hw = ctx.target.hw
    origin = tuning.Candidate()

    def first(units):
        return SimpleNamespace(ok=True, generator=SimpleNamespace(
            metrics=SimpleNamespace(code_units=units)))

    from tensorforge.analysis.icache import code_bytes
    big = 1
    while code_bytes(big, hw) <= hw.icache_size:
        big *= 2
    rolls = tuning._roll_candidates(first(big), ctx, origin, None)
    assert [c.get('k_unroll_max') for c in rolls] == [16, 0]
    assert tuning._roll_candidates(first(10), ctx, origin, None) == []
    assert tuning._roll_candidates(first(big), ctx, origin,
                                   {'k_unroll_max': 8}) == []
    asked = _context(k_unroll_max=32)
    assert tuning._roll_candidates(first(big), asked, origin, None) == []
