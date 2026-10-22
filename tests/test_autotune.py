# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""`Options.autotune`: a generator that picks its own configuration.

Off by default, and switched on by one value.  Held here: that off changes
nothing, that on builds the pick and says so, that only safe knobs are turned,
and that a pick is remembered.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from tensorforge.common.context import Context, Options
from tensorforge.common.options import registry
from tensorforge.generators import tuning
from tensorforge.generators.generator import Generator

CASES = Path(__file__).resolve().parent / "cases"


def _case(name="local_flux.py"):
    path = CASES / name
    spec = importlib.util.spec_from_file_location("tf_autotune__" + path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _generate(arch="sm_100", **opts):
    mod = _case()
    gen = Generator(mod.descr_list(),
                    Context(arch=arch, backend="cuda", fp_type=mod.DTYPE,
                            options=Options(**opts)))
    gen.generate()
    return gen


@pytest.fixture(autouse=True)
def _forget_picks():
    tuning._PICKS.clear()
    tuning._LOADED.clear()
    yield
    tuning._PICKS.clear()
    tuning._LOADED.clear()


def test_autotune_is_off_by_default_and_off_changes_nothing():
    assert registry()['autotune'].default == 'off'
    assert not registry()['autotune'].codegen
    plain = _generate()
    off = _generate(autotune='off')
    assert plain.tuned is None and off.tuned is None
    assert plain.get_kernel() == off.get_kernel()


def test_static_autotune_builds_its_pick_and_says_so():
    gen = _generate(autotune='static', autotune_budget=6)
    assert gen.tuned is not None
    assert gen.lanes.num_threads == gen.tuned.lanes.num_threads
    assert gen.lanes.lead_width == gen.tuned.lanes.lead_width
    assert f'// tuned: {gen.tuned.label()}' in gen.get_kernel()


def test_the_budget_caps_the_builds(monkeypatch):
    built = []
    real = tuning.build

    def counting(factory, context, candidate, generate):
        built.append(candidate)
        return real(factory, context, candidate, generate)
    monkeypatch.setattr(tuning, 'build', counting)
    _generate(autotune='static', autotune_budget=4)
    # four for the walk, the default among them, and the cache key's build
    # is the default's -- so at most four distinct candidates
    assert len(set(built)) <= 4


def test_a_probe_the_generator_builds_for_itself_is_not_tuned(monkeypatch):
    """Tuning is one question about the caller's kernel, asked once.

    The probes of the merge decision are generators this one builds to ask
    something about itself, at a geometry that is already settled, and so is
    the pick it builds.  Tuning them would open a walk inside every probe and
    a probe inside every build of every walk -- hundreds of builds that all
    keep to `autotune_budget` in *distinct* candidates, so the budget test
    above would not see them.  Counted here, in total.
    """
    built = []
    real = tuning.build

    def counting(factory, context, candidate, generate):
        built.append(candidate)
        return real(factory, context, candidate, generate)
    monkeypatch.setattr(tuning, 'build', counting)
    # sm_120: no measured preference, so the walk actually runs
    _generate("sm_120", autotune='static', autotune_budget=4)
    assert len(built) <= 5, f'{len(built)} builds against a budget of four'


def test_a_remembered_pick_costs_one_build(monkeypatch, tmp_path):
    # sm_120: no measured preference for it, so the walk runs and is remembered
    path = tmp_path / "picks.json"
    first = _generate("sm_120", autotune='static', autotune_budget=4,
                      autotune_cache=str(path))
    assert json.loads(path.read_text())
    tuning._PICKS.clear()
    tuning._LOADED.clear()
    built = []
    real = tuning.build

    def counting(factory, context, candidate, generate):
        built.append(candidate)
        return real(factory, context, candidate, generate)
    monkeypatch.setattr(tuning, 'build', counting)
    again = _generate("sm_120", autotune='static', autotune_budget=4,
                      autotune_cache=str(path))
    assert len(built) == 1, 'the default build for the key, and nothing else'
    assert again.tuned == first.tuned
    assert again.get_kernel() == first.get_kernel()


def test_an_explicit_geometry_is_kept_and_the_rest_still_turned(monkeypatch):
    """A caller who states the geometry states that, not "do not tune".

    Standing aside for an explicit `lanes` would turn tuning off for every
    caller that passes one, and `tools/bench/build.py` passes one for every
    build -- so nothing measured through it would be tuned.  What is fixed is
    pinned into the origin and leaves the space; the other knobs are still
    walked.

    The walk itself is another test's business: this one watches what the
    generator asks for, because asking is the decision.
    """
    mod = _case()
    base = tuning.start(mod.descr_list(),
                        Context(arch="sm_100", backend="cuda", fp_type=mod.DTYPE))
    asked = {}

    def watching(factory, context, mode='static', budget=None, cache=None,
                 fixed=None, *, generate):
        asked['fixed'] = fixed
        return None

    monkeypatch.setattr(tuning, 'autotune', watching)
    gen = Generator(mod.descr_list(),
                    Context(arch="sm_100", backend="cuda", fp_type=mod.DTYPE,
                            options=Options(autotune='static')),
                    lanes=base.lanes)
    gen.generate()
    assert asked, 'the geometry switched tuning off'
    assert asked['fixed'] == {'lanes': base.lanes}


def test_a_pinned_knob_leaves_the_space_and_stays_in_the_origin():
    """`fixed` is applied to the origin and its knob is not walked."""
    mod = _case()
    ctx = Context(arch="sm_100", backend="cuda", fp_type=mod.DTYPE)
    descrs = mod.descr_list()
    origin = tuning.start(descrs, ctx)
    knobs = tuning.simple_space(descrs, ctx)
    assert any(k.name == 'lanes' for k in knobs), 'nothing to pin in this case'
    pinned = origin.set('lanes', origin.lanes)
    assert pinned.lanes == origin.lanes
    rest = [k for k in knobs if k.name not in {'lanes'}]
    assert {k.name for k in rest} == {k.name for k in knobs} - {'lanes'}


def _simple(arch, backend="cuda"):
    mod = _case()
    ctx = Context(arch=arch, backend=backend, fp_type=mod.DTYPE)
    descrs = mod.descr_list()
    origin = tuning.start(descrs, ctx)
    return {k.name: list(k.values(origin)) for k in tuning.simple_space(descrs, ctx)}


def test_the_lane_space_stops_at_the_vector_unit_under_spmd():
    """A multiplication narrower than the vector unit idles part of it.

    Under SPMD there is one work-item per lane and nothing to put in the rest
    of the vector, where a wave holding several multiplications fills itself.
    Measured on a 16-wide PVC: with the geometry free, both scorers would take
    `elastic-o6s:derivative` to eight lanes, at 66.1 ns an element against
    29.9 at the deduced 32 and 12.7 at sixteen.
    """
    intel = _simple("pvc", "oneapi")
    assert min(c.num_threads for c in intel['lanes']) >= 16
    nvidia = _simple("sm_100")
    assert min(c.num_threads for c in nvidia['lanes']) < 16, (
        'a narrow multiplication shares its wave there, so the floor is not '
        'the wave width')


def test_the_simple_space_turns_only_what_is_safe_to_ship():
    knobs = _simple("sm_100")
    assert set(knobs) <= {'lanes', 'merge_variants', 'k_roll', 'k_unroll_max',
                          'k_width'}
    widths = {c.num_threads for c in knobs['lanes']}
    assert all(w & (w - 1) == 0 for w in widths), 'powers of two only'
    # 2 beside the divisors: the shortest body, which is what a kernel at
    # the register limit wants and no divisor of 55 offers.
    assert knobs['k_roll'][:2] == [0, 28] and 2 in knobs['k_roll']


def test_the_reduction_width_offers_only_the_other_side():
    """1 is the default and the origin the walk starts from, and `coordinate`
    sets a knob's value as it is given -- so offering it back would spend a
    build to arrive where it started."""
    assert _simple("sm_100")['k_width'] == [2]


def test_a_reduction_too_short_to_have_a_whole_group_is_not_offered_one():
    import tensorforge.generators.tuning as t
    real = t.contraction_lengths
    try:
        t.contraction_lengths = lambda d: [3]
        assert 'k_width' not in _simple("sm_100")
        t.contraction_lengths = lambda d: [4]
        assert 'k_width' in _simple("sm_100")
    finally:
        t.contraction_lengths = real


@pytest.mark.parametrize("arch,backend,paired", [
    ("sm_100", "cuda", True), ("sm_120", "cuda", False), ("sm_86", "cuda", False),
    ("gfx942", "hip", True), ("gfx1250", "hip", True), ("gfx1150", "hip", False)])
def test_width_two_only_where_one_instruction_does_two_fmas(arch, backend, paired):
    knobs = _simple(arch, backend)
    assert any(c.lead_width == 2 for c in knobs['lanes']) == paired


class _FakeGen:
    def __init__(self, threads, mults, peak, blocks=16):
        from tensorforge.generators.lanes import LaneConfig
        #: What the build settled on: the lanes a multiplication spans, and
        #: on Intel how many share one thread's register file.
        self.lanes = LaneConfig(threads, threads, 1)
        from tensorforge.common.metrics import BuildMetrics
        self.metrics = BuildMetrics()
        self.metrics.peak_pressure = peak
        self.metrics.emitted_work = 1000
        self.resident_blocks = blocks

        class _Launch:
            mults_per_block = mults
        self._launch = _Launch()

    def launch_config(self):
        return self._launch


def _fake(arch, backend, threads, mults, peak):
    ctx = Context(arch=arch, backend=backend, fp_type=_case().DTYPE)
    return tuning.Build(tuning.Candidate(), generator=_FakeGen(threads, mults, peak),
                        context=ctx)


def test_amd_is_guarded_by_what_hipcc_allocates_and_nvidia_by_bytes():
    """gfx942 spilled eight lanes at b = 56 (model 2449 B, under its 2048 B
    budget only as bytes); sm_100 did not spill them at 816 B."""
    assert tuning._over_budget(_fake('gfx942', 'hip', 8, 32, 2449))
    assert not tuning._over_budget(_fake('gfx942', 'hip', 32, 4, 1403))
    assert not tuning._over_budget(_fake('sm_100', 'cuda', 8, 16, 816))
    assert tuning._over_budget(_fake('sm_100', 'cuda', 8, 16, 1112))


def test_intel_spmd_shares_one_register_file_over_the_sub_group():
    """The file is a thread's, and under SPMD a thread holds the sub-group.

    Compared against the whole file, as every other target is, the guard
    would never fire on Intel: `elastic-o6s:derivative` models 1716 B a lane
    against 8192 and IGC spills it hard -- 1108 scratch messages in the ISA
    at 32 lanes against 9 at sixteen.  Per thread it is 6.7 times the file
    against 3.9, which is the order the machine agrees with.
    """
    wide = _fake('pvc', 'oneapi', 32, 8, 1716)
    narrow = _fake('pvc', 'oneapi', 16, 8, 1972)
    assert tuning._over_budget(wide) > tuning._over_budget(narrow) > 0
    # Under an explicit vector the work-item *is* the multiplication, so the
    # file is not shared and the figure is the footprint itself.
    esimd = _fake('pvc', 'esimd', 32, 8, 9000)
    assert tuning._over_budget(esimd) == 9000 - 8192


def test_amd_ranks_by_the_waves_its_registers_leave():
    """One wave per SIMD above 256 registers: at 1403 B (about 440 registers)
    a block of four waves fits once per CU; at 400 B (about 126, so 128)
    four waves per SIMD, sixteen per CU, four blocks."""
    heavy = tuning._register_blocks(_fake('gfx942', 'hip', 32, 8, 1403))
    light = tuning._register_blocks(_fake('gfx942', 'hip', 32, 8, 400))
    assert heavy == 1 and light == 4
    assert tuning._register_blocks(_fake('sm_100', 'cuda', 32, 4, 400)) is None


def test_amd_takes_width_two_only_with_one_pair_per_lane():
    knobs = _simple("gfx942", "hip")
    pairs = {c.num_threads for c in knobs['lanes'] if c.lead_width == 2}
    # 56 rows: a pair per lane needs 28 lanes or more
    assert pairs and all(2 * t >= 56 for t in pairs)
    nvidia = {c.num_threads for c in _simple("sm_100")['lanes'] if c.lead_width == 2}
    assert 16 in nvidia, 'ptxas takes two pairs per lane without spilling'


@pytest.mark.parametrize("arch,backend,stated", [
    ("gfx900", "hip", 102 * 4), ("gfx942", "hip", 102 * 4),
    ("gfx1150", "hip", 106 * 4), ("gfx1250", "hip", 106 * 4),
    ("sm_120", "cuda", None), ("pvc", "oneapi", None)])
def test_a_scalar_register_file_is_stated_where_there_is_one(arch, backend,
                                                             stated):
    """AMD keeps what a whole wave agrees on in SGPRs, and a wave has about a
    hundred.  NVIDIA's uniform datapath is not the same offer: it carries
    integer and address arithmetic, so a uniform float is a vector register
    there and no second budget applies."""
    hw = Context(arch=arch, backend=backend,
                 fp_type=_case().DTYPE).target.hw
    assert hw.max_scalar_reg_per_wave == stated


def test_the_scalar_file_is_judged_on_its_own():
    """`local_flux` never fills it; SeisSol's damage step does -- 107 SGPRs
    allocated and 117 spilled on gfx1150, beside full VGPRs.  One figure
    against `max_reg_per_thread` shows neither, so the uniform values are
    weighed against their own budget."""
    fits = _fake('gfx1150', 'hip', 32, 4, 400)
    fits.generator.metrics.peak_uniform_pressure = 100 * 4
    assert tuning._over_scalar_budget(fits) == 0

    over = _fake('gfx1150', 'hip', 32, 4, 400)
    over.generator.metrics.peak_uniform_pressure = 150 * 4
    assert tuning._over_scalar_budget(over) == (150 - 106) * 4

    silent = _fake('sm_120', 'cuda', 32, 4, 400)
    silent.generator.metrics.peak_uniform_pressure = 150 * 4
    assert tuning._over_scalar_budget(silent) == 0, 'no file, no judgement'


def _narrow_product(columns=9, dest=None):
    """C[56 x 9] = A[56 x 56] B[56 x 9]: an operator the batch shares against
    an element's own -- the product a single element leaves CDNA's matrix
    instruction half empty in."""
    from tensorforge.common.basic_types import Addressing, Datatype
    from tensorforge.common.matrix.boundingbox import BoundingBox
    from tensorforge.common.matrix.tensor import SubTensor, Tensor
    from tensorforge.generators.descriptions import GemmDescr

    def view(shape, addressing, alias):
        return SubTensor(Tensor(shape, addressing, BoundingBox([0, 0], shape),
                                alias=alias, datatype=Datatype.F32))
    return [GemmDescr(trans_a=False, trans_b=False,
                      a=view([56, 56], Addressing.NONE, 'A'),
                      b=view([56, columns], Addressing.STRIDED, 'B'),
                      c=view([56, columns], dest or Addressing.STRIDED, 'C'),
                      alpha=1.0, beta=0.0)]


def test_amd_turns_the_stack_and_the_width_under_the_compiled_scorer():
    """Where one element leaves the matrix instruction half empty
    (`Preferences.tune_stack`), the stack is a knob, and so is the tile
    width where the target has several (`MatrixPaths.widths`); each only on
    its other side, the origin being the default.  Under the compiled scorer
    alone, which sees what they cost in registers; NVIDIA has neither."""
    from tensorforge.common.basic_types import Datatype
    ctx = Context(arch='gfx942', backend='hip', fp_type=Datatype.F32)
    descrs = _narrow_product()
    origin = tuning.start(descrs, ctx)
    knobs = {k.name: list(k.values(origin))
             for k in tuning.simple_space(descrs, ctx, mode='compiled')}
    assert knobs['stack_elements'] == [2]
    assert knobs['matrix_columns'] == [4, 16]
    static = {k.name for k in tuning.simple_space(descrs, ctx, mode='static')}
    assert not {'stack_elements', 'matrix_columns'} & static
    assert not {'stack_elements', 'matrix_columns'} & set(_simple('sm_100'))
    full = {k.name: list(k.values(origin))
            for k in tuning.space(descrs, ctx)}
    assert full['stack_elements'] == [1, 2]
    assert full['matrix_columns'] == [0, 4, 16]


def test_a_compiled_walk_without_a_compiler_turns_what_a_static_one_does(
        monkeypatch):
    """`compiled` falls back on the static scorer where no compiler is found,
    and the space follows the scorer that ranks it, not the one asked for."""
    from tensorforge.common.basic_types import Datatype
    seen = {}

    def watching(descr_factory, context, scorer, knobs=None, **rest):
        seen['knobs'] = {k.name for k in knobs}
        return tuning.Outcome(rest['origin'], None, [])

    monkeypatch.setattr(tuning, 'coordinate', watching)
    monkeypatch.setattr(tuning.CompiledScore, 'available',
                        lambda self, context: False)
    ctx = Context(arch='gfx942', backend='hip', fp_type=Datatype.F32,
                  options=Options(autotune='compiled'))
    with pytest.warns(UserWarning, match='ranking statically'):
        tuning.autotune(_narrow_product, ctx, mode='compiled',
                        generate=Generator.trial)
    assert seen['knobs'] and 'stack_elements' not in seen['knobs']


def test_a_list_that_does_not_stack_is_not_offered_the_stack():
    """What `stacking.stack` refuses by name -- here a destination the batch
    shares -- is not a candidate at all."""
    from tensorforge.common.basic_types import Addressing, Datatype
    ctx = Context(arch='gfx942', backend='hip', fp_type=Datatype.F32)
    descrs = _narrow_product(dest=Addressing.NONE)
    assert 'stack_elements' not in {
        k.name for k in tuning.simple_space(descrs, ctx, mode='compiled')}


def test_the_walk_takes_the_stack_where_it_is_worth_it_per_element():
    """The bound is per element (`pipeline.of`), so the stack is weighed in
    the unit the plain build is: on the narrow product the static model puts
    it at 256 clocks an element against 457 for the plain build, and a walk
    over the knob takes it.  The static scorer stands in for the compiled
    one, which needs a compiler: what is held is the unit."""
    from tensorforge.common.basic_types import Datatype
    ctx = Context(arch='gfx942', backend='hip', fp_type=Datatype.F32)
    knobs = [k for k in tuning.simple_space(_narrow_product(), ctx,
                                            mode='compiled')
             if k.name == 'stack_elements']
    out = tuning.coordinate(_narrow_product, ctx, tuning.static_score,
                            knobs=knobs, generate=Generator.trial)
    assert out.best.get('stack_elements') == 2


def test_residency_is_counted_in_elements():
    """A stacked multiplication holds two elements, so the residency the
    scorers rank by after the bound counts them (`static_score`)."""
    from tensorforge.common.basic_types import Datatype
    ctx = Context(arch='gfx942', backend='hip', fp_type=Datatype.F32)
    for stack in (1, 2):
        result = tuning.build(_narrow_product, ctx,
                              tuning.Candidate().set('stack_elements', stack),
                              Generator.trial)
        _, _, resident = tuning._geometry(result)
        blocks = tuning._register_blocks(result)
        if blocks is not None:
            mults = result.generator.launch_config().mults_per_block
            resident = min(resident, blocks * mults)
        assert resident > 0
        assert tuning.static_score(result)[2] == -resident * stack


@pytest.mark.parametrize('asked', [{'stack_elements': 1},
                                   {'matrix_columns': 4},
                                   {'stack_elements': 2, 'matrix_columns': 16}])
def test_an_asked_stack_or_width_is_pinned(monkeypatch, asked):
    """A stack or a width the caller asks for is a decision, as an explicit
    geometry is: pinned in the origin and out of the space."""
    from tensorforge.common.basic_types import Datatype
    from tensorforge.generators import stacking
    seen = {}

    def watching(factory, context, mode='static', budget=None, cache=None,
                 fixed=None, *, generate):
        seen['fixed'] = fixed
        seen['stacked'] = [stacking.factor(v.tensor) for d in factory()
                           for op in d.operations() for v in op.reads()]
        return None

    monkeypatch.setattr(tuning, 'autotune', watching)
    gen = Generator(_narrow_product(),
                    Context(arch='gfx942', backend='hip', fp_type=Datatype.F32,
                            options=Options(autotune='static', **asked)))
    gen.generate()
    assert seen['fixed'] == asked
    # and the candidates come from the list unstacked, whatever this
    # generator's own stack: a candidate states its own
    assert set(seen['stacked']) == {1}


def test_a_knob_with_nothing_to_offer_leaves_the_point_as_it_is():
    """`enumerate_space` over a knob without values at a point: the point,
    not nothing -- the width of the matrix instruction has no values at a
    lane count no wide tile fits."""
    origin = tuning.Candidate()
    knobs = [tuning.Knob('k_roll', lambda c: (0, 2)),
             tuning.Knob('matrix_columns', lambda c: ())]
    points = list(tuning.enumerate_space(knobs, origin))
    assert [p.get('k_roll') for p in points] == [0, 2]
    assert all(p.get('matrix_columns') is None for p in points)
