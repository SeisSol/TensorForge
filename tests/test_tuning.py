# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""The search space over configurations, and what walks it (`generators.tuning`).

The strategies are checked against a scorer that knows the answer, with the
builds stubbed out: what is being held is the walk, and a real build costs a
second each.  The space and the work count are checked on real builds.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from tensorforge.common.basic_types import Datatype
from tensorforge.common.context import Context, Options
from tensorforge.generators import tuning
from tensorforge.generators.generator import Generator
from tensorforge.generators.lanes import LaneConfig

CASES = Path(__file__).resolve().parent / "cases"


def _case(name):
    path = CASES / name
    spec = importlib.util.spec_from_file_location("tf_tuning__" + path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _knobs(case, arch, backend):
    mod = _case(case)
    ctx = Context(arch=arch, backend=backend, fp_type=mod.DTYPE)
    descrs = mod.descr_list()
    origin = tuning.start(descrs, ctx)
    return {k.name: list(k.values(origin)) for k in tuning.space(descrs, ctx)}, origin


# ---------------------------------------------------------------------- #
# candidates
# ---------------------------------------------------------------------- #

def test_a_candidate_round_trips_through_plain_data():
    cand = tuning.Candidate(LaneConfig(8, 56, 1)).set('k_roll', 28).set(
        'merge_variants', True)
    assert tuning.Candidate.from_dict(cand.to_dict()) == cand
    assert cand.get('k_roll') == 28 and cand.get('lanes').num_threads == 8
    assert cand.label() == '8w1 k_roll=28,merge_variants=1'


def test_a_candidate_context_keeps_the_target_and_adds_its_options():
    base = Context(arch='sm_100', backend='cuda', fp_type=Datatype.F32,
                   options=Options(prepare_operands=True))
    ctx = tuning.Candidate(options=(('k_roll', 4),)).context(base)
    hw = ctx.get_vm().get_hw_descr()
    assert (hw.model, hw.backend) == ('sm_100', 'cuda')
    opts = ctx.get_user_options()
    assert opts.prepare_operands is True and opts.k_roll == 4


# ---------------------------------------------------------------------- #
# the space
# ---------------------------------------------------------------------- #

def test_local_flux_on_nvidia_offers_every_knob_that_can_matter():
    knobs, origin = _knobs("local_flux.py", "sm_100", "cuda")
    assert {'lanes', 'merge_variants', 'prepare_operands', 'preload_globals',
            'k_roll', 'tensor_cores', 'mma_prefetch'} <= set(knobs)
    # the reductions are 56 and 9 long: rolled by 28 and by 8, or not at all
    assert knobs['k_roll'] == [0, 28, 8]
    geometries = {(c.num_threads, c.lead_width) for c in knobs['lanes']}
    assert {(32, 1), (8, 1), (16, 2)} <= geometries
    # the matrix path's prefetch depth is a question only where it is taken
    assert knobs['mma_prefetch'] == [None]
    assert list(dict((k.name, k) for k in tuning.space(
        _case("local_flux.py").descr_list(),
        Context(arch='sm_100', backend='cuda', fp_type=Datatype.F32)))
        ['mma_prefetch'].values(origin.set('tensor_cores', True))) == [1, 2]


def test_no_matrix_path_off_nvidia():
    knobs, _ = _knobs("local_flux.py", "gfx942", "hip")
    assert 'tensor_cores' not in knobs and 'mma_prefetch' not in knobs


def test_no_width_two_where_the_backend_cannot_spell_it():
    knobs, _ = _knobs("local_flux.py", "sm_86", "cuda")
    assert any(c.lead_width == 2 for c in knobs['lanes'])
    knobs, _ = _knobs("local_flux.py", "pvc", "acpp")
    assert all(c.lead_width == 1 for c in knobs['lanes'])


# ---------------------------------------------------------------------- #
# strategies, against a scorer that knows the answer
# ---------------------------------------------------------------------- #

def _stub_builds(monkeypatch, fail=()):
    built = []

    def build(factory, context, candidate):
        built.append(candidate)
        if candidate.get('k_roll') in fail:
            return tuning.Build(candidate, context=context, error=ValueError('no'))
        return tuning.Build(candidate, generator=object(), context=context)
    monkeypatch.setattr(tuning, 'build', build)
    return built


def _separable(result):
    """Best at 8 lanes, merged, rolled by 8 -- each on its own."""
    c = result.candidate
    return ((c.lanes.num_threads != 8) + (not c.get('merge_variants'))
            + (c.get('k_roll') != 8))


def test_coordinate_descent_finds_a_separable_optimum(monkeypatch):
    mod = _case("local_flux.py")
    ctx = Context(arch='sm_100', backend='cuda', fp_type=mod.DTYPE)
    built = _stub_builds(monkeypatch)
    out = tuning.tune(mod.descr_list, ctx, scorer=_separable)
    assert out.score == 0
    assert out.best.lanes.num_threads == 8 and out.best.get('k_roll') == 8
    assert out.best.get('merge_variants') is True
    everything = sum(1 for _ in tuning.enumerate_space(
        tuning.space(mod.descr_list(), ctx), tuning.start(mod.descr_list(), ctx)))
    assert len(set(built)) < everything / 4, (
        'a coordinate walk should cost a fraction of the whole space')


def test_a_candidate_that_does_not_build_is_passed_over(monkeypatch):
    mod = _case("local_flux.py")
    ctx = Context(arch='sm_100', backend='cuda', fp_type=mod.DTYPE)
    _stub_builds(monkeypatch, fail=(8,))
    out = tuning.tune(mod.descr_list, ctx, scorer=_separable)
    assert out.best.get('k_roll') != 8
    assert any(t.error is not None for t in out.trials)


def test_exhaustive_agrees_with_coordinate_on_a_separable_score(monkeypatch):
    mod = _case("local_flux.py")
    ctx = Context(arch='sm_100', backend='cuda', fp_type=mod.DTYPE)
    _stub_builds(monkeypatch)
    knobs = [k for k in tuning.space(mod.descr_list(), ctx)
             if k.name in ('lanes', 'merge_variants', 'k_roll')]
    a = tuning.tune(mod.descr_list, ctx, scorer=_separable,
                    strategy=tuning.exhaustive, knobs=knobs)
    b = tuning.tune(mod.descr_list, ctx, scorer=_separable, knobs=knobs)
    assert a.score == b.score == 0


# ---------------------------------------------------------------------- #
# compilers
# ---------------------------------------------------------------------- #

def test_ptxas_output_is_read():
    log = ("ptxas info    : Used 168 registers, used 1 barriers, 384 bytes cmem[0]\n"
           "    328 bytes stack frame, 328 bytes spill stores, 196 bytes spill loads\n")
    r = tuning.parse_ptxas(log)
    assert (r.registers, r.spill_bytes) == (168, 524)


def test_amdgpu_resource_remarks_are_read():
    log = ("remark:     VGPRs: 110 [-Rpass-analysis=kernel-resource-usage]\n"
           "remark:     AGPRs: 16 [-Rpass-analysis=kernel-resource-usage]\n"
           "remark:     ScratchSize [bytes/lane]: 124 [-Rpass-analysis=kernel-resource-usage]\n"
           "remark:     Occupancy [waves/SIMD]: 4 [-Rpass-analysis=kernel-resource-usage]\n")
    r = tuning.parse_amdgpu(log)
    assert (r.registers, r.spill_bytes, r.register_blocks) == (126, 124, 4)


def test_the_compiler_is_the_callers_then_the_environments_then_the_paths(monkeypatch):
    monkeypatch.setenv('TF_NVCC', '/from/env/nvcc')
    monkeypatch.setattr(tuning.shutil, 'which', lambda name: '/on/path/' + name)
    assert tuning.Toolchain(nvcc='/given/nvcc').compiler('nvidia') == '/given/nvcc'
    assert tuning.Toolchain().compiler('nvidia') == '/from/env/nvcc'
    monkeypatch.delenv('TF_NVCC')
    assert tuning.Toolchain().compiler('nvidia') == '/on/path/nvcc'
    assert tuning.Toolchain().compiler('intel') == '/on/path/icpx'
    assert tuning.Toolchain(icpx='/given/icpx').compiler('intel') == '/given/icpx'
    assert tuning.Toolchain().compiler('other') is None


# ---------------------------------------------------------------------- #
# the work count a ranking reads
# ---------------------------------------------------------------------- #

def test_rolling_a_reduction_does_not_change_its_work():
    """A rolled loop is written once and runs its trip count; counted once, a
    roll by 28 made `local_flux` look like a 28th of its arithmetic."""
    mod = _case("local_flux.py")

    def work(**opts):
        ctx = Context(arch='sm_100', backend='cuda', fp_type=mod.DTYPE,
                      options=Options(**opts))
        gen = Generator(mod.descr_list(), ctx)
        gen.generate()
        return gen.emitted_work

    assert work(k_roll=28) == work()
