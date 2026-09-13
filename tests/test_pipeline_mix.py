# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""What a kernel's statements occupy, and the least time that takes.

The emitter sorts every statement it lays down by the pipe it occupies and
counts the bytes a lane moves through each memory space (`pir.emit`,
`Context.record_mix`, `Context.record_bytes`); `analysis.pipeline` turns that
into the least clocks per element of one SM or CU and says which pipe binds.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import warnings
from pathlib import Path

import pytest

from tensorforge.analysis import pipeline
from tensorforge.common.context import Context, Options
from tensorforge.generators.generator import Generator

CASES = Path(__file__).resolve().parent / 'cases'


def _built(name, arch, backend, **options):
    spec = importlib.util.spec_from_file_location(
        'tf_mix__' + name, next(CASES.rglob(f'{name}.py')))
    case = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(case)
    gen = Generator(case.descr_list(), Context(
        arch=arch, backend=backend, fp_type=case.DTYPE,
        options=Options(**options)))
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter('ignore')
        gen.generate()
    return gen


@pytest.mark.parametrize('arch,backend', [('sm_120', 'cuda'),
                                          ('gfx942', 'hip'),
                                          ('gfx1150', 'hip')])
def test_a_build_reports_its_mix_and_what_it_moves(arch, backend):
    gen = _built('local_flux', arch, backend)
    assert set(gen.issue_mix) <= set(pipeline.CATEGORIES)
    issued = {c: v[0] for c, v in gen.issue_mix.items()}
    # the contraction is the bulk of it, however it is spelled
    arithmetic = issued.get('fp', 0) + issued.get('matrix', 0)
    assert arithmetic > issued.get('int', 0)
    assert gen.memory_bytes.get('global.read', 0) > 0
    assert gen.memory_bytes.get('global.write', 0) > 0


def test_a_dpp_fma_is_arithmetic_and_a_matrix_builtin_is_a_matrix_op():
    """On RDNA the contraction is FMAs with a DPP operand, on CDNA MFMAs:
    counted as a move between lanes and as nothing, they left `fp` empty."""
    rdna = {c: v[0] for c, v in _built('local_flux', 'gfx1150', 'hip')
            .issue_mix.items()}
    cdna = {c: v[0] for c, v in _built('local_flux', 'gfx942', 'hip')
            .issue_mix.items()}
    assert rdna['fp'] > rdna.get('xlane', 0)
    assert cdna.get('matrix', 0) > 0
    assert 'other' not in rdna or rdna['other'] < rdna['fp'] / 100


def test_rolling_a_reduction_keeps_what_is_issued_and_shrinks_the_code():
    """Issued follows the trip counts, copies the code: rolling moves only
    the second -- and adds the loop's counter, test and branch."""
    unrolled = _built('local_flux', 'sm_120', 'cuda', merge_variants=False)
    rolled = _built('local_flux', 'sm_120', 'cuda', merge_variants=False,
                    k_roll=8)
    fp = lambda g, i: g.issue_mix['fp'][i]
    assert fp(rolled, 0) == fp(unrolled, 0)
    assert fp(rolled, 1) < fp(unrolled, 1)
    assert rolled.issue_mix['branch'][0] > unrolled.issue_mix['branch'][0]


def test_the_bound_names_its_pipe_and_is_the_largest_of_them():
    gen = _built('local_flux', 'sm_120', 'cuda')
    b = pipeline.of(gen)
    assert b.binding in pipeline.resources(
        gen._context.get_vm().get_hw_descr())
    assert b.cycles == max(b.per_resource.values())
    assert b.share(b.binding) == 1.0


def test_the_bound_is_per_element_and_falls_with_the_lanes_it_shares():
    """At fewer lanes a warp serves more elements with one instruction."""
    wide = _built('local_flux', 'sm_120', 'cuda', lanes_per_mult=32,
                  merge_variants=False)
    narrow = _built('local_flux', 'sm_120', 'cuda', lanes_per_mult=16,
                    merge_variants=False)
    hw = wide._context.get_vm().get_hw_descr()
    per = lambda g: pipeline.instructions_per_element(
        g.issue_mix, hw, g._num_threads)['fp']
    # the same FMAs in total, each warp instruction covering twice the rows
    assert per(narrow) == pytest.approx(per(wide), rel=0.2)


def test_consumer_fp64_is_a_sixteenth_and_data_center_fp64_is_half():
    from tensorforge.common.basic_types import Datatype

    def hw(arch):
        return Context(arch=arch, backend='cuda',
                       fp_type=Datatype.F64).get_vm().get_hw_descr()
    consumer = pipeline.resources(hw('sm_120'))
    center = pipeline.resources(hw('sm_90'))
    assert consumer['fp64'].rate == pytest.approx(consumer['fp32'].rate / 64)
    assert center['fp64'].rate == pytest.approx(center['fp32'].rate / 4)


def test_persistent_rounds_quantize_the_last_one():
    # 100 slots: 250 elements take three rounds, the last half full
    assert pipeline.persistent_efficiency(250, 10, 2, 5) == pytest.approx(
        250 / 300)
    assert pipeline.persistent_efficiency(300, 10, 2, 5) == 1.0
    assert pipeline.persistent_efficiency(0, 10, 2, 5) == 1.0


def test_dram_is_what_an_element_streams_and_binds_when_scarce():
    """The compulsory bytes, not the counted loads: the operators are read
    once for the batch, so they are not in it, and a bandwidth low enough
    makes DRAM the binding pipe."""
    gen = _built('local_flux', 'sm_120', 'cuda')
    stream = pipeline.stream_bytes(gen._given)
    from tensorforge.analysis.cost import list_cost
    assert 0 < stream <= list_cost(gen._given, batch=1).bytes
    scarce = pipeline.of(gen, dram_bytes_per_clock=1e-3)
    assert scarce.binding == 'dram'
    assert 'dram' not in pipeline.of(gen).per_resource


def test_a_measured_search_skips_what_cannot_win():
    """Once a candidate is measured, one whose least time is already longer
    is not run: a proof, not a guess."""
    from types import SimpleNamespace

    from tensorforge.generators.tuning import MeasuredScore

    gen = _built('local_flux', 'sm_120', 'cuda')
    ran = []

    def run(result):
        ran.append(result.candidate)
        return 1e-9 if result.candidate == 'fast' else 1.0

    score = MeasuredScore(run, sms=48, clock_ghz=2.5, batch=262144)
    fast = SimpleNamespace(ok=True, candidate='fast', generator=gen)
    slow = SimpleNamespace(ok=True, candidate='slow', generator=gen)
    assert score(fast) == 1e-9
    # the same kernel's least time is far above a nanosecond
    assert score(slow) == float('inf')
    assert ran == ['fast'] and score.pruned == ['slow']


def test_without_the_device_every_candidate_is_measured():
    from types import SimpleNamespace

    from tensorforge.generators.tuning import MeasuredScore

    gen = _built('local_flux', 'sm_120', 'cuda')
    score = MeasuredScore(lambda result: 1e-9)
    for name in ('a', 'b'):
        assert score(SimpleNamespace(ok=True, candidate=name,
                                     generator=gen)) == 1e-9
    assert score.pruned == []


def test_the_least_time_scales_with_the_batch_and_the_sms():
    b = pipeline.Bound(1000.0, 'fp32', {'fp32': 1000.0})
    one = pipeline.least_seconds(b, 1000, 10, 2.0)
    assert pipeline.least_seconds(b, 2000, 10, 2.0) == pytest.approx(2 * one)
    assert pipeline.least_seconds(b, 1000, 20, 2.0) == pytest.approx(one / 2)
    assert pipeline.least_seconds(b, 1000, 10, 2.0, 0.5) == pytest.approx(
        2 * one)
