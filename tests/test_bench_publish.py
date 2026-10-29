# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""The figures and tables for a paper (`publish.py`, `texout.py`).

What is checked is what a reader of the paper would otherwise take on
trust: the fit a launch cost is read off, a time step put together the way
SeisSol launches it, a performance portability that tells a failure from a
gap in the campaign, a kernel drawn under its own equation only -- and that
what comes out is LaTeX a document can `\\input`, compiled where a TeX
installation is there to compile it.
"""

from __future__ import annotations

import csv
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / 'tools' / 'bench'
if str(BENCH) not in sys.path:
    sys.path.insert(0, str(BENCH))

import compare  # noqa: E402
import publish  # noqa: E402
import texout  # noqa: E402


@pytest.fixture(autouse=True)
def _issued_flops():
    """The flop basis is module state; every test here reads the default."""
    publish.set_flop_basis('executed')
    yield
    publish.set_flop_basis('executed')


# ----------------------------------------------------------------------
# A campaign tree
# ----------------------------------------------------------------------

#: A machine's roof: 10 TFLOP/s and 1 TB/s, the ridge at 10 flop/byte.
CEILING = {'flops_per_s': 1e13, 'bytes_per_s': 1e12,
           'peak_bytes_per_s': 1.2e12}


def _point(kernel, arch, order, efficiency=None, *, equation='elastic',
           batch=65536, intensity=1.0, error=None, precision='d',
           simulations=1, config='baseline', **extra):
    """A row at `efficiency` of the roof (`CEILING`) at `intensity`, or a
    row that produced no number.  Anything in `extra` goes into the row."""
    tag = f'{equation}-{order}-{precision}-s{simulations}'
    row = {'workload': kernel, 'config': config, 'arch': arch,
           'backend': 'cuda', 'device': f'device {arch}',
           'datatype': {'d': 'F64', 's': 'F32'}[precision],
           'capture': tag, 'equation': equation, 'order': order,
           'precision': precision, 'simulations': simulations}
    if error:
        return dict(row, error=error)
    flops = 1e9
    byts = flops / intensity
    roof = min(CEILING['flops_per_s'], CEILING['bytes_per_s'] * intensity)
    ns = flops / (efficiency * roof) * 1e9
    return dict(row, batch=batch, flops=flops, bytes=byts, event_ns=ns,
                wall_ns=ns + 2000.0, **extra)


def _tree(tmp_path, rows, manifest=(), datatypes=('F64',)):
    """One run directory per machine, each with its ceilings beside it."""
    for arch in sorted({r['arch'] for r in rows}):
        d = tmp_path / arch / 'run'
        d.mkdir(parents=True)
        (d / 'bench.json').write_text(json.dumps(
            {'run': {'suite': 'test'},
             'manifest': [m for m in manifest if m['arch'] == arch],
             'results': [r for r in rows if r['arch'] == arch]}))
        for dt in datatypes:
            c = d / 'ceiling' / dt
            c.mkdir(parents=True)
            (c / 'roofline.json').write_text(json.dumps({'ceiling': dict(
                CEILING, arch=arch, backend='cuda', device=f'device {arch}',
                datatype=dt,
                flops_per_s=CEILING['flops_per_s'] * (2 if dt == 'F32'
                                                      else 1))}))
    return compare.collect([tmp_path])


# ----------------------------------------------------------------------
# The cost of a launch
# ----------------------------------------------------------------------

def test_the_fit_recovers_a_launch_cost_and_its_half_rate_batch():
    """`t(n) = 5 us + 2 ns n`: half the asymptotic rate at 2 500 elements,
    and the first measured batch with half the best rate is 4 096."""
    fit = publish.launch_fit([(n, 5000.0 + 2.0 * n)
                              for n in (16, 64, 256, 1024, 4096, 16384)])
    assert fit.t0_ns == pytest.approx(5000.0)
    assert fit.c_ns == pytest.approx(2.0)
    assert fit.half == pytest.approx(2500.0)
    assert fit.residual == pytest.approx(0.0, abs=1e-12)
    assert fit.measured == 4096


def test_no_fit_is_made_up_from_too_little_or_from_no_slope():
    assert publish.launch_fit([(16, 10.0), (32, 12.0)]) is None
    assert publish.launch_fit([(16, 10.0), (32, 10.0), (64, 10.0)]) is None


def test_a_small_launch_counts_as_much_as_a_large_one():
    """Three percent either way on every launch, as repeated runs scatter,
    moves the fixed cost by under one percent.  Unweighted, the fit would
    take it from the largest launches, whose three percent is three times
    the whole fixed cost, and be a fifth off."""
    batches = (16, 64, 256, 1024, 4096, 16384, 65536, 262144)
    fit = publish.launch_fit([(n, (5000.0 + 2.0 * n) * (1 + 0.03 * (-1) ** i))
                              for i, n in enumerate(batches)])
    assert fit.t0_ns == pytest.approx(5000.0, rel=0.01)
    assert fit.c_ns == pytest.approx(2.0, rel=0.02)
    assert fit.residual == pytest.approx(0.03, rel=0.25)


# ----------------------------------------------------------------------
# The time step
# ----------------------------------------------------------------------

def _timed(ns, flops):
    return {'wall_ns': ns, 'event_ns': ns, 'flops': flops,
            'executed_flops': flops}


def test_a_time_step_is_launched_the_way_seissol_launches_it():
    """The neighbour flux: sixteen launches of a quarter of the elements,
    four at a time.  Taken together, four launches of all of them; one
    after another, sixteen of a quarter.  The arithmetic is the same
    either way, the time is not."""
    step = {'derivative': publish.Launch(),
            'flux': publish.Launch(16, 0.25, 4)}
    at = {'derivative': {n: _timed(100.0 + n, 10.0 * n)
                         for n in (64, 128, 256)},
          'flux': {n: _timed(50.0 + n / 2, 4.0 * n)
                   for n in (16, 32, 64, 128, 256)}}
    together = publish.step_series(at, step, 'wall', together=True)
    serial = publish.step_series(at, step, 'wall', together=False)
    assert [n for n, _, _ in together] == [64, 128, 256]
    assert [n for n, _, _ in serial] == [64, 128, 256]
    n, t, f = together[0]
    assert t == pytest.approx(164.0 + 4 * (50.0 + 32.0))
    assert f == pytest.approx(640.0 + 4 * 256.0)
    n, t, f = serial[0]
    assert t == pytest.approx(164.0 + 16 * (50.0 + 8.0))
    assert f == pytest.approx(together[0][2])
    assert publish.Launch(16, 0.25, 4).per_element == 4


def test_a_step_with_a_launch_not_measured_at_its_batch_is_left_out():
    step = {'derivative': publish.Launch(),
            'flux': publish.Launch(16, 0.25, 4)}
    at = {'derivative': {n: _timed(100.0 + n, 10.0 * n)
                         for n in (64, 128, 256)},
          'flux': {n: _timed(50.0 + n / 2, 4.0 * n) for n in (64, 128, 256)}}
    assert [n for n, _, _ in publish.step_series(at, step, 'wall', False)
            ] == [256]
    assert publish.step_series({'flux': at['flux']}, step, 'wall',
                               True) == []


def test_every_kernel_of_a_step_is_in_the_exports_of_its_equation():
    """A name in `STEP` that no export has would leave the breakdown and the
    time step out without a word of why but `skipped`."""
    for equation, step in publish.STEP.items():
        exports = sorted((BENCH / 'data').glob(f'{equation}-*.json'))
        assert exports, equation
        for path in exports:
            names = set(json.loads(path.read_text())['descriptions'])
            assert set(step) <= names, (path.name, set(step) - names)


# ----------------------------------------------------------------------
# Portability
# ----------------------------------------------------------------------

def test_portability_tells_a_failure_from_a_kernel_not_run(tmp_path):
    """Order 6 ran everywhere: the harmonic mean.  Order 4 failed on C: zero,
    as the definition has it.  Order 3 was not asked of C: no number at all
    -- a gap in a campaign says nothing about the code."""
    rows = [_point('gpu_volume', 'A', 6, 0.5), _point('gpu_volume', 'B', 6,
                                                      0.25),
            _point('gpu_volume', 'C', 6, 1.0),
            _point('gpu_volume', 'A', 4, 0.5), _point('gpu_volume', 'B', 4,
                                                      0.5),
            _point('gpu_volume', 'C', 4, error='exit 1: out of registers'),
            _point('gpu_volume', 'A', 3, 0.5), _point('gpu_volume', 'B', 3,
                                                      0.5)]
    field = _tree(tmp_path, rows)
    assert len(field.failed) == 1
    out = publish.table_portability(field, ['gpu_volume'])
    by = {r['order']: r for r in out.rows}
    assert by[6]['pp'] == pytest.approx(3 / (2 + 4 + 1))
    assert by[4]['pp'] == 0.0 and by[4]['failed_on'] == 'C'
    assert by[3]['pp'] is None and by[3]['not_run_on'] == 'C'
    assert r'\texttt{volume} & -- & 0 & 43 \\' in out.tex


def test_one_machine_makes_no_portability(tmp_path):
    field = _tree(tmp_path, [_point('gpu_volume', 'A', 4, 0.5)])
    assert publish.table_portability(field, ['gpu_volume']) is None


# ----------------------------------------------------------------------
# Across machines
# ----------------------------------------------------------------------

#: The kernels of an elastic time step (`publish.STEP`).
ELASTIC_STEP = ('gpu_derivative', 'gpu_volume', 'gpu_localFluxAll',
                'gpu_neighboringFlux[0]')


def test_machines_keep_one_order_and_colour_by_their_numbers(tmp_path):
    rows = [_point('gpu_volume', a, 4, 0.5)
            for a in ('sm_100', 'sm_80', 'gfx90a', 'sm_90')]
    look = publish.Machines(_tree(tmp_path, rows))
    assert look.names == ['gfx90a', 'sm_80', 'sm_90', 'sm_100']
    assert look.fill('sm_100') == 'tfc4' and look('gfx90a') == 'tf s1'


def test_the_peak_turned_round_is_a_bar_per_machine(tmp_path):
    """At one order, each kernel on every machine, as shares of the
    machine's own ceilings: at intensity 1 a kernel at half the roof moves
    half the triad and does a twentieth of the FMA chain (`CEILING`).  A
    kernel that failed on a machine is a cross; one not asked of it, a
    gap."""
    rows = [_point('gpu_volume', 'A', 4, 0.5),
            _point('gpu_volume', 'B', 4, 0.25),
            _point('gpu_volume', 'C', 4, error='exit 1: out of registers'),
            _point('gpu_derivative', 'A', 4, 0.5),
            _point('gpu_derivative', 'C', 4, 0.5),
            _point('gpu_volume', 'A', 6, 0.5),
            _point('gpu_volume', 'B', 6, 0.5)]
    field = _tree(tmp_path, rows)
    peaks, rates, skipped = publish.figure_machines(
        field, ['gpu_volume', 'gpu_derivative'], publish.Machines(field))
    assert not skipped
    assert [o.name for o in peaks] == ['machines-peak-elastic-f64-o4',
                                       'machines-peak-elastic-f64-o6']
    assert [o.name for o in rates] == ['machines-rate-elastic-f64-o4',
                                       'machines-rate-elastic-f64-o6']
    o4 = peaks[0]
    by = {(r['kernel'], r['arch']): r for r in o4.rows}
    assert by[('gpu_volume', 'A')]['fraction_of_roof'] == pytest.approx(0.5)
    assert by[('gpu_volume', 'A')]['fraction_of_triad'] == pytest.approx(0.5)
    assert by[('gpu_volume', 'A')]['fraction_of_fma'] == pytest.approx(0.05)
    assert by[('gpu_volume', 'B')]['fraction_of_roof'] == pytest.approx(0.25)
    assert by[('gpu_volume', 'C')]['failed'] is True
    assert ('gpu_derivative', 'B') not in by
    assert o4.tex.count('$\\times$') == 3          # one per panel
    assert o4.tex.count('\\addlegendentry') == 3   # A, B, C
    assert 'A cross marks' in o4.caption
    assert 'A cross marks' not in peaks[1].caption
    rate = {(r['kernel'], r['arch']): r for r in rates[0].rows}
    assert rate[('gpu_volume', 'A')]['gflops'] == pytest.approx(500.0)


def test_a_machine_without_a_ceiling_has_rates_but_no_shares(tmp_path):
    rows = [_point('gpu_volume', a, 4, 0.5) for a in 'AB']
    field = _tree(tmp_path, rows)
    del field.ceilings[('B', 'F64')]
    peaks, rates, skipped = publish.figure_machines(
        field, ['gpu_volume'], publish.Machines(field))
    assert not peaks and len(rates) == 1
    assert skipped == ['machines-peak: fewer than two machines with a '
                       'ceiling ran at elastic-f64-o4']


def test_a_time_step_per_element_on_every_machine(tmp_path):
    """Each kernel's time per element, the neighbour flux four times; a
    machine that misses one of the step's kernels has no step."""
    rows = [_point(k, a, 4, e) for k in ELASTIC_STEP
            for a, e in (('A', 0.5), ('B', 0.25))]
    rows += [_point(k, 'C', 4, 0.5) for k in ELASTIC_STEP[:3]]
    field = _tree(tmp_path, rows)
    out, skipped = publish.figure_machines_step(
        field, list(ELASTIC_STEP), publish.Styles(),
        publish.Machines(field))
    assert not skipped
    fig, = out
    assert fig.name == 'machines-step-elastic-f64-o4'
    assert {r['arch'] for r in fig.rows} == {'A', 'B'}
    ns = {(r['kernel'], r['arch']): r['ns_per_element'] for r in fig.rows}
    # 1 GFLOP at intensity 1 and half of a 1 TFLOP/s roof: 2 ms a launch.
    assert ns[('gpu_volume', 'A')] == pytest.approx(2e6 / 65536)
    assert ns[('gpu_neighboringFlux[0]', 'A')] == pytest.approx(
        4 * 2e6 / 65536)
    assert texout.num(7 * 2e6 / 65536) in fig.tex
    assert '\\texttt{C}' not in fig.tex


def test_the_time_step_ramps_up_on_every_machine(tmp_path):
    rows = []
    for a, t0 in (('A', 5000.0), ('B', 20000.0)):
        for k in ELASTIC_STEP:
            for b in (1024, 4096, 16384):
                r = _point(k, a, 4, 0.5, batch=b)
                r['wall_ns'] = t0 + 2.0 * b
                rows.append(r)
    field = _tree(tmp_path, rows)
    out, skipped = publish.figure_machines_rampup(
        field, list(ELASTIC_STEP), publish.Machines(field), 'wall')
    assert not skipped
    fig, = out
    assert fig.name == 'machines-rampup-elastic-f64-o4'
    half = {r['arch']: r['half_fit'] for r in fig.rows}
    # Seven launches of t0 + 2 n (the neighbour flux four times, its
    # relations taken together as one launch of n), so n-half is t0 / 2.
    assert half['A'] == pytest.approx(5000.0 / 2)
    assert half['B'] == pytest.approx(20000.0 / 2)


def test_one_roof_stands_for_every_machine(tmp_path):
    """The intensity over the machine's ridge and the rate over its FMA
    chain: under a ridge of 10, intensity 1 is a tenth of the way, and
    half the roof there is 5 % of A's FMA chain -- 2.5 % of B's, whose
    chain is twice as fast and whose ridge is twice as far."""
    import dataclasses
    rows = [_point('gpu_volume', a, 4, 0.5) for a in 'AB']
    rows.append(_point('gpu_derivative', 'B', 4, 0.5))
    field = _tree(tmp_path, rows)
    c = field.ceilings[('B', 'F64')]
    field.ceilings[('B', 'F64')] = dataclasses.replace(
        c, flops_per_s=2 * c.flops_per_s)
    out, skipped = publish.figure_machines_roofline(
        field, ['gpu_derivative', 'gpu_volume'], publish.Machines(field))
    assert not skipped
    fig, = out
    by = {(r['kernel'], r['arch']): r for r in fig.rows}
    assert by[('gpu_volume', 'A')]['intensity_over_ridge'] == \
        pytest.approx(0.1)
    assert by[('gpu_volume', 'A')]['fraction_of_fma'] == pytest.approx(0.05)
    assert by[('gpu_volume', 'B')]['intensity_over_ridge'] == \
        pytest.approx(0.05)
    assert by[('gpu_volume', 'B')]['fraction_of_fma'] == pytest.approx(0.025)
    # Both machines in the key, though A did not run the first panel's
    # kernel.
    assert fig.tex.count('\\addlegendentry') == 2
    assert '\\texttt{derivative}' in fig.tex


def test_a_long_kernel_name_breaks_where_a_word_starts():
    assert publish._breakable('derivativeTaylorExpansion') == (
        '\\texttt{derivative\\allowbreak{}Taylor\\allowbreak{}Expansion}')


def test_one_machine_is_compared_with_nothing(tmp_path):
    field = _tree(tmp_path, [_point('gpu_volume', 'A', 4, 0.5)])
    outputs, skipped = publish.build(field, [], ['gpu_volume'], 4, 'wall')
    assert not [o for o in outputs if o.name.startswith('machines-')]
    assert 'machines-*: one machine only' in skipped


# ----------------------------------------------------------------------
# A kernel is its export
# ----------------------------------------------------------------------

def test_a_roofline_draws_one_equation_and_nothing_of_another(tmp_path):
    """`gpu_derivative` is a kernel of both equations; each roofline has
    the points of its own, told apart here by their intensity."""
    rows = [_point('gpu_derivative', 'A', o, 0.5, equation='elastic',
                   intensity=1.0) for o in (3, 4)]
    rows += [_point('gpu_derivative', 'A', o, 0.5, equation='viscoelastic',
                    intensity=4.0) for o in (3, 4)]
    figures, skipped = publish.figure_rooflines(
        _tree(tmp_path, rows), ['gpu_derivative'], publish.Styles())
    assert not skipped
    by = {o.name: o for o in figures}
    assert set(by) == {'roofline-A-elastic-f64', 'roofline-A-viscoelastic-f64'}
    assert {round(r['intensity'], 9)
            for r in by['roofline-A-elastic-f64'].rows} == {1.0}
    assert {round(r['intensity'], 9)
            for r in by['roofline-A-viscoelastic-f64'].rows} == {4.0}
    assert all(r['fraction'] == pytest.approx(0.5)
               for o in figures for r in o.rows)


def test_a_ramp_up_needs_a_sweep_and_says_so(tmp_path):
    rows = [_point('gpu_volume', 'A', 4, 0.5, batch=b) for b in (4096, 65536)]
    figures, tables, skipped = publish.figure_rampup(
        _tree(tmp_path, rows), ['gpu_volume'], publish.Styles(), 'wall', 4)
    assert figures == [] and tables == []
    assert any('--batches 16,32,' in s for s in skipped)


# ----------------------------------------------------------------------
# From the exports alone
# ----------------------------------------------------------------------

def test_the_kernel_table_has_a_formula_for_every_default_kernel():
    import describe
    kernels = list(publish.DEFAULT_KERNELS)
    described = describe.describe_all([BENCH / 'data'], kernels)
    out = publish.table_kernels(described, kernels, order=4)
    assert out is not None
    rows = {(r['equation'], r['kernel']): r for r in out.rows}
    for equation, step in publish.STEP.items():
        for kernel in step:
            row = rows[(equation, kernel)]
            assert row['order'] == 4 and row['formula'] and row['yateto']
    intensity = publish.table_intensity(described, kernels)
    assert {r['precision'] for r in intensity.rows} == {'s', 'd'}
    assert all(r['intensity_dense'] > 0 for r in intensity.rows)


# ----------------------------------------------------------------------
# Local time stepping and the runs
# ----------------------------------------------------------------------

import seissolrun  # noqa: E402
from capture import Capture, Piece  # noqa: E402

HEADER = ('"profilingId","localId","layerType","size","dynamicRuptureSize",'
          '"rank","localRank","timestep","updateFactor","runId"\n')


def _clustering_csv(tmp_path, layers, run_id='mesh'):
    lines = [f'{i},{k},"{kind}",{cells},0,0,0,{dt},2,"{run_id}"'
             for i, (k, kind, cells, dt) in enumerate(layers)]
    path = tmp_path / 'mesh-clustering.csv'
    path.write_text(HEADER + '\n'.join(lines) + '\n')
    return path


def _fit(arch, t0_us, c_ns, reading='time step'):
    return {'arch': arch, 'kernel': reading, 'equation': 'elastic',
            'order': 4, 'datatype': 'F64', 'clock': 'wall',
            't0_us': t0_us, 'ns_per_element': c_ns,
            'half_fit': t0_us * 1e3 / c_ns}


def test_a_clustering_is_priced_with_each_machines_launch_cost(tmp_path):
    clustering = seissolrun.read_clustering(_clustering_csv(
        tmp_path, [(0, 'Interior', 1000, 1.0), (1, 'Interior', 50000, 2.0),
                   (2, 'Interior', 200000, 4.0)]))
    series = publish.lts_series(
        [(clustering, Capture.from_tag('elastic-4-d-s1'))], [])
    fits = [_fit('A', 50.0, 2.0), _fit('A', 120.0, 2.0, 'time step, serial'),
            _fit('B', 200.0, 4.0)]
    figures, skipped = publish.figure_lts(series, fits, [], None)
    assert not skipped
    out, = figures
    speed = {r['arch']: r for r in out.rows if r['source'] == 'speedup'}
    assert speed['A']['ideal'] == pytest.approx(clustering.ideal_speedup())
    assert speed['A']['predicted_together'] == pytest.approx(
        clustering.predicted_speedup(50e3, 2.0))
    assert speed['A']['predicted_serial'] == pytest.approx(
        clustering.predicted_speedup(120e3, 2.0))
    assert speed['B']['predicted_serial'] is None
    halves = {r['arch']: r for r in out.rows if r['source'] == 'kernels'}
    # n-half 25 000 cells on A: the first cluster's updates are below it.
    updates = [1000 / 1.0, 50000 / 2.0, 200000 / 4.0]
    assert halves['A']['updates_below'] == pytest.approx(
        updates[0] / sum(updates))
    assert 'ybar interval' in out.tex and r'\texttt{A}' in out.tex
    assert 'synthetic' not in out.caption


def test_without_launch_costs_the_clustering_is_still_drawn(tmp_path):
    clustering = seissolrun.read_clustering(
        publish.SYNTHETIC_CLUSTERING)
    series = publish.lts_series(
        [(clustering, Capture.from_tag('elastic-4-d-s1'))], [])
    figures, skipped = publish.figure_lts(series, [], [], None)
    assert len(figures) == 1 and 'synthetic, not from a mesh' in \
        figures[0].caption
    assert any('no time step was swept' in s for s in skipped)


def test_a_clustering_argument_names_its_kernels(tmp_path):
    path = _clustering_csv(tmp_path, [(0, 'Interior', 10, 1.0)])
    clustering, capture = publish.clustering_arg(f'{path}@viscoelastic-6-s',
                                                 4)
    assert capture.tag == 'viscoelastic-6-s-s1' and clustering.name == 'mesh'
    clustering, capture = publish.clustering_arg('synthetic', 3)
    assert clustering.synthetic and capture.tag == 'elastic-3-d-s1'
    with pytest.raises(SystemExit):
        publish.clustering_arg(f'{path}@nonsense', 4)
    with pytest.raises(SystemExit, match='--clustering'):
        publish.clustering_arg(f'{tmp_path / "absent.csv"}@elastic-4-d', 4)


def _summary(tmp_path, name, arch, *, clusters=1, seconds=100.0,
             config='elastic-linearck-o4-f64-stroud'):
    blob = {
        'schema': seissolrun.SCHEMA, 'status': 'completed',
        'build': {'deviceArch': arch, 'deviceBackend': 'cuda'},
        'configurations': [config], 'parallel': {'ranks': 2},
        'mesh': {'cells': {'sum': 1000.0}},
        'simulation': {'startTime': 0.0, 'endTime': 1.0,
                       'cellUpdates': 4e9},
        'time': {'simulation': {'total': {'max': seconds,
                                          'mean': seconds},
                                'io': {'mean': 5.0},
                                'outsideKernels': {'mean': 10.0}}},
        'kernels': {'regions': {
            'computeLocalIntegration': {'time': {'mean': 50.0}},
            'computeNeighboringIntegration': {'time': {'mean': 25.0}}}},
        'lts': {'clusters': [{'id': i, 'timestep': 2.0 ** i, 'cells': 500}
                             for i in range(clusters)]}}
    (tmp_path / f'{name}-summary.json').write_text(json.dumps(blob))


def test_what_the_runs_got_through_and_where_their_time_went(tmp_path):
    _summary(tmp_path, 'a', 'sm_90', seconds=100.0)
    _summary(tmp_path, 'b', 'gfx90a', seconds=200.0)
    runs = seissolrun.collect_runs([tmp_path])
    throughput, = publish.figure_throughput(runs)
    rates = {r['arch']: r['updates_per_s_per_rank'] for r in throughput.rows}
    assert rates == {'sm_90': pytest.approx(4e9 / 100 / 2),
                     'gfx90a': pytest.approx(4e9 / 200 / 2)}
    split, = publish.figure_timesplit(runs)
    first = next(r for r in split.rows if r['arch'] == 'sm_90')
    assert first['other_s'] == pytest.approx(100 - 50 - 25 - 10 - 5)
    assert 'xbar stacked' in split.tex


# ----------------------------------------------------------------------
# Fusion, the cascade, and what explains the curves
# ----------------------------------------------------------------------

def _split_run(tmp_path, rows):
    """The split suite's run, beside the main one as `evaluate.py` lays
    them out: every row from one binary."""
    d = tmp_path / 'A' / 'split' / 'elastic-4-d-s1'
    d.mkdir(parents=True)
    (d / 'bench.json').write_text(json.dumps({'run': {'suite': 'split'},
                                              'results': [
        dict(r, unit='seissolsplit-cuda-A-f64-baseline') for r in rows]}))


def test_the_pieces_of_a_kernel_are_added_up_against_it(tmp_path):
    """Against the whole kernel of their own run, not against the main
    run's measurement of it, which is another clock."""
    main, split = [], []
    for batch in (256, 65536):
        main.append(dict(_point('gpu_volume', 'A', 4, 0.5, batch=batch),
                         wall_ns=4000.0 * batch / 256))
        split.append(dict(_point(Piece('gpu_volume', 0, 1).name, 'A', 4,
                                 0.5, batch=batch),
                          wall_ns=1000.0 * batch / 256))
        for i in range(3):
            split.append(dict(_point(Piece('gpu_volume', i, 3).name, 'A', 4,
                                     0.5, batch=batch),
                              wall_ns=500.0 * batch / 256, bytes=2e9))
    _split_run(tmp_path, split)
    out, = publish.figure_fusion(_tree(tmp_path, main), ['gpu_volume'],
                                 'wall')[0]
    entry, = out.rows
    assert (entry['batch_large'], entry['batch_small']) == (65536, 256)
    assert entry['ratio_large'] == pytest.approx(1.5)
    assert entry['ratio_small'] == pytest.approx(1.5)
    assert entry['bytes_ratio'] == pytest.approx(6.0)


def test_no_pieces_say_where_to_get_them(tmp_path):
    made, why = publish.figure_fusion(
        _tree(tmp_path, [_point('gpu_volume', 'A', 4, 0.5)]),
        ['gpu_volume'], 'wall')
    assert made == [] and 'seissolsplit' in why[0]


def test_pieces_without_their_whole_kernel_in_the_run_say_so(tmp_path):
    _split_run(tmp_path, [_point(Piece('gpu_volume', i, 2).name, 'A', 4,
                                 0.5, batch=256) for i in range(2)])
    made, why = publish.figure_fusion(
        _tree(tmp_path, [_point('gpu_volume', 'A', 4, 0.5, batch=256)]),
        ['gpu_volume'], 'wall')
    assert made == [] and 'same run' in why[0]


def test_the_cascade_puts_the_most_efficient_machine_first(tmp_path):
    rows = [_point('gpu_volume', a, 4, e)
            for a, e in (('A', 0.25), ('B', 0.5), ('C', 1.0))]
    out, = publish.figure_cascade(_tree(tmp_path, rows), ['gpu_volume'], 4,
                                  publish.Styles())
    assert [r['arch'] for r in out.rows] == ['C', 'B', 'A']
    assert [r['pp_first'] for r in out.rows] == pytest.approx(
        [1.0, 2 / 3, 3 / 7])


def test_lanes_and_resources_come_from_what_the_build_recorded(tmp_path):
    # On NVIDIA the stack frame lands under `scratchsize`, and it holds more
    # than spills: what spilled is the traffic, as the ranking reads it.
    rows = [_point('gpu_volume', 'A', o, 0.5, executed_flops=2e9,
                   nonzero_flops=1e9 * share, lanes=32,
                   static={'vgprs': 40 * o, 'spillstores': 16 * (o - 3),
                           'spillloads': 16 * (o - 3), 'scratchsize': 64},
                   code_bytes=30000 * o, icache_bytes=131072)
            for o, share in ((3, 0.25), (4, 0.5))]
    styles = publish.Styles()
    fieldset = _tree(tmp_path, rows)
    lanes, = publish.figure_lanes(fieldset, ['gpu_volume'], styles)
    assert [(r['order'], r['needed_of_issued'], r['dense_of_issued'])
            for r in lanes.rows] == [(3, 0.125, 0.5), (4, 0.25, 0.5)]
    resources, = publish.figure_resources(fieldset, ['gpu_volume'], styles)
    assert {(r['order'], r['registers'], r['spilled_bytes'],
             r['code_bytes']) for r in resources.rows} == {
        (3, 120, 0, 90000), (4, 160, 32, 120000)}
    assert '128\\,KiB, the rule' in resources.caption


def test_double_against_single_beside_the_machines_rates(tmp_path):
    rows = [_point('gpu_volume', 'A', 4, 0.5, precision=p, intensity=i)
            for p, i in (('d', 1.0), ('s', 2.0))]
    out, = publish.figure_precision(
        _tree(tmp_path, rows, datatypes=('F64', 'F32')), ['gpu_volume'])
    row, = out.rows
    # Same rate on a memory roof, half the bytes: twice as fast in single.
    assert row['f64_over_f32'] == pytest.approx(2.0)
    assert '(FMA 2.0:1)' in out.tex


def test_fused_simulations_per_simulation(tmp_path):
    rows = [_point('gpu_volume', 'A', 4, 0.5, simulations=1),
            dict(_point('gpu_volume', 'A', 4, 0.5, simulations=8),
                 event_ns=4 * _point('gpu_volume', 'A', 4, 0.5)['event_ns'])]
    out, = publish.figure_simulations(_tree(tmp_path, rows), ['gpu_volume'],
                                      4, publish.Styles())
    assert [(r['simulations'], r['speedup']) for r in out.rows] == [
        (1, 1.0), (8, pytest.approx(2.0))]


def test_traffic_is_matched_to_the_run_by_arch_and_symbol(tmp_path):
    """The symbol is a digest of the source, which another architecture of
    the same backend can share: a profile counts for the machine it ran
    on."""
    rows = [_point('gpu_volume', 'A', 4, 0.5, batch=b, symbol='kernel_x')
            for b in (1024, 65536)]
    for r in rows:
        r['bytes'] = 1000.0 + 100.0 * r['batch']
    fieldset = _tree(tmp_path, rows)
    for arch, read in (('A', 600000.0), ('B', 9e9)):
        profile = tmp_path / arch / 'profiles' / 'profile.json'
        profile.parent.mkdir(parents=True)
        profile.write_text(json.dumps({
            'batch': 4096,
            'device': {'spec': {'backend': 'cuda', 'arch': arch}},
            'collections': [
                {'workload': 'gpu_volume', 'symbol': 'kernel_x',
                 'rows': [{'duration_ns': 5.0, 'dram_read_bytes': read,
                           'dram_write_bytes': 211400.0}]}]}))
    profiles = publish._profiles([tmp_path])
    assert profiles[('A', 'kernel_x')] == (811400.0, 4096)
    out, = publish.figure_traffic(fieldset, profiles, ['gpu_volume'])
    row, = out.rows
    assert row['compulsory_bytes'] == pytest.approx(1000 + 100 * 4096)
    assert row['ratio'] == pytest.approx(811400 / 410600)


def test_a_profile_counts_for_the_machine_of_its_backend(tmp_path):
    """One GPU under two backends is two machines; the profile names the
    GPU, and the kernel's symbol tells the backends apart."""
    rows = [dict(_point('gpu_volume', 'pvc', 4, 0.5, batch=b,
                        symbol=f'kernel_{backend}'), backend=backend)
            for backend in ('oneapi', 'esimd') for b in (1024, 65536)]
    for r in rows:
        r['bytes'] = 1000.0 + 100.0 * r['batch']
    fieldset = _tree(tmp_path, rows)
    profile = tmp_path / 'pvc' / 'profiles' / 'profile.json'
    profile.parent.mkdir(parents=True)
    profile.write_text(json.dumps({
        'batch': 4096,
        'device': {'spec': {'backend': 'oneapi', 'arch': 'pvc'}},
        'collections': [
            {'workload': 'gpu_volume', 'symbol': 'kernel_esimd',
             'rows': [{'duration_ns': 5.0, 'dram_read_bytes': 600000.0,
                       'dram_write_bytes': 211400.0}]}]}))
    out, = publish.figure_traffic(fieldset, publish._profiles([tmp_path]),
                                  ['gpu_volume'])
    assert out.name == 'traffic-esimd-pvc-elastic-f64'
    assert '(esimd)' in out.caption


def test_what_making_the_kernels_cost_is_summed_per_export(tmp_path):
    manifest = [{'workload': w, 'arch': 'A', 'datatype': 'F64',
                 'config': 'baseline', 'capture': 'elastic-4-d-s1',
                 'error': '', 'generate_s': g, 'compile_s': 10.0,
                 'object_bytes': 2 ** 20, 'code_bytes': c,
                 'icache_bytes': 131072}
                for w, g, c in (('a', 1.5, 1024), ('b', 2.5, 4096))]
    manifest.append(dict(manifest[0], workload='c', error='refused'))
    # Not a kernel of the build: a piece of the split suite.
    manifest.append(dict(manifest[0], workload=Piece('a', 0, 2).name))
    fieldset = _tree(tmp_path, [_point('gpu_volume', 'A', 4, 0.5)],
                     manifest=manifest)
    # The whole configuration, built by another run: `a` again, and `d`.
    build = tmp_path / 'A' / 'build' / 'elastic-4-d-s1'
    build.mkdir(parents=True)
    (build / 'bench.json').write_text(json.dumps({
        'run': {'suite': 'test'}, 'results': [],
        'manifest': [dict(manifest[0], compile_s=None),
                     dict(manifest[1], workload='d')]}))
    out, = publish.table_buildcost(publish._manifests([tmp_path]), fieldset)
    row, = out.rows
    assert (row['kernels'], row['generate_s'], row['compile_s'],
            row['object_bytes'], row['largest_code_bytes']) == (
        3, 6.5, 30.0, 3 * 2 ** 20, 4096)
    assert '128' in out.caption and row['suite'] == 'test'


def test_the_builds_of_two_backends_on_one_gpu_cost_apart(tmp_path):
    """SPMD and ESIMD make kernels of the same names on one GPU: each
    machine its own build cost (`compare.Field.split_backends`)."""
    rows = [dict(_point('gpu_volume', 'pvc', 4, 0.5), backend=b)
            for b in ('oneapi', 'esimd')]
    manifest = [{'workload': 'gpu_volume', 'arch': 'pvc', 'backend': b,
                 'datatype': 'F64', 'config': 'baseline',
                 'capture': 'elastic-4-d-s1', 'error': '', 'generate_s': g,
                 'compile_s': 10.0, 'object_bytes': 2 ** 20,
                 'code_bytes': 1024, 'icache_bytes': 131072}
                for b, g in (('oneapi', 1.0), ('esimd', 3.0))]
    fieldset = _tree(tmp_path, rows, manifest=manifest)
    out = {o.name: o for o in publish.table_buildcost(
        publish._manifests([tmp_path]), fieldset)}
    assert sorted(out) == ['buildcost-esimd-pvc', 'buildcost-oneapi-pvc']
    assert out['buildcost-esimd-pvc'].rows[0]['generate_s'] == 3.0
    assert '(esimd)' in out['buildcost-esimd-pvc'].caption


# ----------------------------------------------------------------------
# The kernels drawn
# ----------------------------------------------------------------------

def _described(*kernels):
    import describe
    return describe.describe_all([BENCH / 'data'], kernels, costs=False)


def test_a_formula_is_drawn_with_what_the_kernel_reaches():
    """The volume kernel at order 4: the stiffness matrices as one family
    over the reach of all three, drawn with the first one's nonzeros; the
    time integral as the ten rows the kernel reads of it."""
    out = {o.name: o for o in publish.figure_operations(
        _described('gpu_volume', 'gpu_derivative'),
        ['gpu_derivative', 'gpu_volume'])}
    assert 'operations-elastic-o4' in out
    assert any(n.startswith('operations-viscoelastic-o4') for n in out)
    rows = [r for r in out['operations-elastic-o4'].rows
            if r['kernel'] == 'gpu_volume']
    by = {r['tensor']: r for r in rows}
    assert (by['Q']['rows'], by['Q']['cols']) == (20, 9)
    k = by['kDivM(0)']
    assert k['members'] == 'kDivM(0) kDivM(1) kDivM(2)'
    assert (k['rows'], k['cols'], k['shared'], k['nonzeros']) == (
        20, 10, True, 33)
    assert (by['I']['rows'], by['I']['cols'], by['I']['shared']) == (
        10, 9, False)
    # The recursion at its first step: the first derivative is ten rows.
    first = [r for r in out['operations-elastic-o4'].rows
             if r['kernel'] == 'gpu_derivative' and r['tensor'] == 'dQ(1)']
    assert first and first[0]['rows'] == 10
    assert '\\sum_{i=0}^{2}' in out['operations-elastic-o4'].tex


def test_a_time_step_taller_than_a_page_comes_in_parts():
    kernels = list(publish.DEFAULT_KERNELS)
    parts = [o for o in publish.figure_operations(_described(*kernels),
                                                  kernels)
             if o.name.startswith('operations-viscoelastic-o6')]
    assert [o.name for o in parts] == [
        f'operations-viscoelastic-o6-{i + 1}' for i in range(len(parts))]
    assert len(parts) > 1 and parts[0].rows and not parts[-1].rows
    # Every part says whose rows it holds, and what the colours mean.
    assert all('\\texttt{' in o.tex and 'each element its own' in o.tex
               for o in parts)
    # As even as that many parts can be: none of them a leftover.
    heights = [-min(float(y) for y in re.findall(r'\(-?[\d.]+,(-?[\d.]+)\)',
                                                  o.tex)) for o in parts]
    assert max(heights) <= publish._OPS_HEIGHT
    assert min(heights) > 0.75 * max(heights)


def test_the_shared_operators_are_drawn_by_family():
    out = {o.name: o for o in publish.figure_operators(
        _described('gpu_volume', 'gpu_localFluxAll'),
        ['gpu_volume', 'gpu_localFluxAll'])}
    rows = out['operators-elastic-o4'].rows
    assert [r['operator'] for r in rows] == [
        'kDivM(0)', 'kDivM(1)', 'kDivM(2)', 'fMrT(0)', 'fMrT(1)', 'fMrT(2)',
        'fMrT(3)', 'rDivM(0)', 'rDivM(1)', 'rDivM(2)', 'rDivM(3)']
    assert [r['nonzeros'] for r in rows[:3]] == [33, 77, 92]


# ----------------------------------------------------------------------
# texout
# ----------------------------------------------------------------------

def test_names_from_data_are_set_as_text():
    assert texout.escape('gpu_neighboringFlux[0] & 50%') == \
        r'gpu\_neighboringFlux[0] \& 50\%'


def test_a_number_in_a_table_has_significant_digits_and_thin_spaces():
    assert texout.num(None) == '--'
    assert texout.num(float('nan')) == '--'
    assert texout.num(12.345) == '12.3'
    assert texout.num(0.012345) == '0.0123'
    assert texout.num(123456.0) == r'123\,456'
    assert texout.num(0.0) == '0'


def test_a_table_is_booktabs_with_its_groups():
    tex = texout.table('l r', [['a', 'b'], r'\cmidrule{1-2}'],
                       [['x', '1'], None, ['y', '2'], texout.GAP,
                        ['z', '3']])
    lines = tex.splitlines()
    assert lines[:2] == [r'\begin{tabular}{l r}', r'\toprule']
    assert r'\cmidrule{1-2}' in lines
    assert lines.index(r'y & 2 \\') == lines.index(r'\midrule', 5) + 1
    assert r'\addlinespace' in lines and lines[-2] == r'\bottomrule'


def test_a_gap_in_a_line_is_left_open():
    plot = texout.Plot('tf s1', [(1, 2.0), (2, None), (4, 3.0)], 'k',
                       labels=['3', '', '6'])
    text = plot.render()
    assert 'nan nan {}' in text and 'unbounded coords=jump' in text
    assert 'meta=m' in text and '1 2 {3}' in text
    assert text.startswith(r'\addplot[tf s1')
    assert r'\addlegendentry{k}' in text


def test_a_rule_spans_the_axis_whatever_its_x_axis():
    rule = texout.hrule(100, 'roof', pos=0.5)
    assert '{rel axis cs:0,0}|-{axis cs:1,100}' in rule
    assert '{rel axis cs:1,0}|-{axis cs:1,100}' in rule
    assert 'node[pos=0.50, anchor=south east, tf label] {roof}' in rule


# ----------------------------------------------------------------------
# The whole of it, typeset
# ----------------------------------------------------------------------

def _tex_with_pgfplots() -> bool:
    if shutil.which('pdflatex') is None or shutil.which('kpsewhich') is None:
        return False
    found = subprocess.run(['kpsewhich', 'pgfplots.sty'], capture_output=True,
                           text=True, check=False)
    return bool(found.stdout.strip())


def test_what_is_written_is_complete_and_says_what_it_left_out(tmp_path):
    rows = [_point(k, a, o, 0.4 + 0.1 * i, batch=b)
            for i, a in enumerate('AB') for o in (3, 4)
            for k in ('gpu_volume', 'gpu_derivative')
            for b in (256, 4096, 65536)]
    field = _tree(tmp_path / 'runs', rows)
    outputs, skipped = publish.build(field, [], ['gpu_volume',
                                                 'gpu_derivative'], 4, 'wall')
    out = tmp_path / 'pub'
    publish.write(outputs, out, 'test', {'skipped': skipped})
    names = {o.name for o in outputs}
    assert {'machines', 'portability', 'roofline-A-elastic-f64',
            'peak-B-elastic-f64', 'rampup-A'} <= names
    assert any(s.startswith('kernels, intensity') for s in skipped)
    for o in outputs:
        where = 'figures' if o.kind == 'figure' else 'tables'
        assert (out / where / f'{o.name}.tex').exists()
    with (out / 'data' / 'portability.csv').open() as fh:
        assert len(list(csv.DictReader(fh))) == 4
    assert 'kernels, intensity' in (out / 'preview.tex').read_text()


@pytest.mark.skipif(not _tex_with_pgfplots(),
                    reason='needs pdflatex with pgfplots')
def test_the_preview_compiles(tmp_path):
    """Every kind of output at once, so that each one's LaTeX is set: two
    machines, two orders, both precisions, fused simulations, a kernel in
    pieces, what the builds recorded, a profile, SeisSol runs and the
    synthetic clustering."""
    import describe
    kernels = ['gpu_volume', 'gpu_derivative']
    batches = (256, 1024, 4096, 16384, 65536)
    built = dict(lanes=32, static={'vgprs': 96}, code_bytes=90000,
                 icache_bytes=131072, executed_flops=1.5e9,
                 nonzero_flops=6e8)
    rows = [_point(k, a, o, 0.4 + 0.1 * i, batch=b, precision=p,
                   symbol=f'kernel_{k}_{a}_{o}_{p}', **built)
            for i, a in enumerate('AB') for o in (3, 4) for p in 'ds'
            for k in kernels for b in batches]
    rows += [_point(k, 'A', 4, 0.6, batch=b, simulations=8, **built)
             for k in kernels for b in batches]
    rows += [dict(_point(Piece('gpu_volume', i, n).name, 'A', o, 0.3,
                         batch=b), bytes=1.5e9)
             for i, n in ((0, 1), (0, 2), (1, 2)) for o in (3, 4)
             for b in batches]
    manifest = [{'workload': k, 'arch': a, 'datatype': 'F64',
                 'config': 'baseline', 'capture': f'elastic-{o}-d-s1',
                 'error': '', 'generate_s': 2.0, 'compile_s': 9.0,
                 'object_bytes': 2 ** 20, 'code_bytes': 90000,
                 'icache_bytes': 131072}
                for k in kernels for a in 'AB' for o in (3, 4)]
    field = _tree(tmp_path / 'runs', rows, manifest=manifest,
                  datatypes=('F64', 'F32'))
    profile = tmp_path / 'runs' / 'A' / 'profiles' / 'profile.json'
    profile.parent.mkdir(parents=True)
    profile.write_text(json.dumps({
        'batch': 65536, 'device': {'spec': {'backend': 'cuda', 'arch': 'A'}},
        'collections': [
            {'workload': 'gpu_volume', 'symbol': 'kernel_gpu_volume_A_4_d',
             'rows': [{'duration_ns': 1.0, 'dram_read_bytes': 7e10}]}]}))
    seissol = tmp_path / 'seissol'
    seissol.mkdir()
    _summary(seissol, 'gts', 'A', seconds=300.0)
    _summary(seissol, 'lts', 'A', clusters=3, seconds=100.0)
    _summary(seissol, 'other', 'B', seconds=150.0)
    extra = publish.Extra(
        runs=seissolrun.collect_runs([seissol]),
        clusterings=[publish.clustering_arg('synthetic', 4)],
        profiles=publish._profiles([tmp_path / 'runs']),
        manifests=publish._manifests([tmp_path / 'runs']))
    described = describe.describe_all([BENCH / 'data'], kernels)
    outputs, skipped = publish.build(field, described, kernels, 4, 'wall',
                                     extra)
    kinds = {o.name.split('-')[0] for o in outputs}
    assert {'lts', 'timesplit', 'throughput', 'cascade', 'machines',
            'fusion', 'lanes', 'resources', 'precision', 'simulations',
            'traffic', 'buildcost'} <= kinds, sorted(kinds)
    out = tmp_path / 'pub'
    publish.write(outputs, out, 'test', {'skipped': skipped})
    ok, why = texout.compile_tex(out / 'preview.tex')
    assert ok, why
    assert (out / 'preview.pdf').read_bytes().startswith(b'%PDF')
    if shutil.which('pdffonts'):
        # Every font embedded, and none a Type 3 bitmap: the two things a
        # publisher's PDF check turns a paper away for.
        fonts = subprocess.run(['pdffonts', str(out / 'preview.pdf')],
                               capture_output=True, text=True,
                               check=True).stdout.splitlines()[2:]
        assert fonts
        assert not [f for f in fonts if 'Type 3' in f]
        assert all(f.split()[-5] == 'yes' for f in fonts)


@pytest.mark.skipif(not _tex_with_pgfplots(),
                    reason='needs pdflatex with pgfplots')
def test_the_figures_across_machines_compile(tmp_path):
    """Each figure across machines, with the time step the bars and the
    ramp-up need and a failure among them, set as the preview sets it."""
    rows = []
    for i, a in enumerate(('sm_80', 'gfx942', 'esimd-pvc')):
        for k in ELASTIC_STEP:
            for b in (1024, 4096, 16384, 65536):
                if (a, k) == ('gfx942', 'gpu_volume'):
                    continue
                r = _point(k, a, 4, 0.3 + 0.2 * i, batch=b)
                r['wall_ns'] = 4000.0 * (i + 1) + 2.0 * b
                rows.append(r)
    rows.append(_point('gpu_volume', 'gfx942', 4, error='exit 1'))
    field = _tree(tmp_path / 'runs', rows)
    outputs, skipped = publish.build(field, [], list(ELASTIC_STEP), 4,
                                     'wall')
    names = {o.name for o in outputs if o.name.startswith('machines-')}
    assert names == {f'machines-{kind}-elastic-f64-o4' for kind in
                     ('peak', 'rate', 'step', 'rampup', 'roofline')}
    out = tmp_path / 'pub'
    publish.write([o for o in outputs if o.name in names], out, 'test',
                  {'skipped': skipped})
    ok, why = texout.compile_tex(out / 'preview.tex')
    assert ok, why
