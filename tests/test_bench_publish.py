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
    assert {'lts', 'timesplit', 'throughput', 'cascade', 'fusion', 'lanes',
            'resources', 'precision', 'simulations', 'traffic',
            'buildcost'} <= kinds, sorted(kinds)
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
