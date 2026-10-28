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
           batch=65536, intensity=1.0, error=None, precision='d'):
    """A row at `efficiency` of the roof (`CEILING`) at `intensity`, or a
    row that produced no number."""
    tag = f'{equation}-{order}-{precision}-s1'
    row = {'workload': kernel, 'config': 'baseline', 'arch': arch,
           'backend': 'cuda', 'device': f'device {arch}',
           'datatype': {'d': 'F64', 's': 'F32'}[precision],
           'capture': tag, 'equation': equation, 'order': order,
           'precision': precision, 'simulations': 1}
    if error:
        return dict(row, error=error)
    flops = 1e9
    byts = flops / intensity
    roof = min(CEILING['flops_per_s'], CEILING['bytes_per_s'] * intensity)
    ns = flops / (efficiency * roof) * 1e9
    return dict(row, batch=batch, flops=flops, bytes=byts, event_ns=ns,
                wall_ns=ns + 2000.0)


def _tree(tmp_path, rows):
    """One run directory per machine, each with its ceiling beside it."""
    for arch in sorted({r['arch'] for r in rows}):
        d = tmp_path / arch / 'run'
        d.mkdir(parents=True)
        (d / 'bench.json').write_text(json.dumps(
            {'results': [r for r in rows if r['arch'] == arch]}))
        (d / 'ceiling.json').write_text(json.dumps({'ceiling': dict(
            CEILING, arch=arch, backend='cuda', device=f'device {arch}',
            datatype='F64')}))
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
    import describe
    kernels = ['gpu_volume', 'gpu_derivative']
    rows = [_point(k, a, o, 0.4 + 0.1 * i, batch=b)
            for i, a in enumerate('AB') for o in (3, 4)
            for k in kernels for b in (256, 1024, 4096, 16384, 65536)]
    field = _tree(tmp_path / 'runs', rows)
    described = describe.describe_all([BENCH / 'data'], kernels)
    outputs, skipped = publish.build(field, described, kernels, 4, 'wall')
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
