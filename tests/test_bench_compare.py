# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""The cross-run views, and the PDF both plotters write.

`test_bench_plot.py` covers one run. What is new here is everything that
only exists across runs -- the order parsed out of a directory name, a
ratio taken between two configurations, a kernel placed against a machine's
ridge -- and the second rendering of a chart, which has no browser to be
forgiving about it.

A PDF is checked as a PDF: the header, the object table pointing at real
objects, and the page count. A file that opens in a viewer here and nowhere
else is a file nobody can put in a paper.
"""

from __future__ import annotations

import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / 'tools' / 'bench'
if str(BENCH) not in sys.path:
    sys.path.insert(0, str(BENCH))

import compare  # noqa: E402
import pdfout  # noqa: E402
import plot  # noqa: E402


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------

def _row(workload, config, arch, batch, ns, flops=1e9, byts=1e8,
         backend='cuda', device='', datatype='F32'):
    return {'workload': workload, 'config': config, 'arch': arch,
            'backend': backend, 'device': device or arch, 'datatype': datatype,
            'batch': batch, 'flops': flops, 'bytes': byts, 'event_ns': ns,
            'gflops': flops / ns, 'gbytes': byts / ns,
            'ns_per_element': ns / batch}


def _write(tmp_path, name, rows, ceiling=None):
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    (d / 'bench.json').write_text(json.dumps({'results': rows}))
    if ceiling:
        (d / 'ceiling.json').write_text(json.dumps({'ceiling': ceiling}))
    return d


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------

@pytest.mark.parametrize('name,expected', [
    ('elastic-6-s-s1', 6),                 # what evaluate.py names an export
    ('elastic-linearck-o8-s', 8),          # what the recorded suite calls one
    ('viscoelastic-3-d-s8', 3),
    ('sm_120', None),                      # a device, not an order
    ('runs', None),
])
def test_the_order_is_read_from_the_run_directory(name, expected):
    assert compare.order_of(Path('/eval/runs') / name, []) == expected


def test_a_row_that_carries_its_order_is_believed_over_the_path():
    """The path is a fallback for a field that does not exist yet; where
    the field appears it is the one that was recorded."""
    assert compare.order_of(Path('/eval/elastic-6-s-s1'),
                            [{'order': 4}]) == 4


def test_an_order_nobody_recorded_is_not_guessed(tmp_path):
    """Placing such a run at zero would put a point on the order axis that
    no configuration was ever built at."""
    _write(tmp_path, 'nameless', [_row('k', 'baseline', 'sm_120', 64, 100.0)])
    assert compare.collect([tmp_path]).orders == []


def test_a_failed_row_is_not_a_point(tmp_path):
    _write(tmp_path, 'elastic-3-s-s1',
           [_row('k', 'baseline', 'sm_120', 64, 100.0),
            {'workload': 'broken', 'error': 'not built'}])
    assert [p.workload for p in compare.collect([tmp_path]).points] == ['k']


def test_one_bench_file_is_read_once_however_many_paths_name_it(tmp_path):
    d = _write(tmp_path, 'elastic-3-s-s1',
               [_row('k', 'baseline', 'sm_120', 64, 100.0)])
    both = compare.collect([tmp_path, d, d / 'bench.json'])
    assert len(both.points) == 1


def test_the_largest_launch_is_the_one_compared(tmp_path):
    """A ratio taken where the scaling curve is still climbing is a ratio
    at an arbitrary place on a slope."""
    _write(tmp_path, 'elastic-3-s-s1',
           [_row('k', 'baseline', 'sm_120', 64, 100.0),
            _row('k', 'baseline', 'sm_120', 4096, 2000.0)])
    got = compare.collect([tmp_path])
    assert [p.batch for p in got.largest_batch(got.points)] == [4096]


# ----------------------------------------------------------------------
# What the views say
# ----------------------------------------------------------------------

def _two_configs(tmp_path):
    rows = []
    for cfg, ns in (('baseline', 1000.0), ('faster', 500.0)):
        rows.append(_row('k', cfg, 'sm_120', 1024, ns))
    return _write(tmp_path, 'elastic-3-s-s1', rows)


def test_a_cost_ratio_points_the_way_a_reader_expects(tmp_path):
    """Half the nanoseconds is twice as fast, so the bar is above one --
    the view inverts for the metrics where lower is better."""
    _two_configs(tmp_path)
    svg = compare.view_configs(compare.collect([tmp_path]), 'ns_per_element')
    assert svg and 'faster is 2.000x baseline' in svg


def test_a_rate_ratio_is_not_inverted(tmp_path):
    _two_configs(tmp_path)
    svg = compare.view_configs(compare.collect([tmp_path]), 'gflops')
    assert svg and 'faster is 2.000x baseline' in svg


def test_one_configuration_has_nothing_to_compare(tmp_path):
    _write(tmp_path, 'elastic-3-s-s1',
           [_row('k', 'baseline', 'sm_120', 64, 100.0)])
    assert compare.view_configs(compare.collect([tmp_path])) is None


def test_a_kernel_is_bound_by_whichever_ceiling_the_machine_runs_out_of(
        tmp_path):
    """The same kernel, two machines, two answers: the intensity is the
    operation's and the ridge is the machine's."""
    rows = [_row('k', 'baseline', 'sm_120', 1024, 100.0, flops=1e9, byts=1e8)]
    _write(tmp_path, 'a', rows,
           {'arch': 'sm_120', 'flops_per_s': 1e13, 'bytes_per_s': 1e12})
    rows2 = [_row('k', 'baseline', 'gfx1', 1024, 100.0, flops=1e9, byts=1e8,
                  backend='hip')]
    _write(tmp_path, 'b', rows2,
           {'arch': 'gfx1', 'flops_per_s': 1e12, 'bytes_per_s': 1e12})
    got = compare.collect([tmp_path])
    # intensity is 10 flop/byte; the ridges are 10 and 1.
    svg = compare.view_intensity(got)
    assert svg
    assert 'memory-bound' in svg or 'compute-bound' in svg
    assert 'ridge of 10.0' in svg and 'ridge of 1.0' in svg


def test_portability_divides_each_machine_by_its_own_roof(tmp_path):
    """Which is the whole point of it: the absolute view ranks the
    machines, this one ranks what was made of them."""
    _write(tmp_path, 'a',
           [_row('k', 'baseline', 'fast', 1024, 100.0, flops=1e9, byts=1e8)],
           {'arch': 'fast', 'flops_per_s': 1e13, 'bytes_per_s': 1e12})
    _write(tmp_path, 'b',
           [_row('k', 'baseline', 'slow', 1024, 1000.0, flops=1e9, byts=1e8,
                 backend='hip')],
           {'arch': 'slow', 'flops_per_s': 1e12, 'bytes_per_s': 1e11})
    svg = compare.view_portability(compare.collect([tmp_path]))
    assert svg
    # Ten times the rate on ten times the machine is the same fraction, so
    # the two bars have to agree -- which is the claim, and an empty match
    # would pass it vacuously.
    fractions = re.findall(r': ([\d.]+)% of its roof', svg)
    assert len(fractions) == 2, svg
    assert len(set(fractions)) == 1, fractions


def test_the_order_view_needs_two_orders(tmp_path):
    _write(tmp_path, 'elastic-3-s-s1',
           [_row('k', 'baseline', 'sm_120', 1024, 100.0)])
    assert compare.view_order(compare.collect([tmp_path])) is None
    _write(tmp_path, 'elastic-6-s-s1',
           [_row('k', 'baseline', 'sm_120', 1024, 400.0)])
    assert compare.view_order(compare.collect([tmp_path])) is not None


def test_a_kernel_missing_from_one_order_leaves_a_gap(tmp_path):
    """Rather than being carried across at its neighbour's value."""
    _write(tmp_path, 'elastic-3-s-s1',
           [_row('a', 'baseline', 'sm_120', 1024, 100.0),
            _row('b', 'baseline', 'sm_120', 1024, 100.0)])
    _write(tmp_path, 'elastic-6-s-s1',
           [_row('a', 'baseline', 'sm_120', 1024, 400.0)])
    _write(tmp_path, 'elastic-8-s-s1',
           [_row('a', 'baseline', 'sm_120', 1024, 900.0),
            _row('b', 'baseline', 'sm_120', 1024, 900.0)])
    svg = compare.view_order(compare.collect([tmp_path]))
    assert svg and 'not run' in svg
    ET.fromstring(svg)


# ----------------------------------------------------------------------
# The PDF
# ----------------------------------------------------------------------

def _canvas():
    c = plot.Canvas('t', 'x', 'y')
    c.fit([0, 4], [0, 4])
    c.axes()
    c.line([(0, 1), (1, 2), (2, 3)], '#2f6fb2', label='s')
    c.bar(c.L + 10, 12, 2.0, '#b2452f', 'a bar')
    return c


def test_a_written_pdf_is_a_pdf(tmp_path):
    out = tmp_path / 'r.pdf'
    pdfout.write_pdf([_canvas().to_page('one'),
                      _canvas().to_page('two')], out, 'title')
    blob = out.read_bytes()
    assert blob.startswith(b'%PDF-1.')
    assert blob.rstrip().endswith(b'%%EOF')
    assert b'/Count 2' in blob


def test_every_offset_in_the_table_points_at_its_object(tmp_path):
    """A cross-reference table that is off by a byte opens in one viewer
    and not the next, which is the failure this catches here instead of in
    somebody's mail client."""
    out = tmp_path / 'r.pdf'
    pdfout.write_pdf([_canvas().to_page('one')], out)
    blob = out.read_bytes()
    start = int(re.search(rb'startxref\s+(\d+)', blob).group(1))
    table = blob[start:].split(b'\n')
    count = int(table[1].split()[1])
    for number, line in enumerate(table[2:2 + count]):
        if line.endswith(b'f '):          # the free head
            continue
        offset = int(line.split()[0])
        assert blob[offset:offset + 16].startswith(f'{number} 0 obj'.encode())


def test_the_page_is_as_large_as_the_chart(tmp_path):
    c = _canvas()
    page = c.to_page()
    assert (page.width, page.height) == (c.W, c.H)


def test_a_dash_survives_the_encoding():
    """Latin-1 has no em dash and the encoder answers `?`, which turned
    every rule label into a question."""
    assert '--' in pdfout.PText(0, 0, 'ridge — below').to_pdf(100)
    assert '?' not in pdfout.PText(0, 0, 'ridge — below').to_pdf(100)


def test_a_long_note_is_wrapped_rather_than_run_off_the_sheet():
    note = 'word ' * 60
    short = plot.Canvas('t', 'x', 'y')
    tall = plot.Canvas('t', 'x', 'y', note=note)
    assert tall.H > short.H
    assert len(tall._note_lines) > 1


def test_a_table_is_a_page_and_a_table(tmp_path):
    page, markup = pdfout.table_page(
        'Machines', ['device', 'GB/s'], [['a', '100'], ['b', '200']],
        'a note')
    assert '<table' in markup and '<td>200</td>' in markup
    out = tmp_path / 't.pdf'
    pdfout.write_pdf([page], out)
    assert out.read_bytes().startswith(b'%PDF-1.')


def test_both_renderings_draw_the_same_things():
    """The SVG and the PDF come out of one list of primitives, so a mark
    that is in one is in the other; this is what keeps a view from being
    added to only half of the output."""
    c = _canvas()
    svg = c.render()
    page = c.to_page()
    assert svg.count('<circle') == sum(
        1 for p in page.prims if isinstance(p, pdfout.PCircle))
    assert svg.count('<polyline') == sum(
        1 for p in page.prims if isinstance(p, pdfout.PPoly))
