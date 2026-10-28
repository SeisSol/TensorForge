# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""A kernel is its name *and* the export it was captured from.

Over the sixteen exports in `tools/bench/data`, the compulsory intensity
under one name -- `gpu_localFluxAll` -- spans 3.6 to 38 flop/byte, so a
reader that keys on the name alone keys on whichever export came first.
What is pinned here is that the identity travels: from the export's
`config` into the workload, from the directory name where a row predates
that, and into the points the readers draw.

Host-only: no GPU, no toolchain.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / 'tools' / 'bench'
if str(BENCH) not in sys.path:
    sys.path.insert(0, str(BENCH))

import capture  # noqa: E402
import compare  # noqa: E402

DATA = BENCH / 'data'


def test_the_tag_is_the_name_evaluate_gives_an_export():
    """The readers place a run by its directory where its rows do not say,
    so the two spellings have to be one."""
    import evaluate
    for entry in evaluate.EXPORT_MATRIX:
        name = evaluate.export_name(*entry)
        assert capture.Capture.from_tag(name).tag == name


@pytest.mark.parametrize('tag,expected', [
    ('elastic-6-s-s1', ('elastic', 6, 's', 1)),
    ('viscoelastic-3-d-s8', ('viscoelastic', 3, 'd', 8)),
    ('elastic-linearck-o8-s', None),     # a recorded configuration, no export
    ('sm_120', None),
    ('runs', None),
])
def test_a_tag_is_read_or_refused(tag, expected):
    got = capture.Capture.from_tag(tag)
    if expected is None:
        assert got is None
    else:
        assert (got.equation, got.order, got.precision,
                got.simulations) == expected


def test_every_vendored_export_says_what_it_is():
    """And says it the way its file is named, which is what lets a run of
    it be placed by its directory."""
    for path in sorted(DATA.glob('*.json')):
        blob = json.loads(path.read_text())
        got = capture.Capture.from_config(blob.get('config'))
        assert got is not None, path.name
        assert got.tag == path.stem


def test_a_row_carries_the_fields_flat():
    c = capture.Capture('viscoelastic', 4, 'd', 1, 3, 'linearckanelastic')
    row = dict(c.fields(), workload='gpu_derivative', datatype='F64')
    assert capture.Capture.from_row(row) == c
    assert row['capture'] == 'viscoelastic-4-d-s1'


def test_the_rows_are_believed_over_the_directory():
    rows = [capture.Capture('elastic', 4, 'd').fields()]
    got = capture.capture_of(Path('/eval/runs/elastic-6-s-s1'), rows)
    assert got.order == 4 and got.precision == 'd'


def test_a_directory_is_read_from_the_innermost_part():
    got = capture.capture_of(
        Path('/gb200/runs/elastic-6-s-s1/sub'), [])
    assert got.tag == 'elastic-6-s-s1'


def test_a_workload_from_an_export_knows_its_capture():
    import suite
    path = DATA / 'viscoelastic-3-s-s1.json'
    workloads = suite.from_dump(path, pattern='gpu_derivative')
    assert len(workloads) == 1
    got = workloads[0].capture
    assert got.tag == 'viscoelastic-3-s-s1'
    assert got.mechanisms == 3


# ----------------------------------------------------------------------
# Into the points
# ----------------------------------------------------------------------

def _row(workload, **extra):
    r = {'workload': workload, 'config': 'baseline', 'arch': 'sm_90',
         'backend': 'cuda', 'datatype': 'F32', 'batch': 1024,
         'flops': 1e9, 'bytes': 1e8, 'event_ns': 1e5}
    r.update(extra)
    return r


def _write(tmp_path, name, rows):
    d = tmp_path / name
    d.mkdir(parents=True)
    (d / 'bench.json').write_text(json.dumps({'results': rows}))
    return d


def test_one_name_in_two_exports_is_two_kernels(tmp_path):
    _write(tmp_path, 'elastic-3-s-s1', [_row('gpu_volume')])
    _write(tmp_path, 'elastic-6-s-s1', [_row('gpu_volume')])
    got = compare.collect([tmp_path])
    assert sorted(p.kernel for p in got.points) == [
        'gpu_volume @ elastic-3-s-s1', 'gpu_volume @ elastic-6-s-s1']


def test_a_capture_the_row_contradicts_is_dropped(tmp_path):
    """A directory named for single precision holding a double row is not
    evidence of which export the row is; the row is the record."""
    _write(tmp_path, 'elastic-6-s-s1', [_row('k', datatype='F64')])
    point = compare.collect([tmp_path]).points[0]
    assert point.capture is None
    assert point.order == 6
