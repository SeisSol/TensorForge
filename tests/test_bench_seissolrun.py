# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Reading what a SeisSol run says about itself (`tools/bench/seissolrun.py`).

The files are SeisSol's own -- `<prefix>-summary.json` and
`<prefix>-clustering.csv` -- so what is checked is that they are read as
SeisSol writes them, and that the model of a clustered run counts what it
says it counts: cell updates for the ideal, a launch per layer and step for
the prediction, the slowest rank for both.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

BENCH = Path(__file__).resolve().parents[1] / 'tools' / 'bench'
if str(BENCH) not in sys.path:
    sys.path.insert(0, str(BENCH))

import seissolrun  # noqa: E402

HEADER = ('"profilingId","localId","layerType","size","dynamicRuptureSize",'
          '"rank","localRank","timestep","updateFactor","runId"\n')


def _clustering(tmp_path, rows, name='run-clustering.csv', run_id='abc'):
    lines = [f'{i},{cluster},"{kind}",{cells},0,{rank},0,{dt},2,"{run_id}"'
             for i, (rank, cluster, kind, cells, dt) in enumerate(rows)]
    path = tmp_path / name
    path.write_text(HEADER + '\n'.join(lines) + '\n')
    return path


def test_a_configuration_name_is_the_export_it_corresponds_to():
    elastic = seissolrun.capture_of_config('elastic-linearck-o4-f64-stroud')
    assert (elastic.tag, elastic.solver) == ('elastic-4-d-s1', 'linearck')
    visco = seissolrun.capture_of_config(
        'viscoelastic-linearckanelastic-m3-o6-f32-dunavant-s8')
    assert visco.tag == 'viscoelastic-6-s-s8' and visco.mechanisms == 3
    assert seissolrun.capture_of_config('nonsense') is None


def test_the_clustering_is_read_as_seissol_writes_it(tmp_path):
    path = _clustering(tmp_path, [(0, 0, 'Interior', 100, 1.0),
                                  (0, 1, 'Interior', 300, 2.0)])
    clustering = seissolrun.read_clustering(path)
    assert clustering.name == 'run' and not clustering.synthetic
    assert [(l.cluster, l.cells, l.timestep) for l in clustering.layers] == [
        (0, 100, 1.0), (1, 300, 2.0)]
    # 400 cells at the smallest timestep against 100 + 300 / 2 updates.
    assert clustering.ideal_speedup() == pytest.approx(400 / 250)


def test_the_synthetic_series_says_it_is_one():
    clustering = seissolrun.read_clustering(
        BENCH / 'clusterings' / 'synthetic-clustering.csv')
    assert clustering.synthetic and clustering.ranks == [0, 1, 2, 3]
    assert clustering.ideal_speedup() > 1


def test_every_layer_is_a_launch_and_the_slowest_rank_decides(tmp_path):
    """Per rank: local time stepping launches each layer once per step of
    its cluster, global time stepping its interior and its copy cells
    once per smallest step."""
    path = _clustering(tmp_path, [(0, 0, 'Interior', 100, 1.0),
                                  (0, 1, 'Interior', 300, 2.0),
                                  (1, 0, 'Copy', 10, 1.0),
                                  (1, 0, 'Interior', 50, 1.0),
                                  (1, 1, 'Interior', 50, 2.0)])
    clustering = seissolrun.read_clustering(path)
    t0, c = 1000.0, 2.0
    lts = clustering.seconds(t0, c)
    gts = clustering.seconds(t0, c, gts=True)
    assert lts[0] == pytest.approx(((t0 + c * 100) + (t0 + c * 300) / 2)
                                   * 1e-9)
    assert lts[1] == pytest.approx(((t0 + c * 10) + (t0 + c * 50)
                                    + (t0 + c * 50) / 2) * 1e-9)
    assert gts[0] == pytest.approx((t0 + c * 400) * 1e-9)
    assert gts[1] == pytest.approx(((t0 + c * 10) + (t0 + c * 100)) * 1e-9)
    assert clustering.predicted_speedup(t0, c) == pytest.approx(
        max(gts.values()) / max(lts.values()))
    # Free launches leave the cell updates: the ideal, on one rank.
    one = seissolrun.Clustering('one', [l for l in clustering.layers
                                        if l.rank == 0])
    assert one.predicted_speedup(0.0, c) == pytest.approx(
        one.ideal_speedup())


def test_a_layer_type_seissol_does_not_write_is_refused(tmp_path):
    path = _clustering(tmp_path, [(0, 0, 'Ghost', 1, 1.0)])
    with pytest.raises(ValueError, match='neither interior nor copy'):
        seissolrun.read_clustering(path)


def _summary(tmp_path, name, *, clusters=3, arch='sm_90', seconds=100.0,
             status='completed', end=2.0, cells=1000.0, ranks=2,
             config='elastic-linearck-o4-f64-stroud'):
    blob = {
        'schema': seissolrun.SCHEMA, 'status': status,
        'build': {'deviceArch': arch, 'deviceBackend': 'cuda'},
        'configurations': [config],
        'parallel': {'ranks': ranks},
        'mesh': {'cells': {'sum': cells, 'mean': cells / ranks}},
        'simulation': {'startTime': 0.0, 'endTime': end,
                       'cellUpdates': 1e9},
        'time': {'simulation': {
            'total': {'max': seconds, 'mean': seconds * 0.9},
            'io': {'mean': 4.0}, 'outsideKernels': {'mean': 6.0}}},
        'kernels': {'regions': {
            'computeLocalIntegration': {
                'time': {'mean': 50.0},
                'regression': {'constant': 2e-5, 'perElement': 1e-8}},
            'computeNeighboringIntegration': {
                'time': {'mean': 20.0},
                'regression': {'constant': 3e-5, 'perElement': 4e-9}}}},
        'lts': {'theoreticalSpeedup': {'clustered': 4.0},
                'clusters': [{'id': i, 'timestep': 2.0 ** i,
                              'cells': cells / clusters}
                             for i in range(clusters)]},
    }
    path = tmp_path / f'{name}-summary.json'
    path.write_text(json.dumps(blob))
    return path


def test_a_summary_says_what_ran_where_and_how_long(tmp_path):
    run = seissolrun.read_summary(_summary(tmp_path, 'a'))
    assert (run.arch, run.capture.tag, run.ranks) == ('sm_90',
                                                     'elastic-4-d-s1', 2)
    assert run.lts and run.clusters == 3 and run.ideal_speedup == 4.0
    assert run.updates_per_second_per_rank == pytest.approx(1e9 / 100 / 2)
    assert run.split['computeLocalIntegration'] == 50.0
    assert run.split['total'] == pytest.approx(90.0)
    # SeisSol's own fit of the regions, as an n-half of a time step.
    assert run.half() == pytest.approx(5e-5 / 1.4e-8)
    # No clustering file beside it: the summary's clusters, spread evenly.
    assert run.clustering.ranks == [0, 1]
    assert {l.cells for l in run.clustering.layers} == {167}


def test_the_clustering_beside_a_summary_is_the_one_read(tmp_path):
    _clustering(tmp_path, [(0, 0, 'Interior', 7, 1.0)],
                name='a-clustering.csv')
    run = seissolrun.read_summary(_summary(tmp_path, 'a'))
    assert [l.cells for l in run.clustering.layers] == [7]


def test_a_run_that_did_not_finish_or_another_file_is_no_run(tmp_path):
    assert seissolrun.read_summary(
        _summary(tmp_path, 'b', status='running')) is None
    other = tmp_path / 'c-summary.json'
    other.write_text(json.dumps({'schema': 'something/1'}))
    assert seissolrun.read_summary(other) is None
    assert [r.name for r in seissolrun.collect_runs([tmp_path])] == []


def test_lts_against_gts_needs_both_of_one_mesh(tmp_path):
    _summary(tmp_path, 'gts', clusters=1, seconds=400.0, end=1.0)
    _summary(tmp_path, 'lts', clusters=3, seconds=150.0, end=1.5)
    _summary(tmp_path, 'other', clusters=3, cells=999.0)
    runs = seissolrun.collect_runs([tmp_path])
    assert sorted(r.name for r in runs) == ['gts', 'lts', 'other']
    speedups = seissolrun.measured_speedups(runs)
    assert speedups == {('sm_90', 'elastic-linearck-o4-f64-stroud', 2,
                         1000): pytest.approx(400.0 / (150.0 / 1.5))}
