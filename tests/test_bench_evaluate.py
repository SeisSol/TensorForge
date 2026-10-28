# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""What `tools/bench/evaluate.py` runs, without running it.

Every step is a subprocess, so the steps are recorded instead of executed and
the plan is checked: that one command yields what the publication reads --
the ramp-up sweep where the launch cost is fitted, the matrix units in the
same run where the target reads the option, each kernel whole and in pieces,
the whole configuration built once, and the publication at the end.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

BENCH = Path(__file__).resolve().parent.parent / 'tools' / 'bench'
if str(BENCH) not in sys.path:
    sys.path.insert(0, str(BENCH))

import evaluate  # noqa: E402
from publish import RAMPUP_BATCHES  # noqa: E402

SWEEP = ','.join(map(str, RAMPUP_BATCHES))


def _plan(monkeypatch, tmp_path, *argv):
    """The steps `evaluate.py` takes for two exports, one of a single
    simulation and one of eight, by name: the command each would run."""
    exports = tmp_path / 'exports'
    exports.mkdir()
    for tag in ('elastic-4-d-s1', 'elastic-4-d-s8'):
        (exports / f'{tag}.json').write_text('{}')
    plan = {}

    def record(name, command, *rest):
        plan[name] = command
        return evaluate.Step(name, command, 'ok')

    monkeypatch.setattr(evaluate, 'run_step', record)
    monkeypatch.setattr(evaluate, 'run_with_env',
                        lambda name, command, env, *rest: record(name,
                                                                 command))
    monkeypatch.setattr(sys, 'argv', ['evaluate.py', '--out',
                                      str(tmp_path / 'eval'), '--exports',
                                      str(exports), *argv])
    assert evaluate.main() == 0
    return plan


def _flag(command, flag):
    return command[command.index(flag) + 1] if flag in command else None


def test_one_command_plans_everything_the_publication_reads(monkeypatch,
                                                             tmp_path):
    plan = _plan(monkeypatch, tmp_path, '--arch', 'sm_90')
    one, eight = plan['run elastic-4-d-s1'], plan['run elastic-4-d-s8']
    # The sweep where the launch cost is fitted, the suite's batches where
    # only the largest is read.
    assert _flag(one, '--batches') == SWEEP and '--batches' not in eight
    # The matrix units beside the suite's configurations, in the same run.
    assert _flag(one, '--configs') == '+tensor-cores'
    # Whole and in pieces for one simulation only.
    assert 'seissolsplit.py' in plan['split elastic-4-d-s1'][2]
    assert 'split elastic-4-d-s8' not in plan
    manifest = json.loads((tmp_path / 'eval' / 'evaluation.json').read_text())
    skipped, = [s for s in manifest['steps'] if s['status'] == 'skipped']
    assert skipped['step'] == 'split elastic-4-d-s8'
    # The whole configuration built, nothing run.
    build = plan['buildcost elastic-4-d-s1']
    assert build[2].endswith('seissol.py') and '--build-only' in build
    assert _flag(build, '--configs') == 'baseline'
    # And the publication over the whole tree, the comparison over runs.
    assert plan['publish'][2] == str(tmp_path / 'eval')
    assert str(tmp_path / 'eval' / 'split') not in plan['compare']
    assert all(_flag(c, '--arch') == 'sm_90' for name, c in plan.items()
               if name.startswith(('run', 'split', 'buildcost')) and c)


def test_amd_is_timed_in_the_suites_configurations_alone(monkeypatch,
                                                         tmp_path):
    plan = _plan(monkeypatch, tmp_path, '--backend', 'hip', '--arch',
                 'gfx90a')
    assert '--configs' not in plan['run elastic-4-d-s1']


@pytest.mark.parametrize('backend,arch,expected', [
    ('cuda', None, True), ('oneapi', None, False), ('esimd', 'pvc', True),
    ('hip', None, False), (None, 'sm_80', True), (None, 'gfx942', False)])
def test_which_targets_read_the_matrix_option(backend, arch, expected):
    assert evaluate.matrix_paths(backend, arch) is expected


def test_asked_for_is_what_is_run(monkeypatch, tmp_path):
    plan = _plan(monkeypatch, tmp_path, '--arch', 'sm_90', '--configs',
                 'baseline', '--batches', '4096', '--no-split',
                 '--no-buildcost', '--no-publish',
                 '--seissol', 'runs/')
    run = plan['run elastic-4-d-s1']
    assert (_flag(run, '--configs'), _flag(run, '--batches')) == (
        'baseline', '4096')
    assert not [n for n in plan if n.startswith(('split', 'buildcost',
                                                 'publish'))]


def test_publish_takes_the_seissol_runs_and_clusterings(monkeypatch,
                                                         tmp_path):
    plan = _plan(monkeypatch, tmp_path, '--arch', 'gfx942', '--seissol',
                 'runs/', '--clustering', 'synthetic')
    command = plan['publish']
    assert (_flag(command, '--seissol'), _flag(command, '--clustering')) == (
        'runs', 'synthetic')
