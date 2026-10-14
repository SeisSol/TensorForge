# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""`tools/mutation_check.py` against the guard in `tests/conftest.py`.

The guard refuses to collect on a tree a killed mutation run left broken, and
it took the harness's own runs for that: the lock is held while the tests
run.  Every mutation was refused at collection, a refusal exits nonzero as a
failure does, and so every one was reported caught, in a second each.  Held
here: the guard lets a live run through and still refuses a stale lock, the
harness marks the runs it makes, and a run that tested nothing is not a
catch.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

import conftest                                   # same directory as this file

ROOT = Path(__file__).resolve().parent.parent


def _harness():
    spec = importlib.util.spec_from_file_location(
        'mutation_check_under_test', ROOT / 'tools' / 'mutation_check.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_guard_refuses_a_stale_lock_and_lets_a_live_run_through(
        tmp_path, monkeypatch):
    lock = tmp_path / '.mutation-in-progress'
    lock.write_text('src/tensorforge/generators/tuning.py\n')
    monkeypatch.setattr(conftest, 'MUTATION_LOCK', lock)
    monkeypatch.delenv(conftest.MUTATION_RUN, raising=False)
    with pytest.raises(pytest.UsageError, match='deliberately broken'):
        conftest.pytest_collection(None)
    monkeypatch.setenv(conftest.MUTATION_RUN, '1')
    conftest.pytest_collection(None)


def test_the_harness_marks_the_runs_it_makes(tmp_path, monkeypatch):
    harness = _harness()
    seen = {}

    def run(cmd, capture_output, text, env):
        seen['env'] = env
        return SimpleNamespace(returncode=1, stdout='', stderr='')

    monkeypatch.setattr(harness.subprocess, 'run', run)
    # Away from the tree: the harness clears `src/tensorforge`'s bytecode
    # caches where it runs.
    monkeypatch.chdir(tmp_path)
    harness._run_tests('tests/test_nothing.py')
    assert seen['env'][conftest.MUTATION_RUN] == '1'


def test_a_run_that_tested_nothing_is_no_catch(tmp_path, monkeypatch):
    harness = _harness()
    assert harness.verdict(1) == harness.verdict(2) == 'caught'
    assert harness.verdict(0) == 'missed'
    for status in (3, 4, 5):
        assert harness.verdict(status).startswith('not run')

    victim = tmp_path / 'victim.py'
    victim.write_text('x = 1\n')
    monkeypatch.chdir(tmp_path)                   # where the lock is written
    statuses = iter([4, 1, 0])
    monkeypatch.setattr(harness, '_run_tests',
                        lambda target: SimpleNamespace(returncode=next(statuses)))
    mutations = [(f'mutation {i}', lambda: (victim, 'x = 2\n'))
                 for i in range(3)]
    assert harness.run('group', 'tests', mutations) == (1, 3)
    assert victim.read_text() == 'x = 1\n'
    assert not (tmp_path / '.mutation-in-progress').exists()
