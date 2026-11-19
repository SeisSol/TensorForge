# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""The targets the controller discovers reach every xdist worker unchanged
(`conftest.TARGETS_ENV`), so that no worker probes -- and collects -- on its own."""

import conftest
from harness.toolchain import Target


def test_targets_survive_the_handover():
    targets = [Target('cuda', 'sm_120', 'nvidia', 0, 'NVIDIA GeForce RTX 5070'),
               Target('hip', 'gfx1150', 'amd', 0, 'AMD Radeon 890M')]
    assert conftest.targets_from_json(conftest.targets_to_json(targets)) == targets


def test_a_worker_takes_the_handed_targets_and_probes_nothing(monkeypatch):
    class Config:
        pass

    handed = [Target('cuda', 'sm_86', 'nvidia', 3, 'NVIDIA GeForce RTX 3080')]
    monkeypatch.setenv(conftest.TARGETS_ENV, conftest.targets_to_json(handed))

    def probed():
        raise AssertionError('a worker probed for GPUs')

    monkeypatch.setattr(conftest, 'detect_all', probed)
    assert conftest._discover_targets_session(Config()) == handed


def test_none_found_is_handed_down_as_none():
    assert conftest.targets_from_json(conftest.targets_to_json([])) == []
