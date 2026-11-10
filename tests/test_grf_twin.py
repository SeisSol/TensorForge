# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""The lanes rule's large-register-file twin (`tuning.GRF_TWIN`): on Intel
under DPC++, a candidate IGC spills is built again asking for 256 general
registers a thread (`Options.sycl_grf`), and taken where IGC spills that one
less."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from tensorforge.common.basic_types import Datatype
from tensorforge.common.context import Context, Options
from tensorforge.generators import tuning
from tensorforge.generators.generator import Generator
from tensorforge.toolchain import Resources


def _case():
    path = Path(__file__).parent / "cases" / "addressing_none.py"
    spec = importlib.util.spec_from_file_location("grf_case", path)
    case = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(case)
    return case.descr_list


@pytest.fixture(autouse=True)
def _forget_picks():
    tuning._PICKS.clear()
    tuning._LOADED.clear()
    yield
    tuning._PICKS.clear()
    tuning._LOADED.clear()


class _Igc:
    """A compiler that spills `small` bytes of every build at the default
    register file and `large` at 256."""

    def __init__(self, small, large):
        self.small, self.large = small, large

    def resources(self, result):
        big = str(result.candidate.get('sycl_grf', 'default')) == '256'
        return Resources(registers=None,
                         spill_bytes=self.large if big else self.small,
                         code_bytes=4096)


def _pick(monkeypatch, small, large, arch='pvc', backend='oneapi', **asked):
    monkeypatch.setattr(tuning, '_settled', lambda result: None)
    monkeypatch.setattr(tuning, 'executed_instructions',
                        lambda result: 1000.0)
    factory = _case()
    context = Context(arch=arch, backend=backend, fp_type=Datatype.F32,
                      options=Options(**asked))
    origin = tuning.start(factory(), context)
    first = tuning.build(factory, context, origin, Generator.trial)
    return tuning.lanes_rule(factory, context, _Igc(small, large), origin,
                             first, generate=Generator.trial)


def test_a_kernel_igc_spills_takes_the_large_file_where_it_spills_less(
        monkeypatch):
    assert str(_pick(monkeypatch, 4096, 512).get('sycl_grf')) == '256'


def test_a_kernel_that_fits_keeps_the_default(monkeypatch):
    assert _pick(monkeypatch, 0, 0).get('sycl_grf') is None


def test_no_gain_in_the_spilling_is_no_reason(monkeypatch):
    assert _pick(monkeypatch, 4096, 4096).get('sycl_grf') is None


def test_a_file_the_caller_asked_for_stands(monkeypatch):
    pick = _pick(monkeypatch, 4096, 512, sycl_grf='128')
    assert str(pick.get('sycl_grf', '128')) == '128'


def test_not_off_intel(monkeypatch):
    assert _pick(monkeypatch, 4096, 512, arch='sm_90',
                 backend='cuda').get('sycl_grf') is None
