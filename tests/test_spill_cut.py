# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""The lanes rule's cut twin on Intel (`tuning.FISSION_REF_SHARE`): IGC
states no code size, so a build it spills is cut in two and the cut taken
where IGC's listing references what it spilled far less often, for little
more to execute."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from tensorforge.common.basic_types import Datatype
from tensorforge.common.context import Context, Options
from tensorforge.generators import tuning
from tensorforge.generators.generator import Generator
from tensorforge.toolchain import Resources


def _chain():
    path = Path(__file__).parent / "cases" / "fission_chain.py"
    spec = importlib.util.spec_from_file_location("fission_chain", path)
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


class _IGC:
    """A compiler that says nothing of its code, spills a build whole `spill`
    bytes and lists `whole` references to them, and spills it cut in
    proportion to the `cut` references it lists then."""

    def __init__(self, spill, whole, cut):
        self.spill, self.whole, self.cut = spill, whole, cut
        self.asked = []
        self.code = 1000

    def resources(self, result):
        cut = result.candidate.get('fission')
        spill = (self.spill * self.cut // self.whole if cut and self.whole
                 else self.spill)
        # Two multiplications of eight lanes in one sub-group spill more than
        # one of sixteen, as IGC does: the rule weighs the cut at sixteen.
        if result.generator.lanes.num_threads < 16:
            spill *= 4
        return Resources(registers=256, spill_bytes=spill)

    def spill_references(self, result):
        self.asked.append(result.candidate)
        return self.cut if result.candidate.get('fission') else self.whole

    def listed_instructions(self, result):
        return self.code


class _Large(_IGC):
    """The same, with `code` instructions listed for every build."""

    def __init__(self, spill, whole, cut, code):
        super().__init__(spill, whole, cut)
        self.code = code


def _pick(monkeypatch, scorer, cut_count, **asked):
    monkeypatch.setattr(tuning, '_settled', lambda result: None)
    monkeypatch.setattr(tuning, 'executed_instructions', lambda result: (
        cut_count if result.candidate.get('fission') else 1000.0))
    factory = _chain()
    context = Context(arch='pvc', backend='oneapi', fp_type=Datatype.F64,
                      options=Options(**asked))
    origin = tuning.start(factory(), context)
    first = tuning.build(factory, context, origin, Generator.trial)
    return tuning.lanes_rule(factory, context, scorer, origin, first,
                             generate=Generator.trial)


def test_a_cut_that_spills_far_less_often_is_taken(monkeypatch):
    pick = _pick(monkeypatch, _IGC(4096, 1000, 500), 1010.0)
    assert pick.get('fission') == 2


def test_a_cut_that_spills_about_as_often_is_not(monkeypatch):
    assert not _pick(monkeypatch, _IGC(4096, 1000, 800), 1010.0).get('fission')


def test_a_cut_that_executes_far_more_is_not(monkeypatch):
    assert not _pick(monkeypatch, _IGC(4096, 1000, 100), 1150.0).get('fission')


def test_a_cut_that_saves_many_references_an_instruction_may_execute_more(
        monkeypatch):
    """The order-7 `derivative` in single: 19192 references against 8426
    for 25652 instructions, 7.4 % more executed, 38 % faster."""
    pick = _pick(monkeypatch, _IGC(4096, 1000, 600), 1050.0)
    assert pick.get('fission') == 2


def test_a_cut_that_saves_few_may_not(monkeypatch):
    assert not _pick(monkeypatch, _IGC(4096, 1000, 750), 1050.0).get('fission')


def test_a_build_that_spills_nothing_is_not_cut(monkeypatch):
    scorer = _IGC(0, 0, 0)
    assert not _pick(monkeypatch, scorer, 900.0).get('fission')
    assert not any(c.get('fission') for c in scorer.asked)


def test_a_cut_the_caller_asked_against_stands(monkeypatch):
    pick = _pick(monkeypatch, _IGC(4096, 1000, 100), 1000.0, fission=0)
    assert not pick.get('fission')


def test_a_large_kernel_is_cut_where_its_references_barely_move(monkeypatch):
    """The order-8 `derivative` in double: 95188 instructions, 0.92 of the
    references cut, 0.9 % more executed, 35 % faster."""
    pick = _pick(monkeypatch, _Large(4096, 1000, 920, 95188), 1050.0)
    assert pick.get('fission') == 2


def test_a_small_one_is_not(monkeypatch):
    assert not _pick(monkeypatch, _Large(4096, 1000, 920, 9000),
                     1050.0).get('fission')


def test_the_listing_counts_its_instructions(tmp_path):
    from tensorforge import toolchain
    (tmp_path / 'a.asm').write_text('//.instCount 95188\n//.spill size 88704\n')
    (tmp_path / 'b.asm').write_text('//.instCount 12\n')
    assert toolchain.igc_instructions(str(tmp_path)) == 95188
    assert toolchain.igc_instructions(str(tmp_path / 'missing')) is None
