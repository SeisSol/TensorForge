# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""The lanes rule's guard on split sub-groups (`tuning.SPLIT_REF_GROWTH`).

Where the default spills, eight lanes on PVC (`Options.lanes_split_floor`,
two multiplications in a sub-group of 16) may spill within `SPILL_GROWTH` of
it and still reference the spilled values far more often: IGC's listing
says so (`toolchain.igc_spill_references`), its scratch does not.  The
figures below are the order-6 `neighboringFlux` in double precision
(IGC 2.41.5), where eight lanes ran 2.3 times as long as sixteen.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from tensorforge import toolchain
from tensorforge.common.basic_types import Datatype
from tensorforge.common.context import Context, Options
from tensorforge.generators import tuning
from tensorforge.generators.generator import Generator
from tensorforge.toolchain import Resources


def _case():
    path = Path(__file__).parent / "cases" / "addressing_none.py"
    spec = importlib.util.spec_from_file_location("split_case", path)
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
    """Spill and spill references by lane count; the same at every register
    file, so the large-file twin never helps."""

    def __init__(self, spill, refs):
        self.spill, self.refs = spill, refs
        self.listed = []

    @staticmethod
    def _lanes(result):
        return result.generator.lanes.num_threads

    def resources(self, result):
        return Resources(registers=None,
                         spill_bytes=self.spill.get(self._lanes(result), 0),
                         code_bytes=4096)

    def spill_references(self, result):
        self.listed.append(self._lanes(result))
        return self.refs.get(self._lanes(result), 0)


def _pick(monkeypatch, spill, refs):
    monkeypatch.setattr(tuning, '_settled', lambda result: None)
    # Eight lanes count the fewest instructions an element, as they do for
    # the kernel above (12321 against 17789 at sixteen).
    monkeypatch.setattr(tuning, 'executed_instructions',
                        lambda result: 125.0 * result.generator.lanes.num_threads)
    factory = _case()
    context = Context(arch='pvc', backend='oneapi', fp_type=Datatype.F64,
                      options=Options(lanes_split_floor=True))
    origin = tuning.start(factory(), context)
    first = tuning.build(factory, context, origin, Generator.trial)
    igc = _Igc(spill, refs)
    pick = tuning.lanes_rule(factory, context, igc, origin, first,
                             generate=Generator.trial)
    return pick.lanes.num_threads if pick.lanes else None, igc.listed


SPILL = {16: 14592, 8: 20160, 32: 5760, 64: 4992}


def test_eight_lanes_that_reference_their_spill_far_more_are_out(monkeypatch):
    lanes, listed = _pick(monkeypatch, SPILL, {16: 672, 8: 5086})
    assert lanes != 8
    assert sorted(listed) == [8, 16]


def test_eight_lanes_that_reference_it_about_as_often_stay(monkeypatch):
    lanes, _ = _pick(monkeypatch, SPILL, {16: 672, 8: 1000})
    assert lanes == 8


def test_no_listing_where_the_default_spills_nothing(monkeypatch):
    lanes, listed = _pick(monkeypatch, {16: 0, 8: 0}, {16: 0, 8: 0})
    assert listed == []


def test_the_listing_is_read_for_its_header(tmp_path):
    (tmp_path / 'a').mkdir()
    (tmp_path / 'a' / 'OCL_asm1_simd16_entry_0001.asm').write_text(
        '//.kernel k\n//.spill size 20160\n//.spill GRF est. ref count 5086\n')
    (tmp_path / 'OCL_asmsm1_simd16_entry_0001.asm').write_text(
        '//.spill GRF est. ref count 5081\n')
    assert toolchain.igc_spill_references(str(tmp_path)) == 5086


def test_a_listing_without_spill_says_none_were_made(tmp_path):
    (tmp_path / 'k.asm').write_text('//.kernel k\n//.instCount 1121\n')
    assert toolchain.igc_spill_references(str(tmp_path)) == 0
    assert toolchain.igc_spill_references(str(tmp_path / 'missing')) is None
