# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""The lanes rule's cut twin (`tuning.FISSION_TWIN`): a candidate whose code
the compiler wrote past `generator.FISSION_PAST` instruction caches is built
cut into pieces as well, and taken where its count, charged per piece, is
the lower."""
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


class _Code:
  """A compiler that writes `code` bytes for every build, whole or cut."""
  def __init__(self, code):
    self.code = code

  def resources(self, result):
    return Resources(registers=128, spill_bytes=0, code_bytes=self.code)


def _pick(monkeypatch, code, cut_count, **asked):
  monkeypatch.setattr(tuning, '_settled', lambda result: None)
  monkeypatch.setattr(tuning, 'executed_instructions', lambda result: (
      cut_count if result.candidate.get('fission') else 1000.0))
  factory = _chain()
  context = Context(arch='sm_90', backend='cuda', fp_type=Datatype.F32,
                    options=Options(**asked))
  origin = tuning.start(factory(), context)
  first = tuning.build(factory, context, origin, Generator.trial)
  return tuning.lanes_rule(factory, context, _Code(code), origin, first,
                           generate=Generator.trial)


def test_a_body_past_the_cache_is_cut_where_that_counts_less(monkeypatch):
  # 480 kB whole: past three caches of 128 kB; three pieces of 160 kB each
  # charge a fifth of that per instruction.
  pick = _pick(monkeypatch, 480 * 1024, 1100.0)
  assert pick.get('fission') == tuning_pieces(480 * 1024)


def test_a_cut_that_counts_far_more_is_not_taken(monkeypatch):
  assert not _pick(monkeypatch, 480 * 1024, 3000.0).get('fission')


def test_a_cut_the_caller_asked_against_stands(monkeypatch):
  assert not _pick(monkeypatch, 480 * 1024, 1100.0, fission=0).get('fission')


def test_a_body_within_the_threshold_is_not_cut(monkeypatch):
  assert not _pick(monkeypatch, 150 * 1024, 900.0).get('fission')


def tuning_pieces(code):
  from tensorforge.generators.generator import fission_pieces
  from tensorforge.common.context import Context as _C
  hw = _C(arch='sm_90', backend='cuda', fp_type=Datatype.F32).target.hw
  return fission_pieces(code, hw)
