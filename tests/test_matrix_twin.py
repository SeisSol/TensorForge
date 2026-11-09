# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""The matrix path as a twin of the default in the lanes rule
(`tuning._matrix_candidates`): offered where the part's matrix units outrun
its vector units in the kernel's precision (`tuning.MATRIX_TWIN`), and taken
where the rule's count says it is faster, as every other twin is."""
from __future__ import annotations

import pytest

from tensorforge.common.basic_types import Datatype
from tensorforge.common.context import Context, Options
from tensorforge.generators import tuning
from tensorforge.generators.generator import Generator
from tensorforge.toolchain import Resources

from test_lanes_rule import _local_flux


@pytest.fixture(autouse=True)
def _forget_picks():
  tuning._PICKS.clear()
  tuning._LOADED.clear()
  yield
  tuning._PICKS.clear()
  tuning._LOADED.clear()


def _offered(arch, dtype, **opts):
  descr_factory = _local_flux()
  context = Context(arch=arch, backend='cuda', fp_type=dtype,
                    options=Options(**opts))
  origin = tuning.start(descr_factory(), context)
  first = tuning.build(descr_factory, context, origin, Generator.trial)
  assert first.ok
  return tuning._matrix_candidates(descr_factory(), first, context, origin,
                                   None)


@pytest.mark.parametrize('arch, dtype, chunked', [
    ('sm_80', Datatype.F64, True), ('sm_80', Datatype.F32, None),
    ('sm_90', Datatype.F64, True), ('sm_90', Datatype.F32, None),
    ('sm_100', Datatype.F32, None)])
def test_offered_where_the_matrix_units_are_faster(arch, dtype, chunked):
  twins = _offered(arch, dtype)
  assert len(twins) == 1
  assert twins[0].get('tensor_cores') is True
  assert twins[0].get('mma_chunked_order') is chunked


@pytest.mark.parametrize('arch, dtype', [
    ('sm_100', Datatype.F64), ('sm_120', Datatype.F32), ('sm_120', Datatype.F64),
    ('sm_86', Datatype.F64), ('sm_89', Datatype.F32)])
def test_not_offered_where_they_are_not(arch, dtype):
  assert _offered(arch, dtype) == []


def test_a_path_the_caller_asked_for_stands():
  assert _offered('sm_90', Datatype.F64, tensor_cores=False) == []
  assert _offered('sm_90', Datatype.F64, tensor_cores=True) == []


class _Clean:
  """A compiler that reports the same clean build for every candidate."""
  def resources(self, result):
    return Resources(registers=128, spill_bytes=0)


@pytest.mark.parametrize('matrix_count, taken', [(1000.0, True),
                                                 (3000.0, False)])
def test_taken_where_the_count_says_it_is_faster(monkeypatch, matrix_count,
                                                 taken):
  monkeypatch.setattr(tuning, '_settled', lambda result: None)
  monkeypatch.setattr(tuning, 'executed_instructions', lambda result: (
      matrix_count if result.candidate.get('tensor_cores') else 2000.0))
  descr_factory = _local_flux()
  context = Context(arch='sm_90', backend='cuda', fp_type=Datatype.F64,
                    options=Options())
  origin = tuning.start(descr_factory(), context)
  first = tuning.build(descr_factory, context, origin, Generator.trial)
  pick = tuning.lanes_rule(descr_factory, context, _Clean(), origin, first,
                           generate=Generator.trial)
  assert bool(pick.get('tensor_cores')) is taken
