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


@pytest.mark.parametrize('arch, dtype, wide', [
    ('sm_80', Datatype.F64, True), ('sm_80', Datatype.F32, None),
    ('sm_90', Datatype.F64, True), ('sm_90', Datatype.F32, None),
    ('sm_100', Datatype.F32, None)])
def test_offered_where_the_matrix_units_are_faster(arch, dtype, wide):
  twins = _offered(arch, dtype)
  assert twins and all(t.get('tensor_cores') is True for t in twins)
  assert all(t.get('mma_operator_copies') is True for t in twins)
  # The double-precision layouts: A in 16-byte chunks, B's temporaries
  # padded for the fragment reads.
  assert all(t.get('mma_chunked_order') is wide for t in twins)
  assert all(t.get('mma_fragment_ld') is wide for t in twins)
  # Deep products only (`MATRIX_TWIN_DEPTH`).
  assert all(t.get('mma_min_depth') == tuning.MATRIX_TWIN_DEPTH for t in twins)
  # The first at the default unroll, the others rolled or written out.
  assert twins[0].get('k_unroll_max') is None
  assert {t.get('k_unroll_max') for t in twins[1:]} <= set(tuning.MATRIX_ROLLS)


def test_rolled_twins_only_where_a_reduction_is_long_enough():
  from tensorforge.generators.descriptions import contraction_lengths
  from tensorforge.generators.tuning import _flat
  longest = max(contraction_lengths(_flat(_local_flux()())))
  rolls = {t.get('k_unroll_max') for t in _offered('sm_90', Datatype.F64)[1:]}
  assert (16 in rolls) == (longest > 16)
  assert (0 in rolls) == (longest > 64)


def test_an_unroll_the_caller_asked_for_stands():
  twins = _offered('sm_90', Datatype.F64, k_unroll_max=16)
  assert len(twins) == 1 and twins[0].get('k_unroll_max') is None


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


def _matrix_issues(arch):
  from tensorforge.common.basic_types import Addressing
  from tensorforge.common.matrix.boundingbox import BoundingBox
  from tensorforge.common.matrix.tensor import SubTensor, Tensor
  from tensorforge.generators.descriptions import GemmDescr
  import contextlib
  import io

  def tensor(shape, addressing, alias):
    return Tensor(list(shape), addressing, BoundingBox([0, 0], list(shape)),
                  alias=alias, datatype=Datatype.F64)
  descrs = [GemmDescr(False, False,
                      SubTensor(tensor((32, 32), Addressing.NONE, 'A')),
                      SubTensor(tensor((32, 8), Addressing.STRIDED, 'B')),
                      SubTensor(tensor((32, 8), Addressing.STRIDED, 'C')),
                      alpha=1.0, beta=0.0)]
  gen = Generator(descrs, Context(arch=arch, backend='cuda',
                                  fp_type=Datatype.F64,
                                  options=Options(tensor_cores=True)))
  with contextlib.redirect_stdout(io.StringIO()):
    gen.generate()
  return gen.metrics.issue_mix['matrix'][0]


def test_an_fp64_matrix_instruction_is_eight_on_blackwell():
  """sm_100 lowers every `mma.sync` f64 shape to `DMMA.8x8x4`: the
  m16n8k16 Hopper issues once is eight there, and the count says so."""
  assert _matrix_issues('sm_100') == 8 * _matrix_issues('sm_90')


def test_a_depth_the_caller_asked_for_stands():
  twins = _offered('sm_90', Datatype.F64, mma_min_depth=16)
  assert twins and all(t.get('mma_min_depth') is None for t in twins)
