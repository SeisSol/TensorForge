# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Copies of a batch-constant operator for its readers on NVIDIA's matrix path
(`Options.mma_operator_copies`, `generators.operator_copies`).

Which readers get one is decided from the descriptions; what a copy is in the
kernel -- a table of the file, in its reader's fragment order, named after
its numbers and passed by nobody -- is the generator's.  The numbers
themselves are checked on the GPU (`tests/cases/mma_operator_cop*`)."""
from __future__ import annotations

import contextlib
import io
import re

import numpy as np

from tensorforge.common.basic_types import Addressing, Datatype, Residence
from tensorforge.common.context import Context, Options
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators.descriptions import GemmDescr
from tensorforge.generators.generator import Generator
from tensorforge.generators.operator_copies import copy_operators

F64 = Datatype.F64
VALUES = np.random.default_rng(7).uniform(0.5, 1.5, size=(56, 56))


def _operator(alias='A'):
  return Tensor([56, 56], Addressing.NONE, BoundingBox([0, 0], [56, 56]),
                alias=alias, datatype=F64, data=VALUES.copy())


def _strided(alias):
  return Tensor([56, 9], Addressing.STRIDED, BoundingBox([0, 0], [56, 9]),
                alias=alias, datatype=F64)


def _product(a, rows, depth, dest, add=False, source=None):
  """`dest[0:rows] (+)= a[0:rows, 1:depth] source[1:depth]`."""
  return GemmDescr(False, False, SubTensor(a, BoundingBox([0, 1], [rows, depth])),
                   SubTensor(source or _strided('B'),
                             BoundingBox([1, 0], [depth, 9])),
                   SubTensor(dest, BoundingBox([0, 0], [rows, 9])),
                   alpha=1.0, beta=1.0 if add else 0.0)


def _derivative_like():
  """Three windows of one operator, as the derivative's orders 1, 2 and 4
  read `kDivMT`: two deep enough for the matrix path, one not."""
  a, d = _operator(), _strided('D')
  return a, [_product(a, 35, 54, d), _product(a, 20, 33, d, add=True),
             _product(a, 10, 9, d, add=True)]


def test_matrix_path_readers_get_copies_the_short_one_the_operator():
  a, descrs = _derivative_like()
  out, copies = copy_operators(descrs, 16)
  assert len(copies) == 2
  assert [d.ops[0].tensor is a for d in out] == [False, False, True]
  for clone in copies:
    assert clone.residence is Residence.MODULE
    assert clone.storage_order is None and clone.packing is None
    assert np.array_equal(clone.data, a.data)
  # The caller's descriptions are not touched.
  assert all(d.ops[0].tensor is a for d in descrs)


def test_one_window_two_readers_two_copies_one_table():
  """A copy read twice would be offered no order, so each reader has its
  own; with the same numbers in the same order they are one table."""
  a, d = _operator(), _strided('D')
  descrs = [_product(a, 35, 54, d), _product(a, 35, 54, d, add=True,
                                             source=_strided('E')),
            _product(a, 10, 9, d, add=True)]
  out, copies = copy_operators(descrs, 16)
  assert len(copies) == 2
  assert out[0].ops[0].tensor is not out[1].ops[0].tensor
  src = _generate(descrs, mma_operator_copies=True, mma_chunked_order=True)
  assert len(_tables(src)) == 1


def test_a_lone_reader_of_the_whole_operator_keeps_it_where_prepared():
  a, d = _operator(), _strided('D')
  whole = [GemmDescr(False, False, SubTensor(a), SubTensor(_strided('B')),
                     SubTensor(d), alpha=1.0, beta=0.0)]
  assert copy_operators(whole, 16, prepared=True)[1] == []
  assert len(copy_operators(whole, 16, prepared=False)[1]) == 1
  # A window narrower than the operator is ordered only where asked by name.
  assert len(copy_operators([_product(a, 35, 54, d)], 16,
                            prepared=True)[1]) == 1


def _generate(descrs, **opts):
  gen = Generator(descrs, Context(arch='sm_90', backend='cuda', fp_type=F64,
                                  options=Options(tensor_cores=True, **opts)))
  with contextlib.redirect_stdout(io.StringIO()):
    gen.generate()
  return gen.get_kernel()


def _tables(src):
  return re.findall(r'__device__ __align__\(16\) const double (tf_op_\w+)\[(\d+)\]',
                    src)


def test_a_copy_is_a_table_of_the_file_and_no_parameter():
  src = _generate(_derivative_like()[1], mma_operator_copies=True,
                  mma_chunked_order=True)
  tables = _tables(src)
  assert len(tables) == 2
  signature = re.search(r'kernel_\w+\(([^)]*)\)', src).group(1)
  assert 'tf_op_' not in signature and 'OPERATOR_COPY' not in signature
  # The placeholder names a binding at most, never the table it binds.
  assert not re.search(r'(?<!\w)TENSORFORGE_OPERATOR_COPY_\d+\[', src)
  for name, _ in tables:
    assert f'&{name}[0]' in src
  # The operator the short product reads is still passed.
  assert signature.count('const double *') == 4


def test_a_copy_is_named_after_its_numbers():
  one = dict(_tables(_generate(_derivative_like()[1], mma_operator_copies=True)))
  two = dict(_tables(_generate(_derivative_like()[1], mma_operator_copies=True)))
  assert one == two
  # Another order of the same numbers is another table.
  other = dict(_tables(_generate(_derivative_like()[1], mma_operator_copies=True,
                                 mma_chunked_order=True)))
  assert set(other).isdisjoint(one)


def test_off_by_default():
  assert _tables(_generate(_derivative_like()[1])) == []


def test_a_copy_is_not_staged_by_the_preload():
  """The preload stages batch-constant operators into shared memory once per
  block; a copy has one reader, which reads it a fragment at a time through
  the caches.  Staged, SeisSol's order-8 derivative took one multiplication
  a block at 225 kB and ran 5.4 times as long (GH200)."""
  _, descrs = _derivative_like()
  gen = Generator(descrs, Context(
      arch='sm_90', backend='cuda', fp_type=F64,
      options=Options(tensor_cores=True, mma_operator_copies=True,
                      preload_globals=True, preload_partial=True)))
  with contextlib.redirect_stdout(io.StringIO()):
    gen.generate()
  built = gen.built
  copies = [t for t in built._module_operands]
  assert copies
  assert not any(id(t) in built._preloaded for t in copies)
  # The operator the short product reads is still staged.
  assert built._preloaded
