# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""`Options.stage_operands`: a multiplication's sources copied into shared
memory instead of loaded into registers.

`all` stages every multiplication, `stream` only one that sums over no axis --
every source number read once into the one output number it adds to, where
the copy's sixteen bytes a lane are all there is to win.  A contraction keeps
its register image under `stream`: it reads each number more than once, and
staged it reads them back from shared memory every time.
"""
from __future__ import annotations

import contextlib
import io
import re

from tensorforge.backend.instructions.builders.multilinear_builder import _contracts
from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.context import Context, Options
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators.descriptions import MultilinearDescr
from tensorforge.generators.generator import Generator

M = 16


def tensor(shape, alias):
  return Tensor(list(shape), Addressing.STRIDED,
                BoundingBox([0] * len(shape), list(shape)), alias=alias,
                datatype=Datatype.F32)


def product():
  """`C[i,j] = X[i,l] K[l,j]`."""
  x, k, c = tensor((M, M), 'X'), tensor((M, M), 'K'), tensor((M, M), 'C')
  return MultilinearDescr(SubTensor(c), [SubTensor(x), SubTensor(k)],
                          [[0, -1], [-1, 1]], [[0, 1], [0, 1]])


def stream():
  """`C[i,j] += X[i,j]`."""
  x, c = tensor((M, M), 'X'), tensor((M, M), 'C')
  return MultilinearDescr(SubTensor(c), [SubTensor(x)], [[0, 1]], [[0, 1]],
                          add=True)


def chain():
  """`T[i,j] = X[i,l] K[l,j]`, then `C[i,j] += T[i,j]`: T is global, and the
  product stores it from the lanes of its rows."""
  x, k = tensor((M, M), 'X'), tensor((M, M), 'K')
  t, c = tensor((M, M), 'T'), tensor((M, M), 'C')
  return [MultilinearDescr(SubTensor(t), [SubTensor(x), SubTensor(k)],
                           [[0, -1], [-1, 1]], [[0, 1], [0, 1]]),
          MultilinearDescr(SubTensor(c), [SubTensor(t)], [[0, 1]], [[0, 1]],
                           add=True)]


def shared_input():
  """`C[i,j] += X[i,j]`, then `D[i,j] = X[i,l] K[l,j]`: X is read twice."""
  x, k = tensor((M, M), 'X'), tensor((M, M), 'K')
  c, d = tensor((M, M), 'C'), tensor((M, M), 'D')
  return [MultilinearDescr(SubTensor(c), [SubTensor(x)], [[0, 1]], [[0, 1]],
                           add=True),
          MultilinearDescr(SubTensor(d), [SubTensor(x), SubTensor(k)],
                           [[0, -1], [-1, 1]], [[0, 1], [0, 1]])]


def body(descr, mode):
  context = Context(arch='sm_90', backend='cuda', fp_type=Datatype.F32,
                    options=Options(stage_operands=mode))
  generator = Generator(descr if isinstance(descr, list) else [descr], context)
  with contextlib.redirect_stdout(io.StringIO()):
    generator.generate()
  text = re.sub(r'kernel_kernel_[0-9a-f]+', 'K', generator.get_kernel())
  return re.sub(r'// options: .*', '', text)


def test_a_summed_axis_is_a_negative_target():
  assert _contracts(product())
  assert not _contracts(stream())


def test_stream_stages_what_sums_over_nothing():
  assert body(stream(), 'stream') == body(stream(), 'all')
  assert body(stream(), 'stream') != body(stream(), 'off')


def test_stream_leaves_a_contraction_in_registers():
  assert body(product(), 'stream') == body(product(), 'off')
  assert body(product(), 'all') != body(product(), 'off')


def test_what_the_kernel_stored_is_not_staged():
  # Staged, `C += T` would copy sixteen bytes a lane -- numbers the product
  # stored from other lanes, with nothing ordering those stores ahead.
  assert body(chain(), 'stream') == body(chain(), 'off')


def test_what_another_operation_reads_is_not_staged():
  # Staged, the copy would be what the product finds for X: staging rebinds
  # the tensor to it, and a merged run's table reads it as a global pointer.
  assert body(shared_input(), 'stream') == body(shared_input(), 'off')
