# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Kernels with a lead dimension blocked wider than one element, against the
same kernel at width one, on the host oracle (`kernel_eval`).

At width `w` lane `l` holds rows `w*l .. w*l+w-1`, side by side, so a read of
them is one `w`-wide access and their arithmetic one vector.  Two readings of
that broke, each only where SeisSol's derivative put them together:

* an operand whose lead axis is not its contiguous one -- `kDivMT`, read
  transposed in place -- has a lane's rows `ld` apart, and one wide access at
  the first read the next elements of the other axis;
* a register image whose window starts inside a vector -- rows 1 to 18 at
  width two -- was stored to shared memory a whole vector a lane, the head
  lane's outside component before the buffer, over the neighbor's data.
"""
from __future__ import annotations

import contextlib
import io
from dataclasses import replace

import numpy as np
import pytest

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.context import Context, Options
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators import lanes
from tensorforge.generators.descriptions import GemmDescr
from tensorforge.generators.generator import Generator
from tensorforge.reference import kernel_eval

F = Datatype.F32
S, N0 = Addressing.STRIDED, Addressing.NONE


def tensor(shape, addressing, alias, lo=None, hi=None, data=None, tmp=False):
  t = Tensor(list(shape), addressing,
             BoundingBox(list(lo or [0] * len(shape)), list(hi or shape)),
             alias=alias, datatype=F, data=data)
  if tmp:
    t.is_tmp = True
  return t


def transposed():
  """`C = A^T B`, A batch-constant and read in place."""
  a = tensor((20, 10), N0, 'A',
             data=np.random.default_rng(3).standard_normal((20, 10)))
  b, c = tensor((20, 9), S, 'B'), tensor((10, 9), S, 'C')
  return [GemmDescr(True, False, SubTensor(a), SubTensor(b), SubTensor(c),
                    alpha=1.0, beta=0.0)]


def chain(lo):
  """`tmp[lo..18] = X[lo..18] S`, then `D = K[:, lo..18] tmp[lo..18]` -- a
  step of SeisSol's derivative, the temporary read along its own lead axis."""
  x = tensor((32, 9), S, 'X', lo=(lo, 0), hi=(18, 9))
  s = tensor((9, 9), S, 'S')
  tmp = tensor((32, 9), S, 'tmp', lo=(lo, 0), hi=(18, 9), tmp=True)
  k = tensor((20, 20), N0, 'K', lo=(0, lo), hi=(10, 18),
             data=np.random.default_rng(4).standard_normal((20, 20)))
  d = tensor((16, 9), S, 'D', hi=(10, 9))
  return [GemmDescr(False, False, SubTensor(x), SubTensor(s), SubTensor(tmp),
                    alpha=1.0, beta=0.0),
          GemmDescr(False, False, SubTensor(k), SubTensor(tmp), SubTensor(d),
                    alpha=1.0, beta=0.0)]


def memory(descrs, width, monkeypatch):
  deduce = lanes.deduce
  monkeypatch.setattr(lanes, 'deduce', lambda *a, **k: replace(
      deduce(*a, **k), lead_width=width))
  gen = Generator(descrs, Context(arch='sm_90', backend='cuda', fp_type=F,
                                  options=Options()))
  with contextlib.redirect_stdout(io.StringIO()):
    gen.generate()
  threads, mults = kernel_eval.launch_geometry(gen.get_launcher())
  monkeypatch.setattr(lanes, 'deduce', deduce)
  return kernel_eval.evaluate_wave(gen.get_kernel(), threads, seed=5,
                                   globals_only=True, mults=mults)


@pytest.mark.parametrize('descrs', [transposed, lambda: chain(1),
                                    lambda: chain(0)],
                         ids=['transposed', 'shifted_chain', 'aligned_chain'])
def test_width_two_computes_what_width_one_does(descrs, monkeypatch):
  one = memory(descrs(), 1, monkeypatch)
  two = memory(descrs(), 2, monkeypatch)
  wrong = [key for key, value in one.items()
           if key not in two or not np.isclose(value, two[key], rtol=1e-4)]
  assert not wrong, f'{len(wrong)} cells differ, first {sorted(wrong)[:4]}'
