# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""SeisSol's local flux, in the shape production actually dispatches.

Four faces, each contributing ``Result += (A_f @ B) @ C_f``.  Every operand
role in it is a different addressing mode, and that is the point of the case:

* ``A_f`` is 56x56 and :attr:`Addressing.NONE` --- a static operator, one copy
  for the whole grid, read by every block.  Four of them, one per face.
* ``B`` is 56x9 and :attr:`Addressing.PTR_BASED`, the element's own degrees of
  freedom.  It is the *same* operand in all four faces.
* ``C_f`` is 9x9 and ``PTR_BASED`` but constant per element --- the flux
  solver.  Four of them.
* ``Result`` is 56x9, ``PTR_BASED``, and accumulated into across all four.

``X = A_f @ B`` goes through a temporary; nothing outside the kernel sees it.

The first face writes ``Result`` and the other three add to it, which is
``Result = 0`` followed by four ``+=`` without an instruction that only zeroes.

Padding: ``B`` and ``Result`` are stored 64x9 with a 56x9 bounding box, which
is what SeisSol does and what makes the leading dimension a multiple of the
lane count.  The operators are not padded --- they are 56x56 exactly, and a
lead of 56 over 32 lanes is what the corpus otherwise lacks.

Why this case exists rather than another GEMM: it is the shape the questions
about batch-constant operands are actually about.  ``A_f`` is read by every
block and never changes, so anything done to it inside the batch loop --- the
load, and on a matrix-core path the *conversion* --- is work repeated for no
reason.  Whether hoisting that into the section prologue pays is a
measurement, and this is the workload it has to be taken on;
``Options.preload_globals`` is the switch, ``tools/bench``'s ``preload`` and
``no-preload`` configurations are the two sides.
"""

import numpy as np

from tensorforge.common.basic_types import (Addressing, DataFlowDirection,
                                            Datatype)
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators.descriptions import MultilinearDescr

NAME = "local_flux"
OUTPUT = "R"
DTYPE = Datatype.F32
BATCH = 8
TOL = (1e-4, 1e-4)

#: Order 6: 56 basis functions, 9 quantities.  The padded leading dimension is
#: what the element-local tensors are stored with.
_M, _N, _PAD = 56, 9, 64
_FACES = 4


def _tensor(shape, alias, addressing, bbox=None, is_tmp=False):
    return Tensor(list(shape), addressing,
                  BoundingBox([0, 0], list(bbox or shape)),
                  alias=alias, datatype=DTYPE, is_tmp=is_tmp)


def descr_list():
    # One operand object per role, reused across the faces where production
    # reuses it: `B` and `Result` are the same buffer in all four.
    b = _tensor([_PAD, _N], "B", Addressing.PTR_BASED, bbox=[_M, _N])
    r = _tensor([_PAD, _N], "R", Addressing.PTR_BASED, bbox=[_M, _N])
    x = _tensor([_M, _N], "X", Addressing.STRIDED, is_tmp=True)

    descrs = []
    for f in range(_FACES):
        a = _tensor([_M, _M], f"A{f}", Addressing.NONE)
        c = _tensor([_N, _N], f"C{f}", Addressing.PTR_BASED)
        # X = A_f @ B  -- a fresh view of the same temporary each face, so the
        # generator sees four writes to one buffer rather than four buffers.
        descrs.append(MultilinearDescr(
            dest=SubTensor(x), ops=[SubTensor(a), SubTensor(b)],
            target=[[0, -1], [-1, 1]], permute=[[0, 1], [0, 1]]))
        # Result += X @ C_f, and the first face writes rather than adds.
        descrs.append(MultilinearDescr(
            dest=SubTensor(r), ops=[SubTensor(x), SubTensor(c)],
            target=[[0, -1], [-1, 1]], permute=[[0, 1], [0, 1]],
            add=(f > 0)))
    return descrs


def reference(inputs, dest_in):
    out = np.array(dest_in, copy=True)
    acc = None
    for f in range(_FACES):
        x = np.einsum("bik,bkj->bij",
                      np.broadcast_to(inputs[f"A{f}"],
                                      (inputs["B"].shape[0], _M, _M)),
                      inputs["B"][:, :_M, :])
        term = np.einsum("bik,bkj->bij", x, inputs[f"C{f}"])
        acc = term if acc is None else acc + term
    out[:, :_M, :] = acc
    return out
