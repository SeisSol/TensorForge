# SPDX-License-Identifier: MIT
"""Accumulate into a tensor, then read it back in the same kernel.

    D          = Q  @ S
    t_i        = Q  @ B_i
    D[0:16, :] += W_i @ t_i        for i = 0, 1, 2
    O          = D  @ C

The row slice on the accumulations makes ``D`` a destination written in
several boxes, so each write goes out to memory as it is produced and the
final descriptor reads ``D`` back from there.  That read is the one thing in
the kernel that must not move: ``MoveLoads`` hoists transfers to hide their
latency, and it used to do so without looking at what lay between, so the read
of ``D`` ended up above the last accumulation's store.  Every term but the
last reached ``O`` --- no crash, no diagnostic, a result that is wrong by one
summand.

This is the shape of an ADER derivative kernel, which is why three of the
poroelastic kernels showed it and nothing in the suite did.
"""

import numpy as np

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.helper import generate_tmp_matrix
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators.descriptions import GemmDescr, MultilinearDescr

NAME = "accumulate_then_read"
# The kernel writes two tensors --- the accumulator D and the result O --- so
# it has to say which one `reference()` returns.
OUTPUT = "O"
DTYPE = Datatype.F32
BATCH = 4
TOL = (1e-4, 1e-4)

M, N, HALF, TERMS = 32, 13, 16, 3


def _t(shape, alias):
    return Tensor(shape, Addressing.STRIDED,
                  BoundingBox([0] * len(shape), list(shape)),
                  alias=alias, datatype=DTYPE)


def descr_list():
    q = _t([M, N], "Q")
    d = _t([M, N], "D")
    out = [GemmDescr(False, False, a=SubTensor(q), b=SubTensor(_t([N, N], "S")),
                     c=SubTensor(d), alpha=1.0, beta=0.0)]
    for i in range(TERMS):
        b = _t([N, N], f"B{i}")
        tmp = SubTensor(generate_tmp_matrix(SubTensor(q), SubTensor(b)))
        out.append(GemmDescr(False, False, a=SubTensor(q), b=SubTensor(b),
                             c=tmp, alpha=1.0, beta=0.0))
        out.append(MultilinearDescr(
            dest=SubTensor(d, bbox=BoundingBox([0, 0], [HALF, N])),
            ops=[SubTensor(_t([HALF, M], f"W{i}")), tmp],
            target=[[0, -1], [-1, 1]], permute=[[0, 1], [0, 1]], add=True))
    out.append(GemmDescr(False, False, a=SubTensor(d),
                         b=SubTensor(_t([N, N], "C")),
                         c=SubTensor(_t([M, N], "O")), alpha=1.0, beta=0.0))
    return out


def reference(inputs, dest_in):
    q = inputs["Q"]
    d = np.einsum("bik,bkj->bij", q, inputs["S"])
    for i in range(TERMS):
        t = np.einsum("bik,bkj->bij", q, inputs[f"B{i}"])
        d[:, :HALF, :] = d[:, :HALF, :] + np.einsum("bik,bkj->bij",
                                                    inputs[f"W{i}"], t)
    return np.einsum("bik,bkj->bij", d, inputs["C"])
