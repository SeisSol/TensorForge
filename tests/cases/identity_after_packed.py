# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""``Q = M (D (R (I A)))``: ``neighboringFlux[16]`` at order 4.

The whole of ``neighboringFlux[16]`` at order 4 but for the ``+=`` into
``Q`` -- ``I A`` the flux solver's product, then ``rT(0)``, ``fP(1)`` and
``rDivM(0)`` -- with the numbers its
description carries: ``R`` is sparse and packed, and ``D`` is the identity
pattern with weights, which collapses to one product per lane
(`MultilinearInstruction._identity_reduction`) -- reading the first
product's result where the residency left it, in registers -- and the
last product reads what that step hands on.

The kernel has to see the numbers the generator was told, so the inputs for
``R`` and ``D`` are replaced by them (`INPUT_TRANSFORM`).
"""

import numpy as np

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators.descriptions import GemmDescr

NAME = "identity_after_packed"
DTYPE = Datatype.F32
BATCH = 4
TOL = (1e-5, 1e-5)

R_VALUES = np.array([
    [1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
    [0.0, 0.5, 1.5, 0.0, 0.0, 0.0, 0.0, -0.5, -1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 1.5, 0.0],
    [0.0, 0.5, -0.5, 0.0, 0.0, 0.0, 0.0, -0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, -0.5, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.16666666666666666, 0.8333333333333334, 1.6666666666666667, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.16666666666666666, -0.8333333333333334, -1.6666666666666667, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.5, 0.5, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5, -0.5, 1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.3333333333333333, -0.3333333333333333, 0.3333333333333333, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.3333333333333333, 0.3333333333333333, -0.3333333333333333, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05, 0.35, 1.05, 1.75, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.75, 0.25, -1.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.45, 0.15, -0.55, 0.75, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25, -0.25, 0.25, -0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
M_VALUES = np.array([
    [6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 30.0, 30.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 30.0, -10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [-10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 35.0, 105.0, 70.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 105.0, 63.0, -42.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 70.0, -42.0, 14.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, -42.0, -42.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, -42.0, 14.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [14.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 25.2, 126.0, 226.8, 126.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 126.0, 270.0, 54.0, -90.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 226.8, 54.0, -118.8, 54.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 126.0, -90.0, 54.0, -18.0],
    [0.0, 0.0, 0.0, -45.0, -135.0, -90.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, -135.0, -81.0, 54.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, -90.0, 54.0, -18.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 54.0, 54.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 54.0, -18.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [-18.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
D_VALUES = np.array([
    [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, -0.08333333333333333, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.03333333333333333, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, -0.05555555555555555, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.16666666666666666, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.017857142857142856, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.025, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.041666666666666664, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.125],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
INPUT_TRANSFORM = {
    "R": lambda x: np.broadcast_to(R_VALUES, x.shape),
    "D": lambda x: np.broadcast_to(D_VALUES, x.shape),
    "M": lambda x: np.broadcast_to(M_VALUES, x.shape),
}


def descr_list():
    r = SubTensor(Tensor([16, 20], Addressing.NONE, BoundingBox([0, 0], [16, 20]),
                         alias="R", datatype=DTYPE, data=R_VALUES.copy()))
    i = SubTensor(Tensor([32, 9], Addressing.STRIDED, BoundingBox([0, 0], [32, 9]),
                         alias="I", datatype=DTYPE), BoundingBox([0, 0], [20, 9]))
    a = SubTensor(Tensor([9, 9], Addressing.STRIDED, BoundingBox([0, 0], [9, 9]),
                         alias="A", datatype=DTYPE))
    x = SubTensor(Tensor([32, 9], Addressing.STRIDED, BoundingBox([0, 0], [32, 9]),
                         alias="X", datatype=DTYPE, is_tmp=True))
    x_in = SubTensor(x.tensor, BoundingBox([0, 0], [20, 9]))
    t = SubTensor(Tensor([16, 9], Addressing.STRIDED, BoundingBox([0, 0], [16, 9]),
                         alias="T", datatype=DTYPE, is_tmp=True))
    d = SubTensor(Tensor([16, 10], Addressing.NONE, BoundingBox([0, 0], [16, 10]),
                         alias="D", datatype=DTYPE, data=D_VALUES.copy()))
    u = SubTensor(Tensor([16, 9], Addressing.STRIDED, BoundingBox([0, 0], [16, 9]),
                         alias="U", datatype=DTYPE, is_tmp=True))
    m = SubTensor(Tensor([32, 10], Addressing.NONE, BoundingBox([0, 0], [32, 10]),
                         alias="M", datatype=DTYPE, data=M_VALUES.copy()))
    q = SubTensor(Tensor([32, 9], Addressing.STRIDED, BoundingBox([0, 0], [32, 9]),
                         alias="Q", datatype=DTYPE))
    t_in = SubTensor(t.tensor, BoundingBox([0, 0], [10, 9]))
    u_in = SubTensor(u.tensor, BoundingBox([0, 0], [10, 9]))
    x_out = SubTensor(x.tensor, BoundingBox([0, 0], [20, 9]))
    return [GemmDescr(False, False, i, a, x_out, alpha=1.0, beta=0.0),
            GemmDescr(False, False, r, x_in, t, alpha=1.0, beta=0.0),
            GemmDescr(False, False, d, t_in, u, alpha=1.0, beta=0.0),
            GemmDescr(False, False, m, u_in, q, alpha=1.0, beta=0.0)]


def reference(inputs, dest_in):
    x = np.einsum("bik,bkj->bij", inputs["I"][:, :20, :], inputs["A"])
    t = np.einsum("Bik,bkj->bij", inputs["R"], x)
    u = np.einsum("Bik,bkj->bij", inputs["D"], t[:, :10, :])
    return np.einsum("Bik,bkj->bij", inputs["M"], u[:, :10, :])
