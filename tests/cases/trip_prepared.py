# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""A prepared operand read through the trip machinery.

`prepare_operands` stores ``A`` in the SIMT interleave, so a lane reads two
of its rows as one vector; `trip_reads` holds what a trip of a rolled
reduction reads in a register array and fills it for the trip after.  The
two did not combine: a trip slot was one scalar, and the pass passed over
every vector load it met -- `local_flux` at `k_roll=2` had 28 trip slots
unprepared and none prepared.

A slot is now one *value*, which may be a vector: the array is scalars as
every register array here is, a slot is `width` of them, and the emitter
writes the same cast it writes for any wide register access.  56 rows over
32 lanes is two slots a lane, which is what makes the interleave offer a
group of two in the first place.
"""

import numpy as np

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators.descriptions import GemmDescr

NAME = "trip_prepared"
DTYPE = Datatype.F32
BATCH = 4
TOL = (1e-5, 1e-5)
OPTIONS = {"prepare_operands": True, "trip_reads": True, "k_roll": 2}
M, K, N = 56, 16, 6


def _values():
    rng = np.random.default_rng(71)
    # Every cell carries a number: a wrong slot cannot be hidden by a zero
    # standing where the right one would have been.
    return rng.uniform(0.5, 1.5, size=(M, K))


A_VALUES = _values()
INPUT_TRANSFORM = {"A": lambda x: np.broadcast_to(A_VALUES, x.shape)}


def descr_list():
    a = SubTensor(Tensor([M, K], Addressing.NONE, BoundingBox([0, 0], [M, K]),
                         alias="A", datatype=DTYPE, data=A_VALUES.copy()))
    b = SubTensor(Tensor([K, N], Addressing.STRIDED, BoundingBox([0, 0], [K, N]),
                         alias="B", datatype=DTYPE))
    c = SubTensor(Tensor([M, N], Addressing.STRIDED, BoundingBox([0, 0], [M, N]),
                         alias="C", datatype=DTYPE))
    return [GemmDescr(False, False, a, b, c, alpha=1.0, beta=0.0)]


def reference(inputs, dest_in):
    return np.einsum("Bik,bkj->bij", inputs["A"], inputs["B"])
