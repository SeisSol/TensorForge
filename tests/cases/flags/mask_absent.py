# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""A kernel whose caller has attributes and did not ask for the mask.

An empty ``ATTRS`` is not the same as no ``ATTRS``: it says the frontend
does have the attribute channel and left the mask out, so the launcher has
no ``flags0`` parameter and the loop body has no guard at all.  A case
without ``ATTRS`` gets the mask with a null check instead, which is what
every other case in this corpus records.

The same GEMM as ``gemm_square_16``, so the two snapshots differ in exactly
the mask.
"""

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators.descriptions import GemmDescr

NAME = "flags_mask_absent"
DTYPE = Datatype.F32
BATCH = 4
TOL = (1e-5, 1e-5)
ATTRS = {}


def descr_list():
    a = SubTensor(Tensor([16, 16], Addressing.STRIDED,
                         BoundingBox([0, 0], [16, 16]),
                         alias="A", datatype=DTYPE))
    b = SubTensor(Tensor([16, 16], Addressing.STRIDED,
                         BoundingBox([0, 0], [16, 16]),
                         alias="B", datatype=DTYPE))
    c = SubTensor(Tensor([16, 16], Addressing.STRIDED,
                         BoundingBox([0, 0], [16, 16]),
                         alias="C", datatype=DTYPE))
    return [GemmDescr(trans_a=False, trans_b=False,
                      a=a, b=b, c=c, alpha=1.0, beta=0.0)]
