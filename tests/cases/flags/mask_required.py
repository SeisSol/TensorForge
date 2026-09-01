# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""A kernel whose caller asked for the per-element mask.

``ATTRS`` names the mask, so the launcher takes a ``flags0`` parameter
without a default and the loop body dereferences it unconditionally.  The
same GEMM as ``gemm_square_16`` otherwise, so the snapshot diff against it
is exactly the mask and nothing else.

The driver passes an all-ones mask (see ``driver_emit``), which enables
every element -- what is under test here is the signature and the guard,
not what masking off an element does to the result.
"""

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators.descriptions import GemmDescr

NAME = "flags_mask_required"
DTYPE = Datatype.F32
BATCH = 4
TOL = (1e-5, 1e-5)
ATTRS = {"flags": True}


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
