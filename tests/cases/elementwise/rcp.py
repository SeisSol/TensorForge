# SPDX-License-Identifier: MIT
"""``B[i, j] = 1 / A[i, j]`` — single-unary-op ElementwiseDescr.

Domain: ``rcp`` blows up near zero. ``INPUT_TRANSFORM`` shifts ``|x|``
by 0.5 to keep magnitudes safely above the singularity (the kernel
sees ``|x| + 0.5`` so reciprocals stay in ``(0, 2]``).

The op is ``optree.rcp``, not ``1.0 / A`` — the latter would route
through ``div`` and constant-fold to a different IR node (see
``optree.div`` at ``optree.py:674``).
"""

import numpy as np

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators import optree
from tensorforge.generators.descriptions import ElementwiseDescr

from harness.optree_helpers import make_tvar

NAME = "elementwise_rcp_16x16"
DTYPE = Datatype.F32
BATCH = 4
TOL = (1e-5, 1e-5)
INPUT_TRANSFORM = {"A": lambda x: np.abs(x) + 0.5}


def descr_list():
    a = SubTensor(Tensor([16, 16], Addressing.STRIDED,
                         BoundingBox([0, 0], [16, 16]),
                         alias="A", datatype=DTYPE))
    b = SubTensor(Tensor([16, 16], Addressing.STRIDED,
                         BoundingBox([0, 0], [16, 16]),
                         alias="B", datatype=DTYPE))
    return [ElementwiseDescr(
        [optree.Assignment(make_tvar(b, 2), optree.rcp(make_tvar(a, 2)))])]


def reference(inputs, dest_in):
    return 1.0 / inputs["A"]
