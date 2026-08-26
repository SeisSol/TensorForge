# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""``B[i, j] = 1 / A[i, j]`` — single-unary-op ElementwiseDescr.

Domain: ``rcp`` blows up near zero. ``INPUT_TRANSFORM`` shifts ``|x|``
by 0.5 to keep magnitudes safely above the singularity (the kernel
sees ``|x| + 0.5`` so reciprocals stay in ``(0, 2]``).

The op is ``ew.rcp``, not ``ew.div(b, 1.0, a)`` — though the latter now
through ``div`` and constant-fold to a different IR node (see
folds to the same RCP in ``generators/elementwise.py``).
"""

import numpy as np

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators import elementwise as ew


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
    return [ew.rcp(b, a)]


def reference(inputs, dest_in):
    return 1.0 / inputs["A"]
