# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""An accumulation onto a temporary that two slices assembled.

    t[0:6]  = a
    t[6:12] = b
    t      += w          <- adds onto both slices
    D       = t

Under the atomic-accumulation policy, the default on AMD, an accumulation
reads no old value: the atomic store adds onto what global memory holds.  A
temporary's buffer is stored plainly, though, so the accumulation onto it
found nothing to add to and `t` came out as `w` alone.
"""

import numpy as np

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators.descriptions import MultilinearDescr

NAME = "temp_accumulate_assembled"
OUTPUT = "D"
DTYPE = Datatype.F32
BATCH = 4
TOL = (1e-4, 1e-4)

N, SPLIT = 12, 6


def _global(alias, size):
    return Tensor([size], Addressing.STRIDED, BoundingBox([0], [size]),
                  alias=alias, datatype=DTYPE)


def _copy(dest, src, add=False):
    return MultilinearDescr(dest=dest, ops=[src], target=[[0]], permute=[[0]], add=add)


def descr_list():
    t = Tensor([N], Addressing.PTR_BASED, BoundingBox([0], [N]),
               alias="t", is_tmp=True, datatype=DTYPE)
    top = SubTensor(t, BoundingBox([0], [SPLIT]), [0], sliced=True)
    bottom = SubTensor(t, BoundingBox([0], [N - SPLIT]), [SPLIT], sliced=True)
    return [
        _copy(top, SubTensor(_global("a", SPLIT))),
        _copy(bottom, SubTensor(_global("b", N - SPLIT))),
        _copy(SubTensor(t), SubTensor(_global("w", N)), add=True),
        _copy(SubTensor(_global("D", N)), SubTensor(t)),
    ]


def reference(inputs, dest_in):
    return np.concatenate([inputs["a"], inputs["b"]], axis=1) + inputs["w"]
