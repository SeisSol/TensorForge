# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""F64 variant of ``add_true.py`` — accumulation in double precision.

Same shape and structure as ``cases/add_true.py``, but with
:data:`Datatype.F64`. Worth its own case because the
``add=True`` path in :mod:`multilinear_builder` flows through a
different load path for the prev-value-of-C (``_get_target_symbol(True)``
at line 269) — the dtype changes which lexic emit functions are
invoked, so a regression that only affects F64 loads is otherwise
invisible.
"""

import numpy as np

from tensorforge.common.basic_types import Addressing, DataFlowDirection, Datatype
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators.descriptions import MultilinearDescr

NAME = "gemm_add_accumulate_f64"
DTYPE = Datatype.F64
BATCH = 4
TOL = (1e-12, 1e-12)        # F64 — tight tolerance


def descr_list():
    a = SubTensor(Tensor([12, 16], Addressing.STRIDED,
                         BoundingBox([0, 0], [12, 16]),
                         alias="A", datatype=DTYPE))
    b = SubTensor(Tensor([16, 8], Addressing.STRIDED,
                         BoundingBox([0, 0], [16, 8]),
                         alias="B", datatype=DTYPE))
    c = SubTensor(Tensor([12, 8], Addressing.STRIDED,
                         BoundingBox([0, 0], [12, 8]),
                         alias="C", datatype=DTYPE))
    descr = MultilinearDescr(
        dest=c, ops=[a, b],
        target=[[0, -1], [-1, 1]],
        permute=[[0, 1], [0, 1]],
        add=True,
    )
    c.tensor.set_data_flow_direction(DataFlowDirection.SOURCE)
    return [descr]


def reference(inputs, dest_in):
    return dest_in + np.einsum("bik,bkj->bij", inputs["A"], inputs["B"])
