# SPDX-License-Identifier: MIT
"""Two GEMMs separated by a :class:`GridFenceDescr`.

``D = A @ B`` in section 0; the fence cuts the descr list; then
``E = D @ C`` in section 1. ``D`` ends up as
:data:`DataFlowDirection.SOURCESINK` automatically (it's the sink of
descr 0 and the source of descr 2, so the second
``set_data_flow_direction`` call promotes it).

What this case exercises specifically:

* the launcher signature gains *two* ``numElements`` and *two*
  ``flags`` parameters (one per section); the harness driver counts
  ``len(gen._sections)`` and emits the right number of each (see
  ``driver_emit.py`` near the launcher call assembly);
* the cross-section dataflow is exact — if section 1 reads stale ``D``
  values (no proper fence semantics), the comparison fails;
* shared memory is still per-kernel because both sections compile into
  one cooperative-style kernel; section 0's barrier-bound shmem usage
  must not corrupt section 1's working set.

``GridFenceDescr.trueBarrier()`` returns ``False`` — the generator
emits two sequential sections inside one kernel but does **not**
switch to cooperative launch (``cudaLaunchCooperativeKernel``); see
``GridBarrierDescr`` for that path.

TODO: it is not; the two kernels will be run simultaneously.
"""

import numpy as np

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators.descriptions import GemmDescr, GridFenceDescr

NAME = "fence_two_gemms_16x16"
# Two tensors are written --- the intermediate D and the result E ---
# so the case has to name the one `reference()` returns.
OUTPUT = "E"
DTYPE = Datatype.F32
BATCH = 4
TOL = (1e-4, 1e-4)         # two-step chain — slightly looser than single-step


def descr_list():
    a = SubTensor(Tensor([16, 16], Addressing.STRIDED,
                         BoundingBox([0, 0], [16, 16]),
                         alias="A", datatype=DTYPE))
    b = SubTensor(Tensor([16, 16], Addressing.STRIDED,
                         BoundingBox([0, 0], [16, 16]),
                         alias="B", datatype=DTYPE))
    d = SubTensor(Tensor([16, 16], Addressing.STRIDED,
                         BoundingBox([0, 0], [16, 16]),
                         alias="D", datatype=DTYPE))
    c = SubTensor(Tensor([16, 16], Addressing.STRIDED,
                         BoundingBox([0, 0], [16, 16]),
                         alias="C", datatype=DTYPE))
    e = SubTensor(Tensor([16, 16], Addressing.STRIDED,
                         BoundingBox([0, 0], [16, 16]),
                         alias="E", datatype=DTYPE))
    # Second descr needs a fresh SubTensor referencing the *same* Tensor —
    # otherwise the generator's matrix-list dedup misses it as a cross-
    # section reuse and the symbol lookup fails on the second pass.
    d_in = SubTensor(d.tensor)
    return [
        GemmDescr(False, False, a=a, b=b, c=d),
        GridFenceDescr(),
        GemmDescr(False, False, a=d_in, b=c, c=e),
    ]


def reference(inputs, dest_in):
    D = np.einsum("bik,bkj->bij", inputs["A"], inputs["B"])
    return np.einsum("bik,bkj->bij", D, inputs["C"])
