# SPDX-License-Identifier: MIT
"""Two GEMMs separated by a :class:`GridBarrierDescr` — cooperative launch.

Same arithmetic as ``fence_two_gemms.py`` (``D = A @ B`` then
``E = D @ C``) but the section boundary is a true grid-wide barrier.
``GridBarrierDescr.trueBarrier()`` returns ``True``, which in turn
sets ``section.barrier`` and triggers ``persistent_threading``:

* the launcher emits ``cudaLaunchCooperativeKernel`` instead of
  ``<<<...>>>``;
* the grid size becomes ``gridsize`` (the persistent worker count
  derived from device occupancy) rather than
  ``min(gridsize, numElements0)``;
* the cooperative-launch path needs ``tensorforge::argsPtrs`` from
  ``tensorforge_aux.h``, which is already on the test driver's
  include path via ``toolchain.py``;
* helper headers (``cooperative_groups.h``,
  ``cooperative_groups/memcpy_async.h``) come in via
  ``gen.get_helper_headers()`` — already wired in ``runner.py:99``.

The compute layer is identical to the fence case, so once the fence
case passes the only failure mode here is the cooperative-launch
plumbing.

Cooperative launch has a device-side prerequisite: the GPU must
support ``cudaDevAttrCooperativeLaunch``. Almost everything ≥ sm_60
does, but a stripped-down emulator might not, in which case
``cudaLaunchCooperativeKernel`` returns
``cudaErrorCooperativeLaunchTooLarge`` / ``cudaErrorNotSupported`` at
runtime and the case fails with a clear error in ``stderr.txt``.
"""

import numpy as np

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators.descriptions import GemmDescr, GridBarrierDescr

NAME = "barrier_two_gemms_16x16"
DTYPE = Datatype.F32
BATCH = 4
TOL = (1e-4, 1e-4)


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
    d_in = SubTensor(d.tensor)
    return [
        GemmDescr(False, False, a=a, b=b, c=d),
        GridBarrierDescr(),
        GemmDescr(False, False, a=d_in, b=c, c=e),
    ]


def reference(inputs, dest_in):
    D = np.einsum("bik,bkj->bij", inputs["A"], inputs["B"])
    return np.einsum("bik,bkj->bij", D, inputs["C"])
