# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Two GEMMs separated by a :class:`GridBarrierDescr` — cooperative launch.

``D = A @ B`` then ``E = D @ C``, and the section boundary between
them is a true grid-wide barrier.
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
* ``cooperative_groups::this_grid()`` comes from
  ``cooperative_groups.h``, which ``tensorforge_device/cuda.h``
  includes.

The fence case, ``fence_two_gemms.py``, runs two *independent* GEMMs
instead, because a fence does not order its sections.  Ordering across
a section boundary is this case's subject alone, which is why its batch
has to exceed the grid.

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
# Two tensors are written --- the intermediate D and the result E ---
# so the case has to name the one `reference()` returns.
OUTPUT = "E"
DTYPE = Datatype.F32
# Past the grid size an occupancy query yields on the machines this runs
# on, so the batch loop takes more than one trip and blocks take *unequal*
# numbers of trips.  What that buys is a guard on where the grid sync sits:
# outside the batch loop it is reached once by every block, which is what
# the generator emits; moved inside it, the blocks with fewer trips stop
# arriving and the kernel hangs.  A batch below the grid size gives every
# block exactly one trip and cannot tell the two placements apart.
#
# It does *not* make the case test grid-wide ordering, and no batch size
# would: a barrier resets the section-1 traversal to `blockId`
# (`Generator._section_traversal`), so both sections walk the same elements
# in the same blocks and every block only ever reads back what it wrote
# itself.
# Ordering across blocks is not expressible as a case while a section's
# element mapping is fixed by the block id.
BATCH = 96
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
