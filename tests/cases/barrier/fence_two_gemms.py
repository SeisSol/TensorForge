# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Two *independent* GEMMs separated by a :class:`GridFenceDescr`.

``D = A @ B`` in section 0; the fence cuts the descr list; then
``E = C @ F`` in section 1.  Nothing section 1 reads is written by
section 0, and that is the point of the case rather than an omission.

A fence is not an ordering.  ``GridFenceDescr.trueBarrier()`` returns
``False``, so ``generator.py`` never sets ``section.barrier`` and never
switches to a cooperative launch -- the two sections may be in flight
across the grid at the same time.  ``_section_traversal`` says so
outright and then acts on it (``generator.py:501-522``): a section that
follows another *without* a barrier starts at
``(blockId + numElements0) % stride`` rather than at ``blockId``,
deliberately, "so that consecutive sections do not all hammer the same
elements".  A case that made section 1 read what section 0 wrote would
therefore be reading another block's output with no synchronisation
between the two -- racy by construction, and green only by luck.  This
case used to be exactly that, and carried a ``TODO`` saying so.

What the fence does provide, and what this case pins down:

* the launcher signature gains *two* ``numElements`` and *two*
  ``flags`` parameters (one per section); the harness driver counts
  ``len(gen._sections)`` and emits the right number of each (see
  ``driver_emit.py`` near the launcher call assembly);
* the offset traversal still *covers*: ``b -> (b + numElements0) %
  stride`` is a bijection on ``[0, stride)``, so every element is
  computed exactly once even though section 1 is walked by different
  blocks than section 0.  A wrong offset shows up here as elements
  computed twice or not at all, which the comparison catches;
* shared memory is per-section but shares one arena, so section 0's
  working set must not corrupt section 1's.

``BATCH`` is deliberately larger than a typical occupancy-derived grid
so that blocks iterate more than once and the modulo wraps.

Ordering across a section boundary is ``barrier_two_gemms.py``'s
subject, not this one's.

Only ``E`` is compared -- the harness compares a single output, and
``D`` is left as a second sink so the ``OUTPUT`` selection path stays
exercised.  Section 0's arithmetic is a plain square GEMM and is
covered by ``cases/gemm_square_16.py``.
"""

import numpy as np

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators.descriptions import GemmDescr, GridFenceDescr

NAME = "fence_two_gemms_16x16"
# Two tensors are written --- D in section 0 and E in section 1 ---
# so the case has to name the one `reference()` returns.
OUTPUT = "E"
DTYPE = Datatype.F32
# Past the grid size an occupancy query yields on the machines this runs
# on, so the batch loop takes more than one trip and the section-1 start
# offset wraps.
BATCH = 96
TOL = (1e-4, 1e-4)


def _square(alias):
    return SubTensor(Tensor([16, 16], Addressing.STRIDED,
                            BoundingBox([0, 0], [16, 16]),
                            alias=alias, datatype=DTYPE))


def descr_list():
    a = _square("A")
    b = _square("B")
    d = _square("D")
    c = _square("C")
    f = _square("F")
    e = _square("E")
    return [
        GemmDescr(False, False, a=a, b=b, c=d),
        GridFenceDescr(),
        GemmDescr(False, False, a=c, b=f, c=e),
    ]


def reference(inputs, dest_in):
    return np.einsum("bik,bkj->bij", inputs["C"], inputs["F"])
