# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""``M = A @ B`` then ``C = abs(M)``, with ``M`` a global output.

The one case in this group that generates.  It is also the only one that is
wrong, which is the reason it exists.

With a global destination the store is deferred to the epilogue so the value
can stay in registers, and the epilogue is emitted after every descriptor of
the section.  The elementwise therefore reads `M` from global memory before
anything has written it, and the contraction's result overwrites `M` afterwards:

    // glb_m3 = abs(glb_m0);     <- reads whatever the caller left there
    // glb_m0 = store{r>g}(r1);  <- writes it only now

`verify` cannot see it, because `m0` is a kernel parameter and so a symbol with
a definition on entry.  `test_mixed_residency.py` states the ordering the
generated source has to have; that test is the failing one, not this snapshot.

Shapes are 8x8 throughout: the elementwise descriptor pins the lane count to the
vector unit length regardless of the tensors, so nothing is gained by going
smaller, and 8x8 keeps a snapshot diff readable.

`abs` is the pointwise operation because it is total and exact -- no input
domain to shape, no tolerance spent on a transcendental, and every backend has
it.  ESIMD, for one, has no `tanh` intrinsic at all.
"""

import numpy as np

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators import elementwise as ew
from tensorforge.generators.descriptions import GemmDescr

DTYPE = Datatype.F32
BATCH = 4
TOL = (1e-5, 1e-5)
N = 8


def _t(alias, shape=(N, N), tmp=False):
    return Tensor(list(shape),
                  Addressing.PTR_BASED if tmp else Addressing.STRIDED,
                  BoundingBox([0] * len(shape), list(shape)),
                  alias=alias, is_tmp=tmp, datatype=DTYPE)


def _s(alias, shape=(N, N), tmp=False):
    return SubTensor(_t(alias, shape, tmp))


NAME = "mixed_ml_glb_then_ew"
#: `M` and `C` are both sinks; the reference describes `C`.
OUTPUT = "C"


def descr_list():
    a, b, m, c = _s("A"), _s("B"), _s("M"), _s("C")
    return [GemmDescr(False, False, a=a, b=b, c=m),
            ew.abs(c, m)]


def reference(inputs, dest_in):
    return np.abs(np.einsum("bik,bkj->bij", inputs["A"], inputs["B"]))
