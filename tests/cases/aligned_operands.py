# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""``D = A B`` with operands that promise a 16-byte aligned stride.

Every other case in this directory leaves ``Tensor.alignment`` at its default
of 0, which means *unknown* -- and unknown is what `widths_for` turns into
"scalar only". So until this case existed, the whole width-selection path was
unreachable from the corpus: `plan_hops` was exercised at width 1 in 52
linearized loads and at no other width anywhere, and the reinterpret casts it
guards were dead code that only the model tests in ``test_vector_hops`` spoke
to.

16 is what ``yateto.py`` attaches when the memory layout reports
``alignedStride()``, which is the ordinary situation for a SeisSol operator:
the leading dimension is padded so that every column of every matrix in the
batch starts on a 16-byte boundary. ``M`` is a multiple of 4 here for the same
reason -- a promise about column 0 is only a promise about column 1 if the
stride carries it.

The shape is otherwise deliberately dull. This case exists to make a width
decision happen, not to test arithmetic; ``accumulate_chain`` and the rest
already cover the numerics.
"""

import numpy as np

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators.descriptions import GemmDescr

NAME = "aligned_operands"
DTYPE = Datatype.F32
BATCH = 4
TOL = (1e-5, 1e-5)

M, N, K = 16, 8, 16

#: What `yateto.py` sets when the layout reports an aligned stride.
ALIGNMENT = 16


def _t(shape, alias):
    return SubTensor(Tensor(shape, Addressing.STRIDED,
                            BoundingBox([0] * len(shape), list(shape)),
                            alias=alias, datatype=DTYPE,
                            alignment=ALIGNMENT))


def descr_list():
    return [GemmDescr(trans_a=False, trans_b=False,
                      a=_t([M, K], "A"), b=_t([K, N], "B"),
                      c=_t([M, N], "D"),
                      alpha=1.0, beta=0.0)]
