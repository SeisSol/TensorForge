# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""``wide = double(I)`` then ``canonical = embed @ wide``, in a float kernel.

SeisSol's `toCanonical` in single precision, padded: the time integral of a
neighbor is read as floats, widened to doubles, and multiplied by a double
operator into a double result.  The kernel's type is float -- what its arena
holds and what it reads -- while everything after the cast is double.

Summed in float, as every target did while a register image took the kernel's
type, the product is right to about seven digits; in double to fifteen, which
is what the tolerance holds it to.  SeisSol's operator is an embedding of
zeros and ones, and multiplying by it is exact in either precision; a random
one is not.  The reference is NumPy's, in double.
"""

import numpy as np

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators import elementwise as ew
from tensorforge.generators.descriptions import GemmDescr

NAME = "widen_then_ml"
DTYPE = Datatype.F32
BATCH = 4
TOL = (1e-12, 1e-12)
OUTPUT = "canonical"

#: An order-3 basis, the canonical order 4, nine quantities.
BASES, CANONICAL, QUANTITIES = 10, 20, 9


def _t(alias, shape, datatype, addressing=Addressing.STRIDED, tmp=False):
    return Tensor(list(shape),
                  Addressing.PTR_BASED if tmp else addressing,
                  None if tmp else BoundingBox([0, 0], list(shape)),
                  is_tmp=tmp, alias=alias, datatype=datatype)


def descr_list():
    integral = SubTensor(_t("I", (BASES, QUANTITIES), Datatype.F32))
    wide = SubTensor(_t("wide", (BASES, QUANTITIES), Datatype.F64, tmp=True))
    embed = SubTensor(_t("embed", (CANONICAL, BASES), Datatype.F64,
                         Addressing.NONE))
    canonical = SubTensor(_t("canonical", (CANONICAL, QUANTITIES),
                             Datatype.F64))
    return [ew.copy(wide, integral),
            GemmDescr(False, False, a=embed, b=wide, c=canonical)]


def reference(inputs, dest_in):
    return np.einsum("kl,blq->bkq", inputs["embed"][0].astype(np.float64),
                     inputs["I"].astype(np.float64))
