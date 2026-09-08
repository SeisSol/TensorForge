# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Reaching one precision through products of a narrower one.

A matrix unit that multiplies in TF32, BF16 or INT8 can still produce an F32
or F64 result: the operands are taken apart into terms whose sum reproduces
the original, the term products are accumulated, and the ones that fall below
the accumulator's own rounding are dropped.  How many terms, and which of
their products are worth keeping, is arithmetic rather than hardware -- the
same two numbers whether the instruction underneath is `mfma`, `dpas` or
`mma.sync`.

Keeping them here is what stops the two answers from drifting.  A path that
writes its term products out by hand states an error analysis nothing checks,
and states it once per target: the number of terms disappears into the arity
of whatever split routine it calls, and the order the products accumulate in
becomes whatever the tuple happened to be written in.

What is *not* here is which arithmetic to reach for.  That is a catalogue
question -- which entries a target has, at what tile -- and it is answered
where the catalogue is.
"""

from math import ceil
from typing import Optional, Tuple

from tensorforge.common.basic_types import Datatype

#: Significand bits each format carries, the matrix unit's rather than the
#: storage type's.  TF32 is stored in 32 bits and multiplies with 11, which no
#: C++ type records, and it is the number every count below is derived from.
MANTISSA = {
    Datatype.F64: 53,
    Datatype.F32: 24,
    Datatype.TF32: 11,
    Datatype.F16: 11,
    Datatype.BF16: 8,
}


def terms(significand: int, target: Datatype) -> int:
    """Terms of `significand` bits whose sum reproduces a `target` one.

    Three BF16 terms cover F32's 24 bits exactly; two cover 16, which is more
    than TF32 and less than F32.  Both are usable and they are not the same
    claim, so a path taking fewer than this many is making a choice -- and
    :func:`covered` is the number that choice costs.
    """
    return max(1, ceil(MANTISSA[target] / significand))


def covered(significand: int, count: int) -> int:
    """Significand bits `count` terms actually carry.

    The counterpart of :func:`terms`, and the reason a reduced split is
    statable: two TF32 terms carry 22 bits against FP32's 24, which is a
    number to weigh rather than a habit to inherit.
    """
    return significand * count


def products(count: int, keep: Optional[int] = None
             ) -> Tuple[Tuple[int, int], ...]:
    """Which `(i, j)` term products to accumulate, smallest contribution first.

    Term `i` is worth about ``2**(-significand*i)`` of the operand, so the
    product `(i, j)` is worth ``2**(-significand*(i+j))``: everything with
    ``i + j >= keep`` sits at or below the target's own rounding error and is
    dropped.  At ``keep == count`` that leaves ``count*(count+1)/2`` products
    -- six for three terms, three for two.

    Smallest first, so the small contributions accumulate before the large one
    rounds them off.  The order is free and never worse; how much it buys
    depends on how much the accumulator already carries from earlier k, which
    is a measurement rather than a derivation.
    """
    keep = count if keep is None else keep
    pairs = [(i, j) for i in range(count) for j in range(count)
             if i + j < keep]
    return tuple(sorted(pairs, key=lambda p: (-(p[0] + p[1]), p)))
