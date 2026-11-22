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

What is *not* here is which arithmetic to reach for.  That is a catalog
question -- which entries a target has, at what tile -- and it is answered
where the catalog is.
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


#: The orders a list of term products is issued in (`products`): the
#: smallest contribution first, or the largest.
ORDERS = ('smallest', 'largest')

#: Which side's term breaks a tie between two products of one contribution
#: (`products`): A's -- `(0, 1)` before `(1, 0)` -- or B's.
MAJORS = ('a', 'b')


def products(count: int, keep: Optional[int] = None, *,
             counts: Optional[Tuple[int, int]] = None,
             order: str = 'smallest', major: str = 'a'
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

    The paths that issue the products do not all take that order, and say
    which they take here rather than in a literal of their own (`terms.plan`):
    `order='largest'` puts the largest contribution first -- NVIDIA's
    `mma.sync`, wgmma and tcgen05 issue 3xTF32 so, the upper halves first --
    and `major='b'` breaks a tie between two products of one contribution by
    B's term rather than A's -- the Intel paths iterate this function's pairs
    as `(B's term, A's term)`, which is that order once the pair is named
    `(A's, B's)`.  `counts` takes the sides apart where they have different
    numbers of terms -- A in two TF32 halves against B whole, `(2, 1)` --
    `keep` then defaulting to the larger; `count` is the number for both
    where it is not given.
    """
    if order not in ORDERS:
        raise ValueError(f'an order of {ORDERS}, not {order!r}')
    if major not in MAJORS:
        raise ValueError(f'a major side of {MAJORS}, not {major!r}')
    a, b = (count, count) if counts is None else (int(c) for c in counts)
    keep = max(a, b) if keep is None else keep
    pairs = [(i, j) for i in range(a) for j in range(b) if i + j < keep]
    sign = -1 if order == 'smallest' else 1

    def key(p):
        tie = p if major == 'a' else (p[1], p[0])
        return (sign * (p[0] + p[1]), tie)
    return tuple(sorted(pairs, key=key))
