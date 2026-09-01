# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Which arrangement computes a contraction.

The companion of :mod:`tensorforge.backend.placement`, and split along the
same seam.

*Legality* is what the operation and the target allow.  A matrix core owns the
lane-to-register mapping of its fragments, so an arrangement that has already
changed that mapping cannot hand it operands; an instruction whose tile does
not fit the thread count cannot be issued at all.  These are facts, and a
kernel that gets one wrong is wrong -- usually silently, since the generated
code still compiles and the numbers are merely not the ones asked for.

*Preference* is which of the legal arrangements is worth taking.  Matrix cores
against a broadcast chain against the plain nest is a throughput question with
a different answer per target and, eventually, per shape.  A kernel that gets
one wrong is slow.

The arrangements are four, and they are distinguished by *where the operand
that is not spread over the lanes comes from* -- which is the question every
one of them answers differently, and the only one that separates them:

* the nest reads it from memory, staged or in place;
* the broadcast chain moves it between lanes and multiplies;
* DPP does the same move as a modifier on the multiply itself;
* a matrix core takes both operands as fragments and does the move internally.

Vector width is deliberately not one of them.  Packing the lead dimension into
`float2`/`float4` is a change to how wide each of these steps is, not a
different arrangement, so it lives on the instruction as `lead_width` and
appears here only as the exclusion in :func:`is_contraction`.  Staging into a
scratch shared buffer is likewise not one: where an operand is copied to is
:mod:`placement`'s question, and the nest reads the answer without knowing
which of them asked.
"""

from dataclasses import dataclass
from enum import Enum
from typing import FrozenSet, Iterable, Tuple

from tensorforge.common.basic_types import Datatype


class Strategy(Enum):
    """How the products of a contraction get emitted."""

    #: The generic loop nest.  Always legal, and the only arrangement that
    #: needs nothing of the target.
    GENERIC = 'generic'
    #: One of the operand's lanes broadcast to all of them, then a scalar FMA
    #: per product.  The broadcast is a real cross-lane instruction under SPMD
    #: -- `readlane`, `group_broadcast` -- and an element read out of the
    #: work-item's own vector under an explicit one, which is the whole
    #: difference in what it costs.
    BROADCAST = 'broadcast'
    #: The same arrangement with the broadcast folded into the multiply as an
    #: instruction modifier, so it is not an instruction of its own.
    DPP = 'dpp'
    #: A matrix core: both operands as fragments, the lane distribution fixed
    #: by the instruction rather than by the loop.
    MATRIX = 'matrix'


@dataclass(frozen=True)
class ComputeShape:
    """What the choice is made from.

    Deliberately without a writer, a symbol or a context: legality is a
    question about the shape and the target, and keeping it answerable from
    numbers is what lets it be asserted directly instead of inferred from
    generated text.
    """

    #: Lanes the lead dimension is spread over.
    threads: int
    #: Accumulator type.  The operand type may be narrower -- an emulated path
    #: splits its inputs and keeps the sum in this.
    dtype: Datatype
    #: Whether the second operand is stored sparsely, which decides both how
    #: it is read and which arrangements can read it that way.
    sparse: bool
    #: Whether the lowering holds a whole wave per work-item.  Not a property
    #: of the hardware: the same target admits both, and what a cross-lane
    #: broadcast costs differs between them by more than its spelling.
    explicit_simd: bool


def is_contraction(operands: int, lead_width: int) -> bool:
    """Whether anything but the nest could compute this at all.

    Two conditions, and neither is a preference.

    Every arrangement below names an `A` and a `B`; a product of three or more
    operands has no such split and only the nest walks it.

    A lead width above one distributes the lead dimension across the lanes in
    blocks rather than cyclically.  The matrix arrangements own that mapping
    -- handing MFMA operands addressed at stride `width` gives it the right
    registers in the wrong places -- and the broadcast arrangements index the
    lanes directly.  Width and arrangement are alternatives rather than
    composable, and saying so here is what stops one from quietly corrupting
    the other.
    """
    return operands == 2 and lead_width == 1


def legal_strategies(offered: Iterable[Strategy]) -> FrozenSet[Strategy]:
    """What may be emitted for this shape on this target.

    Whatever the target offers, and the nest -- which is always available and
    is why every other arrangement is free to decline.
    """
    return frozenset(offered) | {Strategy.GENERIC}


#: The order each target takes the legal arrangements in.  Preference only:
#: every entry is already legal by the time this is read, and a target with no
#: row runs the nest, which is correct and slow.
#:
#: The eventual producer of this order is a cost model with the instruction
#: throughputs and the register pressure in hand.  Until there is one, a fixed
#: order per vendor states the same claim in a form that can be replaced
#: wholesale rather than unpicked from the dispatch.
PREFERENCES = {
    # Matrix cores where a tile fits; the DPP chain otherwise, which is why
    # F64 lands there without the order naming a type.
    'amd': (Strategy.MATRIX, Strategy.DPP, Strategy.GENERIC),
    'nvidia': (Strategy.MATRIX, Strategy.GENERIC),
    # The register-only chain beats staging operands through shared memory
    # here, and whether DPAS beats it in turn is a measurement rather than a
    # preference -- the order says what to try first, not what is known.
    'intel': (Strategy.MATRIX, Strategy.BROADCAST, Strategy.GENERIC),
}

DEFAULT_PREFERENCE: Tuple[Strategy, ...] = (Strategy.GENERIC,)


def choose_strategy(legal: FrozenSet[Strategy], vendor: str) -> Strategy:
    """The first legal arrangement this target prefers."""
    for strategy in PREFERENCES.get(vendor, DEFAULT_PREFERENCE):
        if strategy in legal:
            return strategy
    return Strategy.GENERIC
