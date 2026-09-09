# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""How an operand gets from the distribution it has to the one a consumer
wants.

The question is the same on every target and the answers are not, which is
why they are separated here.  Two of the four rungs are facts about bits:
that there is no gap at all, and that a trip through memory closes any gap
there is.  The two in between are instructions -- a single one where the
target has it, an assembly of swaps and merges where it does not -- and those
belong to whoever has them.

This module owns the first two and the order.  A target hands in the middle
two as `Rungs`, or hands in nothing and gets the two that are always true.
Handing in nothing is a real answer and not a stub: it says this target has
no register route between these layouts, which is exactly NVIDIA's position
and was the reason its refusal had to be a literal while `reach` lived in
`primitives/amd`.
"""

from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

from . import bitlayout, staging


@dataclass(frozen=True)
class Rungs:
    """What a target can put between two register layouts.

    `single` is asked about the two layouts and not about the gap between
    them, even though this module has already computed one.  Handing the gap
    over would fix what a rung is allowed to look at, and a target whose
    instruction is not an exchange of bits -- a broadcast, a replication --
    could then not answer at all.  Recomputing a pure function of two
    arguments is the cheaper mistake.

    `assembled` returns the route or `None`, and `None` covers both "cannot"
    and "could be priced but not written" -- the difference matters to the
    target and not here, and folding it in keeps the reason next to the masks
    it is a reason about.
    """

    #: Whether one instruction of this target's closes this gap.
    single: Optional[Callable[[Any, Any, int], bool]] = None
    #: The same exchange out of smaller moves, or `None`.
    assembled: Optional[Callable[[Any, Any, Sequence, int], Any]] = None
    #: What an assembled route issues, for the comparison against the trip.
    cost: Optional[Callable[[Any], int]] = None


def reach(have, want, ext: int, indices, wave: Optional[int] = None,
          rungs: Optional[Rungs] = None):
    """How the operand gets from one distribution to the other.

    Four answers, cheapest first, and each is a different kind of thing: `0`
    is nothing to emit, `1` is one instruction of the target's, a tuple of
    `Move` is that same exchange assembled out of smaller ones, and a tuple of
    `Transfer` is the trip through memory.  Ordered by what they cost rather
    than by which is convenient, because the order *is* the preference.

    A packed operand is unpacked first and then answered like any other, and
    whether that closes the gap depends on which operand it is.  A shared
    matrix reduces to the layout a nest already reads and costs the extracts.
    A lead operand does not: its width puts the low bits of the index inside
    the register and a fragment wants them across the lanes, so unpacking
    moves them the wrong way and what remains is a permutation between lane
    weights.

    Never `None`.  The staged path closes every gap, so a caller reaching here
    always has an answer -- what it does not always have is one it can afford,
    and `1` and the assembled route are the two it might not be offered.
    """
    # A vector bit is an element of a packed register and reaching one is a
    # subscript rather than a shuffle, so it is not a gap for the cross-lane
    # machinery -- it closes first, and the extracts are what it costs.
    have, _ = bitlayout.unpacked(have)

    gap = bitlayout.is_exchange(bitlayout.displacement(have, want))
    if gap == ():
        return 0
    if (rungs is not None and rungs.single is not None
            and rungs.single(have, want, ext)):
        return 1
    trip = staging.staged(have, want, indices)
    if wave is None or rungs is None or rungs.assembled is None:
        return trip
    composed = rungs.assembled(have, want, indices, wave)
    if composed is None:
        return trip
    # The last two rungs are compared rather than ordered.  Registers beat
    # memory at every width in the AMD catalogue -- 224 instructions against
    # 1024 accesses at width eight -- but that is a count and not a law, and a
    # gap with one region per element would not.
    stores, loads = staging.accesses(trip)
    return composed if rungs.cost(composed) <= stores + loads else trip


def lead_route(shape, rungs: Optional[Rungs] = None):
    """How this shape's lead operand reaches the distribution a fragment
    wants, in the kinds `reach` answers in.

    `0` at width one, and that is not a shortcut: an unpacked lead operand
    already arrives spread one element per lane, which is the fragment's own
    reading, so there is no gap.  What a fragment wants is therefore the lead
    operand at width one and not a third statement of a distribution, which
    is what makes the two sides comparable by construction.

    One function because every target asks it of the same operand and would
    otherwise each derive it, and because what differs between them is not
    the question but the rungs -- which is the whole argument for the
    parameter.  The extent is the thread count: the index space this operand
    spans in one issue, which is also what the trip has to carry.
    """
    from .strategy import lead_layout

    have = shape.lead_layout
    if have is None or not bitlayout.packed(have):
        return 0
    threads = shape.threads
    return reach(have, lead_layout(threads, 1), threads,
                 [(index,) for index in range(threads)],
                 wave=threads if rungs is not None else None, rungs=rungs)
