# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""What a matrix path is handed, and what it owes back.

Every path that stands in for the generic loop nest -- a matrix core, a
broadcast chain, a packed-VALU arrangement -- reads and writes the same three
operands through the same three accessors, and the nest is what builds them.
The accessors are the whole interface: a path is never given a symbol, a data
view or a slicing offset, only ``A(writer, var, i, k)``, ``B(writer, var, j,
k)`` and ``C(writer, value, i, j)``.

Two properties of that interface are invisible at the call site and cost a
silent wrong answer when a path assumes the other one, so they are stated here
rather than left to each path to rediscover.

**The lead index is in slots.**  ``A`` and ``C`` take ``i`` as a register slot
-- one per ``threads`` elements of the lead dimension -- and wrap it at
:attr:`lead_slots`.  A path may walk the lead dimension in elements, and one
that does divides before it asks.  Handing an element count over unchanged
asks for the same slot ``threads`` times and gets the same value back: no
error anywhere, one product accumulated into everything.  Both counts are
named so that a path has to say which it means.

**Declining is free only through the writer.**  A path returns ``False`` to
mean the generic nest should run instead.  The nest calls it inside
:meth:`Writer.speculative` and discards on a decline, so a path may give up
after it has emitted -- but only what went through the writer comes back.  A
reservation made before generation does not; :func:`scratch` is where a path
states what it needs, and the same function answers ``temp_shmem``.
"""

from dataclasses import dataclass
from typing import Callable, Optional

from tensorforge.common.basic_types import Datatype


@dataclass(frozen=True)
class MatmulOperands:
    """``C[i,j] += A[i,k] * B[k,j]`` as the accessors and extents of one call.

    Held together rather than passed as a positional tail, because the two
    lead counts differ by a factor of :attr:`threads` and are both plain
    ``int``: as arguments they are interchangeable to the type checker, to the
    reader, and to the caller.
    """

    #: ``A(writer, var, i, k) -> value | bool``.  ``var=None`` asks for the
    #: value itself rather than for a name to fill in, which is what an
    #: operand of a vendor intrinsic has to be: a value whose definition the
    #: IR cannot see has no def-use edge back to the read that produced it.
    A: Callable
    #: ``B(writer, var, j, k)``, same convention.
    B: Callable
    #: ``C(writer, value, i, j)``, writing one accumulator into the result.
    C: Callable
    #: ``sparse(k, j) -> bool``, or ``None`` where every entry is stored.
    #: Where it is not ``None``, ``B`` reads through the linear index instead
    #: of the coordinate one, so the two travel together.
    sparse: Optional[Callable]

    #: Register slots the lead dimension occupies per lane, and the unit the
    #: accessors index it in.
    lead_slots: int
    #: Elements the lead dimension spans.  ``lead_slots * threads`` rounded up
    #: from it -- a path that walks elements divides by :attr:`threads` before
    #: it reaches an accessor.
    lead_elements: int
    #: Extent of the remaining output indices, flattened.
    n: int
    #: Extent of the contraction, flattened.
    k: int
    #: Where the contraction starts.  ``k + kx`` is the depth a path walks.
    kx: int

    #: Lanes the lead dimension is spread over.
    threads: int
    #: Accumulator type.  Not the operand type: an emulated path splits its
    #: operands into a narrower one and this stays what the sum is kept in.
    dtype: Datatype


def scratch(dtype: Datatype) -> int:
    """Shared-memory elements a path needs, asked before anything is emitted.

    Every vendor module answers this, and the routing table asks the module it
    would dispatch to -- so the reservation and the emission are one decision
    rather than two that have to be kept in step.  A path that stages nothing
    answers 0, which is the default here.
    """
    return 0
