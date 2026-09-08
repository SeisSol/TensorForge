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

**Three types, not one.**  `A`, `B` and the accumulator each have their own,
mirroring the `a`, `b` and `d` fragments a catalogue entry carries.  They
coincide for everything the front end produces today, which is exactly why
collapsing them into a single field survives: nothing reads the difference
until something needs it, and then it is not there to read.

The type the instruction *multiplies in* is deliberately not among them.  An
emulated path splits an F32 operand into TF32 or BF16 terms and accumulates
their products; which substrate it picks is a choice it makes out of the
catalogue against `split_terms`, not something the caller can state.  Putting
it here would let a caller name an arithmetic the hardware does not have.

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

    #: Type `A` is read as.
    a: Datatype
    #: Type `B` is read as.
    b: Datatype
    #: Type the sum is kept in, and the one legality is asked about:
    #: `MatrixOp.available_for` matches an instruction by `d.dtype`, and the
    #: catalogue offers a BF16 entry for an F32 accumulator precisely because
    #: the two are different questions.
    accumulator: Datatype

    #: Scalars `A` occupies per logical element, and therefore how many parts
    #: its accessor has: `A(writer, var, i, k, part)` for `part` in
    #: `range(a_parts)`.
    #:
    #: One for every operand a frontend describes.  More where the generator
    #: decided to keep `A` *prepared* in memory --- two for a TF32 split, three
    #: for a BF16 one --- so that the parts a matrix instruction would compute
    #: are read instead.
    #:
    #: A count and not a field per part, which is the version this replaced:
    #: `A_lo` names "the second of two" and a three-part scheme would have
    #: needed an `A_lo2` beside it, and a four-part one another.  What the
    #: parts *mean* is the instruction mode's business --- the TF32 branch
    #: takes two and multiplies three products of them --- and what they are
    #: *numbered* is this.
    a_parts: int = 1


def scratch(dtype: Datatype) -> int:
    """Shared-memory elements a path needs, asked before anything is emitted.

    Every vendor module answers this, and the routing table asks the module it
    would dispatch to -- so the reservation and the emission are one decision
    rather than two that have to be kept in step.  A path that stages nothing
    answers 0, which is the default here.
    """
    return 0
