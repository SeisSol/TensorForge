# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""What one chunk of a run leaves behind for the next.

:mod:`tensorforge.analysis.families` says which slices of a list are one slice
repeated.  That is the question of whether a run *may* be written once.  It
does not say what the repetition is allowed to become, and the two are not the
same question: a chain of steps and four independent contributions can have
identical skeletons and identical hole counts, and still one of them must stay
in order while the other need not.

The distinction that matters is not whether the chunks are independent but
what they share and how:

* nothing at all -- the chunks may be reordered, overlapped, or given to
  different waves, and a loop over them is a loop in name only;
* one destination that every chunk reads and writes -- an accumulation.  The
  chunks are independent in their inputs and joined only at the sum, so
  splitting them across waves is possible but costs a combine proportional to
  the accumulator, which for the operators this is aimed at is the largest
  object in the kernel;
* a value written by one chunk and read by a later one -- a recurrence.  Order
  is part of the meaning and there is nothing to overlap.

The three want different lowerings, and telling them apart is arithmetic on
the accesses rather than a judgement, so it belongs here rather than in
whoever ends up choosing.

Identity is object identity, deliberately.  Within one descriptor list two
chunks either name the same tensor or they do not, and a name-based comparison
would make two distinct temporaries that happen to share an alias look like a
dependence that is not there.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

from tensorforge.analysis.antiunify import _slots, _tensor_of
from tensorforge.generators.descriptions import (MultilinearDescr,
                                                 OperationDescription)


class Carried(Enum):
    """The ways a later chunk can depend on an earlier one."""

    FLOW = 'flow'        # written, then read -- a recurrence
    ANTI = 'anti'        # read, then overwritten
    OUTPUT = 'output'    # written twice


@dataclass(frozen=True)
class Dependence:
    """What holds between the chunks of a run.

    ``kinds`` is empty exactly when the chunks share nothing beyond their
    accumulators.  ``distance`` is the smallest chunk separation at which any
    dependence appears, which is what a pipeline may not cross and what a
    rotation has to be deep enough for.

    Accumulators are reported apart from the dependence rather than as one,
    because they are the case where the answer is neither yes nor no: the
    chunks are independent in everything except the sum, and whether that is
    worth splitting is a cost question about the accumulator's size.
    """

    kinds: FrozenSet[Carried]
    distance: Optional[int]
    accumulators: Tuple[str, ...]
    tensors: Tuple[str, ...]

    @property
    def independent(self) -> bool:
        """No chunk depends on another, and there is nothing to combine."""
        return not self.kinds and not self.accumulators

    @property
    def only_accumulation(self) -> bool:
        """Joined at the sum and nowhere else."""
        return not self.kinds and bool(self.accumulators)

    @property
    def ordered(self) -> bool:
        """The chunks mean what they mean only in this order."""
        return bool(self.kinds)


def _name(tensor) -> str:
    return (getattr(tensor, 'alias', None)
            or getattr(tensor, 'name', None)
            or f'<{id(tensor):x}>')


def accesses(descr: OperationDescription) -> Tuple[List, List]:
    """The tensors one descriptor reads and the ones it writes.

    A destination is read as well as written when the operation accumulates,
    which is the difference between a chunk that overwrites its output and one
    that adds to it -- and therefore the difference between an output
    dependence and an accumulator.
    """
    reads: List = []
    writes: List = []
    for role, view in _slots(descr):
        tensor = _tensor_of(view)
        if role == 'dest':
            writes.append(tensor)
            if isinstance(descr, MultilinearDescr) and descr.add:
                reads.append(tensor)
        else:
            reads.append(tensor)
    return reads, writes


def chunk_accesses(chunk: Sequence[OperationDescription]
                   ) -> Tuple[FrozenSet[int], FrozenSet[int], Dict[int, object]]:
    """One chunk's reads and writes, as sets of object ids, plus a name table."""
    reads: set = set()
    writes: set = set()
    table: Dict[int, object] = {}
    for descr in chunk:
        r, w = accesses(descr)
        for tensor in r:
            reads.add(id(tensor))
            table[id(tensor)] = tensor
        for tensor in w:
            writes.add(id(tensor))
            table[id(tensor)] = tensor
    return frozenset(reads), frozenset(writes), table


def carried(chunks: Sequence[Sequence[OperationDescription]]) -> Dependence:
    """What holds between the chunks of a run.

    Accumulators are taken out before the chunks are compared.  A destination
    that every chunk both reads and writes is the shape of a sum, and leaving
    it in would report a flow dependence at distance one for every accumulation
    -- true in the letter and useless, since it would put every reduction in
    the same class as a recurrence.
    """
    if len(chunks) < 2:
        return Dependence(frozenset(), None, (), ())

    per_chunk = [chunk_accesses(chunk) for chunk in chunks]
    table: Dict[int, object] = {}
    for _, _, part in per_chunk:
        table.update(part)

    accumulated = frozenset.intersection(
        *[reads & writes for reads, writes, _ in per_chunk])

    kinds: set = set()
    distance: Optional[int] = None
    involved: set = set()

    for earlier in range(len(chunks)):
        e_reads, e_writes, _ = per_chunk[earlier]
        e_reads -= accumulated
        e_writes -= accumulated
        for later in range(earlier + 1, len(chunks)):
            l_reads, l_writes, _ = per_chunk[later]
            l_reads -= accumulated
            l_writes -= accumulated
            found = {
                Carried.FLOW: e_writes & l_reads,
                Carried.ANTI: e_reads & l_writes,
                Carried.OUTPUT: e_writes & l_writes,
            }
            for kind, overlap in found.items():
                if overlap:
                    kinds.add(kind)
                    involved |= overlap
                    gap = later - earlier
                    distance = gap if distance is None else min(distance, gap)

    return Dependence(
        frozenset(kinds), distance,
        tuple(sorted(_name(table[i]) for i in accumulated)),
        tuple(sorted(_name(table[i]) for i in involved)))


def binding_period(bindings: Sequence[object]) -> Optional[int]:
    """The stride at which a hole starts naming the same tensors again.

    A hole whose bindings repeat every ``p`` chunks needs ``p`` buffers and a
    rotation; one whose bindings never repeat needs as many as there are
    chunks, and is an indexed array rather than a rotation.  The two lower
    very differently, and which one a run wants is not visible in the run's
    shape -- only in its table.
    """
    names = [id(_tensor_of(view)) for view in bindings]
    if len(names) < 2:
        return None
    for period in range(1, len(names)):
        if all(names[i] == names[i + period]
               for i in range(len(names) - period)):
            return period
    return None


def escapes(descrs: Sequence[OperationDescription],
            start: int, stop: int) -> Tuple[str, ...]:
    """Tensors written inside ``[start, stop)`` whose value is observable after.

    Two ways to be observable, and both have to be counted.  One is a read
    later in the same list.  The other is being a tensor the caller passed in:
    writing it is the point of passing it, so its value outlives the kernel
    whether or not anything else in the list touches it, and only a temporary
    the generator invented for itself is free of that.

    Counting only the first is how a rotation gets chosen for a sequence whose
    outputs are the reason the kernel exists.  A recursion whose steps each
    write a buffer the caller owns has every step observable, even when the
    list itself never reads one of them again.
    """
    written: Dict[int, object] = {}
    for descr in descrs[start:stop]:
        _, w = accesses(descr)
        for tensor in w:
            written[id(tensor)] = tensor

    later: set = set()
    for descr in descrs[stop:]:
        r, _ = accesses(descr)
        later |= {id(tensor) for tensor in r}

    return tuple(sorted(_name(tensor) for key, tensor in written.items()
                        if key in later or not getattr(tensor, 'is_tmp', False)))


def shifts(bindings: Sequence[Sequence[object]],
           least_overlap: int = 2) -> Tuple[Tuple[int, int, int], ...]:
    """Which hole's table is another hole's table, moved along.

    ``(source, target, distance)`` says that what hole ``source`` names at
    iteration ``k`` is what hole ``target`` names at iteration ``k + distance``
    -- the shape of a recurrence, where each step reads what the one before it
    wrote.

    Worth finding for two reasons.  A table with a shifted column in it is a
    table with a redundant column: only the independent ones have to be
    carried, and the rest are the same values read at an offset.  And the
    relation is the recurrence itself made visible, which no per-iteration
    fact can be -- within one iteration the two are simply different tensors.

    ``least_overlap`` is how many positions must agree before this counts.  A
    single coincidence between two short tables says nothing.
    """
    if not bindings:
        return ()
    arity = len(bindings[0])
    found: List[Tuple[int, int, int]] = []
    for distance in range(1, len(bindings)):
        overlap = len(bindings) - distance
        if overlap < least_overlap:
            break
        for source in range(arity):
            for target in range(arity):
                if source == target:
                    continue
                if all(_tensor_of(bindings[k][source])
                       is _tensor_of(bindings[k + distance][target])
                       for k in range(overlap)):
                    found.append((source, target, distance))
    return tuple(found)
