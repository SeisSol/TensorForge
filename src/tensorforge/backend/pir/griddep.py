# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT

"""Pseudo-IR: where a kernel waits for the grid launched ahead of it
(Programmatic Dependent Launch, `Options.programmatic_launch`).

A kernel launched with `cudaLaunchAttributeProgrammaticStreamSerialization`
may start while the work ahead of it on the stream still runs: its blocks are
scheduled as the blocks of the grid before exit, and `griddepcontrol.wait`
(`engines.grid_wait`) holds a thread until that grid has completed and what it
wrote is visible.  What a thread does ahead of the wait overlaps the tail of
the grid before -- and may read nothing a kernel of the run writes, since PTX
promises the other grid's writes only behind the wait.

So `place_wait` puts the wait ahead of the first statement of a body that
touches anything but what is free:

* free is an operand the kernel reads and does not write that is the batch's
  (`Addressing.NONE`: one buffer for every element, the operators -- and the
  multi-element layout's stage of them) or that its caller states unchanged
  over the run (yateto's `temporal: invariant`, `Tensor.time_invariant`); a
  merged run's stand-in where every member is;
* a global access of a base that names no operand, or in a space the access
  does not name, is not free, and nor is opaque text (`Effect.UNKNOWN`) or
  text its builder states reads global memory (`text_reads`: the flag words,
  which an earlier kernel may compute).

At the body's level, ahead of the loop or the guard the access stands in and
never inside, so that every thread of every block reaches it.  A pointer table
is read through the operand's own base, so the per-element pointers are read
behind the wait as well: the conservative answer, and the measured one (the
loads a wait further down let overlap were worth 0 to 2 % of a chain).  Every
kernel writes an output, so every kernel waits before it writes, and a chain
of such launches is ordered as one of plain launches is.

Placed behind every pass, by the generator, in the first section that touches
anything not free (`Generator._place_grid_wait`): the sections run one after
the other, so behind the first wait the rest of the kernel is covered.
"""

from __future__ import annotations

from typing import FrozenSet, Iterable, Optional, Sequence, Tuple

from tensorforge.common.basic_types import Addressing

from . import engines
from .core import (Effect, MemSpace, Stmt, collect_accesses, collect_effect,
                   walk_stmts)

_M_WRITES = int(Effect.WRITE | Effect.ATOMIC)
_SPACES = (MemSpace.GLOBAL, MemSpace.UNKNOWN)


def _tensors(base) -> Optional[Tuple]:
    """The tensors an access of `base` lands in: the operand's own, with the
    members of a merged run where it is their stand-in; None where the base
    names no operand."""
    obj = getattr(base, 'obj', None)
    if obj is None:
        return None
    return (obj,) + tuple(getattr(obj, 'variant_members', ()) or ())


def written(bodies: Iterable[Sequence[Stmt]]) -> FrozenSet[int]:
    """The ids of the tensors a statement of `bodies` writes: no tensor the
    kernel writes is free, whatever its caller states."""
    out = set()
    for body in bodies:
        for a in collect_accesses(tuple(body)):
            if int(a.kind) & _M_WRITES:
                out.update(id(t) for t in _tensors(a.base) or ())
    return frozenset(out)


def free(tensor, writes: FrozenSet[int]) -> bool:
    """Whether a kernel that writes `writes` may read `tensor` ahead of the
    wait: the batch's, or stated unchanged over the run, and not its own
    output."""
    if id(tensor) in writes:
        return False
    addressing = getattr(tensor, 'addressing', None)
    if addressing is not None and addressing.kind is Addressing.NONE:
        return True
    return bool(getattr(tensor, 'time_invariant', False))


def touches(s: Stmt, writes: FrozenSet[int]) -> bool:
    """Whether `s`, or anything in its regions, touches memory an earlier
    kernel may write."""
    if collect_effect((s,)) & Effect.UNKNOWN:
        return True
    if any(x.attr('text_reads') in _SPACES for x in walk_stmts((s,))):
        return True
    for a in collect_accesses((s,)):
        if a.space not in _SPACES:
            continue
        tensors = _tensors(a.base)
        if not tensors or not all(free(t, writes) for t in tensors):
            return True
    return False


def place_wait(body: Sequence[Stmt], writes: FrozenSet[int]
               ) -> Tuple[Tuple[Stmt, ...], bool]:
    """`body` with the wait (`engines.grid_wait`) ahead of its first
    statement that touches anything not free, and whether it got one -- not
    where no statement does."""
    body = tuple(body)
    for i, s in enumerate(body):
        if touches(s, writes):
            return body[:i] + (engines.grid_wait(),) + body[i:], True
    return body, False


__all__ = ['free', 'place_wait', 'touches', 'written']
