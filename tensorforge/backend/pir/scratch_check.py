# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Is the scratch packing this body declared consistent with what it does?

`IRBuilder.scratch_scope` states a lifetime by nesting, and the suballocator
turns that into offsets --- so two buffers in sibling scopes share space.  That
is a claim, and until now nothing checked it.

The obvious ambition was to derive the packing instead, from a liveness
analysis over the body, and retire the declaration.  That does not work here,
and the reason is worth writing down rather than rediscovering.

`matmul`'s loops are unrolled by the Python that emits them, so every
iteration is laid out in sequence.  The C tile is written and read in
iteration `i`'s epilogue, then again in `i+1`'s, with the A and B staging of
`i+1` in between --- so a live range running from C's first touch to its last
covers A's and B's, and all three interfere.  320 elements, against the 192
the declaration achieves.  The analysis is not wrong; a single interval is
just the wrong shape for a buffer used in bursts.

Splitting the range at a write does not rescue it either.  What makes the
reuse safe is that each burst *fully* overwrites the tile before reading it,
and no single store does that: the C tile is 128 elements filled by 48
separate writes.  Proving the union covers the buffer is an index-set
analysis, which is a great deal of machinery to re-derive a fact the emitter
knew all along.

So this module checks instead of deriving.  It looks for the error that is
actually available to make --- a buffer read after another buffer has written
over its space, with nothing rewriting it in between --- which is decidable
from the accesses already declared, and is what a mis-nested `scratch_scope`
produces.  It is a necessary condition, not a sufficient one: a partial
rewrite between the clobber and the read would satisfy it and still be wrong.
`check_reuse` says so rather than implying a proof it does not have.

When the micro allocator eventually takes over the macro one, this is the part
that survives.  Deriving offsets will need the coverage analysis; refusing a
packing that is visibly unsafe does not, and is useful immediately.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

from .core import Effect, MemSpace, Op, Stmt, Value


@dataclass(frozen=True)
class Touch:
    position: int
    buffer: Value
    reads: bool
    writes: bool


@dataclass(frozen=True)
class Window:
    """Where a buffer sits in the arena, in elements."""
    start: int
    length: int

    @property
    def end(self) -> int:
        return self.start + self.length

    def overlaps(self, other: 'Window') -> bool:
        return self.start < other.end and other.start < self.end


def _walk(body: Sequence[Stmt], pos: List[int]) -> Iterator[Tuple[int, Stmt]]:
    for stmt in body:
        here = pos[0]
        pos[0] += 1
        yield here, stmt
        for region in stmt.regions:
            yield from _walk(region.body, pos)


def windows(body: Sequence[Stmt]) -> Dict[int, Window]:
    """The offsets the suballocator handed out, read back off the body."""
    out: Dict[int, Window] = {}
    for _, stmt in _walk(body, [0]):
        if stmt.op != Op.ALLOC:
            continue
        attrs = dict(stmt.attrs) if stmt.attrs else {}
        offset = attrs.get('offset')
        for t in stmt.target:
            if getattr(t.type, 'space', None) == MemSpace.SHARED:
                out[id(t)] = Window(offset if offset is not None else 0,
                                    t.type.volume)
    return out


def touches(body: Sequence[Stmt], keys) -> Tuple[List[Touch], List[int]]:
    """Every access to a tracked buffer, in emission order.

    The second return value is the positions of statements that declined to
    say what they touch.  Those have to be read as touching everything, and a
    single one of them makes the whole check vacuous --- which is worth
    reporting rather than hiding, because the answer then is not "safe" but
    "not asked".
    """
    found: List[Touch] = []
    opaque: List[int] = []
    for here, stmt in _walk(body, [0]):
        if stmt.op == Op.ALLOC:
            continue                    # where a buffer comes from, not a use
        for a in stmt.accesses:
            if a.space not in (MemSpace.SHARED, MemSpace.UNKNOWN):
                continue
            if a.base is None or a.space == MemSpace.UNKNOWN:
                opaque.append(here)
                continue
            if id(a.base) in keys:
                found.append(Touch(here, a.base,
                                   bool(a.kind & Effect.READ),
                                   bool(a.kind & Effect.WRITE)))
    return found, opaque


@dataclass(frozen=True)
class Violation:
    read_of: Value
    at: int
    clobbered_by: Value
    clobbered_at: int

    def __str__(self) -> str:
        return (f'{self.read_of} is read at {self.at}, but {self.clobbered_by} '
                f'wrote over its space at {self.clobbered_at} and nothing '
                f'rewrote {self.read_of} in between')


def check_reuse(body: Sequence[Stmt]) -> Tuple[List[Violation], List[int]]:
    """Buffers sharing space must not be read across each other's writes.

    For every pair whose windows overlap: walk the accesses in order, and if a
    read of one follows a write of the other with no intervening write of the
    one, the packing is reading data that is no longer there.

    Returns the violations and the positions of any statements that declined
    to declare their accesses.  A non-empty second list means the first is not
    a clean bill of health.
    """
    win = windows(body)
    if len(win) < 2:
        return [], []
    all_touches, opaque = touches(body, set(win))
    by_id = {t.buffer and id(t.buffer): t.buffer for t in all_touches}

    violations: List[Violation] = []
    ids = sorted(win, key=lambda b: (win[b].start, win[b].length))
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            if not win[a].overlaps(win[b]):
                continue
            last_write: Dict[int, Optional[int]] = {a: None, b: None}
            for t in all_touches:
                k = id(t.buffer)
                if k not in (a, b):
                    continue
                other = b if k == a else a
                if t.reads and last_write[other] is not None:
                    mine = last_write[k]
                    if mine is None or mine < last_write[other]:
                        violations.append(
                            Violation(t.buffer, t.position,
                                      by_id[other], last_write[other]))
                        last_write[other] = None      # report once per burst
                if t.writes:
                    last_write[k] = t.position
    return violations, opaque
