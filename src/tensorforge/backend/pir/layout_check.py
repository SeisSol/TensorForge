# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Is the shared-memory layout of a body consistent with what it does?

The allocator (`allocate`) puts two buffers on the same bytes where no
statement occupies both, and it learns where a value of a buffer ends from
the instruction that writes it: a `mark defines` in front of its first write
says the buffer holds nothing anybody wants up to there.  That is a claim the
emitter makes, and a layout derived from it is as good as the claim.

So this checks the layout against the accesses alone, with the marks left
out.  What it looks for is the error a wrong claim produces -- a buffer read
after another buffer wrote over its bytes, with nothing rewriting it in
between -- which is decidable from the accesses already declared.

It is a necessary condition, not a sufficient one.  A partial rewrite between
the clobber and the read satisfies it and may still be wrong, and the walk is
in emission order, so what a loop carries around its back edge is not
followed.  `check_layout` returns the statements that did not say what they
touch alongside the violations, so that no violations cannot be read as safe
when it means not asked.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from .core import Effect, MemSpace, Op, Stmt, Value


@dataclass(frozen=True)
class Touch:
    position: int
    buffer: Any
    reads: bool
    writes: bool


@dataclass(frozen=True)
class Window:
    """Where a buffer sits in its arena, in elements.

    `arena` is part of the position: offsets are counted from each arena's
    own base, so two buffers with overlapping offsets in different arenas do
    not share a byte.
    """
    start: int
    length: int
    arena: Optional[str] = None

    @property
    def end(self) -> int:
        return self.start + self.length

    def overlaps(self, other: 'Window') -> bool:
        if self.arena != other.arena:
            return False
        return self.start < other.end and other.start < self.end


def _walk(body: Sequence[Stmt], pos: List[int]) -> Iterator[Tuple[int, Stmt]]:
    for stmt in body:
        here = pos[0]
        pos[0] += 1
        yield here, stmt
        for region in stmt.regions:
            yield from _walk(region.body, pos)


def _key(stmt: Stmt) -> Any:
    """The buffer a window is one of: what it names as its identity, or the
    window itself."""
    identity = stmt.attr('identity')
    return ('o', id(identity)) if identity is not None \
        else ('v', stmt.target[0].id)


def windows(body: Sequence[Stmt]) -> Tuple[Dict[Any, Window], List[Any]]:
    """Where every shared buffer of the body sits, read back off its
    allocations, keyed by buffer; and the buffers that sit nowhere yet.

    A buffer with several stages takes all of them, from the offset of its
    first: a window into stage `s` names it as ``base + (s) * size``.
    """
    out: Dict[Any, Window] = {}
    unplaced: List[Any] = []
    for _, stmt in _walk(body, [0]):
        if stmt.op != Op.ALLOC or not stmt.target:
            continue
        value = stmt.target[0]
        if getattr(value.type, 'space', None) != MemSpace.SHARED:
            continue
        key = _key(stmt)
        offset = stmt.attr('offset')
        if isinstance(offset, str):
            m = re.match(r'\s*(\d+)\s*\+', offset)
            offset = int(m.group(1)) if m else None
        if offset is None:
            unplaced.append(key)
            continue
        stages = stmt.attr('stages', 1) or 1
        found = Window(offset, value.type.volume * stages, stmt.attr('arena'))
        known = out.get(key)
        if known is not None:
            found = Window(min(known.start, found.start),
                           max(known.end, found.end) - min(known.start,
                                                           found.start),
                           found.arena)
        out[key] = found
    return out, unplaced


def touches(body: Sequence[Stmt], keys) -> Tuple[List[Touch], List[int]]:
    """Every access to a tracked buffer, in emission order, and the
    positions of the statements that did not say what they touch.

    Those have to be read as touching everything, and a single one makes the
    whole check vacuous -- which is worth reporting rather than hiding,
    because the answer then is not "safe" but "not asked".  A mark is a
    claim and not an access, so it is no touch.
    """
    value_key: Dict[int, Any] = {}
    for _, stmt in _walk(body, [0]):
        if stmt.op == Op.ALLOC and stmt.target:
            value_key[stmt.target[0].id] = _key(stmt)
    found: List[Touch] = []
    opaque: List[int] = []
    for here, stmt in _walk(body, [0]):
        if stmt.op in (Op.ALLOC, Op.MARK):
            continue
        for a in stmt.accesses:
            if a.space not in (MemSpace.SHARED, MemSpace.UNKNOWN):
                continue
            if a.base is None or a.space == MemSpace.UNKNOWN:
                opaque.append(here)
                continue
            key = (value_key.get(a.base.id) if isinstance(a.base, Value)
                   else ('o', id(a.base)))
            if key in keys:
                found.append(Touch(here, key, bool(a.kind & Effect.READ),
                                   bool(a.kind & (Effect.WRITE
                                                  | Effect.ATOMIC))))
    return found, opaque


@dataclass(frozen=True)
class Violation:
    read_of: str
    at: int
    clobbered_by: str
    clobbered_at: int

    def __str__(self) -> str:
        return (f'{self.read_of} is read at {self.at}, but {self.clobbered_by} '
                f'wrote over its bytes at {self.clobbered_at} and nothing '
                f'rewrote {self.read_of} in between')


def _names(body: Sequence[Stmt]) -> Dict[Any, str]:
    out: Dict[Any, str] = {}
    for _, stmt in _walk(body, [0]):
        if stmt.op == Op.ALLOC and stmt.target:
            out.setdefault(_key(stmt), str(stmt.target[0]))
    return out


def check_layout(body: Sequence[Stmt]) -> Tuple[List[Violation], List[Any]]:
    """Buffers sharing bytes must not be read across each other's writes.

    For every pair whose windows overlap: walk the accesses in order, and if
    a read of one follows a write of the other with no write of the one in
    between, the layout has it read data that is no longer there.

    Returns the violations, and the positions of the statements that did not
    say what they touch together with the buffers that have no place yet.  A
    non-empty second list means the first is not a clean bill of health.
    """
    win, unplaced = windows(body)
    if len(win) < 2:
        return [], list(unplaced)
    all_touches, opaque = touches(body, set(win))
    by_buffer: Dict[Any, List[Touch]] = {}
    for t in all_touches:
        by_buffer.setdefault(t.buffer, []).append(t)
    names = _names(body)
    violations: List[Violation] = []
    keys = sorted(win, key=lambda k: (win[k].arena or '', win[k].start,
                                      win[k].length, str(k)))
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            if not win[a].overlaps(win[b]):
                continue
            # In order, and within one statement a read ahead of a write:
            # a statement reads its operands before it stores its result.
            pair = sorted(by_buffer.get(a, []) + by_buffer.get(b, []),
                          key=lambda t: (t.position, t.writes and not t.reads))
            last_write: Dict[Any, Optional[int]] = {a: None, b: None}
            for t in pair:
                other = b if t.buffer == a else a
                if t.reads and last_write[other] is not None:
                    mine = last_write[t.buffer]
                    if mine is None or mine < last_write[other]:
                        violations.append(Violation(
                            names[t.buffer], t.position, names[other],
                            last_write[other]))
                        last_write[other] = None   # once per burst
                if t.writes:
                    last_write[t.buffer] = t.position
    return violations, list(opaque) + list(unplaced)
