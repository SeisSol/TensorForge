# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT

"""Pseudo-IR: where each buffer in shared memory goes.

A buffer is an `Op.ALLOC` with no offset; this pass hands the offsets out,
once the body is in its final order.  Before it, a pass that moves a transfer
moves the stretch its buffer is occupied, and one that gives a buffer a
second stage doubles it -- both decisions the placement has to follow rather
than precede.

Two arenas, as the hardware has them.  Each multiplication owns a copy of
one, and what is placed there may share bytes with any buffer it is never
occupied at the same time as.  The other is the block's, for what every
multiplication of a block reads -- an operator preloaded once, a staged
member of a merged run -- written ahead of everything that reads it and read
to the end, so it is laid out back to back, in the order it is allocated.

When a buffer is occupied is liveness over the body, with one thing the body
cannot say by itself: where a value of a buffer *ends*.  A write does not, in
general -- a buffer assembled from slices, or written in bursts of one element
per store, is still wanted for what the earlier writes put there -- and an
analysis that knew only the accesses would keep every buffer alive from its
first write to its last read, across every burst.  So the instruction that
writes a buffer whole says so, with a `mark defines` in front of its first
write: the buffer holds nothing anybody wants up to there.  An `Op.ALLOC`
says the same of a buffer that has no other window.

The answer is per statement and around every back edge: a buffer is occupied
where it is live -- read further on before something defines it anew --
and where it is touched.  Two buffers may share bytes when no statement
occupies both.  A raw block whose head the IR cannot read may run its body
any number of times, so it is treated as a loop, which can only add
occupancy; and a statement that does not say what it touches reads every
buffer, which keeps all of them where they are.  A buffer nothing touches
occupies nothing, and takes no room.

The layout is the largest buffer first, each at the lowest offset clear of
every buffer it is occupied together with, on the alignment every buffer
starts on.  Equal sizes keep the order of their allocation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .core import BufferType, Effect, MemSpace, Op, Stmt, Value, walk_stmts

#: The two arenas.
MULT = 'mult'
BLOCK = 'block'

#: What every buffer starts on, in bytes, and what one multiplication's arena
#: is padded to.  The arena itself is the launch's dynamic shared memory,
#: which the runtime gives at least this; what the placement adds is that
#: every buffer start and the per-multiplication stride keep it.  Without
#: that the guarantee is an accident of the sizes, and nothing that reads a
#: buffer wide could rely on it: `nvidia.matmul` stores through `float4`, a
#: wide read of a staged operand needs its base on 16 bytes, and
#: `MultilinearDescr` states this alignment for a temporary, which is the
#: generator's own storage.
SHARED_ALIGN_BYTES = 16


@dataclass
class Layout:
    """What the placement arrived at: the end of each arena, in elements,
    and where every buffer starts."""
    per_mult: int
    block: int
    offsets: Dict[Any, int]


class _Buffer:
    """One buffer: every window of it, its arena, how much it needs and on
    what it starts."""

    __slots__ = ('key', 'arena', 'size', 'align', 'index', 'windows')

    def __init__(self, key, arena, index):
        self.key = key
        self.arena = arena
        self.size = 0
        self.align = 1
        self.index = index
        self.windows: List[Stmt] = []


def _key(s: Stmt) -> Any:
    identity = s.attr('identity')
    if identity is not None:
        return ('o', id(identity))
    return ('v', s.target[0].id)


def _binds(s: Stmt) -> bool:
    return any(isinstance(t.type, BufferType) for t in s.target)


def _loops(s: Stmt) -> bool:
    if s.op in (Op.FOR, Op.WHILE):
        return True
    if s.op != Op.RAWBLOCK:
        return False
    head = (s.text or '').lstrip()
    return bool(re.match(r'(?:#pragma[^\n]*\n\s*)?(?:for|while)\b', head))


def _bits(mask: int):
    while mask:
        low = mask & -mask
        yield low.bit_length() - 1
        mask ^= low


class _Liveness:
    """Occupancy of the multiplication's buffers, as bit masks."""

    def __init__(self, buffers: Dict[Any, _Buffer], value_key: Dict[int, Any],
                 single: set):
        self.bit = {}
        for b in buffers.values():
            if b.arena == MULT:
                self.bit[b.key] = len(self.bit)
        self.value_key = value_key
        self.single = single
        self.everything = (1 << len(self.bit)) - 1
        self.neighbors = [0] * len(self.bit)
        self.touched = 0
        self.recording = False

    def _mask(self, base) -> int:
        """The buffer an access names, by its window or by the buffer a
        window is one of (`identity`)."""
        if isinstance(base, Value):
            key = self.value_key.get(base.id)
        else:
            key = ('o', id(base))
        bit = self.bit.get(key)
        return 0 if bit is None else 1 << bit

    def info(self, s: Stmt) -> Tuple[int, int, int]:
        """`(uses, defs, kills)` of a statement without regions."""
        if s.op == Op.ALLOC:
            if s.target and s.target[0].id in self.single:
                kill = self._mask(s.target[0])
                return 0, 0, kill
            return 0, 0, 0
        if s.op == Op.MARK:
            if s.attr('mark') == 'defines':
                kill = 0
                for a in s.args:
                    kill |= self._mask(a)
                return 0, 0, kill
            return 0, 0, 0
        if _binds(s):
            return 0, 0, 0
        uses = defs = 0
        for a in s.accesses:
            if a.space == MemSpace.UNKNOWN or (a.space == MemSpace.SHARED
                                               and a.base is None):
                uses |= self.everything
                continue
            if a.space != MemSpace.SHARED:
                continue
            m = self._mask(a.base)
            if not m:
                continue
            if a.kind & (Effect.WRITE | Effect.ATOMIC):
                defs |= m
            if a.kind & Effect.READ:
                uses |= m
        return uses, defs, 0

    def record(self, occupied: int) -> None:
        if not self.recording or not occupied:
            return
        self.touched |= occupied
        for b in _bits(occupied):
            self.neighbors[b] |= occupied

    def backward(self, body: Sequence[Stmt], live: int) -> int:
        for s in reversed(body):
            if s.regions:
                live = self.region(s, live)
                continue
            uses, defs, kills = self.info(s)
            self.record(live | uses | defs)
            live = (live & ~kills) | uses
        return live

    def region(self, s: Stmt, live_out: int) -> int:
        if not _loops(s):
            live = 0
            for r in s.regions:
                live |= self.backward(r.body, live_out)
            if s.op != Op.IF or len(s.regions) == 1:
                # a guard that is not taken, or a raw head that may not run
                live |= live_out
            return live
        recording, self.recording = self.recording, False
        head = live_out
        for _ in range(len(self.bit) + 2):
            entry = live_out
            for r in s.regions:
                entry |= self.backward(r.body, head)
            if entry == head:
                break
            head = entry
        self.recording = recording
        for r in s.regions:
            self.backward(r.body, head)
        return head


def _collect(body: Sequence[Stmt], arenas: Mapping[str, str],
             default_align: Mapping[str, int]):
    buffers: Dict[Any, _Buffer] = {}
    value_key: Dict[int, Any] = {}
    windows_of: Dict[Any, int] = {}
    for s in walk_stmts(body):
        if s.op != Op.ALLOC or not s.target:
            continue
        v = s.target[0]
        if getattr(v.type, 'space', None) != MemSpace.SHARED:
            continue
        arena = arenas.get(s.attr('arena'))
        if arena is None or s.attr('offset') is not None:
            continue
        key = _key(s)
        b = buffers.get(key)
        if b is None:
            b = buffers[key] = _Buffer(key, arena, len(buffers))
            b.align = default_align.get(arena, 1)
        b.windows.append(s)
        stages = s.attr('stages', 1) or 1
        b.size = max(b.size, v.type.volume * stages)
        b.align = max(b.align, s.attr('place_align', 1) or 1)
        value_key[v.id] = key
        windows_of[key] = windows_of.get(key, 0) + 1
    single = {s.target[0].id for b in buffers.values() for s in b.windows
              if windows_of[b.key] == 1}
    return buffers, value_key, single


def _aligned(n: int, align: int) -> int:
    return -(-n // align) * align


def allocate(body: Tuple[Stmt, ...], *, arenas: Mapping[str, str],
             align: int = 1, block_align: int = 1,
             report: Optional[List[str]] = None
             ) -> Tuple[Tuple[Stmt, ...], Layout]:
    """`body` with every unplaced shared buffer at an offset, and where the
    arenas end.

    ``arenas`` maps the arena an `Op.ALLOC` names to `MULT` or `BLOCK`;
    ``align`` is what every buffer of the multiplication's arena starts on,
    ``block_align`` the block's, both in elements -- a window may ask for more
    (`place_align`).
    """
    buffers, value_key, single = _collect(
        body, arenas, {MULT: max(1, align), BLOCK: max(1, block_align)})
    offsets: Dict[Any, int] = {}

    # The block's, back to back in the order allocated.
    end_block = 0
    for b in sorted((b for b in buffers.values() if b.arena == BLOCK),
                    key=lambda b: b.index):
        at = _aligned(end_block, b.align)
        offsets[b.key] = at
        end_block = at + b.size

    # The multiplication's, by occupancy.
    live = _Liveness(buffers, value_key, single)
    live.recording = True
    live.backward(body, 0)
    mult = [b for b in buffers.values() if b.arena == MULT]
    by_bit = {live.bit[b.key]: b for b in mult}
    placed: Dict[int, int] = {}
    end_mult = 0
    for b in sorted(mult, key=lambda b: (-b.size, b.index)):
        bit = live.bit[b.key]
        if not live.touched >> bit & 1:
            offsets[b.key] = 0
            continue
        taken = sorted((placed[n], placed[n] + by_bit[n].size)
                       for n in _bits(live.neighbors[bit])
                       if n != bit and n in placed)
        at = 0
        for lo, hi in taken:
            if at + b.size <= lo:
                break
            at = max(at, _aligned(hi, b.align))
        placed[bit] = at
        offsets[b.key] = at
        end_mult = max(end_mult, at + b.size)

    out = _place(body, value_key, offsets, buffers)
    return out, Layout(end_mult, end_block, offsets)


def _offset_of(s: Stmt, base: int, buffer: _Buffer):
    stage = s.attr('stage')
    if stage is None:
        return base
    size = s.target[0].type.volume
    return f'{base} + ({stage}) * {size}'


def _place(body: Tuple[Stmt, ...], value_key, offsets, buffers):
    out: List[Stmt] = []
    for s in body:
        if s.regions:
            s = replace(s, regions=tuple(
                replace(r, body=_place(r.body, value_key, offsets, buffers))
                for r in s.regions))
        elif s.op == Op.ALLOC and s.target:
            key = value_key.get(s.target[0].id)
            if key is not None and key in offsets:
                s = s.with_attr('offset', _offset_of(s, offsets[key],
                                                     buffers[key]))
        out.append(s)
    return tuple(out)
