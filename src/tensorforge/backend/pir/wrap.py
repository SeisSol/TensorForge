# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
# SPDX-FileContributor: David Schneller

"""Pseudo-IR: the transfer for the next element, issued across the back edge.

The batch loop computes one element per iteration, and each iteration starts
by fetching its operands: the load waits at the first statement that reads
them, with nothing of the iteration left to overlap it with.  What could
overlap it is the previous iteration.  So this pass moves a transfer one
element ahead: iteration ``k`` issues the transfer for element ``k + 1`` once
it is done with the buffer, and the first element's transfer is peeled ahead
of the loop::

    before:   for k { [ l1 c1 l2 c2 ] }
    after:    l1(first)  for k { [ c1 l2 c2 ]  l1(k + 1) }

Where the moved transfer goes is decided by dependence and not by a count of
slots: at the tail of the body, after the per-element flag guard.  Behind
everything that reads its buffer, so one buffer is enough; and outside the
guard, so a masked element still fetches its successor -- element ``k`` being
masked says nothing about ``k + 1``, and a fetch under ``k``'s mask would leave
``k + 1`` without its operands.

*Which transfers* is ``distance``, the number of loads a transfer is moved
ahead by.  In the loop unrolled once, the ``j``-th transfer of an iteration
moves up past ``distance`` loads, and that runs off the top of the body into
the previous iteration exactly when ``j < distance``.  So the first
``distance`` transfers of the body wrap, in body order, and the rest stay.

*What a transfer is* is read off the accesses: a statement of the
per-element body that reads global memory and writes one register array or
one shared window, and nothing else.  The statements of one transfer follow
each other -- a hop loop and its predicated tail, the copies and the guard
around the last of them -- and move together, with the comments and the
`mark defines` that lead them.

*What may move.*  The transfer for ``k + 1`` comes from its place in
iteration ``k + 1``, so it crosses what stands ahead of it there, and then
everything of iteration ``k`` behind the buffer's last access -- which is
nothing it could conflict with, by construction.  So the questions are about
the stretch ahead of it in its own iteration:

* nothing there touches its buffer, or says nothing about what it touches;
* nothing there writes the memory it reads -- the element it fetches for
  ``k + 1`` is written in ``k + 1`` by that store, after the fetch would have
  read it.  Where the source does not depend on the element at all, any store
  to it in the body counts, since then every iteration reads the same data;
* for a shared buffer, no barrier there: it orders what other threads did to
  shared memory, and the buffer is shared;
* the buffer is written by this transfer alone, so that it holds one element
  for the whole iteration;
* and what the transfer reads can be computed for another element: the
  statements of the loop it depends on -- the pointer binding, the arithmetic
  -- are cloned with the next element's index for the tail and the first
  element's for the peel, so they have to be free of effects.

*What the moved transfer needs* besides its place:

* a register buffer declared outside the loop, since a declaration in the
  body is a fresh object every iteration;
* a shared buffer's copies retired where they are read: the loop carries the
  completion tokens across its back edge -- the peel's start it, the tail's
  are yielded -- and the wait at the consumer names the carried ones.  A
  masked element skips that wait and drains on its own path instead, or the
  tail would fill the buffer with an older copy still on its way into it; and
  the last iteration's copy, issued for an element nobody reads, is drained
  after the loop, since the memory it lands in may be the next section's;
* where the address is a pointer of the element's own, read out of an array,
  the transfer follows it only for an element the caller did not mask:
  under that element's flag, the next element's at the tail and the first's
  in the peel.  The array entry itself is read unconditionally, at a clamped
  index that is in range;
* and where the loop has a mask, the flags ride the loop.  Read at the head,
  an element's flag is a load its guard waits on at once, and where the
  counter retires loads in order that wait is for every transfer the previous
  tail issued as well.  So the flag words of this element and the next come
  in with the iteration, and the one two ahead is read at its head -- a load
  with a whole iteration to arrive.  The word and not the condition: a
  comparison right behind the load is a use right behind it.

*A second stage*, where `stages` asks for one, takes the tail out of the
picture for a shared transfer whose copies the hardware carries out on its
own.  At the tail the copy for ``k + 1`` has little ahead of its wait: what
stands at the head of ``k + 1`` in front of the first read.  With the buffer
twice over it goes to the head of ``k`` instead, into the stage ``k`` does not
read, and has the whole iteration::

    l1(first -> 0)  for k { l1(k + 1 -> 1 - s) [ c1(s) l2 c2 ]  s = 1 - s }

The stage ``s`` is a value the loop carries, not one computed from the
element: one thread's elements are a stride apart, and a stage taken from the
element would not alternate where two divides the stride.  The buffer becomes
three windows of one identity -- the peel's, into the first stage, and in the
loop the one the iteration reads and the one it fills -- whose offsets the
allocator completes with the stage.  The waits go ahead of the element guard:
behind a copy issued at the head, a masked element could not drain without
retiring that copy as well, and the two paths would leave different copies in
flight.  A copy written as loads and stores stalls on its loads where it is
issued, at the head as at the tail, and keeps one stage; so does a buffer used
outside the loop, whose windows could not follow the stage there.

Barriers and the buffer's place are not this pass's: the barrier placement
fences the tail's write against the reads before it and the reads at the head
of the next iteration against it (`barriers`), and the allocator keeps a
buffer in flight across the back edge live there (`allocate`).  The windows
of a buffer with two stages are one buffer to both: what the head of ``k``
fills is what ``k - 1`` read.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from tensorforge.common.basic_types import Datatype

from .asyncmem import strip_commits
from .core import (BOOL, INDEX, SCALAR_LAYOUT, SIZE, Access, Effect,
                   MemSpace, Op, Region, ScalarType, Stmt, Value, walk_stmts)

#: The effects a statement the pass computes again for another element may
#: not have.
_SIDE = Effect.WRITE | Effect.ATOMIC | Effect.BARRIER | Effect.UNKNOWN


class Refusal(Exception):
    """Why a transfer, or a loop, was left where it is.  Carried rather than
    logged: the caller asked for a transformation and is entitled to the
    reason it did not happen."""


def wrap_loads(body: Tuple[Stmt, ...], scratch: Callable[[], object], *,
               distance: int = 1, stages: int = 1,
               report: Optional[List[str]] = None) -> Tuple[Stmt, ...]:
    """`body` with the first `distance` transfers of each batch loop issued
    one element ahead.

    `scratch` makes a builder for the statements this adds -- the clones,
    the flag reads, the carried values -- whose values are new in `body`
    (`IRBuilder.scratch`).  `stages` is how many copies of a shared buffer a
    moved transfer may have: two moves it to the head of the body.  `report`
    collects one line per transfer: `+` and the buffer where it moved, `-`
    and the reason where it did not.

    A batch loop is a `for` that names its successor index and its first
    element (`next`, `first`).  A traversal that has no first element to
    name has no peel -- a group of rows driven in lockstep is one -- and is
    left alone.
    """
    if distance < 1:
        raise ValueError(f'move distance must be >= 1, got {distance}')
    if stages not in (1, 2):
        raise ValueError(f'a moved transfer has one stage or two, not {stages}')
    given = body
    # A commit records where a group closed in the schedule it was placed
    # for; the scheduler places them again behind this pass.
    body = strip_commits(body)
    allocs = {s.target[0].id: s for s in walk_stmts(body)
              if s.op is Op.ALLOC and s.target}
    out: List[Stmt] = []
    copied: List[Stmt] = []
    removed: set = set()
    rewritten = False
    for s in body:
        if (s.op is not Op.FOR or s.attr('next') is None
                or s.attr('first') is None):
            out.append(s)
            continue
        try:
            around = tuple(x for x in body if x is not s)
            rewrite = _Loop(s, allocs, scratch, report, stages,
                            around).wrap(distance)
            before, loop, after = rewrite.run()
        except Refusal as why:
            if report is not None:
                report.append(f'- loop: {why}')
            out.append(s)
            continue
        out.extend(before)
        out.append(loop)
        out.extend(after)
        copied.extend(d for p in rewrite.plans for d in p.deps)
        removed |= rewrite.removed
        rewritten = True
    if not rewritten:
        return given
    if removed:
        # The declarations of buffers given a second stage, which the loop
        # now declares itself.
        out = [x for x in out if id(x) not in removed]
    return _drop_unread(tuple(out), copied)


def _drop_unread(body: Tuple[Stmt, ...],
                 copied: Sequence[Stmt]) -> Tuple[Stmt, ...]:
    """`body` without the bindings of `copied` nothing names any more.

    A binding declares a name and so escapes: what reads it may be text, or
    an access that names the operand rather than the value, and neither is a
    use the graph sees.  So it goes only where none of the three reads it --
    no operand, no text but comments, no operand named like it.  Where the moved transfers
    were its only readers, it would otherwise be a declaration of a pointer
    nothing follows.
    """
    from .build import _names_in
    names = {id(s): str(s.attr('extern')) for s in copied
             if s.attr('extern') and len(s.target) == 1}
    if not names:
        return body
    by_value = {s.target[0].id: id(s) for s in copied if id(s) in names}
    read = set()
    for x in walk_stmts(body):
        if id(x) in names:
            continue
        read.update(by_value[v.id] for v in x.operands() if v.id in by_value)
        # A comment names what it describes, and reads nothing.
        texts = [] if (x.text or '').lstrip().startswith('//') else [x.text or '']
        texts += [a for a in x.args if isinstance(a, str)]
        texts += [str(getattr(a, 'name', '')) for a in x.args
                  if not isinstance(a, Value)]
        texts += [v for k, v in x.attrs if isinstance(v, str)]
        for sid, name in names.items():
            if sid not in read and any(_names_in(t, name) for t in texts if t):
                read.add(sid)
    gone = set(names) - read
    if not gone:
        return body

    def strip(stmts):
        out = []
        for s in stmts:
            if id(s) in gone:
                continue
            if s.regions:
                s = replace(s, regions=tuple(replace(r, body=strip(r.body))
                                             for r in s.regions))
            out.append(s)
        return tuple(out)
    return strip(body)


# --------------------------------------------------------------------------- #
# What a transfer is
# --------------------------------------------------------------------------- #

def _writes(a: Access) -> bool:
    return bool(a.kind & (Effect.WRITE | Effect.ATOMIC))


def _opaque(a: Access) -> bool:
    return a.space is MemSpace.UNKNOWN or a.base is None


def _same(x, y) -> bool:
    if isinstance(x, Value) and isinstance(y, Value):
        return x.id == y.id
    return x is y


def _neutral(s: Stmt) -> bool:
    """Text that touches nothing and defines nothing: a comment, a blank
    line.  It moves with the transfer it leads."""
    return (s.op is Op.RAWSTMT and not s.target and not s.args
            and not s.accesses and s.effect == Effect.NONE)


def _glue(s: Stmt) -> bool:
    """Arithmetic: no memory, no effect, no region."""
    return (s.pure and not s.regions and not s.accesses
            and s.effect == Effect.NONE and bool(s.target))


def _defines_whole(s: Stmt, dest: Value) -> bool:
    return (s.op is Op.MARK and s.attr('mark') == 'defines'
            and all(isinstance(a, Value) and a.id == dest.id for a in s.args))


def _fill_target(s: Stmt, fetched=frozenset()) -> Optional[Value]:
    """The buffer `s` fills, where `s` is a piece of a transfer.

    A piece reads global memory and writes one register array or one shared
    window, and touches nothing else -- whatever is inside it included.  A
    statement that reads a register or shared buffer computes rather than
    transfers, and one that says nothing about what it touches is not
    something this pass can account for.

    And what it writes is what it read: a copy, or a store of a value a load
    in the piece produced.  A reduction over global memory reads the same
    and writes the same, and is a computation all the same -- moving it
    would move the arithmetic with it, and count it among the transfers the
    distance is a number of.  `fetched` are values loaded from global memory
    by statements of their own, ahead of this one.
    """
    if s.op in (Op.WAIT, Op.COMMIT_ASYNC, Op.MARK, Op.ALLOC, Op.YIELD):
        return None
    dest = None
    fetches = False
    loaded = {t.id for x in walk_stmts((s,)) if x.op is Op.LOAD
              for t in x.target} | set(fetched)
    if fetched and s.op is Op.STORE:
        fetches = isinstance(s.args[1], Value) and s.args[1].id in fetched
    for x in walk_stmts((s,)):
        if (x.op in (Op.LOAD_ASYNC, Op.WAIT, Op.COMMIT_ASYNC, Op.MARK,
                     Op.ALLOC) or x.effect & Effect.BARRIER):
            return None
        for a in x.accesses:
            if _opaque(a):
                return None
            if _writes(a):
                if (a.kind & Effect.READ
                        or a.space not in (MemSpace.REGISTER, MemSpace.SHARED)
                        or not isinstance(a.base, Value)):
                    return None
                moves = x.op is Op.COPY_ASYNC or (
                    x.op is Op.STORE and isinstance(x.args[1], Value)
                    and x.args[1].id in loaded)
                if not moves:
                    return None
                if dest is None:
                    dest = a.base
                elif dest.id != a.base.id:
                    return None
            elif a.space is MemSpace.GLOBAL:
                fetches = True
            else:
                return None
    return dest if fetches else None


@dataclass
class _Transfer:
    """One transfer of the per-element body: the buffer it fills and its
    statements there, in order."""
    dest: Value
    stmts: List[Stmt] = field(default_factory=list)
    #: The statements that move data, without what leads them.
    pieces: List[Stmt] = field(default_factory=list)

    @property
    def shared(self) -> bool:
        return self.dest.type.space is MemSpace.SHARED

    def tokens(self) -> List[Value]:
        return [t for x in walk_stmts(tuple(self.pieces))
                if x.op is Op.COPY_ASYNC for t in x.target]


def _fetch(s: Stmt) -> bool:
    """A read of global memory into a value, standing on its own: the first
    half of a transfer whose second half is a store of it."""
    return (s.op is Op.LOAD and len(s.target) == 1 and not s.regions
            and bool(s.accesses) and all(
                a.space is MemSpace.GLOBAL and not _writes(a)
                for a in s.accesses))


def _transfers(scope: Sequence[Stmt]) -> List[_Transfer]:
    """The transfers of `scope`, in body order.

    A transfer written out without a loop is a load and a store per hop, each
    a statement of its own, and the loads belong to it as much as the stores
    -- as long as nothing else reads what they loaded.

    Arithmetic between two pieces of one transfer does not end it -- what
    `licm` lifts out of a hop loop lands there -- and stays where it is: a
    moved piece that reads it gets a copy of it (`_Loop._dependencies`).
    Anything else between two pieces ends the transfer, and the buffer is
    then filled by two, which no plan accepts.
    """
    uses: Dict[int, int] = {}
    for x in walk_stmts(tuple(scope)):
        for v in x.operands():
            uses[v.id] = uses.get(v.id, 0) + 1
    out: List[_Transfer] = []
    current: Optional[_Transfer] = None
    between: List[Stmt] = []
    for s in scope:
        fetched = {x.target[0].id: x for x in between if _fetch(x)}
        dest = _fill_target(s, frozenset(fetched))
        if dest is None:
            if _neutral(s) or _glue(s) or _fetch(s) or s.op is Op.MARK:
                between.append(s)
            else:
                current, between = None, []
            continue
        stored = {x.args[1].id for x in walk_stmts((s,))
                  if x.op is Op.STORE and isinstance(x.args[1], Value)}
        if any(uses.get(vid, 0) != 1 for vid in stored & set(fetched)):
            # It stores a value something else reads as well, and the load
            # cannot leave with it.
            current, between = None, []
            continue
        halves = {id(fetched[vid]) for vid in stored & set(fetched)}

        def joins(x):
            return id(x) in halves or _neutral(x) or _defines_whole(x, dest)
        if (current is not None and current.dest.id == dest.id
                and all(joins(x) or _glue(x) for x in between)):
            current.stmts.extend(x for x in between if joins(x))
            current.pieces.extend(x for x in between if id(x) in halves)
        else:
            lead: List[Stmt] = []
            for x in reversed(between):
                if joins(x):
                    lead.append(x)
                elif not _glue(x):
                    break
            lead.reverse()
            if len([x for x in lead if id(x) in halves]) != len(halves):
                # A half stands behind something that is not the transfer's.
                current, between = None, []
                continue
            current = _Transfer(dest)
            current.stmts.extend(lead)
            current.pieces.extend(x for x in lead if id(x) in halves)
            out.append(current)
        current.stmts.append(s)
        current.pieces.append(s)
        between = []
    return out


def _name(v: Value, alloc: Optional[Stmt]) -> str:
    extern = alloc.attr('extern') if alloc is not None else None
    return str(extern or v.hint or v)


def _line(p) -> str:
    """The report's line for a transfer that moves."""
    name = _name(p.transfer.dest, p.alloc)
    if not p.transfer.shared:
        return f'+ {name} [reg]'
    if p.rotate:
        return f'+ {name} [shr, 2 stages]'
    if p.single:
        return f'+ {name} [shr] one stage: {p.single}'
    return f'+ {name} [shr]'


# --------------------------------------------------------------------------- #
# One loop
# --------------------------------------------------------------------------- #

@dataclass
class _Plan:
    """A transfer the loop moves, and what moving it takes."""
    transfer: _Transfer
    alloc: Stmt
    #: Whether the allocation stands in the loop and leaves it.
    alloc_moves: bool
    #: The wait that retires its copies, where it has any.
    wait: Optional[Stmt]
    #: The statements of the loop it reads, to compute again for another
    #: element, in the order they run.
    deps: List[Stmt]
    #: Whether it follows a pointer of the element's own.
    owns_pointer: bool
    #: Whether its buffer gets a second stage and the transfer goes to the
    #: head of the body.
    rotate: bool = False
    #: Why it keeps one stage where two were asked for.
    single: Optional[str] = None


class _Loop:
    """A batch loop, taken apart: its body, the element guard in it and the
    per-element statements the guard holds."""

    def __init__(self, loop: Stmt, allocs: Dict[int, Stmt], scratch, report,
                 stages: int = 1, around: Sequence[Stmt] = ()):
        self.loop = loop
        self.allocs = allocs
        self.scratch = scratch
        self.report = report
        self.stages = stages
        #: The statements of the body around the loop.
        self.around = tuple(around)
        region = loop.regions[0]
        self.region = region
        self.k = region.args[0]
        self.next = loop.attr('next')
        self.first = loop.attr('first')
        self.flag_word = loop.attr('flag_word')
        guards = [i for i, s in enumerate(region.body)
                  if s.op is Op.IF and s.attr('guard') == 'element']
        if len(guards) > 1:
            raise Refusal('the loop has more than one element guard')
        self.guard_at = guards[0] if guards else None
        self.scope: Tuple[Stmt, ...] = (
            region.body[self.guard_at].regions[0].body
            if self.guard_at is not None else region.body)

    def _top(self) -> List[Tuple[Tuple[int, int], Stmt]]:
        """Every statement at the top of the loop's body or of the guard,
        keyed by the order it runs in."""
        out = []
        for i, s in enumerate(self.region.body):
            if i == self.guard_at:
                out.extend(((i, j + 1), x) for j, x in enumerate(self.scope))
            else:
                out.append(((i, 0), s))
        return out

    def _ahead_of(self, first: Stmt) -> List[Stmt]:
        """What stands ahead of `first` in its own iteration."""
        out = []
        for _, s in self._top():
            if s is first:
                return out
            out.append(s)
        raise AssertionError('the transfer is not at the top of the loop')

    # -- planning ---------------------------------------------------------- #

    def wrap(self, distance: int):
        transfers = _transfers(self.scope)
        if not transfers:
            raise Refusal('no transfer in the body')
        plans: List[_Plan] = []
        for t in transfers[:distance]:
            alloc = self.allocs.get(t.dest.id)
            try:
                plans.append(self._plan(t, alloc))
            except Refusal as why:
                if self.report is not None:
                    self.report.append(f'- {_name(t.dest, alloc)}: {why}')
        if not plans:
            raise Refusal('nothing to move')
        if self.report is not None:
            self.report.extend(_line(p) for p in plans)
        return _Rewrite(self, plans)

    def _plan(self, t: _Transfer, alloc: Optional[Stmt]) -> _Plan:
        d = t.dest
        if alloc is None:
            raise Refusal('the buffer is not allocated in this body')
        inside = any(s is alloc for s in walk_stmts(self.region.body))
        if inside and not any(s is alloc for _, s in self._top()):
            raise Refusal('the buffer is allocated inside a construct of the '
                          'body, so its declaration cannot leave the loop')
        if t.shared:
            identity = alloc.attr('identity')
            if alloc.attr('stages') or (identity is not None and any(
                    s is not alloc and s.attr('identity') is identity
                    for s in self.allocs.values())):
                raise Refusal('the buffer has several windows')
        elif alloc.attr('init') not in (None, '', '{}'):
            raise Refusal('the buffer is declared with an initializer, which '
                          'a declaration ahead of the loop would apply once')
        if any(x.predicate is not None for x in walk_stmts(tuple(t.pieces))
               if x.op is Op.COPY_ASYNC):
            raise Refusal('a copy of the transfer is predicated already')

        mine = {id(x) for x in walk_stmts(tuple(t.stmts))}
        tokens = t.tokens()
        wait = None
        if tokens:
            ids = {v.id for v in tokens}
            waits = [w for w in walk_stmts(self.region.body)
                     if w.op is Op.WAIT and any(isinstance(a, Value)
                                                and a.id in ids for a in w.args)]
            named = ({a.id for a in waits[0].args if isinstance(a, Value)}
                     if len(waits) == 1 else set())
            if (len(waits) != 1 or not ids <= named
                    or not any(w is waits[0] for w in self.scope)):
                raise Refusal(f'{len(tokens)} copies retired by {len(waits)} '
                              f'waits; a transfer is what one wait retires')
            wait = waits[0]

        for x in walk_stmts(self.region.body):
            if id(x) in mine or x is wait or x.op in (Op.MARK, Op.ALLOC):
                continue
            if any(_writes(a) and _same(a.base, d) for a in x.accesses):
                raise Refusal('the buffer is written by more than this '
                              'transfer, so it does not hold one element for '
                              'the whole iteration')

        deps, owns = self._dependencies(t)
        reads = [a.base for x in walk_stmts(tuple(t.pieces))
                 for a in x.accesses if a.space is MemSpace.GLOBAL]
        varies = any(self._names_index(s) for s in deps) or any(
            self._names_index(x) for x in walk_stmts(tuple(t.pieces)))

        for x in walk_stmts(tuple(self._ahead_of(t.stmts[0]))):
            for a in x.accesses:
                if _opaque(a):
                    raise Refusal('ahead of the transfer stands a statement '
                                  'that does not say what it touches')
                if _same(a.base, d):
                    raise Refusal('ahead of the transfer stands an access to '
                                  'its buffer, so it cannot leave its own '
                                  'iteration')
                if _writes(a) and any(_same(a.base, r) for r in reads):
                    raise Refusal('ahead of the transfer stands a write of '
                                  'what it reads, which it would read before '
                                  'the write')
            if t.shared and (x.op is Op.BARRIER or x.effect & Effect.BARRIER):
                raise Refusal('ahead of the transfer stands a barrier, which '
                              'orders the shared memory it writes')
        if not varies:
            for x in walk_stmts(self.region.body):
                if id(x) in mine:
                    continue
                if any(_writes(a) and any(_same(a.base, r) for r in reads)
                       for a in x.accesses):
                    raise Refusal('the body writes what the transfer reads, '
                                  'which is the same for every element')
        single = None
        if t.shared and self.stages > 1:
            single = self._single_stage(t, alloc, inside, tokens)
        return _Plan(t, alloc, inside, wait, deps, owns,
                     rotate=t.shared and self.stages > 1 and single is None,
                     single=single)

    def _single_stage(self, t: _Transfer, alloc: Stmt, inside: bool,
                      tokens: Sequence[Value]) -> Optional[str]:
        """Why `t` keeps one stage where two were asked for, or None.

        Its windows follow the stage the loop carries, so they are declared
        in the loop: the buffer has to be one the loop can take the
        declaration of -- declared at the top of it, or ahead of it in the
        same body and used nowhere else.
        """
        if not tokens:
            return ('its copies are loads and stores, which stall where they '
                    'are issued')
        if not inside and not any(s is alloc for s in self.around):
            return 'the buffer is declared where the loop cannot declare it'
        from .build import _names_in
        d = t.dest
        extern = alloc.attr('extern')
        for x in walk_stmts(tuple(s for s in self.around if s is not alloc)):
            if (any(v.id == d.id for v in x.operands())
                    or any(_same(a.base, d) for a in x.accesses)):
                return 'the buffer is used outside the loop'
            text = x.text or ''
            if extern and not text.lstrip().startswith('//') and _names_in(
                    text, str(extern)):
                return 'the buffer is named outside the loop'
        return None

    def _names_index(self, s: Stmt) -> bool:
        return any(v.id == self.k.id for v in s.operands())

    def _dependencies(self, t: _Transfer) -> Tuple[List[Stmt], bool]:
        """The statements of the loop's body `t` reads, transitively, in the
        order they run; and whether one of them reads an element's own
        pointer.

        Only what stands at the top of the body or of the guard: a value
        defined deeper is not visible to the transfer in the first place.
        The loop's own arguments are not statements: the index is replaced,
        and a value the loop carries belongs to the iteration, not to the
        element a clone is for.
        """
        defs: Dict[int, Tuple[Tuple[int, int], Stmt]] = {}
        for key, s in self._top():
            for v in s.target:
                defs[v.id] = (key, s)
        inner = set()
        for x in walk_stmts(tuple(t.pieces)):
            inner.update(v.id for v in x.target)
            for r in x.regions:
                inner.update(v.id for v in r.args)
        # The buffer it fills is not one of them: its declaration leaves the
        # loop whole, and is not computed again.
        inner.add(t.dest.id)
        carried = {v.id for v in self.region.args[1:]}
        wanted = [v.id for x in walk_stmts(tuple(t.pieces))
                  for v in x.operands() if v.id not in inner]
        found: Dict[int, Tuple[Tuple[int, int], Stmt]] = {}
        while wanted:
            vid = wanted.pop()
            if vid in carried:
                raise Refusal('the transfer reads a value the loop carries')
            hit = defs.get(vid)
            if hit is None or id(hit[1]) in found:
                continue
            key, s = hit
            if s.regions or (s.effect & _SIDE) or not s.movable:
                raise Refusal(f'the transfer reads a `{s.op}` of the body, '
                              f'which cannot be computed again for another '
                              f'element')
            if any(_writes(a) or _opaque(a) or a.space is not MemSpace.GLOBAL
                   for a in s.accesses):
                raise Refusal('the transfer reads a value loaded from memory '
                              'other than global, which another element '
                              'cannot read again')
            found[id(s)] = (key, s)
            wanted.extend(v.id for v in s.operands())
        deps = [s for _, s in sorted(found.values(), key=lambda e: e[0])]
        # What the dependencies load must not change under the loop.  A
        # binding is not such a load, though it declares a read of its
        # operand: strided addressing is arithmetic, and the array a pointer
        # is read out of is an argument the kernel never writes -- it writes
        # through the pointers, which its accesses cannot tell apart from the
        # array, being recorded against the operand either way.
        roots = [a.base for s in deps if s.op is Op.LOAD for a in s.accesses]
        for x in walk_stmts(self.region.body):
            if any(_writes(a) and any(_same(a.base, r) for r in roots)
                   for a in x.accesses):
                raise Refusal('the body writes memory the transfer\'s address '
                              'is computed from')
        return deps, any(s.attr('element_pointer') for s in deps)


# --------------------------------------------------------------------------- #
# Rewriting
# --------------------------------------------------------------------------- #

_IDENTIFIER = re.compile(r'\b[A-Za-z_]\w*\b')


class _Copy:
    """Statements of the loop again, for another element: every value they
    define new, every operand through what is known of the element, and
    every name spelled in text that changed spelled anew.

    Text is where a value can be named without being an operand -- a raw
    statement spelling its loop's index, a copy's address -- and a copy for
    another element that kept such a name would compute for the old one, or
    name a value of a scope it is not in.  So the spellings travel with the
    values: a value's own name, and a binding's, which a copy of it changes
    to `wrap_glb_m0` for the next element and `peel_glb_m0` for the first.
    """

    def __init__(self, fresh: Callable[[Value], Value], prefix: str,
                 taken: set):
        self._fresh = fresh
        self._prefix = prefix
        self._taken = taken
        self.mapping: Dict[int, Value] = {}
        self.spell: Dict[str, str] = {}

    def given(self, old: Value, new: Value) -> None:
        """`old` is `new` in what this copies."""
        self.mapping[old.id] = new
        self.spell[str(old)] = str(new)

    def _sub(self, x):
        if isinstance(x, Value):
            return self.mapping.get(x.id, x)
        if isinstance(x, str):
            return self._respell(x)
        return x

    def _respell(self, text: str) -> str:
        if not text or not self.spell:
            return text
        return _IDENTIFIER.sub(lambda m: self.spell.get(m.group(0), m.group(0)),
                               text)

    def stmts(self, stmts: Sequence[Stmt], named: bool = False) -> List[Stmt]:
        """`stmts` copied.  `named`: a copy of a declaration with a name of
        its own gets one of its own as well, and is declared the way the
        original is -- the backend's spelling of its type."""
        out = []
        for s in stmts:
            regions = []
            for r in s.regions:
                args = tuple(self._fresh(v) for v in r.args)
                for o, n in zip(r.args, args):
                    self.given(o, n)
                regions.append(Region(args=args, body=tuple(self.stmts(r.body))))
            target = tuple(self._fresh(v) for v in s.target)
            clone = replace(
                s, target=target, args=tuple(self._sub(a) for a in s.args),
                predicate=(self._sub(s.predicate) if s.predicate is not None
                           else None),
                regions=tuple(regions),
                text=self._respell(s.text) if s.text else s.text,
                accesses=tuple(replace(a, base=self._sub(a.base))
                               if isinstance(a.base, Value) else a
                               for a in s.accesses),
                attrs=tuple((k, v if k in ('decl', 'extern') else self._sub(v))
                            for k, v in s.attrs))
            for o, n in zip(s.target, target):
                self.given(o, n)
            if named and clone.attr('extern'):
                clone = self._name(clone, str(s.attr('extern')))
            out.append(clone)
        return out

    def rename(self, extern: str) -> str:
        """A name of its own for the copy of what is called `extern`, which
        the text this copies spells from then on."""
        name = base = f'{self._prefix}_{extern}'
        n = 1
        while name in self._taken:
            name = f'{base}_{n}'
            n += 1
        self._taken.add(name)
        self.spell[extern] = name
        return name

    def _name(self, s: Stmt, extern: str) -> Stmt:
        name = self.rename(extern)
        attrs = [(k, v) for k, v in s.attrs if k not in ('extern', 'decl')]
        attrs.append(('extern', name))
        decl = s.attr('decl')
        if isinstance(decl, str) and decl.rstrip().endswith(extern):
            attrs.append(('decl', decl.rstrip()[:-len(extern)] + name))
        return replace(s, attrs=tuple(attrs))


class _Rewrite:
    """A loop with its plans carried out: `(before, loop, after)`."""

    def __init__(self, loop: _Loop, plans: List[_Plan]):
        self.l = loop
        self.plans = plans
        self.b = loop.scratch()
        #: Statements ahead of the loop that go: the declarations of the
        #: buffers the loop declares itself, in their stages.
        self.removed: set = set()

    def _fresh(self, v: Value) -> Value:
        return self.b.value(v.type, hint=v.hint, uniform=v.uniformity,
                            layout=v.layout, quals=v.quals)

    # -- making statements ------------------------------------------------- #

    def _word(self, index, name: str) -> Tuple[List[Stmt], Value]:
        """`index`'s flag, read as the word it is stored as."""
        b = self.l.scratch()
        v = b.decl_expr(f'const uint32_t {name}', self.l.flag_word,
                        ScalarType(Datatype.U32), None, args=(index,),
                        kind=Effect.READ, space=MemSpace.GLOBAL, hint=name,
                        extern=name, layout=SCALAR_LAYOUT)
        return list(b.finish()), v

    def _flag(self, word, name: str) -> Tuple[List[Stmt], Value]:
        """A flag word as the condition it stands for."""
        b = self.l.scratch()
        v = b.decl_expr(f'const bool {name}', 'static_cast<bool>({0})', BOOL,
                        None, args=(word,), hint=name, extern=name)
        return list(b.finish()), v

    def _successor(self, index, hint: str) -> Tuple[List[Stmt], Value]:
        """`index` one stride on, clamped the way the loop clamps its own
        successor: the element a transfer issued there is for."""
        b = self.l.scratch()
        _, count, stride = self.l.loop.loop_bounds
        ahead = b.op('add', SIZE, index, stride, hint=f'{hint}Ahead')
        inside = b.op('lt', BOOL, ahead, count, hint=f'{hint}In')
        v = b.op('select', SIZE, inside, ahead, index, hint=hint)
        return list(b.finish()), v

    def _guarded(self, stmts: List[Stmt], cond: Value) -> List[Stmt]:
        """`stmts`, done only where `cond` holds.

        A copy takes it as its predicate rather than sitting in a block: a
        copy in a block is issued on one path only, as far as the count of
        groups in flight can tell, and the loop carrying its token would lose
        the steady state its waits are counted against.  The predicate is a
        branch per copy all the same, so a pointer is still not followed
        where it may not be.  Everything else goes into one block.
        """
        if not any(x.op is Op.COPY_ASYNC for x in walk_stmts(tuple(stmts))):
            b = self.l.scratch()
            with b.if_(cond):
                for s in stmts:
                    b.emit(s)
            return list(b.finish())

        def predicate(s: Stmt) -> Stmt:
            if s.op is Op.COPY_ASYNC:
                return replace(s, predicate=cond)
            if s.regions:
                return replace(s, regions=tuple(
                    replace(r, body=tuple(predicate(x) for x in r.body))
                    for r in s.regions))
            return s
        return [predicate(s) for s in stmts]

    def _windows(self, p: _Plan, stage: Value, following: Value,
                 peel_copy: '_Copy', tail_copy: '_Copy'
                 ) -> Tuple[Stmt, Stmt, Stmt]:
        """The windows of a buffer given a second stage: the peel's, into
        the stage the first iteration reads; and in the loop the one the
        iteration reads, which keeps the buffer's value, and the one it fills
        for the next element.  One buffer to whoever asks of its identity.
        """
        alloc = p.alloc
        dest = p.transfer.dest
        identity = alloc.attr('identity')
        if identity is None:
            identity = object()
        extern = alloc.attr('extern')

        def window(target, args, name):
            attrs = [(k, v) for k, v in alloc.attrs
                     if k not in ('identity', 'stages', 'extern')]
            attrs += [('identity', identity), ('stages', 2)]
            if name is not None:
                attrs.append(('extern', name))
            return replace(alloc, target=(target,), args=args,
                           attrs=tuple(attrs))

        peel_v, fill_v = self._fresh(dest), self._fresh(dest)
        peel_name = fill_name = None
        if extern:
            peel_name = peel_copy.rename(str(extern))
            fill_name = tail_copy.rename(str(extern))
        peel_copy.given(dest, peel_v)
        tail_copy.given(dest, fill_v)
        return (window(peel_v, (), peel_name),
                window(dest, (stage,), extern),
                window(fill_v, (following,), fill_name))

    # -- the loop ---------------------------------------------------------- #

    def run(self):
        l = self.l
        carry = l.flag_word is not None and l.guard_at is not None
        before: List[Stmt] = [p.alloc for p in self.plans
                              if p.alloc_moves and not p.rotate]

        # The flag words the loop starts with: the first element's, which
        # the peel's pointers are followed under, and its successor's.  They
        # are the carried values `own` and `nxt` in the body.
        own = nxt = first_word = None
        if carry:
            stmts, first_word = self._word(l.first, 'flagWordFirst')
            before += stmts
            stmts, second = self._successor(l.first, 'flagSecond')
            before += stmts
            stmts, second_word = self._word(second, 'flagWordSecond')
            before += stmts
            own = self.b.value(first_word.type, hint='flagWord',
                               layout=first_word.layout)
            nxt = self.b.value(second_word.type, hint='flagWordNext',
                               layout=second_word.layout)

        # The conditions a pointer of the element's own is followed under.
        peel_flag = tail_flag = None
        tail_flag_stmts: List[Stmt] = []
        if l.flag_word is not None and any(p.owns_pointer for p in self.plans):
            if first_word is None:
                stmts, first_word = self._word(l.first, 'flagWordFirst')
                before += stmts
            stmts, peel_flag = self._flag(first_word, 'allowed_peel')
            before += stmts
            if nxt is not None:
                tail_flag_stmts, tail_flag = self._flag(nxt, 'allowed_next')
            else:
                stmts, word = self._word(l.next, 'flagWordNext')
                flag_stmts, tail_flag = self._flag(word, 'allowed_next')
                tail_flag_stmts = stmts + flag_stmts

        taken: set = set()
        peel_copy = _Copy(self._fresh, 'peel', taken)
        peel_copy.given(l.k, l.first)
        tail_copy = _Copy(self._fresh, 'wrap', taken)
        tail_copy.given(l.k, l.next)

        # The stage the iteration reads, carried from 0, and the one it
        # fills; the windows into them at the top of the body, ahead of
        # everything that names the buffer.
        stage = following = zero = None
        top: List[Stmt] = []
        if any(p.rotate for p in self.plans):
            b = l.scratch()
            zero = b.const(0, INDEX)
            before += list(b.finish())
            stage = self.b.value(INDEX, hint='stage', uniform=l.k.uniformity)
            b = l.scratch()
            following = b.op('bitxor', INDEX, stage, 1, hint='stageNext')
            top += list(b.finish())
            fills: List[Stmt] = []
            for p in self.plans:
                if not p.rotate:
                    continue
                peel_win, read_win, fill_win = self._windows(
                    p, stage, following, peel_copy, tail_copy)
                before.append(peel_win)
                top.append(read_win)
                fills.append(fill_win)
                if not p.alloc_moves:
                    self.removed.add(id(p.alloc))
            top += fills
        # A pointer of the next element's own is followed under its flag,
        # which a transfer at the head reads there.
        flag_at_head = tail_flag is not None and any(
            p.rotate and p.owns_pointer for p in self.plans)
        done_peel: set = set()
        done_tail: set = set()
        heads: List[Stmt] = []
        tails_reg: List[Stmt] = []
        tails_shared: List[Stmt] = []
        waits: Dict[int, Stmt] = {}
        inits: List[Value] = []
        carried: List[Value] = []
        yields: List[Value] = []
        results: List[Value] = []
        after: List[Stmt] = []
        for p in self.plans:
            # The dependencies, once each: two transfers out of one pointer
            # share its copies.
            new = [s for s in p.deps if id(s) not in done_peel]
            before += peel_copy.stmts(new, named=True)
            done_peel.update(id(s) for s in new)
            new = [s for s in p.deps if id(s) not in done_tail]
            heads += tail_copy.stmts(new, named=True)
            done_tail.update(id(s) for s in new)

            peel = peel_copy.stmts(p.transfer.stmts)
            tail = tail_copy.stmts(p.transfer.stmts)
            if p.rotate:
                # The buffer is one, and the stage it reads is still wanted
                # when the other is filled: no point of the loop holds
                # nothing of it.
                tail = [x for x in tail if not (
                    x.op is Op.MARK and x.attr('mark') == 'defines')]
            if p.owns_pointer and peel_flag is not None:
                peel = self._guarded(peel, peel_flag)
                tail = self._guarded(tail, tail_flag)
            before += peel
            if p.rotate:
                if flag_at_head and tail_flag_stmts:
                    heads += tail_flag_stmts
                    tail_flag_stmts = []
                heads += tail
            else:
                (tails_shared if p.transfer.shared else tails_reg).extend(tail)
            if p.wait is None:
                continue

            # The tokens: the peel's are the first iteration's, the tail's
            # the next one's, and the wait names the iteration's own.
            originals = p.transfer.tokens()
            peel_tokens = [t for x in walk_stmts(tuple(peel))
                           if x.op is Op.COPY_ASYNC for t in x.target]
            tail_tokens = [t for x in walk_stmts(tuple(tail))
                           if x.op is Op.COPY_ASYNC for t in x.target]
            if not (len(originals) == len(peel_tokens) == len(tail_tokens)):
                raise Refusal('the peel and the tail issue different copies')
            args = [self._fresh(t) for t in originals]
            ends = [self._fresh(t) for t in originals]
            in_loop = {t.id: a for t, a in zip(originals, args)}
            past = {t.id: e for t, e in zip(originals, ends)}
            waits[id(p.wait)] = replace(p.wait, args=tuple(
                in_loop.get(a.id, a) if isinstance(a, Value) else a
                for a in p.wait.args))
            accesses = p.wait.accesses
            if p.rotate:
                # Behind the loop the buffer is the peel's window: the stage
                # the loop read is a window of the loop.
                peel_win = peel_copy.mapping[p.transfer.dest.id]
                accesses = tuple(replace(a, base=peel_win)
                                 if _same(a.base, p.transfer.dest) else a
                                 for a in accesses)
            after.append(Stmt(op=Op.WAIT, args=tuple(
                past.get(a.id, a) if isinstance(a, Value) else a
                for a in p.wait.args), pure=False, movable=True,
                effect=p.wait.effect, accesses=accesses))
            inits += peel_tokens
            carried += args
            yields += tail_tokens
            results += ends

        if carry:
            inits += [first_word, second_word]
            carried += [own, nxt]
            results += [self._fresh(own), self._fresh(nxt)]
        if stage is not None:
            inits.append(zero)
            carried.append(stage)
            results.append(self._fresh(stage))
        loop = self._body(heads, tails_reg, tails_shared, tail_flag_stmts,
                          waits, carry, own, nxt, yields, carried, inits,
                          results, top, following)
        return before, loop, after

    def _body(self, heads, tails_reg, tails_shared, tail_flag_stmts, waits,
              carry, own, nxt, yields, carried, inits, results, top,
              following) -> Stmt:
        l = self.l
        region = l.region
        moved = {id(s) for p in self.plans for s in p.transfer.stmts}
        moved |= {id(p.alloc) for p in self.plans if p.alloc_moves}
        # Behind a copy issued at the head, the masked path cannot drain: it
        # would retire the copy just issued as well, and the two paths would
        # hand the next iteration different copies in flight, which a count
        # of groups cannot follow.  So the waits go ahead of the guard, where
        # both paths pass them -- an iteration after a copy at the head was
        # issued, which is all the cover there is to have.
        lift = l.guard_at is not None and any(p.rotate for p in self.plans)
        lifted: List[Stmt] = []
        if lift:
            scope = list(l.scope)
            for p in self.plans:
                if p.wait is None:
                    continue
                # With the comment that says what it waits for.
                at = next(i for i, x in enumerate(scope) if x is p.wait)
                lead = at
                while (lead > 0 and _neutral(scope[lead - 1])
                       and id(scope[lead - 1]) not in moved):
                    lead -= 1
                lifted += scope[lead:at] + [waits[id(p.wait)]]
                moved.update(id(x) for x in scope[lead:at + 1])

        def keep(stmts):
            return [waits.get(id(s), s) for s in stmts if id(s) not in moved]

        body = list(region.body)
        terminator = region.terminator
        if terminator is not None:
            body.pop()

        # The head: the next element's dependencies -- its pointer, before
        # the whole body that hides its latency -- and the flag word two
        # elements ahead.
        flag_ahead: List[Stmt] = []
        word_ahead = None
        if carry:
            stmts, ahead = self._successor(l.next, 'flagAhead')
            word_stmts, word_ahead = self._word(ahead, 'flagWordAhead')
            flag_ahead = stmts + word_stmts

        if l.guard_at is None:
            new = keep(body)
            at = _behind_defs(new, heads + flag_ahead)
            new[at:at] = heads + flag_ahead
            rest = new
        else:
            guard = body[l.guard_at]
            pre = keep(body[:l.guard_at])
            cond = guard.cond
            if carry:
                # The guard reads the word the iteration came in with.  The
                # read of the element's flag it replaces goes, unless
                # something else reads it.
                stmts, allowed = self._flag(own, 'allowed')
                users = [x for x in walk_stmts(tuple(body)) if x is not guard
                         and any(v.id == cond.id for v in x.operands())]
                if not users:
                    pre = [s for s in pre
                           if not any(v.id == cond.id for v in s.target)]
                pre += stmts
                cond = allowed
            at = _behind_defs(pre, heads + flag_ahead)
            pre[at:at] = heads + flag_ahead
            pre += lifted
            regions = [replace(guard.regions[0],
                               body=tuple(keep(guard.regions[0].body)))]
            if waits and not lift:
                # A masked element skips the waits at its consumers, and what
                # they would have retired is still in flight into the buffer
                # the tail fills again -- and the copies of one thread to
                # one place land in no promised order, so the older one could
                # land last.  So it is retired on that path as well: a drain,
                # since a completion token is consumed once.
                drain = waits[next(iter(waits))]
                drain = Stmt(op=Op.WAIT, pure=False, movable=False,
                             effect=drain.effect)
                rest_else = (guard.regions[1].body if len(guard.regions) > 1
                             else ())
                regions.append(Region(args=(), body=tuple(rest_else)
                                      + (drain,)))
            guard = replace(guard, args=(cond,) + tuple(guard.args[1:]),
                            regions=tuple(regions))
            rest = keep(body[l.guard_at + 1:])
            new = pre + [guard]

        # The tail: register transfers ahead of the barrier that closes the
        # iteration, shared ones behind it, where it fences this iteration's
        # reads against the write.
        closing = max((i for i, s in enumerate(rest) if s.op is Op.BARRIER),
                      default=None)
        if closing is None:
            rest += tail_flag_stmts + tails_reg + tails_shared
        else:
            rest[closing + 1:closing + 1] = tails_shared
            rest[closing:closing] = tail_flag_stmts + tails_reg
        if l.guard_at is not None:
            new += rest
        new = top + new

        yielded = list(terminator.args if terminator is not None else ())
        yielded += yields
        if carry:
            yielded += [nxt, word_ahead]
        if following is not None:
            yielded.append(following)
        if yielded:
            new.append(replace(terminator, args=tuple(yielded))
                       if terminator is not None
                       else Stmt(op=Op.YIELD, args=tuple(yielded), pure=False,
                                 movable=False))
        return replace(
            l.loop,
            target=l.loop.target + tuple(results),
            args=l.loop.args + tuple(inits),
            regions=(Region(args=region.args + tuple(carried),
                            body=tuple(new)),))


def _behind_defs(stmts: List[Stmt], clones: Sequence[Stmt]) -> int:
    """The first position in `stmts` behind every statement that defines a
    value `clones` read."""
    wanted = {v.id for s in clones for x in walk_stmts((s,))
              for v in x.operands()}
    at = 0
    for i, s in enumerate(stmts):
        if any(v.id in wanted for v in s.target):
            at = i + 1
    return at
