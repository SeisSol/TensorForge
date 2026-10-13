# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
# SPDX-FileContributor: David Schneller

"""Pseudo-IR: what a transfer is.

The passes that issue a transfer ahead of where it was written -- within the
iteration (`move`), and across the batch loop's back edge (`wrap`) -- agree
on what one is, and read it off the accesses: a statement that reads global
memory and writes one register array or one shared window, and nothing else.
The statements of one transfer follow each other -- a hop loop and its
predicated tail, the copies and the guard around the last of them -- and
move together, with the comments and the `mark defines` that lead them.  A
transfer written out without a loop is a load and a store per hop, and the
loads belong to it as much as the stores.

The wait that retires a copy the hardware carries out on its own is not part
of it: the wait stays where the first read is, and the distance to it is what
issuing the transfer early is for.

A loader tags the statements of its transfer (`IRBuilder.transfer`), and a
transfer is one tag: two loaders that fill one buffer in a row write two
transfers, which the accesses alone would read as one.  What a statement may
be within a transfer is still read off its accesses -- a tagged statement
that, after the passes, reads something besides global memory ends the
transfer as an untagged one does -- and a body nobody tagged is read off its
accesses alone.

How far a transfer is issued ahead is counted in transfers of its own
pipeline, and what a pipeline is, is a key over the transfer
(`PIPELINES`): by default the memory path it takes, so that a copy into
shared memory and a load into registers do not hold each other back.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence

from .core import Access, Effect, MemSpace, Op, Stmt, Value, walk_stmts



def tag(s: Stmt) -> Optional[int]:
    """The transfer `s` was written for (`IRBuilder.transfer`), if any."""
    return s.attr('transfer')


def tagged(s: Stmt, transfer: int) -> Stmt:
    """`s` as a statement of the transfer `transfer`, unless it is one of
    another already."""
    if tag(s) is not None:
        return s
    return replace(s, attrs=s.attrs + (('transfer', transfer),))


def writes(a: Access) -> bool:
    return bool(a.kind & (Effect.WRITE | Effect.ATOMIC))


def opaque(a: Access) -> bool:
    return a.space is MemSpace.UNKNOWN or a.base is None


def same(x, y) -> bool:
    if isinstance(x, Value) and isinstance(y, Value):
        return x.id == y.id
    return x is y


def neutral(s: Stmt) -> bool:
    """Text that touches nothing and defines nothing: a comment, a blank
    line.  It moves with the transfer it leads."""
    return (s.op is Op.RAWSTMT and not s.target and not s.args
            and not s.accesses and s.effect == Effect.NONE)


def glue(s: Stmt) -> bool:
    """Arithmetic: no memory, no effect, no region."""
    return (s.pure and not s.regions and not s.accesses
            and s.effect == Effect.NONE and bool(s.target))


def defines_whole(s: Stmt, dest: Value) -> bool:
    return (s.op is Op.MARK and s.attr('mark') == 'defines'
            and all(isinstance(a, Value) and a.id == dest.id for a in s.args))


def fill_target(s: Stmt, fetched=frozenset()) -> Optional[Value]:
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
    if not s.regions and not any(a.space is MemSpace.GLOBAL and not writes(a)
                                 for a in s.accesses) and not (
            s.op is Op.STORE and isinstance(s.args[1], Value)
            and s.args[1].id in fetched):
        # Nothing read from global memory, in a statement with nothing
        # inside: the one question that answers most statements of a body.
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
            if opaque(a):
                return None
            if writes(a):
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
class Transfer:
    """One transfer of the per-element body: the buffer it fills and its
    statements there, in order."""
    dest: Value
    stmts: List[Stmt] = field(default_factory=list)
    #: The statements that move data, without what leads them.
    pieces: List[Stmt] = field(default_factory=list)
    #: The tag its loader gave it (`IRBuilder.transfer`), or None.
    tag: Optional[int] = None

    @property
    def shared(self) -> bool:
        return self.dest.type.space is MemSpace.SHARED

    def tokens(self) -> List[Value]:
        return [t for x in walk_stmts(tuple(self.pieces))
                if x.op is Op.COPY_ASYNC for t in x.target]


def fetch(s: Stmt) -> bool:
    """A read of global memory into a value, standing on its own: the first
    half of a transfer whose second half is a store of it."""
    return (s.op is Op.LOAD and len(s.target) == 1 and not s.regions
            and bool(s.accesses) and all(
                a.space is MemSpace.GLOBAL and not writes(a)
                for a in s.accesses))


def transfers(scope: Sequence[Stmt]) -> List[Transfer]:
    """The transfers of `scope`, in body order.

    A transfer written out without a loop is a load and a store per hop, each
    a statement of its own, and the loads belong to it as much as the stores
    -- as long as nothing else reads what they loaded.

    Arithmetic between two pieces of one transfer does not end it -- what
    `licm` lifts out of a hop loop lands there -- and is not part of it
    either: what a pass that moves the transfer does with it is the pass's
    to decide.  Anything else between two pieces ends the transfer, and the
    buffer is then filled by two -- as it is by two pieces of different tags.
    """
    uses: Dict[int, int] = {}
    for x in walk_stmts(tuple(scope)):
        for v in x.operands():
            uses[v.id] = uses.get(v.id, 0) + 1
    out: List[Transfer] = []
    current: Optional[Transfer] = None
    between: List[Stmt] = []
    fetched: Dict[int, Stmt] = {}
    for s in scope:
        dest = fill_target(s, frozenset(fetched))
        if dest is None:
            if neutral(s) or glue(s) or s.op is Op.MARK:
                between.append(s)
            elif fetch(s):
                between.append(s)
                fetched[s.target[0].id] = s
            else:
                current, between, fetched = None, [], {}
            continue
        stored = {x.args[1].id for x in walk_stmts((s,))
                  if x.op is Op.STORE and isinstance(x.args[1], Value)}
        if any(uses.get(vid, 0) != 1 for vid in stored & set(fetched)):
            # It stores a value something else reads as well, and the load
            # cannot leave with it.
            current, between, fetched = None, [], {}
            continue
        halves = {id(fetched[vid]) for vid in stored & set(fetched)}

        def joins(x):
            return id(x) in halves or neutral(x) or defines_whole(x, dest)
        if (current is not None and current.dest.id == dest.id
                and current.tag == tag(s)
                and all(joins(x) or glue(x) for x in between)):
            current.stmts.extend(x for x in between if joins(x))
            current.pieces.extend(x for x in between if id(x) in halves)
        else:
            lead: List[Stmt] = []
            for x in reversed(between):
                if joins(x):
                    lead.append(x)
                elif not glue(x):
                    break
            lead.reverse()
            if len([x for x in lead if id(x) in halves]) != len(halves):
                # A half stands behind something that is not the transfer's.
                current, between, fetched = None, [], {}
                continue
            current = Transfer(dest, tag=tag(s))
            current.stmts.extend(lead)
            current.pieces.extend(x for x in lead if id(x) in halves)
            out.append(current)
        current.stmts.append(s)
        current.pieces.append(s)
        between, fetched = [], {}
    return out


# --------------------------------------------------------------------------- #
# Cover
# --------------------------------------------------------------------------- #

#: What lays nothing into the instruction stream, as the emitter counts code
#: (`emit._NO_CODE`).
_NO_CODE = frozenset({Op.CONST, Op.YIELD, Op.DECLARE, Op.ALLOC, Op.EXTRACT,
                      Op.SPLIT, Op.PACK, Op.MARK})


def executed(stmts: Sequence[Stmt]) -> int:
    """How much `stmts` issue when they run, in the emitter's units of code
    -- one a statement, two a branch -- and a counted loop's body once per
    trip: the cover a transfer issued ahead of them has.  A branch counts its
    longer arm, and a loop whose trips are not known its body once.
    """
    n = 0
    for s in stmts:
        if s.op is Op.FOR:
            lo, hi, step = s.loop_bounds
            trips = 1
            if all(isinstance(x, int) for x in (lo, hi, step)) and step > 0:
                trips = max(0, -(-(hi - lo) // step))
            n += trips * sum(executed(r.body) for r in s.regions)
        elif s.op is Op.IF:
            n += 2 + max((executed(r.body) for r in s.regions), default=0)
        elif s.regions:
            n += 1 + sum(executed(r.body) for r in s.regions)
        elif s.op not in _NO_CODE:
            n += 1
    return n


# --------------------------------------------------------------------------- #
# Pipelines
# --------------------------------------------------------------------------- #

#: The pipeline of a bulk copy (`memory_path`).
BULK = 'bulk'


def memory_path(t: Transfer):
    """The pipeline `t` moves in: the way its bytes go, which the space it
    fills names -- global to shared memory or global to registers -- and the
    tensor memory accelerator's bulk copy, which goes a third way.

    They retire apart where the hardware keeps them apart: on NVIDIA a copy
    into shared memory is counted in groups, a load into registers is
    scoreboarded on its own (`asyncmem`) and a bulk copy completes on a
    barrier of its own (`IRBuilder.copy_bulk`), so none waits for another.
    On AMD one counter takes both of the first two in order; there the
    pipelines say only how far each transfer is issued ahead of the next of
    its kind.
    """
    if any(x.op is Op.COPY_ASYNC and x.counter == 'bulk'
           for x in walk_stmts(tuple(t.pieces))):
        return BULK
    return t.dest.type.space


def one_pipeline(t: Transfer) -> None:
    """Every transfer in the one pipeline."""
    return None


#: The pipelines a transfer moves in, by the name `Options.transfer_pipelines`
#: gives them.  A pipeline is whatever the key says belongs together, and a
#: key may group transfers of any path; these are the two asked for.
PIPELINES = {'path': memory_path, 'one': one_pipeline}


def pipelines(name: str):
    """The key function `name` stands for."""
    found = PIPELINES.get(name)
    if found is None:
        raise ValueError(f'transfer_pipelines: {name!r} is none of '
                         f'{", ".join(sorted(PIPELINES))}')
    return found
