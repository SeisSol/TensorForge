# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Pseudo-IR: register blocks as values.

A block (`blocks.split_register_images`) is still memory to the IR: a buffer
with loads and stores, live from its allocation to its last access whatever
it holds in between.  Here it becomes what it holds.  Walked in the order the
body runs, a store makes what it stores the block's value, and a load is the
value the block holds where the load stands:

* a store of the whole block is its value; of part of it, the value with the
  part inserted (`Op.INSERT`) -- under the explicit vector, of a full-lane
  tail (`valid`), the lanes it writes;
* a load of the whole block is the value; of part of it, the part
  (`Op.SLICE`) -- under the explicit vector, of a full-lane tail, the lanes it
  reads in zeros;
* under a predicate, a store is a select between what it stores and what the
  block held, and a load one between what it reads and what it gives where
  the predicate does not hold (`other`);
* before any store, the block holds the zeros it was allocated with: a splat
  (`IRBuilder.splat`), which the emitters spell as the vector it is.

Where a block is stored in a branch or a loop that stays and read behind it,
its value is handed out: a result of the `if`, each arm yielding what the
block holds at its end and an arm that is not there what came in; an
argument of the `for`, the first trip starting from what came in and each
next from what the last yielded, and a result behind it.  That is how the
batch loop carries the image of the next element its wrap loads at the end of
a trip.

Not where a store stands in a loop whose trips are queried (`while`) or in a
block of raw text, which cannot hand a value out; not in a loop whose raw text
may leave a trip, which would leave without yielding; and not where a load is
named for raw text, or what is stored is of another type than the block holds
-- a mask written into a number's image.  Such a block keeps its buffer.

What the passes behind see then is a value a column, with a definition and
uses: which column is live where, and across what (`passes.pressure`).
"""

from __future__ import annotations

from dataclasses import replace
from typing import Dict, List, Optional, Sequence, Set, Tuple

from .blocks import _JUMPS, _Body, _comment, _is_buffer, _width
from .core import (SCALAR_LAYOUT, Effect, LaneAxis, MemSpace, Op, Operand,
                   Region, RegisterLayout, ScalarType, Stmt, Uniformity, Value,
                   walk, walk_stmts)


class _Block:
    """A block taken to values: its name, its width in elements, and the
    type and layout a value of all of it has."""

    __slots__ = ('name', 'width', 'type', 'layout', 'uniformity', 'offsets')

    def __init__(self, name: str, width: int, type_, layout, uniformity,
                 offsets: Dict[int, int]):
        self.name = name
        self.width = width
        self.type = type_
        self.layout = layout
        self.uniformity = uniformity
        #: The element each access of it starts at, by the access's id.
        self.offsets = offsets


def block_values(body: Tuple[Stmt, ...], scratch, *,
                 explicit_simd: bool = False, predicated: bool = True,
                 report: Optional[List[str]] = None) -> Tuple[Stmt, ...]:
    """`body` with every register block that can be taken to values taken
    to them (see the module): its allocation a zero splat, its stores and
    loads gone into what they define and read, and the values a branch or a
    loop hands out results of it.

    A block is a register image every access of which has a known offset
    and lies inside it, under the restrictions `blocks.split_register_images`
    takes an image apart under, an access under a predicate included unless
    `predicated` is False (`blocks.RegisterBlocks.select_predicated`): one
    the split made (`block_of`), or one whose widest access spans it.
    `scratch` makes the builder the values are numbered by; `report`
    collects what was taken and why the rest was not.
    """
    seen = _Body(body, explicit_simd)
    refused: Dict[str, str] = {}
    images = [im for im in seen.images(written_out=True, refused=refused,
                                       predicated=True, whole=True)
              if im.block == im.value.type.volume]
    blocks, why = _blocks(seen, body, images, explicit_simd, predicated)
    if report is not None:
        for name, reason in why.items():
            report.append(f'values: - {name}: {reason}')
    if not blocks:
        return body
    walker = _Values(seen, blocks, scratch, explicit_simd)
    out, _ = walker.walk(body, {})
    if report is not None:
        kinds = walker.kinds
        count = {k: sum(1 for b in blocks if k in kinds.get(b, ()))
                 for k in ('part', 'pred', 'if', 'for')}
        straight = sum(1 for b in blocks
                       if not {'if', 'for'} & kinds.get(b, set()))
        report.append(f'values: {len(blocks)} block(s) as values: '
                      f'{straight} in one list, {count["if"]} through '
                      f'results of a branch, {count["for"]} through '
                      f'arguments of a loop; {count["part"]} with a part '
                      f'accessed, {count["pred"]} under a predicate; '
                      f'{walker.results} result(s), {walker.carried} '
                      f'argument(s), {walker.selects} select(s)')
    return tuple(out)


def _blocks(seen: _Body, body: Tuple[Stmt, ...], images, explicit_simd: bool,
            predicated: bool = True
            ) -> Tuple[Dict[str, _Block], Dict[str, str]]:
    """The blocks of `images` that can be taken to values, by name, and why
    the others cannot -- not one an access under a predicate reaches unless
    `predicated`."""
    where: Dict[int, Tuple[Stmt, ...]] = {}
    for s, parents in walk(body):
        where[id(s)] = parents
    out: Dict[str, _Block] = {}
    why: Dict[str, str] = {}
    for im in images:
        n = im.block
        elem = im.value.type.elem
        whole = []
        reason = None
        around = where[id(im.alloc)]
        for s, flats in im.accesses:
            value = s.target[0] if s.op == Op.LOAD else s.args[1]
            if isinstance(value, Value):
                if value.type.base != elem:
                    reason = (f'a {s.op} of a {value.type.base} where it '
                              f'holds {elem}')
                    break
                if _width(value, explicit_simd) == n:
                    whole.append(value)
            if s.op == Op.LOAD and (s.attr('extern') or s.attr('escapes')):
                reason = 'a load raw text names'
                break
            if s.predicate is not None and not predicated:
                reason = 'an access under a predicate, which would be a select'
                break
            if s.op == Op.STORE:
                inside = where[id(s)][len(around):]
                reason = _stays(inside)
                if reason:
                    break
        if reason is None and len({v.type for v in whole}) > 1:
            reason = 'the whole block in two types'
        if reason is not None:
            why[im.name] = reason
            continue
        if whole:
            ref = whole[0]
            type_, layout, uniformity = ref.type, ref.layout, ref.uniformity
        else:
            if explicit_simd:
                type_ = ScalarType(elem)
                layout = (RegisterLayout((LaneAxis(n),)) if n > 1
                          else SCALAR_LAYOUT)
            else:
                type_ = ScalarType(elem, n if n > 1 else None)
                layout = SCALAR_LAYOUT
            # What a lane holds of its own, as far as anything can tell.
            uniformity = Uniformity.LANE
        out[im.name] = _Block(im.name, n, type_, layout, uniformity,
                              {id(s): flats[0] for s, flats in im.accesses})
    return out, why


def _stays(around: Sequence[Stmt]) -> Optional[str]:
    """Why a store under the statements `around` -- from the block's own
    list inwards -- cannot hand its value out of them, or None."""
    for s in around:
        if s.op == Op.WHILE:
            return 'a store in a loop whose trips are queried'
        if s.op == Op.RAWBLOCK:
            return 'a store in a block of raw text'
        if s.op == Op.FOR and any(
                x.text and not _comment(x) and _JUMPS.search(x.text)
                for x in walk_stmts(s.regions[0].body)):
            return 'a store in a loop raw text may leave'
        if s.op not in (Op.IF, Op.FOR):
            return f'a store in a `{s.op}`'
    return None


class _Values:
    """One walk over a body, in the order it runs, with what each block
    holds."""

    def __init__(self, seen: _Body, blocks: Dict[str, _Block], scratch,
                 explicit_simd: bool):
        self.seen = seen
        self.blocks = blocks
        self.simd = explicit_simd
        self.b = scratch()
        #: What a load that is gone was: the value the block held.
        self.mapping: Dict[int, Operand] = {}
        #: The values that are a block's zeros, which a part of is zeros.
        self.zeros: Set[int] = set()
        self._stored: Dict[int, frozenset] = {}
        self.results = self.carried = self.selects = 0
        #: What each block went through, by name: `part`, `pred`, `if`, `for`.
        self.kinds: Dict[str, Set[str]] = {}

    # -- what a statement touches ------------------------------------------- #

    def _block(self, s: Stmt) -> Optional[_Block]:
        """The block `s` loads or stores, or None."""
        if s.op not in (Op.LOAD, Op.STORE) or not s.args:
            return None
        base = s.args[0]
        if not _is_buffer(base):
            return None
        return self.blocks.get(self.seen.name_of(base))

    def _stored_in(self, s: Stmt) -> frozenset:
        got = self._stored.get(id(s))
        if got is None:
            names = set()
            for x in walk_stmts((s,)):
                if x.op == Op.STORE:
                    blk = self._block(x)
                    if blk is not None:
                        names.add(blk.name)
            got = self._stored[id(s)] = frozenset(names)
        return got

    # -- values ------------------------------------------------------------- #

    def _sub(self, x):
        while isinstance(x, Value) and x.id in self.mapping:
            x = self.mapping[x.id]
        return x

    def _substituted(self, s: Stmt) -> Stmt:
        """`s` reading what the loads that are gone were, in its operands,
        its predicate and its attributes -- `other`, a loop's `next`."""
        if not self.mapping:
            return s
        args = tuple(self._sub(a) for a in s.args)
        pred = self._sub(s.predicate) if s.predicate is not None else None
        attrs = tuple((k, self._sub(v)) for k, v in s.attrs)
        if (all(a is b for a, b in zip(args, s.args)) and pred is s.predicate
                and all(a[1] is b[1] for a, b in zip(attrs, s.attrs))):
            return s
        return replace(s, args=args, predicate=pred, attrs=attrs)

    def _fresh(self, type_, layout, hint: str, uniformity=None) -> Value:
        if uniformity is None:
            return self.b.value(type_, hint=hint, layout=layout)
        return self.b.value(type_, hint=hint, layout=layout,
                            uniform=uniformity)

    def _of(self, blk: _Block, hint: str) -> Value:
        return self._fresh(blk.type, blk.layout, hint, blk.uniformity)

    def _zero(self, type_, layout, hint: str, out: List[Stmt]) -> Value:
        v = self._fresh(type_, layout, hint)
        zero = 0.0 if type_.base.name.startswith(('F', 'BF', 'TF')) else 0
        out.append(Stmt(op=Op.CONST, target=(v,), attrs=(('value', zero),)))
        self.zeros.add(v.id)
        return v

    def _like(self, x: Value, hint: str) -> Value:
        return self._fresh(x.type, x.layout, hint, x.uniformity)

    def _op(self, op: str, target: Value, args, out: List[Stmt]) -> Value:
        out.append(Stmt(op=op, target=(target,), args=tuple(args)))
        return target

    def _part(self, vec: Value, offset: int, type_, layout, hint: str,
              out: List[Stmt]) -> Value:
        """`vec`'s elements from `offset` on, as many as `type_` and `layout`
        span; zeros of a block's zeros."""
        if vec.id in self.zeros:
            return self._zero(type_, layout, hint, out)
        return self._op(Op.SLICE, self._fresh(type_, layout, hint),
                        (vec, offset), out)

    def _narrowed(self, s: Stmt, value) -> Optional[int]:
        """The elements of `value` a full-lane tail's access to it touches,
        where fewer than it spans -- under the explicit vector, whose emitter
        holds the access to them (`EsimdEmitter._valid_width`)."""
        valid = s.attr('valid')
        if not self.simd or valid is None or not isinstance(value, Value):
            return None
        part = valid * (value.type.length or 1)
        return part if part < _width(value, True) else None

    def _lanes(self, value: Value, width: int):
        """The type and layout of `width` of `value`'s elements."""
        if not self.simd or width == 1:
            return ScalarType(value.type.base), SCALAR_LAYOUT
        return (ScalarType(value.type.base),
                RegisterLayout((LaneAxis(width),)))

    # -- the walk ----------------------------------------------------------- #

    def walk(self, stmts: Sequence[Stmt], cur: Dict[str, Value]
             ) -> Tuple[List[Stmt], Dict[str, Value]]:
        out: List[Stmt] = []
        for orig in stmts:
            s = self._substituted(orig)
            if (s.op == Op.ALLOC and s.target
                    and self.seen.name_of(s.target[0]) in self.blocks):
                blk = self.blocks[self.seen.name_of(s.target[0])]
                cur[blk.name] = self._zero(blk.type, blk.layout,
                                           f'{blk.name}_0', out)
                continue
            blk = self._block(s)
            if blk is not None and s.op == Op.STORE:
                cur[blk.name] = self._store(orig, s, blk, cur[blk.name], out)
                continue
            if blk is not None:
                self._load(orig, s, blk, cur[blk.name], out)
                continue
            if s.regions:
                out.append(self._regions(orig, s, cur))
                continue
            out.append(s)
        return out, cur

    def _store(self, orig: Stmt, s: Stmt, blk: _Block, held: Value,
               out: List[Stmt]) -> Value:
        """What `blk` holds behind the store `s`."""
        value = s.args[1]
        offset = blk.offsets[id(orig)]
        if not isinstance(value, Value):
            # A number is one element, the block's element.
            elem = ScalarType(blk.type.base)
            value = self._fresh(elem, SCALAR_LAYOUT, f'{blk.name}_n')
            out.append(Stmt(op=Op.CONST, target=(value,),
                            attrs=(('value', s.args[1]),)))
        width = _width(value, self.simd)
        narrow = self._narrowed(s, value)
        if narrow is not None:
            type_, layout = self._lanes(value, narrow)
            value = self._part(value, 0, type_, layout, f'{blk.name}_p', out)
            width = narrow
        whole = width == blk.width
        if not whole:
            self.kinds.setdefault(blk.name, set()).add('part')
        if s.predicate is not None:
            self.selects += 1
            self.kinds.setdefault(blk.name, set()).add('pred')
            if whole:
                return self._op('select', self._of(blk, blk.name),
                                (s.predicate, value, held), out)
            old = self._part(held, offset, value.type, value.layout,
                             f'{blk.name}_o', out)
            value = self._op('select', self._like(value, f'{blk.name}_s'),
                             (s.predicate, value, old), out)
        if whole:
            return value
        return self._op(Op.INSERT, self._of(blk, blk.name),
                        (held, value, offset), out)

    def _load(self, orig: Stmt, s: Stmt, blk: _Block, held: Value,
              out: List[Stmt]) -> None:
        """The load `s` gone into what `blk` holds where it stands."""
        x = s.target[0]
        offset = blk.offsets[id(orig)]
        width = _width(x, self.simd)
        narrow = self._narrowed(s, x)
        if narrow is not None:
            # The lanes up to `valid`, zeros past them.
            type_, layout = self._lanes(x, narrow)
            part = self._part(held, offset, type_, layout, f'{blk.name}_p',
                              out)
            zero = self._zero(x.type, x.layout, f'{blk.name}_z', out)
            got = self._op(Op.INSERT, self._like(x, x.hint), (zero, part, 0),
                           out)
        elif width == blk.width:
            got = held
        else:
            got = self._part(held, offset, x.type, x.layout, x.hint, out)
        if narrow is not None or width != blk.width:
            self.kinds.setdefault(blk.name, set()).add('part')
        if s.predicate is not None:
            self.selects += 1
            self.kinds.setdefault(blk.name, set()).add('pred')
            other = s.attr('other')
            other = (self._sub(other) if other is not None
                     else self._zero(x.type, x.layout, f'{x.hint}_z', out))
            got = self._op('select', self._like(x, x.hint),
                           (s.predicate, got, other), out)
        self.mapping[x.id] = got

    # -- control flow ------------------------------------------------------- #

    def _regions(self, orig: Stmt, s: Stmt, cur: Dict[str, Value]) -> Stmt:
        """`s`, a statement with regions, walked: what a block holds behind
        it handed out of it where it is stored inside."""
        written = sorted(self._stored_in(orig) & cur.keys())
        if not written:
            regions = tuple(
                replace(r, body=tuple(self.walk(r.body, dict(cur))[0]))
                for r in orig.regions)
            return replace(s, regions=regions)
        if s.op == Op.IF:
            return self._if(orig, s, cur, written)
        assert s.op == Op.FOR, s.op
        return self._for(orig, s, cur, written)

    def _arm(self, region: Region, cur: Dict[str, Value],
             written: List[str]) -> Region:
        """`region` walked from `cur`, yielding what it yielded and then what
        each block of `written` holds at its end."""
        last = region.terminator
        body = region.body[:-1] if last is not None else region.body
        out, after = self.walk(body, dict(cur))
        handed = tuple(after[b] for b in written)
        if last is not None:
            y = replace(last, args=tuple(self._sub(a) for a in last.args)
                        + handed)
        else:
            y = Stmt(op=Op.YIELD, args=handed, pure=False, movable=False)
        return replace(region, body=tuple(out) + (y,))

    def _if(self, orig: Stmt, s: Stmt, cur: Dict[str, Value],
            written: List[str]) -> Stmt:
        arms = [self._arm(r, cur, written) for r in orig.regions]
        if len(arms) == 1:
            # The arm that is not there hands on what came in.
            assert not s.target, 'an `if` with results and one arm'
            arms.append(Region(body=(Stmt(
                op=Op.YIELD, args=tuple(cur[b] for b in written),
                pure=False, movable=False),)))
        results = tuple(self._of(self.blocks[b], self.blocks[b].name)
                        for b in written)
        self.results += len(results)
        for b, v in zip(written, results):
            cur[b] = v
            self.kinds.setdefault(b, set()).add('if')
        return replace(s, target=s.target + results, regions=tuple(arms))

    def _for(self, orig: Stmt, s: Stmt, cur: Dict[str, Value],
             written: List[str]) -> Stmt:
        region = orig.regions[0]
        inits = tuple(cur[b] for b in written)
        args = tuple(self._of(self.blocks[b], self.blocks[b].name)
                     for b in written)
        inner = dict(cur)
        inner.update(zip(written, args))
        arm = self._arm(region, inner, written)
        results = tuple(self._of(self.blocks[b], self.blocks[b].name)
                        for b in written)
        self.carried += len(results)
        for b, v in zip(written, results):
            cur[b] = v
            self.kinds.setdefault(b, set()).add('for')
        return replace(s, args=s.args + inits, target=s.target + results,
                       regions=(replace(arm, args=region.args + args),))


def unguard_register_stores(body: Tuple[Stmt, ...], *, threads: int,
                            arena: Optional[str] = None,
                            sub_group: Optional[int] = None,
                            report: Optional[List[str]] = None
                            ) -> Tuple[Stmt, ...]:
    """`body` with the lane predicate dropped from every store into a
    register buffer where the lanes it leaves out are lanes no read takes
    what the buffer held there before: the store then writes every lane.

    The lanes a read takes are those anything reads of what it loads
    (`demand.Demand`), and a read takes what the buffer held before the store
    where the write it finds there is not one behind the store -- at any
    execution of the read, before the store or after it, where a loop hands
    one trip's registers to the next.  A predicated store keeps the block
    whose image it writes from becoming a value under SPMD on Intel
    (`block_values` without `predicated`): a window of an image written by
    an operation, a sum its epilogue writes to the window's rows alone, a
    copy of a window into the next operation's image.  Where what is stored
    comes of a cross-lane read, the guard is also what IGC 2.40.13 to
    2.42.6 moves the read into (`pin_crosslane_sources`); and the order-6
    dense time derivative reads all but a few of those windows only by
    broadcasts from the window's own rows.  A body too large for the flow
    (`flow.TooLarge`) is left as it is."""
    from .demand import Demand
    from .flow import Flow, TooLarge

    def lane_guard(s: Stmt) -> bool:
        return (s.op == Op.STORE and isinstance(s.predicate, Value)
                and len(s.accesses) == 1
                and s.accesses[0].space is MemSpace.REGISTER
                and s.predicate.uniformity <= Uniformity.LANE)
    stores = [s for s in walk_stmts(body) if lane_guard(s)]
    if not stores:
        return body
    # Only buffers every statement reaches by a load or store of one element
    # the flow places: what else touches one may read any lane of it.
    plain: Dict[int, bool] = {}
    loads: Dict[int, List[Stmt]] = {}
    for s in walk_stmts(body):
        for a in s.accesses:
            if a.space is not MemSpace.REGISTER or a.base is None:
                continue
            ok = (s.op in (Op.LOAD, Op.STORE) and len(s.accesses) == 1
                  and (s.op == Op.STORE or len(s.target) == 1))
            plain[id(a.base)] = plain.get(id(a.base), True) and ok
            if ok and s.op == Op.LOAD:
                loads.setdefault(id(a.base), []).append(s)
    stores = [s for s in stores if plain.get(id(s.accesses[0].base))]
    if not stores:
        return body
    try:
        flow = Flow(body, threads, arena, masks=True, before=False)
    except TooLarge as why:
        if report is not None:
            report.append(f'unguard: {why}; left as it is')
        return body
    demand = Demand(flow, sub_group)
    dropped: Set[int] = set()
    for s in stores:
        execs = flow.executions(s)
        if not execs or id(s) in flow.opaque:
            continue
        reads = [(r, ld) for ld in loads.get(id(s.accesses[0].base), ())
                 for r in flow.executions(ld)]
        if any(r not in flow.reach for r, _ in reads):
            continue
        if all(_unread_beside(flow, demand, n, reads) for n in execs):
            dropped.add(id(s))
    if not dropped:
        return body
    if report is not None:
        report.append(f'unguard: {len(dropped)} of {len(stores)} stores into '
                      f'registers write every lane; the lanes their guard '
                      f'leaves out are read by nothing')

    def rewrite(stmts: Tuple[Stmt, ...]) -> Tuple[Stmt, ...]:
        out = []
        for s in stmts:
            if s.regions:
                s = replace(s, regions=tuple(replace(r, body=rewrite(r.body))
                                             for r in s.regions))
            if id(s) in dropped:
                s = replace(s, predicate=None)
            out.append(s)
        return tuple(out)
    return rewrite(body)


def _unread_beside(flow, demand, n: int, reads) -> bool:
    """Does no read in `reads`, `(execution, load)`, take in a lane the store
    `n` leaves out what the element held before it?"""
    beside = flow.full & ~flow.insts[n].mask
    if not beside:
        return True
    cells = {cell for _, cell in flow.cells_of(n)}
    for r, ld in reads:
        for reach in flow.reach[r]:
            lanes = reach.mask & beside
            if not lanes or reach.cell not in cells:
                continue
            if reach.writer is not None and reach.writer > n and all(
                    w > n for w in reach.maybe):
                # A write behind the store: what the element holds there is
                # not the store's to keep or replace.
                continue
            if demand.lanes(ld.target[0], flow.insts[r].env()) & lanes:
                return False
    return True


#: An empty asm the value goes into: no instruction, but a use of the value
#: right where it is defined (`pin_crosslane_sources`).
_PIN = '__asm__ volatile("" :: "r"({0}));'


class _Guards:
    """Which statements run under a lane guard -- in an `if` on a
    lane-dependent condition, or under a predicate -- and which values keep to
    some lanes only (`kept`): every use of the value under a guard, an arm of
    a select on a lane-dependent condition, a store into a register buffer
    every load of which keeps to some lanes -- IGC sees through register
    buffers -- or a statement whose results all keep to some lanes in turn.
    A value with no use, a yield out of anything but a lane-guarded `if`, a
    store or raw text outside a guard all count as every lane's.  And which
    values come of a cross-lane read (`crossed`), through register buffers
    alike."""

    def __init__(self, body: Tuple[Stmt, ...]):
        self.guarded: Dict[int, bool] = {}

        def lane_dependent(x) -> bool:
            return isinstance(x, Value) and x.uniformity <= Uniformity.LANE
        self.lane_dependent = lane_dependent

        def mark(stmts, inside: bool):
            for st in stmts:
                g = inside or st.predicate is not None
                self.guarded[id(st)] = g
                for r in st.regions:
                    arm = g or (st.op == Op.IF and st.args
                                and lane_dependent(st.args[0]))
                    mark(r.body, arm)
        mark(body, False)
        self.uses: Dict[int, List[Tuple[Stmt, int]]] = {}
        self.order = list(walk_stmts(body))
        self.loads: Dict[int, List[Stmt]] = {}
        stores: Dict[int, List[Stmt]] = {}
        for st in self.order:
            for k, a in enumerate(st.args):
                if isinstance(a, Value):
                    self.uses.setdefault(a.id, []).append((st, k))
            if st.predicate is not None:
                self.uses.setdefault(st.predicate.id, []).append((st, -1))
            if self.register(st):
                (self.loads if st.op == Op.LOAD else stores).setdefault(
                    id(st.args[0]), []).append(st)
        self.kept: Set[int] = set()
        for st in reversed(self.order):
            for t in st.target:
                us = self.uses.get(t.id, ())
                if us and all(self.keeps(u, k) for u, k in us):
                    self.kept.add(t.id)
        self.crossed: Set[int] = set()
        filled: Set[int] = set()
        for _ in range(2):
            for st in self.order:
                hit = bool(st.attr('crosslane')) or any(
                    v.id in self.crossed for v in st.operands())
                if st.op == Op.LOAD and self.register(st):
                    hit = hit or id(st.args[0]) in filled
                if st.op == Op.STORE and self.register(st) and len(
                        st.args) > 1 and isinstance(st.args[1], Value):
                    if st.args[1].id in self.crossed:
                        filled.add(id(st.args[0]))
                if hit:
                    self.crossed.update(t.id for t in st.target)

    @staticmethod
    def register(st: Stmt) -> bool:
        return (st.op in (Op.LOAD, Op.STORE) and bool(st.accesses)
                and st.accesses[0].space is MemSpace.REGISTER)

    def held_back(self, u: Stmt, k: int) -> bool:
        """Whether the use `u` (operand `k`) is one a guard keeps to some
        lanes: under the guard, a select's arm, or a store into a register
        buffer read only so."""
        if self.guarded.get(id(u)):
            return True
        if (u.op == 'select' and k in (1, 2) and u.args
                and self.lane_dependent(u.args[0])):
            return True
        return (u.op == Op.STORE and k == 1 and self.register(u)
                and all(x.id in self.kept for ld in self.loads.get(
                    id(u.args[0]), ()) for x in ld.target))

    def keeps(self, u: Stmt, k: int) -> bool:
        if self.held_back(u, k):
            return True
        return bool(u.target) and u.op not in (Op.STORE, Op.YIELD) and all(
            x.id in self.kept for x in u.target)


def _lane_kept(body: Tuple[Stmt, ...]) -> Set[int]:
    """The values every use of which keeps to some lanes (`_Guards.kept`)."""
    return _Guards(body).kept


def pin_crosslane_sources(body: Tuple[Stmt, ...], at_risk: bool = True,
                          where: str = 'use') -> Tuple[Stmt, ...]:
    """`body` with the result of every cross-lane read handed to an empty
    volatile asm right where it is read (`blocks.RegisterBlocks.
    pin_crosslane`): the read then stays where it stands, with every lane
    taking part.

    IGC 2.40.13 to 2.42.6 moves a sub-group broadcast whose result is used
    only under a lane guard into the guard, and there the lanes the guard
    leaves out read 0.  Measured on the Max 1550, as the right value less
    exactly the products broadcast from those lanes: with the blocks as
    values, the order-6 dense derivative's dQ(1), first column, rows 32 to 34
    under `lane < 3`, less lanes 3 to 15 of one direction (to 1e-13); the
    single-precision `localFluxAll`'s Q, first column, rows 48 to 55 under
    `lane < 8`, less lanes 8 to 15 of one face (F32's last digit); with every
    load that pays put back, the derivative's dQ(4) under `lane < 4`.
    Whether the guard is a select or a branch around a store does not
    matter; as written the kernel escapes only because what it reads goes
    through arrays a loop indexes, which reach IGC later.  The asm emits
    nothing, but it is not free -- every cross-lane read pinned cost the
    order-6 dense `localFluxAll`, all loads that pay put back, 21 % more
    instructions on IGC 2.41.5 -- so `at_risk` pins only where what came of
    a cross-lane read enters a guard: the last value outside it every use of
    which keeps to some lanes (`_Guards`), which holds all it is computed
    from outside too.

    `where` the asm stands: right behind the value (`definition`), or right
    ahead of the first statement of its list that reads it under a guard
    (`use`), so that IGC may compute it as late as that -- behind the value
    where nothing of its list reads it so.  The two were a line apart in
    SeisSol's kernels, and their listings and times the same on the Max 1550
    (order-6 dense `derivative` as values, -12.3 and -13.3 %, job 561193).
    Nor do the pins cost what they first seemed to: without them IGC 2.41.5
    spilled as much or more (the order-6 dense `localFluxAll`, every load
    that pays put back, 14656 B against 4672); what the order-8 sparse
    `derivative` lost with them came of its predicated blocks staying memory
    (`Options.register_selects`).
    """
    sources = {}
    if at_risk:
        # Where what came of a cross-lane read enters a guard: the last value
        # outside it, every use of which keeps to some lanes.  Pinned there,
        # it and all it is computed from stay outside.
        g = _Guards(body)
        for st in g.order:
            if g.guarded.get(id(st)) or st.op == Op.LOAD:
                continue
            for t in st.target:
                if (t.id in g.kept and t.id in g.crossed
                        and isinstance(t.type, ScalarType)
                        and any(g.held_back(u, k)
                                for u, k in g.uses.get(t.id, ()))):
                    sources.setdefault(t.id, t)
    else:
        for s in walk_stmts(body):
            if s.attr('crosslane'):
                for t in s.target:
                    if isinstance(t.type, ScalarType):
                        sources.setdefault(t.id, t)
    if not sources:
        return body
    pinned = set()

    def pin(v: Value) -> Stmt:
        pinned.add(v.id)
        return Stmt(op=Op.RAWSTMT, target=(), args=(v,), text=_PIN,
                    pure=False, movable=False, effect=Effect.UNKNOWN,
                    accesses=(), attrs=(('fmt', True),))

    # The statements that read a source under a guard, with the sources each
    # reads so.
    held: Dict[int, List[int]] = {}
    if at_risk and where == 'use':
        for tid in sources:
            for u, k in g.uses.get(tid, ()):
                if g.held_back(u, k):
                    held.setdefault(id(u), []).append(tid)

    def rewrite(stmts: Tuple[Stmt, ...]) -> Tuple[Stmt, ...]:
        # Per source, the first statement of the list that holds a read of
        # it under a guard: the asm goes right ahead of it.
        first: Dict[int, int] = {}
        if held:
            for j, s in enumerate(stmts):
                for x in walk_stmts((s,)):
                    for tid in held.get(id(x), ()):
                        first.setdefault(tid, j)
        ahead: Dict[int, List[Value]] = {}
        out: List[Stmt] = []
        for j, s in enumerate(stmts):
            for v in ahead.pop(j, ()):
                out.append(pin(v))
            if s.regions:
                s = replace(s, regions=tuple(replace(r, body=rewrite(r.body))
                                             for r in s.regions))
            out.append(s)
            for t in s.target:
                if t.id in sources and t.id not in pinned:
                    at = first.get(t.id)
                    if at is not None and at > j:
                        ahead.setdefault(at, []).append(t)
                    else:
                        out.append(pin(t))
        return tuple(out)

    return rewrite(body)
