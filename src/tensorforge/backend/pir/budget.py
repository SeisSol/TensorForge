# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Pseudo-IR: the register budget.

With the blocks of a register image taken to values (`values`), what a body
holds live at each statement is what the model says it is (`passes.
pressure`): a column version from the statement that defines it to its last
use.  Where that is more than the target's compiler can keep (the budget, in
the model's own bytes -- a lane's under SPMD, a work-item's under the explicit
vector), the compiler spills on its own: VC and IGC to scratch around every
`mad`, which was the measured problem on PVC.  Here the body is made to fit
first, deciding which column versions are not kept across a stretch where
too much is live.

`rematerialize` takes a value that is a load of memory the kernel does not
write -- a batch-constant operator, an element's data it only reads -- and
loads it again right before its next use instead of keeping it across the
gap: no memory is added, and the gap is free of it.  A constant goes the same
way, and costs nothing to put back.  The choice is Belady's,
at every statement over the budget in the order the body runs: of the values
live there that can be loaded again and are not read there, the one read
again furthest ahead goes, until the statement fits or nothing is left to
take.  A value is taken out in the statement list it is defined in, between
the statements of that list that read it -- what a statement reads anywhere
inside it, it reads -- so that the load put back before the next of them
stands where it is seen from every later read in the list.  Under the
explicit vector a register buffer nothing touches goes first, since the model
holds every register buffer through the whole body
(`drop_unused_registers`).

A reload is not free, and the compiler's spill it replaces is not always
there (`Budget`).  The pass acts only on a body that is over what the
target's compiler was measured to hold by more than a margin; a value goes
only if its reload costs the target no more instructions than the spill it
saves (`RELOAD_PRICE`, `Budget.spill_price`); and loads go only where
the body as written was well over the capacity and what pays -- the blocks'
values and the loads put back together -- would close most of that excess
(`CLOSED_MIN`, `EXCESS_MIN`): where what cannot go holds it, the compiler
spills whatever is put back, and only constants go.  A first version took
every load it could wherever the
model was over: the viscoelastic derivative put back 1773 loads for sm_90,
and its SASS grew by 79 % for 40 % less spill; on PVC every volume kernel
lost (order 6, dense: values alone 23 % slower, with the reloads 58 %).
Where nothing but constants would go, the body is left as it came, its
images whole (`RegisterBlocks.remat`): taking them apart is worth it only
for what is put back then.

A value computed in the body (an intermediate the next product reads, a
sum) cannot be loaded again from anywhere.  Where the target has shared
memory beside the registers a multiplication may take, `spill_shared` is the
next tier: such a value is stored once, behind its definition, into a slot of
its own lane, and loaded from there before a read the gap holds it from --
again Belady's choice, over what `rematerialize` left, within the slots the
tier holds (`Budget.shared`).  What it costs is what the compiler's own spill
of the value costs, a store and a load where it is read again; what differs
is where to: shared memory a multiplication has to itself, rather than the
scratch or local memory every lane's spills share the caches with, and the
values the furthest read decides rather than the compiler's.
"""

from __future__ import annotations

import bisect
import heapq
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union

from .core import BufferType, MemSpace, Op, ScalarType, Stmt, Value
from .passes import (_affine_costs, _index, _register_slot_events,
                     _value_index, register_bytes)

#: The spaces a value may be loaded again from: memory that is the same for
#: the whole kernel where nothing in it is written.  Not shared memory, which
#: the other lanes of the block write.
_RELOADABLE = (MemSpace.GLOBAL, MemSpace.CONSTANT, MemSpace.PARAM)

#: Instructions a value put back costs where it is read next, by what it is:
#: a constant one move; a load its address and the load; a load under a
#: predicate besides the zero it gives where the predicate does not hold.
#: Measured on SASS (sm_90): each guarded reload came with 1.4 IMAD, 0.9
#: CS2R and 0.4 MOV beside its LDG -- +443 LDG and +1896 instructions in the
#: order-6 derivative, +2330 and +9744 in the viscoelastic one.
RELOAD_PRICE = {'const': 1, 'load': 2, 'guarded': 4}

#: What goes from a body that what pays could not bring near its capacity:
#: the compiler spills it anyway, and a reload saves at most which value it
#: spills.
CHEAP = RELOAD_PRICE['const']

#: The share of a body's excess over its capacity that may be left once
#: everything that pays is put back, for the pass to put it back, where the
#: body as written is not known (`rematerialize`'s `as_written`).
LEFT_OVER = 0.3

#: Where the body as written is known: the share of *its* excess over the
#: capacity that what pays must take away, the blocks' values and the loads
#: put back together, for loads to go (`CLOSED_MIN`) -- and how large that
#: excess must be against the capacity (`EXCESS_MIN`).  Per body, against
#: the times with every load that pays put back, on PVC (IGC 2.41.5,
#: oneapi, F64, 262144 elements; elastic orders 6 and 8, viscoelastic 6),
#: as (share closed, excess over capacity): faster were the order-8 sparse
#: `derivative` (1.00, 1.09; -55 %), the viscoelastic `fluxLocalAll` (0.62,
#: 3.50; -31 %), the order-8 dense `localFluxAll` (0.55, 2.72; -16 %), the
#: order-6 dense `volume` (0.74, 1.26; -14 %), the viscoelastic `volumeExt`
#: (0.54, 2.13; -13 %) and the order-6 dense `localFluxAll` (0.77, 0.92;
#: -12 %); unchanged the order-8 sparse `localFluxAll` (0.84, 0.90); slower
#: the order-8 sparse `volume` (0.70, 0.87; +19 %), the order-8 dense
#: `volume` (0.46, 2.69; +17 %), the viscoelastic `derivative` (0.40, 5.53;
#: +7 %) and the order-6 sparse `derivative` (1.00, 0.68; +4 %).  Both bars
#: separate them, and both narrowly: a fit to eleven kernels, not a model
#: of IGC, whose response to the values -- the order-8 volumes 43 and 48 %
#: slower with them alone -- the register model does not see.
CLOSED_MIN = 0.53
EXCESS_MIN = 0.9


@dataclass(frozen=True)
class Budget:
    """What a body is held to (`abstract_instruction.register_budget`).

    `capacity`: the model's bytes a lane -- a work-item under the explicit
    vector -- the target's compiler was measured to keep without spilling;
    the pass takes values out down to it.  `margin`: how far above it a body
    may peak before the pass acts at all, for what the model cannot say of
    the compiler -- a body within it is left alone.  `spill_price`: the
    instructions one value the compiler spills costs it, its store and a
    fill; a value is put back only where its reload costs no more
    (`RELOAD_PRICE`).  `left_over`: the share of the excess that may be
    left with everything that pays put back, for loads to go at all;
    negative for `LEFT_OVER`.  A bare number of bytes is a capacity with
    neither margin nor price to stop at."""

    capacity: int
    margin: float = 0.15
    spill_price: int = RELOAD_PRICE['guarded']
    #: The share of the excess that may be left with all that pays put back
    #: for loads to go at all (`LEFT_OVER`).
    left_over: float = -1.0
    #: Bytes of shared memory a multiplication may take for the values it
    #: spills there (`spill_shared`); 0 for no such tier.
    shared: int = 0
    #: Whether those slots are accessed volatile (`spill_shared`).
    shared_volatile: bool = True

    @property
    def trigger(self) -> int:
        """The peak above which the pass acts."""
        return int(self.capacity * (1 + self.margin))


def reload_kind(s: Stmt) -> str:
    """What putting back the value `s` defines is, as `RELOAD_PRICE` prices
    it."""
    if s.op == Op.CONST:
        return 'const'
    if s.predicate is not None:
        return 'guarded'
    return 'load'


class Liveness:
    """`passes.pressure`'s model of a body, kept per value: the statements in
    the order the body runs, each value's definition, its last use as the
    model extends it, the statements that read it and what it weighs; and
    what is live at each statement.

    The same rules as the sweep there -- region arguments live through their
    region, a use inside a region the definition is not in keeps the value to
    the end of that region, an integer offset weighs nothing, a folded
    cross-lane read nothing, register arrays as their slots under SPMD and
    whole under the explicit vector -- so that `profile`'s largest entry is
    what `pressure(in_bytes=True)` answers.
    """

    def __init__(self, body: Tuple[Stmt, ...], explicit_simd: bool,
                 folded_crosslane: bool = False):
        values = _value_index(body)
        order, span = _index(body)
        self.order, self.span = order, span
        pos: Dict[int, int] = {id(st): i for i, st in enumerate(order)}
        self.pos = pos
        define: Dict[int, int] = {}
        last: Dict[int, int] = {}
        for i, st in enumerate(order):
            for r in st.regions:
                for a in r.args:
                    define[a.id] = i
                    last[a.id] = span[i][1]
            for t in st.target:
                define[t.id] = i
                last.setdefault(t.id, i)
        enclosing: Dict[int, List[int]] = {}
        #: Each statement list, by its id: the positions of its statements.
        self.lists: Dict[int, List[int]] = {}
        #: Each statement's list and its index in it, by position.
        self.place: Dict[int, Tuple[int, int]] = {}

        def collect(stmts, chain):
            positions = self.lists.setdefault(id(stmts), [])
            for k, st in enumerate(stmts):
                i = pos[id(st)]
                enclosing[i] = chain
                positions.append(i)
                self.place[i] = (id(stmts), k)
                for r in st.regions:
                    collect(r.body, chain + [i])
        collect(body, [])

        def reach(d: int, i: int) -> int:
            end = i
            for anc in enclosing.get(i, []):
                if anc >= d and not (span[anc][0] <= d <= span[anc][1]):
                    end = max(end, span[anc][1])
                elif span[anc][0] > d:
                    end = max(end, span[anc][1])
            return end

        uses: Dict[int, List[int]] = {}
        for i, st in enumerate(order):
            for v in st.operands():
                d = define.get(v.id)
                if d is None:
                    continue
                last[v.id] = max(last.get(v.id, d), reach(d, i))
                uses.setdefault(v.id, []).append(i)
        root, carrier = _affine_costs(order)
        for vid, keep in root.items():
            if vid in last:
                last[keep] = max(last.get(keep, define.get(keep, 0)),
                                 last[vid])
        folded = {t.id for st in order if folded_crosslane and st.pure
                  and st.attr('crosslane') for t in st.target}
        weight: Dict[int, int] = {}
        for vid, d in define.items():
            v = values.get(vid)
            if v is None or isinstance(v.type, BufferType):
                continue
            if vid in root and vid not in carrier:
                continue
            if vid in folded:
                continue
            w = register_bytes(v, explicit_simd)
            if w:
                weight[vid] = w
        events: List[Tuple[int, int]] = []
        for vid, w in weight.items():
            events += [(define[vid], w), (last.get(vid, define[vid]) + 1, -w)]
        if not explicit_simd:
            events += _register_slot_events(order, span, enclosing, values,
                                            reach)
        else:
            floor = sum(register_bytes(v, explicit_simd)
                        for v in values.values()
                        if isinstance(v.type, BufferType))
            if floor:
                events += [(0, floor), (len(order), -floor)]
        delta = [0] * (len(order) + 1)
        for p, w in events:
            delta[min(p, len(order))] += w
        profile: List[int] = []
        live = 0
        for d in delta[:len(order)]:
            live += d
            profile.append(live)
        self.values = values
        self.define, self.last, self.uses = define, last, uses
        self.weight = weight
        #: Bytes live across each statement, by position.
        self.profile = profile

    @property
    def peak(self) -> int:
        return max(self.profile, default=0)


class _Candidate:
    """A value that can be loaded again: its load, the list it is defined
    in, the indices in that list of the statements that read it, and what
    what a reload of it is and costs (`reload_kind`)."""

    __slots__ = ('vid', 'load', 'list', 'starts', 'ends', 'at', 'reads',
                 'weight', 'last', 'kind', 'price')

    def __init__(self, vid, load, list_id, starts, ends, at, reads, weight,
                 last):
        self.vid = vid
        self.load = load
        self.list = list_id
        self.starts = starts
        self.ends = ends
        #: Where the value is defined now: the load's index, or the last
        #: reload put back.
        self.at = at
        self.reads = reads
        self.weight = weight
        self.last = last
        self.kind = reload_kind(load)
        self.price = RELOAD_PRICE[self.kind]


def _written(order: Sequence[Stmt]):
    """What the body writes: per space, the bases written, and the spaces
    written somewhere nobody can say where (`Access.base` None, or the
    unknown space, which is every space)."""
    bases: Dict[MemSpace, Set[int]] = {}
    anywhere: Set[MemSpace] = set()
    for st in order:
        for a in st.accesses:
            if not a.writes:
                continue
            if a.space is MemSpace.UNKNOWN:
                anywhere.update(MemSpace)
            elif a.base is None:
                anywhere.add(a.space)
            else:
                bases.setdefault(a.space, set()).add(id(a.base))
    return bases, anywhere


def _reloadable(s: Stmt, written) -> bool:
    """Whether `s` defines a value that can be had again where it is read
    next: a constant, which the emitters spell where it is read -- the model
    counts the zero a column starts from as the 256 bytes of its column under
    the explicit vector -- or a load of memory nothing in the body writes,
    a copy of which can stand anywhere: no name raw text spells, one number
    or vector."""
    if (s.op == Op.CONST and len(s.target) == 1
            and isinstance(s.target[0].type, ScalarType)):
        return True
    if (s.op != Op.LOAD or len(s.target) != 1 or len(s.accesses) != 1
            or not isinstance(s.target[0].type, ScalarType)
            or s.attr('extern') or s.attr('escapes')):
        return False
    read = s.accesses[0]
    if read.space not in _RELOADABLE or read.base is None:
        return False
    bases, anywhere = written
    return (read.space not in anywhere
            and id(read.base) not in bases.get(read.space, ()))


def drop_unused_registers(body: Tuple[Stmt, ...], explicit_simd: bool
                          ) -> Tuple[Tuple[Stmt, ...], int, int]:
    """`body` without the allocations of register buffers nothing touches,
    how many there were and what the model counted them at.

    Under the explicit vector the model holds a register buffer from the
    start of the body to its end, read or not (`passes.pressure`).  A buffer
    whose loads and stores the passes before took away keeps its allocation,
    and the emitter declares it: the order-6 derivative declares 23 register
    buffers it never reads, 78 KB of the 122 KB the model gives it under
    ESIMD with its blocks as values (`values`), more than five times the
    budget.  Against such a floor every statement is over the budget, and
    every load that can be had again is put back everywhere: the order-6
    sparse local flux put back 2644 loads with its 7 KB of them counted, and
    1 without, which is all it needs to fit.  What
    touches a buffer: a statement with it among its arguments, as the value
    its allocation defines or as the macro layer's `Symbol` by its name
    (`blocks._Body.name_of`), or raw text that names it.
    """
    from .blocks import _IDENTIFIER, _Body, _comment, _is_buffer
    seen = _Body(body, explicit_simd)
    allocs: Dict[str, Stmt] = {}
    for s in seen.walked:
        if (s.op == Op.ALLOC and s.target
                and isinstance(s.target[0].type, BufferType)
                and s.target[0].type.space is MemSpace.REGISTER):
            name = seen.names[s.target[0].id]
            if name in allocs:
                allocs[name] = None
            else:
                allocs[name] = s
    unused = {name: s for name, s in allocs.items() if s is not None}

    def flat(xs):
        for x in xs:
            if isinstance(x, (tuple, list)):
                yield from flat(x)
            else:
                yield x

    for s in seen.walked:
        if not unused:
            break
        for x in flat((s.args, tuple(v for _, v in s.attrs))):
            if _is_buffer(x) and s is not unused.get(seen.name_of(x)):
                unused.pop(seen.name_of(x), None)
        text = ' '.join(t for t in (s.text, s.attr('decl'))
                        if isinstance(t, str))
        if text and not _comment(s):
            for word in unused.keys() & set(_IDENTIFIER.findall(text)):
                unused.pop(word)
    if not unused:
        return body, 0, 0
    drop = {id(s) for s in unused.values()}
    weight = sum(register_bytes(s.target[0], explicit_simd)
                 for s in unused.values())

    def rewrite(stmts: Tuple[Stmt, ...]) -> Tuple[Stmt, ...]:
        return tuple(
            replace(s, regions=tuple(replace(r, body=rewrite(r.body))
                                     for r in s.regions))
            if s.regions else s
            for s in stmts if id(s) not in drop)

    return rewrite(body), len(drop), weight


def rematerialize(body: Tuple[Stmt, ...], scratch, *,
                  budget: Union[Budget, int], explicit_simd: bool = False,
                  folded_crosslane: bool = False,
                  report: Optional[List[str]] = None,
                  taken: Optional[Dict[str, int]] = None,
                  as_written: Optional[int] = None
                  ) -> Tuple[Stmt, ...]:
    """`body` with the values that are loads of memory nothing writes loaded
    again before their next read where too much is live to keep them and the
    reload pays (see the module): `budget` as `Budget`, or a capacity in
    bytes with neither margin nor price to stop at -- a lane's or a
    work-item's (`explicit_simd`), in the model's own (`Liveness`).

    Under the explicit vector, register buffers nothing touches go first
    (`drop_unused_registers`): the model would count them all the way.

    `scratch` makes the builder the loads put back are numbered by;
    `report` collects the peak before and after, and what was put back;
    `taken` counts the reloads by what they reload (`RELOAD_PRICE`'s keys).
    `as_written` is the model's peak of the body as written, before its
    blocks were values: where given, loads go only where what pays closes enough of
    that body's excess (`CLOSED_MIN`, `EXCESS_MIN`), else where it would
    leave little of this one's (`LEFT_OVER`).
    """
    if not isinstance(budget, Budget):
        budget = Budget(int(budget), margin=0.0, spill_price=10 ** 9,
                        left_over=float(10 ** 9))
    if explicit_simd:
        body, dropped, weight = drop_unused_registers(body, explicit_simd)
        if dropped and report is not None:
            report.append(f'budget: {dropped} register buffer(s) nothing '
                          f'touches dropped, {weight} B the model held')
    live = Liveness(body, explicit_simd, folded_crosslane)
    before = live.peak
    if before <= budget.trigger:
        if report is not None:
            report.append(f'budget: {before} B live at most, within '
                          f'{budget.trigger} B ({budget.capacity} B and '
                          f'{budget.margin:.0%})')
        return body
    order = live.order
    written = _written(order)

    # Where each list's statements end, once a list (`spill_shared`).
    list_ends: Dict[int, List[int]] = {}

    def candidates() -> Dict[int, _Candidate]:
        cands: Dict[int, _Candidate] = {}
        for i, st in enumerate(order):
            if not st.target or st.target[0].id not in live.weight:
                continue
            if not _reloadable(st, written):
                continue
            vid = st.target[0].id
            list_id, k = live.place[i]
            starts = live.lists[list_id]
            ends = list_ends.get(list_id)
            if ends is None:
                ends = list_ends[list_id] = [live.span[p][1] for p in starts]
            reads = sorted({_ancestor(starts, ends, u)
                            for u in live.uses.get(vid, ())} - {None})
            if not reads or reads[-1] <= k:
                continue
            cands[vid] = _Candidate(vid, st, list_id, starts, ends, k,
                                    reads, live.weight[vid],
                                    live.last.get(vid, i))
        return cands

    # What putting back everything that pays would leave of the excess.
    left = _belady(live, candidates(), budget.capacity, budget.spill_price,
                   {}, {k: 0 for k in RELOAD_PRICE})[1]
    share = (max(0, left - budget.capacity)
             / max(1, before - budget.capacity))
    if budget.left_over >= 1:
        acts = True
    elif as_written is not None:
        excess = as_written - budget.capacity
        need = (1 - budget.left_over if budget.left_over >= 0
                else CLOSED_MIN)
        closed = (as_written - left) / max(1, excess)
        acts = excess >= EXCESS_MIN * budget.capacity and closed >= need
        share = 1 - closed
    else:
        acts = share <= (budget.left_over if budget.left_over >= 0
                         else LEFT_OVER)
    cands = candidates()
    reloads: Dict[int, Dict[int, List[int]]] = {}
    kinds: Dict[str, int] = {k: 0 for k in RELOAD_PRICE}
    count, _ = _belady(live, cands, budget.capacity,
                       budget.spill_price if acts else CHEAP, reloads, kinds)
    if taken is not None:
        for k, n in kinds.items():
            taken[k] = taken.get(k, 0) + n
    out = _put_back(body, reloads, cands, scratch) if count else body
    if report is not None:
        again = Liveness(out, explicit_simd, folded_crosslane)
        values = {v for k in reloads.values() for vs in k.values()
                  for v in vs}
        report.append(
            f'budget: {before} B live at most against {budget.capacity} B '
            f'(acting above {budget.trigger}'
            + (f', {as_written} B as written' if as_written is not None
               else '')
            + f'); what pays would leave '
            f'{left} B, {share:.0%} of the excess, so '
            f'{"what pays" if acts else "only constants"}: {count} put back '
            f'for {len(values)} value(s) of {len(cands)} that can be -- '
            f'{kinds["const"]} constant(s), {kinds["load"]} load(s), '
            f'{kinds["guarded"]} guarded; {again.peak} B now')
    return out


def _ancestor(starts: List[int], ends: List[int], p: int) -> Optional[int]:
    """The index of the statement of a list whose span holds position `p`,
    or None where none does."""
    k = bisect.bisect_right(starts, p) - 1
    if k >= 0 and p <= ends[k]:
        return k
    return None


def _gap_at(c: _Candidate, p: int):
    """`((first, last), next read)`: the stretch of positions around `p`
    between the read of `c` it is held from (or its definition) and the next,
    where `c` could be taken out; None where `p` is not in one -- a statement
    of its list that reads it holds `p`, or none reads it later."""
    a = _ancestor(c.starts, c.ends, p)
    if a is None or a <= c.at:
        return None
    j = bisect.bisect_left(c.reads, a)
    if j == len(c.reads) or c.reads[j] == a:
        return None
    # A read before `a` the value is held for, or its definition.
    previous = c.reads[j - 1] if j and c.reads[j - 1] >= c.at else c.at
    gap = (c.ends[previous] + 1, c.starts[c.reads[j]] - 1)
    if not gap[0] <= p <= gap[1]:
        return None
    return gap, c.reads[j]


def _belady(live: Liveness, cands: Dict[int, _Candidate], capacity: int,
            limit: int, reloads: Dict[int, Dict[int, List[int]]],
            kinds: Dict[str, int]) -> Tuple[int, int]:
    """Walk the statements in order and, at each one over `capacity`, take
    out of it the candidates read again furthest ahead whose reload costs no
    more than `limit` (`RELOAD_PRICE`), until it fits: a reload before the
    statement of the candidate's list that reads it next, recorded in
    `reloads` (list id -> index -> value ids), counted in `kinds` by what it
    reloads.  How many reloads, and the most left live at a statement."""
    profile = live.profile
    n = len(profile)
    # What a reload frees ends where the gap does: given back there.
    restore = [0] * (n + 1)
    running = 0
    count = 0
    most = 0
    by_def = sorted(cands.values(), key=lambda c: c.starts[c.at])
    nxt = 0
    active: List[_Candidate] = []
    for p in range(n):
        running += restore[p]
        if profile[p] + running > capacity:
            while (nxt < len(by_def)
                   and by_def[nxt].starts[by_def[nxt].at] < p):
                active.append(by_def[nxt])
                nxt += 1
            active = [c for c in active if c.last >= p]
            # Read again furthest ahead first, the heavier of two alike.
            here = sorted(((c, g) for c in active if c.price <= limit
                           for g in (_gap_at(c, p),) if g is not None),
                          key=lambda x: (x[0].starts[x[1][1]], x[0].weight),
                          reverse=True)
            for c, ((start, end), next_read) in here:
                if profile[p] + running <= capacity:
                    break
                running -= c.weight
                restore[end + 1] += c.weight
                reloads.setdefault(c.list, {}).setdefault(
                    next_read, []).append(c.vid)
                c.at = next_read
                kinds[c.kind] += 1
                count += 1
        most = max(most, profile[p] + running)
    return count, most




class _Spilled(_Candidate):
    """A value computed in the body that may go to shared memory: as
    `_Candidate`, with the slot it holds once it goes (`spill_shared`)."""

    __slots__ = ('slot', 'span')

    def __init__(self, *args):
        super().__init__(*args)
        self.kind = 'shared'
        self.slot: Optional[int] = None
        #: The positions the slot is held over: the definition to the last
        #: read.
        self.span = (self.starts[self.at], self.last)


def _spillable(s: Stmt, live: 'Liveness', mutated: Set[int]) -> bool:
    """Whether `s` defines a value one store and a load per gap can stand
    for: one number a lane, which nothing writes in place -- not an
    accumulator a call adds into, nor a declaration -- and that no raw text
    names."""
    if (len(s.target) != 1 or s.target[0].id not in live.weight
            or s.op in (Op.DECLARE, Op.ALLOC, Op.RAWSTMT)
            or s.attr('extern') or s.attr('escapes')):
        return False
    t = s.target[0].type
    return (isinstance(t, ScalarType) and getattr(t, 'length', None) is None
            and s.target[0].id not in mutated)


def spill_shared(body: Tuple[Stmt, ...], scratch, *, capacity: int,
                 shared: int, threads: int, arena: Optional[str] = None,
                 folded_crosslane: bool = False, volatile: bool = True,
                 report: Optional[List[str]] = None) -> Tuple[Stmt, ...]:
    """`body` with values computed in it spilled to shared memory where more
    is live than `capacity` (see the module), under SPMD: at each statement
    over it, of the values live there and not read there, the one read again
    furthest ahead goes -- stored once behind its definition into a slot of
    its own, one element a lane, and loaded from it before the statement of
    its list that reads it next -- until the statement fits, or no slot is
    free for the stretch from a value's definition to its last read.
    `shared` is the bytes the multiplication may take for the slots, of
    `threads` lanes.  `volatile`: the slots are accessed where the body
    accesses them, as often as it does; else a compiler may take a load back
    from the store or the load before, and keep the register after all.

    The slots are one buffer of the multiplication's arena (`arena`), held
    for the whole body and every lane's own (`barriers.own_buffers`): no
    barrier between a lane's store and its load, nothing else on its
    bytes."""
    live = Liveness(body, False, folded_crosslane)
    before = live.peak
    if before <= capacity:
        if report is not None:
            report.append(f'shared: {before} B live at most, within '
                          f'{capacity} B')
        return body
    order = live.order
    mutated: Set[int] = set()
    for st in order:
        if st.op == Op.CALL:
            for w in (st.attr('writes') or ()):
                if isinstance(w, Value):
                    mutated.add(w.id)
    cands: Dict[int, _Spilled] = {}
    # Where each list's statements end, once a list: a candidate's gaps are
    # between the statements of its list, and the top list of a derivative
    # holds tens of thousands.
    list_ends: Dict[int, List[int]] = {}
    for i, st in enumerate(order):
        if not st.target or not _spillable(st, live, mutated):
            continue
        vid = st.target[0].id
        list_id, k = live.place[i]
        starts = live.lists[list_id]
        ends = list_ends.get(list_id)
        if ends is None:
            ends = list_ends[list_id] = [live.span[p][1] for p in starts]
        reads = sorted({_ancestor(starts, ends, u)
                        for u in live.uses.get(vid, ())} - {None})
        if not reads or reads[-1] <= k:
            continue
        cands[vid] = _Spilled(vid, st, list_id, starts, ends, k, reads,
                              live.weight[vid], live.last.get(vid, i))
    elem = {c.load.target[0].type.base for c in cands.values()}
    if len(elem) != 1:
        # One buffer of one type: the body's numbers, where they are all of
        # one kind (the integers the model weighs go with the addresses).
        fp = max(elem, key=lambda e: sum(1 for c in cands.values()
                                         if c.load.target[0].type.base is e),
                 default=None)
        cands = {v: c for v, c in cands.items()
                 if c.load.target[0].type.base is fp}
    if not cands:
        return body
    size = next(iter(cands.values())).load.target[0].type.base.size()
    slots = shared // max(1, size * threads)
    if slots <= 0:
        return body
    busy: List[List[Tuple[int, int]]] = [[] for _ in range(slots)]

    def take_slot(c: _Spilled) -> bool:
        lo, hi = c.span
        for k, spans in enumerate(busy):
            if all(hi < a or b < lo for a, b in spans):
                spans.append((lo, hi))
                c.slot = k
                return True
        return False

    profile = live.profile
    n = len(profile)
    restore = [0] * (n + 1)
    running = 0
    reloads: Dict[int, Dict[int, List[int]]] = {}
    count = 0
    most = 0
    # Every gap of every candidate, between its definition or a read and its
    # next read in its list, by where it opens: Belady's choice over the gaps
    # open at a statement is the top of a heap of them, furthest next read
    # and heaviest first -- not a sort of every candidate at every statement,
    # which took a minute of the order-6 dense derivative.
    gaps: List[Tuple[int, int, int, int, _Spilled]] = []
    for c in cands.values():
        previous = c.at
        for r in c.reads:
            if r <= previous:
                continue
            first, last = c.ends[previous] + 1, c.starts[r] - 1
            if first <= last:
                gaps.append((first, last, r, c.starts[r], c))
            previous = r
    gaps.sort(key=lambda g: g[0])
    heap: List[Tuple[int, int, int, int, int, _Spilled]] = []
    nxt = 0
    for p in range(n):
        running += restore[p]
        while nxt < len(gaps) and gaps[nxt][0] <= p:
            first, last, r, at, c = gaps[nxt]
            heapq.heappush(heap, (-at, -c.weight, nxt, last, r, c))
            nxt += 1
        while profile[p] + running > capacity and heap:
            _, _, _, last, r, c = heapq.heappop(heap)
            if last < p:
                continue
            if c.slot is None and not take_slot(c):
                # No slot frees up later for the same stretch.
                continue
            running -= c.weight
            restore[last + 1] += c.weight
            reloads.setdefault(c.list, {}).setdefault(r, []).append(c.vid)
            count += 1
        most = max(most, profile[p] + running)
    spilled = {vid: c for vid, c in cands.items() if c.slot is not None}
    if not spilled:
        if report is not None:
            report.append(f'shared: {before} B live at most against '
                          f'{capacity} B; no value spilled ({slots} slot(s))')
        return body
    used = 1 + max(c.slot for c in spilled.values())
    out = _spill(body, spilled, reloads, scratch, used, threads, arena,
                 volatile)
    if report is not None:
        again = Liveness(out, False, folded_crosslane)
        report.append(
            f'shared: {before} B live at most against {capacity} B; '
            f'{len(spilled)} value(s) spilled to {used} of {slots} slot(s) '
            f'({used * threads * size} B), {count} reload(s); '
            f'{again.peak} B now')
    return out


def _spill(body: Tuple[Stmt, ...], spilled: Dict[int, _Spilled],
           reloads: Dict[int, Dict[int, List[int]]], scratch, slots: int,
           threads: int, arena: Optional[str],
           kept: bool = True) -> Tuple[Stmt, ...]:
    """`body` with the slots allocated at its head, each spilled value
    stored into its slot behind its definition, and loaded from it where
    `reloads` says, what follows in the list reading the load."""
    from .core import INDEX
    b = scratch()
    first = next(iter(spilled.values())).load.target[0].type.base
    buf = b.alloc(first, (slots * threads,), MemSpace.SHARED, hint='spill',
                  arena=arena, lifetime='body', own=True)
    lane = b.thread_id('x')
    head = list(b.finish())
    by_def = {id(c.load): c for c in spilled.values()}

    def slot_index(bb, c):
        return bb.op('add', INDEX, lane, c.slot * threads, hint='slot')

    def volatile(stmts, c):
        # Which value an access to a slot is of (`spilled`), for what reads
        # through the slots (`values._Guards`).  And made where they stand: a
        # compiler that took the load back from the store, or from the load
        # of the gap before, would keep the register the slot is for -- IGC
        # 2.41.5 kept 1665 of the order-8 dense `volume`'s 5245 loads back
        # and 1284 of its 1928 stores.
        mark = (('spilled', c.vid),) + ((('volatile', True),) if kept else ())
        return [replace(x, attrs=x.attrs + mark)
                if x.op in (Op.LOAD, Op.STORE) and x.args and x.args[0] is buf
                else x for x in stmts]

    def sub(x, mapping):
        return mapping.get(x.id, x) if isinstance(x, Value) else x

    def rewrite(stmts: Tuple[Stmt, ...], mapping: Dict[int, Value]
                ) -> Tuple[Stmt, ...]:
        here = reloads.get(id(stmts), {})
        out: List[Stmt] = []
        for k, s in enumerate(stmts):
            defined = by_def.get(id(s))
            for vid in here.get(k, ()):
                c = spilled.get(vid)
                if c is None:
                    continue
                old = c.load.target[0]
                bb = scratch()
                fresh = bb.load(buf, slot_index(bb, c), type_=old.type,
                                hint=old.hint, layout=old.layout)
                out.extend(volatile(bb.finish(), c))
                mapping[vid] = fresh
            if mapping:
                s = replace(
                    s, args=tuple(sub(a, mapping) for a in s.args),
                    predicate=(sub(s.predicate, mapping)
                               if s.predicate is not None else None),
                    attrs=tuple((key, sub(v, mapping)) for key, v in s.attrs))
            if s.regions:
                s = replace(s, regions=tuple(
                    replace(r, body=rewrite(r.body, dict(mapping)))
                    for r in s.regions))
            out.append(s)
            if defined is not None:
                bb = scratch()
                bb.store(buf, s.target[0], slot_index(bb, defined))
                out.extend(volatile(bb.finish(), defined))
        return tuple(out)

    return tuple(head) + rewrite(body, {})


def _put_back(body: Tuple[Stmt, ...], reloads: Dict[int, Dict[int, List[int]]],
              cands: Dict[int, _Candidate], scratch) -> Tuple[Stmt, ...]:
    """`body` with a copy of each candidate's load put back where `reloads`
    says, and what follows it in the list reading the copy."""
    b = scratch()

    def sub(x, mapping):
        return mapping.get(x.id, x) if isinstance(x, Value) else x

    def rewrite(stmts: Tuple[Stmt, ...], mapping: Dict[int, Value]
                ) -> Tuple[Stmt, ...]:
        here = reloads.get(id(stmts), {})
        out: List[Stmt] = []
        for k, s in enumerate(stmts):
            for vid in here.get(k, ()):
                load = cands[vid].load
                old = load.target[0]
                fresh = b.value(old.type, hint=old.hint, layout=old.layout,
                                uniform=old.uniformity, quals=old.quals)
                out.append(replace(
                    load, target=(fresh,),
                    args=tuple(sub(a, mapping) for a in load.args),
                    predicate=(sub(load.predicate, mapping)
                               if load.predicate is not None else None)))
                mapping[vid] = fresh
            if mapping:
                s = replace(
                    s, args=tuple(sub(a, mapping) for a in s.args),
                    predicate=(sub(s.predicate, mapping)
                               if s.predicate is not None else None),
                    attrs=tuple((key, sub(v, mapping)) for key, v in s.attrs))
            if s.regions:
                s = replace(s, regions=tuple(
                    replace(r, body=rewrite(r.body, dict(mapping)))
                    for r in s.regions))
            out.append(s)
        return tuple(out)

    return rewrite(body, {})
