# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Each multiplication's row of a group, read as the wave's.

Where the multiplications of one wave take their trips through the batch
loop as a group (`BatchLoop._gen_grouped_ir`), a row is the group's first
element plus the row's place in the group, `threadIdx.y` modulo the group.
What a row reads through it is read per lane: its flag word and the pointers
of its element, where the lanes of one multiplication load one address and
those of the next another.  The compiler issues each as a vector load, since
it cannot tell that the lanes of a multiplication agree.

The group's first element is the same on every lane of the wave once the
loop reads it as the wave's, and so is each multiplication's row once its
place is a constant -- and what it reads, which the compiler then loads with
a scalar load.  `group_rows` gives a statement that reads memory through the
place one copy per multiplication, the place that multiplication's, and
replaces the statement by a pick among the copies on the lane's place
(`Lexic.pick`): as many scalar loads as the group has multiplications where
there was one vector load, and a select per 32-bit word.

It takes on trust only what the builder marks.  The place carries
`mult_lane`, the number of multiplications in the group, and a statement
that reads through it and is to be copied carries `per_mult`.  A copy takes
along what the statement reads through the place -- the row, the flag, the
element they decide -- once per multiplication and scope, and whatever reads
no memory stays the lane's.

Behind `wrap`, whose copies of a binding for the first group and the next
read through the same place and are given theirs alike.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Callable, Dict, List, Sequence, Tuple

from .ahead import Clone
from .core import Effect, Stmt, Value, walk_stmts

#: What a statement does that a copy for another multiplication may not do
#: again.
_SIDE = Effect.WRITE | Effect.ATOMIC | Effect.BARRIER | Effect.UNKNOWN


def group_rows(body: Tuple[Stmt, ...], scratch,
               pick: Callable[[str, Sequence[str]], str]) -> Tuple[Stmt, ...]:
    """`body` with every statement marked `per_mult` that reads through a
    place marked `mult_lane` replaced by a pick among copies of it, one per
    multiplication of the group.

    `scratch` makes the builder the copies' values are numbered by, and
    `pick` spells the pick: the place and one value per multiplication, as
    text (`Lexic.pick`).
    """
    walked = walk_stmts(body)
    places: Dict[int, Tuple[Value, int]] = {
        s.target[0].id: (s.target[0], int(s.attr('mult_lane')))
        for s in walked if s.attr('mult_lane') and s.target}
    if not places:
        return body
    order: Dict[int, int] = {}
    defs: Dict[int, Stmt] = {}
    # What a value follows: the place it is computed from, through
    # statements a copy may compute again -- every operand computed from a
    # place, so that a copy computes all of what it reads through it.
    follows: Dict[int, int] = {}
    # Everything computed from a place, by any statement.
    tainted = set(places)
    for i, s in enumerate(walked):
        order[id(s)] = i
        for t in s.target:
            defs[t.id] = s
        reached = [v.id for v in s.operands() if v.id in tainted]
        if not reached:
            continue
        tainted.update(t.id for t in s.target)
        if s.regions or not s.target or s.predicate is not None or (
                s.effect & _SIDE):
            continue
        clean = [vid for vid in reached if vid in places or vid in follows]
        if len(clean) != len(reached):
            continue
        place = {vid if vid in places else follows[vid] for vid in clean}
        if len(place) == 1:
            for t in s.target:
                follows[t.id] = place.pop()
    taken = {str(s.attr('extern')) for s in walked if s.attr('extern')}
    b = scratch()

    def fresh(v: Value) -> Value:
        return b.value(v.type, hint=v.hint, uniform=v.uniformity,
                       layout=v.layout, quals=v.quals)

    def chain(s: Stmt) -> List[Stmt]:
        """What `s` reads through its place, in the order it runs."""
        found: Dict[int, Stmt] = {}
        wanted = [v.id for v in s.operands()]
        while wanted:
            vid = wanted.pop()
            if vid in places or vid not in follows:
                continue
            d = defs[vid]
            if id(d) not in found:
                found[id(d)] = d
                wanted.extend(v.id for v in d.operands())
        return sorted(found.values(), key=lambda x: order[id(x)])

    def copier(value: Value, mult: int) -> Clone:
        clone = Clone(fresh, f'mult{mult}', taken)
        clone.given(value, mult)
        return clone

    def rewrite(stmts: Sequence[Stmt],
                copiers: Dict[int, List[Clone]]) -> List[Stmt]:
        out: List[Stmt] = []
        for s in stmts:
            if s.regions:
                regions = tuple(
                    replace(r, body=tuple(rewrite(
                        r.body, {p: [c.fork() for c in cs]
                                 for p, cs in copiers.items()})))
                    for r in s.regions)
                out.append(replace(s, regions=regions))
                continue
            place = follows.get(s.target[0].id) if s.target else None
            reads = chain(s) if (place is not None
                                 and s.attr('per_mult')) else None
            if reads is None:
                out.append(s)
                continue
            value, count = places[place]
            mine = copiers.get(place)
            if mine is None:
                mine = copiers[place] = [copier(value, m)
                                         for m in range(count)]
            picked = []
            for clone in mine:
                todo = [x for x in reads + [s]
                        if x.target[0].id not in clone.mapping]
                out.extend(_unmarked(x)
                           for x in clone.stmts(todo, named=True))
                picked.append(clone.mapping[s.target[0].id])
            out.append(replace(
                s, args=(value,) + tuple(picked),
                text=pick('{0}', [f'{{{m + 1}}}' for m in range(count)]),
                pure=True, effect=Effect.NONE, accesses=(),
                attrs=_unmarked(s).attrs))
        return out

    return tuple(rewrite(body, {}))


def _unmarked(s: Stmt) -> Stmt:
    """`s` without the mark that asked for copies, which it has been given."""
    if s.attr('per_mult') is None:
        return s
    return replace(s, attrs=tuple((k, v) for k, v in s.attrs
                                  if k != 'per_mult'))
