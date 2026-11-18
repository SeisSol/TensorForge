# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT

"""Pseudo-IR: AMD's buffer engine (`Options.buffer_transfers`).

An AMD buffer instruction addresses global memory through a resource -- a
base, a size in bytes, and what an access past that size does: it reads zero
and writes nothing.  The resource sits in four scalar registers, the offset
in a vector register, 32 bits a lane where a global access takes 64.  What
that buys here is the guard: a predicate becomes an offset past the
resource (`tensorforge::bufferLoad`, one `v_cndmask`), where a guarded load
is a select -- or under `split_predicated_load` a clamped address -- and a
guarded store a branch.

The accesses it takes:

* a `load` or `store` of 4, 8 or 16 bytes a lane (`width`) of a buffer of
  32- or 64-bit numbers, a vector only of the buffer's own element;
* in global memory, through the binding of an operand the kernel reads at
  an address -- not one it takes the numbers of (`symbol.passed_by_value`),
  whose address may be a private copy no resource reaches.  The base is the
  binding where it is a value; where it is the operand's symbol, the access
  is written through its name (`Emitter.base_name`), and the binding is the
  value declared under that name in scope (`extern`);
* with no cache policy (`nontemporal`, `cache`: those stay global accesses
  for now), no value of their own where the predicate is false (`other`),
  and no lane limit of a full-lane tail (`valid`, `head`);
* on a base the wave holds alike.  The resource is the wave's: on a base
  that differs between its lanes the compiler reads it a lane at a time, in
  a loop around every access (a `readfirstlane` waterfall), which is
  correct and slow.  A base is the wave's where it is the same across every
  multiplication of the wave (`wave_level`): a batch-constant operator, and
  an element's binding where a multiplication is a whole number of waves.

A binding's level is joined over what its definition reads (`_Levels`): its
declaration states none (`IRBuilder.decl_expr`), and an element's pointer is
as uniform as the element index it is computed from.

The resource stands right behind the binding, in the same statement list,
so that it is in scope wherever the binding is; an access names it
(`resource`) and its engine (`engine='buffer'`).  Its size is the binding's
type: what the reads through it reach, one batch element of the operand as
it is stored (`ptr_manip.GetElementPtr._extent`).

Behind every pass that moves, clones or redirects an access
(`abstract_instruction._optimize_shared_body`).  `wrap` copies a read for
the next element onto that element's binding, which needs a resource of its
own, and `trips`, `shards` and `carry` point reads at shared memory and at
registers, which have none (`engines.kept_facts`): behind all of them, the
resource an access names is its binding's by construction.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Tuple

from tensorforge.common.basic_types import Datatype

from .core import (BufferType, Effect, MemSpace, Op, Stmt, Uniformity, Value,
                   def_use)
from .engines import ResourceType

#: Bytes a lane moves through the engine at once: `buffer_load_dword`,
#: `_dwordx2` and `_dwordx4`, and the stores.
WIDTHS = (4, 8, 16)

#: The numbers a buffer of the engine holds: whole 32-bit words, which the
#: helpers move as such (`tensorforge::bufferLoad`).
ELEMENTS = frozenset({Datatype.F32, Datatype.F64, Datatype.I32, Datatype.U32,
                      Datatype.I64, Datatype.SIZE})

#: What an access may ask for that the engine does not spell: a cache
#: policy, a value of its own where the predicate is false -- an access past
#: the resource reads zero -- the lane limits of a full-lane tail, and a
#: spill's `volatile`.
REFUSED = ('nontemporal', 'cache', 'other', 'valid', 'head', 'volatile')

#: How deep a binding's definition is followed before its level is taken as
#: the lane's: an element's binding is a few statements from its loop.
_DEPTH = 64


def wave_level(threads: Optional[int], wave: int) -> Uniformity:
    """The level a base has to reach to be the same on every lane of a wave:
    a multiplication's own where it is a whole number of waves -- the wave
    then holds one -- and otherwise the multiplications' that share one.
    Without the lane count, the latter."""
    if threads is not None and threads >= wave and threads % wave == 0:
        return Uniformity.MULT
    return Uniformity.MULTGROUP


def _by_value(operand) -> bool:
    """Whether an access reaches an operand the kernel takes the numbers of
    (`symbol.passed_by_value`): an argument, which clang may copy into
    private memory, at an address no resource reaches."""
    return (getattr(operand, 'embedded', None) is not None
            or bool(getattr(getattr(operand, 'obj', None), 'passed_by_value',
                            False)))


def buffer_base(s: Stmt, named: Optional[Mapping[str, Value]] = None
                ) -> Optional[Value]:
    """The binding `s` reads or writes through, where the engine takes the
    access (the module's list), else None.  `named` holds what the names in
    scope at `s` are declared as, which is where an access through an
    operand's symbol finds its binding.  The binding's level is the
    caller's to weigh."""
    if s.op == Op.LOAD:
        effect, value = Effect.READ, (s.target[0] if s.target else None)
    elif s.op == Op.STORE:
        effect, value = Effect.WRITE, (s.args[1] if len(s.args) > 1 else None)
    else:
        return None
    if s.effect != effect or s.regions or s.attr('engine') is not None:
        return None
    if any(s.attr(k) not in (None, False) for k in REFUSED):
        return None
    base = s.args[0] if s.args else None
    if not isinstance(base, Value):
        if _by_value(base):
            return None
        base = (named or {}).get(getattr(base, 'name', None))
    if not (isinstance(base, Value) and isinstance(base.type, BufferType)
            and base.type.space is MemSpace.GLOBAL):
        return None
    if (len(s.accesses) != 1 or s.accesses[0].space is not MemSpace.GLOBAL
            or _by_value(s.accesses[0].base)):
        return None
    elem = base.type.elem
    if elem not in ELEMENTS:
        return None
    t = getattr(value, 'type', None)
    length = getattr(t, 'length', None)
    if length is not None and getattr(t, 'base', None) != elem:
        # Reinterpreted rather than converted (`IRBuilder._converting_store`).
        return None
    if elem.size() * (length or 1) not in WIDTHS:
        return None
    return base


def _names(s: Stmt) -> Iterable[Tuple[str, Value]]:
    """The names `s` declares its values under (`extern`), which shadow
    what an enclosing scope declared under them."""
    name = s.attr('extern')
    if isinstance(name, str) and len(s.target) == 1:
        yield name, s.target[0]


def verify_buffer(s: Stmt) -> List[str]:
    """What is wrong with `s`, a `load` or `store` through the engine or a
    `resource`: an access names a resource and reaches global memory with
    nothing the engine does not spell, and a resource is one value over a
    binding in global memory and its size."""
    diag: List[str] = []
    if s.op == Op.RESOURCE:
        if len(s.target) != 1 or not isinstance(s.target[0].type,
                                                ResourceType):
            diag.append('resource: produces exactly one resource')
        if (len(s.args) != 2 or not isinstance(s.args[0], Value)
                or not isinstance(s.args[0].type, BufferType)
                or s.args[0].type.space is not MemSpace.GLOBAL
                or not isinstance(s.args[1], int) or s.args[1] <= 0):
            diag.append('resource: takes a buffer in global memory and its '
                        'size in bytes')
        if s.predicate is not None or s.accesses or s.effect != Effect.NONE:
            diag.append('resource: reads no memory and is not predicated')
        return diag
    resource = s.attr('resource')
    if not (isinstance(resource, Value)
            and isinstance(resource.type, ResourceType)):
        diag.append(f'{s.op}: through the buffer engine without a resource')
    if any(a.space is not MemSpace.GLOBAL for a in s.accesses):
        diag.append(f'{s.op}: through the buffer engine outside global '
                    f'memory')
    refused = [k for k in REFUSED if s.attr(k) not in (None, False)]
    if refused:
        diag.append(f'{s.op}: through the buffer engine, which does not spell '
                    f'{", ".join(refused)}')
    return diag


class _Levels:
    """How uniform a value is: the level it states, joined over what its
    definition reads.  A pointer binding states the grid's whatever it is
    computed from (`IRBuilder.decl_expr`), where it is as uniform as its
    element index; a region's argument reads what its loop starts it at and
    hands it on."""

    def __init__(self, body: Tuple[Stmt, ...]):
        self._defs, _ = def_use(body)
        self._memo: Dict[int, Uniformity] = {}

    def __call__(self, x, depth: int = 0) -> Uniformity:
        if not isinstance(x, Value):
            return Uniformity.GRID
        got = self._memo.get(x.id)
        if got is not None:
            return got
        if depth > _DEPTH:
            return Uniformity.LANE
        # What a loop hands its carried value reads that value again: on the
        # way round, it is what it states.
        self._memo[x.id] = x.uniformity
        level = x.uniformity
        for y in self._reads(x):
            level = min(level, self(y, depth + 1))
        self._memo[x.id] = level
        return level

    def _reads(self, x: Value) -> Iterable:
        s = self._defs.get(x.id)
        if s is None:
            return ()
        if any(t.id == x.id for t in s.target):
            # Of a construct, what any of its regions hands out as well.
            return tuple(s.operands()) + tuple(
                y for r in s.regions for y in r.yielded)
        if s.op == Op.FOR:
            if x.id == s.induction.id:
                return s.loop_bounds
            k = [a.id for a in s.iter_args].index(x.id)
            return (s.loop_inits[k],) + tuple(s.regions[0].yielded[k:k + 1])
        if s.op == Op.WHILE:
            return (s.loop_init,) + tuple(s.regions[0].yielded)
        return ()


def buffer_transfers(body: Tuple[Stmt, ...],
                     scratch: Callable[[], object],
                     threads: Optional[int] = None,
                     wave: int = 64,
                     report: Optional[list] = None) -> Tuple[Stmt, ...]:
    """`body` with every access the engine takes through a buffer resource
    of its binding (the module's list), and the resource behind the
    binding's definition.  `threads` is the lanes of a multiplication and
    `wave` the target's, which decide what the wave holds alike
    (`wave_level`); `scratch` builds the resources' values, numbered from
    the body's."""
    levels = _Levels(body)
    least = wave_level(threads, wave)
    bindings: Dict[int, Value] = {}
    counts: Dict[int, int] = {}

    def taken(s: Stmt, named: Mapping[str, Value]) -> Optional[Value]:
        base = buffer_base(s, named)
        if base is None:
            return None
        nbytes = base.type.volume * base.type.elem.size()
        if not 0 < nbytes < 1 << 31 or levels(base) < least:
            return None
        return base

    def scan(stmts: Tuple[Stmt, ...], named: Dict[str, Value]) -> None:
        named = dict(named)
        for s in stmts:
            for r in s.regions:
                scan(r.body, named)
            base = taken(s, named)
            if base is not None:
                bindings[base.id] = base
                counts[base.id] = counts.get(base.id, 0) + 1
            named.update(_names(s))

    scan(body, {})
    if not bindings:
        return body

    b = scratch()
    made: Dict[int, Stmt] = {}
    for vid, base in bindings.items():
        r = b.value(ResourceType(base.type.elem), hint='rsrc',
                    uniform=levels(base))
        made[vid] = Stmt(op=Op.RESOURCE, target=(r,),
                         args=(base, base.type.volume * base.type.elem.size()),
                         pure=True, movable=True, effect=Effect.NONE)
        if report is not None:
            report.append(f'buffer: {base} through {r}, {counts[vid]} '
                          f'accesses, {made[vid].args[1]} B')
    placed = set()

    def behind(values) -> Tuple[Stmt, ...]:
        out = []
        for v in values:
            if v.id in made and v.id not in placed:
                placed.add(v.id)
                out.append(made[v.id])
        return tuple(out)

    def rewrite(stmts: Tuple[Stmt, ...],
                named: Dict[str, Value]) -> Tuple[Stmt, ...]:
        named = dict(named)
        out = []
        for s in stmts:
            if s.regions:
                s = replace(s, regions=tuple(
                    replace(r, body=behind(r.args) + rewrite(r.body, named))
                    for r in s.regions))
            else:
                base = taken(s, named)
                if base is not None:
                    s = replace(s, attrs=s.attrs + (
                        ('engine', 'buffer'),
                        ('resource', made[base.id].target[0])))
            out.append(s)
            out.extend(behind(s.target))
            named.update(_names(s))
        return tuple(out)

    inner = rewrite(body, {})
    # A binding the body does not define is in scope throughout it.
    return behind(bindings.values()) + inner
