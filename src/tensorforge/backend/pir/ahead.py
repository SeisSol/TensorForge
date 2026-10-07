# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
# SPDX-FileContributor: David Schneller

"""Pseudo-IR: what issuing something for another element than the body's
takes.

The batch loop computes one element per iteration.  A pass that issues work
for the element ahead -- the transfer for it, as `wrap` does -- needs these
things of the loop:

* the loop taken apart: its index, its successor and first element, the
  element guard and the per-element statements it holds (`ElementLoop`);
* what of the body the work reads, to compute again for that element, and
  whether that follows a pointer of the element's own
  (`ElementLoop.dependencies`);
* those statements again, for the element (`Clone`);
* an index one stride on, and an element's flag where the loop has a mask
  (`ElementLoop.successor`, `ElementLoop.word`, `ElementLoop.flag`).
"""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Callable, Dict, List, Sequence, Tuple

from tensorforge.common.basic_types import Datatype

from .core import (BOOL, SCALAR_LAYOUT, SIZE, Effect, MemSpace, Op, Region,
                   ScalarType, Stmt, Value, walk_stmts)
from .transfers import Transfer, opaque, same, writes

#: The effects a statement computed again for another element may not have.
_SIDE = Effect.WRITE | Effect.ATOMIC | Effect.BARRIER | Effect.UNKNOWN


class Refusal(Exception):
    """Why a transfer, or a loop, was left as it is.  Carried rather than
    logged: the caller asked for a transformation and is entitled to the
    reason it did not happen."""


class ElementLoop:
    """A batch loop, taken apart: its body, the element guard in it and the
    per-element statements the guard holds."""

    def __init__(self, loop: Stmt, scratch, report=None,
                 around: Sequence[Stmt] = ()):
        self.loop = loop
        #: Makes the builder the added statements are built with
        #: (`IRBuilder.scratch`).
        self.scratch = scratch
        self.report = report
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

    def top(self) -> List[Tuple[Tuple[int, int], Stmt]]:
        """Every statement at the top of the loop's body or of the guard,
        keyed by the order it runs in."""
        out = []
        for i, s in enumerate(self.region.body):
            if i == self.guard_at:
                out.extend(((i, j + 1), x) for j, x in enumerate(self.scope))
            else:
                out.append(((i, 0), s))
        return out

    def ahead_of(self, first: Stmt) -> List[Stmt]:
        """What stands ahead of `first` in its own iteration."""
        out = []
        for _, s in self.top():
            if s is first:
                return out
            out.append(s)
        raise AssertionError('the transfer is not at the top of the loop')

    def names_index(self, s: Stmt) -> bool:
        return any(v.id == self.k.id for v in s.operands())

    def dependencies(self, t: Transfer) -> Tuple[List[Stmt], bool]:
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
        for key, s in self.top():
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
            if any(writes(a) or opaque(a) or a.space is not MemSpace.GLOBAL
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
            if any(writes(a) and any(same(a.base, r) for r in roots)
                   for a in x.accesses):
                raise Refusal('the body writes memory the transfer\'s address '
                              'is computed from')
        return deps, any(s.attr('element_pointer') for s in deps)

    def word(self, index, name: str) -> Tuple[List[Stmt], Value]:
        """`index`'s flag, read as the word it is stored as."""
        b = self.scratch()
        v = b.decl_expr(f'const uint32_t {name}', self.flag_word,
                        ScalarType(Datatype.U32), None, args=(index,),
                        kind=Effect.READ, space=MemSpace.GLOBAL, hint=name,
                        extern=name, layout=SCALAR_LAYOUT)
        return list(b.finish()), v

    def flag(self, word, name: str) -> Tuple[List[Stmt], Value]:
        """A flag word as the condition it stands for."""
        b = self.scratch()
        v = b.decl_expr(f'const bool {name}', 'static_cast<bool>({0})', BOOL,
                        None, args=(word,), hint=name, extern=name)
        return list(b.finish()), v

    def successor(self, index, hint: str) -> Tuple[List[Stmt], Value]:
        """`index` one stride on, clamped the way the loop clamps its own
        successor: the element a transfer issued there is for."""
        b = self.scratch()
        _, count, stride = self.loop.loop_bounds
        ahead = b.op('add', SIZE, index, stride, hint=f'{hint}Ahead')
        inside = b.op('lt', BOOL, ahead, count, hint=f'{hint}In')
        v = b.op('select', SIZE, inside, ahead, index, hint=hint)
        return list(b.finish()), v


_IDENTIFIER = re.compile(r'\b[A-Za-z_]\w*\b')


class Clone:
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


def behind_defs(stmts: List[Stmt], clones: Sequence[Stmt]) -> int:
    """The first position in `stmts` behind every statement that defines a
    value `clones` read."""
    wanted = {v.id for s in clones for x in walk_stmts((s,))
              for v in x.operands()}
    at = 0
    for i, s in enumerate(stmts):
        if any(v.id in wanted for v in s.target):
            at = i + 1
    return at
