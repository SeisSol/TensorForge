# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
# SPDX-FileContributor: David Schneller

"""Pseudo-IR: cache hints for the element the batch loop reaches next.

The stage of a transfer ahead of its request.  `move` issues a transfer
earlier in its iteration and `wrap` one element earlier; a hint asks for the
lines a transfer of the next element will read and moves nothing.  No value
appears, nothing waits on it, and a target without a data prefetch drops the
statement (`IRBuilder.prefetch`), so a hint in the wrong place costs a
memory request and not a number.  Two kinds, each its own switch:

``pointers``
    Under `Addressing.PTR_BASED` every address of an element hangs off one
    dependent load, its entry in the pointer array.  The hint asks for the
    next element's entry, at the head of the body.
``data``
    The next element's operands, at the tail of the body, behind the
    element guard -- where `wrap` would issue the transfer for the next
    element, with the transfer left where it is.  A pointer to that element
    is bound at the head, its computation cloned with the successor's index,
    and the tail asks for the whole of what the binding reaches, one span
    per `Target.prefetch_line_bytes`.

Which pointers and which transfers: those whose address is computed per
element in the body, each pointer once.  A batch-invariant operand is the
same data for every element and cached already.  The element is the one the
body is on, which a loop over groups of rows binds per row (`element`); the
successor's index takes its place in the copy, and what binds it is not
copied (`ElementLoop.dependencies`).

Both outside the element guard: element ``k`` being masked says nothing about
``k + 1``.  The entry of the pointer array is read at the successor's index,
which the loop clamps into range; the pointer in it is followed only for an
element the caller did not mask, under that element's flag, as `wrap` follows
it.  The flag goes by a name of its own (`allowed_hint`): the wrap runs behind
this pass and declares the one it reads at the same tail.  With the wrap as
well, a transfer it moves is hinted all the same; which ones move is decided
behind this pass.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Callable, List, Optional, Sequence, Tuple

from .ahead import Clone, ElementLoop, Refusal, behind_defs
from .core import BufferType, MemSpace, Op, Region, Stmt, Value
from .transfers import transfers


def prefetch_hints(body: Tuple[Stmt, ...], scratch: Callable[[], object], *,
                   pointers: bool = False, data: bool = False,
                   level: str = 'l2', line_bytes: int = 128,
                   report: Optional[List[str]] = None) -> Tuple[Stmt, ...]:
    """`body` with the hints for the next element in each batch loop.

    A batch loop is a `for` that names its successor index (`next`).
    `scratch` makes the builder for the statements this adds
    (`IRBuilder.scratch`); `level` is the cache asked for, `line_bytes` the
    span one hint covers.  `report` collects a line per hint: `+` and what it
    asks for, `-` and why a pointer or a loop has none.
    """
    if not (pointers or data):
        return body
    out: List[Stmt] = []
    changed = False
    for s in body:
        if s.op is not Op.FOR or s.attr('next') is None:
            out.append(s)
            continue
        try:
            loop = _Hints(ElementLoop(s, scratch, report), pointers=pointers,
                          data=data, level=level,
                          line_bytes=line_bytes).run()
        except Refusal as why:
            if report is not None:
                report.append(f'- loop: {why}')
            out.append(s)
            continue
        out.append(loop)
        changed = True
    return tuple(out) if changed else body


class _Hints:
    def __init__(self, l: ElementLoop, *, pointers: bool, data: bool,
                 level: str, line_bytes: int):
        self.l = l
        self.pointers = pointers
        self.data = data
        self.level = level
        self.line_bytes = line_bytes
        self.b = l.scratch()

    def _note(self, line: str) -> None:
        if self.l.report is not None:
            self.l.report.append(line)

    def _fresh(self, v: Value) -> Value:
        return self.b.value(v.type, hint=v.hint, uniform=v.uniformity,
                            layout=v.layout, quals=v.quals)

    # -- what to ask for ------------------------------------------------------ #

    def _entries(self) -> List[Stmt]:
        """The hints for the next element's entries in the pointer arrays:
        one per array a binding of the body reads its element's pointer
        out of."""
        out: List[Stmt] = []
        seen: set = set()
        for _, s in self.l.top():
            if not (s.attr('element_pointer') and self.l.names_element(s)):
                continue
            base = s.accesses[0].base
            if id(base) in seen:
                continue
            seen.add(id(base))
            b = self.l.scratch()
            b.prefetch(base, self.l.next, level=self.level)
            out += list(b.finish())
            self._note(f'+ {s.attr("extern")} [pointer]')
        return out

    def _operands(self) -> Tuple[List[Stmt], List[Stmt], bool]:
        """The pointers to the next element's operands, bound at the head;
        the hints for what they reach, for the tail; and whether a pointer
        followed there is an element's own."""
        heads: List[Stmt] = []
        tails: List[Stmt] = []
        owns = False
        clone = Clone(self._fresh, 'pf', set())
        clone.given(self.l.element, self.l.next)
        done: set = set()
        for t in transfers(self.l.scope):
            try:
                deps, own = self.l.dependencies(t)
            except Refusal as why:
                self._note(f'- {t.dest.hint or t.dest}: {why}')
                continue
            binding = _source_binding(t, deps)
            if binding is None or not any(self.l.names_element(s)
                                          for s in deps):
                continue
            if id(binding) in done:
                continue
            new = [s for s in deps if id(s) not in done]
            done.update(id(s) for s in new)
            heads += clone.stmts(new, named=True)
            ahead = clone.mapping[binding.target[0].id]
            tails += self._spans(ahead)
            owns = owns or own
            self._note(f'+ {binding.attr("extern")} [data]')
        return heads, tails, owns

    def _spans(self, ahead: Value) -> List[Stmt]:
        """Hints over all that `ahead` reaches, one per line."""
        extent = 1
        for size in ahead.type.shape:
            extent *= int(size)
        elem = ahead.type.elem.size() if ahead.type.elem is not None else 4
        per = max(1, self.line_bytes // elem)
        b = self.l.scratch()
        for start in range(0, extent, per):
            # The run as it is: a target that needs a message-sized length
            # rounds it, one that names a line ignores it.
            b.prefetch(ahead, start, level=self.level,
                       elems=min(per, extent - start))
        return list(b.finish())

    # -- the loop ------------------------------------------------------------- #

    def run(self) -> Stmt:
        l = self.l
        entries = self._entries() if self.pointers else []
        heads, tails, owns = self._operands() if self.data else ([], [],
                                                                 False)
        if not (entries or tails):
            raise Refusal('nothing to hint')
        flag: List[Stmt] = []
        if tails and owns and l.flag_word is not None:
            word_stmts, word = l.word(l.next, 'flagWordHint')
            flag_stmts, allowed = l.flag(word, 'allowed_hint')
            flag = word_stmts + flag_stmts
            b = l.scratch()
            with b.if_(allowed):
                for s in tails:
                    b.emit(s)
            tails = list(b.finish())
        region = l.region
        body = list(region.body)
        terminator = region.terminator
        if terminator is not None:
            body.pop()
        head = heads + entries + flag
        at = behind_defs(body, head)
        body[at:at] = head
        # Behind the element guard, and ahead of the barrier that closes
        # the iteration: the hints touch no shared memory, so nothing has to
        # wait for them to pass it.
        closing = max((i for i, s in enumerate(body) if s.op is Op.BARRIER
                       and i > at + len(head)), default=None)
        if closing is None:
            body += tails
        else:
            body[closing:closing] = tails
        if terminator is not None:
            body.append(terminator)
        return replace(l.loop, regions=(Region(args=region.args,
                                               body=tuple(body)),))


def _source_binding(t, deps: Sequence[Stmt]) -> Optional[Stmt]:
    """The binding of the global pointer `t` reads through: among its
    dependencies, the one that declares a buffer in global memory."""
    for s in reversed(deps):
        if any(isinstance(v.type, BufferType)
               and v.type.space is MemSpace.GLOBAL for v in s.target):
            return s
    return None
