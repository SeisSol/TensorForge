# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
# SPDX-FileContributor: David Schneller

"""Pseudo-IR: lowering to the existing ``backend.writer.Writer``.

This is the *only* file in ``pir`` that knows C++ exists.  Everything
vendor-specific goes through the context's lexic, so a second backend is a
handful of overrides here rather than a fork of the builders.

Declaration placement: the IR is SSA, C++ is not.  Values that a region yields
out are declared in the *parent* scope and assigned at the ``yield``; every
other value is declared at its definition.  That keeps the IR pure while the
generated code still looks like something a human would write.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from tensorforge.common.basic_types import Datatype

from .core import (Access, BufferType, Effect, IRError, MemSpace, Op, Operand,
                   Region, ScalarType, Stmt, Value)

# generic pure ops -> infix C++ operators
_INFIX = {
    'add': '+', 'sub': '-', 'mul': '*', 'div': '/',
    'lt': '<', 'le': '<=', 'gt': '>', 'ge': '>=', 'eq': '==', 'ne': '!=',
    'and': '&&', 'or': '||',
}


class Emitter:
    def __init__(self, writer, context: Any = None):
        self.writer = writer
        self.context = context
        self._names: Dict[int, str] = {}
        self._consts: Dict[int, str] = {}

    # -- naming ------------------------------------------------------------ #

    def name(self, v: Value) -> str:
        return self._names.setdefault(v.id, str(v))

    def bind(self, v: Value, text: str) -> None:
        self._names[v.id] = text

    def operand(self, x: Operand) -> str:
        if isinstance(x, Value):
            if x.id in self._consts:
                return self._consts[x.id]
            return self.name(x)
        if isinstance(x, bool):
            return 'true' if x else 'false'
        return str(x)

    # -- types ------------------------------------------------------------- #

    def ctype(self, t) -> str:
        if isinstance(t, ScalarType):
            if t.length is None:
                return t.base.ctype()
            return f'{t.base.ctype()}{t.length}'
        if isinstance(t, BufferType):
            return t.elem.ctype()
        raise IRError(f'cannot render type {t!r}')

    # -- lexic ------------------------------------------------------------- #

    def _lexic(self):
        if self.context is None:
            return None
        try:
            return self.context.get_vm().get_lexic()
        except Exception:
            return None

    def _sync(self) -> str:
        lex = self._lexic()
        if lex is not None and hasattr(lex, 'sync_threads'):
            return f'{lex.sync_threads()};'
        return '__syncthreads();'

    def _thread_idx(self, axis: str) -> str:
        lex = self._lexic()
        if lex is not None:
            attr = getattr(lex, f'thread_idx_{axis}', None)
            if attr is not None:
                return str(attr)
        return f'threadIdx.{axis}'

    # -- addressing -------------------------------------------------------- #

    def address(self, base: Operand, indices: Sequence[Operand]) -> str:
        idx = [self.operand(i) for i in indices]
        if not idx:
            return '0'
        shape: Optional[Tuple[int, ...]] = None
        if isinstance(base, Value) and isinstance(base.type, BufferType):
            shape = base.type.shape
        else:
            view = getattr(base, 'data_view', None)
            if view is not None:
                shape = tuple(view.shape)
        if shape is None or len(shape) != len(idx):
            return ' + '.join(idx)
        # leading dimension first, matching DataView.get_address
        addr = idx[-1]
        for i in reversed(range(len(idx) - 1)):
            addr = f'{idx[i]} + {shape[i]} * ({addr})'
        return addr

    def base_name(self, base: Operand) -> str:
        if isinstance(base, Value):
            return self.name(base)
        return getattr(base, 'name', str(base))

    # -- driver ------------------------------------------------------------ #

    def run(self, body: Tuple[Stmt, ...]) -> None:
        self._emit_body(body, ())

    def _emit_body(self, body: Tuple[Stmt, ...], yield_to: Tuple[str, ...]) -> None:
        for s in body:
            if s.predicate is not None and s.op not in (Op.YIELD,):
                with self.writer.If(self.operand(s.predicate)):
                    self._emit_stmt(s, yield_to)
            else:
                self._emit_stmt(s, yield_to)

    def _emit_stmt(self, s: Stmt, yield_to: Tuple[str, ...]) -> None:
        w = self.writer
        op = s.op

        if op == Op.CONST:
            v = s.target[0]
            lit = v.type.base.literal(s.attr('value'))
            self._consts[v.id] = lit
            return

        if op == Op.YIELD:
            for target, val in zip(yield_to, s.args):
                src = self.operand(val)
                if src != target:
                    w(f'{target} = {src};')
            return

        if op == Op.RAWSTMT:
            w(s.text)
            return

        if op == Op.RAWEXPR:
            v = s.target[0]
            text = s.text.format(*[self.operand(a) for a in s.args])
            w(f'{self.ctype(v.type)} {self.name(v)} = {text};')
            return

        if op == Op.RAWBLOCK:
            if s.attr('pragma'):
                w(f'#pragma {s.attr("pragma")}')
            with w.Block(s.text):
                self._emit_body(s.regions[0].body, yield_to)
            return

        if op == Op.BARRIER:
            w(self._sync())
            return

        if op == Op.ALLOC:
            v = s.target[0]
            t = v.type
            qual = {MemSpace.SHARED: '__shared__ ', MemSpace.CONSTANT: 'const '}.get(
                t.space, '')
            # Shared/scratch buffers are only *named* here; the byte offset is
            # assigned by opt.mem_region_allocation upstairs.  Until that pass
            # understands pir allocs, fall back to a plain declaration.
            w(f'{qual}{t.elem.ctype()} {self.name(v)}[{t.volume}];')
            return

        if op == Op.LOAD:
            v = s.target[0]
            addr = self.address(s.args[0], s.args[1:])
            w(f'{self.ctype(v.type)} {self.name(v)} = '
              f'{self.base_name(s.args[0])}[{addr}];')
            return

        if op == Op.STORE:
            addr = self.address(s.args[0], s.args[2:])
            w(f'{self.base_name(s.args[0])}[{addr}] = {self.operand(s.args[1])};')
            return

        if op == Op.CALL:
            callee = s.attr('callee')
            if callee is not None and callee.startswith('thread_idx_'):
                v = s.target[0]
                self.bind(v, self._thread_idx(callee[-1]))
                return
            v = s.target[0]
            args = ', '.join(self.operand(a) for a in s.args)
            w(f'{self.ctype(v.type)} {self.name(v)} = {callee}({args});')
            return

        if op == Op.FOR:
            self._emit_for(s)
            return

        if op == Op.IF:
            self._emit_if(s)
            return

        # generic pure op
        if s.target:
            v = s.target[0]
            args = [self.operand(a) for a in s.args]
            if op in _INFIX and len(args) == 2:
                expr = f'{args[0]} {_INFIX[op]} {args[1]}'
            elif op == 'fma' and len(args) == 3:
                expr = f'{args[0]} * {args[1]} + {args[2]}'
            elif op == 'select' and len(args) == 3:
                expr = f'{args[0]} ? {args[1]} : {args[2]}'
            elif op == 'neg' and len(args) == 1:
                expr = f'-{args[0]}'
            else:
                expr = f'{op}({", ".join(args)})'
            w(f'{self.ctype(v.type)} {self.name(v)} = {expr};')
            return

        raise IRError(f'no lowering for op {op!r}')

    # -- control flow ------------------------------------------------------ #

    def _emit_for(self, s: Stmt) -> None:
        w = self.writer
        lo, hi, step = s.loop_bounds
        ind = s.induction

        # iter_args and results share one C++ variable: no copy at the latch,
        # and the loop reads like an ordinary accumulator loop.
        targets: List[str] = []
        for arg, init, res in zip(s.iter_args, s.loop_inits, s.target):
            nm = self.name(arg)
            self.bind(res, nm)
            w(f'{self.ctype(arg.type)} {nm} = {self.operand(init)};')
            targets.append(nm)

        if s.attr('unroll'):
            w('#pragma unroll')
        i = self.name(ind)
        cmp_ = '<' if not isinstance(step, int) or step > 0 else '>'
        with w.For(f'int {i} = {self.operand(lo)}; '
                   f'{i} {cmp_} {self.operand(hi)}; '
                   f'{i} += {self.operand(step)}'):
            self._emit_body(s.regions[0].body, tuple(targets))

    def _emit_if(self, s: Stmt) -> None:
        w = self.writer
        targets: List[str] = []
        for res in s.target:
            nm = self.name(res)
            w(f'{self.ctype(res.type)} {nm};')
            targets.append(nm)

        with w.If(self.operand(s.cond)):
            self._emit_body(s.regions[0].body, tuple(targets))
        if len(s.regions) > 1:
            with w.Block('else'):
                self._emit_body(s.regions[1].body, tuple(targets))


def emit(body: Tuple[Stmt, ...], writer, context: Any = None) -> None:
    """Lower ``body`` into ``writer``."""
    Emitter(writer, context).run(body)
