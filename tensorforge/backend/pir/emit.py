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
                   Region, ScalarType, Stmt, TokenType, Value, def_use, walk)

_ATOM = __import__('re').compile(r'^(?:[A-Za-z_][A-Za-z0-9_.:]*|\d[\w.]*)$')


def _unwrap(expr: str) -> str:
    """Drop one redundant paren level: `Writer.If` adds its own."""
    if not (expr.startswith('(') and expr.endswith(')')):
        return expr
    depth = 0
    for i, c in enumerate(expr):
        depth += (c == '(') - (c == ')')
        if depth == 0 and i < len(expr) - 1:
            return expr                 # e.g. "(a) && (b)" -- not a wrapper
    return expr[1:-1]


def _atomic(expr: str) -> bool:
    """An identifier or literal needs no parentheses when inlined."""
    return bool(_ATOM.match(expr))


def _sm_at_least(model: str, minimum: int) -> bool:
    digits = ''.join(c for c in str(model)[3:] if c.isdigit())
    return str(model).startswith('sm_') and bool(digits) and int(digits) >= minimum


# Which architectures actually have the asynchronous global -> shared path.
# The lexic knows how the call *looks*; this table knows whether it *exists*.
_ASYNC_ARCH = {
    'nvidia': lambda m: _sm_at_least(m, 80),                    # cp.async
    'amd': lambda m: str(m) in ('gfx90a', 'gfx940', 'gfx941',   # global_load_lds
                                'gfx942', 'gfx950'),
}


# generic pure ops -> infix C++ operators
_INFIX = {
    'add': '+', 'sub': '-', 'mul': '*', 'div': '/',
    'rem': '%', 'bitand': '&', 'bitor': '|', 'shl': '<<', 'shr': '>>',
    'lt': '<', 'le': '<=', 'gt': '>', 'ge': '>=', 'eq': '==', 'ne': '!=',
    'and': '&&', 'or': '||',
}


class Emitter:
    def __init__(self, writer, context: Any = None):
        self.writer = writer
        self.context = context
        self._names: Dict[int, str] = {}
        self._consts: Dict[int, str] = {}
        self._async_lex = None
        self._async_note = ''
        self._inline: set = set()
        self._pending: Dict[int, str] = {}   # load.async token id -> C++ name

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
        if isinstance(t, TokenType):
            raise IRError('a completion token has no C++ representation; it '
                          'must not escape into generated code')
        if isinstance(t, ScalarType):
            if t.length is None:
                return t.base.ctype()
            lex = self._lexic()
            if lex is not None:
                return lex.get_fptype(t.base.ctype(), t.length)
            return f'{t.base.ctype()}{t.length}'
        if isinstance(t, BufferType):
            return t.elem.ctype()
        raise IRError(f'cannot render type {t!r}')

    # -- lexic ------------------------------------------------------------- #

    def _vm(self):
        """Accepts either a ``Context`` or a ``VM`` as ``context``."""
        if self.context is None:
            return None
        if hasattr(self.context, 'get_vm'):
            return self.context.get_vm()
        if hasattr(self.context, 'get_lexic'):
            return self.context
        return None

    def _lexic(self):
        vm = self._vm()
        return None if vm is None else vm.get_lexic()

    def _hw(self):
        vm = self._vm()
        return None if vm is None else vm.get_hw_descr()

    def _sync(self) -> str:
        lex = self._lexic()
        if lex is not None:
            return f'{lex.sync_block()};'
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

    def elem_size(self, base: Operand) -> int:
        if isinstance(base, Value) and isinstance(base.type, BufferType):
            return base.type.elem.size()
        dt = getattr(base, 'datatype', None)
        return dt.size() if dt is not None else 4

    def _decide_async(self, body: Tuple[Stmt, ...]) -> None:
        """One decision per kernel body, not per copy.

        The wait counters are a single hardware resource: if even one copy has
        to take the synchronous fallback, the counting no longer describes what
        is in flight, so the whole body goes synchronous.  Mixed mode would be
        silently wrong rather than merely slow.
        """
        self._async_lex = None
        copies = [s for s, _ in walk(body) if s.op == Op.COPY_ASYNC]
        if not copies:
            return

        lex, hw = self._lexic(), self._hw()
        if lex is None or hw is None:
            self._async_note = 'no hardware description available'
            return
        supported = _ASYNC_ARCH.get(getattr(hw, 'vendor', None))
        if supported is None or not supported(getattr(hw, 'model', '')):
            self._async_note = f'{getattr(hw, "model", "?")} has no async copy path'
            return

        sizes = lex.copy_async_sizes()
        for c in copies:
            nbytes = c.attr('elems', 1) * self.elem_size(c.copy_dst)
            if nbytes not in sizes:
                self._async_note = (f'{nbytes} B per thread is not one of '
                                    f'{sizes} on {hw.model}')
                return
        self._async_lex = lex

    def zero(self, t) -> str:
        return t.base.literal(0)

    def declare(self, v: Value, expr: str, s: Stmt, name: str = None) -> None:
        """Emit `Ty name = expr;`, folding a predicate into a select.

        A predicated statement that produces a value must *not* be wrapped in a
        guard block --- the declaration would be scoped inside it and the value
        would be unusable afterwards.  Lowering to a select also keeps the
        statement hoistable, which a guard region never is.
        """
        if s.predicate is not None:
            other = s.attr('other')
            other = (self.operand(other) if other is not None
                     else self.zero(v.type))
            expr = f'{self.operand(s.predicate)} ? ({expr}) : ({other})'
        if name is None and v.id in self._inline:
            self.bind(v, _atomic(expr) and expr or f'({expr})')
            return
        self.writer(f'{self.ctype(v.type)} {name or self.name(v)} = {expr};')

    def base_name(self, base: Operand) -> str:
        if isinstance(base, Value):
            return self.name(base)
        return getattr(base, 'name', str(base))

    # -- driver ------------------------------------------------------------ #

    def _plan_inlining(self, body: Tuple[Stmt, ...]) -> set:
        """Values that should become expressions rather than declarations.

        A pure single-use value is written straight into its consumer, so a
        migrated construct emits as compactly as the string it replaces.
        Without this every structured op leaves a named temporary behind, and
        the generated source grows with the migration instead of staying
        comparable to it.

        The use has to sit *directly* in the same region: pushing a
        computation into a nested loop would change how often it runs.
        """
        _, uses = def_use(body)
        inline: set = set()

        def scan(stmts: Tuple[Stmt, ...]) -> None:
            here: Dict[int, int] = {}
            for s in stmts:
                for v in s.operands():
                    here[v.id] = here.get(v.id, 0) + 1

            pending: set = set()
            for s in stmts:
                for v in s.operands():
                    if v.id in pending:
                        inline.add(v.id)
                        pending.discard(v.id)
                if s.has_side_effects or s.regions:
                    # Everything still pending has to materialize here.  The
                    # arithmetic is pure, so moving it past a wait or a store
                    # would not change the result -- but it would undo a
                    # deliberately interleaved schedule, which is the whole
                    # point of having placed the statement where it is.
                    pending.clear()
                if (s.pure and not s.regions and not s.has_side_effects
                        and s.effect == Effect.NONE and len(s.target) == 1
                        and s.op != Op.CONST and not s.attr('escapes')):
                    t = s.target[0]
                    if len(uses.get(t.id, ())) == 1 and here.get(t.id, 0) == 1:
                        pending.add(t.id)
                for r in s.regions:
                    scan(r.body)

        scan(body)
        return inline

    def run(self, body: Tuple[Stmt, ...]) -> None:
        self._inline = self._plan_inlining(body)
        self._decide_async(body)
        if self._async_lex is None and self._async_note:
            self.writer.Comment(f'async copies lowered synchronously: '
                                f'{self._async_note}')
        self._emit_body(body, ())

    def _emit_body(self, body: Tuple[Stmt, ...],
                   yield_to: Tuple[Optional[str], ...]) -> None:
        for s in body:
            declares = s.op in Op.DECLARING or (s.target and not s.regions)
            if s.predicate is not None and s.op != Op.YIELD and not declares:
                with self.writer.If(self.operand(s.predicate)):
                    self._emit_stmt(s, yield_to)
            else:
                self._emit_stmt(s, yield_to)

    def _emit_stmt(self, s: Stmt, yield_to: Tuple[Optional[str], ...]) -> None:
        w = self.writer
        op = s.op

        if op == Op.CONST:
            v = s.target[0]
            lit = v.type.base.literal(s.attr('value'))
            self._consts[v.id] = lit
            return

        if op == Op.YIELD:
            for target, val in zip(yield_to, s.args):
                if target is None:      # token: lives only in the IR
                    continue
                src = self.operand(val)
                if src != target:
                    w(f'{target} = {src};')
            return

        if op == Op.RAWSTMT:
            if s.attr('bare_newline'):
                self.writer.Emptyline()
            elif s.attr('fmt'):
                w(s.text.format(*[self.operand(a) for a in s.args]))
            else:
                w(s.text)
            return

        if op == Op.RAWEXPR:
            v = s.target[0]
            text = s.text.format(*[self.operand(a) for a in s.args])
            self.declare(v, text, s)
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
            self.declare(v, f'{self.base_name(s.args[0])}[{addr}]', s)
            return

        if op == Op.LOAD_ASYNC:
            tok = s.target[0]
            t = s.attr('types', ())[0]
            name = f'v{tok.id}_{s.attr("hint", "ld")}'
            self._pending[tok.id] = name
            addr = self.address(s.load_base, s.load_index)
            # On AMD this is an ordinary global_load whose s_waitcnt we place
            # ourselves; on NVIDIA the scoreboard stalls at the first use and
            # the matching wait lowers to nothing.
            self.declare(Value(tok.id, t), f'{self.base_name(s.load_base)}[{addr}]',
                         s, name=name)
            return

        if op == Op.STORE:
            addr = self.address(s.args[0], s.args[2:])
            w(f'{self.base_name(s.args[0])}[{addr}] = {self.operand(s.args[1])};')
            return

        if op == Op.COPY_ASYNC:
            dst_b = self.base_name(s.copy_dst)
            src_b = self.base_name(s.copy_src)
            dst_a = self.address(s.copy_dst, s.copy_dst_index)
            src_a = self.address(s.copy_src, s.copy_src_index)
            elems = s.attr('elems', 1)
            if self._async_lex is not None:
                nbytes = elems * self.elem_size(s.copy_dst)
                w(self._async_lex.copy_async(f'&{dst_b}[{dst_a}]',
                                             f'&{src_b}[{src_a}]', nbytes))
                commit = self._async_lex.commit_async()
                if commit:
                    w(commit)
            elif elems == 1:
                w(f'{dst_b}[{dst_a}] = {src_b}[{src_a}];')
            else:
                c = f'c{s.target[0].id}'
                w(f'for (int {c} = 0; {c} < {elems}; ++{c}) '
                  f'{{ {dst_b}[({dst_a}) + {c}] = {src_b}[({src_a}) + {c}]; }}')
            return

        if op == Op.WAIT:
            # released values simply alias the variable the issue declared
            tok = s.waited
            if tok is not None and tok.id in self._pending:
                for v in s.target:
                    self.bind(v, self._pending[tok.id])
            lex = self._lexic()
            if lex is None:
                return
            cls = s.attr('counter', 'copy')
            # AMD counts both classes in one vmcnt -- but only while the copy
            # path is actually the hardware one; if copies fell back to plain
            # assignments they are not in flight and must not be counted.
            unified = (self._async_lex is not None and
                       getattr(self._hw(), 'vendor', None) == 'amd')
            n = s.attr('prior_unified' if unified else 'prior', 0)
            texts = []
            if cls in ('load', 'all'):
                texts.append(lex.wait_async_regs(n))
            if cls in ('copy', 'all') and self._async_lex is not None:
                texts.append(lex.wait_async(n))
            for txt in texts:
                if txt:
                    w(txt)
            return

        if op == Op.CALL:
            callee = s.attr('callee')
            if callee is not None and callee.startswith('thread_idx_'):
                v = s.target[0]
                self.bind(v, self._thread_idx(callee[-1]))
                return
            v = s.target[0]
            args = ', '.join(self.operand(a) for a in s.args)
            self.declare(v, f'{callee}({args})', s)
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
            self.declare(v, expr, s)
            return

        raise IRError(f'no lowering for op {op!r}')

    # -- control flow ------------------------------------------------------ #

    def _emit_for(self, s: Stmt) -> None:
        w = self.writer
        lo, hi, step = s.loop_bounds
        ind = s.induction

        # iter_args and results share one C++ variable: no copy at the latch,
        # and the loop reads like an ordinary accumulator loop.
        targets: List[Optional[str]] = []
        for arg, init, res in zip(s.iter_args, s.loop_inits, s.target):
            if isinstance(arg.type, TokenType):
                # a carried token is pure bookkeeping; the hardware counter is
                # what actually crosses the back edge
                targets.append(None)
                continue
            nm = self.name(arg)
            self.bind(res, nm)
            w(f'{self.ctype(arg.type)} {nm} = {self.operand(init)};')
            targets.append(nm)

        i = self.name(ind)
        cmp_ = '<' if not isinstance(step, int) or step > 0 else '>'
        if step == 1:
            advance = f'++{i}'
        elif step == -1:
            advance = f'--{i}'
        else:
            advance = f'{i} += {self.operand(step)}'
        head = (f'{self.ctype(ind.type)} {i} = {self.operand(lo)}; '
                f'{i} {cmp_} {self.operand(hi)}; {advance}')
        # unroll goes through Writer.For, which folds the pragma into the block
        # head; a separate statement would flush the enclosing speculation and
        # defeat empty-block elision.
        with w.For(head, unroll=bool(s.attr('unroll'))):
            self._emit_body(s.regions[0].body, tuple(targets))

    def _emit_if(self, s: Stmt) -> None:
        w = self.writer
        targets: List[Optional[str]] = []
        for res in s.target:
            if isinstance(res.type, TokenType):
                targets.append(None)
                continue
            nm = self.name(res)
            w(f'{self.ctype(res.type)} {nm};')
            targets.append(nm)

        with w.If(_unwrap(self.operand(s.cond))):
            self._emit_body(s.regions[0].body, tuple(targets))
        if len(s.regions) > 1:
            with w.Block('else'):
                self._emit_body(s.regions[1].body, tuple(targets))


def emit(body: Tuple[Stmt, ...], writer, context: Any = None) -> None:
    """Lower ``body`` into ``writer``."""
    Emitter(writer, context).run(body)
