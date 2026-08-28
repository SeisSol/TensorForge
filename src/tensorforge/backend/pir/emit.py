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

from tensorforge.common.basic_types import GeneralLexicon
from tensorforge.common.operation import Operation
from .core import (Access, BufferType, Effect, IRError, MemSpace, Op, Operand,
                   Region, ScalarType, Stmt, TokenType, Value, def_use, walk)

_ATOM = __import__('re').compile(r'^(?:[A-Za-z_][A-Za-z0-9_.:]*|\d[\w.]*)$')


# A predicate becomes a select only where suppressing the statement is not the
# point.  Reads -- synchronous or asynchronous -- may be evaluated under a
# ternary; anything that writes, is atomic, synchronises or is opaque has to
# keep a real branch, or the effect would happen when it must not.  This used
# to be a test on the *shape* of the statement (does it have a target?), which
# would silently fold a value-returning atomic into a ternary.
_MUST_BRANCH = Effect.WRITE | Effect.ATOMIC | Effect.BARRIER | Effect.UNKNOWN


def _folds_predicate(s: Stmt) -> bool:
    if s.effect & _MUST_BRANCH or s.regions:
        return False
    return bool(s.target) or s.op in Op.DECLARING


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
    'rem': '%', 'bitand': '&', 'bitor': '|', 'bitxor': '^', 'shl': '<<', 'shr': '>>',
    'lt': '<', 'le': '<=', 'gt': '>', 'ge': '>=', 'eq': '==', 'ne': '!=',
    'and': '&&', 'or': '||',
}

# Ops with no infix form that the lexic already spells correctly.  Without
# this they fell through to the generic `f'{op}({args})'`, i.e. unqualified
# `min(a, b)` -- which happens to resolve in CUDA and HIP device code through
# the vendor headers' global-namespace overloads, and so worked by accident
# while silently depending on which headers a translation unit had pulled in.
# `get_operation` gives `fminf`/`fmin` by dtype, which is what the elementwise
# path has always emitted for the same operator.
_LEXIC_BINOP = {'min': Operation.MIN, 'max': Operation.MAX}


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

    def operand(self, x: Operand, type_=None) -> str:
        """A value's name, or an immediate spelled for `type_`.

        `str(x)` is fine for the index arithmetic that supplies most
        immediates, and wrong for a float one: Python prints an infinity as
        `inf`, which is not C++. That only became reachable once a loop
        carried an operator's neutral element -- `MaxOperator`'s is
        `-math.inf` -- as its initial value, since `Op.CONST` spells its
        value through `Datatype.literal` and a loop init did not.
        """
        if isinstance(x, Value):
            if x.id in self._consts:
                return self._consts[x.id]
            return self.name(x)
        if isinstance(x, bool):
            return 'true' if x else 'false'
        if isinstance(x, (int, float)) and isinstance(type_, ScalarType):
            return type_.base.literal(x)
        if x is None:
            # `str(None)` is `None`, which is a perfectly good C++ identifier
            # and a perfectly bad one to emit.  It arrives when a producer
            # answered with nothing and the consumer used the answer anyway --
            # `Symbol.load` returns None for every structured load under
            # `simd_mode`, and the value flowed into an arithmetic op, which
            # came out as `sycl::max(float(acc), float(None))`.  Loud here,
            # because the alternative is a compiler error pointing at the
            # arithmetic rather than at the load that had no value.
            raise IRError(
                'a None operand reached the emitter; some producer returned '
                'no value and its consumer used the result anyway')
        return str(x)

    # -- types ------------------------------------------------------------- #

    def ctype(self, t, value: Optional[Value] = None) -> str:
        """The C++ spelling of a value's type.

        ``value`` is the value being declared, when there is one.  This
        emitter does not need it -- in SPMD the lane is in the *address*, so a
        value's type says nothing about how it is spread across the wave.  An
        explicitly vectorised emitter needs exactly that, and it needs it at
        every declaration; passing the value here rather than threading a
        second parameter through six call sites is what keeps the two
        emitters one class apart instead of one file apart.
        """
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

    def _lexic_binop(self, op: str, v: Value, args: Sequence[str]) -> str:
        """`min`/`max` as the lexic spells them, by result dtype.

        Falls back to the bare call when there is no lexic -- the IR-level
        tests build bodies without a context, and an emitter that raised there
        would make them require one for an op that is not what they test.
        """
        lex = self._lexic()
        if lex is None:
            return f'{op}({", ".join(args)})'
        base = getattr(v.type, 'base', None)
        return lex.get_operation(_LEXIC_BINOP[op], base, args[0], args[1])

    def _sync(self, scope=None) -> str:
        # The scope used to be an unchecked string that never reached here, so
        # every barrier came out as sync_block() regardless of what was asked
        # for.
        lex = self._lexic()
        if lex is None:
            return '__syncthreads();'
        name = getattr(scope, 'name', 'BLOCK')
        if name == 'MULT':
            return lex.sync_simd()
        if name == 'GRID':
            return lex.sync_grid()
        return lex.sync_block()

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
        if s.predicate is not None and _folds_predicate(s):
            other = s.attr('other')
            other = (self.operand(other) if other is not None
                     else self.zero(v.type))
            expr = f'{self.operand(s.predicate)} ? ({expr}) : ({other})'
        if name is None and v.id in self._inline:
            self.bind(v, _atomic(expr) and expr or f'({expr})')
            return
        self.writer(f'{self.ctype(v.type, v)} {name or self.name(v)} = {expr};')

    def base_name(self, base: Operand) -> str:
        if isinstance(base, Value):
            return self.name(base)
        return getattr(base, 'name', str(base))

    def elem_access(self, base: Operand, addr: str, t) -> str:
        """``base[addr]``, reinterpreted when the value is wider than one element.

        A buffer is typed by its element, so a vector-typed access reads or
        writes several of them at once and has to be spelled through a pointer
        of the wider type.  That cast is not new -- ``load_linear`` and
        ``store_linear`` formatted the same one into a raw string.  Putting it
        *here* is what lets a vectorised access stay an ``Op.LOAD``/``Op.STORE``
        with the buffer as an operand: the string form had to leave the
        structured path (``pir_buffer`` was consulted only for ``vec == 1``),
        which cost every pass its view of which buffer the access touches.

        The cast is only defined when ``addr`` is aligned to the wider type.
        Nothing here checks that, exactly as nothing checked it before; the
        legality belongs with whoever chooses the width, not with the spelling.
        """
        access = f'{self.base_name(base)}[{addr}]'
        if isinstance(t, ScalarType) and t.length is not None:
            return f'*({self.ctype(t)}*)&{access}'
        return access

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
                        and s.op != Op.CONST and not s.attr('escapes')
                        and not s.attr('no_inline')):
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
            if (s.predicate is not None and s.op != Op.YIELD
                    and not _folds_predicate(s)):
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
            decl = s.attr('decl')
            if decl is not None:
                # The declarator is the caller's text, not something rendered
                # from `v.type`: a pointer binding reads
                # `const float *const __restrict__ p`, and on the AMD
                # pointer-based path `auto p` with the type inside a cast on
                # the right.  The value still exists, so consumers address the
                # buffer rather than spelling its name.
                extern = s.attr('extern')
                if extern is not None:
                    self.bind(v, extern)
                w(f'{decl} = '
                  f'{s.text.format(*[self.operand(a) for a in s.args])};')
                return
            text = s.text.format(*[self.operand(a) for a in s.args])
            self.declare(v, text, s)
            return

        if op == Op.RAWBLOCK:
            if s.attr('pragma'):
                w(f'#pragma {s.attr("pragma")}')
            for t in s.target:      # a value-producing block declares it first
                w(f'{self.ctype(t.type, t)} {self.name(t)};')
            with w.Block(s.text):
                self._emit_body(s.regions[0].body, yield_to)
            return

        if op == Op.BARRIER:
            sync_instr = self._sync(s.attr('scope'))
            if sync_instr is not None:
                w(sync_instr)
            return

        if op == Op.ALLOC:
            v = s.target[0]
            t = v.type
            arena = s.attr('arena')
            if arena is not None:
                # A shared buffer is a window into the kernel's one arena, at
                # the offset the builder bumped out of this instruction's
                # declared scratch tail.  Declaring `__shared__` here instead
                # would allocate outside the size ShrMemOpt computed, which is
                # what the occupancy calculation and the barrier placement both
                # read.
                off = s.attr('offset', 0)
                extern = s.attr('extern')
                if extern is not None:
                    self.bind(v, extern)
                qual = s.attr('restrict')
                qual = f'{qual} ' if qual else ''
                w(f'{t.elem.ctype()}* {qual}{self.name(v)} = &{arena}[{off}];')
                return
            qual = {MemSpace.CONSTANT: 'const '}.get(t.space, '')
            extern = s.attr('extern')
            if extern is not None:
                self.bind(v, extern)
            align = s.attr('align')
            spec = f'alignas({align}) ' if align else ''
            w(f'{spec}{qual}{t.elem.ctype()} {self.name(v)}[{t.volume}]'
              f'{s.attr("init", "")};')
            return

        if op == Op.LOAD:
            lex = self._lexic()

            v = s.target[0]
            addr = self.address(s.args[0], s.args[1:])
            nontemporal = s.attr('nontemporal')
            access = self.elem_access(s.args[0], addr, v.type)
            if nontemporal:
                self.declare(v, f'{lex.glb_load(access, True)}', s)
            else:
                self.declare(v, access, s)
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
            val = s.args[1]
            vt = val.type if isinstance(val, Value) else None
            access = self.elem_access(s.args[0], addr, vt)
            # A store to global memory goes through the lexic, the same way
            # `Op.LOAD` above goes through `glb_load`: the nontemporal hint is
            # `__stcg` on NVIDIA and `__builtin_nontemporal_store` on AMD, and
            # neither is expressible as an assignment.  Without this the hint
            # would be silently dropped for every store that reaches here --
            # which is why global stores had to stay on the text path.
            space = s.accesses[0].space if s.accesses else None
            lex = self._lexic()
            if space is MemSpace.GLOBAL and lex is not None:
                w(lex.glb_store(access, self.operand(val),
                                bool(s.attr('nontemporal'))))
                return
            w(f'{access} = {self.operand(val)};')
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
            if callee is not None and callee.startswith('batch_id_'):
                # Bound, not declared: the macro layer already emits
                # `batchId{n}` around this body, so the value is a name that
                # exists rather than a call to make.  This is the seam -- the
                # micro IR reasons about the batch id as a MULT-uniform value,
                # the macro IR owns what it is called and how it is computed.
                v = s.target[0]
                self.bind(v, f'{GeneralLexicon.BATCH_ID_NAME}{callee[len("batch_id_"):]}')
                return
            args = ', '.join(self.operand(a) for a in s.args)
            if not s.target:
                # A void intrinsic: invoked for what it does to a register it
                # takes by reference, so there is nothing to declare.  The
                # operands still went through `operand()`, which is the point
                # -- the arguments are values the IR knows, not baked-in names.
                w(f'{callee}({args});')
                return
            v = s.target[0]
            self.declare(v, f'{callee}({args})', s)
            return

        if op == Op.DECLARE:
            v = s.target[0]
            # No initialiser to inline, so `declare()`'s folding machinery does
            # not apply -- this is the plain declaration the raw text used to
            # emit, byte for byte.
            w(f'{self.ctype(v.type, v)} {self.name(v)}{s.attr("init", "{}")};')
            return

        if op == Op.PACK:
            v = s.target[0]
            parts = ', '.join(self.operand(a) for a in s.args)
            self.declare(v, f'{{{parts}}}', s)
            return

        if op == Op.EXTRACT:
            v = s.target[0]
            self.declare(v, f'{self.operand(s.args[0])}[{s.attr("lane")}]', s)
            return

        if op == Op.ACCUM:
            target, value = s.args
            w(f'{self.operand(target)} += {self.operand(value)};')
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
            elif op in _LEXIC_BINOP and len(args) == 2:
                expr = self._lexic_binop(op, v, args)
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
            w(f'{self.ctype(arg.type, arg)} {nm} = {self.operand(init, arg.type)};')
            targets.append(nm)

        i = self.name(ind)
        cmp_ = '<' if not isinstance(step, int) or step > 0 else '>'
        if step == 1:
            advance = f'++{i}'
        elif step == -1:
            advance = f'--{i}'
        else:
            advance = f'{i} += {self.operand(step)}'
        head = (f'{self.ctype(ind.type, ind)} {i} = {self.operand(lo)}; '
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
            w(f'{self.ctype(res.type, res)} {nm};')
            targets.append(nm)

        with w.If(_unwrap(self.operand(s.cond))):
            self._emit_body(s.regions[0].body, tuple(targets))
        if len(s.regions) > 1:
            with w.Block('else'):
                self._emit_body(s.regions[1].body, tuple(targets))


def emit(body: Tuple[Stmt, ...], writer, context: Any = None) -> None:
    """Lower ``body`` into ``writer``, in whichever model the lexic asks for.

    The choice is the lexic's because the lexic is where the rest of the
    model already lives -- the kernel attributes, the broadcast spelling, the
    wave barrier.  Splitting the decision between here and there is how the
    old arrangement ended up with an ESIMD kernel attribute on an SPMD body.
    """
    lex = getattr(context, 'get_vm', None)
    simd = False
    if lex is not None:
        try:
            simd = bool(getattr(context.get_vm().get_lexic(), 'simd_mode', False))
        except Exception:
            simd = False
    if simd:
        from .emit_esimd import EsimdEmitter
        EsimdEmitter(writer, context).run(body)
    else:
        Emitter(writer, context).run(body)
