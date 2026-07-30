# SPDX-FileCopyrightText: 2015 SeisSol Group
#
# SPDX-License-Identifier: MIT
# SPDX-FileContributor: David Schneller

"""Pseudo-IR: the builder.

``IRBuilder`` is deliberately call-compatible with ``backend.writer.Writer``
for the subset that instruction code actually uses (``__call__``, ``varalloc``,
``If``, ``For``, ``Block``, ``Scope``, ``Assignment``, ``VariableDeclaration``,
``Comment``, ``Pragma``).  Legacy call sites keep working unchanged and simply
produce opaque ``raw*`` nodes; new code uses the structured constructors
(``for_``, ``if_``, ``load``, ``store``, ``alloc``, ``op``) and gets real
optimisation.  Migration progress is measurable: count the ``raw*`` nodes.

The builder knows nothing about C++ syntax.  Rendering lives in ``pir.emit``.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from tensorforge.common.basic_types import Datatype

from .core import (BOOL, INDEX, TOKEN, Access, BufferType, Effect, IRError,
                   MemSpace, Op, Operand, Region, ScalarType, Stmt, TokenType,
                   Value, dump)


def access_of(symbol: Any, kind: Effect) -> Access:
    """Build an :class:`Access` from a ``backend.symbol.Symbol``."""
    space = MemSpace.from_symbol_type(getattr(symbol, 'stype', None))
    return Access(kind=kind, space=space, base=symbol)


class _Scope:
    """One entry of the builder's block stack."""

    __slots__ = ('body', 'args', 'kind')

    def __init__(self, args: Tuple[Value, ...] = (), kind: str = 'root'):
        self.body: List[Stmt] = []
        self.args = args
        self.kind = kind


class IRBuilder:
    def __init__(self, fptype: Datatype = Datatype.F32, context: Any = None,
                 alloc: Any = None):
        # -1 so the first value is v0, matching writer.VarAlloc: a mechanism
        # swap should not show up as a diff in generated source.
        #
        # `alloc` is a shared name allocator (a `writer.VarAlloc`).  Value names
        # have to be unique across the whole generated file, not just within
        # one instruction body -- two bodies emitting into the same C++ scope
        # would otherwise both start at v0.  That uniqueness is file-scoped
        # state, and it lives on the Writer.
        self._alloc = alloc
        self._counter = -1
        self._stack: List[_Scope] = [_Scope(kind='root')]
        self._fptype = fptype
        self.context = context
        # token id -> the accesses its copy performs, so that `wait` can carry
        # the same ones without the caller having to repeat them.
        self._token_accesses: Dict[int, Tuple[Access, ...]] = {}
        # token id -> types of the values its `wait` releases (empty for a
        # copy, one entry per loaded value for `load.async`)
        self._token_results: Dict[int, Tuple[Any, ...]] = {}
        self._token_uniform: Dict[int, bool] = {}

    # -- values ------------------------------------------------------------ #

    def value(self, type_, hint: str = '', uniform: bool = True) -> Value:
        if self._alloc is not None:
            ident = self._alloc.next_index()
        else:
            self._counter += 1
            ident = self._counter
        return Value(id=ident, type=type_, uniform=uniform, hint=hint)

    def varalloc(self, prefix: str = 'v') -> Value:
        """Drop-in for ``Writer.varalloc``.

        Returns a ``Value`` whose ``__str__`` is a valid C++ identifier, so
        existing f-strings that interpolate the result keep working.  The type
        defaults to the kernel's floating point type rather than a hard-coded
        F32 --- SeisSol builds both fp32 and fp64.
        """
        return self.value(ScalarType(self._fptype),
                          hint='' if prefix == 'v' else prefix)

    def index(self, hint: str = '', uniform: bool = True) -> Value:
        return self.value(INDEX, hint=hint, uniform=uniform)

    # -- emission core ----------------------------------------------------- #

    def emit(self, stmt: Stmt) -> Stmt:
        self._stack[-1].body.append(stmt)
        return stmt

    def _emit_op(self, op: str, results, args=(), **kw) -> Stmt:
        args = tuple(args)
        results = tuple(results)
        return self.emit(Stmt(op=op, target=results, args=args, **kw))

    # -- structured constructors ------------------------------------------- #

    def const(self, value, type_=None) -> Value:
        type_ = type_ or ScalarType(self._fptype)
        v = self.value(type_, hint='c')
        self._emit_op(Op.CONST, (v,), (), attrs=(('value', value),))
        return v

    def op(self, name: str, type_, *args: Operand,
           hint: str = '', pure: bool = True, escapes: bool = False) -> Value:
        """A generic pure operation (``add``, ``mul``, ``fma``, ``select``...).

        Uniformity is propagated: the result is uniform iff every value operand
        is.  That is what lets the verifier reject a barrier under a
        thread-divergent guard.
        """
        uniform = all(a.uniform for a in args if isinstance(a, Value))
        v = self.value(type_, hint=hint, uniform=uniform)
        # `escapes`: the name is referenced from raw text, so the value must
        # neither be eliminated nor folded into its consumer.  Migration
        # scaffolding -- it disappears once the consumer takes a Value.
        attrs = (('escapes', True),) if escapes else ()
        self._emit_op(name, (v,), args, pure=pure, attrs=attrs)
        return v

    def call(self, callee: str, type_, *args: Operand, hint: str = '',
             pure: bool = True, effect: Effect = Effect.NONE,
             accesses: Tuple[Access, ...] = ()) -> Value:
        """A lexic primitive: ``tensorforge::broadcast<...>``, shuffles, MFMA."""
        uniform = all(a.uniform for a in args if isinstance(a, Value))
        v = self.value(type_, hint=hint, uniform=uniform)
        self._emit_op(Op.CALL, (v,), args, pure=pure, effect=effect,
                      accesses=accesses, attrs=(('callee', callee),))
        return v

    def pin(self, value: Operand) -> Operand:
        """Mark `value`'s producer as escaping.

        For values whose name is interpolated into raw text after the fact, so
        the caller does not have to know at construction time whether the
        result will escape.  Searches the innermost scope backwards first: the
        producer is almost always the statement just emitted.
        """
        if not isinstance(value, Value):
            return value
        for scope in reversed(self._stack):
            for i in range(len(scope.body) - 1, -1, -1):
                st = scope.body[i]
                if st.target and st.target[0] is value:
                    scope.body[i] = st.with_attr('escapes', True)
                    return value
        return value

    def thread_id(self, axis: str = 'x') -> Value:
        """The one intrinsically non-uniform value."""
        v = self.value(INDEX, hint=f'tid{axis}', uniform=False)
        self._emit_op(Op.CALL, (v,), (), attrs=(('callee', f'thread_idx_{axis}'),))
        return v

    def alloc(self, elem: Datatype, shape: Sequence[int], space: MemSpace,
              hint: str = 'buf') -> Value:
        """Request a buffer *symbolically*.

        No offset, no address: the whole-kernel allocator upstairs assigns
        those.  This is the bridge between the two layers --- the micro-IR sees
        the tile as a value and can reason about aliasing on it; the macro-IR
        owns the resource and can still decide to place it in registers.
        """
        v = self.value(BufferType(elem, tuple(shape), space), hint=hint)
        self._emit_op(Op.ALLOC, (v,), (), pure=False, movable=False)
        return v

    def load(self, base: Any, *indices: Operand, type_=None, hint: str = '',
             space: Optional[MemSpace] = None, uniform: Optional[bool] = None,
             predicate: Optional[Value] = None,
             other: Optional[Operand] = None) -> Value:
        if space is None:
            space = (base.type.space if isinstance(base, Value)
                     and isinstance(base.type, BufferType)
                     else MemSpace.from_symbol_type(getattr(base, 'stype', None)))
        if type_ is None:
            type_ = (ScalarType(base.type.elem)
                     if isinstance(base, Value) and isinstance(base.type, BufferType)
                     else ScalarType(self._fptype))
        if uniform is None:
            uniform = all(i.uniform for i in indices if isinstance(i, Value))
        v = self.value(type_, hint=hint or 'ld', uniform=uniform)
        attrs = (('other', other),) if other is not None else ()
        self._emit_op(Op.LOAD, (v,), (base,) + tuple(indices),
                      predicate=predicate, pure=False, movable=True,
                      effect=Effect.READ,
                      accesses=(Access(Effect.READ, space, base),),
                      attrs=attrs)
        return v

    def store(self, base: Any, value: Operand, *indices: Operand,
              space: Optional[MemSpace] = None) -> Stmt:
        if space is None:
            space = (base.type.space if isinstance(base, Value)
                     and isinstance(base.type, BufferType)
                     else MemSpace.from_symbol_type(getattr(base, 'stype', None)))
        return self._emit_op(Op.STORE, (), (base, value) + tuple(indices),
                             pure=False, movable=True, effect=Effect.WRITE,
                             accesses=(Access(Effect.WRITE, space, base),))

    def copy_async(self, dst: Any, src: Any, *,
                   dst_index: Sequence[Operand] = (),
                   src_index: Sequence[Operand] = (),
                   elems: int = 1,
                   dst_space: Optional[MemSpace] = None,
                   src_space: Optional[MemSpace] = None,
                   hint: str = 'cp') -> Value:
        """Issue an asynchronous copy ``src -> dst``; returns its token.

        The write to ``dst`` is not observable until the matching :meth:`wait`.
        Copy and wait carry the *same* accesses, which is what keeps any read
        of ``dst`` from being hoisted above the wait --- the reorder machinery
        needs no notion of asynchrony beyond that.
        """
        dst_space = dst_space if dst_space is not None else self._space_of(dst)
        src_space = src_space if src_space is not None else self._space_of(src)
        accesses = (Access(Effect.READ, src_space, src),
                    Access(Effect.WRITE, dst_space, dst))

        tok = self.value(TOKEN, hint=hint)
        self._emit_op(Op.COPY_ASYNC, (tok,),
                      (dst, src) + tuple(dst_index) + tuple(src_index),
                      pure=False, movable=True,
                      effect=Effect.READ | Effect.WRITE | Effect.ASYNC,
                      accesses=accesses,
                      attrs=(('ndst', len(tuple(dst_index))), ('elems', elems),
                             ('counter', 'copy')))
        self._token_accesses[tok.id] = accesses
        self._token_results[tok.id] = ()
        return tok

    def load_async(self, base: Any, *indices: Operand, type_=None,
                   hint: str = 'ld', space: Optional[MemSpace] = None,
                   predicate: Optional[Value] = None,
                   other: Optional[Operand] = None,
                   uniform: Optional[bool] = None) -> Value:
        """Issue a global -> register load; returns its token.

        The loaded value does not exist until :meth:`wait` releases it, so the
        use-after-wait ordering is plain SSA and needs no extra rule.  On AMD
        this is an ordinary `global_load` whose `s_waitcnt` we place ourselves;
        on NVIDIA the hardware scoreboard does it, and the wait lowers to
        nothing --- the op is still worth having there because it expresses how
        far the load may be hoisted.
        """
        if space is None:
            space = self._space_of(base)
        if type_ is None:
            type_ = (ScalarType(base.type.elem)
                     if isinstance(base, Value) and isinstance(base.type, BufferType)
                     else ScalarType(self._fptype))
        if uniform is None:
            uniform = all(i.uniform for i in indices if isinstance(i, Value))

        accesses = (Access(Effect.READ, space, base),)
        tok = self.value(TOKEN, hint=hint)
        attrs: Tuple[Tuple[str, Any], ...] = (
            ('counter', 'load'), ('types', (type_,)), ('uniform', uniform),
            ('hint', hint))
        if other is not None:
            attrs += (('other', other),)
        self._emit_op(Op.LOAD_ASYNC, (tok,), (base,) + tuple(indices),
                      predicate=predicate, pure=False, movable=True,
                      effect=Effect.READ | Effect.ASYNC,
                      accesses=accesses, attrs=attrs)
        self._token_accesses[tok.id] = accesses
        self._token_results[tok.id] = (type_,)
        self._token_uniform[tok.id] = uniform
        return tok

    def wait(self, token: Optional[Value] = None) -> Stmt:
        """Wait for ``token`` (or, with ``None``, drain every outstanding copy).

        The concrete counter value (``vmcnt(N)`` / ``wait_prior(N)``) is not
        decided here --- ``schedule_async`` derives it from the outstanding set
        once the schedule is final.
        """
        if token is None:
            accesses = (Access(Effect.READ | Effect.WRITE, MemSpace.UNKNOWN, None),)
        else:
            accesses = self._token_accesses.get(
                token.id,
                (Access(Effect.READ | Effect.WRITE, MemSpace.UNKNOWN, None),))
        results: Tuple[Value, ...] = ()
        if token is not None:
            types = self._token_results.get(token.id, ())
            results = tuple(self.value(t, hint=token.hint or 'ld',
                                       uniform=self._token_uniform.get(token.id, True))
                            for t in types)
        stmt = self._emit_op(Op.WAIT, results,
                             () if token is None else (token,),
                             pure=False, movable=True,
                             effect=Effect.READ | Effect.WRITE | Effect.ASYNC,
                             accesses=accesses)
        if len(results) == 1:
            return results[0]
        if results:
            return results
        return stmt

    def _space_of(self, base: Any) -> MemSpace:
        if isinstance(base, Value) and isinstance(base.type, BufferType):
            return base.type.space
        return MemSpace.from_symbol_type(getattr(base, 'stype', None))

    def barrier(self, scope: str = 'block') -> Stmt:
        return self._emit_op(Op.BARRIER, (), (), pure=False, movable=False,
                             effect=Effect.BARRIER, attrs=(('scope', scope),))

    def yield_(self, *values: Operand) -> Stmt:
        return self._emit_op(Op.YIELD, (), values, pure=False, movable=False)

    # -- scopes ------------------------------------------------------------ #

    def push(self, args: Tuple[Value, ...] = (), kind: str = 'region') -> None:
        self._stack.append(_Scope(args, kind))

    def pop(self) -> Region:
        scope = self._stack.pop()
        if not self._stack:
            raise IRError('popped the root scope')
        return Region(args=scope.args, body=tuple(scope.body))

    # -- structured control flow ------------------------------------------- #

    def for_(self, lo: Operand, hi: Operand, step: Operand = 1,
             inits: Sequence[Operand] = (), types: Sequence[Any] = (),
             unroll: bool = False, hint: str = 'i') -> '_ForHandle':
        return _ForHandle(self, lo, hi, step, tuple(inits), tuple(types),
                          unroll, hint)

    def if_(self, cond: Operand) -> '_IfHandle':
        """Guard without results --- the common case (bounds checks)."""
        return _IfHandle(self, cond, ())

    def if_else(self, cond: Operand, types: Sequence[Any]) -> '_IfHandle':
        return _IfHandle(self, cond, tuple(types))

    # -- speculative emission (replaces the throw-away Writer hack) -------- #

    @contextmanager
    def speculative(self):
        """Try something out; keep it only if it worked.

        Replaces the ``op.symbol.load(Writer(), ...)`` probe in
        ``multilinear.py``, where a whole load had to be *emitted* into a
        scratch Writer just to find out whether it would succeed::

            with builder.speculative() as spec:
                ok = all(op.symbol.load(builder, ...) for op in self._ops)
                if not ok:
                    spec.discard()
        """
        scope = self._stack[-1]
        mark = len(scope.body)
        counter = self._counter
        names = getattr(self._alloc, 'counter', None)
        depth = len(self._stack)
        spec = _Speculation()
        try:
            yield spec
        except Exception:
            self._rollback(scope, mark, counter, names, depth)
            raise
        if spec.discarded:
            self._rollback(scope, mark, counter, names, depth)
        elif len(self._stack) != depth:
            raise IRError('speculative block left the scope stack unbalanced')

    def _rollback(self, scope, mark, counter, names, depth):
        del scope.body[mark:]
        self._counter = counter
        if names is not None:
            # a discarded probe must not burn names either
            self._alloc.counter = names
        del self._stack[depth:]

    # -- legacy Writer facade ---------------------------------------------- #

    def __call__(self, code: str) -> Stmt:
        """Raw statement text.  Opaque, therefore impure, pinned and
        conservatively assumed to touch everything."""
        return self._emit_op(Op.RAWSTMT, (), (), pure=False, movable=False,
                             effect=Effect.UNKNOWN, text=code,
                             accesses=(Access(Effect.READ | Effect.WRITE,
                                              MemSpace.UNKNOWN, None),))

    def rawexpr(self, text: str, *args: Operand, type_=None, hint: str = '',
                pure: bool = False, movable: bool = False) -> Value:
        """One escape hatch with a *single* convention: ``{0}`` is ``args[0]``.

        The result is declared by the emitter; the text is an expression, never
        a full statement.  (The old writer mixed both conventions --- ``{0}``
        meant the target in ``write()`` but the loop variable in ``For``.)
        """
        type_ = type_ or ScalarType(self._fptype)
        uniform = all(a.uniform for a in args if isinstance(a, Value))
        v = self.value(type_, hint=hint, uniform=uniform)
        self._emit_op(Op.RAWEXPR, (v,), args, pure=pure, movable=movable,
                      effect=Effect.NONE if pure else Effect.UNKNOWN, text=text)
        return v

    def tempvar(self, prefix: str = 'tmp') -> Value:
        """What ``primitives/{nvidia,amd}.py`` already call.

        ``Writer`` never defined it, so every DPP / shuffle path that reaches
        ``writer.tempvar()`` raises ``AttributeError`` today.
        """
        return self.varalloc(prefix)

    def new_line(self) -> Stmt:
        # Writer.new_line() is __call__(''), which writes no line but *does*
        # flush the pending block head -- keep the side effect.
        return self._emit_op(Op.RAWSTMT, (), (), pure=False, movable=False,
                             effect=Effect.NONE, text='')

    def Emptyline(self) -> Stmt:
        return self._emit_op(Op.RAWSTMT, (), (), pure=False, movable=False,
                             effect=Effect.NONE, text='',
                             attrs=(('bare_newline', True),))

    def access_stmt(self, text: str, base: Any, kind: Effect,
                    args: Sequence[Operand] = (), movable: bool = True,
                    fmt: bool = False) -> Stmt:
        """Raw statement text whose memory effect is *known*.

        The migration end state for a memory access is not that the text
        disappears -- vendor intrinsics and inline assembly will always want
        text -- but that it stops being opaque.  `Effect.UNKNOWN` conflicts
        with everything and makes the alias model a no-op; a declared
        `Access(kind, space, base)` lets two accesses to different symbols be
        seen as independent.

        `args` carries the value operands the text mentions (the address, in
        practice) so the dependency is visible and a pass cannot lift the
        statement above the computation it reads.
        """
        space = self._space_of(base)
        # `fmt`: the text carries `{0}`.. placeholders that the emitter fills
        # in.  Baking a value's *name* into the text at build time would defeat
        # the emitter's decision to inline it -- the name would be gone.
        return self._emit_op(Op.RAWSTMT, (), tuple(args),
                             pure=False, movable=movable, effect=kind,
                             accesses=(Access(kind, space, base),), text=text,
                             attrs=(('fmt', True),) if fmt else ())

    def load_expr(self, text: str, type_, base: Any, *,
                  kind: Effect = Effect.READ, space: Optional[MemSpace] = None,
                  args: Sequence[Operand] = (), hint: str = 'ld') -> Value:
        """A declaration whose right-hand side is still text, but whose result
        is a real SSA value.

        The bridge for migrating a body from the inside out: the access itself
        may stay a vendor-specific string, while everything that consumes it
        becomes structured.
        """
        v = self.value(type_, hint=hint)
        self._emit_op(Op.RAWEXPR, (v,), tuple(args), pure=False, movable=True,
                      effect=kind,
                      accesses=(Access(kind,
                                       self._space_of(base) if space is None
                                       else space, base),),
                      text=text)
        return v

    def Comment(self, text: str) -> Stmt:
        return self._emit_op(Op.RAWSTMT, (), (), pure=False, movable=True,
                             effect=Effect.NONE, text=f'// {text}')

    def Pragma(self, name: str) -> Stmt:
        return self.__call__(f'#pragma {name}')

    def Assignment(self, left, right) -> Stmt:
        return self.__call__(f'{left} = {right};')

    def Accumulate(self, left, right) -> Stmt:
        return self.__call__(f'{left} += {right};')

    def Expression(self, expression) -> Stmt:
        return self.__call__(f'{expression};')

    def VariableDeclaration(self, type_, name, expression=None) -> Stmt:
        if expression is not None:
            return self.__call__(f'{type_} {name} = {expression};')
        return self.__call__(f'{type_} {name};')

    def Block(self, text: str = '') -> '_RawBlock':
        return _RawBlock(self, text)

    def Scope(self) -> '_RawBlock':
        return _RawBlock(self, '')

    def AnonymousScope(self) -> '_RawBlock':
        return _RawBlock(self, '')

    def If(self, expression) -> '_RawBlock':
        return _RawBlock(self, f'if ({expression})')

    def For(self, argument, unroll: bool = False) -> '_RawBlock':
        if unroll:
            return _RawBlock(self, f'#pragma unroll\nfor ({argument})')
        return _RawBlock(self, f'for ({argument})')

    def While(self, argument) -> '_RawBlock':
        return _RawBlock(self, f'while ({argument})')

    # -- result ------------------------------------------------------------ #

    def finish(self) -> Tuple[Stmt, ...]:
        if len(self._stack) != 1:
            raise IRError(f'{len(self._stack) - 1} scope(s) left open')
        return tuple(self._stack[0].body)

    def dump(self) -> str:
        return dump(tuple(self._stack[0].body))


class _Speculation:
    def __init__(self):
        self.discarded = False

    def discard(self):
        self.discarded = True


class _RawBlock:
    """Legacy ``Writer.Block`` equivalent: an opaque head plus a region."""

    def __init__(self, builder: IRBuilder, text: str, pragma: Optional[str] = None):
        self.builder = builder
        self.text = text
        self.pragma = pragma

    def __enter__(self):
        self.builder.push(kind='rawblock')
        return self

    def __exit__(self, exc_type, exc, tb):
        region = self.builder.pop()
        if exc_type is not None:
            return False
        attrs = (('pragma', self.pragma),) if self.pragma else ()
        self.builder.emit(Stmt(op=Op.RAWBLOCK, regions=(region,), text=self.text,
                               pure=False, movable=False, effect=Effect.NONE,
                               attrs=attrs))
        return False

    def __call__(self, line: str):
        return self.builder(line)


class _ForHandle:
    def __init__(self, builder, lo, hi, step, inits, types, unroll, hint):
        if len(inits) != len(types):
            raise IRError('for_: one result type per init value required')
        self.builder = builder
        self._args = (lo, hi, step) + inits
        self._types = types
        self._unroll = unroll
        self.induction = builder.index(hint=hint)
        self.iter_args = tuple(builder.value(t, hint=f'acc{i}')
                               for i, t in enumerate(types))
        self.results: Tuple[Value, ...] = ()
        # a token carried through the loop keeps the accesses of its copy
        for arg, init in zip(self.iter_args, inits):
            if isinstance(arg.type, TokenType) and isinstance(init, Value):
                acc = builder._token_accesses.get(init.id)
                if acc is not None:
                    builder._token_accesses[arg.id] = acc
                builder._token_results[arg.id] = builder._token_results.get(init.id, ())
                builder._token_uniform[arg.id] = builder._token_uniform.get(init.id, True)

    def __enter__(self) -> '_ForHandle':
        self.builder.push((self.induction,) + self.iter_args, kind='for')
        return self

    def __exit__(self, exc_type, exc, tb):
        region = self.builder.pop()
        if exc_type is not None:
            return False
        self.results = tuple(self.builder.value(t, hint=f'res{i}')
                             for i, t in enumerate(self._types))
        for res, y in zip(self.results, region.yielded):
            if isinstance(res.type, TokenType) and isinstance(y, Value):
                acc = self.builder._token_accesses.get(y.id)
                if acc is not None:
                    self.builder._token_accesses[res.id] = acc
                b = self.builder
                b._token_results[res.id] = b._token_results.get(y.id, ())
                b._token_uniform[res.id] = b._token_uniform.get(y.id, True)
        attrs = (('unroll', True),) if self._unroll else ()
        self.builder.emit(Stmt(op=Op.FOR, target=self.results, args=self._args,
                               regions=(region,), pure=False, movable=False,
                               attrs=attrs))
        return False

    def yield_(self, *values: Operand):
        return self.builder.yield_(*values)

    @property
    def result(self) -> Value:
        return self.results[0]


class _IfHandle:
    def __init__(self, builder, cond, types):
        self.builder = builder
        self.cond = cond
        self._types = types
        self._then: Optional[Region] = None
        self._else: Optional[Region] = None
        self.results: Tuple[Value, ...] = ()

    # `with builder.if_(cond):` -- guard, no results
    def __enter__(self) -> '_IfHandle':
        if self._types:
            raise IRError('if_else: use .then()/.otherwise(), not `with`')
        self.builder.push(kind='if')
        return self

    def __exit__(self, exc_type, exc, tb):
        region = self.builder.pop()
        if exc_type is not None:
            return False
        self.builder.emit(Stmt(op=Op.IF, args=(self.cond,), regions=(region,),
                               pure=False, movable=False))
        return False

    @contextmanager
    def then(self):
        self.builder.push(kind='if')
        yield self
        self._then = self.builder.pop()

    @contextmanager
    def otherwise(self):
        self.builder.push(kind='if')
        yield self
        self._else = self.builder.pop()
        self._finish()

    def yield_(self, *values: Operand):
        return self.builder.yield_(*values)

    def _finish(self):
        self.results = tuple(self.builder.value(t, hint=f'sel{i}')
                             for i, t in enumerate(self._types))
        regions = tuple(r for r in (self._then, self._else) if r is not None)
        self.builder.emit(Stmt(op=Op.IF, target=self.results, args=(self.cond,),
                               regions=regions, pure=False, movable=False))

    @property
    def result(self) -> Value:
        return self.results[0]
