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

import re
from contextlib import contextmanager
from dataclasses import replace
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from tensorforge.common.basic_types import Datatype
from tensorforge.common.exceptions import GenerationError

from .core import (BOOL, INDEX, TOKEN, Access, BufferType, Effect, IRError,
                   LaneAxis, MemSpace, Op, Operand, Region, RegisterLayout,
                   ScalarType, Stmt, TokenType, Value, dump, join_layout,
                   Uniformity)


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


_BARRIER_ALIASES = {'lane': Uniformity.LANE, 'simd': Uniformity.MULT,
                    'wave': Uniformity.MULT, 'warp': Uniformity.MULT,
                    'mult': Uniformity.MULT, 'block': Uniformity.BLOCK,
                    'group': Uniformity.BLOCK, 'grid': Uniformity.GRID}


def _as_barrier_scope(x) -> Uniformity:
    """Accepts the old strings so existing callers keep working."""
    if isinstance(x, Uniformity):
        return x
    try:
        return _BARRIER_ALIASES[str(x).lower()]
    except KeyError:
        raise IRError(f'unknown barrier scope {x!r}; '
                      f'expected one of {sorted(_BARRIER_ALIASES)}')


def _as_uniformity(x) -> Uniformity:
    if isinstance(x, Uniformity):
        return x
    return Uniformity.GRID if x else Uniformity.LANE


def _join(operands) -> Uniformity:
    """A result is only as uniform as its least uniform operand."""
    levels = [a.uniformity for a in operands if isinstance(a, Value)]
    return min(levels) if levels else Uniformity.GRID


#: An identifier the builder might have emitted.  Values are named `v{id}` or
#: `v{id}_{hint}`, so this over-matches on purpose and the lookup decides.
_IDENT = re.compile(r'\bv\d+\w*\b')


def _names_in(code: str, name: str) -> bool:
    """Whether `name` appears in `code` as an identifier, not as a substring.

    `v1` must not match `v13`, or the check fires on statements that have
    nothing to do with the buffer and callers learn to pass the argument that
    turns it off.
    """
    return re.search(rf'\b{re.escape(name)}\b', code) is not None


class IRBuilder:
    def __init__(self, fptype: Datatype = Datatype.F32, context: Any = None,
                 alloc: Any = None, scratch: Optional[Tuple[str, int]] = None):
        # -1 so the first value is v0, matching writer.VarAlloc: a mechanism
        # swap should not show up as a diff in generated source.
        #
        # `alloc` is a shared name allocator (a `writer.VarAlloc`).  Value names
        # have to be unique across the whole generated file, not just within
        # one instruction body -- two bodies emitting into the same C++ scope
        # would otherwise both start at v0.  That uniqueness is file-scoped
        # state, and it lives on the Writer.
        self._alloc = alloc
        # The scratch arena this body may suballocate from: (pointer name,
        # budget in elements).  `None` means the instruction declared no
        # budget, and a shared alloc from it is a defect rather than a
        # request to make room -- see `alloc`.
        self._scratch = scratch
        self._scratch_used = 0
        self._scratch_peak = 0
        # Buffers this body allocated, for `_check_declared_accesses`.
        self._shared_buffers: List[Value] = []
        # Every value this body has named, keyed by the identifier it emits
        # as.  A raw statement that narrows its accesses is checked against
        # this, so the lookup has to be by name rather than a scan over all
        # values -- there are thousands of both.
        self._by_name: Dict[str, Value] = {}
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

    def value(self, type_, hint: str = '',
              uniform: Union[bool, Uniformity] = Uniformity.GRID,
              layout: Optional[RegisterLayout] = None) -> Value:
        """``uniform`` accepts a bool for compatibility: True -> GRID,
        False -> LANE.  New code should pass a :class:`Uniformity`.

        ``layout`` is how the value is spread over the lanes, when the caller
        knows.  Left ``None`` it stays untracked, which is what every existing
        call site produces and therefore changes nothing."""
        if self._alloc is not None:
            ident = self._alloc.next_index()
        else:
            self._counter += 1
            ident = self._counter
        v = Value(id=ident, type=type_, uniformity=_as_uniformity(uniform),
                  hint=hint, layout=layout)
        self._by_name[str(v)] = v
        return v

    def varalloc(self, prefix: str = 'v') -> Value:
        """Drop-in for ``Writer.varalloc``.

        Returns a ``Value`` whose ``__str__`` is a valid C++ identifier, so
        existing f-strings that interpolate the result keep working.  The type
        defaults to the kernel's floating point type rather than a hard-coded
        F32 --- SeisSol builds both fp32 and fp64.
        """
        v = self.value(ScalarType(self._fptype),
                       hint='' if prefix == 'v' else prefix)
        # A name reservation, not a value the IR defines anywhere.  Legacy
        # bodies redeclare one of these per scope --- `float v35[4][2]{};`
        # inside each iteration --- which is ordinary C++ and not SSA at all.
        # `_check_declared_accesses` must not ask a raw statement to claim
        # such a name as a use or a definition: there is no defining statement
        # for `dce` to remove, so the argument the check rests on does not
        # apply, and demanding a claim here would only teach callers to drop
        # the `accesses` argument that turns the check on.
        self._by_name.pop(str(v), None)
        return v

    def index(self, hint: str = '',
              uniform: Union[bool, Uniformity] = Uniformity.GRID) -> Value:
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
        uniform = _join(args)
        # Same shape as the uniformity join, and for the same reason: an
        # elementwise result lives where its operands live.  Until something
        # attaches a layout this is `None` in, `None` out.
        v = self.value(type_, hint=hint, uniform=uniform,
                       layout=join_layout(args))
        # `escapes`: the name is referenced from raw text, so the value must
        # neither be eliminated nor folded into its consumer.  Migration
        # scaffolding -- it disappears once the consumer takes a Value.
        attrs = (('escapes', True),) if escapes else ()
        self._emit_op(name, (v,), args, pure=pure, attrs=attrs)
        return v

    def call(self, callee: str, type_, *args: Operand, hint: str = '',
             pure: bool = True, movable: bool = True,
             effect: Effect = Effect.NONE,
             accesses: Tuple[Access, ...] = (),
             layout: Optional[RegisterLayout] = None,
             keep_layout: bool = False, materialize: bool = False) -> Value:
        """A lexic primitive: ``tensorforge::broadcast<...>``, shuffles, MFMA.

        Unlike :meth:`op`, the result layout is *not* inherited by default.
        A broadcast or a shuffle exists precisely to change the distribution,
        so inheriting would be wrong more often than right; the caller says
        what comes out (``layout``), or says that this one does pass it
        through (``keep_layout``).

        ``movable=False`` is what a *cross-lane* primitive wants.  Such an
        instruction reads the registers of other lanes, so it is only well
        defined where the wave is converged; it stays a pure function of its
        operands -- CSE and inlining remain correct and are the point -- but
        it must not be hoisted out of the region it was placed in.

        ``materialize`` keeps the result in a variable of its own.  An
        accumulator chain -- each MFMA consuming the previous one -- is
        single-use at every link, so the inliner would fold the whole chain
        into one nested expression.  That is the same value, but it collapses
        a schedule that was written down deliberately and leaves the register
        pressure estimate with nothing to count.
        """
        uniform = _join(args)
        if layout is None and keep_layout:
            layout = join_layout(args)
        v = self.value(type_, hint=hint, uniform=uniform, layout=layout)
        attrs = (('callee', callee),)
        if materialize:
            attrs += (('no_inline', True),)
        self._emit_op(Op.CALL, (v,), args, pure=pure, movable=movable,
                      effect=effect, accesses=accesses, attrs=attrs)
        return v

    def declare(self, type_=None, *, hint: str = '', init: str = '{}',
                uniform: Union[bool, Uniformity] = Uniformity.GRID,
                layout: Optional[RegisterLayout] = None) -> Value:
        """A definition with no computed initialiser: ``Ty name{};``.

        The accumulator of a hand-written intrinsic sequence is written by
        `fmacdpp` through a reference, so it is not the result of any single
        statement and cannot be an SSA producer.  Until now that meant emitting
        the declaration as raw text, which left the value *used but never
        defined* --- invisible to the verifier, and untouchable by every pass,
        because nothing connected the name to a statement.

        This node closes that hole without changing what is emitted: the value
        has a definition point, so def-use analysis works, while the C++ text
        stays byte-for-byte what the raw statement produced.  It is
        deliberately *not* in ``Op.DECLARING``: there is no initialiser to fold
        a predicate into, so a predicated declaration is rejected rather than
        silently lowered to a select.

        `escapes` is set: the value is written through a reference elsewhere,
        so it must keep its own name and must not be folded into a consumer.
        """
        type_ = type_ or ScalarType(self._fptype)
        v = self.value(type_, hint=hint, uniform=uniform, layout=layout)
        self._emit_op(Op.DECLARE, (v,), (), pure=False, movable=False,
                      effect=Effect.NONE,
                      attrs=(('init', init), ('escapes', True)))
        return v

    def call_stmt(self, callee: str, *args: Operand,
                  writes: Sequence[Operand] = (),
                  effect: Optional[Effect] = None,
                  movable: bool = False) -> Stmt:
        """A vendor intrinsic invoked for its effect, not for a result.

        ``fmacdpp16<0>(c, a, b)`` and ``transpose4x4b32(...)`` return nothing
        and write through a reference, so they cannot be modelled as pure
        SSA producers.  They can still stop being *opaque*: the arguments go
        in as values, so the def-use edges are real, and the registers written
        go in as declared :class:`Access` es keyed on the value itself.  Two
        such calls on different accumulators then provably do not conflict,
        which ``Effect.UNKNOWN`` on a raw statement can never say.

        ``writes`` names the operands mutated in place.  Their producers are
        pinned: a value whose name is handed to a reference parameter must not
        be folded into its consumer, or the assignment would land nowhere.
        """
        for w in writes:
            self._require_addressable(w, callee)
        accesses = tuple(Access(Effect.READ | Effect.WRITE, MemSpace.REGISTER,
                                base=w) for w in writes if isinstance(w, Value))
        for w in writes:
            self.pin(w)
        if effect is None:
            effect = Effect.WRITE if accesses else Effect.NONE
        return self._emit_op(Op.CALL, (), tuple(args), pure=False,
                             movable=movable, effect=effect,
                             accesses=accesses, attrs=(('callee', callee),))

    def _require_addressable(self, value: Operand, callee: str) -> None:
        """A written argument has to be something that has an address.

        The C++ these calls reach takes its outputs by non-const reference, so
        a literal in a written position is ill-formed --- and nothing in this
        repository compiles, so it would surface as a build failure at a user
        site rather than here.  It has happened: a padded MFMA tail block used
        to hand `0.0f` to `transpose4x4b32`'s third and fourth parameters,
        which are `T &`.

        Cheap to check and worth checking eagerly rather than in `verify`,
        which only runs under `TF_IR_DEBUG`.
        """
        if not isinstance(value, Value):
            raise IRError(
                f'{callee}: {value!r} is written but is not a value; a '
                f'reference parameter needs something with an address')
        producer = self._producer(value)
        if producer is not None and producer.op == Op.CONST:
            raise IRError(
                f'{callee}: {value!r} is written but is a constant, which '
                f'renders as a literal; a reference parameter cannot bind it')

    def _producer(self, value: Value) -> Optional[Stmt]:
        for scope in reversed(self._stack):
            for st in reversed(scope.body):
                if st.target and st.target[0] is value:
                    return st
        return None

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
        """The lane index: the narrowest thing there is."""
        v = self.value(INDEX, hint=f'tid{axis}', uniform=Uniformity.LANE)
        self._emit_op(Op.CALL, (v,), (), attrs=(('callee', f'thread_idx_{axis}'),))
        return v

    def batch_id(self, lookahead: int = 0) -> Value:
        """The element this multiplication is working on.

        ``MULT``-uniform, not block-uniform: the macro layer emits
        ``batchId0 = threadIdx.y + blockDim.y * blockIdx.x``, so every thread of
        one multiplication agrees and the multiplications packed into a block do
        not.  Declaring it block-uniform would let ``licm`` hoist an address
        computed from it out of a per-multiplication scope.

        ``lookahead`` names the index the loop binds that many iterations ahead,
        which is what a peeled or advanced transfer consumes.
        """
        v = self.value(INDEX, hint=f'batch{lookahead}', uniform=Uniformity.MULT)
        self._emit_op(Op.CALL, (v,), (),
                      attrs=(('callee', f'batch_id_{lookahead}'),))
        return v

    def alloc(self, elem: Datatype, shape: Sequence[int], space: MemSpace,
              hint: str = 'buf') -> Value:
        """Request a buffer *symbolically*.

        For registers and scratch: no offset, no address --- the whole-kernel
        allocator upstairs assigns those.  This is the bridge between the two
        layers: the micro-IR sees the tile as a value and can reason about
        aliasing on it; the macro-IR owns the resource and can still decide to
        place it in registers.

        For :attr:`MemSpace.SHARED` the answer cannot be deferred that far.
        Shared memory is one arena per kernel, sized by ``ShrMemOpt`` from
        ``temp_shmem()`` *before* any instruction body is built, so by the time
        this runs the budget is already fixed and an independent
        ``__shared__`` array would sit outside it --- uncounted against the
        occupancy limit and invisible to the barrier placement that keys on
        region membership.  So a shared alloc is a suballocation of the tail
        this instruction declared, handed out here by bump.

        The declared budget stays the contract, and it is now checked.  It was
        not before: `nvidia.matmul` carries an
        ``assert 32 * max(aregs + bregs, cregs) <= shmsize`` precisely because
        the size in ``temp_shmem()`` and the hand-written offsets in the body
        are two statements of one fact, kept in agreement by hand.  Every
        caller that allocates through here gets that check for free, against
        what it actually asked for rather than against a formula restated at
        the use site.
        """
        v = self.value(BufferType(elem, tuple(shape), space), hint=hint)
        attrs: Tuple = ()
        if space == MemSpace.SHARED:
            attrs = self._suballocate(v, elem)
            self._shared_buffers.append(v)
        self._emit_op(Op.ALLOC, (v,), (), pure=False, movable=False,
                      attrs=attrs)
        return v

    @contextmanager
    def scratch_scope(self):
        """Buffers allocated inside are dead at the end of it.

        The hand form of what a liveness analysis over this body would derive,
        and it is here only because that analysis cannot yet run: a raw
        statement that does not declare its accesses conflicts with every
        buffer in every space, so a body still made mostly of raw text has an
        interference graph in which everything interferes and a colouring that
        reuses nothing.

        `nvidia.matmul` is the case.  Its A and B windows live only inside the
        k/kk/ii nest and its C window only in the epilogue after it closes, so
        C may sit on top of A and B --- 192 elements rather than 320 for
        m16n8k8.  That is a lifetime argument, stated here once by nesting
        instead of three times as offset constants plus an `assert` restating
        the total.

        Written to be replaceable rather than to last.  What it computes is a
        peak, which is exactly what a colouring computes, so when the body is
        structured enough for liveness the two are comparable and this can go.
        """
        mark = self._scratch_used
        try:
            yield
        finally:
            self._scratch_peak = max(self._scratch_peak, self._scratch_used)
            self._scratch_used = mark

    @property
    def scratch_peak(self) -> int:
        """High-water mark, which is what the budget has to cover.

        Not `_scratch_used`: that falls back at the end of every scope, so on
        a body that uses scopes it under-reports, and a check against it would
        pass a body that overflows.
        """
        return max(self._scratch_peak, self._scratch_used)

    def _suballocate(self, v: Value, elem: Datatype) -> Tuple:
        """Place a shared buffer in this instruction's scratch tail."""
        if self._scratch is None:
            raise GenerationError(
                f'shared alloc of {v.type} with no scratch budget: the '
                f'instruction building this body returns 0 from temp_shmem(), '
                f'so ShrMemOpt reserved nothing for it to sit in')
        name, budget = self._scratch
        # 16 bytes is what the vectorised paths need -- `nvidia.matmul` stores
        # through `float4` -- and matches the alignment ShrMemOpt already pads
        # the arena to.  Aligning every suballocation keeps that property
        # independent of the order they are requested in.
        align = max(1, 16 // elem.size())
        start = ((self._scratch_used + align - 1) // align) * align
        end = start + v.type.volume
        if max(end, self._scratch_peak) > budget:
            raise GenerationError(
                f'scratch overflow: {v.type} at offset {start} needs '
                f'{end} elements, budget is {budget}. Either temp_shmem() '
                f'under-reports what this instruction allocates, or the body '
                f'allocates more than it declared')
        self._scratch_used = end
        return (('arena', name), ('offset', start))

    def load(self, base: Any, *indices: Operand, type_=None, hint: str = '',
             space: Optional[MemSpace] = None, uniform: Optional[bool] = None,
             predicate: Optional[Value] = None,
             other: Optional[Operand] = None,
             layout: Optional[RegisterLayout] = None,
             nontemporal: bool = False) -> Value:
        """``layout`` is how the loaded value ends up spread over the lanes.

        A load is where a distribution *enters* the IR: every later layout is
        derived from one of these or from an explicit relayout, so dropping it
        here leaves the whole chain untracked --- and the vendor emitters
        check operand layouts against what their intrinsics require.
        """
        if space is None:
            space = (base.type.space if isinstance(base, Value)
                     and isinstance(base.type, BufferType)
                     else MemSpace.from_symbol_type(getattr(base, 'stype', None)))
        if type_ is None:
            type_ = (ScalarType(base.type.elem)
                     if isinstance(base, Value) and isinstance(base.type, BufferType)
                     else ScalarType(self._fptype))
        if uniform is None:
            uniform = _join(indices)
        v = self.value(type_, hint=hint or 'ld', uniform=uniform, layout=layout)

        attrs = []
        if other is not None:
            attrs += [('other', other)]
        if nontemporal:
            attrs += [('nontemporal', nontemporal)]
        attrs = tuple(attrs)

        self._emit_op(Op.LOAD, (v,), (base,) + tuple(indices),
                      predicate=predicate, pure=False, movable=True,
                      effect=Effect.READ,
                      accesses=(Access(Effect.READ, space, base),),
                      attrs=attrs)
        return v

    def store(self, base: Any, value: Operand, *indices: Operand,
              space: Optional[MemSpace] = None,
              predicate: Optional[Value] = None,
              atomic: bool = False) -> Stmt:
        if space is None:
            space = (base.type.space if isinstance(base, Value)
                     and isinstance(base.type, BufferType)
                     else MemSpace.from_symbol_type(getattr(base, 'stype', None)))
        kind = Effect.ATOMIC if atomic else Effect.WRITE
        return self._emit_op(Op.STORE, (), (base, value) + tuple(indices),
                             predicate=predicate, pure=False, movable=True,
                             effect=kind,
                             accesses=(Access(kind, space, base),))

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
            uniform = _join(indices)

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

    def barrier(self, scope: Union[str, Uniformity] = Uniformity.BLOCK) -> Stmt:
        """A rendezvous of every thread that agrees at level ``scope``.

        The scope is on the same lattice as value uniformity, and that is the
        point: a barrier at level S inside a construct whose entry is only
        U-uniform deadlocks unless ``U >= S``, because the threads that took the
        other branch, or ran fewer iterations, never arrive.  Previously the
        scope was an unchecked string that never reached the emitter -- every
        barrier came out as sync_block() regardless of what was asked for.
        """
        level = _as_barrier_scope(scope)
        return self._emit_op(Op.BARRIER, (), (), pure=False, movable=False,
                             effect=Effect.BARRIER,
                             attrs=(('scope', level),))

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

    def if_(self, cond: Operand, attrs: Tuple = ()) -> '_IfHandle':
        """Guard without results --- the common case (bounds checks)."""
        return _IfHandle(self, cond, (), attrs)

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

    #: What a raw statement is assumed to touch when it does not say.
    _TOUCHES_EVERYTHING = (Access(Effect.READ | Effect.WRITE,
                                  MemSpace.UNKNOWN, None),)

    def __call__(self, code: str, *args: Operand,
                 defines: Sequence[Value] = (),
                 accesses: Optional[Sequence[Access]] = None) -> Stmt:
        """Raw statement text.  Opaque, therefore impure and pinned.

        Two separable facts, and they were being answered by one default.
        *What it is* --- opaque text, unmovable, not a candidate for CSE ---
        stays `Effect.UNKNOWN` no matter what: nothing here can reason about
        the statement as code.  *What it touches* is a different question, and
        the conservative answer to it has a cost that shows up as soon as
        anything wants to reason about memory.

        `Access(base=None)` conflicts with everything in its space, and
        `MemSpace.UNKNOWN` conflicts with every space, so a single raw
        statement between two shared-memory accesses keeps every buffer live.
        A body that is nine tenths converted therefore analyses exactly as
        badly as one that is not converted at all --- which makes the
        conversion all-or-nothing, and an all-or-nothing conversion is one
        that does not get done.

        So a caller that knows may say.  ``accesses=()`` is the common case:
        this statement touches no memory the IR models (a `__syncwarp`, a
        register declaration).  Omitting the argument keeps the old answer, so
        every existing call site means exactly what it meant before.

        This is a promise the text cannot be made to keep, which is why
        `_check_declared_accesses` holds it to the part that *is* checkable:
        a buffer this body allocated, named in the text, must appear in the
        declared set *and* among ``args``.

        ``args`` are the values the statement uses and ``defines`` the ones it
        introduces, both in `rawexpr`'s convention.  A raw declaration ---
        ``float v35[4][2]{};`` --- names a value without using it, and
        requiring that one as an operand would be asking for a use edge that
        runs backwards.

        ``args`` are the values the statement uses, in `rawexpr`'s convention.
        Declaring an access is not the same as declaring a use, and the
        difference is not academic: an `Access` tells the aliasing question
        which buffers a statement may touch, while the use chain is what keeps
        the buffer's definition alive.  A `float4` store that named its tile
        only inside the text had a correct access set and no use edge, so the
        `alloc` that produced the tile was reachable by nothing, was removed,
        and the kernel referred to an undeclared pointer.  It compiled cleanly
        as IR and not at all as C++.
        """
        if accesses is None:
            accesses = self._TOUCHES_EVERYTHING
        else:
            accesses = tuple(accesses)
            self._check_declared_accesses(code, accesses, args, defines)
        return self._emit_op(Op.RAWSTMT, tuple(defines), tuple(args),
                             pure=False, movable=False,
                             effect=Effect.UNKNOWN, text=code,
                             accesses=accesses)

    def _check_declared_accesses(self, code: str,
                                 accesses: Sequence[Access],
                                 args: Sequence[Operand] = (),
                                 defines: Sequence[Value] = ()) -> None:
        """Hold a narrowed access set to what the text visibly does.

        Only one direction is decidable: if a buffer allocated in this body is
        named in the statement, the statement touches it.  The converse is not
        --- a pointer can be reached through an alias this never sees --- so
        an undeclared *absence* is not an error, and nothing here tries to
        prove one.

        Catching the one direction is enough to matter.  The failure this
        prevents is a `store` whose buffer was left out of the declaration and
        so is reported dead while still being written, and that failure is
        silent: the generated code is unchanged, the allocation moves, and the
        result is a wrong kernel produced by a correct-looking pass.
        """
        covered = {a.base for a in accesses}
        conservative = (None in covered
                        or any(a.space == MemSpace.UNKNOWN for a in accesses))
        named = {id(a) for a in args} | {id(d) for d in defines}
        shared = {id(b) for b in self._shared_buffers}

        for name in set(_IDENT.findall(code)):
            v = self._by_name.get(name)
            if v is None:
                continue                # not one of ours: a kernel parameter,
                                        # a loop variable, a C++ keyword
            if id(v) not in named:
                raise IRError(
                    f'raw statement names {name} without listing it: '
                    f'{code.strip()[:70]!r}. Pass it in `args` if the '
                    f'statement uses it, or in `defines` if this is where it '
                    f'comes from. Left out of both, whatever produces {name} '
                    f'is reachable by nothing and will be removed.')
            if id(v) in shared and not conservative and v not in covered:
                raise IRError(
                    f'raw statement names {name} but does not declare an '
                    f'access to it: {code.strip()[:70]!r}. Either add the '
                    f'access or drop the `accesses` argument.')

    def rawexpr(self, text: str, *args: Operand, type_=None, hint: str = '',
                pure: bool = False, movable: bool = False) -> Value:
        """One escape hatch with a *single* convention: ``{0}`` is ``args[0]``.

        The result is declared by the emitter; the text is an expression, never
        a full statement.  (The old writer mixed both conventions --- ``{0}``
        meant the target in ``write()`` but the loop variable in ``For``.)
        """
        type_ = type_ or ScalarType(self._fptype)
        uniform = _join(args)
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
                  args: Sequence[Operand] = (), hint: str = 'ld',
                  layout: Optional[RegisterLayout] = None) -> Value:
        """A declaration whose right-hand side is still text, but whose result
        is a real SSA value.

        The bridge for migrating a body from the inside out: the access itself
        may stay a vendor-specific string, while everything that consumes it
        becomes structured.

        ``layout`` is how the loaded value ends up spread over the lanes, when
        the caller can say.  A load is where a distribution *enters* the IR:
        every later layout is derived from one of these or from an explicit
        relayout, so leaving them untracked leaves the whole chain untracked.
        """
        v = self.value(type_, hint=hint, layout=layout)
        self._emit_op(Op.RAWEXPR, (v,), tuple(args), pure=False, movable=True,
                      effect=kind,
                      accesses=(Access(kind,
                                       self._space_of(base) if space is None
                                       else space, base),),
                      text=text)
        return v

    def value_block(self, type_, base: Any = None, *,
                    kind: Effect = Effect.READ, hint: str = 'v',
                    layout: Optional[RegisterLayout] = None):
        """A region that produces one value by assigning to it internally.

        The escape hatch for code that is not SSA and cannot cheaply be made
        so --- the sparse loader declares a variable and then assigns to it
        under guards.  Wrapping the whole sequence gives it a *declared
        result* and a *declared memory effect*, so consumers can take the
        value as an operand even though the inside stays opaque.  The name
        comes from the shared allocator, so it no longer needs an enclosing
        scope to avoid colliding with the next instruction.
        """
        return _ValueBlock(self, type_, base, kind, hint, layout)

    def pack(self, type_, *parts: Operand, hint: str = 'pk') -> Value:
        """Aggregate initialisation: ``VecTy v{a, b};``.

        The vendor path builds a short vector to hand a pair of accumulators
        to one cross-lane instruction.  As raw text the elements were names
        baked into a string; here they are operands, so the loads that
        produced them are reachable from this statement.
        """
        v = self.value(type_, hint=hint, uniform=_join(parts),
                       layout=join_layout(parts))
        self._emit_op(Op.PACK, (v,), tuple(parts), pure=True)
        return v

    def extract(self, vec: Value, lane: int, type_=None,
                hint: str = 'el') -> Value:
        """``v[i]`` on a packed vector -- the inverse of :meth:`pack`."""
        type_ = type_ or ScalarType(self._fptype)
        v = self.value(type_, hint=hint, uniform=_join((vec,)),
                       layout=vec.layout)
        self._emit_op(Op.EXTRACT, (v,), (vec,), pure=True,
                      attrs=(('lane', lane),))
        return v

    def accumulate(self, target: Value, value: Operand) -> Stmt:
        """``target += value;`` on a declared register.

        The structured counterpart of :meth:`Accumulate`, for the same reason
        :meth:`call_stmt` exists: an accumulator is mutated in place, so it is
        not an SSA producer, but the mutation can still declare *which*
        register it touches instead of being an opaque write.  `value` goes in
        as an operand, so the computation it comes from cannot be reordered
        past this statement.
        """
        self.pin(target)
        return self._emit_op(Op.ACCUM, (), (target, value), pure=False,
                             movable=False, effect=Effect.WRITE,
                             accesses=(Access(Effect.READ | Effect.WRITE,
                                              MemSpace.REGISTER,
                                              base=target),))

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


class _ValueBlock:
    def __init__(self, builder, type_, base, kind, hint, layout=None):
        self.builder = builder
        self._type = type_
        self._base = base
        self._kind = kind
        self.value = builder.value(type_, hint=hint, layout=layout)

    def __enter__(self) -> Value:
        self.builder.push(kind='valueblock')
        return self.value

    def __exit__(self, exc_type, exc, tb):
        region = self.builder.pop()
        if exc_type is not None:
            return False
        acc = ()
        if self._base is not None:
            acc = (Access(self._kind, self.builder._space_of(self._base),
                          self._base),)
        self.builder.emit(Stmt(op=Op.RAWBLOCK, target=(self.value,),
                               regions=(region,), text='', pure=False,
                               movable=False, effect=self._kind, accesses=acc))
        return False


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
    def __init__(self, builder, cond, types, attrs=()):
        self.builder = builder
        self.cond = cond
        self._types = types
        self._attrs = tuple(attrs)
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
                               pure=False, movable=False, attrs=self._attrs))
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
