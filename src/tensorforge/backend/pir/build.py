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

from .core import (BOOL, INDEX, SCALAR_LAYOUT, TOKEN, Access, BufferType,
                   Effect, IRError,
                   LaneAxis, MemSpace, Op, Operand, Region, RegisterLayout,
                   ScalarType, Stmt, TokenType, Value, XorSwizzle, dump, walk,
                   join_layout, Uniformity)


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
    #: Monotonic, never reused.  `id()` is not an identity for this purpose:
    #: a builder that has been finished and collected frees its address, and
    #: the next builder can be handed the same one.  Anything keyed on `id()`
    #: then treats two different bodies as the same body -- which is exactly
    #: the mistake `Symbol.pir_buffer` is there to prevent, so it must not be
    #: the mechanism that makes it.
    _next_uid = 0

    def __init__(self, fptype: Datatype = Datatype.F32, context: Any = None,
                 alloc: Any = None, scratch: Optional[Tuple[str, int]] = None):
        IRBuilder._next_uid += 1
        self.uid = IRBuilder._next_uid
        #: value id -> the buffer its accesses belong to (see `decl_expr`)
        self._view_root = {}
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
              uniform: Union[bool, Uniformity] = Uniformity.GRID,
              layout: Optional[RegisterLayout] = None) -> Value:
        """An integer, replicated across the wave unless said otherwise.

        A loop counter and an address are the same in every lane: the loop is
        entered the same number of times and the subscript is computed from
        the same operands.  `None` here said *unknown*, which is a weaker
        claim than the truth and one an explicitly vectorised emitter cannot
        act on -- it has to decide between `int` and `simd<int, N>` at the
        declaration, and unknown is not one of the two.

        A genuinely lane-varying index does exist (a gather's address vector),
        and it will have to pass its layout in rather than inherit this
        default.  Stated as a default rather than a hard rule for that reason.
        """
        return self.value(INDEX, hint=hint, uniform=uniform,
                          layout=SCALAR_LAYOUT if layout is None else layout)

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
        """A literal.  Replicated by definition -- there is no lane in which
        it is a different number, so this is the one distribution that needs
        no derivation and can never be wrong."""
        type_ = type_ or ScalarType(self._fptype)
        v = self.value(type_, hint='c', layout=SCALAR_LAYOUT)
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

    def assign(self, target: Value, value: Operand) -> Stmt:
        """``target = value;`` where `target` was produced by `declare`.

        The one shape `declare` left without a verb.  A declared value is not
        an SSA producer --- it exists because something writes it through a
        reference or across a guard --- so the write has to be its own
        statement, and until now that statement was raw text.  864 of them in
        the NVIDIA epilogue alone, each naming two values the IR already knew.

        Modelled exactly as `call_stmt`'s ``writes``: a declared register
        access keyed on the target, so two assignments to different values
        provably do not conflict, and the target pinned because a value whose
        name is assigned to must keep that name rather than be folded into its
        consumer.
        """
        self._require_addressable(target, 'assign')
        self.pin(target)
        return self._emit_op(
            Op.CALL, (), (target, value), pure=False, movable=False,
            effect=Effect.WRITE,
            accesses=(Access(Effect.WRITE, MemSpace.REGISTER, base=target),),
            attrs=(('assign', True),))

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

    def asm_stmt(self, template: str, operands: Sequence[Tuple[str, Operand]],
                 *, movable: bool = False) -> Stmt:
        """Inline assembly whose operands are values, not baked-in names.

        `mma.sync` has no intrinsic, so the NVIDIA path has to emit PTX
        directly.  That does not make it opaque.  What the statement reads and
        writes is exactly as knowable as for any other vendor primitive: the
        operands go in as values, the ones with a read-write or write-only
        constraint go in as declared register accesses, and two `mma.sync`
        calls on different accumulators then provably do not conflict.

        ``operands`` is ordered, `(constraint, value)`, outputs and read-write
        operands first.  That order is not a convention, it is how the
        assembler numbers them: outputs and inputs share one sequence, `%0`
        onwards, so moving an operand renumbers everything after it.  The
        caller writes `%0`-style placeholders in ``template`` and this checks
        that the two agree --- a mismatch there reads the wrong registers and
        still compiles, which is the failure this node exists to make
        impossible.

        Not `Op.ASM`: the effects are `Op.CALL`'s effects and the only
        difference is the rendering, so a second op would be a second set of
        rules for the passes to learn.
        """
        seen_input = False
        for constraint, _ in operands:
            is_out = constraint.startswith(('=', '+'))
            if is_out and seen_input:
                raise IRError(
                    f'asm operands are numbered in one sequence, so outputs '
                    f'have to come first; {constraint!r} follows an input')
            seen_input = seen_input or not is_out

        wanted = {f'%{i}' for i in range(len(operands))}
        found = set(re.findall(r'%\d+', template))
        if found != wanted:
            raise IRError(
                f'asm template references {sorted(found)} but {len(operands)} '
                f'operands were given. Numbering is positional, so a mismatch '
                f'reads the wrong registers and still compiles.')

        writes = [v for c, v in operands if c.startswith(('=', '+'))]
        for w in writes:
            self._require_addressable(w, 'asm')
            self.pin(w)
        accesses = tuple(Access(Effect.READ | Effect.WRITE, MemSpace.REGISTER,
                                base=w) for w in writes if isinstance(w, Value))
        return self._emit_op(
            Op.CALL, (), tuple(v for _, v in operands), pure=False,
            movable=movable,
            effect=Effect.WRITE if accesses else Effect.NONE,
            accesses=accesses,
            attrs=(('asm', template),
                   ('constraints', tuple(c for c, _ in operands))))

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

    def lane_index(self, block: int, stride: int = 1,
                   hint: str = 'lead') -> Value:
        """*Which* element of a distributed dimension a lane holds.

        The third of the three questions SPMD answers with one register, and
        the only one whose answer is not a scalar in both models.
        :meth:`thread_id` asks which thread; :meth:`lane_offset` asks where a
        lane's share starts in the address; this asks which index the lane is
        *at*, which is what a bounds guard compares against.

        SPMD: ``(tid / stride) % block``, one integer per thread.

        Explicitly vectorised: ``0, 1, ... block-1`` as a vector, because the
        work-item holds every element of the dimension and "which index am I
        at" has ``block`` answers at once.  A guard over it is therefore a
        mask, not a branch -- which is a statement about the *whole* enclosing
        region, so `Op.IF` on a value from here has to be if-converted before
        it can be emitted.  The ESIMD emitter refuses it otherwise rather than
        lowering a vector into a branch condition.
        """
        if not self._explicit_simd():
            lane = self.op('div', INDEX, self.thread_id('x'), stride,
                           hint='lane')
            return self.op('rem', INDEX, lane, block, hint=hint)
        # `simd<int, block>(0, 1)`: the ESIMD constructor for a linear
        # progression.  Emitted through `rawexpr` rather than a new op, since
        # what makes it a lane index is the layout, not the spelling.
        return self.rawexpr(
            f'{self._simd_spelling(block)}(0, 1)',
            type_=INDEX, hint=hint, pure=True, movable=True,
            layout=RegisterLayout((LaneAxis(block, stride),)))

    def _simd_spelling(self, block: int) -> str:
        lex = self.context.get_vm().get_lexic()
        return lex.get_simd(INDEX.base.ctype(), block)

    def lane_broadcast(self, vec: Value, lane: int, block: int,
                       hint: str = 'bc') -> Value:
        """The value `vec` holds at `lane`, replicated across the wave.

        The fourth question in the family with :meth:`thread_id`,
        :meth:`lane_offset` and :meth:`lane_index`, and the one where the two
        models differ most in *cost* rather than only in spelling.

        SPMD has to move the value: lane `l`'s register is not readable from
        lane `l'`, so this is a cross-lane instruction -- `__shfl`, a DPP
        broadcast, `group_broadcast`.  It is why `amd/relayout.py` exists.

        Explicitly vectorised, the whole vector is in this work-item's own
        registers and `v[lane]` is an ordinary element read.  The broadcast
        costs nothing, which is what makes a register-only matmul preferable
        to staging operands through shared memory here where it is not on
        AMD.

        The result is *replicated* in both models -- one value, the same in
        every lane -- which is why this is not `extract`: that one indexes the
        slot axis (`ScalarType.length`) and keeps the lane distribution.
        """
        out = self.value(vec.type, hint=hint, uniform=Uniformity.MULT,
                         layout=SCALAR_LAYOUT)
        if self._explicit_simd():
            self._emit_op(Op.EXTRACT, (out,), (vec,), pure=True,
                          attrs=(('lane', lane),))
            return out
        lex = self.context.get_vm().get_lexic()
        text = lex.broadcast('{0}', lane, block)
        return self.rawexpr(text, vec, type_=vec.type, hint=hint,
                            pure=True, movable=False, layout=SCALAR_LAYOUT)

    def lane_offset(self, block: int, stride: int = 1,
                    hint: str = 'lane') -> Operand:
        """The address contribution of one lane within a distributed dimension.

        Not the same question as :meth:`thread_id`, even though SPMD answers
        both with the same register.  ``thread_id`` asks *which thread am I*;
        this asks *where does my share of this dimension start*.  They coincide
        only because SPMD spreads the dimension across the threads -- so the
        two were one call, and separating them is what lets a second model
        answer them differently.

        SPMD: ``(tid / stride) % block``, the lane's element of the dimension.

        Explicitly vectorised: ``0``.  The dimension *is* the vector, the
        work-item holds all of it, and its base offset in the register is
        zero.  The lane term then folds out of the address by the ordinary
        identity rules -- ``add(0, x) -> x`` -- rather than by a second code
        path that has to stay in step with the first.

        ``block == 1`` is not distributed at all, so there is no contribution
        to make in either model.
        """
        if block <= 1:
            return 0
        if self._explicit_simd():
            return 0
        lane = self.op('div', INDEX, self.thread_id('x'), stride, hint=hint)
        return self.op('rem', INDEX, lane, block, hint=hint)

    def _explicit_simd(self) -> bool:
        """Whether the lowering puts the lane in the type rather than the address."""
        try:
            return bool(self.context.get_vm().get_lexic().simd_mode)
        except AttributeError:
            return False

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
        # MULT-uniform is a statement about the lanes: every thread of one
        # multiplication has the same batch id, which is exactly replication.
        v = self.value(INDEX, hint=f'batch{lookahead}', uniform=Uniformity.MULT,
                       layout=SCALAR_LAYOUT)
        self._emit_op(Op.CALL, (v,), (),
                      attrs=(('callee', f'batch_id_{lookahead}'),))
        return v

    def alloc(self, elem: Datatype, shape: Sequence[int], space: MemSpace,
              hint: str = 'buf', extern: str = None,
              init: str = '', arena: str = None, offset=0,
              align: Optional[int] = None,
              restrict: str = None,
              swizzle: Optional[XorSwizzle] = None) -> Value:
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
        v = self.value(BufferType(elem, tuple(shape), space, swizzle),
                       hint=hint)
        attrs: Tuple = ()
        if arena is not None:
            # A window the *region* allocator placed, not the scratch bump
            # allocator.  `_suballocate` hands out offsets inside this
            # instruction's `tempShrMem` tail; `ShrMemOpt` places the shared
            # tiles in `localShrMem0` and hands the offset in from outside.
            # Two allocators, one arena, and only one of them may pick an
            # offset for any given buffer -- so an externally placed window
            # says so rather than asking for one it would then have to ignore.
            attrs = (('arena', arena), ('offset', offset))
            if restrict:
                attrs = attrs + (('restrict', restrict),)
        elif space == MemSpace.SHARED:
            attrs = self._suballocate(v, elem)
            self._shared_buffers.append(v)
        if extern is not None:
            # A name the macro layer owns and other instructions still spell
            # out as text.  Transitional, and measurably so: with one PIR body
            # per loop body, 89.8% of buffers have their definition and all
            # their uses inside one body and need no name at all once the
            # consumers take the value (tools/buffer_spans.py).  What is left
            # is the shared arena, its scratch tail, and the tiles of the two
            # cases that have two batch loops -- things that genuinely outlive
            # a body.  So this set shrinks per migrated consumer rather than
            # becoming the permanent way values are addressed.
            #
            # `escapes` is not decoration here.  Making the allocation
            # structured also makes it deletable, and the reads that justify
            # it are still raw text the IR cannot see, so DCE removes the
            # declaration and leaves the uses referring to a name that no
            # longer exists.  That produces a corpus which still renders and
            # no longer compiles, which the snapshot harness cannot catch.
            attrs = attrs + (('extern', extern), ('escapes', True))
        if init:
            attrs = attrs + (('init', init),)
        if align is not None:
            # Declared, not hoped for.  A plain `float r[8]` is 4-byte aligned
            # by every rule that applies to it; that compilers usually give it
            # more is not something a reinterpret cast may rely on.  This is
            # the other half of `Symbol.linear_align_bytes`, which reports the
            # same number back to whoever picks a width.
            attrs = attrs + (('align', align),)
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

    def _swizzled(self, base: Any, index: Operand) -> Operand:
        """The index a swizzled buffer is actually addressed by.

        Applied here, in the one place every read and write of a buffer passes
        through, rather than at the call sites.  A permutation that only some
        accesses apply is worse than none: the store and the load would
        disagree about where an element lives, and the kernel would be quietly
        wrong instead of merely slow.  Nothing outside this method knows the
        buffer is swizzled, which is the property that makes it safe to turn
        on for a tile that already works.
        """
        # A `Symbol` base resolves to its buffer the same way the rest of the
        # access path does.  Reading `.type` off the base alone missed every
        # macro-level window -- `Symbol.load` passes the symbol, not the value
        # -- so a swizzle set at the alloc was accepted and then applied to
        # nothing.
        buf = base.pir_buffer(self) if hasattr(base, 'pir_buffer') else base
        t = getattr(buf if buf is not None else base, 'type', None)
        swz = getattr(t, 'swizzle', None)
        if swz is None:
            return index
        if isinstance(index, int):
            return swz.apply(index)
        # `i ^ ((i / width) % width)`, as IR ops -- the emitter sees arithmetic
        # it can fold when the index is constant, not a string it cannot read.
        # A shift and a mask, not a divide and a modulo: the width is a power
        # of two, and this way the emitted address needs no strength reduction
        # to be what one would have written by hand.
        bits = swz.width.bit_length() - 1
        row = self.op('shr', INDEX, index, bits, hint='sw')
        sel = self.op('bitand', INDEX, row, swz.width - 1, hint='sw')
        return self.op('bitxor', INDEX, index, sel, hint='sw')

    def load(self, base: Any, *indices: Operand, type_=None, hint: str = '',
             space: Optional[MemSpace] = None, uniform: Optional[bool] = None,
             predicate: Optional[Value] = None,
             other: Optional[Operand] = None,
             layout: Optional[RegisterLayout] = None,
             align: Optional[int] = None,
             nontemporal: bool = False,
             extern: str = None) -> Value:
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
        indices = tuple(self._swizzled(base, i) for i in indices)
        if uniform is None:
            uniform = _join(indices)
        v = self.value(type_, hint=hint or 'ld', uniform=uniform, layout=layout)

        attrs = []
        if other is not None:
            attrs += [('other', other)]
        if nontemporal:
            attrs += [('nontemporal', nontemporal)]
        if align is not None:
            # Either a byte count -- what the caller proved about this
            # address -- or the string `'relaxed'`, meaning the access is
            # spelled with a type that declares element alignment and so
            # needs no proof.  What the*caller* proved about this address, in bytes.  A wide
            # access is spelled as a reinterpret cast, and the cast's legality
            # is not recoverable from the IR: the address is an expression,
            # often a string.  So it is carried rather than derived, and
            # `verify` rejects a wide access that carries nothing -- which
            # makes "nobody checked" a state the IR cannot be in, without the
            # IR having to understand the arithmetic.
            attrs += [('align', align)]
        if extern:
            # The name the macro layer already handed out.  A load that has to
            # produce a particular identifier used to be text for that reason
            # alone, which cost every pass its view of an access that was
            # otherwise fully described.
            attrs += [('extern', extern)]
        attrs = tuple(attrs)

        self._emit_op(Op.LOAD, (v,), (base,) + tuple(indices),
                      predicate=predicate, pure=False, movable=True,
                      effect=Effect.READ,
                      accesses=(Access(Effect.READ, space,
                                       self.alias_root(base)),),
                      attrs=attrs)
        return v

    def store(self, base: Any, value: Operand, *indices: Operand,
              space: Optional[MemSpace] = None,
              predicate: Optional[Value] = None,
              align: Optional[int] = None,
              atomic: bool = False,
              nontemporal: bool = False,
              pointer: Optional[str] = None) -> Stmt:
        """``nontemporal`` is a cache hint, carried the way ``Op.LOAD`` carries
        its own: as an attribute the emitter hands to ``lexic.glb_store``.

        It has to travel with the statement rather than be baked into a string
        at the call site, because baking it in is what kept global stores off
        this path -- ``Symbol.store`` asked the lexic for a finished statement
        and then had nothing structured left to emit."""
        if space is None:
            space = (base.type.space if isinstance(base, Value)
                     and isinstance(base.type, BufferType)
                     else MemSpace.from_symbol_type(getattr(base, 'stype', None)))
        indices = tuple(self._swizzled(base, i) for i in indices)
        kind = Effect.ATOMIC if atomic else Effect.WRITE

        attrs = []
        if nontemporal:
            attrs += [('nontemporal', nontemporal)]
        if pointer is not None:
            # The pointer written *through*, when it is not the symbol's own
            # name.  A rotating shared buffer fills a stage other than the one
            # its consumers read, and `Op.STORE`'s base is the symbol -- so the
            # override is a spelling the emitter applies, not a different
            # destination.  It deliberately does not change `alias_root`: the
            # stages are the same buffer, and a pass that thinks otherwise
            # would reorder a fill past a read of the stage it fills.
            attrs += [('pointer', pointer)]
        if align is not None:
            # See `load`: the alignment a wide access needs is proved by the
            # caller and carried, because the address is an expression the IR
            # cannot evaluate.
            attrs += [('align', align)]
        attrs = tuple(attrs)

        return self._emit_op(Op.STORE, (), (base, value) + tuple(indices),
                             predicate=predicate, pure=False, movable=True,
                             effect=kind, attrs=attrs,
                             accesses=(Access(kind, space,
                                              self.alias_root(base)),))

    def copy_async(self, dst: Any, src: Any, *,
                   dst_index: Sequence[Operand] = (),
                   src_index: Sequence[Operand] = (),
                   elems: int = 1,
                   dst_space: Optional[MemSpace] = None,
                   src_space: Optional[MemSpace] = None,
                   predicate: Optional[Value] = None,
                   hint: str = 'cp') -> Value:
        """Issue an asynchronous copy ``src -> dst``; returns its token.

        The write to ``dst`` is not observable until the matching :meth:`wait`.
        Copy and wait carry the *same* accesses, which is what keeps any read
        of ``dst`` from being hoisted above the wait --- the reorder machinery
        needs no notion of asynchrony beyond that.

        ``elems`` is the extent one lane moves, and ``predicate`` is what makes
        the extent enough.  A macro copy is not a tile: the loader splits a
        transfer into hops of 4, 2 and 1 elements per lane and then has
        ``length % num_threads`` elements left over, which the last hop moves
        under ``linear_idx < rest``.  A shape that only admitted whole tiles
        would have to exclude that hop, and the ragged tail is not the rare
        case --- it is every transfer whose length is not a multiple of the
        block.

        Predication does not change the token.  The copy issues for the wave
        whenever any lane is active, so it counts once against the hardware
        counter either way, and the guard is a real branch rather than a
        select because the token has no C++ value to select on.

        The token granularity is independent of all of this.  A wait retires
        every copy up to and including the one it names, so a wait on the last
        token of a group retires the group --- one token per macro copy falls
        out of that rather than needing a group object to express it.
        """
        dst_space = dst_space if dst_space is not None else self._space_of(dst)
        src_space = src_space if src_space is not None else self._space_of(src)
        accesses = (Access(Effect.READ, src_space, self.alias_root(src)),
                    Access(Effect.WRITE, dst_space, self.alias_root(dst)))

        tok = self.value(TOKEN, hint=hint)
        self._emit_op(Op.COPY_ASYNC, (tok,),
                      (dst, src) + tuple(dst_index) + tuple(src_index),
                      pure=False, movable=True, predicate=predicate,
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

        accesses = (Access(Effect.READ, space, self.alias_root(base)),)
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

    def wait(self, token: Optional[Value] = None, *also: Value) -> Stmt:
        """Wait for ``token`` (or, with ``None``, drain every outstanding copy).

        Several tokens may be named, and for a macro copy they have to be.
        ``schedule_async`` retires the waited token *and everything issued
        before it in its own class* --- that is what the hardware counter does,
        and it is why one wait suffices for a transfer split into hops.  But
        `check_tokens` requires every token to be consumed exactly once, and it
        runs before the schedule exists, so it cannot see that a later wait
        already retired the earlier hops.  Naming them makes the two agree
        without either having to guess: the emitted code is unchanged, since
        ``prior`` is still derived from one position.

        This is also the whole of what a "group" needs to be.  A group object
        would carry a stage count, an acquire and a release; naming the tokens
        a wait retires carries the same information as a def-use edge, which
        every pass already understands.

        The concrete counter value (``vmcnt(N)`` / ``wait_prior(N)``) is not
        decided here --- ``schedule_async`` derives it from the outstanding set
        once the schedule is final.
        """
        tokens = ((token,) + also) if token is not None else ()
        if not tokens:
            accesses = (Access(Effect.READ | Effect.WRITE, MemSpace.UNKNOWN, None),)
        else:
            fallback = (Access(Effect.READ | Effect.WRITE, MemSpace.UNKNOWN, None),)
            accesses = tuple(a for t in tokens
                             for a in self._token_accesses.get(t.id, fallback))
        results: Tuple[Value, ...] = ()
        if tokens:
            # Only the named token can release values; the hops retired
            # alongside it are copies, which produce none.
            types = self._token_results.get(tokens[0].id, ())
            results = tuple(self.value(t, hint=tokens[0].hint or 'ld',
                                       uniform=self._token_uniform.get(tokens[0].id, True))
                            for t in types)
        stmt = self._emit_op(Op.WAIT, results, tokens,
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
             unroll: bool = False, hint: str = 'i', extern: str = None,
             ctype: str = None, next_index=None) -> '_ForHandle':
        """A loop.  ``extern`` and ``ctype`` are for loops the macro layer owns.

        An inner loop is the IR's own: it picks the induction variable's name
        and renders its type from `INDEX`.  A loop that already exists in
        generated code is not -- the batch loop's variable is `batchId0`,
        spelled out by the lookahead bindings, the flag guard and every
        `access_address` in the body, and its type is `size_t` rather than
        `int32_t` because it is compared against `numElements`.

        Same trade as `extern` on `alloc`, and it ends the same way: the name
        is needed while the things that spell it are still text, and stops
        being needed as they migrate.
        """
        return _ForHandle(self, lo, hi, step, tuple(inits), tuple(types),
                          unroll, hint, extern, ctype, next_index)

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
                pure: bool = False, movable: bool = False,
                layout: Optional[RegisterLayout] = None) -> Value:
        """One escape hatch with a *single* convention: ``{0}`` is ``args[0]``.

        The result is declared by the emitter; the text is an expression, never
        a full statement.  (The old writer mixed both conventions --- ``{0}``
        meant the target in ``write()`` but the loop variable in ``For``.)
        """
        type_ = type_ or ScalarType(self._fptype)
        uniform = _join(args)
        # Same join as `op()`, and for the same reason: an expression over
        # operands that are spread across the lanes produces a value spread the
        # same way.  Left off, every elementwise result was untracked -- the
        # text is opaque to the IR, but its *shape* is not, and a raw
        # expression is still elementwise over its operands.
        # An explicit `layout` overrides the join: a raw expression whose
        # *text* introduces a distribution its operands do not have -- a lane
        # index built out of nothing -- can say so, and there is no other way
        # for it to, since the IR cannot read the text.
        v = self.value(type_, hint=hint, uniform=uniform,
                       layout=layout if layout is not None else join_layout(args))
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
                             accesses=(Access(kind, space,
                                              self.alias_root(base)),),
                             text=text,
                             attrs=(('fmt', True),) if fmt else ())

    def decl_expr(self, decl: str, text: str, type_, base: Any, *,
                  kind: Effect = Effect.READ, space: Optional[MemSpace] = None,
                  args: Sequence[Operand] = (), hint: str = 'ptr',
                  extern: str = None, alias_root: Any = None) -> Value:
        """A declaration whose declarator is text too, not only its right side.

        `load_expr` renders `{ctype} {name} = {text};`, which is enough while
        the thing being declared is a scalar.  A pointer binding is not: it
        reads `const float *const __restrict__ p`, or on the AMD pointer-based
        path `auto p` with the type inside a cast on the right.  Rendering
        that from a `type_` would mean teaching the emitter a declarator
        grammar for one caller.

        So the declarator stays text and the *result* becomes a value, which
        is the half that matters: consumers can address the buffer instead of
        interpolating its name, and the def-use edge exists, so a scheduler
        knows the binding cannot sink below a read through it.

        `alias_root` is not optional bookkeeping.  `may_alias` treats two
        distinct bases as never aliasing, so a view that claimed its own
        identity would let a write through the underlying buffer reorder past
        a read through the window.  The root is what the accesses are recorded
        against, so a window is the buffer it is a window into.
        """
        v = self.value(type_, hint=hint)
        if base is None:
            # Pure text with no memory behind it: the lookahead bindings are
            # index arithmetic over the induction variable and the grid shape,
            # and they read nothing.  Declaring an access with no base would
            # say the opposite -- `base=None` conflicts with everything in its
            # space -- so a computation that may move anywhere would pin the
            # body it sits in.
            if extern is None:
                raise IRError('decl_expr needs `extern`; see below')
            self._emit_op(Op.RAWEXPR, (v,), tuple(args), pure=True,
                          movable=True, effect=Effect.NONE, accesses=(),
                          text=text,
                          attrs=(('decl', decl), ('extern', extern),
                                 ('escapes', True)))
            return v
        if extern is None:
            raise IRError(
                'decl_expr needs `extern`: the declarator is caller text and '
                'has to spell the same name the emitter binds, and the caller '
                'cannot know that name otherwise. Passing a declarator with a '
                'name of its own emits a definition of one variable and a use '
                'of another, which compiles only when some other statement '
                'happens to have defined the second.')
        root = base if alias_root is None else alias_root
        self._view_root[v.id] = root
        attrs = (('decl', decl), ('escapes', True))
        if extern is not None:
            attrs = attrs + (('extern', extern),)
        self._emit_op(Op.RAWEXPR, (v,), tuple(args), pure=False, movable=True,
                      effect=kind,
                      accesses=(Access(kind,
                                       self._space_of(base) if space is None
                                       else space, root),),
                      text=text, attrs=attrs)
        return v

    def alias_root(self, base: Any) -> Any:
        """The buffer an operand's accesses should be recorded against.

        A symbol and the value that stands for it *in this body* are one
        buffer, so they have to reach the access model as one object.  They
        did not: the structured path (`load`/`store`) records against the
        value, the text path (`load_expr`/`access_stmt`) against the symbol,
        and `may_alias` compares bases with `is` --- so a write through one
        was invisible to a read through the other and `load_cse` could reuse
        a load across it.  Nothing in the corpus takes both paths for one
        symbol in one body today, which is why it never fired; the gate that
        keeps it that way is `vec == 1` in `Symbol.load_linear`.

        `pir_buffer` returns `None` for a value belonging to another body,
        which is the wanted answer there: in that body the symbol is the only
        identity, and every path uses it.
        """
        buf = base.pir_buffer(self) if hasattr(base, 'pir_buffer') else None
        if buf is not None:
            base = buf
        # `seen` bounds the walk: `_view_root` is built one entry at a time by
        # `decl_expr` and a cycle would be a defect, not something to hang on.
        seen = set()
        vid = getattr(base, 'id', None)
        while vid is not None and vid in self._view_root and vid not in seen:
            seen.add(vid)
            base = self._view_root[vid]
            vid = getattr(base, 'id', None)
        return base

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
                                       else space, self.alias_root(base)),),
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
        body = tuple(self._stack[0].body)
        self._check_swizzles_are_total(body)
        return body

    def _check_swizzles_are_total(self, body: Tuple[Stmt, ...]) -> None:
        """No access to a swizzled buffer may bypass `load` and `store`.

        The earlier form of this asked whether the buffer was `extern`, which
        was the same question only for as long as every named access was text.
        It is not the same question: `extern` is about the *name* escaping, and
        what matters is whether an *access* does.  A buffer whose name the
        macro layer owns is fine as long as every read and write of it goes
        through the two methods that apply the permutation.

        Asked here, over the finished body, because that is the first point at
        which the answer is knowable -- the buffer is allocated long before its
        accesses are emitted.  The failure it prevents is the bad kind: a
        permutation applied to some accesses and not others is a store and a
        load that disagree about where an element lives, which is a wrong
        kernel and not a slow one.
        """
        swizzled = {}
        for stmt, _ in walk(body):
            if stmt.op != Op.ALLOC:
                continue
            for t in stmt.target:
                if getattr(t.type, 'swizzle', None) is None:
                    continue
                swizzled[str(t)] = t
                # Also the name the macro layer gave it, which is what raw
                # text spells.  Checking only the value's own identifier is
                # how an unpermuted `memcpy_async` into a swizzled window got
                # past this: it writes `s0[...]`, not `v12_s0[...]`.
                #
                # Only half a guard even so, and worth saying plainly: the
                # copy is emitted by a *different* instruction's body, which
                # is what `extern` is for, so a check over one body cannot see
                # it.  The loader declines the swizzle in that case instead;
                # this catches the same mistake made locally.
                extern = stmt.attr('extern')
                if extern:
                    swizzled[str(extern)] = t
        if not swizzled:
            return
        for stmt, _ in walk(body):
            text = stmt.text
            if not text or text.strip().startswith('//'):
                # A comment naming the buffer describes it, it does not access
                # it.
                continue
            for name, v in swizzled.items():
                if _names_in(text, name):
                    raise IRError(
                        f'{name} is swizzled but is named in raw text: '
                        f'{text.strip()[:70]!r}. The permutation is applied by '
                        f'load and store; an access that goes around them '
                        f'reads a different element than the one that wrote '
                        f'it. Convert that access, or drop the swizzle.')

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
                          self.builder.alias_root(self._base)),)
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
    def __init__(self, builder, lo, hi, step, inits, types, unroll, hint,
                 extern=None, ctype=None, next_index=None):
        if len(inits) != len(types):
            raise IRError('for_: one result type per init value required')
        self._extern = extern
        self._ctype = ctype
        # What this loop calls the *next* element.  A clamped successor index
        # is a property of the traversal, which the loop knows and the IR does
        # not, so `wrap_prefetch` reads it here rather than deriving it.
        self._next_index = next_index
        self.builder = builder
        self._args = (lo, hi, step) + inits
        self._types = types
        self._unroll = unroll
        self.induction = builder.index(hint=hint)
        # A loop-carried value is distributed exactly like the init it starts
        # from -- the back edge cannot change how a value is spread across the
        # lanes, only what it holds.  Left untracked, an accumulator became a
        # hole in the middle of an otherwise fully tracked body.
        self.iter_args = tuple(
            builder.value(t, hint=f'acc{i}',
                          layout=(inits[i].layout
                                  if i < len(inits) and isinstance(inits[i], Value)
                                  else None))
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
        # Same argument on the way out: a result is the last value the body
        # yielded, and yielding does not redistribute.
        self.results = tuple(
            self.builder.value(t, hint=f'res{i}',
                               layout=(region.yielded[i].layout
                                       if i < len(region.yielded)
                                       and isinstance(region.yielded[i], Value)
                                       else None))
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
        if self._extern is not None:
            attrs = attrs + (('extern', self._extern),)
        if self._ctype is not None:
            attrs = attrs + (('ctype', self._ctype),)
        if self._next_index is not None:
            attrs = attrs + (('next', self._next_index),)
        self.builder.emit(Stmt(op=Op.FOR, target=self.results, args=self._args,
                               regions=(region,), pure=False, movable=False,
                               attrs=attrs))
        return False

    def yield_(self, *values: Operand):
        """Yield the loop-carried values, and settle their distribution.

        An `iter_arg` takes its layout from the `init` it starts at, which is
        right whenever the init has one.  A reduction's does not: it starts at
        the operator's neutral element, a bare literal, and the accumulator
        only becomes lane-distributed once the body combines it with something
        that is.

        So the arg adopts the yielded value's layout when it has none of its
        own.  The two are the same distribution by construction -- an
        `iter_arg` and what is yielded back into it are one value seen at the
        two ends of the back edge, and a loop whose carried value changed
        distribution between them would not be expressible at all.

        Patched rather than passed in, because the yield is the first moment
        it is known: the body has to be built to find out what the
        accumulation produces, and the arg is what the body was built against.
        """
        for arg, val in zip(self.iter_args, values):
            if (arg.layout is None and isinstance(val, Value)
                    and val.layout is not None):
                object.__setattr__(arg, 'layout', val.layout)
                if val.uniformity < arg.uniformity:
                    object.__setattr__(arg, 'uniformity', val.uniformity)
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
