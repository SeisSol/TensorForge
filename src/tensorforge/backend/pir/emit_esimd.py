# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Explicitly vectorised lowering of the pseudo-IR, for Intel ESIMD.

The difference from :class:`~tensorforge.backend.pir.emit.Emitter` is one
sentence long, and everything else follows from it:

    **SPMD puts the lane in the address.  ESIMD puts it in the type.**

A value spread over sixteen lanes is `float x` in SPMD, and which element a
lane holds is decided by the `(tid / stride) % block` term inside the
subscript.  The same value in ESIMD is `simd<float, 16> x`, the subscript has
no lane term at all, and the distribution has moved from the index expression
into the declaration.

Which is why this file could not be written before ``Value`` carried a total
distribution.  The information was always there -- ``LeadIndex`` computed it
and printed it -- but it was spent on an index and not recorded, and a value
whose distribution is unknown cannot be given a type here.  There is no
conservative fallback: in SPMD an untracked value is merely one that passes
cannot optimise, so `None` costs precision; here it is a declaration that
cannot be written, so `None` costs the kernel.

That is deliberate, and the error message says so.  A silent guess would pick
`float` for a value that is really a vector, and the result compiles, runs,
and is wrong -- which is the failure mode the ESIMD stubs already had once.
"""

from __future__ import annotations

import re

from typing import Any, Optional

from tensorforge.common.basic_types import Datatype

from .core import (BufferType, IRError, MemSpace, Op, ScalarType, TokenType,
                   Value, def_use, walk_stmts)
from .emit import Emitter, _folds_predicate


#: `*(VectorT<float, 4>*)&p[i]` -- how the base emitter spells a
#: vector-width access.
_VECTOR_ACCESS = re.compile(r'^\*\(\s*[^)]*?\s*\*\s*\)\s*&\s*(.*)$')


class EsimdEmitter(Emitter):
    """Lowering where a value's distribution is part of its C++ type."""

    #: Values whose type could not be decided, in encounter order.  Collected
    #: rather than raised on the first one: during the migration the useful
    #: question is *how many and which*, and a generator that stops at the
    #: first tells you nothing about the size of what is left.  `run()` raises
    #: at the end if any were collected.
    def __init__(self, writer, context: Any = None, strict: bool = True):
        super().__init__(writer, context)
        self.strict = strict
        self.unresolved: list = []
        #: Register buffers held as one `simd`, by name (`_plan_register_buffers`).
        self._simd_buffers: set = set()
        self._alloc_names: dict = {}
        #: Scalar shared-memory reads served from a register window
        #: (`_plan_windows`): statement id -> (window, element).
        self._windows: dict = {}
        #: The first read of each window: statement id -> (buffer, lo, width, name).
        self._window_heads: dict = {}

    # -- types ------------------------------------------------------------- #

    def ctype(self, t, value: Optional[Value] = None) -> str:
        if isinstance(t, TokenType) or not isinstance(t, ScalarType):
            # Buffers and tokens are unchanged: a buffer is memory, and memory
            # has no lane distribution -- what varies is who reads it.
            return super().ctype(t, value)

        if value is None:
            # A type with no value behind it.  The base emitter has a few of
            # these (loop induction variables built inline); they are scalars
            # by construction, so the base spelling is right.
            return super().ctype(t, value)

        if value.layout is None:
            self.unresolved.append(value)
            # A placeholder that will not compile, so that emission can
            # continue and report *all* of them.  `strict` (the default) turns
            # the collection into an exception before anything sees this.
            return f'/* untracked: {value!r} */ {super().ctype(t, value)}'

        if not value.distributed and (t.length or 1) > 1:
            # Replicated but still a vector: its width comes from the slot
            # axis rather than the lane axis.  A DPAS fragment is the case --
            # a `simd<TF32, 128>` whose element order the hardware fixes, held
            # whole by one work-item and spread over no lanes at all.
            #
            # Still a `simd`, not the `sycl::vec` the base emitter would spell:
            # a `vec` has no `select`, no `copy_from` and nothing a fragment is
            # written through.  Under this lowering *every* vector is a `simd`;
            # what differs between the two cases is only where the width comes
            # from.
            return self.simd_type(t.base.ctype(), t.length)

        if not value.distributed:
            # Replicated: every lane holds the whole thing, which is exactly
            # what a scalar is.  Note this is *not* the same answer as the
            # untracked case above even though the spelling matches -- one is
            # derived, the other is a hole.
            return super().ctype(t, value)

        if t.base is Datatype.BOOL:
            # A distributed boolean is a *mask*, not a vector of bools.
            # `simd<bool, N>` exists as a type but is not what a comparison
            # over a `simd` produces and not what a predicated operation
            # takes; ESIMD keeps masks in their own family precisely because
            # the hardware does.  Spelling this `simd<bool, N>` compiled the
            # declaration and then failed at every use, which is the worst
            # place to find out.
            return self.mask_type(value.lane_span())

        # Distributed.  Two dimensions multiply into one vector length: the
        # lane axis (how many lanes the dimension is spread over) and the slot
        # axis (`ScalarType.length`, how many consecutive elements one lane
        # holds).  `LaneAxis`'s own documentation keeps these apart for
        # precisely this reason -- they are different things that happen to
        # both make the register bigger.
        span = value.lane_span()
        width = span * (t.length or 1)
        return self.simd_type(t.base.ctype(), width)

    def mask_type(self, width: int) -> str:
        lex = self._lexic()
        get = getattr(lex, 'get_simd_mask', None) if lex is not None else None
        if get is None:
            raise IRError('the ESIMD emitter needs a lexic with get_simd_mask()')
        return get(width)

    def simd_type(self, elem: str, width: int) -> str:
        lex = self._lexic()
        get = getattr(lex, 'get_simd', None) if lex is not None else None
        if get is None:
            raise IRError('the ESIMD emitter needs a lexic with get_simd()')
        return get(elem, width)

    def initialiser(self, v: Value, name: str, expr: str) -> str:
        """Direct-initialisation, because the broadcast constructor is explicit.

        `simd<float, 16> acc = 0.0f;` does not compile: ESIMD makes the
        broadcast constructor `explicit`, on purpose -- filling a vector from
        a scalar is a decision and not a conversion.  `simd<float, 16>
        acc(0.0f)` says the same thing and is what is meant.

        It comes up because a reduction starts its accumulator at the
        operator's neutral element, which is a literal; the accumulator itself
        is lane-distributed, so the two sides of the `=` genuinely differ in
        shape.
        """
        return f'{self.ctype(v.type, v)} {name}({expr});'

    def _vector_ctype(self, t, relaxed: bool) -> str:
        """Every vector is a `simd` here, whatever its width came from.

        The base emitter asks the lexic for a `sycl::vec`, which is right for
        an SPMD load of four consecutive elements and wrong for anything this
        lowering does with it: a `vec` has no `select`, no `copy_from`, and
        nothing a DPAS fragment is written through.  `ctype` already answers
        this way; the two have to agree, or a fragment is declared one way and
        assigned the other.
        """
        return self.simd_type(t.base.ctype(), t.length)

    # -- addressing -------------------------------------------------------- #

    def _thread_idx(self, axis: str) -> str:
        """There is no lane index.

        One ESIMD work-item *is* the vector; `item.get_local_id(0)` is the
        work-item's position in the ND-range, not a lane, and using it as one
        is how the old `simd_mode` produced kernels that indexed a vector with
        a work-group coordinate.  Anything that still asks for a lane index
        here is asking a question this model does not have -- so it is an
        error and not a substitution.
        """
        if axis == 'x':
            raise IRError(
                'the ESIMD lowering has no lane index: one work-item is the '
                'whole vector, so a value spread over lanes is a simd<> type '
                'and not a subscript. A caller reaching here is still building '
                'an SPMD address.')
        return super()._thread_idx(axis)

    # -- memory ------------------------------------------------------------ #

    def declare(self, v: Value, expr: str, s, name: str = None) -> None:
        """As the base emitter, plus two things a vector declaration needs.

        A folded predicate is a `merge`, not a ternary.  `m ? a : b` on a
        `simd_mask` does not compile -- there is no single bit to test -- and
        where a conversion existed it would pick one arm for all N elements.
        The vector form declares the else-value and merges the then-value in
        under the mask, which is two statements, so it cannot be an
        initialiser expression.

        And the *result* of a masked select is distributed even when both arms
        are replicated: masked lanes keep one value, unmasked lanes take the
        other, so the mask is what introduces the distribution.  Its width
        therefore decides the type, not `v.layout` -- which was computed before
        `if_convert` attached the predicate and cannot know about it.
        """
        pred = getattr(s, 'predicate', None)
        if self._masked(pred, v):
            other = s.attr('other')
            other = (self.operand(other) if other is not None
                     else self.zero(v.type))
            self._merge(v, pred, expr, other, name)
            return
        return self._declare_unpredicated(v, expr, s, name)

    @staticmethod
    def _masked(cond, v: Value) -> bool:
        """Is `cond` a lane mask, and `v` something a merge can hold?"""
        return (isinstance(cond, Value) and cond.layout is not None
                and cond.distributed and isinstance(v.type, ScalarType))

    def _merge(self, v: Value, cond: Value, then_expr: str, other_expr: str,
               name: str = None, arms: tuple = ()) -> None:
        """`v = cond ? then : other`, as the two statements a vector needs.

        A boolean result is a mask and not a vector, so it takes the mask
        algebra instead: `simd_mask` is its own family, with `&`, `|` and `!`
        and no `merge`.  Only over arms that are themselves masks -- there is
        no conversion from a `bool` to a `simd_mask`, so a replicated arm
        would have to be broadcast, and nothing builds one to say how.
        """
        nm = name or self.name(v)
        if v.type.base is Datatype.BOOL:
            if not all(isinstance(a, Value) and a.layout is not None
                       and a.distributed for a in arms) or len(arms) != 2:
                raise IRError(
                    f'{v!r}: a masked select over a boolean needs both arms to '
                    f'be masks. A replicated arm would have to be broadcast '
                    f'into a `simd_mask`, which the API offers no conversion '
                    f'for; build the arm as a mask, or select on the value '
                    f'the boolean guards instead.')
            ty = self.mask_type(cond.lane_span())
            self.writer(f'{ty} {nm} = ({self.operand(cond)} & {then_expr}) | '
                        f'(!{self.operand(cond)} & {other_expr});')
            return
        width = cond.lane_span() * (v.type.length or 1)
        ty = self.simd_type(v.type.base.ctype(), width)
        self.writer(f'{ty} {nm}({other_expr});')
        # Both arms go through the vector type explicitly.  `merge` takes a
        # `simd`, and the then-value is often a *replicated* load -- the
        # mask is what makes the result distributed, not the operand -- so
        # it has to be broadcast rather than left to an implicit
        # conversion the API does not offer.
        self.writer(f'{nm}.merge({ty}({then_expr}), {self.operand(cond)});')

    def _declare_unpredicated(self, v: Value, expr: str, s, name: str = None) -> None:
        """A distributed value is filled by a transfer, not by an initialiser.

        `simd<T, N>` has no constructor taking a `T` lvalue, and `= p[i]` would
        either fail to compile or -- worse, where a conversion exists --
        broadcast one element into all N.  The vector form is a declaration
        followed by `copy_from`, so this splits what the base emitter writes as
        one statement.

        Only for loads: an arithmetic result of `simd` operands is already a
        `simd` and initialises normally.  `Op.LOAD` marks its own statements,
        so the split is keyed on that rather than guessed from the text.
        """
        if (getattr(s, 'op', None) in (Op.LOAD, Op.LOAD_ASYNC)
                and v.layout is not None and v.distributed):
            nm = name or self.name(v)
            ptr = self._as_pointer(expr)
            part = self._valid_width(s, v)
            if part is not None:
                # A full-lane tail: the vector is the whole wave, memory past
                # `valid` is not ours.  Zero, then read the part that is.
                elem = v.type.base.ctype()
                self.writer(f'{self.ctype(v.type, v)} {nm}({self.zero(v.type)});')
                slm = self._slm_load_width(s.args[0], elem, part, ptr)
                if slm is None:
                    tmp = f'{nm}_part'
                    self.writer(f'{self.simd_type(elem, part)} {tmp};')
                    self.writer(f'{tmp}.copy_from({ptr});')
                    slm = tmp
                self.writer(f'{nm}.template select<{part}, 1>(0) = {slm};')
                return
            slm = self._slm_load(s.args[0], v, ptr)
            if slm is not None:
                # An expression rather than the two statements below, because
                # the SLM read *returns* the vector: there is nothing to fill
                # in place, so the declaration takes its initialiser.
                self.writer(f'{self.ctype(v.type, v)} {nm} = {slm};')
                return
            self.writer(f'{self.ctype(v.type, v)} {nm};')
            self.writer(f'{nm}.copy_from({ptr});')
            return
        super().declare(v, expr, s, name)

    @staticmethod
    def _as_pointer(access: str) -> str:
        """`p[i]` -> `p + (i)`.

        `copy_from` takes the address of the first element, and the base
        emitter has already built the subscript.  Rewriting it here rather
        than teaching `Op.LOAD` to hand out both forms keeps the address
        arithmetic in one place -- it is the same expression either way, and
        two builders of it would drift.
        """
        m = _VECTOR_ACCESS.match(access.strip())
        if m:
            # A vector-width access arrives already wrapped in a reinterpret
            # cast, and the cast has to come off first: splitting
            # `*(simd<float,4>*)&p[i]` at the subscript alone yields
            # `*(simd<float,4>*)&p + (i)`, which dereferences the pointer and
            # then adds the index to the *value*.  Ill-formed, which is the
            # good case; a shape that compiled would have read the wrong
            # address.  `copy_from` takes the width from the `simd` it fills,
            # so the cast carries nothing here.
            access = m.group(1)
        if access.endswith(']') and '[' in access:
            base, _, idx = access[:-1].partition('[')
            return f'{base} + ({idx})'
        return f'&{access}'

    # -- shared memory ----------------------------------------------------- #

    @staticmethod
    def _is_shared(base) -> bool:
        """Both ways a base names its space, because both still occur.

        A migrated access has the buffer as a `Value` and reads the space off
        its type.  One that has not migrated has a `Symbol`, whose `stype`
        says the same thing in the macro layer's vocabulary -- and 35 of the
        39 vector reads in a plain GEMM are still of the second kind, so
        answering only for the first is answering for almost none of them.
        """
        t = getattr(base, 'type', None)
        if isinstance(t, BufferType):
            return t.space is MemSpace.SHARED
        stype = getattr(base, 'stype', None)
        return MemSpace.from_symbol_type(stype) is MemSpace.SHARED

    def _vector_width(self, v: Value) -> int:
        return v.lane_span() * (v.type.length or 1)

    def _slm_load(self, base, v: Value, address: str):
        """A vector read of a staged tile, where the target has its own one.

        None for global and register, which is every other base: there a
        vector read is `copy_from` on an address and nothing about the space
        needs saying.  Shared memory on this target is not addressable that
        way at all -- `copy_from` on a `SlmPtr` does not compile, and on a raw
        pointer into the arena it would compile and read global memory, which
        is the reason this asks rather than assumes.
        """
        lex = self._lexic()
        if lex is None or not self._is_shared(base):
            return None
        return lex.get_slm_load(v.type.base.ctype(), self._vector_width(v),
                                address)

    def _valid_width(self, s, v: Value):
        """Elements of `v` a full-lane tail's access may touch, when fewer
        than `v` spans (`LeadIndex.valid`, carried as the `valid` attribute).
        """
        valid = s.attr('valid') if getattr(s, 'op', None) in (Op.LOAD, Op.STORE) else None
        if valid is None:
            return None
        part = valid * (v.type.length or 1)
        return part if part < self._vector_width(v) else None

    def _slm_load_width(self, base, elem: str, width: int, address: str):
        lex = self._lexic()
        if lex is None or not self._is_shared(base):
            return None
        return lex.get_slm_load(elem, width, address)

    def _slm_store(self, base, v: Value, address: str, value: str):
        lex = self._lexic()
        if lex is None or not self._is_shared(base):
            return None
        return lex.get_slm_store(v.type.base.ctype(), self._vector_width(v),
                                 address, value)

    def _emit_stmt(self, s, yield_to) -> None:
        """A distributed value is written back by a transfer too.

        The symmetric case to the load: `p[i] = v` where `v` is a `simd` is
        either ill-formed or a narrowing to one element, and neither is the
        store that was meant.

        An explicit `select` is the same question as a folded predicate and
        needs the same answer.  It arrives by a different route -- the sparse
        path builds one directly rather than letting `if_convert` attach a
        predicate -- and the base emitter spells it `a ? b : c`, which on a
        `simd_mask` does not compile.  Left to `declare` this would not even
        be reached: a single-use select is inlined into its consumer, so the
        ternary lands inside a `copy_to` argument with no declaration to
        override.
        """
        op = getattr(s, 'op', None)
        if (op == Op.ALLOC and s.target
                and self._buf_name(s.target[0]) in self._simd_buffers):
            self._emit_simd_alloc(s)
            return
        if (op in (Op.LOAD, Op.STORE) and s.args and self._is_buffer(s.args[0])
                and self._buf_name(s.args[0]) in self._simd_buffers):
            if op == Op.LOAD:
                self._emit_simd_load(s)
            else:
                self._emit_simd_store(s)
            return
        if op == Op.LOAD and id(s) in self._windows:
            self._emit_window_load(s)
            return
        if (getattr(s, 'op', None) == 'select' and len(s.args) == 3
                and s.target and self._masked(s.args[0], s.target[0])):
            v = s.target[0]
            self._merge(v, s.args[0],
                        self.operand(s.args[1], v.type),
                        self.operand(s.args[2], v.type),
                        arms=(s.args[1], s.args[2]))
            return
        if getattr(s, 'op', None) == Op.EXTRACT:
            # One element of a vector, as the element.  `x[i]` on a `simd` is
            # a `simd_view`, and ESIMD has no operator between a view and a
            # vector: the broadcast operand of every scalar-times-vector FMA
            # did not compile.  The cast is what the base spelling meant.
            v = s.target[0]
            self.declare(v, f'static_cast<{self.ctype(v.type, v)}>('
                            f'{self.operand(s.args[0])}[{s.attr("lane")}])', s)
            return
        if getattr(s, 'op', None) == Op.STORE:
            val = s.args[1]
            if isinstance(val, Value) and val.layout is not None and val.distributed:
                addr = self.address(s.args[0], s.args[2:])
                ptr = self._as_pointer(f'{self.base_name(s.args[0])}[{addr}]')
                part = self._valid_width(s, val)
                if part is not None:
                    elem = val.type.base.ctype()
                    whole = self.simd_type(elem, self._vector_width(val))
                    narrow = (f'{self.simd_type(elem, part)}({whole}('
                              f'{self.operand(val)}).template select<{part}, 1>(0))')
                    lex = self._lexic()
                    slm = (lex.get_slm_store(elem, part, ptr, narrow)
                           if lex is not None and self._is_shared(s.args[0])
                           else None)
                    self.writer(slm if slm is not None
                                else f'{narrow}.copy_to({ptr});')
                    return
                slm = self._slm_store(s.args[0], val, ptr,
                                      self.operand(val))
                if slm is not None:
                    self.writer(slm)
                    return
                self.writer(f'{self.operand(val)}.copy_to({ptr});')
                return
        super()._emit_stmt(s, yield_to)

    def _emit_if(self, s) -> None:
        """A lane-varying guard is a mask, and a mask is not a branch condition.

        `if (m)` where `m` is a `simd_mask<N>` is not a slow branch -- there is
        no single bit to test, and the whole work-item would take one arm for
        all N elements.  What the guard means is that *some* lanes are
        excluded, which in this model is a property of each statement inside
        rather than of the region.

        `passes.if_convert` is exactly that transformation and already exists;
        it is documented as not being in the default pipeline because nothing
        yet used the freedom it buys.  This lowering does: for an explicitly
        vectorised kernel the conversion is not an optimisation but the only
        legal lowering, so reaching here means it did not run or could not
        convert this guard -- `_convertible` refuses regions containing
        barriers, nested regions, or raw declarations.
        """
        cond = s.cond
        if isinstance(cond, Value) and cond.layout is not None and cond.distributed:
            raise IRError(
                f'branch on a lane-varying condition ({cond!r}): in the ESIMD '
                f'lowering this is a mask over {cond.lane_span()} lanes, not a '
                f'branch. It has to be if-converted into per-statement '
                f'predicates before emission; see passes.if_convert and its '
                f'`_convertible` preconditions.')
        super()._emit_if(s)

    # -- register buffers -------------------------------------------------- #

    _SPACES = {'Global': MemSpace.GLOBAL, 'SharedMem': MemSpace.SHARED,
               'Register': MemSpace.REGISTER}

    #: What a `simd` holds here.  `__float128` is not a device type on pvc at
    #: all, and its literal needs a GNU suffix; its arrays stay as they were.
    _VECTOR_ELEMS = (Datatype.F32, Datatype.F64)

    @staticmethod
    def _width(v) -> int:
        """Elements a value spans: its slots, times its lanes if spread."""
        if not isinstance(v, Value) or not isinstance(v.type, ScalarType):
            return 1
        n = v.type.length or 1
        if v.layout is not None and v.distributed:
            n *= v.lane_span()
        return n

    @staticmethod
    def _is_buffer(x) -> bool:
        return ((isinstance(x, Value) and isinstance(x.type, BufferType))
                or hasattr(x, 'stype'))

    def _buf_name(self, x) -> str:
        """The name a buffer is spelled by.  Accesses reach one either as the
        `Value` its `Op.ALLOC` defined or as the macro layer's `Symbol`, and
        only the name says they are the same buffer."""
        if isinstance(x, Value):
            return self._alloc_names.get(x.id, str(x))
        return getattr(x, 'name', str(x))

    def _space(self, x):
        if isinstance(x, Value) and isinstance(x.type, BufferType):
            return x.type.space
        stype = getattr(x, 'stype', None)
        return self._SPACES.get(getattr(stype, 'name', None), MemSpace.UNKNOWN)

    @staticmethod
    def _comment(x) -> bool:
        """A raw statement that is only a comment: the macro layer writes the
        instruction it lowers above it (`// r1 = +(r0 * s0) + None`), which
        names every buffer and touches none."""
        text = (x.text or '').strip()
        return x.op == Op.RAWSTMT and bool(text) and all(
            line.strip().startswith('//') for line in text.splitlines())

    def _plan_allocs(self, stmts) -> dict:
        """name -> the `Value` its `Op.ALLOC` defines."""
        allocs = {}
        self._alloc_names = {}
        for a in stmts:
            if a.op == Op.ALLOC and a.target and isinstance(a.target[0].type, BufferType):
                v = a.target[0]
                name = a.attr('extern') or str(v)
                self._alloc_names[v.id] = name
                allocs[name] = (a, v)
        return allocs

    def _plan_register_buffers(self, stmts, allocs, consts=None, order=None) -> set:
        """Register arrays that can be one `simd` instead of an array.

        A thread-private array is memory to the compiler: every transfer in
        and out of it is a `copy_from`/`copy_to` through a pointer, and what
        IGC cannot promote it keeps in scratch.  Here the array fits in the
        type -- `simd<float, 576>`, read and written through `select` -- and
        is registers by construction.  `local_flux` on pvc: 33088 B of spill
        against 26048 B, 43934 instructions against 23769.

        Only where every access is the base of an unpredicated load or store:
        a buffer handed on, carried by a loop, named in raw text or accessed
        under a predicate keeps its array.  And only where no access with a
        known offset runs past the end: `select` checks its range where the
        array would have read the next variable.
        """
        consts = consts or {}
        cands = {n for n, (a, v) in allocs.items()
                 if v.type.space == MemSpace.REGISTER and a.attr('arena') is None
                 and a.attr('init') in (None, '', '{}')
                 and v.type.elem in self._VECTOR_ELEMS}
        bad = set()
        for x in stmts:
            if x.op in Op.RAW:
                if self._comment(x):
                    continue
                bad |= {n for n in cands
                        if re.search(rf'\b{re.escape(n)}\b', x.text or '')}
                continue
            for i, arg in enumerate(x.args):
                if not self._is_buffer(arg):
                    continue
                n = self._buf_name(arg)
                if n not in cands:
                    continue
                if not (x.op in (Op.LOAD, Op.STORE) and i == 0
                        and x.predicate is None
                        and (x.op == Op.STORE
                             or isinstance(x.target[0].type, ScalarType))):
                    bad.add(n)
                    continue
                w = self._width(x.target[0] if x.op == Op.LOAD else x.args[1])
                idx = x.args[1:] if x.op == Op.LOAD else x.args[2:]
                flat = self._const_flat(arg, idx, consts)
                if flat is not None and not 0 <= flat <= allocs[n][1].type.volume - w:
                    bad.add(n)
        if bad:
            return set()
        return self._within_budget(order or [], cands, allocs)

    def _within_budget(self, order, names, allocs) -> set:
        """As many of `names` as fit in a thread's registers at once.

        A `simd` is registers whatever its size, where an array too big to
        promote goes to scratch whole and is at least read in blocks.  The
        arrays are taken smallest first, each only while every point of the
        body keeps the ones taken and live there within `max_reg_per_thread`
        (live: first access to last, in program order) -- and then all of them
        or none.  Measured on pvc, with the shared-memory windows in both
        columns:

            local_flux             all 15 fit    1344 B -> 0 B spill
            chain_three_matrices   one of 14 kB  1984 B -> 3392 B
            chain_five_multiplies  two of 14 kB  17536 B -> 17600 B

        Next to an array IGC has to keep in scratch, the small ones did better
        as arrays too.
        """
        hw = self._hw()
        budget = getattr(hw, 'max_reg_per_thread', None) or 8192
        span = {}
        for pos, x in enumerate(order):
            for arg in x.args:
                if self._is_buffer(arg):
                    n = self._buf_name(arg)
                    if n in names:
                        a, b = span.get(n, (pos, pos))
                        span[n] = (min(a, pos), max(b, pos))
        load = [0] * (len(order) + 1)
        taken = set()
        for n in sorted(names, key=lambda n: (allocs[n][1].type.volume, n)):
            v = allocs[n][1]
            size = v.type.volume * v.type.elem.size()
            a, b = span.get(n, (0, -1))
            if b < a:
                taken.add(n)
                continue
            if max(load[a:b + 1]) + size > budget:
                return set()
            for i in range(a, b + 1):
                load[i] += size
            taken.add(n)
        return taken

    def _emit_simd_alloc(self, s) -> None:
        v = s.target[0]
        extern = s.attr('extern')
        if extern is not None:
            self.bind(v, extern)
        init = f'({v.type.elem.literal(0)})' if s.attr('init') == '{}' else ''
        self.writer(f'{self.simd_type(v.type.elem.ctype(), v.type.volume)} '
                    f'{self.name(v)}{init};')

    def _emit_simd_load(self, s) -> None:
        v, buf = s.target[0], s.args[0]
        addr = self.address(buf, s.args[1:])
        w = self._width(v)
        named = s.attr('extern')
        nm = named or self.name(v)
        if w == 1:
            self.writer(f'{self.ctype(v.type, v)} {nm} = {self.base_name(buf)}[{addr}];')
        else:
            self.writer(f'{self.ctype(v.type, v)} {nm}({self.base_name(buf)}'
                        f'.template select<{w}, 1>({addr}));')
        if named:
            self.bind(v, named)

    def _emit_simd_store(self, s) -> None:
        buf, val = s.args[0], s.args[1]
        addr = self.address(buf, s.args[2:])
        w = self._width(val)
        if w == 1:
            dt = self.elem_type(buf)
            rhs = self.operand(val, ScalarType(dt) if dt is not None else None)
            self.writer(f'{self.base_name(buf)}[{addr}] = {rhs};')
        else:
            self.writer(f'{self.base_name(buf)}.template select<{w}, 1>({addr}) = '
                        f'{self.operand(val)};')

    # -- shared-memory windows ---------------------------------------------- #

    #: The widest window, in elements: 64 registers of 64 bytes.
    WINDOW_MAX = 1024

    def _plan_windows(self, body, allocs) -> None:
        """Scalar reads of shared memory, served from one block read.

        A broadcast operand is read one element at a time -- `float b =
        s0[k]` -- and under SPMD that is one load shared by the lanes.  Here
        one work-item is the whole vector and each read is a message of its
        own: 1836 SLM reads in `local_flux`, next to 727 global ones.  A run
        of them with constant addresses and nothing in between that could
        change the buffer is one `copy_from` into a `simd`, and the reads
        become register elements.

        What may sit between two reads of a window: anything that does not
        write shared memory or synchronise.  Any shared store ends every
        window, not just the stored buffer's -- the arena reuses offsets, and
        a store through another name can land on the same bytes.  A window
        reaches into a nested region only if nothing in that region ends it,
        so a loop cannot read a window that its own last iteration made stale.
        """
        shared = {n: v for n, (a, v) in allocs.items()
                  if v.type.space == MemSpace.SHARED
                  and v.type.elem in self._VECTOR_ELEMS
                  and getattr(v.type, 'swizzle', None) is None}
        consts = {}
        seq = []

        def lin(stmts, depth):
            for x in stmts:
                if x.op == Op.CONST and x.target:
                    consts[x.target[0].id] = x.attr('value')
                seq.append((x, depth))
                for r in x.regions:
                    lin(r.body, depth + 1)
        lin(body, 0)

        def ends(x) -> bool:
            if self._comment(x):
                return False
            if x.op in Op.RAW or x.op in (Op.BARRIER, Op.WAIT, Op.CALL,
                                          Op.COMMIT_ASYNC, Op.COPY_ASYNC,
                                          Op.ACCUM):
                return True
            if x.op == Op.STORE:
                b = x.args[0] if x.args else None
                return self._space(b) not in (MemSpace.REGISTER, MemSpace.GLOBAL)
            return (not x.pure and not x.regions and x.op not in (
                Op.LOAD, Op.LOAD_ASYNC, Op.PREFETCH, Op.ALLOC, Op.DECLARE,
                Op.YIELD, Op.EXIT, Op.CONST))

        memo = {}

        def ends_inside(x) -> bool:
            if id(x) not in memo:
                memo[id(x)] = any(ends(y) for y in walk_stmts(
                    tuple(z for r in x.regions for z in r.body)))
            return memo[id(x)]

        open_ = {}
        count = [0]

        def close(name):
            st = open_.pop(name, None)
            if st is None or len(st['loads']) < 2:
                return
            vol = shared[name].type.volume
            flats = [f for _, f in st['loads']]
            lo, hi = min(flats), max(flats)
            width = hi - lo + 1
            if width > self.WINDOW_MAX or width > 8 * len(flats):
                return
            width = min(-(-width // 16) * 16, vol - lo)
            wname = f'{name}_w{count[0]}'
            count[0] += 1
            first = st['loads'][0][0]
            self._window_heads[id(first)] = (first.args[0], lo, width, wname,
                                             shared[name].type.elem.ctype())
            for x, f in st['loads']:
                self._windows[id(x)] = (wname, f - lo)

        for x, depth in seq:
            for n in [n for n, st in open_.items() if depth < st['depth']]:
                close(n)
            if ends(x) or (x.regions and ends_inside(x)):
                for n in list(open_):
                    close(n)
                continue
            if not (x.op == Op.LOAD and x.predicate is None and x.target
                    and x.args and self._is_buffer(x.args[0])):
                continue
            buf, v = x.args[0], x.target[0]
            name = self._buf_name(buf)
            if (name not in shared or x.attr('nontemporal')
                    or not isinstance(v.type, ScalarType) or v.type.length
                    or v.layout is None or v.distributed):
                continue
            flat = self._const_flat(buf, x.args[1:], consts)
            if flat is None or not 0 <= flat < shared[name].type.volume:
                continue
            open_.setdefault(name, {'depth': depth, 'loads': []})['loads'].append((x, flat))
        for n in list(open_):
            close(n)

    @staticmethod
    def _const_flat(buf, indices, consts):
        """The element `address` would compute, if every index is known."""
        nums = []
        for i in indices:
            n = consts.get(i.id) if isinstance(i, Value) else i
            if isinstance(n, bool) or not isinstance(n, int):
                return None
            nums.append(n)
        if not nums:
            return 0
        if isinstance(buf, Value) and isinstance(buf.type, BufferType):
            shape = buf.type.shape
        else:
            view = getattr(buf, 'data_view', None)
            shape = tuple(view.shape) if view is not None else None
        if shape is None or len(shape) != len(nums):
            return sum(nums)
        flat = nums[-1]
        for k in reversed(range(len(nums) - 1)):
            flat = nums[k] + shape[k] * flat
        return flat

    def _emit_window_load(self, s) -> None:
        head = self._window_heads.get(id(s))
        if head is not None:
            buf, lo, width, wname, elem = head
            start = f'{self.base_name(buf)} + {lo}'
            lex = self._lexic()
            # Through the target's own read of shared memory: a window is an
            # offset into SLM, and `copy_from` on it does not compile (or, on
            # a raw pointer, reads global memory).
            slm = lex.get_slm_load(elem, width, start) if lex is not None else None
            if slm is not None:
                self.writer(f'{self.simd_type(elem, width)} {wname} = {slm};')
            else:
                self.writer(f'{self.simd_type(elem, width)} {wname};')
                self.writer(f'{wname}.copy_from({start});')
        wname, k = self._windows[id(s)]
        v = s.target[0]
        named = s.attr('extern')
        self.declare(v, f'{wname}[{k}]', s, name=named)
        if named:
            self.bind(v, named)

    # -- entry ------------------------------------------------------------- #

    def run(self, body) -> None:
        stmts = walk_stmts(body)
        allocs = self._plan_allocs(stmts)
        consts = {x.target[0].id: x.attr('value') for x in stmts
                  if x.op == Op.CONST and x.target}
        order = []

        def lin(xs):
            for x in xs:
                order.append(x)
                for r in x.regions:
                    lin(r.body)
        lin(body)
        self._simd_buffers = self._plan_register_buffers(stmts, allocs, consts, order)
        self._windows, self._window_heads = {}, {}
        self._plan_windows(body, allocs)
        super().run(body)
        if self.unresolved and self.strict:
            names = ', '.join(repr(v) for v in self.unresolved[:8])
            more = ('' if len(self.unresolved) <= 8
                    else f' (and {len(self.unresolved) - 8} more)')
            raise IRError(
                f'{len(self.unresolved)} value(s) have no tracked '
                f'distribution and cannot be given an ESIMD type: {names}'
                f'{more}. Every declaration needs to know how the value is '
                f'spread over the lanes; in the SPMD lowering that is carried '
                f'by the index expression instead, which is why these got '
                f'this far untracked.')
