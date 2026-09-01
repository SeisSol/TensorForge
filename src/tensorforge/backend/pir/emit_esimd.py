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

from .core import IRError, Op, ScalarType, TokenType, Value
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
        if (pred is not None and _folds_predicate(s)
                and isinstance(pred, Value) and pred.layout is not None
                and pred.distributed and isinstance(v.type, ScalarType)):
            width = pred.lane_span() * (v.type.length or 1)
            ty = self.simd_type(v.type.base.ctype(), width)
            other = s.attr('other')
            other = (self.operand(other) if other is not None
                     else self.zero(v.type))
            nm = name or self.name(v)
            self.writer(f'{ty} {nm}({other});')
            # Both arms go through the vector type explicitly.  `merge` takes a
            # `simd`, and the then-value is often a *replicated* load -- the
            # mask is what makes the result distributed, not the operand -- so
            # it has to be broadcast rather than left to an implicit
            # conversion the API does not offer.
            self.writer(f'{nm}.merge({ty}({expr}), {self.operand(pred)});')
            return
        return self._declare_unpredicated(v, expr, s, name)

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
            self.writer(f'{self.ctype(v.type, v)} {nm};')
            self.writer(f'{nm}.copy_from({self._as_pointer(expr)});')
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

    def _emit_stmt(self, s, yield_to) -> None:
        """A distributed value is written back by a transfer too.

        The symmetric case to the load: `p[i] = v` where `v` is a `simd` is
        either ill-formed or a narrowing to one element, and neither is the
        store that was meant.
        """
        if getattr(s, 'op', None) == Op.STORE:
            val = s.args[1]
            if isinstance(val, Value) and val.layout is not None and val.distributed:
                addr = self.address(s.args[0], s.args[2:])
                ptr = self._as_pointer(f'{self.base_name(s.args[0])}[{addr}]')
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

    # -- entry ------------------------------------------------------------- #

    def run(self, body) -> None:
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
