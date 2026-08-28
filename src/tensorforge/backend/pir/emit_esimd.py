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

from typing import Any, Optional

from .core import IRError, Op, ScalarType, TokenType, Value
from .emit import Emitter


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

        if not value.distributed:
            # Replicated: every lane holds the whole thing, which is exactly
            # what a scalar is.  Note this is *not* the same answer as the
            # untracked case above even though the spelling matches -- one is
            # derived, the other is a hole.
            return super().ctype(t, value)

        # Distributed.  Two dimensions multiply into one vector length: the
        # lane axis (how many lanes the dimension is spread over) and the slot
        # axis (`ScalarType.length`, how many consecutive elements one lane
        # holds).  `LaneAxis`'s own documentation keeps these apart for
        # precisely this reason -- they are different things that happen to
        # both make the register bigger.
        span = value.lane_span()
        width = span * (t.length or 1)
        return self.simd_type(t.base.ctype(), width)

    def simd_type(self, elem: str, width: int) -> str:
        lex = self._lexic()
        get = getattr(lex, 'get_simd', None) if lex is not None else None
        if get is None:
            raise IRError('the ESIMD emitter needs a lexic with get_simd()')
        return get(elem, width)

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
