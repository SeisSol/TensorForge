# SPDX-FileCopyrightText: 2015 SeisSol Group
#
# SPDX-License-Identifier: MIT

"""Builders for :class:`ElementwiseDescr`.

Each produces a descriptor directly, so there is no node hierarchy in between.

The algebraic simplifications of ``mul`` / ``div`` / ``pow`` are kept *here*
rather than in the instruction: they rewrite one operation into another before
anything is built, which is a frontend concern.  A fold over the macro stream,
as ``pir.passes.fold`` is one over the IR, would be their home, and these
could go.

``op(dest, *srcs)`` throughout, i.e. destination first, matching assignment
order.
"""

from __future__ import annotations

import math
from typing import Union

from tensorforge.common.operation import Operation
from tensorforge.generators.descriptions import ElementwiseDescr

Operand = Union[object, int, float]


def _ew(op: Operation, dest, *srcs, **kw) -> ElementwiseDescr:
    return ElementwiseDescr(op, dest, list(srcs), **kw)


def _is_num(x) -> bool:
    return isinstance(x, (int, float))


# --------------------------------------------------------------------------- #
# Unary
# --------------------------------------------------------------------------- #

_UNARY = ('abs acos acosh asin asinh atan atanh cbrt ceil cos cosh erf exp '
          'expm1 floor gamma log log1p neg rcbrt rcp round rsqrt sign sin '
          'sinh sqrt tan tanh trunc copy').split()

# --------------------------------------------------------------------------- #
# Binary
# --------------------------------------------------------------------------- #

_BINARY = ('add sub mod max min and or xor shl shr shrs '
           'eq neq lt le gt ge').split()


def _make(name: str, arity: int):
    op = Operation[name.upper()]
    if arity == 1:
        def helper(dest, x, **kw):
            return _ew(op, dest, x, **kw)
    else:
        def helper(dest, x, y, **kw):
            return _ew(op, dest, x, y, **kw)
    helper.__name__ = name
    helper.__qualname__ = name
    helper.__doc__ = f'``dest = {name}(...)``'
    return helper


for _n in _UNARY:
    globals()[_n] = _make(_n, 1)
for _n in _BINARY:
    globals()[_n] = _make(_n, 2)


# --------------------------------------------------------------------------- #
# The three with algebraic simplifications
# --------------------------------------------------------------------------- #

def mul(dest, x, y, **kw) -> ElementwiseDescr:
    if _is_num(x) and x in (1, 1.0):
        return copy(dest, y, **kw)
    if _is_num(x) and x in (-1, -1.0):
        return neg(dest, y, **kw)
    if _is_num(y) and y in (1, 1.0):
        return copy(dest, x, **kw)
    if _is_num(y) and y in (-1, -1.0):
        return neg(dest, x, **kw)
    return _ew(Operation.MUL, dest, x, y, **kw)


def div(dest, x, y, **kw) -> ElementwiseDescr:
    if _is_num(x) and x in (1, 1.0):
        return rcp(dest, y, **kw)
    if _is_num(y) and y in (1, 1.0):
        return copy(dest, x, **kw)
    if _is_num(y) and y in (-1, -1.0):
        return neg(dest, x, **kw)
    return _ew(Operation.DIV, dest, x, y, **kw)


def pow(dest, x, y, **kw) -> ElementwiseDescr:
    """``dest = x ** y``, with special cases for some exponents.

    ``y == 2`` becomes ``MUL(x, x)``, a single instruction with a repeated
    operand; a shared subexpression in general is what CSE over the macro
    stream is for.
    """
    if _is_num(y):
        if y in (2, 2.0):
            return _ew(Operation.MUL, dest, x, x, **kw)
        if y == 0.5:
            return sqrt(dest, x, **kw)
        if y == -0.5:
            return rsqrt(dest, x, **kw)
        if y == 1 / 3:
            return cbrt(dest, x, **kw)
        if y == -1 / 3:
            return rcbrt(dest, x, **kw)
        if y in (-1, -1.0):
            return rcp(dest, x, **kw)
        if y in (1, 1.0):
            return copy(dest, x, **kw)
    if _is_num(x):
        if x == math.e:
            return exp(dest, y, **kw)
        if x in (1, 1.0):
            return copy(dest, x, **kw)
    return _ew(Operation.POW, dest, x, y, **kw)


__all__ = _UNARY + _BINARY + ['mul', 'div', 'pow']
