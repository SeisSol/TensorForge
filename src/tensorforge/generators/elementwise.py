# SPDX-FileCopyrightText: 2015 SeisSol Group
#
# SPDX-License-Identifier: MIT

"""Builders for :class:`ElementwiseDescr`.

Each produces a descriptor directly, so there is no node hierarchy in between.
An identity that makes an operation another one -- ``x * 1``, ``x ** 2`` -- is
applied where the descriptors of every frontend pass (`generators.legalize`).

``op(dest, *srcs)`` throughout, i.e. destination first, matching assignment
order.
"""

from __future__ import annotations

from typing import Union

from tensorforge.common.operation import Operation
from tensorforge.generators.descriptions import ElementwiseDescr

Operand = Union[object, int, float]


def _ew(op: Operation, dest, *srcs, **kw) -> ElementwiseDescr:
    return ElementwiseDescr(op, dest, list(srcs), **kw)


# --------------------------------------------------------------------------- #
# Unary
# --------------------------------------------------------------------------- #

_UNARY = ('abs acos acosh asin asinh atan atanh cbrt ceil cos cosh erf exp '
          'expm1 floor gamma log log1p neg rcbrt rcp round rsqrt sign sin '
          'sinh sqrt tan tanh trunc copy').split()

# --------------------------------------------------------------------------- #
# Binary
# --------------------------------------------------------------------------- #

_BINARY = ('add sub mul div pow mod max min and or xor shl shr shrs '
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


__all__ = _UNARY + _BINARY
