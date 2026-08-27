# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""The operator contract, checked on the host.

Two things had gone wrong here and neither could fail a test, because neither
had one.

`Operator` was a plain class while its methods carried `@abstractmethod`.  The
decorator only sets `__isabstractmethod__`; without `ABCMeta` nothing reads it.
So the seven concrete reduction operators were each marked abstract *on their
own implementation* of `format` -- a contradiction that instantiated fine and
would have turned into a `TypeError` on every case file the moment anyone gave
the base class a metaclass.

`neutral()` answered without being told the type.  For `add` and `mul` that is
fine.  For everything else it is not: `AndOperator` returned `True`, which is
the identity for one bit and clears every bit above it on anything wider, and
`Min`/`Max` returned infinities that no integer type can hold.  A reduction
kernel starting from the wrong identity produces a plausible number, which is
the kind of defect worth a cheap test.

Neither needs a GPU, a toolchain, or a generated kernel.
"""

from __future__ import annotations

import math

import pytest

from tensorforge.common.basic_types import Datatype
from tensorforge.common.operation import (AddOperator, AndOperator,
                                          MaxOperator, MinOperator,
                                          MulOperator, Operator, OrOperator,
                                          ReductionOperator, XorOperator)

REDUCTIONS = [AddOperator, MulOperator, MinOperator, MaxOperator,
              AndOperator, OrOperator, XorOperator]

INTS = [Datatype.I8, Datatype.I16, Datatype.I32, Datatype.I64]
FLOATS = [Datatype.F16, Datatype.BF16, Datatype.F32, Datatype.F64,
          Datatype.F128]


# --- abstractness is real ------------------------------------------------- #

def test_operator_is_abstract():
    """`Operator` cannot be instantiated, which is what makes the rest hold."""
    with pytest.raises(TypeError):
        Operator()


def test_reduction_operator_is_abstract():
    with pytest.raises(TypeError):
        ReductionOperator()


@pytest.mark.parametrize('cls', REDUCTIONS, ids=lambda c: c.__name__)
def test_concrete_reduction_operators_instantiate(cls):
    """Every concrete operator satisfies the contract it inherits.

    This is the test the stray decorators would have failed: a concrete class
    that marks its own `format` abstract is not instantiable under `ABCMeta`.
    """
    op = cls()
    assert op.num_operands() == 2
    assert op.format('a', 'b')
    assert op.datatype()


@pytest.mark.parametrize('cls', REDUCTIONS, ids=lambda c: c.__name__)
def test_no_concrete_method_is_marked_abstract(cls):
    """Guards the specific mistake rather than only its symptom.

    `instantiate` above catches it too, but only while `ABCMeta` is in place.
    This says the decorator itself does not belong on an implementation, which
    stays true whatever the metaclass does later.
    """
    for name in ('format', 'neutral', 'datatype', 'num_operands', 'irop'):
        method = getattr(cls, name, None)
        if method is None:
            continue
        assert not getattr(method, '__isabstractmethod__', False), (
            f'{cls.__name__}.{name} implements the method and marks it '
            'abstract at the same time')


# --- neutral elements ----------------------------------------------------- #

@pytest.mark.parametrize('cls', REDUCTIONS, ids=lambda c: c.__name__)
@pytest.mark.parametrize('dtype', INTS + FLOATS + [Datatype.BOOL],
                         ids=str)
def test_neutral_is_answered_for_every_type(cls, dtype):
    """No type is left without an identity, and none of them is None."""
    assert cls().neutral(dtype) is not None


@pytest.mark.parametrize('dtype', INTS, ids=str)
def test_min_max_neutral_is_representable_in_integers(dtype):
    """The bound has to be the type's own, not an infinity standing in for it.

    An `I32` reduction seeded with `inf` is either a compile error or a
    silently truncated literal, depending on how far down the value gets.
    """
    lo = MaxOperator().neutral(dtype)
    hi = MinOperator().neutral(dtype)
    width = 8 * dtype.size()

    assert lo == -(1 << (width - 1))
    assert hi == (1 << (width - 1)) - 1
    assert not math.isinf(lo) and not math.isinf(hi)


@pytest.mark.parametrize('dtype', FLOATS, ids=str)
def test_min_max_neutral_is_infinite_in_floats(dtype):
    """`lowest()`, not `min()`.

    The distinction the C++ side got wrong: the smallest *positive normal*
    float is a perfectly good-looking seed that makes `max` over negative data
    return roughly zero.
    """
    assert MaxOperator().neutral(dtype) == -math.inf
    assert MinOperator().neutral(dtype) == math.inf


@pytest.mark.parametrize('dtype', INTS, ids=str)
def test_bitwise_and_neutral_is_all_ones(dtype):
    """`-1`, not `True`.

    Masked to the type's width it has to leave every bit set; `True` leaves
    exactly one.
    """
    neutral = AndOperator().neutral(dtype)
    width = 8 * dtype.size()
    mask = (1 << width) - 1
    assert neutral & mask == mask


def test_bitwise_and_neutral_is_true_for_bool():
    assert AndOperator().neutral(Datatype.BOOL) is True


@pytest.mark.parametrize('cls', [OrOperator, XorOperator],
                         ids=lambda c: c.__name__)
@pytest.mark.parametrize('dtype', INTS, ids=str)
def test_or_xor_neutral_is_zero(cls, dtype):
    assert cls().neutral(dtype) == 0


# --- the identity law itself ---------------------------------------------- #

_APPLY = {
    AddOperator: lambda a, b: a + b,
    MulOperator: lambda a, b: a * b,
    MinOperator: min,
    MaxOperator: max,
    AndOperator: lambda a, b: a & b,
    OrOperator: lambda a, b: a | b,
    XorOperator: lambda a, b: a ^ b,
}


@pytest.mark.parametrize('cls', REDUCTIONS, ids=lambda c: c.__name__)
@pytest.mark.parametrize('dtype', INTS, ids=str)
def test_neutral_is_an_identity_on_integers(cls, dtype):
    """`op(neutral, x) == x` for values the type can actually hold.

    Python's ints do not wrap, so the check runs against the width's mask --
    which is what the generated code computes in.
    """
    width = 8 * dtype.size()
    mask = (1 << width) - 1
    neutral = int(cls().neutral(dtype))
    apply = _APPLY[cls]

    for x in (0, 1, 7, (1 << (width - 1)) - 1, -(1 << (width - 1)), -1):
        assert apply(neutral, x) & mask == x & mask, (
            f'{cls.__name__}: {neutral} is not an identity for {x} in {dtype}')


@pytest.mark.parametrize('cls', [AddOperator, MulOperator, MinOperator,
                                 MaxOperator], ids=lambda c: c.__name__)
def test_neutral_is_an_identity_on_floats(cls):
    neutral = cls().neutral(Datatype.F64)
    apply = _APPLY[cls]
    for x in (0.0, 1.0, -1.0, 1e30, -1e30, 1e-30):
        assert apply(neutral, x) == x, (
            f'{cls.__name__}: {neutral} is not an identity for {x}')
