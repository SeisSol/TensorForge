# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""An elementwise operation as one typed statement, spelled by its target.

`IRBuilder.math` records which `common.operation.Operation` a value is and
what it reads; at emission the lexic spells it -- its library's functions
from `Lexic.MATH`, or `Lexic.INTEGER_MATH` for an integer, the operators
every C-like language shares from `lexic.INFIX`.  What is pinned here: the
statement is built and checked like the arithmetic, it comes out as the
target's call with a number in it spelled in the kernel's type, the
elementwise instruction writes nothing else for its operation, and the
tables agree on which function an operation is -- one table per language is
one place per language for a misspelt name to hide.
"""

from __future__ import annotations

import re
from dataclasses import replace

import pytest

from tensorforge.backend.pir import verify
from tensorforge.backend.pir.build import IRBuilder
from tensorforge.backend.pir.core import IRError, Op, ScalarType
from tensorforge.backend.pir.emit import Emitter
from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.context import Context
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.common.operation import Operation
from tensorforge.common.target import Target
from tensorforge.common.vm.lexic.cuda_lexic import CudaLexic
from tensorforge.common.vm.lexic.lexic import INFIX
from tensorforge.common.vm.lexic.sycl_lexic import SyclLexic
from tensorforge.common.vm.lexic.target_lexic import TargetLexic
from tensorforge.generators import elementwise as ew
from tensorforge.generators.generator import Generator

F32 = ScalarType(Datatype.F32)
F64 = ScalarType(Datatype.F64)
I32 = ScalarType(Datatype.I32)
I64 = ScalarType(Datatype.I64)

#: One target per lexic.
CUDA = ('sm_86', 'cuda')
HIP = ('gfx942', 'hip')
SYCL = ('pvc', 'oneapi')
ESIMD = ('pvc', 'esimd')
OPENMP = ('sm_86', 'omptarget')


def _emitted(target, op, type_, *operands):
    """The declaration `op` over `operands` comes out as.  `None` stands for
    a value from outside named `x`; a number is itself."""
    b = IRBuilder(fptype=type_.base)
    x = b.extern_value('x', type_, hint='x')
    b.math(op, type_, *(x if o is None else o for o in operands), hint='e')
    lines = []
    Emitter(lines.append, target).run(b.finish())
    return lines[-1]


# --------------------------------------------------------------------------- #
# The statement
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('count', [0, 3])
def test_one_or_two_operands(count):
    b = IRBuilder(fptype=Datatype.F32)
    x = b.extern_value('x', F32, hint='x')
    with pytest.raises(IRError, match='one or two operands'):
        b.math(Operation.ADD, F32, *[x] * count)


def test_the_same_operation_on_the_same_operands_is_one_value():
    b = IRBuilder(fptype=Datatype.F32)
    x = b.extern_value('x', F32, hint='x')
    e = b.math(Operation.EXP, F32, x)
    assert b.math(Operation.EXP, F32, x) is e
    assert b.math(Operation.LOG, F32, x) is not e
    assert b.math(Operation.POW, F32, x, 2.0) is not b.math(
        Operation.POW, F32, x, 3.0)


def test_a_statement_without_its_operation_is_refused():
    b = IRBuilder(fptype=Datatype.F32)
    x = b.extern_value('x', F32, hint='x')
    b.math(Operation.EXP, F32, x)
    body = b.finish()
    assert not verify(body, strict=False)
    stmt = next(s for s in body if s.op == Op.MATH)
    broken = tuple(replace(s, attrs=()) if s is stmt else s for s in body)
    assert any('no operation' in d for d in verify(broken, strict=False))


# --------------------------------------------------------------------------- #
# Its spelling
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('where,op,type_,operands,expected', [
    (CUDA, Operation.EXP, F32, (None,), 'std::exp(x)'),
    (CUDA, Operation.EXP, F64, (None,), 'std::exp(x)'),
    (HIP, Operation.LOG1P, F32, (None,), 'std::log1p(x)'),
    (CUDA, Operation.POW, F32, (None, 3.0), 'std::pow(x, 3.0f)'),
    (CUDA, Operation.POW, F64, (None, 3.0), 'std::pow(x, 3.0)'),
    (CUDA, Operation.MIN, F32, (None, 0.0), 'std::fmin(x, 0.0f)'),
    (CUDA, Operation.RSQRT, F32, (None,), 'rsqrtf(x)'),
    (CUDA, Operation.RSQRT, F64, (None,), 'rsqrt(x)'),
    (CUDA, Operation.MUL, F32, (2, None), '(2.0f * x)'),
    (CUDA, Operation.RCP, F32, (None,), '(1 / x)'),
    (SYCL, Operation.EXP, F32, (None,), 'sycl::exp(x)'),
    (SYCL, Operation.MAX, F32, (None, 0.0),
     'sycl::fmax(float(x), float(0.0f))'),
    (ESIMD, Operation.EXP, F32, (None,), 'tensorforge::intel_esimd::exp(x)'),
    (ESIMD, Operation.TANH, F64, (None,), 'tensorforge::tanhF64(x)'),
    (OPENMP, Operation.SQRT, F32, (None,), 'std::sqrt(x)'),
    (OPENMP, Operation.ABS, F32, (None,), 'std::fabs(x)'),
    (CUDA, Operation.MIN, I64, (None, 0), 'min(x, 0_i64)'),
    (HIP, Operation.ABS, I32, (None,), 'abs(x)'),
    (SYCL, Operation.MAX, I32, (None, 0),
     'sycl::max(int32_t(x), int32_t(0_i32))'),
    (OPENMP, Operation.MIN, I64, (None, 0),
     'std::min(int64_t(x), int64_t(0_i64))'),
])
def test_the_target_spells_it(where, op, type_, operands, expected):
    """Its library's function, a number in the kernel's type -- `3.0f`, not
    a `double` the call would convert -- and an operator where every C-like
    language has the same one.  An integer's own function where it has one:
    `std::fmin` of two `int64_t` compares two `double`s, and above 2**53 the
    smaller of two integers is no longer one of them."""
    line = _emitted(Target(*where), op, type_, *operands)
    assert line.endswith(f' = {expected};'), line


def test_without_a_lexic_the_operation_is_named():
    """A body built without a target, as the IR's own tests build them."""
    assert _emitted(None, Operation.EXP, F32, None).endswith(' = exp(x);')


def test_what_it_occupies_is_read_off_the_function():
    """A transcendental goes to the special-function unit; an absolute value
    does not."""
    class Counting:
        def __init__(self):
            self.target = Target(*CUDA)
            self.categories = []

        def record_mix(self, category, issued, written):
            self.categories.append(category)

    for op, expected in ((Operation.EXP, 'sfu'), (Operation.ABS, 'fp')):
        counting = Counting()
        b = IRBuilder(fptype=Datatype.F32)
        b.math(op, F32, b.extern_value('x', F32, hint='x'))
        Emitter([].append, counting).run(b.finish())
        assert counting.categories == [expected]


def test_the_elementwise_instruction_writes_the_operation(monkeypatch):
    """The operation reaches the IR as itself, and comes out as its call."""
    built = []
    real = IRBuilder.math

    def math(self, fn, type_, *args, **kw):
        built.append((fn, args))
        return real(self, fn, type_, *args, **kw)

    monkeypatch.setattr(IRBuilder, 'math', math)
    gen = Generator([ew.pow(_tensor('B'), _tensor('A'), 3.0)],
                    Context(arch='sm_86', backend='cuda',
                            fp_type=Datatype.F32))
    gen.generate()
    assert built and all(fn == Operation.POW and args[1] == 3.0
                         for fn, args in built)
    assert 'std::pow(' in gen.get_kernel()


def _tensor(alias):
    return SubTensor(Tensor([16, 16], Addressing.STRIDED,
                            BoundingBox([0, 0], [16, 16]), alias=alias,
                            datatype=Datatype.F32))


@pytest.mark.parametrize('where,root,call', [
    (CUDA, ew.rsqrt, 'rsqrtf('), (HIP, ew.rsqrt, 'rsqrtf('),
    (SYCL, ew.rsqrt, 'sycl::rsqrt('), (ESIMD, ew.rsqrt, '::rsqrt('),
    (CUDA, ew.rcbrt, 'rcbrtf('), (HIP, ew.rcbrt, 'rcbrtf(')])
def test_a_reciprocal_root_is_spelled_where_the_library_has_one(
        where, root, call):
    """A library that has the reciprocal root is asked for it."""
    gen = Generator([root(_tensor('B'), _tensor('A'))],
                    Context(arch=where[0], backend=where[1],
                            fp_type=Datatype.F32))
    gen.generate()
    assert call in gen.get_kernel()


# --------------------------------------------------------------------------- #
# The tables
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('where', [CUDA, HIP, SYCL, ESIMD, OPENMP])
@pytest.mark.parametrize('op', sorted(INFIX, key=lambda o: o.value))
def test_the_operators_are_one_table(where, op):
    """Under an explicit vector a comparison is a mask and is asked for as a
    number, which wraps it; the operator inside is the same."""
    spelled = Target(*where).lexic.get_operation(op, Datatype.F32, 'a', 'b')
    assert f'(a {INFIX[op]} b)' in spelled or f'a {INFIX[op]} b' in spelled


def _function(template: str) -> str:
    """`sycl::exp({0})` and `exp{f}({0})` -> `exp`."""
    return re.match(r'(?:\w+::)*(\w+?)(?:\{f\})?\(', template).group(1)


#: Where the floating-point function is not named after its operation: `abs`
#: is C's integer function, and C++'s `min` of floating-point numbers another
#: answer for a NaN than `fmin`.
FLOATING_POINT = {Operation.ABS: 'fabs', Operation.MIN: 'fmin',
                  Operation.MAX: 'fmax'}


@pytest.mark.parametrize('tables,names', [
    ('MATH', FLOATING_POINT), ('INTEGER_MATH', {})])
def test_a_function_is_the_same_function_in_every_library(tables, names):
    """An operation every library has is the same function in each: a typo
    in one table -- `logp1` for `log1p` -- is a name no other table has, and
    an `fmin` in one where another has `min` is another answer for a NaN."""
    libraries = [getattr(lexic, tables)
                 for lexic in (CudaLexic, SyclLexic, TargetLexic)]
    shared = set.intersection(*map(set, libraries))
    assert shared, 'the libraries share no function'
    for op in sorted(shared, key=lambda o: o.value):
        spelled = {_function(table[op]) for table in libraries}
        expected = names.get(op, op.name.lower())
        assert spelled == {expected}, f'{op}: {sorted(spelled)}'


@pytest.mark.parametrize('where', [CUDA, HIP, SYCL, OPENMP])
@pytest.mark.parametrize('type_', [Datatype.U32, Datatype.SIZE], ids=str)
def test_an_unsigned_integer_is_its_own_absolute_value(where, type_):
    """`abs` of one is ambiguous between the overloads for the signed
    types."""
    lexic = Target(*where).lexic
    assert lexic.get_operation(Operation.ABS, type_, 'x', '') == 'x'


def test_an_integer_minimum_is_the_integer_function():
    """The reduction's combine asks the same table as the elementwise
    operation."""
    b = IRBuilder(fptype=Datatype.I64)
    x = b.extern_value('x', I64, hint='x')
    y = b.extern_value('y', I64, hint='y')
    b.op('min', I64, x, y, hint='m')
    lines = []
    Emitter(lines.append, Target(*CUDA)).run(b.finish())
    assert lines[-1].endswith(' = min(x, y);'), lines[-1]


@pytest.mark.parametrize('where', [CUDA, SYCL, OPENMP])
def test_what_no_table_has_is_refused(where):
    """Not substituted: a function one library has and another does not is a
    numerics question."""
    with pytest.raises(NotImplementedError):
        Target(*where).lexic.get_operation(Operation.SIGN, Datatype.F32,
                                            'a', '')
