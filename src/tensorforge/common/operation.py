# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
from enum import Enum
from typing import Union, List
import math
from abc import ABCMeta, abstractmethod

from tensorforge.common.basic_types import Datatype

class Operation(Enum):
  COPY = 0,
  MIN = 1,
  MAX = 2,
  DIV = 3,
  MUL = 4,
  ADD = 5,
  SUB = 6,
  SQRT = 7,
  POW = 8,
  CBRT = 9,
  MOD = 10,
  NEG = 11,
  RCP = 12,
  RSQRT = 13,
  RCBRT = 14,
  ABS = 15,
  CEIL = 30,
  FLOOR = 31,
  ROUND = 32,
  SIGN = 33,
  TRUNC = 34,
  LOGP1 = 94,
  EXPM1 = 95,
  GAMMA = 96,
  ERF = 97,
  LOG = 98,
  EXP = 99,
  SIN = 100,
  COS = 101,
  TAN = 102,
  ASIN = 103,
  ACOS = 104,
  ATAN = 105,
  SINH = 110,
  COSH = 111,
  TANH = 112,
  ASINH = 113,
  ACOSH = 114,
  ATANH = 115,
  NOT = 1000,
  AND = 1001,
  OR = 1002,
  XOR = 1003,
  SHR = 1100,
  SHL = 1101,
  SHRS = 1102,
  EQ = 2000,
  NEQ = 2001,
  LT = 2002,
  LE = 2003,
  GT = 2004,
  GE = 2005

class OperationType(Enum):
  FLOAT = 0,
  SINT = 1,
  UINT = 2,
  BOOLEAN = 3

_FLOATS = (Datatype.F16, Datatype.BF16, Datatype.F32, Datatype.F64,
           Datatype.F128)


def _is_float(dtype: Datatype) -> bool:
  return dtype in _FLOATS


def _int_max(dtype: Datatype) -> int:
  return (1 << (8 * dtype.size() - 1)) - 1


def _int_min(dtype: Datatype) -> int:
  return -(1 << (8 * dtype.size() - 1))


class Operator(metaclass=ABCMeta):
  """Base for the operators a descriptor can carry.

  `ABCMeta` is the point: without it `@abstractmethod` only sets a flag that
  nothing reads, so an incomplete subclass instantiates happily and fails at
  the first call instead of at construction.  Seven concrete `format` methods
  carried the decorator while also being the implementation, which is a
  contradiction that only stayed harmless because nothing enforced it.
  """

  def absorbing(self):
    """Value `a` with `op(a, x) == a` for all x, or None if there is none."""
    return None

  def irop(self):
    """Name of the matching pseudo-IR op, or None to fall back to `format`."""
    return None

  @abstractmethod
  def num_operands(self) -> Union[None, int]:
    pass

  @abstractmethod
  def datatype(self) -> List[OperationType]:
    return []

  @abstractmethod
  def format(self, *ops):
    pass

class ReductionOperator(Operator):
  @abstractmethod
  def neutral(self, dtype: Datatype):
    """Identity element for `dtype`, as a Python value.

    Type-dependent because the identity genuinely is: `min` starts from
    positive infinity over the reals and from the largest representable
    integer over `I32`, and a bitwise `and` starts from all-ones, whose
    spelling is the width's.  A single dtype-free answer is wrong for every
    operator here except `add` and `mul`.
    """

  def num_operands(self):
    return 2

class AddOperator(ReductionOperator):
  def irop(self):
    return 'add'

  def neutral(self, dtype: Datatype):
    return 0

  def format(self, *ops):
    return f'({ops[0]} + {ops[1]})'

  def datatype(self):
    return [OperationType.FLOAT, OperationType.SINT, OperationType.UINT]

  def __str__(self):
    return '+'

class MulOperator(ReductionOperator):
  def absorbing(self):
    return 0

  def irop(self):
    return 'mul'

  def neutral(self, dtype: Datatype):
    return 1

  def format(self, *ops):
    return f'({ops[0]} * {ops[1]})'

  def datatype(self):
    return [OperationType.FLOAT, OperationType.SINT, OperationType.UINT]

  def __str__(self):
    return '*'

class MinOperator(ReductionOperator):
  def irop(self):
    return 'min'

  def neutral(self, dtype: Datatype):
    if _is_float(dtype):
      return math.inf
    if dtype == Datatype.BOOL:
      return True
    return _int_max(dtype)

  def format(self, *ops):
    return f'min({ops[0]}, {ops[1]})'

  def datatype(self):
    return [OperationType.FLOAT, OperationType.SINT, OperationType.UINT]

  def __str__(self):
    return 'min'

class MaxOperator(ReductionOperator):
  def irop(self):
    return 'max'

  def neutral(self, dtype: Datatype):
    if _is_float(dtype):
      return -math.inf
    if dtype == Datatype.BOOL:
      return False
    return _int_min(dtype)

  def format(self, *ops):
    return f'max({ops[0]}, {ops[1]})'

  def datatype(self):
    return [OperationType.FLOAT, OperationType.SINT, OperationType.UINT]

  def __str__(self):
    return 'max'

class AndOperator(ReductionOperator):
  def absorbing(self):
    return 0

  def irop(self):
    # bitwise, not logical: `and` would render as `&&`
    return 'bitand'

  def neutral(self, dtype: Datatype):
    # All ones, not 1.  `True` is only the identity for a one-bit type; on
    # anything wider it clears every bit above the lowest.  Two's complement
    # makes -1 all-ones at every width, which is what `Datatype` offers --
    # it has no unsigned members.
    return True if dtype == Datatype.BOOL else -1

  def format(self, *ops):
    return f'({ops[0]} & {ops[1]})'

  def datatype(self):
    return [OperationType.BOOLEAN, OperationType.UINT]

  def __str__(self):
    return '&'

class OrOperator(ReductionOperator):
  def irop(self):
    # bitwise, not logical: `and` would render as `&&`
    return 'bitor'

  def neutral(self, dtype: Datatype):
    return False if dtype == Datatype.BOOL else 0

  def format(self, *ops):
    return f'({ops[0]} | {ops[1]})'

  def datatype(self):
    return [OperationType.BOOLEAN, OperationType.UINT]

  def __str__(self):
    return '|'

class XorOperator(ReductionOperator):
  def irop(self):
    # bitwise, not logical: `and` would render as `&&`
    return 'bitxor'

  def neutral(self, dtype: Datatype):
    return False if dtype == Datatype.BOOL else 0

  def format(self, *ops):
    return f'({ops[0]} ^ {ops[1]})'

  def datatype(self):
    return [OperationType.BOOLEAN, OperationType.UINT]

  def __str__(self):
    return '^'

class UnaryOperator(Operator):
  def num_operands(self):
    return 1

class BinaryOperator(Operator):
  def num_operands(self):
    return 2

class NegativeOperator:
  pass

class InverseOperator:
  pass
