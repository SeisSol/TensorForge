from enum import Enum
from typing import Union, List
import math
from abc import abstractmethod

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
  CEIL = 30,
  FLOOR = 31,
  ROUND = 32,
  SIGN = 33,
  TRUNC = 34,
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
  SINH = 100,
  COSH = 101,
  TANH = 102,
  ASINH = 103,
  ACOSH = 104,
  ATANH = 105,
  NOT = 1000,
  AND = 1001,
  OR = 1002,
  XOR = 1003,
  EQ = 2000,
  NEQ = 2001,
  LT = 2002,
  LE = 2003,
  GT = 2004,
  GE = 2005

class OperationType(Enum):
  FLOAT = 0,
  INTEGER = 1,
  BOOLEAN = 2

class Operator:
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
  def neutral(self):
    pass
  
  def num_operands(self):
    return 2

class AddOperator(ReductionOperator):
  def neutral(self):
    return 0
  
  @abstractmethod
  def format(self, *ops):
    return f'({ops[0]} + {ops[1]})'
  
  def datatype(self):
    return [OperationType.FLOAT, OperationType.INTEGER]
  
  def __str__(self):
    return '+'

class MulOperator(ReductionOperator):
  def neutral(self):
    return 1
  
  @abstractmethod
  def format(self, *ops):
    return f'({ops[0]} * {ops[1]})'
  
  def datatype(self):
    return [OperationType.FLOAT, OperationType.INTEGER]
  
  def __str__(self):
    return '*'

class MinOperator(ReductionOperator):
  def neutral(self):
    return math.inf
  
  @abstractmethod
  def format(self, *ops):
    return f'min({ops[0]}, {ops[1]})'
  
  def datatype(self):
    return [OperationType.FLOAT, OperationType.INTEGER]
  
  def __str__(self):
    return 'min'

class MaxOperator(ReductionOperator):
  def neutral(self):
    return -math.inf
  
  @abstractmethod
  def format(self, *ops):
    return f'max({ops[0]}, {ops[1]})'
  
  def datatype(self):
    return [OperationType.FLOAT, OperationType.INTEGER]
  
  def __str__(self):
    return 'max'

class AndOperator(ReductionOperator):
  def neutral(self):
    return True
  
  @abstractmethod
  def format(self, *ops):
    return f'({ops[0]} & {ops[1]})'
  
  def datatype(self):
    return [OperationType.BOOLEAN, OperationType.INTEGER]
  
  def __str__(self):
    return '&'

class OrOperator(ReductionOperator):
  def neutral(self):
    return False
  
  @abstractmethod
  def format(self, *ops):
    return f'({ops[0]} | {ops[1]})'
  
  def datatype(self):
    return [OperationType.BOOLEAN, OperationType.INTEGER]
  
  def __str__(self):
    return '|'

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
