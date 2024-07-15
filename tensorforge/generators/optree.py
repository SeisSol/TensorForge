from typing import List, Union
from tensorforge.backend.writer import Writer
from tensorforge.common.context import Context
from tensorforge.common.operation import Operation
from tensorforge.common.basic_types import FloatingPointType
from tensorforge.backend.scopes import Scopes
import tensorforge.ast.node as ytt
import tensorforge.type as yttt

import numpy as np

class VarAlloc:
    def __init__(self):
        self.counter = -1
    
    def alloc(self):
        self.counter += 1
        return f'v{self.counter}'

class Node:
    def tensors(self, intensors=True, outtensors=True):
        pass

    def pretensors(self, intensors=True, outtensors=True):
        pass
    
    def symbols(self, intensors=True, outtensors=True):
        pass

    def getRanges(self, ranges):
        return ranges

    def assignSymbols(self, scopes: Scopes):
        pass
    
    def assignTensor(self, assigner):
        pass
    
    def declare(self, alloc: VarAlloc, writer: Writer, context: Context):
        pass

    def write(self, alloc: VarAlloc, writer: Writer, context: Context):
        pass

    def __add__(self, other):
        return add(self, other)
    
    def __radd__(self, other):
        return add(other, self)
    
    def __sub__(self, other):
        return sub(self, other)
    
    def __rsub__(self, other):
        return sub(other, self)
    
    def __mul__(self, other):
        return mul(self, other)
    
    def __rmul__(self, other):
        return mul(other, self)
    
    def __div__(self, other):
        return div(self, other)
    
    def __rdiv__(self, other):
        return div(other, self)
    
    def __truediv__(self, other):
        return div(self, other)
    
    def __rtruediv__(self, other):
        return div(other, self)

    def __mod__(self, other):
        return mod(self, other)
    
    def __rmod__(self, other):
        return mod(other, self)
    
    def __or__(self, other):
        return bitor(self, other)
    
    def __ror__(self, other):
        return bitor(other, self)
    
    def __and__(self, other):
        return bitand(self, other)
    
    def __rand__(self, other):
        return bitand(other, self)

    def __xor__(self, other):
        return bitxor(self, other)
    
    def __rxor__(self, other):
        return bitxor(other, self)
    
    def __pow__(self, other):
        return pow(self, other)
    
    def __rpow__(self, other):
        return pow(other, self)
    
    def __lt__(self, other):
        return complt(self, other)
    
    def __le__(self, other):
        return comple(self, other)
    
    def __gt__(self, other):
        return compgt(self, other)
    
    def __ge__(self, other):
        return compge(self, other)
    
    def __eq__(self, other):
        return compeq(self, other)
    
    def __ne__(self, other):
        return compne(self, other)
    
    def __neg__(self):
        return neg(self)
    
    def __pos__(self):
        return self
    
    def __abs__(self):
        return abs(self)

class Variable(Node):
    def store(self, writer: Writer, context: Context, value: str):
        pass

class Statement:
    def tensors(self, intensors=True, outtensors=True):
        pass

    def pretensors(self, intensors=True, outtensors=True):
        pass
    
    def symbols(self, intensors=True, outtensors=True):
        pass

    def getRanges(self, ranges):
        return ranges

    def assignSymbols(self, scopes: Scopes):
        pass

    def assignTensor(self, assigner):
        pass
    
    def declare(self, alloc: VarAlloc, writer: Writer, context: Context):
        pass

    def write(self, alloc: VarAlloc, writer: Writer, context: Context):
        pass

class Assignment(Statement):
    def __init__(self, dest: Variable, optree: Node):
        self.dest = dest
        self.optree = optree

    def symbols(self, intensors=True, outtensors=True):
        tensorlist = []
        if intensors:
            tensorlist += self.optree.symbols(intensors, outtensors)
        if outtensors:
            tensorlist += self.dest.symbols(intensors, outtensors)
        return tensorlist

    def tensors(self, intensors=True, outtensors=True):
        tensorlist = []
        if intensors:
            tensorlist += self.optree.tensors(intensors, outtensors)
        if outtensors:
            tensorlist += self.dest.tensors(intensors, outtensors)
        return tensorlist
    
    def pretensors(self, intensors=True, outtensors=True):
        tensorlist = []
        if intensors:
            tensorlist += self.optree.pretensors(intensors, outtensors)
        if outtensors:
            tensorlist += self.dest.pretensors(intensors, outtensors)
        return tensorlist
    
    def getRanges(self, ranges):
        ranges = self.dest.getRanges(ranges)
        ranges = self.optree.getRanges(ranges)
        return ranges

    def assignTensor(self, assigner):
        self.dest.assignTensor(assigner)
        self.optree.assignTensor(assigner)

    def assignSymbols(self, scopes: Scopes):
        self.dest.assignSymbols(scopes)
        self.optree.assignSymbols(scopes)

    def declare(self, alloc: VarAlloc, writer: Writer, context: Context):
        self.dest.declare(alloc, writer, context)
        self.optree.declare(alloc, writer, context)

    def write(self, alloc: VarAlloc, writer: Writer, context: Context):
        value = self.optree.write(alloc, writer, context)
        self.dest.store(alloc, writer, context, value)

class TensorVar(Variable):
    def __init__(self, tensor, slicing, pretensor=None):
        self.tensor = tensor
        self.slicing = slicing
        self.symbol: Union[None, Symbol] = None
        self.pretensor = pretensor
        self.variable = None
        self.indices = None
        self.offset = None
    
    def tensors(self, intensors=True, outtensors=True):
        return [self.tensor]
    
    def pretensors(self, intensors=True, outtensors=True):
        return [self.pretensor]
    
    def symbols(self, intensors=True, outtensors=True):
        return [self.symbol]
    
    def getRanges(self, ranges):
        for i in range(len(self.indices)):
            if self.indices[i] not in ranges:
                #ranges[self.indices[i]] = (self.tensor.bbox.lower()[i], self.tensor.bbox.upper()[i])
                ranges[self.indices[i]] = (0, self.tensor.bbox.size(i))
            crange = ranges[self.indices[i]]
            crange = (0, np.minimum(crange[1], self.tensor.bbox.size(i)))
            #crange = (np.maximum(crange[0], self.tensor.bbox.lower()[i]), np.minimum(crange[1], self.tensor.bbox.upper()[i]))
            ranges[self.indices[i]] = crange
        return ranges

    def assignSymbols(self, scopes: Scopes):
        if self.symbol is None:
            self.symbol = scopes.get_symbol(self.tensor.tensor)

    def assignTensor(self, assigner):
        self.tensor, self.indices = assigner(self.pretensor)
        self.offset = list(self.tensor.bbox.lower())

    def declare(self, alloc: VarAlloc, writer: Writer, context: Context):
        pass
    
    def write(self, alloc: VarAlloc, writer: Writer, context: Context):
        # TODO: re-enable caching
        # if self.variable is None:
        self.variable = alloc.alloc()
        self.symbol.load(writer, context, self.variable, [f'(n{-i-1} + {o})' for i,o in zip(self.indices, self.offset)], False)
        return self.variable
    
    def store(self, alloc: VarAlloc, writer: Writer, context: Context, value: str):
        # assume that we don't have to reload
        self.symbol.store(writer, context, value, [f'(n{-i-1} + {o})' for i,o in zip(self.indices, self.offset)], False)

class ScalarVar(Variable):
    def __init__(self, tensor, slicing, pretensor=None):
        self.tensor = tensor
        self.slicing = slicing
        self.symbol: Union[None, Symbol] = None
        self.pretensor = pretensor
        self.variable = None
        self.indices = None
    
    def tensors(self, intensors=True, outtensors=True):
        return [self.tensor]
    
    def pretensors(self, intensors=True, outtensors=True):
        return []
    
    def symbols(self, intensors=True, outtensors=True):
        return [self.symbol]
    
    def getRanges(self, ranges):
        return ranges

    def assignSymbols(self, scopes: Scopes):
        if self.symbol is None:
            self.symbol = scopes.get_symbol(self.tensor.tensor)

    def assignTensor(self, assigner):
        self.tensor, self.indices = assigner(self.pretensor)

    def declare(self, alloc: VarAlloc, writer: Writer, context: Context):
        pass
    
    def write(self, alloc: VarAlloc, writer: Writer, context: Context):
        # TODO: re-enable caching
        if self.variable is None:
            self.variable = alloc.alloc()
            self.symbol.load(writer, context, self.variable, [f'n{-i-1}' for i in self.indices], False)
        return self.variable
    
    def store(self, alloc: VarAlloc, writer: Writer, context: Context, value: str):
        # assume that we don't have to reload
        self.symbol.store(writer, context, value, [f'n{-i-1}' for i in self.indices], False)

class TempVar(Variable):
    def __init__(self):
        self.variable = None

    def tensors(self, intensors=True, outtensors=True):
        return []
    
    def symbols(self, intensors=True, outtensors=True):
        return []
    
    def pretensors(self, intensors=True, outtensors=True):
        return []

    def assignSymbols(self, scopes: Scopes):
        pass

    def assignTensor(self, assigner):
        pass

    def getRanges(self, ranges):
        return ranges
    
    def declare(self, alloc: VarAlloc, writer: Writer, context: Context):
        self.variable = alloc.alloc()
        writer(f'{context.fp_as_str()} {self.variable};')
    
    def write(self, alloc: VarAlloc, writer: Writer, context: Context):
        assert self.variable is not None
        return self.variable
    
    def store(self, alloc: VarAlloc, writer: Writer, context: Context, value: str):
        if self.variable is None:
            self.variable = alloc.alloc()
        writer(f'{self.variable} = {value};')

class Immediate(Variable):
    def __init__(self, value, fptype: FloatingPointType):
        self.value = value
        self.fptype = fptype

    def tensors(self, intensors=True, outtensors=True):
        return []
    
    def symbols(self, intensors=True, outtensors=True):
        return []
    
    def pretensors(self, intensors=True, outtensors=True):
        return []
    
    def getRanges(self, ranges):
        return ranges

    def assignSymbols(self, scopes: Scopes):
        pass
    
    def assignTensor(self, assigner):
        pass

    def declare(self, alloc: VarAlloc, writer: Writer, context: Context):
        pass
    
    def write(self, alloc: VarAlloc, writer: Writer, context: Context):
        return self.fptype.literal(self.value)
    
    def store(self, alloc: VarAlloc, writer: Writer, context: Context, value: str):
        pass

class OpNode(Node):
    def __init__(self, operands: List[Node]):
        self.operands = operands
        self.variable = None
    
    def getRanges(self, ranges):
        for op in self.operands:
            ranges = op.getRanges(ranges)
        return ranges

    def tensors(self, intensors=True, outtensors=True):
        return [tensor for operand in self.operands for tensor in operand.tensors()]
    
    def pretensors(self, intensors=True, outtensors=True):
        return [tensor for operand in self.operands for tensor in operand.pretensors()]
    
    def symbols(self, intensors=True, outtensors=True):
        return [symbol for operand in self.operands for symbol in operand.symbols()]
    
    def assignSymbols(self, scopes: Scopes):
        for op in self.operands:
            op.assignSymbols(scopes)

    def declare(self, alloc: VarAlloc, writer: Writer, context: Context):
        for op in self.operands:
            op.declare(alloc, writer, context)

    def assignTensor(self, assigner):
        for op in self.operands:
            op.assignTensor(assigner)

    def operation(self, context: Context, var: List[str]):
        pass

    def write(self, alloc: VarAlloc, writer: Writer, context: Context):
        if self.variable is None:
            self.variable = alloc.alloc()
            var = [0] * len(self.operands)
            for i, op in enumerate(self.operands):
                var[i] = op.write(alloc, writer, context)
            writer(f'const auto {self.variable} = {self.operation(context, var)};')
        return self.variable

class LexicOpNode(OpNode):
    def __init__(self, operands: List[Node], optype: Operation):
        super().__init__(operands)
        self.optype = optype

    def operation(self, context: Context, var: List[str]):
        if len(var) == 1:
            realvar = var + ['']
        else:
            realvar = var
        return context.get_vm().get_lexic().get_operation(self.optype, context.fp_type, *(realvar))

class ConditionalOpNode(OpNode):
    def operation(self, context: Context, var: List[str]):
        return f'({var[0]}) ? ({var[1]}) : ({var[2]})'

class CastOpNode(OpNode):
    def __init__(self, operands: List[Node], targetType: FloatingPointType):
        super().__init__(operands)
        self.targetType = targetType
    
    def operation(self, context: Context, var: List[str]):
        return f'static_cast<{self.targetType}>({var[0]})'

class IfNode(Statement):
    def __init__(self, condition: Node, subassignments: List[Statement]):
        self.condition = condition
        self.subassignments = subassignments

    def getRanges(self, ranges):
        ranges = self.condition.getRanges(ranges)
        for subassignment in self.subassignments:
            ranges = subassignment.getRanges(ranges)
        return ranges

    def assignSymbols(self, scopes: Scopes):
        self.condition.assignSymbols(scopes)
        for subassignment in self.subassignments:
            subassignment.assignSymbols(scopes)

    def assignTensor(self, assigner):
        self.condition.assignTensor(assigner)
        for subassignment in self.subassignments:
            subassignment.assignTensor(assigner)

    def tensors(self, intensors=True, outtensors=True):
        tensorlist = []
        if intensors:
            tensorlist += self.condition.tensors(intensors, outtensors)
        return tensorlist + [tensor for operand in self.subassignments for tensor in operand.tensors(intensors, outtensors)]
    
    def pretensors(self, intensors=True, outtensors=True):
        tensorlist = []
        if intensors:
            tensorlist += self.condition.pretensors(intensors, outtensors)
        return tensorlist + [tensor for operand in self.subassignments for tensor in operand.pretensors(intensors, outtensors)]

    def symbols(self, intensors=True, outtensors=True):
        tensorlist = []
        if intensors:
            tensorlist += self.condition.symbols(intensors, outtensors)
        return tensorlist + [tensor for operand in self.subassignments for tensor in operand.symbols(intensors, outtensors)]
    
    def declare(self, alloc: VarAlloc, writer: Writer, context: Context):
        self.condition.declare(alloc, writer, context)
        for subassignment in self.subassignments:
            subassignment.declare(alloc, writer, context)

    def write(self, alloc: VarAlloc, writer: Writer, context: Context):
        result = self.condition.write(alloc, writer, context)
        with writer.If(result):
            for subassignment in self.subassignments:
                subassignment.write(alloc, writer, context)

class WhileNode(Statement):
    def __init__(self, condition: Node, subassignments: List[Statement]):
        self.condition = condition
        self.subassignments = subassignments
        self.conditionVar = TempVar()

    def getRanges(self, ranges):
        ranges = self.condition.getRanges(ranges)
        for subassignment in self.subassignments:
            ranges = subassignment.getRanges(ranges)
        return ranges

    def assignSymbols(self, scopes: Scopes):
        self.condition.assignSymbols(scopes)
        for subassignment in self.subassignments:
            subassignment.assignSymbols(scopes)

    def assignTensor(self, assigner):
        self.condition.assignTensor(assigner)
        for subassignment in self.subassignments:
            subassignment.assignTensor(assigner)

    def tensors(self, intensors=True, outtensors=True):
        tensorlist = []
        if intensors:
            tensorlist += self.condition.tensors(intensors, outtensors)
        return tensorlist + [tensor for operand in self.subassignments for tensor in operand.tensors(intensors, outtensors)]
    
    def pretensors(self, intensors=True, outtensors=True):
        tensorlist = []
        if intensors:
            tensorlist += self.condition.pretensors(intensors, outtensors)
        return tensorlist + [tensor for operand in self.subassignments for tensor in operand.pretensors(intensors, outtensors)]

    def symbols(self, intensors=True, outtensors=True):
        tensorlist = []
        if intensors:
            tensorlist += self.condition.symbols(intensors, outtensors)
        return tensorlist + [tensor for operand in self.subassignments for tensor in operand.symbols(intensors, outtensors)]

    def declare(self, alloc: VarAlloc, writer: Writer, context: Context):
        self.conditionVar.declare(alloc, writer, context)
        self.condition.declare(alloc, writer, context)
        for subassignment in self.subassignments:
            subassignment.declare(alloc, writer, context)

    def write(self, alloc: VarAlloc, writer: Writer, context: Context):
        resultCondition = self.condition.write(alloc, writer, context)
        self.conditionVar.store(alloc, writer, context, resultCondition)
        result = self.conditionVar.write(alloc, writer, context)
        with writer.While(result):
            for subassignment in self.subassignments:
                subassignment.write(alloc, writer, context)
                resultCondition = self.condition.write(alloc, writer, context)
                self.conditionVar.store(alloc, writer, context, resultCondition)

def writeAssignments(assignments: List[Statement], writer: Writer, context: Context):
    alloc = VarAlloc()
    for assignment in assignments:
        assignment.declare(alloc, writer, context)
    for assignment in assignments:
        assignment.write(alloc, writer, context)

def scalarblock(statements: List[Statement]):
    tensors = set()
    for statement in statements:
        tensors.update(statement.pretensors())
    return ytt.ScalarRegion([tensor for tensor in tensors], statements)

BaseType = Union[ytt.Node, Node, float, int, bool]

def assign(target: Union[ytt.Node, TensorVar], source: BaseType):
    if isinstance(target, ytt.Node):
        target = tensor(target)
    return Assignment(target, immc(source))

def conditional(condition: BaseType, subnodes: List[Statement]):
    return IfNode(immc(condition), subnodes)

def ternary(condition: BaseType, yesnode: BaseType, nonode: BaseType):
    return ConditionalOpNode([immc(condition), immc(yesnode), immc(nonode)])

def loop(condition: BaseType, subnodes: List[Statement]):
    return WhileNode(immc(condition), subnodes)

def imm(value, fptype):
    return Immediate(value, fptype)

def tensor(x: ytt.Node, slicing=None):
    return TensorVar(None, slicing, x)

def scalar(x: yttt.Scalar):
    return ScalarVar(None, None, x)

def immc(x: BaseType):
    if isinstance(x, float):
        return imm(x, FloatingPointType.FLOAT)
    if isinstance(x, int):
        return imm(x, FloatingPointType.INT)
    if isinstance(x, bool):
        return imm(x, FloatingPointType.BOOL)
    if isinstance(x, ytt.Node):
        return tensor(x)
    if isinstance(x, yttt.Scalar):
        return scalar(x)
    return x

def cos(x: BaseType):
    return LexicOpNode([immc(x)], Operation.COS)

def sin(x: BaseType):
    return LexicOpNode([immc(x)], Operation.SIN)

def tan(x: BaseType):
    return LexicOpNode([immc(x)], Operation.TAN)

def acos(x: BaseType):
    return LexicOpNode([immc(x)], Operation.ACOS)

def asin(x: BaseType):
    return LexicOpNode([immc(x)], Operation.ASIN)

def atan(x: BaseType):
    return LexicOpNode([immc(x)], Operation.ATAN)

def cosh(x: BaseType):
    return LexicOpNode([immc(x)], Operation.COSH)

def sinh(x: BaseType):
    return LexicOpNode([immc(x)], Operation.SINH)

def tanh(x: BaseType):
    return LexicOpNode([immc(x)], Operation.TANH)

def acosh(x: BaseType):
    return LexicOpNode([immc(x)], Operation.ACOSH)

def asinh(x: BaseType):
    return LexicOpNode([immc(x)], Operation.ASINH)

def atanh(x: BaseType):
    return LexicOpNode([immc(x)], Operation.ATANH)

def sqrt(x: BaseType):
    return LexicOpNode([immc(x)], Operation.SQRT)

def cbrt(x: BaseType):
    return LexicOpNode([immc(x)], Operation.CBRT)

def max(x: BaseType, y: BaseType):
    return LexicOpNode([immc(x), immc(y)], Operation.MAX)

def min(x: BaseType, y: BaseType):
    return LexicOpNode([immc(x), immc(y)], Operation.MIN)

def add(x: BaseType, y: BaseType):
    return LexicOpNode([immc(x), immc(y)], Operation.ADD)

def sub(x: BaseType, y: BaseType):
    return LexicOpNode([immc(x), immc(y)], Operation.SUB)

def neg(x: BaseType):
    return LexicOpNode([immc(x)], Operation.NEG)

def mul(x: BaseType, y: BaseType):
    xconv = immc(x)
    yconv = immc(y)
    # TODO: move these optimizations to a visitor
    if isinstance(xconv, Immediate):
        if xconv.value == 1 or xconv.value == 1.0:
            return yconv
    if isinstance(yconv, Immediate):
        if yconv.value == 1 or yconv.value == 1.0:
            return xconv
    return LexicOpNode([xconv, yconv], Operation.MUL)

def div(x: BaseType, y: BaseType):
    xconv = immc(x)
    yconv = immc(y)
    # TODO: move these optimizations to a visitor
    if isinstance(xconv, Immediate):
        if xconv.value == 1 or xconv.value == 1.0:
            return LexicOpNode([yconv], Operation.RCP)
    if isinstance(yconv, Immediate):
        if yconv.value == 1 or yconv.value == 1.0:
            return xconv
    return LexicOpNode([xconv, yconv], Operation.DIV)

def mod(x: BaseType, y: BaseType):
    xconv = immc(x)
    yconv = immc(y)
    return LexicOpNode([xconv, yconv], Operation.MOD)

def round(x: BaseType):
    return LexicOpNode([immc(x)], Operation.ROUND)

def neg(x: BaseType):
    return LexicOpNode([immc(x)], Operation.NEG)

def rcp(x: BaseType):
    return LexicOpNode([immc(x)], Operation.RCP)

def pow(x: BaseType, y: BaseType):
    xconv = immc(x)
    yconv = immc(y)
    # TODO: move these optimizations to a visitor
    if isinstance(yconv, Immediate):
        if yconv.value == 2 or yconv.value == 2.0:
            return LexicOpNode([xconv, xconv], Operation.MUL)
        if yconv.value == 0.5:
            return LexicOpNode([xconv], Operation.SQRT)
        if yconv.value == -0.5:
            return LexicOpNode([xconv], Operation.RSQRT)
        if yconv.value == 1/3:
            return LexicOpNode([xconv], Operation.CBRT)
        if yconv.value == -1/3:
            return LexicOpNode([xconv], Operation.RCBRT)
        if yconv.value == -1 or yconv.value == -1.0:
            return LexicOpNode([xconv], Operation.RCP)
        if yconv.value == 1 or yconv.value == 1.0:
            return xconv
    if isinstance(xconv, Immediate):
        if xconv.value == math.e:
            return LexicOpNode([immc(y)], Operation.EXP)
        if xconv.value == 1 or xconv.value == 1.0:
            return xconv
    return LexicOpNode([xconv, yconv], Operation.POW)

def exp(x: BaseType):
    return LexicOpNode([immc(x)], Operation.EXP)

def log(x: BaseType):
    return LexicOpNode([immc(x)], Operation.LOG)

def temp():
    return TempVar()

def cast(x: Node, fptype: FloatingPointType):
    return CastOpNode([immc(x)], fptype)

def bitand(x: BaseType, y: BaseType):
    return LexicOpNode([immc(x), immc(y)], Operation.AND)

def bitor(x: BaseType, y: BaseType):
    return LexicOpNode([immc(x), immc(y)], Operation.OR)

def bitxor(x: BaseType, y: BaseType):
    return LexicOpNode([immc(x), immc(y)], Operation.XOR)

def bitnot(x: BaseType):
    return LexicOpNode([immc(x)], Operation.NOT)

def complt(x: BaseType, y: BaseType):
    return LexicOpNode([immc(x), immc(y)], Operation.LT)

def comple(x: BaseType, y: BaseType):
    return LexicOpNode([immc(x), immc(y)], Operation.LE)

def compgt(x: BaseType, y: BaseType):
    return LexicOpNode([immc(x), immc(y)], Operation.GT)

def compge(x: BaseType, y: BaseType):
    return LexicOpNode([immc(x), immc(y)], Operation.GE)

def compeq(x: BaseType, y: BaseType):
    return LexicOpNode([immc(x), immc(y)], Operation.EQ)

def compne(x: BaseType, y: BaseType):
    return LexicOpNode([immc(x), immc(y)], Operation.NEQ)

def abs(x: BaseType):
    return LexicOpNode([immc(x)], Operation.ABS)

def matmul(x: BaseType, y: BaseType):
    pass

def einsum(op: str, *args):
    pass
