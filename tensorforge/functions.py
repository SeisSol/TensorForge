from kernelforge.common.operation import Operation
from kernelforge.common.basic_types import FloatingPointType
from kernelforge.generators.optree import Statement, Node, TensorVar, OpNode, LexicOpNode, ConditionalOpNode, CastOpNode, IfNode, WhileNode, Assignment, Immediate, TempVar
import yateto.ast.node as ytt
from typing import List, Union
import math

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

def conditional(condition: BaseType, subnodes: list[Statement]):
    return IfNode(immc(condition), subnodes)

def ternary(condition: BaseType, yesnode: BaseType, nonode: BaseType):
    return ConditionalOpNode([immc(condition), immc(yesnode), immc(nonode)], None)

def loop(condition: BaseType, subnodes: list[Statement]):
    return WhileNode(immc(condition), subnodes)

def imm(value, fptype):
    return Immediate(value, FloatingPointType.FLOAT)

def tensor(x: ytt.Node, slicing=None):
    return TensorVar(None, slicing, x)

def immc(x: BaseType):
    if isinstance(x, float):
        return imm(x, FloatingPointType.FLOAT)
    if isinstance(x, int):
        return imm(x, FloatingPointType.INT)
    if isinstance(x, bool):
        return imm(x, FloatingPointType.BOOL)
    if isinstance(x, ytt.Node):
        return tensor(x)
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
