from typing import List
from kernelforge.backend.writer import Writer
from kernelforge.common.context import Context
from kernelforge.common.operation import Operation
from kernelforge.common.basic_types import FloatingPointType

class VarAlloc:
    def __init__(self):
        self.counter = -1
    
    def alloc(self):
        self.counter += 1
        return f'v{self.counter}'

class Node:
    def tensors(self):
        pass
    
    def symbols(self):
        pass

    def assignSymbols(self, scopes: Scopes):
        pass
    
    def declare(self, alloc: VarAlloc, writer: Writer, context: Context):
        pass

    def write(self, alloc: VarAlloc, writer: Writer, context: Context):
        pass

class Variable(Node):
    def store(self, writer: Writer, context: Context, value: str):
        pass

class TensorVar(Variable):
    def __init__(self, tensor, slicing):
        self.tensor = tensor
        self.slicing = slicing
        self.symbol: Union[None, Symbol] = None
        self.variable = None
    
    def tensors(self):
        return [self.tensor]
    
    def symbols(self):
        return [self.symbol]
    
    def assignSymbols(self, scopes: Scopes):
        if self.symbol is None:
            pass

    def declare(self, alloc: VarAlloc, writer: Writer, context: Context):
        pass
    
    def write(self, alloc: VarAlloc, writer: Writer, context: Context):
        if self.variable is None:
            self.variable = alloc.alloc()
        self.symbol.load(context, writer, self.variable, [], False)
    
    def store(self, alloc: VarAlloc, writer: Writer, context: Context, value: str):
        # assume that we don't have to reload
        self.symbol.store(writer, context, value, [], False)

class TempVar(Variable):
    def __init__(self):
        self.variable = None

    def tensors(self):
        return []
    
    def symbols(self):
        return []

    def assignSymbols(self, scopes: Scopes):
        pass
    
    def declare(self, alloc: VarAlloc, writer: Writer, context: Context):
        pass
        # self.variable = alloc.alloc()
        # write(f'{context.fp_as_str()} {self.variable};')
    
    def write(self, alloc: VarAlloc, writer: Writer, context: Context):
        assert self.variable is not None
        return self.variable
    
    def store(self, alloc: VarAlloc, writer: Writer, context: Context, value: str):
        if self.variable is None:
            self.variable = alloc.alloc()
        writer(f'auto {self.variable} = {value};')

class OpNode(Node):
    def __init__(self, operands: List[Node], optype: Operation):
        self.operands = operands
        self.optype = optype
        self.variable = None

    def tensors(self):
        return [tensor for operand in self.operands for tensor in operand.tensors()]
    
    def symbols(self):
        return [symbol for operand in self.operands for symbol in operand.symbols()]
    
    def assignSymbols(self, scopes: Scopes):
        for op in self.operands:
            op.assignSymbols(scopes)

    def declare(self, alloc: VarAlloc, writer: Writer, context: Context):
        for i, op in enumerate(self.operands):
            op.declare(alloc, writer, context)

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
    def __init__(self, optype: Operation):
        self.optype = optype
    
    def operation(self, context: Context, var: List[str]):
        return context.get_vm().get_lexic().get_operation(self.optype, *var)

class ConditionalOpNode(OpNode):
    def __init__(self, optype: Operation):
        self.optype = optype
    
    def operation(self, context: Context, var: List[str]):
        return f'({var[0]}) ? ({var[1]}) : ({var[2]})'

class CastOpNode(OpNode):
    def __init__(self, targetType: FloatingPointType):
        self.targetType = targetType
    
    def operation(self, context: Context, var: List[str]):
        return 'static_cast<{self.targetType}>({var[0]})'

class IfNode(Node):
    def __init__(self, condition: Node, subassignments: List[Assignment]):
        self.condition = condition
        self.subassignments = subassignments

    def assignSymbols(self, scopes: Scopes):
        self.condition.assignSymbols(scopes)
        for subassignment in self.subassignments:
            subassignments.assignSymbols(scopes)

    def tensors(self):
        return [tensor for operand in self.operands for tensor in operand.tensors()]
    
    def symbols(self):
        return [symbol for operand in self.operands for symbol in operand.symbols()]
    
    def declare(self, alloc: VarAlloc, writer: Writer, context: Context):
        condition.declare(alloc, writer, context)
        for subassignment in self.subassignments:
            subassignment.declare(alloc, writer, context)

    def write(self, alloc: VarAlloc, writer: Writer, context: Context):
        result = self.condition.write(alloc, writer, context)
        with writer.If(result):
            for subassignment in self.subassignments:
                subassignment.write(alloc, writer, context)

class WhileNode(Node):
    def __init__(self, condition: Node, subassignments: List[Assignment]):
        self.condition = condition
        self.subassignments = subassignments
        self.conditionVar = TempVar()

    def assignSymbols(self, scopes: Scopes):
        self.condition.assignSymbols(scopes)
        for subassignment in self.subassignments:
            subassignments.assignSymbols(scopes)

    def tensors(self):
        return [tensor for operand in self.operands for tensor in operand.tensors()]
    
    def symbols(self):
        return [symbol for operand in self.operands for symbol in operand.symbols()]
    
    def declare(self, alloc: VarAlloc, writer: Writer, context: Context):
        condition.declare(alloc, writer, context)
        for subassignment in self.subassignments:
            subassignment.declare(alloc, writer, context)

    def write(self, alloc: VarAlloc, writer: Writer, context: Context):
        resultCondition = self.condition.write(alloc, writer, context)
        self.conditionVar.store(resultCondition)
        result = self.conditionVar.write(alloc, writer, context)
        with writer.While(result):
            for subassignment in self.subassignments:
                subassignment.write(alloc, writer, context)
                resultCondition = self.condition.write(alloc, writer, context)
                self.conditionVar.store(resultCondition)

class Assignment:
    def __init__(dest: Variable, optree: Node):
        self.dest = dest
        self.optree = optree
    
    def assignSymbols(self, scopes: Scopes):
        self.dest.assignSymbols(scopes)
        self.optree.assignSymbols(scopes)

    def declare(self, alloc: VarAlloc, writer: Writer, context: Context):
        optree.declare(alloc, writer, context)
        self.dest.declare(alloc, writer, context)

    def write(self, alloc: VarAlloc, writer: Writer, context: Context):
        value = optree.write(alloc, writer, context)
        self.dest.store(alloc, writer, context, value)

def writeAssignments(assignments: List[Assignment], writer: Writer, context: Context):
    alloc = VarAlloc()
    for assignment in assignments:
        assignment.declare(alloc, writer, context)
    for assignment in assignments:
        assignment.write(alloc, writer, context)
