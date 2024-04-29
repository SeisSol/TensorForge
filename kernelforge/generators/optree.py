from typing import List
from kernelforge.backend.writer import Writer
from kernelforge.common.context import Context
from kernelforge.common.operation import Operation
from kernelforge.common.basic_types import FloatingPointType
from kernelforge.backend.scopes import Scopes

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
        pass

    def assignSymbols(self, scopes: Scopes):
        pass
    
    def assignTensor(self, assigner):
        pass
    
    def declare(self, alloc: VarAlloc, writer: Writer, context: Context):
        pass

    def write(self, alloc: VarAlloc, writer: Writer, context: Context):
        pass

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
        pass

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
    
    def tensors(self, intensors=True, outtensors=True):
        return [self.tensor]
    
    def pretensors(self, intensors=True, outtensors=True):
        return [self.pretensor]
    
    def symbols(self, intensors=True, outtensors=True):
        return [self.symbol]
    
    def getRanges(self, ranges):
        for i in range(len(self.indices)):
            if self.indices[i] not in ranges:
                ranges[self.indices[i]] = (self.symbol.data_view.get_bbox().lower()[i], self.symbol.data_view.get_bbox().upper()[i])
            crange = ranges[self.indices[i]]
            crange = (min(crange[0], self.symbol.data_view.get_bbox().lower()[i]), max(crange[1], self.symbol.data_view.get_bbox().upper()[i]))
            ranges[self.indices[i]] = crange
        return ranges

    def assignSymbols(self, scopes: Scopes):
        if self.symbol is None:
            self.symbol = scopes.get_symbol(self.tensor)

    def assignTensor(self, assigner):
        self.tensor, self.indices = assigner(self.pretensor)

    def declare(self, alloc: VarAlloc, writer: Writer, context: Context):
        pass
    
    def write(self, alloc: VarAlloc, writer: Writer, context: Context):
        # TODO: re-enable caching
        # if self.variable is None:
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
    def __init__(self, operands: List[Node], optype: Operation):
        self.operands = operands
        self.optype = optype
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
    def operation(self, context: Context, var: List[str]):
        return context.get_vm().get_lexic().get_operation(self.optype, context.fp_type, *(var + ['']))

class ConditionalOpNode(OpNode):
    def operation(self, context: Context, var: List[str]):
        return f'({var[0]}) ? ({var[1]}) : ({var[2]})'

class CastOpNode(OpNode):
    def __init__(self, operands: List[Node], targetType: FloatingPointType):
        self.operands = operands
        self.targetType = targetType
        self.variable = None
        self.optype = None
    
    def operation(self, context: Context, var: List[str]):
        return 'static_cast<{self.targetType}>({var[0]})'

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
