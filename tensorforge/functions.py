from kernelforge.common.operation import Operation
from kernelforge.common.basic_types import FloatingPointType
from kernelforge.generators.optree import Statement, Node, TensorVar, OpNode, LexicOpNode, ConditionalOpNode, CastOpNode, IfNode, WhileNode, Assignment, Immediate, TempVar
import yateto.ast.node as ytt
from typing import List, Union
import math

