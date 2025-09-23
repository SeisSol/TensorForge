from tensorforge.common.matrix.tensor import Tensor
from . import ComputeInstruction
from tensorforge.common.exceptions import InternalError
from tensorforge.backend.writer import Writer
from tensorforge.common.context import Context
from typing import List
from tensorforge.generators.optree import Assignment, writeAssignments
from tensorforge.backend.scopes import Scopes
from tensorforge.backend.symbol import Loop, LeadLoop, write_loops, LeadIndex

class ElementwiseInstruction(ComputeInstruction):
    def __init__(self,
               context: Context,
               assignments: List[Assignment],
               scopes: Scopes,
               prefer_align: bool,
               num_threads: int):
        super(ElementwiseInstruction, self).__init__(context)
        self._assignments = assignments
        self._prefer_align = prefer_align
        self._is_ready = True
        self._user_options = context.get_user_options()
        self._gemm_meta_data = None
        self._num_threads = num_threads

        self._lead_dims = [0]

        self.registers = None

        # TODO: get index list
        seen_tensors = set()
        ranges = {}
        for assignment in self._assignments:
          assignment.assignSymbols(scopes)
          ranges = assignment.getRanges(ranges)
          for tensor in assignment.symbols():
            if tensor not in seen_tensors:
              tensor.add_user(self)
              seen_tensors.add(tensor)
              if not isinstance(tensor.obj, Tensor):
                raise InternalError('elementwise: op is not a matrix')
        self._ks = [None] * len(ranges)
        for i in range(len(ranges)):
            assert -i-1 in ranges
            self._ks[i] = ranges[-i-1]

    def gen_code_inner(self, writer: Writer):
        self._assignment_loop(writer)

    def _assignment_loop(self, writer: Writer):
        loopstack = []

        for i, (dimmin, dimmax) in enumerate(self._ks):
            if i not in self._lead_dims:
                loopstack += [Loop(f'k{i}', dimmin, dimmax, 1, unroll=False)]
            else:
                loopstack += [LeadLoop(f'k{i}', dimmin, dimmax, self._num_threads, unroll=False)]

        def inner(varlist):
            for i,_ in enumerate(self._ks):
                writer(f'auto n{i} = {varlist[i].write(self._context)};')
            writeAssignments(self._assignments, writer, self._context)
        
        write_loops(self._context, writer, loopstack, inner)

    def get_operands(self):
        return [] # TODO: for now

    def __str__(self):
        return ', '.join(str(assignment) for assignment in self._assignments)
