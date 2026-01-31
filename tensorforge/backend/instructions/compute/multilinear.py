from typing import Union
import math
from . import ComputeInstruction
from tensorforge.backend.symbol import SymbolType, add_offset, Symbol, SymbolView, DataView, Loop, LeadLoop, write_loops, LeadIndex, LinearizedLoop
from tensorforge.common.exceptions import InternalError
from tensorforge.backend.writer import Writer
from tensorforge.common.context import Context
from tensorforge.common.operation import ReductionOperator
from typing import Union, List
from tensorforge.common.basic_types import Datatype
from tensorforge.backend.writer import Writer

from .primitives import nvidia as nvidia
from .primitives import amd as amd

class MultilinearInstruction(ComputeInstruction):
    def __init__(self,
               context: Context,
               dest: Symbol,
               ops: List[SymbolView],
               target: List[List[int]],
               prev: Union[None, Symbol],
               productOperation: ReductionOperator,
               sumOperation: ReductionOperator,
               prefer_align: bool,
               num_threads: int,
               blockcount: int=1):
        super(MultilinearInstruction, self).__init__(context)
        self._dest = dest
        self._ops = ops
        self._target = target
        self._productOperation = productOperation
        self._sumOperation = sumOperation
        self._prefer_align = prefer_align
        self._is_ready = True
        self._user_options = context.get_user_options()
        self._gemm_meta_data = None
        self._num_threads = num_threads
        self._blockcount = blockcount
        self._prev = prev

        assert num_threads % blockcount == 0

        self.registers = None
        if dest.stype != SymbolType.Register:
            raise InternalError(f'gemm: accumulator-register array is not provided. Instead: {dest.stype}')
        else:
            self._dest = dest

        for op in self._ops:
            op.symbol.add_user(self)
        dest.add_user(self)

        self._scalar = []
        ops2 = []
        target2 = []
        for op, target in zip(self._ops, self._target):
            if len(target) == 0:
                self._scalar += [op]
            else:
                ops2 += [op]
                target2 += [target]

        self._ops = ops2
        self._target = target2

        self._analyze()

    def _analyze(self):
        targetrank = 0
        for i, op in enumerate(self._ops):
            for j in range(op.bbox.rank()):
                targetrank = max(self._target[i][j] + 1, targetrank)
        self._ns = [(-math.inf, math.inf)] * targetrank
        preKs = {}
        self._opdim_to_nks = []
        sparseK = {}
        self._sparseN = [False] * targetrank
        for i, op in enumerate(self._ops):
            opdim = [''] * op.bbox.rank()
            for j in range(op.bbox.rank()):
                # TODO: check adding the data_view box here again
                lower = op.bbox.lower()[j] #+ op.symbol.data_view._bbox.lower()[j]
                upper = op.bbox.upper()[j] #+ op.symbol.data_view._bbox.lower()[j]
                #if self._target[i][j] != 0:
                #    lower -= op.offset[j]
                #    upper -= op.offset[j]
                if self._target[i][j] < 0:
                    if self._target[i][j] not in preKs:
                        preKs[self._target[i][j]] = (lower, upper)
                        sparseK[self._target[i][j]] = False
                    preKs[self._target[i][j]] = (max(preKs[self._target[i][j]][0], lower), min(preKs[self._target[i][j]][1], upper))
                    opdim[j] = f'k{-self._target[i][j] - 1}'
                    sparseK[self._target[i][j]] |= op.symbol is not None and op.symbol.obj is not None and not op.symbol.obj.is_dense()
                else:
                    self._ns[self._target[i][j]] = (max(self._ns[self._target[i][j]][0], lower), min(self._ns[self._target[i][j]][1], upper))
                    opdim[j] = f'n{self._target[i][j]}'
                    self._sparseN[self._target[i][j]] |= op.symbol is not None and op.symbol.obj is not None and not op.symbol.obj.is_dense()

            self._opdim_to_nks += [opdim]

        self._ks = [0] * len(preKs)
        self._sparseK = [False] * len(preKs)
        for i in range(len(preKs)):
            assert -i-1 in preKs
            self._ks[i] = preKs[-i-1]
            self._sparseK[i] = sparseK[-i-1]

        iterate_dimensions = []
        loads = []
        reductions = []
        self._is_log = False

        # TODO: do not really optimize here anything any more (on a higher level)... Just generate code
        # i.e.: what can be loaded in early/late, do

        # TODO: handle offsets
        if self._prev is not None:
            self._dest.data_view = self._prev.data_view
        if self._dest.data_view is None:
            self._dest.data_view = DataView(shape = [u - l for l,u in self._ns], permute=[i for i in range(targetrank)])
            self._dest.data_view._bbox._lower = [l for l,_ in self._ns]
            self._dest.data_view._bbox._upper = [u for _,u in self._ns]

        self._lead_dims = [0]#[t for t in self._target[0] if t >= 0]

    def gen_ir(self, writer):
        pass

    def gen_code_inner(self, writer: Writer):
        if not self._nonleading_dim_test(writer):
            self._nonleading_dim(writer)
        if len(self._ns) == 0:
            self._leading_dim(writer)
        self._apply_linear(writer)

    def _nonleading_dim(self, writer: Writer):
        loopstack = []

        # TODO: preload values where necessary (i.e. no N in there)
        # Also, postpone multiplications until necessary

        # thread_mask: TODO
        # writer(f'int32_t n0 = {self._vm.get_lexic().thread_idx_x} % {self._ns[0]};')
        # writer(f'int32_t n1a = {self._vm.get_lexic().thread_idx_x} / {self._ns[0]};')
        # n1i = self._num_threads // self._ns[0]
        # writer(f'int32_t n{i} = dimmin + n1a; n{i} < {dimmax}; n{i} += {n1i}')

        # (for broadcasting)
        force_unroll = False #self._context.get_vm().get_hw_descr().vendor == 'amd'

        matrixK = 1

        loopmap = {}

        outerLoops = []

        # TODO: linearize
        for i, (dimmin, dimmax) in enumerate(self._ks):
            loopmap[f'k{i}'] = len(loopstack) + len(outerLoops)
            if -i-1 not in self._lead_dims:
                step = matrixK if i == len(self._ks) - 1 else 1
                loop = [Loop(f'k{i}', dimmin, dimmax, step, unroll=self._sparseK[i] or force_unroll)]
                if self._sparseK[i] or force_unroll or True:# and False:
                    loopstack += loop
                else:
                    outerLoops += loop

        #loopstack += [LinearizedLoop(outerLoops)]

        stride = 1
        threads = self._num_threads
        for i, (dimmin, dimmax) in enumerate(self._ns):
            loopmap[f'n{i}'] = len(loopstack) + len(outerLoops) #- 1
            if i not in self._lead_dims or threads == 0:
                loopstack += [Loop(f'n{i}', dimmin, dimmax, 1, unroll=self._sparseN[i] or force_unroll)]
            else:
                loopstack += [LeadLoop(f'n{i}', dimmin, dimmax, threads, stride, unroll=self._sparseN[i] or force_unroll)]
                threads //= dimmax - dimmin
                stride *= dimmax - dimmin

        def nonlead_writer(varlist):
            prod = []
            allLoaded = True
            for i, op in enumerate(self._ops):
                # self._ops[i].offset[j]
                allLoaded &= op.symbol.load(Writer(), self._context, f'data{i}', [add_offset(varlist[loopmap[nk]], 0) for j,nk in enumerate(self._opdim_to_nks[i])], False)
            if allLoaded and len(self._ops) > 0:
                for i, op in enumerate(self._ops):
                    loaded = op.symbol.load(writer, self._context, f'data{i}', [add_offset(varlist[loopmap[nk]], 0) for j,nk in enumerate(self._opdim_to_nks[i])], False)
                    if not loaded: break
                    if i > 0:
                        prod += [f'{self._fp_as_str} prod{i} = {self._productOperation.format(f"prod{i-1}", f"data{i}")};']
                    else:
                        prod += [f'{self._fp_as_str} prod{i} = data{i};']
                if len(self._ops) > 0 and len(prod) == len(self._ops):
                    for p in prod:
                        writer(p)
                    self._dest.load(writer, self._context, 'value', [varlist[loopmap[f'n{i}']] for i,_ in enumerate(self._ns)], False)
                    writer(f'{self._fp_as_str} newvalue = {self._sumOperation.format("value", f"prod{len(self._ops)-1}")};')
                    self._dest.store(writer, self._context, 'newvalue', [varlist[loopmap[f'n{i}']] for i,_ in enumerate(self._ns)], False)

        write_loops(self._context, writer, loopstack, nonlead_writer)

    def _nonleading_dim_test(self, writer: Writer):
        can_use = self._context.get_vm().get_hw_descr().vendor in ['amd']
        can_use &= len(self._ops) == 2

        if can_use:
            K = 1
            N = 1
            M = 1

            for mi, mx in self._ks:
                K *= mx - mi

            for mi, mx in self._ns[1:]:
                N *= mx - mi

            M *= -(-(self._ns[0][1] - self._ns[0][0]) // self._num_threads)
            Mx = (self._ns[0][1] - self._ns[0][0])

            def unwindJ(j):
                idx = [None]
                for mi, mx in self._ns[1:]:
                    size = mx - mi
                    idx += [j % size + mi]
                    j //= size
                return idx

            def unwindI(i):
                size = -(-(self._ns[0][1] - self._ns[0][0]) // self._num_threads)
                idx = [LeadIndex(i % size + self._ns[0][0] // self._num_threads, self._num_threads, 1)]
                return idx

            # TODO: remove
            kx = self._ks[0][0]

            def unwindK(k, full):
                size = self._ks[0][1] - self._ks[0][0]
                if full:
                    idx = [k % size + self._ks[0][0]]
                else:
                    sizeL = -(-(size + kx) // self._num_threads)
                    idx = [LeadIndex(k % sizeL + self._ks[0][0] // self._num_threads, self._num_threads, 1)]
                k //= size
                for mi, mx in self._ks[1:]:
                    size = mx - mi
                    idx += [k % size + mi]
                    k //= size
                return idx

            def unwindOp(i, j, k, opid, full):
                iidx = unwindI(i)
                jidx = unwindJ(j)
                kidx = unwindK(k, full)
                idx = []

                if opid is None:
                    nks = [f'n{i}' for i in range(len(self._ns))]
                else:
                    nks = self._opdim_to_nks[opid]
                for nk in nks:
                    if nk == 'n0':
                        idx += [iidx[0]]
                    elif nk[0] == 'k':
                        idx += [kidx[int(nk[1:])]]
                    else:
                        idx += [jidx[int(nk[1:])]]
                return idx

            def C(writer, var, i, j):
                self._dest.store(writer, self._context, var, unwindOp(i, j, 0, None, False), False)

            if self._ops[1].symbol.obj and (not self._ops[1].symbol.obj.is_dense() or self._ops[1].symbol.data_view.shape[0] < 16):
                def sparse(k, j):
                    if self._ops[1].symbol.obj and not self._ops[1].symbol.obj.is_dense():
                        return self._ops[1].symbol.obj.linear_index(unwindOp(0, j, k, 1, True)) is not None
                    return True
            else:
                sparse = None

            def B(writer, var, j, k):
                if sparse:
                    self._ops[1].symbol.load_linear(writer, self._context, var, k)
                    return True
                res = self._ops[1].symbol.load(Writer(), self._context, var, unwindOp(0, j, k, 1, False), False)
                if res:
                    self._ops[1].symbol.load(writer, self._context, var, unwindOp(0, j, k, 1, False), False)
                return res

            def A(writer, var, i, k):
                res = self._ops[0].symbol.load(Writer(), self._context, var, unwindOp(i, 0, k, 0, True), False)
                if res:
                    self._ops[0].symbol.load(writer, self._context, var, unwindOp(i, 0, k, 0, True), False)
                return res

            if self._context.get_vm().get_hw_descr().vendor == 'amd':
                amd.matmul(writer, C, A, B, M, N, K, kx, self._num_threads, self._dest.datatype, sparse, self._context)
            elif self._context.get_vm().get_hw_descr().vendor == 'nvidia':
                return nvidia.matmul(writer, C, A, B, Mx, N, K, kx, self._num_threads, self._dest.datatype, sparse, self._context, 'TODO', 0)
            return True
        return False

    def _nonleading_dim2(self, writer: Writer):

        # TODO: preload values where necessary (i.e. no N in there)
        # Also, postpone multiplications until necessary

        # thread_mask: TODO
        # writer(f'int32_t n0 = {self._vm.get_lexic().thread_idx_x} % {self._ns[0]};')
        # writer(f'int32_t n1a = {self._vm.get_lexic().thread_idx_x} / {self._ns[0]};')
        # n1i = self._num_threads // self._ns[0]
        # writer(f'int32_t n{i} = dimmin + n1a; n{i} < {dimmax}; n{i} += {n1i}')

        matrixK = 1

        strides = [None] * len(self._ops)

        localmaps = [None] * len(self._ops)

        for iop, op in enumerate(self._ops):
            size = 1
            stri = {}

            loopstack = []
            localmap = {}

            loopmap = {}

            # TODO: linearize
            for i, (dimmin, dimmax) in enumerate(self._ks):
                loopmap[f'k{i}'] = len(loopstack)
                if -i-1 not in self._lead_dims and f'k{i}' in self._opdim_to_nks[iop]:
                    step = matrixK if i == len(self._ks) - 1 else 1
                    loop = [Loop(f'k{i}', dimmin, dimmax, step, unroll=True)]
                    stri[f'k{i}'] = size
                    size *= dimmax - dimmin
                    loopstack += loop

            stride = 1
            threads = self._num_threads
            for i, (dimmin, dimmax) in enumerate(self._ns):
                loopmap[f'n{i}'] = len(loopstack)
                if f'n{i}' in self._opdim_to_nks[iop]:
                    stri[f'n{i}'] = size
                    if i not in self._lead_dims or threads == 0:
                        loopstack += [Loop(f'n{i}', dimmin, dimmax, 1, unroll=True)]
                        size *= dimmax - dimmin
                    else:
                        loopstack += [LeadLoop(f'n{i}', dimmin, dimmax, threads, stride, unroll=True)]
                        threads //= dimmax - dimmin
                        stride *= dimmax - dimmin

            writer(f'{self._fp_as_str} op{iop}[{size}];')

            def nonlead_writer(varlist):
                index = ' + '.join(f'{varlist[loopmap[var]].write_nonlead()} * {stri[var]}' for var in self._opdim_to_nks[iop])
                loaded = op.symbol.load(writer, self._context, f'tmp', [varlist[loopmap[nk]] for nk in self._opdim_to_nks[iop]], False)
                if loaded:
                    pos = len(localmap)
                    localmap[index] = pos
                    writer(f'op{iop}[{pos}] = tmp;')

            write_loops(self._context, writer, loopstack, nonlead_writer)

            strides[iop] = stri
            localmaps[iop] = localmap

        loopstack = []

        loopmap = {}

        outerLoops = []

        # TODO: linearize
        for i, (dimmin, dimmax) in enumerate(self._ks):
            loopmap[f'k{i}'] = len(outerLoops)
            if -i-1 not in self._lead_dims:
                step = matrixK if i == len(self._ks) - 1 else 1
                loop = [Loop(f'k{i}', dimmin, dimmax, step, unroll=self._sparseK[i])]
                if self._sparseK[i]:
                    loopstack += loop
                else:
                    outerLoops += loop

        loopstack += [LinearizedLoop(outerLoops)]

        stride = 1
        threads = self._num_threads
        for i, (dimmin, dimmax) in enumerate(self._ns):
            loopmap[f'n{i}'] = len(loopstack) + len(outerLoops) - 1
            if i not in self._lead_dims or threads == 0:
                loopstack += [Loop(f'n{i}', dimmin, dimmax, 1, unroll=self._sparseN[i])]
            else:
                loopstack += [LeadLoop(f'n{i}', dimmin, dimmax, threads, stride, unroll=self._sparseN[i])]
                threads //= dimmax - dimmin
                stride *= dimmax - dimmin

        def nonlead_writer(varlist):
            prodc = 0
            prods = []
            for i, op in enumerate(self._ops):
                index = ' + '.join(f'{varlist[loopmap[var]].write_nonlead()} * {strides[i][var]}' for var in self._opdim_to_nks[i])
                if index in localmaps[i]:
                    data = f'op{i}[{localmaps[i][index]}]'
                    if prodc > 0:
                        prods += [f'const {self._fp_as_str} prod{prodc} = {self._productOperation.format(f"prod{prodc-1}", f"{data}")};']
                    else:
                        prods += [f'const {self._fp_as_str} prod{prodc} = {data};']
                    prodc += 1
            if prodc == len(self._ops):
                for prod in prods:
                    writer(prod)
                self._dest.load(writer, self._context, 'value', [varlist[loopmap[f'n{i}']] for i,_ in enumerate(self._ns)], False)
                writer(f'{self._fp_as_str} newvalue = {self._sumOperation.format("value", f"prod{prodc - 1}")};')
                self._dest.store(writer, self._context, 'newvalue', [varlist[loopmap[f'n{i}']] for i,_ in enumerate(self._ns)], False)

        write_loops(self._context, writer, loopstack, nonlead_writer)

    def _apply_linear(self, writer: Writer):
        if len(self._scalar) == 0 and self._prev is None:
            # no linear needed
            return

        if len(self._scalar) > 0:
            scalar_var = writer.varalloc()
            writer(f'{self._fp_as_str} {scalar_var}{"{}"};')
            with writer.AnonymousScope():
                self._scalar[0].symbol.load(writer, self._context, 'value', [], False)
                writer(f'{scalar_var} = value;')
            for scalar in self._scalar[1:]:
                with writer.AnonymousScope():
                    scalar.symbol.load(writer, self._context, 'value', [], False)
                    writer(f'{scalar_var} = {self._productOperation.format("value", f"{scalar_var}")};')

        loopstack = []
        loopmap = {}

        # TODO: not fully ideal; might need only a copy paritally (i.e. use the original dimmin/dimmax)
        stride = 1
        threads = self._num_threads
        for i, (dimmin, dimmax) in enumerate(self._ns):
            loopmap[f'n{i}'] = len(loopstack)
            dimmin = self._dest.data_view.get_bbox().lower()[i]
            dimmax = self._dest.data_view.get_bbox().upper()[i]
            if i not in self._lead_dims or threads == 0:
                loopstack += [Loop(f'n{i}', dimmin, dimmax, 1, unroll=False)]
            else:
                loopstack += [LeadLoop(f'n{i}', dimmin, dimmax, threads, stride, unroll=False)]
                threads //= dimmax - dimmin
                stride *= dimmax - dimmin

        def nonlead_writer(varlist):
            self._dest.load(writer, self._context, 'value', [varlist[loopmap[f'n{i}']] for i,_ in enumerate(self._ns)], False)
            valvar = 'value'
            if len(self._scalar) > 0:
                writer(f'const {self._dest.get_fptype()} newvalue1 = {self._productOperation.format("value", f"{scalar_var}")};')
                valvar = 'newvalue1'
            if self._prev is not None:
                self._prev.load(writer, self._context, 'oldvalue', [varlist[loopmap[f'n{i}']] for i,_ in enumerate(self._ns)], False)
                writer(f'const {self._dest.get_fptype()} newvalue2 = {self._sumOperation.format("oldvalue", valvar)};')
                valvar = 'newvalue2'
            self._dest.store(writer, self._context, valvar, [varlist[loopmap[f'n{i}']] for i,_ in enumerate(self._ns)], False)

        write_loops(self._context, writer, loopstack, nonlead_writer)

    def _cublasdx_nonleadim_dim(self, writer: Writer):
        assert self._is_log
        with writer.Scope():
            # a _tiny_ bit hacky... But ok.
            writer('using namespace cublasdx;')

            m = 0
            n = 0
            k = 0

            num_threads = self._num_threads

            gemm_traits = []
            gemm_traits += [f'Size<{m}, {n}, {k}>']
            gemm_traits += ['Function<function::MM>']
            gemm_traits += ['Type<type::real>']

            transpose = lambda isTrue: 'transpose_mode::transposed' if isTrue else 'transpose_mode::non_transposed'
            gemm_traits += [f'TransposeMode<{transpose(False)}, {transpose(False)}>']
            gemm_traits += [f'Precision<{self._vm.fp_as_str()}>']

            # gemm_traits += [f'LeadingDimension<A,B,C>']

            sm = self._vm.get_hw_descr().model[3:]
            smprint = f'{sm}0'
            gemm_traits += [f'SM<{smprint}>']
            gemm_traits += ['Block']
            gemm_traits += [f'Block_Dim<{num_threads}>']
            traittype = '+'.join(f'{trait}()' for trait in gemm_traits)
            writer(f'using GemmType = decltype({traittype});')

            # currently, the alpha, beta are handled when storing back to global memory
            writer(f'GemmType().execute(1, {self._op1.name}, {self._op2.name}, 1, {self._dest.name});')

    def _leading_dim(self, writer: Writer):
        with writer.Scope():
            loopstack = []
            for i, (dimmin, dimmax) in enumerate(self._ns[1:]):
                loop = writer.For(f'int32_t n{i+1} = {dimmin}; n{i+1} < {dimmax}; ++n{i+1}', True)
                loop.__enter__()
                loopstack += [loop]

            self._dest.load(writer, self._context, 'value', [self._vm.get_lexic().thread_idx_x] + [f'n{i+1}' for i,_ in enumerate(self._ns[1:])], False)
            #writer(f'auto* shmAddr = &{self._shr_mem.name}[{self._shr_mem_offset}];')
            self._reduction(writer)
            write(f'value = tensorforge::reduction<tensorforge::ReductionOperation<{self._fp_as_str}, tensorforge::Op::Sum>, {self._num_threads}, 1, {self._fp_as_str}>(value);')
            # self._butterfly_reduction_loop(writer, max_array_length = 32, amd = False)
            #writer(f'{self._fp_as_str} newvalue = shmAddr[{sublane_address}];')
            self._dest.store(writer, self._context, 'value', [self._vm.get_lexic().thread_idx_x] + [f'n{i+1}' for i,_ in enumerate(self._ns[1:])], False)

            for loop in loopstack[::-1]:
                loop.__exit__(None, None, None)

    def _reduction(self, var, writer: Writer):
        write(f'{var} = tensorforge::reduction<tensorforge::ReductionOperation<{self._fp_as_str}, tensorforge::Op::Sum>, {self._num_threads}, 1, {self._fp_as_str}>({var});')

    def _sycl_reduction(self, writer: Writer):
        writer(f'{var} = sycl::reduction({var});')

    def _omp_reduction(self, writer: Writer):
        writer(f'#pragma omp for reduction({self._sumOperation}: shmAddr[0:{self._total_shm_size}])')
        with writer.For(f'int32_t i = 0; i < TODO; ++i'):
            writer(f'shmAddr[i] = {self._sumOperation.format("shmAddr[i]", f"value")};')

    def get_operands(self):
        inops = [op.symbol for op in self._ops] + [op.symbol for op in self._scalar]
        if self._prev is None:
            return inops
        else:
            return inops + [self._prev]

    def __str__(self):
        return f'{self._dest.name} = {self._sumOperation}({f" {self._productOperation} ".join(op.symbol.name for op in self._ops)}) {self._sumOperation} {self._prev}' # TODO: dimensions
