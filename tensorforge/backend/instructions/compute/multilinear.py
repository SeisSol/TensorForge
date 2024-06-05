from typing import Union
import math
from . import ComputeInstruction
from tensorforge.backend.symbol import SymbolType, Symbol, SymbolView, DataView, Loop, write_loops
from tensorforge.common.exceptions import InternalError
from tensorforge.backend.writer import Writer
from tensorforge.common.context import Context
from tensorforge.common.operation import ReductionOperator
from typing import Union, List

#from .primitives.amd import shuffle_broadcast_forall

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
        self._lead_dims = [0]
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

        self._analyze()

    def _choose_lead_dim(self):
        self._shm_volume = 0
        pass

    def _analyze(self):
        targetrank = 0
        for i, op in enumerate(self._ops):
            for j in range(op.bbox.rank()):
                targetrank = max(self._target[i][j] + 1, targetrank)
        self._ns = [(-math.inf, math.inf)] * targetrank
        preKs = {}
        self._opdim_to_nks = []
        for i, op in enumerate(self._ops):
            opdim = [''] * op.bbox.rank()
            for j in range(op.bbox.rank()):
                if self._target[i][j] < 0:
                    if self._target[i][j] not in preKs:
                        preKs[self._target[i][j]] = (op.bbox.lower()[j], op.bbox.upper()[j])
                    preKs[self._target[i][j]] = (max(preKs[self._target[i][j]][0], op.bbox.lower()[j]), min(preKs[self._target[i][j]][1], op.bbox.upper()[j]))
                    opdim[j] = f'k{-self._target[i][j] - 1}'
                else:
                    self._ns[self._target[i][j]] = (max(self._ns[self._target[i][j]][0], op.bbox.lower()[j]), min(self._ns[self._target[i][j]][1], op.bbox.upper()[j]))
                    opdim[j] = f'n{self._target[i][j]}'
            self._opdim_to_nks += [opdim]
        self._ks = [0] * len(preKs)
        for i in range(len(preKs)):
            assert -i-1 in preKs
            self._ks[i] = preKs[-i-1]

        iterate_dimensions = []
        loads = []
        reductions = []
        self._is_log = False
        
        # TODO: do not really optimize here anything any more (on a higher level)... Just generate code
        # i.e.: what can be loaded in early/late, do
        
        # TODO: handle offsets
        self._dest.data_view = DataView(shape = [u - l for l,u in self._ns], permute=[i for i in range(targetrank)])
        self._dest.data_view._bbox._lower = [l for l,_ in self._ns]
        self._dest.data_view._bbox._upper = [u for _,u in self._ns]

    def gen_code_inner(self, writer: Writer):
        leading = self._ks[0] if len(self._ns) == 0 else self._ns[0]
        if self._prefer_align:
            ifguard = writer.If(self.gen_range_mask_threads(0, leading[1]))
        else:
            ifguard = writer.If(self.gen_range_mask_threads(leading[0], leading[1]))
        
        with ifguard:
            self._nonleading_dim(writer)
        if len(self._ns) == 0:
            self._leading_dim(writer)
        with ifguard:    
            if self._prev is not None:
                self._add_to_prev(writer)
        
    
    def _add_to_prev(self, writer: Writer):
        loopstack = []

        for i, (dimmin, dimmax) in enumerate(self._ns):
            if i not in self._lead_dims:
                writer.insert_pragma_unroll()
                loop = writer.For(f'int n{i} = {dimmin}; n{i} < {dimmax}; ++n{i}')
                loop.__enter__()
                loopstack += [loop]

        self._dest.load(writer, self._context, 'value', [self._vm.get_lexic().thread_idx_x] + [f'n{i+1}' for i,_ in enumerate(self._ns[1:])], False)
        self._prev.load(writer, self._context, 'oldvalue', [self._vm.get_lexic().thread_idx_x] + [f'n{i+1}' for i,_ in enumerate(self._ns[1:])], False)
        writer(f'{self._fp_as_str} newvalue = {self._sumOperation.format("value", "oldvalue")};')
        self._dest.store(writer, self._context, 'newvalue', [self._vm.get_lexic().thread_idx_x] + [f'n{i+1}' for i,_ in enumerate(self._ns[1:])], False)

        for loop in loopstack[::-1]:
            loop.__exit__(None, None, None)

    def _nonleading_dim(self, writer: Writer):
        loopstack = []

        # TODO: preload values where necessary (i.e. no N in there)
        # Also, postpone multiplications until necessary

        # thread_mask: TODO

        # writer(f'int n0 = {self._vm.get_lexic().thread_idx_x} % {self._ns[0]};')
        # writer(f'int n1a = {self._vm.get_lexic().thread_idx_x} / {self._ns[0]};')
        # n1i = self._num_threads // self._ns[0]
        # writer(f'int n{i} = dimmin + n1a; n{i} < {dimmax}; n{i} += {n1i}')

        for i, (dimmin, dimmax) in enumerate(self._ks):
            if -i-1 not in self._lead_dims:
                writer.insert_pragma_unroll()
                loop = writer.For(f'int k{i} = {dimmin}; k{i} < {dimmax}; ++k{i}')
                loop.__enter__()
                loopstack += [loop]

        for i, (dimmin, dimmax) in enumerate(self._ns):
            if i not in self._lead_dims:
                writer.insert_pragma_unroll()
                loop = writer.For(f'int n{i} = {dimmin}; n{i} < {dimmax}; ++n{i}')
                loop.__enter__()
                loopstack += [loop]

        for i, op in enumerate(self._ops):
            op.symbol.load(writer, self._context, f'data{i}', [self._vm.get_lexic().thread_idx_x if nk == 'n0' else nk for nk in self._opdim_to_nks[i]], False)
            if i > 0:
                writer(f'{self._fp_as_str} prod{i} = {self._productOperation.format(f"prod{i-1}", f"data{i}")};')
            else:
                writer(f'{self._fp_as_str} prod{i} = data{i};')
        if len(self._ops) > 0:
            self._dest.load(writer, self._context, 'value', [self._vm.get_lexic().thread_idx_x] + [f'n{i+1}' for i,_ in enumerate(self._ns[1:])], False)
            writer(f'{self._fp_as_str} newvalue = {self._sumOperation.format("value", f"prod{len(self._ops)-1}")};')
            self._dest.store(writer, self._context, 'newvalue', [self._vm.get_lexic().thread_idx_x] + [f'n{i+1}' for i,_ in enumerate(self._ns[1:])], False)

        for loop in loopstack[::-1]:
            loop.__exit__(None, None, None)

    def _nonleading_dim2(self, writer: Writer):
        loopstack = []

        # TODO: preload values where necessary (i.e. no N in there)
        # Also, postpone multiplications until necessary

        # thread_mask: TODO
        # writer(f'int n0 = {self._vm.get_lexic().thread_idx_x} % {self._ns[0]};')
        # writer(f'int n1a = {self._vm.get_lexic().thread_idx_x} / {self._ns[0]};')
        # n1i = self._num_threads // self._ns[0]
        # writer(f'int n{i} = dimmin + n1a; n{i} < {dimmax}; n{i} += {n1i}')

        loopmap = {}

        for i, (dimmin, dimmax) in enumerate(self._ks):
            if -i-1 not in self._lead_dims:
                loopmap[f'k{i}'] = len(loopstack)
                loopstack += [Loop(dimmin, dimmax, 1, unroll=False)]

        for i, (dimmin, dimmax) in enumerate(self._ns):
            if i not in self._lead_dims:
                loopmap[f'n{i}'] = len(loopstack)
                loopstack += [Loop(dimmin, dimmax, 1, unroll=False)]

        def nonlead_writer(varlist):
#            for op in enumerate(self._ops):
#                if op.symbol.
            for i, op in enumerate(self._ops):
                op.symbol.load(writer, self._context, f'data{i}', [self._vm.get_lexic().thread_idx_x if nk == 'n0' else varlist[loopmap[nk]] for nk in self._opdim_to_nks[i]], False)
                if i > 0:
                    writer(f'{self._fp_as_str} prod{i} = {self._productOperation.format(f"prod{i-1}", f"data{i}")};')
                else:
                    writer(f'{self._fp_as_str} prod{i} = data{i};')
            if len(self._ops) > 0:
                self._dest.load(writer, self._context, 'value', [self._vm.get_lexic().thread_idx_x] + [varlist[loopmap[f'n{i+1}']] for i,_ in enumerate(self._ns[1:])], False)
                writer(f'{self._fp_as_str} newvalue = {self._sumOperation.format("value", f"prod{len(self._ops)-1}")};')
                self._dest.store(writer, self._context, 'newvalue', [self._vm.get_lexic().thread_idx_x] + [varlist[loopmap[f'n{i+1}']] for i,_ in enumerate(self._ns[1:])], False)

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

            # TODO: modify, if there are problems with the Blackwell arch name
            sm = self._vm.get_hw_descr().model[3:]
            smprint = f'{sm}0'
            gemm_traits += [f'SM<{smprint}>']
            gemm_traits += ['Block']
            gemm_traits += [f'Block_Dim<{num_threads}>']
            traittype = '+'.join(f'{trait}()' for trait in gemm_traits)
            writer(f'using GemmType = decltype({traittype});')

            # currently, the alpha, beta are handled when storing back to global memory
            writer(f'GemmType().execute(1, {self._op1.name}, {self._op2.name}, 1, {self._dest.name})')

    def _leading_dim(self, writer: Writer):
        with writer.Scope():
            loopstack = []
            for i, (dimmin, dimmax) in enumerate(self._ns[1:]):
                writer.insert_pragma_unroll()
                loop = writer.For(f'int n{i+1} = {dimmin}; n{i+1} < {dimmax}; ++n{i}')
                loop.__enter__()
                loopstack += [loop]

            self._dest.load(writer, self._context, 'value', [self._vm.get_lexic().thread_idx_x] + [f'n{i+1}' for i,_ in enumerate(self._ns[1:])], False)
            #writer(f'auto* shmAddr = &{self._shr_mem.name}[{self._shr_mem_offset}];')
            self._butterfly_reduction_loop(writer, max_array_length = 32, amd = False)
            #writer(f'{self._fp_as_str} newvalue = shmAddr[{sublane_address}];')
            self._dest.store(writer, self._context, 'value', [self._vm.get_lexic().thread_idx_x] + [f'n{i+1}' for i,_ in enumerate(self._ns[1:])], False)
            
            for loop in loopstack[::-1]:
                loop.__exit__(None, None, None)

    def _reduction(self, writer: Writer, blocks):
        pass

    def _butterfly_reduction_loop(self, writer: Writer, max_array_length: int, amd: bool):
        with writer.Scope():
            loop = writer.For(f'int n = {max_array_length}; n >= 1; n /= 2')
            loop.__enter__()
            if amd:
                writer(f'{self._fp_as_str} rvalue = __shfl_xor(value, n);') # TODO: check if swizzle is used here (or DPP). DONE: it isn't. It's all permute.
                    # __builtin_amdgcn_ds_shuffle() # <-- for wave permute
                    # __amdgcn_move_dpp(int src, int dpp_ctrl, int row_mask, int bank_mask, bool bound_ctrl)
                    # __int_as_float(__amdgcn_move_dpp(__float_as_int(value), 0x12{i}, 0, 0, false)) # 1-8, float
                    # use xor/permute for everything else :(
                    # TODO: look at __builtin_amdgcn_ds_permute for active mask (it's more general than then __shfls)
            else:
                writer(f'{self._fp_as_str} rvalue = __shfl_xor_sync(-1, value, n);')
            writer(f'value = {self._sumOperation.format("value", f"rvalue")};')
            # CUDA: __reduce_OP_sync(mask, value) (if: sm_80 or higher; 32 bit)
            #writer(f'atomicAdd(&shmAddr[{sublane_address}], value);')
            loop.__exit__(None, None, None)

    def _sycl_reduction(self, writer: Writer):
        writer(f'sycl::reduction();')
    
    def _omp_reduction(self, writer: Writer):
        writer(f'#pragma omp for reduction({self._sumOperation}: shmAddr[0:{self._total_shm_size}])')
        with writer.For(f'int i = 0; i < TODO; ++i'):
            writer(f'shmAddr[i] = {self._sumOperation.format("shmAddr[i]", f"value")};')

    def get_operands(self):
        if self._prev is None:
            return [op.symbol for op in self._ops]
        else:
            return [op.symbol for op in self._ops] + [self._prev]

    def __str__(self):
        return f'{self._dest.name} = {self._sumOperation}({f" {self._productOperation} ".join(op.symbol.name for op in self._ops)}) {self._sumOperation} {self._prev}' # TODO: dimensions
