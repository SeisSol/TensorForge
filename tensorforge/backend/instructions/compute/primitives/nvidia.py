from tensorforge.common.basic_types import Datatype
from tensorforge.backend.writer import Writer

def reduction_generic(writer: Writer, operation, blocks):
    var = value
    with writer.Scope():
        for block in blocks:
            tempvar = writer.tempvar()
            shuffle_swap(writer, dtype, tempvar, var, block)
            writer(f'{value} = {operation.format("newvalue", {value})}')
            var = tempvar

def minmaxfloatint(writer: Writer, operation, target, source):
    with writer.Scope():
        writer(f'auto negval = __float_as_uint(max(0, {source}));')
        writer(f'auto posval = __float_as_uint(min(0, {source}));')
        writer(f'auto rednegval = __reduction_min_sync(-1, negval);')
        writer(f'auto redposval = __reduction_max_sync(-1, posval);')
        writer(f'{target} = min(__uint_as_float(rednegval), __uint_as_float(redposval));')

def full_reduction(writer: Writer, operation, dtype, target, source):
    if dtype == Datatype.BOOL and operation == Operation.AND:
        writer(f'{target} = __all_sync(-1, {source});')
    elif dtype == Datatype.BOOL and operation == Operation.OR:
        writer(f'{target} = __any_sync(-1, {source});')
    elif dtype in [Datatype.I32, Datatype.I32] and ARCH > sm80 and operation == Operation.MIN:
        writer(f'{target} = __reduction_min_sync(-1, {source});')
    elif dtype in [Datatype.I32, Datatype.I32] and ARCH > sm80 and operation == Operation.MAX:
        writer(f'{target} = __reduction_max_sync(-1, {source});')
    elif dtype == Datatype.F32 and ARCH > sm80 and operation in [Operation.MIN, Operation.MAX]:
        minmaxfloatint(writer, operation, target, source)
    elif dtype in [Datatype.I32] and ARCH > sm80 and operation == Operation.AND:
        writer(f'{target} = __reduction_and_sync(-1, {source});')
    elif dtype in [Datatype.I32] and ARCH > sm80 and operation == Operation.OR:
        writer(f'{target} = __reduction_or_sync(-1, {source});')
    elif dtype in [Datatype.I32] and ARCH > sm80 and operation == Operation.XOR:
        writer(f'{target} = __reduction_xor_sync(-1, {source});')
    elif dtype in [Datatype.I64] and ARCH > sm80 and operation == Operation.AND:
        writer(f'{target} = __reduction_and_sync(-1, {source});')
    elif dtype in [Datatype.I64] and ARCH > sm80 and operation == Operation.OR:
        writer(f'{target} = __reduction_or_sync(-1, {source});')
    elif dtype in [Datatype.I64] and ARCH > sm80 and operation == Operation.XOR:
        writer(f'{target} = __reduction_xor_sync(-1, {source});')
    # TODO: __reduction_xor_and_or_sync for uint and ulong
    else:
        reduction_generic(writer, operation, [2,4,8,16,32])

def ballot_reduction(writer: Writer, operation, subblock, block, source, target):
    with writer.Scope():
        tempvar = writer.tempvar()
        blockvar = writer.tempvar()
        subblockvar = writer.tempvar()
        maskvar = writer.tempvar()
        writer(f'const auto {tempvar} = __ballot_sync(-1, {source});')
        writer(f'const auto {blockvar} = (threadIdx.x / {block}) * {block};')
        writer(f'const auto {subblockvar} = threadIdx.x % {subblock};')

        maskval = 0
        pos = 0
        while pos < block:
            maskval |= 2**pos
            pos += subblock
        writer(f'const auto {maskvar} = ({maskval} << {subblockvar}) << {blockvar};')

        if operation == Operation.AND:
            writer(f'{target} = ({tempvar} & {maskvar}) == {maskvar};')
        if operation == Operation.OR:
            writer(f'{target} = ({tempvar} & {maskvar}) != 0;')
        if operation == Operation.XOR:
            writer(f'{target} = (__popc({tempvar} & {maskvar}) & 1) == 0;')

def reduction(writer: Writer, source, target, operation, subblock, block):
    if block == 32 and subblock == 1:
        return full_reduction(writer)
    elif dtype == Datatype.BOOL:
        return ballot_reduction(writer)
    else:
        return reduction_generic(writer, blocks)

def reduction(writer: Writer, source, target, operation, blocks):
    if sorted(blocks) == [2,4,8,16,32]:
        return full_reduction(writer)
    else:
        return reduction_generic(writer, blocks)

def shuffle_swap(writer, dtype, target, source, block):
    writer(f'{target} = __shfl_xor_sync(-1, {block >> 1}, {source});')

def shuffle_mirror(writer, dtype, target, source, block):
    writer(f'{target} = __shfl_xor_sync(-1, {block - 1}, {source});')

def shuffle_broadcast(writer, dtype, target, source, lane, subblock, block):
    if subblock == 1:
        writer(f'{target} = __shfl_sync(-1, {source}, {lane}, {block});')
    else:
        # TODO: not correct in all cases
        writer(f'{target} = __shfl_sync(-1, {source}, {lane * subblock} + (threadIdx.x % {subblock}), {block});')

def atomic(writer: Writer, target, source, operation):
    pass

def read_shared(writer: Writer, block):
    pass

def shuffle_broadcast_forall(writer, dtype, size, source, filter, callback, subblock, block):
    if block == subblock:
        if filter(0):
            callback(f'{source}', 0)
    else:
        for b in range(block // subblock):
            tempname = f'{source}temp{block}'
            if filter(b):
                with writer.Scope():
                    writer(f'{dtype} {tempname}[{size}];')
                    for i in range(size):
                        shuffle_broadcast(writer, dtype, f'{tempname}[{i}]', f'{source}[{i}]', b, subblock, block)
                    callback(f'{tempname}', b)

def prefer_rowload():
    return False

def tfconvert(writer: Writer, variables):
    for variable in variables:
        writer(f'uint32_t {variable}u, {variable}l;')
        writer(f'tensorforge::splitFloatTF32({variable}u, {variable}l, {variable});')
    return [(f'{v}u', f'{v}l') for v in variables]

def bfconvert(writer: Writer, variables):
    raise NotImplementedError()
    for v1, v2 in zip(variables[0::2], variables[1::2]):
        writer('const auto {v1}u = __float_to_tf32({v1});')
        writer('const auto {v1}m = __float_to_tf32({v1} - {v1}u);')
        writer('const auto {v1}l = __float_to_tf32({v1} - {v1}u - {v1}m);')
    return [(f'{v}u', f'{v}m', f'{v}l') for v in variables[0::2]]

class MMAMode:
    DIRECT = 0
    TF32 = 1
    BF16 = 2
    I8 = 3

class MatmulCall:
    def setup_code(self):
        return ''

    def call_code(self, a, b, c, d):
        return ''

    def __init__(self, n, m, k, b, d, name, mode):
        self.n = n
        self.m = m
        self.k = k
        self.b = b
        self.d = d
        self.name = name
        self.mode = mode

    def generate(self, writer, context, A, B, C):
        Cstr = ','.join(f'{c}' for c in C)
        with writer.Scope():
            writer(f'auto mma = cute::MMA_Atom<{self.name}>{{}};')
            if self.mode == MMAMode.BF16:
                Abf16 = bfconvert(writer, A)
                Bbf16 = bfconvert(writer, B)

                Austr = ','.join(f'{a[0]}' for a in Abf16)
                Amstr = ','.join(f'{a[1]}' for a in Abf16)
                Alstr = ','.join(f'{a[2]}' for a in Abf16)
                Bustr = ','.join(f'{b[0]}' for b in Bbf16)
                Bmstr = ','.join(f'{b[1]}' for b in Bbf16)
                Blstr = ','.join(f'{b[2]}' for b in Bbf16)

                writer(f'mma.fma({Cstr},{Austr},{Bustr},{Cstr});')
                writer(f'mma.fma({Cstr},{Austr},{Bmstr},{Cstr});')
                writer(f'mma.fma({Cstr},{Austr},{Blstr},{Cstr});')
                writer(f'mma.fma({Cstr},{Amstr},{Bustr},{Cstr});')
                writer(f'mma.fma({Cstr},{Amstr},{Bmstr},{Cstr});')
                writer(f'mma.fma({Cstr},{Alstr},{Bustr},{Cstr});')
            if self.mode == MMAMode.TF32:
                Atf32 = tfconvert(writer, A)
                Btf32 = tfconvert(writer, B)

                Austr = ','.join(f'{a[0]}' for a in Atf32)
                Alstr = ','.join(f'{a[1]}' for a in Atf32)
                Bustr = ','.join(f'{b[0]}' for b in Btf32)
                Blstr = ','.join(f'{b[1]}' for b in Btf32)

                writer(f'mma.fma({Cstr},{Austr},{Bustr},{Cstr});')
                writer(f'mma.fma({Cstr},{Austr},{Blstr},{Cstr});')
                writer(f'mma.fma({Cstr},{Alstr},{Bustr},{Cstr});')
            else:
                Astr = ','.join(f'{a}' for a in A)
                Bstr = ','.join(f'{b}' for b in B)
                writer(f'mma.fma({Cstr},{Astr},{Bstr},{Cstr});')

class MMAInstr:
    def headers(self):
        return []

    def __init__(self, m, n, k, b, d, name, mode):
        self.n = n
        self.m = m
        self.k = k
        self.b = b
        self.d = d
        self.name = name
        self.mode = mode

    def headers(self):
        return []

    def asmcall(self, writer, D, A, B, C):
        typeid = "f" if self.d == Datatype.F32 else "d"
        typeidx = "r" if self.d == Datatype.F32 else "d"

        arggrp = lambda x, b: "{" + ','.join(f"%{i + b}" for i,_ in enumerate(x)) + "}"
        arggrp2 = lambda x, o: ','.join(f'"{o}"({v})' for v in x)

        writer(f"""asm("{self.name} "
"{arggrp(D, 0)}, {arggrp(A, len(D))}, {arggrp(B, len(D + A))}, {arggrp(C, len(D + A + B))};"
: {arggrp2(D, f"={typeid}")}
: {arggrp2(A, f"{typeidx}")}, {arggrp2(B, f"{typeidx}")}, {arggrp2(C, f"{typeid}")}
);""")

    def epilogue(self):
        pass

    def generate(self, writer, context, A, B, C):
        with writer.Scope():
            if self.mode == MMAMode.I8:

                pass
            if self.mode == MMAMode.TF32:
                Atf32 = tfconvert(writer, A)
                Btf32 = tfconvert(writer, B)

                self.asmcall(writer, C, [a[0] for a in Atf32], [b[0] for b in Btf32], C)
                self.asmcall(writer, C, [a[0] for a in Atf32], [b[1] for b in Btf32], C)
                self.asmcall(writer, C, [a[1] for a in Atf32], [b[0] for b in Btf32], C)
            else:
                self.asmcall(writer, C, A, B, C)

class CUTEAtom:
    def __init__(self, m, n, k, b, d, name, mode):
        self.n = n
        self.m = m
        self.k = k
        self.b = b
        self.d = d
        self.name = name
        self.mode = mode

    def headers(self):
        return ['cute/atom.hpp']

    def generate(self, writer, context, A, B, C):
        Cstr = ','.join(f'{c}' for c in C)
        with writer.Scope():
            writer(f'auto mma = cute::MMA_Atom<{self.name}>{{}};')
            if self.mode == MMAMode.BF16:
                Abf16 = bfconvert(writer, A)
                Bbf16 = bfconvert(writer, B)

                Austr = ','.join(f'{a[0]}' for a in Abf16)
                Amstr = ','.join(f'{a[1]}' for a in Abf16)
                Alstr = ','.join(f'{a[2]}' for a in Abf16)
                Bustr = ','.join(f'{b[0]}' for b in Bbf16)
                Bmstr = ','.join(f'{b[1]}' for b in Bbf16)
                Blstr = ','.join(f'{b[2]}' for b in Bbf16)

                writer(f'mma.fma({Cstr},{Austr},{Bustr},{Cstr});')
                writer(f'mma.fma({Cstr},{Austr},{Bmstr},{Cstr});')
                writer(f'mma.fma({Cstr},{Austr},{Blstr},{Cstr});')
                writer(f'mma.fma({Cstr},{Amstr},{Bustr},{Cstr});')
                writer(f'mma.fma({Cstr},{Amstr},{Bmstr},{Cstr});')
                writer(f'mma.fma({Cstr},{Alstr},{Bustr},{Cstr});')
            if self.mode == MMAMode.TF32:
                Atf32 = tfconvert(writer, A)
                Btf32 = tfconvert(writer, B)

                Austr = ','.join(f'{a[0]}' for a in Atf32)
                Alstr = ','.join(f'{a[1]}' for a in Atf32)
                Bustr = ','.join(f'{b[0]}' for b in Btf32)
                Blstr = ','.join(f'{b[1]}' for b in Btf32)

                writer(f'mma.fma({Cstr},{Austr},{Bustr},{Cstr});')
                writer(f'mma.fma({Cstr},{Austr},{Blstr},{Cstr});')
                writer(f'mma.fma({Cstr},{Alstr},{Bustr},{Cstr});')
            else:
                Astr = ','.join(f'{a}' for a in A)
                Bstr = ','.join(f'{b}' for b in B)
                writer(f'mma.fma({Cstr},{Astr},{Bstr},{Cstr});')

ATOMS = [
    CUTEAtom(16,8,4,1,Datatype.F32,'SM80_16x8x4_F32TF32TF32F32_TN', MMAMode.TF32),
    CUTEAtom(16,8,8,1,Datatype.F32,'SM80_16x8x8_F32TF32TF32F32_TN', MMAMode.TF32),
    CUTEAtom(8,8,4,1,Datatype.F64,'SM80_8x8x4_F64F64F64F64_TN', MMAMode.DIRECT),
    CUTEAtom(16,8,4,1,Datatype.F64,'SM90_16x8x4_F64F64F64F64_TN', MMAMode.DIRECT),
    CUTEAtom(16,8,8,1,Datatype.F64,'SM90_16x8x8_F64F64F64F64_TN', MMAMode.DIRECT),
]

INSTRS = [
    MMAInstr(16,8,4,1,Datatype.F32,'mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32', MMAMode.TF32), # SM_80
    MMAInstr(16,8,8,1,Datatype.F32,'mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32', MMAMode.TF32), # SM_80
    MMAInstr(8,8,4,1,Datatype.F64,'mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64', MMAMode.DIRECT), # SM_80
    MMAInstr(16,8,4,1,Datatype.F64,'mma.sync.aligned.m16n8k4.row.col.f64.f64.f64.f64', MMAMode.DIRECT), # SM_90
    MMAInstr(16,8,8,1,Datatype.F64,'mma.sync.aligned.m16n8k8.row.col.f64.f64.f64.f64', MMAMode.DIRECT), # SM_90
    MMAInstr(16,8,16,1,Datatype.F64,'mma.sync.aligned.m16n8k16.row.col.f64.f64.f64.f64', MMAMode.DIRECT), # SM_90
    MMAInstr(8,8,16,1,Datatype.F64,'mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32', MMAMode.I8), # SM_75
    MMAInstr(16,8,16,1,Datatype.F64,'mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32', MMAMode.I8), # SM_80
    MMAInstr(16,8,32,1,Datatype.F64,'mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32', MMAMode.I8), # SM_80
]

def matmul(writer, C, A, B, M, N, K, kx, threads, dtype, sparse, ctx, shmptr, shmsize):
    if sparse:
        return False

    atom = ATOMS[1]

    mma = writer.varalloc()
    mmaT = writer.varalloc()

    Ashm = writer.varalloc()
    Bshm = writer.varalloc()

    Acopy = writer.varalloc()
    Bcopy = writer.varalloc()

    Areg = writer.varalloc()
    Breg = writer.varalloc()
    Creg = writer.varalloc()

    # for now, assume that we're warp level
    assert threads == 32

    writer(f'const auto {mma} = cute::make_tiled_mma(cute::UniversalFMA<{dtype.ctype()}, {dtype.ctype()}, {dtype.ctype()}>(), cute::Layout<cute::Shape<cute::_16, cute::_16, cute::_16>>);')
    writer(f'const auto {mmaT} = {mma}.get_slice(threadIdx.x);')

    nregs = atom.n // 4
    mregs = atom.m // 8
    kregs = atom.k // 4

    aregs = mregs * kregs
    bregs = nregs * kregs
    cregs = nregs * mregs

    for ix in range(0, M):
        for j in range(0, N, atom.n):
            for i in range(0, threads, atom.m):
                writer(f'{atom.d.ctype()} {Creg}[{cregs}]{"{}"};')
                for k in range(0, K, atom.k):
                    writer(f'{atom.d.ctype()} {Areg}[]{"{}"};')
                    writer(f'{atom.d.ctype()} {Breg}[]{"{}"};')

                    atom.generate(writer, ctx, [], [], [])
                    writer(f'cute::copy({Acopy}, {Areg}, {Ashm});')
                    writer(f'cute::copy({Bcopy}, {Breg}, {Bshm});')
                    writer(f'cute::gemm({mma}, {Areg}, {Breg}, {Creg});')
            for i in range(0, threads):
                for jj in range(0, atom.n):
                    C(writer, f'{Creg}', ix * threads + i, j + jj)

    return False

def shmsize(stages):
    atom = INSTRS[1]
    threads = 32
    aregs = (atom.m * atom.k) // threads
    bregs = (atom.n * atom.k) // threads
    cregs = (atom.m * atom.n) // threads

    return 32 * max(aregs + bregs, cregs)

class MMAWrapper:
    def headers(self):
        return []

    def __init__(self, base):
        self.n = base.n
        self.m = 32
        self.k = base.k
        self.b = base.b
        self.d = base.d
        self.name = base.name
        self.mode = base.mode
        self.base = base

    def headers(self):
        return self.base.headers()

    def generate(self, writer, context, A, B, C):
        p = self.m // self.base.m
        Ap = len(A) // p
        Cp = len(C) // p
        for i in range(p):
            self.base.generate(writer, context, A[Ap*i:Ap*(i+1)], B, C[Cp*i:Cp*(i+1)])

def matmul(writer, C, A, B, M, N, K, kx, threads, dtype, sparse, ctx, shmptr, shmsize):
    def threadrange(start, size):
        conditions = []
        if start > 0:
            conditions += [f'threadIdx.x >= {start}']
        if start + size < threads:
            conditions += [f'threadIdx.x < {start + size}']

        if len(conditions) > 0:
            return writer.If(' && '.join(conditions))
        else:
            return writer.AnonymousScope()

    if sparse:
        return False

    atom = INSTRS[1]

    mma = writer.varalloc()
    mmaT = writer.varalloc()

    Ashm = writer.varalloc()
    Bshm = writer.varalloc()

    Areg2 = writer.varalloc()
    Breg2 = writer.varalloc()

    Areg = writer.varalloc()
    Breg = writer.varalloc()
    Creg = writer.varalloc()

    # for now, assume that we're warp level
    assert threads == 32

    ntile = 8
    mtile = 8
    ktile = 4

    nregs = atom.n // ntile
    mregs = atom.m // mtile
    kregs = atom.k // ktile

    aregs = (atom.m * atom.k) // threads
    bregs = (atom.n * atom.k) // threads
    cregs = (atom.m * atom.n) // threads

    assert 32 * max(aregs + bregs, cregs) <= shmsize

    aoffs = 0
    boffs = aregs * 32
    coffs = 0

    for j in range(0, N, atom.n):
        with writer.AnonymousScope():
            for k in range(0, K + kx, threads):
                for jj in range(0, min(atom.n, N - j)):
                    B(writer, f'{Breg}_{k//threads}_{jj}', j + jj, k // threads)
                for jj in range(min(atom.n, N - j), atom.n):
                    writer(f'{atom.d.ctype()} {Breg}_{k//threads}_{jj}{"{}"};')
            for i in range(0, M, threads):
                with writer.AnonymousScope():
                    writer(f'{atom.d.ctype()} {Creg}[{cregs}][{threads // atom.m}]{"{}"};')
                    for k in range(0, K, threads):
                        with writer.AnonymousScope():
                            for kk in range(0, min(threads, K - k), atom.k):
                                with writer.AnonymousScope():
                                    writer('__syncwarp();')
                                    trueK = kk + kx
                                    trueSK = min(atom.k, threads - trueK)
                                    with threadrange(trueK, trueSK):
                                        for jj in range(0, atom.n):
                                            writer(f'{shmptr}[{boffs} + (threadIdx.x - {trueK}) % {atom.k} + {jj * atom.k}] = {Breg}_{k//threads}_{jj};')
                                    if trueSK != atom.k:
                                        with threadrange(0, atom.k - trueSK):
                                            for jj in range(0, atom.n):
                                                writer(f'{shmptr}[{boffs} + (threadIdx.x + {trueSK}) % {atom.k} + {jj * atom.k}] = {Breg}_{k//threads+1}_{jj};')
                                    writer('__syncwarp();')

                                    for jj in range(0, nregs):
                                        for kkk in range(0, kregs):
                                            writer(f'{atom.d.ctype()} {Breg2}_{kkk + jj * kregs} = {shmptr}[{boffs} + (threadIdx.x % {ktile}) + (threadIdx.x / {ktile} + {jj * ntile}) * {atom.k} + {kkk * ktile}];')

                                    for kkk in range(0, min(atom.k, K - k - kk)):
                                        A(writer, f'{Areg}_{kkk}', i // threads, k + kk + kkk)
                                    for kkk in range(min(atom.k, K - k - kk), atom.k):
                                        writer(f'{atom.d.ctype()} {Areg}_{kkk}{"{}"};')

                                    for ii in range(0, min(threads, M - i), atom.m):
                                        with writer.AnonymousScope():
                                            writer('__syncwarp();')
                                            with threadrange(ii, atom.m):
                                                for kkk in range(0, atom.k):
                                                    writer(f'{shmptr}[{aoffs} + (threadIdx.x - {ii}) % {atom.m} + {kkk * atom.m}] = {Areg}_{kkk};')
                                            writer('__syncwarp();')

                                            for kk in range(0, kregs):
                                                for iii in range(0, mregs):
                                                    writer(f'{atom.d.ctype()} {Areg2}_{iii + kk * mregs} = {shmptr}[{aoffs} + (threadIdx.x / {ktile}) + (threadIdx.x % {ktile} + {kk * ktile}) * {atom.m} + {iii * mtile}];')

                                            atom.generate(writer, ctx, [f'{Areg2}_{i}' for i in range (aregs)], [f'{Breg2}_{i}' for i in range (bregs)], [f'{Creg}[{i}][{ii // atom.m}]' for i in range (cregs)])

                    for jj in range(0, atom.n):
                        writer(f'{atom.d.ctype()} {Creg}_{jj}{"{}"};')

                    for ii in range(0, threads, atom.m):
                        with writer.AnonymousScope():
                            for jj in range(0, nregs * 2):
                                for iii in range(0, mregs):
                                    writer(f'{shmptr}[{coffs} + threadIdx.x * 2 + {iii} + {jj * 64}] = {Creg}[{iii + mregs * jj}][{ii // atom.m}];')

                            writer('__syncwarp();')
                            with threadrange(ii, atom.m):
                                for jj in range(0, atom.n):
                                    writer(f'{Creg}_{jj} = {shmptr}[{coffs} + (threadIdx.x % {atom.m}) * {atom.n} + {jj}];')
                            writer('__syncwarp();')

                    for jj in range(0, min(atom.n, N - j)):
                        C(writer, f'{Creg}_{jj}', i // threads, j + jj)

    return True
