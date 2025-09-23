from tensorforge.common.basic_types import FloatingPointType
from tensorforge.backend.writer import Writer

# TODO: for RDNA, wave-64, use __builtin_amdgcn_permlane16

# TODO: look at __builtin_amdgcn_readfirstlane , __builtin_amdgcn_readlane for single-value broadcasting

# TODO: list of fast broadcasts?

def dppctrl_lane4(lane1, lane2, lane3, lane4):
    return (lane4 << 6) | (lane3 << 4) | (lane2 << 2) | (lane1 << 0)

def dppctrl_row_shl(lane):
    return 0x100 + lane

def dppctrl_row_shr(lane):
    return 0x110 + lane

def dppctrl_row_ror(lane):
    return 0x120 + lane

def dppctrl_row_bcst(lane):
    return 0x150 + lane

def dppctrl_wf_shl1():
    return 0x130

def dppctrl_wf_rol1():
    return 0x134

def dppctrl_wf_shr1():
    return 0x138

def dppctrl_wf_ror1():
    return 0x13c

def dppctrl_row_mirr():
    return 0x140

def dppctrl_row_hmir():
    return 0x141

def dppctrl_row_bcast15():
    return 0x142

def dppctrl_row_bcast31():
    return 0x143

def amdgcn_dpp(data, dppctrl, rowmask, bankmask, boundctrl):
    return f'__builtin_amdgcn_mov_dpp({data}, {dppctrl}, {rowmask}, {bankmask}, {boundctrl})'

def amdgcn_swizzle_q(data, andmask, ormask, xormask):
    swizzlepattern = (1 << 15) | (xormask << 10) | (ormask << 5) | (andmask << 0)
    return f'__builtin_amdgcn_ds_swizzle({data}, {swizzlepattern})'

def amdgcn_swizzle_p(data, lane1, lane2, lane3, lane4):
    swizzlepattern = (0 << 15) | (lane4 << 6) | (lane3 << 4) | (lane2 << 2) | (lane1 << 0)
    return f'__builtin_amdgcn_ds_swizzle({data}, {swizzlepattern})'

def amdgcn_bpermute(data, source):
    return f'__builtin_amdgcn_ds_bpermute(4 * {source}, {data})'

def amdgcn_permlane64():
    return f'__builtin_amdgcn_permlane64'

def amdgcn_permlane16():
    return f'__builtin_amdgcn_permlane16'

def shuffle_swap(writer, dtype, target, source, block):
    if block == 1: iscale = 0
    if block == 2: iscale = 1
    if block == 4: iscale = 2
    if block == 8: iscale = 3
    if block == 16: iscale = 4
    if block == 32: iscale = 5
    if block == 64: iscale = 6

    if block == 1:
        writer(f'{target} = {source};')
    else:
        if block in [2,4,16]:
            if block in [2,4]:
                lanes = [0,0,0,0]
                if block == 2:
                    lanes = [1, 0, 3, 2]
                if block == 4:
                    lanes = [2, 3, 0, 1]
                dppctrl = (lanes[3] << 12) | (lanes[2] << 8) | (lanes[1] << 4) | (lanes[0] << 0)
            elif block == 16:
                # rotate right by 8 threads (in each 16-thread SIMD)
                dppctrl = 0x128
            writefun = lambda s,t: writer(f'{t} = __builtin_amdgcn_mov_dpp({s}, {dppctrl}, 15, 15, false);')
        elif block in [8,32]:
            andmask = 31
            ormask = 0
            xormask = block >> 1
            swizzlepattern = (1 << 15) | (xormask << 10) | (ormask << 5) | (andmask << 0)
            writefun = lambda s,t: writer(f'{t} = __builtin_amdgcn_ds_swizzle({s}, {swizzlepattern});')
        else: # block == 64
            # no (__lane_id() % block) needed (since (block == 64) anyways)
            blockswap = block >> 1
            mylane = f'(__lane_id() ^ {blockswap})'
            writefun = lambda s,t: writer(f'{t} = __builtin_amdgcn_ds_bpermute(4 * {mylane}, {s});')
    
        if dtype == FloatingPointType.FLOAT or dtype == FloatingPointType.INT or dtype == FloatingPointType.UINT:
            writefun(f'*(int*)&{target}', f'*(int*)&{source}')
        if dtype == FloatingPointType.DOUBLE or dtype == FloatingPointType.LONG or dtype == FloatingPointType.ULONG:
            writefun(f'*(((int*)&{target}) + 0)', f'*(((int*)&{source}) + 0)')
            writefun(f'*(((int*)&{target}) + 1)', f'*(((int*)&{source}) + 1)')

def shuffle_mirror(writer, dtype, target, source, block):
    if block == 1:
        writer(f'{target} = {source};')
    else:
        if block in [2,4,8,16]:
            if block in [2,4]:
                lanes = [0,0,0,0]
                if block == 2:
                    lanes = [1, 0, 3, 2]
                if block == 4:
                    lanes = [3, 2, 1, 0]
                dppctrl = (lanes[3] << 12) | (lanes[2] << 8) | (lanes[1] << 4) | (lanes[0] << 0)
            elif block == 8:
                dppctrl = 0x141
            elif block == 16:
                dppcrtl = 0x140
            writefun = lambda s,t: writer(f'{t} = __builtin_amdgcn_mov_dpp({s}, {dppctrl}, 15, 15, false);')
        elif block == 32:
            andmask = 31
            ormask = 0
            xormask = 31
            swizzlepattern = (1 << 15) | (xormask << 10) | (ormask << 5) | (andmask << 0)
            writefun = lambda s,t: writer(f'{t} = __builtin_amdgcn_ds_swizzle({s}, {swizzlepattern});')
        else: # block == 64
            blockswap = 63
            mylane = f'(__lane_id() ^ {blockswap})'
            writefun = lambda s,t: writer(f'{t} = __builtin_amdgcn_ds_bpermute(4 * {mylane}, {s});')
    
        if dtype == FloatingPointType.FLOAT or dtype == FloatingPointType.INT or dtype == FloatingPointType.UINT:
            writefun(f'*(int*)&{target}', f'*(int*)&{source}')
        if dtype == FloatingPointType.DOUBLE or dtype == FloatingPointType.LONG or dtype == FloatingPointType.ULONG:
            writefun(f'*(((int*)&{target}) + 0)', f'*(((int*)&{source}) + 0)')
            writefun(f'*(((int*)&{target}) + 1)', f'*(((int*)&{source}) + 1)')

def shuffle_swap_nv(writer, dtype, target, source, block):
    writer(f'{target} = __shfl_xor_sync(-1, {block >> 1}, {source});')

def reduction(writer, target, source, dtype, operator, blocks):
    # TODO: can be done with DPP only? (at least when doing a _complete_ reduction... Maybe do a move otherwise?)
    # we can: reduce to one variable; then use readline to move to all
    # well.......... We can use DPP up to 16
    with writer.Scope():
        writer(f'{dtype} temp, temp2;')
        writer(f'temp = {source};')
        temp, temp2 = 'temp', 'temp2'
        for block in blocks:
            shuffle_swap(writer, dtype, temp2, temp, block)
            writer(f'{temp2} = {operator.format(temp, temp2)};')
            temp, temp2 = temp2, temp
        writer(f'{target} = {temp2};')

def shuffle_broadcast(writer, dtype, target, source, lane, subblock, block):
    assert block % subblock == 0
    assert lane < block // subblock

    if subblock == 1: iscale = 0
    if subblock == 2: iscale = 1
    if subblock == 4: iscale = 2
    if subblock == 8: iscale = 3
    if subblock == 16: iscale = 4
    if subblock == 32: iscale = 5
    if subblock == 64: iscale = 6

    # TODO: are two DPP instructions better than one swizzle?
    if block == subblock:
        writer(f'{target} = {source};')
    else:
        if block in [2,4]:
            lanes = [0,0,0,0]
            if subblock == 1:
                lanes = [lane] * 4
            if subblock == 2:
                lanebase = 2 * lane
                lanes = [lanebase, lanebase + 1] * 2
            dppctrl = (lanes[3] << 12) | (lanes[2] << 8) | (lanes[1] << 4) | (lanes[0] << 0)
            writefun = lambda s,t: writer(f'{t} = __builtin_amdgcn_mov_dpp({s}, {dppctrl}, 15, 15, false);')
        elif block == 8 and subblock == 4:
            if lane == 0:
                dppctrl = 0x114 # shr by 4
                bankmask = 0b1010
            if lane == 1:
                dppctrl = 0x104 # shl by 4
                bankmask = 0b0101
            writefun = lambda s,t: writer(f'{t} = __builtin_amdgcn_mov_dpp({s}, {dppctrl}, 15, {bankmask}, false);')
        elif block == 16 and subblock == 1: # TODO: not on older HW
            dppctrl = 0x150 + lane
            writefun = lambda s,t: writer(f'{t} = __builtin_amdgcn_mov_dpp({s}, {dppctrl}, 15, 15, false);')
        elif block == 16 and subblock == 8:
            dppctrl = 0x128 # ror by 8
            bankmask = 0b1100 if lane == 0 else 0b0011
            writefun = lambda s,t: writer(f'{t} = __builtin_amdgcn_mov_dpp({s}, {dppctrl}, 15, {bankmask}, false);')
        elif block == 32 and subblock == 1 and lane == 15: # TODO: not on RDNA
            dppctrl = 0x142 # bst 15
            rowmask = 0b1010
            writefun = lambda s,t: writer(f'{t} = __builtin_amdgcn_mov_dpp({s}, {dppctrl}, {rowmask}, 15, false);')
        elif block == 64 and subblock == 1 and lane == 31: # TODO: not on RDNA
            dppctrl = 0x143 # bst 31
            rowmask = 0b1100
            writefun = lambda s,t: writer(f'{t} = __builtin_amdgcn_mov_dpp({s}, {dppctrl}, {rowmask}, 15, false);')
        elif block == 64 and subblock == 1:
            writefun = lambda s,t: writer(f'{t} = __builtin_amdgcn_readlane({s}, __builtin_amdgcn_readfirstlane({lane}));')
        elif block in [8,16,32]:
            andmask = (1 << iscale) - 1
            ormask = lane << iscale
            xormask = 0
            swizzlepattern = (1 << 15) | (xormask << 10) | (ormask << 5) | (andmask << 0)
            writefun = lambda s,t: writer(f'{t} = __builtin_amdgcn_ds_swizzle({s}, {swizzlepattern});')
        else: # block == 64
            # no block offset needed (since (block == 64) anyways)
            mylane = f'((__lane_id() % {subblock}) + {subblock * lane})'
            writefun = lambda s,t: writer(f'{t} = __builtin_amdgcn_ds_bpermute(4 * {mylane}, {s});')
    
        if dtype == FloatingPointType.FLOAT or dtype == FloatingPointType.INT or dtype == FloatingPointType.UINT:
            writefun(f'*(int*)&{target}', f'*(int*)&{source}')
        if dtype == FloatingPointType.DOUBLE or dtype == FloatingPointType.LONG or dtype == FloatingPointType.ULONG:
            writefun(f'*(((int*)&{target}) + 0)', f'*(((int*)&{source}) + 0)')
            writefun(f'*(((int*)&{target}) + 1)', f'*(((int*)&{source}) + 1)')

def shuffle_broadcast_forall(writer, dtype, size, source, filter, callback, subblock, block):
    def shuffle_broadcast_forall_sub(block, temp):
        tempname = f'{source}temp{block}'
        if block == subblock:
            if filter(0):
                callback(f'{source}', 0)
        if block in [8,16,32,64] and not (subblock == 1 and block == 64):
            shuffle_broadcast_forall_sub(block // 2)
            with writer.Scope():
                writer(f'{dtype} {tempname}[{size}];')
                # TODO: maybe shuffle_mirror at times?
                shuffle_swap(writer, dtype, target, source, block // 2)
                shuffle_broadcast_forall_sub(block // 2)
        else:
            for b in range(block // subblock):
                if filter(b):
                    with writer.Scope():
                        writer(f'{dtype} {tempname}[{size}];')
                        for i in range(size):
                            shuffle_broadcast(writer, dtype, f'{tempname}[{i}]', f'{source}[{i}]', b, subblock, block)
                        callback(f'{tempname}', b)
    shuffle_broadcast_forall(block, source)

class MatrixCore:
    def __init__(self, n: int, m: int, k: int, b: int, d: FloatingPointType, instr: str):
        self.n = n
        self.m = m
        self.k = k
        self.b = b
        self.d = d
        self.instr = instr
        self.logb = 0

    def matmul(self, writer, A, B, C, n, m, k, bA, bB):
        assert n >= self.n
        assert m >= self.m
        # TODO: add shuffle, if needed
    
    def horizontal_kernel(self, writer, logbA, logbB, A, B, C):
        assert self.k == 1

        assert logbA <= self.logb
        assert logbB <= self.logb

        ascale = self.logb - logbA
        bscale = self.logb - logbB

        if bscale == 0:
            bidx = lambda x: 0
        else:
            if logbB == 1:
                bidx = lambda x: x+1
            if logbB >= 2:
                bidx = lambda x: x+4

        cappedbscale = bscale if bscale <= 2 else 2
        def subkernel(B, k):
            for j in range(2**cappedbscale):
                for i in range(2**ascale):
                    writer(f'C = {self.instr}(A, {B}, C, {ascale}, {i}, {bidx(j)});')

        block = 64 // 2**bscale if bscale <= 2 else 16
        shuffle_broadcast_forall(writer, self.d, self.k, 'B', subkernel, logbB, block)

matrixcores = {
    'cdna1': [
        MatrixCore(16, 16, 1, 4, FloatingPointType.FLOAT, '__builtin_amdgcn_mfma_f32_16x16x1f32'),
        MatrixCore(4, 4, 1, 16, FloatingPointType.FLOAT, '__builtin_amdgcn_mfma_f32_4x4x1f32')
    ],
    'cdna2': [
        MatrixCore(16, 16, 1, 4, FloatingPointType.FLOAT, '__builtin_amdgcn_mfma_f32_16x16x1f32'),
        MatrixCore(4, 4, 1, 16, FloatingPointType.FLOAT, '__builtin_amdgcn_mfma_f32_4x4x1f32'),
        MatrixCore(16, 16, 4, 1, FloatingPointType.DOUBLE, '__builtin_amdgcn_mfma_f64_16x16x4f64'),
        MatrixCore(4, 4, 4, 4, FloatingPointType.DOUBLE, '__builtin_amdgcn_mfma_f64_4x4x4f64')
    ],
    'cdna3': [
        MatrixCore(16, 16, 1, 4, FloatingPointType.FLOAT, '__builtin_amdgcn_mfma_f32_16x16x1f32_b4'),
        MatrixCore(4, 4, 1, 16, FloatingPointType.FLOAT, '__builtin_amdgcn_mfma_f32_4x4x1f32_b16')
    ]
}

archmap = {
    'gfx90a': 'cdna2',
    'gfx940': 'cdna3',
    'gfx941': 'cdna3',
    'gfx942': 'cdna3'
}

def matmul(writer, dtype, sumOp, prodOp, n, m, k, A, B, C, block, zeroinit):
    matrixOp = None
    #if sumOp == TODO and prodOp == TODO:
    #    pass
        # LOOK UP
    if matrixOp:
        pass
        # BLOCK HERE
        # matrixOp.horizontal_kernel(writer, TODO, TODO)
    else:
        totalload = m*k
        for jk in range((totalload + 63) // 64):
            def callback(B, i):
                with writer.Scope():
                    argA = f'{A}[TODO]'
                    argB = f'{B}[TODO]'
                    argC = f'{C}[TODO]'
                    writer(f'auto prod = {prodOp.format(argA, argB)}')
                    writer(f'C[{i}] = {sumOp.format(argC, "prod")}')
            pos = jk * 64
            filter = lambda i: pos + i < totalload
            shuffle_broadcast_forall(writer, dtype, dtype, B, filter, callback, 1, block)

def atomic(writer: Writer, target, source, operation):
    pass

def read_shared(writer: Writer, subblock, block):
    pass

def prefer_rowload():
    return True

def reduction(writer: Writer, source, target, operation, blocks):
    var = value
    with writer.Scope():
        for block in blocks:
            tempvar = writer.tempvar()
            writer(f'auto {tempvar} = __shfl_xor_sync(-1, {block >> 1}, {value});')
            writer(f'{value} = {operation.format("newvalue", {value})}')
            var = tempvar
