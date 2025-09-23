
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
    if dtype == FloatingPointType.BOOL and operation == Operation.AND:
        writer(f'{target} = __all_sync(-1, {source});')
    elif dtype == FloatingPointType.BOOL and operation == Operation.OR:
        writer(f'{target} = __any_sync(-1, {source});')
    elif dtype in [FloatingPointType.INT, FloatingPointType.UINT] and ARCH > sm80 and operation == Operation.MIN:
        writer(f'{target} = __reduction_min_sync(-1, {source});')
    elif dtype in [FloatingPointType.INT, FloatingPointType.UINT] and ARCH > sm80 and operation == Operation.MAX:
        writer(f'{target} = __reduction_max_sync(-1, {source});')
    elif dtype == FloatingPointType.FLOAT and ARCH > sm80 and operation in [Operation.MIN, Operation.MAX]:
        minmaxfloatint(writer, operation, target, source)
    elif dtype in [FloatingPointType.UINT] and ARCH > sm80 and operation == Operation.AND:
        writer(f'{target} = __reduction_and_sync(-1, {source});')
    elif dtype in [FloatingPointType.UINT] and ARCH > sm80 and operation == Operation.OR:
        writer(f'{target} = __reduction_or_sync(-1, {source});')
    elif dtype in [FloatingPointType.UINT] and ARCH > sm80 and operation == Operation.XOR:
        writer(f'{target} = __reduction_xor_sync(-1, {source});')
    elif dtype in [FloatingPointType.ULONG] and ARCH > sm80 and operation == Operation.AND:
        writer(f'{target} = __reduction_and_sync(-1, {source});')
    elif dtype in [FloatingPointType.ULONG] and ARCH > sm80 and operation == Operation.OR:
        writer(f'{target} = __reduction_or_sync(-1, {source});')
    elif dtype in [FloatingPointType.ULONG] and ARCH > sm80 and operation == Operation.XOR:
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
    elif dtype == FloatingPointType.BOOL:
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
        writer('const auto {variable}u = __float_to_tf32({variable});')
        writer('const auto {variable}l = __float_to_tf32({variable} - {variable}u);')
    return [(f'{v}u', f'{v}l') for v in variables]

def bfconvert(writer: Writer, variables):
    raise NotImplementedError()
    for v1, v2 in zip(variables[0::2], variables[1::2]):
        writer('const auto {v1}u = __float_to_tf32({v1});')
        writer('const auto {v1}m = __float_to_tf32({v1} - {v1}u);')
        writer('const auto {v1}l = __float_to_tf32({v1} - {v1}u - {v1}m);')
    return [(f'{v}u', f'{v}m', f'{v}l') for v in variables[0::2]]

class CUTEMode:
    DIRECT = 0
    TF32 = 1
    BF16 = 2

class CUTEAtom:
    def __init__(self, n, m, k, b, d, name, compress):
        self.n = n
        self.m = m
        self.k = k
        self.b = b
        self.d = d
        self.name = name
        self.compress = compress

    def headers(self):
        return ['']

    def generate(self, writer, context, A, B, C):
        Cstr = ','.join(f'{c}' for c in C)
        with writer.Scope():
            writer(f'auto mma = MMA_Atom<{self.name}>{{}};')
            if self.mode == CUTEMode.BF16:
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
            if self.mode == CUTEMode.TF32:
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
    # CUTEAtom(16,4,2,1,FloatingPointType.FLOAT,'SM80_16x8x4_F32TF32TF32F32_TN', True),
    # CUTEAtom(16,4,4,1,FloatingPointType.FLOAT,'SM80_16x8x8_F32TF32TF32F32_TN', True),
    CUTEAtom(16,4,4,1,FloatingPointType.FLOAT,'SM80_16x8x4_F32TF32TF32F32_TN', CUTEMode.TF32),
    CUTEAtom(16,4,8,1,FloatingPointType.FLOAT,'SM80_16x8x8_F32TF32TF32F32_TN', CUTEMode.TF32),
    CUTEAtom(8,8,4,1,FloatingPointType.DOUBLE,'SM80_8x8x4_F64F64F64F64_TN', CUTEMode.DIRECT),
    CUTEAtom(16,8,4,1,FloatingPointType.DOUBLE,'SM90_16x8x4_F64F64F64F64_TN', CUTEMode.DIRECT),
    CUTEAtom(16,8,8,1,FloatingPointType.DOUBLE,'SM90_16x8x8_F64F64F64F64_TN', CUTEMode.DIRECT),
    CUTEAtom(16,8,16,1,FloatingPointType.DOUBLE,'SM90_16x8x8_F64F64F64F64_TN', CUTEMode.DIRECT),
]
