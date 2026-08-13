"""Matrix paths with no call site yet.

Unreachable from `matmul()`, and deliberately not deleted: they are
matrix code, worth repairing rather than rewriting.  Two of them used to
hang off `if False:` branches in `matmul32` -- statically reachable,
never executed.  None of the four is correct as it stands; see the note
on each.
"""

from tensorforge.backend.writer import Writer


def mfma_emu_int8(writer: Writer, C, B, A, c, a, b):
    # BROKEN as written: `Aa`/`Ba`/`Ca` are declared empty and never filled
    # from `A`/`B`, the parameters `a`/`b`/`c` are shadowed by locals of the
    # same name, and the scaling the scheme depends on is still a TODO.
    # cf. the Ozaki II paper
    const = [256, 255, 253, 251, 247, 239, 233, 229, 227, 223, 217, 211, 199, 197, 193, 191]
    constM = 1
    for x in const:
        constM *= x
    constI = [pow(constM // const[i], -1, const[i]) for i in range(len(const))]
    const2 = [float((constM // const[i]) * constI[i]) for i in range(len(const))]
    acc = len(const)

    Aa = writer.varalloc()
    Ba = writer.varalloc()
    Ca = writer.varalloc()
    writer(f'int64x4_t {Aa} {"{}"};')
    writer(f'int64x4_t {Ba} {"{}"};')
    writer(f'int64x4_t {Ca} {"{}"};')

    # TODO: scale

    for x,y in zip(const, const2):
        a = writer.varalloc()
        b = writer.varalloc()
        c = writer.varalloc()
        writer(f'const auto {a} = static_cast<uint8x4_t>({Aa} % {x});')
        writer(f'const auto {b} = static_cast<uint8x4_t>({Ba} % {x});')
        writer(f'{c} = __builtin_amdgcn_mfma_i32_4x4x4i8({a}, {b}, 0, {c}, {a}, {b});')
        writer(f'{Ca} += {c} * {y};')


    # TODO: scale back

def mfma_emu_bf16_f32(writer: Writer, C, B, A, c, a, b):
    # Was reached only from an `if False:` branch in `write_matmul`.
    writer(f'const auto [{A[0]}_p0, {A[0]}_p1, {A[0]}_p2] = tensorforge::splitFloatx4BF16({A[0]}, {A[1]}, {A[2]}, {A[3]});')
    writer(f'const auto [{B[0]}_p0, {B[0]}_p1, {B[0]}_p2] = tensorforge::splitFloatx4BF16({B[0]}, {B[1]}, {B[2]}, {B[3]});')
    writer(f'{C} = __builtin_amdgcn_mfma_f32_4x4x4bf16_1k({A[0]}_p0, {B[0]}_p0, {C}, {c}, {a}, {b});')
    writer(f'{C} = __builtin_amdgcn_mfma_f32_4x4x4bf16_1k({A[0]}_p0, {B[0]}_p1, {C}, {c}, {a}, {b});')
    writer(f'{C} = __builtin_amdgcn_mfma_f32_4x4x4bf16_1k({A[0]}_p1, {B[0]}_p0, {C}, {c}, {a}, {b});')
    writer(f'{C} = __builtin_amdgcn_mfma_f32_4x4x4bf16_1k({A[0]}_p0, {B[0]}_p2, {C}, {c}, {a}, {b});')
    writer(f'{C} = __builtin_amdgcn_mfma_f32_4x4x4bf16_1k({A[0]}_p2, {B[0]}_p0, {C}, {c}, {a}, {b});')
    writer(f'{C} = __builtin_amdgcn_mfma_f32_4x4x4bf16_1k({A[0]}_p1, {B[0]}_p1, {C}, {c}, {a}, {b});')


def mfma_emu_f16_f32(writer: Writer, C, B, A, c, a, b):
    # Was reached only from an `if False:` branch in `write_matmul`.
    writer(f'const auto [{A[0]}_p0, {A[0]}_p1] = tensorforge::splitFloatx4F16({A[0]}, {A[1]}, {A[2]}, {A[3]});')
    writer(f'const auto [{B[0]}_p0, {B[0]}_p1] = tensorforge::splitFloatx4F16({B[0]}, {B[1]}, {B[2]}, {B[3]});')
    writer(f'{C} = __builtin_amdgcn_mfma_f32_4x4x4f16({A[0]}_p0, {B[0]}_p0, {C}, {c}, {a}, {b});')
    writer(f'{C} = __builtin_amdgcn_mfma_f32_4x4x4f16({A[0]}_p1, {B[0]}_p0, {C}, {c}, {a}, {b});')
    writer(f'{C} = __builtin_amdgcn_mfma_f32_4x4x4f16({A[0]}_p0, {B[0]}_p1, {C}, {c}, {a}, {b});')


def wmma3atom(writer, A, B, C, threads):
    # BROKEN as written: `[x, y, z] = f(...)` is not C++ (a structured binding
    # needs `auto [...]`), and `VectorT` is emitted without its namespace.


    a = writer.varalloc()
    b = writer.varalloc()
    c = writer.varalloc()

    assert threads == 32

    N = 16
    M = 16
    K = 16

    for m in range(2):
        with writer.AnonymousScope():
            for i in range(N):
                writer(f'const auto {a}_{i} = tensorforge::broadcast<32, 16, {m}>({A}_{i});')
            for j in range(N):
                writer(f'const auto {b}_{j} = tensorforge::broadcast<32, 16, {m}>({B}_{j});')

            writer(f'tensorforge::transpose16x16({",".join(f"{b}_{i}" for i in range(N))});')

            writer(f'VectorT<short, 16> {a}_p1{"{}"};')
            writer(f'VectorT<short, 16> {a}_p2{"{}"};')
            writer(f'VectorT<short, 16> {a}_p3{"{}"};')
            writer(f'VectorT<short, 16> {b}_p1{"{}"};')
            writer(f'VectorT<short, 16> {b}_p2{"{}"};')
            writer(f'VectorT<short, 16> {b}_p3{"{}"};')

            for i in range(N):
                writer(f'[{a}_p1[{i}], {a}_p2[{i}], {a}_p3[{i}]] = splitFloatBF16({a}_{i});')
            for i in range(N):
                writer(f'[{b}_p1[{i}], {b}_p2[{i}], {b}_p3[{i}]] = splitFloatBF16({b}_{i});')

            writer(f'VectorT<float, 8> {c}{"{}"};')
            writer(f'{c} = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32({a}_p1, {b}_p1, {c});')
            writer(f'{c} = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32({a}_p2, {b}_p1, {c});')
            writer(f'{c} = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32({a}_p1, {b}_p2, {c});')
            writer(f'{c} = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32({a}_p3, {b}_p1, {c});')
            writer(f'{c} = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32({a}_p1, {b}_p3, {c});')
            writer(f'{c} = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32({a}_p2, {b}_p2, {c});')

            for j in range(N):
                writer(f'const auto {c}_{j} = tensorforge::broadcast<32, 16, {m}>({c}[{j}]);')
