# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT

def dpas(writer, C, B, A, rc, sd):
    # cf. https://github.com/intel/intel-graphics-compiler/blob/master/documentation/visa/instructions/DPAS.md
    # sd == depth == k * elemsIn32Bit
    # rc == m [1,2,4,8]
    writer(f'tensorforge::intel_esimd::simd<tensorforge::TF32, 32> {A};')
    writer(f'tensorforge::intel_esimd::simd<tensorforge::TF32, 32> {B};')
    writer(f'tensorforge::intel_esimd::simd<float, 32> {C};')
    writer(f'{C} = tensorforge::intel_xmx::dpas<{sd}, {rc}, float>({C}, {B}, {A});')

def fmadpp(writer, C, B, A, size, offset, lane):
    writer(f'{C}.select<{size}, 1>({offset}) += {A}[{lane}] * {B}.select<{size}, 1>({offset});')

def load(writer, C):
    writer(f'{C}')

def matmul(writer, C, A, B, M, N, K, kx, threads, dtype, sparse, ctx):
    rc = 8
    sd = 8

    dtstr = dtype.ctype()
    writer(f'tensorforge::intel_esimd::simd<{dtstr}, 32> {C};')
