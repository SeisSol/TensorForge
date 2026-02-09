
def dpas(C, B, A, rc, sd):
    # cf. https://github.com/intel/intel-graphics-compiler/blob/master/documentation/visa/instructions/DPAS.md
    # sd == depth == k * elemsIn32Bit
    # rc == m
    writer(f'asm("DPAS.tf32.tf32.{sd}.{rc} (16) %[D], %[C], %[B], %[A]" : [D]"=f"({C}) : [C]"f"({C}), [B]"d"({B}), [A]"d"({A}) :);')

def matmul(writer, C, A, B, M, N, K, kx, threads, dtype, sparse, ctx):


    rc = 8
    sd = 8

    dpas(C, A, B, rc, sd)

    # TODO
