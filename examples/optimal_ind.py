#!/usr/bin/env python3

from tensorforge import *
from itertools import permutations

def add(g, target_input):
    N = 16
    A = Tensor('A', (N, N, N, N))
    B = Tensor('B', (N, N, N, N))
    C = Tensor('C', (N, N, N, N))
    D = Tensor('D', (N, N, N, N))
    S = Tensor('S', (N, N, N, N))

    kernel = S['abij'] <= A['acik'] * B['befl'] * C['dfjk'] * D['cdel']
    g.add('kernel_opt', kernel, target_input)

    tmp1 = Tensor('tmp1', (N, N, N, N))
    tmp2 = Tensor('tmp2', (N, N, N, N))

    indices1 = 'cbdf'
    indices2 = 'cbjk'

    for i1_t in permutations(indices1):
        for i2_t in permutations(indices2):
            i1 = ''.join(i1_t)
            i2 = ''.join(i2_t)

            kernel = [  tmp1[i1]  <= B['befl'] * D['cdel'],
                        tmp2[i2]  <= tmp1[i1]  * C['dfjk'],
                        S['abij'] <= tmp2[i2]  * A['acik'] ]
            g.add('kernel_{}_{}'.format(i1,i2), kernel, target = target_input)


def gemm_cfg(arch, variant):
  if variant == 'Eigen':
    return GeneratorCollection([Eigen(arch)])
  elif variant == 'LIBXSMM':
    return GeneratorCollection([LIBXSMM(arch)])
  elif variant == 'LIBXSMM_JIT':
    return GeneratorCollection([LIBXSMM_JIT(arch)])
  elif variant == 'OpenBLAS':
    return GeneratorCollection([OpenBLAS(arch)])
  elif variant == 'PSpaMM':
    return GeneratorCollection([PSpaMM(arch)])
  else:
    raise ValueError(f'given unsupported variant: '
                     f'{variant}. Use Eigen, LIBXSMM, LIBXSMM_JIT, or OpenBLAS.')

