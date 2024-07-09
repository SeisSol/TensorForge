#!/usr/bin/env python3

from tensorforge import *
from tensorforge.gemm_configuration import *

def add(g, target_input):
  M = 32
  N = 32
  K = 32
  A = Tensor('A', (M, K))
  B = Tensor('B', (K, N))
  C = Tensor('C', (M, N))

  g.add('matmulAB', C['ij'] <= A['ik'] * B['kj'], target = target_input)
  g.add('matmulATB', C['ij'] <= A['ki'] * B['kj'], target = target_input)
  g.add('matmulABT', C['ij'] <= A['ik'] * B['jk'], target = target_input)
  g.add('matmulATBT', C['ij'] <= A['ki'] * B['jk'], target = target_input)


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