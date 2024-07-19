#!/usr/bin/env python3

from tensorforge import *

def add(g, target_input):
  N = 8
  A = Tensor('A', (N, N))
  B = Tensor('B', (N, N, N))
  w = Tensor('w', (N,))
  C = Tensor('C', (N, N))

  kernel = C['ij'] <= 2.0 * C['ij'] + A['lj'] * B['ikl'] * w['k']
  g.add('kernel', kernel, target = target_input)


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