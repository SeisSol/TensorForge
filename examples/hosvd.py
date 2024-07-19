#!/usr/bin/env python3

from string import ascii_lowercase as alph
from functools import reduce
from tensorforge import *

def add(g, target_input):
  N = 4

  n = 64
  r = 16
  X = Tensor('X', tuple(n for i in range(N)))
  G = Tensor('G', tuple(r for i in range(N)))
  A = [Tensor('A({})'.format(i), (n,r)) for i in range(N)]

  Alist = [A[i][alph[13+i] + alph[i]] for i in range(N)]
  hosvd = G[alph[0:N]] <= X[alph[13:13+N]] * reduce(lambda x, y: x * y, Alist)
  g.add('hosvd', hosvd, target = target_input)

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