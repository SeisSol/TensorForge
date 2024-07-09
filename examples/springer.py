#!/usr/bin/env python3

from tensorforge import *
from tensorforge.gemm_configuration import *
import re

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

def cold():
  return True

_bench_no = 0
def add_tensor(name, ind, size):
  shape = tuple(size[k] for k in ind)
  return Tensor(name + str(_bench_no), shape)
  
def add_bench(g, descr, sizes, target_input):
  global _bench_no

  Cind, Aind, Bind = descr.split('-')
  size = {k: int(s) for k,s in re.findall(r'([a-z]):([0-9]+)', sizes)}

  A = add_tensor('A', Aind, size)
  B = add_tensor('B', Bind, size)
  C = add_tensor('C', Cind, size)

  g.add(descr.replace('-','_'), C[Cind] <= A[Aind] * B[Bind], target = target_input)
  _bench_no = _bench_no + 1

def add(g, target_input):
  add_bench(g, 'abc-dca-bd', 'a:312;c:296;b:24;d:312;', target_input)

  add_bench(g, 'abcd-ebad-ce', 'a:72;c:24;b:72;e:72;d:72;', target_input)

  add_bench(g, 'abcd-ea-ebcd', 'a:72;c:72;b:72;e:72;d:72;', target_input)
  add_bench(g, 'abcd-eb-aecd', 'a:72;c:72;b:72;e:72;d:72;', target_input)
  add_bench(g, 'abcd-ec-abed', 'a:72;c:72;b:72;e:72;d:72;', target_input)
  add_bench(g, 'ab-ac-cb', 'a:5136;c:5136;b:5120;', target_input)
  add_bench(g, 'ab-acd-dbc', 'a:312;c:296;b:296;d:312;', target_input)

  add_bench(g, 'abc-acd-db', 'a:312;c:296;b:296;d:312;', target_input)
  add_bench(g, 'abc-ad-bdc', 'a:312;c:296;b:312;d:296;', target_input)
  add_bench(g, 'abc-adc-bd', 'a:312;c:296;b:312;d:296;', target_input)
  add_bench(g, 'abc-adc-db', 'a:312;c:296;b:296;d:312;', target_input)
  add_bench(g, 'abc-adec-ebd', 'a:72;c:72;b:72;e:72;d:72;', target_input)
  add_bench(g, 'abcd-aebf-dfce', 'a:72;c:72;b:72;e:72;d:72;f:72;', target_input)
  add_bench(g, 'abcd-aebf-fdec', 'a:72;c:72;b:72;e:72;d:72;f:72;', target_input)
  add_bench(g, 'abcd-aecf-bfde', 'a:72;c:72;b:72;e:72;d:72;f:72;', target_input)
  add_bench(g, 'abcd-aecf-fbed', 'a:72;c:72;b:72;e:72;d:72;f:72;', target_input)
  add_bench(g, 'abcd-aedf-bfce', 'a:72;c:72;b:72;e:72;d:72;f:72;', target_input)
  add_bench(g, 'abcd-aedf-fbec', 'a:72;c:72;b:72;e:72;d:72;f:72;', target_input)
  add_bench(g, 'abcd-aefb-fdce', 'a:72;c:72;b:72;e:72;d:72;f:72;', target_input)
  add_bench(g, 'abcd-aefc-fbed', 'a:72;c:72;b:72;e:72;d:72;f:72;', target_input)

  add_bench(g, 'abcd-eafc-bfde', 'a:72;c:72;b:72;e:72;d:72;f:72;', target_input)

  add_bench(g, 'abcdef-degb-gfac', 'a:24;c:16;b:16;e:16;d:24;g:24;f:16;', target_input)
  add_bench(g, 'abcdef-degc-gfab', 'a:24;c:16;b:16;e:16;d:24;g:24;f:16;', target_input)

  add_bench(g, 'abcdef-dfgb-geac', 'a:24;c:16;b:16;e:16;d:24;g:24;f:16;', target_input)
  add_bench(g, 'abcdef-dfgc-geab', 'a:24;c:16;b:16;e:16;d:24;g:24;f:16;', target_input)

  add_bench(g, 'abcdef-efgb-gdac', 'a:24;c:16;b:16;e:24;d:16;g:24;f:16;', target_input)
  add_bench(g, 'abcdef-efgc-gdab', 'a:24;c:16;b:16;e:24;d:16;g:24;f:16;', target_input)
  add_bench(g, 'abcdef-gdab-efgc', 'a:24;c:16;b:16;e:24;d:16;g:24;f:16;', target_input)
  add_bench(g, 'abcdef-gdac-efgb', 'a:24;c:16;b:16;e:24;d:16;g:24;f:16;', target_input)

  add_bench(g, 'abcdef-geab-dfgc', 'a:24;c:16;b:16;e:16;d:24;g:24;f:16;', target_input)
  add_bench(g, 'abcdef-geac-dfgb', 'a:24;c:16;b:16;e:16;d:24;g:24;f:16;', target_input)

  add_bench(g, 'abcdef-gfab-degc', 'a:24;c:16;b:16;e:16;d:24;g:24;f:16;', target_input)
  add_bench(g, 'abcdef-gfac-degb', 'a:24;c:16;b:16;e:16;d:24;g:24;f:16;', target_input)
