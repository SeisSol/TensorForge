"""Duplicate output elements, measured without double-wrapping.

`matmul32` calls `matmuldpp` internally, so wrapping the C callback in both
places logs every store of the inner call twice.  Wrap once, at `matmul`, and
attribute the path by asking which function is on the stack.
"""
import importlib.util, inspect
from collections import Counter
from pathlib import Path
from tensorforge.common.context import Context
from tensorforge.generators.generator import Generator
import tensorforge.backend.instructions.compute.primitives.amd as amd

orig = amd.matmul
call = [0]; log = []

def spy(writer, C, A, B, M, N, K, kx, threads, dtype, sparse, ctx):
    call[0] += 1
    n = call[0]
    def C2(w, var, i, j):
        fns = {f.function for f in inspect.stack()[:6]}
        path = 'mfma' if 'write_matmul' in fns else 'dpp'
        log.append((n, path, i, j))
        return C(w, var, i, j)
    return orig(writer, C2, A, B, M, N, K, kx, threads, dtype, sparse, ctx)

print(f'{"case":26s} {"calls":>6s} {"stores":>7s} {"dupes":>6s}  by path')
for case in ['rectangular','square_notrans','trans_b','trans_a',
             'chain_three','chain_five','f64','rectangular']:
    p = Path(f'tests/cases/{case}.py')
    if not p.exists(): continue
    spec = importlib.util.spec_from_file_location('c', p)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    log.clear(); call[0] = 0
    amd.matmul = spy
    Generator(m.descr_list(), Context(arch='gfx90a', backend='hip',
              fp_type=getattr(m,'DTYPE',None))).generate()
    amd.matmul = orig
    cnt = Counter((c, i, j) for c, _, i, j in log)
    dup = {k: v for k, v in cnt.items() if v > 1}
    tags = Counter(t for _, t, _, _ in log)
    print(f'{m.NAME:26s} {call[0]:6d} {len(log):7d} {len(dup):6d}  {dict(tags)}'
          + ('  ' + str(sorted(dup)[:4]) if dup else ''))
