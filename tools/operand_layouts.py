"""Does the A operand of fmacdpp{step} always arrive at LaneAxis(step, 1)?

If it does, that is an invariant the layouts can *check* -- the first thing
they would do beyond describing. If it does not always, the exceptions are
the interesting part.
"""
import importlib.util
from collections import Counter
from pathlib import Path
from tensorforge.common.context import Context
from tensorforge.generators.generator import Generator
from tensorforge.backend.pir import build as pirbuild
from tensorforge.backend.pir.core import LaneAxis, RegisterLayout, Value
import re

stats = Counter()
orig = pirbuild.IRBuilder.call_stmt
def call_stmt(self, callee, *args, **kw):
    m = re.match(r'tensorforge::fmacdpp(\d+)<', callee)
    if m:
        step = int(m.group(1))
        A = args[1] if len(args) > 1 else None
        want = RegisterLayout((LaneAxis(step, 1),))
        got = A.layout if isinstance(A, Value) else None
        if got is None:
            stats['A untracked'] += 1
        elif got == want:
            stats[f'A == LaneAxis({step},1)  as expected'] += 1
        else:
            stats[f'A is {got!r}, expected {want!r}'] += 1
    m2 = re.match(r'tensorforge::transpose4x4b32', callee)
    if m2:
        stats['transposes'] += 1
    return orig(self, callee, *args, **kw)
pirbuild.IRBuilder.call_stmt = call_stmt

# and the MFMA, which goes through call()
origc = pirbuild.IRBuilder.call
def call(self, callee, type_, *args, **kw):
    if 'mfma_f32_4x4x1f32' in callee:
        A = args[0] if args else None
        got = A.layout if isinstance(A, Value) else None
        stats['mfma A: ' + (repr(got) if got else 'untracked')] += 1
    return origc(self, callee, type_, *args, **kw)
pirbuild.IRBuilder.call = call

for p in sorted(Path('tests/cases').rglob('*.py')):
    spec = importlib.util.spec_from_file_location(p.stem, p)
    m = importlib.util.module_from_spec(spec)
    try: spec.loader.exec_module(m)
    except Exception: continue
    if not hasattr(m, 'descr_list'): continue
    for arch in ['gfx900','gfx906','gfx908','gfx90a','gfx942','gfx1010','gfx1100','gfx1251']:
        try:
            Generator(m.descr_list(), Context(arch=arch, backend='hip',
                      fp_type=getattr(m,'DTYPE',None))).generate()
        except Exception: pass
for k, v in stats.most_common():
    print(f'{v:8d}  {k}')
