"""Migration progress, by how much a pass can actually see.

`raw*` is too coarse a bucket: `load_expr` produces a RAWEXPR whose text is
still vendor-specific but whose *result is an SSA value* and whose *memory
effect is declared* -- a pass can reorder around it and reuse it.  A RAWSTMT
with Effect.UNKNOWN can do neither.  Counting them together hides the step.
"""
import importlib.util
from collections import Counter
from pathlib import Path
from tensorforge.common.context import Context
from tensorforge.generators.generator import Generator
from tensorforge.backend.pir import build as pirbuild
from tensorforge.backend.pir.core import Effect, MemSpace

c = Counter()
orig = pirbuild.IRBuilder.emit
def emit(self, stmt):
    if stmt.op in ('rawstmt', 'rawexpr', 'rawblock'):
        opaque = bool(stmt.effect & Effect.UNKNOWN) or not stmt.accesses
        c['opaque' if opaque else 'declared'] += 1
    else:
        c['structured'] += 1
    return orig(self, stmt)
pirbuild.IRBuilder.emit = emit

print(f'{"case":24s} {"arch":8s} {"total":>6s} {"opaque":>8s} {"declared":>9s} {"structured":>11s}')
for case in ['rectangular', 'square_notrans', 'chain_three', 'trans_a']:
    p = Path(f'tests/cases/{case}.py')
    spec = importlib.util.spec_from_file_location('c', p); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    for arch, be in [('gfx90a','hip'), ('sm_86','cuda')]:
        c.clear()
        Generator(m.descr_list(), Context(arch=arch, backend=be, fp_type=getattr(m,'DTYPE',None))).generate()
        t = sum(c.values())
        print(f'{m.NAME:24s} {arch:8s} {t:6d} {c["opaque"]:5d} ({100*c["opaque"]/t:4.1f}%) '
              f'{c["declared"]:4d} ({100*c["declared"]/t:4.1f}%) {c["structured"]:5d} ({100*c["structured"]/t:4.1f}%)')
