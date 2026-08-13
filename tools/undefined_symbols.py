"""Which fmacdpp symbols does the generator actually put in the source,
and does the runtime define each of them for that target?"""
import importlib.util, re
from collections import defaultdict
from pathlib import Path
from tensorforge.common.context import Context
from tensorforge.generators.generator import Generator

ARCHS = ['gfx900','gfx906','gfx908','gfx90a','gfx940','gfx942','gfx950',
         'gfx1010','gfx1030','gfx1100','gfx1200','gfx1250','gfx1251']
SYM = re.compile(r'fmacdpp(\d+)')

def has(sym, arch):
    if sym == 4:  return arch != 'gfx900'
    if sym == 8:  return False
    if sym == 16: return arch in ('gfx90a','gfx940','gfx941','gfx942','gfx950') \
                      or arch[3:5] in ('10','11','12')
    return True

emitted = defaultdict(set)
for p in sorted(Path('tests/cases').rglob('*.py')):
    spec = importlib.util.spec_from_file_location(p.stem, p)
    m = importlib.util.module_from_spec(spec)
    try: spec.loader.exec_module(m)
    except Exception: continue
    if not hasattr(m, 'descr_list'): continue
    for arch in ARCHS:
        try:
            g = Generator(m.descr_list(), Context(arch=arch, backend='hip',
                          fp_type=getattr(m,'DTYPE',None)))
            g.generate()
            for w in SYM.findall(g.get_kernel()):
                emitted[arch].add(int(w))
        except Exception: pass

bad = 0
for arch in ARCHS:
    syms = sorted(emitted[arch])
    marks = []
    for s in syms:
        ok = has(s, arch)
        if not ok: bad += 1
        marks.append(f'fmacdpp{s}{"" if ok else " <-- UNDEFINED"}')
    print(f'{arch:9s} {", ".join(marks) if marks else "(none)"}')
print(f'--- {bad} calls to undefined symbols ---')
