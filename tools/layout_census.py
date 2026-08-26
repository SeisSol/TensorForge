# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""How many *distinct* register layouts does the generator actually produce?

The choice between a layout algebra and a table of named layouts turns on
this number.  If the set is small and closed, a table plus an equality test
answers every question a pass can ask.  If it is open -- if new layouts keep
appearing as the operator shapes change -- the relayouts have to be derived
rather than enumerated, and that needs an algebra.
"""
import importlib.util
from collections import Counter
from pathlib import Path
from tensorforge.common.context import Context
from tensorforge.generators.generator import Generator
from tensorforge.backend.pir import build as pirbuild

seen = Counter()
orig = pirbuild.IRBuilder.value
def value(self, type_, hint='', uniform=None, layout=None):
    if layout is not None:
        seen[repr(layout)] += 1
    return orig(self, type_, hint=hint,
                **({'uniform': uniform} if uniform is not None else {}),
                layout=layout)
pirbuild.IRBuilder.value = value

ARCHS = ['gfx900','gfx906','gfx908','gfx90a','gfx942','gfx950',
         'gfx1010','gfx1030','gfx1100','gfx1200','gfx1250','gfx1251']
n = 0
for p in sorted(Path('tests/cases').rglob('*.py')):
    spec = importlib.util.spec_from_file_location(p.stem, p)
    m = importlib.util.module_from_spec(spec)
    try: spec.loader.exec_module(m)
    except Exception: continue
    if not hasattr(m, 'descr_list'): continue
    n += 1
    for arch in ARCHS:
        try:
            Generator(m.descr_list(), Context(arch=arch, backend='hip',
                      fp_type=getattr(m,'DTYPE',None))).generate()
        except Exception: pass
print(f'{n} cases x {len(ARCHS)} architectures')
print(f'distinct layouts produced: {len(seen)}')
for k, v in seen.most_common():
    print(f'   {k:28s} {v:6d} values')
