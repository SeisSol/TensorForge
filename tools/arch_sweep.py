# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Sweep every AMD arch over the case corpus; record failures and emitted helpers."""
import importlib.util
import re
import sys
from pathlib import Path

sys.path.insert(0, 'tests')

from tensorforge.common.context import Context
from tensorforge.generators.generator import Generator

ARCHS = ['gfx900', 'gfx906', 'gfx908', 'gfx90a', 'gfx940', 'gfx942', 'gfx950',
         'gfx1010', 'gfx1030', 'gfx1100', 'gfx1200', 'gfx1250', 'gfx1251']

CASE_DIR = Path('tests/cases')


def load_cases():
    cases = []
    for p in sorted(CASE_DIR.rglob('*.py')):
        spec = importlib.util.spec_from_file_location(p.stem, p)
        mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
        except Exception:
            continue
        if hasattr(mod, 'descr_list'):
            cases.append((f'{p.parent.name}/{p.stem}', mod))
    return cases


HELPER = re.compile(r'tensorforge::(\w+)|__builtin_amdgcn_(\w+)')

results = {}
helpers = {}
for arch in ARCHS:
    ok = fail = 0
    errs = {}
    for name, mod in load_cases():
        try:
            ctx = Context(arch=arch, backend='hip',
                          fp_type=getattr(mod, 'DTYPE', None))
            gen = Generator(mod.descr_list(), ctx)
            gen.generate()
            src = gen.get_kernel() if hasattr(gen, 'get_kernel') else ''
            for m in HELPER.finditer(str(src)):
                helpers.setdefault(m.group(1) or m.group(2), set()).add(arch)
            ok += 1
        except Exception as exc:
            fail += 1
            key = f'{type(exc).__name__}: {str(exc)[:70]}'
            errs.setdefault(key, []).append(name)
    results[arch] = (ok, fail, errs)

print(f'{"arch":10s} {"ok":>4s} {"fail":>5s}  errors')
for arch, (ok, fail, errs) in results.items():
    print(f'{arch:10s} {ok:4d} {fail:5d}  ', end='')
    if not errs:
        print('-')
    else:
        first = True
        for k, v in sorted(errs.items(), key=lambda x: -len(x[1])):
            pad = '' if first else ' ' * 23
            print(f'{pad}{len(v):3d}x {k}')
            first = False

print()
print('=== emitted helper symbols, by arch coverage ===')
for h, archs in sorted(helpers.items()):
    print(f'  {h:32s} {len(archs):2d} archs  e.g. {sorted(archs)[0]}')
