# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Which macro-owned names still span more than one PIR body?

A PIR value connects a definition to its uses.  A C++ name does the same job,
badly, and is needed exactly when the definition and the uses are built into
*different* IRBuilder instances: there is no value to pass, so the only thing
they can share is text.

With one body per loop body, what still needs a name is what outlives one:
the shared arena and its scratch tail, and the tiles of the cases with two
batch loops (`barrier_two_gemms`, `fence_two_gemms`).  This counts them over
the corpus, so that moving the kernel skeleton into the IR has a number to
bring down.

    python3 tools/buffer_spans.py
"""
import contextlib
import importlib.util
import io
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from tensorforge.common.context import Context
from tensorforge.generators.generator import Generator
from tensorforge.backend.pir import build as pirbuild

# names the macro layer owns: register tiles r0.., shared tiles s0.., the
# arena and its scratch tail, and the rolling/peeled pointers
OWNED = re.compile(r'\b((?:r|s)\d+(?:_w)?|localShrMem\d+|totalShrMem|'
                   r'tempShrMem|pipe_\w+|peel_\w+)\b')

# builder id -> set of owned names it mentions
per_body = defaultdict(set)
_orig_emit = pirbuild.IRBuilder.emit


def emit(self, stmt):
    text = stmt.text or ''
    if text:
        for m in OWNED.findall(text):
            per_body[id(self)].add(m)
    return _orig_emit(self, stmt)


pirbuild.IRBuilder.emit = emit


def run_case(path):
    global per_body
    spec = importlib.util.spec_from_file_location('bs_' + path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            spec.loader.exec_module(mod)
    except Exception:
        return None
    if not hasattr(mod, 'NAME') or not hasattr(mod, 'descr_list'):
        return None
    out = {}
    for arch, backend in (('gfx90a', 'hip'), ('sm_86', 'cuda')):
        per_body = defaultdict(set)
        try:
            ctx = Context(arch=arch, backend=backend,
                          fp_type=getattr(mod, 'DTYPE', None))
            with contextlib.redirect_stdout(io.StringIO()):
                Generator(mod.descr_list(), ctx).generate()
        except Exception:
            continue
        bodies = defaultdict(int)
        for names in per_body.values():
            for n in names:
                bodies[n] += 1
        out[backend] = dict(bodies)
    return out


spread = Counter()
total_names = 0
worst = []
for path in sorted(Path('tests/cases').rglob('*.py')):
    if path.name.startswith('_'):
        continue
    res = run_case(path)
    if not res:
        continue
    for backend, bodies in res.items():
        for name, count in bodies.items():
            total_names += 1
            spread[min(count, 5)] += 1
            if count > 1:
                worst.append((count, path.stem, backend, name))

print(f'{total_names} buffer occurrences over the corpus\n')
print('bodies that mention the same buffer:')
for k in sorted(spread):
    label = f'{k}' if k < 5 else '5+'
    share = 100 * spread[k] / total_names
    tag = '  <- needs no name once consumers are migrated' if k == 1 else ''
    print(f'  {label:>3s} body/bodies: {spread[k]:5d}  ({share:4.1f}%){tag}')

contained = spread[1]
print(f'\ncontained in one body: {contained}/{total_names} '
      f'({100 * contained / total_names:.1f}%)')
if worst:
    print('\nmost spread out:')
    for count, case, backend, name in sorted(worst, reverse=True)[:8]:
        print(f'  {name:16s} {count:2d} bodies   {case} [{backend}]')

print('\nnames that still span, by kind:')
kind = Counter()
for count, case, backend, name in worst:
    if name.startswith(('localShrMem', 'totalShrMem', 'tempShrMem')):
        kind['kernel-scope arena / scratch tail'] += 1
    elif name.startswith(('pipe_', 'peel_')):
        kind['pipeline pointer (crosses the loop by design)'] += 1
    elif name.startswith('r'):
        kind['register tile'] += 1
    elif name.startswith('s'):
        kind['shared tile'] += 1
for k, v in kind.most_common():
    print(f'  {v:4d}  {k}')
