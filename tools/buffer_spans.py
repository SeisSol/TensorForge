# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Does `r0` need a name, or does it need a shorter body?

A PIR value connects a definition to its uses.  A C++ name does the same job,
badly, and is needed exactly when the definition and the uses are built into
*different* IRBuilder instances: there is no value to pass, so the only thing
they can share is text.

So "pin the name, or migrate the consumers" is not a matter of taste, it is a
count.  Over the corpus, 590 buffer occurrences:

                          one body    2+ bodies
    per macro-op body       39.3%        60.7%
    per loop body           89.7%        10.3%

Per macro-op --- today's default --- a majority of buffers are named because
`RegisterAlloc` declares in one body, the loader writes in a second and the
multilinear reads in a third.  Pinning names there would be permanent: three
fifths of every buffer in the corpus would keep a macro-owned name forever,
and `symbol.py`'s access helpers would keep building `f'{self.name}[...]'`
because there would be nothing else to build.

Per loop body the same buffers need nothing.  What is left is 49 arena and
scratch-tail occurrences, 2 shared tiles, and 10 register tiles --- and the 10
are all `barrier_two_gemms` and `fence_two_gemms`, which have two batch loops
apiece.  Every name still required in wide mode belongs to something that
outlives one loop body.  None is required by how the macro layer is factored.

That is the argument for migrating the consumers rather than pinning the name,
and for the order: wide bodies first, or the migration has nothing to migrate
to.  A pinning mechanism is still wanted afterwards, for the arena --- but
scoped to a resource that genuinely is kernel-scope, rather than as a general
bridge that would let every value keep its name.

    python3 tools/buffer_spans.py                  # the default: one body per loop
    TF_IR_WIDE=0 python3 tools/buffer_spans.py     # force one body per macro-op
"""
import contextlib
import importlib.util
import io
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

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

mode = 'narrow (one body per macro-op)' if os.environ.get('TF_IR_WIDE') in ('0','false','False') \
    else 'WIDE (one body per loop body, the default)'
print(f'{mode}: {total_names} buffer occurrences over the corpus\n')
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
