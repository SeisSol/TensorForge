# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""What of a kernel never reaches the pseudo-IR at all.

`ir_opacity.py` measures how much of what *is* in a body is raw.  This measures
the other gap: statements the Writer emits directly, which no body ever
contained and no pass can therefore see.

The distinction matters because a pass moves a statement only inside a body,
and only across what the body contains.  The batch loop is part of the
section's body: its index, the successor and the first element it names and
the per-element flag guard are statements, which is what lets
`enable_wrap_loads` move a transfer across the loop's back edge
(`pir/wrap.py`).  What is left outside is the frame around the bodies -- the
signature, the launch bounds, the declaration of dynamic shared memory -- and
a number that grows here is something the passes have stopped seeing.

    python3 tools/macro_surface.py
"""
import argparse
import contextlib
import importlib.util
import io
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from tensorforge.backend import pir
from tensorforge.backend.instructions import abstract_instruction as _ai
from tensorforge.common.context import Context
from tensorforge.generators.generator import Generator

ROOT = Path(__file__).resolve().parent.parent
CASES = ROOT / 'tests' / 'cases'
TARGETS = [('sm_86', 'cuda'), ('gfx90a', 'hip')]

#: Lines that are scaffolding rather than kernel work, and would be noise.
_SKIP = re.compile(r'^\s*(//|\}|\{|$|#include|#pragma once)')


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()

    emitted = []            # text every PIR body produced
    orig_emit = pir.emit

    def patched_emit(body, writer, context=None, metrics=None):
        before = writer.get_src()
        orig_emit(body, writer, context, metrics)
        emitted.append(writer.get_src()[len(before):])

    pir.emit = patched_emit
    _ai.pir.emit = patched_emit

    totals = Counter()
    rows = []
    paths = sorted(CASES.rglob('*.py'))
    if args.limit:
        paths = paths[:args.limit]
    for path in paths:
        if path.name.startswith('_'):
            continue
        spec = importlib.util.spec_from_file_location('ms_' + path.stem, path)
        mod = importlib.util.module_from_spec(spec)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                spec.loader.exec_module(mod)
        except Exception:
            continue
        if not hasattr(mod, 'NAME') or not hasattr(mod, 'descr_list'):
            continue
        for arch, backend in TARGETS:
            emitted.clear()
            try:
                ctx = Context(arch=arch, backend=backend,
                              fp_type=getattr(mod, 'DTYPE', None))
                with contextlib.redirect_stdout(io.StringIO()):
                    gen = Generator(mod.descr_list(), ctx)
                    gen.generate()
                kernel = gen.get_kernel() or ''
            except Exception:
                continue
            if not kernel:
                continue
            through_pir = sum(1 for chunk in emitted
                              for line in chunk.splitlines()
                              if not _SKIP.match(line))
            total = sum(1 for line in kernel.splitlines()
                        if not _SKIP.match(line))
            outside = max(total - through_pir, 0)
            totals['total'] += total
            totals['pir'] += through_pir
            totals['outside'] += outside
            rows.append((mod.NAME, backend, total, through_pir, outside))

    pir.emit = orig_emit
    _ai.pir.emit = orig_emit

    print(f'{"case":34s} {"be":5s} {"lines":>7s} {"in PIR":>8s} '
          f'{"outside":>8s}')
    for name, backend, total, inside, outside in rows[:25]:
        print(f'{name[:34]:34s} {backend:5s} {total:7d} {inside:8d} '
              f'{outside:8d}')
    if len(rows) > 25:
        print(f'... {len(rows) - 25} more')

    t, p, o = totals['total'], totals['pir'], totals['outside']
    print(f'\n{len(rows)} kernels, {t} lines of body')
    print(f'  through a PIR body: {p} ({100 * p / max(t, 1):.1f}%)')
    print(f'  emitted directly:   {o} ({100 * o / max(t, 1):.1f}%)')
    print('\nThe remainder is the frame around the bodies: the signature, the '
          'launch\nbounds and the declaration of dynamic shared memory.  A '
          'pass moves statements\ninside a body; what is out here, none of '
          'them sees.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
