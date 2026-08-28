# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""What of a kernel never reaches the pseudo-IR at all.

`ir_opacity.py` measures how much of what *is* in a body is raw.  This measures
the other gap: statements the Writer emits directly, which no body ever
contained and no pass can therefore see.

The distinction matters for one reason.  A pass may reorder statements inside a
body; the reorder predicate in `pir/schedule.py` says when.  What no pass can
do is move a statement across a construct that is not in the IR --- and the
batch loop is exactly that.  `BatchLoop.gen_code_inner` opens `writer.For(...)`
and emits the body inside it, so the loop, its back edge, its induction
variable and its lookahead bindings are all Writer text.  A body sits *inside*
the loop and has no way to refer to it.

That is the whole reason `WrapLoads` lives at macro level: moving a transfer to
the previous iteration means peeling a prologue and rewriting an index to
`batchId1`, and neither is expressible against a loop the IR cannot name.

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

from tensorforge.backend import pir
from tensorforge.backend.pir import build as pirbuild
from tensorforge.backend.pir.core import walk
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

    def patched_emit(body, writer, context=None):
        before = writer.get_src()
        orig_emit(body, writer, context)
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
    print('\nThe remainder is the section scaffolding: the batch loop and its '
          'header,\nthe flag guard, the lookahead bindings and the stage '
          'counter.  A pass can\nreorder inside a body; it cannot move '
          'anything across a loop it cannot name.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
