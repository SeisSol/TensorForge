# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""What does each staged shared image actually accomplish?

A shared buffer between two stages of a kernel is there to *move* data: the
fill spreads an operand one way over the lanes and the reads take it back
another way.  Which way each is, is now recorded -- `Symbol._note_layout` and
`_record_linear_layout` state how a fill distributes an image, and
`layout_of` says how a read wants it -- so the pair can be compared, and the
comparison says what the buffer is for:

* **broadcast** -- filled distributed, read replicated.  Every lane wants the
  same element, which another lane holds.  Under SPMD that is what shared
  memory is for; under an explicit vector it is `v[k]`, an element read out of
  this work-item's own registers, and the buffer moves nothing.
* **relayout** -- filled over one lane count, read over another.  A real
  redistribution: the element a lane wants is in a different lane, and no
  register read reaches it.
* **round trip** -- filled and read the same way.  Written and read back with
  the same lane assignment, so it accomplishes nothing at all except spilling.
* **unknown** -- one side has no claim.  Not a category, a gap: see
  `Symbol._note_layout` for which fill paths record one.

Only the first and third are candidates for elimination, and only the third
unconditionally -- turning a broadcast into a register read is what the
`compute/broadcast.py` chain does, and whether that is cheaper than the buffer
is the register-pressure question `pressure(in_bytes=True)` answers.

Run:  python tools/staging_census.py [arch] [backend]
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import sys
import warnings
from collections import Counter
from pathlib import Path

import tensorforge.backend.symbol as sym
from tensorforge.common.basic_types import Datatype
from tensorforge.common.context import Context
from tensorforge.generators.generator import Generator


def classify(read, fill):
    if fill is None or read is None:
        return 'unknown'
    if read == fill:
        return 'round trip'
    if not read.is_distributed:
        return 'broadcast'
    return 'relayout'


def instrument():
    """Watch the fills record and the reads ask, and pair them by symbol."""
    fills = {}
    counts = Counter()
    per_case = Counter()
    state = {'case': ''}

    note = sym.Symbol._note_layout

    def noted(self, layout, writer=None):
        note(self, layout, writer)
        if layout is not None:
            fills[id(self)] = layout

    sym.Symbol._note_layout = noted

    record = sym.Symbol._record_linear_layout

    def recorded(self, index, vec, threads=None, writer=None):
        record(self, index, vec, threads, writer)
        if self.layout is not None:
            fills[id(self)] = self.layout

    sym.Symbol._record_linear_layout = recorded

    load = sym.Symbol.load

    def loaded(self, writer, context, variable, index, *a, **k):
        out = load(self, writer, context, variable, index, *a, **k)
        if self.stype is sym.SymbolType.SharedMem:
            kind = classify(sym.layout_of(index, self.num_threads),
                            fills.get(id(self)))
            counts[kind] += 1
            per_case[(state['case'], kind)] += 1
        return out

    sym.Symbol.load = loaded
    return counts, per_case, state


def main():
    arch = sys.argv[1] if len(sys.argv) > 1 else 'pvc'
    backend = sys.argv[2] if len(sys.argv) > 2 else 'esimd'
    counts, per_case, state = instrument()
    warnings.simplefilter('ignore')

    for path in sorted(Path('tests/cases').rglob('*.py')):
        try:
            spec = importlib.util.spec_from_file_location(path.stem, path)
            case = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(case)
            if not hasattr(case, 'descr_list'):
                continue
        except Exception:
            continue
        state['case'] = path.stem
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                Generator(case.descr_list(),
                          Context(arch=arch, backend=backend,
                                  fp_type=getattr(case, 'DTYPE',
                                                  Datatype.F32))).generate()
        except Exception:
            pass

    total = sum(counts.values()) or 1
    print(f'{arch}/{backend}: reads of a staged shared image\n')
    for kind in ('broadcast', 'relayout', 'round trip', 'unknown'):
        n = counts[kind]
        print(f'  {n:7d}  {100 * n / total:5.1f}%  {kind}')

    print('\nthe cases that stage the most:')
    worst = Counter()
    for (case, _), n in per_case.items():
        worst[case] += n
    for case, n in worst.most_common(8):
        parts = ', '.join(f'{k} {per_case[(case, k)]}'
                          for k in ('broadcast', 'relayout', 'round trip',
                                    'unknown')
                          if per_case[(case, k)])
        print(f'  {n:6d}  {case[:34]:34s} {parts}')


if __name__ == '__main__':
    main()
