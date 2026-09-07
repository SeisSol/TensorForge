# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Would keeping a staged image in registers fit?

`tools/staging_census.py` says what each shared buffer *accomplishes*: 86% of
the reads on the ESIMD path are broadcasts, which under an explicit vector are
an element read out of the work-item's own registers and need no buffer at
all.  That is the case for eliminating them.

This is the case against, and the two have to be weighed rather than one of
them assumed.  A buffer that disappears does not free its contents -- the
values it held have to stay live in registers from the fill to the last read
instead, and the work-item already holds the whole tile.  11 of the corpus's
kernels are over the 8 kB a PVC thread gets before anything moves.

So per case:

    shared        bytes the staged images occupy
    eliminable    of those, the ones every read broadcasts from
    registers     peak register bytes today (`pressure(in_bytes=True)`)
    projected     registers + eliminable, the upper bound if they all move
    verdict       against `max_reg_per_thread`

`projected` is an upper bound and deliberately so.  It assumes every
eliminable image is live at the same moment as the register peak, which is the
worst case: the fills and the peak may not overlap, and two images may not
overlap each other.  A tighter figure needs the live ranges, and a bound that
can only overstate is the right one to refuse on -- it never approves a move
that then spills.

Run:  python tools/register_tradeoff.py [arch] [backend]
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import sys
import warnings
from collections import Counter
from pathlib import Path

import tensorforge.backend.pir as pir
import tensorforge.backend.symbol as sym
from tensorforge.common.basic_types import Datatype
from tensorforge.common.context import Context
from tensorforge.generators.generator import Generator
sys.path.insert(0, str(Path(__file__).resolve().parent))
from staging_census import classify  # noqa: E402


def _volume(view) -> int:
    total = 1
    for extent in (getattr(view, 'shape', None) or []):
        total *= extent
    return total


def instrument():
    """Pair each staged image's fill with its reads, and size it."""
    state = {'case': '', 'peak': 0}
    fills = {}
    #: symbol id -> (name, bytes, {kind: count})
    images = {}

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
            entry = images.setdefault(
                (state['case'], id(self)),
                [self.name,
                 _volume(self.data_view) * self.get_fptype().size(),
                 Counter()])
            entry[2][classify(sym.layout_of(index, self.num_threads),
                              fills.get(id(self)))] += 1
        return out

    sym.Symbol.load = loaded

    optimize = pir.optimize

    def optimized(body, **kw):
        out = optimize(body, **kw)
        state['peak'] = max(state['peak'],
                            pir.pressure(out, in_bytes=True,
                                         explicit_simd=kw.get('explicit_simd',
                                                              True)))
        return out

    pir.optimize = optimized
    return images, state


def main():
    arch = sys.argv[1] if len(sys.argv) > 1 else 'pvc'
    backend = sys.argv[2] if len(sys.argv) > 2 else 'esimd'
    images, state = instrument()
    warnings.simplefilter('ignore')

    budget = None
    rows = []
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
        state['peak'] = 0
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                ctx = Context(arch=arch, backend=backend,
                              fp_type=getattr(case, 'DTYPE', Datatype.F32))
                Generator(case.descr_list(), ctx).generate()
            budget = getattr(ctx.get_vm().get_hw_descr(),
                             'max_reg_per_thread', None)
        except Exception:
            continue
        mine = [v for (c, _), v in images.items() if c == path.stem]
        shared = sum(v[1] for v in mine)
        # Eliminable: every read of the image is a broadcast or a round trip.
        # One genuine relayout is enough to keep it -- the element a lane wants
        # then lives in another lane, and no register read reaches it.
        gone = sum(v[1] for v in mine
                   if v[2] and not v[2]['relayout'] and not v[2]['unknown'])
        rows.append((state['peak'] + gone, state['peak'], shared, gone,
                     path.stem))

    rows.sort(reverse=True)
    if budget is None:
        # Absent means "not stated for this target", not "unlimited" -- see
        # `HwDescription.max_reg_per_thread`.  Comparing against a default
        # would answer a question nobody asked, and answer it wrongly in both
        # directions at once.
        print(f'{arch}/{backend}: no per-thread register budget is stated for '
              f'this target, so there is nothing to weigh the move against.')
        return
    cap = budget
    print(f'{arch}/{backend}: budget {cap} B per work-item\n')
    print(f"{'projected':>9} {'registers':>9} {'shared':>7} {'movable':>7}  case")
    fits = pushed = already = 0
    for projected, peak, shared, gone, name in rows:
        if not shared and peak <= cap:
            continue
        if peak > cap:
            verdict, already = 'already over', already + 1
        elif projected > cap:
            verdict, pushed = 'PUSHED OVER', pushed + 1
        else:
            verdict, fits = 'fits', fits + 1
        print(f'{projected:9d} {peak:9d} {shared:7d} {gone:7d}  '
              f'{name[:28]:28s} {verdict}')

    # Three answers, not two, and the middle one is the only place the move is
    # actually a decision.
    #
    # `already over` means the kernel spills before anything moves -- the
    # staged buffers are 8 kB beside a 44 kB peak, so eliminating them changes
    # nothing that matters.  Whatever is done for those is about how much of
    # the operator one work-item owns, which is not this question.
    print(f'\n  {fits:3d}  fit after the move -- the staging is free to go')
    print(f'  {pushed:3d}  fit today and would not after -- leave them staged')
    print(f'  {already:3d}  over the budget already -- the staging is not why')


if __name__ == '__main__':
    main()
