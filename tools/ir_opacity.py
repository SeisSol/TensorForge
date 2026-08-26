# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Migration progress, by how much a pass can actually see -- and from where.

Two measurements, because they answer different questions and conflating them
is how this tool was wrong once already.

*Constructed* is what the generator emits into the builder. That is the number
attributed to source sites, because a site is what you go and rewrite.

*Lowered* is what survives `pir.optimize` and reaches codegen. That is the
number that says how far along the migration is, and it is smaller: passes
delete raw nodes. `flatten_scopes` alone removes every textless `Op.RAWBLOCK`,
so the 18016 empty scopes `write_loops_inner` used to open counted as 67% of
everything opaque while lowering to no text at all.

Constructing nodes for a pass to delete is still waste, and a region boundary
still constrains the passes that run before the one that removes it. But it is
not opacity in the generated code, and reporting it as such overstates the
work left by a factor of five.

`raw*` is too coarse a bucket: `load_expr` produces a RAWEXPR whose text is
still vendor-specific but whose *result is an SSA value* and whose *memory
effect is declared* -- a pass can reorder around it and reuse it.  A RAWSTMT
with Effect.UNKNOWN can do neither.  Counting them together hides the step.

The per-case totals answer "how far along is this".  They do not answer
"what do I change next", because a percentage is not a work item.  So the
second report attributes every raw node to the function that emitted it, by
walking out of the `pir` and writer frames to the first caller that is
neither.  That names a file and a function, which is a thing one can go and
rewrite.

Two contexts are working on this at once.  The corpus number is the shared
one: if it is quoted anywhere it should come from here rather than from a
subset, so that both sides are talking about the same denominator.

    python3 tools/ir_opacity.py            # per-case table, then the sites
    python3 tools/ir_opacity.py --sites    # sites only
    python3 tools/ir_opacity.py --cases    # per-case only
"""
import importlib.util
import sys
from collections import Counter
from pathlib import Path
from typing import Dict

from tensorforge.common.context import Context
from tensorforge.generators.generator import Generator
from tensorforge.backend import pir
from tensorforge.backend.pir import build as pirbuild
from tensorforge.backend.pir.core import Effect, walk
from tensorforge.backend.instructions import abstract_instruction as absinstr

TARGETS = [('gfx90a', 'hip'), ('sm_86', 'cuda')]

per_case = Counter()
by_site = Counter()
by_op = Counter()
lowered = Counter()
lowered_site = Counter()
# Which site emitted a given piece of raw text.  Attributing *lowered* nodes
# needs this indirection: the passes rebuild statements with `replace`, so the
# emitting frame is long gone by the time `optimize` returns, and putting the
# site in `attrs` would change behaviour -- `flatten_scopes` keys on `attrs`
# being empty.  Text is stable across `replace` and specific enough in
# practice; where two sites emit identical text, the more frequent one wins,
# which is the right guess and a rare case.
text_site: Dict[str, Counter] = {}


def _classify(stmt):
    """`raw*` is too coarse, and so was the two-way split that replaced it.

    Three questions, not one.  *Can a pass reason about this as code?*  No, for
    anything raw.  *Does it pin everything around it?*  Only if its effect is
    `UNKNOWN` or it declares no accesses while claiming one.  *Does it cost
    anything at all?*  A comment has `Effect.NONE`, no accesses and
    `movable=True`: it constrains nothing, reorders freely and lowers to a line
    the compiler discards.

    Counting comments as opaque put `compute/__init__.py:gen_ir` at the head of
    the work list with 1092 nodes, 23% of everything -- and rewriting that site
    would have bought exactly nothing, because the nodes it emits are the
    `sink.Comment(self.__str__())` on line 19.  Corpus-wide that is 686 of 4735
    raw nodes, 14.5%, all inert.

    So `inert` is its own bucket and does not count against the migration.  Of
    what remains, `blocking` is what a scheduler cannot move across and
    `declared` is what it can.
    """
    if stmt.op not in ('rawstmt', 'rawexpr', 'rawblock'):
        return 'structured'
    if stmt.effect & Effect.UNKNOWN:
        return 'blocking'
    if not stmt.accesses:
        if stmt.effect == Effect.NONE and stmt.movable:
            return 'inert'
        return 'blocking'
    return 'declared'


_orig_optimize = pir.optimize


def _counting_optimize(body):
    """Count what reaches codegen, which is what `optimize` returns."""
    out = _orig_optimize(body)
    for stmt, _ in walk(out):
        kind = _classify(stmt)
        lowered[kind] += 1
        if kind != 'structured':
            sites = text_site.get(stmt.text or '')
            site = sites.most_common(1)[0][0] if sites else '?'
            lowered_site[(site, kind)] += 1
    return out


pir.optimize = _counting_optimize
absinstr.pir.optimize = _counting_optimize


def _site():
    """The first frame that is neither the IR builder nor a writer wrapper.

    `emit` is called from inside `build.py` in every case, and often through
    one of the writer shims, so the immediate caller is always the same two
    or three names and tells us nothing.  Walking out of those reaches the
    generator code that decided to emit text, which is the thing that has to
    change for the node to stop being raw.
    """
    f = sys._getframe(2)
    while f is not None:
        name = f.f_code.co_filename
        if '/pir/' not in name and 'writer' not in name:
            return f'{Path(name).name}:{f.f_code.co_name}'
        f = f.f_back
    return '?'


_orig_emit = pirbuild.IRBuilder.emit


def emit(self, stmt):
    kind = _classify(stmt)
    per_case[kind] += 1
    if kind != 'structured':
        site = _site()
        by_site[(site, kind)] += 1
        by_op[stmt.op] += 1
        text_site.setdefault(stmt.text or '', Counter())[site] += 1
    return _orig_emit(self, stmt)


pirbuild.IRBuilder.emit = emit


def _cases():
    """Every case the snapshot harness sees, which is not every case in the
    top level of `tests/cases`.

    `conftest.py` walks recursively; `barrier/`, `elementwise/`, `reduction/`
    and `slicing/` hold 24 further cases between them.  Reporting a corpus
    number off a top-level glob understates it by almost half, and this number
    is meant to be the one two contexts quote at each other.
    """
    root = Path('tests/cases')
    return [p for p in sorted(root.rglob('*.py'))
            if '__pycache__' not in p.parts]


def _load(path):
    spec = importlib.util.spec_from_file_location('case', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main(argv):
    want_cases = '--sites' not in argv
    want_sites = '--cases' not in argv

    cases = _cases()
    total = Counter()
    rows = []
    failed = []

    for path in cases:
        for arch, backend in TARGETS:
            per_case.clear()
            try:
                mod = _load(path)
                ctx = Context(arch=arch, backend=backend,
                              fp_type=getattr(mod, 'DTYPE', None))
                Generator(mod.descr_list(), ctx).generate()
            except Exception as exc:
                # A case that does not generate still contributes whatever it
                # emitted before it stopped.  Dropping it would move the total
                # every time an unrelated defect is fixed or introduced.
                failed.append((path.stem, backend, type(exc).__name__))
            total.update(per_case)
            rows.append((path.stem, arch, Counter(per_case)))

    if want_cases:
        print(f'{"case":34s} {"arch":8s} {"total":>7s} '
              f'{"blocking":>14s} {"declared":>14s} {"inert":>14s} '
              f'{"structured":>14s}')
        for name, arch, c in rows:
            t = sum(c.values())
            if t == 0:
                continue
            cells = [f'{c[k]:6d} ({100 * c[k] / t:4.1f}%)'
                     for k in ('blocking', 'declared', 'inert', 'structured')]
            print(f'{name:34s} {arch:8s} {t:7d} '
                  f'{cells[0]:>14s} {cells[1]:>14s} {cells[2]:>14s} '
                  f'{cells[3]:>14s}')
        print()

    t = sum(total.values())
    raw = total['blocking'] + total['declared'] + total['inert']
    lt = sum(lowered.values()) or 1
    print(f'corpus: {len(cases)} cases x {len(TARGETS)} targets, '
          f'{len(cases) * len(TARGETS) - len(failed)} generated, '
          f'{len(failed)} failed')
    print(f'{"":14s}{"constructed":>22s}{"lowered":>22s}')
    print(f'  {"nodes":11s} {t:9d}{"":12s} {lt:9d}')
    for k in ('structured', 'declared', 'inert', 'blocking'):
        print(f'  {k:11s} {total[k]:9d} ({100 * total[k] / t:5.1f}%) '
              f'{lowered[k]:9d} ({100 * lowered[k] / lt:5.1f}%)')
    print(f'  raw by op (constructed)  {dict(by_op)}')
    if failed:
        print('  did not generate: '
              + ', '.join(f'{n}/{b} ({e})' for n, b, e in failed))

    if want_sites:
        lraw = lowered['blocking'] + lowered['declared'] + lowered['inert']
        print()
        print('raw nodes that REACH CODEGEN, by emitting site '
              '-- this is the work left:')
        print(f'{"site":46s} {"kind":10s} {"count":>8s} {"cum%":>7s}')
        cum = 0
        for (site, kind), n in lowered_site.most_common(18):
            cum += n
            print(f'{site:46s} {kind:10s} {n:8d} {100 * cum / lraw:6.1f}%')
        print()
        print('constructed, including what the passes then delete '
              '-- this is build cost, not opacity:')
        print(f'{"site":46s} {"kind":10s} {"count":>8s} {"cum%":>7s}')
        cum = 0
        for (site, kind), n in by_site.most_common(10):
            cum += n
            print(f'{site:46s} {kind:10s} {n:8d} {100 * cum / raw:6.1f}%')


if __name__ == '__main__':
    main(sys.argv[1:])
