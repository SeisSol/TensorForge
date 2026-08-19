# SPDX-License-Identifier: MIT
"""Migration progress, by how much a pass can actually see -- and from where.

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

from tensorforge.common.context import Context
from tensorforge.generators.generator import Generator
from tensorforge.backend.pir import build as pirbuild
from tensorforge.backend.pir.core import Effect

TARGETS = [('gfx90a', 'hip'), ('sm_86', 'cuda')]

per_case = Counter()
by_site = Counter()
by_op = Counter()


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
    if stmt.op in ('rawstmt', 'rawexpr', 'rawblock'):
        # A declared access is one a pass can reason about: it names the
        # symbol it touches and the effect it has.  UNKNOWN, or no access
        # list at all, means the pass has to assume the worst and stop.
        opaque = bool(stmt.effect & Effect.UNKNOWN) or not stmt.accesses
        kind = 'opaque' if opaque else 'declared'
        per_case[kind] += 1
        by_site[(_site(), kind)] += 1
        by_op[stmt.op] += 1
    else:
        per_case['structured'] += 1
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
              f'{"opaque":>14s} {"declared":>14s} {"structured":>14s}')
        for name, arch, c in rows:
            t = sum(c.values())
            if t == 0:
                continue
            cells = [f'{c[k]:6d} ({100 * c[k] / t:4.1f}%)'
                     for k in ('opaque', 'declared', 'structured')]
            print(f'{name:34s} {arch:8s} {t:7d} '
                  f'{cells[0]:>14s} {cells[1]:>14s} {cells[2]:>14s}')
        print()

    t = sum(total.values())
    raw = total['opaque'] + total['declared']
    print(f'corpus: {len(cases)} cases x {len(TARGETS)} targets, '
          f'{len(cases) * len(TARGETS) - len(failed)} generated, '
          f'{len(failed)} failed')
    print(f'  nodes       {t:7d}')
    for k in ('structured', 'declared', 'opaque'):
        print(f'  {k:11s} {total[k]:7d} ({100 * total[k] / t:4.1f}%)')
    print(f'  raw by op   {dict(by_op)}')
    if failed:
        print('  did not generate: '
              + ', '.join(f'{n}/{b} ({e})' for n, b, e in failed))

    if want_sites:
        print()
        print(f'{"site":46s} {"kind":10s} {"count":>8s} {"cum%":>7s}')
        cum = 0
        for (site, kind), n in by_site.most_common():
            cum += n
            print(f'{site:46s} {kind:10s} {n:8d} {100 * cum / raw:6.1f}%')


if __name__ == '__main__':
    main(sys.argv[1:])
