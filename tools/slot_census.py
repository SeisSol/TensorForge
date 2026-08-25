# SPDX-License-Identifier: MIT
"""How many slots does a loop body actually have?

The wrap-around schedule buys prefetch distance at no extra staging space up to
``d = n - 1``, where ``n`` is the number of compute slots in the body.  Whether
that is worth building depends entirely on the distribution of ``n`` over real
kernels: at ``n = 1`` it buys nothing that ``Pipeline`` does not already do, at
``n = 20`` it buys nineteen slots of cover for free.

So this counts, per case and backend:

  n         compute slots in the batch-loop body
  xfer      global->local transfers in that body
  shr/reg   how many of those write shared memory vs. registers
  dist      slots between issue and first consumer, as scheduled today
  bar       barriers in the body -- the cost that scales with n if the
            staging is in shared memory, and is zero if it is in registers

The last column is the reason the register path matters.  A wrapped write to a
shared buffer needs a barrier against the read it is overtaking, so a
slot-granular schedule costs one barrier per slot where double buffering costs
two per body.  A thread-private register buffer needs none.

    python3 tools/slot_census.py             # per-case table and the summary
    python3 tools/slot_census.py --summary   # summary only
    python3 tools/slot_census.py --case NAME # one case, with per-transfer detail
"""
import argparse
import contextlib
import importlib.util
import io
import sys
from collections import Counter
from pathlib import Path

from tensorforge.common.context import Context
from tensorforge.generators.generator import Generator
from tensorforge.backend.opt import OptimizationStage
from tensorforge.backend.opt.slots import SlotModel, Transfer, models_for

TARGETS = [('sm_86', 'cuda'), ('gfx90a', 'hip')]
ROOT = Path(__file__).resolve().parent.parent
CASES = ROOT / 'tests' / 'cases'


def _capture_models():
    """Hook OptimizationStage so every optimised stream is modelled.

    The stream is only assembled inside `Generator._generate_kernel`, and the
    BatchLoop's region is only populated there.  Rather than re-deriving the
    build, wrap `optimize` and read the result it already has.
    """
    captured = []
    original = OptimizationStage.optimize

    def patched(self):
        original(self)
        captured.extend(models_for(self.get_instructions()))

    OptimizationStage.optimize = patched
    return captured, (lambda: setattr(OptimizationStage, 'optimize', original))


def _load(path):
    spec = importlib.util.spec_from_file_location('tf_slot__' + path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--summary', action='store_true')
    ap.add_argument('--case', default=None)
    args = ap.parse_args()

    rows = []
    hist = Counter()
    detail = []
    for path in sorted(CASES.rglob('*.py')):
        if path.name.startswith('_'):
            continue
        try:
            mod = _load(path)
        except Exception:
            continue
        if not hasattr(mod, 'NAME') or not hasattr(mod, 'descr_list'):
            continue
        if args.case and mod.NAME != args.case:
            continue
        for arch, backend in TARGETS:
            captured, restore = _capture_models()
            try:
                ctx = Context(arch=arch, backend=backend,
                              fp_type=getattr(mod, 'DTYPE', None))
                with contextlib.redirect_stdout(io.StringIO()):
                    Generator(mod.descr_list(), ctx).generate()
            except Exception:
                continue
            finally:
                restore()
            for model in captured:
                rows.append((mod.NAME, backend, model))
                hist[model.n] += 1
                if args.case:
                    detail.append((mod.NAME, backend, model))

    if not args.summary:
        print(f'{"case":38s} {"be":5s} {"n":>3s} {"xfer":>5s} {"shr":>4s} '
              f'{"reg":>4s} {"dist":>10s} {"bar":>4s} {"free_d":>7s}')
        for name, backend, m in rows:
            dists = [t.distance for t in m.transfers if t.distance is not None]
            span = ('-' if not dists
                    else (f'{min(dists)}' if min(dists) == max(dists)
                          else f'{min(dists)}..{max(dists)}'))
            print(f'{name[:38]:38s} {backend:5s} {m.n:3d} '
                  f'{len(m.transfers):5d} '
                  f'{sum(1 for t in m.transfers if t.shared):4d} '
                  f'{sum(1 for t in m.transfers if not t.shared):4d} '
                  f'{span:>10s} {m.barriers:4d} {m.free_distance:7d}')

    for name, backend, m in detail:
        print(f'\n--- {name} [{backend}] ---')
        print(m.report())
        for d in (1, 2, 4, max(m.n - 1, 1), m.n):
            c = m.cost(d)
            print(f'  d={d:<3d} copies={c["copies"]} wrapped={c["wrapped"]} '
                  f'(shared {c["shared_wrapped"]})')

    print(f'\n{len(rows)} batch loops over '
          f'{len({r[0] for r in rows})} cases')
    print('slot count n:  ' + '  '.join(
        f'n={k}:{v}' for k, v in sorted(hist.items())))
    per_backend = Counter()
    shr = Counter()
    for _, backend, m in rows:
        per_backend[backend] += m.n
        shr[backend] += sum(1 for t in m.transfers if t.shared)
    for backend in ('cuda', 'hip'):
        loops = [m for _, b, m in rows if b == backend]
        if not loops:
            continue
        print(f'{backend}: mean n={sum(m.n for m in loops)/len(loops):.2f}  '
              f'max n={max(m.n for m in loops)}  '
              f'shared transfers={shr[backend]}  '
              f'register transfers='
              f'{sum(1 for m in loops for t in m.transfers if not t.shared)}  '
              f'barriers={sum(m.barriers for m in loops)}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
