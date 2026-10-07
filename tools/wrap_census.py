# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Which transfers does `enable_wrap_loads` move, and why not the rest?

The pass (`pir.wrap.wrap_loads`) reports one line per transfer it looks at:
moved, and into what kind of buffer, or left where it is, and why.  This
builds every case of the corpus with the pass on and counts those lines, per
case and target and over all:

  reg     transfers moved into a register array
  shr     transfers moved into a shared window
  left    transfers looked at and left where they are

A loop the pass leaves alone as a whole -- one without a transfer, or one
whose traversal has no first element to peel -- reports that instead of a
line per transfer, and is counted under its reason as well.

Only the first `move_distance` transfers of a body are looked at, so a
larger distance asks about more of them:

    python3 tools/wrap_census.py               # per-case table and the summary
    python3 tools/wrap_census.py --summary     # summary only
    python3 tools/wrap_census.py --distance 2  # at another move distance
    python3 tools/wrap_census.py --target esimd/pvc
"""
import argparse
import contextlib
import importlib.util
import io
import sys
import warnings
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from tensorforge.common.context import Context
from tensorforge.common.options import Options
from tensorforge.generators.generator import Generator

TARGETS = ['cuda/sm_86', 'hip/gfx90a']
ROOT = Path(__file__).resolve().parent.parent
CASES = ROOT / 'tests' / 'cases'


def _load(path):
    spec = importlib.util.spec_from_file_location('tf_wrap__' + path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(mod)
    return mod


def _report(mod, target, distance):
    """The pass's lines for the build the generator settled on, or None
    where the case does not generate."""
    backend, arch = target.split('/')
    ctx = Context(arch=arch, backend=backend,
                  fp_type=getattr(mod, 'DTYPE', None),
                  options=Options(enable_wrap_loads=True,
                                  move_distance=distance))
    try:
        # What a build warns about is not what this counts.
        with contextlib.redirect_stdout(io.StringIO()), \
                warnings.catch_warnings():
            warnings.simplefilter('ignore')
            Generator(mod.descr_list(), ctx,
                      attrs=getattr(mod, 'ATTRS', None)).generate()
    except Exception:
        return None
    return list(ctx.wrap_report or [])


def _reason(line):
    """`- name: why` -> why; `- loop: why` -> `loop: why`."""
    name, _, why = line[2:].partition(': ')
    return f'loop: {why}' if name == 'loop' else why


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--summary', action='store_true')
    ap.add_argument('--distance', type=int, default=1)
    ap.add_argument('--target', action='append', default=None,
                    help='backend/arch, repeatable (default: '
                         + ', '.join(TARGETS) + ')')
    args = ap.parse_args()
    targets = args.target or TARGETS

    moved = Counter()
    reasons = Counter()
    cases = set()
    failed = []
    if not args.summary:
        print(f'{"case":38s} {"target":12s} {"reg":>4s} {"shr":>4s} '
              f'{"left":>5s}')
    for path in sorted(CASES.rglob('*.py')):
        if path.name.startswith('_'):
            continue
        try:
            mod = _load(path)
        except Exception:
            continue
        if not hasattr(mod, 'NAME') or not hasattr(mod, 'descr_list'):
            continue
        for target in targets:
            lines = _report(mod, target, args.distance)
            if lines is None:
                failed.append(f'{mod.NAME} [{target}]')
                continue
            cases.add(mod.NAME)
            reg = sum(1 for l in lines if l.startswith('+ ')
                      and l.endswith('[reg]'))
            shr = sum(1 for l in lines if l.startswith('+ ')
                      and l.endswith('[shr]'))
            left = [l for l in lines if l.startswith('- ')]
            moved['reg'] += reg
            moved['shr'] += shr
            reasons.update(_reason(l) for l in left)
            if not args.summary:
                print(f'{mod.NAME[:38]:38s} {target:12s} {reg:4d} {shr:4d} '
                      f'{len(left):5d}')

    total = moved['reg'] + moved['shr'] + sum(reasons.values())
    print()
    print(f'{total} transfers and loops over {len(cases)} cases, '
          f'move distance {args.distance}')
    print(f'{moved["reg"]:6d} moved into a register array')
    print(f'{moved["shr"]:6d} moved into a shared window')
    print(f'{sum(reasons.values()):6d} left where they are:')
    for why, n in reasons.most_common():
        print(f'{n:6d}   {why}')
    if failed:
        print(f'\ndid not generate: {", ".join(failed)}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
