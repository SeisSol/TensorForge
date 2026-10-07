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
  2st     of those, the ones with a second stage (`--multibuffer`)
  left    transfers looked at and left where they are
  shared  the launch's shared memory, in bytes

A loop the pass leaves alone as a whole -- one without a transfer, or one
whose traversal has no first element to peel -- reports that instead of a
line per transfer, and is counted under its reason as well.  So is a shared
transfer that keeps one stage where two were asked for.

Only the first `move_distance` transfers of a body are looked at, so a
larger distance asks about more of them.  What a second stage costs is the
shared column of two runs, one with `--multibuffer` and one without:

    python3 tools/wrap_census.py               # per-case table and the summary
    python3 tools/wrap_census.py --summary     # summary only
    python3 tools/wrap_census.py --distance 2  # at another move distance
    python3 tools/wrap_census.py --multibuffer # with a second stage
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


def _build(mod, target, options):
    """The pass's lines and the shared bytes for the build the generator
    settled on, or None where the case does not generate."""
    backend, arch = target.split('/')
    ctx = Context(arch=arch, backend=backend,
                  fp_type=getattr(mod, 'DTYPE', None),
                  options=Options(**options))
    try:
        # What a build warns about is not what this counts.
        with contextlib.redirect_stdout(io.StringIO()), \
                warnings.catch_warnings():
            warnings.simplefilter('ignore')
            gen = Generator(mod.descr_list(), ctx,
                            attrs=getattr(mod, 'ATTRS', None))
            gen.generate()
    except Exception:
        return None
    launch = gen.launch_config()
    return list(ctx.wrap_report or []), launch.shared_bytes if launch else 0


def _reason(line):
    """`- name: why` -> why; `- loop: why` -> `loop: why`."""
    name, _, why = line[2:].partition(': ')
    return f'loop: {why}' if name == 'loop' else why


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--summary', action='store_true')
    ap.add_argument('--distance', type=int, default=1)
    ap.add_argument('--multibuffer', action='store_true')
    ap.add_argument('--target', action='append', default=None,
                    help='backend/arch, repeatable (default: '
                         + ', '.join(TARGETS) + ')')
    args = ap.parse_args()
    targets = args.target or TARGETS
    options = dict(enable_wrap_loads=True, move_distance=args.distance,
                   enable_multibuffer=args.multibuffer)

    moved = Counter()
    reasons = Counter()
    single = Counter()
    shared = 0
    cases = set()
    failed = []
    if not args.summary:
        print(f'{"case":38s} {"target":12s} {"reg":>4s} {"shr":>4s} '
              f'{"2st":>4s} {"left":>5s} {"shared":>7s}')
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
            built = _build(mod, target, options)
            if built is None:
                failed.append(f'{mod.NAME} [{target}]')
                continue
            lines, nbytes = built
            cases.add(mod.NAME)
            plus = [l for l in lines if l.startswith('+ ')]
            reg = sum(1 for l in plus if l.endswith('[reg]'))
            shr = sum(1 for l in plus if '[shr' in l)
            staged = sum(1 for l in plus if '[shr, 2 stages]' in l)
            left = [l for l in lines if l.startswith('- ')]
            moved['reg'] += reg
            moved['shr'] += shr
            moved['staged'] += staged
            single.update(l.partition('one stage: ')[2] for l in plus
                          if 'one stage: ' in l)
            reasons.update(_reason(l) for l in left)
            shared += nbytes
            if not args.summary:
                print(f'{mod.NAME[:38]:38s} {target:12s} {reg:4d} {shr:4d} '
                      f'{staged:4d} {len(left):5d} {nbytes:7d}')

    total = moved['reg'] + moved['shr'] + sum(reasons.values())
    print()
    print(f'{total} transfers and loops over {len(cases)} cases, '
          f'move distance {args.distance}'
          + (', two stages asked for' if args.multibuffer else ''))
    print(f'{moved["reg"]:6d} moved into a register array')
    print(f'{moved["shr"]:6d} moved into a shared window')
    if args.multibuffer:
        print(f'{moved["staged"]:6d}   of them with a second stage')
        for why, n in single.most_common():
            print(f'{n:6d}   with one: {why}')
    print(f'{sum(reasons.values()):6d} left where they are:')
    for why, n in reasons.most_common():
        print(f'{n:6d}   {why}')
    print(f'{shared:6d} B of shared memory over all launches')
    if failed:
        print(f'\ndid not generate: {", ".join(failed)}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
