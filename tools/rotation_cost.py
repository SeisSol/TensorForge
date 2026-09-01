# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""What rotation costs, whichever way it is paid for.

`wrap_prefetch` can only move a transfer across the back edge if its
destination has two copies -- `ceil((d + 1) / n)` with `d = n` is two.  Asking
the pass whether a given loop would benefit is exact (`assume_rotated`), but
the answer arrives too late: `ShrMemOpt` sizes the arena before any body
exists.  So the second copy has to be committed to in advance, and there are
two ways to do that.

**Generate twice.**  Build the section once to ask the pass, then again with
the answer.  Exact, and it costs build time.

**Allocate for everyone.**  Give every async shared transfer two stages and
let the pass use what it wants.  Free at build time, and it costs shared
memory on every launch -- including for the transfers that turn out not to
move.

Neither is obviously right until both are numbers, which is what this prints.

    python3 tools/rotation_cost.py
"""
import contextlib
import importlib.util
import io
import re
import sys
import time
from collections import Counter
from pathlib import Path

from tensorforge.backend.instructions.memory.load import GlbToShrLoader
from tensorforge.common.context import Context
from tensorforge.generators.generator import Generator

ROOT = Path(__file__).resolve().parent.parent
CASES = ROOT / 'tests' / 'cases'


def _cases():
    out = []
    for path in sorted(CASES.rglob('*.py')):
        if path.name.startswith('_'):
            continue
        spec = importlib.util.spec_from_file_location('rc_' + path.stem, path)
        mod = importlib.util.module_from_spec(spec)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                spec.loader.exec_module(mod)
        except Exception:
            continue
        if hasattr(mod, 'NAME') and hasattr(mod, 'descr_list'):
            out.append(mod)
    return out


def _async_stages(gen):
    """Stage sizes of the transfers that could rotate, and how many."""
    total = count = 0
    for section in getattr(gen, '_sections', []) or []:
        for instr in getattr(section, 'stream', []) or []:
            members = [instr] + list(getattr(instr, 'region', []) or [])
            for member in members:
                if (isinstance(member, GlbToShrLoader)
                        and member._use_cuda_memcpy):
                    total += member.stage_size()
                    count += 1
    return total, count


def main() -> int:
    cases = _cases()

    # -- what a second pass costs ------------------------------------------ #
    def build(times):
        start = time.time()
        for mod in cases:
            for _ in range(times):
                try:
                    ctx = Context(arch='sm_86', backend='cuda',
                                  fp_type=getattr(mod, 'DTYPE', None))
                    with contextlib.redirect_stdout(io.StringIO()):
                        Generator(mod.descr_list(), ctx).generate()
                except Exception:
                    pass
        return time.time() - start

    one, two = build(1), build(2)

    # -- what over-allocating costs ---------------------------------------- #
    totals = Counter()
    worst = []
    for mod in cases:
        try:
            ctx = Context(arch='sm_86', backend='cuda',
                          fp_type=getattr(mod, 'DTYPE', None))
            gen = Generator(mod.descr_list(), ctx)
            with contextlib.redirect_stdout(io.StringIO()):
                gen.generate()
            launcher = gen.get_launcher() or ''
        except Exception:
            continue
        found = re.search(r'(\d+)\s*\*\s*sizeof', launcher)
        if not found:
            continue
        arena = int(found.group(1))
        extra, count = _async_stages(gen)
        if not count:
            continue
        totals['arena'] += arena
        totals['extra'] += extra
        totals['kernels'] += 1
        worst.append((extra / max(arena, 1), mod.NAME, arena, extra, count))

    print(f'{len(cases)} cases, {totals["kernels"]} of them with an async '
          f'shared transfer\n')
    print(f'generate twice:      {one:.1f}s -> {two:.1f}s '
          f'({two / max(one, 0.01):.2f}x build time)')
    share = 100 * totals['extra'] / max(totals['arena'], 1)
    print(f'allocate for everyone: +{totals["extra"]} floats of arena '
          f'({share:.0f}% overall)')
    print('\n  worst kernels by arena growth:')
    for frac, name, arena, extra, count in sorted(worst, reverse=True)[:5]:
        print(f'    {name[:34]:34s} {arena:6d} +{extra:5d} '
              f'({100 * frac:3.0f}%) over {count} transfer(s)')
    print('\nBuild time is paid once; shared memory is paid on every launch, '
          'and it\nis what occupancy is computed from -- on exactly the '
          'kernels the pipeline\nis for. A quarter of an arena is a block of '
          'occupancy; twice the build is\nnot.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
