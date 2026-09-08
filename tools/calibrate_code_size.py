# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Fit the code-size estimate against kernels this generator actually emits.

`analysis.cost.estimated_lines` answers whether a body is large against an
instruction cache, and it answers it from arithmetic alone: staging, index
arithmetic and guards all ride on one constant.  That constant is only worth
what the corpus it came from is worth, so it is fitted here rather than
asserted, against kernels generated on the spot instead of against a file
somebody shipped.

Reports the fit and, more usefully, the spread.  A constant that is right on
the totals and wrong by half on a mid-sized kernel is fine for the question it
answers and would be misleading anywhere else, so the error distribution is
printed beside the coefficient rather than left to be discovered.
"""

import argparse
import importlib.util
import statistics
import sys
from pathlib import Path
from typing import List, Optional, Tuple

from tensorforge.analysis.cost import (LINES_FIXED, LINES_PER_LANE_FLOP,
                                       list_cost)
from tensorforge.common.basic_types import Datatype
from tensorforge.common.context import Context
from tensorforge.generators import lanes as lane_config
from tensorforge.generators.generator import Generator


def code_lines(text: str) -> int:
    """Lines of kernel that are not comment and not blank."""
    return sum(1 for line in text.splitlines()
               if line.strip() and not line.lstrip().startswith('//'))


def measure(descr_list, arch: str, backend: str,
            datatype: Datatype) -> Optional[Tuple[int, int, int]]:
    """`(flops, lanes, lines)` for one list, or None where it will not build."""
    context = Context(arch=arch, backend=backend, fp_type=datatype)
    try:
        generator = Generator(list(descr_list), context)
        generator.generate()
        lines = code_lines(generator.get_kernel())
    except Exception:
        return None
    flops = list_cost(descr_list, batch=1, datatype=datatype).flops
    if not flops or not lines:
        return None
    try:
        lanes = lane_config.deduce(list(descr_list), context).num_threads
    except Exception:
        lanes = 32
    return flops, lanes, lines


def load_cases(root: Path) -> List[Tuple[str, object]]:
    out = []
    for path in sorted(root.rglob('*.py')):
        if path.name.startswith('_'):
            continue
        spec = importlib.util.spec_from_file_location(path.stem, path)
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception:
            continue
        if hasattr(module, 'descr_list'):
            out.append((path.stem, module))
    return out


def fit(points: List[Tuple[float, float]]) -> Tuple[float, float, float]:
    n = len(points)
    sx = sum(x for x, _ in points)
    sy = sum(y for _, y in points)
    sxx = sum(x * x for x, _ in points)
    sxy = sum(x * y for x, y in points)
    slope = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    intercept = (sy - slope * sx) / n
    mean = sy / n
    residual = sum((y - (slope * x + intercept)) ** 2 for x, y in points)
    total = sum((y - mean) ** 2 for _, y in points)
    return slope, intercept, 1 - residual / total if total else 1.0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cases', default='tests/cases')
    ap.add_argument('--arch', default='sm_86')
    ap.add_argument('--backend', default='cuda')
    args = ap.parse_args()

    points, skipped = [], 0
    for name, module in load_cases(Path(args.cases)):
        dtype = getattr(module, 'DTYPE', Datatype.F32)
        try:
            descrs = module.descr_list()
        except Exception:
            # A case may be built to be refused; it has no size to measure.
            skipped += 1
            continue
        result = measure(descrs, args.arch, args.backend, dtype)
        if result is None:
            skipped += 1
            continue
        flops, lanes, lines = result
        points.append((flops / lanes, float(lines)))

    if len(points) < 3:
        sys.exit(f'too few kernels to fit ({len(points)}, {skipped} skipped)')

    slope, intercept, r2 = fit(points)
    shipped = [abs(LINES_PER_LANE_FLOP * x + LINES_FIXED - y) / y
               for x, y in points]
    fitted = [abs(slope * x + intercept - y) / y for x, y in points]

    print(f'{len(points)} kernels measured, {skipped} skipped')
    print(f'fitted    lines = {slope:.3f} * flops/lanes + {intercept:.1f}   '
          f'R2 = {r2:.4f}')
    print(f'shipped   lines = {LINES_PER_LANE_FLOP} * flops/lanes '
          f'+ {LINES_FIXED}')
    print(f'relative error   shipped: median {statistics.median(shipped):.3f} '
          f'max {max(shipped):.3f}')
    print(f'                 fitted : median {statistics.median(fitted):.3f} '
          f'max {max(fitted):.3f}')


if __name__ == '__main__':
    main()
