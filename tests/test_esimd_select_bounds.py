# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""No explicit-vector read reaches past the vector it reads.

A `select<W, 1>(offset)` on a `simd<T, N>` past `N - W` reads whatever lies
beyond: on the Max 1550 the order-5 `derivative` of SeisSol came out NaN in
rows 32..34 of its time integral.  The epilogue of `I += dQ(1) power(1)`
walked the integral's tail block -- `LeadIndex(0, block=3, offset=32)` under
the explicit vector -- and read the shift-less start as row 0, so the
accumulator of `dQ(1)`'s 20 rows seemed to cover it and was read at the
tail's slots: the next column's, and for the last column past the vector
(`epilogue._first_row`).

Checked on the generated text: every `select` whose offset is a constant or
`(C + (v * K))` in an unrolled loop over `v`, against its vector's length.
"""
import contextlib
import io
import re
import sys
import warnings
from pathlib import Path

import pytest

from tensorforge.backend.instructions.compute.epilogue import _first_row
from tensorforge.backend.symbol import LeadIndex
from tensorforge.common.context import Context, Options
from tensorforge.generators.generator import Generator

BENCH = Path(__file__).resolve().parent.parent / 'tools' / 'bench'
DATA = BENCH / 'data'
if str(BENCH) not in sys.path:
    sys.path.insert(0, str(BENCH))

_DECL = re.compile(r'simd<\w+, (\d+)> (\w+)\(')
_FOR = re.compile(r'for \(int32_t (v\d+_\w+) = 0; \1 < (\d+); \+\+\1\)')
_SELECT = re.compile(r'(\w+)\.template select<(\d+), 1>\(')
_AFFINE = re.compile(r'^\(*(\d+)(?:_i32)? \+ \((v\d+_\w+) \* (\d+)\)\)*$')
_SCALED = re.compile(r'^\(*\((v\d+_\w+) \* (\d+)\)\)*$')


def test_a_tail_block_starts_at_its_shift():
    assert _first_row(LeadIndex(0, block=3, stride=1, offset=32)) == 32
    assert _first_row(LeadIndex(1, block=32, stride=1)) == 32


def _esimd_kernel(capture, workload):
    from suite import from_dump
    w = {x.name: x for x in from_dump(DATA / f'{capture}.json')}[workload]
    ctx = Context(arch='pvc', backend='esimd', fp_type=w.datatype,
                  options=Options())
    gen = Generator(w.descrs(), ctx, attrs=w.attrs)
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter('ignore')
        gen.generate()
    return gen.get_kernel()


def _selects(line):
    """`(vector, width, offset)` of every select on `line`, the offset as
    written between its balanced parentheses."""
    for m in _SELECT.finditer(line):
        depth, i = 1, m.end()
        while i < len(line) and depth:
            depth += {'(': 1, ')': -1}.get(line[i], 0)
            i += 1
        yield m.group(1), m.group(2), line[m.end():i - 1].strip()


def _reaches_past(src):
    """Every select that reads past its vector: (line, text)."""
    sizes, trips, out = {}, {}, []
    for number, line in enumerate(src.splitlines(), 1):
        for size, name in _DECL.findall(line):
            sizes[name] = int(size)
        for var, count in _FOR.findall(line):
            trips[var] = int(count)
        for name, width, offset in _selects(line):
            if name not in sizes:
                continue
            last = None
            if offset.isdigit():
                last = int(offset)
            elif _AFFINE.match(offset):
                base, var, step = _AFFINE.match(offset).groups()
                if var in trips:
                    last = int(base) + (trips[var] - 1) * int(step)
            elif _SCALED.match(offset):
                var, step = _SCALED.match(offset).groups()
                if var in trips:
                    last = (trips[var] - 1) * int(step)
            if last is not None and last + int(width) > sizes[name]:
                out.append((number, line.strip()[:160]))
    return out


@pytest.mark.parametrize('capture', ['elastic-4-s-s1', 'elastic-5-s-s1'])
def test_the_derivative_reads_inside_its_vectors(capture):
    if not (DATA / f'{capture}.json').exists():
        pytest.skip(f'no capture {capture}')
    assert _reaches_past(_esimd_kernel(capture, 'gpu_derivative')) == []
