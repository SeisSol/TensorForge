# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""The host oracle runs a block: every multiplication's lanes, and C's
integer arithmetic.

What a block does together ahead of its batch loop -- copying an operator
every element reads into shared memory, each thread its share -- is only
whole with every thread of the block taking part.  Run with the block's first
multiplication alone, the shares of the others are shared memory nobody
wrote, which `Slot` fills with seeded numbers: plausible, and wrong.

An index the copy computes divides by an integer, and in C that quotient is
an integer.  Python's `/` is not, and a fraction that survives into a sum
moves every element after it.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import re
import warnings
from pathlib import Path

import pytest

from tensorforge.common.context import Context
from tensorforge.common.options import Options
from tensorforge.generators.generator import Generator
from tensorforge.reference import kernel_eval as ke

CASES = Path(__file__).resolve().parent / 'cases'
_SYNC = re.compile(r'__syncthreads\(\);')


def _build(stem: str, **options):
    path = next(CASES.rglob(f'{stem}.py'))
    spec = importlib.util.spec_from_file_location('tf_block__' + stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ctx = Context(arch='sm_86', backend='cuda',
                  fp_type=getattr(mod, 'DTYPE', None),
                  options=Options(**options))
    gen = Generator(mod.descr_list(), ctx, attrs=getattr(mod, 'ATTRS', None))
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter('ignore')
        gen.generate()
    return gen.get_kernel(), ke.launch_geometry(gen.get_launcher())


def _written(src: str, geometry, block: bool = True) -> dict:
    """What the kernel leaves in the operands it writes."""
    lanes, mults = geometry
    signature = re.search(r'\bkernel_\w+\(([^)]*)\)', src).group(1)
    outputs = {p.split()[-1] for p in signature.split(',')
               if '*' in p and 'const' not in p}
    assert outputs, 'the kernel writes no operand'
    mem = ke.evaluate_wave(src, lanes, seed=17, globals_only=True,
                           mults=mults, block=block)
    written = {k: v for k, v in mem.items() if k[0] in outputs}
    assert written, 'nothing written to compare'
    return written


def test_a_copy_the_block_shares_is_whole_with_every_multiplication():
    """`preload_globals` copies its operator with every thread of the block
    -- a sixteenth of it by each multiplication on sm_86 -- so the kernel
    computes what it computes without the copy where all of them run, and
    not where the first runs alone."""
    plain, plain_at = _build('addressing_none')
    staged, staged_at = _build('addressing_none', preload_globals=True)
    assert staged_at[1] > 1 and 'threadIdx.y * blockDim.x' in staged
    expected = _written(plain, plain_at)
    assert _written(staged, staged_at) == expected
    assert _written(staged, staged_at, block=False) != expected


def test_a_copy_the_block_shares_is_ordered_by_its_barrier():
    """Each multiplication reads what the threads of the others copied in,
    so without the barrier behind the copy the oracle reports the reads."""
    src, (lanes, mults) = _build('addressing_none', preload_globals=True)

    def races(text):
        found = []
        ke.evaluate_wave(text, lanes, seed=3, races=found, elements=2,
                         mults=mults, block=True)
        return found

    assert races(src) == []
    lines = src.split('\n')
    loop = next(k for k, line in enumerate(lines) if 'batchId0 = ' in line)
    barrier = max(k for k in range(loop) if _SYNC.search(lines[k]))
    without = '\n'.join(line for k, line in enumerate(lines) if k != barrier)
    assert races(without)


@pytest.mark.parametrize('expr, value', [
    ('a / b', 2), ('-a / b', -2), ('a / -b', -2), ('a / b * b', 8),
    ('a % b', 3), ('-a % b', -3), ('x / 2.0', 1.25)])
def test_the_oracle_divides_as_c_does(expr, value):
    """Between integers, `/` truncates toward zero and `%` keeps the sign of
    the dividend."""
    lane = ke.Interp(ke.Slot(), {'a': 11, 'b': 4, 'x': 2.5})
    assert lane.ev(expr) == value
