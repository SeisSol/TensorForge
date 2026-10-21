# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""The pointer an element owns, read a whole iteration ahead of the copy that
follows it (`pointer_stage`).

Under `Addressing.PTR_BASED` every operand of an element hangs off its entry
in a table of pointers.  With `enable_multibuffer` the copy for the next
element goes to the head of the body and follows that element's pointer right
away, so the read of the table is what the copy waits for.  With the stage
the pointer comes in with the iteration: read an iteration before, with the
element after next, and carried.

The host oracle reads the tables -- entry `e` of `m1` is an array of its own
-- and is the check that the copy follows the right element's pointer: over
three trips of a block, with a mask that holds elements back, the kernel
computes what it computes without the transform.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import re
import warnings
from pathlib import Path

from tensorforge.common.context import Context, Options
from tensorforge.common.options import registry
from tensorforge.generators.generator import Generator
from tensorforge.reference import kernel_eval as ke

CASES = Path(__file__).parent / 'cases'


def _build(**options):
    path = CASES / 'addressing_ptr_based.py'
    spec = importlib.util.spec_from_file_location('tf_stage__' + path.stem,
                                                  path)
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


def _written(src: str, geometry, elements: int, races=None) -> dict:
    """What the kernel leaves in the operands it writes: the arrays the
    entries of their tables point at."""
    lanes, mults = geometry
    flags = [1] * elements
    flags[1] = 0
    flags[mults + 2] = 0
    signature = re.search(r'\bkernel_\w+\(([^)]*)\)', src).group(1)
    outputs = {p.split()[-1] for p in signature.split(',')
               if '*' in p and not p.strip().startswith('const')
               and '<const ' not in p}
    mem = ke.evaluate_wave(src, lanes, seed=7, globals_only=True,
                           mults=mults, block=True, races=races,
                           elements=elements, preset={'flags0': flags})
    written = {k: v for k, v in mem.items()
               if k[0].split('@')[0] in outputs}
    assert written, 'nothing written to compare'
    return written


def test_on_by_default():
    assert registry()['pointer_stage'].default is True


def test_the_oracle_reads_a_table_of_pointers():
    """An operand declared a pointer to pointers is a table: `&m1[e][i]`
    is slot `i` of the array entry `e` points at."""
    assert ke._py('&m1[3][4 + i]') == 'ADDR(m1[3], 4 + i)'
    assert ke._py('&m1[3 * 16 + i]') == 'ADDR(m1, 3 * 16 + i)'
    mem = ke.Slot(1)
    env = ke._base_env('void k(const float ** m1, float * m2) { m1; m2; }',
                       mem, 0)
    assert isinstance(env['m1'], ke.PtrTable)
    assert isinstance(env['m2'], ke.Ptr)
    assert env['m1'][3][4] == mem.read('m1@3', 4)


def test_the_stage_is_where_the_copy_is_at_the_head():
    staged, _ = _build(enable_multibuffer=True)
    plain, _ = _build(enable_multibuffer=True, pointer_stage=False)
    assert 'second_glb_' in staged and 'ahead_glb_' in staged
    assert 'second_glb_' not in plain and 'ahead_glb_' not in plain
    # the copy at the head follows the carried pointer, and the read of the
    # element after next comes first
    lines = staged.splitlines()
    loop = next(i for i, l in enumerate(lines) if 'for (size_t' in l)
    fetch = next(i for i, l in enumerate(lines) if 'ahead_glb_' in l)
    copy = next(i for i in range(loop, len(lines))
                if '__pipeline_memcpy_async' in lines[i])
    assert loop < fetch < copy
    carried = re.search(r'\b(v\d+_glb_m\d+) = second_glb_m\d+;', staged)
    assert carried and carried.group(1) in lines[copy]


def test_without_a_second_stage_nothing_changes():
    def body(kernel):
        # the name hashes the options, and the options are written out
        kernel = re.sub(r'kernel_[0-9a-f]+', 'kernel', kernel)
        return re.sub(r'// options:.*', '', kernel)
    assert body(_build()[0]) == body(_build(pointer_stage=False)[0])


def test_the_staged_kernel_computes_what_it_computes_unwrapped():
    plain, at = _build(enable_wrap_loads=False)
    staged, staged_at = _build(enable_multibuffer=True)
    elements = 3 * at[1]
    races = []
    assert (_written(staged, staged_at, elements, races)
            == _written(plain, at, elements))
    assert not races, races
