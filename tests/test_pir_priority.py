# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Wave priorities on AMD (`Options.amd_setprio`, `pir.priority`): where the
markers stand in a kernel, and that nothing else changes."""
from __future__ import annotations

import contextlib
import importlib.util
import io
import pathlib
import re

import pytest

from tensorforge.backend.pir.core import Op
from tensorforge.common.basic_types import Datatype
from tensorforge.common.context import Context, Options
from tensorforge.generators.generator import Generator

from test_rolling import gemm, make

SETPRIO = re.compile(r'__builtin_amdgcn_s_setprio\((\d)\);')
CASES = pathlib.Path(__file__).parent / 'cases'


def _kernel(backend='hip', arch='gfx942', **opts):
    descrs = [gemm(make('A', [24, 24]), make('B', [24, 5]), make('C', [24, 5])),
              gemm(make('C', [24, 5]), make('D', [5, 5]), make('E', [24, 5]))]
    gen = Generator(descrs, Context(arch=arch, backend=backend,
                                    fp_type=Datatype.F32,
                                    options=Options(**opts)))
    with contextlib.redirect_stdout(io.StringIO()):
        gen.generate()
    return gen.get_kernel()


def _without_priorities(src):
    # The kernel's name carries a digest of its options.
    lines = [re.sub(r'kernel_\w+', 'kernel', ln) for ln in src.splitlines()
             if 'setprio' not in ln and 'tensorforge-meta' not in ln
             and '// options:' not in ln]
    return '\n'.join(lines)


def test_off_by_default():
    assert not SETPRIO.search(_kernel())


def test_loads_raise_a_wave_around_its_transfers():
    """High ahead of a transfer and low behind it, the two alternating; the
    kernel otherwise the same statement for statement."""
    src = _kernel(amd_setprio='loads')
    levels = [int(m.group(1)) for m in SETPRIO.finditer(src)]
    assert levels and levels[0] == 2
    assert all(a != b for a, b in zip(levels, levels[1:]))
    # Every raised stretch holds a load of global memory.  (A load the
    # forward pass took from a register keeps its comment and is no
    # transfer, so it stands unraised.)
    for stretch in re.findall(r's_setprio\(2\);(.*?)s_setprio\(0\);', src,
                              re.S):
        assert 'load{g>' in stretch and ('load' in stretch.split('\n', 2)[-1])
    assert _without_priorities(src) == _without_priorities(_kernel())


def test_compute_is_the_reverse():
    levels = [int(m.group(1)) for m in
              SETPRIO.finditer(_kernel(amd_setprio='compute'))]
    assert levels[0] == 2 and levels[1] == 0


def test_alternate_raises_every_other_wave_once():
    src = _kernel(amd_setprio='alternate')
    assert len(SETPRIO.findall(src)) == 1
    assert 'readfirstlane' in src


@pytest.mark.parametrize('mode', ['loads', 'alternate'])
def test_not_on_nvidia(mode):
    assert not SETPRIO.search(_kernel(backend='cuda', arch='sm_90',
                                      amd_setprio=mode))


def test_an_unknown_mode_is_refused():
    with pytest.raises(ValueError):
        _kernel(amd_setprio='sideways')


@pytest.mark.parametrize('mode, text', [
    ('flat', '__builtin_amdgcn_s_setprio(0);'),
    ('sched', '__builtin_amdgcn_sched_barrier(0);')])
def test_the_controls_stand_where_loads_stands(mode, text):
    """The same places as `loads`, with nothing of the priorities: the
    controls that tell the order from the barrier the instruction is."""
    marked = _kernel(amd_setprio='loads').splitlines()
    control = _kernel(amd_setprio=mode).splitlines()
    where = [n for n, ln in enumerate(marked) if 's_setprio' in ln]
    assert where == [n for n, ln in enumerate(control) if text in ln]


@pytest.mark.parametrize('mode, ops', [
    ('loads', [(Op.SET_PRIORITY, 2), (Op.SET_PRIORITY, 0)]),
    ('compute', [(Op.SET_PRIORITY, 2), (Op.SET_PRIORITY, 0),
                 (Op.SET_PRIORITY, 2)]),
    ('alternate', [(Op.SET_PRIORITY, 2)]),
    ('flat', [(Op.SET_PRIORITY, 0), (Op.SET_PRIORITY, 0)]),
    ('sched', [(Op.FENCE, None), (Op.FENCE, None)])])
def test_the_markers_are_engine_ops(mode, ops):
    """What the target writes (`pir.engines`), not text: a priority, or for
    `sched` a fence of the schedule -- around the transfer of a body that
    has one, and the body verifies."""
    from tensorforge.backend import pir
    from tensorforge.backend.pir.build import IRBuilder
    from tensorforge.backend.pir.core import MemSpace
    b = IRBuilder(fptype=Datatype.F32)
    g = b.alloc(Datatype.F32, (8,), MemSpace.GLOBAL, hint='g', offset=0)
    r = b.alloc(Datatype.F32, (8,), MemSpace.REGISTER, hint='r')
    for i in range(2):
        b.store(r, b.load(g, i, hint='x'), i)
    b.store(r, b.load(r, 0, hint='y'), 1)
    body = pir.set_priority(b.finish(), mode)
    pir.verify(body)
    marks = [s for s in body if s.op in (Op.SET_PRIORITY, Op.FENCE)]
    assert [(s.op, s.attr('level')) for s in marks] == ops
    assert all(s.text is None and not s.has_side_effects for s in marks)
    assert (marks[0].attr('waves') == 'odd') == (mode == 'alternate')


def _case_kernel(case, **opts):
    spec = importlib.util.spec_from_file_location(case, CASES / f'{case}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    options = dict(getattr(module, 'OPTIONS', {}), **opts)
    gen = Generator(module.descr_list(),
                    Context(arch='gfx942', backend='hip', fp_type=module.DTYPE,
                            options=Options(**options)))
    with contextlib.redirect_stdout(io.StringIO()):
        gen.generate()
    return gen.get_kernel()


def _markers_as_one(src):
    src = re.sub(r'__builtin_amdgcn_(s_setprio\(\d\)|sched_barrier\(0\));',
                 'MARK;', src)
    return [re.sub(r'kernel_\w+', 'kernel', ln) for ln in src.splitlines()
            if 'tensorforge-meta' not in ln and '// options:' not in ln]


def test_the_sched_control_differs_from_loads_in_the_marker_alone():
    """A guard the emitter writes into its `if` behind a priority it writes
    there behind the control's fence too -- the transfer guarded after a
    run of stores, here: the fence orders the device compiler, and the
    passes are done (`wall=False`)."""
    loads = _case_kernel('mma_operator_copies_run', amd_setprio='loads')
    control = _case_kernel('mma_operator_copies_run', amd_setprio='sched')
    assert 'sched_barrier' in control
    assert _markers_as_one(loads) == _markers_as_one(control)
