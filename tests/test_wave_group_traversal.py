# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""The multiplications of a wave, driven through the batch loop together
(`wave_group_traversal`, off by default).

Where several multiplications share a wave and nothing in the body needs
them in step, each row takes its own trips: the loop is per row, and a row
whose flag holds it back skips the body.  Asked for, the rows of a wave take
their trips as a group (`BatchLoop._gen_grouped_ir`): the loop runs over the
group's first element, each row binds its own from it, and a row with none --
past the end, or held back -- runs the body on the group's element and writes
nothing.

The host oracle checks that the numbers are those of the loop per row, under
a mask that holds a row back on one trip and not the next.  It runs no
matrix fragment product, which would group the wave whatever was asked, so
the products on AMD are the generic ones.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import re
import warnings
from pathlib import Path

import pytest

from tensorforge.backend.instructions.compute.primitives import amd
from tensorforge.backend.instructions.compute.strategy import Strategy
from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.context import Context
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.common.options import Options
from tensorforge.generators.descriptions import GemmDescr
from tensorforge.generators.generator import Generator
from tensorforge.reference import kernel_eval as ke

CASES = Path(__file__).parent / 'cases'

GROUP_LOOP = re.compile(r'for \(size_t \w*batchIdGroup0\b')


@pytest.fixture(autouse=True)
def _generic_products(monkeypatch):
    monkeypatch.setattr(amd, 'ORDER', (Strategy.GENERIC,))


def _generate(descrs, context, attrs=None):
    generator = Generator(descrs, context, attrs=attrs)
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter('ignore')
        generator.generate()
    return generator


def _gemm(backend, arch, **options):
    """`C[b] = A[b] B` over arrays of pointers, with flags: 32 rows, half a
    wave of 64 on gfx942."""
    rows, k = 32, 9
    a, c = (Tensor([rows, k], Addressing.PTR_BASED,
                   BoundingBox([0, 0], [rows, k]), alias=alias,
                   datatype=Datatype.F32) for alias in 'AC')
    b = Tensor([k, k], Addressing.NONE, BoundingBox([0, 0], [k, k]),
               alias='B', datatype=Datatype.F32)
    context = Context(arch=arch, backend=backend, fp_type=Datatype.F32,
                      options=Options(**options))
    return _generate([GemmDescr(False, False, SubTensor(a), SubTensor(b),
                                SubTensor(c), alpha=1.0, beta=0.0)], context)


def _case(stem, backend, arch, **options):
    """A kernel of `tests/cases`."""
    path = next(CASES.rglob(f'{stem}.py'))
    spec = importlib.util.spec_from_file_location('tf_wgt__' + stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    context = Context(arch=arch, backend=backend,
                      fp_type=getattr(mod, 'DTYPE', None),
                      options=Options(**options))
    return _generate(mod.descr_list(), context, getattr(mod, 'ATTRS', None))


def _unnamed(source):
    """The source without the kernel's name and the options it spells, which
    the options asked for decide."""
    source = re.sub(r'\bkernel_\w+', 'kernel', source)
    return re.sub(r'//[^\n]*', '', source)


@pytest.mark.parametrize('backend,arch,lanes,group', [
    ('hip', 'gfx942', 0, 2),
    ('hip', 'gfx942', 16, 4),
    ('cuda', 'sm_86', 16, 2),
], ids=['gfx942-32', 'gfx942-16', 'sm_86-16'])
def test_a_shared_wave_takes_its_trips_as_a_group(backend, arch, lanes, group):
    """Off, a loop per row; on, a loop over groups of a wave's rows, each
    row's place in its group the row modulo the multiplications a wave
    holds."""
    off = _gemm(backend, arch, lanes_per_mult=lanes).get_kernel()
    assert not GROUP_LOOP.search(off)
    on = _gemm(backend, arch, lanes_per_mult=lanes,
               wave_group_traversal=True).get_kernel()
    assert GROUP_LOOP.search(on)
    assert re.search(rf'\w*batchIdLane0 = \(?threadIdx\.y % {group}\b', on)


@pytest.mark.parametrize('backend,arch', [('hip', 'gfx942'),
                                          ('cuda', 'sm_86')])
def test_off_by_default(backend, arch):
    assert (_gemm(backend, arch, lanes_per_mult=16).get_kernel()
            == _gemm(backend, arch, lanes_per_mult=16,
                     wave_group_traversal=False).get_kernel())


@pytest.mark.parametrize('backend,arch,lanes', [
    ('hip', 'gfx942', 64),
    ('hip', 'gfx1150', 0),
    ('cuda', 'sm_86', 0),
], ids=['gfx942-64', 'gfx1150-32', 'sm_86-32'])
def test_a_wave_of_one_multiplication_is_left_alone(backend, arch, lanes):
    """A multiplication that fills the wave is its own group already."""
    off = _gemm(backend, arch, lanes_per_mult=lanes).get_kernel()
    on = _gemm(backend, arch, lanes_per_mult=lanes,
               wave_group_traversal=True).get_kernel()
    assert _unnamed(on) == _unnamed(off)


def test_a_rotated_start_stays_per_row():
    """Asked for, and not needed: a start taken modulo the stride is no
    group's, so the section keeps its loop per row rather than being
    refused."""
    generator = _gemm('hip', 'gfx942', wave_group_traversal=True)
    plain = generator._get_2d_block_id()
    assert generator._group_size(0, plain) == 2
    assert generator._group_size(0, f'({plain} + numElements1) % (x)') == 1


def _mask(mults: int):
    """A flag per element of three trips: row 1 held back on its first trip
    and not on its second, row 2 the other way round."""
    flags = [1] * (3 * mults)
    flags[1] = 0
    flags[2 + mults] = 0
    return flags


def _written(generator, races=None) -> dict:
    """What the kernel leaves in the operands it writes, over three trips of
    one block, under `_mask`."""
    src = generator.get_kernel()
    lanes, mults = ke.launch_geometry(generator.get_launcher())
    signature = re.search(r'\bkernel_\w+\(([^)]*)\)', src).group(1)
    params = re.split(r',(?![^<]*>)', signature)
    outputs = {p.split()[-1] for p in params
               if ('*' in p or 'SpacePtr' in p)
               and not p.strip().startswith('const') and '<const ' not in p}
    flags = 'flags0' in src
    mem = ke.evaluate_wave(src, lanes, seed=7, globals_only=True, mults=mults,
                           block=True, races=races, elements=3 * mults,
                           preset={'flags0': _mask(mults)} if flags else None)
    written = {k: v for k, v in mem.items()
               if k[0].split('@')[0] in outputs}
    assert written, 'nothing written to compare'
    return written


@pytest.mark.parametrize('build', [
    lambda **o: _gemm('hip', 'gfx942', **o),
    lambda **o: _gemm('hip', 'gfx942', lanes_per_mult=16, **o),
    lambda **o: _gemm('cuda', 'sm_86', lanes_per_mult=16, **o),
    lambda **o: _case('square_notrans', 'cuda', 'sm_86', lanes_per_mult=16,
                      **o),
    lambda **o: _case('aligned_operands', 'hip', 'gfx942', **o),
    lambda **o: _case('addressing_ptr_based', 'hip', 'gfx942', **o),
], ids=['gemm-gfx942-32', 'gemm-gfx942-16', 'gemm-sm_86-16',
        'square_notrans-sm_86-16', 'aligned_operands-gfx942',
        'addressing_ptr_based-gfx942'])
def test_the_numbers_do_not_change(build):
    """Rows held back on one trip and not the next, the element of a held
    back row another row's -- and no access the hardware would not have
    ordered."""
    on = build(wave_group_traversal=True)
    assert GROUP_LOOP.search(on.get_kernel())
    races = []
    assert _written(on, races) == _written(build())
    assert not races, races
