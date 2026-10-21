# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""The arrays of pointers and the flags, promised unwritten and unaliased
(`restrict_batch_arrays`, on by default).

`const` down to the last level and `__restrict__` in the kernel's prototype
let a compiler that sees a whole wave address one slot read it as one scalar
load -- on AMD through the scalar cache, whose waits do not queue behind the
vector loads.  The launcher's prototype is the interface's and stays as it
is: the launcher binds a local of the kernel's type and passes that.
"""

from __future__ import annotations

import contextlib
import io
import re
import warnings

import pytest

from harness import syntax

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


def _gemm(backend, arch, **options):
    """`C[b] = A[b] B` over arrays of pointers, with flags."""
    rows, k = 32, 9
    a, c = (Tensor([rows, k], Addressing.PTR_BASED,
                   BoundingBox([0, 0], [rows, k]), alias=alias,
                   datatype=Datatype.F32) for alias in 'AC')
    b = Tensor([k, k], Addressing.NONE, BoundingBox([0, 0], [k, k]),
               alias='B', datatype=Datatype.F32)
    context = Context(arch=arch, backend=backend, fp_type=Datatype.F32,
                      options=Options(**options))
    generator = Generator([GemmDescr(False, False, SubTensor(a), SubTensor(b),
                                     SubTensor(c), alpha=1.0, beta=0.0)],
                          context)
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter('ignore')
        generator.generate()
    return generator


def _prototype(source, prefix):
    """The parameters of the first function named `prefix...`, split at the
    commas outside template arguments."""
    text = re.search(rf'\b{prefix}\w*\(([^)]*)\)', source).group(1)
    params, depth, start = [], 0, 0
    for i, char in enumerate(text):
        depth += (char == '<') - (char == '>')
        if char == ',' and depth == 0:
            params.append(text[start:i].strip())
            start = i + 1
    return params + [text[start:].strip()]


def _unnamed(source):
    """The source without the kernel's name and the options it spells, which
    the options asked for decide."""
    source = re.sub(r'\bkernel_\w+', 'kernel', source)
    return re.sub(r'//[^\n]*', '', source)


SPELLINGS = {
    'hip': ('tensorforge::SpacePtrRestrict<const unsigned, '
            'tensorforge::GlobalMemspace> flags0'),
    'cuda': 'const unsigned * __restrict__ flags0',
}


@pytest.mark.parametrize('backend,arch', [('hip', 'gfx1150'),
                                          ('hip', 'gfx942'),
                                          ('cuda', 'sm_86')])
def test_the_kernel_promises_its_arrays_and_its_flags(backend, arch):
    """`T *const * __restrict__` for every array of pointers, the read ones
    and the written one alike -- the floats a written one reaches are
    written, its pointers are not -- and the flags `const` and restrict."""
    params = _prototype(_gemm(backend, arch, restrict_batch_arrays=True)
                        .get_kernel(), 'kernel_')
    arrays = [p for p in params if '*const *' in p]
    assert sorted(p.split()[-1] for p in arrays) == ['m0', 'm1']
    assert all('*const * __restrict__' in p for p in arrays)
    assert SPELLINGS[backend] in params
    # The batch's own operand, the counts and the offsets are no part of it.
    plain = _prototype(_gemm(backend, arch, restrict_batch_arrays=False)
                       .get_kernel(), 'kernel_')
    kept = [p for p in params if p not in arrays and p != SPELLINGS[backend]]
    assert kept == [p for p in plain if p.split()[-1] not in
                    ('m0', 'm1', 'flags0')]


@pytest.mark.parametrize('backend,arch', [('hip', 'gfx942'),
                                          ('cuda', 'sm_86')])
def test_the_promise_is_the_default(backend, arch):
    """The interface means it already, so nothing has to ask for it."""
    assert (_gemm(backend, arch).get_kernel()
            == _gemm(backend, arch, restrict_batch_arrays=True).get_kernel())


@pytest.mark.parametrize('backend,arch', [('hip', 'gfx1150'),
                                          ('cuda', 'sm_86')])
def test_the_launcher_keeps_the_interface(backend, arch):
    """The caller's types, and a local of the kernel's for each promised
    parameter, which is what the kernel is called with."""
    promised = _gemm(backend, arch, restrict_batch_arrays=True).get_launcher()
    plain = _gemm(backend, arch, restrict_batch_arrays=False).get_launcher()
    assert _prototype(promised, 'launcher_') == _prototype(plain, 'launcher_')
    for name in ('m0', 'm1', 'flags0'):
        assert re.search(rf'__restrict__\)?\s*{name}Arg = \(', promised) or \
            re.search(rf'SpacePtrRestrict<[^>]*> {name}Arg = \(', promised)
        assert re.search(rf'\bkernel_\w+[^;]*\b{name}Arg\b', promised), (
            f'the kernel called with the caller\'s {name}')


@pytest.mark.parametrize('backend,arch', [('hip', 'gfx1150'),
                                          ('cuda', 'sm_86'),
                                          ('acpp', 'pvc')])
def test_switched_off_nothing_is_promised(backend, arch):
    """Off, and on SYCL, whose kernels say `kernel_args_restrict` already."""
    off = _gemm(backend, arch, restrict_batch_arrays=False).get_kernel()
    assert '*const *' not in off
    if backend == 'acpp':
        assert _unnamed(_gemm(backend, arch, restrict_batch_arrays=True)
                        .get_kernel()) == _unnamed(off)


@pytest.mark.parametrize('backend,arch', [('hip', 'gfx1150'),
                                          ('cuda', 'sm_86')])
def test_the_promised_kernel_is_well_formed(backend, arch):
    """`const` between the stars parses where the prototype puts it."""
    if syntax.compiler() is None:
        pytest.skip('no host C++ compiler')
    result = syntax.check_source(
        _gemm(backend, arch, restrict_batch_arrays=True).get_kernel(),
        shim=syntax.shim_for(backend))
    assert result.ok, result.stderr


@pytest.mark.parametrize('backend,arch', [('hip', 'gfx1150'),
                                          ('cuda', 'sm_86')])
def test_the_numbers_do_not_change(backend, arch, monkeypatch):
    """Through the host oracle, the products generic where AMD would take
    a path the oracle does not run."""
    monkeypatch.setattr(amd, 'ORDER', (Strategy.GENERIC,))

    def written(**switch):
        generator = _gemm(backend, arch, **switch)
        src = generator.get_kernel()
        lanes, mults = ke.launch_geometry(generator.get_launcher())
        mem = ke.evaluate_wave(src, lanes, seed=3, globals_only=True,
                               mults=mults)
        outputs = {p.split()[-1] for p in _prototype(src, 'kernel_')
                   if '*' in p and not p.startswith('const')
                   and 'const float' not in p}
        out = {key: v for key, v in mem.items()
               if key[0].split('@')[0] in outputs}
        assert out, 'nothing written to compare'
        return out

    assert written() == written(restrict_batch_arrays=False)
