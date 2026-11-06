# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""What a vendor path can read, and what it leaves to the nest.

Every vendor path is written for ``C[m,n] += A[m,k] B[k,n]``: one contraction
axis, read across the lanes from `B`, and the lead the only output axis on
`A`.  SeisSol's viscoelastic coupling ``Iane[i,k,p] E[k,p,q]`` contracts over
two axes, flattened into one, of which `B` spread only the first over the
lanes; the AMD paths computed it wrong on gfx942 and gfx1150 alike while the
nest -- which CUDA took for it -- was right.  The nest takes it now.  The
memory update ``dQane(1)[i,j,m] += dQane(0)[i,j,n] W[n,m]``, whose first
operand carries an output axis `A(i, k)` has no place for, was the nest's
already; it stays a guard.  The products around them keep their paths.
"""

from __future__ import annotations

import contextlib
import copy
import io

import seissol_suite as suite
from tensorforge.common.basic_types import Datatype
from tensorforge.common.context import Context, Options
from tensorforge.frontend.yateto import DescriptionReader
from tensorforge.generators.generator import Generator

SYSTEM = 'viscoelastic-linearckanelastic'
CONFIG = 'viscoelastic-linearckanelastic-o6-s'


def _name(ref):
    return ref.get('name') if isinstance(ref, dict) else str(ref)


def _cut(kernel, keep):
    """The kernel's description with the operations `keep(result, args)`
    selects, in their order."""
    d = copy.deepcopy(suite.description(SYSTEM, CONFIG, kernel))
    d['operations'] = [op for op in d['operations']
                       if keep(_name(op['result']),
                               [_name(a) for a in op.get('args', [])])]
    return d


def _built(description, arch='gfx942'):
    descrs = DescriptionReader(None, {}).read(description)[0]
    ctx = Context(arch=arch, backend='hip', fp_type=Datatype.F32,
                  options=Options())
    gen = Generator(descrs, ctx)
    with contextlib.redirect_stdout(io.StringIO()):
        gen.generate()
    return gen.get_kernel()


def _vendor_calls(src):
    return src.count('fmacdpp') + src.count('mfma') + src.count('movdpp')


def test_two_contraction_axes_take_the_nest():
    coupling = _cut('gpu_fluxLocalAll',
                    lambda res, args: res == '_tmp16'
                    or (res == 'Q' and '_tmp16' in args))
    assert _vendor_calls(_built(coupling)) == 0


def test_an_output_axis_on_the_first_operand_stays_with_the_nest():
    update = _cut('gpu_derivative',
                  lambda res, args: res == 'dQane(1)' and 'W' in args)
    assert _vendor_calls(_built(update)) == 0


def test_a_plain_product_keeps_its_path():
    """The first face's products of the same kernel, ``fMrT(0) I`` and
    ``rDivM(0) _tmp0``: one contraction axis each, the lead the first
    operand's only output axis."""
    face = _cut('gpu_fluxLocalAll',
                lambda res, args: res in ('_tmp0', '_tmp1'))
    assert _vendor_calls(_built(face)) > 0
