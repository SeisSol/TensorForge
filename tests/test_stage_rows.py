# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""An image whose rows `stage_row_bytes` pads holds each element where its
reads look for it.

The copy into the image counts in elements of the operand, and each access
takes its address in the image from its first element.  So an access has to
keep a row whole: one over the end of a row puts the first elements of the
next one into the pad.  An operand with an alignment promise is copied four
floats at a time where nothing stops it, and with rows of 9 that is the
access that crosses.  Under an explicit vector a transfer is a vector over
the lanes, which no row of 9 holds whole at any width, so an image copied
from global memory keeps the operand's rows there.
"""

from __future__ import annotations

import contextlib
import inspect
import io
import re
import warnings

import pytest

from tensorforge.backend.instructions.memory.load import GlbToShrLoader
from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.context import Context
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.common.options import Options
from tensorforge.generators.descriptions import GemmDescr
from tensorforge.generators.generator import Generator
from tensorforge.reference import kernel_eval as ke


def _descr(dtype, rows, cols):
    """`C = A @ B` with `B` `rows` long, on operands that promise 16-byte
    alignment -- which every batch member keeps: each operand here is a
    whole number of sixteen bytes.  `B` does not carry the lead dimension,
    so it is staged in shared memory where the target stages."""
    def tensor(shape, alias):
        return SubTensor(Tensor(shape, Addressing.STRIDED,
                                BoundingBox([0, 0], list(shape)),
                                alias=alias, datatype=dtype, alignment=16))
    return [GemmDescr(trans_a=False, trans_b=False,
                      a=tensor([16, rows], 'A'), b=tensor([rows, cols], 'B'),
                      c=tensor([16, cols], 'C'), alpha=1.0, beta=0.0)]


def _build(dtype, rows, cols, arch='sm_86', backend='cuda', **options):
    """The kernel, its launch geometry, and every access the staging copies
    make: their width and the rows (`linscale`) they are copied into."""
    hops = []
    original = GlbToShrLoader._write_hop
    signature = inspect.signature(original)

    def recording(self, *args, **kwargs):
        bound = signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        a = bound.arguments
        if a['end'] > a['start']:
            hops.append((a['increment'], a['linscale']))
        return original(self, *args, **kwargs)

    ctx = Context(arch=arch, backend=backend, fp_type=dtype,
                  options=Options(**options))
    gen = Generator(_descr(dtype, rows, cols), ctx)
    GlbToShrLoader._write_hop = recording
    try:
        with warnings.catch_warnings(), \
                contextlib.redirect_stdout(io.StringIO()):
            warnings.simplefilter('ignore')
            gen.generate()
    finally:
        GlbToShrLoader._write_hop = original
    return gen.get_kernel(), ke.launch_geometry(gen.get_launcher()), hops


def _written(src, geometry):
    """What the kernel leaves in the operands it writes, on the host oracle
    with every multiplication of the block running."""
    lanes, mults = geometry
    signature = re.search(r'\bkernel_\w+\(([^)]*)\)', src).group(1)
    outputs = {p.split()[-1] for p in signature.split(',')
               if '*' in p and not p.strip().startswith('const')
               and '<const ' not in p}
    assert outputs, 'the kernel writes no operand'
    mem = ke.evaluate_wave(src, lanes, seed=17, globals_only=True,
                           mults=mults, block=True)
    written = {k: v for k, v in mem.items() if k[0] in outputs}
    assert written, 'nothing written to compare'
    return written


@pytest.mark.parametrize('dtype,pad', [(Datatype.F32, 16), (Datatype.F32, 8),
                                       (Datatype.F64, 16)],
                         ids=['f32-16', 'f32-8', 'f64-16'])
def test_a_padded_image_computes_what_the_operands_rows_do(dtype, pad):
    """Rows of 9 under a pad: the copy moves an element at a time, and the
    kernel computes bit for bit what it computes on the operand's rows."""
    plain, plain_at, _ = _build(dtype, 9, 12)
    padded, padded_at, hops = _build(dtype, 9, 12, stage_row_bytes=pad)
    assert any(rows is not None for _, rows in hops), \
        'the image is not padded: nothing to check'
    assert _written(padded, padded_at) == _written(plain, plain_at)


@pytest.mark.parametrize('dtype,rows,cols,pad,widest', [
    (Datatype.F32, 9, 12, 16, 1),
    (Datatype.F64, 9, 12, 16, 1),
    (Datatype.F32, 12, 12, 64, 4),
], ids=['f32-9', 'f64-9', 'f32-12'])
def test_every_access_of_a_padded_copy_keeps_its_row_whole(dtype, rows, cols,
                                                           pad, widest):
    """An access of a copy into padded rows is as wide as both rows divide:
    one element for rows of 9, and still four floats for rows of 12 in
    image rows of 16."""
    _, _, hops = _build(dtype, rows, cols, stage_row_bytes=pad)
    padded = [(width, linscale) for width, linscale in hops
              if linscale is not None]
    assert padded, 'the image is not padded: nothing to check'
    assert all(linscale[0] % width == 0 and linscale[1] % width == 0
               for width, linscale in padded), padded
    assert max(width for width, _ in padded) == widest


def test_an_explicit_vector_copies_into_the_operands_rows():
    """ESIMD stages `B` and copies it a vector of 16 lanes at a time, which
    no row of 9 holds: the image keeps the operand's rows, and the kernel is
    the one it is without the option."""
    plain, _, plain_hops = _build(Datatype.F32, 9, 12, arch='pvc',
                                  backend='esimd')
    padded, _, hops = _build(Datatype.F32, 9, 12, arch='pvc',
                             backend='esimd', stage_row_bytes=16)
    assert hops, 'nothing staged: nothing to check'
    assert all(linscale is None for _, linscale in hops), hops
    assert hops == plain_hops

    def unnamed(src):
        # The name and the options comment say which options built it.
        src = re.sub(r'\bkernel_\w+', 'kernel', src)
        return re.sub(r'^\s*// options:.*$', '', src, flags=re.M)

    assert unnamed(padded) == unnamed(plain)
