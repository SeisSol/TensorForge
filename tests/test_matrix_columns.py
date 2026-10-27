# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""`Options.matrix_columns`: the width of CDNA's lane-batched matrix
instruction, asked for rather than counted.

The count of passes (`compute.ranking`) takes the four-wide tile unless the
columns fill the sixteen-wide one as well, and a width asked for is taken for
every product it serves.  What the count does not see -- the lead operand
read once per block of columns -- is the kernel's to weigh, which is why the
tuner turns this (`tuning.simple_space`).  Held here on the kernel text:
which instruction a product issues.
"""

from __future__ import annotations

import contextlib
import io
import re

import pytest

from tensorforge.backend.instructions.compute.primitives import (amd, intel,
                                                                  nvidia)
from tensorforge.backend.instructions.compute.primitives.amd import tiling
from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.context import Context
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.common.options import Options
from tensorforge.generators.descriptions import GemmDescr
from tensorforge.generators.generator import Generator
from tensorforge.generators.lanes import LaneConfig


def _product(columns, dtype=Datatype.F32, a=Addressing.NONE):
    """C[56 x columns] = A[56 x 56] B[56 x columns]: an operator the batch
    shares against an element's own, the shape of SeisSol's volume and flux
    products at order 6."""
    def view(shape, addressing, alias):
        return SubTensor(Tensor(shape, addressing, BoundingBox([0, 0], shape),
                                alias=alias, datatype=dtype))
    return [GemmDescr(trans_a=False, trans_b=False,
                      a=view([56, 56], a, 'A'),
                      b=view([56, columns], Addressing.STRIDED, 'B'),
                      c=view([56, columns], Addressing.STRIDED, 'C'),
                      alpha=1.0, beta=0.0)]


def _kernel(columns, arch='gfx942', a=Addressing.NONE, lanes=None,
            **options):
    gen = Generator(_product(columns, a=a),
                    Context(arch=arch, backend='hip', fp_type=Datatype.F32,
                            options=Options(**options)),
                    lanes=(None if lanes is None else
                           LaneConfig(num_threads=lanes,
                                      num_active_threads=lanes,
                                      lead_width=1)))
    with contextlib.redirect_stdout(io.StringIO()):
        gen.generate()
    return gen.get_kernel()


def _tiles(columns, arch='gfx942', **options):
    return set(re.findall(r'mfma_f32_\w+?f32', _kernel(columns, arch,
                                                        **options)))


FOUR, SIXTEEN = 'mfma_f32_4x4x1f32', 'mfma_f32_16x16x1f32'


def test_the_count_takes_the_wide_tile_only_where_the_columns_fill_it():
    """Nine columns are three four-wide issues of two passes against one
    sixteen-wide issue of eight; sixteen are a tie in passes, which the
    fewer issues decide."""
    assert _tiles(9) == {FOUR}
    assert _tiles(16) == {SIXTEEN}


@pytest.mark.parametrize('columns,asked,tile', [
    (9, 16, SIXTEEN), (18, 16, SIXTEEN), (16, 4, FOUR), (9, 4, FOUR)])
def test_an_asked_width_is_taken_for_every_product_it_serves(columns, asked,
                                                              tile):
    assert _tiles(columns, matrix_columns=asked) == {tile}
    # and in a stack, whose products are twice as wide
    assert _tiles(columns, matrix_columns=asked, stack_elements=2) == {tile}


@pytest.mark.parametrize('lanes', [16, 32])
def test_the_wide_tile_drives_the_wave_together(lanes):
    """The 16-wide tile spreads each block's output over the whole wave, and a
    multiplication narrower than it gathers its columns out of its
    neighbors' lanes, so the neighbors take the same trips through the batch
    loop (`amd.convergence`).  In a loop of their own, those past the end
    had left: on MI300A, sixteen lanes and two elements gave columns 8 to 15
    zero.  An element's own A, so that no `blgp` asks it instead; the 4-wide
    tile keeps a block's output in its lanes and asks nothing."""
    owned = Addressing.STRIDED
    src = _kernel(9, a=owned, lanes=lanes, matrix_columns=16)
    assert SIXTEEN in src and 'batchIdGroup0' in src
    src = _kernel(9, a=owned, lanes=lanes, matrix_columns=4)
    assert FOUR in src and 'batchIdGroup0' not in src
    # A whole wave is one multiplication, with no neighbor to wait for.
    src = _kernel(9, a=owned, lanes=64, matrix_columns=16)
    assert SIXTEEN in src and 'batchIdGroup0' not in src


def test_a_width_the_scheme_has_no_entry_of_leaves_the_count():
    """32 is held back (`catalog.LANE_BATCHED_BLOCKS`), and at eight lanes
    the sixteen-wide tile does not fit a multiplication."""
    assert _tiles(9, matrix_columns=32) == _tiles(9)
    ctx = Context(arch='gfx942', backend='hip', fp_type=Datatype.F32,
                  options=Options(matrix_columns=16))
    assert tiling.choose(8, Datatype.F32, ctx, columns=9).width == 4
    assert tiling.choose(64, Datatype.F32, ctx, columns=9).width == 16


def test_the_widths_are_the_lane_batched_tiles_and_nothing_elsewhere():
    """What the tuner offers (`MatrixPaths.widths`): the tiles that fit at
    the lanes, on a part with the matrix cores, for F32 -- and nothing on
    the other vendors, which do not read the option."""
    def ctx(arch):
        return Context(arch=arch, backend='hip', fp_type=Datatype.F32)
    assert amd.widths(64, Datatype.F32, ctx('gfx942')) == (4, 16)
    assert amd.widths(32, Datatype.F32, ctx('gfx90a')) == (4, 16)
    assert amd.widths(8, Datatype.F32, ctx('gfx942')) == (4,)
    assert amd.widths(64, Datatype.F64, ctx('gfx942')) == ()
    assert amd.widths(32, Datatype.F32, ctx('gfx1150')) == ()
    assert nvidia.widths(32, Datatype.F32, None) == ()
    assert intel.widths(16, Datatype.F32, None) == ()


def test_a_context_without_options_asks_for_nothing():
    """The unit tests' stand-ins carry a target and no options: the count's
    choice, as before the option."""
    class Bare:
        target = Context(arch='gfx942', backend='hip',
                         fp_type=Datatype.F32).target
    assert tiling.asked_width(Bare()) == 0
    assert tiling.asked_width(None) == 0
