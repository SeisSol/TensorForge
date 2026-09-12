# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""A batch-constant lead operand stored in k-quads (`amd.prepared_order`).

The lane-batched MFMA reads one element of the lane's row per contraction
step, so four steps stored side by side are one 16-byte read.  Nothing here
runs an MFMA; the order is checked as a permutation, and what reads it as
generated code.
"""
import re

import pytest

from test_amd_blgp import MFMA, _blgps, _kernel_at

from tensorforge.backend.instructions.compute.primitives import amd

QUAD_READ = re.compile(r'\*\(tensorforge::VectorT<float, 4>\*\)&glb_m0\[')
SCALAR_READ = re.compile(r'= glb_m0\[')


@pytest.mark.parametrize('shape,threads', [((56, 56), 32), ((9, 9), 16),
                                           ((80, 13), 32)])
def test_the_order_stores_every_cell_once_where_a_lane_reads_it(shape, threads):
    rows, cols = shape
    order = amd.quad_order(shape, threads, 4)
    cells = [c for c in order if c != -1]
    assert sorted(cells) == list(range(rows * cols))
    slots = -(-rows // threads)
    for r, k in [(0, 0), (rows - 1, cols - 1), (threads % rows, 5 % cols)]:
        s, lane = divmod(r, threads)
        at = ((k // 4 * slots + s) * threads + lane) * 4 + k % 4
        assert order[at] == r + rows * k


def test_only_a_whole_operand_is_offered_one():
    from tensorforge.common.basic_types import Datatype
    from tensorforge.common.context import Context
    ctx = Context(arch='gfx942', backend='hip', fp_type=Datatype.F32)
    assert amd.prepared_order((56, 56), Datatype.F32, ctx, columns=9, lead=56,
                              depth=56, threads=32) is not None
    assert amd.prepared_order((56, 56), Datatype.F32, ctx, columns=9, lead=48,
                              depth=56, threads=32) is None
    assert amd.prepared_order((56, 56), Datatype.F64, ctx, columns=9, lead=56,
                              depth=56, threads=32) is None


@pytest.mark.parametrize('arch', ['gfx942', 'gfx90a'])
def test_a_prepared_operator_is_read_four_steps_at_a_time(arch):
    """`local_flux` with `prepare_operands`: the operator read as vectors of
    four steps and never by coordinate -- the ninth column's chain included
    -- with as many MFMAs as without, `blgp` still handing the quads on."""
    plain = _kernel_at('local_flux', arch)
    src = _kernel_at('local_flux', arch, prepare_operands=True)
    assert QUAD_READ.search(src)
    assert not SCALAR_READ.search(src)
    assert len(MFMA.findall(src)) == len(MFMA.findall(plain))
    blgps = _blgps(src)
    assert blgps.count(1) == blgps.count(2) > 0
