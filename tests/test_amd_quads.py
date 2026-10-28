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


def test_the_order_is_laid_over_the_rows_an_operation_reads():
    """A lane reads the row of its slot's own origin, a multiple of the lanes
    from the box's corner (`MultilinearInstruction._read_corner`), and every
    slot whole: the order covers those rows -- the ones past the box padding
    -- and the steps the contraction walks.  SeisSol's volume computes rows 1
    to 53 of `kDivM`, which its lanes read as rows 0 to 63."""
    from tensorforge.common.basic_types import Datatype
    from tensorforge.common.context import Context
    ctx = Context(arch='gfx942', backend='hip', fp_type=Datatype.F32)
    whole = amd.prepared_order((56, 56), Datatype.F32, ctx, columns=9, lead=56,
                               depth=56, threads=32)
    assert whole is not None and tuple(whole[:4]) == (0, 56, 112, 168)
    volume = amd.prepared_order((64, 35), Datatype.F32, ctx, columns=9,
                                lead=53, depth=35, threads=32, corner=(0, 0),
                                rows=64)
    assert tuple(volume[:4]) == (0, 64, 128, 192)
    assert volume[(32 + 31) * 4] == 63
    # a window from row 40 at 32 lanes: its slot starts at row 32
    later = amd.prepared_order((64, 35), Datatype.F32, ctx, columns=9,
                               lead=20, depth=35, threads=32, corner=(32, 0),
                               rows=32)
    assert tuple(later[:4]) == (32, 96, 160, 224)
    # 56 rows read as two slots: the last eight lanes are padding
    rdiv = amd.prepared_order((56, 21), Datatype.F32, ctx, columns=9, lead=56,
                              depth=21, threads=32, corner=(0, 0), rows=64)
    assert rdiv[(32 + 24) * 4] == -1 and rdiv[(32 + 23) * 4] == 55
    assert amd.prepared_order((64, 35), Datatype.F32, ctx, columns=9, lead=53,
                              depth=35, threads=32, corner=(64, 0)) is None
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


def _windowed(monkeypatch, emulated, **options):
    """`C[1:54] = A[1:54, :] B` over an operator stored 64 x 35: the rows
    SeisSol's volume computes of `kDivM`."""
    import numpy as np
    from tensorforge.backend.instructions.compute.primitives.amd import tiling
    from tensorforge.common.basic_types import Addressing, Datatype
    from tensorforge.common.context import Context
    from tensorforge.common.matrix.boundingbox import BoundingBox
    from tensorforge.common.matrix.tensor import SubTensor, Tensor
    from tensorforge.common.options import Options
    from tensorforge.generators.descriptions import GemmDescr
    from tensorforge.generators.generator import Generator
    monkeypatch.setattr(tiling, 'EMULATION', emulated)
    a = np.zeros((64, 35))
    a[1:54] = np.random.default_rng(5).standard_normal((53, 35))
    op = Tensor([64, 35], Addressing.NONE, BoundingBox([0, 0], [64, 35]),
                alias='A', datatype=Datatype.F32, data=a)
    own = lambda shape, alias: Tensor(shape, Addressing.STRIDED,
                                      BoundingBox([0, 0], shape), alias=alias,
                                      datatype=Datatype.F32)
    c = own([64, 9], 'C')
    gen = Generator([GemmDescr(False, False,
                               SubTensor(op, BoundingBox([1, 0], [54, 35])),
                               SubTensor(own([35, 9], 'B')),
                               SubTensor(c, BoundingBox([1, 0], [54, 9])),
                               alpha=1.0, beta=0.0)],
                    Context(arch='gfx942', backend='hip', fp_type=Datatype.F32,
                            options=Options(**options)))
    gen.generate()
    return op


def test_an_operator_read_as_a_window_is_stored_for_its_lanes(monkeypatch):
    """Rows 1 to 53 are what the operation computes, and its lanes read rows
    0 to 63: the order starts at the slot's origin, lane 0 at row 0."""
    op = _windowed(monkeypatch, True)
    assert op.storage_parts == 3
    order = op.storage_map()
    assert tuple(order[:5]) == (0, 64, 128, 192, 1)


def test_the_direct_tile_takes_a_window_order_only_when_asked(monkeypatch):
    """Over SeisSol's orders 4 and 6 a window in quads came out even with the
    operand as it lies, so the default keeps it; `prepare_operands` asks."""
    assert _windowed(monkeypatch, False).storage_order is None
    asked = _windowed(monkeypatch, False, prepare_operands=True)
    assert asked.storage_order is not None and asked.storage_parts == 1
