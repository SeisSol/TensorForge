# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""One tile per term product, and the claim that makes it possible.

`matmulemu` hands the instruction the four registers a `kk` step already
holds, as the four slots of a k-vector fragment.  Nothing rearranges them, so
the whole path rests on those four registers already being where the wide
fragment wants them -- and a wrong stacking there yields a correctly typed
operand holding the wrong elements, which no snapshot and no symbolic
comparison would notice, since both treat the intrinsic as opaque.

So it is asserted here against the tabulated layouts rather than argued in a
comment.
"""

from __future__ import annotations

import pytest

from tensorforge.backend.instructions.compute import packing, split
from tensorforge.backend.instructions.compute.primitives import amd
from tensorforge.backend.instructions.compute.primitives.amd import layouts
from tensorforge.backend.pir.build import IRBuilder
from tensorforge.common.basic_types import Datatype
from tensorforge.common.context import Context

from test_amd_tiling import _Recorder, _operand


@pytest.fixture(scope='module')
def hip():
    return Context(arch='gfx90a', backend='hip', fp_type=Datatype.F32)


@pytest.fixture(scope='module')
def selected(hip):
    chosen = amd.emu_tile_for(64, Datatype.F32, hip)
    assert chosen is not None, 'gfx90a has a BF16 tile'
    return chosen


# -- the load-bearing claim ------------------------------------------------ #

def test_the_narrow_registers_are_the_wide_fragments_slots(hip, selected):
    """Element `(m, k)` of the wide fragment sits at slot `k` of the lane the
    narrow one puts `(m, 0)` in -- for every block, and for both operands.

    That is what lets the loop pass its four registers through unchanged.  If
    it stopped holding, the emitter would need a relayout between the two, and
    `reorder.fragment_moves` is where that would be found."""
    wide = selected[0].op
    narrow = amd.mfma_tile_for(64, Datatype.F32, hip).op
    assert narrow.k == 1 and wide.k == narrow.m

    for block in range(min(narrow.blocks, wide.blocks)):
        for m in range(narrow.m):
            base = layouts.position(narrow, 'A', m, 0, block=block)
            for k in range(wide.k):
                assert layouts.position(wide, 'A', m, k, block=block) == (
                    base[0] + k, base[1]), ('A', m, k, block)
        for n in range(narrow.n):
            base = layouts.position(narrow, 'B', 0, n, block=block)
            for k in range(wide.k):
                assert layouts.position(wide, 'B', k, n, block=block) == (
                    base[0] + k, base[1]), ('B', k, n, block)


def test_the_tile_is_only_offered_where_its_widths_agree(hip, selected):
    """The stacking above holds because the k-vector is exactly the block
    width.  An entry where it is not would need the relayout, so it is not
    offered rather than fed wrongly."""
    tile, _ = selected
    assert tile.op.k == tile.op.m == tile.op.n
    assert tile.op.n * tile.op.blocks == 64


# -- what is selected ------------------------------------------------------ #

def test_the_direct_path_is_never_a_split(hip):
    """Every entry tile selection reaches multiplies in the type it
    accumulates, so the emulated question is a separate one rather than a
    wider version of the same one."""
    tile = amd.mfma_tile_for(64, Datatype.F32, hip)
    assert split.terms(split.MANTISSA[tile.op.a.dtype], Datatype.F32) == 1


def test_the_term_count_comes_from_the_arithmetic(hip, selected):
    """Three, because BF16 keeps 8 bits against FP32's 24 -- not because the
    runtime's split happens to return three values."""
    tile, terms = selected
    assert tile.op.a.dtype is Datatype.BF16
    assert terms == split.terms(split.MANTISSA[Datatype.BF16], Datatype.F32)
    assert terms == 3


def test_only_a_split_the_runtime_defines_is_offered(hip):
    """A tile the runtime cannot feed is a call to an undeclared function,
    which is the same failure as a missing transpose."""
    tile, _ = amd.emu_tile_for(64, Datatype.F32, hip)
    assert tile.op.a.dtype in amd.DEFINED_SPLITS


def test_the_deployment_switch_is_separate_from_the_selection():
    """`emu_tile_for` says a tile exists; `EMULATION` says whether to take
    it.  Two facts, and only the second is a decision about the generator."""
    assert amd.EMULATION is False


# -- what it emits --------------------------------------------------------- #

def _run(hip, n, threads=64, lead_slots=2, k=8):
    rec = _Recorder()
    tile, terms = amd.emu_tile_for(threads, Datatype.F32, hip)
    writer = IRBuilder(Datatype.F32, context=hip)
    taken = amd.matmulemu(writer, rec, _operand, _operand, lead_slots, n, k, 0,
                          threads, Datatype.F32, None, hip, 0, n, tile, terms)
    return taken, rec


@pytest.mark.parametrize('n', [1, 2, 3, 4, 5, 7, 8, 12, 13])
def test_every_element_is_computed_exactly_once(hip, n):
    taken, rec = _run(hip, n)
    assert taken
    assert rec.covered == {(i, j) for i in range(2) for j in range(n)}
    assert not rec.duplicates


def test_it_issues_one_instruction_per_term_product(hip, selected):
    """The arrangement's whole claim: `len(products)` calls into one
    accumulator, laid out the way `packing.stages` counts them."""
    tile, terms = selected
    products = split.products(terms)
    assert len(products) == 6

    steps = 8
    layout = packing.stages(products, packing.tiles(steps, tile.op.k),
                            capacity=1)
    # One call per (product, k-vector), which is what the emitter loops over.
    assert packing.instructions(layout) == len(products) * packing.tiles(
        steps, tile.op.k)


def test_a_contraction_shorter_than_the_k_vector_is_zero_filled(hip):
    """The instruction reads all of its slots, so a tail that does not fill
    them contributes nothing only if they hold nothing."""
    taken, rec = _run(hip, n=4, k=2)
    assert taken
    assert rec.covered == {(i, j) for i in range(2) for j in range(4)}


# -- at the widths SeisSol runs, split once ------------------------------- #

def _emulated(monkeypatch, a, b_cols=9, threads=32, **options):
    """`C = A B` with the emulated scheme deployed, `A` an operator the batch
    shares (with its numbers) and `B` an element's own."""
    import numpy as np
    from tensorforge.common.basic_types import Addressing
    from tensorforge.common.matrix.boundingbox import BoundingBox
    from tensorforge.common.matrix.tensor import SubTensor, Tensor
    from tensorforge.common.options import Options
    from tensorforge.generators.descriptions import GemmDescr
    from tensorforge.generators.generator import Generator
    from tensorforge.generators.lanes import LaneConfig
    from tensorforge.backend.instructions.compute.primitives.amd import tiling
    monkeypatch.setattr(tiling, 'EMULATION', True)
    rows, depth = a.shape
    op = Tensor([rows, depth], Addressing.NONE, BoundingBox([0, 0], [rows, depth]),
                alias='A', datatype=Datatype.F32, data=np.asarray(a, dtype=float))

    def own(shape, alias):
        return SubTensor(Tensor(shape, Addressing.STRIDED, BoundingBox([0, 0], shape),
                                alias=alias, datatype=Datatype.F32))
    gen = Generator([GemmDescr(False, False, SubTensor(op), own([depth, b_cols], 'B'),
                               own([rows, b_cols], 'C'), alpha=1.0, beta=0.0)],
                    Context(arch='gfx942', backend='hip', fp_type=Datatype.F32,
                            options=Options(**options)),
                    lanes=LaneConfig(num_threads=threads, num_active_threads=rows,
                                     lead_width=1))
    gen.generate()
    return gen.get_kernel(), op


@pytest.mark.parametrize('threads', [16, 32, 64])
def test_the_emulated_tile_serves_every_width_the_lane_batched_one_does(hip, threads):
    """The tile is lane-batched: each multiplication's blocks broadcast within
    it, so the tile says which multiplications it serves, not the wave."""
    tile, terms = amd.emu_tile_for(threads, Datatype.F32, hip)
    assert tile.op.a.dtype is Datatype.BF16 and terms == 3
    assert tile.fits(threads)


def test_an_operator_is_stored_split_and_read_as_its_terms(monkeypatch):
    """Offered the three BF16 terms in k-quads (`amd.QuadOrder`), the operator
    is split once on the host, exactly, and the kernel packs each term's quad
    instead of splitting it; only the element's own operand is split, once
    per block of columns and step."""
    import numpy as np
    rng = np.random.default_rng(3)
    a = rng.standard_normal((56, 56))
    src, op = _emulated(monkeypatch, a)
    assert op.storage_parts == 3 and op.storage_planar
    values = np.asarray(op.storage_values(), dtype=np.float32)
    planes = values.reshape(3, -1)
    assert ((planes.view(np.uint32) & 0xFFFF) == 0).all()
    order = op.storage_map()
    total = planes.astype(np.float64).sum(axis=0)
    for slot, cell in enumerate(order):
        want = 0.0 if cell < 0 else np.float32(a[cell % 56, cell // 56])
        assert total[slot] == want
    blocks, steps = 3, 2                      # 9 columns, 56 steps at 32 lanes
    assert src.count('splitFloatx4BF16') == blocks * steps
    assert src.count('packBF16x4') > 0


def test_a_window_inside_a_quad_goes_to_the_direct_tile(hip):
    """The lead counts its steps from the window's first, and a quad of it
    has to meet one of the shared matrix: a window from step 1 is declined,
    for `matmul32` to take."""
    tile, terms = amd.emu_tile_for(32, Datatype.F32, hip)
    writer = IRBuilder(Datatype.F32, context=hip)
    assert not amd.matmulemu(writer, _Recorder(), _operand, _operand, 1, 4, 8, 1,
                             32, Datatype.F32, None, hip, 0, 4, tile, terms)
    taken = amd.matmulemu(IRBuilder(Datatype.F32, context=hip), _Recorder(),
                          _operand, _operand, 1, 4, 8, 4, 32, Datatype.F32,
                          None, hip, 0, 4, tile, terms)
    assert taken


def test_a_quad_the_operator_holds_nothing_in_is_left_out(monkeypatch):
    """Steps 8 to 15 of the operator are zero in every row: two quads, and
    with them their products."""
    import numpy as np
    import re
    rng = np.random.default_rng(4)
    a = rng.standard_normal((32, 32))
    full, _ = _emulated(monkeypatch, a.copy())
    a[:, 8:16] = 0.0
    sparse, _ = _emulated(monkeypatch, a)
    count = lambda src: len(re.findall(r'mfma_f32_4x4x4bf16_1k\(', src))
    assert count(sparse) == count(full) * 6 // 8
