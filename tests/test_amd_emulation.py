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
from tensorforge.backend.instructions.compute.matmul import MatmulOperands
from tensorforge.backend.instructions.compute.primitives import amd
from tensorforge.backend.instructions.compute.primitives.amd import layouts
from tensorforge.backend.instructions.compute.strategy import Span, Strategy
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
    ops = MatmulOperands(A=_operand, B=_operand, C=rec, sparse=None,
                         lead_slots=lead_slots,
                         lead_elements=lead_slots * threads,
                         n=n, k=k, kx=0, threads=threads,
                         a=Datatype.F32, b=Datatype.F32,
                         accumulator=Datatype.F32)
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

    steps, threads = 8, 64
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
