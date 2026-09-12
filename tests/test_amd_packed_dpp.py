# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""The packed DPP arrangements, checked on the kernels they produce.

Two of them, both one row-share move followed by packed FMAs:

* column pairs at lead width one -- `B(k, j)` and `B(k, j + 1)` in one
  register pair, moved together, and multiplied into both columns of a slot
  by one `v_pk_fma_f32` with the slot's `A(i, k)` splat;
* lead width above one -- the lanes hold pairs of rows and pairs of
  contraction steps, and one move carries two steps.

Where the move does not repay, the chain stays fused, at width one as it was
and at width two per component.  Numbers for `local_flux` (ROCm 7.2, IGC-free
counts from `hipcc -S`): gfx1251 5892 instructions fused, 5591 with column
pairs; at lead width two 6085 through the nest, 4599 through the packed chain,
161 VGPRs.  The `pin` after each row is what keeps those VGPRs: without it the
moves of a whole body came first and gfx1251 needed 740.
"""

from __future__ import annotations

import contextlib
import dataclasses
import importlib.util
import io
import re
import warnings
from pathlib import Path

from tensorforge.common.context import Context, Options
from tensorforge.generators import lanes as lanes_mod
from tensorforge.generators.generator import Generator

CASES = Path(__file__).resolve().parent / 'cases'

PAIR_MOVE = re.compile(
    r'VectorT<float, 2> v\d+_bc = tensorforge::movdpp16<\d+>')


def _kernel(name, arch, width=1):
    path = next(CASES.rglob(f'{name}.py'))
    spec = importlib.util.spec_from_file_location('tf_pdpp__' + name, path)
    case = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(case)
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter('ignore')
        ctx = Context(arch=arch, backend='hip', fp_type=case.DTYPE,
                      options=Options(lead_vectorize=width > 1))
        lanes = None
        if width > 1:
            lanes = dataclasses.replace(
                lanes_mod.deduce(case.descr_list(), ctx), lead_width=width)
        gen = Generator(case.descr_list(), ctx, lanes=lanes)
        gen.generate()
    return gen.get_kernel()


def test_column_pairs_move_two_broadcast_values_at_once():
    """`local_flux` on gfx1251: two lead slots, so a pair of columns is one
    `v_mov_b64_dpp` and two packed FMAs against four fused ones.  The ninth
    column has no partner and keeps the modifier."""
    src = _kernel('local_flux', 'gfx1251')
    assert PAIR_MOVE.search(src), 'the pairs are moved as pairs'
    assert 'tensorforge::pin(' in src, 'each row pins its accumulators'
    assert 'fmacdpp16<' in src, 'the odd column stays fused'


def test_where_a_pair_takes_two_moves_the_chain_stays_fused():
    """gfx1250 has no 64-bit DPP: at two lead slots a pair is four
    instructions either way, and the fused chain is the shorter code."""
    src = _kernel('local_flux', 'gfx1250')
    assert not PAIR_MOVE.search(src)
    assert 'fmacdpp16<' in src


def test_a_packed_lead_operand_takes_the_dpp_chain():
    """At lead width two the nest used to broadcast every `B` element with a
    lane read or a swizzle.  The DPP chain now moves a lane's pair of
    contraction steps once per row, and each step feeds both rows of the
    accumulator pair."""
    src = _kernel('local_flux', 'gfx942', width=2)
    assert PAIR_MOVE.search(src)
    assert re.search(r'_acc \+= \(\(v\d+_bc\[[01]\]\) \* v\d+_data\)', src)
    assert 'tensorforge::pin(' in src


def test_without_the_move_the_wide_chain_is_fused_per_component():
    """gfx1150 has no packed FMA: the wide chain splits into components and
    keeps the modifier on every product, as the chain at width one does."""
    src = _kernel('local_flux', 'gfx1150', width=2)
    assert not PAIR_MOVE.search(src)
    assert 'fmacdpp16<' in src


def _fused_order(order, name, arch, width=1):
    from tensorforge.backend.instructions.compute.primitives.amd import codegen
    saved = codegen.FUSED_ORDER, codegen.FUSED_WIDE
    codegen.FUSED_ORDER, codegen.FUSED_WIDE = order, True
    try:
        return _kernel(name, arch, width)
    finally:
        codegen.FUSED_ORDER, codegen.FUSED_WIDE = saved


def test_the_fused_chain_walks_rows_with_the_same_products():
    """The row order emits what the column order emits -- the same fused
    products, the same count -- and pins each row's accumulators, which the
    column order never needed: there every `A` value was read first and held
    for the whole chain.  At lead width two with the fused wide chain
    switched on, which it is not by default."""
    for width in (1, 2):
        rows = _fused_order('rows', 'local_flux', 'gfx1150', width)
        cols = _fused_order('columns', 'local_flux', 'gfx1150', width)
        assert rows.count('fmacdpp16<') == cols.count('fmacdpp16<') > 0, width
        assert 'tensorforge::pin(' in rows, width
        assert 'tensorforge::pin(' not in cols, width


def test_the_order_follows_the_size_of_the_a_image():
    """`'auto'` takes the rows where the column order would hold a large `A`
    image across columns: `local_flux` holds 112 values a lane (two slots,
    56 steps) over nine fused columns on gfx1150 and walks rows.  On gfx942
    the matrix core takes eight of the columns, and the one left over reads
    each value once in either order, so it keeps the columns."""
    from tensorforge.backend.instructions.compute.primitives.amd import codegen
    assert codegen.FUSED_ORDER == 'auto'
    assert 'tensorforge::pin(' in _kernel('local_flux', 'gfx1150')
    assert 'tensorforge::pin(' not in _kernel('local_flux', 'gfx942')


def test_the_order_weighs_what_the_columns_would_hold():
    """The two terms of `'auto'` besides the image: an `A` already held in
    registers (`chain_three`'s intermediate) makes the image free, so the
    columns stay; a broadcast one lane at a time (FP64 on gfx1150, which has
    no FP64 DPP) relays a value for every step of every column, and
    `add_true_f64` takes the rows for that alone."""
    assert 'tensorforge::pin(' not in _kernel('chain_three', 'gfx1150')
    assert 'tensorforge::pin(' in _kernel('add_true_f64', 'gfx1150')


def test_an_amd_block_keeps_its_threads_at_lead_width_two():
    """NVIDIA holds the mults and halves the block; on AMD the halved block
    was measured slower (gfx1150, `local_flux`: 242 against 153 ns an
    element), so the lanes a mult covers no longer shrink the block there."""
    from tensorforge.common.basic_types import Datatype
    from tensorforge.generators.generator import RegmaxBlockPolicy

    def mults(arch, backend, width):
        ctx = Context(arch=arch, backend=backend, fp_type=Datatype.F32)
        return RegmaxBlockPolicy(ctx, global_mem=0, mem_size_per_mult=0,
                                 num_threads=32,
                                 lead_width=width).get_num_mults_per_block()

    assert mults('gfx1150', 'hip', 2) == mults('gfx1150', 'hip', 1) == 8
    assert mults('sm_86', 'cuda', 2) * 2 == mults('sm_86', 'cuda', 1)
