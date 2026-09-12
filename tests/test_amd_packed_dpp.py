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
