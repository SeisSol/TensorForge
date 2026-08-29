# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""AMD code generation for the multilinear kernel.

`multilinear.py` enters through `matmul()` and nothing else.  The modules
below are layered in dependency order, and the layering carries the lesson of
the bugs that came out of this file:

* `arch`     -- which family a target is
* `caps`     -- what its runtime defines
* `features` -- which LLVM subtarget features it has
* `catalog`  -- what a matrix instruction is
* `layouts`  -- where each element of its operands sits
* `relayout` -- which instruction turns one lane distribution into another
* `select`   -- which instruction width to use
* `emitters` -- how to write one instruction down
* `codegen`  -- the kernel
* `unused`   -- matrix paths kept for repair, with no call site

The split between `arch` and `caps` is the load-bearing one.  Those were the
same thing here, and a family predicate standing in for a capability is what
let gfx900 emit a call to a template that has only a declaration there.

Removed when this became a package: the `dppctrl_*` constant helpers, the raw
`amdgcn_*` intrinsic wrappers, four `shuffle_*` routines, two `reduction`s
written against CUDA's `__shfl_xor_sync`, the `MatrixCore` class with its
`matrixcores`/`archmap` tables and the `matmul` it served, and three empty
stubs -- 350 lines unreachable from `matmul()` through the call graph, not
merely uncovered by tests.  Two of those names, `reduction` and `matmul`, were
defined twice at module level, so Python had been discarding the first
definition since it was written.  `tests/test_amd_reachability.py` keeps the
property.
"""

from tensorforge.common.basic_types import Datatype

from .arch import amdarch, cdna2, gfx1250, gfx1251, rdna
from .caps import has_fmacdpp4, has_fmacdpp8, has_fmacdpp16
from .catalog import (DEFINED_TRANSPOSES, MANTISSA, MATRIX_OPS, MFMA_TILES,
                      NOT_MODELLED, Call, Fragment, MatrixOp,
                      MfmaTile, lane_batched_ops, mfma_tile_for, ops_for,
                      split_products, split_terms, usable_mfma_tiles)
from .features import FEATURE_TARGETS, has_feature, wave_size
from .layouts import (FRAGMENT_BITS, Provenance, covers, established,
                      position, provenance)
from .codegen import hfma, matmul32, matmuldpp
from .emitters import fmadpp, fmadpp4, fmadpp8, fmadpp16, fmascalar
from .relayout import (BROADCAST, MOVDPP16, RELAYOUTS, TRANSPOSE4X4, Relayout,
                       find_relayout)
from .select import select_fmadpp_step, wanted_fmadpp_step
from .unused import (mfma_emu_bf16_f32, mfma_emu_f16_f32, mfma_emu_int8,
                     wmma3atom)

__all__ = [
    'amdarch', 'cdna2', 'gfx1250', 'gfx1251', 'rdna',
    'has_fmacdpp4', 'has_fmacdpp8', 'has_fmacdpp16',
    'FEATURE_TARGETS', 'has_feature', 'wave_size',
    'Call', 'Fragment', 'MatrixOp', 'MATRIX_OPS', 'MANTISSA',
    'NOT_MODELLED', 'ops_for', 'split_terms',
    'split_products',
    'MfmaTile', 'DEFINED_TRANSPOSES', 'MFMA_TILES', 'usable_mfma_tiles',
    'lane_batched_ops', 'mfma_tile_for',
    'FRAGMENT_BITS', 'Provenance', 'covers', 'established',
    'position', 'provenance',
    'wanted_fmadpp_step', 'select_fmadpp_step',
    'Relayout', 'RELAYOUTS', 'BROADCAST', 'MOVDPP16', 'TRANSPOSE4X4',
    'find_relayout',
    'fmadpp', 'fmadpp4', 'fmadpp8', 'fmadpp16', 'fmascalar',
    'hfma', 'matmul32', 'matmuldpp', 'matmul',
    'mfma_emu_int8', 'mfma_emu_bf16_f32', 'mfma_emu_f16_f32', 'wmma3atom',
]


def matmul(writer, C, A, B, M, N, K, kx, threads, dtype, sparse, ctx):
    """Matrix path where a tile fits, DPP everywhere else.

    The condition used to be `cdna1(ctx) and not gfx1251(ctx) and dtype ==
    F32` --- a family predicate and a type check standing in for a structural
    property.  It gave the right answer, and it gave it for a reason that does
    not survive contact with the rest of the catalogue: widening the type
    check to F64 would have routed `mfma_f64_16x16x4f64` into a loop that
    cannot feed it, because that instruction spends two of its lane bits on
    the contraction and the data operand carries the leading dimension there.
    `MatrixOp.lane_batched` states that as one equation and
    `mfma_tile_for` asks it.

    So F64 still takes the DPP path, which is where `fmacdpp16(double&, ...)`
    already serves it --- now because no tile fits rather than because the
    condition names F32.
    """
    if not sparse and mfma_tile_for(threads, dtype, ctx) is not None:
        matmul32(writer, C, A, B, M, N, K, kx, threads, dtype, sparse, ctx)
    else:
        matmuldpp(writer, 0, C, A, B, M, N, K, kx, threads, dtype, sparse, ctx)
