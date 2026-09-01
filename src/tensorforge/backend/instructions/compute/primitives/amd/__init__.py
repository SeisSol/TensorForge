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
* `reorder`  -- how to build a fragment from the nest's registers
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

from ...strategy import Strategy

from .arch import amdarch, cdna2, gfx1250, gfx1251, rdna
from .caps import has_fmacdpp4, has_fmacdpp8, has_fmacdpp16
from .catalog import (DEFINED_TRANSPOSES, MANTISSA, MATRIX_OPS, MFMA_TILES,
                      NOT_MODELLED, Call, Fragment, MatrixOp,
                      MfmaTile, lane_batched_ops, mfma_tile_for, ops_for,
                      split_products, split_terms, usable_mfma_tiles)
from .features import FEATURE_TARGETS, has_feature, wave_size
from .layouts import (FRAGMENT_BITS, Provenance, covers, established,
                      position, provenance)
from .reorder import (IDENTITY_DPP, ROW, Move, fragment_cost,
                      fragment_moves)
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
    'IDENTITY_DPP', 'ROW', 'Move', 'fragment_cost', 'fragment_moves',
    'wanted_fmadpp_step', 'select_fmadpp_step',
    'Relayout', 'RELAYOUTS', 'BROADCAST', 'MOVDPP16', 'TRANSPOSE4X4',
    'find_relayout',
    'fmadpp', 'fmadpp4', 'fmadpp8', 'fmadpp16', 'fmascalar',
    'hfma', 'matmul32', 'matmuldpp', 'matmul',
    'mfma_emu_int8', 'mfma_emu_bf16_f32', 'mfma_emu_f16_f32', 'wmma3atom',
]


def strategies(shape, ctx):
    """What this target can emit for this shape.

    The DPP chain always: a broadcast modifier on the multiply needs nothing
    of the shape, and where the widest form does not link, `select.py` falls
    to a narrower one rather than to nothing.

    A matrix core only where a tile fits, which is a structural question and
    not a family or a type one.  `mfma_f64_16x16x4f64` spends two of its lane
    bits on the contraction, so the data operand carries the leading dimension
    there and the lane-batched loop cannot feed it; `MatrixOp.lane_batched`
    states that as one equation and `mfma_tile_for` asks it.  F64 therefore
    lands on DPP -- where `fmacdpp16(double&, ...)` serves it -- because no
    tile fits, rather than because a condition names the type.

    A sparse second operand is read by linear index, which no fragment layout
    accepts; the DPP chain has a branch for it and takes it.
    """
    offered = {Strategy.DPP}
    if not shape.sparse \
            and mfma_tile_for(shape.threads, shape.dtype, ctx) is not None:
        offered.add(Strategy.MATRIX)
    return frozenset(offered)


def scratch(strategy, dtype):
    """Nothing: both arrangements here keep their operands in registers."""
    return 0


def matmul(writer, ops, ctx, strategy):
    """Emit the arrangement the caller chose."""
    C, A, B = ops.C, ops.A, ops.B
    M, N, K, kx = ops.lead_slots, ops.n, ops.k, ops.kx
    threads, dtype, sparse = ops.threads, ops.dtype, ops.sparse

    if strategy is Strategy.MATRIX:
        matmul32(writer, C, A, B, M, N, K, kx, threads, dtype, sparse, ctx)
    else:
        matmuldpp(writer, 0, C, A, B, M, N, K, kx, threads, dtype, sparse, ctx)
    return True
