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

from ... import broadcast
from ...strategy import Span, Strategy, whole

from .arch import amdarch, cdna2, gfx1250, gfx1251, rdna
from .caps import has_fmacdpp4, has_fmacdpp8, has_fmacdpp16
from .catalog import (DEFINED_TRANSPOSES, MANTISSA, MATRIX_OPS, MFMA_TILES,
                      NOT_MODELLED, Call, Fragment, MatrixOp,
                      MfmaTile, lane_batched_ops, mfma_tile_for, ops_for,
                      split_products, split_terms, usable_mfma_tiles)
from .features import FEATURE_TARGETS, has_feature, wave_size
from .layouts import (FRAGMENT_BITS, Provenance, covers, established,
                      position, provenance)
from .reorder import (BANK, IDENTITY_DPP, ROW, Move, Select,
                      fragment_cost, fragment_moves)
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
    'BANK', 'IDENTITY_DPP', 'ROW', 'Move', 'Select',
    'fragment_cost', 'fragment_moves',
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
    accepts and which the broadcast chain has no lane to replicate; the DPP
    chain has a branch for it and takes it alone.
    """
    offered = {Strategy.DPP}
    if not shape.sparse:
        # The same chain the DPP one fuses its broadcast into, available here
        # through `readlane`.  It cannot read a sparse operand, which is the
        # one thing the DPP branch does that this does not.
        offered.add(Strategy.BROADCAST)
        if mfma_tile_for(shape.threads, shape.accumulator, ctx) is not None:
            offered.add(Strategy.MATRIX)
    return frozenset(offered)


def scratch(strategy, accumulator):
    """Nothing: both arrangements here keep their operands in registers."""
    return 0


def plan(strategy, shape, n, ctx):
    """How the chosen arrangement is laid out over the output.

    The matrix core covers whole tiles; what is left over is a cost question
    with a threshold behind it.  A tail of two or three columns is cheaper as
    one MFMA block with its spare lanes zeroed than as two or three passes of
    a broadcast chain, and a tail of one is not -- padding a block of four to
    compute one column spends three quarters of it on zeroes.

    So the tail is a second span rather than a handoff inside the emitter,
    and the boundary is stated once instead of computed twice.  Computing it
    twice is what makes the two arrangements overlap: `(n // block) * block`
    is the tail only when the block loop stopped there, and when it padded
    through to the end it points into a block already emitted -- both spans
    then write the same columns, the later store hiding it.
    """
    if strategy is not Strategy.MATRIX:
        return whole(strategy, n)
    tile = mfma_tile_for(shape.threads, shape.accumulator, ctx)
    boundary = ((n // tile.block) * tile.block) if n % tile.block < 2 else n
    if boundary >= n:
        return whole(Strategy.MATRIX, n)
    if boundary <= 0:
        # Fewer columns than one block, and too few to repay padding it: there
        # is no matrix span to name, not an empty one.
        return whole(Strategy.DPP, n)
    return (Span(Strategy.MATRIX, 0, boundary),
            Span(Strategy.DPP, boundary, n))


def matmul(writer, ops, ctx, span):
    """Emit one span of the plan."""
    C, A, B = ops.C, ops.A, ops.B
    M, N, K, kx = ops.lead_slots, ops.n, ops.k, ops.kx
    threads, dtype, sparse = ops.threads, ops.accumulator, ops.sparse

    if span.strategy is Strategy.BROADCAST:
        return broadcast.matmul(writer, ops, ctx, span)
    if span.strategy is Strategy.MATRIX:
        tile = mfma_tile_for(threads, dtype, ctx)
        if (ops.a, ops.b) != (tile.op.a.dtype, tile.op.b.dtype):
            # The tile was selected by what it accumulates in; what it
            # multiplies is a separate property of the same entry.  Where the
            # two operands do not already arrive as its fragments want them,
            # reaching this instruction is a split, and that is not what this
            # emitter does.
            return False
        matmul32(writer, C, A, B, M, N, K, kx, threads, dtype, sparse, ctx,
                 span.start, span.stop)
    else:
        matmuldpp(writer, span.start, C, A, B, M, N, K, kx, threads, dtype,
                  sparse, ctx, span.stop)
    return True
