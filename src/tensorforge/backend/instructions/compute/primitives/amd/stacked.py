# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""The multiplications of a wave stacked along a `k = 4` instruction's M.

`matmul32` lays a product onto `mfma_f32_4x4x1f32` with the lanes as the
instruction's N: every lane a row of the operator, every issue one step of the
contraction, the per-element operand broadcast by the instruction itself.  An
instruction that spends lane bits on its contraction cannot be fed that way
(`MatrixOp.lane_batched`), and the scheme that feeds one at a whole wave
anyway measured two to seven times slower than the DPP chain
(`tiling.EXCHANGE`): its transposes, swaps and gathers cost more than the
instruction saves.

At sixteen lanes a multiplication there is an assignment that gathers nothing
after the issue.  The wave holds four multiplications, the `p`-th
(`threadIdx.y % 4`) in lanes `16 p ..`, and a `16x16x4` instruction computes
``D = A B`` for a tile of sixteen operator rows `rt` and four output columns
`ct` with

* N the operator's row inside the tile -- the multiplication's own lane `t`;
* M the element and the column together: ``m = e + 4 jj`` for the `e`-th
  multiplication of the wave and column `4 ct + jj` on
  `mfma_f64_16x16x4f64`, ``m = jj + 4 e`` on `mfma_f32_16x16x4f32`;
* K four steps of the contraction, one in the lanes of each multiplication.

The two lay D out differently (`layouts.FRAGMENT_BITS`): the F64 one holds
row `4 s + q` of the tile in slot `s` of lane quarter `q`, the F32 one row
``s + 4 q``.  Each M order is the one that makes ``D[m(e, jj)][t]`` slot `jj`
of lane ``t + 16 e``: column `4 ct + jj` of slot `rt` in the `e`-th
multiplication's lane `t`, which is where the nest's register image holds
it.  The accumulator *is* the result, and the image, the temporaries and the
layouts stay as they are.

The operands are what that asks for:

* B, ``B[kk][n] = Op[16 rt + n][4 kt + kk]``: the lane's operator row at step
  `4 kt + p` in the `p`-th multiplication's lanes, which is what `A_wave` reads
  for `blgp` -- one load a tile, where the operator is batch-constant;
* A, ``A[m][kk] = T_e[4 kt + kk][4 ct + jj]``: per element, and the element is
  the one the lane's `t` names as an `m` -- another multiplication's.  Where
  it comes from is the route (`Options.amd_matrix_k4`, `a_source`).

`stacking` derives those three statements, and the M order, from the
fragment tables and declines an instruction or a lane count for which they do
not hold, so nothing above is assumed by the emitter.

A tile whose block of the operator is known zero (`MatmulOperands.A_zero`)
issues nothing, and its fragments are not read.  The columns past the last
four go through the DPP chain (`stacked_plan`), or into one more tile padded
with zeros (`padded`).

The contraction runs in its own order, four steps an issue, and on MI300A
both instructions add them as a chain of fused products in that order:
SeisSol's kernels came out bit-equal over every element to the DPP chain's
in double precision and to the lane-batched tile's in single, which is the
check a build of this scheme can be held to.  Not across a chain of
products (`stacked_chain_tile`), which adds them in another order.

At thirty-two and sixty-four lanes a multiplication spans `H` = 2 or 4
quarters of the wave, and one issue feeds every quarter the same sixteen
operator rows -- where the image holds sixteen other rows in each quarter of
a multiplication.  So the quarters take more columns instead: M is the
multiplication `p`, its quarter `h` and the slot `s` (the D layout's quarter
`q = H p + h`), the tile `4 H` columns wide, column `4 h + s` of it in slot
`s` of quarter `h`, and an issue still uses all four of its steps.  What
lands in quarter `h` are the rows of the issue's tile of sixteen, which the
image keeps in quarter `rt % H`: unchained, the quarters of a multiplication
trade blocks of four slots before the epilogue (`_relayout`, `H - 1` lane
gathers a slot); in a chain every lane reads and writes the output at its
own row and column (`_open_chain`), and nothing moves.  B is the operator
at row `16 rt + n` and step `k + q` in lane `n` of quarter `q`, one shifted
read (`MatmulOperands.A_lanes`).
"""

import contextlib
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

from tensorforge.common.basic_types import Datatype

from ... import bitlayout
from ...strategy import Span, Strategy, whole
from .catalog import MATRIX_OPS
from .layouts import element_at, position

#: The instruction the scheme issues, by the type it accumulates in.  F32
#: has the lane-batched tile besides (`tiling`), which feeds
#: `mfma_f32_4x4x1f32` through quad transposes and the DPP chain; this one
#: feeds a sixteen-wide tile with neither, at a coarser grain of skipped
#: zeros.
INSTRUCTIONS = {Datatype.F64: 'mfma_f64_16x16x4f64',
                Datatype.F32: 'mfma_f32_16x16x4f32'}

#: The instruction an emulated product issues, by the type it accumulates in
#: (`Options.amd_matrix_k4_emulate`): BF16 terms of both operands into the F32
#: accumulator, sixteen steps an issue, four a lane, and D the F32 tile's
#: (`stacking`).
EMULATED = {Datatype.F32: 'mfma_f32_16x16x16bf16_1k'}

#: The BF16 terms an F32 operand is split into, which cover its significand.
BF16_TERMS = 3

#: The two orders in which an instruction lays M out over the quarters of the
#: wave and the slots of a lane in D (`stacking`): the quarter minor, ``m = q
#: + 4 s`` -- `mfma_f64_16x16x4f64` --, or major, ``m = 4 q + s`` --
#: `mfma_f32_16x16x4f32`.
QUARTER_MINOR = 'quarter minor'
QUARTER_MAJOR = 'quarter major'

#: Lanes a multiplication may span: a quarter of the wave, two or all four
#: (see the module).
LANES = (16, 32, 64)

#: The routes to the per-element fragment (`Options.amd_matrix_k4`).
#: `transpose` gathers it out of the registers the operand is held in, a lane
#: gather per column of the tile (`ds_bpermute`, two for a double) and a
#: select by the lane's column; `shard` reads it out of the shared memory the
#: operand already sits in, one read a fragment.
ROUTES = ('shard', 'transpose')


def requested(ctx) -> Optional[str]:
    """The route `Options.amd_matrix_k4` asks for -- `'auto'` where it is
    simply on -- or `None` where it is off, or where the context carries no
    options to ask."""
    options = getattr(ctx, 'get_user_options', None)
    if options is None:
        return None
    asked = getattr(options(), 'amd_matrix_k4', False)
    if asked is None or asked is False:
        return None
    if asked is True:
        return 'auto'
    if asked in ROUTES:
        return asked
    from tensorforge.common.exceptions import GenerationError
    raise GenerationError(f'Options.amd_matrix_k4={asked!r}: a route of '
                          f'{ROUTES}, or a boolean')


@lru_cache(maxsize=None)
def stacking(op) -> Optional[str]:
    """How `op` lays a wave out in quarters -- the M order of its D,
    `QUARTER_MINOR` or `QUARTER_MAJOR` -- or `None` where its fragments are
    not the stacked assignment.

    Three statements, each read off the fragment tables (`layouts`), for
    quarters of `n` lanes, `L` a lane of the wave:

    * A: lane `L` holds ``A[L % n][L // n]``: its step of the contraction is
      its quarter, its M row its lane in the quarter;
    * B: lane `L` holds ``B[L // n][L % n]``: the operator's row `L % n` at
      the step of its quarter;
    * D: ``D[m][n']`` sits in slot `s` of lane ``n' + n q``, for ``m = q + 4
      s`` or ``m = 4 q + s``.

    `mfma_f64_16x16x4f64` has all three with the quarter minor,
    `mfma_f32_16x16x4f32` with it major.  Asked of the table rather than
    assumed, because the D layouts of one shape disagree between
    instructions (`layouts`), and a wrong order here is a correctly typed
    operand holding another lane's numbers.
    """
    quarter = op.n
    count = op.wave // quarter
    if (op.blocks != 1 or op.m != quarter or op.wave % quarter
            or op.k % count or op.m % count):
        return None
    # The steps a lane carries: one, or a k-vector of them (the BF16 tile).
    vector = op.k // count
    slots = op.m // count
    for lane in range(op.wave):
        for s in range(vector):
            step = vector * (lane // quarter) + s
            if element_at(op, 'A', s, lane) != (0, lane % quarter, step):
                return None
            if element_at(op, 'B', s, lane) != (0, step, lane % quarter):
                return None
    for order in (QUARTER_MINOR, QUARTER_MAJOR):
        if all(position(op, 'D', stacked_row(order, q, s, count, slots), n)
               == (s, n + quarter * q)
               for q in range(count) for s in range(slots)
               for n in range(quarter)):
            return order
    return None


def fits(op, threads: int) -> bool:
    """Whether `op` stacks (`stacking`) multiplications of `threads` lanes:
    whole quarters of the wave."""
    return (stacking(op) is not None and threads > 0
            and threads % op.n == 0 and op.wave % threads == 0)


def stacked_row(order, q, s, count=4, slots=4) -> int:
    """The M row D holds in slot `s` of quarter `q` of `count`
    (`stacking`)."""
    return q + count * s if order == QUARTER_MINOR else slots * q + s


def quarters(op, threads: int) -> int:
    """The quarters of the wave a multiplication of `threads` lanes spans:
    `H` in the module."""
    return threads // op.n


def lane_cell(writer, op, threads, t):
    """`(element, column)`: which multiplication's element and which column
    of the tile the lane's `t` names as an M row of `op`'s A -- quarter `q`
    and slot `s` of D (`stacked_row`), the element ``q // H``, the column
    ``4 (q % H) + s`` -- as index expressions."""
    from tensorforge.backend.pir.core import INDEX
    H = quarters(op, threads)
    count = op.wave // op.n
    slots = op.m // count
    minor = stacking(op) == QUARTER_MINOR
    if H == 1:
        if minor:
            return (writer.op('rem', INDEX, t, count, hint='e'),
                    writer.op('div', INDEX, t, count, hint='c'))
        return (writer.op('div', INDEX, t, slots, hint='e'),
                writer.op('rem', INDEX, t, slots, hint='c'))
    m = writer.op('rem', INDEX, t, op.n, hint='m')
    if minor:
        q = writer.op('rem', INDEX, m, count, hint='q')
        s = writer.op('div', INDEX, m, count, hint='s')
    else:
        q = writer.op('div', INDEX, m, slots, hint='q')
        s = writer.op('rem', INDEX, m, slots, hint='s')
    element = writer.op('div', INDEX, q, H, hint='e')
    column = writer.op('add', INDEX,
                       writer.op('mul', INDEX,
                                 writer.op('rem', INDEX, q, H, hint='h'),
                                 slots, hint='c'), s, hint='c')
    return element, column


def emulated(ctx) -> bool:
    """Whether an F32 product is emulated in BF16 terms on the stacked tiles
    (`Options.amd_matrix_k4_emulate`)."""
    options = getattr(ctx, 'get_user_options', None)
    return bool(options) and getattr(options(), 'amd_matrix_k4_emulate',
                                     False) in (True, 'bf16x6')


def instruction(threads: int, accumulator, ctx):
    """The instruction the scheme issues here, or `None`: asked for, defined
    for the accumulator, available on the target, and stacking at this lane
    count (`stacking`).  F32 emulated in BF16 terms takes `EMULATED`'s, at
    sixteen lanes."""
    if requested(ctx) is None or threads not in LANES:
        return None
    name = INSTRUCTIONS.get(accumulator)
    if emulated(ctx) and accumulator in EMULATED:
        if threads != LANES[0]:
            return None
        name = EMULATED[accumulator]
    op = next((o for o in MATRIX_OPS if o.builtin == name), None)
    if op is None or not op.available_for(accumulator, ctx):
        return None
    return op if fits(op, threads) else None


def tile_columns(op, threads: int) -> int:
    """Output columns one issue covers: M over the multiplications a wave."""
    return op.m // (op.wave // threads)


def stacks(shape, ctx) -> bool:
    """Whether the scheme serves `shape`.

    The operator has to be the same for every multiplication of the wave
    (`a_uniform`): one issue multiplies all its elements by one B.  Not
    sparse, which no fragment reads; one scalar a lane at lead width one, which
    is the image D lands in; SPMD, where a wave is lanes; and, for the `shard`
    route, a per-element operand in shared memory (`b_shared`).
    """
    if (shape.sparse or shape.explicit_simd or not shape.a_uniform
            or shape.a_parts not in ((1, BF16_TERMS) if emulated(ctx) else
                                     (1,))
            or bitlayout.packed(shape.lead_layout)):
        return False
    if instruction(shape.threads, shape.accumulator, ctx) is None:
        return False
    # `shard` alone reads the per-element operand nowhere but in shared
    # memory: asked of an operand held in registers -- the time derivative's
    # images -- it leaves the product to the DPP chain whole, rather than
    # planning a span it would then decline (`a_source`), which splits the
    # chain in two and runs the wave in step for nothing.
    return requested(ctx) != 'shard' or shape.b_shared


def stacked_reads_shared(shape, ctx) -> bool:
    """Whether a product the scheme takes reads a per-element operand that an
    earlier operation stored in shared memory there (`MatrixPaths.
    reads_shared`): the `shard` route, which is what `true` takes where the
    operand is in shared memory at all."""
    return stacks(shape, ctx) and requested(ctx) in ('shard', 'auto')


def padded(ctx) -> bool:
    """Whether the columns past the last whole tile go into one more tile,
    padded (`Options.amd_matrix_k4_pad`), rather than through the DPP
    chain -- always where products keep their sums in the accumulators
    across a chain (`Options.mma_d_layout`), whose columns all have to be
    there (`stacked_chain_tile`)."""
    options = getattr(ctx, 'get_user_options', None)
    if options is None:
        return False
    asked = options()
    return bool(getattr(asked, 'amd_matrix_k4_pad', False)
                or getattr(asked, 'mma_d_layout', False))


def pipeline_depth(ctx) -> int:
    """Steps of the contraction whose fragments are read ahead of a step's
    issues (`Options.amd_matrix_k4_pipeline`): 0, or 1 and 2 -- the levels
    that read one step ahead."""
    options = getattr(ctx, 'get_user_options', None)
    level = getattr(options(), 'amd_matrix_k4_pipeline', 0) if options else 0
    return 1 if level else 0


def interleaved(ctx) -> bool:
    """Whether a step's issues and the reads of the next are interleaved by
    scheduling groups (`Options.amd_matrix_k4_pipeline` at 2)."""
    options = getattr(ctx, 'get_user_options', None)
    return bool(options) and getattr(options(), 'amd_matrix_k4_pipeline',
                                     0) >= 2


#: `sched_group_barrier` masks (LLVM's AMDGPU backend): the matrix
#: instructions, and the reads from memory and from the data share -- a
#: fragment's loads, its gathers and its shard reads alike.
SCHED_MFMA = 0x0008
SCHED_READS = 0x0020 | 0x0100


def _interleave(writer, count: int) -> None:
    """A step's `count` issues each followed by a group of reads: the
    scheduler lays the next step's fragment reads between this step's matrix
    instructions rather than ahead of all or behind all of them, which is
    how CK hides them (`__builtin_amdgcn_sched_group_barrier`).  The groups
    are a pattern, not a count the scheduler has to meet: a group it cannot
    fill stays short."""
    for _ in range(count):
        writer(f'__builtin_amdgcn_sched_group_barrier({SCHED_MFMA}, 1, 0);',
               accesses=())
        writer(f'__builtin_amdgcn_sched_group_barrier({SCHED_READS}, 2, 0);',
               accesses=())


def stacked_plan(shape, n: int, ctx):
    """Whole tiles of columns on the instruction, the rest on the DPP chain.

    The rest is SeisSol's ninth column: a tile padded to compute it would
    spend three quarters of its issues on zeros, where the chain spends a
    product a row and step -- and reads the operator once more, a row and a
    step at a time.  Which costs more is a measurement, so the padded tile is
    an option (`padded`).  A product narrower than a tile stays on the chain
    either way at sixteen lanes; wider, where a tile is eight or sixteen
    columns, a padded one takes it if it fills half of it -- SeisSol's nine
    columns at sixty-four lanes.
    """
    op = instruction(shape.threads, shape.accumulator, ctx)
    width = tile_columns(op, shape.threads)
    edge = (n // width) * width
    if edge <= 0:
        if (padded(ctx) and quarters(op, shape.threads) > 1
                and 2 * n >= width):
            return whole(Strategy.MATRIX, n)
        return whole(Strategy.DPP, n)
    if edge >= n or padded(ctx):
        return whole(Strategy.MATRIX, n)
    return (Span(Strategy.MATRIX, 0, edge), Span(Strategy.DPP, edge, n))


@dataclass(frozen=True)
class ChainTile:
    """What a chain of products keeps its sum in on this scheme
    (`MatrixPaths.chain_tile`): `m` operator rows and `n` output columns an
    issue, in `d`, walked a slot -- `block` rows -- at a time.  `mode` names
    the instruction, which the members of one chain share."""
    m: int
    n: int
    d: Datatype
    mode: str
    block: int


def stacked_chain_tile(shape, extents, ctx):
    """The tile a chain keeps its sum in, or why there is none.

    The D layout is the register image (`stacking`), so the output's old
    value is read into the accumulators where the image would hold it, every
    product of the chain issues into them, and the last writes them where
    they were read: one accumulator set for the chain where each product had
    its own and an epilogue that added the old value.  Every column has to
    be in the accumulators for that -- whole tiles, or the last one padded
    (`padded`): a column the DPP chain computes is added nowhere else.  And
    the operator has to be read by its own steps (`a_wave`), since a member
    that declined the instruction would leave the chain's accumulators
    without its products.
    """
    if not stacks(shape, ctx):
        return 'not on the stacked matrix path'
    if not shape.a_wave:
        return 'the operator is not read by its own steps'
    op = instruction(shape.threads, shape.accumulator, ctx)
    width = tile_columns(op, shape.threads)
    columns = extents[1]
    if columns % width and not padded(ctx):
        return 'the columns past the last whole tile go to the DPP chain'
    return ChainTile(m=op.n, n=width, d=shape.accumulator,
                     mode=op.builtin, block=shape.threads)


def serves(ops, ctx) -> bool:
    """Whether a matrix span over `ops` is this scheme's: `stacks`, asked of
    the operands the emission holds rather than of the shape the plan was
    made from.  The two agree; a span `stacked_plan` gave the instruction
    reaches here only through it."""
    return (ops.sparse is None and ops.a_uniform and ops.lead_width == 1
            and ops.a_parts in ((1, BF16_TERMS) if emulated(ctx) else (1,))
            and instruction(ops.threads, ops.accumulator, ctx) is not None)


def matmul_stacked(writer, ops, ctx, span) -> bool:
    """`C[:, span] += A B` through the stacked instruction.

    False, with nothing emitted, where the operands are not what the scheme
    reads: no `A_wave` or `A_lanes` (an operator not in memory in its own
    order), a wave whose multiplications do not run in step, operands of
    another type, a span that is not whole tiles -- but for the last one
    where it is padded (`padded`) -- or a per-element operand whose fragment
    a route cannot reach.  The caller then takes the span through the DPP
    chain, which reads all of these -- rather than handing the whole product
    to the nest, which a decline would.
    """
    threads = ops.threads
    op = instruction(threads, ops.accumulator, ctx)
    if op is None:
        return False
    mults = op.wave // threads
    width = tile_columns(op, threads)
    # The operator's fragment: `A_wave` where a multiplication is a quarter,
    # a read of its own at the rows of a tile of sixteen where it is more.
    operator = ops.A_wave if quarters(op, threads) == 1 else ops.A_lanes
    if op.k > op.wave // op.n:
        # The emulated tile reads the operator by row and step, or split.
        operator = ops.A_slot if ops.a_parts == BF16_TERMS else ops.A_lanes
    # The wave's multiplications in step, where it holds more than one.
    if (operator is None or (mults > 1 and not ops.lockstep)
            or (ops.a, ops.b) != (ops.accumulator, ops.accumulator)
            or span.start % width
            or (len(span) % width and not (padded(ctx)
                                           and span.stop == ops.n))):
        return False
    with writer.speculative() as spec:
        if not _emit(writer, ops, ctx, span, op, mults, width):
            spec.discard()
            return False
    return True


@dataclass(frozen=True)
class _Lanes:
    """Where the lane sits (the module's names), as index values: `t` its
    lane in the multiplication, `p` the multiplication in the wave, `n` its
    lane in its quarter, `h` the quarter in the multiplication, `q` the
    quarter in the wave -- the step of the contraction its A and B carry --,
    of `H` quarters of `quarter` lanes a multiplication."""
    t: object
    p: object
    n: object
    h: object
    q: object
    H: int
    quarter: int


def _lanes(writer, t, p, H, quarter):
    from tensorforge.backend.pir.core import INDEX
    if H == 1:
        return _Lanes(t=t, p=p, n=t, h=0, q=p, H=1, quarter=quarter)
    h = writer.op('div', INDEX, t, quarter, hint='h')
    n = writer.op('rem', INDEX, t, quarter, hint='n')
    q = writer.op('add', INDEX, writer.op('mul', INDEX, p, H, hint='q'), h,
                  hint='q')
    return _Lanes(t=t, p=p, n=n, h=h, q=q, H=H, quarter=quarter)


def _emit(writer, ops, ctx, span, op, mults, width) -> bool:
    from tensorforge.backend.pir.core import BOOL, INDEX, ScalarType
    if op.k > op.wave // op.n:
        return _emit_emulated(writer, ops, ctx, span, op, mults, width)
    C, A, chain = ops.C, ops.A, ops.chain
    threads, M, K, kx = ops.threads, ops.lead_slots, ops.k, ops.kx
    H, Q = quarters(op, threads), op.n
    ftype = ScalarType(ops.accumulator)
    acctype = ScalarType(ops.accumulator, op.d.per_lane)
    zero = writer.const(0.0, ftype)
    # The lane's place in the wave's fragments (`stacking`): its `t` names the
    # element and the column its A carries, its quarter `q` the step of both.
    t = writer.thread_id('x')
    p = writer.op('rem', INDEX, writer.thread_id('y'), mults, hint='m')
    lanes = _lanes(writer, t, p, H, Q)
    q = lanes.q
    tiles = list(range(span.start, span.stop, width))
    # Tiles of sixteen rows: a multiplication's slot is `H` of them.
    row_tiles = H * M

    def steps(s0):
        """The operator's own steps a tile at position `s0` covers: the
        positions count from the block holding the contraction's first step,
        `kx` before it (`MatmulOperands.kx`)."""
        return max(0, s0 - kx), min(K, s0 + op.k - kx)

    def known_zero(rt, s0):
        lo, hi = steps(s0)
        return (lo >= hi or (ops.A_zero is not None and ops.A_zero(
            Q * rt, Q * (rt + 1), lo, hi)))

    def by_step(values):
        """The value of the lane's own step out of one per step, `q` picking:
        a tile at the edge of the contraction, where a step past it reads a
        zero rather than a neighbor's number."""
        out = values[-1]
        for i in range(len(values) - 2, -1, -1):
            out = writer.op('select', ftype, writer.op('eq', BOOL, q, i,
                                                       hint='g'),
                            values[i], out, hint='ks')
        return out

    def operator_fragment(rt, s0):
        """B: the operator row `16 rt + n` at step `s0 - kx + q`."""
        r0 = s0 - kx
        if H > 1:
            return _operator_lanes(writer, ops, lanes, rt, r0)
        if r0 >= 0 and r0 + mults <= K:
            got = ops.A_wave(writer, rt, r0, mults)
            if got is not None and got is not False:
                return got
        values = []
        for i in range(mults):
            got = (A(writer, None, rt, r0 + i) if 0 <= r0 + i < K else None)
            values.append(got if got is not None and got is not False
                          else zero)
        return by_step(values)

    if chain is not None and chain.phase in ('open', 'close'):
        # Ahead of a merged run's loop, or behind it: the chain's
        # accumulators read in or written back, and nothing issued -- the
        # loop carries them through its trips (`AccumulatorChain.phase`).
        if chain.phase == 'open':
            _open_chain(writer, ops, op, chain, tiles, acctype, ftype, lanes)
        else:
            _close_chain(writer, ops, op, chain, chain.state['accs'], ftype,
                         lanes)
        return True

    source = a_source(writer, ops, ctx, op, width, lanes)
    if source is None:
        return False

    def element_fragment(c0, s0):
        """A: the column `c0 + c` of element `e` the lane's `t` names
        (`lane_cell`), at position `s0 + q`, zero for a position outside the
        contraction -- and for a column past the span, in a padded last
        tile."""
        got = source(c0, s0, min(width, span.stop - c0))
        if got is None:
            return None
        edge = [i for i in range(op.k) if not kx <= s0 + i < K + kx]
        if not edge:
            return got
        return by_step([zero if i in edge else got for i in range(op.k)])

    # A chain's accumulators outlive the product: declared where every
    # member reaches them, not in a scope of the first.
    with (writer.AnonymousScope() if chain is None
          else contextlib.nullcontext()):
        if chain is None:
            acc = {(rt, c0): writer.declare(acctype, hint='acc')
                   for rt in range(row_tiles) for c0 in tiles}
        elif chain.opens and chain.phase == 'all':
            acc = _open_chain(writer, ops, op, chain, tiles, acctype, ftype,
                              lanes)
        else:
            acc = dict(chain.state.get('accs', {}))
            if any((rt, c0) not in acc
                   for rt in range(row_tiles) for c0 in tiles):
                from tensorforge.common.exceptions import InternalError
                raise InternalError(
                    'a member of a chain reaches past the accumulators its '
                    'first member declared')
        # The contraction outermost and every accumulator inside: the
        # per-element fragments are built once for all tiles of rows, and an
        # accumulator is read again only behind seven others.
        steps_ = [(s0, [rt for rt in range(row_tiles)
                        if not known_zero(rt, s0)])
                  for s0 in range(0, K + kx, op.k)]
        steps_ = [(s0, live) for s0, live in steps_ if live]

        def fetch(s0, live):
            """A step's fragments: the elements' for every tile of columns,
            the operator's for every tile of rows it issues; `None` where a
            route or a read declines."""
            frags = {}
            for c0 in tiles:
                frags[c0] = element_fragment(c0, s0)
                if frags[c0] is None:
                    return None
            ops_ = {}
            for rt in live:
                ops_[rt] = operator_fragment(rt, s0)
                if ops_[rt] is None:
                    return None
            return frags, ops_

        depth = pipeline_depth(ctx)
        ahead = [fetch(s0, live) for s0, live in steps_[:depth]]
        for i, (s0, live) in enumerate(steps_):
            if not depth:
                # A step's element fragments, then each tile of rows' operator
                # fragment ahead of its issues.
                frags = {}
                for c0 in tiles:
                    frags[c0] = element_fragment(c0, s0)
                    if frags[c0] is None:
                        return False
                for rt in live:
                    b = operator_fragment(rt, s0)
                    if b is None:
                        return False
                    for c0 in tiles:
                        acc[(rt, c0)] = writer.call(
                            op.callee, acctype, frags[c0], b, acc[(rt, c0)],
                            0, 0, 0, hint='acc', movable=False,
                            materialize=True)
                continue
            # The fragments of the step `depth` on, read before this step's
            # issues: their loads have the issues to land behind
            # (`Options.amd_matrix_k4_pipeline`).
            if i + depth < len(steps_):
                ahead.append(fetch(*steps_[i + depth]))
            got = ahead.pop(0)
            if got is None:
                return False
            frags, operators = got
            for rt in live:
                for c0 in tiles:
                    acc[(rt, c0)] = writer.call(
                        op.callee, acctype, frags[c0], operators[rt],
                        acc[(rt, c0)], 0, 0, 0, hint='acc', movable=False,
                        materialize=True)
            if interleaved(ctx) and i + depth < len(steps_):
                _interleave(writer, len(live) * len(tiles))
        if chain is not None:
            chain.state['accs'] = acc
            if chain.closes and chain.phase == 'all':
                _close_chain(writer, ops, op, chain, acc, ftype, lanes)
            return True
        if H > 1:
            _relayout(writer, ops, op, acc, tiles, span, ftype, lanes)
            return True
        for rt in range(M):
            for c0 in tiles:
                for jj in range(min(width, span.stop - c0)):
                    C(writer, writer.extract(acc[(rt, c0)], jj, ftype), rt,
                      c0 + jj)
    return True


def _emit_emulated(writer, ops, ctx, span, op, mults, width) -> bool:
    """`C[:, span] += A B` in BF16 terms (`Options.amd_matrix_k4_emulate`).

    The tile is the F32 one's -- the same M, N and D (`stacking`) -- but an
    issue covers sixteen steps, four in each lane: lane `n` of quarter `q`
    carries steps ``4 q .. 4 q + 3`` of the operator's row and of its
    element's column.  Both are split into three BF16 terms each
    (`tensorforge::splitFloatx4BF16`), and six term products -- the pairs
    whose order of magnitude reaches F32's, smallest first
    (`split.products`) -- add into the F32 accumulator.  The operator's
    steps are one read each at the tile's rows (`MatmulOperands.A_lanes`).
    Sixteen lanes a multiplication."""
    from tensorforge.backend.pir.core import BOOL, INDEX, ScalarType
    from ...split import products
    from .codegen import SPLIT_BF16
    C, chain = ops.C, ops.chain
    threads, M, K, kx = ops.threads, ops.lead_slots, ops.k, ops.kx
    count = op.wave // op.n
    vector = op.k // count
    # The operator as the host split it (`amd.prepared_order`): k-quads of
    # its three terms; else split here, read by row and step.
    prepared = ops.A_slot is not None and ops.a_parts == BF16_TERMS
    if (not prepared and ops.A_lanes is None) or quarters(op, threads) != 1:
        return False
    ftype = ScalarType(ops.accumulator)
    acctype = ScalarType(ops.accumulator, op.d.per_lane)
    termtype = ScalarType(Datatype.I16, vector)
    zero = writer.const(0.0, ftype)
    t = writer.thread_id('x')
    p = writer.op('rem', INDEX, writer.thread_id('y'), mults, hint='m')
    lanes = _lanes(writer, t, p, 1, op.n)
    q = lanes.q
    tiles = list(range(span.start, span.stop, width))
    pairs = products(BF16_TERMS)
    # The lane's steps of an issue: `4 q + i`.
    offsets = [writer.op('add', INDEX,
                         writer.op('mul', INDEX, q, vector, hint='ko'), i,
                         hint='ko') for i in range(vector)]

    def known_zero(rt, s0):
        lo, hi = max(0, s0 - kx), min(K, s0 + op.k - kx)
        return (lo >= hi or (ops.A_zero is not None and ops.A_zero(
            op.n * rt, op.n * (rt + 1), lo, hi)))

    def split(values, hint):
        out = [writer.declare(termtype, hint=hint)
               for _ in range(BF16_TERMS)]
        writer.call_stmt(SPLIT_BF16, *out, *values, writes=tuple(out))
        return out

    if chain is not None and chain.phase in ('open', 'close'):
        if chain.phase == 'open':
            _open_chain(writer, ops, op, chain, tiles, acctype, ftype, lanes)
        else:
            _close_chain(writer, ops, op, chain, chain.state['accs'], ftype,
                         lanes)
        return True
    source = a_source(writer, ops, ctx, op, width, lanes)
    if source is None:
        return False

    def element_terms(c0, s0):
        """The lane's four steps of its element's column, zero outside the
        contraction, as terms."""
        values = []
        for i, offset in enumerate(offsets):
            got = source(c0, s0, min(width, span.stop - c0), offset=offset)
            if got is None:
                return None
            # Quarters whose step `4 q + i` falls outside take a zero.
            outside = [r for r in range(count)
                       if not kx <= s0 + vector * r + i < K + kx]
            for r in outside:
                got = writer.op('select', ftype,
                                writer.op('eq', BOOL, q, r, hint='g'),
                                zero, got, hint='ks')
            values.append(got)
        return split(values, 'at')

    def operator_terms(rt, s0):
        if prepared:
            return _prepared_terms(writer, ops, lanes, rt, s0 - kx, vector,
                                   count)
        values = []
        for offset in offsets:
            got = _operator_lanes(writer, ops, lanes, rt, s0 - kx,
                                  offset=offset, span=op.k)
            if got is None:
                return None
            values.append(got)
        return split(values, 'bt')

    with (writer.AnonymousScope() if chain is None
          else contextlib.nullcontext()):
        if chain is None:
            acc = {(rt, c0): writer.declare(acctype, hint='acc')
                   for rt in range(M) for c0 in tiles}
        elif chain.opens and chain.phase == 'all':
            acc = _open_chain(writer, ops, op, chain, tiles, acctype, ftype,
                              lanes)
        else:
            acc = dict(chain.state.get('accs', {}))
        for s0 in range(0, K + kx, op.k):
            live = [rt for rt in range(M) if not known_zero(rt, s0)]
            if not live:
                continue
            frags = {}
            for c0 in tiles:
                frags[c0] = element_terms(c0, s0)
                if frags[c0] is None:
                    return False
            for rt in live:
                b = operator_terms(rt, s0)
                if b is None:
                    return False
                for c0 in tiles:
                    for i, j in pairs:
                        acc[(rt, c0)] = writer.call(
                            op.callee, acctype, frags[c0][i], b[j],
                            acc[(rt, c0)], 0, 0, 0, hint='acc',
                            movable=False, materialize=True)
        if chain is not None:
            chain.state['accs'] = acc
            if chain.closes and chain.phase == 'all':
                _close_chain(writer, ops, op, chain, acc, ftype, lanes)
            return True
        for rt in range(M):
            for c0 in tiles:
                for jj in range(min(width, span.stop - c0)):
                    C(writer, writer.extract(acc[(rt, c0)], jj, ftype), rt,
                      c0 + jj)
    return True


def _prepared_terms(writer, ops, lanes, rt, r0, vector, count):
    """The emulated tile's B out of the operator the host split
    (`amd.prepared_order`): the quad of the lane's quarter -- ``r0 / 4 + q``
    of tile `rt` -- in each of the three terms' planes, one read a term,
    packed to BF16 (`tensorforge::packBF16x4`).  A quad past either end of the
    contraction reads the nearest one: the element's fragment is zero
    there."""
    from tensorforge.backend.pir.core import INDEX, ScalarType
    from .codegen import PACK_BF16
    threads, slots = ops.threads, ops.lead_slots
    quads = -(-ops.k // vector)
    first = r0 // vector
    shift = writer.op('mul', INDEX, lanes.q, slots, hint='r')
    if first < 0 or first + count > quads:
        # The quarters' quads clamped into the operand: `first + q` in
        # `[0, quads)`.
        quad = writer.op('min', INDEX,
                         writer.op('max', INDEX,
                                   writer.op('add', INDEX, lanes.q, first,
                                             hint='r'), 0, hint='r'),
                         quads - 1, hint='r')
        shift = writer.op('mul', INDEX, quad, slots, hint='r')
        first = 0
    got = ops.A_slot(writer, (first * slots + rt) * threads * vector,
                     parts=BF16_TERMS, shift=shift, width=vector)
    if not got:
        return None
    termtype = ScalarType(Datatype.I16, vector)
    return [writer.call(PACK_BF16, termtype, part, hint='bt') for part in got]


def _operator_lanes(writer, ops, lanes, rt, r0, offset=None, span=4):
    """B where a multiplication spans `H > 1` quarters: lane `n` of quarter
    `q` reads the operator at row ``16 rt + n`` and step ``r0 + q`` -- the
    tile's rows in every quarter, where the lane's own rows are those of
    its slot (`MatmulOperands.A_lanes`).  A row outside the operation's
    window reads its nearest row and takes a zero: no lane's guard keeps it
    out here, and in a chain the row is another member's, which this one has
    to add nothing to.  A step past the contraction reads its last step,
    which the element's fragment then multiplies by zero (`_emit`).  `None`
    where the read declines."""
    from tensorforge.backend.pir.core import BOOL, INDEX, ScalarType
    first = ops.lead_first
    last = first + ops.lead_elements - 1
    own = lanes.q if offset is None else offset
    lo, hi = lanes.quarter * rt, lanes.quarter * (rt + 1) - 1
    row = writer.op('add', INDEX, lanes.n, lo, hint='br') if lo else lanes.n
    inside = None
    if lo < first or hi > last:
        inside = writer.op('and', BOOL,
                           writer.op('ge', BOOL, row, first, hint='g'),
                           writer.op('le', BOOL, row, last, hint='g'),
                           hint='g')
        row = writer.op('min', INDEX, writer.op('max', INDEX, row, first,
                                                hint='br'), last, hint='br')
    step = writer.op('add', INDEX, own, r0, hint='bk') if r0 else own
    if r0 < 0 or r0 + span > ops.k:
        step = writer.op('min', INDEX, writer.op('max', INDEX, step, 0,
                                                 hint='bk'),
                         ops.k - 1, hint='bk')
    got = ops.A_lanes(writer, 0, row, step)
    if got is None or got is False:
        return None
    if inside is None:
        return got
    ftype = ScalarType(ops.a)
    return writer.op('select', ftype, inside, got, writer.const(0.0, ftype),
                     hint='bz')


def _chain_rows(writer, chain, rt, lanes):
    """Row `16 rt + n` of the walk -- the lane's in tile `rt` -- and which
    lanes hold a row of the chain's window: `(row, predicate)`, the
    predicate None where all of them do, and `None` where none does.  At
    sixteen lanes a multiplication the tile is the lane's slot and `n` its
    lane."""
    from tensorforge.backend.pir.core import BOOL, INDEX
    lo, hi = chain.rows
    size = lanes.quarter
    first = size * rt
    if first + size <= lo or first >= hi:
        return None
    row = (writer.op('add', INDEX, lanes.n, first, hint='dr') if first
           else lanes.n)
    cond = None
    if first < lo:
        cond = writer.op('ge', BOOL, row, lo, hint='g')
    if first + size > hi:
        upper = writer.op('lt', BOOL, row, hi, hint='g')
        cond = upper if cond is None else writer.op('and', BOOL, cond, upper,
                                                    hint='g')
    return row, cond


def _chain_cells(writer, ops, op, c0, cond, lanes):
    """`[(column, predicate)]` for the slots of a tile at `c0`: the column D
    gives the slot in the lane -- `c0 + s`, or `c0 + 4 h + s` where a
    multiplication spans `H` quarters -- and the lanes where it is one of the
    output's (`cond` and the column inside it); `None` for a slot no lane
    keeps."""
    from tensorforge.backend.pir.core import BOOL, INDEX
    # The D cells a lane holds, not `m / k`: the two agree for the
    # four-step tiles, and the BF16 tile's sixteen steps made it one -- a
    # chain read and wrote one column of four (`amd_matrix_k4_emulate`).
    slots = op.d.per_lane
    out = []
    for s in range(slots):
        if lanes.H == 1:
            out.append((c0 + s, cond) if c0 + s < ops.n else None)
            continue
        if c0 + s >= ops.n:
            out.append(None)
            continue
        column = writer.op('add', INDEX,
                           writer.op('mul', INDEX, lanes.h, slots, hint='c'),
                           c0 + s, hint='c')
        if c0 + slots * (lanes.H - 1) + s < ops.n:
            out.append((column, cond))
            continue
        inside = writer.op('lt', BOOL, column, ops.n, hint='g')
        out.append((column, inside if cond is None else writer.op(
            'and', BOOL, cond, inside, hint='g')))
    return out


def _open_chain(writer, ops, op, chain, tiles, acctype, ftype, lanes):
    """The chain's accumulators, holding the output's old value: for every
    tile of the chain's window and every tile of columns, the cells the D
    layout gives the lane -- at sixteen lanes the register image's, row
    `16 rt + t` and column `c0 + s` in slot `s`; wider, row `16 rt + n` and
    column `c0 + 4 h + s` (the module).  A row outside the window and a
    padded column start at zero and are never written back."""
    zero = writer.const(0.0, ftype)
    count = max(lanes.H * ops.lead_slots,
                -(-chain.rows[1] // lanes.quarter))
    acc = {}
    for rt in range(count):
        rows = _chain_rows(writer, chain, rt, lanes)
        for c0 in tiles:
            if rows is None:
                acc[(rt, c0)] = writer.declare(acctype, hint='acc')
                continue
            row, cond = rows
            parts = [chain.load(writer, row, cell[0], cell[1])
                     if cell is not None else zero
                     for cell in _chain_cells(writer, ops, op, c0, cond,
                                              lanes)]
            acc[(rt, c0)] = writer.pack(acctype, *parts, hint='acc')
    chain.state.clear()
    chain.state['accs'] = acc
    return dict(acc)


def _close_chain(writer, ops, op, chain, acc, ftype, lanes):
    """The chain's sums written where `_open_chain` read the old values."""
    for (rt, c0), value in sorted(acc.items()):
        rows = _chain_rows(writer, chain, rt, lanes)
        if rows is None:
            continue
        row, cond = rows
        for s, cell in enumerate(_chain_cells(writer, ops, op, c0, cond,
                                              lanes)):
            if cell is None:
                continue
            part = writer.extract(value, s, ftype)
            with (writer.if_(cell[1]) if cell[1] is not None
                  else contextlib.nullcontext()):
                chain.store(writer, part, row, cell[0])


def _relayout(writer, ops, op, acc, tiles, span, ftype, lanes):
    """The image's cells out of the tiles', where a multiplication spans
    `H > 1` quarters and the product keeps no chain.

    Tile `H sigma + r` holds in quarter `h` the rows of quarter `r` of slot
    `sigma` -- its sixteen rows -- for the columns ``c0 + 4 h + s``; the
    image wants in quarter `h` its own rows for every column.  So quarter `h`
    keeps tile `H sigma + h`, and for every `d < H` reads its rows of the
    columns of quarter `h ^ d` out of that quarter, which put them in the
    register it hands on: one lane gather a slot and `d`, and selects by
    the lane's quarter.  The cells then go out through the epilogue
    (`MatmulOperands.C`) at the lane's own row."""
    from tensorforge.backend.pir.core import BOOL, INDEX
    H, size, slots = lanes.H, lanes.quarter, op.d.per_lane
    text = f'tensorforge::gather<{op.wave}>({{0}}, {{1}})'
    lane = writer.op('add', INDEX, lanes.t,
                     writer.op('mul', INDEX, lanes.p, ops.threads, hint='l'),
                     hint='l')
    sources = {d: writer.op('bitxor', INDEX, lane, size * d, hint='src')
               for d in range(1, H)}
    mine = {r: writer.op('eq', BOOL, lanes.h, r, hint='g') for r in range(H)}

    def pick(values):
        """`values[h]` in the lanes of quarter `h`."""
        out = values[-1]
        for r in range(H - 2, -1, -1):
            out = writer.op('select', ftype, mine[r], values[r], out,
                            hint='rl')
        return out

    for sigma in range(ops.lead_slots):
        for c0 in tiles:
            blocks = [acc[(H * sigma + r, c0)] for r in range(H)]
            for s in range(slots):
                parts = [writer.extract(b, s, ftype) for b in blocks]
                own = pick(parts)
                got = {}
                for d in range(1, H):
                    send = pick([parts[r ^ d] for r in range(H)])
                    got[d] = writer.rawexpr(text, send, sources[d],
                                            type_=ftype, hint='ga',
                                            pure=True, movable=True,
                                            crosslane=True)
                for g in range(H):
                    column = c0 + slots * g + s
                    if column >= span.stop:
                        continue
                    ops.C(writer, pick([own if r == g else got[r ^ g]
                                        for r in range(H)]), sigma, column)


def a_source(writer, ops, ctx, op, width, lanes):
    """`fragment(c0, s0, columns)`: the lane's A for the tile of columns from
    `c0` at positions from `s0`, before the edge of the contraction is cut
    off, zero past the first `columns` of the tile -- or `None` where no
    route reaches it.

    The route `Options.amd_matrix_k4` names; `true` the shard where the
    operand is in shared memory (`MatmulOperands.B_shard`), the gathers where
    it is not.  `shard` alone declines an operand that is not there, so that
    a measurement of the route measures the route.
    """
    route = requested(ctx)
    if route in ('shard', 'auto') and ops.B_shard is not None:
        return _sharded(writer, ops, op, width, lanes)
    if route == 'shard':
        return None
    return _gathered(writer, ops, op, width, lanes)


def _sharded(writer, ops, op, width, lanes):
    """The `shard` route: A out of the shared memory the operand sits in.

    An earlier operation of the section stored the per-element operand there,
    every multiplication its own copy in its own arena, `mult_stride`
    elements after its predecessor's, and the operation reads it in place
    (`MatrixPaths.reads_shared`).  The lane's A is position ``s0 + q`` of the
    element `e` and column ``c0 + c`` its `t` names (`lane_cell`): one read
    of that multiplication's copy, ``e - p`` arenas from the lane's own, at
    the lane's own row and column (`MatmulOperands.B_shard`) -- where the
    transpose route gathers `width` registers and selects among them.

    The copies are other multiplications' stores, so the reads stand behind a
    rendezvous of the wave.  On a wave of sixteen-lane multiplications it is
    no instruction, and it owes the compiler a fence (`Lexic.handoff_fence`):
    a lane that stored a cell and reads another at a run-time distance is,
    to the compiler, one thread that may read before it writes.

    A step outside the contraction reads the nearest step inside it, which
    the edge then cuts off (`_emit`): the address stays in the copy.
    """
    from tensorforge.backend.pir.core import BOOL, INDEX, ScalarType
    threads, K, kx = ops.threads, ops.k, ops.kx
    ftype = ScalarType(ops.accumulator)
    q = lanes.q
    element, column = lane_cell(writer, op, threads, lanes.t)
    mult = writer.op('sub', INDEX, element, lanes.p, hint='mo')
    fenced = []

    def fragment(c0, s0, columns, offset=None):
        """`offset`, the lane's step from `s0`, is its quarter `q` but where
        a lane carries several steps of an issue (`_emit_emulated`)."""
        if not fenced:
            writer.barrier('wave', threads=threads, handoff=True)
            fenced.append(True)
        own = q if offset is None else offset
        r0 = s0 - kx
        base = min(max(r0, 0), K - 1)
        if 0 <= r0 and r0 + op.k <= K:
            rows = own
        else:
            step = writer.op('add', INDEX, own, r0, hint='ks')
            step = writer.op('min', INDEX,
                             writer.op('max', INDEX, step, 0, hint='ks'),
                             K - 1, hint='ks')
            rows = writer.op('sub', INDEX, step, base, hint='ks')
        if columns >= width:
            got = ops.B_shard(writer, c0, base, rows, column, mult)
            return got if got is not None and got is not False else None
        # A padded tile: the columns past the span read the last real one,
        # inside the copy, and take a zero.
        inside = writer.op('lt', BOOL, column, columns, hint='g')
        got = ops.B_shard(writer, c0, base, rows,
                          writer.op('min', INDEX, column, columns - 1,
                                    hint='c'), mult)
        if got is None or got is False:
            return None
        return writer.op('select', ftype, inside, got,
                         writer.const(0.0, ftype), hint='af')
    return fragment


def _gathered(writer, ops, op, width, lanes):
    """The `transpose` route: A out of the registers the operand is held in.

    The nest's image of the per-element operand holds position `s` of
    column `j` in lane `s % threads` of block `s // threads`
    (`MatmulOperands.B`).  The lane's A is position ``s0 + q`` of the
    element `e` and column ``c0 + c`` its `t` names (`lane_cell`): wave lane
    ``threads e + (s0 + q) % threads``, register ``B(c0 + c, ...)``.  A lane
    gather (`ds_bpermute`) reads the same register out of every lane it
    names, and the column differs between the lanes of one fragment, so each
    column of the tile is gathered on its own and the lane takes its own:
    `width` gathers and a select a fragment.
    """
    from tensorforge.backend.pir.core import BOOL, INDEX, ScalarType
    threads = ops.threads
    ftype = ScalarType(ops.accumulator)
    element, column = lane_cell(writer, op, threads, lanes.t)
    base = writer.op('add', INDEX,
                     writer.op('mul', INDEX, element, threads, hint='e'),
                     lanes.q, hint='src')
    text = f'tensorforge::gather<{op.wave}>({{0}}, {{1}})'

    def fragment(c0, s0, columns, offset=None):
        """`offset` as in `_sharded`."""
        if offset is None:
            lane = writer.op('add', INDEX, base, s0 % threads, hint='src')
        else:
            lane = writer.op('add', INDEX,
                             writer.op('mul', INDEX, element, threads,
                                       hint='e'),
                             writer.op('add', INDEX, offset, s0 % threads,
                                       hint='src'), hint='src')
        gathered = []
        for jj in range(width):
            if jj >= columns:
                # A padded tile's column past the span: nothing to gather.
                gathered.append(writer.const(0.0, ftype))
                continue
            reg = ops.B(writer, None, c0 + jj, s0 // threads)
            if reg is None or reg is False:
                return None
            gathered.append(writer.rawexpr(
                text, reg, lane, type_=ftype, hint='ga', pure=True,
                movable=True, crosslane=True))
        out = gathered[-1]
        for jj in range(width - 2, -1, -1):
            out = writer.op('select', ftype,
                            writer.op('eq', BOOL, column, jj, hint='g'),
                            gathered[jj], out, hint='af')
        return out
    return fragment
