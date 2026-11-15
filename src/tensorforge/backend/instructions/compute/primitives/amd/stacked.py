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
check a build of this scheme can be held to.

Sixteen lanes only.  At thirty-two the halves of a multiplication hold
different tiles of rows, and one issue feeds every lane quarter the same
sixteen operator rows: half of each issue would compute rows nobody keeps.
"""

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

#: The two orders of a wave's elements and columns along an instruction's M
#: (`stacking`): the element minor, ``m = e + mults jj``, or major, ``m = jj +
#: columns e``.
ELEMENT_MINOR = 'element minor'
ELEMENT_MAJOR = 'element major'

#: Lanes a multiplication may span.  See the module: at thirty-two one issue
#: serves only half of its lanes.
LANES = (16,)

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
def stacking(op, threads: int) -> Optional[str]:
    """How `op` stacks a wave's multiplications at `threads` lanes -- the M
    order of their elements and columns, `ELEMENT_MINOR` or `ELEMENT_MAJOR`
    -- or `None` where its fragments are not the stacked assignment.

    Three statements, each read off the fragment tables (`layouts`), for
    `mults` multiplications a wave of `columns` output columns a tile, `t` a
    multiplication's lane and `p` its place in the wave -- wave lane ``t +
    threads * p``:

    * A: lane ``t + threads p`` holds ``A[t][p]``, so the element and column
      a lane's A carries are those its `t` names as an `m`, and its step is
      its `p`;
    * B: lane ``t + threads p`` holds ``B[p][t]``, the operator's row `t`
      at step `p` -- `A_wave`'s distribution;
    * D: ``D[m(e, jj)][n]`` sits in slot `jj` of lane ``n + threads e`` --
      the register image's register for column `jj` of the `e`-th
      multiplication's lane `n` -- for one of the two orders `m`.

    `mfma_f64_16x16x4f64` has all three at sixteen lanes with the element
    minor, `mfma_f32_16x16x4f32` with it major.  Asked of the table rather
    than assumed, because the D layouts of one shape disagree between
    instructions (`layouts`), and a wrong order here is a correctly typed
    operand holding another lane's numbers.
    """
    wave = op.wave
    if threads <= 0 or wave % threads or op.blocks != 1:
        return None
    mults = wave // threads
    if op.m != threads or op.n != threads or op.k != mults or op.m % mults:
        return None
    columns = op.m // mults
    for p in range(mults):
        for t in range(threads):
            lane = t + threads * p
            if element_at(op, 'A', 0, lane) != (0, t, p):
                return None
            if element_at(op, 'B', 0, lane) != (0, p, t):
                return None
    for order in (ELEMENT_MINOR, ELEMENT_MAJOR):
        if all(position(op, 'D', stacked_row(order, e, jj, mults, columns), n)
               == (jj, n + threads * e)
               for e in range(mults) for jj in range(columns)
               for n in range(threads)):
            return order
    return None


def fits(op, threads: int) -> bool:
    """Whether `op`'s fragments are the stacked assignment at `threads`
    lanes, in either M order (`stacking`)."""
    return stacking(op, threads) is not None


def stacked_row(order, e, jj, mults, columns) -> int:
    """The M row of the `e`-th multiplication's column `jj` (`stacking`)."""
    return e + mults * jj if order == ELEMENT_MINOR else jj + columns * e


def lane_cell(writer, op, threads, t):
    """`(element, column)`: which multiplication's element and which column
    of the tile the lane's `t` names as an M row of `op`'s A -- the inverse
    of `stacked_row` -- as index expressions."""
    from tensorforge.backend.pir.core import INDEX
    mults = op.wave // threads
    columns = op.m // mults
    if stacking(op, threads) == ELEMENT_MINOR:
        return (writer.op('rem', INDEX, t, mults, hint='e'),
                writer.op('div', INDEX, t, mults, hint='c'))
    return (writer.op('div', INDEX, t, columns, hint='e'),
            writer.op('rem', INDEX, t, columns, hint='c'))


def instruction(threads: int, accumulator, ctx):
    """The instruction the scheme issues here, or `None`: asked for, defined
    for the accumulator, available on the target, and stacking at this lane
    count (`stacking`)."""
    if requested(ctx) is None or threads not in LANES:
        return None
    name = INSTRUCTIONS.get(accumulator)
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
    (`a_uniform`): one issue multiplies all four elements by one B.  Not
    sparse, which no fragment reads; one scalar a lane at lead width one, which
    is the image D lands in; SPMD, where a wave is lanes; and, for the `shard`
    route, a per-element operand in shared memory (`b_shared`).
    """
    if (shape.sparse or shape.explicit_simd or not shape.a_uniform
            or shape.a_parts != 1 or bitlayout.packed(shape.lead_layout)):
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
    chain."""
    options = getattr(ctx, 'get_user_options', None)
    if options is None:
        return False
    return bool(getattr(options(), 'amd_matrix_k4_pad', False))


def stacked_plan(shape, n: int, ctx):
    """Whole tiles of columns on the instruction, the rest on the DPP chain.

    The rest is SeisSol's ninth column: a tile padded to compute it would
    spend three quarters of its issues on zeros, where the chain spends a
    product a row and step -- and reads the operator once more, a row and a
    step at a time.  Which costs more is a measurement, so the padded tile is
    an option (`padded`); a product narrower than a tile stays on the chain
    either way.
    """
    op = instruction(shape.threads, shape.accumulator, ctx)
    width = tile_columns(op, shape.threads)
    edge = (n // width) * width
    if edge <= 0:
        return whole(Strategy.DPP, n)
    if edge >= n or padded(ctx):
        return whole(Strategy.MATRIX, n)
    return (Span(Strategy.MATRIX, 0, edge), Span(Strategy.DPP, edge, n))


def serves(ops, ctx) -> bool:
    """Whether a matrix span over `ops` is this scheme's: `stacks`, asked of
    the operands the emission holds rather than of the shape the plan was
    made from.  The two agree; a span `stacked_plan` gave the instruction
    reaches here only through it."""
    return (ops.sparse is None and ops.a_uniform and ops.lead_width == 1
            and ops.a_parts == 1
            and instruction(ops.threads, ops.accumulator, ctx) is not None)


def matmul_stacked(writer, ops, ctx, span) -> bool:
    """`C[:, span] += A B` through the stacked instruction.

    False, with nothing emitted, where the operands are not what the scheme
    reads: no `A_wave` (an operator not in memory in its own order), a wave
    whose multiplications do not run in step, operands of another type, a
    span that is not whole tiles -- but for the last one where it is padded
    (`padded`) -- or a per-element operand whose fragment a route cannot
    reach.  The caller then takes the span through the DPP chain, which reads
    all of these -- rather than handing the whole product to the nest, which
    a decline would.
    """
    threads = ops.threads
    op = instruction(threads, ops.accumulator, ctx)
    if op is None:
        return False
    mults = op.wave // threads
    width = tile_columns(op, threads)
    if (ops.A_wave is None or not ops.lockstep
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


def _emit(writer, ops, ctx, span, op, mults, width) -> bool:
    from tensorforge.backend.pir.core import BOOL, INDEX, ScalarType
    C, A = ops.C, ops.A
    threads, M, K, kx = ops.threads, ops.lead_slots, ops.k, ops.kx
    ftype = ScalarType(ops.accumulator)
    acctype = ScalarType(ops.accumulator, op.d.per_lane)
    zero = writer.const(0.0, ftype)
    # The lane's place in the wave's fragments (`stacking`): its `t` names the
    # element and the column its A carries, its `p` the step of both.
    t = writer.thread_id('x')
    p = writer.op('rem', INDEX, writer.thread_id('y'), mults, hint='m')
    tiles = list(range(span.start, span.stop, width))

    def steps(s0):
        """The operator's own steps a tile at position `s0` covers: the
        positions count from the block holding the contraction's first step,
        `kx` before it (`MatmulOperands.kx`)."""
        return max(0, s0 - kx), min(K, s0 + op.k - kx)

    def known_zero(rt, s0):
        lo, hi = steps(s0)
        return (lo >= hi or (ops.A_zero is not None and ops.A_zero(
            threads * rt, threads * (rt + 1), lo, hi)))

    def by_step(values):
        """The value of the lane's own step out of one per step, `p` picking:
        a tile at the edge of the contraction, where a step past it reads a
        zero rather than a neighbor's number."""
        out = values[-1]
        for q in range(len(values) - 2, -1, -1):
            out = writer.op('select', ftype, writer.op('eq', BOOL, p, q,
                                                       hint='g'),
                            values[q], out, hint='ks')
        return out

    def operator_fragment(rt, s0):
        """B: the lane's operator row at step `s0 - kx + p`."""
        r0 = s0 - kx
        if r0 >= 0 and r0 + mults <= K:
            got = ops.A_wave(writer, rt, r0, mults)
            if got is not None and got is not False:
                return got
        values = []
        for q in range(mults):
            got = (A(writer, None, rt, r0 + q) if 0 <= r0 + q < K else None)
            values.append(got if got is not None and got is not False
                          else zero)
        return by_step(values)

    source = a_source(writer, ops, ctx, op, mults, width, t, p)
    if source is None:
        return False

    def element_fragment(c0, s0):
        """A: the column `c0 + jj` of element `e` the lane's `t` names
        (`lane_cell`), at position `s0 + p`, zero for a position outside the
        contraction -- and for a column past the span, in a padded last
        tile."""
        got = source(c0, s0, min(width, span.stop - c0))
        if got is None:
            return None
        edge = [q for q in range(mults) if not kx <= s0 + q < K + kx]
        if not edge:
            return got
        return by_step([zero if q in edge else got for q in range(mults)])

    with writer.AnonymousScope():
        acc = {(rt, c0): writer.declare(acctype, hint='acc')
               for rt in range(M) for c0 in tiles}
        # The contraction outermost and every accumulator inside: the
        # per-element fragments are built once for all tiles of rows, and an
        # accumulator is read again only behind seven others.
        for s0 in range(0, K + kx, op.k):
            live = [rt for rt in range(M) if not known_zero(rt, s0)]
            if not live:
                continue
            frags = {}
            for c0 in tiles:
                frags[c0] = element_fragment(c0, s0)
                if frags[c0] is None:
                    return False
            for rt in live:
                b = operator_fragment(rt, s0)
                for c0 in tiles:
                    acc[(rt, c0)] = writer.call(
                        op.callee, acctype, frags[c0], b, acc[(rt, c0)],
                        0, 0, 0, hint='acc', movable=False, materialize=True)
        for rt in range(M):
            for c0 in tiles:
                for jj in range(min(width, span.stop - c0)):
                    C(writer, writer.extract(acc[(rt, c0)], jj, ftype), rt,
                      c0 + jj)
    return True


def a_source(writer, ops, ctx, op, mults, width, t, p):
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
        return _sharded(writer, ops, op, mults, width, t, p)
    if route == 'shard':
        return None
    return _gathered(writer, ops, op, mults, width, t, p)


def _sharded(writer, ops, op, mults, width, t, p):
    """The `shard` route: A out of the shared memory the operand sits in.

    An earlier operation of the section stored the per-element operand there,
    every multiplication its own copy in its own arena, `mult_stride`
    elements after its predecessor's, and the operation reads it in place
    (`MatrixPaths.reads_shared`).  The lane's A is position ``s0 + p`` of the
    element `e` and column ``c0 + jj`` its `t` names (`lane_cell`): one read
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
    element, column = lane_cell(writer, op, threads, t)
    mult = writer.op('sub', INDEX, element, p, hint='mo')
    fenced = []

    def fragment(c0, s0, columns):
        if not fenced:
            writer.barrier('wave', threads=threads, handoff=True)
            fenced.append(True)
        r0 = s0 - kx
        base = min(max(r0, 0), K - 1)
        if 0 <= r0 and r0 + mults <= K:
            rows = p
        else:
            step = writer.op('add', INDEX, p, r0, hint='ks')
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


def _gathered(writer, ops, op, mults, width, t, p):
    """The `transpose` route: A out of the registers the operand is held in.

    The nest's image of the per-element operand holds position `s` of
    column `j` in lane `s % threads` of block `s // threads`
    (`MatmulOperands.B`).  The lane's A is position ``s0 + p`` of the
    element `e` and column ``c0 + jj`` its `t` names (`lane_cell`): wave lane
    ``threads e + (s0 + p) % threads``, register ``B(c0 + jj, ...)``.  A lane
    gather (`ds_bpermute`) reads the same register out of every lane it
    names, and the column differs between the lanes of one fragment, so each
    column of the tile is gathered on its own and the lane takes its own:
    `width` gathers and a select a fragment.
    """
    from tensorforge.backend.pir.core import BOOL, INDEX, ScalarType
    threads = ops.threads
    ftype = ScalarType(ops.accumulator)
    element, column = lane_cell(writer, op, threads, t)
    base = writer.op('add', INDEX,
                     writer.op('mul', INDEX, element, threads, hint='e'), p,
                     hint='src')
    text = f'tensorforge::gather<{op.wave}>({{0}}, {{1}})'

    def fragment(c0, s0, columns):
        lane = writer.op('add', INDEX, base, s0 % threads, hint='src')
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
