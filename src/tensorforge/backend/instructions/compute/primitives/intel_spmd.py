# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Intel DPAS from the SPMD lowering (`Options.sycl_dpas`).

`primitives.intel` emits DPAS only under the explicit vector, where a
work-item holds a whole fragment as a `simd` and `esimd::xmx::dpas` takes it.
Under DPC++'s SPMD lowering the same instruction is reachable through an
OpenCL builtin, `intel_sub_group_tf32_tf32_matrix_mad_k8`, and there the
fragments are spread over the sixteen work-items of a sub-group -- which is
the lowering's own distribution, and what makes the instruction worth having
here.  The lane layouts, as measured on the Max 1550 by placing distinct
values (SNG2, IGC 2.41.5; the builtin, the SPIR-V instruction and
`joint_matrix` agree)::

    A  8 x 8,  float4:  lane l, element i  =  A[2i + l/8][l % 8]
    B  8 x 16, float8:  lane l, element k  =  B[k][l]
    D  8 x 16, float8:  lane l, element m  =  D[m][l]

The SPMD lowering holds a register image with a row of the lead dimension
per lane and the columns in registers.  Mapped so that the lanes are the
product's rows -- `N` the sixteen rows of a lane slot, `M` eight output
columns, `K` eight contraction steps::

    D[col][row] += sum_k  T[8c + k][col]  *  Op[16 s + row][8c + k]

the accumulator *is* that image: lane `l` holds row `16 s + l`, element `m`
column `j0 + m` (`d_fragment_bits`).  So a product's result is not laid out
for whoever reads it next, and a chain of products -- each one's output the
next one's operand, as in SeisSol's flux and derivative kernels -- stays in
registers in the one layout the whole way.

What does move is the element's operand, the instruction's A.  It wants
`T[8c + l % 8][j0 + 2i + l / 8]` in lane `l`, and the image has `T[p][j]` in
lane `p % 16` of slot `p / 16`: every element is either the lane's own or its
partner's eight lanes over, so one exchange of the sub-group by `l ^ 8` per
pair of columns serves the two chunks of steps a slot holds
(`exchange_source`).  Through shared memory instead, the probe measured 1.4 ns
a product more and spills at 128 registers.

The batch-constant operator is the instruction's B, a tile of sixteen rows
by eight steps, and it is stored that way on the host: tile by tile, each
lane's eight steps adjacent, as its two TF32 halves (`fragment_order`).  A
lane then reads a part of a tile as one 32-byte vector, and nothing is split
in the kernel.  Or, for a copy the file defines (`Options.sycl_dpas_block2d`,
`surface_order`), as a 2D surface sixteen floats wide, a tile sixteen of its
rows -- its upper halves, then its lower -- which one 2D block message of the
sub-group reads, a kilobyte (`tensorforge::dpasBlock2d`).  The halves are
what the three products of the TF32 split multiply -- `hi*hi + hi*lo +
lo*hi`, `lo*lo` below F32's rounding -- and the upper one has to be *rounded*
(`tensorforge::roundTF32`): DPAS truncates its inputs, so an operand handed
in as it is loses up to a TF32 ulp.  Measured, the three products come to
1.2e-6 normwise, 1.6 times the FMA path's.

The fragments are held in OpenCL's own vectors, which is how the lexic spells
a vector of four or eight floats where the path is on
(`tensorforge::NativeFloat4/8`, `SyclLexic.get_fptype`).  A `sycl::vec` is a
class over an array, which the front end takes apart into scalars, and IGC
then puts eight of them back into one block of registers for every
instruction: at 256 registers the order-6 SeisSol kernels spilled 750 to
5 531 references that way where they hold none, but the volume 631, in
native vectors (IGC 2.41.5).  So is a 2D block of a tile's sixteen rows
(`tensorforge::NativeFloat16`, `SyclLexic.sub_group_block2d`).

Columns past the last whole eight -- SeisSol's ninth -- stay on FMAs over the
operator values the tiles hold anyway (`TAIL_MAX`): an instruction of one
column costs as much as one of eight.

Products that add into one output -- three in SeisSol's volume, four in
`localFluxAll` -- may keep their sum in the accumulators from the first to
the last (`Options.mma_d_layout`, `compute.dchain`): the first reads the
output's old value into them by the register image's own addresses
(`_Product.open_chain`), every member adds into them, and the last writes
them (`close_chain`).  What that leaves out is each member's zeroed start,
its epilogue's additions and the reads and writes of the output between
the members; a product on its own is a chain of one, without the epilogue.  Tiles of the operator that its numbers
say are zero (`Options.skip_known_zeros`) are not read and not multiplied,
as the FMA path leaves out its zero steps.

The instruction contracts eight steps whatever the operator holds, and the
element's operand is exchanged for it whatever its zeros, so a short
contraction stays on the FMA path (`Options.sycl_dpas_min_depth`), as does a
product of fewer than `MIN_COLUMNS` columns.  F64 has no DPAS on this part.
"""

from tensorforge.backend.pir import engines, terms
from tensorforge.backend.pir.core import BOOL, INDEX, ScalarType
from tensorforge.common.basic_types import Datatype
from tensorforge.common.exceptions import GenerationError, InternalError
from .. import rowpack, split
from ..bitlayout import Bit, BitLayout, Place, packed
from ..strategy import Strategy, whole

#: The execution size of `dpas.8x8` for TF32, and so the sub-group the
#: builtin runs in and the lanes of a multiplication it serves.
SUB_GROUP = 16

#: Output columns one instruction covers: the repeat count, `M`.
COLUMNS = 8

#: Contraction steps one instruction takes: the systolic depth, `K`.
DEPTH = 8

#: Elements of one operator tile, the instruction's B: sixteen rows by eight
#: steps, one part of it a 32-byte vector a lane.
TILE = SUB_GROUP * DEPTH

#: The models whose DPAS takes TF32 at an execution size of sixteen: Xe-HPC.
#: Xe-HPG has DPAS at eight, which is another instruction and another layout.
MODELS = ('pvc',)

#: Columns past the last whole tile carried on FMAs rather than padded into a
#: tile of their own (`_Product._carried`): a padded tile costs its three
#: instructions whatever it holds, and SeisSol's ninth column is an eighth of
#: one.  Two at most; three columns or more are cheaper padded.
TAIL_MAX = 2

#: Output columns below which a product stays on the FMA path: one
#: instruction over two columns is three quarters padding, and the
#: element's operand is exchanged for it all the same.
MIN_COLUMNS = 3

#: Terms each operand is split into, and the products of them that are kept,
#: smallest first (`split.products`): three, `lo*lo` dropped.
SPLIT_TERMS = 2
PRODUCTS = split.products(SPLIT_TERMS)

#: The same three as a term plan (`terms.TermPlan`), `(A's half, B's half)`
#: named, in the order they are issued: `PRODUCTS` iterated as `(B's, A's)`
#: -- smallest first, a tie broken by B's half.
PLAN = terms.plan('3xtf32/b', terms.Form(Datatype.TF32, SPLIT_TERMS),
                  terms.Form(Datatype.TF32, SPLIT_TERMS), major='b')

#: What a chain's members issue, as `multilinear._d_atom` names it.
CHAIN_MODE = 'tf32'


class ChainAtom:
    """The accumulator a chain's members share (`MatmulOperands.chain`):
    a lane slot of sixteen rows (`m`) by a tile of eight columns (`n`), F32,
    in `d_fragment_bits` -- which is the register image's own layout, so the
    old value is read into it and the sum written from it by the image's
    addresses."""

    def __init__(self):
        self.m, self.n, self.d, self.mode = (SUB_GROUP, COLUMNS, Datatype.F32,
                                             CHAIN_MODE)


def chain_atom(ctx, dtype, columns=0, lead=0, depth=0):
    """`ChainAtom` where this path takes a product of this shape, `None`
    otherwise -- the gate of `offered` and `plan`, asked of the extents."""
    if not usable(ctx) or dtype != Datatype.F32:
        return None
    if columns and columns < MIN_COLUMNS:
        return None
    if depth and depth < min_depth(ctx):
        return None
    return ChainAtom()


#: The callee of one instruction (`tensorforge_device/sycl_spmd.h`): what
#: the matrix op of engine `dpas_spmd` is written as (`mac`).
DPAS = engines.DPAS_SPMD
#: The rounding of the upper half of a split (`sycl_spmd.h`).
ROUND = 'tensorforge::roundTF32'


# --------------------------------------------------------------------------- #
# Where an element sits in a fragment
# --------------------------------------------------------------------------- #
#
# As measured, in the vocabulary of `bitlayout`: a lane bit moves an element
# between work-items, a vector bit between the elements of one work-item's
# `sycl::vec`.  The three are written down once, here; the exchange, the
# order the operator is stored in and the read-out of the accumulator are
# each checked against them (`tests/test_sycl_dpas.py`) rather than spelled
# a second time as arithmetic.

def a_fragment_bits() -> BitLayout:
    """A, `(m, k)` over 8 x 8: lane `8 (m % 2) + k`, element `m / 2`."""
    return BitLayout(((Bit(Place.LANE, 8), Bit(Place.VECTOR, 1),
                       Bit(Place.VECTOR, 2)),
                      (Bit(Place.LANE, 1), Bit(Place.LANE, 2),
                       Bit(Place.LANE, 4))))


def b_fragment_bits() -> BitLayout:
    """B, `(k, n)` over 8 x 16: lane `n`, element `k`."""
    return BitLayout(((Bit(Place.VECTOR, 1), Bit(Place.VECTOR, 2),
                       Bit(Place.VECTOR, 4)),
                      (Bit(Place.LANE, 1), Bit(Place.LANE, 2),
                       Bit(Place.LANE, 4), Bit(Place.LANE, 8))))


def d_fragment_bits() -> BitLayout:
    """C and D, `(m, n)` over 8 x 16: lane `n`, element `m`."""
    return BitLayout(((Bit(Place.VECTOR, 1), Bit(Place.VECTOR, 2),
                       Bit(Place.VECTOR, 4)),
                      (Bit(Place.LANE, 1), Bit(Place.LANE, 2),
                       Bit(Place.LANE, 4), Bit(Place.LANE, 8))))


def mac(writer, type_, a, b, c, term, plan=None):
    """One DPAS, D = A * B + C: the matrix op of engine `dpas_spmd`
    (`IRBuilder.matrix_mac`), written as `tensorforge::dpasTF32`
    (`tensorforge_device/sycl_spmd.h`) over the sub-group the target spells;
    A, B and C a vector a work-item, D a new value
    of `type_` -- a link of the accumulator chain, in a variable of its own,
    its layout one nobody wrote down -- with the layouts measured above and
    `term`, the halves of A and B the product multiplies, of the emulated
    product `plan` (`terms.TermPlan`)."""
    return writer.matrix_mac(
        'dpas_spmd', (COLUMNS, SUB_GROUP, DEPTH),
        (Datatype.TF32, Datatype.TF32, Datatype.F32, Datatype.F32),
        (a,), (b,), (c,), result=type_, hint='dacc',
        layouts=(a_fragment_bits(), b_fragment_bits(), d_fragment_bits()),
        term=term, plan=plan)


def tile_cells():
    """`b_fragment_bits` inverted, lane outermost: `[(step, lane)]` per slot
    of a stored tile, the step counted within the chunk.  What
    `fragment_order` lays a tile out by -- a lane's elements adjacent, so
    that its share of a tile part is one vector -- and checked to name every
    cell once."""
    from tensorforge.common.exceptions import GenerationError
    place = {}
    bits = b_fragment_bits()
    for k in range(DEPTH):
        for n in range(SUB_GROUP):
            at = bits.locate(k, n)
            place[at.lane, at.element] = (k, n)
    cells = [place.get((lane, element)) for lane in range(SUB_GROUP)
             for element in range(DEPTH)]
    if None in cells or len(set(cells)) != TILE:
        raise GenerationError('the B layout does not fill a tile of '
                              f'{SUB_GROUP} lanes by {DEPTH} elements')
    return cells


def accumulator_element(column: int) -> int:
    """Which element of a lane's accumulator holds column `column` of a
    tile (`d_fragment_bits`): the read-out takes it from there, at every
    lane alike -- the lane is the row."""
    return d_fragment_bits().locate(column, 0).element


def exchange_source(chunk: int, lane: int, element: int):
    """Where lane `lane`'s element `element` of the A fragment for steps
    `8 chunk ..` lies in the register image: `(slot, lane, column)`, the
    column counted from the tile's first.

    Read off `a_fragment_bits` -- that element is `A[m][k]` at `m = 2 element
    + lane / 8`, `k = lane % 8`, and `A[m][k]` is step `8 chunk + k` of column
    `m` -- and the image's own layout, step `p` in lane `p % 16` of slot `p /
    16`.  The source lane is always the lane itself or `lane ^ 8`, which is
    what lets one `permute_group_by_xor` by 8 serve every element; a test
    holds every lane and element to that.
    """
    m, k = None, None
    bits = a_fragment_bits()
    for mm in range(COLUMNS):
        for kk in range(DEPTH):
            at = bits.locate(mm, kk)
            if at.lane == lane and at.element == element:
                m, k = mm, kk
    step = DEPTH * chunk + k
    return step // SUB_GROUP, step % SUB_GROUP, m


def exchange_plan():
    """The exchange `exchange_source` implies, per half of the sub-group:
    `(take, send)`.

    `take[odd, high]` is `(partner, offset)` for a lane of the upper half
    (`high`) building the A fragment of a chunk of that parity: whether its
    element comes from the lane eight over rather than its own registers,
    and which column of the element's pair, `2 i + offset`, it is.
    `send[high]` is the column offset a lane of that half hands its partner.

    Derived, and checked to be the same for every lane of a half and every
    element: a lane's share of the exchange is one value, so the plan is a
    select on the half or it is not one exchange at all.  And the same
    `send` for both parities, which is what lets one exchange per pair of
    columns serve a slot's two chunks.
    """
    from tensorforge.common.exceptions import GenerationError
    take = {}
    for odd in (0, 1):
        for lane in range(SUB_GROUP):
            for element in range(COLUMNS // 2):
                slot, source, column = exchange_source(odd, lane, element)
                partner = source != lane
                if slot != 0 or (partner and source != lane ^ (SUB_GROUP // 2)):
                    raise GenerationError(
                        f'the A layout puts element {element} of lane {lane} '
                        f'at lane {source}, slot {slot}: not one exchange by '
                        f'{SUB_GROUP // 2}')
                entry = (partner, column - 2 * element)
                key = (odd, lane >= SUB_GROUP // 2)
                if take.setdefault(key, entry) != entry:
                    raise GenerationError(
                        f'the A layout asks different things of the lanes of '
                        f'one half of the sub-group: {take[key]} and {entry}')
    send = {}
    for (odd, high), (partner, offset) in take.items():
        if partner and send.setdefault(not high, offset) != offset:
            raise GenerationError(
                'the two chunks of a slot want different columns from one '
                'half of the sub-group, so one exchange cannot serve both')
    return take, send


# --------------------------------------------------------------------------- #
# The operator, stored as the instruction reads it
# --------------------------------------------------------------------------- #

def chunks(depth: int, head: int) -> int:
    """Blocks of `DEPTH` steps a contraction of `depth` steps takes, starting
    `head` steps into its first block of `SUB_GROUP` (`MatmulOperands.kx`)."""
    return -(-(int(head) + int(depth)) // DEPTH)


def tile_slot(slot: int, chunk: int, nchunks: int) -> int:
    """The first storage slot of tile `(slot, chunk)` in one part of the
    stored operator (`fragment_order`): tiles slot-major, `TILE` each."""
    return (slot * nchunks + chunk) * TILE


def fragment_order(shape, window, head: int):
    """Which bounding-box cell each storage slot of the operator holds, in
    F-order, `-1` for padding: `Tensor.storage_order` for one part.

    Tile `(s, c)` -- rows `16 s ..` of lane slot `s`, steps `8 c ..` of the
    contraction as the element's operand counts them -- is `TILE` slots
    from `tile_slot(s, c)`, lane by lane, each lane's eight steps adjacent:
    `b_fragment_bits` with the lane outermost, so that a lane's share of a
    tile is one vector.  The steps are counted from the start of the block
    the element's operand is read in, `head` steps before the contraction's
    first (`kx`), and those before it and past the last are padding: the
    tiles line up with the element's lanes rather than with the operator's
    columns, which costs nothing on the host and keeps every exchange the
    one by eight.

    `window` is `((r0, rows), (c0, depth))`: the rows a multiplication's
    slots hold from the box's row `r0`, and the `depth` steps it contracts
    from the box's column `c0`.  A row past the operand is padding as well.
    """
    total_rows, total_cols = (int(x) for x in shape)
    (r0, rows), (c0, depth) = ((int(a), int(b)) for a, b in window)
    nslots = -(-rows // SUB_GROUP)
    nchunks = chunks(depth, head)
    cells = tile_cells()
    order = []
    for s in range(nslots):
        for c in range(nchunks):
            assert len(order) == tile_slot(s, c, nchunks)
            for k, lane in cells:
                row = SUB_GROUP * s + lane
                step = DEPTH * c + k - int(head)
                inside = (row < rows and 0 <= step < depth
                          and r0 + row < total_rows
                          and c0 + step < total_cols)
                order.append((r0 + row) + total_rows * (c0 + step)
                             if inside else -1)
    return tuple(order)


class SplitTiles(tuple):
    """A tile order (`fragment_order`) and the scalars an element of it
    takes: two, its TF32 halves, planar -- each part's tiles one run, so a
    lane's share of a part is one vector.  No `slot_major`: the order is the
    one multiplication's window, so an operator other operations read is
    offered none (`MultilinearInstruction._offer_order`)."""

    def __new__(cls, order, parts=2):
        self = super().__new__(cls, order)
        self.parts = int(parts)
        return self


# --------------------------------------------------------------------------- #
# The operator as a 2D surface (`Options.sycl_dpas_block2d`)
# --------------------------------------------------------------------------- #

def surface_cells():
    """`b_fragment_bits` inverted, element outermost: `[(step, lane)]` per
    slot of a tile part on the 2D surface (`surface_order`) -- row `e` the
    lanes' element `e`, lane `l` in column `l` -- so that a 2D block of the
    tile's rows hands each work-item its share as the instruction's B takes
    it: a column of the block, its elements in order.  `tile_cells`
    transposed."""
    cells = tile_cells()
    return [cells[lane * DEPTH + element] for element in range(DEPTH)
            for lane in range(SUB_GROUP)]


class SurfaceTiles(SplitTiles):
    """A tile order laid out as a 2D surface (`surface_order`): `surface`
    is `(columns, group)` for `Tensor.storage_surface` -- `SUB_GROUP`
    columns, the parts planar a tile at a time, so that tile `t`'s halves
    are the surface's rows `SPLIT_TERMS DEPTH t ..`, its upper one first."""

    def __new__(cls, order, parts=2, surface=(SUB_GROUP, TILE)):
        self = super().__new__(cls, order, parts)
        self.surface = tuple(int(x) for x in surface)
        return self


def block2d(ctx) -> bool:
    """`Options.sycl_dpas_block2d` where the path is taken (`usable`)."""
    return (ctx is not None
            and bool(ctx.get_user_options().sycl_dpas_block2d)
            and usable(ctx))


def surface_order(order, ctx):
    """`order`'s tiles (`prepared_order`) laid out as a 2D surface for one
    2D block read a tile (`Options.sycl_dpas_block2d`,
    `MatrixPaths.surface_order`), or `None` where they stay as they are.

    The surface is `SUB_GROUP` floats wide -- 64 bytes, the narrowest a 2D
    block reads (`block_reads.SURFACE_MIN_WIDTH`) -- and a tile is sixteen
    of its rows, one kilobyte: the upper halves of its eight steps, a row a
    step and a column a lane (`surface_cells`), then the lower ones.  So
    every slot keeps its cell, a tile its place, and only the slots within a
    tile and the parts move (`Tensor.storage_surface`).  Asked of an
    operator copy the file defines alone (`multilinear._offer_order`): its
    table is aligned to 256 bytes (`Generator.COPY_ALIGN`), and a 2D
    block's surface has to start on a cache line, which an operator the
    caller passes is not known to."""
    if not block2d(ctx) or not isinstance(order, SplitTiles):
        return None
    if order.parts != SPLIT_TERMS or len(order) % TILE:
        return None
    where = {cell: n for n, cell in enumerate(tile_cells())}
    moved = [where[cell] for cell in surface_cells()]
    out = []
    for base in range(0, len(order), TILE):
        out.extend(order[base + n] for n in moved)
    return SurfaceTiles(out, parts=SPLIT_TERMS, surface=(SUB_GROUP, TILE))


# --------------------------------------------------------------------------- #
# Whether the path is taken
# --------------------------------------------------------------------------- #

def asked(value) -> bool:
    """Whether a value of `Options.sycl_dpas` turns the path on: any true
    value -- a caller may hand in `1` as well as `True` -- but `auto`, which
    is the tuner's to answer (`tuning._dpas_candidates`): a build it did not
    answer for takes the FMA path."""
    return value != 'auto' and bool(value)


def enabled(ctx) -> bool:
    """`Options.sycl_dpas` (`asked`), off where there is no context to ask."""
    return ctx is not None and asked(ctx.get_user_options().sycl_dpas)


def usable(ctx) -> bool:
    """Whether this build may emit the builtin at all: asked for, DPC++'s
    SPMD lowering, and a part whose DPAS takes TF32 sixteen wide -- with an
    IGC that resolves the builtin (`resolved`)."""
    if not enabled(ctx):
        return False
    target = ctx.target
    model = str(getattr(target.hw, 'model', ''))
    if (target.explicit_simd or target.backend != 'oneapi'
            or model not in MODELS):
        return False
    resolved(model)
    return True


def resolved(model: str) -> None:
    """Raise where the `ocloc` on PATH -- which the ahead-of-time build runs
    -- has an IGC that does not resolve the builtin
    (`toolchain.intel_dpas_tf32`).

    IGC 1.x builds such a kernel without a word and leaves the call an
    undefined symbol of the binary: the build succeeds and the first launch
    stops at the runtime's link of the module, far from anything that names
    the cause.  Here it is named, at generation, which in a build that
    generates its kernels is build time.  Where there is no `ocloc` to ask --
    a JIT build, whose IGC is the driver's, or a machine that only
    generates -- nothing is checked.
    """
    from tensorforge.toolchain import DPAS_TF32_BUILTIN, intel_dpas_tf32
    if intel_dpas_tf32(model) is False:
        raise GenerationError(
            f'Options.sycl_dpas needs IGC 2.x: the IGC behind the `ocloc` on '
            f'PATH does not resolve `{DPAS_TF32_BUILTIN}` for {model} (IGC '
            f'1.x leaves it an undefined symbol, and the kernel stops at its '
            f'first launch).  Put an ocloc with IGC 2.x first on PATH (IGC '
            f'2.41.5, compute-runtime 26.35, resolves it), or build without '
            f'sycl_dpas.')


def min_depth(ctx) -> int:
    """`Options.sycl_dpas_min_depth`."""
    return int(ctx.get_user_options().sycl_dpas_min_depth)


def offered(shape, ctx) -> bool:
    """Whether the path can take a product of `shape` (`strategies`).

    * Sixteen lanes: the instruction's execution size, and so a
      multiplication that is its sub-group.  Eight would share one with a
      neighbor whose rows the instruction would mix in; thirty-two is two.
    * F32 throughout.  The accumulator chooses the instruction; operands of
      another type are the nest's (`_plan`).
    * A batch-constant operator, dense to its reader: its tiles are stored
      for the instruction, and a per-element operand would have to be laid
      out per element -- the transform being avoided.  Not a sparse second
      operand either, which is read by linear index and has no lanes.
    * One row a lane (`lead_layout`): a packed lead puts two or four rows in
      a lane, and the instruction's B wants one.
    * A contraction long enough to fill the instruction
      (`Options.sycl_dpas_min_depth`); an unknown one is admitted.
    """
    if not usable(ctx) or shape.explicit_simd:
        return False
    if shape.threads != SUB_GROUP or shape.accumulator != Datatype.F32:
        return False
    if shape.sparse or not shape.a_uniform or shape.a_parts not in (1, 2):
        return False
    if packed(shape.lead_layout):
        return False
    return shape.depth == 0 or shape.depth >= min_depth(ctx)


def plan(strategy, shape, n, ctx):
    """One arrangement over all `n` columns -- the tail is carried, not
    split off -- or the nest where there are too few columns to fill an
    instruction (`MIN_COLUMNS`).  Asked before `settle_storage` lays the
    operator out, so a product this declines is never given tiles."""
    if strategy is Strategy.MATRIX and n < MIN_COLUMNS:
        return whole(Strategy.GENERIC, n)
    return whole(strategy, n)


def row_caps(columns: int) -> rowpack.RowCaps:
    """What this path does with a product's columns past its last whole
    tile of `COLUMNS` (`MatrixPaths.row_caps`, `Options.row_packing`):
    carries them on FMAs over the operator values its tiles hold
    (`_Product._carried`), or pads them into a tile of their own.  By
    itself it carries up to `TAIL_MAX` and pads more; a product of no
    more than one tile has nothing to carry."""
    ntail = columns % COLUMNS
    if columns <= COLUMNS:
        return rowpack.RowCaps(COLUMNS, frozenset({rowpack.Packing.PAD}),
                               rowpack.Packing.PAD)
    auto = (rowpack.Packing.FMA if 0 < ntail <= TAIL_MAX
            else rowpack.Packing.PAD)
    return rowpack.RowCaps(
        COLUMNS, frozenset({rowpack.Packing.FMA, rowpack.Packing.PAD}),
        auto)


def prepared_order(shape, dtype, ctx, columns=0, lead=0, depth=0,
                   threads=SUB_GROUP, corner=(0, 0), rows=None, head=0):
    """The operator's tiles (`fragment_order`), split into its TF32 halves
    (`SplitTiles`), or `None` where this path will not read it.

    Over the window the multiplication reads: `rows` rows from the box row
    its first slot starts at -- the slots, rounded up to sixteen -- and
    `depth` steps from `corner`'s column, laid out from `head` steps before
    it.  The same gate as `offered` and `plan`, so a layout is offered only
    to a product that takes the path.
    """
    if not usable(ctx) or len(shape) != 2:
        return None
    if threads != SUB_GROUP or dtype != Datatype.F32:
        return None
    if not lead or not depth or columns < MIN_COLUMNS:
        return None
    if depth < min_depth(ctx):
        return None
    r0, c0 = (int(x) for x in corner)
    if (r0 < 0 or c0 < 0 or r0 >= int(shape[0])
            or c0 + int(depth) > int(shape[1])):
        return None
    read = int(rows) if rows else -(-int(lead) // SUB_GROUP) * SUB_GROUP
    order = fragment_order(shape, ((r0, read), (c0, depth)), head)
    return SplitTiles(order, parts=SPLIT_TERMS)


# --------------------------------------------------------------------------- #
# Emission
# --------------------------------------------------------------------------- #

class _Declined(Exception):
    """An accessor had no value: the product goes to the nest."""


def matmul(writer, ops, ctx, span):
    """Emit the product, or decline and leave the body to the nest.

    Declining an operator stored as its halves is an error rather than a
    fallback: the nest reads one scalar an element, which of such a buffer
    is the upper half, and the result off by about a thousandth.
    """
    taken = False
    if (span.strategy is Strategy.MATRIX and span.start == 0
            and span.stop == ops.n and _servable(ops)):
        try:
            taken = _matmul(writer, ops, ctx)
        except _Declined:
            taken = False
    if not taken and ops.a_parts != 1:
        raise InternalError(
            'an operator stored as its TF32 halves for DPAS, and DPAS '
            'declined the operation; nothing else reads the halves')
    return taken


def _servable(ops) -> bool:
    """What `offered` asks of the shape, asked of what the emission is
    handed: a direct caller, or a shape that was not the whole story."""
    return (ops.sparse is None and ops.lead_width == 1
            and ops.threads == SUB_GROUP
            and ops.accumulator == Datatype.F32
            and ops.a == Datatype.F32 and ops.b == Datatype.F32
            and ops.a_parts in (1, SPLIT_TERMS)
            and (ops.a_parts == 1 or ops.A_slot is not None
                 or ops.A_block2d is not None))


def _matmul(writer, ops, ctx) -> bool:
    """The product, one column tile of eight at a time (`_Product`); fenced
    at both ends where its tiles are (`Options.sycl_dpas_fence`), so that
    IGC keeps the products before and after it, and their loads, out of it."""
    product = _Product(writer, ops, ctx)
    if product.fence is not None:
        product.fence_here()
    chain = ops.chain
    if chain is not None and chain.opens:
        product.open_chain()
    for j0 in product.tiles:
        product.tile(j0)
    if chain is not None and chain.closes:
        product.close_chain()
    if product.fence is not None:
        product.fence_here()
    return True


class _Product:
    """One product's emission: what its tiles share.

    A column tile accumulates over the chunks of eight steps, each chunk
    one A fragment -- the element's operand, exchanged and split -- against
    the operator's tile of every lane slot, three instructions a slot.  The
    columns carried on FMAs go with the last column tile, over the operator
    values its chunks read.
    """

    def __init__(self, writer, ops, ctx):
        self.writer, self.ops = writer, ops
        self.N, self.K, self.kx = ops.n, ops.k, ops.kx
        # Lead slots from the first slot's start, where the accessors count
        # (`MatmulOperands.lead_first`).
        self.nslots = -(-(ops.lead_first + ops.lead_elements) // SUB_GROUP)
        self.walked = self.K + self.kx
        self.nchunks = chunks(self.K, self.kx)
        self.parts = ops.a_parts
        self.fp = ScalarType(Datatype.F32)
        self.v4 = ScalarType(Datatype.F32, 4)
        self.v8 = ScalarType(Datatype.F32, 8)
        self.exchange = ctx.target.lexic.exchange_xor('{0}', SUB_GROUP // 2)
        if self.exchange is None:
            raise _Declined()
        ntail = self.N % COLUMNS
        # The columns past the last whole tile: carried on FMAs or padded
        # into a tile of their own, as `row_packing` asks (`row_caps`; by
        # itself up to `TAIL_MAX` carried).
        packed = rowpack.plan_rows(self.N, row_caps(self.N), ctx)
        tailed = (packed.packing is rowpack.Packing.FMA and ntail > 0
                  and self.N > COLUMNS)
        self.dense = self.N - ntail if tailed else self.N
        self.tail = list(range(self.dense, self.N)) if tailed else []
        self.tiles = list(range(0, self.dense, COLUMNS))
        self.live = {(s, c): not self._zero_tile(s, c)
                     for s in range(self.nslots)
                     for c in range(self.nchunks)}
        self.chunks = [c for c in range(self.nchunks)
                       if any(self.live[s, c] for s in range(self.nslots))]
        self.take, self.send = exchange_plan()
        self._anchor_value = None
        # Where the sub-group has a fence of the schedule the compiler holds
        # to (`Lexic.pin_fence`): what `fence_here` writes.
        fence = (ctx.target.lexic.pin_fence()
                 if ctx.get_user_options().sycl_dpas_fence else None)
        self.fence = fence[0] if fence is not None else None
        self._b = {}
        self._swapped = {}
        self._lanes = None

    def fence_here(self) -> None:
        """A fence of the schedule (`engines.FenceKind`): IGC keeps what
        stands on either side of it there, and the sub-group writes it as its
        barrier, the one `self.fence` names."""
        from tensorforge.backend.pir.engines import FenceKind
        self.writer.fence(FenceKind.SCHEDULE)

    # -- what the operator's numbers say ------------------------------------ #

    def _zero(self, e_lo, e_hi, k_lo, k_hi) -> bool:
        return (self.ops.A_zero is not None
                and self.ops.A_zero(e_lo, e_hi, k_lo, k_hi))

    def _steps(self, chunk):
        """The contraction steps chunk `chunk` holds, as `A` counts them:
        `[lo, hi)` of `[0, K)`, empty where it is all padding."""
        lo = max(0, DEPTH * chunk - self.kx)
        hi = min(self.K, DEPTH * chunk + DEPTH - self.kx)
        return lo, hi

    def _zero_tile(self, s, c) -> bool:
        lo, hi = self._steps(c)
        if lo >= hi:
            return True
        return self._zero(SUB_GROUP * s, SUB_GROUP * (s + 1), lo, hi)

    # -- the element's operand ---------------------------------------------- #

    def _lane_values(self):
        """The lane, whether it is in the lower half of the sub-group, and
        which of a chunk's eight steps it hands in (`a_fragment_bits`)."""
        if self._lanes is None:
            w = self.writer
            lane = w.lane_index(SUB_GROUP, hint='dl')
            low = w.op('lt', BOOL, lane, SUB_GROUP // 2, hint='dlo')
            step = w.op('rem', INDEX, lane, DEPTH, hint='dk')
            self._lanes = (lane, low, step)
        return self._lanes

    def _image(self, column, slot):
        """`B` at `column` and lane slot `slot`, `None` where it is zero."""
        key = (column, slot)
        if key not in self._b:
            got = self.ops.B(self.writer, None, column, slot)
            if got is False:
                raise _Declined()
            self._b[key] = got
        return self._b[key]

    def _swap(self, j0, slot, pair):
        """The values of columns `j0 + 2 pair` and `+ 1` at lane slot
        `slot`, and what the lane eight over hands this one: `(values, z)`,
        a zero as `None`, `z` `None` where both columns are.

        Each lane sends the column its partner takes (`exchange_plan`), so
        one exchange serves both chunks of the slot."""
        key = (j0, slot, pair)
        if key in self._swapped:
            return self._swapped[key]
        w = self.writer
        end = min(j0 + COLUMNS, self.dense)
        values = [self._image(j0 + 2 * pair + offset, slot)
                  if j0 + 2 * pair + offset < end else None
                  for offset in (0, 1)]
        z = None
        if any(v is not None for v in values):
            _, low, _ = self._lane_values()
            zero = w.const(0.0, self.fp)
            low_sends = values[self.send[False]]
            high_sends = values[self.send[True]]
            send = w.op('select', self.fp, low,
                        low_sends if low_sends is not None else zero,
                        high_sends if high_sends is not None else zero,
                        hint='ds')
            z = w.rawexpr(self.exchange, send, type_=self.fp, hint='dx',
                          pure=True, movable=False, crosslane=True)
        self._swapped[key] = (values, z)
        return self._swapped[key]

    def operand(self, j0, chunk):
        """The A fragment of column tile `j0` for chunk `chunk`: the
        element's values exchanged into place, the lanes whose step lies
        outside the contraction zeroed, split into the halves `(hi, lo)`."""
        w = self.writer
        _, low, step = self._lane_values()
        slot, odd = divmod(chunk, 2)
        zero = w.const(0.0, self.fp)
        # Lane `l` hands in step `8 chunk + l % 8`, which has to lie in
        # `[kx, K + kx)`: before it the block's lanes are not the
        # contraction's, past it they are the next block's or nothing.
        first, last = self.kx - DEPTH * chunk, self.walked - DEPTH * chunk
        inside = None
        if first > 0:
            inside = w.op('ge', BOOL, step, first, hint='dv')
        if last < DEPTH:
            upper = w.op('lt', BOOL, step, last, hint='dv')
            inside = upper if inside is None else w.op('and', BOOL, inside,
                                                       upper, hint='dv')
        his, los = [], []
        for pair in range(COLUMNS // 2):
            values, z = self._swap(j0, slot, pair)
            if z is None:
                his.append(zero)
                los.append(zero)
                continue

            def taken(high):
                partner, offset = self.take[odd, high]
                got = z if partner else values[offset]
                return got if got is not None else zero
            v = w.op('select', self.fp, low, taken(False), taken(True),
                     hint='da')
            if inside is not None:
                v = w.op('select', self.fp, inside, v, zero, hint='da')
            hi = w.call(ROUND, self.fp, v, hint='dh')
            his.append(hi)
            los.append(w.op('sub', self.fp, v, hi, hint='dlw'))
        return (w.pack(self.v4, *his, hint='dah'),
                w.pack(self.v4, *los, hint='dal'))

    # -- the operator ------------------------------------------------------- #

    def _anchor(self):
        """Zero, as an index the compiler cannot prove invariant in the batch
        loop: the element's index, a 64-bit `SIZE`, shifted down by 63 bits.

        The tiles are the operator's, the same for every element, so their
        addresses are invariant in the loop, and IGC's first compilation of
        a kernel computes all of them ahead of it and holds them through it:
        a 64-bit address a lane, two registers at SIMD16 for each of the
        o6-s volume's 84 tile reads, 168 of the 333 live at its peak (IGC
        2.41.5, `*_codegen.ll`).  Its retry, where the first attempt spills
        past a threshold, computes each address at its read and spills
        nothing -- which the volume reached without its zero tiles skipped
        (717 spill references in the first attempt) and not with them
        (567, kept).  Read at `lane + (anchor + tile) * 16`, every tile's
        address is one value of the loop plus a constant, and nothing is
        left to hoist; the shift costs one scalar instruction an element.

        No kernel is retried any more, and the volume holds 157 spill
        references.  Measured (Max 1550, batch 262144, ns an element without
        and with the anchor): order 6 derivative 21.2 / 19.9,
        `localFluxAll` 17.9 / 17.6, `neighboringFlux[0]` 8.0 / 7.4, volume
        22.2 / 12.8 (13.9 / 14.6 with every tile read); order 8 125.5 /
        125.0, 81.9 / 72.0, 15.0 / 15.4, 33.8 / 32.9."""
        if self._anchor_value is None:
            from tensorforge.backend.instructions.batch_loop import BatchLoop
            from tensorforge.common.basic_types import GeneralLexicon
            w = self.writer
            name = f'{GeneralLexicon.BATCH_ID_NAME}0'
            bound = BatchLoop.indices_in(w)
            batch = (bound.get(name) if bound is not None and name in bound
                     else w.batch_id('0'))
            self._anchor_value = w.op('shr', INDEX, batch, 63, hint='dz')
        return self._anchor_value

    def operator(self, s, c):
        """Tile `(s, c)` of the operator as the instruction's B, `(hi, lo,
        parts)`: `parts(k)` is step `k`'s value for the carried columns, the
        parts it is stored in, `None` where it is a zero.

        Read as the order stored it where there is one: one 2D block of the
        tile's sixteen rows where it is a surface (`A_block2d`,
        `surface_order`), a vector of eight a lane and part where it is the
        tiles (`A_slot`).  Otherwise by coordinate, a step at a time, and
        split here.  The parts are taken out of the vectors where they
        are multiplied, not here: a value of the IR is a register to the
        pressure count, and one taken out at the read keeps the tile live in
        the count up to its last use, wherever that is."""
        w = self.writer
        ops = self.ops
        if ops.A_block2d is not None:
            # The tile's rows of the surface (`surface_order`), one 2D block
            # read of the sub-group: work-item `l` gets column `l`, the upper
            # halves of its eight steps and then the lower.  At a row the
            # anchor moves (`_anchor`): a read at a constant row is the same
            # message in every trip of the batch loop, and IGC builds the
            # messages' payloads ahead of it and holds them through it.
            # SeisSol's volume without and with the anchor (IGC 2.42.6, 256
            # registers, spilled bytes): order 6 256 / 0, order 8 17 728 /
            # 7 616 -- the tiles' two vector reads 0 and 8 256.
            rows = SPLIT_TERMS * DEPTH
            got = ops.A_block2d(
                w, rows * (tile_slot(s, c, self.nchunks) // TILE), rows,
                shift=self._anchor())
            if got is None or got is False:
                raise _Declined()
            hi = w.slice_(got, 0, self.v8, layout=got.layout, hint='dbh')
            lo = w.slice_(got, DEPTH, self.v8, layout=got.layout,
                          hint='dbl')
            element = {k: e for e, (k, _) in enumerate(tile_cells()[:DEPTH])}
            return hi, lo, (lambda k: (
                w.extract(hi, element[k], self.fp, hint='dt'),
                w.extract(lo, element[k], self.fp, hint='dt')))
        if ops.A_slot is not None:
            got = ops.A_slot(w, tile_slot(s, c, self.nchunks),
                             parts=self.parts, width=DEPTH,
                             shift=self._anchor())
            if got is None or got is False:
                raise _Declined()
            # A lane's elements of a tile are its steps (`tile_cells`).
            element = {k: e for e, (k, _) in enumerate(tile_cells()[:DEPTH])}
            if self.parts == SPLIT_TERMS:
                hi, lo = got
                return hi, lo, (lambda k: (
                    w.extract(hi, element[k], self.fp, hint='dt'),
                    w.extract(lo, element[k], self.fp, hint='dt')))
            values = [w.extract(got, element[k], self.fp, hint='dt')
                      for k in range(DEPTH)]
        else:
            values = []
            for k in range(DEPTH):
                step = DEPTH * c + k - self.kx
                if not 0 <= step < self.K:
                    values.append(None)
                    continue
                got = ops.A(w, None, s, step)
                if got is False:
                    raise _Declined()
                values.append(got)
        zero = w.const(0.0, self.fp)
        his, los = [], []
        for v in values:
            if v is None:
                his.append(zero)
                los.append(zero)
                continue
            hi = w.call(ROUND, self.fp, v, hint='dh')
            his.append(hi)
            los.append(w.op('sub', self.fp, v, hi, hint='dlw'))
        return (w.pack(self.v8, *his, hint='dbh'),
                w.pack(self.v8, *los, hint='dbl'),
                lambda k: None if values[k] is None else (values[k],))

    # -- one column tile ---------------------------------------------------- #

    def _carried(self, s, c, k, t, parts, shared, products):
        """The products of step `k` of chunk `c` into carried column `t` of
        slot `s`, lower half first: added to a partial sum of its own size
        rather than rounded off against the upper one."""
        step = DEPTH * c + k - self.kx
        if not 0 <= step < self.K or self._zero(
                SUB_GROUP * s, SUB_GROUP * (s + 1), step, step + 1):
            return
        terms = parts(k)
        if terms is None:
            return
        if (k, t) not in shared:
            position = DEPTH * c + k
            value = self._image(t, position // SUB_GROUP)
            shared[k, t] = (None if value is None else self.writer.lane_broadcast(
                value, position % SUB_GROUP, SUB_GROUP, hint='dbc'))
        if shared[k, t] is not None:
            products.extend((part, shared[k, t]) for part in reversed(terms))

    def _chain(self, old, products):
        """`old` plus every product, ending in a value of its own
        (`escapes`): left to the inliner, the sum over every chunk would be
        one expression at the product's end, and the parts it reads would
        hold their tiles until then."""
        w = self.writer
        for n, (part, shared) in enumerate(products):
            last = n == len(products) - 1
            old = (w.op('mul', self.fp, part, shared, hint='dts', escapes=last)
                   if old is None else
                   w.op('fma', self.fp, part, shared, old, hint='dts',
                        escapes=last))
        return old

    # -- a chain of products into one output ---------------------------------- #

    def _chain_rows(self, slot):
        """Row `16 slot + lane` of the chain's frame, and the predicate that
        keeps it to the chain's window (`AccumulatorChain.rows`), `None`
        where the whole slot lies inside."""
        w = self.writer
        lo, hi = self.ops.chain.rows
        lane = self._lane_values()[0]
        row = (w.op('add', INDEX, lane, SUB_GROUP * slot, hint='drow')
               if slot else lane)
        cond = None
        if SUB_GROUP * slot < lo:
            cond = w.op('ge', BOOL, row, lo, hint='dg')
        if SUB_GROUP * (slot + 1) > hi:
            upper = w.op('lt', BOOL, row, hi, hint='dg')
            cond = upper if cond is None else w.op('and', BOOL, cond, upper,
                                                   hint='dg')
        return row, cond

    def open_chain(self):
        """The chain's accumulators, one a lane slot and column tile of its
        window, with the output's old value read into them: lane `l` of
        slot `s` reads row `16 s + l`, its element `accumulator_element(m)`
        column `j0 + m` -- the register image's own addresses, so the read
        is the bias load the chain replaces, issued once for every member
        (`compute.dchain`).  A carried column's sum starts the same way."""
        w = self.writer
        chain = self.ops.chain
        nslots = -(-chain.rows[1] // SUB_GROUP)
        zero = w.const(0.0, self.fp)
        accs, tails = {}, {}
        for s in range(nslots):
            row, cond = self._chain_rows(s)
            for j0 in self.tiles:
                elements = [zero] * COLUMNS
                for m in range(min(j0 + COLUMNS, self.dense) - j0):
                    elements[accumulator_element(m)] = chain.load(
                        w, row, j0 + m, cond)
                accs[j0, s] = w.pack(self.v8, *elements, hint='dacc')
            for t in self.tail:
                tails[s, t] = chain.load(w, row, t, cond)
        chain.state.clear()
        chain.state.update(accs=accs, tails=tails, slots=nslots)

    def close_chain(self):
        """The chain's sums written where `open_chain` read the old values,
        each lane its rows of the window."""
        w = self.writer
        chain = self.ops.chain
        state = chain.state
        for s in range(state['slots']):
            row, cond = self._chain_rows(s)
            stores = []
            for j0 in self.tiles:
                for m in range(min(j0 + COLUMNS, self.dense) - j0):
                    stores.append((w.extract(state['accs'][j0, s],
                                             accumulator_element(m), self.fp,
                                             hint='dc'), j0 + m))
            for t in self.tail:
                stores.append((state['tails'][s, t], t))
            if cond is None:
                for value, col in stores:
                    chain.store(w, value, row, col)
                continue
            with w.if_(cond):
                for value, col in stores:
                    chain.store(w, value, row, col)

    def tile(self, j0):
        """Column tile `j0` over every chunk: per chunk the A fragment, then
        slot by slot the tile read, its three products and its share of the
        carried columns.

        IGC schedules the block as it likes whatever the order written, and
        left alone it reads the tiles of later chunks ahead, which costs
        registers.  With `Options.sycl_dpas_fence` the tiles are read one
        ahead of their products and a fence of the sub-group stands between
        one tile's products and the next (`Lexic.pin_fence`, which no load is
        moved across): the next tile's read overlaps this one's products,
        and no more than two tiles are in flight -- at the price of the
        latency the reads ahead would have hidden, which is why it is a
        question per kernel."""
        w = self.writer
        ops = self.ops
        chain = ops.chain
        carried = self.tail if j0 == self.tiles[-1] else []
        if chain is not None:
            # A member adds into the chain's accumulators and hands them on
            # to the next; the last writes them (`close_chain`).
            state = chain.state
            if self.nslots > state.get('slots', 0):
                raise InternalError(
                    f'a member of a chain over {state.get("slots", 0)} lane '
                    f'slots computes {self.nslots}')
            acc = {s: state['accs'][j0, s] for s in range(self.nslots)}
            sums = {(s, t): state['tails'][s, t] for s in range(self.nslots)
                    for t in carried}
        else:
            acc = {s: w.declare(self.v8, hint='dacc')
                   for s in range(self.nslots)}
            sums = {}
        order = [(c, s) for c in self.chunks for s in range(self.nslots)
                 if self.live[s, c]]
        read = {}
        if self.fence is not None and order:
            read[0] = self.operator(order[0][1], order[0][0])
        chunk, aterms, shared = None, None, {}
        for i, (c, s) in enumerate(order):
            if self.fence is not None:
                self.fence_here()
                if i + 1 < len(order):
                    read[i + 1] = self.operator(order[i + 1][1],
                                                order[i + 1][0])
            if c != chunk:
                chunk, aterms, shared = c, self.operand(j0, c), {}
            hi, lo, parts = (read.pop(i) if i in read
                             else self.operator(s, c))
            bterms = (hi, lo)
            for q, p in PLAN.products:
                acc[s] = mac(w, self.v8, aterms[q], bterms[p], acc[s],
                             (q, p), plan=PLAN)
            for t in carried:
                products = []
                for k in range(DEPTH):
                    self._carried(s, c, k, t, parts, shared, products)
                if products:
                    sums[s, t] = self._chain(sums.get((s, t)), products)
        if chain is not None:
            for s in range(self.nslots):
                chain.state['accs'][j0, s] = acc[s]
                for t in carried:
                    chain.state['tails'][s, t] = sums[s, t]
            return
        end = min(j0 + COLUMNS, self.dense)
        zero = w.const(0.0, self.fp)
        for s in range(self.nslots):
            for m in range(end - j0):
                ops.C(w, w.extract(acc[s], accumulator_element(m), self.fp,
                                   hint='dc'), s, j0 + m)
            for t in carried:
                ops.C(w, sums.get((s, t), zero), s, t)
