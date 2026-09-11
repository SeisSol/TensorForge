# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
from tensorforge.common.basic_types import Datatype
from tensorforge.common.exceptions import GenerationError, InternalError
from .. import ranking
from ..bitlayout import Bit, BitLayout, Place
from ..routes import lead_route as routes_lead_route
from ..strategy import Strategy, whole
from tensorforge.backend.pir.core import (BOOL, INDEX, Access, Effect, MemSpace, Uniformity,
                                          XorSwizzle,
                                          ScalarType, Value)
from tensorforge.backend.writer import Writer

#: The two halves an FP32 value splits into for `mma.sync ... .tf32`.
#:
#: `Datatype.TF32` now, not `U32`.  The old spelling said "four bytes of
#: something" and was chosen because `splitFloatTF32` took `uint32_t &` -- it
#: still does, since `tensorforge::tf32` is a typedef on CUDA and has to be
#: (the halves go into PTX under `"r"`, which binds a register and not a class
#: type).  What changes is what the *generator* knows: a value of this type is
#: a converted operand of a matrix instruction, not an integer that happens to
#: be four bytes wide.
#:
#: The same member serves the Intel path, where the C++ type is
#: `esimd::tfloat32` and the distinction is enforced by the compiler.
TF32_HALF = ScalarType(Datatype.TF32)


def _as_tf32(writer: Writer, value):
    """A stored half as the instruction's operand type, without arithmetic.

    A TF32 value *is* a float whose low thirteen mantissa bits are zero, so a
    half that was prepared on the host is already the right number; what the
    `asm` needs is the `r` operand class, which is a reinterpretation and not
    a conversion.  `__float_as_uint` is how CUDA spells that, and it costs no
    instruction.

    This is the whole difference between reading a prepared operand and
    computing one: `tfconvert` below emits two `cvt.rna.tf32.f32` per value,
    and on a part without a native converter each of those is a software
    sequence.
    """
    return writer.rawexpr('__float_as_uint({0})', value, type_=TF32_HALF,
                          hint='u', pure=True, movable=True)


def tfconvert(writer: Writer, variables):
    """Split each operand into the two TF32 halves the MMA multiplies.

    The halves used to be a raw declaration and a raw call --- two statements
    per operand, 4584 across the corpus, and the largest opaque site here after
    `matmul` itself.  Nothing followed from that opacity being cheap to remove:
    the declaration has a value, the call writes through references to it, and
    the IR has had verbs for both since the AMD conversion.

    What does *not* become structured yet is the input.  `generate` is handed
    A and B as C++ identifiers built from `varalloc` names, not as values, so
    the operand goes in as text and there is no def-use edge into the split.
    Closing that is the `matmul` patch; until then this is a boundary, and
    writing it as one is better than writing it as an intrinsic that happens
    to take a string.
    """
    # A pure operation with two results, not a call writing through
    # references.  The reference-out spelling is the vendor's signature and
    # belongs in the emitter; here the split is what it is, and CSE can
    # hash-cons it.  The corpus split the same value twice in 15% of cases --
    # no store and no reload in between, just a second `kk` block asking for
    # the same fragment.
    out = []
    for variable in variables:
        out.append(writer.split_op('tensorforge::splitFloatTF32',
                                   (TF32_HALF, TF32_HALF), variable,
                                   hints=('u', 'l')))
    return out

class MMAMode:
    DIRECT = 0
    TF32 = 1
    BF16 = 2
    I8 = 3

class MMAInstr:
    def __init__(self, m, n, k, b, d, name, mode, sm):
        self.n = n
        self.m = m
        self.k = k
        self.b = b
        self.d = d
        self.name = name
        self.mode = mode
        #: Compute capability the instruction first appears at, times ten.
        #: Promoted from the comment each row already carried, and unchecked
        #: in the way a comment was: nothing here reads the PTX ISA.  It is a
        #: field so that a selection can refuse an entry the target does not
        #: have, which a comment cannot do.
        self.sm = sm

    def headers(self):
        return []

    def asmcall(self, writer, D, A, B, C, uses=()):
        """The `mma.sync` itself.

        `D` and `C` are the same accumulator at every call site -- the
        instruction reads it and writes it back.  Listing it as `"=f"` under
        outputs and again as `"f"` under inputs states two *unrelated* operands
        that happen to name one C++ lvalue, and nothing then requires the
        compiler to give them the same register: it may read the accumulator
        into one and write the result into another, dropping the accumulation.
        `"+f"` says read-and-write, and then the operand is listed once.

        Numbering follows from that.  The assembler numbers outputs and inputs
        in one sequence, so folding C into D shifts A and B down by `len(C)`;
        `asm_stmt` checks the template against the operand list rather than
        trusting that the two were edited together.
        """
        typeid = "f" if self.d == Datatype.F32 else "d"
        typeidx = "r" if self.d == Datatype.F32 else "d"

        inout = D if D is C or list(D) == list(C) else None

        grp = lambda n, b: "{" + ','.join(f"%{i + b}" for i in range(n)) + "}"

        if inout is not None:
            operands = ([(f'+{typeid}', v) for v in inout]
                        + [(typeidx, v) for v in A]
                        + [(typeidx, v) for v in B])
            dgrp = grp(len(inout), 0)
            agrp = grp(len(A), len(inout))
            bgrp = grp(len(B), len(inout) + len(A))
            cgrp = dgrp
        else:
            operands = ([(f'={typeid}', v) for v in D]
                        + [(typeidx, v) for v in A]
                        + [(typeidx, v) for v in B]
                        + [(typeid, v) for v in C])
            dgrp = grp(len(D), 0)
            agrp = grp(len(A), len(D))
            bgrp = grp(len(B), len(D) + len(A))
            cgrp = grp(len(C), len(D) + len(A) + len(B))

        template = (f'"{self.name} "\n'
                    f'"{dgrp}, {agrp}, {bgrp}, {cgrp};"')
        writer.asm_stmt(template, operands)

    def epilogue(self):
        pass

    @property
    def terms(self):
        """How many products one accumulator takes per step (`products`)."""
        return 3 if self.mode == MMAMode.TF32 else 1

    def products(self, writer, A, B, a_split=None):
        """The products one accumulator takes, as `(A, B)` operand lists in
        the order `generate` issues them.

        Separate from issuing them so that a caller with several accumulators
        can interleave: 3xTF32 puts three products into one accumulator, and
        issued back to back each waits for the one before it.

        `a_split` is the A halves where they were read rather than computed.
        Same shape `tfconvert` returns -- a `(upper, lower)` per fragment -- so
        the products are unchanged and only their source differs.  Given, the
        conversion of A does not happen at all; that is the point of storing
        the operand prepared, and it is the whole of what this parameter does.
        """
        if self.mode == MMAMode.TF32:
            Atf32 = a_split if a_split is not None else tfconvert(writer, A)
            Btf32 = tfconvert(writer, B)
            half = lambda xs, h: [x[h] for x in xs]
            return [(half(Atf32, 0), half(Btf32, 0)),
                    (half(Atf32, 0), half(Btf32, 1)),
                    (half(Atf32, 1), half(Btf32, 0))]
        return [(A, B)]

    def generate(self, writer, context, A, B, C, uses=(), a_split=None):
        """Every product of one accumulator, issued in order (`products`)."""
        with writer.Scope():
            for a, b in self.products(writer, A, B, a_split):
                self.asmcall(writer, C, a, b, C, uses)

INSTRS = [
    MMAInstr(16,8,4,1,Datatype.F32,'mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32', MMAMode.TF32, 80), # SM_80
    MMAInstr(16,8,8,1,Datatype.F32,'mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32', MMAMode.TF32, 80), # SM_80
    MMAInstr(8,8,4,1,Datatype.F64,'mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64', MMAMode.DIRECT, 80), # SM_80
    MMAInstr(16,8,4,1,Datatype.F64,'mma.sync.aligned.m16n8k4.row.col.f64.f64.f64.f64', MMAMode.DIRECT, 90), # SM_90
    MMAInstr(16,8,8,1,Datatype.F64,'mma.sync.aligned.m16n8k8.row.col.f64.f64.f64.f64', MMAMode.DIRECT, 90), # SM_90
    MMAInstr(16,8,16,1,Datatype.F64,'mma.sync.aligned.m16n8k16.row.col.f64.f64.f64.f64', MMAMode.DIRECT, 90), # SM_90
    MMAInstr(8,8,16,1,Datatype.F64,'mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32', MMAMode.I8, 75), # SM_75
    MMAInstr(16,8,16,1,Datatype.F64,'mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32', MMAMode.I8, 80), # SM_80
    MMAInstr(16,8,32,1,Datatype.F64,'mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32', MMAMode.I8, 80), # SM_80
]

#: The compute capability to select against when the caller has no target.
#:
#: It used to be 80 because nothing plumbed the target's: `shmsize` was asked
#: for a size before a context existed and `matmul` was handed one it did not
#: pass on, so every arch from sm_60 to sm_120 got the same sm_80 table.  Both
#: now take an `sm`, derived from the context by `sm_of`, and this is what is
#: left: the floor for a caller that genuinely has no target to name.
#:
#: 75 rather than 80 because that is where `mma.sync` first exists at all, and
#: because being the *floor* it should exclude rather than include.  Every F32
#: entry here is sm_80, so a caller with no context now selects nothing for
#: F32 and falls through to the generic nest -- which is the safe direction.
#: At 80 it would instead have emitted sm_80 PTX for a target that may not run
#: it.
BASELINE_SM = 75


def sm_of(ctx) -> int:
    """The target's compute capability times ten, or the floor.

    `hw_descr().model` is the arch string the context was built with
    (`'sm_120'`).  Anything that does not parse as one -- a non-NVIDIA model
    reaching here, or a name shape this does not know -- yields the floor
    rather than a guess, so an unrecognised target declines instructions
    instead of being credited with them.
    """
    try:
        model = str(ctx.get_vm().get_hw_descr().model)
    except AttributeError:
        return BASELINE_SM
    if not model.startswith('sm_'):
        return BASELINE_SM
    digits = ''.join(c for c in model[3:] if c.isdigit())
    return int(digits) if digits else BASELINE_SM

#: Modes the emitter actually emits.  `generate` converts and issues three
#: products for TF32 and issues once for DIRECT; its I8 branch is a `pass`, so
#: those entries would be selected and then emit nothing.
EMITTED_MODES = (MMAMode.TF32, MMAMode.DIRECT)


def accumulator_slots(atom):
    """`(slot, offset)` for every accumulator register one lane holds.

    The D fragment's shape, and the only place it is written down.  Every row
    in `INSTRS` has `n == 8` and holds D as `m / 8` row groups of two adjacent
    columns: lane `t` owns rows `t / 4 + 8 * g` and columns `2 * (t % 4) + e`.
    Since `(t / 4) * n + 2 * (t % 4)` is `2 * t` for `n == 8`, the lane's whole
    contribution to the address is `2 * t`, which the caller adds; what is left
    is the constant offset per slot, and `slot` is `2 * g + e` because that is
    the order the PTX operand list is in.

    Extracted from the epilogue rather than left inline because inline it was
    three constants -- `nregs * 2`, `mregs` and a literal 64 -- that happen to
    describe `m16n8k8` and nothing else in the table.  `m8n8k4.f64` has
    `mregs == 1`, so the loop dropped `e == 1` and wrote the other slot 64
    elements past the end of a 64-element tile.  That returned NaN, silently,
    and nothing could see it: the emitter was the only statement of the layout,
    so there was nothing to check it against.  `tests/test_nvidia_gate.py` now
    checks it against the read-back that follows it.
    """
    return tuple((2 * g + e, e + g * 8 * atom.n)
                 for g in range(atom.m // 8)
                 for e in range(2))


def tile_starts(extent, threads, atom):
    """Where the emitter's nest puts tile origins along one axis.

    Two nested loops and not one, and the difference is not cosmetic: the
    outer walks waves and the inner walks the atom inside a wave, but the
    inner is bounded by what is left of the *wave*, not of the extent.  So
    `range(0, extent, atom)` is a different list whenever the wave remainder
    is not a multiple of the atom, and anything deriving the tiling from a
    ceiling disagrees with the code it is describing.

    Stated here because two readers need the same answer -- the emitter, which
    walks the tiles, and `fragment_order`, which says what is in them -- and a
    layout the two derive separately is a layout they can come to disagree
    about without either being wrong on its own.
    """
    return tuple(outer + inner
                 for outer in range(0, extent, threads)
                 for inner in range(0, min(threads, extent - outer), atom))


def a_fragment_bits(atom, threads: int = 32) -> BitLayout:
    """The PTX A layout of one tile, in the vocabulary a value is read in.

    The counterpart to AMD's `FRAGMENT_BITS`, and it exists for the same
    reason: a distribution stated as arithmetic can be executed and compared
    to nothing, while one stated as bits can be held against what an operand
    actually holds.  Until this was written the packed-operand refusal here
    had to be a literal -- there was no second side to ask about.

    Both axes factor exactly, which is not an accident of these entries but
    of the map: lane `t` holds row ``t / ktile + iii * mtile`` and column
    ``t % ktile + kf * ktile``, and every one of those terms is a shift.  The
    row's low three bits are the lane's bits 2..4, its higher bits are `iii`,
    which is the operand list's low digit; the column's low two bits are the
    lane's 0..1 and its higher bits are `kf`, the high digit, whose place
    value in `f` is `mregs`.  A tile therefore spreads `atom.m * atom.k`
    elements over exactly `aregs * threads` slots, one each.

    The hardware fact itself is `fragment_order`'s to document, and it is
    verified there three ways on real silicon.  This states it once, and
    `fragment_order` reads it back rather than spelling the same arithmetic a
    second time -- two statements of one map can disagree, and the one that
    decides where the host packer writes and the one that decides where the
    kernel reads are exactly the pair that must not.
    """
    mtile, ktile = 8, 4
    mregs, kregs = atom.m // mtile, atom.k // ktile
    rows = tuple([Bit(Place.LANE, 1 << (2 + b)) for b in range(3)]
                 + [Bit(Place.SLOT, 1 << b)
                    for b in range((mregs - 1).bit_length())])
    cols = tuple([Bit(Place.LANE, 1 << b) for b in range(2)]
                 + [Bit(Place.SLOT, mregs << b)
                    for b in range((kregs - 1).bit_length())])
    return BitLayout((rows, cols))


def _fragment_cells(atom, threads: int = 32, bits=None):
    """`a_fragment_bits` inverted: which cell of a tile each slot and lane
    holds.

    A bijection, and checked to be one.  A tile has `atom.m * atom.k` cells
    and `aregs * threads` places for them, and those are equal by
    construction -- so a table that sends two cells to one place has lost a
    third, and an operand packed from it would be missing an element that the
    kernel then reads as whatever was there before.

    `bits` is a parameter so the refusal can be exercised: with the real
    table it never fires, and a guard that cannot be made to fire is a guard
    nobody has read.
    """
    bits = a_fragment_bits(atom, threads) if bits is None else bits
    cells = {}
    for row in range(atom.m):
        for col in range(atom.k):
            at = bits.locate(row, col)
            if (at.slot, at.lane) in cells:
                raise GenerationError(
                    f'the A layout of {atom.name} sends two cells to slot '
                    f'{at.slot} lane {at.lane}, so a tile does not fit its '
                    f'own fragments')
            cells[(at.slot, at.lane)] = (row, col)
    return cells


def fragment_order(shape, atom, threads=32):
    """Which bounding-box cell each slot of a pre-ordered A holds.

    The operand's storage in the order a lane reads it: tile by tile, and
    within a tile lane by lane, each lane's `aregs` fragments adjacent in the
    order the instruction's operand list takes them.  So a lane reads a tile's
    fragments with one access into the register group the instruction wants,
    a warp reads the tile as one contiguous run, and there is no shared round
    trip at all.  Staged, the same fragments come out of `Ashm[lane + threads
    * f]`; stored, lane `lane`'s fragment `f` is slot `aregs * lane + f` of
    its tile.  With the parts planar (`Tensor.storage_planar`) each part is
    such an image of its own.

    Lane by lane rather than fragment by fragment, which is what this was: with
    the 32 lanes contiguous per fragment, a lane read its fragments one scalar
    each, and with the parts adjacent those came in as hi/lo pairs that ptxas
    had to regroup into the instruction's register quads -- about 1230 moves
    per element of `local_flux`, measured on sm_100.

    The tile's own map is the PTX A layout, and it is the one thing here that
    is the hardware's rather than this module's: lane `t` holds rows
    `t / ktile + iii * mtile` and columns `t % ktile + kf * ktile`, with
    `f == iii + kf * mregs` the operand-list order.  Verified against the
    instruction on hardware three ways -- the `wmma` loader, a raw `mma.sync`
    against a host reference, and an impulse response -- and against this
    module's own store/load pair, which computes the same function.

    A slot naming `-1` is padding: the tiling covers `len(tile_starts) * atom`
    along each axis, which is at least the extent and usually more, and a slot
    past the end belongs to no cell.  It reads zero, which is what the
    emitter's own padding registers said before the order moved into memory.

    Returned as `Tensor.storage_order` wants it -- F-order cell per slot -- so
    the host packer and the kernel take the layout from one statement.
    """
    rows, cols = int(shape[0]), int(shape[1])
    mtile, ktile = 8, 4
    mregs, kregs = atom.m // mtile, atom.k // ktile
    aregs = mregs * kregs
    cell = _fragment_cells(atom, threads)
    mstarts = tile_starts(rows, threads, atom.m)
    kstarts = tile_starts(cols, threads, atom.k)
    # The emitter names a tile by `start // atom` (`mt`, `kt` at the fragment
    # read), so the two agree only while that is the position in the list.
    # It is, for every entry in `INSTRS` -- each `m` and `k` divides the wave,
    # so a start is a multiple of the atom and the starts run from zero
    # without gaps -- and an entry that broke it would silently shift every
    # tile, so it is checked rather than assumed.
    for starts, step in ((mstarts, atom.m), (kstarts, atom.k)):
        if tuple(st // step for st in starts) != tuple(range(len(starts))):
            raise GenerationError(
                f'the nest tiles {starts} do not enumerate by start // '
                f'{step}, so `fragment_order` and the emitter would name '
                f'tiles differently')
    ktiles = len(kstarts)
    order = []
    for mt, m0 in enumerate(mstarts):
        for kt, k0 in enumerate(kstarts):
            assert len(order) == (mt * ktiles + kt) * aregs * threads
            for lane in range(threads):
                for f in range(aregs):
                    dm, dk = cell[(f, lane)]
                    row, col = m0 + dm, k0 + dk
                    order.append(row + rows * col
                                 if row < rows and col < cols else -1)
    return tuple(order)


def prepared_order(shape, dtype, ctx, columns=0, lead=0, depth=0,
                   threads=32):
    """The order this target would read a two-dimensional A operand in.

    `None` where there is none to state: no entry serves the accumulator, the
    operand is not a matrix, or the wave is not the one the layout is written
    for.  A caller asks before it decides to prepare an operand, so a refusal
    has to be an answer rather than an exception.
    """
    if threads > WAVE or WAVE % threads or len(shape) != 2:
        return None
    # The same `sm` the emission will select with, for the reason `scratch`
    # takes a context: an order laid out against one table and read by an
    # entry from another is a permutation nothing shares.
    atom = instr_for(dtype, columns=columns, lead=lead, depth=depth,
                     sm=sm_of(ctx))
    if atom is None:
        return None
    # Over the wave and not over one multiplication: the fragment is the
    # warp's, and a narrower multiplication reads it with its neighbours'
    # lanes (`matmul`, `shift`).
    return fragment_order(shape, atom, WAVE)


def _bfrag(writer, ops, threadrange, nbase, kbase, ktile, ntile, N,
           atom, threads):
    """One `B` fragment read where it lies, with the column range guarded.

    The guard is the whole difference between this and the staged read, and it
    is easy to leave out: staged, the columns past the end of the operand are
    a *compile-time* range --- `for jj in range(min(atom.n, N - j))` reads and
    the rest declare a zero --- because the column a lane holds is a Python
    loop variable.  Read directly, the column is `n + t / ktile`, so which
    lanes have a column at all is a question about the hardware, and asking it
    at emission time gives every lane the answer for lane zero.

    What that costs if it is skipped is not a wrong column but an out-of-range
    read: the lanes above the end address past the operand, and on the corpus
    case that returned NaN rather than anything harmless.  The columns
    themselves are independent --- column `c` of the product comes from column
    `c` of `B` --- so the values the guarded-off lanes would hold are never
    read; it is the access that has to not happen.
    """
    live = min(ntile, N - nbase) * ktile
    if live >= threads:
        return ops.B_frag(writer, nbase, kbase, ktile, ntile)
    # Declared outside the guard: a value defined inside one is not visible to
    # the instruction that follows, and a slot nothing writes reads zero.
    value = writer.declare(ScalarType(atom.d), hint='bs')
    with threadrange(0, live):
        writer.assign(value, ops.B_frag(writer, nbase, kbase, ktile, ntile))
    return value


def instrs_for(dtype, sm=None):
    """Every entry the emitter could issue for this accumulator, widest first.

    Widest because a tie in the ranking keeps this order, and a tie is what a
    caller that does not state its shape gets.  Listing them in the order the
    table happens to hold would make such a caller take the narrowest.
    """
    sm = BASELINE_SM if sm is None else sm
    return tuple(sorted((op for op in INSTRS
                         if op.d is dtype and op.mode in EMITTED_MODES
                         and op.sm <= sm),
                        key=lambda op: (-op.m * op.n * op.k, op.name)))


def instr_for(dtype, columns=0, lead=0, depth=0, sm=None):
    """The entry that serves this shape with the fewest issues, or `None`.

    One function, asked twice: once by `shmsize` sizing the staging and once
    by `matmul` issuing.  A size computed for one entry and an issue of
    another is a buffer nobody fills or an overrun, and two dicts indexing the
    same list by hand is how the two come to differ.
    """
    def extent(op):
        # The warp holds `m` of the leading dimension and the accumulator `n`
        # of the output; the contraction is `k`.  A different mapping to the
        # same three numbers than AMD's, which is why the conversion is here.
        return (ranking.Extent(columns=op.n, lanes=op.m, depth=op.k,
                               name=op.name), 1)

    found = ranking.rank(instrs_for(dtype, sm), extent, columns, lead, depth)
    return found[0] if found else None


#: Whether the path is deployed, as opposed to whether it *can* emit for a
#: given shape -- that second question is `supports()`.  Two different facts,
#: so two names: `supports()` is a property of the shape, `ENABLED` is a
#: decision about the generator, and only the second is something to flip.
#:
#: Parked pending a run on real hardware.  `"+f"` versus `"=f"`/`"f"` on the
#: accumulator is a register-allocation difference no front end can see, and
#: the corpus is checked for well-formedness, not executed.
ENABLED = False


#: Contraction steps below which the matrix path is declined.
#:
#: A stand-in for a cost model, and it says so.  The widest F32 entry here
#: contracts 8 steps at a time, so a contraction of 9 issues two tiles to do
#: the work of one and a bit: 44% of what it issues is padding.  Sixteen is
#: twice that tile depth, which bounds the waste at one padded tail tile and
#: at most half the issued work -- a rule of thumb, not a measurement, and the
#: measurement is what should replace it.
#:
#: It exists because SeisSol's local flux contains both kinds in one kernel:
#: `56x56 . 56x9` contracts 56 steps and is what the path was built for, while
#: `56x9 . 9x9` contracts 9 and reaches it only incidentally.  Until the second
#: has been priced, it stays on the generic nest.
MIN_DEPTH = 16


#: Columns past the last whole tile that are carried on the fragments of that
#: tile rather than padded into a tile of their own (`matmul`).  A padded tile
#: costs its three HMMAs whatever it holds, and one column of it -- the ninth
#: of `local_flux` -- holds an eighth of that; carried, it costs a lane four
#: FFMA per step and a shared-memory reduction at the end.  Two at most, which
#: is also the room `shmsize` keeps for it.
TAIL_MAX = 2

#: The modes whose A fragments are the operand's values, row `g + 8 * i` and
#: column `t + 4 * j` of the tile -- TF32 before its split, and FP64, whose
#: `m16n8k*` fragments have the same pattern -- so that a remainder can be
#: multiplied on them directly.
TAIL_MODES = (MMAMode.TF32, MMAMode.DIRECT)

#: Independent accumulator chains a step needs before one accumulator per
#: tile is enough.  An HMMA waits for the one that produced its accumulator
#: -- about 20 cycles on sm_100, against one issued every 8 -- so a chain
#: issues at full rate only with two others between its members.  `matmul`
#: issues term by term across a step's chains, which puts `chains` HMMAs
#: between them; below this count each of a tile's products gets an
#: accumulator of its own, summed before the epilogue, and the distance is
#: `chains * terms`.  Measured on GB200 at one warp per scheduler: two
#: chains back to back 10.1 cycles per HMMA, eight apart 8.17.  Two, not
#: three, because the kernels this reaches run at two warps per scheduler or
#: more, where two chains cost 8.26 even back to back -- and a separate
#: accumulator costs registers the occupancy is counted in (`mmaosm`: 168 of
#: the 170 three blocks allow).
CHAINS_MIN = 2

#: Whether `matmul` loads a pre-ordered operand's fragments one step ahead,
#: so that the instructions stop waiting on them (GB200, package 3: `mmaos`
#: long scoreboard, 54 % of its HMMA stalls, the loads a median 13
#: instructions ahead).  Not what the registers follow: on sm_100a `mmao` has
#: 190 with it and 190 without -- and 141 without the remainder on the
#: fragments (`TAIL_MAX`), which is where they went.  A switch until the two
#: are measured against each other on the part it was meant for.
PREFETCH = True

#: The wave the fragment layouts are written over.  A multiplication spread
#: over fewer lanes shares it with its neighbours, and `matmul` runs one
#: round of the product per multiplication.
WAVE = 32


def convergence(strategy, shape):
    """How far the threads have to run in step for `strategy` over `shape`.

    `mma.sync` is `.aligned`: every lane of the warp issues it together, so
    a multiplication narrower than the warp needs its neighbours in the warp
    to take the same trips through the batch loop.  Asked of the target and
    not of the plan, because "matrix" is not one instruction: another
    target's matrix path may have no such demand at all.
    """
    if strategy is Strategy.MATRIX and shape.threads < WAVE:
        return Uniformity.MULTGROUP
    return None


def supports(threads, dtype, sparse, depth=0) -> bool:
    """Whether `matmul` can emit for this shape, asked *before* it is called.

    This was an `assert` inside the emitter, which was safe only for as long
    as nothing reached it.  Turning the path on makes the difference matter:
    an assertion aborts generation for a case the generic path handles
    perfectly well, so the preconditions have to be a question the caller can
    ask, not a crash the caller cannot avoid.

    * ``threads`` divides the wave.  The instruction is the warp's; a
      multiplication narrower than it shares the warp with its neighbours,
      and `matmul` runs one round of fragments per multiplication, wiring
      each one's `B` in and its `D` out through its own shared region.
      Wider ones are a different instruction, which does not exist here yet.
    * ``not sparse``.  `matmul` already declines these by returning `False`,
      but `temp_shmem` reserves shared memory off the same predicate; if the
      two disagree the reservation is made for a kernel that never uses it.
    * ``depth >= MIN_DEPTH``.  A shallow contraction reaches this path and then
      spends most of its issues on padding; see `MIN_DEPTH`.  `depth == 0` is
      "the caller does not know", which is not the same as "shallow" and is
      admitted -- `shmsize` asks without a shape and must keep its upper bound.
    """
    return (threads <= WAVE and WAVE % threads == 0
            and dtype in (Datatype.F32, Datatype.F64)
            and not sparse and (depth == 0 or depth >= MIN_DEPTH))


def shmsize(stages, dtype, sm=None, a_parts=1, lanes=None):
    """Staging elements to reserve, sized before the entry is chosen.

    Over every candidate rather than over the one `instr_for` would return,
    because this is asked without the shape that ranking reads: a reservation
    made for a narrower entry than the one issued is an overrun, and the two
    cannot be made to agree by ranking twice on different information.  The
    largest is an upper bound, and the difference between the candidates is
    one staging tile.

    `sm` has to be the same one `matmul` will select with.  It is the reason
    `scratch` takes a context: sizing over the sm_75 table and then issuing
    from the sm_120 one reserves a buffer for the narrowest entry and writes
    the widest into it, which is the overrun this docstring already warned
    about -- reached through the arch rather than through the shape.
    """
    threads = 32

    def size(atom):
        # `a_parts` tiles for A and not one: an operand stored prepared is
        # staged a part at a time, each through its own tile, because the
        # fragment read indexes by slot times the wave and a tile holding two
        # parts per slot would change that arithmetic for both.
        aregs = a_parts * ((atom.m * atom.k) // threads)
        bregs = (atom.n * atom.k) // threads
        cregs = (atom.m * atom.n) // threads
        # Room for a remainder carried on the fragments: its columns of `B`
        # beside the tile, and its partials -- one per row and lane position
        # -- beside the epilogue's (`TAIL_MAX`).
        tail = atom.mode in TAIL_MODES
        return max(32 * (aregs + bregs) + tail * TAIL_MAX * atom.k,
                   32 * cregs + tail * TAIL_MAX * atom.m * 4)

    # A warp shared by several multiplications staggers each one's copy of a
    # tile by up to 31 elements (`matmul`, `stagger`), the `A` and `B` tiles
    # side by side and the epilogue tile over them.  Rounded to the 16 bytes
    # every region is aligned to: the reservation is part of what one
    # multiplication owns, and a stride that is not a whole number of vectors
    # misaligns every other multiplication's wide accesses.
    mults = WAVE // lanes if lanes and lanes < WAVE else 1
    pad = 2 * 31 * (mults - 1) if mults <= 8 else 0
    align = max(1, 16 // dtype.size())
    pad = -(-pad // align) * align
    return max((size(atom) for atom in instrs_for(dtype, sm)), default=0) + pad

def lead_route(shape):
    """`routes.lead_route` with this target's rungs, and it has none.

    Not a placeholder.  A rung is an instruction that moves bits between the
    lane index and the register index, and the ones this module issues do not
    do that: `mma.sync` reads a fragment it was given, and the staging around
    it moves elements through shared memory rather than between lanes.  So
    the honest answer is the two rungs that are true everywhere -- no gap, or
    the trip -- and handing in nothing is how that is said.
    """
    return routes_lead_route(shape)


def takes(route) -> bool:
    """Whether this emitter writes this route.

    Only the empty one.  The fragments here are staged, but by `matmul`
    itself and from the operand's own storage; nothing in this module takes a
    `Transfer` plan and emits it, which is the same gap AMD has and a
    different emitter away from closing.
    """
    return route == 0


def strategies(shape, ctx):
    """What this target can emit for this shape.

    One arrangement, and it is off: `ENABLED` is the deployment switch and
    `supports` the shape gate.  Asking both here rather than at the call site
    is what keeps a shape this cannot serve falling through to the nest
    instead of reaching an assertion inside the emitter.

    A packed lead operand is declined by the route it would need, the same
    way AMD declines it and through the same function.  What differs is the
    rungs, and this target hands in none: there is no instruction here that
    permutes a register index against a lane index, so an operand that does
    not already arrive at the fragment's distribution reaches it through
    memory or not at all.  `takes` then says no, because the emitter writes
    the staged fragments it stages itself and not a relayout it was handed.

    That is a refusal with a price attached rather than a literal, which is
    what it was until `reach` stopped living in `primitives/amd`.  It also
    means this lifts the way AMD's will: by an emitter learning a route, not
    by a condition being edited.
    """
    if not takes(lead_route(shape)):
        return frozenset()
    if (ENABLED
            and supports(shape.threads, shape.accumulator, shape.sparse,
                         shape.depth)
            and instrs_for(shape.accumulator, sm_of(ctx))):
        return frozenset({Strategy.MATRIX})
    return frozenset()


def scratch(strategy, shape, ctx):
    """One set of staging tiles, sized off the same atom the emitter picks.

    Asked before generation, so it cannot depend on anything the body decides
    -- but it may depend on the target, and it has to: the entry `matmul`
    issues is selected against the context's compute capability, so a size
    computed without it is a size for a different instruction.
    """
    if strategy is not Strategy.MATRIX:
        return 0
    return shmsize(1, shape.accumulator, sm_of(ctx), shape.a_parts,
                   lanes=shape.threads)


def plan(strategy, shape, n, ctx):
    """One arrangement over the whole output: nothing here splits a tail.

    A partial tile is padded by `threadrange` instead, which the staged
    fragments make cheap -- the spare lanes read zeroes out of the same tile
    the real ones do.
    """
    return whole(strategy, n)


def _index(writer, *, sub=0, mod=None, div=None, scale=1, add=0, lane=None):
    """A lane-derived index, built as operations rather than spelled out.

    Every address this file computes has the same shape --- the thread index,
    an optional shift, an optional wrap, a stride and an offset --- and it was
    written as text six times.  Text is where the address stops being
    analysable: `cse` cannot merge two identical `rawexpr` nodes (they are not
    pure), the bank census has to parse the generated source to answer a
    question the IR could answer directly, and a pass that wanted to reason
    about the access pattern had nothing to reason over.

    Order is `((tid - sub) % mod / div) * scale + add`, which is the order the
    six call sites already used.  `lane` replaces `tid` where the index is the
    warp's rather than the multiplication's (`_warp_group`).
    """
    v = writer.thread_id('x') if lane is None else lane
    if sub:
        v = writer.op('sub', INDEX, v, sub, hint='a')
    if mod is not None:
        v = writer.op('rem', INDEX, v, mod, hint='a')
    if div is not None:
        v = writer.op('div', INDEX, v, div, hint='a')
    if scale != 1:
        v = writer.op('mul', INDEX, v, scale, hint='a')
    if add:
        v = writer.op('add', INDEX, v, add, hint='a')
    return v


def _warp_group(writer, ops, threads, mults):
    """The warp lane, this multiplication's place in the warp, and how far
    another multiplication's copy of a tile is.

    At one multiplication per warp the lane is `threadIdx.x` and there is no
    other copy, so this answers `None` and no distance -- and that path emits
    what it always did.  Narrower, the multiplications of a warp are
    consecutive in `y`: the lane is `threadIdx.x + threads * (threadIdx.y %
    mults)`, and multiplication `p`'s tile is `p - threadIdx.y % mults`
    regions from this one's, since every multiplication owns the same layout
    of shared memory, `mult_stride` elements after its predecessor's.

    The distance is handed to the access as its `shift`, which the builder
    adds after the tile's swizzle: the permutation stays one of the tile's
    own index, the same for the multiplication that writes the tile and for
    the neighbour that reads it.
    """
    if mults == 1:
        return None, None, (lambda p: None)
    stride = ops.mult_stride
    if stride is None:
        raise InternalError(
            'the MMA path shares a warp between multiplications and addresses '
            'the shared memory of its neighbours, but was not told how much '
            'each multiplication owns')
    mine = writer.op('rem', INDEX, writer.thread_id('y'), mults, hint='m')
    lane = writer.op('add', INDEX, writer.thread_id('x'),
                     writer.op('mul', INDEX, mine, threads, hint='m'),
                     hint='wl')
    back = writer.op('mul', INDEX, mine, -stride, hint='r')

    def region(p):
        return writer.op('add', INDEX, back, p * stride, hint='r') if p else back
    return lane, mine, region


def _lanes_of(start, size, threads):
    """`(slot, first lane, lanes, sub)` for rows `start .. start + size` of a
    lead distribution over `threads` lanes.

    Row `r` sits in lane `r % threads` at slot `r // threads`, so a run of
    rows is one slot where it stays below a multiple of the lane count and
    several where it crosses one.  A lane of the run holds row `lane - sub`
    counted from `start`.
    """
    for s in range(start // threads, (start + size - 1) // threads + 1):
        lo = max(start - s * threads, 0)
        hi = min(start + size - s * threads, threads)
        yield s, lo, hi - lo, start - s * threads


def matmul(writer, ops, ctx, span):
    C, A, B = ops.C, ops.A, ops.B
    # Elements, and the loop below walks them a wave at a time.  The
    # accessors take slots, so `i // threads` is what reaches them.
    M = ops.lead_elements
    N, K, kx = span.stop, ops.k, ops.kx
    threads, dtype, sparse = ops.threads, ops.accumulator, ops.sparse
    # The warp the fragments are spread over, and how many multiplications
    # share it.  One is the warp-per-multiplication path this always was.
    # More, and the warp runs one round of MMAs per multiplication: each one
    # stages its own `A` and `B` in its own shared region, round `p` reads
    # its fragments out of multiplication `p`'s, and `D` goes back through
    # that region to the lanes that own the rows.  The staging stores and the
    # epilogue reads are the multiplication's (`threadIdx.x`); the fragment
    # reads and the `D` stores are the warp's (`lane`).
    wave = WAVE
    mults = wave // threads
    # Lead slots per lane.  A wave of rows spans `mults` of them, and one past
    # the operand's last holds no rows at all.
    nslots = -(-M // threads)

    def threadrange(start, size):
        """The lanes that take part in one staging step.

        A structured `if_` rather than `writer.If`, which takes a string and
        emits a raw block.  The text spelled the same guard, and the emitted
        C++ is identical -- what changes is that the condition is a value, so
        a pass walking the body can tell which lanes reach an access inside.
        Without that, `pir/banks.py` counted all 32 into every bank and read
        72 conflict-free accesses in `rectangular` as 2-way.
        """
        cond = None
        tid = writer.thread_id('x')
        if start > 0:
            cond = writer.op('ge', BOOL, tid, start, hint='g')
        if start + size < threads:
            upper = writer.op('lt', BOOL, tid, start + size, hint='g')
            cond = upper if cond is None else writer.op('and', BOOL, cond,
                                                        upper, hint='g')
        return writer.if_(cond) if cond is not None else writer.AnonymousScope()

    if sparse:
        return False

    # for now.
    # TODO for later: split matrix into tiles
    # if too small for matrix tile (or with zero padded), use FMA instead
    atom = instr_for(dtype, columns=ops.n, lead=ops.lead_elements,
                     depth=ops.k + ops.kx, sm=sm_of(ctx))

    mma = writer.varalloc()
    mmaT = writer.varalloc()

    Ashm = writer.varalloc()
    Bshm = writer.varalloc()



    # Staged fragments, by slot.  Dicts because the B index is a pair and the
    # extents are loop-derived; what matters is that these hold values now, not
    # C++ identifiers built out of a `varalloc` name.
    Areg = {}
    AregParts = None            # per part, allocated once the atom is known
    Breg = {}
    Creg = writer.varalloc()

    # `supports()` is the gate; this is the guard for a direct caller.
    assert threads <= wave and wave % threads == 0
    lane, mine, region = _warp_group(writer, ops, threads, mults)
    # A barrier here meets the warp.  Where the warp holds several
    # multiplications all of them have to arrive, and the count is what says
    # so to `barrier` -- as `convergence_scope` says it to the loop around.
    sync = {} if mults == 1 else {'threads': threads}
    shift = {} if mults == 1 else {'shift': mine}

    ntile = 8
    mtile = 8
    ktile = 4

    nregs = atom.n // ntile
    mregs = atom.m // mtile
    kregs = atom.k // ktile

    aregs = (atom.m * atom.k) // wave
    # How many k tiles a pre-ordered operand was laid out in.  Read from
    # `tile_starts` and not from a ceiling, so the emitter names a tile the
    # same way `fragment_order` did -- the two are one layout, and the only
    # way for them to disagree is to derive it twice.
    ktiles = len(tile_starts(K, wave, atom.k))
    bregs = (atom.n * atom.k) // wave
    cregs = (atom.m * atom.n) // wave

    # The three staging windows, taken from the scratch tail this instruction
    # declared to ShrMemOpt rather than placed by hand.
    #
    # `aoffs = 0`, `boffs = aregs * 32`, `coffs = 0` was not three constants
    # but one packing: C deliberately overlaps A and B, which is why the size
    # is `32 * max(aregs + bregs, cregs)` and not their sum -- 192 elements
    # rather than 320 for m16n8k8.  It is legal because A and B are live only
    # inside the k/kk/ii nest and C only in the epilogue after it closes.
    #
    # That is a lifetime argument, and it was being carried by three integers
    # and an `assert` restating the total.  It belongs to a liveness analysis;
    # until the body is structured enough for one to see it, the windows are
    # requested and the overlap is stated in one place instead of three.
    # Fragment slots, filled by the loads and read by the MMA.  Generously
    # sized: the index is `iii + kk * mregs` and `kkk + jj * kregs`, so the
    # bound is a product of loop extents rather than the register count.
    Afrag = [None] * (aregs * mregs * kregs * 8)
    # One list per round: round `p` multiplies multiplication `p`'s `B`.
    Bfrag = [[None] * (bregs * nregs * kregs * 8) for _ in range(mults)]
    # One staging chain per part.  `a_parts == 1` is the ordinary operand and
    # every list below is one long, which is the shape this code had before
    # there was a second part -- so the prepared case is the general one and
    # the ordinary case is its `n = 1`, rather than the two being branches.
    #
    # A part gets its own tile rather than a wider shared one: the fragment
    # read indexes by slot times the wave, and a tile holding several parts
    # per slot would change that arithmetic for all of them to save one
    # allocation.
    aparts = ops.a_parts
    # Whether `A` is stored in the order this reads it.  Where it is, the
    # staging tile below is not an optimisation that was skipped -- it is a
    # transform that already happened, once, on the host, for a batch that
    # shares the operand.
    aordered = ops.A_slot is not None
    # Whether `B` can be read where it lies instead of redistributed.  Asked
    # once and for the whole span: the tile is either needed or it is not, and
    # a path that staged half the fragments would still pay for it.
    # Only on a warp of its own: a neighbour's `B` lies behind a pointer this
    # lane does not hold, so it comes through the neighbour's tile.
    bdirect = (mults == 1
               and ops.B_frag is not None and ops.B_direct is not None
               and ops.B_direct(ktile, ntile) and atom.n == ntile)
    AregParts = [Areg] + [{} for _ in range(1, aparts)]
    # A warp of its own reads the epilogue tile at the index the lane gives;
    # a shared one reads its own rows, one run per slot they span.
    alone = mults == 1
    # The staging stores and the epilogue reads happen in every
    # multiplication of the warp at once, each in its own region and at the
    # same offsets -- so the regions' distance decides their banks, and where
    # it is a multiple of 32 all of them land on the same ones.  So a tile is
    # staggered by `stagger(...)` elements per multiplication, making the
    # distance the one its access pattern needs: `B` is staged in runs of
    # eight (eight banks apart), and the epilogue reads swizzled rows of eight,
    # whose banks repeat every eight (`8 / mults` apart, modulo eight).  `A` is
    # not: it is staged four consecutive elements per lane, which ptxas merges
    # into one 16-byte store, and a 16-byte access is served eight lanes at a
    # time -- one multiplication at eight lanes, half of one at sixteen -- so
    # its regions never meet in a bank, and a stagger that is not a multiple
    # of four only breaks the merge (measured: twice the stores, 3-way).  The
    # fragment reads and the `D` stores see one region per instruction, and a
    # shift common to all lanes does not change their banks.  Measured on
    # `local_flux` at eight lanes: unstaggered, the `B` stores were 3-way and
    # the epilogue reads 4-way.
    def stagger(target):
        if alone or mults > 8:
            return 0
        return (target - ops.mult_stride) % 32
    apad, bpad, cpad = 0, stagger(atom.k), stagger(8 // mults)

    def at(p, pad):
        """The shift to multiplication `p`'s copy of a tile staggered by
        `pad`, or to this lane's own where `p` is `None`."""
        if alone:
            return None
        if p is None:
            return writer.op('mul', INDEX, mine, pad, hint='r') if pad else None
        r = region(p)
        return writer.op('add', INDEX, r, p * pad, hint='r') if pad and p else r

    # The columns past the last whole tile.  Padded into a tile of their own
    # they cost as much as a whole one -- three HMMAs a step for the ninth
    # column of `local_flux` -- so a narrow remainder is carried by the last
    # whole tile instead: every lane multiplies the A fragments it holds anyway
    # by the remainder's B, and the four lanes sharing a row sum their partials
    # in the epilogue, through the tile `D` goes through (`TAIL_MAX`).
    width = N - span.start
    ntail = width % atom.n
    tailed = (0 < ntail <= TAIL_MAX and width > atom.n
              and atom.mode in TAIL_MODES)
    jstops = list(range(span.start, N - ntail if tailed else N, atom.n))
    tailbase = cregs * wave
    tailcol = atom.m * ktile

    with writer.scratch_scope():
        # `aparts` scalars per slot, adjacent, so a fragment's parts are one
        # access rather than one each -- see the note below the B tile.
        Ashm = writer.alloc(atom.d, (aparts * aregs * wave + (mults - 1) * apad,),
                            MemSpace.SHARED, hint='atile')
        # The B tile is written a row at a time and read a column at a time,
        # which no linear stride can serve without bank conflicts: 32 lanes
        # read 32 distinct elements spread over 60, and 240 bytes do not fit
        # in 128 of bank width.  Padding moves the collision, transposing
        # moves it to the store; permuting each row costs nothing and clears
        # both.  Measured over the emitted addresses: 2-way -> 1-way.
        Bshm = writer.alloc(atom.d, (bregs * wave + tailed * ntail * atom.k
                                     + (mults - 1) * bpad,), MemSpace.SHARED,
                            hint='btile', swizzle=XorSwizzle(atom.k))
        # One tile with the parts *adjacent*, not one tile per part.
        #
        # The first arrangement gave each part its own tile, on the grounds
        # that the fragment read indexes by slot times the wave and a shared
        # tile holding several parts per slot would change that arithmetic.
        # True, and the wrong thing to optimise: it saved one allocation and
        # paid one access per fragment per part.  Measured, two parts cost
        # +896 LDS and +904 global loads against the single-part kernel, and
        # the kernel is bound by exactly that traffic.
        #
        # Adjacent, the two halves of one fragment are one 8-byte access, and
        # the same holds in global memory, where `DataView._elem_parts` already
        # interleaves them.  The address gains a factor and an addend; ptxas
        # merges the neighbouring scalar accesses, as it already does for the
        # four consecutive stores below.

    with writer.scratch_scope():
        # Written lane-strided across the whole warp and read row-strided by
        # `atom.n`, so it collides both ways: 4-way on the read, 2-way on the
        # write.  The width is the wave, not the row -- and it has to be
        # chosen per tile rather than fixed, because no single value serves
        # every access here.  Measured over the four patterns this path emits:
        #
        #             none   xor8  xor16  xor32
        #   B load     2-w    1-w    2-w    2-w
        #   C load     4-w    2-w    1-w    1-w
        #   C store    2-w    2-w    2-w    1-w
        #
        # `tools/bank_conflicts.py` is what keeps those honest.
        Cshm = writer.alloc(atom.d, (tailbase + tailed * ntail * tailcol
                                     + (mults - 1) * cpad,), MemSpace.SHARED,
                            hint='ctile', swizzle=XorSwizzle(wave))

    x4type = {
        Datatype.F32: 'float4',
        Datatype.F64: 'double4'
    }[dtype]

    for j in jstops:
        # The last whole tile carries the narrow remainder, if there is one.
        tail = ntail if tailed and j == jstops[-1] else 0
        ncols = atom.n + tail
        with writer.AnonymousScope():
            for k in range(0, K + kx, threads):
                # `var is None` asks the accessor for the value rather than a
                # name to write into -- the protocol has said so since the
                # sparse loader took it, and the MMA path simply never used it.
                # The padding slots are a `declare` for the same reason they
                # were a raw declaration: nothing loads them, and the MMA reads
                # a zero.
                for jj in range(0, min(ncols, N - j)):
                    Breg[k // threads, jj] = B(writer, None, j + jj, k // threads)
                for jj in range(min(ncols, N - j), ncols):
                    Breg[k // threads, jj] = writer.declare(ScalarType(atom.d),
                                                            hint='bs')
            for i in range(0, M, wave):
                with writer.AnonymousScope():
                    # One value per accumulator slot rather than a `[cregs][n]`
                    # array named by `varalloc`.  The array was a C++
                    # identifier the IR knew nothing about, so `mma.sync`'s
                    # read-write operand could not be a value and the asm had
                    # to stay raw text.  Same registers, same initialisation;
                    # the difference is that each slot now has a definition
                    # point and a use chain.  One set per round: round `p`
                    # accumulates multiplication `p`'s rows.
                    tiles = list(range(0, min(wave, M - i), atom.m))
                    # Where the step has too few chains for its HMMAs to
                    # issue back to back, every product gets an accumulator
                    # of its own (`CHAINS_MIN`).  `Cvals` is the first.
                    naccs = atom.terms if len(tiles) * mults < CHAINS_MIN else 1
                    accs = [[[[writer.declare(ScalarType(atom.d), hint='c')
                               for _ in range(wave // atom.m)]
                              for _ in range(cregs)]
                             for _ in range(mults)]
                            for _ in range(naccs)]
                    Cvals = accs[0]
                    # The remainder's partial sums, `(round, tile, column, row
                    # group) -> value`: a lane's share of rows `g` and `g + 8`,
                    # over the `k` it holds.  Values and not declared registers,
                    # because the steps below share one scope.
                    Tvals = {}
                    # Every step of the contraction, in one scope rather than
                    # a block each: a value loaded for the next step has to be
                    # visible there.
                    steps = [(k, kk) for k in range(0, K, wave)
                             for kk in range(0, min(wave, K - k), atom.k)]

                    def quads(k, kk):
                        """Every tile's A fragments for one step, as the order
                        stored them: `aregs` consecutive slots from `tbase +
                        aregs * lane`, one wide access per part -- the
                        register group the instruction takes, loaded in place
                        (`fragment_order`).  The operand is batch-constant, so
                        a warp shared by several multiplications reads it once
                        for all of them, over all of its lanes (`shift`)."""
                        out = {}
                        for ii in tiles:
                            mt = (i + ii) // atom.m
                            kt = (k + kk) // atom.k
                            tbase = (mt * ktiles + kt) * (aregs * wave)
                            got = ops.A_slot(writer, tbase, parts=aparts,
                                             width=aregs, **shift)
                            got = got if aparts > 1 else (got,)
                            out[ii] = {None: [[(got[pt] if aregs == 1 else
                                                writer.extract(got[pt], f, hint='a'))
                                               for f in range(aregs)]
                                              for pt in range(aparts)]}
                        return out

                    # One step ahead where the fragments come straight from
                    # memory and a warp holds one multiplication: nothing else
                    # stands between such a load and its instruction -- the
                    # split that used to is gone when the operand is stored
                    # split -- and measured on GB200 the instructions waited on
                    # it (long scoreboard, 54 % of their stalls, the loads a
                    # median 13 instructions ahead).  A shared warp has its
                    # rounds in between already.
                    ahead = quads(*steps[0]) if aordered and PREFETCH else None
                    for s, (k, kk) in enumerate(steps):
                        trueK = kk + kx
                        if not bdirect:
                            # Read once per lane at one
                            # distribution and handed to the lanes
                            # that want it at another.  Where the
                            # fragment's own address is
                            # expressible, none of this happens.
                            # Every multiplication stages its own
                            # rows in its own region, one run per
                            # slot the step spans -- the remainder's
                            # columns with them.
                            writer.barrier('wave', **sync)
                            for s_, lo, cnt, sub in _lanes_of(k + trueK, atom.k,
                                                              threads):
                                with threadrange(lo, cnt):
                                    for jj in range(0, ncols):
                                        writer.store(Bshm, Breg[s_, jj],
                                                     _index(writer, sub=sub, mod=atom.k,
                                                            add=jj * atom.k),
                                                     shift=at(None, bpad))
                            writer.barrier('wave', **sync)

                        for jj in range(0, nregs):
                            for kkk in range(0, kregs):
                                # The loaded value *is* the
                                # fragment.  Copying it into a
                                # `varalloc` name and handing the
                                # name to the MMA was pure
                                # indirection: a declaration and an
                                # assignment per fragment, and a
                                # C++ identifier where the IR had a
                                # value all along.
                                # The fragment layout: the lane's
                                # column within the tile, plus its
                                # row scaled by the tile width.
                                # Two lane terms, so `_index` does
                                # not fit and the sum is written
                                # out -- still operations, and the
                                # two `thread_id` reads are one
                                # value after `cse`.
                                if bdirect:
                                    Bfrag[0][kkk + jj * kregs] = _bfrag(
                                        writer, ops, threadrange,
                                        j + jj * ntile,
                                        k + trueK + kkk * ktile,
                                        ktile, ntile, N, atom,
                                        threads)
                                    continue
                                col = _index(writer, mod=ktile, lane=lane)
                                row = _index(writer, div=ktile,
                                             add=jj * ntile,
                                             scale=atom.k, lane=lane)
                                addr = writer.op('add', INDEX, col, row, hint='a')
                                if kkk * ktile:
                                    addr = writer.op('add', INDEX, addr,
                                                     kkk * ktile, hint='a')
                                # Round `p` reads multiplication
                                # `p`'s tile.
                                for p in range(mults):
                                    Bfrag[p][kkk + jj * kregs] = writer.load(
                                        Bshm, addr, hint='b', shift=at(p, bpad))

                        # The remainder's B, at the `k` a lane's A fragments
                        # hold -- `t % ktile` plus the fragment's `ktile` step
                        # -- and one column for every lane.  Read where it lies
                        # where `B` is (`nblock` of one: no lane term in the
                        # column), from the staged tile otherwise.
                        btail = {}
                        for p in range(mults):
                            for kf in range(kregs):
                                for ci in range(tail):
                                    if bdirect:
                                        btail[p, kf, ci] = ops.B_frag(
                                            writer, j + atom.n + ci,
                                            k + trueK + kf * ktile, ktile, 1)
                                        continue
                                    at_ = _index(writer, mod=ktile, lane=lane,
                                                 add=kf * ktile + (atom.n + ci) * atom.k)
                                    btail[p, kf, ci] = writer.load(
                                        Bshm, at_, hint='b', shift=at(p, bpad))

                        # Parts innermost, so one element's parts
                        # are read next to each other.  They are
                        # adjacent in memory -- the part index is
                        # the innermost stride -- but a compiler
                        # merges neighbouring accesses only where
                        # they are also neighbours in the
                        # instruction stream, and reading all of
                        # part 0 and then all of part 1 leaves them
                        # far apart.  Measured: separated, the two
                        # parts cost 1809 global loads; adjacent,
                        # ptxas folds them back into the 905 the
                        # single-part kernel issues.
                        if aordered and PREFETCH:
                            frags_by_ii = ahead
                            ahead = (quads(*steps[s + 1])
                                     if s + 1 < len(steps) else None)
                        elif aordered:
                            frags_by_ii = quads(k, kk)
                        else:
                            # Read once per lane and redistributed
                            # through the tile below.  A
                            # pre-ordered operand skips both: it
                            # was redistributed before the kernel
                            # ran, so a lane reads its own
                            # fragments and nothing else's.  A
                            # wave of rows spans `mults` slots of
                            # a lane, and one past the operand's
                            # last has no rows and holds zero.
                            for q in range(mults):
                                slot = i // threads + q
                                live = (min(atom.k, K - k - kk)
                                        if slot < nslots else 0)
                                for kkk in range(0, live):
                                    got = A(writer, None, slot,
                                            k + kk + kkk, parts=aparts)
                                    got = got if aparts > 1 else (got,)
                                    for pt in range(aparts):
                                        AregParts[pt][q, kkk] = got[pt]
                                for kkk in range(live, atom.k):
                                    for pt in range(aparts):
                                        # A padding slot reads zero in
                                        # every part, and for a split
                                        # that is the right answer: it
                                        # says the part before it was
                                        # exact, which for a slot
                                        # nothing multiplies is true.
                                        AregParts[pt][q, kkk] = writer.declare(
                                            ScalarType(atom.d), hint='as')

                            # Every tile's fragments before any product, so
                            # that the products can be issued across tiles.
                            frags_by_ii = {}
                            for ii in tiles:
                                writer.barrier('wave', **sync)
                                for q, lo, cnt, sub in _lanes_of(ii, atom.m, threads):
                                    with threadrange(lo, cnt):
                                        # for kkk in range(0, atom.k):
                                        #     writer(f'{shmptr}[{aoffs} + (threadIdx.x - {ii}) % {atom.m} + {kkk * atom.m}] = {Areg}_{kkk};')
                                        for kkk in range(0, atom.k, ktile):
                                            # `ktile` consecutive
                                            # elements, written one at a
                                            # time rather than packed.
                                            #
                                            # This was a `pack` into
                                            # `ScalarType(atom.d, 4)`
                                            # and one wide store, which
                                            # is what the addresses
                                            # deserve -- and which nvcc
                                            # refuses.  `CudaLexic`
                                            # renders a packed value as
                                            # `tensorforge::VectorT<T,
                                            # 4>`, a GNU `vector_size`
                                            # typedef, and the device
                                            # front end declines a
                                            # *value* of that type: "is
                                            # a vector, which is not
                                            # supported in device code",
                                            # 101 times over a corpus
                                            # case.  `cuda.h` predicted
                                            # exactly this.
                                            #
                                            # The spelling is not fixed
                                            # in the lexic because the
                                            # lexic is right for its own
                                            # reasons: `float4` has no
                                            # arithmetic operators and
                                            # cannot be assigned through
                                            # a `VectorRelaxedT`
                                            # pointer, which the staging
                                            # transfers need.  Neither
                                            # applies here -- this value
                                            # is only ever stored -- so
                                            # the narrower spelling is
                                            # local to the one site that
                                            # cannot have the wider one.
                                            #
                                            # It costs a wide store.
                                            # Reinstating one needs a
                                            # device-legal vector value,
                                            # not a different lexic.
                                            base = _index(
                                                writer, sub=sub, mod=atom.m,
                                                scale=ktile, add=kkk * atom.m)
                                            for n in range(ktile):
                                                addr = base if n == 0 else writer.op(
                                                    'add', INDEX, base, n, hint='a')
                                                wide = (addr if aparts == 1
                                                        else writer.op('mul', INDEX, addr,
                                                                       aparts, hint='a'))
                                                for pt in range(aparts):
                                                    to = (wide if pt == 0
                                                          else writer.op('add', INDEX,
                                                                         wide, pt,
                                                                         hint='a'))
                                                    writer.store(Ashm,
                                                                 AregParts[pt][q, kkk + n],
                                                                 to, shift=at(None, apad))
                                writer.barrier('wave', **sync)

                                # Where `A` is the same for every
                                # multiplication every lane reads
                                # multiplication 0's copy, once
                                # for all rounds -- one region for
                                # the whole warp, so the banks are
                                # the tile's alone.  Otherwise
                                # round `p` reads `p`'s.
                                srcs = ([None] if alone else [0] if ops.a_uniform
                                        else list(range(mults)))
                                frags = {}
                                for src in srcs:
                                    got_parts = [[None] * aregs for _ in range(aparts)]
                                    for kf in range(0, kregs):
                                        for iii in range(0, mregs):
                                            #writer(f'{atom.d.ctype()} {Areg2}_{iii + kk * mregs} = {shmptr}[{aoffs} + (threadIdx.x / {ktile}) + (threadIdx.x % {ktile} + {kk * ktile}) * {atom.m} + {iii * mtile}];')
                                            faddr = _index(writer, add=(iii + kf * mregs) * wave,
                                                           lane=lane)
                                            if aparts > 1:
                                                faddr = writer.op('mul', INDEX, faddr,
                                                                  aparts, hint='a')
                                            far = None if src is None else at(src, apad)
                                            got_parts[0][iii + kf * mregs] = writer.load(
                                                Ashm, faddr, hint='a', shift=far)
                                            for pt in range(1, aparts):
                                                got_parts[pt][iii + kf * mregs] = writer.load(
                                                    Ashm, writer.op('add', INDEX, faddr, pt,
                                                                    hint='a'), hint='a',
                                                    shift=far)
                                    frags[src] = got_parts
                                frags_by_ii[ii] = frags

                        # Where A was stored prepared the
                        # parts are handed over as they
                        # were read, one tuple per
                        # fragment.  What a mode does with
                        # them is its own: the TF32 branch
                        # takes two and issues three
                        # products, a BF16 one would take
                        # three.  The reinterpretation to
                        # the operand type is
                        # arithmetic-free.
                        prods = {}
                        for ii in tiles:
                            frags = frags_by_ii[ii]
                            splits = {src: ([tuple(_as_tf32(writer, got_parts[pt][f])
                                                   for pt in range(aparts))
                                             for f in range(aregs)]
                                            if aparts > 1 else None)
                                      for src, got_parts in frags.items()}
                            for p in range(mults):
                                src = p if p in frags else next(iter(frags))
                                prods[ii, p] = atom.products(
                                    writer, frags[src][0][:aregs], Bfrag[p][:bregs],
                                    a_split=splits[src])
                        # Term by term across every accumulator of the step,
                        # not accumulator by accumulator: 3xTF32 puts its
                        # three products into one accumulator, and issued
                        # back to back each waits for the one before.
                        # Measured on GB200 (HMMA latency ~20 cycles, one
                        # issued per 8): two accumulators between them cost
                        # 25 % at one warp per scheduler, eight cost nothing
                        # -- and a shared warp has `2 * mults`.
                        for term in range(len(prods[tiles[0], 0])):
                            for ii in tiles:
                                for p in range(mults):
                                    a_op, b_op = prods[ii, p][term]
                                    acc = [accs[term % naccs][p][c][ii // atom.m]
                                           for c in range(cregs)]
                                    atom.asmcall(writer, acc, a_op, b_op, acc)

                        # The remainder on the fragments already in hand: a
                        # lane's A fragment `iii + kf * mregs` is row `g + 8
                        # * iii`, column `t + ktile * kf` of the tile, so it
                        # meets the remainder's B at that column.  In full
                        # precision -- the parts added back where A is stored
                        # split.
                        if tail:
                            fp = ScalarType(atom.d)
                            full = {}
                            for ii in tiles:
                                frags = frags_by_ii[ii]
                                for p in range(mults):
                                    src = p if p in frags else next(iter(frags))
                                    if (ii, src) not in full:
                                        parts_ = frags[src]
                                        full[ii, src] = [
                                            parts_[0][f] if aparts == 1 else
                                            writer.op('add', fp, parts_[0][f],
                                                      parts_[1][f], hint='af')
                                            for f in range(aregs)]
                                    af = full[ii, src]
                                    for ci in range(tail):
                                        for iii in range(mregs):
                                            acc = Tvals.get((p, ii, ci, iii))
                                            for kf in range(kregs):
                                                a_ = af[iii + kf * mregs]
                                                b_ = btail[p, kf, ci]
                                                acc = (writer.op('mul', fp, a_, b_, hint='t')
                                                       if acc is None else
                                                       writer.op('fma', fp, a_, b_, acc,
                                                                 hint='t'))
                                            Tvals[p, ii, ci, iii] = acc

                    # The separate accumulators, summed: the corrections first,
                    # which are small against the product they correct.
                    for ii in (tiles if naccs > 1 else ()):
                        for p in range(mults):
                            for c in range(cregs):
                                part = [accs[a][p][c][ii // atom.m]
                                        for a in range(naccs)]
                                corr = part[1]
                                for more in part[2:]:
                                    corr = writer.op('add', ScalarType(atom.d), corr,
                                                     more, hint='c')
                                Cvals[p][c][ii // atom.m] = writer.op(
                                    'add', ScalarType(atom.d), part[0], corr, hint='c')

                    # The epilogue's staging registers.  Assigned inside a
                    # thread guard and read outside it, so they are declared
                    # here and written through `assign` rather than being the
                    # result of the load: a value defined inside the guard
                    # would not be visible to the store that follows.
                    Cout = {(q, jj): writer.declare(ScalarType(atom.d), hint='c')
                            for q in range(mults) for jj in range(ncols)}

                    for ii in range(0, wave, atom.m):
                        with writer.AnonymousScope():
                            # The lane's own term is `2 * t`; the rest is the
                            # instruction's fragment shape, which
                            # `accumulator_slots` states and a test checks.
                            # Round `p`'s `D` goes to multiplication `p`'s
                            # region, so each one reads its rows back out of
                            # its own.  The remainder's partials go after it,
                            # a lane's at row `g + 8 * iii`, position `t`.
                            has_tail = tail and (0, ii, 0, 0) in Tvals
                            for p in range(mults):
                                for slot, off in accumulator_slots(atom):
                                    writer.store(Cshm, Cvals[p][slot][ii // atom.m],
                                                 _index(writer, scale=2, add=off, lane=lane),
                                                 shift=at(p, cpad))
                                for ci in range(tail if has_tail else 0):
                                    for iii in range(mregs):
                                        writer.store(Cshm, Tvals[p, ii, ci, iii],
                                                     _index(writer, lane=lane,
                                                            add=tailbase + ci * tailcol
                                                            + iii * mtile * ktile),
                                                     shift=at(p, cpad))

                            writer.barrier('wave', **sync)
                            for q, lo, cnt, sub in _lanes_of(ii, atom.m, threads):
                                with threadrange(lo, cnt):
                                    for jj in range(0, atom.n):
                                        idx = (_index(writer, mod=atom.m, scale=atom.n, add=jj)
                                               if alone else
                                               _index(writer, sub=sub, mod=atom.m,
                                                      scale=atom.n, add=jj))
                                        _c = writer.load(Cshm, idx, hint='data',
                                                         shift=at(None, cpad))
                                        writer.assign(Cout[q, jj], _c)
                                    # A row's remainder: the four partials of
                                    # the lanes that shared it, summed.
                                    for ci in range(tail if has_tail else 0):
                                        total = None
                                        for tq in range(ktile):
                                            idx = _index(writer, sub=sub, mod=atom.m,
                                                         scale=ktile,
                                                         add=tailbase + ci * tailcol + tq)
                                            v = writer.load(Cshm, idx, hint='data',
                                                            shift=at(None, cpad))
                                            total = v if total is None else writer.op(
                                                'add', ScalarType(atom.d), total, v, hint='t')
                                        writer.assign(Cout[q, atom.n + ci], total)
                            writer.barrier('wave', **sync)

                    for q in range(mults):
                        if i // threads + q >= nslots:
                            continue
                        for jj in range(0, min(ncols, N - j)):
                            C(writer, Cout[q, jj], i // threads + q, j + jj)

    return True
