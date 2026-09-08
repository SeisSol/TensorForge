# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
from tensorforge.common.basic_types import Datatype
from .. import ranking
from ..strategy import Strategy, whole
from tensorforge.backend.pir.core import (BOOL, INDEX, Access, Effect, MemSpace,
                                          XorSwizzle,
                                          Uniformity,
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
    def headers(self):
        return []

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

    def generate(self, writer, context, A, B, C, uses=(), a_split=None):
        """`a_split` is the A halves where they were read rather than computed.

        Same shape `tfconvert` returns -- a `(upper, lower)` per fragment -- so
        the three issues below are unchanged and only their source differs.
        Given, the conversion of A does not happen at all; that is the point of
        storing the operand prepared, and it is the whole of what this
        parameter does.
        """
        with writer.Scope():
            if self.mode == MMAMode.I8:

                pass
            if self.mode == MMAMode.TF32:
                Atf32 = a_split if a_split is not None else tfconvert(writer, A)
                Btf32 = tfconvert(writer, B)

                self.asmcall(writer, C, [a[0] for a in Atf32], [b[0] for b in Btf32], C, uses)
                self.asmcall(writer, C, [a[0] for a in Atf32], [b[1] for b in Btf32], C, uses)
                self.asmcall(writer, C, [a[1] for a in Atf32], [b[0] for b in Btf32], C, uses)
            else:
                self.asmcall(writer, C, A, B, C, uses)

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


def supports(threads, dtype, sparse, depth=0) -> bool:
    """Whether `matmul` can emit for this shape, asked *before* it is called.

    This was an `assert` inside the emitter, which was safe only for as long
    as nothing reached it.  Turning the path on makes the difference matter:
    an assertion aborts generation for a case the generic path handles
    perfectly well, so the preconditions have to be a question the caller can
    ask, not a crash the caller cannot avoid.

    * ``threads == 32``.  The emitter is warp-level throughout -- it stages
      operands through `__syncwarp` and indexes shared memory by
      `threadIdx.x` modulo the atom's `k`.  Narrower waves would need a
      warp-level broadcast and a way back; wider ones are a different
      instruction.  Neither exists here yet.
    * ``not sparse``.  `matmul` already declines these by returning `False`,
      but `temp_shmem` reserves shared memory off the same predicate; if the
      two disagree the reservation is made for a kernel that never uses it.
    * ``depth >= MIN_DEPTH``.  A shallow contraction reaches this path and then
      spends most of its issues on padding; see `MIN_DEPTH`.  `depth == 0` is
      "the caller does not know", which is not the same as "shallow" and is
      admitted -- `shmsize` asks without a shape and must keep its upper bound.
    """
    return (threads == 32 and dtype in (Datatype.F32, Datatype.F64)
            and not sparse and (depth == 0 or depth >= MIN_DEPTH))


def shmsize(stages, dtype, sm=None, a_parts=1):
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
        return 32 * max(aregs + bregs, cregs)

    return max((size(atom) for atom in instrs_for(dtype, sm)), default=0)

def strategies(shape, ctx):
    """What this target can emit for this shape.

    One arrangement, and it is off: `ENABLED` is the deployment switch and
    `supports` the shape gate.  Asking both here rather than at the call site
    is what keeps a shape this cannot serve falling through to the nest
    instead of reaching an assertion inside the emitter.

    A packed lead operand is declined, and here the reason is that the
    question cannot be asked yet: the fragments are staged through shared
    memory at offsets this module writes out by hand, and there is no layout
    table for them -- no counterpart to AMD's `FRAGMENT_BITS`.  Without one
    there is nothing to ask `reach` about, so the refusal is a literal rather
    than a route, and it stays one until the table is written.
    """
    if shape.lead_width > 1:
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
    return shmsize(1, shape.accumulator, sm_of(ctx), shape.a_parts)


def plan(strategy, shape, n, ctx):
    """One arrangement over the whole output: nothing here splits a tail.

    A partial tile is padded by `threadrange` instead, which the staged
    fragments make cheap -- the spare lanes read zeroes out of the same tile
    the real ones do.
    """
    return whole(strategy, n)


def _index(writer, *, sub=0, mod=None, div=None, scale=1, add=0):
    """A lane-derived index, built as operations rather than spelled out.

    Every address this file computes has the same shape --- the thread index,
    an optional shift, an optional wrap, a stride and an offset --- and it was
    written as text six times.  Text is where the address stops being
    analysable: `cse` cannot merge two identical `rawexpr` nodes (they are not
    pure), the bank census has to parse the generated source to answer a
    question the IR could answer directly, and a pass that wanted to reason
    about the access pattern had nothing to reason over.

    Order is `((tid - sub) % mod / div) * scale + add`, which is the order the
    six call sites already used.
    """
    v = writer.thread_id('x')
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


def matmul(writer, ops, ctx, span):
    C, A, B = ops.C, ops.A, ops.B
    # Elements, and the loop below walks them in strides of `threads`.  The
    # accessors take slots, so `i // threads` is what reaches them.
    M = ops.lead_elements
    N, K, kx = span.stop, ops.k, ops.kx
    threads, dtype, sparse = ops.threads, ops.accumulator, ops.sparse

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
    assert threads == 32

    ntile = 8
    mtile = 8
    ktile = 4

    nregs = atom.n // ntile
    mregs = atom.m // mtile
    kregs = atom.k // ktile

    aregs = (atom.m * atom.k) // threads
    bregs = (atom.n * atom.k) // threads
    cregs = (atom.m * atom.n) // threads

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
    Bfrag = [None] * (bregs * nregs * kregs * 8)
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
    AregParts = [Areg] + [{} for _ in range(1, aparts)]
    AfragParts = [[None] * len(Afrag) for _ in range(aparts)]

    with writer.scratch_scope():
        # `aparts` scalars per slot, adjacent, so a fragment's parts are one
        # access rather than one each -- see the note below the B tile.
        Ashm = writer.alloc(atom.d, (aparts * aregs * threads,),
                            MemSpace.SHARED, hint='atile')
        # The B tile is written a row at a time and read a column at a time,
        # which no linear stride can serve without bank conflicts: 32 lanes
        # read 32 distinct elements spread over 60, and 240 bytes do not fit
        # in 128 of bank width.  Padding moves the collision, transposing
        # moves it to the store; permuting each row costs nothing and clears
        # both.  Measured over the emitted addresses: 2-way -> 1-way.
        Bshm = writer.alloc(atom.d, (bregs * threads,), MemSpace.SHARED,
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
        Cshm = writer.alloc(atom.d, (cregs * threads,), MemSpace.SHARED,
                            hint='ctile', swizzle=XorSwizzle(threads))

    x4type = {
        Datatype.F32: 'float4',
        Datatype.F64: 'double4'
    }[dtype]

    for j in range(span.start, N, atom.n):
        with writer.AnonymousScope():
            for k in range(0, K + kx, threads):
                # `var is None` asks the accessor for the value rather than a
                # name to write into -- the protocol has said so since the
                # sparse loader took it, and the MMA path simply never used it.
                # The padding slots are a `declare` for the same reason they
                # were a raw declaration: nothing loads them, and the MMA reads
                # a zero.
                for jj in range(0, min(atom.n, N - j)):
                    Breg[k // threads, jj] = B(writer, None, j + jj, k // threads)
                for jj in range(min(atom.n, N - j), atom.n):
                    Breg[k // threads, jj] = writer.declare(ScalarType(atom.d),
                                                            hint='bs')
            for i in range(0, M, threads):
                with writer.AnonymousScope():
                    # One value per accumulator slot rather than a `[cregs][n]`
                    # array named by `varalloc`.  The array was a C++
                    # identifier the IR knew nothing about, so `mma.sync`'s
                    # read-write operand could not be a value and the asm had
                    # to stay raw text.  Same registers, same initialisation;
                    # the difference is that each slot now has a definition
                    # point and a use chain.
                    Cvals = [[writer.declare(ScalarType(atom.d), hint='c')
                              for _ in range(threads // atom.m)]
                             for _ in range(cregs)]
                    for k in range(0, K, threads):
                        with writer.AnonymousScope():
                            for kk in range(0, min(threads, K - k), atom.k):
                                with writer.AnonymousScope():
                                    writer.barrier(Uniformity.MULT)
                                    trueK = kk + kx
                                    trueSK = min(atom.k, threads - trueK)
                                    with threadrange(trueK, trueSK):
                                        for jj in range(0, atom.n):
                                            writer.store(Bshm, Breg[k // threads, jj],
                                                         _index(writer, sub=trueK, mod=atom.k, add=jj * atom.k))
                                    if trueSK != atom.k:
                                        with threadrange(0, atom.k - trueSK):
                                            for jj in range(0, atom.n):
                                                writer.store(Bshm, Breg[k // threads + 1, jj],
                                                                     _index(writer, sub=-trueSK, mod=atom.k, add=jj * atom.k))
                                    writer.barrier(Uniformity.MULT)

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
                                            col = _index(writer, mod=ktile)
                                            row = _index(writer, div=ktile,
                                                         add=jj * ntile,
                                                         scale=atom.k)
                                            addr = writer.op('add', INDEX, col, row, hint='a')
                                            if kkk * ktile:
                                                addr = writer.op('add', INDEX, addr,
                                                                 kkk * ktile, hint='a')
                                            Bfrag[kkk + jj * kregs] = writer.load(Bshm, addr, hint='b')

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
                                    for kkk in range(0, min(atom.k, K - k - kk)):
                                        for pt in range(aparts):
                                            AregParts[pt][kkk] = A(writer, None,
                                                                   i // threads,
                                                                   k + kk + kkk, pt)
                                    for kkk in range(min(atom.k, K - k - kk), atom.k):
                                        for pt in range(aparts):
                                            # A padding slot reads zero in every
                                            # part, and for a split that is the
                                            # right answer: it says the part
                                            # before it was exact, which for a
                                            # slot nothing multiplies is true.
                                            AregParts[pt][kkk] = writer.declare(
                                                ScalarType(atom.d), hint='as')

                                    for ii in range(0, min(threads, M - i), atom.m):
                                        with writer.AnonymousScope():
                                            writer.barrier(Uniformity.MULT)
                                            with threadrange(ii, atom.m):
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
                                                        writer, sub=ii, mod=atom.m,
                                                        scale=ktile, add=kkk * atom.m)
                                                    for n in range(ktile):
                                                        addr = base if n == 0 else writer.op(
                                                            'add', INDEX, base, n, hint='a')
                                                        wide = (addr if aparts == 1
                                                                else writer.op('mul', INDEX, addr,
                                                                               aparts, hint='a'))
                                                        for pt in range(aparts):
                                                            at = (wide if pt == 0
                                                                  else writer.op('add', INDEX,
                                                                                 wide, pt,
                                                                                 hint='a'))
                                                            writer.store(Ashm,
                                                                         AregParts[pt][kkk + n],
                                                                         at)
                                            writer.barrier(Uniformity.MULT)

                                            for kk in range(0, kregs):
                                                for iii in range(0, mregs):
                                                    #writer(f'{atom.d.ctype()} {Areg2}_{iii + kk * mregs} = {shmptr}[{aoffs} + (threadIdx.x / {ktile}) + (threadIdx.x % {ktile} + {kk * ktile}) * {atom.m} + {iii * mtile}];')
                                                    faddr = _index(writer, add=(iii + kk * mregs) * 32)
                                                    if aparts > 1:
                                                        faddr = writer.op('mul', INDEX, faddr,
                                                                          aparts, hint='a')
                                                    Afrag[iii + kk * mregs] = writer.load(Ashm, faddr, hint='a')
                                                    for pt in range(1, aparts):
                                                        AfragParts[pt][iii + kk * mregs] = writer.load(
                                                            Ashm, writer.op('add', INDEX, faddr, pt,
                                                                            hint='a'), hint='a')

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
                                            AfragParts[0] = Afrag
                                            a_split = ([tuple(_as_tf32(writer, AfragParts[pt][f])
                                                              for pt in range(aparts))
                                                        for f in range(aregs)]
                                                       if aparts > 1 else None)
                                            atom.generate(writer, ctx, Afrag[:aregs], Bfrag[:bregs],
                                                          [Cvals[i][ii // atom.m] for i in range (cregs)],
                                                          a_split=a_split)

                    # The epilogue's staging registers.  Assigned inside a
                    # thread guard and read outside it, so they are declared
                    # here and written through `assign` rather than being the
                    # result of the load: a value defined inside the guard
                    # would not be visible to the store that follows.
                    Cout = [writer.declare(ScalarType(atom.d), hint='c')
                            for _ in range(atom.n)]

                    for ii in range(0, threads, atom.m):
                        with writer.AnonymousScope():
                            # The lane's own term is `2 * t`; the rest is the
                            # instruction's fragment shape, which
                            # `accumulator_slots` states and a test checks.
                            for slot, off in accumulator_slots(atom):
                                writer.store(Cshm, Cvals[slot][ii // atom.m],
                                             _index(writer, scale=2, add=off))

                            writer.barrier(Uniformity.MULT)
                            with threadrange(ii, atom.m):
                                for jj in range(0, atom.n):
                                    _c = writer.load(Cshm, _index(writer, mod=atom.m, scale=atom.n, add=jj), hint='data')
                                    writer.assign(Cout[jj], _c)
                            writer.barrier(Uniformity.MULT)

                    for jj in range(0, min(atom.n, N - j)):
                        C(writer, Cout[jj], i // threads, j + jj)

    return True
