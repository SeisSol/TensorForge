# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
from tensorforge.common.basic_types import Datatype
from tensorforge.backend.pir.core import (INDEX, Access, Effect, MemSpace,
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
    out = []
    for variable in variables:
        upper = writer.declare(TF32_HALF, hint='u')
        lower = writer.declare(TF32_HALF, hint='l')
        writer.call_stmt('tensorforge::splitFloatTF32', upper, lower, variable,
                         writes=(upper, lower))
        out.append((upper, lower))
    return out

class MMAMode:
    DIRECT = 0
    TF32 = 1
    BF16 = 2
    I8 = 3

class MMAInstr:
    def headers(self):
        return []

    def __init__(self, m, n, k, b, d, name, mode):
        self.n = n
        self.m = m
        self.k = k
        self.b = b
        self.d = d
        self.name = name
        self.mode = mode

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

    def generate(self, writer, context, A, B, C, uses=()):
        with writer.Scope():
            if self.mode == MMAMode.I8:

                pass
            if self.mode == MMAMode.TF32:
                Atf32 = tfconvert(writer, A)
                Btf32 = tfconvert(writer, B)

                self.asmcall(writer, C, [a[0] for a in Atf32], [b[0] for b in Btf32], C, uses)
                self.asmcall(writer, C, [a[0] for a in Atf32], [b[1] for b in Btf32], C, uses)
                self.asmcall(writer, C, [a[1] for a in Atf32], [b[0] for b in Btf32], C, uses)
            else:
                self.asmcall(writer, C, A, B, C, uses)

INSTRS = [
    MMAInstr(16,8,4,1,Datatype.F32,'mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32', MMAMode.TF32), # SM_80
    MMAInstr(16,8,8,1,Datatype.F32,'mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32', MMAMode.TF32), # SM_80
    MMAInstr(8,8,4,1,Datatype.F64,'mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64', MMAMode.DIRECT), # SM_80
    MMAInstr(16,8,4,1,Datatype.F64,'mma.sync.aligned.m16n8k4.row.col.f64.f64.f64.f64', MMAMode.DIRECT), # SM_90
    MMAInstr(16,8,8,1,Datatype.F64,'mma.sync.aligned.m16n8k8.row.col.f64.f64.f64.f64', MMAMode.DIRECT), # SM_90
    MMAInstr(16,8,16,1,Datatype.F64,'mma.sync.aligned.m16n8k16.row.col.f64.f64.f64.f64', MMAMode.DIRECT), # SM_90
    MMAInstr(8,8,16,1,Datatype.F64,'mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32', MMAMode.I8), # SM_75
    MMAInstr(16,8,16,1,Datatype.F64,'mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32', MMAMode.I8), # SM_80
    MMAInstr(16,8,32,1,Datatype.F64,'mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32', MMAMode.I8), # SM_80
]

#: Whether the path is deployed, as opposed to whether it *can* emit for a
#: given shape -- that second question is `supports()`.  Two different facts,
#: so two names: `supports()` is a property of the shape, `ENABLED` is a
#: decision about the generator, and only the second is something to flip.
#:
#: Parked pending a run on real hardware.  `"+f"` versus `"=f"`/`"f"` on the
#: accumulator is a register-allocation difference no front end can see, and
#: the corpus is checked for well-formedness, not executed.
ENABLED = False


def supports(threads, dtype, sparse) -> bool:
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
    """
    return threads == 32 and dtype in (Datatype.F32, Datatype.F64) and not sparse


def shmsize(stages, dtype):
    atom = {
        Datatype.F32: INSTRS[1],
        Datatype.F64: INSTRS[2]
    }[dtype]

    threads = 32
    aregs = (atom.m * atom.k) // threads
    bregs = (atom.n * atom.k) // threads
    cregs = (atom.m * atom.n) // threads

    return 32 * max(aregs + bregs, cregs)

def scratch(dtype):
    """One set of staging tiles, sized off the same atom the emitter picks.

    Asked before generation, so it cannot depend on anything the body decides.
    """
    return shmsize(1, dtype)


def matmul(writer, ops, ctx):
    C, A, B = ops.C, ops.A, ops.B
    # Elements, and the loop below walks them in strides of `threads`.  The
    # accessors take slots, so `i // threads` is what reaches them.
    M = ops.lead_elements
    N, K, kx = ops.n, ops.k, ops.kx
    threads, dtype, sparse = ops.threads, ops.dtype, ops.sparse

    def threadrange(start, size):
        conditions = []
        if start > 0:
            conditions += [f'threadIdx.x >= {start}']
        if start + size < threads:
            conditions += [f'threadIdx.x < {start + size}']

        if len(conditions) > 0:
            return writer.If(' && '.join(conditions))
        else:
            return writer.AnonymousScope()

    if sparse:
        return False

    # for now.
    # TODO for later: split matrix into tiles
    # if too small for matrix tile (or with zero padded), use FMA instead
    # use different tile sizes if available

    atom = {
        Datatype.F32: INSTRS[1],
        Datatype.F64: INSTRS[2]
    }[dtype]

    mma = writer.varalloc()
    mmaT = writer.varalloc()

    Ashm = writer.varalloc()
    Bshm = writer.varalloc()



    # Staged fragments, by slot.  Dicts because the B index is a pair and the
    # extents are loop-derived; what matters is that these hold values now, not
    # C++ identifiers built out of a `varalloc` name.
    Areg = {}
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

    with writer.scratch_scope():
        Ashm = writer.alloc(atom.d, (aregs * threads,), MemSpace.SHARED,
                            hint='atile')
        # The B tile is written a row at a time and read a column at a time,
        # which no linear stride can serve without bank conflicts: 32 lanes
        # read 32 distinct elements spread over 60, and 240 bytes do not fit
        # in 128 of bank width.  Padding moves the collision, transposing
        # moves it to the store; permuting each row costs nothing and clears
        # both.  Measured over the emitted addresses: 2-way -> 1-way.
        Bshm = writer.alloc(atom.d, (bregs * threads,), MemSpace.SHARED,
                            hint='btile', swizzle=XorSwizzle(atom.k))
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

    for j in range(0, N, atom.n):
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
                                                         writer.rawexpr(f'(threadIdx.x - {trueK}) % {atom.k} + {jj * atom.k}', type_=INDEX, hint='a'))
                                    if trueSK != atom.k:
                                        with threadrange(0, atom.k - trueSK):
                                            for jj in range(0, atom.n):
                                                writer.store(Bshm, Breg[k // threads + 1, jj],
                                                                     writer.rawexpr(f'(threadIdx.x + {trueSK}) % {atom.k} + {jj * atom.k}', type_=INDEX, hint='a'))
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
                                            Bfrag[kkk + jj * kregs] = writer.load(Bshm, writer.rawexpr(f'(threadIdx.x % {ktile}) + (threadIdx.x / {ktile} + {jj * ntile}) * {atom.k} + {kkk * ktile}', type_=INDEX, hint='a'), hint='b')

                                    for kkk in range(0, min(atom.k, K - k - kk)):
                                        Areg[kkk] = A(writer, None, i // threads, k + kk + kkk)
                                    for kkk in range(min(atom.k, K - k - kk), atom.k):
                                        Areg[kkk] = writer.declare(ScalarType(atom.d),
                                                                   hint='as')

                                    for ii in range(0, min(threads, M - i), atom.m):
                                        with writer.AnonymousScope():
                                            writer.barrier(Uniformity.MULT)
                                            with threadrange(ii, atom.m):
                                                # for kkk in range(0, atom.k):
                                                #     writer(f'{shmptr}[{aoffs} + (threadIdx.x - {ii}) % {atom.m} + {kkk * atom.m}] = {Areg}_{kkk};')
                                                for kkk in range(0, atom.k, ktile):
                                                    # `store` already emits the
                                                    # reinterpret-cast form for
                                                    # a vector-typed value.
                                                    # Writing it by hand meant
                                                    # the shared write was
                                                    # opaque for the sake of a
                                                    # cast the verb performs.
                                                    quad = writer.pack(
                                                        ScalarType(atom.d, 4),
                                                        *(Areg[kkk + n] for n in range(4)),
                                                        hint='q')
                                                    writer.store(
                                                        Ashm, quad,
                                                        writer.rawexpr(f'((threadIdx.x - {ii}) % {atom.m}) * {ktile} + {kkk * atom.m}',
                                                                       type_=INDEX, hint='a'))
                                            writer.barrier(Uniformity.MULT)

                                            for kk in range(0, kregs):
                                                for iii in range(0, mregs):
                                                    #writer(f'{atom.d.ctype()} {Areg2}_{iii + kk * mregs} = {shmptr}[{aoffs} + (threadIdx.x / {ktile}) + (threadIdx.x % {ktile} + {kk * ktile}) * {atom.m} + {iii * mtile}];')
                                                    Afrag[iii + kk * mregs] = writer.load(Ashm, writer.rawexpr(f'threadIdx.x + {(iii + kk * mregs) * 32}', type_=INDEX, hint='a'), hint='a')

                                            atom.generate(writer, ctx, Afrag[:aregs], Bfrag[:bregs],
                                                          [Cvals[i][ii // atom.m] for i in range (cregs)])

                    # The epilogue's staging registers.  Assigned inside a
                    # thread guard and read outside it, so they are declared
                    # here and written through `assign` rather than being the
                    # result of the load: a value defined inside the guard
                    # would not be visible to the store that follows.
                    Cout = [writer.declare(ScalarType(atom.d), hint='c')
                            for _ in range(atom.n)]

                    for ii in range(0, threads, atom.m):
                        with writer.AnonymousScope():
                            for jj in range(0, nregs * 2):
                                for iii in range(0, mregs):
                                    writer.store(Cshm, Cvals[iii + mregs * jj][ii // atom.m],
                                        writer.rawexpr(f'threadIdx.x * 2 + {iii} + {jj * 64}', type_=INDEX, hint='a'))

                            writer.barrier(Uniformity.MULT)
                            with threadrange(ii, atom.m):
                                for jj in range(0, atom.n):
                                    _c = writer.load(Cshm, writer.rawexpr(f'(threadIdx.x % {atom.m}) * {atom.n} + {jj}', type_=INDEX, hint='a'), hint='data')
                                    writer.assign(Cout[jj], _c)
                            writer.barrier(Uniformity.MULT)

                    for jj in range(0, min(atom.n, N - j)):
                        C(writer, Cout[jj], i // threads, j + jj)

    return True
