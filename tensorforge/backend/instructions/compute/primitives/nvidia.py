from tensorforge.common.basic_types import Datatype
from tensorforge.backend.writer import Writer

def tfconvert(writer: Writer, variables):
    for variable in variables:
        writer(f'uint32_t {variable}u, {variable}l;')
        writer(f'tensorforge::splitFloatTF32({variable}u, {variable}l, {variable});')
    return [(f'{v}u', f'{v}l') for v in variables]

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

    def asmcall(self, writer, D, A, B, C):
        """The `mma.sync` itself.

        `D` and `C` are the same accumulator at every call site -- the
        instruction reads it and writes it back.  Listing it as `"=f"` under
        outputs and again as `"f"` under inputs states two *unrelated*
        operands that happen to name one C++ lvalue, and nothing then requires
        the compiler to give them the same register: it is free to read the
        accumulator into one and write the result into another, dropping the
        accumulation.  `"+f"` is the constraint that says read-and-write, and
        then the operand is listed once.

        Numbering follows from that.  PTX numbers outputs and inputs in one
        sequence, so folding C into D shifts A and B down by `len(C)` --
        writing the offsets from the *emitted* groups rather than from the
        argument lists is what keeps the two in step.
        """
        typeid = "f" if self.d == Datatype.F32 else "d"
        typeidx = "r" if self.d == Datatype.F32 else "d"

        inout = D if D is C or list(D) == list(C) else None

        arggrp = lambda x, b: "{" + ','.join(f"%{i + b}" for i,_ in enumerate(x)) + "}"
        arggrp2 = lambda x, o: ','.join(f'"{o}"({v})' for v in x)

        if inout is not None:
            # one operand, read-write: D and C name the same registers
            outs = arggrp2(inout, f"+{typeid}")
            ins = f'{arggrp2(A, typeidx)}, {arggrp2(B, typeidx)}'
            dgrp = arggrp(inout, 0)
            agrp = arggrp(A, len(inout))
            bgrp = arggrp(B, len(inout) + len(A))
            cgrp = dgrp
        else:
            outs = arggrp2(D, f"={typeid}")
            ins = (f'{arggrp2(A, typeidx)}, {arggrp2(B, typeidx)}, '
                   f'{arggrp2(C, typeid)}')
            dgrp = arggrp(D, 0)
            agrp = arggrp(A, len(D))
            bgrp = arggrp(B, len(D) + len(A))
            cgrp = arggrp(C, len(D) + len(A) + len(B))

        writer(f"""asm("{self.name} "
"{dgrp}, {agrp}, {bgrp}, {cgrp};"
: {outs}
: {ins}
);""")

    def epilogue(self):
        pass

    def generate(self, writer, context, A, B, C):
        with writer.Scope():
            if self.mode == MMAMode.I8:

                pass
            if self.mode == MMAMode.TF32:
                Atf32 = tfconvert(writer, A)
                Btf32 = tfconvert(writer, B)

                self.asmcall(writer, C, [a[0] for a in Atf32], [b[0] for b in Btf32], C)
                self.asmcall(writer, C, [a[0] for a in Atf32], [b[1] for b in Btf32], C)
                self.asmcall(writer, C, [a[1] for a in Atf32], [b[0] for b in Btf32], C)
            else:
                self.asmcall(writer, C, A, B, C)

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

#: The atom `matmul` and `shmsize` are both hard-wired to.  Named once so the
#: capability predicate, the shared-memory reservation and the emitter cannot
#: drift apart -- they had three separate copies of `INSTRS[1]` before.
ATOM = INSTRS[1]


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
    * ``dtype == ATOM.d``.  `ATOM` is a TF32 instruction.  Nothing downstream
      compares the operand type against it, so an `f64` case would emit
      `mma.sync...f32.tf32.tf32.f32` over doubles and be quietly wrong rather
      than loudly unsupported.
    * ``not sparse``.  `matmul` already declines these by returning `False`,
      but `temp_shmem` reserves shared memory off the same predicate; if the
      two disagree the reservation is made for a kernel that never uses it.
    """
    return threads == 32 and dtype == ATOM.d and not sparse


def shmsize(stages):
    atom = ATOM
    threads = 32
    aregs = (atom.m * atom.k) // threads
    bregs = (atom.n * atom.k) // threads
    cregs = (atom.m * atom.n) // threads

    return 32 * max(aregs + bregs, cregs)

def matmul(writer, C, A, B, M, N, K, kx, threads, dtype, sparse, ctx, shmptr, shmsize):
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

    atom = ATOM

    mma = writer.varalloc()
    mmaT = writer.varalloc()

    Ashm = writer.varalloc()
    Bshm = writer.varalloc()

    Areg2 = writer.varalloc()
    Breg2 = writer.varalloc()

    Areg = writer.varalloc()
    Breg = writer.varalloc()
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

    assert 32 * max(aregs + bregs, cregs) <= shmsize

    aoffs = 0
    boffs = aregs * 32
    coffs = 0

    for j in range(0, N, atom.n):
        with writer.AnonymousScope():
            for k in range(0, K + kx, threads):
                for jj in range(0, min(atom.n, N - j)):
                    B(writer, f'{Breg}_{k//threads}_{jj}', j + jj, k // threads)
                for jj in range(min(atom.n, N - j), atom.n):
                    writer(f'{atom.d.ctype()} {Breg}_{k//threads}_{jj}{"{}"};')
            for i in range(0, M, threads):
                with writer.AnonymousScope():
                    writer(f'{atom.d.ctype()} {Creg}[{cregs}][{threads // atom.m}]{"{}"};')
                    for k in range(0, K, threads):
                        with writer.AnonymousScope():
                            for kk in range(0, min(threads, K - k), atom.k):
                                with writer.AnonymousScope():
                                    writer('__syncwarp();')
                                    trueK = kk + kx
                                    trueSK = min(atom.k, threads - trueK)
                                    with threadrange(trueK, trueSK):
                                        for jj in range(0, atom.n):
                                            writer(f'{shmptr}[{boffs} + (threadIdx.x - {trueK}) % {atom.k} + {jj * atom.k}] = {Breg}_{k//threads}_{jj};')
                                    if trueSK != atom.k:
                                        with threadrange(0, atom.k - trueSK):
                                            for jj in range(0, atom.n):
                                                writer(f'{shmptr}[{boffs} + (threadIdx.x + {trueSK}) % {atom.k} + {jj * atom.k}] = {Breg}_{k//threads+1}_{jj};')
                                    writer('__syncwarp();')

                                    for jj in range(0, nregs):
                                        for kkk in range(0, kregs):
                                            writer(f'{atom.d.ctype()} {Breg2}_{kkk + jj * kregs} = {shmptr}[{boffs} + (threadIdx.x % {ktile}) + (threadIdx.x / {ktile} + {jj * ntile}) * {atom.k} + {kkk * ktile}];')

                                    for kkk in range(0, min(atom.k, K - k - kk)):
                                        A(writer, f'{Areg}_{kkk}', i // threads, k + kk + kkk)
                                    for kkk in range(min(atom.k, K - k - kk), atom.k):
                                        writer(f'{atom.d.ctype()} {Areg}_{kkk}{"{}"};')

                                    for ii in range(0, min(threads, M - i), atom.m):
                                        with writer.AnonymousScope():
                                            writer('__syncwarp();')
                                            with threadrange(ii, atom.m):
                                                # for kkk in range(0, atom.k):
                                                #     writer(f'{shmptr}[{aoffs} + (threadIdx.x - {ii}) % {atom.m} + {kkk * atom.m}] = {Areg}_{kkk};')
                                                for kkk in range(0, atom.k, ktile):
                                                    writer(f'*(float4*)&{shmptr}[{aoffs} + ((threadIdx.x - {ii}) % {atom.m}) * {ktile} + {kkk * atom.m}] = make_float4({Areg}_{kkk}, {Areg}_{kkk+1}, {Areg}_{kkk+2}, {Areg}_{kkk+3});')
                                            writer('__syncwarp();')

                                            for kk in range(0, kregs):
                                                for iii in range(0, mregs):
                                                    #writer(f'{atom.d.ctype()} {Areg2}_{iii + kk * mregs} = {shmptr}[{aoffs} + (threadIdx.x / {ktile}) + (threadIdx.x % {ktile} + {kk * ktile}) * {atom.m} + {iii * mtile}];')
                                                    writer(f'{atom.d.ctype()} {Areg2}_{iii + kk * mregs} = {shmptr}[{aoffs} + threadIdx.x + {(iii + kk * mregs) * 32}];')

                                            atom.generate(writer, ctx, [f'{Areg2}_{i}' for i in range (aregs)], [f'{Breg2}_{i}' for i in range (bregs)], [f'{Creg}[{i}][{ii // atom.m}]' for i in range (cregs)])

                    for jj in range(0, atom.n):
                        writer(f'{atom.d.ctype()} {Creg}_{jj}{"{}"};')

                    for ii in range(0, threads, atom.m):
                        with writer.AnonymousScope():
                            for jj in range(0, nregs * 2):
                                for iii in range(0, mregs):
                                    writer(f'{shmptr}[{coffs} + threadIdx.x * 2 + {iii} + {jj * 64}] = {Creg}[{iii + mregs * jj}][{ii // atom.m}];')

                            writer('__syncwarp();')
                            with threadrange(ii, atom.m):
                                for jj in range(0, atom.n):
                                    writer(f'{Creg}_{jj} = {shmptr}[{coffs} + (threadIdx.x % {atom.m}) * {atom.n} + {jj}];')
                            writer('__syncwarp();')

                    for jj in range(0, min(atom.n, N - j)):
                        C(writer, f'{Creg}_{jj}', i // threads, j + jj)

    return True
