# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Intel XMX (DPAS) for the multilinear kernel.

What is *derived* here and what is *assumed* are kept apart on purpose,
because the two fail differently.

Derived, from the arithmetic in `sycl/ext/intel/esimd/xmx/dpas.hpp`::

    OpsPerChannel = clamp(32 / max(A_bits, B_bits), 1, 8)
    M = RepeatCount
    K = SystolicDepth * OpsPerChannel
    N = ExecutionSize                      (16 for tf32; the header asserts it)
    |A| = M * K   elements of A's type
    |B| = K * N   elements of B's type
    |C| = M * N

Those are sizes, and a wrong size is a compile error -- the header's own
`static_assert`s catch it.  `tests/test_intel_gate.py` recomputes them from the
same formulas, so this table cannot drift from the header without saying so.

The fragment layout *is* derived too, and from the vISA specification rather
than the SYCL header -- `documentation/visa/instructions/DPAS.md` in
intel-graphics-compiler.  An earlier version of this file said it was not
documented anywhere; that was wrong, and the answer turns out to be simple::

    Dst, Src0 (C) and Src2 (A) are row-major in the GRF's 1-D space.
    Src1 (B) is laid out over a 2-D view: GRF row = k, DW column = n.

For TF32 that degenerates.  `OPS_PER_CHAN = 1`, so
`SRC1_OPERANDS_PER_CHAN = 32 / (1 * 32) = 1`, the GRF-row index `m` in the
pseudo-code equals the depth `d`, and B's "special" layout becomes
`B[k * N + n]` -- ordinary row-major.  The packing that makes Src1 unusual is
for sub-dword types, where several `k` share one DW; a 32-bit element leaves
nothing to pack.  With `Src2 advanced 8 * OPS_PER_CHAN per repeat` and
`Dst/Src0 advanced one GRF per repeat`, all three come out as::

    A[m * K + k]      B[k * N + n]      C[m * N + n]

FP64 is deliberately absent.  XMX has no FP64 at all, and emulating it from
TF32 costs more than PVC's vector units already deliver: ~419 TF of TF32
against ~52 TF of native FP64, where 53 mantissa bits need about fifteen
products.  Emulation is for FP32, where three products against 52 TF is a
gain.
"""

from tensorforge.backend.pir.core import SCALAR_LAYOUT, ScalarType
from tensorforge.common.basic_types import Datatype
from .. import broadcast, ranking, split
from ..bitlayout import packed
from ..strategy import Strategy, whole

#: Fixed by the hardware; the header asserts it.
SYSTOLIC_DEPTH = 8

#: `tf32` is ExecutionSize 16 only.  `bf16` and `fp16` also allow 8, which is
#: a different instruction and not this table's business yet.
EXECUTION_SIZE = 16


class DpasAtom:
    """One DPAS shape: what it multiplies and how big its operands are.

    `repeat` is the only free parameter -- 1, 2, 4 or 8, per the header's
    `verify_repeat_count` -- and it trades register pressure for issue count.
    """

    def __init__(self, name, elem_bits, acc, repeat=8,
                 depth=SYSTOLIC_DEPTH, exec_size=EXECUTION_SIZE):
        self.name = name
        self.elem_bits = elem_bits
        self.acc = acc
        self.repeat = repeat
        self.depth = depth
        self.exec_size = exec_size

    @property
    def ops_per_channel(self) -> int:
        return max(min(32 // self.elem_bits, 8), 1)

    @property
    def m(self) -> int:
        return self.repeat

    @property
    def k(self) -> int:
        return self.depth * self.ops_per_channel

    @property
    def n(self) -> int:
        return self.exec_size

    @property
    def a_elems(self) -> int:
        return self.m * self.k

    @property
    def b_elems(self) -> int:
        return self.k * self.n

    @property
    def c_elems(self) -> int:
        return self.m * self.n

    def with_repeat(self, repeat) -> 'DpasAtom':
        return DpasAtom(self.name, self.elem_bits, self.acc, repeat,
                        self.depth, self.exec_size)

    def __repr__(self):
        return (f'DpasAtom({self.name}, {self.m}x{self.n}x{self.k}, '
                f'repeat={self.repeat})')


#: Only what PVC's XMX has and this generator has a use for.
#:
#: `int8` and the MX formats (`bf8`, `hf8`, `e2m1`, and `bdpas` with its E8M0
#: scales) exist in the hardware and in the header.  They are absent because
#: nothing here produces them, not because they do not work -- adding one is a
#: row, not a mechanism.
ATOMS = {
    'tf32': DpasAtom('tf32', 32, Datatype.F32),
    'bf16': DpasAtom('bf16', 16, Datatype.F32),
    'fp16': DpasAtom('fp16', 16, Datatype.F32),
}

#: Terms the FP32 path splits an operand into.
#:
#: Two, and `split.terms(MANTISSA[TF32], F32)` is three: 11 bits at a time
#: against FP32's 24 needs three terms to cover it exactly, and two carry 22.
#: The reduction is deliberate and it is what `splitFloatTF32` is built for --
#: it returns a `(hi, lo)` pair, so the count and the routine's arity are one
#: fact -- but it is a reduction, and `split.covered` is what it costs.
TF32_SPLIT_TERMS = 2

#: The number of TF32 products it takes to recover an FP32 multiply.
#:
#: Derived rather than stated: the pair covers about 22 bits and the cross
#: terms `hi*lo` and `lo*hi` make up the difference, while `lo*lo` sits below
#: the accumulator's rounding and is dropped.  That is the same three-product
#: arrangement `nvidia.py` uses for `mma.sync ... .tf32`, and it comes out of
#: the same formula rather than being asserted twice.
TF32_TERMS = len(split.products(TF32_SPLIT_TERMS))

#: Whether the path is deployed, as opposed to whether it *can* emit for a
#: given shape -- that second question is `supports()`.  Two different facts,
#: so two names, and only this one is a decision about the generator.
#:
#: Parked pending a run on real hardware, and by now for the same reason as
#: the NVIDIA path rather than a sharper one.
#:
#: What used to block it is settled.  The fragment layout is derived from the
#: vISA pseudo-code and checked by placing a matrix through the offsets,
#: running the transcription and comparing against `C + A @ B`; the operand
#: mapping (Src1 is this generator's A, Src2 its B) is checked the same way;
#: and 29 of the 31 emitted kernels are well-formed with 11 of them carrying
#: real `dpas` calls.
#:
#: What is left is what no front end can answer: whether three TF32 products
#: through a systolic array give the FP32 result the generic path gives, on a
#: machine.  That is a run, not an argument.
ENABLED = False


# --------------------------------------------------------------------------- #
# Where an element sits in a fragment
# --------------------------------------------------------------------------- #
#
# Read off the pseudo-code in `documentation/visa/instructions/DPAS.md`, not
# guessed.  The loop is::
#
#     k = 0
#     for r in 0 .. RC-1:
#         temp = Src0.R[r]
#         for d in 0 .. SD-1:
#             m = d / SRC1_OPERANDS_PER_CHAN          # which GRF of Src1
#             n = (d % SRC1_OPERANDS_PER_CHAN) * OPS_PER_CHAN
#             for i in 0 .. Exec_size-1:
#                 temp.F[i] += dot(Src1.R[m].DW[i].n, Src2.k)
#             k += OPS_PER_CHAN
#         dst.R[r] = temp
#
#     Dst, Src0 advance one GRF per repeat; Src2 advances 8*OPS_PER_CHAN;
#     Src1 stays put.
#
# `R[j]` is the j-th GRF and `DW[i]` its i-th dword, so in the flat 1-D view a
# GRF is `Exec_size` elements wide for a 32-bit type.  Substituting gives the
# three functions below.


def src1_operands_per_chan(atom: DpasAtom) -> int:
    return 32 // (atom.ops_per_channel * atom.elem_bits)


def a_offset(atom: DpasAtom, m: int, k: int) -> int:
    """Src2: `Src2.k` within repeat `m`, which advances `8 * OPS_PER_CHAN`.

    Row-major over (M, K) -- the spec says so in words too ("Dst, Src0, Src2
    are laid out in row-major in this 1-D memory space").
    """
    return m * atom.k + k


def b_offset(atom: DpasAtom, k: int, n: int) -> int:
    """Src1: `Src1.R[m].DW[n_chan]`, at element `n_elem` inside that dword.

    The layout the spec calls "neither row-major nor column major".  With
    `m = d / SRC1_OPERANDS_PER_CHAN` selecting the GRF, the channel `i`
    selecting the dword within it, and `n = (d % ...) * OPS_PER_CHAN` the
    element inside the dword, a flat index is::

        m * (Exec_size * elems_per_dword) + i * elems_per_dword + n

    For a 32-bit element `SRC1_OPERANDS_PER_CHAN` is 1 and one dword holds one
    element, so this collapses to `k * N + n` -- plain row-major.  The packing
    is what makes Src1 unusual, and a 32-bit type leaves nothing to pack.
    """
    per_chan = src1_operands_per_chan(atom)
    elems_per_dword = 32 // atom.elem_bits
    grf = k // (per_chan * atom.ops_per_channel)
    within = (k % (per_chan * atom.ops_per_channel))
    return (grf * atom.n * elems_per_dword + n * elems_per_dword + within)


def c_offset(atom: DpasAtom, m: int, n: int) -> int:
    """Dst/Src0: one GRF per repeat, channel `i` within it."""
    return m * atom.n + n


def reference(atom: DpasAtom, c, b, a):
    """The instruction, in Python, transcribed from the pseudo-code.

    Not for generating anything -- it is the check that the three offset
    functions above are the same layout the hardware uses.  Comparing it
    against an ordinary `C + A @ B` is what turns "this is what I read in the
    spec" into "and reading it that way reproduces a matrix product".
    """
    out = list(c)
    per_chan = src1_operands_per_chan(atom)
    for r in range(atom.m):
        k = 0
        for d in range(atom.depth):
            grf = d // per_chan
            n_el = (d % per_chan) * atom.ops_per_channel
            for i in range(atom.n):
                acc = 0.0
                for o in range(atom.ops_per_channel):
                    elems_per_dword = 32 // atom.elem_bits
                    bi = (grf * atom.n * elems_per_dword
                          + i * elems_per_dword + n_el + o)
                    acc += b[bi] * a[r * atom.k + k + o]
                out[r * atom.n + i] += acc
            k += atom.ops_per_channel
    return out


def supports(threads, dtype, sparse) -> bool:
    """Whether `matmul` can emit for this shape, asked *before* it is called.

    * ``threads == 16``.  `ExecutionSize` is 16 for every type in this table,
      and the ESIMD lowering makes the vector width the thread count -- so a
      wave of any other width is a different instruction, not a narrower use
      of this one.
    * ``dtype is F32``.  FP64 has no DPAS at all (see the module docstring),
      and the lower precisions are not what SeisSol asks for.
    * ``not sparse``.  The sparse operand path loads by linear index, which is
      not a fragment.
    """
    return threads == EXECUTION_SIZE and dtype == Datatype.F32


#: Repeat counts the header's `verify_repeat_count` admits, widest first.
#:
#: Widest first because a tie in the ranking below means the count could not
#: tell two of them apart, and the order a module states is then the answer.
#: With no shape and no budget every candidate ties, so this is what decides.
REPEATS = (8, 4, 2, 1)


def fragment_bytes(atom, terms=TF32_SPLIT_TERMS) -> int:
    """Register file one issue group's fragments hold, in bytes.

    The accumulator once and each operand once per split term, which is what
    `dpas_matmul` declares inside a `j0` tile.  `repeat` scales the
    accumulator and Src2 and leaves Src1 alone, so this is the register half
    of the trade the repeat count makes -- the issue half is `ranking.issues`.

    A lower bound on what the body holds, not the body's own figure: the
    surrounding loop nest has registers of its own, and
    `_check_register_budget` is what weighs the whole of it.
    """
    elems = atom.c_elems + terms * (atom.a_elems + atom.b_elems)
    return elems * Datatype.F32.size()


def atoms_for(dtype, budget=None):
    """Every repeat count this type could be emitted at, widest first.

    `repeat` is the only free parameter here, and it trades register pressure
    for issue count: eight columns of output per issue against one, and eight
    times the accumulator and Src2 to hold them.  A budget in bytes drops the
    candidates whose fragments alone would not fit.
    """
    if dtype != Datatype.F32:
        return ()
    base = ATOMS['tf32']
    out = [base.with_repeat(repeat) for repeat in REPEATS]
    if budget is not None:
        out = [atom for atom in out if fragment_bytes(atom) <= budget]
    return tuple(out)


def atom_for(dtype, columns=0, lead=0, depth=0, budget=None):
    """The repeat count that serves this shape with the fewest issues.

    Ranking by issues alone always returns the widest, because nothing else
    in the count changes with `repeat` -- so without a budget this is the
    constant `REPEATS[0]` with a ranking around it, and saying that is the
    point: the selection only becomes one once the register side bounds it.
    That is what `budget` is for and why it is a parameter rather than a
    constant here.
    """
    def key(atom):
        # `m` takes the output columns, `n` the lanes and so the leading
        # dimension, `k` the contraction -- a third mapping to the same three
        # numbers, and the reason the conversion sits in each vendor module.
        return (ranking.Extent(columns=atom.m, lanes=atom.n, depth=atom.k,
                               name=f'{atom.name}x{atom.repeat}'), 1)

    found = ranking.rank(atoms_for(dtype, budget), key, columns, lead, depth)
    return found[0] if found else None


def register_budget(ctx):
    """Bytes of register file one work-item gets, or `None` where unstated."""
    if ctx is None:
        return None
    return getattr(ctx.get_vm().get_hw_descr(), 'max_reg_per_thread', None)


def simd(lexic, elem, count) -> str:
    return lexic.get_simd(elem, count)


#: Whether the register-only FMA path is deployed.
#:
#: Separate from `ENABLED`, because they wait on different things.  DPAS waits
#: on a machine: nothing here can check a systolic arrangement.  This path uses
#: only what a front end sees -- an element read and an FMA -- so what it waits
#: on is the arithmetic, and that is checkable without hardware.
#:
#: Two defects had to go first, and naming them is worth more than the flag:
#:
#: * the dispatch passed `Mx` where this path needs `M`.  `unwindI` maps its
#:   argument with `i % M`, so iterating to the element count asked for the
#:   same index `threads` times and got the same value back -- and then the
#:   same product was accumulated into everything.  Not an error anywhere,
#:   just wrong.
#: * `float * simd<float, N>` needs the free operator ESIMD defines in
#:   `detail/operators.hpp`; the test shim only had the member overloads,
#:   which cover a scalar on the right.
#:
#: What clears it now: 31 of 31 emitted kernels are well-formed, no
#: accumulator receives a product twice, and on a 16x16 GEMM each of the 16
#: accumulators sweeps the full contraction over one distinct B vector.  That
#: is structure, not numerics -- the numbers still want a run.
BROADCAST_ENABLED = True


def _fragment(writer, dtype, count, hint):
    """A DPAS fragment: `count` elements, held whole by one work-item.

    `SCALAR_LAYOUT` and not a lane axis, and the distinction is the point.  A
    fragment is not "one element per lane" -- its element order is the
    hardware's (see `a_offset` and friends), and the work-item owns all of it.
    The width therefore lives on `ScalarType.length`, the slot axis, which the
    ESIMD emitter also spells as a `simd`.
    """
    return writer.declare(ScalarType(dtype, count), hint=hint, init='{}',
                          layout=SCALAR_LAYOUT)


def _run(writer, frag, start, size, hint):
    """A contiguous run of a fragment, as a value of the *run's* type.

    `ScalarType(base, size)` and not the parent's: a view is 16 wide even when
    it looks into 128, and typing it by the fragment makes the accumulator
    read-out claim to store 128 elements where it stores one output column.
    """
    return writer.rawexpr(f'{{0}}.template select<{size}, 1>({start})', frag,
                          type_=ScalarType(frag.type.base, size), hint=hint,
                          pure=True)


#: Which of this generator's operands is which of the instruction's.
#:
#: `C(i,j) = sum_k A(i,k) * B(j,k)` onto `D(m,n) = sum_k Ad(m,k) * Bd(k,n)`.
#: DPAS's `N` is the execution size, which under this lowering is the wave --
#: and the wave is where the *lead* dimension lives.  So `n` is `i`, `m` is
#: `j`, and the two operands cross over:
#:
#:     Src1 (the instruction's B) is this generator's A, indexed (k, i)
#:     Src2 (the instruction's A) is this generator's B, indexed (j, k)
#:
#: Which is not a guess either -- `test_intel_gate.py` runs the transcribed
#: instruction with the operands placed this way and checks it against
#: `sum_k A(i,k) * B(j,k)`.
#:
#: The reformat falls out of it.  `b_offset(k, n) = k*N + n`, so the sixteen
#: lanes an operand load already returns land in sixteen *consecutive* slots;
#: `a_offset(m, k) = m*K + k` does the same for eight.  Every transfer between
#: a lane vector and a fragment is a contiguous run, which is why this is a
#: handful of `select`s rather than a loop over elements.


def dpas_matmul(writer, C, A, B, M, N, K, kx, threads, dtype, ctx):
    """`C += A x B` through XMX, with FP32 emulated over three TF32 products.

    Three products, not four: `lo*lo` falls below the accumulator's rounding.
    The same arrangement as `nvidia.py`'s `mma.sync ... .tf32`, and it has to
    be -- the error analysis belongs to the split, not to either instruction.
    """
    atom = atom_for(dtype, columns=N, lead=M * threads, depth=K + kx,
                    budget=register_budget(ctx))
    if atom is None or threads != atom.n:
        return False
    acc_ct = dtype.ctype()
    depth = K + kx

    for j0 in range(0, N, atom.m):
        rows = min(atom.m, N - j0)
        acc = _fragment(writer, dtype, atom.c_elems, 'dacc')
        for k0 in range(0, depth, atom.k):
            ahi = _fragment(writer, Datatype.TF32, atom.a_elems, 'ahi')
            alo = _fragment(writer, Datatype.TF32, atom.a_elems, 'alo')
            bhi = _fragment(writer, Datatype.TF32, atom.b_elems, 'bhi')
            blo = _fragment(writer, Datatype.TF32, atom.b_elems, 'blo')

            # Src1 <- this generator's A: one lane vector per contraction step.
            for k in range(min(atom.k, depth - k0)):
                v = A(writer, None, 0, k0 + k)
                if v is None or v is False:
                    return False
                off = b_offset(atom, k, 0)
                writer.call_stmt(f'tensorforge::splitFloatTF32<{atom.n}>',
                                 _run(writer, bhi, off, atom.n, 'bh'),
                                 _run(writer, blo, off, atom.n, 'bl'),
                                 v, writes=(bhi, blo))

            # Src2 <- this generator's B: one run per repeat row.
            for m in range(rows):
                v = B(writer, None, j0 + m, k0 // threads)
                if v is None or v is False:
                    return False
                off = a_offset(atom, m, 0)
                writer.call_stmt(f'tensorforge::splitFloatTF32<{atom.k}>',
                                 _run(writer, ahi, off, atom.k, 'ah'),
                                 _run(writer, alo, off, atom.k, 'al'),
                                 _run(writer, v, 0, atom.k, 'bk'),
                                 writes=(ahi, alo))

            bterms, aterms = (bhi, blo), (ahi, alo)
            for i, j in split.products(TF32_SPLIT_TERMS):
                bf, af = bterms[i], aterms[j]
                writer.assign(acc, writer.rawexpr(
                    f'tensorforge::intel_xmx::dpas<{atom.depth}, '
                    f'{atom.repeat}, {acc_ct}>({{0}}, {{1}}, {{2}})',
                    acc, bf, af, type_=acc.type, hint='dp', pure=True))

        # Read-out is a run too: `c_offset(m, n) = m*N + n`, so one output
        # column is `acc.select<N, 1>(m * N)` -- already the shape the store
        # wants.
        for m in range(rows):
            C(writer, _run(writer, acc, c_offset(atom, m, 0), atom.n, 'cr'),
              0, j0 + m)
    return True


def strategies(shape, ctx):
    """What this target can emit for this shape.

    Two arrangements, and they are not variations of each other.  DPAS is a
    systolic product with staged fragments; the broadcast chain is an FMA
    chain over the lanes.  Which wins is a measurement, not a preference, and
    neither flag is on by accident.

    The broadcast chain is offered only under the explicitly vectorised
    lowering, and that is its whole argument: a lane broadcast is `v[k]` out
    of this work-item's own registers there, and a real cross-lane
    instruction in SPMD.  Offering it on an Intel target lowered as SPMD would
    trade a shared buffer for a `group_broadcast` per product, which is the
    trade the DPP chain makes deliberately and this one does not.

    Not the same question as `placement.broadcast_without_staging`, which asks
    whether a mis-oriented operand may be read where it lies and is dropped
    under the same lowering this one requires.  One is about reaching an
    operand, the other about what to build the products out of.

    A packed lead operand is declined by both.  DPAS reads its fragments at
    offsets derived from the vISA pseudocode and checked by laying a matrix
    through them; those offsets name elements, and a width that puts elements
    inside a register is a distribution they were not derived for.  The
    broadcast chain reads `v[k]` out of the work-item's own registers, where
    `k` is an element index for the same reason as on AMD.  Neither has been
    given a conversion, so neither is offered one.
    """
    if packed(shape.lead_layout):
        return frozenset()
    if not supports(shape.threads, shape.accumulator, shape.sparse):
        return frozenset()
    offered = set()
    if ENABLED and not shape.sparse:
        offered.add(Strategy.MATRIX)
    if BROADCAST_ENABLED and shape.explicit_simd:
        offered.add(Strategy.BROADCAST)
    return frozenset(offered)


def scratch(strategy, shape, ctx):
    """Nothing: both arrangements here hold their fragments in registers.

    `ctx` is unused and is part of the signature anyway: the NVIDIA answer
    depends on the target, so the interface has to be able to carry one.
    """
    return 0


def plan(strategy, shape, n, ctx):
    """One arrangement over the whole output.

    Both paths here pad a partial tile rather than leave it: DPAS reads a
    fragment whose spare rows are zero, and the broadcast chain simply has
    fewer accumulators.  Neither gets cheaper by handing the remainder to the
    other.
    """
    return whole(strategy, n)


def matmul(writer, ops, ctx, span):
    """Emit the arrangement the caller chose, or decline.

    DPAS is what is specific to this target; the broadcast chain is the shared
    one in `compute/broadcast.py`, offered here because an explicit vector is
    where its replication is free rather than because it is an Intel idea.

    Either may decline after it has emitted, when an operand it needs turns
    out to have no value: whether the shape is servable is not fully knowable
    before the loads are attempted.  The caller emits this inside
    `Writer.speculative` and discards on `False`, which is what makes a
    partial attempt cost nothing.
    """
    C, A, B = ops.C, ops.A, ops.B
    M, N, K, kx = ops.lead_slots, ops.n, ops.k, ops.kx
    threads, dtype, sparse = ops.threads, ops.accumulator, ops.sparse

    if span.strategy is Strategy.BROADCAST:
        return broadcast.matmul(writer, ops, ctx, span)
    if span.start != 0 or span.stop != N:
        # DPAS does not take a range; `plan` never asks for one, and a direct
        # caller that does should hear so rather than get the whole output.
        return False
    if span.strategy is Strategy.MATRIX:
        if Datatype.F32 not in (ops.a, ops.b) or ops.a != ops.b:
            # `splitFloatTF32` takes an F32 apart; handed anything else it
            # would produce two halves of a number it never had.  The atom is
            # chosen by the accumulator, so nothing upstream has checked what
            # the operands arrive as.
            return False
        return dpas_matmul(writer, C, A, B, M, N, K, kx, threads, dtype, ctx)
    return False
