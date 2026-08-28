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

from tensorforge.common.basic_types import Datatype

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

#: The number of TF32 products it takes to recover an FP32 multiply.
#:
#: TF32 keeps 11 mantissa bits against FP32's 24, so one split leaves the pair
#: `(hi, lo)` covering about 22 -- and the cross terms `hi*lo` and `lo*hi` make
#: up the difference.  `lo*lo` falls below the accumulator's rounding and is
#: dropped, which is the same three-term arrangement `nvidia.py` uses for
#: `mma.sync ... .tf32`.
TF32_TERMS = 3

#: Whether the path is deployed, as opposed to whether it *can* emit for a
#: given shape -- that second question is `supports()`.  Two different facts,
#: so two names, and only this one is a decision about the generator.
#:
#: Parked pending a run on real hardware -- but for the same reason as the
#: NVIDIA path now, not a sharper one.  The fragment layout turned out to be
#: specified (see the module docstring); what is left unverified is the same
#: class of thing `nvidia.py` names: an arrangement a front end cannot check,
#: on an instruction nothing in this repository executes.
ENABLED = False


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
    return threads == EXECUTION_SIZE and dtype == Datatype.F32 and not sparse


def atom_for(dtype):
    """The atom an operator of this type is emulated with, or None."""
    return ATOMS['tf32'] if dtype == Datatype.F32 else None


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


def broadcast_matmul(writer, C, A, B, M, N, K, kx, threads, dtype, ctx):
    """`C[i][j] += B[k][j] * A[i][k]`, entirely in registers.

    The same shape as the AMD DPP path in `amd/codegen.py`, and preferable
    here for a reason that is specific to this model: the contraction index of
    B lives in the *lanes*, so every product needs one of B's lanes broadcast
    to all of them.  On AMD that is a real cross-lane instruction and the
    reason `relayout.py` has a table of them; under an explicit vector it is
    `v[k]`, an element read out of this work-item's own registers.

    A free broadcast is what makes the register-only arrangement beat staging
    operands through shared memory, which is what the NVIDIA path has to do --
    there is no barrier, no arena, and no round trip.

    A is per-lane in the output index `i`; B is per-lane in the contraction
    index `k`.  Two different meanings of "lane" for the two operands, which
    is exactly what the broadcast reconciles.
    """
    # `None` asks the loader for the value rather than for a name to fill in:
    # these are operands, and an operand whose definition the IR cannot see is
    # invisible to every pass that reasons about ordering or reuse.
    a = {}
    out_layout = None
    for i in range(M):
        for k in range(K + kx):
            v = A(writer, None, i, k)
            if v is not None and v is not False:
                a[(i, k)] = v
                # Taken from the operand rather than constructed: A is indexed
                # by the same output index the accumulator is, so whatever
                # distribution its loads came out with is the one to hold.
                if out_layout is None:
                    out_layout = v.layout

    for j in range(N):
        # The accumulator is spread over the lanes exactly like the output it
        # holds -- one element of the lead dimension per lane.  Declared with
        # that layout rather than left untracked, because untracked is not a
        # conservative default here: an explicitly vectorised declaration
        # cannot be written without it.
        acc = [writer.declare(hint='acc', layout=out_layout) for _ in range(M)]
        for k0 in range(0, K + kx, threads):
            vb = B(writer, None, j, k0 // threads)
            if vb is None or vb is False:
                continue
            for lane in range(min(threads, K + kx - k0)):
                # One of B's lanes, replicated -- free here, a shuffle on AMD.
                bk = writer.lane_broadcast(vb, lane, threads)
                for i in range(M):
                    operand = a.get((i, k0 + lane))
                    if operand is None:
                        continue
                    writer.accumulate(
                        acc[i], writer.op('mul', operand.type, bk, operand,
                                          hint='p'))
        for i in range(M):
            C(writer, acc[i], i, j)
    return True


def matmul(writer, C, A, B, M, N, K, kx, threads, dtype, sparse, ctx):
    """Neither path is deployed; `_is_matmul` asks before calling.

    Returning `False` means the caller falls through to the generic path,
    which is the correct behaviour for a path that is not on.

    What DPAS will need, when there is a machine to check it against:

    * A and B staged row-major into `simd<TF32, M*K>` and `simd<TF32, K*N>`
      -- see the module docstring; for TF32 the vISA layout is plain
      row-major for all three operands.
    * three `xmx::dpas` per product, over `(hi,hi)`, `(hi,lo)`, `(lo,hi)`;
      see `TF32_TERMS`.
    * the accumulator's layout left untracked, exactly as the MFMA path leaves
      its own: `None` means unknown, and every check treats an unknown layout
      as distinct from every other, so a pass stays conservative.  A wrong
      layout is not conservative.
    """
    if BROADCAST_ENABLED and not sparse:
        return broadcast_matmul(writer, C, A, B, M, N, K, kx, threads, dtype,
                                ctx)
    return False
