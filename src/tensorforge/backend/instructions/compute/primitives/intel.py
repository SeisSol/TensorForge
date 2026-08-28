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

Not derived: *which element of the fragment a given lane holds*.  That is a
hardware register assignment, it is in no header, and its failure mode is the
bad one -- a correctly typed vector holding the wrong elements compiles, runs,
and is wrong.  `amd/codegen.py` reaches the same conclusion about the MFMA
accumulator and writes `None` rather than a plausible layout; the same applies
with more force to a systolic array.  Hence `ENABLED`.

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
#: Parked pending a run on real hardware, for a sharper reason than the NVIDIA
#: path's: the DPAS fragment layout -- which lane holds which element of A, B
#: and C -- is in no header and cannot be checked by a front end.  A wrong one
#: produces a kernel that compiles and computes wrong numbers, which nothing in
#: this repository would catch.
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


def matmul(writer, C, A, B, M, N, K, kx, threads, dtype, sparse, ctx):
    """Not emitted.

    `ENABLED` is off, and `_is_matmul` asks before calling -- so this is
    unreachable rather than merely unused.  It stays as the named place for
    the emission to go, and returning `False` means the caller falls through
    to the generic path, which is the correct behaviour for a path that is not
    deployed.

    What it will need, when there is a machine to check it against:

    * A and B staged into `simd<TF32, M*K>` and `simd<TF32, K*N>` in the
      fragment order the hardware expects.  The operand callbacks hand over
      one element at a time and know nothing about that order, so the staging
      is where the unknown sits -- see the module docstring.
    * three `xmx::dpas` per product, over `(hi,hi)`, `(hi,lo)`, `(lo,hi)`;
      see `TF32_TERMS`.
    * the accumulator's layout left untracked, exactly as the MFMA path leaves
      its own: `None` means unknown, and every check treats an unknown layout
      as distinct from every other, so a pass stays conservative.  A wrong
      layout is not conservative.
    """
    return False
