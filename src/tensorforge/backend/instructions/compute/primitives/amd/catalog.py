# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""The matrix instruction catalogue.

What an instruction *is* --- its shape, how many blocks it computes, what each
lane holds of A, B and D --- is a fact about the target; which one to use is a
policy.  They used to be interleaved, and the entries covered exactly one
family: square, ``K == 1``, F32, A fed by a cross-lane transpose.  Every
direction this has to grow breaks a different one of those assumptions.

======================================  ==================================
want                                    breaks
======================================  ==================================
``mfma_f64_4x4x4f64``                   4 blocks, scalar accumulator, K=4
``mfma_f64_16x16x4f64``                 1 block, v4f64 accumulator
``mfma_f32_4x4x4bf16_1k``               operand is a k-vector
``wmma_f32_16x16x16_bf16_w32``          wave 32, operand duplicated
``wmma_f32_16x16x16_bf16_w32_gfx12``    wave 32, operand not duplicated
``wmma_f32_16x16x4_f32``                different argument order
======================================  ==================================

So the descriptor carries the shape, the block count, the per-lane widths, the
wave it is defined against, the LLVM feature that gates it, and the argument
order of the call.  Nothing here says which one to pick.

**Every row is a claim about hardware**, and this package has already been
bitten twice by claims stated in the wrong role.  So none of the rows is
written from memory: `tools/amd_matrix_table.py` extracts the shapes, per-lane
widths and feature gates from LLVM's own ``BuiltinsAMDGPU.td``, ``AMDGPU.td``
and ``GCNProcessors.td`` into ``tests/data/amd_matrix_builtins.json``, and
`tests/test_amd_catalog.py` fails if a row here disagrees with it.

The one thing the table cannot supply is the split between `blocks` and
`replication`.  LLVM states the product --- ``per_lane * wave`` --- and both
factors reproduce it.  `blocks` is declared and `replication` derived, which
puts the undetermined half in one integer per row instead of spreading it.
That integer is where the RDNA 3 fragments differ from every other row: their
A and B hold a 16x16x16 operand *twice* per wave32 (four times per wave64),
and the same shape on gfx12 holds it once.  Deriving `replication` makes that
show up as a number rather than as a comment nobody checks.

Entries are restricted to instructions with an F32 or F64 accumulator.  The
integer, FP8/FP6/FP4 and F16/BF16-accumulator instructions exist and are in
the vendored table; they are absent here because no operator this generator
emits can consume them, and the check against LLVM treats the omission as
`NOT_MODELLED` rather than passing over it.
"""

from dataclasses import dataclass
from enum import Enum
from math import ceil
from typing import Optional, Tuple

from tensorforge.common.basic_types import Datatype
from .features import has_feature, wave_size


class Call(Enum):
    """The argument order of the intrinsic.

    Three shapes cover every entry, and the third is why this is a field
    rather than something the emitter assumes: gfx125x *interleaves* its
    modifiers with the operands, so counting arguments places the accumulator
    in the wrong slot.
    """

    #: ``d = f(a, b, c, cbsz, abid, blgp)``.  LLVM gives these no `ArgNames`;
    #: the order is the one every MFMA has had since gfx908.  `cbsz`/`abid`
    #: broadcast one block's A operand to a group of blocks and `blgp`
    #: permutes B, which is why `broadcast` is true only here.
    MFMA = 'a, b, c, cbsz, abid, blgp'
    #: ``d = f(a, b, c)``.  gfx11 and gfx12.
    WMMA = 'a, b, c'
    #: ``d = f(a_neg, a, b_neg, b, c_mod, c, matrix_a_reuse,
    #: matrix_b_reuse)``.  The two `reuse` flags tell the hardware an operand
    #: is unchanged from the previous WMMA and save it a fetch, which a k-loop
    #: holding A fixed across several B can set for free.
    WMMA_MODS = ('a_neg, a, b_neg, b, c_mod, c, '
                 'matrix_a_reuse, matrix_b_reuse')


#: Significand bits, implicit leading bit included.  What decides how many
#: terms a split-precision emulation needs, so it is stated per *format*, not
#: per storage type: `xf32` arrives in `float` registers and is rounded to a
#: narrower significand inside the matrix unit, which no C++ type records.
MANTISSA = {
    Datatype.F64: 53,
    Datatype.F32: 24,
    Datatype.F16: 11,
    Datatype.BF16: 8,
}

#: XF32's significand width, asserted rather than read: AMD documents the
#: format as TF32-shaped, and TF32 is 10 stored bits plus the implicit one.
#: It is the one number in this module that no vendored table confirms.  The
#: only thing depending on it is how many products a split needs, so a wrong
#: value shows up as an accuracy result rather than as a wrong kernel.
XF32_MANTISSA = 11


@dataclass(frozen=True)
class Fragment:
    """What one lane holds of one operand.

    `per_lane` is a count of elements and becomes ``ScalarType(dtype,
    per_lane)`` at the call.  Deliberately not a `RegisterLayout`: the
    k-extent a lane holds is a vector *type* over the slot dimension, and
    `LaneAxis` says in its own docstring that this is not what it describes.
    """

    dtype: Datatype
    per_lane: int


@dataclass(frozen=True)
class MatrixOp:
    """One matrix instruction, as LLVM defines it."""

    builtin: str
    m: int
    n: int
    k: int
    #: Independent matrix products the instruction computes at once.  Declared
    #: rather than derived; see the module docstring.
    blocks: int
    #: Lanes the fragment layouts are stated against.  An instruction gated on
    #: `wavefrontsize32` cannot be emitted from a wave64 kernel or the other
    #: way round, which `available_for` checks against the hardware.
    wave: int
    a: Fragment
    b: Fragment
    d: Fragment
    #: The subtarget feature this row *requires*, spelled as LLVM spells it.
    feature: str
    call: Call
    #: The feature clang gates the builtin on, when it is weaker than
    #: `feature`.  The two are usually the same and for XF32 they are not:
    #: clang accepts `mfma_f32_16x16x8_xf32` on any `mai-insts` target and the
    #: backend then fails to select it, because the instruction itself is
    #: gated on `xf32-insts` and exists only on gfx942.  Recording both keeps
    #: the row honest in each direction --- offering the instruction where the
    #: hardware lacks it, or naming a feature clang would reject.
    gate: Optional[str] = None
    #: Significand bits of the *arithmetic*, when it is narrower than the
    #: operand type suggests.  `None` means "whatever `a.dtype` has".
    mantissa: Optional[int] = None

    def __post_init__(self):
        for name, frag, extent in (('a', self.a, self.m * self.k),
                                   ('b', self.b, self.k * self.n),
                                   ('d', self.d, self.m * self.n)):
            held = frag.per_lane * self.wave
            if held % (extent * self.blocks):
                raise ValueError(
                    f'{self.builtin}: {name} holds {held} slots for '
                    f'{extent * self.blocks} elements, which does not divide')

    @property
    def callee(self) -> str:
        """The name a call site writes.

        The catalogue stores the bare name because that is how the vendored
        table stores it, and the table is what the rows are checked against.
        The prefix is added here so a mismatch cannot hide behind it.
        """
        return f'__builtin_amdgcn_{self.builtin}'

    @property
    def broadcast(self) -> bool:
        """Does the instruction have `cbsz`/`abid`/`blgp`?"""
        return self.call is Call.MFMA

    @property
    def significand(self) -> int:
        return self.mantissa or MANTISSA[self.a.dtype]

    def replication(self, which: str = 'a') -> int:
        """How many times over the wave holds each element of an operand.

        1 everywhere except the `wmma-256b-insts` fragments, where A and B are
        duplicated across the half-waves: 2 at wave32, 4 at wave64.  Not a
        quirk to work around but the reason those instructions cost 8 VGPRs
        per operand where gfx12 costs 4, and that belongs wherever the two get
        compared.
        """
        frag, extent = {'a': (self.a, self.m * self.k),
                        'b': (self.b, self.k * self.n),
                        'd': (self.d, self.m * self.n)}[which]
        return frag.per_lane * self.wave // (extent * self.blocks)

    def available_for(self, dtype, ctx) -> bool:
        """Can this instruction be emitted here, for an accumulator of `dtype`?

        Three separate questions, kept separate on purpose: does the target
        have the feature, does its wave match, and does the accumulator type
        match what the caller accumulates in.  The first is the one that
        bites --- calling a builtin without its subtarget feature is a compile
        error, and the families do not line up with the features: gfx1250 and
        gfx1251 differ by exactly one instruction.
        """
        return (has_feature(ctx, self.feature)
                and wave_size(ctx) == self.wave
                and self.d.dtype == dtype)

    def fits(self, threads: int) -> bool:
        """Does one multiplication's thread count reach a whole wave?

        A matrix instruction is a whole-wave operation.  A kernel spreading a
        multiplication over fewer threads than the wave has could still use
        one by giving the spare lanes something to do, which nothing here
        arranges yet.
        """
        return threads == self.wave

    def cbsz(self, threads: int) -> int:
        """How many blocks share one A operand: the instruction's `cbsz`.

        `min(blocks, threads // n)`, because two different things bound it.
        The instruction cannot broadcast to more blocks than it has, and the
        scheme cannot broadcast across more lanes than one multiplication
        owns: below a full wave, several multiplications share it and each
        needs its own A.

        This was `threads // block`, and it was right for as long as every
        entry had `k == 1` --- there `blocks == wave // n` and the two
        expressions agree, which is why a table of hand-written constants and
        then a formula both reproduced the same numbers.  They stop agreeing
        at the first `k > 1` entry: `mfma_f64_4x4x4f64` has four blocks where
        `threads // n` is sixteen, and a `cbsz` of 4 on a four-block
        instruction names a block that does not exist.
        """
        if threads <= 0 or threads % self.n:
            raise ValueError(f'{self.builtin} needs threads to be a multiple '
                             f'of {self.n}, got {threads}')
        return min(self.blocks, threads // self.n).bit_length() - 1

    def lane_batched(self) -> bool:
        """Does the scheme `matmul32` implements fit this instruction?

        That scheme maps the generator's dimensions onto the instruction's
        like this:

        * the instruction's **M** takes the output columns, which live in the
          accumulator's registers;
        * the instruction's **N**, times the block count, takes the *lanes*,
          which carry the generator's leading dimension;
        * the instruction's **K** takes the contraction, one value per
          instruction, selected through `abid`.

        The middle line is the constraint.  The lanes have to be entirely the
        leading dimension, so ``n * blocks`` has to be the whole wave --- and
        with one element per lane the operand invariant makes that the same
        statement as ``k == 1``::

            b.per_lane * wave == k * n * blocks     (the invariant)
            b.per_lane == 1
            => n * blocks == wave / k

        Against the *wave*, not the thread count.  Below a full wave several
        multiplications share it and each still sees whole blocks; what
        narrows then is the broadcast, which is `cbsz`\'s business.  Writing
        this against `threads` would refuse the 32-thread case that works
        today.

        So every `k > 1` instruction spends part of its lane index on the
        contraction.  The generator's data operand does not have it there: it
        carries the leading dimension in lanes and the contraction in
        registers, because `unwindI` hands the leading dimension a `LeadIndex`
        and `unwindK(..., full=True)` hands the contraction a plain one.

        Which makes the FP64 MFMAs *not* a drop-in for this path, contrary to
        what a matching per-lane width suggests.  Same width, different
        assignment: fitting them is not a matter of a wider tile but of
        staging the data operand so that k reaches the lanes --- the same
        staging the split-precision paths need for their k-vectors.
        """
        return (self.a.per_lane == 1 and self.b.per_lane == 1
                and self.k == 1 and self.m == self.n
                and self.n * self.blocks == self.wave)


# --------------------------------------------------------------------------- #
# The entries
# --------------------------------------------------------------------------- #

def _ops(feature, call, wave, rows, mantissa=None, gate=None):
    """One feature's rows.

    A row is ``(builtin, m, n, k, blocks, in_dtype, in_per_lane, out_dtype,
    out_per_lane)`` --- A and B always share a format and a width, on every
    instruction in the table.
    """
    return tuple(
        MatrixOp(builtin=builtin, m=m, n=n, k=k, blocks=blocks, wave=wave,
                 a=Fragment(din, wa), b=Fragment(din, wa),
                 d=Fragment(dout, wd),
                 feature=feature, call=call, mantissa=mantissa,
                 gate=gate)
        for builtin, m, n, k, blocks, din, wa, dout, wd in rows)


F32, F64, F16, BF16 = (Datatype.F32, Datatype.F64, Datatype.F16, Datatype.BF16)

#: gfx908 and up.  The K=1 F32 tiles are what `matmul32` emits today; the rest
#: of the block is the same generation's F16 and two-wide BF16.
_MAI = _ops('mai-insts', Call.MFMA, 64, (
    ('mfma_f32_4x4x1f32',      4,  4,  1, 16, F32,  1, F32,  4),
    ('mfma_f32_4x4x2bf16',     4,  4,  2, 16, BF16, 2, F32,  4),
    ('mfma_f32_4x4x4f16',      4,  4,  4, 16, F16,  4, F32,  4),
    ('mfma_f32_16x16x1f32',   16, 16,  1,  4, F32,  1, F32, 16),
    ('mfma_f32_16x16x2bf16',  16, 16,  2,  4, BF16, 2, F32, 16),
    ('mfma_f32_16x16x4f16',   16, 16,  4,  4, F16,  4, F32, 16),
    ('mfma_f32_16x16x4f32',   16, 16,  4,  1, F32,  1, F32,  4),
    ('mfma_f32_16x16x8bf16',  16, 16,  8,  1, BF16, 2, F32,  4),
    ('mfma_f32_16x16x16f16',  16, 16, 16,  1, F16,  4, F32,  4),
    ('mfma_f32_32x32x1f32',   32, 32,  1,  2, F32,  1, F32, 32),
    ('mfma_f32_32x32x2bf16',  32, 32,  2,  2, BF16, 2, F32, 32),
    ('mfma_f32_32x32x2f32',   32, 32,  2,  1, F32,  1, F32, 16),
    ('mfma_f32_32x32x4bf16',  32, 32,  4,  1, BF16, 2, F32, 16),
    ('mfma_f32_32x32x4f16',   32, 32,  4,  2, F16,  4, F32, 32),
    ('mfma_f32_32x32x8f16',   32, 32,  8,  1, F16,  4, F32, 16),
))

#: gfx942 only, and not carried forward to gfx950 --- a path built on it does
#: not survive the next generation.  The operands are `float` and the
#: arithmetic is not: this is the AMD counterpart of the 3xTF32 path in
#: `primitives/nvidia.py`, and its split is smaller than BF16's because the
#: format keeps three more significand bits.
_XF32 = _ops('xf32-insts', Call.MFMA, 64, (
    ('mfma_f32_16x16x8_xf32', 16, 16,  8,  1, F32,  2, F32,  4),
    ('mfma_f32_32x32x4_xf32', 32, 32,  4,  1, F32,  2, F32, 16),
), mantissa=XF32_MANTISSA, gate='mai-insts')

#: gfx90a and up: four-wide BF16 operands (`_1k`), and FP64.
#:
#: The FP64 pair is the cheapest entry to reach from where the generator
#: stands.  Both take *scalar* A and B, which is the fragment shape the K=1
#: F32 path already produces --- no k-vector, no split, no staging.
_GFX90A = _ops('gfx90a-insts', Call.MFMA, 64, (
    ('mfma_f32_4x4x4bf16_1k',     4,  4,  4, 16, BF16, 4, F32,  4),
    ('mfma_f32_16x16x4bf16_1k',  16, 16,  4,  4, BF16, 4, F32, 16),
    ('mfma_f32_16x16x16bf16_1k', 16, 16, 16,  1, BF16, 4, F32,  4),
    ('mfma_f32_32x32x4bf16_1k',  32, 32,  4,  2, BF16, 4, F32, 32),
    ('mfma_f32_32x32x8bf16_1k',  32, 32,  8,  1, BF16, 4, F32, 16),
    ('mfma_f64_4x4x4f64',         4,  4,  4,  4, F64,  1, F64,  1),
    ('mfma_f64_16x16x4f64',      16, 16,  4,  1, F64,  1, F64,  4),
))

#: gfx950: the same 16-bit formats at twice the K.
_GFX950 = _ops('gfx950-insts', Call.MFMA, 64, (
    ('mfma_f32_16x16x32_bf16', 16, 16, 32,  1, BF16, 8, F32,  4),
    ('mfma_f32_16x16x32_f16',  16, 16, 32,  1, F16,  8, F32,  4),
    ('mfma_f32_32x32x16_bf16', 32, 32, 16,  1, BF16, 8, F32, 16),
    ('mfma_f32_32x32x16_f16',  32, 32, 16,  1, F16,  8, F32, 16),
))

#: RDNA 3.  16 elements per lane for a 16x16x16 operand is twice what the
#: shape needs at wave32 and four times at wave64; `replication()` says so.
_WMMA_256B = _ops('wmma-256b-insts', Call.WMMA, 32, (
    ('wmma_f32_16x16x16_bf16_w32', 16, 16, 16, 1, BF16, 16, F32, 8),
    ('wmma_f32_16x16x16_f16_w32',  16, 16, 16, 1, F16,  16, F32, 8),
)) + _ops('wmma-256b-insts', Call.WMMA, 64, (
    ('wmma_f32_16x16x16_bf16_w64', 16, 16, 16, 1, BF16, 16, F32, 4),
    ('wmma_f32_16x16x16_f16_w64',  16, 16, 16, 1, F16,  16, F32, 4),
))

#: RDNA 4, and the gfx117x refresh that got the same fragments.  Same shape as
#: RDNA 3, half the registers.
_WMMA_128B = _ops('wmma-128b-insts', Call.WMMA, 32, (
    ('wmma_f32_16x16x16_bf16_w32_gfx12', 16, 16, 16, 1, BF16, 8, F32, 8),
    ('wmma_f32_16x16x16_f16_w32_gfx12',  16, 16, 16, 1, F16,  8, F32, 8),
)) + _ops('wmma-128b-insts', Call.WMMA, 64, (
    ('wmma_f32_16x16x16_bf16_w64_gfx12', 16, 16, 16, 1, BF16, 4, F32, 4),
    ('wmma_f32_16x16x16_f16_w64_gfx12',  16, 16, 16, 1, F16,  4, F32, 4),
))

#: gfx1250 and gfx1251: K doubled again, and the modifier-carrying call shape.
_WMMA_N16 = _ops('wmma-n16-insts', Call.WMMA_MODS, 32, (
    ('wmma_f32_16x16x32_bf16', 16, 16, 32, 1, BF16, 16, F32, 8),
    ('wmma_f32_16x16x32_f16',  16, 16, 32, 1, F16,  16, F32, 8),
))

#: The two rows that make the emulation unnecessary where they exist: gfx1250
#: has a native F32 matrix instruction, and gfx1251 adds F64.  The latter is
#: the only builtin gated on `gfx1251-gemm-insts`, which is what that feature
#: is for.
_GFX1250 = _ops('gfx1250-insts', Call.WMMA_MODS, 32, (
    ('wmma_f32_16x16x4_f32', 16, 16, 4, 1, F32, 2, F32, 8),
))

_GFX1251 = _ops('gfx1251-gemm-insts', Call.WMMA_MODS, 32, (
    ('wmma_f64_16x16x4_f64', 16, 16, 4, 1, F64, 2, F64, 8),
))

MATRIX_OPS = (_MAI + _XF32 + _GFX90A + _GFX950
              + _WMMA_256B + _WMMA_128B + _WMMA_N16 + _GFX1250 + _GFX1251)

#: Families the vendored table lists and this catalogue leaves out, with the
#: reason.  Stated so the check against LLVM can be an equality: a new builtin
#: matching none of these shows up as a failure rather than as silence.
NOT_MODELLED = {
    'i32 accumulator': 'no operator accumulates in integers',
    'fp8/bf8/fp6/fp4 operands': 'below what any split reaches back to F32',
    'f16/bf16 accumulator': 'every operator accumulates in F32 or F64',
    'swmmac': 'structured sparsity; the sparse path does not select tiles',
}


def ops_for(dtype, ctx, threads=None):
    """Every catalogue entry emittable here, largest tile first.

    Says nothing about whether one *should* be used.  A BF16 entry offered for
    an F32 accumulator is offered as the substrate of a split, and whether
    that split beats the direct path is a cost question, which is `select`'s.
    """
    out = [op for op in MATRIX_OPS if op.available_for(dtype, ctx)]
    if threads is not None:
        out = [op for op in out if op.fits(threads)]
    return tuple(sorted(out, key=lambda op: (-op.m * op.n, -op.k, op.builtin)))


def lane_batched_ops(dtype, ctx):
    """Every catalogue entry the lane-batched scheme could emit, largest first.

    The F32 policy does not go through this --- it goes through `MFMA_TILES`,
    which carries the transposes as well.  This is the same question asked of
    the whole catalogue, for the emitter that will need it: it is what says
    that widening the type check to F64 finds nothing, rather than finding
    `mfma_f64_16x16x4f64` and feeding it wrongly.
    """
    return tuple(op for op in ops_for(dtype, ctx)
                 if op.broadcast and op.lane_batched())


def split_terms(op, dtype) -> int:
    """Operand terms whose sum reproduces a `dtype` significand.

    Three BF16 terms cover F32's 24 bits exactly; two cover 16, which is more
    than TF32 and less than F32.  The formula is here so that the reduced
    variants are a choice with a number attached rather than a habit ---
    `mfma_emu_f16_f32` used two F16 terms, which is 22 bits, and nothing said
    so.

    It says nothing about *range*.  F16 carries five exponent bits, so an F16
    split of an F32 operand also needs scaling to stay inside them; BF16 and
    XF32 carry F32's exponent and need none.
    """
    return max(1, ceil(MANTISSA[dtype] / op.significand))


def split_products(terms: int, keep: Optional[int] = None
                   ) -> Tuple[Tuple[int, int], ...]:
    """Which `(i, j)` term products to accumulate, smallest contribution first.

    Term `i` is worth about ``2**(-significand*i)`` of the operand, so the
    product `(i, j)` is worth ``2**(-significand*(i+j))``: everything with
    ``i + j >= keep`` sits at or below the target's own rounding error and is
    dropped.  At ``keep == terms`` that leaves ``terms*(terms+1)/2``
    products --- six for BF16 into F32, which is what `mfma_emu_bf16_f32`
    emits.

    Smallest first, so the small contributions accumulate before the large one
    rounds them off.  The order is free and never worse; how much it buys
    depends on how much the accumulator already carries from earlier k, which
    is a measurement rather than a derivation.
    """
    keep = terms if keep is None else keep
    pairs = [(i, j) for i in range(terms) for j in range(terms)
             if i + j < keep]
    return tuple(sorted(pairs, key=lambda p: (-(p[0] + p[1]), p)))


# --------------------------------------------------------------------------- #
# The F32 K=1 tiling policy
# --------------------------------------------------------------------------- #
#
# What follows is not a second catalogue.  It is the policy `matmul32` runs on
# today: three square K=1 F32 tiles, each fed through a cross-lane transpose.
# The transpose is the part that is *ours* --- it is how this generator
# arranges A, not something the instruction requires --- which is why it lives
# here and not on `MatrixOp`.

#: Transposes `include/tensorforge_device/hip.h` actually defines.
#: `tests/test_amd_catalog.py` checks this against the header, the same way
#: the `fmacdpp` capabilities are checked --- the list is a copy of a C++ fact
#: and copies drift.
#:
#: `transpose16x2` and `transpose16x4` are in the runtime and unused by any
#: tile here.  They are not square: they permute 2 or 4 registers *within* a
#: 16-lane row rather than transposing a 16x16 tile, so they do not fit the
#: `MfmaTile` shape.  Listing them anyway keeps this set meaning what its name
#: says, which is what makes the check against the header a real check.
DEFINED_TRANSPOSES = frozenset({
    'tensorforge::transpose4x4b32',
    'tensorforge::transpose16x16b32',
    'tensorforge::transpose16x2',
    'tensorforge::transpose16x4',
})

#: Tile width -> the transpose that feeds its A operand, and whether that
#: transpose declares its outputs separately from its inputs.  No
#: `transpose32x32b32` exists, so `available_for` refuses the 32-wide tile and
#: the wider path stays unreachable --- which is what the commented-out
#: `write_matmul(32, ...)` call used to express, silently and without saying
#: why.
_TILE_TRANSPOSES = {
    4: ('tensorforge::transpose4x4b32', True),
    16: ('tensorforge::transpose16x16b32', False),
    32: ('tensorforge::transpose32x32b32', False),
}


@dataclass(frozen=True)
class MfmaTile:
    """One square K=1 MFMA tile: the intrinsic, and how its A operand is fed.

    A `MatrixOp` plus the one fact the catalogue does not carry --- the
    cross-lane transpose that feeds A.  That transpose is *ours*: it is how
    this generator arranges the operand, not something the instruction
    requires, and it is the reason this type still exists beside `MatrixOp`.
    Everything else is delegated, so the intrinsic name and the broadcast
    control have one source and the check against LLVM covers them.
    """

    op: MatrixOp
    #: Cross-lane transpose applied to the A registers before the tile runs.
    #: `None` means the tile needs none; a name that `hip.h` does not define
    #: means the tile is unusable, which `available_for` reports.
    transpose: Optional[str]
    #: True when the transpose declares its outputs separately from its
    #: inputs --- `tp(w1..wn, v1..vn)` rather than `tp(w1..wn)`.  That is what
    #: lets it be emitted in SSA form: fresh values come out, so they can
    #: carry the layout the exchange produced, and the inputs are left alone
    #: instead of being rewritten underneath whatever else still reads them.
    #: `transpose4x4b32` has it; `transpose16x16b32` is in-place only.
    transpose_has_separate_outputs: bool = False

    @property
    def block(self) -> int:
        """The square tile width."""
        return self.op.m

    @property
    def builtin(self) -> str:
        return self.op.callee

    def scale(self, threads: int) -> int:
        """The intrinsic's `cbsz`, from the instruction rather than the tile.

        `MatrixOp.cbsz` says why the two used to be the same expression and
        are not one.
        """
        if not self.fits(threads):
            raise ValueError(
                f'{self.builtin} needs threads to be a multiple of '
                f'{self.block}, got {threads}')
        return self.op.cbsz(threads)

    def fits(self, threads: int) -> bool:
        return threads >= self.block and threads % self.block == 0

    def available_for(self, threads: int, dtype, ctx) -> bool:
        """Can this tile be emitted here at all?

        Three separate questions, kept separate on purpose: does the hardware
        have MFMA, does the tile divide the thread count, and does the runtime
        define the transpose it needs.  The third is the one that bites --- a
        tile whose transpose is missing produces a call to an undeclared
        template, exactly like `fmacdpp4` on gfx900.

        The first used to be `cdna1(ctx) and not gfx1251(ctx)`, a family
        predicate standing in for the `mai-insts` feature.  They agree on
        every target this is tested against and disagree on gfx90b--gfx90f,
        which the range admits and the hardware does not have.
        """
        if dtype != Datatype.F32:
            return False
        if not has_feature(ctx, 'mai-insts'):
            return False
        if not self.fits(threads):
            return False
        if not self.op.lane_batched():
            return False
        return self.transpose is None or self.transpose in DEFINED_TRANSPOSES


def _tile(block):
    op = next(o for o in MATRIX_OPS
              if o.m == o.n == block and o.k == 1
              and o.a.dtype is Datatype.F32)
    transpose, separate = _TILE_TRANSPOSES[block]
    return MfmaTile(op, transpose, separate)


#: Derived from `MATRIX_OPS`, so the intrinsic names have one source and the
#: check against LLVM covers them too.
MFMA_TILES = tuple(_tile(block) for block in (4, 16, 32))


def usable_mfma_tiles(threads, dtype, ctx):
    """Widest first -- the order the tiling loop wants to try them in."""
    return tuple(sorted((t for t in MFMA_TILES
                         if t.available_for(threads, dtype, ctx)),
                        key=lambda t: -t.block))


def mfma_tile_for(threads, dtype, ctx):
    """The tile `matmul32` would emit here, or `None`.

    One function, asked twice: once by `matmul()` deciding which path to take
    and once by `matmul32` picking the tile. They used to ask different
    questions --- a family predicate at the router, a `next()` over the usable
    tiles at the emitter --- and agreed only because both happened to be true
    on the same targets. A router that says yes where the emitter finds
    nothing raises `StopIteration` out of code generation, which is not a
    diagnosis of anything.
    """
    # Policy, not capability: the 16-wide tile needs a staging step that is
    # not written and the 32-wide one has no transpose in the runtime.
    return next((t for t in usable_mfma_tiles(threads, dtype, ctx)
                 if t.block == 4), None)
