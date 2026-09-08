# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Which LLVM subtarget features a target has.

`caps` answers what the *runtime* defines --- the `#if` guards in `hip.h`.
This answers what the *ISA* defines, which is a different question with a
different owner: the guards in `hip.h` are ours and can be fixed here, the
subtarget features are LLVM's and can only be read.  A matrix builtin is gated
on one of these, so calling it without the feature is a compile error rather
than the link error `caps` exists to prevent.

Some rows gate nothing and describe throughput instead -- how many FMAs an
issue retires, how wide one DPP move is.  Getting one of those wrong costs a
slower kernel rather than a broken build, which is a reason to check them
against the same source, not a reason to keep them somewhere looser.

Reading them off the architecture number, as `arch` does for families, is what
this module avoids.  The families and the features do not nest the same way:
`gfx1250` and `gfx1251` share `gfx1250-insts` but only `gfx1251` has
`gfx1251-gemm-insts`, and `wmma-128b-insts` covers gfx1170--gfx1172 *and*
gfx1200--gfx1201 while skipping everything between.  A predicate written as a
range gets both wrong.

The lists are a copy of an LLVM fact, so they are checked:
`tests/test_amd_catalog.py` compares them against
`tests/data/amd_matrix_builtins.json`, which `tools/amd_matrix_table.py`
extracts from `AMDGPU.td` and `GCNProcessors.td`.
"""

from .arch import amdarch

#: gfx940 and gfx941 are absent from LLVM main --- the targets were removed,
#: not the instructions --- so they are listed here where the hardware had the
#: feature and the vendored table cannot confirm it.  The check is therefore
#: containment, not equality: everything LLVM names must appear here, and the
#: surplus is these two.
#:
#: Where the vendored table cannot confirm, what to assume depends on which
#: way being wrong hurts.  For a feature that says *an instruction exists*,
#: guessing wrong costs a compile error, so these two are listed with the rest
#: of CDNA3.  For one that says *no assurance is needed from the caller* ---
#: `agent-scope-fine-grained-remote-memory-atomics` and
#: `memory-atomic-fadd-f32-denormal-support` --- guessing wrong costs a silent
#: no-op on fine-grained memory, so they are left out and the assurance is
#: asked for.
_REMOVED_FROM_LLVM = (0x940, 0x941)

#: Feature string -> the targets that carry it.  Same spelling as the
#: `AMDGPUBuiltin` gate, so an entry in `catalog` can name the feature it needs
#: in exactly the form LLVM does.
FEATURE_TARGETS = {
    # CDNA.  `mai-insts` is the Matrix Arithmetic Instructions themselves;
    # `gfx90a-insts` adds the four-wide bf16 operands (`_1k`) and FP64 MFMA.
    'mai-insts': (0x908, 0x90a, *_REMOVED_FROM_LLVM, 0x942, 0x950),
    'gfx90a-insts': (0x90a, *_REMOVED_FROM_LLVM, 0x942, 0x950),
    'gfx940-insts': (*_REMOVED_FROM_LLVM, 0x942, 0x950),
    'gfx950-insts': (0x950,),
    # XF32 is gfx942 only.  It was not carried forward to gfx950, so a path
    # built on it does not survive the next generation --- which is a reason
    # to keep it behind the same selection policy as everything else rather
    # than special-casing MI300.
    'xf32-insts': (*_REMOVED_FROM_LLVM, 0x942),

    # RDNA.  The split is the fragment width, not the family: `256b` holds a
    # 16x16x16 operand duplicated across the half-waves, `128b` holds it once.
    'wmma-256b-insts': (0x1100, 0x1101, 0x1102, 0x1103,
                        0x1150, 0x1151, 0x1152, 0x1153, 0x1154),
    'wmma-128b-insts': (0x1170, 0x1171, 0x1172, 0x1200, 0x1201),

    # gfx125x.  `gfx1250-insts` reaches gfx1310 as well, so it is not a
    # synonym for "is a gfx125x part".
    'wmma-n16-insts': (0x1250, 0x1251),
    'gfx1250-insts': (0x1250, 0x1251, 0x1310),
    'gfx1251-gemm-insts': (0x1251,),
    'swmmac-gfx1250-insts': (0x1250, 0x1251),

    #: Not an instruction gate.  gfx1251 buys its FP64 GEMM support at the
    #: price of the family's slowest WMMA, which is a selection input: the
    #: split-precision paths are worth less there than on gfx1250, and the
    #: native FP64 WMMA is worth more.
    'gfx125x-lowest-rate-wmma': (0x1251,),

    # Floating-point atomics, read by `backend/atomics.py`.  They are here and
    # not in a table of their own for the reason the module exists: they are
    # LLVM's facts, checked against LLVM's sources by the same test as the
    # matrix ones.
    #
    # The returning and non-returning forms are separate features and the
    # difference is not cosmetic.  gfx908 carries only the non-returning one,
    # which is the form an accumulation wants -- and
    # `__builtin_amdgcn_global_atomic_fadd_f32` returns, so the builtin is
    # unavailable exactly where the instruction is not.
    'atomic-fadd-rtn-insts': (0x90a, 0x942, 0x950,
                              0x1100, 0x1101, 0x1102, 0x1103,
                              0x1150, 0x1151, 0x1152, 0x1153, 0x1154,
                              0x1170, 0x1171, 0x1172,
                              0x1200, 0x1201, 0x1250, 0x1251, 0x1310,
                              *_REMOVED_FROM_LLVM),
    'atomic-fadd-no-rtn-insts': (0x908, 0x90a, 0x942, 0x950,
                                 0x1100, 0x1101, 0x1102, 0x1103,
                                 0x1150, 0x1151, 0x1152, 0x1153, 0x1154,
                                 0x1170, 0x1171, 0x1172,
                                 0x1200, 0x1201, 0x1250, 0x1251, 0x1310,
                                 *_REMOVED_FROM_LLVM),
    #: FP64 global add.  CDNA2 and up, and gfx125x -- and *not* RDNA, at any
    #: generation, which is the gate a `>= gfx1000` range gets backwards.
    'flat-buffer-global-fadd-f64-inst': (0x90a, 0x942, 0x950, 0x1250, 0x1251,
                                         *_REMOVED_FROM_LLVM),
    #: The LDS counterpart, carried by the same targets.  No caller yet; it is
    #: listed so the answer is looked up rather than assumed when a
    #: shared-memory accumulation first asks.
    'lds-atomic-add-f64': (0x90a, 0x942, 0x950, 0x1250, 0x1251,
                           *_REMOVED_FROM_LLVM),
    'atomic-global-pk-add-bf16-inst': (0x942, 0x950, 0x1200, 0x1201,
                                       0x1250, 0x1251, 0x1310),
    #: Whether the hardware add honours the denormal mode.  Where it does not,
    #: the compiler wants `-fatomic-ignore-denormal-mode` before it will emit
    #: the instruction -- which is why gfx90a needs the flag and gfx942 does
    #: not, on hardware that has the same instruction.
    'memory-atomic-fadd-f32-denormal-support': (0x942, 0x950,
                                                0x1100, 0x1101, 0x1102, 0x1103,
                                                0x1150, 0x1151, 0x1152, 0x1153,
                                                0x1154, 0x1170, 0x1171, 0x1172,
                                                0x1200, 0x1201, 0x1250, 0x1251),
    #: Agent-scope atomics reach fine-grained memory without an assurance from
    #: the caller.  The one entry here that is about the *allocator* rather
    #: than the instruction, and the reason `unsafe_fp_atomics_required`
    #: answers False for gfx942 and True for gfx90a.
    'agent-scope-fine-grained-remote-memory-atomics': (0x942, 0x950,
                                                       0x1200, 0x1201,
                                                       0x1250, 0x1251, 0x1310),

    # How many FMAs an issue retires, read by `select.broadcast_form`.  A DPP
    # modifier excludes every one of these -- an instruction carries a
    # modifier or packed operands, and a VOPD pair carries neither -- so they
    # are what a fused broadcast is paid for.
    #
    #: `v_pk_fma_f32`, on CDNA and on gfx125x, and the two are named
    #: differently in AMDGPU.td: CDNA carries `FeaturePackedFP32Ops` and
    #: gfx125x gates the same arithmetic on the target, so the `single-sgpr`
    #: refinement is the only record that names it.  Read as an existence
    #: marker, which is what makes it worth two rows rather than one -- a
    #: single hand-merged row would state a fact LLVM does not.
    'packed-fp32-ops': (0x90a, *_REMOVED_FROM_LLVM, 0x942, 0x950),
    'packed-fp32-single-sgpr-ops': (0x1250, 0x1251),
    #: `v_pk_fma_f64`, on gfx1251 alone.
    'packed-fp64-single-sgpr-ops': (0x1251,),
    #: Wave32 dual issue: two *independent scalar* FMAs paired by the
    #: compiler, which is a differently shaped kernel from packed math and not
    #: a second spelling of it.  RDNA 3 onwards, gfx125x included, and not
    #: RDNA 1 or 2 -- so a range over `>= gfx1000` claims it four generations
    #: too early.
    'vopd': (0x1100, 0x1101, 0x1102, 0x1103,
             0x1150, 0x1151, 0x1152, 0x1153, 0x1154,
             0x1170, 0x1171, 0x1172,
             0x1200, 0x1201, 0x1250, 0x1251, 0x1310),
    #: DPP over the 64-bit DP ALU: `v_mov_b64_dpp` and `v_fmac_f64_dpp` move
    #: or multiply a whole 64-bit unit -- a `double`, or a pair of floats --
    #: where two 32-bit instructions are needed otherwise.  gfx1251 carries it
    #: and gfx1250 does not, so a family predicate over gfx125x gets one of
    #: the two wrong whichever way it answers.  What will read it is the
    #: packed arrangement, which pays one move where the unit goes in one
    #: piece and two where it does not; it is listed so that answer is looked
    #: up rather than assumed.
    'dpp-64bit': (0x90a, *_REMOVED_FROM_LLVM, 0x942, 0x950, 0x1251),
}


def has_feature(ctx, feature: str) -> bool:
    """Does this target carry `feature`?

    Unknown feature names raise rather than answering `False`: a typo in a
    catalogue entry would otherwise turn into "this instruction is available
    nowhere", which is indistinguishable from a correct entry for hardware we
    do not target.
    """
    if feature not in FEATURE_TARGETS:
        raise KeyError(f'unknown AMDGPU subtarget feature {feature!r}; '
                       f'known: {sorted(FEATURE_TARGETS)}')
    return amdarch(ctx) in FEATURE_TARGETS[feature]


def wave_size(ctx) -> int:
    """Lanes per wavefront on this target.

    Distinct from the `threads` a multiplication is spread over, which is a
    property of the kernel and can be narrower.  A matrix instruction is
    always a whole-wave operation, so its fragment layout is stated against
    this number and an entry whose `wave` disagrees is unusable here.
    """
    return ctx.get_vm().get_hw_descr().vec_unit_length
