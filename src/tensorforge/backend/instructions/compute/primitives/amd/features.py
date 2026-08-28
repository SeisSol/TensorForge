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
