# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Architecture predicates: which family a target belongs to.

Nothing here says what a target can *do* -- that is `caps`.  Keeping the
two apart is not tidiness: answering a capability question with a family
predicate is what let the generator emit `fmacdpp4` on gfx900, where the
specialisations are switched off.
"""


def amdarch(ctx):
    archstr = ctx.get_vm().get_hw_descr().model
    return int(archstr[3:], base=16)


def gfx1251(ctx): # no details known yet about the name
    # gfx1251 supports DPP64 (inferred by the LLVM tests)
    return amdarch(ctx) == 0x1251


def gfx1250(ctx): # no details known yet about the name
    # gfx1251 supports DPP64 (inferred by the LLVM tests)
    return amdarch(ctx) == 0x1250


def cdna2(ctx):
    return (amdarch(ctx) < 0x1000 and amdarch(ctx) >= 0x90a) or gfx1251(ctx)


# `cdna1` used to sit here, and its only caller was the matmul router asking
# "does this target have MFMA".  That is `features.has_feature(ctx,
# 'mai-insts')`, which answers it without admitting gfx90b--gfx90f the way a
# range does.  Removed rather than left unreferenced: an unused family
# predicate is the shape the gfx900 bug came in.


def rdna(ctx):
    # TODO: gfx1250 ?
    return amdarch(ctx) >= 0x1000 and amdarch(ctx) < 0x1250

def gfx906(ctx):
    return amdarch(ctx) >= 0x906 and not amdarch(ctx) in (0x907, 0x909, 0x90b, 0x90c, 0x90d, 0x90f)
