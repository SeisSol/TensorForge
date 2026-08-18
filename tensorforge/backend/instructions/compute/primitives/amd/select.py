"""Broadcast width selection.

Strategy first, then narrowed to what the runtime defines.  The split is
the point: a performance tweak to the first must not be able to turn into
a link error.
"""

from tensorforge.common.basic_types import Datatype
from .arch import cdna2, gfx1250, gfx1251, rdna, gfx906
from .caps import has_fmacdpp4, has_fmacdpp8, has_fmacdpp16


def wanted_fmadpp_step(datatype, threads, ctx):
    """The widest broadcast the *hardware strategy* wants -- performance only.

    Deliberately says nothing about whether the target defines it; see
    `select_fmadpp_step`.  Keeping the two apart is what stops a performance
    tweak from silently becoming a link error, which is how gfx900 came to
    emit a call to a template that has only a declaration there.
    """
    step = 1
    if threads >= 4 and datatype == Datatype.F32 and gfx906(ctx):
        step = 4
    if threads >= 8 and datatype == Datatype.F32 and (rdna(ctx) or gfx1251(ctx) or gfx1250(ctx)):
        step = 8
    if threads >= 16 and datatype == Datatype.F32 and (rdna(ctx) or gfx1251(ctx) or gfx1250(ctx)):
        step = 16
    if threads >= 16 and (cdna2(ctx) or gfx1251(ctx)) and datatype in (Datatype.F32, Datatype.F64):
        step = 16
    return step


def select_fmadpp_step(datatype, threads, ctx):
    """What we can actually emit: the strategy, narrowed to what links.

    Falls to the next *narrower* width the target defines.  Narrower is always
    correct -- it just costs more broadcasts -- so an unavailable instruction
    degrades performance instead of breaking the build.
    """
    wanted = wanted_fmadpp_step(datatype, threads, ctx)
    available = {
        1: lambda: True,                       # plain FMA, always there
        4: lambda: has_fmacdpp4(ctx),
        8: lambda: has_fmacdpp8(ctx),
        16: lambda: has_fmacdpp16(ctx, datatype),
    }
    return next(s for s in (wanted, 8, 4, 1) if s <= wanted and available[s]())
