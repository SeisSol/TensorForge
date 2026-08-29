# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""How many lanes a section's lead dimension is spread over.

One number with more leverage than it looks like.  A register array is sized
`extent / lanes` along the distributed axis, so the lane count scales *every*
register array in the section at once -- where a placement decision only says
which arrays exist.  Order 6 in double on gfx90a is the case that shows it: the
stiffness operator needs 148 register slots per lane at 32 lanes and 74 at 64,
which at 8 bytes a slot is the difference between 296 and 148 VGPRs.

It is per section and not per operation.  The descriptors of a section share
one register image, so a lane count one of them needs is a lane count all of
them must have -- which is also why changing it means rebuilding the section
rather than revising one operand.

Stated as a value rather than computed in place so that it can be *chosen*.
Nothing here chooses yet: `deduce` returns what the generator has always
produced, and an explicit config overrides it. That is the whole point --
"build this section with 64 lanes instead" was not previously expressible, and
a search over configurations cannot start from a constant.
"""

from dataclasses import dataclass
from typing import List, Optional

from tensorforge.common.context import Context
from tensorforge.generators.descriptions import (ElementwiseDescr,
                                                 OperationDescription)

#: The default ceiling on lanes per multiplication, and *not* a hardware fact:
#: it bites only where the wave is wider, which today is AMD alone (NVIDIA and
#: Intel report 32 and 16, so the minimum is a no-op there).
#:
#: It is also not arbitrary. Running gfx90a at the full wave halves the
#: per-lane register footprint and has measured *slower* on some kernels, so
#: the ceiling encodes a result rather than an oversight. What it lacked was
#: anywhere to say so, and any way to ask for the other answer.
DEFAULT_LANE_CEILING = 32


@dataclass(frozen=True)
class LaneConfig:
    """The lane geometry one section is built with."""

    #: Lanes the lead dimension is distributed over.
    num_threads: int
    #: Of those, how many do useful work; the rest are masked off at the edges.
    num_active_threads: int
    #: Lead-dimension elements one lane covers where it used to cover one.
    lead_width: int


def deduce(descr_list: List[OperationDescription],
           context: Context,
           ceiling: Optional[int] = DEFAULT_LANE_CEILING) -> LaneConfig:
    """The lane geometry a descriptor list asks for.

    A maximum over the lane counts and a *minimum* over the widths, and the
    asymmetry is not an oversight: one register image is shared across the
    section, so a width one descriptor cannot take is a width none may take,
    while a lane count one descriptor needs is one they must all have.

    `ceiling` caps the result. `None` means the hardware's wave width, which is
    the largest value that is meaningful -- a barrier inside a batch loop is
    only simd-uniform, so a section wider than a wave would need a group
    barrier where it may not have one.

    An elementwise descriptor waives the ceiling. Its iteration space is the
    vector unit's, not a contraction's lead dimension, so compressing it below
    that would leave lanes idle for no gain.
    """
    num_threads = 0
    num_active = 0
    widths = []
    for descr in descr_list:
        threads, active = descr.get_num_threads(context)
        num_threads = max(threads, num_threads)
        num_active = max(active, num_active)
        widths.append(getattr(descr, 'lead_width', lambda _c: 1)(context))

    # Deliberately *not* also clamped to the wave width, which is what the
    # generator has always done.  Clamping would be a change, and a large one:
    # the Intel targets report a 16-wide vector unit, so a ceiling of 16 would
    # take 35 kernels from refusing to generate -- a group barrier inside a
    # simd-uniform loop -- to generating. That is a lead worth following and
    # not a side effect to take while extracting a decision.
    cap = (context.get_vm().get_hw_descr().vec_unit_length
           if ceiling is None else ceiling)
    if not any(isinstance(d, ElementwiseDescr) for d in descr_list):
        num_threads = min(cap, num_threads)

    return LaneConfig(num_threads=num_threads,
                      num_active_threads=num_active,
                      lead_width=min(widths) if widths else 1)
