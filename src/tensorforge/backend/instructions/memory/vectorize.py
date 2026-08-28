# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Which widths a linearized transfer may use, and where the hops go.

Two questions that have to be answered together, because the answer to
either alone is wrong.

*How wide* is a property of the base: reinterpreting ``&buf[i]`` as a
``T2``/``T4`` is defined only when that address is aligned to the wider type,
and the base's alignment is the only thing that can promise it.  Nothing
promised anything before -- ``GlbToRegLoader`` carried ``for g in [4, 2, 1]``
commented down to ``[1]``, which is a width decision written as a disabled
list, so re-enabling it would have cast unaligned addresses on whichever
operand happened not to be padded.

*Where the hops go* is a property of the extent, and it was wrong in a way
the width made much worse.  The old loop emitted a hop per ``range`` step and
then set ``start = (total // granularity) * granularity`` from the granularity
it had just finished with, so a partial hop at the end both overran the buffer
and was covered again by the next, narrower width.  At width 1 that overruns
by up to ``threads - 1`` elements; at width 4 by four times as many, and into
a 16-byte access that a padded batch stride no longer covers.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

#: No target loads more than 16 bytes in one instruction: `LDG.128`/`LDS.128`
#: on NVIDIA, `global_load_dwordx4`/`ds_read_b128` on AMD.  So `double4` is
#: not a width, and `float4` is the widest there is.
MAX_ACCESS_BYTES = 16


def widths_for(elem_bytes: int, align_bytes: int,
               cap_bytes: int = MAX_ACCESS_BYTES) -> List[int]:
    """The widths a base of this alignment may be accessed at, widest first.

    ``align_bytes`` is what is *proven* about the base, not what it happens to
    be at runtime.  An unproven alignment answers ``[1]``, the same way
    ``lane_span`` refuses rather than returning 1: "not known to be 16-byte
    aligned" and "known to be 4-byte aligned" are the same permission, and a
    cast that needs 16 must not be able to acquire one by default.
    """
    if elem_bytes < 1:
        raise ValueError(f'element size must be >= 1, got {elem_bytes}')
    return [w for w in (4, 2, 1)
            if w * elem_bytes <= cap_bytes and w * elem_bytes <= max(align_bytes, elem_bytes)]


def plan_hops(total: int, threads: int,
              widths: Sequence[int]) -> Tuple[List[Tuple[int, int]], int]:
    """Cover ``[0, total)`` with whole hops, widest first.

    Returns ``(hops, tail)``: ``hops`` is a list of ``(offset, width)`` where a
    hop moves ``threads * width`` consecutive elements, and ``tail`` is what is
    left over -- fewer than ``threads`` elements once ``widths`` ends in 1.

    Three properties the old arithmetic did not have, and which
    :mod:`tests.test_vector_hops` states as tests rather than as prose:

    * no hop runs past ``total`` -- ``(total - pos) // step`` counts *whole*
      hops, where ``range(pos, total, step)`` counted started ones;
    * no element is covered twice -- ``pos`` advances by the hops actually
      emitted, not by a quantity recomputed from the granularity;
    * every hop offset is a multiple of its own width, which is what makes
      the reinterpret cast at that offset legal given an aligned base.

    The tail is returned rather than emitted as a narrow hop, because covering
    it is a different question: fewer than ``threads`` elements means some
    lanes have nothing to do, and whether they are predicated off or allowed
    to read past the end is a decision about the buffer, not about the width.
    """
    if threads < 1:
        raise ValueError(f'thread count must be >= 1, got {threads}')
    hops: List[Tuple[int, int]] = []
    pos = 0
    for w in widths:
        if w < 1:
            raise ValueError(f'width must be >= 1, got {w}')
        step = threads * w
        n = (total - pos) // step
        hops += [(pos + k * step, w) for k in range(n)]
        pos += n * step
    return hops, total - pos


def register_array_align(volume_bytes: int,
                         cap_bytes: int = MAX_ACCESS_BYTES) -> int:
    """How far a register staging array is declared aligned, in bytes.

    Stated once and read from both ends -- `allocate.py` requests it, and
    `Symbol.linear_align_bytes` reports it -- because the two are the same
    fact, and a width chosen from a guarantee the declaration does not make
    is exactly the bug this whole path is about.

    Declaring it breaks a circularity rather than adding a knob.  The width
    depends on the minimum of the two bases' alignment, and if the register
    end were decided *from* the width there would be nothing to start from;
    with the array always as aligned as one access can use, the width depends
    on the source alone.

    A small array is not over-aligned: an 8-byte one asks for 8, not 16.  The
    padding would be free but the claim would not be true of anything.
    """
    a = 1
    while a * 2 <= min(cap_bytes, max(volume_bytes, 1)):
        a *= 2
    return a
