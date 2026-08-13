# SPDX-License-Identifier: MIT
"""A wave simulator, to derive what a cross-lane instruction actually does.

The layouts in `amd/relayout.py` are claims about hardware: *this* instruction
turns *that* distribution into *this other* one.  Claims like that cannot be
checked by reading, and the attempt has already failed twice in this codebase
--- once in the `LaneAxis` docstring, once in the broadcast annotation derived
from it, both times producing the right numbers in the wrong roles.

So the claims are checked against an execution instead.  Each primitive is
modelled from its own definition in `include/tensorforge_device/hip.h`: give
every (register, lane) slot a distinct tag, run the instruction, and read off
where each tag ended up.  That map is ground truth; the table has to match it.

Only what is actually established lives here.  A primitive whose definition is
inline assembly this cannot model has no entry, and therefore no table row ---
which is the right outcome: no row means no relayout is offered, and a pass
stays conservative rather than acting on a guess.
"""

from __future__ import annotations

from typing import Callable, List, Sequence, Tuple

Lanes = List          # one entry per lane


# --------------------------------------------------------------------------- #
# DPP
# --------------------------------------------------------------------------- #

def quad_perm(ctrl: int) -> Tuple[int, int, int, int]:
    """DPP_CTRL 0x000-0x0FF: two bits per lane of a quad."""
    return tuple((ctrl >> (2 * i)) & 3 for i in range(4))


def dpp(ctrl: int, vals: Sequence) -> Lanes:
    """`dpp<ctrl, 0xf, 0xf, true>` restricted to the quad-permute range.

    Full row and wave controls are not modelled: nothing in the table needs
    them, and a half-modelled instruction would be worse than an absent one.
    """
    if ctrl > 0xFF:
        raise NotImplementedError(
            f'dpp ctrl 0x{ctrl:x} is outside the quad-permute range; '
            f'model it before putting an instruction that uses it in the table')
    perm = quad_perm(ctrl)
    return [vals[(l & ~3) + perm[l % 4]] for l in range(len(vals))]


# --------------------------------------------------------------------------- #
# The primitives the table names
# --------------------------------------------------------------------------- #

def broadcast(vals: Sequence, block: int, subblock: int, lane: int) -> Lanes:
    """`tensorforge::broadcast<Block, Subblock, Lane>`.

    Result lane `l` takes the source value from lane `Lane*Subblock + l %
    Subblock` within its own block, so the result repeats every `Subblock`
    lanes with neighbours differing.
    """
    assert block % subblock == 0
    assert lane * subblock < block
    out = []
    for l in range(len(vals)):
        base = (l // block) * block
        out.append(vals[base + lane * subblock + (l % subblock)])
    return out


def transpose4x4b32(regs: Sequence[Sequence]) -> List[Lanes]:
    """`tensorforge::transpose4x4b32(w1..w4, v1..v4)`, from its own body.

    Follows the non-assembly path in hip.h verbatim.  The result is the 4x4
    transpose that exchanges the *register* index with the lane index inside a
    quad: `w[r][l] == v[l % 4][(l & ~3) + r]`.

    That exchange is the interesting one for layouts: it moves a tensor
    dimension out of the slots and into the lanes, and the other one back.
    """
    threads = len(regs[0])
    v1, v2, v3, v4 = regs
    vv2, vv4 = dpp(0xa0, v2), dpp(0xa0, v4)
    vv1, vv3 = dpp(0xf5, v1), dpp(0xf5, v3)
    u1 = [v1[l] if l % 2 == 0 else vv2[l] for l in range(threads)]
    u2 = [v2[l] if l % 2 == 1 else vv1[l] for l in range(threads)]
    u3 = [v3[l] if l % 2 == 0 else vv4[l] for l in range(threads)]
    u4 = [v4[l] if l % 2 == 1 else vv3[l] for l in range(threads)]
    uu1, uu2 = dpp(0xee, u1), dpp(0xee, u2)
    uu3, uu4 = dpp(0x44, u3), dpp(0x44, u4)
    return [
        [u1[l] if l % 4 < 2 else uu3[l] for l in range(threads)],
        [u2[l] if l % 4 < 2 else uu4[l] for l in range(threads)],
        [uu1[l] if l % 4 < 2 else u3[l] for l in range(threads)],
        [uu2[l] if l % 4 < 2 else u4[l] for l in range(threads)],
    ]


def movdpp16(vals: Sequence, row: int) -> Lanes:
    """`tensorforge::movdpp16<Row>` -- row share within 16 lanes.

    Every lane of a 16-lane row takes lane `Row` of that row, so one distinct
    value per row and rows sit 16 consecutive threads apart.
    """
    return [vals[(l // 16) * 16 + row] for l in range(len(vals))]


# --------------------------------------------------------------------------- #
# Reading a layout back off a simulation
# --------------------------------------------------------------------------- #

def tagged(threads: int, regs: int = 1) -> List[Lanes]:
    """Distinct tags for every (register, lane) slot."""
    return [[(r, l) for l in range(threads)] for r in range(regs)]


def lane_axis_of(result: Lanes, threads: int):
    """Recover the `(block, stride)` of a single-axis distribution, or None.

    A pattern is described by `LaneAxis(block, stride)` exactly when threads
    hold equal values precisely where `(t // stride) % block` agrees.  Both
    halves of that biconditional matter: checking only that equal indices give
    equal values would accept an axis that merges distinct elements.
    """
    for stride in (1, 2, 4, 8, 16, 32, 64):
        for block in (1, 2, 4, 8, 16, 32, 64):
            if stride * block > threads * 64:
                continue
            if all((result[a] == result[b])
                   == (((a // stride) % block) == ((b // stride) % block))
                   for a in range(threads) for b in range(threads)):
                return block, stride
    return None
