# SPDX-FileCopyrightText: 2026 SeisSol Group
#
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
    """`dpp<ctrl, 0xf, 0xf, true>`, quad permute and row rotate.

    0x121-0x12F is `row_ror:n`: within each row of sixteen, lane `l` reads
    lane `(l + n) % 16`. Modelled because `swap<16>` is `row_ror:8` and that
    is the only way to check what `swap` does.

    `row_shl` and `row_shr` stay unmodelled even though they sit next door,
    and the wave controls likewise: nothing names them, and a half-modelled
    instruction would be worse than an absent one.
    """
    if 0x121 <= ctrl <= 0x12F:
        n = ctrl - 0x120
        return [vals[(l & ~15) + ((l + n) % 16)] for l in range(len(vals))]
    if ctrl > 0xFF:
        raise NotImplementedError(
            f'dpp ctrl 0x{ctrl:x} is outside the quad-permute and row-rotate '
            f'ranges; model it before putting an instruction that uses it in '
            f'the table')
    perm = quad_perm(ctrl)
    return [vals[(l & ~3) + perm[l % 4]] for l in range(len(vals))]


def dpp_update(ctrl: int, row_mask: int, bank_mask: int,
               src: Sequence, old: Sequence) -> Lanes:
    """`tensorforge::dppUpdate<ctrl, row_mask, bank_mask, true>(src, old)`.

    The move runs over the whole wave and the masks decide which lanes keep
    the result: `row_mask` bit `r` enables lanes `16r..16r+15`, `bank_mask`
    bit `b` enables lanes `4b..4b+3` of every enabled row, and a lane needs
    both. Disabled lanes keep `old`, which is what makes the pair a merge.
    """
    moved = dpp(ctrl, src)
    return [moved[l] if (row_mask >> (l // 16) & 1)
            and (bank_mask >> ((l % 16) // 4) & 1) else old[l]
            for l in range(len(src))]


def transpose16x16b32(regs: Sequence[Sequence]) -> List[Lanes]:
    """`tensorforge::transpose16x16b32`, from its own source.

    Sixteen registers over sixteen lanes, built as a butterfly: four
    `transpose4x4b32` blocks, then an 8x8 stage on `row_ror:4` and `row_ror:12`
    with alternating bank masks, then a 16x16 stage on `row_ror:8`.

    Two of its controls were the wrong way round when this was written, and
    that is how it was found. Modelled because nothing checked it. The runtime has had it since before
    the relayout table, and `test_amd_relayout.py` covers `transpose4x4b32`
    only -- so what this one does to a lane index was, until now, whatever the
    reader assumed.
    """
    if len(regs) != 16:
        raise ValueError(f'transpose16x16b32 takes 16 registers, got {len(regs)}')
    v = []
    for base in range(0, 16, 4):
        v += transpose4x4b32(regs[base:base + 4])

    u = [None] * 16
    for i in range(4):
        u[i] = dpp_update(0x12c, 0b1111, 0b1010, v[4 + i], v[i])
        u[4 + i] = dpp_update(0x124, 0b1111, 0b0101, v[i], v[4 + i])
        u[8 + i] = dpp_update(0x12c, 0b1111, 0b1010, v[12 + i], v[8 + i])
        u[12 + i] = dpp_update(0x124, 0b1111, 0b0101, v[8 + i], v[12 + i])

    w = [None] * 16
    for i in range(8):
        w[i] = dpp_update(0x128, 0b1111, 0b1100, u[8 + i], u[i])
        w[8 + i] = dpp_update(0x128, 0b1111, 0b0011, u[i], u[8 + i])
    return w


def swizzle(and_mask: int, or_mask: int, xor_mask: int,
            vals: Sequence) -> Lanes:
    """`tensorforge::swizzle<And, Or, Xor>`.

    `ds_swizzle_b32` in bitmask mode: lane `l` reads `((l & and) | or) ^ xor`
    of its own group of 32. The grouping matters -- the masks are five bits
    and cannot reach across the halves of a wave64 -- so the high bit of the
    lane is carried through untouched.
    """
    return [vals[(l & ~31) | ((((l & 31) & and_mask) | or_mask) ^ xor_mask)]
            for l in range(len(vals))]


def swap(vals: Sequence, block: int) -> Lanes:
    """`tensorforge::swap<Block>`, from the branch that Block selects.

    Each branch is modelled from its own source rather than from the map the
    template documents, which is the point: the template documented one map
    and two of its branches implemented another.
    """
    if block == 1:
        return list(vals)
    if block == 64:
        return [vals[(l % 32) + (1 - l // 32) * 32] for l in range(len(vals))]
    if block in (8, 32):
        return swizzle(0x1f, 0x0, block // 2, vals)
    if block == 16:
        return dpp(0x128, vals)
    if block == 4:
        return dpp(0b01001110, vals)
    if block == 2:
        return dpp(0b10110001, vals)
    raise NotImplementedError(f'swap<{block}> has no branch in hip.h')


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


def transpose16x4(regs: Sequence[Sequence]) -> List[Lanes]:
    """`tensorforge::transpose16x4`, from its own source.

    Four registers against the top two bits of the lane index inside each row
    of sixteen: an 8x8 butterfly on `row_ror`, then `transpose16x2` on
    `row_ror:8`. It carried the same swapped pair as the 8x8 stage of
    `transpose16x16b32`, from the same shape of mistake.
    """
    if len(regs) != 4:
        raise ValueError(f'transpose16x4 takes 4 registers, got {len(regs)}')
    v = regs
    u1 = dpp_update(0x12c, 0b1111, 0b1010, v[1], v[0])
    u2 = dpp_update(0x124, 0b1111, 0b0101, v[0], v[1])
    u3 = dpp_update(0x12c, 0b1111, 0b1010, v[3], v[2])
    u4 = dpp_update(0x124, 0b1111, 0b0101, v[2], v[3])
    w1, w3 = transpose16x2([u1, u3])
    w2, w4 = transpose16x2([u2, u4])
    return [w1, w2, w3, w4]


def transpose16x2(regs: Sequence[Sequence]) -> List[Lanes]:
    """`tensorforge::transpose16x2`.

    Unaffected by the swap above: `row_ror:8` is its own inverse over sixteen
    lanes, so both directions are the same control and there was no pair to
    get the wrong way round.
    """
    if len(regs) != 2:
        raise ValueError(f'transpose16x2 takes 2 registers, got {len(regs)}')
    return [dpp_update(0x128, 0b1111, 0b1100, regs[1], regs[0]),
            dpp_update(0x128, 0b1111, 0b0011, regs[0], regs[1])]
