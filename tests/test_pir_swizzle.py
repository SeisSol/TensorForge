# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""A tile can carry a permutation, and every access applies it.

Shared memory is 32 banks of four bytes, and a warp's access costs as many
cycles as the largest number of distinct addresses landing in one bank.  A
tile written a row at a time and read a column at a time cannot avoid that
with padding: the NVIDIA MMA's B fragment has 32 lanes reading 32 distinct
elements spread over 60, and 240 bytes do not fit in 128 of bank width, so
some bank is hit twice whatever the stride.  Padding moves the collision;
transposing moves it to the store.

Permuting each row costs nothing and clears both, and the reason it can live
on the buffer rather than at every access is arithmetic::

    n * width + (k ^ (n % width))  ==  i ^ ((i // width) % width)

`k` occupies the low `log2(width)` bits and `n % width` is smaller than
`width`, so the XOR cannot carry out of them.  The permutation is therefore a
function of the linear index alone, which is what lets `load` and `store`
apply it without any caller knowing.

That placement is the point.  A permutation only some accesses apply is worse
than none: the store and the load would disagree about where an element lives
and the kernel would be quietly wrong rather than merely slow.
"""

from __future__ import annotations

import pytest

from tensorforge.backend.pir.build import IRBuilder
from tensorforge.backend.pir.core import (INDEX, IRError, MemSpace, XorSwizzle,
                                          BufferType)
from tensorforge.backend.pir.emit import Emitter
from tensorforge.common.basic_types import Datatype

BANKS = 32


def builder(budget=512):
    return IRBuilder(fptype=Datatype.F32, scratch=('tempShrMem', budget))


def emitted(body):
    lines = []
    Emitter(lines.append).run(body)
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# The permutation itself
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("width", [2, 4, 8, 16])
def test_the_linear_form_equals_the_two_dimensional_one(width):
    swz = XorSwizzle(width)
    for n in range(4 * width):
        for k in range(width):
            assert swz.apply(n * width + k) == n * width + (k ^ (n % width))


@pytest.mark.parametrize("width", [2, 4, 8, 16])
def test_it_is_a_bijection_so_nothing_is_lost(width):
    """Every element still has exactly one home.  A permutation that collided
    would drop data, which is a different and much worse failure than a bank
    conflict."""
    swz = XorSwizzle(width)
    for rows in (width, 4 * width):
        n = rows * width
        assert sorted(swz.apply(i) for i in range(n)) == list(range(n))


@pytest.mark.parametrize("width", [3, 5, 6, 0])
def test_a_width_that_is_not_a_power_of_two_is_refused(width):
    """The identity above depends on `k` fitting exactly in the low bits."""
    with pytest.raises(IRError, match="power of two"):
        XorSwizzle(width)


def test_no_element_leaves_its_row():
    """Which is what keeps the store conflict-free: a row-wise write still
    covers `width` distinct banks, just in a different order."""
    swz = XorSwizzle(8)
    for i in range(64):
        assert i // 8 == swz.apply(i) // 8


# --------------------------------------------------------------------------- #
# Applied once, by the buffer
# --------------------------------------------------------------------------- #

def test_both_a_load_and_a_store_apply_it():
    b = builder()
    tile = b.alloc(Datatype.F32, (64,), MemSpace.SHARED, hint='btile',
                   swizzle=XorSwizzle(8))
    idx = b.rawexpr('threadIdx.x', type_=INDEX, hint='a')
    b.store(tile, b.load(tile, idx, hint='v'), idx)
    text = emitted(b.finish())
    assert text.count('^') == 2, f"the permutation is not on both sides:\n{text}"


def test_an_unswizzled_buffer_is_untouched():
    b = builder()
    tile = b.alloc(Datatype.F32, (64,), MemSpace.SHARED, hint='atile')
    idx = b.rawexpr('threadIdx.x', type_=INDEX, hint='a')
    b.store(tile, b.load(tile, idx, hint='v'), idx)
    assert '^' not in emitted(b.finish())


def test_a_constant_index_is_folded():
    """Nothing is paid for a permutation the generator can do itself."""
    b = builder()
    tile = b.alloc(Datatype.F32, (64,), MemSpace.SHARED, hint='btile',
                   swizzle=XorSwizzle(8))
    b.load(tile, 9, hint='v')
    text = emitted(b.finish())
    assert '[8]' in text, text
    assert '^' not in text


def test_it_is_a_shift_and_a_mask():
    """A power-of-two width, so the emitted address is what one would write by
    hand rather than something a later pass has to strength-reduce."""
    b = builder()
    tile = b.alloc(Datatype.F32, (64,), MemSpace.SHARED, hint='btile',
                   swizzle=XorSwizzle(8))
    idx = b.rawexpr('threadIdx.x', type_=INDEX, hint='a')
    b.load(tile, idx, hint='v')
    text = emitted(b.finish())
    assert '>> 3' in text and '& 7' in text, text
    assert '/' not in text.split('=')[-1] and '%' not in text.split('=')[-1]


def test_the_swizzle_shows_in_the_type():
    t = BufferType(Datatype.F32, (64,), MemSpace.SHARED, XorSwizzle(8))
    assert 'xor8' in repr(t)


# --------------------------------------------------------------------------- #
# What it was for
# --------------------------------------------------------------------------- #

def _ways(addr, lanes=32):
    per = {}
    for t in range(lanes):
        a = addr(t)
        per.setdefault(a % BANKS, set()).add(a)
    return max(len(v) for v in per.values())


def test_the_mma_fragment_load_stops_conflicting():
    """32 lanes read `(t%4) + (t/4)*8 + ...`, which is 32 distinct addresses
    over 60 elements -- 2-way on any linear stride, 1-way permuted."""
    plain = lambda t: (t % 4) + (t // 4) * 8
    swz = XorSwizzle(8)
    assert _ways(plain) == 2
    assert _ways(lambda t: swz.apply(plain(t))) == 1


def test_the_staging_store_still_does_not_conflict():
    """Eight lanes write one column each.  The permutation must not buy the
    load's conflict at the store's expense, which is what transposing does."""
    swz = XorSwizzle(8)
    for jj in range(8):
        plain = lambda t, jj=jj: (t % 8) + jj * 8
        assert _ways(plain, lanes=8) == 1
        assert _ways(lambda t, p=plain: swz.apply(p(t)), lanes=8) == 1


def test_padding_would_not_have_worked():
    """Recorded because it is the obvious first idea and it is wrong: eight
    rows of nine wrap past 64 and collide again."""
    padded = lambda t: (t % 4) + (t // 4) * 9
    assert _ways(padded) == 2
