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


# --------------------------------------------------------------------------- #
# Where it may not go
# --------------------------------------------------------------------------- #

def test_an_extern_buffer_may_be_swizzled_if_nothing_names_it_in_text():
    """`extern` is about the *name* escaping; what matters is whether an
    *access* does.

    Those were the same question only while every named access was text.  The
    macro layer's windows are extern -- other instructions still spell `s0`
    out -- and every read and write of one now goes through `load` and `store`,
    which is what applies the permutation.
    """
    b = builder()
    tile = b.alloc(Datatype.F32, (64,), MemSpace.SHARED, hint='s0', extern='s0',
                   arena='arena', offset=0, swizzle=XorSwizzle(8))
    idx = b.rawexpr('threadIdx.x', type_=INDEX, hint='a')
    b.load(tile, idx, hint='v')
    assert b.finish()


def test_a_raw_access_to_a_swizzled_buffer_is_refused():
    """The failure the earlier guard was aiming at, asked properly.

    A permutation applied to some accesses and not others is a store and a
    load that disagree about where an element lives: a wrong kernel, not a
    slow one.  Checked over the finished body, because that is the first point
    at which the answer is knowable -- the buffer is allocated long before its
    accesses are emitted.
    """
    b = builder()
    tile = b.alloc(Datatype.F32, (64,), MemSpace.SHARED, hint='s0',
                   swizzle=XorSwizzle(8))
    # No `accesses` argument, so the statement is conservative and the
    # earlier check does not object -- which is the case that has to reach the
    # one at `finish`.
    b(f'float x = {tile}[0];')
    with pytest.raises(IRError, match="named in raw text"):
        b.finish()


def test_extern_without_a_swizzle_is_still_fine():
    """The macro layer's windows are extern and unswizzled, which is the
    arrangement today and has to keep working."""
    b = builder()
    assert b.alloc(Datatype.F32, (64,), MemSpace.SHARED, hint='s0',
                   extern='s0', arena='arena', offset=0) is not None


def test_a_named_load_still_takes_the_structured_path():
    """The prerequisite that made the rest of this reachable.

    `Symbol.load` used to leave the structured path whenever the consumer
    needed a particular identifier, which was almost always.  Measured on the
    cases with the worst bank conflicts, all 5650 accesses to a shared symbol
    took the text path -- so a swizzle on such a buffer would have applied to
    nothing, and would have become wrong rather than useless the moment one of
    them was converted.

    `extern` on the load supplies the name, so the address goes in as an
    operand and the emitted line is unchanged.
    """
    b = builder()
    tile = b.alloc(Datatype.F32, (64,), MemSpace.SHARED, hint='s0')
    idx = b.rawexpr('threadIdx.x', type_=INDEX, hint='a')
    v = b.load(tile, idx, hint='data', extern='v42_data')
    b('use(v42_data);', v, accesses=())
    text = emitted(b.finish())
    assert 'float v42_data = ' in text, text
    assert 'use(v42_data);' in text


# --------------------------------------------------------------------------- #
# The width is a per-tile choice, not a constant
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("name,access,lanes,widths", [
    # (pattern, active lanes, {width: expected ways})
    ("B fragment load", lambda i, t: (t % 4) + (t // 4) * 8, 32,
     {None: 2, 8: 1, 16: 2, 32: 2}),
    ("C fragment load", lambda i, t: (t % 16) * 8, 16,
     {None: 4, 8: 2, 16: 1, 32: 1}),
    ("C staging store", lambda i, t: t * 2, 32,
     {None: 2, 8: 2, 16: 2, 32: 1}),
])
def test_no_single_width_serves_every_tile(name, access, lanes, widths):
    """Which is why the width is chosen per buffer and not fixed.

    The B fragment wants 8 and the C tile wants 32, and each is 2-way under
    the other's choice.  A constant would have looked like a simplification
    and cost one of them.
    """
    for width, want in widths.items():
        swz = XorSwizzle(width) if width else None
        f = (lambda t: swz.apply(access(None, t))) if swz else (
            lambda t: access(None, t))
        assert _ways(f, lanes) == want, (
            f"{name} under {'xor%d' % width if width else 'no swizzle'}")


@pytest.mark.parametrize("stride", [9, 13])
def test_an_odd_stride_is_already_clean_and_a_swizzle_would_hurt(stride):
    """Why the gate stays conservative.

    A row width coprime with 32 spreads a column read over all the banks by
    itself.  Permuting it moves elements the plain layout had already placed
    well, and the access gets worse -- so a swizzle applied to everything
    would trade a real win on some tiles for a real loss on others.
    """
    plain = lambda t: t * stride
    assert _ways(plain, 32) == 1
    for width in (8, 16, 32):
        assert _ways(lambda t: XorSwizzle(width).apply(plain(t)), 32) > 1


# --------------------------------------------------------------------------- #
# Choosing the width for a macro window
# --------------------------------------------------------------------------- #

def _window_width(volume, banks=32):
    """The rule `AbstractShrMemWrite._swizzle` applies, in one place to test."""
    width = 1
    while width * 2 <= banks and volume % (width * 2) == 0:
        width *= 2
    return width


@pytest.mark.parametrize("rows,cols,want", [
    (32, 16, 32),   # 512
    (16, 16, 32),   # 256 -- takes 32, not the row width, and reads 1-way
    (56, 13, 8),    # 728 = 8 * 91; the row-width rule declined outright
    (12, 8, 32),    # 96
    (32, 32, 32),   # 1024, capped at the bank count
    (9, 9, 1),      # 81 is odd: no swizzle, and that is the right answer
    (13, 13, 1),    # 169
])
def test_the_width_divides_the_volume(rows, cols, want):
    assert _window_width(rows * cols) == want


@pytest.mark.parametrize("rows,cols", [(32, 16), (16, 16), (56, 13), (12, 8),
                                       (9, 9), (13, 13), (12, 12)])
def test_the_permutation_never_leaves_the_buffer(rows, cols):
    """Why the width has to divide the volume rather than match the row.

    The permutation maps each block of `width` elements onto itself, so a
    buffer whose last block is partial has indices that permute past its end:
    728 elements swizzled at 32 puts eight of them into the next window.
    Shared memory, silently, which is the worst failure available here.
    """
    volume = rows * cols
    width = _window_width(volume)
    if width < 2:
        return
    swz = XorSwizzle(width)
    images = [swz.apply(i) for i in range(volume)]
    assert max(images) < volume, "the permutation escaped the buffer"
    assert sorted(images) == list(range(volume)), "not a bijection"


def test_an_odd_volume_declines_rather_than_falls_back():
    """A row width coprime with 32 already spreads a column read over every
    bank; permuting it would move elements the plain layout had placed well."""
    for volume in (81, 169, 91):
        assert _window_width(volume) == 1


# --------------------------------------------------------------------------- #
# Where a permutation must not go
# --------------------------------------------------------------------------- #

def test_a_contiguous_run_does_not_survive_an_element_permutation():
    """Why a bulk or vector access rules the swizzle out.

    `memcpy_async` and ESIMD's `copy_from` both move several *contiguous*
    positions at once, and the permutation acts per element: the components
    arrive transposed, and once the block key exceeds the run they come from
    outside it altogether.
    """
    swz = XorSwizzle(8)
    # a four-wide run starting at 12: the block key is 1, so pairs swap
    assert [swz.apply(i) for i in range(12, 16)] == [13, 12, 15, 14]
    # the elements are all still in the block, just not in order -- which is
    # the good case; a key wider than the run takes them out of it
    assert sorted(swz.apply(i) for i in range(12, 16)) == list(range(12, 16))
    assert swz.apply(4) == 4 and swz.apply(36) == 32, (
        'a run at 36 leaves its own four positions entirely')


def test_a_granular_permutation_would_keep_them_together():
    """Recorded because it is the option, not a defect.

    Permuting *granules* of `g` elements moves the components of one vector
    together and in order.  It costs exactly the spreading it preserves: a
    coarser unit has proportionally fewer distinct keys, so a stride-32 column
    read goes 1-way at granule 1, 2-way at granule 2, 4-way at granule 4.
    """
    def granular(width, g):
        def f(i):
            gran, off = divmod(i, g)
            return (gran ^ ((gran // width) % width)) * g + off
        return f

    for g in (2, 4):
        f = granular(8, g)
        for start in range(0, 64, g):
            run = [f(start + k) for k in range(g)]
            assert run == list(range(run[0], run[0] + g)), (
                f'granule {g} split a run at {start}: {run}')

    def ways(f, stride):
        per = {}
        for t in range(32):
            a = f(t * stride)
            per.setdefault(a % 32, set()).add(a)
        return max(len(v) for v in per.values())

    assert ways(granular(32, 1), 32) == 1
    assert ways(granular(16, 2), 32) == 2
    assert ways(granular(8, 4), 32) == 4
