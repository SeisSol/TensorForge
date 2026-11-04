# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""The permutation of a staged window, decided from the accesses a body
makes (`pir.banks.propose`, `Options.swizzle` `decide`).

The window is built plain, its accesses are read lane by lane, and a
permutation is proposed only where it lowers the bank cycles by more than its
own arithmetic costs.  The second operand of a product is read at the same
address in every lane, which no permutation helps -- and which a rolled loop
pays for in four integer operations an access and a scalar read where a
vector one would do (SeisSol's `localFluxAll` at order 6 on sm_120: 254
instructions a trip permuted, 91 plain, for the same 36 FMAs).
"""

from __future__ import annotations

import pytest

from tensorforge.backend.pir import banks
from tensorforge.backend.pir.build import IRBuilder
from tensorforge.backend.pir.core import INDEX, MemSpace, ScalarType, XorSwizzle
from tensorforge.common.basic_types import Datatype

F32 = Datatype.F32
REAL = ScalarType(F32)
ROWS, COLS = 32, 32


def body_with(read, rows=ROWS, cols=COLS, need=1, unroll=True):
    """A window written a row at a time, the lane along the row, and read by
    `read(b, tile, lane)`."""
    b = IRBuilder(fptype=F32, arena='shrMem')
    tile = b.alloc(F32, (rows * cols,), MemSpace.SHARED, hint='s',
                   identity='w', permutable=need)
    lane = b.thread_id('x')
    value = b.op('mul', REAL, 2.0, 3.0)
    with b.for_(0, cols, 1, unroll=True) as hop:
        b.store(tile, value,
                b.op('add', INDEX, lane,
                     b.op('mul', INDEX, hop.induction, rows)))
    read(b, tile, lane)
    return b.finish()


def broadcast(unroll):
    """The second operand of a product: one address for every lane, the
    reduction index running along it."""
    def read(b, tile, lane):
        with b.for_(0, COLS, 1, unroll=unroll) as k:
            b.load(tile, b.op('add', INDEX, k.induction, 3 * ROWS))
    return read


def column(b, tile, lane):
    """The classic column read: lane `l` at row `l`, 32 lanes in one bank."""
    with b.for_(0, COLS, 1, unroll=True) as k:
        b.load(tile, b.op('add', INDEX, b.op('mul', INDEX, lane, ROWS),
                          k.induction))


@pytest.mark.parametrize("unroll", [True, 4, False])
def test_an_operand_every_lane_reads_alike_stays_plain(unroll):
    """Nothing to spread: one bank cycle for a broadcast whatever the layout.
    Unrolled the permutation would be folded away; rolled it would be
    arithmetic in every trip -- either way no reason to have it."""
    got = banks.propose(body_with(broadcast(unroll)))
    assert got['w'].swizzle is None
    assert got['w'].unread == ''
    assert got['w'].plain == got['w'].permuted


def test_a_column_read_is_permuted():
    """32-way plain; a permutation 32 wide takes it to one cycle, and the
    arithmetic it costs per access is far less than the 31 cycles it saves."""
    got = banks.propose(body_with(column))['w']
    assert got.swizzle == XorSwizzle(32)
    assert got.permuted < got.plain


def test_a_rolled_column_read_still_pays():
    """The permutation's arithmetic is counted against what it saves, not
    instead of it."""
    def read(b, tile, lane):
        with b.for_(0, COLS, 1, unroll=False) as k:
            b.load(tile, b.op('add', INDEX, b.op('mul', INDEX, lane, ROWS),
                              k.induction))
    got = banks.propose(body_with(read))['w']
    assert got.swizzle is not None


def test_the_granule_covers_the_widest_access():
    """A two-wide read uses its index once, so the permutation has to move
    pairs, or the second element would be read from where it was not
    written."""
    def read(b, tile, lane):
        pair = ScalarType(F32, 2)
        with b.for_(0, COLS // 2, 1, unroll=True) as k:
            b.load(tile, b.op('add', INDEX, b.op('mul', INDEX, lane, ROWS),
                              b.op('mul', INDEX, k.induction, 2)),
                   type_=pair, align=8)
    got = banks.propose(body_with(read))['w']
    assert got.swizzle is not None and got.swizzle.granule == 2


def test_the_writer_need_is_kept():
    """A transfer that moves four elements at once needs a granule of four,
    whatever its readers take."""
    got = banks.propose(body_with(column, need=4))['w']
    assert got.swizzle is None or got.swizzle.granule >= 4


def test_one_multiplication_is_read_over_its_own_lanes():
    """Eight lanes a multiplication: the column read reaches eight rows, an
    eight-way conflict at most, and the lanes past them are another
    multiplication's, in a window of its own."""
    got = banks.propose(body_with(column), lanes=8)['w']
    assert got.swizzle is not None
    assert got.swizzle.width >= 8


def test_an_index_from_a_table_is_read():
    """The column an ELL gather takes comes from a table of the generated
    file; with its numbers at hand the address is known per lane."""
    def read(b, tile, lane):
        col = b.rawexpr('tab[{0}]', lane, type_=INDEX, pure=True,
                        movable=True, hint='col')
        with b.for_(0, COLS, 1, unroll=True) as k:
            b.load(tile, b.op('add', INDEX, b.op('mul', INDEX, col, ROWS),
                              k.induction))
    unread = banks.propose(body_with(read))['w']
    assert unread.swizzle is None and unread.unread
    got = banks.propose(body_with(read), tables={'tab': list(range(32))})['w']
    assert got.unread == '' and got.swizzle is not None


def test_a_window_already_permuted_is_left_alone():
    b = IRBuilder(fptype=F32, arena='shrMem')
    tile = b.alloc(F32, (ROWS * COLS,), MemSpace.SHARED, hint='s',
                   identity='w', permutable=1, swizzle=XorSwizzle(32))
    b.load(tile, b.op('mul', INDEX, b.thread_id('x'), ROWS))
    assert banks.propose(b.finish()) == {}
