# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Which lanes of a value anything reads (`pir.demand`), and the loads the
forward pass takes because the lanes it would give something else than the
load are lanes nothing reads (`pir.forward`).

The lane maps are claims about hardware, and they are run the way
`test_amd_relayout.py` runs the relayout table: every lane tagged, the
instruction executed from its definition (`harness.wavesim`), and every
lane's result has to be the tag of the lane the map names.
"""

from __future__ import annotations

import pytest

from harness import wavesim
from test_forward import F32, LANES, REAL, Body, _loads

from tensorforge.backend.pir.core import (BOOL, Access, Effect, MemSpace,
                                          walk_stmts)
from tensorforge.backend.pir.demand import Demand, lane_map
from tensorforge.backend.pir.flow import Flow

THREADS = [16, 32, 64]


def _mapped(threads, block, lane):
    """What every lane of `threads` reads under the map `lane` of `block`:
    the tag of that lane."""
    tags = wavesim.tagged(threads)[0]
    return [tags[p - p % block + lane(p % block)] for p in range(threads)]


# -- the maps, against the simulation ----------------------------------------

@pytest.mark.parametrize('threads', THREADS)
def test_a_broadcast_reads_the_lane_the_simulation_takes(threads):
    for block in (2, 4, 8, 16, 32, 64):
        if block > threads:
            break
        for sub in (s for s in (1, 2, 4, 8, 16, 32) if block % s == 0
                    and s <= block):
            for lane in range(block // sub):
                operand, of_result, b, f = lane_map(
                    f'tensorforge::broadcast<{block}, {sub}, {lane}>')
                assert (operand, of_result, b) == (0, True, block)
                assert wavesim.broadcast(wavesim.tagged(threads)[0], block,
                                         sub, lane) == _mapped(threads, block,
                                                               f)


@pytest.mark.parametrize('threads', THREADS)
def test_a_row_share_reads_the_lane_the_simulation_takes(threads):
    for row in range(16):
        for callee in (f'tensorforge::movdpp16<{row}>',
                       f'tensorforge::movdpp16Kept<{row}>'):
            operand, of_result, b, f = lane_map(callee)
            assert (operand, of_result, b) == (0, True, 16)
            assert wavesim.movdpp16(wavesim.tagged(threads)[0],
                                    row) == _mapped(threads, 16, f)


@pytest.mark.parametrize('threads', THREADS)
@pytest.mark.parametrize('width', [4, 8, 16])
def test_a_dpp_fma_multiplies_with_the_lane_its_control_names(threads, width):
    """`FMADPP4` is `quad_perm:[pos,pos,pos,pos]`, `FMADPP8` `dpp8` with
    `pos` eight times, `FMADPP16` the row share `movdpp16` is: what each
    moves the operand `a` to is what the map says the FMA reads."""
    for row in range(width):
        operand, of_result, b, f = lane_map(f'tensorforge::fmacdpp{width}<{row}>')
        assert (operand, of_result, b) == (1, False, width)
        tags = wavesim.tagged(threads)[0]
        moved = (wavesim.dpp(row * 0x55, tags) if width == 4
                 else wavesim.dpp8([row] * 8, tags) if width == 8
                 else wavesim.movdpp16(tags, row))
        assert moved == _mapped(threads, width, f)


def test_a_primitive_without_a_row_has_no_map():
    for callee in ('tensorforge::transpose4x4b32', 'tensorforge::swap<16>',
                   'tensorforge::swizzle<31, 0, 16>', 'tensorforge::readlane',
                   'tensorforge::broadcast_warp', None):
        assert lane_map(callee) is None


# -- the lanes a value's uses read --------------------------------------------

def _rows(lane, threads=LANES, width=16):
    """The mask of lane `lane` of every row of `width`."""
    return sum(1 << (r + lane) for r in range(0, threads, width))


class _Use(Body):
    """`x`, a value of every lane, and `y`, another; `acc`, an accumulator
    an FMA under a DPP control adds into, stored where the body keeps it."""

    def __init__(self):
        super().__init__()
        b = self.b
        self.x = b.op('mul', REAL, 2.0, b.load(self.p, self.lane), hint='x')
        self.y = b.op('mul', REAL, 3.0, b.load(self.p, self.lane), hint='y')
        self.acc = b.declare(REAL, hint='acc')

    def fma(self, row, a, b_=None, width=16):
        self.b.call_stmt(f'tensorforge::fmacdpp{width}<{row}>', self.acc, a,
                         self.y if b_ is None else b_, writes=(self.acc,))

    def lanes(self):
        self.b.store(self.p, self.acc, self.lane)
        body = self.b.finish()
        flow = Flow(body, LANES, 'shrMem', values=True, before=False)
        return Demand(flow).lanes(self.x, {})


def test_an_fma_under_a_dpp_control_reads_one_lane_a_row():
    t = _Use()
    t.fma(3, t.x)
    assert t.lanes() == _rows(3)


def test_the_lanes_of_two_fmas_add_up():
    t = _Use()
    t.fma(3, t.x)
    t.fma(4, t.x, width=8)
    assert t.lanes() == _rows(3) | _rows(4, width=8)


def test_the_operand_an_fma_multiplies_its_lane_with_reads_what_the_sum_is_read_in():
    """`acc += a[row] * x`: lane `p` of `x` goes into lane `p` of `acc`,
    which is stored where the body keeps it -- every lane -- or read back by
    an FMA that takes one lane a row of it."""
    t = _Use()
    t.fma(3, t.y, t.x)
    assert t.lanes() == (1 << LANES) - 1
    t = _Use()
    t.fma(3, t.y, t.x)
    r = t.registers('r0', 1)
    t.b.store(r, t.acc, 0)
    other = t.b.declare(REAL, hint='acc2')
    t.b.call_stmt('tensorforge::fmacdpp16<5>', other, t.b.load(r, 0), t.y,
                  writes=(other,))
    t.acc = other
    assert t.lanes() == _rows(5)


def test_the_operand_an_fma_moves_reads_the_rows_of_what_the_sum_is_read_in():
    """`acc += x[row] * y`, `acc` read back by an FMA that takes lane 5
    of every row of it: of `x`, lane `row` of the rows lane 5 is in."""
    t = _Use()
    t.fma(3, t.x)
    r = t.registers('r0', 1)
    t.b.store(r, t.acc, 0)
    other = t.b.declare(REAL, hint='acc2')
    t.b.call_stmt('tensorforge::fmacdpp16<5>', other, t.b.load(r, 0), t.y,
                  writes=(other,))
    t.acc = other
    assert t.lanes() == _rows(3)


def test_an_accumulator_read_by_what_it_does_not_know_reads_every_lane():
    t = _Use()
    t.fma(3, t.y, t.x)
    t.b.call_stmt('tensorforge::pin', t.acc, writes=(t.acc,))
    t.b.store(t.registers('r0', 1), t.acc, 0)
    t.acc = t.b.declare(REAL, hint='unused')
    assert t.lanes() == (1 << LANES) - 1


def test_a_broadcast_hands_on_the_lanes_its_result_is_read_in():
    t = _Use()
    bc = t.b.call('tensorforge::broadcast<32, 16, 1>', REAL, t.x,
                  movable=False)
    t.fma(3, bc)
    assert t.lanes() == 1 << 19


def test_the_broadcast_spelled_as_text_is_the_broadcast():
    t = _Use()
    bc = t.b.rawexpr('tensorforge::broadcast<32, 1, 1>({0})', t.x,
                     crosslane=True, pure=True)
    t.fma(3, bc)
    assert t.lanes() == 1 << 1


def test_arithmetic_reads_the_lanes_its_result_is_read_in():
    t = _Use()
    t.fma(5, t.b.op('add', REAL, t.x, t.y))
    assert t.lanes() == _rows(5)


def test_a_register_hands_on_the_lanes_its_load_back_is_read_in():
    t = _Use()
    r = t.registers('r0', 1)
    t.b.store(r, t.x, 0)
    t.fma(6, t.b.load(r, 0))
    assert t.lanes() == _rows(6)


def test_a_register_read_by_nothing_reads_no_lane():
    t = _Use()
    t.b.store(t.registers('r0', 1), t.x, 0)
    t.fma(6, t.y)
    assert t.lanes() == 0


@pytest.mark.parametrize('use', ['memory', 'unknown', 'predicate', 'text',
                                 'opaque'])
def test_what_it_does_not_know_reads_every_lane(use):
    t = _Use()
    b = t.b
    if use == 'memory':
        b.store(t.q, t.x, t.lane)
    elif use == 'opaque':
        r = t.registers('r0', 1)
        b.store(r, t.x, 0)
        b.touches([Access(Effect.READ, MemSpace.REGISTER, base=r)])
    elif use == 'unknown':
        t.fma(2, b.call('tensorforge::swizzle<31, 0, 16>', REAL, t.x,
                        movable=False))
    elif use == 'predicate':
        cond = b.op('lt', BOOL, t.x, 1.0)
        t.fma(2, b.load(t.p, t.lane, predicate=cond))
    else:
        t.fma(2, b.rawexpr('__shfl_xor_sync(~0u, {0}, 1)', t.x,
                           crosslane=True))
    assert t.lanes() == (1 << LANES) - 1


# -- what the forward pass takes ----------------------------------------------

def _handed_to_an_fma(row, predicated=False):
    """`r` computed on every lane, its rows below 20 stored into `s` and
    loaded back into `back` by those lanes, the rest of `back` its zeros --
    no promise of what `r` holds past row 20 -- and `back` the operand an FMA
    under a DPP control takes from lane `row` of every row of sixteen."""
    t = _Use()
    b = t.b
    s = b.alloc(F32, (LANES,), MemSpace.SHARED, hint='s', extern='s0',
                arena='shrMem')
    r = t.registers('r0', 1)
    b.store(r, b.op('mul', REAL, 2.0, b.load(t.p, t.lane)), 0)
    rows = b.op('lt', BOOL, t.lane, 20, hint='g')
    with b.if_(rows):
        b.store(s, b.load(r, 0), t.lane)
    back = t.registers('r1', 1)
    if predicated:
        b.store(back, b.load(s, t.lane, predicate=rows), 0)
    else:
        with b.if_(rows):
            b.store(back, b.load(s, t.lane), 0)
    t.fma(row, b.load(back, 0))
    b.store(t.p, t.acc, t.lane)
    return t


def _selects(body):
    return [x for x in walk_stmts(body) if x.op == 'select']


@pytest.mark.parametrize('predicated', [False, True])
def test_lanes_nothing_reads_take_the_register_as_the_rest(predicated):
    """The FMA reads lanes 3 and 19, both rows `r` stored: the load back is
    `r`, on every lane, though `r` holds no zero past row 20."""
    t = _handed_to_an_fma(3, predicated)
    out = t.run()
    assert not _selects(out)
    assert not _loads(out, MemSpace.SHARED)
    assert not [x for x in walk_stmts(out) if x.predicate is not None
                and x.accesses and x.accesses[0].space is MemSpace.REGISTER]


@pytest.mark.parametrize('predicated', [False, True])
def test_a_lane_read_past_the_rows_keeps_the_zero(predicated):
    """Lane 21, past the rows, is read as well: there the load gives the
    zero `r` does not hold, and a select or a predicate says so."""
    t = _handed_to_an_fma(5, predicated)
    out = t.run()
    assert _selects(out) or [x for x in walk_stmts(out)
                             if x.predicate is not None and x.accesses
                             and x.accesses[0].space is MemSpace.REGISTER]
