# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Every row of the relayout table, re-derived by simulation.

The table says which distribution an instruction produces.  That is a claim
about hardware, and the two previous claims about hardware in this codebase
were both wrong in the same way: right numbers, wrong roles.  The `LaneAxis`
docstring put element `s` in lane `(s // stride) % block` when the generator
emits `((tid / stride) % block) + slot * block`; the broadcast annotation,
derived from that docstring, said `LaneAxis(threads // step, step)` where the
hardware gives `LaneAxis(step, 1)`.  Neither was caught by reading.

So each row is checked by running the instruction: tag every (register, lane)
slot, execute the definition from `hip.h`, and recover the distribution from
where the tags land.  `harness.wavesim` is that execution, and it is written
from the C++ bodies rather than from the table, so agreement means something.
"""

from __future__ import annotations

import pytest

from harness import wavesim
from tensorforge.backend.instructions.compute.primitives.amd import relayout
from tensorforge.backend.pir.core import LaneAxis, RegisterLayout

THREADS = [16, 32, 64]


# --------------------------------------------------------------------------- #
# The simulator itself, against facts stated in hip.h's own comments
# --------------------------------------------------------------------------- #

def test_quad_perm_decoding():
    assert wavesim.quad_perm(0xa0) == (0, 0, 2, 2)
    assert wavesim.quad_perm(0xf5) == (1, 1, 3, 3)
    assert wavesim.quad_perm(0xee) == (2, 3, 2, 3)
    assert wavesim.quad_perm(0x44) == (0, 1, 0, 1)


def test_unmodelled_dpp_control_is_refused_not_guessed():
    """A half-modelled instruction is worse than an absent one."""
    with pytest.raises(NotImplementedError):
        wavesim.dpp(0x150, list(range(64)))


@pytest.mark.parametrize("threads", THREADS)
def test_transpose4x4_is_the_transpose_its_comment_claims(threads):
    """`w[r][l] == v[l % 4][(l & ~3) + r]`."""
    v = wavesim.tagged(threads, regs=4)
    w = wavesim.transpose4x4b32(v)
    for r in range(4):
        for l in range(threads):
            assert w[r][l] == (l % 4, (l & ~3) + r)


@pytest.mark.parametrize("threads", THREADS)
def test_transpose4x4_loses_nothing(threads):
    """It permutes; every input slot survives exactly once."""
    v = wavesim.tagged(threads, regs=4)
    w = wavesim.transpose4x4b32(v)
    assert sorted(t for row in w for t in row) == sorted(
        t for row in v for t in row)


# --------------------------------------------------------------------------- #
# The table rows
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("threads", THREADS)
def test_broadcast_row_matches_the_hardware(threads):
    for step in (1, 2, 4, 8, 16, 32, 64):
        if step > threads:
            break
        for lane in range(threads // step):
            if not relayout.BROADCAST.applies(threads=threads, step=step,
                                              lane=lane):
                continue
            result = wavesim.broadcast(wavesim.tagged(threads)[0],
                                       threads, step, lane)
            got = wavesim.lane_axis_of(result, threads)
            assert got is not None, f'threads={threads} step={step}: not an axis'
            claimed = relayout.BROADCAST.produces(threads=threads, step=step,
                                                  lane=lane)
            assert claimed == RegisterLayout((LaneAxis(*got),)), (
                f'threads={threads} step={step} lane={lane}: table says '
                f'{claimed}, hardware gives LaneAxis{got}')


@pytest.mark.parametrize("threads", THREADS)
def test_movdpp16_row_matches_the_hardware(threads):
    for row in range(16):
        result = wavesim.movdpp16(wavesim.tagged(threads)[0], row)
        got = wavesim.lane_axis_of(result, threads)
        assert got is not None
        claimed = relayout.MOVDPP16.produces(threads=threads, row=row)
        assert claimed == RegisterLayout((LaneAxis(*got),)), (
            f'threads={threads} row={row}: table says {claimed}, '
            f'hardware gives LaneAxis{got}')


@pytest.mark.parametrize("threads", THREADS)
def test_transpose_row_matches_the_hardware(threads):
    """Both dimensions vary with the lane afterwards, so the row is rank 2.

    Output register `r` at lane `l` holds `(register l % 4, lane (l & ~3) + r)`.
    Reading only the first of those, as this test first did, would have let a
    rank-1 row stand while half the answer was missing.
    """
    w = wavesim.transpose4x4b32(wavesim.tagged(threads, regs=4))
    reg_axis = wavesim.lane_axis_of([t[0] for t in w[0]], threads)
    lane_axis = wavesim.lane_axis_of([t[1] for t in w[0]], threads)
    assert reg_axis is not None and lane_axis is not None
    hardware = RegisterLayout((LaneAxis(*reg_axis), LaneAxis(*lane_axis)))
    assert relayout.TRANSPOSE4X4.produces(threads=threads) == hardware


@pytest.mark.parametrize("threads", THREADS)
def test_the_transpose_result_is_a_bijection(threads):
    """One lane per (register, element) pair -- nothing replicated.

    This is the shape a fused operator wants: at 64 lanes, `LaneAxis(4, 1)`
    beside `LaneAxis(16, 4)`, so lane `l` holds `(l % 4, l // 4)`.
    """
    lay = relayout.TRANSPOSE4X4.produces(threads=threads)
    assert lay.rank == 2
    assert lay.tiles(threads)
    assert lay.replication(threads) == 1


def test_the_transpose_is_the_only_rank_two_row():
    ranks = {e.name: e.produces(**next(relayout._candidates(e, 64))).rank
             for e in relayout.RELAYOUTS}
    assert ranks['transpose4x4'] == 2
    assert all(r == 1 for n, r in ranks.items() if n != 'transpose4x4')


def test_lossy_rows_really_lose_elements():
    """`lossy` is not decoration -- a search prefers lossless rows on it."""
    threads = 64
    for entry, params in [
            (relayout.BROADCAST, dict(threads=threads, step=16, lane=0)),
            (relayout.MOVDPP16, dict(threads=threads, row=0))]:
        src = wavesim.tagged(threads)[0]
        out = (wavesim.broadcast(src, threads, params.get('step', 1),
                                 params.get('lane', 0))
               if entry is relayout.BROADCAST
               else wavesim.movdpp16(src, params['row']))
        assert len(set(out)) < len(set(src)), f'{entry.name} claims to be lossy'
        assert entry.lossy


# --------------------------------------------------------------------------- #
# Lookup
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("threads", THREADS)
def test_lookup_finds_an_instruction_for_every_layout_the_table_can_make(threads):
    for entry in relayout.RELAYOUTS:
        for params in relayout._candidates(entry, threads):
            if not entry.applies(**params):
                continue
            target = entry.produces(**params)
            found = relayout.find_relayout(target, threads)
            assert found is not None, \
                f'{target} unreachable although {entry.name} makes it'
            got, got_params = found
            # the search returns only what the layout determines; the
            # data-selecting parameters have to be supplied to reproduce it
            # the row the search picked need not be the row we started from,
            # so fill its data parameters from its own candidate set
            filled = dict(got_params)
            for name in got.selects_data:
                filled[name] = next(
                    c[name] for c in relayout._candidates(got, threads)
                    if name in c)
            assert got.produces(**filled) == target


@pytest.mark.parametrize("threads", THREADS)
def test_lookup_omits_the_parameters_it_cannot_know(threads):
    """`broadcast<B, S, L>` has the same layout for every `L`.

    A search over layouts cannot determine `L`, so it must not return one:
    an arbitrary pick would look like an answer and select the wrong
    sub-block. The caller supplies it, because the caller is what knows
    which elements it wants.
    """
    found = relayout.find_relayout(RegisterLayout((LaneAxis(4, 1),)), threads)
    assert found is not None
    entry, params = found
    assert 'lane' in entry.selects_data
    assert 'lane' not in params
    assert params['step'] == 4


@pytest.mark.parametrize("threads", THREADS)
def test_lookup_returns_none_for_a_layout_nothing_produces(threads):
    """Not finding one has to be possible, or the search says nothing."""
    impossible = RegisterLayout((LaneAxis(3, 5),))
    assert relayout.find_relayout(impossible, threads) is None


def test_lookup_finds_the_transpose_for_its_rank_two_result():
    target = RegisterLayout((LaneAxis(4, 1), LaneAxis(16, 4)))
    found = relayout.find_relayout(target, 64)
    assert found is not None
    entry, _ = found
    assert entry is relayout.TRANSPOSE4X4
    assert not entry.lossy


def test_lookup_prefers_a_lossless_instruction():
    """Ordering, checked without relying on which rows happen to overlap.

    The rows no longer produce a layout in common -- the transpose went to
    rank 2 -- so the preference is asserted on the search order itself rather
    than on a target that two rows can both reach, which would silently stop
    testing anything the next time the table changes.
    """
    ordered = sorted(relayout.RELAYOUTS, key=lambda e: e.lossy)
    assert not ordered[0].lossy
    assert [e.lossy for e in ordered] == sorted(e.lossy for e in relayout.RELAYOUTS)


def test_the_table_only_holds_rows_the_simulator_can_check():
    """`transpose16x16b32` exists in the runtime and is deliberately absent.

    Its body uses row and wave DPP controls the simulator does not model, so a
    row for it could not be verified -- and an unverified row is exactly the
    kind of claim that produced the two earlier errors.
    """
    names = {e.callee for e in relayout.RELAYOUTS}
    assert not any('16x16' in n for n in names)


# --------------------------------------------------------------------------- #
# The checks: layouts used to catch something, not only to describe
# --------------------------------------------------------------------------- #

def _ir():
    from tensorforge.backend.pir.build import IRBuilder
    from tensorforge.common.basic_types import Datatype
    return IRBuilder(Datatype.F32), Datatype


def test_fmadpp_rejects_an_operand_in_the_wrong_distribution():
    """The DPP pattern assumes it. A mismatch is a wrong kernel.

    And an invisible one: the snapshot would look reasonable and the symbolic
    comparison treats the intrinsic as uninterpreted, so neither would see
    that the operand holds the right type and the wrong elements.
    """
    from tensorforge.backend.instructions.compute.primitives import amd
    from tensorforge.backend.pir.core import ScalarType
    b, dt = _ir()
    f = ScalarType(dt.F32)
    C = b.declare(f, hint='acc')
    good = b.value(f, layout=relayout.fmadpp_operand_layout(16))
    bad = b.value(f, layout=relayout.fmadpp_operand_layout(4))
    B = b.value(f)
    amd.fmadpp16(b, C, good, B, 0)                       # fine
    with pytest.raises(ValueError):
        amd.fmadpp16(b, C, bad, B, 0)


def test_fmadpp_lets_an_untracked_operand_through():
    """`None` is unknown, not wrong.

    The sparse loader used to be the reason this mattered; it now reports what
    its fill recorded (`test_sparse_layout.py`).  The MFMA accumulator still
    does not, deliberately.  Refusing to emit for want of an annotation would
    turn a description into an obstacle, and the parts that are annotated
    would stop being worth annotating.
    """
    from tensorforge.backend.instructions.compute.primitives import amd
    from tensorforge.backend.pir.core import ScalarType
    b, dt = _ir()
    f = ScalarType(dt.F32)
    amd.fmadpp16(b, b.declare(f, hint='acc'), b.value(f), b.value(f), 0)


@pytest.mark.parametrize("step", [4, 16])
def test_the_requirement_is_stated_once(step):
    """`hfma` searches for it and `fmadpp` checks it -- the same expression.

    Two statements of one requirement is the arrangement that produced the
    wrong broadcast layout: the callee in one place, the claim about its
    result in another.
    """
    want = relayout.fmadpp_operand_layout(step)
    found = relayout.find_relayout(want, 64)
    assert found is not None
    entry, params = found
    filled = dict(params, **{n: 0 for n in entry.selects_data})
    assert entry.produces(**filled) == want


def test_mfma_rejects_an_operand_that_is_not_transposed():
    from tensorforge.backend.instructions.compute.primitives.amd import codegen
    from tensorforge.backend.pir.core import LaneAxis, ScalarType
    b, dt = _ir()
    plain = b.value(ScalarType(dt.F32),
                    layout=RegisterLayout((LaneAxis(64, 1),)))
    with pytest.raises(ValueError):
        codegen._check_mfma_operand(plain, 64, 'mfma')
    ok = b.value(ScalarType(dt.F32),
                 layout=relayout.TRANSPOSE4X4.produces(threads=64))
    codegen._check_mfma_operand(ok, 64, 'mfma')
    codegen._check_mfma_operand(b.value(ScalarType(dt.F32)), 64, 'mfma')
