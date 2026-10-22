# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""The host oracle says where the hardware would not have ordered two lanes.

The interpreter runs its lanes in lockstep, so a missing barrier never
changes a value there: a reader always finds the write done.  What
`kernel_eval.Races` adds is the question the hardware asks -- were the two
lanes' accesses of one slot separated by a barrier both took part in -- and
for an asynchronous copy, whether the slot is still in flight.

A check like this is worth its runtime only if it fails where it should, so
most of what follows takes a barrier out of a generated kernel and expects
the race it was there to prevent.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import re
from pathlib import Path

import pytest

from tensorforge.common.context import Context
from tensorforge.common.options import Options
from tensorforge.generators.generator import Generator
from tensorforge.reference import kernel_eval as ke

CASES = Path(__file__).resolve().parent / "cases"
_SYNC = re.compile(r'__syncwarp|__syncthreads|barrier\.sync')


def _kernel(case: str, **options):
    path = next(CASES.rglob(f'{case}.py'))
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ctx = Context(arch='sm_86', backend='cuda',
                  fp_type=getattr(mod, 'DTYPE', None),
                  options=Options(**options))
    gen = Generator(mod.descr_list(), ctx)
    with contextlib.redirect_stdout(io.StringIO()):
        gen.generate()
    return gen.get_kernel(), ke.launch_geometry(gen.get_launcher())


def _races(src, lanes, elements=2):
    found = []
    ke.evaluate_wave(src, lanes, seed=3, races=found, elements=elements)
    return found


@pytest.mark.parametrize('name', ['chain_three', 'accumulate_then_read',
                                  'temp_slice_after_whole',
                                  'temp_dead_slice_reassign'])
def test_a_generated_kernel_is_ordered(name):
    src, (lanes, _) = _kernel(name)
    assert _races(src, lanes) == []


@pytest.mark.parametrize('name', ['chain_three'])
def test_every_barrier_of_a_generated_kernel_is_there_for_a_reason(name):
    """Take any one out and the oracle finds what it was ordering -- the
    barrier placement puts none that nothing needs, and the oracle misses
    none that is gone.

    All but the last: that one ends the persistent loop's body and is there
    whatever the body does (`Generator._generate_bound`), so where nothing
    crosses the back edge it orders nothing.

    Not `temp_dead_slice_reassign`: every row of its temporary is written and
    read in the lane of its row (`MultilinearBuilder._lead_origin_shift`),
    and the barriers between those hand-overs order nothing the oracle
    finds.  The placement asks buffers, not lanes (`pir.barriers`)."""
    src, (lanes, _) = _kernel(name)
    lines = src.split('\n')
    sync = [k for k, line in enumerate(lines) if _SYNC.search(line)]
    assert len(sync) > 1, 'no barrier placed in the kernel; nothing to test'
    for k in sync[:-1]:
        without = '\n'.join(line for j, line in enumerate(lines) if j != k)
        assert _races(without, lanes, elements=3), (
            f'nothing races without line {k}: {lines[k].strip()}')


def test_a_race_names_the_slot_the_lanes_and_the_statement():
    src, (lanes, _) = _kernel('chain_three')
    lines = src.split('\n')
    k = next(k for k, line in enumerate(lines) if _SYNC.search(line))
    without = '\n'.join(line for j, line in enumerate(lines) if j != k)
    race = _races(without, lanes)[0]
    assert race.kind in ('RAW', 'WAR', 'WAW', 'FLIGHT')
    assert race.lane != race.other or race.kind == 'FLIGHT'
    assert race.statement and str(race.slot) in str(race)


def test_every_spelling_of_a_barrier_is_one():
    races = ke.Races()
    for stmt in ('__syncthreads()', '__syncwarp(0xffff)',
                 '__builtin_amdgcn_s_barrier()',
                 'asm volatile("barrier.sync.aligned 1, 64;")',
                 'asm volatile("bar.sync 1, 64;")',
                 'cooperative_groups::this_grid().sync()'):
        assert races.statement_kind(stmt) == 'barrier', stmt
    assert races.statement_kind('__pipeline_commit()') == 'commit'
    assert races.statement_kind('__pipeline_wait_prior(1)') == 'wait'
    assert races.statement_kind('float x = s0[0]') is None


def test_a_slot_in_flight_is_touched_by_nobody():
    """Until the wait retires the copy's group, the copy may still land: a
    read of its slot is a race even from the lane that issued it."""
    races = ke.Races()
    races.lane = 0
    races.issuing = True
    races.write(7)
    races.issuing = False
    races.read(7)
    assert [r.kind for r in races.found] == ['FLIGHT']


def test_a_retired_copy_is_ordered_for_its_own_lane_only():
    races = ke.Races()
    races.lane = 0
    races.issuing = True
    races.write(7)
    races.issuing = False
    races.tick('__pipeline_commit()')
    races.commit(0)
    races.tick('__pipeline_wait_prior(0)')
    races.wait(0, '__pipeline_wait_prior(0)')
    races.read(7)
    assert races.found == []
    races.lane = 1
    races.read(7)
    assert [r.kind for r in races.found] == ['RAW']


# --------------------------------------------------------------------------- #
# whom a barrier meets
# --------------------------------------------------------------------------- #

def test_each_spelling_says_whom_it_meets():
    races = ke.Races()
    assert races.reach('__syncthreads()') == ('block',)
    assert races.reach('__syncwarp()') == ('warp', None)
    assert races.reach('__syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16))') \
        == ('warp', '0x0000ffffu << (threadIdx.y % 2 * 16)')
    assert races.reach('asm volatile("barrier.sync %0, %1;" :: '
                       '"r"(threadIdx.y + 1), "r"(64) : "memory")') \
        == ('named', 'threadIdx.y + 1', '64')
    assert races.reach('asm volatile("bar.sync 1, 64;")') \
        == ('named', '1', '64')


@pytest.mark.parametrize('lanes', [64, 96])
@pytest.mark.parametrize('name', ['chain_three', 'accumulate_then_read',
                                  'temp_dead_slice_reassign'])
def test_a_multiplication_wider_than_a_warp_is_ordered(name, lanes):
    """Its barrier is a named one, met by the warps of the multiplication
    and nobody else, and every lane of it reaches each."""
    src, (got, _) = _kernel(name, lanes_per_mult=lanes)
    assert got == lanes and 'barrier.sync' in src
    assert _races(src, got, elements=3) == []


def test_the_multiplications_of_a_warp_meet_on_their_own_masks():
    """Two multiplications share a warp and each barrier names its own half:
    run together, each reaches the barrier it waits at, and nothing races."""
    src, (lanes, mults) = _kernel('chain_three', lanes_per_mult=16)
    assert lanes == 16 and mults > 1 and '__syncwarp(0x' in src
    found = []
    ke.evaluate_wave(src, lanes, seed=3, races=found, elements=2,
                     mults=mults, block=True)
    assert found == []


def test_a_multiplication_meets_on_its_own_half_where_the_other_has_none():
    """Where the batch ends at the first multiplication of a warp, the second
    has no element and skips the body: the barrier the first reaches names
    its own half, and it waits for nobody beyond it."""
    src, (lanes, mults) = _kernel('chain_three', lanes_per_mult=16)
    assert lanes == 16 and mults > 1 and '__syncwarp(0x' in src
    found = []
    ke.evaluate_wave(src, lanes, seed=3, races=found, elements=1,
                     mults=mults, block=True)
    assert found == []


def _wide(name='chain_three', lanes=64):
    src, (got, _) = _kernel(name, lanes_per_mult=lanes)
    lines = src.split('\n')
    named = [k for k, line in enumerate(lines) if 'barrier.sync' in line]
    assert named, 'no named barrier in a multiplication wider than a warp'
    return lines, named, got


def test_a_barrier_some_lanes_of_the_multiplication_skip_is_reported():
    """Under a lane guard the lockstep run goes on without the lanes outside;
    the hardware would wait for them."""
    lines, named, lanes = _wide()
    k = named[0]
    guarded = lines[:k] + ['if (threadIdx.x < 40) {', lines[k], '}'] \
        + lines[k + 1:]
    found = _races('\n'.join(guarded), lanes)
    arrivals = [r for r in found if r.kind == 'ARRIVAL']
    assert arrivals and arrivals[0].lane == 40 and arrivals[0].other == 40, \
        found
    assert 'does not reach' in str(arrivals[0])


def test_a_count_its_warps_do_not_fill_is_reported():
    lines, named, lanes = _wide()
    half = [line.replace(f'"r"({lanes})', f'"r"({lanes // 2})')
            for line in lines]
    found = _races('\n'.join(half), lanes)
    assert [r.kind for r in found if r.kind == 'COUNT'], found


def test_a_warp_barrier_does_not_order_the_warps_of_one_multiplication():
    """`__syncwarp` executed by all 64 lanes is two rendezvous, one per warp:
    what one warp stages, the other reads unordered."""
    lines, named, lanes = _wide()
    warp = [('__syncwarp();' if k in named else line)
            for k, line in enumerate(lines)]
    found = _races('\n'.join(warp), lanes)
    across = [r for r in found if r.kind in ('RAW', 'WAR', 'WAW')
              and (r.lane < 32) != (r.other < 32)]
    assert across, found
