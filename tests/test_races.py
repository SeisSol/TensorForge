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


@pytest.mark.parametrize('name', ['chain_three', 'temp_dead_slice_reassign'])
def test_every_barrier_of_a_generated_kernel_is_there_for_a_reason(name):
    """Take any one out and the oracle finds what it was ordering -- the
    barrier placement puts none that nothing needs, and the oracle misses
    none that is gone.

    All but the last: that one ends the persistent loop's body and is there
    whatever the body does (`Generator._generate_bound`), so where nothing
    crosses the back edge it orders nothing."""
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
