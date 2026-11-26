# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""The host oracle's pool regions (`cuda.h`'s `warpsApart` and `ticketTake`,
`kernel_eval.Lockstep`): the warps of a region each a coroutine of their
own, their tickets one fetch-add a warp in the order a schedule seeds, the
race tracker watching them as it watches the roles -- what the high-order
tile's mixer pool runs on.  Every kernel without a region runs as before."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from tensorforge.reference import kernel_eval as ke

INCLUDE = Path(__file__).resolve().parents[1] / 'src' / 'tensorforge' / 'include'

#: Three warps of a block, a multiplication each (`evaluate_wave`'s `mults`).
WARPS = 3


def _head(rest: str) -> str:
    """A block whose first thread zeroes the ticket counter and sets up
    three mbarriers of 32 arrivals, ahead of the block's barrier, then
    `rest`."""
    return ('void k(float* m0, const float* m1) {\n'
            '  auto* shrMem = reinterpret_cast<float*>(totalShrMemPtr);\n'
            '  int32_t* ctr = reinterpret_cast<int32_t*>(shrMem);\n'
            '  uint64_t* bar = reinterpret_cast<uint64_t*>(shrMem + 8);\n'
            '  float* box = shrMem + 16;\n'
            '  if (threadIdx.x == 0 && threadIdx.y == 0) {\n'
            '    ctr[0] = 0;\n'
            '    tensorforge::mbarrierInit(&bar[0], 32);\n'
            '    tensorforge::mbarrierInit(&bar[1], 32);\n'
            '    tensorforge::mbarrierInit(&bar[2], 32);\n'
            '  }\n'
            '  __syncthreads();\n' + rest + '}\n')


#: Twelve tickets over the warps, each writing its warp's number where its
#: ticket says, and each warp how many it took.
TICKETS = _head(
    '  if (tensorforge::warpsApart()) {\n'
    '    float taken = 0.0f;\n'
    '    int32_t t = tensorforge::ticketTake(&ctr[0]);\n'
    '    while (t < 12) {\n'
    '      m0[t * 32 + threadIdx.x] = threadIdx.y;\n'
    '      taken = taken + 1.0f;\n'
    '      t = tensorforge::ticketTake(&ctr[0]);\n'
    '    }\n'
    '    m0[400 + threadIdx.y * 32 + threadIdx.x] = taken;\n'
    '  }\n')


def _chain(region: str) -> str:
    """Three tickets, a hand-off from each to the next through a slot of
    shared memory and an mbarrier: ticket t waits for t - 1's arrival, adds
    its column of `m1` to what t - 1 left and arrives in turn -- every warp
    at the same statements, each at its own barrier."""
    return _head(
        f'  if ({region}) {{\n'
        '    int32_t t = tensorforge::ticketTake(&ctr[0]);\n'
        '    while (t < 3) {\n'
        '      if (t > 0) {\n'
        '        tensorforge::mbarrierWait(&bar[t - 1], 0);\n'
        '      }\n'
        '      box[t * 32 + threadIdx.x] = (t > 0 ? box[(t - 1) * 32 + '
        'threadIdx.x] : 0.0f) + m1[t * 32 + threadIdx.x];\n'
        '      m0[t * 32 + threadIdx.x] = box[t * 32 + threadIdx.x];\n'
        '      tensorforge::mbarrierArrive(&bar[t], true);\n'
        '      t = tensorforge::ticketTake(&ctr[0]);\n'
        '    }\n'
        '  }\n')


def _run(src: str, schedule=None, seed: int = 0):
    races = []
    mem = ke.evaluate_wave(src, 32, seed=seed, globals_only=True,
                           races=races, mults=WARPS, block=True,
                           schedule=schedule)
    return mem, races


def _owners(mem, tickets=12):
    """The warp each ticket went to, and how many each warp took."""
    owners = [mem.get(('m0', t * 32)) for t in range(tickets)]
    taken = [mem.get(('m0', 400 + w * 32)) for w in range(WARPS)]
    assert None not in owners and set(owners) <= set(range(WARPS))
    assert taken == [owners.count(w) for w in range(WARPS)]
    return owners


def test_every_ticket_once_in_the_warps_order():
    """Without a schedule the warps take their turns in their own order, a
    turn ending behind each ticket: tickets 0 to 11 round the three warps,
    each exactly once, every lane of a warp its warp's -- no race."""
    mem, races = _run(TICKETS)
    assert races == []
    assert _owners(mem) == [t % WARPS for t in range(12)]
    assert all(mem.get(('m0', t * 32 + lane)) == t % WARPS
               for t in range(12) for lane in range(32))


def test_the_schedule_seeds_the_warps_turns():
    """A schedule shuffles each round of the warps' turns: every seed hands
    out every ticket exactly once, the seeds' orders not all one, and a seed
    again its own order."""
    orders = {}
    for seed in range(1, 7):
        mem, races = _run(TICKETS, schedule=seed)
        assert races == []
        orders[seed] = tuple(_owners(mem))
    assert len(set(orders.values())) > 1
    again, _ = _run(TICKETS, schedule=3)
    assert tuple(_owners(again)) == orders[3]


@pytest.mark.parametrize('schedule', [None, 1, 2, 5])
def test_a_hand_off_chain_between_the_warps_of_a_region(schedule):
    """Three tickets handing a sum on through shared memory and mbarriers:
    in a pool region each warp waits at its own barrier while the one ahead
    goes on, the sums every column's of `m1` so far, no race, whatever the
    schedule."""
    mem, races = _run(_chain('tensorforge::warpsApart()'), schedule)
    assert races == []
    fill, total = ke.Slot(0), [0.0] * 32
    for t in range(3):
        for lane in range(32):
            total[lane] += fill.read('m1', t * 32 + lane)
            assert mem.get(('m0', t * 32 + lane)) == pytest.approx(
                total[lane], rel=1e-6)


def test_the_chain_hangs_where_the_warps_run_together():
    """The same chain outside a pool region: the warps of the `if (t > 0)`
    run together, the second's wait holding the first's back -- the hang
    the oracle refuses, which a region of warps apart is for."""
    with pytest.raises(ke.Abort, match='would never return'):
        _run(_chain('true'))


def test_a_region_through_a_variable():
    """The region's mark kept in a variable, as an emitter that declares
    its conditions writes it: the same coroutines."""
    src = _chain('apart').replace(
        '  if (apart) {\n',
        '  bool apart = tensorforge::warpsApart();\n  if (apart) {\n')
    mem, races = _run(src, schedule=2)
    assert races == []
    assert mem.get(('m0', 2 * 32)) is not None


def test_a_race_between_the_warps_of_a_region_is_found():
    """Each ticket's warp writing one slot of shared memory, nothing
    ordering the writes: the tracker reports it, as it would between
    roles."""
    src = _head(
        '  if (tensorforge::warpsApart()) {\n'
        '    int32_t t = tensorforge::ticketTake(&ctr[0]);\n'
        '    while (t < 3) {\n'
        '      box[threadIdx.x] = t;\n'
        '      t = tensorforge::ticketTake(&ctr[0]);\n'
        '    }\n'
        '  }\n')
    _, races = _run(src)
    assert any(r.kind == 'WAW' for r in races)


def test_a_counter_set_up_unordered_is_a_race():
    """The counter zeroed by one thread without the block's barrier behind
    it: the tickets' reads of it race the write -- an atomic's increment
    races no other warp's, its read of what set the counter up does."""
    src = TICKETS.replace('  __syncthreads();\n', '')
    _, races = _run(src)
    assert any(r.kind == 'RAW' for r in races)


def test_a_ticket_asks_for_the_whole_warp():
    """`ticketTake`'s shuffle has every lane of the warp take part: a warp
    some of whose lanes reach it is an `Abort`."""
    src = _head(
        '  if (threadIdx.x < 16) {\n'
        '    int32_t t = tensorforge::ticketTake(&ctr[0]);\n'
        '  }\n')
    with pytest.raises(ke.Abort, match='ticketTake in 16 lanes of warp'):
        _run(src)


def test_a_region_every_lane_enters():
    """A region some lanes do not enter is an `Abort`: its warps are the
    region's arms, whole."""
    src = _head(
        '  if (threadIdx.x < 16 && tensorforge::warpsApart()) {\n'
        '    m0[threadIdx.x] = 1.0f;\n'
        '  }\n')
    with pytest.raises(ke.Abort, match='a pool region some lanes do not '
                                       'enter'):
        _run(src)


def test_the_helpers_build(tmp_path):
    """`warpsApart` and `ticketTake` built for sm_100a: a kernel taking
    tickets in a region compiles."""
    nvcc = shutil.which('nvcc')
    if nvcc is None:
        pytest.skip('nvcc not found')
    src = tmp_path / 'pool.cu'
    src.write_text(
        '#include "tensorforge_device/cuda.h"\n'
        '__global__ void k(float *out) {\n'
        '  __shared__ int32_t ctr[1];\n'
        '  if (threadIdx.x == 0) ctr[0] = 0;\n'
        '  __syncthreads();\n'
        '  if (tensorforge::warpsApart()) {\n'
        '    int32_t t = tensorforge::ticketTake(&ctr[0]);\n'
        '    out[t * 32 + (threadIdx.x & 31)] = t;\n'
        '  }\n'
        '}\n')
    run = subprocess.run([nvcc, '-std=c++17', '-arch=sm_100a', '-c', '-o',
                          str(tmp_path / 'pool.o'), '-I', str(INCLUDE),
                          str(src)], capture_output=True, text=True,
                         timeout=600)
    if run.returncode and 'sm_100a' in run.stderr and 'not defined' in \
            run.stderr:
        pytest.skip('nvcc builds no sm_100a')
    assert run.returncode == 0, run.stderr[-2000:]
