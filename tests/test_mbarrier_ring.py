# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""The roles of a block (`IRBuilder.roles`) and the ring of mbarriers they
hand slots over through (`IRBuilder.mbarrier_ring`): a producer fills the
slots, its consumers drain them, each slot a pair of mbarriers -- the full
one the producer's bytes complete on, the empty one the consumers arrive
at once done with the slot.  Every role counts its rounds of the ring from
its trips (`ring_round`): the parity it waits for is the round's, the
producer's on the empty side the other one, so that its first round finds
the slots free.  Inside its region a role is the block: its barriers are
its own (`Participants.ROLE`), a named barrier the other role's threads are
no members of.

What is checked here: the statements and their text (`cuda.h`'s
`mbarrierInit`, `mbarrierWait`, `mbarrierArrive`, `mbarrierExpect`,
`bulkCopyOn`), the verifier on the roles' barriers, that the barrier
placement leaves the hand-over to the ring and meets a role's hazards on
the role's barrier, the host oracle (`reference.kernel_eval`) on a
producer warp and two consumer warps -- the values, and each mistake the
protocol can make refused -- the build with the SASS read back, and on a
GPU of sm_90 or newer the body run against its input.
"""

from __future__ import annotations

import contextlib
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from tensorforge import toolchain
from tensorforge.backend.pir import engines
from tensorforge.backend.pir.allocate import MULT, allocate
from tensorforge.backend.pir.barriers import BLOCK, Arena, place_barriers
from tensorforge.backend.pir.build import (IRBuilder, RingSlot, Role,
                                           barrier_stmt)
from tensorforge.backend.pir.core import (BOOL, INDEX, IRError, MemSpace, Op,
                                          Participants, ScalarType,
                                          Uniformity, walk)
from tensorforge.backend.pir.emit import Emitter
from tensorforge.backend.pir.passes import verify
from tensorforge.common.basic_types import Datatype
from tensorforge.common.context import Context
from tensorforge.reference import kernel_eval

INCLUDE = Path(__file__).resolve().parents[1] / 'src' / 'tensorforge' / 'include'
F32 = Datatype.F32
#: Elements a trip moves: one a consumer lane, of two consumer warps.
WIDTH = 64
#: The roles' named barriers: the last two of the sixteen, apart from the
#: ids the multiplications take from 1 on.
IDENT = '(14 + (threadIdx.y >= 2))'


def _ctx(arch='sm_90a'):
    return Context(arch=arch, backend='cuda', fp_type=F32)


def _body(trips=5, slots=2, warp_sync=True, release=True, expect=True,
          tx=WIDTH * 4, producer_parity=True, exchange=True,
          role_barrier=True, block_barrier=False, cross=False):
    """Two consumer warps (`threadIdx.y` 0 and 1) and a producer warp (2)
    over a ring of `slots`, in roles of their own: each trip the producer
    waits for the slot's round on its empty side, copies the trip's `WIDTH`
    floats of `m1` into it by one bulk copy and announces their bytes on its
    full side; the consumers wait for the slot on the full side, each lane
    hands its float to the lane of the other warp through a buffer of their
    own -- `exchange`, met on their role's barrier -- writes it to `m0`,
    and each warp meets and releases the slot by its first lane.  The
    switches leave a step out, or get it wrong: the mistakes the oracle
    has to refuse."""
    b = IRBuilder(fptype=F32, arena='shrMem', context=_ctx())
    ring = b.mbarrier_ring('pc', slots, full=1, empty=2,
                           leader='threadIdx.x == 0 && threadIdx.y == 0')
    slot_buf = b.alloc(F32, (slots * WIDTH,), MemSpace.SHARED, hint='slot',
                       extern='slot', place_align=4)
    swap = b.alloc(F32, (WIDTH,), MemSpace.SHARED, hint='swap',
                   extern='swap')
    src = b.alloc(F32, (trips * WIDTH,), MemSpace.GLOBAL, hint='g',
                  extern='m1')
    out = b.alloc(F32, (trips * WIDTH,), MemSpace.GLOBAL, hint='o',
                  extern='m0')
    x, y = b.thread_id('x'), b.thread_id('y')
    lane = b.op('add', INDEX, x, b.op('mul', INDEX, y, 32), hint='c')
    producer = b.op('ge', BOOL, y, 2, hint='prod')
    with b.for_(0, trips, 1, hint='t', uniform=Uniformity.BLOCK) as loop:
        t = loop.induction
        slot, parity = b.ring_round(ring, t)
        at = b.op('mul', INDEX, slot, WIDTH, hint='at')
        h = b.roles(producer, Role('producer', 32, IDENT),
                    Role('consumers', WIDTH, IDENT))
        with h.then():
            issue = b.op('eq', BOOL, x, 0, hint='issue')
            if cross:
                # The consumers' own buffer, written by the producer: no
                # barrier of either role orders the two.
                b.store(swap, b.const(0.0, ScalarType(F32)), x)
            free = (b.op('sub', INDEX, 1, parity, hint='free')
                    if producer_parity else parity)
            b.ring_wait(ring, 'empty', slot, free, writes=[slot_buf])
            tok = b.copy_bulk(slot_buf, src, barrier=RingSlot(ring, slot),
                              dst_index=[at],
                              src_index=[b.op('mul', INDEX, t, WIDTH)],
                              elems=WIDTH, predicate=issue)
            if expect:
                b.ring_arrive(ring, 'full', slot, issue, tx=tx,
                              tokens=[tok])
            else:
                b.ring_arrive(ring, 'full', slot, b.op('eq', BOOL, x, -1),
                              tx=tx, tokens=[tok])
        with h.otherwise():
            b.ring_wait(ring, 'full', slot, parity, writes=[slot_buf])
            value = b.load(slot_buf, b.op('add', INDEX, at, lane), hint='v')
            if exchange:
                other = b.op('rem', INDEX, b.op('add', INDEX, lane, 32),
                             WIDTH, hint='o')
                b.store(swap, value, other)
                if role_barrier:
                    b.barrier(Participants.ROLE)
                if block_barrier:
                    b.barrier(Participants.BLOCK)
                value = b.load(swap, lane, hint='w')
                lane_out = other
            else:
                lane_out = lane
            b.store(out, value, b.op('add', INDEX, b.op(
                'mul', INDEX, t, WIDTH), lane_out))
            if warp_sync:
                b.barrier(Participants.WAVE)
            if release:
                b.ring_arrive(ring, 'empty', slot,
                              b.op('eq', BOOL, x, 0, hint='first'))
    built = b.finish()
    placed, layout = allocate(_placed(built), arenas={'shrMem': MULT})
    return built, placed, layout


def _make_barrier(handoff, block):
    """The barriers the placement adds outside the roles: the block's, and
    the multiplication's -- one warp."""
    if block:
        return barrier_stmt(Participants.BLOCK, 32, handoff=handoff)
    return barrier_stmt(Participants.MULT, 32, 32, handoff)


def _placed(built, report=None):
    return place_barriers(
        tuple(built), arena=Arena({'shrMem': (MULT, 0), 'block': (BLOCK, 0)}),
        make_barrier=_make_barrier, arrival=Uniformity.MULT, report=report)


class _Sink:
    def __init__(self):
        self.lines = []

    def __call__(self, text, *args, **kwargs):
        self.lines.append(text)

    def __getattr__(self, _):
        return lambda *a, **k: None

    @contextlib.contextmanager
    def _block(self, head):
        self.lines.append(f'{head} {{')
        yield
        self.lines.append('}')

    def If(self, cond):
        return self._block(f'if ({cond})')

    def Else(self):
        return self._block('else')

    def Block(self, head=''):
        return self._block(head)

    def For(self, head, unroll=False):
        return self._block(f'for ({head})')


def _emit(body, arch='sm_90a'):
    sink = _Sink()
    Emitter(sink, _ctx(arch)).run(body)
    return sink.lines


def _oracle(lines, trips=5, seed=3):
    """The body on the host oracle, a producer warp and a consumer warp:
    the elements of `m0` that differ from `m1`'s, and the races."""
    keep = [line for line in lines if not re.match(r'^float m\d+\[', line)]
    src = ('void k() {\n'
           '  auto* shrMem = reinterpret_cast<float*>(totalShrMemPtr);\n'
           + ''.join(f'  {line}\n' for line in keep) + '}')
    races = []
    mem = kernel_eval.evaluate_wave(src, 32, seed=seed, globals_only=True,
                                    races=races, mults=3, block=True)
    fill = kernel_eval.Slot(seed)
    bad = [i for i in range(trips * WIDTH)
           if mem.get(('m0', i)) != fill.read('m1', i)]
    return bad, races


# -- the statements ------------------------------------------------------------ #

def test_the_ring_is_set_up_once_ahead_of_everything():
    """Its storage -- a pair of barriers a slot, for the whole body -- the
    setup by the leader, and the block's barrier behind both."""
    built, _, _ = _body()
    alloc, init, barrier = built[:3]
    assert alloc.op == Op.ALLOC and alloc.attr('lifetime') == 'body'
    assert alloc.target[0].type.shape == (2 * 2 * 2,)
    assert init.op == Op.SYNC_INIT and init.attr('extern') == 'pc_ring'
    ring = init.target[0]
    assert ring.type.slots == 2 and ring.type.kind is engines.SyncKind.MBARRIER
    assert (init.attr('full'), init.attr('empty')) == (1, 2)
    assert barrier.op == Op.BARRIER


def test_the_ring_verifies():
    """Each copy's token taken once, by the producer's arrival; the ring's
    waits and arrivals wherever a role stands, and the consumers' barrier
    inside their role."""
    built, _, _ = _body()
    assert verify(built, strict=False) == []


def test_a_role_barrier_outside_the_roles_is_refused():
    b = IRBuilder(fptype=F32, arena='shrMem', context=_ctx())
    with pytest.raises(IRError, match='outside the roles'):
        b.barrier(Participants.ROLE)


def test_a_block_barrier_inside_a_role_is_refused():
    """The other role never arrives: a deadlock the verifier names."""
    built, _, _ = _body(role_barrier=False, block_barrier=True)
    diag = verify(built, strict=False)
    assert any('block-wide barrier' in d for d in diag), diag


def test_the_text():
    lines = _emit(_body()[1])
    text = '\n'.join(lines)
    assert 'uint64_t * pc_ring = reinterpret_cast<uint64_t *>(' in text
    assert 'tensorforge::mbarrierInit(&pc_ring[r], 1);' in text
    assert 'tensorforge::mbarrierInit(&pc_ring[2 + r], 2);' in text
    waits = re.findall(r'tensorforge::mbarrierWait\(&pc_ring\[([^\]]+)\]',
                       text)
    assert len(waits) == 2 and waits[0].startswith('2 + ')
    assert re.search(r'tensorforge::bulkCopyOn\(&pc_ring\[\w+\], &slot\[',
                     text)
    assert re.search(r'tensorforge::mbarrierExpect\(&pc_ring\[\w+\], 256u, ',
                     text)
    assert re.search(r'tensorforge::mbarrierArrive\(&pc_ring\[2 \+ \(\w+\)\]',
                     text)
    assert ('asm volatile("barrier.sync %0, %1;" :: "r"((14 + (threadIdx.y '
            '>= 2))), "r"(64) : "memory");') in text


def test_a_copy_into_a_slot_is_one_threads():
    b = IRBuilder(fptype=F32, arena='shrMem', context=_ctx())
    ring = b.mbarrier_ring('pc', 2, full=1, empty=1, leader='true')
    dst = b.alloc(F32, (64,), MemSpace.SHARED, hint='s')
    src = b.alloc(F32, (64,), MemSpace.GLOBAL, hint='g', extern='m1')
    with pytest.raises(IRError, match='one thread'):
        b.copy_bulk(dst, src, barrier=RingSlot(ring, 0), dst_index=[0],
                    src_index=[0], elems=32)


def test_bytes_are_announced_on_the_full_side():
    with pytest.raises(IRError, match='full side'):
        engines.ring_arrive(None, 'empty', 0, True, tx=128)
    with pytest.raises(IRError, match='no'):
        engines.ring_wait(None, 'half', 0, 0)


def _loop_barriers(body):
    loop = next(s for s in body if s.op == Op.FOR)
    return [engines.object_of(s).participants
            for s, _ in walk(loop.regions[0].body) if s.op == Op.BARRIER]


def test_the_roles_meet_on_the_ring_alone():
    """The barrier placement leaves the hand-over to the ring: no barrier
    of the block inside the loop, where it would wait for the other role,
    and nothing reported unfenced -- the consumers' own exchange met on
    their barrier, and what a trip leaves the next one there too."""
    report = []
    built, _, _ = _body()
    found = _loop_barriers(_placed(built, report))
    assert Participants.BLOCK not in found and not report
    assert found.count(Participants.ROLE) == 2, found


def test_an_access_across_the_roles_is_reported():
    """The producer writes a buffer the consumers read, outside the ring:
    neither role's barrier orders the other, and the placement says so."""
    report = []
    built, _, _ = _body(cross=True)
    _placed(built, report)
    assert report and 'meet an earlier one' in report[0]


def test_a_role_hazard_is_met_on_the_roles_barrier():
    """Without the consumers' barrier between the exchange's store and its
    load, the placement puts the role's own there -- not the block's."""
    built, _, _ = _body(role_barrier=False)
    found = _loop_barriers(_placed(built))
    assert found.count(Participants.ROLE) == 2
    assert Participants.BLOCK not in found


# -- the roles' registers ------------------------------------------------------ #

def _budget_lines(count, change, arch='sm_90a'):
    b = IRBuilder(fptype=F32, arena='shrMem', context=_ctx(arch))
    b.emit(engines.register_budget(count, change))
    return _emit(b.finish(), arch)


@pytest.mark.parametrize('arch', ['sm_90a', 'sm_100a'])
def test_a_register_budget_is_setmaxnreg(arch):
    """A warpgroup's registers from here on: lowered to the block's pool
    by a producer, raised from it by its consumers."""
    assert _budget_lines(40, 'dec', arch) == [
        'asm volatile("setmaxnreg.dec.sync.aligned.u32 40;" ::: "memory");']
    assert _budget_lines(232, 'inc', arch) == [
        'asm volatile("setmaxnreg.inc.sync.aligned.u32 232;" ::: "memory");']


@pytest.mark.parametrize('count,change,arch,why', [
    (41, 'dec', 'sm_90a', 'multiple of eight'),
    (16, 'dec', 'sm_90a', 'multiple of eight'),
    (40, 'up', 'sm_90a', 'inc or dec'),
    (40, 'dec', 'sm_90', 'no setmaxnreg'),
    (40, 'dec', 'sm_80', 'no setmaxnreg')])
def test_a_register_budget_the_target_cannot_take_is_refused(count, change,
                                                             arch, why):
    with pytest.raises(IRError, match=why):
        _budget_lines(count, change, arch)


def test_a_register_budget_survives_dead_code_removal():
    """It has no result and no effect a pass could see: its content is
    where it stands."""
    from tensorforge.backend.pir.passes import _dce_protected
    assert _dce_protected(engines.register_budget(40, 'dec'))


# -- the host oracle ------------------------------------------------------------ #

@pytest.mark.parametrize('slots,trips', [(1, 3), (2, 5), (3, 7)])
def test_the_ring_hands_every_slot_over_on_the_oracle(slots, trips):
    """Every trip's floats arrive, and no access is unordered: the
    consumers' reads of a slot ahead of the producer's next copy into it
    through the warp's barrier and the first lane's release, the copy's
    bytes ahead of their reads through the full side."""
    bad, races = _oracle(_emit(_body(trips=trips, slots=slots)[1]),
                         trips=trips)
    assert not bad and not races, (bad[:4], races[:3])


def test_a_release_by_the_first_lane_alone_is_a_race():
    """Without the warp's barrier -- and without the exchange, whose role
    barrier would order every lane's read ahead of the release -- the first
    lane's arrival releases its own reads of the slot and none of the
    others'."""
    bad, races = _oracle(_emit(_body(warp_sync=False, exchange=False)[1]))
    assert {r.kind for r in races} == {'WAR'}


def test_a_slot_never_released_hangs():
    with pytest.raises(kernel_eval.Abort, match='never return'):
        _oracle(_emit(_body(release=False)[1]))


def test_bytes_never_announced_hang():
    with pytest.raises(kernel_eval.Abort, match='never return'):
        _oracle(_emit(_body(expect=False)[1]))


def test_bytes_announced_short_hang():
    """The copy completes 128 bytes on the full side, 64 announced: the
    phase never completes."""
    with pytest.raises(kernel_eval.Abort, match='never return'):
        _oracle(_emit(_body(tx=64)[1]))


def test_the_producer_waiting_for_its_consumers_parity_hangs():
    """Its first round has to find the slots free: the parity before the
    empty side's first phase."""
    with pytest.raises(kernel_eval.Abort, match='never return'):
        _oracle(_emit(_body(producer_parity=False)[1]))


def test_an_arrival_past_the_count_is_refused():
    """The producer's announcement twice: its full side counts one."""
    lines = _emit(_body()[1])
    i = next(i for i, line in enumerate(lines)
             if line.startswith('tensorforge::mbarrierExpect('))
    twice = lines[:i + 1] + [lines[i]] + lines[i + 1:]
    with pytest.raises(kernel_eval.Abort, match='past the 1'):
        _oracle(twice)


def test_a_block_barrier_in_a_role_never_meets_the_other_role():
    """The producer never comes: the oracle sees the consumers wait for
    lanes that do not arrive."""
    _, placed, _ = _body(role_barrier=False, block_barrier=True)
    _, races = _oracle(_emit(placed))
    assert 'ARRIVAL' in {r.kind for r in races}


def test_a_wait_ahead_of_the_setup_is_a_race():
    lines = _emit(_body()[1])
    i = lines.index('__syncthreads();')
    bad, races = _oracle(lines[:i] + lines[i + 1:])
    assert 'RAW' in {r.kind for r in races}


# -- the build and the hardware ---------------------------------------------- #

def _nvcc():
    found = toolchain.COMPILERS['cuda'].find()
    if not found or not shutil.which(found):
        pytest.skip('nvcc not found')
    return found


def _nvcc_builds(arch):
    """The toolkit's nvcc, or a skip where it builds no `arch`: the base
    target missing from `nvcc --list-gpu-code`, as CUDA 12.4 has no sm_120
    ("Value 'sm_120' is not defined for option 'gpu-architecture'")."""
    nvcc = _nvcc()
    listed = subprocess.run([nvcc, '--list-gpu-code'], capture_output=True,
                            text=True, timeout=60).stdout.split()
    if arch.rstrip('af') not in listed:
        pytest.skip(f'{nvcc} builds no {arch}')
    return nvcc


def _cuda_source(lines, layout, trips=5):
    keep = [line for line in lines if not re.match(r'^float m\d+\[', line)]
    body = ''.join(f'  {line}\n' for line in keep)
    n = trips * WIDTH
    return f'''#include <cstdio>
#include <vector>
#include "tensorforge_device/cuda.h"

__global__ void __launch_bounds__(96) ring_case(float *__restrict__ m0,
                                                 const float *__restrict__ m1) {{
  extern __shared__ __align__(128) float shrMem[];
{body}}}

#define CHECK(x) do {{ cudaError_t e_ = (x); if (e_ != cudaSuccess) {{ \\
  std::fprintf(stderr, "%s: %s\\n", #x, cudaGetErrorString(e_)); return 1; }} }} while (0)

int main() {{
  const size_t n = {n};
  std::vector<float> h(n), d(n);
  for (size_t i = 0; i < n; ++i) h[i] = float(i % 1021) + 0.5f;
  float *m0, *m1;
  CHECK(cudaMalloc(&m0, n * 4));
  CHECK(cudaMalloc(&m1, n * 4));
  CHECK(cudaMemcpy(m1, h.data(), n * 4, cudaMemcpyHostToDevice));
  CHECK(cudaMemset(m0, 0, n * 4));
  ring_case<<<1, dim3(32, 3), {layout.per_mult * 4}>>>(m0, m1);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());
  CHECK(cudaMemcpy(d.data(), m0, n * 4, cudaMemcpyDeviceToHost));
  size_t bad = 0;
  for (size_t i = 0; i < n; ++i) bad += d[i] != h[i];
  std::printf("%zu of %zu differ\\n", bad, n);
  return bad ? 4 : 0;
}}
'''


def _build(tmp_path, arch, product, trips=5):
    _, placed, layout = _body(trips=trips)
    src = tmp_path / 'ring.cu'
    src.write_text(_cuda_source(_emit(placed, arch), layout, trips))
    entry = toolchain.COMPILERS['cuda']
    out = tmp_path / ('ring.cubin' if product == 'cubin' else 'ring')
    cmd = [_nvcc_builds(arch), *entry.language_flags(),
           *entry.target_flags(arch), '-I', str(INCLUDE), '-o', str(out),
           str(src)]
    if product == 'cubin':
        cmd.insert(1, '-cubin')
    run = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    assert run.returncode == 0, run.stderr[-4000:]
    return out


@pytest.mark.parametrize('arch', ['sm_90a', 'sm_120'])
def test_compiled_the_ring_is_the_hardwares_barriers(arch, tmp_path):
    """The waits spin on `try_wait.parity`, the copy is one bulk copy
    completing on the barrier, the arrivals the barrier's."""
    cubin = _build(tmp_path, arch, 'cubin')
    tool = shutil.which('cuobjdump')
    if tool is None:
        pytest.skip('cuobjdump not found')
    sass = subprocess.run([tool, '-sass', str(cubin)], capture_output=True,
                          text=True, timeout=600).stdout
    assert re.search(r'\bSYNCS\.PHASECHK\.TRANS64\.TRYWAIT\b', sass)
    assert re.search(r'\bUBLKCP\b', sass)
    assert re.search(r'\bSYNCS\.ARRIVE\.TRANS64', sass)


def _gpu():
    tool = shutil.which('nvidia-smi')
    if tool is None:
        return None
    try:
        run = subprocess.run([tool, '--query-gpu=compute_cap',
                              '--format=csv,noheader'], capture_output=True,
                             text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    first = run.stdout.strip().splitlines()[:1]
    if run.returncode or not first:
        return None
    return 'sm_' + first[0].strip().replace('.', '')


@pytest.mark.parametrize('trips', [5, 64])
def test_gpu_the_ring_hands_every_slot_over(trips, tmp_path):
    """Built for the GPU present, of sm_90 or newer: every trip's floats
    through the ring, to the bit; a hang is the test's timeout."""
    arch = _gpu()
    if arch is None or int(arch[3:]) < 90:
        pytest.skip('no GPU of sm_90 or newer')
    exe = _build(tmp_path, arch, 'binary', trips=trips)
    run = subprocess.run([str(exe)], capture_output=True, text=True,
                         timeout=120)
    assert run.returncode == 0, (run.stdout, run.stderr[-2000:])
    assert run.stdout.startswith('0 of ')
