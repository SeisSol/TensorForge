# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""The roles of a block and their rings on AMD and Intel (`pir.roles`,
`engines.RingSpec`): what each target spells them as, and where it runs
none.

* gfx1250 and gfx1251: a role meets on a named barrier object of its own --
  every wave counts every role's object at the kernel's head and joins its
  own (`hip.h`, `joinRoleBarriers`), a role's barrier is a signal and a wait
  on it (`syncNamedBarrier`) -- and a ring's barriers are LDS counters
  (`ldsRingInit`, `ldsRingWait`, `ldsRingArrive`): the arrivals a phase and
  so far in one word, the parity protocol of an mbarrier, no transaction
  count.  Shown on the host oracle (`reference.kernel_eval`) and in the
  compiled ISA; no gfx1250 has run it.
* CDNA (gfx9): no roles, with the reasons (`roles.CDNA_REFUSAL`); the rings
  of LDS counters compile for it, a starting point for a producer there.
* PVC under ESIMD: named barriers -- a role's a signal and a wait of its
  threads (`namedBarrierSync`), a ring's a barrier a side of each slot, the
  producers signaling and the consumers signaling and waiting, the
  consumers' first release of every slot ahead of their first round
  (`ring_prime`), and the kernel's count of them at its head
  (`namedBarriersInit`).  Every id and count a compile-time constant -- a
  role's its own, a ring's side picked from the slot by a branch a slot --
  as IGC 1.0.15468 (SNG2's) takes no other.  Shown in the text and compiled
  for PVC, every id reaching IGC a constant; run on SNG2's PVC under IGC
  2.41.5 (job 564784), and refused where the IGC behind the build writes a
  signal as IGC 1.0.15468 does, from the wrong operands -- every kernel
  hangs there (jobs 564774, 564780, 564782): the build's IGC asked how it
  writes one (`toolchain.intel_named_barrier_signal`, `engines.
  pvc_named_refusal`); `engines.PVC_NAMED_BARRIERS` overrides the
  question for the tests that lower them.
* SYCL's SPMD lowering: no roles and no rings, with the reason.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

import test_mbarrier_ring as tmr
import test_named_barriers as tnb
import test_roles as tr
from tensorforge.backend.pir import engines, roles as pr
from tensorforge.backend.pir.allocate import BLOCK, MULT, allocate
from tensorforge.backend.pir.build import IRBuilder
from tensorforge.backend.pir.core import (BOOL, INDEX, IRError, MemSpace, Op,
                                          Participants, Uniformity,
                                          walk_stmts)
from tensorforge.backend.pir.emit import Emitter
from tensorforge.backend.pir.engines import SyncKind
from tensorforge.backend.pir.passes import verify
from tensorforge.backend.pir.roles import Role, RoleSplit
from tensorforge.common.basic_types import Datatype
from tensorforge.common.context import Context
from tensorforge.reference import kernel_eval

F32 = Datatype.F32
WIDTH = tr.WIDTH
INCLUDE = Path(__file__).resolve().parents[1] / 'src' / 'tensorforge' / 'include'


def _ctx(arch, backend='hip'):
    return Context(arch=arch, backend=backend, fp_type=F32)


def _emit(body, ctx):
    sink = tmr._Sink()
    Emitter(sink, ctx).run(body)
    return sink.lines


# -- gfx1250: the pipelines of roles over rings of LDS counters ---------------- #

def _hip_pipeline(stages=3, trips=5, slots=2, looped=True, release=True,
                  role_sync=True, arch='gfx1250'):
    """`test_roles._pipeline` on AMD: a producer wave and `stages - 1`
    stages of two waves each, a ring between each two -- of LDS counters
    (`IRBuilder.ring`).  The producer's lanes store the trip's floats into
    the slot, meet on their role's barrier, and its first lane publishes;
    each stage takes its slot, a mixer hands each float to the lane of the
    other wave through a buffer of its own (met on its role's barrier),
    gives the slot back by each wave's first lane behind that barrier,
    fills the next ring's slot and publishes it by its first lane; the
    consumer writes `m0` and gives its slot back.  A slot's full side
    counts the one arrival of its producer's first lane, its empty side one
    of each wave of its consumers.  `release` off: the first mixer never
    gives a slot back; `role_sync` off: the consumer's waves give the slot
    back without their role's barrier."""
    ctx = _ctx(arch)
    b = IRBuilder(fptype=F32, arena='shrMem', context=ctx)
    names = ['producer'] + [f'mixer{i}' for i in range(1, stages - 1)] + [
        'consumer']
    ranges = [(tr.STAGE_WARPS * (stages - 1), 1)] + [
        (tr.STAGE_WARPS * i, tr.STAGE_WARPS) for i in range(stages - 1)]
    split = RoleSplit(tuple(Role(n, 32 * c, warps=(lo, c))
                            for n, (lo, c) in zip(names, ranges)))
    rings = [b.ring(f'p{i}', slots, full=1, empty=tr.STAGE_WARPS,
                    leader='threadIdx.x == 0 && threadIdx.y == 0',
                    arena='block', producer=names[i],
                    consumers=(names[i + 1],))
             for i in range(stages - 1)]
    bufs = [b.alloc(F32, (slots * WIDTH,), MemSpace.SHARED, hint=f's{i}',
                    extern=f's{i}', place_align=4, handed=True,
                    arena='block')
            for i in range(stages - 1)]
    swaps = [b.alloc(F32, (WIDTH,), MemSpace.SHARED, hint=f'w{i}',
                     extern=f'w{i}', arena='block')
             for i in range(stages - 2)]
    src = b.alloc(F32, (trips * WIDTH,), MemSpace.GLOBAL, hint='g',
                  extern='m1')
    out = b.alloc(F32, (trips * WIDTH,), MemSpace.GLOBAL, hint='o',
                  extern='m0')
    x, y = b.thread_id('x'), b.thread_id('y')

    def lane_of(role):
        lo = split.role(role).warps[0]
        return b.op('add', INDEX, x, b.op('mul', INDEX, b.op(
            'sub', INDEX, y, lo, hint='w'), 32, hint='w'), hint='c')

    def first_of(role):
        return b.op('eq', BOOL, lane_of(role), 0, hint='first')

    def wave_first():
        return b.op('eq', BOOL, x, 0, hint='wfirst')

    def produce(t):
        ring = rings[0]
        slot, parity = b.ring_round(ring, t)
        at = b.op('mul', INDEX, slot, WIDTH, hint='at')
        b.ring_acquire(ring, slot, parity, writes=[bufs[0]])
        for half in (0, 32):
            i = b.op('add', INDEX, x, half, hint='i')
            value = b.load(src, b.op('add', INDEX, b.op(
                'mul', INDEX, t, WIDTH), i), hint='g')
            b.store(bufs[0], value, b.op('add', INDEX, at, i))
        b.barrier(Participants.ROLE)
        b.ring_publish(ring, slot, first_of('producer'))

    def mix(t, i):
        into, outof = rings[i], rings[i - 1]
        slot, parity = b.ring_round(into, t)
        at = b.op('mul', INDEX, slot, WIDTH, hint='at')
        lane = lane_of(f'mixer{i}')
        first = first_of(f'mixer{i}')
        b.ring_take(outof, slot, parity, writes=[bufs[i - 1]])
        value = b.load(bufs[i - 1], b.op('add', INDEX, at, lane), hint='v')
        other = b.op('rem', INDEX, b.op('add', INDEX, lane, 32), WIDTH,
                     hint='o')
        b.store(swaps[i - 1], value, other)
        b.barrier(Participants.ROLE)
        value = b.load(swaps[i - 1], lane, hint='w')
        if release or i > 1:
            b.ring_release(outof, slot, wave_first())
        b.ring_acquire(into, slot, parity, writes=[bufs[i]])
        b.store(bufs[i], value, b.op('add', INDEX, at, lane))
        b.barrier(Participants.ROLE)
        b.ring_publish(into, slot, first)

    def consume(t):
        ring = rings[-1]
        slot, parity = b.ring_round(ring, t)
        at = b.op('mul', INDEX, slot, WIDTH, hint='at')
        lane = lane_of('consumer')
        b.ring_take(ring, slot, parity, writes=[bufs[-1]])
        value = b.load(bufs[-1], b.op('add', INDEX, at, lane), hint='v')
        b.store(out, value, b.op('add', INDEX, b.op('mul', INDEX, t, WIDTH),
                                 lane))
        if role_sync:
            b.barrier(Participants.ROLE)
        b.ring_release(ring, slot, wave_first())

    def body(name, t):
        if name == 'producer':
            produce(t)
        elif name == 'consumer':
            consume(t)
        else:
            mix(t, int(name[len('mixer'):]))

    def trips_of(step):
        with b.for_(0, trips, 1, hint='t', uniform=Uniformity.BLOCK) as loop:
            step(loop.induction)

    if looped:
        h = b.split_roles(split)
        for name in names:
            with h.role(name):
                trips_of(lambda t, name=name: body(name, t))
    else:
        def trip(t):
            h = b.split_roles(split)
            for name in names:
                with h.role(name):
                    body(name, t)
        trips_of(trip)
    built = b.finish()
    report = []
    placed, _ = allocate(tmr._placed(built, report),
                         arenas={'shrMem': MULT, 'block': BLOCK})
    return built, placed, report, ctx


def test_the_split_takes_the_objects_of_gfx1250():
    """The roles' ids are indices into the kernel's objects from 0, in the
    order of their warps; the objects are set up at the body's head, every
    role's count of waves, each wave joining its own."""
    built, placed, _, ctx = _hip_pipeline()
    (setup,) = [s for s in built if s.op == Op.SYNC_INIT
                and s.attr('waves') is not None]
    assert setup.attr('waves') == (2, 2, 1)
    assert setup.attr('ident') == '(0 + (threadIdx.y >= 2) + (threadIdx.y >= 4))'
    lines = _emit(placed, ctx)
    assert ('__shared__ __amdgpu_named_workgroup_barrier_t tfRoles[3];'
            in lines)
    assert ('tensorforge::joinRoleBarriers<2, 2, 1>(tfRoles, (0 + '
            '(threadIdx.y >= 2) + (threadIdx.y >= 4)));' in lines)
    assert ('tensorforge::syncNamedBarrier(&tfRoles[(0 + (threadIdx.y >= 2)'
            ' + (threadIdx.y >= 4))]);' in lines)


def test_the_rings_of_gfx1250_are_lds_counters():
    """A ring's statements as the CUDA ring's, its barriers LDS counters:
    the setup by the leader, the waits for a parity, the arrivals by the
    thread their operand picks -- no bytes."""
    built, placed, _, ctx = _hip_pipeline()
    rings = [s.target[0] for s in built if s.op == Op.SYNC_INIT
             and s.target]
    assert all(r.type.kind is SyncKind.LDS_ARRIVE for r in rings)
    text = '\n'.join(_emit(placed, ctx))
    assert 'tensorforge::ldsRingInit(&p0_ring[r], 1);' in text
    assert 'tensorforge::ldsRingInit(&p0_ring[2 + r], 2);' in text
    assert re.search(r'tensorforge::ldsRingWait\(&p0_ring\[2 \+ \(\w+\)\], '
                     r'\(1 - ', text)
    assert re.search(r'tensorforge::ldsRingArrive\(&p1_ring\[\w+\], ', text)
    assert 'mbarrier' not in text
    assert verify(built, strict=False) == []


def _run(lines, stages, trips=5, seed=3):
    return tr._run(lines, stages, trips=trips, seed=seed)


@pytest.mark.parametrize('stages', [2, 3, 4])
@pytest.mark.parametrize('looped', [True, False], ids=['looped', 'trip'])
@pytest.mark.parametrize('slots,trips', [(1, 3), (2, 5), (3, 7)])
def test_every_float_through_the_roles_of_gfx1250_on_the_oracle(
        stages, looped, slots, trips):
    """Two to four roles on gfx1250's objects over rings of LDS counters:
    every float where the pipeline puts it, no access unordered, the arms
    of the nested IFs as coroutines."""
    _, placed, report, ctx = _hip_pipeline(stages=stages, trips=trips,
                                           slots=slots, looped=looped)
    assert not report
    bad, races = _run(_emit(placed, ctx), stages, trips=trips)
    assert not bad and not races, (bad[:4], races[:3])


def test_an_lds_slot_never_given_back_hangs_on_the_oracle():
    _, placed, _, ctx = _hip_pipeline(release=False)
    with pytest.raises(kernel_eval.Abort, match='never return|every arm'):
        _run(_emit(placed, ctx), 3)


def test_a_release_ahead_of_the_roles_reads_is_a_race_on_the_oracle():
    """The consumer's first lane gives the slot back without its role's
    barrier: the producer's next stores into the slot race the other
    lanes' reads of it."""
    _, placed, _, ctx = _hip_pipeline(stages=2, role_sync=False)
    _, races = _run(_emit(placed, ctx), 2)
    assert 'WAR' in {r.kind for r in races}, races[:3]


def test_an_lds_ring_announces_no_bytes():
    """A producer waits for its own copies ahead of its arrival: the LDS
    counters count no transactions, and an arrival with bytes is
    refused where it would be spelled."""
    b = IRBuilder(fptype=F32, arena='shrMem', context=_ctx('gfx1250'))
    ring = b.ring('pc', 2, full=1, empty=1, leader='true')
    s = b.ring_publish(ring, 0, True, tx=256)
    with pytest.raises(IRError, match='count no transactions'):
        engines.lower(s, _ctx('gfx1250').target, name=str)


def test_one_split_a_body_on_gfx1250():
    """A wave joins one object: a body sets up one split's."""
    b = IRBuilder(fptype=F32, arena='shrMem', context=_ctx('gfx1250'))
    b.split_roles(tr._split((0, 1), (1, 1), names=['a', 'b']))
    with pytest.raises(IRError, match='a second split'):
        b.split_roles(tr._split((0, 2), (2, 1), names=['a', 'b']))


def _hip_kernel(placed, ctx, threads):
    keep = [line for line in _emit(placed, ctx)
            if not re.match(r'^float m\d+\[', line)]
    return ('#include "tensorforge_device/hip.h"\n'
            f'__global__ void __launch_bounds__({threads}) k('
            'float *__restrict__ m0, const float *__restrict__ m1) {\n'
            '  extern __shared__ float block[];\n'
            + ''.join(f'  {line}\n' for line in keep) + '}\n')


@pytest.mark.parametrize('arch', ['gfx1250', 'gfx1251'])
def test_the_roles_and_rings_of_gfx1250_compile(arch, tmp_path):
    """Read off the instructions: every role's object counted and the
    wave's own joined, the roles' rendezvous a signal and a wait, the
    arrivals 64-bit LDS atomics, the waits LDS loads with a sleep between,
    and the objects asked of the dispatch."""
    _, placed, _, ctx = _hip_pipeline(arch=arch)
    run = tnb._assemble(_hip_kernel(placed, ctx, 160), arch, tmp_path)
    assert run.returncode == 0, run.stderr[-3000:]
    asm = run.stdout
    assert len(re.findall(r'\ts_barrier_init\b', asm)) == 3
    assert re.search(r'\ts_barrier_join\b', asm)
    assert re.search(r'\ts_barrier_signal\b', asm)
    assert re.search(r'\ts_barrier_wait\b', asm)
    assert re.search(r'\tds_add_u64\b', asm)
    assert re.search(r'\tds_load_b64\b', asm)
    assert re.search(r'\ts_sleep 1\b', asm)
    assert re.search(r'\.amdhsa_named_barrier_count', asm)


# -- CDNA: no roles, the rings compile ------------------------------------------ #

@pytest.mark.parametrize('arch', ['gfx90a', 'gfx942', 'gfx950'])
def test_cdna_runs_no_roles(arch):
    """The reasons stand in the refusal: registers per kernel, the matrix
    cores in the wave, the work-group's one barrier, the ping-pong as
    CDNA's measured form -- and the rings that compile there."""
    ctx = _ctx(arch)
    why = pr.roles_refusal(ctx.target, 2)
    assert why == pr.CDNA_REFUSAL
    for said in ('allocated per kernel', 'ping-pong', 'ldsRing',
                 'LDS-DMA'):
        assert said in why
    b = IRBuilder(fptype=F32, arena='shrMem', context=ctx)
    with pytest.raises(IRError, match='CDNA runs no roles'):
        b.split_roles(tr._split((0, 1), (1, 1), names=['a', 'b']))


def test_an_amd_device_without_objects_runs_no_roles():
    why = pr.roles_refusal(_ctx('gfx1150').target, 2)
    assert 'no named barriers' in why


def _cdna_ring(arch):
    """A producer and consumer over a ring of LDS counters without roles:
    the first wave fills a slot, the second drains it, each loop its own --
    the ring alone, as a CDNA producer would use it."""
    ctx = _ctx(arch)
    b = IRBuilder(fptype=F32, arena='shrMem', context=ctx)
    ring = b.ring('pc', 2, full=1, empty=1,
                  leader='threadIdx.x == 0 && threadIdx.y == 0',
                  arena='block')
    buf = b.alloc(F32, (2 * WIDTH,), MemSpace.SHARED, hint='s', extern='s',
                  handed=True, arena='block')
    src = b.alloc(F32, (5 * WIDTH,), MemSpace.GLOBAL, hint='g', extern='m1')
    out = b.alloc(F32, (5 * WIDTH,), MemSpace.GLOBAL, hint='o', extern='m0')
    x, y = b.thread_id('x'), b.thread_id('y')
    first = b.op('eq', BOOL, x, 0, hint='first')
    with b.if_(b.op('eq', BOOL, y, 0, hint='prod')):
        with b.for_(0, 5, 1, hint='t', uniform=Uniformity.BLOCK) as loop:
            t = loop.induction
            slot, parity = b.ring_round(ring, t)
            b.ring_acquire(ring, slot, parity, writes=[buf])
            b.store(buf, b.load(src, b.op('add', INDEX, b.op(
                'mul', INDEX, t, WIDTH), x)), b.op('add', INDEX, b.op(
                    'mul', INDEX, slot, WIDTH), x))
            b.ring_publish(ring, slot, first)
    return b, ctx, ring, buf, out


@pytest.mark.parametrize('arch', ['gfx90a', 'gfx942'])
def test_the_rings_compile_for_cdna(arch, tmp_path):
    """The LDS counters on CDNA: 64-bit LDS atomics, LDS loads polled with
    a sleep between -- a starting point for a producer there."""
    b, ctx, ring, buf, out = _cdna_ring(arch)
    with b.if_(b.op('eq', BOOL, b.thread_id('y'), 1, hint='cons')):
        with b.for_(0, 5, 1, hint='t', uniform=Uniformity.BLOCK) as loop:
            t = loop.induction
            slot, parity = b.ring_round(ring, t)
            b.ring_take(ring, slot, parity, writes=[buf])
            x = b.thread_id('x')
            b.store(out, b.load(buf, b.op('add', INDEX, b.op(
                'mul', INDEX, slot, WIDTH), x)), b.op('add', INDEX, b.op(
                    'mul', INDEX, t, WIDTH), x))
            b.ring_release(ring, slot, b.op('eq', BOOL, x, 0, hint='first'))
    placed, _ = allocate(b.finish(), arenas={'shrMem': MULT, 'block': BLOCK})
    run = tnb._assemble(_hip_kernel(placed, ctx, 128), arch, tmp_path)
    assert run.returncode == 0, run.stderr[-3000:]
    assert re.search(r'\tds_add_u64\b', run.stdout)
    assert re.search(r'\ts_sleep 1\b', run.stdout)


# -- PVC under ESIMD: named barriers ------------------------------------------- #

@pytest.mark.parametrize('form,why', [
    ('flag-producers', 'PVC_REFUSAL'), (None, 'PVC_UNPROBED'),
    ('gateway', None)], ids=['old-form', 'no-igc', 'gateway'])
def test_pvc_takes_roles_and_rings_where_its_igc_writes_the_signal_right(
        form, why, monkeypatch):
    """The IGC behind the build asked how it writes a named barrier's
    signal (`toolchain.intel_named_barrier_signal`): the whole gateway
    message, and PVC takes roles and rings; IGC 1.0.15468's old form, whose
    every wait hangs on SNG2, or no IGC to ask, and they are refused with
    the reason -- `engines.PVC_NAMED_BARRIERS` overriding the question."""
    from tensorforge import toolchain
    monkeypatch.setattr(toolchain, 'intel_named_barrier_signal',
                        lambda *args, **kwargs: form)
    ctx = _ctx('pvc', 'esimd')
    assert engines.PVC_NAMED_BARRIERS is None
    expected = None if why is None else getattr(engines, why)
    assert pr.roles_refusal(ctx.target, 2) == expected
    assert engines.ring_refusal(ctx.target) == expected
    b = IRBuilder(fptype=F32, arena='shrMem', context=ctx)
    if why is not None:
        with pytest.raises(IRError, match='roles and rings on PVC refused'):
            b.split_roles(tr._split((0, 1), (1, 1), names=['a', 'b']))
        with pytest.raises(IRError, match='roles and rings on PVC refused'):
            b.ring('pc', 2, full=1, empty=1, leader='true', waiters=(1, 1))
    for forced, refused in ((True, None), (False, engines.PVC_REFUSAL)):
        monkeypatch.setattr(engines, 'PVC_NAMED_BARRIERS', forced)
        assert engines.ring_refusal(ctx.target) == refused


def test_the_signal_form_is_read_from_the_listing():
    """vISA's `nbarrier.signal <flag> <producers>`, as IGC 1.0.15468 wrote
    it on SNG2 (job 564782), against the gateway message IGC 2.41.5
    writes."""
    from tensorforge.toolchain import named_barrier_signal_form
    old = ('    nbarrier.signal 0x1:b 0x1:b   /// $41\n'
           '    nbarrier.wait 0x1:b   /// $42\n')
    new = ('    raw_sends.3.1.0.0 (M1, 1) 0x0:ud 0x2000004:ud V41.0 %null.0 '
           '%null.0   /// $12\n    nbarrier.wait 0x1:b   /// $13\n')
    assert named_barrier_signal_form(old) == 'flag-producers'
    assert named_barrier_signal_form(new) == 'gateway'
    assert named_barrier_signal_form('    nbarrier.wait 0x1:b\n') is None


def test_the_igc_here_writes_the_gateway_message():
    """The toolchain of this machine asked for real -- where icpx and ocloc
    are on PATH."""
    from tensorforge import toolchain
    if toolchain.COMPILERS['esimd'].find() is None or \
            shutil.which('ocloc') is None:
        pytest.skip('icpx (`TF_ICPX`) and ocloc not found')
    assert toolchain.intel_named_barrier_signal() == 'gateway'


@pytest.fixture
def named_pvc(monkeypatch):
    """PVC's roles and rings lowered all the same, for their text and their
    build (`engines.PVC_NAMED_BARRIERS`)."""
    monkeypatch.setattr(engines, 'PVC_NAMED_BARRIERS', True)


def _pvc(**ring):
    ctx = _ctx('pvc', 'esimd')
    b = IRBuilder(fptype=F32, arena='shrMem', context=ctx)
    return ctx, b


def test_the_roles_of_pvc_take_named_barriers(named_pvc):
    """Ids from 1 on, each role's its own as a constant (`RoleSplit.
    constant`) -- the warps' order's, as on NVIDIA: the consumers' range
    starts lower; a role's barrier a signal and a wait of its threads; the
    kernel's count of named barriers at the body's head."""
    ctx, b = _pvc()
    split = RoleSplit((Role('producer', 1, warps=(2, 1)),
                       Role('consumers', 2, warps=(0, 2))),
                      warp='tid', axis='y')
    h = b.split_roles(split)
    with h.role('producer'):
        alone = b.barrier(Participants.ROLE)
    with h.role('consumers'):
        barrier = b.barrier(Participants.ROLE)
    body = b.finish()
    assert body[0].attr('named') == 2
    assert engines.lower(body[0], ctx.target, name=str).lines == (
        'tensorforge::namedBarriersInit<2>();',)
    assert engines.lower(barrier, ctx.target, name=str).lines == (
        'tensorforge::namedBarrierSync<1, 2>();',)
    assert engines.lower(alone, ctx.target, name=str).lines == (
        'tensorforge::namedBarrierSync<2, 1>();',)
    assert verify(body, strict=False) == []


def test_a_role_barrier_of_pvc_named_by_a_value_is_refused():
    """A split whose roles share the id each thread computes, a value:
    IGC 1.0.15468 takes no named barrier so, and the lowering says so."""
    ctx = _ctx('pvc', 'esimd')
    obj = engines.sync_object(Participants.ROLE, 16, 2, '(1 + (tid >= 2))')
    with pytest.raises(IRError, match='names its id as a value'):
        engines.lower(engines.barrier(obj), ctx.target, name=str)


def _pvc_ring(slots=2, waiters=(2, 1)):
    ctx, b = _pvc()
    ring = b.ring('pc', slots, full=1, empty=2, leader='true',
                  producer='producer', consumers=('consumers',),
                  waiters=waiters)
    return ctx, b, ring


def test_a_ring_of_pvc_is_a_named_barrier_a_side_of_each_slot(named_pvc):
    """The full sides from the ring's first id on, the empty sides behind
    them; a publish and a release signal as producers, an acquire and a
    take signal as consumers and wait, each with the side's counts; the
    consumers' first release signals every empty side once."""
    ctx, b, ring = _pvc_ring()
    spec = ring.spec
    assert spec.kind is SyncKind.NAMED and spec.ident == 1
    lower = lambda s: engines.lower(s, ctx.target, name=str).lines  # noqa
    assert lower(b.ring_publish(ring, 's', 'issue')) == (
        'tensorforge::namedRingArrive<1, 2, 1, 2>(s, issue);',)
    assert lower(b.ring_take(ring, 's', 'p')) == (
        'tensorforge::namedRingWait<1, 2, 1, 2>(s);',)
    assert lower(b.ring_release(ring, 's', 'issue')) == (
        'tensorforge::namedRingArrive<3, 2, 2, 1>(s, issue);',)
    acquire = b.ring_acquire(ring, 's', 'p')
    assert lower(acquire) == ('tensorforge::namedRingWait<3, 2, 2, 1>(s);',)
    assert lower(b.ring_prime(ring, 'issue')) == (
        'tensorforge::namedRingPrime<3, 2, 2, 1>(issue);',)
    assert b.finish()[0].attr('named') == 4


def test_a_parity_ring_takes_no_prime():
    """An mbarrier's and an LDS counter's waits find the slots free by the
    parity before their first phase: no first release written."""
    for ctx in (tmr._ctx(), _ctx('gfx1250')):
        b = IRBuilder(fptype=F32, arena='shrMem', context=ctx)
        ring = b.ring('pc', 2, full=1, empty=1, leader='true')
        assert b.ring_prime(ring, True) is None
        assert not [s for s in b.finish() if s.op == Op.ARRIVE]


@pytest.mark.parametrize('ring,why', [
    (dict(waiters=(0, 0)), 'the threads that wait'),
    (dict(tx=128), 'count no transactions')])
def test_a_named_ring_without_its_counts_is_refused(ring, why, named_pvc):
    ctx, b = _pvc()
    with pytest.raises(IRError, match=why):
        r = b.ring('pc', 2, full=1, empty=1, leader='true',
                   **{'waiters': (1, 1), **ring})
        engines.lower(b.ring_publish(r, 0, True, tx=ring.get('tx', 0)),
                      ctx.target, name=str)


#: A producer thread and two consumer threads over a ring of two slots,
#: the consumers meeting on their role's barrier: the named barriers'
#: helpers of `isycl.h` as the lowerings write them -- or, `{value}`, a
#: barrier whose id is a value, as the lowering wrote them before.  The
#: rounds a run-time count, so that no unrolled loop makes a slot a
#: constant on its own.
PVC_RING = r'''#include "tensorforge_device/isycl.h"
int main(int argc, char **) {
  sycl::queue q;
  float *m0 = sycl::malloc_device<float>(64, q);
  const int trips = 7 + argc;
  q.parallel_for(sycl::nd_range<1>(3, 3), [=](sycl::nd_item<1> it)
                 SYCL_ESIMD_KERNEL {
    tensorforge::namedBarriersInit<5>();
    const unsigned tid = it.get_local_id(0);
    if (tid >= 2) {
      for (int t = 0; t < trips; ++t) {
        tensorforge::namedRingWait<3, 2, 2, 1>(t % 2);
        tensorforge::namedRingArrive<1, 2, 1, 2>(t % 2, true);
      }
    } else {
      tensorforge::namedRingPrime<3, 2, 2, 1>(true);
      for (int t = 0; t < trips; ++t) {
        tensorforge::namedRingWait<1, 2, 1, 2>(t % 2);
        tensorforge::namedBarrierSync<5, 2>();
        tensorforge::namedRingArrive<3, 2, 2, 1>(t % 2, true);
        {value}
      }
    }
  }).wait();
  sycl::free(m0, q);
}
'''


def _pvc_ir(tmp_path, value=''):
    """`PVC_RING` built for PVC by icpx and ocloc, IGC dumping: the IR as
    IGC's vector compiler read it (`*_after_spirv_reader.ll`) -- what SNG2's
    IGC 1.0.15468 gets, whatever IGC is on PATH -- and IGC's last
    (`*_final.ll`).  Where icpx and ocloc are on PATH."""
    import os
    from tensorforge import toolchain
    icpx = toolchain.COMPILERS['esimd'].find()
    if icpx is None or shutil.which('ocloc') is None:
        pytest.skip('icpx (`TF_ICPX`) and ocloc not found')
    source = tmp_path / 'ring.cpp'
    source.write_text(PVC_RING.replace('{value}', value))
    dump = tmp_path / 'dump'
    dump.mkdir()
    run = subprocess.run([icpx, '-fsycl', '-fsycl-targets=spir64_gen',
                          '-Xs', '-device pvc', '-I', str(INCLUDE),
                          str(source), '-o', str(tmp_path / 'ring')],
                         capture_output=True, text=True, timeout=900,
                         env={**os.environ, 'IGC_ShaderDumpEnable': '1',
                              'IGC_DumpToCustomDir': str(dump)})
    assert run.returncode == 0, run.stderr[-3000:]
    read = sorted(dump.glob('*_after_spirv_reader.ll'))
    final = sorted(dump.glob('*_final.ll'))
    if not read or not final:
        pytest.skip('IGC wrote no dumps (IGC_ShaderDumpEnable)')
    return read[0].read_text(), final[0].read_text()


def _nbarriers(ir):
    """The named barriers' calls of `ir`, their arguments as text: a
    signal's (`nbarrier.arrive`: the id, the mode, the counts) and a wait's
    (`nbarrier`: the operation, the id, a count)."""
    return re.findall(r'call void @llvm\.genx\.nbarrier(?:\.arrive)?'
                      r'\(([^)]*)\)', ir)


def test_every_named_barrier_id_reaches_igc_as_a_constant(tmp_path):
    """The helpers built for PVC: every named barrier's id and count a
    constant in the IR IGC reads and in its last -- the ring's sides
    picked from the slot by a branch a slot, no branch merged back into a
    value."""
    for ir in _pvc_ir(tmp_path):
        calls = _nbarriers(ir)
        assert calls and not [c for c in calls if '%' in c], calls[:8]


def test_a_named_barrier_id_of_a_value_is_seen(tmp_path):
    """The check's other side: a wait on a barrier whose id is a value, as
    the lowering wrote a ring's before, reaches IGC as a value -- what IGC
    1.0.15468 refuses ("Incorrect args to intrinsic call")."""
    read, _ = _pvc_ir(tmp_path, value='tensorforge::intel_esimd::'
                      'named_barrier_wait(1 + tid);')
    assert [c for c in _nbarriers(read) if '%' in c]


# -- SYCL's SPMD lowering: none ---------------------------------------------- #

def test_spmd_runs_no_roles_and_no_rings():
    ctx = _ctx('pvc', 'oneapi')
    assert pr.roles_refusal(ctx.target, 2) == pr.SPMD_REFUSAL
    b = IRBuilder(fptype=F32, arena='shrMem', context=ctx)
    with pytest.raises(IRError, match='no barrier over part of a '
                       'work-group'):
        b.split_roles(tr._split((0, 1), (1, 1), names=['a', 'b']))
    with pytest.raises(IRError, match='SPMD lowering has no barrier'):
        b.ring('pc', 2, full=1, empty=1, leader='true')


def test_cuda_below_sm_90_has_no_rings_but_roles():
    """`bar.sync` names a role's barrier on any NVIDIA part; a ring waits
    for a phase's parity from sm_90 on."""
    ctx = Context(arch='sm_80', backend='cuda', fp_type=F32)
    assert pr.roles_refusal(ctx.target, 3) is None
    b = IRBuilder(fptype=F32, arena='shrMem', context=ctx)
    with pytest.raises(IRError, match='from sm_90 on'):
        b.ring('pc', 2, full=1, empty=1, leader='true')
