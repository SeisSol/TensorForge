# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""The synchronization and ordering ops, through the registry
(`pir.engines`).

A barrier, a fence, a counter wait and a wave priority are each one entry of
`engines.REGISTRY`: what a statement of the op declares, what the passes may
assume of it, and how each backend writes it.  What existed before is
written as it was -- every barrier as the lexic spelled it, the fences as the
raw text they replace -- which is what most of these check, against the
lexic directly rather than against a copy of its strings.  What is new -- the
arrive without the wait, counter waits, priorities -- is checked against the
instruction a device compiler makes of it where one is installed.
"""

from __future__ import annotations

import re
import subprocess

import pytest

from tensorforge.backend.pir import engines
from tensorforge.backend.pir.build import IRBuilder, barrier_stmt
from tensorforge.backend.pir.core import (Effect, IRError, MemSpace, Op,
                                          Participants, Stmt, TokenType,
                                          Uniformity)
from tensorforge.backend.pir.engines import (Completion, FenceKind, SyncKind,
                                             SyncObjectType, SyncScope)
from tensorforge.common.basic_types import Datatype
from tensorforge.common.context import Context

TARGETS = [('cuda', 'sm_86'), ('cuda', 'sm_90'), ('hip', 'gfx90a'),
           ('hip', 'gfx942'), ('hip', 'gfx1150'), ('hip', 'gfx1250'),
           ('acpp', 'pvc'), ('oneapi', 'pvc'), ('esimd', 'pvc')]


def _target(backend, arch):
    return Context(arch=arch, backend=backend, fp_type=Datatype.F32).target


def _lines(stmt, target):
    return engines.lower(stmt, target).lines


def _mult(target, factor=2):
    wave = target.hw.vec_unit_length
    return engines.sync_object(Participants.MULT, wave, factor * wave)


# -- the registry ------------------------------------------------------------ #

def test_every_engine_op_is_a_name_pir_knows():
    """An op the registry describes is one `Stmt` accepts by name, and every
    one has a lowering for a target it does not name (`'*'`)."""
    assert set(engines.REGISTRY) == {
        Op.BARRIER, Op.SYNC_INIT, Op.ARRIVE, Op.WAIT_PHASE, Op.FENCE,
        Op.WAIT_COUNTER, Op.SET_PRIORITY}
    for name, spec in engines.REGISTRY.items():
        assert name in Op.KNOWN and spec.name == name
        assert '*' in spec.lower
        assert spec.units >= 0 and spec.pipe


def _built():
    obj = engines.sync_object(Participants.MULT, 32, 64)
    return [engines.barrier(obj), engines.barrier(obj, handoff=True),
            engines.sync_init(obj, objects=4), engines.arrive(obj),
            engines.wait_phase(obj), engines.fence_stmt(FenceKind.SCHEDULE),
            engines.fence_stmt('memory', scope=SyncScope.WARP),
            engines.fence_stmt(FenceKind.SCHEDULE, group=(8, 1, 0)),
            engines.wait_counter('vmcnt', 0), engines.set_priority(2)]


def test_a_statement_says_what_its_entry_says():
    """Every pass asks the statement, not the registry, so the two have to
    agree: the effect, whether it moves, and the operands it holds."""
    for s in _built():
        spec = engines.spec_of(s.op)
        assert s.effect == spec.effect and s.movable == spec.movable
        assert not s.pure
        assert engines.verify_engine(s) == [], s


def test_the_verifier_holds_an_engine_statement_to_its_operands():
    from tensorforge.backend.pir import verify
    bare = Stmt(op=Op.BARRIER, pure=False, movable=False,
                effect=Effect.BARRIER)
    assert any('no object' in d for d in engines.verify_engine(bare))
    wrong = Stmt(op=Op.FENCE, pure=False, movable=False, effect=Effect.UNKNOWN,
                 attrs=(('kind', 'schedule'),))
    assert any('is not a FenceKind' in d for d in engines.verify_engine(wrong))
    loose = Stmt(op=Op.FENCE, pure=False, movable=True, effect=Effect.UNKNOWN,
                 attrs=(('kind', FenceKind.SCHEDULE),))
    assert any('registry says' in d for d in engines.verify_engine(loose))
    with pytest.raises(IRError, match='no object'):
        verify((bare,))


def test_the_completion_kinds_and_the_token_that_carries_one():
    """Four ways an operation's completion is observed; a token says which
    where its issue does, and says nothing -- and prints as it did -- where
    it does not."""
    assert {c.value for c in Completion} == {'group', 'barrier', 'counter',
                                             'sync'}
    assert repr(TokenType()) == 'token'
    assert repr(TokenType(Completion.GROUP)) == 'token<group>'
    assert TokenType() == TokenType(None) != TokenType(Completion.COUNTER)
    spec = engines.REGISTRY
    assert spec[Op.WAIT_PHASE].retires is Completion.BARRIER
    assert spec[Op.WAIT_COUNTER].retires is Completion.COUNTER
    assert spec[Op.ARRIVE].completes is Completion.BARRIER


# -- objects ------------------------------------------------------------------ #

def test_the_object_follows_from_the_participants_and_the_geometry():
    """The block barrier is object 0, a warp's barrier the warp's, a
    multiplication narrower than the warp a mask of it, and one at least as
    wide an object of its own."""
    so = engines.sync_object
    assert (so(Participants.BLOCK, 32).scope, so(Participants.BLOCK, 32).kind
            ) == (SyncScope.BLOCK, SyncKind.IMPLICIT)
    assert so(Participants.MULTGROUP, 32, 48).scope is SyncScope.BLOCK
    assert so(Participants.WAVE, 32, 32).scope is SyncScope.WARP
    narrow = so(Participants.MULT, 32, 16)
    assert (narrow.scope, narrow.kind) == (SyncScope.WARP, SyncKind.IMPLICIT)
    assert not narrow.per_mult
    wide = so(Participants.MULT, 32, 64)
    assert (wide.scope, wide.kind) == (SyncScope.MULT, SyncKind.NAMED)
    assert wide.per_mult and so(Participants.MULT, 1).per_mult
    assert so(Participants.GRID, 32).scope is SyncScope.GRID
    assert repr(wide) == 'sync<mult:named, mult, 64/32>'


def test_an_object_whose_parts_disagree_is_refused():
    with pytest.raises(IRError):
        SyncObjectType(SyncScope.BLOCK, SyncKind.NAMED, Participants.MULT)
    with pytest.raises(IRError):
        SyncObjectType(SyncScope.WARP, SyncKind.IMPLICIT, Participants.BLOCK)
    with pytest.raises(IRError):
        SyncObjectType(SyncScope.MULT, SyncKind.NAMED, Participants.MULT, 0)
    # An object a body creates may sit at any scope.
    SyncObjectType(SyncScope.BLOCK, SyncKind.MBARRIER, Participants.MULT)


@pytest.mark.parametrize('who', list(Participants))
@pytest.mark.parametrize('threads', [None, 16, 32, 64, 96])
def test_the_arrival_level_is_the_participants(who, threads):
    """What `verify` and the barrier placement read off a barrier is what
    `Participants.arrival` says, as the `scope` attribute said before."""
    s = barrier_stmt(who, 32, threads)
    want = who.arrival(threads if threads is not None else 32, 32)
    assert engines.object_of(s).arrival is want
    assert engines.arrival_of(s) is want
    assert engines.settles(s) is want


def test_what_a_rendezvous_settles_and_what_an_arrive_does_not():
    """A barrier and a wait settle at their object's level; an arrive has
    waited for nobody and a setup is no rendezvous of the body.  A call that
    holds a block barrier (the launch cursor's `next`) settles the block."""
    obj = engines.sync_object(Participants.MULT, 32, 64)
    assert engines.settles(engines.wait_phase(obj)) is Uniformity.MULT
    assert engines.settles(engines.arrive(obj)) is None
    assert engines.settles(engines.sync_init(obj)) is None
    assert engines.arrival_of(engines.sync_init(obj)) is Uniformity.BLOCK
    assert engines.settles(engines.fence_stmt(FenceKind.SCHEDULE)) is None
    call = Stmt(op=Op.CALL, pure=False, movable=False, effect=Effect.BARRIER,
                attrs=(('callee', 'launchCursor0.next'),))
    assert engines.settles(call) is Uniformity.BLOCK


def test_a_block_takes_an_object_a_multiplication_only_where_one_meets_alone():
    """The budget the thread-block policy is held to: one named object a
    multiplication of the block, wherever a barrier meets a multiplication
    on its own, at any depth -- and none for the block's or a warp's."""
    b = IRBuilder(fptype=Datatype.F32)
    with b.for_(0, 4, 1):
        b.barrier(Participants.MULT, threads=64)
    assert engines.object_demand(b.finish(), 6) == 6
    b = IRBuilder(fptype=Datatype.F32)
    b.barrier(Participants.BLOCK)
    b.barrier(Participants.WAVE, threads=32)
    assert engines.object_demand(b.finish(), 6) == 0


# -- what each backend writes ------------------------------------------------ #

def _lexic_barrier(target, who, threads, handoff):
    """The barrier as the emitter wrote it before the registry
    (`Emitter._sync` and the hand-off fence at dd12ebfb), from the lexic."""
    lex = target.lexic
    if who is Participants.WAVE:
        text = lex.sync_simd()
    elif who is Participants.GRID:
        text = lex.sync_grid()
    elif who is Participants.MULT and threads is not None:
        if not target.sync_mult(threads):
            raise IRError('no rendezvous of one multiplication')
        text = lex.sync_mult(threads, target.hw)
    else:
        text = lex.sync_block()
    if not text and handoff:
        text = lex.handoff_fence()
    return () if text is None else (text,)


@pytest.mark.parametrize('backend,arch', TARGETS)
def test_a_barrier_is_written_as_the_lexic_wrote_it(backend, arch):
    """Every participant set, width and hand-off, against the lexic: the
    combined form on an object is the instruction the barrier was."""
    target = _target(backend, arch)
    wave = target.hw.vec_unit_length
    for who in Participants:
        for threads in (None, wave // 4, wave // 2, wave, 2 * wave,
                        3 * wave // 2, 4 * wave):
            for handoff in (False, True):
                stmt = barrier_stmt(who, wave, threads, handoff)
                try:
                    want = _lexic_barrier(target, who, threads, handoff)
                except (IRError, NotImplementedError) as exc:
                    with pytest.raises(type(exc)):
                        _lines(stmt, target)
                    continue
                assert _lines(stmt, target) == want, (who, threads, handoff)
                assert _lines(engines.wait_phase(engines.object_of(stmt)),
                              target) == _lexic_barrier(target, who, threads,
                                                        False)


def test_without_a_target_a_barrier_is_the_block_s():
    """The IR-level tests build bodies without one."""
    for who in Participants:
        assert _lines(barrier_stmt(who, 1, 64), None) == ('__syncthreads();',)


def test_the_setup_is_the_lexic_s_prologue_where_the_target_has_objects():
    """gfx1250 declares its objects and has every wave join its own; an
    NVIDIA id and every implicit object need nothing."""
    for backend, arch in TARGETS:
        target = _target(backend, arch)
        obj = _mult(target)
        want = tuple(target.lexic.named_barrier_prologue(
            obj.threads, 3, target.hw))
        assert _lines(engines.sync_init(obj, objects=3), target) == want
        block = engines.sync_object(Participants.BLOCK,
                                    target.hw.vec_unit_length)
        assert _lines(engines.sync_init(block), target) == ()
    gfx1250 = _target('hip', 'gfx1250')
    assert _lines(engines.sync_init(_mult(gfx1250), objects=3), gfx1250) == (
        '__shared__ __amdgpu_named_workgroup_barrier_t tfBarriers[3];',
        'tensorforge::joinNamedBarrier<2>(&tfBarriers[threadIdx.y]);')


def test_an_arrive_is_the_first_half_of_the_named_rendezvous():
    """`bar.arrive` on the multiplication's id, the signal of its object on
    gfx1250; refused where the hardware has no split, rather than spelled as
    a whole rendezvous."""
    sm86 = _target('cuda', 'sm_86')
    assert _lines(engines.arrive(_mult(sm86)), sm86) == (
        'asm volatile("barrier.arrive %0, %1;" :: "r"(threadIdx.y + 1), '
        '"r"(64) : "memory");',)
    gfx1250 = _target('hip', 'gfx1250')
    assert _lines(engines.arrive(_mult(gfx1250)), gfx1250) == (
        'tensorforge::arriveNamedBarrier(&tfBarriers[threadIdx.y]);',)
    for backend, arch in (('hip', 'gfx942'), ('oneapi', 'pvc'),
                          ('esimd', 'pvc')):
        target = _target(backend, arch)
        with pytest.raises(IRError):
            _lines(engines.arrive(_mult(target)), target)
    block = engines.sync_object(Participants.BLOCK, 32)
    with pytest.raises(IRError):
        _lines(engines.arrive(block), sm86)


def test_an_object_of_the_transfer_engines_is_not_lowered_yet():
    mbarrier = SyncObjectType(SyncScope.BLOCK, SyncKind.MBARRIER,
                              Participants.BLOCK)
    sm90 = _target('cuda', 'sm_90')
    for stmt in (engines.barrier(mbarrier), engines.arrive(mbarrier, tx=128),
                 engines.wait_phase(mbarrier), engines.sync_init(mbarrier)):
        with pytest.raises(IRError, match='transfers'):
            _lines(stmt, sm90)


def test_a_schedule_fence_is_the_statement_the_compiler_holds_to():
    """Where the target has one, `pin_fence`'s, with whether every lane has
    to arrive; on AMD the scheduling barrier the batch loop's head had."""
    for backend, arch in TARGETS:
        target = _target(backend, arch)
        got = engines.lower(engines.fence_stmt(FenceKind.SCHEDULE), target)
        if backend == 'esimd':
            assert got.lines == ('tensorforge::intel_esimd::fence<128>();',)
            continue
        fence = target.lexic.pin_fence()
        assert got == ((fence[0],), fence[1]) if fence else got == ((), False)
        if backend == 'hip':
            assert got.lines == (target.lexic.loop_body_fence(),)
    sm86 = _target('cuda', 'sm_86')
    assert engines.lower(engines.fence_stmt('schedule'), sm86) == (
        ('__syncwarp();',), True)


def test_a_scheduling_group_is_amd_s_and_a_hint_elsewhere():
    group = engines.fence_stmt(FenceKind.SCHEDULE, group=(0x120, 2, 0))
    gfx942 = _target('hip', 'gfx942')
    assert _lines(group, gfx942) == (
        '__builtin_amdgcn_sched_group_barrier(288, 2, 0);',)
    for backend, arch in (('cuda', 'sm_86'), ('oneapi', 'pvc'),
                          ('esimd', 'pvc')):
        assert _lines(group, _target(backend, arch)) == ()
    masked = engines.fence_stmt(FenceKind.SCHEDULE, mask=8)
    assert _lines(masked, gfx942) == ('__builtin_amdgcn_sched_barrier(8);',)


def test_the_explicit_vector_writes_vISA_s_fence_with_its_mask():
    esimd = _target('esimd', 'pvc')
    for mask, kind in ((128, FenceKind.SCHEDULE), (33, FenceKind.MEMORY),
                       (161, FenceKind.BOTH)):
        assert _lines(engines.fence_stmt(kind, mask=mask), esimd) == (
            f'tensorforge::intel_esimd::fence<{mask}>();',)


def test_a_memory_fence_of_the_wave_and_of_the_block():
    """At the wave's scope the hand-off fence where the wave barrier is no
    instruction, the wave barrier where it is; at the block's the
    work-group's release-acquire fence.  Refused where there is none."""
    warp = engines.fence_stmt(FenceKind.MEMORY, scope=SyncScope.WARP)
    block = engines.fence_stmt(FenceKind.MEMORY)
    gfx942, sm86 = _target('hip', 'gfx942'), _target('cuda', 'sm_86')
    oneapi, esimd = _target('oneapi', 'pvc'), _target('esimd', 'pvc')
    assert _lines(warp, gfx942) == (gfx942.lexic.handoff_fence(),)
    assert engines.lower(warp, sm86) == (('__syncwarp();',), True)
    assert _lines(warp, oneapi) == (
        'sycl::group_barrier(item.get_sub_group());',)
    assert _lines(warp, esimd) == ()
    assert _lines(block, sm86) == ('__threadfence_block();',)
    assert _lines(block, gfx942) == (
        '__builtin_amdgcn_fence(__ATOMIC_ACQ_REL, "workgroup");',)
    assert _lines(block, oneapi) == (
        'sycl::atomic_fence(sycl::memory_order::acq_rel, '
        'sycl::memory_scope::work_group);',)
    with pytest.raises(IRError):
        _lines(block, esimd)
    grid = engines.fence_stmt(FenceKind.MEMORY, scope=SyncScope.GRID)
    with pytest.raises(IRError):
        _lines(grid, sm86)


def test_a_fence_of_both_writes_one_statement_where_one_does_both():
    both = engines.fence_stmt(FenceKind.BOTH, scope=SyncScope.WARP)
    assert _lines(both, _target('cuda', 'sm_86')) == ('__syncwarp();',)
    assert _lines(both, _target('hip', 'gfx942')) == (
        '__builtin_amdgcn_fence(__ATOMIC_ACQ_REL, "wavefront");',
        '__builtin_amdgcn_sched_barrier(0);')


def test_counter_waits_per_target():
    """CUDA's commit groups as the copies' own wait spells them; CDNA's
    `s_waitcnt` immediate with every other field at its maximum; gfx1250's
    asynchronous counters; refused where the target has no such counter."""
    sm86 = _target('cuda', 'sm_86')
    assert _lines(engines.wait_counter('group', 1), sm86) == (
        sm86.lexic.wait_async(1),)
    for arch in ('gfx90a', 'gfx942'):
        target = _target('hip', arch)
        assert _lines(engines.wait_counter('vmcnt', 0), target) == (
            '__builtin_amdgcn_s_waitcnt(3952);',)        # 0x0f70
        assert _lines(engines.wait_counter('vmcnt', 21), target) == (
            f'__builtin_amdgcn_s_waitcnt({0x4f75});',)   # 21 = 0b01_0101
        assert _lines(engines.wait_counter('lgkmcnt', 3), target) == (
            f'__builtin_amdgcn_s_waitcnt({0xc37f});',)
        with pytest.raises(IRError):
            _lines(engines.wait_counter('vmcnt', 64), target)
    gfx1250 = _target('hip', 'gfx1250')
    assert _lines(engines.wait_counter('asynccnt', 1), gfx1250) == (
        '__builtin_amdgcn_s_wait_asynccnt(1);',)
    assert _lines(engines.wait_counter('tensorcnt', 0), gfx1250) == (
        '__builtin_amdgcn_s_wait_tensorcnt(0);',)
    for backend, arch, counter in (('hip', 'gfx1150', 'vmcnt'),
                                   ('hip', 'gfx942', 'asynccnt'),
                                   ('cuda', 'sm_86', 'vmcnt'),
                                   ('oneapi', 'pvc', 'group')):
        with pytest.raises(IRError):
            _lines(engines.wait_counter(counter, 0), _target(backend, arch))


def test_a_wave_priority_is_amd_s_and_nothing_elsewhere():
    for arch in ('gfx90a', 'gfx942', 'gfx1250'):
        assert _lines(engines.set_priority(2), _target('hip', arch)) == (
            '__builtin_amdgcn_s_setprio(2);',)
    for backend, arch in (('cuda', 'sm_86'), ('oneapi', 'pvc')):
        assert _lines(engines.set_priority(2), _target(backend, arch)) == ()
    with pytest.raises(IRError):
        engines.set_priority(4)


def test_a_wave_priority_for_the_odd_waves_only():
    """`amd_setprio=alternate`'s: behind the wave's index in the block, read
    from the first lane, so that the branch is uniform."""
    odd = engines.set_priority(2, waves='odd')
    assert engines.verify_engine(odd) == []
    (line,) = _lines(odd, _target('hip', 'gfx942'))
    assert line.startswith('if (__builtin_amdgcn_readfirstlane(')
    assert line.endswith(') & 1) { __builtin_amdgcn_s_setprio(2); }')
    assert _lines(odd, _target('cuda', 'sm_90')) == ()
    with pytest.raises(IRError):
        engines.set_priority(2, waves='even')


def test_a_fence_of_the_schedule_need_not_be_a_wall():
    """Placed behind the passes for the device compiler alone: no effect,
    still pinned, written as a wall of the schedule is.  A memory fence is a
    wall by what it does."""
    gfx942 = _target('hip', 'gfx942')
    soft = engines.fence_stmt(FenceKind.SCHEDULE, wall=False)
    hard = engines.fence_stmt(FenceKind.SCHEDULE)
    assert soft.effect == Effect.NONE and hard.effect == Effect.UNKNOWN
    assert not soft.movable and engines.verify_engine(soft) == []
    assert _lines(soft, gfx942) == _lines(hard, gfx942) == (
        '__builtin_amdgcn_sched_barrier(0);',)
    with pytest.raises(IRError):
        engines.fence_stmt(FenceKind.MEMORY, wall=False)


# -- the emitter and the passes ---------------------------------------------- #

class _Sink:
    def __init__(self):
        self.lines = []

    def __call__(self, text, *args, **kwargs):
        self.lines.append(text)

    def __getattr__(self, _):
        return lambda *a, **k: None


class _Metrics:
    def __init__(self):
        self.code = 0
        self.mix = []

    def record_code(self, units):
        self.code += units

    def record_mix(self, category, issued, copies):
        self.mix.append(category)


def test_the_emitter_writes_and_counts_an_engine_op_as_the_registry_says():
    """What is written is the lowering; what is counted is the entry's units
    and pipe -- a fence one unit of `other`, as the raw text was."""
    from tensorforge.backend.pir.emit import Emitter
    gfx942 = _target('hip', 'gfx942')
    b = IRBuilder(fptype=Datatype.F32, context=Context(
        arch='gfx942', backend='hip', fp_type=Datatype.F32))
    b.fence(FenceKind.SCHEDULE)
    b.set_priority(1)
    b.wait_counter('vmcnt', 0)
    b.barrier(Participants.BLOCK)
    sink, metrics = _Sink(), _Metrics()
    Emitter(sink, gfx942, metrics).run(b.finish())
    assert sink.lines == ['__builtin_amdgcn_sched_barrier(0);',
                          '__builtin_amdgcn_s_setprio(1);',
                          '__builtin_amdgcn_s_waitcnt(3952);',
                          '__syncthreads();']
    assert metrics.code == 4
    assert metrics.mix == ['other', 'other', 'sync', 'barrier']


def _rawfence():
    return Stmt(op=Op.RAWSTMT, pure=False, movable=False,
                effect=Effect.UNKNOWN, text='__builtin_amdgcn_sched_barrier(0);')


def test_a_fence_is_a_wall_where_the_raw_text_was_one():
    """Relaxing that is for later: for now nothing moves across a fence that
    did not move across the text it replaces."""
    from tensorforge.backend.pir.schedule import can_reorder, may_cross
    b = IRBuilder(fptype=Datatype.F32, arena='shrMem')
    buf = b.alloc(Datatype.F32, (8,), MemSpace.SHARED, hint='s', offset=0)
    b.load(buf, 0, hint='x')
    read = next(s for s in b.finish() if s.op == Op.LOAD)
    for fence in (_rawfence(), engines.fence_stmt(FenceKind.SCHEDULE),
                  engines.fence_stmt(FenceKind.MEMORY)):
        assert not can_reorder(read, fence) and not can_reorder(fence, read)
        assert not may_cross(read, fence)


def test_a_body_of_every_op_verifies_and_comes_through_the_pipeline():
    """Every op as the builder makes it: legal where every thread arrives,
    and in its place, all of it, behind the standard passes."""
    from tensorforge.backend import pir
    b = IRBuilder(fptype=Datatype.F32)
    obj = b.sync_object(Participants.MULT, 64)
    b.sync_init(obj, objects=2)
    b.arrive(obj)
    b.wait_phase(obj)
    b.fence(FenceKind.SCHEDULE)
    b.fence(FenceKind.MEMORY)
    b.wait_counter('vmcnt', 0)
    b.set_priority(1)
    b.barrier()
    body = b.finish()
    pir.verify(body)
    assert pir.optimize(body) == body


def test_dce_keeps_a_priority_and_a_fence():
    """Neither has a result or a memory effect a reader asks for: what each
    does is where it stands."""
    from tensorforge.backend.pir import dce
    body = (engines.set_priority(2), engines.fence_stmt(FenceKind.SCHEDULE))
    assert dce(body) == body


def test_the_pin_fence_is_the_registry_s():
    """`Emitter._pin_load` writes the fence of the schedule, and leaves out
    one every lane has to reach where the lanes do not agree."""
    from tensorforge.backend.pir.emit import Emitter
    b = IRBuilder(fptype=Datatype.F32)
    buf = b.alloc(Datatype.F32, (8,), MemSpace.GLOBAL, hint='g', offset=0)
    for i in range(4):
        b.load(buf, i, hint='x')
    load = next(s for s in b.finish() if s.op == Op.LOAD)
    for backend, arch, converged, want in (
            ('cuda', 'sm_86', True, ['__syncwarp();']),
            ('cuda', 'sm_86', False, []),
            ('hip', 'gfx942', False, ['__builtin_amdgcn_sched_barrier(0);'])):
        sink = _Sink()
        emitter = Emitter(sink, _target(backend, arch))
        emitter._pin_every, emitter._converged = 1, converged
        emitter._pin_load(load)
        assert sink.lines == want


def test_the_stacked_product_groups_its_reads_with_fences():
    """`Options.amd_matrix_k4_pipeline` at 2: an issue and a group of reads,
    each a fence of the schedule with its group."""
    from tensorforge.backend.instructions.compute.primitives.amd import stacked
    b = IRBuilder(fptype=Datatype.F32)
    stacked._interleave(b, 2)
    groups = [s.attr('group') for s in b.finish()]
    assert groups == [(stacked.SCHED_MFMA, 1, 0), (stacked.SCHED_READS, 2, 0)] * 2
    gfx942 = _target('hip', 'gfx942')
    assert [_lines(s, gfx942)[0] for s in b.finish()][:2] == [
        '__builtin_amdgcn_sched_group_barrier(8, 1, 0);',
        '__builtin_amdgcn_sched_group_barrier(288, 2, 0);']


def test_the_kernel_opens_with_the_setup_of_its_objects():
    """The gfx1250 prologue is a body of its own: one setup of the objects,
    one a multiplication of the block (`Generator._barrier_prologue`)."""
    from test_named_barriers import _fold
    gen = _fold('gfx1250', 64)
    prologue = gen._barrier_prologue()
    assert [s.op for s in prologue] == [Op.SYNC_INIT]
    assert prologue[0].attr('objects') == gen._launch.mults_per_block
    assert engines.object_of(prologue[0]).per_mult
    assert _fold('gfx1250', 32)._barrier_prologue() == ()


# -- what a device compiler makes of it -------------------------------------- #

def _assemble(source, arch, tmp_path):
    from harness import syntax
    from tensorforge import toolchain
    hipcc = syntax.device_compiler('hip')
    if hipcc is None:
        pytest.skip('hipcc not found')
    path = tmp_path / f'{arch}.hip'
    path.write_text(source)
    run = subprocess.run([hipcc, '-O3', '-std=c++17', f'--offload-arch={arch}',
                          '-I', toolchain.include_dir(), '--cuda-device-only',
                          '-S', '-o', '-', str(path)],
                         capture_output=True, text=True, timeout=600)
    if run.returncode and re.search(r'unsupported .*architecture|'
                                    r'invalid offload arch', run.stderr):
        pytest.skip(f'hipcc does not know {arch}')
    assert run.returncode == 0, run.stderr[-3000:]
    return [line.strip() for line in run.stdout.splitlines()
            if line.startswith('\t') and not line.strip().startswith(';')]


def _kernel(lines, decls=()):
    body = '\n'.join(f'  {line}' for line in lines)
    head = '\n'.join(f'  {d}' for d in decls)
    return ('#include "tensorforge_device/hip.h"\n'
            '__global__ void __launch_bounds__(256) k(float *x, '
            'const float *y) {\n' + head + '\n'
            '  float v[8];\n'
            '  for (int i = 0; i < 8; ++i) v[i] = y[threadIdx.x + 256 * i];\n'
            + body + '\n'
            '  float a = 0.f;\n'
            '  for (int i = 0; i < 8; ++i) a += v[i];\n'
            '  x[threadIdx.x] = a;\n}\n')


def test_the_split_rendezvous_compiles_to_a_signal_and_a_meet(tmp_path):
    """gfx1250: the setup, then a signal alone for the arrive, and a signal
    and a wait for the wait on the named object."""
    target = _target('hip', 'gfx1250')
    obj = _mult(target)
    init = _lines(engines.sync_init(obj, objects=4), target)
    body = (_lines(engines.arrive(obj), target)
            + _lines(engines.wait_phase(obj), target)
            + _lines(engines.wait_counter('asynccnt', 1), target)
            + _lines(engines.set_priority(2), target))
    asm = _assemble(_kernel(body, decls=init), 'gfx1250', tmp_path)
    join = asm.index('s_barrier_join m0')
    signals = [i for i, ins in enumerate(asm) if ins == 's_barrier_signal m0']
    assert len(signals) == 2 and all(i > join for i in signals)
    assert asm.count('s_barrier_wait 1') == 1
    assert asm[signals[1] + 1] == 's_barrier_wait 1'
    assert 's_wait_asynccnt 0x1' in asm and 's_setprio 2' in asm


def test_a_counter_wait_compiles_to_the_count(tmp_path):
    """gfx942: eight loads in flight, and the wait lets five stay."""
    target = _target('hip', 'gfx942')
    asm = _assemble(_kernel(_lines(engines.wait_counter('vmcnt', 5), target)),
                    'gfx942', tmp_path)
    assert any(re.match(r's_waitcnt vmcnt\(5\)', ins) for ins in asm), asm


def test_the_split_named_barrier_is_bar_arrive_and_bar_sync(tmp_path):
    """sm_86: the arrive is `barrier.arrive` and the wait `barrier.sync`,
    both on the multiplication's id and its count, and the counter wait on
    the commit groups is `cp.async.wait_group`."""
    from harness import syntax
    nvcc = syntax.device_compiler('cuda')
    if nvcc is None:
        pytest.skip('nvcc not found')
    target = _target('cuda', 'sm_86')
    obj = _mult(target)
    body = (_lines(engines.arrive(obj), target)
            + _lines(engines.wait_phase(obj), target)
            + _lines(engines.wait_counter('group', 1), target))
    path = tmp_path / 'split.cu'
    path.write_text('#include <cuda_pipeline.h>\n'
                    '__global__ void k(float *x) {\n'
                    + '\n'.join(f'  {line}' for line in body)
                    + '\n  x[threadIdx.x] = 1.f;\n}\n')
    run = subprocess.run([nvcc, '-arch=sm_86', '-ptx', '-o', '-', str(path)],
                         capture_output=True, text=True, timeout=600)
    assert run.returncode == 0, run.stderr[-3000:]
    assert re.search(r'barrier\.arrive\s+%r\d+, %r\d+;', run.stdout)
    assert re.search(r'barrier\.sync\s+%r\d+, %r\d+;', run.stdout)
    assert re.search(r'cp\.async\.wait_group\s+1;', run.stdout)
