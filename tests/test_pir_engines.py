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

import contextlib
import re
import subprocess

import pytest

from tensorforge.backend.pir import engines
from tensorforge.backend.pir.build import IRBuilder, barrier_stmt
from tensorforge.backend.pir.core import (BOOL, INDEX, Effect, IRError,
                                          MemSpace, Op, Participants, Stmt,
                                          TokenType, Uniformity)
from tensorforge.backend.pir.engines import (Completion, FenceKind, SyncKind,
                                             SyncObjectType, SyncScope)
from tensorforge.common.basic_types import Datatype
from tensorforge.common.context import Context, Options

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
        Op.WAIT_COUNTER, Op.SET_PRIORITY, Op.COMMIT_GROUP, Op.MATRIX_MAC,
        Op.TMEM_ALLOC, Op.TMEM_DEALLOC, Op.TMEM_LD, Op.TMEM_ST}
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
    it does not.  Not part of the type's identity: a loop that carries a
    token declares its argument as `TOKEN`, whatever the kind."""
    assert {c.value for c in Completion} == {'group', 'barrier', 'counter',
                                             'sync'}
    assert repr(TokenType()) == 'token'
    assert repr(TokenType(Completion.GROUP)) == 'token<group>'
    assert TokenType() == TokenType(None) == TokenType(Completion.COUNTER)
    assert hash(TokenType()) == hash(TokenType(Completion.BARRIER))
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


#: What a barrier may cover: every set but the warpgroup, which is what
#: wgmma executes together and no barrier's object (`sync_object`).
BARRIER_SETS = [p for p in Participants if p is not Participants.WARPGROUP]


def test_a_warpgroup_is_no_barrier_s():
    with pytest.raises(IRError, match='warpgroup'):
        engines.sync_object(Participants.WARPGROUP, 32, 128)


@pytest.mark.parametrize('who', BARRIER_SETS)
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
    for who in BARRIER_SETS:
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
    for who in BARRIER_SETS:
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


def test_an_mbarrier_is_spelled_as_its_cursor_and_nothing_else():
    """A described mbarrier has no storage and no cursor to name, and an
    arrive at one is the bulk copy's own: refused.  The rendezvous of all
    participants is no use of one either."""
    mbarrier = SyncObjectType(SyncScope.BLOCK, SyncKind.MBARRIER,
                              Participants.BLOCK)
    sm90 = _target('cuda', 'sm_90')
    for stmt in (engines.barrier(mbarrier), engines.arrive(mbarrier, tx=128),
                 engines.wait_phase(mbarrier), engines.sync_init(mbarrier)):
        with pytest.raises(IRError):
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
        assert got == (engines.Lowered((fence[0],), fence[1]) if fence
                       else engines.Lowered((), False))
        if backend == 'hip':
            assert got.lines == (target.lexic.loop_body_fence(),)
    sm86 = _target('cuda', 'sm_86')
    assert engines.lower(engines.fence_stmt('schedule'), sm86) == (
        engines.Lowered(('__syncwarp();',), True))


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
    assert engines.lower(warp, sm86) == engines.Lowered(('__syncwarp();',),
                                                        True)
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
    # And the compiler's fence behind it (`test_async_lds`).
    assert _lines(engines.wait_counter('asynccnt', 1), gfx1250)[:-1] == (
        '__builtin_amdgcn_s_wait_asynccnt(1);',)
    # The tensor copies' wait, and the compiler's fence behind it
    # (`engines.TENSOR_FENCE`).
    assert _lines(engines.wait_counter('tensorcnt', 0), gfx1250) == (
        '__builtin_amdgcn_s_wait_tensorcnt(0);',
        '__atomic_signal_fence(__ATOMIC_SEQ_CST);')
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


# -- transfer engines (phase 1a) --------------------------------------------- #

def _bulk_body():
    """A window, its bulk copy through the channel's mbarrier, and the wait
    for it -- on sm_90, where the target spells one."""
    from tensorforge.backend.pir.core import SIZE, BufferType
    ctx = Context(arch='sm_90', backend='cuda', fp_type=Datatype.F32)
    b = IRBuilder(fptype=Datatype.F32, arena='shrMem', context=ctx)
    k = b.extern_value('element', SIZE, uniform=Uniformity.MULT, hint='k')
    src = b.decl_expr(
        'const float *const __restrict__ glb_m0', '&m0[{0} * 64]',
        BufferType(Datatype.F32, (64,), MemSpace.GLOBAL, readonly=True),
        'm0', args=(k,), kind=Effect.READ, space=MemSpace.GLOBAL,
        hint='glb_m0', extern='glb_m0')
    s0 = b.alloc(Datatype.F32, (64,), MemSpace.SHARED, hint='s0',
                 extern='s0', offset=0)
    barrier = b.bulk_channel('s0', 'threadIdx.x == 0', threads=32)
    tok = b.copy_bulk(s0, src, barrier=barrier, dst_index=(0,),
                      src_index=(0,), elems=64)
    b.wait(tok)
    from tensorforge.backend.pir.allocate import MULT, allocate
    body, _ = allocate(b.finish(), arenas={'shrMem': MULT})
    return ctx, b, barrier, body


def test_every_transfer_engine_completes_one_way_and_is_counted_by_it():
    """The class a wait counts a transfer in follows from how its engine
    completes: commit groups, a counter, a barrier."""
    assert set(engines.TRANSFER_ENGINES) == {
        'scalar', 'vector', 'cp_async', 'lds_dma', 'bulk', 'load', 'buffer', 'async_lds',
        'tdm'}
    assert engines.counter_of('cp_async') == 'copy'
    assert engines.counter_of('lds_dma') == 'load'
    assert engines.counter_of('load') == 'load'
    assert engines.counter_of('bulk') == 'bulk'
    for name, engine in engines.TRANSFER_ENGINES.items():
        assert all(op in Op.KNOWN for op in engine.ops), name


@pytest.mark.parametrize('arch,backend,engine', [
    ('sm_86', 'cuda', 'cp_async'), ('sm_86', 'hip', 'cp_async'),
    ('gfx942', 'hip', 'lds_dma'), ('gfx90a', 'hip', 'lds_dma'),
    ('gfx1150', 'hip', 'cp_async'), ('pvc', 'oneapi', 'cp_async')])
def test_a_copy_goes_through_the_engine_its_target_has(arch, backend,
                                                       engine):
    """AMD's LDS-DMA where HIP spells one for the device, `cp.async`
    elsewhere -- which a target without the path lowers synchronously."""
    target = _target(backend, arch)
    assert engines.copy_engine(target) == engine
    assert engines.copy_engine(None) == 'cp_async'


def test_a_copy_says_its_engine_width_and_cache_policy():
    """Sixteen bytes of `cp.async` go past L1 (`cg`), four and eight through
    it (`ca`); a token reports the engine's kind of completion."""
    ctx = Context(arch='sm_86', backend='cuda', fp_type=Datatype.F32)
    b = IRBuilder(fptype=Datatype.F32, arena='shrMem', context=ctx)
    s0 = b.alloc(Datatype.F32, (64,), MemSpace.SHARED, hint='s0', offset=0)
    g = b.alloc(Datatype.F32, (64,), MemSpace.GLOBAL, hint='g', extern='g')
    wide = b.copy_async(s0, g, dst_index=(0,), src_index=(0,), elems=4)
    narrow = b.copy_async(s0, g, dst_index=(4,), src_index=(4,), elems=2)
    copies = [s for s in b.finish() if s.op == Op.COPY_ASYNC]
    assert [(c.attr('engine'), c.attr('width'), c.attr('cache'))
            for c in copies] == [('cp_async', 16, 'cg'), ('cp_async', 8, 'ca')]
    assert all(c.counter == 'copy' for c in copies)
    assert wide.type.completes is Completion.GROUP
    assert narrow.type == TokenType()
    amd = Context(arch='gfx942', backend='hip', fp_type=Datatype.F32)
    b = IRBuilder(fptype=Datatype.F32, arena='shrMem', context=amd)
    s0 = b.alloc(Datatype.F32, (64,), MemSpace.SHARED, hint='s0', offset=0)
    g = b.alloc(Datatype.F32, (64,), MemSpace.GLOBAL, hint='g', extern='g')
    tok = b.copy_async(s0, g, dst_index=(0,), src_index=(0,))
    copy = next(s for s in b.finish() if s.op == Op.COPY_ASYNC)
    assert copy.attr('engine') == 'lds_dma' and copy.attr('cache') is None
    assert copy.counter == 'load' and tok.type.completes is Completion.COUNTER


def test_a_load_and_a_store_say_what_a_lane_moves():
    """`width` in bytes, a wide access one of the vector engine: what the
    emitter spells as the value's type, and the verifier holds the two to
    each other."""
    from tensorforge.backend import pir
    from tensorforge.backend.pir.core import ScalarType
    b = IRBuilder(fptype=Datatype.F32)
    g = b.alloc(Datatype.F32, (64,), MemSpace.GLOBAL, hint='g', extern='g')
    one = b.load(g, 0, hint='x')
    four = b.load(g, 4, type_=ScalarType(Datatype.F32, 4), hint='q', align=16)
    b.store(g, four, 8, align=16)
    b.store(g, 1.0, 12)
    body = b.finish()
    widths = [(s.op, s.attr('width'), engines.engine_of(s)) for s in body
              if s.op in (Op.LOAD, Op.STORE)]
    assert widths == [(Op.LOAD, 4, 'scalar'), (Op.LOAD, 16, 'vector'),
                      (Op.STORE, 16, 'vector'), (Op.STORE, 4, 'scalar')]
    pir.verify(body)
    stale = tuple(s.with_attr('width', 8) if s.op == Op.LOAD and
                  s.target[0] is one else s for s in body)
    with pytest.raises(IRError, match='width'):
        pir.verify(stale)


@pytest.mark.parametrize('arch,backend,width,cache', [
    ('sm_86', 'cuda', 1, 'cg'), ('sm_86', 'cuda', 4, None),
    ('gfx942', 'hip', 4, 'cg'), ('pvc', 'oneapi', 1, None)])
def test_a_cache_hint_is_the_one_the_target_has(arch, backend, width, cache):
    """The request (`nontemporal`) stays for the passes that read it; the
    policy is what the target spells for the access's type and width --
    CUDA's `__ldcg` for one element, AMD's builtin for any, SYCL none."""
    from tensorforge.backend.pir.core import ScalarType
    ctx = Context(arch=arch, backend=backend, fp_type=Datatype.F32)
    b = IRBuilder(fptype=Datatype.F32, context=ctx)
    g = b.alloc(Datatype.F32, (64,), MemSpace.GLOBAL, hint='g', extern='g')
    type_ = None if width == 1 else ScalarType(Datatype.F32, width)
    b.load(g, 0, type_=type_, nontemporal='cg', align=16)
    load = next(s for s in b.finish() if s.op == Op.LOAD)
    assert load.attr('nontemporal') == 'cg' and load.attr('cache') == cache
    bare = IRBuilder(fptype=Datatype.F32)
    g = bare.alloc(Datatype.F32, (64,), MemSpace.GLOBAL, hint='g', extern='g')
    bare.load(g, 0, nontemporal='cg')
    load = next(s for s in bare.finish() if s.op == Op.LOAD)
    assert load.attr('cache', engines.UNRESOLVED) is engines.UNRESOLVED


def test_the_bulk_path_is_an_mbarrier_its_copy_and_the_wait_on_it():
    """The channel sets the object up in its storage, the copy names it,
    and the wait is the wait for its phase, consuming the copy's token as a
    `wait` would -- every one spelled as the cursor's calls, as before."""
    from tensorforge.backend import pir
    from tensorforge.backend.pir.emit import Emitter
    ctx, b, barrier, body = _bulk_body()
    assert barrier.type.kind is SyncKind.MBARRIER and barrier.id < 0
    pir.verify(body)
    out, diag = pir.schedule_async(body)
    assert not diag
    wait = next(s for s in out if s.op == Op.WAIT_PHASE)
    assert wait.movable and wait.effect & Effect.ASYNC
    assert not wait.effect & Effect.BARRIER
    assert engines.settles(wait) is None
    sink = _Sink()
    Emitter(sink, ctx).run(out)
    text = '\n'.join(sink.lines)
    assert ('tensorforge::BulkCursor s0_bulk = tensorforge::bulk_arm('
            's0_barrier, threadIdx.x == 0);' in text)
    assert 'tensorforge::bulk_copy(s0_bulk, &s0[0], &glb_m0[0], 256);' in text
    assert text.rstrip().endswith('tensorforge::bulk_wait(s0_bulk);')


def test_a_drain_waits_on_every_cursor_by_its_object():
    """A full drain names no token, and waits on every cursor of the body,
    by the name the objects' setups give them."""
    from tensorforge.backend import pir
    from tensorforge.backend.pir.emit import Emitter
    ctx, b, barrier, body = _bulk_body()
    drain = IRBuilder(fptype=Datatype.F32, context=ctx)
    drain.wait()
    out, _ = pir.schedule_async(body + drain.finish())
    sink = _Sink()
    Emitter(sink, ctx).run(out)
    assert sink.lines[-1] == 'tensorforge::bulk_wait(s0_bulk);'
    assert sink.lines.count('tensorforge::bulk_wait(s0_bulk);') == 2


def test_a_wait_of_the_engines_retires_only_where_it_names_tokens():
    """`core.retires`: a `wait`, and a wait on an object or a counter that
    names the tokens it retires -- not the rendezvous of a named barrier."""
    from tensorforge.backend.pir.core import retires
    _, _, _, body = _bulk_body()
    wait = next(s for s in body if s.op == Op.WAIT_PHASE)
    assert retires(wait)
    named = engines.wait_phase(engines.sync_object(Participants.MULT, 32, 64))
    assert not retires(named)
    assert not retires(engines.wait_counter('vmcnt', 0))


# -- the reference interpreter ------------------------------------------------ #

def test_the_interpreter_reads_an_arrive_as_half_a_rendezvous():
    """`bar.arrive` and gfx1250's signal (`arriveNamedBarrier`) are arrivals
    that wait for nobody: the lanes that wait at the same barrier next are
    ordered behind what the arriving lanes did ahead of their arrival, and
    a lane that arrived is no lane the barrier missed."""
    from tensorforge.reference import kernel_eval
    races = kernel_eval.Races()
    for text in ('asm volatile("barrier.arrive %0, %1;" :: '
                 '"r"(threadIdx.y + 1), "r"(64) : "memory")',
                 'tensorforge::arriveNamedBarrier(&tfBarriers[tfMult])'):
        assert races.statement_kind(text) == 'arrive'
        assert races.reach(text)[0] == 'named'
    assert races.reach('tensorforge::syncNamedBarrier(&tfBarriers[tfMult])'
                       ) == ('named', 'tfMult', None)
    assert races.statement_kind(
        'tensorforge::syncNamedBarrier(&tfBarriers[tfMult])') == 'barrier'

    def run(arrive: bool):
        r = kernel_eval.Races()
        r.tick('store')
        r.lane = 0
        r.write(5)
        if arrive:
            r.tick('arrive')
            r.arrive(1, 0)
        r.tick('sync')
        r.barrier([32, 33], r.arrivals(1))
        r.tick('load')
        r.lane = 32
        r.read(5)
        return [x.kind for x in r.found]
    assert run(True) == []
    assert run(False) == ['RAW']


# -- AMD's LDS-DMA, waited for on vmcnt ------------------------------------- #

def _dma_body(copies=2, threads=64, dst=None, guard=None, extra=None):
    """`copies` LDS-DMA copies of one wave each, the lane past 64 elements a
    copy, then each waited for and read back reversed -- every lane reading
    another's element, within the copy's own 64.  `dst` replaces the
    destination of the first, `guard` puts it under an `if`, and `extra` is
    called between the copies and the waits -- a token it returns is waited
    for behind them."""
    ctx = Context(arch='gfx942', backend='hip', fp_type=Datatype.F32)
    b = IRBuilder(fptype=Datatype.F32, context=ctx)
    s0 = b.alloc(Datatype.F32, (64 * copies,), MemSpace.SHARED, hint='s0',
                 extern='s0', arena='blk', offset=0)
    g = b.alloc(Datatype.F32, (64 * copies,), MemSpace.GLOBAL, hint='g',
                extern='m1')
    out = b.alloc(Datatype.F32, (64 * copies,), MemSpace.GLOBAL, hint='o',
                  extern='m0')
    lane = b.thread_id('x')
    ats = [lane] + [b.op('add', INDEX, lane, 64 * k, hint='at')
                    for k in range(1, copies)]
    toks = []
    for k, at in enumerate(ats):
        to = dst(b, lane) if k == 0 and dst is not None else at
        if k == 0 and guard is not None:
            with b.if_(guard(b, lane)):
                toks.append(b.copy_async(s0, g, dst_index=(to,),
                                         src_index=(at,), threads=threads))
        else:
            toks.append(b.copy_async(s0, g, dst_index=(to,),
                                     src_index=(at,), threads=threads))
    later = extra(b, g, lane) if extra is not None else None
    for k, (tok, at) in enumerate(zip(toks, ats)):
        b.wait(tok)
        back = b.op('sub', INDEX, 64 * k + 63, lane, hint='bk')
        b.store(out, b.load(s0, back, hint='u'), at)
    if later is not None:
        b.store(out, b.wait(later), lane)
    return ctx, b.finish()


class _Commented(_Sink):
    """A sink that keeps the comments and the blocks too."""

    def Comment(self, text):
        self.lines.append(f'// {text}')

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


def _dma_emit(ctx, body, quiet=True):
    """Schedule and write `body`; `quiet`, with no diagnostic on the way --
    a copy under an `if` or a drain leaves the waits behind it counting
    nothing, and says so."""
    from tensorforge.backend import pir
    from tensorforge.backend.pir.emit import Emitter
    out, diag = pir.schedule_async(body)
    assert not (quiet and diag), diag
    sink = _Commented()
    Emitter(sink, ctx).run(out)
    return out, sink.lines


def _dma_kernel(lines):
    """The lines as a kernel the interpreter runs: the arena the shared
    buffer is a window of, the globals by their own names."""
    keep = [l for l in lines if not re.match(r'^float m\d+\[', l)]
    return ('void k() {\n'
            '  auto* blk = reinterpret_cast<float*>(totalShrMemPtr);\n'
            + ''.join(f'  {l}\n' for l in keep) + '}')


def test_an_lds_dma_wait_is_the_counter_s_and_names_its_copies():
    """The wait for an LDS-DMA copy is `vmcnt`'s, written by TF: it names
    the copies it retires, moves as a `wait` does, and is counted one
    operation at a time -- no group closes the copies."""
    ctx, body = _dma_body()
    waits = [s for s in body if s.op == Op.WAIT_COUNTER]
    assert len(waits) == 2 and not any(s.op == Op.WAIT for s in body)
    for w in waits:
        assert (w.attr('counter'), w.attr('engine')) == ('vmcnt', 'lds_dma')
        assert len(w.args) == 1 and w.movable
        assert w.effect == Effect.READ | Effect.WRITE | Effect.ASYNC
        assert engines.verify_engine(w) == []
        assert not engines.completes_on_barrier(w)
    copies = [s for s in body if s.op == Op.COPY_ASYNC]
    assert all(c.attr('threads') == 64 for c in copies)
    out, lines = _dma_emit(ctx, body)
    assert not any(s.op == Op.COMMIT_ASYNC for s in out)
    assert [s.attr('count') for s in out if s.op == Op.WAIT_COUNTER] == [1, 0]
    dma = [i for i, l in enumerate(lines) if 'global_load_lds' in l]
    waited = [i for i, l in enumerate(lines)
              if l.startswith('__builtin_amdgcn_s_waitcnt(')]
    assert len(dma) == 2 and dma[-1] < waited[0]
    assert [lines[i] for i in waited] == [
        '__builtin_amdgcn_s_waitcnt(3953);', '__builtin_amdgcn_s_waitcnt(3952);']


def test_the_count_is_the_copies_issued_behind_it_and_nothing_else():
    """vmcnt counts the loads into registers too, but the compiler moves
    those across a copy, merges two into one and drops a dead one: a count
    that includes them can come out larger than what is in flight.  So the
    wait counts the copies behind the one it waits for, which the compiler
    issues as written -- `prior_unified` still counts the loads."""
    def load(b, g, lane):
        return b.load_async(g, lane)
    ctx, body = _dma_body(copies=3, extra=load)
    out, _ = _dma_emit(ctx, body)
    first = next(s for s in out if s.op == Op.WAIT_COUNTER)
    assert first.attr('count') == 2
    assert first.attr('prior_unified') == 3


def test_a_predicated_copy_is_not_counted_behind_a_wait():
    """A wave none of whose lanes take a predicated copy branches around it,
    and issues one copy fewer than a count that includes it."""
    from tensorforge.backend import pir
    ctx = Context(arch='gfx942', backend='hip', fp_type=Datatype.F32)
    b = IRBuilder(fptype=Datatype.F32, context=ctx)
    s0 = b.alloc(Datatype.F32, (128,), MemSpace.SHARED, hint='s0',
                 extern='s0', arena='blk', offset=0)
    g = b.alloc(Datatype.F32, (128,), MemSpace.GLOBAL, hint='g', extern='m1')
    lane = b.thread_id('x')
    first = b.copy_async(s0, g, dst_index=(lane,), src_index=(lane,),
                         threads=64)
    at = b.op('add', INDEX, lane, 64, hint='at')
    some = b.op('lt', BOOL, lane, 40, hint='g')
    second = b.copy_async(s0, g, dst_index=(at,), src_index=(at,),
                          threads=64, predicate=some)
    b.wait(first)
    b.wait(second)
    out, diag = pir.schedule_async(b.finish())
    assert not diag
    assert [s.attr('count') for s in out
            if s.op == Op.WAIT_COUNTER] == [0, 0]


@pytest.mark.parametrize('change,why', [
    (dict(threads=32), 'no whole number of waves'),
    (dict(dst=lambda b, lane: b.op('mul', INDEX, lane, 2)),
     'not the lane past a base'),
    (dict(dst=lambda b, lane: 0), 'not the lane past a base'),
    (dict(guard=lambda b, lane: b.op('lt', BOOL, 40, lane)),
     'first lane of a wave'),
], ids=['half-wave', 'strided', 'one-place', 'suffix-guard'])
def test_a_copy_the_dma_cannot_write_keeps_the_body_to_loads(change, why):
    """The instruction writes lane t's four bytes t places past one base a
    wave -- the first active lane's address -- and nothing else: a body with
    one copy it cannot write moves all of them as loads and stores, and
    waits for none."""
    ctx, body = _dma_body(**change)
    _, lines = _dma_emit(ctx, body, quiet=False)
    assert any('LDS-DMA cannot write it' in l and why in l for l in lines)
    assert not any('global_load_lds' in l or 's_waitcnt' in l
                   for l in lines)


def test_a_prefix_guard_keeps_the_first_lane_and_the_copy():
    """`lane < k` leaves every wave whole or a prefix of its lanes, whose
    first is the one the base is taken from."""
    ctx, body = _dma_body(
        guard=lambda b, lane: b.op('lt', BOOL, lane, 40))
    _, lines = _dma_emit(ctx, body, quiet=False)
    assert sum('global_load_lds' in l for l in lines) == 2


def test_a_drain_in_a_body_of_dma_copies_is_the_counter_at_zero():
    """A wait for everything is written too where the copies are the DMA's:
    nothing else would wait for them in front of the reads."""
    from tensorforge.backend import pir
    from tensorforge.backend.pir.emit import Emitter
    ctx = Context(arch='gfx942', backend='hip', fp_type=Datatype.F32)
    b = IRBuilder(fptype=Datatype.F32, context=ctx)
    s0 = b.alloc(Datatype.F32, (128,), MemSpace.SHARED, hint='s0',
                 extern='s0', arena='blk', offset=0)
    g = b.alloc(Datatype.F32, (128,), MemSpace.GLOBAL, hint='g', extern='m1')
    out = b.alloc(Datatype.F32, (128,), MemSpace.GLOBAL, hint='o',
                  extern='m0')
    lane = b.thread_id('x')
    at = b.op('add', INDEX, lane, 64, hint='at')
    for where in (lane, at):
        b.copy_async(s0, g, dst_index=(where,), src_index=(where,),
                     threads=64)
    b.wait()
    for where in (lane, at):
        b.store(out, b.load(s0, where, hint='u'), where)
    body, _ = pir.schedule_async(b.finish())
    sink = _Commented()
    Emitter(sink, ctx).run(body)
    lines = sink.lines
    last = max(i for i, l in enumerate(lines) if 'global_load_lds' in l)
    read = next(i for i, l in enumerate(lines) if '= s0[' in l)
    assert [l for l in lines[last:read] if 's_waitcnt' in l] == [
        '__builtin_amdgcn_s_waitcnt(3952);']


def test_the_interpreter_finds_what_each_wait_is_for():
    """Run on the host: every lane reads another's element of each copy
    behind its wait, which is the copy's value and no race.  A wait one
    copy laxer reads the first copy in flight, and so does no wait."""
    from tensorforge.reference import kernel_eval
    ctx, body = _dma_body()
    _, lines = _dma_emit(ctx, body)
    src = _dma_kernel(lines)

    def run(text):
        races = []
        mem = kernel_eval.evaluate_wave(text, 64, seed=5, globals_only=True,
                                        races=races)
        return mem, sorted({r.kind for r in races})
    mem, races = run(src)
    seed = kernel_eval.Slot(5)
    assert races == []
    assert all(mem[('m0', t)] == seed.read('m1', 63 - t) and
               mem[('m0', 64 + t)] == seed.read('m1', 127 - t)
               for t in range(64))
    lax = src.replace('s_waitcnt(3953)', 's_waitcnt(3954)', 1)
    assert run(lax)[1] == ['FLIGHT']
    none = src.replace('__builtin_amdgcn_s_waitcnt(3953);', '', 1)
    assert run(none)[1] == ['FLIGHT']


def test_the_interpreter_writes_a_dma_where_the_hardware_does():
    """One base a wave, the address of its first lane, and lane t's bytes t
    places past it: a lane whose own address is another finds the value
    somewhere else; a block barrier lands every copy of the waves that
    reach it, and its lanes read them in order."""
    from tensorforge.reference import kernel_eval
    dma = ('__builtin_amdgcn_global_load_lds((__attribute__((address_space(1)))'
           ' uint32_t*)(&m1[threadIdx.x]), (__attribute__((address_space(3)))'
           ' uint32_t*)(&s0[{}]), 4, 0, 0);')

    def run(*stmts, lanes=64):
        src = ('void k() {\n'
               '  auto* blk = reinterpret_cast<float*>(totalShrMemPtr);\n'
               '  float * __restrict__ s0 = &blk[0];\n'
               + ''.join(f'  {s}\n' for s in stmts) + '}')
        races = []
        mem = kernel_eval.evaluate_wave(src, lanes, seed=7, globals_only=True,
                                        races=races)
        return mem, sorted({r.kind for r in races})
    seed = kernel_eval.Slot(7)
    mem, races = run(dma.format('2 * threadIdx.x'),
                     '__builtin_amdgcn_s_waitcnt(3952);',
                     'm0[threadIdx.x] = s0[threadIdx.x];')
    assert all(mem[('m0', t)] == seed.read('m1', t) for t in range(64))
    mem, races = run(dma.format('threadIdx.x'), '__syncthreads();',
                     'm0[threadIdx.x] = s0[(threadIdx.x + 64) % 128];',
                     lanes=128)
    assert races == []
    assert all(mem[('m0', t)] == seed.read('m1', (t + 64) % 128)
               for t in range(128))
    assert run(dma.format('threadIdx.x'),
               'm0[threadIdx.x] = s0[threadIdx.x];')[1] == ['FLIGHT']


def test_the_lane_past_a_shared_base_is_what_the_dma_writes():
    """`threadIdx.x` once, everything else what the multiplication shares:
    the lane plus a constant or a row's offset is, twice the lane or a
    lane-dependent offset is not."""
    from tensorforge.backend.pir.core import walk_stmts
    b = IRBuilder(fptype=Datatype.F32)
    lane = b.thread_id('x')
    row = b.thread_id('y')
    good = [lane, b.op('add', INDEX, lane, 64),
            b.op('add', INDEX, b.op('mul', INDEX, row, 64), lane)]
    bad = [b.op('mul', INDEX, lane, 2), b.op('add', INDEX, lane, lane), 0,
           row]
    defs = {v.id: s for s in walk_stmts(b.finish()) for v in s.target}
    assert all(engines.lane_linear(v, defs) for v in good)
    assert not any(engines.lane_linear(v, defs) for v in bad)


def test_amd_stages_through_the_dma_only_where_asked_and_whole_waves():
    """Off by default; asked for, on a device that has the copy, for four
    bytes an element and a multiplication of whole waves."""
    from tensorforge.common.options import registry
    assert registry()['amd_async_copies'].default is False

    def staging(arch, threads, elem=4, on=True):
        ctx = Context(arch=arch, backend='hip', fp_type=Datatype.F32,
                      options=Options(amd_async_copies=on))
        return ctx.target.async_staging(ctx.get_user_options(), threads,
                                        elem)
    assert staging('gfx942', 64) and staging('gfx90a', 128)
    assert not staging('gfx942', 64, on=False)
    assert not staging('gfx942', 32) and not staging('gfx942', 64, elem=8)
    # gfx1250's copy into the LDS has neither constraint (`test_async_lds`).
    assert not staging('gfx1150', 64)
    assert Context(arch='sm_86', backend='cuda',
                   fp_type=Datatype.F32).target.async_staging()


def test_a_counter_wait_s_effect_depends_on_what_it_names():
    """Without copies it orders any shared memory a counter tracks, stays
    where it is and retires no token; naming copies it is their `wait`:
    what they touched, movable, and the tokens consumed."""
    from tensorforge.backend.pir.core import Access, retires
    plain = engines.wait_counter('vmcnt', 0)
    assert not plain.movable and plain.args == () and not retires(plain)
    assert [(a.space, a.base) for a in plain.accesses] == [
        (MemSpace.SHARED, None)]
    b = IRBuilder(fptype=Datatype.F32, arena='shrMem')
    s0 = b.alloc(Datatype.F32, (4,), MemSpace.SHARED, hint='s0', offset=0)
    tok = b.value(TokenType(Completion.COUNTER), hint='t')
    touched = (Access(Effect.WRITE, MemSpace.SHARED, s0),)
    named = engines.wait_counter('vmcnt', 0, (tok,), touched,
                                 engine='lds_dma')
    assert named.movable and named.args == (tok,) and retires(named)
    assert named.accesses == touched
    assert engines.verify_engine(plain) == engines.verify_engine(named) == []


def _generated(name, arch='gfx942', **options):
    """The kernel of the corpus case `name` for HIP on `arch`."""
    import contextlib
    import importlib.util
    import io
    from pathlib import Path
    from tensorforge.generators.generator import Generator
    path = next((Path(__file__).resolve().parent / 'cases').rglob(f'{name}.py'))
    spec = importlib.util.spec_from_file_location('tf_dma__' + name, path)
    mod = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(mod)
    opts = {**getattr(mod, 'OPTIONS', {}), **options}
    gen = Generator(mod.descr_list(), Context(arch=arch, backend='hip',
                                              fp_type=mod.DTYPE,
                                              options=Options(**opts)))
    with contextlib.redirect_stdout(io.StringIO()):
        gen.generate()
    return gen.get_kernel()


@pytest.mark.parametrize('name', ['trip_prepared', 'k_unroll_tail'])
def test_a_preloaded_operator_goes_through_the_dma_where_asked(name):
    """At 64 lanes a multiplication is a wave, and the operator the block
    preloads into the LDS moves by DMA, every copy waited for on vmcnt in
    front of the block barrier -- a transfer short enough to be written out
    by hand as well (`trip_prepared`'s four hops), since the destination has
    to be values to be checked.  Off, or at the default lanes, where a wave
    holds several multiplications, it moves as before."""
    src = _generated(name, amd_async_copies=True, lanes_per_mult=64)
    lines = src.splitlines()
    dma = [i for i, l in enumerate(lines) if 'global_load_lds(' in l]
    wait = next(i for i, l in enumerate(lines)
                if '__builtin_amdgcn_s_waitcnt(3952);' in l)
    barrier = next(i for i in range(wait, len(lines))
                   if '__syncthreads();' in lines[i])
    assert dma and max(dma) < wait < barrier
    assert 'cuda_pipeline' not in src
    for plain in (_generated(name, lanes_per_mult=64),
                  _generated(name, amd_async_copies=True)):
        assert 'global_load_lds' not in plain and 's_waitcnt' not in plain


@pytest.mark.parametrize('arch', ['gfx90a', 'gfx942', 'gfx950'])
def test_the_preload_compiles_to_the_dma_and_its_wait(arch, tmp_path):
    """Read off the instructions: the copies are the DMA's, and the wait
    in front of the barrier drains vmcnt."""
    src = _generated('trip_prepared', arch, amd_async_copies=True,
                     lanes_per_mult=64)
    asm = _assemble('#include "tensorforge_device/hip.h"\n' + src, arch,
                    tmp_path)
    dma = [i for i, ins in enumerate(asm)
           if re.match(r'global_load_(lds_)?dword\b.*', ins)
           and (ins.startswith('global_load_lds') or ins.endswith(' lds'))]
    barrier = asm.index('s_barrier')
    assert len(dma) == 4 and max(dma) < barrier
    assert any(re.match(r's_waitcnt vmcnt\(0\)', ins)
               for ins in asm[max(dma):barrier])


def test_the_compiler_keeps_the_copies_and_the_count(tmp_path):
    """The body of three copies and a load into registers, compiled: the
    copies stay three, and the first wait is no laxer than written.  (Where
    the load goes is the compiler's: ROCm 7.2 issues it in front of the
    last copy, behind which it was written -- why a count may not include
    it.)"""
    ctx, body = _dma_body(copies=3, extra=lambda b, g, lane:
                          b.load_async(g, lane))
    _, lines = _dma_emit(ctx, body)
    keep = [l for l in lines if not re.match(r'^float m\d+\[', l)]
    src = ('#include "tensorforge_device/hip.h"\n'
           '__global__ void __launch_bounds__(64) k(float *__restrict__ m0, '
           'float *__restrict__ m1) {\n  extern __shared__ float blk[];\n'
           + ''.join(f'  {l}\n' for l in keep) + '}\n')
    asm = _assemble(src, 'gfx942', tmp_path)
    vmem = [ins for ins in asm if re.match(r'global_', ins)]
    kinds = ['dma' if ins.startswith('global_load_lds') else
             'load' if ins.startswith('global_load') else 'store'
             for ins in vmem]
    assert kinds.count('dma') == 3 and kinds.count('load') == 1
    first = next(ins for ins in asm if ins.startswith('s_waitcnt vmcnt'))
    assert int(re.search(r'vmcnt\((\d+)\)', first).group(1)) <= 2
