# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
# SPDX-FileContributor: David Schneller

"""Pseudo-IR: engines -- the hardware constructs a kernel uses, as ops.

A barrier, a fence, a counter wait or a wave priority is one instruction or
two on the device, and until here each was either a `BARRIER` whose meaning
lived in the emitter (`Emitter._sync`) or raw text with `Effect.UNKNOWN`: a
wall no pass could see into, spelled by whoever wrote it.  The tile kernels
need more of these constructs and need the passes to see them -- an arrive
apart from its wait, an object initialized ahead of the body, a fence that
orders the compiler and not the memory -- so each gets an op name of its
own and one entry here.

**The registry** (`REGISTRY`, `OpSpec`).  One entry an op: its operand and
result kinds, its memory effect, how it completes and what it retires, the
issue units and pipe the analysis passes count it under, its hazards, and
its lowering per backend.  The emitter and the passes read the entry;
nothing reads text for these ops.

**Sync objects** (`SyncObjectType`).  Every barrier arrives at and waits on
an object: the block's own (object 0, implicit), a warp's (implicit, a mask
of its lanes), a multiplication's (named: `bar.sync id, n` on NVIDIA, an
object of gfx1250's), and later mbarriers and gfx1250's LDS arrive counters.
`BARRIER` is the combined form -- every participant arrives and waits -- and
the split ops (`ARRIVE`, `WAIT_PHASE`) and the setup (`SYNC_INIT`) name the
same objects.  Which multiplications need an object of their own is what the
budget check counts (`object_demand`), and what the kernel prologue sets up.

**Completion kinds** (`Completion`).  How the completion of an operation is
observed: a commit group, an mbarrier phase, a hardware counter, or on issue.
Tokens carry one (`TokenType.completes`); the transfers that set it come with
the transfer engines.

What this increment keeps, on purpose:

* **The spellings stay the lexic's** where it had one -- `sync_block`,
  `sync_mult`, `pin_fence`, `handoff_fence`, `named_barrier_prologue` --
  since `Target`, the thread-block policy and the tests ask it too.  A
  lowering here decides which spelling an op is, and writes itself only
  what the lexic had no spelling for: the split barrier, counter waits and
  wave priorities, and the texts the producers wrote themselves -- AMD's
  scheduling groups, vISA's fence.  So every existing option emits the
  bytes it emitted before, by construction.
* **A fence is a wall wherever the raw text was one** (`Effect.UNKNOWN`).
  `FENCE(schedule)` orders the compiler and no memory, which is what the
  passes may exploit once they know the op; until then nothing moves across
  one that did not before.
* **Objects are static.**  The objects in use today are named by the
  hardware -- object 0, a warp's mask, the multiplication's id -- and exist
  for the whole kernel, so an op names its object in the attribute `object`
  rather than as an operand.  An object a body creates (an mbarrier) will be
  a `Value` of type `SyncObjectType`, its op's first operand; `object_of`
  reads either.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import (Any, Callable, Dict, Iterator, List, Mapping, NamedTuple,
                    Optional, Sequence, Tuple)

from .core import (Access, Effect, IRError, MemSpace, Op, Participants, Stmt,
                   Uniformity, Value, walk_stmts)


# --------------------------------------------------------------------------- #
# Kinds
# --------------------------------------------------------------------------- #

class Completion(Enum):
    """How the completion of an operation is observed.

    Four mechanisms, and a wait has to know which one it retires: they are
    counted differently, and a count of one kind says nothing about another.
    """

    GROUP = 'group'      # cp.async commit groups: `cp.async.wait_group N`
    BARRIER = 'barrier'  # an mbarrier phase: `mbarrier.try_wait.parity`
    COUNTER = 'counter'  # a hardware counter: vmcnt, asynccnt, tensorcnt, SBID
    SYNC = 'sync'        # complete on issue: nothing to wait for


class SyncScope(Enum):
    """Which threads an object can hold, in hardware.

    A level of the machine, not a number of threads: `MULT` is the object
    one multiplication of the block takes for its own (a named barrier),
    which is a different resource from the block's object 0 whatever the
    widths.
    """

    WARP = 'warp'          # a warp / wave / sub-group, or a mask of its lanes
    MULT = 'mult'          # one multiplication's threads: a group of waves
    BLOCK = 'block'        # the thread block / work-group
    CLUSTER = 'cluster'    # a cluster of blocks (later)
    GRID = 'grid'          # the cooperative grid


class SyncKind(Enum):
    """What an object is on the device, which decides whether it needs
    setting up (`SYNC_INIT`) and what its arrive and wait are."""

    IMPLICIT = 'implicit'      # needs no setup: object 0, a warp's barrier
    NAMED = 'named'            # a hardware id: NVIDIA `bar`, gfx1250 objects
    MBARRIER = 'mbarrier'      # 8 B in shared memory with a phase bit
    LDS_ARRIVE = 'lds_arrive'  # gfx1250's arrive counter in the LDS


class FenceKind(Enum):
    """What a fence orders.

    `SCHEDULE` orders the compiler's schedule and no memory: `sched_barrier`,
    vISA's software fence, and a `__syncwarp()` where it stands for "ptxas
    moves no load across here".  `MEMORY` orders the memory accesses of the
    threads its scope holds.  `BOTH` is the two at once.
    """

    SCHEDULE = 'schedule'
    MEMORY = 'memory'
    BOTH = 'both'


# --------------------------------------------------------------------------- #
# Sync objects
# --------------------------------------------------------------------------- #

#: The scopes and kinds that go together.  A named object is a
#: multiplication's, an implicit one the warp's, the block's or the grid's;
#: the objects a body creates may be at any scope.
_KIND_SCOPES = {
    SyncKind.IMPLICIT: (SyncScope.WARP, SyncScope.BLOCK, SyncScope.GRID),
    SyncKind.NAMED: (SyncScope.MULT,),
    SyncKind.MBARRIER: tuple(SyncScope),
    SyncKind.LDS_ARRIVE: tuple(SyncScope),
}

#: The participant sets a scope can meet.  A multiplication may meet at a
#: warp's barrier, as a mask of the warp's lanes, where it is narrower than
#: the warp; a group of multiplications meets at the block's, which is sized
#: to hold one group wherever a group has to meet (`SyncThreads`).
_SCOPE_PARTICIPANTS = {
    SyncScope.WARP: (Participants.WAVE, Participants.MULT),
    SyncScope.MULT: (Participants.MULT,),
    SyncScope.BLOCK: (Participants.BLOCK, Participants.MULTGROUP),
    SyncScope.CLUSTER: (Participants.BLOCK,),
    SyncScope.GRID: (Participants.GRID,),
}


@dataclass(frozen=True)
class SyncObjectType:
    """A barrier object: where it lives, what it is, and who arrives.

    Both the type of an object a body creates (a `Value` of this type, later)
    and the description of one the hardware names (the attribute `object`):
    the latter needs no identity beyond its description, since two barriers
    of one multiplication meet on the multiplication's one id.

    ``participants`` is what the barrier covers -- the set the emitter's
    spelling is chosen by (`Participants`) -- and ``threads`` who arrives, in
    threads, where that is fewer than the scope holds: a count is what every
    sub-block rendezvous needs, in threads, waves or sub-groups depending on
    the target, so the count and not a pre-divided number is what travels.
    ``wave`` is the geometry the arrival level is derived from (`arrival`).

    Scope and kind are not derived here but checked against the participants
    (`__post_init__`): an mbarrier's kind is not a function of who arrives,
    and a description whose parts disagree would be spelled by one part and
    budgeted by the other.
    """

    scope: SyncScope
    kind: SyncKind
    participants: Participants
    threads: Optional[int] = None
    wave: int = 1

    def __post_init__(self):
        if self.scope not in _KIND_SCOPES[self.kind]:
            raise IRError(f'a {self.kind.value} object has no '
                          f'{self.scope.value} scope')
        if (self.kind in (SyncKind.IMPLICIT, SyncKind.NAMED)
                and self.participants not in _SCOPE_PARTICIPANTS[self.scope]):
            raise IRError(f'a {self.scope.value} object cannot meet '
                          f'{self.participants.value}')
        if self.threads is not None and self.threads < 1:
            raise IRError(f'an object of {self.threads} threads')

    @property
    def arrival(self) -> Uniformity:
        """The uniformity a region needs for a barrier on this object to be
        legal: who has to arrive (`Participants.arrival`)."""
        return self.participants.arrival(
            self.threads if self.threads is not None else self.wave,
            self.wave)

    @property
    def per_mult(self) -> bool:
        """Is this one object a multiplication of the block -- an id each
        multiplication takes of the block's few (`object_demand`)?"""
        return self.kind is SyncKind.NAMED and self.scope is SyncScope.MULT

    def __repr__(self):
        count = '' if self.threads is None else f', {self.threads}'
        return (f'sync<{self.scope.value}:{self.kind.value}, '
                f'{self.participants.value}{count}/{self.wave}>')


def sync_object(participants: Participants, wave: int,
                threads: Optional[int] = None) -> SyncObjectType:
    """The object a barrier of ``participants`` arrives at and waits on.

    The object follows from what the barrier covers and the geometry, and
    from nothing about the target: a multiplication narrower than the warp
    meets at the warp's barrier, as a mask of its lanes; one at least as wide
    on an object of its own, which each target spells -- an id on NVIDIA, an
    object on gfx1250, the block's barrier where the target has nothing
    narrower and the block is sized to one multiplication (`Lexic.sync_mult`).
    Built without the target, so that a pass placing a barrier in a body it
    did not build needs none.
    """
    if participants is Participants.WAVE:
        return SyncObjectType(SyncScope.WARP, SyncKind.IMPLICIT, participants,
                              threads, wave)
    if participants is Participants.MULT:
        if threads is not None and threads < wave:
            return SyncObjectType(SyncScope.WARP, SyncKind.IMPLICIT,
                                  participants, threads, wave)
        return SyncObjectType(SyncScope.MULT, SyncKind.NAMED, participants,
                              threads, wave)
    if participants is Participants.GRID:
        return SyncObjectType(SyncScope.GRID, SyncKind.IMPLICIT, participants,
                              threads, wave)
    return SyncObjectType(SyncScope.BLOCK, SyncKind.IMPLICIT, participants,
                          threads, wave)


def object_of(s: Stmt) -> Optional[SyncObjectType]:
    """The object `s` arrives at, waits on or sets up, or None.

    The attribute for an object the hardware names, the first operand for
    one a body created -- a `Value` whose type is the object's.
    """
    obj = s.attr('object')
    if isinstance(obj, SyncObjectType):
        return obj
    if s.args and isinstance(s.args[0], Value) and isinstance(
            s.args[0].type, SyncObjectType):
        return s.args[0].type
    return None


def objects(body: Sequence[Stmt]) -> Iterator[SyncObjectType]:
    """The objects the engine ops of `body` name, at any depth, in order."""
    for s in walk_stmts(tuple(body)):
        if s.op in REGISTRY:
            obj = object_of(s)
            if obj is not None:
                yield obj


def object_demand(body: Sequence[Stmt], mults_per_block: int) -> int:
    """How many named objects a block of `mults_per_block` multiplications
    takes for the barriers of `body`: one a multiplication where any barrier
    meets a multiplication on its own, since every barrier of the
    multiplication meets on its one id -- and none for the implicit objects,
    which are no resource.

    What a block may hold is weighed against this (`check_barrier_budget`):
    one more multiplication than the target has objects for waits at an
    object that does not exist.
    """
    if any(obj.per_mult for obj in objects(body)):
        return mults_per_block
    return 0


# --------------------------------------------------------------------------- #
# The registry
# --------------------------------------------------------------------------- #

class Lowered(NamedTuple):
    """What an op is written as: its lines, and whether every lane of the
    warp has to execute them -- a full-mask `__syncwarp`, a sub-group
    barrier -- which a caller writing them under a lane-varying condition
    has to know (`Emitter._pin_load`)."""

    lines: Tuple[str, ...] = ()
    converged: bool = False


class Spelling:
    """What a lowering reads of the target: the target itself (`sync_mult`),
    its lexic and its device, and which family of backends it is.

    Built from whatever the emitter was given, which may be no target at all
    -- the IR-level tests build bodies without one -- and then answers None.
    """

    __slots__ = ('target', 'lexic', 'hw', 'family')

    def __init__(self, target: Any):
        self.target = target
        self.lexic = getattr(target, 'lexic', None)
        self.hw = getattr(target, 'hw', None)
        self.family = family(target)

    @property
    def amd(self) -> bool:
        return getattr(self.hw, 'vendor', None) == 'amd'

    @property
    def model(self) -> str:
        return str(getattr(self.hw, 'model', '') or '')


def family(target: Any) -> str:
    """The backend family a lowering is chosen by: `cuda`, `hip`, `sycl`
    (SPMD), `esimd` (the explicit vector), the backend's own name for one
    without engines, or `''` without a target."""
    if target is None:
        return ''
    if getattr(target, 'explicit_simd', False):
        return 'esimd'
    backend = str(getattr(target, 'backend', '') or '')
    if backend in ('acpp', 'oneapi', 'hipsycl', 'dpcpp'):
        return 'sycl'
    return backend


Lowering = Callable[[Stmt, Spelling], Lowered]


@dataclass(frozen=True)
class Slot:
    """One operand of an engine op: the attribute it is held in and what it
    has to be.  The ops' operands are attributes, not values -- an object
    the hardware names, a count, a kind -- so this is what `verify_engine`
    holds a statement to."""

    name: str
    kinds: Tuple[type, ...]
    required: bool = True


@dataclass(frozen=True)
class Hazard:
    """A wait the hardware does not interlock: `states` wait states between
    this op and a following `then`.  None of the ops here has one; the shape
    is for the matrix engines and the DPP blocks (an AMD VALU write read by
    an MFMA as A or B, two states), which come with them."""

    then: str
    states: int


@dataclass(frozen=True)
class OpSpec:
    """Everything the passes and the emitter know about one op.

    ``effect`` and ``movable`` are what a statement of the op declares, and
    so what every pass that asks a statement sees; ``settles`` whether
    behind it what the other participants did before their arrival is
    visible -- an arrive publishes but waits for nobody.  ``arrival``
    overrides the object's arrival level where the op reaches further than
    its object (a setup that holds the block's barrier).

    ``completes`` is how the op's own effect completes, ``retires`` which
    kind of completion it waits for.  ``units`` is what it adds to the code
    count (`Emitter._record_code`), ``pipe`` the category `analysis.pipeline`
    counts its issue under -- the ones the statements it replaces were
    counted as, so that no figure moves.  ``lower`` maps a backend family
    (`family`) to its lowering, `'*'` for every other.
    """

    name: str
    operands: Tuple[Slot, ...]
    results: Tuple[str, ...]
    effect: Effect
    movable: bool
    completes: Completion
    retires: Optional[Completion]
    settles: bool
    units: int
    pipe: str
    lower: Mapping[str, Lowering]
    hazards: Tuple[Hazard, ...] = ()
    arrival: Optional[Uniformity] = None

    def lowering(self, fam: str) -> Lowering:
        got = self.lower.get(fam, self.lower.get('*'))
        if got is None:
            raise IRError(f'{self.name} has no lowering for '
                          f'{fam or "no target"}')
        return got


# --------------------------------------------------------------------------- #
# Lowerings
# --------------------------------------------------------------------------- #

def _no_split(op: str, obj: SyncObjectType, env: Spelling) -> IRError:
    return IRError(f'{op} on {obj!r} has no spelling on '
                   f'{env.family or "no target"}: the hardware meets it in '
                   f'the combined form only (`BARRIER`)')


def _phase1(op: str, obj: SyncObjectType) -> IRError:
    return IRError(f'{op} on a {obj.kind.value} object is lowered with the '
                   f'transfers that complete on one, which are not here yet')


def _combined(obj: SyncObjectType, env: Spelling) -> Optional[str]:
    """Every participant arrives and waits: the instruction for what the
    object's barrier covers.

    Read off `participants` alone, never off the arrival level: the two
    are different questions and a wave barrier answers them differently.
    `Lexic.sync_mult` is asked only for `MULT`, and only after
    `Target.sync_mult` agreed at the same width -- so a target with no
    sub-block rendezvous is never handed a request it has to approximate.

    Falls back to the bare call without a lexic: the IR-level tests build
    bodies without a context, and a lowering that raised there would make
    them require one for an op that is not what they test.
    """
    if obj.kind in (SyncKind.MBARRIER, SyncKind.LDS_ARRIVE):
        raise _phase1('a rendezvous', obj)
    lex = env.lexic
    if lex is None:
        return '__syncthreads();'
    who = obj.participants
    if who is Participants.WAVE:
        return lex.sync_simd()
    if who is Participants.GRID:
        return lex.sync_grid()
    if who is Participants.MULT and obj.threads is not None and (
            env.hw is not None):
        sync_mult = getattr(env.target, 'sync_mult', None)
        if sync_mult is not None and not sync_mult(obj.threads):
            # Spelled anyway, it would be a rendezvous of the wrong threads:
            # a count of 48 that no warp boundary falls on, an id that two
            # interleaved multiplications share.
            raise IRError(f'a barrier of one multiplication of {obj.threads} '
                          f'threads, which this target cannot meet on its own '
                          f'(`Target.sync_mult`)')
        return lex.sync_mult(obj.threads, env.hw)
    # `multgroup` and `block` alike: the block is sized to hold one group
    # wherever a group is what has to meet, so the block barrier is the
    # group's own and there is nothing narrower to reach for.
    return lex.sync_block()


def _lower_barrier(s: Stmt, env: Spelling) -> Lowered:
    """The combined form, and where it is spelled as nothing -- a wave in
    lockstep -- the memory fence a one-lane store still owes the lanes that
    read it back (`handoff`, `Lexic.handoff_fence`)."""
    text = _combined(object_of(s), env)
    if not text and s.attr('handoff') and env.lexic is not None:
        fenced = _lower_fence(fence_stmt(FenceKind.MEMORY,
                                         scope=SyncScope.WARP), env)
        text = fenced.lines[0] if fenced.lines else None
    return Lowered(() if text is None else (text,))


def _lower_wait_phase(s: Stmt, env: Spelling) -> Lowered:
    """Wait for the object's current phase.

    On a named object the wait arrives too -- `bar.sync` counts the waiting
    threads, and gfx1250's rendezvous is a signal and a wait -- so one count
    serves both and a producer that only arrives (`ARRIVE`) is the split:
    the combined form, written by every participant, is this.
    """
    text = _combined(object_of(s), env)
    return Lowered(() if text is None else (text,))


def _lower_arrive_cuda(s: Stmt, env: Spelling) -> Lowered:
    """`bar.arrive id, n` on the multiplication's id: the arrival without
    the wait, the id and count `CudaLexic.sync_mult` meets with."""
    obj = object_of(s)
    if obj.kind in (SyncKind.MBARRIER, SyncKind.LDS_ARRIVE):
        raise _phase1(Op.ARRIVE, obj)
    if not obj.per_mult or obj.threads is None or env.lexic is None:
        raise _no_split(Op.ARRIVE, obj, env)
    row = env.lexic.thread_idx_y
    return Lowered((f'asm volatile("barrier.arrive %0, %1;" :: '
                    f'"r"({row} + 1), "r"({obj.threads}) : "memory");',))


def _lower_arrive_hip(s: Stmt, env: Spelling) -> Lowered:
    """The signal of the multiplication's object, without the wait
    (`tensorforge::arriveNamedBarrier`), where the device has objects."""
    obj = object_of(s)
    if obj.kind in (SyncKind.MBARRIER, SyncKind.LDS_ARRIVE):
        raise _phase1(Op.ARRIVE, obj)
    lex = env.lexic
    if (not obj.per_mult or lex is None or obj.threads is None
            or not getattr(env.hw, 'named_barriers', 0) or not env.amd
            or obj.threads <= getattr(env.hw, 'vec_unit_length', 0)):
        raise _no_split(Op.ARRIVE, obj, env)
    return Lowered((f'tensorforge::arriveNamedBarrier('
                    f'&{lex.NAMED_BARRIERS}[{lex.thread_idx_y}]);',))


def _lower_arrive_other(s: Stmt, env: Spelling) -> Lowered:
    obj = object_of(s)
    if obj.kind in (SyncKind.MBARRIER, SyncKind.LDS_ARRIVE):
        raise _phase1(Op.ARRIVE, obj)
    raise _no_split(Op.ARRIVE, obj, env)


def _lower_sync_init(s: Stmt, env: Spelling) -> Lowered:
    """Set up `objects` objects ahead of their first use, `count` threads
    arriving at each.

    An implicit object needs nothing, nor does an NVIDIA id, which
    `bar.sync` names by number and counts at the instruction.  gfx1250's
    objects keep their count and members as state: the kernel declares them,
    one a multiplication, and every wave sets its multiplication's count and
    joins it -- the lexic's prologue (`HipLexic.named_barrier_prologue`),
    which the base lexic answers with nothing.
    """
    obj = object_of(s)
    if obj.kind in (SyncKind.MBARRIER, SyncKind.LDS_ARRIVE):
        raise _phase1(Op.SYNC_INIT, obj)
    if not obj.per_mult or env.lexic is None or env.hw is None:
        return Lowered()
    count = s.attr('count', obj.threads)
    return Lowered(tuple(env.lexic.named_barrier_prologue(
        count, s.attr('objects', 1), env.hw)))


#: vISA's software fence: a barrier to the scheduler that emits nothing
#: (`Options.esimd_unroll_fence`).
ESIMD_SCHEDULE_MASK = 128


def _schedule_fence(s: Stmt, env: Spelling) -> Lowered:
    """A fence of the compiler's schedule.

    The scheduling barrier of the backend's compiler -- `sched_barrier` on
    AMD, mask 0 unless the fence says which instructions may cross; vISA's
    software fence under the explicit vector -- and where it has none, the
    statement its compiler moves no memory access across, `pin_fence`'s:
    `__syncwarp()` on CUDA and a sub-group barrier under SPMD SYCL, both of
    which need every lane of the warp.

    A scheduling group (`group`: mask, size, id) is AMD's pattern rather
    than a wall, and a target without one drops it: it is a hint.
    """
    lex = env.lexic
    group = s.attr('group')
    mask = s.attr('mask')
    if env.family == 'esimd':
        if group is not None:
            return Lowered()
        return Lowered((f'tensorforge::intel_esimd::fence<'
                        f'{ESIMD_SCHEDULE_MASK}>();',))
    if group is not None:
        if env.family != 'hip' or not env.amd:
            return Lowered()
        kinds, size, sync_id = group
        return Lowered((f'__builtin_amdgcn_sched_group_barrier('
                        f'{kinds}, {size}, {sync_id});',))
    if mask and env.family == 'hip' and env.amd:
        return Lowered((f'__builtin_amdgcn_sched_barrier({int(mask)});',))
    fence = lex.pin_fence() if lex is not None else None
    if fence is None:
        return Lowered()
    text, converged = fence
    return Lowered((text,), converged)


def _memory_fence(s: Stmt, env: Spelling) -> Lowered:
    """A fence of the memory accesses of the threads `scope` holds.

    At a warp's scope: the lexic's hand-off fence where the warp barrier is
    no instruction -- AMD's lanes run in step, and only the compiler has to
    be told (`Lexic.handoff_fence`) -- and the warp barrier itself where it
    is one, since `__syncwarp` and a sub-group barrier order the memory of
    the lanes they meet.  At the block's: the release-acquire fence of the
    work-group.  A memory fence is not a hint, so a scope a target has no
    spelling for is refused rather than dropped.
    """
    lex = env.lexic
    scope = s.attr('scope', SyncScope.BLOCK)
    if lex is None:
        raise IRError(f'a memory fence at {scope.value} scope without a '
                      f'target')
    if scope is SyncScope.WARP:
        text = lex.handoff_fence()
        if text:
            return Lowered((text,))
        text = lex.sync_simd()
        return Lowered(() if text is None else (text,), bool(text))
    if scope is SyncScope.BLOCK:
        text = {'cuda': '__threadfence_block();',
                'sycl': 'sycl::atomic_fence(sycl::memory_order::acq_rel, '
                        'sycl::memory_scope::work_group);'}.get(env.family)
        if env.family == 'hip':
            text = ('__builtin_amdgcn_fence(__ATOMIC_ACQ_REL, "workgroup");'
                    if env.amd else '__threadfence_block();')
        if text is not None:
            return Lowered((text,))
    raise IRError(f'a memory fence at {scope.value} scope has no spelling on '
                  f'{env.family or "no target"}')


def _lower_fence(s: Stmt, env: Spelling) -> Lowered:
    kind = s.attr('kind')
    mask = s.attr('mask')
    if env.family == 'esimd' and mask is not None and s.attr('group') is None:
        # vISA's fence bits say what it orders, whichever kind the fence
        # was asked for as (`Options.esimd_unroll_fence`).
        return Lowered((f'tensorforge::intel_esimd::fence<{int(mask)}>();',))
    if kind is FenceKind.SCHEDULE:
        return _schedule_fence(s, env)
    if kind is FenceKind.MEMORY:
        return _memory_fence(s, env)
    # Both: the memory fence, and the schedule's where it is a statement of
    # its own -- `__syncwarp()` is both at once.
    memory = _memory_fence(s, env)
    schedule = _schedule_fence(s, env)
    extra = tuple(t for t in schedule.lines if t not in memory.lines)
    return Lowered(memory.lines + extra,
                   memory.converged or schedule.converged)


#: The counters of the gfx9 `s_waitcnt` immediate (CDNA: gfx90a, gfx94x,
#: gfx950): `(shift, width)` of each field, the vector-memory count split in
#: two.  A field at its maximum waits for nothing.
_GFX9_WAITCNT = {'vmcnt': ((0, 4), (14, 2)), 'expcnt': ((4, 3),),
                 'lgkmcnt': ((8, 4),)}


def _gfx9_limit(counter: str) -> int:
    """The largest count a field of the gfx9 immediate holds."""
    return (1 << sum(width for _, width in _GFX9_WAITCNT[counter])) - 1


def _gfx9_waitcnt(counter: str, count: int) -> int:
    imm = 0
    for name, fields in _GFX9_WAITCNT.items():
        value = count if name == counter else _gfx9_limit(name)
        for shift, width in fields:
            imm |= (value & ((1 << width) - 1)) << shift
            value >>= width
    return imm


def _lower_wait_counter_cuda(s: Stmt, env: Spelling) -> Lowered:
    """`cp.async.wait_group`: at most `count` commit groups in flight, as
    the copies' own wait spells it (`Lexic.wait_async`)."""
    counter, count = s.attr('counter'), s.attr('count')
    if counter != 'group' or env.lexic is None:
        raise IRError(f'no {counter} counter to wait on under CUDA')
    return Lowered((env.lexic.wait_async(count),))


def _lower_wait_counter_hip(s: Stmt, env: Spelling) -> Lowered:
    """A counter wait written where TF wants it rather than where
    `SIInsertWaitcnts` would put it.

    The builtins and not inline assembly: an asm statement is opaque to the
    very pass that computes the counts and to the scheduler (`HipLexic.
    wait_async`).  Of gfx12's split counters clang 22 has builtins for
    gfx1250's asynchronous ones alone (`s_wait_asynccnt`, `s_wait_tensorcnt`;
    `s_wait_loadcnt` and `s_wait_dscnt` have none), so only those are
    spelled there; CDNA takes the gfx9 immediate with every other field at
    its maximum, which waits for nothing.  RDNA 3, with an immediate of its
    own, is not spelled yet.
    """
    counter, count = s.attr('counter'), s.attr('count')
    model = env.model
    if (env.amd and model.startswith('gfx9')
            and counter in ('vmcnt', 'lgkmcnt')):
        limit = _gfx9_limit(counter)
        if not 0 <= count <= limit:
            raise IRError(f'{counter}({count}) is past the counter\'s '
                          f'{limit} on {model}')
        return Lowered((f'__builtin_amdgcn_s_waitcnt('
                        f'{_gfx9_waitcnt(counter, count)});',))
    if (env.amd and model in ('gfx1250', 'gfx1251')
            and counter in ('asynccnt', 'tensorcnt')):
        return Lowered((f'__builtin_amdgcn_s_wait_{counter}({count});',))
    raise IRError(f'no {counter} counter to wait on on {model or "HIP"}')


def _lower_wait_counter_other(s: Stmt, env: Spelling) -> Lowered:
    raise IRError(f'no {s.attr("counter")} counter to wait on on '
                  f'{env.family or "no target"}: the compiler tracks the '
                  f'transfers there')


#: The waves a priority may be set for (`set_priority`): every wave where
#: None, or those of an odd index in the block.
PRIORITY_WAVES = (None, 'odd')


def _lower_set_priority(s: Stmt, env: Spelling) -> Lowered:
    """The wave's issue priority, 0 to 3 (`s_setprio`), on AMD; nothing
    elsewhere -- a priority is a hint, and a target without one schedules
    the same work.

    For the odd waves only, behind the wave's index in the block, read from
    the first lane so that the branch is uniform."""
    if not (env.family == 'hip' and env.amd):
        return Lowered()
    text = f'__builtin_amdgcn_s_setprio({s.attr("level")});'
    if s.attr('waves') == 'odd':
        text = ('if (__builtin_amdgcn_readfirstlane((threadIdx.x + '
                'blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z)) '
                f'/ warpSize) & 1) {{ {text} }}')
    return Lowered((text,))


# --------------------------------------------------------------------------- #
# The entries
# --------------------------------------------------------------------------- #

_OBJECT = Slot('object', (SyncObjectType,))
_SYNC = Effect.BARRIER
_INT = (int,)

REGISTRY: Dict[str, OpSpec] = {spec.name: spec for spec in (
    # Every participant arrives and waits: the combined form, which the
    # passes have always seen as `BARRIER` and still do.
    OpSpec(Op.BARRIER, operands=(_OBJECT, Slot('handoff', (bool,), False)),
           results=(), effect=_SYNC, movable=False,
           completes=Completion.SYNC, retires=None, settles=True,
           units=1, pipe='barrier', lower={'*': _lower_barrier}),
    # Set an object up ahead of its first use: at the kernel's head, by the
    # whole block -- gfx1250's join holds the block barrier.
    OpSpec(Op.SYNC_INIT, operands=(_OBJECT, Slot('count', _INT, False),
                                   Slot('objects', _INT, False)),
           results=(), effect=_SYNC, movable=False,
           completes=Completion.SYNC, retires=None, settles=False,
           units=1, pipe='other', lower={'*': _lower_sync_init},
           arrival=Uniformity.BLOCK),
    OpSpec(Op.ARRIVE, operands=(_OBJECT, Slot('count', _INT, False),
                                Slot('tx', _INT, False)),
           results=(), effect=_SYNC, movable=False,
           completes=Completion.BARRIER, retires=None, settles=False,
           units=1, pipe='barrier',
           lower={'cuda': _lower_arrive_cuda, 'hip': _lower_arrive_hip,
                  '*': _lower_arrive_other}),
    OpSpec(Op.WAIT_PHASE, operands=(_OBJECT,),
           results=(), effect=_SYNC, movable=False,
           completes=Completion.SYNC, retires=Completion.BARRIER,
           settles=True, units=1, pipe='barrier',
           lower={'*': _lower_wait_phase}),
    # A wall wherever the raw text it replaces was one: opaque to every
    # pass, though what it orders is known (see the module).  A fence of the
    # schedule placed behind the passes (`wall=False`) orders the device
    # compiler alone and has no effect here (`effect_of`).
    OpSpec(Op.FENCE, operands=(Slot('kind', (FenceKind,)),
                               Slot('scope', (SyncScope,), False),
                               Slot('mask', _INT, False),
                               Slot('group', (tuple,), False),
                               Slot('wall', (bool,), False)),
           results=(), effect=Effect.UNKNOWN, movable=False,
           completes=Completion.SYNC, retires=None, settles=False,
           units=1, pipe='other', lower={'*': _lower_fence}),
    OpSpec(Op.WAIT_COUNTER, operands=(Slot('counter', (str,)),
                                      Slot('count', _INT)),
           results=(), effect=Effect.READ | Effect.WRITE | Effect.ASYNC,
           movable=False, completes=Completion.SYNC,
           retires=Completion.COUNTER, settles=False, units=1, pipe='sync',
           lower={'cuda': _lower_wait_counter_cuda,
                  'hip': _lower_wait_counter_hip,
                  '*': _lower_wait_counter_other}),
    # No memory effect, and pinned: what it changes is which wave issues
    # first from where it stands, and nothing moves it.
    OpSpec(Op.SET_PRIORITY, operands=(Slot('level', _INT),
                                      Slot('waves', (str,), False)),
           results=(), effect=Effect.NONE, movable=False,
           completes=Completion.SYNC, retires=None, settles=False,
           units=1, pipe='other', lower={'*': _lower_set_priority}),
)}


def spec_of(op: str) -> Optional[OpSpec]:
    return REGISTRY.get(op)


def lower(s: Stmt, target: Any) -> Lowered:
    """`s` as written for `target` (a `Target`, or None)."""
    spec = REGISTRY[s.op]
    env = target if isinstance(target, Spelling) else Spelling(target)
    return spec.lowering(env.family)(s, env)


def verify_engine(s: Stmt) -> List[str]:
    """What is wrong with an engine statement: an operand missing or of the
    wrong kind, or a declared effect that is not the registry's -- which a
    pass would act on and the registry would not know about."""
    spec = REGISTRY[s.op]
    diag = []
    for slot in spec.operands:
        value = s.attr(slot.name)
        if slot.name == 'object' and object_of(s) is not None:
            continue
        if value is None:
            if slot.required:
                diag.append(f'{s.op}: no {slot.name}')
            continue
        if not isinstance(value, slot.kinds):
            diag.append(f'{s.op}: {slot.name} {value!r} is not a '
                        f'{" or ".join(k.__name__ for k in slot.kinds)}')
    effect = effect_of(s.op, s.attrs)
    if s.effect != effect or s.movable != spec.movable:
        diag.append(f'{s.op}: declares {s.effect!r}, '
                    f'{"movable" if s.movable else "pinned"}, where the '
                    f'registry says {effect!r}, '
                    f'{"movable" if spec.movable else "pinned"}')
    return diag


def effect_of(op: str, attrs: Tuple = ()) -> Effect:
    """The effect a statement of `op` with `attrs` declares: the registry's,
    but none for a fence that is no wall (`fence_stmt`)."""
    if op == Op.FENCE and dict(attrs).get('wall') is False:
        return Effect.NONE
    return REGISTRY[op].effect


def arrival_of(s: Stmt) -> Uniformity:
    """Who has to reach a statement that synchronizes: the object's arrival
    level for an engine op, the `scope` a call holding a rendezvous states
    (the launch cursor's `next`), and the block for anything that says
    nothing."""
    spec = REGISTRY.get(s.op)
    if spec is not None:
        if spec.arrival is not None:
            return spec.arrival
        obj = object_of(s)
        if obj is not None:
            return obj.arrival
    scope = s.attr('scope')
    return scope if isinstance(scope, Uniformity) else Uniformity.BLOCK


_M_BARRIER = int(Effect.BARRIER)


def settles(s: Stmt) -> Optional[Uniformity]:
    """The level behind `s` at which every participant's accesses ahead of
    their arrival are visible, or None where `s` settles nothing.

    A barrier's arrival level, as `place_barriers` has always read it: a
    block-wide one settles everything, a narrower one what its
    multiplication did to its own memory.  An arrive settles nothing -- the
    lanes behind it have waited for nobody -- nor does a setup.
    """
    if not int(s.effect) & _M_BARRIER:
        return None
    spec = REGISTRY.get(s.op)
    if spec is not None and not spec.settles:
        return None
    return arrival_of(s)


# --------------------------------------------------------------------------- #
# Statements
# --------------------------------------------------------------------------- #

def _stmt(op: str, attrs: Tuple = (), args: Tuple = (), target: Tuple = (),
          accesses: Tuple[Access, ...] = ()) -> Stmt:
    """A statement of `op`, with what the registry says it does."""
    spec = REGISTRY[op]
    return Stmt(op=op, target=target, args=args, pure=False,
                movable=spec.movable, effect=effect_of(op, attrs),
                accesses=accesses, attrs=attrs)


def barrier(obj: SyncObjectType, handoff: bool = False) -> Stmt:
    """Every participant of `obj` arrives and waits; `handoff`, where the
    rendezvous carries one lane's store to the others' loads of the same
    address (`Lexic.handoff_fence`)."""
    return _stmt(Op.BARRIER, (('object', obj),)
                 + ((('handoff', True),) if handoff else ()))


def sync_init(obj: SyncObjectType, objects: int = 1,
              count: Optional[int] = None) -> Stmt:
    """Set up `objects` objects like `obj`, one a multiplication of the
    block where it is a multiplication's, `count` threads arriving at each
    (the object's own threads where None)."""
    attrs = (('object', obj), ('objects', int(objects)))
    if count is not None:
        attrs += (('count', int(count)),)
    return _stmt(Op.SYNC_INIT, attrs)


def arrive(obj: SyncObjectType, count: int = 1, tx: int = 0) -> Stmt:
    """Arrive at `obj` without waiting: `count` arrivals, `tx` bytes it
    expects to complete on it (an mbarrier's transaction count)."""
    attrs = (('object', obj),)
    if count != 1:
        attrs += (('count', int(count)),)
    if tx:
        attrs += (('tx', int(tx)),)
    return _stmt(Op.ARRIVE, attrs)


def wait_phase(obj: SyncObjectType) -> Stmt:
    """Wait for the current phase of `obj` (see `_lower_wait_phase` for the
    arrival a named object's wait is)."""
    return _stmt(Op.WAIT_PHASE, (('object', obj),))


def fence_stmt(kind: FenceKind, *, scope: Optional[SyncScope] = None,
               mask: Optional[int] = None,
               group: Optional[Tuple[int, int, int]] = None,
               wall: bool = True) -> Stmt:
    """A fence of `kind`.  `scope` is whose memory a memory fence orders
    (the block's where None), `mask` the backend's own scheduling mask (AMD
    `sched_barrier`, vISA's fence bits), and `group` an AMD scheduling group
    -- mask, size, id -- rather than a wall.

    Not a `wall`: a fence of the schedule placed where the passes are done,
    for the device compiler alone -- pinned, and no effect the emitter keeps
    a value in front of (`pir.priority`'s control, which stands where its
    raw text, declared without one, stood).  A memory fence is always one."""
    if not isinstance(kind, FenceKind):
        kind = FenceKind(kind)
    if not wall and kind is not FenceKind.SCHEDULE:
        raise IRError(f'a {kind.value} fence is a wall: it orders memory')
    attrs = (('kind', kind),)
    if scope is not None:
        attrs += (('scope', scope),)
    if mask is not None:
        attrs += (('mask', int(mask)),)
    if group is not None:
        attrs += (('group', tuple(int(x) for x in group)),)
    if not wall:
        attrs += (('wall', False),)
    return _stmt(Op.FENCE, attrs)


def wait_counter(counter: str, count: int) -> Stmt:
    """Wait until at most `count` operations are outstanding on `counter`
    (`group`, `vmcnt`, `lgkmcnt`, `asynccnt`, `tensorcnt`).

    It writes what the transfers it retires wrote, and which those are is
    the counter's to know, not the statement's: so any buffer of shared
    memory, where the transfers a counter tracks land -- one into a register
    is ordered by its own def-use edge.  Which tokens a wait retires comes
    with the transfer engines that count on these.
    """
    accesses = (Access(Effect.READ | Effect.WRITE, MemSpace.SHARED, None),)
    return _stmt(Op.WAIT_COUNTER,
                 (('counter', str(counter)), ('count', int(count))),
                 accesses=accesses)


def set_priority(level: int, waves: Optional[str] = None) -> Stmt:
    """The wave's issue priority from here on, 0 (lowest) to 3, for the
    waves `waves` names (`PRIORITY_WAVES`): all where None."""
    if not 0 <= int(level) <= 3:
        raise IRError(f'a wave priority of {level}; the hardware has 0 to 3')
    if waves not in PRIORITY_WAVES:
        raise IRError(f'a wave priority for the waves {waves!r}; there are '
                      f'{", ".join(str(w) for w in PRIORITY_WAVES)}')
    attrs = (('level', int(level)),)
    if waves is not None:
        attrs += (('waves', waves),)
    return _stmt(Op.SET_PRIORITY, attrs)


__all__ = [
    'Completion', 'FenceKind', 'Hazard', 'Lowered', 'OpSpec',
    'PRIORITY_WAVES', 'REGISTRY', 'Slot', 'Spelling', 'SyncKind',
    'SyncObjectType', 'SyncScope', 'arrival_of', 'arrive', 'barrier',
    'effect_of', 'family', 'fence_stmt', 'lower', 'object_demand',
    'object_of', 'objects', 'set_priority', 'settles', 'spec_of',
    'sync_init', 'sync_object', 'verify_engine', 'wait_counter',
    'wait_phase',
]
