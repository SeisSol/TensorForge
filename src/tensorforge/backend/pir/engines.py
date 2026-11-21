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
Tokens carry one (`TokenType.completes`), the kind of the engine that issued
them.

**Transfer engines** (`TRANSFER_ENGINES`).  What moves the bytes of a
transfer: one element a lane (`scalar`) or several (`vector`, 8 or 16 bytes:
float4, a planar quad, a wide staging hop) for a `load` or `store`, which
completes on issue; `cp.async` into shared memory, counted in commit groups;
AMD's LDS-DMA (`global_load_lds`), counted one operation at a time in
`vmcnt`; the bulk copy of the tensor memory accelerator, completing on an
mbarrier; and a load into registers ahead of its use (`load.async`).  An
asynchronous transfer names its engine and the kind it completes by
(`engine`, `completes`), and the class `asyncmem` counts it in derives from
that (`counter_of`).  Every transfer says how many bytes a lane moves
(`width`), the cache policy where one is set (`cache`), and a block its
`shape`.  A copy through an engine that fills (`TransferEngine.fill`:
`cp.async`, whose source size is an operand) may end in the condition
under which it reads (`Stmt.copy_valid`): issued by every lane it is
for, it reads nothing where that fails and writes the fill, so it writes
its destination whole and is no predicated copy to the passes.  The
LDS-DMA's waits are TF's to write, on the counter
(`waited_on`): the compiler places none in front of a read of what a copy
wrote, and the instruction's one destination -- a base a wave, lane t's
bytes t places past it -- is checked where the copy is written
(`lds_dma_refusal`).  gfx1250's tensor data mover (`tdm`) copies a whole
tile, `shape` rows of a `pitch` apart, through a descriptor
(`Op.TENSOR_DESC`), issued once by the first wave of the lanes it is for
(`leader`, `tdm_refusal`) and waited for on that wave's `tensorcnt`.
NVIDIA's tensor memory accelerator (`tma`) copies a box of a tensor --
two or three dimensions, the tensor's ends filled -- through a tensor map
the launcher encodes and the kernel takes as a parameter (`TensorMap`),
issued by one thread for the lanes the mbarrier it completes on is for,
as the bulk copy is (`tma_refusal`), into shared memory in the layout of
the map's swizzle, which is a TensorForge swizzle (`swizzle_form`); and it
adds such a box back into the tensor (`tma_reduce`), one thread's, counted
in that thread's bulk groups.  Which of these stages an operand's batch is
the batch's stride's question (`staging_engine`).  NVIDIA's `ldmatrix` and
`stmatrix` move a warp's 8x8 matrices between shared memory and the
registers of its lanes as one collective (`Op.MATRIX_LOAD`,
`Op.MATRIX_STORE`): each lane names a row, holds a word of every matrix,
and the op carries where that puts each element of the tile the matrices
lie in (`matrix_transfer_bits`).

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
* **Objects named by the hardware are static.**  Object 0, a warp's mask,
  the multiplication's id exist for the whole kernel, so an op names one in
  the attribute `object` rather than as an operand.  An object a body
  creates -- the mbarrier a bulk copy completes on -- is a `Value` of type
  `SyncObjectType`, numbered in the objects' own space (negative, apart
  from the values', `IRBuilder.new_object`), defined by its `SYNC_INIT` in
  the body's prologue and named, again in `object`, by the copies and the
  waits on it: defined once ahead of everything and cloned by no pass, it
  needs no def-use edge, and an operand would sit where the copy's
  positional convention has none.  `object_of` reads every form.
"""

from __future__ import annotations

import itertools
import string
from dataclasses import dataclass
from enum import Enum
from typing import (Any, Callable, Dict, Iterator, List, Mapping, NamedTuple,
                    Optional, Sequence, Tuple)

from tensorforge.common.basic_types import RUNTIME, Addressing, Datatype

from .core import (Access, BufferType, Effect, IRError, MemSpace, Op,
                   Participants, Stmt, Uniformity, Value, XorSwizzle,
                   walk_stmts)


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

    `REGISTERS` orders the accesses of the registers it names by the threads
    against those of an asynchronous matrix engine (`engine`) that reads and
    writes them, in the compiler's schedule and in the hardware:
    `wgmma.fence`, ahead of a product whose accumulator or A fragment
    anything else touched (PTX ISA, "wgmma.fence").  `PROXY` orders the
    accesses of shared memory through the generic proxy -- an ordinary load
    or store -- against those through the async proxy -- a bulk or tensor
    copy, a matrix engine reading an operand through its descriptor
    (`fence.proxy.async.shared::cta`, PTX ISA, "membar/fence", ISA 8.0,
    sm_90).

    `THREAD_SYNC` orders an asynchronous engine's operations (`engine`)
    against the synchronization of the threads: those issued ahead of it
    before the barrier or wait behind it (`side` 'before'), those behind it
    after the barrier or wait ahead of it (`side` 'after') --
    `tcgen05.fence::before_thread_sync` and `::after_thread_sync` (PTX ISA,
    "tcgen05.fence", sm_100a).
    """

    SCHEDULE = 'schedule'
    MEMORY = 'memory'
    BOTH = 'both'
    REGISTERS = 'registers'
    PROXY = 'proxy'
    THREAD_SYNC = 'thread_sync'


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

    The attribute for an object the hardware names; for one a body created
    -- a `Value` whose type is the object's -- the value its users name in
    the same attribute, its setup's target, or an operand.
    """
    obj = s.attr('object')
    if isinstance(obj, SyncObjectType):
        return obj
    value = object_value(s)
    return None if value is None else value.type


def object_value(s: Stmt) -> Optional[Value]:
    """The object a body created that `s` names or sets up, as the `Value`
    it is -- which is what is spelled, where the object is the cursor of a
    bulk copy -- or None for an object the hardware names."""
    obj = s.attr('object')
    if isinstance(obj, Value) and isinstance(obj.type, SyncObjectType):
        return obj
    for v in s.target[:1]:
        if isinstance(v.type, SyncObjectType):
            return v
    if s.args and isinstance(s.args[0], Value) and isinstance(
            s.args[0].type, SyncObjectType):
        return s.args[0]
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
# Transfer engines
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class TransferEngine:
    """What moves the bytes of a transfer, and how its completion is seen.

    ``ops`` are the statements that go through it; ``completes`` the kind of
    completion it reports (`Completion`), which decides how a wait counts
    it (`counter_of`).
    """

    name: str
    ops: Tuple[str, ...]
    completes: Completion
    doc: str = ''
    #: The hardware counter a wait for it is written on, where TF writes
    #: that wait itself (`WAIT_COUNTER`) rather than the compiler or a
    #: group's own primitive.
    waited_on: Optional[str] = None
    #: Whether each transfer is one instruction the compiler issues as
    #: written -- an intrinsic it neither merges with another nor drops --
    #: so that a wait may count it among those issued behind the one it
    #: waits for (`asyncmem`).  An ordinary load is not: the compiler moves
    #: it across the copy, merges two into one and drops a dead one, and a
    #: count that includes it can come out larger than what is in flight.
    one_to_one: bool = False
    #: What a transfer through it writes where it reads nothing -- `zero`,
    #: or None where it reads every byte it moves: a copy that fills
    #: (`IRBuilder.copy_async`'s `valid`, `Stmt.copy_valid`) is issued by
    #: every lane it is for, and where its condition fails moves no byte of
    #: its source and this into its destination.
    fill: Optional[str] = None


TRANSFER_ENGINES: Dict[str, TransferEngine] = {e.name: e for e in (
    TransferEngine('scalar', (Op.LOAD, Op.STORE), Completion.SYNC,
                   'one element a lane'),
    TransferEngine('vector', (Op.LOAD, Op.STORE), Completion.SYNC,
                   'several elements a lane in one access, 8 or 16 bytes: '
                   'float4, a planar quad, a wide staging hop'),
    TransferEngine('cp_async', (Op.COPY_ASYNC,), Completion.GROUP,
                   'cp.async into shared memory, sm_80 and newer, counted in '
                   'commit groups; its source size, an operand, fills: zero '
                   'reads nothing and writes zeros (PTX ISA, "cp.async")',
                   fill='zero'),
    TransferEngine('lds_dma', (Op.COPY_ASYNC,), Completion.COUNTER,
                   "AMD's global_load_lds into the LDS (CDNA2/3), counted "
                   'in vmcnt one operation at a time; its waits are written '
                   'out, since the compiler places none in front of a read '
                   'of what it wrote',
                   waited_on='vmcnt', one_to_one=True),
    TransferEngine('bulk', (Op.COPY_ASYNC,), Completion.BARRIER,
                   'cp.async.bulk of a whole window by one thread, '
                   'completing on an mbarrier, sm_90 and newer'),
    TransferEngine('load', (Op.LOAD_ASYNC,), Completion.COUNTER,
                   'a load into registers issued ahead of its use: vmcnt on '
                   "AMD, NVIDIA's scoreboard"),
    TransferEngine('async_lds', (Op.COPY_ASYNC,), Completion.COUNTER,
                   "gfx1250's global_load_async_to_lds into the LDS, each "
                   'lane to its own address, counted in asynccnt one '
                   'operation at a time -- which counts nothing else; its '
                   'waits are written out, since neither the compiler nor a '
                   'block barrier waits for it',
                   waited_on='asynccnt', one_to_one=True),
    TransferEngine('buffer', (Op.LOAD, Op.STORE), Completion.SYNC,
                   "AMD's buffer instructions, 4, 8 or 16 bytes a lane "
                   'through a resource the wave holds (`resource`, '
                   '`pir.buffers`): an offset past its size reads zero and '
                   'writes nothing, which is what a predicate becomes; its '
                   "waits stay the compiler's, as a load's do"),
    TransferEngine('tma', (Op.COPY_ASYNC,), Completion.BARRIER,
                   "the tensor memory accelerator's copy of a box of a tensor "
                   '(`cp.async.bulk.tensor`) through a tensor map the '
                   'launcher encodes (`TensorMap`), by one thread, into '
                   'shared memory in the layout of the map\'s swizzle, '
                   'completing on an mbarrier with the bytes of the box, '
                   'sm_90 and newer'),
    TransferEngine('tma_reduce', (Op.COPY_ASYNC,), Completion.COUNTER,
                   "the tensor memory accelerator's reduction of a box of "
                   'shared memory into the tensor of a map '
                   '(`cp.reduce.async.bulk.tensor`, `.add`), by one thread, '
                   'each closed into a bulk group of that thread\'s own and '
                   'counted in its bulk groups, which count nothing else; '
                   'its waits are written out (`cp.async.bulk.wait_group.'
                   'read`), behind which the box may be written again, and '
                   'a barrier behind the wait shows that to the other '
                   'threads; sm_90 and newer',
                   waited_on='bulk_group', one_to_one=True),
    TransferEngine('tdm', (Op.COPY_ASYNC,), Completion.COUNTER,
                   "gfx1250's tensor data mover (`tensor_load_to_lds`): a "
                   'tile of `shape` rows a `pitch` apart into the LDS, row '
                   'after row, through a descriptor (`Op.TENSOR_DESC`), '
                   'issued once by the first wave of the lanes it is for and '
                   "counted in that wave's tensorcnt, which counts nothing "
                   'else; its waits are written out, and a barrier behind '
                   'the wait shows the tile to the other waves',
                   waited_on='tensorcnt', one_to_one=True),
    TransferEngine('ldmatrix', (Op.MATRIX_LOAD,), Completion.SYNC,
                   "NVIDIA's ldmatrix (sm_75 on): one, two or four 8x8 "
                   'matrices of 16-byte rows from shared memory into the '
                   'registers of a warp, as one collective of it -- each '
                   'lane names a row and holds a word of every matrix, '
                   'where the fragment layout puts it '
                   '(`matrix_transfer_bits`)'),
    TransferEngine('stmatrix', (Op.MATRIX_STORE,), Completion.SYNC,
                   "NVIDIA's stmatrix (sm_90 on): ldmatrix the other way "
                   "round, a warp's registers into the rows its lanes name"),
)}

#: The class `asyncmem` counts a transfer in, by the kind of completion its
#: engine reports: what closes into groups, what is counted one operation
#: at a time, and what completes on a barrier of its own.
_COUNTER_OF = {Completion.GROUP: 'copy', Completion.COUNTER: 'load',
               Completion.BARRIER: 'bulk'}


def counter_of(engine: str) -> str:
    """The class an asynchronous transfer through `engine` is counted in
    (`Stmt.counter`): derived from how the engine completes, which is the
    one fact a wait has to know about it."""
    return _COUNTER_OF[TRANSFER_ENGINES[engine].completes]


def completes_of(engine: str) -> Completion:
    return TRANSFER_ENGINES[engine].completes


def waited_on(engine: Optional[str]) -> Optional[str]:
    """The counter a wait for a transfer through `engine` is written on
    (`IRBuilder.wait`), or None where the wait is not a counter's."""
    found = TRANSFER_ENGINES.get(engine) if engine is not None else None
    return None if found is None else found.waited_on


def one_to_one(engine: Optional[str]) -> bool:
    """Whether a transfer through `engine` is one instruction a later wait
    may count (`TransferEngine.one_to_one`)."""
    found = TRANSFER_ENGINES.get(engine) if engine is not None else None
    return found is not None and found.one_to_one


def fill_of(engine: Optional[str]) -> Optional[str]:
    """What a copy through `engine` writes where it reads nothing
    (`TransferEngine.fill`), or None where no copy through it fills."""
    found = TRANSFER_ENGINES.get(engine) if engine is not None else None
    return None if found is None else found.fill


def fill_refusal(copy: Stmt) -> Optional[str]:
    """Why `copy`, a copy that fills, is none its engine issues, or None.

    Its engine fills, with what the copy says it writes (`fill`), and the
    copy ends in the condition under which it reads -- a boolean value,
    which every lane it is for evaluates for itself (`Stmt.copy_valid`)."""
    fill = copy.attr('fill')
    engine = copy.attr('engine')
    if fill_of(engine) is None:
        return (f'{engine} reads every byte it moves: no copy through it '
                f'fills')
    if fill != fill_of(engine):
        return f'a fill of {fill!r}, where {engine} writes {fill_of(engine)!r}'
    valid = copy.copy_valid
    t = getattr(valid, 'type', None)
    if not (isinstance(valid, Value) and getattr(t, 'length', 0) is None
            and getattr(getattr(t, 'base', None), 'name', None) == 'BOOL'):
        return f'it reads under {valid!r}, which is no boolean value'
    return None


def copy_engine(target: Any) -> str:
    """The engine `IRBuilder.copy_async` issues through on `target`.

    AMD's LDS-DMA where HIP spells one for the device (`Target.
    async_copy_path`), and on gfx1250 and gfx1251 their copy into the LDS
    (`Target.async_lds`); `cp.async` everywhere else and where the target
    is not known.  Whether the body's copies then go through it at all is
    the emitter's decision, once for the whole body (`Emitter.
    _decide_async`): a size or a device without the path lowers every copy
    as a load and a store.
    """
    hw = getattr(target, 'hw', None)
    if (getattr(target, 'backend', None) == 'hip'
            and getattr(hw, 'vendor', None) == 'amd'
            and target.async_copy_path()):
        return 'async_lds' if target.async_lds() else 'lds_dma'
    return 'cp_async'


#: The calls that read the lane along `x` (`IRBuilder.thread_id`).
_LANE_CALLEES = ('thread_idx_x', 'raw_thread_idx_x')


def _lane_terms(v: Any, defs: Mapping[int, Stmt], depth: int = 0
                ) -> Optional[int]:
    """How often `v` counts the lane, where it is a sum of the lane and of
    terms its multiplication agrees on; None where it is anything else."""
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return 0
    if not isinstance(v, Value) or depth > 32:
        return None
    if v.uniformity >= Uniformity.MULT:
        return 0
    d = defs.get(v.id)
    if d is None or d.predicate is not None:
        return None
    if d.op == Op.CALL and d.attr('callee') in _LANE_CALLEES:
        return 1
    if d.op == 'add' and len(d.args) == 2:
        a = _lane_terms(d.args[0], defs, depth + 1)
        b = _lane_terms(d.args[1], defs, depth + 1)
        return None if a is None or b is None else a + b
    return None


def lane_linear(index: Any, defs: Mapping[int, Stmt]) -> bool:
    """Whether `index` is the lane plus terms its multiplication agrees on:
    `threadIdx.x` once, with a coefficient of one, and everything else a
    constant or a value of `Uniformity.MULT` or wider.  `defs` maps a
    value's id to the statement that defines it.

    Where a multiplication is a whole number of waves that is one base the
    wave shares plus the lane in it -- the wave holds lanes of that one
    multiplication, and `threadIdx.x` is its lane plus a multiple of the
    wave -- which is the destination AMD's LDS-DMA writes: one base a wave
    (M0, the address of its first active lane), and lane t's bytes t places
    past it.  Any other index is cut to the first lane's and lands
    elsewhere, without a word from the compiler.
    """
    return _lane_terms(index, defs) == 1


def lane_prefix(cond: Any, defs: Mapping[int, Stmt]) -> bool:
    """Whether a guard `cond` leaves every wave either whole or a prefix of
    its lanes: a value the multiplication agrees on, or `x < bound` with `x`
    the lane past a shared base (`lane_linear`) and `bound` shared -- under
    which a wave's first active lane is its first, whose address the
    LDS-DMA takes for the wave's."""
    if isinstance(cond, Value) and cond.uniformity >= Uniformity.MULT:
        return True
    d = defs.get(cond.id) if isinstance(cond, Value) else None
    return (d is not None and d.op == 'lt' and len(d.args) == 2
            and d.predicate is None
            and _lane_terms(d.args[0], defs) == 1
            and _lane_terms(d.args[1], defs) == 0)


def lds_dma_refusal(copy: Stmt, defs: Mapping[int, Stmt],
                    guards: Sequence[Any], wave: int) -> Optional[str]:
    """Why AMD's LDS-DMA cannot write `copy`, or None where it can.

    `guards` are the conditions of the regions it stands in, a `('not',
    cond)` for an else arm and None for a region whose condition is not a
    value (a raw block, a loop whose bounds the lanes do not share).  The
    instruction takes one base a wave and writes each lane's four bytes in
    the lane's own place past it (`lane_linear`): so the multiplication has
    to be a whole number of waves (`threads`, which the issuing transfer
    states), the copy one element of four bytes a lane, its destination the
    lane past a base the wave shares, and every guard around it one that
    keeps the wave's first lane in where any lane is (`lane_prefix`).
    """
    threads = copy.attr('threads')
    if not threads:
        return 'the copy does not say how many lanes its multiplication has'
    if threads % wave:
        return (f'a multiplication of {threads} lanes is no whole number of '
                f'waves of {wave}')
    if copy.attr('elems', 1) != 1 or copy.attr('width') != 4:
        return (f'a lane moves {copy.attr("width")} bytes, where the copy '
                f'moves one element of four')
    dst = copy.copy_dst_index
    if len(dst) != 1 or not lane_linear(dst[0], defs):
        return 'the destination is not the lane past a base the wave shares'
    conds = tuple(guards)
    if copy.predicate is not None:
        conds += (copy.predicate,)
    for cond in conds:
        if cond is None or isinstance(cond, tuple) and not (
                isinstance(cond[1], Value)
                and cond[1].uniformity >= Uniformity.MULT):
            return 'a guard around it is not one the lanes share'
        if not isinstance(cond, tuple) and not lane_prefix(cond, defs):
            return 'a guard around it can leave out the first lane of a wave'
    return None


#: What a tensor descriptor holds a tile's extents in, and its pitch, in
#: elements: the rows and the columns in sixteen bits each (`tile_dim1`,
#: `tile_dim0`), the pitch in forty-eight (`tensor_dim0_stride`) -- see
#: `hip.h`, `tensorDescriptorWords`.
TDM_TILE_LIMIT = (1 << 16) - 1
TDM_PITCH_LIMIT = (1 << 48) - 1

#: The element sizes a tensor descriptor's data size spells, in bytes.
TDM_ELEMENT_BYTES = (1, 2, 4, 8)

#: The lanes a tensor copy is for (`team`): a multiplication's, whose
#: guards it has to share, or the block's.
TDM_TEAMS = {'mult': Uniformity.MULT, 'block': Uniformity.BLOCK}


def tdm_refusal(copy: Stmt, defs: Mapping[int, Stmt], level: Uniformity,
                wave: int) -> Optional[str]:
    """Why gfx1250's tensor data mover cannot issue `copy`, or None where it
    can.

    One instruction a wave, and every operand of it the wave's: so the copy
    is issued once for the `threads` lanes it is for -- a multiplication's or
    the block's (`team`) -- by the first wave of them, under the predicate `x
    < wave` with `x` the lane past a base they share (`lane_linear`).  They
    are a whole number of waves, so that their first `wave` lanes are that
    wave and no wave holds lanes of two tiles, and `level`, the least
    uniform of the guards and loop bounds around the copy, is one they all
    share, so that the first wave reaches the copy wherever any of them
    does.  The tile's extents and pitch fit the descriptor's fields
    (`TDM_TILE_LIMIT`, `TDM_PITCH_LIMIT`), and its element is a size the
    descriptor spells (`TDM_ELEMENT_BYTES`).
    """
    threads = copy.attr('threads')
    if not threads:
        return 'the copy does not say how many lanes it is for'
    if threads % wave:
        return f'{threads} lanes are no whole number of waves of {wave}'
    team = TDM_TEAMS.get(copy.attr('team'))
    if team is None:
        return f'the copy is for the lanes of {copy.attr("team")!r}'
    shape, pitch = copy.attr('shape'), copy.attr('pitch')
    if not (isinstance(shape, tuple) and len(shape) == 2
            and all(isinstance(n, int) and 1 <= n <= TDM_TILE_LIMIT
                    for n in shape)):
        return (f'a tile of {shape!r} is not two extents of 1 to '
                f'{TDM_TILE_LIMIT}')
    if not (isinstance(pitch, int) and shape[1] <= pitch <= TDM_PITCH_LIMIT):
        return (f'a pitch of {pitch!r} is shorter than a row of {shape[1]} '
                f'or longer than {TDM_PITCH_LIMIT}')
    width, elems = copy.attr('width'), copy.attr('elems')
    if not (width and elems == shape[0] * shape[1]
            and width % elems == 0 and width // elems in TDM_ELEMENT_BYTES):
        return (f'{width!r} bytes in {elems!r} elements is no tile of '
                f'elements of {TDM_ELEMENT_BYTES} bytes')
    lead = copy.predicate
    d = defs.get(lead.id) if isinstance(lead, Value) else None
    if not (copy.attr('leader') and d is not None and d.op == 'lt'
            and len(d.args) == 2 and d.predicate is None
            and _lane_terms(d.args[0], defs) == 1 and d.args[1] == wave):
        return f'it is not issued by the first wave of its lanes (x < {wave})'
    if level < team:
        return 'a guard around it is not one its lanes share'
    return None


@dataclass(frozen=True)
class TensorDescType:
    """The type of a tensor descriptor (`Op.TENSOR_DESC`): what gfx1250's
    tensor data mover copies a tile through -- the tile's LDS and global
    addresses, its rows and columns, the pitch of its rows -- held once for
    the wave, in scalar registers.  Spelled by the target
    (`Lexic.tensor_descriptor_type`); no part of what a lane's registers
    hold, as a buffer resource is not.

    ``elem`` is the element of the tile, whose size the descriptor says."""

    elem: Any = None

    def __repr__(self):
        return 'tdesc' if self.elem is None else f'tdesc<{self.elem}>'


# --------------------------------------------------------------------------- #
# Tensor maps: boxes through the tensor memory accelerator (`tma`)
# --------------------------------------------------------------------------- #

#: The ranks of a box TensorForge writes the copy for: PTX has `.1d` to
#: `.5d` (PTX ISA, "cp.async.bulk.tensor"), the tiles here are two and
#: three dimensions.
TMA_RANKS = (2, 3)
#: A box's extent along a dimension, at most; a tensor's extent, at most;
#: and a stride of the tensor in bytes, below (CUDA's tensor map:
#: `cuTensorMapEncodeTiled`'s boxDim, globalDim and globalStrides).
TMA_BOX_LIMIT = 256
TMA_DIM_LIMIT = 1 << 32
TMA_STRIDE_LIMIT = 1 << 40
#: Bytes a box starts on in shared memory, and the tensor's base address,
#: its strides and a box's first element along its rows in global memory:
#: the tensor map's rules, the last one the hardware's (an illegal
#: instruction on the RTX 5070, `~/tf-probe/engines3b/probe`).
TMA_SHARED_ALIGN = 128
TMA_GLOBAL_ALIGN = 16
#: What a box reads past the tensor's ends: zeros, or a NaN that the
#: tensor core's FMA reads as zero (`CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE`,
#: `..._NAN_REQUEST_ZERO_FMA`).
TMA_FILLS = ('zero', 'nan')
#: The elements a reduction of a box adds (PTX ISA, "cp.reduce.async.bulk.
#: tensor": `.add` of the types of the map), of those a map spells.
TMA_REDUCE_TYPES = (Datatype.F32, Datatype.I32, Datatype.U32)
#: The elements a map spells, by the suffix of their name in the driver's
#: `CUtensorMapDataType` (`CU_TENSOR_MAP_DATA_TYPE_FLOAT32`, ...), which
#: `tensorforge::TensorMapElement` names for the launcher (`cuda_tma.h`).
TMA_TYPES = {Datatype.F32: 'FLOAT32', Datatype.F64: 'FLOAT64',
             Datatype.I32: 'INT32', Datatype.U32: 'UINT32',
             Datatype.I64: 'INT64'}


@dataclass(frozen=True)
class TensorMap:
    """A tensor map: what the tensor memory accelerator copies a box of a
    tensor in global memory through (PTX ISA, "Tensor-map" and
    "cp.async.bulk.tensor"; CUDA's `cuTensorMapEncodeTiled`).

    128 bytes the driver encodes on the host and the kernel takes as a
    parameter, in param space (`__grid_constant__`), by its address: so the
    launcher encodes it from the pointer, the strides and the extents it is
    handed (`kernel_params.TensorMapParam`, `tensorforge::tensorMapTiled`),
    and a copy names it (`IRBuilder.copy_tma`).  This is its static half,
    what the generator knows of it, under the parameter's `name`.

    Outermost dimension first, as TensorForge's shapes are; PTX and the
    driver count from the innermost one, and the lowering and the launcher
    turn the order around.  `dims` are the tensor's extents, None where only
    the launch knows one -- the batch's elements.  `strides` are the
    elements from one index of a dimension to the next, for every dimension
    but the innermost, whose elements are adjacent; None is dense, the
    extents' products.  `box` is what a copy moves.  `swizzle`, 0 or 32, 64
    or 128 bytes, is how the box lies in shared memory: NVIDIA's swizzle of
    that span, which is TensorForge's `XorSwizzle` of `swizzle_form`, and
    whose rows are the span apart (`pitch`).  `fill` is what a box reads past
    the tensor's ends (`TMA_FILLS`).
    """

    name: str
    elem: Any
    dims: Tuple[Optional[int], ...]
    box: Tuple[int, ...]
    strides: Optional[Tuple[Optional[int], ...]] = None
    swizzle: int = 0
    fill: str = 'zero'

    def __post_init__(self):
        why = self._refusal()
        if why is not None:
            raise IRError(f'a tensor map {self.name!r}: {why}')

    def _refusal(self) -> Optional[str]:
        if not (isinstance(self.name, str) and self.name.isidentifier()):
            return 'its name is no identifier'
        if self.elem not in TMA_TYPES:
            return f'of {self.elem}; a map spells {", ".join(map(str, TMA_TYPES))}'
        dims, box = tuple(self.dims), tuple(self.box)
        if len(dims) != len(box) or len(box) not in TMA_RANKS:
            return (f'{len(dims)} extents and a box of {len(box)}; the '
                    f'copies here are of {TMA_RANKS} dimensions')
        for n in dims:
            if n is not None and not (isinstance(n, int)
                                      and 1 <= n <= TMA_DIM_LIMIT):
                return f'an extent of {n!r}'
        for n in box:
            if not (isinstance(n, int) and 1 <= n <= TMA_BOX_LIMIT):
                return f'a box extent of {n!r}, not 1 to {TMA_BOX_LIMIT}'
        size = self.elem.size()
        inner = box[-1] * size
        if inner % 16:
            return (f'a box row of {inner} bytes, which is no multiple of '
                    f'16')
        if self.swizzle not in SWIZZLE_BYTES:
            return f'a swizzle of {self.swizzle!r}; there are {SWIZZLE_BYTES}'
        if self.swizzle and inner > self.swizzle:
            return (f'a box row of {inner} bytes, wider than its swizzle of '
                    f'{self.swizzle}')
        if self.fill not in TMA_FILLS:
            return f'a fill of {self.fill!r}; there are {TMA_FILLS}'
        if self.strides is None:
            if any(n is None for n in dims[1:]):
                return 'dense, with an inner extent only the launch knows'
            return None
        if len(self.strides) != len(dims) - 1:
            return (f'{len(self.strides)} strides for {len(dims)} '
                    f'dimensions; the innermost has none')
        for n in self.strides:
            if n is None:
                continue
            if not isinstance(n, int) or n < 1:
                return f'a stride of {n!r}'
            if (n * size) % TMA_GLOBAL_ALIGN or n * size >= TMA_STRIDE_LIMIT:
                return (f'a stride of {n * size} bytes, no multiple of '
                        f'{TMA_GLOBAL_ALIGN} below {TMA_STRIDE_LIMIT}')
        return None

    @property
    def rank(self) -> int:
        return len(self.box)

    def element_strides(self) -> Tuple[Optional[int], ...]:
        """Elements from one index of each dimension to the next, outermost
        first, the innermost's 1; None where the launch binds one."""
        if self.strides is not None:
            return tuple(self.strides) + (1,)
        out, step = [], 1
        for n in reversed(self.dims):
            out.append(step)
            step = None if step is None or n is None else step * n
        return tuple(reversed(out))

    def box_bytes(self) -> int:
        """The bytes of a box: what its copy announces on the mbarrier, the
        elements past the tensor's ends included -- a box is filled whole."""
        volume = 1
        for n in self.box:
            volume *= n
        return volume * self.elem.size()

    def rows(self) -> int:
        """The rows of a box: every dimension but the innermost."""
        rows = 1
        for n in self.box[:-1]:
            rows *= n
        return rows

    def pitch(self) -> int:
        """Bytes from a row of a box to the next in shared memory: the
        swizzle's span where there is one, a row narrower than it padded
        to it (GB200's and the RTX 5070's reading); the row itself where
        not."""
        return self.swizzle or self.box[-1] * self.elem.size()

    def shared_bytes(self) -> int:
        """The bytes a box covers in shared memory, its padding included."""
        return self.rows() * self.pitch()

    def shared_index(self, *at: int) -> int:
        """The element of the destination that element `at` of a box --
        outermost first -- goes to, before the swizzle permutes it."""
        row = 0
        for n, i in zip(self.box[:-1], at[:-1]):
            row = row * n + i
        return row * (self.pitch() // self.elem.size()) + at[-1]

    def __repr__(self):
        dims = 'x'.join('?' if n is None else str(n) for n in self.dims)
        box = 'x'.join(map(str, self.box))
        swz = f', sw{self.swizzle}' if self.swizzle else ''
        return f'tmap<{self.name}: {dims}x{self.elem}, box {box}{swz}>'


def tma_refusal(copy: Stmt) -> Optional[str]:
    """Why the tensor memory accelerator cannot copy `copy`, or None where
    it can: a box into shared memory (`tma`), or a box of shared memory
    added into the tensor (`tma_reduce`), the same rules for the side in
    shared memory -- its source then -- and the coordinates on the side of
    the tensor, its destination.  A reduction is issued by the thread its
    predicate picks, as every thread would issue one otherwise, and adds
    an element of `TMA_REDUCE_TYPES`.

    The map (`map`) a `TensorMap`, a coordinate for each of its dimensions
    (`copy_src_index`) -- the innermost one on `TMA_GLOBAL_ALIGN` bytes,
    which the RTX 5070 refuses as an illegal instruction where it is not; a
    value is the caller's to keep there -- and the mbarrier the copy
    completes on (`object`);
    the destination a buffer of shared memory of the map's element, laid
    out as the map's swizzle lays the box out (`swizzle_form`, None without
    one) and long enough for the box from where it starts; that start on
    `TMA_SHARED_ALIGN` bytes where it is a number -- a value is the caller's
    to keep there, as the buffer's placement is (`IRBuilder.copy_tma`) --
    and the bytes the copy announces the box's.
    """
    tmap = copy.attr('map')
    if not isinstance(tmap, TensorMap):
        return 'it names no tensor map'
    reduce = copy.attr('engine') == 'tma_reduce'
    coords = copy.copy_dst_index if reduce else copy.copy_src_index
    if len(coords) != tmap.rank:
        return (f'{len(coords)} coordinates into a map of {tmap.rank} '
                f'dimensions')
    inner = coords[-1]
    if (isinstance(inner, int) and not isinstance(inner, bool)
            and (inner * tmap.elem.size()) % TMA_GLOBAL_ALIGN):
        return (f'the box starts {inner * tmap.elem.size()} bytes into a row, '
                f'off {TMA_GLOBAL_ALIGN}: an illegal instruction')
    if reduce:
        if copy.predicate is None:
            return 'every thread would add the box: no thread issues it alone'
        if tmap.elem not in TMA_REDUCE_TYPES:
            return (f'a reduction of {tmap.elem}; it adds '
                    f'{", ".join(map(str, TMA_REDUCE_TYPES))}')
    elif copy.attr('object') is None:
        return 'it names no mbarrier to complete on'
    side = copy.copy_src if reduce else copy.copy_dst
    t = getattr(side, 'type', None)
    if not (isinstance(t, BufferType) and t.space == MemSpace.SHARED):
        return (f'the box is {side!r} in shared memory, which is no buffer '
                f'there')
    elem = getattr(t.elem, 'base', t.elem)
    if elem.size() != tmap.elem.size():
        return f'a box of {tmap.elem} in a buffer of {elem}'
    form = swizzle_form(tmap.swizzle, tmap.elem)
    if t.swizzle != form:
        return (f'the box lies as {form!r} in shared memory, and the buffer '
                f'is laid out as {t.swizzle!r}')
    index = copy.copy_src_index if reduce else copy.copy_dst_index
    start = index[0] if len(index) == 1 else None
    if len(index) != 1:
        return 'the box in shared memory is not named by one index'
    need = tmap.shared_bytes() // elem.size()
    if isinstance(start, int) and not isinstance(start, bool):
        if (start * elem.size()) % TMA_SHARED_ALIGN:
            return (f'the box starts {start * elem.size()} bytes into the '
                    f'buffer, off {TMA_SHARED_ALIGN}')
        if start + need > t.volume:
            return (f'the box covers {need} elements from {start}, past the '
                    f'{t.volume} of the buffer')
    elif need > t.volume:
        return f'the box covers {need} elements, past the {t.volume} of the buffer'
    if copy.attr('width') != tmap.box_bytes():
        return (f'it announces {copy.attr("width")!r} bytes for a box of '
                f'{tmap.box_bytes()}')
    return None


def tensor_map_refusal(addressing: Any, *, stride_bytes: Optional[int],
                       alignment: int, row_bytes: int) -> Optional[str]:
    """Why the batch of an operand addressed as `addressing` (an
    `Addressing` or an `ElementAddressing`) cannot be read through a tensor
    map, or None where it can.

    A map states one stride from an element's data to the next, which the
    launcher has to know when it encodes the map: the batch's own operand
    (`NONE`, no stride at all), one at a stride from the last (`STRIDED`),
    or one an index list names at a stride from the base (`INDEXED`, the
    box's coordinate along the elements its entry).  An array of pointers
    states none (`PTR_BASED`, `INDEXED_POINTER`): an element's data is
    wherever its pointer says, and only the device reads the pointers.  A
    stride bound at run time is not taken either, as the map could refuse
    it once the generator has decided.  Every element's data starts on 16
    bytes, which only the operand's promise says (`alignment`, `Tensor.
    alignment`: the base and its offsets included), and so does each of
    its rows (`row_bytes`) and the stride (`stride_bytes`, the stride in
    bytes; None for the batch's operand).
    """
    kind = getattr(addressing, 'kind', addressing)
    if kind in (Addressing.PTR_BASED, Addressing.INDEXED_POINTER):
        return ("an element's data is reached through a pointer of its "
                'own, at no stride from the last the launcher knows')
    if kind not in (Addressing.NONE, Addressing.STRIDED, Addressing.INDEXED):
        return f'an operand addressed as {kind} is no tensor in memory'
    if getattr(addressing, 'stride', None) is RUNTIME:
        return ('a stride bound at run time, which the map could refuse '
                'after the kernel is generated')
    if alignment % TMA_GLOBAL_ALIGN:
        return (f'the operand promises {alignment} bytes of alignment; a '
                f'map needs {TMA_GLOBAL_ALIGN}')
    if row_bytes % TMA_GLOBAL_ALIGN:
        return f'rows of {row_bytes} bytes, off {TMA_GLOBAL_ALIGN}'
    if kind is not Addressing.NONE and (
            stride_bytes is None or stride_bytes % TMA_GLOBAL_ALIGN
            or stride_bytes >= TMA_STRIDE_LIMIT):
        return (f'elements {stride_bytes} bytes apart, no multiple of '
                f'{TMA_GLOBAL_ALIGN} below {TMA_STRIDE_LIMIT}')
    return None


def staging_engine(target: Any, addressing: Any, *,
                   stride_bytes: Optional[int], alignment: int,
                   row_bytes: int, window_bytes: int,
                   contiguous: bool) -> Tuple[str, Tuple[str, ...]]:
    """The engine that copies an element's window of an operand into shared
    memory on `target`, and why each better one was not taken: a box through
    a tensor map (`tma`) where the batch has one stride the launcher knows
    (`tensor_map_refusal`); else the bulk copy of the window, one for each
    element (`bulk`), where the window is one run of a multiple of 16 bytes
    on 16 -- which an array of pointers does not keep from it; else the copy
    of every lane's few bytes (`copy_engine`: `cp.async`, or AMD's)."""
    why = []
    if not (target is not None and target.tensor_maps()):
        why.append('the target has no tensor maps')
    else:
        refusal = tensor_map_refusal(addressing, stride_bytes=stride_bytes,
                                     alignment=alignment, row_bytes=row_bytes)
        if refusal is None:
            return 'tma', ()
        why.append(refusal)
    if not (target is not None and target.bulk_copy_path()):
        why.append('the target has no bulk copy')
    elif not contiguous:
        why.append('the window is no one run of bytes')
    elif window_bytes % 16 or alignment % 16:
        why.append(f'a window of {window_bytes} bytes on {alignment}, off '
                   f'16')
    else:
        return 'bulk', tuple(why)
    return copy_engine(target), tuple(why)


# --------------------------------------------------------------------------- #
# Matrix transfers: a warp's 8x8 matrices between shared memory and registers
# --------------------------------------------------------------------------- #

#: The matrices one transfer moves (`.x1`, `.x2`, `.x4`), each eight rows of
#: 16 bytes -- PTX's `.m8n8` of 16-bit elements (`MATRIX_TRANSFER_SHAPE`),
#: four of 32 bits a row (PTX ISA, "Warp-level matrix load instruction:
#: ldmatrix" and "Warp-level matrix store instruction: stmatrix").
MATRIX_TRANSFER_COUNTS = (1, 2, 4)
MATRIX_TRANSFER_SHAPE = (8, 8)
MATRIX_ROWS = 8
MATRIX_ROW_BYTES = 16

#: The first architecture each is on: ldmatrix's `.m8n8` from sm_75 (PTX ISA
#: 6.5), stmatrix's from sm_90 (PTX ISA 7.8).
MATRIX_TRANSFER_SM = {'ldmatrix': 75, 'stmatrix': 90}

#: The op of each, by its engine.
MATRIX_TRANSFER_OPS = {'ldmatrix': Op.MATRIX_LOAD, 'stmatrix': Op.MATRIX_STORE}


def matrix_transfer_bits(origins: Sequence[Tuple[int, int]]) -> Any:
    """The fragment layout of a matrix transfer of 32-bit elements -- a
    `BitLayout` over the rows and the columns of the tile its matrices lie
    in -- or None where that is no layout of bits.  `origins` is the element
    of the tile each matrix starts at, (row, column): matrix `i`, the one
    whose rows lanes `8 i` to `8 i + 7` name.

    What the instruction states (PTX ISA, "ldmatrix", "stmatrix"): lane `t`
    holds in register `i` the 32-bit word `t % 4` of row `t / 4` of matrix
    `i`.  Over 32-bit elements a word is one, so a matrix's row is the lane's
    bits 2 to 4 and its element the lane's bits 0 and 1, and the matrix is
    the register: a bit of the register's number for each bit of the tile
    past a matrix's eight rows or four columns.  That takes a placement in
    which each bit of a matrix's number moves it by one such bit -- `origins`
    the sums of the steps of their numbers' bits -- and the steps along each
    axis fill its bits from the matrix's own up.  `((0, 0), (8, 0), (0, 4),
    (8, 4))` is the A fragment of `mma.sync`'s m16n8k8 over TF32
    (`nvidia.a_fragment_bits`).
    """
    from tensorforge.backend.instructions.compute.bitlayout import (
        Bit, BitLayout, Place)
    origins = tuple(tuple(o) for o in origins)
    count = len(origins)
    if count not in MATRIX_TRANSFER_COUNTS or origins[0] != (0, 0):
        return None
    steps = [origins[1 << b] for b in range((count - 1).bit_length())]
    for i, at in enumerate(origins):
        want = tuple(sum(step[axis] for b, step in enumerate(steps)
                         if i >> b & 1) for axis in (0, 1))
        if at != want:
            return None
    words = MATRIX_ROW_BYTES // 4
    down = sorted((r, b) for b, (r, c) in enumerate(steps) if c == 0 and r)
    across = sorted((c, b) for b, (r, c) in enumerate(steps) if r == 0 and c)
    if len(down) + len(across) != len(steps):
        return None
    if any(r != MATRIX_ROWS << k for k, (r, _) in enumerate(down)) or any(
            c != words << k for k, (c, _) in enumerate(across)):
        return None
    rows = (tuple(Bit(Place.LANE, 4 << k) for k in range(3))
            + tuple(Bit(Place.SLOT, 1 << b) for _, b in down))
    cols = (tuple(Bit(Place.LANE, 1 << k) for k in range(2))
            + tuple(Bit(Place.SLOT, 1 << b) for _, b in across))
    return BitLayout((rows, cols))


def matrix_registers(s: Stmt) -> Tuple[Any, ...]:
    """The registers of a matrix transfer: what a load defines, what a store
    reads (its operands behind the base and the index)."""
    return tuple(s.target) if s.op == Op.MATRIX_LOAD else tuple(s.args[2:])


def matrix_transfer_refusal(s: Stmt) -> Optional[str]:
    """Why `s` is no matrix transfer this lowering writes, or None.

    The engine's own op -- ldmatrix loads, stmatrix stores -- of one, two or
    four matrices of `MATRIX_TRANSFER_SHAPE`, a register of each, and each
    register an element of 32 bits: a word of a row, which the transposing
    form (`.trans`) and the 16-bit elements would split, and neither is
    built.  Every lane of the warp takes part (`.aligned`), so there is no
    predicate; the operands are the buffer and the lane's row in it, which
    the op declares as one read or one write of shared memory; a layout it
    carries is one of bits (`matrix_transfer_bits`).
    """
    engine = s.attr('engine')
    if MATRIX_TRANSFER_OPS.get(engine) != s.op:
        return (f'{s.op} through {engine!r}: ldmatrix loads and stmatrix '
                f'stores')
    count = s.attr('count')
    if count not in MATRIX_TRANSFER_COUNTS:
        return (f'{count!r} matrices; the instruction moves one of '
                f'{MATRIX_TRANSFER_COUNTS}')
    if s.attr('shape') != MATRIX_TRANSFER_SHAPE:
        return (f'matrices of {s.attr("shape")!r}; the instruction moves '
                f'{MATRIX_TRANSFER_SHAPE} of 16-bit elements')
    registers = matrix_registers(s)
    want = 2 if s.op == Op.MATRIX_LOAD else 2 + count
    if len(s.args) != want or len(registers) != count:
        return f'{len(registers)} registers for {count} matrices'
    for v in registers:
        t = getattr(v, 'type', None)
        size = getattr(getattr(t, 'base', None), 'size', None)
        if not (isinstance(v, Value) and getattr(t, 'length', 0) is None
                and callable(size) and size() == 4):
            return (f'{v!r} is no register of a 32-bit element, a word of a '
                    f'row of a matrix')
    if s.predicate is not None:
        return ('every lane of the warp takes part (`.aligned`): a predicate '
                'would leave lanes out')
    kind = Effect.READ if s.op == Op.MATRIX_LOAD else Effect.WRITE
    if len(s.accesses) != 1 or s.accesses[0].kind != kind or \
            s.accesses[0].space != MemSpace.SHARED:
        return (f'{s.op} declares {s.accesses!r}, where it is one '
                f'{"read" if kind == Effect.READ else "write"} of shared '
                f'memory')
    layout = s.attr('layout')
    if layout is not None:
        from tensorforge.backend.instructions.compute.bitlayout import (
            BitLayout)
        if not isinstance(layout, BitLayout):
            return f'a fragment layout of {layout!r}, which is no BitLayout'
    return None


# --------------------------------------------------------------------------- #
# Relayout: registers brought from one distribution over the lanes to another
# --------------------------------------------------------------------------- #

#: The backend families AMD's code generation runs under: HIP, and SYCL
#: compiled for AMD (AdaptiveCpp's gfx targets), which writes the same
#: calls and builtins.
AMD_FAMILIES = ('hip', 'sycl')

#: The forms a relayout is written in (`RelayoutStrategy.form`): a value of a
#: value (`value`), returned by what it is written as; registers written
#: through references (`in_place`), declared ahead of it -- the transposes
#: of AMD's quads and sixteens, whose outputs are the first `written`
#: operands; and registers renamed (`relabel`), each result a register the
#: source holds already, which writes no instruction.
RELAYOUT_FORMS = ('value', 'in_place', 'relabel')


def _xlane(params: Mapping) -> Optional[str]:
    return 'xlane'


def _xlane_wide(params: Mapping) -> Optional[str]:
    return 'xlane.wide'


def _broadcast_pipe(params: Mapping) -> Optional[str]:
    """A broadcast over a block wider than a row of sixteen lanes crosses
    the rows (`emit._crosslane_category`, `analysis.pipeline`)."""
    return 'xlane.wide' if int(params['threads']) > 16 else 'xlane'


def _no_issue(params: Mapping) -> Optional[str]:
    return None


@dataclass(frozen=True)
class RelayoutStrategy:
    """One way of bringing registers from the distribution over the lanes
    they have to another (`Op.RELAYOUT`): the instruction it is written as,
    what it costs, and which lanes it reads.

    `families` are the backends it is written for (`family`); `form` how
    (`RELAYOUT_FORMS`); `callee` the function it is written as, a template
    over its parameters (`params`) -- whose lanes `pir.demand` reads off
    it, as it reads a call's (`demand.lane_map`); `units` and `pipe` what
    it adds to the code count and the issue mix -- the figures the calls
    it replaced were counted as.  `lossy` where the result holds fewer
    distinct elements than the source, a sub-block repeated.

    Which strategy closes a gap between two layouts is the target's
    question (`amd.relayout.find_relayout`, which searches AMD's rows; a
    relabel where the two agree but for the registers, `relabel_order`)."""

    name: str
    families: Tuple[str, ...]
    form: str
    callee: Optional[str] = None
    pipe: Callable[[Mapping], Optional[str]] = _xlane
    units: int = 1
    lossy: bool = False
    doc: str = ''

    def spelled(self, params: Mapping) -> str:
        """The function a relayout of these parameters is written as."""
        return self.callee.format(**dict(params))

    @property
    def fields(self) -> Tuple[str, ...]:
        """The parameters its callee names: what tells two relayouts of it
        apart, as the text of the call they were did."""
        if self.callee is None:
            return ()
        return tuple(name for _, name, _, _ in
                     string.Formatter().parse(self.callee) if name)


RELAYOUT_STRATEGIES: Dict[str, RelayoutStrategy] = {r.name: r for r in (
    RelayoutStrategy(
        'broadcast', AMD_FAMILIES, 'value',
        'tensorforge::broadcast<{threads}, {step}, {lane}>',
        pipe=_broadcast_pipe, lossy=True,
        doc="AMD's DPP broadcast (`hip.h`): sub-block `lane` of `step` "
            'lanes, repeated over each block of `threads`'),
    RelayoutStrategy(
        'movdpp16', AMD_FAMILIES, 'value', 'tensorforge::movdpp16<{row}>',
        lossy=True,
        doc="AMD's DPP row share: lane `row` of each row of sixteen"),
    RelayoutStrategy(
        'movdpp16_kept', AMD_FAMILIES, 'value',
        'tensorforge::movdpp16Kept<{row}>', lossy=True,
        doc='the row share, kept where it is issued (`hip.h`)'),
    RelayoutStrategy(
        'transpose4x4', AMD_FAMILIES, 'in_place', 'tensorforge::transpose4x4b32',
        doc='the 4x4 transpose of (register, lane % 4) into four registers '
            'declared ahead of it'),
    RelayoutStrategy(
        'transpose4x4_fused', AMD_FAMILIES, 'in_place',
        'tensorforge::transpose4x4b32Fused',
        doc='the same exchange, its shuffle and selection fused '
            '(`Options.amd_fused_transpose`)'),
    RelayoutStrategy(
        'transpose16x16', AMD_FAMILIES, 'in_place',
        'tensorforge::transpose16x16b32',
        doc='the 16x16 transpose of (register, lane % 16), in place on its '
            'sixteen operands'),
    RelayoutStrategy(
        'relabel', ('cuda', 'hip', 'sycl', 'esimd', ''), 'relabel',
        pipe=_no_issue, units=0,
        doc='the registers renamed: each result a register the source holds '
            'already, where the two layouts agree but for the order of the '
            'registers (`relabel_order`) -- a D fragment read as the next '
            "product's A, its K relabeled; written as copies the compiler "
            'folds, no instruction'),
    RelayoutStrategy(
        'shfl', ('cuda',), 'value', 'tensorforge::gather<32>',
        pipe=_xlane_wide,
        doc="NVIDIA's warp shuffle (`SHFL.IDX`, `cuda.h`'s gather): each "
            'lane the register of the lane its second operand names'),
)}


def relayout_strategy(s: Stmt) -> RelayoutStrategy:
    """The strategy a `RELAYOUT` names."""
    found = RELAYOUT_STRATEGIES.get(s.attr('strategy'))
    if found is None:
        raise IRError(f'a relayout by {s.attr("strategy")!r}; there are '
                      f'{", ".join(sorted(RELAYOUT_STRATEGIES))}')
    return found


def relayout_callee(s: Stmt) -> Optional[str]:
    """The function a `RELAYOUT` is written as, or None for one written as
    no call (a relabel): what `pir.demand`'s lane maps and the issue mix
    read, as they read a call's."""
    strategy = relayout_strategy(s)
    if strategy.callee is None:
        return None
    return strategy.spelled(dict(s.attr('params') or ()))


def relabel_order(source: Any, target: Any,
                  relabel: Optional[Sequence[Callable[[int], int]]] = None
                  ) -> Optional[Tuple[int, ...]]:
    """Which register of the source each register of the target is, where
    the two layouts (`BitLayout`s over the same indices) put every element
    in the same lane -- the registers renamed, no instruction -- or None
    where some element would change lanes.

    `relabel`, one function an index, renames the indices of the target:
    element `(i, j)` of the source is element `(relabel[0](i), relabel[1]
    (j))` of the target.  So a D fragment is the next product's A where B's
    K rows are relabeled to match (`d-to-a-is-a-column-relabel`): mma.sync's
    D of m16n8 and its A of m16n8k8 over TF32 differ by a rotation of a
    column's three bits, which a relabel of K undoes."""
    if len(source.axes) != len(target.axes):
        return None
    sizes = []
    for bits_s, bits_t in zip(source.axes, target.axes):
        if len(bits_s) != len(bits_t):
            return None
        sizes.append(1 << len(bits_s))
    order: Dict[int, int] = {}
    for index in itertools.product(*(range(n) for n in sizes)):
        renamed = tuple(index) if relabel is None else tuple(
            f(i) for f, i in zip(relabel, index))
        if any(not 0 <= i < n for i, n in zip(renamed, sizes)):
            return None
        at_s = source.locate(*index)
        at_t = target.locate(*renamed)
        if at_s.lane != at_t.lane or at_s.element != at_t.element:
            return None
        if order.setdefault(at_t.slot, at_s.slot) != at_s.slot:
            return None
    slots = sorted(order)
    if slots != list(range(len(slots))) or sorted(order.values()) != slots:
        return None
    return tuple(order[j] for j in slots)


def relabeling(source: Any, target: Any
               ) -> Optional[Tuple[Tuple[Callable[[int], int], ...],
                                   Tuple[int, ...]]]:
    """The relabel of the indices under which `target` holds every element
    in the lane `source` holds it -- a permutation of each index's bits --
    and the order of the registers then (`relabel_order`), or None where no
    relabel does it.

    A bit of an index that lies on a lane, or inside a vector register, has
    to land on the same lane bit or vector bit in the target, of the same
    index; a bit that picks the register may land on any register bit of
    it.  So a gap between two layouts that is the registers alone, once the
    indices are renamed, is closed with no instruction: what a consumer
    reading the same values under the renamed indices -- B's K rows
    relabeled to match -- takes as it stands.  mma.sync's D of m16n8 is
    the A of the next m16n8k8 over TF32 that way, its column's low bit the
    register's and its two higher the lane's, where A's two low bits are
    the lane's and its high bit the register's (`d-to-a-is-a-column-
    relabel`)."""
    from tensorforge.backend.instructions.compute.bitlayout import Place
    if len(source.axes) != len(target.axes):
        return None
    maps = []
    for bits_s, bits_t in zip(source.axes, target.axes):
        if len(bits_s) != len(bits_t):
            return None
        mapping = [None] * len(bits_s)
        free = [k for k, b in enumerate(bits_t) if b.place is Place.SLOT]
        for k, b in enumerate(bits_s):
            if b.place is Place.SLOT:
                if not free:
                    return None
                mapping[k] = free.pop(0)
                continue
            hit = [m for m, c in enumerate(bits_t)
                   if c.place is b.place and c.weight == b.weight]
            if len(hit) != 1:
                return None
            mapping[k] = hit[0]
        if sorted(mapping) != list(range(len(bits_s))):
            return None

        def move(i, mapping=tuple(mapping)):
            return sum(1 << m for k, m in enumerate(mapping) if i >> k & 1)
        maps.append(move)
    relabel = tuple(maps)
    order = relabel_order(source, target, relabel)
    return None if order is None else (relabel, order)


def relayout_refusal(s: Stmt) -> Optional[str]:
    """Why `s` is no relayout its strategy writes, or None.

    Its strategy one of `RELAYOUT_STRATEGIES`, in its form: a value of a
    value -- and for a shuffle the lane it reads -- one result; registers
    written through references, as many as `written` says, declared as a
    read and a write each, and no result; registers renamed, a result a
    register in the order `order` names, which is a permutation, and the
    layouts a relabel in that order where it states them (`layouts`:
    source, target and the relabel of the indices, `relabel_order`).  A
    relayout reads other lanes' registers, so it is under no predicate."""
    strategy = RELAYOUT_STRATEGIES.get(s.attr('strategy'))
    if strategy is None:
        return f'no strategy {s.attr("strategy")!r}'
    form = strategy.form
    if form == 'value':
        want = 2 if strategy.name == 'shfl' else 1
        if len(s.target) != 1 or len(s.args) != want:
            return (f'{strategy.name} takes {want} operand(s) and gives one '
                    f'value, not {len(s.args)} and {len(s.target)}')
    elif form == 'in_place':
        written = s.attr('written')
        if s.target or not isinstance(written, int) or not (
                0 < written <= len(s.args)):
            return (f'{strategy.name} writes the first `written` of its '
                    f'operands, and gives no value')
        regs = list(s.args[:written])
        if [a.base for a in s.accesses] != regs or any(
                a.space != MemSpace.REGISTER
                or a.kind != Effect.READ | Effect.WRITE for a in s.accesses):
            return (f'{strategy.name} without a read and a write of each '
                    f'register it writes')
    else:
        order = s.attr('order')
        if (not isinstance(order, tuple) or len(order) != len(s.target)
                or sorted(order) != list(range(len(s.args)))):
            return (f'a relabel of {len(s.args)} registers into '
                    f'{len(s.target)} in the order {order!r}: a permutation')
        layouts = s.attr('layouts')
        if layouts is not None:
            source, target, relabel = (tuple(layouts) + (None,))[:3]
            if relabel_order(source, target, relabel) != order:
                return (f'the layouts are no relabel in the order {order!r}: '
                        f'an element would change lanes')
    if s.predicate is not None:
        return ('a relayout reads the registers of other lanes, which a '
                'predicate would leave out')
    return None


def element_of(type_: Any, base: Any) -> Any:
    """The element a transfer moves a lane of: the value's, or where the
    value has no type of its own -- a literal stored -- the buffer's."""
    elem = getattr(type_, 'base', None)
    if elem is not None:
        return elem
    if isinstance(base, Value) and hasattr(base.type, 'elem'):
        return base.type.elem
    return getattr(base, 'datatype', None)


def width_of(type_: Any, base: Any) -> Optional[int]:
    """Bytes a lane moves at once: the element times the value's slots, or
    None where the element is not known (a raw base)."""
    elem = element_of(type_, base)
    try:
        size = elem.size()
    except (AttributeError, TypeError):
        return None
    return size * (getattr(type_, 'length', None) or 1)


def engine_of(s: Stmt) -> Optional[str]:
    """The engine a transfer goes through: the one an asynchronous one
    names, and for a `load` or `store` the scalar or the vector one its
    width makes it -- a wide access is one instruction of 8 or 16 bytes,
    however it is spelled."""
    engine = s.attr('engine')
    if engine is not None:
        return engine
    if s.op not in (Op.LOAD, Op.STORE):
        return None
    value = s.target[0] if s.op == Op.LOAD else (
        s.args[1] if len(s.args) > 1 else None)
    length = getattr(getattr(value, 'type', None), 'length', None)
    return 'scalar' if length is None else 'vector'


def copy_cache(engine: str, width: Optional[int]) -> Optional[str]:
    """The cache policy a copy's width gives it: sixteen bytes of `cp.async`
    lower to `cp.async.cg` and go past L1, four and eight to `cp.async.ca`
    and fill it on the way (`CudaLexic.copy_async`) -- which is why a
    staging transfer takes sixteen where it can (`_bypass_covering`)."""
    if engine != 'cp_async' or width is None:
        return None
    return 'cg' if width == 16 else 'ca'


#: What `cache_policy` answers where the builder did not know the target:
#: the access's request is resolved where it is lowered.
UNRESOLVED = object()


def cache_policy(request: Any, target: Any, datatype: Any,
                 length: int = 1) -> Any:
    """The cache hint an access that asks for `request` (`hints.cache_hint`:
    `'cg'`, `'cs'`, or False) gets on `target`, for an access of `length` x
    `datatype`: the request where the target spells it for that type
    (`Target.nontemporal`), None where it does not -- and `UNRESOLVED`
    without a target, for the emitter to ask the one it lowers for."""
    if not request:
        return None
    if target is None:
        return UNRESOLVED
    if datatype is None or not target.nontemporal(datatype, length):
        return None
    return request


#: The attributes of an access that are global memory's alone, which a pass
#: that points an access at shared memory drops (`trips`, `shards`).
GLOBAL_POLICY = ('nontemporal', 'cache')

#: What a transfer's statement says that its other facts already decide --
#: the width its value's type, the policy its request and type -- which is
#: therefore no part of what makes two accesses one (`passes._cse_key`).
DERIVED = ('width', 'cache')


def kept_facts(s: Stmt) -> Tuple:
    """The attributes an access keeps where a pass points it elsewhere and
    drops the rest -- at a register, which takes no hint and no alignment:
    what a lane moves, which is still its value's.

    Not the buffer engine's (`engine`, `resource`): a register is no buffer.
    Nor do the passes that point a read at shared memory (`trips`, `shards`)
    or at the next element's binding (`wrap`) have anything of it to keep or
    to drop, since the engine is taken behind all of them (`pir.buffers`) --
    a read moved to the next element and still naming this element's
    resource would read this element."""
    return tuple((k, v) for k, v in s.attrs if k == 'width')


@dataclass(frozen=True)
class ResourceType:
    """The type of a buffer resource (`Op.RESOURCE`): what an AMD buffer
    instruction addresses global memory through -- a base, a size in bytes,
    and what an access past that size does -- held once for the wave, in
    four scalar registers.  Spelled by the target
    (`Lexic.buffer_resource_type`); no part of what a lane's registers hold,
    so the pressure counts it as nothing, as it does a token.

    ``elem`` is the element of the buffer it is made over, which an access
    through it counts its index in and reads its words as -- an access
    through an operand's symbol names no binding whose type would say."""

    elem: Any = None

    def __repr__(self):
        return 'resource' if self.elem is None else f'resource<{self.elem}>'


# --------------------------------------------------------------------------- #
# The registry
# --------------------------------------------------------------------------- #

class Lowered(NamedTuple):
    """What an op is written as: its lines, and whether every lane of the
    warp has to execute them -- a full-mask `__syncwarp`, a sub-group
    barrier -- which a caller writing them under a lane-varying condition
    has to know (`Emitter._pin_load`).

    `result` is the expression an op with a result is: the emitter declares
    the result with it as its initializer, spelling the declaration as its
    lexic and its model of a value do (`Emitter.declare`) -- a scalar, a
    vector type of the target's, a `simd` under the explicit vector."""

    lines: Tuple[str, ...] = ()
    converged: bool = False
    result: Optional[str] = None


class Spelling:
    """What a lowering reads of the target: the target itself (`sync_mult`),
    its lexic and its device, which family of backends it is -- and how the
    emitter names a value, for an op whose object or storage is one.

    Built from whatever the emitter was given, which may be no target at all
    -- the IR-level tests build bodies without one -- and then answers None.
    """

    __slots__ = ('target', 'lexic', 'hw', 'family', 'name')

    def __init__(self, target: Any, name: Optional[Callable] = None):
        self.target = target
        self.lexic = getattr(target, 'lexic', None)
        self.hw = getattr(target, 'hw', None)
        self.family = family(target)
        self.name = name if name is not None else str

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
    this op and a following `then` -- a matrix engine by its name, which is
    what the DPP blocks' VALU writes owe an MFMA reading them as A or B, two
    states (`amd.dpp_block.HAZARD`)."""

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
    counted as, so that no figure moves -- or a function of the statement
    where the category is one of its shape (`pipe_of`).  ``lower`` maps a
    backend family (`family`) to its lowering, `'*'` for every other;
    ``check`` is what `verify_engine` asks of a statement beyond its slots.
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
    pipe: Any
    lower: Mapping[str, Lowering]
    hazards: Tuple[Hazard, ...] = ()
    arrival: Optional[Uniformity] = None
    check: Optional[Callable[[Stmt], List[str]]] = None

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
    return IRError(f'{op} on a {obj.kind.value} object has no spelling of '
                   f'its own yet: the transfer that completes on it arrives '
                   f'there (`tensorforge::bulk_copy`)')


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

    On an mbarrier it is the wait for what completes on it, the bulk copies
    whose tokens it names: the cursor's (`tensorforge::bulk_wait`), which
    holds the phase the wait is for.
    """
    obj = object_of(s)
    if obj.kind is SyncKind.MBARRIER:
        return _wait_mbarrier(s, env)
    text = _combined(obj, env)
    return Lowered(() if text is None else (text,))


def _wait_mbarrier(s: Stmt, env: Spelling) -> Lowered:
    _spelled_mbarrier(s, env, object_value(s))
    return Lowered((f'tensorforge::bulk_wait({env.name(object_value(s))});',))


def _spelled_mbarrier(s: Stmt, env: Spelling, value) -> None:
    """Refuse an mbarrier the target cannot spell, or one that is no value:
    a description has no storage, and no cursor to name."""
    if value is None:
        raise IRError(f'{s.op} on an mbarrier that no body set up: it is '
                      f'spelled by the cursor its setup declares')
    if env.family != 'cuda':
        raise IRError(f'an mbarrier has no spelling on '
                      f'{env.family or "no target"}')


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
    if obj.kind is SyncKind.MBARRIER:
        return _init_mbarrier(s, env)
    if obj.kind is SyncKind.LDS_ARRIVE:
        raise _phase1(Op.SYNC_INIT, obj)
    if not obj.per_mult or env.lexic is None or env.hw is None:
        return Lowered()
    count = s.attr('count', obj.threads)
    return Lowered(tuple(env.lexic.named_barrier_prologue(
        count, s.attr('objects', 1), env.hw)))


def _init_mbarrier(s: Stmt, env: Spelling) -> Lowered:
    """An mbarrier in the eight bytes of shared memory its setup names,
    armed by the thread `leader` names for one arrival a phase: the cursor
    the bulk copies through it and their waits name (`cuda.h`,
    `tensorforge::bulk_arm`), which holds the phase bit as well."""
    _spelled_mbarrier(s, env, s.target[0] if s.target else None)
    return Lowered((f'tensorforge::BulkCursor {env.name(s.target[0])} = '
                    f'tensorforge::bulk_arm({env.name(s.args[0])}, '
                    f'{s.attr("leader")});',))


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


def _register_fence(s: Stmt, env: Spelling) -> Lowered:
    """`wgmma.fence` (PTX ISA, "wgmma.fence", sm_90a), behind the operand
    fence over the registers it orders: the threads' accesses of them ahead
    of it, in the compiler's schedule and in the hardware, before any
    product the warpgroup issues behind it reads or writes them."""
    engine = s.attr('engine')
    if engine != 'wgmma' or env.family != 'cuda':
        raise IRError(f'a fence of the registers of {engine!r} on '
                      f'{env.family or "no target"}: wgmma\'s, on CUDA')
    _wgmma_available(env)
    registers = [v for v in s.args if isinstance(v, Value)]
    lines = (operand_fence(registers, env.name),) if registers else ()
    return Lowered(lines + ('tensorforge::wgmmaFence();',))


def _proxy_fence(s: Stmt, env: Spelling) -> Lowered:
    """`fence.proxy.async.shared::cta` (PTX ISA, "membar/fence", ISA 8.0,
    sm_90): what the threads of the block wrote to shared memory through the
    generic proxy, visible to what reads it through the async proxy behind
    it -- a matrix engine through a descriptor, a bulk or tensor copy --
    and the other way round.  CUDA from sm_90 on."""
    if env.family != 'cuda':
        raise IRError(f'a fence of the async proxy on '
                      f'{env.family or "no target"}: CUDA has one')
    level = env.hw.sm_level() if env.hw is not None else None
    if level is not None and level < 90:
        raise IRError(f'a fence of the async proxy on {env.model}: sm_90 '
                      f'and newer have one')
    return Lowered(('tensorforge::fenceProxyAsync();',))


def _check_fence(s: Stmt) -> List[str]:
    """A fence of registers names the engine whose accesses it orders and
    declares a read and a write of each register it names; no other fence
    names an operand."""
    kind = s.attr('kind')
    registers = [v for v in s.args if isinstance(v, Value)]
    if kind is FenceKind.THREAD_SYNC:
        if s.attr('engine') != 'tcgen05' or s.attr('side') not in (
                'before', 'after') or s.args:
            return [f'{s.op}: a fence of the threads\' synchronization names '
                    f'tcgen05, a side, before or after, and no operands']
        return []
    if kind is not FenceKind.REGISTERS:
        return [f'{s.op}: a {getattr(kind, "value", kind)} fence names no '
                f'operands'] if s.args else []
    diag = []
    if s.attr('engine') not in GROUP_ENGINES:
        diag.append(f'{s.op}: a fence of the registers of '
                    f'{s.attr("engine")!r}')
    if len(registers) != len(s.args) or [a.base for a in s.accesses] != \
            registers or any(a.space != MemSpace.REGISTER
                             or a.kind != Effect.READ | Effect.WRITE
                             for a in s.accesses):
        diag.append(f'{s.op}: a fence of registers without a read and a '
                    f'write of each')
    return diag


def _lower_fence(s: Stmt, env: Spelling) -> Lowered:
    kind = s.attr('kind')
    mask = s.attr('mask')
    if kind is FenceKind.REGISTERS:
        return _register_fence(s, env)
    if kind is FenceKind.PROXY:
        return _proxy_fence(s, env)
    if kind is FenceKind.THREAD_SYNC:
        return _thread_sync_fence(s, env)
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
    if counter == 'wgmma':
        return _wait_wgmma(s, env)
    if counter == 'bulk_group':
        # The thread's bulk groups, of which at most `count` may still be
        # reading their boxes (`tma_reduce`; PTX ISA,
        # "cp.async.bulk.wait_group": `.read`).
        return Lowered((f'tensorforge::bulkWaitRead<{count}>();',))
    if counter != 'group' or env.lexic is None:
        raise IRError(f'no {counter} counter to wait on under CUDA')
    return Lowered((env.lexic.wait_async(count),))


#: A fence of the compiler alone, which emits no instruction: no access to
#: memory moves across it.  It stands ahead of gfx1250's copy into the LDS
#: (`async_lds`, `HipLexic.copy_async`) and behind each wait on `asynccnt`.
#: clang 22 and 23 declare the copy to write only the LDS address its own
#: lane names (`memory(argmem: ...)`), and the wait to touch no memory a
#: program names; but every lane's copy writes its own place, which the
#: other lanes read.  Compiled for gfx1250 and gfx1251, a lane's read of its
#: neighbor's place, written ahead of the copy, was issued behind it -- when
#: the neighbor's copy may already have written there (2026-11-19).  Nor
#: does that model keep a read of what the copies wrote behind their wait:
#: no probe has shown one moved ahead of it, but nothing there forbids it.
COMPILER_FENCE = '__atomic_signal_fence(__ATOMIC_SEQ_CST);'


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
            and counter == 'asynccnt'):
        # The fence behind it keeps a read of what the copies wrote behind
        # the wait (`COMPILER_FENCE`).
        return Lowered((f'__builtin_amdgcn_s_wait_asynccnt({count});',
                        COMPILER_FENCE))
    if (env.amd and model in ('gfx1250', 'gfx1251')
            and counter in ('asynccnt', 'tensorcnt')):
        lines = (f'__builtin_amdgcn_s_wait_{counter}({count});',)
        if counter == 'tensorcnt':
            lines += (TENSOR_FENCE,)
        return Lowered(lines)
    raise IRError(f'no {counter} counter to wait on on {model or "HIP"}')


#: What follows a wait on `tensorcnt`: a fence of the compiler alone, which
#: emits no instruction.  clang 22 declares the tensor load to touch no
#: memory a program names -- its operands are the descriptor's words, no
#: pointer -- and the wait to touch none either, so a read of the tile
#: behind both is free to move ahead of them, or to be answered by an
#: earlier read of the same address: compiled for gfx1250, a read of an
#: LDS address behind a tensor load into it and its wait came out as the
#: value read before the load (2026-11-18).  The load's own fence, ahead of
#: it, is `tensorforge::tensorLoad`'s (`hip.h`).  The same fence as the
#: asynchronous copies' (`COMPILER_FENCE`), for the same reason.
TENSOR_FENCE = COMPILER_FENCE


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
# Matrix transfers
# --------------------------------------------------------------------------- #

def _matrix_transfer_available(s: Stmt, env: Spelling) -> None:
    """Refuse a CUDA target older than the architecture the engine is on
    (`MATRIX_TRANSFER_SM`); every other backend has no lowering at all
    (`_no_matrix_transfer`).  Without a target it is written as for one."""
    engine = s.attr('engine')
    sm_level = getattr(env.hw, 'sm_level', None)
    level = sm_level() if callable(sm_level) else None
    floor = MATRIX_TRANSFER_SM.get(engine, 0)
    if level is not None and level < floor:
        raise IRError(f'{engine} on {env.model}: sm_{floor} and newer have '
                      f'it')


def _lower_matrix_transfer(s: Stmt, env: Spelling) -> Lowered:
    """A matrix transfer as `cuda.h` writes it: `tensorforge::ldmatrixX{n}(
    r0, ..., base, index)`, the registers declared ahead of it and written
    through references, and `tensorforge::stmatrixX{n}(base, index, r0,
    ...)` -- the lane's row `base + index`, 16 bytes on 16.  The helpers
    move floats."""
    _matrix_transfer_available(s, env)
    why = matrix_transfer_refusal(s)
    if why is not None:
        raise IRError(why)
    registers = matrix_registers(s)
    for v in registers:
        if v.type.base != Datatype.F32:
            raise IRError(f'{s.attr("engine")} of {v.type.base}: the '
                          f'helpers move floats (`cuda.h`)')
    names = [env.name(v) for v in registers]
    where = f'{env.name(s.args[0])}, {env.name(s.args[1])}'
    count = s.attr('count')
    if s.op == Op.MATRIX_LOAD:
        return Lowered(tuple(f'{v.type.base.ctype()} {n}{{}};'
                             for v, n in zip(registers, names))
                       + (f'tensorforge::ldmatrixX{count}('
                          f'{", ".join(names)}, {where});',))
    return Lowered((f'tensorforge::stmatrixX{count}({where}, '
                    f'{", ".join(names)});',))


def _no_matrix_transfer(s: Stmt, env: Spelling) -> Lowered:
    raise IRError(f'{s.attr("engine")} has no spelling on '
                  f'{env.family or "no target"}: it is NVIDIA\'s, on CUDA')


def _check_matrix_transfer(s: Stmt) -> List[str]:
    why = matrix_transfer_refusal(s)
    return [] if why is None else [f'{s.op}: {why}']


def _matrix_transfer_pipe(s: Stmt) -> Optional[str]:
    """The issue category of a matrix transfer: a store of shared memory for
    stmatrix.  ldmatrix counts as nothing -- as the call it was written as
    before it was an op, which the emitter took for one that lays no code
    down (`emit._NO_CODE`) -- so that no figure of a kernel that loads its
    fragments by it moves; it is one issue of the shared memory pipe."""
    return 'shared.store' if s.op == Op.MATRIX_STORE else None


def _lower_relayout(s: Stmt, env: Spelling) -> Lowered:
    """A relayout as its strategy writes it (`RELAYOUT_STRATEGIES`): a value
    returned by the function it is called as, the result's initializer --
    the text the call it replaced wrote; registers written through
    references, the function called on its operands; registers renamed, a
    copy of each source register under the result's name, which the
    compiler folds."""
    strategy = relayout_strategy(s)
    if env.family and env.family not in strategy.families:
        raise IRError(f'{strategy.name} has no spelling on {env.family}')
    why = relayout_refusal(s)
    if why is not None:
        raise IRError(why)
    names = [env.name(a) for a in s.args]
    if strategy.form == 'relabel':
        lines = []
        for v, at in zip(s.target, s.attr('order')):
            if getattr(v.type, 'length', None) is not None:
                raise IRError(f'a relabel of {v.type}: of scalar registers')
            lines.append(f'{v.type.base.ctype()} {env.name(v)} = '
                         f'{names[at]};')
        return Lowered(tuple(lines))
    callee = strategy.spelled(dict(s.attr('params') or ()))
    if strategy.form == 'in_place':
        return Lowered((f'{callee}({", ".join(names)});',))
    return Lowered(result=f'{callee}({", ".join(names)})')


def _check_relayout(s: Stmt) -> List[str]:
    why = relayout_refusal(s)
    return [] if why is None else [f'{s.op}: {why}']


def _relayout_pipe(s: Stmt) -> Optional[str]:
    """The issue category of a relayout, its strategy's: the one the call
    it replaced was counted under."""
    return relayout_strategy(s).pipe(dict(s.attr('params') or ()))


def _relayout_units(s: Stmt) -> int:
    """A relayout's code, its strategy's: one call, or no instruction for a
    relabel."""
    return relayout_strategy(s).units


# --------------------------------------------------------------------------- #
# Matrix engines
# --------------------------------------------------------------------------- #

class Source(Enum):
    """Where a matrix engine reads an operand from, or keeps D.

    `REGISTER`: the fragment's registers, as every synchronous engine takes
    them.  `SHARED`: shared memory, through one descriptor
    (`MatrixDescType`, `Op.MATRIX_DESC`) -- wgmma's B always, its A where A
    is not in registers -- whose read the matrix op declares, since the
    descriptor reads nothing.  `TMEM`: NVIDIA's tensor memory (tcgen05), by
    its address.
    """

    REGISTER = 'register'
    SHARED = 'shared'
    TMEM = 'tmem'


#: The swizzles a matrix descriptor states, in bytes: the width of the rows
#: of an atom of eight rows, whose 16-byte chunks are permuted by the row's
#: place in the atom (`swizzled`) -- and 0 for none.
SWIZZLE_BYTES = (0, 32, 64, 128)

#: The formats of a matrix descriptor: wgmma's (sm_90a) and tcgen05's
#: (sm_100a), which differ in a fixed field and in how a swizzle is
#: numbered (`matrix_descriptor_bits`).
DESC_FORMATS = ('wgmma', 'tcgen05')

#: The swizzle field of each format, by the swizzle in bytes: wgmma's two
#: bits from bit 62, tcgen05's three from bit 61 (PTX ISA, wgmma's "Matrix
#: Descriptor Format" and tcgen05's "Shared Memory Descriptor").
_SWIZZLE_FIELD = {'wgmma': ({0: 0, 128: 1, 64: 2, 32: 3}, 62),
                  'tcgen05': ({0: 0, 128: 2, 64: 4, 32: 6}, 61)}


@dataclass(frozen=True)
class MatrixDescType:
    """The type of a shared-memory matrix descriptor (`Op.MATRIX_DESC`):
    the 64 bits NVIDIA's asynchronous matrix engines read an operand through,
    where it starts and how it is laid out (PTX ISA, "Asynchronous
    Warpgroup Level Matrix Shared Memory Layout" and its "Matrix Descriptor
    Format"; tcgen05's "Shared Memory Descriptor").

    The layout is a canonical one, K-major -- K contiguous in each row of M
    or N -- and built of core matrices of eight rows of 16 bytes of K.
    Without a swizzle a core matrix is 128 contiguous bytes, its rows 16
    apart; `lbo`, the leading byte offset, is between the core matrices next
    to each other along K, and `sbo`, the stride byte offset, between those
    next to each other along M or N.  With a swizzle (`swizzle`: 32, 64 or
    128 bytes) the eight rows of an atom are `swizzle` bytes each, K along
    them, and their 16-byte chunks are permuted by the row's place in the
    atom; `sbo` is between atoms along M or N, and `lbo` is not read for a
    product whose K fits a row (16, its field 1, is what the prototypes
    gave).  Which of the two offsets is which was settled on the hardware:
    the swapped reading computed garbage on GH200 and on GB200
    (`~/tf-probe/tcgen05`, `wg_selftest`, `gemm56`).

    `elem` is the element the operand holds; `form` the engine whose format
    the bits are in (`DESC_FORMATS`).  Like a resource, no part of what a
    lane computes with: a 64-bit value every lane of the collective holds
    alike.
    """

    elem: Any
    swizzle: int = 0
    lbo: int = 0
    sbo: int = 0
    form: str = 'wgmma'

    def __post_init__(self):
        if self.swizzle not in SWIZZLE_BYTES:
            raise IRError(f'a matrix descriptor swizzles {self.swizzle!r} '
                          f'bytes; there are {SWIZZLE_BYTES}')
        if self.form not in DESC_FORMATS:
            raise IRError(f'a matrix descriptor of the format {self.form!r}; '
                          f'there are {DESC_FORMATS}')
        for name in ('lbo', 'sbo'):
            value = getattr(self, name)
            if (not isinstance(value, int) or value < 0 or value % 16
                    or value >> 4 >= 1 << 14):
                raise IRError(f'a matrix descriptor with {name} {value!r}: a '
                              f'multiple of 16 bytes below {16 << 14}')

    def __repr__(self):
        swz = f', sw{self.swizzle}' if self.swizzle else ''
        return (f'mdesc<{self.elem}, {self.form}{swz}, lbo {self.lbo}, '
                f'sbo {self.sbo}>')


def matrix_descriptor_bits(desc: MatrixDescType) -> int:
    """Every bit of a descriptor of `desc` but the start address, which the
    device fills in from the operand's place (`tensorforge::
    matrixDescriptor`, bits 0 to 13, the address in 16-byte units).

    The leading and the stride byte offset in 16-byte units from bit 16 and
    from bit 32, and the swizzle at the top (`_SWIZZLE_FIELD`); tcgen05's
    format besides holds the constant 0b001 from bit 46.  The base offset,
    bits 49 to 51, is zero: a swizzled operand starts on the alignment of
    its atom (`swizzle_alignment`), where the pattern starts."""
    field, shift = _SWIZZLE_FIELD[desc.form]
    bits = (((desc.lbo >> 4) & 0x3FFF) << 16
            | ((desc.sbo >> 4) & 0x3FFF) << 32
            | field[desc.swizzle] << shift)
    if desc.form == 'tcgen05':
        bits |= 1 << 46
    return bits


def swizzle_alignment(swizzle: int) -> int:
    """Bytes a swizzled operand starts on: the eight rows of its atom, over
    which the permutation repeats -- 256, 512 and 1024 for 32, 64 and 128
    bytes; 16 without a swizzle, the unit of the start address."""
    return 8 * swizzle if swizzle else 16


def swizzled(address: int, swizzle: int) -> int:
    """`address` as a swizzle of `swizzle` bytes permutes it: the 16-byte
    chunk -- bits 4 on -- XORed with the row of the atom -- bits 7 on --
    one, two or three bits of each for 32, 64 and 128 bytes.  On the address
    as the hardware forms it, so an operand starts on `swizzle_alignment`.
    The same for a matrix engine reading through a descriptor and for a
    tensor copy writing the tile."""
    if not swizzle:
        return address
    bits = {32: 1, 64: 2, 128: 3}[swizzle]
    return address ^ (((address >> 7) & ((1 << bits) - 1)) << 4)


def swizzle_form(swizzle: int, elem: Any) -> Optional[XorSwizzle]:
    """TensorForge's swizzle (`XorSwizzle`) that is NVIDIA's of `swizzle`
    bytes over elements of `elem`, None for none: the 16-byte chunk a
    granule, a row eight of them -- the 128-byte line -- and the first two,
    four or eight permuted, by one, two or three bits of the line's number
    (`swizzled`).  So a buffer of that swizzle is addressed, element by
    element, where the tensor memory accelerator writes a box of that
    swizzle and wgmma and tcgen05 read an operand of it -- placed on the
    period (`swizzle_alignment`), since the hardware takes the bits from the
    shared address itself (GB200's and the RTX 5070's reading).  At 128
    bytes it is the swizzle TensorForge already had, `XorSwizzle(8, 16 /
    size)`."""
    if not swizzle:
        return None
    if swizzle not in SWIZZLE_BYTES:
        raise IRError(f'a swizzle of {swizzle!r} bytes; there are '
                      f'{SWIZZLE_BYTES}')
    size = elem.size()
    if size > 16 or 16 % size:
        raise IRError(f'a swizzle of 16-byte chunks over elements of {size}')
    return XorSwizzle(swizzle // 16, 16 // size, line=8)


def swizzle_bytes(form: Optional[XorSwizzle], elem: Any) -> Optional[int]:
    """The NVIDIA swizzle a buffer laid out as `form` is, in bytes -- 0 for
    none -- or None where it is a permutation of TensorForge's own that no
    copy or matrix engine of NVIDIA's reads (`swizzle_form`)."""
    if form is None:
        return 0
    for swizzle in SWIZZLE_BYTES[1:]:
        if form == swizzle_form(swizzle, elem):
            return swizzle
    return None


def descriptor_offset(desc: MatrixDescType, row: int, k: int) -> int:
    """Bytes past the start of an operand laid out as `desc` states where
    its element (`row`, `k`) is -- `row` along M or N, `k` along K -- before
    the swizzle, which the hardware applies to the address (`swizzled`)."""
    kb = k * desc.elem.size()
    if not desc.swizzle:
        return ((row // 8) * desc.sbo + (kb // 16) * desc.lbo
                + (row % 8) * 16 + kb % 16)
    return (row // 8) * desc.sbo + (row % 8) * desc.swizzle + kb


def asm_text(template: str, operands: Sequence[Tuple[str, Any]],
             name: Callable, volatile: bool = False,
             clobbers: Sequence[str] = ()) -> str:
    """Inline assembly: `template`, then the outputs and the inputs of
    `operands` -- `(constraint, value)`, outputs first, which is how the
    assembler numbers them -- each value spelled by `name`.

    `volatile` for an instruction whose order among the others of its kind
    the compiler has to keep, and which it must neither fold nor drop: an
    asynchronous one, whose effect the operands do not state.  `clobbers`,
    where there are any, follow the inputs (`"memory"`)."""
    outs = ', '.join(f'"{c}"({name(v)})' for c, v in operands
                     if c.startswith(('=', '+')))
    ins = ', '.join(f'"{c}"({name(v)})' for c, v in operands
                    if not c.startswith(('=', '+')))
    head = 'asm volatile(' if volatile else 'asm('
    if clobbers:
        listed = ', '.join(f'"{c}"' for c in clobbers)
        return f'{head}{template}\n: {outs}\n: {ins}\n: {listed}\n);'
    return f'{head}{template}\n: {outs}\n: {ins}\n);'


def fragments(s: Stmt) -> Tuple[Tuple[Any, ...], Tuple[Any, ...],
                                Tuple[Any, ...]]:
    """The A, B and C fragments of a `MATRIX_MAC`, the registers of each in
    the order the instruction takes them."""
    na, nb, nc = s.attr('fragments')
    return (s.args[:na], s.args[na:na + nb],
            s.args[na + nb:na + nb + nc])


#: The PTX type of an `mma.sync` operand, by what it holds.
_PTX_TYPES = {Datatype.TF32: 'tf32', Datatype.F32: 'f32',
              Datatype.F64: 'f64', Datatype.I8: 's8', Datatype.I32: 's32'}


def mma_sync_name(shape: Tuple[int, int, int],
                  types: Tuple[Any, Any, Any, Any]) -> str:
    """The `mma.sync` instruction of `shape` (m, n, k) over `types` (A, B,
    C, D): A row-major and B column-major, as every entry of the NVIDIA
    path takes them (`primitives.nvidia.INSTRS`)."""
    m, n, k = shape
    a, b, c, d = types
    return (f'mma.sync.aligned.m{m}n{n}k{k}.row.col.{_PTX_TYPES[d]}.'
            f'{_PTX_TYPES[a]}.{_PTX_TYPES[b]}.{_PTX_TYPES[c]}')


def _lower_mma_sync(s: Stmt, env: Spelling) -> Lowered:
    """`mma.sync` as inline PTX: C read and written in place, under `"+f"`
    (`"+d"` for FP64), and so listed once -- as D, at `%0` on -- and named
    again where the instruction reads C; A and B in 32-bit registers
    (`"r"`, a TF32 half each) or in FP64 ones (`"d"`).

    Listing the accumulator under `"=f"` and again under `"f"` would state
    two unrelated operands that happen to name one C++ lvalue, and nothing
    would then make the compiler give them one register: it could read the
    accumulator from one and write the product to another, dropping the
    accumulation.  So there is no form but the one in place."""
    if not s.attr('in_place'):
        raise IRError('mma.sync accumulates into its operand: a MATRIX_MAC '
                      'of it is in place')
    a, b, c = fragments(s)
    types = s.attr('types')
    if types[3] not in (Datatype.F32, Datatype.F64):
        raise IRError(f'mma.sync into {types[3]} is not written: the '
                      f'accumulator is FP32 or FP64 here')
    wide = types[3] == Datatype.F64
    acc, reg = ('d', 'd') if wide else ('f', 'r')
    operands = ([(f'+{acc}', v) for v in c] + [(reg, v) for v in a]
                + [(reg, v) for v in b])

    def group(count: int, first: int) -> str:
        return '{' + ','.join(f'%{i + first}' for i in range(count)) + '}'

    d = group(len(c), 0)
    template = (f'"{mma_sync_name(s.attr("shape"), types)} "\n'
                f'"{d}, {group(len(a), len(c))}, '
                f'{group(len(b), len(c) + len(a))}, {d};"')
    return Lowered((asm_text(template, operands, env.name),))


@dataclass(frozen=True)
class MatrixEngine:
    """What issues a `MATRIX_MAC`: D = A * B + C on the fragments the
    engine takes, as one instruction.

    ``collective`` is who executes the instruction together -- every lane
    of the wave for a warp-wide `mma.sync`, which a lane-varying branch
    around it breaks.  ``work`` is whether its issue counts as arithmetic
    (`Emitter._record_work`: one operation over a tile the emitter is not
    told the flops of).  ``pipe`` is the category `analysis.pipeline`
    counts the issue under, a function of the statement where the price is
    one of the shape.  ``spell`` writes the statement.

    ``completes`` is how the completion of a product is observed: on issue
    for the synchronous engines, by group for wgmma -- closed by
    `COMMIT_GROUP` and waited for by `WAIT_COUNTER` on the engine's counter
    (`GROUP_ENGINES`).  ``sources`` are where it may take A, B and C from,
    each a tuple of `Source`s.
    """

    name: str
    collective: Optional[Participants]
    work: bool
    pipe: Callable[[Stmt], str]
    spell: Lowering
    doc: str = ''
    hazards: Tuple[Hazard, ...] = ()
    completes: Completion = Completion.SYNC
    sources: Tuple[Tuple['Source', ...], ...] = (
        (Source.REGISTER,), (Source.REGISTER,), (Source.REGISTER,))


def _matrix(s: Stmt) -> str:
    return 'matrix'


#: The modifiers of AMD's MFMA builtins, in the order they follow the
#: operands: `cbsz` and `abid` broadcast a block of A to the others, `blgp`
#: swizzles B's lane groups.  Zero where a modifier is not stated.
MFMA_MODIFIERS = ('cbsz', 'abid', 'blgp')


def mfma_category(instruction: str) -> str:
    """The issue category an MFMA counts under, `mfma.<shape>` -- the shape
    as its builtin names it (`mfma.f32_4x4x1f32`, of `mfma_f32_4x4x1f32`):
    the instructions differ by an order of magnitude in how long they hold
    the SIMD, and `analysis.pipeline` prices each (`MFMA_CLOCKS`)."""
    if instruction.startswith('mfma_'):
        return f'mfma.{instruction[len("mfma_"):]}'
    return 'mfma'


def _mfma_pipe(s: Stmt) -> str:
    return mfma_category(s.attr('instruction') or '')


def _lower_mfma(s: Stmt, env: Spelling) -> Lowered:
    """An MFMA as its builtin, `__builtin_amdgcn_<instruction>(A, B, C,
    cbsz, abid, blgp)`: D is what it returns, the result's initializer."""
    if not s.target or s.attr('instruction') is None:
        raise IRError('an MFMA returns D and names its instruction: its '
                      'matrix op has a result and an `instruction`')
    a, b, c = fragments(s)
    modifiers = dict(s.attr('modifiers', ()))
    args = [env.name(v) for v in (*a, *b, *c)]
    args += [str(modifiers.get(name, 0)) for name in MFMA_MODIFIERS]
    return Lowered(result=f'__builtin_amdgcn_{s.attr("instruction")}('
                          f'{", ".join(args)})')


def _other(s: Stmt) -> str:
    return 'other'


def sources_of(s: Stmt) -> Tuple[Source, Source, Source]:
    """Where a `MATRIX_MAC` takes A, B and C from: its `sources`, or the
    registers where it states none -- every synchronous engine's."""
    return s.attr('sources') or (Source.REGISTER,) * 3


#: The PTX type of a `wgmma` operand, by what it holds.
_WGMMA_TYPES = {Datatype.TF32: 'tf32', Datatype.F32: 'f32'}

#: The macro nvcc defines compiling for sm_90a, and only then: not for the
#: `compute_90` PTX that `-arch=sm_90a` builds besides, where ptxas refuses
#: every wgmma instruction (CUDA 13.4).  The product is written inside it,
#: and the fence, the commit and the wait trap outside it (`cuda.h`), so a
#: build of either kind compiles and none runs a kernel without them.
WGMMA_FEATURE = '__CUDA_ARCH_FEAT_SM90_ALL'

#: The N a TF32 `wgmma` takes: 8 to 256 in steps of 8 (PTX ISA,
#: "wgmma.mma_async", the shapes of `.tf32`).
WGMMA_N = tuple(range(8, 257, 8))


def wgmma_name(shape: Tuple[int, int, int],
               types: Tuple[Any, Any, Any, Any]) -> str:
    """The `wgmma.mma_async` instruction of `shape` (m, n, k) over `types`
    (A, B, C, D): `.dtype.atype.btype` behind the shape (PTX ISA,
    "wgmma.mma_async", ISA 8.0, sm_90a)."""
    m, n, k = shape
    a, b, _, d = types
    return (f'wgmma.mma_async.sync.aligned.m{m}n{n}k{k}.'
            f'{_WGMMA_TYPES[d]}.{_WGMMA_TYPES[a]}.{_WGMMA_TYPES[b]}')


def wgmma_refusal(s: Stmt) -> Optional[str]:
    """Why `s` is no `wgmma` this lowering writes, or None.

    TF32 operands into an FP32 accumulator, which is what the F32 paths
    multiply (3xTF32), at m64nNk8: N a multiple of 8 up to 256.  D is N / 2
    registers a lane -- the warpgroup's 128 lanes hold the 64 x N tile --
    and A, where it is in registers, four (64 x 8 over 128 lanes).  A
    descriptor is one operand.  The one scalar operand is `scale-d`: 0
    writes D = A * B, 1 adds to what D holds; a value of either, or the
    literal.  TF32 is K-major in both operands, so there is no transpose to
    state, and the immediate scales of A and B are 1."""
    shape, types = s.attr('shape'), s.attr('types')
    if tuple(types or ()) != (Datatype.TF32, Datatype.TF32, Datatype.F32,
                              Datatype.F32):
        return (f'a wgmma over {types!r}: TF32 operands into an FP32 '
                f'accumulator are written here')
    if not (isinstance(shape, tuple) and len(shape) == 3 and shape[0] == 64
            and shape[1] in WGMMA_N and shape[2] == 8):
        return f'a wgmma of shape {shape!r}: m64nNk8, N in 8, 16, ..., 256'
    a, b, c = fragments(s)
    src_a, src_b, src_c = sources_of(s)
    if src_b is not Source.SHARED or src_c is not Source.REGISTER:
        return 'a wgmma takes B from shared memory and keeps D in registers'
    if len(c) != shape[1] // 2:
        return (f'{len(c)} registers of D for N = {shape[1]}: a lane holds '
                f'{shape[1] // 2}')
    if src_a is Source.REGISTER and len(a) != 4:
        return f'{len(a)} registers of A, where a lane holds four'
    for name, frag, src in (('A', a, src_a), ('B', b, src_b)):
        if src is Source.SHARED and not all(
                isinstance(v, Value) and isinstance(v.type, MatrixDescType)
                and v.type.form == 'wgmma' for v in frag):
            return f"{name} through no descriptor of wgmma's format"
    scalars = s.args[len(a) + len(b) + len(c):]
    if s.attr('scalars') != ('scale_d',) or len(scalars) != 1 or (
            not isinstance(scalars[0], Value) and scalars[0] not in (0, 1)):
        return (f'scale-d is one operand, 0 or 1 or a value of either, not '
                f'{scalars!r}')
    # The hardware takes the descriptors and scale-d from uniform registers
    # (`gdesc[UR]`, `UPT` in the SASS of sm_90a): one value for the whole
    # warpgroup, which a value of each lane is not.
    shared = [v for frag, src in zip((a, b), (src_a, src_b))
              if src is Source.SHARED for v in frag]
    for v in shared + list(scalars):
        if isinstance(v, Value) and v.uniformity <= Uniformity.LANE:
            return (f'{v!r} differs between the lanes, and the warpgroup '
                    f'takes one of it')
    # And D is the lane's own: every lane holds other elements of it.  An
    # accumulator declared the same in every lane is one a pass may take
    # for that, as one of `mma.sync`'s once was (`nvidia.matmul`).
    for v in c:
        if not isinstance(v, Value) or v.uniformity > Uniformity.LANE:
            return (f'{v!r} is a register of D, which differs between the '
                    f'lanes, and is declared as one they agree on')
    return None


def _wgmma_available(env: Spelling) -> None:
    """Refuse a target that has no wgmma: on CUDA, sm_90a alone
    (`Target.wgmma`).  Without a target -- an IR-level test -- it is
    written as it would be for one."""
    if env.family not in ('cuda', ''):
        raise IRError(f'wgmma has no spelling on {env.family}')
    has = getattr(env.target, 'wgmma', None)
    if callable(has) and not has():
        raise IRError(f'wgmma on {env.model or "this target"}: sm_90a is the '
                      f'target that has it')


def _lower_wgmma(s: Stmt, env: Spelling) -> Lowered:
    """`wgmma.mma_async` as inline PTX (PTX ISA, "wgmma.mma_async", ISA
    8.0, sm_90a): D read and written in place under `"+f"`, at `%0` on; A
    through its descriptor (`"l"`) or in four 32-bit registers (`"r"`); B
    through its descriptor; `scale-d` the literal where it is one, and
    where it is a value a predicate set from it.

    Volatile: the instruction reads shared memory through its descriptors,
    which no operand states, and it is one of a sequence -- the fence ahead
    of it, the commit and the wait behind -- whose order the compiler has
    to keep; an `asm` that is not volatile may be moved across the others,
    or two identical ones folded into one (as nvcc does with `mma.sync`).
    Inside the guard of the target that has it (`WGMMA_FEATURE`)."""
    if not s.attr('in_place'):
        raise IRError('wgmma accumulates into its D registers: a MATRIX_MAC '
                      'of it is in place')
    _wgmma_available(env)
    why = wgmma_refusal(s)
    if why is not None:
        raise IRError(why)
    a, b, c = fragments(s)
    scale = s.args[len(a) + len(b) + len(c)]
    shared_a = sources_of(s)[0] is Source.SHARED
    operands = ([('+f', v) for v in c]
                + [('l' if shared_a else 'r', v) for v in a]
                + [('l', v) for v in b])

    def group(count: int, first: int) -> str:
        return '{' + ', '.join(f'%{i + first}' for i in range(count)) + '}'

    d = group(len(c), 0)
    a_text = f'%{len(c)}' if shared_a else group(len(a), len(c))
    b_text = f'%{len(c) + len(a)}'
    name = wgmma_name(s.attr('shape'), s.attr('types'))
    flag = scale if isinstance(scale, Value) else None

    def spell(v):
        # "r" takes 32 bits, and a flag is one byte: widened, the asm
        # operand is the register `setp` tests (nvcc refuses the byte).
        text = env.name(v)
        return (f'static_cast<uint32_t>({text})'
                if flag is not None and v is flag else text)

    if flag is not None:
        operands.append(('r', scale))
        template = (f'"{{\\n.reg .pred p;\\nsetp.ne.b32 p, '
                    f'%{len(operands) - 1}, 0;\\n"\n'
                    f'"{name} "\n'
                    f'"{d}, {a_text}, {b_text}, p, 1, 1;\\n}}"')
    else:
        template = (f'"{name} "\n'
                    f'"{d}, {a_text}, {b_text}, {int(scale)}, 1, 1;"')
    return Lowered((f'#if defined({WGMMA_FEATURE})',
                    asm_text(template, operands, spell, volatile=True),
                    '#endif'))


#: What the source of a product guards a tcgen05 instruction with: defined
#: by `cuda.h` where nvcc compiles for a target that has them (sm_100a,
#: sm_103a, and their families), and nowhere else -- not for the
#: `compute_100` PTX that `-arch=sm_100a` builds besides.
TCGEN05_FEATURE = 'TENSORFORGE_TCGEN05'

#: The N a TF32 tcgen05.mma takes at M = 128 on one CTA: 16 to 256 in
#: steps of 16 (PTX ISA, "tcgen05.mma", the valid shapes of `.kind::tf32`).
TCGEN05_N = tuple(range(16, 257, 16))

#: The columns of tensor memory a block may allocate: a power of two from
#: 32 to 512 (PTX ISA, "tcgen05.alloc").
TMEM_COLUMNS = (32, 64, 128, 256, 512)

#: The lanes of tensor memory: one a row of an M = 128 product's D.
TMEM_LANES = 128


def tcgen05_idesc(shape: Tuple[int, int, int]) -> int:
    """The instruction descriptor of a `kind::tf32` tcgen05.mma of `shape`
    (PTX ISA, "tcgen05 Instruction descriptor"): D in FP32 (bits 4-5: 1), A
    and B TF32 (bits 7-9 and 10-12: 2), both K-major (bits 15 and 16: 0),
    N >> 3 from bit 17 and M >> 4 from bit 24.  GB200's reading
    (`~/tf-probe/tcgen05/gemm56.cu`, `volume8.cu`)."""
    m, n, _ = shape
    return (1 << 4) | (2 << 7) | (2 << 10) | ((n >> 3) << 17) | ((m >> 4) << 24)


def _is_tmem(v: Any) -> bool:
    return (isinstance(v, Value) and isinstance(v.type, BufferType)
            and v.type.space is MemSpace.TMEM)


def tcgen05_refusal(s: Stmt) -> Optional[str]:
    """Why `s` is no tcgen05.mma this lowering writes, or None.

    TF32 operands into an FP32 accumulator at M = 128 on one CTA, N a
    multiple of 16 up to 256, K = 8 (PTX ISA, "tcgen05.mma"): D the columns
    `d_column` on of a tensor memory buffer, one row a lane; A one
    descriptor of tcgen05's format, or from tensor memory (`a_column` on,
    eight columns); B one descriptor.  The scalar operands are
    `enable_input_d`, 0 writes D = A * B, 1 adds to D -- a value of
    either, or the literal -- and `issue`, the value that picks the one
    thread that issues it, which is the thread that commits it too
    (`commit_group`): a product commits only what its own thread issued."""
    shape, types = s.attr('shape'), s.attr('types')
    if tuple(types or ()) != (Datatype.TF32, Datatype.TF32, Datatype.F32,
                              Datatype.F32):
        return (f'a tcgen05.mma over {types!r}: TF32 operands into an FP32 '
                f'accumulator are written here')
    if not (isinstance(shape, tuple) and len(shape) == 3 and shape[0] == 128
            and shape[1] in TCGEN05_N and shape[2] == 8):
        return (f'a tcgen05.mma of shape {shape!r}: m128nNk8, N in 16, 32, '
                f'..., 256')
    if s.attr('fragments') != (1, 1, 1):
        return 'a tcgen05.mma takes one operand each of A, B and D'
    (a,), (b,), (c,) = fragments(s)
    src_a, src_b, src_c = sources_of(s)
    if src_b is not Source.SHARED or src_c is not Source.TMEM:
        return ('a tcgen05.mma takes B from shared memory and keeps D in '
                'tensor memory')
    for name, v, src in (('A', a, src_a), ('B', b, src_b)):
        if src is Source.SHARED and not (
                isinstance(v, Value) and isinstance(v.type, MatrixDescType)
                and v.type.form == 'tcgen05'):
            return f'{name} through {v!r}, no descriptor of tcgen05\'s format'
    columns = {'d_column': (c, shape[1])}
    if src_a is Source.TMEM:
        columns['a_column'] = (a, shape[2])
    for key, (buffer, width) in columns.items():
        at = s.attr(key)
        if not _is_tmem(buffer):
            return f'{buffer!r} is no buffer of tensor memory'
        if not isinstance(at, int) or at < 0 \
                or at + width > buffer.type.shape[1]:
            return (f'{key} {at!r} and {width} columns past the '
                    f'{buffer.type.shape[1]} of {buffer}')
    scalars = s.args[3:]
    if s.attr('scalars') != ('enable_input_d', 'issue') or len(scalars) != 2:
        return 'the scalars of a tcgen05.mma are enable_input_d and issue'
    enable, issue = scalars
    if not isinstance(enable, Value) and enable not in (0, 1):
        return f'enable-input-d is 0 or 1 or a value of either, not {enable!r}'
    if not isinstance(issue, Value):
        return (f'issue is the value that picks the thread, not {issue!r}: '
                f'every thread would issue it')
    return None


def _tcgen05_available(env: Spelling) -> None:
    """Refuse a target that has no tcgen05: on CUDA, sm_100a
    (`Target.tcgen05`).  Without a target it is written as for one."""
    if env.family not in ('cuda', ''):
        raise IRError(f'tcgen05 has no spelling on {env.family}')
    has = getattr(env.target, 'tcgen05', None)
    if callable(has) and not has():
        raise IRError(f'tcgen05 on {env.model or "this target"}: sm_100a is '
                      f'the target that has it')


def _guarded(feature: str, lines: Sequence[str]) -> Tuple[str, ...]:
    return (f'#if defined({feature})', *lines, '#endif')


def _lower_tcgen05(s: Stmt, env: Spelling) -> Lowered:
    """`tcgen05.mma.cta_group::1.kind::tf32` as inline PTX (PTX ISA,
    "tcgen05.mma", ISA 8.6, sm_100a), issued by the one thread `issue`
    picks, under a predicate inside the asm: D's address in tensor memory,
    A through its descriptor (`"l"`) or at its address in tensor memory
    (`[%1]`), B through its descriptor, the instruction descriptor an
    immediate (`tcgen05_idesc`), `enable-input-d` the literal or a
    predicate set from a value.  The flags are widened to the 32 bits "r"
    takes.  Volatile and clobbering memory, as every tcgen05 instruction
    here: what it reads and writes, no operand states.  Inside the guard of
    the targets that have it (`TCGEN05_FEATURE`)."""
    if not s.attr('in_place'):
        raise IRError('tcgen05.mma accumulates into tensor memory: a '
                      'MATRIX_MAC of it is in place')
    _tcgen05_available(env)
    why = tcgen05_refusal(s)
    if why is not None:
        raise IRError(why)
    (a,), (b,), (c,) = fragments(s)
    enable, issue = s.args[3:]
    tmem_a = sources_of(s)[0] is Source.TMEM

    def at(buffer, key):
        column = s.attr(key)
        return f'{env.name(buffer)} + {column}u' if column else env.name(buffer)

    def flag(v):
        return f'static_cast<uint32_t>({env.name(v)})'

    operands = [('r', at(c, 'd_column')),
                ('r', at(a, 'a_column')) if tmem_a else ('l', env.name(a)),
                ('l', env.name(b)),
                ('n', f'{tcgen05_idesc(s.attr("shape")):#010x}u'),
                ('r', flag(issue))]
    if isinstance(enable, Value):
        operands.append(('r', flag(enable)))
        head, predicate = 'setp.ne.b32 p, %5, 0;\\n', 'p'
    else:
        head, predicate = '', str(int(enable))
    a_text = '[%1]' if tmem_a else '%1'
    template = (f'"{{\\n.reg .pred e, p;\\nsetp.ne.b32 e, %4, 0;\\n{head}"\n'
                f'"@e tcgen05.mma.cta_group::1.kind::tf32 [%0], {a_text}, %2, '
                f'%3, {predicate};\\n}}"')
    return Lowered(_guarded(TCGEN05_FEATURE, (asm_text(
        template, operands, lambda x: x, volatile=True,
        clobbers=('memory',)),)))


#: The function the SPMD lowering's DPAS is written as
#: (`tensorforge_device/sycl_spmd.h`).
DPAS_SPMD = 'tensorforge::dpasTF32'


def _lower_dpas_spmd(s: Stmt, env: Spelling) -> Lowered:
    """DPAS from the SPMD lowering, `tensorforge::dpasTF32(sub-group, A, B,
    C)` (`DPAS_SPMD`): D is what it returns, the sub-group the one the lexic
    spells."""
    group = (env.lexic.active_sub_group_mask() if env.lexic is not None
             else None)
    if not s.target or group is None:
        raise IRError('DPAS under SPMD returns D and runs over the sub-group '
                      'the target spells')
    a, b, c = fragments(s)
    args = [group] + [env.name(v) for v in (*a, *b, *c)]
    return Lowered(result=f'{DPAS_SPMD}({", ".join(args)})')


def _lower_dpas_esimd(s: Stmt, env: Spelling) -> Lowered:
    """DPAS under the explicit vector, `tensorforge::intel_xmx::dpas<depth,
    repeat, T>(C, B, A)`: the systolic depth, the repeat count -- M --, the
    accumulator's element, and the operands in XMX's order, B its Src1 and
    A its Src2.  D is what it returns."""
    if not s.target:
        raise IRError('DPAS returns D: its matrix op has a result')
    a, b, c = fragments(s)
    m, _, _ = s.attr('shape')
    depth = dict(s.attr('modifiers', ())).get('depth')
    types = s.attr('types')
    args = [env.name(v) for v in (*c, *b, *a)]
    return Lowered(result=f'tensorforge::intel_xmx::dpas<{depth}, {m}, '
                          f'{types[3].ctype()}>({", ".join(args)})')


MATRIX_ENGINES: Dict[str, MatrixEngine] = {e.name: e for e in (
    MatrixEngine('mma_sync', Participants.WAVE, True, _matrix,
                 _lower_mma_sync,
                 "NVIDIA's warp-wide mma.sync, TF32 operands in 32-bit "
                 'registers into an FP32 accumulator (sm_80 on)'),
    MatrixEngine('dmma', Participants.WAVE, True, _matrix, _lower_mma_sync,
                 "NVIDIA's FP64 mma.sync, which ptxas lowers to DMMA.8x8x4 "
                 'whatever its shape from sm_100 on (`Emitter._issues`)'),
    MatrixEngine('mfma', Participants.WAVE, True, _mfma_pipe, _lower_mfma,
                 "AMD's MFMA (CDNA), one builtin returning D, the "
                 'instruction named by the catalog row it is '
                 '(`amd.catalog.MatrixOp`) and priced by its shape '
                 '(`mfma_category`)'),
    # Intel's DPAS counts as other code, and as no work: the figures it was
    # counted under as a call and as raw text, kept as they were -- so a
    # build that issues it counts its arithmetic short.
    MatrixEngine('dpas_spmd', Participants.WAVE, False, _other,
                 _lower_dpas_spmd,
                 "Intel's DPAS from the SPMD lowering, TF32 into FP32 at an "
                 'execution size of 16: the sub-group of 16 issues it '
                 'together (`primitives.intel_spmd`)'),
    MatrixEngine('dpas_esimd', None, False, _other, _lower_dpas_esimd,
                 "Intel's DPAS under the explicit vector (`xmx::dpas`): the "
                 'work-item issues it alone, the sub-group being its vector '
                 '(`primitives.intel`)'),
    MatrixEngine('wgmma', Participants.WARPGROUP, True, _matrix,
                 _lower_wgmma,
                 "NVIDIA's asynchronous product of a warpgroup (sm_90a): A "
                 'and B in shared memory through descriptors, A also in '
                 'registers, D in registers, written in place; complete '
                 'once a wait on its counter retires the group it was '
                 'committed in (`COMMIT_GROUP`, `matrix_wait`)',
                 completes=Completion.GROUP,
                 sources=((Source.SHARED, Source.REGISTER), (Source.SHARED,),
                          (Source.REGISTER,))),
    # One thread issues it, the one `issue` picks: no collective executes
    # it together.
    MatrixEngine('tcgen05', None, True, _matrix, _lower_tcgen05,
                 "NVIDIA's asynchronous product of the 5th generation tensor "
                 'core (sm_100a): A and B in shared memory through '
                 'descriptors, A also in tensor memory, D in tensor memory; '
                 'issued by one thread, complete once the mbarrier its '
                 'thread committed it to has moved on (`COMMIT_GROUP` onto '
                 'an object, `WAIT_PHASE`)',
                 completes=Completion.BARRIER,
                 sources=((Source.SHARED, Source.TMEM), (Source.SHARED,),
                          (Source.TMEM,))),
)}

#: The engines whose products complete by group, and the counter a wait for
#: their groups is written on (`matrix_wait`): wgmma's own, which counts
#: nothing else.
GROUP_ENGINES = {'wgmma': 'wgmma'}

#: The engines that commit their products (`COMMIT_GROUP`), by how a
#: committed group completes: on the engine's counter, or onto the mbarrier
#: the commit names, which a wait for its phase retires (`WAIT_PHASE`).
COMMIT_ENGINES = {'wgmma': Completion.GROUP, 'tcgen05': Completion.BARRIER}


def matrix_engine(s: Stmt) -> MatrixEngine:
    """The engine a `MATRIX_MAC` names."""
    found = MATRIX_ENGINES.get(s.attr('engine'))
    if found is None:
        raise IRError(f'a matrix op on engine {s.attr("engine")!r}; there '
                      f'are {", ".join(sorted(MATRIX_ENGINES))}')
    return found


def _lower_matrix_mac(s: Stmt, env: Spelling) -> Lowered:
    return matrix_engine(s).spell(s, env)


def _matrix_pipe(s: Stmt) -> str:
    return matrix_engine(s).pipe(s)


def _check_matrix_mac(s: Stmt) -> List[str]:
    """A `MATRIX_MAC` takes the registers its fragments count, and has a
    result unless it writes C in place -- which it then declares as a read
    and a write of each of C's registers.

    An operand in shared memory is one descriptor (`MatrixDescType`), from
    a source its engine takes it from, and the op declares a read of shared
    memory for it, behind C's registers; an operand in registers is no
    descriptor.  The operands behind the fragments are the ones `scalars`
    names.  An engine that completes after the issue writes D in place."""
    diag = []
    engine = MATRIX_ENGINES.get(s.attr('engine'))
    if engine is None:
        diag.append(f'{s.op}: engine {s.attr("engine")!r} is none of '
                    f'{", ".join(sorted(MATRIX_ENGINES))}')
    counts = s.attr('fragments')
    scalars = s.attr('scalars') or ()
    if (not isinstance(counts, tuple) or len(counts) != 3
            or sum(counts) + len(scalars) != len(s.args)):
        diag.append(f'{s.op}: {len(s.args)} operands for the fragments '
                    f'{counts!r}' + (f' and {scalars!r}' if scalars else ''))
        return diag
    shared = 0
    for i, (name, frag, source) in enumerate(zip('ABC', fragments(s),
                                                 sources_of(s))):
        if engine is not None and source not in engine.sources[i]:
            diag.append(f'{s.op}: {engine.name} takes no {name} from '
                        f'{getattr(source, "value", source)}')
        descs = [v for v in frag if isinstance(v, Value)
                 and isinstance(v.type, MatrixDescType)]
        if source is Source.SHARED:
            shared += 1
            if len(frag) != 1 or len(descs) != 1:
                diag.append(f'{s.op}: {name} in shared memory is one '
                            f'descriptor, not {len(frag)} operand(s)')
        elif descs:
            diag.append(f'{s.op}: {name} in {source.value} through a '
                        f'descriptor')
    if s.attr('in_place'):
        if s.target:
            diag.append(f'{s.op}: in place, and a result besides')
        c = fragments(s)[2]
        src_a, src_b, src_c = sources_of(s)
        want = [v for v in c if isinstance(v, Value)]
        space_c = (MemSpace.TMEM if src_c is Source.TMEM
                   else MemSpace.REGISTER)
        regs = s.accesses[:len(want)]
        if [a.base for a in regs] != want or any(
                a.space != space_c
                or a.kind != Effect.READ | Effect.WRITE
                for a in regs):
            diag.append(f'{s.op}: in place without a read and a write of '
                        f'each register of C')
        reads = s.accesses[len(want):]
        spaces = [MemSpace.SHARED if src is Source.SHARED else MemSpace.TMEM
                  for src in (src_a, src_b) if src is not Source.REGISTER]
        if [a.space for a in reads] != spaces or any(
                a.kind != Effect.READ for a in reads):
            diag.append(f'{s.op}: {shared} operand(s) in shared memory and '
                        f'{len(reads)} other access(es) declared, where each '
                        f'is a read of the memory an operand is in')
    elif engine is not None and engine.completes is not Completion.SYNC:
        diag.append(f'{s.op}: {engine.name} completes behind its issue, and '
                    f'writes D in place')
    elif len(s.target) != 1:
        diag.append(f'{s.op}: {len(s.target)} results, where D is one')
    return diag


def _lower_commit_group_cuda(s: Stmt, env: Spelling) -> Lowered:
    """`wgmma.commit_group`: the products this warpgroup issued since its
    last commit, one group (PTX ISA, "wgmma.commit_group", sm_90a).

    tcgen05's: the products the thread `issue` picks issued, tracked by the
    mbarrier the object is, which they arrive at once complete
    (`tcgen05.commit.cta_group::1.mbarrier::arrive::one`, PTX ISA,
    "tcgen05.commit", sm_100a) -- written by every thread, which notes on
    its cursor that a phase is pending, and issued by that one
    (`tensorforge::tcgen05Commit`)."""
    engine = s.attr('engine')
    if engine == 'wgmma':
        _wgmma_available(env)
        return Lowered(('tensorforge::wgmmaCommit();',))
    if engine == 'tcgen05':
        _tcgen05_available(env)
        _spelled_mbarrier(s, env, object_value(s))
        return Lowered((f'tensorforge::tcgen05Commit('
                        f'{env.name(object_value(s))}, '
                        f'{env.name(s.args[0])});',))
    raise IRError(f'no group of {engine!r} to commit under CUDA')


def _lower_commit_group_other(s: Stmt, env: Spelling) -> Lowered:
    raise IRError(f'no group of {s.attr("engine")!r} to commit on '
                  f'{env.family or "no target"}')


def _check_commit_group(s: Stmt) -> List[str]:
    """A commit names an engine whose products complete by group: on its
    counter, with no object and no operand (wgmma), or onto an mbarrier,
    the object, with the value picking the thread that issued them
    (tcgen05)."""
    engine = s.attr('engine')
    if engine not in COMMIT_ENGINES:
        return [f'{s.op}: {engine!r} commits no groups; '
                f'{", ".join(sorted(COMMIT_ENGINES))} do']
    obj = object_of(s)
    if COMMIT_ENGINES[engine] is Completion.BARRIER:
        if obj is None or obj.kind is not SyncKind.MBARRIER \
                or object_value(s) is None:
            return [f'{s.op}: {engine} commits onto an mbarrier the body '
                    f'set up, not {obj!r}']
        if len(s.args) != 1 or not isinstance(s.args[0], Value):
            return [f'{s.op}: {engine} commits what the thread one value '
                    f'picks issued']
    elif obj is not None or s.args:
        return [f'{s.op}: {engine} commits onto its counter, with no object '
                f'and no operand']
    return []


#: The constraint an operand fence names a register with, by its element:
#: a TF32 fragment is a 32-bit integer register (`tensorforge::tf32`).
_REGISTER_CONSTRAINTS = {Datatype.F32: 'f', Datatype.F64: 'd',
                         Datatype.TF32: 'r', Datatype.I32: 'r'}


def operand_fence(registers: Sequence[Value], name: Callable) -> str:
    """An empty `asm volatile` that reads and writes `registers`, so that
    the compiler keeps every other access of them on the side of it where
    it stands.  The fence, the commit and the wait of an asynchronous
    engine are volatile statements, which the compiler keeps in order, but
    they name no register: without this a read of D could be scheduled
    ahead of the wait that makes it valid, or a write of D behind the fence
    that orders it before the product."""
    operands = []
    for v in registers:
        base = getattr(getattr(v, 'type', None), 'base', None)
        letter = _REGISTER_CONSTRAINTS.get(base)
        if letter is None:
            raise IRError(f'an operand fence over {v!r}, whose register '
                          f'class is not known')
        operands.append(f'"+{letter}"({name(v)})')
    return f'asm volatile("" : {", ".join(operands)} :: "memory");'


def _no_tmem(s: Stmt, env: Spelling) -> Lowered:
    raise IRError(f'{s.op}: no tensor memory on {env.family or "no target"}')


def _check_tmem(s: Stmt) -> List[str]:
    """The tensor memory ops: a power of two of columns from 32 to 512, and
    the buffer each names in tensor memory."""
    diag = []
    if s.op in (Op.TMEM_ALLOC, Op.TMEM_DEALLOC):
        if s.attr('columns') not in TMEM_COLUMNS:
            diag.append(f'{s.op}: {s.attr("columns")!r} columns, where a '
                        f'block allocates one of {TMEM_COLUMNS}')
    buffer = s.target[0] if s.op == Op.TMEM_ALLOC else (
        s.args[0] if s.args else None)
    if not _is_tmem(buffer):
        diag.append(f'{s.op}: {buffer!r} is no buffer of tensor memory')
    if s.op in (Op.TMEM_LD, Op.TMEM_ST):
        count = len(s.target) if s.op == Op.TMEM_LD else len(s.args) - 1
        if count not in TMEM_COUNTS:
            diag.append(f'{s.op}: {count} columns a thread, where the '
                        f'instruction moves one of {TMEM_COUNTS}')
    return diag


#: The columns a thread loads or stores at once (`.32x32b.x{N}`, PTX ISA,
#: "tcgen05.ld" and "tcgen05.st").
TMEM_COUNTS = (1, 2, 4, 8, 16, 32, 64, 128)


def _lower_tmem_alloc(s: Stmt, env: Spelling) -> Lowered:
    """`columns` columns of tensor memory for the block, allocated by its
    first warp into the shared slot the operand is, given up for others to
    allocate behind it, and handed to every thread through the slot behind
    a block barrier between the engine's fences (`tensorforge::tmemAlloc`;
    PTX ISA, "tcgen05.alloc", "tcgen05.relinquish_alloc_permit")."""
    _tcgen05_available(env)
    return Lowered(result=f'tensorforge::tmemAlloc<{s.attr("columns")}>('
                          f'&{env.name(s.args[0])}[0])')


def _lower_tmem_dealloc(s: Stmt, env: Spelling) -> Lowered:
    """The columns freed by the first warp, behind a block barrier between
    the engine's fences, so that no thread reads them again
    (`tensorforge::tmemDealloc`; PTX ISA, "tcgen05.dealloc")."""
    _tcgen05_available(env)
    return Lowered((f'tensorforge::tmemDealloc<{s.attr("columns")}>('
                    f'{env.name(s.args[0])});',))


def _tmem_address(s: Stmt, env: Spelling) -> str:
    """The warp's quarter of the lanes, from `column` on: a warp reaches
    the lanes 32 (w % 4) to 32 (w % 4) + 31 alone (PTX ISA, "Tensor
    Memory": "Access restrictions")."""
    return (f'tensorforge::tmemWarpAddress({env.name(s.args[0])}, '
            f'{s.attr("column")}u)')


def _lower_tmem_ld(s: Stmt, env: Spelling) -> Lowered:
    """`tcgen05.ld.sync.aligned.32x32b.x{N}.b32` and the wait for it (PTX
    ISA, "tcgen05.ld", "tcgen05.wait"): thread t of a warp the lane 32 (w
    % 4) + t of the buffer, N columns from `column` on, into its N
    results -- declared here, as FP32."""
    _tcgen05_available(env)
    names = [env.name(v) for v in s.target]
    count = len(names)
    if count not in TMEM_COUNTS:
        raise IRError(f'a load of {count} columns of tensor memory a thread; '
                      f'the instruction moves one of {TMEM_COUNTS}')
    regs = '{' + ', '.join(f'%{i}' for i in range(count)) + '}'
    template = (f'"tcgen05.ld.sync.aligned.32x32b.x{count}.b32 {regs}, '
                f'[%{count}];"')
    operands = [('=f', n) for n in names] + [('r', _tmem_address(s, env))]
    return Lowered(tuple(f'float {n}{{}};' for n in names)
                   + _guarded(TCGEN05_FEATURE, (asm_text(
                       template, operands, lambda x: x, volatile=True),))
                   + ('tensorforge::tmemWaitLd();',))


def _lower_tmem_st(s: Stmt, env: Spelling) -> Lowered:
    """`tcgen05.st.sync.aligned.32x32b.x{N}.b32` and the wait for it: thread
    t of a warp the lane 32 (w % 4) + t, N columns from `column` on."""
    _tcgen05_available(env)
    values = s.args[1:]
    count = len(values)
    if count not in TMEM_COUNTS:
        raise IRError(f'a store of {count} columns of tensor memory a '
                      f'thread; the instruction moves one of {TMEM_COUNTS}')
    regs = '{' + ', '.join(f'%{i + 1}' for i in range(count)) + '}'
    template = (f'"tcgen05.st.sync.aligned.32x32b.x{count}.b32 [%0], '
                f'{regs};"')
    operands = ([('r', _tmem_address(s, env))]
                + [('f', env.name(v)) for v in values])
    return Lowered(_guarded(TCGEN05_FEATURE, (asm_text(
        template, operands, lambda x: x, volatile=True,
        clobbers=('memory',)),)) + ('tensorforge::tmemWaitSt();',))


def _thread_sync_fence(s: Stmt, env: Spelling) -> Lowered:
    """`tcgen05.fence::before_thread_sync` or `::after_thread_sync` (PTX
    ISA, "tcgen05.fence", sm_100a): the engine's operations ahead of it
    before the barrier or wait behind it, or those behind it after the one
    ahead of it."""
    if s.attr('engine') != 'tcgen05' or env.family != 'cuda':
        raise IRError(f'a fence of {s.attr("engine")!r} and the threads\' '
                      f'synchronization on {env.family or "no target"}: '
                      f'tcgen05\'s, on CUDA')
    _tcgen05_available(env)
    side = {'before': 'Before', 'after': 'After'}.get(s.attr('side'))
    if side is None:
        raise IRError(f'a fence on the side {s.attr("side")!r} of a '
                      f'synchronization')
    return Lowered((f'tensorforge::tcgen05Fence{side}();',))


def _wait_wgmma(s: Stmt, env: Spelling) -> Lowered:
    """`wgmma.wait_group N` (PTX ISA, "wgmma.wait_group", sm_90a): at most
    `count` of this warpgroup's groups still in flight; then the operand
    fence over the registers the wait names, so that none of their reads
    moves ahead of it (`operand_fence`)."""
    count = s.attr('count')
    if not isinstance(count, int) or count < 0:
        raise IRError(f'a wait for at most {count!r} groups of wgmma')
    _wgmma_available(env)
    lines = (f'tensorforge::wgmmaWait<{count}>();',)
    registers = [v for v in s.args if isinstance(v, Value)]
    if registers:
        lines += (operand_fence(registers, env.name),)
    return Lowered(lines)


def pipe_of(s: Stmt) -> str:
    """The category `analysis.pipeline` counts the issue of `s`, an engine
    op, under (`OpSpec.pipe`)."""
    pipe = REGISTRY[s.op].pipe
    return pipe(s) if callable(pipe) else pipe


def units_of(s: Stmt) -> int:
    """What `s`, a statement without regions that lays code down, adds to
    the code count in the emitter's units: its entry's `units` for an engine
    op (`OpSpec.units`), one for any other (`transfers.executed`)."""
    spec = REGISTRY.get(s.op)
    if spec is None:
        return 1
    return spec.units(s) if callable(spec.units) else spec.units


# --------------------------------------------------------------------------- #
# The entries
# --------------------------------------------------------------------------- #

_OBJECT = Slot('object', (SyncObjectType, Value))
_SYNC = Effect.BARRIER
_INT = (int,)
_MATRIX_TRANSFER_SLOTS = (Slot('engine', (str,)), Slot('shape', (tuple,)),
                          Slot('count', _INT), Slot('width', _INT, False),
                          Slot('layout', (object,), False))

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
                                   Slot('objects', _INT, False),
                                   Slot('leader', (str,), False),
                                   Slot('extern', (str,), False)),
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
    # A wait, counted as the waits are (`sync`); on an mbarrier the wait for
    # the copies whose tokens it names, an access of what they wrote as
    # their `wait` is (`effect_of`).
    OpSpec(Op.WAIT_PHASE, operands=(_OBJECT,),
           results=(), effect=_SYNC, movable=False,
           completes=Completion.SYNC, retires=Completion.BARRIER,
           settles=True, units=1, pipe='sync',
           lower={'*': _lower_wait_phase}),
    # A wall wherever the raw text it replaces was one: opaque to every
    # pass, though what it orders is known (see the module).  A fence of the
    # schedule placed behind the passes (`wall=False`) orders the device
    # compiler alone and has no effect here (`effect_of`).
    OpSpec(Op.FENCE, operands=(Slot('kind', (FenceKind,)),
                               Slot('scope', (SyncScope,), False),
                               Slot('mask', _INT, False),
                               Slot('group', (tuple,), False),
                               Slot('wall', (bool,), False),
                               Slot('engine', (str,), False),
                               Slot('side', (str,), False)),
           results=(), effect=Effect.UNKNOWN, movable=False,
           completes=Completion.SYNC, retires=None, settles=False,
           units=1, pipe='other', lower={'*': _lower_fence},
           check=_check_fence),
    OpSpec(Op.WAIT_COUNTER, operands=(Slot('counter', (str,)),
                                      Slot('count', _INT),
                                      Slot('engine', (str,), False)),
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
    # Close what an asynchronous engine issued since its last commit into
    # one group, which a wait on the engine's counter retires
    # (`GROUP_ENGINES`, `matrix_wait`).  Pinned: which products a group
    # holds is where it stands.  Asynchronous, as an issue is: nothing of
    # it may be predicated away, and DCE keeps it.
    OpSpec(Op.COMMIT_GROUP, operands=(Slot('engine', (str,)),
                                      Slot('object', (SyncObjectType, Value),
                                           False)),
           results=(), effect=Effect.ASYNC, movable=False,
           completes=Completion.GROUP, retires=None, settles=False,
           units=1, pipe='sync',
           lower={'cuda': _lower_commit_group_cuda,
                  '': _lower_commit_group_cuda,
                  '*': _lower_commit_group_other},
           check=_check_commit_group),
    # Tensor memory for the block (tcgen05): its columns allocated by the
    # first warp and handed to every thread through a shared slot behind a
    # block barrier -- which the passes see, as they see a barrier -- and
    # freed the same way; a warp's loads and stores of its quarter of the
    # lanes, collectives of the warp, pinned.
    OpSpec(Op.TMEM_ALLOC, operands=(Slot('columns', _INT),),
           results=('tmem',), effect=Effect.BARRIER | Effect.WRITE,
           movable=False, completes=Completion.SYNC, retires=None,
           settles=True, units=1, pipe='barrier',
           lower={'cuda': _lower_tmem_alloc, '': _lower_tmem_alloc,
                  '*': _no_tmem},
           arrival=Uniformity.BLOCK, check=_check_tmem),
    OpSpec(Op.TMEM_DEALLOC, operands=(Slot('columns', _INT),),
           results=(), effect=Effect.BARRIER | Effect.WRITE,
           movable=False, completes=Completion.SYNC, retires=None,
           settles=True, units=1, pipe='barrier',
           lower={'cuda': _lower_tmem_dealloc, '': _lower_tmem_dealloc,
                  '*': _no_tmem},
           arrival=Uniformity.BLOCK, check=_check_tmem),
    OpSpec(Op.TMEM_LD, operands=(Slot('column', _INT),),
           results=('values',), effect=Effect.READ, movable=False,
           completes=Completion.SYNC, retires=None, settles=False,
           units=1, pipe='other',
           lower={'cuda': _lower_tmem_ld, '': _lower_tmem_ld,
                  '*': _no_tmem}, check=_check_tmem),
    OpSpec(Op.TMEM_ST, operands=(Slot('column', _INT),),
           results=(), effect=Effect.WRITE, movable=False,
           completes=Completion.SYNC, retires=None, settles=False,
           units=1, pipe='other',
           lower={'cuda': _lower_tmem_st, '': _lower_tmem_st,
                  '*': _no_tmem}, check=_check_tmem),
    # A warp's matrices between shared memory and its registers, one
    # collective transfer (`TRANSFER_ENGINES`: ldmatrix, stmatrix): a read or
    # a write of the buffer whose rows the lanes name, declared as one, and
    # pinned where it stands -- every lane of the warp takes part.  The
    # slots: the engine, the shape of a matrix as PTX names it, how many
    # (`count`), the bytes a lane holds (`width`), and the fragment layout
    # over the tile the matrices lie in, where the producer states one
    # (`matrix_transfer_bits`).  ldmatrix adds no code (`units`) and no issue
    # (`_matrix_transfer_pipe`), as the call it was before it was an op.
    OpSpec(Op.MATRIX_LOAD, operands=_MATRIX_TRANSFER_SLOTS,
           results=('registers',), effect=Effect.READ, movable=False,
           completes=Completion.SYNC, retires=None, settles=False,
           units=0, pipe=_matrix_transfer_pipe,
           lower={'cuda': _lower_matrix_transfer,
                  '': _lower_matrix_transfer, '*': _no_matrix_transfer},
           check=_check_matrix_transfer),
    OpSpec(Op.MATRIX_STORE, operands=_MATRIX_TRANSFER_SLOTS,
           results=(), effect=Effect.WRITE, movable=False,
           completes=Completion.SYNC, retires=None, settles=False,
           units=1, pipe=_matrix_transfer_pipe,
           lower={'cuda': _lower_matrix_transfer,
                  '': _lower_matrix_transfer, '*': _no_matrix_transfer},
           check=_check_matrix_transfer),
    # Registers from one distribution over the lanes to another
    # (`RELAYOUT_STRATEGIES`): a value of a value, registers written through
    # references, or registers renamed.  Pinned, as the cross-lane calls it
    # replaced were -- defined only where the lanes it reads are converged
    # -- and of no memory; a value pure where the caller says so, for CSE
    # and inlining; one written through references declares the writes
    # (`effect_of`).  The slots: the strategy, its parameters, the operands
    # it writes, a relabel's order, and the layouts where they are stated
    # as bits (source, target, the relabel of the indices).
    OpSpec(Op.RELAYOUT, operands=(Slot('strategy', (str,)),
                                  Slot('params', (tuple,), False),
                                  Slot('written', _INT, False),
                                  Slot('order', (tuple,), False),
                                  Slot('layouts', (tuple,), False)),
           results=('registers',), effect=Effect.NONE, movable=False,
           completes=Completion.SYNC, retires=None, settles=False,
           units=_relayout_units, pipe=_relayout_pipe,
           lower={'*': _lower_relayout}, check=_check_relayout),
    # D = A * B + C on a matrix engine (`MATRIX_ENGINES`).  The fragments
    # are values -- the operands, A's registers, B's, then C's -- and D the
    # result, or C itself where the instruction writes it in place
    # (`in_place`), declared then as a read and a write of each of C's
    # registers (`effect_of`).  Pinned: one instruction of the wave, and
    # the passes reorder no collective.  The other slots: the engine, the
    # shape (m, n, k), the element types of A, B, C and D, the register
    # counts of the fragments, the instruction where the engine has several
    # of a shape (an MFMA's builtin), the fragment layouts where the vendor
    # states them (`BitLayout`s of A, B and D, None where it does not), the
    # term of a split product (which halves of A and B), the vendor's
    # modifiers, and the collective that executes it.
    OpSpec(Op.MATRIX_MAC, operands=(Slot('engine', (str,)),
                                    Slot('shape', (tuple,)),
                                    Slot('types', (tuple,)),
                                    Slot('fragments', (tuple,)),
                                    Slot('instruction', (str,), False),
                                    Slot('in_place', (bool,), False),
                                    Slot('layouts', (tuple,), False),
                                    Slot('term', (tuple,), False),
                                    Slot('modifiers', (tuple,), False),
                                    Slot('collective', (Participants,),
                                         False),
                                    Slot('sources', (tuple,), False),
                                    Slot('scalars', (tuple,), False),
                                    Slot('a_column', _INT, False),
                                    Slot('d_column', _INT, False)),
           results=('d',), effect=Effect.NONE, movable=False,
           completes=Completion.SYNC, retires=None, settles=False,
           units=1, pipe=_matrix_pipe, lower={'*': _lower_matrix_mac},
           check=_check_matrix_mac),
)}


def spec_of(op: str) -> Optional[OpSpec]:
    return REGISTRY.get(op)


def lower(s: Stmt, target: Any, name: Optional[Callable] = None) -> Lowered:
    """`s` as written for `target` (a `Target`, or None); `name` spells a
    value the op names (`Emitter.operand`)."""
    spec = REGISTRY[s.op]
    env = target if isinstance(target, Spelling) else Spelling(target, name)
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
    if spec.check is not None:
        diag.extend(spec.check(s))
    effect = effect_of(s.op, s.attrs)
    movable = movable_of(s.op, s.attrs)
    if s.effect != effect or s.movable != movable:
        diag.append(f'{s.op}: declares {s.effect!r}, '
                    f'{"movable" if s.movable else "pinned"}, where the '
                    f'registry says {effect!r}, '
                    f'{"movable" if movable else "pinned"}')
    return diag


def _waits_on_mbarrier(op: str, attrs: Tuple) -> bool:
    """A wait on an mbarrier: the wait for the copies that complete on it,
    which is what a `wait` was before it was spelled as the object's."""
    if op != Op.WAIT_PHASE:
        return False
    obj = dict(attrs).get('object')
    return (isinstance(obj, Value) and isinstance(obj.type, SyncObjectType)
            and obj.type.kind is SyncKind.MBARRIER)


def _retires_copies(op: str, attrs: Tuple) -> bool:
    """A wait that stands for a `wait` on transfers: the one on an mbarrier,
    and a counter's that names the engine whose transfers it retires
    (`wait_counter`)."""
    if _waits_on_mbarrier(op, attrs):
        return True
    return op == Op.WAIT_COUNTER and dict(attrs).get('engine') is not None


def effect_of(op: str, attrs: Tuple = ()) -> Effect:
    """The effect a statement of `op` with `attrs` declares: the registry's,
    but none for a fence that is no wall (`fence_stmt`), a write for a
    matrix op that writes C in place -- of its registers, the accesses say
    -- and for the wait on
    an mbarrier or a counter's for transfers it names the `wait`'s -- it
    makes what the copies wrote visible, the accesses they declared, and
    orders nobody else's (`wait_phase`, `wait_counter`)."""
    if op == Op.FENCE and dict(attrs).get('wall') is False:
        return Effect.NONE
    if op == Op.MATRIX_MAC and _matrix_effect(attrs) is not None:
        return _matrix_effect(attrs)
    if op == Op.MATRIX_MAC and dict(attrs).get('in_place'):
        return Effect.WRITE
    if op == Op.RELAYOUT and dict(attrs).get('written'):
        return Effect.WRITE
    if _retires_copies(op, attrs):
        return Effect.READ | Effect.WRITE | Effect.ASYNC
    return REGISTRY[op].effect


def _matrix_effect(attrs: Tuple) -> Optional[Effect]:
    """The effect of a matrix op that reads an operand in shared memory, or
    whose engine completes behind the issue: the write of D in place, the
    read, and `ASYNC` -- which keeps it out of a predicated region and out
    of DCE's reach, as it does an asynchronous copy.  None for any other,
    whose effect is the one it always had."""
    a = dict(attrs)
    engine = MATRIX_ENGINES.get(a.get('engine'))
    asynchronous = (engine is not None
                    and engine.completes is not Completion.SYNC)
    shared = any(src in (Source.SHARED, Source.TMEM)
                 for src in (a.get('sources') or ())[:2])
    if not asynchronous and not shared:
        return None
    effect = Effect.WRITE if a.get('in_place') else Effect.NONE
    if shared:
        effect |= Effect.READ
    if asynchronous:
        effect |= Effect.ASYNC
    return effect


def completes_on_barrier(s: Stmt) -> bool:
    """A bulk copy, or the wait for one: what every thread waits for on the
    barrier the copy completes on, which makes the bytes visible to each --
    so no lane hands them to another (`barriers`)."""
    return (s.attr('engine') == 'bulk'
            or _waits_on_mbarrier(s.op, s.attrs))


def movable_of(op: str, attrs: Tuple = ()) -> bool:
    """Whether a statement of `op` with `attrs` may move: the registry's
    answer, but a wait on an mbarrier, or a counter's for the transfers it
    names, moves as the `wait` it was moved, subject to the accesses of the
    copies it retires."""
    if _retires_copies(op, attrs):
        return True
    return REGISTRY[op].movable


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
    return Stmt(op=op, target=target, args=args, pure=False,
                movable=movable_of(op, attrs), effect=effect_of(op, attrs),
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


def wait_phase(obj, tokens: Sequence[Value] = (),
               accesses: Tuple[Access, ...] = ()) -> Stmt:
    """Wait for the current phase of `obj` (see `_lower_wait_phase` for the
    arrival a named object's wait is).

    On an mbarrier -- `obj` the `Value` its setup defined -- the wait for
    the copies that complete on it: `tokens` are theirs, which it consumes
    as a `wait` would (`check_tokens`, `asyncmem`), and `accesses` what they
    wrote and read, which it makes visible."""
    return _stmt(Op.WAIT_PHASE, (('object', obj),), args=tuple(tokens),
                 accesses=tuple(accesses))


def mbarrier(participants: Participants, wave: int,
             threads: Optional[int] = None) -> SyncObjectType:
    """The object a bulk copy completes on, waited on by `participants`:
    eight bytes of shared memory with a phase bit, one arrival a phase --
    the issuing thread's, with the bytes it expects (`tensorforge::
    bulk_copy`)."""
    scope = {Participants.WAVE: SyncScope.WARP,
             Participants.MULT: (SyncScope.WARP
                                 if threads is not None and threads < wave
                                 else SyncScope.MULT),
             Participants.GRID: SyncScope.GRID}.get(participants,
                                                    SyncScope.BLOCK)
    return SyncObjectType(scope, SyncKind.MBARRIER, participants, threads,
                          wave)


def sync_init_value(obj: Value, storage: Value, *, leader: str,
                    extern: Optional[str] = None,
                    count: int = 1) -> Stmt:
    """Set up the object `obj` a body creates, in `storage`, armed by the
    thread `leader` (text, the condition that picks it) for `count`
    arrivals a phase: the statement defines `obj`, which `extern` names in
    the source where something else spells it."""
    attrs = (('count', int(count)), ('leader', leader))
    if extern is not None:
        attrs += (('extern', extern),)
    return _stmt(Op.SYNC_INIT, attrs, args=(storage,), target=(obj,),
                 accesses=(Access(Effect.WRITE, MemSpace.SHARED, storage),))


def fence_stmt(kind: FenceKind, *, scope: Optional[SyncScope] = None,
               mask: Optional[int] = None,
               group: Optional[Tuple[int, int, int]] = None,
               wall: bool = True, engine: Optional[str] = None,
               registers: Sequence[Value] = (),
               side: Optional[str] = None) -> Stmt:
    """A fence of `kind`.  `scope` is whose memory a memory fence orders
    (the block's where None), `mask` the backend's own scheduling mask (AMD
    `sched_barrier`, vISA's fence bits), and `group` an AMD scheduling group
    -- mask, size, id -- rather than a wall.

    A fence of registers (`FenceKind.REGISTERS`) names the `engine` whose
    accesses it orders and the `registers` it orders them for, which it
    declares as read and written: the accumulators and the A fragments a
    product behind it reads, where something else touched them.

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
    if engine is not None:
        attrs += (('engine', engine),)
    if side is not None:
        attrs += (('side', side),)
    registers = tuple(registers)
    if registers and kind is not FenceKind.REGISTERS:
        raise IRError(f'a {kind.value} fence names no registers')
    accesses = tuple(Access(Effect.READ | Effect.WRITE, MemSpace.REGISTER, v)
                     for v in registers)
    return _stmt(Op.FENCE, attrs, args=registers, accesses=accesses)


def wait_counter(counter: str, count: int, tokens: Sequence[Value] = (),
                 accesses: Optional[Tuple[Access, ...]] = None,
                 engine: Optional[str] = None) -> Stmt:
    """Wait until at most `count` operations are outstanding on `counter`
    (`group`, `vmcnt`, `lgkmcnt`, `asynccnt`, `tensorcnt`; `bulk_group`, a
    thread's bulk groups, which its reductions of boxes are closed into).

    Without `tokens` it writes what the transfers it retires wrote, and
    which those are is the counter's to know, not the statement's: so any
    buffer of shared memory, where the transfers a counter tracks land --
    one into a register is ordered by its own def-use edge.

    With them, the wait for the transfers through `engine` they name
    (`IRBuilder.wait`): it consumes them as a `wait` would (`check_tokens`,
    `core.retires`), makes visible what they declared, `accesses`, and
    moves as a `wait` does.  Its count is then not the caller's but
    `schedule_async`'s, once the schedule is final: the transfers issued
    behind the latest one it names that the counter certainly counts
    (`TransferEngine.one_to_one`) -- zero, the full drain, until then.
    """
    attrs = (('counter', str(counter)), ('count', int(count)))
    if engine is not None:
        attrs += (('engine', engine),)
    if accesses is None:
        accesses = (Access(Effect.READ | Effect.WRITE, MemSpace.SHARED,
                           None),)
    return _stmt(Op.WAIT_COUNTER, attrs, args=tuple(tokens),
                 accesses=tuple(accesses))


def commit_group(engine: str, obj: Optional[Value] = None,
                 issue: Optional[Value] = None) -> Stmt:
    """Close what `engine` issued since its last commit into one group
    (`COMMIT_ENGINES`): `wgmma.commit_group`; tcgen05's onto the mbarrier
    `obj` (`IRBuilder.bulk_channel`), what the thread `issue` picks issued.
    Pinned and asynchronous; the object named in `object`, as a bulk copy
    names it."""
    if engine not in COMMIT_ENGINES:
        raise IRError(f'{engine!r} commits no groups; '
                      f'{", ".join(sorted(COMMIT_ENGINES))} do')
    attrs = (('engine', engine),)
    args = ()
    if COMMIT_ENGINES[engine] is Completion.BARRIER:
        if obj is None or issue is None:
            raise IRError(f'{engine} commits onto an mbarrier, what the '
                          f'thread `issue` picks issued')
        attrs += (('object', obj),)
        args = (issue,)
    elif obj is not None or issue is not None:
        raise IRError(f'{engine} commits onto its counter')
    return _stmt(Op.COMMIT_GROUP, attrs, args=args)


def tmem_alloc(tmem: Value, slot: Value, columns: int) -> Stmt:
    """Allocate `columns` columns of tensor memory as the buffer `tmem`,
    through the shared slot `slot` (`_lower_tmem_alloc`): it writes and
    reads the slot, and is a barrier of the block."""
    if columns not in TMEM_COLUMNS:
        raise IRError(f'{columns} columns of tensor memory; a block '
                      f'allocates one of {TMEM_COLUMNS}')
    return _stmt(Op.TMEM_ALLOC, (('columns', int(columns)),), args=(slot,),
                 target=(tmem,),
                 accesses=(Access(Effect.READ | Effect.WRITE, MemSpace.SHARED,
                                  slot),))


def tmem_dealloc(tmem: Value) -> Stmt:
    """Free the columns of `tmem`: a write of the buffer, so that no access
    of it moves behind, and a barrier of the block."""
    return _stmt(Op.TMEM_DEALLOC, (('columns', int(tmem.type.shape[1])),),
                 args=(tmem,),
                 accesses=(Access(Effect.WRITE, MemSpace.TMEM, tmem),))


def tmem_ld(tmem: Value, column: int, targets: Sequence[Value]) -> Stmt:
    """A warp's load of `len(targets)` columns of its quarter of `tmem`'s
    lanes, from `column` on, into `targets`."""
    return _stmt(Op.TMEM_LD, (('column', int(column)),), args=(tmem,),
                 target=tuple(targets),
                 accesses=(Access(Effect.READ, MemSpace.TMEM, tmem),))


def tmem_st(tmem: Value, column: int, values: Sequence[Any]) -> Stmt:
    """A warp's store of `values` to its quarter of `tmem`'s lanes, one
    column each from `column` on."""
    return _stmt(Op.TMEM_ST, (('column', int(column)),),
                 args=(tmem,) + tuple(values),
                 accesses=(Access(Effect.WRITE, MemSpace.TMEM, tmem),))


def _matrix_transfer_attrs(engine: str, count: int, layout: Any) -> Tuple:
    attrs = (('engine', engine), ('shape', MATRIX_TRANSFER_SHAPE),
             ('count', int(count)), ('width', 4 * int(count)))
    return attrs + ((('layout', layout),) if layout is not None else ())


def matrix_load(base: Any, index: Any, registers: Sequence[Value], *,
                access: Access, layout: Any = None) -> Stmt:
    """`len(registers)` matrices into `registers` by ldmatrix, from the rows
    the lanes name at `base + index`: one read of the buffer, `access`
    (`IRBuilder.matrix_load`); `layout` the fragment layout over the tile
    the matrices lie in, where the caller states one."""
    return _stmt(Op.MATRIX_LOAD,
                 _matrix_transfer_attrs('ldmatrix', len(registers), layout),
                 args=(base, index), target=tuple(registers),
                 accesses=(access,))


def matrix_store(base: Any, index: Any, registers: Sequence[Any], *,
                 access: Access, layout: Any = None) -> Stmt:
    """`registers` into `len(registers)` matrices by stmatrix, the rows the
    lanes name at `base + index`: one write of the buffer, `access`
    (`IRBuilder.matrix_store`)."""
    return _stmt(Op.MATRIX_STORE,
                 _matrix_transfer_attrs('stmatrix', len(registers), layout),
                 args=(base, index) + tuple(registers), accesses=(access,))


def relayout(strategy: str, params: Mapping, args: Sequence[Any],
             targets: Sequence[Value] = (), *, pure: bool = True,
             written: int = 0, order: Optional[Tuple[int, ...]] = None,
             layouts: Optional[Tuple] = None,
             no_inline: bool = False) -> Stmt:
    """A relayout by `strategy` (`RELAYOUT_STRATEGIES`) of `args`: into
    `targets` -- one value, or the registers renamed in `order` -- or, with
    `written`, into its first `written` operands through references,
    declared as a read and a write each.  `params` the strategy's, in a
    fixed order; `pure` whether CSE may merge two alike, as the call it
    replaced said; `no_inline` keeps a value in a variable of its own."""
    attrs: Tuple = (('strategy', strategy),)
    fields = RELAYOUT_STRATEGIES[strategy].fields
    params = {k: v for k, v in dict(params).items() if k in fields}
    if params:
        attrs += (('params', tuple(sorted(params.items()))),)
    if written:
        attrs += (('written', int(written)),)
    if order is not None:
        attrs += (('order', tuple(order)),)
    if layouts is not None:
        attrs += (('layouts', tuple(layouts)),)
    if no_inline:
        attrs += (('no_inline', True),)
    accesses = tuple(Access(Effect.READ | Effect.WRITE, MemSpace.REGISTER, w)
                     for w in args[:written] if isinstance(w, Value))
    return Stmt(op=Op.RELAYOUT, target=tuple(targets), args=tuple(args),
                pure=pure and not written, movable=movable_of(
                    Op.RELAYOUT, attrs),
                effect=effect_of(Op.RELAYOUT, attrs), accesses=accesses,
                attrs=attrs)


def matrix_wait(engine: str, count: int, registers: Sequence[Value] = (),
                reads: Sequence[Any] = ()) -> Stmt:
    """Wait until at most `count` of the groups `engine` committed are in
    flight (`wgmma.wait_group`), on the engine's own counter
    (`GROUP_ENGINES`).

    What the products of the retired groups wrote is valid behind it, and
    what they read may be written again: the `registers` -- their D, and
    their A where it is in registers -- which it names and declares as read
    and written, and the shared buffers `reads`, read through their
    descriptors, which it declares as read.  Pinned, as the products and
    the commit are: its count is the caller's, a fact about where the
    groups stand."""
    counter = GROUP_ENGINES.get(engine)
    if counter is None:
        raise IRError(f'{engine!r} commits no groups to wait for; '
                      f'{", ".join(sorted(GROUP_ENGINES))} do')
    if int(count) < 0:
        raise IRError(f'a wait for at most {count} groups')
    registers = tuple(registers)
    accesses = (tuple(Access(Effect.READ | Effect.WRITE, MemSpace.REGISTER, v)
                      for v in registers)
                + tuple(Access(Effect.READ, MemSpace.SHARED, b)
                        for b in reads))
    return _stmt(Op.WAIT_COUNTER, (('counter', counter),
                                   ('count', int(count))),
                 args=registers, accesses=accesses)


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
    'Completion', 'DERIVED', 'FenceKind', 'GLOBAL_POLICY', 'Hazard',
    'Lowered', 'kept_facts',
    'OpSpec', 'TRANSFER_ENGINES', 'TransferEngine', 'UNRESOLVED',
    'cache_policy', 'completes_of', 'completes_on_barrier',
    'copy_cache', 'copy_engine',
    'counter_of', 'element_of', 'engine_of', 'lane_linear', 'lane_prefix',
    'lds_dma_refusal', 'mbarrier', 'movable_of', 'object_value',
    'one_to_one', 'sync_init_value', 'waited_on', 'width_of',
    'PRIORITY_WAVES', 'REGISTRY', 'Slot', 'Spelling', 'SyncKind',
    'SyncObjectType', 'SyncScope', 'arrival_of', 'arrive', 'barrier',
    'effect_of', 'family', 'fence_stmt', 'lower', 'object_demand',
    'object_of', 'objects', 'set_priority', 'settles', 'spec_of',
    'sync_init', 'sync_object', 'verify_engine', 'wait_counter',
    'wait_phase', 'ResourceType',
    'TDM_ELEMENT_BYTES', 'TDM_PITCH_LIMIT', 'TDM_TEAMS', 'TDM_TILE_LIMIT',
    'TENSOR_FENCE', 'TensorDescType', 'tdm_refusal',
    'MATRIX_ENGINES', 'MatrixEngine', 'asm_text', 'fragments',
    'matrix_engine', 'mma_sync_name', 'pipe_of',
    'MFMA_MODIFIERS', 'mfma_category', 'DPAS_SPMD',
    'Source', 'MatrixDescType', 'SWIZZLE_BYTES', 'DESC_FORMATS',
    'matrix_descriptor_bits', 'swizzle_alignment', 'swizzled',
    'descriptor_offset', 'sources_of', 'WGMMA_FEATURE', 'WGMMA_N',
    'wgmma_name', 'wgmma_refusal', 'GROUP_ENGINES', 'operand_fence',
    'commit_group', 'matrix_wait',
    'COMMIT_ENGINES', 'TCGEN05_FEATURE', 'TCGEN05_N', 'TMEM_COLUMNS',
    'TMEM_COUNTS', 'TMEM_LANES', 'tcgen05_idesc', 'tcgen05_refusal',
    'tmem_alloc', 'tmem_dealloc', 'tmem_ld', 'tmem_st',
    'TMA_BOX_LIMIT', 'TMA_DIM_LIMIT', 'TMA_FILLS', 'TMA_GLOBAL_ALIGN',
    'TMA_RANKS', 'TMA_SHARED_ALIGN', 'TMA_STRIDE_LIMIT', 'TMA_TYPES',
    'TMA_REDUCE_TYPES', 'TensorMap', 'staging_engine', 'swizzle_bytes',
    'swizzle_form',
    'tensor_map_refusal', 'tma_refusal',
    'MATRIX_ROWS', 'MATRIX_ROW_BYTES', 'MATRIX_TRANSFER_COUNTS',
    'MATRIX_TRANSFER_OPS', 'MATRIX_TRANSFER_SHAPE', 'MATRIX_TRANSFER_SM',
    'matrix_load', 'matrix_registers', 'matrix_store',
    'matrix_transfer_bits', 'matrix_transfer_refusal', 'units_of',
    'fill_of', 'fill_refusal',
    'AMD_FAMILIES', 'RELAYOUT_FORMS', 'RELAYOUT_STRATEGIES', 'RelayoutStrategy',
    'relabel_order', 'relabeling', 'relayout', 'relayout_callee',
    'relayout_refusal',
    'relayout_strategy',
]
