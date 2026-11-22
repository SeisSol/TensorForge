# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""A block in N roles (`pir.roles`, `IRBuilder.split_roles`): each role a
range of the block's warps with a body of its own, meeting the others on
rings of mbarriers alone and within itself on its own named barrier.  The
statement is a chain of two-region IFs, one link a role but the last two,
which every pass reads as the IFs it is; two roles are the statement
`IRBuilder.roles` has always built.

What is checked here: the split's ranges and barrier ids; the chain and its
shape; the verifier on the roles' barriers -- a block's barrier, another
role's, one between two links, a chain a statement broke; the barrier
placement, each role's hazards met on its own barrier and the accesses
across any two roles reported; the host oracle (`reference.kernel_eval`) on
pipelines of three and four roles over rings of mbarriers -- every value,
no access unordered, the arms of the nested IFs as coroutines, and the
hangs and races the protocol can make refused; and the roles' register
budgets, which draw on the block's own allocation.
"""

from __future__ import annotations

import re

import pytest

import test_mbarrier_ring as tmr
from tensorforge.backend.pir import engines, roles as pr
from tensorforge.backend.pir.allocate import BLOCK, MULT, allocate
from tensorforge.backend.pir.build import IRBuilder, RingSlot
from tensorforge.backend.pir.core import (BOOL, INDEX, IRError, MemSpace, Op,
                                          Participants, ScalarType, Stmt,
                                          Uniformity, walk, walk_stmts)
from tensorforge.backend.pir.passes import verify
from tensorforge.backend.pir.roles import Role, RoleSplit
from tensorforge.common.basic_types import Datatype
from tensorforge.reference import kernel_eval

F32 = Datatype.F32
#: Floats a stage's slot holds: a lane of two warps each.
WIDTH = 64


# -- the split ------------------------------------------------------------------ #

def _split(*ranges, names=None, first=None):
    names = names or [f'r{i}' for i in range(len(ranges))]
    return RoleSplit(tuple(Role(n, 32 * c, warps=(lo, c))
                           for n, (lo, c) in zip(names, ranges)),
                     first=pr.first_id(len(ranges)) if first is None
                     else first)


def test_two_roles_are_the_producers_split():
    """The producer's warp behind sixteen element warps: one step at the
    element warps' end, the two top ids -- the text the multi-element
    producer has always written."""
    split = _split((16, 1), (0, 16), names=['producer', 'elements'])
    assert split.ident == '(14 + (threadIdx.y >= 16))'
    assert [r.ident for r in split.roles] == [split.ident] * 2
    assert (split.id_of('elements'), split.id_of('producer')) == (14, 15)


def test_n_roles_take_the_top_ids_in_their_warps_order():
    """Any order of the roles: the id is a step at each range's start, so
    the role on the lowest warps takes the first id and every other the
    next -- the same text in every role's region."""
    split = _split((5, 1), (0, 4), (4, 1), (6, 4),
                   names=['operator', 'star', 'issuer', 'epilogue'])
    assert split.first == 12 and split.warps == 10
    assert split.ident == ('(12 + (threadIdx.y >= 4) + (threadIdx.y >= 5)'
                           ' + (threadIdx.y >= 6))')
    assert [split.id_of(n) for n in ('star', 'issuer', 'operator',
                                     'epilogue')] == [12, 13, 14, 15]
    for warp in range(10):
        held = [r.name for r in split.roles
                if r.warps[0] <= warp < r.warps[0] + r.warps[1]]
        evaluated = eval(split.ident.replace('threadIdx.y', str(warp)))
        assert evaluated == split.id_of(held[0])


@pytest.mark.parametrize('ranges,why', [
    (((0, 2),), 'two or more'),
    (((0, 2), (1, 2)), 'take it twice'),
    (((0, 2), (3, 1)), 'leave warp 2 out'),
    (((1, 2), (3, 1)), 'leave warp 0 out'),
    (((0, 2), (2, 0)), 'holds 0 warps')])
def test_a_split_that_does_not_tile_the_warps_is_refused(ranges, why):
    with pytest.raises(IRError, match=why):
        _split(*ranges, first=14)


def test_a_split_names_each_role_once():
    with pytest.raises(IRError, match='two roles of one name'):
        _split((0, 1), (1, 1), names=['a', 'a'])


def test_sixteen_roles_take_more_ids_than_a_block_has():
    with pytest.raises(IRError, match='named barriers'):
        pr.first_id(16)


# -- the chain ------------------------------------------------------------------ #

def _three(b, split=None):
    """Three roles over an empty body each: the chain `split_roles`
    builds."""
    split = split or _split((3, 1), (0, 2), (2, 1),
                            names=['producer', 'mixer', 'consumer'])
    h = b.split_roles(split)
    for role in split.roles:
        with h.role(role.name):
            b.barrier(Participants.ROLE)
    return split


def test_three_roles_are_a_chain_of_two_links():
    """The first link: the producer's region, then the next link alone --
    the mixer's region and the consumer's -- each link carrying the roles
    from its own on, its condition the role's range."""
    b = IRBuilder(fptype=F32)
    split = _three(b)
    body = b.finish()
    first = body[-1]
    assert pr.is_split(first) and pr.chain_refusal(first) is None
    assert first.attr('roles') == split.roles
    (link,) = first.regions[1].body
    assert link.attr('roles') == split.roles[1:]
    assert [r.name for r, _ in pr.regions(first)] == ['producer', 'mixer',
                                                      'consumer']
    conds = [s for s in body[:-1] if s.op in ('ge', 'lt')]
    assert [(s.op, s.args[1]) for s in conds] == [('ge', 3), ('lt', 2)]
    assert verify(body, strict=False) == []


def test_two_roles_are_the_statement_roles_builds():
    """The split of two roles and `IRBuilder.roles` with the same condition
    build the same statement."""
    split = _split((1, 1), (0, 1), names=['producer', 'elements'])
    a = IRBuilder(fptype=F32)
    h = a.split_roles(split)
    with h.role('producer'):
        a.barrier(Participants.ROLE)
    with h.role('elements'):
        a.barrier(Participants.ROLE)
    b = IRBuilder(fptype=F32)
    cond = b.op('ge', BOOL, b.thread_id('y'), 1, hint='producer')
    h = b.roles(cond, *split.roles)
    with h.then():
        b.barrier(Participants.ROLE)
    with h.otherwise():
        b.barrier(Participants.ROLE)
    left, right = a.finish(), b.finish()
    assert len(left) == len(right)
    for x, y in zip(walk_stmts(left), walk_stmts(right)):
        assert (x.op, x.attrs, len(x.regions)) == (y.op, y.attrs,
                                                   len(y.regions))


def test_the_regions_go_in_the_splits_order():
    b = IRBuilder(fptype=F32)
    h = b.split_roles(_split((0, 1), (1, 1), names=['a', 'b']))
    with pytest.raises(IRError, match='in the split\'s order'):
        with h.role('b'):
            pass


def test_rebuild_puts_the_regions_back_in_the_chain():
    b = IRBuilder(fptype=F32)
    _three(b)
    first = b.finish()[-1]
    marks = []
    placed = []
    for i, (_, r) in enumerate(pr.regions(first)):
        mark = Stmt(op=Op.MARK, attrs=(('mark', f'm{i}'),))
        marks.append(mark)
        placed.append(type(r)(args=r.args, body=(mark,) + r.body))
    again = pr.rebuild(first, placed)
    assert [r.body[0] for _, r in pr.regions(again)] == marks
    assert pr.chain_refusal(again) is None


# -- the verifier --------------------------------------------------------------- #

def _verified(build):
    b = IRBuilder(fptype=F32)
    split = _split((3, 1), (0, 2), (2, 1),
                   names=['producer', 'mixer', 'consumer'])
    h = b.split_roles(split)
    for role in split.roles:
        with h.role(role.name):
            build(b, role, split)
    return verify(b.finish(), strict=False)


def test_each_role_meets_on_its_own_barrier():
    assert _verified(lambda b, role, split: b.barrier(Participants.ROLE)) \
        == []


def test_a_block_barrier_inside_a_role_is_refused():
    """The other roles never come."""
    def build(b, role, split):
        if role.name == 'mixer':
            b.barrier(Participants.BLOCK)
    diag = _verified(build)
    assert any('block-wide barrier' in d for d in diag), diag


def test_another_roles_barrier_is_refused():
    """The consumer's barrier in the mixer's region: the mixer's threads
    compute their own id, and wait for the consumer's count of them."""
    def build(b, role, split):
        if role.name == 'mixer':
            b.emit(engines.barrier(engines.sync_object(
                Participants.ROLE, 1, split.role('consumer').threads,
                split.ident)))
    diag = _verified(build)
    assert any('in the region of the role mixer' in d for d in diag), diag


def test_a_broken_chain_is_refused():
    """A statement between two links runs on the threads of two roles:
    the chain's shape is held strictly."""
    b = IRBuilder(fptype=F32)
    _three(b)
    first = b.finish()[-1]
    rest = first.regions[1]
    stray = Stmt(op=Op.MARK, attrs=(('mark', 'stray'),))
    broken = type(rest)(args=rest.args, body=(stray,) + rest.body)
    body = b.finish()[:-1] + (Stmt(op=Op.IF, args=first.args,
                                   regions=(first.regions[0], broken),
                                   attrs=first.attrs),)
    diag = verify(body, strict=False)
    assert any('chain of roles is broken' in d for d in diag), diag


def test_a_role_barrier_between_two_links_is_refused():
    """A role's barrier in the second region of a link that goes on to the
    next link: no role's, and its count reaches none."""
    b = IRBuilder(fptype=F32)
    split = _three(b)
    first = b.finish()[-1]
    rest = first.regions[1]
    barrier = engines.barrier(engines.sync_object(
        Participants.ROLE, 1, split.roles[1].threads, split.ident))
    stray = type(rest)(args=rest.args, body=(barrier,) + rest.body)
    body = b.finish()[:-1] + (Stmt(op=Op.IF, args=first.args,
                                   regions=(first.regions[0], stray),
                                   attrs=first.attrs),)
    diag = verify(body, strict=False)
    assert any('no role\'s' in d for d in diag), diag


def test_a_later_roles_hazards_carry_around_the_loop():
    """The state past a split is every role's end, merged: the third role's
    reads of its own buffer, outstanding where a trip ends, meet its next
    trip's store on the role's barrier -- one barrier for that, ahead of
    the store, and one between the store and the reads."""
    b = IRBuilder(fptype=F32, arena='shrMem', context=tmr._ctx())
    split = _split((0, 1), (1, 1), (2, 2), names=['a', 'b', 'c'])
    own = b.alloc(F32, (WIDTH,), MemSpace.SHARED, hint='own', extern='own',
                  arena='block')
    out = b.alloc(F32, (WIDTH,), MemSpace.GLOBAL, hint='o', extern='m0')
    x, y = b.thread_id('x'), b.thread_id('y')
    with b.for_(0, 4, 1, hint='t', uniform=Uniformity.BLOCK):
        h = b.split_roles(split)
        with h.role('a'):
            pass
        with h.role('b'):
            pass
        with h.role('c'):
            lane = b.op('add', INDEX, x, b.op('mul', INDEX, b.op(
                'sub', INDEX, y, 2, hint='w'), 32, hint='w'), hint='c')
            b.store(own, b.const(1.0, ScalarType(F32)),
                    b.op('rem', INDEX, b.op('add', INDEX, lane, 32), WIDTH))
            b.store(out, b.load(own, lane, hint='v'), lane)
    placed = tmr._placed(b.finish())
    (loop,) = [s for s in placed if s.op == Op.FOR]
    (first,) = [s for s in loop.regions[0].body if pr.is_split(s)]
    third = pr.regions(first)[2][1]
    found = [engines.object_of(s) for s in walk_stmts(third.body)
             if s.op == Op.BARRIER]
    assert [o.participants for o in found] == [Participants.ROLE] * 2


# -- pipelines of roles over rings ---------------------------------------------- #

#: The warps of a pipeline's stages after the producer's one: two each.
STAGE_WARPS = 2


def _pipeline(stages=3, trips=5, slots=2, looped=True, release=True,
              warp_sync=True, exchange=True, cross=False, named=False):
    """A producer warp and `stages - 1` stages of two warps each, one ring
    between each two: the producer copies a trip's `WIDTH` floats of `m1`
    into ring 0's slot by one bulk copy and announces their bytes; each
    middle stage -- a mixer -- waits for its ring's slot, hands each float
    to the lane of the other warp through a buffer of its own (`exchange`,
    met on its role's barrier), writes it into the next ring's slot, and
    gives the first slot back and announces the second, each by its warps'
    first lanes behind the warp's barrier; the last stage writes its slot
    to `m0` and gives it back.  `looped`: each role a loop of its own over
    the trips; otherwise the split inside the trip loop.  The switches
    leave a step out: `release` (the first mixer never gives ring 0's slot
    back), `warp_sync` (the first mixer announces without the warp's
    barrier), `cross` (the consumer reads the first mixer's buffer).
    `named`: each ring's description names the roles on its two sides
    (`engines.RingSpec`)."""
    b = IRBuilder(fptype=F32, arena='shrMem', context=tmr._ctx())
    names = ['producer'] + [f'mixer{i}' for i in range(1, stages - 1)] + [
        'consumer']
    ranges = [(STAGE_WARPS * (stages - 1), 1)] + [
        (STAGE_WARPS * i, STAGE_WARPS) for i in range(stages - 1)]
    split = RoleSplit(tuple(Role(n, 32 * c, warps=(lo, c))
                            for n, (lo, c) in zip(names, ranges)),
                      first=pr.first_id(stages))
    # Everything shared is the block's: a role is two warps, which no
    # barrier of one warp orders.
    rings = [b.mbarrier_ring(f'p{i}', slots, full=1 if i == 0 else
                             STAGE_WARPS, empty=STAGE_WARPS,
                             leader='threadIdx.x == 0 && threadIdx.y == 0',
                             arena='block',
                             **(dict(producer=names[i],
                                     consumers=(names[i + 1],))
                                if named else {}))
             for i in range(stages - 1)]
    # The slots' windows, handed from one role to the next by the rings'
    # barriers (`handed`), which the placement leaves to them.
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

    def produce(t):
        ring = rings[0]
        slot, parity = b.ring_round(ring, t)
        at = b.op('mul', INDEX, slot, WIDTH, hint='at')
        issue = b.op('eq', BOOL, x, 0, hint='issue')
        b.ring_wait(ring, 'empty', slot,
                    b.op('sub', INDEX, 1, parity, hint='free'),
                    writes=[bufs[0]])
        tok = b.copy_bulk(bufs[0], src, barrier=RingSlot(ring, slot),
                          dst_index=[at],
                          src_index=[b.op('mul', INDEX, t, WIDTH)],
                          elems=WIDTH, predicate=issue)
        b.ring_arrive(ring, 'full', slot, issue, tx=WIDTH * 4, tokens=[tok])

    def mix(t, i):
        into, outof = rings[i], rings[i - 1]
        slot, parity = b.ring_round(into, t)
        at = b.op('mul', INDEX, slot, WIDTH, hint='at')
        lane = lane_of(f'mixer{i}')
        first = b.op('eq', BOOL, x, 0, hint='first')
        b.ring_wait(outof, 'full', slot, parity, writes=[bufs[i - 1]])
        value = b.load(bufs[i - 1], b.op('add', INDEX, at, lane), hint='v')
        if exchange:
            other = b.op('rem', INDEX, b.op('add', INDEX, lane, 32), WIDTH,
                         hint='o')
            b.store(swaps[i - 1], value, other)
            b.barrier(Participants.ROLE)
            value = b.load(swaps[i - 1], lane, hint='w')
        b.barrier(Participants.WAVE)
        if release or i > 1:
            b.ring_arrive(outof, 'empty', slot, first)
        b.ring_wait(into, 'empty', slot,
                    b.op('sub', INDEX, 1, parity, hint='free'),
                    writes=[bufs[i]])
        b.store(bufs[i], value, b.op('add', INDEX, at, lane))
        if warp_sync or i > 1:
            b.barrier(Participants.WAVE)
        b.ring_arrive(into, 'full', slot, first)

    def consume(t):
        ring = rings[-1]
        slot, parity = b.ring_round(ring, t)
        at = b.op('mul', INDEX, slot, WIDTH, hint='at')
        lane = lane_of('consumer')
        b.ring_wait(ring, 'full', slot, parity, writes=[bufs[-1]])
        value = b.load(bufs[-1], b.op('add', INDEX, at, lane), hint='v')
        if cross:
            value = b.op('add', F32, value, b.load(swaps[0], lane, hint='x'),
                         hint='v')
        b.store(out, value, b.op('add', INDEX, b.op('mul', INDEX, t, WIDTH),
                                 lane))
        b.barrier(Participants.WAVE)
        b.ring_arrive(ring, 'empty', slot, b.op('eq', BOOL, x, 0,
                                               hint='first'))

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
    return built, placed, report, split


def _expected(stages, trips, seed=3):
    """What `m0` holds: `m1`, each mixer's exchange moving a float to the
    lane 32 past its own."""
    fill = kernel_eval.Slot(seed)
    shift = 32 * (stages - 2) % WIDTH
    return [fill.read('m1', (i // WIDTH) * WIDTH + (i % WIDTH + shift)
                      % WIDTH) for i in range(trips * WIDTH)]


def _run(lines, stages, trips=5, seed=3):
    """The body on the host oracle, a block of the pipeline's warps: the
    elements of `m0` that differ from what the pipeline makes of `m1`, and
    the races."""
    keep = [line for line in lines if not re.match(r'^float m\d+\[', line)]
    src = ('void k() {\n'
           '  auto* block = reinterpret_cast<float*>(totalShrMemPtr);\n'
           + ''.join(f'  {line}\n' for line in keep) + '}')
    races = []
    mem = kernel_eval.evaluate_wave(
        src, 32, seed=seed, globals_only=True, races=races,
        mults=1 + STAGE_WARPS * (stages - 1), block=True)
    expected = _expected(stages, trips, seed)
    bad = [i for i in range(trips * WIDTH)
           if mem.get(('m0', i)) != expected[i]]
    return bad, races


def test_a_pipeline_of_three_roles_verifies_and_is_placed():
    """Every token taken once, the roles' barriers in their roles; the
    placement leaves the hand-overs to the rings and meets each mixer's
    exchange on the mixer's barrier, reporting nothing across the
    roles."""
    built, placed, report, split = _pipeline()
    assert verify(built, strict=False) == []
    assert not report
    (first,) = [s for s in placed if pr.is_split(s)]
    found = [engines.object_of(s) for s in walk_stmts((first,))
             if s.op == Op.BARRIER]
    assert Participants.BLOCK not in {o.participants for o in found}
    roles = [o for o in found if o.participants is Participants.ROLE]
    assert roles and {o.threads for o in roles} == {64}
    assert all(o.ident == split.ident for o in roles)


@pytest.mark.parametrize('stages', [3, 4])
@pytest.mark.parametrize('looped', [True, False], ids=['looped', 'trip'])
@pytest.mark.parametrize('slots,trips', [(1, 3), (2, 5), (3, 7)])
def test_every_float_through_n_roles_on_the_oracle(stages, looped, slots,
                                                   trips):
    """Three and four roles, each a loop of its own or the split inside the
    trip loop: every float arrives where the pipeline puts it, and no
    access is unordered -- the nested arms of the chain run as
    coroutines, an inner pair that cannot go on handing its wait to the
    outer one."""
    _, placed, _, _ = _pipeline(stages=stages, trips=trips, slots=slots,
                                looped=looped)
    bad, races = _run(tmr._emit(placed), stages, trips=trips)
    assert not bad and not races, (bad[:4], races[:3])


def test_the_old_lockstep_hangs_on_the_nested_arms(monkeypatch):
    """An arm whole before the other: the producer's loop waits, at its
    trip `slots`, for a release the mixer -- an arm of the inner IF, behind
    it -- has not run yet.  The device runs it; the arms as coroutines
    run it too (`test_every_float_through_n_roles_on_the_oracle`)."""
    _, placed, _, _ = _pipeline(trips=5, slots=2)
    monkeypatch.setattr(kernel_eval, 'COROUTINE_ARMS', False)
    with pytest.raises(kernel_eval.Abort, match='never return'):
        _run(tmr._emit(placed), 3)


def test_the_chain_is_written_as_nested_ifs():
    """Each link an `if` and an `else`, the role barriers' ids each
    thread's own: `(13 + (threadIdx.y >= 2) + (threadIdx.y >= 4))`."""
    _, placed, _, split = _pipeline()
    text = '\n'.join(tmr._emit(placed))
    assert split.ident == '(13 + (threadIdx.y >= 2) + (threadIdx.y >= 4))'
    assert (f'asm volatile("barrier.sync %0, %1;" :: "r"({split.ident}), '
            f'"r"(64) : "memory");') in text
    # The second link inside the first's `else`.
    assert re.search(r'^if \(threadIdx\.y >= 4\) \{$', text, re.M)
    assert re.search(r'^else \{\nif \(v\d+_mixer1\) \{$', text, re.M)


def test_a_slot_a_mixer_never_gives_back_hangs():
    _, placed, _, _ = _pipeline(release=False)
    with pytest.raises(kernel_eval.Abort, match='never return|every arm'):
        _run(tmr._emit(placed), 3)


def test_a_mixer_announcing_ahead_of_its_warps_stores_is_a_race():
    """The first lane's arrival on the next ring's full side hands over
    its own stores alone: without the warp's barrier the consumer reads
    the other lanes' unordered."""
    _, placed, _, _ = _pipeline(warp_sync=False, exchange=False)
    _, races = _run(tmr._emit(placed), 3)
    assert 'RAW' in {r.kind for r in races}, races[:3]


def test_an_access_across_two_later_roles_is_reported():
    """The consumer reads the mixer's own buffer: no barrier of either
    orders the two, and the placement says so -- between the second and
    the third role, past the chain's first link."""
    _, _, report, _ = _pipeline(cross=True)
    assert report and 'meet an earlier one' in report[0]


# -- register budgets ----------------------------------------------------------- #

def test_the_producers_moves_are_the_roles_rule():
    """The multi-element producer's budgets (`element_group.
    register_budgets`) are two roles' of the general rule: the share of
    the block's own allocation."""
    assert pr.register_budgets((('producer', 128), ('elements', 512)),
                               {'producer': 40}) == (
        96, {'producer': 40, 'elements': 104})
    assert pr.register_budgets((('producer', 128), ('elements', 384)),
                               {'producer': 40}) == (
        128, {'producer': 40, 'elements': 152})


def test_three_roles_share_what_two_give_up():
    """A producer and an issuer keep 40 and 56, the mixing warpgroups
    share the rest of the block's 128 x 640."""
    entry, budgets = pr.register_budgets(
        (('producer', 128), ('issuer', 128), ('mixer', 384)),
        {'producer': 40, 'issuer': 56})
    assert entry == 96
    assert budgets == {'producer': 40, 'issuer': 56, 'mixer': 128}
    assert 128 * 40 + 128 * 56 + 384 * 128 <= 96 * 640


@pytest.mark.parametrize('keep', [{'p': 96}, {'p': 120}])
def test_no_move_where_nothing_is_given(keep):
    assert pr.register_budgets((('p', 128), ('e', 512)), keep) is None


def test_budgets_are_whole_warpgroups_in_the_blocks_pool():
    split = _split((0, 4), (4, 4), names=['a', 'b'])
    assert pr.budget_refusal(split, {'a': 40, 'b': 152}, 96) is None
    assert 'past the' in pr.budget_refusal(split, {'a': 40, 'b': 160}, 96)
    odd = _split((0, 2), (2, 6), names=['a', 'b'])
    assert 'no whole warpgroups' in pr.budget_refusal(
        odd, {'a': 40, 'b': 104}, 96)


def test_the_moves_are_written_ahead_of_the_split():
    """Every role that gives lowers its registers, the block meets, every
    role that takes raises them -- each under its range's condition."""
    b = IRBuilder(fptype=F32)
    split = _split((8, 4), (0, 4), (4, 4), names=['producer', 'mixer',
                                                   'issuer'])
    pr.emit_budgets(b, split, {'producer': 40, 'mixer': 160, 'issuer': 56},
                    96)
    body = b.finish()
    moves = [(s.attr('change'), s.attr('count')) for s in walk_stmts(body)
             if s.op == Op.REGISTER_BUDGET]
    assert moves == [('dec', 40), ('dec', 56), ('inc', 160)]
    barriers = [i for i, s in enumerate(body) if s.op == Op.BARRIER]
    decs = [i for i, s in enumerate(body) if s.op == Op.IF and any(
        x.attr('change') == 'dec' for x in walk_stmts((s,)))]
    incs = [i for i, s in enumerate(body) if s.op == Op.IF and any(
        x.attr('change') == 'inc' for x in walk_stmts((s,)))]
    assert max(decs) < barriers[0] < min(incs)
    assert verify(body, strict=False) == []


def test_moves_past_the_pool_are_refused():
    b = IRBuilder(fptype=F32)
    split = _split((0, 4), (4, 4), names=['a', 'b'])
    with pytest.raises(IRError, match='past the'):
        pr.emit_budgets(b, split, {'a': 40, 'b': 160}, 96)
