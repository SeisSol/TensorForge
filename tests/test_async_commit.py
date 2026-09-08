# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Where a group closes, and what a wait counts.

`cp.async.commit_group` and `__pipeline_commit` are per *thread*.  A commit
inside a lane predicate or a hop loop therefore makes the number of groups in
flight a property of which lanes ran, while the wait that counts them is one
statement for all of them.  Nothing showed for a long time because every wait
in the corpus was a full drain, and a drain does not care how many groups it
retires -- which is exactly why this is pinned here rather than left to the
snapshots, where it would go on not showing.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tensorforge.backend.pir import walk
from tensorforge.backend.pir.asyncmem import (check_commits, place_commits,
                                              schedule_async, strip_commits)
from tensorforge.backend.pir.build import IRBuilder
from tensorforge.backend.pir.core import MemSpace, Op
from tensorforge.common.basic_types import Datatype

SNAPSHOTS = Path(__file__).parent / 'snapshots'


def _commits(body):
    return [(s, p) for s, p in walk(body) if s.op == Op.COMMIT_ASYNC]


def _waits(body):
    return [(s, p) for s, p in walk(body) if s.op == Op.WAIT]


def _scratch(size=4096):
    return IRBuilder(fptype=Datatype.F32, scratch=('tempShrMem', size))


# --------------------------------------------------------------------------- #
# Placement
# --------------------------------------------------------------------------- #

def test_a_hop_loop_closes_once_outside_itself():
    """The 16 copies of a split transfer are one group, not sixteen.

    A commit per copy is not merely wasteful.  The group is the unit a wait
    counts, so a transfer that occupies sixteen of them cannot be left in
    flight behind another one without the count saying something no reader
    would guess.
    """
    b = _scratch()
    dst = b.alloc(Datatype.F32, (512,), MemSpace.SHARED, hint='s')
    glb = b.alloc(Datatype.F32, (4096,), MemSpace.GLOBAL, hint='g')
    lane = b.thread_id('x')
    with b.for_(0, 15, 1) as f:
        tok = b.copy_async(dst, glb, dst_index=(f.induction,),
                           src_index=(f.induction,))
    b.wait(tok)
    body = place_commits(b.finish())

    commits = _commits(body)
    assert len(commits) == 1
    stmt, parents = commits[0]
    assert parents == (), 'the commit stayed inside the hop loop'
    assert not check_commits(body)


def test_a_predicated_tail_hop_commits_for_every_lane():
    """The lane that copies nothing still closes a group.

    An empty group is legal and retires immediately, which is what makes this
    the cheap fix: the copy may keep its predicate, and only the commit has to
    leave it.
    """
    b = _scratch()
    dst = b.alloc(Datatype.F32, (128,), MemSpace.SHARED, hint='s')
    glb = b.alloc(Datatype.F32, (4096,), MemSpace.GLOBAL, hint='g')
    lane = b.thread_id('x')
    with b.if_(b.op('lt', None, lane, 24, hint='tail')):
        tok = b.copy_async(dst, glb, dst_index=(lane,), src_index=(lane,))
    b.wait(tok)
    body = place_commits(b.finish())

    commits = _commits(body)
    assert len(commits) == 1
    _, parents = commits[0]
    assert parents == (), 'the commit stayed under the lane predicate'


def test_the_commit_follows_the_wait_into_a_guard():
    """Out of the *lane* predicate, but into the per-element flag guard.

    The rule is one rule -- the commit goes where the wait is -- and this is
    the direction that shows it is not simply "hoist as far as possible".  A
    commit outside a guard whose wait is inside would be just as wrong, one
    group the other way.
    """
    b = _scratch()
    dst = b.alloc(Datatype.F32, (128,), MemSpace.SHARED, hint='s')
    glb = b.alloc(Datatype.F32, (4096,), MemSpace.GLOBAL, hint='g')
    lane = b.thread_id('x')
    allowed = b.op('lt', None, b.thread_id('y'), 1, hint='allowed')
    with b.if_(allowed):
        tok = b.copy_async(dst, glb, dst_index=(lane,), src_index=(lane,))
        b.wait(tok)
    body = place_commits(b.finish())

    commits = _commits(body)
    assert len(commits) == 1
    _, parents = commits[0]
    assert len(parents) == 1 and parents[0].op is Op.IF
    assert not check_commits(body)


def test_the_commit_closes_where_it_was_issued_not_before_the_wait():
    """Two transfers keep the order they were issued in.

    Putting each commit immediately before its own wait would be uniform and
    still wrong: the groups would then close in the order the *waits* appear,
    and a schedule that leaves the first transfer in flight past the second
    would count them the other way round.
    """
    b = _scratch()
    d0 = b.alloc(Datatype.F32, (128,), MemSpace.SHARED, hint='s')
    d1 = b.alloc(Datatype.F32, (128,), MemSpace.SHARED, hint='t')
    glb = b.alloc(Datatype.F32, (4096,), MemSpace.GLOBAL, hint='g')
    lane = b.thread_id('x')
    first = b.copy_async(d0, glb, dst_index=(lane,), src_index=(lane,))
    b.wait(first)
    second = b.copy_async(d1, glb, dst_index=(lane,), src_index=(lane,))
    b.wait(second)
    body = place_commits(b.finish())

    order = [s.op for s in body if s.op in (Op.COPY_ASYNC, Op.COMMIT_ASYNC,
                                            Op.WAIT)]
    assert order == [Op.COPY_ASYNC, Op.COMMIT_ASYNC, Op.WAIT,
                     Op.COPY_ASYNC, Op.COMMIT_ASYNC, Op.WAIT]


# --------------------------------------------------------------------------- #
# Counting
# --------------------------------------------------------------------------- #

def _two_groups():
    """Two transfers issued back to back, the first waited first."""
    b = _scratch()
    d0 = b.alloc(Datatype.F32, (128,), MemSpace.SHARED, hint='s')
    d1 = b.alloc(Datatype.F32, (128,), MemSpace.SHARED, hint='t')
    glb = b.alloc(Datatype.F32, (4096,), MemSpace.GLOBAL, hint='g')
    lane = b.thread_id('x')
    a = b.copy_async(d0, glb, dst_index=(lane,), src_index=(lane,))
    a2 = b.copy_async(d0, glb, dst_index=(lane,), src_index=(lane,))
    c = b.copy_async(d1, glb, dst_index=(lane,), src_index=(lane,))
    b.wait(a2, a)
    b.wait(c)
    return schedule_async(b.finish())


def test_prior_counts_groups_and_not_copies():
    """Three copies, two groups: the first wait leaves one group, not one copy.

    This is the half of the change that is not about lanes.  `prior` used to
    count tokens, which agreed with the hardware only because a commit sat
    behind every copy; with a group per transfer the two would drift, and the
    wait would leave a number of things in flight that nobody counts.
    """
    body, diag = _two_groups()
    assert not diag
    priors = [s.attr('prior') for s, _ in _waits(body)]
    assert priors == [1, 0]


def test_a_conditional_group_is_left_out_of_the_operation_count():
    """`prior_unified` under-counts rather than guessing high.

    Too large an N waits for *fewer* retirements, so a bound may not be used
    as a count.  A group whose copies sit under a predicate holds at most as
    many operations as it names, so the per-operation total leaves it out --
    which waits longer and is the safe direction.
    """
    b = _scratch()
    d0 = b.alloc(Datatype.F32, (128,), MemSpace.SHARED, hint='s')
    d1 = b.alloc(Datatype.F32, (128,), MemSpace.SHARED, hint='t')
    glb = b.alloc(Datatype.F32, (4096,), MemSpace.GLOBAL, hint='g')
    lane = b.thread_id('x')
    a = b.copy_async(d0, glb, dst_index=(lane,), src_index=(lane,))
    with b.if_(b.op('lt', None, lane, 24, hint='tail')):
        c = b.copy_async(d1, glb, dst_index=(lane,), src_index=(lane,))
    b.wait(a)
    b.wait(c)
    body, _ = schedule_async(b.finish())

    first = _waits(body)[0][0]
    assert first.attr('prior') == 1, 'the guarded group is still a group'
    assert first.attr('prior_unified') == 0, (
        'a group that may hold no operation was counted as holding one')


def test_scheduling_twice_changes_nothing():
    """`wrap_prefetch` runs between two schedules, so this has to hold.

    The commits describe one schedule.  Re-deriving them has to start by
    dropping the ones already there, or the second run would count the first
    run's groups as statements of its own.
    """
    once, _ = _two_groups()
    twice, _ = schedule_async(once)
    assert twice == once
    assert len(_commits(strip_commits(twice))) == 0


def test_a_body_with_no_copies_gains_no_commit():
    b = _scratch()
    glb = b.alloc(Datatype.F32, (4096,), MemSpace.GLOBAL, hint='g')
    lane = b.thread_id('x')
    tok = b.load_async(glb, lane, hint='r')
    b.wait(tok)
    body, diag = schedule_async(b.finish())
    assert not diag
    assert not _commits(body)


# --------------------------------------------------------------------------- #
# The generated source
# --------------------------------------------------------------------------- #

_HEAD = re.compile(r'^\}?\s*(if|for|while|else|do)\b')


def _blocks_around(path: Path, needle: str):
    """Every `needle` line with the block headers enclosing it.

    Braces are counted left to right rather than closes-then-opens, and
    comments are cut first.  Both matter: `float r0[8]{};` closes and opens on
    one line, and taking the close first pops the guard the declaration is
    *inside*; the operand comments carry `{0..9}`.  Either mistake loses the
    nesting a few hundred lines in, silently, and in the direction that makes
    this pass.
    """
    stack, found = [], []
    for line in path.read_text().splitlines():
        cut = line.find('//')
        s = (line[:cut] if cut >= 0 else line).strip()
        if not s or s.startswith('#'):
            continue
        if needle in s:
            found.append(tuple(h for h in stack if _HEAD.match(h)))
        for ch in s:
            if ch == '{':
                stack.append(s)
            elif ch == '}' and stack:
                stack.pop()
    return found


@pytest.mark.parametrize('path', sorted(SNAPSHOTS.glob('*.cuda.cpp')),
                         ids=lambda p: p.stem)
def test_no_commit_is_more_conditional_than_its_wait(path):
    """The property, on the text the compiler sees.

    Checked against the *last* wait rather than pairing them up: a commit is
    only ever counted by a wait that follows it, so every enclosing block of
    the commit has to enclose one of those too.
    """
    commits = _blocks_around(path, '__pipeline_commit')
    waits = _blocks_around(path, '__pipeline_wait_prior')
    if not commits:
        pytest.skip('no async copies in this kernel')
    assert waits, 'copies are committed and never waited for'
    for blocks in commits:
        assert any(blocks == w[:len(blocks)] for w in waits), (
            f'{path.name}: a commit sits inside {blocks[-1]!r}, which no wait '
            f'that counts it is inside')
