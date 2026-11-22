# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""The ring as one object (`engines.RingSpec`, `IRBuilder.mbarrier_ring`):
N slots a producer fills and its consumers drain, a pair of barriers a slot
-- the full side's arrivals and bytes, the empty side's arrivals -- each
round of a slot a phase; who fills and who drains it, by role; the scope
its waits and arrivals order, stated on every one; and the protocol in one
place (`ring_acquire`, `ring_publish`, `ring_take`, `ring_release`), which
the three rings of the multi-element layout go through -- the producer's
input ring, the prepared operator ring and the folded star's A ring --
writing what they wrote before, byte for byte.

What is checked here: the description and its refusals; the protocol's
statements, the same the raw waits and arrivals were; the scope on every
statement, a cluster's refused where it would be spelled; the verifier
holding a ring's acts to the roles it names; and the hand-offs a pass reads
of a body (`pir.rings.handoffs`) -- on the roles' pipelines, and on the
kernels of the three rings.
"""

from __future__ import annotations

import contextlib
import io
import re
import warnings

import pytest

import test_folded_star as tfs
import test_mbarrier_ring as tmr
import test_multi_element as me
import test_multi_element_async as tma
import test_operator_ring as tor
import test_roles as tr
from tensorforge.backend.pir import engines, rings
from tensorforge.backend.pir.build import IRBuilder, RingSlot
from tensorforge.backend.pir.core import (BOOL, INDEX, IRError, MemSpace, Op,
                                          walk_stmts)
from tensorforge.backend.pir.engines import RingSpec, SyncKind, SyncScope
from tensorforge.backend.pir.passes import verify
from tensorforge.common.basic_types import Datatype

F32 = Datatype.F32


# -- the description ------------------------------------------------------------ #

def test_a_ring_carries_its_description():
    """The ring's object is a ring of its slots, and its type says who fills
    and who drains it, the arrivals of each side and the scope."""
    b = IRBuilder(fptype=F32, arena='shrMem', context=tmr._ctx())
    ring = b.mbarrier_ring('pc', 3, full=1, empty=2, leader='true',
                           producer='producer', consumers=('consumers',),
                           tx=256)
    spec = ring.spec
    assert spec == RingSpec(3, 1, 2, 'producer', ('consumers',), 256,
                            SyncKind.MBARRIER, SyncScope.BLOCK)
    assert ring.object.type.slots == 3 and ring.object.type.ring is spec
    assert engines.ring_spec(ring) is spec
    assert engines.ring_spec(ring.object) is spec


@pytest.mark.parametrize('args,why', [
    (dict(slots=0, full=1, empty=1), 'of 0 slots'),
    (dict(slots=2, full=0, empty=1), 'count 0 and 1'),
    (dict(slots=2, full=1, empty=1, tx=-4), '-4 bytes'),
    (dict(slots=2, full=1, empty=1, kind=SyncKind.IMPLICIT), 'implicit'),
    (dict(slots=2, full=1, empty=1, scope=SyncScope.WARP), 'warp scope'),
    (dict(slots=2, full=1, empty=1, producer='a', consumers=('b', 'a')),
     'fills a ring it drains')])
def test_a_ring_that_cannot_be_is_refused(args, why):
    with pytest.raises(IRError, match=why):
        RingSpec(**args)


@pytest.mark.parametrize('role,side,waits,refused', [
    ('producer', 'empty', True, False),     # acquire
    ('producer', 'full', False, False),     # publish (and the copies)
    ('consumer', 'full', True, False),      # take
    ('consumer', 'empty', False, False),    # release
    ('consumer', 'empty', True, True),      # a consumer acquiring
    ('producer', 'full', True, True),       # the producer taking
    ('producer', 'empty', False, True),     # the producer releasing
    ('consumer', 'full', False, True),      # a consumer publishing
    (None, 'full', True, True)])            # outside the roles
def test_the_acts_of_a_ring_belong_to_its_roles(role, side, waits, refused):
    spec = RingSpec(2, 1, 1, 'producer', ('consumer',))
    assert (spec.role_refusal(role, side, waits) is not None) is refused
    assert RingSpec(2, 1, 1).role_refusal(role, side, waits) is None


# -- the protocol --------------------------------------------------------------- #

def _ring(**spec):
    b = IRBuilder(fptype=F32, arena='shrMem', context=tmr._ctx())
    ring = b.mbarrier_ring('pc', 2, full=1, empty=2, leader='true', **spec)
    return b, ring


def _shape(s):
    return (s.op, s.attrs, tuple(a.id if hasattr(a, 'id') else a
                                 for a in s.args))


def test_the_protocol_is_the_waits_and_arrivals_it_names():
    """`ring_acquire` is the wait on the empty side for the other parity
    -- an op `1 - parity` named `free` ahead of it, as the rings built it
    by hand -- `ring_publish` the arrival on the full side with its bytes
    and tokens, `ring_take` the wait on the full side, `ring_release` the
    arrival on the empty side or tcgen05's commit onto it."""
    b, ring = _ring()
    slot = b.op('rem', INDEX, b.thread_id('y'), 2, hint='s')
    parity = b.op('rem', INDEX, b.thread_id('y'), 4, hint='p')
    issue = b.op('eq', BOOL, b.thread_id('x'), 0, hint='issue')
    acquire = b.ring_acquire(ring, slot, parity)
    free = b.finish()[-2]
    assert free.op == 'sub' and free.args == (1, parity) and \
        free.target[0].hint == 'free'
    assert _shape(acquire) == _shape(engines.ring_wait(
        ring.object, 'empty', slot, free.target[0]))
    assert _shape(b.ring_publish(ring, slot, issue, tx=128)) == _shape(
        engines.ring_arrive(ring.object, 'full', slot, issue, tx=128))
    assert _shape(b.ring_take(ring, slot, parity)) == _shape(
        engines.ring_wait(ring.object, 'full', slot, parity))
    assert _shape(b.ring_release(ring, slot, issue)) == _shape(
        engines.ring_arrive(ring.object, 'empty', slot, issue))
    commit = b.ring_release(ring, slot, issue, engine='tcgen05')
    assert _shape(commit) == _shape(engines.commit_group(
        'tcgen05', ring.object, issue, slot))
    assert commit.attr('side') == 'empty'


def test_every_wait_and_arrival_states_its_scope():
    """The block's, which the lowering spells as it always did; the op
    definitions take it (`engines.REGISTRY`), so `verify_engine` checks
    it."""
    b, ring = _ring()
    issue = b.op('eq', BOOL, b.thread_id('x'), 0, hint='issue')
    made = [b.ring_acquire(ring, 0, 0), b.ring_publish(ring, 0, issue),
            b.ring_take(ring, 0, 0), b.ring_release(ring, 0, issue),
            b.ring_release(ring, 0, issue, engine='tcgen05')]
    for s in made:
        assert s.attr('scope') is SyncScope.BLOCK, s
        assert engines.verify_engine(s) == [], s
    lines = [engines.lower(s, tmr._ctx('sm_100a').target, name=str).lines
             for s in made]
    assert lines[2] == (f'tensorforge::mbarrierWait(&{ring.object}[0], 0);',)


def test_a_cluster_scope_has_no_spelling_yet():
    """A ring at a cluster's scope is a description the IR holds -- every
    statement on it says so -- and no lowering spells: clusters come
    later, and a wait at the cluster's scope costs every wait an L1
    invalidation on the hardware (wave 3b)."""
    b, ring = _ring(scope=SyncScope.CLUSTER)
    issue = b.op('eq', BOOL, b.thread_id('x'), 0, hint='issue')
    for s in (b.ring_acquire(ring, 0, 0), b.ring_publish(ring, 0, issue),
              b.ring_release(ring, 0, issue, engine='tcgen05')):
        assert s.attr('scope') is SyncScope.CLUSTER
        with pytest.raises(IRError, match='until clusters are built'):
            engines.lower(s, tmr._ctx('sm_100a').target, name=str)


# -- the verifier --------------------------------------------------------------- #

def test_a_ring_with_roles_verifies_where_its_roles_act():
    built, _, _, _ = tr._pipeline(stages=3, named=True)
    assert verify(built, strict=False) == []


def test_a_consumer_waiting_on_the_empty_side_is_refused():
    """The consumers waiting for the empty side's phase before the full
    side's: the producer completes that one, waiting for it themselves is
    a wait for a phase their own role completes."""
    b = IRBuilder(fptype=F32, arena='shrMem', context=tmr._ctx())
    split = tr._split((1, 1), (0, 1), names=['producer', 'consumer'])
    ring = b.mbarrier_ring('pc', 2, full=1, empty=1, leader='true',
                           producer='producer', consumers=('consumer',))
    h = b.split_roles(split)
    with h.role('producer'):
        b.ring_acquire(ring, 0, 0)
    with h.role('consumer'):
        b.ring_acquire(ring, 0, 0)
    diag = verify(b.finish(), strict=False)
    assert any("the producer's (producer), not consumer's" in d
               for d in diag), diag


def test_a_ring_act_outside_its_roles_is_refused():
    b = IRBuilder(fptype=F32, arena='shrMem', context=tmr._ctx())
    ring = b.mbarrier_ring('pc', 2, full=1, empty=1, leader='true',
                           producer='producer', consumers=('consumer',))
    b.ring_take(ring, 0, 0)
    diag = verify(b.finish(), strict=False)
    assert any("not the block's" in d for d in diag), diag


def test_a_publish_of_other_bytes_than_a_round_is_refused():
    b = IRBuilder(fptype=F32, arena='shrMem', context=tmr._ctx())
    split = tr._split((1, 1), (0, 1), names=['producer', 'consumer'])
    ring = b.mbarrier_ring('pc', 2, full=1, empty=1, leader='true',
                           producer='producer', consumers=('consumer',),
                           tx=256)
    issue = b.op('eq', BOOL, b.thread_id('x'), 0, hint='issue')
    h = b.split_roles(split)
    with h.role('producer'):
        b.ring_publish(ring, 0, issue, tx=128)
    with h.role('consumer'):
        b.ring_release(ring, 0, issue)
    diag = verify(b.finish(), strict=False)
    assert any('rounds are 256 bytes' in d for d in diag), diag


# -- the hand-offs -------------------------------------------------------------- #

def test_the_handoffs_of_a_pipeline_by_role():
    """Three roles over two rings: ring 0 acquired, copied into and
    published by the producer, taken and released by the mixer; ring 1
    filled by the mixer and drained by the consumer -- each with its
    bytes."""
    built, _, _, _ = tr._pipeline(stages=3, named=True)
    uses = sorted(rings.handoffs(built).values(),
                  key=lambda u: u.ring.hint)
    assert [u.ring.hint for u in uses] == ['p0_ring', 'p1_ring']
    first, second = uses
    assert first.setup is not None and first.spec.producer == 'producer'
    assert first.roles('acquire') == ('producer',)
    assert first.roles('copy') == ('producer',)
    assert [h.bytes for h in first.acts('copy')] == [tr.WIDTH * 4]
    assert [h.bytes for h in first.acts('publish')] == [tr.WIDTH * 4]
    assert first.roles('take') == first.roles('release') == ('mixer1',)
    assert second.roles('acquire') == second.roles('publish') == (
        'mixer1',)
    assert second.roles('take') == second.roles('release') == (
        'consumer',)
    assert not second.acts('copy')


def _bodies(shape='volume', order=6, arch='sm_90a', **options):
    from tensorforge.generators.generator import Generator
    gen = Generator(me._descrs(shape, order, 's'),
                    me._context(arch, 's', multi_element='force',
                                matrix_engine='auto', **options))
    with warnings.catch_warnings(), \
            contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter('ignore')
        gen.generate()
    return [section.body for section in gen.built._sections]


def _uses(bodies):
    out = {}
    for body in bodies:
        for use in rings.handoffs(body).values():
            out[use.ring.hint] = use
    return out


@pytest.mark.parametrize('arch', ['sm_90a', 'sm_100a'])
def test_the_producers_ring_hands_over_between_its_two_roles(arch):
    """The multi-element producer's input ring: the producer's warp
    acquires each slot, copies into it and publishes its bytes; the
    element warps take and release it -- its description names both.  A
    copy is a column of an element's box, one a lane of the producer's
    warp, an element a lane: the bytes the publish announces are the
    columns' times the group's elements."""
    from tensorforge.generators import element_group as eg
    uses = _uses(_bodies(arch=arch, element_producer=True))
    use = uses['me_ring']
    assert (use.spec.producer, use.spec.consumers) == ('producer',
                                                       ('elements',))
    assert use.roles('acquire') == use.roles('copy') == use.roles(
        'publish') == ('producer',)
    assert use.roles('take') == use.roles('release') == ('elements',)
    group = eg.plan(me._descrs('volume', 6, 's'), me._context(
        arch, 's', multi_element='force', matrix_engine='auto',
        element_producer=True))
    (publish,) = use.acts('publish')
    assert publish.bytes == group.elements * sum(
        h.bytes for h in use.acts('copy'))


@pytest.mark.parametrize('arch,engine', [('sm_90a', None),
                                         ('sm_100a', 'tcgen05')])
def test_the_operator_ring_stays_inside_the_block(arch, engine):
    """The prepared operator ring at o8: the block's first thread fills it
    and the products drain it, in no role -- on tcgen05 every release is
    the products' commit."""
    uses = _uses(_bodies(order=8, arch=arch, matrix_ring='prepared'))
    (use,) = [u for name, u in uses.items() if name.startswith('bk')]
    assert use.spec.producer is None and use.spec.consumers == ()
    assert {h.role for h in use.handoffs} == {None}
    assert use.acts('acquire') and use.acts('copy') and use.acts('take')
    releases = use.acts('release')
    assert releases and all(
        (h.stmt.op == Op.COMMIT_GROUP) == (engine == 'tcgen05')
        for h in releases)


def test_the_folded_stars_ring_hands_a_to_the_products():
    """tcgen05's folded star: the bands acquire a slot of A's ring and
    publish it, their 128 threads a phase; the issuing thread takes it and
    its products' commit releases it."""
    uses = _uses(_bodies(order=8, arch='sm_100a', matrix_ring='prepared',
                         matrix_star='engine'))
    (use,) = [u for name, u in uses.items() if name.startswith('ta')]
    assert (use.spec.full, use.spec.empty) == (128, 1)
    assert use.acts('acquire') and use.acts('publish') and use.acts('take')
    assert all(h.stmt.op == Op.COMMIT_GROUP for h in use.acts('release'))


@pytest.mark.parametrize('options', [
    dict(element_producer=True), dict(order=8, matrix_ring='prepared'),
    dict(order=8, matrix_ring='prepared', matrix_star='engine')],
    ids=['producer', 'operator', 'folded'])
@pytest.mark.parametrize('arch', ['sm_90a', 'sm_100a'])
def test_the_generated_rings_verify(arch, options):
    """The three rings as the multi-element layout builds them, through
    every pass: no act of a ring where its roles do not stand, no role's
    barrier out of its role, no chain broken."""
    found = []
    for body in _bodies(arch=arch, **options):
        found += [d for d in verify(body, strict=False)
                  if "slot's" in d or 'role' in d or 'chain' in d]
    assert not found, found[:3]


# -- the release point ------------------------------------------------------------ #

def test_a_ring_holds_its_slots_for_a_group_by_default():
    """Where the consumers give a slot back (`RingSpec.hold`): behind their
    own group by default, behind more where the ring says, at most its
    slots."""
    assert RingSpec(2, 1, 1).hold == 1
    b = IRBuilder(fptype=F32, arena='shrMem', context=tmr._ctx())
    ring = b.mbarrier_ring('pc', 3, full=1, empty=2, leader='true', hold=2)
    assert ring.spec.hold == ring.hold == 2
    with pytest.raises(IRError, match='for 0 of their groups'):
        RingSpec(2, 1, 1, hold=0)
    with pytest.raises(IRError, match='for 3 of their groups; 1 to its 2'):
        RingSpec(2, 1, 1, hold=3)


@pytest.mark.parametrize('arch,hold', [('sm_90a', 2), ('sm_100a', 1)])
def test_each_ring_holds_what_it_did(arch, hold):
    """The rings as they were: the producer's input ring and the folded
    star's A ring give a slot back behind their own group -- the element
    warps' trip, the block's products' commit; the operator ring behind
    the next chunk's issue on wgmma (`wait_group 1`), at once on tcgen05
    (the commit)."""
    uses = _uses(_bodies(arch=arch, element_producer=True))
    assert uses['me_ring'].spec.hold == 1
    uses = _uses(_bodies(order=8, arch=arch, matrix_ring='prepared',
                         matrix_star='engine'))
    holds = {name[:2]: u.spec.hold for name, u in uses.items()}
    assert holds['bk'] == hold
    if arch == 'sm_100a':
        assert holds['ta'] == 1


def _events(src, ring='bk3'):
    """The operator ring's acts in the text's order: `('fill', chunk)` --
    the chunk by where its copy reads the table --, `('take', chunk)`,
    `('wait', pending)` for wgmma's waits, and `('release', chunk)`, the
    chunk the slot was taken for (a release names the take's slot)."""
    offsets = sorted({int(m) for m in re.findall(
        rf'bulkCopyOn\(&{ring}_ring\[[^]]*\], [^,]*, &tf_bk_\w+\[(\d+)\]',
        src)})
    taken, out = {}, []
    for line in src.splitlines():
        fill = re.search(rf'bulkCopyOn\(&{ring}_ring\[[^]]*\], [^,]*, '
                         rf'&tf_bk_\w+\[(\d+)\]', line)
        take = re.search(rf'mbarrierWait\(&{ring}_ring\[(\w+)\],', line)
        wait = re.search(r'wgmmaWait<(\d)>', line)
        release = re.search(rf'mbarrierArrive\(&{ring}_ring\[\d+ \+ '
                            rf'\((\w+)\)\]', line)
        if fill:
            out.append(('fill', offsets.index(int(fill.group(1)))))
        elif take:
            taken[take.group(1)] = len(taken)
            out.append(('take', taken[take.group(1)]))
        elif wait:
            out.append(('wait', int(wait.group(1))))
        elif release:
            out.append(('release', taken[release.group(1)]))
    return out


def _held(src, chunks, slots, hold):
    """`src`'s operator ring on wgmma holds a slot for `hold` chunks: behind
    chunk `i`'s products the wait leaves `hold - 1` chunks in flight and
    gives back the slot of the chunk it retired, `i - hold + 1`, and the
    first thread fills the chunk `slots` past that one into it at once; at
    the trip's head it fills the first `slots - hold + 1`, and behind the
    last chunk the wait for all of them gives back the slots still held."""
    (ring,) = set(re.findall(r'\b(bk\d+)_ring\b', src))
    events = _events(src, ring)
    assert 1 + max(c for act, c in events if act == 'take') == chunks
    lead = slots - hold + 1
    expected = [('fill', j) for j in range(min(lead, chunks))]
    for i in range(chunks):
        expected.append(('take', i))
        if i >= hold - 1:
            expected += [('wait', hold - 1), ('release', i - hold + 1)]
        if i + lead < chunks:
            expected.append(('fill', i + lead))
    if hold > 1:
        expected.append(('wait', 0))
        expected += [('release', i) for i in range(chunks - hold + 1,
                                                   chunks)]
    assert events == expected


@pytest.mark.parametrize('hold', [None, 1, 3])
def test_wgmma_gives_a_slot_back_behind_its_hold(hold):
    """The prepared ring on wgmma at o8, four slots and eighteen chunks a
    trip (`_held`): by default the wait behind a chunk retires the one
    before (`wait_group 1`), with a hold of 1 its own (`wait_group 0`)."""
    options = {} if hold is None else {'matrix_ring_hold': hold}
    src = tma._kernel('sm_90a', 'volume', 8, 's', matrix_ring='prepared',
                      **options).get_kernel()
    _held(src, 18, 4, hold or 2)


@pytest.mark.parametrize('hold', [None, 1])
def test_wgmmas_folded_star_gives_b_back_behind_its_hold(hold):
    """wgmma's folded star at o8, a block of every product a chunk: the
    wait behind a block gives B's slot back as the ring holds it -- and A's
    with it (`multi_element._wgmma_star`)."""
    options = {} if hold is None else {'matrix_ring_hold': hold}
    src = tma._kernel('sm_90a', 'volume', 8, 's', matrix_ring='prepared',
                      matrix_star='engine', **options).get_kernel()
    _held(src, 11, 4, hold or 2)


@pytest.mark.parametrize('slots,hold', [(2, 1), (4, 1), (4, 3), (3, 3)])
def test_a_ring_held_otherwise_on_the_oracle(slots, hold, monkeypatch):
    """wgmma's prepared ring over two trips of a group with masked
    elements, held for its own chunk alone and for up to every slot: against
    NumPy, without a race -- a slot written while products still read it --
    and without a hang (`test_operator_ring`)."""
    tor._prepared_two_trips('volume', 'sm_90a', monkeypatch, True,
                            matrix_ring_slots=slots, matrix_ring_hold=hold)


def test_the_folded_star_held_for_its_own_block_on_the_oracle(monkeypatch):
    """wgmma's folded star with B's ring held for its own block alone: the
    wait behind a block retires it, its slot of B and of A given back at
    once (`multi_element._wgmma_star`)."""
    tfs._two_trips_at(6, 'volume', 'sm_90a', monkeypatch, True,
                      matrix_star='engine', matrix_ring='prepared',
                      matrix_ring_rows=8, matrix_ring_hold=1, **tma.AUTO)


@pytest.mark.parametrize('arch,options,why', [
    ('sm_100a', {}, "tcgen05 commits a chunk's products"),
    ('sm_90a', {'matrix_ring_slots': 2},
     "a hold of 3 chunks is past the ring's 2 slots"),
    ('sm_90a', {'matrix_star': 'engine'}, "the folded star's A has 2 slots")])
def test_a_hold_a_ring_cannot_take_is_declined(arch, options, why):
    """A hold of 3 where a ring cannot take it: every ring keeps its
    engine's own, and the kernel's comment says why."""
    from tensorforge.generators.element_group import RING_HOLDS
    asked = dict(matrix_ring='prepared', matrix_ring_hold=3, **options)
    src = tma._kernel(arch, 'volume', 8, 's', **asked).get_kernel()
    assert re.search(rf'matrix_ring_hold=3 declined: {re.escape(why)}', src)
    assert 'a slot held for' not in src
    uses = _uses(_bodies(order=8, arch=arch, **asked))
    engine = 'wgmma' if arch == 'sm_90a' else 'tcgen05'
    assert {u.spec.hold for name, u in uses.items()
            if name.startswith('bk')} == {RING_HOLDS[engine]}


def test_a_hold_taken_is_in_the_comment():
    src = tma._kernel('sm_90a', 'volume', 8, 's', matrix_ring='prepared',
                      matrix_ring_hold=1).get_kernel()
    assert '18 chunks a trip, a slot held for 1 chunk;' in src
    assert 'declined' not in src


def test_the_option_reads_a_hold_or_auto():
    from tensorforge.common.options import parse_matrix_ring_hold
    assert parse_matrix_ring_hold('auto') is None
    assert parse_matrix_ring_hold(' 3 ') == 3
    for text in ('0', '9'):
        with pytest.raises(ValueError, match='matrix_ring_hold is one of'):
            parse_matrix_ring_hold(text)
