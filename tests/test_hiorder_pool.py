# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""The high-order tile's mixer pool on tcgen05 (`Options.matrix_pool`,
`element_group.TileRows.pool`, `compute.multi_element._Pool`): where the
plan takes it and why it declines it, the kernel's text -- the counters,
the two pool regions, the mixers' barrier and the reset behind it -- the
host oracle under several schedules against the star bands, and o8's
build.  The protocol itself, every interleaving at a small size, its rules
dropped one at a time: tests/test_pool_model.py.

The plan refuses the pool everywhere since P3 (`element_group.
POOL_REFUSED`, tests/test_pool_refusal.py); these tests let it be, so that
its implementation stays tested for a pool that pays."""

from __future__ import annotations

import re

import pytest

import test_hiorder_tile as hi
import test_multi_element as me
import test_multi_element_async as tma
from tensorforge.generators import element_group as eg

#: The pool asked for.
POOL = {'matrix_pool': 'on'}


@pytest.fixture(autouse=True)
def _the_pool_let_be(monkeypatch):
    """The plan's refusal of the pool since P3 lifted (`element_group.
    POOL_REFUSED`): the implementation under test."""
    monkeypatch.setattr(eg, 'POOL_REFUSED', False)


def _group(order=8, arch='sm_100a', **options):
    return hi._group(order, arch, **options)


def _notes(group, arch='sm_100a', **options):
    return eg.tile_notes(group, me._context(
        arch, 's', multi_element='force', **{**hi.HIGH, **options}))


def _text(order=8, **options):
    return hi._kernel(order, 'sm_100a', **{**POOL, **options}).get_kernel()


# -- the plan ------------------------------------------------------------------- #

def test_the_option_defaults_to_the_bands():
    from tensorforge.common.options import parse_matrix_pool, registry
    assert registry()['matrix_pool'].default == 'off'
    assert parse_matrix_pool(' On ') == 'on'
    with pytest.raises(ValueError):
        parse_matrix_pool('auto')
    group = _group(8)
    assert group.section.tile.pool == 'off'
    assert not any('pool' in note for note in _notes(group))


@pytest.mark.parametrize('order', [4, 6, 8, 10])
def test_the_pool_the_options_ask_for(order):
    """tcgen05's tile at o4 to o10: the pool, A's four slots at least a
    quarter's three mixers, and the kernel's note saying so."""
    group = _group(order, **POOL)
    tile = group.section.tile
    assert (tile.pool, tile.aslots, eg.pool_mixers(tile.bands)) == ('on', 4,
                                                                   3)
    assert any(note.startswith('A mixed in a pool of 12 warps')
               for note in _notes(group, **POOL))


#: Where the plan declines the pool: the target, the order, the options
#: besides it, and the note's reason.
DECLINED = [
    ('sm_90a', 8, {}, 'wgmma\'s warpgroups each fill their own A'),
    ('sm_100a', 8, {'tile_schedule': 'clc'},
     'the tiles go by claims, a trip\'s tile no stride says'),
    ('sm_100a', 8, {'matrix_groups': 'unrolled'},
     'the group\'s blocks written out, a block its band\'s by number'),
    ('sm_100a', 8, {'matrix_handoffs': 'one'},
     'one ring for A and B, its full side the bands\' and the operator '
     'warp\'s'),
    ('sm_100a', 8, {'matrix_bands': 4},
     'A\'s ring has 4 slots, fewer than a quarter\'s 5 mixers: a mixer\'s '
     'wait for a slot would pass on the parity of a phase two short of it'),
    ('sm_100a', 12, {},
     'A\'s ring has 2 slots, fewer than a quarter\'s 3 mixers: a mixer\'s '
     'wait for a slot would pass on the parity of a phase two short of it')]


@pytest.mark.parametrize('arch,order,options,why', DECLINED)
def test_the_pool_declined(arch, order, options, why):
    """Declined on wgmma, where the tiles go by claims, with the group's
    blocks written out, with one ring for A and B, and where A's ring has
    fewer slots than a quarter has mixers -- four bands at o8, o12's two
    slots: the bands' tile, the note saying why, and the kernel the bands'
    but for its name, its options' line and that note."""
    group = _group(order, arch, **POOL, **options)
    assert group.section.tile.pool == 'off'
    assert (f'matrix_pool=on declined: {why}'
            in _notes(group, arch, **POOL, **options))

    def lines(src):
        return [line for line in re.sub(r'kernel_kernel_[0-9a-f]+', 'k',
                                        src).splitlines()
                if '// options: ' not in line
                and 'matrix_pool=on declined' not in line]
    assert lines(hi._kernel(order, arch, **POOL, **options).get_kernel()) \
        == lines(hi._kernel(order, arch, **options).get_kernel())


def test_a_quarter_s_mixers_on_as_many_slots():
    """A's slots as many as a quarter has mixers suffice -- the model's
    pool on two slots for a band warp and an epilogue warp a quarter
    (tests/test_pool_model.py) -- one fewer does not."""
    for bands in (1, 2, 3):
        mixers = eg.pool_mixers(bands)
        assert eg.pool_refusal('tcgen05', 'static', 'loop', 'two', mixers,
                               bands) is None
        assert eg.pool_refusal('tcgen05', 'static', 'loop', 'two',
                               mixers - 1, bands)


@pytest.mark.parametrize('bands,more', [(2, 0), (3, 1)])
def test_the_pool_s_shared_memory(bands, more):
    """o8's pool takes `POOL_BYTES` more than the bands -- the eight
    counters and the mixers' barrier, 48 bytes -- and with three bands A's
    ring's fourth slot's pair of mbarriers besides: the bands' ring as many
    slots a band as every band has, the pool's every slot the mixers'."""
    off = _group(8, matrix_bands=bands)
    on = _group(8, matrix_bands=bands, **POOL)
    assert on.elements == off.elements
    assert (on.section.tile.aslots, off.section.tile.aslots) == (4,
                                                                 4 - more)
    assert eg.POOL_BYTES == 4 * 2 * 4 + 2 * 8
    assert (eg.shared_bytes(on.section, on.elements)
            - eg.shared_bytes(off.section, off.elements)) \
        == eg.POOL_BYTES + 2 * 8 * more


# -- the text ------------------------------------------------------------------- #

def _ring_slots(src, ring):
    """How many slots the kernel's setup gives `ring`'s barriers."""
    return [int(n) for n in re.findall(
        r'for \(int r = 0; r < (\d+); \+\+r\) \{\s*tensorforge::'
        r'mbarrierInit\(&' + ring + r'\[r\]', src)]


@pytest.mark.parametrize('chunks', [32, 'whole'])
def test_the_pool_s_text(chunks):
    """o8 on tcgen05, I in chunks and whole: eight counters of shared
    memory the block's first thread zeroes in the setup, behind the block's
    barrier; the mixers' barrier a slot whose phase the twelve mixers'
    elected arrivals complete, and the input ring's slots -- and I's
    chunks' -- given back by the twelve, where the bands give them back
    eight; two pool regions, the star bands' and the epilogue's, each
    drawing of the counter of its warp's quarter and the trip's parity
    ahead of its blocks and behind each; then in both roles a warp's
    arrival and the wait for the trip's phase -- in the star role the first
    band warp's first lane resetting the trip's four counters behind it,
    the only stores to them besides the setup's, the epilogue waiting for
    D right behind it.  No barrier besides: the bands' named barriers, and
    the setup's two block barriers more."""
    src = _text(8, matrix_input_chunks=chunks)
    bands = hi._kernel(8, 'sm_100a',
                       matrix_input_chunks=chunks).get_kernel()
    setup = re.search(
        r'int32_t \* (v\d+_tk) = reinterpret_cast<int32_t\*>\(&totalShrMem'
        r'\[\d+\]\);\s*bool (v\d+_lead) = threadIdx\.x == 0 && threadIdx\.y '
        r'== 0;\s*if \(\2\) \{\s*'
        + ''.join(r'\1\[' + str(k) + r'\] = 0_i32;\s*' for k in range(8))
        + r'\}\s*__syncthreads\(\);', src)
    assert setup
    tk = setup.group(1)
    assert _ring_slots(src, 'tp_ring') == [1]
    assert 'tensorforge::mbarrierInit(&tp_ring[r], 12);' in src
    for ring in ('me_ring', 'ik_ring') if chunks == 32 else ('me_ring',):
        assert re.search(r'mbarrierInit\(&' + ring + r'\[\d \+ r\], 12\);',
                         src)
        assert re.search(r'mbarrierInit\(&' + ring + r'\[\d \+ r\], 8\);',
                         bands)
    regions = re.findall(
        r'int32_t (v\d+_tk) = \((v\d+_q) \* 2\) \+ (v\d+_\w+);\s*'
        r'bool (v\d+_apart) = tensorforge::warpsApart\(\);\s*if \(\4\) \{',
        src)
    assert len(regions) == 2 and src.count('warpsApart') == 2
    draws = re.findall(r'tensorforge::ticketTake\(&' + tk + r'\[(v\d+_tk)\]\)',
                       src)
    assert sorted(draws) == sorted(2 * [r[0] for r in regions])
    meet = (r'tensorforge::mbarrierArrive\(&tp_ring\[(v\d+_\w+)\], '
            r'(v\d+_first)\);\s*tensorforge::mbarrierWait\(&tp_ring\[\1\], '
            r'(v\d+_\w+)\);\s*')
    star = re.search(meet + r'if \(\(threadIdx\.y == 0\) && \2\) \{\s*'
                     + tk + r'\[\3\] = 0_i32;\s*'
                     + ''.join(tk + r'\[\(\3 \+ ' + str(k)
                               + r'\)\] = 0_i32;\s*' for k in (2, 4, 6))
                     + r'\}', src)
    epilogue = re.search(meet + r'tensorforge::mbarrierWait\(&td_ring\[',
                         src)
    assert star and epilogue and star.start() < epilogue.start()
    assert src.count('&tp_ring[') == 2 + 2 * 2
    parity = star.group(3)
    assert epilogue.group(3) == parity
    assert all(r[2] == parity for r in regions)
    assert len(re.findall(r'\b' + tk + r'\[[^\]]*\] = ', src)) == 8 + 4
    assert src.count('barrier.sync') == bands.count('barrier.sync')
    assert src.count('__syncthreads()') == bands.count('__syncthreads()') + 2
    assert 'A mixed in a pool of 12 warps' in src


def test_three_bands_mix_on_every_slot():
    """Three bands in the pool: A's ring its four slots, a quarter's four
    mixers -- where the bands keep three, a slot each -- and sixteen
    warps mixing."""
    assert _ring_slots(_text(8, matrix_bands=3), 'ta3_ring') == [4]
    assert _ring_slots(hi._kernel(8, 'sm_100a', matrix_bands=3).get_kernel(),
                       'ta3_ring') == [3]
    assert 'tensorforge::mbarrierInit(&tp_ring[r], 16);' in _text(
        8, matrix_bands=3)


def test_the_mixers_barrier_orders_the_reset():
    """The reset is ordered by the mixers' barrier's own forms: each warp's
    elected lane draws, then arrives by `mbarrier.arrive` -- release at the
    CTA's scope, PTX's default -- and the resetting lane waits by
    `mbarrier.try_wait.parity`, acquire at the CTA's scope; behind the next
    phase, which the resetting warp's arrival completes, the trip two on
    draws.  Neither of the helpers relaxed or of a wider scope, and the
    barrier's wait that form whatever `matrix_wait` asks of the mma warp."""
    text = (tma.INCLUDE / 'tensorforge_device' / 'cuda.h').read_text()

    def body(name):
        found = re.search(r'__device__ __forceinline__ void ' + name
                          + r'\(void \*barrier, [^)]*\) \{(.*?)\n\}', text,
                          re.S)
        assert found, name
        return found.group(1)
    arrive, wait = body('mbarrierArrive'), body('mbarrierWait')
    assert 'mbarrier.arrive.shared::cta.b64 _, [%0];' in arrive
    assert 'mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;' in wait
    for helper in (arrive, wait):
        assert not re.search(r'\.(relaxed|cluster|gpu|sys)\b', helper)
    for form in ('suspend', 'test'):
        src = _text(8, matrix_wait=form)
        assert len(re.findall(r'tensorforge::mbarrierWait\(&tp_ring\[',
                              src)) == 2


# -- the oracle ----------------------------------------------------------------- #

def _tickets(monkeypatch):
    """Every ticket the oracle's warps take, in the order they take them:
    `(warp, counter, ticket)`, the counter its word's offset."""
    from tensorforge.reference import kernel_eval
    taken = []
    real = kernel_eval.Interp._ticket

    def ticket(self, counter):
        value = real(self, counter)
        entry = (self.lane // kernel_eval.Races.WARP, counter.off, value)
        # Every lane of the warp asks; the first one draws.
        if not taken or taken[-1] != entry:
            taken.append(entry)
        return value
    monkeypatch.setattr(kernel_eval.Interp, '_ticket', ticket)
    return taken


#: The blocks of K a trip of the tile's star step: I's rows (o4 10, o6 35)
#: eight a block.
BLOCKS = {4: 2, 6: 5}


@pytest.mark.parametrize('order,chunks', [(4, None), (4, 16), (6, None),
                                          (6, 16)])
def test_the_pool_on_the_oracle(order, chunks, monkeypatch):
    """31 elements -- three trips of fourteen, the last one's eleven past
    the batch masked, so that the counters of the first trip's parity are
    reset behind its barrier and drawn again -- I whole and in chunks of 16
    rows: under the warps' own order and three schedules, against NumPy
    without a race (`test_hiorder_tile._on_the_oracle`), bit for bit the
    star bands' output.  Each counter of a quarter and a trip's parity
    hands out every trip it counts from 0, the trip's blocks and a ticket
    past them to each of the quarter's three mixers -- its two band warps
    and its epilogue warp, no other; the epilogue's warps fill blocks too,
    and who fills which differs with the schedule."""
    options = {} if chunks is None else {'matrix_input_chunks': chunks}
    _, bands = hi._on_the_oracle('sm_100a', order, 31, **options)
    taken = _tickets(monkeypatch)
    blocks = BLOCKS[order]
    owners, fillers = set(), set()
    for seed in (None, 1, 2, 3):
        del taken[:]
        gen, got = hi._on_the_oracle('sm_100a', order, 31, schedule=seed,
                                     **POOL, **options)
        assert gen.built._element_group.section.tile.pool == 'on'
        assert (got == bands).all()
        first = min(off for _, off, _ in taken)
        counters = {}
        for warp, off, value in taken:
            counters.setdefault(off - first, []).append((warp, value))
        assert sorted(counters) == list(range(8))
        for index, draws in counters.items():
            quarter, parity = divmod(index, 2)
            trips = 2 if parity == 0 else 1
            assert [v for _, v in draws] == list(range(blocks + 3)) * trips
            assert {w for w, _ in draws} == {quarter, quarter + 4,
                                             quarter + 8}
        owners.add(tuple(taken))
        fillers |= {warp for warp, _, value in taken if value < blocks}
    assert len(owners) > 1
    assert fillers & {8, 9, 10, 11}


# -- the build ------------------------------------------------------------------ #

#: A spill instruction of the SASS.
SPILL = re.compile(r'(?:@!?U?P\w+\s+)?(?:LDL|STL)\b')


def _drawing_loops(sass):
    """The SASS's innermost loops that draw a ticket (ATOMS.ADD): each a
    backward branch's body, ahead of the stubs the waits branch out to
    behind the last EXIT -- their instructions."""
    ops = [(int(m.group(1), 16), m.group(2).strip()) for m in re.finditer(
        r'/\*([0-9a-f]{4,})\*/\s+([^;]*);', sass)]
    end = max(at for at, op in ops if re.match(r'(@!?U?P\w+\s+)?EXIT\b', op))
    loops = []
    for at, op in ops:
        branch = re.search(r'\bBRA\b.*?0x([0-9a-f]+)', op)
        if at <= end and branch and int(branch.group(1), 16) <= at:
            body = [o for a, o in ops if int(branch.group(1), 16) <= a <= at]
            if any('ATOMS.ADD' in o for o in body):
                loops.append((int(branch.group(1), 16), at, body))
    return [body for lo, hi, body in loops
            if not any(lo <= a and b <= hi and (a, b) != (lo, hi)
                       for a, b, _ in loops)]


@pytest.mark.parametrize('chunks', [32, 'whole'])
def test_the_pool_builds(chunks, tmp_path):
    """o8 built for sm_100a with the pool, I in chunks as the options leave
    it and whole: ptxas's report free of its warnings and as many registers
    a thread as the bands' build, 128; the group loop's nine UTCHMMA; the
    draws' fetch-adds on shared memory (ATOMS.ADD), none of the bands', two
    of each of the two regions -- each region's loop over its tickets,
    which draws one, with nothing spilled or reloaded.  The trip's own
    values spill a little more than the bands' (o8 in chunks 40 bytes
    stored against 32, whole 16 against 12): at most 16 bytes more."""
    def build(**options):
        sass, report = hi._ptxas('sm_100a', tmp_path,
                                 matrix_input_chunks=chunks, **options)
        assert not set(re.findall(r'\bC75\d\d\b', report)), report[-2000:]
        ops = [m.group(1).strip() for m in re.finditer(
            r'/\*[0-9a-f]{4,}\*/\s+([^;]*);', sass)]
        registers = [int(n) for n in re.findall(r'Used (\d+) registers',
                                                report)]
        spill = sum(int(n) for n in re.findall(r'(\d+) bytes spill stores',
                                               report))
        return (registers, spill, sum('UTCHMMA' in op for op in ops),
                sum('ATOMS.ADD' in op for op in ops), sass)
    bands, pool = build(), build(**POOL)
    assert pool[0] == bands[0] == [128]
    assert pool[1] <= bands[1] + 16
    assert pool[2] == bands[2] == 9
    assert (bands[3], pool[3]) == (0, 4)
    loops = _drawing_loops(pool[4])
    assert len(loops) == 2
    assert not any(SPILL.match(op) for body in loops for op in body)
