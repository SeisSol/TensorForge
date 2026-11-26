# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""The high-order tile of the multi-element layout's asynchronous tiling
(`Options.matrix_tile` `high`, `element_group.TileRows`,
`compute.multi_element._tile`): the group's rows dense on the engine's --
on wgmma (sm_90a) a tile of whole elements a consumer warpgroup, each a
role of its own, sharing the prepared operator ring that a producer's warp
fills while another stages the inputs; on tcgen05 (sm_100a) the three roles
at as many elements as tensor memory's lanes hold.

B streams in kept spans (`Options.matrix_spans`, `element_group.
tile_chunks`): of each product's block of K only its tiles of N from the
first that holds an entry to the last, in the engine's granules, each
multiplied at its own N into D's columns of it.  The blocks are one loop
(`Options.matrix_groups`, `multi_element._wgmma_groups`) -- its body written
once, one dispatch a block on the block's width -- or written out, as each
engine measured faster (`element_group.tile_groups`).

What is checked here: the plan -- the options' defaults, the rows' map, the
group and the block the fill rule takes on each engine, the refusals and
the kernel's note of them; the kept spans against the operators' entries,
and the prepared table as the descriptors read it; the kernel's text -- the
roles, their barriers, the rings' counts, the spans' widths and the group
loop; the PIR -- the rings' hand-offs by role, verified
through every pass; the host oracle over two trips, the second's elements
masked past the batch, against NumPy and without a race, and with dense
spans bit for bit what the kept ones give; the build for sm_90a and
sm_100a read back; and on a GH200 / GB200 the volume against NumPy.
"""

from __future__ import annotations

import dataclasses
import re
import shutil
import subprocess
import types
import warnings

import numpy as np
import pytest

import test_multi_element as me
import test_multi_element_async as tma
import test_ring_object as tro
from tensorforge.backend.pir.passes import verify
from tensorforge.common.basic_types import Datatype
from tensorforge.generators import element_group as eg

#: The high-order tile asked for.
HIGH = {'matrix_engine': 'auto', 'matrix_tile': 'high'}


def _group(order=8, arch='sm_90a', shape='volume', precision='s',
           **options):
    group = eg.plan(me._descrs(shape, order, precision), me._context(
        arch, precision, multi_element='force', **{**HIGH, **options}))
    assert group is not None
    return group


def _kernel(order=8, arch='sm_90a', **options):
    return tma._kernel(arch, 'volume', order, 's', **{**HIGH, **options})


# -- the plan ------------------------------------------------------------------- #

def test_the_options_default_to_the_tilings_as_before():
    from tensorforge.common.options import registry
    assert registry()['matrix_tile'].default == 'off'
    assert registry()['matrix_consumers'].default == 2


def test_the_options_parse():
    from tensorforge.common.options import (parse_matrix_consumers,
                                            parse_matrix_tile)
    assert [parse_matrix_tile(t) for t in ('off', 'HIGH')] == ['off', 'high']
    assert [parse_matrix_consumers(t) for t in '1234'] == [1, 2, 3, 4]
    for text in ('0', '5'):
        with pytest.raises(ValueError):
            parse_matrix_consumers(text)
    with pytest.raises(ValueError):
        parse_matrix_tile('low')


def test_the_rows_are_whole_elements_a_tile():
    """Seven elements of nine quantities a wgmma tile of 64 rows, the 64th
    padding; fourteen on tcgen05's 128 lanes, two rows of padding."""
    wg = eg.TileRows('wgmma', 9, 7, 2)
    assert (wg.m, wg.used, wg.elements) == (64, 63, 14)
    assert wg.place(0, 0) == (0, 0) and wg.place(6, 8) == (0, 62)
    assert wg.place(7, 0) == (1, 0) and wg.place(13, 8) == (1, 62)
    tc = eg.TileRows('tcgen05', 9, 14, 1)
    assert (tc.m, tc.used, tc.elements) == (128, 126, 14)
    assert tc.place(13, 8) == (0, 125)


@pytest.mark.parametrize('order', [4, 6, 8])
def test_on_wgmma_two_consumer_warpgroups_of_seven_elements(order):
    """sm_90a: fourteen elements, each consumer warpgroup a tile of seven
    on 63 of its 64 rows -- not E/4 warpgroups each issuing 64 rows over
    9E -- and two producer warps behind them; no warp an element's, the
    batch loop's rows all a producer's; the star folded, B streamed
    prepared through its ring, the inputs through the producer's -- I whole,
    which the tile fits on wgmma at o4 to o8 (`tile_plan`, `TileRows.
    chunk`)."""
    group = _group(order)
    section = group.section
    assert section.tile == eg.TileRows('wgmma', 9, 7, 2, groups='unrolled')
    assert group.elements == 14 and group.warps == 10
    assert group.rows == 0 and group.producers == 10
    assert eg.rowless(section) and not section.roles
    assert section.star == 'engine' and section.ring >= 1
    (step,) = [s for s in section.steps if s.lowering == 'wgmma']
    assert all(st.prepared and st.streamed
               for st in eg._step_stages(section, step))


def test_one_consumer_warpgroup_where_the_options_say_so():
    group = _group(8, matrix_consumers=1)
    assert group.section.tile == eg.TileRows('wgmma', 9, 7, 1,
                                             groups='unrolled')
    assert group.elements == 7 and group.warps == 6


@pytest.mark.parametrize('order', [4, 6, 8])
def test_on_tcgen05_the_tile_s_own_layout_at_fourteen_elements(order):
    """sm_100a: fourteen elements on 126 of tensor memory's 128 lanes, the
    tile's own layout -- two star bands (`TILE_BANDS`), the epilogue
    warpgroup and the input, operator and mma warps, 15, no warp an
    element's -- D in two slots of tensor memory, added by bulk reductions
    (`tile_output`: a column's at o4 and o6, the element's box at o8), the
    inputs through a ring, whatever the table's group or `matrix_roles` --
    I in chunks of 32 rows of K at o8, whole at o4 (ten rows) and o6 (its
    last chunk's 3, `tile_plan`), its copies by the elected lane
    (`tile_input_issue`)."""
    group = _group(order, 'sm_100a')
    section = group.section
    chunk = 32 if order > 6 else 0
    assert section.tile == eg.TileRows('tcgen05', 9, 14, 1, groups='loop',
                                       bands=2, dslots=2, aslots=4, chunk=chunk,
                                       cslots=2 if chunk else 0,
                                       output='box' if order == 8 else 'bulk',
                                       inputs='elect')
    assert not section.roles and eg.rowless(section) and section.ring >= 1
    assert group.elements == 14 and group.rows == 0
    assert group.warps == 4 * 2 + 4 + 3 == eg.tile_warps(section.tile)


def test_d_s_slots_where_tensor_memory_holds_them():
    """D in two slots where they fit beside A's ring of four slots in the
    plan's budget of tensor memory (`TMEM_BUDGET`, one number): o8 (N 128)
    448 of 512; o10 (224) and o12 (368) one."""
    assert eg.TMEM_BUDGET == 512
    assert [eg.tile_dslots(n, 3) for n in (32, 64, 128, 224, 368)] == [
        2, 2, 2, 1, 1]
    assert eg.tile_dslots(224, 3, budget=1024) == 2
    from tensorforge.backend.instructions.compute import multi_element as mm
    section = _group(8, 'sm_100a').section
    (step,) = [s for s in section.steps if s.lowering == 'tcgen05']
    columns, at, _ = mm._tcgen05_columns(section, step)
    # A's ring four slots, two a band (`_Fold`, `_fold_bands`); three at
    # three bands, one each.
    assert (columns, at) == (2 * 128 + 4 * 48, (256, 272, 288))
    three = _group(8, 'sm_100a', matrix_bands=3).section
    (step3,) = [s for s in three.steps if s.lowering == 'tcgen05']
    assert mm._tcgen05_columns(three, step3)[0] == 2 * 128 + 3 * 48
    assert mm.tmem_columns(section) == 512
    # Each slot of D a pair of mbarriers in shared memory.
    single = dataclasses.replace(section, tile=dataclasses.replace(
        section.tile, dslots=1))
    assert eg.shared_bytes(section, 14) - eg.shared_bytes(single, 14) == 16


def test_a_trip_s_d_is_its_slot_s():
    """o8 on tcgen05: a trip's D in the slot D's ring gives it, its first
    column the slot times N -- the mma warp's acquire of that slot, the
    products' columns and the epilogue's loads all from it -- so that a
    trip's epilogue reads the slot the next trip's products leave alone
    (which the oracle, running the roles in turn, cannot see)."""
    src = _kernel(8, 'sm_100a').get_kernel()
    (base, slot), = re.findall(r'int32_t (v\d+_db) = (v\d+_rs) \* 128;',
                               src)
    assert re.search(rf'mbarrierWait\(&td_ring\[2 \+ \({slot}\)\]', src)
    assert re.search(rf'tcgen05CommitOn\(&td_ring\[{slot}\]', src)
    uses = [line for line in src.splitlines()
            if re.search(rf'\b{base}\b', line)
            and not line.strip().startswith(f'int32_t {base} =')]
    # The products' first column of D, and the epilogue's loads.
    assert any('tmemWarpAddress' not in line for line in uses), uses
    assert any('tmemWarpAddress' in line for line in uses), uses


@pytest.mark.parametrize('output', ['red', 'bulk', 'box'])
def test_the_epilogue_s_order_in_the_kernel(output):
    """tcgen05's epilogue in the order the oracle, running the roles in
    turn, cannot check: D's slot given back behind every load of it; with
    bulk reductions the box's stores fenced for the async proxy, the role
    met, then the reductions; and the role met again behind the waits for
    their reads, before the next trip writes the box."""
    lines = _kernel(6, 'sm_100a', matrix_output=output).get_kernel(
        ).splitlines()

    def at(text):
        return [k for k, line in enumerate(lines) if text in line]
    (give,) = at('mbarrierArrive(&td_ring[2 + ')
    assert at('tcgen05.ld.sync') and max(at('tcgen05.ld.sync')) < give
    if output == 'red':
        return
    (reduce,) = at('tensorforge::bulkReduceAdd(')
    stores = [k for k in at('_dbox[') if '] = ' in lines[k]]
    fences = [k for k in at('fenceProxyAsync') if max(stores) < k < reduce]
    assert fences
    assert any(fences[0] < k < reduce for k in at('barrier.sync'))
    (wait,) = at('bulkWaitRead')
    assert any(wait < k <= wait + 2 for k in at('barrier.sync'))


def _capped(monkeypatch, cap):
    """The plan's shared memory capped so that no group of more than `cap`
    elements fits: the tile on fewer elements than its star warps, as
    o12's bulk reductions (five) and I whole (seven) take it."""
    shared = eg.shared_bytes
    monkeypatch.setattr(eg, 'shared_bytes', lambda section, elements: (
        shared(section, elements) if elements <= cap else 1 << 30))


@pytest.mark.parametrize('cap', [5, 7])
def test_fewer_elements_than_star_warps_on_the_oracle(monkeypatch, cap):
    """tcgen05's tile on five and seven elements, fewer than its two bands'
    eight star warps: each warp its place among the block's warps, not its
    element -- the modulo put warps 5 to 7 in the first band's place, the
    second band's A short of their arrivals, and the block hung (J3, o12
    bulk, box and I whole on a GB200).  Three trips and more, against NumPy
    without a race, and bit for bit what fourteen elements give."""
    _, whole = _on_the_oracle('sm_100a', 4, 31)
    _capped(monkeypatch, cap)
    gen, got = _on_the_oracle('sm_100a', 4, 31)
    assert gen.built._element_group.elements == cap
    assert (got == whole).all()
    src = gen.get_kernel()
    # The bands by the warp, the batch loop's lanes by the element.
    assert re.search(r'_band = threadIdx\.y / 4;', src)
    assert f'threadIdx.y % {cap};' in src


def test_two_bands_over_o6_s_blocks_on_the_oracle():
    """Two star bands over o6's five blocks of K, two trips: each band its
    blocks two apart, the second band's three past the first's -- where
    o4's two blocks would take one each whatever the bands' stride."""
    gen, _ = _on_the_oracle('sm_100a', 6, 17, matrix_bands=2)
    assert gen.built._element_group.section.tile.bands == 2


def test_two_bands_are_fewer_warps():
    group = _group(8, 'sm_100a')
    assert group.section.tile.bands == 2 and group.warps == 15
    three = _group(8, 'sm_100a', matrix_bands=3)
    assert three.section.tile.bands == 3 and three.warps == 19


def test_the_tile_s_defaults_on_tcgen05():
    """What a GB200 measured (JD, 262144 elements): two star bands -- o8
    7.20 ns an element against three's 11.39, o6 4.67 against 4.86 -- and
    D added by bulk reductions -- o6 a column's 3.06 against the box's 3.45
    and a reduction a cell's 4.67, o8 the box's 6.61 against 7.00 and 7.20
    -- so `matrix_bands` 2 and `matrix_output` `auto`: bulk where an
    element's rows of D are fewer than `TILE_BOX_ROWS`, box from there on,
    bulk again from `TILE_BULK_ROWS` -- o10's bulk 13.31, 13.24 and 13.23
    against the box's 13.49, 13.49 and 13.52 (J3b, J4a, J4b) -- a
    reduction a cell where the bulk reductions' staged D would take the
    group more than a quarter of its elements (o12) and on wgmma -- o10's
    box at twelve elements 13.33 against red's 17.11 at fourteen.  `auto`
    is `red` everywhere else: a kernel without the tile is the kernel it
    was."""
    from tensorforge.common.options import parse_matrix_output, registry
    assert registry()['matrix_bands'].default == eg.TILE_BANDS == 2
    assert registry()['matrix_output'].default == 'auto'
    assert parse_matrix_output('Auto') == 'auto'
    context = me._context('sm_100a', 's', multi_element='force', **HIGH)
    assert (eg.TILE_BOX_ROWS, eg.TILE_BULK_ROWS) == (64, 168)
    assert [eg.tile_output('tcgen05', rows, context)
            for rows in (20, 56, 63, 64, 120, 167, 168, 220, 364)] == [
        'bulk', 'bulk', 'bulk', 'box', 'box', 'box', 'bulk', 'bulk', 'bulk']
    assert eg.tile_output('wgmma', 120, context) == 'red'
    for asked in ('red', 'bulk', 'box'):
        asked_context = me._context('sm_100a', 's', multi_element='force',
                                    **{**HIGH, 'matrix_output': asked})
        assert eg.tile_output('tcgen05', 120, asked_context) == asked
    assert [_group(o, 'sm_100a').section.tile.output
            for o in (4, 6, 8, 10, 12)] == ['bulk', 'bulk', 'box', 'bulk',
                                             'red']
    # Where they keep three quarters of a reduction a cell's elements:
    # o10's twelve of fourteen, which a GB200 measured faster; not o12's
    # five.
    assert eg.TILE_OUTPUT_KEEP == 0.75
    assert _group(10, 'sm_100a').elements == 12
    assert _group(10, 'sm_100a', matrix_output='red').elements == 14
    assert _group(12, 'sm_100a', matrix_output='box').elements == 5
    assert _group(12, 'sm_100a').elements == 14
    # Three bands asked: at o12 two, on A's two slots of tensor memory.
    assert _group(12, 'sm_100a', matrix_bands=3).section.tile.bands == 2
    assert _group(8, 'sm_90a').section.tile.output == 'red'

    def body(gen):
        # The kernel but its name and the options' comment, which lists
        # `red` asked now that it is not the default.
        text = re.sub(r'kernel_kernel_[0-9a-f]+', 'kernel', gen.get_kernel())
        return [line for line in text.splitlines()
                if not line.strip().startswith('// options:')]
    for arch in ('sm_90a', 'sm_100a'):
        auto = tma._kernel(arch, 'volume', 6, 's')
        red = tma._kernel(arch, 'volume', 6, 's', matrix_output='red')
        assert body(auto) == body(red)
        assert 'matrix_output' not in auto.get_kernel()


def test_the_option_off_is_the_kernel_without_it():
    for arch in ('sm_90a', 'sm_100a'):
        off = tma._kernel(arch, 'volume', 6, 's', matrix_tile='off')
        none = tma._kernel(arch, 'volume', 6, 's')
        assert off.get_kernel() == none.get_kernel()
        assert '// tile:' not in off.get_kernel()


@pytest.mark.parametrize('arch,precision,options,why', [
    ('sm_90a', 'd', {}, 'steps on the asynchronous engines'),
    ('sm_80', 's', {}, 'steps on the asynchronous engines'),
])
def test_where_the_tile_does_not_take_the_kernel_says_why(arch, precision,
                                                          options, why):
    group = _group(6, arch, precision=precision, **options)
    assert group.section.tile is None
    assert why in group.tile_note
    context = me._context(arch, precision, multi_element='force',
                          **{**HIGH, **options})
    (note,) = eg.tile_notes(group, context)
    assert note.startswith('matrix_tile=high declined: ') and why in note


def test_the_kernel_says_what_the_tile_took():
    src = _kernel(8).get_kernel()
    (note, _) = [line.strip() for line in src.splitlines()
                 if line.strip().startswith('// tile:')]
    assert '14 elements, 2 tiles of 7 on 63 of 64 rows' in note
    src = _kernel(8, 'sm_100a').get_kernel()
    (note, _) = [line.strip() for line in src.splitlines()
                 if line.strip().startswith('// tile:')]
    assert 'warps 0 to 7 its star bands' in note
    assert '(2 slots of D)' in note and 'warp 14 issues' in note
    assert 'staged in shared memory, a bulk reduction an element' in note


# -- the kernel ------------------------------------------------------------------ #

#: A role's barrier id as the split spells it: the top three of NVIDIA's
#: sixteen, by the warp (`roles.first_id`).
IDENT = r'\(13 \+ \(threadIdx\.y >= 4\) \+ \(threadIdx\.y >= 8\)\)'


def test_the_kernel_splits_the_block_into_its_roles():
    """The two tiles' roles and the producer's, each meeting on its own
    named barrier of its threads -- a warpgroup's 128 -- and no block
    barrier in the trip; both rings' empty sides counting a consumer
    warp's arrival each."""
    src = _kernel(8).get_kernel()
    assert re.search(r'bool v\d+_tile0 = threadIdx\.y < 4;', src)
    assert re.search(r'bool v\d+_tile1 = \(threadIdx\.y >= 4\) && '
                     r'\(threadIdx\.y < 8\);', src)
    counts = set(re.findall(rf'barrier\.sync %0, %1;" :: "r"\({IDENT}\), '
                            rf'"r"\((\d+)\)', src))
    assert counts == {'128'}
    loop = src[src.index('for (size_t'):]
    assert '__syncthreads()' not in loop
    assert re.search(r'mbarrierInit\(&me_ring\[\d+ \+ r\], 8\)', src)
    assert re.search(r'mbarrierInit\(&bk\d+_ring\[\d+ \+ r\], 8\)', src)
    # Each tile's blocks written out at o8, as the options leave them: 99
    # products, nine of each of the eleven blocks -- three of 3xTF32 of
    # each of its three; the group loop's a block's nine once for each of
    # the blocks' six widths.
    assert src.count('wgmma.mma_async') == 2 * 99
    loop = _kernel(8, matrix_groups='loop').get_kernel()
    assert loop.count('wgmma.mma_async') == 2 * 6 * 9


def _bodies(**options):
    return tro._bodies(order=6, arch='sm_90a', **{
        k: v for k, v in {**HIGH, **options}.items()
        if k != 'matrix_engine'})


@pytest.mark.parametrize('chunks', [32, 'auto'])
def test_the_rings_hand_over_between_the_roles(chunks):
    """The rings' acts by role (`pir.rings.handoffs`): the producer fills
    them all -- the inputs', B's, and with I in chunks theirs -- and the
    tiles drain them all.  With I in chunks the input ring is the star
    matrices', a cell a copy of the producer's warp, none a bulk copy
    into the slot; with I whole -- as the options leave it on wgmma -- the
    producer's bulk copies are I's."""
    uses = tro._uses(_bodies(matrix_input_chunks=chunks))
    inputs = uses['me_ring']
    (b,) = [u for name, u in uses.items() if name.startswith('bk')]
    rings = [inputs, b] + ([uses['ik_ring']] if chunks == 32 else [])
    assert ('ik_ring' in uses) == (chunks == 32)
    for use in rings:
        assert (use.spec.producer, use.spec.consumers) == (
            'producer', ('tile0', 'tile1'))
        assert use.roles('acquire') == use.roles('publish') == ('producer',)
        assert use.roles('copy') == (() if use is inputs and chunks == 32
                                     else ('producer',))
        assert use.roles('take') == use.roles('release') == ('tile0',
                                                             'tile1')


@pytest.mark.parametrize('arch,chunks', [('sm_90a', 'auto'),
                                         ('sm_100a', 'auto'),
                                         ('sm_100a', 32)])
def test_the_roles_verify(arch, chunks):
    """Through every pass: no act of a ring out of its roles, no role's
    barrier out of its role, the chain of the split whole -- at o6, I whole
    as `auto` takes it on both engines (`tile_plan`), and on tcgen05 in
    chunks, their ring's acts too."""
    found = []
    for body in tro._bodies(order=6, arch=arch, matrix_tile='high',
                            matrix_input_chunks=chunks):
        found += [d for d in verify(body, strict=False)
                  if "slot's" in d or 'role' in d or 'chain' in d]
    assert not found, found[:3]


def test_tcgen05_s_rings_hand_over_between_its_roles():
    """tcgen05's layout: the input warp fills the inputs' ring and the star
    bands drain it; the operator warp fills B's and the mma warp's products
    commit its slots back; the bands fill A's ring in tensor memory, the
    products give its slots back; the mma warp's products commit D's slot
    full and the epilogue gives it back."""
    uses = tro._uses(tro._bodies(order=6, arch='sm_100a', matrix_tile='high'))
    expect = {'me_ring': ('input', ('star',)),
              'bk_ring': ('operator', ('mma',)),
              'ta_ring': ('star', ('mma',)),
              'td_ring': ('mma', ('epilogue',))}
    for name, (producer, consumers) in expect.items():
        # The rings by their names, the step's number left out (`bk3`).
        (use,) = [u for n, u in uses.items() if re.sub(r'\d+', '', n) == name]
        assert (use.spec.producer, use.spec.consumers) == (producer,
                                                           consumers), name
    (d,) = [u for n, u in uses.items() if n.startswith('td')]
    assert d.roles('publish') == ('mma',)
    assert d.roles('take') == d.roles('release') == ('epilogue',)
    assert d.roles('acquire') == ('mma',)
    assert d.spec.slots == 2


def test_tcgen05_s_kernel_text():
    """o8: the five roles by the warp, D's ring of two slots -- full the
    products' commit, empty the epilogue's four warps -- the products'
    commit onto its full side, the inputs' ring drained by the two bands'
    eight warps."""
    src = _kernel(8, 'sm_100a').get_kernel()
    assert re.search(r'bool v\d+_star = threadIdx\.y < 8;', src)
    assert re.search(r'mbarrierInit\(&td\w*\[\d+ \+ r\], 4\)', src)
    assert re.search(r'mbarrierInit\(&td\w*\[r\], 1\)', src)
    assert re.search(r'tcgen05CommitOn\(&td\w*\[v\w+\], ', src)
    assert re.search(r'mbarrierInit\(&me_ring\[\d+ \+ r\], 8\)', src)


# -- the oracle ------------------------------------------------------------------ #

def _on_the_oracle(arch, order, count, grid=1, launches=(), **options):
    """Every element of `count` against NumPy on the host oracle, the
    group's trips over them -- the last one's elements past the batch
    masked -- and no access a ring, a role's barrier or a wait does not
    order (`kernel_eval.Races`).  `grid` blocks in turn, `launches` what
    each block's claims are granted (`kernel_eval.evaluate_wave`)."""
    import numpy as np
    from tensorforge.common.basic_types import Addressing
    from tensorforge.reference import kernel_eval
    gen = tma._kernel(arch, 'volume', order, 's', **{**HIGH, **options})
    tensors = {}
    for outer in gen.descr_list:
        for d in outer.operations():
            for m in d.matrix_list():
                t = getattr(m, 'tensor', m)
                tensors.setdefault(t.alias, t)

    def box(t):
        return [(int(lo), int(hi)) for lo, hi in zip(t.bbox.lower(),
                                                      t.bbox.upper())]

    def memory(t, read):
        (r0, r1), (c0, c1) = box(t)
        out = np.zeros((count, r1 - r0, c1 - c0))
        for e in range(count):
            for c in range(c1 - c0):
                for r in range(r1 - r0):
                    out[e, r, c] = read(e, r + c * (r1 - r0))
        return out

    fill = kernel_eval.Slot(me.SEED)
    preset, inputs = {}, {}
    for alias, t in tensors.items():
        if t.is_tmp:
            continue
        if t.addressing.kind is Addressing.NONE:
            preset[t.name] = [float(v) for v in t.storage_values()]
            (r0, r1), (c0, c1) = box(t)
            inputs[alias] = np.asarray(t.get_values())[None, r0:r1, c0:c1]
        else:
            inputs[alias] = memory(t, lambda e, s, t=t: fill.read(
                f'{t.name}@{e}', s))
    output = next(t for t in tensors.values() if not t.is_tmp
                  and t.addressing.kind is not Addressing.NONE
                  and any(d.writes().tensor is t for outer in gen.descr_list
                          for d in outer.operations()))
    lanes, mults = kernel_eval.launch_geometry(gen.get_launcher())
    races = []
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        mem = kernel_eval.evaluate_wave(
            gen.get_kernel(), lanes, seed=me.SEED, globals_only=True,
            mults=mults, block=True, preset=preset, elements=count,
            races=races, grid=grid, launches=launches)
    assert not races, races[:3]
    got = memory(output, lambda e, s: mem.get((f'{output.name}@{e}', s),
                                              fill.read(f'{output.name}@{e}',
                                                        s)))
    want = me._reference(me._descrs('volume', order, 's'))(
        inputs, inputs[output.alias])
    for e in range(count):
        scale = np.abs(want[e]).max() or 1.0
        error = np.abs(got[e] - want[e]).max() / scale
        assert error <= 1e-6, (e, error)
    return gen, got


@pytest.mark.parametrize('arch', ['sm_90a', 'sm_100a'])
def test_two_trips_of_the_tile_on_the_oracle(arch):
    """Seventeen elements: a trip of fourteen and one of three, the eleven
    past the batch masked -- on wgmma the second tile's rows all of masked
    elements, its products run, its writes none."""
    gen, _ = _on_the_oracle(arch, 4, 17)
    assert gen.built._element_group.elements == 14


def test_one_consumer_warpgroup_on_the_oracle():
    gen, _ = _on_the_oracle('sm_90a', 4, 9, matrix_consumers=1)
    assert gen.built._element_group.elements == 7


@pytest.mark.parametrize('groups', ['loop', 'unrolled'])
def test_one_slot_of_the_input_ring_on_the_oracle(monkeypatch, groups):
    """The input ring of one slot, the plan's last choice (`tile_plan`):
    the second trip's inputs go into the slot the first trip's mixing
    read, the producer behind every consumer warp's last block of A
    (`_wgmma_tile`'s `done`) -- which the oracle's ring order checks."""
    monkeypatch.setattr(eg, 'PRODUCER_SLOTS', ())
    gen, _ = _on_the_oracle('sm_90a', 4, 17, matrix_groups=groups)
    assert gen.built._element_group.section.ring == 1


@pytest.mark.parametrize('arch', ['sm_90a', 'sm_100a'])
@pytest.mark.parametrize('groups', ['loop', 'unrolled'])
def test_two_slots_of_b_s_ring_on_the_oracle(arch, groups):
    """B's prepared ring at two slots: o4's two trips' four chunks through
    them, each slot taken again behind the wait that retired the products
    that read it and the warps' arrivals that gave it back -- which the
    oracle's ring order checks."""
    gen, _ = _on_the_oracle(arch, 4, 17, matrix_ring_slots=2,
                            matrix_groups=groups)
    assert gen.built._element_group.section.ring_slots == 2


@pytest.mark.parametrize('arch', ['sm_90a', 'sm_100a'])
def test_the_group_loop_computes_what_the_unrolled_groups_do(arch):
    """The group loop against the groups written out, both against NumPy
    on the oracle: bit for bit -- the same products in the same order; on
    wgmma the loop's D started at the accumulators' zeros where the
    unrolled groups' first product writes it fresh, on tcgen05 N and D's
    columns values."""
    _, loop = _on_the_oracle(arch, 4, 17, matrix_groups='loop')
    _, unrolled = _on_the_oracle(arch, 4, 17, matrix_groups='unrolled')
    assert (loop == unrolled).all()


# -- the kept spans ------------------------------------------------------------- #

def _chunks(order=8, arch='sm_90a', **options):
    section = _group(order, arch, **options).section
    (step,) = [s for s in section.steps if s.lowering in eg.ASYNC_LOWERINGS]
    return section, step, eg.tile_chunks(section, step)


def test_the_spans_option_defaults_to_kept():
    from tensorforge.common.options import parse_matrix_spans, registry
    assert registry()['matrix_spans'].default == 'kept'
    assert [parse_matrix_spans(t) for t in ('kept', 'Dense')] == ['kept',
                                                                 'dense']
    with pytest.raises(ValueError):
        parse_matrix_spans('sparse')


@pytest.mark.parametrize('arch', ['sm_90a', 'sm_100a'])
@pytest.mark.parametrize('order', [4, 6, 8])
def test_every_entry_of_b_is_in_exactly_one_kept_piece(arch, order):
    """The kept spans against the operators' entries: every entry inside a
    product's window in exactly one piece of its product -- the block of K
    it is in, the rows of N of its span; the spans whole granules of the
    engine's N (8 on wgmma, 16 on tcgen05) inside the stage, on wgmma each
    on to the stage's last row (`TileRows.descending`), the first block's
    first piece whole; at o6 and o8 fewer rows of N than the dense pieces'
    (o8: 62 % of them on wgmma, 61 % on tcgen05, its group loop's union of
    a block's spans)."""
    section, step, chunks = _chunks(order, arch)
    granule = section.tile.granule
    assert granule == (16 if arch == 'sm_100a' else 8)
    assert section.tile.descending == (arch == 'sm_90a')
    stages = eg._step_stages(section, step)
    first = chunks[0][0]
    assert first[3:] == (0, stages[first[0]].width)
    for chunk in chunks:
        for p, k0, rows, lo, hi in chunk:
            assert 0 <= lo < hi <= stages[p].width
            assert lo % granule == 0 and hi % granule == 0
            assert hi == stages[p].width or not section.tile.descending
    for p, product in enumerate(step.products):
        stage = stages[p]
        (r0, r1), (d0, d1) = stage.window
        pattern = product.first.tensor.nonzero_pattern
        for kk in range(d0, d1):
            for row in range(r0, r1):
                if not pattern.is_nz((row, kk)):
                    continue
                n, k = row - stage.n0, kk - d0
                held = [None for chunk in chunks
                        for q, k0, rows, lo, hi in chunk
                        if q == p and k0 <= k < k0 + rows and lo <= n < hi]
                assert len(held) == 1, (p, row, kk)
    kept = sum(rows * (hi - lo) for chunk in chunks
               for _, _, rows, lo, hi in chunk)
    dense = sum(rows * stages[p].width for chunk in chunks
                for p, _, rows, _, _ in chunk)
    assert kept <= dense
    if order == 8:
        # A block's products one span in the group loop, their union: 61 %
        # on tcgen05 where each product's own would keep 59 %.
        assert round(kept / dense, 2) == (0.61 if arch == 'sm_100a'
                                          else 0.62)


def test_dense_spans_are_whole_widths():
    section, step, chunks = _chunks(8, matrix_spans='dense')
    stages = eg._step_stages(section, step)
    assert section.tile.spans == 'dense'
    assert all((lo, hi) == (0, stages[p].width) for chunk in chunks
               for p, _, _, lo, hi in chunk)


def test_the_spans_are_dense_where_known_zeros_are_not_skipped():
    section, _, _ = _chunks(8, skip_known_zeros=False)
    assert section.tile.spans == 'dense'


def test_a_block_of_no_entry_is_refused(monkeypatch):
    """A block of K that no product has an entry in -- an inactive group,
    which the passes bring -- is refused with its reason, not taken."""
    kept = eg.kept_span

    def none_in_the_second_block(section, step, product, k0, rows, granule):
        return None if k0 == 8 else kept(section, step, product, k0, rows,
                                          granule)
    monkeypatch.setattr(eg, 'kept_span', none_in_the_second_block)
    group = _group(6)
    assert group.section.tile is None
    assert 'block 1 of K hold' in group.tile_note
    assert 'inactive group' in group.tile_note


@pytest.mark.parametrize('arch,sm', [('sm_90a', 90), ('sm_100a', 100)])
def test_the_table_holds_each_kept_piece_as_its_descriptors_read_it(arch, sm):
    """The prepared table walked as the descriptors read it (`_Piece`): a
    chunk's pieces one after another, each its rows of K by its span's rows
    of N in the canonical K-major layout over them -- core matrices of
    eight rows of N by four of K, on wgmma the span's tiles of N last first
    (`TileRows.descending`) -- the upper TF32 halves, then the lower; each
    entry of the operator's window split (`tf32_halves`), zero outside it;
    a chunk's elements what the ring copies (`chunk_volume`)."""
    import numpy as np
    from tensorforge.common.matrix.prepare import tf32_halves
    section, step, chunks = _chunks(6, arch)
    _, numbers, offsets = eg.prepared_table(section, step, sm)
    numbers = np.asarray(numbers, dtype=np.float32)
    stages = eg._step_stages(section, step)
    at = 0
    for chunk in chunks:
        size = 0
        for p, k0, rows, lo, hi in chunk:
            stage = stages[p]
            values = np.asarray(stage.tensor.get_values(), dtype=np.float64)
            (r0, r1), (d0, d1) = stage.window
            width = hi - lo
            assert offsets[(p, k0)] == at
            for k in range(rows):
                for n in range(width):
                    place = ((n // 8) * rows * 8 + (k // 4) * 32
                             + (n % 8) * 4 + k % 4)
                    # The laid row `n`: the span's `n` on tcgen05; on wgmma
                    # its tile `n / 8` counted from the span's last.
                    row = stage.n0 + (hi - 8 * (n // 8 + 1) + n % 8
                                      if section.tile.descending else lo + n)
                    col = d0 + k0 + k
                    value = (values[row, col] if r0 <= row < r1 and col < d1
                             else 0.0)
                    upper, lower = tf32_halves(np.array([value],
                                                        dtype=np.float32))
                    assert numbers[at + place] == upper[0], (p, k0, n, k)
                    assert numbers[at + rows * width + place] == lower[0]
            at += 2 * rows * width
            size += 2 * rows * width
        assert eg.chunk_volume(section, step, chunk) == size
    assert at == len(numbers)


@pytest.mark.parametrize('groups', ['loop', 'unrolled'])
def test_the_kernel_multiplies_each_span_at_its_width(groups):
    """o8 on wgmma: a product a piece, its term and its tile -- three of
    3xTF32 a piece, each tile its own -- at the piece's width: several N;
    written out, their sum the spans' rows of N; in the group loop, a
    block's products once for each width the blocks have.  Every one of
    them into the accumulators from their first register
    (`TileRows.descending`)."""
    section, step, chunks = _chunks(8, matrix_groups=groups)
    src = _kernel(8, matrix_groups=groups).get_kernel()
    widths = [int(n) for n in re.findall(r'wgmma\.mma_async\.sync\.aligned\.'
                                         r'm64n(\d+)k8', src)]
    pieces = [hi - lo for chunk in chunks for *_, lo, hi in chunk]
    if groups == 'loop':
        assert sorted(widths) == sorted(2 * 9 * sorted(set(pieces)))
    else:
        assert len(widths) == 2 * 3 * len(pieces)
        assert sorted(widths) == sorted(2 * 3 * pieces)
    assert len(set(widths)) > 1
    firsts = re.findall(r'wgmma\.mma_async\.sync\.aligned\.m64n\d+k8[^;]*;"'
                        r'\s*:\s*"\+f"\((\w+)\)', src)
    assert len(firsts) == len(widths) and len(set(firsts)) == 2


def test_the_kernel_says_what_the_spans_keep():
    src = _kernel(8).get_kernel()
    notes = [line.strip() for line in src.splitlines()
             if line.strip().startswith('// tile:')]
    assert any('B in kept spans: 62 % of the dense rows of N, 6 widths '
               'from 40 to 120, each to the stage\'s last row' in note
               for note in notes), notes
    src = _kernel(8, matrix_spans='dense').get_kernel()
    assert any('B dense' in line for line in src.splitlines()
               if line.strip().startswith('// tile:'))


def test_the_groups_option_defaults_to_what_each_engine_measured():
    """`auto`: the blocks written out on wgmma, which a GH200 measured
    faster than the loop at o6 to o12 (o8 11.49 ns an element against
    13.47, o12 72.74 against 96.32 at one warpgroup) -- but the loop for
    two warpgroups past o8's rows, which do not hold them -- and the loop
    on tcgen05, which a GB200 measured faster (o8 11.39 against 13.33);
    either as asked."""
    from tensorforge.common.options import parse_matrix_groups, registry
    assert registry()['matrix_groups'].default == 'auto'
    assert [parse_matrix_groups(t) for t in ('auto', 'loop', 'Unrolled')] == [
        'auto', 'loop', 'unrolled']
    with pytest.raises(ValueError):
        parse_matrix_groups('rolled')
    assert _group(8, 'sm_100a').section.tile.groups == 'loop'
    assert _group(8, 'sm_90a').section.tile.groups == 'unrolled'
    for engine in ('wgmma', 'tcgen05'):
        for asked in ('loop', 'unrolled'):
            context = me._context('sm_90a', 's', multi_element='force',
                                  **{**HIGH, 'matrix_groups': asked})
            assert eg.tile_groups(engine, context) == asked
    assert _group(8, 'sm_90a', matrix_groups='loop').section.tile.groups == (
        'loop')
    # Two warpgroups past o8 the loop: the blocks written out do not fit the
    # registers ptxas leaves ten warps; one warpgroup takes them written out.
    assert eg.TILE_UNROLLED_ROWS == 128
    context = me._context('sm_90a', 's', multi_element='force', **HIGH)
    assert [eg.tile_groups('wgmma', context, tiles, rows)
            for tiles, rows in ((2, 120), (2, 128), (2, 220), (1, 364))] == [
        'unrolled', 'unrolled', 'loop', 'unrolled']
    assert [_group(o, 'sm_90a', matrix_input_chunks=32).section.tile.groups
            for o in (10, 12)] == ['loop', 'loop']
    # As the options leave them, I whole: one warpgroup, written out.
    assert [_group(o, 'sm_90a').section.tile.groups for o in (10, 12)] == [
        'unrolled', 'unrolled']
    assert _group(10, 'sm_90a', matrix_consumers=1).section.tile.groups == (
        'unrolled')


@pytest.mark.parametrize('arch', ['sm_90a', 'sm_100a'])
@pytest.mark.parametrize('order', [4, 6, 8])
def test_the_group_loop_takes_one_width_a_block(arch, order):
    """In the group loop a block's products are one width, its key in the
    dispatch: the union of their kept spans -- every entry still in its
    product's piece of its block -- the first block whole."""
    section, step, chunks = _chunks(order, arch, matrix_groups='loop')
    assert section.tile.groups == 'loop'
    stages = eg._step_stages(section, step)
    for chunk in chunks:
        assert len(chunk) == len(step.products)
        assert len({piece[3:] for piece in chunk}) == 1
    assert chunks[0][0][3:] == (0, stages[0].width)
    _, _, apart = _chunks(order, arch, matrix_groups='unrolled')
    for loop, unrolled in zip(chunks, apart):
        lo = min(piece[3] for piece in unrolled)
        assert all(piece[3] == lo for piece in loop)


def test_the_kernel_walks_the_blocks_in_one_loop():
    """o8 in the group loop: each tile's eleven blocks one rolled loop,
    their widths a table of the generated file read once a block, one
    dispatch on them: six widths, a branch each -- the widths 40 to 120 as
    the spans give them; tcgen05's as the options leave it."""
    section, step, chunks = _chunks(8, matrix_groups='loop')
    src = _kernel(8, matrix_groups='loop').get_kernel()
    assert len(re.findall(r'#pragma unroll 1\s*for \(int32_t \w+ = 0; '
                          r'\w+ < 11; \+\+\w+\)', src)) == 2
    (table,) = re.findall(r'const int32_t (tf_tw_\w+)\[11\] = \{([^}]*)\}',
                          src)
    eighths = [int(x) for x in table[1].split(',')]
    assert eighths == [(chunk[0][4] - chunk[0][3]) // 8 for chunk in chunks]
    assert sorted(set(eighths)) == [5, 9, 11, 13, 14, 15]
    assert src.count(f'{table[0]}[') == 2 + 1
    # tcgen05: the star role's band loop -- in a loop of I's chunks
    # (`TileRows.chunk`) -- and the mma role's, one block's nine products,
    # N and D's first column read off two tables.
    section, step, chunks = _chunks(8, 'sm_100a')
    src = _kernel(8, 'sm_100a').get_kernel()
    assert src.count('#pragma unroll 1') == 3
    assert _kernel(8, 'sm_100a', matrix_input_chunks='whole').get_kernel(
        ).count('#pragma unroll 1') == 2
    assert src.count('tcgen05.mma.cta_group') == 9
    spans = [chunk[0][3:] for chunk in chunks]
    for prefix, numbers in (('tl', [lo for lo, _ in spans]),
                            ('tn', [hi - lo for lo, hi in spans])):
        (found,) = re.findall(rf'const int32_t tf_{prefix}_\w+\[11\] = '
                              rf'\{{([^}}]*)\}}', src)
        assert [int(x) for x in found.split(',')] == numbers


@pytest.mark.parametrize('output', ['bulk', 'box'])
def test_tcgen05_s_epilogue_by_bulk_reductions(output):
    """`matrix_output` asked on tcgen05: the tile's epilogue stages D in
    shared memory and adds it by bulk reductions -- a column, or an
    element's box, a reduction -- no reduction a cell; the box counted in
    the plan's shared memory, at o8 beside two slots of the inputs' ring
    with I in chunks, one with I whole, where two do not fit.  wgmma's tile
    takes `bulk` a row of D a reduction, and declines the box, saying so."""
    whole = _group(8, 'sm_100a', matrix_output=output,
                   matrix_input_chunks='whole')
    assert whole.section.ring == 1
    group = _group(8, 'sm_100a', matrix_output=output)
    (step,) = [s for s in group.section.steps if s.lowering == 'tcgen05']
    assert step.output == output and group.section.ring == 2
    section = group.section
    plain = dataclasses.replace(section, steps=tuple(
        dataclasses.replace(s, output='') for s in section.steps))
    box = eg.output_volume(section, 14)
    _, _, stride = eg.output_geometry(section, step)
    assert box == 14 * 9 * stride > 0
    assert eg.shared_bytes(section, 14) == eg.shared_bytes(plain, 14) + 4 * box
    gen = _kernel(6, 'sm_100a', matrix_output=output)
    src = gen.get_kernel()
    assert src.count('tensorforge::bulkReduceAdd') == 1
    assert 'atomicAdd' not in src
    what = 'a column' if output == 'bulk' else 'an element'
    assert f'a bulk reduction {what}' in ''.join(
        l for l in src.splitlines() if l.strip().startswith('// tile:'))
    wgmma = _group(8, 'sm_90a', matrix_output=output)
    if output == 'bulk':
        # A row of D a reduction on wgmma (`multi_element._wgmma_bulk`).
        assert any(s.output == 'bulk' for s in wgmma.section.steps)
        return
    assert not any(s.output for s in wgmma.section.steps)
    notes = eg.output_notes(wgmma.section, me._context(
        'sm_90a', 's', multi_element='force',
        **{**HIGH, 'matrix_output': output}))
    assert any('the high-order tile on wgmma adds D a cell at a time, or a '
               'row a reduction (`bulk`)' in n for n in notes)


@pytest.mark.parametrize('output', ['bulk', 'box'])
def test_tcgen05_s_bulk_epilogue_on_the_oracle(output):
    """Three trips with D added by bulk reductions: the box written again
    in each trip behind the reductions' reads of the last, the masked
    elements of the last trip added nowhere -- against NumPy, without a
    race, and bit for bit what a reduction a cell gives."""
    _, red = _on_the_oracle('sm_100a', 4, 31, matrix_output='red')
    _, bulk = _on_the_oracle('sm_100a', 4, 31, matrix_output=output)
    assert (bulk == red).all()


@pytest.mark.parametrize('options', [{}, {'matrix_groups': 'unrolled'},
                                     {'matrix_bands': 3},
                                     {'matrix_output': 'red'}])
def test_three_trips_of_tcgen05_s_layout_on_the_oracle(options):
    """Thirty-one elements, three trips: D's slots in turn, the first
    taken again in the third trip behind the epilogue's release of it, the
    inputs' ring of two slots likewise -- which the oracle's ring order
    checks -- with the group loop, the blocks written out, three bands and
    D added a cell at a time."""
    gen, _ = _on_the_oracle('sm_100a', 4, 31, **options)
    assert gen.built._element_group.section.tile.dslots == 2


def test_two_trips_of_kept_spans_on_tcgen05_on_the_oracle():
    """o6 on tcgen05, seventeen elements: two trips, the second's D written
    fresh over the first's in tensor memory -- which, unlike wgmma's
    accumulators declared zero each trip, holds the last trip's D."""
    _on_the_oracle('sm_100a', 6, 17)


@pytest.mark.parametrize('arch', ['sm_90a', 'sm_100a'])
def test_the_first_block_s_first_piece_is_whole(monkeypatch, arch):
    """The first block's first piece whole whatever its product's own span
    -- its product writes D first, every column fresh, which on tcgen05
    tensor memory holds the last trip's D in otherwise -- here with every
    span of the first block narrowed to a granule: the volume's operators
    have entries on every row of N there, so that only this shows the rule
    (`tile_chunks`)."""
    kept = eg.kept_span

    def narrowed(section, step, product, k0, rows, granule):
        span = kept(section, step, product, k0, rows, granule)
        return (0, granule) if k0 == 0 and span is not None else span
    monkeypatch.setattr(eg, 'kept_span', narrowed)
    section, step, chunks = _chunks(6, arch)
    stages = eg._step_stages(section, step)
    assert chunks[0][0][3:] == (0, stages[chunks[0][0][0]].width)


@pytest.mark.parametrize('arch,order,count,groups', [
    ('sm_90a', 4, 17, 'loop'), ('sm_100a', 6, 14, 'loop'),
    ('sm_100a', 6, 14, 'unrolled')])
def test_kept_and_dense_spans_compute_alike_on_the_oracle(arch, order,
                                                          count, groups):
    """The kept spans leave out products of zeros alone: on the oracle every
    output bit for bit what dense spans give, both against NumPy -- where
    the spans are narrower than the stage: o4 on wgmma (83 % of the dense
    rows of N), two trips, the second's elements masked past the batch; o6
    on tcgen05 (73 %; at o4 its granule of 16 keeps every row), a trip, its
    blocks in the group loop and written out -- there each product's own
    span, at its own N and D's columns from its first."""
    _, kept = _on_the_oracle(arch, order, count, matrix_groups=groups)
    _, dense = _on_the_oracle(arch, order, count, matrix_spans='dense',
                              matrix_groups=groups)
    assert (kept == dense).all()


# -- the schedule (H3d) ----------------------------------------------------------- #

def test_the_schedule_option_and_where_it_is_taken():
    """`tile_schedule`: static by default; clc and auto where the target has
    cluster launch control (sm_100 on: tcgen05's tile, whose input warp
    claims), declined with a note on sm_90a; its shared memory counted."""
    from tensorforge.common.options import parse_tile_schedule, registry
    assert registry()['tile_schedule'].default == 'static'
    assert [parse_tile_schedule(x) for x in ('static', 'CLC', 'auto')] == [
        'static', 'clc', 'auto']
    with pytest.raises(ValueError):
        parse_tile_schedule('dynamic')
    assert _group(6, 'sm_100a').section.tile.schedule == 'static'
    for asked in ('clc', 'auto'):
        group = _group(6, 'sm_100a', tile_schedule=asked)
        assert group.section.tile.schedule == asked
        plain = dataclasses.replace(group.section, tile=dataclasses.replace(
            group.section.tile, schedule='static'))
        assert (eg.shared_bytes(group.section, 14)
                - eg.shared_bytes(plain, 14)) == eg.SCHEDULE_BYTES
    group = _group(8, 'sm_90a', tile_schedule='clc')
    assert group.section.tile.schedule == 'static'
    notes = eg.tile_notes(group, me._context(
        'sm_90a', 's', multi_element='force',
        **{**HIGH, 'tile_schedule': 'clc'}))
    assert ('tile_schedule=clc declined: no cluster launch control on '
            'sm_90a; the static grid') in notes


@pytest.mark.parametrize('asked', ['clc', 'auto'])
def test_tcgen05_s_tile_keeps_the_static_grid_without_launch_control(
        asked, monkeypatch):
    """The schedule asks the target, not the engine: a tcgen05 target
    without cluster launch control (`Target.launch_control` false) keeps
    the static grid -- the input warp would claim with an instruction the
    part has not."""
    from tensorforge.common.target import Target
    monkeypatch.setattr(Target, 'launch_control', lambda self: False)
    context = me._context('sm_100a', 's', multi_element='force',
                          **{**HIGH, 'tile_schedule': asked})
    assert eg.tile_schedule('tcgen05', context) == 'static'
    assert _group(6, 'sm_100a', tile_schedule=asked).section.tile.schedule \
        == 'static'


def test_the_scheduled_kernel_and_its_launch():
    """o6 on tcgen05 under clc: the blocks' loop by the tiles the ring hands
    every warp -- 15 of them at two bands giving each slot back -- one claim
    and one next tile a trip, by the input warp's lane 0, and the launch a
    block a tile; under auto the launcher's choice from two waves of the
    device on; static as before."""
    src = _kernel(6, 'sm_100a', tile_schedule='clc').get_kernel()
    assert src.count('tensorforge::tileClaim(') == 1
    assert src.count('tensorforge::tileNext(') == 1
    assert re.search(r'mbarrierInit\(&tq_ring\[2 \+ r\], 15\)', src)
    assert re.search(r'mbarrierInit\(&tc_ring\[r\], 1\)', src)
    assert re.search(r'while \(true\)', src)
    launch = _kernel(6, 'sm_100a', tile_schedule='clc').get_launcher()
    assert 'config.grid[0] = ((numElements0 + 13) / 14);' in launch
    launch = _kernel(6, 'sm_100a', tile_schedule='auto').get_launcher()
    assert ('config.grid[0] = (((numElements0 + 13) / 14) >= 2 * gridsize ? '
            '((numElements0 + 13) / 14) : std::min(gridsize, '
            '((numElements0 + 13) / 14)));') in launch
    static = _kernel(6, 'sm_100a')
    assert 'tileClaim' not in static.get_kernel()
    assert 'config.grid[0] = std::min(gridsize, numElements0);' in \
        static.get_launcher()


def test_the_schedule_s_ring_hands_over_and_verifies():
    """The tiles' ring: published by the input warp, taken and given back
    by every warp outside the roles -- a ring of the block's -- and every
    pass's `verify` clean."""
    bodies = tro._bodies(order=6, arch='sm_100a', matrix_tile='high',
                         tile_schedule='clc')
    found = []
    for body in bodies:
        found += [d for d in verify(body, strict=False)
                  if "slot's" in d or 'role' in d or 'chain' in d]
    assert not found, found[:3]
    uses = tro._uses(bodies)
    (ring,) = [u for n, u in uses.items() if n.startswith('tq')]
    assert ring.roles('publish') == ring.roles('acquire') == ('input',)
    assert ring.roles('take') == ring.roles('release') == (None,)


@pytest.mark.parametrize('grid,launches', [
    (3, {0: [1, 2]}), (3, {0: [2]}), (3, {}), (2, {}), (1, {})],
    ids=['one-block-claims-all', 'claims-one', 'claims-none', 'stride-2',
         'stride-1'])
def test_the_scheduled_tiles_on_the_oracle(grid, launches):
    """Thirty-one elements, three tiles, the blocks in turn: a block a tile
    whose claims are granted every other block (the blocks it cancelled
    never run), one of them, none; and fewer blocks than tiles, which take
    the static stride through the same ring -- every element against
    NumPy without a race, and bit for bit what the static grid gives."""
    _, static = _on_the_oracle('sm_100a', 4, 31)
    _, got = _on_the_oracle('sm_100a', 4, 31, grid=grid, launches=launches,
                            tile_schedule='clc')
    assert (got == static).all()


def test_the_oracle_refuses_a_block_granted_twice():
    """A script granting one block's claims a CTA twice is no launch the
    hardware makes: the oracle refuses it rather than run a tile twice."""
    from tensorforge.reference import kernel_eval
    with pytest.raises(kernel_eval.Abort, match='ran or was granted before'):
        _on_the_oracle('sm_100a', 4, 31, grid=3, launches={0: [2, 2]},
                       tile_schedule='clc')


# -- orders 10 and 12 (H4) ------------------------------------------------------- #

@pytest.mark.parametrize('order', [10, 12])
def test_the_fixture_s_volume_at_high_orders(order):
    """SeisSol's volume at o10 and o12 from the fixture (`hiorder_fixture`):
    order 8's description at the order's basis -- the functions padded to
    64 bytes, the derivatives' columns the order below's -- each operator on
    its pattern, no value zero, its rows of Q from its first entry to its
    last."""
    import hiorder_fixture as hf
    n, m = hf.basis(order), hf.basis(order - 1)
    assert (n, m) == {10: (220, 165), 12: (364, 286)}[order]
    desc = hf.description(order)
    tensors = {t['name']: t for t in desc['tensors']}
    assert tensors['I']['storage']['shape'] == [-(-n // 16) * 16, 9]
    for d in range(3):
        name = f'kDivM({d})'
        where = hf.pattern(order, name)
        entries = tensors[name]['values']['data']
        assert len(entries) == int(where.sum()) > 0
        assert all(v != 0 for _, v in entries)
        assert not where[:, m:].any()
        rows = np.nonzero(where.any(axis=1))[0]
        (op,) = [o for o in desc['operations'] if o['result']['name'] == 'Q'
                 and o['args'][0]['name'] == name]
        assert op['result']['bbox'] == [[int(rows[0]), 0],
                                        [int(rows[-1]) + 1, 9]]
    assert me._descrs('volume', order, 's') is not None


def test_tensor_memory_s_slots_at_high_orders():
    """D's and A's slots in tensor memory, the most of D's first: o8 two
    and four, o10 one and four, o12 one and two (368 + 4 x 48 = 560 past
    512), past o12 none (the passes)."""
    assert [eg.tile_tmem(n, 3) for n in (128, 224, 368, 464)] == [
        (2, 4), (1, 4), (1, 2), None]


@pytest.mark.parametrize('order,chunks,elements,slots', [
    (10, 32, 14, (1, 4, 4, 1)), (12, 32, 14, (1, 2, 2, 2)),
    (10, 'whole', 14, (1, 4, 3, 1)), (12, 'whole', 7, (1, 2, 2, 1))])
def test_the_tile_on_tcgen05_at_high_orders(order, chunks, elements, slots):
    """o10 and o12 on tcgen05 (N 224 and 368): D one slot, A four and two
    (tensor memory's, the two bands' own alike), B's ring and the inputs'
    -- the search trying B's slots from the options' down, then the
    inputs' -- and at o12 a step wider than an instruction's N taken, its
    products cut at 256.  With I in chunks (`TileRows.chunk`) fourteen
    elements at both, B's ring four slots at o10; with I whole three, and
    at o12 seven elements -- each with D added a cell at a time, asked
    (where `auto` takes I whole at o10, its last chunk 5 rows)."""
    group = _group(order, 'sm_100a', matrix_input_chunks=chunks,
                   matrix_output='red')
    tile = group.section.tile
    assert group.elements == elements
    assert bool(tile.chunk) == (chunks == 32)
    assert (tile.dslots, tile.aslots, group.section.ring_slots,
            group.section.ring) == slots
    # A band its own slots of A's ring: two bands, at o12 on its two.
    assert tile.bands == 2 and tile.output == 'red'


def test_wgmma_takes_i_whole_where_the_tile_fits_it(monkeypatch):
    """`auto` on wgmma: I whole wherever the tile fits it (`tile_plan`), as
    a GH200 measured it faster at every order -- o4 to o8 fourteen elements
    on two warpgroups, o10 and o12 seven on one; where whole fits no group
    of the tile, I in chunks: o12 with 175000 B of the block's shared
    memory, one warpgroup's whole taking 175668 B at the least (B's ring of
    two slots), in chunks two warpgroups 165752.  (tcgen05's rule is its
    own: `test_tcgen05_takes_i_whole_where_its_last_chunk_is_short`.)"""
    plans = {o: _group(o, 'sm_90a').section.tile for o in (4, 6, 8, 10, 12)}
    assert {o: (t.chunk, t.tiles) for o, t in plans.items()} == {
        4: (0, 2), 6: (0, 2), 8: (0, 2), 10: (0, 1), 12: (0, 1)}
    assert _group(12, 'sm_90a').shared_bytes == 211012
    hw = me._context('sm_90a', 's').target.hw
    monkeypatch.setattr(eg, 'SHARED_RESERVE',
                        hw.max_local_mem_size_per_block - 175000)
    group = _group(12, 'sm_90a')
    assert (group.section.tile.chunk, group.section.tile.tiles) == (32, 2)
    assert group.shared_bytes == 165752


def test_tcgen05_takes_i_whole_where_its_last_chunk_is_short(monkeypatch):
    """`auto` on tcgen05: I whole where its last chunk of I in chunks takes
    less than a block of K (`STAR_ROWS`) and whole keeps the elements and
    the epilogue's form -- o6, 35 rows, its last chunk's 3: fourteen
    elements, bulk reductions either way -- as a GB200 measured o6 29 %
    faster and o8 alike (J4b); in chunks at o7 and o8, their last 24 and
    20 rows, at o10, its last 5 but whole a reduction a cell (the staged D
    no room), and at o12, its last 30; o4's I is one chunk; chunks asked for
    taken as asked.  With reductions a cell asked, o10's I whole, as a
    GB200 measured it faster there (J3b: 16.10 ns an element against the
    chunks' 17.04).  With chunks of 40 rows, o8's last 4 taken whole, o12's
    last 6 not: whole takes half the elements."""
    def plans(*orders):
        return {o: (g.section.tile.chunk, g.elements, g.section.tile.output)
                for o, g in ((o, _group(o, 'sm_100a')) for o in orders)}
    assert plans(4, 6, 7, 8, 10, 12) == {
        4: (0, 14, 'bulk'), 6: (0, 14, 'bulk'), 7: (32, 14, 'box'),
        8: (32, 14, 'box'), 10: (32, 12, 'bulk'), 12: (32, 14, 'red')}
    whole = _group(10, 'sm_100a', matrix_input_chunks='whole')
    assert (whole.elements, whole.section.tile.output) == (14, 'red')
    red = _group(10, 'sm_100a', matrix_output='red')
    assert (red.section.tile.chunk, red.elements) == (0, 14)
    assert _group(6, 'sm_100a', matrix_input_chunks=32).section.tile.chunk \
        == 32
    monkeypatch.setattr(eg, 'TILE_CHUNK', 40)
    assert plans(8, 12) == {8: (0, 14, 'box'), 12: (40, 14, 'red')}


@pytest.mark.parametrize('order,chunks,tiles,slots', [
    (10, 32, 2, 3), (12, 32, 2, 3), (10, 'whole', 1, 3),
    (12, 'whole', 1, 3), (10, 'auto', 1, 3), (12, 'auto', 1, 3)])
def test_wgmma_at_high_orders(order, chunks, tiles, slots):
    """o10 and o12 on wgmma: with I in chunks two consumer warpgroups,
    fourteen elements, B's ring at three slots; with I whole one of seven
    -- two warpgroups' I whole does not fit -- which `auto` takes there, as
    the tile fits it (`tile_plan`)."""
    group = _group(order, 'sm_90a', matrix_input_chunks=chunks)
    assert group.section.tile.tiles == tiles
    assert group.elements == 7 * tiles
    assert group.section.ring_slots == slots


@pytest.mark.parametrize('groups', ['loop', 'unrolled'])
def test_products_cut_at_an_instruction_s_n_on_the_oracle(monkeypatch,
                                                          groups):
    """tcgen05's products of a span wider than an instruction's N cut in
    two -- the first 256 columns and the rest, each its own of D's columns
    and B's core matrices that far on -- here at 16 (o4's 32 columns in two
    halves): bit for bit what whole products give, both against NumPy."""
    from tensorforge.backend.instructions.compute import multi_element as mm
    _, whole = _on_the_oracle('sm_100a', 4, 17, matrix_groups=groups)
    monkeypatch.setattr(mm, 'TCGEN05_N_MAX', 16)
    gen, cut = _on_the_oracle('sm_100a', 4, 17, matrix_groups=groups)
    assert gen.get_kernel().count('tcgen05.mma.cta_group') == 2 * (
        9 if groups == 'loop' else 18)
    assert (cut == whole).all()


def test_the_passes_option_and_how_many():
    """`matrix_passes`: auto by default -- on wgmma the fewest whose D a
    consumer thread holds in `TILE_ACCUMULATORS` registers, each at most an
    instruction's 256 rows (o8 and o10 one, o12's 368 rows two) -- forced 1
    to 8; tcgen05 one, D whole in tensor memory."""
    from tensorforge.common.options import parse_matrix_passes, registry
    assert registry()['matrix_passes'].default == 0
    assert [parse_matrix_passes(x) for x in ('auto', '1', '3')] == [0, 1, 3]
    with pytest.raises(ValueError):
        parse_matrix_passes('9')
    assert [_group(o, 'sm_90a').section.tile.passes for o in (8, 10, 12)] == [
        1, 1, 2]
    assert _group(4, 'sm_90a', matrix_passes=3).section.tile.passes == 3
    assert _group(12, 'sm_100a').section.tile.passes == 1


@pytest.mark.parametrize('passes,spans', [(2, 'kept'), (3, 'kept'),
                                          (2, 'dense')])
def test_a_pass_s_pieces_hold_its_rows_of_b(passes, spans):
    """With passes over N each pass's rows `[plo, phi)` in the engine's
    granules, the passes in turn over the stage; a block's piece of a pass
    its span cut to the pass's rows and on wgmma on to `phi`, the pass's
    first block's first piece whole over the pass; every entry of B in
    exactly one piece -- of the pass whose rows hold it -- a block with
    none in a pass no chunk of it; and the chunks' table offsets their
    own, a block's piece of each pass apart."""
    section, step, chunks = _chunks(6, 'sm_90a', matrix_passes=passes,
                                    matrix_spans=spans)
    bounds = eg.tile_pass_bounds(section, step)
    stages = eg._step_stages(section, step)
    assert len(bounds) == passes and bounds[0][0] == 0
    assert bounds[-1][1] == stages[0].width
    assert all(a[1] == b[0] for a, b in zip(bounds, bounds[1:]))
    assert all(lo % 8 == 0 for lo, _ in bounds)
    ranges = eg.tile_pass_chunks(section, step)
    assert sum(count for _, count in ranges) == len(chunks)
    for (first, count), (plo, phi) in zip(ranges, bounds):
        mine = chunks[first:first + count]
        assert mine[0][0][3:] == (plo, phi)
        for chunk in mine:
            for p, k0, rows, lo, hi in chunk:
                assert plo <= lo < hi == phi
    for p, product in enumerate(step.products):
        stage = stages[p]
        (r0, r1), (d0, d1) = stage.window
        pattern = product.first.tensor.nonzero_pattern
        for kk in range(d0, d1):
            for row in range(r0, r1):
                if not pattern.is_nz((row, kk)):
                    continue
                n, k = row - stage.n0, kk - d0
                held = [None for c in chunks for q, k0, rows, lo, hi in c
                        if q == p and k0 <= k < k0 + rows and lo <= n < hi]
                assert len(held) == 1, (p, row, kk)
    _, _, offsets = eg.prepared_table(section, step, 90)
    starts = [offsets[tuple(chunk[0])] for chunk in chunks]
    assert len(set(starts)) == len(starts)


@pytest.mark.parametrize('narrow', ['first', 'last'])
@pytest.mark.parametrize('passes', [2, 3])
def test_a_pass_s_first_piece_is_whole_over_the_pass(monkeypatch, passes,
                                                     narrow):
    """A pass's first piece whole over the pass's rows whatever its
    product's own span -- it writes the pass's D first, every row fresh --
    here with the first block's spans narrowed: from their second granule
    on (the first pass's first piece would start past the pass's first
    row), or to their last granule (the last pass's first piece the first
    block's, narrower than the pass).  The volume's operators have entries
    on every row of N in the first block, so that only this shows the rule
    (`tile_chunks`)."""
    kept = eg.kept_span

    def narrowed(section, step, product, k0, rows, granule):
        span = kept(section, step, product, k0, rows, granule)
        if k0 != 0 or span is None or span[1] - span[0] <= granule:
            return span
        return ((span[0] + granule, span[1]) if narrow == 'first'
                else (span[1] - granule, span[1]))
    monkeypatch.setattr(eg, 'kept_span', narrowed)
    section, step, chunks = _chunks(6, 'sm_90a', matrix_passes=passes,
                                    matrix_spans='kept')
    granule = section.tile.granule
    shown = 0
    for (first, _), (plo, phi) in zip(eg.tile_pass_chunks(section, step),
                                      eg.tile_pass_bounds(section, step)):
        p, k0, rows = chunks[first][0][:3]
        lo, _ = narrowed(section, step, p, k0, rows, granule)
        shown += lo > plo
        assert chunks[first][0][3:] == (plo, phi)
    assert shown


@pytest.mark.parametrize('groups', ['loop', 'unrolled'])
@pytest.mark.parametrize('passes', [2, 3])
def test_passes_compute_what_one_does_on_the_oracle(passes, groups):
    """wgmma's passes over N on the oracle, o4's 24 rows in two and in three
    (one of eight rows): each pass's D from fresh, its blocks' A computed
    again, its D's columns read back from the pass's last row, and a block
    with no entry in a pass skipped -- bit for bit what one pass gives, both
    against NumPy over two trips."""
    _, one = _on_the_oracle('sm_90a', 4, 17, matrix_groups=groups)
    gen, got = _on_the_oracle('sm_90a', 4, 17, matrix_passes=passes,
                              matrix_groups=groups)
    assert gen.built._element_group.section.tile.passes == passes
    assert (got == one).all()


@pytest.mark.parametrize('passes', [2, 3])
@pytest.mark.parametrize('groups', ['loop', 'unrolled'])
def test_a_pass_reads_its_own_rows_of_d_back(passes, groups):
    """Each pass reads back its own rows of D alone -- o4's 24 rows in a
    pass of 8 and one of 16, or three of 8 -- so that over the passes the
    kernel adds every row of D once: as many reductions as one pass's, no
    register of a narrower pass read past its rows into another's."""
    one = _kernel(4, matrix_groups=groups).get_kernel()
    src = _kernel(4, matrix_groups=groups, matrix_passes=passes).get_kernel()
    assert src.count('red.global.add') == one.count('red.global.add') > 0


@pytest.mark.parametrize('groups', ['loop', 'unrolled'])
def test_one_input_slot_through_the_passes_on_the_oracle(monkeypatch,
                                                         groups):
    """The input ring at one slot under two passes over N: the slot given
    back behind the last pass's blocks alone -- every pass reads the trip's
    inputs again -- so that the next trip's inputs land in it only once the
    last pass has read them; over three trips, against NumPy and bit for
    bit what one pass gives, which the oracle's ring order checks."""
    monkeypatch.setattr(eg, 'PRODUCER_SLOTS', ())
    _, one = _on_the_oracle('sm_90a', 4, 31, matrix_groups=groups)
    gen, got = _on_the_oracle('sm_90a', 4, 31, matrix_passes=2,
                              matrix_groups=groups)
    section = gen.built._element_group.section
    assert (section.ring, section.tile.passes) == (1, 2)
    assert (got == one).all()



# -- I in chunks ------------------------------------------------------------------ #

def test_the_input_chunks_option_parses():
    from tensorforge.common.options import (INPUT_CHUNKS_WHOLE,
                                            parse_matrix_input_chunks,
                                            registry)
    assert registry()['matrix_input_chunks'].default == 0
    assert [parse_matrix_input_chunks(t) for t in
            ('auto', 'WHOLE', '8', '32', '128', 0, -1)] == [
        0, INPUT_CHUNKS_WHOLE, 8, 32, 128, 0, INPUT_CHUNKS_WHOLE]
    for text in ('4', '12', '136', 'half'):
        with pytest.raises(ValueError):
            parse_matrix_input_chunks(text)


@pytest.mark.parametrize('arch', ['sm_90a', 'sm_100a'])
def test_i_in_chunks_where_it_has_more_rows_than_one(arch):
    """I in chunks of 32 rows of K (`TILE_CHUNK`) through a ring of two
    slots where it has more rows -- o6's 35 in two chunks, o8's 84 in
    three, the last 20 -- whole at o4's ten under `auto`; `whole` as
    before; forced, o4's ten in chunks of eight, the second two rows.  A
    trip's rounds every chunk in turn."""
    # `auto` takes I whole on wgmma where the tile fits it, and on tcgen05
    # where its last chunk is short of a block of K, o6's (`tile_plan`):
    # the chunks asked for here, at o8 on tcgen05 `auto`'s.
    def asked(order):
        return ({} if arch == 'sm_100a' and order == 8
                else {'matrix_input_chunks': 32})

    def tile(order, **options):
        return _group(order, arch, **{**asked(order), **options}).section.tile
    assert (_group(4, arch).section.tile.chunk,
            _group(4, arch).section.tile.cslots) == (0, 0)
    assert (tile(6).chunk, tile(6).cslots) == (32, eg.TILE_CHUNK_SLOTS) \
        == (32, 2)
    assert tile(8, matrix_input_chunks='whole').chunk == 0
    for order, cuts in ((6, ((0, 32), (32, 35))),
                        (8, ((0, 32), (32, 64), (64, 84)))):
        section = _group(order, arch, **asked(order)).section
        (step,) = [s for s in section.steps
                   if s.lowering in eg.ASYNC_LOWERINGS]
        buffer = eg.chunked_input(section)
        assert buffer.tensor.alias == 'I'
        assert eg.input_chunks(buffer, 32) == cuts
        assert eg.tile_input_rounds(section, step) == tuple(range(len(cuts)))
    forced = _group(4, arch, matrix_input_chunks=8).section
    assert forced.tile.chunk == 8
    assert eg.input_chunks(eg.chunked_input(forced), 8) == ((0, 8), (8, 10))


def test_a_pass_reads_its_own_chunks_again():
    """wgmma's passes over N at o12: the first pass's blocks of K those
    with an entry in its rows -- the first six chunks of I -- the second's
    all nine, fifteen rounds a trip; each block's chunk, the round it reads
    and whether it is the pass's first or last to read it, which take the
    round and give it back, each round exactly once."""
    section = _group(12, 'sm_90a', matrix_input_chunks=32).section
    (step,) = [s for s in section.steps if s.lowering == 'wgmma']
    rounds = eg.tile_input_rounds(section, step)
    assert rounds == tuple(range(6)) + tuple(range(9))
    reads = eg.tile_input_reads(section, step)
    assert len(reads) == len(eg.tile_chunks(section, step))
    assert all(rounds[i] == h for h, i, _, _ in reads)
    assert sorted(i for _, i, take, _ in reads if take) == list(range(15))
    assert sorted(i for _, i, _, give in reads if give) == list(range(15))


@pytest.mark.parametrize('arch', ['sm_90a', 'sm_100a'])
def test_the_chunks_in_the_plan_s_shared_memory(arch):
    """I's ring of chunks counted in the block's shared memory in place of
    I in every slot of the input ring: two slots of the fourteen elements'
    32 rows of its nine columns, and a pair of mbarriers a slot."""
    section = _group(8, arch, **({'matrix_input_chunks': 32}
                                 if arch == 'sm_90a' else {})).section
    whole = dataclasses.replace(section, tile=dataclasses.replace(
        section.tile, chunk=0, cslots=0))
    assert eg.chunked_input(whole) is None
    i = eg.chunked_input(section)
    assert eg.shared_bytes(section, 14) - eg.shared_bytes(whole, 14) == (
        4 * 14 * (2 * 32 * 9 - section.ring * i.per_element) + 2 * 2 * 8)


@pytest.mark.parametrize('arch,consumers', [('sm_90a', 8), ('sm_100a', 8)])
def test_the_chunk_ring_s_text(arch, consumers):
    """o8: I's ring of two slots, full counting the producer's arrival,
    empty a consumer warp's each -- wgmma's two warpgroups' eight, tcgen05's
    two bands' eight; a chunk's copies a column of 32 rows of every
    element, 16128 bytes announced, the last chunk's 20 rows 10080; the
    input ring no bulk copy, the star matrices its alone, and its slot
    given back before the first chunk is waited for -- the star numbers in
    registers by then.  Each element's lane's copies, asked for on tcgen05,
    where `auto` takes the elected lane's (`test_the_input_issue_text`)."""
    src = _kernel(8, arch, **({'matrix_input_chunks': 32}
                              if arch == 'sm_90a'
                              else {'matrix_input_issue': 'lane'})
                  ).get_kernel()
    assert 'mbarrierInit(&ik_ring[r], 1)' in src
    assert f'mbarrierInit(&ik_ring[2 + r], {consumers})' in src
    copies = re.findall(r'bulkCopyOn\(&ik_ring\[\w+\], [^,]+, [^,]+, '
                        r'(\d+), ', src)
    assert sorted(set(copies)) == ['128', '80'] and len(copies) == 3 * 9
    assert re.findall(r'mbarrierExpect\(&ik_ring\[\w+\], (\d+)u', src) == [
        '16128', '16128', '10080']
    assert 'bulkCopyOn(&me_ring' not in src
    assert src.index('mbarrierArrive(&me_ring[') < \
        src.index('mbarrierWait(&ik_ring[')


@pytest.mark.parametrize('arch', ['sm_90a', 'sm_100a'])
@pytest.mark.parametrize('groups', ['loop', 'unrolled'])
def test_i_in_chunks_computes_what_i_whole_does_on_the_oracle(arch, groups):
    """o4's ten rows of I in chunks of eight -- two, the second two rows, a
    copy of four -- over two trips: each chunk's slot taken by its first
    block and given back behind its last, by every consumer warp, and the
    input ring's slot back once the star numbers are read; bit for bit
    what I whole gives, both against NumPy, without a race."""
    _, whole = _on_the_oracle(arch, 4, 17, matrix_groups=groups,
                              matrix_input_chunks='whole')
    gen, got = _on_the_oracle(arch, 4, 17, matrix_groups=groups,
                              matrix_input_chunks=8)
    assert gen.built._element_group.section.tile.chunk == 8
    assert (got == whole).all()


@pytest.mark.parametrize('arch', ['sm_90a', 'sm_100a'])
@pytest.mark.parametrize('groups', ['loop', 'unrolled'])
def test_chunks_of_two_blocks_on_the_oracle(arch, groups):
    """o6's 35 rows of I in chunks of sixteen, two blocks of K a chunk --
    as the default's 32 rows take four: a chunk's slot taken at its first
    block and given back behind its second, the group loop's rounds and
    gives off its tables by the chunk, not the block -- over three trips,
    the chunks' ring of two slots taken again; bit for bit what I whole
    gives, both against NumPy, without a race."""
    _, whole = _on_the_oracle(arch, 6, 31, matrix_groups=groups,
                              matrix_input_chunks='whole')
    gen, got = _on_the_oracle(arch, 6, 31, matrix_groups=groups,
                              matrix_input_chunks=16)
    assert gen.built._element_group.section.tile.chunk == 16
    assert (got == whole).all()


@pytest.mark.parametrize('groups', ['loop', 'unrolled'])
def test_each_pass_s_chunks_on_the_oracle(groups):
    """wgmma's passes over N with I in chunks: o4's 24 rows of N in three
    passes, each streaming its blocks' chunks of eight rows again -- a pass
    whose blocks read one chunk one round -- over two trips, bit for bit
    what I whole gives."""
    _, whole = _on_the_oracle('sm_90a', 4, 17, matrix_groups=groups,
                              matrix_passes=3, matrix_input_chunks='whole')
    gen, got = _on_the_oracle('sm_90a', 4, 17, matrix_groups=groups,
                              matrix_passes=3, matrix_input_chunks=8)
    section = gen.built._element_group.section
    (step,) = [s for s in section.steps if s.lowering == 'wgmma']
    assert section.tile.passes == 3
    assert len(eg.tile_input_rounds(section, step)) > 2
    assert (got == whole).all()


@pytest.mark.parametrize('bands', [2, 3])
@pytest.mark.parametrize('groups', ['loop', 'unrolled'])
def test_every_band_gives_every_chunk_back_on_the_oracle(monkeypatch, bands,
                                                         groups):
    """tcgen05's bands over o6's five blocks with I in chunks of eight, a
    block a chunk -- each chunk one band's to read, every band's to take
    and give back -- over three trips, the input ring at one slot, its
    slot back at once: bit for bit what I whole gives."""
    monkeypatch.setattr(eg, 'PRODUCER_SLOTS', ())
    _, whole = _on_the_oracle('sm_100a', 6, 31, matrix_groups=groups,
                              matrix_bands=bands,
                              matrix_input_chunks='whole')
    gen, got = _on_the_oracle('sm_100a', 6, 31, matrix_groups=groups,
                              matrix_bands=bands, matrix_input_chunks=8)
    section = gen.built._element_group.section
    assert (section.tile.chunk, section.tile.bands, section.ring) == (
        8, bands, 1)
    assert (got == whole).all()


@pytest.mark.parametrize('groups', ['loop', 'unrolled'])
def test_the_star_slot_back_at_once_on_the_oracle(monkeypatch, groups):
    """wgmma with the input ring at one slot and I in chunks: the next
    trip's star matrices staged as soon as every consumer warp has read
    this trip's, under its blocks -- over three trips, bit for bit what I
    whole gives."""
    monkeypatch.setattr(eg, 'PRODUCER_SLOTS', ())
    _, whole = _on_the_oracle('sm_90a', 4, 31, matrix_groups=groups,
                              matrix_input_chunks='whole')
    gen, got = _on_the_oracle('sm_90a', 4, 31, matrix_groups=groups,
                              matrix_input_chunks=8)
    assert gen.built._element_group.section.ring == 1
    assert (got == whole).all()


# -- A's slots, each band's own ------------------------------------------------- #

def _a_slots(src):
    """The slots of a kernel's ring of A in tensor memory: its mbarriers set
    up, a full side of a band's 128 threads each."""
    (count,) = re.findall(r'for \(int r = 0; r < (\d+); \+\+r\) \{\s*'
                          r'tensorforge::mbarrierInit\(&ta\d+_ring\[r\], 128\)',
                          src)
    return int(count)


@pytest.mark.parametrize('bands,slots', [(3, 3), (2, 4), (1, 4), (4, 4)])
def test_the_star_bands_own_their_slots_of_a(bands, slots):
    """tcgen05's tile at o8: A's ring in tensor memory as many slots as its
    bands own alike -- three of the four at three bands -- so that block
    `j`'s slot `j % slots` is band `j % bands`'s and every round of a slot
    one band's.  Without it a band asks for a slot's round two phases
    ahead of the slot's empty side, which a wait on the parity passes: the
    hardware race of GB200's J2 (three bands, the default then, wrong and
    hung where two bands were right)."""
    src = _kernel(8, 'sm_100a', matrix_bands=bands).get_kernel()
    assert _a_slots(src) == slots


@pytest.mark.parametrize('blocks', [5, 11])
def test_the_ring_model_sees_the_tile_s_race(blocks):
    """The protocol model of A's ring (`ring_model`), each warp its own
    agent under seeded random schedules, the bands never meeting -- the
    tile's star bands run a trip ahead of each other on two input slots:
    three bands on four slots hand the products another block's A, on
    three slots never; nor two bands on four.  At o6's five blocks and
    o8's eleven."""
    import ring_model
    assert ring_model.violations(3, 4, blocks, sync='none') > 0
    assert ring_model.violations(3, 3, blocks, sync='none') == 0
    assert ring_model.violations(2, 4, blocks, sync='none') == 0


# -- the bulk epilogue on wgmma ------------------------------------------------------ #

@pytest.mark.parametrize('order', [8, 12])
def test_wgmma_s_bulk_epilogue_plan(order):
    """`matrix_output` `bulk` on wgmma's tile: a row of D -- an element's
    column of the output -- a bulk reduction, a pass's rows of it out to
    sixteen bytes, staged in the warpgroup's slots of A, free behind the
    pass's last wait: no shared memory past what `red` takes.  o8: rows 0
    to 120, 124 floats apart (an odd number of sixteen bytes), eight rows a
    warp a round; o12's two passes each its own rows."""
    red = _group(order)
    group = _group(order, matrix_output='bulk')
    section = group.section
    (step,) = [s for s in section.steps if s.lowering == 'wgmma']
    assert step.output == 'bulk'
    assert group.shared_bytes == red.shared_bytes
    assert eg.output_volume(section, group.elements) == 0
    geometry = [eg.wgmma_bulk_geometry(section, step, lo, hi)
                for lo, hi in eg.tile_pass_bounds(section, step)]
    assert geometry == ([(0, 120, 124, 8)] if order == 8 else
                        [(0, 188, 188, 8), (184, 364, 180, 8)])


def test_wgmma_s_bulk_epilogue_text():
    """o8: a bulk reduction a round of a warp's rows -- two tiles, the
    rows' two halves of eight, one round each -- no reduction a cell; each
    round fenced for the async proxy behind the rows' stores, and the
    warpgroup meeting behind the last before the next blocks of A."""
    src = _kernel(8, matrix_output='bulk').get_kernel()
    assert src.count('tensorforge::bulkReduceAdd') == 2 * 2
    assert 'red.global.add.f32' not in src
    lines = src.splitlines()

    def at(text):
        return [k for k, line in enumerate(lines) if text in line]
    stores = [k for k in at('_astar') if '] = ' in lines[k]]
    for reduce in at('tensorforge::bulkReduceAdd('):
        # The round's stores, then the fence, then the reduction: the
        # async proxy reads the slots only behind a fence of them.
        store = max(k for k in stores if k < reduce)
        fences = [k for k in at('fenceProxyAsync') if store < k < reduce]
        assert fences, (store, reduce)
    red = _kernel(8).get_kernel()
    assert 'tensorforge::bulkReduceAdd' not in red
    assert 'red.global.add.f32' in red


@pytest.mark.parametrize('groups', ['loop', 'unrolled'])
@pytest.mark.parametrize('count,passes', [(17, 1), (31, 1), (17, 3)])
def test_wgmma_s_bulk_epilogue_on_the_oracle(groups, count, passes):
    """D by bulk reductions a row on wgmma's tile against a reduction a cell
    over o4's elements -- two and three trips, the last's elements past the
    batch masked, and in three passes over N -- bit for bit, both against
    NumPy, without a race: the slots of A staged and read again behind the
    reductions' reads."""
    options = dict(matrix_groups=groups)
    if passes > 1:
        options['matrix_passes'] = passes
    _, red = _on_the_oracle('sm_90a', 4, count, **options)
    gen, bulk = _on_the_oracle('sm_90a', 4, count, matrix_output='bulk',
                               **options)
    (step,) = [s for s in gen.built._element_group.section.steps
               if s.lowering == 'wgmma']
    assert step.output == 'bulk'
    assert (bulk == red).all()


@pytest.mark.parametrize('rows', [1, 2, 4])
def test_wgmma_s_bulk_epilogue_in_rounds_on_the_oracle(monkeypatch, rows):
    """A warp's rows of D staged `rows` at a time, as where A's slots hold
    no more -- past the orders the volume reaches, so forced here
    (`wgmma_bulk_geometry`): each round's quads alone staging, at their
    place in the round, the slots handed on by the warp's barrier -- bit for
    bit what a reduction a cell gives over two trips, against NumPy without
    a race, a reduction a round of each half."""
    geometry = eg.wgmma_bulk_geometry

    def fewer(section, step, plo, phi):
        lo, hi, stride, _ = geometry(section, step, plo, phi)
        return lo, hi, stride, rows
    _, red = _on_the_oracle('sm_90a', 4, 17, matrix_groups='loop')
    monkeypatch.setattr(eg, 'wgmma_bulk_geometry', fewer)
    gen, bulk = _on_the_oracle('sm_90a', 4, 17, matrix_output='bulk',
                               matrix_groups='loop')
    assert gen.get_kernel().count('tensorforge::bulkReduceAdd(') == (
        2 * 2 * (8 // rows))
    assert (bulk == red).all()


# -- the mma warp's issue (H3e) ------------------------------------------------ #

def test_the_issue_options_parse():
    from tensorforge.common.options import (parse_matrix_issue,
                                            parse_matrix_slots,
                                            parse_tile_tables, registry)
    assert registry()['matrix_issue'].default == 'thread'
    assert registry()['tile_tables'].default == 'device'
    assert registry()['matrix_slots'].default == 'round'
    assert [parse_matrix_slots(t) for t in ('round', 'TRIP')] == [
        'round', 'trip']
    with pytest.raises(ValueError):
        parse_matrix_slots('block')
    from tensorforge.common.options import (parse_matrix_handoffs,
                                            parse_matrix_wait)
    assert registry()['matrix_handoffs'].default == 'two'
    assert registry()['matrix_wait'].default == 'try'
    assert [parse_matrix_handoffs(t) for t in ('two', 'One')] == [
        'two', 'one']
    assert [parse_matrix_wait(t) for t in ('try', 'Suspend', 'test')] == [
        'try', 'suspend', 'test']
    for parse, text in ((parse_matrix_handoffs, 'three'),
                        (parse_matrix_wait, 'spin')):
        with pytest.raises(ValueError):
            parse(text)
    assert [parse_matrix_issue(t) for t in ('thread', 'ELECT')] == [
        'thread', 'elect']
    assert [parse_tile_tables(t) for t in ('device', 'Constant')] == [
        'device', 'constant']
    with pytest.raises(ValueError):
        parse_matrix_issue('warp')
    with pytest.raises(ValueError):
        parse_tile_tables('shared')


@pytest.mark.parametrize('order', [4, 8, 12])
def test_the_issue_the_options_ask_for(order):
    """`elect` on tcgen05's tile; on wgmma, whose products are the
    warpgroup's, declined, as its comment says; `thread` as the options
    leave it."""
    assert _group(order, 'sm_100a').section.tile.issue == 'thread'
    assert _group(order, 'sm_100a',
                  matrix_issue='elect').section.tile.issue == 'elect'
    wgmma = _group(order, 'sm_90a', matrix_issue='elect')
    assert wgmma.section.tile.issue == 'thread'
    notes = eg.tile_notes(wgmma, me._context(
        'sm_90a', 's', multi_element='force',
        **{**HIGH, 'matrix_issue': 'elect'}))
    assert ('matrix_issue=elect declined: wgmma\'s products are the '
            'warpgroup\'s') in notes


def test_the_tables_the_options_ask_for():
    """`constant` on tcgen05's group loop; where the blocks are written out,
    which read no table, and on wgmma, `device` -- declined there in its
    comment."""
    assert _group(8, 'sm_100a').section.tile.tables == 'device'
    asked = {'tile_tables': 'constant'}
    assert _group(8, 'sm_100a', **asked).section.tile.tables == 'constant'
    assert _group(8, 'sm_100a', matrix_groups='unrolled',
                  **asked).section.tile.tables == 'device'
    wgmma = _group(8, 'sm_90a', matrix_groups='loop', **asked)
    assert wgmma.section.tile.tables == 'device'
    notes = eg.tile_notes(wgmma, me._context(
        'sm_90a', 's', multi_element='force',
        **{**HIGH, 'matrix_groups': 'loop', **asked}))
    assert ('tile_tables=constant declined: tcgen05\'s group loop\'s '
            'alone') in notes


def test_the_loop_s_tables_in_constant_memory():
    """`constant`: the group loop's first columns and widths in constant
    memory under names of their own -- a kernel of the file reading the
    same numbers from global memory names its own tables -- the comment
    saying so; as the options leave them, in global memory."""
    plain = _kernel(8, 'sm_100a').get_kernel()
    src = _kernel(8, 'sm_100a', tile_tables='constant').get_kernel()
    assert re.search(r'^__device__ const int32_t tf_tl_\w+\[11\]', plain,
                     re.M)
    assert not re.search(r'__constant__', plain)
    for prefix in ('tlc', 'tnc'):
        assert len(re.findall(rf'^__constant__ const int32_t tf_{prefix}_\w+'
                              rf'\[11\] = ', src, re.M)) == 1
        assert re.search(rf'\(?tf_{prefix}_\w+\[v\d+_g\]', src)
    assert not re.search(r'\btf_t[ln]_', src)
    assert 'one loop, a dispatch on their width, its tables in constant ' \
           'memory' in src


@pytest.mark.parametrize('order,bands,depth', [(4, 2, 4), (8, 2, 4),
                                               (8, 3, 12), (12, 2, 2)])
def test_the_slots_the_options_ask_for(order, bands, depth):
    """`trip` on tcgen05's tile, the group loop written out a slot of both
    rings at a time -- their slots' least common multiple, A's three of
    three bands against B's four twelve, o12's two of each two; on wgmma,
    whose passes walk B's
    chunks again, `round`, declined in its comment; `round` as the options
    leave it."""
    assert _group(order, 'sm_100a').section.tile.slots == 'round'
    group = _group(order, 'sm_100a', matrix_slots='trip', matrix_bands=bands)
    assert group.section.tile.slots == 'trip'
    assert eg.tile_depth(group.section) == depth
    assert eg.tile_depth(_group(order, 'sm_100a').section) == 1
    wgmma = _group(order, 'sm_90a', matrix_slots='trip')
    assert wgmma.section.tile.slots == 'round'
    notes = eg.tile_notes(wgmma, me._context(
        'sm_90a', 's', multi_element='force',
        **{**HIGH, 'matrix_slots': 'trip'}))
    assert ('matrix_slots=trip declined: wgmma\'s passes walk B\'s chunks '
            'again') in notes


def test_the_ring_model_numbers_b_s_ring_by_the_trip():
    """The protocol model of B's ring (`ring_model.operator_violations`),
    the operator warp and the issuing thread each an agent: numbered by the
    trip as by its rounds over the trips, at o4's two blocks, o6's five and
    o8's eleven on four slots, it hands every chunk over -- and where the
    two sides number it differently, never."""
    import ring_model
    for blocks in (2, 5, 11):
        for numbering in ring_model.NUMBERINGS:
            assert ring_model.operator_violations(4, blocks,
                                                  numbering=numbering) == 0
        assert ring_model.operator_violations(
            4, blocks, numbering=('round', 'trip')) > 0


@pytest.mark.parametrize('order,bands', [(4, 2), (8, 2), (8, 3), (12, 2)])
def test_the_handoffs_the_options_ask_for(order, bands):
    """`one` on tcgen05's tile: the rings numbered by the trip and B's of
    A's slots -- three at three bands, two at o12 --, the group loop a slot
    of them a body; `matrix_wait` as asked there; on wgmma, whose
    warpgroups each fill their own A, `two` and `try`, declined in its
    comment; as the options leave them, `two` and `try`."""
    tile = _group(order, 'sm_100a').section.tile
    assert (tile.handoffs, tile.wait) == ('two', 'try')
    group = _group(order, 'sm_100a', matrix_handoffs='one',
                   matrix_wait='suspend', matrix_bands=bands)
    tile = group.section.tile
    assert (tile.handoffs, tile.slots, tile.wait) == ('one', 'trip',
                                                      'suspend')
    assert group.section.ring_slots == tile.aslots
    assert eg.tile_depth(group.section) == tile.aslots
    asked = {'matrix_handoffs': 'one', 'matrix_wait': 'test'}
    wgmma = _group(order, 'sm_90a', **asked)
    assert (wgmma.section.tile.handoffs, wgmma.section.tile.wait) == (
        'two', 'try')
    notes = eg.tile_notes(wgmma, me._context(
        'sm_90a', 's', multi_element='force', **{**HIGH, **asked}))
    assert ('matrix_handoffs=one declined: wgmma\'s warpgroups each fill '
            'their own A') in notes
    assert 'matrix_wait=test declined: tcgen05\'s mma warp\'s alone' in notes


@pytest.mark.parametrize('groups', ['loop', 'unrolled'])
def test_one_ring_for_a_and_b_text(groups):
    """`one` at o8: no ring of A's own -- B's ring its slots, its full
    barrier set up for the band's 128 threads and the operator warp's
    arrival, its empty one for the products' one commit; the issuing
    thread's one wait and one commit a block, onto that ring; the star
    bands' waits and arrivals on it; the comment says so."""
    src = _kernel(8, 'sm_100a', matrix_groups=groups,
                  matrix_handoffs='one').get_kernel()
    assert 'ta3_ring' not in src
    assert re.search(r'mbarrierInit\(&bk3_ring\[r\], 129\)', src)
    assert re.search(r'mbarrierInit\(&bk3_ring\[4 \+ r\], 1\)', src)
    waits = re.findall(r'mbarrierWait\(&bk3_ring\[(?!4 \+)', src)
    commits = re.findall(r'tcgen05CommitOn\(&bk3_ring\[', src)
    if groups == 'loop':
        # Four bodies a trip of the loop, a wait and a commit each.
        assert (len(waits), len(commits)) == (4, 4)
    else:
        assert (len(waits), len(commits)) == (11, 11)
    assert re.search(r'mbarrierArrive\(&bk3_ring\[', src)
    assert ('A\'s ring and B\'s one, block j its slot j modulo the slots '
            'every trip, a wait and a commit a block') in src


@pytest.mark.parametrize('groups', ['loop', 'unrolled'])
def test_one_ring_hands_over_between_its_roles(groups):
    """`one` at o6: B's ring the one ring -- its description naming the
    star bands and the operator warp its producers, each acquiring and
    publishing its slots, the operator warp's bulk copies into them, the
    mma warp taking them and committing them back -- no ring of A's own,
    and every pass's `verify` clean (`rings.refusal`: no act of the ring
    out of its roles)."""
    bodies = tro._bodies(order=6, arch='sm_100a', matrix_tile='high',
                         matrix_groups=groups, matrix_handoffs='one')
    found = []
    for body in bodies:
        found += [d for d in verify(body, strict=False)
                  if "slot's" in d or 'role' in d or 'chain' in d]
    assert not found, found[:3]
    uses = tro._uses(bodies)
    assert not [name for name in uses if name.startswith('ta')]
    (ring,) = [u for name, u in uses.items() if name.startswith('bk')]
    assert ring.spec.producers == ('star', 'operator')
    assert ring.spec.consumers == ('mma',)
    assert set(ring.roles('acquire')) == set(ring.roles('publish')) == {
        'star', 'operator'}
    assert ring.roles('copy') == ('operator',)
    assert ring.roles('take') == ring.roles('release') == ('mma',)


@pytest.mark.parametrize('groups', ['loop', 'unrolled'])
def test_one_ring_s_wait_makes_b_s_slot_visible(monkeypatch, groups):
    """`one` at o8: the issuing thread's one wait a block, on B's ring,
    declares B's slot written behind it (`IRBuilder.ring_take`'s `writes`)
    as B's own wait does with two rings -- so that no read of B moves
    ahead of it."""
    from tensorforge.backend.pir.build import IRBuilder
    take = IRBuilder.ring_take
    taken = {}

    def recorded(self, ring, slot, parity, *, writes=(), **kwargs):
        taken.setdefault(ring.name, set()).add(
            tuple(sorted(x.hint for x in writes)))
        return take(self, ring, slot, parity, writes=writes, **kwargs)
    monkeypatch.setattr(IRBuilder, 'ring_take', recorded)
    seen = {}
    for handoffs in ('two', 'one'):
        taken.clear()
        _kernel(8, 'sm_100a', matrix_groups=groups,
                matrix_handoffs=handoffs).get_kernel()
        seen[handoffs] = dict(taken)
    assert 'ta3' in seen['two'] and 'ta3' not in seen['one']
    (written,) = seen['two']['bk3']
    assert written and seen['one']['bk3'] == {written}


@pytest.mark.parametrize('groups', ['loop', 'unrolled'])
@pytest.mark.parametrize('handoffs', ['two', 'one'])
def test_the_wait_forms_text(groups, handoffs):
    """`suspend` and `test` at o8, the mma warp's waits alone: A's slot's
    and B's -- or the one ring's -- a block, the group loop one body a trip
    of it by the rounds, four by the trip, and D's slot back; each other
    wait of the kernel `try_wait` again; the comment says so."""
    options = dict(matrix_groups=groups, matrix_handoffs=handoffs)
    plain = _kernel(8, 'sm_100a', **options).get_kernel().count(
        'mbarrierWait(')
    bodies = {'loop': 1 if handoffs == 'two' else 4, 'unrolled': 11}[groups]
    for form, name in (('suspend', 'mbarrierWaitSuspend'),
                       ('test', 'mbarrierWaitTest')):
        text = _kernel(8, 'sm_100a', matrix_wait=form,
                       **options).get_kernel()
        mma = text.count(f'tensorforge::{name}(')
        assert mma == bodies * (2 if handoffs == 'two' else 1) + 1
        assert text.count('mbarrierWait(') == plain - mma
        assert ('waiting by try_wait with a suspend-time hint'
                if form == 'suspend' else 'waiting by test_wait in a spin') \
            in text


def test_a_ring_s_wait_by_a_form_of_its_own():
    """A ring's wait in a form of `engines.RING_WAIT_FORMS` carries it to
    the lowering (`engines.ring_wait`), none by default; any other form
    refused, as the options' parse refuses it."""
    from tensorforge.backend.pir import engines
    from tensorforge.backend.pir.build import IRBuilder
    from tensorforge.backend.pir.core import IRError
    from tensorforge.common.context import Context
    b = IRBuilder(fptype=Datatype.F32, arena='shrMem', context=Context(
        arch='sm_100a', backend='cuda', fp_type=Datatype.F32))
    ring = b.mbarrier_ring('pc', 2, full=1, empty=1,
                           leader='threadIdx.x == 0')
    assert [b.ring_take(ring, 0, 0, form=form).attr('form')
            for form in (None, 'suspend', 'test')] == [None, 'suspend',
                                                       'test']
    assert b.ring_acquire(ring, 0, 0, form='test').attr('form') == 'test'
    with pytest.raises(IRError):
        b.ring_take(ring, 0, 0, form='spin')


def test_the_ring_model_shares_a_s_ring_and_b_s():
    """The protocol model of A's ring and B's as one (`ring_model.
    shared_violations`) -- a slot's full barrier the band's 128 threads and
    the operator warp's arrival with B's bytes, its empty one the products'
    one commit, which both producers wait for -- hands every block over
    wherever A's ring alone does, the bands owning their slots (two bands
    on two or four, three on three, one on four), and races where it does:
    three bands on four slots, two on three.  At o6's five blocks and o8's
    eleven, the bands never meeting."""
    import ring_model
    for blocks in (5, 11):
        for bands, slots in ((1, 4), (2, 4), (2, 2), (3, 3)):
            assert ring_model.shared_violations(bands, slots, blocks) == 0
        for bands, slots in ((3, 4), (2, 3)):
            assert ring_model.shared_violations(bands, slots, blocks) > 0
            assert ring_model.violations(bands, slots, blocks,
                                         sync='none') > 0


@pytest.mark.parametrize('groups', ['loop', 'unrolled'])
def test_the_slots_by_the_trip_text(groups):
    """`trip` at o8: the issuing thread's every wait and commit of A's and
    B's rings at a slot that is a number -- the group loop four bodies a
    trip of it, the last one's past the eleventh block guarded -- and B's
    producer alike, its waits, copies and bytes; the comment says so.  (A's
    producers, the star bands, take their slots by their band.)"""
    src = _kernel(8, 'sm_100a', matrix_groups=groups,
                  matrix_slots='trip').get_kernel()
    number = r'(?:4 \+ \()?\d+(?:_i32)?\)?'
    taken = (re.findall(r'mbarrierWait\(&ta3_ring\[(?!4 \+)([^]]*)\]', src)
             + re.findall(r'tcgen05CommitOn\(&ta3_ring\[([^]]*)\]', src)
             + re.findall(r'(?:mbarrierWait|tcgen05CommitOn|bulkCopyOn|'
                          r'mbarrierExpect)\(&bk3_ring\[([^]]*)\]', src))
    assert taken and all(re.fullmatch(number, x) for x in taken), taken
    if groups == 'loop':
        assert re.search(r'for \(int32_t (v\d+_g) = 0; \1 < 11; \1 \+= 4\)',
                         src)
        assert len(re.findall(r'if \(v\d+_g < 11\)', src)) == 1
        assert 'its body 4 times, a slot of the rings\' each' in src
    assert ('fills B\'s ring, block j its slot j modulo the slots every '
            'trip, as A\'s') in src


def _region(src: str, head: int) -> str:
    """The text of the braces opened at `src[head]`, string literals
    skipped (the asm templates hold braces of their own)."""
    depth, j = 0, head
    while True:
        c = src[j]
        if c == '"':
            j += 1
            while src[j] != '"':
                j += 2 if src[j] == '\\' else 1
        elif c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                return src[head:j + 1]
        j += 1


@pytest.mark.parametrize('groups', ['loop', 'unrolled'])
def test_the_elect_issue_text(groups):
    """`elect`: the mma warp's lanes elect one at the role's head -- the
    kernel's one elect -- and the trip in one branch on it: D's slot, every
    block's waits, products and commits, B's releases and D's commit, each
    under that lane's predicate; the comment says so.  As the options leave
    it, no elect -- the input warp's copies each element's lane's, asked
    for, where `auto` elects one of its lanes as well
    (`tile_input_issue`)."""
    assert 'electOne' not in _kernel(8, 'sm_100a', matrix_groups=groups,
                                     matrix_input_issue='lane').get_kernel()
    src = _kernel(8, 'sm_100a', matrix_groups=groups, matrix_issue='elect',
                  matrix_input_issue='lane').get_kernel()
    assert src.count('tensorforge::electOne()') == 1
    head = re.search(r'bool (v\d+_elect) = tensorforge::electOne\(\);\s*'
                     r'if \(\1\) \{', src)
    assert head is not None
    elect = head.group(1)
    branch = _region(src, head.end() - 1)
    products = src.count('tcgen05.mma.cta_group')
    assert products and branch.count('tcgen05.mma.cta_group') == products
    assert branch.count(f'"r"(static_cast<uint32_t>({elect}))') == products
    commits = re.findall(r'tensorforge::tcgen05CommitOn\(&\w+\[[^]]*\], '
                         r'(\w+)\)', src)
    assert commits and set(commits) == {elect}
    assert len(re.findall(r'tensorforge::tcgen05CommitOn\(', branch)) == len(
        commits)
    assert re.search(rf'if \({elect}\) \{{\s*tensorforge::mbarrierWait\('
                     rf'&td_ring', branch)
    assert ('warp 14 issues -- the lane elect.sync picks, the trip in one '
            'branch of its own') in ''.join(
        l for l in src.splitlines() if l.strip().startswith('// tile:'))


def test_the_oracle_elects_a_converged_warp_s_lowest_lane():
    """`tensorforge::electOne()` on the oracle: true in the lowest lane of
    each warp the run brings to it whole; a warp short of a lane there --
    which the full mask does not allow -- an `Abort`."""
    from tensorforge.reference import kernel_eval as ke
    lanes = []
    for t in range(64):
        lane = ke.Interp(ke.Slot(), {})
        lane.lane = t
        lanes.append(lane)
    run = ke.Lockstep(lanes)
    run.current, run.wave_cache = lanes, {}
    assert [it.lane for it in lanes
            if it.ev('tensorforge::electOne()')] == [0, 32]
    run.current, run.wave_cache = lanes[32:], {}
    assert [it.lane for it in lanes[32:]
            if it.ev('tensorforge::electOne()')] == [32]
    run.current, run.wave_cache = lanes[1:32], {}
    with pytest.raises(ke.Abort, match='elect.sync in 31 lanes of warp 0'):
        lanes[1].ev('tensorforge::electOne()')


@pytest.mark.parametrize('order,groups,tables', [
    (4, 'loop', 'device'), (4, 'unrolled', 'device'), (6, 'loop', 'device'),
    (6, 'unrolled', 'device'), (6, 'loop', 'constant')])
def test_the_elect_issue_on_the_oracle(order, groups, tables):
    """`elect` over three trips, o4's two blocks and o6's five, the loop and
    written out, and the loop's tables in constant memory: against NumPy
    without a race, and bit for bit what the first lane's issue gives --
    the lane elect.sync picks is the warp's lowest, which the oracle checks
    is all there."""
    _, thread = _on_the_oracle('sm_100a', order, 31, matrix_groups=groups)
    _, elect = _on_the_oracle('sm_100a', order, 31, matrix_groups=groups,
                              matrix_issue='elect', tile_tables=tables)
    assert (elect == thread).all()


def test_the_elect_issue_over_o8_s_blocks_on_the_oracle():
    """`elect` at o8, its eleven blocks over two trips of the group loop,
    against NumPy without a race."""
    gen, _ = _on_the_oracle('sm_100a', 8, 17, matrix_issue='elect')
    assert gen.built._element_group.section.tile.issue == 'elect'


@pytest.mark.parametrize('order,groups,options', [
    (4, 'loop', {}), (4, 'unrolled', {}), (6, 'loop', {}),
    (6, 'unrolled', {}), (6, 'loop', {'matrix_bands': 3}),
    (6, 'loop', {'matrix_issue': 'elect', 'tile_tables': 'constant'})])
def test_the_slots_by_the_trip_on_the_oracle(order, groups, options):
    """`trip` over three trips, o4's two blocks and o6's five, the loop --
    four bodies a trip of it, twelve at three bands -- and written out,
    with the elected lane's issue and the tables in constant memory too:
    against NumPy without a race, and bit for bit what B's rounds over the
    trips give."""
    _, rounds = _on_the_oracle('sm_100a', order, 31, matrix_groups=groups,
                               **{k: v for k, v in options.items()
                                  if k == 'matrix_bands'})
    _, trip = _on_the_oracle('sm_100a', order, 31, matrix_groups=groups,
                             matrix_slots='trip', **options)
    assert (trip == rounds).all()


@pytest.mark.parametrize('order,groups,options', [
    (4, 'loop', {}), (4, 'unrolled', {}), (6, 'loop', {}),
    (6, 'unrolled', {'matrix_wait': 'test'}),
    (6, 'loop', {'matrix_bands': 3, 'matrix_wait': 'suspend'}),
    (6, 'loop', {'matrix_issue': 'elect', 'tile_tables': 'constant'})])
def test_one_ring_for_a_and_b_on_the_oracle(order, groups, options):
    """`one` over three trips, o4's two blocks and o6's five, the loop and
    written out, three bands, the wait forms, and the elected lane's issue
    with the tables in constant memory: against NumPy without a race --
    the oracle's barriers count the band's arrivals and the operator
    warp's with its bytes alike -- and bit for bit what two rings give."""
    _, two = _on_the_oracle('sm_100a', order, 31, matrix_groups=groups,
                            **{k: v for k, v in options.items()
                               if k == 'matrix_bands'})
    _, one = _on_the_oracle('sm_100a', order, 31, matrix_groups=groups,
                            matrix_handoffs='one', **options)
    assert (one == two).all()


def test_the_slots_by_the_trip_over_o8_s_blocks_on_the_oracle():
    """`trip` at o8 with the elected lane's issue and the tables in
    constant memory: eleven blocks over two trips, the loop's last body
    past them guarded, against NumPy without a race."""
    gen, _ = _on_the_oracle('sm_100a', 8, 17, matrix_issue='elect',
                            tile_tables='constant', matrix_slots='trip')
    assert gen.built._element_group.section.tile.slots == 'trip'


# -- I's supply: the input warp's copies ----------------------------------------- #

def test_the_input_issue_option_parses():
    from tensorforge.common.options import (parse_matrix_input_ahead,
                                            parse_matrix_input_issue,
                                            registry)
    assert registry()['matrix_input_issue'].default == 'auto'
    assert registry()['matrix_input_ahead'].default == 'off'
    assert [parse_matrix_input_issue(t) for t in ('Auto', 'lane', 'Elect')] \
        == ['auto', 'lane', 'elect']
    assert [parse_matrix_input_ahead(t) for t in ('off', 'L2')] == [
        'off', 'l2']
    for parse, text in ((parse_matrix_input_issue, 'warp'),
                        (parse_matrix_input_ahead, 'l1')):
        with pytest.raises(ValueError):
            parse(text)


@pytest.mark.parametrize('arch', ['sm_90a', 'sm_100a'])
def test_the_input_issue_the_options_ask_for(arch):
    """`elect` on both engines' tiles -- tcgen05's input warp and wgmma's
    producer copy I alike -- the comment saying so; `lane` as asked; and
    `auto`, as the options leave it, the elected lane on tcgen05, which a
    GB200 measured faster at every order (J5), each element's lane on
    wgmma, which no GH200 has measured -- at o6 and o8 alike."""
    default = 'elect' if arch == 'sm_100a' else 'lane'
    assert [_group(o, arch).section.tile.inputs for o in (6, 8)] == [
        default, default]
    assert _group(8, arch, matrix_input_issue='lane').section.tile.inputs \
        == 'lane'
    group = _group(8, arch, matrix_input_issue='elect')
    assert group.section.tile.inputs == 'elect'
    notes = eg.tile_notes(group, me._context(
        arch, 's', multi_element='force', matrix_input_issue='elect',
        **HIGH))
    assert any('every element\'s bulk copies by the lane elect.sync picks'
               in note for note in notes)


@pytest.mark.parametrize('chunks', [32, 'whole'])
def test_the_input_issue_text(chunks):
    """`elect` at o8 on tcgen05: the input warp elects a lane at the input
    ring's trip and, I in chunks, at its chunks' -- every lane of the warp
    there -- and
    that lane issues every element's bulk copies of I, each under its
    predicate, the elements' pointers loaded by every lane alike, I's
    chunks' again each round: fourteen elements' nine columns a chunk,
    three chunks a trip, where each element's lane issues its own nine
    under its predicate; and the elected lane announces their bytes."""
    lane = _kernel(8, 'sm_100a', matrix_input_chunks=chunks,
                   matrix_input_issue='lane').get_kernel()
    src = _kernel(8, 'sm_100a', matrix_input_chunks=chunks,
                  matrix_input_issue='elect').get_kernel()
    picks = re.findall(r'bool (v\d+_ie) = (v\d+_pw) && (v\d+_ie);', src)
    elections = 2 if chunks == 32 else 1
    assert len(picks) == elections
    assert src.count('tensorforge::electOne()') == elections
    issuers = {p[0] for p in picks}
    ring = 'ik_ring' if chunks == 32 else 'me_ring'
    copies = re.findall(r'if \((v\d+_\w+)\) \{\s*tensorforge::bulkCopyOn\(&'
                        + ring + r'\[', src)
    assert copies and set(copies) <= issuers
    assert len(copies) == 14 * 9 * (3 if chunks == 32 else 1)
    per_lane = re.findall(r'if \(v\d+_mine\) \{\s*tensorforge::bulkCopyOn\(&'
                          + ring + r'\[', lane)
    assert len(per_lane) == 9 * (3 if chunks == 32 else 1)
    assert 'electOne' not in lane
    assert re.search(r'mbarrierExpect\(&' + ring + r'\[[^,]*\], \d+u, ('
                     + '|'.join(issuers) + r')\)', src)
    # Every element's pointer through the batch's clamp -- the group's first
    # element where the element is past the batch's end -- none unclamped.
    clamped = re.findall(r'const float \*const eI_\d+ = &m0\[\(batchIdActive'
                         r'0e\d+ \? \w+ : v\d+_batchIdGroup0\)\]', src)
    assert len(clamped) == 14 * (3 if chunks == 32 else 1)
    assert not re.search(r'&m0\[v\d+_e\]', src)
    assert ('every element\'s bulk copies by the lane elect.sync picks'
            in src)


@pytest.mark.parametrize('arch', ['sm_90a', 'sm_100a'])
def test_the_input_ahead_the_options_ask_for(arch):
    """`l2` on both engines' tiles where the blocks stride over the batch,
    the comment saying so; declined where tcgen05's tiles go by claims, the
    next one no stride on; `off` as the options leave it."""
    assert _group(8, arch).section.tile.ahead == 'off'
    group = _group(8, arch, matrix_input_ahead='l2')
    assert group.section.tile.ahead == 'l2'
    assert any('the next trip\'s I asked of L2 a trip ahead' in note
               for note in eg.tile_notes(group, me._context(
                   arch, 's', multi_element='force',
                   matrix_input_ahead='l2', **HIGH)))
    if arch == 'sm_100a':
        asked = {'matrix_input_ahead': 'l2', 'tile_schedule': 'clc'}
        claimed = _group(8, arch, **asked)
        assert claimed.section.tile.ahead == 'off'
        assert ('matrix_input_ahead=l2 declined: the tiles go by claims, '
                'the next one no stride on') in eg.tile_notes(
                    claimed, me._context(arch, 's', multi_element='force',
                                         **{**HIGH, **asked}))


@pytest.mark.parametrize('chunks', [32, 'whole'])
def test_the_input_ahead_text(chunks):
    """`l2` at o8 on tcgen05: at the trip's head each element's lane loads
    its element of the next trip's group -- the loop's stride on, clamped
    past the batch's end -- its pointer to I, and at the trip's end asks L2
    for its element's nine columns of I a line at a time, three lines of
    84 rows each, under its predicate, behind the trip's last copies; none
    as the options leave it."""
    off = _kernel(8, 'sm_100a', matrix_input_chunks=chunks).get_kernel()
    src = _kernel(8, 'sm_100a', matrix_input_chunks=chunks,
                  matrix_input_ahead='l2').get_kernel()
    assert 'prefetchL2' not in off
    ahead = re.search(r'size_t (v\d+_gn) = (v\d+_batchIdGroup0) \+ '
                      r'\(gridDim\.x \* 14\);', src)
    assert ahead
    clamp = re.search(r'size_t (v\d+_gn) = \(' + ahead.group(1)
                      + r' < numElements0\) \? ' + ahead.group(1) + r' : '
                      + ahead.group(2) + ';', src)
    assert clamp
    pointer = re.search(r'const float \*const (eI_\d+) = &m0\[\(\w+ \? '
                        r'(v\d+_en) : ' + clamp.group(1) + r'\)\]', src)
    assert pointer
    hints = re.findall(r'if \((v\d+_mine)\) \{\s*tensorforge::prefetchL2'
                       r'\(&' + pointer.group(1) + r'\[(\d+)\]\);', src)
    assert len(hints) == 9 * 3
    assert sorted({int(at) % 128 for _, at in hints}) == [0, 32, 64]
    ring = 'ik_ring' if chunks == 32 else 'me_ring'
    last = max(m.start() for m in re.finditer(
        r'tensorforge::bulkCopyOn\(&' + ring, src))
    assert src.index('tensorforge::prefetchL2') > last
    assert 'the next trip\'s I asked of L2 a trip ahead' in src


@pytest.mark.parametrize('arch,order,options', [
    ('sm_100a', 4, {}), ('sm_100a', 6, {'matrix_input_chunks': 32}),
    ('sm_100a', 6, {'matrix_input_chunks': 'whole'}),
    ('sm_100a', 6, {'matrix_groups': 'unrolled', 'matrix_input_chunks': 32}),
    ('sm_100a', 6, {'matrix_input_ahead': 'l2', 'matrix_input_chunks': 32}),
    ('sm_90a', 4, {}), ('sm_90a', 6, {'matrix_input_chunks': 16}),
    ('sm_90a', 6, {'matrix_input_ahead': 'l2'})])
def test_the_input_issue_on_the_oracle(arch, order, options):
    """The elected lane's copies of I over three trips, both engines, I in
    chunks and whole, the loop and written out, and with the next trip's I
    asked of L2 a trip ahead: against NumPy without a race, and bit for bit
    what each element's lane's copies give, none asked ahead."""
    _, lane = _on_the_oracle(arch, order, 31, matrix_input_issue='lane', **{
        k: v for k, v in options.items() if k != 'matrix_input_ahead'})
    gen, got = _on_the_oracle(arch, order, 31, matrix_input_issue='elect',
                              **options)
    tile = gen.built._element_group.section.tile
    assert (tile.inputs, tile.ahead) == (
        'elect', options.get('matrix_input_ahead', 'off'))
    assert (got == lane).all()


# -- the build and the hardware ------------------------------------------------- #

def _ptxas(arch, tmp_path, order=8, **options):
    """The kernel built for `arch` with ptxas's report: `(sass, report)`."""
    from tensorforge import toolchain
    nvcc = tma._nvcc_builds(arch)
    gen = _kernel(order, arch, **options)
    context = me._context(arch, 's', multi_element='force',
                          **{**HIGH, **options})
    headers = list(context.target.headers()) + list(gen.get_helper_headers())
    src = tmp_path / 'kernel.cu'
    src.write_text(''.join(f'#include "{h}"\n' for h in headers)
                   + '#include "tensorforge_aux.h"\n' + gen.get_kernel()
                   + '\n' + gen.get_launcher())
    entry = toolchain.COMPILERS['cuda']
    obj = tmp_path / 'kernel.o'
    run = subprocess.run([nvcc, *entry.language_flags(),
                          *entry.target_flags(arch), '-Xptxas', '-v',
                          '-I', str(tma.INCLUDE), '-c', str(src), '-o',
                          str(obj)], capture_output=True, text=True,
                         timeout=1800)
    assert run.returncode == 0, run.stderr[-4000:]
    tool = shutil.which('cuobjdump')
    if tool is None:
        pytest.skip('cuobjdump not found')
    sass = subprocess.run([tool, '-sass', str(obj)], capture_output=True,
                          text=True, timeout=600).stdout
    return sass, run.stderr


@pytest.mark.parametrize('arch,engine,products,options', [
    ('sm_90a', 'HGMMA', 2 * 99, {}),
    ('sm_90a', 'HGMMA', 2 * 6 * 9, {'matrix_groups': 'loop'}),
    ('sm_100a', 'UTCHMMA', 9, {}),
    ('sm_100a', 'UTCHMMA', 9, {'matrix_bands': 3}),
    ('sm_100a', 'UTCHMMA', 9, {'matrix_output': 'bulk'}),
    ('sm_100a', 'UTCHMMA', 9, {'matrix_output': 'red'}),
    ('sm_100a', 'UTCHMMA', 9, {'tile_schedule': 'clc'}),
    ('sm_100a', 'UTCHMMA', 9, {'order': 10}),
    ('sm_100a', 'UTCHMMA', 18, {'order': 12}),
    ('sm_90a', 'HGMMA', 2 * 72, {'order': 10, 'matrix_input_chunks': 32,
                                 'matrix_groups': 'loop'}),
    ('sm_90a', 'HGMMA', 2 * 162, {'order': 12, 'matrix_input_chunks': 32,
                                  'matrix_groups': 'loop'}),
    ('sm_90a', 'HGMMA', 72, {'order': 10, 'matrix_input_chunks': 'whole',
                             'matrix_groups': 'loop'}),
    ('sm_90a', 'HGMMA', 162, {'order': 12, 'matrix_input_chunks': 'whole',
                              'matrix_groups': 'loop'}),
    ('sm_100a', 'UTCHMMA', 18, {'order': 12,
                                'matrix_input_chunks': 'whole'}),
    ('sm_90a', 'HGMMA', 216, {'matrix_passes': 2, 'matrix_groups': 'loop'}),
    ('sm_90a', 'HGMMA', 2 * 6 * 9, {'matrix_output': 'bulk',
                                    'matrix_groups': 'loop'}),
    ('sm_100a', 'UTCHMMA', 99, {'matrix_groups': 'unrolled'})])
def test_the_build_reads_back(arch, engine, products, options, tmp_path):
    """o8 built: every product an instruction of the engine's, as many as
    the text has -- wgmma's blocks written out, the 33 kept pieces' three
    each, or its group loop's six widths' nine each, tcgen05's nine, its N
    a value, or written out 99, every tile's -- and none of them
    serialized by ptxas: no C7510 (a wgmma serialized for a register's
    use), C7511, C7514, C7517 or C7519 (for an arrive in a divergent path:
    the group loop's dispatch on a width ptxas cannot tell is the warp's
    alike) in its report.  tcgen05's o8 as the options leave it adds D by
    one bulk reduction an element (`tile_output`), o10 by a column's (past
    `TILE_BULK_ROWS`), as where asked; wgmma's a round of a warp's rows a
    reduction where asked; no reduction a cell."""
    sass, report = _ptxas(arch, tmp_path, **options)
    auto = ({8: 'box', 10: 'bulk'}.get(options.get('order', 8), 'red')
            if arch == 'sm_100a' else 'red')
    output = options.get('matrix_output', auto)
    if output in ('bulk', 'box'):
        # D by bulk reductions -- tcgen05's one a thread, wgmma's a round of
        # a warp's rows -- no reduction a cell.
        assert len(re.findall(r'\bUBLKRED\b', sass)) == (
            1 if arch == 'sm_100a' else 2 * 2)
        assert not re.search(r'\bREDG\b', sass)
    if engine == 'HGMMA':
        assert len(re.findall(r'\bHGMMA\.64x\d+x8\b', sass)) == products
    else:
        assert tma._count(sass, engine) == products
    for code in ('C7510', 'C7511', 'C7514', 'C7517', 'C7519'):
        assert code not in report, report[-2000:]


@pytest.mark.parametrize('groups,products,tables', [
    ('loop', 9, 'device'), ('unrolled', 99, 'device'),
    ('loop', 9, 'constant')])
def test_the_elect_issue_builds(groups, products, tables, tmp_path):
    """o8 built for sm_100a with `elect`: the products the first lane's
    issue builds, none of them in a loop over the lanes that might issue
    it -- no ELECT or BRA.U.ANY from the first to the last -- and nothing
    spilled or reloaded between them; in the group loop at most one operand
    moved into a uniform register there (R2UR), where the first lane's
    issue moves 48, and none with its tables in constant memory, which it
    reads by uniform loads (LDCU)."""
    sass, report = _ptxas('sm_100a', tmp_path, matrix_groups=groups,
                          matrix_issue='elect', tile_tables=tables)
    ops = [m.group(1).strip() for m in re.finditer(
        r'/\*[0-9a-f]{4,}\*/\s+([^;]*);', sass)]
    mmas = [i for i, op in enumerate(ops) if 'UTCHMMA' in op]
    assert len(mmas) == products
    span = ops[mmas[0]:mmas[-1] + 1]
    assert not any('ELECT' in op or 'BRA.U.ANY' in op for op in span)
    assert not any(re.match(r'(?:@!?U?P\w+\s+)?(?:LDL|STL)\b', op)
                   for op in span)
    moved = sum(bool(re.search(r'\bR2UR\b', op)) for op in span)
    if groups == 'loop':
        assert moved <= (1 if tables == 'device' else 0)
    if tables == 'constant':
        assert sum(op.startswith('LDCU') for op in ops) >= 2
    for code in ('C7510', 'C7511', 'C7514', 'C7517', 'C7519'):
        assert code not in report, report[-2000:]


def test_the_slots_by_the_trip_build(tmp_path):
    """o8 built for sm_100a with both rings numbered by the trip, the
    elected lane's issue and the tables in constant memory: the group loop
    four bodies a trip of it, 36 UTCHMMA, none of them in a loop over the
    lanes and nothing spilled or reloaded from the first to the last."""
    sass, report = _ptxas('sm_100a', tmp_path, matrix_issue='elect',
                          tile_tables='constant', matrix_slots='trip')
    ops = [m.group(1).strip() for m in re.finditer(
        r'/\*[0-9a-f]{4,}\*/\s+([^;]*);', sass)]
    mmas = [i for i, op in enumerate(ops) if 'UTCHMMA' in op]
    assert len(mmas) == 4 * 9
    span = ops[mmas[0]:mmas[-1] + 1]
    assert not any('ELECT' in op or 'BRA.U.ANY' in op for op in span)
    assert not any(re.match(r'(?:@!?U?P\w+\s+)?(?:LDL|STL)\b', op)
                   for op in span)
    for code in ('C7510', 'C7511', 'C7514', 'C7517', 'C7519'):
        assert code not in report, report[-2000:]


@pytest.mark.parametrize('chunks', [32, 'whole'])
def test_the_input_issue_builds(chunks, tmp_path):
    """o8 built for sm_100a, the mma warp's issue by the elected lane and
    its tables in constant memory, I in chunks and whole: by each element's
    lane the input warp's bulk copies are a loop ptxas wraps over the lanes
    (ELECT, R2UR.BROADCAST, BRA.U.ANY a copy); by the elected lane they are
    every element's written out -- fourteen elements' nine columns a round
    -- with none of that loop's ELECT or R2UR.BROADCAST, and no more than a
    few bytes more spilled; as the options leave it (`auto`), the elected
    lane's."""
    def build(**more):
        sass, report = _ptxas('sm_100a', tmp_path, matrix_issue='elect',
                              tile_tables='constant',
                              matrix_input_chunks=chunks, **more)
        assert not set(re.findall(r'\bC75\d\d\b', report)), report[-2000:]
        ops = [m.group(1).strip() for m in re.finditer(
            r'/\*[0-9a-f]{4,}\*/\s+([^;]*);', sass)]
        spill = sum(int(n) for n in re.findall(r'(\d+) bytes spill stores',
                                               report))
        return (sum(bool(re.search(r'\bELECT\b', op)) for op in ops),
                sum('R2UR.BROADCAST' in op for op in ops),
                sum('UBLKCP' in op for op in ops), spill)
    lane, elect = (build(matrix_input_issue='lane'),
                   build(matrix_input_issue='elect'))
    rounds = 3 if chunks == 32 else 1
    assert elect[2] >= 14 * 9 * rounds > lane[2]
    assert elect[0] < lane[0] and elect[1] < lane[1]
    assert elect[3] <= lane[3] + 16
    # `auto` on tcgen05: the elected lane's.
    assert build() == elect


#: The one ring's builds besides it (`test_one_ring_build`): the elected
#: lane's issue, the tables in constant memory, the rings numbered by the
#: trip.
ONE_RING_BUILD = dict(matrix_issue='elect', tile_tables='constant',
                      matrix_slots='trip')


def _tile_ops(tmp_path, **options):
    """o8 built for sm_100a as `options` ask, ptxas's report free of its
    warnings: the SASS's instructions, and the group loop's iteration's --
    from the first product's block to the back edge behind the last --
    four bodies a trip of it, 36 UTCHMMA, none in a loop over the lanes."""
    sass, report = _ptxas('sm_100a', tmp_path, **options)
    for code in ('C7510', 'C7511', 'C7514', 'C7517', 'C7519'):
        assert code not in report, report[-2000:]
    ops = [m.group(1).strip() for m in re.finditer(
        r'/\*[0-9a-f]{4,}\*/\s+([^;]*);', sass)]
    mmas = [i for i, op in enumerate(ops) if 'UTCHMMA' in op]
    assert len(mmas) == 4 * 9
    span = ops[mmas[0]:mmas[-1] + 1]
    assert not any('ELECT' in op or 'BRA.U.ANY' in op for op in span)
    return ops, span


def test_one_ring_build(tmp_path):
    """o8 built for sm_100a with A's ring and B's as one (`ONE_RING_BUILD`):
    a block one wait for its slot and one commit (SYNCS, UTCBAR), where two
    rings take two of each; no thread parked in an mbarrier's wait
    (NANOSLEEP.SYNCS), waiting by `try` (`test_the_wait_forms_build`)."""
    counts = {}
    for handoffs in ('two', 'one'):
        ops, span = _tile_ops(tmp_path, **ONE_RING_BUILD,
                              matrix_handoffs=handoffs)
        counts[handoffs] = (sum('SYNCS.PHASECHK' in op for op in span),
                            sum(op.startswith('UTCBAR') for op in span))
        assert not any('NANOSLEEP.SYNCS' in op for op in ops)
    # Between the first product and the last, three blocks' hand-offs.
    assert counts == {'two': (6, 6), 'one': (3, 3)}


@pytest.mark.parametrize('form', ['suspend', 'test'])
def test_the_wait_forms_build(form, tmp_path):
    """The one ring's o8 build (`test_one_ring_build`) with the mma warp
    waiting as `matrix_wait` asks: `test` its three hand-offs between the
    first product and the last a test of the phase (SYNCS.PHASECHK without
    TRYWAIT) in a spin, no thread parked in an mbarrier's wait; `suspend` a
    try_wait, behind which ptxas parks the thread (NANOSLEEP.SYNCS) for the
    hint's 10 ms (0x989680 ns) where the phase is not complete.  (The
    tensor memory's allocation retries behind a NANOSLEEP of its own.)"""
    ops, span = _tile_ops(tmp_path, **ONE_RING_BUILD, matrix_handoffs='one',
                          matrix_wait=form)
    checks = [op for op in span if 'SYNCS.PHASECHK' in op]
    sleeps = [op for op in ops if 'NANOSLEEP.SYNCS' in op]
    assert len(checks) == 3
    if form == 'test':
        assert not any('TRYWAIT' in op for op in checks) and not sleeps
    else:
        assert all('TRYWAIT' in op for op in checks)
        assert sleeps and all('NANOSLEEP.SYNCS 0x989680' in op
                              for op in sleeps)


#: The tile's variants on the GPU: as the options leave it, its blocks
#: written out and in one loop, I whole in the input ring and in chunks
#: (wgmma's as the options leave it is I whole where it fits, tcgen05's in
#: chunks), and
#: tcgen05's own levers -- three star bands, D added a cell at a time, by
#: bulk reductions a column and an element's box, the mma warp's issue by
#: the lane elect.sync picks, and with it the group loop's tables in
#: constant memory, and the rings numbered by the trip, alone and with
#: both; A's ring and B's as one, alone and with both, and so with the mma
#: warp waiting by try_wait with a suspend-time hint and by test_wait; and
#: I's bulk copies by the lane elect.sync picks, in chunks and whole, and
#: the next trip's I asked of L2 a trip ahead, alone and with it.
GPU_VARIANTS = {'': {}, 'unrolled': {'matrix_groups': 'unrolled'},
                'loop': {'matrix_groups': 'loop'},
                'whole': {'matrix_input_chunks': 'whole'},
                'chunks': {'matrix_input_chunks': 32},
                'b3': {'matrix_bands': 3}, 'red': {'matrix_output': 'red'},
                'bulk': {'matrix_output': 'bulk'},
                'box': {'matrix_output': 'box'},
                'elect': {'matrix_issue': 'elect'},
                'constant': {'matrix_issue': 'elect',
                             'tile_tables': 'constant'},
                'trip': {'matrix_slots': 'trip'},
                'trip-ec': {'matrix_slots': 'trip', 'matrix_issue': 'elect',
                            'tile_tables': 'constant'},
                'one': {'matrix_handoffs': 'one'},
                **{name: {'matrix_handoffs': 'one', 'matrix_issue': 'elect',
                          'tile_tables': 'constant', **wait}
                   for name, wait in (('one-ec', {}),
                                      ('one-ec-s', {'matrix_wait': 'suspend'}),
                                      ('one-ec-t', {'matrix_wait': 'test'}))},
                'ie': {'matrix_input_issue': 'elect'},
                'il': {'matrix_input_issue': 'lane'},
                'ie-whole': {'matrix_input_issue': 'elect',
                             'matrix_input_chunks': 'whole'},
                'ahead': {'matrix_input_ahead': 'l2'},
                'ie-ahead': {'matrix_input_issue': 'elect',
                             'matrix_input_ahead': 'l2'}}
#: The variants of tcgen05's levers alone (`test_the_tile_computes_on_the_
#: gpu`): wgmma's tile has no bands, adds D a cell or a row a reduction,
#: and its products are the warpgroup's.
TCGEN05_VARIANTS = ('b3', 'red', 'box', 'elect', 'constant', 'trip',
                    'trip-ec', 'one', 'one-ec', 'one-ec-s', 'one-ec-t')


@pytest.mark.parametrize('variant', sorted(GPU_VARIANTS))
@pytest.mark.parametrize('order', [6, 8, 10, 12])
def test_the_tile_computes_on_the_gpu(order, variant, target, cache_root,
                                      tensorforge_include):
    """On a GH200 (wgmma) or GB200 (tcgen05): the volume in the high-order
    tile against NumPy over a batch ending in a partial group, several
    trips a block, the operators their description's numbers
    (`test_multi_element._described`)."""
    if target.backend != 'cuda' or target.arch not in ('sm_90', 'sm_100'):
        pytest.skip('no GPU with wgmma or tcgen05')
    if target.arch == 'sm_90' and variant in TCGEN05_VARIANTS:
        pytest.skip('tcgen05\'s levers: wgmma\'s tile has no bands, adds D '
                    'a cell or a row a reduction, and its products are the '
                    'warpgroup\'s')
    if order > 8 and variant in ('b3', 'red', 'bulk', 'box'):
        pytest.skip('o10 and o12: the tile as the options leave it, its '
                    'blocks written out and in one loop, I whole and in '
                    'chunks, and the mma warp\'s issue by elect.sync\'s '
                    'lane, its loop\'s tables in constant memory too, the '
                    'rings numbered by the trip, and A\'s ring and B\'s as '
                    'one')
    from harness.runner import run_case
    options = {**HIGH, **GPU_VARIANTS[variant]}
    special = dataclasses.replace(target, arch=target.arch + 'a')
    tma._nvcc_builds(special.arch)
    group = eg.plan(me._descrs('volume', order, 's'), me._context(
        special.arch, 's', multi_element='force', **options))
    assert group is not None and group.section.tile is not None
    case = types.SimpleNamespace(
        NAME=f'hiorder_tile_volume_o{order}s' + (f'_{variant}' if variant
                                                  else ''),
        DTYPE=Datatype.F32, BATCH=4099, TOL=me._normwise(),
        OPTIONS={'multi_element': 'force', **options},
        INPUT_TRANSFORM=me._described(me._descrs('volume', order, 's')),
        descr_list=lambda: me._descrs('volume', order, 's'),
        reference=me._reference(me._descrs('volume', order, 's')))
    result = run_case(case, special, cache_root, tensorforge_include)
    assert result.passed, (result.max_abs_err, result.max_rel_err,
                           result.detail)


@pytest.mark.parametrize('variant', ['', 'unrolled', 'red', 'bulk', 'box',
                                     'elect', 'constant', 'trip', 'trip-ec',
                                     'one', 'one-ec', 'one-ec-s', 'one-ec-t',
                                     'ie', 'ie-whole', 'ahead', 'ie-ahead',
                                     'il'])
@pytest.mark.parametrize('order', [6, 8])
def test_many_trips_of_the_tile_on_the_gpu(order, variant, target, cache_root,
                                           tensorforge_include):
    """On a GB200, where J2 found three bands on two input slots wrong and
    hung (A's ring's slots in turn among the bands): the tile over 65537
    elements on a twentieth of the grid the device holds -- some 650 trips a
    block (`Options.resident_grid`) -- three bands (the default J2 ran),
    bit for bit what two bands (the default now) give over the same
    elements (the same products in the same order), each output form
    against its own."""
    if target.backend != 'cuda' or target.arch != 'sm_100':
        pytest.skip('no GPU with tcgen05')
    import numpy as np
    from harness import driver_emit
    from harness.runner import execute
    special = dataclasses.replace(target, arch='sm_100a')
    tma._nvcc_builds(special.arch)
    options = {**HIGH, **GPU_VARIANTS[variant]}
    builds = [me._generate(me._descrs('volume', order, 's'), me._context(
        special.arch, 's', multi_element='force', **options, matrix_bands=b,
        resident_grid=0.05)) for b in (3, 2)]
    tiles = [gen.built._element_group.section.tile for gen in builds]
    assert [t.bands for t in tiles] == [3, 2]
    batch = 65537
    rng = np.random.default_rng(19)
    inputs = {}
    for op in driver_emit.collect_operands(builds[0]):
        if op.is_scalar or not op.is_source:
            continue
        key = op.alias or op.kernel_name
        if op.values is not None:
            inputs[key] = np.asarray(op.values, dtype=np.float32).reshape(
                op.shape, order='F')[None]
            continue
        rows = 1 if op.addressing == 'none' else batch
        inputs[key] = rng.standard_normal((rows, *op.shape)).astype(
            np.float32)
    got, want = (execute(gen, special, cache_root, tensorforge_include,
                         batch, inputs, runtime_timeout=300.0)
                 for gen in builds)
    assert set(got) == set(want)
    for name in want:
        assert np.array_equal(got[name], want[name]), name
