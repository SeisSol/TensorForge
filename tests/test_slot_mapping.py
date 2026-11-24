# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""The multi-element layout's slot form (`Options.multi_element_fma`
`slots`, `element_group.SlotMapping`, `compute.multi_element._SlotTrip`):
the lanes form with its lanes laid out by a lead -- a warp's elements and
their columns flattened, `lane = i mod 32`, `slot = i div 32` -- one slot a
lane on CUDA: nine lanes an element, three elements a warp (L9), each lane
one column of its element, and the mixing's other columns brought by
shuffles planned at generation.

What is checked here: the mappings and the exchange plan, the fill rule
and each reason it refuses, the kernel's shape -- the lanes' binding and
mask, the shuffles a row, no shared memory -- the build for sm_90a, the host
oracle over two trips of a group with partial and masked warps on every
shape the form takes, and on the local GPU against NumPy.
"""

from __future__ import annotations

import re
import types

import pytest

import test_multi_element as me
from tensorforge.common.basic_types import Addressing
from tensorforge.common.exceptions import InternalError
from tensorforge.generators import element_group as eg

#: The slot form, where the layout is forced.
SLOTS = {'multi_element': 'force', 'multi_element_fma': 'slots'}


def _diagonal_star_description(order=4, precision='s'):
    """SeisSol's sparse star with its diagonal stored as well: a column
    multiplies its own entry, and the columns that read two others receive
    nothing by the third shuffle -- whose coefficient, the diagonal's cell
    were it read, has to be zero."""
    d = me._sparse_star_description(order, precision)
    entries = sorted([list(c) for c in me.STAR_ENTRIES]
                     + [[q, q] for q in range(9)], key=lambda c: (c[1], c[0]))
    for t in d['tensors']:
        if t['name'].startswith('star('):
            t['storage']['entries'] = entries
    return d


def _copied_description(order=4, precision='s'):
    """The volume, then its output copied, `P = Q` over the rows it adds
    into: a chained section of one phase, whose copy reads an output that
    accumulated from the column the lane holds -- what the output holds
    after the phase, not the phase's sums alone."""
    import copy
    import json
    d = json.loads(json.dumps(me._volume_description(order, precision)))
    q = next(t for t in d['tensors'] if t['name'] == 'Q')
    d['tensors'].append({**copy.deepcopy(q), 'name': 'P'})
    last = d['operations'][-1]
    box = [list(last['result']['bbox'][0]), list(last['result']['bbox'][1])]

    def view(name):
        return {'name': name, 'indices': ['a', 'b'], 'bbox': box,
                'offset': [0, 0], 'offset_from': None, 'sliced': False}
    d['operations'].append({
        'type': 'multilinear', 'result': view('P'), 'args': [view('Q')],
        'condition': [], 'permute': [[0, 1]], 'target': [[0, 1]],
        'linear': {'alpha': last['linear']['alpha'], 'add': False}})
    return d


#: The descriptions made here, beside `test_multi_element`'s.
DESCRIPTIONS = {'diagonal-star': _diagonal_star_description,
                'copied': _copied_description}


def _descrs(shape='volume', order=4, precision='s'):
    if shape not in DESCRIPTIONS:
        return me._descrs(shape, order, precision)
    from tensorforge.frontend.yateto import DescriptionReader
    return DescriptionReader(None, {}).read(
        DESCRIPTIONS[shape](order, precision))[0]


def _plan(shape='volume', order=4, precision='s', arch='sm_90', **options):
    context = me._context(arch, precision, **{**SLOTS, **options})
    section, why = eg.classify(_descrs(shape, order, precision))
    assert section is not None, why
    return eg.fill(section, context), section, context


# -- the mappings ------------------------------------------------------------ #

def test_nine_lanes_an_element_is_three_elements_a_warp():
    """L9: element-outer and unpadded, one slot a lane -- lane `9 e + q` --
    27 of 32 lanes, each one element's."""
    m = eg.split_mapping(9, 9, 32)
    assert (m.name, m.elements, m.slots, m.lanes, m.busy) == ('L9', 3, 1, 9,
                                                              27)
    assert all(m.item(lane, 0) == divmod(lane, 9) for lane in range(27))
    assert all(m.item(lane, 0) is None for lane in range(27, 32))
    assert m.place(2, 4) == (22, 0) and m.one_element()


def test_three_lanes_an_element_take_three_slots():
    """L3: quantity `q` on lane `q mod 3` of its element's three, at slot
    `q div 3` -- ten elements on 30 lanes."""
    m = eg.split_mapping(3, 9, 32)
    assert (m.elements, m.slots, m.busy) == (10, 3, 30)
    assert m.place(4, 7) == (13, 2) and m.one_element()


def test_a_wrapped_lead_puts_two_elements_on_a_lane():
    """E7 element-outer: 63 items in two slots of 32 lanes, a lane's slots
    of two elements where the wrap falls -- no lane binds one element."""
    m = eg.wrapped_mapping(7, 9, 32)
    assert (m.elements, m.slots, len(m.items)) == (7, 2, 63)
    assert m.place(3, 5) == (0, 1) and m.item(31, 1) is None
    assert not m.one_element()


# -- the exchange ------------------------------------------------------------ #

def _needs(mapping, pattern):
    """Each busy lane's needs under `pattern(q_from, q_to)`."""
    return {lane: {x for x in range(mapping.quantities)
                   if x != q and pattern(x, q)}
            for lane, _, _, q in mapping.items}


def test_a_dense_star_takes_nine_broadcasts_a_row():
    """Every lane needs every other quantity of its element: broadcasts
    (`element_group.BROADCASTS`) -- shuffle k brings quantity k to every
    lane of the element, from its lane `9 e + k`, the lane holding it
    reading itself: nine a row, one more than the eight the needs take, so
    that a lane's coefficient of term k is its column's entry k.  The idle
    lanes read themselves."""
    m = eg.split_mapping(9, 9, 32)
    x = eg.plan_exchange(m, _needs(m, lambda a, b: True))
    assert (x.count, x.bound, x.broadcast) == (9, 8, True)
    for lane in range(27):
        e, q = divmod(lane, 9)
        assert [x.received[k][lane] for k in range(9)] == list(range(9))
        assert [x.sources[k][lane] for k in range(9)] == [
            9 * e + k for k in range(9)]
    assert all(x.sources[k][lane] == lane and x.received[k][lane] == -1
               for k in range(9) for lane in range(27, 32))
    assert all(x.full(k, m) for k in range(9))


def test_needs_short_of_every_quantity_take_a_rotation():
    """Where a lane needs fewer than all -- here all but the quantity four
    after its own -- the needs plan: as many shuffles as the most a lane
    needs, each lane receiving its needs once, in the order after its own
    quantity, each from that quantity's lane."""
    m = eg.split_mapping(9, 9, 32)
    x = eg.plan_exchange(m, _needs(m, lambda a, b: (a - b) % 9 != 4))
    assert (x.count, x.bound, x.broadcast) == (7, 7, False)
    assert set(x.kinds) == {'lanes'}
    for lane in range(27):
        e, q = divmod(lane, 9)
        got = [x.received[k][lane] for k in range(7)]
        assert got == [(q + d) % 9 for d in (1, 2, 3, 5, 6, 7, 8)]
        assert [x.sources[k][lane] for k in range(7)] == [9 * e + g
                                                          for g in got]


def test_seissols_sparse_star_takes_three():
    """The star as SeisSol's GPU layout stores it (`me.STAR_ENTRIES`): an
    output column reads two or three others, so three shuffles a row; a
    column of two receives nothing by the third."""
    m = eg.split_mapping(9, 9, 32)
    stored = {tuple(c) for c in me.STAR_ENTRIES}
    x = eg.plan_exchange(m, _needs(m, lambda a, b: (a, b) in stored))
    assert x.count == 3
    assert not x.full(2, m) and x.full(0, m)
    for lane in range(27):
        q = lane % 9
        got = {x.received[k][lane] for k in range(3)} - {-1}
        assert got == {a for a, b in stored if b == q}


def test_several_slots_a_lane_are_not_exchanged_yet():
    with pytest.raises(ValueError, match='3 slots a lane'):
        eg.plan_exchange(eg.split_mapping(3, 9, 32), {})


# -- the fill rule ----------------------------------------------------------- #

@pytest.mark.parametrize('order,precision,arch', [
    (4, 's', 'sm_90'), (4, 'd', 'sm_120'), (6, 's', 'sm_90'),
    (6, 'd', 'sm_80')])
def test_the_volume_takes_l9_in_blocks_of_four_warps(order, precision, arch):
    """L9 on CUDA whatever the order and the precision: a block of four
    warps, each its own three elements, a group of twelve; the star dense,
    nine broadcasts and every one a term of each product, the lane's own
    column among them."""
    (group, why), section, _ = _plan('volume', order, precision, arch)
    assert group is not None, why
    plan = group.lanes
    assert (group.elements, group.warps, group.threads) == (12, 4, 32)
    assert plan.mapping == eg.split_mapping(9, 9, 32)
    assert (plan.exchange.count, plan.exchange.broadcast) == (9, True)
    mixing = [p for s in section.steps for p in s.products
              if p.kind is eg.Kind.MIXING]
    assert all(eg.slot_terms(plan, p, 0) == list(range(9))
               for p in mixing)
    assert all(s.lowering == 'lane' for s in group.section.steps)


def test_the_sparse_star_drops_the_own_term():
    """SeisSol's star stores no diagonal: no lane multiplies its own
    column, three shuffles a row, three terms a product."""
    (group, why), section, _ = _plan('sparse-star', 6, 's')
    assert group is not None, why
    plan = group.lanes
    assert (plan.exchange.count, plan.exchange.broadcast) == (3, False)
    mixing = [p for s in section.steps for p in s.products
              if p.kind is eg.Kind.MIXING]
    assert all(eg.slot_terms(plan, p, 0) == [0, 1, 2] for p in mixing)
    assert plan.registers < eg.slots_plan(
        eg.classify(me._descrs('volume', 6, 's'))[0],
        me._context('sm_90', 's', **SLOTS))[0].registers


def test_a_stored_diagonal_keeps_the_own_term_and_zeroes_the_rest():
    """With the diagonal stored, every column multiplies its own entry;
    three shuffles still, the third bringing three of the columns nothing
    (`Exchange.full`)."""
    (group, why), section, _ = _plan('diagonal-star', 4, 'd')
    assert group is not None, why
    plan = group.lanes
    assert plan.exchange.count == 3
    assert [plan.exchange.full(k, plan.mapping) for k in range(3)] == [
        True, True, False]
    mixing = [p for s in section.steps for p in s.products
              if p.kind is eg.Kind.MIXING]
    assert all(eg.slot_terms(plan, p, 0) == [-1, 0, 1, 2] for p in mixing)


def test_the_option_keeps_the_description_unreassociated():
    """`reassociate_products` would put the operator first and mix its
    intermediate after it, which the slot form does not take: the form
    taken, as the lanes form, keeps the description as yateto wrote it.
    `localFlux`, whose face matrix has fewer rows than the element's
    operand -- the volume's operators have more, and the rewrite leaves it
    as it is anyway."""
    gen = me._generate(me._descrs('localFlux', 4, 's'), me._context(
        'sm_90', 's', reassociate_products=True, **SLOTS))
    group = gen.built._element_group
    assert group is not None and group.lanes is not None
    assert group.lanes.mapping is not None


def test_an_operator_section_takes_as_many_lanes_as_columns():
    """No mixing at all: a lane a column of the output, no shuffle --
    `plConvertToModal`'s six columns, five elements a warp."""
    (group, why), _, _ = _plan('plConvertToModal', 4, 's')
    assert group is not None, why
    assert group.lanes.mapping == eg.split_mapping(6, 6, 32)
    assert group.lanes.exchange.count == 0 and group.elements == 20


@pytest.mark.parametrize('arch,backend', [('gfx942', 'hip'),
                                          ('pvc', 'oneapi')])
def test_the_slot_form_is_cudas(arch, backend):
    context = me._context(arch, 's', backend=backend, **SLOTS)
    section, _ = eg.classify(me._descrs('volume', 4, 's'))
    plan, why = eg.slots_plan(section, context)
    assert plan is None and 'written for CUDA on NVIDIA' in why


@pytest.mark.parametrize('shape,options,why', [
    ('volume', {'multi_element_values': 'table'}, 'reads no table'),
    ('chain', {}, 'an intermediate, is mixed'),
], ids=['table', 'mixed-intermediate'])
def test_what_the_slot_form_does_not_take_says_why(shape, options, why):
    context = me._context('sm_90', 's', **{**SLOTS, **options})
    section, _ = eg.classify(me._descrs(shape, 4, 's'))
    plan, said = eg.slots_plan(section, context)
    assert plan is None and why in said


def test_a_section_it_refuses_falls_back_to_the_warp_form():
    """As the lanes form does: the warp form's plan, where the section is no
    chained one (the operator first, its intermediate mixed after)."""
    (group, why), _, _ = _plan('chain', 6, 's')
    assert group is not None and group.lanes is None, why


def test_the_derivative_takes_its_phases_in_each_warp():
    """A chained section a phase after another in every warp: the copies
    alone first, no plan of their own, then each order's mixing and
    operator steps on the one lead -- L9, eight shuffles a row."""
    (group, why), _, _ = _plan('derivative', 6, 'd')
    assert group is not None, why
    plans = group.lanes.phases
    assert (group.elements, group.warps) == (12, 4) and plans[0] is None
    assert len(plans) == 6 and all(
        p.mapping == eg.split_mapping(9, 9, 32) and p.exchange.count == 9
        and p.exchange.broadcast for p in plans[1:])
    assert [p.ranges for p in plans[1:]] == [
        ((0, 35),), ((0, 20),), ((0, 10),), ((0, 4),), ((0, 1),)]


def test_the_tuner_tries_the_slot_forms_caps_reads_and_grids():
    """Where the kernel takes the slot form: `min_blocks_per_sm` 1, 2 and 3
    (255, 255 and 168 registers a thread of its block of four warps), the
    output read ahead and masked, the grid by occupancy and in one round --
    twelve builds where every one is tried; none of them where it does
    not."""
    from tensorforge.generators import tuning
    descrs = me._descrs('volume', 6, 's')
    knobs = {k.name: k for k in tuning.space(
        descrs, me._context('sm_90', 's', **SLOTS))}
    origin = tuning.Candidate()
    assert list(knobs['min_blocks_per_sm'].values(origin)) == [1, 2, 3]
    assert list(knobs['slot_output_read'].values(origin)) == ['ahead',
                                                              'masked']
    assert list(knobs['one_round_grid'].values(origin)) == [False, True]
    slots = [knobs[n] for n in ('min_blocks_per_sm', 'slot_output_read',
                                'one_round_grid')]
    assert len(list(tuning.enumerate_space(slots, origin))) == 12
    lanes = {k.name for k in tuning.space(
        descrs, me._context('sm_90', 's', multi_element='force',
                            multi_element_fma='lanes'))}
    assert not lanes & {'min_blocks_per_sm', 'slot_output_read',
                        'one_round_grid'}


def test_autotunes_walk_turns_neither_the_placement_nor_the_grid():
    """`Options.autotune`'s walk (`simple_space`) turns neither: the grid is
    a launch no build shows, and the placement leaves the bound its margin
    is read off where it was."""
    from tensorforge.generators import tuning
    descrs = me._descrs('volume', 6, 's')
    context = me._context('sm_90', 's', **SLOTS)
    for mode in ('static', 'compiled'):
        knobs = {k.name for k in tuning.simple_space(descrs, context,
                                                     mode=mode)}
        assert not knobs & {'slot_output_read', 'one_round_grid'}


# -- `auto`: the slot form where it pays, the element form where it does not -- #

AUTO = {'multi_element': 'auto'}


def _seissol(kernel, order, precision):
    """SeisSol's kernel as yateto sent it (the fixture's capture); with
    `/sparse` after its name, its stars stored as SeisSol's GPU builds store
    them (`spp`, `me.STAR_ENTRIES`), as the jobs' `vols` and `ders` do."""
    import json
    from tensorforge.frontend.yateto import DescriptionReader
    name, _, stored = kernel.partition('/')
    d = json.loads(json.dumps(me._seissol(name)(order, precision)))
    if stored == 'sparse':
        for t in d['tensors']:
            if t['name'].startswith('star('):
                t['storage'] = {'shape': [9, 9], 'type': 'spp',
                                'entries': me.STAR_ENTRIES}
    return DescriptionReader(None, {}).read(d)[0]


#: (kernel, order, precision, arch, taken): what `auto` measured to pay and
#: to lose, on GH200 (sm_90a), GB200 (sm_100a), sm_120 and a target nobody
#: measured; `/sparse`, the stars stored sparse (`_seissol`).
AUTO_CASES = [
    ('gpu_volume', 4, 's', 'sm_90a', True), ('gpu_volume', 6, 'd', 'sm_90a',
                                             True),
    ('gpu_derivative', 6, 's', 'sm_90a', True),
    ('gpu_derivative', 6, 'd', 'sm_90a', False),
    ('gpu_derivative/sparse', 6, 'd', 'sm_90a', True),
    ('gpu_derivative', 6, 'd', 'sm_100a', True),
    ('gpu_localFlux[0]', 4, 'd', 'sm_90a', True),
    ('gpu_localFlux[0]', 6, 's', 'sm_90a', False),
    ('gpu_localFlux[0]', 6, 'd', 'sm_100a', False),
    ('gpu_localFlux[0]', 6, 's', 'sm_80', False),
    ('gpu_localFluxAll', 6, 's', 'sm_90a', True),
    ('gpu_localFluxAll', 6, 'd', 'sm_100a', False),
    ('gpu_localFluxAll', 6, 'd', 'sm_120', False),
    ('gpu_derivative', 4, 's', 'sm_120', False),
    ('gpu_derivative/sparse', 4, 's', 'sm_120', False),
    ('gpu_derivative', 4, 'd', 'sm_120', True),
    ('gpu_derivative', 4, 's', 'sm_100a', True),
    ('gpu_derivative/sparse', 4, 's', 'sm_90a', True),
    ('gpu_volume/sparse', 6, 'd', 'sm_90a', True)]


@pytest.mark.parametrize('kernel,order,precision,arch,taken', AUTO_CASES,
                         ids=[f'{k.replace("/", "-")}-o{o}{p}-{a}'
                              for k, o, p, a, _ in AUTO_CASES])
def test_auto_takes_the_slot_form_on_nvidia_where_it_pays(
        kernel, order, precision, arch, taken):
    """`multi_element=auto` on CUDA: the slot form -- the group `force`
    takes in it -- where nothing in `SLOT_DECLINES` declines the section,
    and no plan, the per-element kernel, where an entry does: localFlux at
    o6 on every target, localFluxAll at o6 in F64 too, the derivative at o6
    in F64 with its stars dense on sm_90 alone -- with them sparse the slot
    form -- and at o4 in F32 on sm_120 alone."""
    descrs = _seissol(kernel, order, precision)
    group = eg.plan(descrs, me._context(arch, precision, **AUTO))
    section = eg.classify(descrs)[0]
    if taken:
        assert group is not None and group.lanes.mapping is not None
        forced = eg.plan(descrs, me._context(arch, precision, **SLOTS))
        assert (group.elements, group.lanes) == (forced.elements,
                                                 forced.lanes)
        assert eg.slot_decline(section, me._context(arch, precision)) is None
    else:
        assert group is None
        lanes, why = eg.auto_takes(section, me._context(arch, precision,
                                                        **AUTO))
        assert lanes is None and ('slower' in why or 'no faster' in why)


def test_auto_takes_no_other_form_on_nvidia():
    """Asked for the warp or the lanes form, `auto` takes nothing on CUDA;
    `slots` and `auto` take the slot form."""
    descrs = _seissol('gpu_volume', 4, 's')
    section = eg.classify(descrs)[0]
    for form in ('warp', 'lanes'):
        context = me._context('sm_90a', 's', multi_element_fma=form, **AUTO)
        lanes, why = eg.auto_takes(section, context)
        assert lanes is None and f'the {form} form is asked for' in why
        assert eg.plan(descrs, context) is None
    for form in ('auto', 'slots'):
        context = me._context('sm_90a', 's', multi_element_fma=form, **AUTO)
        assert eg.plan(descrs, context).lanes.mapping is not None


def test_every_decline_names_a_seissol_kernel_and_its_jobs():
    """Each entry of `SLOT_DECLINES` is the signature of a SeisSol kernel at
    an order the jobs measured, and says what was measured where: a job's
    number, or the local run's date."""
    signatures = {eg.slot_signature(eg.classify(_seissol(k, o, 's'))[0])
                  for k in ('gpu_volume', 'gpu_localFlux[0]',
                            'gpu_localFluxAll', 'gpu_derivative',
                            'gpu_volume/sparse', 'gpu_derivative/sparse')
                  for o in (4, 6)}
    for entry in eg.SLOT_DECLINES:
        assert entry.signature in signatures
        assert re.search(r'(Vista|Horizon) \d+|local run, \d{4}-\d\d-\d\d',
                         entry.why)


#: (kernel, order, precision, arch, read, one round): the defaults' rule on
#: GH200 (sm_90a), GB200 (sm_100a) and a target nobody measured, which takes
#: GB200's.
DEFAULTS = [
    ('gpu_volume', 4, 's', 'sm_90a', 'ahead', True),
    ('gpu_volume', 4, 'd', 'sm_90a', 'ahead', True),
    ('gpu_volume', 6, 's', 'sm_90a', 'masked', True),
    ('gpu_volume', 6, 'd', 'sm_90a', 'masked', True),
    ('gpu_localFlux[0]', 4, 's', 'sm_90a', 'ahead', True),
    ('gpu_localFlux[0]', 4, 'd', 'sm_90a', 'masked', True),
    ('gpu_localFluxAll', 4, 's', 'sm_90a', 'ahead', True),
    ('gpu_localFluxAll', 4, 'd', 'sm_90a', 'masked', True),
    ('gpu_localFluxAll', 6, 's', 'sm_90a', 'ahead', True),
    ('gpu_volume', 4, 's', 'sm_100a', 'masked', False),
    ('gpu_volume', 4, 'd', 'sm_100a', 'masked', False),
    ('gpu_volume', 6, 's', 'sm_100a', 'masked', True),
    ('gpu_localFlux[0]', 4, 's', 'sm_100a', 'ahead', True),
    ('gpu_localFlux[0]', 4, 'd', 'sm_100a', 'masked', True),
    ('gpu_localFluxAll', 4, 's', 'sm_100a', 'ahead', True),
    ('gpu_localFluxAll', 4, 'd', 'sm_100a', 'masked', True),
    ('gpu_derivative', 4, 's', 'sm_100a', 'ahead', True),
    ('gpu_volume', 4, 's', 'sm_120', 'masked', False)]


@pytest.mark.parametrize('kernel,order,precision,arch,read,one_round',
                         DEFAULTS, ids=[f'{k}-o{o}{p}-{a}' for k, o, p, a, *_
                                        in DEFAULTS])
def test_the_slot_forms_defaults_follow_the_measured_rule(
        kernel, order, precision, arch, read, one_round):
    """`slot_output_read=auto` (`slot_read`): a section that holds columns
    -- the fluxes' faces, the derivative's orders -- reads masked in F64
    and ahead in F32; any other masked, on GH200 only past 32 rows of
    output.  `one_round_grid=auto` (`slot_one_round`): one round on GH200,
    elsewhere where the section holds columns or its output is long.
    Asked, as asked."""
    descrs = _seissol(kernel, order, precision)
    section = eg.classify(descrs)[0]
    context = me._context(arch, precision)
    assert eg.slot_read(section, context) == read
    other = 'ahead' if read == 'masked' else 'masked'
    assert eg.slot_read(section, me._context(
        arch, precision, slot_output_read=other)) == other
    group = eg.plan(descrs, me._context(arch, precision, **SLOTS))
    assert eg.slot_one_round(group, context) is one_round
    assert eg.slot_one_round(None, context) is False


def test_the_slot_forms_grid_covers_the_batch_in_one_round():
    """`one_round_grid=auto`: a block an element group where the kernel
    takes the slot form and the rule asks for it (GH200 always, GB200 at
    o6), the occupancy grid where it does not or where it is asked off, and
    a resident grid where one is asked for."""
    def launcher(order=4, arch='sm_90', **options):
        return me._generate(me._descrs('volume', order, 's'),
                            me._context(arch, 's', **options)
                            ).get_launcher()
    one_round = 'config.grid[0] = (numElements0 + 12 - 1) / 12;'
    occupancy = 'gridsize = smCount * blocksPerSM;'
    assert one_round in launcher(**SLOTS)
    assert occupancy in launcher(**SLOTS, one_round_grid=False)
    assert occupancy in launcher()
    assert occupancy in launcher(multi_element='force')  # the warp form
    assert occupancy in launcher(arch='sm_100a', **SLOTS)
    assert one_round in launcher(6, 'sm_100a', **SLOTS)
    resident = launcher(**SLOTS, resident_grid=1.0)
    assert 'static_cast<std::size_t>(1.0 * gridsize)' in resident
    kernel = me._generate(me._descrs('volume', 4, 's'),
                          me._context('sm_90', 's', **SLOTS)).get_kernel()
    assert '"persistent":false' in kernel


def test_only_a_group_on_the_lanes_has_a_lane_map():
    from tensorforge.backend.instructions.batch_loop import (BatchLoop,
                                                             LoopMode, Trip)
    context = me._context('sm_90', 's')
    with pytest.raises(InternalError, match='only a group on the lanes'):
        BatchLoop(context, 0, LoopMode.PERSISTENT, '0', '1', [],
                  group_size=12, trip=Trip.ELEMENT_GROUP, lane_map=(3, 9))


# -- the kernel -------------------------------------------------------------- #

def _kernel(shape='volume', order=4, precision='s', arch='sm_90', **options):
    gen = me._generate(me._descrs(shape, order, precision),
                       me._context(arch, precision, **{**SLOTS, **options}))
    assert gen.built._element_group.lanes.mapping is not None
    return gen.get_kernel()


def test_a_lane_binds_its_element_and_the_idle_lanes_none():
    """The loop's group on the lanes (`Trip.LANE_GROUP`) with a lane map:
    each warp's three elements, nine lanes each; lanes 27 to 31 masked, and
    asked first, so that they read no flag."""
    kernel = _kernel()
    assert 'const auto batchIdLane0 = 3 * threadIdx.y + threadIdx.x / 9;' \
        in kernel
    assert re.search(r'batchIdActive0 = threadIdx\.x < 27 && batchIdGroup0 '
                     r'\+ batchIdLane0 < numElements0', kernel)
    assert '__launch_bounds__(128, 1)' in kernel
    assert 'batchIdGroup0 += (gridDim.x * 12)' in kernel


@pytest.mark.parametrize('shape,order,precision,shuffles,rows,tables', [
    ('volume', 4, 's', 9, 10, 0), ('volume', 6, 'd', 9, 35, 0),
    ('sparse-star', 4, 'd', 3, 10, 3)])
def test_a_row_of_the_star_is_its_shuffles(shape, order, precision,
                                           shuffles, rows, tables):
    """A shuffle a planned exchange a row of `I` -- the shuffles shared by
    the three directions -- from a table of source lanes each, or under
    broadcasts from the element's first lane plus the quantity; no shared
    memory."""
    kernel = _kernel(shape, order, precision)
    assert kernel.count('tensorforge::gather<32>(') == shuffles * rows
    found = re.findall(r'__device__ const int32_t (tf_mes_\w+)\[32\]',
                       kernel)
    assert len(found) == tables
    assert '__shared__ ' not in kernel.replace(
        'extern __shared__ char totalShrMemPtr[];', '')


def test_a_column_is_written_back_in_whole_tiles():
    """Row 0 of `Q` is no row of the volume's, but its tile is written as
    one access, the row as it was read."""
    kernel = _kernel()
    stores = re.findall(r'^\s*\*\(tensorforge::VectorT<float, 4>\*\)&glb_\w+'
                        r'\[[^\]]+\] = ', kernel, re.M)
    assert len(stores) == 5
    assert not re.search(r'^\s*glb_\w+\[[^\]]+\] = ', kernel, re.M)


def test_a_broadcast_reads_its_coefficients_at_the_columns_offsets():
    """Under broadcasts the source of shuffle k is the element's first lane
    plus k, and a lane's coefficient of term k its column's entry k: the
    row a number in every read of the three stars -- an offset of the
    column's own, where the needs' rotation put it at an offset of the
    lane's, a 64-bit address add a read."""
    kernel = _kernel()
    sources = re.findall(r'int32_t v\d+_src = v\d+_f \+ (\d+);', kernel)
    assert sorted(int(k) for k in sources) == list(range(1, 9))
    base = re.search(r'int32_t (v\d+_c) = v\d+_q \* 9;', kernel).group(1)
    rows = re.findall(rf'int32_t v\d+_a = (\d+) \+ {base};', kernel)
    assert sorted(int(k) for k in rows) == list(range(1, 9))
    assert 'tf_mes_' not in kernel


def _output_reads(kernel):
    """Where the kernel reads the tensor it stores to, and that name."""
    out = re.search(r'\*\(tensorforge::VectorT<float, 4>\*\)&(glb_\w+)\['
                    r'[^\]]+\] = ', kernel).group(1)
    return [m.start() for m in re.finditer(
        re.escape(f'= *(tensorforge::VectorT<float, 4>*)&{out}['), kernel)]


def test_the_output_is_read_ahead_of_the_walk():
    """The column's sums start at zero and the output's tiles are read
    ahead of the first shuffle (`slot_output_read` `ahead`): read behind it
    with nothing around them, they were read in the stores' branch, one
    tile ahead of the lane's branch and four behind it."""
    kernel = _kernel()
    reads = _output_reads(kernel)
    assert len(reads) == 5
    assert max(reads) < kernel.index('tensorforge::gather<32>(')


def test_masked_the_output_is_read_with_its_stores_in_one_block():
    """`masked`: behind the last shuffle, the five tiles' reads, their adds
    and their stores one block under the lane's mask, the stores in it
    unguarded."""
    kernel = _kernel(slot_output_read='masked')
    reads = _output_reads(kernel)
    assert len(reads) == 5
    assert kernel.count('if (batchIdActive0) {') == 1
    block = kernel.index('if (batchIdActive0) {')
    assert kernel.rindex('tensorforge::gather<32>(') < block < min(reads)
    tail = kernel[block:]
    end = tail.index('\n        }\n')
    stores = re.findall(r'^\s*\*\(tensorforge::VectorT<float, 4>\*\)&glb_\w+'
                        r'\[[^\]]+\] = ', tail[:end], re.M)
    assert len(stores) == 5


def test_masked_a_phase_whose_copies_read_its_column_reads_ahead():
    """The derivative's phases each have scaled copies that read the column
    as written, behind its block: under `masked` the kernel is the one
    `ahead` gives, its name and options line aside."""
    def body(kernel):
        return re.sub(r'kernel_\w+|// options:.*', '', kernel)
    assert body(_kernel('derivative', 4, 's', slot_output_read='masked')) \
        == body(_kernel('derivative', 4, 's'))


def test_every_fma_is_a_statement_where_it_is_computed():
    """A row's accumulations stand behind its shuffles and sums, one FMA a
    statement, as the prototype wrote them -- not one expression a row of
    the output at its write, which held every row's sums to the end.  (A
    sum's first product is folded into its second term.)"""
    kernel = _kernel(order=4, precision='d')
    values = [line for line in kernel.splitlines()
              if re.match(r'\s*double v\d+_\w+ = ', line)]
    assert len(values) > 400
    products = [line for line in values if line.count(' * ') > 2]
    assert not products, products[:2]


# -- built and run ----------------------------------------------------------- #

@pytest.mark.parametrize('read', ['ahead', 'masked'])
def test_compiled_for_sm90a_a_row_is_nine_broadcasts(tmp_path, read):
    """ptxas keeps the plan: 90 SHFL.IDX at o4, a lane's own loads, no
    shared memory -- and the coefficients' reads at immediate offsets: 55
    instructions of address arithmetic in the kernel with nvcc 13.4, 171
    under the needs' rotation (l9b), the lanes prototype's L9 about 34.  And
    the output's five tiles read with no branch among them, ahead of the
    walk or in their masked block: read behind it with nothing around them
    (l9c), ptxas read the first ahead of the lane's branch and the other
    four behind it."""
    import shutil
    import subprocess
    import test_multi_element_async as tma
    from tensorforge import toolchain
    nvcc = tma._nvcc_builds('sm_90a')
    context = me._context('sm_90a', 's', **SLOTS, slot_output_read=read)
    gen = me._generate(me._descrs('volume', 4, 's'), context)
    headers = list(context.target.headers()) + list(gen.get_helper_headers())
    src = tmp_path / 'kernel.cu'
    src.write_text(''.join(f'#include "{h}"\n' for h in headers)
                   + '#include "tensorforge_aux.h"\n' + gen.get_kernel()
                   + '\n' + gen.get_launcher())
    entry = toolchain.COMPILERS['cuda']
    obj = tmp_path / 'kernel.o'
    run = subprocess.run([nvcc, *entry.language_flags(),
                          *entry.target_flags('sm_90a'), '-I',
                          str(tma.INCLUDE), '-c', str(src), '-o', str(obj)],
                         capture_output=True, text=True, timeout=1200)
    assert run.returncode == 0, run.stderr[-4000:]
    tool = shutil.which('cuobjdump')
    if tool is None:
        pytest.skip('cuobjdump not found')
    sass = subprocess.run([tool, '-sass', str(obj)], capture_output=True,
                          text=True, timeout=600).stdout
    assert len(re.findall(r'\bSHFL\.IDX\b', sass)) == 90
    assert not re.search(r'\b(LDS|STS)\b', sass)
    ops = re.findall(r'/\*[0-9a-f]{4,}\*/\s+(?:@!?U?P[0-9T] )?([A-Z0-9_]+)',
                     sass)
    assert sum(op in ('IADD3', 'LEA', 'IMAD', 'SHF', 'LOP3')
               for op in ops) <= 100
    lines = [ln for ln in sass.splitlines()
             if re.match(r'\s+/\*[0-9a-f]{4,}\*/', ln)]
    tiles = [n for n, ln in enumerate(lines) if 'LDG.E.128' in ln][-5:]
    assert not [ln for ln in lines[tiles[0]:tiles[-1]]
                if re.search(r'\bBRA\b', ln)]


ORACLE = [('volume', 4, 's'), ('volume', 4, 'd'), ('volume', 6, 'd'),
          ('sparse-star', 4, 's'), ('sparse-star', 4, 'd'),
          ('diagonal-star', 4, 'd'),
          ('localFlux', 4, 's'), ('neighboringFlux', 4, 'd'),
          ('localFluxAll', 4, 's'), ('columns', 4, 'd'),
          ('direction', 4, 's'), ('assign', 4, 'd'), ('assign-low', 4, 's'),
          ('window', 6, 's'), ('plConvertToModal', 4, 's')]


#: The output read in one masked block with its stores (`slot_output_read`
#: `masked`): kept rows of a partial tile, windows and assigned outputs.
MASKED = [('volume', 4, 's'), ('volume', 6, 'd'), ('sparse-star', 4, 'd'),
          ('diagonal-star', 4, 'd'), ('localFlux', 4, 's'),
          ('localFluxAll', 4, 's'), ('columns', 4, 'd'), ('assign', 4, 'd'),
          ('assign-low', 4, 's'), ('window', 6, 's'),
          ('plConvertToModal', 4, 's')]
TRIPS = ([(s, o, p, 'ahead') for s, o, p in ORACLE]
         + [(s, o, p, 'masked') for s, o, p in MASKED])


@pytest.mark.parametrize('shape,order,precision,read', TRIPS,
                         ids=[f'{s}-o{o}{p}' + ('-masked' if r == 'masked'
                                                else '')
                              for s, o, p, r in TRIPS])
def test_every_element_of_two_trips_on_the_oracle(shape, order, precision,
                                                  read):
    """The group over itself and seven more elements on the host oracle:
    the second trip's first two warps whole, its third one element of three
    -- the rest of its lanes masked -- and its fourth masked; every
    element's output against NumPy, from the oracle's own fill of memory,
    the operators their description's numbers; and no access to shared
    memory that no barrier orders.  Dense and sparse stars, the fluxes'
    held columns, windows of the output's columns and rows, assigned
    outputs, a section of operators alone; the output read ahead of the
    walk and in its masked block."""
    import numpy as np
    gen = me._generate(_descrs(shape, order, precision),
                       me._context('sm_90', precision, **SLOTS,
                                   slot_output_read=read))
    built = gen.built._element_group
    assert built.lanes is not None and built.lanes.mapping is not None
    count = built.elements + 7
    tensors, inputs, _, after = me._on_the_oracle(gen, count)
    output = next(t for t in tensors.values() if not t.is_tmp
                  and t.addressing.kind is not Addressing.NONE
                  and any(d.writes().tensor is t
                          for outer in gen.descr_list
                          for d in outer.operations()))
    got = after(output)
    want = me._reference(_descrs(shape, order, precision))(
        inputs, inputs[output.alias])
    tolerance = 1e-12 if precision == 'd' else 1e-6
    for e in range(count):
        scale = np.abs(want[e]).max() or 1.0
        error = np.abs(got[e] - want[e]).max() / scale
        assert error <= tolerance, (e, error)


CHAINED = [('derivative', 4, 's'), ('derivative', 4, 'd'),
           ('derivative', 6, 's'), ('plConvertToNodal', 4, 's'),
           ('copied', 4, 's'), ('copied', 4, 'd')]


@pytest.mark.parametrize('shape,order,precision', CHAINED,
                         ids=[f'{s}-o{o}{p}' for s, o, p in CHAINED])
def test_a_chained_section_on_the_oracle(shape, order, precision):
    """A chained section's phases on the host oracle over a group and seven
    elements more: the derivative -- the time integral and every order's
    derivative, each order's mixing reading back the order before -- a copy
    added to a product's output after it, and an output that accumulated
    copied from the column the lane holds; every output against NumPy."""
    import numpy as np
    descrs = _descrs(shape, order, precision)
    gen = me._generate(descrs, me._context('sm_90', precision, **SLOTS))
    group = gen.built._element_group
    assert group.lanes is not None and group.lanes.mapping is not None
    assert group.lanes.phases
    count = group.elements + 7
    tensors, inputs, scalars, after = me._on_the_oracle(gen, count)
    want = me._reference_all(descrs)(inputs, scalars)
    tolerance = 1e-12 if precision == 'd' else 1e-6
    for alias in want:
        got = after(tensors[alias])
        for e in range(count):
            scale = np.abs(want[alias][e]).max() or 1.0
            error = np.abs(got[e] - want[alias][e]).max() / scale
            assert error <= tolerance, (alias, e, error)


@pytest.mark.parametrize('order,precision', [
    (4, 's'), (6, 's'), (4, 'd'), (6, 'd')])
def test_the_slot_forms_derivative_computes_on_the_gpu(
        order, precision, target, cache_root, tensorforge_include):
    """The derivative in the slot form built and run on the local GPU
    against NumPy, over a batch whose last group is partial and which gives
    a block of the persistent grid several trips: the time integral, which
    sums every order, and the last order's derivative, the chain's end."""
    from harness.runner import run_case
    if (target.backend, target.vendor) not in eg.SLOT_TARGETS:
        pytest.skip(f'the slot form is not written for {target.backend} on '
                    f'{target.vendor}')
    group = eg.plan(me._descrs('derivative', order, precision),
                    me._context(target.arch, precision,
                                backend=target.backend, **SLOTS))
    assert group is not None and group.lanes.phases
    for output in ('I', f'dQ({order - 1})'):
        result = run_case(
            me._derivative_case(order, precision, output, SLOTS,
                                name='multi_element_slots'),
            target, cache_root, tensorforge_include)
        assert result.passed, (output, result.max_abs_err,
                               result.max_rel_err, result.detail)


SLOT_CASES = [
    ('volume', 4, 's'), ('volume', 6, 's'), ('volume', 4, 'd'),
    ('volume', 6, 'd'), ('sparse-star', 4, 's'), ('sparse-star', 6, 's'),
    ('sparse-star', 6, 'd'), ('localFlux', 6, 'd'),
    ('neighboringFlux', 4, 's'), ('localFluxAll', 6, 's'),
    ('localFluxAll', 4, 'd'), ('direction', 6, 'd'), ('columns', 6, 's')]


#: Beside the defaults: the output read in its masked block, the grid
#: covering the batch in one round (`one_round_grid`), one trip a block, and
#: `multi_element=auto` where it takes the slot form and where it declines.
SLOT_VARIANTS = [
    ('volume', 4, 's', {'slot_output_read': 'masked'}),
    ('volume', 6, 'd', {'slot_output_read': 'masked'}),
    ('sparse-star', 6, 's', {'slot_output_read': 'masked'}),
    ('localFluxAll', 4, 'd', {'slot_output_read': 'masked'}),
    ('volume', 4, 's', {'one_round_grid': True}),
    ('localFlux', 6, 'd', {'one_round_grid': True,
                           'slot_output_read': 'masked'}),
    # `auto`: the slot form taken, and declined for the per-element kernel.
    ('volume', 6, 's', {'multi_element': 'auto'}),
    ('localFlux', 6, 's', {'multi_element': 'auto'})]
GPU_CASES = [(s, o, p, {}) for s, o, p in SLOT_CASES] + SLOT_VARIANTS


def _case_id(shape, order, precision, options):
    short = {'one_round_grid': '1r', 'multi_element': 'auto'}
    suffix = ''.join(f'-{short.get(k, v)}' for k, v in sorted(options.items()))
    return f'{shape}-o{order}{precision}{suffix}'


@pytest.mark.parametrize('shape,order,precision,options', GPU_CASES,
                         ids=[_case_id(*c) for c in GPU_CASES])
def test_the_slot_form_computes_on_the_gpu(shape, order, precision, options,
                                           target, cache_root,
                                           tensorforge_include):
    """Built and run on the local GPU against NumPy, over a batch whose last
    group is partial and which gives a block of the persistent grid several
    trips -- or, in one round, every block one; the operators the
    description's numbers, which its literals are.  F32 held to the
    outputs' size as the lanes form is."""
    from harness.runner import run_case
    if (target.backend, target.vendor) not in eg.SLOT_TARGETS:
        pytest.skip(f'the slot form is not written for {target.backend} on '
                    f'{target.vendor}')
    group = eg.plan(me._descrs(shape, order, precision),
                    me._context(target.arch, precision,
                                backend=target.backend, **SLOTS))
    assert group is not None and group.lanes.mapping is not None
    case = types.SimpleNamespace(
        NAME='multi_element_slots_' + _case_id(
            shape, order, precision, options).replace('-', '_'),
        DTYPE=me._dtype(precision), BATCH=4099,
        TOL=(1e-12, 1e-12) if precision == 'd' else (1e-4, 1e-2),
        OPTIONS={**SLOTS, **options},
        INPUT_TRANSFORM=me._described(me._descrs(shape, order, precision)),
        descr_list=lambda: me._descrs(shape, order, precision),
        reference=me._reference(me._descrs(shape, order, precision)))
    result = run_case(case, target, cache_root, tensorforge_include)
    assert result.passed, (result.max_abs_err, result.max_rel_err,
                           result.detail)
