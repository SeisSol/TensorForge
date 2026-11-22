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


def test_a_dense_star_takes_eight_shuffles_a_row():
    """Every quantity of the element from the other eight lanes: as many
    shuffles as the most a lane needs, each lane receiving each of its
    element's other quantities once, from that quantity's lane -- a rotation
    of the nine -- and the idle lanes reading themselves."""
    m = eg.split_mapping(9, 9, 32)
    x = eg.plan_exchange(m, _needs(m, lambda a, b: True))
    assert (x.count, x.bound) == (8, 8)
    assert set(x.kinds) == {'lanes'}
    for lane in range(27):
        e, q = divmod(lane, 9)
        got = [x.received[k][lane] for k in range(8)]
        assert got == [(q + k + 1) % 9 for k in range(8)]
        assert [x.sources[k][lane] for k in range(8)] == [9 * e + g
                                                          for g in got]
    assert all(x.sources[k][lane] == lane and x.received[k][lane] == -1
               for k in range(8) for lane in range(27, 32))
    assert all(x.full(k, m) for k in range(8))


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
    eight shuffles and every term in each product."""
    (group, why), section, _ = _plan('volume', order, precision, arch)
    assert group is not None, why
    plan = group.lanes
    assert (group.elements, group.warps, group.threads) == (12, 4, 32)
    assert plan.mapping == eg.split_mapping(9, 9, 32)
    assert plan.exchange.count == 8
    mixing = [p for s in section.steps for p in s.products
              if p.kind is eg.Kind.MIXING]
    assert all(eg.slot_terms(plan, p, 0) == [-1] + list(range(8))
               for p in mixing)
    assert all(s.lowering == 'lane' for s in group.section.steps)


def test_the_sparse_star_drops_the_own_term():
    """SeisSol's star stores no diagonal: no lane multiplies its own
    column, three shuffles a row, three terms a product."""
    (group, why), section, _ = _plan('sparse-star', 6, 's')
    assert group is not None, why
    plan = group.lanes
    assert plan.exchange.count == 3
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
        p.mapping == eg.split_mapping(9, 9, 32) and p.exchange.count == 8
        for p in plans[1:])
    assert [p.ranges for p in plans[1:]] == [
        ((0, 35),), ((0, 20),), ((0, 10),), ((0, 4),), ((0, 1),)]


def test_the_tuner_tries_the_slot_forms_register_caps():
    """`min_blocks_per_sm` 1, 2 and 3 where the kernel takes the slot form
    (255, 255 and 168 registers a thread of its block of four warps), and
    no such knob where it does not."""
    from tensorforge.generators import tuning
    descrs = me._descrs('volume', 6, 's')
    knobs = {k.name: k for k in tuning.space(
        descrs, me._context('sm_90', 's', **SLOTS))}
    assert list(knobs['min_blocks_per_sm'].values(
        tuning.Candidate())) == [1, 2, 3]
    assert 'min_blocks_per_sm' not in {k.name for k in tuning.space(
        descrs, me._context('sm_90', 's', multi_element='force',
                            multi_element_fma='lanes'))}


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


@pytest.mark.parametrize('shape,order,precision,shuffles,rows', [
    ('volume', 4, 's', 8, 10), ('volume', 6, 'd', 8, 35),
    ('sparse-star', 4, 'd', 3, 10)])
def test_a_row_of_the_star_is_its_shuffles(shape, order, precision,
                                           shuffles, rows):
    """A shuffle a planned exchange a row of `I` -- the shuffles shared by
    the three directions -- from a table of source lanes each; no shared
    memory."""
    kernel = _kernel(shape, order, precision)
    assert kernel.count('tensorforge::gather<32>(') == shuffles * rows
    tables = re.findall(r'__device__ const int32_t (tf_mes_\w+)\[32\]',
                        kernel)
    assert len(tables) == shuffles
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


def test_the_output_is_read_behind_the_walk():
    """The column's sums start at zero, and the output's tiles are read
    only after the last shuffle -- held from the start, they cost the o6
    volume a spill in F64."""
    kernel = _kernel()
    out = re.search(r'\*\(tensorforge::VectorT<float, 4>\*\)&(glb_\w+)\['
                    r'[^\]]+\] = ', kernel).group(1)
    first = kernel.index(f'= *(tensorforge::VectorT<float, 4>*)&{out}[')
    assert first > kernel.rindex('tensorforge::gather<32>(')


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

def test_compiled_for_sm90a_a_row_is_eight_shuffles(tmp_path):
    """ptxas keeps the plan: 80 SHFL.IDX at o4, a lane's own loads, no
    shared memory."""
    import shutil
    import subprocess
    import test_multi_element_async as tma
    from tensorforge import toolchain
    nvcc = tma._nvcc_builds('sm_90a')
    context = me._context('sm_90a', 's', **SLOTS)
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
    assert len(re.findall(r'\bSHFL\.IDX\b', sass)) == 80
    assert not re.search(r'\b(LDS|STS)\b', sass)


ORACLE = [('volume', 4, 's'), ('volume', 4, 'd'), ('volume', 6, 'd'),
          ('sparse-star', 4, 's'), ('sparse-star', 4, 'd'),
          ('diagonal-star', 4, 'd'),
          ('localFlux', 4, 's'), ('neighboringFlux', 4, 'd'),
          ('localFluxAll', 4, 's'), ('columns', 4, 'd'),
          ('direction', 4, 's'), ('assign', 4, 'd'), ('assign-low', 4, 's'),
          ('window', 6, 's'), ('plConvertToModal', 4, 's')]


@pytest.mark.parametrize('shape,order,precision', ORACLE,
                         ids=[f'{s}-o{o}{p}' for s, o, p in ORACLE])
def test_every_element_of_two_trips_on_the_oracle(shape, order, precision):
    """The group over itself and seven more elements on the host oracle:
    the second trip's first two warps whole, its third one element of three
    -- the rest of its lanes masked -- and its fourth masked; every
    element's output against NumPy, from the oracle's own fill of memory,
    the operators their description's numbers; and no access to shared
    memory that no barrier orders.  Dense and sparse stars, the fluxes'
    held columns, windows of the output's columns and rows, assigned
    outputs, a section of operators alone."""
    import numpy as np
    gen = me._generate(_descrs(shape, order, precision),
                       me._context('sm_90', precision, **SLOTS))
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


@pytest.mark.parametrize('shape,order,precision', SLOT_CASES,
                         ids=[f'{s}-o{o}{p}' for s, o, p in SLOT_CASES])
def test_the_slot_form_computes_on_the_gpu(shape, order, precision, target,
                                           cache_root, tensorforge_include):
    """Built and run on the local GPU against NumPy, over a batch whose last
    group is partial and which gives a block of the persistent grid several
    trips; the operators the description's numbers, which its literals are.
    F32 held to the outputs' size as the lanes form is."""
    from harness.runner import run_case
    if (target.backend, target.vendor) not in eg.SLOT_TARGETS:
        pytest.skip(f'the slot form is not written for {target.backend} on '
                    f'{target.vendor}')
    group = eg.plan(me._descrs(shape, order, precision),
                    me._context(target.arch, precision,
                                backend=target.backend, **SLOTS))
    assert group is not None and group.lanes.mapping is not None
    case = types.SimpleNamespace(
        NAME=f'multi_element_slots_{shape}_o{order}{precision}',
        DTYPE=me._dtype(precision), BATCH=4099,
        TOL=(1e-12, 1e-12) if precision == 'd' else (1e-4, 1e-2),
        OPTIONS=dict(SLOTS),
        INPUT_TRANSFORM=me._described(me._descrs(shape, order, precision)),
        descr_list=lambda: me._descrs(shape, order, precision),
        reference=me._reference(me._descrs(shape, order, precision)))
    result = run_case(case, target, cache_root, tensorforge_include)
    assert result.passed, (result.max_abs_err, result.max_rel_err,
                           result.detail)
