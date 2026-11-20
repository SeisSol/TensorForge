# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""The multi-element layout (`Options.multi_element`,
`generators.element_group`, `compute.multi_element`): a block of E warps
takes E elements of the batch together, their operands staged side by side,
the batch-constant operators once for the block, and every product of the
section lowered by its kind.

Two shapes keep the dispatch general: SeisSol's volume, where the element's
small operand -- the star -- mixes the degrees of freedom ahead of the
operator, and a chain with the star at the end, the operator's product
mixed afterwards (as in the local flux and the derivative).  What is checked
here: the classification and each reason a section is refused for, the fill
rule, the kernel's shape -- one group a block, the operators staged ahead
of the loop, the barriers, the element's write masked -- that the switch off
changes nothing, and the numbers: on the host oracle, and on the local GPU
where there is one.
"""

from __future__ import annotations

import contextlib
import copy
import importlib.util
import io
import json
import re
import types
import warnings
from pathlib import Path

import pytest

from tensorforge.common.basic_types import Datatype
from tensorforge.common.context import Context
from tensorforge.common.options import Options
from tensorforge.generators import element_group as eg

ROOT = Path(__file__).resolve().parent.parent
SYSTEM = 'elastic-linearck'


def _volume_description(order=6, precision='s'):
    import seissol_suite as fx
    return fx.description(SYSTEM, f'{SYSTEM}-o{order}-{precision}',
                          'gpu_volume')


def _chain_description(order=6, precision='s', reuse=False):
    """A section with the element's small operand at the end of the chain:
    `T_d = kDivM(d) I`, then `Q += T_d star(d)`, for two directions over
    the volume's row windows -- the operator's product mixed afterwards,
    from the volume's own tensors and two temporaries as tall as `Q`.
    `reuse` gives both directions one temporary, written again after the
    first mixing read it."""
    d = json.loads(json.dumps(_volume_description(order, precision)))
    tensors = {t['name']: t for t in d['tensors']}
    ops = [op for op in d['operations'] if op['result']['name'] == 'Q']
    q = tensors['Q']
    operations = []
    for n, op in enumerate(ops[:2]):
        name = '_chain0' if reuse else f'_chain{n}'
        if not any(t['name'] == name for t in d['tensors']):
            tmp = copy.deepcopy(tensors['_tmp0'])
            tmp['name'] = name
            tmp['storage'] = copy.deepcopy(q['storage'])
            d['tensors'].append(tmp)
        rows = op['result']['bbox'][0][0], op['result']['bbox'][1][0]
        depth = op['args'][0]['bbox'][1][1]
        one = op['linear']['alpha']
        operations.append({
            'type': 'multilinear',
            'result': {'name': name, 'indices': ['k', 'q'],
                       'bbox': [[rows[0], 0], [rows[1], 9]],
                       'offset': [0, 0], 'offset_from': None,
                       'sliced': False},
            'args': [{'name': op['args'][0]['name'], 'indices': ['k', 'l'],
                      'bbox': [[rows[0], 0], [rows[1], depth]],
                      'offset': [0, 0], 'offset_from': None,
                      'sliced': False},
                     {'name': 'I', 'indices': ['l', 'q'],
                      'bbox': [[0, 0], [depth, 9]], 'offset': [0, 0],
                      'offset_from': None, 'sliced': False}],
            'condition': [], 'permute': [[0, 1], [0, 1]],
            'target': [[0, -1], [-1, 1]],
            'linear': {'alpha': one, 'add': False}})
        operations.append({
            'type': 'multilinear',
            'result': {'name': 'Q', 'indices': ['k', 'p'],
                       'bbox': [[rows[0], 0], [rows[1], 9]],
                       'offset': [0, 0], 'offset_from': None,
                       'sliced': False},
            'args': [{'name': name, 'indices': ['k', 'q'],
                      'bbox': [[rows[0], 0], [rows[1], 9]],
                      'offset': [0, 0], 'offset_from': None,
                      'sliced': False},
                     {'name': f'star({n})', 'indices': ['q', 'p'],
                      'bbox': [[0, 0], [9, 9]], 'offset': [0, 0],
                      'offset_from': None, 'sliced': False}],
            'condition': [], 'permute': [[0, 1], [0, 1]],
            'target': [[0, -1], [-1, 1]],
            'linear': {'alpha': one, 'add': [0, 1]}})
    d['operations'] = operations
    return d


def _seissol(kernel):
    """A kernel of the SeisSol suite as yateto sent it."""
    def description(order=6, precision='s'):
        import seissol_suite as fx
        return fx.description(SYSTEM, f'{SYSTEM}-o{order}-{precision}',
                              kernel)
    return description


def _direction_description(order=6, precision='s'):
    """The volume's first direction alone, `Q += kDivM(0) (I star(0))`: an
    accumulating operator step with rows the operator has no nonzero in."""
    d = json.loads(json.dumps(_volume_description(order, precision)))
    d['operations'] = d['operations'][:2]
    return d


def _columns_description(order=6, precision='s'):
    """The volume with its second and third directions over their own
    columns of `Q`, 0..5 and 3..9: one step whose products each add into a
    window of the step's columns of its own."""
    d = json.loads(json.dumps(_volume_description(order, precision)))
    for n, (lo, hi) in ((3, (0, 5)), (5, (3, 9))):
        op = d['operations'][n]
        for view in (op['result'], op['args'][1]):
            view['bbox'][0][1], view['bbox'][1][1] = lo, hi
    return d


DESCRIPTIONS = {
    'volume': _volume_description, 'chain': _chain_description,
    'direction': _direction_description, 'columns': _columns_description,
    'reuse': lambda order, precision: _chain_description(order, precision,
                                                         reuse=True),
    # The flux kernels the layout takes besides the volume: a mixing
    # product, then two operator products, the first into an intermediate.
    'localFlux': _seissol('gpu_localFlux[0]'),
    'neighboringFlux': _seissol('gpu_neighboringFlux[0]'),
    'localFluxAll': _seissol('gpu_localFluxAll')}


def _descrs(shape='volume', order=6, precision='s'):
    from tensorforge.frontend.yateto import DescriptionReader
    return DescriptionReader(None, {}).read(
        DESCRIPTIONS[shape](order, precision))[0]


def _volume(order=6, precision='s'):
    return _descrs('volume', order, precision)


def _dtype(precision):
    return Datatype.F64 if precision == 'd' else Datatype.F32


def _context(arch='sm_90', precision='s', backend='cuda', **options):
    return Context(arch=arch, backend=backend, fp_type=_dtype(precision),
                   options=Options(**options))


def _generate(descrs, context):
    from tensorforge.generators.generator import Generator
    gen = Generator(descrs, context)
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter('ignore')
        gen.generate()
    return gen


# -- the classification ------------------------------------------------------ #

def _steps(section):
    return [(s.kind.value, s.lowering, s.dest.alias, s.assigns,
             [(p.first.tensor.alias, p.second.tensor.alias) for p in
              s.products]) for s in section.steps]


@pytest.mark.parametrize('order,precision,basis,depth', [
    (4, 's', 20, 10), (6, 's', 56, 35), (6, 'd', 56, 35), (8, 's', 120, 84)])
def test_the_volume_mixes_first_and_sums_the_operators(order, precision,
                                                       basis, depth):
    """Three mixing products into an intermediate each, then the three
    operator products as one step into the element's own `Q`."""
    section, why = eg.classify(_volume(order, precision))
    assert section is not None, why
    assert _steps(section) == [
        ('mixing', 'rows', f'_tmp{2 * d}', True, [('I', f'star({d})')])
        for d in range(3)] + [
        ('operator', 'fma', 'Q', False,
         [(f'kDivM({d})', f'_tmp{2 * d}') for d in range(3)])]
    assert section.basis == basis and section.dtype is _dtype(precision)
    assert [t.alias for t in section.operators] == [
        'kDivM(0)', 'kDivM(1)', 'kDivM(2)']
    assert [(b.tensor.alias, b.role) for b in section.inputs] == [
        ('I', eg.Role.GROUP)] + [(f'star({d})', eg.Role.SMALL)
                                 for d in range(3)]
    assert section.buffer(section.inputs[0].tensor).rows == (0, depth)
    assert [t.alias for t in section.outputs] == ['Q']


def test_the_chain_applies_the_operator_first_and_mixes_last():
    """The second shape: each operator product into an intermediate, then
    the two mixing products as one step into `Q`, each over its own rows."""
    section, why = eg.classify(_descrs('chain'))
    assert section is not None, why
    assert _steps(section) == [
        ('operator', 'fma', '_chain0', True, [('kDivM(0)', 'I')]),
        ('operator', 'fma', '_chain1', True, [('kDivM(1)', 'I')]),
        ('mixing', 'rows', 'Q', False,
         [('_chain0', 'star(0)'), ('_chain1', 'star(1)')])]
    rows = [p.rows for p in section.steps[-1].products]
    assert rows == [(1, 54), (1, 55)] and section.steps[-1].rows == (1, 55)
    assert [b.role for b in section.intermediates] == [
        eg.Role.INTERMEDIATE] * 2


def test_a_reused_intermediate_splits_the_steps():
    """One temporary for both directions: the first mixing runs before the
    temporary is written again, and the two mixings into `Q` stay apart."""
    section, why = eg.classify(_descrs('reuse'))
    assert section is not None, why
    assert _steps(section) == [
        ('operator', 'fma', '_chain0', True, [('kDivM(0)', 'I')]),
        ('mixing', 'rows', 'Q', False, [('_chain0', 'star(0)')]),
        ('operator', 'fma', '_chain0', True, [('kDivM(1)', 'I')]),
        ('mixing', 'rows', 'Q', False, [('_chain0', 'star(1)')])]


def test_a_small_operand_is_small(monkeypatch):
    """An element's own operand wider than `SMALL_EXTENT` mixes nothing."""
    monkeypatch.setattr(eg, 'SMALL_EXTENT', 8)
    assert 'too large for a small operand' in eg.classify(_volume())[1]


def _changed(change):
    descrs = copy.deepcopy(_volume())
    change(descrs)
    return eg.classify(descrs)[1]


def _per_element_operator(descrs):
    from tensorforge.common.basic_types import Addressing
    descrs[1].ops[0].tensor.addressing = Addressing.PTR_BASED


def _transposed(descrs):
    descrs[1].target = [[-1, 0], [1, -1]]


def _scaled(descrs):
    descrs[1].ops = list(descrs[1].ops) + [descrs[1].ops[0]]


def _guarded(descrs):
    descrs[0].condition = ['x']


def _reads_its_output(descrs):
    """The second star product reads `Q`, over the window it read of `I`."""
    from tensorforge.common.matrix.tensor import SubTensor
    descrs[2].ops = [SubTensor(descrs[1].dest.tensor, descrs[2].ops[0].bbox),
                     descrs[2].ops[1]]


@pytest.mark.parametrize('change,why', [
    (lambda d: setattr(d[1], 'add', False), 'assigns a window of Q'),
    (lambda d: setattr(d[0], 'add', True), 'which nothing assigned'),
    (_per_element_operator, 'neither an operator'),
    (lambda d: d.insert(0, d.pop(1)), 'before anything writes it'),
    (_transposed, 'in their order'),
    (_scaled, 'not a product of two matrices'),
    (_guarded, 'is guarded'),
    (_reads_its_output, 'written and read'),
], ids=['assigns-a-window', 'accumulates-unassigned', 'operator-per-element',
        'order', 'transposed', 'scaled', 'guarded', 'reads-its-output'])
def test_what_the_layout_does_not_take_says_why(change, why):
    assert why in _changed(change)


# -- the fill rule ----------------------------------------------------------- #

@pytest.mark.parametrize('shape,order,precision,arch,elements,lowering', [
    ('volume', 6, 's', 'sm_90', 16, 'mma_sync'),
    ('volume', 6, 'd', 'sm_90', 8, 'dmma'),
    ('volume', 6, 'd', 'sm_80', 4, 'dmma'),
    ('volume', 6, 's', 'sm_120', 8, 'mma_sync'),
    ('volume', 6, 'd', 'sm_120', 4, 'fma'),
    ('volume', 4, 's', 'sm_90', 32, 'mma_sync'),
    ('volume', 8, 's', 'sm_90', 4, 'mma_sync'),
    ('chain', 6, 's', 'sm_90', 16, 'mma_sync'),
    ('chain', 6, 's', 'sm_120', 8, 'mma_sync')])
def test_the_group_is_the_tables_halved_until_it_fits(shape, order,
                                                      precision, arch,
                                                      elements, lowering):
    """The design's size for the basis and the precision, halved while the
    operators and the group's buffers do not fit a block, the last size
    tried the smallest; the matrix tiling where a group of it fits, the FMA
    forms where none does (o6 F64 on sm_120's 99 kB)."""
    section = eg.classify(_descrs(shape, order, precision))[0]
    group, why = eg.fill(section, _context(arch, precision))
    assert group is not None, why
    assert group.elements == elements
    operator = next(s for s in group.section.steps
                    if s.kind is eg.Kind.OPERATOR)
    assert operator.lowering == lowering
    assert group.shared_bytes == eg.shared_bytes(group.section, elements)
    assert group.shared_bytes <= (_context(arch, precision).target.hw
                                  .max_local_mem_size_per_block)


@pytest.mark.parametrize('options,default,lowering', [
    ({}, True, 'mma_sync'), ({}, False, 'fma'),
    ({'tensor_cores': True}, False, 'mma_sync'),
    ({'tensor_cores': False}, True, 'fma')],
    ids=['default-on', 'default-off', 'asked-on', 'asked-off'])
def test_the_matrix_engines_follow_the_switch(options, default, lowering,
                                              monkeypatch):
    """`Options.tensor_cores` where it is set, the layout's own `MATRIX`
    where it is not; the mixing steps stay on the FMA units either way."""
    monkeypatch.setattr(eg, 'MATRIX', default)
    section = eg.classify(_volume())[0]
    group = eg.fill(section, _context(**options))[0]
    assert [s.lowering for s in group.section.steps] == ['rows'] * 3 + [
        lowering]


@pytest.mark.parametrize('precision,depth,ld', [('s', 40, 56), ('d', 36, 60)])
def test_a_product_reads_its_window_of_the_operator(precision, depth, ld):
    """Each operator product's B is its own window, the step's rows its
    columns from the step's first: the contraction padded to the
    instruction's depth, the row length to a stride that meets no bank
    twice; the operators as stored are not staged where nothing reads
    them."""
    section = eg.classify(_volume(6, precision))[0]
    group = eg.fill(section, _context('sm_90', precision))[0]
    stages = group.section.stages
    assert [(st.tensor.alias, st.window) for st in stages] == [
        (f'kDivM({d})', ((1, 54 + d), (0, 35))) for d in range(3)]
    assert {(st.n0, st.depth, st.ld) for st in stages} == {(1, depth, ld)}
    assert [(b.tensor.alias, b.rows, b.cols) for b in group.section.results] \
        == [('Q', (1, 56), (0, 9))]


def test_an_operator_too_large_for_a_block_is_refused():
    section = eg.classify(_volume(8, 's'))[0]
    group, why = eg.fill(section, _context('sm_120'))
    assert group is None and 'streamed' in why


# -- the operators' nonzeros (the sparse FMA form) ---------------------------- #

def _fma_section(shape='volume', order=6, precision='s', **options):
    section = eg.classify(_descrs(shape, order, precision))[0]
    return eg.lower(section, _context('sm_90', precision, tensor_cores=False,
                                      **options))


def _nonzeros(product):
    """`(row, k)` of the product's operator that are not zero, in its
    window, from the numbers the description carries."""
    import numpy as np
    data = np.asarray(product.first.tensor.get_values())
    (r0, r1), (k0, k1) = product.rows, product.depth
    return {(i, k) for i in range(r0, r1) for k in range(k0, k1)
            if data[i, k] != 0}


@pytest.mark.parametrize('shape,order,precision', [
    ('volume', 6, 's'), ('volume', 4, 'd'), ('chain', 6, 's'),
    ('localFlux', 6, 'd'), ('localFluxAll', 4, 's')])
def test_an_fma_step_walks_its_operators_nonzeros(shape, order, precision):
    """Each operator step of the FMA form takes its terms: per product
    exactly the operator's nonzeros in the product's window, each at its
    cell of the operator's storage, the rows by how many nonzeros the
    step's products have there, the most first."""
    section = _fma_section(shape, order, precision)
    steps = [s for s in section.steps if s.kind is eg.Kind.OPERATOR]
    assert steps and all(s.lowering == 'fma' and s.terms is not None
                         for s in steps)
    for step in steps:
        terms = step.terms
        counts = {i: sum(terms.counts[p][s]
                         for p in range(len(step.products)))
                  for s, i in enumerate(terms.order)}
        assert list(terms.order) == sorted(terms.order,
                                           key=lambda i: (-counts[i], i))
        for p, product in enumerate(step.products):
            walked = []
            at = 0
            for s, i in enumerate(terms.order):
                n = terms.counts[p][s]
                walked += [(i, k) for k in terms.cols[p][at:at + n]]
                at += n
            assert at == len(terms.cols[p]) == len(terms.cells[p])
            assert sorted(walked) == sorted(_nonzeros(product))
            tensor = product.first.tensor
            assert [tensor.bbox.lower()[0] + c % tensor.bbox.sizes()[0]
                    for c in terms.cells[p]] == [i for i, _ in walked]
            assert [tensor.bbox.lower()[1] + c // tensor.bbox.sizes()[0]
                    for c in terms.cells[p]] == [k for _, k in walked]


def test_an_accumulating_step_leaves_rows_without_terms_out():
    """Adding nothing to a row of the output is not writing it -- the
    volume's first direction alone, where `kDivM(0)` is zero in 18 of its
    53 rows; assigning an intermediate writes its zeros -- the chain's
    first, the same operator into `_chain0`."""
    adds = _fma_section('direction', 6, 's').steps[-1]
    nonzero = {i for p in adds.products for i, _ in _nonzeros(p)}
    assert not adds.assigns and len(nonzero) == 53 - 18
    assert sorted(adds.terms.order) == sorted(nonzero)
    assigns = _fma_section('chain', 6, 's').steps[0]
    assert assigns.assigns and assigns.terms.counts[0][-1] == 0
    assert sorted(assigns.terms.order) == list(range(*assigns.rows))


def test_the_terms_row_table():
    """A row's data: the row, then each product's first nonzero in the
    tables and its count; the products' nonzeros one after another."""
    terms = eg.Terms(order=(5, 2), counts=((2, 1), (0, 3)),
                     cols=((0, 3, 1), (4, 5, 6)), cells=((), ()))
    assert terms.entries == 6 and terms.base(1) == 3 and terms.stride == 5
    assert terms.rows_table() == (5, 0, 2, 3, 0, 2, 2, 1, 3, 3)
    assert terms.volume == 2 * 6 + 2 * 5


@pytest.mark.parametrize('options', [
    {'skip_known_zeros': False}, {'tensor_cores': True}],
    ids=['zeros-not-skipped', 'matrix'])
def test_no_terms_without_the_fma_form_skipping_zeros(options):
    """`skip_known_zeros` off keeps the operators' boxes, as stored, and the
    matrix tiling reads its own stages."""
    section = eg.classify(_volume())[0]
    lowered = eg.lower(section, _context('sm_90', **{'tensor_cores': False,
                                                      **options}))
    assert all(s.terms is None for s in lowered.steps)


def test_an_operator_without_numbers_is_read_whole(monkeypatch):
    """Where the description gives no numbers for an operator, the step's
    zeros are not known and it reads the operators as stored."""
    descrs = _volume()
    section = eg.classify(descrs)[0]
    monkeypatch.setattr(section.steps[-1].products[1].first.tensor, 'data',
                        None)
    lowered = eg.lower(section, _context('sm_90', tensor_cores=False))
    assert lowered.steps[-1].terms is None
    assert [st.tensor.alias for st in lowered.stages] == [
        f'kDivM({d})' for d in range(3)]


@pytest.mark.parametrize('precision,sparse,dense', [
    ('s', 15204, 26880), ('d', 30408, 47040)])
def test_the_terms_take_less_room_than_the_operators(precision, sparse,
                                                     dense):
    """At o6 the volume's three operators: 1708 values, as many indices,
    55 rows of row data -- in place of 3 x 35 columns of the boxes."""
    size = _dtype(precision).size()
    lowered = _fma_section(precision=precision)
    assert lowered.stages == ()
    assert eg.shared_bytes(lowered, 0) == sparse
    dense_section = _fma_section(precision=precision, skip_known_zeros=False)
    assert eg.shared_bytes(dense_section, 0) == dense
    assert sum(st.volume for st in dense_section.stages) * size == dense


def _executed(shape='volume', order=6, precision='s', **options):
    from tensorforge.analysis import pipeline
    gen = _generate(_descrs(shape, order, precision),
                    _context('sm_90', precision, multi_element='force',
                             tensor_cores=False, **options))
    return pipeline.executed_flops(gen)


def _section_flops(shape, order, precision, sparse):
    """Flops a sparse (or dense) FMA section performs an element.  A step
    over the operators' nonzeros: each nonzero once for each of the
    product's columns, the products summed in one accumulator.  A step over
    the windows (a mixing step, or an operator step without its terms):
    each product's outputs over its contraction -- the first term a
    multiplication where the contraction is written out
    (`multi_element.UNROLL`), an FMA from zero where it is a loop -- and the
    products' sums added.  A step that accumulates adds to what is
    there."""
    from tensorforge.backend.instructions.compute.multi_element import UNROLL
    section = _fma_section(shape, order, precision,
                           skip_known_zeros=sparse)
    flops = 0
    for step in section.steps:
        height = step.rows[1] - step.rows[0]
        width = step.cols[1] - step.cols[0]
        written = height * width
        if step.terms is not None:
            flops += 2 * sum(len(_nonzeros(p)) * (p.cols[1] - p.cols[0])
                             for p in step.products)
            written = len(step.terms.order) * width
        else:
            for p in step.products:
                depth = p.depth[1] - p.depth[0]
                flops += ((p.rows[1] - p.rows[0]) * (p.cols[1] - p.cols[0])
                          * (2 * depth - (depth <= UNROLL)))
            flops += written * (len(step.products) - 1)
        if not step.assigns:
            flops += written
    return flops


@pytest.mark.parametrize('shape,order,precision', [
    ('volume', 6, 's'), ('volume', 4, 'd'), ('chain', 6, 's'),
    ('localFlux', 6, 'd'), ('neighboringFlux', 4, 's'), ('columns', 4, 'd')])
def test_the_executed_flops_are_the_nonzeros(shape, order, precision):
    """The emitter's count of the sparse form (`pipeline.executed_flops`):
    every lane's term under a guard that says how many lanes take it, so
    what the kernel performs is exactly the operators' nonzeros -- and the
    dense form's is its windows'."""
    assert _executed(shape, order, precision) == _section_flops(
        shape, order, precision, True)
    assert _executed(shape, order, precision,
                     skip_known_zeros=False) == _section_flops(
        shape, order, precision, False)


def test_the_layout_is_cudas_on_nvidia():
    section = eg.classify(_volume())[0]
    group, why = eg.fill(section, _context('gfx942', backend='hip'))
    assert group is None and 'NVIDIA' in why


@pytest.mark.parametrize('options,why', [
    ({'launch_control': True}, 'hands out elements'),
    ({'stack_elements': 2}, 'stacked elements')],
    ids=['launch-control', 'stacked'])
def test_a_loop_without_groups_is_refused(options, why):
    """The batch loop's element-group trip (`Trip.ELEMENT_GROUP`) is a
    group of the block's warps on every trip: a loop that hands out
    single elements or stacks of them takes none."""
    section = eg.classify(_volume())[0]
    group, reason = eg.fill(section, _context('sm_100', **options))
    assert group is None and why in reason


def test_the_loop_takes_the_group_or_says_why():
    """An element-group loop is the block's group or an internal error, never
    the per-row loop it would otherwise fall back to."""
    from tensorforge.backend.instructions.batch_loop import (BatchLoop,
                                                             LoopMode, Trip)
    from tensorforge.backend.pir.core import Uniformity
    from tensorforge.common.exceptions import InternalError
    context = _context()

    def loop(**kw):
        args = dict(context=context, section_index=0,
                    mode=LoopMode.PERSISTENT, start='s', stride='t',
                    region=[], group_size=8, trip=Trip.ELEMENT_GROUP)
        args.update(kw)
        return BatchLoop(**args)

    grouped = loop()
    grouped.set_mults_per_block(8)
    assert grouped._grouped() and grouped.trip is Trip.ELEMENT_GROUP
    assert 'element_group(8)' in str(grouped)
    assert grouped.uniform_scope() is Uniformity.BLOCK
    with pytest.raises(InternalError, match='the group is the block'):
        loop().set_mults_per_block(16)
    with pytest.raises(InternalError, match='work queue'):
        loop(mode=LoopMode.LAUNCHCTRL)
    with pytest.raises(InternalError, match='stacked'):
        loop(stack=2)
    with pytest.raises(InternalError, match='one element'):
        loop(group_size=1)


@pytest.mark.parametrize('value', ['off', 'auto'])
def test_only_force_takes_the_layout(value):
    """`off` never, and `auto` nowhere until a measurement says where."""
    assert eg.plan(_volume(), _context(multi_element=value)) is None
    assert eg.plan(_volume(), _context(multi_element='force')) is not None


def test_the_option_is_off_by_default():
    from tensorforge.common.options import registry
    assert registry()['multi_element'].default == 'off'


# -- the lanes form ----------------------------------------------------------- #

LANES = {'multi_element': 'force', 'multi_element_fma': 'lanes'}


@pytest.mark.parametrize('shape,order,precision,classes,ranges', [
    ('volume', 4, 's', 3, ((1, 20),)), ('volume', 6, 's', 3, ((1, 56),)),
    ('volume', 6, 'd', 2, ((1, 32), (32, 37), (37, 43), (43, 56))),
    ('localFlux', 6, 'd', 3, ((0, 35), (35, 56))),
    ('localFluxAll', 4, 'd', 3, ((0, 20),))])
def test_the_lanes_form_cuts_the_output_until_a_lane_holds_it(
        shape, order, precision, classes, ranges):
    """As few row ranges as keep a lane's registers within the plan's
    budget -- one in F32, more at o6 F64, each cut where the work of its
    two parts is closest -- and three column classes where the block's
    warps allow (`LANE_WARPS`)."""
    section = eg.classify(_descrs(shape, order, precision))[0]
    lanes, why = eg.lanes_plan(section, _context('sm_90', precision))
    assert lanes is not None, why
    assert lanes.classes == classes and lanes.ranges == ranges
    assert lanes.warps == classes * len(ranges) <= eg.LANE_WARPS
    assert lanes.registers <= eg.LANE_REGISTERS
    size = 2 if precision == 'd' else 1
    tile = 16 // (8 if precision == 'd' else 4)
    cones = eg.Cones(section)
    one = cones.live(range(*cones.rows), tile) * size
    assert (one > eg.LANE_REGISTERS) is (len(ranges) > 1)


def test_a_cone_reads_the_operands_rows_its_nonzeros_name():
    """The rows each value's column needs, by the step that writes it: the
    volume's first rows read the first contraction indices of each
    direction's intermediate, the rows its own operator's nonzeros name."""
    section = eg.classify(_volume(6, 's'))[0]
    cones = eg.Cones(section)
    need = cones.need(range(1, 20))
    for p in section.steps[-1].products:
        reads = {k for i in range(1, 20)
                 for k, _ in cones.nonzeros[id(p)].get(i, ())}
        assert cones.source(p).dest is p.second.tensor
        assert need[id(cones.source(p))] == reads and max(reads) < 10
    assert need[id(section.steps[-1])] == set(range(1, 20))


def test_a_reused_intermediate_is_a_value_a_write():
    """The reused temporary: each mixing product reads what the operator
    step just ahead of it wrote, and the cones keep the two apart."""
    section = eg.classify(_descrs('reuse'))[0]
    cones = eg.Cones(section)
    first, mix0, second, mix1 = section.steps
    assert cones.source(mix0.products[0]) is first
    assert cones.source(mix1.products[0]) is second
    assert cones.source(first.products[0]) is None


@pytest.mark.parametrize('change,why', [
    ('chain', 'a mixing product of _chain0'),
    ('literals', 'more than the 64 literals'),
    ('numbers', 'carries no numbers')],
    ids=['mixes-an-intermediate', 'too-many-literals', 'no-numbers'])
def test_what_the_lanes_form_does_not_take_says_why(change, why,
                                                    monkeypatch):
    """A mixing product of an intermediate -- every column of it a lane
    would hold -- operators beyond `inline_constants`, and an operator
    without its numbers are the warp form's."""
    options = {}
    if change == 'chain':
        section = eg.classify(_descrs('chain'))[0]
    else:
        section = eg.classify(_volume())[0]
        if change == 'literals':
            options['inline_constants'] = 64
        else:
            monkeypatch.setattr(section.steps[-1].products[0].first.tensor,
                                'data', None)
    lanes, reason = eg.lanes_plan(section, _context('sm_90', **options))
    assert lanes is None and why in reason
    group = eg.fill(section, _context('sm_90', **options, **LANES))[0]
    assert group is not None and group.lanes is None


@pytest.mark.parametrize('options,lanes', [
    ({}, False), ({'multi_element_fma': 'warp'}, False),
    ({'multi_element_fma': 'lanes'}, True),
    ({'multi_element_fma': 'lanes', 'tensor_cores': True}, True)],
    ids=['auto', 'warp', 'lanes', 'lanes-over-matrix'])
def test_the_lanes_form_where_it_is_asked_for(options, lanes):
    """`multi_element_fma`: `auto` is the warp form everywhere so far, and
    `lanes` takes the lanes form ahead of the matrix tiling: a group of a
    warp's lanes, each step `lane`, nothing staged."""
    section = eg.classify(_volume())[0]
    group = eg.fill(section, _context('sm_90', **options))[0]
    assert (group.lanes is not None) is lanes
    if lanes:
        assert group.elements == eg.LANE_ELEMENTS == 32
        assert group.warps == 3 and group.shared_bytes == 0
        assert {s.lowering for s in group.section.steps} == {'lane'}
        assert group.section.stages == () and group.section.results == ()


def test_the_lane_group_loop():
    """The lanes form's trip: the lane is the element, the group the
    block's, its warps whatever their number."""
    from tensorforge.backend.instructions.batch_loop import (BatchLoop,
                                                             LoopMode, Trip)
    from tensorforge.backend.pir.core import Uniformity
    from tensorforge.common.exceptions import InternalError
    loop = BatchLoop(context=_context(), section_index=0,
                     mode=LoopMode.PERSISTENT, start='s', stride='t',
                     region=[], group_size=32, trip=Trip.LANE_GROUP)
    loop.set_mults_per_block(6)
    assert loop._grouped() and loop.uniform_scope() is Uniformity.BLOCK
    assert loop._group_start() == '32 * (blockIdx.x)'
    assert 'lane_group(32)' in str(loop)
    with pytest.raises(InternalError, match='work queue'):
        BatchLoop(context=_context(), section_index=0,
                  mode=LoopMode.LAUNCHCTRL, start='s', stride='t',
                  region=[], group_size=32, trip=Trip.LANE_GROUP)


# -- the kernel -------------------------------------------------------------- #

def _kernel(arch='sm_90', precision='s', shape='volume', **options):
    return _generate(_descrs(shape, 6, precision),
                     _context(arch, precision, **options)).get_kernel()


def test_a_block_is_one_group_of_warps():
    """E warps a block, a warp a multiplication; the launch says so."""
    kernel = _kernel(multi_element='force')
    assert 'block 32x16x1' in kernel and '32 lanes x 16 per block' in kernel


@pytest.mark.parametrize('options,names', [
    ({}, [f'bkDivM{d}' for d in range(3)]),
    ({'tensor_cores': False}, ['vQ', 'lQ', 'rQ']),
    ({'tensor_cores': False, 'skip_known_zeros': False},
     [f'kkDivM{d}' for d in range(3)])],
    ids=['matrix', 'fma', 'fma-dense'])
def test_the_operators_are_staged_once_ahead_of_the_loop(options, names):
    """As the matrix tiling reads them (`b`, a window each), as the sparse
    FMA form does -- the step's nonzeros packed: values, contraction
    indices and row data (`v`, `l`, `r`) -- or as stored for the FMA form
    without them (`k`), before the loop and behind one barrier."""
    kernel = _kernel(multi_element='force', **options)
    loop = kernel.index('batchIdGroup0')
    for name in names:
        assert -1 < kernel.index(name) < loop, name
    assert kernel[:loop].count('__syncthreads();') == 1


@pytest.mark.parametrize('options,barriers', [
    ({}, 5), ({'tensor_cores': False}, 4)], ids=['matrix', 'fma'])
def test_the_group_takes_its_trips_together(options, barriers):
    """The traversal is the group's: every warp's element from the group's
    first, the block's barriers around the staging -- and, with the matrix
    tiling, one more between D's tiles and the outputs."""
    kernel = _kernel(multi_element='force', **options)
    assert re.search(r'batchIdGroup0 \+ batchIdLane0', kernel)
    assert kernel.count('__syncthreads();') == barriers


@pytest.mark.parametrize('arch,precision,instruction,count', [
    ('sm_90', 's', 'm16n8k8.row.col.f32.tf32.tf32.f32', 45),
    ('sm_90', 'd', 'm16n8k4.row.col.f64.f64.f64.f64', 27),
    ('sm_80', 'd', 'm8n8k4.row.col.f64.f64.f64.f64', 27)])
def test_an_operator_step_is_tiled_on_the_matrix_engine(arch, precision,
                                                        instruction, count):
    """The volume's three operator products one product over K: every
    step of K of each, three products of TF32 halves in F32, one in F64."""
    kernel = _kernel(arch=arch, precision=precision, multi_element='force')
    assert kernel.count('mma.sync.aligned.') == count
    assert kernel.count('mma.sync.aligned.' + instruction) == count


def _stores(lines, pattern):
    return [i for i, line in enumerate(lines)
            if re.search(pattern + r'\[[^]]*\] = ', line)]


def test_a_reused_intermediate_waits_for_its_readers():
    """Without the barrier optimization: the block meets before the
    temporary is written again, its first readers done -- six barriers,
    the copy's, the trip's head, and one ahead of each step."""
    kernel = _kernel(shape='reuse', multi_element='force',
                     enable_sync_block_opt=False)
    lines = kernel.splitlines()
    barriers = [i for i, line in enumerate(lines)
                if line.strip() == '__syncthreads();']
    writes = _stores(lines, r'_g_chain0')
    outputs = _stores(lines, r'glb_m\d+')
    assert len(barriers) == 6
    first, second = min(writes), max(writes)
    assert any(min(outputs) < b < second for b in barriers)
    assert any(first < b < min(outputs) for b in barriers)


STAGED = r'_g(I|star\d)'
OUTPUT = r'glb_m\d+'


@pytest.mark.parametrize('shape,options,phases', [
    ('volume', {}, [STAGED, r'_g_tmp\d', r'_dQ', OUTPUT]),
    ('volume', {'tensor_cores': False}, [STAGED, r'_g_tmp\d', OUTPUT]),
    ('chain', {}, [STAGED, r'_g_chain\d', OUTPUT])],
    ids=['volume-matrix', 'volume-fma', 'chain-matrix'])
def test_the_barriers_stand_without_the_placement(shape, options, phases):
    """A build without the barrier optimization (`Options.
    enable_sync_block_opt`) keeps the instructions' own block barriers: one
    after the operators' copy, ahead of the loop, and in the trip one
    ahead of each phase whose stores another warp may still be reading --
    the staging, the intermediates, the matrix tiling's D, the outputs."""
    kernel = _kernel(shape=shape, multi_element='force',
                     enable_sync_block_opt=False, **options)
    lines = kernel.splitlines()
    loop = next(i for i, line in enumerate(lines) if 'batchIdGroup0 <' in line)
    barriers = [i for i, line in enumerate(lines)
                if line.strip() == '__syncthreads();']
    assert len(barriers) == len(phases) + 1 and barriers[0] < loop
    for n, phase in enumerate(phases):
        stores = _stores(lines, phase)
        upper = barriers[n + 2] if n + 2 < len(barriers) else len(lines)
        assert stores and barriers[n + 1] < min(stores), phase
        assert max(stores) < upper, phase


@pytest.mark.parametrize('shape,options', [
    ('volume', {}), ('chain', {}), ('volume', LANES)],
    ids=['volume', 'chain', 'volume-lanes'])
def test_an_element_past_the_end_writes_nothing(shape, options):
    """Every write of `Q` runs under the element's mask: the warp's, or in
    the lanes form the lane's."""
    lines = _kernel(shape=shape, **{'multi_element': 'force',
                                    **options}).splitlines()
    writes = _stores(lines, r'glb_m\d+')
    assert writes and all(lines[i - 1].strip() == 'if (batchIdActive0) {'
                          for i in writes)


@pytest.mark.parametrize('precision,warps', [('s', 3), ('d', 8)])
def test_a_lanes_block_takes_a_group_on_its_lanes(precision, warps):
    """The lanes form's kernel: a block of the plan's warps over a group of
    32 elements, a lane each -- the group the block's, a trip apart by the
    grid's groups -- no shared memory and no block barrier, the operators'
    pointers never read: their nonzeros are literals."""
    kernel = _kernel(precision=precision, **LANES)
    assert f'block 32x{warps}x1, 0 B shared' in kernel
    assert 'const auto batchIdLane0 = threadIdx.x;' in kernel
    assert 'batchIdGroup0 = 32 * (blockIdx.x)' in kernel
    assert 'batchIdGroup0 += (gridDim.x * 32)' in kernel
    assert '__syncthreads' not in kernel
    section = eg.classify(_volume(6, precision))[0]
    operators = {t.alias for t in section.operators}
    names = re.findall(r'//   (m\d+) \S+ \S+ none', kernel)
    assert len(names) == len(operators) == 3
    assert not any(re.search(rf'glb_{n}\[', kernel) for n in names)


def test_a_lanes_walk_reads_the_element_for_itself():
    """The rows of `I` a column reads, read in the column loop -- not ahead
    of it, held across every column -- and by each walk again: the four
    faces of `localFluxAll` read them four times, 16 bytes at a time, rather
    than one face's reads held through the other three."""
    kernel = _generate(_descrs('localFluxAll', 4, 's'),
                       _context('sm_90', 's', **LANES)).get_kernel()
    lines = kernel.splitlines()
    loop = next(i for i, line in enumerate(lines)
                if re.search(r'for \(int32_t v\d+_c = 0;', line))
    reads = [i for i, line in enumerate(lines)
             if re.search(r'VectorT<float, 4>\*\)&glb_m0\[', line)]
    # I is 20 rows by 9 at o4: five tiles of four rows a column of it, and
    # every face reads every row (its operator's contraction is the basis).
    assert reads and min(reads) > loop
    assert len(reads) == 4 * 5 * 9


def _lanes_flops(shape, order, precision):
    """Flops the lanes form performs an element: for every row range, a
    column of its cone in every column of the output -- each mixing step's
    entries at the rows the cone needs, each operator step's nonzeros
    there, the first of a row a multiplication where the column starts
    from zero, and the output's accumulated onto what it holds.  A term
    whose operand row nothing writes -- a row of an intermediate no
    nonzero or no mixing product reaches -- is a known zero and left
    out; rows of one column that start with the same literal times the
    same entry share that multiplication (the IR's common
    subexpressions), and a start that multiplies by one is folded."""
    section = eg.classify(_descrs(shape, order, precision))[0]
    lanes = eg.lanes_plan(section, _context('sm_90', precision))[0]
    cones = eg.Cones(section)
    last = section.steps[-1]
    width = last.cols[1] - last.cols[0]

    def zero(step, row):
        if step is None:
            return False
        if step.kind is eg.Kind.MIXING:
            return not any(p.rows[0] <= row < p.rows[1]
                           for p in step.products)
        return not terms(step, row)

    def live(step, row):
        """The row's terms the walk adds, in the order it adds them."""
        return sorted(((k, id(p), v) for p in step.products
                       for k, v in cones.nonzeros[id(p)].get(row, ())
                       if not zero(cones.source(p), k)),
                      key=lambda term: term[0])

    def terms(step, row):
        return len(live(step, row))

    total = 0
    for rows in lanes.ranges:
        need = cones.need(range(*rows))
        for step in section.steps:
            mine = need.get(id(step), set())
            if step is last:
                mine = {i for i in range(*rows)
                        if step.rows[0] <= i < step.rows[1]}
            starts = set()
            for i in mine:
                if step.kind is eg.Kind.MIXING:
                    covering = [p for p in step.products
                                if p.rows[0] <= i < p.rows[1]]
                    total += width * (sum(2 * (p.depth[1] - p.depth[0]) - 1
                                          for p in covering)
                                      + max(0, len(covering) - 1))
                    continue
                added = live(step, i)
                if not added:
                    continue
                if step is not last or step.assigns:
                    # The first term a multiplication, shared by the rows
                    # that start alike; the rest multiply-adds.
                    starts.add(added[0][1:] + (added[0][0],))
                    total += width * 2 * (len(added) - 1)
                else:
                    total += width * 2 * len(added)
            # A first term that multiplies by one is the entry itself.
            total += width * sum(1 for pid, value, k in starts
                                 if value != 1.0)
    return total


@pytest.mark.parametrize('shape,order,precision', [
    ('volume', 4, 's'), ('volume', 6, 'd'), ('localFlux', 6, 'd'),
    ('neighboringFlux', 4, 's'), ('localFluxAll', 4, 's'),
    ('localFluxAll', 6, 'd'), ('columns', 4, 'd')])
def test_the_lanes_forms_flops_are_its_cones(shape, order, precision):
    """The emitter's count of the lanes form: the column loop's trips and
    each range's share of the warps (`lanes_active`), divided over the
    warps an element has (`pipeline.executed_flops`) -- what every warp
    computes of every element, the operators at their nonzeros."""
    assert _executed(shape, order, precision, multi_element_fma='lanes') \
        == _lanes_flops(shape, order, precision)


@pytest.mark.parametrize('shape', ['volume', 'chain'])
def test_off_is_the_kernel_without_the_option(shape):
    """Byte for byte: the switch off builds what the default builds."""
    assert _kernel(shape=shape, multi_element='off') == _kernel(shape=shape)


def test_a_refused_plan_builds_the_default_kernel():
    """o8 on sm_120: the operators do not fit, and the kernel is the one
    each element a multiplication writes."""
    forced = _generate(_volume(8, 's'),
                       _context('sm_120', multi_element='force')).get_kernel()
    assert 'batchIdGroup0' not in forced


# -- the numbers ------------------------------------------------------------- #

def _host():
    path = ROOT / 'tools' / 'host' / 'corpus_check.py'
    spec = importlib.util.spec_from_file_location('me_corpus_check', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize('shape,options', [
    ('volume', {}), ('chain', {}), ('reuse', {}),
    ('volume', {'tensor_cores': False})],
    ids=['volume', 'chain', 'reuse', 'volume-fma'])
@pytest.mark.parametrize('precision,tolerance', [('s', 1e-6), ('d', 1e-12)])
def test_the_section_computes_on_the_oracle(shape, options, precision,
                                            tolerance):
    """The kernel run by the host interpreter against NumPy, normwise --
    the section's real operators, on sm_90: the matrix tiling, and once the
    FMA forms."""
    host = _host()
    kernel = host.Kernel(DESCRIPTIONS[shape](6, precision), _dtype(precision),
                         'sm_90')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        got, _ = kernel.run({'multi_element': 'force', **options})
    want = kernel.expected()
    for alias in {alias for alias, _ in want}:
        keys = [key for key in want if key[0] == alias]
        scale = max(abs(want[key]) for key in keys) or 1.0
        error = max(abs(got[key] - want[key]) for key in keys) / scale
        assert error <= tolerance, (alias, error)


#: The oracle's seed for its fill of memory (`kernel_eval.Slot`).
SEED = 29


@pytest.mark.parametrize('shape,order,precision,options', [
    ('volume', 4, 'd', {'tensor_cores': False}),
    ('volume', 4, 's', {'tensor_cores': False, 'skip_known_zeros': False}),
    ('reuse', 4, 's', {'tensor_cores': False}),
    ('localFlux', 4, 'd', {'tensor_cores': False}),
    ('neighboringFlux', 4, 's', {'tensor_cores': False,
                                 'enable_sync_block_opt': False}),
    ('localFluxAll', 4, 'd', {'tensor_cores': False}),
    ('direction', 4, 's', {'tensor_cores': False}),
    ('columns', 4, 'd', {'tensor_cores': False}),
    ('volume', 4, 'd', LANES), ('volume', 4, 's', LANES),
    ('localFlux', 4, 's', LANES), ('neighboringFlux', 4, 'd', LANES),
    ('localFluxAll', 4, 's', LANES), ('columns', 4, 's', LANES),
    ('direction', 4, 'd', LANES)],
    ids=['volume-o4d', 'volume-o4s-dense', 'reuse-o4s', 'localFlux-o4d',
         'neighboringFlux-o4s-own-barriers', 'localFluxAll-o4d',
         'direction-o4s', 'columns-o4d', 'lanes-volume-o4d',
         'lanes-volume-o4s', 'lanes-localFlux-o4s',
         'lanes-neighboringFlux-o4d', 'lanes-localFluxAll-o4s',
         'lanes-columns-o4s', 'lanes-direction-o4d'])
def test_every_element_of_two_trips_on_the_oracle(shape, order, precision,
                                                  options, monkeypatch):
    """A group over itself and seven more elements on the host oracle: two
    trips of the group, the second partial -- four warps over seven
    elements, one warp masked, in the warp form; a warp's lanes over 39, 25
    of them masked, in the lanes form -- every element's output against
    NumPy, from the oracle's own fill of memory, the operators their
    description's numbers; and no access to shared memory that no barrier
    orders (`kernel_eval.Races`)."""
    import numpy as np
    from tensorforge.common.basic_types import Addressing
    from tensorforge.reference import kernel_eval
    lanes = options.get('multi_element_fma') == 'lanes'
    if lanes:
        group, count = eg.LANE_ELEMENTS, eg.LANE_ELEMENTS + 7
    else:
        group, count = 4, 7
        monkeypatch.setattr(eg, 'ELEMENTS',
                            {key: group for key in eg.ELEMENTS})
        monkeypatch.setattr(eg, 'DEFAULT_ELEMENTS', group)
    gen = _generate(_descrs(shape, order, precision),
                    _context('sm_90', precision, **{'multi_element': 'force',
                                                    **options}))
    built = gen.built._element_group
    assert built.elements == group and (built.lanes is not None) is lanes
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
        """The tensor's box, element by element, as `read(element, slot)`
        gives its storage slots (column by column over the box)."""
        (r0, r1), (c0, c1) = box(t)
        out = np.zeros((count, r1 - r0, c1 - c0))
        for e in range(count):
            for c in range(c1 - c0):
                for r in range(r1 - r0):
                    out[e, r, c] = read(e, r + c * (r1 - r0))
        return out

    fill = kernel_eval.Slot(SEED)
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
            gen.get_kernel(), lanes, seed=SEED, globals_only=True,
            mults=mults, block=True, preset=preset, elements=count,
            races=races)
    assert not races, races[:3]
    got = memory(output, lambda e, s: mem.get((f'{output.name}@{e}', s),
                                              fill.read(f'{output.name}@{e}',
                                                        s)))
    want = _reference(_descrs(shape, order, precision))(
        inputs, inputs[output.alias])
    tolerance = 1e-12 if precision == 'd' else 1e-6
    for e in range(count):
        scale = np.abs(want[e]).max() or 1.0
        error = np.abs(got[e] - want[e]).max() / scale
        assert error <= tolerance, (e, error)


def _reference(descrs):
    """NumPy over the description, every element at once, in double
    precision: each product into its window, an intermediate zero outside
    what is written.  The harness's arrays are the tensors' boxes."""
    import numpy as np
    from tensorforge.generators.descriptions import MultilinearDescr
    ops = [op for d in descrs for op in d.operations()]

    def reference(inputs, dest_in):
        full = {}

        def whole(tensor, data):
            lo = tuple(int(x) for x in tensor.bbox.lower())
            hi = tuple(int(x) for x in tensor.bbox.upper())
            data = np.asarray(data, dtype=np.float64)
            out = np.zeros((data.shape[0],) + tuple(hi))
            out[:, lo[0]:hi[0], lo[1]:hi[1]] = data
            return out

        def array(tensor):
            if tensor.alias not in full:
                full[tensor.alias] = whole(tensor, inputs[tensor.alias])
            return full[tensor.alias]

        output = None
        for op in ops:
            assert isinstance(op, MultilinearDescr)
            a, b = op.ops
            (r0, k0), (r1, k1) = a.bbox.lower(), a.bbox.upper()
            c0, c1 = b.bbox.lower()[1], b.bbox.upper()[1]
            value = np.einsum('...ik,...kj->...ij',
                              array(a.tensor)[:, r0:r1, k0:k1],
                              array(b.tensor)[:, k0:k1, c0:c1])
            dest = op.dest.tensor
            if not dest.is_tmp:
                if output is None:
                    output = dest
                    full[dest.alias] = whole(dest, dest_in)
            elif not op.add:
                full[dest.alias] = np.zeros(
                    (value.shape[0],) + tuple(int(x) for x in
                                              dest.bbox.upper()))
            target = full[dest.alias]
            if op.add:
                target[:, r0:r1, c0:c1] += value
            else:
                target[:, r0:r1, c0:c1] = value
        lo = tuple(int(x) for x in output.bbox.lower())
        hi = tuple(int(x) for x in output.bbox.upper())
        return full[output.alias][:, lo[0]:hi[0], lo[1]:hi[1]]

    return reference


def _known_zeros(descrs):
    """The harness's fill of each operator that carries its numbers, zero
    where they are: a kernel that skips the known zeros reads only the rest,
    and the reference has to multiply the same matrix.  Random numbers
    elsewhere, as the harness draws them, so that a term read at another
    cell than its own is a wrong number and not, by chance, an equal one.
    The arrays are the operators' boxes."""
    import numpy as np
    from tensorforge.common.basic_types import Addressing
    out = {}
    for descr in descrs:
        for op in descr.operations():
            for sub in op.ops:
                t = getattr(sub, 'tensor', None)
                if (t is None or t.is_tmp or not t.has_values()
                        or t.addressing.kind is not Addressing.NONE):
                    continue
                box = tuple(slice(int(lo), int(hi)) for lo, hi in
                            zip(t.bbox.lower(), t.bbox.upper()))
                mask = np.asarray(t.get_values())[box] != 0
                out[t.alias] = lambda a, mask=mask: a * mask
    return out


GPU_CASES = [
    ('volume', 4, 's', {}), ('volume', 6, 's', {}), ('volume', 4, 'd', {}),
    ('volume', 6, 'd', {}),
    ('volume', 6, 's', {'enable_sync_block_opt': False}),
    ('volume', 6, 's', {'tensor_cores': False}),
    ('volume', 6, 's', {'tensor_cores': False, 'skip_known_zeros': False}),
    ('volume', 4, 'd', {'tensor_cores': False}),
    ('volume', 6, 'd', {'tensor_cores': False,
                        'enable_sync_block_opt': False}),
    ('chain', 6, 's', {}), ('chain', 6, 'd', {}), ('reuse', 6, 's', {}),
    ('chain', 6, 'd', {'tensor_cores': False}),
    ('reuse', 6, 's', {'tensor_cores': False}),
    ('localFlux', 6, 'd', {'tensor_cores': False}),
    ('neighboringFlux', 4, 's', {'tensor_cores': False}),
    ('localFluxAll', 6, 's', {'tensor_cores': False}),
    ('direction', 6, 'd', {'tensor_cores': False}),
    ('columns', 6, 's', {'tensor_cores': False})]


@pytest.mark.parametrize('shape,order,precision,options', GPU_CASES, ids=[
    'o4s', 'o6s', 'o4d', 'o6d', 'o6s-own-barriers', 'o6s-fma',
    'o6s-fma-dense', 'o4d-fma', 'o6d-fma-own-barriers', 'chain-o6s',
    'chain-o6d', 'reuse-o6s', 'chain-o6d-fma', 'reuse-o6s-fma',
    'localFlux-o6d-fma', 'neighboringFlux-o4s-fma', 'localFluxAll-o6s-fma',
    'direction-o6d-fma', 'columns-o6s-fma'])
def test_the_section_computes_on_the_gpu(shape, order, precision, options,
                                         target, cache_root,
                                         tensorforge_include):
    """Built and run on the local GPU against NumPy, over a batch that ends
    in a partial group and gives a block of the persistent grid several
    trips -- the group's buffers rewritten while another warp of the block
    may still read the last trip's -- under the barriers the placement
    derives and, once, under the instructions' own.  The operators are zero
    where the description says (`_known_zeros`), as the sparse FMA form
    takes them to be."""
    if target.backend != 'cuda':
        pytest.skip('the multi-element layout is CUDA\'s')
    from harness.runner import run_case
    # The layout's, and not the default build a refused plan leaves.
    assert eg.plan(_descrs(shape, order, precision), _context(
        target.arch, precision, multi_element='force', **options)) is not None
    suffix = ''.join(f'_{k}{int(v)}' for k, v in sorted(options.items()))
    case = types.SimpleNamespace(
        NAME=f'multi_element_{shape}_o{order}{precision}{suffix}',
        DTYPE=_dtype(precision), BATCH=4099,
        TOL=(1e-12, 1e-12) if precision == 'd' else (1e-4, 1e-4),
        OPTIONS={'multi_element': 'force', **options},
        INPUT_TRANSFORM=_known_zeros(_descrs(shape, order, precision)),
        descr_list=lambda: _descrs(shape, order, precision),
        reference=_reference(_descrs(shape, order, precision)))
    result = run_case(case, target, cache_root, tensorforge_include)
    assert result.passed, (result.max_abs_err, result.max_rel_err,
                           result.detail)


def _described(descrs):
    """The harness's fill of each operator that carries its numbers, those
    numbers: the lanes form takes them as literals, and its operators'
    memory is never read."""
    import numpy as np
    from tensorforge.common.basic_types import Addressing
    out = {}
    for descr in descrs:
        for op in descr.operations():
            for sub in op.ops:
                t = getattr(sub, 'tensor', None)
                if (t is None or t.is_tmp or not t.has_values()
                        or t.addressing.kind is not Addressing.NONE):
                    continue
                box = tuple(slice(int(lo), int(hi)) for lo, hi in
                            zip(t.bbox.lower(), t.bbox.upper()))
                values = np.asarray(t.get_values())[box]
                out[t.alias] = lambda a, values=values: np.broadcast_to(
                    values, a.shape)
    return out


LANE_CASES = [
    ('volume', 4, 's'), ('volume', 6, 's'), ('volume', 4, 'd'),
    ('volume', 6, 'd'), ('localFlux', 6, 'd'), ('neighboringFlux', 4, 's'),
    ('localFluxAll', 6, 's'), ('localFluxAll', 6, 'd'), ('direction', 6, 'd'),
    ('columns', 6, 's')]


@pytest.mark.parametrize('shape,order,precision', LANE_CASES,
                         ids=[f'{s}-o{o}{p}' for s, o, p in LANE_CASES])
def test_the_lanes_form_computes_on_the_gpu(shape, order, precision, target,
                                            cache_root, tensorforge_include):
    """The lanes form built and run on the local GPU against NumPy, over a
    batch whose last group of 32 is partial and which gives a block of the
    persistent grid several trips; the operators the description's
    numbers, which its literals are.  F32 is held to a tolerance of the
    outputs' size, about 1e3: a sum of a hundred terms of that size rounds
    to about 1e-3 in single precision, and an output near zero meets it
    absolutely."""
    if target.backend != 'cuda':
        pytest.skip('the multi-element layout is CUDA\'s')
    from harness.runner import run_case
    group = eg.plan(_descrs(shape, order, precision),
                    _context(target.arch, precision, **LANES))
    assert group is not None and group.lanes is not None
    case = types.SimpleNamespace(
        NAME=f'multi_element_lanes_{shape}_o{order}{precision}',
        DTYPE=_dtype(precision), BATCH=4099,
        TOL=(1e-12, 1e-12) if precision == 'd' else (1e-4, 1e-2),
        OPTIONS=dict(LANES),
        INPUT_TRANSFORM=_described(_descrs(shape, order, precision)),
        descr_list=lambda: _descrs(shape, order, precision),
        reference=_reference(_descrs(shape, order, precision)))
    result = run_case(case, target, cache_root, tensorforge_include)
    assert result.passed, (result.max_abs_err, result.max_rel_err,
                           result.detail)
