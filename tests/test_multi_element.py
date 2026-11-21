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


def _assign_description(order=6, precision='s'):
    """The volume's last direction alone, assigning all of `Q`: `T = I
    star(2)`, then `Q = kDivM(2) T` over `Q`'s box -- an operator step
    that stores its D where the volume's add theirs."""
    d = json.loads(json.dumps(_volume_description(order, precision)))
    box = next(t for t in d['tensors'] if t['name'] == 'Q')['storage']
    rows = [box['start'][0], box['start'][0] + box['sizes'][0]]
    mixing, operator = d['operations'][-2:]
    depth = operator['args'][0]['bbox'][1][1]
    operator['result']['bbox'] = [[rows[0], 0], [rows[1], 9]]
    operator['args'][0]['bbox'] = [[rows[0], 0], [rows[1], depth]]
    operator['linear']['add'] = False
    d['operations'] = [mixing, operator]
    return d


def _window_description(order=6, precision='s'):
    """The volume with its first direction's product cut to rows 1..41 of
    `Q`, where its operator has nonzeros further down: in the step's rows,
    outside the product's window, a B that must stay zero there."""
    d = json.loads(json.dumps(_volume_description(order, precision)))
    operator = d['operations'][1]
    for view in (operator['result'], operator['args'][0]):
        view['bbox'][1][0] = 41
    return d


#: The cells of an elastic star SeisSol's GPU memory layout stores
#: (`config/gpu/tensorforge.xml`, `star` sparse) -- 24 of 81, in their
#: storage order, the same for the three directions -- as a re-export of
#: o6 under it lists them (`~/tf-probe/mfma/export`, 11-19).
STAR_ENTRIES = [[6, 0], [7, 0], [8, 0], [6, 1], [7, 1], [8, 1], [6, 2],
                [7, 2], [8, 2], [6, 3], [7, 3], [7, 4], [8, 4], [6, 5],
                [8, 5], [0, 6], [3, 6], [5, 6], [1, 7], [3, 7], [4, 7],
                [2, 8], [4, 8], [5, 8]]


def _sparse_star_description(order=6, precision='s'):
    """The volume with its stars stored as a SeisSol GPU build stores them:
    `spp`, the cells of `STAR_ENTRIES` alone."""
    d = json.loads(json.dumps(_volume_description(order, precision)))
    for t in d['tensors']:
        if t['name'].startswith('star('):
            t['storage'] = {'shape': [9, 9], 'type': 'spp',
                            'entries': STAR_ENTRIES}
    return d


DESCRIPTIONS = {
    'volume': _volume_description, 'chain': _chain_description,
    'sparse-star': _sparse_star_description,
    'direction': _direction_description, 'columns': _columns_description,
    'reuse': lambda order, precision: _chain_description(order, precision,
                                                         reuse=True),
    'assign': _assign_description, 'window': _window_description,
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


#: The options that take every input of the volume's trip a trip ahead into
#: a second stage (`element_group.second_stages`).
TWO_STAGES = {'enable_multibuffer': True, 'move_distance': 4}


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


def _sparse_operator_description(order=4, precision='s'):
    """The sparse-star volume with an operator stored sparse as well."""
    d = _sparse_star_description(order, precision)
    for t in d['tensors']:
        if t['name'] == 'kDivM(0)':
            box = t['storage']['shape']
            t['storage'] = {'shape': box, 'type': 'spp',
                            'entries': [[0, 0], [1, 0]]}
    return d


def test_an_elements_small_operand_may_be_stored_sparse():
    """SeisSol's GPU memory layout stores `star` sparse, 24 of its 81
    cells (`STAR_ENTRIES`): the layout takes the volume so, the stars an
    element's small operands as before, and reads each stored cell where it
    is stored -- through a table of the cells' places, zero where a cell is
    not stored -- in either FMA form.  An operator stored sparse is still
    the layout's refusal."""
    from tensorforge.frontend.yateto import DescriptionReader
    section, why = eg.classify(_descrs('sparse-star', 4, 's'))
    assert section is not None, why
    stars = [b for b in section.inputs if b.tensor.alias.startswith('star')]
    assert len(stars) == 3 and all(b.role is eg.Role.SMALL for b in stars)
    assert all(not b.tensor.is_dense() and b.per_element == 81
               for b in stars)
    descrs = DescriptionReader(None, {}).read(
        _sparse_operator_description())[0]
    assert eg.classify(descrs)[1] == 'kDivM(0) is stored sparse'
    for options in ({'multi_element': 'force', 'tensor_cores': False},
                    LANES):
        gen = _generate(_descrs('sparse-star', 4, 's'),
                        _context('sm_90', 's', **options))
        kernel = gen.get_kernel()
        assert gen.built._element_group is not None
        tables = re.findall(r'(tf_mep_\w+)\[81\] = \{([^}]*)\}', kernel)
        assert len(tables) == 1, tables
        numbers = [int(x) for x in tables[0][1].split(',')]
        assert sorted(x for x in numbers if x >= 0) == list(range(24))
        assert numbers.count(-1) == 81 - 24
        assert f'{tables[0][0]}[' in kernel.split(tables[0][1])[-1]


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
    ('volume', 6, 'd', 'sm_80', 8, 'dmma'),
    ('volume', 6, 's', 'sm_120', 8, 'mma_sync'),
    ('volume', 6, 'd', 'sm_120', 4, 'dmma'),
    ('volume', 4, 's', 'sm_90', 32, 'mma_sync'),
    ('volume', 4, 'd', 'sm_120', 16, 'dmma'),
    ('volume', 8, 's', 'sm_90', 7, 'mma_sync'),
    ('volume', 8, 's', 'sm_80', 7, 'fma'),
    ('chain', 6, 's', 'sm_90', 16, 'mma_sync'),
    ('chain', 6, 's', 'sm_120', 8, 'mma_sync')])
def test_the_group_is_the_tables_halved_until_it_fits(shape, order,
                                                      precision, arch,
                                                      elements, lowering):
    """The design's size for the basis and the precision, halved while the
    operators and the group's buffers do not fit a block, the last size
    tried the smallest; the matrix tiling where a group of it fits, the FMA
    forms where none does (o8 F32 on sm_80's 164 kB)."""
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


def test_the_halving_tries_the_smallest_group_last(monkeypatch):
    """14 halves to 7, and 7 to `MIN_ELEMENTS` rather than to 3: o8 F32 on
    sm_90, the block a little short of a group of 7 in every form of the
    tiling."""
    section = eg.classify(_volume(8, 's'))[0]
    context = _context('sm_90', 's')
    need = min(eg.shared_bytes(eg.lower(section, context, split=halves), 7)
               for halves in (True, False))
    room = context.target.hw.max_local_mem_size_per_block
    monkeypatch.setattr(eg, 'SHARED_RESERVE', room - need + 1)
    group, why = eg.fill(section, context)
    assert group is not None, why
    assert group.elements == eg.MIN_ELEMENTS


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


@pytest.mark.parametrize('precision,depth,k,regs', [('s', 40, 8, 4),
                                                     ('d', 36, 4, 1)])
def test_a_product_reads_its_window_of_the_operator(precision, depth, k, regs):
    """Each operator product's B is its own window, the step's rows its
    columns from the step's first: the contraction padded to the
    instruction's depth, the rows to its width, in fragment order -- a
    lane's registers side by side, in F32 both TF32 halves of each where
    the group has the room; the operators as stored are not staged where
    nothing reads them."""
    section = eg.classify(_volume(6, precision))[0]
    group = eg.fill(section, _context('sm_90', precision))[0]
    stages = group.section.stages
    assert [(st.tensor.alias, st.window) for st in stages] == [
        (f'kDivM({d})', ((1, 54 + d), (0, 35))) for d in range(3)]
    assert {(st.n0, st.depth, st.width, st.k, st.regs) for st in stages} \
        == {(1, depth, 56, k, regs)}


@pytest.mark.parametrize('order,arch,elements,split', [
    (6, 'sm_90', 16, True), (6, 'sm_120', 8, False), (4, 'sm_120', 16, True)])
def test_b_comes_split_where_it_costs_no_group_size(order, arch, elements,
                                                    split):
    """The halves take twice B's room: the fill rule takes them at the
    largest group that fits either way, and only where they fit there."""
    section = eg.classify(_volume(order, 's'))[0]
    group = eg.fill(section, _context(arch))[0]
    assert group.elements == elements
    assert eg.split(group.section) is split


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


@pytest.mark.parametrize('backend,arch,taken', [
    ('cuda', 'sm_90', True), ('hip', 'gfx942', True), ('hip', 'gfx1150', True),
    ('oneapi', 'pvc', True), ('esimd', 'pvc', False), ('acpp', 'pvc', False)])
def test_the_layout_is_written_for_three_targets(backend, arch, taken):
    """CUDA on NVIDIA, HIP on AMD and DPC++ under SPMD on Intel
    (`TARGETS`); the explicit vector, and AdaptiveCpp, whose device picks the
    sub-group, are refused and say so."""
    section = eg.classify(_volume())[0]
    for options in ({}, LANES):
        group, why = eg.fill(section, _context(arch, backend=backend,
                                               **options))
        assert (group is not None) is taken
        if not taken:
            assert 'DPC++ under SPMD on Intel' in why


@pytest.mark.parametrize('backend,arch', [('hip', 'gfx942'),
                                          ('oneapi', 'pvc')])
def test_the_matrix_tiling_is_nvidias(backend, arch):
    """Elsewhere every step takes its FMA form, the matrix engines asked
    for or not."""
    section = eg.classify(_volume())[0]
    lowered = eg.lower(section, _context(arch, backend=backend,
                                         tensor_cores=True))
    assert not eg.tiled(lowered)
    assert {s.lowering for s in lowered.steps} == {'fma', 'rows'}


@pytest.mark.parametrize('backend,arch,options,elements,registers,grf', [
    ('cuda', 'sm_90', {}, 32, 168, None),
    ('hip', 'gfx942', {}, 64, 168, None),
    ('hip', 'gfx1150', {}, 32, 168, None),
    ('oneapi', 'pvc', {}, 16, 168, '256'),
    ('oneapi', 'pvc', {'sycl_grf': 'auto'}, 16, 168, 'auto'),
    ('oneapi', 'pvc', {'sycl_grf': '128'}, 16, 84, '128')],
    ids=['sm_90', 'gfx942', 'gfx1150', 'pvc', 'pvc-auto', 'pvc-grf128'])
def test_the_lanes_group_is_the_targets_warp(backend, arch, options,
                                             elements, registers, grf):
    """The lanes form's group is a warp's lanes on the target -- a CUDA
    warp, a CDNA or RDNA wave, a sub-group of Xe-HPC's 16 lanes -- and a
    lane holds as much of what it addresses as on CUDA: 168 of 256 VGPRs on
    AMD, and on Intel its share of the thread's file, the large one unless
    `sycl_grf` asks for another."""
    context = _context(arch, backend=backend, **options)
    assert eg.lane_elements(context) == elements
    assert eg.lane_registers(context, elements) == registers
    assert eg.lane_grf(context) == grf
    for order in (4, 6):
        section = eg.classify(_volume(order, 'd'))[0]
        group = eg.fill(section, _context(arch, 'd', backend=backend,
                                          **options, **LANES))[0]
        if group.lanes is None:
            # F64 in the small file: the warp form's.
            assert registers == 84
            why = eg.lanes_plan(section, _context(arch, 'd', backend=backend,
                                                  **options))[1]
            assert f'more than the {registers} the lanes form' in why
            continue
        assert group.elements == elements and group.threads == elements
        assert group.warps <= eg.LANE_WARPS
        assert group.lanes.registers <= registers


@pytest.mark.parametrize('backend,arch,options,values', [
    ('cuda', 'sm_90', {}, 'literal'), ('hip', 'gfx942', {}, 'memory'),
    ('oneapi', 'pvc', {}, 'table'),
    ('cuda', 'sm_90', {'multi_element_values': 'memory'}, 'memory'),
    ('cuda', 'sm_90', {'multi_element_values': 'table'}, 'table'),
    ('hip', 'gfx942', {'multi_element_values': 'literal'}, 'literal')],
    ids=['sm_90', 'gfx942', 'pvc', 'sm_90-memory', 'sm_90-table',
         'gfx942-literal'])
def test_where_the_lanes_forms_numbers_come_from(backend, arch, options,
                                                 values):
    """Literals on NVIDIA, the operator's memory on AMD, a table of the
    numbers in the order they are multiplied by on Intel
    (`multi_element_values`): as literals the description's numbers are in
    the kernel's text; read from memory the kernel reads the operators'
    pointers; from the table it reads the table, a warp's width at a time,
    and neither of the others.  Only literals count against
    `inline_constants`."""
    context = _context(arch, backend=backend, **options)
    assert eg.lane_values(context) == values
    assert eg.lane_literals(context) is (values == 'literal')
    section = eg.classify(_volume(4, 's'))[0]
    lanes, why = eg.lanes_plan(section, _context(
        arch, backend=backend, inline_constants=8, **options))
    assert (lanes is None) is (values == 'literal')
    gen = _generate(_volume(4, 's'), _context(arch, backend=backend,
                                              inline_constants=4096,
                                              **options, **LANES))
    kernel = gen.get_kernel()
    names = re.findall(r'//   (m\d+) \S+ \S+ none', kernel)
    assert len(names) == 3
    # The code, without the tables the file defines ahead of it.
    code = [line for line in kernel.splitlines()
            if not line.strip().startswith(('//', '#', 'constexpr',
                                            '__device__ const'))]
    reads = [n for n in names
             if any(re.search(rf'\bglb_{n}\[', line) for line in code)]
    assert reads == (names if values == 'memory' else [])
    operator = section.steps[-1].products[0].first.tensor
    value = next(v for v in operator.storage_values() if abs(v) > 1.5)
    assert (Datatype.F32.literal(value) in '\n'.join(code)) is (
        values == 'literal')
    tables = [d for d in gen.module_table_definitions() if 'tf_mel_' in d]
    assert bool(tables) is (values == 'table')
    if values == 'table':
        name = re.search(r'(tf_mel_\w+)\[', tables[0]).group(1)
        assert any(name in line for line in code)
        assert Datatype.F32.literal(value) in tables[0]


@pytest.mark.parametrize('fission', ['auto', 2])
def test_a_kernel_the_layout_takes_is_not_cut(fission):
    """Fission (`Options.fission`, `auto` on Intel under SPMD) cuts a list
    into runs of the batch, each a section of its own: the layout plans
    the kernel's one section whole, and is not cut."""
    for options in (LANES, {'multi_element': 'force'}):
        gen = _generate(_descrs('localFluxAll', 4, 'd'), _context(
            'pvc', 'd', backend='oneapi', fission=fission, **options))
        built = gen.built
        assert built._element_group is not None
        assert len(built._sections) == 1


@pytest.mark.parametrize('backend,arch,spelled', [
    ('cuda', 'sm_90', None), ('hip', 'gfx942', 'tensorforge::opaqueZero()'),
    ('oneapi', 'pvc', 'sycl::group_broadcast(item.get_sub_group(), ')])
def test_the_lanes_form_anchors_its_reads_to_the_column(backend, arch,
                                                        spelled):
    """Off NVIDIA the column's reads of what does not change from column
    to column -- the element's rows its mixing products read -- carry an
    anchor of the column, a zero the compiler cannot see through, so that
    it forms their addresses in the column loop (`lane_anchors`)."""
    assert eg.lane_anchors(_context(arch, backend=backend)) is (
        spelled is not None)
    kernel = _generate(_volume(4, 's'), _context(
        arch, backend=backend, **LANES)).get_kernel()
    lines = kernel.splitlines()
    loop = next(i for i, line in enumerate(lines)
                if re.search(r'for \(int32_t v\d+_c = 0;', line))
    anchors = [i for i, line in enumerate(lines) if '_anc = ' in line]
    if spelled is None:
        assert not anchors
        return
    assert len(anchors) == 1 and anchors[0] > loop
    assert spelled in lines[anchors[0]]
    name = re.search(r'(v\d+_anc) = ', lines[anchors[0]]).group(1)
    # Read at the anchor, or at an index formed of it.
    anchored = {name} | set(re.findall(
        rf'(v\d+_a) = [^;]*\b{name}\b', kernel))
    reads = re.findall(r'glb_m0\[([^\]]+)\]', kernel)
    assert reads and all(set(re.findall(r'v\d+_an?c?\b', at)) & anchored
                         for at in reads)


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
        assert group.elements == eg.lane_elements(_context('sm_90')) == 32
        assert group.warps == 3 and group.shared_bytes == 0
        assert {s.lowering for s in group.section.steps} == {'lane'}
        assert group.section.stages == ()


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


@pytest.mark.parametrize('options,sub_group,grf', [
    (LANES, 16, '256'), ({**LANES, 'sycl_grf': '128'}, 16, '128'),
    ({'multi_element': 'force'}, 32, None)],
    ids=['lanes', 'lanes-grf128', 'warp'])
def test_a_sycl_kernel_states_its_sub_group_and_file(options, sub_group,
                                                     grf):
    """Under DPC++ SPMD the lanes form's kernel states a sub-group of its
    16 elements and the large register file, `sycl_grf` where it says
    another; the warp form's states its warp of 32 and no file."""
    kernel = _generate(_volume(4, 's'), _context(
        'pvc', backend='oneapi', **options)).get_kernel()
    assert f'[[intel::reqd_sub_group_size({sub_group})]]' in kernel
    files = re.findall(r'grf_size<(\d+)>', kernel)
    assert files == ([] if grf is None else [grf])


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
    ({}, 4), ({'tensor_cores': False}, 3)], ids=['matrix', 'fma'])
def test_the_group_takes_its_trips_together(options, barriers):
    """The traversal is the group's, a loop the IR sees: every warp's
    element from the group's first.  The block meets where the placement
    finds the staging and the steps need it (`pir.barriers`) -- the trip's
    head no more, its copies retired behind the trip's last reads -- and the
    matrix tiling adds into the outputs from its accumulators, behind no
    barrier of its own."""
    kernel = _kernel(multi_element='force', **options)
    assert re.search(r'batchIdGroup0 \+ (v\d+_)?batchIdLane0', kernel)
    assert kernel.count('__syncthreads();') == barriers


@pytest.mark.parametrize('shape,arch,precision,adds,stores', [
    ('volume', 'sm_90', 's', 4, 0), ('volume', 'sm_80', 'd', 2, 0),
    ('volume', 'sm_120', 'd', 4, 0), ('assign', 'sm_90', 's', 0, 4),
    ('assign', 'sm_80', 'd', 0, 2), ('chain', 'sm_90', 's', 0, 0)])
def test_a_matrix_step_writes_its_output_from_the_accumulators(
        shape, arch, precision, adds, stores):
    """A lane's cells of D go into the output from its registers, each
    through its row's element's pointer (`eQ_*`, read from the block's
    table `pQ`): one reduction a cell where the step accumulates, a store
    where it assigns, no buffer and no barrier between.  A section whose
    matrix steps write intermediates only keeps no table."""
    section = eg.classify(_descrs(shape, 6, precision))[0]
    group = eg.fill(section, _context(arch, precision))[0]
    tables = [t.alias for t in group.section.pointer_tables]
    assert tables == (['Q'] if adds or stores else [])
    lines = _kernel(arch=arch, precision=precision, shape=shape,
                    multi_element='force').splitlines()
    reductions = [line for line in lines if 'red.global.add.' in line]
    assert len(reductions) == len([r for r in reductions if '&eQ_' in r])
    assert len(reductions) == adds
    assert len(_stores(lines, r'eQ_\d+')) == adds + stores
    assert any('_pQ[' in line for line in lines) is bool(tables)


def _loop(lines):
    """The line the batch loop's group traversal starts on."""
    return next(i for i, line in enumerate(lines)
                if re.search(r'for \(size_t \w*batchIdGroup0 =', line))


def test_the_staging_is_copied_a_trip_ahead():
    """A trip's operands come by `cp.async`, and the first of them a trip
    ahead (`pir.wrap`): the first group's copy ahead of the loop, each next
    group's behind the last read of the buffer, one stage."""
    lines = _kernel(multi_element='force').splitlines()
    loop = _loop(lines)
    peel = [i for i, line in enumerate(lines) if 'peel_glb_m0[' in line]
    ahead = [i for i, line in enumerate(lines) if 'wrap_glb_m0[' in line]
    assert peel and max(peel) < loop
    assert ahead and min(ahead) > loop
    assert all('__pipeline_memcpy_async(&' in lines[i] for i in peel + ahead)
    assert any('__pipeline_wait_prior' in line for line in lines[loop:])


@pytest.mark.parametrize('order,precision,arch,elements,doubled', [
    (6, 's', 'sm_90', 16, 4), (6, 'd', 'sm_90', 8, 4),
    (8, 's', 'sm_90', 7, 0), (6, 's', 'sm_120', 8, 0),
    (4, 's', 'sm_120', 16, 4)])
def test_a_second_stage_where_it_costs_no_group_size(order, precision, arch,
                                                     elements, doubled):
    """`enable_multibuffer` and a `move_distance` over the volume's four
    inputs: each takes a second stage where the group keeps its size,
    copied a trip ahead into the stage the trip does not read; where the
    stage would cost the group -- o8 F32 on sm_90, from 7 to 4 -- the
    group keeps its size and the inputs one stage, the trip's own barrier
    ahead of their copies keeping them in the trip."""
    section = eg.classify(_volume(order, precision))[0]
    context = _context(arch, precision, **TWO_STAGES)
    group = eg.fill(section, context)[0]
    assert (group.elements, group.section.doubled) == (elements, doubled)
    assert group.shared_bytes == eg.shared_bytes(group.section, elements)
    plain = eg.fill(section, _context(arch, precision))[0]
    assert plain.elements == elements
    lines = _generate(_volume(order, precision), _context(
        arch, precision, multi_element='force', **TWO_STAGES)
    ).get_kernel().splitlines()
    loop = _loop(lines)
    peel = {m for line in lines[:loop]
            for m in re.findall(r'peel_glb_(m\d+)\[', line)}
    assert len(peel) == doubled
    if not doubled:
        first = next(i for i, line in enumerate(lines)
                     if i > loop and '__pipeline_memcpy_async' in line)
        assert any(line.strip() == '__syncthreads();'
                   for line in lines[loop:first])


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


@pytest.mark.parametrize('shape,precision,pitched,count', [
    ('volume', 's', {'_tmp0': 44, '_tmp2': 44, '_tmp4': 44}, 15),
    ('volume', 'd', {}, 0),
    ('chain', 's', {'I': 44}, 10)])
def test_f32_a_fragments_come_by_ldmatrix(shape, precision, pitched, count):
    """In F32 a matrix step reads A by ldmatrix, a load of four 8x8 matrices
    a warp, so the buffers it reads have a pitch: rows of 16 bytes, four
    floats more than a multiple of eight, long enough for every padded step
    of K; F64 reads A lane by lane."""
    section = eg.classify(_descrs(shape, 6, precision))[0]
    group = eg.fill(section, _context('sm_90', precision))[0]
    got = {b.tensor.alias: b.pitch for b in group.section.inputs
           + group.section.intermediates if b.stride}
    assert got == pitched
    assert all(pitch % 8 == 4 for pitch in got.values())
    kernel = _kernel(precision=precision, shape=shape,
                     multi_element='force')
    assert kernel.count('tensorforge::ldmatrixX4(') == count


def test_the_padding_is_zero_before_the_tiling_reads_it():
    """The pitch's padding of every column is written with zeros in each
    trip, ahead of the barrier the matrix step waits behind -- what its
    padded steps of K multiply the operator's zero rows with."""
    kernel = _kernel(multi_element='force', enable_sync_block_opt=False)
    lines = kernel.splitlines()
    barriers = [i for i, line in enumerate(lines)
                if line.strip() == '__syncthreads();']
    first_read = min(i for i, line in enumerate(lines)
                     if 'tensorforge::ldmatrixX4(' in line)
    ahead = max(b for b in barriers if b < first_read)
    for d in (0, 2, 4):
        zeros = _stores(lines, rf'_g_tmp{d}', zeros=True)
        assert zeros and max(zeros) < ahead, d


def test_the_barrier_placement_sees_the_ldmatrix_reads():
    """With the barrier optimization (`pir.barriers`) the block still meets
    between the intermediates' last write and ldmatrix's first read of
    them."""
    lines = _kernel(multi_element='force').splitlines()
    writes = _stores(lines, r'_g_tmp\d')
    first_read = min(i for i, line in enumerate(lines)
                     if 'tensorforge::ldmatrixX4(' in line)
    assert any(max(writes) < i < first_read for i, line in enumerate(lines)
               if line.strip() == '__syncthreads();')


def test_ldmatrix_is_a_declared_read_of_its_buffer():
    """Every pass orders ldmatrix by what it says it reads: each
    `ldmatrixX4` is a read of the buffer its lanes name, in shared memory,
    and no pure function a pass may move or merge."""
    from tensorforge.backend.pir.core import Effect, MemSpace, Op, walk_stmts
    gen = _generate(_volume(6, 's'), _context(multi_element='force'))
    loads = [st for section in gen.built._sections
             for st in walk_stmts(section.body)
             if st.op == Op.SPLIT
             and st.attr('callee') == 'tensorforge::ldmatrixX4']
    assert len(loads) == 15
    for st in loads:
        assert not st.pure and not st.movable
        assert [(a.kind, a.space) for a in st.accesses] == [
            (Effect.READ, MemSpace.SHARED)]


def _stores(lines, pattern, zeros=False):
    """The lines that store into the buffers `pattern` names -- their data,
    or (`zeros`) the zeros of their padding -- and that add data into them
    by a reduction (`red.global.add`) or copy it in (`cp.async`)."""
    zero = re.compile(f'(?:{pattern})' + r'\[[^]]*\] = 0\.0f?;')
    store = re.compile(f'(?:{pattern})' + r'\[[^]]*\] = '
                       + r'|red\.global\.add\.\w+ .*&' + f'(?:{pattern})\\['
                       + r'|__pipeline_memcpy_async\(&\w*' + f'(?:{pattern})\\[')
    return [i for i, line in enumerate(lines)
            if store.search(line) and bool(zero.search(line)) == zeros]


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
#: A matrix step's adds into `Q` from its accumulators, through its rows'
#: elements' pointers.
ADDED = r'eQ_\d+'


@pytest.mark.parametrize('shape,options,phases', [
    ('volume', {}, [STAGED + r'|_pQ', r'_g_tmp\d', ADDED]),
    ('volume', {'tensor_cores': False}, [STAGED, r'_g_tmp\d', OUTPUT]),
    ('chain', {}, [STAGED, r'_g_chain\d', OUTPUT])],
    ids=['volume-matrix', 'volume-fma', 'chain-matrix'])
def test_the_barriers_stand_without_the_placement(shape, options, phases):
    """A build without the barrier optimization (`Options.
    enable_sync_block_opt`) keeps the instructions' own block barriers: one
    after the operators' copy, ahead of the loop, and in the trip one
    ahead of each phase whose stores another warp may still be reading --
    the staging with the pointer tables, the intermediates, the outputs."""
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
    ('volume', {}), ('chain', {}), ('volume', LANES),
    ('volume', {'arch': 'sm_80', 'precision': 'd'})],
    ids=['volume', 'chain', 'volume-lanes', 'volume-sm80-d'])
def test_an_element_past_the_end_writes_nothing(shape, options):
    """Every write of `Q` runs under the element's mask: the warp's, or in
    the lanes form the lane's -- and where a matrix step adds a cell of any
    warp's element from its accumulators, the element's pointer being
    there, which its warp publishes null where the element is masked."""
    lines = _kernel(shape=shape, **{'multi_element': 'force',
                                    **options}).splitlines()
    text = '\n'.join(lines)
    writes = _stores(lines, OUTPUT)
    adds = _stores(lines, ADDED)
    assert writes or adds
    assert all(lines[i - 1].strip() == 'if (batchIdActive0) {'
               for i in writes)
    published = re.findall(r'_pQ\[\w+\] = (.*);', text)
    assert bool(published) is bool(adds)
    # The word stored, where the scheduler named it ahead of the store.
    published = [re.search(r'int64_t ' + p + r' = (.*);', text).group(1)
                 if re.fullmatch(r'\w+', p) else p for p in published]
    assert all(re.fullmatch(r'\(?batchIdActive0 \? \(reinterpret_cast<int64_t>'
                            r'\(glb_m\d+\)\) : 0_i64\)?', p)
               for p in published)
    for i in adds:
        pointer = re.search(r'&(eQ_\d+)\[', lines[i]).group(1)
        word = re.search(pointer + r' = reinterpret_cast<\w+\*>\((\w+)\);',
                         text).group(1)
        assert re.search(r'int64_t ' + word + r' = \w+_pQ\[', text)
        held = re.search(r'bool (\w+) = ' + word + ' != 0;', text).group(1)
        assert lines[i - 1].strip().startswith('if (')
        assert re.search(r'\b' + held + r'\b', lines[i - 1])


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


@pytest.mark.parametrize('shape,order,precision', [
    ('volume', 4, 's'), ('localFlux', 4, 's'), ('columns', 4, 's')])
def test_a_pass_of_several_columns(shape, order, precision, monkeypatch):
    """Where the registers allow it (`LANE_COLUMNS`), a pass takes a class's
    columns together (`Lanes.columns`): each of the operators' numbers once
    for all of them, the same arithmetic in fewer passes -- at the same row
    ranges the count the same, the passes a third.  (Where the columns take
    more ranges, each range computes its mixing entries again.)"""
    section = eg.classify(_descrs(shape, order, precision))[0]
    one = eg.lanes_plan(section, _context('sm_90', precision))[0]
    flops = _executed(shape, order, precision, multi_element_fma='lanes')
    monkeypatch.setitem(eg.LANE_COLUMNS, 'nvidia', 3)
    monkeypatch.setattr(eg, 'LANE_REGISTERS', 255)
    three = eg.lanes_plan(section, _context('sm_90', precision))[0]
    assert one.columns == 1 and three.columns == 3
    assert three.ranges == one.ranges and three.classes == 3
    assert _executed(shape, order, precision,
                     multi_element_fma='lanes') == flops
    kernel = _generate(_descrs(shape, order, precision), _context(
        'sm_90', precision, **LANES)).get_kernel()
    assert re.search(r'for \(int32_t v\d+_c = 0; v\d+_c < 1;', kernel)


@pytest.mark.parametrize('backend,arch,spelled', [
    ('cuda', 'sm_90', None), ('hip', 'gfx942', 'fma('),
    ('oneapi', 'pvc', 'sycl::fma(')])
def test_the_lanes_forms_fmas_stand_where_they_are(backend, arch, spelled):
    """Where the target spells an FMA as a call (`Lexic.fused_fma`), the
    lanes form's FMAs are such calls, each a statement of its own: icpx may
    reassociate an infix chain, and a chain inlined into one expression at
    its last use moves the arithmetic behind every read of the operators'
    numbers, which IGC then held and spilled.  The warp form's text, and
    CUDA's, keep the infix form."""
    lanes = _generate(_volume(4, 's'), _context(
        arch, backend=backend, **LANES)).get_kernel()
    warp = _generate(_volume(4, 's'), _context(
        arch, backend=backend, multi_element='force')).get_kernel()
    if spelled is None:
        assert 'fma(' not in lanes
        return
    calls = re.findall(r'^\s*float v\d+_[xt] = ' + re.escape(spelled), lanes,
                       re.M)
    assert calls
    assert not re.search(r'= ' + re.escape(spelled) + r'[^;]*' +
                         re.escape(spelled), lanes)
    assert spelled not in warp


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


def _oracle_errors(shape, precision, arch, options):
    """The kernel run by the host interpreter against NumPy: each output's
    error, normwise."""
    host = _host()
    kernel = host.Kernel(DESCRIPTIONS[shape](6, precision), _dtype(precision),
                         arch)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        got, _ = kernel.run({'multi_element': 'force', **options})
    want = kernel.expected()
    errors = {}
    for alias in {alias for alias, _ in want}:
        keys = [key for key in want if key[0] == alias]
        scale = max(abs(want[key]) for key in keys) or 1.0
        errors[alias] = max(abs(got[key] - want[key]) for key in keys) / scale
    return errors


@pytest.mark.parametrize('shape,options', [
    ('volume', {}), ('chain', {}), ('reuse', {}),
    ('volume', {'tensor_cores': False}), ('assign', {}), ('window', {})],
    ids=['volume', 'chain', 'reuse', 'volume-fma', 'assign', 'window'])
@pytest.mark.parametrize('precision,tolerance', [('s', 1e-6), ('d', 1e-12)])
def test_the_section_computes_on_the_oracle(shape, options, precision,
                                            tolerance):
    """The kernel run by the host interpreter against NumPy, normwise --
    the section's real operators, on sm_90: the matrix tiling, and once the
    FMA forms."""
    for alias, error in _oracle_errors(shape, precision, 'sm_90',
                                       options).items():
        assert error <= tolerance, (alias, error)


@pytest.mark.parametrize('shape', ['volume', 'assign'])
def test_sm_80s_shape_computes_on_the_oracle(shape):
    """F64 on sm_80, `m8n8k4`: two cells of D a lane, each added into the
    output from the accumulators, or stored where the step assigns."""
    for alias, error in _oracle_errors(shape, 'd', 'sm_80', {}).items():
        assert error <= 1e-12, (alias, error)


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
    ('direction', 4, 'd', LANES),
    ('volume', 4, 's', {}), ('volume', 4, 'd', {}),
    ('volume', 4, 's', TWO_STAGES), ('chain', 4, 's', TWO_STAGES),
    ('volume', 4, 'd', {'tensor_cores': False, 'arch': 'gfx942'}),
    ('localFlux', 4, 's', {'arch': 'gfx1150'}),
    ('volume', 4, 'd', {**LANES, 'arch': 'gfx942'}),
    ('localFluxAll', 4, 's', {**LANES, 'arch': 'gfx942'}),
    ('localFlux', 4, 'd', {**LANES, 'arch': 'gfx1150'}),
    ('volume', 4, 's', {**LANES, 'multi_element_values': 'memory'}),
    ('volume', 4, 's', {**LANES, 'multi_element_values': 'table'}),
    ('localFlux', 4, 'd', {**LANES, 'arch': 'gfx942',
                           'multi_element_values': 'table'}),
    ('volume', 4, 'd', {**LANES, 'columns': 3}),
    ('localFluxAll', 4, 's', {**LANES, 'columns': 3, 'arch': 'gfx942'}),
    ('sparse-star', 4, 'd', {'tensor_cores': False}),
    ('sparse-star', 4, 's', {'arch': 'gfx942'}),
    ('sparse-star', 4, 's', LANES),
    ('sparse-star', 4, 'd', {**LANES, 'arch': 'gfx942'})],
    ids=['volume-o4d', 'volume-o4s-dense', 'reuse-o4s', 'localFlux-o4d',
         'neighboringFlux-o4s-own-barriers', 'localFluxAll-o4d',
         'direction-o4s', 'columns-o4d', 'lanes-volume-o4d',
         'lanes-volume-o4s', 'lanes-localFlux-o4s',
         'lanes-neighboringFlux-o4d', 'lanes-localFluxAll-o4s',
         'lanes-columns-o4s', 'lanes-direction-o4d', 'matrix-volume-o4s',
         'matrix-volume-o4d', 'matrix-volume-o4s-two-stages',
         'matrix-chain-o4s-two-stages', 'gfx942-volume-o4d',
         'gfx1150-localFlux-o4s', 'gfx942-lanes-volume-o4d',
         'gfx942-lanes-localFluxAll-o4s', 'gfx1150-lanes-localFlux-o4d',
         'lanes-volume-o4s-memory', 'lanes-volume-o4s-table',
         'gfx942-lanes-localFlux-o4d-table', 'lanes-volume-o4d-columns',
         'gfx942-lanes-localFluxAll-o4s-columns', 'sparse-star-o4d',
         'gfx942-sparse-star-o4s', 'lanes-sparse-star-o4s',
         'gfx942-lanes-sparse-star-o4d'])
def test_every_element_of_two_trips_on_the_oracle(shape, order, precision,
                                                  options, monkeypatch):
    """A group over itself and seven more elements on the host oracle: two
    trips of the group, the second partial -- four warps over seven
    elements, one warp masked, in the warp form; a warp's lanes over seven
    more, the rest of the second trip masked, in the lanes form -- on CUDA
    and on HIP, a CDNA wave of 64 lanes and RDNA's of 32, the operators'
    nonzeros literals or read from memory -- every element's output against
    NumPy, from the oracle's own fill of memory, the operators their
    description's numbers; and no access to shared memory that no barrier
    orders (`kernel_eval.Races`)."""
    import numpy as np
    from tensorforge.common.basic_types import Addressing
    from tensorforge.reference import kernel_eval
    options = dict(options)
    arch = options.pop('arch', 'sm_90')
    columns = options.pop('columns', None)
    if columns is not None:
        for vendor in ('nvidia', 'amd'):
            monkeypatch.setitem(eg.LANE_COLUMNS, vendor, columns)
        monkeypatch.setattr(eg, 'LANE_REGISTERS', 255)
    backend = 'cuda' if arch.startswith('sm_') else 'hip'
    context = _context(arch, precision, backend=backend,
                       **{'multi_element': 'force', **options})
    lanes = options.get('multi_element_fma') == 'lanes'
    if lanes:
        group = eg.lane_elements(context)
        count = group + 7
    else:
        group, count = 4, 7
        monkeypatch.setattr(eg, 'ELEMENTS',
                            {key: group for key in eg.ELEMENTS})
        monkeypatch.setattr(eg, 'DEFAULT_ELEMENTS', group)
    gen = _generate(_descrs(shape, order, precision), context)
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
        gives its storage slots (column by column over the box -- or, for a
        tensor stored sparse, its stored cells' slots, zero elsewhere)."""
        (r0, r1), (c0, c1) = box(t)
        out = np.zeros((count, r1 - r0, c1 - c0))
        for e in range(count):
            for c in range(c1 - c0):
                for r in range(r1 - r0):
                    if t.is_dense():
                        out[e, r, c] = read(e, r + c * (r1 - r0))
                    elif t.spp.is_nz((r0 + r, c0 + c)):
                        out[e, r, c] = read(e, t.linear_index((r0 + r,
                                                               c0 + c)))
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
    out.update(_stored_only(descrs))
    return out


def _stored_only(descrs):
    """The harness's fill of each element's operand stored sparse, zero at
    the cells it does not store: the harness hands the kernel the stored
    cells alone (`DriverOperand.pack_index`) and the reference the box, which
    has to be the same matrix."""
    import numpy as np
    out = {}
    for descr in descrs:
        for op in descr.operations():
            for sub in op.ops:
                t = getattr(sub, 'tensor', None)
                if t is None or t.is_tmp or t.is_dense():
                    continue
                (r0, c0), (r1, c1) = ([int(x) for x in t.bbox.lower()],
                                      [int(x) for x in t.bbox.upper()])
                mask = np.array([[t.spp.is_nz((r, c)) for c in range(c0, c1)]
                                 for r in range(r0, r1)])
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
    ('columns', 6, 's', {'tensor_cores': False}),
    ('volume', 4, 's', TWO_STAGES), ('chain', 6, 's', TWO_STAGES),
    ('sparse-star', 6, 'd', {'tensor_cores': False})]


@pytest.mark.parametrize('shape,order,precision,options', GPU_CASES, ids=[
    'o4s', 'o6s', 'o4d', 'o6d', 'o6s-own-barriers', 'o6s-fma',
    'o6s-fma-dense', 'o4d-fma', 'o6d-fma-own-barriers', 'chain-o6s',
    'chain-o6d', 'reuse-o6s', 'chain-o6d-fma', 'reuse-o6s-fma',
    'localFlux-o6d-fma', 'neighboringFlux-o4s-fma', 'localFluxAll-o6s-fma',
    'direction-o6d-fma', 'columns-o6s-fma', 'o4s-two-stages',
    'chain-o6s-two-stages', 'sparse-star-o6d-fma'])
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
    from harness.runner import run_case
    # The layout's, and not the default build a refused plan leaves -- on
    # CUDA every case; elsewhere a group the block's shared memory holds.
    context = _context(target.arch, precision, backend=target.backend,
                       multi_element='force', **options)
    if eg.plan(_descrs(shape, order, precision), context) is None:
        section = eg.classify(_descrs(shape, order, precision))[0]
        why = eg.fill(section, context)[1]
        assert target.backend != 'cuda', why
        pytest.skip(f'{target.backend} {target.arch}: {why}')
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
    out.update(_stored_only(descrs))
    return out


LANE_CASES = [
    ('volume', 4, 's'), ('volume', 6, 's'), ('volume', 4, 'd'),
    ('volume', 6, 'd'), ('localFlux', 6, 'd'), ('neighboringFlux', 4, 's'),
    ('localFluxAll', 6, 's'), ('localFluxAll', 6, 'd'), ('direction', 6, 'd'),
    ('columns', 6, 's'), ('sparse-star', 6, 's')]


@pytest.mark.parametrize('shape,order,precision', LANE_CASES,
                         ids=[f'{s}-o{o}{p}' for s, o, p in LANE_CASES])
def test_the_lanes_form_computes_on_the_gpu(shape, order, precision, target,
                                            cache_root, tensorforge_include):
    """The lanes form built and run on the local GPU against NumPy, over a
    batch whose last group is partial and which gives a block of the
    persistent grid several trips; the operators the description's
    numbers, which its literals are -- or which it reads where it reads
    them (`multi_element_values`).  F32 is held to a tolerance of the
    outputs' size, about 1e3: a sum of a hundred terms of that size rounds
    to about 1e-3 in single precision, and an output near zero meets it
    absolutely."""
    from harness.runner import run_case
    if (target.backend, target.vendor) not in eg.TARGETS:
        pytest.skip(f'the layout is not written for {target.backend} on '
                    f'{target.vendor}')
    group = eg.plan(_descrs(shape, order, precision),
                    _context(target.arch, precision, backend=target.backend,
                             **LANES))
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
