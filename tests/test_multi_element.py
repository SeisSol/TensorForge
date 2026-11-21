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
import dataclasses
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


def _assign_description(order=6, precision='s', first=None):
    """The volume's last direction alone, assigning all of `Q`: `T = I
    star(2)`, then `Q = kDivM(2) T` over `Q`'s box -- an operator step
    that stores its D where the volume's add theirs; `Q` stored from row
    `first` to its last basis function where it is given, so that a
    column's last row is the next one's first stored cell but one."""
    d = json.loads(json.dumps(_volume_description(order, precision)))
    box = next(t for t in d['tensors'] if t['name'] == 'Q')['storage']
    rows = [box['start'][0], box['start'][0] + box['sizes'][0]]
    if first is not None:
        rows = [first, order * (order + 1) * (order + 2) // 6]
        box['start'][0], box['sizes'][0] = first, rows[1] - first
    mixing, operator = d['operations'][-2:]
    depth = operator['args'][0]['bbox'][1][1]
    operator['result']['bbox'] = [[rows[0], 0], [rows[1], 9]]
    operator['args'][0]['bbox'] = [[rows[0], 0], [rows[1], depth]]
    operator['linear']['add'] = False
    d['operations'] = [mixing, operator]
    return d


def _window_description(order=6, precision='s'):
    """The volume with its first direction's product cut to rows 1..39 of
    `Q`, where its operator has nonzeros further down: in the step's rows,
    outside the product's window, a B that must stay zero there -- inside
    a tile the stage keeps (rows 33..40 on the grid from row 1), so that the
    tile's rows past the window are stored and multiplied."""
    d = json.loads(json.dumps(_volume_description(order, precision)))
    operator = d['operations'][1]
    for view in (operator['result'], operator['args'][0]):
        view['bbox'][1][0] = 39
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
    'assign-low': lambda order, precision: _assign_description(
        order, precision, first=1),
    # The flux kernels the layout takes besides the volume: a mixing
    # product, then two operator products, the first into an intermediate.
    'localFlux': _seissol('gpu_localFlux[0]'),
    'neighboringFlux': _seissol('gpu_neighboringFlux[0]'),
    'localFluxAll': _seissol('gpu_localFluxAll'),
    # A chained section: the time derivative, an order a phase, each reading
    # the order before back, and the time integral its scaled sum.
    'derivative': _seissol('gpu_derivative'),
    # Another: a product into the output, then a copy added to it.
    'plConvertToNodal': _seissol('gpu_plConvertToNodal'),
    # An operator product of the element's own operand, read from its memory
    # (`_LaneTrip.entry`), the output's rows in three ranges.
    'plConvertToModal': _seissol('gpu_plConvertToModal')}


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
    ('volume', 6, 'd', 'sm_90', 16, 'dmma'),
    ('volume', 6, 'd', 'sm_80', 8, 'dmma'),
    ('volume', 6, 's', 'sm_120', 8, 'mma_sync'),
    ('volume', 6, 'd', 'sm_120', 4, 'dmma'),
    ('volume', 4, 's', 'sm_90', 32, 'mma_sync'),
    ('volume', 4, 'd', 'sm_120', 16, 'dmma'),
    ('volume', 8, 's', 'sm_90', 7, 'mma_sync'),
    ('volume', 8, 's', 'sm_80', 4, 'mma_sync'),
    ('chain', 6, 's', 'sm_90', 16, 'mma_sync'),
    ('chain', 6, 's', 'sm_120', 8, 'mma_sync')])
def test_the_group_is_the_tables_halved_until_it_fits(shape, order,
                                                      precision, arch,
                                                      elements, lowering):
    """The design's size for the basis and the precision, halved while the
    operators and the group's buffers do not fit a block, the last size
    tried the smallest; the matrix tiling where a group of it fits, even
    the smallest -- o8 F32 on sm_80's 164 kB, its B tiles of zeros left
    out (`Stage.tiles`)."""
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
    where it is not; the mixing steps stay on the FMA units either way --
    block-wide beside the tiling (`mix`), a warp an element beside the FMA
    form (`rows`)."""
    monkeypatch.setattr(eg, 'MATRIX', default)
    section = eg.classify(_volume())[0]
    group = eg.fill(section, _context(**options))[0]
    mixing = 'mix' if lowering == 'mma_sync' else 'rows'
    assert [s.lowering for s in group.section.steps] == [mixing] * 3 + [
        lowering]


@pytest.mark.parametrize('arch,precision,source,vector', [
    ('sm_120', 's', 36, 4), ('sm_90', 's', 36, 4), ('sm_90', 'd', 38, 2)])
def test_the_star_step_beside_the_tiling_is_block_wide(arch, precision,
                                                      source, vector):
    """The tiling's star step (`mix`, `multi_element._mix`): a thread a row
    of A -- an element's column, 72 rows for sm_120's eight elements at o6,
    three threads of the block's 256 a row, each every third group of
    rows -- its
    27 star numbers read once, the group operand 16 bytes of a column at a
    time and the three directions in one pass.  The group operand's pitch
    an odd number of 16 bytes; the consumer's buffers as the tiling has
    them, written 16 bytes at a time where their pitch allows (F32's 44),
    an element at a time where it does not (F64's 35, unpitched)."""
    context = _context(arch, precision, multi_element='force')
    group = eg.plan(_descrs('volume', 6, precision), context)
    section = group.section
    assert [s.lowering for s in section.steps][:3] == ['mix'] * 3
    assert section.buffer(section.steps[0].products[0].first.tensor).pitch \
        == source
    tiled = eg.lower(eg.classify(_descrs('volume', 6, precision))[0],
                     context)
    plain = dataclasses.replace(tiled, steps=tuple(
        dataclasses.replace(s, lowering='rows') if s.lowering == 'mix'
        else s for s in tiled.steps))
    assert [b.pitch for b in section.intermediates] == [
        b.pitch for b in eg._pitched(plain).intermediates]
    kernel = _generate(_descrs('volume', 6, precision), context).get_kernel()
    rows = group.elements * 9
    assert f'% {rows};' in kernel and f'/ {rows};' in kernel
    # The threads' guard where it stands, or ahead of the loop, where it
    # does not change from trip to trip.
    assert re.search(rf'< {32 * group.elements // rows}(\) {{|;)', kernel)
    vec = re.escape(f'tensorforge::VectorT<{_dtype(precision).ctype()}, '
                    f'{vector}>')
    gi = re.findall(rf'= \*\({vec}\*\)&v\d+_gI\[', kernel)
    assert len(gi) == 9
    stores = re.findall(rf'\*\({vec}\*\)&v\d+_g_tmp\d\[[^]]*\] = ', kernel)
    assert len(stores) == (3 if precision == 's' else 0)
    star = re.findall(r'= v\d+_gstar\d\[v\d+_a\];', kernel)
    assert len(star) == 27


@pytest.mark.parametrize('arch,backend,options', [
    ('sm_120', 'cuda', {'tensor_cores': False}), ('gfx942', 'hip', {}),
    ('pvc', 'oneapi', {})])
def test_the_star_step_without_the_tiling_is_a_warp_an_element(
        arch, backend, options):
    """Without the matrix tiling -- the FMA form on NVIDIA, the warp form
    elsewhere -- the mixing steps keep the `rows` form, and the group
    operand its box's rows."""
    group = eg.plan(_descrs('volume', 4, 's'), _context(
        arch, 's', backend=backend, multi_element='force', **options))
    if group is None:
        pytest.skip(f'{arch}: no group')
    section = group.section
    assert {s.lowering for s in section.steps if s.kind is eg.Kind.MIXING} \
        == {'rows'}
    source = section.buffer(section.steps[0].products[0].first.tensor)
    assert source.pitch == source.height


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
    (6, 'sm_90', 16, True), (6, 'sm_120', 8, True), (4, 'sm_120', 16, True)])
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


@pytest.mark.parametrize('backend,arch,precision,options,values', [
    ('cuda', 'sm_90', 's', {}, 'literal'),
    ('hip', 'gfx942', 's', {}, 'literal'),
    ('hip', 'gfx1150', 's', {}, 'literal'),
    ('hip', 'gfx942', 'd', {}, 'memory'),
    ('oneapi', 'pvc', 's', {}, 'table'),
    ('cuda', 'sm_90', 's', {'multi_element_values': 'memory'}, 'memory'),
    ('cuda', 'sm_90', 's', {'multi_element_values': 'table'}, 'table'),
    ('hip', 'gfx942', 'd', {'multi_element_values': 'literal'}, 'literal')],
    ids=['sm_90', 'gfx942', 'gfx1150', 'gfx942-f64', 'pvc', 'sm_90-memory',
         'sm_90-table', 'gfx942-f64-literal'])
def test_where_the_lanes_forms_numbers_come_from(backend, arch, precision,
                                                 options, values):
    """Literals on NVIDIA and on AMD at F32, the operator's memory on AMD
    at F64, a table of the numbers in the order they are multiplied by on
    Intel (`multi_element_values`): as literals the description's numbers
    are in the kernel's text; read from memory the kernel reads the
    operators' pointers; from the table it reads the table, a warp's width
    at a time, and neither of the others.  Only literals count against
    `inline_constants`, and not on AMD, where a literal is a dword of its
    FMA (`LANE_LITERALS`)."""
    context = _context(arch, precision, backend=backend, **options)
    assert eg.lane_values(context) == values
    assert eg.lane_literals(context) is (values == 'literal')
    capped = values == 'literal' and backend != 'hip'
    assert (eg.lane_literal_cap(context) is None) is (backend == 'hip')
    section = eg.classify(_volume(4, precision))[0]
    lanes, why = eg.lanes_plan(section, _context(
        arch, precision, backend=backend, inline_constants=8, **options))
    assert (lanes is None) is capped
    gen = _generate(_volume(4, precision), _context(
        arch, precision, backend=backend, inline_constants=4096, **options,
        **LANES))
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
    assert (_dtype(precision).literal(value) in '\n'.join(code)) is (
        values == 'literal')
    tables = [d for d in gen.module_table_definitions() if 'tf_mel_' in d]
    assert bool(tables) is (values == 'table')
    if values == 'table':
        name = re.search(r'(tf_mel_\w+)\[', tables[0]).group(1)
        assert any(name in line for line in code)
        assert _dtype(precision).literal(value) in tables[0]


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
def test_the_lanes_form_anchors_its_reads_to_the_walk(backend, arch,
                                                      spelled):
    """Off NVIDIA the column's reads of what does not change from column
    to column -- the element's rows its mixing products read -- carry an
    anchor, a zero the compiler cannot see through, so that it forms their
    addresses in the column loop (`lane_anchors`); and a walk's anchor is
    its own: the four faces of `localFluxAll` read the element's rows each
    at anchors of its own, not the first face's reads held through the
    other three."""
    assert eg.lane_anchors(_context(arch, backend=backend)) is (
        spelled is not None)
    for descrs, walks in ((_volume(4, 's'), 1),
                          (_descrs('localFluxAll', 4, 's'), 4)):
        kernel = _generate(descrs, _context(
            arch, backend=backend, **LANES)).get_kernel()
        lines = kernel.splitlines()
        loop = next(i for i, line in enumerate(lines)
                    if re.search(r'for \(int32_t v\d+_c = 0;', line))
        anchors = {re.search(r'(v\d+_anc) = ', line).group(1): i
                   for i, line in enumerate(lines) if '_anc = ' in line}
        if spelled is None:
            assert not anchors
            continue
        # The walk's anchors, and on AMD a held walk's tiles' after the
        # arithmetic before them (`chained_zero`).
        assert all(i > loop and (spelled in lines[i]
                                 or 'tensorforge::opaqueZeroAfter(' in lines[i])
                   for i in anchors.values())
        # Read at an anchor, or at an index formed of one.
        of = dict(re.findall(r'(v\d+_a) = [^;]*\b(v\d+_anc)\b', kernel))
        at = [set(re.findall(r'v\d+_anc\b', read))
              | {of[a] for a in re.findall(r'v\d+_a\b', read) if a in of}
              for read in re.findall(r'glb_m0\[([^\]]+)\]', kernel)]
        assert at and all(at)
        assert len(set().union(*at)) >= walks


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
def test_force_takes_the_layout_where_off_and_auto_do_not(value):
    """`off` never, and `auto` not on CUDA, where nothing has measured it
    to pay yet (`AUTO_LANES`)."""
    assert eg.plan(_volume(), _context(multi_element=value)) is None
    assert eg.plan(_volume(), _context(multi_element='force')) is not None


def test_the_option_is_off_by_default():
    from tensorforge.common.options import registry
    assert registry()['multi_element'].default == 'off'


# -- the lanes form ----------------------------------------------------------- #

LANES = {'multi_element': 'force', 'multi_element_fma': 'lanes'}


@pytest.mark.parametrize('shape,order,precision,classes,ranges', [
    ('volume', 4, 's', 3, ((1, 20),)), ('volume', 6, 's', 3, ((1, 56),)),
    ('volume', 6, 'd', 3, ((1, 35), (35, 56))),
    ('localFlux', 6, 'd', 3, ((0, 35), (35, 56))),
    ('localFluxAll', 4, 'd', 3, ((0, 20),))])
def test_the_lanes_form_cuts_the_output_until_a_lane_holds_it(
        shape, order, precision, classes, ranges):
    """As few row ranges as keep a lane's registers within the plan's
    budget -- one in F32, two at o6 F64, cut at the basis's degree five:
    the rows below need the volume's intermediate up to degree three only,
    the PVC prototype's two halves -- and three column classes where the
    block's warps allow (`LANE_WARPS`)."""
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


def test_the_column_classes_share_the_blocks_warps(monkeypatch):
    """Classes times ranges are at most the block's warps (`LANE_WARPS`):
    at four warps the o6 F64 volume's two row ranges take two column
    classes, not the three a range alone would."""
    monkeypatch.setattr(eg, 'LANE_WARPS', 4)
    section = eg.classify(_descrs('volume', 6, 'd'))[0]
    lanes, why = eg.lanes_plan(section, _context('sm_90', 'd'))
    assert lanes is not None, why
    assert (lanes.classes, len(lanes.ranges)) == (2, 2)
    assert lanes.warps <= eg.LANE_WARPS


@pytest.mark.parametrize('shape', ['volume', 'localFlux'])
def test_a_cut_costs_the_least_work_a_lane_holds(shape):
    """Of the cuts in two whose ranges a lane holds, the plan's has the
    least work, all of it and its costlier range's added (`Cones.ranges`),
    then the earliest: the o6 F64 volume's at row 35 -- the rows below
    need its intermediate's first 20 rows, the rows above all 35 -- where
    the cut of the closest two halves, at row 37, computes the 35 in both,
    and the cut of the least work, at row 21, leaves the rows above three
    times the rows below."""
    section = eg.classify(_descrs(shape, 6, 'd'))[0]
    cones = eg.Cones(section)
    limit = eg.LANE_REGISTERS // 2
    lo, hi = cones.rows
    held = {c: (cones.work(range(lo, c)), cones.work(range(c, hi)))
            for c in range(lo + 1, hi)
            if max(cones.live(range(lo, c), 2),
                   cones.live(range(c, hi), 2)) <= limit}
    best = min(held, key=lambda c: (sum(held[c]) + max(held[c]), c))
    assert cones.ranges(2, limit, 2) == ((lo, best), (best, hi))
    assert cones.ranges(1, limit, 2) is None
    if shape == 'volume':
        assert best == 35 and 37 not in held
        below, above = cones.need(range(lo, 35)), cones.need(range(35, hi))
        for p in section.steps[-1].products:
            source = id(cones.source(p))
            assert max(below[source]) < 20 and len(above[source]) == 35
        least = min(held, key=lambda c: sum(held[c]))
        assert least == 21 and sum(held[least]) < sum(held[35])
        assert held[21][1] > 3 * held[21][0]


def test_a_range_costs_its_cone_and_ends_where_a_lane_no_longer_holds_it():
    """The work of a range found a row at a time (`Cones.costs`) is its
    cone's (`work`) -- a cone is the union of its rows' -- and a range
    from a row is in the table up to the last a lane holds."""
    section = eg.classify(_volume(6, 'd'))[0]
    cones = eg.Cones(section)
    lo, hi = cones.rows
    every = cones.costs()
    assert len(every) == (hi - lo) * (hi - lo + 1) // 2
    for a, b in ((lo, hi), (lo, 21), (21, hi), (35, hi), (10, 20)):
        assert every[(a, b)] == cones.work(range(a, b))
    held = cones.costs(84, 2)
    for a in range(lo, hi):
        ends = [b for (start, b) in held if start == a]
        last = max(ends)
        assert ends == list(range(a + 1, last + 1))
        assert cones.live(range(a, last), 2) <= 84
        assert last == hi or cones.live(range(a, last + 1), 2) > 84


@pytest.mark.parametrize('shape,arch,backend,precision,held', [
    ('localFlux', 'gfx942', 'hip', 's', True),
    ('localFluxAll', 'gfx1150', 'hip', 's', True),
    ('volume', 'gfx942', 'hip', 's', False),
    ('localFlux', 'gfx942', 'hip', 'd', False),
    ('localFlux', 'sm_90', 'cuda', 's', False),
    ('localFlux', 'pvc', 'oneapi', 's', False)])
def test_the_lanes_form_holds_what_it_writes_on_amd(shape, arch, backend,
                                                   precision, held):
    """On AMD every value the lanes form writes under the lane's mask is
    handed through an empty `asm` ahead of its guard (`opaque_value`), where
    a walk holds a column an operator step computes -- the flux kernels'
    faces -- and the numbers are literals: the compiler otherwise sinks the
    value, and every FMA it takes, behind the guard and every load ahead of
    it.  Not the volume, whose sums read what is held anyway; not with
    numbers read from memory, whose loads the backend then gathers; not
    off AMD."""
    kernel = _generate(_descrs(shape, 4, precision), _context(
        arch, precision, backend=backend, **LANES)).get_kernel()
    stores = re.findall(r'if \(batchIdActive\d*\) \{\s*\n\s*\*?\(?[^=;]*'
                        r'glb_m\d+\[[^\]]*\] = (v\d+_\w+);', kernel)
    assert stores
    passed = re.findall(r'(v\d+_w) = tensorforge::opaque\((v\d+_\w+)\);',
                        kernel)
    if not held:
        assert not passed and 'tensorforge::opaque(' not in kernel
        return
    assert {name for name, _ in passed} == set(stores)


@pytest.mark.parametrize('shape,arch,backend,precision,chained', [
    ('localFlux', 'gfx942', 'hip', 's', True),
    ('localFluxAll', 'gfx1150', 'hip', 'd', True),
    ('volume', 'gfx942', 'hip', 's', False),
    ('localFlux', 'sm_90', 'cuda', 's', False),
    ('localFlux', 'pvc', 'oneapi', 's', False)])
def test_the_lanes_form_reads_a_tile_after_the_one_before_on_amd(
        shape, arch, backend, precision, chained):
    """On AMD, where a walk holds a column an operator step computes, each
    tile of the element's rows after a walk's first is read at an anchor
    formed after the arithmetic before it (`chained_zero`): instruction
    selection puts a block's reads ahead of its FMAs otherwise.  The
    anchor reads the latest product of the walk; the reads of the tile go
    through it.  Not the volume, whose sums are not held; not off AMD."""
    kernel = _generate(_descrs(shape, 4, precision), _context(
        arch, precision, backend=backend, **LANES)).get_kernel()
    lines = kernel.splitlines()
    after = [(i, m.group(1), m.group(2)) for i, line in enumerate(lines)
             for m in [re.search(r'(v\d+_anc) = tensorforge::opaqueZeroAfter'
                                 r'\((v\d+_\w+)\);', line)] if m]
    if not chained:
        assert not after and 'opaqueZeroAfter' not in kernel
        return
    assert after
    for i, name, value in after:
        # Formed after a product the walk has made, ...
        made = next(j for j, line in enumerate(lines)
                    if re.search(rf'\b{value} = ', line))
        assert made < i and value.endswith('_x')
        # ... and read through by the element's rows that follow it.
        reads = [line for line in lines[i + 1:i + 40] if 'glb_m0[' in line]
        assert reads and any(name in line or re.search(r'v\d+_a\]', line)
                             for line in reads)


@pytest.mark.parametrize('arch,backend,precision,form,off', [
    ('gfx942', 'hip', 's', LANES, True),
    ('gfx90a', 'hip', 's', LANES, True),
    ('gfx942', 'hip', 'd', LANES, False),
    ('gfx1150', 'hip', 's', LANES, False),
    ('gfx942', 'hip', 's', {'multi_element': 'force'}, False),
    ('gfx942', 'hip', 's', {}, False),
    ('sm_90', 'cuda', 's', LANES, False)])
def test_the_lanes_form_is_compiled_without_packed_fp32_on_cdna(
        arch, backend, precision, form, off):
    """The lanes form's kernel at F32 on a target with packed FP32
    arithmetic (CDNA2 and later) is compiled without it
    (`lane_features`): the SLP vectorizer pairs its FMAs otherwise, and
    their numbers become registers gathered ahead of the pairs.  Not at
    F64, not on RDNA, not the warp form or the default kernel, not CUDA."""
    kernel = _generate(_volume(4, precision), _context(
        arch, precision, backend=backend, **form)).get_kernel()
    spelled = '__global__ void __attribute__((target("no-packed-fp32-ops")))'
    assert (spelled in kernel) is off
    assert kernel.count('__attribute__((target(') == int(off)


# -- a chained section: the derivative --------------------------------------- #

@pytest.mark.parametrize('order,precision', [(4, 's'), (6, 'd')])
def test_the_derivative_is_a_chained_section(order, precision):
    """SeisSol's time derivative classifies whole: the time integral and the
    zeroth derivative copies of `Q` (`Kind.SCALE`), then an order a phase --
    the volume's three mixing steps and one operator step into `dQ(d)`, of
    the order before read back -- each with the time integral's scaled sum
    of it after (`GroupSection.phases`); the factors are the kernel's
    scalars, an order's rows of its derivative a window of the buffer."""
    section, why = eg.classify(_descrs('derivative', order, precision))
    assert section is not None, why
    assert section.chained
    assert [t.alias for t in section.scalars] == [
        f'power({d})' for d in range(order)]
    phases = section.phases()
    assert len(phases) == order
    copies = phases[0]
    assert not copies.main and sorted(s.dest.alias for s in copies.scales) \
        == ['I', 'dQ(0)']
    assert all(s.products[0].first.tensor.alias == 'Q' for s in copies.scales)
    for d, phase in enumerate(phases[1:], 1):
        kinds = [s.kind for s in phase.main]
        assert kinds == [eg.Kind.MIXING] * 3 + [eg.Kind.OPERATOR]
        assert phase.main[-1].dest.alias == f'dQ({d})'
        read = 'Q' if d == 1 else f'dQ({d - 1})'
        assert all(s.products[0].first.tensor.alias == read
                   for s in phase.main[:3])
        (taylor,) = phase.scales
        p = taylor.products[0]
        assert (taylor.dest.alias, p.first.tensor.alias, p.add) == (
            'I', f'dQ({d})', True)
        assert p.second.tensor.alias == f'power({d})'
    assert not eg.classify(_volume())[0].chained


def test_a_copy_added_after_a_product_runs_after_it():
    """`Q = M P` then `Q += R` (SeisSol's `plConvertToNodal`): the copy adds
    to what the product wrote, so it runs in the product's phase, after it
    -- in the copies' own phase ahead of it, the product's assignment would
    overwrite what it added."""
    section, why = eg.classify(_descrs('plConvertToNodal', 4, 's'))
    assert section is not None and section.chained, why
    (phase,) = section.phases()
    assert [s.kind for s in phase.main] == [eg.Kind.OPERATOR]
    (copy,) = phase.scales
    assert copy.dest is phase.main[0].dest and copy.products[0].add


def _derivative_changed(change, order=4, precision='s'):
    """The derivative's description with one thing changed: `early`, the
    first mixing product of the second order read before the last product
    into the first; `unwritten`, the time integral's sum of the first order
    a row past what the order wrote; `window`, the time integral's first
    copy onto other rows than it copies."""
    d = json.loads(json.dumps(DESCRIPTIONS['derivative'](order, precision)))
    ops = d['operations']
    if change == 'early':
        reads = next(n for n, op in enumerate(ops)
                     if any(a['name'] == 'dQ(1)' for a in op['args'])
                     and op['result']['name'].startswith('_tmp'))
        ops.insert(reads - 3, ops.pop(reads))
    elif change == 'unwritten':
        op = next(op for op in ops if op['result']['name'] == 'I'
                  and op['args'][0]['name'] == 'dQ(1)')
        op['args'][0]['bbox'][1][0] += 1
    else:
        first = ops[0]
        first['result']['bbox'][0][0] += 1
    from tensorforge.frontend.yateto import DescriptionReader
    return DescriptionReader(None, {}).read(d)[0]


@pytest.mark.parametrize('change,why', [
    ('early', 'read before its last write'),
    ('unwritten', 'reads dQ(1) where nothing wrote it'),
    ('window', 'copies a window of Q onto another')])
def test_a_chained_section_reads_an_output_whole(change, why):
    """An output read back is read after its last write and where it was
    written; a scaled copy copies its window onto the same rows."""
    section, reason = eg.classify(_derivative_changed(change))
    assert section is None and why in reason


def test_a_window_assigned_outside_a_chain_is_refused():
    """Assigning a window of an element's tensor, the rest left as it is,
    only a chained section's phases do -- what reads the rest is a later
    phase's business there; the volume with its first product assigning a
    window of `Q` is refused, as it was."""
    d = json.loads(json.dumps(DESCRIPTIONS['volume'](4, 's')))
    op = d['operations'][1]
    op.setdefault('linear', {})['add'] = False
    from tensorforge.frontend.yateto import DescriptionReader
    section, reason = eg.classify(DescriptionReader(None, {}).read(d)[0])
    assert section is None and 'whose rest it would have to zero' in reason


def test_a_chained_section_is_the_lanes_forms_alone():
    """The warp form stages an element's operands ahead of its steps and
    writes an output's tiles whole: a chained section is the lanes form's;
    `auto` takes none, none measured yet."""
    section = eg.classify(_descrs('derivative', 4, 's'))[0]
    for arch, backend in (('sm_90', 'cuda'), ('pvc', 'oneapi')):
        group, why = eg.fill(section, _context(arch, 's', backend=backend,
                                               multi_element='force'))
        assert group is None and 'the lanes form\'s alone' in why
    taken, why = eg.auto_takes(section, _context('pvc', 's', backend='oneapi',
                                                multi_element='auto'))
    assert taken is None and 'chained' in why
    assert eg.plan(_descrs('derivative', 4, 's'), _context(
        'pvc', 's', backend='oneapi', multi_element='auto')) is None


@pytest.mark.parametrize('arch,backend,precision', [
    ('sm_90', 'cuda', 's'), ('pvc', 'oneapi', 's'), ('gfx942', 'hip', 'd')])
def test_the_lanes_form_takes_the_derivative_a_phase_a_warp(arch, backend,
                                                            precision):
    """One warp takes every phase -- every column of an order, at one range
    of its rows -- so that a lane reads back the order before as it wrote
    it, no barrier between; the copies a pass of their own, and each order's
    scaled sum into the time integral in its column pass, of the registers
    that hold the order's column (`_chained`)."""
    context = _context(arch, precision, backend=backend, **LANES)
    group = eg.plan(_descrs('derivative', 6, precision), context)
    assert group is not None and group.lanes is not None
    plans = group.lanes.phases
    assert group.lanes.warps == 1 and plans[0] is None
    rows = [(0, 35), (0, 20), (0, 10), (0, 4), (0, 1)]
    assert [(p.classes, p.ranges) for p in plans[1:]] == [
        (1, (r,)) for r in rows]
    kernel = _generate(_descrs('derivative', 6, precision),
                       context).get_kernel()
    loops = re.findall(r'for \(int32_t v\d+_c = 0; v\d+_c < (\d+);', kernel)
    assert loops == ['9'] * 6
    # The integral's sum takes an order's column from the registers that
    # computed it: the last order, which no later order reads, is only
    # written; the order before it is read back by the last.
    dests = re.findall(r'//   (m\d+)\[i,j\](?:@\S+)? \+?= ', kernel)
    orders = [m for m in dict.fromkeys(dests) if m != dests[-1]]
    assert len(orders) == 6
    last, before = orders[-1], orders[-2]
    assert re.search(rf'glb_{last}\[[^]]*\] = ', kernel)
    assert not re.search(rf'= [^;]*glb_{last}\[', kernel)
    assert re.search(rf'= [^;]*glb_{before}\[', kernel)


@pytest.mark.parametrize('shape,order,precision,arch,values', [
    ('derivative', 4, 's', 'sm_90', None),
    ('derivative', 4, 'd', 'gfx942', None),
    ('derivative', 4, 's', 'gfx1150', None),
    ('derivative', 6, 's', 'sm_90', None),
    ('derivative', 4, 's', 'sm_90', 'memory'),
    ('derivative', 4, 'd', 'gfx942', 'table'),
    ('plConvertToNodal', 4, 's', 'sm_90', None),
    ('plConvertToNodal', 6, 'd', 'gfx942', None)])
def test_the_derivative_computes_on_the_oracle(shape, order, precision,
                                               arch, values):
    """The lanes form's derivative on the host oracle, over a group and seven
    elements more -- a trip and a partial one -- every output against NumPy:
    the time integral and each order's derivative, the rows of a buffer past
    its order's window as the oracle filled them; the operators' numbers
    literals, read from their memory or from a table.  And a copy added to
    a product's output after it (`plConvertToNodal`)."""
    import numpy as np
    backend = 'cuda' if arch.startswith('sm_') else 'hip'
    options = {'multi_element_values': values} if values else {}
    context = _context(arch, precision, backend=backend, **LANES, **options)
    count = eg.lane_elements(context) + 7
    descrs = _descrs(shape, order, precision)
    tensors, inputs, scalars, after = _on_the_oracle(
        _generate(descrs, context), count)
    want = _reference_all(descrs)(inputs, scalars)
    if shape == 'derivative':
        assert sorted(scalars) == [f'power({d})' for d in range(order)]
        assert sorted(want) == sorted(
            ['I'] + [f'dQ({d})' for d in range(order)])
    tolerance = 1e-12 if precision == 'd' else 1e-6
    for alias in want:
        got = after(tensors[alias])
        for e in range(count):
            scale = np.abs(want[alias][e]).max() or 1.0
            error = np.abs(got[e] - want[alias][e]).max() / scale
            assert error <= tolerance, (alias, e, error)


@pytest.mark.parametrize('order,precision', [
    (4, 's'), (6, 's'), (4, 'd'), (6, 'd')])
def test_the_derivative_computes_on_the_gpu(order, precision, target,
                                            cache_root, tensorforge_include):
    """The lanes form's derivative built and run on the local GPU against
    NumPy, over a batch whose last group is partial and which gives a block
    of the persistent grid several trips: the time integral, which sums
    every order, and the last order's derivative, the chain's end."""
    from harness.runner import run_case
    if (target.backend, target.vendor) not in eg.TARGETS:
        pytest.skip(f'the layout is not written for {target.backend} on '
                    f'{target.vendor}')
    group = eg.plan(_descrs('derivative', order, precision),
                    _context(target.arch, precision, backend=target.backend,
                             **LANES))
    assert group is not None and group.lanes is not None
    assert group.lanes.phases
    for output in ('I', f'dQ({order - 1})'):
        result = run_case(_derivative_case(order, precision, output, LANES),
                          target, cache_root, tensorforge_include)
        assert result.passed, (output, result.max_abs_err,
                               result.max_rel_err, result.detail)


# -- `auto` ------------------------------------------------------------------ #

AUTO = {'multi_element': 'auto'}


def _pvc(precision='s', **options):
    return _context('pvc', precision, backend='oneapi', **options)


@pytest.mark.parametrize('shape,order', [
    ('volume', 4), ('volume', 6), ('sparse-star', 6), ('localFlux', 4),
    ('localFlux', 6), ('neighboringFlux', 6), ('localFluxAll', 4),
    ('localFluxAll', 6)])
def test_auto_takes_the_lanes_form_at_f32_on_intel(shape, order):
    """`auto` on PVC: the lanes form of an F32 section whose plan holds the
    output in one row range (`AUTO_LANES`) -- the volume, with its stars
    dense or stored sparse, and the flux kernels -- the group `force`
    takes in the lanes form."""
    group = eg.plan(_descrs(shape, order, 's'), _pvc(**AUTO))
    assert group is not None and group.lanes is not None
    assert len(group.lanes.ranges) == 1
    forced = eg.plan(_descrs(shape, order, 's'), _pvc(**LANES))
    assert (group.elements, group.lanes) == (forced.elements, forced.lanes)
    assert {s.lowering for s in group.section.steps} == {'lane'}


@pytest.mark.parametrize('order', [4, 6])
def test_auto_takes_no_f64_section_yet(order):
    """At F64 the lanes form lost on PVC under the plans measured so far:
    `auto` takes none, though the lanes form plans one."""
    section = eg.classify(_volume(order, 'd'))[0]
    assert eg.lanes_plan(section, _pvc('d', **AUTO))[0] is not None
    lanes, why = eg.auto_takes(section, _pvc('d', **AUTO))
    assert lanes is None and 'no F64 section on intel' in why
    assert eg.plan(_volume(order, 'd'), _pvc('d', **AUTO)) is None


@pytest.mark.parametrize('backend,arch', [
    ('cuda', 'sm_90'), ('cuda', 'sm_120'), ('hip', 'gfx942'),
    ('hip', 'gfx1150'), ('esimd', 'pvc')])
def test_auto_takes_nothing_but_intel_under_spmd(backend, arch):
    """Not on CUDA or HIP, which nothing has measured yet, and not under
    the explicit vector, which the layout is not written for (`TARGETS`)."""
    context = _context(arch, 's', backend=backend, **AUTO)
    assert eg.plan(_volume(4, 's'), context) is None
    if backend != 'esimd':
        section = eg.classify(_volume(4, 's'))[0]
        assert 'no F32 section on' in eg.auto_takes(section, context)[1]


def test_auto_takes_one_row_range(monkeypatch):
    """Where the plan cuts the output into ranges -- here at a budget that
    holds no o6 volume in one -- `auto` takes nothing: each range computes
    again the rows of the intermediate the other needs too."""
    monkeypatch.setattr(eg, 'LANE_REGISTERS', 120)
    section = eg.classify(_volume(6, 's'))[0]
    lanes = eg.lanes_plan(section, _pvc(**AUTO))[0]
    assert lanes is not None and len(lanes.ranges) == 2
    taken, why = eg.auto_takes(section, _pvc(**AUTO))
    assert taken is None and '2 row ranges' in why
    assert eg.plan(_volume(6, 's'), _pvc(**AUTO)) is None


@pytest.mark.parametrize('fma,taken', [
    ('auto', True), ('lanes', True), ('warp', False)])
def test_auto_is_the_lanes_form_and_warp_declines_it(fma, taken):
    """`auto`'s form is its rule's, the lanes form, whatever
    `multi_element_fma` says short of `warp`, which declines it; `force`
    keeps the option's own form, the warp form for `auto` on Intel."""
    group = eg.plan(_volume(4, 's'), _pvc(multi_element='auto',
                                          multi_element_fma=fma))
    assert (group is not None) is taken
    assert group is None or group.lanes is not None
    forced = eg.plan(_volume(4, 's'), _pvc(multi_element='force',
                                           multi_element_fma=fma))
    assert (forced.lanes is not None) is (fma == 'lanes')


def test_a_kernel_auto_takes_is_neither_cut_nor_reassociated():
    """The generator keeps its gates for `auto` as for `force`: fission,
    `auto` on Intel under SPMD, builds no probe of the list to decide its
    cuts by -- the layout plans the kernel's one section whole -- and the
    operator product put first (`reassociate_products`, asked for here) is
    the order the lanes form refuses."""
    for shape, order in (('localFluxAll', 6), ('localFlux', 4)):
        gen = _generate(_descrs(shape, order, 's'), _pvc(
            fission='auto', reassociate_products=True, **AUTO))
        assert gen._fission_pieces == 0
        built = gen.built
        assert built._element_group is not None
        assert built._element_group.lanes is not None
        assert len(built._sections) == 1


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
    where it assigns -- for each tile of D the walk holds, its M tiles
    together and its slots -- no buffer and no barrier between.  A section
    whose matrix steps write intermediates only keeps no table."""
    section = eg.classify(_descrs(shape, 6, precision))[0]
    group = eg.fill(section, _context(arch, precision))[0]
    tables = [t.alias for t in group.section.pointer_tables]
    assert tables == (['Q'] if adds or stores else [])
    lines = _kernel(arch=arch, precision=precision, shape=shape,
                    multi_element='force').splitlines()
    held = sum(units.group * units.slots for step, units in _walks(
        shape, precision=precision, arch=arch) if not step.dest.is_tmp)
    reductions = [line for line in lines if 'red.global.add.' in line]
    assert len(reductions) == len([r for r in reductions if '&eQ_' in r])
    assert len(reductions) == adds * held
    assert len(_stores(lines, r'eQ_\d+')) == (adds + stores) * held
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
    (6, 's', 'sm_90', 16, 4), (6, 'd', 'sm_90', 16, 0),
    (8, 's', 'sm_90', 7, 4), (6, 's', 'sm_120', 8, 4),
    (4, 's', 'sm_120', 16, 4)])
def test_a_second_stage_where_it_costs_no_group_size(order, precision, arch,
                                                     elements, doubled):
    """`enable_multibuffer` and a `move_distance` over the volume's four
    inputs: each takes a second stage where the group keeps its size,
    copied a trip ahead into the stage the trip does not read; where the
    stage would cost the group -- o6 F64 on sm_90, from 16 to 8 -- the
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


def _walks(shape='volume', order=6, precision='s', arch='sm_90'):
    """The plan's matrix steps, each with how its tiles of D go to the
    block's warps (`element_group.tile_units`)."""
    group = eg.plan(_descrs(shape, order, precision),
                    _context(arch, precision, multi_element='force'))
    return [(step, eg.tile_units(group.section, step, group.elements,
                                 int(arch[3:])))
            for step in group.section.steps
            if step.lowering in eg.MATRIX_LOWERINGS]


@pytest.mark.parametrize('arch,precision,instruction,terms', [
    ('sm_90', 's', 'm16n8k8.row.col.f32.tf32.tf32.f32', 3),
    ('sm_90', 'd', 'm16n8k4.row.col.f64.f64.f64.f64', 1),
    ('sm_80', 'd', 'm8n8k4.row.col.f64.f64.f64.f64', 1)])
def test_an_operator_step_is_tiled_on_the_matrix_engine(arch, precision,
                                                        instruction, terms):
    """The volume's three operator products one product over K, walked a
    kept tile of B at a time (`element_group.TileUnits`): the walk's body
    a product each, the MMAs of every M tile it takes together and every
    slot in it -- three products of TF32 halves in F32, one in F64."""
    (step, units), = _walks(precision=precision, arch=arch)
    count = len(step.products) * units.group * units.slots * terms
    kernel = _kernel(arch=arch, precision=precision, multi_element='force')
    assert kernel.count('mma.sync.aligned.') == count
    assert kernel.count('mma.sync.aligned.' + instruction) == count
    if terms == 3:
        # A split once an M tile and k-block, for every slot -- B comes
        # split, from the stages' copy, one split each.
        assert kernel.count('tensorforge::splitFloatTF32(') == len(
            step.products) * (1 + units.group * 4)


@pytest.mark.parametrize('order,precision,kept,tiles', [
    (6, 's', 71, 105), (6, 'd', 112, 189), (8, 's', 285, 495)])
def test_a_tile_of_zeros_is_neither_stored_nor_multiplied(order, precision,
                                                          kept, tiles):
    """B's tiles where the operator has no entry in the product's window
    are left out of its stage (`Stage.tiles`), and the walk's entries
    (`TileUnits.entries`) list each stored tile of a unit's slots once, at
    the place it is stored; with `skip_known_zeros` off, every tile."""
    def stages(**options):
        group = eg.plan(_descrs('volume', order, precision), _context(
            'sm_90', precision, multi_element='force', **options))
        step = next(s for s in group.section.steps
                    if s.lowering in eg.MATRIX_LOWERINGS)
        return [group.section.stage(p.first.tensor, (p.rows, p.depth))
                for p in step.products]
    staged = stages()
    assert sum(len(st.tiles) for st in staged) == kept
    assert sum((st.depth // st.k) * (st.width // 8) for st in staged) \
        == tiles
    (_, units), = _walks(order=order, precision=precision)
    for p, stage in enumerate(staged):
        listed = {(kt, nt, place)
                  for (_, _, nts), walked in zip(units.units, units.entries)
                  for kt, places in walked[p]
                  for nt, place in zip(nts, places) if place >= 0}
        assert sorted((kt, nt) for kt, nt, _ in listed) == sorted(
            stage.tiles)
        assert all(stage.tiles[place] == (kt, nt)
                   for kt, nt, place in listed)
    assert sum(len(st.tiles) for st in stages(skip_known_zeros=False)) \
        == tiles


@pytest.mark.parametrize('order,window,aligned', [(6, 71, 72),
                                                  (8, 285, 281)])
def test_the_grid_of_b_starts_at_the_window_or_a_tile(order, window,
                                                      aligned):
    """The tiles of the volume's operator step kept on its two grids
    (`kept_tiles`): N from the step's first row, as the volume stages them,
    and from that rounded down to a tile, where a consumer's K tiles have
    to meet a producer's N tiles -- neither fewer everywhere."""
    step = next(s for s in eg.classify(_volume(order, 's'))[0].steps
                if s.kind is eg.Kind.OPERATOR)

    def kept(n0):
        return sum(len(eg.kept_tiles(p, n0, 8, -(-(p.depth[1] - p.depth[0])
                                                 // 8) * 8,
                                     -(-(step.rows[1] - n0) // 8) * 8))
                   for p in step.products)
    assert step.rows[0] % 8
    assert kept(step.rows[0]) == window
    assert kept(step.rows[0] - step.rows[0] % 8) == aligned


@pytest.mark.parametrize('order,precision,arch', [
    (6, 's', 'sm_90'), (6, 'd', 'sm_90'), (8, 's', 'sm_90'),
    (6, 's', 'sm_120'), (4, 's', 'sm_120'), (6, 'd', 'sm_80')])
def test_the_units_hold_every_tile_of_d_once(order, precision, arch):
    """A matrix step's tiles of D over the block's warps, fixed at
    generation (`tile_units`): every M tile with every kept tile of B in
    exactly one unit -- a tile of D in several, its entries shared out,
    only where the step adds into its output (split K) -- every unit one
    warp's; a unit's M tiles walked a group at a time."""
    group = eg.plan(_descrs('volume', order, precision),
                    _context(arch, precision, multi_element='force'))
    step = next(s for s in group.section.steps
                if s.lowering in eg.MATRIX_LOWERINGS)
    units = eg.tile_units(group.section, step, group.elements,
                          int(arch[3:]))
    stages = [group.section.stage(p.first.tensor, (p.rows, p.depth))
              for p in step.products]
    m, _, _ = eg.matrix_shape(step.lowering, int(arch[3:]))
    mtiles = -(-group.elements * (step.cols[1] - step.cols[0]) // m)
    walked = sorted(
        (mt, nt, p, kt)
        for (lo, hi, nts), listed in zip(units.units, units.entries)
        for p, entries in enumerate(listed) for kt, places in entries
        for nt, place in zip(nts, places) if place >= 0
        for mt in range(lo, hi))
    assert walked == sorted((mt, nt, p, kt) for p, st in enumerate(stages)
                            for kt, nt in st.tiles for mt in range(mtiles))
    if step.dest.is_tmp or step.assigns:
        held = [(mt, nt) for lo, hi, nts in units.units
                for mt in range(lo, hi) for nt in nts if nt >= 0]
        assert len(held) == len(set(held))
    assert len(units.warps) == group.elements
    assert sorted(u for taken in units.warps for u in taken) == list(
        range(len(units.units)))
    if any(not taken for taken in units.warps):
        # Spread: no warp takes a second unit while another has none.
        assert all(len(taken) <= 1 for taken in units.warps)
    assert all(len(nts) == units.slots for _, _, nts in units.units)
    assert 1 <= units.group <= max(hi - lo for lo, hi, _ in units.units)


def test_the_walk_in_slots_computes_on_the_oracle(monkeypatch):
    """A warp's N tiles in several slots (`TILE_SLOTS`, one by default),
    which o6 F32 on sm_90 takes three of where it may: A kept over the
    slots, a slot's tile multiplied where it holds one -- the walk that
    tests each slot's stored tile against -1."""
    monkeypatch.setattr(eg, 'TILE_SLOTS', 3)
    (_, units), = _walks()
    assert units.slots > 1 and any(
        place < 0 for walked in units.entries for entries in walked
        for _, places in entries for place in places)
    for alias, error in _oracle_errors('volume', 's', 'sm_90', {}).items():
        assert error <= 1e-6, (alias, error)


@pytest.mark.parametrize('shape', ['volume', 'assign-low'])
def test_a_grid_aligned_to_the_tiles_computes_on_the_oracle(shape,
                                                            monkeypatch):
    """B's N grid from the tile below the step's first row (`_staged`'s
    `aligned`): the steps' rows start at 1, so a tile's first row is no
    row of the step, and the walk leaves it out of D -- nothing is written
    outside the output's stored cells, where `Q` is stored from row 1
    (`assign-low`) as where it is not."""
    from tensorforge.reference import kernel_eval
    staged = eg._staged
    monkeypatch.setattr(eg, '_staged', lambda *args, **options: staged(
        *args, **{**options, 'aligned': True}))
    group = eg.plan(_descrs(shape, 6, 's'), _context(multi_element='force'))
    assert {st.n0 for st in group.section.stages if st.window} == {0}
    host = _host()
    kernel = host.Kernel(DESCRIPTIONS[shape](6, 's'), _dtype('s'), 'sm_90')
    base = kernel._name('Q') + '@0'
    stored = set(host._slots(kernel.tensors['Q']))
    written, write = set(), kernel_eval.Slot.write

    def record(self, name, index, value):
        if name == base:
            written.add(int(index))
        return write(self, name, index, value)
    monkeypatch.setattr(kernel_eval.Slot, 'write', record)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        got, _ = kernel.run({'multi_element': 'force'})
    want = kernel.expected()
    assert written and written <= stored, sorted(written - stored)
    scale = max(abs(v) for v in want.values())
    assert max(abs(got[key] - want[key]) for key in want) <= 1e-6 * scale


@pytest.mark.parametrize('shape,precision,pitched,ldmatrix', [
    ('volume', 's', {'I': 36, '_tmp0': 44, '_tmp2': 44, '_tmp4': 44}, True),
    ('volume', 'd', {'I': 38}, False),
    ('chain', 's', {'I': 44}, True)])
def test_f32_a_fragments_come_by_ldmatrix(shape, precision, pitched,
                                          ldmatrix):
    """In F32 a matrix step reads A by ldmatrix, a load of four 8x8 matrices
    a warp -- in the walk's body, once for each M tile it takes together --
    so the buffers it reads have a pitch: rows of 16 bytes, four floats
    more than a multiple of eight, long enough for every padded step of K;
    F64 reads A lane by lane.  The star step's group operand has one as
    well (`mix`), which it reads 16 bytes at a time: every pitch an odd
    number of 16 bytes."""
    section = eg.classify(_descrs(shape, 6, precision))[0]
    group = eg.fill(section, _context('sm_90', precision))[0]
    got = {b.tensor.alias: b.pitch for b in group.section.inputs
           + group.section.intermediates if b.stride}
    assert got == pitched
    size = _dtype(precision).size()
    assert all(pitch * size % 32 == 16 for pitch in got.values())
    kernel = _kernel(precision=precision, shape=shape,
                     multi_element='force')
    count = sum(len(step.products) * units.group
                for step, units in _walks(shape, precision=precision))
    assert kernel.count('tensorforge::ldmatrixX4(') == (
        count if ldmatrix else 0)


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
    """Every pass orders ldmatrix by what it says it reads: each load of
    four matrices (`Op.MATRIX_LOAD`) is a read of the buffer its lanes
    name, in shared memory, and no pure function a pass may move or merge;
    and its fragments are where the instruction that reads them takes A
    (`engines.matrix_transfer_bits`, `nvidia.a_fragment_bits`)."""
    from tensorforge.backend.instructions.compute.primitives import nvidia
    from tensorforge.backend.pir.core import Effect, MemSpace, Op, walk_stmts
    gen = _generate(_volume(6, 's'), _context(multi_element='force'))
    loads = [st for section in gen.built._sections
             for st in walk_stmts(section.body)
             if st.op == Op.MATRIX_LOAD and st.attr('engine') == 'ldmatrix']
    (step, units), = _walks()
    assert len(loads) == len(step.products) * units.group
    atom = next(a for a in nvidia.INSTRS if (a.m, a.n, a.k) == (16, 8, 8)
                and a.mode == nvidia.MMAMode.TF32)
    for st in loads:
        assert not st.pure and not st.movable
        assert [(a.kind, a.space) for a in st.accesses] == [
            (Effect.READ, MemSpace.SHARED)]
        assert st.attr('count') == 4
        assert st.attr('layout') == nvidia.a_fragment_bits(atom)


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


@pytest.mark.parametrize('precision,warps', [('s', 3), ('d', 6)])
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


def test_a_number_in_the_padding_reaches_no_result(monkeypatch):
    """A pitched input's padding holds copies of its column's first row
    (`MultiElementSection._stage`), never zeroed, since B's padding is.
    The chain's I is the tiling's A, read by ldmatrix over K padded past
    its 35 rows, in a buffer of 44 rows a column: with a large number in
    its first row, and so in every column's padding, the outputs still
    agree with NumPy."""
    from tensorforge.reference import kernel_eval
    plan = eg.plan(_descrs('chain', 6, 's'),
                   _context('sm_90', 's', multi_element='force'))
    staged = {b.tensor.alias: b for b in plan.section.inputs}['I']
    assert staged.pitch > staged.height
    host = _host()
    kernel = host.Kernel(DESCRIPTIONS['chain'](6, 's'), _dtype('s'), 'sm_90')
    base = kernel._name('I') + '@0'
    first = {slot for slot, cell in host._slots(kernel.tensors['I']).items()
             if cell[0] == 0}
    read = kernel_eval.Slot.read

    def sentinel(self, name, index):
        if name == base and int(index) in first:
            self.data.setdefault((name, int(index)), 1e15)
        return read(self, name, index)
    monkeypatch.setattr(kernel_eval.Slot, 'read', sentinel)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        got, _ = kernel.run({'multi_element': 'force'})
    want = kernel.expected()
    assert max(abs(want[key]) for key in want) > 1e15
    for alias in {alias for alias, _ in want}:
        keys = [key for key in want if key[0] == alias]
        scale = max(abs(want[key]) for key in keys)
        error = max(abs(got[key] - want[key]) for key in keys) / scale
        assert error <= 1e-6, (alias, error)


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
    ('volume', 4, 's', {**LANES, 'arch': 'gfx942'}),
    ('localFlux', 4, 's', {**LANES, 'arch': 'gfx1150'}),
    ('volume', 4, 's', {**LANES, 'multi_element_values': 'memory'}),
    ('volume', 4, 's', {**LANES, 'multi_element_values': 'table'}),
    ('localFlux', 4, 'd', {**LANES, 'arch': 'gfx942',
                           'multi_element_values': 'table'}),
    ('volume', 4, 'd', {**LANES, 'columns': 3}),
    ('localFluxAll', 4, 's', {**LANES, 'columns': 3, 'arch': 'gfx942'}),
    ('sparse-star', 4, 'd', {'tensor_cores': False}),
    ('sparse-star', 4, 's', {'arch': 'gfx942'}),
    ('sparse-star', 4, 's', LANES),
    ('sparse-star', 4, 'd', {**LANES, 'arch': 'gfx942'}),
    ('plConvertToModal', 4, 's', {**LANES, 'columns': 3}),
    ('plConvertToModal', 6, 'd', {**LANES, 'columns': 3, 'arch': 'gfx942'})],
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
         'gfx942-lanes-volume-o4s', 'gfx1150-lanes-localFlux-o4s',
         'lanes-volume-o4s-memory', 'lanes-volume-o4s-table',
         'gfx942-lanes-localFlux-o4d-table', 'lanes-volume-o4d-columns',
         'gfx942-lanes-localFluxAll-o4s-columns', 'sparse-star-o4d',
         'gfx942-sparse-star-o4s', 'lanes-sparse-star-o4s',
         'gfx942-lanes-sparse-star-o4d', 'lanes-plConvertToModal-o4s-columns',
         'gfx942-lanes-plConvertToModal-o6d-columns'])
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
    tensors, inputs, _, after = _on_the_oracle(gen, count)
    output = next(t for t in tensors.values() if not t.is_tmp
                  and t.addressing.kind is not Addressing.NONE
                  and any(d.writes().tensor is t for outer in gen.descr_list
                          for d in outer.operations()))
    got = after(output)
    want = _reference(_descrs(shape, order, precision))(
        inputs, inputs[output.alias])
    tolerance = 1e-12 if precision == 'd' else 1e-6
    for e in range(count):
        scale = np.abs(want[e]).max() or 1.0
        error = np.abs(got[e] - want[e]).max() / scale
        assert error <= tolerance, (e, error)


def _on_the_oracle(gen, count):
    """`gen`'s kernel run on the host oracle over `count` elements, from the
    oracle's own fill of memory, the operators their description's numbers,
    and no access to shared memory that no barrier orders
    (`kernel_eval.Races`): `(tensors, inputs, scalars, after)` -- the
    description's tensors by alias; each one's box but an intermediate's,
    element by element, as the kernel finds it; the scalar factors' values;
    and `after(t)`, a tensor's box as the kernel leaves it."""
    import numpy as np
    from tensorforge.common.basic_types import Addressing
    from tensorforge.reference import kernel_eval
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
    preset, inputs, scalars = {}, {}, {}
    for alias, t in tensors.items():
        if t.is_tmp:
            continue
        if not t.bbox.lower():
            # A scalar passed by value: the oracle's fill of its one slot.
            scalars[alias] = fill.read(t.name, 0)
        elif t.addressing.kind is Addressing.NONE:
            preset[t.name] = [float(v) for v in t.storage_values()]
            (r0, r1), (c0, c1) = box(t)
            inputs[alias] = np.asarray(t.get_values())[None, r0:r1, c0:c1]
        else:
            inputs[alias] = memory(t, lambda e, s, t=t: fill.read(
                f'{t.name}@{e}', s))
    lanes, mults = kernel_eval.launch_geometry(gen.get_launcher())
    races = []
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        mem = kernel_eval.evaluate_wave(
            gen.get_kernel(), lanes, seed=SEED, globals_only=True,
            mults=mults, block=True, preset=preset, elements=count,
            races=races)
    assert not races, races[:3]

    def after(t):
        return memory(t, lambda e, s: mem.get((f'{t.name}@{e}', s),
                                              fill.read(f'{t.name}@{e}', s)))

    return tensors, inputs, scalars, after


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


def _reference_all(descrs):
    """NumPy over a description of several outputs -- a chained section,
    its scaled copies too -- every element at once, in double precision:
    `reference(inputs, scalars)` is each element output's box after the
    description, from its box in `inputs` where nothing writes it.  The
    scalars are the factors' values by alias."""
    import numpy as np
    ops = [op for d in descrs for op in d.operations()]

    def reference(inputs, scalars):
        full = {}

        def whole(tensor):
            if tensor.alias not in full:
                lo = tuple(int(x) for x in tensor.bbox.lower())
                hi = tuple(int(x) for x in tensor.bbox.upper())
                data = np.asarray(inputs[tensor.alias], dtype=np.float64)
                out = np.zeros((data.shape[0],) + tuple(hi))
                out[:, lo[0]:hi[0], lo[1]:hi[1]] = data
                full[tensor.alias] = out
            return full[tensor.alias]

        written = []
        for op in ops:
            scaled = eg._scaled(op)
            if scaled is not None:
                src, factor = scaled
                (r0, c0), (r1, c1) = src.bbox.lower(), src.bbox.upper()
                value = whole(src.tensor)[:, r0:r1, c0:c1] * (
                    1.0 if factor is None else scalars[factor.tensor.alias])
            else:
                a, b = op.ops
                (r0, k0), (r1, k1) = a.bbox.lower(), a.bbox.upper()
                c0, c1 = b.bbox.lower()[1], b.bbox.upper()[1]
                value = np.einsum('...ik,...kj->...ij',
                                  whole(a.tensor)[:, r0:r1, k0:k1],
                                  whole(b.tensor)[:, k0:k1, c0:c1])
            dest = op.dest.tensor
            if dest.is_tmp and not op.add:
                full[dest.alias] = np.zeros(
                    (value.shape[0],) + tuple(int(x) for x in
                                              dest.bbox.upper()))
            target = full[dest.alias] if dest.is_tmp else whole(dest)
            if not dest.is_tmp and dest.alias not in written:
                written.append(dest.alias)
            if op.add:
                target[:, r0:r1, c0:c1] += value
            else:
                target[:, r0:r1, c0:c1] = value
        out = {}
        for alias in written:
            tensor = next(op.dest.tensor for op in ops
                          if op.dest.tensor.alias == alias)
            lo = tuple(int(x) for x in tensor.bbox.lower())
            hi = tuple(int(x) for x in tensor.bbox.upper())
            out[alias] = full[alias][:, lo[0]:hi[0], lo[1]:hi[1]]
        return out

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
    ('sparse-star', 6, 'd', {'tensor_cores': False}),
    ('localFlux', 6, 's', {}), ('neighboringFlux', 4, 'd', {}),
    ('localFluxAll', 4, 'd', {}), ('sparse-star', 6, 's', {}),
    ('direction', 4, 's', {})]


@pytest.mark.parametrize('shape,order,precision,options', GPU_CASES, ids=[
    'o4s', 'o6s', 'o4d', 'o6d', 'o6s-own-barriers', 'o6s-fma',
    'o6s-fma-dense', 'o4d-fma', 'o6d-fma-own-barriers', 'chain-o6s',
    'chain-o6d', 'reuse-o6s', 'chain-o6d-fma', 'reuse-o6s-fma',
    'localFlux-o6d-fma', 'neighboringFlux-o4s-fma', 'localFluxAll-o6s-fma',
    'direction-o6d-fma', 'columns-o6s-fma', 'o4s-two-stages',
    'chain-o6s-two-stages', 'sparse-star-o6d-fma', 'localFlux-o6s',
    'neighboringFlux-o4d', 'localFluxAll-o4d', 'sparse-star-o6s',
    'direction-o4s'])
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


def _derivative_case(order, precision, output, options, batch=4099,
                     name='multi_element_lanes'):
    """The harness's case of the derivative, compared at `output` -- the
    time integral or an order's derivative: the factors the driver's 0.5
    (`driver_emit.collect_operands`), an output nothing gives the kernel
    zero, as the driver hands it over.  The rows grow by orders of magnitude
    from an order to the next, up to 1e9 at o6, and a row's rounding with
    them: each row is held to its size over the batch, 1e-5 of it in F32
    and 1e-13 in F64 -- the GPUs' worst is 2e-7 and 6e-16."""
    import numpy as np
    descrs = _descrs('derivative', order, precision)
    outputs, scalars = {}, {}
    for descr in descrs:
        for op in descr.operations():
            scaled = eg._scaled(op)
            if scaled is not None and scaled[1] is not None:
                t = scaled[1].tensor
                scalars[t.alias] = (
                    float(np.asarray(t.get_values()).reshape(-1)[0])
                    if t.has_values() else 0.5)
            if not op.dest.tensor.is_tmp:
                outputs.setdefault(op.dest.tensor.alias, op.dest.tensor)
    every = _reference_all(descrs)

    def reference(inputs, dest_in):
        given = dict(inputs)
        for alias, t in outputs.items():
            if alias not in given:
                given[alias] = np.zeros((len(dest_in),) + tuple(
                    int(hi) - int(lo) for lo, hi in zip(t.bbox.lower(),
                                                        t.bbox.upper())))
        return every(given, scalars)[output]

    def tolerance(expected):
        rows = np.abs(expected).max(axis=(0, 2), keepdims=True)
        return 0.0, rows * (1e-13 if precision == 'd' else 1e-5)

    return types.SimpleNamespace(
        NAME=f'{name}_derivative_o{order}{precision}', DTYPE=_dtype(precision),
        BATCH=batch, OUTPUT=output, TOL=tolerance,
        OPTIONS=dict(options), INPUT_TRANSFORM=_described(descrs),
        descr_list=lambda: _descrs('derivative', order, precision),
        reference=reference)


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
