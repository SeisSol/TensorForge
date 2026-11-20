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


DESCRIPTIONS = {
    'volume': _volume_description, 'chain': _chain_description,
    'reuse': lambda order, precision: _chain_description(order, precision,
                                                         reuse=True)}


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
    assert [(b.tensor.alias, b.rows, b.cols) for b in group.section.results] \
        == [('Q', (1, 56), (0, 9))]


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


# -- the kernel -------------------------------------------------------------- #

def _kernel(arch='sm_90', precision='s', shape='volume', **options):
    return _generate(_descrs(shape, 6, precision),
                     _context(arch, precision, **options)).get_kernel()


def test_a_block_is_one_group_of_warps():
    """E warps a block, a warp a multiplication; the launch says so."""
    kernel = _kernel(multi_element='force')
    assert 'block 32x16x1' in kernel and '32 lanes x 16 per block' in kernel


@pytest.mark.parametrize('options,prefix', [
    ({}, 'b'), ({'tensor_cores': False}, 'k')], ids=['matrix', 'fma'])
def test_the_operators_are_staged_once_ahead_of_the_loop(options, prefix):
    """As the matrix tiling reads them (`b`, a window each) or as stored
    for the FMA form (`k`), before the loop and behind one barrier."""
    kernel = _kernel(multi_element='force', **options)
    loop = kernel.index('batchIdGroup0')
    for d in range(3):
        assert -1 < kernel.index(f'{prefix}kDivM{d}') < loop
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
    """The lines that store into the buffers `pattern` names: their data,
    or (`zeros`) the zeros of their padding."""
    zero = re.compile(pattern + r'\[[^]]*\] = 0\.0f?;')
    store = re.compile(pattern + r'\[[^]]*\] = ')
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


@pytest.mark.parametrize('shape', ['volume', 'chain'])
def test_an_element_past_the_end_writes_nothing(shape):
    """Every write of `Q` runs under the warp's element's mask."""
    lines = _kernel(shape=shape, multi_element='force').splitlines()
    writes = _stores(lines, r'glb_m\d+')
    assert writes and all(lines[i - 1].strip() == 'if (batchIdActive0) {'
                          for i in writes)


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


@pytest.mark.parametrize('shape,order,precision,options', [
    ('volume', 4, 's', {}), ('volume', 6, 's', {}), ('volume', 4, 'd', {}),
    ('volume', 6, 'd', {}),
    ('volume', 6, 's', {'enable_sync_block_opt': False}),
    ('volume', 6, 's', {'tensor_cores': False}),
    ('chain', 6, 's', {}), ('chain', 6, 'd', {}), ('reuse', 6, 's', {})],
    ids=['o4s', 'o6s', 'o4d', 'o6d', 'o6s-own-barriers', 'o6s-fma',
         'chain-o6s', 'chain-o6d', 'reuse-o6s'])
def test_the_section_computes_on_the_gpu(shape, order, precision, options,
                                         target, cache_root,
                                         tensorforge_include):
    """Built and run on the local GPU against NumPy, over a batch that ends
    in a partial group and gives a block of the persistent grid several
    trips -- the group's buffers rewritten while another warp of the block
    may still read the last trip's -- under the barriers the placement
    derives and, once, under the instructions' own."""
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
        descr_list=lambda: _descrs(shape, order, precision),
        reference=_reference(_descrs(shape, order, precision)))
    result = run_case(case, target, cache_root, tensorforge_include)
    assert result.passed, (result.max_abs_err, result.max_rel_err,
                           result.detail)
