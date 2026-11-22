# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""A matrix step's D added into the element's output by bulk reductions
(`Options.matrix_output`, `element_group.Step.output`): D staged in shared
memory as columns of the output, and added by `cp.reduce.async.bulk .add`
(F32 and F64, no tensor map) through each element's own pointer -- `bulk`
a column a reduction, a lane of the element's warp each, the columns an
odd number of sixteen bytes apart; `box` the element's whole box in one --
instead of a `red.global.add` a cell.  One reduction a column (or element)
a launch: deterministic, refused where two steps add into one output.

What is checked here: the refusals and their reasons; the geometry; the
kernel's text and its note of the +0.0 the reductions add outside the
step's windows; the host oracle over two trips of a group with a masked
warp on every matrix lowering (mma_sync and DMMA, wgmma, tcgen05; with the
operator ring and the folded star) against NumPy and without a race; the
output bit for bit the reductions' a cell inside the windows and +0.0
added outside them, a -0.0 there turned +0.0; and on the local GPU and on
a GH200 or a GB200, the volume against NumPy.
"""

from __future__ import annotations

import contextlib
import dataclasses
import io
import math
import types
import warnings

import numpy as np
import pytest

import test_multi_element as me
from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.generators import element_group as eg


def _plan(arch, precision='s', order=4, shape='volume', **options):
    context = me._context(arch, precision, multi_element='force', **options)
    section, why = eg.classify(me._descrs(shape, order, precision))
    assert section is not None, why
    group, why = eg.fill(section, context)
    assert group is not None, why
    return group, context


def _operator(group):
    (step,) = [s for s in group.section.steps if s.kind is eg.Kind.OPERATOR]
    return step


# -- the plan ------------------------------------------------------------------ #

@pytest.mark.parametrize('arch,precision,lowering', [
    ('sm_90', 's', 'mma_sync'), ('sm_90', 'd', 'dmma'),
    ('sm_90a', 's', 'wgmma'), ('sm_100a', 's', 'tcgen05'),
    ('sm_120', 's', 'mma_sync')])
def test_every_matrix_lowering_takes_the_bulk_output(arch, precision,
                                                     lowering):
    options = {'matrix_engine': 'auto'} if arch.endswith('a') else {}
    group, context = _plan(arch, precision, matrix_output='bulk', **options)
    step = _operator(group)
    assert step.lowering == lowering and step.output == 'bulk'
    (note,) = eg.output_notes(group.section, context)
    assert note.startswith('Q += D by a bulk reduction a column of the '
                           'element, rows 0 to 20')
    assert 'receive exactly +0.0 (a -0.0 becomes +0.0)' in note


def test_the_default_adds_a_cell_at_a_time():
    group, context = _plan('sm_90')
    assert all(not s.output for s in group.section.steps)
    assert eg.output_notes(group.section, context) == []


def test_a_target_without_the_bulk_copy_keeps_the_reductions():
    group, context = _plan('sm_80', matrix_output='bulk')
    assert not _operator(group).output
    (note,) = eg.output_notes(group.section, context)
    assert note == ('matrix_output=bulk declined for Q: no bulk copy on the '
                    'target; a reduction a cell')


def test_the_refusals_say_why():
    """Two steps adding into one output, a step reading it, a step that
    assigns, an output that promises less than sixteen bytes, an
    intermediate."""
    group, context = _plan('sm_90', matrix_output='bulk')
    section = group.section
    step = _operator(group)
    twice = dataclasses.replace(section, steps=section.steps + (step,))
    assert eg.output_refusal(twice, step, context) == (
        '2 steps add into Q: their reductions would land in an order of '
        'their own')
    (mix, *_) = [s for s in section.steps if s.kind is eg.Kind.MIXING]
    reading = dataclasses.replace(mix, products=(dataclasses.replace(
        mix.products[0], first=types.SimpleNamespace(tensor=step.dest)),)
        + mix.products[1:])
    read = dataclasses.replace(section, steps=section.steps + (reading,))
    assert eg.output_refusal(read, step, context) == (
        'a step reads Q: the reductions\' writes land behind the wait for '
        'their reads only')
    assigning = dataclasses.replace(step, products=(dataclasses.replace(
        step.products[0], add=False),) + step.products[1:])
    assert eg.output_refusal(section, assigning, context) == \
        'the step assigns Q'
    tensor = step.dest

    class Loose:
        """Q, promising eight bytes."""
        def __getattr__(self, name):
            return getattr(tensor, name)
        alignment = 8
        is_tmp = False
    loose = dataclasses.replace(step, products=tuple(dataclasses.replace(
        p, dest=types.SimpleNamespace(tensor=Loose(), bbox=p.dest.bbox))
        for p in step.products))
    assert eg.output_refusal(section, loose, context) == (
        'Q promises 8 bytes of alignment, the bulk reduction 16')
    (mix, *_) = [s for s in section.steps if s.kind is eg.Kind.MIXING]
    assert eg.output_refusal(section, mix, context) == 'mix writes no D'


@pytest.mark.parametrize('arch,precision,order,options,elements,taken', [
    ('sm_100a', 's', 8, {'matrix_engine': 'auto', 'matrix_ring': 'prepared',
                         'matrix_star': 'engine'}, 14, True),
    ('sm_100a', 's', 8, {'matrix_engine': 'auto', 'matrix_ring': 'prepared'},
     7, True),
    ('sm_90a', 's', 8, {'matrix_engine': 'auto', 'matrix_ring': 'prepared'},
     12, False),
    ('sm_90', 'd', 6, {}, 16, False),
    ('sm_90', 's', 6, {}, 16, True)],
    ids=['tcgen05-o8-folded', 'tcgen05-o8', 'wgmma-o8', 'dmma-o6d',
         'mma_sync-o6s'])
@pytest.mark.parametrize('form', ['bulk', 'box'])
def test_the_bulk_output_costs_no_group_size(form, arch, precision, order,
                                             options, elements, taken):
    """The staged columns take room the reductions a cell do not: the fill
    rule takes them only where the group keeps its size and its lowering's
    other choices -- on tcgen05, which stages D in shared memory anyway, at
    o8 with and without the folded star; not on wgmma at o8 (12 elements
    would be 4) nor for F64 at o6 on sm_90 (16 would be 8) -- and the
    kernel's comment says where it declined them."""
    group, context = _plan(arch, precision, order=order, **options)
    asked, context = _plan(arch, precision, order=order,
                           matrix_output=form, **options)
    assert group.elements == asked.elements == elements
    assert (eg.output_of(asked.section) == form) is taken
    (note,) = eg.output_notes(asked.section, context)
    if taken:
        assert note.startswith('Q += D by ')
    else:
        assert note == (f'matrix_output={form} declined for Q: its staged '
                        f'columns do not fit beside the group the fill rule '
                        f'chose -- its size and its lowering\'s other choices '
                        f'kept; a reduction a cell')
        assert asked.section.shape() == group.section.shape()


@pytest.mark.parametrize('form,expected', [('bulk', (0, 120, 124)),
                                           ('box', (0, 128, 128))])
def test_the_staged_columns(form, expected):
    """At o8 the step adds rows 1 to 120 of Q's 128: `bulk` stages rows 0
    to 120 of each column, 124 floats apart (31 sixteen bytes: a warp's
    columns on distinct banks); `box` the whole column, Q's 128 apart."""
    group, _ = _plan('sm_100a', order=8, matrix_engine='auto',
                     matrix_output=form)
    assert eg.output_geometry(group.section, _operator(group)) == expected


def test_f64_columns_are_sixteen_bytes_too():
    group, _ = _plan('sm_90', 'd', order=6, matrix_output='bulk')
    lo, hi, stride = eg.output_geometry(group.section, _operator(group))
    assert (lo * 8) % 16 == 0 and ((hi - lo) * 8) % 16 == 0
    assert (stride * 8 // 16) % 2 == 1


# -- the kernel ----------------------------------------------------------------- #

@pytest.mark.parametrize('form', ['bulk', 'box'])
def test_the_kernel_adds_by_bulk_reductions(form):
    import test_multi_element_async as tma
    src = tma._kernel('sm_100a', 'volume', 8, 's', matrix_output=form,
                      matrix_ring='prepared', matrix_star='engine'
                      ).get_kernel()
    assert src.count('tensorforge::bulkReduceAdd(') == 1
    assert 'tensorforge::bulkWaitRead<0>();' in src
    assert '// output: Q += D by ' in src
    # The staged columns' last stores fenced for the async proxy, and the
    # element warps met, ahead of the reduction that reads them.
    at = src.index('tensorforge::bulkReduceAdd(')
    fence = src.rfind('tensorforge::fenceProxyAsync();', 0, at)
    assert src.rfind('_dbox[', 0, at) < fence
    assert '__syncthreads();' in src[fence:at]
    # tcgen05's rows of D sixteen bytes a store over the span the reduction
    # adds, zeros included: Q's rows 0 to 120 (`bulk`) or 128 (`box`).
    stores = [line for line in src.splitlines()
              if '_dbox[' in line and ' = ' in line]
    assert all('tensorforge::VectorT<float, 4>' in line for line in stores)
    assert len(stores) == {'bulk': 30, 'box': 32}[form]
    assert 'red.global' not in src and 'atomicAdd' not in src


# -- the oracle ------------------------------------------------------------------ #

def _output(gen):
    """The tensor the generated kernel adds into: the one written that is
    neither an intermediate nor a value in the kernel (`Addressing.NONE`)."""
    for outer in gen.descr_list:
        for d in outer.operations():
            t = d.writes().tensor
            if not t.is_tmp and t.addressing.kind is not Addressing.NONE:
                return t
    raise AssertionError('no output')


def _run(shape, arch, precision, monkeypatch, order=4, count=7, preset=None,
         **options):
    """The section over `count` elements, groups of four, on the host oracle
    against NumPy: `(got, inputs, output, gen)` -- the output's boxes as the
    kernel leaves them.  `preset(output)`, where given, the oracle's first
    value of some slots, by `(name@element, slot)` -- the generated output's
    name, which the description's tensor has only once generated."""
    from tensorforge.reference import kernel_eval
    monkeypatch.setattr(eg, 'ELEMENTS', {key: 4 for key in eg.ELEMENTS})
    monkeypatch.setattr(eg, 'DEFAULT_ELEMENTS', 4)
    context = me._context(arch, precision, multi_element='force', **options)
    gen = me._generate(me._descrs(shape, order, precision), context)
    if preset is not None:
        preset = preset(_output(gen))
        read = kernel_eval.Slot.read

        def poked(self, base, index):
            key = (base, int(index))
            if key in preset and key not in self.data:
                return preset[key]
            return read(self, base, index)
        monkeypatch.setattr(kernel_eval.Slot, 'read', poked)
    tensors, inputs, _, after = me._on_the_oracle(gen, count)
    output = _output(gen)
    assert tensors[output.alias] is output
    got = after(output)
    want = me._reference(me._descrs(shape, order, precision))(
        inputs, inputs[output.alias])
    tolerance = 1e-12 if precision == 'd' else 1e-6
    for e in range(count):
        scale = np.abs(want[e]).max() or 1.0
        assert np.abs(got[e] - want[e]).max() / scale <= tolerance, e
    return got, inputs, output, gen


@pytest.mark.parametrize('arch,precision,options', [
    ('sm_90', 's', {}), ('sm_90', 'd', {}), ('sm_120', 's', {}),
    ('sm_90a', 's', {'matrix_engine': 'auto'}),
    ('sm_100a', 's', {'matrix_engine': 'auto'}),
    ('sm_90a', 's', {'matrix_engine': 'auto', 'matrix_star': 'engine'}),
    ('sm_100a', 's', {'matrix_engine': 'auto', 'matrix_star': 'engine'})],
    ids=['mma_sync', 'dmma', 'sm120', 'wgmma', 'tcgen05', 'wgmma-folded',
         'tcgen05-folded'])
@pytest.mark.parametrize('form', ['bulk', 'box'])
def test_every_element_of_two_trips_on_the_oracle(form, arch, precision,
                                                  options, monkeypatch):
    """Two trips of a group, a warp masked in the second: every element
    against NumPy, and no access to shared memory that no barrier or wait
    orders -- the staged columns read by the reductions until their wait,
    written again only behind it."""
    _run('volume', arch, precision, monkeypatch, matrix_output=form,
         **options)


@pytest.mark.parametrize('arch,options', [
    ('sm_90', {'element_producer': True}),
    ('sm_90a', {'matrix_engine': 'auto', 'element_producer': True}),
    ('sm_100a', {'matrix_engine': 'auto', 'element_producer': True}),
    ('sm_100a', {'matrix_engine': 'auto', 'matrix_ring': 'prepared',
                 'matrix_star': 'engine', 'element_producer': True}),
    ('sm_90', {'enable_sync_block_opt': False}),
    ('sm_100a', {'matrix_engine': 'auto', 'enable_sync_block_opt': False})],
    ids=['mma_sync-producer', 'wgmma-producer', 'tcgen05-producer',
         'tcgen05-folded-producer', 'mma_sync-own-barriers',
         'tcgen05-own-barriers'])
@pytest.mark.parametrize('form', ['bulk', 'box'])
def test_the_reductions_own_wait_and_barrier_on_the_oracle(form, arch,
                                                           options,
                                                           monkeypatch):
    """Where the inputs come from the producer's ring, no drain of the
    element warps' copies waits on the issuing thread's bulk groups ahead of
    the next trip's staged columns: the reductions' own wait does; and
    without the barrier placement, the element warps' own barrier ahead of
    the reductions is all that orders the staged columns' stores before
    them -- two trips of a group against NumPy and without a race."""
    _run('volume', arch, 's', monkeypatch, matrix_output=form, **options)


@pytest.mark.parametrize('form', ['bulk', 'box'])
def test_a_staged_column_written_again_before_the_wait_is_a_race(form,
                                                                 monkeypatch):
    """The bulk engine reads the staged columns until the issuing thread's
    wait retires its reductions (`bulkWaitRead`): without that wait the
    next trip's stores into them are a race the oracle names, `FLIGHT`, as
    it names a TMA box's (test_tma)."""
    monkeypatch.setattr(eg, 'ELEMENTS', {key: 4 for key in eg.ELEMENTS})
    monkeypatch.setattr(eg, 'DEFAULT_ELEMENTS', 4)
    context = me._context('sm_90', 's', multi_element='force',
                          matrix_output=form)
    gen = me._generate(me._descrs('volume', 4, 's'), context)
    wait = 'tensorforge::bulkWaitRead<0>();'
    src = gen.get_kernel()
    assert wait in src
    monkeypatch.setattr(gen, 'get_kernel', lambda: src.replace(wait, ''))
    with pytest.raises(AssertionError, match='FLIGHT'):
        me._on_the_oracle(gen, 7)


def test_a_body_whose_one_copy_is_a_bulk_reduction_keeps_its_wait():
    """A body whose only copy is a bulk reduction has none to lower as
    loads (`Emitter._decide_async`), and the wait such a body's copies
    drop is never the reduction's: the issuing thread's wait on its bulk
    groups stays, behind which the staged run may be written again -- as a
    TMA reduction's does (test_tma)."""
    from tensorforge.backend.pir.allocate import MULT, allocate
    from tensorforge.backend.pir.build import IRBuilder
    from tensorforge.backend.pir.core import BOOL, MemSpace
    from tensorforge.backend.pir.engines import FenceKind
    from tensorforge.common.context import Context
    import test_tma
    ctx = Context(arch='sm_90', backend='cuda', fp_type=Datatype.F32)
    b = IRBuilder(fptype=Datatype.F32, arena='shrMem', context=ctx)
    sa = b.alloc(Datatype.F32, (256,), MemSpace.SHARED, hint='sa',
                 extern='sa', place_align=16)
    out = b.alloc(Datatype.F32, (4096,), MemSpace.GLOBAL, hint='o',
                  extern='m0')
    lane = b.thread_id('x')
    b.store(sa, 1.0, lane)
    b.fence(FenceKind.PROXY)
    b.barrier('block')
    issue = b.op('eq', BOOL, lane, 0)
    b.wait(b.reduce_bulk(out, sa, elems=256, issue=issue, dst_index=(0,),
                         src_index=(0,)))
    body, _ = allocate(b.finish(), arenas={'shrMem': MULT})
    lines = test_tma._emit(body, ctx)
    assert any(line.startswith('tensorforge::bulkReduceAdd(') for line in lines)
    assert 'tensorforge::bulkWaitRead<0>();' in lines


@pytest.mark.parametrize('arch', ['sm_90', 'sm_100a'])
def test_the_window_is_bit_for_bit_and_the_rest_gets_plus_zero(arch,
                                                               monkeypatch):
    """The output after the bulk reductions against after a reduction a
    cell: the rows the step's windows reach bit for bit -- one addition of
    the same D a cell either way -- and the rows outside them, Q's padding
    included where the box takes it, their old value plus exactly +0.0: a
    -0.0 there becomes +0.0, every other number stays."""
    options = {'matrix_engine': 'auto'} if arch.endswith('a') else {}
    group, _ = _plan(arch, matrix_output='box', **options)
    step = _operator(group)
    first = int(step.dest.bbox.lower()[0])
    # Row 0 -- the box's first slot -- outside the windows: the volume's
    # start at row 1 (the constant's derivative is none).
    assert first == 0 and step.rows[0] > 0

    def preset(output):
        """A -0.0 in the first row of every element's first column."""
        return {(f'{output.name}@{e}', 0): -0.0 for e in range(7)}
    boxed, _, _, _ = _run('volume', arch, 's', monkeypatch, preset=preset,
                          matrix_output='box', **options)
    monkeypatch.undo()
    celled, _, _, _ = _run('volume', arch, 's', monkeypatch, preset=preset,
                           **options)
    r0, r1 = step.rows[0] - first, step.rows[1] - first

    def bits(x):
        return np.ascontiguousarray(x).view(np.uint64)
    assert np.array_equal(bits(boxed[:, r0:r1]), bits(celled[:, r0:r1]))
    rest = np.concatenate([boxed[:, :r0], boxed[:, r1:]], axis=1)
    kept = np.concatenate([celled[:, :r0], celled[:, r1:]], axis=1)
    assert np.array_equal(bits(rest), bits(kept + 0.0))
    assert not np.array_equal(bits(rest), bits(kept))
    assert all(math.copysign(1.0, celled[e, 0, 0]) < 0 for e in range(7))
    assert all(math.copysign(1.0, boxed[e, 0, 0]) > 0 for e in range(7))


# -- on the hardware ------------------------------------------------------------- #

@pytest.mark.parametrize('form', ['bulk', 'box'])
@pytest.mark.parametrize('order', [4, 6])
def test_the_bulk_output_computes_on_the_gpu(order, form, target, cache_root,
                                             tensorforge_include):
    """On any GPU with the bulk copy (sm_90 on, the local sm_120 among
    them): the volume against NumPy over a batch ending in a partial group,
    the asynchronous engine where the target has one."""
    if target.backend != 'cuda' or target.arch not in ('sm_90', 'sm_100',
                                                        'sm_120'):
        pytest.skip('no GPU with the bulk copy')
    from harness.runner import run_case
    special = target
    options = {'multi_element': 'force', 'matrix_output': form}
    if target.arch in ('sm_90', 'sm_100'):
        import test_multi_element_async as tma
        special = dataclasses.replace(target, arch=target.arch + 'a')
        tma._nvcc_builds(special.arch)
        options['matrix_engine'] = 'auto'
    group = eg.plan(me._descrs('volume', order, 's'), me._context(
        special.arch, 's', **options))
    if group is None or not _operator(group).output:
        pytest.skip(f'no plan with the bulk output at o{order} on '
                    f'{special.arch}')
    case = types.SimpleNamespace(
        NAME=f'bulk_output_volume_o{order}s_{form}',
        DTYPE=Datatype.F32, BATCH=4099, TOL=(1e-4, 1e-4),
        OPTIONS=options,
        INPUT_TRANSFORM=me._known_zeros(me._descrs('volume', order, 's')),
        descr_list=lambda: me._descrs('volume', order, 's'),
        reference=me._reference(me._descrs('volume', order, 's')))
    result = run_case(case, special, cache_root, tensorforge_include)
    assert result.passed, (result.max_abs_err, result.max_rel_err,
                           result.detail)
