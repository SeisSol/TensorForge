# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""The tensor memory accelerator's copy of a box (`Op.COPY_ASYNC`, engine
`tma`) through a tensor map (`engines.TensorMap`): sm_90 and newer.

A box of a tensor of two or three dimensions goes into shared memory by one
thread, through a map the launcher encodes and the kernel takes in param
space (`kernel_params.TensorMapParam`, `tensorforge::tensorMapTiled`),
completing on the mbarrier of a bulk cursor with the box's bytes.  The box
lies in the layout of the map's swizzle, which is a TensorForge swizzle
(`engines.swizzle_form`), so the box is read as any swizzled buffer is.
What is checked here: the vocabulary, the swizzle forms against the
hardware's address function, the map and what it refuses, the copy and what
it refuses, the text, the host oracle (`reference.kernel_eval`) on boxes of
every form, the parameter's surfaces, the build with the SASS and the PTX
read back, and on a GPU of sm_90 or newer the boxes copied and read back,
the map encoded by the launcher's code.
"""

from __future__ import annotations

import contextlib
import dataclasses
import math
import re
import shutil
import struct
import subprocess
from pathlib import Path

import pytest

from tensorforge import toolchain
from tensorforge.backend.pir import engines
from tensorforge.backend.pir.allocate import MULT, allocate
from tensorforge.backend.pir.asyncmem import schedule_async
from tensorforge.backend.pir.build import IRBuilder
from tensorforge.backend.pir.core import (BOOL, INDEX, IRError, MemSpace, Op,
                                          Participants, XorSwizzle, walk)
from tensorforge.backend.pir.emit import Emitter
from tensorforge.backend.pir.engines import (Completion, FenceKind,
                                             MatrixDescType, TensorMap)
from tensorforge.backend.pir.passes import verify
from tensorforge.common.basic_types import (RUNTIME, Addressing, Datatype,
                                            ElementAddressing)
from tensorforge.common.context import Context
from tensorforge.generators.kernel_params import TensorMapParam
from tensorforge.reference import kernel_eval

INCLUDE = Path(__file__).resolve().parents[1] / 'src' / 'tensorforge' / 'include'

F32 = Datatype.F32


def _ctx(arch='sm_90', backend='cuda'):
    return Context(arch=arch, backend=backend, fp_type=F32)


class _Sink:
    def __init__(self):
        self.lines = []

    def __call__(self, text, *args, **kwargs):
        self.lines.append(text)

    def __getattr__(self, _):
        return lambda *a, **k: None

    @contextlib.contextmanager
    def _block(self, head):
        self.lines.append(f'{head} {{')
        yield
        self.lines.append('}')

    def If(self, cond):
        return self._block(f'if ({cond})')

    def Else(self):
        return self._block('else')

    def Block(self, head=''):
        return self._block(head)


def _emit(body, ctx=None):
    sink = _Sink()
    Emitter(sink, ctx if ctx is not None else _ctx()).run(body)
    return sink.lines


@dataclasses.dataclass
class Case:
    """A body of one block of 128 threads that copies boxes of the tensor
    `m1` -- `dims`, outermost first, `strides` apart -- one after the other
    into the shared buffer `sa`, through the map `tmA`, each from its
    `origins` entry on and waiting for the cursor's phase, and has every
    thread read its elements of the box from `sa` -- as TensorForge reads a
    buffer of the map's swizzle -- and store them, box after box, to `m0`.
    `computed` makes the coordinates values rather than literals."""
    dims: tuple
    box: tuple
    origins: tuple
    swizzle: int = 0
    fill: str = 'zero'
    strides: tuple = None
    computed: bool = False

    @property
    def tmap(self):
        return TensorMap('tmA', F32, self.dims, self.box, strides=self.strides,
                         swizzle=self.swizzle, fill=self.fill)

    @property
    def volume(self):
        return math.prod(self.box)

    @property
    def extent(self):
        """Elements of `m1`: past the last element of the tensor."""
        strides = self.tmap.element_strides()
        return 1 + sum((n - 1) * t for n, t in zip(self.dims, strides))


CASES = {
    'plain': Case((64, 64), (8, 32), ((3, 4),)),
    'plain-narrow': Case((64, 64), (16, 4), ((5, 8),)),
    'sw128': Case((64, 64), (8, 32), ((8, 32),), swizzle=128),
    'sw64': Case((64, 64), (16, 16), ((16, 4),), swizzle=64),
    'sw32': Case((64, 64), (16, 8), ((1, 8),), swizzle=32),
    'sw128-narrow': Case((64, 64), (16, 16), ((0, 0),), swizzle=128),
    'past-the-end': Case((64, 64), (8, 32), ((60, 48),)),
    'before-the-start-nan': Case((64, 64), (8, 32), ((-2, -4),), fill='nan'),
    '3d': Case((16, 16, 64), (3, 4, 16), ((2, 4, 16),)),
    'sw128-3d': Case((16, 32, 32), (4, 4, 32), ((1, 8, 0),), swizzle=128),
    'two-phases': Case((64, 64), (8, 32), ((0, 0), (8, 32)), swizzle=128),
    'computed': Case((64, 64), (8, 16), ((4, 16), (12, 0)), swizzle=64,
                     computed=True),
    'batch': Case((4, 9, 56), (2, 9, 32), ((1, 0, 24),), strides=(512, 56)),
}


def _body(case: Case, ctx=None, place=None):
    """`case` built and placed: (context, body, the arena in floats)."""
    ctx = ctx if ctx is not None else _ctx()
    tmap = case.tmap
    b = IRBuilder(fptype=F32, arena='shrMem', context=ctx)
    form = engines.swizzle_form(tmap.swizzle, F32)
    align = max(engines.TMA_SHARED_ALIGN,
                engines.swizzle_alignment(tmap.swizzle)) // 4
    sa = b.alloc(F32, (tmap.shared_bytes() // 4,), MemSpace.SHARED, hint='sa',
                 extern='sa', place_align=place or align, swizzle=form)
    src = b.alloc(F32, (case.extent,), MemSpace.GLOBAL, hint='g',
                  extern='m1')
    out = b.alloc(F32, (case.volume * len(case.origins),), MemSpace.GLOBAL,
                  hint='o', extern='m0')
    cursor = b.bulk_channel('tma', 'threadIdx.x == 0',
                            participants=Participants.BLOCK)
    lane = b.thread_id('x')
    inner, pitch = case.box[-1], tmap.pitch() // 4
    for p, origin in enumerate(case.origins):
        if p:
            # The reads of the last box, ahead of the copy over it.
            b.barrier('block')
        coords = [b.op('add', INDEX, b.thread_id('y'), c) if case.computed
                  else c for c in origin]
        b.wait(b.copy_tma(sa, src, tmap, coords, barrier=cursor))
        for j in range((case.volume + 127) // 128):
            k = b.op('add', INDEX, lane, 128 * j) if j else lane
            at = b.op('add', INDEX,
                      b.op('mul', INDEX, b.op('div', INDEX, k, inner), pitch),
                      b.op('rem', INDEX, k, inner), hint='at')

            def move():
                b.store(out, b.load(sa, at),
                        b.op('add', INDEX, k, p * case.volume))
            if case.volume % 128 and j == case.volume // 128:
                with b.if_(b.op('lt', BOOL, k, case.volume)):
                    move()
            else:
                move()
    body, layout = allocate(b.finish(), arenas={'shrMem': MULT})
    return ctx, body, layout.per_mult


def _expected(case, values):
    """The boxes as the hardware fills them, element after element."""
    out = []
    strides = case.tmap.element_strides()
    for origin in case.origins:
        for k in range(case.volume):
            at, rest = [], k
            for n in reversed(case.box):
                at.append(rest % n)
                rest //= n
            where = [o + i for o, i in zip(origin, reversed(at))]
            if all(0 <= w < n for w, n in zip(where, case.dims)):
                out.append(values[sum(w * t for w, t in zip(where, strides))])
            else:
                out.append(math.nan if case.fill == 'nan' else 0.0)
    return out


def _same(got, want):
    return [(i, g, w) for i, (g, w) in enumerate(zip(got, want))
            if not (g == w or (math.isnan(g) and math.isnan(w)))]


def _oracle_source(lines):
    keep = [line for line in lines if not re.match(r'^float m\d+\[', line)]
    return ('void k() {\n'
            '  auto* shrMem = reinterpret_cast<float*>(totalShrMemPtr);\n'
            + ''.join(f'  {line}\n' for line in keep) + '}')


def _run_oracle(case, src=None, seed=5, races=None):
    ctx, body, _ = _body(case)
    if src is None:
        src = _oracle_source(_emit(body, ctx))
    arg = kernel_eval.TensorMapArg.of(case.tmap, 'm1')
    mem = kernel_eval.evaluate_wave(src, 128, seed=seed, globals_only=True,
                                    scalars={'tmA': arg}, races=races)
    slot = kernel_eval.Slot(seed)
    values = [slot.read('m1', i) for i in range(case.extent)]
    count = case.volume * len(case.origins)
    return [mem[('m0', i)] for i in range(count)], _expected(case, values)


# -- the vocabulary ------------------------------------------------------------ #

def test_the_engine_completes_on_a_barrier_as_the_bulk_copy():
    tma = engines.TRANSFER_ENGINES['tma']
    assert tma.ops == (Op.COPY_ASYNC,)
    assert tma.completes is Completion.BARRIER
    assert engines.counter_of('tma') == engines.counter_of('bulk') == 'bulk'


@pytest.mark.parametrize('arch,backend,has', [
    ('sm_90', 'cuda', True), ('sm_90a', 'cuda', True),
    ('sm_100', 'cuda', True), ('sm_100a', 'cuda', True),
    ('sm_120', 'cuda', True), ('sm_80', 'cuda', False),
    ('sm_86', 'cuda', False), ('sm_90', 'hip', False)])
def test_the_targets_with_tensor_maps(arch, backend, has):
    assert _ctx(arch, backend).target.tensor_maps() is has


@pytest.mark.parametrize('elem', [Datatype.F32, Datatype.F64])
@pytest.mark.parametrize('swizzle', [32, 64, 128])
def test_the_swizzle_forms_are_the_hardwares_address_function(swizzle, elem):
    """A buffer of the form, read element by element, reads where NVIDIA's
    swizzle puts the bytes: the chunk XORed with the 128-byte line."""
    form = engines.swizzle_form(swizzle, elem)
    size = elem.size()
    for i in range(8192 // size):
        assert form.apply(i) * size == engines.swizzled(i * size, swizzle)
    assert engines.swizzle_bytes(form, elem) == swizzle
    assert engines.swizzle_bytes(None, elem) == 0


def test_the_128_byte_swizzle_is_tensorforges_own():
    assert engines.swizzle_form(128, Datatype.F32) == XorSwizzle(8, 4)
    assert engines.swizzle_form(128, Datatype.F64) == XorSwizzle(8, 2)
    assert engines.swizzle_bytes(XorSwizzle(8, 4), Datatype.F32) == 128
    # TensorForge's other swizzles are no layout of NVIDIA's.
    assert engines.swizzle_bytes(XorSwizzle(4, 4), Datatype.F32) is None
    assert engines.swizzle_bytes(XorSwizzle(32), Datatype.F32) is None
    assert engines.swizzle_form(0, Datatype.F32) is None


def test_a_swizzle_row_longer_than_its_width():
    assert XorSwizzle(8, 4, line=8) == XorSwizzle(8, 4)
    assert XorSwizzle(8, 4, line=8).line is None
    assert repr(XorSwizzle(4, 4, line=8)) == 'xor4g4l8'
    assert repr(XorSwizzle(2, line=8)) == 'xor2l8'
    assert XorSwizzle(4, 4, line=8).span == 8
    for line in (2, 6):
        with pytest.raises(IRError):
            XorSwizzle(4, 4, line=line)


@pytest.mark.parametrize('swizzle', [32, 64, 128])
def test_the_builder_addresses_a_buffer_of_the_form_as_it_permutes(swizzle):
    """Every lane stores its number at its own index of a buffer of the
    form: the arena holds it where the form puts the index."""
    form = engines.swizzle_form(swizzle, F32)
    b = IRBuilder(fptype=F32, arena='shrMem', context=_ctx())
    sa = b.alloc(F32, (512,), MemSpace.SHARED, hint='sa', extern='sa',
                 place_align=256, swizzle=form)
    lane = b.thread_id('x')
    for j in range(4):
        k = b.op('add', INDEX, lane, 128 * j) if j else lane
        b.store(sa, b.op('add', INDEX, k, 1000), k)
    body, _ = allocate(b.finish(), arenas={'shrMem': MULT})
    mem = kernel_eval.evaluate_wave(_oracle_source(_emit(body)), 128, seed=1)
    base = next(s.attr('offset') for s, _ in walk(body) if s.op == Op.ALLOC
                and s.attr('extern') == 'sa')
    for i in range(512):
        assert mem[('shr', base + form.apply(i))] == 1000 + i


# -- the map ----------------------------------------------------------------- #

def test_the_facts_of_a_map():
    m = TensorMap('tmA', F32, (None, 9, 56), (4, 9, 32), swizzle=128)
    assert m.rank == 3
    assert m.element_strides() == (504, 56, 1)
    assert m.box_bytes() == 4 * 9 * 32 * 4
    assert m.pitch() == 128 and m.rows() == 36
    assert m.shared_bytes() == 36 * 128
    assert m.shared_index(1, 2, 3) == (9 + 2) * 32 + 3
    narrow = TensorMap('tmB', F32, (64, 64), (16, 16), swizzle=128)
    assert narrow.pitch() == 128 and narrow.shared_index(1, 3) == 35
    plain = TensorMap('tmC', Datatype.F64, (64, 64), (16, 6),
                      strides=(None,))
    assert plain.pitch() == 48 and plain.element_strides() == (None, 1)


@pytest.mark.parametrize('kwargs,why', [
    (dict(dims=(64,), box=(16,)), 'dimensions'),
    (dict(dims=(4, 4, 4, 64), box=(1, 1, 1, 16)), 'dimensions'),
    (dict(dims=(64, 64), box=(0, 16)), 'box extent'),
    (dict(dims=(64, 64), box=(257, 16)), 'box extent'),
    (dict(dims=(64, 64), box=(8, 3)), 'no multiple of 16'),
    (dict(dims=(64, 128), box=(8, 64), swizzle=128), 'wider than'),
    (dict(dims=(64, 64), box=(8, 16), swizzle=48), 'swizzle'),
    (dict(dims=(64, 64), box=(8, 16), strides=(9,)), 'stride of 36'),
    (dict(dims=(64, None), box=(8, 16)), 'dense'),
    (dict(dims=(64, 64), box=(8, 16), fill='inf'), 'fill'),
    (dict(dims=(64, 64), box=(8, 16), elem=Datatype.F16), 'spells'),
    (dict(dims=(64, 64), box=(8, 16), name='1x'), 'identifier'),
    (dict(dims=(64, 64), box=(8, 16), strides=(64, 1)), 'innermost'),
])
def test_a_map_the_driver_does_not_encode_is_refused(kwargs, why):
    args = dict(name='tmA', elem=F32)
    args.update(kwargs)
    with pytest.raises(IRError, match=why):
        TensorMap(**args)


# -- the copy ---------------------------------------------------------------- #

def _refused_body(*, swizzle=128, buffer_swizzle=True, place=None, coords=2,
                  start=0, length=None, space=MemSpace.SHARED, inner=0):
    tmap = TensorMap('tmA', F32, (64, 64), (8, 32), swizzle=swizzle)
    b = IRBuilder(fptype=F32, arena='shrMem', context=_ctx())
    form = engines.swizzle_form(swizzle, F32) if buffer_swizzle else None
    align = max(128, engines.swizzle_alignment(swizzle)) // 4
    sa = b.alloc(F32, (length or 256,), space, hint='sa', extern='sa',
                 place_align=place or align, swizzle=form)
    src = b.alloc(F32, (4096,), MemSpace.GLOBAL, hint='g', extern='m1')
    cursor = b.bulk_channel('tma', 'threadIdx.x == 0',
                            participants=Participants.BLOCK)
    return b.copy_tma(sa, src, tmap, (0,) * (coords - 1) + (inner,),
                      barrier=cursor, dst_index=start)


@pytest.mark.parametrize('kwargs,why', [
    (dict(place=128), 'not placed on 256'),
    (dict(buffer_swizzle=False), 'lies as xor8g4'),
    (dict(swizzle=0, place=32), 'lies as None'),
    (dict(coords=3), '3 coordinates'),
    (dict(start=4, length=512), 'off 128'),
    (dict(start=32, length=256), 'past the 256'),
    (dict(length=200), 'past the 200'),
    (dict(inner=2), '8 bytes into a row'),
])
def test_a_copy_the_accelerator_cannot_make_is_refused(kwargs, why):
    if kwargs.get('swizzle') == 0:
        kwargs['buffer_swizzle'] = True
        # A plain box into a buffer of TensorForge's own swizzle.
        tmap = TensorMap('tmA', F32, (64, 64), (8, 32))
        b = IRBuilder(fptype=F32, arena='shrMem', context=_ctx())
        sa = b.alloc(F32, (256,), MemSpace.SHARED, hint='sa', extern='sa',
                     place_align=32, swizzle=XorSwizzle(8, 4))
        src = b.alloc(F32, (4096,), MemSpace.GLOBAL, hint='g', extern='m1')
        cursor = b.bulk_channel('tma', 'threadIdx.x == 0',
                                participants=Participants.BLOCK)
        with pytest.raises(IRError, match='lies as None'):
            b.copy_tma(sa, src, tmap, (0, 0), barrier=cursor)
        return
    with pytest.raises(IRError, match=why):
        _refused_body(**kwargs)


def test_a_copy_through_no_map_or_into_no_buffer_is_refused():
    b = IRBuilder(fptype=F32, arena='shrMem', context=_ctx())
    src = b.alloc(F32, (4096,), MemSpace.GLOBAL, hint='g', extern='m1')
    cursor = b.bulk_channel('tma', 'threadIdx.x == 0',
                            participants=Participants.BLOCK)
    with pytest.raises(IRError, match='no tensor map'):
        b.copy_tma(src, src, 'tmA', (0, 0), barrier=cursor)
    tmap = TensorMap('tmA', F32, (64, 64), (8, 32))
    with pytest.raises(IRError, match='no buffer'):
        b.copy_tma(1, src, tmap, (0, 0), barrier=cursor)


@pytest.mark.parametrize('name', sorted(CASES))
def test_every_body_verifies(name):
    verify(_body(CASES[name])[1])


def test_the_copy_is_a_bulk_copy_of_the_box():
    ctx, body, _ = _body(CASES['batch'])
    (copy,) = [s for s, _ in walk(body) if s.op == Op.COPY_ASYNC]
    tmap = CASES['batch'].tmap
    assert copy.attr('engine') == 'tma' and copy.counter == 'bulk'
    assert copy.attr('map') == tmap and copy.attr('shape') == (2, 9, 32)
    assert copy.attr('width') == tmap.box_bytes() == 2 * 9 * 32 * 4
    assert copy.copy_src_index == (1, 0, 24)
    assert engines.tma_refusal(copy) is None
    (wait,) = [s for s, _ in walk(body) if s.op == Op.WAIT_PHASE]
    assert wait.args == (copy.target[0],)


def test_the_text_of_a_copy():
    """The box through the map's address, the bytes of the box, the
    coordinates innermost first; the wait the cursor's."""
    lines = _emit(_body(CASES['3d'])[1])
    copy = [line for line in lines if 'tmaLoad' in line]
    assert copy == ['tensorforge::tmaLoad(tma_bulk, &sa[0], &tmA, 768, 16, '
                    '4, 2);']
    assert 'tensorforge::bulk_wait(tma_bulk);' in lines
    computed = [line for line in _emit(_body(CASES['computed'])[1])
                if 'tmaLoad' in line]
    assert computed == [
        'tensorforge::tmaLoad(tma_bulk, &sa[0], &tmA, 512, (threadIdx.y + 16), '
        '(threadIdx.y + 4));',
        'tensorforge::tmaLoad(tma_bulk, &sa[0], &tmA, 512, (threadIdx.y + 0), '
        '(threadIdx.y + 12));']


@pytest.mark.parametrize('arch,backend', [('sm_80', 'cuda'),
                                          ('sm_90', 'hip')])
def test_a_target_without_the_accelerator_refuses_the_copy(arch, backend):
    _, body, _ = _body(CASES['plain'])
    with pytest.raises(IRError, match='tensor_maps'):
        _emit(body, _ctx(arch, backend))


def test_a_descriptor_reads_a_buffer_in_its_swizzle_alone():
    """One vocabulary: a matrix descriptor over a buffer TensorForge
    addresses through a swizzle has to state that swizzle."""
    sw128 = MatrixDescType(Datatype.TF32, swizzle=128, lbo=16, sbo=1024)
    for form, ok in ((engines.swizzle_form(128, F32), True),
                     (None, True),
                     (engines.swizzle_form(64, F32), False),
                     (XorSwizzle(32), False)):
        b = IRBuilder(fptype=F32, arena='shrMem', context=_ctx('sm_90a'))
        sb = b.alloc(F32, (512,), MemSpace.SHARED, hint='sb', extern='sb',
                     place_align=256, swizzle=form)
        if ok:
            b.matrix_desc(sb, 0, desc=sw128)
        else:
            with pytest.raises(IRError, match='laid out as'):
                b.matrix_desc(sb, 0, desc=sw128)


# -- the host oracle ------------------------------------------------------------ #

@pytest.mark.parametrize('name', sorted(CASES))
def test_the_oracle_copies_boxes_as_the_hardware_lays_them_out(name):
    got, want = _run_oracle(CASES[name])
    bad = _same(got, want)
    assert not bad, bad[:6]


def test_a_box_read_ahead_of_its_phase_is_a_race():
    ctx, body, _ = _body(CASES['plain'])
    lines = _emit(body, ctx)
    wait = lines.index('tensorforge::bulk_wait(tma_bulk);')
    # The wait moved past the first reads.
    first = next(i for i, line in enumerate(lines) if i > wait
                 and 'sa[' in line)
    moved = lines[:wait] + lines[wait + 1:first + 1] + [lines[wait]] \
        + lines[first + 1:]
    races = []
    _run_oracle(CASES['plain'], _oracle_source(moved), races=races)
    assert any(r.kind == 'FLIGHT' for r in races)
    races = []
    _run_oracle(CASES['plain'], races=races)
    assert not races


def test_announcing_other_bytes_than_the_box_is_refused():
    lines = _emit(_body(CASES['plain'])[1])
    lines = [line.replace('&tmA, 1024,', '&tmA, 512,') for line in lines]
    with pytest.raises(kernel_eval.Abort, match='would not complete'):
        _run_oracle(CASES['plain'], _oracle_source(lines))


def test_a_box_off_16_bytes_along_its_rows_is_refused():
    """A coordinate the copy computes is the caller's to keep on 16 bytes;
    the oracle refuses one that is not, as the hardware does."""
    lines = _emit(_body(CASES['computed'])[1])
    lines = [line.replace('(threadIdx.y + 16)', '(threadIdx.y + 17)')
             for line in lines]
    with pytest.raises(kernel_eval.Abort, match='illegal instruction'):
        _run_oracle(CASES['computed'], _oracle_source(lines))


def test_a_box_off_128_bytes_is_refused():
    lines = _emit(_body(CASES['plain'])[1])
    lines = [line.replace('&sa[0]', '&sa[4]') for line in lines]
    with pytest.raises(kernel_eval.Abort, match='off 128'):
        _run_oracle(CASES['plain'], _oracle_source(lines))


def test_a_box_off_the_swizzles_period_lands_where_the_hardware_puts_it():
    """The swizzle is the shared address's: a buffer placed on 128 bytes but
    off the 1024 of a 128-byte swizzle reads the box permuted otherwise
    (the RTX 5070 does the same, `~/tf-probe/engines3b/probe/tma_probe.cu`)."""
    case = CASES['sw128']
    ctx, body, _ = _body(case)
    lines = _emit(body, ctx)
    window = [line for line in lines if re.match(r'float ?\* ?sa = ', line)]
    assert len(window) == 1
    offset = int(re.search(r'shrMem\[(\d+)\]', window[0]).group(1))
    assert offset % 256 == 0
    moved = [line.replace(f'shrMem[{offset}]', f'shrMem[{offset + 32}]')
             if line == window[0] else line for line in lines]
    got, want = _run_oracle(case, _oracle_source(moved))
    assert _same(got, want)


# -- the parameter ------------------------------------------------------------- #

def test_the_parameter_is_the_kernels_alone_and_encoded_by_the_launcher():
    lexic = _ctx().target.lexic
    tmap = TensorMap('tmA', F32, (None, 9, 56), (4, 9, 32), swizzle=128)
    p = TensorMapParam.of(tmap, 'A + extraOffset_A',
                          dims={0: 'numElements0'})
    assert p.declaration(lexic) == '__grid_constant__ const CUtensorMap tmA'
    assert p.declaration(lexic, host=True) is None
    assert p.argument() is None and p.argument(lexic) == 'tmAArg'
    assert p.binding(lexic) == (
        'CUtensorMap tmAArg = tensorforge::tensorMapTiled<float, 3>('
        'A + extraOffset_A, {static_cast<unsigned long long>(numElements0), '
        '9ull, 56ull}, {504ull, 56ull}, {4u, 9u, 32u}, 128);')
    runtime = TensorMap('tmB', Datatype.F64, (None, 9, 56), (1, 9, 56),
                        strides=(None, 56), fill='nan')
    q = TensorMapParam.of(runtime, 'B', dims={0: 'n'}, strides={0: 'B_stride'})
    assert q.binding(lexic) == (
        'CUtensorMap tmBArg = tensorforge::tensorMapTiled<double, 3>('
        'B, {static_cast<unsigned long long>(n), 9ull, 56ull}, '
        '{static_cast<unsigned long long>(B_stride), 56ull}, {1u, 9u, 56u}, '
        '0, true);')
    with pytest.raises(ValueError, match='dimension 0'):
        TensorMapParam.of(tmap, 'A')
    with pytest.raises(ValueError, match='stride of dimension 0'):
        TensorMapParam.of(runtime, 'B', dims={0: 'n'})


# -- which engine stages a batch ------------------------------------------------ #

def _staging(arch, addressing, *, stride=504 * 4, alignment=16, row=56 * 4,
             window=9 * 56 * 4, contiguous=True):
    return engines.staging_engine(_ctx(arch).target, addressing,
                                  stride_bytes=stride, alignment=alignment,
                                  row_bytes=row, window_bytes=window,
                                  contiguous=contiguous)


@pytest.mark.parametrize('addressing', [
    Addressing.NONE, Addressing.STRIDED, Addressing.INDEXED,
    ElementAddressing(Addressing.STRIDED, stride=512),
    ElementAddressing(Addressing.INDEXED, stride=512, offset=8)])
@pytest.mark.parametrize('arch', ['sm_90', 'sm_100a', 'sm_120'])
def test_a_batch_at_one_stride_goes_through_a_map(addressing, arch):
    assert _staging(arch, addressing) == ('tma', ())


@pytest.mark.parametrize('addressing', [
    Addressing.PTR_BASED,
    ElementAddressing(Addressing.INDEXED_POINTER, slots=4, slot=1)])
def test_an_array_of_pointers_falls_back_to_a_bulk_copy_an_element(
        addressing):
    engine, why = _staging('sm_90', addressing)
    assert engine == 'bulk'
    assert why == ("an element's data is reached through a pointer of its "
                   'own, at no stride from the last the launcher knows',)
    # A window the bulk copy cannot move whole: every lane's few bytes.
    engine, why = _staging('sm_90', addressing, window=9 * 4)
    assert engine == 'cp_async' and len(why) == 2 and 'off 16' in why[1]
    engine, why = _staging('sm_90', addressing, contiguous=False)
    assert engine == 'cp_async' and 'one run' in why[1]


@pytest.mark.parametrize('kwargs,why', [
    (dict(addressing=ElementAddressing(Addressing.STRIDED, stride=RUNTIME)),
     'bound at run time'),
    (dict(alignment=8), 'promises 8 bytes'),
    (dict(row=9 * 4), 'rows of 36 bytes'),
    (dict(stride=9 * 4), 'elements 36 bytes apart'),
    (dict(addressing=Addressing.SCALAR), 'no tensor'),
])
def test_what_a_map_cannot_state_falls_back(kwargs, why):
    addressing = kwargs.pop('addressing', Addressing.STRIDED)
    engine, reasons = _staging('sm_90', addressing, **kwargs)
    assert engine in ('bulk', 'cp_async') and why in reasons[0]


@pytest.mark.parametrize('arch,want', [('sm_80', 'cp_async'),
                                       ('sm_86', 'cp_async')])
def test_without_the_accelerator_the_copy_engine_stages(arch, want):
    engine, why = _staging(arch, Addressing.STRIDED)
    assert engine == want
    assert why == ('the target has no tensor maps',
                   'the target has no bulk copy')


# -- the reduction of a box into the tensor -------------------------------------- #

@dataclasses.dataclass
class Reduce:
    """A body of one block of 128 threads that adds boxes into the tensor
    `m0` -- `dims`, outermost first, `strides` apart -- through the map
    `tmA`: for each of `origins`, every thread writes its elements of the
    box, `m2`'s box after box, to the shared buffer `sa` as TensorForge
    writes a buffer of the map's swizzle and fences them for the
    accelerator; behind a barrier the first thread adds the box into the
    tensor, every thread waits for its bulk groups, and a barrier stands
    ahead of the next box's writes."""
    dims: tuple
    box: tuple
    origins: tuple
    swizzle: int = 0
    strides: tuple = None

    @property
    def tmap(self):
        return TensorMap('tmA', F32, self.dims, self.box, strides=self.strides,
                         swizzle=self.swizzle)

    @property
    def volume(self):
        return math.prod(self.box)

    @property
    def extent(self):
        strides = self.tmap.element_strides()
        return 1 + sum((n - 1) * t for n, t in zip(self.dims, strides))


REDUCES = {
    'plain': Reduce((64, 64), (8, 32), ((3, 4),)),
    'sw128': Reduce((64, 64), (8, 32), ((8, 32),), swizzle=128),
    'sw32': Reduce((64, 64), (16, 8), ((1, 8),), swizzle=32),
    'sw128-narrow': Reduce((64, 64), (16, 16), ((0, 16),), swizzle=128),
    'past-the-end': Reduce((64, 64), (8, 32), ((60, 48),)),
    '3d': Reduce((16, 16, 64), (3, 4, 16), ((2, 4, 16),)),
    'overlapping': Reduce((64, 64), (8, 32), ((0, 0), (4, 16)), swizzle=128),
    'batch': Reduce((4, 9, 56), (2, 9, 32), ((1, 0, 24),), strides=(512, 56)),
}


def _reduce_body(case: Reduce, ctx=None):
    ctx = ctx if ctx is not None else _ctx()
    tmap = case.tmap
    b = IRBuilder(fptype=F32, arena='shrMem', context=ctx)
    form = engines.swizzle_form(tmap.swizzle, F32)
    align = max(engines.TMA_SHARED_ALIGN,
                engines.swizzle_alignment(tmap.swizzle)) // 4
    sa = b.alloc(F32, (tmap.shared_bytes() // 4,), MemSpace.SHARED, hint='sa',
                 extern='sa', place_align=align, swizzle=form)
    values = b.alloc(F32, (case.volume * len(case.origins),),
                     MemSpace.GLOBAL, hint='v', extern='m2')
    out = b.alloc(F32, (case.extent,), MemSpace.GLOBAL, hint='o',
                  extern='m0')
    lane = b.thread_id('x')
    issue = b.op('eq', BOOL, lane, 0, hint='issue')
    inner, pitch = case.box[-1], tmap.pitch() // 4
    for p, origin in enumerate(case.origins):
        if p:
            # The last box read, ahead of the writes over it.
            b.barrier('block')
        for j in range((case.volume + 127) // 128):
            k = b.op('add', INDEX, lane, 128 * j) if j else lane
            at = b.op('add', INDEX,
                      b.op('mul', INDEX, b.op('div', INDEX, k, inner), pitch),
                      b.op('rem', INDEX, k, inner), hint='at')

            def move():
                b.store(sa, b.load(values, b.op('add', INDEX, k,
                                                p * case.volume)), at)
            if case.volume % 128 and j == case.volume // 128:
                with b.if_(b.op('lt', BOOL, k, case.volume)):
                    move()
            else:
                move()
        b.fence(FenceKind.PROXY)
        b.barrier('block')
        b.wait(b.reduce_tma(out, sa, tmap, origin, issue=issue))
    body, layout = allocate(b.finish(), arenas={'shrMem': MULT})
    return ctx, body, layout.per_mult


def _reduced(case, initial, values):
    """The tensor with the boxes added, one after the other."""
    out = list(initial)
    strides = case.tmap.element_strides()
    for p, origin in enumerate(case.origins):
        for k in range(case.volume):
            at, rest = [], k
            for n in reversed(case.box):
                at.append(rest % n)
                rest //= n
            where = [o + i for o, i in zip(origin, reversed(at))]
            if all(0 <= w < n for w, n in zip(where, case.dims)):
                i = sum(w * t for w, t in zip(where, strides))
                out[i] = out[i] + values[p * case.volume + k]
    return out


def _run_reduce_oracle(case, src=None, seed=7, races=None):
    ctx, body, _ = _reduce_body(case)
    if src is None:
        src = _oracle_source(_emit(body, ctx))
    arg = kernel_eval.TensorMapArg.of(case.tmap, 'm0')
    mem = kernel_eval.evaluate_wave(src, 128, seed=seed, globals_only=True,
                                    scalars={'tmA': arg}, races=races)
    slot = kernel_eval.Slot(seed)
    initial = [slot.read('m0', i) for i in range(case.extent)]
    values = [slot.read('m2', i)
              for i in range(case.volume * len(case.origins))]
    got = [mem.get(('m0', i), initial[i]) for i in range(case.extent)]
    return got, _reduced(case, initial, values)


def test_the_reduction_is_counted_in_the_issuing_threads_bulk_groups():
    engine = engines.TRANSFER_ENGINES['tma_reduce']
    assert engine.completes is Completion.COUNTER
    assert engine.waited_on == 'bulk_group' and engine.one_to_one
    _, body, _ = _reduce_body(REDUCES['batch'])
    (copy,) = [s for s, _ in walk(body) if s.op == Op.COPY_ASYNC]
    assert copy.attr('engine') == 'tma_reduce' and copy.predicate is not None
    assert copy.copy_dst_index == (1, 0, 24) and copy.copy_src_index == (0,)
    assert engines.tma_refusal(copy) is None
    (wait,) = [s for s, _ in walk(body) if s.op == Op.WAIT_COUNTER]
    assert wait.attr('counter') == 'bulk_group' and wait.args == copy.target
    verify(body)


def test_the_text_of_a_reduction():
    lines = _emit(_reduce_body(REDUCES['3d'])[1])
    at = lines.index('tensorforge::tmaReduceAdd(&tmA, &sa[0], 16, 4, 2);')
    assert re.fullmatch(r'if \(\w+\) \{', lines[at - 1]) and lines[at + 1] == '}'
    assert 'tensorforge::bulkWaitRead<0>();' in lines[at + 2:]
    assert 'tensorforge::fenceProxyAsync();' in lines[:at]


def test_a_reduction_every_thread_issues_or_of_doubles_is_refused():
    tmap = TensorMap('tmA', F32, (64, 64), (8, 32))
    b = IRBuilder(fptype=F32, arena='shrMem', context=_ctx())
    sa = b.alloc(F32, (256,), MemSpace.SHARED, hint='sa', extern='sa',
                 place_align=32)
    out = b.alloc(F32, (4096,), MemSpace.GLOBAL, hint='o', extern='m0')
    with pytest.raises(IRError, match='every thread would add'):
        b.reduce_tma(out, sa, tmap, (0, 0), issue=None)
    issue = b.op('eq', BOOL, b.thread_id('x'), 0)
    with pytest.raises(IRError, match='not placed on 32'):
        sb = b.alloc(F32, (256,), MemSpace.SHARED, hint='sb', extern='sb',
                     place_align=4)
        b.reduce_tma(out, sb, tmap, (0, 0), issue=issue)
    doubles = TensorMap('tmD', Datatype.F64, (64, 64), (8, 16))
    sd = b.alloc(Datatype.F64, (128,), MemSpace.SHARED, hint='sd',
                 extern='sd', place_align=16)
    with pytest.raises(IRError, match='a reduction of double'):
        b.reduce_tma(out, sd, doubles, (0, 0), issue=issue)


@pytest.mark.parametrize('name', sorted(REDUCES))
def test_the_oracle_adds_the_boxes_where_the_tensor_is(name):
    races = []
    got, want = _run_reduce_oracle(REDUCES[name], races=races)
    bad = _same(got, want)
    assert not bad, bad[:6]
    assert not races


def test_a_reduction_leaves_the_bodys_copies_asynchronous():
    """A reduction is counted in its thread's bulk groups alone: the
    copies of `cp.async` beside it stay asynchronous, the box's bytes being
    no size of theirs."""
    b = IRBuilder(fptype=F32, arena='shrMem', context=_ctx())
    sa = b.alloc(F32, (256,), MemSpace.SHARED, hint='sa', extern='sa',
                 place_align=32)
    sc = b.alloc(F32, (512,), MemSpace.SHARED, hint='sc', extern='sc',
                 place_align=4)
    src = b.alloc(F32, (4096,), MemSpace.GLOBAL, hint='g', extern='m1')
    out = b.alloc(F32, (4096,), MemSpace.GLOBAL, hint='o', extern='m0')
    lane = b.thread_id('x')
    four = b.op('mul', INDEX, lane, 4)
    b.wait(b.copy_async(sc, src, dst_index=(four,), src_index=(four,),
                        elems=4))
    b.store(sa, b.load(sc, lane), lane)
    b.fence(FenceKind.PROXY)
    b.barrier('block')
    issue = b.op('eq', BOOL, lane, 0)
    b.wait(b.reduce_tma(out, sa, REDUCES['plain'].tmap, (0, 0), issue=issue))
    body, _ = allocate(b.finish(), arenas={'shrMem': MULT})
    lines = _emit(body)
    assert any(line.startswith('__pipeline_memcpy_async(') for line in lines)
    assert 'tensorforge::bulkWaitRead<0>();' in lines


def test_a_drain_waits_for_the_reductions_and_the_schedule_keeps_them():
    """A full drain waits on the thread's bulk groups too; the scheduled
    waits of a reduction stay full ones."""
    ctx = _ctx()
    b = IRBuilder(fptype=F32, arena='shrMem', context=ctx)
    sa = b.alloc(F32, (256,), MemSpace.SHARED, hint='sa', extern='sa',
                 place_align=32)
    out = b.alloc(F32, (4096,), MemSpace.GLOBAL, hint='o', extern='m0')
    lane = b.thread_id('x')
    b.store(sa, 1.0, lane)
    b.fence(FenceKind.PROXY)
    b.barrier('block')
    issue = b.op('eq', BOOL, lane, 0)
    b.reduce_tma(out, sa, REDUCES['plain'].tmap, (0, 0), issue=issue)
    b.wait()
    body, _ = allocate(b.finish(), arenas={'shrMem': MULT})
    scheduled, diag = schedule_async(body)
    assert not diag
    assert 'tensorforge::bulkWaitRead<0>();' in _emit(scheduled, ctx)
    scheduled, diag = schedule_async(_reduce_body(REDUCES['overlapping'])[1])
    assert not diag
    assert [line for line in _emit(scheduled, ctx) if 'bulkWait' in line] == [
        'tensorforge::bulkWaitRead<0>();'] * 2


def test_a_box_written_again_before_the_wait_is_a_race():
    """The accelerator reads the box until the issuing thread's wait, a
    barrier between the issue and the writes notwithstanding."""
    case = REDUCES['overlapping']
    lines = _emit(_reduce_body(case)[1])
    first = lines.index('tensorforge::bulkWaitRead<0>();')
    moved = lines[:first] + lines[first + 1:]
    races = []
    _run_reduce_oracle(case, _oracle_source(moved), races=races)
    assert any(r.kind == 'FLIGHT' for r in races)


# -- the build ----------------------------------------------------------------- #

def _cuda_source(case, lines, arena):
    lexic = _ctx().target.lexic
    param = TensorMapParam.of(case.tmap, 'm1')
    keep = [line for line in lines if not re.match(r'^float m\d+\[', line)]
    body = ''.join(f'  {line}\n' for line in keep)
    count = case.volume * len(case.origins)
    return f'''#include <cstdio>
#include <vector>
#include "tensorforge_device/cuda_tma.h"

__global__ void __launch_bounds__(128) tma_case(float *__restrict__ m0,
    const float *__restrict__ m1, {param.declaration(lexic)}) {{
  extern __shared__ __align__(1024) unsigned char tfRaw[];
  float *shrMem = reinterpret_cast<float *>(tfRaw + ((1024u -
      (static_cast<unsigned>(__cvta_generic_to_shared(tfRaw)) & 1023u)) & 1023u));
{body}}}

#define CHECK(x) do {{ cudaError_t e_ = (x); if (e_ != cudaSuccess) {{ \\
  std::fprintf(stderr, "%s: %s\\n", #x, cudaGetErrorString(e_)); return 1; }} }} while (0)

int main(int argc, char **argv) {{
  const size_t n = {case.extent}, nd = {count};
  std::vector<float> h(n), d(nd);
  FILE *in = std::fopen(argv[1], "rb");
  if (!in || std::fread(h.data(), 4, n, in) != n) return 2;
  std::fclose(in);
  float *m0, *m1;
  CHECK(cudaMalloc(&m0, nd * 4));
  CHECK(cudaMalloc(&m1, n * 4));
  CHECK(cudaMemcpy(m1, h.data(), n * 4, cudaMemcpyHostToDevice));
  CHECK(cudaMemset(m0, 0, nd * 4));
  {param.binding(lexic)}
  const int smem = {arena * 4} + 1024;
  CHECK(cudaFuncSetAttribute(tma_case,
      cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  tma_case<<<1, 128, smem>>>(m0, m1, {param.argument(lexic)});
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());
  CHECK(cudaMemcpy(d.data(), m0, nd * 4, cudaMemcpyDeviceToHost));
  FILE *out = std::fopen(argv[2], "wb");
  if (!out || std::fwrite(d.data(), 4, nd, out) != nd) return 3;
  std::fclose(out);
  return 0;
}}
'''


def _nvcc():
    found = toolchain.COMPILERS['cuda'].find()
    if not found or not shutil.which(found):
        pytest.skip('nvcc not found')
    return found


#: The first CUDA release that builds each arch-specific target: an older
#: toolkit refuses it ("Unsupported gpu architecture"), as x-wing0's CUDA
#: 12.4 refuses sm_100a.
ARCH_RELEASE = {'sm_90a': (12, 0), 'sm_100a': (12, 8)}


def _nvcc_builds(arch):
    """The toolkit's nvcc, or a skip where it builds no `arch`: the base
    target missing from `nvcc --list-gpu-code` (CUDA 12.4 has neither sm_100
    nor sm_120), or a release older than the one that brought the
    arch-specific target (`ARCH_RELEASE`)."""
    nvcc = _nvcc()
    listed = subprocess.run([nvcc, '--list-gpu-code'], capture_output=True,
                            text=True, timeout=60).stdout.split()
    found = re.search(r'release (\d+)\.(\d+)', subprocess.run(
        [nvcc, '--version'], capture_output=True, text=True,
        timeout=60).stdout)
    release = (int(found.group(1)), int(found.group(2))) if found else (0, 0)
    if (arch.rstrip('af') not in listed
            or release < ARCH_RELEASE.get(arch, (0, 0))):
        pytest.skip(f'nvcc {release[0]}.{release[1]} builds no {arch}')
    return nvcc


def _build(case, tmp_path, flags, product='cubin', arch='sm_90'):
    ctx, body, arena = _body(case, _ctx(arch))
    src = tmp_path / 'tma.cu'
    src.write_text(_cuda_source(case, _emit(body, ctx), arena))
    nvcc = toolchain.COMPILERS['cuda']
    out = tmp_path / {'cubin': 'k.cubin', 'ptx': 'k.ptx'}.get(product, 'tma')
    cmd = [_nvcc_builds(arch), *nvcc.language_flags(), *flags,
           '-I', str(INCLUDE), '-o', str(out), str(src)]
    if product in ('cubin', 'ptx'):
        cmd.insert(1, f'-{product}')
    run = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    assert run.returncode == 0, run.stderr[-4000:]
    return out


@pytest.mark.parametrize('arch', ['sm_90', 'sm_100a', 'sm_120'])
@pytest.mark.parametrize('name', ['sw128', '3d', 'two-phases'])
def test_compiled_each_copy_is_one_utmaldg(name, arch, tmp_path):
    case = CASES[name]
    cubin = _build(case, tmp_path, toolchain.COMPILERS['cuda'].target_flags(
        arch), arch=arch)
    tool = shutil.which('cuobjdump')
    if tool is None:
        pytest.skip('cuobjdump not found')
    sass = subprocess.run([tool, '-sass', str(cubin)], capture_output=True,
                          text=True, timeout=600).stdout
    rank = len(case.box)
    assert len(re.findall(rf'\bUTMALDG\.{rank}D\b', sass)) == \
        len(case.origins)


@pytest.mark.parametrize('name', ['sw128', '3d', 'two-phases'])
def test_compiled_the_ptx_copies_the_boxes_the_body_states(name, tmp_path):
    case = CASES[name]
    ptx = _build(case, tmp_path, toolchain.COMPILERS['cuda'].target_flags(
        'sm_90'), product='ptx').read_text()
    rank = len(case.box)
    copies = re.findall(r'cp\.async\.bulk\.tensor\.(\d)d\.shared::cluster\.'
                        r'global\.tile\.mbarrier::complete_tx::bytes', ptx)
    assert copies == [str(rank)] * len(case.origins)
    assert 'trap;' not in ptx


def _reduce_cuda_source(case, lines, arena):
    lexic = _ctx().target.lexic
    param = TensorMapParam.of(case.tmap, 'm0')
    keep = [line for line in lines if not re.match(r'^float m\d+\[', line)]
    body = ''.join(f'  {line}\n' for line in keep)
    count = case.volume * len(case.origins)
    return f'''#include <cstdio>
#include <vector>
#include "tensorforge_device/cuda_tma.h"

__global__ void __launch_bounds__(128) tma_reduce(float *__restrict__ m0,
    const float *__restrict__ m2, {param.declaration(lexic)}) {{
  extern __shared__ __align__(1024) unsigned char tfRaw[];
  float *shrMem = reinterpret_cast<float *>(tfRaw + ((1024u -
      (static_cast<unsigned>(__cvta_generic_to_shared(tfRaw)) & 1023u)) & 1023u));
{body}}}

#define CHECK(x) do {{ cudaError_t e_ = (x); if (e_ != cudaSuccess) {{ \\
  std::fprintf(stderr, "%s: %s\\n", #x, cudaGetErrorString(e_)); return 1; }} }} while (0)

int main(int argc, char **argv) {{
  const size_t n = {case.extent}, nv = {count};
  std::vector<float> h(n + nv);
  FILE *in = std::fopen(argv[1], "rb");
  if (!in || std::fread(h.data(), 4, n + nv, in) != n + nv) return 2;
  std::fclose(in);
  float *m0, *m2;
  CHECK(cudaMalloc(&m0, n * 4));
  CHECK(cudaMalloc(&m2, nv * 4));
  CHECK(cudaMemcpy(m0, h.data(), n * 4, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(m2, h.data() + n, nv * 4, cudaMemcpyHostToDevice));
  {param.binding(lexic)}
  const int smem = {arena * 4} + 1024;
  CHECK(cudaFuncSetAttribute(tma_reduce,
      cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  tma_reduce<<<1, 128, smem>>>(m0, m2, {param.argument(lexic)});
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());
  CHECK(cudaMemcpy(h.data(), m0, n * 4, cudaMemcpyDeviceToHost));
  FILE *out = std::fopen(argv[2], "wb");
  if (!out || std::fwrite(h.data(), 4, n, out) != n) return 3;
  std::fclose(out);
  return 0;
}}
'''


def _build_reduce(case, tmp_path, flags, product='cubin', arch='sm_90'):
    ctx, body, arena = _reduce_body(case, _ctx(arch))
    src = tmp_path / 'tma_reduce.cu'
    src.write_text(_reduce_cuda_source(case, _emit(body, ctx), arena))
    nvcc = toolchain.COMPILERS['cuda']
    out = tmp_path / {'cubin': 'k.cubin', 'ptx': 'k.ptx'}.get(product,
                                                            'tma_reduce')
    cmd = [_nvcc_builds(arch), *nvcc.language_flags(), *flags,
           '-I', str(INCLUDE), '-o', str(out), str(src)]
    if product in ('cubin', 'ptx'):
        cmd.insert(1, f'-{product}')
    run = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    assert run.returncode == 0, run.stderr[-4000:]
    return out


@pytest.mark.parametrize('name', ['sw128', '3d', 'overlapping'])
def test_compiled_each_reduction_is_one_tensor_reduce_and_a_group(name,
                                                                  tmp_path):
    case = REDUCES[name]
    ptx = _build_reduce(case, tmp_path, toolchain.COMPILERS[
        'cuda'].target_flags('sm_90'), product='ptx').read_text()
    rank = len(case.box)
    adds = re.findall(r'cp\.reduce\.async\.bulk\.tensor\.(\d)d\.global\.'
                      r'shared::cta\.add\.tile\.bulk_group', ptx)
    assert adds == [str(rank)] * len(case.origins)
    assert len(re.findall(r'cp\.async\.bulk\.commit_group', ptx)) == \
        len(case.origins)
    assert len(re.findall(r'cp\.async\.bulk\.wait_group\.read 0', ptx)) == \
        len(case.origins)
    assert 'trap;' not in ptx


@pytest.mark.parametrize('arch', ['sm_90', 'sm_120'])
@pytest.mark.parametrize('name', ['3d', 'overlapping'])
def test_compiled_each_reduction_is_one_utmaredg_and_a_flush(name, arch,
                                                             tmp_path):
    case = REDUCES[name]
    cubin = _build_reduce(case, tmp_path, toolchain.COMPILERS[
        'cuda'].target_flags(arch), arch=arch)
    tool = shutil.which('cuobjdump')
    if tool is None:
        pytest.skip('cuobjdump not found')
    sass = subprocess.run([tool, '-sass', str(cubin)], capture_output=True,
                          text=True, timeout=600).stdout
    rank = len(case.box)
    count = len(case.origins)
    assert len(re.findall(rf'\bUTMAREDG\.{rank}D\.ADD\b', sass)) == count
    assert len(re.findall(r'\bUTMACMDFLUSH\b', sass)) == count


# -- on the hardware ---------------------------------------------------------- #

def _gpu():
    tool = shutil.which('nvidia-smi')
    if tool is None:
        return None
    try:
        run = subprocess.run([tool, '--query-gpu=compute_cap',
                              '--format=csv,noheader'], capture_output=True,
                             text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    first = run.stdout.strip().splitlines()[:1]
    if run.returncode or not first:
        return None
    return 'sm_' + first[0].strip().replace('.', '')


def _inputs(case):
    return [float(i % 4093) + 0.25 for i in range(case.extent)]


@pytest.mark.parametrize('name', sorted(CASES))
def test_gpu_the_boxes_land_where_tensorforge_reads_them(name, tmp_path):
    """Built for the GPU present, of sm_90 or newer: the map encoded by the
    launcher's binding, each box copied and read back through the buffer's
    swizzle, against the boxes computed here, to the bit."""
    arch = _gpu()
    if arch is None or int(arch[3:]) < 90:
        pytest.skip('no GPU of sm_90 or newer')
    case = CASES[name]
    exe = _build(case, tmp_path,
                 toolchain.COMPILERS['cuda'].target_flags(arch),
                 product='binary', arch=arch)
    values = _inputs(case)
    (tmp_path / 'in.bin').write_bytes(struct.pack(f'<{len(values)}f',
                                                  *values))
    run = subprocess.run([str(exe), str(tmp_path / 'in.bin'),
                          str(tmp_path / 'out.bin')], capture_output=True,
                         text=True, timeout=120)
    assert run.returncode == 0, run.stderr[-2000:]
    count = case.volume * len(case.origins)
    got = struct.unpack(f'<{count}f', (tmp_path / 'out.bin').read_bytes())
    bad = _same(got, _expected(case, values))
    assert not bad, (f'{len(bad)} of {count} elements differ (index, got, '
                     f'want): {bad[:6]}')


@pytest.mark.parametrize('name', sorted(REDUCES))
def test_gpu_the_boxes_are_added_where_the_tensor_is(name, tmp_path):
    """Built for the GPU present, of sm_90 or newer: each box written as
    TensorForge writes a buffer of the map's swizzle and added into the
    tensor through the map the launcher encoded, against the sums computed
    here -- exact, the numbers being quarters."""
    arch = _gpu()
    if arch is None or int(arch[3:]) < 90:
        pytest.skip('no GPU of sm_90 or newer')
    case = REDUCES[name]
    exe = _build_reduce(case, tmp_path,
                        toolchain.COMPILERS['cuda'].target_flags(arch),
                        product='binary', arch=arch)
    count = case.volume * len(case.origins)
    initial = [float(i % 509) for i in range(case.extent)]
    values = [0.25 * (k % 97) + 1.0 for k in range(count)]
    data = initial + values
    (tmp_path / 'in.bin').write_bytes(struct.pack(f'<{len(data)}f', *data))
    run = subprocess.run([str(exe), str(tmp_path / 'in.bin'),
                          str(tmp_path / 'out.bin')], capture_output=True,
                         text=True, timeout=120)
    assert run.returncode == 0, run.stderr[-2000:]
    got = struct.unpack(f'<{case.extent}f', (tmp_path / 'out.bin').read_bytes())
    bad = _same(got, _reduced(case, initial, values))
    assert not bad, (f'{len(bad)} of {case.extent} elements differ (index, '
                     f'got, want): {bad[:6]}')
