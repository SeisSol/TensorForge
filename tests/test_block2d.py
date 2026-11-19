# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""2D block reads (`pir.block_reads.load_block2d`): one LSC 2D block message
for a block of a surface, which hands each lane of a sub-group of 16 its
column of the block, or its row where the read transposes.

The statement is a `load` with `engine='block'` as a window or a run is, told
apart by its pitch.  Under the explicit vector it is one `load_2d`
(`tensorforge::blockLoad2d`); under SPMD the sub-group's 2D block read
(`tensorforge::subGroupBlock2d`).  Nothing produces one yet: the limits are
pinned here, each with its own answer, and the device code is checked by the
on-slice harness of the item, compiled for PVC and run under the CPU
stand-in.
"""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from tensorforge.backend.pir import block_reads
from tensorforge.backend.pir.build import IRBuilder
from tensorforge.backend.pir.core import (BOOL, INDEX, Effect, IRError,
                                          LaneAxis, MemSpace, Op,
                                          RegisterLayout, ScalarType)
from tensorforge.backend.pir.emit import Emitter
from tensorforge.backend.pir.emit_esimd import EsimdEmitter
from tensorforge.backend.writer import Writer
from tensorforge.common.basic_types import (Addressing, Datatype, indexed,
                                            pointer_based)
from tensorforge.common.context import Context, Options
from harness.placement import placed

F32 = Datatype.F32
F64 = Datatype.F64
SURFACE = dict(width=256, height=64, pitch=256, align=64)
HEADER = (Path(block_reads.__file__).resolve().parents[2]
          / 'include' / 'tensorforge_device' / 'sycl_spmd.h')


def _refusal(elem=F32, rows=8, cols=16, **change):
    args = dict(SURFACE)
    args.update(change)
    return block_reads.block2d_refusal(elem, rows, cols, **args)


# -- the limits ---------------------------------------------------------------- #

@pytest.mark.parametrize('rows,cols,transpose', [
    *[(r, 16, False) for r in block_reads.BLOCK2D_ROWS],
    *[(16, c, True) for c in block_reads.BLOCK2D_COLS],
])
def test_the_shapes_a_lane_reads(rows, cols, transpose):
    """Untransposed a lane's column of 16, 1 to 32 rows of it; transposed a
    lane's row of 16, 1 to 8 columns of it."""
    assert _refusal(rows=rows, cols=cols, transpose=transpose) is None


@pytest.mark.parametrize('elem,rows,cols,change,why', [
    (F32, 8, 16, dict(store=True, transpose=True),
     'no transposed 2D block store'),
    (F32, 8, 16, dict(store=True), 'no 2D block store is built'),
    (F32, 16, 8, dict(transpose=True, table=True),
     'share no pitch'),
    (F32, 8, 16, dict(vnni=True), 'VNNI'),
    (Datatype.F16, 8, 16, {}, 'not 16-bit'),
    (F64, 16, 8, dict(transpose=True), 'transposes at most 8 rows'),
    (F64, 8, 8, dict(transpose=True), 'transposes at most 8 rows'),
    (F64, 8, 16, {}, '128 bytes'),
    (F32, 8, 16, dict(lanes=32), 'not by 32 lanes'),
    (F32, 3, 16, {}, 'untransposed'),
    (F32, 8, 8, {}, 'untransposed'),
    (F32, 64, 16, {}, 'untransposed'),
    (F32, 8, 4, dict(transpose=True), 'transposed'),
    (F32, 16, 16, dict(transpose=True), 'transposed'),
    (F32, 16, 3, dict(transpose=True), 'transposed'),
    (F32, 8, 16, dict(align=None), '64-byte aligned'),
    (F32, 8, 16, dict(align=32), '64-byte aligned'),
    (F32, 8, 16, dict(width=32, pitch=32), 'from 64'),
    (F32, 8, 16, dict(width=(1 << 24) + 4, pitch=(1 << 24) + 16),
     'from 64'),
    (F32, 8, 16, dict(width=66, pitch=80), 'no whole number of 4-byte'),
    (F32, 8, 16, dict(height=0), 'from 1'),
    (F32, 8, 16, dict(height=(1 << 24) + 1), 'from 1'),
    (F32, 8, 16, dict(pitch=128), 'at least the width'),
    (F32, 8, 16, dict(pitch=264), 'a multiple of 16'),
    (F32, 8, 16, dict(width=1 << 24, pitch=(1 << 24) + 16), 'at most 2^24'),
], ids=['transposed-store', 'store', 'pointer-table', 'vnni', 'f16',
        'f64-transposed', 'f64-transposed-8-rows', 'f64', 'lanes',
        'rows-3', 'cols-8', 'rows-64', 'transposed-rows-8',
        'transposed-cols-16', 'transposed-cols-3', 'align-unknown',
        'align-32', 'narrow', 'wide', 'width-not-elements', 'height-0',
        'height-past-2^24', 'pitch-under-width', 'pitch-not-16',
        'pitch-past-2^24'])
def test_each_limit_is_its_own_answer(elem, rows, cols, change, why):
    """No transposed 2D store and no store built; elements reached through
    a table have no pitch; VNNI, 16-bit and 64-bit data; a sub-group other
    than 16; the shapes; and the surface rules of the 2D block messages."""
    answer = _refusal(elem, rows, cols, **change)
    assert answer is not None and why in answer, answer


def _symbol(addressing, name='A'):
    """A global operand as the generator hands one over: a name, its space
    and the tensor, which states how an element reaches its data."""
    return SimpleNamespace(name=name, datatype=F32,
                           stype=SimpleNamespace(name='Global'),
                           obj=SimpleNamespace(addressing=addressing))


@pytest.mark.parametrize('addressing,refused', [
    (Addressing.STRIDED, False),
    (Addressing.PTR_BASED, True),
    (pointer_based(offset=4), True),
    (Addressing.INDEXED, True),
    (indexed(stride=96), True),
    (Addressing.INDEXED_POINTER, True),
], ids=['strided', 'pointer-based', 'pointer-based-offset', 'indexed',
        'indexed-stride', 'indexed-pointer'])
def test_elements_through_a_table_have_no_pitch(addressing, refused):
    """Read as rows of one surface, elements have to be equally strided: a
    table of pointers or of indices would need a check at run time and a
    fallback.  The kind decides, a parameterized form included; a block
    within one element's data takes any."""
    def read(elements):
        b = IRBuilder(fptype=F32)
        return block_reads.load_block2d(
            b, _symbol(addressing), 0, 0, rows=16, cols=8, transpose=True,
            elements=elements, **SURFACE)

    read(elements=False)
    if refused:
        with pytest.raises(IRError, match='share no pitch'):
            read(elements=True)
    else:
        read(elements=True)


def test_the_builder_answers_with_the_limit():
    b = IRBuilder(fptype=F32)
    g = b.alloc(F32, (4096,), MemSpace.GLOBAL, extern='m1')
    with pytest.raises(IRError, match='64-byte aligned'):
        block_reads.load_block2d(b, g, 0, 0, rows=8, cols=16, width=256,
                                 height=64, pitch=256, align=16)
    with pytest.raises(IRError, match='VNNI'):
        block_reads.load_block2d(b, g, 0, 0, rows=8, cols=16, vnni=True,
                                 **SURFACE)


def test_a_surface_is_global_memory():
    b = IRBuilder(fptype=F32, arena='shrMem')
    s = b.alloc(F32, (4096,), MemSpace.SHARED, extern='s1')
    with pytest.raises(IRError, match='reads global memory, not shared'):
        block_reads.load_block2d(b, s, 0, 0, rows=8, cols=16, **SURFACE)


# -- the statement ------------------------------------------------------------- #

def _read(b, rows=8, cols=16, transpose=False, x=0, y=3):
    g = b.alloc(F32, (4096,), MemSpace.GLOBAL, extern='m1')
    return block_reads.load_block2d(b, g, x, y, rows=rows, cols=cols,
                                    transpose=transpose, **SURFACE)


@pytest.mark.parametrize('rows,cols,transpose,count', [
    (8, 16, False, 8), (16, 4, True, 4), (1, 16, False, None),
    (16, 1, True, None),
], ids=['rows', 'transposed', 'one-row', 'one-column'])
def test_a_2d_block_is_one_load_of_the_lanes_share(rows, cols, transpose,
                                                   count):
    """A value over the 16 lanes, a vector of the share -- a number where it
    is one element -- defined by a `load` with `engine='block'` and its
    surface, read from global memory."""
    b = IRBuilder(fptype=F32)
    v = _read(b, rows, cols, transpose)
    s = next(x for x in b.finish() if x.op == Op.LOAD)
    assert s.target == (v,)
    assert v.type == ScalarType(F32, count)
    assert v.layout == RegisterLayout((LaneAxis(16),))
    assert s.attr('engine') == 'block' and s.attr('shape') == (rows, cols)
    assert s.attr('pitch') == 256 and s.attr('surface') == (256, 64)
    assert s.attr('transpose') is transpose and s.attr('vnni') is False
    assert s.attr('width') == 4 * (count or 1)
    assert [(a.kind, a.space) for a in s.accesses] == [
        (Effect.READ, MemSpace.GLOBAL)]
    assert s.args[1:] == (0, 3) and not s.pure and s.predicate is None
    assert block_reads.is_block2d(s) and block_reads.is_block(s)
    assert not block_reads.is_part(s)
    assert block_reads.block2d_geometry(s) == (cols, rows, count or 1)


def test_a_window_or_a_run_is_no_2d_block():
    """The same `load` with `engine='block'`, told apart by the pitch."""
    from test_block_reads import _body, _shared_reads
    out = block_reads.block_reads(_body(_shared_reads))
    heads = [s for s in out if block_reads.is_block(s)]
    assert heads and not any(block_reads.is_block2d(s) for s in heads)


def test_the_planners_leave_a_2d_block_alone():
    """No window, run or sub-group run takes a 2D block as a read of its
    own: its arguments are a corner, not an index."""
    b = IRBuilder(fptype=F32, arena='shrMem')
    _read(b)
    _read(b, rows=16, cols=8, transpose=True)
    body = placed(b.finish())
    assert block_reads.block_reads(body) is body
    stmts = block_reads.walk_stmts(body)
    allocs, names = block_reads.allocations(stmts)
    assert block_reads.plan_spmd_runs(body, allocs, names, 16) == []


# -- under the explicit vector ------------------------------------------------- #

def _esimd(build):
    b = IRBuilder(fptype=F32, arena='shrMem')
    build(b)
    ctx = Context(arch='pvc', backend='esimd', fp_type=F32)
    writer = Writer()
    EsimdEmitter(writer=writer, context=ctx, strict=False).run(
        placed(b.finish()))
    return writer.get_src()


@pytest.mark.parametrize('rows,cols,transpose,spelled', [
    (8, 16, False, 'simd<float, 128> v1_blk = tensorforge::blockLoad2d'
                   '<float, 16, 8, false>(m1, 256, 64, 256, 0, 3);'),
    (16, 8, True, 'simd<float, 128> v1_blk = tensorforge::blockLoad2d'
                  '<float, 8, 16, true>(m1, 256, 64, 256, 0, 3);'),
    (32, 16, False, 'simd<float, 512> v1_blk = tensorforge::blockLoad2d'
                    '<float, 16, 32, false>(m1, 256, 64, 256, 0, 3);'),
    (16, 1, True, 'simd<float, 16> v1_blk = tensorforge::blockLoad2d'
                  '<float, 1, 16, true>(m1, 256, 64, 256, 0, 3);'),
], ids=['rows', 'transposed', 'rows-32', 'one-column'])
def test_esimd_reads_the_block_with_one_load_2d(rows, cols, transpose,
                                                spelled):
    """The block `load_2d` is given -- `Width` the block's columns, `Height`
    its rows -- and the `simd` it returns as the initializer: in the order
    of the lanes' share already, nothing copied."""
    src = _esimd(lambda b: _read(b, rows, cols, transpose))
    assert src.count(spelled) == 1, src
    assert 'copy_from' not in src, src


def _predicated(build):
    """`build`'s body, its 2D block under a predicate."""
    def wrapped(b):
        flag = b.op('lt', BOOL, b.const(1, INDEX), 8, hint='g')
        build(b)
        return flag
    return wrapped


def _guard(body, flag):
    return tuple(dataclasses.replace(s, predicate=flag)
                 if block_reads.is_block2d(s) else s for s in body)


def test_esimd_refuses_a_predicated_block():
    """One message reads the block whole: there is no lane to leave out."""
    b = IRBuilder(fptype=F32, arena='shrMem')
    flag = _predicated(_read)(b)
    body = _guard(placed(b.finish()), flag)
    ctx = Context(arch='pvc', backend='esimd', fp_type=F32)
    with pytest.raises(IRError, match='under a predicate'):
        EsimdEmitter(writer=Writer(), context=ctx, strict=False).run(body)


# -- under SPMD ---------------------------------------------------------------- #

def _spmd(build, lanes=16):
    b = IRBuilder(fptype=F32)
    build(b)
    ctx = Context(arch='pvc', backend='oneapi', fp_type=F32,
                  options=Options())
    ctx.mult_lanes = lanes
    writer = Writer()
    Emitter(writer, ctx).run(b.finish())
    return writer.get_src(), ctx


@pytest.mark.parametrize('rows,cols,transpose,spelled', [
    (8, 16, False, 'sycl::vec<float, 8> v1_blk = tensorforge::subGroupBlock2d'
                   '<float, 8, 16, false>(item.get_sub_group(), m1, 256, 64, '
                   '256, 0, 3);'),
    (16, 4, True, 'sycl::vec<float, 4> v1_blk = tensorforge::subGroupBlock2d'
                  '<float, 16, 4, true>(item.get_sub_group(), m1, 256, 64, '
                  '256, 0, 3);'),
    (1, 16, False, 'float v1_blk = tensorforge::subGroupBlock2d'
                   '<float, 1, 16, false>(item.get_sub_group(), m1, 256, 64, '
                   '256, 0, 3);'),
], ids=['rows', 'transposed', 'one-row'])
def test_spmd_reads_the_block_with_the_sub_groups_read(rows, cols, transpose,
                                                       spelled):
    """Each work-item its column, or its row, as a `sycl::vec` -- a number
    where it is one element."""
    src, _ = _spmd(lambda b: _read(b, rows, cols, transpose))
    assert src.count(spelled) == 1, src


@pytest.mark.parametrize('lanes', [32, 8, None],
                         ids=['sub-group-32', 'sub-group-8', 'unknown'])
def test_spmd_reads_a_block_only_with_a_sub_group_of_16(lanes):
    """The message hands the block to the 16 work-items of a full
    sub-group, which have to be the multiplication's lanes."""
    with pytest.raises(IRError, match='no sub-group of 16'):
        _spmd(lambda b: _read(b), lanes=lanes)


def test_spmd_refuses_a_predicated_block():
    b = IRBuilder(fptype=F32)
    flag = _predicated(_read)(b)
    body = _guard(b.finish(), flag)
    ctx = Context(arch='pvc', backend='oneapi', fp_type=F32,
                  options=Options())
    ctx.mult_lanes = 16
    with pytest.raises(IRError, match='under a predicate'):
        Emitter(Writer(), ctx).run(body)


def test_spmd_has_no_share_of_32():
    """A lane's share is one `sycl::vec`, at most 16: 32 rows are the
    explicit vector's."""
    with pytest.raises(IRError, match='one sycl::vec of at most 16'):
        _spmd(lambda b: _read(b, rows=32))


def test_the_device_header_declares_the_reads_the_shapes_take():
    """The builtins `sycl_spmd.h` declares, one for each shape the SPMD
    lowering takes: rows up to `BLOCK2D_SPMD_MAX`, the transposed columns."""
    text = HEADER.read_text()
    declared = set(re.findall(
        r'TENSORFORGE_BLOCK2D_READ\((intel_sub_group_2d_block_read_\w+)\)',
        text))
    wanted = ({f'intel_sub_group_2d_block_read_32b_{r}r16x1c'
               for r in block_reads.BLOCK2D_ROWS
               if r <= block_reads.BLOCK2D_SPMD_MAX}
              | {f'intel_sub_group_2d_block_read_transpose_32b_16r{c}x1c'
                 for c in block_reads.BLOCK2D_COLS})
    assert declared == wanted
    called = set(re.findall(r'\b(intel_sub_group_2d_block_read_\w+)\(\s*surface',
                            text))
    assert called == wanted


def test_the_header_comes_with_a_2d_block():
    """`sycl_spmd.h` is included where a body reads a 2D block, which the
    builder of one says (`SyclLexic.block2d`)."""
    _, ctx = _spmd(lambda b: _read(b))
    lex = ctx.target.lexic
    lex.split_broadcast = lex.dpas = lex.block_runs = False
    assert 'tensorforge_device/sycl_spmd.h' not in ctx.target.headers()
    lex.block2d = True
    assert 'tensorforge_device/sycl_spmd.h' in ctx.target.headers()


def test_the_shim_parses_both_reads():
    """The generated calls against the syntax shim's declarations."""
    from harness import syntax
    if syntax.compiler() is None:
        pytest.skip('no host compiler')
    spmd, _ = _spmd(lambda b: _read(b, 16, 8, True))
    esimd = _esimd(lambda b: _read(b))
    for src, backend in ((spmd, 'oneapi'), (esimd, 'esimd')):
        body = '\n'.join(l for l in src.splitlines()
                         if not re.match(r'\s*float m\d+\[', l))
        kernel = ('void k(sycl::nd_item<3> item, const float *m1) {\n'
                  + body + '\n}\n')
        result = syntax.check_source(kernel, shim=syntax.shim_for(backend))
        assert result.ok, (backend, result.errors())
