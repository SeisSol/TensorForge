# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Reads served from one block message, as statements (`pir.block_reads`).

A window -- scalar reads of shared memory at constant addresses -- and a run
-- reads of one buffer at consecutive offsets, a lane each -- are one `load`
with `engine='block'` and its `shape`, in front of the first read it serves,
and the reads become pure extracts of it where they stood.  Under the
explicit vector the text is what the emitter wrote before from tables of its
own (`test_esimd_emitter.py` holds those spellings); under SPMD a run is a
sub-group block read where `Options.spmd_block_runs` asks for one.
"""

from __future__ import annotations

import re

import pytest

from tensorforge.backend.pir import block_reads
from tensorforge.backend.pir.build import IRBuilder
from tensorforge.backend.pir.core import (BOOL, INDEX, SCALAR_LAYOUT, LaneAxis,
                                          MemSpace, Op, RegisterLayout)
from tensorforge.backend.pir.emit import Emitter
from tensorforge.backend.pir.emit_esimd import EsimdEmitter
from tensorforge.backend.writer import Writer
from tensorforge.common.basic_types import Datatype
from tensorforge.common.context import Context, Options
from harness.placement import placed

F32 = Datatype.F32
SPREAD16 = RegisterLayout((LaneAxis(16),))
SPREAD32 = RegisterLayout((LaneAxis(32),))


def _shared_reads(b):
    """Three scalar reads of `s7`, folded into a register vector."""
    s = b.alloc(F32, (64,), MemSpace.SHARED, extern='s7')
    acc = b.alloc(F32, (16,), MemSpace.REGISTER, extern='r7', init='{}')
    x = b.load(acc, 0, hint='x', layout=SPREAD16)
    for k in (3, 5, 9):
        c = b.load(s, k, hint='c', layout=SCALAR_LAYOUT)
        x = b.op('mul', F32, x, c, hint='x')
    b.store(acc, x, 0)


def _lane_reads(b, offsets=(0, 32), layout=SPREAD32):
    """Reads of `g` at consecutive offsets, a lane each, summed."""
    g = b.alloc(F32, (128,), MemSpace.SHARED, extern='g')
    acc = b.alloc(F32, (32,), MemSpace.REGISTER, extern='r7', init='{}')
    x = b.load(acc, 0, hint='x', layout=layout)
    for k in offsets:
        x = b.op('add', F32, x, b.load(g, k, hint='u', layout=layout),
                 hint='x')
    b.store(acc, x, 0)


def _body(build):
    b = IRBuilder(fptype=F32, arena='shrMem')
    build(b)
    return placed(b.finish())


def _esimd(body):
    ctx = Context(arch='pvc', backend='esimd', fp_type=F32)
    writer = Writer()
    EsimdEmitter(writer=writer, context=ctx, strict=False).run(body)
    return writer.get_src()


# -- under the explicit vector ------------------------------------------------ #

def test_a_window_is_one_read_and_extracts_where_the_reads_stood():
    """One `load` of the window's 16 elements from its lowest, in front of
    its first read, named as the vector is spelled; each read the extract of
    its element, kept a declaration of its own."""
    body = _body(_shared_reads)
    out = block_reads.block_reads(body)
    heads = [s for s in out if block_reads.is_block(s)]
    assert len(heads) == 1
    head = heads[0]
    assert head.attr('shape') == (16,) and head.attr('extern') == 's7_w0'
    assert block_reads.const_flat(head.args[0], head.args[1:], {}) == 3
    parts = [s for s in out if block_reads.is_part(s)]
    assert [(s.op, s.attr('lane')) for s in parts] == [
        (Op.EXTRACT, 0), (Op.EXTRACT, 2), (Op.EXTRACT, 6)]
    assert all(s.pure and s.attr('no_inline') for s in parts)
    assert out.index(head) == out.index(parts[0]) - 1
    reads = [s for s in body if s.op == Op.LOAD
             and getattr(s.args[0], 'type', None) is not None
             and s.args[0].type.space is MemSpace.SHARED]
    assert [s.target for s in parts] == [s.target for s in reads]


def test_a_window_reads_as_it_did():
    """The text of the vector and its elements, as the emitter wrote them
    from its own tables before."""
    src = _esimd(_body(_shared_reads))
    assert src.count('slmLoad<float, 16>(s7 + 3)') == 1, src
    assert re.search(r'float \w+ = s7_w0\[0\];', src), src
    assert 's7_w0[6]' in src and 's7[' not in src, src


def test_a_run_is_one_read_and_slices():
    """Two 32-lane reads at 0 and 32 of `g`: one read of 64, sliced."""
    body = _body(_lane_reads)
    out = block_reads.block_reads(body)
    head = next(s for s in out if block_reads.is_block(s))
    assert head.attr('shape') == (32, 2) and head.attr('extern') == 'g_run0'
    parts = [s for s in out if block_reads.is_part(s)]
    assert [(s.op, s.args[1]) for s in parts] == [(Op.SLICE, 0),
                                                  (Op.SLICE, 32)]
    src = _esimd(body)
    assert src.count('g_run0') == 3, src
    assert 'g_run0.template select<32, 1>(32)' in src, src


def test_a_body_without_either_is_left_as_it_is():
    body = _body(lambda b: _lane_reads(b, offsets=(0,)))
    assert block_reads.block_reads(body) is body


def test_the_vectors_are_numbered_apart_from_every_value():
    """Below every object and every value: nothing built after the pass
    is renumbered, and a read a block serves is told by its source."""
    out = block_reads.block_reads(_body(_shared_reads))
    head = next(s for s in out if block_reads.is_block(s))
    assert head.target[0].id < -(1 << 39)
    assert not any(block_reads.is_part(s) for s in out
                   if s.op == Op.EXTRACT and s.args[0].id >= 0)


# -- under SPMD --------------------------------------------------------------- #

def _spmd_reads(b, flats=(0, 16, 32, 48), lanes=16, between=None,
                guarded=()):
    """Reads of global `m1` at `rem(tid, lanes) + c`, every other statement
    a store of what they read; `between(b)` after the second read, and the
    reads at `guarded` under a predicate."""
    from tensorforge.backend.pir.core import BufferType
    g = b.alloc(F32, (256,), MemSpace.GLOBAL, extern='m1')
    out = b.alloc(F32, (256,), MemSpace.GLOBAL, extern='m0')
    lane = b.op('rem', INDEX, b.thread_id('x'), lanes, hint='lead')
    layout = RegisterLayout((LaneAxis(lanes),))
    for i, c in enumerate(flats):
        at = b.op('add', INDEX, lane, c, hint='a')
        pred = (b.op('lt', BOOL, lane, 8, hint='g') if c in guarded
                else None)
        v = b.load(g, at, hint='u', layout=layout, predicate=pred)
        b.store(out, v, at)
        if i == 1 and between is not None:
            between(b, g, lane)


def _spmd_runs(build, lanes=16):
    b = IRBuilder(fptype=F32)
    build(b)
    body = b.finish()
    stmts = block_reads.walk_stmts(body)
    allocs, names = block_reads.allocations(stmts)
    return body, block_reads.plan_spmd_runs(body, allocs, names, lanes)


def test_an_spmd_run_is_reads_one_sub_group_apart_in_any_order():
    """Taken together wherever they stand in the list: the block read goes
    in front of the first, and read `j` is element `j`."""
    _, runs = _spmd_runs(lambda b: _spmd_reads(b, flats=(16, 0, 48, 32)))
    assert len(runs) == 1
    assert [f for _, f in runs[0].reads] == [16, 0, 48, 32]
    body, _ = _spmd_runs(lambda b: _spmd_reads(b, flats=(16, 0, 48, 32)))
    out = block_reads.block_reads(body, windows=False, runs=False,
                                  spmd_lanes=16)
    head = next(s for s in out if block_reads.is_block(s))
    assert head.args[1] == 0 and head.attr('shape') == (16, 4)
    lanes = [s.attr('lane') for s in out if block_reads.is_part(s)]
    assert lanes == [1, 0, 3, 2]


@pytest.mark.parametrize('change,expected', [
    (dict(flats=(0, 16, 32)), [[0, 16]]),
    (dict(guarded=(32,)), [[0, 16]]),
    (dict(between=lambda b, g, lane: b.store(g, b.const(1.0), lane)),
     [[0, 16], [32, 48]]),
    (dict(flats=(0, 24)), []),
], ids=['three', 'guarded', 'store-between', 'not-a-sub-group-apart'])
def test_what_an_spmd_run_takes(change, expected):
    """A power of two of reads, none predicated, each a sub-group past the
    last -- and a store to the buffer between them parts them."""
    _, runs = _spmd_runs(lambda b: _spmd_reads(b, **change))
    assert [[f for _, f in r.reads] for r in runs] == expected


def _spmd_src(lanes, reads=16, **options):
    b = IRBuilder(fptype=F32)
    _spmd_reads(b, flats=tuple(k * reads for k in range(4)), lanes=reads)
    ctx = Context(arch='pvc', backend='oneapi', fp_type=F32,
                  options=Options(**options))
    ctx.mult_lanes = lanes
    writer = Writer()
    Emitter(writer, ctx).run(b.finish())
    return writer.get_src(), ctx


def test_an_spmd_run_is_a_sub_group_block_read_where_asked():
    src, ctx = _spmd_src(16, spmd_block_runs=True)
    assert ('sycl::vec<float, 4> m1_run0 = tensorforge::subGroupRun<4>('
            'item.get_sub_group(), &m1[0]);') in src, src
    assert re.search(r'float \w+ = m1_run0\[3\];', src), src
    assert 'tensorforge_device/sycl_spmd.h' in ctx.target.headers()


@pytest.mark.parametrize('lanes,reads,options', [
    (16, 16, {}), (32, 16, {'spmd_block_runs': True}),
    (24, 24, {'spmd_block_runs': True}), (None, 16, {'spmd_block_runs': True}),
], ids=['off', 'narrower-than-the-multiplication', 'no-sub-group-of-24',
        'lanes-unknown'])
def test_no_spmd_run_where_not_asked_or_not_the_sub_group(lanes, reads,
                                                          options):
    """Off; a read spread over fewer lanes than its multiplication; a
    multiplication of 24, which a sub-group of 32 holds with lanes to
    spare, so its lane is no sub-group's; and lanes nobody stated."""
    src, _ = _spmd_src(lanes, reads, **options)
    assert 'subGroupRun' not in src, src


def test_spmd_runs_are_off_by_default():
    from tensorforge.common.options import registry
    assert registry()['spmd_block_runs'].default is False


def test_the_shim_parses_an_spmd_run():
    """The generated call against the syntax shim's declaration of it."""
    from harness import syntax
    if syntax.compiler() is None:
        pytest.skip('no host compiler')
    src, _ = _spmd_src(16, spmd_block_runs=True)
    body = '\n'.join(l for l in src.splitlines()
                     if not re.match(r'\s*float m\d+\[', l))
    kernel = ('void k(sycl::nd_item<3> item, float *m0, float *m1) {\n'
              + body + '\n}\n')
    result = syntax.check_source(kernel, shim=syntax.shim_for('oneapi'))
    assert result.ok, result.errors()
