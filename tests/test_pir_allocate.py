# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Where each shared buffer goes, decided once the body is in its final order.

A shared alloc is a window into the arena the launch is sized for, never an
array of its own: `__shared__ float buf[N];` written inside a body would be
memory the occupancy calculation does not know about, which compiles and
silently costs blocks.  So a window has no offset until `pir.allocate` gives
it one, and the emitter refuses a window nobody placed.

Two buffers may share bytes where no statement occupies both, and a buffer is
occupied where it is touched and where it is live -- read further on before
something defines it anew.  What defines a buffer anew is said by the
instruction that writes it whole (`mark defines`): a write alone does not,
since a buffer assembled from slices is still wanted for the earlier ones.
"""

from __future__ import annotations

import re

import pytest

from tensorforge.backend.pir import emit
from tensorforge.backend.pir.allocate import BLOCK, MULT, allocate
from tensorforge.backend.pir.build import IRBuilder
from tensorforge.backend.pir.core import INDEX, IRError, MemSpace, Op, walk
from tensorforge.backend.writer import Writer
from tensorforge.common.basic_types import Datatype
from tensorforge.common.target import Target

ARENA = 'shrMem'


def _builder(fptype=Datatype.F32):
    return IRBuilder(fptype=fptype, arena=ARENA)


def _layout(body, align=1, **kw):
    arenas = kw.pop('arenas', {ARENA: MULT})
    out, layout = allocate(tuple(body), arenas=arenas, align=align, **kw)
    offsets = {}
    for s, _ in walk(out):
        if s.op == Op.ALLOC and s.target and s.attr('arena') is not None:
            offsets[s.target[0].hint] = s.attr('offset')
    return out, layout, offsets


def _burst(b, buf, lane, defines=True):
    """Write `buf` whole and read it back: one stretch it is occupied for.

    Saying so first, as every instruction that writes a buffer whole does: a
    store alone does not end what the buffer held, and without the mark its
    value starts at its window."""
    if defines:
        b.mark('defines', buf)
    b.store(buf, b.const(1.0), lane)
    return b.load(buf, lane, hint='d')


def _src(body) -> str:
    w = Writer()
    emit(body, w, Target('sm_86', 'cuda'))
    return w.get_src()


# --------------------------------------------------------------------------- #
# What a shared buffer is
# --------------------------------------------------------------------------- #

def test_a_shared_buffer_is_a_window_into_the_arena():
    """The regression this whole module exists for: no array of its own."""
    b = _builder()
    lane = b.thread_id('x')
    tile = b.alloc(Datatype.F32, (16, 4), MemSpace.SHARED, hint='tile')
    _burst(b, tile, lane)
    out, _, _ = _layout(b.finish())
    src = _src(out)
    assert '__shared__' not in src
    assert re.search(r'\*\s*\w+_tile\s*=\s*&shrMem\[0\]', src), src


def test_a_window_nobody_placed_is_refused_at_emission():
    b = _builder()
    tile = b.alloc(Datatype.F32, (8,), MemSpace.SHARED, hint='tile')
    b.store(tile, b.const(1.0), b.thread_id('x'))
    with pytest.raises(IRError, match='nothing placed'):
        _src(b.finish())


def test_register_and_global_buffers_are_not_the_allocator_s():
    b = _builder()
    b.alloc(Datatype.F32, (8,), MemSpace.REGISTER, hint='r')
    b.alloc(Datatype.F32, (8,), MemSpace.GLOBAL, hint='g')
    out, layout, offsets = _layout(b.finish())
    assert offsets == {}
    assert layout.per_mult == 0


# --------------------------------------------------------------------------- #
# Who shares bytes with whom
# --------------------------------------------------------------------------- #

def test_buffers_occupied_together_get_bytes_of_their_own():
    b = _builder()
    lane = b.thread_id('x')
    a = b.alloc(Datatype.F32, (64,), MemSpace.SHARED, hint='a')
    c = b.alloc(Datatype.F32, (32,), MemSpace.SHARED, hint='c')
    b.store(a, b.const(1.0), lane)
    b.store(c, b.const(2.0), lane)
    b.load(a, lane, hint='d')
    b.load(c, lane, hint='d')
    _, layout, offsets = _layout(b.finish())
    assert offsets == {'a': 0, 'c': 64}
    assert layout.per_mult == 96


def test_buffers_never_occupied_together_share_bytes():
    b = _builder()
    lane = b.thread_id('x')
    a = b.alloc(Datatype.F32, (64,), MemSpace.SHARED, hint='a')
    c = b.alloc(Datatype.F32, (32,), MemSpace.SHARED, hint='c')
    _burst(b, a, lane)
    _burst(b, c, lane)
    _, layout, offsets = _layout(b.finish())
    assert offsets == {'a': 0, 'c': 0}
    assert layout.per_mult == 64


@pytest.mark.parametrize('defines', [False, True])
def test_a_value_ends_where_the_writer_says_the_buffer_is_defined_anew(defines):
    """`a` is written in two bursts with `c` in between.  A write alone does
    not end the first burst's value -- `a` may be assembled from both -- so
    without the mark `a` is occupied throughout and `c` gets its own bytes.
    With it, the stretch between the bursts holds nothing anybody wants."""
    b = _builder()
    lane = b.thread_id('x')
    a = b.alloc(Datatype.F32, (64,), MemSpace.SHARED, hint='a')
    c = b.alloc(Datatype.F32, (32,), MemSpace.SHARED, hint='c')
    _burst(b, a, lane)
    _burst(b, c, lane)
    _burst(b, a, lane, defines=defines)
    _, layout, offsets = _layout(b.finish())
    if defines:
        assert offsets == {'a': 0, 'c': 0}
    else:
        assert offsets['c'] >= 64 or offsets['a'] >= 32


def test_a_buffer_carried_around_a_back_edge_is_occupied_throughout():
    """Read at the top of the body before anything defines it there: the
    value comes from the previous iteration, so `c`, used in between, may
    not take its bytes."""
    b = _builder()
    lane = b.thread_id('x')
    a = b.alloc(Datatype.F32, (32,), MemSpace.SHARED, hint='a')
    c = b.alloc(Datatype.F32, (32,), MemSpace.SHARED, hint='c')
    b.mark('defines', a)
    b.store(a, b.const(0.0), lane)
    with b.for_(0, 4, 1):
        b.load(a, lane, hint='d')
        _burst(b, c, lane)
        b.store(a, b.const(1.0), lane)
    _, _, offsets = _layout(b.finish())
    assert offsets['a'] != offsets['c']


def test_a_statement_that_does_not_say_what_it_touches_reads_everything():
    """So nothing shares bytes across it: `a` is live there."""
    b = _builder()
    lane = b.thread_id('x')
    a = b.alloc(Datatype.F32, (32,), MemSpace.SHARED, hint='a')
    c = b.alloc(Datatype.F32, (32,), MemSpace.SHARED, hint='c')
    b.mark('defines', a)
    b.store(a, b.const(0.0), lane)
    b('opaque();')
    b.mark('defines', c)
    _burst(b, c, lane)
    _, _, offsets = _layout(b.finish())
    assert offsets['a'] != offsets['c']


def test_a_buffer_nothing_touches_takes_no_room():
    b = _builder()
    lane = b.thread_id('x')
    b.alloc(Datatype.F32, (1024,), MemSpace.SHARED, hint='unused')
    a = b.alloc(Datatype.F32, (32,), MemSpace.SHARED, hint='a')
    _burst(b, a, lane)
    _, layout, _ = _layout(b.finish())
    assert layout.per_mult == 32


def test_the_windows_of_one_buffer_are_placed_once():
    """Two windows naming one buffer as their identity are the same bytes:
    a later user writes through the window an earlier one declared."""
    b = _builder()
    lane = b.thread_id('x')
    owner = object()
    first = b.alloc(Datatype.F32, (32,), MemSpace.SHARED, hint='first',
                    identity=owner)
    other = b.alloc(Datatype.F32, (32,), MemSpace.SHARED, hint='other')
    b.store(first, b.const(0.0), lane)
    _burst(b, other, lane)
    second = b.alloc(Datatype.F32, (32,), MemSpace.SHARED, hint='second',
                     identity=owner)
    b.load(second, lane, hint='d')
    _, _, offsets = _layout(b.finish())
    assert offsets['first'] == offsets['second']
    assert offsets['other'] != offsets['first']


# --------------------------------------------------------------------------- #
# Where a buffer may start
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('dtype,align', [(Datatype.F32, 4), (Datatype.F64, 2)])
def test_every_buffer_starts_on_sixteen_bytes(dtype, align):
    """`nvidia.matmul` stores through `float4`; an unaligned window faults.
    In elements, so it depends on the element size."""
    b = _builder(dtype)
    lane = b.thread_id('x')
    odd = b.alloc(dtype, (3,), MemSpace.SHARED, hint='odd')
    nxt = b.alloc(dtype, (8,), MemSpace.SHARED, hint='next')
    b.store(odd, b.const(1.0), lane)
    b.store(nxt, b.const(1.0), lane)
    b.load(odd, lane, hint='d')
    b.load(nxt, lane, hint='d')
    _, _, offsets = _layout(b.finish(), align=align)
    assert all(o % align == 0 for o in offsets.values()), offsets
    assert offsets['next'] >= 3 or offsets['odd'] >= 8


def test_a_window_may_ask_for_more():
    b = _builder()
    lane = b.thread_id('x')
    big = b.alloc(Datatype.F32, (5,), MemSpace.SHARED, hint='big')
    wide = b.alloc(Datatype.F32, (4,), MemSpace.SHARED, hint='wide',
                   place_align=8)
    b.store(big, b.const(1.0), lane)
    b.store(wide, b.const(1.0), lane)
    b.load(big, lane, hint='d')
    b.load(wide, lane, hint='d')
    _, _, offsets = _layout(b.finish())
    assert offsets['wide'] % 8 == 0


def test_a_buffer_with_two_stages_takes_both_and_names_its_stage():
    """Both stages reserved, and the window's offset the stage's: its
    operand, which the emitter spells."""
    b = _builder()
    lane = b.thread_id('x')
    stage = b.op('bitand', INDEX, lane, 1, hint='stage')
    owner = object()
    w = b.alloc(Datatype.F32, (16,), MemSpace.SHARED, hint='w',
                identity=owner, stages=2, stage=stage)
    other = b.alloc(Datatype.F32, (8,), MemSpace.SHARED, hint='other')
    b.store(w, b.const(1.0), lane)
    b.store(other, b.const(1.0), lane)
    b.load(w, lane, hint='d')
    b.load(other, lane, hint='d')
    out, layout, offsets = _layout(b.finish())
    assert offsets['w'] == '0 + ({0}) * 16'
    assert offsets['other'] == 32
    assert layout.per_mult == 40
    writer = Writer()
    emit(out, writer, Target('sm_86', 'cuda'))
    assert '= &shrMem[0 + ((threadIdx.x & 1)) * 16];' in writer.get_src(), (
        writer.get_src())


def test_the_block_s_buffers_are_back_to_back_in_allocation_order():
    """Written ahead of everything that reads them and read to the end, so
    there is nothing to share: each starts where the one before it ends."""
    b = _builder()
    lane = b.thread_id('x')
    first = b.alloc(Datatype.F32, (10,), MemSpace.SHARED, hint='first',
                    arena='block')
    second = b.alloc(Datatype.F32, (6,), MemSpace.SHARED, hint='second',
                     arena='block')
    for buf in (first, second):
        b.store(buf, b.const(1.0), lane)
    for buf in (first, second):
        b.load(buf, lane, hint='d')
    _, layout, offsets = _layout(b.finish(), arenas={'block': BLOCK},
                                 block_align=4)
    assert offsets == {'first': 0, 'second': 12}
    assert layout.block == 18


# --------------------------------------------------------------------------- #
# What the generator makes of it
# --------------------------------------------------------------------------- #

def _sections(case: str, **options):
    import importlib.util
    from pathlib import Path

    from tensorforge.common.context import Context
    from tensorforge.common.options import Options
    from tensorforge.generators.generator import Generator

    path = next((Path(__file__).resolve().parent / 'cases').rglob(case))
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ctx = Context(arch='sm_86', backend='cuda',
                  fp_type=getattr(mod, 'DTYPE', None) or Datatype.F32,
                  options=Options(**options))
    gen = Generator(mod.descr_list(), ctx)
    gen.generate()
    return gen._sections


def _windows(section):
    out = {}
    for s, _ in walk(section.body):
        if (s.op == Op.ALLOC and s.target
                and getattr(s.target[0].type, 'space', None)
                == MemSpace.SHARED and s.attr('offset') is not None):
            out[s.target[0].hint] = (s.attr('offset'), s.target[0].type.volume)
    return out


def test_the_epilogue_tile_goes_over_the_staging_tiles():
    """`nvidia.matmul` writes `C` in the epilogue, after the last read of `A`
    and `B`, and says so in front of each burst -- so the three tiles share
    bytes without anything stating the packing by hand."""
    for section in _sections('rectangular.py', tensor_cores=True):
        tiles = {k: v for k, v in _windows(section).items()
                 if k in ('atile', 'btile', 'ctile')}
        if not tiles:
            continue
        (co, cn), (ao, an) = tiles['ctile'], tiles['atile']
        assert co < ao + an and ao < co + cn, tiles
        return
    pytest.fail('no staging tiles: the matrix path was not taken')
