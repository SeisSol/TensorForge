# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""A buffer of the multiplication's own shared memory that every lane reads
only what it wrote itself, kept in registers over the statements that touch
it (`pir.carry.keep_own`), asked of the bodies `test_carry.py` builds.

The carry keeps such a buffer in registers over a merged run's trips, and
asks of the code ahead of the run that it stores the buffer from registers
(`test_carry.py`).  Here the code ahead stores into the same registers as
the trips, whatever it stores, and the arena loses the buffer -- where
every lane reads, in one register for every element it reaches, what it
wrote there before.
"""

from __future__ import annotations

import contextlib
import io
import re
import warnings

import numpy as np
import pytest

from test_carry import COLS, LANES, Own, _shared
from test_oracle_fold import HOST, SEED, _case, _tensors
from tensorforge.common.context import Context
from tensorforge.common.options import Options
from tensorforge.generators.generator import Generator
from tensorforge.reference import kernel_eval as ke
from tensorforge.backend.pir import IRBuilder, verify
from tensorforge.backend.pir.carry import keep_own
from tensorforge.backend.pir.core import (BOOL, INDEX, SIZE, BufferType,
                                          Effect, MemSpace, Op, ScalarType,
                                          Uniformity, walk_stmts)
from tensorforge.common.basic_types import Datatype


def _keep(run):
    run.report = []
    out = keep_own(run.body, run.b.scratch, threads=LANES, arena='shrMem',
                   report=run.report)
    verify(out)
    return out


def _declared(body, buffer) -> bool:
    return any(s.op is Op.ALLOC and buffer in s.target
               for s in walk_stmts(body))


def _image(body):
    image, = [s.target[0] for s in walk_stmts(body) if s.op is Op.ALLOC
              and s.attr('extern') == 'own_s0']
    return image


def test_a_temporary_each_lane_keeps_to_itself_leaves_the_arena():
    """Every lane adds onto its own rows of `S` over the trips of a run,
    which the code ahead of it stored: all of it in registers of the lane's
    own, the store ahead of the run as well, and `S` declared no more."""
    run = Own()
    out = _keep(run)
    assert not _shared(out, run.s)
    assert not _declared(out, run.s)
    image = _image(out)
    assert image.type.space is MemSpace.REGISTER
    assert image.type.shape == (COLS,)
    assert any(f'own: s0 in {COLS} registers' in line for line in run.report)


def test_the_elements_the_trips_leave_alone_are_kept_as_well():
    """The trips leave the last column alone, which the code ahead stores:
    a register of the image for it, written and read by nobody else."""
    run = Own(extra=True)
    out = _keep(run)
    assert not _shared(out, run.s)
    assert _image(out).type.shape == (COLS,)


def test_a_run_in_a_loop_of_its_own_keeps_it_in_that_loop():
    """The run stands in a loop of its own, whose body stores `S` ahead of
    the run: every trip of that loop writes before it reads, and the image
    is declared in its body."""
    run = Own(outer=True)
    out = _keep(run)
    assert not _shared(out, run.s)
    outer = next(s for s in out if s.op is Op.FOR)
    assert any(s.op is Op.ALLOC and s.attr('extern') == 'own_s0'
               for s in outer.regions[0].body)


def test_a_mark_that_defines_it_anew_goes_with_it():
    run = Own(mark=True)
    out = _keep(run)
    assert not _shared(out, run.s)
    assert not any(s.op is Op.MARK and run.s in s.args
                   for s in walk_stmts(out))


def test_what_another_lane_reads_stays_in_memory():
    """Lane `l` reads the row of lane `l + 1` behind the run."""
    run = Own(across=True)
    assert _keep(run) == run.body
    assert any('another lane writes' in line for line in run.report)


def test_a_lane_that_adds_another_lanes_row_keeps_it_in_memory():
    run = Own(other=True)
    assert _keep(run) == run.body
    assert any('another lane writes' in line for line in run.report)


def test_what_a_lane_reads_before_it_writes_it_stays_in_memory():
    """The code ahead of the run leaves the last column out, which the
    trips read before they write it: what a lane reads there no statement
    that touches `S` wrote."""
    run = Own(partial=True)
    assert _keep(run) == run.body
    assert any('did not write there before' in line for line in run.report)


def test_a_register_that_would_hold_two_elements_keeps_it_in_memory():
    run = Own(parity=True)
    assert _keep(run) == run.body
    assert any("two of a lane's elements" in line for line in run.report)


def test_a_register_a_loop_picks_that_is_not_unrolled_stays_in_memory():
    run = Own(plain=True)
    assert _keep(run) == run.body
    assert any('not unrolled' in line for line in run.report)


def test_a_buffer_text_writes_stays_in_memory():
    """A statement spelled in text writes the buffer, which no register of
    a lane's image can say it reaches."""
    b = IRBuilder(fptype=Datatype.F32, arena='shrMem')
    k = b.extern_value('element', SIZE, uniform=Uniformity.MULT, hint='k')
    p = b.decl_expr(
        'float *const __restrict__ glb_m1', '&m1[{0} * 32]',
        BufferType(Datatype.F32, (LANES,), MemSpace.GLOBAL), 'm1', args=(k,),
        kind=Effect.READ, space=MemSpace.GLOBAL, hint='glb_m1',
        extern='glb_m1')
    lane = b.op('rem', INDEX, b.thread_id('x'), LANES, hint='lead')
    s = b.alloc(Datatype.F32, (LANES,), MemSpace.SHARED, hint='s',
                extern='s0')
    b.store(s, 1.0, lane)
    b.access_stmt('s0[0] = 2.0f;', s, Effect.WRITE)
    b.store(p, b.load(s, lane), lane)
    body = b.finish()
    report = []
    assert keep_own(body, b.scratch, threads=LANES, arena='shrMem',
                    report=report) == body
    assert any('a rawstmt touches it' in line for line in report)


def test_a_statement_that_says_nothing_of_what_it_touches_keeps_it():
    """Between the store and the load, text that declares nothing about what
    it touches: it may read the buffer, which its registers would not be."""
    b = IRBuilder(fptype=Datatype.F32, arena='shrMem')
    k = b.extern_value('element', SIZE, uniform=Uniformity.MULT, hint='k')
    p = b.decl_expr(
        'float *const __restrict__ glb_m1', '&m1[{0} * 32]',
        BufferType(Datatype.F32, (LANES,), MemSpace.GLOBAL), 'm1', args=(k,),
        kind=Effect.READ, space=MemSpace.GLOBAL, hint='glb_m1',
        extern='glb_m1')
    lane = b.op('rem', INDEX, b.thread_id('x'), LANES, hint='lead')
    s = b.alloc(Datatype.F32, (LANES,), MemSpace.SHARED, hint='s',
                extern='s0')
    b.store(s, 1.0, lane)
    b('opaque();')
    b.store(p, b.load(s, lane), lane)
    body = b.finish()
    report = []
    assert keep_own(body, b.scratch, threads=LANES, arena='shrMem',
                    report=report) == body
    assert any('says nothing about what it touches' in line
               for line in report)


def _lane_body():
    """A builder with `p`, a lane's element of global memory, the lane, and
    `s`, a buffer of the arena: `(b, p, lane, s, k)`, `k` the element."""
    b = IRBuilder(fptype=Datatype.F32, arena='shrMem')
    k = b.extern_value('element', SIZE, uniform=Uniformity.MULT, hint='k')
    p = b.decl_expr(
        'float *const __restrict__ glb_m1', '&m1[{0} * 32]',
        BufferType(Datatype.F32, (LANES,), MemSpace.GLOBAL), 'm1', args=(k,),
        kind=Effect.READ, space=MemSpace.GLOBAL, hint='glb_m1',
        extern='glb_m1')
    lane = b.op('rem', INDEX, b.thread_id('x'), LANES, hint='lead')
    s = b.alloc(Datatype.F32, (LANES,), MemSpace.SHARED, hint='s',
                extern='s0')
    return b, p, lane, s, k


def test_a_loop_whose_trips_hand_it_on_is_asked_whole():
    """Every trip of a run but the first adds what the trip before stored:
    asked of the trip alone, the read stands under a hop nothing names; over
    the loop, every lane reads what it wrote the trip before."""
    b, p, lane, s, _ = _lane_body()
    real = ScalarType(Datatype.F32)
    with b.for_(0, 3, 1, rolled=True, hint='i') as hop:
        x = b.load(p, lane)
        with b.if_(b.op('gt', BOOL, hop.induction, 0, hint='later')):
            b.store(p, b.op('add', real, x, b.load(s, lane)), lane)
        b.store(s, x, lane)
    body = b.finish()
    verify(body)
    report = []
    out = keep_own(body, b.scratch, threads=LANES, arena='shrMem',
                   report=report)
    verify(out)
    assert not _shared(out, s)
    assert any('own: s0 in 1 registers' in line for line in report)


def test_a_loop_that_does_not_touch_it_is_not_asked():
    """Between the store and the load, a loop over a count the body reads
    from memory: it touches nothing of the buffer, and what its trips are is
    no question here."""
    b, p, lane, s, k = _lane_body()
    b.store(s, 1.0, lane)
    with b.for_(0, k, 1, hint='n'):
        b.store(p, 0.0, lane)
    b.store(p, b.load(s, lane), lane)
    body = b.finish()
    verify(body)
    out = keep_own(body, b.scratch, threads=LANES, arena='shrMem')
    verify(out)
    assert not _shared(out, s)


def _built(case, arch, descrs=None, **options):
    descrs = case.descr_list() if descrs is None else descrs
    gen = Generator(descrs, Context(
        arch=arch, backend='cuda' if arch.startswith('sm') else 'hip',
        fp_type=case.DTYPE,
        options=Options(**{**getattr(case, 'OPTIONS', {}), **options})))
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter('ignore')
        gen.generate()
    return gen


@pytest.mark.parametrize('stem', [
    'ml_slices_then_ew', 'temp_accumulate_assembled',
    'temp_dead_slice_reassign', 'temp_reassign_narrower_slice'])
def test_a_temporary_kept_in_registers_computes_what_numpy_does(stem):
    """Temporaries of the corpus that every lane reads only what it wrote
    itself of: in registers of the lane's own, and no more in shared memory;
    and on the host oracle what the kernel computes is what NumPy does."""
    case = _case(stem)
    held = _built(case, 'sm_86').get_kernel()
    descrs = case.descr_list()
    gen = _built(case, 'sm_86', descrs, keep_own=True)
    src = gen.get_kernel()
    kept = set(re.findall(r'\bown_(s\d+)\b', src))
    assert kept
    for name in kept:
        assert re.search(rf'\b{name}\[', held)
        assert not re.search(rf'\b{name}\[', src)
    lanes, mults = ke.launch_geometry(gen.get_launcher())
    mem = ke.evaluate_wave(src, lanes, seed=SEED, globals_only=True,
                           mults=mults, block=True)
    fill = ke.Slot(SEED)
    tensors = _tensors(descrs)
    sinks = [d.writes().tensor.alias for d in descrs
             if not d.writes().tensor.is_tmp]
    out = getattr(case, 'OUTPUT', None) or sinks[-1]
    arrays = {}
    for alias, t in tensors.items():
        if t.is_tmp:
            continue
        arr = np.zeros((1,) + tuple(int(x) for x in t.shape))
        for slot, cell in HOST._slots(t).items():
            arr[(0,) + tuple(cell)] = fill.read(t.name, slot)
        arrays[alias] = arr
    want = np.asarray(case.reference(arrays, arrays[out].copy()))
    t = tensors[out]
    rtol, atol = getattr(case, 'TOL', (1e-5, 1e-5))
    for slot, cell in HOST._slots(t).items():
        got = mem.get((t.name, slot), fill.read(t.name, slot))
        expected = want.reshape((1,) + tuple(int(x) for x in t.shape))[
            (0,) + tuple(cell)]
        assert got == pytest.approx(expected, rel=rtol, abs=atol), (slot,
                                                                     cell)
