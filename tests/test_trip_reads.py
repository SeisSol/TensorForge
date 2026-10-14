# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""`trip_reads`: what a trip of a counted loop reads from global memory in
place, read into a register array instead, which `wrap` fills a trip ahead.

A rolled reduction (`k_roll`) reads its operator where it lies, and the
products of a trip take the load's value at once.  Gathered into a register
array at the head of the trip, the reads are a transfer `wrap` issues for the
next trip right behind this trip's reads of the array, and for the first trip
ahead of the loop (`pir/trips.py`).
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import re
from pathlib import Path

import pytest

from tensorforge.backend.pir import verify, walk
from tensorforge.backend.pir.build import IRBuilder
from tensorforge.backend.pir.core import (INDEX, SIZE, BufferType, Effect,
                                          MemSpace, Op, ScalarType,
                                          Uniformity)
from tensorforge.backend.pir.trips import TRIP_SLOTS, trip_reads
from tensorforge.backend.pir.wrap import wrap_loads
from tensorforge.common.basic_types import Datatype
from tensorforge.common.context import Context
from tensorforge.common.options import Options, registry
from tensorforge.generators.generator import Generator
from tensorforge.reference import kernel_eval as ke

CASES = Path(__file__).resolve().parent / 'cases'
F32 = Datatype.F32
ROLLED = ['addressing_none', 'local_flux']


def _build(case, target='cuda/sm_86', **options):
    path = next(CASES.rglob(f'{case}.py'))
    spec = importlib.util.spec_from_file_location('tf_trips__' + case, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    backend, arch = target.split('/')
    ctx = Context(arch=arch, backend=backend,
                  fp_type=getattr(mod, 'DTYPE', None),
                  options=Options(**options))
    gen = Generator(mod.descr_list(), ctx)
    with contextlib.redirect_stdout(io.StringIO()):
        gen.generate()
    return gen.get_kernel(), ke.launch_geometry(gen.get_launcher())


# --------------------------------------------------------------------------- #
# the generated code
# --------------------------------------------------------------------------- #

def test_off_by_default():
    assert registry()['trip_reads'].default is False
    src, _ = _build('addressing_none', k_roll=2)
    assert '_trip[' not in src


@pytest.mark.parametrize('name', ROLLED)
def test_a_trip_reads_the_operator_from_registers_filled_a_trip_ahead(name):
    """The array is declared and filled for the first trip ahead of the
    loop; in the loop the trip reads its slots first and fills them for the
    trip after right behind, at the counter one on, held at the last trip."""
    src, _ = _build(name, k_roll=2, trip_reads=True)
    loops = [m.start() for m in re.finditer(r'#pragma unroll 2\n', src)]
    assert loops, 'no rolled loop'
    head = src[:loops[0]]
    arrays = re.findall(r'float (v\d+_trip)\[(\d+)\]\{\};', head)
    assert arrays, 'no register array ahead of the rolled loop'
    name, slots = arrays[-1]
    assert re.search(rf'{name}\[0_i32\] = v\d+_tripRead;', head), (
        'the first trip is not filled ahead of the loop')
    body = src[loops[0]:]
    body = body[:body.index('\n          }\n')]
    lines = [line.strip() for line in body.splitlines()]
    reads = [k for k, line in enumerate(lines)
             if re.fullmatch(rf'float v\d+_data = {name}\[\d+_i32\];', line)]
    fills = [k for k, line in enumerate(lines)
             if re.fullmatch(rf'{name}\[\d+_i32\] = v\d+_tripRead;', line)]
    assert len(reads) == len(fills) == int(slots), (reads, fills)
    assert max(reads) < min(fills), 'a slot filled before it is read'
    ahead = next(line for line in lines if 'tripAhead =' in line)
    assert '+ 1' in ahead
    # All slots read together, ahead of the trip's arithmetic, and the fill
    # for the next trip right behind them: it has the whole trip ahead.
    assert reads == list(range(reads[0], reads[0] + len(reads))), reads
    assert fills[0] - max(reads) <= 2, (
        'the fill does not follow the reads: it is to have the trip ahead')
    assert not any(re.search(r'\b(?:s\d+|ir\d+|r\d+)\[', line)
                   for line in lines[:reads[0]]), (
        'the trip computes before it reads its slots')


@pytest.mark.parametrize('name', ROLLED)
def test_it_computes_what_the_operator_read_in_place_computes(name):
    """Over three elements of one multiplication, and with a mask holding
    the middle one back, which the rolled loop is inside of."""
    for preset in (None, {'flags0': [1, 0, 1]}):
        mems = []
        for options in ({}, {'trip_reads': True}):
            src, (lanes, _) = _build(name, k_roll=2, **options)
            races = []
            mems.append(ke.evaluate_wave(src, lanes, seed=7,
                                         globals_only=True, races=races,
                                         elements=3, preset=preset))
            assert races == [], races
        plain, gathered = mems
        assert gathered.keys() == plain.keys()
        worst = max(abs(gathered[k] - plain[k]) for k in plain)
        assert worst < 1e-5, (name, preset, worst)


def test_with_a_second_stage_as_well():
    src, (lanes, _) = _build('local_flux', k_roll=2, trip_reads=True,
                             enable_multibuffer=True)
    assert '_trip[' in src
    races = []
    gathered = ke.evaluate_wave(src, lanes, seed=7, globals_only=True,
                                races=races, elements=3)
    plain_src, _ = _build('local_flux', k_roll=2, enable_multibuffer=True)
    plain = ke.evaluate_wave(plain_src, lanes, seed=7, globals_only=True,
                             elements=3)
    assert races == []
    assert max(abs(gathered[k] - plain[k]) for k in plain) < 1e-5


def test_without_the_wrap_nothing_is_gathered():
    """The array is for the wrap to fill ahead; without it the loads would
    stand at the head of the trip and wait there."""
    src, _ = _build('addressing_none', k_roll=2, trip_reads=True,
                    enable_wrap_loads=False)
    assert '_trip[' not in src


def _unnamed(src):
    """The kernel without its options line and its name, which hash them."""
    src = re.sub(r'// options:.*', '', src)
    return re.sub(r'kernel_[0-9a-f]+', 'kernel', src)


@pytest.mark.parametrize('name,options', [
    ('addressing_none', {}),
    ('trans_a', {'k_roll': 2}),
    ('sqrt', {'k_roll': 2}),
], ids=['unrolled', 'rolled-from-shared-memory', 'loops-unrolled-whole'])
def test_a_loop_without_reads_in_place_is_left_alone(name, options):
    """Fully unrolled, the reduction has no counted loop to gather in;
    `trans_a` rolls one that reads both operands from shared memory; and the
    loops of an elementwise kernel, which read global memory a trip, are
    unrolled whole, which the compiler schedules as one."""
    plain, _ = _build(name, **options)
    gathered, _ = _build(name, trip_reads=True, **options)
    assert _unnamed(plain) == _unnamed(gathered)


# --------------------------------------------------------------------------- #
# the pass
# --------------------------------------------------------------------------- #

class Trips:
    """A counted loop reading a slice of a batch-constant operand a trip,
    and writing the products into an element's output."""

    def __init__(self, *, lane=True, write_operand=False, indirect=False,
                 loads=1):
        b = self.b = IRBuilder(fptype=F32, arena='shrMem')
        k = b.extern_value('element', SIZE, uniform=Uniformity.MULT, hint='k')
        self.operand = b.decl_expr(
            'const float *const __restrict__ glb_m0', '&m0[0]',
            BufferType(F32, (64,), MemSpace.GLOBAL, readonly=True), 'm0',
            kind=Effect.READ, space=MemSpace.GLOBAL, hint='glb_m0',
            extern='glb_m0')
        out = b.decl_expr(
            'float *const __restrict__ glb_m1', '&m1[{0} * 8]',
            BufferType(F32, (8,), MemSpace.GLOBAL), 'm1', args=(k,),
            kind=Effect.READ, space=MemSpace.GLOBAL, hint='glb_m1',
            extern='glb_m1')
        tid = b.thread_id('x')
        with b.for_(0, 8, 1, unroll=2, hint='kt') as loop:
            kt = loop.induction
            index = b.op('mul', INDEX, kt, 8, hint='a')
            if lane:
                index = b.op('add', INDEX, index, tid, hint='a')
            if indirect:
                index = b.op('add', INDEX, tid,
                             b.load(out, kt, type_=INDEX, hint='i'),
                             hint='a')
            a = b.load(self.operand, index, hint='op')
            for more in range(1, loads):
                at = b.op('add', INDEX, index, more, hint='a')
                a = b.op('add', ScalarType(F32), a,
                         b.load(self.operand, at, hint='op'), hint='s')
            if write_operand:
                b.store(self.operand, a, kt)
            b.store(out, b.op('mul', ScalarType(F32), a, 2.0, hint='p'), kt)
        self.body = b.finish()
        verify(self.body)

    def gather(self):
        self.report = []
        out = trip_reads(self.body, self.b.scratch, report=self.report)
        verify(out)
        return out


def _loop(body):
    return next(s for s, _ in walk(tuple(body)) if s.op is Op.FOR)


def _arrays(body):
    return [s for s, _ in walk(tuple(body)) if s.op is Op.ALLOC
            and s.target[0].type.space is MemSpace.REGISTER]


def test_the_trip_reads_its_slot_and_the_array_is_declared_ahead():
    t = Trips()
    out = t.gather()
    arrays = _arrays(out)
    assert len(arrays) == 1 and arrays[0].target[0].type.shape == (1,)
    assert out.index(arrays[0]) < out.index(_loop(out))
    stmts = _loop(out).regions[0].body
    loads = [s for s in stmts if s.op is Op.LOAD]
    assert [s.accesses[0].space for s in loads] == [MemSpace.GLOBAL,
                                                    MemSpace.REGISTER]
    store = next(s for s in stmts if s.op is Op.STORE)
    assert store.args[0] is arrays[0].target[0]
    assert stmts.index(store) < stmts.index(loads[1])
    assert any(line.startswith('+ trip reads: 1') for line in t.report)


def test_the_wrap_fills_the_array_for_the_trip_after():
    t = Trips()
    out = wrap_loads(t.gather(), t.b.scratch, report=t.report)
    verify(out)
    loop = _loop(out)
    ahead = [s for s in out[:out.index(loop)] if s.op is Op.LOAD
             and s.accesses[0].space is MemSpace.GLOBAL]
    assert len(ahead) == 1, 'the first trip is not read ahead of the loop'
    stmts = loop.regions[0].body
    global_loads = [s for s in stmts if s.op is Op.LOAD
                    and s.accesses[0].space is MemSpace.GLOBAL]
    slot_reads = [s for s in stmts if s.op is Op.LOAD
                  and s.accesses[0].space is MemSpace.REGISTER]
    assert len(global_loads) == 1 and len(slot_reads) == 1
    assert stmts.index(slot_reads[0]) < stmts.index(global_loads[0])
    assert any(line.startswith('+') and line.endswith('in a counted loop')
               for line in t.report), t.report


@pytest.mark.parametrize('kind,why', [
    (dict(lane=False), 'the lanes read one element'),
    (dict(write_operand=True), 'a buffer written in the loop'),
    (dict(indirect=True), 'an index the loop computes by more than'),
])
def test_what_is_left_in_place(kind, why):
    """A load all lanes read alike -- a scalar load on AMD -- one from a
    buffer the loop writes, and one whose index the loop loads: none of them
    is something to fill a trip ahead."""
    t = Trips(**kind)
    out = t.gather()
    assert out is t.body
    assert any(why in line for line in t.report), t.report


def test_a_trip_reading_more_than_an_array_takes_is_left_in_place():
    """A trip of more loads than `TRIP_SLOTS` hides their latency among
    themselves, and an array as wide would hold as many registers through
    the trip; one of as many as it takes gets an array of that width."""
    t = Trips(loads=TRIP_SLOTS + 1)
    assert t.gather() is t.body
    assert any(f'{TRIP_SLOTS + 1} loads a trip' in line
               for line in t.report), t.report
    t = Trips(loads=TRIP_SLOTS)
    arrays = _arrays(t.gather())
    assert [a.target[0].type.shape for a in arrays] == [(TRIP_SLOTS,)]
