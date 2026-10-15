# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""`trip_shards`: what a trip of a counted loop reads from global memory in
place, copied into the multiplication's shared memory a shard a trip.

The run of the operand a trip's lanes read is copied by those lanes, an
element each, with the asynchronous copy, into a window of the
multiplication's own arena; `wrap` issues the copy a trip ahead, and the
barrier between the copy and the reads is the multiplication's -- no block
barrier, since no other multiplication reads the window (`pir/trips.py`).
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import re
from pathlib import Path

import pytest

from tensorforge.backend.pir import verify
from tensorforge.backend.pir.build import IRBuilder
from tensorforge.backend.pir.core import (INDEX, BOOL, BufferType, Effect,
                                          MemSpace, Op, ScalarType)
from tensorforge.backend.pir.shards import _Indices
from tensorforge.backend.pir.trips import _trip_span
from tensorforge.common.basic_types import Datatype
from tensorforge.common.context import Context
from tensorforge.common.options import Options, registry
from tensorforge.generators.generator import Generator
from tensorforge.reference import kernel_eval as ke

CASES = Path(__file__).resolve().parent / 'cases'
F32 = Datatype.F32


def _build(case, target='cuda/sm_86', **options):
    path = next(CASES.rglob(f'{case}.py'))
    spec = importlib.util.spec_from_file_location('tf_shards__' + case, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # Unprepared, which is what these are about.  A re-encoded operand is
    # read through its own accessor, and the trip machinery holds a
    # *coordinate* read in registers -- the two do not combine today:
    # `local_flux` at `k_roll=2` has 28 trip slots without `prepare_operands`
    # and none with it.  Worth knowing, and not worth hiding inside a test
    # that is about something else.
    options.setdefault('prepare_operands', False)
    backend, arch = target.split('/')
    ctx = Context(arch=arch, backend=backend,
                  fp_type=getattr(mod, 'DTYPE', None),
                  options=Options(**options))
    gen = Generator(mod.descr_list(), ctx)
    with contextlib.redirect_stdout(io.StringIO()):
        gen.generate()
    return gen.get_kernel(), ke.launch_geometry(gen.get_launcher())


def _rolled(src):
    """The first rolled loop's body, to its closing brace."""
    start = src.index('#pragma unroll 2\n')
    body = src[start:]
    return body[:body.index('\n          }\n')]


def test_off_by_default():
    assert registry()['trip_shards'].default is False
    src, _ = _build('addressing_none', k_roll=2)
    assert 'tripShard' not in src


@pytest.mark.parametrize('name', ['addressing_none', 'local_flux'])
def test_a_trip_reads_its_shard_from_the_multiplications_window(name):
    """The window is in the multiplication's arena and filled for the first
    trip ahead of the loop; in the loop the trip waits, meets its lanes, reads
    the shard and copies the next trip's -- no block barrier anywhere."""
    src, _ = _build(name, k_roll=2, trip_shards=True)
    windows = re.findall(r'float \* (v\d+_tripShard) = &localShrMem0\[\d+\];',
                         src)
    assert windows, 'no window in the multiplication\'s arena'
    loop = _rolled(src)
    head = src[:src.index(loop)]
    window = windows[0]
    assert re.search(rf'__pipeline_memcpy_async\(&{window}\[', head), (
        'the first trip is not copied ahead of the loop')
    lines = [line.strip() for line in loop.splitlines()]
    wait = lines.index('__pipeline_wait_prior(0);')
    reads = [k for k, line in enumerate(lines) if f'{window}[' in line
             and 'memcpy' not in line]
    copies = [k for k, line in enumerate(lines)
              if line.startswith(f'__pipeline_memcpy_async(&{window}[')]
    assert reads and copies and wait < min(reads) < min(copies), (
        wait, reads, copies)
    assert any(line.startswith('__syncwarp') for line in lines[wait:reads[0]])
    assert '__syncthreads' not in loop
    ahead = next(line for line in lines if 'tripAhead =' in line)
    assert '+ 1' in ahead


def test_with_two_stages_a_trip_waits_for_its_own_shard_alone():
    """With `enable_multibuffer` the next trip's shard is copied into the
    other stage at the head of the trip, and the trip waits for its own
    behind that copy: `__pipeline_wait_prior(1)`, the copy just issued still
    in flight while the trip computes."""
    src, _ = _build('local_flux', k_roll=2, trip_shards=True,
                    enable_multibuffer=True)
    loop = _rolled(src)
    lines = [line.strip() for line in loop.splitlines()]
    copies = [k for k, line in enumerate(lines)
              if line.startswith('__pipeline_memcpy_async(&')
              and 'tripShard' in line]
    wait = lines.index('__pipeline_wait_prior(1);')
    assert copies and max(copies) < wait, (copies, wait)
    assert '__pipeline_wait_prior(0);' not in loop


@pytest.mark.parametrize('name', ['addressing_none', 'local_flux'])
@pytest.mark.parametrize('extra', [{}, {'enable_multibuffer': True}],
                         ids=['one-stage', 'two-stages'])
def test_it_computes_what_the_operator_read_in_place_computes(name, extra):
    """Over three elements, and with a mask holding the middle one back."""
    for preset in (None, {'flags0': [1, 0, 1]}):
        mems = []
        for options in ({}, {'trip_shards': True}):
            src, (lanes, _) = _build(name, k_roll=2, **extra, **options)
            races = []
            mems.append(ke.evaluate_wave(src, lanes, seed=7,
                                         globals_only=True, races=races,
                                         elements=3, preset=preset))
            assert races == [], races
        plain, staged = mems
        assert staged.keys() == plain.keys()
        assert max(abs(staged[k] - plain[k]) for k in plain) < 1e-5


def test_a_multiplication_wider_than_a_warp_meets_on_its_named_barrier():
    """At 64 lanes the multiplication's barrier is a named one, and the
    shard is copied by both of its warps."""
    src, (lanes, mults) = _build('local_flux', k_roll=2, trip_shards=True,
                                 lanes_per_mult=64)
    assert lanes == 64
    loop = _rolled(src)
    assert 'barrier.sync' in loop and '__syncthreads' not in loop
    plain, _ = _build('local_flux', k_roll=2, lanes_per_mult=64)
    for program in (src, plain):
        races = []
        ke.evaluate_wave(program, lanes, seed=7, races=races, elements=2,
                         mults=mults, block=True)
        assert races == [], races
    got = ke.evaluate_wave(src, lanes, seed=7, globals_only=True, elements=2)
    want = ke.evaluate_wave(plain, lanes, seed=7, globals_only=True,
                            elements=2)
    assert max(abs(got[k] - want[k]) for k in want) < 1e-5


@pytest.mark.parametrize('where', ['hip/gfx942', 'cuda/sm_70'])
def test_only_where_the_target_has_the_asynchronous_copy(where):
    src, _ = _build('addressing_none', target=where, k_roll=2,
                    trip_shards=True)
    assert 'tripShard' not in src


def test_a_shard_takes_the_reads_ahead_of_the_register_array():
    """With both asked for, the loads go into the window, and nothing is
    left for a register array."""
    src, _ = _build('local_flux', k_roll=2, trip_shards=True,
                    trip_reads=True)
    assert 'tripShard' in src and '_trip[' not in src


def test_the_lanes_past_the_run_copy_its_last_element_again():
    """56 elements over 32 lanes: a window of 64, and the second hop's
    source held at the run's last element -- a copy with no predicate, which
    the wrap moves."""
    src, _ = _build('local_flux', k_roll=2, trip_shards=True)
    loop = _rolled(src)
    assert re.search(r'= min\(v\d+_hop, 55\);', src), 'no source held'
    assert re.search(r'float \* v\d+_tripShard = &localShrMem0\[\d+\];', src)
    window = re.search(r'(v\d+_tripShard) = &localShrMem0', src).group(1)
    assert not re.search(rf'if \([^)]*\) \{{\s*__pipeline_memcpy_async\(&'
                         rf'{window}', loop), 'a predicated copy'


# --------------------------------------------------------------------------- #
# the span a trip's lanes read
# --------------------------------------------------------------------------- #

def _loop_with(index_of, predicate_of=None, threads=4):
    """A counted loop over eight trips with one load of a bound operand at
    `index_of(builder, lane, trip)`."""
    b = IRBuilder(fptype=F32, arena='shrMem')
    operand = b.decl_expr(
        'const float *const __restrict__ glb_m0', '&m0[0]',
        BufferType(F32, (128,), MemSpace.GLOBAL, readonly=True), 'm0',
        kind=Effect.READ, space=MemSpace.GLOBAL, hint='glb_m0',
        extern='glb_m0')
    lane = b.thread_id('x')
    with b.for_(0, 8, 1, unroll=2, hint='kt') as loop:
        kt = loop.induction
        pred = predicate_of(b, lane, kt) if predicate_of else None
        b.load(operand, index_of(b, lane, kt), hint='op', predicate=pred)
    body = b.finish()
    verify(body)
    stmt = next(s for s in body if s.op is Op.FOR)
    load = next(s for s in stmt.regions[0].body if s.op is Op.LOAD)
    return body, stmt, load


def test_the_span_is_a_run_moving_by_a_step():
    body, loop, load = _loop_with(
        lambda b, lane, kt: b.op('add', INDEX, b.op('add', INDEX, lane, 3),
                                 b.op('mul', INDEX, kt, 8)))
    span = _trip_span(load, loop.regions[0].args[0], range(0, 8), 4,
                      _Indices(body, {'x': 4}))
    assert span == (3, 8, 4)


def test_a_predicate_the_lanes_decide_holds_lanes_out():
    body, loop, load = _loop_with(
        lambda b, lane, kt: b.op('add', INDEX, lane,
                                 b.op('mul', INDEX, kt, 8)),
        lambda b, lane, kt: b.op('lt', BOOL, lane, 3))
    span = _trip_span(load, loop.regions[0].args[0], range(0, 8), 4,
                      _Indices(body, {'x': 4}))
    assert span == (0, 8, 3)


def test_a_run_that_does_not_move_by_a_step_is_none():
    body, loop, load = _loop_with(
        lambda b, lane, kt: b.op('add', INDEX, lane,
                                 b.op('mul', INDEX, kt, kt)))
    assert _trip_span(load, loop.regions[0].args[0], range(0, 8), 4,
                      _Indices(body, {'x': 4})) is None
