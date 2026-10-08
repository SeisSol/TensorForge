# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""`preload_shards`: the shards of the batch-constant operands that the
block's shared memory holds (`pir.shards`).

Two things are pinned.  That a kernel computes with its shards held what it
computes without -- on the host oracle with every multiplication of the
block taking part, since the copy is the block's and a run of the first
multiplication alone sees only that multiplication's share of it.  And what
is held: the shards the budget admits, least cover first, each keeping the
alignment its operand gives it, copied by the whole block in as few copies
as their layout allows and behind a barrier the oracle cannot do without.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import re
import warnings
from pathlib import Path

import pytest

from tensorforge.backend.pir import shards as S
from tensorforge.backend.pir.build import IRBuilder
from tensorforge.backend.pir.core import (INDEX, SIZE, BufferType, Effect,
                                          MemSpace, Uniformity)
from tensorforge.common.basic_types import Datatype
from tensorforge.common.context import Context
from tensorforge.common.options import Options
from tensorforge.generators.generator import Generator
from tensorforge.generators.lanes import LaneConfig
from tensorforge.reference import kernel_eval as ke

CASES = Path(__file__).resolve().parent / 'cases'

#: Held: a shard buffer, the loads that read it, and its copy.
SHARD = re.compile(r'\bfloat \* (v\d+_shard) = &totalShrMem\[')
_SYNC = re.compile(r'__syncthreads\(\);')


def _module(stem: str):
    path = next(CASES.rglob(f'{stem}.py'))
    spec = importlib.util.spec_from_file_location('tf_shard__' + stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _build(stem: str, arch: str = 'sm_86', backend: str = 'cuda',
           report: list = None, lanes: LaneConfig = None, **options):
    """The kernel, its launcher's geometry, and -- where asked for -- what
    the pass reports."""
    mod = _module(stem)
    if report is not None:
        options['ir_debug'] = 'report'
    ctx = Context(arch=arch, backend=backend,
                  fp_type=getattr(mod, 'DTYPE', None),
                  options=Options(**options))
    gen = Generator(mod.descr_list(), ctx, attrs=getattr(mod, 'ATTRS', None),
                    lanes=lanes)
    out = io.StringIO()
    with warnings.catch_warnings(), contextlib.redirect_stdout(out):
        warnings.simplefilter('ignore')
        gen.generate()
    if report is not None:
        report.extend(line[len('shards: '):]
                      for line in out.getvalue().splitlines()
                      if line.startswith('shards: '))
    return gen.get_kernel(), ke.launch_geometry(gen.get_launcher())


def _written(src: str, geometry, elements: int = 1) -> dict:
    """What the kernel leaves in the operands it writes, every
    multiplication of the block taking part."""
    lanes, mults = geometry
    signature = re.search(r'\bkernel_\w+\(([^)]*)\)', src).group(1)
    outputs = {p.split()[-1] for p in signature.split(',')
               if '*' in p and 'const' not in p}
    assert outputs, 'the kernel writes no operand'
    mem = ke.evaluate_wave(src, lanes, seed=17, globals_only=True,
                           mults=mults, block=True, elements=elements)
    written = {k: v for k, v in mem.items() if k[0] in outputs}
    assert written, 'nothing written to compare'
    return written


def _reads(src: str) -> int:
    """How many loads read a shard buffer."""
    held = SHARD.findall(src)
    return sum(len(re.findall(rf'(?<!\w){b}\[', line))
               for b in held for line in src.splitlines()
               if not re.match(rf'\s*{b}\[', line))


# --------------------------------------------------------------------------- #
# What the kernel computes
# --------------------------------------------------------------------------- #

#: Cases with a batch-constant operand read at an index every thread knows,
#: whose kernels the oracle runs.
HELD = ['addressing_none', 'known_zero_rows']


@pytest.mark.parametrize('budget', [0, 512], ids=['all', 'some'])
@pytest.mark.parametrize('stem', HELD + ['known_zero_tiles'])
def test_the_kernel_computes_what_it_computes_without(stem, budget):
    """Bit for bit: a shard holds the operand's own numbers in the operand's
    own order, and the loads read them there.  `known_zero_tiles` copies
    more elements than its block has threads, a few for each."""
    plain, plain_at = _build(stem)
    held, held_at = _build(stem, preload_shards=True,
                           preload_shard_budget=budget)
    assert SHARD.search(held) and _reads(held), 'nothing held'
    assert _written(held, held_at) == _written(plain, plain_at)


def test_a_wide_load_is_held_whole():
    """Four rows a lane, read as one vector: the shard holds the four."""
    wide = LaneConfig(num_threads=4, num_active_threads=4, lead_width=4)
    plain, plain_at = _build('addressing_none', lanes=wide)
    held, held_at = _build('addressing_none', lanes=wide, preload_shards=True,
                           preload_shard_budget=1 << 20)
    assert re.search(r'\*\([^()]*, 4>\*\)&v\d+_shard\[', held)
    assert _written(held, held_at) == _written(plain, plain_at)


@pytest.mark.parametrize('stem', HELD)
def test_the_copy_is_ordered_ahead_of_the_reads(stem):
    """No race with the barrier behind the copy, and one without it: the
    elements a multiplication reads were copied in by threads of the
    others."""
    src, (lanes, mults) = _build(stem, preload_shards=True)
    assert mults > 1, 'one multiplication per block: nobody else copies'

    def races(text):
        found = []
        ke.evaluate_wave(text, lanes, seed=3, races=found, elements=2,
                         mults=mults, block=True)
        return found

    assert races(src) == []
    lines = src.split('\n')
    loop = next(k for k, line in enumerate(lines) if 'batchId0 = ' in line)
    barrier = max(k for k in range(loop) if _SYNC.search(lines[k]))
    without = '\n'.join(line for k, line in enumerate(lines) if k != barrier)
    assert races(without)


@pytest.mark.parametrize('stem', HELD)
def test_every_thread_of_the_block_copies_its_share(stem):
    """The copy steps by the threads of the block the launcher starts, not
    by the block the body was first built for: a block wider than its
    stride copies an element twice, a narrower one leaves some out."""
    src, (lanes, mults) = _build(stem, preload_shards=True)
    flat = re.search(r'int32_t (v\d+_flat) = threadIdx\.x \+ \(threadIdx\.y '
                     r'\* (\d+)\);', src)
    assert flat and int(flat.group(2)) == lanes
    steps = re.findall(rf'{flat.group(1)} < (\d+); v\d+_s \+= (\d+)\)', src)
    guards = re.findall(rf'if \({flat.group(1)} < (\d+)\)', src)
    assert steps or guards
    assert all(int(step) == lanes * mults for _, step in steps)
    assert all(int(n) <= lanes * mults for n in guards)


def test_the_budget_bounds_what_is_held():
    """Bytes of the operand, at most, and more of it for more bytes."""
    held = []
    for budget in (256, 512, 1024):
        report = []
        _build('addressing_none', preload_shards=True,
               preload_shard_budget=budget, report=report)
        line = next(r for r in report if r.startswith('+ '))
        count, total = map(int, re.search(r'(\d+) of (\d+) element',
                                          line).groups())
        assert count * 4 <= budget
        held.append(count)
    assert held == sorted(held) and held[0] < held[-1] <= total


def test_the_shards_leave_the_sm_as_many_blocks_as_it_holds_without():
    """By default the shards take what each block can have more and the SM
    still hold as many blocks -- by shared memory and threads, the block
    sized as one that holds a copy of operator data.  `local_flux`'s
    operators are far more than that on sm_86, so the budget is what
    decides."""
    report = []
    src, (lanes, mults) = _build('local_flux', report=report,
                                 preload_shards=True)
    held = [int(m.group(1)) for m in
            (re.search(r'(\d+) of (\d+) element', r) for r in report) if m]
    total = int(re.search(r'of (\d+) element', report[-1]).group(1))
    assert 0 < held[-1] < total
    shared = int(re.search(r'// launch: .*?, (\d+) B shared', src).group(1))
    arena = int(re.search(r'localShrMem0 = &totalShrMem\[\d+ \* threadIdx\.y '
                          r'\+ (\d+)\]', src).group(1))
    hw = Context(arch='sm_86', backend='cuda',
                 fp_type=Datatype.F32).target.hw

    def blocks(per_block):
        return min(hw.max_block_per_sm,
                   hw.max_local_mem_size_per_block // per_block,
                   hw.max_threads_per_sm // (lanes * mults))

    assert blocks(shared) == blocks(shared - arena * 4)


def test_nothing_to_hold_leaves_the_kernel_as_it_was():
    """`local_flux` stages all of its operators on gfx942: nothing is left
    for a shard, and the kernel is the one without the option."""
    def normal(src):
        return re.sub(r'kernel_\w*?_[0-9a-f]{16}', 'K', '\n'.join(
            line for line in src.splitlines() if '// options:' not in line))

    report = []
    with_, _ = _build('local_flux', 'gfx942', 'hip', report=report,
                      preload_shards=True)
    without, _ = _build('local_flux', 'gfx942', 'hip')
    assert report and all(r.startswith('- ') for r in report)
    assert normal(with_) == normal(without)


# --------------------------------------------------------------------------- #
# What a load to hold is
# --------------------------------------------------------------------------- #

def _section(index='lane', written=False, trips=0):
    """A batch loop reading a batch-constant operand bound ahead of it, at
    `index`: the lane's own element, every lane the same one, or one of the
    element's own; `written` stores to the operand in the loop as well, and
    `trips` reads it in a counted loop too, a lane block every trip."""
    b = IRBuilder(fptype=Datatype.F32, arena='shrMem')

    def binding(name, base, args=(), writable=False, offset=''):
        return b.decl_expr(
            f'const float *const __restrict__ {name}', f'&{base}[{offset}0]',
            BufferType(Datatype.F32, (64,), MemSpace.GLOBAL,
                       readonly=not writable), base, args=args,
            kind=Effect.READ, space=MemSpace.GLOBAL, hint=name, extern=name)

    operand = binding('glb_m1', 'm1')
    count = b.extern_value('numElements0', SIZE, hint='count')
    with b.for_('start', count, 'stride', hint='batchId0', index_type=SIZE,
                uniform=Uniformity.MULT) as f:
        f._next_index = b.op('add', SIZE, f.induction, 'stride', hint='next')
        out = binding('glb_m0', 'm0', args=(f.induction,), writable=True,
                      offset='{0} * 64 + ')
        lane = b.thread_id('x')
        at = {'lane': lane, 'uniform': 3,
              'element': b.op('rem', INDEX, f.induction, 64)}[index]
        b.store(out, b.load(operand, at, hint='a'), lane)
        if written:
            b.store(binding('glb_w', 'm1', writable=True), 0.0, lane)
        if trips:
            with b.for_(0, trips, 1, hint='t') as t:
                at = b.op('add', INDEX, lane,
                          b.op('mul', INDEX, t.induction, 32))
                b.store(out, b.load(operand, at, hint='t'), lane)
    return b, b.finish()


@pytest.mark.parametrize('index, written, expected', [
    ('lane', False, '+ v0_glb_m1: 1 of 1 shard(s), 32 of 32 element(s)'),
    ('uniform', False, '- no load to hold'),
    ('element', False, '- 1 load(s): an index not known per thread '
                       '(the element)'),
    ('lane', True, '- 1 load(s): the kernel writes the operand')])
def test_a_load_held_reads_an_operand_nothing_writes_at_an_index_each_lane_knows(
        index, written, expected):
    """A lane reads its own element: held.  Every lane one element: a
    scalar load, left alone.  An element's own: a shard per element is no
    shard.  An operand the kernel writes: what a shard holds would go
    stale."""
    b, body = _section(index, written)
    report = []
    out = S.shard_loads(body, b.scratch, arena='totalShrMem', align=4,
                        budget=1024, mults=2, threads=32,
                        fptype=Datatype.F32, report=report)
    assert expected in report
    assert (out is body) == (not expected.startswith('+'))


def test_a_load_over_the_trips_of_a_loop_takes_no_lane_block_with_it():
    """The load in the loop reads two lane blocks, one of them the one the
    load ahead of it reads.  Its shard is both and does not fit; the lane
    block alone does, and is held for the load that reads only it."""
    b, body = _section(trips=2)
    report = []
    S.shard_loads(body, b.scratch, arena='totalShrMem', align=4, budget=40,
                  mults=2, threads=32, fptype=Datatype.F32, report=report)
    assert report[-1] == '+ v0_glb_m1: 1 of 2 shard(s), 32 of 96 element(s)'


# --------------------------------------------------------------------------- #
# What is held, and how it is copied
# --------------------------------------------------------------------------- #

class _Buffer:
    """What `shards` reads of an operand: its identity."""

    def __init__(self, ident):
        self.id = ident


def _load(cover, position=0):
    return S._Load(stmt=None, buffer=None, lo=0, hi=0, cover=cover,
                   position=position)


def _shard(buffer, lo, size, cover=0, position=0):
    return S._Shard(buffer, lo, lo + size - 1,
                    [_load(cover, position)])


def test_the_shards_whose_loads_wait_soonest_go_first():
    """Least cover first, then in the loop's order, each as long as it
    fits."""
    a = _Buffer(1)
    late, soon, sooner = (_shard(a, 0, 32, cover=9), _shard(a, 64, 32, cover=2),
                          _shard(a, 128, 32, cover=0, position=5))
    assert S._choose([late, soon, sooner], free=64, align=4) == [sooner, soon]
    assert S._choose([late, soon, sooner], free=63, align=4) == [sooner]


def test_a_shard_is_counted_with_the_gap_its_alignment_takes():
    """The gap ahead of it in its buffer, where `_layout` puts it, and a
    buffer after the first with what its start's alignment may take."""
    a, b = _Buffer(1), _Buffer(2)
    first, second = _shard(a, 1, 30), _shard(a, 33, 30)
    assert S._choose([first], free=31, align=4) == [first]
    assert S._choose([first], free=30, align=4) == []
    assert S._choose([first, second], free=63, align=4) == [first, second]
    assert S._choose([first, second], free=62, align=4) == [first]
    size, _ = S._layout([first, second], align=4)[a.id]
    assert size == 63
    other = _shard(b, 0, 30)
    assert S._choose([first, other], free=64, align=4) == [first, other]
    assert S._choose([first, other], free=63, align=4) == [first]


def test_a_shard_keeps_the_alignment_its_operand_gives_it():
    """A shard starts in its buffer where its first element is on the
    alignment it has in the operand, so a wide load stays as aligned."""
    a = _Buffer(1)
    shards = [_shard(a, lo, 21) for lo in (3, 59, 115)]
    size, placed = S._layout(shards, align=4)[a.id]
    assert [base % 4 for _, base in placed] == [lo % 4 for lo in (3, 59, 115)]
    assert all(b1 >= b0 + 21 for (_, b0), (_, b1) in zip(placed, placed[1:]))
    assert size == placed[-1][1] + 21


def test_runs_at_one_distance_are_one_copy():
    """A lane block of every column -- one length, one distance in the
    operand and in the buffer -- is copied by one copy, and what follows
    each other in both is one run."""
    a = _Buffer(1)
    columns = [(_shard(a, 32 + 56 * k, 24), 24 * k) for k in range(9)]
    assert S._copies(columns) == [S._Copy(32, 0, 24, 9, 56, 24)]
    joined = [(_shard(a, 0, 16), 0), (_shard(a, 16, 16), 16)]
    assert S._copies(joined) == [S._Copy(0, 0, 32)]
    ragged = [(_shard(a, 0, 16), 0), (_shard(a, 40, 8), 16),
              (_shard(a, 80, 16), 24)]
    assert [c.count for c in S._copies(ragged)] == [16, 8, 16]
