# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""RELAYOUT: registers from one distribution over the lanes to another, an
op of the engines layer (`Op.RELAYOUT`, `engines.RELAYOUT_STRATEGIES`).

AMD's relayouts -- the DPP broadcast, the row share, the transposes of the
MFMA tiling -- were calls; they are the op now, written as the same text,
counted as the same figures, merged and inlined as before, and their lanes
read off as before (`pir.demand`).  NVIDIA's: a renaming of registers that
writes no instruction (`relabel`), which is how mma.sync's D is the next
product's A once B's K rows are relabeled (`engines.relabeling`,
`nvidia.d_to_a`), and the warp shuffle (`shfl`).  What is checked here: the
registry, the AMD forms against the calls they replaced, the relabel's
solver on mma.sync's three shapes, a chain of two FP64 products through a
relabel on the host oracle and on the local GPU, and the shuffle.
"""

from __future__ import annotations

import dataclasses
import re
import shutil
import struct
import subprocess
from pathlib import Path

import numpy as np
import pytest

from tensorforge import toolchain
from tensorforge.backend.instructions.compute.primitives import nvidia
from tensorforge.backend.instructions.compute.primitives.amd import relayout as amd
from tensorforge.backend.pir import demand, engines, verify, walk
from tensorforge.backend.pir.build import IRBuilder
from tensorforge.backend.pir.core import (INDEX, Effect, IRError, MemSpace,
                                          Op, ScalarType, Stmt, Uniformity)
from tensorforge.backend.pir.emit import Emitter
from tensorforge.backend.pir.passes import cse
from tensorforge.common.basic_types import Datatype
from tensorforge.common.context import Context
from tensorforge.reference import kernel_eval

F32, F64 = Datatype.F32, Datatype.F64
INCLUDE = Path(__file__).resolve().parents[1] / 'src' / 'tensorforge' / 'include'


class _Metrics:
    def __init__(self):
        self.code = 0
        self.mix = {}

    def record_code(self, units):
        self.code += units

    def record_mix(self, category, issued, code):
        self.mix[category] = self.mix.get(category, 0) + issued


def _emit(body, ctx=None, metrics=None):
    lines = []
    Emitter(lines.append, ctx, metrics).run(body)
    return lines


def _hip(arch='gfx942'):
    return Context(arch=arch, backend='hip', fp_type=F32)


def _atom(m, n, k, mode):
    return next(a for a in nvidia.INSTRS
                if (a.m, a.n, a.k) == (m, n, k) and a.mode == mode)


# -- the registry ------------------------------------------------------------- #

def test_the_strategies_and_their_forms():
    s = engines.RELAYOUT_STRATEGIES
    assert {n: r.form for n, r in s.items()} == {
        'broadcast': 'value', 'movdpp16': 'value', 'movdpp16_kept': 'value',
        'transpose4x4': 'in_place', 'transpose4x4_fused': 'in_place',
        'transpose16x16': 'in_place', 'relabel': 'relabel', 'shfl': 'value'}
    assert all(r.families == engines.AMD_FAMILIES == ('hip', 'sycl')
               for n, r in s.items() if n not in ('relabel', 'shfl'))
    assert s['shfl'].families == ('cuda',) and 'cuda' in s['relabel'].families
    assert s['relabel'].units == 0 and s['broadcast'].units == 1
    assert s['broadcast'].fields == ('threads', 'step', 'lane')
    assert s['movdpp16'].fields == ('row',)
    spec = engines.REGISTRY[Op.RELAYOUT]
    assert spec.effect == Effect.NONE and not spec.movable


def test_amd_s_rows_name_their_strategies():
    """Every row of AMD's table is a strategy of the registry, by name: the
    table says how to reach a layout, the registry how that is written."""
    for row in amd.RELAYOUTS:
        assert row.name in engines.RELAYOUT_STRATEGIES
        strategy = engines.RELAYOUT_STRATEGIES[row.name]
        assert strategy.callee == row.callee and strategy.lossy == row.lossy


# -- AMD's relayouts, as the calls they replaced ------------------------------ #

def _value_body(strategy, params, twice=False, other=None, pure=True):
    """A value relaid by `strategy` -- and, `twice`, the same again, and
    `other`'s parameters once more -- each stored."""
    ctx = _hip()
    b = IRBuilder(fptype=F32, context=ctx)
    out = b.alloc(F32, (64,), MemSpace.GLOBAL, hint='o', extern='m0')
    src = b.alloc(F32, (64,), MemSpace.GLOBAL, hint='g', extern='m1')
    lane = b.thread_id('x')
    x = b.load(src, lane, hint='x')
    row = amd.RELAYOUTS[[r.name for r in amd.RELAYOUTS].index(
        'movdpp16' if strategy.startswith('movdpp16') else strategy)]
    vs = [b.relayout(strategy, params, x, type_=ScalarType(F32), hint='bc',
                     layout=row.produces(**params), pure=pure)]
    if twice:
        vs.append(b.relayout(strategy, params, x, type_=ScalarType(F32),
                             hint='bc', layout=row.produces(**params),
                             pure=pure))
    if other is not None:
        vs.append(b.relayout(strategy, other, x, type_=ScalarType(F32),
                             hint='bc', layout=row.produces(**other),
                             pure=pure))
    for i, v in enumerate(vs):
        b.store(out, v, i)
    return ctx, b, vs


def _as_calls(body):
    """`body` with each relayout as the call it replaced: what `call` and
    `call_stmt` built for it."""
    def call(s):
        if s.op != Op.RELAYOUT:
            return s
        attrs = (('callee', engines.relayout_callee(s)),)
        if s.attr('no_inline'):
            attrs += (('no_inline', True),)
        return Stmt(op=Op.CALL, target=s.target, args=s.args, pure=s.pure,
                    movable=False, effect=s.effect, accesses=s.accesses,
                    attrs=attrs)
    return tuple(call(s) for s in body)


@pytest.mark.parametrize('strategy,params', [
    ('broadcast', dict(threads=64, step=4, lane=3)),
    ('broadcast', dict(threads=16, step=4, lane=1)),
    ('movdpp16', dict(threads=64, row=5)),
    ('movdpp16_kept', dict(threads=64, row=2))])
def test_a_value_relayout_is_the_call_it_was(strategy, params):
    """The text, inlined into its one consumer as the call was, and the code
    and the issue mix: a cross-lane instruction, wide where a broadcast's
    block crosses the rows of sixteen."""
    ctx, b, vs = _value_body(strategy, params)
    body = b.finish()
    now, then = _Metrics(), _Metrics()
    lines = _emit(body, ctx, now)
    assert lines == _emit(_as_calls(body), ctx, then)
    assert (now.code, now.mix) == (then.code, then.mix)
    callee = engines.RELAYOUT_STRATEGIES[strategy].spelled(params)
    assert any(f'{callee}(' in line for line in lines)
    wide = strategy == 'broadcast' and params['threads'] > 16
    assert now.mix.get('xlane.wide' if wide else 'xlane') == 1


def test_the_same_relayout_twice_is_one_and_another_is_two():
    """Shared within the scope, and by CSE, on the callee's parameters as
    the call was on its text: the same broadcast once, another lane apart;
    a relayout the caller says is no pure function, never."""
    p = dict(threads=64, step=4, lane=3)
    _, b, vs = _value_body('broadcast', p, twice=True,
                           other=dict(p, lane=2))
    assert vs[0] is vs[1] and vs[2] is not vs[0]
    _, b, vs = _value_body('movdpp16', dict(threads=64, row=1), twice=True,
                           pure=False)
    assert vs[0] is not vs[1]
    body = cse(b.finish())
    assert len([s for s, _ in walk(body) if s.op == Op.RELAYOUT]) == 2


def test_demand_hands_a_relayout_s_lanes_on_as_the_call_s():
    """`pir.demand` over a broadcast relayout: lane 1 of each block of 16
    of what the FMA in place reads, as `test_demand` finds for the call."""
    from test_demand import REAL, _Use
    t = _Use()
    bc = t.b.relayout('broadcast', dict(threads=32, step=16, lane=1), t.x,
                      type_=REAL)
    t.fma(3, bc)
    assert t.lanes() == 1 << 19


def test_demand_reads_the_lanes_off_the_strategy_s_call():
    """`pir.demand` reads a relayout's lanes as it read the call's."""
    _, b, vs = _value_body('broadcast', dict(threads=64, step=4, lane=3))
    s = next(s for s, _ in walk(b.finish()) if s.op == Op.RELAYOUT)
    assert demand.lane_map(engines.relayout_callee(s))[2] == 64
    _, b, vs = _value_body('movdpp16', dict(threads=64, row=5))
    s = next(s for s, _ in walk(b.finish()) if s.op == Op.RELAYOUT)
    block, lane = demand.lane_map(engines.relayout_callee(s))[2:]
    assert block == 16 and lane(9) == 5


@pytest.mark.parametrize('strategy,count,separate', [
    ('transpose4x4', 4, True), ('transpose4x4_fused', 4, True),
    ('transpose16x16', 16, False)])
def test_a_transpose_is_the_call_statement_it_was(strategy, count, separate):
    """Into four registers declared ahead of it, or in place on its sixteen:
    a read and a write of each, its producer pinned, the text and the
    figures the call statement had."""
    ctx = _hip()
    b = IRBuilder(fptype=F32, context=ctx)
    src = b.alloc(F32, (64,), MemSpace.GLOBAL, hint='g', extern='m1')
    out = b.alloc(F32, (64,), MemSpace.GLOBAL, hint='o', extern='m0')
    lane = b.thread_id('x')
    regs = [b.load(src, b.op('add', INDEX, lane, i), hint='r')
            for i in range(count)]
    if separate:
        outs = [b.declare(ScalarType(F32), hint='tp') for _ in regs]
        stmt = b.relayout_in_place(strategy, outs, regs)
    else:
        outs = regs
        stmt = b.relayout_in_place(strategy, regs)
    for i, v in enumerate(outs):
        b.store(out, v, i)
    body = b.finish()
    assert stmt.attr('written') == count and not stmt.pure
    assert stmt.effect == Effect.WRITE and not stmt.target
    assert [a.base for a in stmt.accesses] == list(outs)
    assert engines.verify_engine(stmt) == []
    # The producers of what it writes keep their names: a reference binds
    # them.
    defs = {x.target[0].id: x for x, _ in walk(body) if x.target}
    assert all(defs[v.id].attr('escapes') for v in outs)
    unsaid = dataclasses.replace(stmt, accesses=())
    assert any('without a read and a write' in d
               for d in engines.verify_engine(unsaid))
    now, then = _Metrics(), _Metrics()
    lines = _emit(body, ctx, now)
    assert lines == _emit(_as_calls(body), ctx, then)
    assert (now.code, now.mix) == (then.code, then.mix)
    callee = engines.RELAYOUT_STRATEGIES[strategy].callee
    assert sum(line.startswith(f'{callee}(') for line in lines) == 1


def test_a_strategy_is_written_for_its_own_backends():
    """AMD's relayouts where AMD's code generation runs -- HIP, and SYCL
    compiled for AMD, which wrote the same calls -- and nowhere else."""
    _, b, _ = _value_body('broadcast', dict(threads=64, step=4, lane=3))
    s = next(s for s, _ in walk(b.finish()) if s.op == Op.RELAYOUT)
    cuda = Context(arch='sm_86', backend='cuda', fp_type=F32)
    with pytest.raises(IRError, match='no spelling on cuda'):
        engines.lower(s, cuda.target, name=str)
    hip = engines.lower(s, _hip().target, name=str).result
    acpp = Context(arch='gfx942', backend='acpp', fp_type=F32)
    assert hip and engines.lower(s, acpp.target, name=str).result == hip


# -- the relabel -------------------------------------------------------------- #

@pytest.mark.parametrize('shape,mode,order', [
    ((16, 8, 8), nvidia.MMAMode.TF32, (0, 2, 1, 3)),
    ((16, 8, 4), nvidia.MMAMode.DIRECT, (0, 2, 1, 3)),
    ((8, 8, 4), nvidia.MMAMode.DIRECT, (0, 1))])
def test_d_is_the_next_a_under_a_relabel_of_k(shape, mode, order):
    """mma.sync's D is the next product's A with no instruction: A's
    registers D's in `order`, and column `2t + b` of D column `t + 4b` of
    A, the rows as they are (`d-to-a-is-a-column-relabel`)."""
    atom = _atom(*shape, mode)
    relabel, got, layouts = nvidia.d_to_a(atom, atom)
    assert got == order
    assert [relabel[1](c) for c in range(8)] == [0, 4, 1, 5, 2, 6, 3, 7]
    assert [relabel[0](r) for r in range(shape[0])] == list(range(shape[0]))
    assert engines.relabel_order(*layouts) == order


def test_a_gap_that_moves_elements_between_lanes_is_no_relabel():
    """A's layout is no relabel of itself transposed: the lanes would
    change, which takes instructions."""
    from tensorforge.backend.instructions.compute.bitlayout import BitLayout
    a = nvidia.a_fragment_bits(_atom(16, 8, 8, nvidia.MMAMode.TF32))
    swapped = BitLayout((a.axes[1] + (a.axes[0][-1],), a.axes[0][:-1]))
    assert engines.relabeling(a, swapped) is None
    assert engines.relabel_order(a, a) == (0, 1, 2, 3)


def _relabel_body():
    b = IRBuilder(fptype=F64)
    regs = [b.declare(ScalarType(F64), hint='c') for _ in range(4)]
    return b, regs


def test_a_relabel_is_copies_and_no_instruction():
    atom = _atom(16, 8, 4, nvidia.MMAMode.DIRECT)
    relabel, order, layouts = nvidia.d_to_a(atom, atom)
    b, regs = _relabel_body()
    a = b.relabel(regs, order, layouts=layouts)
    s = next(s for s, _ in walk(b.finish()) if s.op == Op.RELAYOUT)
    assert s.attr('order') == (0, 2, 1, 3) and s.pure
    assert engines.verify_engine(s) == []
    metrics = _Metrics()
    lines = _emit(b.finish(), Context(arch='sm_90', backend='cuda',
                                      fp_type=F64), metrics)
    assert lines[-4:] == [f'double {a[j]} = {regs[order[j]]};'
                          for j in range(4)]
    assert metrics.code == 0 and not metrics.mix


def test_a_relabel_its_layouts_do_not_allow_is_refused():
    atom = _atom(16, 8, 4, nvidia.MMAMode.DIRECT)
    _, _, layouts = nvidia.d_to_a(atom, atom)
    b, regs = _relabel_body()
    with pytest.raises(IRError, match='an element would change lanes'):
        b.relabel(regs, (0, 1, 2, 3), layouts=layouts)
    with pytest.raises(IRError, match='a permutation'):
        b.relabel(regs, (0, 0, 1, 2))


# -- a chain of two products through a relabel ------------------------------- #

DMMA = _atom(8, 8, 4, nvidia.MMAMode.DIRECT)


def _chain_body(arch='sm_86'):
    """One warp: D1 = A1 B1 on m8n8k4 over FP64, D1 handed on as the A of
    two products of the next K -- B2's rows relabeled to match -- by a
    relabel, D2 = D1 B2 stored.  A1 is 8 x 4, B1 4 x 8, B2 8 x 8, row-major,
    in `m1`, `m2`, `m3`; D2 in `m0`."""
    ctx = Context(arch=arch, backend='cuda', fp_type=F64)
    b = IRBuilder(fptype=F64, context=ctx)
    a1 = b.alloc(F64, (32,), MemSpace.GLOBAL, hint='a', extern='m1')
    b1 = b.alloc(F64, (32,), MemSpace.GLOBAL, hint='b', extern='m2')
    b2 = b.alloc(F64, (64,), MemSpace.GLOBAL, hint='b', extern='m3')
    out = b.alloc(F64, (64,), MemSpace.GLOBAL, hint='o', extern='m0')
    lane = b.thread_id('x')
    q = b.op('div', INDEX, lane, 4, hint='q')
    r = b.op('rem', INDEX, lane, 4, hint='r')
    a = b.load(a1, b.op('add', INDEX, b.op('mul', INDEX, q, 4), r), hint='a')
    bb = b.load(b1, b.op('add', INDEX, b.op('mul', INDEX, r, 8), q),
                hint='b')
    d1 = [b.declare(ScalarType(F64), hint='d', uniform=Uniformity.LANE)
          for _ in range(2)]
    DMMA.asmcall(b, d1, [a], [bb], d1)
    relabel, order, layouts = nvidia.d_to_a(DMMA, DMMA)
    a2 = b.relabel(d1, order, layouts=layouts)
    d2 = [b.declare(ScalarType(F64), hint='e', uniform=Uniformity.LANE)
          for _ in range(2)]
    for block in range(2):
        # Column t of block `block` of A2 is column 2t + block of D1: B2's
        # row of that column, for the lane's k and n.
        row = b.op('add', INDEX, b.op('mul', INDEX, r, 2), block, hint='k')
        bb2 = b.load(b2, b.op('add', INDEX, b.op('mul', INDEX, row, 8), q),
                     hint='b')
        DMMA.asmcall(b, d2, [a2[block]], [bb2], d2)
    for e in range(2):
        at = b.op('add', INDEX, b.op('mul', INDEX, q, 8),
                  b.op('add', INDEX, b.op('mul', INDEX, r, 2), e), hint='o')
        b.store(out, d2[e], at)
    body = b.finish()
    verify(body)
    return ctx, body


def _oracle_source(lines):
    keep = [line for line in lines if not re.match(r'^double m\d+\[', line)]
    return 'void k() {\n' + ''.join(f'  {line}\n' for line in keep) + '}'


def test_the_oracle_multiplies_through_the_relabel():
    """D2 = (A1 B1) B2 to the products' rounding, D1 never leaving the
    registers it was accumulated in."""
    ctx, body = _chain_body()
    src = _oracle_source(_emit(body, ctx))
    mem = kernel_eval.evaluate_wave(src, 32, seed=11, globals_only=True)
    slot = kernel_eval.Slot(11)
    a1 = np.array([slot.read('m1', i) for i in range(32)]).reshape(8, 4)
    b1 = np.array([slot.read('m2', i) for i in range(32)]).reshape(4, 8)
    b2 = np.array([slot.read('m3', i) for i in range(64)]).reshape(8, 8)
    got = np.array([mem[('m0', i)] for i in range(64)]).reshape(8, 8)
    assert np.allclose(got, (a1 @ b1) @ b2, rtol=1e-12, atol=1e-12)


def _cuda_source(lines):
    keep = [line for line in lines if not re.match(r'^double m\d+\[', line)]
    body = ''.join(f'  {line}\n' for line in keep)
    return f'''#include <cstdio>
#include <vector>
#include "tensorforge_device/cuda.h"

__global__ void chain(double *m0, const double *m1, const double *m2,
                      const double *m3) {{
{body}}}

int main(int argc, char **argv) {{
  std::vector<double> h(128), d(64);
  FILE *in = std::fopen(argv[1], "rb");
  if (!in || std::fread(h.data(), 8, 128, in) != 128) return 2;
  std::fclose(in);
  double *m0, *m1, *m2, *m3;
  cudaMalloc(&m0, 64 * 8); cudaMalloc(&m1, 32 * 8);
  cudaMalloc(&m2, 32 * 8); cudaMalloc(&m3, 64 * 8);
  cudaMemcpy(m1, h.data(), 32 * 8, cudaMemcpyHostToDevice);
  cudaMemcpy(m2, h.data() + 32, 32 * 8, cudaMemcpyHostToDevice);
  cudaMemcpy(m3, h.data() + 64, 64 * 8, cudaMemcpyHostToDevice);
  chain<<<1, 32>>>(m0, m1, m2, m3);
  if (cudaDeviceSynchronize() != cudaSuccess) return 4;
  cudaMemcpy(d.data(), m0, 64 * 8, cudaMemcpyDeviceToHost);
  FILE *out = std::fopen(argv[2], "wb");
  if (!out || std::fwrite(d.data(), 8, 64, out) != 64) return 3;
  std::fclose(out);
  return 0;
}}
'''


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


def _build(lines, tmp_path, arch):
    found = toolchain.COMPILERS['cuda'].find()
    if not found or not shutil.which(found):
        pytest.skip('nvcc not found')
    src = tmp_path / 'k.cu'
    src.write_text(_cuda_source(lines))
    nvcc = toolchain.COMPILERS['cuda']
    exe = tmp_path / 'k'
    run = subprocess.run([found, *nvcc.language_flags(),
                          *nvcc.target_flags(arch), '-I', str(INCLUDE),
                          '-o', str(exe), str(src)],
                         capture_output=True, text=True, timeout=600)
    assert run.returncode == 0, run.stderr[-4000:]
    return exe


def test_gpu_multiplies_through_the_relabel(tmp_path):
    """The chain on the local GPU (sm_80 or newer), integer operands, to the
    bit: D1 handed on as A by a renaming of registers alone."""
    arch = _gpu()
    if arch is None or int(arch[3:]) < 80:
        pytest.skip('no GPU of sm_80 or newer')
    ctx, body = _chain_body(arch)
    exe = _build(_emit(body, ctx), tmp_path, arch)
    values = [float((i * 7 + 3) % 11 - 5) for i in range(128)]
    (tmp_path / 'in.bin').write_bytes(struct.pack('<128d', *values))
    run = subprocess.run([str(exe), str(tmp_path / 'in.bin'),
                          str(tmp_path / 'out.bin')], capture_output=True,
                         text=True, timeout=600)
    assert run.returncode == 0, run.stderr[-2000:]
    got = np.array(struct.unpack('<64d', (tmp_path / 'out.bin')
                                 .read_bytes())).reshape(8, 8)
    a1 = np.array(values[:32]).reshape(8, 4)
    b1 = np.array(values[32:64]).reshape(4, 8)
    b2 = np.array(values[64:]).reshape(8, 8)
    assert (got == (a1 @ b1) @ b2).all()


# -- the shuffle -------------------------------------------------------------- #

def _shfl_body(arch='sm_86'):
    """Each lane the value of the lane after it, by a shuffle."""
    ctx = Context(arch=arch, backend='cuda', fp_type=F32)
    b = IRBuilder(fptype=F32, context=ctx)
    src = b.alloc(F32, (32,), MemSpace.GLOBAL, hint='g', extern='m1')
    out = b.alloc(F32, (32,), MemSpace.GLOBAL, hint='o', extern='m0')
    lane = b.thread_id('x')
    x = b.load(src, lane, hint='x')
    nxt = b.op('rem', INDEX, b.op('add', INDEX, lane, 1), 32, hint='n')
    y = b.relayout('shfl', {}, x, nxt, type_=ScalarType(F32), hint='y')
    b.store(out, y, lane)
    return ctx, b.finish()


def test_the_shuffle_is_the_gather_of_cuda_h_on_the_oracle():
    ctx, body = _shfl_body()
    lines = _emit(body, ctx)
    assert any('tensorforge::gather<32>(' in line for line in lines)
    src = ('void k() {\n' + ''.join(
        f'  {line}\n' for line in lines
        if not re.match(r'^float m\d+\[', line)) + '}')
    mem = kernel_eval.evaluate_wave(src, 32, seed=2, globals_only=True)
    slot = kernel_eval.Slot(2)
    assert [mem[('m0', i)] for i in range(32)] == [
        slot.read('m1', (i + 1) % 32) for i in range(32)]


def test_the_shuffle_is_cuda_s():
    ctx, body = _shfl_body()
    s = next(s for s, _ in walk(body) if s.op == Op.RELAYOUT)
    with pytest.raises(IRError, match='no spelling on hip'):
        engines.lower(s, _hip().target, name=str)
    metrics = _Metrics()
    _emit(body, ctx, metrics)
    assert metrics.mix.get('xlane.wide') == 1
