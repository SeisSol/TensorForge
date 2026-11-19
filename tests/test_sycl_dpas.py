# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""DPAS from the SPMD lowering (`Options.sycl_dpas`, `primitives/intel_spmd`).

What a wrong answer would come from, in the order it would come:

* the lane layouts of the three operands, which were measured on the Max
  1550 and are stated once, as bits -- held here against the probe's tables;
* the program the emitter writes around them: the exchange that lays the
  element's operand out, the lanes whose step is not the contraction's, the
  TF32 split and the three products, the operator stored in tiles, the
  columns carried on FMAs, the tiles left out for zeros.  Modeled in NumPy
  from the module's own planning functions and run against `Op @ T`, over
  windows that start inside a slot and contractions that start inside a
  block;
* what the generator does with it: which products take the path, what is
  refused, how the operator is stored, what a kernel needs included.

What it cannot check is the device: the tfemu CPU emulator runs the emitted
kernels (`~/tf-probe/esimd/emu_check.py`), and the Max 1550 times them.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import math
import os
import struct
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

from tensorforge.backend.instructions.compute import bitlayout
from tensorforge.backend.instructions.compute.bitlayout import (Bit,
                                                                BitLayout,
                                                                Place,
                                                                Position)
from tensorforge.backend.instructions.compute.primitives import intel
from tensorforge.backend.instructions.compute.primitives import \
    intel_spmd as ds
from tensorforge.backend.instructions.compute.strategy import (ComputeShape,
                                                               Strategy,
                                                               lead_layout)
from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge import toolchain
from tensorforge.common.exceptions import GenerationError
from tensorforge.common.matrix.prepare import split_tf32

ROOT = Path(__file__).resolve().parent.parent

#: The probe of the `ocloc` on PATH (`intel_spmd.resolved`), before the
#: fixture below stands in for it.
PROBE = toolchain.intel_dpas_tf32


@pytest.fixture(autouse=True)
def _no_ocloc(monkeypatch):
    """Every build here as on a machine with no `ocloc` to ask: what the
    generator writes does not depend on which IGC the test machine has."""
    monkeypatch.setattr(toolchain, 'intel_dpas_tf32', lambda *a, **k: None)


# --------------------------------------------------------------------------
# The layouts, against the probe's tables
# --------------------------------------------------------------------------
#
# SNG2, 11-14 (`~/tf-probe/pvc/dpas-probe/route.cpp`, job 561387): one slot
# set at a time against distinct values in the other operand, then random
# integers compared exactly.  Lane `l`, element `i` of each operand holds:

def _measured_a(lane, element):
    return 2 * element + lane // 8, lane % 8


def _measured_b(lane, element):
    return element, lane


def _measured_d(lane, element):
    return element, lane


def test_the_layouts_are_the_measured_ones():
    for lane in range(ds.SUB_GROUP):
        for element in range(4):
            assert (ds.a_fragment_bits().locate(*_measured_a(lane, element))
                    == Position(lane=lane, element=element))
        for element in range(8):
            assert (ds.b_fragment_bits().locate(*_measured_b(lane, element))
                    == Position(lane=lane, element=element))
            assert (ds.d_fragment_bits().locate(*_measured_d(lane, element))
                    == Position(lane=lane, element=element))


@pytest.mark.parametrize('bits,shape,elements', [
    (ds.a_fragment_bits, (8, 8), 4),
    (ds.b_fragment_bits, (8, 16), 8),
    (ds.d_fragment_bits, (8, 16), 8)], ids=['A', 'B', 'D'])
def test_each_layout_puts_one_element_in_each_place(bits, shape, elements):
    places = [bits().locate(r, c) for r in range(shape[0])
              for c in range(shape[1])]
    assert sorted(places, key=lambda p: (p.lane, p.element)) == [
        Position(lane=lane, element=e) for lane in range(ds.SUB_GROUP)
        for e in range(elements)]


def test_the_accumulator_is_the_register_image():
    """The SPMD image holds row `16 s + r` of a slot in lane `r` and column
    `j0 + c` of a tile in its register `c`; D holds `D[c][r]` in lane `r`,
    element `c`.  The same place, so a product's result is handed on as it
    lies."""
    for row in range(ds.SUB_GROUP):
        for column in range(ds.COLUMNS):
            assert (ds.d_fragment_bits().locate(column, row)
                    == Position(lane=row, element=column))


def test_one_exchange_by_eight_serves_every_element():
    """Every element of the A fragment is the lane's own or its partner's
    eight over, at the slot holding the chunk's steps, in one of the pair's
    two columns -- and a lane sends the same column for either chunk."""
    for chunk in range(4):
        for lane in range(ds.SUB_GROUP):
            for element in range(4):
                slot, source, column = ds.exchange_source(chunk, lane, element)
                assert slot == chunk // 2
                assert source in (lane, lane ^ 8)
                assert column in (2 * element, 2 * element + 1)
    take, send = ds.exchange_plan()
    assert sorted(take) == [(0, False), (0, True), (1, False), (1, True)]
    assert sorted(send) == [False, True]
    # The upper half of a chunk of even parity, the lower of an odd one, is
    # what comes from the partner: the rows of the other half of the slot.
    assert take[0, True][0] and take[1, False][0]
    assert not take[0, False][0] and not take[1, True][0]


def test_a_layout_one_exchange_cannot_serve_is_refused(monkeypatch):
    """The plan is derived, so a layout with the lanes four apart is not
    silently exchanged by eight."""
    wrong = BitLayout(((Bit(Place.LANE, 4), Bit(Place.VECTOR, 1),
                        Bit(Place.VECTOR, 2)),
                       (Bit(Place.LANE, 1), Bit(Place.LANE, 2),
                        Bit(Place.LANE, 8))))
    monkeypatch.setattr(ds, 'a_fragment_bits', lambda: wrong)
    with pytest.raises(GenerationError):
        ds.exchange_plan()


# --------------------------------------------------------------------------
# The program the emitter writes, in NumPy
# --------------------------------------------------------------------------

def _round(x):
    """`tensorforge::roundTF32`."""
    u = np.asarray(x, np.float32).view(np.uint32).astype(np.uint64)
    return ((u + 0x1000) & 0xFFFFE000).astype(np.uint32).view(np.float32)


def _truncate(x):
    """What DPAS does to its inputs."""
    u = np.asarray(x, np.float32).view(np.uint32)
    return (u & np.uint32(0xFFFFE000)).view(np.float32)


def _gather(bits, shares, shape):
    out = np.zeros(shape)
    for r in range(shape[0]):
        for c in range(shape[1]):
            at = bits.locate(r, c)
            out[r, c] = shares[at.lane][at.element]
    return out


def _scatter(bits, matrix, elements):
    shares = np.zeros((ds.SUB_GROUP, elements))
    for r in range(matrix.shape[0]):
        for c in range(matrix.shape[1]):
            at = bits.locate(r, c)
            shares[at.lane][at.element] = matrix[r, c]
    return shares


def _dpas(a, b, acc):
    """The instruction on the work-items' shares, from the three layouts:
    both factors truncated to TF32, the sum in double."""
    am = _gather(ds.a_fragment_bits(), a, (8, 8))
    bm = _gather(ds.b_fragment_bits(), b, (8, 16))
    cm = _gather(ds.d_fragment_bits(), acc, (8, 16))
    dm = cm + (_truncate(am).astype(np.float64)
               @ _truncate(bm).astype(np.float64))
    return _scatter(ds.d_fragment_bits(), dm, 8)


def test_the_layouts_make_a_matrix_product():
    rng = np.random.default_rng(3)
    a = rng.integers(-8, 8, (8, 8)).astype(np.float64)
    b = rng.integers(-8, 8, (8, 16)).astype(np.float64)
    c = rng.integers(-8, 8, (8, 16)).astype(np.float64)
    d = _dpas(_scatter(ds.a_fragment_bits(), a, 4),
              _scatter(ds.b_fragment_bits(), b, 8),
              _scatter(ds.d_fragment_bits(), c, 8))
    assert np.array_equal(_gather(ds.d_fragment_bits(), d, (8, 16)),
                          c + a @ b)


def _program(op, t, rows, c0, depth, kx, n, skip=True):
    """`Op[rows, c0:c0+depth] @ T` as `_Product` computes it: `(out,
    instructions)`, `out` over the rows of the slots from row 0.

    `t` is the element's operand by position, `kx` positions before the
    contraction's first step: the register image the emitter reads with
    `B(j, slot)`, lane `p % 16` of slot `p / 16`.  The positions before
    `kx` and past the contraction hold NaN, which the masks have to keep out.
    The operator is stored as `prepared_order` stores it and split as the
    host splits it."""
    lo_row, hi_row = rows
    nslots = -(-hi_row // ds.SUB_GROUP)
    nchunks = ds.chunks(depth, kx)
    order = ds.fragment_order(op.shape, ((0, ds.SUB_GROUP * nslots),
                                         (c0, depth)), kx)
    flat = op.flatten(order='F')
    values = np.array([flat[c] if c >= 0 else 0.0 for c in order],
                      np.float32)
    stored = split_tf32(values, planar=len(values))
    planes = (stored[:len(values)], stored[len(values):])
    positions = ds.SUB_GROUP * -(-(depth + kx) // ds.SUB_GROUP)
    image = np.full((positions, n), np.nan, np.float32)
    image[kx:kx + depth] = t
    lanes = np.arange(ds.SUB_GROUP)
    low, step = lanes < 8, lanes % 8
    take, send = ds.exchange_plan()
    ntail = n % ds.COLUMNS
    tailed = 0 < ntail <= ds.TAIL_MAX and n > ds.COLUMNS
    dense = n - ntail if tailed else n
    window = np.zeros((ds.SUB_GROUP * nslots, op.shape[1]))
    window[lo_row:hi_row] = op[lo_row:hi_row]

    def zero(s, k_lo, k_hi):
        rows_ = window[ds.SUB_GROUP * s:ds.SUB_GROUP * (s + 1)]
        return skip and not np.any(rows_[:, c0 + k_lo:c0 + k_hi])

    out = np.zeros((ds.SUB_GROUP * nslots, n))
    count = 0
    tiles = list(range(0, dense, ds.COLUMNS))
    for j0 in tiles:
        acc = np.zeros((nslots, ds.SUB_GROUP, 8))
        carried = list(range(dense, n)) if j0 == tiles[-1] and tailed else []
        for c in range(nchunks):
            k_lo, k_hi = max(0, 8 * c - kx), min(depth, 8 * c + 8 - kx)
            live = [s for s in range(nslots)
                    if k_lo < k_hi and not zero(s, k_lo, k_hi)]
            if not live:
                continue
            slot, odd = divmod(c, 2)
            end = min(j0 + ds.COLUMNS, dense)
            inside = (step >= kx - 8 * c) & (step < depth + kx - 8 * c)
            ahi, alo = np.zeros((16, 4), np.float32), np.zeros((16, 4),
                                                                np.float32)
            for pair in range(4):
                held = [image[16 * slot:16 * slot + 16, col] if col < end
                        else np.zeros(16, np.float32)
                        for col in (j0 + 2 * pair, j0 + 2 * pair + 1)]
                sent = np.where(low, held[send[False]], held[send[True]])
                swapped = sent[lanes ^ 8]

                def taken(high):
                    partner, offset = take[odd, high]
                    return swapped if partner else held[offset]
                v = np.where(low, taken(False), taken(True))
                v = np.where(inside, v, np.float32(0)).astype(np.float32)
                ahi[:, pair] = _round(v)
                alo[:, pair] = v - ahi[:, pair]
            for s in live:
                base = ds.tile_slot(s, c, nchunks)
                b = [plane[base:base + ds.TILE].reshape(16, 8)
                     for plane in planes]
                for p, q in ds.PRODUCTS:
                    acc[s] = _dpas((ahi, alo)[q], b[p], acc[s])
                    count += 1
                for k in range(8):
                    position = 8 * c + k
                    if not k_lo <= position - kx < k_hi or zero(
                            s, position - kx, position - kx + 1):
                        continue
                    for col in carried:
                        shared = np.float64(image[position, col])
                        out[16 * s:16 * s + 16, col] += (
                            b[1][:, k].astype(np.float64) * shared
                            + b[0][:, k].astype(np.float64) * shared)
        for s in range(nslots):
            out[16 * s:16 * s + 16, j0:min(j0 + 8, dense)] = \
                acc[s][:, :min(8, dense - j0)]
    return out, count


#: `(rows, c0, depth, kx, n)`: SeisSol's shapes and the edges between them.
SHAPES = {
    'flux_operator': ((0, 21), 0, 56, 0, 9),
    'flux_back': ((0, 56), 0, 21, 0, 9),
    'volume_row_one': ((1, 54), 0, 35, 0, 9),
    'derivative_column_one': ((0, 35), 1, 53, 1, 9),
    'padded_tile': ((0, 40), 2, 20, 5, 6),
    'two_tiles': ((0, 32), 0, 16, 0, 16),
    'across_blocks': ((0, 16), 3, 17, 15, 10),
}


@pytest.mark.parametrize('name', sorted(SHAPES))
def test_the_program_computes_the_product(name):
    """Normwise against `Op @ T` in double, where 3xTF32 keeps about 22
    bits; and the same numbers with the zero tiles left out as with every
    tile, which is what makes leaving them out a matter of time alone."""
    rows, c0, depth, kx, n = SHAPES[name]
    rng = np.random.default_rng(len(name))
    op = rng.uniform(-1, 1, (64, c0 + depth + 3)).astype(np.float32)
    # A block of zeros a tile wide, as SeisSol's operators have them.
    op[16:32, c0 + 8:c0 + 16] = 0
    t = rng.uniform(-1, 1, (depth, n)).astype(np.float32)
    want = (op[rows[0]:rows[1], c0:c0 + depth].astype(np.float64)
            @ t.astype(np.float64))
    got, count = _program(op, t, rows, c0, depth, kx, n)
    got = got[rows[0]:rows[1]]
    assert np.all(np.isfinite(got))
    assert np.max(np.abs(got - want)) / np.max(np.abs(want)) < 2e-6
    dense, every = _program(op, t, rows, c0, depth, kx, n, skip=False)
    assert np.allclose(dense[rows[0]:rows[1]], got, rtol=0, atol=1e-6)
    assert count <= every


# --------------------------------------------------------------------------
# The generator
# --------------------------------------------------------------------------

def _ctx(backend='oneapi', arch='pvc', dtype=Datatype.F32, **options):
    from tensorforge.common.context import Context, Options
    return Context(arch=arch, backend=backend, fp_type=dtype,
                   options=Options(**options))


def _generate(descrs, ctx):
    from tensorforge.generators.generator import Generator
    gen = Generator(descrs, ctx)
    with contextlib.redirect_stdout(io.StringIO()), \
            warnings.catch_warnings():
        warnings.simplefilter('ignore')
        gen.generate()
    return gen


def _case(name):
    path = ROOT / 'tests' / 'cases' / f'{name}.py'
    spec = importlib.util.spec_from_file_location(f'dpas_{name}', path)
    mod = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(mod)
    return mod.descr_list()


def _seissol(kernel, order=6, precision='s'):
    import seissol_suite as fx
    spec = importlib.util.spec_from_file_location(
        'dpas_corpus_check', ROOT / 'tools' / 'host' / 'corpus_check.py')
    host = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(host)
    system = 'elastic-linearck'
    description = fx.description(system, f'{system}-o{order}-{precision}',
                                 kernel)
    dtype = Datatype.F64 if precision == 'd' else Datatype.F32
    return host.Kernel(description, dtype, 'pvc').descrs()


def _constants(gen):
    return {s.obj.alias: s.obj
            for s in gen._scopes.get_global_scope().values()
            if getattr(s.obj, 'addressing', None) is Addressing.NONE}


def _calls(src):
    return src.count(f'{ds.DPAS}(')


def test_the_default_build_takes_no_dpas(monkeypatch):
    monkeypatch.delenv('TF_SYCL_DPAS', raising=False)
    ctx = _ctx(lanes_per_mult=16)
    assert ctx.get_user_options().sycl_dpas is False
    src = _generate(_case('local_flux'), ctx).get_kernel()
    assert _calls(src) == 0 and ds.ROUND not in src


@pytest.mark.parametrize('value,taken', [(True, True), (1, True),
                                         ('auto', False), (0, False)])
def test_what_turns_the_path_on(value, taken):
    """A caller may hand in `1` (the bench harness reads `sycl_dpas=1` as an
    integer); `auto` is the tuner's to answer and builds the FMA path."""
    assert ds.enabled(_ctx(sycl_dpas=value)) is taken
    assert ds.usable(_ctx(sycl_dpas=value)) is taken


def test_the_option_reaches_the_emitter():
    """`local_flux` at sixteen lanes: per face `X = A_f @ B`, four slots of
    56 rows by seven chunks of 56 steps, three products each, its ninth
    column carried; `R += X @ C_f` contracts nine steps and stays on the
    nest.  The operators carry no numbers, so no tile is left out."""
    gen = _generate(_case('local_flux'),
                    _ctx(lanes_per_mult=16, sycl_dpas=True))
    src = gen.get_kernel()
    faces, slots, chunks = 4, 4, 7
    assert _calls(src) == faces * slots * chunks * len(ds.PRODUCTS)
    assert 'permute_group_by_xor' in src


def test_the_operator_is_stored_split_in_tiles():
    """Each `A_f` in `fragment_order` over its window, its two TF32 halves
    planar; what the host writes is `split_tf32` of the order's cells."""
    gen = _generate(_case('local_flux'),
                    _ctx(lanes_per_mult=16, sycl_dpas=True))
    operators = [t for alias, t in _constants(gen).items()
                 if alias.startswith('A')]
    assert len(operators) == 4
    for t in operators:
        assert t.storage_parts == ds.SPLIT_TERMS and t.storage_planar
        assert tuple(t.storage_order) == ds.fragment_order(
            (56, 56), ((0, 64), (0, 56)), 0)


def test_zero_tiles_are_left_out():
    """SeisSol's order-6 volume, single precision: `kDivM(d)` over rows 1 to
    53, 54 and 55 and 35 steps -- four slots by five chunks, 180
    instructions with every tile.  The operators' numbers leave a quarter of
    the tiles empty, and those are neither read nor multiplied."""
    descrs = _seissol('gpu_volume')
    every = _calls(_generate(descrs, _ctx(sycl_dpas=True,
                                          skip_known_zeros=False))
                   .get_kernel())
    gen = _generate(_seissol('gpu_volume'), _ctx(sycl_dpas=True))
    assert every == 3 * 4 * 5 * len(ds.PRODUCTS)
    live = 0
    for alias, t in _constants(gen).items():
        if not alias.startswith('kDivM'):
            continue
        # A copy of the operator for its reader (`mma_operator_copies`, the
        # default under the path) is named after it, `@` its number.
        last = {'kDivM(0)': 54, 'kDivM(1)': 55,
                'kDivM(2)': 56}[alias.split('@')[0]]
        data = np.asarray(t.data)
        for s in range(4):
            for c in range(5):
                block = data[max(16 * s, 1):min(16 * s + 16, last),
                             8 * c:min(8 * c + 8, 35)]
                live += bool(np.any(block))
    assert _calls(gen.get_kernel()) == live * len(ds.PRODUCTS) < every


def test_an_operator_with_several_readers_is_read_by_coordinate():
    """The order-6 derivative reads each `kDivMT` five times, over windows
    from column 1: without copies (`mma_operator_copies` off) no order is
    one reader's, so the operator stays as it is passed and the path splits
    each tile it reads.  The two shortest contractions, 7 and 1 steps, stay
    on the nest."""
    gen = _generate(_seissol('gpu_derivative'),
                    _ctx(sycl_dpas=True, mma_operator_copies=False))
    src = gen.get_kernel()
    operators = [t for alias, t in _constants(gen).items()
                 if alias.startswith('kDivMT')]
    assert operators and all(t.storage_order is None
                             and t.storage_parts == 1 for t in operators)
    assert _calls(src) > 0
    # Every operator value a tile holds is split in the kernel: more
    # roundings than the element's operand alone would take.
    assert src.count(f'{ds.ROUND}(') > _calls(src) // len(ds.PRODUCTS) * 4


def _dense(**fields):
    base = dict(threads=16, accumulator=Datatype.F32, sparse=False,
                explicit_simd=False, lead=56, depth=56, a_uniform=True)
    base.update(fields)
    return ComputeShape(**base)


def test_the_shape_gate():
    ctx = _ctx(sycl_dpas=True)
    assert ds.offered(_dense(), ctx)
    assert intel.strategies(_dense(), ctx) == {Strategy.MATRIX}
    refused = {
        'eight lanes': _dense(threads=8),
        'thirty-two lanes': _dense(threads=32),
        'double': _dense(accumulator=Datatype.F64),
        'a sparse second operand': _dense(sparse=True),
        'a per-element operator': _dense(a_uniform=False),
        'two rows a lane': _dense(lead_layout=lead_layout(16, 2)),
        'a short contraction': _dense(depth=9),
        'the explicit vector': _dense(explicit_simd=True),
    }
    for why, shape in refused.items():
        assert not ds.offered(shape, ctx), why
        assert Strategy.MATRIX not in intel.strategies(shape, ctx) or (
            shape.explicit_simd), why
    assert ds.offered(_dense(depth=9), _ctx(sycl_dpas=True,
                                            sycl_dpas_min_depth=8))


@pytest.mark.parametrize('backend,arch,options', [
    ('oneapi', 'pvc', {}),
    ('esimd', 'pvc', {'sycl_dpas': True}),
    ('acpp', 'pvc', {'sycl_dpas': True}),
    ('oneapi', 'dg1', {'sycl_dpas': True}),
    ('oneapi', 'pvc', {'sycl_dpas': 'auto'}),
], ids=['not asked', 'esimd', 'acpp', 'dg1', 'auto, untuned'])
def test_only_dpcpp_spmd_on_pvc(backend, arch, options):
    """The builtin is IGC's under DPC++, and sixteen wide on Xe-HPC; the
    explicit vector has DPAS of its own (`Options.tensor_cores`)."""
    assert not ds.usable(_ctx(backend, arch, **options))


def test_too_few_columns_go_to_the_nest():
    shape = _dense()
    ctx = _ctx(sycl_dpas=True)
    assert intel.plan(Strategy.MATRIX, shape, 2, ctx)[0].strategy \
        is Strategy.GENERIC
    assert intel.plan(Strategy.MATRIX, shape, 9, ctx)[0].strategy \
        is Strategy.MATRIX
    assert intel.prepared_order((56, 56), Datatype.F32, ctx, columns=2,
                                lead=56, depth=56, threads=16) is None


@pytest.mark.parametrize('lanes', [8, 32])
def test_other_widths_take_no_dpas(lanes):
    src = _generate(_case('local_flux'),
                    _ctx(lanes_per_mult=lanes, sycl_dpas=True)).get_kernel()
    assert _calls(src) == 0


def test_the_header_is_asked_for():
    """`dpasTF32` is declared where the split broadcast is, and a kernel
    that takes the path needs it whether or not the split is on.  The
    sub-group runs, on by default under `oneapi`, ask for it as well, so
    they are off where nothing is to."""
    header = 'tensorforge_device/sycl_spmd.h'
    assert header in _ctx(sycl_dpas=True, sycl_split_broadcast=0,
                          spmd_block_runs=False).target.headers()
    assert header not in _ctx(sycl_split_broadcast=0,
                              spmd_block_runs=False).target.headers()


def test_a_dpas_kernel_parses():
    from harness import syntax
    if syntax.compiler() is None:
        pytest.skip('no host C++ compiler')
    src = _generate(_case('local_flux'),
                    _ctx(lanes_per_mult=16, sycl_dpas=True)).get_kernel()
    result = syntax.check_source(src, shim=syntax.SYCL_SHIM)
    assert result.ok, result.errors()


def test_declining_an_operator_stored_split_is_an_error():
    """Only the path reads the halves; handed back to the nest, the upper
    one would be read for the value."""
    from tensorforge.backend.instructions.compute.matmul import \
        MatmulOperands
    from tensorforge.backend.instructions.compute.strategy import Span
    from tensorforge.common.exceptions import InternalError
    ops = MatmulOperands(A=None, B=None, C=None, sparse=None, lead_slots=1,
                         lead_elements=16, n=9, k=16, kx=0, threads=16,
                         a=Datatype.F64, b=Datatype.F64,
                         accumulator=Datatype.F32, a_parts=2)
    with pytest.raises(InternalError):
        ds.matmul(None, ops, _ctx(sycl_dpas=True),
                  Span(Strategy.MATRIX, 0, 9))


def test_the_fence_reads_one_tile_ahead():
    """`Options.sycl_dpas_fence`: a fence of the sub-group in front of every
    tile's products, the next tile read behind it, and one at either end of
    the product -- in `local_flux`, one a tile and two a product, and
    nothing else fenced.  Off by default."""
    fenced = _generate(_case('local_flux'),
                       _ctx(lanes_per_mult=16, sycl_dpas=True,
                            sycl_dpas_fence=True)).get_kernel()
    plain = _generate(_case('local_flux'),
                      _ctx(lanes_per_mult=16, sycl_dpas=True)).get_kernel()
    fence = 'sycl::group_barrier(item.get_sub_group());'
    tiles = _calls(fenced) // len(ds.PRODUCTS)
    products = 4
    assert fenced.count(fence) - plain.count(fence) == tiles + 2 * products
    assert _calls(fenced) == _calls(plain)


def test_each_reader_of_a_shared_operator_gets_its_tiles():
    """Under `sycl_dpas` operator copies are the default
    (`mma_operator_copies` unset): the derivative's DPAS readers of `kDivMT`
    each read a copy of their own, defined in the file, aligned for the
    vector reads, in their tiles and split -- only the element's operand is
    split in the kernel, four values a chunk."""
    gen = _generate(_seissol('gpu_derivative'), _ctx(sycl_dpas=True))
    src = gen.get_kernel()
    assert 'alignas(256) constexpr float' in src
    copies = [t for t in gen._module_operands
              if gen._scopes.get_symbol(t) is not None]
    assert copies and all(t.storage_parts == ds.SPLIT_TERMS
                          and t.storage_order is not None for t in copies)
    unordered = _generate(_seissol('gpu_derivative'),
                          _ctx(sycl_dpas=True,
                               mma_operator_copies=False)).get_kernel()
    assert _calls(src) == _calls(unordered)
    assert src.count(f'{ds.ROUND}(') < unordered.count(f'{ds.ROUND}(')
    assert src.count(f'{ds.ROUND}(') % 4 == 0


# --------------------------------------------------------------------------
# IGC 1.x, named at generation
# --------------------------------------------------------------------------

def _elf(symbols):
    """A 64-bit ELF file holding a symbol table of `(name, defined)`."""
    strings = b'\0'
    table = [struct.pack('<IBBHQQ', 0, 0, 0, 0, 0, 0)]
    for name, defined in symbols:
        table.append(struct.pack('<IBBHQQ', len(strings), 0x10, 0,
                                 1 if defined else 0, 0, 0))
        strings += name.encode() + b'\0'
    symtab = b''.join(table)
    names = b'\0.symtab\0.strtab\0.shstrtab\0'
    at_symtab = 64
    at_strings = at_symtab + len(symtab)
    at_names = at_strings + len(strings)
    pad = (-(at_names + len(names))) % 8
    shoff = at_names + len(names) + pad

    def section(name, kind, offset, size, link=0, entsize=0):
        return struct.pack('<IIQQQQIIQQ', name, kind, 0, 0, offset, size,
                           link, 0, 1, entsize)

    sections = (section(0, 0, 0, 0)
                + section(1, 2, at_symtab, len(symtab), link=2, entsize=24)
                + section(9, 3, at_strings, len(strings))
                + section(17, 3, at_names, len(names)))
    header = (b'\x7fELF' + bytes([2, 1, 1]) + bytes(9)
              + struct.pack('<HHIQQQIHHHHHH', 1, 0, 1, 0, 0, shoff, 0, 64, 0,
                            0, 64, 4, 3))
    return header + symtab + strings + names + bytes(pad) + sections


#: How IGC 1.0.15468 leaves the call in the probe's binary (SNG2, ocloc of
#: compute-runtime 23.43): an undefined symbol, beside the patch token every
#: binary imports.
UNRESOLVED = '_Z39intel_sub_group_tf32_tf32_matrix_mad_k8Dv4_fDv8_fS0_'


def test_the_undefined_symbols_of_a_binary(tmp_path):
    path = tmp_path / 'probe'
    path.write_bytes(_elf([('probe', True), (UNRESOLVED, False),
                           ('__INTEL_PATCH_CROSS_THREAD_OFFSET_OFF_R0',
                            False)]))
    assert toolchain.elf_undefined_symbols(str(path)) == [
        UNRESOLVED, '__INTEL_PATCH_CROSS_THREAD_OFFSET_OFF_R0']
    (tmp_path / 'text').write_text('Build succeeded.')
    assert toolchain.elf_undefined_symbols(str(tmp_path / 'text')) is None


def _ocloc(tmp_path, symbols, status=0):
    """A stand-in `ocloc` that "builds" the probe into a binary of
    `symbols`, as both IGCs say: successfully."""
    image = tmp_path / 'image'
    image.write_bytes(_elf(symbols))
    script = tmp_path / 'ocloc'
    script.write_text(
        f'#!{sys.executable}\n'
        'import os, shutil, sys\n'
        'args = sys.argv[1:]\n'
        "out = args[args.index('-out_dir') + 1]\n"
        'os.makedirs(out, exist_ok=True)\n'
        f"shutil.copy({str(image)!r}, os.path.join(out, 'probe'))\n"
        "print('Build succeeded.')\n"
        f'sys.exit({status})\n')
    script.chmod(0o755)
    return str(script)


def test_the_probe_reads_the_binary_not_the_word(tmp_path):
    """IGC 1.x says "Build succeeded" and leaves the call undefined: a no.
    IGC 2.x resolves it: a yes.  A failed build is a no, and no `ocloc` at
    all is no answer."""
    old = tmp_path / 'old'
    new = tmp_path / 'new'
    failed = tmp_path / 'failed'
    for d in (old, new, failed):
        d.mkdir()
    patch = ('__INTEL_PATCH_CROSS_THREAD_OFFSET_OFF_R0', False)
    assert PROBE(ocloc=_ocloc(old, [('probe', True), (UNRESOLVED, False),
                                    patch])) is False
    assert PROBE(ocloc=_ocloc(new, [('probe', True), patch])) is True
    assert PROBE(ocloc=_ocloc(failed, [('probe', True), patch],
                              status=1)) is False
    assert PROBE(ocloc=str(tmp_path / 'nowhere')) is None


def test_an_igc_without_the_builtin_is_named_at_generation(monkeypatch):
    monkeypatch.setattr(toolchain, 'intel_dpas_tf32', lambda *a, **k: False)
    with pytest.raises(GenerationError, match='IGC 2.x'):
        _generate(_case('local_flux'), _ctx(lanes_per_mult=16,
                                            sycl_dpas=True))

    def asked(*args, **kwargs):
        raise AssertionError('the default build asked for the IGC')

    # Nothing to ask where the path is off: the default build never runs it.
    monkeypatch.setattr(toolchain, 'intel_dpas_tf32', asked)
    _generate(_case('local_flux'), _ctx(lanes_per_mult=16))


# --------------------------------------------------------------------------
# Per kernel: the twins of `autotune=lanes`
# --------------------------------------------------------------------------

@pytest.fixture
def _forget_picks():
    from tensorforge.generators import tuning
    tuning._PICKS.clear()
    tuning._LOADED.clear()
    yield
    tuning._PICKS.clear()
    tuning._LOADED.clear()


def _tuned(dtype=Datatype.F32, backend='oneapi', arch='pvc', **options):
    """`local_flux` at sixteen lanes as the lanes rule starts it: the
    factory, the context, the origin and its build."""
    from tensorforge.generators import tuning
    from tensorforge.generators.generator import Generator
    from test_lanes_rule import _local_flux
    factory = _local_flux(m=56, n=9, datatype=dtype)
    context = _ctx(backend, arch, dtype, **options)
    origin = tuning.start(factory(), context)
    first = tuning.build(factory, context, origin, Generator.trial)
    assert first.ok
    return factory, context, origin, first


def _twins(*args, **options):
    from tensorforge.generators import tuning
    factory, context, origin, first = _tuned(*args, **options)
    return tuning._dpas_candidates(factory(), first, context, origin, None)


def test_auto_offers_the_path_with_and_without_the_fence():
    """Not the D chains: their count is lower, and they run slower."""
    twins = _twins(sycl_dpas='auto')
    assert [t.get('sycl_dpas_fence') for t in twins] == [None, True]
    assert all(t.get('sycl_dpas') is True and t.get('mma_d_layout') is None
               for t in twins)
    # A fence the caller states stays as stated.
    stated = _twins(sycl_dpas='auto', sycl_dpas_fence=False)
    assert len(stated) == 1 and stated[0].get('sycl_dpas_fence') is None


@pytest.mark.parametrize('dtype,backend,options', [
    (Datatype.F32, 'oneapi', {}),
    (Datatype.F32, 'oneapi', {'sycl_dpas': True}),
    (Datatype.F32, 'oneapi', {'sycl_dpas': False}),
    (Datatype.F64, 'oneapi', {'sycl_dpas': 'auto'}),
    (Datatype.F32, 'esimd', {'sycl_dpas': 'auto'}),
], ids=['not asked', 'asked on', 'asked off', 'double', 'esimd'])
def test_auto_alone_offers_twins(dtype, backend, options):
    assert _twins(dtype, backend, **options) == []


def test_no_twins_where_the_igc_lacks_the_builtin(monkeypatch):
    monkeypatch.setattr(toolchain, 'intel_dpas_tf32', lambda *a, **k: False)
    assert _twins(sycl_dpas='auto') == []


class _Compiler:
    """A compiler that reports `default` bytes spilled for every build but
    the DPAS twins', `dpas` for the twin and `fence` for the fenced one."""
    def __init__(self, default=0, dpas=0, fence=None):
        self.default, self.dpas = default, dpas
        self.fence = dpas if fence is None else fence

    def resources(self, result):
        c = result.candidate
        spilled = (self.default if c.get('sycl_dpas') is not True
                   else self.fence if c.get('sycl_dpas_fence') is True
                   else self.dpas)
        return toolchain.Resources(registers=128, spill_bytes=spilled)


def _rule(monkeypatch, dpas_count, compiler):
    from tensorforge.generators import tuning
    from tensorforge.generators.generator import Generator
    monkeypatch.setattr(tuning, 'executed_instructions', lambda result: (
        dpas_count if result.candidate.get('sycl_dpas') is True else 2000.0))
    factory, context, origin, first = _tuned(sycl_dpas='auto')
    return tuning.lanes_rule(factory, context, compiler, origin, first,
                             generate=Generator.trial)


@pytest.mark.parametrize('dpas_count,taken', [(1000.0, True),
                                              (1950.0, False),
                                              (3000.0, False)])
def test_the_rule_takes_the_path_where_the_count_says(
        monkeypatch, _forget_picks, dpas_count, taken):
    """By the margin every pick has to win by (`DEVIATION_MARGIN`); the
    fence, which spills the same here, is left."""
    pick = _rule(monkeypatch, dpas_count, _Compiler())
    assert (pick.get('sycl_dpas') is True) is taken
    assert pick.get('sycl_dpas_fence') is None


@pytest.mark.parametrize('default,dpas,taken', [
    (0, 17280, True),          # spills where the default does not
    (11264, 3328, True),       # less than the default
    (5056, 9024, True),        # less than twice the default
    (29888, 66816, False),     # more than twice the default
    (0, 19 * 1024, False),     # past the floor
])
def test_the_rule_leaves_the_path_where_igc_spills_it(
        monkeypatch, _forget_picks, default, dpas, taken):
    """`DPAS_SPILL_GROWTH` over the default's spilling, at least
    `DPAS_SPILL_FLOOR`: the walk's price of a spilled byte would leave the
    twin wherever it spills at all."""
    from tensorforge.generators import tuning
    assert (tuning.DPAS_SPILL_GROWTH, tuning.DPAS_SPILL_FLOOR) == (2.0,
                                                                  18 * 1024)
    pick = _rule(monkeypatch, 1000.0, _Compiler(default, dpas))
    assert (pick.get('sycl_dpas') is True) is taken


@pytest.mark.parametrize('dpas,fence,fenced', [(3648, 7360, False),
                                               (17280, 9408, True),
                                               (0, 20000, False)])
def test_the_fence_where_igc_spills_it_less(monkeypatch, _forget_picks,
                                            dpas, fence, fenced):
    pick = _rule(monkeypatch, 1000.0, _Compiler(0, dpas, fence))
    assert pick.get('sycl_dpas') is True
    assert (pick.get('sycl_dpas_fence') is True) is fenced


# --------------------------------------------------------------------------
# D chains: the accumulators start at the output's old value
# --------------------------------------------------------------------------

def test_chains_where_the_spmd_path_is():
    """`mma_d_layout` reaches the SPMD path's accumulators, which are the
    register image; the explicit vector's are not, and keep no chains."""
    assert intel.d_chains(_ctx(sycl_dpas=True))
    assert not intel.d_chains(_ctx())
    assert not intel.d_chains(_ctx('esimd', 'pvc', sycl_dpas=True))
    atom = intel.d_chain_atom(_ctx(sycl_dpas=True), Datatype.F32, columns=9,
                              lead=56, depth=56)
    assert (atom.m, atom.n, atom.mode) == (ds.SUB_GROUP, ds.COLUMNS,
                                           ds.CHAIN_MODE)
    assert atom.mode in intel.EMITTED_MODES
    assert intel.d_chain_atom(_ctx(sycl_dpas=True), Datatype.F32,
                              columns=2, lead=56, depth=56) is None
    assert intel.d_chain_atom(_ctx(sycl_dpas=True), Datatype.F64,
                              columns=9, lead=56, depth=56) is None


def _chains(monkeypatch):
    """The sizes of the chains `dchain.link` forms, as they are formed."""
    from tensorforge.backend.instructions.compute import dchain
    seen = []
    real = dchain.link

    def link(context, instrs):
        out = real(context, instrs)
        seen.extend(len(x._dchain.members) for x in dchain._flat(instrs)
                    if getattr(x, '_dchain', None) is not None
                    and x._dchain.members[0] is x)
        return out
    monkeypatch.setattr(dchain, 'link', link)
    return seen


def test_the_volume_sums_its_three_products_in_the_accumulators(monkeypatch):
    """SeisSol's order-6 volume adds three products into `Q`: one chain,
    the old value read into the accumulators before the first and the sum
    written after the last -- the same instructions in between."""
    plain = _generate(_seissol('gpu_volume'),
                      _ctx(sycl_dpas=True)).get_kernel()
    seen = _chains(monkeypatch)
    src = _generate(_seissol('gpu_volume'),
                    _ctx(sycl_dpas=True, mma_d_layout=True)).get_kernel()
    assert seen and set(seen) == {3}
    assert _calls(src) == _calls(plain)


def test_no_chain_ends_in_an_output_read_again(monkeypatch):
    """The time derivative reads each order's sum for the next order: a chain
    would leave it in memory, where the product's register image holds it
    without one (`intel.D_CHAIN_READ_AFTER`).  So the derivative chains
    nothing, and its kernel is the one without chains."""
    plain = _generate(_seissol('gpu_derivative'),
                      _ctx(sycl_dpas=True)).get_kernel()
    seen = _chains(monkeypatch)
    src = _generate(_seissol('gpu_derivative'),
                    _ctx(sycl_dpas=True, mma_d_layout=True)).get_kernel()
    assert seen == []

    def body(text):
        # The kernel's name and its options comment carry the options.
        return [line for line in text.splitlines()
                if 'kernel_kernel_' not in line and '// options:' not in line]
    assert body(src) == body(plain)
