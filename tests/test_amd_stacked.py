# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""The multiplications of a wave stacked along a 16x16x4 instruction's M.

`primitives.amd.stacked` claims that the instruction's D lands in the register
image the nest already has, so that nothing moves after the issue, and builds
the two operand fragments to make that true -- for `mfma_f64_16x16x4f64`,
which stacks the elements minor in M, and `mfma_f32_16x16x4f32`, which stacks
them major.  Three kinds of check:

* the claim against the fragment tables (`layouts`), which come from AMD's
  calculator rather than from the scheme;
* the emitter run in a wave simulator: every call it makes evaluated over the
  64 lanes of a wave, the instruction from those tables, the per-element
  operand read where the nest holds it -- and a NaN wherever it holds no
  position of the contraction, so that a fragment reaching past the edge
  cannot pass;
* what the generator makes of it: how many issues a kernel has, and where the
  scheme declines.
"""

from __future__ import annotations

import contextlib
import dataclasses
import importlib.util
import io
import re
import warnings
from pathlib import Path

import numpy as np
import pytest

from tensorforge.backend.instructions.compute.matmul import MatmulOperands
from tensorforge.backend.instructions.compute.primitives import amd
from tensorforge.backend.instructions.compute.primitives.amd import (
    catalog, layouts, stacked)
from tensorforge.backend.instructions.compute.strategy import (ComputeShape,
                                                               Span, Strategy,
                                                               lead_layout)
from tensorforge.backend.pir.core import Uniformity
from tensorforge.common.basic_types import Datatype
from tensorforge.common.context import Context, Options
from tensorforge.generators.generator import Generator
from tensorforge.generators.lanes import LaneConfig

OP = next(o for o in catalog.MATRIX_OPS if o.builtin == "mfma_f64_16x16x4f64")
F32_OP = next(o for o in catalog.MATRIX_OPS
              if o.builtin == "mfma_f32_16x16x4f32")
WAVE, THREADS, MULTS, COLUMNS = 64, 16, 4, 4

#: The instruction by the type it accumulates in, and its M order.
OPS = {Datatype.F64: OP, Datatype.F32: F32_OP}
ORDERS = {Datatype.F64: stacked.ELEMENT_MINOR,
          Datatype.F32: stacked.ELEMENT_MAJOR}
TYPES = [Datatype.F64, Datatype.F32]

CASES = Path(__file__).parent / "cases"
MFMA = re.compile(r"__builtin_amdgcn_mfma_f64_16x16x4f64\(")
F32_MFMA = re.compile(r"__builtin_amdgcn_mfma_f32_16x16x4f32\(")


def _ctx(arch="gfx942", dtype=Datatype.F64, **options):
    options.setdefault("amd_matrix_k4", "transpose")
    return Context(arch=arch, backend="hip", fp_type=dtype,
                   options=Options(**options))


def _row(dtype, e, jj):
    """The M row of the `e`-th multiplication's column `jj`."""
    return e + MULTS * jj if dtype is Datatype.F64 else jj + COLUMNS * e


# -- the claim against the tables ------------------------------------------ #

@pytest.mark.parametrize("dtype", TYPES)
def test_d_lands_in_the_register_image(dtype):
    """``D[m(e, jj)][t]`` is slot `jj` of lane ``t + 16 e``: column `jj` of
    the `e`-th multiplication's lane `t`, as the nest's image holds it --
    ``m = e + 4 jj`` on the F64 instruction, ``m = jj + 4 e`` on the F32
    one."""
    op = OPS[dtype]
    for e in range(MULTS):
        for jj in range(COLUMNS):
            for t in range(THREADS):
                assert layouts.position(op, "D", _row(dtype, e, jj), t) == (
                    jj, t + THREADS * e)
    # And it is a bijection onto the accumulator: every slot of every lane
    # is one element's column.
    held = {layouts.position(op, "D", m, n) for m in range(16)
            for n in range(16)}
    assert held == {(s, lane) for s in range(4) for lane in range(WAVE)}


@pytest.mark.parametrize("dtype", TYPES)
def test_a_and_b_are_the_lanes_own_step(dtype):
    """A's step and B's step are the lane's place in the wave, its row of
    the tile its lane in the multiplication -- what `A_wave` reads."""
    op = OPS[dtype]
    for p in range(MULTS):
        for t in range(THREADS):
            lane = t + THREADS * p
            assert layouts.element_at(op, "A", 0, lane) == (0, t, p)
            assert layouts.element_at(op, "B", 0, lane) == (0, p, t)


def test_the_mapping_is_derived_and_declines_where_it_does_not_hold():
    # The two instructions stack the other way round in M.
    assert stacked.stacking(OP, 16) == stacked.ELEMENT_MINOR
    assert stacked.stacking(F32_OP, 16) == stacked.ELEMENT_MAJOR
    for op in (OP, F32_OP):
        assert stacked.fits(op, 16)
        # Thirty-two lanes: the halves of a multiplication hold other rows.
        assert not stacked.fits(op, 32)
        assert not stacked.fits(op, 64)
    # Neither the lane-batched tile nor a 16x16x1 one: their K is not the
    # wave's multiplications.
    for name in ("mfma_f32_4x4x1f32", "mfma_f32_16x16x1f32"):
        other = next(o for o in catalog.MATRIX_OPS if o.builtin == name)
        assert stacked.stacking(other, 16) is None


# -- the emitter in a wave simulator --------------------------------------- #

class _Spec:
    def __init__(self):
        self.discarded = False

    def discard(self):
        self.discarded = True


class SimWriter:
    """The writer the emitter is handed, evaluating instead of building: a
    value is what it holds in each of the 64 lanes of a wave, the
    multiplication a lane belongs to its `lane // 16`.  The instruction is
    evaluated from the fragment tables, the lane gather as `ds_bpermute` reads
    -- any lane of the wave.  Enough of the builder for this emitter and no
    more: a call the simulator does not know fails the test."""

    def __init__(self):
        self.lanes = np.arange(WAVE)
        self.issues = 0
        self.gathers = 0
        self.barriers = 0
        self.shard_reads = 0
        self.mask = np.ones(WAVE, dtype=bool)
        self.issued = None

    # structure
    @contextlib.contextmanager
    def speculative(self):
        yield _Spec()

    @contextlib.contextmanager
    def AnonymousScope(self):
        yield self

    # values
    def thread_id(self, axis="x", raw=False):
        if axis == "x":
            return self.lanes % THREADS
        # A wave of a block 16 wide: rows `4w .. 4w + 3`; any `w` will do.
        return 4 * 7 + self.lanes // THREADS

    def const(self, value, type_=None):
        return np.full(WAVE, value, dtype=float)

    def declare(self, type_=None, **kw):
        length = getattr(type_, "length", 1) or 1
        return np.zeros((WAVE, length)) if length > 1 else np.zeros(WAVE)

    def extract(self, vec, lane, type_=None, hint=""):
        return vec[:, lane]

    def op(self, name, type_, *args, hint="", **kw):
        a = [np.asarray(x) for x in args]
        if name == "rem":
            return np.remainder(a[0], a[1])
        if name == "div":
            return np.floor_divide(a[0], a[1])
        if name == "add":
            return a[0] + a[1]
        if name == "sub":
            return a[0] - a[1]
        if name == "min":
            return np.minimum(a[0], a[1])
        if name == "max":
            return np.maximum(a[0], a[1])
        if name == "mul":
            return a[0] * a[1]
        if name == "eq":
            return a[0] == a[1]
        if name == "lt":
            return a[0] < a[1]
        if name == "ge":
            return a[0] >= a[1]
        if name == "and":
            return a[0] & a[1]
        if name == "select":
            return np.where(a[0], a[1], a[2])
        raise AssertionError(f"the simulator has no {name!r}")

    def pack(self, type_, *parts, hint=""):
        return np.stack([np.asarray(x, dtype=float) for x in parts], axis=1)

    @contextlib.contextmanager
    def if_(self, cond):
        # The lanes a guarded statement runs for: what a chain's store under
        # it writes (`_close_chain`).
        outer = self.mask
        self.mask = np.asarray(cond) & outer
        try:
            yield self
        finally:
            self.mask = outer

    def barrier(self, participants="block", threads=None, handoff=False):
        # The rendezvous the shard route owes the wave: in the simulator the
        # lanes run in step anyway, so what is checked is that it is asked
        # for, of the wave, before any read (`shard_reads`).
        assert participants == "wave" and threads == THREADS and handoff
        self.barriers += 1

    def rawexpr(self, text, *args, **kw):
        m = re.fullmatch(r"tensorforge::gather<(\d+)>\(\{0\}, \{1\}\)", text)
        assert m, f"the simulator has no {text!r}"
        assert int(m.group(1)) == WAVE and kw.get("crosslane")
        self.gathers += 1
        value, lane = np.asarray(args[0]), np.asarray(args[1])
        assert ((0 <= lane) & (lane < WAVE)).all()
        return value[lane]

    def call(self, callee, type_, *args, **kw):
        op = next((o for o in OPS.values() if o.callee == callee), None)
        assert op is not None, callee
        assert self.issued in (None, op), "two instructions in one product"
        self.issued = op
        a, b, c, cbsz, abid, blgp = args
        assert (cbsz, abid, blgp) == (0, 0, 0)
        self.issues += 1
        amat = np.empty((op.m, op.k))
        bmat = np.empty((op.k, op.n))
        cmat = np.empty((op.m, op.n))
        for m in range(op.m):
            for k in range(op.k):
                slot, lane = layouts.position(op, "A", m, k)
                amat[m, k] = a[lane]
        for k in range(op.k):
            for n in range(op.n):
                slot, lane = layouts.position(op, "B", k, n)
                bmat[k, n] = b[lane]
        for m in range(op.m):
            for n in range(op.n):
                slot, lane = layouts.position(op, "D", m, n)
                cmat[m, n] = c[lane, slot]
        dmat = amat @ bmat + cmat
        out = np.empty_like(c)
        for m in range(op.m):
            for n in range(op.n):
                slot, lane = layouts.position(op, "D", m, n)
                out[lane, slot] = dmat[m, n]
        return out


def _operands(op_rows, elements, kx, columns, zero_known=True, wave=True,
              shard=False, writer=None, dtype=Datatype.F64):
    """`MatmulOperands` over NumPy data, read the way the nest's accessors
    read: `op_rows` is the operator counted from the first slot's row, one
    column a step of the contraction; `elements[e]` is the `e`-th
    multiplication's per-element operand, one row a step.  The image of a
    per-element operand holds position `s` (`kx` before the first step) in
    lane `s % 16` of block `s // 16`; a position outside the contraction
    holds NaN, which no product may read."""
    rows, depth = op_rows.shape
    slots = -(-rows // THREADS)
    lanes = np.arange(WAVE)
    t, mult = lanes % THREADS, lanes // THREADS
    out = np.full((MULTS, slots * THREADS, columns), np.nan)

    def op_at(r, k):
        r = np.asarray(r)
        inside = (0 <= r) & (r < rows)
        return np.where(inside, op_rows[np.clip(r, 0, rows - 1), k], 0.0)

    def A(writer, var, i, k, *rest):
        assert var is None and not rest
        return op_at(THREADS * i + t, k)

    def A_wave(writer, i, k, mults):
        assert mults == MULTS and k + mults <= depth
        return np.array([op_at(THREADS * i + t[l], k + mult[l])[()]
                         for l in lanes])

    def B(writer, var, j, kb):
        assert var is None
        s = THREADS * kb + t - kx
        inside = (0 <= s) & (s < depth)
        values = np.array([elements[mult[l]][min(max(s[l], 0), depth - 1), j]
                           for l in lanes])
        return np.where(inside, values, np.nan)

    def C(writer, value, i, j):
        out[mult, THREADS * i + t, j] = value

    def A_zero(e_lo, e_hi, k_lo, k_hi):
        return bool((op_rows[e_lo:min(e_hi, rows), k_lo:k_hi] == 0).all())

    def B_shard(w, j, k, rows_, columns_, mults_):
        """Each multiplication's copy in shared memory, as the frontend
        described it: element `mult + mults` at step `k + rows`, column
        `j + columns`, which have to stay inside the copy."""
        assert w.barriers, "a read of other multiplications' copies " \
                           "before the wave met"
        w.shard_reads += 1
        e = mult + np.asarray(mults_)
        step = k + np.asarray(rows_)
        col = j + np.asarray(columns_)
        assert ((0 <= e) & (e < MULTS)).all()
        assert ((0 <= step) & (step < depth)).all()
        assert ((0 <= col) & (col < np.asarray(elements).shape[2])).all()
        return np.array([elements[e[l]][step[l], col[l]] for l in lanes])

    ops = MatmulOperands(
        A=A, B=B, C=C, sparse=None, lead_slots=slots, lead_elements=rows,
        n=columns, k=depth, kx=kx, threads=THREADS, a=dtype,
        b=dtype, accumulator=dtype, a_uniform=True,
        A_wave=A_wave if wave else None, lockstep=True,
        A_zero=A_zero if zero_known else None,
        B_shard=B_shard if shard else None)
    return ops, out


def _simulate(op_rows, elements, kx=0, columns=8, route="transpose",
              options=None, dtype=Datatype.F64, **kw):
    ops, out = _operands(op_rows, elements, kx, columns,
                         shard=route == "shard", dtype=dtype, **kw)
    writer = SimWriter()
    assert stacked.matmul_stacked(writer, ops,
                                  _ctx(dtype=dtype, amd_matrix_k4=route,
                                       **(options or {})),
                                  Span(Strategy.MATRIX, 0, columns))
    assert writer.issued is OPS[dtype]
    want = np.einsum("rk,ekj->erj", op_rows, np.asarray(elements)[:, :, :columns])
    got = out[:, :op_rows.shape[0], :columns]
    return got, want, writer


def _random(rows, depth, columns, seed, zero_blocks=()):
    rng = np.random.default_rng(seed)
    op_rows = rng.uniform(-1, 1, (rows, depth))
    for (r0, r1), (k0, k1) in zero_blocks:
        op_rows[r0:r1, k0:k1] = 0.0
    elements = rng.uniform(-1, 1, (MULTS, depth, columns))
    return op_rows, elements


@pytest.mark.parametrize("dtype", TYPES)
@pytest.mark.parametrize("route", ["transpose", "shard"])
@pytest.mark.parametrize("rows,depth,kx", [
    (55, 35, 0),     # SeisSol's order-6 volume: rows 1..54 from the slot
    (64, 36, 0),     # whole tiles
    (21, 56, 0),     # a 21-row chain, fMrT -> rDivM
    (20, 10, 0),     # order 4
    (55, 34, 1),     # a contraction from step 1, the derivative's
    (40, 13, 3),     # from step 3, ending inside a tile
])
def test_the_emitted_program_computes_every_element(rows, depth, kx, route,
                                                    dtype):
    op_rows, elements = _random(rows, depth, 8, seed=rows * 100 + depth)
    got, want, writer = _simulate(op_rows, elements, kx=kx, route=route,
                                  dtype=dtype)
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12)
    if route == "shard":
        # One read a fragment, behind one rendezvous, and no gather.
        assert writer.barriers == 1 and writer.gathers == 0
        assert writer.shard_reads == 2 * -(-(depth + kx) // 4)


@pytest.mark.parametrize("dtype", TYPES)
@pytest.mark.parametrize("route", ["transpose", "shard"])
def test_a_padded_last_tile_computes_the_ninth_column(route, dtype):
    """`Options.amd_matrix_k4_pad`: the ninth column in a third tile of four,
    its other three columns zeros -- which the gathers do not read and the
    shard reads inside its copy and drops."""
    op_rows, elements = _random(55, 35, 9, seed=9)
    got, want, writer = _simulate(op_rows, elements, columns=9, route=route,
                                  options={"amd_matrix_k4_pad": True},
                                  dtype=dtype)
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12)
    assert writer.issues == 3 * 4 * 9
    if route == "transpose":
        assert writer.gathers == 9 * (2 * COLUMNS + 1)
    ctx = _ctx(amd_matrix_k4=route, amd_matrix_k4_pad=True)
    assert amd.plan(Strategy.MATRIX, _shape(shared=True), 9, ctx) == (
        Span(Strategy.MATRIX, 0, 9),)
    # Narrower than a tile, the product stays on the chain.
    assert amd.plan(Strategy.MATRIX, _shape(shared=True), 3, ctx) == (
        Span(Strategy.DPP, 0, 3),)


def test_the_elements_are_not_interchangeable():
    """Every multiplication's own numbers come out in its own lanes: the
    same operator against four different elements -- swapped, two of them
    would agree with the reference only if all four were alike."""
    op_rows, elements = _random(55, 35, 8, seed=5)
    got, want, _ = _simulate(op_rows, elements)
    for e in range(MULTS):
        for other in range(MULTS):
            close = np.allclose(got[e], want[other], rtol=1e-12, atol=1e-12)
            assert close == (e == other)


def test_issues_are_twice_the_nonzero_operator_blocks():
    """A 16x4 block of the operator known zero issues nothing, for either
    tile of columns; two tiles of four columns for the eight."""
    blocks = [((0, 16), (4, 8)), ((16, 32), (0, 12)), ((48, 55), (28, 35))]
    op_rows, elements = _random(55, 35, 8, seed=7, zero_blocks=blocks)
    got, want, writer = _simulate(op_rows, elements)
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12)
    nonzero = sum(1 for rt in range(4) for kt in range(9)
                  if (op_rows[16 * rt:16 * rt + 16, 4 * kt:4 * kt + 4] != 0)
                  .any())
    assert nonzero == 36 - 1 - 3 - 2
    assert writer.issues == 2 * nonzero


def test_the_gathers_are_one_a_column_and_fragment():
    op_rows, elements = _random(55, 35, 8, seed=3)
    _, _, writer = _simulate(op_rows, elements)
    assert writer.gathers == 9 * 2 * COLUMNS


def test_the_shard_route_alone_declines_an_operand_not_in_shared_memory():
    """`shard` asked for and the operand in registers: nothing emitted, for
    the DPP chain to take; `true` takes the gathers instead."""
    op_rows, elements = _random(55, 35, 8, seed=2)
    ops, _ = _operands(op_rows, elements, 0, 8, shard=False)
    writer = SimWriter()
    assert not stacked.matmul_stacked(writer, ops, _ctx(amd_matrix_k4="shard"),
                              Span(Strategy.MATRIX, 0, 8))
    assert writer.issues == 0
    got, want, writer = _simulate(op_rows, elements, route=True)
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12)
    assert writer.gathers and not writer.barriers


def test_without_a_wave_read_the_span_is_declined():
    """No `A_wave` -- an operator not in memory in its own order -- and the
    scheme emits nothing, for the caller to take the DPP chain."""
    op_rows, elements = _random(55, 35, 8, seed=1)
    ops, _ = _operands(op_rows, elements, 0, 8, wave=False)
    writer = SimWriter()
    assert not stacked.matmul_stacked(writer, ops, _ctx(), Span(Strategy.MATRIX, 0, 8))
    assert writer.issues == 0


# -- what the generator makes of it ---------------------------------------- #

def _shape(threads=16, accumulator=Datatype.F64, uniform=True, width=1,
           sparse=False, shared=False):
    return ComputeShape(threads=threads, accumulator=accumulator,
                        sparse=sparse, explicit_simd=False, lead=54, depth=35,
                        lead_layout=lead_layout(threads, width),
                        a_uniform=uniform, b_shared=shared)


def test_the_shard_route_is_planned_only_over_an_operand_in_shared_memory():
    """`shard` asked of a per-element operand held in registers -- the time
    derivative's images -- plans nothing for the instruction: the DPP chain
    keeps the product whole, as without the option.  `true` takes the
    gathers there; the shared operand is what the builder keeps in place."""
    shard, auto = _ctx(amd_matrix_k4="shard"), _ctx(amd_matrix_k4=True)
    assert not stacked.stacks(_shape(), shard)
    assert stacked.stacks(_shape(shared=True), shard)
    assert stacked.stacks(_shape(), auto)
    assert amd.reads_shared(_shape(shared=True), shard)
    assert amd.reads_shared(_shape(shared=True), auto)
    assert not amd.reads_shared(_shape(shared=True), _ctx())


def test_the_scheme_is_offered_only_where_it_holds():
    ctx = _ctx()
    assert stacked.stacks(_shape(), ctx)
    assert Strategy.MATRIX in amd.strategies(_shape(), ctx)
    assert amd.plan(Strategy.MATRIX, _shape(), 9, ctx) == (
        Span(Strategy.MATRIX, 0, 8), Span(Strategy.DPP, 8, 9))
    assert amd.plan(Strategy.MATRIX, _shape(), 8, ctx) == (
        Span(Strategy.MATRIX, 0, 8),)
    assert amd.plan(Strategy.MATRIX, _shape(), 3, ctx) == (
        Span(Strategy.DPP, 0, 3),)
    # Off by default.
    off = Context(arch="gfx942", backend="hip", fp_type=Datatype.F64)
    assert not stacked.stacks(_shape(), off)
    assert Strategy.MATRIX not in amd.strategies(_shape(), off)
    # Thirty-two lanes, an operator per element, a packed lead operand, a
    # sparse second operand: declined, and the DPP chain keeps them.
    for shape in (_shape(threads=32), _shape(uniform=False), _shape(width=2),
                  _shape(sparse=True)):
        assert not stacked.stacks(shape, ctx)
    # F32 at sixteen lanes takes its own instruction, planned alike; at
    # thirty-two, and with the option off, the lane-batched tile.
    f32 = _ctx(dtype=Datatype.F32)
    single = _shape(accumulator=Datatype.F32)
    assert stacked.stacks(single, f32)
    assert amd.plan(Strategy.MATRIX, single, 9, f32) == (
        Span(Strategy.MATRIX, 0, 8), Span(Strategy.DPP, 8, 9))
    assert not stacked.stacks(_shape(threads=32, accumulator=Datatype.F32),
                              f32)
    assert not stacked.stacks(single, _ctx(dtype=Datatype.F32,
                                           amd_matrix_k4=False))
    # No F64 matrix core on RDNA; CDNA 2 has the instruction.
    assert not stacked.stacks(_shape(), _ctx(arch="gfx1150"))
    assert stacked.stacks(_shape(), _ctx(arch="gfx90a"))


def test_a_stacked_operator_is_not_prepared_in_k_quads():
    """`prepared_order` under the option at sixteen lanes: the scheme reads
    the operator by its own steps (`A_wave`), so no k-quad order is offered
    for it -- one would hand the product back to the lane-batched tile --
    while without the option, or at thirty-two lanes, the tile's quads
    are."""
    asked = dict(columns=9, lead=56, depth=35)
    on, off = (_ctx(dtype=Datatype.F32),
               _ctx(dtype=Datatype.F32, amd_matrix_k4=False))
    assert amd.prepared_order((56, 56), Datatype.F32, on, threads=16,
                              **asked) is None
    assert amd.prepared_order((56, 56), Datatype.F32, off, threads=16,
                              **asked) is not None
    assert amd.prepared_order((56, 56), Datatype.F32, on, threads=32,
                              **asked) is not None


def test_the_multiplications_of_a_wave_run_in_step():
    ctx = _ctx()
    assert amd.convergence(Strategy.MATRIX, _shape(), 9, ctx) is \
        Uniformity.MULTGROUP


def _kernel(name, arch="gfx942", threads=16, **options):
    path = next(CASES.rglob(f"{name}.py"))
    spec = importlib.util.spec_from_file_location("tf_stacked__" + name, path)
    case = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(case)
    asked = dict(getattr(case, "OPTIONS", {}))
    asked.update(options)
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore")
        ctx = Context(arch=arch, backend="hip", fp_type=case.DTYPE,
                      options=Options(**asked))
        lanes = LaneConfig(num_threads=threads, num_active_threads=threads,
                           lead_width=1)
        gen = Generator(case.descr_list(), ctx, lanes=lanes)
        gen.generate()
    return case, gen.get_kernel()


def _nonzero_blocks(values, r0, r1, k0, k1):
    """16x4 blocks of the window `[r0, r1) x [k0, k1)`, counted from the slot
    holding row `r0` -- the tiles the scheme issues for."""
    first = r0 - r0 % THREADS
    count = 0
    for rt in range(first, r1, THREADS):
        for kt in range(k0, k1, 4):
            block = values[max(rt, r0):min(rt + THREADS, r1), kt:min(kt + 4, k1)]
            count += bool((block != 0).any())
    return count


def test_the_volume_issues_twice_its_nonzero_blocks():
    case, src = _kernel("stacked_volume_f64")
    want = sum(2 * _nonzero_blocks(v, r0, r1, 0, 35)
               for v, (r0, r1) in zip(case.OPERATORS, case.WINDOWS))
    assert len(MFMA.findall(src)) == want


def test_the_f32_volume_issues_its_own_instruction():
    """Single precision at sixteen lanes: `mfma_f32_16x16x4f32` for every
    operator product, twice its nonzero blocks as in double precision; the
    lane-batched tile is left the products whose operator is per element,
    the star matrices -- and has them all without the option."""
    case, src = _kernel("stacked_volume_f32")
    want = sum(2 * _nonzero_blocks(v, r0, r1, 0, 35)
               for v, (r0, r1) in zip(case.OPERATORS, case.WINDOWS))
    assert len(F32_MFMA.findall(src)) == want
    assert not MFMA.search(src)
    _, off = _kernel("stacked_volume_f32", amd_matrix_k4=False)
    assert not F32_MFMA.search(off)
    assert "__builtin_amdgcn_mfma_f32_4x4x1f32(" in off


@pytest.mark.parametrize("name", ["stacked_from_step_one_f64",
                                  "stacked_chain21_f64",
                                  "stacked_ragged_f64"])
def test_the_other_shapes_take_it(name):
    """A contraction from step one, a 21-row chain and a batch ending inside
    a wave: on CDNA every operator product of them issues the instruction --
    through the shard where its per-element operand is a temporary in
    shared memory, through the gathers where it is not."""
    _, src = _kernel(name)
    assert MFMA.search(src)
    _, gathered = _kernel(name, amd_matrix_k4="transpose")
    assert "tensorforge::gather<64>" in gathered


def test_the_shard_reads_the_temporary_where_its_producer_stored_it():
    """Under `shard` the volume's temporaries stay in shared memory, read
    across the wave behind a fence of it, and nothing is gathered; under
    `transpose` they are register images the gathers read, and no
    multiplication reads another's shared memory."""
    _, shard = _kernel("stacked_volume_f64", amd_matrix_k4="shard")
    assert "__builtin_amdgcn_fence(__ATOMIC_ACQ_REL, \"wavefront\")" in shard
    assert "tensorforge::gather<64>" not in shard
    assert len(MFMA.findall(shard)) > 0
    _, gathered = _kernel("stacked_volume_f64", amd_matrix_k4="transpose")
    assert "__builtin_amdgcn_fence" not in gathered
    assert "tensorforge::gather<64>" in gathered
    assert len(MFMA.findall(gathered)) == len(MFMA.findall(shard))


@pytest.mark.parametrize("arch,threads,options", [
    ("gfx942", 16, {"amd_matrix_k4": False}),
    ("gfx942", 32, {}),
    ("gfx1150", 16, {}),
])
def test_where_it_declines_the_kernel_is_todays(arch, threads, options):
    _, src = _kernel("stacked_volume_f64", arch=arch, threads=threads,
                     **options)
    assert not MFMA.search(src)


def test_gfx90a_issues_it_too():
    _, src = _kernel("stacked_volume_f64", arch="gfx90a")
    assert MFMA.search(src)
