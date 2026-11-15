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
import json
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
BF16_OP = next(o for o in catalog.MATRIX_OPS
               if o.builtin == "mfma_f32_16x16x16bf16_1k")
ORDERS = {Datatype.F64: stacked.QUARTER_MINOR,
          Datatype.F32: stacked.QUARTER_MAJOR}
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
    # The two instructions lay M out the other way round over the quarters.
    assert stacked.stacking(OP) == stacked.QUARTER_MINOR
    assert stacked.stacking(F32_OP) == stacked.QUARTER_MAJOR
    for op in (OP, F32_OP):
        # A multiplication of whole quarters of the wave: one, two or four.
        for threads in (16, 32, 64):
            assert stacked.fits(op, threads)
        for threads in (8, 48, 128):
            assert not stacked.fits(op, threads)
    # Neither the lane-batched tile nor a 16x16x1 one: their K is not the
    # wave's quarters.
    for name in ("mfma_f32_4x4x1f32", "mfma_f32_16x16x1f32"):
        other = next(o for o in catalog.MATRIX_OPS if o.builtin == name)
        assert stacked.stacking(other) is None


# -- the emitter in a wave simulator --------------------------------------- #

class _Spec:
    def __init__(self):
        self.discarded = False

    def discard(self):
        self.discarded = True


class SimWriter:
    """The writer the emitter is handed, evaluating instead of building: a
    value is what it holds in each of the 64 lanes of a wave, the
    multiplication a lane belongs to its `lane // threads`.  The instruction
    is evaluated from the fragment tables, the lane gather as `ds_bpermute`
    reads -- any lane of the wave.  Enough of the builder for this emitter
    and no more: a call the simulator does not know fails the test."""

    def __init__(self, threads=THREADS):
        self.threads = threads
        self.lanes = np.arange(WAVE)
        self.issues = 0
        self.gathers = 0
        self.barriers = 0
        self.shard_reads = 0
        self.mask = np.ones(WAVE, dtype=bool)
        self.issued = None
        self.raw = []
        self.splits = 0

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
            return self.lanes % self.threads
        # A wave of a block 16 wide: rows `4w .. 4w + 3`; any `w` will do.
        return 4 * 7 + self.lanes // self.threads

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
        if name == "le":
            return a[0] <= a[1]
        if name == "and":
            return a[0] & a[1]
        if name == "bitxor":
            return np.bitwise_xor(a[0], a[1])
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

    def __call__(self, text, *args, accesses=None, **kw):
        # A raw statement: the scheduling groups of a pipelined product
        # (`Options.amd_matrix_k4_pipeline`), which order nothing here.
        assert text.startswith("__builtin_amdgcn_sched_group_barrier("), text
        assert accesses == ()
        self.raw.append(text)

    def barrier(self, participants="block", threads=None, handoff=False):
        # The rendezvous the shard route owes the wave: in the simulator the
        # lanes run in step anyway, so what is checked is that it is asked
        # for, of the wave, before any read (`shard_reads`).
        assert participants == "wave" and threads == self.threads and handoff
        self.barriers += 1

    def rawexpr(self, text, *args, **kw):
        m = re.fullmatch(r"tensorforge::gather<(\d+)>\(\{0\}, \{1\}\)", text)
        assert m, f"the simulator has no {text!r}"
        assert int(m.group(1)) == WAVE and kw.get("crosslane")
        self.gathers += 1
        value, lane = np.asarray(args[0]), np.asarray(args[1])
        assert ((0 <= lane) & (lane < WAVE)).all()
        return value[lane]

    def call_stmt(self, callee, *args, writes=(), **kw):
        # The runtime split of four floats into three BF16 terms each,
        # rounded to nearest even as `static_cast<__bf16>` does, the terms
        # kept as the floats they stand for.
        assert callee == "tensorforge::splitFloatx4BF16", callee
        outs, values = args[:3], args[3:]
        assert len(values) == 4 and len(writes) == 3
        self.splits += 1
        rest = np.stack([np.asarray(v, dtype=np.float32) for v in values],
                        axis=1)
        for out in outs:
            term = _bf16(rest)
            out[...] = term
            rest = (rest - term).astype(np.float32)

    def call(self, callee, type_, *args, **kw):
        if callee == "tensorforge::packBF16x4":
            # Terms the host split, stored as the floats they stand for.
            return np.asarray(args[0])
        op = next((o for o in list(OPS.values()) + [BF16_OP]
                   if o.callee == callee), None)
        assert op is not None, callee
        assert self.issued in (None, op), "two instructions in one product"
        self.issued = op
        a, b, c, cbsz, abid, blgp = args
        assert (cbsz, abid, blgp) == (0, 0, 0)
        self.issues += 1
        amat = np.empty((op.m, op.k))
        bmat = np.empty((op.k, op.n))
        cmat = np.empty((op.m, op.n))
        a, b = np.asarray(a), np.asarray(b)
        for m in range(op.m):
            for k in range(op.k):
                slot, lane = layouts.position(op, "A", m, k)
                amat[m, k] = a[lane, slot] if a.ndim == 2 else a[lane]
        for k in range(op.k):
            for n in range(op.n):
                slot, lane = layouts.position(op, "B", k, n)
                bmat[k, n] = b[lane, slot] if b.ndim == 2 else b[lane]
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


def _bf16(x):
    """`x` rounded to BF16, to nearest even, as a float32."""
    x = np.asarray(x, dtype=np.float32)
    bits = x.view(np.uint32).astype(np.uint64)
    bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return bits.astype(np.uint32).view(np.float32)


def _split_bf16(values):
    """Three BF16 terms of `values`, as `prepare.split_bf16` stores them."""
    rest = np.asarray(values, dtype=np.float32)
    terms = []
    for _ in range(3):
        term = _bf16(rest)
        terms.append(term.astype(float))
        rest = (rest - term).astype(np.float32)
    return terms


def _operands(op_rows, elements, kx, columns, zero_known=True, wave=True,
              shard=False, writer=None, chain=None, dtype=Datatype.F64,
              threads=THREADS, window=None, prepared=False):
    """`MatmulOperands` over NumPy data, read the way the nest's accessors
    read: `op_rows` is the operator counted from the first slot's row, one
    column a step of the contraction; `elements[e]` is the `e`-th
    multiplication's per-element operand, one row a step.  The image of a
    per-element operand holds position `s` (`kx` before the first step) in
    lane `s % threads` of block `s // threads`; a position outside the
    contraction holds NaN, which no product may read.  `window`, rows
    `[r0, r1)` of the walk, is the operation's own, which a read by row
    (`A_lanes`) has to stay in."""
    rows, depth = op_rows.shape
    first, end = window if window is not None else (0, rows)
    T, mults = threads, WAVE // threads
    slots = -(-rows // T)
    lanes = np.arange(WAVE)
    t, mult = lanes % T, lanes // T
    out = np.full((mults, slots * T, columns), np.nan)

    def op_at(r, k):
        r = np.asarray(r)
        inside = (0 <= r) & (r < rows)
        return np.where(inside, op_rows[np.clip(r, 0, rows - 1), k], 0.0)

    def A(writer, var, i, k, *rest):
        assert var is None and not rest
        return op_at(T * i + t, k)

    def A_wave(writer, i, k, count):
        assert count == mults and k + count <= depth
        return np.array([op_at(T * i + t[l], k + mult[l])[()]
                         for l in lanes])

    def A_lanes(writer, k, rows_, steps_):
        """The operator at a row and a step of every lane's own, which have
        to stay inside it."""
        r = np.broadcast_to(np.asarray(rows_), (WAVE,))
        step = k + np.broadcast_to(np.asarray(steps_), (WAVE,))
        assert ((first <= r) & (r < end)).all()
        assert ((0 <= step) & (step < depth)).all()
        return op_rows[r, step]

    def B(writer, var, j, kb):
        assert var is None
        s = T * kb + t - kx
        inside = (0 <= s) & (s < depth)
        values = np.array([elements[mult[l]][min(max(s[l], 0), depth - 1), j]
                           for l in lanes])
        return np.where(inside, values, np.nan)

    def C(writer, value, i, j):
        out[mult, T * i + t, j] = value

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
        assert ((0 <= e) & (e < mults)).all()
        assert ((0 <= step) & (step < depth)).all()
        assert ((0 <= col) & (col < np.asarray(elements).shape[2])).all()
        return np.array([elements[e[l]][step[l], col[l]] for l in lanes])

    def A_slot(writer, slot, parts=1, shift=None, width=1):
        """An operator the host split into k-quads of its three terms
        (`amd.prepared_order`): slot `(q slots + i) T width` holds quad `q`
        of row slot `i`, `shift` more slots on in the lane."""
        assert parts == 3 and width == 4
        nonlead = slot // (T * width) + np.broadcast_to(
            np.asarray(0 if shift is None else shift), (WAVE,))
        quad, i = nonlead // slots, nonlead % slots
        row = T * i + t
        assert ((0 <= quad) & (quad * 4 < depth + 3)).all()
        cells = np.zeros((WAVE, 4))
        for c in range(4):
            step = quad * 4 + c
            ok = (row < rows) & (step < depth)
            cells[:, c] = np.where(ok, op_rows[np.clip(row, 0, rows - 1),
                                               np.clip(step, 0, depth - 1)],
                                   0.0)
        return _split_bf16(cells)

    ops = MatmulOperands(
        A=A, B=B, C=C, sparse=None, lead_slots=slots,
        lead_elements=end - first,
        n=columns, k=depth, kx=kx, threads=T, a=dtype,
        lead_first=first,
        b=dtype, accumulator=dtype, a_uniform=True,
        A_wave=A_wave if wave else None, lockstep=True,
        A_lanes=A_lanes if wave else None,
        A_zero=A_zero if zero_known else None,
        B_shard=B_shard if shard else None, chain=chain,
        A_slot=A_slot if prepared else None, a_parts=3 if prepared else 1)
    return ops, out


def _simulate(op_rows, elements, kx=0, columns=8, route="transpose",
              options=None, dtype=Datatype.F64, threads=THREADS, **kw):
    ops, out = _operands(op_rows, elements, kx, columns,
                         shard=route == "shard", dtype=dtype,
                         threads=threads, **kw)
    writer = SimWriter(threads)
    assert stacked.matmul_stacked(writer, ops,
                                  _ctx(dtype=dtype, amd_matrix_k4=route,
                                       **(options or {})),
                                  Span(Strategy.MATRIX, 0, columns))
    assert writer.issued is (BF16_OP if (options or {}).get(
        "amd_matrix_k4_emulate") else OPS[dtype])
    want = np.einsum("rk,ekj->erj", op_rows, np.asarray(elements)[:, :, :columns])
    got = out[:, :op_rows.shape[0], :columns]
    return got, want, writer


def _random(rows, depth, columns, seed, zero_blocks=(), threads=THREADS):
    rng = np.random.default_rng(seed)
    op_rows = rng.uniform(-1, 1, (rows, depth))
    for (r0, r1), (k0, k1) in zero_blocks:
        op_rows[r0:r1, k0:k1] = 0.0
    elements = rng.uniform(-1, 1, (WAVE // threads, depth, columns))
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


class _Output:
    """An output in global memory, an element a multiplication of the wave,
    read and written the way a chain reads and writes it
    (`AccumulatorChain`): at a row of the walk and a column, every lane its
    own multiplication's element."""

    def __init__(self, values, threads=THREADS):
        self.values = np.array(values, dtype=float)
        self.threads = threads
        self.reads = self.writes = 0

    def load(self, writer, row, col, predicate=None):
        self.reads += 1
        lanes = np.arange(WAVE)
        row = np.broadcast_to(np.asarray(row), (WAVE,))
        col = np.broadcast_to(np.asarray(col), (WAVE,))
        rows, cols = self.values.shape[1:]
        got = self.values[lanes // self.threads, np.clip(row, 0, rows - 1),
                          np.clip(col, 0, cols - 1)]
        inside = (0 <= row) & (row < rows) & (0 <= col) & (col < cols)
        if predicate is not None:
            inside = inside & np.asarray(predicate)
        return np.where(inside, got, 0.0)

    def store(self, writer, value, row, col):
        self.writes += 1
        lanes = np.arange(WAVE)
        row = np.broadcast_to(np.asarray(row), (WAVE,))
        col = np.broadcast_to(np.asarray(col), (WAVE,))
        mask = writer.mask & (0 <= row) & (row < self.values.shape[1])
        assert (col[mask] < self.values.shape[2]).all(), \
            "a padded column written"
        self.values[lanes[mask] // self.threads, row[mask], col[mask]] = \
            np.asarray(value)[mask]


@pytest.mark.parametrize("dtype", TYPES)
@pytest.mark.parametrize("route", ["transpose", "shard"])
def test_a_chain_adds_every_product_into_one_set_of_accumulators(route,
                                                                 dtype):
    """Three products into one output, SeisSol's volume: the old value read
    into the accumulators by the first, every product issued into them, the
    sum written by the last -- each row and column once, and nothing past
    the chain's window.  The windows differ, rows 1 to 54, 55 and 56 of the
    slots' walk, and a product adds zeros where its operator has no rows."""
    from tensorforge.backend.instructions.compute.matmul import AccumulatorChain
    rng = np.random.default_rng(41)
    windows = ((1, 54), (1, 55), (1, 56))
    rows, depth, columns = 64, 35, 9
    q0 = rng.uniform(-1, 1, (MULTS, rows, columns))
    out = _Output(q0)
    state = {}
    want = q0.copy()
    writer = SimWriter()
    ctx = _ctx(dtype=dtype, amd_matrix_k4=route, mma_d_layout=True)
    for d, (r0, r1) in enumerate(windows):
        op_rows = np.zeros((rows, depth))
        op_rows[r0:r1] = rng.uniform(-1, 1, (r1 - r0, depth))
        elements = rng.uniform(-1, 1, (MULTS, depth, columns))
        chain = AccumulatorChain(opens=d == 0, closes=d == len(windows) - 1,
                                 state=state, rows=(1, 56), load=out.load,
                                 store=out.store)
        ops, _ = _operands(op_rows, elements, 0, columns,
                           shard=route == "shard", chain=chain, dtype=dtype)
        ops = dataclasses.replace(ops, C=_no_write)
        assert stacked.matmul_stacked(writer, ops, ctx,
                                      Span(Strategy.MATRIX, 0, columns))
        want[:, r0:r1] += np.einsum("rk,ekj->erj", op_rows[r0:r1], elements)
    np.testing.assert_allclose(out.values, want, rtol=1e-12, atol=1e-12)
    # Read once and written once: the window's 4 slots of 9 columns.
    assert out.reads == out.writes == 4 * columns
    # Every issue into the one set: three products, 3 tiles of columns.
    assert writer.issues == 3 * 4 * 9 * 3


#: The wider multiplications: lanes, and a tile's columns (`4 H`).
WIDE = [(32, 8), (64, 16)]


@pytest.mark.parametrize("dtype", TYPES)
@pytest.mark.parametrize("route", ["transpose", "shard"])
@pytest.mark.parametrize("threads,width", WIDE)
@pytest.mark.parametrize("rows,depth,kx", [
    (55, 35, 0),     # SeisSol's order-6 volume
    (64, 36, 0),     # whole tiles
    (21, 56, 0),     # a 21-row chain
    (120, 84, 0),    # order 8
    (55, 34, 1),     # a contraction from step 1, the derivative's
    (40, 13, 3),     # from step 3, ending inside a tile
])
def test_a_wide_multiplication_computes_every_element(rows, depth, kx,
                                                      threads, width, route,
                                                      dtype):
    """Thirty-two and sixty-four lanes a multiplication: the quarters of one
    take more columns, the rows of an issue's tile land in every quarter,
    and the quarters trade them back into the image before the epilogue
    (`stacked._relayout`) -- one tile of columns, its every element where
    the nest keeps it."""
    op_rows, elements = _random(rows, depth, width, seed=rows + depth,
                                threads=threads)
    got, want, writer = _simulate(op_rows, elements, kx=kx, columns=width,
                                  route=route, dtype=dtype, threads=threads)
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12)
    H = threads // 16
    slots = -(-rows // threads)
    # `H - 1` gathers a slot of D, four slots a tile, a tile of rows each.
    relayout = slots * 4 * (H - 1)
    if route == "shard":
        assert writer.barriers == 1
        assert writer.gathers == relayout
        assert writer.shard_reads == -(-(depth + kx) // 4)
    else:
        assert writer.gathers == relayout + width * -(-(depth + kx) // 4)


@pytest.mark.parametrize("dtype", TYPES)
@pytest.mark.parametrize("threads,width", WIDE)
def test_a_wide_padded_tile_and_its_issues(threads, width, dtype):
    """The ninth column in one more tile -- the second of eight columns, or
    the only one of sixteen -- padded with zeros; and an issue a nonzero
    block of sixteen rows and four steps for every tile, as at sixteen
    lanes."""
    blocks = [((0, 16), (4, 8)), ((16, 32), (0, 12)), ((48, 55), (28, 35))]
    op_rows, elements = _random(55, 35, 9, seed=11, zero_blocks=blocks,
                                threads=threads)
    got, want, writer = _simulate(op_rows, elements, columns=9,
                                  options={"amd_matrix_k4_pad": True},
                                  dtype=dtype, threads=threads)
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12)
    nonzero = sum(1 for rt in range(4) for kt in range(9)
                  if (op_rows[16 * rt:16 * rt + 16, 4 * kt:4 * kt + 4] != 0)
                  .any())
    assert writer.issues == -(-9 // width) * nonzero


@pytest.mark.parametrize("dtype", TYPES)
@pytest.mark.parametrize("route", ["transpose", "shard"])
@pytest.mark.parametrize("threads,width", WIDE)
def test_a_wide_chain_reads_and_writes_at_the_lanes_own_cells(threads, width,
                                                              route, dtype):
    """A chain at thirty-two and sixty-four lanes: D is not the image there,
    and needs not be -- the first product reads the output's old value at
    the row and column D gives every lane, the last writes it back there,
    each cell once, and nothing is gathered after the issues."""
    from tensorforge.backend.instructions.compute.matmul import AccumulatorChain
    rng = np.random.default_rng(43)
    mults = WAVE // threads
    windows = ((1, 54), (1, 55), (1, 56))
    rows, depth, columns = 64, 35, 9
    q0 = rng.uniform(-1, 1, (mults, rows, columns))
    out = _Output(q0, threads)
    state = {}
    want = q0.copy()
    writer = SimWriter(threads)
    ctx = _ctx(dtype=dtype, amd_matrix_k4=route, mma_d_layout=True)
    for d, (r0, r1) in enumerate(windows):
        # Outside a member's window the operator holds numbers that are not
        # its product's: a read by row has to keep them out.
        op_rows = rng.uniform(5, 9, (rows, depth))
        op_rows[r0:r1] = rng.uniform(-1, 1, (r1 - r0, depth))
        elements = rng.uniform(-1, 1, (mults, depth, columns))
        chain = AccumulatorChain(opens=d == 0, closes=d == len(windows) - 1,
                                 state=state, rows=(1, 56), load=out.load,
                                 store=out.store)
        ops, _ = _operands(op_rows, elements, 0, columns,
                           shard=route == "shard", chain=chain, dtype=dtype,
                           threads=threads, window=(r0, r1), zero_known=False)
        ops = dataclasses.replace(ops, C=_no_write)
        assert stacked.matmul_stacked(writer, ops, ctx,
                                      Span(Strategy.MATRIX, 0, columns))
        want[:, r0:r1] += np.einsum("rk,ekj->erj", op_rows[r0:r1], elements)
    np.testing.assert_allclose(out.values, want, rtol=1e-12, atol=1e-12)
    # Four tiles of sixteen rows, and every slot of D some lane keeps: a
    # slot's column is `c0 + 4 h + s`, and it reads at all where `c0 + s`
    # is one of the output's.
    kept = sum(1 for c0 in range(0, columns, width) for s in range(4)
               if c0 + s < columns)
    assert out.reads == out.writes == 4 * kept
    if route == "shard":
        assert writer.gathers == 0


@pytest.mark.parametrize("dtype", TYPES)
@pytest.mark.parametrize("threads", [16, 32])
def test_a_chain_over_a_merged_runs_trips(threads, dtype):
    """`localFluxAll`'s four faces as one merged loop (`dchain.DChain.loop`):
    the member opens the chain ahead of the loop, issues into the
    accumulators the loop carries on every trip -- with that trip's operator
    and element -- and closes it behind; the output is read and written once,
    and every trip's product is in it."""
    from tensorforge.backend.instructions.compute.matmul import AccumulatorChain
    rng = np.random.default_rng(47)
    mults = WAVE // threads
    rows, depth, columns, trips = 56, 21, 9, 4
    q0 = rng.uniform(-1, 1, (mults, rows, columns))
    out = _Output(q0, threads)
    state = {}
    want = q0.copy()
    writer = SimWriter(threads)
    ctx = _ctx(dtype=dtype, mma_d_layout=True)
    data = [(rng.uniform(-1, 1, (rows, depth)),
             rng.uniform(-1, 1, (mults, depth, columns)))
            for _ in range(trips)]
    phases = ["open"] + ["body"] * trips + ["close"]
    for n, phase in enumerate(phases):
        op_rows, elements = data[min(max(n - 1, 0), trips - 1)]
        chain = AccumulatorChain(opens=True, closes=True, state=state,
                                 rows=(0, rows), load=out.load,
                                 store=out.store, phase=phase)
        ops, _ = _operands(op_rows, elements, 0, columns, chain=chain,
                           dtype=dtype, threads=threads)
        ops = dataclasses.replace(ops, C=_no_write)
        issued = writer.issues
        assert stacked.matmul_stacked(writer, ops, ctx,
                                      Span(Strategy.MATRIX, 0, columns))
        assert (writer.issues > issued) == (phase == "body")
        if phase == "body":
            want += np.einsum("rk,ekj->erj", op_rows, elements)
    np.testing.assert_allclose(out.values, want, rtol=1e-12, atol=1e-12)
    width = 16 // (WAVE // threads)
    kept = sum(1 for c0 in range(0, columns, width) for s in range(4)
               if c0 + s < columns)
    assert out.reads == out.writes == -(-rows // 16) * kept


@pytest.mark.parametrize("dtype", TYPES)
@pytest.mark.parametrize("route", ["transpose", "shard"])
@pytest.mark.parametrize("threads", [16, 32])
@pytest.mark.parametrize("level", [1, 2])
def test_a_pipelined_product_computes_every_element(level, threads, route,
                                                    dtype):
    """`Options.amd_matrix_k4_pipeline`: the fragments of the next step read
    ahead of this step's issues, and at level 2 every issue but the last
    step's followed by a scheduling group of reads -- the same numbers."""
    width = 16 // (WAVE // threads)
    op_rows, elements = _random(55, 35, width, seed=59, threads=threads)
    got, want, writer = _simulate(op_rows, elements, columns=width,
                                  route=route, dtype=dtype, threads=threads,
                                  options={"amd_matrix_k4_pipeline": level})
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12)
    last = sum(1 for rt in range(4) if (op_rows[16 * rt:16 * rt + 16,
                                                32:35] != 0).any())
    assert len(writer.raw) == (2 * (writer.issues - last) if level == 2
                               else 0)


@pytest.mark.parametrize("route", ["transpose", "shard"])
@pytest.mark.parametrize("rows,depth,kx", [
    (55, 35, 0),     # SeisSol's order-6 volume
    (21, 56, 0),     # a 21-row chain
    (20, 10, 0),     # order 4, one issue of sixteen steps
    (55, 34, 1),     # from step 1, the derivative's
    (40, 13, 3),     # from step 3, ending inside an issue
])
@pytest.mark.parametrize("prepared", [False, True])
def test_an_emulated_product_holds_single_precision(rows, depth, kx, route,
                                                   prepared):
    """`Options.amd_matrix_k4_emulate`: the F32 product as six BF16 term
    products on `mfma_f32_16x16x16bf16_1k`, sixteen steps an issue and four a
    lane, into the F32 tile's accumulator -- every element at the precision
    of an F32 product (1e-6 normwise), where one BF16 product would hold 1e-2."""
    op_rows, elements = _random(rows, depth, 8, seed=rows * 7 + depth)
    op_rows = op_rows.astype(np.float32).astype(float)
    elements = np.asarray(elements, dtype=np.float32).astype(float)
    if prepared and kx % 4:
        pytest.skip("the operator is offered split where it starts on a quad")
    got, want, writer = _simulate(op_rows, elements, kx=kx, route=route,
                                  dtype=Datatype.F32,
                                  options={"amd_matrix_k4_emulate": True},
                                  prepared=prepared)
    assert writer.issued is BF16_OP
    # The operator split by the host is read, not split again: only the
    # elements' fragments are, one a tile of columns and issue.
    if prepared:
        assert writer.splits == 2 * -(-(depth + kx) // 16)
    scale = np.abs(want).max()
    assert np.abs(got - want).max() / scale < 1e-6
    # Six products a nonzero block of sixteen rows and sixteen steps, every
    # tile of columns.
    blocks = -(-rows // 16) * -(-(depth + kx) // 16)
    assert writer.issues <= 6 * 2 * blocks


@pytest.mark.parametrize("route", ["transpose", "shard"])
@pytest.mark.parametrize("prepared", [False, True])
def test_an_emulated_chain_carries_every_cell_of_the_tile(route, prepared):
    """The BF16 tile in a chain: its D holds four cells a lane, as the F32
    tile's does, though its sixteen steps an issue made `m / k` one -- read
    so, the chain opened and closed one column of every four."""
    from tensorforge.backend.instructions.compute.matmul import AccumulatorChain
    rng = np.random.default_rng(43)
    windows = ((1, 54), (1, 55), (1, 56))
    rows, depth, columns = 64, 35, 9
    q0 = rng.uniform(-1, 1, (MULTS, rows, columns))
    out = _Output(q0)
    state = {}
    want = q0.copy()
    writer = SimWriter()
    ctx = _ctx(dtype=Datatype.F32, amd_matrix_k4=route, mma_d_layout=True,
               amd_matrix_k4_emulate=True)
    for d, (r0, r1) in enumerate(windows):
        op_rows = np.zeros((rows, depth))
        op_rows[r0:r1] = rng.uniform(-1, 1, (r1 - r0, depth))
        op_rows = op_rows.astype(np.float32).astype(float)
        elements = rng.uniform(-1, 1, (MULTS, depth, columns)).astype(
            np.float32).astype(float)
        chain = AccumulatorChain(opens=d == 0, closes=d == len(windows) - 1,
                                 state=state, rows=(1, 56), load=out.load,
                                 store=out.store)
        ops, _ = _operands(op_rows, elements, 0, columns,
                           shard=route == "shard", chain=chain,
                           dtype=Datatype.F32, prepared=prepared)
        ops = dataclasses.replace(ops, C=_no_write)
        assert stacked.matmul_stacked(writer, ops, ctx,
                                      Span(Strategy.MATRIX, 0, columns))
        want[:, r0:r1] += np.einsum("rk,ekj->erj", op_rows[r0:r1], elements)
    assert writer.issued is BF16_OP
    assert np.abs(out.values - want).max() / np.abs(want).max() < 1e-6
    assert out.reads == out.writes == 4 * columns


def _no_write(writer, value, i, j):
    raise AssertionError("a member of a chain writes the output only at its "
                         "end, through the chain")


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
    # Eight lanes, an operator per element, a packed lead operand, a sparse
    # second operand: declined, and the DPP chain keeps them.
    for shape in (_shape(threads=8), _shape(uniform=False), _shape(width=2),
                  _shape(sparse=True)):
        assert not stacked.stacks(shape, ctx)
    # Thirty-two and sixty-four lanes: tiles of eight and sixteen columns.
    assert amd.plan(Strategy.MATRIX, _shape(threads=32), 9, ctx) == (
        Span(Strategy.MATRIX, 0, 8), Span(Strategy.DPP, 8, 9))
    assert amd.plan(Strategy.MATRIX, _shape(threads=64), 9, ctx) == (
        Span(Strategy.DPP, 0, 9),)
    padded = _ctx(amd_matrix_k4_pad=True)
    assert amd.plan(Strategy.MATRIX, _shape(threads=64), 9, padded) == (
        Span(Strategy.MATRIX, 0, 9),)
    assert amd.plan(Strategy.MATRIX, _shape(threads=64), 7, padded) == (
        Span(Strategy.DPP, 0, 7),)
    # F32 takes its own instruction, planned alike; with the option off the
    # lane-batched tile.
    f32 = _ctx(dtype=Datatype.F32)
    single = _shape(accumulator=Datatype.F32)
    assert stacked.stacks(single, f32)
    assert amd.plan(Strategy.MATRIX, single, 9, f32) == (
        Span(Strategy.MATRIX, 0, 8), Span(Strategy.DPP, 8, 9))
    assert stacked.stacks(_shape(threads=32, accumulator=Datatype.F32), f32)
    assert not stacked.stacks(single, _ctx(dtype=Datatype.F32,
                                           amd_matrix_k4=False))
    # No F64 matrix core on RDNA; CDNA 2 has the instruction.
    assert not stacked.stacks(_shape(), _ctx(arch="gfx1150"))
    assert stacked.stacks(_shape(), _ctx(arch="gfx90a"))


def test_a_stacked_operator_is_not_prepared_in_k_quads():
    """`prepared_order` under the option: the scheme reads the operator in
    its own order (`A_wave`, `A_lanes`), so no k-quad order is offered for
    it -- one would hand the product back to the lane-batched tile --
    while without the option, or at a lane count the scheme does not take,
    the tile's quads are."""
    asked = dict(columns=9, lead=56, depth=35)
    on, off = (_ctx(dtype=Datatype.F32),
               _ctx(dtype=Datatype.F32, amd_matrix_k4=False))
    for threads in (16, 32):
        assert amd.prepared_order((56, 56), Datatype.F32, on, threads=threads,
                                  **asked) is None
        assert amd.prepared_order((56, 56), Datatype.F32, off,
                                  threads=threads, **asked) is not None
    assert amd.prepared_order((56, 56), Datatype.F32, on, threads=8,
                              **asked) is not None


def test_the_emulated_tile_takes_the_operator_split_only_unpadded():
    """The emulated tile's operator is split on the host -- three BF16 terms
    in k-quads -- only where the ninth column cannot go into a padded tile:
    under `mma_d_layout` or `amd_matrix_k4_pad` it is split in the kernel, the
    form that was never wrong (`amd.prepared_order`)."""
    asked = dict(columns=9, lead=56, depth=36)
    emulated = dict(dtype=Datatype.F32, amd_matrix_k4_emulate=True)
    split = amd.prepared_order((56, 56), Datatype.F32, _ctx(**emulated),
                               threads=16, **asked)
    assert split is not None and split.parts == 3
    for padding in ({"mma_d_layout": True}, {"amd_matrix_k4_pad": True}):
        assert amd.prepared_order((56, 56), Datatype.F32,
                                  _ctx(**emulated, **padding), threads=16,
                                  **asked) is None


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


def _operand_name(src, alias):
    """The kernel's name of the operand the description calls `alias`."""
    meta = json.loads(re.search(r"tensorforge-meta: (\{.*\})", src).group(1))
    return next(o["name"] for o in meta["operands"] if o["alias"] == alias)


@pytest.mark.parametrize("name,issue", [("stacked_volume_f64", MFMA),
                                        ("stacked_volume_f32", F32_MFMA)])
@pytest.mark.parametrize("route", ["transpose", "shard"])
def test_a_chain_reads_and_writes_the_output_once(name, issue, route):
    """`mma_d_layout`: the volume's three products add into one set of
    accumulators, the ninth column in a padded tile -- three issues a
    nonzero block -- `Q` read into them before the first and written after
    the last, once for each of the 4 slots and 9 columns of its window, and
    nothing added atomically."""
    case, src = _kernel(name, amd_matrix_k4=route, mma_d_layout=True)
    want = sum(3 * _nonzero_blocks(v, r0, r1, 0, 35)
               for v, (r0, r1) in zip(case.OPERATORS, case.WINDOWS))
    assert len(issue.findall(src)) == want
    q = re.escape("glb_" + _operand_name(src, "Q"))
    stores = len(re.findall(rf"^\s*{q}\[[^\]]*\] = ", src, re.MULTILINE))
    loads = len(re.findall(rf"{q}\[[^\]]*\]", src)) - stores
    assert loads == stores == 4 * 9
    assert not re.search(r"atomic_fadd|atomicAdd", src)


def test_a_chain_spans_a_merged_runs_trips():
    """Four products into `Q` merged into one loop (`stacked_merged_chain_f64`):
    the chain spans the loop's trips -- the body issues the product once,
    `Q` is read ahead of the loop and written behind it, a cell each, and
    nothing is added atomically -- where unchained every trip adds its tile
    into `Q` through the epilogue."""
    loop = re.compile(r"for \(int32_t (\w+) = 0; \1 < 4; \+\+\1\)")
    case, src = _kernel("stacked_merged_chain_f64")
    assert len(loop.findall(src)) == 1
    # 20 rows: two tiles; nine columns: three tiles, the last padded;
    # ten steps: three.
    assert len(MFMA.findall(src)) == 2 * 3 * 3
    assert not re.search(r"atomic_fadd|atomicAdd", src)
    q = re.escape("glb_" + _operand_name(src, "Q"))
    stores = len(re.findall(rf"^\s*{q}\[[^\]]*\] = ", src, re.MULTILINE))
    loads = len(re.findall(rf"{q}\[[^\]]*\]", src)) - stores
    # Two tiles of sixteen rows, nine columns in slots of four.
    assert loads == stores == 2 * 9
    _, alone = _kernel("stacked_merged_chain_f64", mma_d_layout=False)
    assert len(loop.findall(alone)) == 1
    assert re.search(r"atomic_fadd", alone)


@pytest.mark.parametrize("name,want", [("stacked_volume_f64", [3]),
                                       ("stacked_chain21_f64", [])])
def test_a_product_on_its_own_keeps_its_epilogue(name, want, monkeypatch):
    """`D_CHAIN_SINGLES`: the volume's three products into `Q` are one chain,
    while the 21-row product, the only one into its output, is none -- its
    D layout is the register image already, and its old value read before
    the issues would be held across them, where the epilogue adds it
    after."""
    from tensorforge.backend.instructions.compute import dchain
    made = []
    real = dchain.link

    def spy(context, instrs):
        out = real(context, instrs)
        chains = {id(x._dchain): x._dchain for x in instrs
                  if getattr(x, "_dchain", None) is not None}
        made.extend(len(c.members) for c in chains.values())
        return out
    monkeypatch.setattr(dchain, "link", spy)
    _kernel(name, mma_d_layout=True)
    assert sorted(made) == want
    assert amd.D_CHAINS and not amd.D_CHAIN_SINGLES


@pytest.mark.parametrize("name,size", [("stacked_volume_f64", 8),
                                       ("stacked_volume_f32", 4)])
def test_the_shard_arenas_start_a_quarter_bank_row_apart(name, size):
    """Under `shard` a fragment reads the four copies of a wave at once
    (`neighbor_arenas`): the arenas are staggered by a quarter of the 32
    banks' row, so that four copies by four columns of the volume's odd
    leading dimension (35) fall on distinct banks -- the stride the shard
    read steps between copies and the one the arenas are laid out at."""
    _, src = _kernel(name, amd_matrix_k4="shard")
    stride = int(re.search(r"\(\(threadIdx\.x [%/] 4\) - v\d+_m\) \* (\d+);",
                           src).group(1))
    row = 32 * 4 // size
    assert stride % row == row // 4
    assert f"[{stride} * threadIdx.y + " in src


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
    ("gfx942", 8, {}),
    ("gfx1150", 16, {}),
])
def test_where_it_declines_the_kernel_is_todays(arch, threads, options):
    _, src = _kernel("stacked_volume_f64", arch=arch, threads=threads,
                     **options)
    assert not MFMA.search(src)


def test_gfx90a_issues_it_too():
    _, src = _kernel("stacked_volume_f64", arch="gfx90a")
    assert MFMA.search(src)
