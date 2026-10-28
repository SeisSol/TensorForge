# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""The emitter against the plans it emits.

`exchange_codegen` turns three plans into calls. What can be checked here is
that it is faithful to them -- every swap, every merge, every issue, in the
order the plans give and with the operands they name. What cannot is whether
the instruction then computes the right thing: that is the plans' claim, which
`test_amd_reorder.py` checks in the wave simulator, and beyond that it wants a
machine.

The path is deliberately not routed. `offers` does not name it, so nothing
generates it and no snapshot moves; wiring it up changes every F64 kernel on
CDNA 2 and later.
"""

from __future__ import annotations

import collections

import pytest

from tensorforge.common.basic_types import Datatype
from tensorforge.backend.instructions.compute.primitives.amd import (
    exchange_codegen, reorder)


class Recorder:
    """Records the call sequence instead of building IR.

    Enough of the writer for this path and no more: it emits calls, statements,
    declarations, assignments, constants and extracts, and nothing else. A
    recorder that accepted more would let the emitter grow a dependency without
    the test noticing.
    """

    class _Scope:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def __init__(self):
        self.calls = []
        self._n = 0

    def _value(self, tag):
        self._n += 1
        return f"{tag}{self._n}"

    def AnonymousScope(self):
        return Recorder._Scope()

    def declare(self, type_, **kw):
        return self._value("acc")

    def const(self, value, type_=None):
        return self._value("zero")

    def assign(self, target, value):
        self.calls.append(("assign", (target, value)))

    def extract(self, vec, slot, type_=None, **kw):
        self.calls.append(("extract", (vec, slot)))
        return self._value("el")

    def call(self, callee, type_, *args, **kw):
        out = self._value("v")
        self.calls.append((callee, args + (out,)))
        return out

    def call_stmt(self, callee, *args, **kw):
        self.calls.append((callee, args))
        return None

    def kinds(self):
        return collections.Counter(name.split("<")[0] for name, _ in self.calls)


class Ctx:
    """Just enough context for the architecture and wave queries."""

    class _Descr:
        model = "gfx90a"
        vec_unit_length = 64

    class _Target:
        def __init__(self):
            self.hw = Ctx._Descr()

    def __init__(self):
        self.target = Ctx._Target()


def _emit(M=1, N=16, K=64, threads=64, dtype=Datatype.F64, sparse=None):
    writer, stores = Recorder(), []
    ok = exchange_codegen.matmul_exchange(
        writer, lambda w, value, i, j: stores.append((i, j, value)),
        lambda w, var, j, k: f"sh{j}_{k}",
        lambda w, var, i, k: f"da{i}_{k}",
        M=M, N=N, K=K, kx=0, threads=threads, dtype=dtype, sparse=sparse,
        ctx=Ctx(), start=0, stop=N)
    return ok, writer, stores


# --------------------------------------------------------------------------- #
# what it selects, and when it declines
# --------------------------------------------------------------------------- #

def test_it_selects_the_instruction_the_plans_cover():
    op = exchange_codegen.exchange_op(Datatype.F64, 64, Ctx())
    assert op.builtin == "mfma_f64_16x16x4f64"
    assert reorder.a_exchange(op) is not None
    assert not reorder.broadcast_feeds_a(op)


@pytest.mark.parametrize("kwargs", [
    {"sparse": lambda k, j: True},
    {"threads": 32},
])
def test_it_declines_without_emitting(kwargs):
    """`False` and an empty writer, not `False` after a partial emission.

    The caller runs this inside `Writer.speculative`, so emitting and then
    declining is survivable -- but only what went through the writer is rolled
    back, and a decision that can be made up front costs nothing to make up
    front. A sparse second operand is read by linear index, which no fragment
    accepts; 32 threads because a matrix instruction is a whole-wave
    operation.
    """
    ok, writer, stores = _emit(**kwargs)
    assert ok is False
    assert writer.calls == [] and stores == []


def test_f32_reaches_this_path_too_and_that_is_a_routing_question():
    """`mfma_f32_16x16x4f32` is the same shape as the F64 one.

    One block, `k == 4`, one element per lane -- so it has an exchange and a
    plan, and this emitter takes it. Which is a reason routing has to be
    decided rather than derived: `matmul32` already owns F32 through the K=1
    tiles, and two paths that both answer for a type have to be ordered by
    something. Recorded here so that whoever routes finds the collision stated
    rather than discovering it as a duplicate span.
    """
    assert exchange_codegen.exchange_op(
        Datatype.F32, 64, Ctx()).builtin == "mfma_f32_16x16x4f32"
    ok, writer, stores = _emit(dtype=Datatype.F32)
    assert ok and stores


# --------------------------------------------------------------------------- #
# faithfulness to the plans
# --------------------------------------------------------------------------- #

def test_the_call_counts_are_what_the_plans_say():
    """Every count derived, not written down.

    One transpose per k-block, one issue per (contraction group, lead group,
    lead slot), and the swaps and merges of every fragment and every
    writeback. A count that came out of running the emitter and was then
    pasted here would pass forever; these come from the plans, so an emitter
    that stops following them fails.
    """
    op = exchange_codegen.exchange_op(Datatype.F64, 64, Ctx())
    exchange = reorder.a_exchange(op)
    lead_groups = op.wave // (op.n * op.blocks)
    M, N, K = 1, 16, 64

    ok, writer, stores = _emit(M=M, N=N, K=K)
    assert ok
    kinds = writer.kinds()

    blocks = K // (exchange.stride * op.k)
    issues = blocks * exchange.groups * M * lead_groups
    assert kinds[op.callee] == issues
    assert kinds["tensorforge::transpose16x16b32"] == blocks

    fragment = sum(len(m.swaps) for m in reorder.fragment_moves(op, "B", 0))
    merges = len(reorder.fragment_moves(op, "B", 0))
    writeback = sum(sum(len(g.swaps)
                        for g in reorder.accumulator_gathers(op, column))
                    for column in range(min(op.m, N)))
    gathers = sum(len(reorder.accumulator_gathers(op, column))
                  for column in range(min(op.m, N)))

    assert kinds["tensorforge::swap"] == issues * fragment + writeback * M
    assert kinds["tensorforge::dppUpdate"] == issues * merges + gathers * M
    assert kinds["extract"] == gathers * M
    assert len(stores) == N * M


def test_every_issue_reads_a_transpose_output_and_its_own_accumulator():
    """Where the A operand comes from, which the emitter must not get wrong.

    It is a register of the transpose -- register `g` for contraction group
    `g` -- and never a freshly built value. Feeding it from the wrong place
    produces correctly typed code that computes something else, which is
    exactly what no snapshot would catch. The accumulator is checked
    separately, because after the first issue it is a produced value rather
    than the declaration and the two properties want different assertions.
    """
    op = exchange_codegen.exchange_op(Datatype.F64, 64, Ctx())
    ok, writer, _ = _emit()
    assert ok

    transposed = None
    issues = 0
    for name, args in writer.calls:
        if name.startswith("tensorforge::transpose"):
            transposed = list(args)
        elif name == op.callee:
            a_operand = args[0]
            assert transposed is not None, "an issue before the transpose"
            assert a_operand in transposed, "A operand is not a fragment"
            issues += 1
    assert issues == writer.kinds()[op.callee]


def test_the_accumulators_are_threaded_not_reused():
    """One chain per lead group, each step reading the previous result.

    The instruction returns its accumulator, so the chain is SSA and every
    issue after the first has to read a value some earlier issue produced. An
    emitter that passed the original declaration every time would drop every
    contribution but the last, and the shapes and types would all still match.
    """
    op = exchange_codegen.exchange_op(Datatype.F64, 64, Ctx())
    ok, writer, _ = _emit()
    assert ok

    produced, read_back = set(), 0
    for name, args in writer.calls:
        if name != op.callee:
            continue
        accumulator, out = args[2], args[-1]
        if accumulator in produced:
            read_back += 1
        produced.add(out)

    lead_groups = op.wave // (op.n * op.blocks)
    issues = writer.kinds()[op.callee]
    assert read_back == issues - lead_groups, (
        "every issue but the first of each chain reads a produced value")


def test_the_transpose_writes_registers_of_its_own():
    """Never a constant, which a reference parameter cannot bind.

    The transpose writes its operands in place, and a column is a constant in
    the padding and wherever the operator's numbers are spelled as literals --
    SeisSol's elastic kernels at order 6 in double stopped at the first k-block
    on exactly that.  Every operand is a declaration a column was assigned to.
    """
    ok, writer, _ = _emit(N=9)
    assert ok
    assigned = {args[0] for name, args in writer.calls if name == "assign"}
    transposes = [args for name, args in writer.calls
                  if name.startswith("tensorforge::transpose")]
    assert transposes
    for args in transposes:
        assert all(a in assigned for a in args)


def test_the_dispatch_hands_the_shared_operand_to_the_transpose(monkeypatch):
    """`matmul_exchange` takes `shared` before `lead`; the dispatch holds them
    as `B` and `A`.

    Handed over in the dispatch's own order, the transpose read the lead
    operand with a column for its slot and the fragments the shared one with a
    lead slot for its column -- typed, emitted, compiled for gfx942, and wrong.
    Nothing ran the routing before SeisSol's order-6 kernels did, and the
    tests above call the emitter with its own names.
    """
    from tensorforge.backend.pir.build import IRBuilder
    from tensorforge.backend.instructions.compute.matmul import MatmulOperands
    from tensorforge.backend.instructions.compute.primitives import amd
    from tensorforge.backend.instructions.compute.primitives.amd import tiling
    from tensorforge.backend.instructions.compute.strategy import Span, Strategy
    monkeypatch.setattr(tiling, "EXCHANGE", True)
    reads = {"A": [], "B": []}

    def reader(name):
        def read(writer, var, first, k):
            reads[name].append(first)
            return writer.declare(hint=name.lower())
        return read

    stores = []
    M, N = 1, 16
    ops = MatmulOperands(A=reader("A"), B=reader("B"),
                         C=lambda writer, value, i, j: stores.append((i, j)),
                         sparse=None, lead_slots=M, lead_elements=M * 64,
                         n=N, k=64, kx=0, threads=64, a=Datatype.F64,
                         b=Datatype.F64, accumulator=Datatype.F64)
    assert amd.matmul(IRBuilder(Datatype.F64), ops, Ctx(),
                      Span(Strategy.MATRIX, 0, N))
    assert set(reads["A"]) == set(range(M)), "A is read by lead slot"
    assert set(reads["B"]) == set(range(N)), "B is read by column"
    assert sorted(stores) == [(i, j) for i in range(M) for j in range(N)]


def test_the_lead_counts_from_the_contraction_head():
    """Position `p` of the shared operand's block is the lead's step `p - kx`.

    The shared operand is read a block at a time from the block holding the
    contraction's first step, and the lead accessor counts from that step --
    the pairing `matmul32` states.  Counted from the block's start instead,
    the lead read `kx` steps ahead of the shared operand and past its own end,
    which wraps (SeisSol's time derivative, from depth 1).
    """
    steps = []

    def lead(writer, var, i, k):
        steps.append(k)
        return f"da{i}_{k}"

    writer, stores = Recorder(), []
    ok = exchange_codegen.matmul_exchange(
        writer, lambda w, value, i, j: stores.append((i, j)),
        lambda w, var, j, k: f"sh{j}_{k}", lead,
        M=1, N=16, K=60, kx=4, threads=64, dtype=Datatype.F64, sparse=None,
        ctx=Ctx(), start=0, stop=16)
    assert ok
    assert set(steps) == set(range(60))


@pytest.mark.parametrize("dtype", [Datatype.F64, Datatype.F32])
@pytest.mark.parametrize("head", [0, 1, 5])
def test_the_exchange_order_holds_what_the_plan_would_swap_there(dtype, head):
    """Stored where the swaps would have put it.

    The plan, run in the wave simulator, says which lane's row and which step
    every lane of every fragment ends up holding.  The order has to have put
    exactly that element there, at the step the head shifts it to, and
    padding wherever the step or the row is outside the window.
    """
    from harness import wavesim
    op = exchange_codegen.exchange_op(dtype, 64, Ctx())
    exchange = reorder.a_exchange(op)
    span = op.n * op.blocks
    depth = exchange.stride * op.k
    M, steps, (r0, c0) = 2, 60, (2, 5)
    rows = M * 64 - 9
    shape = (r0 + rows + 3, c0 + steps + 7)
    order = exchange_codegen.exchange_order(
        op, shape, 64, ((r0, rows), (c0, steps)), head)
    seen = 0
    for base in range(0, steps + head, depth):
        for group in range(exchange.groups):
            for i in range(M):
                for g in range(op.wave // span):
                    got = [None] * op.wave
                    for move in reorder.fragment_moves(op, "B", 0, g):
                        lanes = [(move.contraction, lane)
                                 for lane in range(op.wave)]
                        for block in move.swaps:
                            lanes = wavesim.swap(lanes, block)
                        for lane in move.select.lanes:
                            got[lane] = lanes[lane]
                    index = exchange_codegen.fragment_index(
                        op, M, base // depth, group, i, g)
                    for lane in range(op.wave):
                        kk, source = got[lane]
                        r = base + exchange.stride * kk + group - head
                        row = i * 64 + source
                        want = ((r0 + row) + shape[0] * (c0 + r)
                                if 0 <= r < steps and row < rows else -1)
                        assert order[index * 64 + lane] == want, (
                            f"{op.builtin} fragment {index} lane {lane}")
                    seen += 1
    assert len(order) == seen * 64


def test_a_prepared_lead_operand_is_read_a_fragment_at_a_time():
    """Stored in `exchange_order`, the leading operand takes no swap and no
    merge: one read per fragment, every issue reading the one it names, and
    the plan's moves are the writeback's alone."""
    op = exchange_codegen.exchange_op(Datatype.F64, 64, Ctx())
    M, N, K = 1, 16, 64
    reads = []

    def lead_slot(writer, index):
        reads.append(index)
        return f"fr{index}"

    def lead(*args):
        pytest.fail("the coordinate accessor of a prepared operand")

    writer, stores = Recorder(), []
    ok = exchange_codegen.matmul_exchange(
        writer, lambda w, value, i, j: stores.append((i, j)),
        lambda w, var, j, k: f"sh{j}_{k}", lead,
        M=M, N=N, K=K, kx=0, threads=64, dtype=Datatype.F64, sparse=None,
        ctx=Ctx(), start=0, stop=N, lead_slot=lead_slot)
    assert ok
    assert reads == list(range(len(reads)))
    kinds = writer.kinds()
    writeback = sum(sum(len(g.swaps)
                        for g in reorder.accumulator_gathers(op, column))
                    for column in range(min(op.m, N)))
    gathers = sum(len(reorder.accumulator_gathers(op, column))
                  for column in range(min(op.m, N)))
    assert kinds["tensorforge::swap"] == writeback * M
    assert kinds["tensorforge::dppUpdate"] == gathers * M
    issues = [args for name, args in writer.calls if name == op.callee]
    assert issues and all(args[1].startswith("fr") for args in issues)


def test_the_dispatch_reads_a_prepared_operand_by_fragment(monkeypatch):
    """With `A_slot` -- `A` stored as `prepared_order` offered it -- the
    exchange scheme reads it whole fragments at a time and never by
    coordinate, which would address a permuted buffer."""
    from tensorforge.backend.pir.build import IRBuilder
    from tensorforge.backend.instructions.compute.matmul import MatmulOperands
    from tensorforge.backend.instructions.compute.primitives import amd
    from tensorforge.backend.instructions.compute.primitives.amd import tiling
    from tensorforge.backend.instructions.compute.strategy import Span, Strategy
    monkeypatch.setattr(tiling, "EXCHANGE", True)
    slots = []

    def coordinate(writer, var, i, k):
        pytest.fail("A read by coordinate")

    def a_slot(writer, slot, parts=1, shift=None, width=1):
        slots.append(slot)
        return writer.declare(hint="frag")

    M, N = 1, 16
    ops = MatmulOperands(A=coordinate,
                         B=lambda writer, var, j, k: writer.declare(hint="b"),
                         C=lambda writer, value, i, j: None,
                         sparse=None, lead_slots=M, lead_elements=M * 64,
                         n=N, k=64, kx=0, threads=64, a=Datatype.F64,
                         b=Datatype.F64, accumulator=Datatype.F64,
                         A_slot=a_slot)
    assert amd.matmul(IRBuilder(Datatype.F64), ops, Ctx(),
                      Span(Strategy.MATRIX, 0, N))
    assert slots and all(s % 64 == 0 for s in slots)
    assert sorted(slots) == [64 * n for n in range(len(slots))]
    order = amd.prepared_order((64, 64), Datatype.F64, Ctx(), columns=N,
                               lead=64, depth=64, threads=64)
    assert order == exchange_codegen.exchange_order(
        exchange_codegen.exchange_op(Datatype.F64, 64, Ctx()), (64, 64), 64,
        ((0, 64), (0, 64)))


def test_a_partial_column_block_is_padded_with_zeroes():
    """Fewer columns than the tile is width, not a reason to decline.

    The instruction covers sixteen either way, so the columns that do not
    exist are fed real zeroes and contribute nothing -- the same answer
    `matmul32` gives for a partial block, and cheaper than falling back for
    the remainder.
    """
    ok, writer, stores = _emit(N=9)
    assert ok
    assert {j for _, j, _ in stores} == set(range(9))
