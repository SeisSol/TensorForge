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
    catalog, exchange_codegen, reorder)


class Recorder:
    """Records the call sequence instead of building IR.

    Enough of the writer for this path and no more: it emits calls, statements,
    declarations, constants and extracts, and nothing else. A recorder that
    accepted more would let the emitter grow a dependency without the test
    noticing.
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

    class _VM:
        def get_hw_descr(self):
            return Ctx._Descr()

    def get_vm(self):
        return Ctx._VM()


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
