# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""`barrier(scope)` says which threads have to meet; the lowering has to agree.

`Uniformity` is a lattice — LANE, MULT, BLOCK, GRID — and `IRBuilder.barrier`
takes a level from it. The docstring there is explicit about why: a barrier at
level S inside a construct whose entry is only U-uniform deadlocks unless
`U >= S`. So the level is a claim about how many threads arrive.

`emit._sync` lowers MULT to `sync_simd()` — `__syncwarp()` on CUDA — and that
is correct, because a multiplication currently cannot outgrow a wave. It is a
lowering tied to an invariant, not a defect, and these tests exist to say
which invariant, so that the day it is lifted the lowering is not left behind.

The invariant is not stated anywhere; it holds by construction.
`_deduce_num_threads` clamps to 32 unless an elementwise descriptor is
present, and the one configuration that gets past the clamp — a multilinear
whose lead dimension aligns above 32, alongside an elementwise — does not
produce a wrong kernel either. It produces no kernel at all: the PIR verifier
rejects it with "group barrier inside a construct whose trip count is only
simd-uniform". Checked, not assumed; see the last test below.

So the super-wave reduction is blocked, but not by a hazard. When the cap
lifts, MULT has to lower to `sync_block()` wherever a multiplication spans
several waves — correct on its own terms, a block barrier being a superset —
and safe only with one multiple per block, since the batch loop is otherwise
just MULT-uniform and a block barrier inside it deadlocks on a ragged tail.
That is a thread-block policy decision
(`RegmaxBlockPolicy.get_num_mults_per_block`), not one the emitter can take.
"""

from __future__ import annotations

import pytest

from tensorforge.backend.pir.build import IRBuilder
from tensorforge.backend.pir.core import Uniformity
from tensorforge.common.basic_types import Datatype
from tensorforge.common.context import Context


def _sync_text(context, scope):
    from tensorforge.backend.pir.emit import Emitter

    lines = []

    class _Sink:
        def __call__(self, text, *args, **kwargs):
            lines.append(text)

        def __getattr__(self, _):
            return lambda *a, **k: None

    return Emitter(_Sink(), context)._sync(scope)


@pytest.fixture
def context():
    return Context(arch="sm_86", backend="cuda", fp_type=Datatype.F32)


def test_barrier_scope_reaches_the_emitter(context):
    """The regression that motivated the lattice: every scope came out block."""
    assert _sync_text(context, Uniformity.GRID) != _sync_text(context,
                                                              Uniformity.BLOCK)
    assert _sync_text(context, Uniformity.MULT) != _sync_text(context,
                                                              Uniformity.BLOCK)


def test_a_mult_barrier_is_a_warp_barrier(context):
    """What it lowers to today, stated so the change is visible when it moves."""
    assert "syncwarp" in _sync_text(context, Uniformity.MULT)


@pytest.mark.xfail(strict=True, reason=(
    "the thread count cannot exceed a wave today, so _sync has no reason to "
    "know the configuration; lifting the cap is what makes this assertion due"))
def test_a_mult_barrier_spanning_waves_is_not_a_warp_barrier(context):
    """What has to become true at the same commit that lifts the cap.

    Written against the lowering rather than a generated kernel, because
    nothing emits a MULT barrier at that width: the assertion has to be able
    to fail before the configuration that would need it exists. `_sync` takes
    only the scope, so there is no way to hand it a thread count -- which is
    the work, not an obstacle to describing it.
    """
    from tensorforge.common.threadconfig import ThreadConfig

    config = ThreadConfig(context, threadcount=64)
    assert config.superwarp(), "the fixture no longer sets up the case"
    assert config.warps_per_multiple() == 2

    assert "syncwarp" not in _sync_text(context, Uniformity.MULT)


def test_a_thread_count_above_a_wave_is_refused_rather_than_miscompiled():
    """The invariant, checked from the outside.

    `_deduce_num_threads` clamps to 32 only when no elementwise descriptor is
    present, so the clamp alone does not carry it. What does is the verifier:
    the one combination that gets past the clamp -- a multilinear aligning
    above 32 next to an elementwise -- is rejected outright.

    If this ever starts generating, the barrier lowering above is no longer
    covered by an invariant and the xfail turns due.
    """
    from tensorforge.common.basic_types import Addressing
    from tensorforge.common.exceptions import GenerationError
    from tensorforge.common.matrix.boundingbox import BoundingBox
    from tensorforge.common.matrix.tensor import SubTensor, Tensor
    from tensorforge.common.operation import Operation
    from tensorforge.generators.descriptions import (ElementwiseDescr,
                                                     MultilinearDescr)
    from tensorforge.generators.generator import Generator

    def tensor(shape, alias):
        return SubTensor(Tensor(shape, Addressing.STRIDED,
                                BoundingBox([0] * len(shape), list(shape)),
                                alias=alias, datatype=Datatype.F32))

    ctx = Context(arch="sm_86", backend="cuda", fp_type=Datatype.F32)
    assert ctx.align(num=56) > 32, "56 no longer aligns above a wave"

    gemm = MultilinearDescr(tensor([56, 18], "C"),
                            [tensor([56, 18], "A"), tensor([18, 18], "B")],
                            [[0, -1], [-1, 1]], [[0, 1], [0, 1]])
    ew = ElementwiseDescr(Operation.ABS, tensor([56, 18], "F"),
                          [tensor([56, 18], "E")])

    with pytest.raises(GenerationError, match="barrier"):
        Generator([gemm, ew], ctx).generate()


def test_the_reduction_refuses_a_cross_lane_fold_it_cannot_synchronise():
    """Better a named refusal than a generic barrier diagnostic.

    `ReductionInstruction` reads the same fact from the other side. The
    verifier would catch the configuration anyway; refusing here says which
    feature is missing rather than which invariant was violated.
    """
    from tensorforge.backend.instructions.compute.reduction import \
        ReductionInstruction

    source = ReductionInstruction._check_cross_lane_is_available.__doc__
    assert source and "sync_simd" in source, (
        "the refusal has stopped naming the reason it refuses")
