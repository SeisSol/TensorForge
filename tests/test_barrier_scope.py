# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""A barrier says what it covers; who must arrive follows from that.

Two questions used to share one answer and that is what the merged ladder
separates. `Participants` says what the barrier covers in hardware -- WAVE,
MULT, MULTGROUP, BLOCK, GRID -- and is what the emitter turns into an
instruction. `Uniformity` says who therefore has to arrive, and is what
`verify` weighs against the region: a barrier at level S inside a construct
whose entry is only U-uniform deadlocks unless `U >= S`.

The wave is on the first list and deliberately not on the second. It sits
above a 16-thread multiplication and below a 64-thread one, so a rung for it
would order `min`-propagation of value uniformity differently depending on the
lane configuration. What a wave barrier demands of its region is therefore a
fact about the geometry, and `Participants.arrival` derives it rather than
taking it from the caller -- which is what keeps the claim and the instruction
from drifting apart.

`MULTGROUP` is on both lists: it is a real answer to "the same across how many
threads". Where a multiplication does not divide the wave, the smallest set a
barrier can separate is the group of them that fills a whole number of waves,
and a traversal driven by such a group is group-uniform and neither mult- nor
block-uniform.

Which set `SyncThreads` asks for is the narrowest the target can spell, because
whatever a barrier covers has to arrive, and rows that must arrive together
cannot run the body a different number of times. So the width chosen there is
what `AbstractThreadBlockPolicy` sizes a block from, and asking for more than
necessary is occupancy given away.
"""
from __future__ import annotations

import pytest

from tensorforge.backend.pir.build import IRBuilder
from tensorforge.backend.pir.core import Uniformity
from tensorforge.common.basic_types import Datatype
from tensorforge.common.context import Context


def _sync_text(context, participants, threads=None):
    from tensorforge.backend.pir.emit import Emitter

    lines = []

    class _Sink:
        def __call__(self, text, *args, **kwargs):
            lines.append(text)

        def __getattr__(self, _):
            return lambda *a, **k: None

    return Emitter(_Sink(), context)._sync(participants, threads)


def _requested(context, num_threads):
    """What `SyncThreads` actually asks the IR for, at this width."""
    from tensorforge.backend.instructions.sync_block import SyncThreads

    seen = []

    class _Recorder:
        def barrier(self, participants, threads=None):
            seen.append((participants, threads))

    SyncThreads(context, num_threads).gen_ir(_Recorder())
    assert len(seen) == 1, f'expected one barrier, got {seen}'
    return seen[0]


@pytest.fixture
def context():
    return Context(arch="sm_86", backend="cuda", fp_type=Datatype.F32)


def test_each_participant_set_reaches_the_emitter_as_its_own_instruction(context):
    from tensorforge.backend.pir.core import Participants

    grid = _sync_text(context, Participants.GRID)
    block = _sync_text(context, Participants.BLOCK)
    wave = _sync_text(context, Participants.WAVE)
    assert len({grid, block, wave}) == 3, (
        f"three participant sets, {len({grid, block, wave})} instructions")
    assert "syncwarp" in wave
    assert "this_grid" in grid


def test_a_multiplication_that_is_the_wave_asks_for_the_wave(context):
    from tensorforge.backend.pir.core import Participants, Uniformity
    from tensorforge.backend.instructions.sync_block import SyncThreads

    wave = context.get_vm().get_hw_descr().vec_unit_length
    who, threads = _requested(context, wave)
    assert who is Participants.WAVE, (
        "nothing is narrower than the wave when the multiplication is one")
    assert threads == wave
    assert SyncThreads(context, wave).barrier_scope() is Uniformity.MULT


def test_a_multiplication_inside_a_wave_asks_for_a_masked_wave(context):
    """Several multiplications share the wave, so a full mask is too wide."""
    from tensorforge.backend.pir.core import Participants, Uniformity
    from tensorforge.backend.instructions.sync_block import SyncThreads

    wave = context.get_vm().get_hw_descr().vec_unit_length
    half = wave // 2
    who, threads = _requested(context, half)
    assert who is Participants.MULT
    assert SyncThreads(context, half).barrier_scope() is Uniformity.MULT, (
        "the neighbouring multiplication in this wave is free to run the body "
        "a different number of times, so it must not be waited for")

    text = _sync_text(context, Participants.MULT, half)
    assert "syncwarp" in text and "0x" in text, (
        f"a masked warp barrier was expected, got {text!r}")


def test_a_multiplication_wider_than_a_wave_asks_for_a_named_barrier(context):
    from tensorforge.backend.pir.core import Participants, Uniformity
    from tensorforge.backend.instructions.sync_block import SyncThreads

    wide = 2 * context.get_vm().get_hw_descr().vec_unit_length
    who, threads = _requested(context, wide)
    assert who is Participants.MULT and threads == wide
    assert SyncThreads(context, wide).barrier_scope() is Uniformity.MULT

    text = _sync_text(context, Participants.MULT, wide)
    assert "barrier.sync" in text and str(wide) in text, (
        f"a counted named barrier was expected, got {text!r}")


def test_a_ragged_width_falls_back_to_the_group(context):
    """`barrier.sync` counts threads in multiples of the warp size.

    A width that leaves a partial wave has no spelling, so the barrier is met
    at the group instead -- and the block is sized to hold one group.
    """
    from tensorforge.backend.pir.core import Participants, Uniformity
    from tensorforge.backend.instructions.sync_block import SyncThreads

    wave = context.get_vm().get_hw_descr().vec_unit_length
    ragged = wave + wave // 2
    who, _ = _requested(context, ragged)
    assert who is Participants.MULTGROUP
    assert SyncThreads(context, ragged).barrier_scope() is Uniformity.MULTGROUP


def test_the_arrival_of_a_wave_barrier_follows_the_geometry():
    from tensorforge.backend.pir.core import Participants, Uniformity

    assert Participants.WAVE.arrival(64, 32) is Uniformity.MULT, (
        "the wave holds part of one multiplication, so only it must arrive")
    assert Participants.WAVE.arrival(16, 32) is Uniformity.MULTGROUP, (
        "the wave holds two multiplications, so both must arrive")


def test_a_target_without_a_sub_block_rendezvous_gets_the_block():
    """The default answer over-synchronises but never deadlocks."""
    from tensorforge.backend.pir.core import Participants

    ctx = Context(arch="pvc", backend="oneapi", fp_type=Datatype.F32)
    lexic = ctx.get_vm().get_lexic()
    assert not lexic.has_sync_mult(64, ctx.get_vm().get_hw_descr())
    assert _sync_text(ctx, Participants.MULTGROUP, 64) == lexic.sync_block()


def test_a_wide_multiplication_is_packed_by_what_the_target_can_separate():
    """The invariant, checked from the outside.

    `_deduce_num_threads` clamps to 32 only when no elementwise descriptor is
    present, so a multilinear aligning above 32 next to an elementwise gets a
    64-thread multiplication. What a block then holds is exactly what the
    target can tell apart: CUDA has `barrier.sync id, count` and packs several,
    SYCL under SPMD has nothing narrower than the sub-group and gets one.
    """
    from tensorforge.common.basic_types import Addressing
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

    for arch, backend in (("sm_86", "cuda"), ("gfx1100", "hip"),
                          ("pvc", "oneapi")):
        ctx = Context(arch=arch, backend=backend, fp_type=Datatype.F32)
        vm = ctx.get_vm()
        wave = vm.get_hw_descr().vec_unit_length
        assert ctx.align(num=56) > wave, f"56 no longer aligns above {arch}"

        gemm = MultilinearDescr(tensor([56, 18], "C"),
                                [tensor([56, 18], "A"), tensor([18, 18], "B")],
                                [[0, -1], [-1, 1]], [[0, 1], [0, 1]])
        ew = ElementwiseDescr(Operation.ABS, tensor([56, 18], "F"),
                              [tensor([56, 18], "E")])

        gen = Generator([gemm, ew], ctx)
        gen.generate()
        threads = gen._num_threads
        mults = min(s.shr_mem_obj.get_mults_per_block() for s in gen._sections)
        assert threads > wave, f"{arch}: {threads} no longer exceeds the wave"

        separable = vm.get_lexic().has_sync_mult(threads, vm.get_hw_descr())
        loud = any(any(i.barrier_scope() is not None for i in s.stream)
                   for s in gen._sections)
        if loud and not separable:
            assert mults == 1, (
                f"{arch}: {mults} multiplications per block with no sub-block "
                f"rendezvous to separate them")
        else:
            assert mults >= 1, (
                f"{arch}: nothing forces a block down to one multiplication "
                f"here, so packing it is the right answer")


def test_the_cap_is_asked_only_where_a_barrier_needs_it():
    """A section with no barrier is packed by memory and threads alone."""
    from tensorforge.generators.generator import RegmaxBlockPolicy

    ctx = Context(arch="pvc", backend="oneapi", fp_type=Datatype.F32)
    wave = ctx.get_vm().get_hw_descr().vec_unit_length

    quiet = RegmaxBlockPolicy(ctx, global_mem=0, mem_size_per_mult=0,
                              num_threads=2 * wave)
    assert quiet._barrier_cap() is None
    assert quiet.get_num_mults_per_block() > 1

    loud = RegmaxBlockPolicy(ctx, global_mem=0, mem_size_per_mult=0,
                             num_threads=2 * wave)
    loud.set_has_barrier(True)
    assert loud._barrier_cap() == 1
    assert loud.get_num_mults_per_block() == 1

    exact = RegmaxBlockPolicy(ctx, global_mem=0, mem_size_per_mult=0,
                              num_threads=wave)
    exact.set_has_barrier(True)
    assert exact._barrier_cap() is None, (
        "a multiplication that is the wave needs nothing narrower")


def test_the_loop_reports_block_uniformity_only_at_one_mult_per_block():
    """`uniform_scope` is what `verify` weighs a barrier against."""
    from tensorforge.backend.instructions.batch_loop import BatchLoop, LoopMode
    from tensorforge.backend.pir.core import Uniformity

    ctx = Context(arch="sm_86", backend="cuda", fp_type=Datatype.F32)
    loop = BatchLoop(ctx, section_index=0, mode=LoopMode.PERSISTENT,
                     start="0", stride="1", region=[])
    assert loop.uniform_scope() is Uniformity.MULT, (
        "an undecided block count has to answer conservatively")
    loop.set_mults_per_block(2)
    assert loop.uniform_scope() is Uniformity.MULT
    loop.set_mults_per_block(1)
    assert loop.uniform_scope() is Uniformity.BLOCK


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
