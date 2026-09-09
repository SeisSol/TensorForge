# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""`barrier(scope)` says which threads have to meet; the lowering has to agree.

`Uniformity` is a lattice — LANE, MULT, BLOCK, GRID — and `IRBuilder.barrier`
takes a level from it. The docstring there is explicit about why: a barrier at
level S inside a construct whose entry is only U-uniform deadlocks unless
`U >= S`. So the level is a claim about how many threads arrive.

MULT lowers to `sync_simd()` — `__syncwarp()` on CUDA — and that is not a
lowering tied to an invariant but a definition: a barrier spelled MULT *means*
the wave, and `nvidia.py` says `'wave'` at the six places where it stages an
mma fragment, because a fragment's distribution is a property of the warp and
not of whatever the multiplication happens to be.

A multiplication wider than a wave is therefore not spelled MULT at all. It
cannot be: a wave sits above a 16-thread multiplication and below a 64-thread
one, so a rung for it on this lattice would order `min`-propagation of value
uniformity differently depending on the lane configuration. What happens
instead is that `SyncThreads` resolves the width where the numbers are —
`barrier_scope` already weighs the thread count against `vec_unit_length` —
and asks for a BLOCK barrier carrying that width, so a vendor with a sub-block
rendezvous can narrow it in `Lexic.sync_mult`.

That resolution is the fix these tests now pin. The two used to disagree:
`barrier_scope` answered GROUP for a wide multiplication while `gen_ir` asked
for MULT regardless, so `verify` would refuse the construct while the emitter,
had it run, would have synchronised a warp where a block was needed.

Which leaves the question of when a block barrier inside the batch loop is
legal at all. It is legal exactly at one multiplication per block: the loop is
otherwise only MULT-uniform and the barrier deadlocks on a ragged tail. So a
width above a wave and a block holding several multiplications are mutually
exclusive unless the target can rendezvous a sub-block, and
`AbstractThreadBlockPolicy._barrier_cap` is where the two meet — it asks
`Lexic.has_sync_mult` and caps the block at one where the answer is no. The
last test below reads that from the outside.
"""
from __future__ import annotations

import pytest

from tensorforge.backend.pir.build import IRBuilder
from tensorforge.backend.pir.core import Uniformity
from tensorforge.common.basic_types import Datatype
from tensorforge.common.context import Context


def _sync_text(context, scope, threads=None):
    from tensorforge.backend.pir.emit import Emitter

    lines = []

    class _Sink:
        def __call__(self, text, *args, **kwargs):
            lines.append(text)

        def __getattr__(self, _):
            return lambda *a, **k: None

    return Emitter(_Sink(), context)._sync(scope, threads)


def _emitted_scope(context, num_threads):
    """What `SyncThreads` actually asks the IR for, at this width.

    Through the instruction rather than through `_sync`, because the width is
    resolved there: that is the whole point of the change these tests pin.
    """
    from tensorforge.backend.instructions.sync_block import SyncThreads

    seen = []

    class _Recorder:
        def barrier(self, scope, threads=None):
            seen.append((scope, threads))

    SyncThreads(context, num_threads).gen_ir(_Recorder())
    assert len(seen) == 1, f'expected one barrier, got {seen}'
    return seen[0]


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


def test_a_multiplication_wider_than_a_wave_does_not_ask_for_a_warp_barrier(context):
    """This used to be an xfail waiting on `_sync` learning the width.

    It no longer waits, and the reason is that the width is not resolved in
    the emitter at all.  `SyncThreads.barrier_scope` already weighed the
    thread count against the wave -- that is why it takes one -- and `gen_ir`
    now asks for what it decided instead of a second opinion.  A multiplication
    that outgrows its wave therefore reaches the IR as a block barrier that
    carries its own width, and never as `MULT`.

    Checkable without lifting the lane cap, because the instruction can be
    built at any width; what the cap governs is whether a *kernel* reaches
    this, not whether the lowering is right.
    """
    scope, threads = _emitted_scope(context, num_threads=64)
    assert scope.name == 'BLOCK', (
        f'a 64-thread multiplication on a 32-wide wave asked for {scope.name}; '
        f'a warp barrier reaches half the threads it was told to reach')
    assert threads == 64, 'the width has to ride along or no vendor can narrow it'
    assert 'syncwarp' not in _sync_text(context, scope, threads)


def test_a_multiplication_inside_its_wave_still_asks_for_the_warp(context):
    """The other half: the common case must not have moved."""
    scope, threads = _emitted_scope(context, num_threads=32)
    assert scope.name == 'MULT'
    assert threads is None, (
        'a wave barrier carries no width -- the threads are in lockstep, so '
        'there is nothing for a vendor to narrow')
    assert 'syncwarp' in _sync_text(context, scope, threads)


def test_the_width_reaches_the_vendor_hook(context):
    """`sync_mult` is where a sub-block rendezvous would go, so it must be asked."""
    asked = []

    class _Lexic:
        def sync_mult(self, n):
            asked.append(n)
            return f'named_barrier({n});'

        def sync_block(self):
            return '__syncthreads();'

        def sync_simd(self):
            return '__syncwarp();'

        def sync_grid(self):
            return 'grid.sync();'

    from tensorforge.backend.pir.emit import Emitter

    emitter = Emitter(lambda *a, **k: None, context)
    emitter._lexic = lambda: _Lexic()
    assert emitter._sync(Uniformity.BLOCK, 64) == 'named_barrier(64);'
    assert asked == [64]
    # ...and a block barrier with no width stays a block barrier
    assert emitter._sync(Uniformity.BLOCK, None) == '__syncthreads();'


def test_the_default_hook_over_synchronises_rather_than_deadlocking(context):
    """No vendor implements the narrow form; the fallback has to be a superset."""
    lex = context.get_vm().get_lexic()
    assert lex.sync_mult(64) == lex.sync_block()


def test_a_thread_count_above_a_wave_gets_a_block_of_its_own():
    """The invariant, checked from the outside.

    `_deduce_num_threads` clamps to 32 only when no elementwise descriptor is
    present, so a multilinear aligning above 32 next to an elementwise gets a
    64-thread multiplication. None of the three backends implements
    `sync_mult`, so all three have to answer it the same way: one
    multiplication per block, which is what makes the block barrier the
    multiplication's own and the loop around it block-uniform.

    Checked through the launch geometry rather than the policy, because the
    number that matters is the one the kernel is launched with.
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
        wave = ctx.get_vm().get_hw_descr().vec_unit_length
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
        assert mults == 1, (
            f"{arch}: {threads} threads per multiplication and {mults} "
            f"multiplications per block, with no sub-block rendezvous to "
            f"separate them")


def test_a_width_within_a_wave_still_packs_a_block():
    """The cap is about the barrier, so it must not reach a narrow width.

    A multiplication inside a wave needs no block-level rendezvous at all, and
    capping it at one per block would spend occupancy on a barrier nobody asks
    for.
    """
    from tensorforge.generators.generator import RegmaxBlockPolicy

    ctx = Context(arch="sm_86", backend="cuda", fp_type=Datatype.F32)
    wave = ctx.get_vm().get_hw_descr().vec_unit_length
    narrow = RegmaxBlockPolicy(ctx, global_mem=0, mem_size_per_mult=0,
                               num_threads=wave)
    wide = RegmaxBlockPolicy(ctx, global_mem=0, mem_size_per_mult=0,
                             num_threads=2 * wave)
    assert narrow._barrier_cap() is None
    assert narrow.get_num_mults_per_block() > 1
    assert wide._barrier_cap() == 1
    assert wide.get_num_mults_per_block() == 1


def test_the_loop_reports_block_uniformity_only_at_one_mult_per_block():
    """`uniform_scope` is what `verify` weighs a barrier against."""
    from tensorforge.backend.instructions.abstract_instruction import \
        BarrierScope
    from tensorforge.backend.instructions.batch_loop import BatchLoop, LoopMode

    ctx = Context(arch="sm_86", backend="cuda", fp_type=Datatype.F32)
    loop = BatchLoop(ctx, section_index=0, mode=LoopMode.PERSISTENT,
                     start="0", stride="1", region=[])
    assert loop.uniform_scope() is BarrierScope.SIMD, (
        "an undecided block count has to answer conservatively")
    loop.set_mults_per_block(2)
    assert loop.uniform_scope() is BarrierScope.SIMD
    loop.set_mults_per_block(1)
    assert loop.uniform_scope() is BarrierScope.GROUP


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
