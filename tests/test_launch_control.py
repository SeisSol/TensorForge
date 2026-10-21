# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Blackwell's work queue, and four ways to get it wrong.

`clusterlaunchcontrol.try_cancel` asks the launcher not to launch a CTA that
has not started yet and hands the caller its id, so a resident block drains
the grid without an occupancy query.  The mode rests on four things, each of
which this module pins:

*The phase is not a copy.*  The mbarrier parity lives in
`ClusterLaunchCursor`, where a caller cannot hold it wrong.  A query that took
the parity by value and flipped its own local would leave the caller's parity
unchanged, so from the second iteration on the wait would be against a phase
that had already completed, return at once, and the block would read a
response another block is also acting on.  One character, and the whole mode.

*The response type is a device type.*  `std::optional` is not `__device__`; it
compiles only under `--expt-relaxed-constexpr`, which this repository happens
to pass and a consumer need not.

*The bookkeeping is per-thread.*  A cursor in shared memory would be one
parity that every thread of the block flips, which is a race, not a counter.
A per-thread cursor stays in lockstep because each value it derives is read
from a response behind a barrier.

*The loop-carried barrier is outside the element guard.*  The hand-off carries
its own barrier, outside the guard, so nothing needs to be appended at all.  A
`SyncThreads` appended per iteration, as the grid-stride loop has it, would
land inside `if (batchId0 < numElements0)` -- a predicate the rows of a block
decide differently.  That does not reliably deadlock, which is worse than if
it did: `bar.sync` pairs arrivals by barrier *number*, so the rows that skip
it rendezvous at the next barrier instead and the loop runs on one barrier out
of step, reusing the tile an iteration early.  With a flag mask making the
rows disagree persistently it does hang.

The numbers that decide the default are in the option's own declaration.

The queue loop itself is built without `enable_wrap_loads`, which at depth
one is refused; what the wrap makes of it at depth two is the last part.
"""

from __future__ import annotations

import re

import pytest

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.context import Context
from tensorforge.common.exceptions import GenerationError
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.common.options import Options
from tensorforge.generators.descriptions import GemmDescr, GridBarrierDescr
from tensorforge.generators.generator import Generator

#: Blackwell and above.  `sm_90` is the interesting negative: it has thread
#: block clusters and not the cancel instruction.
BLACKWELL = "sm_120"
HOPPER = "sm_90"


def _descr_list():
    dt = Datatype.F32
    a = SubTensor(Tensor([56, 56], Addressing.STRIDED,
                         BoundingBox([0, 0], [56, 56]), alias="A", datatype=dt))
    b = SubTensor(Tensor([56, 9], Addressing.STRIDED,
                         BoundingBox([0, 0], [56, 9]), alias="B", datatype=dt))
    c = SubTensor(Tensor([56, 9], Addressing.STRIDED,
                         BoundingBox([0, 0], [56, 9]), alias="C", datatype=dt))
    return [GemmDescr(False, False, a=a, b=b, c=c, alpha=1.0, beta=0.0)]


def _kernel(arch=BLACKWELL, **options):
    ctx = Context(arch=arch, backend="cuda", fp_type=Datatype.F32,
                  options=Options(**options))
    gen = Generator(_descr_list(), ctx)
    gen.generate()
    return gen


def _depth_of(text, needle):
    """Brace nesting at the first line containing ``needle``.

    Cheaper than parsing and enough for the one question asked here: is the
    hand-off inside the element guard or beside it.
    """
    depth = 0
    for line in text.splitlines():
        if needle in line:
            return depth
        depth += line.count("{") - line.count("}")
    raise AssertionError(f"{needle!r} not in the generated kernel")


# -- the mode is reachable at all ------------------------------------------ #

def test_the_queue_is_emitted_when_asked_for():
    src = _kernel(launch_control=True, enable_wrap_loads=False).get_kernel()
    assert "ClusterLaunchQueue<1>" in src
    assert "ClusterLaunchCursor<1>" in src
    assert "while (true)" in src


def test_the_grid_stride_loop_is_what_the_default_still_gets():
    src = _kernel().get_kernel()
    assert "ClusterLaunchQueue" not in src
    # `batchId0` is the induction value's hint since the body takes the
    # index as an operand, so the emitter puts the value id in front of it.
    assert re.search(r"for \(size_t \w*batchId0\b", src)


def test_the_depth_reaches_the_type():
    """Two is the first depth at which the next index is known at the top."""
    src = _kernel(launch_control=True, launch_control_depth=2).get_kernel()
    assert "ClusterLaunchQueue<2>" in src
    assert "ClusterLaunchCursor<2>" in src


# -- four ways to get it wrong --------------------------------------------- #

def test_the_cursor_is_not_in_shared_memory():
    """Per-thread state; in shared memory the parity is a race, not a counter."""
    src = _kernel(launch_control=True, enable_wrap_loads=False).get_kernel()
    cursor = next(line for line in src.splitlines()
                  if "ClusterLaunchCursor" in line)
    assert "__shared__" not in cursor, (
        f"the cursor must be a local, got: {cursor.strip()}")
    queue = next(line for line in src.splitlines()
                 if "ClusterLaunchQueue" in line)
    assert "__shared__" in queue, (
        f"the response slots must be shared, got: {queue.strip()}")


def test_the_handoff_sits_outside_the_element_guard():
    """The barrier it carries has to be reached by every row of the block."""
    src = _kernel(launch_control=True, enable_wrap_loads=False).get_kernel()
    guard = _depth_of(src, re.search(r"if \(\w*batchId0 < numElements0\)",
                                     src).group(0))
    handoff = _depth_of(src, ".next(launchQueue0)")
    assert handoff == guard, (
        f"the hand-off is nested {handoff - guard} level(s) deeper than the "
        f"element guard, so its barrier is reached by some rows only")


def test_no_loop_carried_barrier_is_appended_under_the_guard():
    """The hand-off is the barrier; a second one would land inside the guard.

    `next` carries a `__syncthreads` of its own, in the header and outside the
    guard, so the traversal has nothing left to append.  Anything block-wide
    appearing in the generated body would be inside the guard, which is where
    a barrier hangs the mode.
    """
    src = _kernel(launch_control=True, enable_wrap_loads=False).get_kernel()
    barriers = [line.strip() for line in src.splitlines()
                if "__syncthreads()" in line]
    assert not barriers, (
        f"{len(barriers)} block barrier(s) emitted in the launch-control body, "
        f"first: {barriers[0]}; the hand-off already carries one outside the "
        f"guard, and a second lands inside it")


def test_the_grid_stride_loop_still_gets_its_loop_carried_barrier():
    """Leaving the append out of the queue loop must not leave it out of the
    other mode.

    The grid-stride loop has no hand-off to carry one, so its separation
    between this iteration's reads and the next one's writes is exactly this
    appended barrier -- one per element whatever the body does, appended
    after optimization so that it closes the body it is appended to.
    """
    from tensorforge.backend.instructions.batch_loop import BatchLoop
    from tensorforge.backend.instructions.sync_block import SyncThreads

    stream = _kernel()._sections[0].stream
    loop = next((i for i in stream if isinstance(i, BatchLoop)), None)
    assert loop is not None, "no batch loop in the persistent section"
    assert any(isinstance(i, SyncThreads) for i in loop.region), (
        "the persistent loop lost the barrier that guards the next "
        "iteration's writes against this one's reads")


def test_no_host_only_types_reach_the_device():
    """`std::optional` compiles only under --expt-relaxed-constexpr."""
    from pathlib import Path
    header = (Path(__file__).resolve().parent.parent / "src" / "tensorforge" /
              "include" / "tensorforge_device" / "cuda.h").read_text()
    assert "std::optional" not in header
    assert "#include <optional>" not in header


# -- refusals rather than silent substitutions ----------------------------- #

def test_an_older_target_is_refused_rather_than_downgraded():
    """Asked for and unavailable must not quietly give the other traversal."""
    with pytest.raises(GenerationError, match="sm_100"):
        _kernel(arch=HOPPER, launch_control=True)


@pytest.mark.parametrize("extra", [{}, {"enable_multibuffer": True}])
def test_prefetching_at_depth_one_is_refused(extra):
    """The element a transfer is moved ahead for is the one the queue hands
    out next, and at depth one that answer is asked for at the bottom of the
    iteration before: read at the top, it would be waited for with no body
    in between.  A second stage changes which buffer is filled, not when the
    element is known.
    """
    with pytest.raises(GenerationError, match="launch_control_depth >= 2"):
        _kernel(launch_control=True, enable_wrap_loads=True, **extra)


# -- the next element, read off the queue ---------------------------------- #

def _lookahead(**extra):
    return _kernel(launch_control=True, launch_control_depth=2,
                   **extra).get_kernel()


def test_at_depth_two_the_next_element_is_read_off_the_queue():
    """`peek` at the top of the iteration answers what `next` answers at its
    bottom; the transfer for the next element is addressed from it, and the
    peel ahead of the loop from the first element."""
    src = _lookahead()
    lines = src.splitlines()
    loop = next(i for i, l in enumerate(lines) if "while (true)" in l)
    peek = next(i for i, l in enumerate(lines) if ".peek(launchQueue0)" in l)
    guard = next(i for i, l in enumerate(lines) if "if (allowed)" in l)
    handoff = next(i for i, l in enumerate(lines)
                   if ".next(launchQueue0)" in l)
    assert loop < peek < guard < handoff
    ahead = re.search(r"(\w+_ahead) = launchCursor0\.peek", src).group(1)
    following = re.search(rf"(\w+_batchId1) = \(\({ahead} >= 0\)", src)
    assert following, "the next element is not taken from the queue's answer"
    assert re.search(rf"wrap_glb_\w+ = &\w+\[{following.group(1)} \*", src)
    peeled = [i for i, l in enumerate(lines) if re.search(r"peel_glb_\w+ =", l)]
    assert peeled and max(peeled) < loop


def test_the_size_and_the_mask_are_one_element_guard():
    """The transfers of the element stand in one guard, which a pass looks
    into; the flag is read only for an element in range."""
    src = _lookahead()
    assert re.search(r"const bool allowed = \w+_batchId0 < numElements0 && "
                     r"\(flags0 == nullptr \|\| static_cast<bool>"
                     r"\(flags0\[\w+_batchId0\]\)\);", src)
    assert not re.search(r"if \(\w*batchId0 < numElements0\)", src)


def test_the_handoff_stays_outside_the_element_guard_with_the_lookahead():
    src = _lookahead()
    guard = _depth_of(src, "if (allowed)")
    assert _depth_of(src, ".next(launchQueue0)") == guard
    assert _depth_of(src, ".peek(launchQueue0)") == guard


def test_without_the_wrap_the_queue_loop_is_as_it_was():
    """Nothing is moved ahead, so nothing is read ahead of the queue."""
    src = _kernel(launch_control=True, launch_control_depth=2,
                  enable_wrap_loads=False).get_kernel()
    assert ".peek(" not in src
    assert re.search(r"if \(\w*batchId0 < numElements0\)", src)


def test_the_device_cursor_can_peek():
    """What the generated loop calls exists, beside `next`."""
    from pathlib import Path
    header = (Path(__file__).resolve().parent.parent / "src" / "tensorforge" /
              "include" / "tensorforge_device" / "cuda.h").read_text()
    assert "int peek(ClusterLaunchQueue<Depth> &queue)" in header


def test_a_grid_barrier_is_refused():
    """A cooperative launch waits for blocks the queue prevents from starting.

    The mechanism *is* that the launcher never starts most of the grid: a
    resident block cancels the CTAs it then runs itself.  So the blocks a grid
    barrier waits for are exactly the ones that will not exist, and the ones
    that do run disagree on how many iterations they take.  Refused with a
    `GenerationError`, because a bare `assert not coop` in the launcher would
    fire with no message and read as an internal defect rather than as the
    answer to a question the caller asked.
    """
    dt = Datatype.F32

    def square(alias):
        return SubTensor(Tensor([16, 16], Addressing.STRIDED,
                                BoundingBox([0, 0], [16, 16]),
                                alias=alias, datatype=dt))

    a, b, d, c, e = (square(x) for x in "ABDCE")
    descrs = [GemmDescr(False, False, a=a, b=b, c=d, alpha=1.0, beta=0.0),
              GridBarrierDescr(),
              GemmDescr(False, False, a=d, b=c, c=e, alpha=1.0, beta=0.0)]
    ctx = Context(arch=BLACKWELL, backend="cuda", fp_type=dt,
                  options=Options(launch_control=True,
                                  enable_wrap_loads=False))
    gen = Generator(descrs, ctx)
    with pytest.raises(GenerationError, match="grid-wide barrier"):
        gen.generate()


# -- the loop as a construct ----------------------------------------------- #

def _bodies(**options):
    """Every PIR body one kernel produces, captured on its way to the writer."""
    import tensorforge.backend.pir as pir

    out = []
    real = pir.emit

    def capture(body, writer, context=None, metrics=None):
        out.append(body)
        return real(body, writer, context, metrics)

    pir.emit = capture
    try:
        _kernel(**options).get_kernel()
    finally:
        pir.emit = real
    return out


def _the_loop(op, **options):
    from tensorforge.backend.pir.core import walk

    for body in _bodies(**options):
        for stmt, _ in walk(body):
            # The loop over the batch is the one whose induction carries the
            # macro layer's name -- as a hint, since nothing spells it, so
            # this reads the value rather than an `extern` attribute, which
            # the batch loop does not set.
            if stmt.op == op and stmt.induction.hint == "batchId0":
                return stmt
    raise AssertionError(f"no {op} over the batch in the generated body")


def test_the_queue_loop_is_a_construct_the_passes_can_walk():
    """The region is the whole loop, so the hand-off is inside what a pass sees.

    Emitted around the region instead, nothing could be moved across it -- not
    because the move is illegal but because the statement is not there to
    reason about.
    """
    from tensorforge.backend.pir.core import Effect, MemSpace, Op, walk

    loop = _the_loop(Op.WHILE, launch_control=True,
                      enable_wrap_loads=False)
    inner = [s for s, _ in walk(loop.regions[0].body)]

    guard = [s for s in inner if s.op == Op.IF]
    assert guard, "the size guard is not an `if` the passes can see"

    handoff = [s for s in inner
               if s.op == Op.CALL and "next" in (s.attr("callee") or "")]
    assert len(handoff) == 1, f"{len(handoff)} hand-offs in the loop"
    assert handoff[0].effect & Effect.BARRIER, (
        "`next` carries a block barrier; a pass that cannot see it is free to "
        "move a shared-memory access across the point where one element's "
        "tile stops being live")
    assert any(a.space is MemSpace.SHARED for a in handoff[0].accesses), (
        "the hand-off reads and writes the response slots")

    assert loop.exits, "the loop has no exit and so no trip count"


def test_the_queue_loop_is_block_uniform_and_its_element_guard_is_not():
    """The two are different questions and the form answers both.

    Every thread reads the same cancel response out of shared memory behind a
    barrier, so all of them leave on the same iteration -- the loop is entered
    block-uniformly.  The guard inside it is per element, and the rows of a
    block hold different elements.
    """
    from tensorforge.backend.pir.core import Op, Uniformity
    from tensorforge.backend.pir.passes import _entry_uniformity

    loop = _the_loop(Op.WHILE, launch_control=True,
                      enable_wrap_loads=False)
    assert _entry_uniformity(loop) is Uniformity.BLOCK

    guard = next(s for s in loop.regions[0].body if s.op == Op.IF)
    assert _entry_uniformity(guard) is Uniformity.MULT


def test_the_counted_loop_states_its_own_non_uniformity():
    """`batchId0` differs between the rows of a block, and the loop says so.

    The bounds are the macro layer's text and carry no uniformity of their
    own, so the induction is what the answer has to come from.
    """
    from tensorforge.backend.pir.core import Op, Uniformity
    from tensorforge.backend.pir.passes import _entry_uniformity

    assert _entry_uniformity(_the_loop(Op.FOR)) is Uniformity.MULT


def _loop_with_a_block_barrier(queried: bool):
    """One loop over the batch, with a block barrier directly in its body."""
    from tensorforge.backend.pir.build import IRBuilder
    from tensorforge.backend.pir.core import (BOOL, INDEX, SIZE, Participants,
                                              Uniformity)

    builder = IRBuilder(fptype=Datatype.F32)
    if queried:
        with builder.while_("start", extern="batchId0", index_type=SIZE,
                            uniform=Uniformity.MULT) as loop:
            # `Participants` says what the barrier covers in hardware;
            # `uniform` says what the region guarantees.  Two ladders, and
            # this call names the first one.
            builder.barrier(Participants.BLOCK)
            nxt = builder.call("cursor.next", INDEX, "queue", pure=False,
                               movable=False, uniform=Uniformity.BLOCK,
                               materialize=True)
            loop.exit_when(builder.op("lt", BOOL, nxt, 0))
            loop.yield_(builder.op("add", SIZE, nxt, 0))
    else:
        with builder.for_("start", "numElements0", "stride", extern="batchId0",
                          index_type=SIZE, uniform=Uniformity.MULT):
            builder.barrier(Participants.BLOCK)
    return builder.finish()


def test_a_block_barrier_is_legal_in_the_queue_loop_and_not_in_the_counted_one():
    """The one place where the queue is structurally better than the loop.

    Both loops run over the batch and neither can carry a grid barrier.  They
    differ on the block: the queue hands every thread of a block the same
    element sequence, so all of them reach a block barrier the same number of
    times, and a strided loop over `threadIdx.y + blockDim.y * blockIdx.x`
    does not.
    """
    from tensorforge.backend.pir.passes import verify

    assert verify(_loop_with_a_block_barrier(queried=True), strict=False) == []

    counted = verify(_loop_with_a_block_barrier(queried=False), strict=False)
    assert any("block-wide barrier" in d for d in counted), (
        f"the counted loop accepted a block barrier: {counted}")


def test_an_exit_belongs_to_the_loop_it_leaves():
    """Nested, the condition it fires on is not the one the loop is entered under."""
    from tensorforge.backend.pir.build import IRBuilder
    from tensorforge.backend.pir.core import BOOL, INDEX, SIZE
    from tensorforge.backend.pir.passes import verify

    builder = IRBuilder(fptype=Datatype.F32)
    with builder.while_("start", extern="batchId0", index_type=SIZE) as loop:
        nxt = builder.call("cursor.next", INDEX, "queue", pure=False,
                           movable=False, materialize=True)
        with builder.if_(builder.op("lt", BOOL, nxt, 0)):
            loop.exit_when(True)
        loop.yield_(builder.op("add", SIZE, nxt, 0))

    diag = verify(builder.finish(), strict=False)
    assert any("directly in the region of a `while`" in d for d in diag), diag


def _square(**options):
    """`square_notrans` under the queue: a load into registers the wrap
    moves ahead, and a copy into shared memory it leaves."""
    import contextlib
    import importlib.util
    import io
    from pathlib import Path

    from tensorforge.reference import kernel_eval as ke
    path = next((Path(__file__).resolve().parent / "cases").rglob(
        "square_notrans.py"))
    spec = importlib.util.spec_from_file_location("tf_lc__square", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ctx = Context(arch=BLACKWELL, backend="cuda",
                  fp_type=getattr(mod, "DTYPE", None),
                  options=Options(launch_control=True, **options))
    gen = Generator(mod.descr_list(), ctx)
    with contextlib.redirect_stdout(io.StringIO()):
        gen.generate()
    return gen.get_kernel(), ke.launch_geometry(gen.get_launcher())


def test_the_lookahead_computes_what_the_queue_loop_computes():
    """On the host oracle, with the queue handing the block its CTAs out of
    order -- the element after this one is not a stride on -- and a mask
    that holds elements back: the transfer moved ahead for the element the
    queue answers fills the buffer with that element's operands."""
    from tensorforge.reference import kernel_eval as ke

    def written(src, geometry):
        lanes, mults = geometry
        elements = 3 * mults - 1
        flags = [1] * elements
        flags[1] = 0
        flags[mults + 2] = 0
        races = []
        mem = ke.evaluate_wave(src, lanes, seed=3, globals_only=True,
                               mults=mults, block=True, races=races,
                               elements=elements, launches=(2, 1),
                               preset={"flags0": flags})
        assert not races, races
        signature = re.search(r"\bkernel_\w+\(([^)]*)\)", src).group(1)
        outputs = {p.split()[-1] for p in signature.split(",")
                   if "*" in p and not p.strip().startswith("const")
                   and "<const " not in p}
        return {k: v for k, v in mem.items() if k[0] in outputs}

    plain, at = _square(enable_wrap_loads=False)
    ahead, ahead_at = _square(launch_control_depth=2)
    assert "wrap_glb_" in ahead and "wrap_glb_" not in plain
    assert at == ahead_at
    expected = written(plain, at)
    assert expected and written(ahead, ahead_at) == expected


def test_the_oracles_queue_answers_in_order_then_nothing():
    from tensorforge.reference.kernel_eval import _LaunchCursor
    cursor = _LaunchCursor((5, 3))
    cursor.start(None)
    assert cursor.peek(None) == 5 and cursor.peek(None) == 5
    assert cursor.next(None) == 5
    assert cursor.peek(None) == 3 and cursor.next(None) == 3
    assert cursor.peek(None) == -1 and cursor.next(None) == -1
