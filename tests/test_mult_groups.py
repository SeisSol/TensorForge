# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""A multiplication that does not divide the wave shares one with its neighbour.

A barrier reaches waves, never parts of one. So where `num_threads` is not a
multiple of the wave, the smallest set of multiplications a barrier can
separate is a *group* of them -- two 48-thread multiplications fill three
32-lane waves, and no barrier can tell one from the other.

Which is not a corner: `Context.align` rounds to
`vec_unit_length * hw_fp_word_size / fp_size`, and at eight bytes per element
that is half a wave, so half the widths above the wave are odd multiples of it.
Double precision is where this lives.

The consequence is that the whole group has to reach every barrier the same
number of times. So the traversal is driven by the group's leader rather than
by each row, and the row whose element is out of range is not branched around
-- it runs the body, reads a clamped duplicate, and writes nothing.
"""
from __future__ import annotations

import re

import pytest

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.context import Context
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.common.operation import Operation
from tensorforge.common.threads import mults_per_group
from tensorforge.generators.descriptions import (ElementwiseDescr,
                                                 MultilinearDescr)
from tensorforge.generators.generator import Generator


def _tensor(shape, alias, dtype):
    return SubTensor(Tensor(shape, Addressing.STRIDED,
                            BoundingBox([0] * len(shape), list(shape)),
                            alias=alias, datatype=dtype))


def _wide_section(lead, dtype):
    """A multilinear beside an elementwise, which is what waives the lane cap."""
    gemm = MultilinearDescr(_tensor([lead, 18], "C", dtype),
                            [_tensor([lead, 18], "A", dtype),
                             _tensor([18, 18], "B", dtype)],
                            [[0, -1], [-1, 1]], [[0, 1], [0, 1]])
    ew = ElementwiseDescr(Operation.ABS, _tensor([lead, 18], "F", dtype),
                          [_tensor([lead, 18], "E", dtype)])
    return [gemm, ew]


def test_group_size_is_the_waves_a_multiplication_shares():
    assert mults_per_group(64, 32) == 1, "a whole wave is its own group"
    assert mults_per_group(48, 32) == 2, "two of these fill three waves"
    assert mults_per_group(96, 64) == 2
    assert mults_per_group(16, 32) == 2, "two of these fill one wave"
    assert mults_per_group(0, 32) == 1, "no width is not a crash"


def test_double_precision_reaches_the_ragged_case():
    """Stated as a test because the whole group path hangs off it."""
    ctx = Context(arch="sm_86", backend="cuda", fp_type=Datatype.F64)
    wave = ctx.get_vm().get_hw_descr().vec_unit_length
    ragged = [w for w in (ctx.align(n) for n in range(1, 200))
              if w > wave and mults_per_group(w, wave) > 1]
    assert ragged, "no width above the wave is ragged; the group path is dead"


def test_the_block_holds_exactly_one_group():
    """A group narrower than the block would not be separable either.

    The barrier the group buys reaches the block, so a second group inside it
    is free to run the body a different number of times and the barrier is back
    to being a deadlock.
    """
    ctx = Context(arch="sm_86", backend="cuda", fp_type=Datatype.F64)
    wave = ctx.get_vm().get_hw_descr().vec_unit_length
    gen = Generator(_wide_section(40, Datatype.F64), ctx)
    gen.generate()

    threads = gen._num_threads
    assert threads % wave != 0, f"{threads} divides the wave; not the ragged case"
    group = mults_per_group(threads, wave)
    assert group == 2
    for section in gen._sections:
        assert section.shr_mem_obj.get_mults_per_block() == group


def test_the_grouped_traversal_is_driven_by_the_leader():
    ctx = Context(arch="sm_86", backend="cuda", fp_type=Datatype.F64)
    gen = Generator(_wide_section(40, Datatype.F64), ctx)
    gen.generate()
    src = gen.get_kernel()

    assert "batchIdLane0 = threadIdx.y % 2" in src
    assert "batchIdGroup0 = (threadIdx.y - batchIdLane0)" in src, (
        "the loop has to start at the group's lowest index, so that it runs as "
        "often as the row with the most to do")
    assert "const size_t batchId0 = batchIdActive0 ?" in src, (
        "a row with no element of its own still reads, so its index has to be "
        "clamped to one that exists")


def test_every_global_write_of_a_grouped_body_runs_under_the_mask():
    """The row that has no element runs the body and writes nothing."""
    ctx = Context(arch="sm_86", backend="cuda", fp_type=Datatype.F64)
    gen = Generator(_wide_section(40, Datatype.F64), ctx)
    gen.generate()
    lines = [line.strip() for line in gen.get_kernel().splitlines()]

    writes = [i for i, line in enumerate(lines)
              if re.match(r"glb_\w+\[.*\] = ", line)]
    assert writes, "no global write found; the check would pass vacuously"
    for i in writes:
        assert any("batchIdActive0" in lines[j] for j in range(max(0, i - 3), i)), (
            f"unmasked global write: {lines[i]}")


def test_the_body_of_a_grouped_loop_is_group_uniform():
    from tensorforge.backend.instructions.batch_loop import BatchLoop

    ctx = Context(arch="sm_86", backend="cuda", fp_type=Datatype.F64)
    gen = Generator(_wide_section(40, Datatype.F64), ctx)
    gen.generate()
    loops = [i for s in gen._sections for i in s.stream
             if isinstance(i, BatchLoop)]
    assert loops, "no batch loop in the stream"
    for loop in loops:
        assert loop.uniform_scope().name == "GROUP"


def test_a_rotated_start_stays_per_row():
    """The modulo does not distribute over the lane offset.

    A rotated start is taken modulo the stride, so the leader's start plus a
    lane is not the row's own start once the wrap falls between them. Rows
    would collide on one element and miss another, which is a wrong answer
    rather than a deadlock -- so the group is refused instead.
    """
    ctx = Context(arch="sm_86", backend="cuda", fp_type=Datatype.F64)
    gen = Generator(_wide_section(40, Datatype.F64), ctx)
    gen._num_threads = 48
    plain = gen._get_2d_block_id()
    assert gen._group_size(0, plain) == 2
    assert gen._group_size(0, f"({plain} + numElements1) % (x)") == 1


def test_a_mask_never_reaches_shared_memory_or_registers():
    """Scratch is the multiplication's own; what a row writes there it reads."""
    from tensorforge.backend import elementmask

    assert elementmask.active() == (None, None)
    with elementmask.element_mask(None, "active0"):
        assert elementmask.active() == (None, "active0")
        with elementmask.element_mask(None, "inner0"):
            assert elementmask.active()[1] == "inner0"
        assert elementmask.active()[1] == "active0"
    assert elementmask.active() == (None, None)
