# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""A shared alloc is a window into the arena, not an array of its own.

Shared memory in a generated kernel is one arena per thread block. `ShrMemOpt`
sizes it before any instruction body is built: the coloured region area, plus
a bank-conflict pad, plus a scratch tail whose size is the maximum of
`temp_shmem()` over the stream. That number is what the occupancy calculation
reads and what the barrier placement keys on.

So `__shared__ float buf[N];` written inside a body would be memory the
allocator does not know about. It would compile, and it would silently reduce
the occupancy the generator believes it has -- the failure mode is a
performance cliff with no diagnostic, which is the kind that survives for a
long time.

These tests pin the alternative: `alloc(space=SHARED)` hands out a window of
the tail this instruction declared, the windows do not overlap, and asking for
more than was declared is an error at generation time rather than a silent
overrun into whatever the pad happened to leave free.
"""

from __future__ import annotations

import re

import pytest

from tensorforge.backend.pir import emit
from tensorforge.backend.pir.build import IRBuilder
from tensorforge.backend.pir.core import MemSpace
from tensorforge.backend.writer import Writer
from tensorforge.common.basic_types import Datatype
from tensorforge.common.exceptions import GenerationError
from tensorforge.common.vm.vm import vm_factory


ARENA = 'tempShrMem'


def _src(builder) -> str:
    """Emit without the optimiser: DCE drops a buffer nothing reads.

    That is correct of DCE and unhelpful here -- these tests are about where a
    buffer is placed, not about whether it survives. The tests that do go
    through the full path (below) store into the buffer, which is what a real
    caller does anyway.
    """
    w = Writer()
    emit(builder.finish(), w, vm_factory('sm_86', 'cuda', 'float'))
    return w.get_src()


def _windows(src: str):
    """(name, offset) for every arena suballocation in the emitted source."""
    return [(m.group(1), int(m.group(2))) for m in
            re.finditer(rf'\*\s*(\w+)\s*=\s*&{ARENA}\[(\d+)\]', src)]


# ---------------------------------------------------------------------- #
# where the buffer comes from
# ---------------------------------------------------------------------- #

def test_a_shared_alloc_points_into_the_arena():
    b = IRBuilder(fptype=Datatype.F32, scratch=(ARENA, 256))
    b.alloc(Datatype.F32, (16, 4), MemSpace.SHARED, hint='tile')
    src = _src(b)
    assert _windows(src) == [('v0_tile', 0)], src


def test_no_shared_array_is_declared_inside_a_body():
    """The regression this whole file exists for."""
    b = IRBuilder(fptype=Datatype.F32, scratch=(ARENA, 256))
    b.alloc(Datatype.F32, (16, 4), MemSpace.SHARED, hint='tile')
    assert '__shared__' not in _src(b)


def test_register_and_global_allocs_are_untouched():
    """Only shared memory is arena-managed; the others keep their declaration.

    A register buffer is a real local array and a global one is a parameter
    stand-in. Neither is a suballocation of anything, and routing them through
    the arena would be wrong rather than merely unnecessary.
    """
    b = IRBuilder(fptype=Datatype.F32, scratch=(ARENA, 256))
    b.alloc(Datatype.F32, (8,), MemSpace.REGISTER, hint='r')
    b.alloc(Datatype.F32, (8,), MemSpace.GLOBAL, hint='g')
    src = _src(b)
    assert _windows(src) == []
    assert 'float v0_r[8];' in src
    assert 'float v1_g[8];' in src


# ---------------------------------------------------------------------- #
# how the windows are placed
# ---------------------------------------------------------------------- #

def test_windows_do_not_overlap():
    sizes = [(16, 4), (8,), (3,), (12, 2)]
    b = IRBuilder(fptype=Datatype.F32, scratch=(ARENA, 512))
    vols = []
    for i, shape in enumerate(sizes):
        v = b.alloc(Datatype.F32, shape, MemSpace.SHARED, hint=f't{i}')
        vols.append(v.type.volume)
    offsets = [off for _, off in _windows(_src(b))]
    assert len(offsets) == len(sizes)
    ends = [o + v for o, v in zip(offsets, vols)]
    for i in range(1, len(offsets)):
        assert offsets[i] >= ends[i - 1], (
            f'window {i} at {offsets[i]} starts inside window {i-1} '
            f'which runs to {ends[i-1]}')


@pytest.mark.parametrize('dtype,align', [(Datatype.F32, 4), (Datatype.F64, 2)])
def test_windows_are_16_byte_aligned(dtype, align):
    """`nvidia.matmul` stores through `float4`; an unaligned window faults.

    The alignment is in elements, so it depends on the element size -- which is
    why this is parametrised rather than asserting `% 4 == 0`. `ShrMemOpt` pads
    the arena to the same boundary, so a window that is aligned within the tail
    is aligned absolutely.
    """
    b = IRBuilder(fptype=dtype, scratch=(ARENA, 512))
    # A deliberately awkward first size: 3 elements would leave the next
    # window at 3 if nothing rounded it up.
    b.alloc(dtype, (3,), MemSpace.SHARED, hint='odd')
    b.alloc(dtype, (8,), MemSpace.SHARED, hint='next')
    offsets = [off for _, off in _windows(_src(b))]
    assert offsets[0] == 0
    assert offsets[1] % align == 0, offsets
    assert offsets[1] >= 3


# ---------------------------------------------------------------------- #
# what the declared budget buys
# ---------------------------------------------------------------------- #

def test_overflowing_the_declared_budget_raises():
    b = IRBuilder(fptype=Datatype.F32, scratch=(ARENA, 16))
    with pytest.raises(GenerationError, match='scratch overflow'):
        b.alloc(Datatype.F32, (32,), MemSpace.SHARED, hint='big')


def test_overflow_counts_the_windows_already_handed_out():
    """Each alloc fits; together they do not. The second one is the error."""
    b = IRBuilder(fptype=Datatype.F32, scratch=(ARENA, 24))
    b.alloc(Datatype.F32, (16,), MemSpace.SHARED, hint='a')
    with pytest.raises(GenerationError, match='scratch overflow'):
        b.alloc(Datatype.F32, (16,), MemSpace.SHARED, hint='b')


def test_a_budget_of_zero_is_not_a_budget():
    """An instruction that declared nothing may not allocate.

    Falling back to a bare `__shared__` array here is the tempting reading --
    it generates, after all. It is also exactly the uncounted allocation this
    file is about, so `temp_shmem()` returning 0 has to mean *no*, not
    *unlimited*.
    """
    b = IRBuilder(fptype=Datatype.F32)
    with pytest.raises(GenerationError, match='no scratch budget'):
        b.alloc(Datatype.F32, (4,), MemSpace.SHARED, hint='x')


def test_alignment_padding_counts_against_the_budget():
    """The pad is real memory; a budget check that ignores it under-counts."""
    b = IRBuilder(fptype=Datatype.F32, scratch=(ARENA, 7))
    b.alloc(Datatype.F32, (3,), MemSpace.SHARED, hint='a')
    # 3 rounds up to 4, leaving 3 of the 7 -- so a 4-element request fails
    # even though 7 - 3 == 4 would suggest it fits.
    with pytest.raises(GenerationError, match='scratch overflow'):
        b.alloc(Datatype.F32, (4,), MemSpace.SHARED, hint='b')


# ---------------------------------------------------------------------- #
# the wiring
# ---------------------------------------------------------------------- #

def test_an_instruction_body_is_given_the_budget_it_declared():
    """`through_pir` must hand `temp_shmem()` to the builder it creates.

    Nothing in the current corpus declares a nonzero budget -- the NVIDIA
    matmul path that does is gated off -- so without this test the wiring is
    unexercised and would break silently the day that path is enabled.
    """
    from tensorforge.backend.instructions.abstract_instruction import (
        AbstractInstruction)

    class _Declares(AbstractInstruction):
        """Minimal instruction: declares a tail and allocates out of it."""

        def __init__(self, context, budget):
            super().__init__(context)
            self._budget = budget
            self.seen = None

        def temp_shmem(self):
            return self._budget

        def gen_ir(self, builder):
            self.seen = builder._scratch
            buf = builder.alloc(builder._fptype, (8,), MemSpace.SHARED,
                                hint='t')
            # A buffer nothing touches is dead and DCE says so, correctly.
            builder.store(buf, builder.const(0.0), builder.thread_id('x'))

        def __str__(self):
            return 'declares'

    from tensorforge.common.context import Context
    ctx = Context(arch='sm_86', backend='cuda', fp_type=Datatype.F32)
    instr = _Declares(ctx, 64)
    w = Writer()
    instr.gen_code(w)

    assert instr.seen == (ARENA, 64)
    assert _windows(w.get_src()) == [('v0_t', 0)], w.get_src()


def test_a_body_that_declared_nothing_is_given_nothing():
    from tensorforge.backend.instructions.abstract_instruction import (
        AbstractInstruction)

    class _Silent(AbstractInstruction):
        def __init__(self, context):
            super().__init__(context)
            self.seen = 'unset'

        def gen_ir(self, builder):
            self.seen = builder._scratch

        def __str__(self):
            return 'silent'

    from tensorforge.common.context import Context
    ctx = Context(arch='sm_86', backend='cuda', fp_type=Datatype.F32)
    instr = _Silent(ctx)
    instr.gen_code(Writer())
    assert instr.seen is None
