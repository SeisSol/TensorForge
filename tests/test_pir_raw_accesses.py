# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""A raw statement may say what it touches, and must say what it uses.

`Op.RAWSTMT` answers two questions.  *What it is* -- opaque text, unmovable,
not a candidate for CSE -- is settled and stays `Effect.UNKNOWN`.  *What it
touches* defaults to the same answer, `Access(READ|WRITE, UNKNOWN, None)`,
which conflicts with every buffer in every space.

That default has a specific cost.  One raw statement between two shared-memory
accesses keeps every buffer live, so a body that is nine tenths converted lays
out like one that is not converted at all: the allocator gives no buffer the
bytes of another across it (`pir.allocate`).  The conversion would be
all-or-nothing, and an all-or-nothing conversion does not get done.
`accesses=` is the way out; omitting it keeps the conservative default.

The operand requirement is here because declaring an access is not declaring a
use.  Take a `float4` staging store that names its tile only inside the text:
given a correct `accesses=` and no operand, the alloc that produces the tile is
reachable by nothing, `dce` removes it, and the kernel refers to an undeclared
pointer.  Correct as IR, not C++ at all -- a snapshot shows a diff and the
verifier sees nothing wrong.
"""

from __future__ import annotations

import pytest

from tensorforge.backend.pir import passes
from tensorforge.backend.pir.build import IRBuilder
from tensorforge.backend.pir.core import (INDEX, Access, Effect, IRError,
                                          MemSpace, Op, walk)
from tensorforge.backend.pir.emit import Emitter
from tensorforge.common.basic_types import Datatype
from tensorforge.common.exceptions import GenerationError
from harness.placement import placed


def builder(arena=True):
    return IRBuilder(fptype=Datatype.F32, arena='shrMem' if arena else None)


def emitted(body):
    lines = []
    Emitter(lines.append).run(placed(body))
    return lines


# --------------------------------------------------------------------------- #
# What it touches
# --------------------------------------------------------------------------- #

def test_omitting_the_argument_keeps_the_conservative_answer():
    """A call site that says nothing gets the conservative answer."""
    b = builder()
    s = b('__syncwarp();')
    assert len(s.accesses) == 1
    assert s.accesses[0].space == MemSpace.UNKNOWN
    assert s.accesses[0].base is None


def test_a_statement_that_touches_nothing_may_say_so():
    b = builder()
    assert b('__syncwarp();', accesses=()).accesses == ()


def test_the_statement_stays_opaque_either_way():
    """Narrowing the memory answer must not make the text look analysable.

    Nothing here can reason about a raw statement *as code*, so it stays
    pinned and impure whatever it says about memory.  A statement that became
    movable because it declared no accesses could be hoisted across the
    barrier it was ordering.
    """
    b = builder()
    for s in (b('__syncwarp();'), b('__syncwarp();', accesses=())):
        assert s.effect == Effect.UNKNOWN
        assert not s.movable
        assert not s.pure


# --------------------------------------------------------------------------- #
# What it uses
# --------------------------------------------------------------------------- #

def test_a_buffer_named_in_the_text_must_be_an_operand():
    b = builder()
    tile = b.alloc(Datatype.F32, (16,), MemSpace.SHARED, hint='tile')
    write = (Access(Effect.WRITE, MemSpace.SHARED, tile),)
    with pytest.raises(IRError, match="without listing it"):
        b(f'*(float4*)&{tile}[0] = x;', accesses=write)


def test_a_buffer_named_in_the_text_must_have_a_declared_access():
    b = builder()
    tile = b.alloc(Datatype.F32, (16,), MemSpace.SHARED, hint='tile')
    with pytest.raises(IRError, match="does not declare an access"):
        b(f'{tile}[0] = 1.0f;', tile, accesses=())


def test_any_named_value_must_be_an_operand_not_only_buffers():
    """The load result is the same defect one step along: named in the text,
    invisible to the use chain, and `dce` is right to remove what defines
    it."""
    b = builder()
    tile = b.alloc(Datatype.F32, (16,), MemSpace.SHARED, hint='tile')
    idx = b.rawexpr('threadIdx.x', type_=INDEX, hint='a')
    val = b.load(tile, idx, hint='data')
    with pytest.raises(IRError, match="without listing it"):
        b(f'float x = {val};', accesses=())
    assert b(f'float x = {val};', val, accesses=()) is not None


def test_a_declaration_says_defines_not_args():
    """A raw declaration names its value without using it.

    `float v35[4][2]{};` is where the name comes from, so requiring it in
    `args` would be asking for a use edge that runs backwards.
    """
    b = builder()
    tile = b.alloc(Datatype.F32, (16,), MemSpace.SHARED, hint='tile')
    idx = b.rawexpr('threadIdx.x', type_=INDEX, hint='a')
    val = b.load(tile, idx, hint='data')
    stmt = b(f'float copy = {val};', val, accesses=())
    assert val in stmt.args


def test_a_varalloc_name_is_not_asked_to_claim_anything():
    """`varalloc` reserves a C++ identifier; it does not define an IR value.

    Text-based bodies redeclare one per scope --- `float v35[4][2]{};` inside
    each iteration --- which is ordinary C++ and not SSA at all.  There is no
    defining statement for `dce` to remove, so the argument the check rests on
    does not apply, and demanding a claim would only teach callers to drop the
    `accesses` argument that turns the check on.
    """
    b = builder()
    name = b.varalloc()
    assert b(f'float {name}[4][2]{{}};', accesses=()) is not None


def test_a_name_that_is_not_ours_is_left_alone():
    """Kernel parameters, loop variables and C++ keywords are not values this
    body defined, and a check that fired on them would teach callers to drop
    the argument that turns it on."""
    b = builder()
    assert b('float x = glb_m0[threadIdx.x] + v_not_a_value;',
             accesses=()) is not None


def test_a_prefix_of_a_value_name_does_not_count_as_a_use():
    """`v1` must not match `v13`."""
    b = builder()
    for _ in range(13):
        b.rawexpr('0', type_=INDEX, hint='')
    b('float x = 1;', accesses=())          # names nothing


# --------------------------------------------------------------------------- #
# The use edge
# --------------------------------------------------------------------------- #

def test_an_operand_keeps_the_allocation_alive_through_dce():
    """The whole reason the operand requirement exists.

    Without the use edge the `alloc` is reachable by nothing, `dce` removes
    it, and the emitted kernel subscripts a pointer that was never declared.
    """
    b = builder()
    tile = b.alloc(Datatype.F32, (16,), MemSpace.SHARED, hint='tile')
    b(f'*(float4*)&{tile}[0] = make_float4(1, 2, 3, 4);', tile,
      accesses=(Access(Effect.WRITE, MemSpace.SHARED, tile),))
    body = passes.dce(b.finish())
    assert any(s.op == Op.ALLOC for s, _ in walk(body)), (
        "the allocation was removed despite being used")
    assert any('_tile = &shrMem' in line for line in emitted(body))


# --------------------------------------------------------------------------- #
# Where a shared buffer goes
# --------------------------------------------------------------------------- #

def test_a_body_with_no_arena_cannot_allocate_shared():
    """A shared buffer is a window into an arena the allocator lays out,
    and a body without one has nowhere to put it."""
    b = builder(arena=False)
    with pytest.raises(GenerationError, match="no arena"):
        b.alloc(Datatype.F32, (4,), MemSpace.SHARED, hint='nope')
