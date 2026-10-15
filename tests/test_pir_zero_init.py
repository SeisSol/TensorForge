# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""`fold` knows a pre-zeroed register buffer still holds its zeros.

A reduction allocates its accumulator with `init='{}'` and folds in one term
at a time, so the first fold reads a slot nothing has written: the addition
is an identity and the load behind it is dead.  Nothing in the IR said so
before, and the additions reached the machine -- `ncu` counted 315 of them
per element in SeisSol's `derivativeTaylorExpansion`, a fifth of everything
that kernel computes.

What the pass may *not* do is the reason for most of these tests.  A slot
that has been stored into holds what was stored; a buffer in shared memory
holds what another lane put there; and a loop body is walked once while it
runs many times, so what it stored on one trip it reads on the next.
"""

from __future__ import annotations

from tensorforge.backend.pir import passes
from tensorforge.backend.pir.build import IRBuilder
from tensorforge.backend.pir.core import (INDEX, MemSpace, Op, ScalarType,
                                          walk_stmts)
from tensorforge.common.basic_types import Datatype

F32 = ScalarType(Datatype.F32)


def _ops(body, op):
    return [s for s in walk_stmts(body) if s.op == op]


def _accumulator(space=MemSpace.REGISTER, init='{}'):
    """`acc[0] = acc[0] + x`, the shape every reduction starts with."""
    b = IRBuilder(arena='shrMem')
    acc = b.alloc(Datatype.F32, (4,), space, hint='acc', init=init)
    src = b.alloc(Datatype.F32, (4,), MemSpace.REGISTER, hint='src')
    x = b.load(src, 0, space=MemSpace.REGISTER)
    prev = b.load(acc, 0, space=space)
    b.store(acc, b.op('add', F32, prev, x), 0, space=space)
    return b, acc


def test_the_first_fold_into_a_zeroed_accumulator_is_an_assignment():
    b, _ = _accumulator()
    body = passes.dce(passes.fold(b.finish()))
    assert not _ops(body, 'add'), 'the addition against zero should be gone'
    stores = _ops(body, Op.STORE)
    assert len(stores) == 1
    # and what it stores is the term itself, not a sum
    assert stores[0].args[1].id == _ops(body, Op.LOAD)[0].target[0].id


def test_a_slot_already_written_is_not_zero_any_more():
    """The second fold reads what the first stored."""
    b = IRBuilder()
    acc = b.alloc(Datatype.F32, (4,), MemSpace.REGISTER, hint='acc', init='{}')
    src = b.alloc(Datatype.F32, (4,), MemSpace.REGISTER, hint='src')
    for term in range(2):
        x = b.load(src, term, space=MemSpace.REGISTER)
        prev = b.load(acc, 0, space=MemSpace.REGISTER)
        b.store(acc, b.op('add', F32, prev, x), 0, space=MemSpace.REGISTER)
    body = passes.dce(passes.fold(b.finish()))
    assert len(_ops(body, 'add')) == 1, 'only the first fold is an identity'


def test_a_vector_slot_is_not_given_a_scalar_zero():
    """One zero where a lane-wide one belongs is a missing splat.

    `a * b + 0.0` over a four-wide value is not a narrower constant: the
    reference evaluator rejects it outright ("vector op with a scalar"),
    and a target may or may not broadcast it.  Writing the splat here would
    mean inventing the width's spelling in a pass that has no target, so
    the fold does not apply to a vector at all.
    """
    b = IRBuilder(arena='shrMem')
    acc = b.alloc(Datatype.F32, (4,), MemSpace.REGISTER, hint='acc', init='{}')
    src = b.alloc(Datatype.F32, (4,), MemSpace.REGISTER, hint='src')
    wide = ScalarType(Datatype.F32, 2)
    x = b.load(src, 0, type_=wide, space=MemSpace.REGISTER)
    prev = b.load(acc, 0, type_=wide, space=MemSpace.REGISTER)
    b.store(acc, b.op('add', wide, prev, x), 0, space=MemSpace.REGISTER)
    body = passes.dce(passes.fold(b.finish()))
    assert len(_ops(body, 'add')) == 1, 'a vector zero must not be folded away'


def test_a_shared_accumulator_keeps_its_addition():
    """Another lane may have written it, and this body cannot see that."""
    b, _ = _accumulator(space=MemSpace.SHARED)
    body = passes.dce(passes.fold(b.finish()))
    assert len(_ops(body, 'add')) == 1


def test_an_uninitialized_accumulator_keeps_its_addition():
    b, _ = _accumulator(init=None)
    body = passes.dce(passes.fold(b.finish()))
    assert len(_ops(body, 'add')) == 1


def test_a_variable_index_forgets_the_whole_buffer():
    """A store nobody can place may have landed anywhere in it."""
    b = IRBuilder()
    acc = b.alloc(Datatype.F32, (4,), MemSpace.REGISTER, hint='acc', init='{}')
    src = b.alloc(Datatype.F32, (4,), MemSpace.REGISTER, hint='src')
    unknown = b.rawexpr('threadIdx.x', type_=INDEX, hint='tid')
    b.store(acc, b.load(src, 0, space=MemSpace.REGISTER), unknown,
            space=MemSpace.REGISTER)
    prev = b.load(acc, 0, space=MemSpace.REGISTER)
    b.store(acc, b.op('add', F32, prev, b.load(src, 1,
                                               space=MemSpace.REGISTER)), 0,
            space=MemSpace.REGISTER)
    body = passes.dce(passes.fold(b.finish()))
    assert len(_ops(body, 'add')) >= 1, 'the unplaceable store forgets it'
