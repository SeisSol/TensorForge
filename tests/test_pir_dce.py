# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""`dce` counts the reads of a statement off once.

A dead statement with a region takes what is inside it along, and everything
in there loses its reads.  A statement in there can also die on its own: the
load whose only reader was the region's `yield` dies once the branch has.
Counting its reads off a second time takes a read away from a value it shares
with a live statement elsewhere -- the lane index `cse` merges across the
branches of the predicated loads of one operand -- and the producer of that
value goes while the live statement still reads it.

That is what the literals of `inline_constants` bring about: a zero entry
drops its product, the predicated load feeding only that product is dead, and
the kernel came out reading an index it never declared (the order-3 fused
SeisSol kernels on oneAPI, `use of undeclared identifier`).
"""

from __future__ import annotations

from tensorforge.backend.pir import passes
from tensorforge.backend.pir.build import IRBuilder
from tensorforge.backend.pir.core import BOOL, INDEX, MemSpace, Op, ScalarType, walk_stmts
from tensorforge.common.basic_types import Datatype

F32 = ScalarType(Datatype.F32)


def _predicated_load(builder, image, index, guard):
    handle = builder.if_else(guard, (F32,))
    with handle.then():
        handle.yield_(builder.load(image, index, hint='stored'))
    with handle.otherwise():
        handle.yield_(builder.const(0.0))
    return handle.result


def _kernel():
    """Two predicated loads at one index; only the second one is read."""
    builder = IRBuilder(fptype=Datatype.F32)
    image = builder.alloc(Datatype.F32, (16,), MemSpace.REGISTER, hint='image')
    lane = builder.thread_id('x')
    index = builder.op('add', INDEX, lane, 4, hint='idx')
    guard = builder.op('lt', BOOL, lane, 9, hint='guard')
    _predicated_load(builder, image, index, guard)
    builder.store(image, _predicated_load(builder, image, index, guard), 0)
    return builder.finish(), index


def test_the_index_of_a_live_load_survives_a_dead_load_at_the_same_index():
    body, index = _kernel()
    body = passes.dce(body)
    produced = {x.id for s in walk_stmts(body) for x in s.target}
    read = {v.id for s in walk_stmts(body) for v in s.operands()}
    assert index.id in read, 'the live load no longer reads the index'
    assert index.id in produced, 'the index is read but nothing produces it any more'


def test_the_dead_load_goes_with_its_branch():
    body, _ = _kernel()
    body = passes.dce(body)
    branches = [s for s in walk_stmts(body) if s.regions]
    loads = [s for s in walk_stmts(body) if s.op == Op.LOAD]
    assert len(branches) == 1 and len(loads) == 1, [s.op for s in walk_stmts(body)]
