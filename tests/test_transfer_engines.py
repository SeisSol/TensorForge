# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Every transfer goes through a registered engine of its op
(`engines.TRANSFER_ENGINES`, `engines.transfer_engine_refusal`).

The block reads are engines of the registry now: a window's or a run's block
message (`block`) and Intel's 2D block read (`block2d`), which was a `block`
read told apart by its pitch.  The verifier holds a transfer that names an
engine to one that exists and moves its op.
"""

from __future__ import annotations

import dataclasses

import pytest

from tensorforge.backend.pir import block_reads, engines
from tensorforge.backend.pir.build import IRBuilder
from tensorforge.backend.pir.core import (IRError, MemSpace, Op, ScalarType,
                                          Stmt)
from tensorforge.backend.pir.passes import verify
from tensorforge.common.basic_types import Datatype

F32 = Datatype.F32


def _load(engine=None):
    b = IRBuilder(fptype=F32)
    g = b.alloc(F32, (64,), MemSpace.GLOBAL, hint='g', extern='m0')
    b.load(g, 3, hint='x')
    load = next(s for s in b.finish() if s.op == Op.LOAD)
    if engine is not None:
        load = load.with_attr('engine', engine)
    return b, load


def test_the_block_engines_are_registered_for_loads():
    for name in block_reads.BLOCK_ENGINES:
        engine = engines.TRANSFER_ENGINES[name]
        assert engine.ops == (Op.LOAD,)
        assert engine.completes is engines.Completion.SYNC


@pytest.mark.parametrize('engine', [None, 'scalar', 'block', 'block2d',
                                    'buffer'])
def test_a_load_through_an_engine_of_loads_is_one(engine):
    _, load = _load(engine)
    assert engines.transfer_engine_refusal(load) is None


def test_a_load_through_an_engine_that_is_none_is_refused():
    _, load = _load('slm2d')
    assert 'no transfer engine' in engines.transfer_engine_refusal(load)
    with pytest.raises(IRError, match='no transfer engine'):
        verify((load,))


def test_a_transfer_through_an_engine_of_another_op_is_refused():
    """A 2D block reads; a copy through cp.async's engine is a copy."""
    b = IRBuilder(fptype=F32)
    g = b.alloc(F32, (64,), MemSpace.GLOBAL, hint='g', extern='m0')
    b.store(g, 1.0, 0)
    store = next(s for s in b.finish() if s.op == Op.STORE)
    store = store.with_attr('engine', 'block2d')
    assert 'which moves load' in engines.transfer_engine_refusal(store)
    with pytest.raises(IRError, match='which moves load'):
        verify((store,))


def test_an_engine_of_another_kind_is_not_a_transfer_s():
    """A matrix op, a fence, a wait and a commit name engines of their
    own, which are no transfer engines and are not held to them."""
    s = Stmt(op=Op.FENCE, pure=False, movable=False,
             attrs=(('engine', 'wgmma'),), effect=engines.Effect.UNKNOWN)
    assert engines.transfer_engine_refusal(s) is None
