# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""The broadcast chain, and the claim that it belongs to no vendor.

It sat in `primitives/intel.py` and its own docstring said what it was: the
same shape as the AMD DPP path, differing in what the replication costs rather
than in what it computes.  A shared arrangement kept inside one target's module
is one no other target can reach without importing that module, which is the
thing these check has stopped being true.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

from tensorforge.backend.instructions.compute import broadcast
from tensorforge.backend.instructions.compute.matmul import MatmulOperands
from tensorforge.backend.instructions.compute.primitives import amd, intel
from tensorforge.backend.instructions.compute.strategy import (
    PREFERENCES, ComputeShape, Span, Strategy, choose_strategy,
    legal_strategies)
from tensorforge.backend.pir.build import IRBuilder
from tensorforge.common.basic_types import Datatype
from tensorforge.common.context import Context

from test_amd_tiling import _Recorder, _operand


SOURCE = pathlib.Path(broadcast.__file__)


def _ops(recorder, n, threads=32, lead_slots=2, sparse=None):
    return MatmulOperands(A=_operand, B=_operand, C=recorder, sparse=sparse,
                          lead_slots=lead_slots,
                          lead_elements=lead_slots * threads,
                          n=n, k=8, kx=0, threads=threads,
                          dtype=Datatype.F32)


# -- it names no target ---------------------------------------------------- #

def test_the_chain_imports_no_vendor_module():
    """The property the move was for.  A shared arrangement that reaches into
    one target's package is shared in name only: the next target to want it
    inherits that target's gates along with the emitter."""
    tree = ast.parse(SOURCE.read_text())
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
        elif isinstance(node, ast.Import):
            imported += [alias.name for alias in node.names]
    assert not [m for m in imported if 'primitives' in m], imported


@pytest.mark.parametrize('vendor', ['amd', 'intel'])
def test_neither_target_keeps_its_own_copy(vendor):
    """Two emitters for one arrangement is how they drift: the AMD catalogue
    orders its split terms smallest-first and the DPAS path orders them
    largest-first, which is exactly the disagreement two copies produce."""
    module = {'amd': amd, 'intel': intel}[vendor]
    assert not hasattr(module, 'broadcast_matmul')
    assert Strategy.BROADCAST in PREFERENCES[vendor]


# -- what it computes ------------------------------------------------------ #

@pytest.fixture(scope='module')
def hip():
    """A real context: `lane_broadcast` reaches the lexic under SPMD, so a
    stub that answers only the architecture predicates is not enough."""
    return Context(arch='gfx90a', backend='hip', fp_type=Datatype.F32)


@pytest.mark.parametrize('n', [1, 2, 5, 7, 8, 16, 23])
def test_it_covers_what_the_dpp_chain_covers(hip, n):
    """The two arrangements differ in where the replication happens and in
    nothing else, so the set of products they emit has to match.  Asserted
    against the DPP chain rather than against a recomputed expectation,
    because the chain is the thing the AMD offer is claiming to be an
    alternative to."""
    def run(strategy):
        rec = _Recorder()
        amd.matmul(IRBuilder(Datatype.F32, context=hip), _ops(rec, n), hip,
                   Span(strategy, 0, n))
        return rec

    chain, dpp = run(Strategy.BROADCAST), run(Strategy.DPP)
    assert chain.covered == dpp.covered
    assert not chain.duplicates


def test_a_span_bounds_what_it_writes(hip):
    """It takes a range where the DPAS path does not, so a plan is free to
    give it part of the output."""
    rec = _Recorder()
    broadcast.matmul(IRBuilder(Datatype.F32, context=hip), _ops(rec, 8), hip,
                     Span(Strategy.BROADCAST, 3, 6))
    assert {j for _, _, j in rec.stores} == {3, 4, 5}


def test_it_declines_a_sparse_operand():
    """B is reached through its linear index there, and the contraction here
    walks B's lanes; there is no lane to replicate."""
    ops = _ops(None, 4, sparse=lambda k, j: True)
    assert broadcast.matmul(None, ops, None,
                            Span(Strategy.BROADCAST, 0, 4)) is False


# -- where it sits in the order -------------------------------------------- #

def test_amd_prefers_the_fused_broadcast(hip):
    """DPP folds the replication into the multiply as a modifier, so it costs
    no instruction of its own.  Both are offered; the ranking is what says
    which is taken, and turning it around is a measurement rather than an
    edit to a gate."""
    shape = ComputeShape(threads=32, dtype=Datatype.F64, sparse=False,
                         explicit_simd=False)
    offered = amd.strategies(shape, hip)
    assert {Strategy.DPP, Strategy.BROADCAST} <= offered
    assert choose_strategy(legal_strategies(offered), 'amd') is Strategy.DPP


def test_amd_offers_neither_chain_a_sparse_operand_it_cannot_read(hip):
    shape = ComputeShape(threads=32, dtype=Datatype.F32, sparse=True,
                         explicit_simd=False)
    assert amd.strategies(shape, hip) == frozenset({Strategy.DPP})
