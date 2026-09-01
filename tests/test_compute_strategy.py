# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""The compute strategy, asserted from numbers rather than from output.

Which arrangement a contraction is emitted as was four conditions in three
forms spread over the dispatch and the vendor modules, and the only way to ask
what it had decided was to read the generated code.  Written as a function of
the shape it can be asked directly -- which is what these do.

The split is the same one `placement.py` makes.  Legality is about what the
target and the shape allow; preference is about which of the legal answers is
worth taking.  A test that mixes them passes for the wrong reason as soon as
the order changes.
"""

from __future__ import annotations

import pytest

from tensorforge.backend.instructions.compute.strategy import (
    DEFAULT_PREFERENCE, PREFERENCES, ComputeShape, Strategy, choose_strategy,
    is_contraction, legal_strategies)
from tensorforge.backend.instructions.compute.primitives import amd, intel
from tensorforge.backend.instructions.compute.primitives import nvidia
from tensorforge.common.basic_types import Datatype


class _FakeCtx:
    """Enough context for the AMD predicates, which read the model name."""

    class _HW:
        def __init__(self, model):
            self.model = model
            self.vendor = 'amd'

    class _VM:
        def __init__(self, model):
            self._hw = _FakeCtx._HW(model)

        def get_hw_descr(self):
            return self._hw

    def __init__(self, model='gfx90a'):
        self._vm = _FakeCtx._VM(model)

    def get_vm(self):
        return self._vm


def _shape(threads=32, dtype=Datatype.F32, sparse=False, explicit_simd=False):
    return ComputeShape(threads=threads, dtype=dtype, sparse=sparse,
                        explicit_simd=explicit_simd)


# -- shape-independent legality -------------------------------------------- #

def test_three_operands_have_no_a_and_b():
    """Every arrangement but the nest names two operands; a longer product
    has no such split to name."""
    assert not is_contraction(operands=3, lead_width=1)


def test_a_widened_lead_dimension_excludes_every_arrangement():
    """The matrix cores own the lane-to-register mapping their fragments use
    and the broadcast chains index the lanes directly; a blocked lead
    distribution is a change to exactly that.  Composing the two is silent --
    right registers, wrong places -- so it is excluded rather than ranked."""
    assert not is_contraction(operands=2, lead_width=2)
    assert is_contraction(operands=2, lead_width=1)


def test_the_nest_is_always_legal():
    """Which is why every other arrangement is free to decline."""
    assert Strategy.GENERIC in legal_strategies(frozenset())
    assert Strategy.GENERIC in legal_strategies({Strategy.MATRIX})


# -- what each target offers ----------------------------------------------- #

def test_amd_always_offers_the_dpp_chain():
    """A broadcast modifier on the multiply asks nothing of the shape, and
    where the widest form does not link `select.py` narrows rather than
    declines."""
    for dtype in (Datatype.F32, Datatype.F64):
        offered = amd.strategies(_shape(dtype=dtype), _FakeCtx())
        assert Strategy.DPP in offered


def test_amd_offers_a_matrix_core_only_where_a_tile_fits():
    """F64's MFMAs spend two lane bits on the contraction, so the lane-batched
    loop cannot feed them.  The offer follows that structural fact rather than
    a type name -- which is why the F64 answer needs no special case."""
    ctx = _FakeCtx('gfx90a')
    assert Strategy.MATRIX in amd.strategies(_shape(dtype=Datatype.F32), ctx)
    assert Strategy.MATRIX not in amd.strategies(_shape(dtype=Datatype.F64),
                                                 ctx)


def test_amd_declines_a_matrix_core_for_a_sparse_operand():
    """Read by linear index, which no fragment layout accepts.  The DPP chain
    has a branch for it and stays on offer."""
    ctx = _FakeCtx()
    offered = amd.strategies(_shape(sparse=True), ctx)
    assert offered == frozenset({Strategy.DPP})


def test_nvidia_offers_nothing_while_the_switch_is_off():
    """`ENABLED` is a deployment switch, not a capability, and it reads as one
    here: flipping it is the whole of turning the path on."""
    assert not nvidia.ENABLED
    assert nvidia.strategies(_shape(threads=32), None) == frozenset()


def test_intel_offers_the_broadcast_chain_only_under_explicit_simd():
    """A lane broadcast is an element read out of the work-item's own vector
    there and a real cross-lane instruction in SPMD.  The arrangement is built
    entirely out of that operation, so the lowering decides whether it is
    worth offering at all."""
    assert intel.BROADCAST_ENABLED
    simd = intel.strategies(_shape(threads=16, explicit_simd=True), None)
    spmd = intel.strategies(_shape(threads=16, explicit_simd=False), None)
    assert Strategy.BROADCAST in simd
    assert Strategy.BROADCAST not in spmd


def test_intel_offers_nothing_for_a_wave_that_is_not_the_execution_size():
    """`ExecutionSize` is 16 for every atom in the table, and the ESIMD
    lowering makes the vector width the thread count."""
    assert intel.strategies(_shape(threads=32, explicit_simd=True),
                            None) == frozenset()


# -- preference ------------------------------------------------------------ #

def test_a_target_with_no_row_runs_the_nest():
    """Correct and slow, which is the right default: adding a row is the whole
    of enabling a target, and forgetting to costs performance, not answers."""
    assert DEFAULT_PREFERENCE == (Strategy.GENERIC,)
    assert choose_strategy(frozenset({Strategy.MATRIX, Strategy.GENERIC}),
                           'moore') is Strategy.GENERIC


def test_amd_prefers_the_matrix_core_to_the_dpp_chain():
    legal = legal_strategies({Strategy.MATRIX, Strategy.DPP})
    assert choose_strategy(legal, 'amd') is Strategy.MATRIX


def test_amd_falls_to_dpp_when_no_tile_fits():
    legal = legal_strategies({Strategy.DPP})
    assert choose_strategy(legal, 'amd') is Strategy.DPP


def test_preference_never_invents_an_arrangement():
    """Ranking only chooses among what is already legal.  A row naming an
    arrangement the shape excluded has to be skipped, not taken -- otherwise
    the preference table can overrule a correctness condition."""
    for vendor, order in PREFERENCES.items():
        for strategy in order:
            legal = legal_strategies(frozenset())
            chosen = choose_strategy(legal, vendor)
            assert chosen is Strategy.GENERIC, (vendor, strategy)


def test_every_preferred_arrangement_can_be_emitted():
    """A row that names an arrangement no module in that vendor's package
    emits would be chosen and then have nothing to do."""
    modules = {'amd': amd, 'nvidia': nvidia, 'intel': intel}
    for vendor, order in PREFERENCES.items():
        assert Strategy.GENERIC in order, vendor
        assert vendor in modules, vendor
    assert set(PREFERENCES) == set(modules)


@pytest.mark.parametrize('vendor', sorted(PREFERENCES))
def test_the_nest_is_last_in_every_row(vendor):
    """It is legal for every shape, so anything after it is unreachable."""
    order = PREFERENCES[vendor]
    assert order[-1] is Strategy.GENERIC
    assert order.count(Strategy.GENERIC) == 1
