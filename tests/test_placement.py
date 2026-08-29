# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""The placement decision, asked directly.

Legality and preference were four booleans set from a vendor string in a
constructor and read in the middle of the code that emits loads.  In that shape
neither half could be asked a question: the only way to find out what a vendor
did was to generate a kernel and count instructions, and the only way to add a
target was to guess which of the four to flip.

These are the two halves stated apart.  The legality tests say what would be
*wrong*; the preference tests say what each vendor picks among answers that are
all correct. A change that moves a preference moves snapshots; one that moves a
legality claim moves whether a kernel is right at all, and only the first kind
should ever be routine.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from tensorforge.backend.placement import (DEFAULT_POLICY, POLICIES, Placement,
                                           ResultPlacement, VendorPolicy,
                                           choose_operand_placement,
                                           choose_result_placement,
                                           legal_operand_placements,
                                           legal_result_placements,
                                           policy_for, result_is_atomic)

NVIDIA = POLICIES["nvidia"]
AMD = POLICIES["amd"]
INTEL = POLICIES["intel"]


class _Hw:
    def __init__(self, vendor):
        self.vendor = vendor


def _legal(policy, *, addressable=True, transposed=False,
           carries_lead_dim=True):
    return legal_operand_placements(addressable=addressable,
                                    transposed=transposed,
                                    carries_lead_dim=carries_lead_dim,
                                    policy=policy)


# ----------------------------------------------------------------------
# legality: what would be wrong
# ----------------------------------------------------------------------

@pytest.mark.parametrize("policy", [NVIDIA, DEFAULT_POLICY],
                         ids=["nvidia", "unnamed"])
def test_a_transposed_operand_can_only_be_staged_through_shared(policy):
    """A register image fixes its lane axis when it is written.

    Reading a transposed operand where it lies would need the address to move
    data between lanes, and an address cannot; a register copy cannot either,
    since it has the same problem one step earlier.  Only a shared buffer can
    reorder.
    """
    assert _legal(policy, transposed=True) == frozenset({Placement.SHARED})


@pytest.mark.parametrize("policy", [NVIDIA, DEFAULT_POLICY],
                         ids=["nvidia", "unnamed"])
def test_an_operand_without_the_lead_index_can_only_be_staged(policy):
    """Same problem from the other side: there is no lane axis to spread."""
    assert _legal(policy, carries_lead_dim=False) \
        == frozenset({Placement.SHARED})


@pytest.mark.parametrize("policy", [AMD, INTEL], ids=["amd", "intel"])
def test_a_hardware_broadcast_makes_the_missing_lane_axis_legal_in_place(
        policy):
    """Where a broadcast is cheap, every lane can read the same element.

    This is the one legality claim that is hardware-dependent, and it is the
    reason it lives in the policy rather than next to the others: the fact
    being asserted is about the machine, not about the data.
    """
    assert Placement.IN_PLACE in _legal(policy, transposed=True)
    assert Placement.IN_PLACE in _legal(policy, carries_lead_dim=False)


def test_the_broadcast_claim_is_dropped_under_explicit_simd():
    """The one field the vendor does not decide on its own.

    A broadcast needs a value whose spread over the lanes is known.  The SPMD
    lowering carries that in the index expression; the explicit-SIMD one
    carries it in the type, where a value read once and used across the whole
    vector has no spread to give -- the emitter says so outright, refusing the
    declaration for want of a tracked distribution.

    So the same Intel hardware admits the placement under one lowering and not
    under the other, and a table keyed on the vendor alone gets it wrong for
    half the targets: setting it for Intel outright fixed ten kernels under
    SYCL and broke twenty-five under ESIMD.
    """
    hw = _Hw("intel")
    assert policy_for(hw).broadcast_without_staging
    assert not policy_for(hw, explicit_simd=True).broadcast_without_staging


@pytest.mark.parametrize("name", sorted(POLICIES))
def test_the_lowering_gate_subtracts_and_nothing_else(name):
    """It is a property of the lowering, not of any one vendor's row.

    Applied to every row that carries the claim, and it removes that field and
    leaves the rest, so a target that gains an explicit-SIMD lowering later
    does not also silently lose its register staging.
    """
    hw = _Hw(name)
    spmd = policy_for(hw)
    simd = policy_for(hw, explicit_simd=True)

    assert not simd.broadcast_without_staging
    assert replace(spmd, broadcast_without_staging=False) == simd


@pytest.mark.parametrize("policy", [NVIDIA, AMD, INTEL, DEFAULT_POLICY],
                         ids=["nvidia", "amd", "intel", "unnamed"])
def test_an_unaddressable_operand_can_only_be_read_in_place(policy):
    """A scalar has nothing to stage, whatever the vendor prefers."""
    assert _legal(policy, addressable=False) \
        == frozenset({Placement.IN_PLACE})


def test_a_destination_written_in_slices_cannot_stay_in_registers():
    """The residency holds one entry per name.

    A second slice displaces the first, and the first slice's whole
    contribution is computed and then thrown away.
    """
    assert legal_result_placements(written_in_slices=True) \
        == frozenset({ResultPlacement.MEMORY})


# ----------------------------------------------------------------------
# preference: what each vendor picks among correct answers
# ----------------------------------------------------------------------

@pytest.mark.parametrize("policy,expected", [
    (NVIDIA, Placement.REGISTER),
    (AMD, Placement.REGISTER),
    (INTEL, Placement.REGISTER),
    (DEFAULT_POLICY, Placement.IN_PLACE),
], ids=["nvidia", "amd", "intel", "unnamed"])
def test_what_each_vendor_does_with_a_plain_operand(policy, expected):
    """A vendor with no row reads global memory on every iteration.

    Correct and slow, and it was every vendor but two: Intel read a 16x16 GEMM
    out of global memory 263 times in its loop body against 5 on NVIDIA, purely
    because the flags named a list rather than describing a machine.
    """
    assert choose_operand_placement(_legal(policy), policy) is expected


def test_preference_never_overrides_legality():
    """A vendor that prefers registers still stages a transposed operand."""
    legal = _legal(NVIDIA, transposed=True)
    assert choose_operand_placement(legal, NVIDIA) is Placement.SHARED


def test_an_unclaimed_alternative_is_named_rather_than_dead():
    """No vendor preloads operands into shared memory today.

    The flag was `vendor in []`, which reads as a mistake and is not one: it is
    an alternative nobody currently picks. Stated as a table field it stays
    reachable by writing a row, instead of by editing a branch.
    """
    assert not any(p.preload_operands_into_shared for p in POLICIES.values())
    shared_first = VendorPolicy(preload_operands_into_shared=True)
    assert choose_operand_placement(_legal(shared_first), shared_first) \
        is Placement.SHARED


# ----------------------------------------------------------------------
# where and how are two decisions
# ----------------------------------------------------------------------

def test_a_deferred_result_can_still_be_atomic():
    """The distinction that a three-valued placement collapses.

    An accumulation on AMD is kept in its register array *and* reaches memory
    as an atomic add: the update goes out at the section boundary rather than
    here, and is still an add rather than an overwrite. Folding atomicity into
    the placement enum turns that store into a plain one, which silently makes
    the accumulation an assignment.
    """
    atomic = result_is_atomic(accumulating=True, pending_is_atomic=True,
                              policy=AMD)
    assert atomic
    assert choose_result_placement(legal_result_placements(
        written_in_slices=False), atomic=atomic, policy=AMD) \
        is ResultPlacement.REGISTER


def test_an_atomic_result_written_in_slices_goes_out_now():
    """Deferring an atomic is what makes it collide with the next slice."""
    atomic = result_is_atomic(accumulating=True, pending_is_atomic=True,
                              policy=AMD)
    assert choose_result_placement(legal_result_placements(
        written_in_slices=True), atomic=atomic, policy=AMD) \
        is ResultPlacement.MEMORY


def test_an_atomic_needs_nothing_non_atomic_already_pending():
    """Mixing the two lets a plain store overwrite an update."""
    assert not result_is_atomic(accumulating=True, pending_is_atomic=False,
                                policy=AMD)


def test_only_an_accumulation_is_atomic():
    assert not result_is_atomic(accumulating=False, pending_is_atomic=True,
                                policy=AMD)


@pytest.mark.parametrize("policy", [NVIDIA, INTEL, DEFAULT_POLICY],
                         ids=["nvidia", "intel", "unnamed"])
def test_a_vendor_without_atomics_never_gets_one(policy):
    assert not result_is_atomic(accumulating=True, pending_is_atomic=True,
                                policy=policy)
