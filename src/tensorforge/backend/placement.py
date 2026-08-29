# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Where an operand is read from, and where a result is kept.

Two questions that look like one and are not.

*Legality* is what the operation and the data allow.  An operand whose lane
axis is not the one it is stored along cannot be read where it lies: the
address would have to transpose across lanes, which an address cannot do.  A
result assembled from several partial writes cannot stay in an accumulator,
because each operation holds only its own part.  These are facts; getting one
wrong produces a kernel that is wrong.

*Preference* is what the hardware makes worthwhile among the legal answers.
Staging an operand into registers pays on one vendor and not on another;
deferring a store pays when it saves a read-modify-write and costs when it
serialises against the next slice.  Getting one wrong produces a kernel that is
slower.

They were interleaved, as five booleans set in a constructor from a vendor
string and consulted in the middle of the code that emits loads.  Written out,
the five say less than they look like: two of them carry the same list, one is
empty, and every vendor not named reads every operand out of global memory on
every iteration -- which is why a 16x16 GEMM on Intel had 263 global reads in
its loop body against 5 on NVIDIA.

Keeping preference in a table is what makes it replaceable.  The eventual
producer of these decisions is a pass with the shared-memory budget and the
register pressure in hand, or the PIR once staging arrays are values with a
memory space rather than symbols with a baked-in address.  Either way what it
produces is one of these, and the code that emits loads does not have to know
which of them asked.
"""

from dataclasses import dataclass, replace
from enum import Enum
from typing import FrozenSet


class Placement(Enum):
    """Where an operand is read from."""

    #: Read where it already is: global memory, a shared buffer, or a register
    #: image an earlier operation left behind.
    IN_PLACE = 'in_place'
    #: Copy into a shared buffer first.  The only answer that can reorder,
    #: so the only one available to a transposed operand.
    SHARED = 'shared'
    #: Copy into a register array first.  Cheapest to read and the least
    #: flexible: the lane axis is fixed when the image is written.
    REGISTER = 'register'


class ResultPlacement(Enum):
    """Where a result is kept once the operation has computed it.

    *Where* it is kept and *how* it is written are two decisions, not one.  A
    deferred store can itself be atomic: the update goes out at the section
    boundary rather than here, and is still an add rather than an overwrite.
    Modelling atomicity as a third placement collapses that and quietly turns
    an accumulation into an assignment.
    """

    #: Leave it in the accumulator and write it out when someone needs it.
    #: Saves the read-modify-write of an accumulation chain.
    REGISTER = 'register'
    #: Write it out now.
    MEMORY = 'memory'


@dataclass(frozen=True)
class VendorPolicy:
    """One row of the preference table.

    Nothing here is about correctness: every field picks among answers that
    are already legal.
    """

    #: Copy an operand into registers rather than reading global memory on
    #: every iteration.
    preload_operands_into_registers: bool = False
    #: The same for a shared buffer.  No vendor takes it today; it stays as a
    #: named alternative rather than a dead branch.
    preload_operands_into_shared: bool = False
    #: Keep a result in its accumulator instead of storing it eagerly.
    keep_results_in_registers: bool = False
    #: Accumulate into global memory atomically, which lets a store go out on
    #: its own instead of waiting for the tensor's other writers.
    atomic_accumulation: bool = False
    #: A cross-lane broadcast is cheap enough that an operand missing the lane
    #: axis can be read in place instead of staged.
    broadcast_without_staging: bool = False


#: What each vendor prefers.  A vendor with no row reads everything in place,
#: which is correct and slow; adding a row is the whole of enabling a target.
POLICIES = {
    'nvidia': VendorPolicy(preload_operands_into_registers=True,
                           keep_results_in_registers=True),
    'amd': VendorPolicy(preload_operands_into_registers=True,
                        keep_results_in_registers=True,
                        atomic_accumulation=True,
                        broadcast_without_staging=True),
    # The sub-group broadcast is cheap enough here that an operand whose lane
    # axis is not where a reader expects it can be read in place.  Dropped
    # again under the explicit-SIMD lowering by `policy_for`, which is where
    # the reason for that lives.
    'intel': VendorPolicy(preload_operands_into_registers=True,
                          keep_results_in_registers=True,
                          broadcast_without_staging=True),
}

DEFAULT_POLICY = VendorPolicy()


def policy_for(hw, explicit_simd: bool = False) -> VendorPolicy:
    """This machine's row, with anything the lowering cannot express removed.

    The table is keyed by vendor, and one field is not a fact about the vendor.
    A cross-lane broadcast needs a value whose distribution over the lanes is
    known; the SPMD lowering carries that in the index expression, and the
    explicit-SIMD one carries it in the type -- where a value read once and
    used across the whole vector has no distribution to give.  So the same
    Intel hardware admits reading a mis-oriented operand in place under one
    lowering and not under the other, and asking the vendor alone gets it wrong
    for half the targets.

    Dropped rather than made a separate row, because it is a subtraction: the
    lowering cannot express the answer, so the answer is not available, whatever
    the hardware could do.
    """
    policy = POLICIES.get(hw.vendor, DEFAULT_POLICY)
    if explicit_simd and policy.broadcast_without_staging:
        policy = replace(policy, broadcast_without_staging=False)
    return policy


# -- legality ------------------------------------------------------------- #

def legal_operand_placements(*,
                             addressable: bool,
                             transposed: bool,
                             carries_lead_dim: bool,
                             policy: VendorPolicy) -> FrozenSet[Placement]:
    """Which placements would produce a correct read of this operand.

    An operand that is not addressable -- a scalar, or a tensor with no
    addressing -- has nothing to stage and can only be read where it is.

    An operand that is transposed, or that does not carry the destination's
    lead index at all, has its lane axis somewhere other than where a reader
    expects it.  Reading it in place would need the address to move data
    between lanes; only a copy can do that, and only a copy through shared
    memory, since a register image fixes its lane axis when it is written.
    The exception is a hardware broadcast: where one is cheap, every lane can
    read the same element and the missing lane axis stops mattering.
    """
    if not addressable:
        return frozenset({Placement.IN_PLACE})
    if (transposed or not carries_lead_dim) \
            and not policy.broadcast_without_staging:
        return frozenset({Placement.SHARED})
    return frozenset({Placement.IN_PLACE, Placement.SHARED,
                      Placement.REGISTER})


def choose_operand_placement(legal: FrozenSet[Placement],
                             policy: VendorPolicy) -> Placement:
    """The best of the legal answers for this hardware."""
    if len(legal) == 1:
        return next(iter(legal))
    if policy.preload_operands_into_registers and Placement.REGISTER in legal:
        return Placement.REGISTER
    if policy.preload_operands_into_shared and Placement.SHARED in legal:
        return Placement.SHARED
    return Placement.IN_PLACE


def result_is_atomic(*, accumulating: bool, pending_is_atomic: bool,
                     policy: VendorPolicy) -> bool:
    """Whether this result reaches memory as an atomic update.

    Only an accumulation can be one, and only while nothing non-atomic is
    already pending for the same tensor: mixing the two lets a plain store
    overwrite an update instead of adding to it.
    """
    return policy.atomic_accumulation and accumulating and pending_is_atomic


def legal_result_placements(*, written_in_slices: bool
                            ) -> FrozenSet[ResultPlacement]:
    """Which placements would produce a correct result.

    A destination assembled from several partial writes cannot stay in an
    accumulator: the residency holds one entry per name, so a second slice
    would displace the first and its whole contribution would be computed and
    thrown away.  Being atomic does not change that -- an atomic that is
    deferred is exactly one that collides with the next slice -- so it has to
    go out here, which it can afford to, being order-independent.
    """
    if written_in_slices:
        return frozenset({ResultPlacement.MEMORY})
    return frozenset({ResultPlacement.MEMORY, ResultPlacement.REGISTER})


def choose_result_placement(legal: FrozenSet[ResultPlacement], *,
                            atomic: bool,
                            policy: VendorPolicy) -> ResultPlacement:
    """The best of the legal answers for this hardware.

    Keeping a result in registers wins where it is legal, since it saves the
    read-modify-write outright.

    An atomic result defers too, and would even where the policy does not ask
    for deferral in general.  That disjunct is redundant today -- the one
    vendor with atomics also keeps results in registers -- and it is kept
    because the two are separate claims and a table row could separate them.
    """
    if ResultPlacement.REGISTER in legal \
            and (atomic or policy.keep_results_in_registers):
        return ResultPlacement.REGISTER
    return ResultPlacement.MEMORY
