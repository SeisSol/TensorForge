# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Everything in `primitives/nvidia.py` is reachable, or is listed as not.

The same guard as `test_amd_reachability.py`, on the same machinery, for the
same reason, and with the whole module at stake: a dispatch that never
selected it would leave nothing in it covered, nothing in it able to fail,
and definitions that no case can reach free to accumulate unseen.

The sharpest form of that is a second module-level definition of a name
already taken.  Python keeps the *last* definition, so where a working
emitter and a broken twin share a name, the working one runs only because it
comes second.  That is the specific reason to delete duplicates before
turning a path on rather than after: with the path live and the twins
present, "which one runs" is decided by file order.

An unreachable definition that references names which do not exist at all is
not repairable in the sense the AMD `unused.py` entries are.  It carries no
known defect to fix; it is code that has never run once, and it is deleted
rather than kept.

So the allow-list here is empty, and that is the point: an entry would mean
someone decided a specific thing is worth keeping unreachable, with a reason
attached.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from harness import reachability

PRIMITIVES = (Path(__file__).parent.parent / "src" / "tensorforge" / "backend" /
              "instructions" / "compute" / "primitives")

MODULES = ["nvidia"]

#: What `multilinear.py` reads.  `supports` is an entry point in its own
#: right, not something `matmul` reaches -- the gate is asked *before* the
#: emitter -- and `ENABLED` likewise: it is a module-level constant the
#: caller consults, and without it here the deployment switch reads as dead.
#: `prepared_order` is asked at a different time from all of them: before the
#: operand's buffer is sized, because the answer decides how big it is.
#: `fragment_order` hangs off it and is reached rather than listed.
#: `convergence` asks whether the plan needs the multiplications of a warp in
#: step, while the batch loop is built -- before any body, and of the target
#: rather than the plan, since `mma.sync` is `.aligned` and another target's
#: matrix path is not.
ENTRIES = ["matmul", "supports", "strategies", "plan", "ENABLED",
           "prepared_order", "convergence"]

# Unreachable on purpose.  Each entry would need a reason that says why
# deleting it would be worse than keeping it.  There are none.
KEPT_UNREACHABLE: dict = {}


@pytest.fixture(scope="module")
def analysis():
    return reachability.analyze(PRIMITIVES, MODULES, ENTRIES)


def test_entry_points_exist(analysis):
    defs, _ = analysis
    missing = [e for e in ENTRIES if e not in defs]
    assert not missing, f"multilinear.py calls these: {missing}"


def test_no_name_is_defined_twice(analysis):
    """A second definition of the same name silently discards the first.

    Where the twin that loses is the working one, the file breaks; where it
    is the broken one, the file works by accident of ordering.
    """
    defs, _ = analysis
    dupes = reachability.duplicate_definitions(defs)
    assert not dupes, f"shadowed definitions: {dupes}"


def test_nothing_is_unreachable_without_a_reason(analysis):
    defs, reach = analysis
    undeclared = (set(defs) - reach) - set(KEPT_UNREACHABLE)
    assert not undeclared, (
        f"unreachable from {ENTRIES}: {sorted(undeclared)}. "
        f"Delete it, wire it up, or add it to KEPT_UNREACHABLE with a reason.")


def test_allow_list_does_not_outlive_its_entries(analysis):
    defs, reach = analysis
    stale = {n for n in KEPT_UNREACHABLE if n not in defs or n in reach}
    assert not stale, f"KEPT_UNREACHABLE is out of date for: {sorted(stale)}"


@pytest.mark.parametrize("mod", MODULES)
def test_no_amd_intrinsics_in_the_nvidia_module(mod):
    """The mirror of the CUDA-intrinsic check on the AMD side.

    Nothing here uses one today.  A routine written against one vendor's
    intrinsics can land in the other vendor's package, and nothing makes that
    traffic go only one way.
    """
    src = reachability.code_only(PRIMITIVES / f"{mod}.py")
    for token in ("__builtin_amdgcn_", "fmacdpp", "transpose4x4b32",
                  "__ockl_", "s_barrier"):
        assert token not in src, f"{mod}.py: {token} is an AMD intrinsic"


def test_module_has_no_empty_stubs(analysis):
    """`def f(...): pass` reads as an implemented hook and is not one."""
    defs, _ = analysis
    assert not reachability.empty_stubs(defs)
