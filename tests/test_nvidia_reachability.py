# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Everything in `primitives/nvidia.py` is reachable, or is listed as not.

The same guard as `test_amd_reachability.py`, on the same machinery, for the
same reason -- but the module it guards had a sharper version of the problem.
`nvidia.py` was never reached at all: `_is_matmul` asked
`vendor in ['amd']`, so the `elif vendor == 'nvidia'` branch below it could
not run, and had not been able to since the line was written.  Nothing in the
file was covered, so nothing in it could fail, so 23 definitions accumulated
that no case could reach.

Two of those were second module-level definitions of a name already taken.
`matmul` was one, and the shadowed twin was the broken one: it declared
`{Areg}[]{}`, an array of size zero, and called `atom.generate(..., [], [],
[])` with empty operand lists.  The working emitter survived only because
Python keeps the *last* definition.  That is the specific reason to delete
duplicates before turning a path on rather than after: with the path live and
the twins still present, "which one runs" is decided by file order.

The rest -- `reduction_generic`, `full_reduction`, `ballot_reduction`,
`minmaxfloatint`, the four `shuffle_*` helpers, `atomic`, `read_shared`,
`CUTEAtom`, `ATOMS`, `MatmulCall`, `MMAWrapper`, `bfconvert`,
`prefer_rowload` -- was deleted rather than kept, because none of it was
repairable in the sense the AMD `unused.py` entries are.  These do not carry
a known defect to fix; they reference names that do not exist at all.
`reduction_generic` uses `value` and `dtype`, `full_reduction` uses `ARCH`,
`sm80` and `Operation`, and calls itself with one argument where it takes
five.  That is not code with bugs in it, it is code that was never run once.
`bfconvert` opens with `raise NotImplementedError()` and then has a body.

So the allow-list here is empty, and that is the point: an entry would mean
someone decided a specific thing is worth keeping unreachable, with a reason
attached.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from harness import reachability

PRIMITIVES = (Path(__file__).parent.parent / "tensorforge" / "backend" /
              "instructions" / "compute" / "primitives")

MODULES = ["nvidia"]

#: What `multilinear.py` reads.  `supports` is an entry point in its own
#: right, not something `matmul` reaches -- the gate is asked *before* the
#: emitter, which is the whole change that made this file live -- and
#: `ENABLED` likewise: it is a module-level constant the caller consults, and
#: without it here the deployment switch reads as dead.
ENTRIES = ["matmul", "supports", "shmsize", "ENABLED"]

# Unreachable on purpose.  Each entry would need a reason that says why
# deleting it would be worse than keeping it.  There are none.
KEPT_UNREACHABLE: dict = {}


@pytest.fixture(scope="module")
def analysis():
    return reachability.analyse(PRIMITIVES, MODULES, ENTRIES)


def test_entry_points_exist(analysis):
    defs, _ = analysis
    missing = [e for e in ENTRIES if e not in defs]
    assert not missing, f"multilinear.py calls these: {missing}"


def test_no_name_is_defined_twice(analysis):
    """A second definition of the same name silently discards the first.

    Both `reduction` and `matmul` had one here.  The `matmul` case is the one
    that matters: the twin that lost is the one with the zero-size array, so
    the file worked by accident of ordering.
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

    Nothing here uses one today; the check exists because the AMD package did
    acquire two routines written against `__shfl_xor_sync`, and there is no
    reason the traffic cannot go the other way.
    """
    src = reachability.code_only(PRIMITIVES / f"{mod}.py")
    for token in ("__builtin_amdgcn_", "fmacdpp", "transpose4x4b32",
                  "__ockl_", "s_barrier"):
        assert token not in src, f"{mod}.py: {token} is an AMD intrinsic"


def test_module_has_no_empty_stubs(analysis):
    """`def f(...): pass` reads as an implemented hook and is not one.

    `atomic` and `read_shared` were exactly that.
    """
    defs, _ = analysis
    assert not reachability.empty_stubs(defs)


def test_the_atom_is_named_once(analysis):
    """`INSTRS[1]` had three separate copies: `shmsize`, the CUTE `matmul` and
    the live one.  A shared-memory reservation computed from one atom and an
    emitter using another is a silent overflow, so the choice is `ATOM`."""
    import ast

    tree = ast.parse((PRIMITIVES / "nvidia.py").read_text())
    subscripts = [n for n in ast.walk(tree)
                  if isinstance(n, ast.Subscript)
                  and isinstance(n.value, ast.Name) and n.value.id == "INSTRS"]
    assert len(subscripts) <= 1, (
        f"INSTRS is subscripted at lines {[n.lineno for n in subscripts]}; "
        f"the selected atom is named once, as ATOM")
