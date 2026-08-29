# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Everything in `primitives/amd.py` is reachable, or is listed as not.

The module had accumulated 350 lines that nothing could call: constant
helpers, intrinsic wrappers, two routines written against CUDA's
`__shfl_xor_sync`, a class with its dispatch tables, and -- twice -- a
second module-level definition of a name that silently replaced the first.
None of it was caught by the tests, because unreachable code cannot fail.

So the property is asserted directly rather than left to review.  Reachability
is computed over the call graph from the single entry point `multilinear.py`
uses, which is a stronger statement than test coverage: coverage says "no case
happened to run this", reachability says "no case can".

The allow-list is the interesting part.  Adding a name to it is a deliberate
act with a reason attached; growing it silently is the failure this guards.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from harness import reachability

AMD = (Path(__file__).parent.parent / "src" / "tensorforge" / "backend" /
       "instructions" / "compute" / "primitives" / "amd")

#: Modules of the package, in dependency order.  Reachability is computed over
#: all of them at once: a name that is unreachable only because it sits in
#: another file is still unreachable, and splitting a module must not be a way
#: to launder dead code past this check.
MODULES = ["__init__", "arch", "caps", "features", "catalog", "layouts",
           "relayout", "select", "emitters", "codegen", "unused"]

ENTRY = "matmul"

# Unreachable on purpose.  Each entry needs a reason that says why deleting it
# would be worse than keeping it.
#
# The relayout table's lookup half used to be here too, until `hfma` started
# reaching it through `find_relayout`: `matmul` -> `hfma` -> `find_relayout`
# -> `RELAYOUTS` -> every row. Nothing in the table is unreachable now.
#
# `vega7nm` used to be here, on the claim that it was the missing `fmacdpp4`
# guard.  It was not: it excludes gfx900, which is what made it look right,
# but it also excludes all of RDNA, where the instruction does exist.  Using
# it would have turned a link error on one target into silently slower code on
# five.  The capability predicates in `amd.py` replaced it and it was deleted.
KEPT_UNREACHABLE = {
    "mfma_emu_int8":
        "matrix path, to be repaired rather than rewritten",
    "mfma_emu_bf16_f32":
        "matrix path, to be repaired rather than rewritten",
    "mfma_emu_f16_f32":
        "matrix path, to be repaired rather than rewritten",
    "wmma3atom":
        "matrix path, to be repaired rather than rewritten",
    # The catalogue describes every float matrix instruction; `matmul()` still
    # selects from the three K=1 F32 tiles through `usable_mfma_tiles`, so the
    # general query and the split arithmetic have no call site yet. They lose
    # their entry here when the emitter that consumes them lands -- which is
    # what `test_allow_list_does_not_outlive_its_entries` enforces.
    "ops_for":
        "catalogue query; the emitter that selects from it is not written",
    "_place":
        "the table's decoder; reached only from `position`",
    "position":
        "fragment placement; the emitter that stages an operand into one is "
        "not written",
    "covers":
        "guards `position`; same call site, not written yet",
    "lane_batched_ops":
        "the same precondition asked of the whole catalogue; the F32 policy "
        "reaches it through MFMA_TILES instead",
    "split_terms":
        "split-precision arithmetic; no emitter consumes it yet",
    "split_products":
        "split-precision arithmetic; no emitter consumes it yet",
    "NOT_MODELLED":
        "documents the catalogue's boundary; read by the LLVM cross-check",
}


@pytest.fixture(scope="module")
def analysis():
    return reachability.analyse(AMD, MODULES, [ENTRY])


def test_entry_point_exists(analysis):
    defs, _ = analysis
    assert ENTRY in defs, f"{ENTRY}() is what multilinear.py calls"


def test_no_name_is_defined_twice(analysis):
    """A second definition of the same name silently discards the first.

    Both `reduction` and `matmul` had one.  The `matmul` case meant the whole
    MatrixCore dispatch path had been unreachable since it was written -- not
    by design, by name collision.  Across a package the failure is quieter
    still: two modules can each define the name and the `__init__` re-export
    picks whichever it imports last.
    """
    defs, _ = analysis
    dupes = reachability.duplicate_definitions(defs)
    assert not dupes, f"shadowed definitions: {dupes}"


def test_nothing_is_unreachable_without_a_reason(analysis):
    defs, reach = analysis
    unreachable = set(defs) - reach
    undeclared = unreachable - set(KEPT_UNREACHABLE)
    assert not undeclared, (
        f"unreachable from {ENTRY}(): {sorted(undeclared)}. "
        f"Delete it, wire it up, or add it to KEPT_UNREACHABLE with a reason.")


def test_allow_list_does_not_outlive_its_entries(analysis):
    """A name that became reachable, or was deleted, should leave the list."""
    defs, reach = analysis
    stale = {n for n in KEPT_UNREACHABLE if n not in defs or n in reach}
    assert not stale, f"KEPT_UNREACHABLE is out of date for: {sorted(stale)}"


@pytest.mark.parametrize("mod", MODULES)
def test_no_cuda_intrinsics_in_the_amd_package(mod):
    """`__shfl_xor_sync` is CUDA.  Two routines here were written against it."""
    src = reachability.code_only(AMD / f"{mod}.py")
    for token in ("__shfl_xor_sync", "__shfl_sync", "__ballot_sync"):
        assert token not in src, f"{mod}.py: {token} is a CUDA intrinsic"


def test_module_has_no_empty_stubs(analysis):
    """`def f(...): pass` reads as an implemented hook and is not one."""
    defs, _ = analysis
    stubs = reachability.empty_stubs(defs)
    assert not stubs, f"empty stubs: {stubs}"
