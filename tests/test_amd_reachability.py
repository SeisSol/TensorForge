# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Everything in `primitives/amd/` is reachable, or is listed as not.

Code that nothing can call cannot fail, so no test notices it: a constant
helper, an intrinsic wrapper, a routine written against CUDA's
`__shfl_xor_sync`, a class with its dispatch tables, or a second module-level
definition of a name that silently replaces the first.

So the property is asserted directly rather than left to review.  Reachability
is computed over the call graph from the entry points `multilinear.py` uses,
which is a stronger statement than test coverage: coverage says "no case
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
           "reorder", "relayout", "select", "emitters", "codegen",
           "exchange_codegen", "tiling", "unused"]

#: What the dispatch calls into this package, every one an entry point in the
#: same sense: `matmul` emits, and the others are asked before it -- what the
#: target can emit for a shape, how the arrangement is laid out over the
#: output, what has to be staged, how far the threads run in step, which order
#: the A operand is read in.  Computing reachability from the emitter alone
#: would count them as dead.
ENTRIES = ["matmul", "scratch", "strategies", "plan", "convergence",
           "prepared_order"]

# Unreachable on purpose.  Each entry needs a reason that says why deleting it
# would be worse than keeping it.
#
# The relayout table needs no entry: `matmul` -> `hfma` -> `find_relayout` ->
# `RELAYOUTS` reaches every row.
KEPT_UNREACHABLE = {
    "mfma_emu_int8":
        "matrix path, to be repaired rather than rewritten",
    "mfma_emu_bf16_f32":
        "matrix path, to be repaired rather than rewritten",
    "mfma_emu_f16_f32":
        "matrix path, to be repaired rather than rewritten",
    "wmma3atom":
        "matrix path, to be repaired rather than rewritten",
    # The catalog describes every float matrix instruction; `matmul()` still
    # selects from the three K=1 F32 tiles through `usable_mfma_tiles`, so the
    # general query and the split arithmetic have no call site yet. They lose
    # their entry here when the emitter that consumes them lands -- which is
    # what `test_allow_list_does_not_outlive_its_entries` enforces.
    "_place":
        "the table's decoder; reached only from `position`",
    "established":
        "the strict query an emitter uses to decline a derived layout",
    "_row":
        "measured row or derived one; reached only from `position`",
    "AXES":
        "names the operand index order for `index_terms`",
    "FED_BY":
        "which accessor feeds which fragment; the names collide and the "
        "mapping is not symmetric",
    "accumulator_cost":
        "prices the epilogue against the contraction loop that filled it",
    "fragment_cost":
        "prices a plan against staging or against not taking the path",
    "Term":
        "one shift-and-mask contribution to a fragment address",
    "_bit_sources":
        "reads the table the other way; reached from `element_at` and "
        "`index_terms`",
    "index_terms":
        "fragment addressing for a staging emitter, which is not written",
    "element_at":
        "the inverse of `position`; same call site, not written yet",
    "position":
        "fragment placement; the emitter that stages an operand into one is "
        "not written",
    "lane_batched_ops":
        "the same precondition asked of the whole catalog; the F32 policy "
        "reaches it through MFMA_TILES instead",
    "issues":
        "the axis count, read by the ranking through `ranking.issues` and by "
        "the tests; this wrapper is what a caller holding a `MatrixOp` uses",
    "spare_products":
        "the output-axis predicate the emulated emitter needs once it can "
        "take an entry wider than its k-vector; nothing reads it before then",
    "fragment_layout":
        "the same row `position` reads, in the vocabulary a value can also "
        "be read in; nothing compares the two sides yet, which is what the "
        "relayout solver would do",
    "extracts":
        "what unpacking a packed operand costs; `reach` unpacks before it "
        "answers, but nothing weighs the price yet -- no emitter writes a "
        "route that starts from a packed operand",
    "NOT_MODELED":
        "documents the catalog's boundary; read by the LLVM cross-check",
}


@pytest.fixture(scope="module")
def analysis():
    return reachability.analyze(AMD, MODULES, ENTRIES)


def test_entry_points_exist(analysis):
    defs, _ = analysis
    missing = [e for e in ENTRIES if e not in defs]
    assert not missing, f"{missing} is what multilinear.py calls"


def test_no_name_is_defined_twice(analysis):
    """A second definition of the same name silently discards the first.

    A second `matmul` would leave the first, a whole dispatch path,
    unreachable -- not by design, by name collision.  Across a package the
    failure is quieter still: two modules can each define the name and the
    `__init__` re-export picks whichever it imports last.
    """
    defs, _ = analysis
    dupes = reachability.duplicate_definitions(defs)
    assert not dupes, f"shadowed definitions: {dupes}"


def test_nothing_is_unreachable_without_a_reason(analysis):
    defs, reach = analysis
    unreachable = set(defs) - reach
    undeclared = unreachable - set(KEPT_UNREACHABLE)
    assert not undeclared, (
        f"unreachable from {ENTRIES}: {sorted(undeclared)}. "
        f"Delete it, wire it up, or add it to KEPT_UNREACHABLE with a reason.")


def test_allow_list_does_not_outlive_its_entries(analysis):
    """A name that became reachable, or was deleted, should leave the list."""
    defs, reach = analysis
    stale = {n for n in KEPT_UNREACHABLE if n not in defs or n in reach}
    assert not stale, f"KEPT_UNREACHABLE is out of date for: {sorted(stale)}"


@pytest.mark.parametrize("mod", MODULES)
def test_no_cuda_intrinsics_in_the_amd_package(mod):
    """`__shfl_xor_sync` is CUDA; nothing in the AMD package may use it."""
    src = reachability.code_only(AMD / f"{mod}.py")
    for token in ("__shfl_xor_sync", "__shfl_sync", "__ballot_sync"):
        assert token not in src, f"{mod}.py: {token} is a CUDA intrinsic"


def test_module_has_no_empty_stubs(analysis):
    """`def f(...): pass` reads as an implemented hook and is not one."""
    defs, _ = analysis
    stubs = reachability.empty_stubs(defs)
    assert not stubs, f"empty stubs: {stubs}"
