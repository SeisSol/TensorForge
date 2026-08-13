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

import ast
from collections import defaultdict
from pathlib import Path

import pytest

AMD = (Path(__file__).parent.parent / "tensorforge" / "backend" /
       "instructions" / "compute" / "primitives" / "amd")

#: Modules of the package, in dependency order.  Reachability is computed over
#: all of them at once: a name that is unreachable only because it sits in
#: another file is still unreachable, and splitting a module must not be a way
#: to launder dead code past this check.
MODULES = ["__init__", "arch", "caps", "catalog", "relayout", "select",
           "emitters", "codegen", "unused"]

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
}


def _sources():
    for name in MODULES:
        p = AMD / f"{name}.py"
        assert p.exists(), f"{p} is missing; MODULES is out of date"
        yield name, p.read_text()


def _module_defs(tree):
    """Module-level definitions, keyed by name, newest last."""
    defs = defaultdict(list)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)):
            defs[node.name].append(node)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                # `__all__` is the export list, not a definition -- it names
                # everything by construction, so treating it as one would make
                # the reachability check vacuous.
                if isinstance(t, ast.Name) and t.id != "__all__":
                    defs[t.id].append(node)
    return defs


def _reachable(defs, entry):
    live = {n: v[-1] for n, v in defs.items()}
    seen, work = set(), [entry]
    while work:
        name = work.pop()
        if name in seen or name not in live:
            continue
        seen.add(name)
        for n in ast.walk(live[name]):
            if isinstance(n, ast.Name) and n.id in live and n.id not in seen:
                work.append(n.id)
    return seen


@pytest.fixture(scope="module")
def analysis():
    """Definitions and reachability across the whole package.

    Cross-module `from .x import y` lines are not treated as definitions --
    they are edges, and the definition they point at is already in the merged
    table. Counting them would make every re-export look like a second
    definition of the same name.
    """
    defs = defaultdict(list)
    for mod, src in _sources():
        for name, nodes in _module_defs(ast.parse(src)).items():
            defs[name].extend((mod, n) for n in nodes)
    flat = {n: [x[1] for x in v] for n, v in defs.items()}
    return flat, _reachable(flat, ENTRY)


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
    dupes = {n: [d.lineno for d in v] for n, v in defs.items() if len(v) > 1}
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


def _code_only(path: Path) -> str:
    """The module's source with comments and docstrings stripped.

    The invariant is about what the module *emits*, not about what it is
    allowed to say.  A comment recording that two routines used to be written
    against a CUDA intrinsic is exactly the note a reader wants, and a check
    that forbade naming it would delete its own explanation.
    """
    import io
    import tokenize

    out = []
    with open(path, "rb") as fh:
        tokens = list(tokenize.tokenize(fh.readline))
    prev_end = (1, 0)
    for tok in tokens:
        if tok.type == tokenize.COMMENT:
            continue
        if tok.type == tokenize.STRING and tok.line.strip().startswith(
                ('"""', "'''", 'r"""', "r'''")):
            continue                      # docstring on its own line
        out.append(tok.string)
    return "\n".join(out)


@pytest.mark.parametrize("mod", MODULES)
def test_no_cuda_intrinsics_in_the_amd_package(mod):
    """`__shfl_xor_sync` is CUDA.  Two routines here were written against it."""
    src = _code_only(AMD / f"{mod}.py")
    for token in ("__shfl_xor_sync", "__shfl_sync", "__ballot_sync"):
        assert token not in src, f"{mod}.py: {token} is a CUDA intrinsic"


def test_module_has_no_empty_stubs(analysis):
    """`def f(...): pass` reads as an implemented hook and is not one."""
    defs, _ = analysis
    stubs = []
    for name, nodes in defs.items():
        node = nodes[-1]
        if not isinstance(node, ast.FunctionDef):
            continue
        body = [s for s in node.body if not isinstance(s, ast.Expr)
                or not isinstance(s.value, ast.Constant)]
        if len(body) == 1 and isinstance(body[0], ast.Pass):
            stubs.append(name)
    assert not stubs, f"empty stubs: {stubs}"
