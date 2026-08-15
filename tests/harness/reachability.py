# SPDX-License-Identifier: MIT
"""Reachability over a vendor package's call graph.

Lifted out of `test_amd_reachability.py` unchanged when the same guard was
needed for `primitives/nvidia.py`.  Two copies of a static analysis is two
places for it to be subtly different, and the interesting content -- which
names are unreachable on purpose, and why -- belongs to each vendor, not to
the mechanism.

Reachability is a stronger statement than coverage: coverage says no case
happened to run this, reachability says no case can.
"""

from __future__ import annotations

import ast
import io
import tokenize
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple


def module_defs(tree: ast.Module) -> Dict[str, List[ast.AST]]:
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


def reachable(defs: Dict[str, List[ast.AST]],
              entries: Sequence[str]) -> Set[str]:
    live = {n: v[-1] for n, v in defs.items()}
    seen: Set[str] = set()
    work = list(entries)
    while work:
        name = work.pop()
        if name in seen or name not in live:
            continue
        seen.add(name)
        for n in ast.walk(live[name]):
            if isinstance(n, ast.Name) and n.id in live and n.id not in seen:
                work.append(n.id)
    return seen


def analyse(root: Path, modules: Sequence[str],
            entries: Sequence[str]) -> Tuple[Dict[str, List[ast.AST]],
                                             Set[str]]:
    """Definitions and reachability across a whole package.

    Cross-module `from .x import y` lines are not treated as definitions --
    they are edges, and the definition they point at is already in the merged
    table.  Counting them would make every re-export look like a second
    definition of the same name.

    Merging the modules before computing reachability is deliberate: a name
    that is unreachable only because it sits in another file is still
    unreachable, and splitting a module must not become a way to launder dead
    code past this check.
    """
    defs = defaultdict(list)
    for mod in modules:
        path = root / f"{mod}.py"
        assert path.exists(), f"{path} is missing; the module list is stale"
        for name, nodes in module_defs(ast.parse(path.read_text())).items():
            defs[name].extend(nodes)
    flat = dict(defs)
    return flat, reachable(flat, entries)


def code_only(path: Path) -> str:
    """The module's source with comments and docstrings stripped.

    Vendor-purity checks are about what a module *emits*, not about what it is
    allowed to say.  A comment recording that two routines used to be written
    against the other vendor's intrinsic is exactly the note a reader wants,
    and a check that forbade naming it would delete its own explanation.
    """
    out = []
    with open(path, "rb") as fh:
        tokens = list(tokenize.tokenize(fh.readline))
    for tok in tokens:
        if tok.type == tokenize.COMMENT:
            continue
        if tok.type == tokenize.STRING and tok.line.strip().startswith(
                ('"""', "'''", 'r"""', "r'''")):
            continue                      # docstring on its own line
        out.append(tok.string)
    return "\n".join(out)


def duplicate_definitions(defs) -> Dict[str, List[int]]:
    return {n: [d.lineno for d in v] for n, v in defs.items() if len(v) > 1}


def empty_stubs(defs) -> List[str]:
    """`def f(...): pass` reads as an implemented hook and is not one."""
    stubs = []
    for name, nodes in defs.items():
        node = nodes[-1]
        if not isinstance(node, ast.FunctionDef):
            continue
        body = [s for s in node.body
                if not isinstance(s, ast.Expr)
                or not isinstance(s.value, ast.Constant)]
        if len(body) == 1 and isinstance(body[0], ast.Pass):
            stubs.append(name)
    return stubs
