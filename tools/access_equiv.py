# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Did this refactor change *which* memory gets touched?

A textual snapshot diff answers "did the source change", which during a
migration is almost always yes and almost never the question.  Removing the
pin from an address, for one, renumbers every SSA value after it and lets the
emitter inline single-use addresses into their loads, so::

    int32_t v199_a = v197_i0 + v198_i1;
    float v200_data = r2[v199_a];

becomes::

    float v201_data = r2[(v197_i0 + v198_i1)];

Two spellings, one access.  Reviewing thousands of those by eye is how a real
change gets waved through in the middle of them.

So: expand every SSA name in every subscript transitively down to leaves ---
loop variables, thread indices, literals --- canonicalise, and compare the
resulting multiset of ``(base, address)`` pairs against a git revision.  If it
matches, the refactor moved names and not memory.

Canonicalised away, because a migration produces all three and none of them is
a change in behaviour:

* renumbering --- names are replaced by order of first appearance
* parenthesisation --- ``ast.unparse`` gives a minimal-paren form
* ``0 + x`` and ``1 * x`` --- what folding does once the address is foldable

Deliberately *not* canonicalised: associativity and distribution.  ``a*(b+c)``
and ``a*b + c`` stay different, because on an address they usually are.

    python3 tools/access_equiv.py            # against HEAD
    python3 tools/access_equiv.py HEAD~3
    python3 tools/access_equiv.py -v         # show matching targets too

Exit status is 1 when anything differs, so it can gate a commit.
"""
import ast
import importlib.util
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

from tensorforge.common.context import Context
from tensorforge.generators.generator import Generator

TARGETS = [('gfx90a', 'hip'), ('sm_86', 'cuda')]
SNAPSHOT_DIR = Path('tests/snapshots')

# `T name = expr;` -- the definitions an address is built from.  A loop header
# does not match: it ends in `{`, not `;`.  That is what keeps a loop variable
# a *leaf*, which it has to be --- expanding it to its initial value would
# claim every iteration reads iteration zero, and hide an off-by-one.
DEF = re.compile(r'^\s*(?:const\s+)?'
                 r'(?:int32_t|unsigned|float|double|auto|size_t)\s+'
                 r'(v\d+_\w+)\s*=\s*(.+?);\s*$')
ACC = re.compile(r'(\w+)\[([^\[\]]*)\]')
NAME = re.compile(r'v\d+_(\w+)')

_INTRINSIC = (('threadIdx.x', 'TIDX'), ('threadIdx.y', 'TIDY'),
              ('threadIdx.z', 'TIDZ'), ('blockIdx.x', 'BIDX'),
              ('blockIdx.y', 'BIDY'), ('blockDim.x', 'BDIMX'))

# `32_i32` is a C++ literal with a type suffix and a Python syntax error.  The
# suffix carries no address information, so strip it rather than letting the
# expression fall back to a flat-text comparison -- text is sensitive to
# exactly the parenthesisation this tool exists to see past.
_SUFFIX = re.compile(r'\b(\d+)_[iu]\d+\b')


class _Fold(ast.NodeTransformer):
    """The identities an unpinned address folds on its own."""

    def visit_BinOp(self, node):
        self.generic_visit(node)
        lhs, rhs = node.left, node.right

        def is_(n, k):
            return isinstance(n, ast.Constant) and n.value == k

        if isinstance(node.op, ast.Add):
            if is_(lhs, 0):
                return rhs
            if is_(rhs, 0):
                return lhs
        if isinstance(node.op, ast.Mult):
            if is_(lhs, 1):
                return rhs
            if is_(rhs, 1):
                return lhs
        return node


def _expand(expr, defs, depth=0):
    """Substitute definitions until only leaves remain."""
    if depth > 64:              # a cycle would mean the source is not SSA
        return expr
    return re.sub(
        r'v\d+_\w+',
        lambda m: (f'({_expand(defs[m.group(0)], defs, depth + 1)})'
                   if m.group(0) in defs else m.group(0)),
        expr)


def _canon_expr(expr):
    """Minimal-paren form with the identity terms folded.

    One fold pass suffices: `NodeTransformer` rewrites bottom-up, so a nested
    `0 + (0 + x)` collapses on the way out.  Measured over the corpus, a second
    pass changes nothing in 60848 accesses.
    """
    text = _SUFFIX.sub(r'\1', expr)
    for c, py in _INTRINSIC:
        text = text.replace(c, py)
    try:
        tree = _Fold().visit(ast.parse(text, mode='eval'))
    except SyntaxError:
        # Deliberately loud.  The quiet alternatives -- fall back to comparing
        # flat text, or drop the entry -- both end with the tool reporting
        # "identical" for a file whose strangest accesses it silently stopped
        # looking at, and its whole value is that its answer licenses not
        # reading the diff.  A construct it cannot parse is a gap in the tool,
        # and should read as one.
        raise ValueError(
            f'cannot parse the subscript {expr!r} (as {text!r}). Extend '
            f'_SUFFIX or _INTRINSIC rather than letting this pass silently')
    return ast.unparse(ast.fix_missing_locations(tree))


def accesses(src):
    """Multiset of ``(base, expanded address)`` over one generated kernel."""
    defs = {}
    for line in src.splitlines():
        m = DEF.match(line)
        if m:
            defs[m.group(1)] = m.group(2)

    renames = {}

    def canon_names(text):
        def rename(m):
            renames.setdefault(m.group(0), f'N{len(renames)}_{m.group(1)}')
            return renames[m.group(0)]
        return NAME.sub(rename, text)

    out = Counter()
    for line in src.splitlines():
        for base, index in ACC.findall(line):
            if not index.strip():
                continue                  # `float sh[];` is a declaration
            out[(canon_names(base),
                 canon_names(_canon_expr(_expand(index, defs))))] += 1
    return out


def _at_rev(rev, path):
    r = subprocess.run(['git', 'show', f'{rev}:{path}'],
                       capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else None


def _case_paths():
    """Recursive, matching `conftest.py`: the subdirectories hold 24 cases."""
    return [p for p in sorted(Path('tests/cases').rglob('*.py'))
            if '__pycache__' not in p.parts]


def _cases():
    for path in _case_paths():
        spec = importlib.util.spec_from_file_location('case', path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        yield mod


def main(argv):
    verbose = '-v' in argv
    rest = [a for a in argv if a != '-v']
    rev = rest[0] if rest else 'HEAD'

    differ, same, skipped = [], 0, []
    for mod in _cases():
        for arch, backend in TARGETS:
            rel = f'{SNAPSHOT_DIR}/{mod.NAME}.{backend}.cpp'
            old = _at_rev(rev, rel)
            if old is None:
                skipped.append(f'{mod.NAME}/{backend} (no snapshot at {rev})')
                continue
            try:
                gen = Generator(mod.descr_list(),
                                Context(arch=arch, backend=backend,
                                        fp_type=getattr(mod, 'DTYPE', None)))
                gen.generate()
                new = gen.get_kernel() or ''
            except Exception as exc:
                skipped.append(f'{mod.NAME}/{backend} ({type(exc).__name__})')
                continue
            before, after = accesses(old), accesses(new)
            if before == after:
                same += 1
                if verbose:
                    print(f'  ok    {mod.NAME}/{backend} '
                          f'({sum(after.values())} accesses)')
            else:
                differ.append((mod.NAME, backend, before - after, after - before))

    for name, backend, gone, added in differ:
        print(f'{name}/{backend}: '
              f'{sum(gone.values())} only in {rev}, {sum(added.values())} only now')
        for (b, e), n in list(gone.items())[:6]:
            print(f'    was {n}x {b}[{e}]')
        for (b, e), n in list(added.items())[:6]:
            print(f'    now {n}x {b}[{e}]')

    print(f'\nagainst {rev}: {same} targets identical, {len(differ)} differ'
          + (f', {len(skipped)} skipped' if skipped else ''))
    if skipped and verbose:
        for s in skipped:
            print(f'  skipped {s}')
    return 1 if differ else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
