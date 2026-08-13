"""What in the AMD package is reachable from its single entry point?

`multilinear.py` calls exactly one name: `amd.matmul`.  Anything not reachable
from there through the call graph is unreachable *by construction* --- a
stronger statement than "no test covered it", and the one a deletion needs.

`tests/test_amd_reachability.py` asserts this property; this prints it.  Use it
when deciding what a change made dead, or before deleting something.

    python3 tools/reachability.py
"""

import ast
from collections import defaultdict
from pathlib import Path

PKG = Path('tensorforge/backend/instructions/compute/primitives/amd')
MODULES = ['__init__', 'arch', 'caps', 'catalog', 'relayout', 'select',
           'emitters', 'codegen', 'unused']
ENTRY = 'matmul'


def module_defs(tree):
    defs = defaultdict(list)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)):
            defs[node.name].append(node)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id != '__all__':
                    defs[t.id].append(node)
    return defs


def main():
    defs, home = defaultdict(list), {}
    for mod in MODULES:
        path = PKG / f'{mod}.py'
        if not path.exists():
            print(f'  (missing: {path})')
            continue
        for name, nodes in module_defs(ast.parse(path.read_text())).items():
            defs[name].extend(nodes)
            home.setdefault(name, mod)

    live = {n: v[-1] for n, v in defs.items()}
    reach, work = set(), [ENTRY]
    while work:
        name = work.pop()
        if name in reach or name not in live:
            continue
        reach.add(name)
        for n in ast.walk(live[name]):
            if isinstance(n, ast.Name) and n.id in live and n.id not in reach:
                work.append(n.id)

    dupes = {n: [d.lineno for d in v] for n, v in defs.items() if len(v) > 1}
    if dupes:
        print('SHADOWED -- a later definition silently discards the earlier:')
        for n, lines in sorted(dupes.items()):
            print(f'    {n:26s} {home[n]}.py lines {lines}')
        print()

    print(f'entry point: {ENTRY}()\n')
    print(f'REACHABLE ({len(reach)}):')
    for n in sorted(reach):
        print(f'    {n:26s} {home[n]}.py')
    unreachable = sorted(set(live) - reach)
    print(f'\nUNREACHABLE ({len(unreachable)}):')
    for n in unreachable:
        print(f'    {n:26s} {home[n]}.py')
    if not unreachable:
        print('    -')


if __name__ == '__main__':
    main()
