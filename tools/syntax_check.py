# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Does every generated kernel parse, and does every call in it resolve?

`hipcc -fsyntax-only` for people without hipcc: the `// === kernel ===` section
of each snapshot, on top of the declaration-only shim in
`tests/shim/tensorforge_host.h`, through `g++`.  Same machinery as
`tests/test_syntax.py` -- this is the form that fits next to the other tools.

    python3 tools/syntax_check.py            # the whole corpus
    python3 tools/syntax_check.py '*.hip.*'  # one backend
"""
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tests'))
from harness import syntax  # noqa: E402

pattern = sys.argv[1] if len(sys.argv) > 1 else '*.cpp'

cxx = syntax.compiler()
if cxx is None:
    print('no host C++ compiler found (set TF_HOST_CXX to pick one)')
    raise SystemExit(2)

files = syntax.snapshots(pattern)
if not files:
    print(f'no snapshots matching {pattern!r}; run pytest --snapshot-update')
    raise SystemExit(2)

with ThreadPoolExecutor(8) as ex:
    results = list(ex.map(lambda p: syntax.check_snapshot(p, cxx), files))

kinds = Counter()
bad = 0
for r in results:
    if r.ok is None:
        kinds['skipped'] += 1
        continue
    if r.ok:
        kinds['ok'] += 1
        continue
    bad += 1
    print(f'--- {r.path.name}')
    for line in r.errors():
        print(f'    {line.strip()}')

print(f'\n--- {kinds["ok"]} well-formed, {bad} ill-formed, '
      f'{kinds["skipped"]} without a kernel section '
      f'({Path(cxx).name}) ---')
raise SystemExit(1 if bad else 0)
