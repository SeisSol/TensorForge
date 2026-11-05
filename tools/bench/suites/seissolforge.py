# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""The kernels of a time step against the generators SeisSol used before
TensorForge: gemmforge and chainforge.

Every kernel in five configurations, in one run, as a ladder of single
steps (`suite.CONFIGS`):

| configuration | launches | built by |
|---|---|---|
| `baseline` | one | TensorForge |
| `chains` | as gemmforge-chainforge cuts the kernel | TensorForge |
| `gemmforge-chainforge` | the same | gemmforge and chainforge |
| `ops` | one per operation | TensorForge |
| `gemmforge` | the same | gemmforge |

`baseline` against `gemmforge-chainforge` is what SeisSol gained; `chains`
between them parts that into what fusing the whole kernel is worth and what
the code is worth, at the launches the old generators made.  `ops` and
`gemmforge` are the same comparison one product at a time, yateto's path
where chainforge cannot be imported.  Where a kernel is one chain --
`volume`, `localFluxAll`, the neighbour flux -- `chains` builds the kernel
`baseline` builds, and the comparison is of the code alone.

The elastic system with one simulation only: products of several
simulations are of a rank the old generators do not take.  The workloads
carry the kernel's name with a suffix (`capture.Legacy`), so that no reader
of the timing run sees a second measurement of a kernel, from another run on
another clock.  Every configuration's outputs are checked against
`baseline`'s before anything is timed (`CHECK`): a kernel of the old
generators that computes something else is a finding, not a number.

The batches are the ramp-up's ends and its middle: the launches a sequence
adds weigh most at the small ones, where local time stepping's clusters are.
"""

import dataclasses
import os
import sys
from pathlib import Path

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from capture import Legacy  # noqa: E402
from seissolbench import KERNELS  # noqa: E402
from suite import from_dump  # noqa: E402

NAME = 'seissolforge'
DESCRIPTION = ('captured kernels against gemmforge and chainforge, whole, '
               'as they cut them, and one launch per operation')

BATCHES = (256, 1024, 4096, 16384, 65536, 262144)
CONFIGS = ['baseline', 'chains', 'gemmforge-chainforge', 'ops', 'gemmforge']
CHECK = True

#: The equations the old generators are measured on.
EQUATIONS = ('elastic',)

DUMP = Path(os.environ.get('TF_BENCH_DUMP', 'descriptors.json'))


def workloads():
    if not DUMP.exists():
        raise SystemExit(
            f'no descriptor capture at {DUMP}; point $TF_BENCH_DUMP at an '
            f'export, as for suites/seissolbench.py.')
    out = []
    for w in from_dump(DUMP):
        if w.name not in KERNELS or w.capture is None:
            continue
        if (w.capture.equation not in EQUATIONS
                or w.capture.simulations != 1):
            continue
        out.append(dataclasses.replace(w, name=Legacy(w.name).name))
    if not out:
        raise SystemExit(f'{DUMP}: no kernel of the elastic system with one '
                         'simulation, which is what the old generators are '
                         'measured on')
    return out
