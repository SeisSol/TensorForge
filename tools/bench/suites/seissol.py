# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Kernels captured from a SeisSol codegen run.

Point `TF_BENCH_DUMP` at a `descriptors.json` produced by
`tools/host/dump_descriptors.py`.  These are the production shapes: the sparsity
is the operator matrices' own, the chains are as long as yateto made them, and
the batch is the element count of a real mesh partition.

Deliberately not vendored into the repository.  A capture is tied to an order,
an equation set and a memory layout, and a checked-in one would be measured
long after it stopped describing what SeisSol generates.
"""

import os
from pathlib import Path

from suite import from_dump

NAME = 'seissol'
DESCRIPTION = 'kernels from a SeisSol descriptor capture'

BATCHES = (4096, 65536, 262144)
CONFIGS = ['baseline', 'wave', 'pipeline', 'wrap1', 'wrap2']

DUMP = Path(os.environ.get('TF_BENCH_DUMP', 'descriptors.json'))


def workloads():
    if not DUMP.exists():
        raise SystemExit(
            f'no descriptor capture at {DUMP}. Produce one with\n'
            f'  python3 tools/host/dump_descriptors.py --out descriptors.json '
            f'-- <generate.py arguments>\n'
            f'and point $TF_BENCH_DUMP at it.')
    return from_dump(DUMP)
