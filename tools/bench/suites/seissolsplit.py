# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""The kernels of `seissolbench`, whole and as one launch per operation.

What fusion is worth, measured rather than argued: each kernel beside the
chain a generator that does not fuse launches -- one kernel per product, the
temporaries in global memory between them (`suite.split_description`) --
built by the same generator and timed in the same run, because two runs
differ by the device's clock as much as by the code (README, "Why a
configuration is measured against another in one run").  `publish.py` adds
the pieces of a kernel up per batch and sets the sum against the whole.

The whole kernel is named as its own single piece, `gpu_volume~0-of-1`
(`capture.Piece`), so that it is measured here without being measured twice
under its name: every other reader sees `seissolbench`'s run of it only.  A
kernel of one operation has no chain to split and is left out.  Two batches,
the smallest and the largest, which are what the figure reads: over the
sixteen exports there are 952 pieces against 86 kernels.

Each piece is timed on its own and warm, so what one piece leaves in the cache
for the next is not modelled.  At a small batch that changes little -- the
chain would find its temporaries in L2, and a piece timed alone finds them
there too, from its own previous launch -- and at a large batch neither finds
them there.
"""

import dataclasses
import os
import sys
from collections import Counter
from pathlib import Path

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from capture import Piece  # noqa: E402
from seissolbench import KERNELS  # noqa: E402
from suite import from_dump  # noqa: E402

NAME = 'seissolsplit'
DESCRIPTION = 'captured kernels, whole and split into one launch per operation'

BATCHES = (256, 262144)
CONFIGS = ['baseline']

DUMP = Path(os.environ.get('TF_BENCH_DUMP', 'descriptors.json'))


def workloads():
    if not DUMP.exists():
        raise SystemExit(
            f'no descriptor capture at {DUMP}; point $TF_BENCH_DUMP at an '
            f'export, as for suites/seissolbench.py.')
    pieces = [w for w in from_dump(DUMP, split=True)
              if Piece.of(w.name).kernel in KERNELS]
    chained = {k for k, n in Counter(Piece.of(w.name).kernel
                                     for w in pieces).items() if n > 1}
    whole = [dataclasses.replace(w, name=Piece(w.name, 0, 1).name)
             for w in from_dump(DUMP) if w.name in chained]
    return whole + [w for w in pieces if Piece.of(w.name).kernel in chained]
