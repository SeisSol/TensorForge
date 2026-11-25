# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""The high-order tile's fixture: the nonzero patterns of SeisSol's elastic
volume operators kDivM(0..2) at orders 10 and 12, packed as bits into
tests/fixtures/seissol/hiorder-kdivm.npz, which tests/hiorder_fixture.py
builds the volume's descriptions from.

The patterns come from the operators of the modal (Dubiner) basis the
hand-written high-order kernels used -- computed by
~/tf-probe/matrices-hi/compute.py, `linear-ader-<order>.npz`; at order 8
they are SeisSol's own, entry for entry -- and only the patterns are kept:
the tree's tests fill them with seeded values, and the jobs that measure
accuracy load the real ones (`hiorder_fixture.description`'s `values`).

    python3 tools/hiorder_fixture.py <matrices-hi out dir> [out.npz]
"""

import sys
from pathlib import Path

import numpy as np

ORDERS = (10, 12)
OPERATORS = ('kDivM(0)', 'kDivM(1)', 'kDivM(2)')


def main(source: Path, out: Path) -> None:
    arrays = {}
    for order in ORDERS:
        found = np.load(source / f'linear-ader-{order}.npz')
        for name in OPERATORS:
            pattern = found[name] != 0
            key = f'o{order}/{name}'
            arrays[f'{key}/shape'] = np.array(pattern.shape, dtype=np.int32)
            arrays[f'{key}/bits'] = np.packbits(pattern.reshape(-1))
            print(f'{key}: {pattern.shape}, {int(pattern.sum())} entries')
    np.savez_compressed(out, **arrays)
    print(f'{out}: {out.stat().st_size} B')


if __name__ == '__main__':
    here = Path(__file__).resolve().parent.parent
    main(Path(sys.argv[1]), Path(sys.argv[2]) if len(sys.argv) > 2 else
         here / 'tests' / 'fixtures' / 'seissol' / 'hiorder-kdivm.npz')
