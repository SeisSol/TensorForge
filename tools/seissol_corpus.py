# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Phase-1 measurement over SeisSol's `matrices_N.xml`.

The operator matrices are the corpus the sparse-constant cost model has to be
fitted to, so they get measured before anything is modelled.  This reads the
XML SeisSol ships, groups the matrices into families, and runs
`tools/spp_metrics.py` over each one.

Two conventions the numbers depend on, stated rather than assumed:

* The XML is 1-based in both `row` and `column`; the arrays here are 0-based.
* Axis 0 is the *lead* axis, which is the one TensorForge spreads across
  lanes and the one a wide access runs along.  Axis 1 is the contraction
  axis for a left operand.  Both are reported, because a matrix that is a
  left operand in one kernel is a right operand in another.

Usage::

    python tools/seissol_corpus.py path/to/matrices_*.xml
    python tools/seissol_corpus.py --json out.json path/to/matrices_*.xml
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from spp_metrics import measure, PatternMetrics  # noqa: E402

#: Tile shapes worth scoring.  ``K = 4`` is the contraction extent every FP64
#: matrix instruction agrees on -- `mfma_f64_4x4x4f64`, `mfma_f64_16x16x4f64`
#: and `wmma_f64_16x16x4_f64` alike -- so it is the one the FP64 path has to
#: live with.  The free extent varies with the lane block, and the degenerate
#: ``1`` covers the single row or column the 4x4 instruction accepts.
TILES_LEFT = ((1, 4), (4, 4), (16, 4), (32, 4))
#: The same shapes for an operand whose contraction axis is axis 0.
TILES_RIGHT = ((4, 1), (4, 4), (4, 16), (4, 32))

#: A family name with its instance index stripped: ``kDivM(0)`` -> ``kDivM``.
_INSTANCE = re.compile(r'\(\s*\d+(?:\s*,\s*\d+)*\s*\)\s*$')


def family_of(name: str) -> str:
    return _INSTANCE.sub('', name)


def load(path: Path) -> "OrderedDict[str, np.ndarray]":
    """Every matrix in one file, dense, Fortran-ordered, 0-based."""
    out: "OrderedDict[str, np.ndarray]" = OrderedDict()
    for node in ET.parse(str(path)).getroot().findall('matrix'):
        rows = int(node.get('rows'))
        cols = int(node.get('columns'))
        arr = np.zeros((rows, cols), dtype=np.float64, order='F')
        for entry in node.findall('entry'):
            # Maple writes `.5e-1` and `-.125`, which float() reads as is.
            arr[int(entry.get('row')) - 1,
                int(entry.get('column')) - 1] = float(entry.get('value'))
        out[node.get('name')] = arr
    return out


def order_of(path: Path) -> int:
    """The basis size in the filename, which is what names these files."""
    m = re.search(r'_(\d+)\.xml$', path.name)
    if not m:
        raise ValueError(f'{path.name}: no basis size in the filename')
    return int(m.group(1))


def measure_file(path: Path) -> List[Tuple[str, str, PatternMetrics]]:
    """One `PatternMetrics` per matrix, tagged with basis size and family."""
    basis = order_of(path)
    results = []
    for name, arr in load(path).items():
        fam = family_of(name)
        tiles = TILES_LEFT + TILES_RIGHT
        # A tile larger than the matrix scores as one occupied tile, which is
        # true but uninformative; drop those rather than report them.
        tiles = tuple(t for t in tiles
                      if t[0] <= arr.shape[0] and t[1] <= arr.shape[1])
        results.append((str(basis), fam,
                        measure(arr, name=f'{basis}/{name}', values=arr,
                                tile_shapes=tiles)))
    return results


def _agg(ms: List[PatternMetrics]) -> Dict[str, float]:
    """Family-level aggregate.

    Non-zeros are the weight throughout: a family's ten instances differ in
    where their non-zeros sit, and averaging the per-matrix ratios would give
    a nearly empty instance the same say as a full one.
    """
    nnz = sum(m.nnz for m in ms)
    vol = sum(m.volume for m in ms)
    lens0 = [l for m in ms for l in m.runs[0].lengths]
    lens1 = [l for m in ms for l in m.runs[1].lengths]
    inline = ([m.inline_fraction * m.nnz for m in ms]
              if all(m.inline_fraction is not None for m in ms) else None)
    return {
        'count': len(ms),
        'nnz': nnz,
        'volume': vol,
        'density': nnz / vol if vol else 0.0,
        'runs0': len(lens0),
        'meanlen0': float(np.mean(lens0)) if lens0 else 0.0,
        'runs1': len(lens1),
        'meanlen1': float(np.mean(lens1)) if lens1 else 0.0,
        'ge2_0': sum(l for l in lens0 if l >= 2) / nnz if nnz else 0.0,
        'ge4_0': sum(l for l in lens0 if l >= 4) / nnz if nnz else 0.0,
        'ge2_1': sum(l for l in lens1 if l >= 2) / nnz if nnz else 0.0,
        'ge4_1': sum(l for l in lens1 if l >= 4) / nnz if nnz else 0.0,
        'inline': sum(inline) / nnz if inline and nnz else 0.0,
    }


def report(paths: List[Path]) -> Tuple[str, dict]:
    rows = []
    blob: dict = {}
    for path in sorted(paths, key=order_of):
        per_family: Dict[str, List[PatternMetrics]] = {}
        for basis, fam, m in measure_file(path):
            per_family.setdefault(fam, []).append(m)
        for fam, ms in per_family.items():
            a = _agg(ms)
            a['basis'] = order_of(path)
            a['family'] = fam
            a['shape'] = list(ms[0].shape)
            fills = {}
            for tile in ms[0].blocks:
                stored = sum(m.blocks[tile].stored for m in ms)
                fills['×'.join(str(t) for t in tile)] = {
                    'fill': sum(m.blocks[tile].nnz for m in ms) / stored,
                    'stored': stored,
                }
            a['tiles'] = fills
            meta = {'runs': 0, 'bitmap': 0, 'bitmap_g4': 0}
            for m in ms:
                meta['runs'] += m.metadata_bytes(0, 1)['runs']
                meta['bitmap'] += m.metadata_bytes(0, 1)['bitmap']
                meta['bitmap_g4'] += m.metadata_bytes(0, 4)['bitmap']
            a['metadata'] = meta
            rows.append(a)
        blob[str(order_of(path))] = [r for r in rows
                                     if r['basis'] == order_of(path)]

    head = (f'{"N":>4} {"family":<8} {"shape":>9} {"#":>3} {"nnz":>7} '
            f'{"dens":>5} {"run0":>6} {"len0":>6} {"≥2":>5} {"≥4":>5} '
            f'{"len1":>6} {"inline":>6}')
    lines = [head, '-' * len(head)]
    for r in rows:
        lines.append(
            f'{r["basis"]:>4} {r["family"]:<8} '
            f'{r["shape"][0]}×{r["shape"][1]:<7} {r["count"]:>3} '
            f'{r["nnz"]:>7} {r["density"]:>5.2f} {r["runs0"]:>6} '
            f'{r["meanlen0"]:>6.1f} {r["ge2_0"]:>5.2f} {r["ge4_0"]:>5.2f} '
            f'{r["meanlen1"]:>6.1f} {r["inline"]:>6.2f}')
    return '\n'.join(lines), blob


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('files', nargs='+', type=Path)
    ap.add_argument('--json', type=Path, default=None,
                    help='also write the full numbers here')
    args = ap.parse_args(argv)

    text, blob = report(args.files)
    print(text)
    if args.json:
        args.json.write_text(json.dumps(blob, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
