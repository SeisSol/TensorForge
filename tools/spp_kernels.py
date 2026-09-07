# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""The constant operands as the code generator actually sees them.

`seissol_corpus.py` measures `matrices_N.xml`, which is the operators as
SeisSol ships them.  That is not what reaches a kernel: yateto slices them,
fuses them, and hands the backend a bounding box and an eqspp window, and a
kernel takes some of them and not others.  The staging decision is made per
kernel over the operands that kernel holds, so it has to be measured there.

A capture from `tools/host/dump_descriptors.py` has everything needed and no
guesses in it:

* which operands are batch-constant, from `addressing`
* their exact stored pattern, from the `pack` map -- the same map the kernel
  and the harness address them through, so the pattern here is the pattern
  the kernel has, not one reconstructed from a threshold on the values
* how often one element reads each of them, by counting the descriptors that
  name it

That last one replaces the assumption `spp_sweep.py` carries.  There the
traversal count follows the ADER recursion because nothing better was
available; here it is counted.

    python3 tools/spp_kernels.py descriptors.json
    python3 tools/spp_kernels.py --arch gfx90a --fp 8 descriptors.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from spp_metrics import measure                     # noqa: E402
from spp_occupancy import Machine, frontier         # noqa: E402
from spp_plan import placements                     # noqa: E402
from spp_sweep import TILES, _hw                    # noqa: E402
from seissol_corpus import COVER_WIDTHS             # noqa: E402


def captures(path: Path) -> Dict[str, list]:
    """kernel name -> its descriptor rows."""
    blob = json.loads(Path(path).read_text())
    return blob['all'] if 'all' in blob else {'(single)': blob['descrs']}


def pattern_of(operand: dict) -> Optional[np.ndarray]:
    """The operand's occupancy mask over its bounding box.

    From the pack map when there is one, because that is the storage the
    kernel addresses.  Otherwise from the baked-in values, where a structural
    zero and a numeric one are indistinguishable and the mask is a lower bound
    on how sparse the operand really is.  Neither available means the operand
    is dense as far as anything here can tell, and it is reported as such
    rather than skipped.
    """
    shape = tuple(operand.get('ashape') or operand['shape'])
    if not shape:
        return None
    mask = np.zeros(shape, dtype=bool, order='F')
    flat = mask.reshape(-1, order='F')
    pack = operand.get('pack')
    if pack is not None:
        flat[np.asarray(pack, dtype=int)] = True
        return flat.reshape(shape, order='F')
    data = operand.get('data')
    if data is not None and len(data) == flat.size:
        return (np.asarray(data, dtype=float).reshape(shape, order='F') != 0.0)
    flat[:] = True
    return flat.reshape(shape, order='F')


def values_of(operand: dict) -> Optional[np.ndarray]:
    shape = tuple(operand.get('ashape') or operand['shape'])
    data = operand.get('data')
    if data is None or not shape:
        return None
    arr = np.asarray(data, dtype=float)
    if arr.size != int(np.prod(shape)):
        return None
    return arr.reshape(shape, order='F')


def constants_of(rows: Sequence[Optional[dict]]) -> Tuple[Dict[str, dict],
                                                          Counter]:
    """Batch-constant operands of one kernel, and how often each is read.

    Keyed by name rather than alias: two aliases of one buffer are one
    staging decision, and counting them apart would double the traffic of a
    tensor that is read once.
    """
    found: Dict[str, dict] = {}
    uses: Counter = Counter()
    for row in rows:
        if row is None:
            continue
        for op in row['ops']:
            if op is None or 'none' not in str(op.get('addressing', '')).lower():
                continue
            found.setdefault(op['name'], op)
            uses[op['name']] += 1
    return found, uses


def analyse(path: Path, arch: str = 'gfx90a', backend: str = 'hip',
            fp_bytes: int = 8, threads_per_block: int = 256, mults: int = 8,
            fixed_lds: int = 0, latency: float = 600.0,
            cache_hit: float = 0.0, bytes_per_cycle: float = 64.0) -> List[dict]:
    machine = Machine.from_hw(_hw(arch, backend))
    out: List[dict] = []
    for kernel, rows in sorted(captures(path).items()):
        found, uses = constants_of(rows)
        if not found:
            continue
        groups, names = [], []
        for name, op in sorted(found.items()):
            mask = pattern_of(op)
            if mask is None:
                continue
            metrics = measure(mask, name=name, values=values_of(op),
                              tile_shapes=TILES, cover_widths=COVER_WIDTHS)
            groups.append(placements(metrics, fp_bytes,
                                     traversals=uses[name] * mults))
            names.append((name, metrics, uses[name]))
        if not groups:
            continue
        levels = frontier(groups, machine, threads_per_block, mults,
                          fixed_lds=fixed_lds, bytes_per_cycle=bytes_per_cycle,
                          latency=latency, cache_hit=cache_hit)
        if not levels:
            continue
        best = levels[0]
        staged = [p for p in best.plan.chosen if p.staged]
        out.append(dict(
            kernel=kernel,
            operands=len(names),
            nnz=sum(m.nnz for _, m, _ in names),
            dense=sum(m.volume for _, m, _ in names),
            uses=sorted({u for _, _, u in names}),
            blocks=best.residency.blocks,
            limit=best.residency.limit,
            lds=best.lds_per_block,
            staged=len(staged),
            layouts=sorted({p.label for p in staged}) or ['—'],
            cycles=best.seconds_per_element(bytes_per_cycle, latency,
                                            cache_hit),
        ))
    return out


def report(rows: Sequence[dict]) -> str:
    head = (f'{"kernel":<22} {"ops":>4} {"nnz":>7} {"dens":>5} {"uses":>8} '
            f'{"blk":>4} {"limit":<8} {"lds/blk":>9} {"staged":>7} layouts')
    lines = [head, '-' * (len(head) + 10)]
    for r in rows:
        density = r['nnz'] / r['dense'] if r['dense'] else 0.0
        uses = ','.join(str(u) for u in r['uses'][:3])
        lines.append(
            f'{r["kernel"][:22]:<22} {r["operands"]:>4} {r["nnz"]:>7} '
            f'{density:>5.2f} {uses:>8} {r["blocks"]:>4} {r["limit"]:<8} '
            f'{r["lds"] / 1024:>8.1f}K {r["staged"]:>3}/{r["operands"]:<3} '
            f'{",".join(r["layouts"])}')
    keys = {(r['blocks'], r['staged'], tuple(r['layouts'])) for r in rows}
    lines.append(f'\n{len(rows)} Kernel mit konstanten Operanden, '
                 f'{len(keys)} verschiedene Entscheidungen')
    return '\n'.join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('capture', type=Path)
    ap.add_argument('--arch', default='gfx90a')
    ap.add_argument('--backend', default='hip')
    ap.add_argument('--fp', type=int, default=8, choices=(4, 8))
    ap.add_argument('--threads', type=int, default=256)
    ap.add_argument('--mults', type=int, default=8)
    ap.add_argument('--fixed-lds', type=int, default=0)
    ap.add_argument('--latency', type=float, default=600.0)
    ap.add_argument('--cache-hit', type=float, default=0.0)
    ap.add_argument('--json', type=Path, default=None)
    args = ap.parse_args(argv)

    rows = analyse(args.capture, arch=args.arch, backend=args.backend,
                   fp_bytes=args.fp, threads_per_block=args.threads,
                   mults=args.mults, fixed_lds=args.fixed_lds,
                   latency=args.latency, cache_hit=args.cache_hit)
    print(report(rows))
    if args.json:
        args.json.write_text(json.dumps(rows, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
