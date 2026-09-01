# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Where the staging decision actually changes, across the whole corpus.

`spp_occupancy` answers one operand set on one target.  The question that
decides how much machinery is worth building is a different one: over every
family, every order and every target, how many *distinct* answers are there?
A decision that comes out the same everywhere wants a constant, not a pass.

So this sweeps and then marks the flips.  A row is starred where its choice
differs from the same family and target one order down -- the staged count,
the residency, or the set of layouts.  The stars are the output; the rest of
the table is there to see why.

Two inputs are assumptions and not measurements, and both are flags:

``traversals``
    How often one element reads the operand.  The stiffness matrices carry
    the ADER recursion and are applied about `order - 1` times; the surface
    matrices are applied once per face.  This drives the whole cost of leaving
    an operand in global memory, so it is worth checking against the kernels
    rather than taking on trust.

``fixed_lds``
    Shared memory the block needs regardless of the constants.  Zero is right
    for the operators measured on AMD, where the operands sit in registers and
    the only thing in LDS is the staged constant, and wrong as soon as a
    kernel stages its operands too.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

from seissol_corpus import COVER_WIDTHS, family_of, load, order_of  # noqa: E402
from spp_metrics import measure                                     # noqa: E402
from spp_occupancy import Machine, frontier                         # noqa: E402
from spp_plan import placements                                     # noqa: E402

#: Targets to sweep.  One CDNA and two NVIDIA generations, which between them
#: span the shared memory sizes that matter: 64 kB against 163 and 227.
TARGETS = (('gfx90a', 'hip'), ('sm_80', 'cuda'), ('sm_90', 'cuda'))

#: Tile shapes to offer.  Four is the contraction extent every FP64 matrix
#: instruction agrees on, and one row deep is what the 4x4 instruction accepts.
TILES = ((1, 4), (2, 4), (4, 4))


def traversals_for(family: str, basis: int, order: int) -> int:
    """How often one element reads this operand, under the ADER structure.

    An assumption about the kernels rather than a fact about the matrices;
    `--traversals` overrides it wholesale when it is wrong.
    """
    return max(1, order - 1) if family.startswith('kDivM') else 1


def basis_to_order(basis: int) -> int:
    """`O(O+1)(O+2)/6 = basis`, solved by trying."""
    for order in range(1, 20):
        if order * (order + 1) * (order + 2) // 6 == basis:
            return order
    raise ValueError(f'{basis} is not an ADER-DG basis size')


class Row:
    __slots__ = ('basis', 'order', 'family', 'arch', 'blocks', 'limit',
                 'lds', 'staged', 'total', 'layouts', 'cycles', 'flip')

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)
        self.flip = ''

    @property
    def key(self) -> Tuple:
        """What has to change for this to count as a different decision."""
        return (self.blocks, self.staged, tuple(sorted(self.layouts)))


def sweep(paths: Sequence[Path], fp_bytes: int = 8,
          threads_per_block: int = 256, mults: int = 8,
          fixed_lds: int = 0, traversals: Optional[int] = None,
          latency: float = 600.0, cache_hit: float = 0.0,
          bytes_per_cycle: float = 64.0) -> List[Row]:
    machines = {arch: Machine.from_hw(_hw(arch, backend))
                for arch, backend in TARGETS}
    rows: List[Row] = []
    for path in sorted(paths, key=order_of):
        basis = order_of(path)
        order = basis_to_order(basis)
        families: Dict[str, list] = {}
        for name, arr in load(path).items():
            families.setdefault(family_of(name), []).append((name, arr))
        for family, members in sorted(families.items()):
            trav = (traversals if traversals is not None
                    else traversals_for(family, basis, order))
            groups = [
                placements(measure(arr, name=name, tile_shapes=TILES,
                                   cover_widths=COVER_WIDTHS),
                           fp_bytes, traversals=trav * mults)
                for name, arr in members]
            for arch, _ in TARGETS:
                levels = frontier(groups, machines[arch], threads_per_block,
                                  mults, fixed_lds=fixed_lds,
                                  bytes_per_cycle=bytes_per_cycle,
                                  latency=latency, cache_hit=cache_hit)
                if not levels:
                    continue
                best = levels[0]
                staged = [p for p in best.plan.chosen if p.staged]
                rows.append(Row(
                    basis=basis, order=order, family=family, arch=arch,
                    blocks=best.residency.blocks,
                    limit=best.residency.limit,
                    lds=best.lds_per_block, staged=len(staged),
                    total=len(members),
                    layouts=sorted({p.label for p in staged}) or ['—'],
                    cycles=best.seconds_per_element(bytes_per_cycle, latency,
                                                    cache_hit)))
    _mark_flips(rows)
    return rows


def _hw(arch: str, backend: str):
    from tensorforge.common.basic_types import Datatype
    from tensorforge.common.context import Context
    return Context(arch=arch, backend=backend,
                   fp_type=Datatype.F64).get_vm().get_hw_descr()


def _mark_flips(rows: List[Row]) -> None:
    """Star a row whose decision differs from the next order down.

    Against the previous *order* and not the previous row: the sweep walks
    orders outermost, so the comparison that means something is the one along
    the axis the operators grow on.
    """
    seen: Dict[Tuple[str, str], Tuple] = {}
    for row in rows:
        ident = (row.family, row.arch)
        prev = seen.get(ident)
        if prev is not None and prev != row.key:
            row.flip = '*'
        seen[ident] = row.key


def report(rows: Sequence[Row], only_flips: bool = False) -> str:
    head = (f'{"":1} {"N":>4} {"O":>2} {"family":<8} {"arch":<7} '
            f'{"blk":>4} {"limit":<8} {"lds/blk":>9} {"staged":>7} '
            f'{"cyc/elem":>10}  layouts')
    lines = [head, '-' * (len(head) + 12)]
    for row in rows:
        if only_flips and not row.flip:
            continue
        lines.append(
            f'{row.flip:1} {row.basis:>4} {row.order:>2} {row.family:<8} '
            f'{row.arch:<7} {row.blocks:>4} {row.limit:<8} '
            f'{row.lds / 1024:>8.1f}K {row.staged:>3}/{row.total:<3} '
            f'{row.cycles:>10.0f}  {",".join(row.layouts)}')
    return '\n'.join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('files', nargs='+', type=Path)
    ap.add_argument('--traversals', type=int, default=None,
                    help='override the per-family ADER assumption')
    ap.add_argument('--fixed-lds', type=int, default=0)
    ap.add_argument('--mults', type=int, default=8)
    ap.add_argument('--threads', type=int, default=256)
    ap.add_argument('--latency', type=float, default=600.0)
    ap.add_argument('--cache-hit', type=float, default=0.0)
    ap.add_argument('--fp', type=int, default=8, choices=(4, 8))
    ap.add_argument('--flips-only', action='store_true')
    args = ap.parse_args(argv)

    rows = sweep(args.files, fp_bytes=args.fp, threads_per_block=args.threads,
                 mults=args.mults, fixed_lds=args.fixed_lds,
                 traversals=args.traversals, latency=args.latency,
                 cache_hit=args.cache_hit)
    print(report(rows, only_flips=args.flips_only))
    distinct = len({r.key for r in rows})
    print(f'\n{len(rows)} Zeilen, {sum(1 for r in rows if r.flip)} Wechsel, '
          f'{distinct} verschiedene Entscheidungen')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
