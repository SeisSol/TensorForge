# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Figures and tables for a write-up, as LaTeX, from what was measured.

`compare.py` is for looking at a campaign; this is for putting one in a
paper.  The difference is not the data -- it reads the same tree, through
`compare.collect` -- but the medium: pgfplots and booktabs in the document's
own fonts and at its column width (`texout.py`), each beside the CSV of the
numbers it draws, so a figure can be re-set without being re-measured and a
number in the text can be checked against the one in the figure.

    python3 tools/bench/publish.py eval/ --out pub/
    python3 tools/bench/publish.py gb200/ mi300a/ lumi/ --out pub/ --compile
    python3 tools/bench/publish.py eval/ --seissol runs/ --clustering \\
        mesh-clustering.csv@elastic-4-d --out pub/
    python3 tools/bench/publish.py --out pub/          # the kernels alone

## What it writes

| output | from | the question |
|---|---|---|
| `tables/kernels` | exports | what does each kernel compute? |
| `tables/intensity`, `figures/intensity` | exports | how many flops per byte, at each order and precision? |
| `figures/operations-*` | exports | what does each kernel multiply, drawn to scale? |
| `figures/operators-*` | exports | where are the shared operators' nonzeros? |
| `tables/machines` | ceilings | what did each machine turn out to do, per precision? |
| `figures/roofline-*` | runs, ceilings | where does each kernel sit, and how does it move with the order? |
| `figures/peak-*` | runs, ceilings | how close to the roof, the bus and the FMA rate does a full launch get? |
| `figures/rampup-*`, `tables/rampup-*` | runs | from how many elements on is a launch worth its cost? |
| `figures/lts-*` | clusterings, runs | how much of local time stepping do the launch costs leave? |
| `figures/breakdown-*` | runs | where does one time step's time go, per order? |
| `figures/timesplit-*` | SeisSol runs | where did a simulation's time go? |
| `figures/throughput-*` | SeisSol runs | what did SeisSol get through on each machine? |
| `tables/portability`, `figures/cascade-*` | runs, ceilings | how evenly does a kernel reach each machine's roof? |
| `figures/configs-*` | runs | does a configuration -- or another generator -- help? |
| `figures/fusion-*` | split runs | what is fusing a kernel's chain worth? |
| `figures/lanes-*` | runs | how much of what a kernel issues is needed? |
| `figures/resources-*` | runs | registers, spills and code, against the order |
| `figures/precision-*` | runs, ceilings | what does double cost against single? |
| `figures/simulations-*` | runs | what are fused simulations worth? |
| `figures/traffic-*` | profiles | how much more than it had to did a kernel move? |
| `tables/buildcost-*` | manifests | what did making the kernels cost? |

Every figure is a bare `tikzpicture` in `figures/`, every table a bare
`tabular` in `tables/`, both `\\input`-able; `tfstyle.tex` holds what they
share.  `preview.tex` sets all of them with a suggested caption, and with
`--compile` becomes `preview.pdf`, beside one PDF per figure for a paper
that includes graphics rather than pgfplots.

## Which numbers

The flop count is chosen with `--flops` as in the other readers and named
on every axis that divides by it.  The clock is the device's for the
roofline, the peaks and the portability -- what an achieved rate should be
divided by -- and the wall clock over back-to-back launches for the ramp-up,
which is about the launch overhead SeisSol pays and the device clock leaves
out (`--rampup-clock`).  A "full launch" is the largest batch measured; a
ramp-up wants a sweep of them (`RAMPUP_BATCHES`).

A kernel is its name and its export (`capture.py`): a figure is drawn for
one machine, one precision and one equation, and a curve joins the orders
of one kernel and nothing else.

## What the runs say

`--seissol` reads what SeisSol writes about a run itself (`seissolrun.py`):
the summary -- configuration, machine, cell updates, the time per compute
region and SeisSol's own fit of it against the cells a launch carried -- and
the clustering beside it.  The LTS figure reads a clustering against the
launch costs the ramp-up measured; `--clustering` names one directly, and
`synthetic` the series in `clusterings/` until a mesh's is at hand, which
every figure drawn from it says.  Profiles (`profile.py`) and what the
builds cost (`build.py`) are found in the campaign trees themselves.

## What is written by hand

`STEP`, how SeisSol launches the kernels of a time step, which no
measurement says; `SLOTS`, which kernel keeps which colour in every figure.
The rest comes from the runs, the ceilings and the exports.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import texout                                                    # noqa: E402
from capture import DATATYPE, Capture, Piece                     # noqa: E402
from compare import Ceiling, Field, Point, collect               # noqa: E402
from seissolrun import (Clustering, Run, collect_runs,
                        measured_speedups, read_clustering)      # noqa: E402
from plot import (FLOP_FIELD, FLOP_LABEL, fallback_note, flop_basis,
                  flops_of, set_flop_basis)                      # noqa: E402

# ----------------------------------------------------------------------
# What only a person can say
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class Launch:
    """How one kernel is launched in a time step of `n` elements: `count`
    launches of `share * n` elements each, of which `together` at a time
    go to streams of their own."""
    count: int = 1
    share: float = 1.0
    together: int = 1

    @property
    def per_element(self) -> float:
        """How many times an element passes through the kernel per step."""
        return self.count * self.share

    def text(self) -> str:
        if self.count == 1 and self.share == 1:
            return 'one launch of n'
        return (f'{self.count} launches of n/{1 / self.share:g}'
                + (f', {self.together} at a time' if self.together > 1
                   else ''))


#: One time step of global time stepping, by equation, as SeisSol launches
#: it on a GPU (its `src/Kernels/LinearCK*/{Time,Local,Neighbor}.cpp`, with
#: the local fluxes combined, the default): the neighbour flux once per
#: face and face relation -- four faces, four relations each, the four
#: relations of a face on streams of their own -- and taken as spread
#: evenly over the relations, which a mesh only approximates.  One
#: relation's kernel stands for all of them.  What the breakdown and the
#: time step's ramp-up are made of; under local time stepping a cluster
#: adds the Taylor expansion for every neighbour in another cluster, which
#: no export says.
STEP: Dict[str, Dict[str, Launch]] = {
    'elastic': {'gpu_derivative': Launch(), 'gpu_volume': Launch(),
                'gpu_localFluxAll': Launch(),
                'gpu_neighboringFlux[0]': Launch(16, 0.25, 4)},
    'viscoelastic': {'gpu_derivative': Launch(), 'gpu_volumeExt': Launch(),
                     'gpu_fluxLocalAll': Launch(),
                     'gpu_neighborFluxExt[0]': Launch(16, 0.25, 4),
                     'gpu_neighbor': Launch()},
}


def step_text(step: Dict[str, Launch]) -> str:
    return '; '.join(f'{short(k)} {launch.text()}'
                     for k, launch in step.items())


#: The kernels drawn unless `--kernels` says otherwise: those of a time step
#: and the Taylor expansion local time stepping adds, which is what
#: `suites/seissolbench.py` measures.
DEFAULT_KERNELS = (
    'gpu_derivative', 'gpu_volume', 'gpu_localFluxAll',
    'gpu_neighboringFlux[0]', 'gpu_derivativeTaylorExpansion',
    'gpu_volumeExt', 'gpu_fluxLocalAll', 'gpu_neighborFluxExt[0]',
    'gpu_neighbor', 'gpu_derivativeTaylorExpansionEla')

#: Which categorical slot a kernel keeps in every figure, by its family: a
#: kernel that is blue in the roofline is blue in the breakdown, and the
#: viscoelastic counterpart of an elastic kernel takes its colour.  A
#: kernel not listed takes the next free slot in order of appearance.
SLOTS: Dict[str, int] = {
    'derivative': 0, 'volume': 1, 'volumeExt': 1, 'localFluxAll': 2,
    'fluxLocalAll': 2, 'neighboringFlux': 3, 'neighborFluxExt': 3,
    'derivativeTaylorExpansion': 4, 'derivativeTaylorExpansionEla': 4,
    'neighbor': 5, 'local': 6,
}


def short(workload: str) -> str:
    """A kernel's name in a figure: no `gpu_`, and no `[0]` where the first
    variant stands for the family."""
    name = workload[4:] if workload.startswith('gpu_') else workload
    return name[:-3] if name.endswith('[0]') else name


def _family(workload: str) -> str:
    name = short(workload)
    return name.split('[')[0]


class Styles:
    """The series style of every kernel, fixed for the whole publication."""

    def __init__(self) -> None:
        self._given: Dict[str, int] = {}

    def slot(self, workload: str) -> int:
        fam = _family(workload)
        if fam in SLOTS:
            return SLOTS[fam]
        if fam not in self._given:
            self._given[fam] = len(set(SLOTS.values())) + len(self._given)
        return self._given[fam]

    def __call__(self, workload: str) -> str:
        """The line-and-mark style (`tf s<n>` in `tfstyle.tex`)."""
        return f'tf s{self.slot(workload) + 1}'

    def fill(self, workload: str) -> str:
        """The same identity as a fill, for a bar."""
        return f'tfc{self.slot(workload) % len(texout.CATEGORICAL) + 1}'


# ----------------------------------------------------------------------
# Outputs
# ----------------------------------------------------------------------

@dataclass
class Output:
    """One figure or table: its LaTeX, the rows of its CSV, and the caption
    a reader of the preview is offered."""
    name: str
    kind: str                        # 'figure' | 'table'
    tex: str
    rows: List[Dict] = field(default_factory=list)
    caption: str = ''


def _write_csv(rows: Sequence[Dict], path: Path) -> None:
    if not rows:
        return
    keys: List[str] = []
    for r in rows:
        keys.extend(k for k in r if k not in keys)
    with path.open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _nanos(row: Dict, clock: str) -> Optional[float]:
    """Nanoseconds per launch on `clock`: the device's (`event`) or the
    wall clock's over back-to-back launches (`wall`) -- falling back to the
    other only where the row has none of the asked one, which the caller
    is told about through `fell_back`."""
    first, second = (('event_ns', 'wall_ns') if clock == 'event'
                     else ('wall_ns', 'event_ns'))
    return row.get(first) or row.get(second) or None


def _fell_back(rows: Iterable[Dict], clock: str) -> int:
    key = 'event_ns' if clock == 'event' else 'wall_ns'
    return sum(1 for r in rows if not r.get(key))


def _gflops(row: Dict, clock: str) -> Optional[float]:
    nanos, flops = _nanos(row, clock), flops_of(row)
    return flops / nanos if nanos and flops else None


def _gbytes(row: Dict, clock: str) -> Optional[float]:
    nanos = _nanos(row, clock)
    return row['bytes'] / nanos if nanos and row.get('bytes') else None


def _intensity(row: Dict) -> Optional[float]:
    f = flops_of(row)
    return f / row['bytes'] if f and row.get('bytes') else None


def _dt_name(datatype: str) -> str:
    return {'F32': 'FP32', 'F64': 'FP64'}.get(datatype, datatype)


def _basis_label() -> str:
    return {'executed': 'issued', 'dense': 'dense',
            'nonzero': 'needed'}[flop_basis()]


def _flops_are() -> str:
    """The sentence a caption names the flop count in."""
    return {'executed': 'The flops are those the kernel issues.',
            'dense': 'The flops are those a dense implementation performs.',
            'nonzero': 'The flops are those the operation needs, its '
                       'structural zeros taken out.'}[flop_basis()]


def _flops_note() -> str:
    """The same for the comment atop a generated file, with what the count
    is worth (`analysis.pipeline.executed_flops`)."""
    return (f'flops {FLOP_LABEL[flop_basis()]}'
            + (' -- never below the hardware count and up to 1.53x above it '
               'where arithmetic reaches fewer lanes than its guard claims, '
               'so a fraction of the roof is a ceiling'
               if flop_basis() == 'executed' else ''))


# ----------------------------------------------------------------------
# The kernels: from the exports alone
# ----------------------------------------------------------------------

def table_kernels(described: Sequence, kernels: Sequence[str],
                  order: int) -> Optional[Output]:
    """What each kernel computes, at one order in double with one
    simulation: the formula in a paper's notation, its step of the scheme,
    and how many operations and steps the export has for it."""
    import describe as dsc
    chosen: Dict[Tuple[str, str], object] = {}
    for d in described:
        c = d.capture
        if (c is None or d.workload not in kernels or c.simulations != 1
                or c.precision != 'd'):
            continue
        key = (c.equation, d.workload)
        prev = chosen.get(key)
        # The asked order, else the one nearest it.
        if prev is None or (abs(c.order - order)
                            < abs(prev.capture.order - order)):
            chosen[key] = d
    if not chosen:
        return None
    rows_tex: List[Optional[List[str]]] = []
    rows_csv: List[Dict] = []
    equations = sorted({eq for eq, _ in chosen})
    for eq in equations:
        if rows_tex:
            rows_tex.append(None)
        if len(equations) > 1:
            rows_tex.append([r'\multicolumn{5}{@{}l}{\textit{'
                             + texout.escape(eq) + '}}'])
        for k in kernels:
            d = chosen.get((eq, k))
            if d is None:
                continue
            tex_lines = dsc.lines(d.formula,
                                  lambda s: dsc.to_latex(s, dsc.NOTATION))
            rows_tex.append([
                r'\texttt{' + texout.escape(short(k)) + '}',
                texout.escape(d.stage),
                texout.stack(f'${line}$' for line in tex_lines),
                str(d.operations), str(d.depth)])
            rows_csv.append({
                'equation': eq, 'kernel': k, 'order': d.capture.order,
                'stage': d.stage, 'role': d.role,
                'formula': '; '.join(dsc.lines(d.formula, dsc.to_text)),
                'yateto': '; '.join(dsc.lines(d.formula, dsc.to_yateto)),
                'operations': d.operations, 'temporaries': d.temporaries,
                'steps': d.depth})
    ref = sorted({d.capture.order for d in chosen.values()})
    tex = texout.table(
        'l l l r r',
        [['Kernel', 'Step', 'Computes', 'Ops', 'Steps']],
        rows_tex,
        comment=('generated by tools/bench/publish.py -- the formulas are '
                 'read off the exports (describe.py);\n"Ops" counts the '
                 'export\'s operations, "Steps" its longest chain of them, '
                 f'at order {"/".join(map(str, ref))}, double, one '
                 'simulation'))
    caption = ('The kernels of one time step, as the export hands them to '
               'TensorForge. The formula is reconstructed from the export\'s '
               "operations; ``Ops'' is how many operations it has, ``Steps'' "
               f'the longest chain through them (order '
               f'{"/".join(map(str, ref))}, double precision).')
    return Output('kernels', 'table', tex, rows_csv, caption)


def _static_rows(described: Sequence, kernels: Sequence[str]) -> List[Dict]:
    rows = []
    for d in described:
        c = d.capture
        if c is None or d.workload not in kernels or not d.bytes:
            continue
        rows.append({
            'equation': c.equation, 'kernel': d.workload,
            'order': c.order, 'precision': c.precision,
            'simulations': c.simulations,
            'flops_dense': d.flops, 'flops_needed': d.nonzero_flops,
            'bytes': d.bytes,
            'intensity_dense': d.intensity,
            'intensity_needed': d.nonzero_intensity})
    return rows


def table_intensity(described: Sequence, kernels: Sequence[str]
                    ) -> Optional[Output]:
    """Flops and bytes per element, and the intensity they make, per kernel
    and order -- with the precision as columns, since only the bytes move
    with it -- for one simulation and, where the exports have them, eight."""
    rows = _static_rows(described, kernels)
    if not rows:
        return None
    by: Dict[Tuple, Dict[str, Dict]] = defaultdict(dict)
    for r in rows:
        by[(r['equation'], r['simulations'], r['kernel'], r['order'])][
            r['precision']] = r
    body: List = []
    last = None
    for (eq, sims, k, order) in sorted(
            by, key=lambda t: (t[0], t[1], kernels.index(t[2]), t[3])):
        group = (eq, sims, k)
        if last is None or (eq, sims) != last[:2]:
            if last is not None:
                body.append(None)
            body.append([r'\multicolumn{9}{@{}l}{\textit{'
                         + texout.escape(eq)
                         + (f', {sims} simulations' if sims > 1 else '')
                         + '}}'])
        elif group != last:
            body.append(texout.GAP)
        cells = by[(eq, sims, k, order)]
        ref = cells.get('d') or cells.get('s')
        name = (r'\texttt{' + texout.escape(short(k)) + '}'
                if group != last else '')
        d, s = cells.get('d', {}), cells.get('s', {})
        body.append([
            name, str(order),
            texout.num(ref['flops_needed']), texout.num(ref['flops_dense']),
            texout.num(d.get('bytes')),
            texout.num(d.get('intensity_needed')),
            texout.num(d.get('intensity_dense')),
            texout.num(s.get('intensity_needed')),
            texout.num(s.get('intensity_dense'))])
        last = group
    tex = texout.table(
        'l r r r r r r r r',
        [['', '', r'\multicolumn{2}{c}{flop/element}', 'byte/el.',
          r'\multicolumn{2}{c}{flop/byte, FP64}',
          r'\multicolumn{2}{c}{flop/byte, FP32}'],
         r'\cmidrule(lr){3-4}\cmidrule(lr){6-7}\cmidrule(lr){8-9}',
         ['Kernel', '$O$', 'needed', 'dense', 'FP64', 'needed', 'dense',
          'needed', 'dense']],
        body,
        comment=('generated by tools/bench/publish.py from the exports '
                 '(describe.py); per element, bytes compulsory\nand '
                 'asymptotic: an operator every element shares is read once '
                 'per launch'))
    caption = ('Arithmetic per element -- what the operation needs, with '
               'its structural zeros taken out, and what a dense '
               'implementation performs -- against the bytes an element '
               'cannot avoid moving, and the intensity the two make, per '
               'order and precision. An operator every element shares is '
               'read once per launch and not counted per element.')
    return Output('intensity', 'table', tex, rows, caption)


def figure_intensity(described: Sequence, kernels: Sequence[str],
                     fieldset: Optional[Field], styles: Styles
                     ) -> List[Output]:
    """Intensity against order, per equation, one panel per precision, with
    every measured machine's ridge of that precision as a rule: where a
    kernel crosses from the memory side to the arithmetic one, and on
    which machine."""
    rows = [r for r in _static_rows(described, kernels)
            if r['simulations'] == 1]
    out: List[Output] = []
    for eq in sorted({r['equation'] for r in rows}):
        mine = [r for r in rows if r['equation'] == eq]
        precisions = [p for p in ('d', 's')
                      if any(r['precision'] == p for r in mine)]
        orders = sorted({r['order'] for r in mine})
        if len(orders) < 2:
            continue
        lo, hi = orders[0] - 0.4, orders[-1] + 0.4
        axes = []
        drawn: List[Dict] = []
        for i, prec in enumerate(precisions):
            dt = DATATYPE[prec]
            ax = texout.Axis(options=[
                'ymode=log', f'xtick={{{",".join(map(str, orders))}}}',
                f'xmin={lo}, xmax={hi}',
                f'xlabel={{order $O$, {_dt_name(dt)}}}'])
            if i == 0:
                ax.options += ['ylabel={flop/byte}',
                               'legend style={at={(0,1.12)}, anchor=south '
                               'west, legend columns=3, /tikz/every even '
                               'column/.append style={column sep=0.6em}}']
            for k in kernels:
                pts = sorted((r['order'], r['intensity_needed'],
                              r['intensity_dense']) for r in mine
                             if r['kernel'] == k and r['precision'] == prec)
                if not pts:
                    continue
                style = styles(k)
                ax.plots.append(texout.Plot(
                    style, [(o, n) for o, n, _ in pts],
                    texout.escape(short(k)) if i == 0 else None))
                ax.plots.append(texout.Plot(
                    style + ', densely dashed, mark=none, forget plot',
                    [(o, dn) for o, _, dn in pts]))
                drawn.extend(r for r in mine
                             if r['kernel'] == k and r['precision'] == prec)
            if fieldset is not None:
                ridges = [(arch, ceil.ridge) for (arch, cdt), ceil
                          in sorted(fieldset.ceilings.items())
                          if cdt == dt and ceil.ridge]
                for n, (arch, ridge) in enumerate(ridges):
                    # Each label at its own place along the rule: two
                    # machines' ridges are often a few percent apart.
                    ax.raw.append(texout.hrule(
                        ridge, r'\texttt{' + texout.escape(arch) + '}',
                        anchor='south west', pos=n / len(ridges)))
            axes.append(ax)
        if not axes:
            continue
        if len(axes) == 1:
            axes[0].options.insert(0, 'tf')
            tex = texout.picture(axes, comment=(
                f'generated by tools/bench/publish.py: {eq}, solid what the '
                'operation needs, dashed a dense implementation'))
        else:
            tex = texout.picture(
                axes, f'group size={len(axes)} by 1, horizontal sep=0.9cm',
                comment=(f'generated by tools/bench/publish.py: {eq}, solid '
                         'what the operation needs, dashed a dense '
                         'implementation'),
                options=['width=0.56\\tfwidth', 'height=0.62\\tfwidth'])
        caption = (f'Arithmetic intensity of the {eq} kernels against the '
                   'order: solid what the operation needs, dashed what a '
                   'dense implementation performs, per compulsory byte. '
                   'A rule is a measured machine\'s ridge in that '
                   'precision; above it the machine runs out of arithmetic '
                   'before bandwidth.')
        out.append(Output(f'intensity-{eq}', 'figure', tex, drawn, caption))
    return out


# ----------------------------------------------------------------------
# The kernels drawn: the formula to scale, the operators' nonzeros
# ----------------------------------------------------------------------

#: The width the drawings are laid out at -- a two-column paper's column,
#: 8.6 cm -- and shrunk to `\tfwidth` where that is narrower.  Text keeps its
#: size where the drawing would only stretch to fill a wider one.
_OPS_WIDTH = 244.0
#: A matrix entry, at most, in points: an order-3 operator of ten columns
#: stays the size of a word rather than of the figure.
_OPS_CELL = 4.0
#: A character of `\scriptsize`, a little over the average, for placing
#: labels apart.
_OPS_CHAR = 3.9
_OPS_GAP = 5.0
_OPS_LABEL = 17.0
_OPS_HEADER = 11.0
#: Room below a statement's labels, before the next statement.
_OPS_BELOW = 7.0


def _visible(tex: str) -> int:
    """How many characters a short piece of LaTeX shows, a times sign and a
    thin space counted as the room they take."""
    text = tex.replace('\\times', '++').replace('\\,', ' ')
    text = re.sub(r'\\[a-zA-Z]+', '', text)
    return len(re.sub(r'[{}^_$\\]', '', text))


@dataclass
class _Tensor:
    """One factor as drawn: its reach, how deep it goes past two axes, and
    what is shared by every element -- with its nonzeros -- against what
    every element has its own of."""
    tex: str
    rows: int
    cols: int
    depth: int
    constant: bool
    nonzeros: Tuple[Tuple[int, int], ...]
    members: Tuple[str, ...]
    densities: Tuple[float, ...]

    def label(self, densities: bool = True) -> str:
        sizes = [self.rows, self.cols] + ([self.depth] if self.depth > 1
                                          else [])
        text = '$\\times$'.join(str(s) for s in sizes)
        if densities and self.densities:
            lo, hi = min(self.densities), max(self.densities)
            text += (f', {100 * lo:.0f}\\,\\%' if round(100 * lo)
                     == round(100 * hi) else
                     f', {100 * lo:.0f}--{100 * hi:.0f}\\,\\%')
        return text

    #: Whether the label gives the operator's share of nonzeros: the
    #: drawing of the formula leaves that to the drawing of the operators.
    show_density: bool = True

    def text(self) -> str:
        return self.label(self.show_density)

    def label_width(self) -> float:
        return _OPS_CHAR * max(_visible(self.tex), _visible(self.text())) + 3

    def width(self, cell: float) -> float:
        return self.cols * cell + 1.5 * min(self.depth - 1, 2)

    def height(self, cell: float) -> float:
        return self.rows * cell + 1.5 * min(self.depth - 1, 2)


def _ops_tensor(d, name: str, bindings: Dict[str, Sequence[int]],
                notation) -> Optional[_Tensor]:
    """A folded name as drawn: the reach of all the members it stands for,
    and the nonzeros of the first -- the one the label names."""
    import describe as dsc
    names = dsc.members(name, bindings)
    found = [(n, d.extents[n]) for n in names if n in d.extents]
    if not found:
        return None
    sizes = [max(e.sizes[a] if a < len(e.sizes) else 1 for _, e in found)
             for a in range(max(len(e.sizes) for _, e in found))]
    rows = sizes[0] if sizes else 1
    cols = sizes[1] if len(sizes) > 1 else 1
    depth = 1
    for size in sizes[2:]:
        depth *= size
    first = found[0][1]
    nonzeros = tuple((ix[0], ix[1] if len(ix) > 1 else 0)
                     for ix in first.nonzeros)
    densities = tuple(e.density for _, e in found if e.density is not None)
    return _Tensor(dsc._tex_name(name, notation), rows, cols, depth,
                   first.constant, nonzeros, tuple(n for n, _ in found),
                   densities, show_density=False)


def _ops_sum(sym: str, values: Tuple[int, ...]) -> str:
    """A sum over a family, its range on the symbol: `\\sum_{i=0}^{3}`."""
    if values == tuple(range(values[0], values[0] + len(values))):
        return f'$\\sum_{{{sym}={values[0]}}}^{{{values[-1]}}}$'
    return (f'$\\sum_{{{sym}\\in\\{{{",".join(map(str, values))}'
            '\\}}$')


def _ops_rows(d, notation) -> Tuple[List[List], List[str]]:
    """The statements of a kernel as rows of things to draw -- a tensor,
    or a piece of text: the relation, a sum, a scale factor -- and the
    ranges of its recursions, which a row draws at their first step."""
    import describe as dsc
    out: List[List] = []
    steps: List[str] = []
    for item in d.formula:
        loop = item if isinstance(item, dsc.Loop) else None
        if loop:
            steps.append(f'${loop.symbol} = 0,\\dots,{loop.count - 1}$')
        for s in (item.body if loop else (item,)):
            bindings: Dict[str, Sequence[int]] = {}
            for sym, values, moves in s.over:
                bindings[sym] = values[:1] if moves else values
                if moves:
                    steps.append(f'${sym} = '
                                 + dsc._range(values).replace(
                                     '..', r',\dots,') + '$')
            if loop:
                bindings[loop.symbol] = (0,)
            row: List = []
            dest = _ops_tensor(d, s.dest.name, bindings, notation)
            if dest is None:
                continue
            row += [dest, ('$\\mathrel{+}=$' if s.add else '$=$')]
            row += [_ops_sum(sym, values) for sym, values, moves in s.over
                    if not moves]
            row += [f'${dsc._tex_name(x, notation)}$'
                    if not re.match(r'^-?[\d.]', x) else f'${x}$'
                    for x in s.scalars]
            if s.op:
                row.append(r'$\operatorname{' + s.op.lower() + '}$')
            for i, f in enumerate(s.factors):
                if i:
                    row.append(r'$\cdot$')
                factor = _ops_tensor(d, f.name, bindings, notation)
                row.append(factor if factor is not None
                           else f'${dsc._tex_name(f.name, notation)}$')
            out.append(row)
    return out, list(dict.fromkeys(steps))


def _ops_symbol(tex: str) -> float:
    """The width of a relation, a sum or a scale factor, set in `\\small`."""
    return 5.1 * max(_visible(tex), 1) + 4.0


def _ops_width(row: Sequence, cell: float) -> float:
    total = 0.0
    for item in row:
        if isinstance(item, _Tensor):
            total += max(item.width(cell), item.label_width()) + _OPS_GAP
        else:
            total += _ops_symbol(item) + _OPS_GAP
    return total


def _ops_cell(rows: Sequence[Sequence], width: float = _OPS_WIDTH) -> float:
    """The largest entry, up to `_OPS_CELL`, at which every row fits -- or,
    where a row's labels alone are wider, at which it is no wider than they
    make it: a smaller entry would only shrink the drawing, not the row."""
    floor = 0.6
    limits = [max(width, _ops_width(r, floor)) for r in rows]
    cell = _OPS_CELL
    while cell > floor and any(_ops_width(r, cell) > limit + 0.01
                               for r, limit in zip(rows, limits)):
        cell *= 0.92
    return max(cell, floor)


def _ops_draw(t: _Tensor, x: float, top: float, cell: float) -> List[str]:
    """One tensor at (x, top): a stack behind it where it has more than two
    axes, then the tensor -- filled where each element has its own, framed
    with its nonzeros where all elements share it."""
    out = []
    w, h = t.cols * cell, t.rows * cell
    for k in range(min(t.depth - 1, 2), 0, -1):
        out.append(f'\\draw[{"tfmuted" if t.constant else "tfc3"}, '
                   f'line width=0.3pt] ({x + 1.5 * k:.2f},{top + 1.5 * k:.2f})'
                   f' rectangle +({w:.2f},{-h:.2f});')
    if t.constant:
        out.append(f'\\draw[tfmuted, line width=0.3pt, fill=white] '
                   f'({x:.2f},{top:.2f}) rectangle +({w:.2f},{-h:.2f});')
        inset = 0.12 * cell if cell > 1.5 else 0.0
        cells = [f'({x + c * cell + inset:.2f},{top - r * cell - inset:.2f})'
                 f' rectangle +({cell - 2 * inset:.2f},'
                 f'{-(cell - 2 * inset):.2f})'
                 for r, c in t.nonzeros if r < t.rows and c < t.cols]
        if cells:
            out.append('\\fill[tfc7] ' + ' '.join(cells) + ';')
    else:
        out.append(f'\\draw[tfc3, line width=0.3pt, fill=tfc3!22] '
                   f'({x:.2f},{top:.2f}) rectangle +({w:.2f},{-h:.2f});')
    return out


def _ops_legend(y: float) -> List[str]:
    return [f'\\draw[tfc3, line width=0.3pt, fill=tfc3!22] (0,{y:.2f}) '
            f'rectangle +(6,-6);',
            f'\\node[tf label, anchor=west] at (8,{y - 3:.2f}) '
            '{each element its own};',
            f'\\draw[tfmuted, line width=0.3pt] (90,{y:.2f}) '
            'rectangle +(6,-6);',
            f'\\fill[tfc7] (91.5,{y - 1.5:.2f}) rectangle +(3,-3);',
            f'\\node[tf label, anchor=west] at (98,{y - 3:.2f}) '
            '{shared by all elements, its nonzeros};']


def _ops_picture(body: Sequence[str], comment: str) -> str:
    """A TikZ picture laid out at `_OPS_WIDTH`, shrunk to `\\tfwidth`
    where that is narrower."""
    return ('% ' + comment.replace('\n', '\n% ') + '\n'
            '\\sbox0{\\begin{tikzpicture}[x=1pt, y=1pt, '
            'font=\\scriptsize]\n' + '\n'.join(body)
            + '\n\\end{tikzpicture}}%\n'
            '\\ifdim\\wd0>\\tfwidth\\resizebox{\\tfwidth}{!}{\\usebox0}'
            '\\else\\usebox0\\fi\n')


def _ops_choice(described: Sequence, kernels: Sequence[str]
                ) -> Dict[Tuple[str, int], Dict[str, object]]:
    """The kernels by equation and order, in double with one simulation --
    a drawing has no precision, and one simulation is the one SeisSol
    runs."""
    chosen: Dict[Tuple[str, int], Dict[str, object]] = defaultdict(dict)
    for d in described:
        c = d.capture
        if (c is None or d.workload not in kernels or c.simulations != 1
                or c.precision != 'd' or not getattr(d, 'extents', None)):
            continue
        chosen[(c.equation, c.order)][d.workload] = d
    return chosen


#: How tall a drawing may be, in points: a column of a two-column paper
#: less a caption.  A time step that needs more goes into further parts.
_OPS_HEIGHT = 560.0


def _ops_header(k: str, steps: Sequence[str], y: float,
                continued: bool) -> str:
    note = []
    if continued:
        note.append('continued')
    if steps:
        note.append(f'drawn at the first step, {", ".join(steps)}')
    return (f'\\node[anchor=north west, inner sep=0pt, text=tfink] at '
            f'(0,{y:.2f}) {{\\footnotesize\\texttt{{'
            f'{texout.escape(short(k))}}}'
            + (f'\\quad\\scriptsize\\color{{tfsecondary}}{"; ".join(note)}'
               if note else '') + '};')


def _ops_row(row: Sequence, y: float, cell: float) -> Tuple[List[str], float]:
    """One statement at `y`, its tensors centred on one line and their
    labels on one line below the tallest; the TikZ and the height."""
    body: List[str] = []
    tall = max((t.height(cell) for t in row if isinstance(t, _Tensor)),
               default=8.0)
    middle = y - tall / 2
    x = 0.0
    for item in row:
        if isinstance(item, _Tensor):
            slot = max(item.width(cell), item.label_width())
            body += _ops_draw(item, x + (slot - item.width(cell)) / 2,
                              middle + item.height(cell) / 2, cell)
            body.append(f'\\node[tf label, anchor=north, align=center] at '
                        f'({x + slot / 2:.2f},{y - tall - 1.5:.2f}) '
                        f'{{${item.tex}$\\\\{item.text()}}};')
            x += slot + _OPS_GAP
        else:
            wide = _ops_symbol(item)
            body.append(f'\\node[text=tfink, font=\\small] at '
                        f'({x + wide / 2:.2f},{middle:.2f}) {{{item}}};')
            x += wide + _OPS_GAP
    return body, tall + _OPS_LABEL + _OPS_BELOW


def figure_operations(described: Sequence, kernels: Sequence[str]
                      ) -> List[Output]:
    """Every kernel of a time step as its formula drawn to scale, per
    equation and order: each tensor as the box of what the kernel reaches
    of it, the ones every element has its own of filled, the operators all
    elements share framed with their nonzeros.  A sum over a family is
    drawn with its first member and the reach of all of them, a recursion
    at its first step.  What does not fit a page goes into further parts,
    at the same scale."""
    import describe as dsc
    out = []
    for (eq, order), mine in sorted(_ops_choice(described, kernels).items()):
        blocks = [(k,) + _ops_rows(mine[k], dsc.NOTATION) for k in kernels
                  if k in mine]
        blocks = [(k, rows, steps) for k, rows, steps in blocks if rows]
        if not blocks:
            continue
        cell = _ops_cell([r for _, rows, _ in blocks for r in rows])
        csv_rows: List[Dict] = []
        for k, rows, _ in blocks:
            for n, row in enumerate(rows):
                for item in row:
                    if isinstance(item, _Tensor):
                        csv_rows.append({
                            'equation': eq, 'order': order, 'kernel': k,
                            'statement': n, 'tensor': item.members[0],
                            'members': ' '.join(item.members),
                            'rows': item.rows, 'cols': item.cols,
                            'depth': item.depth, 'shared': item.constant,
                            'nonzeros': len(item.nonzeros),
                            'densities': ' '.join(
                                f'{v:.4f}' for v in item.densities)})

        def lay_out(limit: float) -> List[List[str]]:
            """Into parts no taller than `limit`, a kernel continued where
            it has to be, its header drawn again."""
            parts: List[List[str]] = [[]]
            y = 0.0
            for k, rows, steps in blocks:
                parts[-1].append(_ops_header(k, steps, y, False))
                y -= _OPS_HEADER
                for row in rows:
                    drawn, height = _ops_row(row, y, cell)
                    # The legend below the last row takes a line of its own.
                    if y - height < 14.0 - limit and len(parts[-1]) > 1:
                        if parts[-1][-1].startswith('\\node[anchor=north west'):
                            parts[-1].pop()
                            y += _OPS_HEADER
                        parts[-1] += _ops_legend(y)
                        parts.append([])
                        y = 0.0
                        parts[-1].append(_ops_header(
                            k, steps, y, continued=row is not rows[0]))
                        y -= _OPS_HEADER
                        drawn, height = _ops_row(row, y, cell)
                    parts[-1] += drawn
                    y -= height
                y -= 4.0
            parts[-1] += _ops_legend(y)
            return parts

        parts = lay_out(_OPS_HEIGHT)
        if len(parts) > 1:
            # As even as that many parts allow: the least height at which
            # there are no more of them.
            low, high = 0.0, _OPS_HEIGHT
            for _ in range(20):
                middle = (low + high) / 2
                if len(lay_out(middle)) > len(parts):
                    low = middle
                else:
                    high = middle
            parts = lay_out(high)
        for i, body in enumerate(parts):
            name = f'operations-{eq}-o{order}' + (f'-{i + 1}' if len(parts)
                                                   > 1 else '')
            tex = _ops_picture(body, (
                f'generated by tools/bench/publish.py from the export: '
                f'{eq}, order {order}, double, one simulation; an entry is '
                f'{cell:.2f}pt\nreach and nonzeros per tensor in the CSV'))
            caption = (f'The {eq} kernels of one time step at order {order}'
                       + (f', part {i + 1} of {len(parts)}'
                          if len(parts) > 1 else '')
                       + ', each statement drawn to scale: a box is the part '
                       'of a tensor the kernel reaches, below it its extent. '
                       'Filled, what every element has its own of; framed, '
                       'the operators all elements share, with their '
                       'nonzeros. A sum over a family is drawn with its first '
                       'member and the reach of all of them, a recursion at '
                       'its first step.')
            out.append(Output(name, 'figure', tex,
                              csv_rows if i == 0 else [], caption))
    return out


def figure_operators(described: Sequence, kernels: Sequence[str]
                     ) -> List[Output]:
    """The operators the kernels of a time step share, per equation and
    order, each with its nonzeros: what the kernels skip, and how that
    changes from one face or direction to the next."""
    import describe as dsc
    out = []
    for (eq, order), mine in sorted(_ops_choice(described, kernels).items()):
        seen: Dict[str, object] = {}
        for k in kernels:
            d = mine.get(k)
            if d is None:
                continue
            for name, e in d.extents.items():
                if e.constant and e.nonzeros and name not in seen:
                    seen[name] = e
        if not seen:
            continue
        # A family together, its members in order, the families as they
        # first appear.
        families = list(dict.fromkeys(dsc.Ref(n, ()).base for n in seen))
        seen = dict(sorted(seen.items(), key=lambda item: (
            families.index(dsc.Ref(item[0], ()).base),
            dsc.Ref(item[0], ()).family)))
        thumbs = [_Tensor(dsc._tex_name(n, dsc.NOTATION), e.sizes[0],
                          e.sizes[1] if len(e.sizes) > 1 else 1, 1, True,
                          tuple((i[0], i[1] if len(i) > 1 else 0)
                                for i in e.nonzeros), (n,),
                          (e.density,) if e.density is not None else ())
                  for n, e in seen.items()]
        widest = max(t.cols for t in thumbs)
        cell = min(_OPS_CELL, (_OPS_WIDTH / 4 - _OPS_GAP) / widest)
        body: List[str] = []
        csv_rows: List[Dict] = []
        x, y, tall = 0.0, 0.0, 0.0
        for t in thumbs:
            slot = max(t.width(cell), t.label_width())
            if x and x + slot > _OPS_WIDTH:
                y -= tall + _OPS_LABEL + 6.0
                x, tall = 0.0, 0.0
            body += _ops_draw(t, x + (slot - t.width(cell)) / 2, y, cell)
            tall = max(tall, t.height(cell))
            body.append(f'\\node[tf label, anchor=north, align=center] at '
                        f'({x + slot / 2:.2f},{y - t.height(cell) - 1.5:.2f})'
                        f' {{${t.tex}$\\\\{t.text()}}};')
            csv_rows.append({'equation': eq, 'order': order,
                             'operator': t.members[0], 'rows': t.rows,
                             'cols': t.cols, 'nonzeros': len(t.nonzeros),
                             'density': t.densities[0] if t.densities
                             else None})
            x += slot + _OPS_GAP
        tex = _ops_picture(body, (
            f'generated by tools/bench/publish.py from the export: {eq}, '
            f'order {order}, double; an entry is {cell:.2f}pt'))
        caption = (f'The operators the {eq} kernels share at order {order}, '
                   'each over the part the kernels reach of it, with its '
                   'nonzeros, its extent and their share.')
        out.append(Output(f'operators-{eq}-o{order}', 'figure', tex,
                          csv_rows, caption))
    return out


# ----------------------------------------------------------------------
# The machines
# ----------------------------------------------------------------------

def table_machines(fieldset: Field) -> Optional[Output]:
    """What each machine turned out to do, per precision: the FMA chain,
    the further roofs recorded beside it, the triad and the bus."""
    archs = sorted({a for (a, _) in fieldset.ceilings})
    if not archs:
        return None
    datatypes = sorted({dt for (_, dt) in fieldset.ceilings if dt})
    rows_tex, rows_csv = [], []
    for arch in archs:
        m = fieldset.machine(arch)
        per = {dt: fieldset.ceiling(arch, dt) for dt in datatypes}
        units = sorted({(u.name, u.flops_per_s, dt) for dt, c in per.items()
                        if c for u in c.units})
        rows_tex.append(
            [texout.escape(fieldset.device_of(arch)),
             r'\texttt{' + texout.escape(arch) + '}']
            + [texout.num(c.flops_per_s / 1e12) if c and c.flops_per_s
               else '--' for c in per.values()]
            + [texout.num(m.bytes_per_s / 1e9) if m and m.bytes_per_s
               else '--',
               texout.num(m.peak_bytes_per_s / 1e9)
               if m and m.peak_bytes_per_s else '--']
            + [texout.num(c.ridge) if c and c.ridge else '--'
               for c in per.values()])
        rows_csv.append(dict(
            {'arch': arch, 'device': fieldset.device_of(arch),
             'gbytes_triad': m.bytes_per_s / 1e9 if m else None,
             'gbytes_bus': m.peak_bytes_per_s / 1e9 if m else None},
            **{f'tflops_{dt}': (c.flops_per_s / 1e12 if c else None)
               for dt, c in per.items()},
            **{f'ridge_{dt}': (c.ridge if c else None)
               for dt, c in per.items()},
            **{f'tflops_{dt}_{name}': f / 1e12 for name, f, dt in units}))
    headers = (['Device', 'Arch']
               + [f'{_dt_name(dt)} TF/s' for dt in datatypes]
               + ['GB/s', 'bus GB/s']
               + [f'ridge {_dt_name(dt)}' for dt in datatypes])
    tex = texout.table(
        'l l ' + 'r ' * (len(headers) - 2), [headers], rows_tex,
        comment=('generated by tools/bench/publish.py from roofline.py\'s '
                 'ceilings: an FMA chain per precision, a triad,\nand the '
                 'bus the device reports; further roofs (`--unit`) are in '
                 'the CSV'))
    caption = ('The machines, as measured: the rate of an FMA chain per '
               'precision, a STREAM triad, the bandwidth the memory bus '
               'allows, and the ridge where the two roofs meet (flop/byte).')
    return Output('machines', 'table', tex, rows_csv, caption)


# ----------------------------------------------------------------------
# The runs
# ----------------------------------------------------------------------

def _full(fieldset: Field, kernels: Sequence[str]) -> List[Point]:
    """One point per kernel, export, configuration and machine: the largest
    launch, in the kernel set."""
    return [p for p in fieldset.largest_batch(fieldset.points)
            if p.workload in kernels and p.capture is not None]


def _panels(points: Iterable[Point]) -> List[Tuple[str, str, str, int]]:
    """The figures the points make: machine, precision, equation,
    simulations."""
    return sorted({(p.arch, p.datatype, p.capture.equation,
                    p.capture.simulations) for p in points})


def _roof_lines(ax: texout.Axis, ceiling: Ceiling, x0: float, x1: float
                ) -> None:
    """The two measured roofs, and any further one, as lines; what they
    are, in the corner above the memory slope that no kernel can reach."""
    ridge = ceiling.ridge
    ax.raw.append(
        f'\\addplot[tf roof, forget plot] coordinates {{'
        f'({x0:.6g},{ceiling.bytes_per_s * x0 / 1e9:.6g}) '
        f'({ridge:.6g},{ceiling.flops_per_s / 1e9:.6g}) '
        f'({x1:.6g},{ceiling.flops_per_s / 1e9:.6g})}};\n')
    for unit in ceiling.units:
        knee = unit.flops_per_s / ceiling.bytes_per_s
        ax.raw.append(
            f'\\addplot[tf unit, forget plot] coordinates {{'
            f'({ridge:.6g},{ceiling.flops_per_s / 1e9:.6g}) '
            f'({knee:.6g},{unit.flops_per_s / 1e9:.6g}) '
            f'({x1:.6g},{unit.flops_per_s / 1e9:.6g})}};\n')
    lines = [f'FMA chain {texout.num(ceiling.flops_per_s / 1e12)}\\,TFLOP/s',
             f'triad {texout.num(ceiling.bytes_per_s / 1e9)}\\,GB/s']
    lines += [f'{texout.escape(u.name)} '
              f'{texout.num(u.flops_per_s / 1e12)}\\,TFLOP/s (dashed)'
              for u in ceiling.units]
    ax.raw.append(
        '\\node[tf label, anchor=north west, align=left] at '
        '(rel axis cs:0.02,0.98) {' + r' \\ '.join(lines) + '};\n')


def figure_rooflines(fieldset: Field, kernels: Sequence[str],
                     styles: Styles) -> Tuple[List[Output], List[str]]:
    """Each kernel at every order it was run at, under one machine's roof
    in one precision: a line per kernel through its orders, each point
    labelled with its order, so the figure shows where a kernel sits and
    which way the order moves it."""
    out, skipped = [], []
    full = [p for p in _full(fieldset, kernels)
            if p.config == fieldset.reference]
    for arch, dt, eq, sims in _panels(full):
        ceiling = fieldset.ceiling(arch, dt)
        tag = f'{arch}-{eq}-{dt.lower()}' + (f'-s{sims}' if sims > 1 else '')
        if ceiling is None or not ceiling.ridge:
            skipped.append(f'roofline-{tag}: no {dt} ceiling for {arch}')
            continue
        mine = [p for p in full if (p.arch, p.datatype, p.capture.equation,
                                    p.capture.simulations)
                == (arch, dt, eq, sims)]
        pts = [(p, _intensity(p.row), _gflops(p.row, 'event'))
               for p in mine]
        pts = [t for t in pts if t[1] and t[2]]
        if not pts:
            continue
        xs = [i for _, i, _ in pts]
        ys = [g for _, _, g in pts]
        x0 = min(min(xs) / 2, ceiling.ridge / 8)
        x1 = max(max(xs) * 2, ceiling.ridge * 4,
                 *(u.flops_per_s / ceiling.bytes_per_s * 2
                   for u in ceiling.units))
        top = max([ceiling.flops_per_s] + [u.flops_per_s
                                           for u in ceiling.units]) / 1e9
        ax = texout.Axis(options=[
            'tf', 'tf legend above', 'xmode=log', 'ymode=log',
            f'xmin={x0:.4g}, xmax={x1:.4g}',
            f'ymin={min(ys) / 3:.4g}, ymax={top * 2.2:.4g}',
            f'xlabel={{flop/byte ({_basis_label()})}}',
            f'ylabel={{GFLOP/s ({_basis_label()})}}'])
        _roof_lines(ax, ceiling, x0, x1)
        rows = []
        for k in kernels:
            ks = sorted((p.order, i, g, p) for p, i, g in pts
                        if p.workload == k)
            if not ks:
                continue
            # The ends only: the line says which way the order moves the
            # kernel, and a number at every point collides with the next.
            ax.plots.append(texout.Plot(
                styles(k), [(i, g) for _, i, g, _ in ks],
                texout.escape(short(k)),
                labels=[str(o) if n in (0, len(ks) - 1) else ''
                        for n, (o, _, _, _) in enumerate(ks)]))
            for o, i, g, p in ks:
                roof = ceiling.bound(i) / 1e9
                rows.append({'kernel': k, 'order': o, 'batch': p.batch,
                             'intensity': i, 'gflops': g, 'roof': roof,
                             'fraction': g / roof if roof else None,
                             'flops': flop_basis()})
        note = fallback_note([p.row for p, _, _ in pts])
        fell = _fell_back((p.row for p, _, _ in pts), 'event')
        comment = (f'generated by tools/bench/publish.py: {arch}, '
                   f'{_dt_name(dt)}, {eq}, {sims} simulation(s); device '
                   f'clock; the largest launch measured\n{_flops_note()}'
                   + (f'\n{note.strip()}' if note else '')
                   + (f'\n{fell} rows had no device clock and are on the '
                      f'wall clock' if fell else ''))
        tex = texout.picture([ax], comment=comment)
        caption = (f'Roofline of the {eq} kernels on '
                   f'{texout.escape(fieldset.device_of(arch))}, '
                   f'{_dt_name(dt)}'
                   + (f', {sims} fused simulations' if sims > 1 else '')
                   + '. Each line is one kernel through the orders it was '
                   'run at, the lowest and the highest at its ends, at the '
                   'largest launch measured; the roofs are measured, an FMA '
                   'chain and a triad'
                   + ''.join(f'; dashed, {texout.escape(u.name)}, '
                             + ('supplied rather than measured'
                                if u.source == 'supplied' else 'measured')
                             for u in ceiling.units)
                   + f'. {_flops_are()}')
        out.append(Output(f'roofline-{tag}', 'figure', tex, rows, caption))
    return out, skipped


def figure_peaks(fieldset: Field, kernels: Sequence[str]
                 ) -> Tuple[List[Output], List[str]]:
    """A full launch of each kernel at each order, three ways: as a
    fraction of the roof its intensity allows, as compulsory bandwidth
    against the triad, and as an arithmetic rate against the FMA chain.
    Three panels over one kernel axis rather than one chart with three
    scales."""
    out, skipped = [], []
    full = [p for p in _full(fieldset, kernels)
            if p.config == fieldset.reference]
    for arch, dt, eq, sims in _panels(full):
        if sims > 1:
            continue
        ceiling = fieldset.ceiling(arch, dt)
        tag = f'{arch}-{eq}-{dt.lower()}'
        if ceiling is None or not ceiling.ridge:
            skipped.append(f'peak-{tag}: no {dt} ceiling for {arch}')
            continue
        mine = [p for p in full if (p.arch, p.datatype, p.capture.equation,
                                    p.capture.simulations)
                == (arch, dt, eq, 1)]
        names = [k for k in kernels if any(p.workload == k for p in mine)]
        orders = sorted({p.order for p in mine})
        if not names or not orders:
            continue
        steps = texout.ordinal(len(orders))
        rows = []
        values: Dict[Tuple[str, int], Tuple[float, float, float]] = {}
        for p in mine:
            g, b, i = (_gflops(p.row, 'event'), _gbytes(p.row, 'event'),
                       _intensity(p.row))
            if not (g and b and i):
                continue
            frac = g * 1e9 / ceiling.bound(i)
            values[(p.workload, p.order)] = (frac, b, g)
            rows.append({'kernel': p.workload, 'order': p.order,
                         'batch': p.batch, 'fraction_of_roof': frac,
                         'gbytes': b, 'gflops': g, 'flops': flop_basis()})
        if not values:
            continue
        xticks = ','.join(str(i) for i in range(len(names)))
        labels = ','.join(r'\strut\texttt{' + texout.escape(short(k)) + '}'
                          for k in names)
        common = ['ybar=0pt', 'area legend',
                  f'bar width={min(0.8 / len(orders), 0.22):.3f}',
                  f'xtick={{{xticks}}}', 'xticklabels={}',
                  f'xmin=-0.6, xmax={len(names) - 0.4}',
                  'ymin=0', 'enlarge y limits={upper, value=0.12}',
                  'xmajorgrids=false', 'height=0.36\\tfwidth']
        metric = [
            ('fraction', r'\% of roof', 100.0, None),
            ('gbytes', 'GB/s', 1.0, ceiling.bytes_per_s / 1e9),
            ('gflops', f'TFLOP/s ({_basis_label()})', 1e-3,
             ceiling.flops_per_s / 1e12)]
        axes = []
        for j, (what, label, scale, rule) in enumerate(metric):
            ax = texout.Axis(options=list(common) + [f'ylabel={{{label}}}'])
            top = 0.0
            for n, o in enumerate(orders):
                pts = []
                for x, k in enumerate(names):
                    v = values.get((k, o))
                    val = None if v is None else v[j] * scale
                    if val:
                        top = max(top, val)
                    pts.append((x, val))
                ax.plots.append(texout.Plot(
                    f'fill=tfo{steps[n] + 1}, draw=none, mark=none',
                    pts, f'$O={o}$' if j == 0 else None))
            name = ('triad' if what == 'gbytes' else 'FMA chain') + ' ' \
                + texout.num(rule or 0) + ('\\,GB/s' if what == 'gbytes'
                                           else '\\,TFLOP/s')
            headroom = 'enlarge y limits={upper, value=0.12}'
            if rule and rule <= top * 1.5:
                # Up to the rule as well as the bars, which may stop short,
                # and room above it for its label.
                ax.options.remove(headroom)
                ax.options.append(f'ymax={max(top, rule) * 1.3:.4g}')
                ax.raw.append(texout.hrule(rule, name))
            elif rule:
                # Room above the bars for saying where the roof would be.
                ax.options[ax.options.index(headroom)] = \
                    'enlarge y limits={upper, value=0.35}'
                ax.raw.append(
                    '\\node[tf label, anchor=north east] at '
                    f'(rel axis cs:0.99,0.97) {{{name}, off scale}};\n')
            if j == 0:
                # Up to the roof whatever the bars reach: a fraction is
                # read against the whole.
                ax.options.remove(headroom)
                ax.options += ['tf legend above',
                               f'ymax={max(100.0, top) * 1.06:.4g}']
                ax.raw.append(texout.hrule(100))
            if j == len(metric) - 1:
                ax.options.remove('xticklabels={}')
                ax.options += [f'xticklabels={{{labels}}}',
                               'xticklabel style={rotate=30, anchor=north '
                               'east, inner sep=1pt}']
            axes.append(ax)
        tex = texout.picture(
            axes, 'group size=1 by 3, vertical sep=0.25cm, '
                  'x descriptions at=edge bottom',
            comment=(f'generated by tools/bench/publish.py: {arch}, '
                     f'{_dt_name(dt)}, {eq}; the largest launch, device '
                     f'clock\n{_flops_note()}\n'
                     + fallback_note([p.row for p in mine]).strip()))
        caption = (f'A full launch of each {eq} kernel on '
                   f'{texout.escape(fieldset.device_of(arch))}, '
                   f'{_dt_name(dt)}, per order: the fraction of the roof '
                   'its intensity allows, the compulsory bandwidth against '
                   'the measured triad, and the arithmetic rate against the '
                   f'measured FMA chain. {_flops_are()}')
        out.append(Output(f'peak-{tag}', 'figure', tex, rows, caption))
    return out, skipped


# ----------------------------------------------------------------------
# Ramp-up
# ----------------------------------------------------------------------

@dataclass
class Fit:
    """`t(n) = t0 + c n`, least squares over the measured batches: the
    launch's fixed cost and the per-element cost a large launch tends to.
    Their ratio is Hockney's n-half, the batch at which a launch reaches
    half its asymptotic rate.  `measured` is the smallest batch that reached
    half the best measured rate, read off the points rather than the line,
    and `residual` the furthest a point lies from the line, relative to its
    time: how far the line is to be trusted."""
    t0_ns: float
    c_ns: float
    half: Optional[float]
    measured: Optional[int]
    residual: float = 0.0


def launch_fit(points: Sequence[Tuple[int, float]]) -> Optional[Fit]:
    """`points` are (batch, nanoseconds per launch).  Weighted by the
    inverse square of the time, so that a small launch -- whose time is the
    launch cost the fit is for -- is not drowned by a large one's."""
    pts = [(n, t) for n, t in points if n and t]
    if len(pts) < 3:
        return None
    w = [1.0 / (t * t) for _, t in pts]
    sw = sum(w)
    sx = sum(wi * n for wi, (n, _) in zip(w, pts))
    sy = sum(wi * t for wi, (_, t) in zip(w, pts))
    sxx = sum(wi * n * n for wi, (n, _) in zip(w, pts))
    sxy = sum(wi * n * t for wi, (n, t) in zip(w, pts))
    det = sw * sxx - sx * sx
    if det <= 0:
        return None
    c = (sw * sxy - sx * sy) / det
    t0 = (sy - c * sx) / sw
    if c <= 0:
        return None
    best = max(n / t for n, t in pts)
    measured = next((n for n, t in sorted(pts) if n / t >= 0.5 * best), None)
    residual = max(abs(t - t0 - c * n) / t for n, t in pts)
    return Fit(t0_ns=t0, c_ns=c, half=(t0 / c if t0 > 0 else None),
               measured=measured, residual=residual)


def _by_batch(points: Iterable[Point], clock: str
              ) -> Dict[str, Dict[int, Dict]]:
    """Each kernel's rows by batch, where a row has a time."""
    out: Dict[str, Dict[int, Dict]] = defaultdict(dict)
    for p in points:
        if _nanos(p.row, clock):
            out[p.workload][p.batch] = p.row
    return out


#: One time step of n elements: (n, nanoseconds, flops), per order and per
#: reading (`step_series`).
StepSeries = Dict[int, Dict[bool, List[Tuple[int, float, float]]]]


def step_series(at: Dict[str, Dict[int, Dict]], step: Dict[str, Launch],
                clock: str, together: bool
                ) -> List[Tuple[int, float, float]]:
    """One time step of `n` elements (`STEP`), at every `n` for which each
    of its launches was measured at the batch it carries: `(n, nanoseconds,
    flops)`.  With `together`, the launches SeisSol puts on streams of their
    own are taken as one launch of all their elements -- the most the
    streams can give; without, they run one after another -- the least.
    A step one of whose launches was not measured at its batch is left out,
    not filled in."""
    if not step or any(k not in at for k in step):
        return []
    out = []
    for n in sorted({b for rows in at.values() for b in rows}):
        t = f = 0.0
        for k, launch in step.items():
            group = launch.together if together else 1
            batch = launch.share * group * n
            row = (at[k].get(int(batch)) if float(batch).is_integer()
                   else None)
            if row is None:
                break
            launches = launch.count / group
            t += launches * _nanos(row, clock)
            f += launches * (flops_of(row) or 0.0)
        else:
            out.append((n, t, f))
    return out


#: What the ramp-up table calls the two readings of a time step.
STEP_ROWS = ('time step', 'time step, serial')

#: The sweep a ramp-up wants (`run.py --batches`, `evaluate.py --batches`):
#: every power of two from 16 elements, where a launch is all overhead, to
#: 2^18, a large partition -- so that each batch has the one a quarter of
#: it, which a neighbour-flux launch carries in a step of the batch
#: (`STEP`).
RAMPUP_BATCHES = tuple(2 ** e for e in range(4, 19))


def _fit_row(arch: str, dt: str, eq: str, order: int, kernel: str,
             clock: str, times: Sequence[Tuple[int, float]],
             best: Optional[float]) -> Dict:
    fit = launch_fit(times)
    return {'arch': arch, 'datatype': dt, 'equation': eq, 'order': order,
            'kernel': kernel, 'clock': clock,
            't0_us': fit.t0_ns / 1e3 if fit else None,
            'ns_per_element': fit.c_ns if fit else None,
            'half_fit': fit.half if fit else None,
            'half_measured': fit.measured if fit else None,
            'fit_residual': fit.residual if fit else None,
            'gflops_best': best}


def _rampup_axes(xlabel: str) -> Tuple[texout.Axis, texout.Axis]:
    """Above, the rate; below, the rate as a fraction of the best one --
    the form a knee is read off, and the same for every machine."""
    top = texout.Axis(options=[
        'tf legend above', 'xmode=log', 'log basis x=2', 'ymode=log',
        f'ylabel={{GFLOP/s ({_basis_label()})}}',
        'xticklabels={}', 'height=0.5\\tfwidth'])
    low = texout.Axis(options=[
        'xmode=log', 'log basis x=2', 'ymin=0, ymax=1.08',
        'ytick={0,0.25,0.5,0.75,1}',
        'yticklabels={0,,50\\,\\%,,100\\,\\%}',
        f'xlabel={{{xlabel}}}', 'ylabel={of the best}',
        'height=0.32\\tfwidth'])
    low.raw.append(texout.hrule(0.5))
    return top, low


def _rampup_pair(top: texout.Axis, low: texout.Axis, style: str,
                 rates: Sequence[Tuple[int, Optional[float]]],
                 legend: Optional[str]) -> Optional[float]:
    best = max((g for _, g in rates if g), default=None)
    if best is None:
        return None
    top.plots.append(texout.Plot(style, rates, legend))
    low.plots.append(texout.Plot(style + ', forget plot',
                                 [(b, g / best if g else None)
                                  for b, g in rates]))
    return best


def _clock_sentence(clock: str) -> str:
    return ('Wall clock over back-to-back launches, so the launch overhead '
            'is in it.' if clock == 'wall' else
            'Device clock, which leaves the launch overhead out.')


def figure_rampup(fieldset: Field, kernels: Sequence[str], styles: Styles,
                  clock: str, order: int
                  ) -> Tuple[List[Output], List[Output], List[str]]:
    """The rate against the launch size, per machine, precision and
    equation: every kernel at one order with the time step they make
    (`STEP`), and the time step at every order -- how the knee moves with
    the order.  A fit of `t0 + c n` per kernel, order and time step goes
    into a table per machine."""
    figures: List[Output] = []
    skipped: List[str] = []
    ref = fieldset.reference
    pts = [p for p in fieldset.points
           if p.workload in kernels and p.capture is not None
           and p.config == ref and p.capture.simulations == 1]
    fit_rows: List[Dict] = []
    for arch, dt, eq in sorted({(p.arch, p.datatype, p.capture.equation)
                                for p in pts}):
        tag = f'{arch}-{eq}-{dt.lower()}'
        step = STEP.get(eq, {})
        mine = [p for p in pts if (p.arch, p.datatype, p.capture.equation)
                == (arch, dt, eq)]
        swept: Dict[int, Dict[str, Dict[int, Dict]]] = {}
        steps: StepSeries = {}
        for o in sorted({p.order for p in mine}):
            at = {k: v for k, v in _by_batch(
                (p for p in mine if p.order == o), clock).items()
                if len(v) > 2}
            if not at:
                continue
            swept[o] = at
            for k in kernels:
                if k in at:
                    seq = sorted(at[k].items())
                    fit_rows.append(_fit_row(
                        arch, dt, eq, o, k, clock,
                        [(b, _nanos(r, clock)) for b, r in seq],
                        max((_gflops(r, clock) or 0) for _, r in seq)))
            steps[o] = {t: step_series(at, step, clock, t)
                        for t in (True, False)}
            for together, name in zip((True, False), STEP_ROWS):
                seq = steps[o][together]
                if len(seq) > 2:
                    fit_rows.append(_fit_row(
                        arch, dt, eq, o, name, clock,
                        [(n, t) for n, t, _ in seq],
                        max(f / t for _, t, f in seq)))
        if not swept:
            skipped.append(f'rampup-{tag}: fewer than three batches -- '
                           'run with --batches '
                           + ','.join(map(str, RAMPUP_BATCHES)))
            continue
        shown = min(swept, key=lambda o: (abs(o - order), o))
        figures.append(_rampup_kernels(fieldset, kernels, styles, clock,
                                       arch, dt, eq, shown, swept[shown],
                                       steps[shown], step))
        made = _rampup_orders(fieldset, clock, arch, dt, eq, steps, step)
        if made is not None:
            figures.append(made)
        elif step:
            skipped.append(f'rampup-{tag}: the time step needs each of '
                           f'{", ".join(map(short, step))} swept at one '
                           'order')
        else:
            skipped.append(f'rampup-{tag}: no time step is written down '
                           f'for {eq} (STEP)')
    tables = [_rampup_table(fit_rows, fieldset, arch, kernels)
              for arch in sorted({r['arch'] for r in fit_rows})]
    return figures, tables, skipped


def _rampup_kernels(fieldset: Field, kernels: Sequence[str], styles: Styles,
                    clock: str, arch: str, dt: str, eq: str, order: int,
                    at: Dict[str, Dict[int, Dict]],
                    steps: Dict[bool, List[Tuple[int, float, float]]],
                    step: Dict[str, Launch]) -> Output:
    top, low = _rampup_axes('elements $n$')
    rows = []
    for k in kernels:
        if k not in at:
            continue
        seq = sorted(at[k].items())
        _rampup_pair(top, low, styles(k),
                     [(b, _gflops(r, clock)) for b, r in seq],
                     texout.escape(short(k)))
        rows += [{'kernel': k, 'batch': b, 'gflops': _gflops(r, clock),
                  'ns_per_element': _nanos(r, clock) / b} for b, r in seq]
    for together, name, dash in zip((True, False), STEP_ROWS,
                                    ('', ', densely dashed')):
        seq = steps.get(together) or []
        if len(seq) < 2 or (not together and seq == steps.get(True)):
            continue
        _rampup_pair(top, low, f'tfink, line width=1.2pt, mark=none{dash}',
                     [(n, f / t) for n, t, f in seq], name)
        rows += [{'kernel': name, 'batch': n, 'gflops': f / t,
                  'ns_per_element': t / n} for n, t, f in seq]
    tex = texout.picture(
        [top, low], 'group size=1 by 2, vertical sep=0.25cm, '
                    'x descriptions at=edge bottom',
        comment=(f'generated by tools/bench/publish.py: {arch}, '
                 f'{_dt_name(dt)}, {eq}, order {order}; {clock} clock\n'
                 f'{_flops_note()}\nthe time step: {step_text(step)}'))
    caption = (f'Ramp-up of the {eq} kernels at order {order} on '
               f'{texout.escape(fieldset.device_of(arch))}, '
               f'{_dt_name(dt)}: the rate against the elements one launch '
               'carries, and below it as a fraction of the best. The black '
               'lines are a time step of $n$ elements as SeisSol launches '
               'it, the neighbour flux once per face and face relation: '
               'solid with the relations of a face, which go to streams of '
               'their own, taken as one launch, dashed with them one after '
               f'another. {_clock_sentence(clock)}')
    return Output(f'rampup-{arch}-{eq}-{dt.lower()}-o{order}', 'figure',
                  tex, rows, caption)


def _rampup_orders(fieldset: Field, clock: str, arch: str, dt: str,
                   eq: str, steps: StepSeries, step: Dict[str, Launch]
                   ) -> Optional[Output]:
    orders = [o for o in sorted(steps) if len(steps[o][True]) > 2]
    if not orders:
        return None
    top, low = _rampup_axes('elements per time step $n$')
    shades = texout.ordinal(len(orders))
    rows = []
    for o, shade in zip(orders, shades):
        seq = steps[o][True]
        _rampup_pair(top, low, f'color=tfo{shade + 1}, mark=*, '
                               'mark size=1.2pt',
                     [(n, f / t) for n, t, f in seq], f'$O={o}$')
        rows += [{'order': o, 'batch': n, 'gflops': f / t,
                  'ns_per_element': t / n} for n, t, f in seq]
    tex = texout.picture(
        [top, low], 'group size=1 by 2, vertical sep=0.25cm, '
                    'x descriptions at=edge bottom',
        comment=(f'generated by tools/bench/publish.py: {arch}, '
                 f'{_dt_name(dt)}, {eq}; {clock} clock\n{_flops_note()}\n'
                 f'the time step: {step_text(step)}, the launches on '
                 'streams of their own taken as one'))
    caption = (f'Ramp-up of one {eq} time step on '
               f'{texout.escape(fieldset.device_of(arch))}, {_dt_name(dt)}, '
               'per order: the rate of a step of $n$ elements, and below it '
               'as a fraction of the best, with the relations of a face, '
               'which SeisSol puts on streams of their own, taken as one '
               'launch -- the most the streams can give. '
               f'{_clock_sentence(clock)}')
    return Output(f'rampup-{arch}-{eq}-{dt.lower()}', 'figure', tex, rows,
                  caption)


def _rampup_table(rows: List[Dict], fieldset: Field, arch: str,
                  kernels: Sequence[str]) -> Output:
    """One machine: per equation and precision, the fixed cost of a launch
    and the batch at which it reaches half its asymptotic rate, per kernel
    and order, and the same for the time step read both ways."""
    mine = [r for r in rows if r['arch'] == arch]
    orders = sorted({r['order'] for r in mine})
    names = [k for k in kernels if any(r['kernel'] == k for r in mine)]
    names += [n for n in STEP_ROWS if any(r['kernel'] == n for r in mine)]
    body: List = []
    width = 2 + len(orders)
    for eq, dt in sorted({(r['equation'], r['datatype']) for r in mine}):
        if body:
            body.append(None)
        body.append([f'\\multicolumn{{{width}}}{{@{{}}l}}{{\\textit{{'
                     + texout.escape(eq) + f'}}, {_dt_name(dt)}}}'])
        for name in names:
            cells = {r['order']: r for r in mine
                     if (r['equation'], r['datatype'], r['kernel'])
                     == (eq, dt, name)}
            if not cells:
                continue
            if name == STEP_ROWS[0]:
                body.append(texout.GAP)
            t0 = [c['t0_us'] for c in cells.values()
                  if c['t0_us'] is not None]
            label = (texout.escape(name) if name in STEP_ROWS
                     else r'\texttt{' + texout.escape(short(name)) + '}')
            lo, hi = ((texout.num(min(t0)), texout.num(max(t0))) if t0
                      else ('--', '--'))
            body.append(
                [label, lo if lo == hi else f'{lo}--{hi}']
                + [texout.num(cells[o]['half_fit'])
                   if o in cells and cells[o]['half_fit'] else '--'
                   for o in orders])
    tex = texout.table(
        'l r ' + 'r ' * len(orders),
        [['', '', f'\\multicolumn{{{len(orders)}}}{{c}}{{$n_{{1/2}}$ at '
                  'order $O$}'],
         f'\\cmidrule(l){{3-{width}}}',
         # `\mu` and not `\textmu`: the text one comes from a bitmap font
         # where the document has no Type 1 one for it, and a Type 3 font
         # in a PDF is what a publisher's checker turns away.
         ['Kernel', r'$t_0$ [$\mu$s]'] + [str(o) for o in orders]],
        body,
        comment=('generated by tools/bench/publish.py: t(n) = t0 + c n, '
                 'least squares weighted by 1/t^2; t0 from the lowest to the '
                 'highest over the orders,\nn_1/2 = t0/c; the smallest '
                 'measured batch reaching half the best rate and the worst '
                 'relative residual are in the CSV'))
    clock = mine[0]['clock'] if mine else 'wall'
    caption = (f'The cost of a launch on '
               f'{texout.escape(fieldset.device_of(arch))}: its fixed part '
               '$t_0$, from the lowest to the highest over the orders, and '
               'the elements $n_{1/2}=t_0/c$ at which a launch reaches half '
               'its asymptotic rate, from $t(n)=t_0+cn$ fitted to the '
               'measured times; for the time step with the relations of a '
               'face taken as one launch and one after another. '
               f'{_clock_sentence(clock)}')
    return Output(f'rampup-{arch}', 'table', tex, mine, caption)


def figure_breakdown(fieldset: Field, kernels: Sequence[str], styles: Styles,
                     clock: str = 'event'
                     ) -> Tuple[List[Output], List[str]]:
    """Where one time step's time goes, per order: the kernels' time per
    element at a full launch, stacked and weighed by how many times an
    element passes through each (`STEP`)."""
    out, skipped = [], []
    full = [p for p in _full(fieldset, kernels)
            if p.config == fieldset.reference and p.capture.simulations == 1]
    for arch, dt, eq in sorted({(p.arch, p.datatype, p.capture.equation)
                                for p in full}):
        step = STEP.get(eq)
        if not step:
            skipped.append(f'breakdown-{arch}-{eq}-{dt.lower()}: no time '
                           f'step is written down for {eq} (STEP)')
            continue
        mine = {(p.workload, p.order): p for p in full
                if (p.arch, p.datatype, p.capture.equation) == (arch, dt, eq)}
        orders = sorted({o for (_, o) in mine
                         if all((k, o) in mine for k in step)})
        tag = f'{arch}-{eq}-{dt.lower()}'
        if not orders:
            skipped.append(f'breakdown-{tag}: no order has every kernel of '
                           f'the step ({", ".join(step)})')
            continue
        ax = texout.Axis(options=[
            'tf', 'ybar stacked', 'bar width=0.5',
            f'xtick={{{",".join(map(str, orders))}}}',
            f'xmin={orders[0] - 0.7}, xmax={orders[-1] + 0.7}',
            'xmajorgrids=false', 'ymin=0',
            'xlabel={order $O$}', 'ylabel={ns per element and step}',
            'tf legend above', 'reverse legend', 'area legend'])
        rows = []
        for k, launch in step.items():
            w = launch.per_element
            pts = []
            for o in orders:
                p = mine[(k, o)]
                ns = _nanos(p.row, clock) / p.batch * w
                pts.append((o, ns))
                rows.append({'kernel': k, 'order': o, 'weight': w,
                             'batch': p.batch, 'ns_per_element': ns})
            label = texout.escape(short(k)) + (f' $\\times{w:g}$'
                                               if w != 1 else '')
            ax.plots.append(texout.Plot(
                f'fill={styles.fill(k)}, draw=white, line width=0.6pt, '
                'mark=none', pts, label))
        tex = texout.picture([ax], comment=(
            f'generated by tools/bench/publish.py: {arch}, {_dt_name(dt)}, '
            f'{eq}; {clock} clock at the largest launch\nthe time step: '
            f'{step_text(step)}'))
        caption = (f'One time step of the {eq} system on '
                   f'{texout.escape(fieldset.device_of(arch))}, '
                   f'{_dt_name(dt)}, per element and order: each kernel\'s '
                   'time per element at a full launch, times how many times '
                   'an element passes through it in a step of global time '
                   'stepping -- the neighbour flux once per face.')
        out.append(Output(f'breakdown-{tag}', 'figure', tex, rows, caption))
    return out, skipped


# ----------------------------------------------------------------------
# Portability and configurations
# ----------------------------------------------------------------------

@dataclass
class Efficiencies:
    """The fraction of its own roof each machine's full launch of each
    kernel reached, at the kernel's intensity: (equation, datatype, kernel,
    order) -> {arch: fraction}.  Beside it, what was tried and produced no
    number, and the machines each (equation, datatype) is over -- those
    with a roof in the precision that ran the equation in it."""
    eff: Dict[Tuple, Dict[str, float]]
    failed: set
    platforms: Dict[Tuple[str, str], set]


def efficiencies(fieldset: Field, kernels: Sequence[str]) -> Efficiencies:
    ref = fieldset.reference
    full = [p for p in _full(fieldset, kernels)
            if p.config == ref and p.capture.simulations == 1]
    tried = [p for p in fieldset.failed
             if p.workload in kernels and p.capture is not None
             and p.config == ref and p.capture.simulations == 1]
    eff: Dict[Tuple, Dict[str, float]] = defaultdict(dict)
    for p in full:
        ceiling = fieldset.ceiling(p.arch, p.datatype)
        g, i = _gflops(p.row, 'event'), _intensity(p.row)
        if ceiling is None or not g or not i:
            continue
        eff[(p.capture.equation, p.datatype, p.workload, p.order)][
            p.arch] = g * 1e9 / ceiling.bound(i)
    failed = {(p.capture.equation, p.datatype, p.workload, p.order, p.arch)
              for p in tried}
    platforms: Dict[Tuple[str, str], set] = defaultdict(set)
    for p in full + tried:
        if fieldset.ceiling(p.arch, p.datatype):
            platforms[(p.capture.equation, p.datatype)].add(p.arch)
    return Efficiencies(eff, failed, platforms)


def table_portability(fieldset: Field, kernels: Sequence[str]
                      ) -> Optional[Output]:
    """Pennycook's performance portability per kernel, order and precision:
    the harmonic mean over the machines of the fraction of each one's own
    roof a full launch reaches at the kernel's intensity (`efficiencies`).
    One of the machines that tried the kernel and produced no number makes
    it zero, as the definition has it, and one that was not asked leaves it
    open -- a gap in a campaign is not a failure of the code."""
    found = efficiencies(fieldset, kernels)
    eff, failed, platforms = found.eff, found.failed, found.platforms
    groups = sorted(g for g, archs in platforms.items() if len(archs) > 1)
    if not groups:
        return None
    orders = sorted({key[3] for key in list(eff) + list(failed)
                     if key[:2] in groups})
    every = sorted(set().union(*(platforms[g] for g in groups)))
    body: List = []
    rows_csv: List[Dict] = []
    for eq, dt in groups:
        machines = sorted(platforms[(eq, dt)])
        if body:
            body.append(None)
        body.append([f'\\multicolumn{{{1 + len(orders)}}}{{@{{}}l}}{{'
                     f'\\textit{{{texout.escape(eq)}}}, {_dt_name(dt)}'
                     + ('' if machines == every else ', over ' + ', '.join(
                         r'\texttt{' + texout.escape(a) + '}'
                         for a in machines)) + '}'])
        for k in kernels:
            cells, seen = [], False
            for o in orders:
                e = eff.get((eq, dt, k, o), {})
                broke = [a for a in machines
                         if a not in e and (eq, dt, k, o, a) in failed]
                if not e and not broke:
                    cells.append('--')
                    continue
                seen = True
                absent = [a for a in machines if a not in e and a not in broke]
                if broke or any(e[a] <= 0 for a in e):
                    pp: Optional[float] = 0.0
                elif absent:
                    pp = None
                else:
                    pp = len(machines) / sum(1.0 / e[a] for a in machines)
                cells.append('--' if pp is None
                             else texout.num(100 * pp, 2) if pp else '0')
                rows_csv.append(dict(
                    {'equation': eq, 'datatype': dt, 'kernel': k,
                     'order': o, 'pp': pp, 'machines': ' '.join(machines),
                     'failed_on': ' '.join(broke),
                     'not_run_on': ' '.join(absent)},
                    **{f'efficiency_{a}': e.get(a) for a in machines}))
            if seen:
                body.append([r'\texttt{' + texout.escape(short(k)) + '}']
                            + cells)
    tex = texout.table(
        'l ' + 'r ' * len(orders),
        [['', f'\\multicolumn{{{len(orders)}}}{{c}}{{$\\mathcal{{P}}$ '
              '[\\%] at order $O$}'],
         f'\\cmidrule(l){{2-{1 + len(orders)}}}',
         ['Kernel'] + [str(o) for o in orders]],
        body,
        comment=('generated by tools/bench/publish.py: the harmonic mean '
                 'over the machines of the fraction of each one\'s measured '
                 'roof\nat the kernel\'s intensity, largest launch, device '
                 'clock; 0 where a machine produced no number, -- where one '
                 f'was not asked\n{_flops_note()}\nthe efficiency on each '
                 'machine is in the CSV'))
    caption = ('Performance portability $\\mathcal{P}$ (Pennycook et al.) '
               'of each kernel over '
               + ', '.join(texout.escape(fieldset.device_of(a))
                           for a in every)
               + ', in percent: the harmonic mean of the fraction of each '
               'machine\'s own measured roof a full launch reaches at the '
               'kernel\'s intensity. Zero where a machine built or ran the '
               'kernel and produced no number; a dash where one was not '
               f'asked to. {_flops_are()}')
    return Output('portability', 'table', tex, rows_csv, caption)


def figure_configs(fieldset: Field, kernels: Sequence[str]
                   ) -> List[Output]:
    """Every configuration -- or a run of another generator brought in
    under its own configuration name -- against the reference, per kernel
    and order: the ratio of the times per element at a full launch on one
    machine, in one precision, right of one where the candidate is faster.
    A row per kernel and order, so that the names are read across rather
    than tilted, and a logarithmic axis, so that twice as fast and twice as
    slow are as far from one."""
    configs = fieldset.configs
    ref = fieldset.reference
    if len(configs) < 2 or ref is None:
        return []
    others = [c for c in configs if c != ref]
    full = [p for p in _full(fieldset, kernels)
            if p.capture.simulations == 1]
    out = []
    for arch, dt, eq in sorted({(p.arch, p.datatype, p.capture.equation)
                                for p in full}):
        mine = {(p.workload, p.order, p.config): p for p in full
                if (p.arch, p.datatype, p.capture.equation) == (arch, dt, eq)}
        groups = sorted({(k, o) for (k, o, _) in mine},
                        key=lambda t: (kernels.index(t[0]), t[1]))
        groups = [g for g in groups if (g[0], g[1], ref) in mine
                  and any((g[0], g[1], c) in mine for c in others)]
        if not groups:
            continue

        def per_element(p: Optional[Point]) -> Optional[float]:
            # Per element: two configurations' largest launches need not
            # be the same size.
            return _per_element(p.row, p.batch, 'event') if p else None

        rows, plots, ratios = [], [], []
        for n, c in enumerate(others):
            pts = []
            for y, (k, o) in enumerate(groups):
                tb = per_element(mine.get((k, o, ref)))
                tc = per_element(mine.get((k, o, c)))
                r = tb / tc if tb and tc else None
                pts.append((r, y))
                if r:
                    ratios.append(r)
                    rows.append({'kernel': k, 'order': o, 'config': c,
                                 'reference': ref, 'speedup': r})
            plots.append(texout.Plot(f'tf s{n + 1}, only marks', pts,
                                     r'\texttt{' + texout.escape(c) + '}'))
        if not ratios:
            continue
        ax = _dot_axis(groups, ratios, 'speed-up over \\texttt{'
                       + texout.escape(ref) + '}')
        ax.plots = plots
        tex = texout.picture([ax], comment=(
            f'generated by tools/bench/publish.py: {arch}, {_dt_name(dt)}, '
            f'{eq}; reference time over candidate time per element, device '
            f'clock, largest launch'))
        name = r'\texttt{' + texout.escape(ref) + '}'
        caption = (f'Each configuration against {name} on '
                   f'{texout.escape(fieldset.device_of(arch))}, '
                   f'{_dt_name(dt)}, {eq}: the reference\'s time per element '
                   'over the candidate\'s at a full launch, per kernel and '
                   'order; right of one is faster.')
        out.append(Output(f'configs-{arch}-{eq}-{dt.lower()}', 'figure', tex,
                          rows, caption))
    return out


# ----------------------------------------------------------------------
# A row per kernel and order
# ----------------------------------------------------------------------

def _ratio_axis_range(values: Sequence[float], rule: float = 1.0
                      ) -> Tuple[float, float, List[float]]:
    """A logarithmic axis for ratios: from the power of two below the
    smallest to the one above the largest, the rule included."""
    values = [v for v in values if v and v > 0] + [rule]
    lo = 2.0 ** math.floor(math.log2(min(values) / 1.05))
    hi = 2.0 ** math.ceil(math.log2(max(values) * 1.05))
    ticks = [2.0 ** e for e in range(int(round(math.log2(lo))),
                                     int(round(math.log2(hi))) + 1)]
    return lo, hi, ticks


def _dot_axis(groups: Sequence[Tuple[str, int]], values: Sequence[float],
              xlabel: str, rule: Optional[float] = 1.0) -> texout.Axis:
    """A row per kernel and order, so that the names are read across
    rather than tilted, on a logarithmic axis of ratios -- twice and half
    are as far from one -- with the rule drawn and a hairline between one
    kernel's orders and the next's."""
    lo, hi, ticks = _ratio_axis_range(values, rule or 1.0)
    labels = [(r'\texttt{' + texout.escape(short(k)) + '} '
               if i == 0 or groups[i - 1][0] != k else '') + f'{o}'
              for i, (k, o) in enumerate(groups)]
    ax = texout.Axis(options=[
        'tf', 'tf legend above', 'xmode=log', 'log basis x=2',
        'y dir=reverse', f'xmin={lo:g}, xmax={hi:g}',
        f'xtick={{{",".join(f"{t:g}" for t in ticks)}}}',
        f'xticklabels={{{",".join(f"{t:g}" for t in ticks)}}}',
        f'ytick={{{",".join(str(i) for i in range(len(groups)))}}}',
        'yticklabels={' + ','.join(labels) + '}',
        'yticklabel style={font=\\scriptsize, color=tfink}',
        f'ymin=-0.7, ymax={len(groups) - 0.3}', 'ymajorgrids=false',
        f'height={max(0.3, 0.12 + 0.05 * len(groups)):.2f}\\tfwidth',
        f'xlabel={{{xlabel}}}'])
    if rule:
        ax.raw.append(f'\\draw[tf rule] ({{axis cs:{rule:g},0}}|-{{rel axis '
                      f'cs:0,0}}) -- ({{axis cs:{rule:g},0}}|-{{rel axis '
                      f'cs:0,1}});\n')
    for i in range(1, len(groups)):
        if groups[i][0] != groups[i - 1][0]:
            ax.raw.append(
                f'\\draw[tfgrid] ({{rel axis cs:0,0}}|-{{axis cs:1,'
                f'{i - 0.5}}}) -- ({{rel axis cs:1,0}}|-{{axis cs:1,'
                f'{i - 0.5}}});\n')
    return ax


def _per_element(row: Optional[Dict], batch: int, clock: str
                 ) -> Optional[float]:
    ns = _nanos(row, clock) if row else None
    return ns / batch if ns and batch else None


# ----------------------------------------------------------------------
# What fusing a chain is worth
# ----------------------------------------------------------------------

def figure_fusion(fieldset: Field, kernels: Sequence[str], clock: str
                  ) -> Tuple[List[Output], List[str]]:
    """Each kernel against its pieces -- one launch per operation, the
    temporaries through memory (`suites/seissolsplit.py`) -- added up per
    batch: the pieces' time over the whole kernel's, both from one run and
    one build, at the largest batch and at the smallest, where the pieces'
    extra launches weigh most.  Beside them, what the pieces move against
    what the kernel moves, compulsorily.

    The whole kernel is its own single piece (`gpu_volume~0-of-1`), and
    only that one counts: `seissolbench` times the kernel too, but in
    another run, which is another clock."""
    ref = fieldset.reference
    pieces: Dict[Tuple, Dict[int, Dict[int, Dict]]] = defaultdict(
        lambda: defaultdict(dict))
    counts: Dict[Tuple, int] = {}
    wholes: Dict[Tuple, Dict[int, Dict]] = defaultdict(dict)
    for p in fieldset.points:
        if (p.config != ref or p.capture is None
                or p.capture.simulations != 1):
            continue
        piece = Piece.of(p.workload)
        if piece is None or piece.kernel not in kernels:
            continue
        # The run and the binary: within them the pieces and the whole kernel
        # were timed in the same passes.
        key = (p.arch, p.datatype, p.capture, piece.kernel, p.run,
               p.row.get('unit'))
        if piece.count == 1:
            wholes[key][p.batch] = p.row
        else:
            pieces[key][p.batch][piece.index] = p.row
            counts[key] = piece.count
    if not pieces:
        return [], ['fusion: no kernel measured in pieces '
                    '(suites/seissolsplit.py)']
    out, skipped = [], []
    for arch, dt, eq in sorted({key[:2] + (key[2].equation,)
                                for key in pieces}):
        entries = []
        for key, by_batch in pieces.items():
            a, d, capture, kernel = key[:4]
            if (a, d, capture.equation) != (arch, dt, eq):
                continue
            whole = wholes.get(key, {})
            complete = sorted(
                b for b, rows in by_batch.items()
                if len(rows) == counts[key] and b in whole
                and _nanos(whole[b], clock)
                and all(_nanos(r, clock) for r in rows.values()))
            if not complete:
                continue

            def ratio(b: int) -> float:
                return (sum(_nanos(r, clock) for r in by_batch[b].values())
                        / _nanos(whole[b], clock))

            big, small = complete[-1], complete[0]
            moved = sum(r.get('bytes') or 0 for r in by_batch[big].values())
            entries.append({
                'kernel': kernel, 'order': capture.order,
                'pieces': counts[key], 'batch_large': big,
                'ratio_large': ratio(big), 'batch_small': small,
                'ratio_small': ratio(small) if small != big else None,
                'bytes_ratio': (moved / whole[big]['bytes']
                                if whole[big].get('bytes') else None),
                'clock': clock})
        if not entries:
            skipped.append(f'fusion-{arch}-{eq}-{dt.lower()}: pieces without '
                           'their whole kernel in the same run '
                           '(suites/seissolsplit.py times both)')
            continue
        entries.sort(key=lambda e: (kernels.index(e['kernel']), e['order']))
        groups = [(e['kernel'], e['order']) for e in entries]
        values = [v for e in entries for v in (e['ratio_large'],
                                               e['ratio_small'],
                                               e['bytes_ratio']) if v]
        ax = _dot_axis(groups, values,
                       'pieces over the fused kernel')
        ax.plots = [
            texout.Plot('tf s1, only marks',
                        [(e['ratio_large'], y)
                         for y, e in enumerate(entries)],
                        'time, largest launch'),
            texout.Plot('tf s2, only marks',
                        [(e['ratio_small'], y)
                         for y, e in enumerate(entries)],
                        'time, smallest launch'),
            texout.Plot('tfmuted, only marks, mark=|, mark size=3.5pt, '
                        'line width=1pt',
                        [(e['bytes_ratio'], y)
                         for y, e in enumerate(entries)],
                        'compulsory bytes')]
        tex = texout.picture([ax], comment=(
            f'generated by tools/bench/publish.py: {arch}, {_dt_name(dt)}, '
            f'{eq}; the pieces\' {clock}-clock times summed per batch over '
            'the whole kernel\'s, both from one run; batches in the CSV'))
        caption = (f'What fusion is worth on '
                   f'{texout.escape(fieldset.device_of(arch))}, '
                   f'{_dt_name(dt)}, {eq}: each kernel taken apart into one '
                   'launch per operation, its temporaries through global '
                   'memory, built by the same generator and timed in the '
                   'same run -- the pieces\' time over the whole kernel\'s '
                   'at the largest and the '
                   'smallest launch measured, and the bytes they move '
                   f'against the kernel\'s. {_clock_sentence(clock)}')
        out.append(Output(f'fusion-{arch}-{eq}-{dt.lower()}', 'figure', tex,
                          entries, caption))
    return out, skipped


# ----------------------------------------------------------------------
# Portability as a cascade
# ----------------------------------------------------------------------

def figure_cascade(fieldset: Field, kernels: Sequence[str], order: int,
                   styles: Styles) -> List[Output]:
    """Sewall et al.'s cascade, per equation and precision at one order:
    each kernel's efficiency on every machine, most efficient first; the
    performance portability of the first so many of them, as machines are
    added in that order; and which machine stands at which place.  The
    table gives the number; this shows what it is made of."""
    found = efficiencies(fieldset, kernels)
    out = []
    for (eq, dt), machines in sorted(found.platforms.items()):
        machines = sorted(machines)
        if len(machines) < 2:
            continue
        orders = sorted({o for (e, d, _, o) in found.eff
                         if (e, d) == (eq, dt)})
        if not orders:
            continue
        o = min(orders, key=lambda x: (abs(x - order), x))
        series = []
        for k in kernels:
            e = found.eff.get((eq, dt, k, o), {})
            values = [(a, e[a]) for a in machines if a in e]
            values += [(a, 0.0) for a in machines if a not in e
                       and (eq, dt, k, o, a) in found.failed]
            if len(values) < 2:
                continue
            values.sort(key=lambda t: (-t[1], t[0]))
            series.append((k, values))
        if not series:
            continue
        n = len(machines)
        ticks = ','.join(str(i) for i in range(1, n + 1))
        common = [f'xmin=0.5, xmax={n + 0.5}', f'xtick={{{ticks}}}',
                  'xmajorgrids=false']
        top = texout.Axis(options=common + [
            'tf legend above', 'ymin=0', 'xticklabels={}',
            'ylabel={\\% of own roof}', 'height=0.42\\tfwidth'])
        mid = texout.Axis(options=common + [
            'ymin=0', 'xticklabels={}', 'ylabel={$\\mathcal{P}$ [\\%]}',
            'height=0.32\\tfwidth'])
        low = texout.Axis(options=common + [
            f'ymin=-0.6, ymax={len(series) - 0.4}', 'y dir=reverse',
            f'ytick={{{",".join(str(i) for i in range(len(series)))}}}',
            'yticklabels={' + ','.join(
                r'\texttt{' + texout.escape(short(k)) + '}'
                for k, _ in series) + '}',
            'yticklabel style={font=\\scriptsize, color=tfink}',
            'ymajorgrids=false', 'xlabel={place, most efficient first}',
            f'height={max(0.24, 0.1 + 0.05 * len(series)):.2f}\\tfwidth'])
        rows = []
        for y, (k, values) in enumerate(series):
            top.plots.append(texout.Plot(
                styles(k), [(i + 1, 100 * v)
                            for i, (_, v) in enumerate(values)],
                texout.escape(short(k))))
            cumulative = []
            for i in range(len(values)):
                vs = [v for _, v in values[:i + 1]]
                pp = (len(vs) / sum(1.0 / v for v in vs)
                      if all(v > 0 for v in vs) else 0.0)
                cumulative.append((i + 1, 100 * pp))
                rows.append({'kernel': k, 'order': o, 'place': i + 1,
                             'arch': values[i][0],
                             'efficiency': values[i][1], 'pp_first': pp})
            mid.plots.append(texout.Plot(styles(k) + ', forget plot',
                                         cumulative))
            for i, (a, _) in enumerate(values):
                # Named, not coloured: the colours are the kernels'.
                low.raw.append(
                    f'\\node[tf label, font=\\tiny\\ttfamily] at (axis cs:'
                    f'{i + 1},{y}) {{{texout.escape(a)}}};\n')
        tex = texout.picture(
            [top, mid, low], 'group size=1 by 3, vertical sep=0.3cm, '
                             'x descriptions at=edge bottom',
            comment=(f'generated by tools/bench/publish.py: {eq}, '
                     f'{_dt_name(dt)}, order {o}; the cascade of Sewall et '
                     f'al. (P3HPC 2020) over {", ".join(machines)}\n'
                     f'{_flops_note()}'))
        caption = (f'Performance portability of the {eq} kernels at order '
                   f'{o}, {_dt_name(dt)}, as a cascade (Sewall et al.): each '
                   'kernel\'s efficiency on every machine, most efficient '
                   'first; $\\mathcal{P}$ over the first so many of them; '
                   'and which machine stands at which place. Zero where a '
                   f'machine produced no number. {_flops_are()}')
        out.append(Output(f'cascade-{eq}-{dt.lower()}', 'figure', tex, rows,
                          caption))
    return out


# ----------------------------------------------------------------------
# What explains the curves
# ----------------------------------------------------------------------

def _grid(n: int) -> Tuple[int, int]:
    cols = min(3, n)
    return cols, -(-n // cols)


def figure_lanes(fieldset: Field, kernels: Sequence[str], styles: Styles
                 ) -> List[Output]:
    """How much of the arithmetic a kernel issues the operation needs --
    solid -- and how much a dense implementation would do -- dashed, over
    the same -- against the order, one panel per machine.  Issued is never
    below needed; it is below dense where the kernel skips more structural
    zeros than the lanes of a destination narrower than the warp or the
    wave leave idle, and above where it is the other way round."""
    ref = fieldset.reference
    full = [p for p in _full(fieldset, kernels)
            if p.config == ref and p.capture.simulations == 1
            and p.row.get('executed_flops')]
    out = []
    for eq, dt in sorted({(p.capture.equation, p.datatype) for p in full}):
        mine = [p for p in full if (p.capture.equation, p.datatype)
                == (eq, dt)]
        archs = sorted({p.arch for p in mine})
        orders = sorted({p.order for p in mine})
        if len(orders) < 2:
            continue
        cols, nrows = _grid(len(archs))
        axes, rows = [], []
        # One scale for every panel: their tick labels are drawn once.
        top = 1.08 * 100 * max(
            [1.0] + [max(p.row.get('nonzero_flops') or 0,
                         p.row.get('flops') or 0) / p.row['executed_flops']
                     for p in mine])
        for i, arch in enumerate(archs):
            here = [p for p in mine if p.arch == arch]
            lanes = sorted({p.row.get('lanes') for p in here
                            if p.row.get('lanes')})
            ax = texout.Axis(options=[
                f'xtick={{{",".join(map(str, orders))}}}',
                f'xmin={orders[0] - 0.4}, xmax={orders[-1] + 0.4}',
                f'ymin=0, ymax={top:.4g}', 'xlabel={order $O$}',
                'title style={font=\\scriptsize, align=center}',
                f'title={{\\texttt{{{texout.escape(arch)}}}\\\\'
                f'{"/".join(map(str, lanes))} lanes}}']
                + (['ylabel={\\% of issued flops}'] if i % cols == 0
                   else ['yticklabels={}']))
            if i == 0:
                ax.options.append('legend to name=tflanes')
            for k in kernels:
                pts = sorted((p.order,
                              (p.row.get('nonzero_flops') or 0)
                              / p.row['executed_flops'],
                              (p.row.get('flops') or 0)
                              / p.row['executed_flops'])
                             for p in here if p.workload == k)
                if not pts:
                    continue
                ax.plots.append(texout.Plot(
                    styles(k), [(o, 100 * n) for o, n, _ in pts],
                    texout.escape(short(k)) if i == 0 else None))
                ax.plots.append(texout.Plot(
                    styles(k) + ', densely dashed, mark=none, forget plot',
                    [(o, 100 * d) for o, _, d in pts]))
                rows += [{'arch': arch, 'kernel': k, 'order': o,
                          'needed_of_issued': n, 'dense_of_issued': d}
                         for o, n, d in pts]
            axes.append(ax)
        tex = texout.picture(
            axes, f'group size={cols} by {nrows}, horizontal sep=0.25cm, '
                  'vertical sep=1.1cm',
            comment=(f'generated by tools/bench/publish.py: {eq}, '
                     f'{_dt_name(dt)}; flops per element: needed and dense '
                     'over issued, at the largest launch'),
            options=[f'width={0.98 / cols + 0.06:.2f}\\tfwidth',
                     f'height={0.52 if cols > 1 else 0.62}\\tfwidth',
                     'legend columns=3'])
        tex = tex.replace('\\end{tikzpicture}\n',
                          '\\node[anchor=south] at (current bounding box.'
                          'north) {\\pgfplotslegendfromname{tflanes}};\n'
                          '\\end{tikzpicture}\n')
        caption = (f'How much of what the {eq} kernels issue is needed, '
                   f'{_dt_name(dt)}, per order and machine: solid, the '
                   'flops the operation needs over those the kernel issues; '
                   'dashed, those a dense implementation performs over the '
                   'same -- above 100\\,\\% where the kernel skips more '
                   'structural zeros than its idle lanes cost, below where a '
                   'destination narrower than the warp or the wave leaves '
                   'lanes idle.')
        out.append(Output(f'lanes-{eq}-{dt.lower()}', 'figure', tex, rows,
                          caption))
    return out


def _spill_bytes(row: Dict) -> Optional[int]:
    """What a build spilled, read as `toolchain.resources` reads it: on
    NVIDIA the spill traffic, stores and loads together -- the stack frame
    holds more than spills -- on AMD the scratch, on Intel the spill size
    IGC states."""
    static = row.get('static')
    if not static:
        return None
    backend = row.get('backend')
    if backend == 'cuda':
        return ((static.get('spillstores') or 0)
                + (static.get('spillloads') or 0))
    if backend == 'hip':
        return static.get('scratchsize') or 0
    return static.get('vgprsspill') or 0


def figure_resources(fieldset: Field, kernels: Sequence[str],
                     styles: Styles) -> List[Output]:
    """What a kernel takes of the machine, against the order: registers a
    lane, what spilled, and the code against the instruction cache it has to
    stay resident in -- which is what the larger orders run out of first."""
    ref = fieldset.reference
    full = [p for p in _full(fieldset, kernels)
            if p.config == ref and p.capture.simulations == 1]
    out = []
    for arch, dt, eq in sorted({(p.arch, p.datatype, p.capture.equation)
                                for p in full}):
        mine = [p for p in full if (p.arch, p.datatype, p.capture.equation)
                == (arch, dt, eq)]
        orders = sorted({p.order for p in mine})
        if len(orders) < 2:
            continue
        registers = [p for p in mine if (p.row.get('static') or {}).get(
            'vgprs')]
        spills = [p for p in mine if _spill_bytes(p.row)]
        code = [p for p in mine if p.row.get('code_bytes')]
        if not registers and not code:
            continue
        common = [f'xtick={{{",".join(map(str, orders))}}}',
                  f'xmin={orders[0] - 0.4}, xmax={orders[-1] + 0.4}',
                  'ymin=0', 'height=0.4\\tfwidth']
        panels = []
        rows = []

        def panel(points, value, label, legend):
            ax = texout.Axis(options=common + [f'ylabel={{{label}}}'])
            if legend:
                ax.options.append('tf legend above')
            for k in kernels:
                pts = sorted((p.order, value(p)) for p in points
                             if p.workload == k)
                if pts:
                    ax.plots.append(texout.Plot(
                        styles(k), pts,
                        texout.escape(short(k)) if legend else None))
            return ax

        if registers:
            panels.append(panel(
                registers, lambda p: (p.row['static'].get('vgprs', 0)
                                      + p.row['static'].get('agprs', 0)),
                'registers', True))
        if spills:
            panels.append(panel(spills, lambda p: _spill_bytes(p.row),
                                'spills [B]', not panels))
        kib = None
        if code:
            ax = panel(code, lambda p: p.row['code_bytes'] / 1024,
                       'code [KiB]', not panels)
            cache = {p.row.get('icache_bytes') for p in code} - {None}
            if len(cache) == 1:
                # The rule is named in the caption: the code grows into it
                # with the order, so a label at either end meets a curve.
                kib = cache.pop() / 1024
                ax.raw.append(texout.hrule(kib))
                highest = max([kib] + [p.row['code_bytes'] / 1024
                                       for p in code])
                ax.options.append(f'ymax={1.1 * highest:.4g}')
            panels.append(ax)
        for p in mine:
            static = p.row.get('static') or {}
            rows.append({'kernel': p.workload, 'order': p.order,
                         'registers': (static.get('vgprs', 0)
                                       + static.get('agprs', 0)) or None,
                         'spilled_bytes': _spill_bytes(p.row),
                         'code_bytes': p.row.get('code_bytes'),
                         'icache_bytes': p.row.get('icache_bytes')})
        tex = texout.picture(
            panels, f'group size=1 by {len(panels)}, vertical sep=0.25cm, '
                    'x descriptions at=edge bottom',
            comment=(f'generated by tools/bench/publish.py: {arch}, '
                     f'{_dt_name(dt)}, {eq}; registers and spills as the '
                     'compiler reported them, code as the emitter laid it '
                     'down (analysis.icache)'))
        panels[-1].options.append('xlabel={order $O$}')
        caption = (f'What the {eq} kernels take of '
                   f'{texout.escape(fieldset.device_of(arch))}, '
                   f'{_dt_name(dt)}, per order: registers a lane and what '
                   'spilled, as the compiler reports them, and the code '
                   'against the instruction cache TensorForge assumes for '
                   'the target'
                   + (f', {kib:g}\\,KiB, the rule' if kib else '')
                   + ', which a kernel\'s batch loop has to stay resident '
                   'in.')
        out.append(Output(f'resources-{arch}-{eq}-{dt.lower()}', 'figure',
                          tex, rows, caption))
    return out


def figure_precision(fieldset: Field, kernels: Sequence[str]
                     ) -> List[Output]:
    """Double against single, per kernel and order, on every machine: the
    ratio of the times per element at a full launch, beside the two
    ratios it would be if the kernel were bound by one thing -- 2, the
    bytes, and the machine's own ratio of FMA rates, in its legend."""
    ref = fieldset.reference
    full = [p for p in _full(fieldset, kernels)
            if p.config == ref and p.capture.simulations == 1]
    times: Dict[Tuple, Dict[str, float]] = defaultdict(dict)
    for p in full:
        per = _per_element(p.row, p.batch, 'event')
        if per:
            times[(p.capture.equation, p.workload, p.order, p.arch)][
                p.datatype] = per
    out = []
    for eq in sorted({k[0] for k in times}):
        archs = sorted({a for (e, _, _, a), t in times.items()
                        if e == eq and len(t) == 2})
        groups = sorted({(k, o) for (e, k, o, a), t in times.items()
                         if e == eq and len(t) == 2},
                        key=lambda t: (kernels.index(t[0]), t[1]))
        if not archs or not groups:
            continue
        rows, values, plots = [], [], []
        for m, arch in enumerate(archs):
            pts = []
            for y, (k, o) in enumerate(groups):
                t = times.get((eq, k, o, arch), {})
                r = t['F64'] / t['F32'] if len(t) == 2 else None
                pts.append((r, y))
                if r:
                    values.append(r)
                    rows.append({'kernel': k, 'order': o, 'arch': arch,
                                 'f64_over_f32': r})
            single, double = (fieldset.ceiling(arch, 'F32'),
                              fieldset.ceiling(arch, 'F64'))
            fma = (single.flops_per_s / double.flops_per_s
                   if single and double and double.flops_per_s else None)
            label = (r'\texttt{' + texout.escape(arch) + '}'
                     + (f' (FMA {texout.num(fma, 2)}:1)' if fma else ''))
            plots.append(texout.Plot(f'tf s{m + 1}, only marks', pts, label))
        ax = _dot_axis(groups, values, 'FP64 time over FP32 time', rule=2.0)
        ax.raw.append('\\node[tf label, anchor=south west, rotate=90] at '
                      '({axis cs:2,0}|-{rel axis cs:0,0}) {bytes};\n')
        ax.plots = plots
        tex = texout.picture([ax], comment=(
            f'generated by tools/bench/publish.py: {eq}; double over single '
            'per element, device clock, largest launch; the legend gives '
            'each machine\'s measured FMA rates, single over double'))
        caption = (f'Double against single precision for the {eq} kernels: '
                   'the time per element at a full launch in FP64 over that '
                   'in FP32, per kernel, order and machine. A kernel bound '
                   'by memory sits near 2, the ratio of the bytes; one bound '
                   'by arithmetic near its machine\'s ratio of FMA rates, '
                   'given in the legend.')
        out.append(Output(f'precision-{eq}', 'figure', tex, rows, caption))
    return out


def figure_simulations(fieldset: Field, kernels: Sequence[str], order: int,
                       styles: Styles) -> List[Output]:
    """Fused simulations: the speed-up per element and simulation over one
    simulation, against how many are fused, per kernel at one order -- what
    an operator read once for several simulations is worth."""
    ref = fieldset.reference
    full = [p for p in _full(fieldset, kernels) if p.config == ref]
    out = []
    for arch, dt, eq in sorted({(p.arch, p.datatype, p.capture.equation)
                                for p in full}):
        mine = [p for p in full if (p.arch, p.datatype, p.capture.equation)
                == (arch, dt, eq)]
        counts: Dict[int, set] = defaultdict(set)
        for p in mine:
            counts[p.order].add(p.capture.simulations)
        usable = [o for o, sims in counts.items() if len(sims) > 1
                  and 1 in sims]
        if not usable:
            continue
        o = min(usable, key=lambda x: (abs(x - order), x))
        sims = sorted(counts[o])
        ax = texout.Axis(options=[
            'tf', 'tf legend above', 'xmode=log', 'log basis x=2',
            f'xtick={{{",".join(map(str, sims))}}}',
            f'xticklabels={{{",".join(map(str, sims))}}}',
            f'xmin={sims[0] / 1.3:g}, xmax={sims[-1] * 1.3:g}', 'ymin=0',
            'xlabel={fused simulations}',
            'ylabel={speed-up per simulation}'])
        ax.raw.append(texout.hrule(1.0))
        rows = []
        for k in kernels:
            per = {}
            for p in mine:
                if p.workload == k and p.order == o:
                    t = _per_element(p.row, p.batch, 'event')
                    if t:
                        per[p.capture.simulations] = t / \
                            p.capture.simulations
            if 1 not in per or len(per) < 2:
                continue
            pts = [(s, per[1] / per[s]) for s in sorted(per)]
            ax.plots.append(texout.Plot(styles(k), pts,
                                        texout.escape(short(k))))
            rows += [{'kernel': k, 'order': o, 'simulations': s,
                      'speedup': v} for s, v in pts]
        if not rows:
            continue
        tex = texout.picture([ax], comment=(
            f'generated by tools/bench/publish.py: {arch}, {_dt_name(dt)}, '
            f'{eq}, order {o}; device clock, largest launch'))
        caption = (f'Fused simulations of the {eq} kernels at order {o} on '
                   f'{texout.escape(fieldset.device_of(arch))}, '
                   f'{_dt_name(dt)}: the time per element and simulation '
                   'with one simulation over that with several fused into '
                   'one launch, at a full launch.')
        out.append(Output(f'simulations-{arch}-{eq}-{dt.lower()}', 'figure',
                          tex, rows, caption))
    return out


#: A profiled kernel: the architecture it ran on -- None where the profile
#: does not say -- and its symbol.  The symbol alone is not enough: it is a
#: digest of the kernel's source, which two architectures of one backend can
#: share.
ProfileKey = Tuple[Optional[str], str]


def _profiles(paths: Iterable[Path]) -> Dict[ProfileKey, Tuple[float, int]]:
    """`profile.py`'s measurements by the kernel they are of: the DRAM
    bytes the longest-running kernel of a workload moved, and the batch it
    moved them at."""
    out: Dict[ProfileKey, Tuple[float, int]] = {}
    for target in paths:
        target = Path(target)
        found = ([target] if target.is_file() and target.name ==
                 'profile.json' else sorted(target.rglob('profile.json'))
                 if target.is_dir() else [])
        for path in found:
            try:
                blob = json.loads(path.read_text())
            except (OSError, ValueError):
                continue
            batch = blob.get('batch')
            arch = ((blob.get('device') or {}).get('spec') or {}).get('arch')
            for collection in blob.get('collections') or []:
                rows = [r for r in collection.get('rows') or [] if r]
                symbol = collection.get('symbol')
                if not rows or not symbol or not batch:
                    continue
                hottest = max(rows, key=lambda r: r.get('duration_ns') or 0)
                read = hottest.get('dram_read_bytes')
                write = hottest.get('dram_write_bytes')
                if read is None and write is None:
                    continue
                out[(arch, symbol)] = ((read or 0.0) + (write or 0.0),
                                       int(batch))
    return out


def _bytes_at(rows: Sequence[Dict], batch: int) -> Optional[float]:
    """The compulsory bytes of a launch of `batch`, from the rows of other
    batches: they are linear in it -- per element, plus what a launch reads
    once."""
    pts = sorted({(r['batch'], r['bytes']) for r in rows
                  if r.get('batch') and r.get('bytes')})
    for b, v in pts:
        if b == batch:
            return float(v)
    if len(pts) < 2:
        return None
    (b0, v0), (b1, v1) = pts[0], pts[-1]
    per = (v1 - v0) / (b1 - b0)
    return v0 + per * (batch - b0)


def figure_traffic(fieldset: Field,
                   profiles: Dict[ProfileKey, Tuple[float, int]],
                   kernels: Sequence[str]) -> List[Output]:
    """What a kernel moved through DRAM, as the profiler counted it, over
    what it had to move: near one, the traffic is compulsory and the code is
    not the lever; far above, the same bytes were fetched again."""
    if not profiles:
        return []
    ref = fieldset.reference
    by: Dict[Tuple, List[Point]] = defaultdict(list)
    for p in fieldset.points:
        if (p.workload in kernels and p.capture is not None
                and p.config == ref and p.capture.simulations == 1):
            by[(p.arch, p.datatype, p.capture, p.workload)].append(p)
    out = []
    for arch, dt, eq in sorted({(a, d, c.equation) for (a, d, c, _) in by}):
        entries = []
        for (a, d, capture, k), points in by.items():
            if (a, d, capture.equation) != (arch, dt, eq):
                continue
            symbol = points[0].row.get('symbol')
            # A profile names the architecture, not the machine.
            found = (profiles.get((fieldset.architecture(arch), symbol))
                     or profiles.get((None, symbol)))
            if found is None:
                continue
            measured, batch = found
            needed = _bytes_at([p.row for p in points], batch)
            if needed:
                entries.append({'kernel': k, 'order': capture.order,
                                'batch': batch, 'measured_bytes': measured,
                                'compulsory_bytes': needed,
                                'ratio': measured / needed})
        if not entries:
            continue
        entries.sort(key=lambda e: (kernels.index(e['kernel']), e['order']))
        groups = [(e['kernel'], e['order']) for e in entries]
        ax = _dot_axis(groups, [e['ratio'] for e in entries],
                       'DRAM bytes measured over compulsory')
        ax.options.remove('tf legend above')
        ax.plots = [texout.Plot('tf s1, only marks',
                                [(e['ratio'], y)
                                 for y, e in enumerate(entries)])]
        tex = texout.picture([ax], comment=(
            f'generated by tools/bench/publish.py: {arch}, {_dt_name(dt)}, '
            f'{eq}; profile.py\'s DRAM read and write bytes over the '
            'compulsory bytes at the profiled batch'))
        caption = (f'Traffic of the {eq} kernels on '
                   f'{texout.escape(fieldset.device_of(arch))}, '
                   f'{_dt_name(dt)}: the DRAM bytes the profiler counted '
                   'over those the operation has to move, per kernel and '
                   'order. Near one the traffic is compulsory; below, a '
                   'cache held what was counted as compulsory.')
        out.append(Output(f'traffic-{arch}-{eq}-{dt.lower()}', 'figure', tex,
                          entries, caption))
    return out


def _manifests(paths: Iterable[Path]) -> List[Dict]:
    """What every run built, from the manifests beside its results: one
    entry per workload and unit, with the suite it belonged to."""
    out: List[Dict] = []
    seen = set()
    for target in paths:
        target = Path(target)
        found = ([target] if target.is_file()
                 else sorted(target.rglob('bench.json')))
        for path in found:
            if path.resolve() in seen:
                continue
            seen.add(path.resolve())
            try:
                blob = json.loads(path.read_text())
            except (OSError, ValueError):
                continue
            if not isinstance(blob, dict):
                continue
            suite = (blob.get('run') or {}).get('suite')
            for entry in blob.get('manifest') or []:
                out.append(dict(entry, suite=suite))
    return out


def table_buildcost(manifests: Sequence[Dict], fieldset: Optional[Field]
                    ) -> List[Output]:
    """What making the kernels cost on each machine, per export: how many
    there were, the seconds generating and compiling them, the objects, and
    the largest kernel's code -- against the instruction cache, named in
    the heading.

    Over the kernels a production build makes: a piece of the split suite
    is none of them, and a kernel two runs of a campaign built -- the
    timing run's few and the build of the whole configuration
    (`evaluate.py`) -- is one kernel, counted once, with the compile time
    where one of them has it."""
    kernels: Dict[Tuple, Dict[str, Dict]] = defaultdict(dict)
    for entry in manifests:
        workload = entry.get('workload') or ''
        if (entry.get('error') or entry.get('generate_s') is None
                or Piece.of(workload) is not None):
            continue
        capture = Capture.from_tag(entry.get('capture') or '')
        if capture is None:
            continue
        arch = entry.get('arch') or '?'
        if fieldset is not None:
            arch = fieldset.machine_name(arch, entry.get('backend') or '?')
        seen = kernels[(arch, entry.get('config') or '',
                        entry.get('datatype') or '', capture.equation,
                        capture.simulations, capture.order)]
        if workload not in seen or (seen[workload].get('compile_s') is None
                                    and entry.get('compile_s') is not None):
            seen[workload] = entry
    groups = {key: list(seen.values()) for key, seen in kernels.items()}
    out = []
    for arch in sorted({key[0] for key in groups}):
        mine = {key: entries for key, entries in groups.items()
                if key[0] == arch}
        caches = {e.get('icache_bytes') for entries in mine.values()
                  for e in entries} - {None}
        device = fieldset.device_of(arch) if fieldset else arch
        configs = sorted({key[1] for key in mine})
        body: List = []
        rows_csv: List[Dict] = []
        last = None
        for key in sorted(mine, key=lambda k: (k[1], k[2] != 'F64', k[3],
                                                k[4], k[5])):
            _, config, dt, eq, sims, order = key
            entries = mine[key]
            if (config, dt) != last:
                if body:
                    body.append(None)
                body.append([r'\multicolumn{7}{@{}l}{\textit{'
                             + _dt_name(dt) + '}'
                             + (r', \texttt{' + texout.escape(config) + '}'
                                if len(configs) > 1 else '') + '}'])
                last = (config, dt)
            generate = sum(e['generate_s'] for e in entries)
            compiled = [e['compile_s'] for e in entries
                        if e.get('compile_s') is not None]
            objects = [e['object_bytes'] for e in entries
                       if e.get('object_bytes')]
            code = [e['code_bytes'] for e in entries if e.get('code_bytes')]
            body.append([
                texout.escape(eq) + (f', {sims} sim.' if sims > 1 else ''),
                str(order), str(len(entries)), texout.num(generate),
                texout.num(sum(compiled)) if compiled else '--',
                texout.num(sum(objects) / 2 ** 20) if objects else '--',
                texout.num(max(code) / 1024) if code else '--'])
            rows_csv.append({'arch': arch, 'config': config, 'datatype': dt,
                             'equation': eq, 'simulations': sims,
                             'order': order, 'kernels': len(entries),
                             'suite': ' '.join(sorted({str(e.get('suite'))
                                                       for e in entries})),
                             'generate_s': generate,
                             'compile_s': sum(compiled) if compiled else None,
                             'object_bytes': (sum(objects) if objects
                                              else None),
                             'largest_code_bytes': max(code) if code
                             else None})
        cache = (f'{caches.pop() / 1024:g}\\,KiB'
                 if len(caches) == 1 else None)
        tex = texout.table(
            'l r r r r r r',
            [['Export', '$O$', 'Kernels', 'Generate [s]', 'Compile [s]',
              'Objects [MiB]', 'Largest code [KiB]']],
            body,
            comment=('generated by tools/bench/publish.py from the run '
                     'manifests: seconds summed over the kernels, which '
                     'compile in parallel, so not a build\'s wall time;\n'
                     'code as the emitter laid it down (analysis.icache)'
                     + (f'; instruction cache {cache}' if cache else '')))
        caption = (f'What making the kernels cost on '
                   f'{texout.escape(device)}, per export: how many there '
                   'were, the seconds TensorForge spent generating them and '
                   'the compiler compiling them, summed over the kernels, '
                   'the size of the objects, and the largest kernel\'s code'
                   + (f' against an instruction cache of {cache}'
                      if cache else '') + '.')
        out.append(Output(f'buildcost-{arch}', 'table', tex, rows_csv,
                          caption))
    return out


# ----------------------------------------------------------------------
# What the runs did
# ----------------------------------------------------------------------

@dataclass
class LtsSeries:
    """A clustering to read the launch costs against: the configuration
    its kernels are at, and the runs it came from, where it did."""
    clustering: Clustering
    capture: Capture
    runs: List[Run] = field(default_factory=list)


def lts_series(clusterings: Sequence[Tuple[Clustering, Capture]],
               runs: Sequence[Run]) -> List[LtsSeries]:
    """The clusterings given, and those of the runs with more than one
    cluster -- one series per clustering and configuration, with every run
    that had it."""
    out: List[LtsSeries] = [LtsSeries(c, cap) for c, cap in clusterings]
    by: Dict[Tuple, LtsSeries] = {}
    for run in runs:
        if not run.lts or run.clustering is None or run.capture is None:
            continue
        key = (tuple(sorted(run.clustering.layers,
                            key=lambda l: (l.rank, l.cluster, l.copy))),
               run.capture.tag)
        if key not in by:
            by[key] = LtsSeries(run.clustering, run.capture)
            out.append(by[key])
        by[key].runs.append(run)
    return out


def figure_lts(series: Sequence[LtsSeries], fits: Sequence[Dict],
               runs: Sequence[Run], fieldset: Optional[Field]
               ) -> Tuple[List[Output], List[str]]:
    """Local time stepping against what a launch costs.  Above, where the
    cell updates of a clustering fall by the cells a launch carries, with
    the batch at which a time step reaches half its rate on each machine --
    from the kernel benchmark (solid) and from SeisSol's own fit of a run
    (dashed).  Below, the speed-up over global time stepping: the ideal of
    the clustering, what the measured launch costs leave of it on each
    machine, and what a pair of runs measured."""
    out, skipped = [], []
    measured = measured_speedups(runs)
    for item in series:
        clustering, capture = item.clustering, item.capture
        dt = capture.datatype
        tag = f'{clustering.name}-{capture.tag}'
        fit = {}
        for reading, name in zip(('together', 'serial'), STEP_ROWS):
            for r in fits:
                if (r['kernel'] == name and r['equation'] == capture.equation
                        and r['order'] == capture.order
                        and r['datatype'] == dt
                        and r.get('t0_us') is not None
                        and r.get('ns_per_element')):
                    fit.setdefault(r['arch'], {})[reading] = r
        archs = sorted(fit)
        if not archs:
            skipped.append(f'lts-{tag}: the clustering alone, no speed-ups -- '
                           f'no time step was swept at {capture.tag}, and '
                           'the launch costs come from the ramp-up '
                           '(--batches)')
        updates = clustering.updates()
        total = sum(u for _, u in updates) or 1.0
        shares: Dict[int, float] = defaultdict(float)
        for layer, u in updates:
            shares[int(math.floor(math.log2(layer.cells)))] += u / total
        ks = list(range(min(shares), max(shares) + 1))
        hist = ([(2.0 ** k, 100 * shares.get(k, 0.0)) for k in ks]
                + [(2.0 ** (ks[-1] + 1), 100 * shares.get(ks[-1], 0.0))])
        top = texout.Axis(options=[
            'xmode=log', 'log basis x=2', 'ymin=0',
            f'xmin={2.0 ** ks[0]:g}, xmax={2.0 ** (ks[-1] + 1):g}',
            'xlabel={cells a launch carries}',
            'ylabel={\\% of cell updates}', 'height=0.42\\tfwidth',
            'xmajorgrids=false',
            'legend style={at={(0.5,1.03)}, anchor=south, legend columns=2, '
            '/tikz/every even column/.append style={column sep=0.8em}}'])
        top.plots.append(texout.Plot(
            'ybar interval, fill=tfgrid, draw=tfaxis, mark=none, forget plot',
            hist))
        # A run names the architecture; every machine of it gets the run.
        under = {a: fieldset.architecture(a) if fieldset is not None else a
                 for a in archs}
        rows: List[Dict] = []
        dashed = False
        for m, arch in enumerate(archs):
            half = fit[arch].get('together', {}).get('half_fit')
            app = [r.half() for r in item.runs if r.arch == under[arch]]
            app = [h for h in app if h]
            colour = f'tfc{m % len(texout.CATEGORICAL) + 1}'
            for value, dash, source in ((half, '', 'kernels'),
                                        (app[0] if app else None,
                                         ', densely dashed', 'seissol')):
                if not value:
                    continue
                below = sum(u for layer, u in updates
                            if layer.cells < value) / total
                top.raw.append(
                    f'\\draw[{colour}, line width=0.9pt{dash}] '
                    f'({{axis cs:{value:.6g},0}}|-{{rel axis cs:0,0}}) -- '
                    f'({{axis cs:{value:.6g},0}}|-{{rel axis cs:0,1}});\n')
                if dash:
                    dashed = True
                else:
                    # A key rather than labels at the lines: two machines'
                    # half-rate batches are often a few percent apart.
                    top.raw.append(
                        f'\\addlegendimage{{{colour}, line width=0.9pt}}\n'
                        f'\\addlegendentry{{\\texttt{{{texout.escape(arch)}}}'
                        f': $n_{{1/2}}={texout.num(value)}$, '
                        f'{100 * below:.0f}\\,\\% below}}\n')
                rows.append({'series': clustering.name, 'arch': arch,
                             'source': source, 'half': value,
                             'updates_below': below})
        if dashed:
            top.raw.append('\\addlegendimage{tfsecondary, densely dashed, '
                           'line width=0.9pt}\n\\addlegendentry{SeisSol\'s '
                           'own fit}\n')
        ideal = clustering.ideal_speedup()
        axes = [top]
        if archs:
            bars = texout.Axis(options=[
                'ybar=0pt', 'bar width=0.25', 'ymin=0',
                f'xtick={{{",".join(str(i) for i in range(len(archs)))}}}',
                'xticklabels={' + ','.join(
                    r'\texttt{' + texout.escape(a) + '}' for a in archs)
                + '}',
                f'xmin=-0.6, xmax={len(archs) - 0.4}', 'xmajorgrids=false',
                'ylabel={speed-up over GTS}', 'height=0.48\\tfwidth',
                'tf legend above', 'area legend'])
            predicted = {}
            for reading, colour, label in (
                    ('together', 'tfo6', 'launch costs, overlapped'),
                    ('serial', 'tfo2', 'one after another')):
                pts = []
                for x, arch in enumerate(archs):
                    r = fit[arch].get(reading)
                    v = (clustering.predicted_speedup(
                        r['t0_us'] * 1e3, r['ns_per_element'])
                        if r else None)
                    predicted[(arch, reading)] = v
                    pts.append((x, v))
                bars.plots.append(texout.Plot(
                    f'fill={colour}, draw=none, mark=none', pts, label))
            seen = {}
            for x, arch in enumerate(archs):
                for run in item.runs:
                    key = (run.arch, run.configuration, run.ranks,
                           round(run.cells))
                    if run.arch == under[arch] and key in measured:
                        seen[x] = measured[key]
                        break
            if seen:
                bars.plots.append(texout.Plot(
                    'fill=tfink, draw=none, mark=none',
                    [(x, seen.get(x)) for x in range(len(archs))],
                    'measured'))
            # Room above the tallest bar and the ideal for the rule's label.
            shown = [v for v in list(predicted.values()) + list(seen.values())
                     if v]
            bars.options.append(f'ymax={1.15 * max([ideal] + shown):.4g}')
            bars.raw.append(texout.hrule(
                ideal, f'ideal {texout.num(ideal, 2)}$\\times$'))
            for arch in archs:
                rows.append({'series': clustering.name, 'arch': arch,
                             'source': 'speedup', 'ideal': ideal,
                             'predicted_together': predicted.get(
                                 (arch, 'together')),
                             'predicted_serial': predicted.get(
                                 (arch, 'serial')),
                             'measured': seen.get(archs.index(arch))})
            axes.append(bars)
        label = (texout.escape(clustering.name)
                 + (' (synthetic, not from a mesh)'
                    if clustering.synthetic else ''))
        tex = texout.picture(
            axes, f'group size=1 by {len(axes)}, vertical sep=1.3cm',
            comment=(f'generated by tools/bench/publish.py: clustering '
                     f'{clustering.name}'
                     + (' -- SYNTHETIC' if clustering.synthetic else '')
                     + f', {len(clustering.ranks)} rank(s), kernels at '
                     f'{capture.tag}; the time step as in STEP; LTS adds the '
                     'Taylor expansion at cluster borders, which is not '
                     'counted'))
        caption = (f'Local time stepping on clustering {label}, '
                   f'{len(clustering.ranks)} rank(s), {capture.equation} at '
                   f'order {capture.order}, {_dt_name(dt)}. Above, the cell '
                   'updates by the cells a launch carries, every layer of '
                   'every cluster a launch of its own, and the batch at '
                   'which a time step reaches half its rate on each machine '
                   '-- from the kernel benchmark, with the share of the '
                   'updates below it, and dashed from SeisSol\'s own fit of '
                   'a run. Below, the speed-up over global time stepping: '
                   'the ideal of the clustering, what the measured launch '
                   'costs leave of it with the relations of a face '
                   'overlapped and one after another, and, where both were '
                   'run, the measured one.')
        out.append(Output(f'lts-{tag}', 'figure', tex, rows, caption))
    return out, skipped


def _run_label(run: Run) -> str:
    capture = run.capture
    return (r'\texttt{' + texout.escape(run.arch) + '}'
            + (f' $O{capture.order}$ {_dt_name(capture.datatype)}'
               if capture else ''))


def figure_throughput(runs: Sequence[Run]) -> List[Output]:
    """What SeisSol got through on each machine: cell updates per second
    and rank, per order and precision, a bar per machine -- the number the
    kernels are for."""
    out = []
    groups: Dict[Tuple[str, bool], List[Run]] = defaultdict(list)
    for run in runs:
        if run.capture and run.updates_per_second_per_rank:
            groups[(run.capture.equation, run.lts)].append(run)
    for (eq, lts), mine in sorted(groups.items()):
        archs = sorted({r.arch for r in mine})
        datatypes = sorted({r.capture.datatype for r in mine},
                           reverse=True)
        orders = sorted({r.capture.order for r in mine})
        # At least three orders wide, so that a single order's bars keep
        # the width they have among several.
        middle = (orders[0] + orders[-1]) / 2
        half = max(orders[-1] - orders[0] + 1.2, 3.0) / 2
        axes, rows = [], []
        for i, dt in enumerate(datatypes):
            ax = texout.Axis(options=[
                'ybar=0pt', f'bar width={min(0.8 / len(archs), 0.25):.3f}',
                'ymin=0', f'xtick={{{",".join(map(str, orders))}}}',
                f'xmin={middle - half:g}, xmax={middle + half:g}',
                'xmajorgrids=false', f'xlabel={{order $O$, {_dt_name(dt)}}}',
                'area legend']
                + (['ylabel={M cell updates/s per rank}', 'tf legend above']
                   if i == 0 else []))
            for m, arch in enumerate(archs):
                pts = []
                for o in orders:
                    values = sorted(
                        r.updates_per_second_per_rank for r in mine
                        if (r.arch, r.capture.datatype, r.capture.order)
                        == (arch, dt, o))
                    v = values[len(values) // 2] if values else None
                    pts.append((o, v / 1e6 if v else None))
                    if v:
                        rows.append({'arch': arch, 'datatype': dt,
                                     'order': o, 'runs': len(values),
                                     'updates_per_s_per_rank': v})
                ax.plots.append(texout.Plot(
                    f'fill=tfc{m % len(texout.CATEGORICAL) + 1}, draw=none, '
                    'mark=none', pts,
                    (r'\texttt{' + texout.escape(arch) + '}')
                    if i == 0 else None))
            axes.append(ax)
        if len(axes) == 1:
            axes[0].options.insert(0, 'tf')
            tex = texout.picture(axes, comment='generated by '
                                 'tools/bench/publish.py from SeisSol\'s '
                                 'run summaries')
        else:
            tex = texout.picture(
                axes, f'group size={len(axes)} by 1, horizontal sep=1.2cm',
                comment='generated by tools/bench/publish.py from SeisSol\'s '
                        'run summaries',
                options=['width=0.56\\tfwidth', 'height=0.62\\tfwidth'])
        mode = 'local' if lts else 'global'
        caption = (f'What SeisSol got through, {eq} with {mode} time '
                   'stepping: cell updates per second and rank over the '
                   'whole simulation, per order, precision and machine -- '
                   'the median where several runs were given -- as the '
                   'run summaries state them.')
        out.append(Output(f'throughput-{eq}-{mode}', 'figure', tex, rows,
                          caption))
    return out


#: The parts of a simulation's time a run summary states, in the order a
#: bar stacks them, with their colours.
_SPLIT = (('computeLocalIntegration', 'local', 'tfc1'),
          ('computeNeighboringIntegration', 'neighbour', 'tfc2'),
          ('computeDynamicRupture', 'dynamic rupture', 'tfc3'),
          ('computePointSources', 'point sources', 'tfc4'),
          ('outsideKernels', 'outside the kernels', 'tfc5'),
          ('io', 'blocking IO', 'tfc7'),
          ('other', 'other', 'tfgrid'))


def figure_timesplit(runs: Sequence[Run]) -> List[Output]:
    """Where a simulation's time went, per run: SeisSol's compute regions,
    the compute outside them -- waiting, communication, scheduling -- and
    the blocking IO, as shares of the total, means over the ranks."""
    out = []
    groups: Dict[Tuple[str, bool], List[Run]] = defaultdict(list)
    for run in runs:
        if run.capture and run.split.get('total'):
            groups[(run.capture.equation, run.lts)].append(run)
    for (eq, lts), mine in sorted(groups.items()):
        mine = sorted(mine, key=lambda r: (r.arch, r.capture.order,
                                           r.capture.datatype, r.name))
        rows = []
        shares = []
        for run in mine:
            total = run.split['total']
            parts = {key: run.split.get(key, 0.0) for key, _, _ in _SPLIT
                     if key != 'other'}
            parts['other'] = max(0.0, total - sum(parts.values()))
            shares.append({k: 100 * v / total for k, v in parts.items()})
            rows.append(dict({'run': run.name, 'arch': run.arch,
                              'configuration': run.configuration,
                              'total_s': total},
                             **{f'{k}_s': v for k, v in parts.items()}))
        ax = texout.Axis(options=[
            'tf', 'xbar stacked', 'bar width=0.6', 'xmin=0, xmax=100',
            'y dir=reverse',
            f'ytick={{{",".join(str(i) for i in range(len(mine)))}}}',
            'yticklabels={' + ','.join(_run_label(r) for r in mine) + '}',
            'yticklabel style={font=\\scriptsize, color=tfink}',
            f'ymin=-0.6, ymax={len(mine) - 0.4}', 'ymajorgrids=false',
            'xlabel={\\% of the simulation}', 'tf legend above',
            'area legend',
            f'height={max(0.3, 0.16 + 0.07 * len(mine)):.2f}\\tfwidth'])
        for key, label, colour in _SPLIT:
            if not any(s.get(key) for s in shares):
                continue
            ax.plots.append(texout.Plot(
                f'fill={colour}, draw=white, line width=0.4pt, mark=none',
                [(s.get(key, 0.0), y) for y, s in enumerate(shares)],
                label))
        tex = texout.picture([ax], comment=(
            'generated by tools/bench/publish.py from SeisSol\'s run '
            'summaries: means over the ranks; other is the total less the '
            'parts'))
        mode = 'local' if lts else 'global'
        caption = (f'Where the time of a SeisSol simulation went, {eq} with '
                   f'{mode} time stepping, per machine and configuration: '
                   'SeisSol\'s compute regions, the compute outside them -- '
                   'waiting, communication, scheduling -- and the blocking '
                   'IO, as shares of the simulation, means over the ranks.')
        out.append(Output(f'timesplit-{eq}-{mode}', 'figure', tex, rows,
                          caption))
    return out


# ----------------------------------------------------------------------
# Assembly
# ----------------------------------------------------------------------

def preview(outputs: Sequence[Output], title: str,
            skipped: Sequence[str] = ()) -> str:
    """A document with every output and its suggested caption, after a
    page that names them and says what was not made, and why."""
    def names(kind: str) -> str:
        return ', '.join(r'\texttt{' + texout.escape(o.name) + '}'
                         for o in outputs if o.kind == kind) or 'none'

    # Ragged: a name breaks nowhere but at the space after its comma.
    contents = ('{\\raggedright\\noindent\\textbf{Tables:} '
                + names('table') + '\n\n\\noindent\\textbf{Figures:} '
                + names('figure') + '\\par}\n')
    if skipped:
        # The math bullet: the text one is a bitmap font where the
        # document has no Type 1 one for it (`_rampup_table`).
        contents += ('\n\\noindent\\textbf{Not made:}\n'
                     '\\begin{itemize}\\small'
                     '\\renewcommand{\\labelitemi}{$\\bullet$}\n'
                     + ''.join(f'\\item {texout.escape(s)}\n'
                               for s in skipped)
                     + '\\end{itemize}\n')
    parts = [contents, '\\clearpage\n']
    for n, o in enumerate(outputs):
        if n and n % 4 == 0:
            # LaTeX holds eighteen floats at most; a preview of every
            # machine, precision and order has many more.
            parts.append('\\clearpage\n')
        where = 'figures' if o.kind == 'figure' else 'tables'
        if o.kind == 'figure':
            parts.append(
                '\\begin{figure}[htbp]\n\\centering\n'
                f'\\input{{{where}/{o.name}}}\n'
                f'\\caption{{{o.caption}}}\n'
                f'\\label{{fig:{o.name}}}\n\\end{{figure}}\n')
        else:
            parts.append(
                '\\begin{table}[htbp]\n\\centering\\footnotesize\n'
                f'\\caption{{{o.caption}}}\n'
                f'\\label{{tab:{o.name}}}\n'
                f'\\input{{{where}/{o.name}}}\n\\end{{table}}\n')
    return ('\\documentclass[10pt]{article}\n'
            '\\usepackage[a4paper,margin=2cm]{geometry}\n'
            + texout.PREAMBLE
            + '\\newlength{\\tfwidth}\\setlength{\\tfwidth}{8.6cm}\n'
            '\\input{tfstyle}\n'
            f'\\title{{{texout.escape(title)}}}\n\\date{{}}\n'
            '\\begin{document}\n\\maketitle\n'
            + '\n'.join(parts) + '\\end{document}\n')


@dataclass
class Extra:
    """What a publication draws on beside the kernel runs: SeisSol's own run
    summaries, clusterings to read the launch costs against, the profiler's
    measurements, and what the builds cost."""
    runs: List[Run] = field(default_factory=list)
    clusterings: List[Tuple[Clustering, Capture]] = field(
        default_factory=list)
    profiles: Dict[ProfileKey, Tuple[float, int]] = field(
        default_factory=dict)
    manifests: List[Dict] = field(default_factory=list)


def build(fieldset: Optional[Field], described: Sequence,
          kernels: Sequence[str], order: int, clock: str,
          extra: Optional[Extra] = None
          ) -> Tuple[List[Output], List[str]]:
    extra = extra or Extra()
    styles = Styles()
    outputs: List[Output] = []
    skipped: List[str] = []

    def add(made, why: str = '') -> None:
        if isinstance(made, tuple):
            made, reasons = made
            skipped.extend(reasons)
        if made is None or made == []:
            if why:
                skipped.append(why)
            return
        outputs.extend(made if isinstance(made, list) else [made])

    # What each kernel is, from the exports alone.
    if described:
        add(table_kernels(described, kernels, order),
            'kernels: no export in double with one simulation')
        add(table_intensity(described, kernels), 'intensity: no exports')
        add(figure_intensity(described, kernels, fieldset, styles),
            'intensity figure: fewer than two orders exported')
        add(figure_operations(described, kernels),
            'operations: no export in double with one simulation')
        add(figure_operators(described, kernels),
            'operators: no operator with its numbers in the exports')
    else:
        skipped.append('kernels, intensity, operations, operators: no '
                       'exports (--exports)')

    # What the kernels did.
    fits: List[Dict] = []
    measured = fieldset is not None and bool(fieldset.points)
    if measured:
        add(table_machines(fieldset), 'machines: no ceilings in the runs')
        add(figure_rooflines(fieldset, kernels, styles))
        add(figure_peaks(fieldset, kernels))
        figs, tabs, why = figure_rampup(fieldset, kernels, styles, clock,
                                        order)
        add(figs)
        add(tabs)
        skipped.extend(why)
        fits = [r for t in tabs for r in t.rows]
    else:
        skipped.append('everything the kernels measured: no runs given')

    # What a time step and a run made of them.
    add(figure_lts(lts_series(extra.clusterings, extra.runs), fits,
                   extra.runs, fieldset),
        'lts: no clustering (--clustering, or a run with several clusters '
        'in --seissol)')
    if measured:
        add(figure_breakdown(fieldset, kernels, styles))
    add(figure_timesplit(extra.runs),
        'timesplit: no SeisSol run summaries (--seissol)')
    add(figure_throughput(extra.runs),
        'throughput: no SeisSol run summaries (--seissol)')

    # Across machines and configurations, and what explains the curves.
    if measured:
        add(table_portability(fieldset, kernels),
            'portability: fewer than two machines with a ceiling')
        add(figure_cascade(fieldset, kernels, order, styles),
            'cascade: fewer than two machines with a ceiling')
        add(figure_configs(fieldset, kernels),
            'configs: one configuration only')
        add(figure_fusion(fieldset, kernels, clock))
        add(figure_lanes(fieldset, kernels, styles),
            'lanes: no executed flops, or one order only')
        add(figure_resources(fieldset, kernels, styles),
            'resources: no registers or code sizes recorded')
        add(figure_precision(fieldset, kernels),
            'precision: no kernel measured in both precisions')
        add(figure_simulations(fieldset, kernels, order, styles),
            'simulations: one simulation count only')
        add(figure_traffic(fieldset, extra.profiles, kernels),
            'traffic: no profile.json with DRAM counters')
    add(table_buildcost(extra.manifests, fieldset),
        'buildcost: no manifest that recorded what a build cost')
    return outputs, skipped


def write(outputs: Sequence[Output], out: Path, title: str,
          manifest: Dict) -> None:
    for sub in ('figures', 'tables', 'data', 'standalone'):
        (out / sub).mkdir(parents=True, exist_ok=True)
    (out / 'tfstyle.tex').write_text(texout.style())
    for o in outputs:
        where = 'figures' if o.kind == 'figure' else 'tables'
        (out / where / f'{o.name}.tex').write_text(o.tex)
        _write_csv(o.rows, out / 'data' / f'{o.name}.csv')
        if o.kind == 'figure':
            (out / 'standalone' / f'{o.name}.tex').write_text(
                texout.standalone(f'../figures/{o.name}', '../tfstyle'))
    (out / 'preview.tex').write_text(preview(outputs, title,
                                             manifest.get('skipped', ())))
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2,
                                                  default=str))


#: The clustering `--clustering synthetic` stands for: written down to lay
#: the figures out with until a mesh's is at hand, and named as such in
#: every figure drawn from it.
SYNTHETIC_CLUSTERING = HERE / 'clusterings' / 'synthetic-clustering.csv'


def clustering_arg(text: str, order: int) -> Tuple[Clustering, Capture]:
    """`CSV[@TAG]`: a clustering, and the export its kernels are at --
    `elastic-4-d`, or with the simulations, `elastic-4-d-s1`."""
    path, _, tag = text.partition('@')
    source = SYNTHETIC_CLUSTERING if path == 'synthetic' else Path(path)
    tag = tag or f'elastic-{order}-d'
    capture = Capture.from_tag(tag if tag.count('-') == 3 else f'{tag}-s1')
    if capture is None:
        raise SystemExit(f'--clustering {text}: {tag!r} is no export tag '
                         '(equation-order-precision, e.g. elastic-4-d)')
    try:
        return read_clustering(source), capture
    except (OSError, ValueError, KeyError) as exc:
        raise SystemExit(f'--clustering {text}: {exc}') from exc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('paths', type=Path, nargs='*',
                    help='campaign trees or bench.json files, as compare.py '
                         'takes them; none for the kernels alone')
    ap.add_argument('--exports', type=Path, default=HERE / 'data',
                    help='the exports the kernels were measured from, for '
                         'the formulas and the static figures')
    ap.add_argument('--out', type=Path, default=Path('publish'))
    ap.add_argument('--kernels', default=','.join(DEFAULT_KERNELS),
                    help='comma-separated kernel names, in the order the '
                         'tables list them')
    ap.add_argument('--order', type=int, default=4,
                    help='the order the kernel table shows the formulas at, '
                         'and the ramp-up its kernels at')
    ap.add_argument('--flops', default='executed', choices=sorted(FLOP_FIELD),
                    help='which flop count a rate and an intensity divide '
                         '(as in plot.py and compare.py)')
    ap.add_argument('--rampup-clock', default='wall',
                    choices=('wall', 'event'),
                    help='the clock of the ramp-up: the wall clock over '
                         'back-to-back launches, launch overhead and all '
                         '(default), or the device\'s')
    ap.add_argument('--seissol', type=Path, action='append', default=[],
                    help='SeisSol output directories or <prefix>-summary.json '
                         'files: what the runs got through, where their time '
                         'went, and their clusterings (repeatable)')
    ap.add_argument('--clustering', action='append', default=[],
                    metavar='CSV[@TAG]',
                    help='a <prefix>-clustering.csv as SeisSol writes it, or '
                         '"synthetic" for the series in clusterings/, to read '
                         'the launch costs against; TAG names the kernels\' '
                         'export, e.g. elastic-4-d (default: elastic at '
                         '--order, double) (repeatable)')
    ap.add_argument('--title', default='TensorForge measurements')
    ap.add_argument('--compile', action='store_true',
                    help='run pdflatex on preview.tex and on every '
                         'standalone figure')
    args = ap.parse_args()

    set_flop_basis(args.flops)
    kernels = [k.strip() for k in args.kernels.split(',') if k.strip()]
    fieldset = collect(args.paths) if args.paths else None

    import describe as dsc
    described = []
    if args.exports and args.exports.exists():
        described = dsc.describe_all([args.exports], kernels)

    extra = Extra(runs=collect_runs(args.seissol),
                  clusterings=[clustering_arg(text, args.order)
                               for text in args.clustering],
                  profiles=_profiles(args.paths),
                  manifests=_manifests(args.paths))
    outputs, skipped = build(fieldset, described, kernels, args.order,
                             args.rampup_clock, extra)
    if not outputs:
        print('nothing to publish:\n  ' + '\n  '.join(skipped),
              file=sys.stderr)
        return 2
    manifest = {
        'inputs': [str(p) for p in args.paths],
        'seissol': [str(r.path) for r in extra.runs],
        'clusterings': [{'name': c.name, 'synthetic': c.synthetic,
                         'kernels': cap.tag}
                        for c, cap in extra.clusterings],
        'exports': str(args.exports), 'kernels': kernels,
        'flops': args.flops, 'rampup_clock': args.rampup_clock,
        'step': {eq: {k: asdict(launch) for k, launch in step.items()}
                 for eq, step in STEP.items()},
        'outputs': [{'name': o.name, 'kind': o.kind, 'rows': len(o.rows)}
                    for o in outputs],
        'skipped': skipped,
    }
    write(outputs, args.out, args.title, manifest)
    for o in outputs:
        print(f'written: {o.kind} {o.name}')
    for s in skipped:
        print(f'skipped {s}')
    if args.compile:
        targets = [args.out / 'preview.tex'] + sorted(
            (args.out / 'standalone').glob('*.tex'))
        for path in targets:
            ok, why = texout.compile_tex(path)
            print(f'{"compiled" if ok else "FAILED"}: {path}'
                  + (f' -- {why}' if why else ''))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
