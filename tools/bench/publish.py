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
    python3 tools/bench/publish.py --out pub/          # the kernels alone

## What it writes

| output | from | the question |
|---|---|---|
| `tables/kernels` | exports | what does each kernel compute? |
| `tables/intensity`, `figures/intensity` | exports | how many flops per byte, at each order and precision? |
| `tables/machines` | ceilings | what did each machine turn out to do, per precision? |
| `figures/roofline-*` | runs, ceilings | where does each kernel sit, and how does it move with the order? |
| `figures/peak-*` | runs, ceilings | how close to the roof, the bus and the FMA rate does a full launch get? |
| `figures/rampup-*`, `tables/rampup-*` | runs | from how many elements on is a launch worth its cost? |
| `figures/breakdown-*` | runs | where does one time step's time go, per order? |
| `tables/portability` | runs, ceilings | how evenly does a kernel reach each machine's roof? |
| `figures/configs-*` | runs | does a configuration -- or another generator -- help? |

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
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import texout                                                    # noqa: E402
from capture import DATATYPE                                     # noqa: E402
from compare import Ceiling, Field, Point, collect               # noqa: E402
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

def table_portability(fieldset: Field, kernels: Sequence[str]
                      ) -> Optional[Output]:
    """Pennycook's performance portability per kernel, order and precision:
    the harmonic mean over the machines of the fraction of each one's own
    roof a full launch reaches at the kernel's intensity.  The machines are
    those with a roof in the precision that ran the equation in it; one of
    them that tried the kernel and produced no number makes it zero, as the
    definition has it, and one that was not asked leaves it open -- a gap
    in a campaign is not a failure of the code."""
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
            ns = _nanos(p.row, 'event') if p is not None else None
            return ns / p.batch if ns and p.batch else None

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
        # The ticks a ratio is read off, powers of two around one.
        lo = 2.0 ** math.floor(math.log2(min(ratios + [1.0]) / 1.05))
        hi = 2.0 ** math.ceil(math.log2(max(ratios + [1.0]) * 1.05))
        ticks = [2.0 ** e for e in range(int(math.log2(lo)),
                                         int(math.log2(hi)) + 1)]
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
            f'height={0.12 + 0.05 * len(groups):.2f}\\tfwidth',
            f'xlabel={{speed-up over \\texttt{{{texout.escape(ref)}}}}}'])
        ax.plots = plots
        # One, and a hairline between one kernel's orders and the next's.
        ax.raw.append('\\draw[tf rule] ({axis cs:1,0}|-{rel axis cs:0,0}) '
                      '-- ({axis cs:1,0}|-{rel axis cs:0,1});\n')
        for i in range(1, len(groups)):
            if groups[i][0] != groups[i - 1][0]:
                ax.raw.append(
                    f'\\draw[tfgrid] ({{rel axis cs:0,0}}|-{{axis cs:1,'
                    f'{i - 0.5}}}) -- ({{rel axis cs:1,0}}|-{{axis cs:1,'
                    f'{i - 0.5}}});\n')
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


def build(fieldset: Optional[Field], described: Sequence,
          kernels: Sequence[str], order: int, clock: str
          ) -> Tuple[List[Output], List[str]]:
    styles = Styles()
    outputs: List[Output] = []
    skipped: List[str] = []

    def add(made, why: str = '') -> None:
        if made is None or made == []:
            if why:
                skipped.append(why)
            return
        outputs.extend(made if isinstance(made, list) else [made])

    if described:
        add(table_kernels(described, kernels, order),
            'kernels: no export in double with one simulation')
        add(table_intensity(described, kernels), 'intensity: no exports')
        add(figure_intensity(described, kernels, fieldset, styles),
            'intensity figure: fewer than two orders exported')
    else:
        skipped.append('kernels, intensity: no exports (--exports)')
    if fieldset is None or not fieldset.points:
        skipped.append('everything measured: no runs given')
        return outputs, skipped
    add(table_machines(fieldset), 'machines: no ceilings in the runs')
    made, why = figure_rooflines(fieldset, kernels, styles)
    add(made)
    skipped.extend(why)
    made, why = figure_peaks(fieldset, kernels)
    add(made)
    skipped.extend(why)
    figs, tabs, why = figure_rampup(fieldset, kernels, styles, clock,
                                    order)
    add(figs)
    add(tabs)
    skipped.extend(why)
    made, why = figure_breakdown(fieldset, kernels, styles)
    add(made)
    skipped.extend(why)
    add(table_portability(fieldset, kernels),
        'portability: fewer than two machines with a ceiling')
    add(figure_configs(fieldset, kernels),
        'configs: one configuration only')
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

    outputs, skipped = build(fieldset, described, kernels, args.order,
                             args.rampup_clock)
    if not outputs:
        print('nothing to publish:\n  ' + '\n  '.join(skipped),
              file=sys.stderr)
        return 2
    manifest = {
        'inputs': [str(p) for p in args.paths],
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
