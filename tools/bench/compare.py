# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Charts across runs: several kernels, several configurations, several
machines, several orders.

`plot.py` draws one run well. Everything it answers is about one device and
one build: where a kernel saturates, whether it is allowed to go faster,
which of two builds is ahead. The questions that come after a campaign are
about the *field* rather than a point in it -- does this configuration help
everywhere or only here, does the picture hold as the order grows, does a
kernel that is memory-bound on one machine stay memory-bound on the next.

So this reads a whole tree of runs at once and writes a PDF. It launches
nothing and builds nothing; it is the offline half of `plot.py`, and it
shares that file's `Canvas`, so a chart from either reads the same.

    python3 tools/bench/compare.py eval/runs --pdf --out eval/compare
    python3 tools/bench/compare.py a/bench.json b/bench.json --pdf

## What each view is for

| view | the question |
|---|---|
| `devices` | which machines are these, and what did they turn out to do? |
| `bandwidth-peak` | how close is the measured ceiling to the one the memory bus allows? |
| `intensity` | is this kernel memory- or compute-bound, on each machine, and by how far? |
| `kernels` | what does each kernel cost and carry -- the table behind the charts |
| `configs` | does a configuration help everywhere, or only on one kernel? |
| `order` | how does the cost move as the discretisation order grows? |
| `archs` | the same kernel on each machine, as it ran |
| `portability` | the same, with each machine's own ceiling divided out |

## Why not matplotlib

The question came up, and the answer is the one `plot.py` already gives for
the cluster: nothing here is a chart type a library would draw better. These
are lines, bars, dots and rules. What a library would add is a dependency on
a login node, a second drawing idiom beside the one in `Canvas`, and a
second set of defaults to keep in step with it. The one thing it would
genuinely buy -- a PDF writer -- is `pdfout.py`, which is two hundred lines
because a chart is lines and text.

## What these views deliberately do not do

They do not average a kernel's configurations together, for the reason
`plot.py` gives. They do not compare two machines by absolute rate without
saying so: `archs` is the honest absolute view and `portability` is the one
that divides each machine's own ceiling out, and a reader who wants to rank
machines wants the second. And a ratio is never taken across a gap -- a
kernel missing from one run is missing from the comparison rather than
being carried at its neighbour's value.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from pdfout import Page, PText, write_pdf, table_page          # noqa: E402
from plot import (Canvas, FLOP_FIELD, FLOP_LABEL, MARKS, PALETTE, RATE_LABEL,
                  _fmt, _results, flop_basis, flops_of, index_html,
                  intensity_of, rate, rate_label, series_style,
                  set_flop_basis)                            # noqa: E402

#: Magnitude against a midpoint of 1: two hues and a neutral grey between
#: them, never a hue at the middle (a rainbow midpoint reads as a third
#: category).  Cool is "the reference is ahead", warm is "the candidate is".
DIVERGING = ('#184f95', '#2a78d6', '#86b6ef', '#e8e8e4',
             '#f0a58c', '#e34948', '#a82424')


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------

#: `elastic-6-s-s1` (what `evaluate.py` names an export) and
#: `elastic-linearck-o6-s` (what the recorded suite calls a config).
_ORDER_PATTERNS = (re.compile(r'(?:^|[-_])o(\d+)(?:[-_]|$)'),
                   re.compile(r'^[A-Za-z]+-(\d+)-'))


def order_of(path: Path, rows: Sequence[Dict]) -> Optional[int]:
    """The discretisation order this run is of, from the rows or the path.

    A row would be the better place and does not carry it today, so the
    path is read -- but only the path, never the *content*: guessing an
    order from a kernel's shapes would be a number nobody recorded.
    Returns `None` where neither says, and the order views then leave the
    run out rather than placing it at zero.
    """
    for row in rows:
        if row.get('order'):
            return int(row['order'])
    for part in reversed(path.parts):
        for pattern in _ORDER_PATTERNS:
            m = pattern.search(part)
            if m:
                return int(m.group(1))
    return None


@dataclass
class Point:
    """One measurement, flattened out of whichever file carried it."""
    workload: str
    config: str
    arch: str
    backend: str
    device: str
    datatype: str
    batch: int
    row: Dict
    order: Optional[int] = None
    run: str = ''

    @property
    def target(self) -> str:
        return f'{self.backend}-{self.arch}'

    @property
    def intensity(self) -> Optional[float]:
        """Flops per compulsory byte, on the basis the charts are drawn
        against -- a property of the operation, which is why it is the
        same on every machine and the ridge is not."""
        return intensity_of(self.row)


@dataclass
class Ceiling:
    """What one machine turned out to do (`roofline.py`), and what it is
    built to do where the device says so."""
    arch: str
    backend: str = ''
    device: str = ''
    compute_units: int = 0
    flops_per_s: float = 0.0
    bytes_per_s: float = 0.0
    memory_bytes: int = 0
    peak_bytes_per_s: float = 0.0
    peak_flops_per_s: float = 0.0

    @property
    def ridge(self) -> float:
        return self.flops_per_s / self.bytes_per_s if self.bytes_per_s else 0.0

    def bound(self, intensity: float) -> float:
        return min(self.flops_per_s, self.bytes_per_s * intensity)


@dataclass
class Field:
    """Every point of a campaign, and the ceilings that go with them."""
    points: List[Point] = field(default_factory=list)
    ceilings: Dict[str, Ceiling] = field(default_factory=dict)

    @property
    def archs(self) -> List[str]:
        return sorted({p.arch for p in self.points})

    @property
    def workloads(self) -> List[str]:
        return sorted({p.workload for p in self.points})

    @property
    def configs(self) -> List[str]:
        return sorted({p.config for p in self.points})

    @property
    def orders(self) -> List[int]:
        return sorted({p.order for p in self.points if p.order is not None})

    def device_of(self, arch: str) -> str:
        for p in self.points:
            if p.arch == arch and p.device:
                return p.device
        return self.ceilings.get(arch, Ceiling(arch)).device or arch

    def at(self, **keys) -> List[Point]:
        return [p for p in self.points
                if all(getattr(p, k) == v for k, v in keys.items())]

    def largest_batch(self, points: Sequence[Point]) -> List[Point]:
        """One point per (workload, config, arch, order): the biggest launch
        there is, because a ratio taken on the rising part of a scaling
        curve is a ratio at an arbitrary place on a slope."""
        best: Dict[Tuple, Point] = {}
        for p in points:
            key = (p.workload, p.config, p.arch, p.order, p.datatype)
            if key not in best or p.batch > best[key].batch:
                best[key] = p
        return list(best.values())


def _load_ceiling(path: Path) -> Optional[Ceiling]:
    try:
        blob = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    c = blob.get('ceiling', blob)
    if not c.get('arch'):
        return None
    return Ceiling(
        arch=c['arch'], backend=c.get('backend', ''),
        device=c.get('device', ''),
        compute_units=int(c.get('compute_units') or c.get('cus') or 0),
        flops_per_s=float(c.get('flops_per_s') or 0.0),
        bytes_per_s=float(c.get('bytes_per_s') or 0.0),
        memory_bytes=int(c.get('memory_bytes') or 0),
        peak_bytes_per_s=float(c.get('peak_bytes_per_s') or 0.0),
        peak_flops_per_s=float(c.get('peak_flops_per_s') or 0.0))


def collect(targets: Sequence[Path]) -> Field:
    """Every `bench.json` under the paths given, with the ceilings beside
    them.

    A directory is walked; a file is read as it stands. The run's name is
    its directory, which is what the order is parsed out of, so pointing
    this at a tree `evaluate.py` wrote needs no further argument.
    """
    out = Field()
    seen: set = set()
    for target in targets:
        benches = ([target] if target.is_file()
                   else sorted(target.rglob('bench.json')))
        for bench in benches:
            if bench.resolve() in seen:
                continue
            seen.add(bench.resolve())
            try:
                rows = [r for r in _results(json.loads(bench.read_text()))
                        if not r.get('error')]
            except (OSError, ValueError):
                continue
            order = order_of(bench.parent, rows)
            run = bench.parent.name
            for row in rows:
                if not row.get('workload'):
                    continue
                out.points.append(Point(
                    workload=row['workload'],
                    config=row.get('config') or 'default',
                    arch=row.get('arch') or '?',
                    backend=row.get('backend') or '?',
                    device=row.get('device') or '',
                    datatype=row.get('datatype') or '',
                    batch=int(row.get('batch') or 0),
                    row=row, order=order, run=run))
            for name in ('ceiling.json', 'roofline.json'):
                ceiling = _load_ceiling(bench.parent / name)
                if ceiling:
                    out.ceilings.setdefault(ceiling.arch, ceiling)
        # A tree `evaluate.py` wrote keeps its ceilings in one place rather
        # than beside every run, which is the point of measuring them once.
        if target.is_dir():
            for name in sorted(target.rglob('*.json')):
                if name.parent.name == 'ceiling' or name.name in (
                        'ceiling.json', 'roofline.json'):
                    ceiling = _load_ceiling(name)
                    if ceiling:
                        out.ceilings.setdefault(ceiling.arch, ceiling)
    return out


# ----------------------------------------------------------------------
# Helpers the views share
# ----------------------------------------------------------------------

def _colors(names: Sequence[str]) -> Dict[str, str]:
    """A fixed colour per name, assigned in order and never recycled within
    a chart: beyond the palette the mark carries the identity instead."""
    return {n: series_style(i)[0] for i, n in enumerate(names)}


def _marks(names: Sequence[str]) -> Dict[str, str]:
    """The second channel, from the same rule `plot.py` draws by."""
    return {n: series_style(i)[1] for i, n in enumerate(names)}


def _geomean(values: Iterable[float]) -> Optional[float]:
    vals = [v for v in values if v and v > 0]
    if not vals:
        return None
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


def _diverging(ratio: float, span: float = 2.0) -> str:
    """A colour for a ratio against 1, on the diverging ramp.

    `span` is the ratio that saturates the ramp -- 2.0 means twice and half
    are the ends, which keeps the interesting band readable instead of
    being compressed by one outlier.
    """
    if not ratio or ratio <= 0:
        return '#e8e8e4'
    t = math.log(ratio) / math.log(span)
    t = max(-1.0, min(1.0, t))
    idx = int(round((t + 1) / 2 * (len(DIVERGING) - 1)))
    return DIVERGING[len(DIVERGING) - 1 - idx]


LOWER_IS_BETTER = ('ns_per_element', 'ns')


# ----------------------------------------------------------------------
# Metadata: the machines
# ----------------------------------------------------------------------

def view_devices(fieldset: Field) -> Optional[Tuple[Page, str]]:
    """What ran this, as a table.

    A table and not a chart on purpose: six unrelated quantities about four
    machines is a reference a reader looks things up in, and a chart of it
    would be four bar charts sharing an axis they have no business sharing.
    """
    archs = fieldset.archs
    if not archs:
        return None
    headers = ['device', 'arch', 'backend', 'CUs', 'memory',
               'FLOP/s measured', 'GB/s measured', 'GB/s peak', 'ridge']
    rows = []
    for arch in archs:
        c = fieldset.ceilings.get(arch)
        rows.append([
            fieldset.device_of(arch), arch,
            next((p.backend for p in fieldset.points if p.arch == arch), '?'),
            str(c.compute_units or '?') if c else '?',
            f'{c.memory_bytes / 2 ** 30:.0f} GiB' if c and c.memory_bytes
            else 'unknown',
            f'{c.flops_per_s / 1e12:.2f} T' if c and c.flops_per_s else '—',
            f'{c.bytes_per_s / 1e9:.0f}' if c and c.bytes_per_s else '—',
            f'{c.peak_bytes_per_s / 1e9:.0f}' if c and c.peak_bytes_per_s
            else 'unknown',
            f'{c.ridge:.1f} F/B' if c and c.ridge else '—',
        ])
    note = ('"measured" is what `roofline.py` got out of the machine; "peak" '
            'is what the memory bus allows, where the device reports its '
            'clock and width. A theoretical FLOP/s is not listed: it needs '
            'a cores-per-unit table per architecture, which would be a '
            'guess for any part newer than the table.')
    return table_page('Machines', headers, rows, note)


def view_bandwidth_peak(fieldset: Field) -> Optional[str]:
    """How much of its own bus each machine actually delivers.

    The *fraction* and not the pair of absolutes: a set that holds a laptop
    part and a data-centre one spans two orders of magnitude, and on a
    shared linear axis the laptop is a line at the bottom. The absolute
    figures are in the `devices` table, where a reader looks a number up;
    what a chart is for here is the comparison, and a streaming loop
    reaching 84 % of the bus against one reaching 45 % is the finding.
    """
    have = [(a, fieldset.ceilings[a]) for a in fieldset.archs
            if a in fieldset.ceilings
            and fieldset.ceilings[a].bytes_per_s
            and fieldset.ceilings[a].peak_bytes_per_s]
    if not have:
        return None
    c = Canvas('What each machine delivers of its own memory bus', '',
               'measured ÷ bus', width=max(520, 110 * len(have) + 180),
               tilted_labels=True,
               note='The bus figure is the device\'s own memory clock and '
                    'width, which is arithmetic rather than an estimate; '
                    'the measured one is a streaming loop. Absolutes are in '
                    'the machines table.')
    fracs = [x.bytes_per_s / x.peak_bytes_per_s for _, x in have]
    c.fit([0, len(have)], fracs + [1.0], y_from_zero=True)
    c.axes(xticks=[], ytick_label=lambda v: f'{v:.0%}')
    c.rule(1.0, 'the bus', '#777')
    step = (c.W - c.L - c.R) / max(len(have), 1)
    for i, (arch, ceil) in enumerate(have):
        frac = ceil.bytes_per_s / ceil.peak_bytes_per_s
        c.bar(c.L + step * (i + 0.25), step * 0.5, frac, PALETTE[0],
              f'{fieldset.device_of(arch)}: {ceil.bytes_per_s / 1e9:.0f} of '
              f'{ceil.peak_bytes_per_s / 1e9:.0f} GB/s ({frac:.1%})')
        c.tilted_label(c.L + step * (i + 0.5), fieldset.device_of(arch))
    return c.render()


# ----------------------------------------------------------------------
# Metadata: the kernels
# ----------------------------------------------------------------------

def _intensity_table(fieldset: Field) -> Dict[str, float]:
    """Compulsory flops per byte, per workload.

    One number per kernel and not per machine: it is a property of the
    operation. Taken at the largest launch, where the per-element figures
    have settled.
    """
    out: Dict[str, float] = {}
    for p in fieldset.largest_batch(fieldset.points):
        i = p.intensity
        if i:
            out.setdefault(p.workload, i)
    return out


def view_intensity(fieldset: Field) -> Optional[str]:
    """Where each kernel sits against each machine's ridge.

    The quantity is the kernel's intensity divided by the machine's ridge,
    on a log axis with a rule at one. Below the rule the kernel is
    memory-bound on that machine, above it compute-bound, and the distance
    is how far from the corner it is -- which is the part a yes/no label
    throws away. A kernel can be memory-bound on one machine and
    compute-bound on the next, which is exactly why the rule cannot be a
    property of the kernel.
    """
    intensities = _intensity_table(fieldset)
    archs = [a for a in fieldset.archs
             if a in fieldset.ceilings and fieldset.ceilings[a].ridge]
    names = [w for w in fieldset.workloads if w in intensities]
    if not archs or len(names) < 1:
        return None
    colors, marks = _colors(archs), _marks(archs)
    c = Canvas('Memory- or compute-bound, per machine',
               '', 'arithmetic intensity ÷ ridge', ylog=True,
               width=max(760, 46 * len(names) + 220), legend_width=190,
               tilted_labels=True,
               note='Below 1 the machine runs out of bandwidth first; above '
                    'it, of arithmetic. The ridge is the measured one.')
    ratios = [intensities[w] / fieldset.ceilings[a].ridge
              for w in names for a in archs]
    c.fit([0, len(names)], ratios + [1.0], ypad=1.6)
    c.axes(xticks=[])
    c.rule(1.0, 'ridge — bandwidth below, arithmetic above', '#777')
    step = (c.W - c.L - c.R) / max(len(names), 1)
    for i, w in enumerate(names):
        for a in archs:
            ridge = fieldset.ceilings[a].ridge
            r = intensities[w] / ridge
            x = c.L + step * (i + 0.5)
            c._mark(x, c.py(r), colors[a], marks[a],
                    f'{w} on {a}: {intensities[w]:.2f} flop/byte against a '
                    f'ridge of {ridge:.1f} — '
                    + ('memory-bound' if r < 1 else 'compute-bound')
                    + f', by {max(r, 1 / r):.1f}x')
        c.tilted_label(c.L + step * (i + 0.5), w)
    for a in archs:
        c.legend.append((f'{a} (ridge {fieldset.ceilings[a].ridge:.0f})',
                         colors[a]))
    return c.render()


def view_flopmix(fieldset: Field) -> Optional[str]:
    """What the kernel issues, against what the operation needs.

    Three counts and two gaps.  The first gap is the sparsity a dense
    implementation pays for and the generator partly avoids; the second is
    the lanes: a destination twenty rows wide computes on thirty-two, and
    the multiply-add that fills the other twelve is issued, retired and
    counted by the profiler.  A roofline drawn on the issued count is
    honest about the machine and says nothing about this, which is why it
    is a chart of its own.

    Drawn as a fraction of the issued count, so kernels of wildly
    different size compare: the bar is what the operation needs and the
    space above it is arithmetic spent on nothing.
    """
    points = fieldset.largest_batch(fieldset.points)
    by_workload: Dict[str, Point] = {}
    for p in points:
        if p.row.get('executed_flops') and p.row.get('nonzero_flops'):
            by_workload.setdefault(p.workload, p)
    if not by_workload:
        return None
    names = sorted(by_workload)
    c = Canvas('How much of the issued arithmetic is needed', '',
               'fraction of what the kernel issues',
               width=max(760, 54 * len(names) + 200), legend_width=170,
               tilted_labels=True,
               note='The bar is what the operation needs; the first segment '
                    'above it is what a dense implementation adds, and the '
                    'rest is the lanes a narrow destination leaves idle. '
                    'Per element at the largest launch.')
    c.fit([0, len(names)], [0.0, 1.0], y_from_zero=True)
    c.axes(xticks=[], ytick_label=lambda v: f'{v:.0%}')
    c.rule(1.0, 'issued', '#777')
    step = (c.W - c.L - c.R) / max(len(names), 1)
    for i, w in enumerate(names):
        p = by_workload[w]
        issued = float(p.row['executed_flops'])
        needed = float(p.row['nonzero_flops']) / issued
        dense = min(float(p.row.get('flops') or 0) / issued, 1.0)
        x = c.L + step * (i + 0.22)
        c.bar(x, step * 0.56, needed, PALETTE[0],
              f'{w}: needs {needed:.1%} of what it issues')
        if dense > needed:
            c.bar(x, step * 0.56, dense, PALETTE[1],
                  f'{w}: a dense implementation would perform {dense:.1%}',
                  y_base=needed)
        if dense < 1.0:
            c.bar(x, step * 0.56, 1.0, PALETTE[4],
                  f'{w}: the lanes add the rest, to {issued:,.0f} flop',
                  y_base=dense)
        c.tilted_label(c.L + step * (i + 0.5), w)
    c.legend.extend([('needed', PALETTE[0]), ('dense adds', PALETTE[1]),
                     ('idle lanes add', PALETTE[4])])
    return c.render()


def view_kernels(fieldset: Field) -> Optional[Tuple[Page, str]]:
    """What each kernel is, as a table: the numbers the charts divide by.

    The three flop counts are kept apart rather than reconciled -- what the
    operation needs, what a dense implementation performs, and what the
    built kernel performs are three different things and the generator
    reports the third.
    """
    points = fieldset.largest_batch(fieldset.points)
    if not points:
        return None
    archs = [a for a in fieldset.archs
             if a in fieldset.ceilings and fieldset.ceilings[a].ridge]
    by_workload: Dict[str, Point] = {}
    for p in points:
        by_workload.setdefault(p.workload, p)
    headers = ['kernel', 'needed', 'dense', 'issued', 'useful',
               'byte/element', f'flop/byte ({flop_basis()})'] + [
                   f'bound on {a}' for a in archs]
    rows = []
    for w in sorted(by_workload):
        p = by_workload[w]
        batch = p.batch or 1
        b = p.row.get('bytes')
        needed = p.row.get('nonzero_flops')
        dense = p.row.get('flops')
        issued = p.row.get('executed_flops')
        inten = p.intensity
        cells = [w,
                 f'{needed / batch:,.0f}' if needed else '—',
                 f'{dense / batch:,.0f}' if dense else '—',
                 f'{issued / batch:,.0f}' if issued else '—',
                 f'{needed / issued:.0%}' if needed and issued else '—',
                 f'{b / batch:,.0f}' if b else '—',
                 f'{inten:.2f}' if inten else '—']
        for a in archs:
            ridge = fieldset.ceilings[a].ridge
            if not inten or not ridge:
                cells.append('—')
                continue
            r = inten / ridge
            cells.append(('memory' if r < 1 else 'compute')
                         + f' {max(r, 1 / r):.1f}x')
        rows.append(cells)
    note = ('Per element at the largest launch. The three counts are kept '
            'apart because they are three things: "needed" is the operation '
            'with its structural zeros taken out, "dense" is what a dense '
            'implementation performs, and "issued" is what the built kernel '
            'performs -- a destination twenty rows wide computes on '
            'thirty-two lanes. "useful" is needed over issued, which is the '
            'arithmetic the generator spends to nothing. "bound" is the '
            'intensity against that machine\'s measured ridge, and the '
            'factor is how far from the corner.')
    return table_page('Kernels', headers, rows, note)


# ----------------------------------------------------------------------
# Comparisons
# ----------------------------------------------------------------------

def view_configs(fieldset: Field, what: str = 'ns_per_element',
                 reference: Optional[str] = None) -> Optional[str]:
    """Every configuration against a reference, per kernel.

    One row per kernel and one mark per configuration, so a configuration
    that helps everywhere reads as a column and one that helps a single
    kernel reads as a single mark away from the rest. That shape is the
    question -- "does this generalise" -- and a bar chart per kernel buries
    it in a forest.
    """
    configs = fieldset.configs
    if len(configs) < 2:
        return None
    ref = reference or ('baseline' if 'baseline' in configs else configs[0])
    if ref not in configs:
        return None
    others = [c for c in configs if c != ref]
    table: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    for p in fieldset.largest_batch(fieldset.points):
        v = rate(p.row, what)
        if v:
            table[(p.workload, p.config)].append(v)
    names = [w for w in fieldset.workloads
             if (w, ref) in table and any((w, o) in table for o in others)]
    if not names:
        return None
    lower_better = what in LOWER_IS_BETTER
    colors, marks = _colors(others), _marks(others)
    c = Canvas(f'Configurations against {ref} — {rate_label(what)}',
               '', f'ratio to {ref}', ylog=True,
               width=max(760, 46 * len(names) + 220), legend_width=190,
               tilted_labels=True,
               note='Above 1 is faster. Each mark is one configuration on '
                    'one kernel, at the largest launch, over every machine '
                    'that ran it.')
    ratios: Dict[Tuple[str, str], float] = {}
    for w in names:
        base = _geomean(table[(w, ref)])
        for o in others:
            cand = _geomean(table.get((w, o), []))
            if base and cand:
                ratios[(w, o)] = base / cand if lower_better else cand / base
    if not ratios:
        return None
    c.fit([0, len(names)], list(ratios.values()) + [1.0], ypad=1.4)
    c.axes(xticks=[])
    c.rule(1.0, ref, '#777')
    step = (c.W - c.L - c.R) / max(len(names), 1)
    for i, w in enumerate(names):
        for j, o in enumerate(others):
            r = ratios.get((w, o))
            if not r:
                continue
            x = c.L + step * (i + 0.3 + 0.4 * j / max(len(others) - 1, 1))
            c._mark(x, c.py(r), colors[o], marks[o],
                    f'{w}: {o} is {r:.3f}x {ref}')
        c.tilted_label(c.L + step * (i + 0.5), w)
    for o in others:
        geo = _geomean([v for (w, oo), v in ratios.items() if oo == o])
        c.legend.append((f'{o} (geo {geo:.3f}x)' if geo else o, colors[o]))
    return c.render()


def view_order(fieldset: Field, what: str = 'ns_per_element',
               arch: Optional[str] = None) -> Optional[str]:
    """Cost against the discretisation order, one line per kernel.

    The axis is the order and not the problem size, because that is the
    knob a solve is configured with. The cost per element grows steeply
    with it -- the operators are larger and there are more of them -- so
    the y axis is logarithmic and a kernel that grows *less* steeply than
    its neighbours is the thing to see.
    """
    orders = fieldset.orders
    if len(orders) < 2:
        return None
    points = [p for p in fieldset.largest_batch(fieldset.points)
              if p.order is not None and (arch is None or p.arch == arch)]
    series: Dict[str, Dict[int, List[float]]] = defaultdict(
        lambda: defaultdict(list))
    for p in points:
        v = rate(p.row, what)
        if v:
            series[p.workload][p.order].append(v)
    series = {k: v for k, v in series.items() if len(v) > 1}
    if not series:
        return None
    names = sorted(series)
    colors, marks = _colors(names), _marks(names)
    # The points first, the axis second: pooling several machines into one
    # curve and then fitting the axis to the *unpooled* values leaves the
    # scale a decade below anything drawn, and half the chart empty.
    curves = {name: [(o, _geomean(series[name].get(o, [])) or None)
                     for o in orders] for name in names}
    drawn = [v for pts in curves.values() for _, v in pts if v]
    if not drawn:
        return None
    where = f' on {arch}' if arch else ''
    pooled = (arch is None
              and len({p.arch for p in points}) > 1)
    c = Canvas(f'Cost against order{where} — {rate_label(what)}',
               'discretisation order', rate_label(what), ylog=True,
               width=760, legend_width=200,
               note='One point per order at the largest launch; a gap is a '
                    'configuration that was not run rather than a zero.'
                    + (' Pooled over every machine as a geometric mean -- '
                       'pass --per-arch-order for one chart each.'
                       if pooled else ''))
    c.fit(orders, drawn)
    c.axes(xticks=orders, xtick_label=lambda v: f'{int(v)}')
    for name in names:
        pts = curves[name]
        c.line(pts, colors[name], label=name, mark=marks[name],
               tips=[f'{name} @ order {o}: '
                     + (f'{v:.4g}' if v else 'not run')
                     for o, v in pts])
    return c.render()


def view_archs(fieldset: Field, what: str = 'gflops') -> Optional[str]:
    """The same kernel on each machine, as it ran.

    Absolute and unapologetic about it: this says which machine finished
    first, which is a fair question and a different one from which machine
    was used well. `portability` is the second question.
    """
    archs = fieldset.archs
    if len(archs) < 2:
        return None
    table: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    for p in fieldset.largest_batch(fieldset.points):
        v = rate(p.row, what)
        if v:
            table[(p.workload, p.arch)].append(v)
    names = [w for w in fieldset.workloads
             if sum(1 for a in archs if (w, a) in table) > 1]
    if not names:
        return None
    colors = _colors(archs)
    c = Canvas(f'Each machine on each kernel — {rate_label(what)}', '',
               rate_label(what), ylog=True,
               width=max(760, 60 * len(names) + 220), legend_width=190,
               tilted_labels=True,
               note='At the largest launch each ran. Logarithmic, because a '
                    'set that holds a laptop part and a data-centre one '
                    'spans two orders of magnitude. Absolute rates rank the '
                    'machines, not the code -- see portability.')
    vals = [v for vs in table.values() for v in vs]
    c.fit([0, len(names)], vals)
    c.axes(xticks=[])
    step = (c.W - c.L - c.R) / max(len(names), 1)
    bw = step / (len(archs) + 0.8)
    for i, w in enumerate(names):
        for j, a in enumerate(archs):
            v = _geomean(table.get((w, a), []))
            if not v:
                continue
            c.bar(c.L + step * i + bw * (j + 0.5), bw * 0.88, v, colors[a],
                  f'{w} on {fieldset.device_of(a)}: {v:.4g}',
                  y_base=c.y0)
        c.tilted_label(c.L + step * (i + 0.5), w)
    for a in archs:
        c.legend.append((fieldset.device_of(a)[:26], colors[a]))
    return c.render()


def view_portability(fieldset: Field) -> Optional[str]:
    """The same, with each machine's own ceiling divided out.

    What fraction of what *this* machine can do did the kernel get? That is
    the number that compares a laptop part with a data-centre one without
    rewarding the one that is simply larger, and it is the number a
    generator is answerable for.
    """
    archs = [a for a in fieldset.archs
             if a in fieldset.ceilings and fieldset.ceilings[a].bytes_per_s]
    if len(archs) < 2:
        return None
    table: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    for p in fieldset.largest_batch(fieldset.points):
        if p.arch not in fieldset.ceilings:
            continue
        achieved, inten = rate(p.row, 'gflops'), p.intensity
        if not achieved or not inten:
            continue
        roof = fieldset.ceilings[p.arch].bound(inten) / 1e9
        if roof > 0:
            table[(p.workload, p.arch)].append(achieved / roof)
    names = [w for w in fieldset.workloads
             if sum(1 for a in archs if (w, a) in table) > 1]
    if not names:
        return None
    colors = _colors(archs)
    c = Canvas('Fraction of its own roof, per machine', '',
               'achieved ÷ roof at this intensity',
               width=max(760, 60 * len(names) + 220), legend_width=190,
               tilted_labels=True,
               note='1.0 would be the machine saturated at this kernel\'s '
                    'intensity. Comparable across machines; the absolute '
                    'rates in `archs` are not.')
    vals = [v for vs in table.values() for v in vs]
    c.fit([0, len(names)], vals + [1.0], y_from_zero=True)
    c.axes(xticks=[], ytick_label=lambda v: f'{v:.0%}')
    c.rule(1.0, 'the roof', '#777')
    step = (c.W - c.L - c.R) / max(len(names), 1)
    bw = step / (len(archs) + 0.8)
    for i, w in enumerate(names):
        for j, a in enumerate(archs):
            v = _geomean(table.get((w, a), []))
            if not v:
                continue
            c.bar(c.L + step * i + bw * (j + 0.5), bw * 0.88, v, colors[a],
                  f'{w} on {fieldset.device_of(a)}: {v:.1%} of its roof')
        c.tilted_label(c.L + step * (i + 0.5), w)
    for a in archs:
        geo = _geomean([v for (w, aa), vs in table.items() if aa == a
                        for v in vs])
        c.legend.append((f'{fieldset.device_of(a)[:20]} '
                         f'({geo:.0%})' if geo else a, colors[a]))
    return c.render()


# ----------------------------------------------------------------------
# Assembly
# ----------------------------------------------------------------------

QUESTIONS = {
    'devices': 'Which machines are these, and what did they turn out to do?',
    'bandwidth-peak': 'How close is the measured ceiling to the one the '
                      'memory bus allows?',
    'intensity': 'Is each kernel memory- or compute-bound, on each machine, '
                 'and by how far from the corner?',
    'kernels': 'What does each kernel need and carry — the numbers the '
               'charts divide by.',
    'configs': 'Does a configuration help everywhere, or only on one kernel?',
    'order': 'How does the cost move as the discretisation order grows?',
    'archs': 'The same kernel on each machine, as it ran.',
    'portability': 'The same, with each machine\'s own ceiling divided out.',
}


def build(fieldset: Field, what: str, reference: Optional[str],
          per_arch_order: bool) -> Tuple[List[Tuple[str, str, str]],
                                         List[Page], List[str]]:
    charts: List[Tuple[str, str, str]] = []
    pages: List[Page] = []
    skipped: List[str] = []

    def add_chart(name: str, svg: Optional[str], why: str) -> None:
        if svg:
            charts.append((name, QUESTIONS.get(name, ''), svg))
            from plot import _LAST_CANVAS
            if _LAST_CANVAS is not None:
                pages.append(_LAST_CANVAS.to_page(name))
        else:
            skipped.append(f'{name}: {why}')

    def add_table(name: str, made: Optional[Tuple[Page, str]],
                  why: str) -> None:
        if made:
            page, markup = made
            charts.append((name, QUESTIONS.get(name, ''), markup))
            pages.append(page)
        else:
            skipped.append(f'{name}: {why}')

    add_table('devices', view_devices(fieldset), 'no runs')
    add_chart('bandwidth-peak', view_bandwidth_peak(fieldset),
              'no device reported its bus clock and width -- re-measure the '
              'ceilings with a roofline.py that records them')
    add_chart('intensity', view_intensity(fieldset),
              'no ceilings, so there is no ridge to place a kernel against')
    add_table('kernels', view_kernels(fieldset), 'no runs')
    add_chart('flopmix', view_flopmix(fieldset),
              'the runs carry no executed_flops -- the analysis pass could '
              'not count the bodies')
    add_chart('configs', view_configs(fieldset, what, reference),
              'one configuration only -- the comparison needs two')
    if per_arch_order:
        for arch in fieldset.archs:
            add_chart(f'order-{arch}', view_order(fieldset, what, arch),
                      'fewer than two orders for this machine')
    add_chart('order', view_order(fieldset, what),
              'fewer than two orders in the tree; the order is read from the '
              'run directory name')
    add_chart('archs', view_archs(fieldset, 'gflops'),
              'one machine only')
    add_chart('portability', view_portability(fieldset),
              'fewer than two machines with a measured ceiling')
    return charts, pages, skipped


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        epilog='Point it at the tree evaluate.py wrote, or at individual '
               'bench.json files.')
    ap.add_argument('paths', type=Path, nargs='+',
                    help='directories to walk for bench.json, or the files')
    ap.add_argument('--out', type=Path, default=Path('bench-compare'),
                    help='directory for the charts, the tables and the PDF')
    ap.add_argument('--metric', default='ns_per_element',
                    choices=sorted(RATE_LABEL),
                    help='what the configuration and order views compare')
    ap.add_argument('--reference', default=None,
                    help='the configuration the others are measured against '
                         '(default: baseline, else the first by name)')
    ap.add_argument('--per-arch-order', action='store_true',
                    help='also one order chart per machine, not only the '
                         'pooled one')
    ap.add_argument('--flops', default='executed', choices=sorted(FLOP_FIELD),
                    help='which of the three flop counts a FLOP/s and an '
                         'intensity are against (default: what the kernel '
                         'issues, which is what a profiler reports)')
    ap.add_argument('--pdf', action='store_true', default=True,
                    help='write report.pdf (on by default)')
    ap.add_argument('--no-pdf', dest='pdf', action='store_false')
    args = ap.parse_args()

    set_flop_basis(args.flops)
    fieldset = collect(args.paths)
    if not fieldset.points:
        print('no usable rows under ' + ', '.join(str(p) for p in args.paths),
              file=sys.stderr)
        return 2

    charts, pages, skipped = build(fieldset, args.metric, args.reference,
                                   args.per_arch_order)
    args.out.mkdir(parents=True, exist_ok=True)
    for name, _, markup in charts:
        suffix = 'svg' if markup.lstrip().startswith('<svg') else 'html'
        (args.out / f'{name}.{suffix}').write_text(markup)
    title = (f'{len(fieldset.workloads)} kernels, {len(fieldset.configs)} '
             f'configurations, {len(fieldset.archs)} machines'
             + (f', orders {"/".join(str(o) for o in fieldset.orders)}'
                if fieldset.orders else ''))
    (args.out / 'index.html').write_text(index_html(charts, title))
    if args.pdf and pages:
        write_pdf(pages, args.out / 'report.pdf', title)

    for name, _, _ in charts:
        print(f'written: {name}')
    if args.pdf and pages:
        print(f'written: {args.out / "report.pdf"} ({len(pages)} pages)')
    print(f'written: {args.out / "index.html"}')
    for note in skipped:
        print(f'skipped {note}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
