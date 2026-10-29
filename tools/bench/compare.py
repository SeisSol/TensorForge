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

from capture import Capture, capture_of                        # noqa: E402
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

    A row that carries its capture (`run.py` writes `order` beside the rest
    of `capture.Capture.fields`) is believed first; a run that predates that
    is placed by its directory -- but only by the path, never by the
    *content*: guessing an order from a kernel's shapes would be a number
    nobody recorded.  Returns `None` where neither says, and the order views
    then leave the run out rather than placing it at zero.
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
    #: The machine, as every view names it: the architecture the row
    #: gives, or where one architecture ran under several backends, the
    #: backend and the architecture (`Field.split_backends`).
    arch: str
    backend: str
    device: str
    datatype: str
    batch: int
    row: Dict
    order: Optional[int] = None
    run: str = ''
    #: The export the kernel was captured from, where the row or the run's
    #: directory says (`capture.capture_of`).
    capture: Optional[Capture] = None

    @property
    def kernel(self) -> str:
        """The kernel this point measured, which the workload name alone
        does not identify: the same name is a different operation at every
        order, precision, simulation count and equation."""
        return (f'{self.workload} @ {self.capture.tag}' if self.capture
                else self.workload)

    @property
    def intensity(self) -> Optional[float]:
        """Flops per compulsory byte, on the basis the charts are drawn
        against -- a property of the operation, which is why it is the
        same on every machine and the ridge is not."""
        return intensity_of(self.row)


@dataclass(frozen=True)
class Unit:
    """A further compute roof over the same memory roof: a matrix unit, or
    a precision emulated on one (`roofline.py --unit`).

    Drawn beside the vector roof and never instead of it.  Which one a
    kernel is answerable to depends on the instructions it was built from,
    and nothing in a row says that yet -- so the fraction of the roof stays
    taken against the FMA chain, and these are the lines a kernel built for
    the unit could reach.
    """
    name: str
    flops_per_s: float
    source: str = ''


@dataclass
class Ceiling:
    """What one machine turned out to do (`roofline.py`), and what it is
    built to do where the device says so.

    One per machine *and precision*: the FMA chain is built for one type,
    and a double-precision kernel placed under a single-precision roof sits
    as far below it as the part's two rates differ -- twice on a data-centre
    part, sixty-four times on a consumer one.  `datatype` is empty only for
    a ceiling that did not say, which then stands for every precision of
    its machine: what somebody supplied by hand, as asked.
    """
    arch: str
    backend: str = ''
    device: str = ''
    compute_units: int = 0
    flops_per_s: float = 0.0
    bytes_per_s: float = 0.0
    memory_bytes: int = 0
    peak_bytes_per_s: float = 0.0
    peak_flops_per_s: float = 0.0
    datatype: str = ''
    units: Tuple[Unit, ...] = ()

    @property
    def ridge(self) -> float:
        return self.flops_per_s / self.bytes_per_s if self.bytes_per_s else 0.0

    def bound(self, intensity: float) -> float:
        return min(self.flops_per_s, self.bytes_per_s * intensity)


@dataclass
class Field:
    """Every point of a campaign, and the ceilings that go with them,
    keyed by machine and precision (`ceiling`)."""
    points: List[Point] = field(default_factory=list)
    ceilings: Dict[Tuple[str, str], Ceiling] = field(default_factory=dict)
    #: What was asked for and produced no number -- not built, not run,
    #: or timed out -- with the row's `error` saying which.  Kept apart
    #: from `points`, which every view draws, for a reader that has to
    #: tell a kernel that failed on a machine from one that was not run.
    failed: List[Point] = field(default_factory=list)
    #: The machines that are one architecture under one of several
    #: backends, by name: their architecture and backend.
    split: Dict[str, Tuple[str, str]] = field(default_factory=dict)

    def split_backends(self) -> None:
        """An architecture measured under more than one backend -- SPMD and
        ESIMD on an Intel GPU -- is a machine per backend, `esimd-pvc`
        beside `oneapi-pvc`, each under the architecture's ceilings: every
        view keys a measurement by its machine, and one machine of both
        would hold one backend's kernel and drop the other's."""
        backends: Dict[str, set] = defaultdict(set)
        for p in self.points + self.failed:
            backends[p.arch].add(p.backend)
        for arch, found in sorted(backends.items()):
            if len(found) < 2:
                continue
            for backend in sorted(found):
                name = f'{backend}-{arch}'
                self.split[name] = (arch, backend)
                for (a, datatype), ceiling in list(self.ceilings.items()):
                    if a == arch:
                        self.ceilings.setdefault((name, datatype), ceiling)
            for p in self.points + self.failed:
                if p.arch == arch:
                    p.arch = f'{p.backend}-{arch}'

    def machine_name(self, arch: str, backend: str) -> str:
        """What the views call the machine of a row of `arch` and
        `backend` (`split_backends`)."""
        name = f'{backend}-{arch}'
        return name if name in self.split else arch

    def architecture(self, machine: str) -> str:
        """The architecture a machine is, as a row or a profile names it
        (`split_backends`)."""
        return self.split.get(machine, (machine, ''))[0]

    def ceiling(self, arch: str, datatype: str) -> Optional[Ceiling]:
        """The roof a point of this precision on this machine is placed
        under: the one measured for it, else one that named no precision.
        Never the other precision's -- that is the error this is keyed
        against."""
        return (self.ceilings.get((arch, datatype))
                or self.ceilings.get((arch, '')))

    def machine(self, arch: str) -> Optional[Ceiling]:
        """Any ceiling of the machine, for what does not depend on the
        precision: its name, its memory, its bus.  The fastest memory roof
        measured on it, since a triad over doubles and one over floats
        measure the same bus."""
        have = [c for (a, _), c in self.ceilings.items() if a == arch]
        return max(have, key=lambda c: c.bytes_per_s) if have else None

    @property
    def ceiling_archs(self) -> List[str]:
        return sorted({a for (a, _) in self.ceilings
                       if a in set(self.archs)})

    @property
    def archs(self) -> List[str]:
        return sorted({p.arch for p in self.points})

    @property
    def workloads(self) -> List[str]:
        return sorted({p.workload for p in self.points})

    @property
    def kernels(self) -> List[str]:
        """Every kernel measured (`Point.kernel`), by name and then by the
        export -- equation, simulations, precision, order -- so the orders
        of one kernel stand side by side."""
        first: Dict[str, Point] = {}
        for p in self.points:
            first.setdefault(p.kernel, p)
        return sorted(first, key=lambda k: _kernel_order(first[k]))

    @property
    def captures(self) -> List[str]:
        return sorted({p.capture.tag for p in self.points if p.capture})

    @property
    def configs(self) -> List[str]:
        return sorted({p.config for p in self.points})

    @property
    def orders(self) -> List[int]:
        return sorted({p.order for p in self.points if p.order is not None})

    @property
    def reference(self) -> Optional[str]:
        """The configuration a single-configuration view is drawn for:
        `baseline` where it was run, else the first by name."""
        configs = self.configs
        if not configs:
            return None
        return 'baseline' if 'baseline' in configs else configs[0]

    def label(self, point: Point) -> str:
        """A kernel's name in a chart.  The export is spelled out only where
        the field holds more than one, for the reason `plot.distinguish`
        gives: a legend repeating the same suffix on every entry is noise."""
        return point.kernel if len(self.captures) > 1 else point.workload

    def device_of(self, arch: str) -> str:
        """The device a machine is, by name; with its backend where the
        device is several machines (`split_backends`)."""
        device = next((p.device for p in self.points
                       if p.arch == arch and p.device), '')
        if not device:
            found = self.machine(arch)
            device = (found.device if found else '') or arch
        if arch in self.split and device != arch:
            device += f' ({self.split[arch][1]})'
        return device

    def at(self, **keys) -> List[Point]:
        return [p for p in self.points
                if all(getattr(p, k) == v for k, v in keys.items())]

    def largest_batch(self, points: Sequence[Point]) -> List[Point]:
        """One point per kernel, configuration and machine: the biggest
        launch there is, because a ratio taken on the rising part of a
        scaling curve is a ratio at an arbitrary place on a slope.

        Keyed by the kernel and not by its name, so the orders, precisions
        and equations of one name stay apart rather than the largest of
        them standing for all."""
        best: Dict[Tuple, Point] = {}
        for p in points:
            key = (p.workload, p.capture, p.config, p.arch, p.order,
                   p.datatype)
            if key not in best or p.batch > best[key].batch:
                best[key] = p
        return list(best.values())

    def one_per_kernel(self, points: Sequence[Point]) -> Dict[Tuple[str, str],
                                                              Point]:
        """`{(kernel, arch): point}` at the largest launch, in the
        reference configuration where the kernel was run in it.

        One configuration and not a pool of them: what a kernel issues
        depends on how it was built, and the views that use this draw one
        mark per kernel and machine.
        """
        ref = self.reference
        out: Dict[Tuple[str, str], Point] = {}
        for p in sorted(self.largest_batch(points),
                        key=lambda p: (p.config != ref, p.config)):
            out.setdefault((p.kernel, p.arch), p)
        return out


def _kernel_order(p: Point) -> Tuple:
    c = p.capture
    return (p.workload,
            (c.equation, c.simulations, c.precision, c.order) if c
            else ('', 0, '', 0))


def _load_ceiling(path: Path) -> Optional[Ceiling]:
    try:
        blob = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    c = blob.get('ceiling', blob)
    if not isinstance(c, dict) or not c.get('arch'):
        return None
    units = tuple(
        Unit(name=str(u.get('unit') or 'unit')
             + (f" {u['shape']}" if u.get('shape') else ''),
             flops_per_s=float(u['flops_per_s']),
             source=str(u.get('source') or ''))
        for u in c.get('units') or [] if u.get('flops_per_s'))
    return Ceiling(
        arch=c['arch'], backend=c.get('backend', ''),
        device=c.get('device', ''),
        compute_units=int(c.get('compute_units') or c.get('cus') or 0),
        flops_per_s=float(c.get('flops_per_s') or 0.0),
        bytes_per_s=float(c.get('bytes_per_s') or 0.0),
        memory_bytes=int(c.get('memory_bytes') or 0),
        peak_bytes_per_s=float(c.get('peak_bytes_per_s') or 0.0),
        peak_flops_per_s=float(c.get('peak_flops_per_s') or 0.0),
        datatype=str(c.get('datatype') or ''), units=units)


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
                rows = _results(json.loads(bench.read_text()))
            except (OSError, ValueError):
                continue
            # Per row, since one file may hold several exports; the
            # directory only for rows that do not say.
            by_path = capture_of(bench.parent, [])
            run = bench.parent.name
            for row in rows:
                if not row.get('workload'):
                    continue
                order = order_of(bench.parent, [row])
                captured = Capture.from_row(row) or by_path
                if captured is not None and (
                        captured.order != order
                        or (row.get('datatype')
                            and captured.datatype != row['datatype'])):
                    # The row states an order or a precision its directory
                    # contradicts; the row is the record, and a capture that
                    # disagrees with it identifies nothing.
                    captured = None
                (out.failed if row.get('error') else out.points).append(Point(
                    workload=row['workload'],
                    config=row.get('config') or 'default',
                    arch=row.get('arch') or '?',
                    backend=row.get('backend') or '?',
                    device=row.get('device') or '',
                    datatype=row.get('datatype') or '',
                    batch=int(row.get('batch') or 0),
                    row=row, order=order, run=run, capture=captured))
            for name in ('ceiling.json', 'roofline.json'):
                ceiling = _load_ceiling(bench.parent / name)
                if ceiling:
                    out.ceilings.setdefault(
                        (ceiling.arch, ceiling.datatype), ceiling)
        # A tree `evaluate.py` wrote keeps its ceilings in one place rather
        # than beside every run, which is the point of measuring them once.
        if target.is_dir():
            for name in sorted(target.rglob('*.json')):
                if name.parent.name == 'ceiling' or name.name in (
                        'ceiling.json', 'roofline.json'):
                    ceiling = _load_ceiling(name)
                    if ceiling:
                        out.ceilings.setdefault(
                            (ceiling.arch, ceiling.datatype), ceiling)
    out.split_backends()
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
    datatypes = sorted({p.datatype for p in fieldset.points if p.datatype}
                       | {dt for (_, dt) in fieldset.ceilings if dt})
    headers = (['device', 'arch', 'backend', 'CUs', 'memory']
               + [f'{dt} TFLOP/s' for dt in datatypes]
               + ['GB/s measured', 'GB/s peak']
               + [f'ridge {dt}' for dt in datatypes])
    rows = []
    for arch in archs:
        c = fieldset.machine(arch)
        per = [fieldset.ceiling(arch, dt) for dt in datatypes]
        rows.append([
            fieldset.device_of(arch), arch,
            next((p.backend for p in fieldset.points if p.arch == arch), '?'),
            str(c.compute_units or '?') if c else '?',
            f'{c.memory_bytes / 2 ** 30:.0f} GiB' if c and c.memory_bytes
            else 'unknown']
            + [f'{x.flops_per_s / 1e12:.2f}' if x and x.flops_per_s else '—'
               for x in per]
            + [f'{c.bytes_per_s / 1e9:.0f}' if c and c.bytes_per_s else '—',
               f'{c.peak_bytes_per_s / 1e9:.0f}' if c and c.peak_bytes_per_s
               else 'unknown']
            + [f'{x.ridge:.1f} F/B' if x and x.ridge else '—' for x in per])
    note = ('"measured" is what `roofline.py` got out of the machine, an FMA '
            'chain per precision and a triad; "peak" is what the memory bus '
            'allows, where the device reports its clock and width. A '
            'theoretical FLOP/s is not listed: it needs a cores-per-unit '
            'table per architecture, which would be a guess for any part '
            'newer than the table. A precision with no ceiling of its own '
            'is a dash rather than the other precision\'s figure.')
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
    have = [(a, fieldset.machine(a)) for a in fieldset.archs
            if fieldset.machine(a) is not None
            and fieldset.machine(a).bytes_per_s
            and fieldset.machine(a).peak_bytes_per_s]
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

def view_intensity(fieldset: Field) -> Optional[str]:
    """Where each kernel sits against each machine's ridge.

    The quantity is the kernel's intensity divided by the machine's ridge,
    on a log axis with a rule at one. Below the rule the kernel is
    memory-bound on that machine, above it compute-bound, and the distance
    is how far from the corner it is -- which is the part a yes/no label
    throws away. A kernel can be memory-bound on one machine and
    compute-bound on the next, which is exactly why the rule cannot be a
    property of the kernel.

    One mark per kernel *and* machine, each from that machine's own run:
    what a kernel issues depends on what it was built for -- a wave of 64
    pads differently from a warp of 32 -- so on the default basis the
    intensity is not one number per kernel either.
    """
    per = fieldset.one_per_kernel(fieldset.points)

    def ridge_of(k: str, a: str) -> float:
        p = per.get((k, a))
        roof = fieldset.ceiling(a, p.datatype) if p is not None else None
        return roof.ridge if roof is not None and p.intensity else 0.0

    archs = fieldset.ceiling_archs
    names = [k for k in fieldset.kernels
             if any(ridge_of(k, a) for a in archs)]
    if not archs or not names:
        return None
    labels = {k: fieldset.label(next(per[(k, a)] for a in archs
                                     if (k, a) in per)) for k in names}
    colors, marks = _colors(archs), _marks(archs)
    c = Canvas('Memory- or compute-bound, per machine',
               '', 'arithmetic intensity ÷ ridge', ylog=True,
               width=max(760, 46 * len(names) + 220), legend_width=190,
               tilted_labels=max(len(v) for v in labels.values()),
               note='Below 1 the machine runs out of bandwidth first; above '
                    'it, of arithmetic. The ridge is the measured one, of '
                    'the kernel\'s own precision; a kernel with no ceiling '
                    'of its precision on a machine has no mark there.')
    ratios = [per[(k, a)].intensity / ridge_of(k, a)
              for k in names for a in archs if ridge_of(k, a)]
    c.fit([0, len(names)], ratios + [1.0], ypad=1.6)
    c.axes(xticks=[])
    c.rule(1.0, 'ridge — bandwidth below, arithmetic above', '#777')
    step = (c.W - c.L - c.R) / max(len(names), 1)
    for i, k in enumerate(names):
        for a in archs:
            ridge = ridge_of(k, a)
            if not ridge:
                continue
            p = per[(k, a)]
            r = p.intensity / ridge
            x = c.L + step * (i + 0.5)
            c._mark(x, c.py(r), colors[a], marks[a],
                    f'{k} on {a}: {p.intensity:.2f} flop/byte against a '
                    f'ridge of {ridge:.1f} — '
                    + ('memory-bound' if r < 1 else 'compute-bound')
                    + f', by {max(r, 1 / r):.1f}x')
        c.tilted_label(c.L + step * (i + 0.5), labels[k])
    for a in archs:
        ridges = '/'.join(f'{fieldset.ceilings[(aa, dt)].ridge:.0f}'
                          for (aa, dt) in sorted(fieldset.ceilings)
                          if aa == a and fieldset.ceilings[(aa, dt)].ridge)
        c.legend.append((f'{a} (ridge {ridges})', colors[a]))
    return c.render()


def view_flopmix(fieldset: Field, arch: Optional[str] = None
                 ) -> Optional[str]:
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

    One machine per chart: the lanes are the target's, so the same kernel
    issues differently on a warp of 32 and a wave of 64.  Without `arch`
    the field must hold one machine.
    """
    archs = [arch] if arch else fieldset.archs
    if len(archs) != 1:
        return None
    by_kernel: Dict[str, Point] = {}
    for (kernel, a), p in fieldset.one_per_kernel(fieldset.points).items():
        if (a == archs[0] and p.row.get('executed_flops')
                and p.row.get('nonzero_flops')):
            by_kernel[kernel] = p
    if not by_kernel:
        return None
    names = [k for k in fieldset.kernels if k in by_kernel]
    labels = {k: fieldset.label(by_kernel[k]) for k in names}
    where = f' — {archs[0]}' if arch else ''
    c = Canvas(f'How much of the issued arithmetic is needed{where}', '',
               'fraction of what the kernel issues',
               width=max(760, 54 * len(names) + 200), legend_width=170,
               tilted_labels=max(len(v) for v in labels.values()),
               note='The bar is what the operation needs; the first segment '
                    'above it is what a dense implementation adds, and the '
                    'rest is the lanes a narrow destination leaves idle. '
                    'Per element at the largest launch. The issued count is '
                    'an upper bound (up to 1.53x the hardware figure), so '
                    'the needed fraction is a floor.')
    c.fit([0, len(names)], [0.0, 1.0], y_from_zero=True)
    c.axes(xticks=[], ytick_label=lambda v: f'{v:.0%}')
    c.rule(1.0, 'issued', '#777')
    step = (c.W - c.L - c.R) / max(len(names), 1)
    for i, k in enumerate(names):
        p = by_kernel[k]
        issued = float(p.row['executed_flops'])
        needed = float(p.row['nonzero_flops']) / issued
        dense = min(float(p.row.get('flops') or 0) / issued, 1.0)
        x = c.L + step * (i + 0.22)
        c.bar(x, step * 0.56, needed, PALETTE[0],
              f'{k}: needs {needed:.1%} of what it issues')
        if dense > needed:
            c.bar(x, step * 0.56, dense, PALETTE[1],
                  f'{k}: a dense implementation would perform {dense:.1%}',
                  y_base=needed)
        if dense < 1.0:
            c.bar(x, step * 0.56, 1.0, PALETTE[4],
                  f'{k}: the lanes add the rest, to {issued:,.0f} flop',
                  y_base=dense)
        c.tilted_label(c.L + step * (i + 0.5), labels[k])
    c.legend.extend([('needed', PALETTE[0]), ('dense adds', PALETTE[1]),
                     ('idle lanes add', PALETTE[4])])
    return c.render()


def view_kernels(fieldset: Field) -> Optional[Tuple[Page, str]]:
    """What each kernel is, as a table: the numbers the charts divide by.

    The three flop counts are kept apart rather than reconciled -- what the
    operation needs, what a dense implementation performs, and what the
    built kernel performs are three different things and the generator
    reports the third.

    One row per kernel and machine, since what is issued is the target's;
    the first two counts and the bytes repeat across the machines of one
    kernel, which is the check that the rows are of the same operation.
    """
    per = fieldset.one_per_kernel(fieldset.points)
    if not per:
        return None
    headers = ['kernel', 'machine', 'needed', 'dense', 'issued', 'useful',
               'byte/element', f'flop/byte ({flop_basis()})', 'bound']
    rows = []
    order = {k: i for i, k in enumerate(fieldset.kernels)}
    for (k, a) in sorted(per, key=lambda ka: (order[ka[0]], ka[1])):
        p = per[(k, a)]
        batch = p.batch or 1
        b = p.row.get('bytes')
        needed = p.row.get('nonzero_flops')
        dense = p.row.get('flops')
        issued = p.row.get('executed_flops')
        inten = p.intensity
        cells = [fieldset.label(p), a,
                 f'{needed / batch:,.0f}' if needed else '—',
                 f'{dense / batch:,.0f}' if dense else '—',
                 f'{issued / batch:,.0f}' if issued else '—',
                 f'{needed / issued:.0%}' if needed and issued else '—',
                 f'{b / batch:,.0f}' if b else '—',
                 f'{inten:.2f}' if inten else '—']
        ceiling = fieldset.ceiling(a, p.datatype)
        ridge = ceiling.ridge if ceiling else 0.0
        if not inten or not ridge:
            cells.append('—')
        else:
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

    A ratio is taken between two builds of the *same* kernel on the *same*
    machine and only then pooled over machines, as a geometric mean: pooling
    each configuration's rates first would divide one set of machines by
    another wherever the two configurations were not run on the same ones.
    """
    configs = fieldset.configs
    if len(configs) < 2:
        return None
    ref = reference or fieldset.reference
    if ref not in configs:
        return None
    others = [c for c in configs if c != ref]
    table: Dict[Tuple[str, str, str], float] = {}
    first: Dict[str, Point] = {}
    for p in fieldset.largest_batch(fieldset.points):
        v = rate(p.row, what)
        if v:
            table[(p.kernel, p.arch, p.config)] = v
            first.setdefault(p.kernel, p)
    lower_better = what in LOWER_IS_BETTER
    pooled: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    for (k, a, cfg), cand in table.items():
        base = table.get((k, a, ref))
        if cfg == ref or not base:
            continue
        pooled[(k, cfg)].append(base / cand if lower_better else cand / base)
    ratios = {key: _geomean(vs) for key, vs in pooled.items()}
    names = [k for k in fieldset.kernels
             if any((k, o) in ratios for o in others)]
    if not names:
        return None
    labels = {k: fieldset.label(first[k]) for k in names}
    colors, marks = _colors(others), _marks(others)
    c = Canvas(f'Configurations against {ref} — {rate_label(what)}',
               '', f'ratio to {ref}', ylog=True,
               width=max(760, 46 * len(names) + 220), legend_width=190,
               tilted_labels=max(len(v) for v in labels.values()),
               note='Above 1 is faster. Each mark is one configuration on '
                    'one kernel, at the largest launch, against the '
                    'reference on the same machine and pooled over every '
                    'machine that ran both.')
    c.fit([0, len(names)], list(ratios.values()) + [1.0], ypad=1.4)
    c.axes(xticks=[])
    c.rule(1.0, ref, '#777')
    step = (c.W - c.L - c.R) / max(len(names), 1)
    for i, k in enumerate(names):
        for j, o in enumerate(others):
            r = ratios.get((k, o))
            if not r:
                continue
            x = c.L + step * (i + 0.3 + 0.4 * j / max(len(others) - 1, 1))
            c._mark(x, c.py(r), colors[o], marks[o],
                    f'{k}: {o} is {r:.3f}x {ref}')
        c.tilted_label(c.L + step * (i + 0.5), labels[k])
    for o in others:
        geo = _geomean([v for (k, oo), v in ratios.items() if oo == o])
        c.legend.append((f'{o} (geo {geo:.3f}x)' if geo else o, colors[o]))
    return c.render()


def _series_name(p: Point, varying: Sequence[str]) -> str:
    """A curve's name in the order view: the kernel's, and of its export
    whichever parts differ between the curves of the chart."""
    parts = []
    c = p.capture
    if c is not None:
        if 'equation' in varying:
            parts.append(c.equation)
        if 'simulations' in varying:
            parts.append(f's{c.simulations}')
    if 'datatype' in varying:
        parts.append(p.datatype.lower())
    if 'config' in varying:
        parts.append(p.config)
    return p.workload + (f' [{" ".join(parts)}]' if parts else '')


def view_order(fieldset: Field, what: str = 'ns_per_element',
               arch: Optional[str] = None) -> Optional[str]:
    """Cost against the discretisation order, one line per kernel.

    The axis is the order and not the problem size, because that is the
    knob a solve is configured with. The cost per element grows steeply
    with it -- the operators are larger and there are more of them -- so
    the y axis is logarithmic and a kernel that grows *less* steeply than
    its neighbours is the thing to see.

    A curve is one kernel of one equation, simulation count, precision and
    configuration; only the order varies along it.  Joining the orders of
    single and double, or of one and eight simulations, would draw a line
    through points that are not the same kernel at another order.
    """
    orders = fieldset.orders
    if len(orders) < 2:
        return None
    points = [p for p in fieldset.largest_batch(fieldset.points)
              if p.order is not None and (arch is None or p.arch == arch)]

    def key(p: Point) -> Tuple:
        c = p.capture
        return (p.workload, c.equation if c else '',
                c.simulations if c else 1, p.datatype, p.config)

    varying = [name for name, i in (('equation', 1), ('simulations', 2),
                                    ('datatype', 3), ('config', 4))
               if len({key(p)[i] for p in points}) > 1]
    series: Dict[Tuple, Dict[int, List[float]]] = defaultdict(
        lambda: defaultdict(list))
    named: Dict[Tuple, str] = {}
    for p in points:
        v = rate(p.row, what)
        if v:
            series[key(p)][p.order].append(v)
            named.setdefault(key(p), _series_name(p, varying))
    series = {k: v for k, v in series.items() if len(v) > 1}
    if not series:
        return None
    keys = sorted(series, key=lambda k: named[k])
    names = [named[k] for k in keys]
    colors, marks = _colors(names), _marks(names)
    # The points first, the axis second: pooling several machines into one
    # curve and then fitting the axis to the *unpooled* values leaves the
    # scale a decade below anything drawn, and half the chart empty.
    curves = {named[k]: [(o, _geomean(series[k].get(o, [])) or None)
                         for o in orders] for k in keys}
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

    In the reference configuration: two configurations of one kernel are
    two builds, and pooling them would rank a machine by whichever it
    happened to run more of.
    """
    archs = fieldset.archs
    if len(archs) < 2:
        return None
    per = fieldset.one_per_kernel(fieldset.points)
    table = {ka: rate(p.row, what) for ka, p in per.items()}
    table = {ka: v for ka, v in table.items() if v}
    names = [k for k in fieldset.kernels
             if sum(1 for a in archs if (k, a) in table) > 1]
    if not names:
        return None
    labels = {k: fieldset.label(next(per[(k, a)] for a in archs
                                     if (k, a) in per)) for k in names}
    colors = _colors(archs)
    c = Canvas(f'Each machine on each kernel — {rate_label(what)}', '',
               rate_label(what), ylog=True,
               width=max(760, 60 * len(names) + 220), legend_width=190,
               tilted_labels=max(len(v) for v in labels.values()),
               note='At the largest launch each ran. Logarithmic, because a '
                    'set that holds a laptop part and a data-centre one '
                    'spans two orders of magnitude. Absolute rates rank the '
                    'machines, not the code -- see portability.')
    c.fit([0, len(names)], list(table.values()))
    c.axes(xticks=[])
    step = (c.W - c.L - c.R) / max(len(names), 1)
    bw = step / (len(archs) + 0.8)
    for i, k in enumerate(names):
        for j, a in enumerate(archs):
            v = table.get((k, a))
            if not v:
                continue
            c.bar(c.L + step * i + bw * (j + 0.5), bw * 0.88, v, colors[a],
                  f'{k} on {fieldset.device_of(a)}: {v:.4g}',
                  y_base=c.y0)
        c.tilted_label(c.L + step * (i + 0.5), labels[k])
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
    archs = [a for a in fieldset.ceiling_archs
             if fieldset.machine(a).bytes_per_s]
    if len(archs) < 2:
        return None
    per = fieldset.one_per_kernel(fieldset.points)
    table: Dict[Tuple[str, str], float] = {}
    for (k, a), p in per.items():
        ceiling = fieldset.ceiling(a, p.datatype)
        if ceiling is None or not ceiling.bytes_per_s:
            continue
        achieved, inten = rate(p.row, 'gflops'), p.intensity
        if not achieved or not inten:
            continue
        roof = ceiling.bound(inten) / 1e9
        if roof > 0:
            table[(k, a)] = achieved / roof
    names = [k for k in fieldset.kernels
             if sum(1 for a in archs if (k, a) in table) > 1]
    if not names:
        return None
    labels = {k: fieldset.label(next(per[(k, a)] for a in archs
                                     if (k, a) in per)) for k in names}
    colors = _colors(archs)
    c = Canvas('Fraction of its own roof, per machine', '',
               'achieved ÷ roof at this intensity',
               width=max(760, 60 * len(names) + 220), legend_width=190,
               tilted_labels=max(len(v) for v in labels.values()),
               note='1.0 would be the machine saturated at this kernel\'s '
                    'intensity. Comparable across machines; the absolute '
                    'rates in `archs` are not.')
    c.fit([0, len(names)], list(table.values()) + [1.0], y_from_zero=True)
    c.axes(xticks=[], ytick_label=lambda v: f'{v:.0%}')
    c.rule(1.0, 'the roof', '#777')
    step = (c.W - c.L - c.R) / max(len(names), 1)
    bw = step / (len(archs) + 0.8)
    for i, k in enumerate(names):
        for j, a in enumerate(archs):
            v = table.get((k, a))
            if not v:
                continue
            c.bar(c.L + step * i + bw * (j + 0.5), bw * 0.88, v, colors[a],
                  f'{k} on {fieldset.device_of(a)}: {v:.1%} of its roof')
        c.tilted_label(c.L + step * (i + 0.5), labels[k])
    for a in archs:
        geo = _geomean([v for (k, aa), v in table.items() if aa == a])
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
    # Per machine where there are several: the lanes a narrow destination
    # leaves idle are the target's.
    for arch in (fieldset.archs if len(fieldset.archs) > 1 else [None]):
        add_chart(f'flopmix-{arch}' if arch else 'flopmix',
                  view_flopmix(fieldset, arch),
                  'the runs carry no executed_flops -- the analysis pass '
                  'could not count the bodies')
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
