# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Charts from what `run.py` and `profile.py` already wrote.

`roofline.py` draws one view and draws it well: where a kernel sits under the
machine's two ceilings. That view answers one question -- *is this kernel
allowed to go faster* -- and it answers it at one launch size. The questions
that come up next are about the shape of the curve rather than a point on it:
where a kernel saturates, what it costs to get there, whether the traffic it
pays for is the traffic it needs, and which of two builds is actually ahead
once the batch is large enough to mean anything.

So this is a family of views over the same two files, and nothing else. It
runs no kernels and builds nothing. Point it at a `bench.json` (and at a
`profile.json` where one exists) and it writes SVG.

## What each view is for

| view | reads | the question |
|---|---|---|
| `scaling` | bench | where does the rate stop rising with the batch? |
| `latency` | bench | what does one element cost, and from which batch on? |
| `spread` | bench | is the measurement stable enough to compare? |
| `roofline` | bench | is the kernel allowed to go faster? |
| `trajectory` | bench | how does it climb toward its roof as the launch grows? |
| `efficiency` | bench | what fraction of the roof, against the batch? |
| `compare` | 2+ bench | which build is ahead, and by how much, per kernel? |
| `occupancy` | + profile | does the rate follow the occupancy, or not? |
| `traffic` | + profile | how much of the traffic is compulsory? |
| `measured-roofline` | + profile | the same point at the intensity it really ran at |

The last three need a profiler, which on a shared machine is often not
available (see `README.md`); they are skipped with a note rather than being a
reason for the rest to fail.

## Why hand-written SVG again

The same reason `roofline.py` gives: this directory is meant to run on a login
node of a machine somebody else administers, where `pip install matplotlib` is
not a thing that reliably happens, and where the result has to travel back as
a file. SVG opens in a browser, scales, carries `<title>` tooltips with the
exact figures, and diffs as text. The cost is that every axis is drawn here;
`Canvas` keeps that in one place.

## What these charts deliberately do not do

They do not smooth, fit or extrapolate. A line joins measured points and stops
where the measurements stop. Where a quantity is missing -- a batch that was
not run, a metric the profiler dropped -- the line breaks rather than
interpolating across the gap, because an interpolated roofline point is a
statement about a kernel that was never launched.

Nor do they average over configurations. Two configurations of one kernel are
two curves; collapsing them to a mean is how a configuration that wins
everywhere and one that wins nowhere come to look alike.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

# ----------------------------------------------------------------------
# Reading what the other tools wrote
# ----------------------------------------------------------------------


def _results(blob) -> List[Dict]:
    """`run.py` has written both shapes: a bare list and `{'results': [...]}`."""
    if isinstance(blob, dict) and 'results' in blob:
        return list(blob['results'])
    return list(blob) if isinstance(blob, list) else []


@dataclass
class Run:
    """One `bench.json`, with the label it is drawn under."""
    label: str
    path: Path
    rows: List[Dict]

    @property
    def device(self) -> str:
        for row in self.rows:
            if row.get('arch'):
                return f"{row['arch']} / {row.get('backend', '?')}"
        return '?'

    @property
    def datatypes(self) -> List[str]:
        return sorted({r.get('datatype', '') for r in self.rows if r.get('datatype')})


def load_run(path: Path, label: Optional[str] = None) -> Run:
    path = path / 'bench.json' if path.is_dir() else path
    rows = [r for r in _results(json.loads(path.read_text())) if not r.get('error')]
    return Run(label or path.parent.name, path, rows)


def load_profile(path: Path) -> Dict[str, List[Dict]]:
    """`profile.json` to `{workload: [row, ...]}`.

    One workload can carry several rows where the profiler saw several
    kernels in the unit; they are kept, and the views that need a single
    number take the longest-running one, which is the one the workload is
    named after.
    """
    path = path / 'profile.json' if path.is_dir() else path
    blob = json.loads(path.read_text())
    out: Dict[str, List[Dict]] = {}
    for collection in blob.get('collections', []):
        rows = [r for r in collection.get('rows', []) if r]
        if rows:
            out.setdefault(collection['workload'], []).extend(rows)
    return out


def hottest(rows: Sequence[Dict]) -> Dict:
    return max(rows, key=lambda r: r.get('duration_ns') or 0.0)


# ----------------------------------------------------------------------
# Drawing
# ----------------------------------------------------------------------

#: Eight lines are as many as a legend can carry before it stops being read.
#: Chosen to stay apart in the common forms of colour blindness and, failing
#: that, in luminance -- the charts are also read printed.
PALETTE = ('#2f6fb2', '#b2452f', '#3f8f4f', '#8a5fb0',
           '#c08a1e', '#2a8f9a', '#9a2a60', '#606060')

MARKS = ('circle', 'square', 'triangle', 'diamond')


def _fmt(value: float) -> str:
    """A number a reader can say out loud."""
    a = abs(value)
    if a == 0:
        return '0'
    if a >= 1e12:
        return f'{value / 1e12:.3g}T'
    if a >= 1e9:
        return f'{value / 1e9:.3g}G'
    if a >= 1e6:
        return f'{value / 1e6:.3g}M'
    if a >= 1e3:
        return f'{value / 1e3:.3g}k'
    if a >= 1:
        return f'{value:.3g}'
    return f'{value:.2g}'


class Canvas:
    """Axes, ticks, a legend and a frame -- the part every view repeats.

    Both axes are independently linear or logarithmic. A log axis with a
    non-positive bound is a bug in the caller, not something to clamp
    silently, so it raises.
    """

    def __init__(self, title: str, xlabel: str, ylabel: str,
                 xlog: bool = False, ylog: bool = False,
                 width: int = 760, height: int = 460, note: str = '',
                 legend_width: int = 0, tilted_labels: bool = False):
        """`legend_width` reserves a column to the right for the key and
        `tilted_labels` a band below for names written at an angle.  Both are
        margins rather than overlays: a legend drawn on top of the data hides
        exactly the crowded part a reader came to look at."""
        self.title, self.note = title, note
        self.xlabel, self.ylabel = xlabel, ylabel
        self.xlog, self.ylog = xlog, ylog
        self.W = width + legend_width
        self.H = height + (58 if tilted_labels else 0)
        self.L, self.T = 74, 46
        self.R = 22 + legend_width
        self.B = 62 + (58 if tilted_labels else 0)
        self.legend_width = legend_width
        self.tilted = tilted_labels
        self.body: List[str] = []
        self.legend: List[Tuple[str, str]] = []
        self.x0 = self.x1 = self.y0 = self.y1 = 0.0

    def tilted_label(self, x: float, text: str) -> None:
        """A name under a bar, written at an angle so that neighbours do not
        collide.  Anchored at its end so the text runs away from the axis
        rather than through it."""
        y = self.H - self.B + 13
        self.body.append(
            f'<text x="{x:.1f}" y="{y}" text-anchor="end" fill="#555" '
            f'font-size="9" transform="rotate(-45 {x:.1f} {y})">'
            f'{html.escape(text[:30])}</text>')

    # -- scales ---------------------------------------------------------

    def fit(self, xs: Iterable[float], ys: Iterable[float],
            ypad: float = 1.1, y_from_zero: bool = False) -> None:
        xs = [x for x in xs if x is not None]
        ys = [y for y in ys if y is not None]
        if not xs or not ys:
            xs, ys = [0.0, 1.0], [0.0, 1.0]
        self.x0, self.x1 = min(xs), max(xs)
        self.y0, self.y1 = min(ys), max(ys)
        if self.xlog:
            if self.x0 <= 0:
                raise ValueError('a logarithmic x axis needs positive data')
            self.x0, self.x1 = self.x0 / 1.3, self.x1 * 1.3
        else:
            span = (self.x1 - self.x0) or abs(self.x1) or 1.0
            self.x0 -= span * 0.04
            self.x1 += span * 0.04
        if self.ylog:
            if self.y0 <= 0:
                raise ValueError('a logarithmic y axis needs positive data')
            self.y0, self.y1 = self.y0 / ypad, self.y1 * ypad
        else:
            self.y0 = 0.0 if y_from_zero else self.y0 - (self.y1 - self.y0) * 0.08
            self.y1 = self.y1 * ypad if self.y1 > 0 else 1.0
        if self.x1 <= self.x0:
            self.x1 = self.x0 + 1.0
        if self.y1 <= self.y0:
            self.y1 = self.y0 + 1.0

    def px(self, x: float) -> float:
        lo, hi = self.x0, self.x1
        if self.xlog:
            x, lo, hi = math.log10(max(x, lo)), math.log10(lo), math.log10(hi)
        return self.L + (x - lo) / (hi - lo) * (self.W - self.L - self.R)

    def py(self, y: float) -> float:
        lo, hi = self.y0, self.y1
        if self.ylog:
            y, lo, hi = math.log10(max(y, lo)), math.log10(lo), math.log10(hi)
        return self.H - self.B - (y - lo) / (hi - lo) * (self.H - self.T - self.B)

    # -- ticks ----------------------------------------------------------

    @staticmethod
    def _linear_ticks(lo: float, hi: float, want: int = 6) -> List[float]:
        span = hi - lo
        if span <= 0:
            return [lo]
        raw = span / want
        mag = 10 ** math.floor(math.log10(raw))
        step = min((m * mag for m in (1, 2, 2.5, 5, 10)),
                   key=lambda s: abs(s - raw))
        first = math.ceil(lo / step) * step
        out, t = [], first
        while t <= hi + step * 1e-9:
            out.append(t)
            t += step
        return out

    @staticmethod
    def _log_ticks(lo: float, hi: float) -> List[float]:
        out = []
        e = math.floor(math.log10(lo))
        while 10 ** e <= hi * 1.0001:
            for m in (1, 2, 5):
                v = m * 10 ** e
                if lo <= v <= hi:
                    out.append(v)
            e += 1
        return out or [lo, hi]

    def axes(self, xticks: Optional[Sequence[float]] = None,
             xtick_label: Optional[Callable[[float], str]] = None,
             ytick_label: Optional[Callable[[float], str]] = None) -> None:
        fx = xtick_label or _fmt
        fy = ytick_label or _fmt
        xs = list(xticks) if xticks is not None else (
            self._log_ticks(self.x0, self.x1) if self.xlog
            else self._linear_ticks(self.x0, self.x1))
        ys = (self._log_ticks(self.y0, self.y1) if self.ylog
              else self._linear_ticks(self.y0, self.y1))
        for x in xs:
            px = self.px(x)
            self.body.append(
                f'<line x1="{px:.1f}" y1="{self.T}" x2="{px:.1f}" '
                f'y2="{self.H - self.B}" stroke="#eee"/>')
            self.body.append(
                f'<text x="{px:.1f}" y="{self.H - self.B + 15}" '
                f'text-anchor="middle" fill="#444">{html.escape(fx(x))}</text>')
        for y in ys:
            py = self.py(y)
            self.body.append(
                f'<line x1="{self.L}" y1="{py:.1f}" x2="{self.W - self.R}" '
                f'y2="{py:.1f}" stroke="#eee"/>')
            self.body.append(
                f'<text x="{self.L - 8}" y="{py + 3.5:.1f}" text-anchor="end" '
                f'fill="#444">{html.escape(fy(y))}</text>')

    # -- marks ----------------------------------------------------------

    def line(self, points: Sequence[Tuple[float, float]], color: str,
             label: str = '', dashed: bool = False, mark: str = 'circle',
             tips: Optional[Sequence[str]] = None) -> None:
        """A polyline through the points, with a mark on each.

        A `None` in either coordinate breaks the line: see the module
        docstring on why a gap is left open.
        """
        run: List[str] = []
        for i, (x, y) in enumerate(points):
            if x is None or y is None:
                if len(run) > 1:
                    self._polyline(run, color, dashed)
                run = []
                continue
            run.append(f'{self.px(x):.1f},{self.py(y):.1f}')
            tip = tips[i] if tips and i < len(tips) else f'{_fmt(x)}, {_fmt(y)}'
            self._mark(self.px(x), self.py(y), color, mark, tip)
        if len(run) > 1:
            self._polyline(run, color, dashed)
        if label:
            self.legend.append((label, color))

    def _polyline(self, run: Sequence[str], color: str, dashed: bool) -> None:
        dash = ' stroke-dasharray="5 4"' if dashed else ''
        self.body.append(
            f'<polyline points="{" ".join(run)}" fill="none" stroke="{color}" '
            f'stroke-width="1.8" stroke-linejoin="round"{dash}/>')

    def _mark(self, px: float, py: float, color: str, kind: str,
              tip: str) -> None:
        t = f'<title>{html.escape(tip)}</title>'
        if kind == 'square':
            shape = (f'<rect x="{px - 3.2:.1f}" y="{py - 3.2:.1f}" width="6.4" '
                     f'height="6.4" fill="{color}">{t}</rect>')
        elif kind == 'triangle':
            shape = (f'<polygon points="{px:.1f},{py - 4:.1f} '
                     f'{px - 3.7:.1f},{py + 2.6:.1f} {px + 3.7:.1f},'
                     f'{py + 2.6:.1f}" fill="{color}">{t}</polygon>')
        elif kind == 'diamond':
            shape = (f'<polygon points="{px:.1f},{py - 4.2:.1f} '
                     f'{px + 4.2:.1f},{py:.1f} {px:.1f},{py + 4.2:.1f} '
                     f'{px - 4.2:.1f},{py:.1f}" fill="{color}">{t}</polygon>')
        else:
            shape = (f'<circle cx="{px:.1f}" cy="{py:.1f}" r="3.4" '
                     f'fill="{color}">{t}</circle>')
        self.body.append(shape)

    def bar(self, x: float, w: float, y: float, color: str, tip: str,
            y_base: Optional[float] = None) -> None:
        base = self.py(self.y0 if y_base is None else y_base)
        top = self.py(y)
        self.body.append(
            f'<rect x="{x:.1f}" y="{min(top, base):.1f}" width="{w:.1f}" '
            f'height="{abs(base - top):.1f}" fill="{color}" opacity="0.85">'
            f'<title>{html.escape(tip)}</title></rect>')

    def rule(self, y: float, text: str, color: str = '#999') -> None:
        py = self.py(y)
        self.body.append(
            f'<line x1="{self.L}" y1="{py:.1f}" x2="{self.W - self.R}" '
            f'y2="{py:.1f}" stroke="{color}" stroke-dasharray="4 3"/>')
        self.body.append(
            f'<text x="{self.L + 6}" y="{py - 4:.1f}" '
            f'fill="{color}">{html.escape(text)}</text>')

    def text(self, x: float, y: float, s: str, anchor: str = 'start',
             fill: str = '#444', size: int = 11) -> None:
        self.body.append(
            f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
            f'fill="{fill}" font-size="{size}">{html.escape(s)}</text>')

    # -- output ---------------------------------------------------------

    def render(self) -> str:
        fits = max(int((self.H - self.T - 24) // 14), 1)
        entries = list(self.legend)
        more = ''
        if len(entries) > fits:
            more = f'+{len(entries) - fits + 1} more, hover the marks'
            entries = entries[:fits - 1]
        x = self.W - self.legend_width + 8 if self.legend_width else self.L + 8
        legend = []
        for i, (label, color) in enumerate(entries):
            y = self.T + 10 + i * 14
            legend.append(
                f'<rect x="{x}" y="{y - 8}" width="8" height="8" '
                f'fill="{color}"/>'
                f'<text x="{x + 12}" y="{y}" fill="#333" font-size="10">'
                f'{html.escape(label[:28])}</text>')
        if more:
            legend.append(
                f'<text x="{x}" y="{self.T + 10 + len(entries) * 14}" '
                f'fill="#888" font-size="10">{html.escape(more)}</text>')
        note = (f'<text x="{self.L}" y="{self.H - 10}" fill="#777" '
                f'font-size="10">{html.escape(self.note)}</text>'
                if self.note else '')
        return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{self.W}" \
height="{self.H}" viewBox="0 0 {self.W} {self.H}" font-family="sans-serif" \
font-size="11">
<rect width="{self.W}" height="{self.H}" fill="white"/>
<text x="{self.L}" y="22" font-size="14" fill="#111">\
{html.escape(self.title)}</text>
{chr(10).join(self.body)}
<line x1="{self.L}" y1="{self.H - self.B}" x2="{self.W - self.R}" \
y2="{self.H - self.B}" stroke="#333"/>
<line x1="{self.L}" y1="{self.T}" x2="{self.L}" y2="{self.H - self.B}" \
stroke="#333"/>
<text x="{(self.L + self.W - self.R) / 2:.0f}" y="{self.H - self.B + 34}" \
text-anchor="middle" fill="#222">{html.escape(self.xlabel)}</text>
<text x="16" y="{(self.T + self.H - self.B) / 2:.0f}" text-anchor="middle" \
fill="#222" transform="rotate(-90 16 {(self.T + self.H - self.B) / 2:.0f})">\
{html.escape(self.ylabel)}</text>
{chr(10).join(legend)}
{note}
</svg>
'''


# ----------------------------------------------------------------------
# Series
# ----------------------------------------------------------------------

def series_key(row: Dict) -> str:
    """One curve per workload *and* configuration -- see the docstring on why
    they are not averaged together."""
    config = row.get('config') or ''
    return f"{row['workload']}" + (f' [{config}]' if config and config != 'default' else '')


def by_series(rows: Sequence[Dict]) -> Dict[str, List[Dict]]:
    out: Dict[str, List[Dict]] = {}
    for row in rows:
        out.setdefault(series_key(row), []).append(row)
    for rows_ in out.values():
        rows_.sort(key=lambda r: r.get('batch') or 0)
    return out


def rate(row: Dict, what: str) -> Optional[float]:
    """The quantity a view is plotting, from whichever field carries it.

    `gflops` and `gbytes` are written by `run.py`; recomputing them from
    `flops` and the clock is the same arithmetic and disagrees when the run
    used a different clock, so the stored figure wins where it exists.
    """
    nanos = row.get('event_ns') or row.get('wall_ns')
    if what == 'gflops':
        if row.get('gflops'):
            return float(row['gflops'])
        return row['flops'] / nanos if (nanos and row.get('flops')) else None
    if what == 'gbytes':
        if row.get('gbytes'):
            return float(row['gbytes'])
        return row['bytes'] / nanos if (nanos and row.get('bytes')) else None
    if what == 'ns_per_element':
        if row.get('ns_per_element'):
            return float(row['ns_per_element'])
        return nanos / row['batch'] if (nanos and row.get('batch')) else None
    if what == 'ns':
        return float(nanos) if nanos else None
    raise KeyError(what)


RATE_LABEL = {'gflops': 'GFLOP/s', 'gbytes': 'GB/s (compulsory)',
              'ns_per_element': 'ns per element', 'ns': 'kernel ns'}


# ----------------------------------------------------------------------
# Views over the timing run alone
# ----------------------------------------------------------------------

def view_scaling(run: Run, what: str = 'gflops') -> Optional[str]:
    """Rate against launch size.

    The one that answers "is the batch large enough to mean anything". A
    curve still rising at the largest batch measured has not saturated, and
    every ratio taken from its last point is a ratio at an arbitrary place on
    a slope.
    """
    groups = by_series(run.rows)
    if not any(len(v) > 1 for v in groups.values()):
        return None                    # one batch: nothing to plot against
    c = Canvas(f'{RATE_LABEL[what]} against launch size — {run.label}',
               'elements per launch', RATE_LABEL[what], xlog=True,
               legend_width=190,
               note=f'{run.device}; a curve still rising at the right edge '
                    f'has not saturated.')
    xs = [r['batch'] for rows in groups.values() for r in rows if r.get('batch')]
    ys = [y for rows in groups.values() for y in
          (rate(r, what) for r in rows) if y]
    if not xs or not ys:
        return None
    c.fit(xs, ys, y_from_zero=(what != 'ns_per_element'))
    c.axes(xticks=sorted(set(xs)), xtick_label=_fmt)
    for i, (name, rows) in enumerate(sorted(groups.items())):
        pts = [(r.get('batch'), rate(r, what)) for r in rows]
        tips = [f'{name} @ {_fmt(r.get("batch") or 0)}: '
                f'{rate(r, what) or 0:.2f} {RATE_LABEL[what]}' for r in rows]
        c.line(pts, PALETTE[i % len(PALETTE)], name,
               mark=MARKS[(i // len(PALETTE)) % len(MARKS)], tips=tips)
    return c.render()


def view_spread(run: Run) -> Optional[str]:
    """Min, median and p90 of the kernel clock, per series.

    Drawn because every comparison in this directory is a ratio of two
    measurements, and a ratio is only worth reading where the spread is
    smaller than the difference claimed. `run.py` already records the three
    figures; nothing else here shows them.
    """
    rows = [r for r in run.rows
            if r.get('event_ns_p50') and r.get('event_ns_min')]
    if not rows:
        return None
    rows = sorted(rows, key=lambda r: (r['workload'], r.get('batch') or 0))
    c = Canvas(f'Kernel time, min to p90 — {run.label}', '',
               'kernel ns', ylog=True, width=max(760, 46 * len(rows) + 140),
               tilted_labels=True,
               note=f'{run.device}; the bar is min to p90, the tick is the '
                    f'median. A difference smaller than the bar is not one.')
    ys = ([r['event_ns_min'] for r in rows]
          + [r.get('event_ns_p90') or r['event_ns_p50'] for r in rows])
    c.fit([0, len(rows)], ys)
    c.axes(xticks=[])
    step = (c.W - c.L - c.R) / max(len(rows), 1)
    for i, r in enumerate(rows):
        x = c.L + step * (i + 0.5)
        lo, mid = r['event_ns_min'], r['event_ns_p50']
        hi = r.get('event_ns_p90') or mid
        color = PALETTE[i % len(PALETTE)]
        c.body.append(
            f'<line x1="{x:.1f}" y1="{c.py(lo):.1f}" x2="{x:.1f}" '
            f'y2="{c.py(hi):.1f}" stroke="{color}" stroke-width="7" '
            f'opacity="0.45"><title>{html.escape(series_key(r))} @ '
            f'{_fmt(r.get("batch") or 0)}: min {lo:.0f}, p50 {mid:.0f}, '
            f'p90 {hi:.0f} ns ({(hi - lo) / mid * 100:.1f} % spread)</title>'
            f'</line>')
        c.body.append(
            f'<line x1="{x - 6:.1f}" y1="{c.py(mid):.1f}" x2="{x + 6:.1f}" '
            f'y2="{c.py(mid):.1f}" stroke="{color}" stroke-width="2"/>')
        c.tilted_label(x, f'{r["workload"]} @ {_fmt(r.get("batch") or 0)}')
    return c.render()


def view_compare(runs: Sequence[Run], what: str = 'ns_per_element',
                 batch: Optional[int] = None) -> Optional[str]:
    """Two or more runs, per workload, as bars against the first.

    The first run is the reference and is drawn at 1. What the other bars
    say is the ratio, which is the only thing a reader of two builds wants
    and the thing a pair of absolute axes makes them compute by hand.
    """
    if len(runs) < 2:
        return None
    tables = []
    for run in runs:
        t: Dict[str, Dict[int, float]] = {}
        for row in run.rows:
            v = rate(row, what)
            if v:
                t.setdefault(series_key(row), {})[row.get('batch') or 0] = v
        tables.append(t)
    common = sorted(set.intersection(*[set(t) for t in tables]))
    if not common:
        return None

    def pick(t: Dict[int, float]) -> Optional[float]:
        if batch is not None:
            return t.get(batch)
        return t[max(t)] if t else None

    lower_is_better = what in ('ns_per_element', 'ns')
    c = Canvas(f'{runs[0].label} against ' + ', '.join(r.label for r in runs[1:])
               + f' — {RATE_LABEL[what]}',
               '', f'ratio to {runs[0].label}',
               width=max(760, 54 * len(common) + 180),
               tilted_labels=True, legend_width=150,
               note=f'{runs[0].device}; above 1 is faster than '
                    f'{runs[0].label}. '
                    + ('Lower ' + RATE_LABEL[what] + ' is better, so the bar '
                       'is the reference over the candidate.'
                       if lower_is_better else
                       'Higher is better, so the bar is the candidate over '
                       'the reference.'))
    ratios: List[List[Optional[float]]] = []
    for name in common:
        ref = pick(tables[0][name])
        row = []
        for t in tables[1:]:
            v = pick(t[name])
            if not ref or not v:
                row.append(None)
            else:
                row.append(ref / v if lower_is_better else v / ref)
        ratios.append(row)
    flat = [v for row in ratios for v in row if v]
    if not flat:
        return None
    c.fit([0, len(common)], flat + [1.0], y_from_zero=True)
    c.axes(xticks=[])
    c.rule(1.0, runs[0].label)
    step = (c.W - c.L - c.R) / max(len(common), 1)
    bw = step / (len(runs) + 0.6)
    for i, name in enumerate(common):
        for j, v in enumerate(ratios[i]):
            if v is None:
                continue
            x = c.L + step * i + bw * (j + 0.4)
            c.bar(x, bw * 0.9, v, PALETTE[(j + 1) % len(PALETTE)],
                  f'{name}: {runs[j + 1].label} is {v:.3f}x '
                  f'{runs[0].label}')
        c.tilted_label(c.L + step * (i + 0.5), name)
    for j, run in enumerate(runs[1:]):
        c.legend.append((run.label, PALETTE[(j + 1) % len(PALETTE)]))
    geo = math.exp(sum(math.log(v) for v in flat) / len(flat))
    c.text(c.L, c.T - 8, f'geometric mean over {len(flat)} points: {geo:.4f}x')
    return c.render()


# ----------------------------------------------------------------------
# Views that need the ceilings
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class Roof:
    """The two ceilings, however they were obtained.

    `roofline.py` measures them; this takes them as given so that a chart can
    be drawn from a saved run on a machine that is not the one measured, and
    says in the note which it was.
    """
    flops_per_s: float
    bytes_per_s: float
    source: str = 'given'
    datatype: str = ''

    @property
    def ridge(self) -> float:
        return self.flops_per_s / self.bytes_per_s

    def bound(self, intensity: float) -> float:
        return min(self.flops_per_s, self.bytes_per_s * intensity)


def _roof_frame(c: Canvas, roof: Roof, xs: Sequence[float],
                ys: Sequence[float]) -> None:
    c.fit(list(xs) + [roof.ridge / 8, roof.ridge * 8],
          list(ys) + [roof.flops_per_s * 1.4, roof.flops_per_s / 500])
    c.axes()
    pts = ' '.join(f'{c.px(x):.1f},{c.py(roof.bound(x)):.1f}'
                   for x in (c.x0, roof.ridge, c.x1))
    c.body.append(f'<polyline points="{pts}" fill="none" stroke="#222" '
                  f'stroke-width="2"/>')
    c.body.append(
        f'<line x1="{c.px(roof.ridge):.1f}" y1="{c.T}" '
        f'x2="{c.px(roof.ridge):.1f}" y2="{c.H - c.B}" stroke="#bbb" '
        f'stroke-dasharray="4 3"/>')
    c.text(c.px(roof.ridge) + 4, c.T + 12,
           f'ridge {roof.ridge:.1f} flop/byte', fill='#888')


def view_roofline(run: Run, roof: Roof) -> Optional[str]:
    """Every measured point under the two ceilings, at compulsory intensity."""
    rows = [r for r in run.rows if r.get('intensity') and rate(r, 'gflops')]
    if not rows:
        return None
    c = Canvas(f'Roofline — {run.label}', 'arithmetic intensity '
               '(flop/byte, compulsory)', 'achieved FLOP/s',
               xlog=True, ylog=True, legend_width=190,
               note=f'{run.device}; roofs {roof.source}. Compulsory intensity '
                    f'is what the description implies, not what the kernel '
                    f'paid: see the measured roofline where a profile exists.')
    xs = [r['intensity'] for r in rows]
    ys = [rate(r, 'gflops') * 1e9 for r in rows]
    _roof_frame(c, roof, xs, ys)
    groups = by_series(rows)
    for i, (name, rs) in enumerate(sorted(groups.items())):
        color = PALETTE[i % len(PALETTE)]
        for r in rs:
            achieved = rate(r, 'gflops') * 1e9
            frac = achieved / roof.bound(r['intensity'])
            c._mark(c.px(r['intensity']), c.py(achieved), color,
                    MARKS[(i // len(PALETTE)) % len(MARKS)],
                    f'{name} @ {_fmt(r.get("batch") or 0)}: '
                    f'{achieved / 1e9:.1f} GFLOP/s, {frac * 100:.0f} % of roof')
        c.legend.append((name, color))
    return c.render()


def view_trajectory(run: Run, roof: Roof) -> Optional[str]:
    """The same axes, but each kernel is the path it walks as the batch grows.

    Compulsory intensity does not depend on the launch size, so the path is
    vertical: it shows at which batch a kernel stops climbing, and how far
    under its roof it stopped. A kernel whose path is still rising at the top
    point was not measured far enough.
    """
    groups = {k: v for k, v in by_series(run.rows).items() if len(v) > 1}
    if not groups:
        return None
    c = Canvas(f'Climb toward the roof — {run.label}',
               'arithmetic intensity (flop/byte, compulsory)',
               'achieved FLOP/s', xlog=True, ylog=True, legend_width=190,
               note=f'{run.device}; one path per kernel, smallest batch at '
                    f'the bottom. Still rising at the top means the launch '
                    f'never saturated.')
    xs = [r['intensity'] for rs in groups.values() for r in rs
          if r.get('intensity')]
    ys = [rate(r, 'gflops') * 1e9 for rs in groups.values() for r in rs
          if rate(r, 'gflops')]
    if not xs:
        return None
    _roof_frame(c, roof, xs, ys)
    for i, (name, rs) in enumerate(sorted(groups.items())):
        pts = [(r.get('intensity'),
                (rate(r, 'gflops') or 0) * 1e9 or None) for r in rs]
        tips = [f'{name} @ {_fmt(r.get("batch") or 0)}: '
                f'{(rate(r, "gflops") or 0):.1f} GFLOP/s' for r in rs]
        c.line(pts, PALETTE[i % len(PALETTE)], name,
               mark=MARKS[(i // len(PALETTE)) % len(MARKS)], tips=tips)
    return c.render()


def view_efficiency(run: Run, roof: Roof) -> Optional[str]:
    """Fraction of the roof against launch size.

    The same data as `trajectory` with the roof divided out, which is the
    form that compares kernels of different intensity against each other.
    """
    groups = {k: v for k, v in by_series(run.rows).items()
              if len(v) > 1 and all(r.get('intensity') for r in v)}
    if not groups:
        return None
    c = Canvas(f'Fraction of the roof — {run.label}', 'elements per launch',
               '% of the bound the intensity allows', xlog=True,
               legend_width=190,
               note=f'{run.device}; roofs {roof.source}. Below the lower '
                    f'rule neither line is the constraint -- look at '
                    f'occupancy, at launch overhead, or at whether the batch '
                    f'filled the grid.')
    xs = [r['batch'] for rs in groups.values() for r in rs if r.get('batch')]
    fr = []
    for rs in groups.values():
        for r in rs:
            g = rate(r, 'gflops')
            if g:
                fr.append(g * 1e9 / roof.bound(r['intensity']) * 100)
    if not xs or not fr:
        return None
    c.fit(xs, fr + [100.0], y_from_zero=True)
    c.axes(xticks=sorted(set(xs)))
    c.rule(80.0, 'at the roof')
    c.rule(50.0, 'neither line binds below here')
    for i, (name, rs) in enumerate(sorted(groups.items())):
        pts, tips = [], []
        for r in rs:
            g = rate(r, 'gflops')
            v = g * 1e9 / roof.bound(r['intensity']) * 100 if g else None
            pts.append((r.get('batch'), v))
            tips.append(f'{name} @ {_fmt(r.get("batch") or 0)}: '
                        f'{v or 0:.1f} % of roof')
        c.line(pts, PALETTE[i % len(PALETTE)], name,
               mark=MARKS[(i // len(PALETTE)) % len(MARKS)], tips=tips)
    return c.render()


# ----------------------------------------------------------------------
# Views that need a profiler
# ----------------------------------------------------------------------

def _measured_bytes(row: Dict) -> Optional[float]:
    r = row.get('dram_read_bytes')
    w = row.get('dram_write_bytes')
    if r is None and w is None:
        return None
    return (r or 0.0) + (w or 0.0)


def view_traffic(run: Run, profile: Dict[str, List[Dict]]) -> Optional[str]:
    """What the kernel moved against what the description required.

    The ratio is `profile.py`'s `traffic_amplification`, drawn rather than
    tabulated because its shape over a set of kernels is the interesting
    part: a ratio near one means the traffic is compulsory and the only
    lever is reuse that does not exist; a large one means the same bytes
    were fetched repeatedly and a blocking change can take them back.

    Below one is not an error -- it means a cache held what the description
    counted as compulsory, which is what an operator small enough to stay
    resident does.
    """
    pairs = []
    for row in run.rows:
        rows = profile.get(row['workload'])
        if not rows or not row.get('bytes'):
            continue
        measured = _measured_bytes(hottest(rows))
        if measured:
            pairs.append((series_key(row), row['bytes'], measured))
    if not pairs:
        return None
    pairs.sort(key=lambda p: -(p[2] / p[1]))
    c = Canvas(f'Traffic against what the description requires — {run.label}',
               '', 'measured DRAM bytes / compulsory bytes',
               width=max(760, 50 * len(pairs) + 160), ylog=True,
               tilted_labels=True,
               note=f'{run.device}; above one, the same bytes were fetched '
                    f'more than once. Below one, a cache held them.')
    c.fit([0, len(pairs)], [p[2] / p[1] for p in pairs] + [1.0])
    c.axes(xticks=[])
    c.rule(1.0, 'compulsory')
    step = (c.W - c.L - c.R) / max(len(pairs), 1)
    for i, (name, compulsory, measured) in enumerate(pairs):
        x = c.L + step * i + step * 0.15
        c.bar(x, step * 0.7, measured / compulsory,
              PALETTE[i % len(PALETTE)],
              f'{name}: {measured / 1e6:.1f} MB measured against '
              f'{compulsory / 1e6:.1f} MB compulsory '
              f'({measured / compulsory:.2f}x)', y_base=1.0)
        c.tilted_label(x + step * 0.7, name)
    return c.render()


def view_occupancy(run: Run, profile: Dict[str, List[Dict]]) -> Optional[str]:
    """Achieved rate against occupancy, one point per kernel.

    The view that separates "not enough parallelism" from "enough and still
    slow". Points climbing to the right are limited by how much of the
    machine was busy; points far right and low are not -- that is a
    dependent chain or a spill, and the fix is in the kernel rather than in
    the launch geometry.
    """
    pts = []
    for row in run.rows:
        rows = profile.get(row['workload'])
        g = rate(row, 'gflops')
        if not rows or not g:
            continue
        occ = hottest(rows).get('occupancy')
        if occ:
            pts.append((series_key(row), occ * (100 if occ <= 1.0 else 1), g))
    if not pts:
        return None
    c = Canvas(f'Rate against occupancy — {run.label}',
               'occupancy (% of peak)', 'GFLOP/s', legend_width=190,
               note=f'{run.device}; far right and low is not an occupancy '
                    f'problem.')
    c.fit([p[1] for p in pts], [p[2] for p in pts], y_from_zero=True)
    c.axes()
    for i, (name, occ, g) in enumerate(sorted(pts)):
        color = PALETTE[i % len(PALETTE)]
        c._mark(c.px(occ), c.py(g), color,
                MARKS[(i // len(PALETTE)) % len(MARKS)],
                f'{name}: {g:.1f} GFLOP/s at {occ:.1f} % occupancy')
        c.legend.append((name, color))
    return c.render()


def view_measured_roofline(run: Run, roof: Roof,
                           profile: Dict[str, List[Dict]]) -> Optional[str]:
    """The roofline at the intensity the kernel actually ran at.

    Compulsory intensity is a property of the description; measured
    intensity is flops over the bytes that crossed the memory controller.
    Where a kernel re-reads an operand the second is smaller than the first,
    and the point slides *left* -- often from the flat part of the roof onto
    the slope, which changes what the chart says to do about it. Both points
    are drawn, joined by a line, so the move is visible rather than
    asserted.
    """
    arrows = []
    for row in run.rows:
        rows = profile.get(row['workload'])
        g = rate(row, 'gflops')
        if not rows or not g or not row.get('intensity') or not row.get('flops'):
            continue
        measured = _measured_bytes(hottest(rows))
        if not measured:
            continue
        arrows.append((series_key(row), row['intensity'],
                       row['flops'] / measured, g * 1e9))
    if not arrows:
        return None
    c = Canvas(f'Compulsory against measured intensity — {run.label}',
               'arithmetic intensity (flop/byte)', 'achieved FLOP/s',
               xlog=True, ylog=True, legend_width=190,
               note=f'{run.device}; roofs {roof.source}. Hollow is what the '
                    f'description implies, filled is what crossed the memory '
                    f'controller.')
    xs = [a[1] for a in arrows] + [a[2] for a in arrows]
    ys = [a[3] for a in arrows]
    _roof_frame(c, roof, xs, ys)
    for i, (name, compulsory, measured, achieved) in enumerate(sorted(arrows)):
        color = PALETTE[i % len(PALETTE)]
        y = c.py(achieved)
        c.body.append(
            f'<line x1="{c.px(compulsory):.1f}" y1="{y:.1f}" '
            f'x2="{c.px(measured):.1f}" y2="{y:.1f}" stroke="{color}" '
            f'stroke-width="1.2" opacity="0.7"/>')
        c.body.append(
            f'<circle cx="{c.px(compulsory):.1f}" cy="{y:.1f}" r="3.6" '
            f'fill="white" stroke="{color}" stroke-width="1.6">'
            f'<title>{html.escape(name)}: compulsory {compulsory:.2f} '
            f'flop/byte</title></circle>')
        c._mark(c.px(measured), y, color, 'circle',
                f'{name}: measured {measured:.2f} flop/byte, '
                f'{achieved / 1e9:.1f} GFLOP/s '
                f'({measured / compulsory:.2f}x the compulsory intensity)')
        c.legend.append((name, color))
    return c.render()


# ----------------------------------------------------------------------
# Assembly
# ----------------------------------------------------------------------

def index_html(charts: Sequence[Tuple[str, str, str]], title: str) -> str:
    """One page with every chart that had data, and its question above it."""
    blocks = '\n'.join(
        f'<section><h2>{html.escape(name)}</h2>'
        f'<p>{html.escape(question)}</p>{svg}</section>'
        for name, question, svg in charts)
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>
 body {{ font-family: sans-serif; margin: 2rem auto; max-width: 62rem;
        color: #222; line-height: 1.5; }}
 h1 {{ font-size: 1.4rem; }} h2 {{ font-size: 1.05rem; margin-bottom: .2rem; }}
 section {{ margin: 2.4rem 0; }}
 section p {{ margin: 0 0 .6rem; color: #555; font-size: .92rem; }}
 svg {{ border: 1px solid #e3e3e3; max-width: 100%; height: auto; }}
</style></head><body>
<h1>{html.escape(title)}</h1>
<p>Hover any mark for the figures behind it.</p>
{blocks}
</body></html>
'''


#: The question each view answers, shown above it on the index page.  Kept
#: here rather than in the functions so that the page reads as one document.
QUESTIONS = {
    'scaling': 'Where does the rate stop rising with the launch size? A ratio '
               'taken from a curve that is still climbing is a ratio at an '
               'arbitrary place on a slope.',
    'bandwidth': 'The same against compulsory bytes rather than flops.',
    'latency': 'What does one element cost, and from which batch on is that '
               'cost flat?',
    'spread': 'Is the measurement steady enough for the difference being '
              'claimed?',
    'roofline': 'Is the kernel allowed to go faster, at the intensity its '
                'description implies?',
    'trajectory': 'How does each kernel climb toward its roof as the launch '
                  'grows, and where does it stop?',
    'efficiency': 'The same climb with the roof divided out, so kernels of '
                  'different intensity compare.',
    'compare': 'Which build is ahead, per kernel, and by how much?',
    'traffic': 'How much of the traffic was compulsory, and how much was the '
               'same bytes fetched again?',
    'occupancy': 'Does the rate follow how much of the machine was busy?',
    'measured-roofline': 'Where does the point move once the intensity is the '
                         'one the memory controller saw?',
}


def build_charts(runs: Sequence[Run], roof: Optional[Roof],
                 profile: Optional[Dict[str, List[Dict]]],
                 batch: Optional[int]) -> Tuple[List[Tuple[str, str, str]], List[str]]:
    charts: List[Tuple[str, str, str]] = []
    skipped: List[str] = []

    def add(name: str, svg: Optional[str], why: str = '') -> None:
        if svg:
            charts.append((name, QUESTIONS.get(name, ''), svg))
        else:
            skipped.append(f'{name}: {why or "no data for it in this run"}')

    primary = runs[0]
    add('scaling', view_scaling(primary, 'gflops'),
        'only one batch size in the run -- give the suite several')
    add('bandwidth', view_scaling(primary, 'gbytes'),
        'only one batch size in the run')
    add('latency', view_scaling(primary, 'ns_per_element'),
        'only one batch size in the run')
    add('spread', view_spread(primary),
        'the run carries no percentile columns')
    if roof is None:
        for name in ('roofline', 'trajectory', 'efficiency'):
            skipped.append(f'{name}: no ceilings -- pass --peak-flops and '
                           f'--peak-bytes, or --roofline-json')
    else:
        add('roofline', view_roofline(primary, roof))
        add('trajectory', view_trajectory(primary, roof),
            'only one batch size in the run')
        add('efficiency', view_efficiency(primary, roof),
            'only one batch size in the run')
    if len(runs) > 1:
        add('compare', view_compare(runs, 'ns_per_element', batch))
    else:
        skipped.append('compare: pass a second bench.json')
    if profile:
        add('traffic', view_traffic(primary, profile),
            'the profiler reported no DRAM counters')
        add('occupancy', view_occupancy(primary, profile),
            'the profiler reported no occupancy')
        if roof is not None:
            add('measured-roofline',
                view_measured_roofline(primary, roof, profile),
                'the profiler reported no DRAM counters')
    else:
        for name in ('traffic', 'occupancy', 'measured-roofline'):
            skipped.append(f'{name}: no profile.json (see --profile)')
    return charts, skipped


def roof_from(args, runs: Sequence[Run]) -> Optional[Roof]:
    """Ceilings from the flags, or from a `roofline.py --out` file."""
    if args.roofline_json:
        blob = json.loads(Path(args.roofline_json).read_text())
        ceiling = blob.get('ceiling', blob)
        flops = ceiling.get('flops_per_s')
        byts = ceiling.get('bytes_per_s')
        if flops and byts:
            return Roof(float(flops), float(byts),
                        f'from {Path(args.roofline_json).name}',
                        ceiling.get('datatype', ''))
    if args.peak_flops and args.peak_bytes:
        return Roof(args.peak_flops * 1e12, args.peak_bytes * 1e9,
                    'given on the command line',
                    (runs[0].datatypes or [''])[0])
    return None


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        epilog='The first bench.json is the reference the others compare '
               'against.')
    ap.add_argument('bench', type=Path, nargs='+',
                    help='bench.json, or the directory run.py --out wrote')
    ap.add_argument('--label', action='append', default=None,
                    help='name for each run, in order; defaults to the '
                         'directory name')
    ap.add_argument('--profile', type=Path, default=None,
                    help='profile.json, or the directory profile.py --out '
                         'wrote')
    ap.add_argument('--roofline-json', default=None,
                    help='a file carrying flops_per_s and bytes_per_s, as '
                         'roofline.py --out writes')
    ap.add_argument('--peak-flops', type=float, default=None,
                    help='compute ceiling in TFLOP/s, where no file has it')
    ap.add_argument('--peak-bytes', type=float, default=None,
                    help='memory ceiling in GB/s, where no file has it')
    ap.add_argument('--batch', type=int, default=None,
                    help='the launch size the comparison is taken at '
                         '(default: the largest both runs have)')
    ap.add_argument('--out', type=Path, default=Path('bench-plots'),
                    help='directory for the SVGs and index.html')
    args = ap.parse_args()

    labels = args.label or []
    runs = []
    for i, path in enumerate(args.bench):
        try:
            runs.append(load_run(path, labels[i] if i < len(labels) else None))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f'{path}: {exc}', file=sys.stderr)
            return 2
    if not runs[0].rows:
        print(f'{runs[0].path}: no usable rows', file=sys.stderr)
        return 2

    profile = None
    if args.profile:
        try:
            profile = load_profile(args.profile)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f'{args.profile}: {exc}', file=sys.stderr)

    roof = roof_from(args, runs)
    charts, skipped = build_charts(runs, roof, profile, args.batch)

    args.out.mkdir(parents=True, exist_ok=True)
    for name, _, svg in charts:
        (args.out / f'{name}.svg').write_text(svg)
    title = (f'{runs[0].label} — {runs[0].device}'
             + (f' against {", ".join(r.label for r in runs[1:])}'
                if len(runs) > 1 else ''))
    (args.out / 'index.html').write_text(index_html(charts, title))

    for name, _, _ in charts:
        print(f'written: {args.out / (name + ".svg")}')
    print(f'written: {args.out / "index.html"}')
    for note in skipped:
        print(f'skipped {note}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
