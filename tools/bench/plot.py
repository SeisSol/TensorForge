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

from pdfout import (Page, PCircle, PLine, PPolygon, PPoly, PRect, PText,
                    Prim, _wrap, write_pdf)

# ----------------------------------------------------------------------
# Reading what the other tools wrote
# ----------------------------------------------------------------------


def _results(blob) -> List[Dict]:
    """The rows of a results file, a bare list or `{'results': [...]}`."""
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


def series_style(i: int) -> Tuple[str, str]:
    """The colour and the mark for curve `i`.

    Both channels, and the mark varies *within* the palette rather than
    only once it wraps.  Eight hues cannot all stay apart under
    deuteranopia -- the validator puts the closest adjacent pair of this
    set at a few units of Delta E, where eight is the floor that secondary
    encoding makes legal -- so a reader who cannot separate two of them
    separates a circle from a triangle instead.

    The pair stays unique for 32 curves: the mark advances with the index
    inside a block of eight *and* with the block, so curve 0 and curve 8
    share a colour and differ in shape.
    """
    return PALETTE[i % len(PALETTE)], MARKS[(i % len(PALETTE)
                                             + i // len(PALETTE)) % len(MARKS)]


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


#: The canvas most recently rendered (`Canvas.render`), which is how a view
#: that returns SVG still yields a PDF page.
_LAST_CANVAS: Optional['Canvas'] = None


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
        # The note is wrapped to the drawing and the sheet grows for it.
        # Written on one line it ran off the right edge of every chart
        # whose note said more than a dozen words, and a note nobody can
        # read is a note nobody wrote.
        self._note_lines = _wrap(
            note, max(int((width - 96) / 5.2), 40)) if note else []
        extra = 12 * max(len(self._note_lines) - 1, 0)
        self.H = height + (58 if tilted_labels else 0) + extra
        self.L, self.T = 74, 46
        self.R = 22 + legend_width
        # The bottom margin grows with the note as the sheet does, so the
        # wrapped lines land below the axis label instead of across it.
        self.B = 62 + (58 if tilted_labels else 0) + extra
        self.legend_width = legend_width
        self.tilted = tilted_labels
        #: What was drawn, as primitives.  `body` is the same list spelled
        #: as SVG and is appended to in step (`_emit`), so a caller that
        #: puts a string in it directly still draws -- into the SVG only,
        #: which is why nothing in this file does that any more.
        self.prims: List[Prim] = []
        self.body: List[str] = []
        self.legend: List[Tuple[str, str]] = []
        self.x0 = self.x1 = self.y0 = self.y1 = 0.0

    def _emit(self, prim: Prim) -> None:
        self.prims.append(prim)
        self.body.append(prim.to_svg())

    def tilted_label(self, x: float, text: str) -> None:
        """A name under a bar, written at an angle so that neighbours do not
        collide.  Anchored at its end so the text runs away from the axis
        rather than through it."""
        y = self.H - self.B + 13
        self._emit(PText(x, y, text[:30], anchor='end', fill='#555', size=9,
                         rotate=-45))

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
            self._emit(PLine(px, self.T, px, self.H - self.B, '#eee'))
            self._emit(PText(px, self.H - self.B + 15, fx(x),
                             anchor='middle', fill='#444'))
        for y in ys:
            py = self.py(y)
            self._emit(PLine(self.L, py, self.W - self.R, py, '#eee'))
            self._emit(PText(self.L - 8, py + 3.5, fy(y), anchor='end',
                             fill='#444'))

    # -- marks ----------------------------------------------------------

    def line(self, points: Sequence[Tuple[float, float]], color: str,
             label: str = '', dashed: bool = False, mark: str = 'circle',
             tips: Optional[Sequence[str]] = None) -> None:
        """A polyline through the points, with a mark on each.

        A `None` in either coordinate breaks the line: see the module
        docstring on why a gap is left open.
        """
        run: List[Tuple[float, float]] = []
        for i, (x, y) in enumerate(points):
            if x is None or y is None:
                if len(run) > 1:
                    self._polyline(run, color, dashed)
                run = []
                continue
            run.append((self.px(x), self.py(y)))
            tip = tips[i] if tips and i < len(tips) else f'{_fmt(x)}, {_fmt(y)}'
            self._mark(self.px(x), self.py(y), color, mark, tip)
        if len(run) > 1:
            self._polyline(run, color, dashed)
        if label:
            self.legend.append((label, color))

    def _polyline(self, run: Sequence[Tuple[float, float]], color: str,
                  dashed: bool) -> None:
        self._emit(PPoly(list(run), color, 1.8, '5 4' if dashed else ''))

    def _mark(self, px: float, py: float, color: str, kind: str,
              tip: str) -> None:
        if kind == 'square':
            self._emit(PRect(px - 3.2, py - 3.2, 6.4, 6.4, color, tip=tip))
        elif kind == 'triangle':
            self._emit(PPolygon([(px, py - 4), (px - 3.7, py + 2.6),
                                 (px + 3.7, py + 2.6)], color, tip))
        elif kind == 'diamond':
            self._emit(PPolygon([(px, py - 4.2), (px + 4.2, py),
                                 (px, py + 4.2), (px - 4.2, py)], color, tip))
        else:
            self._emit(PCircle(px, py, 3.4, color, tip=tip))

    def bar(self, x: float, w: float, y: float, color: str, tip: str,
            y_base: Optional[float] = None) -> None:
        base = self.py(self.y0 if y_base is None else y_base)
        top = self.py(y)
        self._emit(PRect(x, min(top, base), w, abs(base - top), color,
                         opacity=0.85, tip=tip))

    def rule(self, y: float, text: str, color: str = '#999') -> None:
        py = self.py(y)
        self._emit(PLine(self.L, py, self.W - self.R, py, color, dash='4 3'))
        self._emit(PText(self.L + 6, py - 4, text, fill=color))

    def text(self, x: float, y: float, s: str, anchor: str = 'start',
             fill: str = '#444', size: int = 11) -> None:
        self._emit(PText(x, y, s, anchor=anchor, fill=fill, size=size))

    # -- the three shapes the views used to spell in SVG themselves ------

    def segment(self, x1: float, y1: float, x2: float, y2: float,
                color: str, width: float = 1.0, opacity: float = 1.0,
                tip: str = '', dashed: bool = False) -> None:
        """A straight line between two points *already in pixels*: a
        whisker, a ridge, an arrow from one intensity to another."""
        self._emit(PLine(x1, y1, x2, y2, color, width,
                         '4 3' if dashed else '', opacity, tip))

    def path(self, points: Sequence[Tuple[float, float]], color: str,
             width: float = 2.0) -> None:
        """A polyline in pixels -- the roof, which is a function of the
        axes rather than of the data."""
        self._emit(PPoly(list(points), color, width))

    def ring(self, px: float, py: float, r: float, color: str,
             tip: str = '') -> None:
        """A hollow mark, for the end of a pair that the filled one
        begins."""
        self._emit(PCircle(px, py, r, '#ffffff', stroke=color, tip=tip))

    # -- output ---------------------------------------------------------

    def _chrome(self) -> List[Prim]:
        """The frame, the labels and the legend, as primitives.

        Built here rather than written into the SVG at the end, because the
        PDF needs the same furniture and a second spelling of it is a second
        thing to keep in step.
        """
        out: List[Prim] = [PRect(0, 0, self.W, self.H, '#ffffff')]
        out.append(PText(self.L, 22, self.title, fill='#111', size=14))
        out.append(PLine(self.L, self.H - self.B, self.W - self.R,
                         self.H - self.B, '#333'))
        out.append(PLine(self.L, self.T, self.L, self.H - self.B, '#333'))
        out.append(PText((self.L + self.W - self.R) / 2, self.H - self.B + 34,
                         self.xlabel, anchor='middle', fill='#222'))
        mid = (self.T + self.H - self.B) / 2
        out.append(PText(16, mid, self.ylabel, anchor='middle', fill='#222',
                         rotate=-90))
        fits = max(int((self.H - self.T - 24) // 14), 1)
        entries = list(self.legend)
        more = ''
        if len(entries) > fits:
            more = f'+{len(entries) - fits + 1} more, hover the marks'
            entries = entries[:fits - 1]
        x = self.W - self.legend_width + 8 if self.legend_width else self.L + 8
        for i, (label, color) in enumerate(entries):
            y = self.T + 10 + i * 14
            out.append(PRect(x, y - 8, 8, 8, color))
            out.append(PText(x + 12, y, label[:28], fill='#333', size=10))
        if more:
            out.append(PText(x, self.T + 10 + len(entries) * 14, more,
                             fill='#888', size=10))
        for i, line in enumerate(self._note_lines):
            out.append(PText(
                self.L, self.H - 10 - 12 * (len(self._note_lines) - 1 - i),
                line, fill='#777', size=10))
        return out

    def render(self) -> str:
        """The chart as SVG: the background and title first, the data, then
        the furniture, so that an axis label is never covered by a mark.

        The canvas registers itself as the one most recently rendered, so
        that `build_charts` can collect a PDF page per view without every
        view changing what it returns.
        """
        global _LAST_CANVAS
        _LAST_CANVAS = self
        chrome = self._chrome()
        head, tail = chrome[:2], chrome[2:]
        return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{self.W}" \
height="{self.H}" viewBox="0 0 {self.W} {self.H}" font-family="sans-serif" \
font-size="11">
{chr(10).join(p.to_svg() for p in head)}
{chr(10).join(self.body)}
{chr(10).join(p.to_svg() for p in tail)}
</svg>
'''

    def to_page(self, title: str = '') -> Page:
        """The same chart as one page of a PDF (`pdfout.write_pdf`)."""
        chrome = self._chrome()
        return Page(self.W, self.H,
                    list(chrome[:2]) + list(self.prims) + list(chrome[2:]),
                    title or self.title)


# ----------------------------------------------------------------------
# Series
# ----------------------------------------------------------------------

#: What a series is keyed by beyond the workload, decided once per file from
#: what actually varies in it (`distinguish`).  A module global because
#: `series_key` is called from a dozen places that have one row and no view of
#: the run, and threading the set through all of them would say the same thing
#: twelve times.
_EXTRA_KEYS: Tuple[str, ...] = ()


def distinguish(rows: Sequence[Dict]) -> None:
    """Fix the fields a curve is separated by, for this file.

    The configuration always separates.  The *target* and the *datatype* do so
    only where the rows disagree about them -- a run over two devices used to
    produce one curve per workload with both devices' points on it, sorted by
    batch, so the line zig-zagged between an RTX 5070 and an 890M and read as
    a scaling collapse.  Where only one target is present the key stays short,
    because a legend repeating the same arch on every entry is noise.
    """
    global _EXTRA_KEYS
    extra = []
    if len({(r.get('backend'), r.get('arch')) for r in rows}) > 1:
        extra.append('target')
    if len({r.get('datatype') for r in rows}) > 1:
        extra.append('datatype')
    _EXTRA_KEYS = tuple(extra)


def series_key(row: Dict) -> str:
    """One curve per workload, configuration, and whatever else varies -- see
    the docstring on why they are not averaged together."""
    parts = []
    config = row.get('config') or ''
    if config and config != 'default':
        parts.append(config)
    if 'target' in _EXTRA_KEYS:
        parts.append(f"{row.get('backend')}-{row.get('arch')}")
    if 'datatype' in _EXTRA_KEYS:
        parts.append(str(row.get('datatype', '')).lower())
    return f"{row['workload']}" + (f' [{" ".join(parts)}]' if parts else '')


def by_series(rows: Sequence[Dict]) -> Dict[str, List[Dict]]:
    out: Dict[str, List[Dict]] = {}
    for row in rows:
        out.setdefault(series_key(row), []).append(row)
    for rows_ in out.values():
        rows_.sort(key=lambda r: r.get('batch') or 0)
    return out


#: Which of the three flop counts a rate and an intensity are against
#: (`README.md`, "Three flop counts").  A module global, set once per
#: invocation, because `rate` is called from a dozen places that have one
#: row and no view of the run -- the same reason `_EXTRA_KEYS` is one.
#:
#: `executed` is the default and the roofline is why.  The question a roof
#: asks is whether the machine's arithmetic is the limit, and the machine
#: issues what the *built kernel* performs: a destination twenty rows wide
#: computes on thirty-two lanes, and the multiply-add that fills the other
#: twelve is issued, retired and counted by the profiler.  Against the
#: dense figure the same point sits 2.2x too low on `gpu_localFluxAll`
#: (90 576 executed against 40 680 dense), which is the difference between
#: a kernel at a third of its roof and one at two thirds.
_FLOP_BASIS = 'executed'

#: The row field each basis reads, and what it means.
FLOP_FIELD = {'executed': 'executed_flops',
              'dense': 'flops',
              'nonzero': 'nonzero_flops'}

FLOP_LABEL = {'executed': 'issued by the kernel',
              'dense': 'a dense implementation performs',
              'nonzero': 'the operation needs'}


def set_flop_basis(name: str) -> None:
    if name not in FLOP_FIELD:
        raise KeyError(f'{name}: the flop basis is one of '
                       f'{", ".join(sorted(FLOP_FIELD))}')
    global _FLOP_BASIS
    _FLOP_BASIS = name


def flop_basis() -> str:
    return _FLOP_BASIS


def flops_of(row: Dict, basis: Optional[str] = None) -> Optional[float]:
    """The flop count on the chosen basis, or the dense one where the
    chosen one was not recorded.

    Falling back rather than dropping the row: `executed_flops` comes from
    the analysis pass and is `None` where it could not count a body, and a
    kernel vanishing from a roofline because of that is worse than a
    kernel drawn on the count beside it.  The views that care say how many
    fell back (`fallback_note`).
    """
    value = row.get(FLOP_FIELD[basis or _FLOP_BASIS])
    if value:
        return float(value)
    return float(row['flops']) if row.get('flops') else None


def on_dense(rows: Sequence[Dict], basis: Optional[str] = None) -> int:
    """How many of `rows` had to fall back to the dense count."""
    field = FLOP_FIELD[basis or _FLOP_BASIS]
    return sum(1 for r in rows if not r.get(field) and r.get('flops'))


def fallback_note(rows: Sequence[Dict], basis: Optional[str] = None) -> str:
    n = on_dense(rows, basis)
    return (f' {n} of {len(rows)} points carry no '
            f'{FLOP_FIELD[basis or _FLOP_BASIS]} and are drawn on the dense '
            f'count.' if n else '')


def intensity_of(row: Dict, basis: Optional[str] = None) -> Optional[float]:
    """Flops per compulsory byte, on the same basis as the rate.

    The two have to agree: a point placed at the dense intensity and drawn
    at the issued rate is on neither roofline.  `run.py` stores the dense
    pair as `intensity`, which is what this reproduces for `dense`.
    """
    f = flops_of(row, basis)
    b = row.get('bytes')
    return f / b if f and b else None


def rate(row: Dict, what: str) -> Optional[float]:
    """The quantity a view is plotting, from whichever field carries it.

    `gbytes` is written by `run.py`; recomputing it from `bytes` and the
    clock is the same arithmetic and disagrees when the run used a
    different clock, so the stored figure wins where it exists.  The
    stored `gflops` is against the *dense* count, so it wins on that basis
    and on no other -- there is nothing stored for the other two, and
    taking the dense figure for them would label an axis with a number
    from a different numerator.
    """
    nanos = row.get('event_ns') or row.get('wall_ns')
    if what == 'gflops':
        if _FLOP_BASIS == 'dense' and row.get('gflops'):
            return float(row['gflops'])
        f = flops_of(row)
        return f / nanos if (nanos and f) else None
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


def rate_label(what: str) -> str:
    """The axis name, saying which flop count it is against -- a GFLOP/s
    that does not say is three different numbers."""
    if what == 'gflops':
        return f'GFLOP/s ({_FLOP_BASIS})'
    return RATE_LABEL[what]


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
    c = Canvas(f'{rate_label(what)} against launch size — {run.label}',
               'elements per launch', rate_label(what), xlog=True,
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
                f'{rate(r, what) or 0:.2f} {rate_label(what)}' for r in rows]
        c.line(pts, PALETTE[i % len(PALETTE)], name,
               mark=series_style(i)[1], tips=tips)
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
        c.segment(x, c.py(lo), x, c.py(hi), color, width=7, opacity=0.45,
                  tip=f'{series_key(r)} @ {_fmt(r.get("batch") or 0)}: '
                      f'min {lo:.0f}, p50 {mid:.0f}, p90 {hi:.0f} ns '
                      f'({(hi - lo) / mid * 100:.1f} % spread)')
        c.segment(x - 6, c.py(mid), x + 6, c.py(mid), color, width=2)
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
               + f' — {rate_label(what)}',
               '', f'ratio to {runs[0].label}',
               width=max(760, 54 * len(common) + 180),
               tilted_labels=True, legend_width=150,
               note=f'{runs[0].device}; above 1 is faster than '
                    f'{runs[0].label}. '
                    + ('Lower ' + rate_label(what) + ' is better, so the bar '
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
    c.path([(c.px(x), c.py(roof.bound(x)))
            for x in (c.x0, roof.ridge, c.x1)], '#222')
    c.segment(c.px(roof.ridge), c.T, c.px(roof.ridge), c.H - c.B, '#bbb',
              dashed=True)
    c.text(c.px(roof.ridge) + 4, c.T + 12,
           f'ridge {roof.ridge:.1f} flop/byte', fill='#888')


def view_roofline(run: Run, roof: Roof) -> Optional[str]:
    """Every measured point under the two ceilings, at compulsory intensity."""
    rows = [r for r in run.rows if intensity_of(r) and rate(r, 'gflops')]
    if not rows:
        return None
    c = Canvas(f'Roofline — {run.label}', 'arithmetic intensity '
               f'(flop/byte, compulsory, {flop_basis()})',
               f'achieved FLOP/s ({flop_basis()})',
               xlog=True, ylog=True, legend_width=190,
               note=f'{run.device}; roofs {roof.source}. Flops are what '
                    f'{FLOP_LABEL[flop_basis()]}. Compulsory intensity is '
                    f'what the description implies, not what the kernel '
                    f'paid: see the measured roofline where a profile '
                    f'exists.' + fallback_note(rows))
    xs = [intensity_of(r) for r in rows]
    ys = [rate(r, 'gflops') * 1e9 for r in rows]
    _roof_frame(c, roof, xs, ys)
    groups = by_series(rows)
    for i, (name, rs) in enumerate(sorted(groups.items())):
        color = PALETTE[i % len(PALETTE)]
        for r in rs:
            achieved = rate(r, 'gflops') * 1e9
            frac = achieved / roof.bound(intensity_of(r))
            c._mark(c.px(intensity_of(r)), c.py(achieved), color,
                    series_style(i)[1],
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
    xs = [intensity_of(r) for rs in groups.values() for r in rs
          if intensity_of(r)]
    ys = [rate(r, 'gflops') * 1e9 for rs in groups.values() for r in rs
          if rate(r, 'gflops')]
    if not xs:
        return None
    _roof_frame(c, roof, xs, ys)
    for i, (name, rs) in enumerate(sorted(groups.items())):
        pts = [(intensity_of(r),
                (rate(r, 'gflops') or 0) * 1e9 or None) for r in rs]
        tips = [f'{name} @ {_fmt(r.get("batch") or 0)}: '
                f'{(rate(r, "gflops") or 0):.1f} GFLOP/s' for r in rs]
        c.line(pts, PALETTE[i % len(PALETTE)], name,
               mark=series_style(i)[1], tips=tips)
    return c.render()


def view_efficiency(run: Run, roof: Roof) -> Optional[str]:
    """Fraction of the roof against launch size.

    The same data as `trajectory` with the roof divided out, which is the
    form that compares kernels of different intensity against each other.
    """
    groups = {k: v for k, v in by_series(run.rows).items()
              if len(v) > 1 and all(intensity_of(r) for r in v)}
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
                fr.append(g * 1e9 / roof.bound(intensity_of(r)) * 100)
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
            v = g * 1e9 / roof.bound(intensity_of(r)) * 100 if g else None
            pts.append((r.get('batch'), v))
            tips.append(f'{name} @ {_fmt(r.get("batch") or 0)}: '
                        f'{v or 0:.1f} % of roof')
        c.line(pts, PALETTE[i % len(PALETTE)], name,
               mark=series_style(i)[1], tips=tips)
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
                series_style(i)[1],
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
        compulsory, f = intensity_of(row), flops_of(row)
        if not rows or not g or not compulsory or not f:
            continue
        measured = _measured_bytes(hottest(rows))
        if not measured:
            continue
        # Both intensities on the chosen basis: the move the arrow draws is
        # the *bytes* changing, and a numerator that changed with them
        # would hide it.
        arrows.append((series_key(row), compulsory, f / measured, g * 1e9))
    if not arrows:
        return None
    c = Canvas(f'Compulsory against measured intensity — {run.label}',
               f'arithmetic intensity (flop/byte, {flop_basis()})',
               f'achieved FLOP/s ({flop_basis()})',
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
        c.segment(c.px(compulsory), y, c.px(measured), y, color,
                  width=1.2, opacity=0.7)
        c.ring(c.px(compulsory), y, 3.6, color,
               f'{name}: compulsory {compulsory:.2f} flop/byte')
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
#: One PDF page per view that had data, filled by `build_charts`.
PAGES: List[Page] = []

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
    # The views return SVG, which is what the index page wants; the PDF
    # wants the canvas they drew on.  `Canvas.render` records the last one
    # per view (`_LAST_PAGE`) rather than every view changing its return
    # type, which would touch ten functions to serve one output.
    charts: List[Tuple[str, str, str]] = []
    skipped: List[str] = []
    PAGES.clear()

    def add(name: str, svg: Optional[str], why: str = '') -> None:
        if svg:
            charts.append((name, QUESTIONS.get(name, ''), svg))
            if _LAST_CANVAS is not None:
                PAGES.append(_LAST_CANVAS.to_page(name))
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
    ap.add_argument('--flops', default='executed', choices=sorted(FLOP_FIELD),
                    help='which of the three flop counts a FLOP/s and an '
                         'arithmetic intensity are against: what the kernel '
                         'issues (default, and what a profiler reports), '
                         'what a dense implementation performs, or what the '
                         'operation needs')
    ap.add_argument('--pdf', action='store_true',
                    help='also write every chart as a page of report.pdf, '
                         'and one PDF per chart beside its SVG')
    args = ap.parse_args()

    set_flop_basis(args.flops)
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

    # Over every run at once: a comparison of two files must key its curves
    # the same way in both, or `compare` would pair a CUDA series with a HIP
    # one of the same name.
    distinguish([r for run in runs for r in run.rows])

    profile = None
    if args.profile:
        try:
            profile = load_profile(args.profile)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f'{args.profile}: {exc}', file=sys.stderr)

    roof = roof_from(args, runs)
    charts, skipped = build_charts(runs, roof, profile, args.batch)

    args.out.mkdir(parents=True, exist_ok=True)
    title = (f'{runs[0].label} — {runs[0].device}'
             + (f' against {", ".join(r.label for r in runs[1:])}'
                if len(runs) > 1 else ''))
    for name, _, svg in charts:
        (args.out / f'{name}.svg').write_text(svg)
    if args.pdf and PAGES:
        write_pdf(PAGES, args.out / 'report.pdf', title)
        for page in PAGES:
            write_pdf([page], args.out / f'{page.title}.pdf', page.title)
    (args.out / 'index.html').write_text(index_html(charts, title))

    for name, _, _ in charts:
        print(f'written: {args.out / (name + ".svg")}')
    print(f'written: {args.out / "index.html"}')
    if args.pdf and PAGES:
        print(f'written: {args.out / "report.pdf"} ({len(PAGES)} pages)')
    for note in skipped:
        print(f'skipped {note}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
