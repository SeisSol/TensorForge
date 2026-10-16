# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""The same charts as a PDF, written by hand like the SVG beside it.

`plot.py` explains why its charts are hand-written SVG: this directory runs
on a login node somebody else administers, where `pip install` is not a
thing that reliably happens.  A PDF asked for the same answer and gets the
same one.  A vector PDF of lines, rectangles and Helvetica text is a few
hundred bytes of operators and an object table; what a library would add
here is a dependency, not a capability.

So `Canvas` records what it drew as *primitives* and renders them twice:
once as SVG, once as PDF content streams.  Neither is converted from the
other -- an SVG-to-PDF converter would be a parser for output this module
already has in structured form, and it would break on the first primitive
somebody adds without telling it.

## What a PDF buys over the SVG

The SVG is the better thing to read at the machine: it carries `<title>`
tooltips with the exact figures, and a browser is already there.  The PDF
is the better thing to *send*: one file holds every chart of an
evaluation in order, it prints at any size, and it goes into a paper or an
issue without a screenshot.  Both are written, and neither is the primary.

## What is deliberately not here

No raster fallback, no font embedding and no Unicode beyond Latin-1.  The
charts carry axis numbers, kernel names and architecture names, which are
ASCII; a label that is not gets transliterated rather than silently
dropping a glyph.  Embedding a font to render a kernel name nobody has
written yet is a cost with no case behind it.
"""

from __future__ import annotations

import html
import math
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

__all__ = ['Prim', 'PLine', 'PPoly', 'PRect', 'PCircle', 'PPolygon', 'PText',
           'Page', 'write_pdf', 'table_page']


# ----------------------------------------------------------------------
# Colour
# ----------------------------------------------------------------------

def _rgb(color: str) -> Tuple[float, float, float]:
    """`#rrggbb` (or a few names) as PDF's 0..1 triple."""
    named = {'white': '#ffffff', 'black': '#000000', 'none': '#ffffff'}
    c = named.get(color, color).lstrip('#')
    if len(c) == 3:
        c = ''.join(ch * 2 for ch in c)
    if len(c) != 6:
        return (0.0, 0.0, 0.0)
    try:
        return tuple(int(c[i:i + 2], 16) / 255.0 for i in (0, 2, 4))  # type: ignore
    except ValueError:
        return (0.0, 0.0, 0.0)


def _num(v: float) -> str:
    """A coordinate, short: PDF has no use for the sixteenth decimal and a
    content stream is read by people often enough to keep it narrow."""
    return f'{v:.2f}'.rstrip('0').rstrip('.') or '0'


# ----------------------------------------------------------------------
# Primitives
# ----------------------------------------------------------------------
#
# Each one knows both spellings of itself.  The SVG is what `Canvas.body`
# has always held, byte for byte, so that the tests that count `<polyline`
# in it keep saying what they said.

@dataclass
class Prim:
    """One drawn thing, in SVG user space (y down, origin top left)."""

    def to_svg(self) -> str:                      # pragma: no cover - abstract
        raise NotImplementedError

    def to_pdf(self, h: float) -> str:            # pragma: no cover - abstract
        raise NotImplementedError

    @staticmethod
    def _title(tip: str) -> str:
        return f'<title>{html.escape(tip)}</title>' if tip else ''

    @staticmethod
    def _stroke(color: str, width: float, dash: str, opacity: float,
                h: float) -> str:
        r, g, b = _rgb(color)
        out = [f'{_num(r)} {_num(g)} {_num(b)} RG', f'{_num(width)} w']
        out.append(f'[{dash}] 0 d' if dash else '[] 0 d')
        return ' '.join(out)

    @staticmethod
    def _fill(color: str) -> str:
        r, g, b = _rgb(color)
        return f'{_num(r)} {_num(g)} {_num(b)} rg'


@dataclass
class PLine(Prim):
    x1: float
    y1: float
    x2: float
    y2: float
    color: str = '#000'
    width: float = 1.0
    dash: str = ''
    opacity: float = 1.0
    tip: str = ''

    def to_svg(self) -> str:
        d = f' stroke-dasharray="{self.dash}"' if self.dash else ''
        o = f' opacity="{self.opacity}"' if self.opacity != 1.0 else ''
        w = f' stroke-width="{self.width}"' if self.width != 1.0 else ''
        body = self._title(self.tip)
        return (f'<line x1="{self.x1:.1f}" y1="{self.y1:.1f}" '
                f'x2="{self.x2:.1f}" y2="{self.y2:.1f}" '
                f'stroke="{self.color}"{w}{d}{o}>{body}</line>'
                if body else
                f'<line x1="{self.x1:.1f}" y1="{self.y1:.1f}" '
                f'x2="{self.x2:.1f}" y2="{self.y2:.1f}" '
                f'stroke="{self.color}"{w}{d}{o}/>')

    def to_pdf(self, h: float) -> str:
        return (f'q {_alpha(self.opacity)}'
                f'{self._stroke(self.color, self.width, _dash(self.dash), self.opacity, h)} '
                f'{_num(self.x1)} {_num(h - self.y1)} m '
                f'{_num(self.x2)} {_num(h - self.y2)} l S Q')


@dataclass
class PPoly(Prim):
    """An open polyline through points already in user space."""
    points: List[Tuple[float, float]] = field(default_factory=list)
    color: str = '#000'
    width: float = 1.8
    dash: str = ''

    def to_svg(self) -> str:
        pts = ' '.join(f'{x:.1f},{y:.1f}' for x, y in self.points)
        d = f' stroke-dasharray="{self.dash}"' if self.dash else ''
        return (f'<polyline points="{pts}" fill="none" stroke="{self.color}" '
                f'stroke-width="{self.width}" stroke-linejoin="round"{d}/>')

    def to_pdf(self, h: float) -> str:
        if not self.points:
            return ''
        ops = [f'q {self._stroke(self.color, self.width, _dash(self.dash), 1.0, h)} 1 j']
        for i, (x, y) in enumerate(self.points):
            ops.append(f'{_num(x)} {_num(h - y)} {"m" if i == 0 else "l"}')
        ops.append('S Q')
        return ' '.join(ops)


@dataclass
class PRect(Prim):
    x: float
    y: float
    w: float
    h: float
    fill: str = '#000'
    opacity: float = 1.0
    tip: str = ''

    def to_svg(self) -> str:
        o = f' opacity="{self.opacity}"' if self.opacity != 1.0 else ''
        body = self._title(self.tip)
        return (f'<rect x="{self.x:.1f}" y="{self.y:.1f}" width="{self.w:.1f}" '
                f'height="{self.h:.1f}" fill="{self.fill}"{o}>{body}</rect>'
                if body else
                f'<rect x="{self.x:.1f}" y="{self.y:.1f}" width="{self.w:.1f}" '
                f'height="{self.h:.1f}" fill="{self.fill}"{o}/>')

    def to_pdf(self, h: float) -> str:
        return (f'q {_alpha(self.opacity)}{self._fill(self.fill)} '
                f'{_num(self.x)} {_num(h - self.y - self.h)} '
                f'{_num(self.w)} {_num(self.h)} re f Q')


@dataclass
class PCircle(Prim):
    cx: float
    cy: float
    r: float
    fill: str = '#000'
    stroke: str = ''
    width: float = 1.6
    tip: str = ''

    def to_svg(self) -> str:
        s = (f' stroke="{self.stroke}" stroke-width="{self.width}"'
             if self.stroke else '')
        body = self._title(self.tip)
        return (f'<circle cx="{self.cx:.1f}" cy="{self.cy:.1f}" '
                f'r="{self.r:.1f}" fill="{self.fill}"{s}>{body}</circle>'
                if body else
                f'<circle cx="{self.cx:.1f}" cy="{self.cy:.1f}" '
                f'r="{self.r:.1f}" fill="{self.fill}"{s}/>')

    def to_pdf(self, h: float) -> str:
        # Four Bézier arcs: the magic constant is the one that puts a cubic
        # within half a thousandth of a quarter circle.
        k = self.r * 0.5523
        x, y, r = self.cx, h - self.cy, self.r
        path = (f'{_num(x - r)} {_num(y)} m '
                f'{_num(x - r)} {_num(y + k)} {_num(x - k)} {_num(y + r)} '
                f'{_num(x)} {_num(y + r)} c '
                f'{_num(x + k)} {_num(y + r)} {_num(x + r)} {_num(y + k)} '
                f'{_num(x + r)} {_num(y)} c '
                f'{_num(x + r)} {_num(y - k)} {_num(x + k)} {_num(y - r)} '
                f'{_num(x)} {_num(y - r)} c '
                f'{_num(x - k)} {_num(y - r)} {_num(x - r)} {_num(y - k)} '
                f'{_num(x - r)} {_num(y)} c')
        if self.stroke:
            return (f'q {self._fill(self.fill)} '
                    f'{self._stroke(self.stroke, self.width, "", 1.0, h)} '
                    f'{path} B Q')
        return f'q {self._fill(self.fill)} {path} f Q'


@dataclass
class PPolygon(Prim):
    points: List[Tuple[float, float]] = field(default_factory=list)
    fill: str = '#000'
    tip: str = ''

    def to_svg(self) -> str:
        pts = ' '.join(f'{x:.1f},{y:.1f}' for x, y in self.points)
        body = self._title(self.tip)
        return (f'<polygon points="{pts}" fill="{self.fill}">{body}</polygon>'
                if body else f'<polygon points="{pts}" fill="{self.fill}"/>')

    def to_pdf(self, h: float) -> str:
        if not self.points:
            return ''
        ops = [f'q {self._fill(self.fill)}']
        for i, (x, y) in enumerate(self.points):
            ops.append(f'{_num(x)} {_num(h - y)} {"m" if i == 0 else "l"}')
        ops.append('h f Q')
        return ' '.join(ops)


@dataclass
class PText(Prim):
    x: float
    y: float
    s: str
    anchor: str = 'start'
    fill: str = '#444'
    size: float = 11
    rotate: float = 0.0

    def to_svg(self) -> str:
        t = (f' transform="rotate({self.rotate:g} {self.x:.1f} {self.y:.1f})"'
             if self.rotate else '')
        return (f'<text x="{self.x:.1f}" y="{self.y:.1f}" '
                f'text-anchor="{self.anchor}" fill="{self.fill}" '
                f'font-size="{self.size:g}"{t}>{html.escape(self.s)}</text>')

    def to_pdf(self, h: float) -> str:
        # Helvetica's average advance is about 0.5 em over mixed text, which
        # is what the anchor correction needs and all it needs: the glyph
        # widths would buy a pixel of placement and cost a width table.
        width = len(self.s.translate(_TRANSLITERATE)) * self.size * 0.5
        dx = {'middle': -width / 2, 'end': -width}.get(self.anchor, 0.0)
        r, g, b = _rgb(self.fill)
        y = h - self.y
        if self.rotate:
            a = math.radians(self.rotate)
            ca, sa = math.cos(a), math.sin(a)
            # Rotate about the anchor, which SVG does in its own y-down
            # frame; in PDF the sign of the shear flips with the axis.
            ox = self.x + dx * ca
            oy = y - dx * sa
            tm = (f'{_num(ca)} {_num(-sa)} {_num(sa)} {_num(ca)} '
                  f'{_num(ox)} {_num(oy)} Tm')
        else:
            tm = f'1 0 0 1 {_num(self.x + dx)} {_num(y)} Tm'
        return (f'BT {_num(r)} {_num(g)} {_num(b)} rg /F1 {_num(self.size)} Tf '
                f'{tm} ({_escape(self.s)}) Tj ET')


#: Typographic characters the charts use that WinAnsi has no slot for.
#: Spelled out rather than replaced with `?`, which is what the encoder
#: does on its own and which turned every em dash in a rule label into a
#: question mark.
_TRANSLITERATE = str.maketrans({
    '\u2014': '--', '\u2013': '-', '\u2212': '-', '\u2018': "'",
    '\u2019': "'", '\u201c': '"', '\u201d': '"', '\u2026': '...',
    '\u00a0': ' ', '\u2032': "'", '\u2033': '"', '\u2248': '~',
    '\u2264': '<=', '\u2265': '>=', '\u2192': '->',
})


def _escape(s: str) -> str:
    out = s.translate(_TRANSLITERATE).encode('latin-1', 'replace')
    return (out.decode('latin-1').replace('\\', r'\\')
            .replace('(', r'\(').replace(')', r'\)'))


def _dash(svg_dash: str) -> str:
    """`stroke-dasharray="5 4"` as PDF's dash array contents."""
    return ' '.join(svg_dash.replace(',', ' ').split())


def _alpha(opacity: float) -> str:
    return f'/GS{int(round(opacity * 100))} gs ' if opacity != 1.0 else ''


# ----------------------------------------------------------------------
# The document
# ----------------------------------------------------------------------

@dataclass
class Page:
    """One chart, as wide and tall as it was drawn."""
    width: float
    height: float
    prims: Sequence[Prim]
    title: str = ''


def write_pdf(pages: Sequence[Page], path: Path,
              title: str = 'TensorForge benchmark') -> None:
    """Every page into one file, in the order given.

    Each page keeps the pixel size the chart was laid out at, so a wide
    comparison stays wide instead of being squeezed onto a portrait sheet.
    A reader who wants paper scales it there; a reader who wants the
    figures reads the SVG.
    """
    if not pages:
        raise ValueError('a PDF needs at least one page')
    alphas = sorted({p.opacity for page in pages for p in page.prims
                     if isinstance(p, (PLine, PRect)) and p.opacity != 1.0})

    objects: List[bytes] = []

    def add(body: bytes) -> int:
        objects.append(body)
        return len(objects)

    font = add(b'<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica '
               b'/Encoding /WinAnsiEncoding >>')
    gs_names = {}
    gs_entries = []
    for a in alphas:
        name = f'GS{int(round(a * 100))}'
        num = add(f'<< /Type /ExtGState /CA {a:.2f} /ca {a:.2f} >>'
                  .encode('latin-1'))
        gs_names[a] = num
        gs_entries.append(f'/{name} {num} 0 R')
    gs_dict = f'/ExtGState << {" ".join(gs_entries)} >>' if gs_entries else ''

    pages_num = len(objects) + 1 + 2 * len(pages)   # reserved below
    page_nums: List[int] = []
    for page in pages:
        stream = '\n'.join(
            op for op in (p.to_pdf(page.height) for p in page.prims) if op)
        data = zlib.compress(stream.encode('latin-1', 'replace'))
        content = add(b'<< /Length ' + str(len(data)).encode()
                      + b' /Filter /FlateDecode >>\nstream\n' + data
                      + b'\nendstream')
        page_num = add(
            f'<< /Type /Page /Parent {pages_num} 0 R '
            f'/MediaBox [0 0 {_num(page.width)} {_num(page.height)}] '
            f'/Resources << /Font << /F1 {font} 0 R >> {gs_dict} >> '
            f'/Contents {content} 0 R >>'.encode('latin-1'))
        page_nums.append(page_num)
    kids = ' '.join(f'{n} 0 R' for n in page_nums)
    tree = add(f'<< /Type /Pages /Kids [{kids}] /Count {len(page_nums)} >>'
               .encode('latin-1'))
    # `/Parent` was written before the tree object existed; it is the next
    # number after the last page, which is what `pages_num` reserved.
    assert tree == pages_num, (tree, pages_num)
    info = add(f'<< /Title ({_escape(title)}) /Producer '
               f'(TensorForge tools/bench) >>'.encode('latin-1'))
    catalog = add(f'<< /Type /Catalog /Pages {tree} 0 R >>'.encode('latin-1'))

    out = bytearray(b'%PDF-1.4\n%\xe2\xe3\xcf\xd3\n')
    offsets = [0]
    for i, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f'{i} 0 obj\n'.encode() + body + b'\nendobj\n'
    start = len(out)
    out += f'xref\n0 {len(objects) + 1}\n'.encode()
    out += b'0000000000 65535 f \n'
    for off in offsets[1:]:
        out += f'{off:010d} 00000 n \n'.encode()
    out += (f'trailer\n<< /Size {len(objects) + 1} /Root {catalog} 0 R '
            f'/Info {info} 0 R >>\nstartxref\n{start}\n%%EOF\n').encode()
    path.write_bytes(bytes(out))


def table_page(title: str, headers: Sequence[str],
               rows: Sequence[Sequence[str]], note: str = '',
               ) -> Tuple[Page, str]:
    """A reference table, as a PDF page and as HTML.

    Some answers are a table.  Six unrelated quantities about four machines
    is something a reader looks a value up in, and a chart of it would be
    six bar charts sharing an axis they have no business sharing -- so this
    exists beside the charts rather than being drawn as one.

    The two spellings come out of one call for the same reason the chart
    primitives do: a table written twice is a table that disagrees with
    itself on the third edit.
    """
    pad, lead, head_h = 10.0, 15.0, 26.0
    size = 9.5
    widths = [max(len(str(h)), *(len(str(r[i])) for r in rows)) if rows
              else len(str(h)) for i, h in enumerate(headers)]
    # Helvetica averages about half an em over mixed text; the column is
    # that plus a gutter, which is close enough that nothing collides.
    cols = [w * size * 0.52 + 14 for w in widths]
    width = sum(cols) + 2 * pad
    # The note is wrapped to the sheet rather than the sheet widened to the
    # note: a table two columns wide with a paragraph under it would be a
    # page of mostly margin.
    per_line = max(int((width - 2 * pad) / (8.5 * 0.5)), 40)
    note_lines = _wrap(note, per_line) if note else []
    height = (head_h + 24 + lead * (len(rows) + 1)
              + (12 * len(note_lines) + 10 if note_lines else 0) + pad)

    prims: List[Prim] = [PRect(0, 0, width, height, '#ffffff')]
    prims.append(PText(pad, 20, title, fill='#111', size=14))
    y = head_h + 16
    x = pad
    for i, h in enumerate(headers):
        prims.append(PText(x, y, str(h), fill='#111', size=size))
        x += cols[i]
    prims.append(PLine(pad, y + 5, width - pad, y + 5, '#999'))
    for r, row in enumerate(rows):
        y += lead
        if r % 2:
            prims.append(PRect(pad - 3, y - lead + 6, width - 2 * pad + 6,
                               lead, '#f4f4f1'))
        x = pad
        for i, cell in enumerate(row):
            prims.append(PText(x, y + 5, str(cell),
                               fill='#111' if i == 0 else '#444', size=size))
            x += cols[i]
    for i, line in enumerate(note_lines):
        prims.append(PText(pad, height - 10 - 12 * (len(note_lines) - 1 - i),
                           line, fill='#777', size=8.5))

    head = ''.join(f'<th>{html.escape(str(h))}</th>' for h in headers)
    body = ''.join(
        '<tr>' + ''.join(f'<td>{html.escape(str(c))}</td>' for c in row)
        + '</tr>' for row in rows)
    markup = (
        '<table style="border-collapse:collapse;font-size:.86rem">'
        '<thead style="text-align:left;border-bottom:1px solid #999">'
        f'<tr>{head}</tr></thead><tbody>{body}</tbody></table>'
        + (f'<p style="color:#777;font-size:.8rem">{html.escape(note)}</p>'
           if note else ''))
    return Page(width, height, prims, title), markup


def _wrap(text: str, width: int) -> List[str]:
    """Greedy word wrap.  No hyphenation and no justification: a note under
    a table is read once."""
    out: List[str] = []
    line = ''
    for word in text.split():
        if line and len(line) + 1 + len(word) > width:
            out.append(line)
            line = word
        else:
            line = f'{line} {word}'.strip()
    if line:
        out.append(line)
    return out
