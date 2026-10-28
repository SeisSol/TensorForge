# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""What each captured kernel computes, read off the export itself.

A table of kernels that is written by hand describes the kernels somebody
remembered, at the order they remembered them at.  An export from
`tools/seissol_export.py` carries every operation yateto handed the backend,
with the tensors' own names and index labels, so the formula is in it: a
chain `_tmp0[lp] = I[lq] AplusT[qp]`, `_tmp1[mp] = fMrT[ml] _tmp0[lp]`,
`Q[kp] += rDivM[km] _tmp1[mp]` is the one product
`Q[kp] += rDivM[km] fMrT[ml] I[lq] AplusT[qp]`, and four of them that differ
only in the face they take their operators from are one sum over the faces.

So this rebuilds the formula from the operations -- temporaries written once
and read once are put back into their reader, and a run of statements that
differ only in a family index collapses into a sum over it -- and reports it
beside what the export says about the kernel's structure and cost.  What is
written by hand is only what no export can say: which step of the scheme a
kernel is (`ROLES`), and how a tensor is spelled in a paper's notation
(`NOTATION`).

    python3 tools/bench/describe.py tools/bench/data/elastic-4-d-s1.json
    python3 tools/bench/describe.py tools/bench/data --kernels gpu_volume
    python3 tools/bench/describe.py tools/bench/data --csv kernels.csv

## What is reconstructed, and what is not

The algebra is the export's: no factor is reordered within a statement, and
an index is renamed only where a summed one would collide with a label of
the reader it is put into -- or where the reader only copies, which then
takes the writer's labels.  A temporary that is accumulated into, or read
twice, stays a statement of its own, because putting it back would compute it
twice or change what is summed.

A recursion is folded where it is regular.  An input copied into a family's
member 0 is read as that member (`dQ(0) = Q`); the kernel's outputs --
tensors nothing in it reads, like the time integral -- are written last, past
everything they commute with; and a statement or a block of statements
repeated with every family member one further each time becomes one rule,
`dQ(j+1) = sum_i kDivMT(i) dQ(j) star(i)`.  What does not fold this way stays
listed, in the export's order.

The figures beside the formula need TensorForge (`analysis.cost`) and are
left out where it cannot be imported, so the formulas can be had anywhere.
Bytes are per element and asymptotic: an operator every element shares is
read once per launch, which at the batches worth measuring is no term at
all, and counting it per element would make every small kernel look
memory-bound by a constant that is the launch's and not the kernel's.
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import json
import re
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from capture import Capture  # noqa: E402

# ----------------------------------------------------------------------
# What only a person can say
# ----------------------------------------------------------------------

#: The step of the scheme a kernel belongs to, by the kernel's family name
#: (`gpu_` and a trailing `[n]` taken off).  Kept short on purpose: a role is
#: a table cell, and what the kernel computes is the formula's job.
ROLES: Dict[str, Tuple[str, str]] = {
    'derivative': (
        'predictor',
        'ADER time derivatives by the Cauchy-Kovalevskaya procedure, '
        'accumulated into the time integral over the step'),
    'derivativeTaylorExpansion': (
        'predictor',
        'time integral of the predictor over a sub-interval, as a Taylor '
        'series in the derivatives'),
    'derivativeTaylorExpansionEla': (
        'predictor',
        'the same for the elastic part of the viscoelastic system'),
    'volume': (
        'corrector',
        'volume integral: stiffness matrices against the time-integrated '
        'DOFs and the star matrices'),
    'volumeExt': (
        'corrector',
        'volume integral into the extended DOFs of the viscoelastic system'),
    'localFluxAll': (
        'corrector',
        'local surface integral over the four faces'),
    'localFlux': (
        'corrector',
        'local surface integral over one face'),
    'localFluxExt': (
        'corrector',
        'local surface integral over one face, into the extended DOFs'),
    'fluxLocalAll': (
        'corrector',
        'local surface integral over the four faces and the anelastic '
        'update after it'),
    'local': (
        'corrector',
        'anelastic update: memory variables and their coupling back into '
        'the elastic DOFs'),
    'neighboringFlux': (
        'corrector',
        'neighbour surface integral for one face and orientation'),
    'neighborFluxExt': (
        'corrector',
        'neighbour surface integral for one face and orientation, into the '
        'extended DOFs'),
    'neighbor': (
        'corrector',
        'anelastic update after the neighbour integrals'),
    'localFluxNodal': (
        'boundary',
        'local surface integral of one face from nodal boundary values'),
    'dirichletFlux': ('boundary', 'Dirichlet boundary flux of one face'),
    'fsgFlux': ('boundary', 'free-surface-gravity boundary flux of one face'),
    'fsgKernel': ('boundary', 'free-surface-gravity boundary update'),
    'addVelocity': ('boundary', 'face velocity added for a boundary'),
    'rotateFaceDisplacement': (
        'boundary', 'face displacement rotated into the face frame'),
    'evaluateAndRotateQAtInterpolationPoints': (
        'rupture',
        'DOFs evaluated at the fault\'s points and rotated into its frame'),
    'projectToDR': (
        'rupture',
        'predictor evaluated at the fault\'s points per time point'),
    'nodalFlux': ('rupture', 'fault flux back into the element'),
    'plConvertToNodal': ('plasticity', 'stress DOFs to nodal values'),
    'plConvertToModal': ('plasticity', 'stress back to modal DOFs'),
    'accumulateIntegrals': ('time stepping', 'time integrals accumulated'),
    'stateToTransport': ('time stepping', 'copy, no arithmetic'),
    'transportToState': ('time stepping', 'copy, no arithmetic'),
}

#: Prefixes of the output projections, which are a role of their own and
#: too many to list one by one.
OUTPUT_PREFIXES = ('projectBasisToVtk', 'projectNodalToVtk')

#: How a tensor is spelled in the notation of a paper, by its name in the
#: export; `#` is where the family index goes (`_tex_name`).  Anything not
#: here is set upright under its own name, which is the honest default: it
#: is the name the generated code uses.
NOTATION: Dict[str, str] = {
    'Q': r'Q',
    'I': r'I',
    'dQ': r'\partial_t^{#} Q',
    'kDivM': r'\hat{K}^{(#)}',
    'kDivMT': r'\hat{K}^{(#)T}',
    'star': r'A^{*(#)}',
    'rDivM': r'\hat{R}^{(#)}',
    'fMrT': r'\hat{F}^{-(#)}',
    'fPrT': r'\hat{F}^{+(#)}',
    'AplusT': r'\hat{A}^{+(#)}',
    'AplusTAll': r'\hat{A}^{+(#)}',
    'AminusT': r'\hat{A}^{-(#)}',
    'power': r'\tau_{#}',
}


def family_of(workload: str) -> str:
    """`gpu_neighboringFlux[3]` -> `neighboringFlux`: what `ROLES` keys by."""
    name = re.sub(r'\[\d+\]$', '', workload)
    return name[4:] if name.startswith('gpu_') else name


def role_of(workload: str) -> Tuple[str, str]:
    fam = family_of(workload)
    if fam in ROLES:
        return ROLES[fam]
    if fam.startswith(OUTPUT_PREFIXES):
        return 'output', 'DOFs projected to output points'
    return '', ''


# ----------------------------------------------------------------------
# The statements
# ----------------------------------------------------------------------

#: A family index in a tensor's name: `kDivM(0)`, `V3mTo2n(0,0)`.
_FAMILY = re.compile(r'^(?P<base>.*?)\((?P<values>\d+(?:,\d+)*)\)$')


@dataclass(frozen=True)
class Ref:
    """A tensor as one statement reads or writes it."""
    name: str
    indices: Tuple[str, ...]

    @property
    def base(self) -> str:
        m = _FAMILY.match(self.name)
        return m['base'] if m else self.name

    @property
    def family(self) -> Tuple[int, ...]:
        m = _FAMILY.match(self.name)
        return tuple(int(v) for v in m['values'].split(',')) if m else ()


@dataclass(frozen=True)
class Statement:
    """`dest (+)= scalars * factors`, summed over every index the
    destination does not carry -- or, for an elementwise `op`, the factors
    combined by it.  `over` is a family index collapsed into this one
    statement: its symbol, the values it takes, and whether the
    destination varies with it (a statement per value) or not (a sum)."""
    dest: Ref
    add: bool
    factors: Tuple[Ref, ...]
    scalars: Tuple[str, ...] = ()
    op: str = ''
    over: Tuple[Tuple[str, Tuple[int, ...], bool], ...] = ()

    @property
    def labels(self) -> List[str]:
        out: List[str] = list(self.dest.indices)
        for f in self.factors:
            out.extend(i for i in f.indices if i not in out)
        return out


@dataclass(frozen=True)
class Loop:
    """A block of statements repeated with every family member moved by
    one: a recursion, each step from the one before.  The body is spelled
    for step `symbol`, which runs through `count` steps from zero."""
    symbol: str
    count: int
    body: Tuple[Statement, ...]


def _ref(blob: dict) -> Ref:
    return Ref(blob['name'], tuple(blob.get('indices') or ()))


def _scalar(alpha: Optional[dict], tensors: Dict[str, dict]) -> Tuple[str, ...]:
    """A scale factor as written: nothing for one, the number otherwise,
    and the name of a runtime scalar (`power(1)`, `dt`)."""
    if not alpha:
        return ()
    tensor = tensors.get(alpha.get('name', ''))
    values = (tensor or {}).get('values') or {}
    data = values.get('data') if values.get('kind') == 'entries' else None
    if data:
        value = float(data[0][1])
        return () if value == 1.0 else (f'{value:g}',)
    return (alpha['name'],)


def statements(description: dict) -> List[Statement]:
    """The export's operations, in order, as statements."""
    tensors = {t['name']: t for t in description.get('tensors', [])}
    out: List[Statement] = []
    for op in description.get('operations', []):
        linear = op.get('linear') or {}
        # `add` is a mask over the destination's axes, or False: the empty
        # mask still accumulates (`frontend/yateto.py`).
        add = linear.get('add', False)
        accumulate = add is not False and add is not None
        kind = op.get('type')
        dest = _ref(op['result'])
        factors = tuple(_ref(a) for a in op.get('args', []))
        name = '' if kind == 'multilinear' else str(op.get('optype') or kind)
        if name == 'Mul' and all(set(f.indices) <= set(dest.indices)
                                 for f in factors):
            # A product that sums over nothing is the multilinear one
            # without a contracted label, and reads as one.
            name = ''
        out.append(Statement(
            dest=dest, add=accumulate, factors=factors,
            scalars=_scalar(linear.get('alpha'), tensors), op=name))
    return out


def temporaries(description: dict) -> List[str]:
    return [t['name'] for t in description.get('tensors', [])
            if (t.get('flags') or {}).get('temporary')]


# ----------------------------------------------------------------------
# Putting the temporaries back
# ----------------------------------------------------------------------

_LETTERS = 'abcdefghijklmnopqrstuvwxyz'


def _fresh(taken: Iterable[str]) -> str:
    used = set(taken)
    for c in _LETTERS:
        if c not in used:
            return c
    raise ValueError('every index letter is taken')


def _substitute(reader: Statement, at: int, writer: Statement) -> Statement:
    """`reader` with its factor `at` replaced by what `writer` computed.

    The writer's destination labels become the ones the reader used; a
    label the writer sums over is renamed only where the reader already
    uses it, so the result reads as the export does wherever it can.
    """
    use = reader.factors[at]
    if len(reader.factors) == 1 and use.indices == reader.dest.indices:
        # The reader only moves what was computed: name it in the writer's
        # labels, which are the ones the export chose for the arithmetic.
        return replace(reader, dest=Ref(reader.dest.name, writer.dest.indices),
                       factors=writer.factors,
                       scalars=reader.scalars + writer.scalars)
    mapping = dict(zip(writer.dest.indices, use.indices))
    taken = set(reader.labels)
    for label in writer.labels:
        if label in mapping:
            continue
        if label in taken or label in mapping.values():
            new = _fresh(taken | set(mapping.values()) | set(writer.labels))
            mapping[label] = new
            taken.add(new)
        else:
            mapping[label] = label
            taken.add(label)
    moved = tuple(Ref(f.name, tuple(mapping[i] for i in f.indices))
                  for f in writer.factors)
    return replace(reader,
                   factors=reader.factors[:at] + moved
                   + reader.factors[at + 1:],
                   scalars=reader.scalars + writer.scalars)


def inline(stmts: Sequence[Statement], temps: Iterable[str]
           ) -> List[Statement]:
    """Every temporary written once by a plain contraction and read once,
    put back into the statement that reads it.  The rest stay."""
    temps = set(temps)
    writes: Dict[str, int] = {}
    reads: Dict[str, int] = {}
    for s in stmts:
        writes[s.dest.name] = writes.get(s.dest.name, 0) + 1
        for f in s.factors:
            reads[f.name] = reads.get(f.name, 0) + 1

    def foldable(s: Statement) -> bool:
        return (s.dest.name in temps and not s.add and not s.op
                and writes[s.dest.name] == 1 and reads.get(s.dest.name) == 1)

    pending: Dict[str, Statement] = {}
    out: List[Statement] = []
    for s in stmts:
        if s.op:
            # An elementwise statement takes no product into it: what it
            # reads is written out before it, where the export had it.
            for f in s.factors:
                if f.name in pending:
                    out.append(pending.pop(f.name))
        else:
            at = 0
            while at < len(s.factors):
                name = s.factors[at].name
                if name in pending:
                    writer = pending.pop(name)
                    s = _substitute(s, at, writer)
                    continue          # the moved factors may be pending too
                at += 1
        if foldable(s):
            pending[s.dest.name] = s
        else:
            out.append(s)
    out.extend(pending.values())       # never read after all: keep them
    return out


def _reads(s: Statement) -> set:
    return {f.name for f in s.factors} | ({s.dest.name} if s.add else set())


def _commutes(a: Statement, b: Statement) -> bool:
    """Whether two statements may change places: neither writes what the
    other reads or writes."""
    return (a.dest.name not in _reads(b) and b.dest.name not in _reads(a)
            and a.dest.name != b.dest.name)


def alias(stmts: Sequence[Statement]) -> List[Statement]:
    """`dQ(0) = Q` read as a name for Q: every other read of `Q` becomes one
    of `dQ(0)`, so that the first derivative is the recursion's first step
    rather than a statement of its own.

    Only for an input -- a tensor the kernel never writes -- copied without
    a factor into a family's member 0, and only where the copy can move to
    the top without passing anything that touches what it writes.
    """
    stmts = list(stmts)
    written = {s.dest.name for s in stmts}
    for k, s in enumerate(stmts):
        if (s.add or s.op or s.scalars or len(s.factors) != 1
                or s.dest.family != (0,)
                or s.factors[0].name in written
                or sum(1 for t in stmts if t.dest.name == s.dest.name) != 1
                or any(s.dest.name in _reads(t) for t in stmts[:k])):
            continue
        source, member = s.factors[0].name, s.dest.name
        moved = [s] + [t for t in stmts if t is not s]
        return [t if t is s else replace(
            t, factors=tuple(Ref(member, f.indices) if f.name == source
                             else f for f in t.factors))
                for t in moved]
    return stmts


def sink(stmts: Sequence[Statement]) -> List[Statement]:
    """The kernel's outputs written last, one output after the other.

    An output here is a tensor no other statement reads -- the time
    integral, accumulated between the derivatives it is taken from.  Its
    statements move past everything they commute with, which is all of it
    once nothing reads them, and the outputs are then ordered by first
    appearance; two statements of one output keep their order.  What
    stays in front is the recursion, now one statement after the other.
    """
    readers: Dict[str, int] = {}
    for st in stmts:
        for f in st.factors:
            readers[f.name] = readers.get(f.name, 0) + 1
    body, tail = [], []
    for k, st in enumerate(stmts):
        later = stmts[k + 1:]
        if (not readers.get(st.dest.name)
                and all(_commutes(st, t) or t.dest.name == st.dest.name
                        for t in later)):
            tail.append(st)
        else:
            body.append(st)
    first = {}
    for st in tail:
        first.setdefault(_shape(st), len(first))
    return body + sorted(tail, key=lambda st: first[_shape(st)])


# ----------------------------------------------------------------------
# Collapsing the families
# ----------------------------------------------------------------------

def _shape(s: Statement) -> Tuple:
    """A statement with its family indices taken out of the names."""
    def strip(r: Ref) -> Tuple:
        return (r.base, r.indices, len(r.family))
    return (strip(s.dest), tuple(strip(f) for f in s.factors),
            tuple(_FAMILY.sub(lambda m: m['base'], x) for x in s.scalars),
            s.op)


def _families(s: Statement) -> List[Tuple[int, ...]]:
    refs = [s.dest, *s.factors]
    out = [r.family for r in refs]
    for x in s.scalars:
        m = _FAMILY.match(x)
        out.append(tuple(int(v) for v in m['values'].split(','))
                   if m else ())
    return out


def _collapsible(group: Sequence[Statement]) -> Optional[Tuple[int, ...]]:
    """The values a single family index runs through over `group`, or None
    where the statements differ in anything else, or in two ways at once.

    Two members may move by a fixed distance from each other -- `dQ(n+1)`
    from `dQ(n)`, `coeffDR(n+4)` beside `dQ(n)` -- which is still one
    index; the smallest of them names it."""
    if len(group) < 2:
        return None
    columns = list(zip(*[_families(s) for s in group]))
    moving = [c for c in columns if len(set(c)) > 1]
    if not moving or any(len(v) != 1 for c in moving for v in c):
        return None
    base = min(moving, key=lambda c: c[0][0])
    if any(len({v[0] - b[0] for v, b in zip(c, base)}) != 1 for c in moving):
        return None
    return tuple(v[0] for v in base)


def collapse(stmts: Sequence[Statement]) -> List[Statement]:
    """A run of statements that differ only in one family index, as one.

    Where the destination stays the same the run is a sum, and every
    statement after the first has to accumulate -- the first may assign.
    Where the destination moves with the index it is a statement per value.
    """
    out: List[Statement] = []
    i = 0
    while i < len(stmts):
        j = i + 1
        while j < len(stmts) and _shape(stmts[j]) == _shape(stmts[i]):
            j += 1
        group = list(stmts[i:j])
        values = _collapsible(group)
        dest_moves = len({s.dest.name for s in group}) > 1
        sums_into_one = not dest_moves and all(s.add for s in group[1:])
        if values is None or not (dest_moves or sums_into_one):
            out.append(stmts[i])
            i += 1
            continue
        first = group[0]
        symbol = _symbol(_taken([first]))
        refs = [[s.dest, *s.factors] for s in group]
        moving = {k for k in range(len(refs[0]))
                  if len({r[k].family for r in refs}) > 1}

        def member(base: str, value: int) -> str:
            shift = value - values[0]
            return (f'{base}({symbol})' if not shift else
                    f'{base}({symbol}{"+" if shift > 0 else "-"}{abs(shift)})')

        def generalize(k: int, r: Ref) -> Ref:
            return (Ref(member(r.base, r.family[0]), r.indices)
                    if k in moving else r)

        scalars = []
        for k, x in enumerate(first.scalars):
            m = _FAMILY.match(x)
            if m and len({s.scalars[k] for s in group}) > 1:
                x = member(m['base'], int(m['values']))
            scalars.append(x)
        out.append(Statement(
            dest=generalize(0, first.dest), add=first.add,
            factors=tuple(generalize(k + 1, f)
                          for k, f in enumerate(first.factors)),
            scalars=tuple(scalars), op=first.op,
            over=first.over + ((symbol, values, dest_moves),)))
        i = j
    return out


def fold(stmts: Sequence[Statement]) -> List:
    """Blocks of two or more statements repeated with every family member
    one further each time, as a `Loop` -- the viscoelastic derivatives,
    which take three statements per step where the elastic ones take one
    and `collapse` already has them."""
    out: List = []
    i = 0
    while i < len(stmts):
        found = None
        for period in range(2, (len(stmts) - i) // 2 + 1):
            shapes = [_shape(x) for x in stmts[i:i + period]]
            reps = 1
            while (i + (reps + 1) * period <= len(stmts)
                   and [_shape(x) for x in stmts[i + reps * period:
                                                 i + (reps + 1) * period]]
                   == shapes):
                reps += 1
            if reps >= 2 and _steps(stmts, i, period, reps):
                found = (period, reps)
                break
        if found is None:
            out.append(stmts[i])
            i += 1
            continue
        period, reps = found
        symbol = _symbol(_taken(stmts[i:i + period]))
        body = []
        for k in range(period):
            first = stmts[i + k]
            group = [stmts[i + b * period + k] for b in range(reps)]

            def step(r: Ref, where: int) -> Ref:
                members = [[g.dest, *g.factors][where] for g in group]
                if len({m.family for m in members}) < 2:
                    return r
                shift = r.family[0]
                return Ref(f'{r.base}({symbol}' + (
                    f'+{shift})' if shift > 0 else
                    f'-{-shift})' if shift < 0 else ')'), r.indices)

            scalars = tuple(
                _step_scalar(x, [g.scalars[j] for g in group], symbol)
                for j, x in enumerate(first.scalars))
            body.append(replace(
                first, dest=step(first.dest, 0),
                factors=tuple(step(f, j + 1)
                              for j, f in enumerate(first.factors)),
                scalars=scalars))
        out.append(Loop(symbol, reps, tuple(body)))
        i += period * reps
    return out


def _steps(stmts: Sequence[Statement], i: int, period: int,
           reps: int) -> bool:
    """Whether, from block to block, every family member that moves moves
    by exactly one, and every statement has one that does."""
    for k in range(period):
        group = [stmts[i + b * period + k] for b in range(reps)]
        columns = list(zip(*[_families(g) for g in group]))
        moving = [c for c in columns if len(set(c)) > 1]
        if not moving:
            return False
        for c in moving:
            if any(len(v) != 1 for v in c):
                return False
            if any(c[b][0] - c[0][0] != b for b in range(reps)):
                return False
    return True


def _step_scalar(x: str, members: Sequence[str], symbol: str) -> str:
    m = _FAMILY.match(x)
    if not m or len(set(members)) < 2:
        return x
    shift = int(m['values'])
    return (f"{m['base']}({symbol}" + (f'+{shift})' if shift > 0 else
                                       f'-{-shift})' if shift < 0 else ')'))


#: The letters a collapsed family index is named by, in order of
#: preference; the first one the statement does not use already.
_FAMILY_SYMBOLS = 'ijfchnrt'


def _taken(stmts: Sequence[Statement]) -> set:
    """Every letter the statements already use: their index labels, the
    symbols of what is collapsed in them, and the letters inside a family
    index already made symbolic (`kDivM(i)`)."""
    out = set()
    for st in stmts:
        out.update(st.labels)
        out.update(sym for sym, _, _ in st.over)
        for name in [st.dest.name, *(f.name for f in st.factors),
                     *st.scalars]:
            m = _ANY_FAMILY.match(name)
            if m:
                out.update(c for c in m['values'] if c.isalpha())
    return out


def _symbol(taken: Iterable[str]) -> str:
    used = set(taken)
    for c in _FAMILY_SYMBOLS:
        if c not in used:
            return c
    return _fresh(used)


def formula(description: dict) -> List:
    """The kernel as the fewest statements the export allows: temporaries
    put back, then the sums over a family, then -- with an input named by
    the family member it is copied to and the statements of one shape
    brought together -- the recursions and the sums that were interleaved
    with them."""
    once = collapse(inline(statements(description),
                           temporaries(description)))
    return fold(collapse(sink(alias(once))))


# ----------------------------------------------------------------------
# Spelling it
# ----------------------------------------------------------------------

#: A family index as written after folding: `kDivM(i)`, `dQ(n+1)`,
#: `V3mTo2n(0,0)`.
_ANY_FAMILY = re.compile(r'^(?P<base>.*?)\((?P<values>[^()]*)\)$')


def _range(values: Tuple[int, ...]) -> str:
    if values == tuple(range(values[0], values[0] + len(values))):
        return f'{values[0]}..{values[-1]}'
    return ','.join(str(v) for v in values)


def lines(formula: Sequence, spell) -> List[str]:
    """A formula in one spelling, a line per statement, a loop's body
    indented under its header."""
    out: List[str] = []
    for item in formula:
        if isinstance(item, Loop):
            out.append(spell(item))
            out.extend('  ' + spell(st) for st in item.body)
        else:
            out.append(spell(item))
    return out


def to_text(s) -> str:
    """Plain text: `Q[kp] += sum_i rDivM(i)[km] I[lq] AplusT(i)[qp]`."""
    if isinstance(s, Loop):
        return f'for {s.symbol} in 0..{s.count - 1}:'

    def ref(r: Ref) -> str:
        return f'{r.name}[{"".join(r.indices)}]'
    sums = ''.join(f'sum_{sym} ' for sym, _, each in s.over if not each)
    body = ' '.join(list(s.scalars) + [ref(f) for f in s.factors])
    if s.op:
        body = f'{s.op.lower()}(' + ', '.join(
            list(s.scalars) + [ref(f) for f in s.factors]) + ')'
    ranges = ''.join(f'  ({sym} in {_range(v)})'
                     for sym, v, moves in s.over if not moves)
    each = ''.join(f'  for {sym} in {_range(v)}'
                   for sym, v, moves in s.over if moves)
    return (f'{ref(s.dest)} {"+=" if s.add else "="} {sums}{body}'
            f'{ranges}{each}')


def _tex_name(name: str, notation: Optional[Dict[str, str]]) -> str:
    """A tensor's name in math mode.  `notation` maps a name to a symbol in
    which `#` marks where the family index goes; where the name has none,
    the marks are taken out with what only frames them."""
    m = _ANY_FAMILY.match(name)
    base, family = (m['base'], m['values']) if m else (name, '')
    family = family.replace(',', ',\\,')
    if notation is not None and base in notation:
        sym = notation[base]
        if '#' in sym:
            if family:
                return sym.replace('#', family)
            for frame in ('^{(#)}', '(#)', '^{#}', '_{#}', '#'):
                sym = sym.replace(frame, '')
            return sym
        return sym + (f'^{{({family})}}' if family else '')
    sym = base if len(base) == 1 else (
        r'\mathrm{' + base.replace('_', r'\_') + '}')
    return sym + (f'^{{({family})}}' if family else '')


def to_latex(s, notation: Optional[Dict[str, str]] = None) -> str:
    """Math-mode LaTeX.  The tensors keep the names the generated code uses
    unless `notation` says how a paper spells them (`NOTATION`)."""
    if isinstance(s, Loop):
        return (rf'\text{{for }} {s.symbol} = 0,\dots,{s.count - 1}\colon')

    def ref(r: Ref) -> str:
        sub = ''.join(r.indices)
        return _tex_name(r.name, notation) + (f'_{{{sub}}}' if sub else '')
    sums = ''.join(f'\\sum_{{{sym}}} ' for sym, _, moves in s.over
                   if not moves)
    scal = [_tex_name(x, notation) if not re.match(r'^-?[\d.]', x) else x
            for x in s.scalars]
    if s.op:
        body = (r'\operatorname{' + s.op.lower() + '}('
                + ', '.join(scal + [ref(f) for f in s.factors]) + ')')
    else:
        body = r'\,'.join(scal + [ref(f) for f in s.factors])
    rel = r'\mathrel{+}=' if s.add else '='
    tail = ''.join(
        f',\\quad {sym} = ' + _range(v).replace('..', ',\\dots,')
        for sym, v, moves in s.over if moves)
    return f'{ref(s.dest)} {rel} {sums}{body}{tail}'


def to_yateto(s) -> str:
    """The statement as yateto source, the way SeisSol's codegen writes
    one: `Q['kp'] <= Q['kp'] + sum(rDivM[i]['km'] * ... for i in
    range(4))` -- a sum over a family as a generator, a statement per
    member as a loop around it."""
    if isinstance(s, Loop):
        return f'for {s.symbol} in range({s.count}):'

    def name(x: str) -> str:
        m = _ANY_FAMILY.match(x)
        return f"{m['base']}[{m['values']}]" if m else x

    def ref(r: Ref) -> str:
        return f"{name(r.name)}['{''.join(r.indices)}']"

    def bounds(v: Tuple[int, ...]) -> str:
        if v == tuple(range(v[0], v[0] + len(v))):
            return (f'range({v[-1] + 1})' if v[0] == 0
                    else f'range({v[0]}, {v[-1] + 1})')
        return '(' + ', '.join(str(x) for x in v) + ')'

    factors = [name(x) for x in s.scalars] + [ref(f) for f in s.factors]
    body = (f'{s.op}(' + ', '.join(factors) + ')' if s.op
            else ' * '.join(factors))
    for sym, v, moves in s.over:
        if not moves:
            body = f'sum({body} for {sym} in {bounds(v)})'
    lhs = ref(s.dest)
    rhs = f'{lhs} + {body}' if s.add else body
    loops = ''.join(f'for {sym} in {bounds(v)}: '
                    for sym, v, moves in s.over if moves)
    return f'{loops}{lhs} <= {rhs}'


# ----------------------------------------------------------------------
# What the export says about the kernel
# ----------------------------------------------------------------------

@dataclass
class Described:
    """One kernel of one export: its formula, its shape, its cost."""
    workload: str
    capture: Optional[Capture]
    formula: List[Statement]
    operations: int
    temporaries: int
    depth: int
    operators: List[Tuple[str, Tuple[int, ...], Optional[float]]]
    operands: List[Tuple[str, Tuple[int, ...]]]
    #: Per element, from `analysis.cost`; None where TensorForge could not
    #: be imported or the kernel could not be read.
    flops: Optional[int] = None
    nonzero_flops: Optional[int] = None
    nonzero_exact: Optional[bool] = None
    bytes: Optional[int] = None
    shared_bytes: Optional[int] = None
    notes: List[str] = field(default_factory=list)

    @property
    def family(self) -> str:
        return family_of(self.workload)

    @property
    def stage(self) -> str:
        return role_of(self.workload)[0]

    @property
    def role(self) -> str:
        return role_of(self.workload)[1]

    @property
    def intensity(self) -> Optional[float]:
        return self.flops / self.bytes if self.flops and self.bytes else None

    @property
    def nonzero_intensity(self) -> Optional[float]:
        return (self.nonzero_flops / self.bytes
                if self.nonzero_flops and self.bytes else None)


def _depth(stmts: Sequence[Statement]) -> int:
    """The longest chain of statements each reading what the one before
    wrote: how many steps a kernel is, rather than how many lines."""
    level: Dict[str, int] = {}
    deepest = 0
    for s in stmts:
        d = 1 + max((level.get(f.name, 0) for f in s.factors), default=0)
        if s.add:
            d = max(d, level.get(s.dest.name, 0))
        level[s.dest.name] = max(level.get(s.dest.name, 0), d)
        deepest = max(deepest, d)
    return deepest


def _density(tensor: dict) -> Optional[float]:
    """Nonzeros over the stored box, for an operator that carries numbers."""
    values = tensor.get('values') or {}
    if values.get('kind') != 'entries':
        return None
    storage = tensor.get('storage') or {}
    sizes = storage.get('sizes') or storage.get('shape') or []
    volume = 1
    for s in sizes:
        volume *= int(s)
    nonzero = sum(1 for _, v in values.get('data') or [] if v != 0)
    return nonzero / volume if volume else None


def describe(workload: str, description: dict,
             capture: Optional[Capture] = None,
             costs: bool = True) -> Described:
    """Everything this module knows about one kernel of one export."""
    stmts = statements(description)
    temps = set(temporaries(description))
    operators, operands = [], []
    for t in description.get('tensors', []):
        flags = t.get('flags') or {}
        if t.get('residence') != 'memory' or flags.get('temporary'):
            continue
        shape = tuple(int(x) for x in (t.get('storage') or {}).get('shape')
                      or ())
        if flags.get('constant'):
            operators.append((t['name'], shape, _density(t)))
        else:
            operands.append((t['name'], shape))
    out = Described(
        workload=workload, capture=capture, formula=formula(description),
        operations=len(stmts), temporaries=len(temps), depth=_depth(stmts),
        operators=operators, operands=operands)
    if costs:
        _attach_cost(out, description)
    return out


def _attach_cost(out: Described, description: dict) -> None:
    """The per-element figures from `analysis.cost`, which needs TensorForge
    -- left absent, with a note, where it cannot be had."""
    try:
        root = HERE.parents[1] / 'src'
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from tensorforge.analysis.cost import list_cost
        from tensorforge.common.basic_types import Datatype
        from tensorforge.frontend.yateto import DescriptionReader
    except ImportError as exc:                   # pragma: no cover - env
        out.notes.append(f'no cost figures: {exc}')
        return
    datatype = {'s': Datatype.F32, 'd': Datatype.F64}.get(
        out.capture.precision if out.capture else '', Datatype.F64)
    try:
        descrs = DescriptionReader(arch=None).read(description)[0]
        cost = list_cost(descrs, batch=1, datatype=datatype)
    except Exception as exc:                     # noqa: BLE001
        out.notes.append(f'no cost figures: {type(exc).__name__}: {exc}')
        return
    out.flops = cost.flops
    out.nonzero_flops = cost.nonzero_flops
    out.nonzero_exact = cost.nonzero_exact
    out.bytes = sum(t.read + t.write for t in cost.tensors if t.per_batch)
    out.shared_bytes = sum(t.read + t.write for t in cost.tensors
                           if not t.per_batch)


# ----------------------------------------------------------------------
# Reading exports
# ----------------------------------------------------------------------

def exports(paths: Sequence[Path]) -> List[Path]:
    out: List[Path] = []
    for p in paths:
        out.extend(sorted(p.glob('*.json')) if p.is_dir() else [p])
    return out


def describe_all(paths: Sequence[Path], kernels: Sequence[str] = ('*',),
                 costs: bool = True) -> List[Described]:
    """Every kernel matching one of `kernels` (globs over the names) in
    every export under `paths`."""
    out: List[Described] = []
    for path in exports(paths):
        blob = json.loads(path.read_text())
        captured = (Capture.from_config(blob.get('config'))
                    or Capture.from_tag(path.stem))
        for name, description in sorted(
                (blob.get('descriptions') or {}).items()):
            # The name first: a kernel name carries brackets, which a glob
            # reads as a character class.
            if not any(name == k or fnmatch.fnmatch(name, k)
                       for k in kernels):
                continue
            out.append(describe(name, description, captured, costs=costs))
    return out


CSV_FIELDS = ('capture', 'equation', 'order', 'precision', 'simulations',
              'workload', 'family', 'stage', 'role', 'operations',
              'temporaries', 'depth', 'flops', 'nonzero_flops',
              'nonzero_exact', 'bytes', 'shared_bytes', 'intensity',
              'nonzero_intensity', 'formula')


def to_row(d: Described) -> Dict:
    c = d.capture
    return {
        'capture': c.tag if c else '', 'equation': c.equation if c else '',
        'order': c.order if c else '', 'precision': c.precision if c else '',
        'simulations': c.simulations if c else '',
        'workload': d.workload, 'family': d.family, 'stage': d.stage,
        'role': d.role, 'operations': d.operations,
        'temporaries': d.temporaries, 'depth': d.depth,
        'flops': d.flops, 'nonzero_flops': d.nonzero_flops,
        'nonzero_exact': d.nonzero_exact, 'bytes': d.bytes,
        'shared_bytes': d.shared_bytes,
        'intensity': f'{d.intensity:.4g}' if d.intensity else '',
        'nonzero_intensity': (f'{d.nonzero_intensity:.4g}'
                              if d.nonzero_intensity else ''),
        'formula': '; '.join(lines(d.formula, to_text)),
    }


def write_csv(described: Sequence[Described], path: Path) -> None:
    with path.open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        w.writeheader()
        for d in described:
            w.writerow(to_row(d))


def report(d: Described) -> str:
    """One kernel, for a terminal."""
    c = d.capture
    out = [f'{d.workload}' + (f'  [{c.tag}]' if c else '')]
    if d.role:
        out.append(f'  {d.stage}: {d.role}')
    out.append(f'  {d.operations} operations, {d.temporaries} temporaries, '
               f'{d.depth} steps deep')
    out.extend(f'    {line}' for line in lines(d.formula, to_text))
    if d.operators:
        out.append('  operators: ' + ', '.join(
            f'{n} {"x".join(map(str, sh))}'
            + (f' ({dens:.0%} nonzero)' if dens is not None else '')
            for n, sh, dens in d.operators))
    if d.flops is not None:
        out.append(
            f'  per element: {d.flops:,} flop dense, {d.nonzero_flops:,} '
            f'needed{"" if d.nonzero_exact else " (no pattern: dense)"}, '
            f'{d.bytes:,} byte -> {d.intensity or 0:.2f} / '
            f'{d.nonzero_intensity or 0:.2f} flop/byte')
    out.extend(f'  note: {n}' for n in d.notes)
    return '\n'.join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('exports', type=Path, nargs='+',
                    help='export files, or directories of them')
    ap.add_argument('--kernels', default='*',
                    help='comma-separated globs over the kernel names')
    ap.add_argument('--csv', type=Path, default=None,
                    help='write every kernel of every export as a CSV row')
    ap.add_argument('--no-cost', action='store_true',
                    help='formulas and structure only; needs no TensorForge')
    args = ap.parse_args()

    described = describe_all(args.exports,
                             [k.strip() for k in args.kernels.split(',')],
                             costs=not args.no_cost)
    if not described:
        print('no kernel matched', file=sys.stderr)
        return 2
    if args.csv:
        write_csv(described, args.csv)
        print(f'written: {args.csv} ({len(described)} kernels)')
    else:
        print('\n\n'.join(report(d) for d in described))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
