# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Which SeisSol configuration a kernel was captured from.

A kernel name is not a kernel.  `gpu_localFluxAll` is a different operation at
every order, in single and in double, with one simulation and with eight, and
for the elastic and the viscoelastic system -- over the sixteen exports in
`data/`, the compulsory intensity under that one name spans 3.6 to 38
flop/byte (dense, per element), and 141 of the 142 names spread by more than a
factor of two.  So anything that keys a figure by the workload name alone keys
it by whichever of those happened to come first.

What tells them apart is the export's own configuration (`config` in a
capture from `tools/seissol_export.py`), and this module is the one spelling
of it.  It imports nothing beyond the standard library, because the readers
(`compare.py` and those built on it) run on a laptop after the data was copied
back, where TensorForge itself need not be importable.

The tag is the name `evaluate.py` gives an export and a run directory,
`elastic-6-s-s1`, so a run that predates rows carrying the fields can still
be placed by its directory.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

#: `<equation>-<order>-<precision>-s<simulations>`, what `evaluate.py` writes.
_TAG = re.compile(r'^(?P<equation>[A-Za-z][A-Za-z0-9]*)-(?P<order>\d+)-'
                  r'(?P<precision>[sd])-s(?P<simulations>\d+)$')

#: SeisSol's spelling of the precision against the bench's `Datatype` name.
DATATYPE = {'s': 'F32', 'd': 'F64'}
PRECISION = {v: k for k, v in DATATYPE.items()}


@dataclass(frozen=True)
class Capture:
    """The configuration an export was generated at.

    `mechanisms` and `solver` are carried because they change the kernels
    (a viscoelastic system with three mechanisms is not the one with five),
    but they are not part of the tag: the export matrix varies neither
    within an equation, and the tag is a name people type.
    """
    equation: str
    order: int
    precision: str
    simulations: int = 1
    mechanisms: int = 0
    solver: str = ''

    @property
    def tag(self) -> str:
        return (f'{self.equation}-{self.order}-{self.precision}-'
                f's{self.simulations}')

    @property
    def datatype(self) -> str:
        """The bench's name for the precision, as rows spell it."""
        return DATATYPE.get(self.precision, self.precision)

    def fields(self) -> Dict:
        """What a row carries, flat, so a reader needs no nesting rule."""
        return {'capture': self.tag, 'equation': self.equation,
                'order': self.order, 'precision': self.precision,
                'simulations': self.simulations,
                'mechanisms': self.mechanisms, 'solver': self.solver}

    @classmethod
    def from_config(cls, config: Optional[Mapping]) -> Optional['Capture']:
        """From an export's `config` block; `None` where it lacks the three
        fields nothing can be placed without."""
        if not config:
            return None
        try:
            return cls(equation=str(config['equation']),
                       order=int(config['order']),
                       precision=str(config['precision']),
                       simulations=int(config.get('multiple_simulations')
                                       or config.get('simulations') or 1),
                       mechanisms=int(config.get('mechanisms') or 0),
                       solver=str(config.get('solver') or ''))
        except (KeyError, TypeError, ValueError):
            return None

    @classmethod
    def from_tag(cls, tag: str) -> Optional['Capture']:
        m = _TAG.match(tag)
        if not m:
            return None
        return cls(equation=m['equation'], order=int(m['order']),
                   precision=m['precision'],
                   simulations=int(m['simulations']))

    @classmethod
    def from_row(cls, row: Mapping) -> Optional['Capture']:
        """From the fields `run.py` writes into a row (`fields`)."""
        if not row.get('equation') or not row.get('order'):
            return None
        precision = row.get('precision') or PRECISION.get(
            str(row.get('datatype') or ''), '')
        if not precision:
            return None
        return cls(equation=str(row['equation']), order=int(row['order']),
                   precision=str(precision),
                   simulations=int(row.get('simulations') or 1),
                   mechanisms=int(row.get('mechanisms') or 0),
                   solver=str(row.get('solver') or ''))


def capture_of(path: Path, rows: Sequence[Mapping]) -> Optional[Capture]:
    """The capture a run is of: what its rows recorded, else its directory.

    The rows win: a directory can be renamed and a row cannot.  The path is
    read part by part from the innermost, so `eval/runs/elastic-6-s-s1` and
    `gb200/runs/elastic-6-s-s1/bench.json` both resolve.  Never guessed from
    the kernels' shapes -- that would be a number nobody recorded.
    """
    for row in rows:
        found = Capture.from_row(row)
        if found is not None:
            return found
    for part in reversed(Path(path).parts):
        found = Capture.from_tag(part)
        if found is not None:
            return found
    return None
