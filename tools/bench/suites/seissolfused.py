# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""SeisSol's local step as one kernel, against the three launches SeisSol
makes of it.

SeisSol's GPU path computes an element's local step -- the "predict" of its
time cluster -- in three launches over the same cells: `gpu_derivative`
(the time derivatives `dQ(k)` and their integral `I` from the DOFs `Q`),
`gpu_volume` and `gpu_localFluxAll` (both adding onto `Q` from `I`).  The
three read and write per element only, so one kernel can compute all of
them; what keeps them apart today are the operators, which do not fit one
kernel's code and shared memory at the higher orders.  Here the three
descriptions are combined into one (`combine_descriptions`): every
temporary and scalar constant renamed per kernel, the operands they share
-- `I`, `Q`, `star(d)` -- one tensor, the operations concatenated.
`Options.fission` then cuts the combined kernel where the generator
chooses, or where a configuration forces it.

Two forms of the combination (`VARIANTS`):

* `a` -- the derivatives stay operands and are written out, as today.
* `b` -- the derivatives are temporaries: only `I` and `Q` leave the kernel.
  SeisSol needs an element's derivatives in memory only where another
  kernel reads them -- a dynamic-rupture face, a neighbor in a finer time
  cluster, a free surface with gravity -- and writes every other element's
  into scratch memory nobody reads.

Where they come from: with `TF_FUSED_CAPTURES` naming a directory of
exports (`tools/seissol_export.py`, `<system>-o<order>-<p>.json`) of a
SeisSol whose predictor is a family -- `gpu_derivative[0]` writes the
derivatives, `[1]` keeps them as temporaries (SeisSol branch
`davschneller/derivative-free-ader`) -- `a` is member 0 with the volume and
the local flux, `b` member 1.  Without, the recorded descriptions of
`tests/seissol_suite.py`, and `b` that derivative with its `dQ(k)` marked
temporaries by hand (`keep_derivatives`), which keeps one dead copy the
family's member drops.

Configurations (`CONFIGS`): `three`, the three kernels launched one after
the other (`suite.kernels_partition`), each built as SeisSol builds it;
`f0`, the combined kernel whole; `fauto`, cut where `fission='auto'`
decides; `f2` to `f4`, cut into that many pieces at the flop-balanced legal
points.  `CHECK` compares every configuration's outputs with the
reference's before anything is timed.

The orders, precisions and forms come from the environment, so that a job
can split the work: `TF_FUSED_ORDERS` (`4,6,8`), `TF_FUSED_PRECISIONS`
(`s,d`), `TF_FUSED_VARIANTS` (`a,b`), `TF_FUSED_SYSTEM`
(`elastic-linearck`).
"""

from __future__ import annotations

import copy
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[3]
for _extra in (ROOT / 'src', ROOT / 'tests', Path(__file__).resolve().parents[1]):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

import seissol_suite  # noqa: E402
from suite import Config, Workload  # noqa: E402
from tensorforge.common.basic_types import Datatype  # noqa: E402
from tensorforge.common.context import Options  # noqa: E402
from tensorforge.frontend.yateto import DescriptionReader  # noqa: E402

NAME = 'seissolfused'
DESCRIPTION = ("SeisSol's local step (derivative, volume, local flux) as one "
               'kernel, whole and cut, against its three launches')

#: The kernels of the local step, in the order SeisSol launches them.
KERNELS = ('gpu_derivative', 'gpu_volume', 'gpu_localFluxAll')

#: The predictor's members in a SeisSol whose predictor is a family: the one
#: that writes the derivatives (`a`) and the one that keeps them (`b`).
MEMBERS = {'a': 'gpu_derivative[0]', 'b': 'gpu_derivative[1]'}

#: The two forms of the combination: the derivatives written out (`a`), or
#: kept inside the kernel as temporaries (`b`).
VARIANTS = ('a', 'b')

#: The name of a time derivative.
DERIVATIVE = re.compile(r'^dQ\(\d+\)$')

BATCHES = (10000, 100000, 262144)
CHECK = True

CONFIGS = [
    Config('three', partition='kernels'),
    Config('f0', Options(fission=0)),
    Config('fauto', Options(fission='auto')),
    Config('f2', Options(fission=2)),
    Config('f3', Options(fission=3)),
    Config('f4', Options(fission=4)),
]

PRECISION = {'s': Datatype.F32, 'd': Datatype.F64}


def _rename(node, table: Dict[str, str]):
    """`node` with every `name` that `table` holds replaced."""
    if isinstance(node, dict):
        return {k: (table.get(v, v) if k == 'name' and isinstance(v, str)
                    else _rename(v, table)) for k, v in node.items()}
    if isinstance(node, list):
        return [_rename(x, table) for x in node]
    return node


def combine_descriptions(described: Sequence[Tuple[str, Dict]],
                         keep_derivatives: bool = True) -> Dict:
    """One description computing the `(name, description)` kernels one
    after the other, per element.

    A kernel's own temporaries and scalar constants (`_tmp*`, `_scalar*`)
    are renamed `_k<n>...`, so that two kernels' never meet; every other
    tensor is the kernels' operand and is merged by name -- where two
    kernels state it differently, that is an error, not a choice.  `parts`
    names the kernels with their counts of operations, which is where
    `suite.kernels_partition` cuts the description into the launches
    SeisSol makes.  With `keep_derivatives` false, the time derivatives
    `dQ(k)` are marked temporaries."""
    tensors: Dict[str, Dict] = {}
    operations: List[Dict] = []
    parts: List[List] = []
    version = None
    for n, (kernel, d) in enumerate(described):
        if version is not None and d['version'] != version:
            raise ValueError(f'{kernel} states interface version '
                             f'{d["version"]}, the kernels before it {version}')
        version = d['version']
        own = {t['name']: f'_k{n}{t["name"]}' for t in d['tensors']
               if t['name'].startswith('_')}
        d = _rename(d, own)
        for t in d['tensors']:
            seen = tensors.get(t['name'])
            if seen is not None and seen != t:
                differ = sorted(k for k in set(seen) | set(t)
                                if seen.get(k) != t.get(k))
                raise ValueError(f'{t["name"]} differs between the kernels '
                                 f'in {differ}')
            tensors.setdefault(t['name'], copy.deepcopy(t))
        operations += d['operations']
        parts.append([kernel, len(d['operations'])])
    if not keep_derivatives:
        for t in tensors.values():
            if DERIVATIVE.match(t['name']):
                t['flags'] = dict(t['flags'], temporary=True)
                # As yateto states a temporary: strided per element, which
                # no one binds.
                t['addressing'] = 'n*N+o&'
    return {'version': version, 'tensors': list(tensors.values()),
            'operations': operations, 'parts': parts}


def combine(system: str, config: str, kernels: Sequence[str] = KERNELS,
            keep_derivatives: bool = True) -> Dict:
    """`combine_descriptions` over the recorded descriptions of
    `tests/seissol_suite.py`."""
    return combine_descriptions(
        [(k, seissol_suite.description(system, config, k)) for k in kernels],
        keep_derivatives)


def captured(system: str, order: int, precision: str) -> Optional[Dict]:
    """The export of `TF_FUSED_CAPTURES` for one configuration, or None."""
    where = os.environ.get('TF_FUSED_CAPTURES')
    if not where:
        return None
    path = Path(where) / f'{system}-o{order}-{precision}.json'
    if not path.exists():
        raise SystemExit(f'TF_FUSED_CAPTURES: no {path}')
    return json.loads(path.read_text())


def kernels_of(system: str, order: int, precision: str, variant: str
               ) -> List[Tuple[str, Dict]]:
    """The local step's kernels for one form, `(name, description)`: the
    predictor's member from an export where there is one, else the recorded
    derivative (`b` then marks its derivatives temporaries itself)."""
    blob = captured(system, order, precision)
    if blob is not None:
        names = (MEMBERS[variant],) + KERNELS[1:]
        return [(k, copy.deepcopy(blob['descriptions'][k])) for k in names]
    config = f'{system}-o{order}-{precision}'
    return [(k, seissol_suite.description(system, config, k))
            for k in KERNELS]


def workload(system: str, order: int, precision: str, variant: str
             ) -> Workload:
    described = kernels_of(system, order, precision, variant)
    keep = variant == 'a' or captured(system, order, precision) is not None
    description = combine_descriptions(described, keep_derivatives=keep)

    def factory(description=description):
        return DescriptionReader(arch=None).read(
            copy.deepcopy(description))[0]
    source = ('export' if captured(system, order, precision) is not None
              else 'recorded')
    return Workload(name=f'local-o{order}-{precision}-{variant}',
                    descrs=factory, datatype=PRECISION[precision],
                    origin=(f'combined:{source}:{system}-o{order}-{precision}:'
                            + ','.join(k for k, _ in described)),
                    description=description)


def _env(name: str, default: str) -> List[str]:
    return [x.strip() for x in os.environ.get(name, default).split(',')
            if x.strip()]


def workloads() -> List[Workload]:
    system = os.environ.get('TF_FUSED_SYSTEM', 'elastic-linearck')
    return [workload(system, int(order), precision, variant)
            for precision in _env('TF_FUSED_PRECISIONS', 's,d')
            for order in _env('TF_FUSED_ORDERS', '4,6,8')
            for variant in _env('TF_FUSED_VARIANTS', ','.join(VARIANTS))]
