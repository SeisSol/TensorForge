# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""What a SeisSol run says about itself, for the figures that are about runs.

The kernel benchmark measures one kernel at a time; a paper about a code that
uses the kernels also has to say what the code did with them.  SeisSol writes
that down itself, and this reads it rather than asking for it to be copied
into a table by hand:

- `<prefix>-summary.json` (`seissol-run-summary/1`): the build (device arch,
  backend), the configuration, the ranks, the cell updates and the time they
  took, the time per compute region with SeisSol's own fit of it against the
  cells a launch carried, and the LTS layout with its theoretical speedup.
- `<prefix>-clustering.csv`: per rank, cluster and layer -- interior or copy --
  the cells a launch carries and the cluster's timestep.  The clustering is
  what local time stepping makes of a mesh, and the launch sizes in it are
  what decides how much of the time step goes into launch overhead.

Standard library only, as for `capture.py`: the readers run where SeisSol and
TensorForge are not installed.

## The model of a clustered run

A time step of `n` cells costs `t0 + c n` (`publish.launch_fit`, over the
kernels of a step as SeisSol launches them).  Every layer of every cluster is
launched apart, once per step of its cluster: per rank, over a stretch of
simulated time `T`, local time stepping costs the sum over its layers of
`T / dt * (t0 + c n)` and global time stepping that of its two layers --
interior and copy -- at the smallest timestep.  The slowest rank decides.
The ideal, against which both are read, counts cell updates only, which is
what SeisSol's "theoretical speedup" is (up to its wiggle factor, which the
cluster timesteps here include).  What the model leaves out it says: the
Taylor expansion a cell computes for a neighbour in another cluster, dynamic
rupture, and everything outside the kernels.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from capture import Capture

#: What `summary.json` says it is; a reader of another schema reads nothing.
SCHEMA = 'seissol-run-summary/1'

#: SeisSol's configuration name (`configName` in `src/Common/ConfigValue.cpp`):
#: `elastic-linearck-o4-f64-stroud`, `viscoelastic-linearckanelastic-m3-o4-
#: f32-stroud-s8`.
_CONFIG = re.compile(
    r'^(?P<material>[a-z]+)-(?P<solver>[a-z]+)(?:-m(?P<mechanisms>\d+))?'
    r'-o(?P<order>\d+)-(?P<precision>f32|f64)-(?P<quadrature>[a-z]+)'
    r'(?:-s(?P<simulations>\d+))?$')

#: The compute regions of SeisSol's loop statistics, in the order a time
#: step runs them, with what a figure calls them.
REGIONS = (('computeLocalIntegration', 'local'),
           ('computeNeighboringIntegration', 'neighbour'),
           ('computeDynamicRupture', 'dynamic rupture'),
           ('computePointSources', 'point sources'))


def capture_of_config(name: str) -> Optional[Capture]:
    """A SeisSol configuration name as the export it corresponds to."""
    m = _CONFIG.match(name or '')
    if not m:
        return None
    return Capture(equation=m['material'], order=int(m['order']),
                   precision={'f64': 'd', 'f32': 's'}[m['precision']],
                   simulations=int(m['simulations'] or 1),
                   mechanisms=int(m['mechanisms'] or 0),
                   solver=m['solver'])


# ----------------------------------------------------------------------
# The clustering
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class Layer:
    """One layer of one cluster on one rank, launched on its own: its
    interior cells, or its copy cells -- a copy cell once per neighbour it
    is sent to, since every copy is computed."""
    rank: int
    cluster: int
    copy: bool
    cells: int
    timestep: float


@dataclass
class Clustering:
    """A mesh as local time stepping made it: every layer of every rank."""
    name: str
    layers: List[Layer]
    #: Not from a mesh: a series written down to lay figures out with,
    #: which every figure drawn from it says.
    synthetic: bool = False

    @property
    def ranks(self) -> List[int]:
        return sorted({layer.rank for layer in self.layers})

    @property
    def timestep(self) -> float:
        """The smallest cluster timestep: what global time stepping takes."""
        return min(layer.timestep for layer in self.layers)

    def updates(self) -> List[Tuple[Layer, float]]:
        """Each layer with its cell updates per unit of simulated time."""
        return [(layer, layer.cells / layer.timestep)
                for layer in self.layers if layer.cells]

    def ideal_speedup(self) -> float:
        """Cell updates at the smallest timestep over those of the
        clustering: local time stepping with every launch free."""
        cells = sum(layer.cells for layer in self.layers)
        lts = sum(u for _, u in self.updates())
        return cells / self.timestep / lts if lts else 1.0

    def seconds(self, t0_ns: float, c_ns: float, gts: bool = False
                ) -> Dict[int, float]:
        """Seconds per rank for one unit of simulated time, a step of `n`
        cells costing `t0 + c n` nanoseconds (`module docstring`)."""
        out: Dict[int, float] = {}
        for rank in self.ranks:
            mine = [layer for layer in self.layers if layer.rank == rank]
            if gts:
                halves = [sum(layer.cells for layer in mine
                              if layer.copy == copy) for copy in (False, True)]
                ns = sum(t0_ns + c_ns * n for n in halves if n) / \
                    self.timestep
            else:
                ns = sum((t0_ns + c_ns * layer.cells) / layer.timestep
                         for layer in mine if layer.cells)
            out[rank] = ns * 1e-9
        return out

    def predicted_speedup(self, t0_ns: float, c_ns: float) -> float:
        """Global over local time stepping, on the slowest rank of each."""
        lts = max(self.seconds(t0_ns, c_ns).values())
        gts = max(self.seconds(t0_ns, c_ns, gts=True).values())
        return gts / lts if lts else 1.0


def read_clustering(path: Path, name: Optional[str] = None) -> Clustering:
    """`<prefix>-clustering.csv` as SeisSol writes it
    (`src/ResultWriter/ClusteringWriter.cpp`)."""
    path = Path(path)
    layers: List[Layer] = []
    run_ids = set()
    with path.open(newline='') as fh:
        for row in csv.DictReader(fh):
            layer_type = (row.get('layerType') or '').strip().lower()
            if layer_type not in ('interior', 'copy'):
                raise ValueError(f'{path}: layer type {layer_type!r} is '
                                 'neither interior nor copy')
            layers.append(Layer(rank=int(row['rank']),
                                cluster=int(row['localId']),
                                copy=layer_type == 'copy',
                                cells=int(row['size']),
                                timestep=float(row['timestep'])))
            run_ids.add((row.get('runId') or '').strip())
    if not layers:
        raise ValueError(f'{path}: no clusters')
    stem = path.name[:-len('-clustering.csv')] \
        if path.name.endswith('-clustering.csv') else path.stem
    synthetic = run_ids == {'synthetic'}
    return Clustering(name or stem, layers, synthetic)


def clustering_from_summary(summary: Dict, name: str) -> Optional[Clustering]:
    """The clustering a summary's LTS section describes, where no
    `-clustering.csv` came with it: the cells of every cluster over all
    ranks, spread evenly over them and all interior -- the launch sizes are
    then a mean, which the CSV makes exact."""
    lts = summary.get('lts') or {}
    clusters = lts.get('clusters') or []
    ranks = int((summary.get('parallel') or {}).get('ranks') or 1)
    layers = []
    for cluster in clusters:
        cells = cluster.get('cells')
        if cells is None or not cluster.get('timestep'):
            return None
        for rank in range(ranks):
            layers.append(Layer(rank, int(cluster['id']), False,
                                int(round(cells / ranks)),
                                float(cluster['timestep'])))
    return Clustering(name, layers) if layers else None


# ----------------------------------------------------------------------
# The run
# ----------------------------------------------------------------------

def _mean(value) -> Optional[float]:
    """A figure the summary keeps over ranks, by its mean; a bare number
    as it stands."""
    if isinstance(value, dict):
        return value.get('mean')
    return value if isinstance(value, (int, float)) else None


def _max(value) -> Optional[float]:
    if isinstance(value, dict):
        return value.get('max')
    return value if isinstance(value, (int, float)) else None


def _sum(value) -> Optional[float]:
    if isinstance(value, dict):
        return value.get('sum')
    return value if isinstance(value, (int, float)) else None


@dataclass
class Run:
    """One SeisSol run, as its summary states it."""
    path: Path
    name: str
    arch: str
    backend: str
    configuration: str
    capture: Optional[Capture]
    ranks: int
    cells: float
    cell_updates: float
    #: Seconds of the whole simulation on the slowest rank, and of the
    #: simulated time it covered.
    seconds: float
    simulated: float
    #: Mean seconds per rank: each compute region, the compute outside the
    #: kernels, the blocking IO, and the simulation in total.
    split: Dict[str, float] = field(default_factory=dict)
    #: SeisSol's fit of each region's time against the cells a launch
    #: carried: region -> (seconds per launch, seconds per cell).
    fits: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    clusters: int = 1
    ideal_speedup: Optional[float] = None
    clustering: Optional[Clustering] = None

    @property
    def lts(self) -> bool:
        return self.clusters > 1

    @property
    def updates_per_second_per_rank(self) -> Optional[float]:
        if not self.seconds or not self.ranks:
            return None
        return self.cell_updates / self.seconds / self.ranks

    def half(self, regions: Sequence[str] = ('computeLocalIntegration',
                                             'computeNeighboringIntegration')
             ) -> Optional[float]:
        """The cells at which a launch of these regions reaches half its
        asymptotic rate, from SeisSol's own fit -- the measured counterpart
        of `publish`'s n-half of a time step."""
        if not all(r in self.fits for r in regions):
            return None
        t0 = sum(self.fits[r][0] for r in regions)
        c = sum(self.fits[r][1] for r in regions)
        return t0 / c if t0 > 0 and c > 0 else None


def read_summary(path: Path) -> Optional[Run]:
    """`<prefix>-summary.json`, with the `-clustering.csv` beside it where
    there is one.  None for another schema, or for a run that did not
    finish."""
    path = Path(path)
    try:
        blob = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    if blob.get('schema') != SCHEMA or blob.get('status') != 'completed':
        return None
    build = blob.get('build') or {}
    configurations = (blob.get('configurations')
                      or build.get('configurations') or [])
    configuration = configurations[0] if configurations else ''
    simulation = blob.get('simulation') or {}
    times = (blob.get('time') or {}).get('simulation') or {}
    kernels = blob.get('kernels') or {}
    regions = kernels.get('regions') or {}
    split: Dict[str, float] = {}
    fits: Dict[str, Tuple[float, float]] = {}
    for region, _ in REGIONS:
        entry = regions.get(region)
        if not entry:
            continue
        seconds = _mean(entry.get('time'))
        if seconds is not None:
            split[region] = seconds
        regression = entry.get('regression') or {}
        if 'constant' in regression and 'perElement' in regression:
            fits[region] = (float(regression['constant']),
                            float(regression['perElement']))
    for key in ('outsideKernels', 'io', 'total'):
        value = _mean(times.get(key))
        if value is not None:
            split[key] = value
    lts = blob.get('lts') or {}
    stem = path.name[:-len('-summary.json')] \
        if path.name.endswith('-summary.json') else path.stem
    clustering = None
    beside = path.with_name(f'{stem}-clustering.csv')
    if beside.exists():
        clustering = read_clustering(beside, stem)
    else:
        clustering = clustering_from_summary(blob, stem)
    return Run(
        path=path, name=stem,
        arch=str(build.get('deviceArch') or '?'),
        backend=str(build.get('deviceBackend') or ''),
        configuration=configuration,
        capture=capture_of_config(configuration),
        ranks=int((blob.get('parallel') or {}).get('ranks') or 1),
        cells=float(_sum((blob.get('mesh') or {}).get('cells')) or 0.0),
        cell_updates=float(simulation.get('cellUpdates') or 0.0),
        seconds=float(_max(times.get('total')) or 0.0),
        simulated=float((simulation.get('endTime') or 0.0)
                        - (simulation.get('startTime') or 0.0)),
        split=split, fits=fits,
        clusters=len(lts.get('clusters') or []) or 1,
        ideal_speedup=((lts.get('theoreticalSpeedup') or {})
                       .get('clustered')),
        clustering=clustering)


def collect_runs(targets: Iterable[Path]) -> List[Run]:
    """Every finished run under the paths: a directory is walked for
    `*-summary.json`, a file is read as it stands."""
    out: List[Run] = []
    seen = set()
    for target in targets:
        target = Path(target)
        paths = ([target] if target.is_file()
                 else sorted(target.rglob('*-summary.json')))
        for path in paths:
            if path.resolve() in seen:
                continue
            seen.add(path.resolve())
            run = read_summary(path)
            if run is not None:
                out.append(run)
    return out


def measured_speedups(runs: Sequence[Run]) -> Dict[Tuple, float]:
    """Local over global time stepping where both were run: the same
    configuration, machine, ranks and mesh -- told by its cell count --
    once with one cluster and once with several.  Per simulated second, so
    two runs of different length still compare.  Keyed by (arch,
    configuration, ranks, cells)."""
    pairs: Dict[Tuple, Dict[bool, Run]] = {}
    for run in runs:
        if not run.simulated or not run.seconds:
            continue
        key = (run.arch, run.configuration, run.ranks, round(run.cells))
        pairs.setdefault(key, {})[run.lts] = run
    out = {}
    for key, both in pairs.items():
        if True in both and False in both:
            gts, lts = both[False], both[True]
            out[key] = ((gts.seconds / gts.simulated)
                        / (lts.seconds / lts.simulated))
    return out
