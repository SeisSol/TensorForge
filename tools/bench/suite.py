# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""What to measure: a suite, and how one expands into things to build and run.

A suite is a Python module, the same shape a test case is, for the same reason:
the corpus it draws from is Python, the descriptor lists it may inline are
Python objects, and a YAML spec would have ended up carrying Python expressions
as strings.  A suite module defines `NAME` and `workloads()`, optionally
`TARGETS`, `BATCHES` and `CONFIGS`.

## Why a workload carries a factory and not a list

`MultilinearDescr.__init__` calls `set_data_flow_direction` on every operand,
so a descriptor list is not inert: building from one twice is fine, but handing
the same objects to two configurations and comparing them is not obviously
fine, and `lanes.search` already refuses lists for this reason.  A factory
makes the repetition explicit at the call site that does it.

## Why the build unit is a configuration and not a kernel

The emitted symbol is `kernel_kernel_<md5>` where the hash covers the
descriptor list and the flag mode -- and nothing else.  Two builds that differ
only in `Options` (pipelining, wrapped loads, wide bodies) therefore emit *the
same symbol* for different code.  A profiler keys its report on that symbol, so
putting two configurations of one workload in one binary produces a report in
which they are indistinguishable.

So the unit is one binary per `(target, datatype, options, lane ceiling)`,
holding every workload in the suite.  Within a binary the configuration is
fixed and the hash is unique per workload, which is exactly the property a
report needs; across binaries the configuration is the binary's identity and is
recorded in the manifest.  The batch is a runtime argument and needs no rebuild.
"""

from __future__ import annotations

import contextlib
import fnmatch
import importlib.util
import io
import json
import os
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))

from tensorforge.common.basic_types import Addressing, Datatype  # noqa: E402
from tensorforge.common.context import Options  # noqa: E402
from tensorforge.common.matrix.boundingbox import BoundingBox  # noqa: E402
from tensorforge.common.matrix.tensor import SubTensor, Tensor  # noqa: E402
from tensorforge.generators import lanes  # noqa: E402
from tensorforge.generators.descriptions import MultilinearDescr  # noqa: E402

#: Where the case corpus lives.  `TF_TESTS` overrides, the way
#: `tools/host/tfpaths.py` does, so a suite can be run against a checkout that
#: is not the installed package.
CASES = Path(os.environ.get('TF_TESTS', ROOT / 'tests')) / 'cases'


# -- the pieces ------------------------------------------------------------- #

@dataclass(frozen=True)
class Workload:
    """One kernel to build, and the shapes it runs at.

    `attrs` is the kernel attribute dict a frontend would pass.  `None` and
    `{}` are different: `None` is what a frontend without attributes gets and
    produces a flag mask with a null check, `{}` produces no mask at all.  The
    distinction reaches the kernel name through the flag mode, so it is part of
    the workload's identity and not a detail.
    """
    name: str
    descrs: Callable[[], List]
    datatype: Datatype
    attrs: Optional[dict] = None
    #: Free-form, carried into the manifest: where this came from.
    origin: str = ''

    def __post_init__(self):
        if not callable(self.descrs):
            raise TypeError(
                f'workload {self.name!r}: descrs must be a factory returning a '
                f'fresh descriptor list, not a list -- see the module docstring')


@dataclass(frozen=True)
class Config:
    """One code-generation configuration, and the binary it identifies."""
    label: str
    options: Optional[Options] = None
    #: `lanes.DEFAULT_LANE_CEILING`, `None` for the wave width, or an integer.
    lane_ceiling: Optional[int] = lanes.DEFAULT_LANE_CEILING
    #: True where `lane_ceiling` is meant as "the wave width", which `None`
    #: also spells; the two are told apart so a manifest can say which.
    wave_wide: bool = False


@dataclass(frozen=True)
class TargetSpec:
    """A build target, before it is joined with a detected device."""
    backend: str            # cuda | hip | oneapi | acpp | esimd
    arch: str


@dataclass(frozen=True)
class Suite:
    name: str
    workloads: Tuple[Workload, ...]
    batches: Tuple[int, ...]
    configs: Tuple[Config, ...]
    targets: Optional[Tuple[TargetSpec, ...]]   # None: detect at run time
    datatypes: Tuple[Datatype, ...]
    description: str = ''


@dataclass(frozen=True)
class BuildUnit:
    """Everything that goes into one binary, plus the batches it runs at."""
    suite: str
    target: TargetSpec
    datatype: Datatype
    config: Config
    workloads: Tuple[Workload, ...]
    batches: Tuple[int, ...]

    @property
    def label(self) -> str:
        return (f'{self.suite}-{self.target.backend}-{self.target.arch}-'
                f'{self.datatype.name.lower()}-{self.config.label}')


# -- named configurations --------------------------------------------------- #

#: The configurations worth sweeping, named once so a report and a command line
#: agree on the spelling.  Each is a single axis away from `baseline`, because a
#: comparison between two configurations differing in three things measures
#: nothing.
CONFIGS: Dict[str, Config] = {
    'baseline': Config('baseline'),
    'wave': Config('wave', lane_ceiling=None, wave_wide=True),
    'pipeline': Config('pipeline', Options(enable_pipeline=True)),
    'multibuffer': Config('multibuffer',
                          Options(enable_pipeline=True,
                                  enable_multibuffer=True)),
    'wrap1': Config('wrap1', Options(enable_wrap_loads=True, wrap_distance=1)),
    'wrap2': Config('wrap2', Options(enable_wrap_loads=True, wrap_distance=2)),
    'narrow-bodies': Config('narrow-bodies', Options(wide_bodies=False)),
    # The two sides of the prologue question.  Named rather than left to the
    # vendor default so a run states which one it measured: `preload` stages
    # every batch-constant operand into shared memory once per block,
    # `no-preload` reads it from global inside the batch loop.
    'preload': Config('preload', Options(preload_globals=True)),
    'no-preload': Config('no-preload', Options(preload_globals=False)),
    # Whether the backend may re-encode a batch-constant operand for the
    # instruction that reads it.  Paired with `no-preload` because the two
    # answer the same question in opposite directions -- preloading stages the
    # operand into shared memory once per block, preparing removes the reason
    # to stage it at all -- and a run that does not say which it measured has
    # measured neither.
    'prepare': Config('prepare', Options(preload_globals=False,
                                         prepare_operands=True)),
}


# -- workload sources ------------------------------------------------------- #

def from_cases(pattern: str = '*', root: Optional[Path] = None
               ) -> List[Workload]:
    """Workloads from the correctness corpus, matched on `NAME`.

    The corpus is the right source of *shapes* and the wrong source of *load*:
    every case runs at `BATCH` two to four, and the launcher sizes its grid as
    `min(occupancy_gridsize, numElements0)`, so at those batches a run measures
    launch overhead and nothing else.  The case's own `BATCH` is therefore
    dropped here rather than carried; the suite says what to run at.

    A case whose descriptors refuse to construct is skipped, not raised: the
    corpus deliberately contains some (`beta_nonzero` asserts on `beta`), and a
    benchmark that stops on them measures nothing rather than most things.
    """
    root = root or CASES
    out: List[Workload] = []
    for path in sorted(root.rglob('*.py')):
        if path.name.startswith('_'):
            continue
        spec = importlib.util.spec_from_file_location(f'suite_{path.stem}', path)
        mod = importlib.util.module_from_spec(spec)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                spec.loader.exec_module(mod)
        except Exception:
            continue
        if not hasattr(mod, 'NAME') or not hasattr(mod, 'descr_list'):
            continue
        if not fnmatch.fnmatch(mod.NAME, pattern):
            continue
        try:
            mod.descr_list()
        except Exception:
            continue
        out.append(Workload(
            name=mod.NAME, descrs=mod.descr_list, datatype=mod.DTYPE,
            attrs=getattr(mod, 'ATTRS', None),
            origin=f'cases/{path.relative_to(root)}'))
    return out


def from_dump(path: Path, pattern: str = '*',
              datatype: Datatype = Datatype.F32) -> List[Workload]:
    """Workloads from a `tools/host/dump_descriptors.py` capture.

    The real corpus: what yateto handed the backend on an actual SeisSol
    codegen run, so the shapes, the sparsity and the chain lengths are the
    production ones rather than a test's approximation of them.

    Tensors are shared across the descriptors of one kernel by name, which is
    what makes a chain a chain -- rebuilding each descriptor's operands
    independently would give the generator two tensors where the kernel has
    one, and no temporary would ever be recognised as such.
    """
    blob = json.loads(Path(path).read_text())
    kernels = blob['all'] if 'all' in blob else blob
    addressing = {str(a): a for a in Addressing}

    out: List[Workload] = []
    for symbol, rows in sorted(kernels.items()):
        if not fnmatch.fnmatch(symbol, pattern):
            continue
        if not any(r for r in rows):
            continue

        def factory(rows=rows):
            tensors: Dict[str, Tensor] = {}

            def tensor(x):
                if x['name'] not in tensors:
                    data = x['data']
                    tensors[x['name']] = Tensor(
                        x['shape'], addressing[x['addressing']],
                        BoundingBox(list(x['tbbox'][0]), list(x['tbbox'][1])),
                        alias=x['name'], is_tmp=x['is_tmp'],
                        data=None if data is None else np.array(data),
                        datatype=datatype)
                return tensors[x['name']]

            def sub(x):
                return SubTensor(tensor(x),
                                 BoundingBox(list(x['bbox'][0]),
                                             list(x['bbox'][1])),
                                 list(x['offset']), sliced=x['sliced'])

            built = []
            for row in rows:
                if row is None:
                    continue
                built.append(MultilinearDescr(
                    dest=sub(row['dest']),
                    ops=[sub(o) for o in row['ops'] if o],
                    target=[list(t) for t, o in zip(row['target'], row['ops'])
                            if o],
                    permute=[list(p) for p, o in zip(row['permute'], row['ops'])
                             if o],
                    add=row['add']))
            return built

        out.append(Workload(name=symbol, descrs=factory, datatype=datatype,
                            origin=f'dump:{Path(path).name}'))
    return out


# -- loading and expanding -------------------------------------------------- #

def load(path: Path) -> Suite:
    """Read a suite module.

    Required: `NAME`, `workloads()`.  Optional: `DESCRIPTION`, `BATCHES`,
    `CONFIGS` (labels into the table above, or `Config` objects), `TARGETS`,
    `DATATYPES`.
    """
    path = Path(path)
    spec = importlib.util.spec_from_file_location(f'tfsuite_{path.stem}', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    for required in ('NAME', 'workloads'):
        if not hasattr(mod, required):
            raise ValueError(f'{path}: a suite needs {required}')

    configs = []
    for entry in getattr(mod, 'CONFIGS', ['baseline']):
        if isinstance(entry, Config):
            configs.append(entry)
        elif entry in CONFIGS:
            configs.append(CONFIGS[entry])
        else:
            raise ValueError(
                f'{path}: unknown configuration {entry!r}; known are '
                f'{sorted(CONFIGS)}')

    workloads = tuple(mod.workloads())
    if not workloads:
        raise ValueError(f'{path}: workloads() returned nothing')
    seen = [w.name for w in workloads]
    if len(set(seen)) != len(seen):
        dupes = sorted({n for n in seen if seen.count(n) > 1})
        raise ValueError(
            f'{path}: workload names must be unique within a suite, they are '
            f'the report keys; repeated: {dupes}')

    targets = getattr(mod, 'TARGETS', None)
    return Suite(
        name=mod.NAME,
        workloads=workloads,
        batches=tuple(getattr(mod, 'BATCHES', (1024, 8192, 65536))),
        configs=tuple(configs),
        targets=None if targets is None else tuple(targets),
        datatypes=tuple(getattr(mod, 'DATATYPES',
                                sorted({w.datatype for w in workloads},
                                       key=lambda d: d.name))),
        description=getattr(mod, 'DESCRIPTION', ''))


def expand(suite: Suite, targets: Sequence[TargetSpec]) -> List[BuildUnit]:
    """One `BuildUnit` per binary that has to exist.

    A workload is included in a unit only when its own datatype matches the
    unit's: a suite mixing F32 and F64 workloads yields two units per
    configuration rather than one binary whose kernels disagree about the
    context's floating-point type.
    """
    units: List[BuildUnit] = []
    for target in targets:
        for datatype in suite.datatypes:
            members = tuple(w for w in suite.workloads
                            if w.datatype == datatype)
            if not members:
                continue
            for config in suite.configs:
                units.append(BuildUnit(
                    suite=suite.name, target=target, datatype=datatype,
                    config=config, workloads=members, batches=suite.batches))
    return units
