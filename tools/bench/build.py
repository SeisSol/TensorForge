# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Turn a build unit into one binary, and say what the compiler said about it.

Deliberately not `tests/harness/toolchain.py`. That one compiles with no
optimisation flag at all, which is right where the question is whether the
numbers come out correct and wrong where the question is how long they take:
under `hipcc` and `icpx` it means `-O0`, and a measurement of `-O0` code is a
measurement of nothing anybody runs. `tools/register_usage.py` already passes
`-O3` itself for the same reason.

`-DNDEBUG` comes with it, which turns `CHECK_ERR` into a no-op -- so a binary
from this path checks nothing, and correctness has to have been established on
the harness path before a number from here means anything.

## One object per workload, dropped on failure

A corpus contains cases a given toolchain refuses; the snapshots record several
per backend. Compiling each workload separately and linking the survivors keeps
one refusal from costing the other sixty measurements, and records which
refused and why -- which is a result rather than an accident, and is the same
arrangement `lanes.search` uses for a candidate that does not build.

## What the compiler is asked on the way past

The resource remarks cost nothing extra: `-Xptxas=-v`, ROCm's
`-Rpass-analysis=kernel-resource-usage`, IGC's spill warnings. The parsers for
all three already exist in `tools/register_usage.py` and are imported rather
than written again, so registers, spills and occupancy land in the manifest
beside the timings without a second compilation.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
for extra in (ROOT / 'src', ROOT / 'tests', ROOT / 'tools'):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from harness import driver_bench                                # noqa: E402
from register_usage import parse_igc, parse_ptxas, parse_remarks  # noqa: E402
from tensorforge.analysis.cost import list_cost                 # noqa: E402
from tensorforge.common.context import Context, Options         # noqa: E402
from tensorforge.generators import lanes                        # noqa: E402
from tensorforge.generators.generator import Generator          # noqa: E402

from suite import BuildUnit, Workload                           # noqa: E402

INCLUDE = ROOT / 'src' / 'tensorforge' / 'include'

#: Where built binaries live.  Per user and content-addressed, so a rerun of an
#: unchanged unit skips the compilation entirely -- which matters because a
#: configuration sweep rebuilds the same workloads at every point that did not
#: change.
CACHE = Path(os.environ.get(
    'TF_BENCH_CACHE', Path.home() / '.cache' / 'tensorforge-bench'))


@dataclass(frozen=True)
class Compiler:
    """One toolchain: how to compile an object, how to link, how to read it.

    `aux` is the runtime translation unit `tensorforge_aux` lives in, which is
    per-language and has to be linked in even though nothing here calls it
    directly: the generated launcher does, through `CHECK_ERR`.
    """
    name: str
    env: str
    default: str
    aux: str
    parse: object

    def compile_flags(self, arch: str) -> List[str]:
        raise NotImplementedError

    def link_flags(self, arch: str) -> List[str]:
        return self.compile_flags(arch)

    def source_suffix(self) -> str:
        return '.cpp'


@dataclass(frozen=True)
class Nvcc(Compiler):
    def compile_flags(self, arch):
        return ['-std=c++17', '-O3', '-DNDEBUG', f'-arch={arch}',
                '--expt-relaxed-constexpr', '-Xptxas=-v', '-lineinfo']

    def source_suffix(self):
        return '.cu'


@dataclass(frozen=True)
class Hipcc(Compiler):
    def compile_flags(self, arch):
        return ['-std=c++17', '-O3', '-DNDEBUG', f'--offload-arch={arch}',
                '-Rpass-analysis=kernel-resource-usage', '-g1']


@dataclass(frozen=True)
class Icpx(Compiler):
    def compile_flags(self, arch):
        # Ahead of time.  A JIT build never reaches IGC, so it reports nothing
        # about registers or spills -- and it also moves the first kernel
        # launch's cost into the measurement, where a warm-up cannot reach it
        # because the compilation happens once per process and not once per
        # launcher.
        return ['-fsycl', '-std=c++17', '-O3', '-DNDEBUG',
                '-fsycl-targets=spir64_gen',
                '-Xsycl-target-backend', f'-device {arch}']


@dataclass(frozen=True)
class Acpp(Compiler):
    def compile_flags(self, arch):
        return ['-std=c++17', '-O3', '-DNDEBUG', f'--acpp-targets={arch}']


#: Keyed by the *generator* backend, since that is what a `TargetSpec` carries.
#: `esimd` and `oneapi` are one toolchain and two code generators, which is
#: exactly why they are separate build units: the binaries differ, the compiler
#: does not.
COMPILERS: Dict[str, Compiler] = {
    'cuda': Nvcc('cuda', 'TF_NVCC', 'nvcc', 'tensorforge_aux.cu', parse_ptxas),
    'hip': Hipcc('hip', 'TF_HIPCC', 'hipcc', 'tensorforge_aux.cpp',
                 parse_remarks),
    'oneapi': Icpx('oneapi', 'TF_ICPX', 'icpx', 'tensorforge_aux_sycl.cpp',
                   parse_igc),
    'esimd': Icpx('esimd', 'TF_ICPX', 'icpx', 'tensorforge_aux_sycl.cpp',
                  parse_igc),
    'acpp': Acpp('acpp', 'TF_ACPP', 'acpp', 'tensorforge_aux_sycl.cpp',
                 lambda err: {}),
}


def compiler_binary(compiler: Compiler) -> Optional[str]:
    return (os.environ.get(compiler.env)
            or shutil.which(compiler.default))


@dataclass
class WorkloadBuild:
    """What became of one workload in one unit."""
    name: str
    origin: str
    symbol: Optional[str] = None
    obj: Optional[Path] = None
    #: Registers, spills, occupancy -- whatever the toolchain volunteered.
    static: Dict[str, int] = field(default_factory=dict)
    cost: Optional[object] = None
    lanes: Optional[int] = None
    error: str = ''

    @property
    def ok(self) -> bool:
        return self.obj is not None and not self.error


@dataclass
class UnitBuild:
    unit: BuildUnit
    exe: Optional[Path]
    workloads: List[WorkloadBuild]
    error: str = ''

    @property
    def usable(self) -> List[WorkloadBuild]:
        return [w for w in self.workloads if w.ok]


# ----------------------------------------------------------------------
# Generation
# ----------------------------------------------------------------------

def generate(workload: Workload, unit: BuildUnit) -> Tuple[Optional[str],
                                                           WorkloadBuild]:
    """`(translation unit source, record)` for one workload in one unit.

    Generation is not thread-safe -- `Context.peak_pressure` is per-context but
    the lane deduction reaches into module state -- so callers run this
    serially and parallelise the compilation, which is the slow part anyway.
    """
    record = WorkloadBuild(name=workload.name, origin=workload.origin)
    options = unit.config.options or Options()
    try:
        descrs = workload.descrs()
        ctx = Context(arch=unit.target.arch, backend=unit.target.backend,
                      fp_type=unit.datatype, options=options)
        config = lanes.deduce(descrs, ctx, ceiling=unit.config.lane_ceiling)
        gen = Generator(descrs, ctx, attrs=workload.attrs, lanes=config)
        with contextlib.redirect_stdout(io.StringIO()):
            gen.generate()
        headers = list(ctx.get_vm().get_headers()) + list(
            gen.get_helper_headers())
        includes = '\n'.join(f'#include "{h}"' for h in headers)
        src = driver_bench.emit_workload_tu(
            gen, unit.target.backend, workload.name, includes)
    except Exception as exc:                      # noqa: BLE001
        record.error = f'{type(exc).__name__}: {exc}'
        return None, record

    record.symbol = f'kernel_{gen.get_base_name()}'
    record.lanes = config.num_threads
    record.cost = list_cost(workload.descrs(), batch=1,
                            datatype=unit.datatype)
    return src, record


# ----------------------------------------------------------------------
# Compilation
# ----------------------------------------------------------------------

def _digest(unit: BuildUnit, sources: Dict[str, str], driver: str) -> str:
    h = hashlib.sha256()
    h.update(unit.label.encode())
    for name in sorted(sources):
        h.update(name.encode())
        h.update(sources[name].encode())
    h.update(driver.encode())
    return h.hexdigest()[:16]


def _compile_one(cc: str, compiler: Compiler, arch: str, src: Path,
                 obj: Path) -> Tuple[bool, str]:
    cmd = [cc, *compiler.compile_flags(arch), '-c', '-I', str(INCLUDE),
           '-I', str(src.parent), str(src), '-o', str(obj)]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        (src.with_suffix('.log')).write_text(
            ' '.join(cmd) + '\n\n' + proc.stdout + '\n' + proc.stderr)
        head = [ln for ln in proc.stderr.splitlines() if 'error' in ln.lower()]
        return False, '\n'.join(head[:3]) or 'compile failed'
    return True, proc.stderr


def build(unit: BuildUnit, cache: Path = CACHE,
          jobs: Optional[int] = None) -> UnitBuild:
    """Generate, compile, link.  Returns what exists, and what did not.

    A unit whose workloads all refuse is not an error either -- it comes back
    with `exe=None` and every refusal recorded, because "this configuration
    does not build on this target" is a finding and a traceback is not.
    """
    compiler = COMPILERS.get(unit.target.backend)
    if compiler is None:
        return UnitBuild(unit, None, [], f'no recipe for backend '
                                        f'{unit.target.backend!r}')
    cc = compiler_binary(compiler)
    if cc is None:
        return UnitBuild(unit, None, [],
                         f'{compiler.default} not found; set ${compiler.env}')

    sources: Dict[str, str] = {}
    records: List[WorkloadBuild] = []
    for workload in unit.workloads:
        src, record = generate(workload, unit)
        records.append(record)
        if src is not None:
            sources[workload.name] = src

    if not sources:
        return UnitBuild(unit, None, records, 'nothing generated')

    driver = driver_bench.emit_driver(sorted(sources), unit.target.backend)
    out = cache / unit.label / _digest(unit, sources, driver)
    exe = out / 'bench'
    if exe.exists():
        # The cache is keyed on the source text, so a hit is the same binary.
        # The static figures are re-read from the stored logs rather than
        # recompiled: they are what the compiler said about this exact source.
        for record in records:
            log = out / f'{driver_bench.slug(record.name)}.remarks'
            if log.exists():
                record.static = compiler.parse(log.read_text())
                record.obj = out / f'{driver_bench.slug(record.name)}.o'
        return UnitBuild(unit, exe, records)

    out.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}
    for name, src in sources.items():
        path = out / f'{driver_bench.slug(name)}{compiler.source_suffix()}'
        path.write_text(src)
        paths[name] = path
    driver_path = out / f'driver{compiler.source_suffix()}'
    driver_path.write_text(driver)

    by_name = {r.name: r for r in records}

    def compile_workload(item):
        name, path = item
        obj = path.with_suffix('.o')
        ok, message = _compile_one(cc, compiler, unit.target.arch, path, obj)
        return name, obj, ok, message

    with ThreadPoolExecutor(max_workers=jobs or (os.cpu_count() or 1)) as pool:
        for name, obj, ok, message in pool.map(compile_workload,
                                               sorted(paths.items())):
            record = by_name[name]
            if not ok:
                record.error = message
                continue
            record.obj = obj
            record.static = compiler.parse(message)
            (out / f'{driver_bench.slug(name)}.remarks').write_text(message)

    objects = [str(r.obj) for r in records if r.ok]
    if not objects:
        return UnitBuild(unit, None, records, 'every workload refused to build')

    # The driver is rewritten for the survivors only: a symbol declared in the
    # table but never defined is a link error, and losing the whole binary to
    # one workload is exactly what compiling them apart was meant to avoid.
    driver_path.write_text(
        driver_bench.emit_driver([r.name for r in records if r.ok],
                                 unit.target.backend))
    link = [cc, *compiler.link_flags(unit.target.arch), '-I', str(INCLUDE),
            str(driver_path), *objects, str(INCLUDE / compiler.aux),
            '-o', str(exe)]
    proc = subprocess.run(link, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        (out / 'link.log').write_text(
            ' '.join(link) + '\n\n' + proc.stdout + '\n' + proc.stderr)
        return UnitBuild(unit, None, records,
                         f'link failed; see {out / "link.log"}')
    return UnitBuild(unit, exe, records)
