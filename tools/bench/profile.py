# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Drive the vendor's profiler over a suite, and normalise just enough of it.

The same binary `run.py` times, run in its `profile` mode: a fixed number of
dispatches after the warm-up and no timing of its own, so the tool attached to
it is the only clock. One workload per process, which keeps a counter
collection small and keeps `rocprof-compute` -- which re-runs the whole
application once per counter pass -- to a runtime measured in seconds.

## What is normalised, and what is not

A small common set: duration, DRAM bytes each way, L2 bytes, achieved
occupancy, launch geometry. Everything else stays in the vendor's own file,
which is kept beside the normalised rows rather than parsed. Normalising more
would mean claiming that `dram__bytes_read.sum` and `FETCH_SIZE` are the same
quantity in more places than they are, and the failure mode of that claim is a
table that looks comparable and is not.

Nothing here makes a cross-vendor comparison legitimate. `run.py` is the tool
for that, because a launch counted with a steady clock is the same measurement
everywhere and a hardware counter is not.

## The number worth the trouble

The counters give bytes that moved; :mod:`tensorforge.analysis.cost` gives
bytes that had to. Their ratio is the traffic amplification, and for a batched
small-operator kernel it is the whole question: an `Addressing.NONE` operator
matrix is read by every block and should be an L2 hit, so a ratio near one says
the cache did its job and a ratio near the block count says it did not. No
timing run can tell those apart -- they differ in where the bytes came from,
not in how many arrived.

## Flags are probed, not trusted

Vendor command lines move between releases. Every flag this builds lives in one
adapter, `--dry-run` prints the exact command without running it, and where the
vendor ships a validator (`rocprofv3-avail pmc-check`, `ncu --query-metrics`)
the metric set is checked against the installed tool before a collection
starts. A metric the tool does not know is dropped with a note rather than
failing the run: a partial collection is worth having and a rejected command
line is not.
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import io
import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for extra in (str(HERE), str(ROOT / 'src'), str(ROOT / 'tests')):
    if extra not in sys.path:
        sys.path.insert(0, extra)

import build as bench_build                                     # noqa: E402
import run as bench_run                                         # noqa: E402
import suite as suite_mod                                       # noqa: E402
from tensorforge.analysis.cost import list_cost                 # noqa: E402


@dataclass(frozen=True)
class Metric:
    """One normalised quantity and the vendor metric that supplies it.

    `scale` converts the vendor's unit to the normalised one -- bytes for
    traffic, nanoseconds for time, a fraction for occupancy. Kept beside the
    name because the unit is part of what the name means: ROCm's `FETCH_SIZE`
    is kilobytes and Nsight's `dram__bytes_read.sum` is bytes, and a table
    mixing them silently is off by 1024 in a direction nobody notices.
    """
    key: str
    expr: str
    scale: float = 1.0
    note: str = ''


#: The normalised set.  Deliberately short; see the module docstring.
NORMALISED = ('duration_ns', 'dram_read_bytes', 'dram_write_bytes',
              'l2_bytes', 'occupancy', 'grid_size', 'block_size')


@dataclass
class Collection:
    """What one profiler invocation produced."""
    workload: str
    symbol: Optional[str]
    commands: List[List[str]]
    outdir: Path
    rows: List[Dict] = field(default_factory=list)
    raw: List[str] = field(default_factory=list)
    dropped: List[str] = field(default_factory=list)
    error: str = ''


# ----------------------------------------------------------------------
# Long-format CSV
# ----------------------------------------------------------------------

#: Column headings that carry the kernel, the metric and the value.  Both
#: `ncu --csv` and `rocprofv3 --output-format csv` emit *long* format -- one
#: row per kernel per metric -- which is why one reader serves both.  Matched
#: case-insensitively and by substring, because the exact heading has changed
#: between releases of each while the shape has not.
_KERNEL_COLUMNS = ('kernel name', 'kernel_name', 'kernel')
_METRIC_COLUMNS = ('metric name', 'counter_name', 'metric')
_VALUE_COLUMNS = ('metric value', 'counter_value', 'value')


def _pick(header: Sequence[str], candidates: Sequence[str]) -> Optional[int]:
    lowered = [h.strip().strip('"').lower() for h in header]
    for candidate in candidates:
        for i, name in enumerate(lowered):
            if name == candidate:
                return i
    for candidate in candidates:
        for i, name in enumerate(lowered):
            if candidate in name:
                return i
    return None


def read_long_csv(text: str) -> Tuple[Dict[str, Dict[str, float]], str]:
    """`{kernel: {metric: value}}` from a long-format CSV, plus what went wrong.

    Tolerant on purpose. A profiler's CSV carries banner lines, a units row,
    thousands separators and occasionally an `n/a`; refusing the file over any
    of those would throw away a collection that took minutes to gather and that
    nobody can reproduce without the machine.
    """
    rows = list(csv.reader(io.StringIO(text)))
    header_at = None
    for i, row in enumerate(rows[:40]):
        if _pick(row, _METRIC_COLUMNS) is not None and \
                _pick(row, _VALUE_COLUMNS) is not None:
            header_at = i
            break
    if header_at is None:
        return {}, 'no header row carrying a metric and a value column'

    header = rows[header_at]
    k_at = _pick(header, _KERNEL_COLUMNS)
    m_at = _pick(header, _METRIC_COLUMNS)
    v_at = _pick(header, _VALUE_COLUMNS)

    out: Dict[str, Dict[str, float]] = {}
    skipped = 0
    for row in rows[header_at + 1:]:
        if len(row) <= max(m_at, v_at, k_at or 0):
            continue
        raw = row[v_at].strip().replace(',', '')
        try:
            value = float(raw)
        except ValueError:
            skipped += 1
            continue
        kernel = row[k_at].strip() if k_at is not None else '(unnamed)'
        out.setdefault(kernel, {})[row[m_at].strip()] = value
    note = f'{skipped} non-numeric values skipped' if skipped else ''
    return out, note


def normalise(per_kernel: Dict[str, Dict[str, float]],
              metrics: Sequence[Metric]) -> List[Dict]:
    """Vendor metric names to the common set, keeping the rest verbatim."""
    by_expr = {m.expr: m for m in metrics}
    out = []
    for kernel, values in sorted(per_kernel.items()):
        row: Dict[str, object] = {'kernel': kernel, 'vendor': dict(values)}
        for expr, value in values.items():
            metric = by_expr.get(expr)
            if metric is not None:
                row[metric.key] = value * metric.scale
        out.append(row)
    return out


# ----------------------------------------------------------------------
# Adapters
# ----------------------------------------------------------------------

@dataclass
class Profiler:
    name: str
    env: str
    default: str
    metrics: Tuple[Metric, ...]
    #: Whether the tool can be told which kernel to collect at collection time.
    #: Where it cannot, the narrowing comes from the invocation instead: the
    #: driver runs one workload per process, so one kernel is dispatched
    #: whatever the binary holds. The filter is a second line and not the
    #: first.
    kernel_filter: bool = True

    def binary(self) -> Optional[str]:
        return os.environ.get(self.env) or shutil.which(self.default)

    def invocation(self) -> str:
        """The path to run, or the bare name when the tool is not installed.

        A dry run has to print something a reader can check, and it is most
        often run on a machine that has neither the profiler nor the compiler.
        """
        return self.binary() or self.default

    def version(self) -> str:
        exe = self.binary()
        if exe is None:
            return ''
        proc = subprocess.run([exe, '--version'], capture_output=True,
                              text=True, check=False)
        lines = (proc.stdout or proc.stderr).strip().splitlines()
        return lines[0] if lines else ''

    def validate(self, metrics: Sequence[Metric]
                 ) -> Tuple[List[Metric], List[str]]:
        """Split the metric set into what this installation knows and what not.

        The default answer is "all of them, unverified": a profiler that ships
        no enumerator gets the benefit of the doubt, and a metric it rejects
        surfaces as a failed command with the tool's own message, which is
        more useful than a guess made here.
        """
        return list(metrics), []

    def commands(self, exe: Path, workload: str, batch: int, iters: int,
                 warmup: int, symbol: Optional[str], outdir: Path,
                 metrics: Sequence[Metric]) -> List[List[str]]:
        """The invocations to run, in order.

        A list rather than one command because not every tool collects in a
        single step: VTune collects and then reports, and `rocprof-compute`
        adds a roofline pass beside the counter pass. A caller runs them in
        sequence and stops at the first failure.
        """
        raise NotImplementedError

    def collect(self, outdir: Path,
                metrics: Sequence[Metric]) -> Tuple[List[Dict], List[str], str]:
        raise NotImplementedError


class NsightCompute(Profiler):
    """`ncu`.

    Counter collection needs permission: without `CAP_PERFMON`, or with the
    driver's `NVreg_RestrictProfilingToAdminUsers` left at its default, `ncu`
    reports `ERR_NVGPUCTRPERM` and collects nothing. That is a machine
    configuration and not something a flag here can work around, which is the
    reason `run.py` exists as a path that never needs it.

    Kernel replay is the default and re-runs a kernel once per pass. A
    cooperative launch cannot be replayed that way, so the barrier cases need
    `--replay-mode application`, and the adapter says so rather than producing
    an empty report.
    """

    def commands(self, exe, workload, batch, iters, warmup, symbol, outdir,
                 metrics):
        args = [self.invocation(), '--csv', '--page', 'raw',
                '--target-processes', 'all',
                '--metrics', ','.join(m.expr for m in metrics),
                '--launch-skip', str(warmup), '--launch-count', str(iters),
                '--log-file', str(outdir / 'ncu.csv')]
        if symbol:
            args += ['--kernel-name', f'regex:{symbol}']
        return [args + [str(exe), 'profile', workload, str(batch), str(iters),
                        str(warmup), '0']]

    def validate(self, metrics):
        exe = self.binary()
        if exe is None:
            return list(metrics), []
        proc = subprocess.run([exe, '--query-metrics'], capture_output=True,
                              text=True, check=False)
        if proc.returncode != 0 or not proc.stdout.strip():
            return list(metrics), []
        known = proc.stdout
        good = [m for m in metrics if m.expr.split('.')[0] in known]
        dropped = [f'{m.expr}: not offered by this ncu'
                   for m in metrics if m not in good]
        return good, dropped

    def collect(self, outdir, metrics):
        path = outdir / 'ncu.csv'
        if not path.exists():
            return [], [], 'ncu wrote no CSV'
        per_kernel, note = read_long_csv(path.read_text())
        return normalise(per_kernel, metrics), [str(path)], note


class RocprofV3(Profiler):
    """`rocprofv3`.

    `--kernel-iteration-range` is what makes the warm-up skippable: the driver
    dispatches `warmup + iters` times and the range names the tail, so the
    launch that pays for the occupancy query and `hipFuncSetAttribute` is not
    in the sample.

    Counters are collected in passes and the hardware has room for only so many
    at once, so a large `--pmc` set silently becomes several application runs.
    `rocprofv3-avail pmc-check` is asked first, which is the vendor's own answer
    to whether a set fits.
    """

    def commands(self, exe, workload, batch, iters, warmup, symbol, outdir,
                 metrics):
        args = [self.invocation(), '--kernel-trace', '--stats',
                '--output-format', 'csv', '-d', str(outdir), '-o', 'rocprof']
        if metrics:
            args += ['--pmc', *[m.expr for m in metrics]]
        if symbol:
            args += ['--kernel-include-regex', symbol]
        # 1-based and inclusive: the first `warmup` dispatches are skipped.
        args += ['--kernel-iteration-range', f'{warmup + 1}-{warmup + iters}']
        return [args + ['--', str(exe), 'profile', workload, str(batch),
                        str(iters), str(warmup), '0']]

    def validate(self, metrics):
        exe = shutil.which('rocprofv3-avail')
        if exe is None or not metrics:
            return list(metrics), []
        proc = subprocess.run(
            [exe, 'pmc-check', *[m.expr for m in metrics]],
            capture_output=True, text=True, check=False)
        if proc.returncode == 0:
            return list(metrics), []
        # The check reports which entries it could not satisfy; anything it
        # names is dropped and the rest is still worth collecting.
        text = (proc.stdout + proc.stderr)
        good = [m for m in metrics if m.expr not in text]
        dropped = [f'{m.expr}: rejected by rocprofv3-avail pmc-check'
                   for m in metrics if m not in good]
        if not good:
            dropped.append('pmc-check rejected the whole set; '
                           'run `rocprofv3-avail list --pmc` on this agent')
        return good, dropped

    def collect(self, outdir, metrics):
        raw, rows, notes = [], {}, []
        for path in sorted(outdir.rglob('*.csv')):
            raw.append(str(path))
            per_kernel, note = read_long_csv(path.read_text())
            if note:
                notes.append(f'{path.name}: {note}')
            for kernel, values in per_kernel.items():
                rows.setdefault(kernel, {}).update(values)
        if not rows:
            return [], raw, 'rocprofv3 wrote no counter rows'
        return normalise(rows, metrics), raw, '; '.join(notes)


class Unitrace(Profiler):
    """`unitrace`, from intel/pti-gpu.

    Level Zero metrics rather than a counter file: the tool writes its own
    report, and what is normalised out of it is the kernel timing. The deeper
    memory counters on this stack come from VTune (`gpu-hotspots`) and the
    roofline from Advisor, both of which are separate programs with their own
    output formats -- so this adapter collects what unitrace gives and names
    the other two rather than pretending to wrap them.

    An ahead-of-time build matters here more than elsewhere: a JIT build spends
    its first launch inside IGC, and no warm-up count reaches that because the
    compilation happens once per process rather than once per launcher.
    `build.py` passes `-fsycl-targets=spir64_gen` for this reason.
    """

    def commands(self, exe, workload, batch, iters, warmup, symbol, outdir,
                 metrics):
        args = [self.invocation(), '--device-timing',
                '--chrome-kernel-logging',
                '--output', str(outdir / 'unitrace')]
        if symbol:
            args += ['--include-kernels', symbol]
        return [args + ['--', str(exe), 'profile', workload, str(batch),
                        str(iters), str(warmup), '0']]

    def collect(self, outdir, metrics):
        raw = [str(p) for p in sorted(outdir.rglob('unitrace*'))]
        if not raw:
            return [], [], 'unitrace wrote no report'
        # Its CSV, where it produces one, is the same long shape; its text
        # report is not, and is kept rather than guessed at.
        rows: Dict[str, Dict[str, float]] = {}
        for path in sorted(outdir.rglob('*.csv')):
            per_kernel, _ = read_long_csv(Path(path).read_text())
            for kernel, values in per_kernel.items():
                rows.setdefault(kernel, {}).update(values)
        if not rows:
            return [], raw, ('kernel timings are in the unitrace report, '
                             'which is kept verbatim; no CSV to normalise')
        return normalise(rows, metrics), raw, ''


class Vtune(Profiler):
    """`vtune -collect gpu-hotspots`, then `vtune -report` into CSV.

    The Intel counterpart of the counter half. `unitrace` is the timing tool on
    this stack and reports its own text; the memory events that make a traffic
    amplification computable come from VTune, and they come out of a second
    invocation -- collection writes a result directory, reporting turns it into
    a table. Hence two commands rather than one.

    A roofline on Intel is Advisor's (`advisor --collect=roofline
    --profile-gpu`), not this one and not a reimplementation: it runs its own
    calibration and knows the stack's own ceilings. Named here rather than
    wrapped, because a wrapper that got the calibration wrong would produce a
    plot that looks like Advisor's and is not.
    """

    def commands(self, exe, workload, batch, iters, warmup, symbol, outdir,
                 metrics):
        result = outdir / 'vtune'
        program = [str(exe), 'profile', workload, str(batch), str(iters),
                   str(warmup), '0']
        collect = [self.invocation(), '-collect', 'gpu-hotspots',
                   '-knob', 'profiling-mode=source-analysis',
                   '-knob', 'source-analysis=mem-latency',
                   '-result-dir', str(result), '--', *program]
        # The report step is what produces something readable; `-format csv`
        # lands in the same long shape ncu and rocprofv3 use.
        report = [self.invocation(), '-report', 'hw-events',
                  '-result-dir', str(result), '-format', 'csv',
                  '-csv-delimiter', 'comma',
                  '-report-output', str(outdir / 'vtune.csv')]
        return [collect, report]

    def collect(self, outdir, metrics):
        path = outdir / 'vtune.csv'
        if not path.exists():
            return [], [], 'vtune wrote no CSV report'
        per_kernel, note = read_long_csv(path.read_text())
        return normalise(per_kernel, metrics), [str(path)], note


#: Metric sets, per vendor.  These are the names to check first on a new
#: driver or ROCm release -- `--dry-run` prints the command they build, and
#: `probe` asks the installed tool which of them it knows.
PROFILERS: Dict[str, Profiler] = {
    'ncu': NsightCompute('ncu', 'TF_NCU', 'ncu', (
        Metric('duration_ns', 'gpu__time_duration.sum'),
        Metric('dram_read_bytes', 'dram__bytes_read.sum'),
        Metric('dram_write_bytes', 'dram__bytes_write.sum'),
        Metric('l2_bytes', 'lts__t_bytes.sum'),
        Metric('occupancy',
               'sm__warps_active.avg.pct_of_peak_sustained_active',
               scale=0.01, note='reported as a percentage'),
        Metric('grid_size', 'launch__grid_size'),
        Metric('block_size', 'launch__block_size'),
    )),
    'rocprofv3': RocprofV3('rocprofv3', 'TF_ROCPROFV3', 'rocprofv3', (
        Metric('dram_read_bytes', 'FETCH_SIZE', scale=1024.0,
               note='ROCm reports kilobytes'),
        Metric('dram_write_bytes', 'WRITE_SIZE', scale=1024.0,
               note='ROCm reports kilobytes'),
        Metric('occupancy', 'OccupancyPercent', scale=0.01),
    )),
    'unitrace': Unitrace('unitrace', 'TF_UNITRACE', 'unitrace', (
        # unitrace's own CSV, where it emits one, names the kernel duration
        # this way.  The deeper counters are VTune's; see the class docstring.
        Metric('duration_ns', 'Kernel Time (ns)'),
    )),
    'vtune': Vtune('vtune', 'TF_VTUNE', 'vtune', kernel_filter=False,
                   metrics=(
        Metric('duration_ns', 'Elapsed Time', scale=1e9,
               note='VTune reports seconds'),
        Metric('dram_read_bytes', 'GPU_MEMORY_READ_BYTES'),
        Metric('dram_write_bytes', 'GPU_MEMORY_WRITE_BYTES'),
        Metric('l2_bytes', 'GPU_L3_BYTES'),
    )),
}

#: Which profiler answers for which device vendor, when none was named.
#: Intel has two and the default is the one that needs no licence; `--tool
#: vtune` is the way to the memory counters.
BY_VENDOR = {'nvidia': 'ncu', 'amd': 'rocprofv3', 'intel': 'unitrace'}


# ----------------------------------------------------------------------
# Driving one suite
# ----------------------------------------------------------------------

def profile_unit(unit_build: bench_build.UnitBuild,
                 device: bench_run.Device, profiler: Profiler,
                 metrics: Sequence[Metric], args) -> List[Collection]:
    out: List[Collection] = []
    by_name = {w.name: w for w in unit_build.unit.workloads}
    # A dry run reports on everything that *generated*, which needs no
    # compiler: the point of printing a command without running it is to check
    # it on a machine that has the profiler, which is rarely the machine that
    # has the toolchain.
    exe = unit_build.exe or Path('<unbuilt>')
    for record in unit_build.workloads:
        if args.dry_run:
            if record.symbol is None:
                continue
        elif not record.ok:
            continue
        outdir = (args.out / unit_build.unit.label /
                  bench_build.driver_bench.slug(record.name))
        if not args.dry_run:
            outdir.mkdir(parents=True, exist_ok=True)
        commands = profiler.commands(
            exe, record.name, args.batch, args.iters, args.warmup,
            record.symbol, outdir, metrics)
        collection = Collection(workload=record.name, symbol=record.symbol,
                                commands=commands, outdir=outdir)
        if args.dry_run:
            out.append(collection)
            continue

        env = os.environ.copy()
        if device.vendor == 'nvidia':
            env['CUDA_VISIBLE_DEVICES'] = str(device.index)
        elif device.vendor == 'amd':
            env['HIP_VISIBLE_DEVICES'] = str(device.index)
        log = []
        for step, command in enumerate(commands):
            try:
                proc = subprocess.run(command, capture_output=True, text=True,
                                      timeout=args.timeout, check=False,
                                      env=env)
            except subprocess.TimeoutExpired:
                collection.error = (f'step {step + 1} timed out after '
                                    f'{args.timeout}s')
                break
            log.append(' '.join(command) + '\n\n' + proc.stdout + '\n'
                       + proc.stderr)
            if proc.returncode != 0:
                tail = (proc.stderr.strip().splitlines()
                        or ['(no output)'])[-2:]
                collection.error = (f'step {step + 1} exited '
                                    f'{proc.returncode}: ' + ' / '.join(tail))
                break
        (outdir / 'tool.log').write_text('\n\n'.join(log))

        rows, raw, note = profiler.collect(outdir, metrics)
        collection.rows, collection.raw = rows, [*raw, str(outdir / 'tool.log')]
        if note:
            collection.dropped.append(note)

        # The join that makes the counters mean something: measured bytes
        # against the bytes the operation could not avoid.
        cost = list_cost(by_name[record.name].descrs(), batch=args.batch,
                         datatype=unit_build.unit.datatype)
        for row in rows:
            row['compulsory_bytes'] = cost.bytes
            row['flops'] = cost.flops
            measured = ((row.get('dram_read_bytes') or 0.0)
                        + (row.get('dram_write_bytes') or 0.0))
            if cost.bytes and measured:
                row['traffic_amplification'] = measured / cost.bytes
            if row.get('duration_ns'):
                row['gflops'] = cost.flops / row['duration_ns']
        out.append(collection)
    return out


def report(collections: Sequence[Collection], dry_run: bool) -> None:
    if dry_run:
        for c in collections:
            print(f'# {c.workload}')
            for command in c.commands:
                print('  ' + ' '.join(command))
        return

    good = [(c, r) for c in collections for r in c.rows]
    if good:
        head = (f'{"workload":28s} {"kernel":22s} {"dur ns":>10s} '
                f'{"DRAM MB":>9s} {"compulsory MB":>14s} {"x":>6s} '
                f'{"GFLOP/s":>9s} {"occ":>5s}')
        print(head)
        print('-' * len(head))
        for c, r in good:
            measured = ((r.get('dram_read_bytes') or 0.0)
                        + (r.get('dram_write_bytes') or 0.0))
            print(f'{c.workload[:28]:28s} {str(r["kernel"])[:22]:22s} '
                  f'{(r.get("duration_ns") or 0):10.1f} '
                  f'{measured / 1e6:9.2f} '
                  f'{(r.get("compulsory_bytes") or 0) / 1e6:14.2f} '
                  f'{(r.get("traffic_amplification") or 0):6.2f} '
                  f'{(r.get("gflops") or 0):9.1f} '
                  f'{(r.get("occupancy") or 0):5.2f}')

    failed = [c for c in collections if c.error]
    if failed:
        print(f'\n{len(failed)} collections failed:')
        for c in failed:
            print(f'  {c.workload}: {c.error}')
    notes = {n for c in collections for n in c.dropped}
    for note in sorted(notes):
        print(f'note: {note}')


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('suite', type=Path, nargs='?')
    ap.add_argument('--tool', choices=sorted(PROFILERS), default=None)
    ap.add_argument('--probe', action='store_true',
                    help='report which profilers are here and which metrics '
                         'they know, and run nothing')
    ap.add_argument('--dry-run', action='store_true',
                    help='print the command lines without running them')
    ap.add_argument('--backend', default=None)
    ap.add_argument('--arch', default=None)
    ap.add_argument('--workloads', default='*')
    ap.add_argument('--configs', default=None)
    ap.add_argument('--batch', type=int, default=65536)
    ap.add_argument('--iters', type=int, default=3)
    ap.add_argument('--warmup', type=int, default=5)
    ap.add_argument('--metrics', default=None,
                    help='comma-separated vendor metric names, replacing the '
                         'default set for the chosen tool')
    ap.add_argument('--timeout', type=float, default=1800.0)
    ap.add_argument('--jobs', type=int, default=os.cpu_count() or 1)
    ap.add_argument('--out', type=Path, default=Path('profile-out'))
    args = ap.parse_args()

    devices = bench_run.detect_devices(args.backend, args.arch)

    if args.probe:
        for name, profiler in sorted(PROFILERS.items()):
            exe = profiler.binary()
            if exe is None:
                print(f'{name:11s} not found (set ${profiler.env})')
                continue
            good, dropped = profiler.validate(profiler.metrics)
            print(f'{name:11s} {exe}')
            print(f'            {profiler.version()}')
            print(f'            {len(good)}/{len(profiler.metrics)} '
                  f'default metrics accepted')
            for message in dropped:
                print(f'            - {message}')
        for device in devices:
            print(f'device      {device.vendor} {device.spec.arch} '
                  f'({device.name}) -> {BY_VENDOR.get(device.vendor, "?")}')
        return 0

    if args.suite is None:
        ap.error('a suite is required unless --probe is given')

    if not devices and not args.dry_run:
        print('no device detected; pass --arch to build, --dry-run to look',
              file=sys.stderr)
        return 2
    device = devices[0] if devices else bench_run.Device(
        suite_mod.TargetSpec(args.backend or 'cuda', args.arch or 'sm_80'),
        'nvidia', 0, 'none')

    tool = args.tool or BY_VENDOR.get(device.vendor)
    profiler = PROFILERS.get(tool)
    if profiler is None:
        print(f'no profiler for vendor {device.vendor!r}; pass --tool',
              file=sys.stderr)
        return 2
    if profiler.binary() is None and not args.dry_run:
        print(f'{profiler.default} not found; set ${profiler.env}, or use '
              f'tools/bench/run.py, which needs no profiler', file=sys.stderr)
        return 2

    metrics = profiler.metrics
    if args.metrics:
        metrics = tuple(Metric(key=expr, expr=expr)
                        for expr in args.metrics.split(','))
    dropped: List[str] = []
    if not args.dry_run:
        metrics, dropped = profiler.validate(metrics)
    for message in dropped:
        print(f'note: {message}', file=sys.stderr)

    spec = suite_mod.load(args.suite)
    if args.workloads != '*':
        spec = replace(spec, workloads=tuple(
            w for w in spec.workloads
            if fnmatch.fnmatch(w.name, args.workloads)))
    if args.configs:
        wanted = {c.strip() for c in args.configs.split(',')}
        spec = replace(spec, configs=tuple(
            c for c in spec.configs if c.label in wanted))
    if not spec.workloads or not spec.configs:
        print('nothing left after filtering', file=sys.stderr)
        return 2

    if not args.dry_run:
        args.out.mkdir(parents=True, exist_ok=True)
    collections: List[Collection] = []
    for unit in suite_mod.expand(spec, [device.spec]):
        built = bench_build.build(unit, jobs=args.jobs)
        if built.exe is None and not args.dry_run:
            print(f'{unit.label}: {built.error or "no binary"}',
                  file=sys.stderr)
            continue
        collections.extend(
            profile_unit(built, device, profiler, metrics, args))

    report(collections, args.dry_run)

    if not args.dry_run:
        blob = {
            'tool': profiler.name,
            'version': profiler.version(),
            'device': asdict(device),
            'batch': args.batch, 'iters': args.iters, 'warmup': args.warmup,
            'metrics': [asdict(m) for m in metrics],
            'dropped': dropped,
            'collections': [
                {**{k: v for k, v in asdict(c).items() if k != 'outdir'},
                 'outdir': str(c.outdir)}
                for c in collections],
        }
        (args.out / 'profile.json').write_text(
            json.dumps(blob, indent=2, default=str))
        print(f'\nwritten: {args.out / "profile.json"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
