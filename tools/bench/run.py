# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Time a suite on whatever device is here, without a profiler.

The number this produces is comparable across vendors, which is the one thing
the profiling path cannot offer: `ncu`, `rocprofv3` and `unitrace` measure
different quantities under the same words, and a table mixing them is a table
of three things. A launch counted with a steady clock is the same measurement
everywhere.

Two clocks per workload, because they answer different questions. Wall time
over back-to-back launches on one stream includes the launch overhead, which is
what SeisSol pays: it dispatches thousands of small kernels per timestep and
the overhead is a real term, not an artifact of the harness. Device event time
around a single launch excludes the queueing and is what an achieved
FLOP-per-second should be divided by. Both are reported; neither is called
*the* time.

## Read the batch column before the throughput column

The launcher sizes its grid `min(occupancy_gridsize, numElements0)`. Below
saturation the kernel is launch-bound and its throughput is a statement about
the runtime, not about the code. The runner therefore sweeps the batch and
prints nanoseconds per element next to the totals: where that column has
stopped falling, the grid is full and the number means what it looks like it
means. A single batch would have produced one number and no way to tell which
regime it came from.

    python3 tools/bench/run.py suites/corpus.py
    python3 tools/bench/run.py suites/corpus.py --configs baseline,wave -v
    TF_BENCH_DUMP=descriptors.json python3 tools/bench/run.py suites/seissol.py
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for extra in (str(HERE), str(ROOT / 'src'), str(ROOT / 'tests')):
    if extra not in sys.path:
        sys.path.insert(0, extra)

import build as bench_build                                     # noqa: E402
import suite as suite_mod                                       # noqa: E402
from tensorforge import interop                                 # noqa: E402
from tensorforge.analysis.cost import list_cost                 # noqa: E402


@dataclass(frozen=True)
class Device:
    """A target joined with a device to run it on."""
    spec: suite_mod.TargetSpec
    vendor: str
    index: int
    name: str


def detect_devices(explicit_backend: Optional[str],
                   explicit_arch: Optional[str]) -> List[Device]:
    """What is here, or what was asked for.

    An explicit `--arch` is taken at face value and paired with device 0: a
    cross-compile for an architecture the host does not have is a legitimate
    thing to want from the *build* half, and refusing it here would mean the
    only way to check that a configuration compiles is to own the hardware.
    Running it will fail, loudly, which is the right place for that to happen.
    """
    if explicit_arch:
        backend = explicit_backend or 'cuda'
        vendor = {'cuda': 'nvidia', 'hip': 'amd'}.get(backend, 'intel')
        return [Device(suite_mod.TargetSpec(backend, explicit_arch),
                       vendor, 0, f'requested {explicit_arch}')]

    from harness.gpu_detect import detect_all
    backends = {'nvidia': ['cuda'], 'amd': ['hip'],
                'intel': ['oneapi', 'esimd']}
    out: List[Device] = []
    for gpu in detect_all():
        for backend in backends.get(gpu.vendor, []):
            if explicit_backend and backend != explicit_backend:
                continue
            out.append(Device(suite_mod.TargetSpec(backend, gpu.arch),
                              gpu.vendor, gpu.index, gpu.name))
    return out


def run_binary(exe: Path, workload: str, batch: int, iters: int, warmup: int,
               device: Device, timeout: float) -> Dict:
    """One invocation, one workload.

    One process per measurement rather than a loop inside the binary. It costs
    a device initialization each time, and it buys two things worth more than
    that: a workload that hangs or faults takes down its own run and not the
    sweep, and the profiling path -- which wants exactly one workload per
    process so a counter collection stays small -- drives the same binary the
    same way.
    """
    env = os.environ.copy()
    if device.vendor == 'nvidia':
        env['CUDA_VISIBLE_DEVICES'] = str(device.index)
    elif device.vendor == 'amd':
        env['HIP_VISIBLE_DEVICES'] = str(device.index)
    # The index is consumed by the environment above, so the binary always
    # opens device 0 of what it can see.  Passing both would select the second
    # device of a one-device view.
    cmd = [str(exe), 'time', workload, str(batch), str(iters), str(warmup), '0']
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout, check=False, env=env)
    except subprocess.TimeoutExpired:
        return {'error': f'timeout after {timeout}s'}
    if proc.returncode != 0:
        tail = (proc.stderr.strip().splitlines() or ['(no output)'])[-3:]
        return {'error': f'exit {proc.returncode}: ' + ' / '.join(tail)}
    for line in proc.stdout.splitlines():
        line = line.strip()
        if line.startswith('{'):
            return json.loads(line)
    return {'error': 'the driver printed no measurement'}


def measure(unit_build: bench_build.UnitBuild, device: Device,
            workloads: List[suite_mod.Workload], args) -> List[Dict]:
    """Every surviving workload at every batch, joined with its cost.

    The rows come back *without* their timings: `time_rows` fills those in,
    once every unit has been built, so that the configurations can be
    measured against each other rather than one after the other.
    """
    by_name = {w.name: w for w in workloads}
    rows: List[Dict] = []
    for record in unit_build.workloads:
        base = {
            'unit': unit_build.unit.label,
            'workload': record.name,
            'origin': record.origin,
            'symbol': record.symbol,
            'backend': unit_build.unit.target.backend,
            'arch': unit_build.unit.target.arch,
            'device': device.name,
            'datatype': unit_build.unit.datatype.name,
            'config': unit_build.unit.config.label,
            'lanes': record.lanes,
            'static': record.static,
        }
        if not record.ok:
            # A refusal always carries a reason: `report` takes a row without
            # an error for a measurement, and a measurement has a batch.
            rows.append({**base, 'error': record.error or 'not built'})
            continue
        for batch in unit_build.unit.batches:
            cost = list_cost(by_name[record.name].descrs(), batch=batch,
                             datatype=unit_build.unit.datatype)
            row = {**base, 'batch': batch,
                   'flops': cost.flops, 'bytes': cost.bytes,
                   # What the operation needs once the structural zeros are
                   # taken out, beside what a dense implementation performs.
                   # Recorded and not divided by: the generator emits the
                   # dense figure today, so a rate against this one would
                   # report work nobody does.  The pair is what says how much
                   # the sparsity is worth.
                   'nonzero_flops': cost.nonzero_flops,
                   'nonzero_exact': cost.nonzero_exact,
                   # And what the built kernel performs, which is above both:
                   # a destination twenty rows wide computes on thirty-two
                   # lanes.  Recorded, not divided by -- see the manifest.
                   'executed_flops': (record.executed_flops * batch
                                      if record.executed_flops else None),
                   'transcendental': cost.transcendental,
                   'intensity': cost.intensity}
            row['_run'] = (unit_build.exe, record.name, batch, device)
            row['_cost'] = cost
            rows.append(row)
    return rows


def time_rows(rows: List[Dict], args) -> None:
    """Fill in the timings: several passes over *all* rows, keeping the best.

    A pass is a round over every row rather than a repetition of one.  The
    device's clock moves between processes and the runner starts one process
    per measurement: the same binary timed six times over
    `elastic-o4s:localFluxAll` spanned 6.3 % while a single process's own
    min-to-p90 stayed under 1 %.  Two configurations timed one after the
    other therefore differ by the clock as much as by the code -- which is
    how two builds of a *byte-identical* kernel came out 7.5 % apart and the
    difference was read as a result.

    Interleaving spreads the drift over every row instead of over the order
    they happen to be built in, and the minimum is what the machine did on
    the pass where it was not throttled.  The spread across passes is kept
    beside it, because a difference smaller than that is not a difference.
    """
    jobs = [row for row in rows if '_run' in row]
    for pass_no in range(max(1, args.repeat)):
        for row in jobs:
            exe, name, batch, device = row['_run']
            timing = run_binary(exe, name, batch, args.iters, args.warmup,
                                device, args.timeout)
            row.setdefault('_samples', []).append(timing)
        if args.verbose and args.repeat > 1:
            print(f'  pass {pass_no + 1}/{args.repeat} done', file=sys.stderr)

    for row in jobs:
        samples = row.pop('_samples', [])
        cost = row.pop('_cost')
        row.pop('_run')
        good = [t for t in samples if not t.get('error')]
        if not good:
            row.update(samples[0] if samples else
                       {'error': 'no measurement attempted'})
            continue

        def clock(t):
            return t.get('event_ns') or t.get('wall_ns') or float('inf')

        best = min(good, key=clock)
        row.update(best)
        row['passes'] = len(good)
        if len(good) > 1:
            lo, hi = clock(best), max(clock(t) for t in good)
            row['pass_spread'] = (hi / lo - 1.0) if lo else None
        # Achieved rates use the device clock where there is one and the
        # wall clock where there is not, and say which -- dividing a
        # compulsory byte count by a time that includes launch overhead
        # gives a bandwidth no memory system ever delivered.
        nanos = best.get('event_ns') or best.get('wall_ns')
        row['rate_clock'] = ('event' if best.get('event_ns')
                             else 'wall' if best.get('wall_ns') else None)
        if nanos:
            row['gflops'] = cost.flops / nanos
            row['gbytes'] = cost.bytes / nanos
            row['ns_per_element'] = nanos / row['batch']


def report(rows: List[Dict], verbose: bool) -> None:
    errors = [r for r in rows if r.get('error')]
    good = [r for r in rows if not r.get('error')]

    if good:
        # The target and the datatype are columns only where they vary.  A run
        # on two devices printed `chain_five_multiplies baseline 4096` twice
        # with different numbers and nothing saying which was which; a run on
        # one device does not need the column and reads worse with it.
        def variant(row: Dict) -> str:
            parts = []
            if len({(r['backend'], r['arch']) for r in good}) > 1:
                parts.append(f'{row["backend"]}-{row["arch"]}')
            if len({r['datatype'] for r in good}) > 1:
                parts.append(row['datatype'].lower())
            return ' '.join(parts)

        width = max((len(variant(r)) for r in good), default=0)
        column = f'{"target":{width}s} ' if width else ''
        spread = any(r.get('pass_spread') is not None for r in good)
        head = (f'{"workload":30s} {column}{"cfg":12s} {"batch":>8s} '
                f'{"wall ns":>10s} {"dev ns":>10s} {"ns/elem":>8s} '
                f'{"GFLOP/s":>9s} {"GB/s":>8s} {"AI":>6s}'
                + (f' {"+/-":>6s}' if spread else ''))
        print(head)
        print('-' * len(head))
        for r in sorted(good, key=lambda r: (r['workload'], variant(r),
                                             r['config'], r['batch'])):
            cell = f'{variant(r):{width}s} ' if width else ''
            print(f'{r["workload"][:30]:30s} {cell}{r["config"][:12]:12s} '
                  f'{r["batch"]:8d} {r.get("wall_ns", 0):10.1f} '
                  f'{(r.get("event_ns") or float("nan")):10.1f} '
                  f'{r.get("ns_per_element", 0):8.4f} '
                  f'{r.get("gflops", 0):9.1f} {r.get("gbytes", 0):8.1f} '
                  f'{(r.get("intensity") or 0):6.2f}'
                  + (f' {r["pass_spread"] * 100:5.1f}%'
                     if spread and r.get('pass_spread') is not None
                     else '       ' if spread else ''))

    if errors:
        print(f'\n{len(errors)} workloads produced no number:')
        seen: Dict[str, List[str]] = {}
        for r in errors:
            seen.setdefault(r['error'].splitlines()[0][:90], []).append(
                f'{r["workload"]}/{r["config"]}')
        for message, who in sorted(seen.items(), key=lambda x: -len(x[1])):
            print(f'  {len(who):3d}x {message}')
            if verbose:
                for name in who:
                    print(f'         {name}')


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('suite', type=Path)
    ap.add_argument('--backend', default=None,
                    help='restrict to one generator backend')
    ap.add_argument('--arch', default=None,
                    help='build for this architecture instead of detecting; '
                         'running still needs the hardware')
    ap.add_argument('--workloads', default='*',
                    help='glob over workload names')
    ap.add_argument('--configs', default=None,
                    help='comma-separated subset of the suite\'s configs')
    ap.add_argument('--batches', default=None,
                    help='comma-separated batch sizes, overriding the suite')
    ap.add_argument('--iters', type=int, default=200)
    ap.add_argument('--repeat', type=int, default=3,
                    help='passes over every row; the best is kept and the '
                         'spread across passes reported (1 to time each row '
                         'once, as before)')
    ap.add_argument('--warmup', type=int, default=10)
    ap.add_argument('--timeout', type=float, default=300.0)
    ap.add_argument('--jobs', type=int, default=os.cpu_count() or 1)
    ap.add_argument('--out', type=Path, default=None,
                    help='write results and manifest here as JSON')
    ap.add_argument('--build-only', action='store_true')
    ap.add_argument('-v', '--verbose', action='store_true')
    args = ap.parse_args()

    spec = suite_mod.load(args.suite)
    if args.workloads != '*':
        spec = replace(spec, workloads=tuple(
            w for w in spec.workloads
            if fnmatch.fnmatch(w.name, args.workloads)))
    if args.configs:
        wanted = {c.strip() for c in args.configs.split(',')}
        spec = replace(spec, configs=tuple(
            c for c in spec.configs if c.label in wanted))
    if args.batches:
        spec = replace(spec, batches=tuple(
            int(b) for b in args.batches.split(',')))
    if not spec.workloads or not spec.configs:
        print('nothing left to run after filtering', file=sys.stderr)
        return 2

    devices = detect_devices(args.backend, args.arch)
    if spec.targets is not None:
        wanted = set(spec.targets)
        devices = [d for d in devices if d.spec in wanted]
    if not devices:
        print('no usable device; pass --arch to build without running',
              file=sys.stderr)
        return 2

    print(f'suite {spec.name}: {len(spec.workloads)} workloads, '
          f'{len(spec.configs)} configs, batches {spec.batches}')
    for device in devices:
        print(f'  - {device.spec.backend}-{device.spec.arch} ({device.name})')

    started = time.time()
    rows: List[Dict] = []
    manifest: List[Dict] = []
    for device in devices:
        for unit in suite_mod.expand(spec, [device.spec]):
            built = bench_build.build(unit, jobs=args.jobs)
            if built.error:
                print(f'{unit.label}: {built.error}', file=sys.stderr)
            for record in built.workloads:
                manifest.append({
                    'unit': unit.label, 'workload': record.name,
                    'symbol': record.symbol, 'origin': record.origin,
                    'lanes': record.lanes, 'static': record.static,
                    'error': record.error,
                    'flops_per_element': (record.cost.flops
                                          if record.cost else None),
                    'nonzero_flops_per_element': (record.cost.nonzero_flops
                                                  if record.cost else None),
                    'executed_flops_per_element': record.executed_flops,
                    'bytes_per_element': (record.cost.bytes
                                          if record.cost else None),
                })
            if built.exe is None or args.build_only:
                continue
            rows.extend(measure(built, device,
                                list(unit.workloads), args))

    if not args.build_only:
        time_rows(rows, args)
        report(rows, args.verbose)

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        blob = {
            'run': {
                'suite': spec.name,
                'tensorforge': interop.get_version(),
                'started': started,
                'elapsed_s': time.time() - started,
                'iters': args.iters, 'warmup': args.warmup,
                'repeat': args.repeat,
                'devices': [asdict(d) for d in devices],
            },
            'manifest': manifest,
            'results': rows,
        }
        (args.out / 'bench.json').write_text(json.dumps(blob, indent=2,
                                                        default=str))
        print(f'\nwritten: {args.out / "bench.json"}')

    return 0 if any(not r.get('error') for r in rows) or args.build_only else 1


if __name__ == '__main__':
    raise SystemExit(main())
