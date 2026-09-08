# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Where the roof is, measured, and where the kernels sit under it.

A roofline needs three things. Two of them exist: `analysis/cost.py` gives the
arithmetic intensity and `run.py` gives the achieved rate. The third -- the
machine's ceilings -- is not anywhere. `hw_descr_db.yml` carries what the code
generator may emit and deliberately not what the hardware sustains, which is
the right split: a clock rate is not a code-generation constraint, and putting
one there would make the database wrong every time a machine is
power-capped.

So the ceilings are measured, here, by two microbenchmarks built with the same
recipe as everything else in this directory.

## Why measured and not looked up

A datasheet peak is a number at a boost clock the machine may never hold. It is
also a number for the instruction mix the vendor chose: an FMA peak has nothing
to say about a kernel that cannot issue FMAs back to back. Dividing a real
kernel by a marketing figure produces a percentage that is wrong by an unknown
amount in a known direction, and the usual response -- "we are at 12% of peak"
-- is then a statement about the divisor.

Measuring also makes the roofline available on every stack. AMD's own roofline
is `rocprof-compute --roof-only` and needs MI200 or newer; Intel's is Advisor's.
Neither is wrapped here, and both are better than this when they are available:
they know their own machine. What this gives is a roof on the machine in front
of you, from the toolchain you already needed, without a profiler and without a
licence.

## The two ceilings

`fma` is a dependent chain per accumulator with enough accumulators to cover
the pipeline, no memory traffic in the loop. That is the compute roof for the
type it was built for, and it is a roof for *this* instruction mix -- a machine
with matrix units will sit far above it on a kernel that uses them, which the
report says rather than hides.

`triad` is `c = a + s*b` over a buffer far larger than the last level of cache,
counted at three arrays' worth of bytes the way STREAM counts it. On hardware
that allocates on write the true traffic is four; the convention is stated
rather than corrected, because a corrected number is no longer the one everyone
else quotes.

## Reading the result

A point sitting on the memory slope is bound by traffic and the useful lever is
reuse; on the flat it is bound by arithmetic. For the batched small-operator
kernels this code generates, the interesting failure is neither: a point far
*below* both roofs is bound by occupancy, launch overhead or latency, and the
report says so instead of leaving a reader to infer a roof that was never the
constraint.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for extra in (str(HERE), str(ROOT / 'src'), str(ROOT / 'tests')):
    if extra not in sys.path:
        sys.path.insert(0, extra)

import build as bench_build                                     # noqa: E402
import run as bench_run                                         # noqa: E402
from tensorforge.common.basic_types import Datatype             # noqa: E402

#: Independent accumulators in the FMA chain.  Enough to cover the issue
#: latency of an FMA on every current stack; more costs registers and stops
#: buying throughput, fewer leaves the pipeline stalling on its own result and
#: measures the latency instead of the rate.
ACCUMULATORS = 8

_CTYPE = {Datatype.F32: 'float', Datatype.F64: 'double'}


# ----------------------------------------------------------------------
# The microbenchmark
# ----------------------------------------------------------------------

_CUDAHIP = r'''
%(include)s
#include <chrono>
#include <cstdio>
#include <cstdlib>

using T = %(ctype)s;

__global__ void tfc_fma(T* out, unsigned long long iters) {
  T a[%(acc)d];
  for (int k = 0; k < %(acc)d; ++k) a[k] = (T)(threadIdx.x + k) * (T)1e-3;
  const T b = (T)1.0000001, c = (T)0.9999999;
  for (unsigned long long i = 0; i < iters; ++i) {
#pragma unroll
    for (int k = 0; k < %(acc)d; ++k) a[k] = a[k] * b + c;
  }
  T s = (T)0;
  for (int k = 0; k < %(acc)d; ++k) s += a[k];
  /* Unconditional, so nothing above may be eliminated; one store per thread
     against `iters * ACCUMULATORS` fused multiply-adds is not measurable. */
  out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}

__global__ void tfc_triad(const T* __restrict__ a, const T* __restrict__ b,
                          T* __restrict__ c, size_t n, T s) {
  size_t stride = (size_t)gridDim.x * blockDim.x;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += stride)
    c[i] = a[i] + s * b[i];
}

#define CHK(x) do { auto e_ = (x); if (e_ != %(prefix)sSuccess) { \
    std::fprintf(stderr, "%%s\n", %(prefix)sGetErrorString(e_)); \
    std::exit(2); } } while (0)

int main(int argc, char** argv) {
  const int device = (argc > 1) ? std::atoi(argv[1]) : 0;
  CHK(%(prefix)sSetDevice(device));
  %(propty)s prop;
  CHK(%(prefix)sGetDeviceProperties(&prop, device));

  const int block = 256;
  const int grid = prop.multiProcessorCount * 32;
  const unsigned long long iters = 4096;

  T* out = nullptr;
  CHK(%(prefix)sMalloc(&out, (size_t)grid * block * sizeof(T)));

  double fma_best = 0.0;
  for (int rep = 0; rep < 5; ++rep) {
    CHK(%(prefix)sDeviceSynchronize());
    auto t0 = std::chrono::steady_clock::now();
    tfc_fma<<<grid, block>>>(out, iters);
    CHK(%(prefix)sDeviceSynchronize());
    auto t1 = std::chrono::steady_clock::now();
    double s = std::chrono::duration<double>(t1 - t0).count();
    /* Two flops per fused multiply-add. */
    double flops = 2.0 * %(acc)d * (double)iters * (double)grid * block;
    if (rep > 0) fma_best = fma_best > flops / s ? fma_best : flops / s;
  }

  /* Far past any last-level cache on current parts. */
  const size_t n = (size_t)1 << 26;
  T *a = nullptr, *b = nullptr, *c = nullptr;
  CHK(%(prefix)sMalloc(&a, n * sizeof(T)));
  CHK(%(prefix)sMalloc(&b, n * sizeof(T)));
  CHK(%(prefix)sMalloc(&c, n * sizeof(T)));
  CHK(%(prefix)sMemset(a, 1, n * sizeof(T)));
  CHK(%(prefix)sMemset(b, 1, n * sizeof(T)));

  double bw_best = 0.0;
  for (int rep = 0; rep < 5; ++rep) {
    CHK(%(prefix)sDeviceSynchronize());
    auto t0 = std::chrono::steady_clock::now();
    tfc_triad<<<grid, block>>>(a, b, c, n, (T)1.5);
    CHK(%(prefix)sDeviceSynchronize());
    auto t1 = std::chrono::steady_clock::now();
    double s = std::chrono::duration<double>(t1 - t0).count();
    double bytes = 3.0 * (double)n * sizeof(T);
    if (rep > 0) bw_best = bw_best > bytes / s ? bw_best : bytes / s;
  }

  std::printf("{\"flops_per_s\": %%.6e, \"bytes_per_s\": %%.6e, "
              "\"device\": \"%%s\", \"cus\": %%d}\n",
              fma_best, bw_best, prop.name, prop.multiProcessorCount);
  return 0;
}
'''

_SYCL = r'''
#include <sycl/sycl.hpp>
#include <chrono>
#include <cstdio>
#include <cstdlib>

using T = %(ctype)s;

int main(int argc, char** argv) {
  sycl::queue q{sycl::default_selector_v, sycl::property::queue::in_order()};
  const size_t block = 256;
  const size_t cus =
      q.get_device().get_info<sycl::info::device::max_compute_units>();
  const size_t grid = cus * 32;
  const unsigned long long iters = 4096;

  T* out = sycl::malloc_device<T>(grid * block, q);

  double fma_best = 0.0;
  for (int rep = 0; rep < 5; ++rep) {
    q.wait();
    auto t0 = std::chrono::steady_clock::now();
    q.parallel_for(sycl::nd_range<1>{grid * block, block},
                   [=](sycl::nd_item<1> it) {
      T a[%(acc)d];
      size_t lid = it.get_local_id(0);
      for (int k = 0; k < %(acc)d; ++k) a[k] = (T)(lid + k) * (T)1e-3;
      const T b = (T)1.0000001, c = (T)0.9999999;
      for (unsigned long long i = 0; i < iters; ++i)
#pragma unroll
        for (int k = 0; k < %(acc)d; ++k) a[k] = a[k] * b + c;
      T s = (T)0;
      for (int k = 0; k < %(acc)d; ++k) s += a[k];
      out[it.get_global_id(0)] = s;
    });
    q.wait();
    auto t1 = std::chrono::steady_clock::now();
    double s = std::chrono::duration<double>(t1 - t0).count();
    double flops = 2.0 * %(acc)d * (double)iters * (double)(grid * block);
    if (rep > 0) fma_best = fma_best > flops / s ? fma_best : flops / s;
  }

  const size_t n = (size_t)1 << 26;
  T* a = sycl::malloc_device<T>(n, q);
  T* b = sycl::malloc_device<T>(n, q);
  T* c = sycl::malloc_device<T>(n, q);
  q.memset(a, 1, n * sizeof(T)).wait();
  q.memset(b, 1, n * sizeof(T)).wait();

  double bw_best = 0.0;
  for (int rep = 0; rep < 5; ++rep) {
    q.wait();
    auto t0 = std::chrono::steady_clock::now();
    q.parallel_for(sycl::range<1>{n}, [=](sycl::id<1> i) {
      c[i] = a[i] + (T)1.5 * b[i];
    });
    q.wait();
    auto t1 = std::chrono::steady_clock::now();
    double s = std::chrono::duration<double>(t1 - t0).count();
    double bytes = 3.0 * (double)n * sizeof(T);
    if (rep > 0) bw_best = bw_best > bytes / s ? bw_best : bytes / s;
  }

  std::printf("{\"flops_per_s\": %%.6e, \"bytes_per_s\": %%.6e, "
              "\"device\": \"%%s\", \"cus\": %%zu}\n",
              fma_best, bw_best,
              q.get_device().get_info<sycl::info::device::name>().c_str(), cus);
  return 0;
}
'''


def emit_ceiling(backend: str, datatype: Datatype) -> str:
    """The microbenchmark source for one backend and one element type.

    The first repetition of each loop is discarded rather than averaged in: it
    carries the module load and, on a JIT stack, the compile. The rest are
    reduced with a maximum and not a mean -- a ceiling is the best the machine
    did, and a mean over a run that was descheduled once reports the
    scheduler.
    """
    ctype = _CTYPE.get(datatype)
    if ctype is None:
        raise NotImplementedError(
            f'no ceiling microbenchmark for {datatype.name}; the roof for a '
            f'type the machine emulates is not a hardware property')
    if backend in ('oneapi', 'acpp', 'esimd'):
        return _SYCL % {'ctype': ctype, 'acc': ACCUMULATORS}
    return _CUDAHIP % {
        'ctype': ctype, 'acc': ACCUMULATORS,
        'prefix': 'cuda' if backend == 'cuda' else 'hip',
        # CUDA spells the properties struct `cudaDeviceProp` and HIP spells it
        # `hipDeviceProp_t`; the prefix substitution does not reach that.
        'propty': ('cudaDeviceProp' if backend == 'cuda'
                   else 'hipDeviceProp_t'),
        'include': ('#include <cuda_runtime.h>' if backend == 'cuda'
                    else '#include <hip/hip_runtime.h>'),
    }


@dataclass(frozen=True)
class Ceiling:
    """What the machine did, not what it is sold as."""
    backend: str
    arch: str
    datatype: str
    device: str
    compute_units: int
    flops_per_s: float
    bytes_per_s: float

    @property
    def ridge(self) -> float:
        """Flops per byte where the two roofs meet."""
        return self.flops_per_s / self.bytes_per_s if self.bytes_per_s else 0.0

    def bound(self, intensity: float) -> float:
        """The roof at one arithmetic intensity."""
        return min(self.flops_per_s, self.bytes_per_s * intensity)


def measure_ceiling(device: bench_run.Device, datatype: Datatype,
                    cache: Path = bench_build.CACHE,
                    timeout: float = 600.0) -> Tuple[Optional[Ceiling], str]:
    """Build and run the microbenchmark, or say why not."""
    backend = device.spec.backend
    compiler = bench_build.COMPILERS.get(backend)
    if compiler is None:
        return None, f'no compiler recipe for {backend!r}'
    cc = bench_build.compiler_binary(compiler)
    if cc is None:
        return None, f'{compiler.default} not found; set ${compiler.env}'

    try:
        source = emit_ceiling(backend, datatype)
    except NotImplementedError as exc:
        return None, str(exc)

    out = cache / 'ceiling' / f'{backend}-{device.spec.arch}-{datatype.name}'
    out.mkdir(parents=True, exist_ok=True)
    src = out / f'ceiling{compiler.source_suffix()}'
    src.write_text(source)
    exe = out / 'ceiling'

    if not exe.exists():
        cmd = [cc, *compiler.link_flags(device.spec.arch), str(src),
               '-o', str(exe)]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            (out / 'build.log').write_text(
                ' '.join(cmd) + '\n\n' + proc.stdout + '\n' + proc.stderr)
            return None, f'ceiling did not build; see {out / "build.log"}'

    env = os.environ.copy()
    if device.vendor == 'nvidia':
        env['CUDA_VISIBLE_DEVICES'] = str(device.index)
    elif device.vendor == 'amd':
        env['HIP_VISIBLE_DEVICES'] = str(device.index)
    try:
        proc = subprocess.run([str(exe), '0'], capture_output=True, text=True,
                              timeout=timeout, check=False, env=env)
    except subprocess.TimeoutExpired:
        return None, f'ceiling run timed out after {timeout}s'
    if proc.returncode != 0:
        return None, f'ceiling run exited {proc.returncode}: ' \
                     f'{proc.stderr.strip()[:120]}'
    for line in proc.stdout.splitlines():
        if line.strip().startswith('{'):
            blob = json.loads(line)
            return Ceiling(
                backend=backend, arch=device.spec.arch,
                datatype=datatype.name, device=blob['device'],
                compute_units=int(blob['cus']),
                flops_per_s=float(blob['flops_per_s']),
                bytes_per_s=float(blob['bytes_per_s'])), ''
    return None, 'the ceiling benchmark printed no measurement'


# ----------------------------------------------------------------------
# Placing the kernels
# ----------------------------------------------------------------------

@dataclass
class Point:
    workload: str
    config: str
    batch: int
    intensity: float
    achieved: float          # flops per second
    roof: float
    #: `memory`, `compute`, or `neither` -- see the module docstring.
    regime: str
    fraction: float


#: How close to a roof counts as reaching it.  A kernel at 80% of the bound its
#: intensity puts it under is doing what the machine allows; below half of it,
#: the binding constraint is something the roofline does not draw.
AT_ROOF = 0.8
FAR_BELOW = 0.5


def place(rows: List[Dict], ceiling: Ceiling) -> List[Point]:
    points: List[Point] = []
    for row in rows:
        if row.get('error') or not row.get('intensity'):
            continue
        nanos = row.get('event_ns') or row.get('wall_ns')
        if not nanos or not row.get('flops'):
            continue
        achieved = row['flops'] / (nanos * 1e-9)
        roof = ceiling.bound(row['intensity'])
        fraction = achieved / roof if roof else 0.0
        if fraction < FAR_BELOW:
            regime = 'neither'
        elif row['intensity'] < ceiling.ridge:
            regime = 'memory'
        else:
            regime = 'compute'
        points.append(Point(
            workload=row['workload'], config=row.get('config', ''),
            batch=row.get('batch', 0), intensity=row['intensity'],
            achieved=achieved, roof=roof, regime=regime, fraction=fraction))
    return points


def svg(points: List[Point], ceiling: Ceiling) -> str:
    """A standalone log-log plot, written by hand.

    No plotting dependency: this directory's whole point is to run on a login
    node of a machine somebody else administers, and `pip install matplotlib`
    is not always a thing that happens there. The output opens in a browser and
    is a text file a diff can read.
    """
    W, H, PAD = 720, 440, 60
    xs = [p.intensity for p in points] or [1.0]
    ys = [p.achieved for p in points] or [1.0]
    x0 = min(min(xs) / 4, ceiling.ridge / 8)
    x1 = max(max(xs) * 4, ceiling.ridge * 8)
    y0 = min(min(ys) / 8, ceiling.flops_per_s / 1000)
    y1 = ceiling.flops_per_s * 2

    def px(x):
        return PAD + (math.log10(max(x, x0)) - math.log10(x0)) / \
            (math.log10(x1) - math.log10(x0)) * (W - 2 * PAD)

    def py(y):
        return H - PAD - (math.log10(max(y, y0)) - math.log10(y0)) / \
            (math.log10(y1) - math.log10(y0)) * (H - 2 * PAD)

    roof = ' '.join(
        f'{px(x):.1f},{py(ceiling.bound(x)):.1f}'
        for x in (x0, ceiling.ridge, x1))
    colours = {'memory': '#2f6fb2', 'compute': '#b2452f', 'neither': '#888888'}
    dots = '\n'.join(
        f'<circle cx="{px(p.intensity):.1f}" cy="{py(p.achieved):.1f}" r="4" '
        f'fill="{colours[p.regime]}" opacity="0.75">'
        f'<title>{p.workload} / {p.config} @ {p.batch}: '
        f'{p.achieved / 1e9:.1f} GFLOP/s, {p.fraction * 100:.0f}% of roof'
        f'</title></circle>'
        for p in points)

    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W}" \
height="{H}" font-family="sans-serif" font-size="11">
<rect width="{W}" height="{H}" fill="white"/>
<polyline points="{roof}" fill="none" stroke="#222" stroke-width="2"/>
<line x1="{px(ceiling.ridge):.1f}" y1="{PAD}" x2="{px(ceiling.ridge):.1f}" \
y2="{H - PAD}" stroke="#bbb" stroke-dasharray="4 3"/>
{dots}
<line x1="{PAD}" y1="{H - PAD}" x2="{W - PAD}" y2="{H - PAD}" stroke="#222"/>
<line x1="{PAD}" y1="{PAD}" x2="{PAD}" y2="{H - PAD}" stroke="#222"/>
<text x="{W / 2}" y="{H - 18}" text-anchor="middle">arithmetic intensity \
(flop/byte, compulsory)</text>
<text x="16" y="{H / 2}" text-anchor="middle" \
transform="rotate(-90 16 {H / 2})">achieved GFLOP/s</text>
<text x="{PAD + 6}" y="{PAD - 12}">{ceiling.device} — \
{ceiling.flops_per_s / 1e12:.2f} TFLOP/s, \
{ceiling.bytes_per_s / 1e9:.0f} GB/s, ridge \
{ceiling.ridge:.1f} flop/byte ({ceiling.datatype}, measured)</text>
</svg>
'''


def report(points: List[Point], ceiling: Ceiling) -> None:
    print(f'{ceiling.device}: {ceiling.flops_per_s / 1e12:.2f} TFLOP/s '
          f'({ceiling.datatype} FMA), {ceiling.bytes_per_s / 1e9:.0f} GB/s '
          f'(triad), ridge at {ceiling.ridge:.1f} flop/byte')
    print()
    head = (f'{"workload":30s} {"cfg":10s} {"batch":>8s} {"AI":>7s} '
            f'{"GFLOP/s":>9s} {"roof":>9s} {"% roof":>7s}  regime')
    print(head)
    print('-' * len(head))
    for p in sorted(points, key=lambda p: -p.fraction):
        print(f'{p.workload[:30]:30s} {p.config[:10]:10s} {p.batch:8d} '
              f'{p.intensity:7.2f} {p.achieved / 1e9:9.1f} '
              f'{p.roof / 1e9:9.1f} {p.fraction * 100:6.1f}%  {p.regime}')

    stranded = [p for p in points if p.regime == 'neither']
    if stranded:
        print(f'\n{len(stranded)} of {len(points)} points sit below half of '
              f'the roof their intensity allows. Neither line is the '
              f'constraint there: look at occupancy, at the launch overhead '
              f'in the wall column, or at whether the batch saturated the '
              f'grid at all.')


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('bench', type=Path, nargs='?',
                    help='bench.json from tools/bench/run.py --out')
    ap.add_argument('--backend', default=None)
    ap.add_argument('--arch', default=None)
    # By name: `Datatype.__str__` is the C spelling, so passing the members
    # as choices would have argparse advertise `float` and reject `F32`.
    ap.add_argument('--datatype', default='F32',
                    choices=sorted(d.name for d in _CTYPE))
    ap.add_argument('--ceiling-only', action='store_true',
                    help='measure the roof and print it, and place nothing')
    ap.add_argument('--peak-flops', type=float, default=None,
                    help='use this instead of measuring, in flop/s')
    ap.add_argument('--peak-bytes', type=float, default=None,
                    help='use this instead of measuring, in byte/s')
    ap.add_argument('--out', type=Path, default=None)
    args = ap.parse_args()

    datatype = Datatype[args.datatype]
    devices = bench_run.detect_devices(args.backend, args.arch)
    if not devices:
        print('no device detected', file=sys.stderr)
        return 2
    device = devices[0]

    if args.peak_flops and args.peak_bytes:
        # Supplied rather than measured: a vendor roofline, or a figure from
        # rocprof-compute. Recorded as supplied so a later reader can tell it
        # from a measurement.
        ceiling = Ceiling(device.spec.backend, device.spec.arch,
                          datatype.name, f'{device.name} (supplied)', 0,
                          args.peak_flops, args.peak_bytes)
    else:
        ceiling, why = measure_ceiling(device, datatype)
        if ceiling is None:
            print(f'no ceiling: {why}', file=sys.stderr)
            print('pass --peak-flops and --peak-bytes to supply one, e.g. '
                  'from rocprof-compute --roof-only or Intel Advisor',
                  file=sys.stderr)
            return 2

    if args.ceiling_only or args.bench is None:
        print(json.dumps(asdict(ceiling), indent=2))
        print(f'ridge: {ceiling.ridge:.2f} flop/byte')
        return 0

    blob = json.loads(args.bench.read_text())
    rows = [r for r in blob.get('results', [])
            if r.get('datatype') == datatype.name]
    if not rows:
        print(f'no {datatype.name} results in {args.bench}', file=sys.stderr)
        return 2

    points = place(rows, ceiling)
    report(points, ceiling)

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / 'roofline.svg').write_text(svg(points, ceiling))
        (args.out / 'roofline.json').write_text(json.dumps({
            'ceiling': asdict(ceiling),
            'points': [asdict(p) for p in points],
        }, indent=2))
        print(f'\nwritten: {args.out / "roofline.svg"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
