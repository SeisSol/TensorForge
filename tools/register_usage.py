# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Does the pressure model rank two lane configurations the way hipcc does?

The question a search over configurations has to answer before it is worth
building.  `pir.pressure(body, in_bytes=True, explicit_simd=False)` is fast and
needs no toolchain; the compiler's VGPR count is the truth and is far too slow
to sit inside a loop over configurations times sections.  So the model drives
and the compiler calibrates -- if the two agree well enough, which is what this
measures.

Two numbers come out, and they answer different things:

* **Rank agreement.** Of the cases where the two lane configurations differ,
  how often does the model prefer the one the compiler gives fewer VGPRs?
  This is the number that decides whether a search can use the model at all.
  A search only ever compares configurations; it never needs the absolute
  figure.
* **Correlation and scale.** How the model's bytes track the reported VGPRs.
  This is what a *threshold* would need -- "will this fit under 256" -- and it
  is the harder ask, since the model counts live value bytes while the
  compiler reports registers after allocation, coalescing and rematerialization.

Run it where ROCm is installed::

    python3 tools/register_usage.py --arch gfx90a
    python3 tools/register_usage.py --arch gfx942 --cases 'chain*' -v
    python3 tools/register_usage.py --arch gfx90a --json out.json

It compiles each case twice -- at the default lane ceiling and at the wave
width -- so a run costs two compilations per case.  `--jobs` parallelizes them.

Nothing here changes generated code.  A failed compilation is reported and
skipped, not raised: the point is a measurement over whatever compiles, and a
corpus that has some cases the toolchain refuses is still worth the answer for
the rest.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import os
import re
import statistics
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))

import tensorforge.backend.pir as pir  # noqa: E402
from tensorforge import toolchain  # noqa: E402
from tensorforge.common.context import Context  # noqa: E402
from tensorforge.generators import lanes  # noqa: E402
from tensorforge.generators.generator import Generator  # noqa: E402

@dataclass
class Measurement:
    case: str
    lanes: int
    model_bytes: int
    model_values: int
    register_slots: int
    vgprs: Optional[int] = None
    agprs: Optional[int] = None
    sgprs: Optional[int] = None
    scratch: Optional[int] = None
    spills: Optional[int] = None
    occupancy: Optional[int] = None
    error: str = ''


# -- generation ------------------------------------------------------------ #

def _peak_pressure_hook():
    """Capture the peak per-lane pressure of every body as it is emitted.

    Hooked rather than recomputed, because the body a section emits is not
    reachable afterwards -- `Generator` keeps the rendered text, not the IR.
    """
    peak: Dict[str, int] = {}
    original = pir.emit

    def emit(body, writer, context, *args, **kwargs):
        peak['bytes'] = max(peak.get('bytes', 0),
                            pir.pressure(body, in_bytes=True,
                                         explicit_simd=False))
        peak['values'] = max(peak.get('values', 0), pir.pressure(body))
        return original(body, writer, context, *args, **kwargs)

    return peak, emit, original


def _register_slots(src: str) -> int:
    """Declared register-array elements per lane, straight out of the source.

    An independent check on the model: it is the term the lane count halves,
    and if the model stops tracking it the two columns part company visibly.
    """
    return sum(int(n) for _, n in
               re.findall(r'\b(?:double|float|_Float16)\s+(r\d+)\[(\d+)\]',
                          src))


def load_cases(pattern: str) -> List:
    import fnmatch
    out = []
    for path in sorted((ROOT / 'tests' / 'cases').rglob('*.py')):
        if path.name.startswith('_'):
            continue
        spec = importlib.util.spec_from_file_location(f'ru_{path.stem}', path)
        mod = importlib.util.module_from_spec(spec)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                spec.loader.exec_module(mod)
        except Exception:
            continue
        if not hasattr(mod, 'descr_list') or not hasattr(mod, 'NAME'):
            continue
        if fnmatch.fnmatch(mod.NAME, pattern):
            out.append(mod)
    return out


def generate(mod, arch: str, ceiling: Optional[int],
             backend: Backend = None):
    """`(source, measurement)` for one case at one lane configuration."""
    peak, hook, original = _peak_pressure_hook()
    import tensorforge.backend.instructions.abstract_instruction as ai
    pir.emit, ai.pir.emit = hook, hook
    try:
        ctx = Context(arch=arch,
                      backend=(backend or BACKENDS['hip']).generator_backend,
                      fp_type=getattr(mod, 'DTYPE', None))
        config = lanes.deduce(mod.descr_list(), ctx, ceiling=ceiling)
        gen = Generator(mod.descr_list(), ctx, lanes=config)
        with contextlib.redirect_stdout(io.StringIO()):
            gen.generate()
        # the headers the generator asks for, on top of the backend's own: an
        # instruction can need one the backend does not list
        wanted = dict.fromkeys(ctx.target.headers()
                               + gen.get_helper_headers())
        src = ''.join(f'#include "{h}"\n' for h in wanted) + gen.get_kernel()
    except Exception as exc:
        return None, Measurement(mod.NAME, 0, 0, 0, 0,
                                 error=f'{type(exc).__name__}: {exc}')
    finally:
        pir.emit, ai.pir.emit = original, original

    return src, Measurement(case=mod.NAME,
                            lanes=config.num_threads,
                            model_bytes=peak.get('bytes', 0),
                            model_values=peak.get('values', 0),
                            register_slots=_register_slots(src))


# -- compilation ----------------------------------------------------------- #

@dataclass(frozen=True)
class Backend:
    """One toolchain: how to build an object for it, under which headers.

    Three of them, and they are not equally informative.  AMD reports every
    field per kernel, NVIDIA reports registers and spills, Intel reports only
    that a kernel spilled (`toolchain.report_fields`).  A caller gets what the
    toolchain gives and can tell absence from zero.
    """

    name: str
    #: What the generator emits for it ...
    generator_backend: str
    #: ... and which entry of `toolchain.COMPILERS` builds it: the SYCL
    #: backend generates SPMD SYCL and asks `icpx`, since IGC is what reports.
    compiler_backend: str
    headers: str

    @property
    def compiler(self) -> toolchain.Compiler:
        return toolchain.COMPILERS[self.compiler_backend]

    def parse(self, log: str) -> Dict[str, int]:
        return toolchain.report_fields(self.compiler_backend, log)

    def command(self, compiler: str, arch: str, src: Path, obj: Path,
                include: Path, extra: List[str]) -> List[str]:
        raise NotImplementedError

    def translation_unit(self, kernel: str) -> str:
        """The kernel under the real device headers, not the host shim.

        `tests/harness/syntax.py` puts a shim on top so a host `g++` accepts
        device code, which is right for a syntax check and wrong here: the
        numbers only mean anything if the device compiler sees what it will
        actually see.
        """
        return f'{self.headers}\n\n{kernel}\n'


@dataclass(frozen=True)
class HipBackend(Backend):
    def command(self, compiler, arch, src, obj, include, extra):
        c = self.compiler
        return [compiler, '-x', 'hip', *c.language_flags(),
                *c.target_flags(arch), '-O3', '-c', *c.report_flags(),
                '-I', str(include), *extra, str(src), '-o', str(obj)]


@dataclass(frozen=True)
class CudaBackend(Backend):
    def command(self, compiler, arch, src, obj, include, extra):
        c = self.compiler
        return [compiler, '-x', 'cu', *c.language_flags(),
                *c.target_flags(arch), '-O3', '-c', *c.report_flags(),
                '-I', str(include), *extra, str(src), '-o', str(obj)]


@dataclass(frozen=True)
class SyclBackend(Backend):
    def command(self, compiler, arch, src, obj, include, extra):
        # Ahead of time (`target_flags`), because a JIT build never reaches
        # IGC and so never says anything at all.
        c = self.compiler
        return [compiler, *c.language_flags(), *c.target_flags(arch), '-O3',
                '-c', '-I', str(include), *extra, str(src), '-o', str(obj)]


BACKENDS = {
    'hip': HipBackend(
        name='hip', generator_backend='hip', compiler_backend='hip',
        headers=('#include <hip/hip_runtime.h>\n'
                 '#include "tensorforge_device/hip.h"')),
    'cuda': CudaBackend(
        name='cuda', generator_backend='cuda', compiler_backend='cuda',
        headers='#include "tensorforge_device/cuda.h"'),
    'sycl': SyclBackend(
        name='sycl', generator_backend='acpp', compiler_backend='oneapi',
        headers=('#include <sycl/sycl.hpp>\n'
                 '#include "tensorforge_device/isycl.h"')),
}


def compile_and_read(kernel: str, arch: str, compiler: str, backend: Backend,
                     extra: List[str]) -> Dict[str, int]:
    """Compile for `arch` and return whatever the toolchain reported."""
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / f'k.{backend.name}.cpp'
        src.write_text(backend.translation_unit(kernel))
        cmd = backend.command(compiler, arch, src, Path(tmp) / 'k.o',
                              ROOT / 'src' / 'tensorforge' / 'include', extra)
        proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        head = (proc.stderr.strip().splitlines() or ['(no output)'])[:4]
        raise RuntimeError('\n'.join(head))

    fields = backend.parse(proc.stderr)
    if not fields and backend.name != 'sycl':
        # Not an error on SYCL: a clean build there says nothing, because
        # there is nothing per kernel to say -- silence means "did not spill".
        sample = '\n'.join(proc.stderr.strip().splitlines()[:6]) or '(silent)'
        raise RuntimeError(
            f'nothing was parsed from {compiler}. Either it is older than the '
            f'flag this asks for, or it prints a shape this does not read. '
            f'What it printed:\n{sample}')
    return fields


def measure(mod, arch: str, hipcc: Optional[str], extra: List[str],
            backend: Backend = None) -> List[Measurement]:
    out = []
    backend = backend or BACKENDS['hip']
    for ceiling in (lanes.DEFAULT_LANE_CEILING, None):
        src, m = generate(mod, arch, ceiling, backend)
        if src is not None and hipcc:
            try:
                f = compile_and_read(src, arch, hipcc, backend, extra)
                m.vgprs = f.get('vgprs')
                m.agprs = f.get('agprs')
                m.sgprs = f.get('sgprs')
                m.scratch = f.get('scratchsize')
                m.spills = ((f.get('vgprsspill') or 0)
                            + (f.get('sgprsspill') or 0)) or None
                m.occupancy = f.get('occupancy')
            except RuntimeError as exc:
                m.error = str(exc)
        out.append(m)
    return out


# -- reporting ------------------------------------------------------------- #

def report(rows: List[Measurement], verbose: bool,
           model_only: bool = False) -> int:
    by_case: Dict[str, List[Measurement]] = {}
    for r in rows:
        by_case.setdefault(r.case, []).append(r)

    pairs = [(a, b) for a, b in by_case.values()
             if a.lanes and b.lanes and a.lanes != b.lanes
             and a.vgprs and b.vgprs]

    if verbose or not pairs:
        print(f'{"case":32s} {"lanes":>7s} {"model B":>9s} {"slots":>7s} '
              f'{"VGPR":>6s} {"AGPR":>6s} {"scratch":>8s} {"occ":>4s}')
        for case in sorted(by_case):
            for r in by_case[case]:
                if r.error:
                    print(f'{case:32s} {r.lanes:7d}  {r.error.splitlines()[0][:60]}')
                    continue
                print(f'{case:32s} {r.lanes:7d} {r.model_bytes:9d} '
                      f'{r.register_slots:7d} {_n(r.vgprs):>6s} '
                      f'{_n(r.agprs):>6s} {_n(r.scratch):>8s} '
                      f'{_n(r.occupancy):>4s}')

    if not pairs:
        print('\nno case produced two comparable configurations with VGPR '
              'numbers; nothing to correlate')
        return 0 if model_only else 1

    agree = ties = 0
    for a, b in pairs:
        model_prefers = a if a.model_bytes <= b.model_bytes else b
        real_cost_a = (a.vgprs or 0) + (a.agprs or 0)
        real_cost_b = (b.vgprs or 0) + (b.agprs or 0)
        if real_cost_a == real_cost_b:
            ties += 1
            continue
        compiler_prefers = a if real_cost_a < real_cost_b else b
        agree += model_prefers.lanes == compiler_prefers.lanes

    decided = len(pairs) - ties
    print(f'\n{len(pairs)} cases with two comparable configurations, '
          f'{ties} of them tied')
    if decided:
        print(f'rank agreement model/compiler: {agree}/{decided} '
              f'({100.0 * agree / decided:.0f}%)')
        print('  This is the number that decides whether a search can use '
              'the model.')

    flat = [r for r in rows if r.vgprs and r.model_bytes]
    if len(flat) >= 3:
        xs = [r.model_bytes for r in flat]
        ys = [(r.vgprs or 0) + (r.agprs or 0) for r in flat]
        print(f'\nmodel bytes against VGPR+AGPR over {len(flat)} '
              f'measurements:')
        if len(set(ys)) == 1:
            # Said rather than printed as `nan`: a correlation against a
            # constant is undefined, and the reason is worth naming -- either
            # every kernel really does use the same registers, or the numbers
            # are not coming from where they are supposed to.
            print(f'  the compiler reports {ys[0]} everywhere; without '
                  f'spread there is nothing to correlate')
        else:
            print(f'  Pearson  r = {_pearson(xs, ys):+.3f}')
            print(f'  Spearman r = {_pearson(_ranks(xs), _ranks(ys)):+.3f}')
        ratio = [y / x for x, y in zip(xs, ys) if x]
        if ratio and len(set(ys)) > 1:
            print(f'  VGPRs per model byte: median '
                  f'{statistics.median(ratio):.4f}, '
                  f'range {min(ratio):.4f}..{max(ratio):.4f}')
            print('  A narrow range is what a threshold would need; a wide '
                  'one says only the rank carries.')

    spilled = [r for r in rows if r.spills]
    if spilled:
        print(f'\n{len(spilled)} configurations that spill:')
        for r in spilled[:10]:
            print(f'  {r.case} @ {r.lanes} lanes: {r.spills} '
                  f'({r.scratch} B scratch)')
    return 0


def _n(v) -> str:
    return '-' if v is None else str(v)


def _ranks(xs: List[float]) -> List[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    out = [0.0] * len(xs)
    for rank, i in enumerate(order):
        out[i] = float(rank)
    return out


def _pearson(xs: List[float], ys: List[float]) -> float:
    n = len(xs)
    if n < 2:
        return float('nan')
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy = sum((y - my) ** 2 for y in ys) ** 0.5
    return num / (dx * dy) if dx and dy else float('nan')


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--backend', default='hip', choices=sorted(BACKENDS),
                    help='which toolchain to ask. They do not answer equally: '
                         'hip reports every field per kernel, cuda reports '
                         'registers and spills, sycl reports only that a '
                         'kernel spilled -- IGC puts the register count in a '
                         'shader dump, not on the command line')
    ap.add_argument('--arch', default=None,
                    help='defaults to gfx90a / sm_80 / pvc for the backend')
    ap.add_argument('--compiler', '--hipcc', dest='compiler',
                    default=None,
                    help="defaults to the backend's environment variable "
                         '($TF_HIPCC, $TF_NVCC, $TF_ICPX), else the compiler '
                         'on PATH; omit compilation with --model-only')
    ap.add_argument('--cases', default='*')
    ap.add_argument('--jobs', type=int, default=os.cpu_count() or 1)
    ap.add_argument('--model-only', action='store_true',
                    help='generate and report the model figures without '
                         'compiling; useful to check the corpus first')
    ap.add_argument('--json', type=Path)
    ap.add_argument('-v', '--verbose', action='store_true')
    ap.add_argument('cflags', nargs='*',
                    help='extra flags passed through to hipcc')
    args = ap.parse_args()

    backend = BACKENDS[args.backend]
    args.arch = args.arch or {'hip': 'gfx90a', 'cuda': 'sm_80',
                              'sycl': 'pvc'}[args.backend]
    compiler = None if args.model_only else backend.compiler.find(
        args.compiler)
    if not args.model_only and not compiler:
        print(f'no {backend.compiler.binary} found; pass --compiler or set '
              f'${backend.compiler.env[0]}, or use --model-only',
              file=sys.stderr)
        return 2

    mods = load_cases(args.cases)
    if not mods:
        print(f'no case matches {args.cases!r}', file=sys.stderr)
        return 2
    print(f'{len(mods)} cases, backend={backend.name}, arch={args.arch}, '
          f'{"model only" if not compiler else compiler}')

    rows: List[Measurement] = []
    # Generation mutates module-level hooks, so it runs serially; only the
    # compilations, which are the slow part, are spread out.
    plans = [(mod, generate(mod, args.arch, c, backend))
             for mod in mods
             for c in (lanes.DEFAULT_LANE_CEILING, None)]

    def finish(item):
        mod, (src, m) = item
        if src is not None and compiler:
            try:
                f = compile_and_read(src, args.arch, compiler, backend,
                                     args.cflags)
                m.vgprs, m.agprs = f.get('vgprs'), f.get('agprs')
                m.sgprs, m.scratch = f.get('sgprs'), f.get('scratchsize')
                m.spills = ((f.get('vgprsspill') or 0)
                            + (f.get('sgprsspill') or 0)) or None
                m.occupancy = f.get('occupancy')
            except RuntimeError as exc:
                m.error = str(exc)
        return m

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        rows = list(pool.map(finish, plans))

    if args.json:
        args.json.write_text(json.dumps([asdict(r) for r in rows], indent=2))
        print(f'written: {args.json}')
    return report(rows, args.verbose, model_only=not compiler)


if __name__ == '__main__':
    raise SystemExit(main())
