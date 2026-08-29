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
  compiler reports registers after allocation, coalescing and rematerialisation.

Run it where ROCm is installed::

    python3 tools/register_usage.py --arch gfx90a
    python3 tools/register_usage.py --arch gfx942 --cases 'chain*' -v
    python3 tools/register_usage.py --arch gfx90a --json out.json

It compiles each case twice -- at the default lane ceiling and at the wave
width -- so a run costs two compilations per case.  `--jobs` parallelises them.

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
import shutil
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
from tensorforge.common.context import Context  # noqa: E402
from tensorforge.generators import lanes  # noqa: E402
from tensorforge.generators.generator import Generator  # noqa: E402

#: What `-Rpass-analysis=kernel-resource-usage` prints, one remark per field.
#: Parsed rather than read off an object file because it needs no extra tool
#: and survives the object being discarded.
#:
#: Deliberately not a list of the field names.  Half of them carry their unit
#: in brackets before the colon -- `ScratchSize [bytes/lane]`, `Occupancy
#: [waves/SIMD]`, `LDS Size [bytes/block]` -- and the spill fields are `SGPRs
#: Spill` and `VGPRs Spill`, not one count.  Matching a name list against that
#: silently returns nothing for the fields that matter most, on a tool whose
#: whole job is to report numbers, so this takes whatever `name: integer` the
#: remarks contain and normalises the name afterwards.  Non-numeric remarks
#: (`Function Name`, `Dynamic Stack: False`) simply do not match.
_REMARK = re.compile(
    r'remark:\s*(?P<field>[A-Za-z][A-Za-z ]*?)\s*'
    r'(?:\[[^\]]*\])?\s*:\s*(?P<value>-?[0-9]+)\b')


def parse_remarks(stderr: str) -> Dict[str, int]:
    """Every `name: integer` remark, keyed by a squashed lower-case name.

    The maximum per field, because a translation unit may hold more than one
    kernel and the budget is per kernel: the largest is the only reading that
    cannot understate.
    """
    fields: Dict[str, int] = {}
    for m in _REMARK.finditer(stderr):
        key = m.group('field').strip().lower().replace(' ', '')
        fields[key] = max(fields.get(key, 0), int(m.group('value')))
    return fields


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


def generate(mod, arch: str, ceiling: Optional[int]):
    """`(source, measurement)` for one case at one lane configuration."""
    peak, hook, original = _peak_pressure_hook()
    import tensorforge.backend.instructions.abstract_instruction as ai
    pir.emit, ai.pir.emit = hook, hook
    try:
        ctx = Context(arch=arch, backend='hip',
                      fp_type=getattr(mod, 'DTYPE', None))
        config = lanes.deduce(mod.descr_list(), ctx, ceiling=ceiling)
        gen = Generator(mod.descr_list(), ctx, lanes=config)
        with contextlib.redirect_stdout(io.StringIO()):
            gen.generate()
        src = gen.get_kernel()
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

def translation_unit(kernel: str) -> str:
    """The kernel under the real device header, not the host shim.

    `tests/harness/syntax.py` puts a shim on top so that a host `g++` accepts
    device code, which is right for a syntax check and wrong here: the numbers
    only mean anything if the device compiler sees what it will actually see.
    """
    return ('#include <hip/hip_runtime.h>\n'
            '#include "tensorforge_device/hip.h"\n\n'
            f'{kernel}\n')


def compile_and_read(kernel: str, arch: str, hipcc: str,
                     extra: List[str]) -> Dict[str, int]:
    """Compile for `arch` and return whatever resource remarks came back."""
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / 'k.hip.cpp'
        src.write_text(translation_unit(kernel))
        cmd = [hipcc, '-x', 'hip', f'--offload-arch={arch}', '-O3', '-c',
               '-Rpass-analysis=kernel-resource-usage',
               '-I', str(ROOT / 'src' / 'tensorforge' / 'include'),
               *extra, str(src), '-o', str(Path(tmp) / 'k.o')]
        proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        head = (proc.stderr.strip().splitlines() or ['(no output)'])[:4]
        raise RuntimeError('\n'.join(head))

    fields = parse_remarks(proc.stderr)
    if not fields:
        sample = '\n'.join(proc.stderr.strip().splitlines()[:6]) or '(silent)'
        raise RuntimeError(
            'no kernel-resource-usage remarks were parsed. Either this hipcc '
            'predates -Rpass-analysis=kernel-resource-usage, or it prints a '
            f'shape this does not read. What it printed:\n{sample}')
    return fields


def measure(mod, arch: str, hipcc: Optional[str], extra: List[str]
            ) -> List[Measurement]:
    out = []
    for ceiling in (lanes.DEFAULT_LANE_CEILING, None):
        src, m = generate(mod, arch, ceiling)
        if src is not None and hipcc:
            try:
                f = compile_and_read(src, arch, hipcc, extra)
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
    print(f'\n{len(pairs)} Fälle mit zwei vergleichbaren Konfigurationen, '
          f'{ties} davon gleichteuer')
    if decided:
        print(f'Rangübereinstimmung Modell/Compiler: {agree}/{decided} '
              f'({100.0 * agree / decided:.0f}%)')
        print('  Das ist die Zahl, an der hängt, ob eine Suche das Modell '
              'benutzen kann.')

    flat = [r for r in rows if r.vgprs and r.model_bytes]
    if len(flat) >= 3:
        xs = [r.model_bytes for r in flat]
        ys = [(r.vgprs or 0) + (r.agprs or 0) for r in flat]
        print(f'\nModellbytes gegen VGPR+AGPR über {len(flat)} Messungen:')
        if len(set(ys)) == 1:
            # Said rather than printed as `nan`: a correlation against a
            # constant is undefined, and the reason is worth naming -- either
            # every kernel really does use the same registers, or the numbers
            # are not coming from where they are supposed to.
            print(f'  der Compiler meldet überall {ys[0]}; ohne Streuung gibt '
                  f'es nichts zu korrelieren')
        else:
            print(f'  Pearson  r = {_pearson(xs, ys):+.3f}')
            print(f'  Spearman r = {_pearson(_ranks(xs), _ranks(ys)):+.3f}')
        ratio = [y / x for x, y in zip(xs, ys) if x]
        if ratio and len(set(ys)) > 1:
            print(f'  VGPR je Modellbyte: Median {statistics.median(ratio):.4f}, '
                  f'Spanne {min(ratio):.4f}..{max(ratio):.4f}')
            print('  Eine enge Spanne ist, was ein Schwellwert bräuchte; '
                  'eine weite sagt, dass nur der Rang trägt.')

    spilled = [r for r in rows if r.spills]
    if spilled:
        print(f'\n{len(spilled)} Konfigurationen mit Spills:')
        for r in spilled[:10]:
            print(f'  {r.case} @ {r.lanes} Lanes: {r.spills} '
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
    ap.add_argument('--arch', default='gfx90a')
    ap.add_argument('--hipcc', default=os.environ.get('TF_HIPCC'),
                    help='defaults to $TF_HIPCC, else whichever hipcc is on '
                         'PATH; omit compilation entirely with --model-only')
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

    hipcc = None if args.model_only else (args.hipcc or shutil.which('hipcc'))
    if not args.model_only and not hipcc:
        print('no hipcc found; pass --hipcc or set TF_HIPCC, or use '
              '--model-only', file=sys.stderr)
        return 2

    mods = load_cases(args.cases)
    if not mods:
        print(f'no case matches {args.cases!r}', file=sys.stderr)
        return 2
    print(f'{len(mods)} Fälle, arch={args.arch}, '
          f'{"nur Modell" if not hipcc else hipcc}')

    rows: List[Measurement] = []
    # Generation mutates module-level hooks, so it runs serially; only the
    # compilations, which are the slow part, are spread out.
    plans = [(mod, generate(mod, args.arch, c))
             for mod in mods
             for c in (lanes.DEFAULT_LANE_CEILING, None)]

    def finish(item):
        mod, (src, m) = item
        if src is not None and hipcc:
            try:
                f = compile_and_read(src, args.arch, hipcc, args.cflags)
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
        print(f'geschrieben: {args.json}')
    return report(rows, args.verbose, model_only=not hipcc)


if __name__ == '__main__':
    raise SystemExit(main())
