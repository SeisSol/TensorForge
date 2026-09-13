# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Fit the instruction-cache estimate against what the compilers emit.

`analysis.icache` weighs the emitter's units (`Context.record_code`) against a
target's instruction cache, through `INSTRUCTIONS_PER_UNIT` -- machine
instructions per unit -- and the target's `instruction_bytes`.  This fits the
first against the case set: each case is generated, its device code compiled
on its own (nvcc to a cubin and `cuobjdump -sass`; hipcc to device assembly),
and the instructions of the kernel counted.  The fit is through the origin,
and the spread is printed beside it: a factor that is right on the totals and
off by half on a mid-sized kernel is fine for a cliff and for nothing finer.

  python tools/calibrate_icache.py --arch sm_120 [--arch gfx942 ...] [--cases GLOB]
"""

import argparse
import contextlib
import importlib.util
import io
import re
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

from tensorforge.common.context import Context
from tensorforge.generators.generator import Generator

ROOT = Path(__file__).resolve().parents[1]
INCLUDE = ROOT / 'src' / 'tensorforge' / 'include'


def generate(path: Path, arch: str):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(module)
    if not hasattr(module, 'descr_list'):
        return None
    backend = 'cuda' if arch.startswith('sm_') else 'hip'
    context = Context(arch=arch, backend=backend, fp_type=module.DTYPE)
    generator = Generator(module.descr_list(), context)
    with contextlib.redirect_stdout(io.StringIO()):
        generator.generate()
    headers = context.get_vm().get_headers() + generator.get_helper_headers()
    def include(header):
        if header.startswith('#'):
            return header
        if header.startswith(('<', '"')):
            return f'#include {header}'
        return f'#include <{header}>'
    includes = '\n'.join(include(h) for h in dict.fromkeys(headers))
    return generator, f'{includes}\n{generator.get_kernel()}\n'


def sass_instructions(source: str, arch: str, tmp: Path) -> int:
    cu, cubin = tmp / 'k.cu', tmp / 'k.cubin'
    cu.write_text(source)
    subprocess.run(['nvcc', '-O3', '-std=c++17', '--expt-relaxed-constexpr',
                    f'-arch={arch}', '-cubin', '-I', str(INCLUDE),
                    '-o', str(cubin), str(cu)], check=True, capture_output=True)
    sass = subprocess.run(['cuobjdump', '-sass', str(cubin)], check=True,
                          capture_output=True, text=True).stdout
    # an offset past 0xffff has five digits: `{4}` stopped counting at 4096
    return len(re.findall(r'^\s+/\*[0-9a-f]{4,}\*/\s+\S', sass, re.M))


def isa_instructions(source: str, arch: str, tmp: Path) -> int:
    hip, asm = tmp / 'k.hip', tmp / 'k.s'
    hip.write_text(source)
    subprocess.run(['hipcc', '-O3', f'--offload-arch={arch}', '-x', 'hip',
                    '--cuda-device-only', '-S', '-I', str(INCLUDE),
                    '-o', str(asm), str(hip)], check=True, capture_output=True)
    count, inside = 0, False
    for line in asm.read_text().splitlines():
        if re.match(r'^_Z\w*kernel_\w*:', line):
            inside = True
        elif inside and line.startswith('.Lfunc_end'):
            inside = False
        elif inside and re.match(r'^\s+[a-z][a-z0-9_]*(\s|$)', line):
            count += 1
    return count


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--arch', action='append', required=True)
    parser.add_argument('--cases', default='**/*.py',
                        help='glob under tests/cases (default: all)')
    args = parser.parse_args(argv)
    for arch in args.arch:
        points = []
        for path in sorted((ROOT / 'tests' / 'cases').glob(args.cases)):
            if path.name.startswith('_'):
                continue
            try:
                built = generate(path, arch)
                if built is None or not built[0].code_units:
                    continue
                generator, source = built
                with tempfile.TemporaryDirectory() as tmp:
                    count = (sass_instructions if arch.startswith('sm_')
                             else isa_instructions)(source, arch, Path(tmp))
            except Exception as error:   # a case this target does not build
                print(f'  {path.stem}: skipped ({type(error).__name__})',
                      file=sys.stderr)
                continue
            points.append((path.stem, generator.code_units, count))
        if not points:
            print(f'{arch}: nothing measured')
            continue
        # least squares through the origin: count = a * units
        a = (sum(u * c for _, u, c in points)
             / sum(u * u for _, u, _ in points))
        errors = sorted((c - a * u) / c for _, u, c in points if c)
        print(f'{arch}: {len(points)} kernels, instructions per unit {a:.3f}; '
              f'relative error median {statistics.median(errors):+.2f}, '
              f'10/90 % {errors[len(errors) // 10]:+.2f} / '
              f'{errors[(9 * len(errors)) // 10]:+.2f}')
        for name, units, count in sorted(points, key=lambda p: -p[2])[:8]:
            print(f'   {name:40s} units {units:7d}  instructions {count:7d}  '
                  f'ratio {count / units:.2f}')


if __name__ == '__main__':
    main()
