# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Fit machine instructions per emitted statement, by category.

`analysis.pipeline` bounds a kernel's time by the busiest pipe, from the
statements the emitter counted per category (`Context.record_mix`).  How many
machine instructions a statement becomes is the one thing there that is not
derived from the program: an offset folds into a load, a division becomes a
sequence, an address computation disappears into the addressing mode.  This
fits that factor per category against the case set: each case is generated,
compiled on its own (nvcc and `cuobjdump -sass`; hipcc to device assembly),
and its instructions sorted into the same categories by opcode.  The factor is
the ratio of the totals, over the copies laid down (static, as compiled), with
the spread per kernel beside it.

  python tools/calibrate_mix.py --arch sm_120 [--arch gfx942 ...] [--cases GLOB]
"""

import argparse
import collections
import re
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibrate_icache import INCLUDE, ROOT, generate  # noqa: E402

#: SASS opcode (without modifiers) -> category.
SASS = [
    (r'^(HMMA|IMMA|DMMA|BMMA|HGMMA|UTC\w*)$', 'matrix'),
    (r'^D(FMA|ADD|MUL|MNMX|SETP|SET)$', 'fp64'),
    (r'^(FFMA2?|FADD2?|FMUL2?|FMNMX|FSEL|FSETP|FSET|FCHK|FRND|F2F|HFMA2|HADD2'
     r'|HMUL2|HMNMX2|FSWZADD)$', 'fp'),
    (r'^MUFU$', 'sfu'),
    (r'^(SHFL|VOTE|VOTEU|MATCH|REDUX)$', 'xlane'),
    (r'^(LDG|LD)$', 'global.load'),
    (r'^(STG|ST|RED|ATOM|ATOMG)$', 'global.store'),
    (r'^(LDC|LDCU)$', 'constant.load'),
    (r'^(NOP)$', 'padding'),
    (r'^(ENDCOLLECTIVE)$', 'sync'),
    (r'^(LDS|LDSM)$', 'shared.load'),
    (r'^(STS|ATOMS)$', 'shared.store'),
    (r'^LDL$', 'local.load'),
    (r'^STL$', 'local.store'),
    (r'^(LDGSTS|UBLKCP|UTMALDG)$', 'async.copy'),
    (r'^(CCTL|CCTLL)$', 'global.prefetch'),
    (r'^(BAR|WARPSYNC)$', 'barrier'),
    (r'^(DEPBAR|MEMBAR|LDGDEPBAR|ERRBAR|ARRIVES)$', 'sync'),
    (r'^(BRA|BRX|EXIT|RET|CALL|JMP|BSSY|BSYNC|YIELD|BREAK)$', 'branch'),
    (r'^(U?IMAD\w*|U?IADD3|IADD|U?LEA|U?SHF|SHL|SHR|U?LOP3|LOP|U?ISETP|IMNMX|U?SEL'
     r'|U?MOV|IABS|I2F\w*|F2I\w*|U?PRMT|SGXT|BMSK|POPC|FLO|S2R|S2UR|CS2R|ULDC'
     r'|IMUL|VIADD|VIMNMX|R2UR|P2R|R2P|U?PLOP3|I2I\w*|F2FP\w*|IDP\w*)$', 'int'),
]

#: AMD mnemonic -> category, first match wins.
ISA = [
    (r'^v_(mfma|smfmac|wmma|swmmac)', 'matrix'),
    (r'^v_(exp|log|rcp|rsq|sqrt|sin|cos)_', 'sfu'),
    (r'^v_\w*_f64', 'fp64'),
    (r'^v_(pk_)?(fma|fmac|mac|add|sub|subrev|mul|min|max|dot2\w*|med3|ldexp|cndmask'
     r'|cvt_pk)\w*_(f32|f16|bf16)', 'fp'),
    (r'^v_\w*_(f32|f16|bf16)', 'fp'),
    (r'^(v_readlane|v_readfirstlane|v_writelane|v_permlane|ds_swizzle|ds_bpermute'
     r'|ds_permute|v_mov_b32_dpp|v_mov_b64_dpp)', 'xlane'),
    (r'^(ds_read|ds_load)', 'shared.load'),
    (r'^(ds_write|ds_store|ds_add)', 'shared.store'),
    (r'^scratch_load', 'local.load'),
    (r'^scratch_store', 'local.store'),
    (r'^(global_load_lds|buffer_load_\w*lds)', 'async.copy'),
    (r'^(global_load|buffer_load|flat_load)', 'global.load'),
    (r'^(global_store|buffer_store|flat_store|global_atomic|buffer_atomic)',
     'global.store'),
    (r'^s_(load|buffer_load)', 'constant.load'),
    (r'^s_barrier', 'barrier'),
    (r'^(s_waitcnt|s_wait_|s_sleep)', 'sync'),
    (r'^s_(cbranch|branch|setpc|endpgm|getpc)', 'branch'),
    (r'^(v_|s_)', 'int'),
]


def classify(opcode: str, table) -> str:
    for pattern, category in table:
        if re.match(pattern, opcode):
            return category
    return 'other'


def sass_mix(source: str, arch: str, tmp: Path) -> collections.Counter:
    cu, cubin = tmp / 'k.cu', tmp / 'k.cubin'
    cu.write_text(source)
    subprocess.run(['nvcc', '-O3', '-std=c++17', '--expt-relaxed-constexpr',
                    f'-arch={arch}', '-cubin', '-I', str(INCLUDE),
                    '-o', str(cubin), str(cu)], check=True, capture_output=True)
    sass = subprocess.run(['cuobjdump', '-sass', str(cubin)], check=True,
                          capture_output=True, text=True).stdout
    mix = collections.Counter()
    for m in re.finditer(r'^\s+/\*[0-9a-f]{4,}\*/\s+(?:@!?U?P\w+\s+)?([A-Z0-9_]+)',
                         sass, re.M):
        mix[classify(m.group(1).split('.')[0], SASS)] += 1
    return mix


def isa_mix(source: str, arch: str, tmp: Path) -> collections.Counter:
    hip, asm = tmp / 'k.hip', tmp / 'k.s'
    hip.write_text(source)
    subprocess.run(['hipcc', '-O3', f'--offload-arch={arch}', '-x', 'hip',
                    '--cuda-device-only', '-S', '-I', str(INCLUDE),
                    '-o', str(asm), str(hip)], check=True, capture_output=True)
    mix, inside = collections.Counter(), False
    for line in asm.read_text().splitlines():
        if re.match(r'^_Z\w*kernel_\w*:', line):
            inside = True
        elif inside and line.startswith('.Lfunc_end'):
            inside = False
        elif inside:
            m = re.match(r'^\s+([a-z][a-z0-9_]*)(\s|$)', line)
            if m:
                mix[classify(m.group(1), ISA)] += 1
    return mix


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--arch', action='append', required=True)
    parser.add_argument('--cases', default='**/*.py',
                        help='glob under tests/cases (default: all)')
    args = parser.parse_args(argv)
    for arch in args.arch:
        emitted, measured, per_kernel = (collections.Counter(),
                                         collections.Counter(), [])
        for path in sorted((ROOT / 'tests' / 'cases').glob(args.cases)):
            if path.name.startswith('_'):
                continue
            try:
                built = generate(path, arch)
                if built is None or not built[0].issue_mix:
                    continue
                generator, source = built
                with tempfile.TemporaryDirectory() as tmp:
                    mix = (sass_mix if arch.startswith('sm_') else isa_mix)(
                        source, arch, Path(tmp))
            except Exception as error:   # a case this target does not build
                print(f'  {path.stem}: skipped ({type(error).__name__})',
                      file=sys.stderr)
                continue
            copies = {c: v[1] for c, v in generator.issue_mix.items()}
            emitted.update(copies)
            measured.update(mix)
            per_kernel.append((path.stem, copies, mix))
        if not per_kernel:
            print(f'{arch}: nothing measured')
            continue
        print(f'{arch}: {len(per_kernel)} kernels; machine instructions per '
              f'emitted statement (ratio of totals, per-kernel median and 10/90 %)')
        for category in sorted(set(emitted) | set(measured),
                               key=lambda c: -measured.get(c, 0)):
            e, m = emitted.get(category, 0), measured.get(category, 0)
            ratios = sorted(k[2].get(category, 0) / k[1][category]
                            for k in per_kernel if k[1].get(category))
            spread = (f'{statistics.median(ratios):.2f} '
                      f'[{ratios[len(ratios) // 10]:.2f}, '
                      f'{ratios[(9 * len(ratios)) // 10]:.2f}]'
                      if ratios else '-')
            factor = f'{m / e:.3f}' if e else '  (none emitted)'
            print(f'   {category:16s} emitted {e:8d}  compiled {m:8d}  '
                  f'factor {factor:>8s}  {spread}')


if __name__ == '__main__':
    main()
