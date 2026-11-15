# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Regenerate `tests/data/amd_matrix_layouts.json` from AMD's calculator.

`catalog.py` says how *many* elements of each operand a lane holds.  It does
not say *which*, and which is what an emitter needs: the fragment layout is
the difference between a correct kernel and one that reads the wrong lanes
with correctly typed registers.

AMD publishes those layouts also via a tool --- the matrix
instruction calculator, https://github.com/ROCm/amd_matrix_instruction_calculator.
This drives it and vendors what it says, for the same reason
`amd_matrix_table.py` vendors LLVM: the suite stays offline, the derived facts
show up in review as a diff, and nothing in code generation grows a dependency
on a 340 kB argparse script.

The tool covers CDNA1--3 and RDNA3--4.  It does not know gfx950, gfx1250 or
gfx1251, so it cannot be the only source and the generated table records the
gap instead of hiding it.

For gfx1250 AMD publishes the layouts in a document instead: the CDNA 5 ISA
reference guide tabulates where each element of a WMMA operand is stored
(`ISA_GUIDE`).  Those tables are transcribed below as what they say for one
element -- its slot and lane -- and expanded here into the same form, in a
section of their own (`documented`), so that a reviewer compares the
transcription with the guide rather than with a bit table.  The guide does
not list gfx1251's FP64 WMMA, which stays out.

Usage::

    python tools/amd_matrix_layouts.py --calculator ../amd_matrix_instruction_calculator
    python tools/amd_matrix_layouts.py --isa-only    # the guide's section alone

Every layout it reports is *linear in the index bits*: flipping one bit of a
matrix index moves the element by a fixed amount in (slot, lane), independent
of the other bits.  That is checked here rather than assumed, and it is what
makes the vendored form compact --- one (slot, lane) pair per index bit
instead of one row per element.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

#: Feature -> the name the calculator knows the architecture by.  A feature
#: absent here is one the calculator does not cover.
CALCULATOR_ARCH = {
    'mai-insts': 'CDNA1',
    'gfx90a-insts': 'CDNA2',
    'gfx940-insts': 'CDNA3',
    'xf32-insts': 'CDNA3',
    'wmma-256b-insts': 'RDNA3',
    'wmma-128b-insts': 'RDNA4',
}

#: `A[3][1].B2` -- matrix, first index, second index, block.
_CELL = re.compile(r'([ABD])\[(\d+)\](?:\[(\d+)\])?(?:\.B(\d+))?')


# --------------------------------------------------------------------------- #
# The CDNA 5 ISA reference guide
# --------------------------------------------------------------------------- #

#: Where the transcribed layouts come from.
ISA_GUIDE = {
    'document': '"CDNA5" Instruction Set Architecture: Reference Guide, '
                '27 July 2026',
    'url': 'https://www.amd.com/content/dam/amd/en/documents/'
           'instinct-tech-docs/instruction-set-architectures/'
           'amd-instinct-cdna5-instruction-set-architecture.pdf',
    'section': '7.12.2 Matrix Element Storage in VGPRs',
    'slots': 'an element of the operand type, in the order the builtin\'s '
             'vector holds them: a 16-bit element is half `slot % 2` of VGPR '
             '`slot // 2`, an 8-bit one byte `slot % 4` of VGPR `slot // 4`',
}


def _isa_a_32bit_16x4(row, k):
    """"32-bit A-Matrix 16x4": lanes 0-15 are rows 0-15 and hold K=0 in the
    first VGPR, K=1 in the second; lanes 16-31 hold K=2 and K=3 the same
    way."""
    return k % 2, row + 16 * (k // 2)


def _isa_a_16bit_16x32(row, k):
    """"16-bit A-Matrix 16x32": two elements a VGPR, the lower in [15:0];
    VGPRs 0-3 hold K=0-7 in lanes 0-15 and K=8-15 in lanes 16-31, VGPRs 4-7
    the same for K=16-31."""
    return k % 8 + 8 * (k // 16), row + 16 * (k // 8 % 2)


def _isa_a_8bit_16x64(row, k):
    """"8-bit A-Matrix 16x64": four elements a VGPR, the lowest in [7:0];
    VGPRs 0-1 hold K=0-7 in lanes 0-15 and K=8-15 in lanes 16-31, VGPRs 2-3
    K=16-23 and K=24-31, VGPRs 4-7 the same 32 higher."""
    return k % 8 + 8 * (k // 16), row + 16 * (k // 8 % 2)


def _isa_cd_32bit_16x16(m, n):
    """"32-bit C/D-Matrix 16x16": lanes 0-15 are columns 0-15 and hold rows
    0-7, one a VGPR; lanes 16-31 the same columns, rows 8-15."""
    return m % 8, n + 16 * (m // 8)


#: builtin -> (opcode, M, N, K, the guide's A table, its C/D table).  The
#: guide's A tables describe B as well, with N for M: the same positions with
#: the two indices swapped (`isa_positions`).
ISA_GUIDE_LAYOUTS = {
    'wmma_f32_16x16x4_f32': ('v_wmma_f32_16x16x4_f32', 16, 16, 4,
                             _isa_a_32bit_16x4, _isa_cd_32bit_16x16),
    'wmma_f32_16x16x32_bf16': ('v_wmma_f32_16x16x32_bf16', 16, 16, 32,
                               _isa_a_16bit_16x32, _isa_cd_32bit_16x16),
    'wmma_f32_16x16x32_f16': ('v_wmma_f32_16x16x32_f16', 16, 16, 32,
                              _isa_a_16bit_16x32, _isa_cd_32bit_16x16),
    'wmma_i32_16x16x64_iu8': ('v_wmma_i32_16x16x64_iu8', 16, 16, 64,
                              _isa_a_8bit_16x64, _isa_cd_32bit_16x16),
}


def isa_positions(builtin):
    """`{'A': {(0, m, k): [(slot, lane)]}, 'B': ..., 'D': ...}` for one
    transcribed instruction, in the form `read_layout` returns."""
    _, m, n, k, a_table, d_table = ISA_GUIDE_LAYOUTS[builtin]
    return {
        'A': {(0, i, kk): [a_table(i, kk)]
              for i in range(m) for kk in range(k)},
        'B': {(0, kk, j): [a_table(j, kk)]
              for kk in range(k) for j in range(n)},
        'D': {(0, i, j): [d_table(i, j)]
              for i in range(m) for j in range(n)},
    }


def isa_section():
    """The `documented` section of the table: every transcribed layout,
    compressed and checked as the calculator's are."""
    layouts = {}
    for builtin, (opcode, *_) in ISA_GUIDE_LAYOUTS.items():
        entry = {'arch': 'CDNA5', 'wave': 32, 'opcode': opcode}
        for which, positions in isa_positions(builtin).items():
            base, deltas = bit_deltas(positions, 2)
            entry[which] = {'base': base, 'bits': deltas,
                            'replication': replication_of(positions)}
        layouts[builtin] = entry
    return {'source': ISA_GUIDE, 'layouts': layouts}


def mnemonic(builtin: str) -> str:
    """The ISA opcode behind a clang builtin name.

    They agree for MFMA and not for WMMA: `_w32`, `_w64` and `_gfx12` are
    clang's way of naming the overload, and the hardware has one opcode whose
    wave the calculator takes as `-w`.  Getting this wrong is not quiet --- the
    calculator refuses an unknown opcode --- which is why it is a rewrite here
    rather than a second name in the catalog.
    """
    return 'v_' + re.sub(r'_(?:w32|w64)(?:_gfx12)?$|_gfx12$', '', builtin)


def read_layout(calculator, arch, instruction, which, wave=None):
    """`{(block, i, j): [(slot, lane), ...]}` for one operand.

    A list, not a single position, because the RDNA 3 fragments hold each
    element more than once: lanes 0--15 and 16--31 carry the same 16x16x16
    operand at wave32.  Keeping only the last one seen puts the origin of the
    layout at lane 16 and makes the base look like an offset, which is a
    parsing artifact dressed as a hardware fact.
    """
    cmd = [sys.executable, str(calculator), '-a', arch, '-i', instruction,
           f'-{which}', '--matrix-layout', '--csv']
    if wave is not None:
        cmd += ['-w', str(wave)]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if res.returncode:
        raise RuntimeError(f'{" ".join(cmd)}\n{res.stderr or res.stdout}')
    lines = res.stdout.splitlines()
    head = next((i for i, l in enumerate(lines) if l.startswith('lane,')), None)
    if head is None:
        raise RuntimeError(f'no matrix layout in output of {" ".join(cmd)}')
    out = {}
    for line in lines[head + 1:]:
        if not line[:1].isdigit():
            continue
        cells = line.split(',')
        lane = int(cells[0])
        for slot, cell in enumerate(cells[1:]):
            m = _CELL.match(cell.strip())
            if m is None:
                continue           # an unused slot prints as blank
            _, first, second, block = m.groups()
            index = (int(first),) if second is None else (int(first), int(second))
            out.setdefault((int(block or 0),) + index, []).append((slot, lane))
    return out


def replication_of(positions):
    """How many (slot, lane) slots hold each element.

    Uniform by construction on every instruction the tool knows; asserted
    anyway, because a non-uniform one would make `min()` below an arbitrary
    choice rather than a canonical one.  Cross-checks
    `MatrixOp.replication()`, which comes from LLVM's operand widths --- two
    independent sources for the same number.
    """
    counts = {len(v) for v in positions.values()}
    if len(counts) != 1:
        raise ValueError(f'element multiplicity is not uniform: {sorted(counts)}')
    return counts.pop()


def bit_deltas(positions, ndims):
    """Compress a layout to one (slot, lane) contribution per index bit.

    Returns `(base, deltas)` where `deltas[d][b]` is what setting bit `b` of
    dimension `d` adds, dimension 0 being the block.  Raises if the layout is
    not linear in the bits, because then this form would be a lossy summary
    rather than the same fact written shorter.
    """
    positions = {k: min(v) for k, v in positions.items()}
    zero = tuple([0] * (ndims + 1))
    base = positions[zero]
    deltas = []
    for dim in range(ndims + 1):
        extent = max(key[dim] for key in positions) + 1
        bits, place = [], 1
        while place < extent:
            key = [0] * (ndims + 1)
            key[dim] = place
            slot, lane = positions[tuple(key)]
            bits.append([slot - base[0], lane - base[1]])
            place *= 2
        deltas.append(bits)

    for key, expected in positions.items():
        slot, lane = base
        for dim, value in enumerate(key):
            for bit, (dslot, dlane) in enumerate(deltas[dim]):
                if value >> bit & 1:
                    slot += dslot
                    lane += dlane
        if (slot, lane) != expected:
            raise ValueError(f'layout is not linear in the index bits at {key}: '
                             f'model says {(slot, lane)}, tool says {expected}')
    return list(base), deltas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--calculator',
                    help='path to an amd_matrix_instruction_calculator checkout '
                         'or to matrix_calculator.py itself')
    ap.add_argument('--isa-only', action='store_true',
                    help='rewrite the `documented` section from the ISA '
                         'guide\'s transcription alone, keeping the rest')
    ap.add_argument('-o', '--out', default='tests/data/amd_matrix_layouts.json')
    args = ap.parse_args()

    if args.isa_only:
        # The calculator's part is left as it was generated: the guide's
        # transcription is what changes here, and nothing else has to run.
        out = Path(args.out)
        table = json.loads(out.read_text())
        table['documented'] = isa_section()
        out.write_text(json.dumps(table, indent=2) + '\n')
        print(f'{out}: {len(table["documented"]["layouts"])} documented '
              f'layouts')
        return
    if args.calculator is None:
        ap.error('--calculator is required unless --isa-only')

    calc = Path(args.calculator)
    if calc.is_dir():
        calc = calc / 'matrix_calculator.py'
    if not calc.is_file():
        ap.error(f'{calc} not found')

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
    from tensorforge.backend.instructions.compute.primitives.amd import catalog

    version = subprocess.run([sys.executable, str(calc), '-v'],
                             capture_output=True, text=True).stdout.strip()

    covered, uncovered = {}, {}
    for op in catalog.MATRIX_OPS:
        arch = CALCULATOR_ARCH.get(op.feature)
        if arch is None:
            uncovered.setdefault(op.feature, []).append(op.builtin)
            continue
        wave = op.wave if op.call is not catalog.Call.MFMA else None
        entry = {'arch': arch, 'wave': op.wave, 'opcode': mnemonic(op.builtin)}
        for which in ('A', 'B', 'D'):
            positions = read_layout(calc, arch, entry['opcode'], which, wave)
            base, deltas = bit_deltas(positions, 2)
            entry[which] = {'base': base, 'bits': deltas,
                            'replication': replication_of(positions)}
        covered[op.builtin] = entry

    table = {
        'note': 'generated by tools/amd_matrix_layouts.py -- do not edit by hand',
        'source': 'https://github.com/ROCm/amd_matrix_instruction_calculator',
        'source_version': version,
        'bits': 'per index bit, the (slot, lane) it contributes; '
                'dimension order is (block, first index, second index)',
        'not_covered': {k: sorted(v) for k, v in sorted(uncovered.items())},
        'layouts': covered,
        'documented': isa_section(),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(table, indent=2) + '\n')
    print(f'{out}: {len(covered)} layouts, '
          f'{sum(len(v) for v in uncovered.values())} instructions not covered')


if __name__ == '__main__':
    main()
