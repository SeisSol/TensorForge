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

Usage::

    python tools/amd_matrix_layouts.py --calculator ../amd_matrix_instruction_calculator

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


def mnemonic(builtin: str) -> str:
    """The ISA opcode behind a clang builtin name.

    They agree for MFMA and not for WMMA: `_w32`, `_w64` and `_gfx12` are
    clang's way of naming the overload, and the hardware has one opcode whose
    wave the calculator takes as `-w`.  Getting this wrong is not quiet --- the
    calculator refuses an unknown opcode --- which is why it is a rewrite here
    rather than a second name in the catalogue.
    """
    return 'v_' + re.sub(r'_(?:w32|w64)(?:_gfx12)?$|_gfx12$', '', builtin)


def read_layout(calculator, arch, instruction, which, wave=None):
    """`{(block, i, j): [(slot, lane), ...]}` for one operand.

    A list, not a single position, because the RDNA 3 fragments hold each
    element more than once: lanes 0--15 and 16--31 carry the same 16x16x16
    operand at wave32.  Keeping only the last one seen puts the origin of the
    layout at lane 16 and makes the base look like an offset, which is a
    parsing artefact dressed as a hardware fact.
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
    ap.add_argument('--calculator', required=True,
                    help='path to an amd_matrix_instruction_calculator checkout '
                         'or to matrix_calculator.py itself')
    ap.add_argument('-o', '--out', default='tests/data/amd_matrix_layouts.json')
    args = ap.parse_args()

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
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(table, indent=1) + '\n')
    print(f'{out}: {len(covered)} layouts, '
          f'{sum(len(v) for v in uncovered.values())} instructions not covered')


if __name__ == '__main__':
    main()
