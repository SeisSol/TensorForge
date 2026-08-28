# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Regenerate `tests/data/amd_matrix_builtins.json` from LLVM.

The catalogue in `primitives/amd/catalog.py` states shapes, block counts and
fragment widths.  Those are facts about instructions LLVM already describes,
so they are a copy -- and copies drift.  This extracts the same facts from
LLVM's own sources into a vendored table, which `tests/test_amd_catalog.py`
checks the catalogue against.  Vendored rather than fetched, so the test suite
stays offline and a change to the table shows up in review as a diff.

Usage::

    python tools/amd_matrix_table.py --llvm <path-to-llvm-project>
    python tools/amd_matrix_table.py --fetch          # from github main

Three files are read:

* ``clang/include/clang/Basic/BuiltinsAMDGPU.td`` -- the builtin signatures,
  which give the per-lane fragment widths and element types.
* ``llvm/lib/Target/AMDGPU/AMDGPU.td`` -- the subtarget feature sets.
* ``llvm/lib/Target/AMDGPU/GCNProcessors.td`` -- which processor has which
  feature set.
"""

from __future__ import annotations

import argparse
import json
import re
import urllib.request
from pathlib import Path

RAW = 'https://raw.githubusercontent.com/llvm/llvm-project/main/'
SOURCES = {
    'builtins': 'clang/include/clang/Basic/BuiltinsAMDGPU.td',
    'features': 'llvm/lib/Target/AMDGPU/AMDGPU.td',
    'processors': 'llvm/lib/Target/AMDGPU/GCNProcessors.td',
}

#: Element type as clang spells it -> (our name, bits).  A `short` operand is
#: bf16: the pre-gfx950 bf16 builtins predate clang's `__bf16` and take the
#: bit pattern as an integer.  Which one it is follows from the builtin name,
#: not from the type, so the name is what decides below.
ELEMENTS = {'float': 'f32', 'double': 'f64', '_Float16': 'f16',
            '__bf16': 'bf16', 'short': 'i16', 'int': 'i32'}


def _read(args, which):
    if args.llvm:
        return (Path(args.llvm) / SOURCES[which]).read_text()
    with urllib.request.urlopen(RAW + SOURCES[which]) as fh:
        return fh.read().decode()


# --------------------------------------------------------------------------- #
# features -> processors
# --------------------------------------------------------------------------- #

def feature_map(features_td, processors_td):
    """Every ``gfxNNNN`` that carries each subtarget feature.

    The feature sets nest -- ``FeatureISAVersion12_51`` is a `listconcat` over
    ``FeatureISAVersion12_50_Common`` -- so the references have to be followed
    rather than read off one record.
    """
    sets = {}
    for m in re.finditer(
            r'def\s+(\w+)\s*:\s*(?:FeatureSet|GCNSubtargetFeatureGeneration)<',
            features_td):
        depth, i = 1, m.end()
        while depth and i < len(features_td):
            depth += {'<': 1, '>': -1}.get(features_td[i], 0)
            i += 1
        body = features_td[m.end():i - 1]
        sets[m.group(1)] = (re.findall(r'\bFeature\w+\b', body),
                            re.findall(r'(\w+)\.Features', body))

    def resolve(name, seen=None):
        seen = set() if seen is None else seen
        if name in seen or name not in sets:
            return set()
        seen.add(name)
        feats, refs = sets[name]
        out = set(feats)
        for ref in refs + [f for f in feats if f in sets]:
            out |= resolve(ref, seen)
        return out

    out = {}
    for m in re.finditer(
            r'ProcessorModel<\s*"([^"]+)"\s*,\s*\w+\s*,\s*(\w+)\.Features',
            processors_td):
        target, version = m.group(1), m.group(2)
        if not re.fullmatch(r'gfx[0-9a-f]+', target):
            continue        # `gfx11-generic`, `gfx1250-strict`, ...
        for feature in resolve(version):
            out.setdefault(feature, []).append(target)
    return {k: sorted(set(v)) for k, v in out.items()}


#: tablegen record name -> the string the builtin definitions use.  Only the
#: ones a matrix builtin is gated on; anything else is dropped.
FEATURE_NAMES = {
    'FeatureMAIInsts': 'mai-insts',
    'FeatureGFX90AInsts': 'gfx90a-insts',
    'FeatureGFX940Insts': 'gfx940-insts',
    'FeatureGFX950Insts': 'gfx950-insts',
    'FeatureXF32Insts': 'xf32-insts',
    'FeatureWMMA256bInsts': 'wmma-256b-insts',
    'FeatureWMMA128bInsts': 'wmma-128b-insts',
    'FeatureWMMAN16Insts': 'wmma-n16-insts',
    'FeatureGFX1250Insts': 'gfx1250-insts',
    'FeatureGFX1251GEMMInsts': 'gfx1251-gemm-insts',
    'FeatureSWMMACGfx1250Insts': 'swmmac-gfx1250-insts',
    'FeatureGFX125xLowestRateWMMA': 'gfx125x-lowest-rate-wmma',
}


# --------------------------------------------------------------------------- #
# builtins
# --------------------------------------------------------------------------- #

def _operand(text):
    """``_ExtVector<4, short>`` -> ``(4, 'i16')``; ``double`` -> ``(1, 'f64')``."""
    text = text.strip()
    m = re.fullmatch(r'_ExtVector<(\d+),\s*([\w_]+)>', text)
    if m:
        return int(m.group(1)), ELEMENTS.get(m.group(2), m.group(2))
    return (1, ELEMENTS[text]) if text in ELEMENTS else None


#: `mfma_f32_4x4x4bf16_1k` and `mfma_f32_4x4x2bf16` differ in operand width,
#: not in element format: `_1k` is gfx90a's four-wide bf16 operand, gfx908's
#: is two-wide.  The width is already in `a`/`b`, so the suffix is dropped
#: here rather than becoming a second spelling of `bf16`.
def _dtype(name):
    return re.sub(r'_1k$|_+$', '', name)


def builtins(builtins_td):
    rows = []
    pattern = (r'def (__builtin_amdgcn_(?:mfma|wmma)_\w+) : '
               r'AMDGPUBuiltin<"([^"]+)"[^"]*"([^"]*)">\s*(\{.*?\n\})?')
    for m in re.finditer(pattern, builtins_td, re.S):
        name, sig, features = m.group(1), m.group(2), m.group(3)
        # `ArgNames` is what makes the call shape checkable instead of
        # inferred from the position of the `_Constant` arguments.  gfx1250
        # interleaves them -- `a_neg, a, b_neg, b, c_mod, c, ...` -- so
        # counting constants would place the accumulator wrongly.
        argnames = re.search(r'let ArgNames = \[([^\]]*)\]', m.group(4) or '')
        argnames = re.findall(r'"([^"]+)"', argnames.group(1)) if argnames else []
        shape = re.search(
            r'(mfma|wmma)_(\w+?)_(\d+)x(\d+)x(\d+)_?([a-z0-9_]*?)'
            r'(?:_w(32|64))?(?:_gfx12)?$', name)
        if shape is None:
            continue
        ret, arglist = re.fullmatch(r'([^(]+)\((.*)\)', sig).groups()
        args = [a.strip() for a in re.split(r',\s*(?![^<]*>)', arglist)]
        operands = [_operand(a) for a in args]
        operands = [o for o in operands if o is not None]
        if len(operands) < 3:
            continue
        wave = 64
        if 'wavefrontsize32' in features:
            wave = 32
        elif shape.group(7):
            wave = int(shape.group(7))
        rows.append({
            'builtin': name.replace('__builtin_amdgcn_', ''),
            'feature': features.split(',')[0],
            'wave': wave,
            'kind': shape.group(1),
            'm': int(shape.group(3)),
            'n': int(shape.group(4)),
            'k': int(shape.group(5)),
            'out': _dtype(shape.group(2)),
            'in': _dtype(shape.group(6) or shape.group(2)),
            'a': list(operands[0]),
            'b': list(operands[1]),
            'd': list(_operand(ret)),
            'args': args,
            'argnames': argnames,
        })
    return sorted(rows, key=lambda r: r['builtin'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--llvm', help='path to an llvm-project checkout')
    ap.add_argument('--fetch', action='store_true', help='read from github main')
    ap.add_argument('-o', '--out', default='tests/data/amd_matrix_builtins.json')
    args = ap.parse_args()
    if not args.llvm and not args.fetch:
        ap.error('pass --llvm <path> or --fetch')

    feats = feature_map(_read(args, 'features'), _read(args, 'processors'))
    table = {
        'note': 'generated by tools/amd_matrix_table.py -- do not edit by hand',
        'features': {name: feats.get(record, [])
                     for record, name in sorted(FEATURE_NAMES.items(),
                                                key=lambda kv: kv[1])},
        'builtins': builtins(_read(args, 'builtins')),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(table, indent=1) + '\n')
    print(f'{out}: {len(table["builtins"])} builtins, '
          f'{len(table["features"])} features')


if __name__ == '__main__':
    main()
