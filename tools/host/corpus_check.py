# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Build kernels of a corpus file under several sets of options, and check
every build against NumPy and against the first.

    python3 corpus_check.py ../bench/data/viscoelastic-4-d-s1.json gpu_derivative
    python3 corpus_check.py ../bench/data/viscoelastic-4-d-s1.json \\
        --build merge_variants=1 --build merge_variants=1,rotate_slots=1

A build is a set of `Options`, comma-separated; without `--build`, the list
is built written out (`merge_variants=0`) and merged (`merge_variants=1`).
Without kernels, every kernel of the file.

Each build runs on the host oracle (`tensorforge.reference.kernel_eval`, all
lanes of the block, CUDA only), from the same memory: the oracle's seeded
fill, a function of the slot alone, and the operators' own numbers where the
description has them -- a build may inline a literal or skip a known zero,
so memory has to hold what the description says.  The NumPy evaluation of
the descriptor list (`tensorforge.reference.descriptors.apply`) starts from
that memory too, outputs included: `+=` adds onto what was there.

Two comparisons per build, over every slot of every output: against NumPy,
and against the first build.  The second needs no reference, so it stands
where NumPy cannot follow -- a permuted operand, an operation other than a
product.  Exit status 1 where a build disagrees or the oracle cannot run it.
"""
import argparse
import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))

from tensorforge.common.basic_types import Addressing, Datatype  # noqa: E402
from tensorforge.common.context import Context, Options  # noqa: E402
from tensorforge.frontend.yateto import DescriptionReader  # noqa: E402
from tensorforge.generators.descriptions import (  # noqa: E402
    MultilinearDescr, OperationDescription)
from tensorforge.generators.generator import Generator  # noqa: E402
from tensorforge.reference import descriptors as reference  # noqa: E402
from tensorforge.reference import kernel_eval  # noqa: E402

SEED = 29
#: Both sides compute in double precision on the host, so anything above
#: rounding is a disagreement.
TOLERANCE = 1e-9


def parse_build(text):
    """`name=value,...` as keyword arguments of `Options`."""
    out = {}
    for item in filter(None, text.split(',')):
        name, _, value = item.partition('=')
        out[name.strip()] = _value(value.strip())
    return out


def _value(text):
    for conv in (int, float):
        try:
            return conv(text)
        except ValueError:
            pass
    return {'true': True, 'false': False}.get(text.lower(), text)


def _slots(tensor):
    """Storage slot -> logical cell, over what the tensor stores."""
    box = tuple(int(x) for x in tensor.get_actual_shape())
    lower = tuple(int(x) for x in tensor.bbox.lower())
    smap = tensor.storage_map()
    cells = range(int(np.prod(box))) if smap is None else smap
    return {s: tuple(lo + c for lo, c in zip(
                lower, np.unravel_index(int(cell), box, order='F')))
            for s, cell in enumerate(cells) if cell is not None and cell >= 0}


def _row(view):
    return {'name': view.tensor.alias,
            'bbox': [[int(x) for x in view.bbox.lower()],
                     [int(x) for x in view.bbox.upper()]],
            'offset': [int(x) for x in view.offset],
            'tbbox': [[int(x) for x in view.tensor.bbox.lower()],
                      [int(x) for x in view.tensor.bbox.upper()]],
            'sliced': bool(view.sliced), 'is_tmp': bool(view.tensor.is_tmp)}


class Kernel:
    """One kernel of a corpus file, and the memory every build of it starts
    from."""

    def __init__(self, description, dtype, arch):
        self._description = description
        self._dtype = dtype
        self._arch = arch
        #: Alias -> the name its buffer has in every build, so that the seeded
        #: fill gives each tensor the same numbers whatever a build calls it.
        self._names = {}
        listing = [d for d in self.descrs()
                   if isinstance(d, OperationDescription)
                   and d.writes() is not None]
        self.products = [d for d in listing if isinstance(d, MultilinearDescr)]
        self.tensors = {}
        for d in listing:
            for m in d.matrix_list():
                t = getattr(m, 'tensor', m)
                self.tensors.setdefault(t.alias, t)
        self.written = {d.writes().tensor.alias for d in listing}
        others = [type(d).__name__ for d in listing
                  if not isinstance(d, MultilinearDescr)]
        permuted = any(list(p) != sorted(p) for d in self.products
                       for p in d.permute)
        #: Why NumPy cannot evaluate the list, or None.
        self.unevaluable = (f'{others[0]} is not a product' if others
                            else 'a permuted operand' if permuted else None)

    def descrs(self):
        return DescriptionReader(arch=None).read(self._description)[0]

    def _name(self, alias):
        return f'm{900 + self._names.setdefault(alias, len(self._names))}'

    @staticmethod
    def _sub(tensor):
        """The part of a buffer's name past `@`: a pointer-based tensor's
        element, of which the oracle runs the first."""
        return '0' if tensor.addressing == Addressing.PTR_BASED else ''

    def initial(self, alias, slot):
        """What memory holds in `slot` of `alias` before the kernel runs."""
        tensor = self.tensors[alias]
        sub = self._sub(tensor)
        base = self._name(alias) + (f'@{sub}' if sub else '')
        return kernel_eval.Slot(SEED).read(base, slot)

    def run(self, options):
        """What a build leaves in the outputs, (alias, slot) -> value."""
        gen = Generator(self.descrs(), Context(
            arch=self._arch, backend='cuda', fp_type=self._dtype,
            options=Options(**options)))
        gen.generate()
        names, tensors = {}, {}
        for outer in gen.descr_list:
            for d in outer.operations():
                for m in d.matrix_list():
                    t = getattr(m, 'tensor', m)
                    if not getattr(t, 'is_variant', False):
                        names.setdefault(t.name, t.alias)
                        tensors.setdefault(t.name, t)
        canonical = {n: self._name(a) for n, a in names.items()}
        preset = {canonical[n]: [float(v) for v in t.storage_values()]
                  for n, t in tensors.items()
                  if t.addressing == Addressing.NONE
                  and t.storage_values() is not None}
        src = re.sub(r'\bm\d+\b',
                     lambda m: canonical.get(m.group(0), m.group(0)),
                     gen.get_kernel())
        lanes, mults = kernel_eval.launch_geometry(gen.get_launcher())
        mem = kernel_eval.evaluate_wave(src, lanes, seed=SEED,
                                        globals_only=True, mults=mults,
                                        block=True, preset=preset)
        out = {}
        for alias in self.written:
            tensor = self.tensors[alias]
            if tensor.is_tmp:
                continue
            sub = self._sub(tensor)
            base = self._name(alias) + (f'@{sub}' if sub else '')
            for slot in _slots(tensor):
                value = mem.get((base, slot))
                out[(alias, slot)] = (self.initial(alias, slot)
                                      if value is None else value)
        return out

    def expected(self):
        """The outputs as NumPy evaluates the list, (alias, slot) -> value."""
        arrays = {}
        for alias, t in self.tensors.items():
            arr = np.zeros(tuple(int(s) for s in t.shape) or (1,))
            if t.is_tmp:
                pass
            elif (t.addressing in (Addressing.NONE, Addressing.SCALAR)
                    and t.has_values()):
                data = np.asarray(t.data, dtype=np.float64)
                arr[tuple(slice(0, n) for n in data.shape)] = data
            else:
                for slot, cell in _slots(t).items():
                    arr[cell] = self.initial(alias, slot)
            arrays[alias] = arr
        for d in self.products:
            reference.apply({'dest': _row(d.writes()),
                             'ops': [None if o is None else _row(o)
                                     for o in d.ops],
                             'target': [list(t) for t in d.target],
                             'add': bool(d.add)}, arrays)
        return {(alias, slot): float(np.reshape(arrays[alias][cell], -1)[0])
                for alias in self.written if not self.tensors[alias].is_tmp
                for slot, cell in _slots(self.tensors[alias]).items()}


def _compare(got, want):
    """(slots, disagreeing, worst relative deviation, where)."""
    bad, worst, where = 0, 0.0, None
    for key, w in want.items():
        g = got[key]
        rel = abs(g - w) / max(abs(g), abs(w), 1e-30) if g != w else 0.0
        if rel > TOLERANCE:
            bad += 1
        if rel > worst:
            worst, where = rel, key
    return len(want), bad, worst, where


def check(kernel, builds, out=sys.stdout):
    """Every build of `kernel` against NumPy and the first; False where one
    disagrees."""
    ok = True
    width = max(len(label) for label, _ in builds)
    expected = None if kernel.unevaluable else kernel.expected()
    first = None
    for label, options in builds:
        try:
            got = kernel.run(options)
        except kernel_eval.Abort as e:
            print(f'  {label:<{width}}  oracle: {e}', file=out)
            ok = False
            continue
        parts = []
        if expected is None:
            parts.append(f'numpy: n/a ({kernel.unevaluable})')
        else:
            n, bad, worst, where = _compare(got, expected)
            parts.append(f'numpy: {bad}/{n} disagree, worst {worst:.2g}'
                         + (f' at {where}' if bad else ''))
            ok &= bad == 0
        if first is None:
            first = got
        else:
            n, bad, worst, where = _compare(got, first)
            parts.append(f'first: {bad}/{n} differ'
                         + (f', worst {worst:.2g} at {where}' if bad else ''))
            ok &= bad == 0
        print(f'  {label:<{width}}  ' + '; '.join(parts), file=out)
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('corpus', help='a corpus file, tools/bench/data/*.json')
    ap.add_argument('kernels', nargs='*', help='kernel names; all by default')
    ap.add_argument('--build', action='append', default=[],
                    help='name=value,... for one build; repeatable')
    ap.add_argument('--arch', default='sm_90')
    args = ap.parse_args()
    warnings.simplefilter('ignore')

    blob = json.loads(Path(args.corpus).read_text())
    dtype = Datatype.F64 if blob['config']['precision'] == 'd' else Datatype.F32
    builds = [(text, parse_build(text))
              for text in (args.build or ['merge_variants=0',
                                          'merge_variants=1'])]
    names = args.kernels or list(blob['descriptions'])
    ok = True
    for name in names:
        print(name)
        kernel = Kernel(blob['descriptions'][name], dtype, args.arch)
        ok &= check(kernel, builds)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
