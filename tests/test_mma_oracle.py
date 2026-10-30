# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""NVIDIA's matrix path on the host oracle.

`kernel_eval` runs `mma.sync` from PTX's fragment tables
(`Interp._mma`), so a kernel built with `tensor_cores` computes on the
host, where every number can be compared with NumPy -- without a GPU, and
for any `sm` the emitter selects instructions for.  What it checks is the
part a wrong answer comes from: which lane hands in which element of `A`
and `B`, and which element of `D` each lane gets back.

SeisSol's volume kernel at order 6 is the case: three products of a
prepared operator into one output, a window that starts at row one, and a
ninth column carried on the fragments (`TAIL_MAX`).  sm_86 takes `m8n8k4`
for FP64 and sm_90 `m16n8k16`, two accumulator layouts.  The comparison is
normwise, per output: 3xTF32 keeps about 22 bits, and an element the sum
cancels in is compared against what it lost.
"""

from __future__ import annotations

import importlib.util
import json
import warnings
from pathlib import Path

import pytest

from tensorforge.common.basic_types import Datatype

ROOT = Path(__file__).resolve().parent.parent

#: Normwise, by precision: 3xTF32 and FP64.
TOLERANCE = {'s': 1e-6, 'd': 1e-12}


def _host():
    path = ROOT / 'tools' / 'host' / 'corpus_check.py'
    spec = importlib.util.spec_from_file_location('mma_corpus_check', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


HOST = _host()


def _kernel(description, precision, arch):
    dtype = Datatype.F64 if precision == 'd' else Datatype.F32
    return HOST.Kernel(description, dtype, arch)


def _volume(precision):
    import seissol_suite as fx
    system = 'elastic-linearck'
    return fx.description(system, f'{system}-o6-{precision}', 'gpu_volume')


def _normwise(got, want):
    """The largest deviation per output, against its largest magnitude --
    infinite where a cell is not a number, which `max` would pass over."""
    out = {}
    for alias in {alias for alias, _ in want}:
        keys = [key for key in want if key[0] == alias]
        scale = max(abs(want[key]) for key in keys) or 1.0
        if any(got[key] != got[key] for key in keys):
            out[alias] = float('inf')
            continue
        out[alias] = max(abs(got[key] - want[key]) for key in keys) / scale
    return out


def _source(description, precision, arch, **options):
    """The kernel a build of `description` emits."""
    import contextlib
    import io

    from tensorforge.common.context import Context, Options
    from tensorforge.generators.generator import Generator
    kernel = _kernel(description, precision, arch)
    gen = Generator(kernel.descrs(), Context(
        arch=arch, backend='cuda', fp_type=kernel._dtype,
        options=Options(**options)))
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter('ignore')
        gen.generate()
    return gen.get_kernel()


@pytest.mark.parametrize('arch', ['sm_86', 'sm_90'])
@pytest.mark.parametrize('precision', ['s', 'd'])
def test_the_matrix_path_computes_on_the_oracle(arch, precision):
    description = _volume(precision)
    assert 'mma.sync' in _source(description, precision, arch,
                                 tensor_cores=True)
    kernel = _kernel(description, precision, arch)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        got, _ = kernel.run({'tensor_cores': True})
    for alias, error in _normwise(got, kernel.expected()).items():
        assert error <= TOLERANCE[precision], (alias, error)


@pytest.mark.parametrize('window', [(2, 49), (16, 49)])
def test_a_window_that_starts_inside_a_slot_is_walked_to_its_end(window):
    """A window from row `lo` to `hi`: the path walks the rows from the
    start of the window's first slot, and so has to walk to `hi`, not to
    `hi - lo`.  At `[2, 49)` the walk ended with the tile before row 48's,
    and row 48 -- the window's last, alone in its tile -- kept the zero its
    accumulator was declared with (`MatmulOperands.lead_first`)."""
    description = json.loads(json.dumps(_volume('d')))
    lo, hi = window
    for op in description['operations']:
        if op['result']['name'] == 'Q':
            op['result']['bbox'][0][0], op['result']['bbox'][1][0] = lo, hi
            op['args'][0]['bbox'][0][0], op['args'][0]['bbox'][1][0] = lo, hi
            break
    kernel = _kernel(description, 'd', 'sm_90')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        got, _ = kernel.run({'tensor_cores': True})
    for alias, error in _normwise(got, kernel.expected()).items():
        assert error <= TOLERANCE['d'], (alias, error)


@pytest.mark.parametrize('precision', ['s', 'd'])
def test_the_matrix_path_reads_no_shared_memory_it_did_not_write(precision,
                                                                 monkeypatch):
    """Shared memory nothing wrote reads as not a number: the deep atoms
    of sm_90 step past the volume's 35 rows of `B`, and read where it lies,
    the column carried on the fragments read the rows after the operand's
    last -- a gap of the arena, or the next operator's transfer still in
    flight.  `A`'s padding made each such product zero, but for a value
    that is not a number."""
    from tensorforge.reference import kernel_eval as ke
    real = ke.Slot.read

    def read(self, base, idx):
        if base == 'shr' and (base, int(idx)) not in self.data:
            return float('nan')
        return real(self, base, idx)
    monkeypatch.setattr(ke.Slot, 'read', read)
    kernel = _kernel(_volume(precision), precision, 'sm_90')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        got, _ = kernel.run({'tensor_cores': True})
    for alias, error in _normwise(got, kernel.expected()).items():
        assert error <= TOLERANCE[precision], (alias, error)


def test_a_dropped_correction_term_is_seen():
    """The oracle is not blind to the split: with one of the three TF32
    products left out, the volume is wrong by what a single product of
    halves loses, far past the tolerance."""
    from tensorforge.backend.instructions.compute.primitives import nvidia
    description = _volume('s')
    kernel = _kernel(description, 's', 'sm_90')
    real = nvidia.MMAInstr.products

    def two(self, writer, A, B, a_split=None, b_split=None):
        return real(self, writer, A, B, a_split, b_split)[:2]
    nvidia.MMAInstr.products = two
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            got, _ = kernel.run({'tensor_cores': True})
    finally:
        nvidia.MMAInstr.products = real
    errors = _normwise(got, kernel.expected())
    assert max(errors.values()) > 100 * TOLERANCE['s'], errors
