# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""`Options.prefetch_data`: the next element's operands, hinted where
`WrapLoads` would fetch them, with the transfers left where they are.

Held here: off unless asked; under ESIMD a hint is a block of up to 64 dwords,
elsewhere one line; the transfers themselves do not move; and the pointer
hints of `enable_prefetch` sit outside the flag guard, which is what lets
them stand next to a wrapped transfer at all.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import re
import warnings
from pathlib import Path

from tensorforge.common.context import Context, Options
from tensorforge.common.options import registry
from tensorforge.generators.generator import Generator

CASES = Path(__file__).resolve().parent / "cases"


def _kernel(backend='esimd', arch='pvc', **opts):
    path = CASES / 'local_flux.py'
    spec = importlib.util.spec_from_file_location('tf_pfd__local_flux', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter('ignore')
        gen = Generator(mod.descr_list(),
                        Context(arch=arch, backend=backend, fp_type=mod.DTYPE,
                                options=Options(**opts)))
        gen.generate()
    return gen.get_kernel()


def test_off_by_default():
    assert registry()['prefetch_data'].default is False
    assert 'pf_' not in _kernel()


def test_esimd_asks_for_whole_operands_in_few_messages():
    """`local_flux`'s per-element operands are one 56 x 9 matrix and four
    9 x 9 ones: 504 and 4 x 81 floats.  One statement asks for up to 31 lines
    -- a gather with a lane per line -- so the 2016 bytes of the first are two
    hints and each of the others one: six, where 64-dword blocks were 16."""
    src = _kernel(prefetch_data=True)
    runs = [int(n) for n in re.findall(r'prefetchL2<(\d+)>\(&?pf_', src)]
    assert sum(runs) == 504 + 4 * 81
    assert len(runs) == 6


def test_cuda_asks_one_line_per_hint():
    """A `prefetch.global.L2` names the 128-byte line holding its address:
    16 lines for 504 floats and 3 for each of 81."""
    src = _kernel('cuda', 'sm_100', prefetch_data=True)
    assert len(re.findall(r'prefetchL2\(&pf_', src)) == 16 + 4 * 3


def test_the_transfers_stay_where_they_are():
    off = _kernel('cuda', 'sm_100')
    on = _kernel('cuda', 'sm_100', prefetch_data=True)
    for name in ('glb_m1', 'glb_m3', 'glb_m9'):
        # the name standing alone: the hints read through `pf_glb_m1`
        read = re.compile(rf'(?<!\w){name}\[')
        assert len(read.findall(off)) == len(read.findall(on)), name


def test_pointer_hints_stand_beside_a_wrapped_transfer():
    """The pointer hints used to sit inside the flag guard, ahead of the
    unguarded prefix `WrapLoads` makes -- and the guard, one block, could not
    be formed: neither option combination generated."""
    for backend, arch in (('esimd', 'pvc'), ('cuda', 'sm_100')):
        src = _kernel(backend, arch, enable_prefetch=True,
                      enable_wrap_loads=True, prefetch_data=True)
        assert 'prefetchL2' in src
