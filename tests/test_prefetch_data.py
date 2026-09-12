# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""`Options.prefetch_data`: the next element's operands, hinted where
`WrapLoads` would fetch them, with the transfers left where they are.

Held here: on under ESIMD and off elsewhere unless asked; under ESIMD the
hints of a body are gathered a line per lane, elsewhere one line each; the transfers themselves do not move; and the pointer
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


def test_the_default_follows_the_backend():
    """On where one work-item is a whole element and nothing else hides the
    next one's latency; elsewhere off until measured."""
    assert 'pf_' in _kernel()
    assert 'pf_' not in _kernel(prefetch_data=False)
    assert 'pf_' not in _kernel('cuda', 'sm_100')
    assert 'pf_' not in _kernel('hip', 'gfx942')


def _esimd_hints(src):
    """`(statements, bytes asked for)` of the data hints in an ESIMD kernel."""
    single = [4 * int(n) for n in re.findall(r'prefetchL2<(\d+)>\(&?pf_', src)]
    runs = [sum(int(b) for b in m.split(','))
            for m in re.findall(r'prefetchRunsL2<([\d, ]+)>\(&?pf_', src)]
    return len(single) + len(runs), sum(single) + sum(runs)


def test_esimd_asks_for_whole_operands_in_few_messages():
    """`local_flux`'s per-element operands are one 56 x 9 matrix and four
    9 x 9 ones: 504 and 4 x 81 floats.  A gather asks for up to 32 lines with
    a lane each, and different operands are only different addresses: the
    first 31 lines of the matrix are one message, its last line and the four
    small ones another -- two, where 64-dword blocks were 16."""
    statements, asked = _esimd_hints(_kernel(prefetch_data=True))
    assert asked == 4 * (504 + 4 * 81)
    assert statements == 2


def test_esimd_asks_for_the_pointers_in_one_message():
    """Six pointer hints side by side at the head of the body: one gather."""
    src = _kernel(enable_prefetch=True, prefetch_data=False)
    assert src.count('prefetchRunsL2<') == 1, src
    assert src.count('tensorforge::prefetchL2(') == 0


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
        assert 'prefetchL2' in src or 'prefetchRunsL2' in src
