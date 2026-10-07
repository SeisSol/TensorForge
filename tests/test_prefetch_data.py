# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""`Options.prefetch_data`: the next element's operands, hinted where
`enable_wrap_loads` would fetch them, with the transfers left where they are.

Held here: off unless asked, ESIMD included; under ESIMD the hints of a body
are gathered a line per lane, elsewhere one line each; a compressed operand
is asked for as it is stored; the transfers themselves do not move; a group
of rows hints each row's successor; and the hints stand beside a wrapped
transfer, under a flag of their own.
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


def _kernel(backend='esimd', arch='pvc', case='local_flux', **opts):
    # unmerged: the hints counted here are one per operator of the unmerged
    # kernel, and a merged run reads its operators through a table
    opts.setdefault('merge_variants', False)
    path = CASES / f'{case}.py'
    spec = importlib.util.spec_from_file_location(
        f'tf_pfd__{path.stem}', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter('ignore')
        gen = Generator(mod.descr_list(),
                        Context(arch=arch, backend=backend, fp_type=mod.DTYPE,
                                options=Options(**opts)))
        gen.generate()
    return gen.get_kernel()


def test_the_default_is_off_everywhere():
    """ESIMD included: on pvc the hints cost more than the latency they hide
    (see the option's doc).  Asking still turns them on."""
    assert 'pf_' not in _kernel()
    assert 'pf_' in _kernel(prefetch_data=True)
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
    small ones another -- two, where 64-dword blocks would take 16."""
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
    """The pointer hints sit outside the flag guard, at the head of the body,
    and the wrap moves a transfer out of it, to the tail: both are for the
    successor, which a masked element has as well, and the three options
    generate together."""
    for backend, arch in (('esimd', 'pvc'), ('cuda', 'sm_100')):
        src = _kernel(backend, arch, enable_prefetch=True,
                      enable_wrap_loads=True, prefetch_data=True)
        assert 'prefetchL2' in src or 'prefetchRunsL2' in src


def test_a_group_of_rows_hints_each_rows_successor():
    """A traversal over groups of rows binds each row's element in its body
    (`mixed/ml_then_ew` on pvc under SYCL is one).  The hints are for that
    element's successor: the pointer is bound at `batchId1`, and what binds
    the row's own element is not computed again -- for the successor, it
    would offset the group's index by the row a second time."""
    src = _kernel('oneapi', 'pvc', case='mixed/ml_then_ew', prefetch_data=True)
    bindings = re.findall(r'pf_glb_\w+ = &\w+\[(\w+) \*', src)
    assert bindings and all(i.endswith('batchId1') for i in bindings), src
    assert 'pf_batchIdActive' not in src


def test_the_hints_flag_is_not_the_wraps():
    """A pointer of the element's own is followed under the successor's
    flag, by the hints and by the wrapped transfer alike, at one tail: each
    declares its own (`allowed_hint`, `allowed_next`), once."""
    src = _kernel('cuda', 'sm_86', case='addressing_ptr_based',
                  enable_prefetch=True, prefetch_data=True,
                  enable_wrap_loads=True)
    assert src.count('const bool allowed_hint ') == 1, src
    assert src.count('const bool allowed_next ') == 1, src
    assert 'if (allowed_hint)' in src


def test_a_compressed_operand_is_asked_for_as_it_is_stored():
    """`sparsity_band` stores B compressed, 46 floats an element where its
    view is 16 x 16: two lines of 128 bytes, and nothing of the elements
    behind it."""
    src = _kernel('cuda', 'sm_86', case='sparsity_band', prefetch_data=True)
    starts = re.findall(r'prefetchL2\(&pf_glb_m2\[(\d+)\]\)', src)
    assert starts == ['0', '32'], src
