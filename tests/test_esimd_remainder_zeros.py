# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""The rows of a destination's storage past its eqspp window are written
zero under the explicit vector too, all of them.

`StoreRegToGlb._fill_lead_remainder` writes a constant over the rows an
assignment's promise covers past the accumulator's.  Every SPMD lane stores
it at its own row; one ESIMD work-item stores for the whole block, as wide
as the value -- and a constant is one element wide, so of `dQ(1)`'s rows
10..15 in SeisSol's order-4 time derivative only row 10 was written, and
the rest kept what the buffer held (`StoreRegToGlb._spread_zero`).  Run on
the CPU (tfemu) against NumPy, the kernel now leaves the storage as the
SPMD lowering does.
"""

import re

import pytest

from test_esimd_select_bounds import DATA, _esimd_kernel


def test_the_remainder_of_dq1_is_one_vector_of_zeros():
    if not (DATA / 'elastic-4-s-s1.json').exists():
        pytest.skip('no capture elastic-4-s-s1')
    src = _esimd_kernel('elastic-4-s-s1', 'gpu_derivative')
    # dQ(1) is m5: rows 0..9 computed, 16 stored per column.
    store = re.search(r'(v\d+_z)\.copy_to\(glb_m5 \+ \(\(10_i32 \+ \(v\d+_z1 \* 16\)\)\)\);', src)
    assert store, 'the zero fill of dQ(1) past row 10'
    decl = re.search(rf'simd<float, (\d+)> {store.group(1)}\(', src)
    assert decl and int(decl.group(1)) == 6
    # No scalar store of a single row is left for it.
    assert not re.search(r'glb_m5\[\(10_i32 \+ \(v\d+_z1 \* 16\)\)\] = 0\.0f;', src)
