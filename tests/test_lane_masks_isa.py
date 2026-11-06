# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Lane patterns known at compile time, as hipcc lowers them (`laneIn`).

The builtin 4x4 transpose, `laneMerge` and the row-of-sixteen broadcast of
gfx10 on select by a fixed pattern of lanes.  Asked of the lane id, that is
two `v_mbcnt` (one in a wave of 32) and a comparison per pattern; asked as
`__builtin_amdgcn_inverse_ballot`, it is the pattern itself in scalar
registers, which the select takes as its mask.  On gfx1150 the three give
the lanes their names claim (checked on the device: every value of a block
of 256).  Skipped where hipcc is absent or does not know the target.
"""

from __future__ import annotations

import subprocess

import pytest

from harness import syntax

SNIPPET = r'''
#include "tensorforge_device/hip.h"
__global__ void k(float *o) {
  const int l = threadIdx.x;
  float w1, w2, w3, w4;
  tensorforge::transpose4x4b32(w1, w2, w3, w4, o[l], o[l + 256], o[l + 512],
                               o[l + 768]);
  float m = tensorforge::laneMerge<0x00ff00ff00ff00ffULL>(w1, w2);
#if defined(__GFX10__) || defined(__GFX11__) || defined(__GFX12__)
  m += tensorforge::broadcast<32, 16, 1>(w3);
#endif
  o[l] = m + w3 * w4;
}
'''


@pytest.mark.parametrize('arch', ['gfx90a', 'gfx942', 'gfx1150', 'gfx1201',
                                  'gfx1250'])
def test_a_fixed_lane_pattern_reads_no_lane_id(arch, tmp_path):
    hipcc = syntax.device_compiler('hip')
    if hipcc is None:
        pytest.skip('hipcc not found')
    from tensorforge import toolchain
    src = tmp_path / 'masks.hip'
    src.write_text(SNIPPET)
    run = subprocess.run([hipcc, '-O3', '-std=c++17', f'--offload-arch={arch}',
                          '-I', toolchain.include_dir(), '--cuda-device-only',
                          '-S', '-o', '-', str(src)],
                         capture_output=True, text=True, timeout=600)
    if run.returncode and 'offload-arch' in run.stderr:
        pytest.skip(f'hipcc does not know {arch}')
    assert run.returncode == 0, run.stderr[-2000:]
    body = [line.strip() for line in run.stdout.splitlines()
            if line.startswith('\t') and not line.strip().startswith(';')]
    assert not any(ins.startswith('v_mbcnt') for ins in body)
    assert not any(ins.startswith('v_cmp') for ins in body)
    # gfx1250 pairs two selects into one `v_dual_cndmask ... :: v_dual_cndmask`.
    assert sum(ins.count('cndmask') for ins in body) >= 9
