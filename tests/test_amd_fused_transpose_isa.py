# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""`transpose4x4b32Fused` as hipcc assembles it, per target.

The exchange is inline assembly, so the hazard recognizer does not see into
it and the assembler is the first to read it: whatever a target takes is
decided here and nowhere else.  Asked of the instructions the device
compiler writes rather than of the source: eight fused selections where the
target has a fused form -- the VOP2 one with the mask in VCC on gfx9, behind
the one wait gfx9's DPP hazard asks for; the VOP3 one with the mask in an
SGPR of its own on gfx11 and gfx12, behind the wait gfx12 asks of a VALU
reading what the SALU just wrote -- and the builtin form elsewhere.

What the eight selections compute is checked on the device, not here: on
gfx1150 (Radeon 890M, ROCm 7.2) the gfx11 form gave the exchange the name
claims for every lane and agreed with the builtin form on 6.6e8 lane values
written by VALU right in front of it.  Skipped where hipcc is absent or does
not know the target.
"""

from __future__ import annotations

import re
import subprocess

import pytest

from harness import syntax

SNIPPET = r'''
#include "tensorforge_device/hip.h"
__global__ void k(float *o) {
  const float v1 = o[threadIdx.x] * 2.0f, v2 = o[threadIdx.x + 256] * 3.0f;
  const float v3 = o[threadIdx.x + 512] * 5.0f, v4 = o[threadIdx.x + 768] * 7.0f;
  float w1, w2, w3, w4;
  tensorforge::transpose4x4b32Fused(w1, w2, w3, w4, v1, v2, v3, v4);
  o[threadIdx.x] = w1 + 3.0f * w2 + 5.0f * w3 + 7.0f * w4;
}
'''


def _assemble(arch, tmp_path):
    hipcc = syntax.device_compiler('hip')
    if hipcc is None:
        pytest.skip('hipcc not found')
    from tensorforge import toolchain
    src = tmp_path / 'fused.hip'
    src.write_text(SNIPPET)
    run = subprocess.run([hipcc, '-O3', '-std=c++17', f'--offload-arch={arch}',
                          '-I', toolchain.include_dir(), '--cuda-device-only',
                          '-S', '-o', '-', str(src)],
                         capture_output=True, text=True, timeout=600)
    if run.returncode and 'offload-arch' in run.stderr:
        pytest.skip(f'hipcc does not know {arch}')
    assert run.returncode == 0, run.stderr[-2000:]
    return [line.strip() for line in run.stdout.splitlines()
            if line.startswith('\t') and not line.strip().startswith(';')]


@pytest.mark.parametrize('arch', ['gfx90a', 'gfx942'])
def test_on_gfx9_the_selections_take_vcc_behind_one_wait(arch, tmp_path):
    body = _assemble(arch, tmp_path)
    fused = [i for i, ins in enumerate(body)
             if ins.startswith('v_cndmask_b32_dpp')]
    assert len(fused) == 8
    assert not any(ins.startswith('v_mov_b32_dpp') for ins in body)
    assert any(ins.startswith('s_nop') for ins in body[:fused[0]])


@pytest.mark.parametrize('arch', ['gfx1100', 'gfx1150', 'gfx1201',
                                  'gfx1250'])
def test_on_gfx11_and_gfx12_each_selection_names_its_mask(arch, tmp_path):
    body = _assemble(arch, tmp_path)
    fused = [i for i, ins in enumerate(body)
             if ins.startswith('v_cndmask_b32_e64_dpp')]
    assert len(fused) == 8
    assert not any(ins.startswith('v_mov_b32_dpp') for ins in body)
    # The masks are SGPRs the compiler filled, none of them VCC, and the
    # block writes none of them.
    masks = {re.split(r',\s*', ins.split(None, 1)[1])[3] for ins in
             (body[i] for i in fused)}
    assert len(masks) == 4 and not any(m.startswith('vcc') for m in masks)
    block = body[fused[0]:fused[-1] + 1]
    assert all(ins.startswith('v_cndmask_b32_e64_dpp') for ins in block)
    waited = body[fused[0] - 1].startswith('s_wait_alu')
    assert waited == arch.startswith('gfx12')


@pytest.mark.parametrize('arch', ['gfx1030'])
def test_without_vop3_dpp_the_builtin_form_stands(arch, tmp_path):
    body = _assemble(arch, tmp_path)
    assert not any('cndmask_b32_dpp' in ins or 'cndmask_b32_e64_dpp' in ins
                   for ins in body)
    assert sum(ins.startswith('v_mov_b32_dpp') for ins in body) == 8
