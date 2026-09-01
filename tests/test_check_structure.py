# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""The register-bounds check sees the indices it claims to see.

An index that leaves its register array is invisible to the numeric oracle:
the interpreter keeps registers in a dict, so it happily reads and writes past
the end and still produces an answer. On a GPU the same access lands on a
neighbouring register or spill slot. That makes this a check nothing else in
the suite can make, which is reason enough to pin what it covers.

The case that motivated the pinning is a base offset of -1 on a carry-in: the
array is filled at 0..5 and read at -1..4, so the accumulation picks up an
uninitialised slot and drops the last element. Both halves of it are easy to
miss -- the index is composed rather than a bare name, and it is negative
rather than too large.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHECKER = ROOT / "tools" / "host" / "check_structure.py"

PROLOGUE = """__global__ void
 kernel_kernel_{name}(float** m0, size_t numElements0) {{
  {{
    // generated with TensorForge. Version: 0.0.1
    float r0[6]{{}};
    // r0 = load{{g>r}}(glb_m0);
    for (int32_t v10_i1 = 0; v10_i1 < 1; ++v10_i1) {{
      for (int32_t v11_i2 = 0; v11_i2 < 6; ++v11_i2) {{
        r0[(v10_i1 + v11_i2)] = glb_m0[v11_i2];
      }}
    }}
    float r1[6]{{}};
    for (int32_t v20_n1 = 0; v20_n1 < 1; ++v20_n1) {{
      int32_t v21_a = {base} + v20_n1;
      for (int32_t v22_n2 = 0; v22_n2 < 6; ++v22_n2) {{
        float v23_data = r0[(v21_a + v22_n2)];
        r1[(v20_n1 + v22_n2)] = v23_data;
      }}
    }}
  }}
}}
void launcher_kernel_{name}(float** m0, size_t numElements0, void* streamPtr) {{
}}
"""


def run(tmp_path, kernels):
    dump = tmp_path / "gpulike_subroutine.cpp"
    dump.write_text("".join(PROLOGUE.format(name=name, base=base)
                            for name, base in kernels))
    finished = subprocess.run([sys.executable, str(CHECKER), str(dump)],
                              capture_output=True, text=True, timeout=300)
    assert finished.returncode == 0, finished.stderr
    return finished.stdout


def test_a_matching_base_offset_is_not_flagged(tmp_path):
    output = run(tmp_path, [("aaaaaaaaaa", "0")])
    assert "REG OOB" not in output
    assert "flagged: 0 of 1" in output


def test_a_negative_base_offset_is_flagged(tmp_path):
    output = run(tmp_path, [("bbbbbbbbbb", "-1")])
    assert "REG OOB" in output
    assert "('r0', -1, 5, 6)" in output


def test_an_index_past_the_end_is_flagged(tmp_path):
    output = run(tmp_path, [("cccccccccc", "1")])
    assert "REG OOB" in output
    assert "('r0', 0, 6, 6)" in output


def test_only_the_offending_kernel_is_flagged(tmp_path):
    output = run(tmp_path, [("aaaaaaaaaa", "0"), ("bbbbbbbbbb", "-1")])
    assert "flagged: 1 of 2" in output
    assert "kernel_bbbbbbbbbb" in output
    assert "kernel_aaaaaaaaaa" not in output
