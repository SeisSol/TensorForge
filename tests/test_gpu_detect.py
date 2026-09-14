# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""`gpu_detect.detect_intel` against what `sycl-ls` actually prints."""

from harness import gpu_detect

# oneAPI 2025.0 on a node with four Data Center GPU Max 1550, whose DRM cards
# the user may not open: Level Zero lists nothing, OpenCL lists three of them.
OPENCL_ONLY = """\
[opencl:cpu][opencl:0] Intel(R) OpenCL, Intel(R) Xeon(R) CPU Max 9468 OpenCL 3.0 (Build 0) [2024.18.12.0.05_160000]
[opencl:gpu][opencl:1] Intel(R) OpenCL Graphics, Intel(R) Data Center GPU Max 1550 OpenCL 3.0 NEO  [23.22.26516.25]
[opencl:gpu][opencl:2] Intel(R) OpenCL Graphics, Intel(R) Data Center GPU Max 1550 OpenCL 3.0 NEO  [23.22.26516.25]
[opencl:gpu][opencl:3] Intel(R) OpenCL Graphics, Intel(R) Data Center GPU Max 1550 OpenCL 3.0 NEO  [23.22.26516.25]
"""

# The same device seen by both backends.
BOTH = """\
[level_zero:gpu][level_zero:0] Intel(R) oneAPI Unified Runtime over Level-Zero, Intel(R) Data Center GPU Max 1550 12.60.7 [1.3.26516]
[opencl:gpu][opencl:0] Intel(R) OpenCL Graphics, Intel(R) Data Center GPU Max 1550 OpenCL 3.0 NEO  [23.22.26516.25]
"""


def _detect(monkeypatch, text):
    monkeypatch.setattr(gpu_detect, "_run", lambda cmd, timeout=5.0: text)
    return gpu_detect.detect_intel()


def test_a_trailing_version_does_not_hide_the_device(monkeypatch):
    gpus = _detect(monkeypatch, OPENCL_ONLY)
    assert [g.arch for g in gpus] == ["pvc"] * 3
    assert [g.index for g in gpus] == [0, 1, 2]
    assert all(g.name.startswith("Intel(R) Data Center GPU Max 1550") for g in gpus)


def test_a_device_seen_by_two_backends_is_one_gpu(monkeypatch):
    gpus = _detect(monkeypatch, BOTH)
    assert len(gpus) == 1 and gpus[0].arch == "pvc"


def test_no_gpu_lines_is_no_gpu(monkeypatch):
    assert _detect(monkeypatch, OPENCL_ONLY.splitlines()[0] + "\n") == []
