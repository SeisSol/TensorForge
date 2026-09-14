# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Discover GPUs on the local machine.

MVP scope: NVIDIA via ``nvidia-smi``. AMD/Intel are stubbed out with clear
extension points.

A :class:`DetectedGPU` is purely a statement about present hardware; it
does not imply that a working toolchain for the arch is installed. The
target-selection logic in :mod:`toolchain` joins the two.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class DetectedGPU:
    vendor: str      # "nvidia" | "amd" | "intel"
    arch: str        # matches entries in tensorforge/common/vm/hw_descr_db.yml
    index: int       # device index as exposed by the vendor runtime
    name: str        # human-readable product name


def _run(cmd: List[str], timeout: float = 5.0) -> str | None:
    """Run ``cmd`` and return stdout, or None if the tool is missing / fails."""
    if shutil.which(cmd[0]) is None:
        return None
    try:
        out = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout


def detect_nvidia() -> List[DetectedGPU]:
    """Return NVIDIA GPUs via ``nvidia-smi``.

    Compute capability ``X.Y`` maps to arch string ``sm_XY`` — the same
    form used in ``hw_descr_db.yml`` and passed to ``nvcc -arch=``.
    """
    out = _run(
        ["nvidia-smi",
         "--query-gpu=index,compute_cap,name",
         "--format=csv,noheader,nounits"]
    )
    if out is None:
        return []

    gpus: List[DetectedGPU] = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            idx = int(parts[0])
        except ValueError:
            continue
        cc = parts[1]                      # e.g. "8.6"
        name = parts[2]
        if "." not in cc:
            continue
        major, minor = cc.split(".", 1)
        arch = f"sm_{major}{minor}"
        gpus.append(DetectedGPU("nvidia", arch, idx, name))
    return gpus


def detect_amd() -> List[DetectedGPU]:
    """Return AMD GPUs via ``rocminfo``.

    Parses the ``Name:`` line of each ``HSA Agent`` whose Device Type is
    GPU. ROCm indexing is trickier than CUDA's — we assign sequential
    indices in discovery order, which matches what ``HIP_VISIBLE_DEVICES``
    selects.
    """
    out = _run(["rocminfo"], timeout=10.0)
    if out is None:
        return []

    gpus: List[DetectedGPU] = []
    current_name: str | None = None
    current_display_name: str | None = None
    current_is_gpu = False
    idx = 0

    for raw in out.splitlines():
        line = raw.strip()
        if line.startswith("Agent "):
            current_name = None
            current_display_name = None
            current_is_gpu = False
            continue
        if line.startswith("Device Type:") and "GPU" in line:
            if current_name is not None:
                current_display_name = current_display_name or current_name
                gpus.append(DetectedGPU("amd", current_name, idx, current_display_name))
                idx += 1
        if line.startswith("Marketing Name:") and current_display_name is None:
            current_display_name = line.split(":", 1)[1].strip()
        if line.startswith("Name:") and current_name is None:
            val = line.split(":", 1)[1].strip()
            if val.startswith("gfx"):
                current_name = val
    return gpus


def detect_intel() -> List[DetectedGPU]:
    """Discover Intel GPUs via ``sycl-ls``.

    The mapping from device name to a TensorForge ``arch`` string is
    fuzzy: ``sycl-ls`` reports product names ("Intel(R) Data Center GPU
    Max 1100", "Intel(R) UHD Graphics 770"), not architecture codes.
    We only emit a target when the product name clearly matches one of
    the entries in ``hw_descr_db.yml``; ambiguous devices are returned
    as ``arch="intel-unknown"`` so the operator notices and the
    toolchain probe filters them out.

    A line reads ``[backend:gpu][backend:N] <platform>, <device> <driver>
    [<version>]``.  The version in brackets at the end is why the name is
    not "whatever follows the last ``]``" -- that is empty.  Every device is
    listed once per backend that sees it, so one backend is counted: Level
    Zero where it lists any GPU, OpenCL otherwise (a node whose DRM cards
    the user may not open still shows its GPUs through OpenCL).
    """
    # sycl-ls loads every adapter first; five seconds was not always enough.
    out = _run(["sycl-ls"], timeout=60.0)
    if out is None:
        return []

    # Map common product-name fragments to arch strings used in the DB.
    name_to_arch = [
        ("Data Center GPU Max", "pvc"),    # Ponte Vecchio
        ("Arc(TM) A",          "dg2"),     # Alchemist
        ("Iris Xe",            "tgl"),
        ("UHD Graphics 7",     "rkl"),
        ("UHD Graphics 6",     "skl"),
    ]

    by_backend: dict = {}
    for line in out.splitlines():
        m = re.match(r"\s*\[(\w+):gpu\]\[[^\]]*\]\s*(.*)$", line)
        if m is None or "Intel" not in m.group(2):
            continue
        text = re.sub(r"\s*\[[^\]]*\]\s*$", "", m.group(2))
        # "<platform>, <device>": the device is the last comma-separated part.
        name = text.rsplit(", ", 1)[-1].strip()
        by_backend.setdefault(m.group(1), []).append(name)

    names = by_backend.get("level_zero") or next(iter(by_backend.values()), [])
    gpus: List[DetectedGPU] = []
    for idx, name in enumerate(names):
        arch = next((value for needle, value in name_to_arch if needle in name),
                    "intel-unknown")
        gpus.append(DetectedGPU("intel", arch, idx, name))
    return gpus


def detect_all() -> List[DetectedGPU]:
    """Discover every GPU the host can see. Never raises."""
    gpus: List[DetectedGPU] = []
    gpus.extend(detect_nvidia())
    gpus.extend(detect_amd())
    gpus.extend(detect_intel())

    # TF_TEST_FAKE_GPUS="sm_86,gfx90a" — lets CI construct a target set on
    # hosts without real GPUs (compile-only runs). The runner will still
    # refuse to execute the resulting binary.
    fake = os.environ.get("TF_TEST_FAKE_GPUS", "").strip()
    if fake:
        for arch in (a.strip() for a in fake.split(",") if a.strip()):
            vendor = _vendor_from_arch(arch)
            if vendor is not None:
                gpus.append(DetectedGPU(vendor, arch, -1, f"fake-{arch}"))
    return gpus


def _vendor_from_arch(arch: str) -> str | None:
    if arch.startswith("sm_"):
        return "nvidia"
    if arch.startswith("gfx"):
        return "amd"
    if arch in {"pvc", "dg1", "bdw", "skl"} or arch.startswith("Gen"):
        return "intel"
    return None
