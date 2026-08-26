# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Hand a generated kernel to a real C++ front end.

The whole mechanism: take the ``// === kernel ===`` section out of a snapshot,
put it on top of ``tests/shim/tensorforge_host.h``, run ``g++ -fsyntax-only``.
Shared between ``tests/test_syntax.py`` and ``tools/syntax_check.py`` so that
the pytest and the command-line runner cannot answer differently.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

HERE = Path(__file__).resolve().parent
TESTS = HERE.parent
SHIM = TESTS / "shim" / "tensorforge_host.h"
SNAPSHOT_DIR = TESTS / "snapshots"

#: A snapshot of a case that failed to generate records the exception instead
#: of source.  `test_snapshots.py` writes those with this marker.
_FAILURE_MARKER = "!!"

_KERNEL = re.compile(r"^// === kernel ===\n(.*?)(?=^// === |\Z)", re.M | re.S)


def compiler() -> Optional[str]:
    return (os.environ.get("TF_HOST_CXX")
            or shutil.which("g++") or shutil.which("clang++"))


def kernel_section(text: str) -> Optional[str]:
    """The generated kernel, or None when there is nothing to check."""
    if text.lstrip().startswith(_FAILURE_MARKER):
        return None                    # a recorded generation failure
    m = _KERNEL.search(text)
    return m.group(1) if m else None


def translation_unit(kernel: str) -> str:
    return f'#include "{SHIM}"\n\n{kernel}'


@dataclass(frozen=True)
class Result:
    path: Path
    ok: Optional[bool]                 # None: nothing to check
    stderr: str = ""
    reason: str = ""

    def errors(self, limit: int = 12) -> List[str]:
        out = [ln for ln in self.stderr.splitlines() if ": error:" in ln]
        return out[:limit]


def check_source(kernel: str, cxx: Optional[str] = None,
                 path: Optional[Path] = None) -> Result:
    cxx = cxx or compiler()
    with tempfile.NamedTemporaryFile("w", suffix=".cpp", delete=False) as f:
        f.write(translation_unit(kernel))
        tmp = f.name
    try:
        r = subprocess.run(
            [cxx, "-fsyntax-only", "-std=c++17", "-w", tmp],
            capture_output=True, text=True)
    finally:
        os.unlink(tmp)
    # The temporary name is noise in a failure report and makes it unstable
    # from run to run; put the snapshot's name there instead.
    stderr = r.stderr.replace(tmp, str(path) if path else "<generated>")
    return Result(path or Path("<generated>"), r.returncode == 0, stderr)


def check_snapshot(path: Path, cxx: Optional[str] = None) -> Result:
    kernel = kernel_section(path.read_text())
    if kernel is None:
        return Result(path, None, reason="no kernel section (generation "
                                         "failure or unrecognised layout)")
    return check_source(kernel, cxx, path)


def snapshots(pattern: str = "*.cpp") -> List[Path]:
    return sorted(SNAPSHOT_DIR.glob(pattern))
