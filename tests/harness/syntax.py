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
SYCL_SHIM = TESTS / "shim" / "tensorforge_sycl.h"
SNAPSHOT_DIR = TESTS / "snapshots"

#: Which shim answers for which backend.  A snapshot is named
#: ``<case>.<backend>.cpp``, so the backend is recoverable from the path and
#: no caller has to pass it -- the alternative was a default argument, and a
#: default here means a CUDA shim silently checking a SYCL kernel and
#: reporting that `sycl::queue` does not exist as if the *generator* were at
#: fault.
_SHIMS = {
    "cuda": SHIM,
    "hip": SHIM,
    "oneapi": SYCL_SHIM,
    "acpp": SYCL_SHIM,
    "esimd": SYCL_SHIM,
}


def shim_for(backend: Optional[str]) -> Path:
    return _SHIMS.get(backend or "", SHIM)


def backend_of(path: Path) -> Optional[str]:
    """The backend a snapshot was generated for, from ``<case>.<backend>.cpp``."""
    parts = path.name.split(".")
    return parts[-2] if len(parts) >= 3 else None

#: A snapshot of a case that failed to generate records the exception instead
#: of source.  `test_snapshots.py` writes those with this marker.
_FAILURE_MARKER = "!!"

_KERNEL = re.compile(r"^// === kernel ===\n(.*?)(?=^// === |\Z)", re.M | re.S)


#: Generated source known not to compile, by snapshot name.
#:
#: Empty, and the mechanism stays because emptying it was the point.
#:
#: It held six ESIMD snapshots that reached a predicated store: a guard on the
#: lead axis that narrowing could not remove, because the vector had to
#: *start* somewhere other than element zero and `LeadIndex` had no base
#: offset to say it with.  It has one since the `VarOffset` merge, and
#: `DataView.split_lead_shift` puts the leftover lanes into a register address
#: -- so a head block, a ragged tail and a later slot are all just a vector
#: with a base now, and no mask survives the corpus.
#:
#: A list and not a pattern.  The first version matched on "contains a mask
#: and an `if`", which stopped describing the set as soon as narrowing changed
#: which cases failed and why -- and a heuristic that quietly misclassifies is
#: worse than none, because the entry it wrongly excuses looks reviewed.
NOT_YET_ESIMD: dict = {}


def known_bad(path) -> str:
    """The recorded reason this snapshot does not compile, or ``''``.

    Here rather than in `test_syntax.py` because `tools/syntax_check.py` needs
    the same answer.  It did not have it, so the command-line runner reported
    three permanent failures for cases the suite already tracks as expected --
    and three standing reds are how a check stops being read.
    """
    return NOT_YET_ESIMD.get(path.name, "")


def compiler() -> Optional[str]:
    return (os.environ.get("TF_HOST_CXX")
            or shutil.which("g++") or shutil.which("clang++"))


def kernel_section(text: str) -> Optional[str]:
    """The generated kernel, or None when there is nothing to check."""
    if text.lstrip().startswith(_FAILURE_MARKER):
        return None                    # a recorded generation failure
    m = _KERNEL.search(text)
    return m.group(1) if m else None


def translation_unit(kernel: str, shim: Optional[Path] = None) -> str:
    return f'#include "{shim or SHIM}"\n\n{kernel}'


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
                 path: Optional[Path] = None,
                 shim: Optional[Path] = None) -> Result:
    cxx = cxx or compiler()
    with tempfile.NamedTemporaryFile("w", suffix=".cpp", delete=False) as f:
        f.write(translation_unit(kernel, shim))
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
    return check_source(kernel, cxx, path, shim_for(backend_of(path)))


def snapshots(pattern: str = "*.cpp") -> List[Path]:
    return sorted(SNAPSHOT_DIR.glob(pattern))


# ----------------------------------------------------------------------
# The device front end
# ----------------------------------------------------------------------
#
# `g++ -fsyntax-only` answers "is this well-formed C++", which is the class of
# defect that escaped everything else and is worth the four seconds.  It does
# not answer "will the *device* front end take it", and the two differ: a GNU
# `vector_size` typedef is well-formed to g++ and rejected by nvcc in device
# code with "is a vector, which is not supported in device code".
#
# That difference is not hypothetical.  `CudaLexic.get_fptype` renders a packed
# value as `tensorforge::VectorT<float, 4>`, deliberately and with its reasons
# written down; the NVIDIA matrix path is the only live emitter that makes a
# *value* of that type, and it produces 101 nvcc errors on a kernel that g++
# passes without a word.  The path is parked behind `nvidia.ENABLED`, so the
# corpus never reaches it -- which is exactly the arrangement where a defect
# waits.  `cuda.h` predicted this one in as many words: "If nvcc ever ...
# declines `vector_size` in device code at all -- these turn a silent
# quarter-width copy into a build error."
#
# Neither invocation below generates an object: `-ptx` stops nvcc after the
# device compile, `--cuda-device-only -fsyntax-only` stops clang before code
# generation.  Half a second per case, against four for the whole host corpus.

@dataclass(frozen=True)
class _DeviceFrontEnd:
    #: Overriding variable, the same names `toolchain.py` honours.
    env: str
    default: str
    #: The architecture to compile *for*.  A front end targets any
    #: architecture it knows without the hardware present, so this is a
    #: property of the check and not of the machine running it -- the same
    #: pair `test_snapshots.py` freezes, so a failure names a target the
    #: corpus already carries.
    arch: str
    flags: tuple


_DEVICE_FRONT_ENDS = {
    "cuda": _DeviceFrontEnd(
        env="NVCC", default="nvcc", arch="sm_86",
        flags=("-x", "cu", "--expt-relaxed-constexpr", "-ptx")),
    "hip": _DeviceFrontEnd(
        env="HIPCC", default="hipcc", arch="gfx90a",
        flags=("-x", "hip", "--cuda-device-only", "-fsyntax-only")),
}

INCLUDE = TESTS.parent / "src" / "tensorforge" / "include"


#: Generated source the device front end is known to refuse, by
#: ``(case, backend)``, with the reason.
#:
#: The same arrangement as `NOT_YET_ESIMD` and for the same reason: a check
#: that is permanently red is a check nobody reads.  An entry here is a claim
#: that the refusal is understood, not that it is acceptable.
#:
#: `gemm_square_16_f128` is refused twice over, and only one of the two is
#: ours.  Below Blackwell nvcc rejects `__float128` in device code outright
#: -- an architecture fact, and `tests/README.md` already says the case needs
#: a compiler that supports the type.  On `sm_120`, where the type is taken,
#: one error is left and it is a generator defect: the load path emits
#: `__ldcg`, which has no `__float128` overload at any architecture.  Removing
#: this entry is what a fix for that has to do.
DEVICE_KNOWN_BAD = {
    ("gemm_square_16_f128", "cuda"):
        "nvcc has no `__ldcg` overload for `__float128`, and below sm_120 it "
        "declines the type in device code at all",
}


def device_known_bad(case_name: str, backend: str) -> str:
    """The recorded reason this pair does not compile, or ``''``."""
    return DEVICE_KNOWN_BAD.get((case_name, backend), "")


def device_front_end(backend: str) -> Optional[_DeviceFrontEnd]:
    """The front end for this backend, or None where there is not one.

    SYCL is absent on purpose rather than by omission: `acpp` and `icpx` are
    whole-program compilers with no device-only mode that stops this early,
    and a check that quietly compiled the host half too would be a slower
    version of the one above wearing this one's name.
    """
    return _DEVICE_FRONT_ENDS.get(backend)


def device_compiler(backend: str) -> Optional[str]:
    fe = device_front_end(backend)
    if fe is None:
        return None
    return os.environ.get(fe.env) or shutil.which(fe.default)


def check_device_source(kernel: str, headers, backend: str,
                        arch: Optional[str] = None,
                        path: Optional[Path] = None,
                        cc: Optional[str] = None) -> Result:
    """Hand one generated kernel to the device front end.

    `headers` is the generator's own list --- `vm.get_headers()` plus
    `gen.get_helper_headers()` --- and not a fixed preamble, because which
    helper headers a kernel needs is a property of what it emitted.  A
    barrier case pulls in cooperative groups and a plain GEMM does not.
    """
    fe = device_front_end(backend)
    if fe is None:
        return Result(path or Path("<generated>"), None,
                      reason=f"no device front end for backend {backend!r}")
    cc = cc or device_compiler(backend)
    if cc is None:
        return Result(path or Path("<generated>"), None,
                      reason=f"{fe.default} not found; set ${fe.env}")

    suffix = ".cu" if backend == "cuda" else ".hip.cpp"
    preamble = "\n".join(f'#include "{h}"' for h in headers)
    with tempfile.NamedTemporaryFile("w", suffix=suffix, delete=False) as f:
        f.write(preamble + "\n" + kernel)
        tmp = f.name
    arch_flag = (f"-arch={arch or fe.arch}" if backend == "cuda"
                 else f"--offload-arch={arch or fe.arch}")
    try:
        r = subprocess.run(
            [cc, "-std=c++17", "-w", arch_flag, *fe.flags,
             "-I", str(INCLUDE), tmp, "-o", os.devnull],
            capture_output=True, text=True)
    finally:
        os.unlink(tmp)
    stderr = r.stderr.replace(tmp, str(path) if path else "<generated>")
    return Result(path or Path("<generated>"), r.returncode == 0, stderr)
