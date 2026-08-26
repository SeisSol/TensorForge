# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Toolchain discovery, per-arch feasibility probing, and compilation.

Two things gate a test ``Target``: the GPU is present (from
:mod:`gpu_detect`) *and* a toolchain exists that can target its arch.
A probe compile is done once per ``(backend, arch)`` and cached in
memory for the session.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .gpu_detect import DetectedGPU


@dataclass(frozen=True)
class Target:
    """A single runnable configuration: a device plus a backend."""
    backend: str     # "cuda" | "hip" | "sycl"
    arch: str
    vendor: str
    device_index: int
    device_name: str

    @property
    def id(self) -> str:
        return f"{self.backend}-{self.arch}-dev{self.device_index}"


# ----------------------------------------------------------------------
# Probing
# ----------------------------------------------------------------------

_probe_cache: Dict[Tuple[str, str], bool] = {}


def _compiler_for(backend: str) -> Optional[str]:
    exe = {
        "cuda": os.environ.get("NVCC", "nvcc"),
        "hip": os.environ.get("HIPCC", "hipcc"),
        "oneapi": os.environ.get("ICPX", "icpx"),
        "acpp": os.environ.get("ACPP", "acpp"),
    }.get(backend)
    if exe is None or shutil.which(exe) is None:
        return None
    return exe


def _probe_compile(backend: str, arch: str, scratch: Path) -> bool:
    """Compile a trivial kernel for ``arch`` to see if the toolchain can."""
    key = (backend, arch)
    if key in _probe_cache:
        return _probe_cache[key]

    cc = _compiler_for(backend)
    if cc is None:
        _probe_cache[key] = False
        return False

    scratch.mkdir(parents=True, exist_ok=True)
    if backend == "cuda":
        src = scratch / "probe.cu"
        src.write_text("__global__ void k() {}\n")
        obj = scratch / "probe.o"
        cmd = [cc, "-std=c++17", f"-arch={arch}", "-c", str(src), "-o", str(obj)]
    elif backend == "hip":
        src = scratch / "probe.cpp"
        src.write_text("#include <hip/hip_runtime.h>\n__global__ void k() {}\n")
        obj = scratch / "probe.o"
        cmd = [cc, "-std=c++17", f"--offload-arch={arch}", "-c", str(src), "-o", str(obj)]
    elif backend == "oneapi":
        # icpx -fsycl with the JIT path; AOT (-fsycl-targets=...) is per-arch
        # and out of MVP scope.
        src = scratch / "probe.cpp"
        src.write_text(
            "#include <sycl/sycl.hpp>\n"
            "int main() { sycl::queue q; q.wait(); return 0; }\n"
        )
        obj = scratch / "probe.bin"
        cmd = [cc, "-fsycl", "-std=c++17", str(src), "-o", str(obj)]
    elif backend == "acpp":
        src = scratch / "probe.cpp"
        src.write_text(
            "#include <sycl/sycl.hpp>\n"
            "int main() { sycl::queue q; q.wait(); return 0; }\n"
        )
        obj = scratch / "probe.bin"
        cmd = [cc, "-std=c++17", str(src), "-o", str(obj)]
    else:
        _probe_cache[key] = False
        return False

    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=120, check=False)
    except (OSError, subprocess.SubprocessError):
        _probe_cache[key] = False
        return False

    ok = (res.returncode == 0)
    _probe_cache[key] = ok
    return ok


def discover_targets(gpus: List[DetectedGPU], scratch: Path,
                     backends: Optional[List[str]] = None) -> List[Target]:
    """Cross-product GPUs with backends whose toolchain can build for them.

    Backend / vendor compatibility:

    * ``cuda``    only on NVIDIA
    * ``hip``     only on AMD
    * ``oneapi``  preferentially Intel; also runs on NVIDIA via plug-ins
                  but we don't auto-enable that — too many bespoke
                  configurations.
    * ``acpp``    runs on whatever AdaptiveCpp is configured for; we
                  enable it on every GPU and let the probe filter.
    """
    if backends is None:
        backends = ["cuda", "hip", "oneapi", "acpp"]

    targets: List[Target] = []
    for gpu in gpus:
        for backend in backends:
            if backend == "cuda" and gpu.vendor != "nvidia":
                continue
            if backend == "hip" and gpu.vendor != "amd":
                continue
            if backend == "oneapi" and gpu.vendor != "intel":
                continue
            if not _probe_compile(backend, gpu.arch, scratch / "probe"):
                continue
            targets.append(Target(
                backend=backend, arch=gpu.arch, vendor=gpu.vendor,
                device_index=gpu.index, device_name=gpu.name,
            ))
    return targets


# ----------------------------------------------------------------------
# Compilation
# ----------------------------------------------------------------------

@dataclass
class BuildInputs:
    """Everything the emit stage produced, plus metadata needed to build."""
    workdir: Path
    includes_src: str           # from gen.get_helper_headers()
    kernel_src: str             # from gen.get_kernel()
    header_src: str             # from gen.get_header()
    launcher_src: str           # from gen.get_launcher()
    driver_src: str             # from driver_emit
    target: Target
    tensorforge_include: Path   # path to tensorforge/include/


def _cache_hash(b: BuildInputs) -> str:
    h = hashlib.sha256()
    h.update(b.target.id.encode())
    for part in (b.includes_src, b.kernel_src, b.header_src, b.launcher_src, b.driver_src):
        h.update(part.encode())
    return h.hexdigest()[:16]


def build(b: BuildInputs, cache_root: Path) -> Path:
    """Compile the driver; return the path to the executable.

    Uses a content-hash cache so repeated runs of the same case skip
    compilation. The cache key covers kernel/launcher/driver text and
    the target, so any generator change invalidates cleanly.
    """
    key = _cache_hash(b)
    out_dir = cache_root / key
    exe = out_dir / "test"
    if exe.exists():
        return exe

    out_dir.mkdir(parents=True, exist_ok=True)

    # Layout mirrors what the generator assumes: kernel+header together,
    # driver + launcher each as its own TU. Keeping the driver separate
    # makes retargeting to hipcc/acpp a one-file change.
    (out_dir / "kernels.h").write_text(b.header_src)
    (out_dir / "main.cu" if b.target.backend == "cuda" else
     out_dir / "main.cpp").write_text(b.driver_src)

    if b.target.backend == "cuda":
        kernel_file = out_dir / "kernels.cu"
        kernel_file.write_text(b.includes_src + "\n" + b.kernel_src + "\n" + b.launcher_src)
        driver_file = out_dir / "main.cu"
        _compile_cuda(b, kernel_file, driver_file, exe)
    elif b.target.backend == "hip":
        kernel_file = out_dir / "kernels.cpp"
        kernel_file.write_text(b.includes_src + "\n" + b.kernel_src + "\n" + b.launcher_src)
        driver_file = out_dir / "main.cpp"
        _compile_hip(b, kernel_file, driver_file, exe)
    elif b.target.backend in ("oneapi", "acpp"):
        kernel_file = out_dir / "kernels.cpp"
        # SYCL launchers reference sycl::queue / sycl::range; the kernel
        # source itself is plain SYCL C++. Both go in one TU.
        kernel_file.write_text(
            "#include <sycl/sycl.hpp>\n#include \"kernels.h\"\n"
            + b.includes_src + "\n" + b.kernel_src + "\n" + b.launcher_src
        )
        driver_file = out_dir / "main.cpp"
        _compile_sycl(b, kernel_file, driver_file, exe)
    else:
        raise RuntimeError(f"no build recipe for backend {b.target.backend!r}")

    return exe


def _compile_cuda(b: BuildInputs, kernel: Path, driver: Path, exe: Path) -> None:
    cc = _compiler_for("cuda")
    if cc is None:
        raise RuntimeError("nvcc not found")

    aux_cu = b.tensorforge_include / "tensorforge_aux.cu"
    cmd = [
        cc, "-std=c++17", f"-arch={b.target.arch}", "--expt-relaxed-constexpr",
        "-I", str(b.tensorforge_include),
        "-I", str(exe.parent),
        str(driver), str(kernel), str(aux_cu),
        "-o", str(exe),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if res.returncode != 0:
        _dump_failure(exe.parent, cmd, res)
        raise RuntimeError(f"nvcc failed; see {exe.parent}/compile.log")


def _compile_hip(b: BuildInputs, kernel: Path, driver: Path, exe: Path) -> None:
    cc = _compiler_for("hip")
    if cc is None:
        raise RuntimeError("hipcc not found")

    aux = b.tensorforge_include / "tensorforge_aux.cpp"
    cmd = [
        cc, "-std=c++17", f"--offload-arch={b.target.arch}",
        "-I", str(b.tensorforge_include),
        "-I", str(exe.parent),
        str(driver), str(kernel), str(aux),
        "-o", str(exe),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if res.returncode != 0:
        _dump_failure(exe.parent, cmd, res)
        raise RuntimeError(f"hipcc failed; see {exe.parent}/compile.log")


def _compile_sycl(b: BuildInputs, kernel: Path, driver: Path, exe: Path) -> None:
    """Build a SYCL test binary.

    ``oneapi`` and ``acpp`` differ mainly in the JIT/AOT flag soup, but
    for a JIT build (sufficient for the harness — we run on whatever
    device the queue selector finds) the command lines are nearly the
    same. Targeting a specific arch (sm_86, gfx90a, pvc) would mean
    ``-fsycl-targets=...``; we don't enforce that here, leaving it to
    runtime device selection.
    """
    cc = _compiler_for(b.target.backend)
    if cc is None:
        raise RuntimeError(f"{b.target.backend} compiler not found")

    aux = b.tensorforge_include / "tensorforge_aux_sycl.cpp"
    base = [cc, "-std=c++17",
            "-I", str(b.tensorforge_include),
            "-I", str(exe.parent),
            str(driver), str(kernel), str(aux),
            "-o", str(exe)]
    if b.target.backend == "oneapi":
        cmd = [cc, "-fsycl"] + base[1:]
    else:        # acpp
        cmd = base
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if res.returncode != 0:
        _dump_failure(exe.parent, cmd, res)
        raise RuntimeError(f"{b.target.backend} compile failed; see {exe.parent}/compile.log")


def _dump_failure(where: Path, cmd: List[str], res: subprocess.CompletedProcess) -> None:
    log = where / "compile.log"
    log.write_text(textwrap.dedent(f"""\
        cmd: {' '.join(cmd)}
        returncode: {res.returncode}

        --- stdout ---
        {res.stdout}
        --- stderr ---
        {res.stderr}
    """))
