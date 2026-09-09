# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""`cuda::pipeline` has a floor, and two files have to agree on where it is.

`<cuda/pipeline>` reaches `<cuda/barrier>`, which is a hard `#error` below
sm_70.  So the pipeline object is not a declaration the compiler drops for want
of a use on Pascal --- it is a translation unit that does not build, and the
generator puts one in every NVIDIA kernel.  Nothing about that is visible in
the generated text, which renders and snapshots identically either way; it only
shows up in a compiler that no unit test runs.

Two numbers state the floor: the preprocessor guard in `cuda.h` and
:meth:`HwDecription.has_cuda_pipeline`.  They are in different languages and
neither can read the other, so the last test here pins them to each other
rather than to a literal --- a floor moved on one side and not the other is
exactly the shape of the bug, and hard-coding 70 in the test would let both
sides drift together past it.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

from tensorforge.common.context import Context
from tensorforge.common.vm.hw_descr import hw_descr_factory
from tensorforge.generators.generator import Generator

CASES = Path(__file__).resolve().parent / "cases"
CUDA_H = (Path(__file__).resolve().parent.parent / "src" / "tensorforge" /
          "include" / "tensorforge_device" / "cuda.h")

#: Enough of the NVIDIA table to cover both sides of the floor, plus a
#: three-digit model: `sm_100` sorts below `sm_60` under a two-character read
#: of the suffix, and that read is the plausible way to get this wrong.
BELOW = ("sm_60", "sm_61", "sm_62")
ABOVE = ("sm_70", "sm_75", "sm_80", "sm_86", "sm_90", "sm_100", "sm_120")


def _case(name: str = "square_notrans"):
    path = CASES / f"{name}.py"
    spec = importlib.util.spec_from_file_location("tf_floor__" + name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _render(arch: str, backend: str = "cuda") -> str:
    mod = _case()
    ctx = Context(arch=arch, backend=backend,
                  fp_type=getattr(mod, "DTYPE", None))
    gen = Generator(mod.descr_list(), ctx)
    gen.generate()
    return str(gen.get_kernel())


@pytest.mark.parametrize("arch", BELOW)
def test_no_pipeline_object_below_the_floor(arch):
    assert "cuda::pipeline" not in _render(arch), (
        f"{arch} cannot include <cuda/pipeline>, so a kernel naming the type "
        f"does not compile there")


@pytest.mark.parametrize("arch", ABOVE)
def test_pipeline_object_above_the_floor(arch):
    """The declaration is unconditional where the type exists.

    Whether any transfer drives it is decided per body, after this point, so an
    unused local is the deliberate price of not building the section twice.
    """
    assert "cuda::pipeline" in _render(arch)


def test_hip_on_an_nvidia_model_names_no_cuda_type():
    """The vendor is NVIDIA and the spelling is still unavailable.

    `hip_lexic` emits the HIP headers, and `cuda::pipeline` is in none of them.
    """
    assert not hw_descr_factory("sm_90", "hip").has_cuda_pipeline()


@pytest.mark.parametrize("arch,expected", [
    ("sm_60", 60), ("sm_86", 86), ("sm_100", 100), ("sm_120", 120)])
def test_sm_level_reads_every_digit(arch, expected):
    assert hw_descr_factory(arch, "cuda").sm_level() == expected


def test_amd_model_has_no_compute_capability():
    assert hw_descr_factory("gfx90a", "hip").sm_level() is None


def test_header_guard_and_generator_agree_on_the_floor():
    guard = re.search(r"__CUDA_ARCH__\s*>=\s*(\d+)", CUDA_H.read_text())
    assert guard, "cuda.h no longer guards <cuda/pipeline> by architecture"

    lowest = min(int(a[3:]) for a in BELOW + ABOVE
                 if hw_descr_factory(a, "cuda").has_cuda_pipeline())
    assert int(guard.group(1)) == lowest * 10, (
        "the header admits <cuda/pipeline> from one architecture and the "
        "generator names the type from another")
