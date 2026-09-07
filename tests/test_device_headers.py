# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""The device headers' reduction operators, checked by a host compiler.

`tests/cpp/reduction_ops.cpp` is the actual test; this runs it. Everything it
asserts is `static_assert`, so a successful compile *is* the pass and the
binary only exists because running nothing would be a strange way to report a
result.

`base.h` needs no device: no intrinsics, no execution-space keywords, not even
the shim that `test_syntax.py` uses. It also had three wrong neutral elements
and a mistagged specialisation, none of which any test could have caught,
because the C++ side of this repository had no test at all until the syntax
check arrived --- and that one only asks whether generated code parses, which
these headers do regardless of what they compute.

The rest of `cuda.h` and `hip.h` is not covered here and cannot be by this
route: a shuffle butterfly needs lanes. `-fsyntax-only` against the shim (see
`test_syntax.py`) answers for well-formedness, and the arithmetic waits for a
machine.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from harness import syntax

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "cpp" / "reduction_ops.cpp"
INCLUDE = HERE.parent / "src" / "tensorforge" / "include"

pytestmark = pytest.mark.skipif(
    syntax.compiler() is None,
    reason="no host C++ compiler to check the device headers with")


#: Every executable check under `tests/cpp`.  Listed rather than globbed so
#: that a file added without being wired up is a missing test rather than a
#: silently skipped one.
PROGRAMS = ("reduction_ops.cpp", "esimd_reduction.cpp", "sycl_shim.cpp")


def _build_and_run(tmp_path, name):
    source = HERE / "cpp" / name
    binary = tmp_path / name.replace(".cpp", "")
    built = subprocess.run(
        [syntax.compiler(), "-std=c++17", "-Wall", "-Wextra", "-Werror",
         f"-I{INCLUDE}", "-o", str(binary), str(source)],
        capture_output=True, text=True, timeout=120)
    assert built.returncode == 0, (
        f"tests/cpp/{name} did not compile; a failing static_assert names the "
        f"property that broke:\n" + (built.stderr or built.stdout))
    ran = subprocess.run([str(binary)], capture_output=True, text=True,
                         timeout=60)
    assert ran.returncode == 0, (ran.stderr or ran.stdout)


@pytest.mark.parametrize("name", PROGRAMS)
def test_the_cpp_checks_pass(tmp_path, name):
    _build_and_run(tmp_path, name)
