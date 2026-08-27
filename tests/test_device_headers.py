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


def test_reduction_operators_hold_their_contract(tmp_path):
    binary = tmp_path / "reduction_ops"
    compile_cmd = [
        syntax.compiler(), "-std=c++17", "-Wall", "-Wextra", "-Werror",
        f"-I{INCLUDE}", "-o", str(binary), str(SOURCE),
    ]

    built = subprocess.run(compile_cmd, capture_output=True, text=True,
                           timeout=120)
    assert built.returncode == 0, (
        "tests/cpp/reduction_ops.cpp did not compile; a failing static_assert "
        "names the property that broke:\n" + (built.stderr or built.stdout))

    ran = subprocess.run([str(binary)], capture_output=True, text=True,
                         timeout=60)
    assert ran.returncode == 0, (ran.stderr or ran.stdout)
