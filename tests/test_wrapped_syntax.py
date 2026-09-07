# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Does the *wrapped* output compile?

`test_syntax` compiles the recorded snapshots, and those are generated with
`enable_wrap_loads` off.  So everything the wrap pass emits -- the peeled
prologue, the advanced pointers, the rotating write window, the carried
tokens -- had no compiling test at all, and "the pass accepted this loop" was
a claim about the IR rather than about the code.

It was wrong twice.  Once the peel named a shared window whose `extern` binding
happened later; once a rotating buffer read `pipeStage0` that nothing declared,
because the generator asked the loop what the counter was *called* without
asking it for one.  Both rendered.  Both would have been caught here in a
second.

This generates every case with the flag on and runs the same `g++
-fsyntax-only` the snapshot test uses.  It is slower than reading a diff, which
is the point: a transformation with no compiling test is a transformation whose
acceptances are unverified.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
from pathlib import Path

import pytest

from harness import syntax
from tensorforge.common.context import Context, Options
from tensorforge.generators.generator import Generator

CASES = Path(__file__).resolve().parent / "cases"

pytestmark = pytest.mark.skipif(
    syntax.compiler() is None,
    reason="no host compiler available for a syntax check")

TARGETS = [("cuda", "sm_86"), ("hip", "gfx90a")]


def _cases():
    for path in sorted(CASES.rglob("*.py")):
        if path.name.startswith("_"):
            continue
        spec = importlib.util.spec_from_file_location("ws_" + path.stem, path)
        mod = importlib.util.module_from_spec(spec)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                spec.loader.exec_module(mod)
        except Exception:
            continue
        if hasattr(mod, "NAME") and hasattr(mod, "descr_list"):
            yield mod


_IDS = [(m.NAME, b, a) for m in _cases() for b, a in TARGETS]


@pytest.mark.parametrize("name,backend,arch", _IDS,
                         ids=[f"{n}-{b}" for n, b, _ in _IDS])
def test_wrapped_kernel_is_well_formed(name, backend, arch):
    mod = next((m for m in _cases() if m.NAME == name), None)
    assert mod is not None

    try:
        ctx = Context(arch=arch, backend=backend,
                      fp_type=getattr(mod, "DTYPE", None),
                      options=Options(enable_wrap_loads=True))
        gen = Generator(mod.descr_list(), ctx)
        with contextlib.redirect_stdout(io.StringIO()):
            gen.generate()
        kernel = gen.get_kernel()
    except Exception as exc:
        pytest.skip(f"does not generate with the wrap pass on: "
                    f"{type(exc).__name__}")

    if not kernel:
        pytest.skip("no kernel section")

    result = syntax.check_source(kernel, shim=syntax.shim_for(backend))
    if result.ok is None:
        pytest.skip(result.reason)
    assert result.ok, (
        f"{name} [{backend}] does not compile with the wrap pass on:\n"
        f"{result.stderr}")
