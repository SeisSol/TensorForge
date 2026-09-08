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

# All four, because the pass is not backend-specific and two of them were the
# ones this test was written after missing something on.
TARGETS = [("cuda", "sm_86"), ("hip", "gfx90a"),
           ("acpp", "pvc"), ("esimd", "pvc")]

#: Defects this test found, pinned rather than described.
#:
#: The clone `_advance` makes of a slice member drops `decl` and `extern`,
#: because a declarator with a name in it cannot be emitted twice.  The
#: emitter then renders the type itself, and for a vectorised transfer that is
#: `tensorforge::VectorT<float, 4>` -- which is what the CUDA and HIP paths
#: use and is not what the SPMD lowering wants, where the destination is a
#: `sycl::vec<float, 4>`.  So the clone is correct C++ and the wrong type.
#:
#: It compiles with the pass off and not with it on, which is the definition
#: of a regression the pass causes; it is xfailed rather than fixed here
#: because the fix is to give the clone a declarator, and that is a change to
#: `decl_expr` rather than to this test.
KNOWN_BAD = {
    ("aligned_operands", "acpp"):
        "the advanced clone renders VectorT where the SPMD lowering wants "
        "sycl::vec -- dropping `decl` on a clone drops the backend's spelling",
}


def _generate(mod, backend, arch, *, wrap):
    ctx = Context(arch=arch, backend=backend,
                  fp_type=getattr(mod, "DTYPE", None),
                  options=Options(enable_wrap_loads=wrap))
    gen = Generator(mod.descr_list(), ctx)
    with contextlib.redirect_stdout(io.StringIO()):
        gen.generate()
    return gen.get_kernel()


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
        kernel = _generate(mod, backend, arch, wrap=True)
    except Exception as exc:
        # Only a skip if it fails *both* ways.  A case that generates without
        # the pass and not with it is a regression the pass caused, and
        # skipping on any exception is how this test would hide exactly what
        # it exists to find: the reason `pipeStage0` reached a kernel
        # undeclared was that nothing compiled the wrapped output, and a skip
        # here is the same hole one level in.
        try:
            _generate(mod, backend, arch, wrap=False)
        except Exception:
            pytest.skip(f"does not generate either way: {type(exc).__name__}")
        raise AssertionError(
            f"{name} [{backend}] generates with the wrap pass off and not on: "
            f"{type(exc).__name__}: {exc}") from exc

    if not kernel:
        pytest.skip("no kernel section")

    if (name, backend) in KNOWN_BAD:
        pytest.xfail(KNOWN_BAD[(name, backend)])

    result = syntax.check_source(kernel, shim=syntax.shim_for(backend))
    if result.ok is None:
        pytest.skip(result.reason)
    assert result.ok, (
        f"{name} [{backend}] does not compile with the wrap pass on:\n"
        f"{result.stderr}")
