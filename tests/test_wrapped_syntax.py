# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Does the *wrapped* output compile?

`test_syntax` compiles the recorded snapshots, and those are generated with
`enable_wrap_loads` off.  So everything the wrap pass emits -- the peel ahead
of the loop, the pointers bound for the next element, the carried tokens and
flag words -- would otherwise have no compiling test at all, and "the pass
accepted this loop" would be a claim about the IR rather than about the code.

Such a claim can be wrong and still render: a peel that names a pointer whose
binding comes later, or the copy of a declaration that kept the original's
name.  A compile catches either in a second.

This generates every case with the flag on and runs the same `g++
-fsyntax-only` the snapshot test uses -- and once more with a second stage
where the copies are asynchronous, which adds the stage the loop carries and
the windows into it.  It is slower than reading a diff, which is the point: a
transformation with no compiling test is a transformation whose acceptances
are unverified.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
from pathlib import Path

import pytest

from harness import UNSUPPORTED, syntax
from tensorforge.common.context import Context, Options
from tensorforge.generators.generator import Generator

CASES = Path(__file__).resolve().parent / "cases"

pytestmark = pytest.mark.skipif(
    syntax.compiler() is None,
    reason="no host compiler available for a syntax check")

# All four, because the pass is not backend-specific.
TARGETS = [("cuda", "sm_86"), ("hip", "gfx90a"),
           ("acpp", "pvc"), ("esimd", "pvc")]

#: What the pass is asked for: the move everywhere, and a second stage on top
#: where copies are asynchronous -- anywhere else it changes nothing.
CONFIGS = {
    "wrap": dict(enable_wrap_loads=True),
    "multibuffer": dict(enable_wrap_loads=True, enable_multibuffer=True),
}
STAGED = {("cuda", "sm_86")}


def _generate(mod, backend, arch, options):
    ctx = Context(arch=arch, backend=backend,
                  fp_type=getattr(mod, "DTYPE", None),
                  options=Options(**options))
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


_IDS = [(m.NAME, b, a, c) for m in _cases() for b, a in TARGETS
        for c in CONFIGS if c == "wrap" or (b, a) in STAGED]


@pytest.mark.parametrize("name,backend,arch,config", _IDS,
                         ids=[f"{n}-{b}" if c == "wrap" else f"{n}-{b}-{c}"
                              for n, b, _, c in _IDS])
def test_wrapped_kernel_is_well_formed(name, backend, arch, config):
    mod = next((m for m in _cases() if m.NAME == name), None)
    assert mod is not None

    try:
        kernel = _generate(mod, backend, arch, CONFIGS[config])
    except UNSUPPORTED as exc:
        # Only a skip if it fails *both* ways.  A case that generates without
        # the pass and not with it is a regression the pass caused, and
        # skipping on any exception is how this test would hide exactly what
        # it exists to find: a wrapped kernel nothing compiles can carry a
        # name nothing declares, and a skip here is the same hole one level
        # in.
        try:
            _generate(mod, backend, arch, {})
        except UNSUPPORTED:
            pytest.skip(f"does not generate either way: {type(exc).__name__}")
        raise AssertionError(
            f"{name} [{backend}] generates with the wrap pass off and not on: "
            f"{type(exc).__name__}: {exc}") from exc

    if not kernel:
        pytest.skip("no kernel section")

    result = syntax.check_source(kernel, shim=syntax.shim_for(backend))
    if result.ok is None:
        pytest.skip(result.reason)

    assert result.ok, (
        f"{name} [{backend}] does not compile with {config}:\n"
        f"{result.stderr}")
