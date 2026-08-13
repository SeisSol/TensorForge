# SPDX-License-Identifier: MIT
"""The codegen's idea of what the runtime defines has to match the runtime.

`amd.py` decides which `fmacdpp` width to emit; `hip.h` decides which ones
exist, behind `#if` guards.  Those are two copies of the same fact in two
languages, and they had drifted: the generator emitted `fmacdpp4` for gfx900,
where the specialisations are switched off, and `fmacdpp8`, which the runtime
does not declare for any target at all.  Neither showed up in a test, because
a call to a template with no definition is a *link* error and nothing here
links.

So the guards are parsed out of the header and compared against the Python
predicates.  The duplication stays -- the generator cannot include a C++
header -- but it stops being silent.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tensorforge.backend.instructions.compute.primitives import amd
from tensorforge.common.basic_types import Datatype
from tensorforge.common.context import Context

HIP_H = (Path(__file__).parent.parent / "tensorforge" / "include" /
         "tensorforge_device" / "hip.h")

ARCHS = ["gfx900", "gfx906", "gfx908", "gfx90a", "gfx940", "gfx942", "gfx950",
         "gfx1010", "gfx1030", "gfx1100", "gfx1200", "gfx1250", "gfx1251"]


def _family_macros(arch: str) -> set:
    """The LLVM target macros a given arch is compiled with.

    UNVERIFIED for gfx1250/gfx1251.  Everything up to gfx1200 follows the
    documented `__GFX<n>__` family scheme, but gfx125x is treated as its own
    generation by `amd.py` itself -- `rdna()` stops below 0x1250 and `cdna2()`
    folds gfx1251 in -- so whether clang defines `__GFX12__` there, or
    something like `__GFX12_5__`, is a question about a toolchain that is not
    installed here.

    It matters: if `__GFX12__` is *not* defined for gfx1250, then hip.h's
    guard is false there, `fmacdpp16` has no specialisations, and the
    generator emits a call to an undeclared template -- exactly the gfx900
    failure, one family over.  The assumption is written down here rather than
    buried in a string slice so that it can be settled with one compile.
    """
    macros = {f"__{arch}__"}
    n = int(arch[3:], 16)
    if n < 0x1000:
        macros.add("__GFX9__")
    elif n >= 0x1250:
        macros.add("__GFX12__")     # ASSUMPTION -- see docstring
    else:
        macros.add(f"__GFX{arch[3:5]}__")
    return macros


#: Architectures whose family macro above is an assumption rather than a fact.
UNVERIFIED_FAMILY = {"gfx1250", "gfx1251"}


def _guard_holds(cond: str, arch: str) -> bool:
    """Evaluate a `#if` line made of `defined(...)`, `!`, `||` for one arch."""
    macros = _family_macros(arch)
    expr = re.sub(r"defined\s*\(\s*(\w+)\s*\)",
                  lambda m: str(m.group(1) in macros), cond)
    expr = expr.replace("||", " or ").replace("&&", " and ").replace("!", " not ")
    return bool(eval(expr, {"__builtins__": {}}, {}))  # noqa: S307


def _guard_for(symbol: str) -> str:
    """The `#if` condition guarding the first specialisation of `symbol`.

    Scans for the `constexpr bool Has...` flag that each block opens with,
    which is what the header uses to advertise the capability, then walks back
    over the directive's line continuations to the `#if` itself.
    """
    src = HIP_H.read_text()
    flag = {"fmacdpp4": "HasFmacDpp4", "fmacdpp16": "HasFmacDpp16"}[symbol]
    lines = src.splitlines()
    for i, line in enumerate(lines):
        if re.match(rf"\s*constexpr bool {flag}\s*=\s*true", line):
            # The directive may span several lines; only the non-final ones
            # carry the trailing backslash, so walk back until a line starts
            # with `#if` and join what was collected.
            parts, j = [], i - 1
            while j >= 0:
                s = lines[j].rstrip()
                parts.append(s[:-1].strip() if s.endswith("\\") else s.strip())
                if re.match(r"\s*#if\b", s):
                    break
                j -= 1
            else:
                pytest.fail(f"no #if above {flag}")
            cond = " ".join(reversed(parts))
            return re.sub(r"^#if\s+", "", cond).strip()
    pytest.fail(f"no `constexpr bool {flag} = true` in hip.h")


def _ctx(arch: str, dtype=Datatype.F32):
    return Context(arch=arch, backend="hip", fp_type=dtype)


@pytest.mark.parametrize("arch", ARCHS)
def test_has_fmacdpp4_matches_the_header(arch):
    expected = _guard_holds(_guard_for("fmacdpp4"), arch)
    assert amd.has_fmacdpp4(_ctx(arch)) == expected, (
        f"{arch}: codegen says {amd.has_fmacdpp4(_ctx(arch))}, "
        f"hip.h says {expected}")


@pytest.mark.parametrize("arch", ARCHS)
@pytest.mark.parametrize("dtype", [Datatype.F32, Datatype.F64])
def test_has_fmacdpp16_matches_the_header(arch, dtype):
    expected = _guard_holds(_guard_for("fmacdpp16"), arch)
    got = amd.has_fmacdpp16(_ctx(arch, dtype), dtype)
    assert got == expected, (
        f"{arch}/{dtype}: codegen says {got}, hip.h says {expected}")


def test_fmacdpp8_does_not_exist_in_the_runtime():
    """Guards against the reverse mistake: enabling a width nobody wrote."""
    assert "fmacdpp8" not in HIP_H.read_text()
    for arch in ARCHS:
        assert not amd.has_fmacdpp8(_ctx(arch))


@pytest.mark.parametrize("arch", ARCHS)
@pytest.mark.parametrize("dtype", [Datatype.F32, Datatype.F64])
@pytest.mark.parametrize("threads", [1, 2, 4, 8, 16, 32, 64])
def test_hfma_never_selects_an_undefined_width(arch, dtype, threads):
    """The property that matters, over the whole selection space.

    `hfma` picks a width from the hardware strategy and then narrows it to
    what the target defines.  Whatever comes out has to be linkable --- for
    every architecture, datatype and thread count, not just the ones the
    snapshot corpus happens to reach.
    """
    ctx = _ctx(arch, dtype)
    step = amd.select_fmadpp_step(dtype, threads, ctx)
    available = {
        1: True,
        4: amd.has_fmacdpp4(ctx),
        8: amd.has_fmacdpp8(ctx),
        16: amd.has_fmacdpp16(ctx, dtype),
    }
    assert step in available, f"unknown width {step}"
    assert available[step], (
        f"{arch}/{dtype}/threads={threads} selects fmacdpp{step}, "
        f"which the runtime does not define there")


@pytest.mark.parametrize("arch", ARCHS)
def test_narrowing_never_widens(arch):
    """Falling back must go *down*.  A wider width than the strategy asked
    for would change the broadcast pattern, not just the instruction."""
    for dtype in (Datatype.F32, Datatype.F64):
        for threads in (1, 2, 4, 8, 16, 32, 64):
            ctx = _ctx(arch, dtype)
            wanted = amd.wanted_fmadpp_step(dtype, threads, ctx)
            got = amd.select_fmadpp_step(dtype, threads, ctx)
            assert got <= wanted, (
                f"{arch}/{dtype}/threads={threads}: widened {wanted} -> {got}")


def test_the_unverified_assumption_is_load_bearing():
    """Records what turns on the gfx125x family macro, so it is not forgotten.

    If `__GFX12__` is not what clang defines for gfx1250, hip.h's guard is
    false there and the generator is emitting an undeclared `fmacdpp16` --
    which no test here can see, because the test shares the assumption with
    the code under test.  Asserting the *consequence* at least makes the
    exposure explicit: these are the targets whose answer rests on a guess.
    """
    exposed = {a for a in UNVERIFIED_FAMILY
               if amd.has_fmacdpp16(_ctx(a), Datatype.F32)}
    assert exposed == UNVERIFIED_FAMILY, (
        "the assumption changed; re-check it against a real toolchain")
