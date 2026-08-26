# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Does the generated code parse, and do its calls resolve?

Nothing in this repository compiled, and that turned out to be a hole with a
shape.  A padded MFMA tail block handed ``0.0f`` to ``transpose4x4b32``'s third
and fourth parameters, which are ``T &``.  Ill-formed C++ --- and the snapshot
corpus, the symbolic equivalence checker and the PIR verifier all passed it,
because none of them models overload resolution.  It surfaced by accident,
while chasing an unrelated difference, and the fix for it was a side effect of
something else.

A full device compile needs a GPU toolchain.  Deciding whether the *source is
well-formed* does not: the intrinsics get declaration-only stubs in
``tests/shim/tensorforge_host.h``, and ``g++ -fsyntax-only`` answers for every
line the generator emitted.  It is not a statement about semantics --- the
stubs have no bodies --- but well-formedness is exactly the class of defect
that escaped everything else, and it costs about four seconds for both
corpora.

This replaces the earlier ``test_signatures.py``, which lifted
reference-taking calls out with a regex and checked those alone.  That check
bailed out (silently, as a skip) on 17 of 46 HIP snapshots whenever an operand
did not match its declaration pattern, and it saw nothing outside the argument
lists.  Two overlapping checks where one is strictly weaker is the arrangement
this codebase keeps finding at the bottom of its bugs, so there is now one.

Skipped when no host compiler is present, rather than silently passing.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from harness import syntax

pytestmark = pytest.mark.skipif(
    syntax.compiler() is None,
    reason="no host C++ compiler to check the generated source with")


def _snapshots():
    files = syntax.snapshots()
    if not files:
        pytest.skip("snapshots not generated; run pytest --snapshot-update")
    return files


@pytest.mark.parametrize("path", _snapshots(), ids=lambda p: p.name)
def test_generated_kernel_is_well_formed(path):
    r = syntax.check_snapshot(path)
    if r.ok is None:
        pytest.skip(r.reason)
    assert r.ok, (
        f"{path.name}: the generated kernel is not well-formed C++\n"
        + "\n".join(r.errors()))


# --------------------------------------------------------------------------
# The shim is a copy of a C++ fact, and copies drift.
# --------------------------------------------------------------------------

INCLUDE = Path(__file__).resolve().parent.parent / "src" / "tensorforge" / "include"
DEVICE_HEADERS = (INCLUDE / "tensorforge_device" / "hip.h",
                  INCLUDE / "tensorforge_device" / "cuda.h")

#: Names the shim declares that also exist in the device headers.  Anything
#: outside this set is either a compiler builtin (no declaration anywhere to
#: compare against) or host-runtime scaffolding.
CHECKED = (
    "transpose4x4b32", "transpose16x16b32", "transpose16x2", "transpose16x4",
    "fmacdpp4", "fmacdpp16", "broadcast", "movdpp16", "splitFloatTF32",
)

_DECL = re.compile(
    r"^(?:template\s*<(?P<tparams>[^>]*)>\s*)?"
    r"(?:(?:__device__|__host__|__forceinline__|inline|constexpr|static)\s+)*"
    r"(?P<ret>[\w:]+(?:\s*<[^<>]*>)?\s*[*&]?)\s+"
    r"(?P<name>\w+)\s*\((?P<params>[^{;()]*)\)\s*(?=[{;])",
    re.S | re.M)


def _declarations_only(text: str) -> str:
    """Comments, preprocessor directives and string literals removed.

    Without this the `#if defined(__GFX10__) || ...` block in `hip.h` parses
    as a function named `defined` whose parameter list runs for two thousand
    characters --- swallowing the `fmacdpp4` declaration that sits inside it
    and reporting the shim as the thing that is wrong.  A checker that
    misreads its own reference is worse than no checker.
    """
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    text = re.sub(r"//[^\n]*", "", text)
    text = re.sub(r'"(?:[^"\\\n]|\\.)*"', '""', text)
    # a directive, including backslash continuations
    text = re.sub(r"^[ \t]*#(?:[^\n\\]|\\\n|\\[^\n])*", "", text, flags=re.M)
    return text


def _norm_type(s: str) -> str:
    """A parameter's type, with the parameter name and `std::` removed."""
    s = re.sub(r"\s+", " ", s).strip()
    s = s.replace("std::", "")
    # drop the declarator name: the last identifier, unless the whole thing is
    # a type (`float`, `T`)
    s = re.sub(r"\b([A-Za-z_]\w*)\s*$", lambda m: "" if "&" in s or "*" in s
               or " " in s.rstrip() else m.group(1), s)
    return re.sub(r"\s+", " ", s).strip()


def _norm_tparams(s: str) -> str:
    """Template parameter *kinds*; the names are not part of a call."""
    if not s:
        return ""
    kinds = []
    for p in s.split(","):
        p = re.sub(r"\s+", " ", p).strip().replace("std::", "")
        kinds.append(p.split()[0] if p else "")
    return ",".join(kinds)


def _signatures(text: str, names) -> dict:
    out = {}
    for m in _DECL.finditer(_declarations_only(text)):
        name = m.group("name")
        if name not in names:
            continue
        params = tuple(_norm_type(p) for p in m.group("params").split(",")
                       if p.strip())
        sig = (_norm_tparams(m.group("tparams") or ""),
               _norm_type(m.group("ret")), params)
        out.setdefault(name, set()).add(sig)
    return out


def test_shim_matches_the_device_headers():
    """The shim declares what the runtime declares --- no more, no less.

    Not tidiness.  ``test_amd_caps.py`` already records the failure mode: a
    test that shares an assumption with the code it checks cannot report the
    assumption being wrong.  A shim that is *more permissive* than the header
    is that failure mode in this file --- it would accept a call the runtime
    rejects, which is precisely the situation this whole check exists to
    catch.
    """
    shim = _signatures(syntax.SHIM.read_text(), CHECKED)
    real = {}
    for header in DEVICE_HEADERS:
        for name, sigs in _signatures(header.read_text(), CHECKED).items():
            real.setdefault(name, set()).update(sigs)

    missing = sorted(set(real) - set(shim))
    assert not missing, (
        f"declared in the device headers, absent from the shim: {missing}. "
        f"A call to one of these would fail to resolve and be reported as a "
        f"generator defect.")

    extra = sorted(set(shim) - set(real))
    assert not extra, (
        f"declared in the shim, absent from the device headers: {extra}. "
        f"The shim would accept a call the runtime cannot link.")

    for name in sorted(shim):
        assert shim[name] == real[name], (
            f"{name}: the shim and the device headers disagree.\n"
            f"  shim:   {sorted(shim[name])}\n"
            f"  header: {sorted(real[name])}")


# --------------------------------------------------------------------------
# Does the check have teeth?  Each of these is a real defect shape.
# --------------------------------------------------------------------------

_GOOD = """
__global__ void k(float* out) {
  float a{}, b{}, c{}, d{}, x{}, y{}, z{}, w{};
  tensorforge::transpose4x4b32(a, b, c, d, x, y, z, w);
  tensorforge::fmacdpp16<3>(a, x, y);
  tensorforge::VectorT<float, 4> acc{};
  acc = __builtin_amdgcn_mfma_f32_4x4x1f32(a, b, acc, 2, 0, 0);
  out[threadIdx.x] = acc[0];
}
"""


def _rejects(src: str) -> bool:
    return not syntax.check_source(src).ok


def test_the_check_accepts_well_formed_source():
    r = syntax.check_source(_GOOD)
    assert r.ok, "\n".join(r.errors())


@pytest.mark.parametrize("name,mutation", [
    # the defect that motivated all of this
    ("a literal in a reference position",
     lambda s: s.replace("transpose4x4b32(a, b, c, d,",
                         "transpose4x4b32(a, b, 0.0f, d,")),
    ("an argument dropped from a transpose",
     lambda s: s.replace(", x, y, z, w);", ", x, y, z);")),
    # things the old extracted-call check could not see
    ("an undeclared operand",
     lambda s: s.replace("tensorforge::fmacdpp16<3>(a, x, y);",
                         "tensorforge::fmacdpp16<3>(a, x, undeclared);")),
    ("a misspelt intrinsic",
     lambda s: s.replace("transpose4x4b32", "transpose4x4b64")),
    ("an accumulator of the wrong width",
     lambda s: s.replace("VectorT<float, 4> acc",
                         "VectorT<float, 2> acc")),
    ("a double handed to the float fmacdpp",
     lambda s: s.replace("float a{}, b{}", "double a{}; float b{}")),
])
def test_the_check_rejects(name, mutation):
    mutated = mutation(_GOOD)
    assert mutated != _GOOD, "mutation did not apply"
    assert _rejects(mutated), f"{name}: not caught"
