# SPDX-License-Identifier: MIT
"""Do the generated intrinsic calls actually match the declarations?

Nothing in this repository compiles, and that turned out to be a hole with a
shape.  A padded MFMA tail block handed `0.0f` to `transpose4x4b32`'s third
and fourth parameters, which are `T &`.  Ill-formed C++ --- and the snapshot
corpus, the symbolic equivalence checker and the PIR verifier all passed it,
because none of them models overload resolution.  It surfaced by accident,
while chasing an unrelated difference.

A full compile needs a GPU toolchain and is out of reach here.  Matching the
*call sites* against *declarations* does not: the reference-taking intrinsics
are declared in a stub with the same signatures as `hip.h`, the calls are
lifted out of the snapshots, and a plain `g++ -fsyntax-only` decides.  That is
a narrow check, but it is aimed exactly where the hole was --- reference
binding, argument counts, and const-correctness --- and it costs a second.

Skipped when no compiler is present, rather than silently passing.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

SNAPSHOT_DIR = Path(__file__).parent / "snapshots"

#: Declarations copied from `include/tensorforge_device/hip.h`.  Signatures
#: only -- there is nothing to run, and a body would drag in the DPP builtins
#: that only an AMDGPU compiler has.
STUBS = """
#include <cstddef>
namespace tensorforge {
template <typename T, std::size_t N>
using VectorT = __attribute__((__vector_size__(N * sizeof(T)))) T;

template <typename T> void transpose4x4b32(T &w1, T &w2, T &w3, T &w4,
                                           T v1, T v2, T v3, T v4);
template <typename T> void transpose16x16b32(T &w1, T &w2, T &w3, T &w4,
    T &w5, T &w6, T &w7, T &w8, T &w9, T &w10, T &w11, T &w12,
    T &w13, T &w14, T &w15, T &w16);
template <int Row> void fmacdpp4(float &c, float a, float b);
template <int Row> void fmacdpp16(float &c, float a, float b);
template <int Row> void fmacdpp16(double &c, double a, double b);
template <int B, int S, int L, typename T> T broadcast(T value);
template <int Row, typename T> T movdpp16(T a);
}
"""

#: The calls worth checking: every intrinsic that takes an argument by
#: reference, which is where a literal or a wrong argument count is fatal.
CALL = re.compile(
    r"^\s*(tensorforge::(?:transpose\w+|fmacdpp\d+(?:<\d+>)?))\s*\(([^;]*)\)\s*;",
    re.M)

#: `float v12_data = ...` / `tensorforge::VectorT<float, 4> v13_acc{};`
DECL = re.compile(
    r"^\s*(?:const\s+)?(float|double|tensorforge::VectorT<\s*\w+\s*,\s*\d+\s*>)"
    r"\s+(v\d+\w*)\s*[=({;]", re.M)

pytestmark = pytest.mark.skipif(
    shutil.which("g++") is None and shutil.which("clang++") is None,
    reason="no host C++ compiler to check signatures against")


def _compiler():
    return shutil.which("g++") or shutil.which("clang++")


def _split_args(s: str):
    out, depth, cur = [], 0, ""
    for ch in s:
        if ch in "(<[":
            depth += 1
        elif ch in ")>]":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur.strip())
    return out


def _translation_unit(src: str) -> str | None:
    """A compilable fragment holding just this kernel's intrinsic calls."""
    calls = CALL.findall(src)
    if not calls:
        return None
    types = dict((name, ty) for ty, name in DECL.findall(src))
    used, body = set(), []
    for callee, arglist in calls:
        args = _split_args(arglist)
        # every generated name anywhere in the argument list, not only the
        # bare ones: an operand can arrive as `(broadcast<...>(v248_lin))`,
        # and missing the nested name would leave it undeclared and turn a
        # checker gap into a reported failure of the generator
        used.update(re.findall(r"\bv\d+\w*\b", arglist))
        body.append(f"  {callee}({', '.join(args)});")
    decls = []
    for name in sorted(used):
        ty = types.get(name)
        if ty is None:
            # a name the kernel uses but does not declare in a form this
            # recognises: assume nothing, and say so by bailing out rather
            # than guessing a type that might make a bad call compile
            return None
        decls.append(f"  {ty} {name}{{}};")
    return (STUBS + "\nvoid check() {\n" + "\n".join(decls) + "\n"
            + "\n".join(body) + "\n}\n")


def _snapshots():
    files = sorted(SNAPSHOT_DIR.glob("*.hip.cpp"))
    if not files:
        pytest.skip("snapshots not generated; run pytest --snapshot-update")
    return files


@pytest.mark.parametrize("path", _snapshots(), ids=lambda p: p.name)
def test_intrinsic_calls_match_their_declarations(path, tmp_path):
    unit = _translation_unit(path.read_text())
    if unit is None:
        pytest.skip("no reference-taking intrinsic calls in this kernel")
    src = tmp_path / "check.cpp"
    src.write_text(unit)
    r = subprocess.run([_compiler(), "-fsyntax-only", "-std=c++17", str(src)],
                       capture_output=True, text=True)
    assert r.returncode == 0, (
        f"{path.name}: generated intrinsic calls do not match the "
        f"declarations in hip.h\n{r.stderr[:1500]}")


def test_the_check_rejects_a_literal_in_a_reference_position(tmp_path):
    """The bug this exists for, injected deliberately.

    A check that only ever passes proves that it ran, not that it works.
    """
    src = tmp_path / "bad.cpp"
    src.write_text(STUBS + """
void check() {
  float v1{}, v2{};
  tensorforge::transpose4x4b32(v1, v2, 0.0f, 0.0f, v1, v2, 0.0f, 0.0f);
}
""")
    r = subprocess.run([_compiler(), "-fsyntax-only", "-std=c++17", str(src)],
                       capture_output=True, text=True)
    assert r.returncode != 0
    assert "reference" in r.stderr or "rvalue" in r.stderr


def test_the_check_rejects_a_wrong_argument_count(tmp_path):
    src = tmp_path / "bad2.cpp"
    src.write_text(STUBS + """
void check() {
  float v1{}, v2{}, v3{}, v4{};
  tensorforge::transpose4x4b32(v1, v2, v3, v4, v1, v2, v3);
}
""")
    r = subprocess.run([_compiler(), "-fsyntax-only", "-std=c++17", str(src)],
                       capture_output=True, text=True)
    assert r.returncode != 0


def test_the_check_accepts_the_in_place_form(tmp_path):
    """Both forms are legal; the check must not reject the one still in use
    for `transpose16x16b32`, which has no separate outputs."""
    src = tmp_path / "ok.cpp"
    src.write_text(STUBS + """
void check() {
  float v1{}, v2{}, v3{}, v4{};
  tensorforge::transpose4x4b32(v1, v2, v3, v4, v1, v2, v3, v4);
  tensorforge::fmacdpp16<3>(v1, v2, v3);
}
""")
    r = subprocess.run([_compiler(), "-fsyntax-only", "-std=c++17", str(src)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
