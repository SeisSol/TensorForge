# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""The shims name what the device headers name.

`test_syntax.py` checks every snapshot's kernel with a host compiler, against
a shim that declares the device helpers instead of defining them.  A shim is a
second statement of the headers' names, and a second statement drifts: a
helper renamed in `hip.h` but not in the shim leaves the syntax check green
over kernels no vendor compiler accepts, and a shim declaration of something
no header defines lets a kernel call it.  These hold the two statements to
each other by name.  Signatures are the vendor compiler's to check; a name is
what a host compiler cannot.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DEVICE = ROOT / "src" / "tensorforge" / "include" / "tensorforge_device"
SHIM = ROOT / "tests" / "shim"
SNAPSHOTS = ROOT / "tests" / "snapshots"

#: The headers a kernel includes, for each backend the snapshots record.
HEADERS = {
    "cuda": ("base.h", "cuda.h"),
    "hip": ("base.h", "hip.h"),
    "acpp": ("base.h", "isycl.h"),
    "esimd": ("base.h", "isycl.h"),
}

#: Each shim, and the headers it stands in for.
SHIMS = {
    "tensorforge_host.h": ("base.h", "cuda.h", "hip.h"),
    "tensorforge_sycl.h": ("base.h", "isycl.h"),
}

_KERNEL = re.compile(r"^// === kernel ===\n(.*?)(?=^// === |\Z)", re.M | re.S)
_COMMENT = re.compile(r"//[^\n]*|/\*.*?\*/", re.S)
#: A declaration at the top level of a namespace: an alias, a type, a
#: constant, or a function (its name is what comes before the parameters).
_DECLARED = re.compile(
    r"\busing\s+(\w+)\s*=|\bstruct\s+(\w+)|\bclass\s+(\w+)"
    r"|\benum\s+class\s+(\w+)|\bconstexpr\s+[\w:]+\s+(\w+)\s*="
    r"|(\w+)\s*\([^;{]*\)\s*(?:const\s*)?[;{]")


def _text(names) -> str:
    return "\n".join((DEVICE / name).read_text() for name in names)


def _mentions(text: str, name: str) -> bool:
    return re.search(rf"\b{re.escape(name)}\b", text) is not None


def _declared_in_namespace(source: str, namespace: str = "tensorforge"):
    """The names declared at the top level of every `namespace` block."""
    source = _COMMENT.sub("", source)
    names = set()
    for opening in re.finditer(rf"namespace\s+{namespace}\s*{{", source):
        depth, top = 1, []
        for ch in source[opening.end():]:
            if ch == "{":
                depth += 1
                top.append("{" if depth == 2 else "")
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    break
                top.append("}" if depth == 1 else "")
            elif depth == 1:
                top.append(ch)
        for match in _DECLARED.finditer("".join(top)):
            names.add(next(group for group in match.groups() if group))
    return names - {"if", "for", "while", "return", "sizeof", "static_assert"}


@pytest.mark.parametrize("backend", sorted(HEADERS))
def test_every_helper_a_kernel_calls_is_in_its_headers(backend):
    """What the generator emits resolves against the real headers, not only
    against the shim the syntax check uses."""
    headers = _text(HEADERS[backend])
    called = set()
    for snapshot in sorted(SNAPSHOTS.glob(f"*.{backend}.cpp")):
        kernel = _KERNEL.search(snapshot.read_text())
        if kernel:
            called |= set(re.findall(r"tensorforge::(\w+)", kernel.group(1)))
    assert called, f"no {backend} snapshot calls a helper"
    missing = sorted(name for name in called if not _mentions(headers, name))
    assert not missing, (
        f"{backend} kernels call {missing}, which "
        f"{' and '.join(HEADERS[backend])} do not name")


@pytest.mark.parametrize("shim", sorted(SHIMS))
def test_every_helper_a_shim_declares_is_in_its_headers(shim):
    declared = _declared_in_namespace((SHIM / shim).read_text())
    assert declared, f"found nothing declared in {shim}"
    headers = _text(SHIMS[shim])
    missing = sorted(name for name in declared if not _mentions(headers, name))
    assert not missing, (
        f"{shim} declares {missing}, which {', '.join(SHIMS[shim])} do not "
        f"name: a kernel calling them would pass the syntax check and fail "
        f"on the device")
