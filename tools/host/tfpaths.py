# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Where the pieces live.

`kernel_eval` is not part of the installed package --- it sits in the
repository's `tests/` directory --- so an editable install is the easiest way
to find it.  Set `TF_TESTS` to override.
"""
import os
import sys


def tests_dir():
    override = os.environ.get("TF_TESTS")
    if override:
        return override
    import tensorforge
    # walk up from the package: `tests/` sits at the repository root, which is
    # one level up in a flat layout and two under `src/`
    here = os.path.dirname(os.path.abspath(tensorforge.__file__))
    for _ in range(4):
        here = os.path.dirname(here)
        candidate = os.path.join(here, "tests")
        if os.path.isdir(candidate):
            return candidate
    raise SystemExit(
        f"cannot find TensorForge's tests/ above {tensorforge.__file__}; "
        f"set TF_TESTS to the directory holding kernel_eval.py")


def add_tests_to_path():
    d = tests_dir()
    if d not in sys.path:
        sys.path.insert(0, d)
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    return d
