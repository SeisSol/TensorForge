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
    repo = os.path.dirname(os.path.dirname(os.path.abspath(tensorforge.__file__)))
    candidate = os.path.join(repo, "tests")
    if not os.path.isdir(candidate):
        raise SystemExit(
            f"cannot find TensorForge's tests/ next to {tensorforge.__file__}; "
            f"set TF_TESTS to the directory holding kernel_eval.py")
    return candidate


def add_tests_to_path():
    d = tests_dir()
    if d not in sys.path:
        sys.path.insert(0, d)
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    return d
