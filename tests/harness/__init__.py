# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Numerical end-to-end test harness for TensorForge kernels.

Orchestrates the pipeline:
    case.descr_list()  ->  Generator  ->  emit main.cu  ->  compile  ->  run  ->  compare

Public entrypoints live in :mod:`runner`; pytest wiring lives in the
top-level ``conftest.py``.
"""

from tensorforge.common.exceptions import GenerationError

#: What generating a case raises where the target does not take it: a
#: refusal (`GenerationError`) or a lowering nobody has written
#: (`NotImplementedError`).  A test over the whole corpus skips these and
#: nothing else -- anything else the generator raises is a defect, and a
#: skip would hide it where it happens.
UNSUPPORTED = (GenerationError, NotImplementedError)
