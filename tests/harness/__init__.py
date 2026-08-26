# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Numerical end-to-end test harness for TensorForge kernels.

Orchestrates the pipeline:
    case.descr_list()  ->  Generator  ->  emit main.cu  ->  compile  ->  run  ->  compare

Public entrypoints live in :mod:`runner`; pytest wiring lives in the
top-level ``conftest.py``.
"""
