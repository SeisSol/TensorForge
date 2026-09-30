# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""What a kernel computes, established on the host.

Two halves, which a check puts side by side: `descriptors` evaluates a
descriptor list in NumPy with yateto's semantics, and `kernel_eval` executes
the generated kernel itself, statement by statement, for the lanes of one
multiplication.  The test suite and the tools under `tools/host` both use
them, so there is one statement of each.
"""
