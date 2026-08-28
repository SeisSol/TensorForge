# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""`cuda::pipeline` calls have to come in pairs, and nothing checks that.

The async path emits `producer_acquire`, `memcpy_async`, `producer_commit`,
`consumer_wait` and `consumer_release` as raw statements.  Raw statements carry
`Effect.UNKNOWN`, which is enough to stop a pass reordering them and not nearly
enough to say anything about whether they are *used* correctly: nothing in the
generator, the IR or the snapshot harness relates an acquire to the wait that
retires it.

libcu++ states the precondition plainly --- `consumer_wait()` requires a
committed stage.  Waiting on a pipeline nothing was committed to is undefined
behaviour, and it generates, renders and snapshots exactly like correct code.

`asyncmem.py` already checks the property this test approximates: every token
consumed exactly once, no wait naming a token that is not in flight.  It checks
it on `copy.async`/`wait`, which the corpus uses zero times.  So this counts
brackets in the generated text instead --- a poor substitute for a def-use edge,
and the reason to want the edge.

The two entries below are the state at the time of writing, not a target.  A
case that starts violating this is a regression; the two that stop are the
migration arriving.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from tensorforge.common.context import Context
from tensorforge.generators.generator import Generator

CASES = Path(__file__).resolve().parent / "cases"

#: Empty, and it should stay empty.
#:
#: Both entries came out on the commit that stopped `LoadWait` keying on
#: `_use_cuda_memcpy`.  That flag is a static choice, not a record: the
#: reordering path ignores it and moves the data with ordinary loads and
#: stores, so a wait fired for a transfer that never went in flight.  The rest
#: of the async path migrated to `copy.async` in the same commit, so the
#: brackets this counts are mostly gone as well -- which is why an entry
#: appearing here again would mean something specific: a `cuda::pipeline`
#: driven by hand, and unbalanced.
KNOWN_UNBALANCED = {}


def _kernels():
    for path in sorted(CASES.rglob("*.py")):
        if path.name.startswith("_"):
            continue
        spec = importlib.util.spec_from_file_location("tf_pipe__" + path.stem,
                                                      path)
        mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
        except Exception:
            continue
        if not hasattr(mod, "NAME") or not hasattr(mod, "descr_list"):
            continue
        try:
            ctx = Context(arch="sm_86", backend="cuda",
                          fp_type=getattr(mod, "DTYPE", None))
            gen = Generator(mod.descr_list(), ctx)
            gen.generate()
        except Exception:
            continue
        kernel = gen.get_kernel()
        if kernel:
            yield mod.NAME, kernel


def _counts(kernel: str):
    return (kernel.count("producer_acquire"), kernel.count("producer_commit"),
            kernel.count("consumer_wait"), kernel.count("consumer_release"))


def test_pipeline_brackets_balance():
    unbalanced = {}
    for name, kernel in _kernels():
        acquire, commit, wait, release = counts = _counts(kernel)
        if not (acquire or wait):
            continue
        if acquire == commit == wait == release:
            continue
        unbalanced[name] = counts

    new = {k: v for k, v in unbalanced.items() if k not in KNOWN_UNBALANCED}
    assert not new, (
        "these kernels acquire, commit, wait and release different numbers of "
        f"times: {new}. A `consumer_wait()` with no committed stage is "
        "undefined behaviour and generates like anything else")

    fixed = set(KNOWN_UNBALANCED) - set(unbalanced)
    assert not fixed, (
        f"{sorted(fixed)} balance now -- drop them from KNOWN_UNBALANCED")

    for name, counts in unbalanced.items():
        assert counts == KNOWN_UNBALANCED[name], (
            f"{name} was {KNOWN_UNBALANCED[name]}, is now {counts}")


def test_no_wait_without_a_commit_anywhere_new():
    """The sharper half: waiting on a pipeline never committed to."""
    offenders = {name: _counts(k)
                 for name, k in _kernels()
                 if _counts(k)[2] and not _counts(k)[1]}
    assert set(offenders) <= {"gemm_trans_b_12x16"}, (
        f"new kernels wait on an uncommitted pipeline: "
        f"{set(offenders) - {'gemm_trans_b_12x16'}}")
