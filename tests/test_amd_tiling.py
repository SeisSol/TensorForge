# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""No output element is computed twice.

`matmul32` splits the N columns between two instruction paths: whole blocks of
four through MFMA, and whatever is left through DPP.  The handoff point used
to be recomputed as `(N // 4) * 4` instead of read off what the MFMA path had
actually covered.  Those agree only when the tail is 0 or 1 columns wide; at
`N % 4 >= 2` the MFMA path pads the last block and does the tail as well, and
both paths then computed the same elements.

The result stayed correct -- the stores overwrite rather than accumulate, so
whichever landed last won -- which is why nothing failed.  What it cost was a
full padded MFMA block of dead work per tail, at every one of the operator
widths where `N % 4` lands on 2 or 3.  N=18 is one; it is not exotic.

The property is asserted over a range of N rather than on the two corpus cases
that happened to hit it, because "no case in the corpus has N % 4 == 2" is a
fact about the corpus, not about the generator.
"""

from __future__ import annotations

import inspect
from collections import Counter

import pytest

from tensorforge.backend.instructions.compute.primitives import amd
from tensorforge.common.basic_types import Datatype


class _Recorder:
    """Stands in for the C store callback and records what it is asked to write.

    Wrapping at `matmul` and asking the stack which path is calling avoids
    wrapping twice: `matmul32` calls `matmuldpp` itself, so instrumenting both
    would log every store of the inner call a second time and manufacture the
    very duplicates this is looking for.
    """

    def __init__(self):
        self.stores = []

    def __call__(self, writer, var, i, j):
        frames = {f.function for f in inspect.stack()[:6]}
        self.stores.append(("mfma" if "write_matmul" in frames else "dpp", i, j))
        return True

    @property
    def duplicates(self):
        counts = Counter((i, j) for _, i, j in self.stores)
        return {k: v for k, v in counts.items() if v > 1}

    @property
    def covered(self):
        return {(i, j) for _, i, j in self.stores}


class _FakeCtx:
    """Just enough context for the architecture predicates."""

    class _Descr:
        def __init__(self, model):
            self.model = model

    class _VM:
        def __init__(self, model):
            self._d = _FakeCtx._Descr(model)

        def get_hw_descr(self):
            return self._d

    def __init__(self, model="gfx90a"):
        self._vm = _FakeCtx._VM(model)

    def get_vm(self):
        return self._vm


def _operand(writer, var, *idx):
    """A loader that always succeeds, returning something truthy and non-None."""
    return writer.declare(hint="op") if var is None else True


def _run(M, N, K, threads=32, arch="gfx90a"):
    from tensorforge.backend.pir.build import IRBuilder

    writer = IRBuilder(Datatype.F32)
    rec = _Recorder()
    amd.matmul(writer, rec, _operand, _operand, M, N, K, 0, threads,
               Datatype.F32, None, _FakeCtx(arch))
    return rec


@pytest.mark.parametrize("N", range(1, 25))
@pytest.mark.parametrize("M", [1, 2, 3])
def test_no_element_is_computed_twice(M, N):
    rec = _run(M, N, K=8)
    assert not rec.duplicates, (
        f"M={M} N={N} (N%4={N % 4}): {len(rec.duplicates)} elements computed "
        f"twice, e.g. {sorted(rec.duplicates)[:3]}")


@pytest.mark.parametrize("N", range(1, 25))
@pytest.mark.parametrize("M", [1, 2, 3])
def test_every_element_is_computed_once(M, N):
    """The other half: fixing an overlap by narrowing too far would leave a
    gap, and a missing column is a worse bug than a duplicated one."""
    rec = _run(M, N, K=8)
    expected = {(i, j) for i in range(M) for j in range(N)}
    assert rec.covered == expected, (
        f"M={M} N={N}: missing {sorted(expected - rec.covered)[:4]}, "
        f"extra {sorted(rec.covered - expected)[:4]}")


@pytest.mark.parametrize("N", [2, 3, 6, 7, 18, 19, 22, 23])
def test_the_tail_goes_to_one_path_only(N):
    """At `N % 4 >= 2` the tail belongs to the padded MFMA block, and the DPP
    path must not see it at all -- that split is the point of `cap4`."""
    rec = _run(M=2, N=N, K=8)
    tail = set(range((N // 4) * 4, N))
    dpp_cols = {j for path, _, j in rec.stores if path == "dpp"}
    mfma_cols = {j for path, _, j in rec.stores if path == "mfma"}
    assert not (dpp_cols & mfma_cols), \
        f"N={N}: columns handled by both paths: {sorted(dpp_cols & mfma_cols)}"
    assert tail <= mfma_cols, f"N={N}: tail {sorted(tail)} not in the MFMA path"


@pytest.mark.parametrize("N", [4, 8, 16, 20])
def test_exact_multiples_leave_nothing_for_the_dpp_path(N):
    rec = _run(M=2, N=N, K=8)
    assert not [s for s in rec.stores if s[0] == "dpp"], \
        f"N={N} divides by 4; the DPP path should have had nothing to do"
