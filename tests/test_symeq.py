# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""The equivalence checker has to be able to fail.

`harness.symeq` compares two generated kernels by building a symbolic
expression per output slot, treating every vendor intrinsic as an
uninterpreted function.  It is the tool that backs the claim "this refactor
changed how the arithmetic is written down, not what it computes" --- and a
comparison that always returns "equivalent" would back that claim equally
well while being worthless.

So these tests perturb a kernel in the two ways the AMD conversion could
plausibly have gone wrong and require the checker to notice:

* a link dropped from an MFMA accumulator chain --- what a mis-set `movable`
  or an over-eager DCE would produce;
* an operand reused from *before* an in-place `transpose4x4b32` --- what a
  CSE that did not see the mutation would produce.

The second is the one worth having: the transpose writes through references,
so the register it overwrites is still, textually, the result of a load that
an unguarded load-CSE would happily reuse.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from harness.symeq import compare, interpret, outputs

SNAPSHOT_DIR = Path(__file__).parent / "snapshots"

# Cases that exercise the MFMA path; skipped rather than failed when the
# snapshot corpus has not been generated yet.
CASES = [
    "gemm_square_16.hip.cpp",
    "gemm_56x18_x_18x18.hip.cpp",
    "chain_three_matrices.hip.cpp",
]

_MFMA_DECL = re.compile(
    r"([\w:<>, ]+?) (v\d+_acc) = (__builtin_amdgcn_mfma\w+)"
    r"\((\w+), (\w+), (\w+), ([^)]*)\);")
_TRANSPOSE = re.compile(r"tensorforge::transpose4x4b32\((\w+), (\w+),")


def _snapshot(name: str) -> Path:
    p = SNAPSHOT_DIR / name
    if not p.exists():
        pytest.skip(f"{name} not generated; run pytest --snapshot-update")
    return p


@pytest.fixture
def perturbed(tmp_path):
    def make(src_path: Path, text: str) -> Path:
        out = tmp_path / src_path.name
        out.write_text(text)
        return out
    return make


@pytest.mark.parametrize("name", CASES)
def test_kernel_is_equivalent_to_itself(name):
    p = _snapshot(name)
    assert compare(p, p) is None


@pytest.mark.parametrize("name", CASES)
def test_dropped_mfma_is_detected(name, perturbed):
    """Forward the accumulator past one MFMA, as a lost link would."""
    p = _snapshot(name)
    src = p.read_text()
    hits = list(_MFMA_DECL.finditer(src))
    if len(hits) < 3:
        pytest.skip("not enough MFMA links to drop one")
    h = hits[1]
    broken = f"{src[:h.start()]}{h.group(1)} {h.group(2)} = {h.group(6)};{src[h.end():]}"
    assert compare(p, perturbed(p, broken)) is not None, \
        "dropping an MFMA from the chain went unnoticed"


@pytest.mark.parametrize("name", CASES)
def test_stale_operand_across_transpose_is_detected(name, perturbed):
    """Reuse a register the transpose has since overwritten in place."""
    p = _snapshot(name)
    src = p.read_text()
    tr = _TRANSPOSE.search(src)
    if tr is None:
        pytest.skip("no in-place transpose in this kernel")
    after = src[tr.end():].replace(tr.group(2), tr.group(1), 1)
    assert compare(p, perturbed(p, src[:tr.end()] + after)) is not None, \
        "a stale pre-transpose operand went unnoticed"


@pytest.mark.parametrize("name", CASES)
def test_renaming_temporaries_is_not_a_difference(name, perturbed):
    """Equivalence is up to renaming: the allocator's numbering is not
    semantics, and a checker that flagged it would flag every refactor."""
    p = _snapshot(name)
    renamed = re.sub(r"\bv(\d+)_", lambda m: f"v{int(m.group(1)) + 1000}_",
                     p.read_text())
    assert compare(p, perturbed(p, renamed)) is None


def test_accumulator_chain_is_visible_to_the_checker():
    """Guards the checker's own blind spot.

    The pre-SSA kernels update the accumulator with a bare `v15 = mfma(...)`.
    A parser that recognises declarations and array stores but not scalar
    reassignment silently drops the entire chain --- and then every output
    slot compares equal as an untouched zero, which reads as success.
    """
    p = _snapshot(CASES[0])
    env = interpret(p)
    outs = outputs(env)
    assert outs, "no output slots recovered at all"
    flat = str(sorted(outs.items(), key=str))
    assert "mfma" in flat, "output slots do not depend on any MFMA"
    assert "zero" not in flat or "mfma" in flat
