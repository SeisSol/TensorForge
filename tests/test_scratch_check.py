# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""`scratch_scope` makes a claim; this is what holds it to account.

Sibling scopes share space, which is how `nvidia.matmul` fits three tiles into
192 elements rather than 320.  Nothing checked that the sharing was safe.

Deriving the packing instead was the first plan and it does not work.  A
liveness analysis over the body computes a single interval per buffer, and
`matmul`'s loops are unrolled, so the C tile's interval --- first touch in
iteration `i`'s epilogue to last touch in the final one --- covers the A and B
staging of every iteration in between.  All three interfere, 320 elements, and
the analysis is right about the question it was asked.  What makes the reuse
safe is that each burst overwrites the tile completely before reading it, and
no single store does that: 128 elements, 48 writes.  Proving the union covers
the buffer is an index-set analysis, and it would be a great deal of machinery
to re-derive something the emitter knew when it wrote the scopes.

So the claim stays where it is made and this checks it, the same arrangement
as the sparse loader's layout.  The check is necessary, not sufficient, and
`check_reuse` returns the opaque statements alongside the violations so that
"no violations" cannot be read as "safe" when it means "not asked".
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

from tensorforge.backend.pir import scratch_check as sc
from tensorforge.backend.pir.build import IRBuilder
from tensorforge.backend.pir.core import INDEX, MemSpace
from tensorforge.common.basic_types import Datatype
from tensorforge.common.context import Context
from tensorforge.generators.generator import Generator

CASES = Path(__file__).resolve().parent / "cases"


def _body(read_a_after_c: bool):
    b = IRBuilder(fptype=Datatype.F32, scratch=('tempShrMem', 192))
    i = b.rawexpr('threadIdx.x', type_=INDEX, hint='a')
    with b.scratch_scope():
        a = b.alloc(Datatype.F32, (128,), MemSpace.SHARED, hint='atile')
        bb = b.alloc(Datatype.F32, (64,), MemSpace.SHARED, hint='btile')
        b.store(a, b.rawexpr('1.0f', hint='v'), i)
        b.store(bb, b.rawexpr('2.0f', hint='v'), i)
        b.load(a, i, hint='d')
    with b.scratch_scope():
        c = b.alloc(Datatype.F32, (128,), MemSpace.SHARED, hint='ctile')
        b.store(c, b.rawexpr('3.0f', hint='v'), i)
        b.load(c, i, hint='d')
        if read_a_after_c:
            b.load(a, i, hint='d')
    return b.finish()


def test_sibling_scopes_that_do_not_outlive_each_other_are_accepted():
    violations, opaque = sc.check_reuse(_body(read_a_after_c=False))
    assert not violations
    assert not opaque


def test_a_read_across_a_reused_window_is_caught():
    """The error a mis-nested scope actually produces."""
    violations, opaque = sc.check_reuse(_body(read_a_after_c=True))
    assert len(violations) == 1
    v = violations[0]
    assert 'atile' in str(v.read_of) and 'ctile' in str(v.clobbered_by)
    assert v.clobbered_at < v.at


def test_a_rewrite_between_the_clobber_and_the_read_is_allowed():
    """Which is exactly what every iteration after the first does."""
    b = IRBuilder(fptype=Datatype.F32, scratch=('tempShrMem', 128))
    i = b.rawexpr('threadIdx.x', type_=INDEX, hint='a')
    with b.scratch_scope():
        a = b.alloc(Datatype.F32, (128,), MemSpace.SHARED, hint='atile')
        b.store(a, b.rawexpr('1.0f', hint='v'), i)
    with b.scratch_scope():
        c = b.alloc(Datatype.F32, (128,), MemSpace.SHARED, hint='ctile')
        b.store(c, b.rawexpr('2.0f', hint='v'), i)
    b.store(a, b.rawexpr('3.0f', hint='v'), i)      # a is rewritten
    b.load(a, i, hint='d')
    violations, _ = sc.check_reuse(b.finish())
    assert not violations


def test_windows_that_do_not_overlap_are_not_compared():
    b = IRBuilder(fptype=Datatype.F32, scratch=('tempShrMem', 256))
    i = b.rawexpr('threadIdx.x', type_=INDEX, hint='a')
    a = b.alloc(Datatype.F32, (128,), MemSpace.SHARED, hint='atile')
    c = b.alloc(Datatype.F32, (128,), MemSpace.SHARED, hint='ctile')
    b.store(a, b.rawexpr('1.0f', hint='v'), i)
    b.store(c, b.rawexpr('2.0f', hint='v'), i)
    b.load(a, i, hint='d')
    placed, unplaced = sc.windows(b.finish())
    assert placed and not unplaced
    violations, _ = sc.check_reuse(b.finish())
    assert not violations


def test_an_undeclared_statement_is_reported_not_ignored():
    """A body with an opaque statement has not been checked, and saying so is
    the difference between a result and a false reassurance."""
    b = IRBuilder(fptype=Datatype.F32, scratch=('tempShrMem', 192))
    i = b.rawexpr('threadIdx.x', type_=INDEX, hint='a')
    with b.scratch_scope():
        a = b.alloc(Datatype.F32, (128,), MemSpace.SHARED, hint='atile')
        b.store(a, b.rawexpr('1.0f', hint='v'), i)
    with b.scratch_scope():
        c = b.alloc(Datatype.F32, (128,), MemSpace.SHARED, hint='ctile')
        b.store(c, b.rawexpr('2.0f', hint='v'), i)
    b('someOpaqueThing();')                    # no accesses argument
    _, opaque = sc.check_reuse(b.finish())
    assert opaque


# --------------------------------------------------------------------------- #
# Over the corpus
# --------------------------------------------------------------------------- #

def _bodies(case: str, backend: str, arch: str):
    path = CASES / case
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    from tensorforge.backend.instructions.compute.primitives import nvidia
    collected = []
    original = IRBuilder.finish

    def finish(self):
        body = original(self)
        if getattr(self, '_scratch', None):
            collected.append(body)
        return body

    IRBuilder.finish = finish
    enabled = nvidia.ENABLED
    nvidia.ENABLED = True
    try:
        ctx = Context(arch=arch, backend=backend,
                      fp_type=getattr(mod, "DTYPE", None) or Datatype.F32)
        Generator(mod.descr_list(), ctx).generate()
    finally:
        IRBuilder.finish = original
        nvidia.ENABLED = enabled
    return collected


# How many statements in each case still refuse to say what they touch, so the
# packing check has to skip them.  A ratchet, not a target: these numbers may
# go down and must never go up.
#
# Back to zero, one commit after the loop pushed it to 4.
#
# Those four were the loop's own scaffolding, raw text that had simply never
# been inside a body before: two lookahead bindings and the two statements
# that computed `allowed`.  They are values now -- the bindings a chain of
# conditional expressions over the induction value, the guard one conditional
# expression instead of an assignment and a guarded overwrite.
#
# The way down, for whoever adds the next entry: 16 before `RegisterAlloc`
# allocated through the builder, 14 before the `glb_m*` bindings declared
# their read, 11 before the shared window became a value, 10 before the
# transfers became `copy.async`, 3 while the `__syncwarp()` calls were left,
# 0, then 4 when the batch loop moved into the IR and brought its scaffolding
# with it, then 0 again.
STILL_OPAQUE = {"rectangular.py": 0, "square_notrans.py": 0}


# `case` is auto-parametrized across the whole corpus by conftest.
@pytest.mark.parametrize("case_file", ["rectangular.py", "square_notrans.py"])
def test_the_generated_packing_is_consistent(case_file):
    """The point of all of it: `matmul` overlaps C onto A and gets away with
    it, and now that is a checked statement rather than a comment."""
    budget = STILL_OPAQUE[case_file]
    seen = 0
    for body in _bodies(case_file, "cuda", "sm_86"):
        violations, opaque = sc.check_reuse(body)
        assert not violations, "\n".join(str(v) for v in violations)
        seen += len(opaque)
    assert seen <= budget, (
        f"{seen} statements did not declare their accesses, up from {budget}: "
        f"something new is emitting raw text into a body that a pass has to "
        f"reason about")
    if os.environ.get("TF_IR_WIDE") is not None:
        # The counts are a property of the body boundaries, not of the
        # generator: with one body per macro instruction the scratch-carrying
        # body holds the matmul alone, so the `glb_m*` bindings and the `r0`
        # declaration that make up six of the sixteen are simply in other
        # bodies and never reach the check.  The upper bound above still
        # applies; the exact ratchet is asserted for the shipped
        # configuration only, so the override stays usable for bisecting.
        return
    assert seen == budget, (
        f"only {seen} opaque statements now, down from {budget} -- lower the "
        f"entry in STILL_OPAQUE so the ratchet holds")
