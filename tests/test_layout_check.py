# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""The layout the allocator derives, held to account by the accesses alone.

The allocator puts two buffers on the same bytes where no statement occupies
both, and it learns where a value ends from the instruction that writes the
buffer whole (`mark defines`).  That is a claim the emitter makes; a wrong
one -- a mark in front of a write that does not define the whole buffer --
lets another buffer take bytes something still reads.  `pir.layout_check`
looks for exactly that error, without reading the marks.

The check is necessary, not sufficient, and `check_layout` returns the
statements that did not say what they touch alongside the violations, so
that "no violations" cannot be read as "safe" when it means "not asked".
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from tensorforge.backend.pir import layout_check as lc
from tensorforge.backend.pir.build import IRBuilder
from tensorforge.backend.pir.core import MemSpace
from tensorforge.common.basic_types import Datatype
from tensorforge.common.context import Context
from tensorforge.common.options import Options
from tensorforge.generators.generator import Generator

CASES = Path(__file__).resolve().parent / "cases"


def _body(read_a_after_c: bool):
    """`a` and `c` on the same bytes, `b` beside them -- placed by hand."""
    b = IRBuilder(fptype=Datatype.F32, arena='shrMem')
    i = b.thread_id('x')
    a = b.alloc(Datatype.F32, (128,), MemSpace.SHARED, hint='atile', offset=0)
    bb = b.alloc(Datatype.F32, (64,), MemSpace.SHARED, hint='btile',
                 offset=128)
    c = b.alloc(Datatype.F32, (128,), MemSpace.SHARED, hint='ctile', offset=0)
    b.store(a, b.const(1.0), i)
    b.store(bb, b.const(2.0), i)
    b.load(a, i, hint='d')
    b.store(c, b.const(3.0), i)
    b.load(c, i, hint='d')
    if read_a_after_c:
        b.load(a, i, hint='d')
    return b.finish()


def test_buffers_that_do_not_outlive_each_other_may_share_bytes():
    violations, opaque = lc.check_layout(_body(read_a_after_c=False))
    assert not violations
    assert not opaque


def test_a_read_across_another_buffer_s_write_is_caught():
    violations, opaque = lc.check_layout(_body(read_a_after_c=True))
    assert len(violations) == 1
    v = violations[0]
    assert 'atile' in v.read_of and 'ctile' in v.clobbered_by
    assert v.clobbered_at < v.at


def test_a_rewrite_between_the_clobber_and_the_read_is_allowed():
    """Which is what every burst after the first does."""
    b = IRBuilder(fptype=Datatype.F32, arena='shrMem')
    i = b.thread_id('x')
    a = b.alloc(Datatype.F32, (128,), MemSpace.SHARED, hint='atile', offset=0)
    c = b.alloc(Datatype.F32, (128,), MemSpace.SHARED, hint='ctile', offset=0)
    b.store(a, b.const(1.0), i)
    b.store(c, b.const(2.0), i)
    b.store(a, b.const(3.0), i)
    b.load(a, i, hint='d')
    violations, _ = lc.check_layout(b.finish())
    assert not violations


def test_a_mark_is_a_claim_and_not_a_rewrite():
    """The check has to stand apart from what it checks: a mark in front of
    nothing does not make the read of a clobbered buffer safe."""
    b = IRBuilder(fptype=Datatype.F32, arena='shrMem')
    i = b.thread_id('x')
    a = b.alloc(Datatype.F32, (128,), MemSpace.SHARED, hint='atile', offset=0)
    c = b.alloc(Datatype.F32, (128,), MemSpace.SHARED, hint='ctile', offset=0)
    b.store(a, b.const(1.0), i)
    b.store(c, b.const(2.0), i)
    b.mark('defines', a)
    b.load(a, i, hint='d')
    violations, _ = lc.check_layout(b.finish())
    assert len(violations) == 1


def test_windows_that_do_not_overlap_are_not_compared():
    b = IRBuilder(fptype=Datatype.F32, arena='shrMem')
    i = b.thread_id('x')
    a = b.alloc(Datatype.F32, (128,), MemSpace.SHARED, hint='atile', offset=0)
    c = b.alloc(Datatype.F32, (128,), MemSpace.SHARED, hint='ctile',
                offset=128)
    b.store(a, b.const(1.0), i)
    b.store(c, b.const(2.0), i)
    b.load(a, i, hint='d')
    placed, unplaced = lc.windows(b.finish())
    assert len(placed) == 2 and not unplaced
    violations, _ = lc.check_layout(b.finish())
    assert not violations


def test_two_windows_of_one_buffer_are_not_two_buffers():
    """A later writer goes through the window an earlier user declared; read
    through the second, the first one's write is no clobber."""
    b = IRBuilder(fptype=Datatype.F32, arena='shrMem')
    i = b.thread_id('x')
    owner = object()
    first = b.alloc(Datatype.F32, (64,), MemSpace.SHARED, hint='first',
                    offset=0, identity=owner)
    b.store(first, b.const(1.0), i)
    second = b.alloc(Datatype.F32, (64,), MemSpace.SHARED, hint='second',
                     offset=0, identity=owner)
    b.load(second, i, hint='d')
    placed, _ = lc.windows(b.finish())
    assert len(placed) == 1
    violations, _ = lc.check_layout(b.finish())
    assert not violations


def test_an_undeclared_statement_is_reported_not_ignored():
    """A body with an opaque statement has not been checked, and saying so is
    the difference between a result and a false reassurance."""
    b = IRBuilder(fptype=Datatype.F32, arena='shrMem')
    i = b.thread_id('x')
    a = b.alloc(Datatype.F32, (128,), MemSpace.SHARED, hint='atile', offset=0)
    c = b.alloc(Datatype.F32, (128,), MemSpace.SHARED, hint='ctile', offset=0)
    b.store(a, b.const(1.0), i)
    b.store(c, b.const(2.0), i)
    b('someOpaqueThing();')                    # no accesses argument
    _, opaque = lc.check_layout(b.finish())
    assert opaque


# --------------------------------------------------------------------------- #
# Over the corpus
# --------------------------------------------------------------------------- #

def _bodies(case: Path, backend: str, arch: str, **options):
    spec = importlib.util.spec_from_file_location(case.stem, case)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ctx = Context(arch=arch, backend=backend,
                  fp_type=getattr(mod, "DTYPE", None) or Datatype.F32,
                  options=Options(**options))
    gen = Generator(mod.descr_list(), ctx)
    gen.generate()
    return [section.body for section in gen.built._sections]


# How many statements in each case refuse to say what they touch, so the
# layout check has to skip them.  A ratchet, not a target: these numbers may
# go down and must never go up.
STILL_OPAQUE = {"rectangular.py": 0, "square_notrans.py": 0,
                "local_flux.py": 0, "accumulate_then_read.py": 0,
                "temp_slice_after_whole.py": 0}


@pytest.mark.parametrize("tensor_cores", [False, True])
@pytest.mark.parametrize("case_file", sorted(STILL_OPAQUE))
def test_the_generated_layout_is_consistent(case_file, tensor_cores):
    """The point of all of it: `matmul` puts `C` over `A` and `B`, and the
    allocator overlaps a whole kernel's buffers by their lifetimes, and both
    are checked statements rather than comments."""
    budget = STILL_OPAQUE[case_file]
    seen = 0
    case = next(CASES.rglob(case_file))
    for body in _bodies(case, "cuda", "sm_86", tensor_cores=tensor_cores):
        violations, opaque = lc.check_layout(body)
        assert not violations, "\n".join(str(v) for v in violations)
        seen += len(opaque)
    assert seen <= budget, (
        f"{seen} statements did not declare their accesses, up from {budget}: "
        f"something new is emitting raw text into a body that a pass has to "
        f"reason about")
    assert seen == budget, (
        f"only {seen} opaque statements now, down from {budget} -- lower the "
        f"entry in STILL_OPAQUE so the ratchet holds")
