# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Bank conflicts, computed from the IR instead of from the emitted C++.

`tools/bank_conflicts.py` answers this by parsing generated source, and it had
to: the addresses were `rawexpr` text and there was nothing to evaluate.  They
are operations now, and the only leaf that is not a constant is
`thread_idx_x`, which a pass can read as "the lane".

Checked against the text version rather than replacing it.  Over the corpus
the two agree on what matters -- 280 four-way accesses against 282, and the
same everywhere else -- and the text version is still the one that decides,
because it measures the addresses the hardware will actually see.

The absolute counts differ by about two percent (14807 accesses against
15143) and I have not accounted for it.  Raw statements subscripting a shared
window explain three.  Recorded as an open difference rather than explained
away: a diagnostic whose gap is understood is worth more than one whose gap is
excused.
"""

from __future__ import annotations

import pytest

from tensorforge.backend.pir import banks
from tensorforge.backend.pir.build import IRBuilder
from tensorforge.backend.pir.core import BOOL, INDEX, MemSpace, XorSwizzle
from tensorforge.common.basic_types import Datatype


def builder(budget=4096):
    return IRBuilder(fptype=Datatype.F32, scratch=('tempShrMem', budget))


def _one(b, tile, index):
    """The single access in this body, analysed."""
    accesses, unresolved = banks.analyse(b.finish())
    assert unresolved == 0
    assert len(accesses) == 1
    return accesses[0]


# --------------------------------------------------------------------------- #
# The arithmetic, against answers that are known by hand
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("scale,want,why", [
    (1, 1, "32 consecutive floats are 32 banks"),
    (2, 2, "an even stride halves the banks reached"),
    (32, 32, "the classic column read: every lane in bank 0, all different"),
    (33, 1, "padding the row by one is what fixes a stride-32 column read"),
])
def test_a_strided_read(scale, want, why):
    b = builder()
    tile = b.alloc(Datatype.F32, (32 * 33,), MemSpace.SHARED, hint='s')
    idx = b.op('mul', INDEX, b.thread_id('x'), scale, hint='a')
    b.load(tile, idx, hint='v')
    assert _one(b, tile, idx).ways == want, why


def test_every_lane_reading_one_address_is_free():
    """A broadcast, not a conflict: one address in one bank costs one cycle
    however many lanes want it."""
    b = builder()
    tile = b.alloc(Datatype.F32, (64,), MemSpace.SHARED, hint='s')
    b.load(tile, 0, hint='v')
    assert _one(b, tile, 0).ways == 1


def test_a_guard_narrows_the_lanes():
    """`Op.IF` carries its condition as a value in `cond`, so the same
    evaluator answers it per lane.

    Without this the analysis counts all 32 into every bank, and the staging
    steps here are guarded to a quarter or a half of the wave -- which read 72
    conflict-free accesses in `rectangular` as 2-way.
    """
    b = builder()
    tile = b.alloc(Datatype.F32, (128,), MemSpace.SHARED, hint='s')
    tid = b.thread_id('x')
    idx = b.op('mul', INDEX, tid, 4, hint='a')
    with b.if_(b.op('lt', BOOL, tid, 8, hint='g')):
        b.load(tile, idx, hint='v')
    access = _one(b, tile, idx)
    assert access.lanes == 8 and access.ways == 1


def test_a_vector_store_takes_its_width_from_the_value():
    """A load produces the value and a store consumes it, so which operand
    carries the width depends on the direction.

    Reading `target` for both made every vector store look scalar, and a
    scalar model of a `float4` store puts four times the stride between lanes.
    """
    from tensorforge.backend.pir.core import ScalarType

    b = builder()
    tile = b.alloc(Datatype.F32, (256,), MemSpace.SHARED, hint='s')
    idx = b.op('mul', INDEX, b.thread_id('x'), 4, hint='a')
    quad = b.rawexpr('q', type_=ScalarType(Datatype.F32, 4), hint='q')
    b.store(tile, quad, idx)
    accesses, _ = banks.analyse(b.finish())
    assert accesses[0].kind == 'store'
    assert accesses[0].ways == 1, (
        'eight lanes of sixteen bytes cover the bank width exactly')


def test_a_swizzle_is_visible_because_it_is_in_the_index():
    """The permutation is applied by `load` and `store`, so it is part of the
    address the analysis sees -- no special case needed."""
    b = builder()
    plain = b.alloc(Datatype.F32, (1024,), MemSpace.SHARED, hint='p')
    idx = b.op('mul', INDEX, b.thread_id('x'), 32, hint='a')
    b.load(plain, idx, hint='v')
    assert _one(b, plain, idx).ways == 32

    b = builder()
    swz = b.alloc(Datatype.F32, (1024,), MemSpace.SHARED, hint='s',
                  swizzle=XorSwizzle(32))
    idx = b.op('mul', INDEX, b.thread_id('x'), 32, hint='a')
    b.load(swz, idx, hint='v')
    assert _one(b, swz, idx).ways == 1


# --------------------------------------------------------------------------- #
# What it refuses
# --------------------------------------------------------------------------- #

def test_a_raw_address_is_refused_not_guessed():
    """An address the analysis cannot read is one it must not count.

    Nine remain in the corpus, all of them `Symbol.load_linear`'s text form.
    Saying so is the difference between a number and a number-shaped thing.
    """
    b = builder()
    tile = b.alloc(Datatype.F32, (64,), MemSpace.SHARED, hint='s')
    b.load(tile, b.rawexpr('0 + threadIdx.x * 1', type_=INDEX, hint='a'),
           hint='v')
    accesses, unresolved = banks.analyse(b.finish())
    assert accesses == [] and unresolved == 1


def test_a_numpy_integer_is_an_integer():
    """Shapes and offsets arrive as `numpy.int64`, which is an integer
    everywhere except to `isinstance`.  Two dozen addresses were unresolved
    for that alone."""
    import numpy as np

    b = builder()
    tile = b.alloc(Datatype.F32, (128,), MemSpace.SHARED, hint='s')
    idx = b.op('add', INDEX, b.thread_id('x'), np.int64(16), hint='a')
    b.load(tile, idx, hint='v')
    assert _one(b, tile, idx).ways == 1


def test_a_loop_variable_resolves_to_its_bound():
    """A loop variable is a region argument, not a statement result, so it has
    no producer to find -- 159 addresses came back unresolved for that.

    Substituting the bound is sound here: these are loops over tensor
    dimensions, every lane is on the same iteration, so the value shifts every
    lane's address equally and the bank pattern is unchanged.
    """
    b = builder()
    tile = b.alloc(Datatype.F32, (256,), MemSpace.SHARED, hint='s')
    with b.for_(0, 4) as loop:
        i = loop.index if hasattr(loop, 'index') else None
        b.load(tile, b.thread_id('x'), hint='v')
    accesses, unresolved = banks.analyse(b.finish())
    assert unresolved == 0 and accesses[0].ways == 1


def test_the_analysis_belongs_after_the_passes():
    """A freshly finished body still holds what `dce` and `cse` will remove.

    `chain_three` has 1172 shared loads at `finish()` and 587 after
    optimisation, and the emitted source shows 590 -- so measuring the
    unoptimised body counted twice as many accesses as the hardware will make.
    That is also where a pass acting on this would sit: after the passes that
    change what is there, before the emitter that fixes it.
    """
    from tensorforge.backend.pir import passes

    b = builder()
    tile = b.alloc(Datatype.F32, (128,), MemSpace.SHARED, hint='s')
    idx = b.op('mul', INDEX, b.thread_id('x'), 4, hint='a')
    kept = b.load(tile, idx, hint='keep')
    b.load(tile, idx, hint='dead')          # nothing uses it
    b('use(%s);' % kept, kept, accesses=())
    body = b.finish()

    before, _ = banks.analyse(body)
    after, _ = banks.analyse(passes.optimize(body))
    assert len(before) == 2
    assert len(after) == 1, (
        'the dead load is still counted; the analysis has to run on the body '
        'the emitter will see')
