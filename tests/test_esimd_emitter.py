# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""The ESIMD emitter turns a distribution into a type, and refuses to guess.

Tested on hand-built values rather than through the generator, because the
generator cannot reach the emitter yet: `LeadIndex.build` still constructs an
SPMD address, so every case stops at the lane index before a declaration is
ever written.  That is the next piece of work, and it is not a reason to leave
the piece that *is* written unverified -- the type mapping is the part the
whole lowering rests on, and it is decidable from a `Value` alone.
"""

from __future__ import annotations

import pytest

from tensorforge.backend.pir.core import (SCALAR_LAYOUT, IRError, LaneAxis,
                                          RegisterLayout, ScalarType, Value)
from tensorforge.backend.pir.emit_esimd import EsimdEmitter
from tensorforge.common.basic_types import Datatype
from tensorforge.common.context import Context

F32 = ScalarType(Datatype.F32)
F64 = ScalarType(Datatype.F64)
SPREAD16 = RegisterLayout((LaneAxis(16),))


@pytest.fixture
def emitter():
    ctx = Context(arch='pvc', backend='oneapi', fp_type=Datatype.F32)
    ctx.get_vm().get_lexic().simd_mode = True
    return EsimdEmitter(writer=None, context=ctx, strict=False)


def val(vid, type_=F32, layout=None):
    return Value(id=vid, type=type_, layout=layout)


# --------------------------------------------------------------------------
# distributed -> a vector whose width is lanes x slots
# --------------------------------------------------------------------------

def test_a_distributed_scalar_becomes_a_lane_wide_vector(emitter):
    assert 'simd<float, 16>' in emitter.ctype(F32, val(1, F32, SPREAD16))


def test_slots_multiply_the_lane_span(emitter):
    """`ScalarType.length` and `LaneAxis.block` are different axes.

    One lane holding four consecutive elements of a dimension spread over
    sixteen lanes is sixty-four elements in the register, and getting this
    product wrong is invisible in SPMD -- where the two are spelled by
    different mechanisms entirely -- and a wrong-sized register here.
    """
    v = val(2, ScalarType(Datatype.F32, 4), SPREAD16)
    assert 'simd<float, 64>' in emitter.ctype(v.type, v)


def test_multi_axis_layouts_multiply(emitter):
    lay = RegisterLayout((LaneAxis(4, 1), LaneAxis(4, 4)))
    v = val(3, F32, lay)
    assert 'simd<float, 16>' in emitter.ctype(v.type, v)


def test_the_element_type_is_carried_through(emitter):
    v = val(4, F64, SPREAD16)
    assert 'simd<double, 16>' in emitter.ctype(v.type, v)


# --------------------------------------------------------------------------
# replicated -> a scalar, and that is a derived answer
# --------------------------------------------------------------------------

def test_a_replicated_value_is_a_plain_scalar(emitter):
    assert emitter.ctype(F32, val(5, F32, SCALAR_LAYOUT)) == 'float'
    assert emitter.unresolved == []


# --------------------------------------------------------------------------
# untracked -> refused, and recorded
# --------------------------------------------------------------------------

def test_an_untracked_value_is_recorded_not_guessed(emitter):
    out = emitter.ctype(F32, val(6))
    assert emitter.unresolved == [val(6)] or len(emitter.unresolved) == 1
    # The placeholder must not be mistakable for a working declaration.
    assert 'untracked' in out


def test_untracked_and_replicated_do_not_collapse(emitter):
    """Both hold one value per lane; only one of them is *known* to.

    In SPMD they are spelled identically and nothing noticed the difference.
    Here the replicated case is an answer and the untracked case is a hole,
    and a lowering that treats them alike writes `float` where a vector
    belongs -- which compiles, runs, and is wrong.
    """
    replicated = emitter.ctype(F32, val(7, F32, SCALAR_LAYOUT))
    untracked = emitter.ctype(F32, val(8))
    assert replicated != untracked
    assert len(emitter.unresolved) == 1


def test_strict_mode_raises_at_the_end_of_a_body():
    ctx = Context(arch='pvc', backend='oneapi', fp_type=Datatype.F32)
    ctx.get_vm().get_lexic().simd_mode = True
    em = EsimdEmitter(writer=None, context=ctx, strict=True)
    em.ctype(F32, val(9))
    with pytest.raises(IRError, match='no tracked distribution'):
        em.run(())


# --------------------------------------------------------------------------
# there is no lane index
# --------------------------------------------------------------------------

def test_asking_for_a_lane_index_is_an_error_not_a_substitution(emitter):
    """One work-item *is* the vector.

    `item.get_local_id(0)` is the work-item's place in the ND-range, and the
    previous ESIMD attempt used it as a lane -- indexing a vector with a
    work-group coordinate.  Refusing here is what turns that from a silent
    wrong answer into the message that names the next piece of work.
    """
    with pytest.raises(IRError, match='no lane index'):
        emitter._thread_idx('x')


def test_the_other_axes_still_answer(emitter):
    """`y` and `z` are work-group coordinates in both models: which element
    this work-item handles, not which lane of it."""
    assert emitter._thread_idx('y')


# --------------------------------------------------------------------------
# memory: a distributed value moves by transfer, not by initialiser
# --------------------------------------------------------------------------

def test_a_subscript_becomes_a_pointer(emitter):
    """`copy_from` takes the address of the first element.

    Rewritten from the subscript the base emitter already built, rather than
    asking `Op.LOAD` for a second form of the same address -- two builders of
    one expression drift, and this one is not simple (`DataView.get_address`
    folds the shape in).
    """
    assert emitter._as_pointer('glb_m0[i + 16 * j]') == 'glb_m0 + (i + 16 * j)'
    assert emitter._as_pointer('x') == '&x'
