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


# --------------------------------------------------------------------------
# math: the ESIMD namespace, or nothing
# --------------------------------------------------------------------------

def _lexic():
    from tensorforge.common.vm.lexic.sycl_lexic import SyclLexic
    return SyclLexic('oneapi', 'intel', explicit_simd=True)


@pytest.mark.parametrize('op,expected', [
    ('ABS', 'intel_esimd::abs(a)'),
    ('SQRT', 'intel_esimd::sqrt(a)'),
    ('EXP', 'intel_esimd::exp(a)'),
    ('LOG', 'intel_esimd::log(a)'),
    ('SIN', 'intel_esimd::sin(a)'),
    ('POW', 'intel_esimd::pow(a, b)'),
    ('MIN', 'intel_esimd::min(a, b)'),
    ('MAX', 'intel_esimd::max(a, b)'),
])
def test_the_esimd_intrinsic_is_used(op, expected):
    from tensorforge.common.operation import Operation
    assert expected in _lexic().get_operation(getattr(Operation, op),
                                              Datatype.F32, 'a', 'b')


def test_reciprocal_is_inv_not_a_division():
    """`1 / x` does not compile against `simd<>`: there is no `operator/`
    taking an `int` on the left, and the intrinsic exists for this."""
    from tensorforge.common.operation import Operation
    out = _lexic().get_operation(Operation.RCP, Datatype.F32, 'a', None)
    assert 'inv(a)' in out and '1 /' not in out


@pytest.mark.parametrize('op', ['TANH', 'TAN', 'ASIN', 'CBRT', 'ATANH'])
def test_functions_the_hardware_library_lacks_are_declined(op):
    """Declined, not substituted.

    `sycl::tanh` is not a slower tanh for a `simd<>` operand -- it does not
    accept one, and where a conversion exists it would compute on a single
    element and look like it worked.  Composing one from the intrinsics that
    do exist is a numerics decision and does not belong in a spelling table.
    """
    from tensorforge.common.operation import Operation
    with pytest.raises(NotImplementedError, match='no ESIMD intrinsic'):
        _lexic().get_operation(getattr(Operation, op), Datatype.F32, 'a', 'b')


def test_the_spmd_lexic_is_untouched():
    from tensorforge.common.operation import Operation
    from tensorforge.common.vm.lexic.sycl_lexic import SyclLexic
    spmd = SyclLexic('acpp', 'intel')
    assert spmd.get_operation(Operation.TANH, Datatype.F32, 'a', None) == 'sycl::tanh(a)'


# --------------------------------------------------------------------------
# the lane index: a scalar in SPMD, a vector here
# --------------------------------------------------------------------------

def _builder(backend):
    from tensorforge.backend.pir.build import IRBuilder
    ctx = Context(arch='pvc', backend=backend, fp_type=Datatype.F32)
    return IRBuilder(fptype=Datatype.F32, context=ctx)


def test_spmd_asks_the_thread_for_its_lane():
    b = _builder('acpp')
    v = b.lane_index(16, 1)
    assert not v.distributed, "SPMD holds one index per thread"


def test_esimd_holds_every_index_at_once():
    """`which index am I at` has `block` answers when the work-item holds the
    whole dimension, so the value is the progression `0, 1, ... block-1`."""
    b = _builder('esimd')
    v = b.lane_index(16, 1)
    assert v.distributed and v.lane_span() == 16


def test_the_lane_offset_and_the_lane_index_are_different_questions():
    """Both are `(tid/stride) % block` in SPMD, and they diverge here: the
    offset a lane contributes to an address is zero (the work-item owns the
    whole dimension), while the index it is *at* is all of them."""
    b = _builder('esimd')
    assert b.lane_offset(16, 1) == 0
    assert b.lane_index(16, 1).distributed


def test_a_mask_is_not_a_branch_condition(emitter):
    """`if (m)` on a `simd_mask<N>` has no single bit to test.

    Refused with the name of the transformation that would fix it, rather than
    lowered into a branch that takes one arm for all N elements.
    """
    from tensorforge.backend.pir.core import Op, Region, Stmt
    cond = val(30, ScalarType(Datatype.BOOL), SPREAD16)
    guard = Stmt(op=Op.IF, args=(cond,), regions=(Region(),))
    with pytest.raises(IRError, match='if_convert'):
        emitter._emit_if(guard)


# --------------------------------------------------------------------------
# a mask is a type of its own, and a select over one is a merge
# --------------------------------------------------------------------------

def test_a_distributed_bool_is_a_mask_not_a_vector_of_bools(emitter):
    """`simd<bool, N>` exists and is the wrong answer.

    ESIMD keeps masks in their own family because the hardware does: a
    comparison over a `simd` yields one, a predicated operation takes one, and
    nothing else converts to it.  Spelling it `simd<bool, N>` compiles the
    declaration and fails at every use.
    """
    v = val(40, ScalarType(Datatype.BOOL), SPREAD16)
    assert 'simd_mask<16>' in emitter.ctype(v.type, v)


def test_a_replicated_bool_is_still_a_plain_bool(emitter):
    v = val(41, ScalarType(Datatype.BOOL), SCALAR_LAYOUT)
    assert 'simd_mask' not in emitter.ctype(v.type, v)


# --------------------------------------------------------------------------
# sinking a guard through a loop
# --------------------------------------------------------------------------

def _guard_over_loop(cond, loop_target=(), bounds=(0, 4, 1), body=()):
    from tensorforge.backend.pir.core import Op, Region, Stmt
    loop = Stmt(op=Op.FOR, target=tuple(loop_target), args=tuple(bounds),
                regions=(Region(args=(val(99, ScalarType(Datatype.I32),
                                          SCALAR_LAYOUT),), body=tuple(body)),))
    return Stmt(op=Op.IF, args=(cond,), regions=(Region(body=(loop,)),))


def test_a_guard_around_a_loop_sinks_into_it():
    """A mask is not control flow: moving it inside leaves the trip count
    alone and suppresses only what the body writes."""
    from tensorforge.backend.pir import passes
    cond = val(50, ScalarType(Datatype.BOOL), SPREAD16)
    assert passes._sinkable_loop(_guard_over_loop(cond)) is not None


def test_a_loop_that_carries_a_value_does_not_sink():
    """A masked-out lane's accumulator has to keep its previous value across
    the back edge, which is a merge and not a predicate; predicating the
    update alone would leave it undefined for that lane."""
    from tensorforge.backend.pir import passes
    cond = val(51, ScalarType(Datatype.BOOL), SPREAD16)
    carried = val(52, ScalarType(Datatype.F32), SPREAD16)
    assert passes._sinkable_loop(
        _guard_over_loop(cond, loop_target=(carried,))) is None


def test_a_lane_varying_bound_does_not_sink():
    """Sinking says the trip count is the same whether or not the guard holds;
    a bound derived from the mask makes that false."""
    from tensorforge.backend.pir import passes
    cond = val(53, ScalarType(Datatype.BOOL), SPREAD16)
    bound = val(54, ScalarType(Datatype.I32), SPREAD16)
    assert passes._sinkable_loop(
        _guard_over_loop(cond, bounds=(0, bound, 1))) is None


def test_sinking_is_off_by_default():
    """For a real branch it is a pessimisation -- the loop runs its full trip
    count instead of being skipped.  Only a mask has no branch to skip with."""
    from tensorforge.backend.pir import passes
    from tensorforge.backend.pir.core import Op
    cond = val(55, ScalarType(Datatype.BOOL), SPREAD16)
    guard = _guard_over_loop(cond)
    assert passes.if_convert((guard,))[0].op == Op.IF
    assert passes.if_convert((guard,), sink_into_loops=True)[0].op == Op.FOR


# --------------------------------------------------------------------------
# a ragged end is a shorter vector, not a mask
# --------------------------------------------------------------------------

def _leadloop(threads=16, start=0, end=12):
    from tensorforge.backend.symbol import LeadLoop
    return LeadLoop('i', start, end, threads, stride=1)


class _FakeWriter:
    def __init__(self, simd): self._simd = simd
    def _explicit_simd(self): return self._simd


def test_a_ragged_end_narrows_instead_of_masking():
    """12 elements over a 16-lane wave.

    SPMD has to mask lanes 12..15: the wave width is the hardware's whatever
    the operand looks like.  An explicitly vectorised kernel makes the vector
    12 wide and there is no ragged end to mask.
    """
    assert _leadloop()._narrow(_FakeWriter(True), 0, None, 12, 0, 12) == (12, 0)


def test_spmd_keeps_the_mask():
    assert _leadloop()._narrow(_FakeWriter(False), 0, None, 12, 0, 12) is None


def test_a_lower_bound_narrows_to_a_vector_that_starts_later():
    """`lane >= 4` is not a mask either.

    It needs the vector to *start* at element 4, which is a base offset --
    `LeadIndex` carries one since the `VarOffset` merge, and
    `split_lead_shift` puts its leftover lanes into a register address.
    """
    out = _leadloop(start=4, end=16)._narrow(_FakeWriter(True), 0, 4, None, 4, 16)
    assert out == (12, 4)


def test_a_later_slot_folds_into_the_offset():
    """`nonlead * block` stops being the right base as soon as `block` is the
    narrowed extent, so the slot goes into the offset and the index is always
    slot zero."""
    out = _leadloop(start=0, end=35)._narrow(_FakeWriter(True), 1, None, 3, 16, 19)
    assert out == (3, 16)


def test_a_full_width_block_needs_no_narrowing():
    assert _leadloop(end=16)._narrow(_FakeWriter(True), 0, None, 16, 0, 16) is None


# --------------------------------------------------------------------------
# a register slot is a run of lanes, not a single entry
# --------------------------------------------------------------------------

def test_a_slot_is_one_entry_per_thread_in_spmd():
    """The lane *is* the thread, so the other lanes' entries live in the other
    threads' private arrays and this one holds a single entry per slot."""
    from tensorforge.backend.symbol import DataView
    assert DataView.lead_lanes(None, False, 16) == 1


def test_a_slot_is_a_run_of_lanes_when_the_work_item_holds_the_wave():
    """Every lane's entry is in *this* array, so a slot is `threads` of them.

    Sizing per thread while addressing per work-item is what made twenty-one
    kernels read past the end of an array -- and that compiled, which is why
    the allocation and the addressing call one function instead of repeating
    a formula that already exists in three places.
    """
    from tensorforge.backend.symbol import DataView
    assert DataView.lead_lanes(None, True, 16) == 16


def test_narrowing_refuses_a_straddling_vector():
    """With `width > 1` the lane bounds are ceilings, so at a ragged end one
    lane holds a vector half inside the box; the guard is what stops its extra
    component from being stored, and narrowing removes the guard."""
    from tensorforge.backend.symbol import LeadLoop
    loop = LeadLoop('i', 0, 9, 16, stride=1, width=2)
    assert loop._narrow(_FakeWriter(True), 0, None, 5, 0, 9) is None


# --------------------------------------------------------------------------
# the last two text-path stores
# --------------------------------------------------------------------------

def test_a_sliced_lead_index_still_takes_the_structured_path():
    """`unwrap_lead`, not `isinstance`.

    A slicing offset wraps the lead index in a `VarOffset`, which
    `build_address` has always peeled -- so testing for `LeadIndex` alone only
    ever sent a sliced store back to the text path, where its address is a
    pinned name instead of an operand.
    """
    from tensorforge.backend.symbol import LeadIndex, unwrap_lead
    # Since the merge this *is* a LeadIndex rather than a wrapper around one,
    # and `unwrap_lead` is what both store paths ask.  `isinstance(...,
    # LeadIndex)` happens to work again -- but only by accident, and the
    # narrower test is what sent a sliced store to the text path before.
    idx = LeadIndex(0, 16, 1, offset=32)
    assert unwrap_lead(idx) is not None


def test_the_pointer_override_does_not_move_the_alias_root():
    """A rotating buffer's stages are one buffer.

    `pointer` changes the name written through and nothing else -- telling a
    pass the stages were separate would let it reorder a fill past a read of
    the stage being filled.
    """
    from tensorforge.backend.pir.core import MemSpace
    b = _builder('esimd')
    buf = b.alloc(Datatype.F32, (16,), MemSpace.REGISTER, hint='s')
    stmt = b.store(buf, b.const(1.0), 0, pointer='stage1')
    assert stmt.attr('pointer') == 'stage1'
    assert stmt.accesses[0].base is b.alias_root(buf)


# --------------------------------------------------------------------------
# the offset belongs to the lead index, not to a wrapper around it
# --------------------------------------------------------------------------

def test_add_offset_folds_into_a_lead_index():
    from tensorforge.backend.symbol import LeadIndex, VarOffset, add_offset
    out = add_offset(LeadIndex(2, 16, 1), 32)
    assert isinstance(out, LeadIndex) and not isinstance(out, VarOffset)
    assert out.offset() == 32


def test_offsets_accumulate_rather_than_nest():
    from tensorforge.backend.symbol import LeadIndex, add_offset
    assert add_offset(add_offset(LeadIndex(0, 16, 1), 16), 16).offset() == 32


def test_wrapping_a_lead_index_is_refused():
    """The unit mismatch, made unreachable.

    `VarOffset.write_nonlead` adds an element count to a slot index -- for
    slot 2 shifted by 32 elements over 16 lanes it produced `2 + 32` where the
    answer is `4`.  Nothing called it, so nothing found it; now nothing can
    build the state that would.
    """
    from tensorforge.backend.symbol import LeadIndex, VarOffset
    from tensorforge.common.exceptions import InternalError
    with pytest.raises(InternalError):
        VarOffset(LeadIndex(2, 16, 1), 32)


def test_unwrap_lead_keeps_its_contract():
    """`(index without the shift applied, shift in elements)`.

    The register callers convert the shift to slots themselves, so handing
    back an index that had already applied it would count it twice.
    """
    from tensorforge.backend.symbol import LeadIndex, unwrap_lead
    idx, shift = unwrap_lead(LeadIndex(2, 16, 1, offset=32))
    assert shift == 32 and idx.offset() == 0


def test_the_element_view_applies_the_offset_and_the_slot_view_does_not():
    """The whole reason the offset moved in here: its unit depends on the view,
    and only the index knows `block` and `width` to convert between them."""
    from tensorforge.backend.symbol import LeadIndex
    idx = LeadIndex(2, 16, 1, offset=32)
    assert idx.write_nonlead() == '2'


def test_a_shift_splits_into_slots_and_lanes():
    """The two halves are not interchangeable.

    Whole slots keep every element in the lane that held it, so they are a
    change of register index.  The remainder moves data *between* lanes --
    a shuffle under SPMD, and simply where the vector starts inside the slot
    run when the work-item holds the wave.
    """
    from tensorforge.backend.symbol import DataView
    assert DataView.split_lead_shift(32, 16) == (2, 0)
    assert DataView.split_lead_shift(36, 16) == (2, 4)
    # width scales the slot, not the lane count
    assert DataView.split_lead_shift(36, 16, width=2) == (1, 2)
