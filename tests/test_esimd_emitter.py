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


# --------------------------------------------------------------------------
# register pressure in bytes
# --------------------------------------------------------------------------

def test_a_vector_costs_its_whole_width():
    """Counting values says one; counting registers says sixteen.

    The difference is the wave width, and it is in the direction that matters:
    under SPMD a value is one register per thread, so a count *is* a register
    count.  When the work-item holds the whole wave it understates by 16 on
    PVC.
    """
    from tensorforge.backend.pir.passes import register_bytes
    assert register_bytes(val(60, F32, SCALAR_LAYOUT)) == 4
    assert register_bytes(val(61, F32, SPREAD16)) == 64


def test_the_slot_axis_multiplies_the_lane_axis():
    """`ScalarType.length` and the layout are different things everywhere else
    and still are here -- they just both make the register bigger."""
    from tensorforge.backend.pir.passes import register_bytes
    v = val(62, ScalarType(Datatype.F32, 4), SPREAD16)
    assert register_bytes(v) == 16 * 4 * 4


def test_an_untracked_layout_counts_as_one_lane():
    """A floor, not an estimate: it is what SPMD would need, so the number
    never overstates and a budget comparison stays on the safe side."""
    from tensorforge.backend.pir.passes import register_bytes
    assert register_bytes(val(63, F32)) == 4


def test_a_register_allocation_is_register_file():
    """And on this path it is the biggest thing in it.

    `lead_window_spans_two_blocks` peaks at 540 bytes of SSA values beside a
    `float r0[4992]` -- 19 KB.  Counting only the values reported that kernel
    as comfortable.
    """
    from tensorforge.backend.pir.core import BufferType, MemSpace
    from tensorforge.backend.pir.passes import register_bytes
    from tensorforge.backend.pir.core import Value as V
    reg = V(id=64, type=BufferType(Datatype.F32, (128,), MemSpace.REGISTER))
    shared = V(id=65, type=BufferType(Datatype.F32, (128,), MemSpace.SHARED))
    assert register_bytes(reg) == 128 * 4
    assert register_bytes(shared) == 0, 'shared memory costs no registers'


def test_a_replicated_vector_is_still_a_simd(emitter):
    """Its width comes from the slot axis, not the lane axis -- and it is a
    `simd` either way.

    A `sycl::vec` has no `select`, no `copy_from` and nothing a fragment is
    written through, so spelling a replicated vector that way compiles the
    declaration and fails at every use.  Under this lowering every vector is a
    `simd`; what differs is only where the width came from.
    """
    v = val(70, ScalarType(Datatype.F32, 128), SCALAR_LAYOUT)
    out = emitter.ctype(v.type, v)
    assert 'simd<float, 128>' in out and 'vec' not in out


def test_dpas_emission_declines_a_wave_it_cannot_use():
    """`ExecutionSize` is 16 for every type in the table, and under this
    lowering the vector width *is* the thread count -- so another width is a
    different instruction, not a narrower use of this one."""
    from tensorforge.backend.instructions.compute.primitives import intel
    assert intel.dpas_matmul(None, None, None, None, 8, 16, 8, 0, 32,
                             Datatype.F32, None) is False


def test_dpas_emission_declines_a_type_without_an_atom():
    """FP64 has no DPAS at all; the caller falls through to the generic path."""
    from tensorforge.backend.instructions.compute.primitives import intel
    assert intel.dpas_matmul(None, None, None, None, 8, 16, 8, 0, 16,
                             Datatype.F64, None) is False


# --------------------------------------------------------------------------
# a multiplication is one work-item, so it has nobody to wait for
# --------------------------------------------------------------------------

def _sync(threads, backend):
    from tensorforge.backend.instructions.sync_block import SyncThreads
    ctx = Context(arch='pvc', backend=backend, fp_type=Datatype.F32)
    return SyncThreads(ctx, threads).barrier_scope()


def test_a_wide_multiplication_needs_a_group_barrier_in_spmd():
    """PVC's sub-group is 16 wide, so 32 threads span two of them.

    SPMD has no spelling for a rendezvous narrower than the sub-group, so the
    barrier is met at the group -- and the thread-block policy sizes the block
    to hold one group, which is what makes it legal inside a `BatchLoop`.
    """
    from tensorforge.backend.pir.core import Uniformity
    assert _sync(16, 'acpp') is Uniformity.MULT, (
        "a 16-thread multiplication is the sub-group, so the sub-group "
        "barrier meets exactly it")
    assert _sync(32, 'acpp') is Uniformity.MULTGROUP


def test_an_explicit_vector_has_nobody_to_wait_for():
    """The wave is not a hardware sub-group the multiplication must fit inside
    -- it *is* the work-item, and `num_threads` is the length of its
    registers.  A 32-thread multiplication is a 32-wide vector executed in
    order by one work-item.

    This is the structural difference the path was chosen for, and it is what
    the sub-group-16 finding from the very first review turns into.
    """
    from tensorforge.backend.pir.core import Uniformity
    assert _sync(16, 'esimd') is Uniformity.MULT
    assert _sync(32, 'esimd') is Uniformity.MULT
    assert _sync(64, 'esimd') is Uniformity.MULT


# --------------------------------------------------------------------------
# the register budget
# --------------------------------------------------------------------------

def test_a_budget_is_stated_where_it_is_known_and_absent_elsewhere():
    """Absent means "not stated", not "unlimited".

    A default would quietly pass everything, which is the failure mode a
    budget check exists to avoid.  So where a budget is stated it is the
    documented figure -- 128 GRF of 64 bytes on PVC, 255 registers on NVIDIA,
    LLVM's addressable VGPRs on AMD (the unified file from gfx90a on) -- and a
    target nobody looked up still has none.
    """
    from tensorforge.common.vm.hw_descr import hw_descr_factory
    assert hw_descr_factory('pvc', 'oneapi').max_reg_per_thread == 8 * 1024
    assert hw_descr_factory('sm_86', 'cuda').max_reg_per_thread == 255 * 4
    assert hw_descr_factory('gfx942', 'hip').max_reg_per_thread == 512 * 4
    assert hw_descr_factory('gfx1150', 'hip').max_reg_per_thread == 256 * 4
    assert hw_descr_factory('dg1', 'oneapi').max_reg_per_thread is None


def test_a_narrower_vector_does_not_shrink_the_tile():
    """`lanes * slots` is the lead dimension rounded up to a multiple of the
    thread count, so a narrower vector trims the rounding waste and leaves
    `lead * nonlead` alone.  Which is why the budget warning points at how
    much of the operator a work-item owns rather than at the vector width.
    """
    def tile(lead, nonlead, threads):
        return -(-lead // threads) * threads * nonlead

    assert tile(64, 78, 8) == tile(64, 78, 16) == tile(64, 78, 32) == 4992
    # 56 does not divide 16, so the rounding waste *is* visible -- and small
    assert tile(56, 9, 8) == 504
    assert tile(56, 9, 16) == 576


def test_spmd_is_not_warned_about():
    """The check is about a work-item holding a whole tile, which is what this
    lowering does and SPMD does not."""
    from tensorforge.backend.instructions.abstract_instruction import (
        _check_register_budget, RegisterBudgetWarning)
    import warnings as _w
    ctx = Context(arch='pvc', backend='acpp', fp_type=Datatype.F32)
    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter('always', RegisterBudgetWarning)
        _check_register_budget((), False, ctx, 'x')
    assert not caught


# --------------------------------------------------------------------------
# a reduction's accumulator
# --------------------------------------------------------------------------

def test_a_loop_carried_value_adopts_the_distribution_it_is_yielded():
    """An `iter_arg` takes its layout from its `init`, and a reduction's init
    is the operator's neutral element -- a bare literal with no distribution.
    The accumulator only becomes lane-distributed once the body combines it
    with something that is, so the yield is the first moment it is known.
    """
    from tensorforge.backend.pir.build import IRBuilder
    from tensorforge.backend.pir.core import ScalarType
    b = _builder('esimd')
    loop = b.for_(0, 4, 1, inits=(0.0,), types=(ScalarType(Datatype.F32),))
    with loop:
        acc = loop.iter_args[0]
        assert acc.layout is None, 'a literal init carries no distribution'
        spread = b.value(ScalarType(Datatype.F32), layout=SPREAD16)
        loop.yield_(spread)
    assert loop.iter_args[0].layout == SPREAD16


def test_a_vector_accumulator_is_direct_initialised(emitter):
    """`simd<float, 16> acc = 0.0f;` does not compile.

    ESIMD makes the broadcast constructor explicit on purpose -- filling a
    vector from a scalar is a decision, not a conversion -- and a reduction
    starting at its neutral element is exactly that case.
    """
    v = val(80, F32, SPREAD16)
    out = emitter.initialiser(v, 'acc', '0.0f')
    assert out == 'tensorforge::intel_esimd::simd<float, 16> acc(0.0f);'


# --------------------------------------------------------------------------
# a cross-lane reduction, where there are no lanes to cross
# --------------------------------------------------------------------------

def _lexic(simd=True):
    from tensorforge.common.vm.lexic.sycl_lexic import SyclLexic
    return SyclLexic('oneapi' if simd else 'acpp', 'intel', explicit_simd=simd)


@pytest.mark.parametrize('op,spelling', [
    ('ADD', 'reduce<float>(v, std::plus<>())'),
    ('MUL', 'reduce<float>(v, std::multiplies<>())'),
    ('MAX', 'hmax<float>(v)'),
    ('MIN', 'hmin<float>(v)'),
])
def test_the_esimd_entry_points(op, spelling):
    from tensorforge.common.operation import Operation
    out = _lexic().reduction('v', getattr(Operation, op), Datatype.F32, 16)
    assert spelling in out


def test_the_result_is_a_scalar_not_a_broadcast():
    """The reduction *collapses* the lane axis.

    In SPMD an all-reduce leaves every thread holding a copy and "the result"
    is that copy, so the two readings coincide.  Here they do not: broadcasting
    back produced `glb_m1[k] = simd<float, 16>(...)`, a sixteen-wide value
    assigned to a scalar destination.  A caller that wants it in every lane
    spells `simd<T, N>(scalar)` itself.
    """
    from tensorforge.common.operation import Operation
    out = _lexic().reduction('v', Operation.ADD, Datatype.F32, 16)
    assert not out.startswith('tensorforge::intel_esimd::simd<')


def test_nothing_to_combine_is_the_value_itself():
    from tensorforge.common.operation import Operation
    assert _lexic().reduction('v', Operation.ADD, Datatype.F32, 16, 16) == 'v'


@pytest.mark.parametrize('sub', [2, 4, 8])
def test_a_segmented_reduction_keeps_its_group(sub):
    """`subblock > 1` is a different shape, not a narrower one.

    A group survives, so the answer is a *vector* -- unlike the collapse,
    which is a scalar -- and the intrinsics do not answer it: `reduce` and
    `hmax` return one value for the whole thing.  `segmentedReduction` in
    `isycl.h` is the butterfly, and `tests/cpp/esimd_reduction.cpp` checks it
    computes what the CUDA shuffle version computes.
    """
    from tensorforge.common.operation import Operation
    out = _lexic().reduction('v', Operation.ADD, Datatype.F32, 16, sub)
    assert f'segmentedReduction<' in out and f', 16, {sub}, float>' in out


def test_the_segmented_form_covers_the_bitwise_operations():
    """It goes through `ReductionOperation`, which `base.h` defines for every
    backend -- so unlike the collapse, this shape is not limited to the four
    the ESIMD intrinsics happen to have."""
    from tensorforge.common.operation import Operation
    for op in (Operation.XOR, Operation.AND, Operation.OR):
        out = _lexic().reduction('v', op, Datatype.I32, 16, 4)
        assert 'segmentedReduction<' in out


def test_bitwise_reductions_have_no_entry_point():
    from tensorforge.common.operation import Operation
    with pytest.raises(NotImplementedError, match='no ESIMD entry point'):
        _lexic().reduction('v', Operation.XOR, Datatype.I32, 16)


def test_spmd_sycl_still_declines_and_says_why():
    """`reduce_over_group` answers for `subblock == 1` and `block == the
    sub-group size`, and the lexic cannot see the sub-group size to check the
    second -- a reduction over the wrong width is wrong quietly."""
    from tensorforge.common.operation import Operation
    with pytest.raises(NotImplementedError, match='sub-group size'):
        _lexic(simd=False).reduction('v', Operation.ADD, Datatype.F32, 16)


# --------------------------------------------------------------------------
# a change of lane count is not a movement
# --------------------------------------------------------------------------

def test_the_lane_count_does_not_change_where_an_element_lives():
    """Under an explicit vector, an element's address is its index.

    `lane_offset` contributes 0 -- the work-item owns the whole dimension --
    and the slot multiplier *is* the lane count, so element `e` of a lead axis
    sits at `slot * lanes + lane = e` whatever `lanes` is.  Two readers with
    different lane counts therefore see the same storage chunked differently,
    and going between them moves nothing.

    Not so under SPMD, where `lane_offset` is the thread index: there
    `LaneAxis(32)` and `LaneAxis(16)` put element `e` in different *threads*,
    and crossing between them is a shuffle.  Which is why the census counts
    2142 of these as relayouts -- the category is right for the model it was
    written for.
    """
    def address(element, lanes):
        return (element // lanes) * lanes + element % lanes

    for a, b in ((32, 16), (24, 9), (16, 32), (4, 16)):
        assert all(address(e, a) == address(e, b) for e in range(256)), (a, b)


def test_spmd_puts_the_same_element_in_a_different_thread():
    """The counterpart, so the asymmetry is stated rather than implied."""
    def thread_of(element, lanes):
        return element % lanes

    assert thread_of(20, 32) != thread_of(20, 16)


# --------------------------------------------------------------------------
# what a vector width costs to issue
# --------------------------------------------------------------------------

def test_only_powers_of_two_are_one_issue():
    """`Exec_size` is a three-bit field: 1, 2, 4, 8, 16, 32 and nothing else.

    So an operation on a 24-wide vector is issued as 16 + 8 -- two
    instructions -- while the same 24 channels of a 32-wide one are a single
    issue with eight masked off, the mask being bits [7..4] of the same field
    and therefore free.  See `documentation/visa/instructions/MOV.md`.
    """
    from tensorforge.backend.symbol import LeadLoop
    for width in LeadLoop.EXEC_SIZES:
        assert LeadLoop.issues(width) == 1
    for width in (3, 6, 9, 12, 20, 24):
        assert LeadLoop.issues(width) == 2


def test_the_narrowed_widths_are_the_expensive_ones():
    """Which is the tension `_narrow` sits in.

    Replacing a guard with a shorter vector is right for *correctness* -- a
    `simd_mask` is not a branch condition, and the store would have been
    predicated on one.  It is not automatically right for cost: 360 of the
    706 narrowings on the corpus land on a length the hardware cannot issue
    in one go, and the largest bucket is 24, which is two issues where a
    masked 32 would be one.
    """
    from tensorforge.backend.symbol import LeadLoop
    # the widths `_narrow` actually chooses, and what they cost
    assert LeadLoop.issues(24) == 2 and LeadLoop.issues(32) == 1
    assert LeadLoop.issues(12) == 2 and LeadLoop.issues(16) == 1
    assert LeadLoop.issues(9) == 2 and LeadLoop.issues(16) == 1


# --------------------------------------------------------------------------
# an explicit `select` is a merge too
# --------------------------------------------------------------------------

def _select_src(result_type=F32, cond_layout=SPREAD16, other=0.0):
    """Emit `store(image, cond ? load : other)` through the ESIMD lowering."""
    from tensorforge.backend.pir import BOOL, IRBuilder, MemSpace
    from tensorforge.backend.writer import Writer

    builder = IRBuilder(fptype=Datatype.F32)
    image = builder.alloc(Datatype.F32, (16,), MemSpace.REGISTER, hint='image')
    x = builder.load(image, 0, hint='x', layout=cond_layout)
    cond = builder.op('lt', BOOL, x, 9.0, hint='occupied')
    loaded = builder.load(image, 1, hint='stored', layout=SPREAD16)
    picked = builder.op('select', result_type, cond, loaded,
                        builder.const(other), hint='masked')
    builder.store(image, picked, 0)

    ctx = Context(arch='pvc', backend='oneapi', fp_type=Datatype.F32)
    ctx.get_vm().get_lexic().simd_mode = True
    writer = Writer()
    EsimdEmitter(writer=writer, context=ctx, strict=False).run(builder.finish())
    return writer.get_src()


def test_a_masked_select_is_a_merge_and_not_a_ternary():
    """`m ? a : b` on a `simd_mask` has no single bit to test.

    The same answer `declare` already gave a folded predicate, for the select
    the sparse path builds directly.  Reached through a different route --
    `if_convert` attaches a predicate, `Symbol.encode_values` emits the op --
    and lowered by the base emitter as a ternary until this was here.
    """
    src = _select_src()
    assert '.merge(' in src, src
    assert '?' not in src, src


def test_the_merge_is_not_inlined_into_its_consumer():
    """Which is why the interception cannot live in `declare`.

    A single-use select is inlined, so the ternary landed inside a `copy_to`
    argument and no declaration was ever written to override.
    """
    src = _select_src()
    assert 'copy_to' in src
    assert not any('copy_to' in line and 'merge' in line
                   for line in src.splitlines()), src


def test_a_replicated_condition_stays_a_ternary():
    """A uniform bool is a branch condition and a `?:` operand like any other.

    Only a *lane-varying* condition is a mask; spelling both as a merge would
    make a vector out of a value that is one scalar per work-item.
    """
    src = _select_src(cond_layout=SCALAR_LAYOUT)
    assert '?' in src, src
    assert '.merge(' not in src, src


def test_a_boolean_select_over_a_replicated_arm_is_refused():
    """`simd_mask` has no conversion from `bool`, so the arm cannot broadcast.

    Refused with what it would take, rather than emitting mask algebra with a
    `false` in it that does not compile.
    """
    with pytest.raises(IRError, match='both arms to be masks'):
        _select_src(result_type=ScalarType(Datatype.BOOL), other=False)
