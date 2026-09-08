# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Stating a run once and putting it back.

The load-bearing test is the round trip: whatever `roll` does, `unroll` has to
undo exactly.  Everything else here is about what `roll` declines to do, which
matters more than what it does -- a loop that hides a barrier or that tidies
away a repetition nobody asked about is worse than no loop.
"""

import pytest

from tensorforge.analysis.antiunify import skeleton
from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators.descriptions import (ForDescr, GemmDescr,
                                                 GridBarrierDescr,
                                                 GridFenceDescr)
from tensorforge.generators.rolling import roll, unroll

DTYPE = Datatype.F32


def make(alias, shape, is_tmp=False):
    key = (alias, tuple(shape))
    if key not in make.pool:
        make.pool[key] = Tensor(list(shape), Addressing.STRIDED,
                                BoundingBox([0] * len(shape), list(shape)),
                                alias=alias, is_tmp=is_tmp, datatype=DTYPE)
    return SubTensor(make.pool[key])


make.pool = {}


@pytest.fixture(autouse=True)
def fresh_pool():
    make.pool = {}
    yield
    make.pool = {}


def gemm(a, b, c, add=False):
    return GemmDescr(trans_a=False, trans_b=False, a=a, b=b, c=c,
                     alpha=1.0, beta=1.0 if add else 0.0)


def recursion(steps=6):
    return [gemm(make('kDivM', [56, 56]), make(f'deriv{k}', [56, 9]),
                 make(f'deriv{k + 1}', [56, 9])) for k in range(steps)]


def accumulation(faces=4):
    return [gemm(make(f'fPrT{i}', [56, 56]), make('I', [56, 9]),
                 make('Q', [56, 9]), add=True) for i in range(faces)]


def separate(faces=4):
    return [gemm(make(f'fPrT{i}', [56, 56]), make('I', [56, 9]),
                 make(f'face{i}', [56, 9])) for i in range(faces)]


def same_shape(descrs):
    return skeleton(descrs)[0]


# --- the round trip ---------------------------------------------------------


@pytest.mark.parametrize('build', [recursion, accumulation, separate])
def test_unroll_undoes_roll(build):
    original = build()
    assert same_shape(unroll(roll(original))) == same_shape(original)


def test_unroll_undoes_roll_with_a_prologue_and_an_epilogue():
    head = [gemm(make('W', [56, 56]), make('U', [56, 9]), make('I', [56, 9]))]
    tail = [gemm(make('M', [56, 56]), make('Q', [56, 9]), make('R', [56, 9]))]
    original = head + accumulation() + tail
    rolled = roll(original, max_arity=1)
    assert any(isinstance(d, ForDescr) for d in rolled)
    assert same_shape(unroll(rolled)) == same_shape(original)


def test_unroll_leaves_a_list_without_loops_alone():
    original = separate(2)
    assert unroll(original) is not original
    assert same_shape(unroll(original)) == same_shape(original)


def test_rolling_twice_changes_nothing_more():
    once = roll(recursion())
    twice = roll(once)
    assert same_shape(unroll(twice)) == same_shape(unroll(once))


# --- what ends up in the loop -----------------------------------------------


def test_a_recursion_rolls_into_one_sequential_loop():
    out = roll(recursion())
    assert len(out) == 1
    loop = out[0]
    assert isinstance(loop, ForDescr)
    assert (loop.iterations, loop.arity) == (6, 2)
    assert loop.sequential


def test_an_accumulation_rolls_and_may_be_reordered():
    out = roll(accumulation())
    loop = out[0]
    assert (loop.iterations, loop.arity) == (4, 1)
    assert not loop.sequential
    assert loop.dependence.accumulators == ('Q',)
    assert loop.writes().tensor.alias == 'Q'


def test_separate_destinations_leave_the_loop_free():
    loop = roll(separate())[0]
    assert not loop.sequential
    assert loop.dependence.independent
    assert loop.writes() is None
    assert len(loop.destinations()) == 4


def test_the_loop_carries_what_decides_the_buffers():
    """A recursion whose intermediates are summed afterwards is an array."""
    steps = recursion()
    tail = [gemm(make('w', [56, 56]), make(f'deriv{k}', [56, 9]),
                 make('total', [56, 9]), add=True) for k in range(1, 7)]
    loop = roll(steps + tail, max_arity=2)[0]
    assert loop.periods == (None, None)
    assert loop.escaping == tuple(sorted(f'deriv{k}' for k in range(1, 7)))


def test_alternating_buffers_show_a_period_and_still_escape():
    """Both halves of the rotation test, and why they disagree here.

    Two named buffers taking turns give each hole a period of two.  They are
    also tensors the caller passed in, so writing them is observable and a
    rotation over them would not be a rotation but a change of what the kernel
    leaves behind.  Period alone is not the licence; that is the point of
    asking both.
    """
    descrs = [gemm(make('A', [56, 56]), make(f'buf{k % 2}', [56, 9]),
                   make(f'buf{(k + 1) % 2}', [56, 9])) for k in range(6)]
    loop = roll(descrs, max_period=1)[0]
    assert loop.periods == (2, 2)
    assert loop.escaping == ('buf0', 'buf1')


def test_scratch_that_does_not_leave_its_chunk_is_not_a_table():
    """A temporary has no identity outside the body that makes it.

    Which is right for scratch: the generator may rename its own, so two
    chunks holding one each hold the same thing.  It also means a rotation
    over scratch is invisible here, so the rotation test -- a period with
    nothing escaping -- is currently only reachable for buffers the anti-
    unifier can see, and those escape.  Stated rather than worked around,
    since closing it means telling scratch that stays inside a chunk from
    scratch that crosses into the next.
    """
    descrs = [d for k in range(4) for d in
              [gemm(make('A', [56, 56]), make(f'in{k}', [56, 9]),
                    make(f's{k}', [56, 9], True)),
               gemm(make('B', [56, 56]), make(f's{k}', [56, 9], True),
                    make(f'out{k}', [56, 9]))]]
    loop = roll(descrs)[0]
    assert (loop.iterations, loop.arity) == (4, 2)
    assert loop.periods == (None, None)


def test_a_chain_carries_its_shift_and_drops_the_derived_column():
    loop = roll(recursion())[0]
    assert loop.shifts == ((0, 1, 1),)
    assert loop.independent_holes == (0,)


def test_a_loop_reports_the_reads_of_every_iteration():
    loop = roll(accumulation())[0]
    aliases = sorted(v.tensor.alias for v in loop.reads())
    assert aliases == ['I'] + sorted(f'fPrT{i}' for i in range(4))


def test_an_accumulated_destination_is_a_write_here_and_a_read_next_door():
    """`reads()` follows the descriptor contract; the dependence does not.

    A descriptor states the operands it was given, so an accumulated
    destination appears under `writes()` and not under `reads()`.  Asking
    whether one chunk depends on another is a different question and has to
    count that same tensor as read, or every accumulation would look like a
    set of chunks that share nothing.  Both readings are used, so the two
    disagreeing is worth pinning rather than reconciling.
    """
    loop = roll(accumulation())[0]
    assert 'Q' not in [v.tensor.alias for v in loop.reads()]
    assert loop.writes().tensor.alias == 'Q'
    assert loop.dependence.accumulators == ('Q',)


# --- what roll declines to do -----------------------------------------------


def test_a_barrier_in_the_body_stops_the_roll():
    """The section split happens on this list; a loop would hide the barrier."""
    descrs = [d for i in range(4)
              for d in accumulation(4)[i:i + 1] + [GridBarrierDescr()]]
    out = roll(descrs)
    assert not any(isinstance(d, ForDescr) for d in out)
    assert same_shape(out) == same_shape(descrs)


def test_a_fence_in_the_body_stops_the_roll():
    descrs = [d for i in range(4)
              for d in accumulation(4)[i:i + 1] + [GridFenceDescr()]]
    assert not any(isinstance(d, ForDescr) for d in roll(descrs))


def test_a_barrier_may_be_rolled_when_asked_for():
    descrs = [d for i in range(4)
              for d in accumulation(4)[i:i + 1] + [GridFenceDescr()]]
    out = roll(descrs, allow_barriers=True)
    assert isinstance(out[0], ForDescr)
    assert out[0].barrier()
    assert same_shape(unroll(out)) == same_shape(descrs)


def test_a_run_that_varies_in_nothing_is_left_alone():
    """The same computation three times is a question, not an opportunity."""
    once = accumulation(1)
    descrs = once + once + once
    out = roll(descrs)
    assert not any(isinstance(d, ForDescr) for d in out)


def test_a_list_with_nothing_repeated_comes_back_unchanged():
    descrs = [gemm(make('A', [56, 56]), make('I', [56, 9]),
                   make('Q', [56, 9])),
              gemm(make('B', [9, 9]), make('J', [9, 4]), make('R', [9, 4]))]
    out = roll(descrs)
    assert len(out) == 2
    assert not any(isinstance(d, ForDescr) for d in out)


def test_an_empty_list_rolls_to_an_empty_list():
    assert roll([]) == []
    assert unroll([]) == []


# --- the loop where it first meets the backend ------------------------------


def test_the_section_geometry_is_the_same_rolled_or_not():
    """A loop is transparent to whoever asks what a section reads and writes.

    The first thing the generator computes from a descriptor list is its
    geometry, and it decides staging widths and whether a destination may stay
    in registers.  Stating a repetition once rather than writing it out is not
    supposed to change any of that, so the two lists are compared rather than
    the rolled one merely being accepted.
    """
    from tensorforge.backend.scopes import Scopes
    from tensorforge.backend.section_plan import SectionPlan

    for build in (recursion, accumulation, separate):
        original = build()
        rolled = roll(original)
        assert any(isinstance(d, ForDescr) for d in rolled)

        plain = SectionPlan(original, Scopes())
        looped = SectionPlan(rolled, Scopes())

        tensors = {v.tensor for d in original
                   for v in d.reads() + [d.writes()] if v is not None}
        for tensor in tensors:
            assert looped.dest_union(tensor) == plain.dest_union(tensor)
            assert looped.written_in_slices(tensor) == \
                plain.written_in_slices(tensor)


def test_a_plain_descriptor_stands_for_itself():
    descrs = separate(2)
    assert [d.operations() for d in descrs] == [[d] for d in descrs]


def test_a_loop_stands_for_every_operation_of_every_iteration():
    original = accumulation()
    loop = roll(original)[0]
    assert len(loop.operations()) == len(original)
    assert same_shape(loop.operations()) == same_shape(original)


# --- the loop through the generator -----------------------------------------


def _generated(descrs):
    from tensorforge.common.context import Context
    from tensorforge.generators.generator import Generator
    gen = Generator(descrs, Context(arch='sm_86', backend='cuda',
                                    fp_type=DTYPE))
    gen.generate()
    return gen.get_kernel()


def _code_only(text):
    """The kernel without its descriptor comment block or its name."""
    import re
    text = re.sub(r'kernel_[0-9a-f]+', 'K', text)
    return '\n'.join(line for line in text.splitlines()
                     if not line.lstrip().startswith('//'))


def _contributions(count=3):
    return [gemm(make('A', [9, 9]), make(f'i{k}', [9, 4]),
                 make(f'o{k}', [9, 4])) for k in range(count)]


def test_a_rolled_list_generates_the_same_kernel_body():
    """While a loop lowers to its iterations, rolling changes no code.

    The state the loop's own lowering has to be measured against: every walk
    over the descriptor list -- lane geometry, operand naming, the section's
    plan, the builders -- goes over the expansion, so a repetition stated once
    and the same repetition written out reach the same instructions.  A walk
    that was missed shows up here as a difference rather than as a kernel that
    silently drops a body.
    """
    plain = _contributions()
    rolled = roll(_contributions())
    assert any(isinstance(d, ForDescr) for d in rolled)
    assert _code_only(_generated(plain)) == _code_only(_generated(rolled))


def test_the_descriptor_comment_states_the_rolled_form():
    """What does differ, and why it is left differing.

    The comment block above a kernel is the list the generator was handed, and
    a rolled list is not the same list.  It feeds the kernel's name, so a
    rolled kernel is a distinct kernel from the start -- which is what keeps a
    later change of lowering from colliding with a cached build of the same
    name.
    """
    plain = _generated(_contributions())
    rolled = _generated(roll(_contributions()))
    assert plain != rolled
    assert 'for 3' in rolled


def _chain(count=3):
    return [gemm(make('A', [9, 9]), make(f'd{k}', [9, 4]),
                 make(f'd{k + 1}', [9, 4])) for k in range(count)]


def test_a_chain_through_a_published_buffer_generates():
    """The shape of a recursion: each step reads what the last one wrote.

    The intermediate is a tensor the caller passed in, not a temporary, so the
    register image holding it has a global home.  Staging it for the next step
    has to move its lane axis, and a store into shared memory has no way to
    name a global destination -- so the write goes home first and is read back
    from there.

    Pinned because it is the body the first loop will be built from, and
    because it is the one chain shape the temporary-based cases do not cover.
    """
    for count in (2, 3, 5):
        assert _generated(_chain(count))


def test_a_rolled_chain_generates_the_same_body():
    plain, rolled = _chain(4), roll(_chain(4))
    assert any(isinstance(d, ForDescr) for d in rolled)
    assert _code_only(_generated(plain)) == _code_only(_generated(rolled))


# --- whether a run is worth rolling -----------------------------------------


def test_a_small_run_is_left_alone_under_a_budget():
    """A body worth a few hundred lines is cheaper written out.

    Rolled it costs a counter, an indexed load per varying operand and a
    residency that has to survive the back edge; written out it costs lines
    and keeps every operand at a compile-time address.  Which way that goes is
    a size question, so the budget is where the caller states it.
    """
    from tensorforge.analysis.cost import estimated_lines

    descrs = _contributions(3)
    size = estimated_lines(descrs, 32)
    assert not any(isinstance(d, ForDescr) for d in
                   roll(_contributions(3), keep_unrolled_under=size * 2))
    assert any(isinstance(d, ForDescr) for d in
               roll(_contributions(3), keep_unrolled_under=size // 2))


def test_without_a_budget_every_run_rolls():
    assert any(isinstance(d, ForDescr) for d in roll(_contributions(3)))


def test_the_estimate_follows_the_arithmetic_and_the_lanes():
    from tensorforge.analysis.cost import estimated_lines

    small = estimated_lines(_contributions(2), 32)
    large = estimated_lines(_contributions(6), 32)
    assert large > small
    assert estimated_lines(_contributions(6), 64) < large
    with pytest.raises(ValueError):
        estimated_lines(_contributions(2), 0)


def test_a_list_that_already_fits_keeps_every_operand_where_it_was():
    """The question an instruction cache asks is about the list, not a run."""
    from tensorforge.analysis.cost import estimated_lines

    descrs = _contributions(4)
    total = estimated_lines(descrs, 32)
    assert not any(isinstance(d, ForDescr) for d in
                   roll(_contributions(4), fit_within=total + 1))
    assert any(isinstance(d, ForDescr) for d in
               roll(_contributions(4), fit_within=total // 2))


def test_only_as_many_runs_are_rolled_as_the_total_needs():
    from tensorforge.analysis.cost import estimated_lines

    both = _contributions(4) + _chain(4)
    budget = estimated_lines(both, 32) - estimated_lines(_chain(4), 32) // 2
    rolled = roll(_contributions(4) + _chain(4), fit_within=budget)
    assert sum(isinstance(d, ForDescr) for d in rolled) == 1


def test_the_budget_does_not_change_what_the_list_means():
    from tensorforge.analysis.cost import estimated_lines

    original = _contributions(4) + _chain(4)
    total = estimated_lines(original, 32)
    for budget in (total * 2, total, total // 2, 1):
        rolled = roll(_contributions(4) + _chain(4), fit_within=budget)
        assert same_shape(unroll(rolled)) == \
            same_shape(_contributions(4) + _chain(4))


def test_the_two_questions_are_asked_separately():
    """A run below the floor is left alone however tight the budget is."""
    from tensorforge.analysis.cost import estimated_lines

    descrs = _contributions(3)
    floor = estimated_lines(descrs, 32) * 2
    assert not any(isinstance(d, ForDescr) for d in
                   roll(_contributions(3), keep_unrolled_under=floor,
                        fit_within=1))


# --- what a builder gets ----------------------------------------------------


def test_the_body_names_a_stand_in_and_not_a_member():
    """A member's name is right for one iteration in four and wrong for three."""
    from tensorforge.generators.rolling import variant_body

    loop = roll(accumulation())[0]
    body, variants = variant_body(loop)
    named = {v.tensor.alias for d in body for v in d.reads() + [d.writes()]
             if v is not None}
    assert 'variant0' in named
    assert not any(n.startswith('fPrT') for n in named)


def test_a_stand_in_is_interchangeable_with_what_it_stands_for():
    from tensorforge.analysis.antiunify import operand_key
    from tensorforge.generators.rolling import variant_body

    _, variants = variant_body(roll(accumulation())[0])
    for variant in variants:
        keys = {operand_key(m) for m in variant.members}
        assert keys == {operand_key(variant.stand_in)}


def test_one_variant_per_hole_holding_one_member_per_iteration():
    from tensorforge.generators.rolling import variant_body

    loop = roll(recursion())[0]
    body, variants = variant_body(loop)
    assert len(variants) == loop.arity
    assert all(v.count == loop.iterations for v in variants)
    assert [m.tensor.alias for m in variants[0].members] == \
        [f'deriv{k + 1}' for k in range(6)]


def test_the_body_is_one_iteration_long():
    from tensorforge.generators.rolling import variant_body

    loop = roll(accumulation())[0]
    body, _ = variant_body(loop)
    assert len(body) == len(loop.general.template)
    assert same_shape(body) != same_shape(accumulation())


def test_binding_the_members_back_gives_the_iterations_again():
    """The stand-in is a name, not a change of meaning."""
    from tensorforge.analysis.antiunify import instantiate

    loop = roll(accumulation())[0]
    for member in range(loop.iterations):
        assert same_shape(instantiate(loop.general, member)) == \
            same_shape(loop.body(member))


def test_the_prefix_names_the_stand_ins():
    from tensorforge.generators.rolling import variant_body

    _, variants = variant_body(roll(accumulation())[0], prefix='face')
    assert [v.stand_in.tensor.alias for v in variants] == ['face0']


def test_the_decomposition_is_the_same_one_every_time():
    """The generator names operands in one phase and resolves them in another.

    A decomposition rebuilt between the two would hand the second phase
    tensors the first has never seen, so the stand-ins are identities and not
    values.
    """
    loop = roll(accumulation())[0]
    first, second = loop.decompose(), loop.decompose()
    assert first[0] is second[0] and first[1] is second[1]
    assert loop.stand_ins()[0].tensor is loop.stand_ins()[0].tensor


def test_the_stand_ins_are_not_among_the_members():
    loop = roll(accumulation())[0]
    members = {m.tensor.alias for v in loop.variants() for m in v.members}
    assert {s.tensor.alias for s in loop.stand_ins()}.isdisjoint(members)


def test_a_loop_reports_one_stand_in_per_hole():
    loop = roll(recursion())[0]
    assert len(loop.stand_ins()) == loop.arity


def test_a_stand_in_does_not_move_the_parameters_along():
    """Its own naming series, so a rolled kernel's parameters keep their places."""
    from tensorforge.common.context import Context
    from tensorforge.generators.generator import Generator

    def named(descrs):
        gen = Generator(descrs, Context(arch='sm_86', backend='cuda',
                                        fp_type=DTYPE))
        gen.register()
        return {m.alias: m.name for m in gen._matrix_list
                if not getattr(m, 'is_variant', False)}

    assert named(roll(accumulation())) == named(accumulation())


def test_a_stand_in_is_named_apart_and_kept_out_of_the_signature():
    from tensorforge.common.context import Context
    from tensorforge.generators.generator import Generator

    gen = Generator(roll(accumulation()), Context(arch='sm_86', backend='cuda',
                                                  fp_type=DTYPE))
    gen.generate()
    variants = [m for m in gen._matrix_list if getattr(m, 'is_variant', False)]
    assert [m.name for m in variants] == ['v0']
    signature = next(l for l in gen.get_kernel().splitlines()
                     if 'kernel_kernel' in l)
    assert ' v0' not in signature


def test_a_stand_in_is_neither_a_parameter_nor_a_temporary():
    loop = roll(accumulation())[0]
    stand_in = loop.stand_ins()[0].tensor
    assert stand_in.is_variant
    assert not stand_in.is_tmp


def test_an_ordinary_tensor_is_not_a_variant():
    assert not accumulation()[0].writes().tensor.is_variant


def _built(loops):
    from tensorforge.common.context import Context
    from tensorforge.generators.generator import Generator
    gen = Generator(roll(accumulation()),
                    Context(arch='sm_86', backend='cuda', fp_type=DTYPE))
    gen._emit_loops = loops
    gen.generate()
    return gen.get_kernel()


def test_the_loop_generates_and_verifies():
    """Both forms come out; the default is still expansion.

    Off not because it cannot be built -- it can, and the verifier accepts it --
    but because nothing has yet compared the two numerically, and expansion is
    exact.
    """
    assert _built(True)
    assert _built(False)


def test_the_loop_form_is_one_body_and_a_counter():
    kernel = _built(True)
    assert kernel.count('for (int batchIdv') == 1
    assert 'Table = (batchIdv' in kernel


def test_the_registers_are_declared_outside_the_loop():
    """An allocation is not a per-iteration act, and its users sit outside."""
    lines = _built(True).splitlines()
    header = next(i for i, l in enumerate(lines) if 'for (int batchIdv' in l)
    allocs = [i for i, l in enumerate(lines)
              if l.strip().startswith('float r') and '[' in l]
    assert allocs and all(i < header for i in allocs)


def _loop_region():
    from tensorforge.common.context import Context
    from tensorforge.generators.generator import Generator
    gen = Generator(roll(accumulation()),
                    Context(arch='sm_86', backend='cuda', fp_type=DTYPE))
    gen._emit_loops = True
    gen.generate()
    loop = next(i for i in gen._sections[0].ir
                if type(i).__name__ == 'VariantLoop')
    return loop


def test_the_peeled_iteration_leaves_the_invariants_outside():
    """What every iteration shares is done once, without a pass for it.

    The shared staging of the operand all iterations read and the load of the
    destination both happen in the peeled copy, and the residency keeps the
    body from repeating them.
    """
    loop = _loop_region()
    kinds = [type(i).__name__ for i in loop.region]
    assert kinds.count('GlbToShrLoader') == 0
    assert kinds.count('GlbToRegLoader') == 1
    assert len(loop.region) < 6


def test_the_loop_runs_the_iterations_the_peel_did_not():
    loop = _loop_region()
    assert (loop.start, loop.count) == (1, 4)
    assert 'for (int' in loop.header().join(('for (int ', ''))


def _loop_of(descrs):
    from tensorforge.common.context import Context
    from tensorforge.generators.generator import Generator
    gen = Generator(descrs, Context(arch='sm_86', backend='cuda',
                                    fp_type=DTYPE))
    gen._emit_loops = True
    gen.generate()
    return next(i for i in gen._sections[0].ir
                if type(i).__name__ == 'VariantLoop')


def test_the_loop_names_the_value_its_body_threads_through_itself():
    """The pair a back edge has to close, read off either side of the build.

    A destination is not accumulated in place -- each build takes a fresh
    register and reads the last -- so what the body holds is one turn of the
    expanded form's chain, and the two links being different registers is the
    whole of what is missing.
    """
    loop = _loop_of(roll(accumulation()))
    assert len(loop.carried) == 1
    init, result = loop.carried[0]
    # Reported after the substitution closed it, so the two links are one.
    assert init is result

    compute = next(i for i in loop.region
                   if type(i).__name__ == 'MultilinearInstruction')
    assert init in compute.uses()
    assert init in compute.defs()


def test_a_body_that_accumulates_nothing_carries_nothing():
    """Empty means nothing to carry, not a carried value gone unnoticed."""
    assert _loop_of(roll(separate())).carried == ()


def test_the_body_reads_and_writes_the_same_register():
    """The chain closed: a repeated body has to land where it started.

    Substituted on the built region rather than arranged during the build,
    because what the residency hands out is its business and the fact that a
    body will be repeated is not something it can know.
    """
    loop = _loop_of(roll(accumulation()))
    compute = next(i for i in loop.region
                   if type(i).__name__ == 'MultilinearInstruction')
    written = {s.name for s in compute.defs()}
    assert written & {s.name for s in compute.uses()} == written


def test_the_writeback_stores_the_register_the_loop_kept():
    """Emitted after the loop is assembled, so the residency has to be told."""
    from tensorforge.common.context import Context
    from tensorforge.generators.generator import Generator
    gen = Generator(roll(accumulation()),
                    Context(arch='sm_86', backend='cuda', fp_type=DTYPE))
    gen._emit_loops = True
    gen.generate()
    loop = next(i for i in gen._sections[0].ir
                if type(i).__name__ == 'VariantLoop')
    stores = [i for i in gen._sections[0].ir
              if type(i).__name__ == 'StoreRegToGlb']
    kept = {s.name for i in loop.region for s in i.defs()}
    assert len(stores) == 1
    assert {s.name for s in stores[0].uses()} <= kept


def test_substitution_reports_whether_it_reached_anything():
    """So a substitution that did nothing is not mistaken for one applied."""
    loop = _loop_of(roll(accumulation()))
    compute = next(i for i in loop.region
                   if type(i).__name__ == 'MultilinearInstruction')
    absent = object()
    assert compute.substitute(absent, absent) is False
