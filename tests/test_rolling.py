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
from tensorforge.generators.descriptions import (GemmDescr,
                                                 MultilinearDescr,
                                                 GridBarrierDescr,
                                                 GridFenceDescr)
from tensorforge.analysis.families import find_repeats
from tensorforge.generators.rolling import ForDescr, _chunks, roll, unroll

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


def _bound(descrs):
    """Each descriptor as the tensors it names, by identity.

    `same_shape` compares skeletons, which is what a hole abstracts away -- so
    a roll that binds the wrong tensor comes back as the same shape and the
    round trip says nothing about it.  This says which tensors.

    The kind is left out deliberately: `rebuild` makes a `MultilinearDescr`
    of a `GemmDescr`, which is the same operation stated in the general form,
    and `same_shape` is where that belongs.
    """
    out = []
    for descr in descrs:
        dest = descr.writes()
        out.append((bool(getattr(descr, 'add', False)),
                    None if dest is None else id(dest.tensor),
                    tuple(id(v.tensor) for v in descr.reads())))
    return out


def nodal_sources(nodes=3):
    """The damage step's shape: two sums over nodal sources, all temporaries.

    Per node, one contribution to `alphaNodal` and one to `breakageNodal`,
    each weighted by that node's own scalar.  The sources are written before
    the run and the sums read after it, so every tensor in the run carries a
    value in or out -- and all six are temporaries, which is the whole point:
    identified by position they all look alike, and a run of six chunks then
    varies in nothing but the scalar.
    """
    def number(alias):
        return SubTensor(Tensor([], Addressing.SCALAR, BoundingBox([], []),
                                alias=alias, datatype=DTYPE))

    def contribution(dest, source, scalar):
        return MultilinearDescr(dest=dest, ops=[source, number(scalar)],
                                target=[[0], []], permute=[[0], []], add=True)

    before, run = [], []
    for k in range(nodes):
        for kind in ('alpha', 'breakage'):
            source = make(f'source{kind.capitalize()}{k}', [343], True)
            before.append(MultilinearDescr(
                dest=source, ops=[make(f'state{k}', [343])],
                target=[[0]], permute=[[0]], add=False))
            run.append(contribution(make(f'{kind}Nodal', [343], True),
                                    source, f'weight{k}'))
    after = [MultilinearDescr(dest=make('out', [343]),
                              ops=[make(f'{kind}Nodal', [343], True)],
                              target=[[0]], permute=[[0]], add=True)
             for kind in ('alpha', 'breakage')]
    return before + run + after


def test_a_run_over_temporaries_keeps_which_temporary_it_names():
    """Two sums over nodal sources, rolled: iteration k adds source k.

    Identifying a temporary by the slot it sits in rests on the generator
    naming its own scratch, so that two chunks each holding one hold the same
    thing.  These are not that: `sourceAlpha0` and `sourceAlpha1` are two
    values, written elsewhere, and `alphaNodal` and `breakageNodal` are two
    sums read afterwards.  With position standing in for all four, the six
    additions would come out as one chunk varying in the scalar alone -- so
    the loop would add the first source six times and never compute the
    breakage sums.  SeisSol's damage step is this list.
    """
    original = nodal_sources()
    assert _bound(unroll(roll(original, min_count=3))) == _bound(original)


def temporary_chain(steps=6):
    """A recursion through temporaries: each step reads what the last wrote."""
    return [gemm(make('kDivM', [56, 56]), make(f'deriv{k}', [56, 9], True),
                 make(f'deriv{k + 1}', [56, 9], True)) for k in range(steps)]


def test_scratch_that_crosses_into_the_next_chunk_is_not_scratch():
    """The other half of the same question.

    `deriv{k}` is written by one chunk and read by the next, so it is a value
    passed on and not scratch, even though nothing outside the run ever looks
    at one.  The anti-unifier makes a hole of it -- identified by its slot it
    would be invisible, and every chunk would read the first step's input.
    """
    from tensorforge.analysis.families import find_repeats
    run = find_repeats(temporary_chain(), min_count=3)[0]
    assert run.arity >= 2


def test_a_run_that_carries_a_temporary_rolls_and_rotates_it():
    """A member is reached through a table of addresses and a temporary has
    none until something asks for its buffer -- so the loop hands them out
    itself, as many as the carry needs rather than one per member.

    `temporary_chain` is the shape: `deriv{k}` is written by one chunk and
    read by the next, a flow at distance one, so two buffers serve the
    whole chain and the members take them in turn
    (`Generator._rotate_members`).  Unrolling it again gives back the list
    that went in, which is the property the rotation must not break.
    """
    descrs = temporary_chain()
    rolled = roll(descrs, min_count=3)
    assert any(isinstance(d, ForDescr) for d in rolled)
    assert _bound(unroll(rolled)) == _bound(descrs)


def _slot_buffers(gen):
    """The distinct shared buffers the run's tables select from."""
    from tensorforge.backend.instructions.ptr_manip import VariantLoop
    from tensorforge.backend.symbol import SymbolType
    ir = getattr(gen, 'built', gen)._sections[0].ir
    loop, = [i for i in ir if isinstance(i, VariantLoop)]
    return {m.name for t in loop.tables for m in t.get_operands()
            if m.stype == SymbolType.SharedMem
            and getattr(m, 'rotation_slot', False)}


def _carried_chain(steps=5):
    """`temporary_chain` with ends: an input to start it and a sink to read
    it, so it is a kernel that can be built and not only rolled."""
    out = [gemm(make('kDivM', [56, 56]),
                make(f'deriv{k}', [56, 9], is_tmp=k > 0),
                make(f'deriv{k + 1}', [56, 9], True)) for k in range(steps)]
    out.append(gemm(make('P', [56, 56]), make(f'deriv{steps}', [56, 9], True),
                    make('Q', [56, 9])))
    return out


def test_a_family_comes_out_in_the_order_its_chain_is_written():
    """The member's position decides its slot, so the order is the rotation.

    A hole and the hole it feeds are the same list shifted -- `deriv1..4`
    read against `deriv2..5` written -- and the member only the reading hole
    names is written *before* the loop, so it belongs at the front.  Merged
    by appending it landed last, took the slot its own successor wanted, and
    the one buffer a width of 1 should collapse to came out as two.
    """
    from tensorforge.analysis.families import find_repeats
    from tensorforge.generators.rolling import rotation_families
    descrs = _carried_chain()
    run = find_repeats(descrs, min_count=2)[0]
    holes = [[getattr(v, 'tensor', v) for v in hole]
             for hole in zip(*run.general.bindings)]
    groups = [[t for t in h if getattr(t, 'is_tmp', False)] for h in holes]
    family, = [f for f in rotation_families(groups) if len(f) > 1]
    assert [t.alias for t in family] == [f'deriv{k}' for k in range(1, 6)]


@pytest.mark.parametrize('slots, want', [(0, 2), (3, 3)])
def test_the_slots_a_carried_family_turns_through_are_asked_for(slots, want):
    """`rotate_slots` is the width of the rotation, and 0 is the carry's own.

    The default is one buffer more than the distance the run carries over,
    which is what the value's lifetime needs.  More buys nothing and is
    allowed because refusing it would be a rule with no case behind it.
    One is the other end and has no buffer at all -- see below.
    """
    gen = _merged_build(roll(_carried_chain(), min_count=3),
                        **({'rotate_slots': slots} if slots else {}))
    assert len(_slot_buffers(gen)) == want


def test_one_slot_carries_the_family_in_registers():
    """At one slot the family selects nothing, so its holes need no table --
    and a hole with no table needs no address, which was the only reason the
    value was in memory.  The loop then reads and writes the register image
    the prologue left, and the buffer the rotation would have taken is not
    taken at all: on the five-step chain that is 16 kB of a gfx942 block,
    and the list written out takes none either.
    """
    gen = _merged_build(roll(_carried_chain(), min_count=2),
                        arch='gfx942', backend='hip', rotate_slots=1)
    assert not _slot_buffers(gen)
    src = gen.get_kernel()
    assert 'store{r>s}' not in src, 'the carry still goes through memory'
    assert 'Table' not in src, 'a hole with one member still selects'


def test_a_temporary_that_is_not_carried_is_left_written_out():
    """Rotating it would take away the registers the body gives it.

    A scratch written and read inside one chunk is dead at its end, and the
    loop already starts it from zeros on every trip
    (`Generator._reset_temporaries`).  Only a family that crosses an
    iteration boundary has anything to rotate, and this one does not: each
    chunk names its own and nothing reads it afterwards.
    """
    descrs = [d for k in range(4) for d in
              [gemm(make('A', [56, 56]), make(f'in{k}', [56, 9]),
                    make(f'scratch{k}', [56, 9], True)),
               gemm(make('B', [56, 56]), make(f'scratch{k}', [56, 9], True),
                    make(f'out{k}', [56, 9]))]]
    run = find_repeats(descrs, min_count=2)[0]
    chunks = _chunks(descrs, run.start, run.period, run.count)
    # The anti-unifier keeps such a scratch out of the holes entirely, so
    # there is nothing for the rotation to refuse -- stated here because
    # the refusal would be the fallback if that ever changed.
    holes = [[getattr(v, 'tensor', v) for v in hole]
             for hole in zip(*run.general.bindings)]
    assert not any(getattr(t, 'is_tmp', False) for h in holes for t in h)


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
    leaves behind.  Period alone is not the license; that is the point of
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

    Pinned because it is the body a loop is built from, and because it is
    the one chain shape the temporary-based cases do not cover.
    """
    for count in (2, 3, 5):
        assert _generated(_chain(count))


def test_a_rolled_chain_generates_the_same_body():
    plain, rolled = _chain(4), roll(_chain(4))
    assert any(isinstance(d, ForDescr) for d in rolled)
    assert _code_only(_generated(plain)) == _code_only(_generated(rolled))


# --- whether a run is worth rolling -----------------------------------------


def _flops(descrs):
    """A size for the budgets below.  The generator's is bytes of code from
    a build; any measure that adds up over a list asks the same question."""
    from tensorforge.analysis.cost import list_cost
    return list_cost(list(descrs)).flops


def test_a_small_run_is_left_alone_under_a_budget():
    """A small body is cheaper written out.

    Rolled it costs a counter, an indexed load per varying operand and a
    residency that has to survive the back edge; written out it keeps every
    operand at a compile-time address.  Which way that goes is a size
    question, so the budget is where the caller states it.
    """
    size = _flops(_contributions(3))
    assert not any(isinstance(d, ForDescr) for d in
                   roll(_contributions(3), keep_unrolled_under=size * 2,
                        size=_flops))
    assert any(isinstance(d, ForDescr) for d in
               roll(_contributions(3), keep_unrolled_under=size // 2,
                    size=_flops))


def test_without_a_budget_every_run_rolls():
    assert any(isinstance(d, ForDescr) for d in roll(_contributions(3)))


def test_a_budget_without_a_size_is_refused():
    with pytest.raises(ValueError):
        roll(_contributions(3), fit_within=100)


def test_a_list_that_already_fits_keeps_every_operand_where_it_was():
    """The question an instruction cache asks is about the list, not a run."""
    total = _flops(_contributions(4))
    assert not any(isinstance(d, ForDescr) for d in
                   roll(_contributions(4), fit_within=total + 1, size=_flops))
    assert any(isinstance(d, ForDescr) for d in
               roll(_contributions(4), fit_within=total // 2, size=_flops))


def test_only_as_many_runs_are_rolled_as_the_total_needs():
    both = _contributions(4) + _chain(4)
    budget = _flops(both) - _flops(_chain(4)) // 2
    rolled = roll(_contributions(4) + _chain(4), fit_within=budget,
                  size=_flops)
    assert sum(isinstance(d, ForDescr) for d in rolled) == 1


def test_the_budget_does_not_change_what_the_list_means():
    total = _flops(_contributions(4) + _chain(4))
    for budget in (total * 2, total, total // 2, 1):
        rolled = roll(_contributions(4) + _chain(4), fit_within=budget,
                      size=_flops)
        assert same_shape(unroll(rolled)) == \
            same_shape(_contributions(4) + _chain(4))


def test_the_two_questions_are_asked_separately():
    """A run below the floor is left alone however tight the budget is."""
    floor = _flops(_contributions(3)) * 2
    assert not any(isinstance(d, ForDescr) for d in
                   roll(_contributions(3), keep_unrolled_under=floor,
                        fit_within=1, size=_flops))


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
    assert kernel.count('for (int32_t batchIdv') == 1
    # selected by the counter -- the next trip's, where the operand is
    # fetched a trip ahead (`wrap`)
    assert 'Table = (' in kernel


def test_the_registers_are_declared_outside_the_loop():
    """An allocation is not a per-iteration act, and its users sit outside."""
    lines = _built(True).splitlines()
    header = next(i for i, l in enumerate(lines) if 'for (int32_t batchIdv' in l)
    allocs = [i for i, l in enumerate(lines)
              if l.strip().startswith('float r') and '[' in l]
    assert allocs and all(i < header for i in allocs)


def _merged_build(descrs, arch='sm_86', backend='cuda', **options):
    from tensorforge.common.context import Context, Options
    from tensorforge.generators.generator import Generator
    gen = Generator(descrs, Context(arch=arch, backend=backend, fp_type=DTYPE,
                                    options=Options(**options)))
    gen._emit_loops = True
    gen.generate()
    return gen


def test_the_counter_selects_through_a_value():
    """The select chain is a definition the bindings read, with the counter
    as its operand: no wall in the body, and something to substitute for a
    pass that builds a trip's statements for another one."""
    from tensorforge.backend.pir.core import Effect, Op, walk_stmts
    gen = _merged_build(roll(accumulation()))
    loop = next(s for s in walk_stmts(gen._sections[0].body)
                if s.op is Op.FOR and s.attr('rolled'))
    body = loop.regions[0].body
    assert not any(s.effect & Effect.UNKNOWN for s in walk_stmts(body))

    def named(suffix):
        return next(s for s in body if s.op is Op.RAWEXPR
                    and str(s.attr('extern', '')).endswith(suffix))
    defs = {t.id: s for s in walk_stmts(body) for t in s.target}

    def of_the_counter(v):
        if v is loop.regions[0].args[0]:
            return True
        s = defs.get(getattr(v, 'id', None))
        return s is not None and any(of_the_counter(a) for a in s.operands())
    chain = named('v0Table')
    assert of_the_counter(chain.args[0])
    assert chain.target[0] in named('glb_v0').args


def _around_the_loop(gen):
    """`(ahead, loop, behind)`: the section's instructions on either side of
    its merged run."""
    ir = getattr(gen, 'built', gen)._sections[0].ir
    at = next(k for k, i in enumerate(ir) if type(i).__name__ == 'VariantLoop')
    return ir[:at], ir[at], ir[at + 1:]


def _kinds(instrs):
    return [type(i).__name__ for i in instrs]


def _rolled_body(gen):
    """The statements of the merged run's loop in the section's pseudo-IR,
    nested ones included."""
    from tensorforge.backend.pir.core import Op, walk_stmts
    loop = next(s for s in walk_stmts(gen._sections[0].body)
                if s.op is Op.FOR and s.attr('rolled'))
    return list(walk_stmts(loop.regions[0].body))


def test_the_loop_runs_every_iteration():
    """Nothing is built beside the loop: the destination it accumulates onto
    is made resident ahead of it instead (`Generator._establish`)."""
    ahead, loop, _ = _around_the_loop(_merged_build(roll(accumulation())))
    assert (loop.start, loop.count) == (0, 4)
    assert loop.header().startswith('int32_t ')
    assert 'MultilinearInstruction' not in _kinds(ahead)


def test_the_destination_is_loaded_ahead_of_the_loop():
    """Where the policy keeps results in registers, the first iteration finds
    the destination loaded, as the written-out list's first operation would
    have loaded it, and every iteration adds onto the one before it."""
    ahead, loop, behind = _around_the_loop(_merged_build(roll(accumulation())))
    loads = [i for i in ahead if type(i).__name__ == 'GlbToRegLoader']
    assert len(loads) == 1
    assert loads[0].uses()[0].obj.alias == 'Q'
    # the varying operand is still loaded every iteration, and the sum goes
    # out once, after the loop
    assert _kinds(loop.region).count('GlbToRegLoader') == 1
    assert _kinds(behind).count('StoreRegToGlb') == 1


def test_an_atomic_destination_starts_from_zeros():
    """Where the policy adds atomically, the run is built as everywhere else
    -- the destination loaded ahead of the loop, the sum stored behind it --
    and the body adds it atomically, since nothing reads the sum
    (`pir.atomics`): the load ahead becomes zeros and goes, and the store
    behind adds."""
    from tensorforge.backend.pir.core import Effect, Op, walk_stmts
    gen = _merged_build(roll(accumulation()), arch='gfx942', backend='hip')
    ahead, loop, behind = _around_the_loop(gen)
    assert loop.start == 0
    assert any('Loader' in k for k in _kinds(ahead))
    q = _output(gen)
    body = tuple(walk_stmts(gen._sections[0].body))
    assert not any(a.base is q for s in body if s.op is Op.LOAD
                   for a in s.accesses)
    assert any(a.base is q and a.kind & Effect.ATOMIC for s in body
               if s.op is Op.STORE for a in s.accesses)


def test_what_every_iteration_shares_leaves_the_loop():
    """The operand every iteration reads is staged in the body, the same copy
    on every trip, and leaves the loop in the pseudo-IR (`pir.move`) -- with
    the wait that retires it, so nothing is in flight across the loop."""
    from tensorforge.backend.pir.core import Effect, MemSpace, Op
    gen = _merged_build(roll(accumulation()))
    _, loop, _ = _around_the_loop(gen)
    assert 'GlbToShrLoader' in _kinds(loop.region)
    body = _rolled_body(gen)
    assert not any(a.space is MemSpace.SHARED and a.kind & Effect.WRITE
                   for s in body for a in s.accesses)
    assert not any(s.op is Op.WAIT for s in body)


def _writer_of_the_destination(loop):
    """The instruction that reads the accumulated value and writes it back.

    An accumulation's epilogue (`MultilinearEpilogue`): the contraction only
    fills its own accumulator, and the previous value and the destination are
    the epilogue's.  Where there is none the contraction itself.
    """
    kinds = [type(i).__name__ for i in loop.region]
    kind = ('MultilinearEpilogue' if 'MultilinearEpilogue' in kinds
            else 'MultilinearInstruction')
    return next(i for i in loop.region if type(i).__name__ == kind)


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

    compute = _writer_of_the_destination(loop)
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
    compute = _writer_of_the_destination(loop)
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


def _stores(descrs):
    from tensorforge.common.context import Context
    from tensorforge.generators.generator import Generator
    gen = Generator(descrs, Context(arch='sm_86', backend='cuda',
                                    fp_type=DTYPE))
    gen._emit_loops = True
    gen.generate()
    ir = gen._sections[0].ir
    loop = next(i for i in ir if type(i).__name__ == 'VariantLoop')
    inside = sum('Store' in type(i).__name__ for i in loop.region)
    after = sum('Store' in type(i).__name__ for i in ir[ir.index(loop) + 1:])
    return inside, after


def test_an_accumulator_is_written_once_after_the_loop():
    """Which is the whole reason for rolling this one: the sum goes out once."""
    assert _stores(roll(accumulation())) == (0, 1)


def test_a_destination_that_varies_is_written_every_iteration():
    """Its address is a different tensor each time.

    The residency flushes once, against whatever the entry holds by then --
    the last iteration's address -- so without a store inside, every iteration
    computes and only the last is kept.
    """
    inside, _ = _stores(roll(recursion()))
    assert inside == 1


def test_the_rule_is_whether_the_destination_varies_not_whether_it_escapes():
    """An accumulator escapes too, and asking that stores the sum every pass."""
    loop = [d for d in roll(accumulation()) if isinstance(d, ForDescr)][0]
    assert 'Q' in loop.escaping
    assert _stores(roll(accumulation()))[0] == 0


def _flux(count=4):
    """First contribution assigns, the rest add -- what a frontend emits."""
    return [gemm(make(f'fPrT{i}', [9, 9]), make('I', [9, 4]),
                 make('Q', [9, 4]), add=(i > 0)) for i in range(count)]


def test_a_run_after_an_assigning_contribution_is_not_peeled_again():
    """The residency it would establish is already established.

    A frontend writes the first contribution as an assignment and the rest as
    additions, so the first falls out of the run -- and it is exactly the
    iteration a peel would add.  Peeling on top of it writes a second copy of
    the body for nothing.
    """
    from tensorforge.common.context import Context
    from tensorforge.generators.generator import Generator

    gen = Generator(roll(_flux()), Context(arch='sm_86', backend='cuda',
                                           fp_type=DTYPE))
    gen._emit_loops = True
    gen.generate()
    ir = gen._sections[0].ir
    loop = next(i for i in ir if type(i).__name__ == 'VariantLoop')
    assert loop.start == 0
    outside = [i for i in ir if type(i).__name__ == 'MultilinearInstruction']
    assert len(outside) == 1


def test_where_every_result_goes_out_to_memory_the_run_is_peeled():
    """A policy that stores each result as it is computed leaves nothing in
    registers for the next iteration to add onto, so there is no image to
    make resident ahead of the loop (`MultilinearBuilder.establish`)."""
    from dataclasses import replace

    from tensorforge.common import target
    base = target.PREFERENCES['nvidia']
    target.PREFERENCES['nvidia'] = replace(base, placement=replace(
        base.placement, keep_results_in_registers=False))
    try:
        _, loop, _ = _around_the_loop(_merged_build(roll(accumulation())))
    finally:
        target.PREFERENCES['nvidia'] = base
    assert loop.start == 1


def _read_first(count=4):
    """`T_i = W_i D; D += V_i T_i`: each iteration reads the destination
    before it adds to it."""
    out = []
    for i in range(count):
        tmp = make('tmp', [9, 4], is_tmp=True)
        out.append(gemm(make(f'W{i}', [9, 9]), make('D', [9, 4]), tmp))
        out.append(gemm(make(f'V{i}', [9, 9]), tmp, make('D', [9, 4]),
                        add=True))
    return out


def test_a_run_that_reads_its_destination_first_is_peeled():
    """A chain that starts with a read of the destination needs its value,
    not a place to add onto, so the first iteration is built beside the loop
    and leaves the residency behind."""
    ahead, loop, _ = _around_the_loop(_merged_build(roll(_read_first())))
    assert (loop.start, loop.count) == (1, 4)
    assert 'MultilinearInstruction' in _kinds(ahead)


def _add_onto_a_read(count=4):
    """`D += W_i D`: each iteration's first touch of the destination is an
    accumulation that also reads it."""
    return [gemm(make(f'W{i}', [9, 9]), make('D', [9, 4]), make('D', [9, 4]),
                 add=True) for i in range(count)]


def test_a_run_that_adds_onto_what_it_reads_is_peeled():
    """An accumulation that reads its own destination needs the value as
    well as a place to add onto -- decided at the chain's first statement,
    which here is the accumulation itself."""
    ahead, loop, _ = _around_the_loop(_merged_build(roll(_add_onto_a_read())))
    assert (loop.start, loop.count) == (1, 4)
    assert 'MultilinearInstruction' in _kinds(ahead)


def test_a_destination_assembled_from_several_writes_is_peeled():
    """Each write of a destination written in slices goes out to memory as
    it is produced, so there is no image to carry from one iteration to the
    next (`MultilinearBuilder.establish`)."""
    gen = _with_option(_accumulate_then_read().descr_list(),
                       merge_variants=True)
    _, loop, _ = _around_the_loop(gen)
    assert loop.start == 1


def test_the_driver_counts_what_the_launcher_takes():
    """A stand-in is in the scope and out of the signature.

    Counting it hands the launcher two arguments more than it has parameters,
    which fails at the call site and says nothing about where it came from.
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from harness.driver_emit import collect_operands
    from tensorforge.common.context import Context
    from tensorforge.generators.generator import Generator

    gen = Generator(roll(_flux()), Context(arch='sm_86', backend='cuda',
                                           fp_type=DTYPE))
    gen._emit_loops = True
    gen.generate()
    counted = {op.kernel_name for op in collect_operands(gen)}
    variants = {m.name for m in gen._matrix_list
                if getattr(m, 'is_variant', False)}
    assert variants
    assert counted.isdisjoint(variants)


# --- one switch -------------------------------------------------------------


def _with_option(descrs, arch='sm_86', backend='cuda', **options):
    from tensorforge.common.context import Context, Options
    from tensorforge.generators.generator import Generator
    gen = Generator(descrs, Context(arch=arch, backend=backend,
                                    fp_type=DTYPE, options=Options(**options)))
    gen.generate()
    return gen


def _merged(gen):
    return any(type(i).__name__ == 'VariantLoop'
               for i in gen.built._sections[0].ir)


def test_the_option_rewrites_and_emits_in_one_step():
    """Two separately reachable switches would let a list be rolled and then
    silently expanded again."""
    plain = _with_option(_flux(), merge_variants=False)
    merged = _with_option(_flux(), merge_variants=True)
    assert not _merged(plain)
    assert _merged(merged)
    assert len(merged.get_kernel()) < len(plain.get_kernel())


def test_by_default_a_kernel_that_fits_the_cache_is_written_out():
    """`auto` weighs the kernel written out against the instruction cache,
    and four small contributions take a sliver of it."""
    assert not _merged(_with_option(_flux()))


def test_by_default_a_kernel_that_crowds_the_cache_is_merged():
    assert _merged(_with_option(_flux(), merge_icache_fraction=1e-6))


def test_auto_merges_nothing_where_the_cache_is_unknown():
    """Intel states no instruction cache, and unknown is not over."""
    assert not _merged(_with_option(_flux(), arch='pvc', backend='oneapi',
                                    merge_icache_fraction=1e-6))


def _counting_builds(monkeypatch):
    from tensorforge.generators.generator import Generator
    builds = []
    real = Generator._generate_bound

    def counting(self):
        builds.append(self)
        return real(self)

    monkeypatch.setattr(Generator, '_generate_bound', counting)
    return builds


def _unnamed(gen):
    """The kernel, without what differs only by how it was asked for: the
    options line and the name derived from the whole text."""
    import re
    return re.sub(r'kernel_[0-9a-f]{16}|// options:.*', '', gen.get_kernel())


# On sm_86 the four contributions lay down 5036 B written out, 3495 B merged,
# and the merged build bounds the written-out size from above at 5469 B --
# so a fraction of the 128 KB cache puts the budget on either side of each.
BETWEEN_AND_FITS = 0.0401       # 5256 B
BETWEEN_AND_OVER = 0.0324       # 4247 B


def test_a_kernel_under_the_upper_bound_is_built_as_given(monkeypatch):
    """The probe merges everything, and the size it bounds from above fits:
    the list is built as given, once more."""
    builds = _counting_builds(monkeypatch)
    gen = _with_option(_flux())
    assert not _merged(gen)
    assert len(builds) == 2
    assert _unnamed(gen) == _unnamed(_with_option(_flux(),
                                                  merge_variants=False))


def _flux_through_scratch(count=4):
    """Each contribution through a temporary of its own, read in the same
    iteration -- scratch, which the merged run holds once -- and a temporary
    after the run, which the two lists therefore number differently."""
    out = []
    for i in range(count):
        scratch = make(f'scratch{i}', [9, 4], is_tmp=True)
        out.append(gemm(make(f'fPrT{i}', [9, 9]), make('I', [9, 4]), scratch))
        out.append(gemm(make('W', [9, 9]), scratch, make('Q', [9, 4]),
                        add=(i > 0)))
    after = make('after', [9, 4], is_tmp=True)
    out.append(gemm(make('C', [9, 9]), make('Q', [9, 4]), after))
    out.append(gemm(make('D', [9, 9]), after, make('R', [9, 4])))
    return out


def test_the_list_built_after_a_probe_is_named_as_given():
    """The probes build on the caller's tensors, and a merged probe numbers
    the temporaries of a shorter list: what is built written out afterwards
    is named for its own list, as a build without a probe names it."""
    gen = _with_option(_flux_through_scratch())
    assert not _merged(gen)
    plain = _with_option(_flux_through_scratch(), merge_variants=False)
    assert _unnamed(gen) == _unnamed(plain)


def test_a_kernel_over_the_lower_bound_is_the_probe(monkeypatch):
    """What the merged probe laid down is already over, and every run is
    needed: the probe is the build."""
    builds = _counting_builds(monkeypatch)
    gen = _with_option(_flux(), merge_icache_fraction=1e-6)
    assert _merged(gen)
    assert len(builds) == 1


@pytest.mark.parametrize('fraction,merged', [(BETWEEN_AND_FITS, False),
                                             (BETWEEN_AND_OVER, True)])
def test_between_the_bounds_the_written_out_size_decides(fraction, merged,
                                                         monkeypatch):
    """Between the bounds only the list written out says, so it is built --
    and then one of the two probes is the kernel, whichever side it fell."""
    builds = _counting_builds(monkeypatch)
    gen = _with_option(_flux(), merge_icache_fraction=fraction)
    assert _merged(gen) == merged
    assert len(builds) == 2
    assert _unnamed(gen) == _unnamed(_with_option(_flux(),
                                                  merge_variants=merged))


def test_a_generator_that_hands_its_build_on_keeps_none_of_its_own():
    """Its sections and scopes would be an empty build, read off it for the
    kernel's; the kernel's are those of the generator that built it."""
    gen = _with_option(_flux(), merge_icache_fraction=1e-6)
    assert gen.built is not gen
    assert gen._sections is None and gen._scopes is None
    assert gen.built._sections and _merged(gen)


def test_a_written_out_probe_is_taken_over_only_where_it_is_the_build(
        monkeypatch):
    """What is taken over is what building it again gives, which a generator
    told to emit loops does not get from a probe that was not."""
    from tensorforge.common.context import Context, Options
    from tensorforge.generators.generator import Generator

    builds = _counting_builds(monkeypatch)
    taken = _with_option(_flux(), merge_icache_fraction=BETWEEN_AND_FITS)
    builds.clear()
    again = Generator(_flux(), Context(
        arch='sm_86', backend='cuda', fp_type=DTYPE,
        options=Options(merge_icache_fraction=BETWEEN_AND_FITS)))
    again._emit_loops = True       # told to emit loops: the probe is not taken
    again.generate()
    assert len(builds) == 3
    assert again.get_kernel() == taken.get_kernel()


def test_a_sibling_builds_what_its_generator_would():
    """The probes and the merged build are siblings (`Generator._sibling`),
    and what was settled after construction goes with them: a pinned name,
    the configuration a tuner picked, the geometry, the attributes."""
    from tensorforge.common.context import Context
    from tensorforge.generators.generator import Generator
    from tensorforge.generators.lanes import LaneConfig

    lanes = LaneConfig(32, 32, 1)
    gen = Generator(_flux(), Context(arch='sm_86', backend='cuda',
                                     fp_type=DTYPE), lanes=lanes, attrs={})
    gen.set_kernel_name('kernel_pinned')
    gen._tuned = object()
    sibling = gen._sibling()
    assert sibling._lanes == lanes and sibling._attrs == {}
    assert sibling._base_kernel_name == 'kernel_pinned'
    assert sibling.tuned is gen.tuned
    assert sibling._thread_block_policy_type is gen._thread_block_policy_type
    # and it settles nothing again
    assert not sibling._may_tune and sibling._merge_decided
    assert not sibling._announce_identity


def _accumulate_then_read():
    import importlib.util
    from pathlib import Path
    path = Path(__file__).parent / 'cases' / 'accumulate_then_read.py'
    spec = importlib.util.spec_from_file_location('accumulate_then_read', path)
    case = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(case)
    return case


@pytest.mark.parametrize('arch', ['sm_70', 'sm_86'])
def test_a_merged_accumulation_is_read_back_after_the_loop(arch):
    """`D += W_i Q B_i` three times, merged, then `O = D C`.

    A transfer moved up across the loop -- the final product's read of `D`
    issued ahead of the accumulation into it -- would compute every row of
    `O` from a `D` two terms short.  The merged build is checked
    on its own against numpy, since it names its buffers differently."""
    import numpy as np

    from tensorforge.reference import kernel_eval
    case = _accumulate_then_read()
    descrs = case.descr_list()
    gen = _with_option(descrs, arch=arch, merge_variants=True)
    assert _merged(gen)
    tensors = {}
    for d in descrs:
        for m in d.matrix_list():
            t = getattr(m, 'tensor', m)
            tensors.setdefault(t.alias, t)
    lanes, mults = kernel_eval.launch_geometry(gen.get_launcher())
    mem = kernel_eval.evaluate_wave(gen.get_kernel(), lanes, seed=11,
                                    globals_only=True, mults=mults)

    def read(alias, rows, cols):
        return np.array([[mem.get((tensors[alias].name, i + rows * j), np.nan)
                          for j in range(cols)] for i in range(rows)])

    m, n, half = case.M, case.N, case.HALF
    q = read('Q', m, n)
    d = q @ read('S', n, n)
    for i in range(case.TERMS):
        d[:half] += read(f'W{i}', half, m) @ (q @ read(f'B{i}', n, n))
    assert np.allclose(read('D', m, n), d, rtol=1e-4, atol=1e-3)
    assert np.allclose(read('O', m, n), d @ read('C', n, n),
                       rtol=1e-4, atol=1e-2)


def _written_by_alias(gen, descrs, numbering):
    """What the kernel leaves in the operands it writes, on the host oracle,
    keyed by alias and element -- with every operand given its inputs by
    alias, so that two builds naming their parameters differently read the
    same numbers.  `numbering` is shared between the builds compared."""
    import re

    from tensorforge.reference import kernel_eval
    names = {}
    for outer in descrs:
        for d in outer.operations():
            for m in d.matrix_list():
                t = getattr(m, 'tensor', m)
                if not getattr(t, 'is_variant', False):
                    names.setdefault(t.name, t.alias)
    canonical = {name: f'm{900 + numbering.setdefault(alias, len(numbering))}'
                 for name, alias in names.items()}
    src = re.sub(r'\bm\d+\b', lambda m: canonical.get(m.group(0), m.group(0)),
                 gen.get_kernel())
    signature = re.search(r'\bkernel_\w+\(([^)]*)\)', src).group(1)
    outputs = {p.split()[-1] for p in signature.split(',')
               if '*' in p and not p.strip().startswith('const')
               and '<const ' not in p}
    lanes, mults = kernel_eval.launch_geometry(gen.get_launcher())
    mem = kernel_eval.evaluate_wave(src, lanes, seed=29, globals_only=True,
                                    mults=mults, block=True)
    alias = {v: names[k] for k, v in canonical.items()}
    return {(alias[k[0]], k[1]): v for k, v in mem.items() if k[0] in outputs}


def _staged(count=4):
    """`Q += K_i I` over batch-constant operators, which `stage_members`
    stages a block barrier apart in the loop, and `O = I S` behind it."""
    operator = Addressing.NONE
    out = []
    for i in range(count):
        k = Tensor([24, 24], operator, BoundingBox([0, 0], [24, 24]),
                   alias=f'K{i}', datatype=DTYPE)
        out.append(gemm(SubTensor(k), make('I', [24, 5]), make('Q', [24, 5]),
                        add=True))
    s = Tensor([5, 5], operator, BoundingBox([0, 0], [5, 5]), alias='S',
               datatype=DTYPE)
    return out + [gemm(make('I', [24, 5]), SubTensor(s), make('O', [24, 5]))]


def _scratch(count=4):
    """`t_i = I A_i` and `Q += P_i t_i`, a temporary assigned afresh on every
    iteration and added onto the destination, as SeisSol's `localFluxAll`
    does."""
    out = []
    for i in range(count):
        scratch = make(f'scratch{i}', [24, 5], is_tmp=True)
        out.append(gemm(make('I', [24, 5]), make(f'A{i}', [5, 5]), scratch))
        out.append(gemm(make(f'P{i}', [24, 24]), scratch, make('Q', [24, 5]),
                        add=True))
    return out


def _started(terms=4, twice=False):
    """`Q = L_0 C_0` and `Q += L_k C_k` behind it: an accumulator the list
    begins ahead of the run, as SeisSol's Taylor expansion sums its first
    term there.  `twice` sums `R = M_0 C_0 + ...` beside it, as the
    anelastic time derivative carries two levels."""
    out = [gemm(make('L0', [24, 5]), make('C0', [5, 5]), make('Q', [24, 5]))]
    if twice:
        out.append(gemm(make('M0', [24, 5]), make('D0', [5, 3]),
                        make('R', [24, 3])))
    for k in range(1, terms):
        out.append(gemm(make(f'L{k}', [24, 5]), make(f'C{k}', [5, 5]),
                        make('Q', [24, 5]), add=True))
        if twice:
            out.append(gemm(make(f'M{k}', [24, 5]), make(f'D{k}', [5, 3]),
                            make('R', [24, 3]), add=True))
    return out


def _own(terms=4):
    """`T = L_k C` and `A += T R_k`, with `A = T R_0` ahead of the run and
    `Q = W A` behind it: a temporary each lane adds its own rows onto, which
    the product behind the run reads across the lanes -- in shared memory,
    then, as the temporaries of SeisSol's free-surface kernel are."""
    out = [gemm(make('L0', [24, 9]), make('C', [9, 1]),
                make('T', [24, 1], True)),
           gemm(make('T', [24, 1], True), make('R0', [1, 3]),
                make('A', [24, 3], True))]
    for k in range(1, terms):
        out.append(gemm(make(f'L{k}', [24, 9]), make('C', [9, 1]),
                        make('T', [24, 1], True)))
        out.append(gemm(make('T', [24, 1], True), make(f'R{k}', [1, 3]),
                        make('A', [24, 3], True), add=True))
    return out + [gemm(make('W', [16, 24]), make('A', [24, 3], True),
                       make('Q', [16, 3]))]


def _argument_chain(steps=3):
    """`L(k + 1) = A L(k)` and `Q += L(k + 1) C(k)`, with the levels the
    kernel's own arguments: the shape of SeisSol's anelastic time derivative,
    whose levels are its outputs.  Rolled, one hole reads `L(0..steps-1)` and
    another writes `L(1..steps)` -- one family of members, shifted by a
    trip."""
    out = []
    for k in range(steps):
        out.append(gemm(make('A', [24, 24]), make(f'L{k}', [24, 5]),
                        make(f'L{k + 1}', [24, 5])))
        out.append(gemm(make(f'L{k + 1}', [24, 5]), make(f'C{k}', [5, 5]),
                        make('Q', [24, 5]), add=True))
    return out


def _run_body(gen):
    """The statements of the merged run's loop, nested ones included."""
    from tensorforge.backend.pir.core import Op, walk_stmts
    loop = next(s for s in walk_stmts(gen._sections[0].body)
                if s.op is Op.FOR and s.attr('rolled'))
    return loop, list(walk_stmts(loop.regions[0].body))


def test_holes_that_share_members_are_one_buffer_to_the_passes():
    """The hole a trip reads and the hole it writes select from members they
    share, so a write through one and a read through the other may meet --
    on this trip's pointers or on the next trip's.  Their accesses are
    recorded against one base (`pir.AliasClass`), and neither pointer
    promises `restrict`: the one fetched for the next trip reaches what this
    trip writes."""
    from tensorforge.backend.pir.core import AliasClass, MemSpace
    from tensorforge.common.basic_types import GeneralLexicon
    gen = _merged_build(roll(_argument_chain()))
    _, body = _run_body(gen)
    bases = {id(a.base): a.base for s in body for a in s.accesses
             if a.space is MemSpace.GLOBAL and isinstance(a.base, AliasClass)}
    family, = bases.values()
    reads = [s for s in body for a in s.accesses
             if a.base is family and not a.writes]
    writes = [s for s in body for a in s.accesses
              if a.base is family and a.writes]
    assert reads and writes
    # The levels both holes reach, as the bindings the tables hold.
    shared = {GeneralLexicon.GLOBAL_MEM_PREFIX
              + gen._scopes.get_symbol(make(f'L{k}', [24, 5]).tensor).name
              for k in range(1, 3)}
    assert {m.name for m in family.members} >= shared
    declared = [s.attr('decl') for s in body if s.attr('decl')
                and any(a.base is family for a in s.accesses)]
    assert declared and not any('__restrict__' in d for d in declared)


def test_holes_that_only_read_shared_members_stay_apart():
    """Two holes that read overlapping members and write nothing conflict
    nowhere, and one buffer would only cost them their `restrict`."""
    from tensorforge.backend.pir.core import AliasClass
    descrs = []
    for k in range(3):
        descrs.append(gemm(make(f'x{k}', [24, 5]), make('C', [5, 5]),
                           make('Q', [24, 5]), add=True))
        descrs.append(gemm(make('B', [24, 24]), make(f'x{k + 1}', [24, 5]),
                           make('R', [24, 5]), add=True))
    gen = _merged_build(roll(descrs))
    loop, = [i for i in gen._sections[0].ir
             if type(i).__name__ == 'VariantLoop']
    reached = [{id(m) for m in t.reachable()} for t in loop.tables]
    assert any(a & b for i, a in enumerate(reached) for b in reached[i + 1:]), \
        'two tables that share members'
    _, body = _run_body(gen)
    assert not any(isinstance(a.base, AliasClass)
                   for s in body for a in s.accesses)


def test_the_next_trips_level_is_fetched_behind_this_trips_write_of_it():
    """Issued a trip ahead (`pir.wrap`), the copy of the level the next trip
    reads stands behind the store this trip makes of that level -- however
    early its buffer is free.  A batch loop's elements own their data, so
    nothing of the kind holds there; a counted loop's trips do not."""
    from tensorforge.backend.pir.core import AliasClass, MemSpace, walk_stmts
    for early in (True, False):
        gen = _merged_build(roll(_argument_chain()), enable_wrap_loads=True,
                            wrap_early=early)
        loop, body = _run_body(gen)
        # The pointers the wrap computed again for the next trip.
        ahead = {s.target[0].id for s in body if s.target
                 and str(s.attr('extern') or '').startswith('wrap_glb')}
        assert ahead, 'nothing issued a trip ahead'
        top = list(loop.regions[0].body)

        def holds(s, test):
            return any(test(x) for x in walk_stmts((s,)))

        def writes_family(x):
            return any(isinstance(a.base, AliasClass) and a.writes
                       and a.space is MemSpace.GLOBAL for a in x.accesses)

        def reads_ahead(x):
            return (any(v.id in ahead for v in x.operands())
                    and any(isinstance(a.base, AliasClass) and not a.writes
                            for a in x.accesses))
        written = max(i for i, s in enumerate(top) if holds(s, writes_family))
        fetched = [i for i, s in enumerate(top) if holds(s, reads_ahead)]
        assert fetched, 'nothing read through the next trip\'s pointer'
        assert min(fetched) > written, (early, min(fetched), written)


def test_a_window_the_loop_fills_is_declared_ahead_of_it():
    """A barrier in the run keeps the staging of what every iteration shares
    in the body (`pir.move`), and the operation behind the loop reads the
    staged copy: its window is declared ahead of the header, or the name
    would be scoped to the loop (`VariantLoop._declare_windows_early`)."""
    import re
    gen = _merged_build(roll(_staged()), stage_members=True)
    _, loop, behind = _around_the_loop(gen)
    staged = [i.get_dest() for i in loop.region
              if type(i).__name__ == 'GlbToShrLoader'
              and not getattr(i.get_dest(), 'block_shared', False)]
    window, = staged
    assert any(window in i.uses() for i in behind)
    lines = gen.get_kernel().splitlines()
    header = next(k for k, line in enumerate(lines)
                  if 'for (int32_t batchIdv' in line)
    declared = next(k for k, line in enumerate(lines)
                    if re.search(rf'\b{window.name} = &', line))
    assert declared < header
    assert any(window.name in line for line in lines[header + 1:]
               if 'load{g>s}' in line), 'staged in the loop all the same'


@pytest.mark.parametrize('shape, atomics, options', [
    ('accumulation', False, {}),
    ('accumulation', True, {}),
    ('read_first', False, {}),
    ('staged', False, {'stage_members': True}),
    ('scratch', False, {}),
    ('scratch', True, {}),
    ('carried', False, {}),
    ('chain', False, {}),
    ('chain', False, {'wrap_early': False}),
    ('chain', False, {'enable_wrap_loads': False}),
    ('accumulation', False, {'closed_trips': True}),
    ('accumulation', True, {'closed_trips': True}),
    ('accumulation', False, {'closed_trips': True, 'carry_trips': False}),
    ('accumulation', True, {'closed_trips': True, 'carry_trips': False}),
    ('read_first', False, {'closed_trips': True}),
    ('staged', False, {'stage_members': True, 'closed_trips': True}),
    ('scratch', False, {'closed_trips': True}),
    ('scratch', True, {'closed_trips': True}),
    ('carried', False, {'closed_trips': True}),
    ('chain', False, {'closed_trips': True}),
    ('started', False, {}),
    ('started', True, {}),
    ('started', False, {'closed_trips': True}),
    ('started', True, {'closed_trips': True}),
    ('started_twice', False, {}),
    ('started_twice', False, {'closed_trips': True}),
    ('started_twice', True, {'closed_trips': True}),
    ('read_first_temporary', False, {}),
    ('read_first_temporary', False, {'closed_trips': True}),
    ('read_by_the_next_trip', False, {}),
    ('read_by_the_next_trip', False, {'closed_trips': True}),
    ('read_by_the_next_trip', False,
     {'closed_trips': True, 'carry_trips': False}),
    ('projected', False, {}),
    ('projected', False, {'closed_trips': True}),
    ('carried', False, {'closed_trips': True, 'rotate_slots': 1}),
    ('carried_from_ahead', False, {}),
    ('carried_from_ahead', False, {'rotate_slots': 1}),
    ('carried_from_ahead', False, {'closed_trips': True}),
    ('carried_from_ahead', False, {'closed_trips': True, 'rotate_slots': 1}),
    ('own', False, {}),
    ('own', False, {'closed_trips': True}),
])
def test_a_merged_run_computes_what_the_list_written_out_does(shape, atomics,
                                                              options):
    """The run built as one loop leaves the numbers the operations built one
    by one do -- with the destination made resident ahead of the loop, from
    zeros where the policy adds atomically, and with the first iteration
    built beside the loop where it cannot be (`read_first`; a destination
    written in slices is checked against numpy above).  `staged` keeps the
    staging of what every iteration shares in the loop -- a barrier stands
    there -- and reads it again behind the loop.  `scratch` assigns a
    temporary afresh on every iteration and adds it onto the destination, as
    SeisSol's `localFluxAll` does (`t = I A(i)`, `Q += P(i) t`): the registers
    of the temporary are allocated ahead of the loop, and every trip has to
    start it from zeros again (`Generator._reset_temporaries`).  `chain`
    carries its levels through the kernel's own arguments, as SeisSol's
    anelastic time derivative does: the hole a trip writes is the hole the
    next one reads, and the read of the next trip's level, issued a trip
    ahead (`pir.wrap`), has to wait for this trip's write of it.  And all of
    them again with the trips closed (`Options.closed_trips`), where nothing
    but memory crosses the back edge and the carry takes the accumulator
    back into registers (`pir.carry`) -- or, without it, leaves it going
    through memory on every trip."""
    from dataclasses import replace

    import numpy as np

    from tensorforge.common import target

    def descrs():
        # Small, since the oracle interprets every lane of the block.
        make.pool = {}
        if shape == 'accumulation':
            return [gemm(make(f'P{i}', [24, 24]), make('I', [24, 5]),
                         make('Q', [24, 5]), add=True) for i in range(4)]
        if shape == 'read_first':
            return _read_first()
        if shape in ('carried', 'carried_from_ahead'):
            # A chain through temporaries: each step reads what the last
            # wrote, which is what the rotation serves
            # (`Generator._rotate_members`).  The oracle is the same list
            # written out, so a slot bound to the wrong member, or one
            # overwritten before its reader, shows up as numbers and not as
            # a build that happens to succeed.  `carried_from_ahead` begins
            # the chain with a step of another shape, ahead of the run, which
            # leaves the run's first member in registers.
            out = []
            if shape == 'carried_from_ahead':
                out.append(gemm(make('B', [24, 16]), make('carry', [16, 5]),
                                make('carry0', [24, 5], is_tmp=True)))
            for i in range(4):
                out.append(gemm(make(f'A{i}', [24, 24]),
                                make(f'carry{i}', [24, 5],
                                     is_tmp=i > 0 or shape != 'carried'),
                                make(f'carry{i + 1}', [24, 5], is_tmp=True)))
            out.append(gemm(make('P', [24, 24]), make('carry4', [24, 5], True),
                            make('Q', [24, 5])))
            return out
        if shape == 'scratch':
            return _scratch()
        if shape == 'started':
            return _started()
        if shape == 'started_twice':
            return _started(twice=True)
        if shape == 'read_first_temporary':
            return _read_first_temporary()
        if shape == 'read_by_the_next_trip':
            return _read_first_temporary(behind=False)
        if shape == 'projected':
            return _projected()
        if shape == 'chain':
            return _argument_chain()
        if shape == 'own':
            return _own()
        return _staged()

    base = target.PREFERENCES['nvidia']
    target.PREFERENCES['nvidia'] = replace(
        base, placement=replace(base.placement, atomic_accumulation=atomics))
    try:
        numbering = {}
        plain = descrs()
        expected = _written_by_alias(_with_option(plain, merge_variants=False,
                                                  **options),
                                     plain, numbering)
        merged = roll(descrs())
        gen = _merged_build(merged, **options)
        assert any(isinstance(d, ForDescr) for d in merged)
        got = _written_by_alias(gen, merged, numbering)
    finally:
        target.PREFERENCES['nvidia'] = base
    assert expected and set(got) == set(expected)
    bad = [k for k in expected
           if not np.isclose(got[k], expected[k], rtol=1e-4, atol=1e-3)]
    assert not bad, f'{len(bad)} of {len(expected)} differ, e.g. {bad[:3]}'


@pytest.mark.parametrize('wrap', [False, True])
def test_a_merged_loop_fences_its_head_against_its_tail(wrap):
    """The staged operand reuses the window the product read the iteration
    before, and nothing appends a barrier to this loop the way the generator
    does to a batch loop: the barrier placement has to see the back edge
    (`pir.barriers`).  Copied at the head, the copy waits for the reads at
    the tail of the trip before; copied at the tail for the next trip
    (`wrap`), it waits for this trip's reads, and the reads at the head wait
    for it."""
    from tensorforge.backend.pir.core import Effect, MemSpace, Op
    gen = _with_option(_accumulate_then_read().descr_list(), arch='sm_70',
                       merge_variants=True, enable_wrap_loads=wrap)
    loop, = [i for i in gen._sections[0].ir
             if type(i).__name__ == 'VariantLoop']

    def stmts(body):
        for st in body:
            yield st
            for r in st.regions:
                yield from stmts(r.body)
    run = next(st for st in stmts(gen._sections[0].body)
               if st.op == Op.FOR and st.attr('extern') == loop.counter)
    body = list(stmts(run.regions[0].body))

    staged = {id(a.base) for st in body if st.op is Op.COPY_ASYNC
              for a in st.accesses
              if a.space == MemSpace.SHARED and a.kind & Effect.WRITE}

    def touching(kind):
        # the wait retires a copy; the copy is where the bytes are written
        return [k for k, st in enumerate(body)
                if st.op not in (Op.ALLOC, Op.MARK, Op.WAIT) and any(
                    a.space == MemSpace.SHARED and a.kind & kind
                    and id(a.base) in staged for a in st.accesses)]
    barriers = [k for k, st in enumerate(body) if st.op == Op.BARRIER]
    assert any(k < touching(Effect.WRITE)[0] for k in barriers)
    if wrap:
        assert touching(Effect.WRITE)[0] > touching(Effect.READ)[-1]
        assert any(k < touching(Effect.READ)[0] for k in barriers)


def test_the_probe_leaves_nothing_behind():
    """The unmerged build that measures the code is a probe: the counters
    the rebuild reports are its own."""
    merged = _with_option(_flux(), merge_icache_fraction=1e-6)
    explicit = _with_option(_flux(), merge_variants=True)
    assert merged.metrics.code_units == explicit.metrics.code_units


def test_a_pair_is_not_a_run_by_default():
    """Two contributions are cheaper written out than a counter and a select."""
    merged = _with_option(_flux(3), merge_variants=True)
    assert not any(type(i).__name__ == 'VariantLoop'
                   for i in merged._sections[0].ir)


def test_the_minimum_is_the_caller_s_to_lower():
    merged = _with_option(_flux(3), merge_variants=True, merge_min_count=2)
    assert any(type(i).__name__ == 'VariantLoop'
               for i in merged._sections[0].ir)


def test_a_stand_in_reads_what_its_members_store():
    """The body reads every member through the stand-in, so the stand-in has
    to carry the members' storage: a member held as two TF32 halves per
    element, read through a stand-in that says one, is a wrong kernel.  The
    order is left to the instruction that decides it for all of them."""
    from tensorforge.generators.rolling import _stand_in
    t = Tensor([4, 4], Addressing.NONE, BoundingBox([0, 0], [4, 4]),
               alias='Asplit', datatype=DTYPE)
    t.storage_parts = 2
    clone = _stand_in(SubTensor(t), 'variant0').tensor
    assert clone.is_variant
    assert clone.storage_parts == 2
    assert clone.storage_order is None


def test_a_factor_the_loop_binds_is_read_inside_it():
    """A scalar hole is read where its name exists, not before the loop.

    The body reads a factor passed by value, and the loop binds that name from
    the table its counter selects.  Read as a bare name the read is a value
    with no operands, which is what `licm` calls loop-invariant -- so it could
    be hoisted past the header, out of the scope the name is declared in, and
    the kernel would stop compiling rather than slow down.

    Stated on the emitted text because that is where it goes wrong: a name
    used before its declaration is a C++ error and nothing earlier in the
    pipeline has an opinion about it.  SeisSol's damage step has this shape in
    a long run; this is the same shape in six.
    """
    import re
    scalar = Addressing.SCALAR

    def number(alias):
        return SubTensor(Tensor([], scalar, BoundingBox([], []),
                                alias=alias, datatype=DTYPE))

    from tensorforge.generators.descriptions import MultilinearDescr
    acc = make('acc', [])
    descrs = []
    for k in range(6):
        # A destination without axes: read as a value, not folded into a
        # product's text, which is the read that has to stay inside.  The
        # contraction beside it is what gives the body something to hoist
        # past, which is how the read leaves the loop at all.
        descrs.append(MultilinearDescr(dest=acc,
                                       ops=[make(f'u{k}', []),
                                            number(f's{k}')],
                                       target=[[], []], permute=[[], []],
                                       add=k > 0))
        descrs.append(gemm(make(f'A{k}', [56, 56]), make('I', [56, 9]),
                           make('Q', [56, 9]), add=True))
    text = _generated(roll(descrs, min_count=3))
    declared, used = {}, {}
    for i, line in enumerate(text.splitlines(), 1):
        # A comment names what it describes, which is no use of the name: a
        # wait keeps the line naming the copy it retires after the copy has
        # moved to a binding of its own.
        line = line.split('//', 1)[0]
        for m in re.finditer(r'\bglb_(\w+)\b', line):
            if re.search(rf'(float|double)[^;=]*\bglb_{m.group(1)}\b\s*=', line):
                declared.setdefault(m.group(1), i)
            else:
                used.setdefault(m.group(1), []).append(i)
    early = {n: (used[n][0], declared.get(n)) for n in used
             if n not in declared or used[n][0] < declared[n]}
    assert not early, f'read before it is bound: {early}'


def test_a_run_reads_its_table_and_its_factor_inside_the_loop():
    """A merged run over ten scalars: nothing it binds is read before the loop.

    Two reads could leave the loop, and neither is the one the test beside
    this covers.  The table a counter selects from is an array once the run is
    longer than a chain is worth, and a binding that names the counter in text
    alone has no operand that varies, so `licm` would move it ahead of the
    header.  The factor's own read is the same shape one step later: a
    stand-in read as a bare name is a value with no operands too, and a
    multilinear that takes it as an epilogue factor would have it hoisted out
    from under the loop that declares it.

    Ten iterations because eight is where a select chain gives way to an
    array, and a destination with an axis because a factor beside a rank-0
    destination is read by the scalar contraction instead.
    """
    import re
    from tensorforge.generators.descriptions import MultilinearDescr

    def number(alias):
        return SubTensor(Tensor([], Addressing.SCALAR, BoundingBox([], []),
                                alias=alias, datatype=DTYPE))

    descrs = []
    for k in range(10):
        descrs.append(MultilinearDescr(dest=make('acc', [56]),
                                       ops=[make(f'b{k}', [56]),
                                            number(f's{k}')],
                                       target=[[0], []], permute=[[0], []],
                                       add=k > 0))
        descrs.append(gemm(make(f'A{k}', [56, 56]), make('I', [56, 9]),
                           make('Q', [56, 9]), add=True))
    text = _generated(roll(descrs, min_count=3))

    lines = text.splitlines()
    counters = {}
    for i, line in enumerate(lines, 1):
        for m in re.finditer(r'for \(int32_t (batchIdv\d+) =', line):
            counters[m.group(1)] = i
    assert counters, 'expected the run to be emitted as a loop'
    early = []
    for i, line in enumerate(lines, 1):
        if re.search(r'for \(int32_t batchIdv\d+ =', line):
            continue
        for m in re.finditer(r'\bbatchIdv\d+\b', line):
            if i < counters.get(m.group(0), 0) or m.group(0) not in counters:
                early.append((i, m.group(0), line.strip()[:70]))
    assert not early, f'counter read outside its loop: {early[:3]}'

    declared, used = {}, {}
    for i, line in enumerate(lines, 1):
        for m in re.finditer(r'\bglb_(\w+)\b', line):
            if re.search(rf'(float|double)[^;=]*\bglb_{m.group(1)}\b\s*=', line):
                declared.setdefault(m.group(1), i)
            else:
                used.setdefault(m.group(1), []).append(i)
    early = {n: (used[n][0], declared.get(n)) for n in used
             if n in declared and used[n][0] < declared[n]}
    assert not early, f'read before it is bound: {early}'


# -- closed trips (`Options.closed_trips`) and the carry (`pir.carry`) ------- #

def _around_the_run(gen):
    """`(ahead, loop, behind)` in the section's pseudo-IR: the statements of
    the list the merged run's loop stands in, on either side of it."""
    from tensorforge.backend.pir.core import Op

    def find(stmts):
        for i, s in enumerate(stmts):
            if s.op is Op.FOR and s.attr('rolled'):
                return list(stmts[:i]), s, list(stmts[i + 1:])
            for r in s.regions:
                found = find(r.body)
                if found is not None:
                    return found
        return None
    return find(gen._sections[0].body)


def _output(gen):
    """The one buffer in global memory the section writes: `Q` below."""
    from tensorforge.backend.pir.core import MemSpace, walk_stmts
    bases = {id(a.base): a.base for s in walk_stmts(gen._sections[0].body)
             for a in s.accesses if a.space is MemSpace.GLOBAL and a.writes}
    base, = bases.values()
    return base


def _images(stmts, buffer, loading):
    """The register images the statements of `stmts` move `buffer` through:
    filled from it where `loading`, else stored into it."""
    from tensorforge.backend.pir.core import MemSpace, collect_accesses
    out = set()
    for s in stmts:
        found = collect_accesses((s,))
        if any(a.base is buffer and a.writes != loading for a in found):
            out |= {id(a.base) for a in found
                    if a.space is MemSpace.REGISTER and a.writes == loading}
    return out


def test_a_closed_trip_hands_the_next_one_nothing_but_memory():
    """With its trips closed, the loop runs every iteration, its body
    declares its own registers, and no register is threaded through the
    trips or set back on each.  Without the carry the accumulator goes to
    memory at the tail of every trip and comes back at the head of the
    next."""
    from tensorforge.backend.pir.core import walk_stmts
    gen = _merged_build(roll(accumulation()), closed_trips=True,
                        carry_trips=False)
    ahead, loop, _ = _around_the_loop(gen)
    assert (loop.start, loop.carried) == (0, ())
    assert 'RegisterAlloc' not in _kinds(ahead)
    assert 'RegisterReset' not in _kinds(loop.region)
    q = _output(gen)
    _, run, _ = _around_the_run(gen)
    touched = [a for s in walk_stmts(run.regions[0].body)
               for a in s.accesses if a.base is q]
    assert any(a.writes for a in touched)
    assert any(not a.writes for a in touched)


def test_the_carry_holds_an_accumulator_in_registers_over_the_trips():
    """What the trips hand each other through memory, the carry hands over in
    registers: `Q` is loaded once ahead of the loop and stored once behind
    it, and the trips add into the image it was loaded into -- one image,
    and no copy between two at the end of each trip."""
    from tensorforge.backend.pir.core import MemSpace, walk_stmts
    gen = _merged_build(roll(accumulation()), closed_trips=True)
    q = _output(gen)
    ahead, run, behind = _around_the_run(gen)
    inside = list(walk_stmts(run.regions[0].body))
    assert not any(a.base is q for s in inside for a in s.accesses)
    loaded = _images(ahead, q, loading=True)
    assert loaded and loaded == _images(behind, q, loading=False)
    assert any(a.space is MemSpace.REGISTER and a.writes
               and id(a.base) in loaded for s in inside for a in s.accesses)


@pytest.mark.parametrize('twice', [False, True])
def test_an_accumulator_begun_ahead_of_the_run_stays_in_its_registers(twice):
    """`Q = L_0 C_0` ahead of the run leaves `Q` in registers, which a closed
    trip finds in memory: so the carry has the load it puts ahead of the
    loop read those registers, and drops the store ahead of it, since the
    store behind the loop writes all of `Q` again (`pir.carry`).  With `R`
    summed beside it, the same for each -- the second load found where the
    first left the statements around it."""
    from tensorforge.backend.pir.core import MemSpace, Op, walk_stmts
    # Without the forwarding: it takes the load ahead of the loop and drops
    # the store ahead on its own (`pir.forward`), and what is asked here is
    # whether the carry does.
    gen = _merged_build(roll(_started(twice=twice)), closed_trips=True,
                        forward_memory=False)
    ahead, _, behind = _around_the_run(gen)
    outputs = {id(a.base): a.base
               for s in walk_stmts(gen._sections[0].body)
               for a in s.accesses if a.space is MemSpace.GLOBAL and a.writes}
    assert len(outputs) == (2 if twice else 1)

    def moved(stmts, buffer):
        return [a for s in walk_stmts(tuple(stmts)) for a in s.accesses
                if s.op in (Op.LOAD, Op.STORE) and a.base is buffer]

    for buffer in outputs.values():
        assert not moved(ahead, buffer), 'stored or loaded ahead'
        assert any(a.writes for a in moved(behind, buffer))


def test_an_atomic_accumulator_is_summed_in_registers_and_added_once():
    """Where the policy adds atomically, the trips are built as everywhere
    else: each loads the accumulator and stores its sum.  The carry keeps it
    in registers over the trips, and the body adds the sum behind the loop
    atomically (`pir.atomics`), from an image that starts as zeros -- the
    order the list written out adds in."""
    from dataclasses import replace

    from tensorforge.backend.pir.core import Effect, Op, walk_stmts
    from tensorforge.common import target
    base = target.PREFERENCES['nvidia']
    target.PREFERENCES['nvidia'] = replace(
        base, placement=replace(base.placement, atomic_accumulation=True))
    try:
        gen = _merged_build(roll(accumulation()), closed_trips=True)
    finally:
        target.PREFERENCES['nvidia'] = base
    q = _output(gen)
    ahead, run, behind = _around_the_run(gen)
    assert not any(a.base is q for s in walk_stmts(run.regions[0].body)
                   for a in s.accesses)
    summed = _images(behind, q, loading=False)
    assert summed
    assert any(a.kind & Effect.ATOMIC for s in walk_stmts(tuple(behind))
               for a in s.accesses if a.base is q)
    declared, = [s for s in ahead if s.op is Op.ALLOC
                 and any(id(t) in summed for t in s.target)]
    assert declared.attr('init') == '{}'


def test_the_carry_leaves_what_a_table_reaches_in_memory():
    """A chain's levels are the kernel's arguments, written through one hole
    and read through another (`pir.AliasClass`): a trip's level is no other
    trip's, so none of them is carried, and the loop goes on writing them.
    The accumulator it adds them into is carried all the same: written
    behind the loop, and nowhere in it."""
    from tensorforge.backend.pir.core import (AliasClass, MemSpace,
                                              walk_stmts)
    gen = _merged_build(roll(_argument_chain()), closed_trips=True)
    _, run, behind = _around_the_run(gen)

    def written(stmts):
        return {id(a.base): a.base for s in walk_stmts(tuple(stmts))
                for a in s.accesses
                if a.space is MemSpace.GLOBAL and a.writes}
    inside = written(run.regions[0].body)
    assert any(isinstance(b, AliasClass) for b in inside.values())
    after = written(behind)
    assert after and not set(after) & set(inside)


def _projected(count=4):
    """`T = X_0 A_i + X_1 B_i`, `U = T W` and `O_i = V U`: temporaries a trip
    computes and uses itself, as SeisSol's `projectToDR` sums the time
    derivatives of an element before it projects them."""
    out = []
    for i in range(count):
        out.append(gemm(make('X0', [24, 9]), make(f'A{i}', [9, 9]),
                        make('T', [24, 9], True)))
        out.append(gemm(make('X1', [24, 9]), make(f'B{i}', [9, 9]),
                        make('T', [24, 9], True), add=True))
        out.append(gemm(make('T', [24, 9], True), make('W', [9, 9]),
                        make('U', [24, 9], True)))
        out.append(gemm(make('V', [32, 24]), make('U', [24, 9], True),
                        make(f'O{i}', [32, 9])))
    return out


@pytest.mark.parametrize('shape', ['scratch', 'projected'])
def test_a_closed_trip_stores_what_outlives_it_and_nothing_it_used_up(shape):
    """At the end of a closed trip, what the next trip or the code behind the
    loop reads goes to memory, and a temporary the trip computes and uses
    itself does not (`Generator._trip_outlives`): `t_i = I A_i`,
    `Q += P_i t_i` ends its trip storing `Q` and not `t` -- and
    `T = X_0 A_i + X_1 B_i`, `O_i = V T W` its trip storing `O_i` and not
    `T`, which the trip ends with in registers."""
    descrs = _scratch() if shape == 'scratch' else _projected()
    gen = _merged_build(roll(descrs), closed_trips=True)
    _, loop, _ = _around_the_loop(gen)
    tail = []
    for instr in reversed(loop.region):
        if type(instr).__name__ not in ('StoreRegToGlb', 'StoreRegToShr'):
            break
        tail.append(instr)
    assert _kinds(tail) == ['StoreRegToGlb']


def test_a_temporary_each_lane_adds_onto_is_held_in_its_registers():
    """`A += T R_k` adds each lane's rows of `A` onto what the lane itself
    wrote, and `A` is in shared memory for the product behind the run, which
    reads it across the lanes.  With the trips closed, the lanes hold their
    elements in registers over the trips (`pir.carry`): nothing in the loop
    reaches `A`, the code ahead of it hands `A` over in registers instead of
    storing it, and it is stored once behind the loop, for the product."""
    from tensorforge.backend.pir.core import MemSpace, Op, walk_stmts
    gen = _merged_build(roll(_own()), closed_trips=True)
    ahead, run, behind = _around_the_run(gen)

    def stores(stmts):
        return {id(a.base) for s in walk_stmts(tuple(stmts))
                for a in s.accesses if s.op is Op.STORE
                and a.space is MemSpace.SHARED}

    def reached(stmts):
        return {id(a.base) for s in walk_stmts(tuple(stmts))
                for a in s.accesses if s.op in (Op.LOAD, Op.STORE)
                and a.space is MemSpace.SHARED}

    held = stores(behind)
    assert held
    assert not held & reached(run.regions[0].body)
    assert not held & stores(ahead)


def _read_first_temporary(count=4, behind=True):
    """`T_i = W_i S; S += V_i T_i` over a temporary `S` that the list sets
    ahead of the run and reads behind it -- or, without `behind`, that only
    the next trip reads, and the trips leave `O += U T_i`."""
    out = [gemm(make('Y', [9, 9]), make('X', [9, 4]),
                make('S', [9, 4], is_tmp=True))]
    for i in range(count):
        tmp = make('T', [9, 4], is_tmp=True)
        out.append(gemm(make(f'W{i}', [9, 9]), make('S', [9, 4], True), tmp))
        out.append(gemm(make(f'V{i}', [9, 9]), tmp, make('S', [9, 4], True),
                        add=True))
        if not behind:
            out.append(gemm(make('U', [9, 9]), tmp, make('O', [9, 4]),
                            add=True))
    if not behind:
        return out
    return out + [gemm(make('Z', [9, 9]), make('S', [9, 4], True),
                       make('O', [9, 4]))]


@pytest.mark.parametrize('behind', [True, False])
def test_a_closed_trip_stores_a_temporary_the_next_trip_reads(behind):
    """A temporary the trip reads before it writes it is the last trip's, and
    a closed trip finds it in memory: what the trip leaves in `S` is stored
    at its end, on every trip -- and `T`, which the trip computes and uses
    itself, is not.  So it is where nothing behind the loop reads `S`: the
    next trip does."""
    gen = _merged_build(roll(_read_first_temporary(behind=behind)),
                        closed_trips=True, carry_trips=False)
    _, loop, _ = _around_the_loop(gen)
    assert (loop.start, loop.count) == (0, 4)
    tail = []
    for instr in reversed(loop.region):
        if type(instr).__name__ not in ('StoreRegToGlb', 'StoreRegToShr'):
            break
        tail.append(instr)
    shared = [i for i in tail if type(i).__name__ == 'StoreRegToShr']
    assert [i._dest.obj.alias for i in shared] == ['S']
    assert len(tail) == (1 if behind else 2)
