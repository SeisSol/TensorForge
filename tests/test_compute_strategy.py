# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""The compute strategy, asserted from numbers rather than from output.

Which arrangement a contraction is emitted as was four conditions in three
forms spread over the dispatch and the vendor modules, and the only way to ask
what it had decided was to read the generated code.  Written as a function of
the shape it can be asked directly -- which is what these do.

The split is the same one `placement.py` makes.  Legality is about what the
target and the shape allow; preference is about which of the legal answers is
worth taking.  A test that mixes them passes for the wrong reason as soon as
the order changes.
"""

from __future__ import annotations

import pytest

from tensorforge.backend.instructions.compute.strategy import (
    DEFAULT_PREFERENCE, PREFERENCES, ComputeShape, Span, Strategy,
    choose_strategy, is_contraction, lead_layout, legal_strategies)
from tensorforge.backend.instructions.compute import (bitlayout,
                                                      routes,
                                                      staging)
from tensorforge.backend.instructions.compute.bitlayout import Position
from tensorforge.backend.symbol import LeadIndex
from tensorforge.common.context import Context
from tensorforge.backend.instructions.compute.primitives import amd, intel
from tensorforge.backend.instructions.compute.primitives import nvidia
from tensorforge.common.basic_types import Datatype


class _FakeCtx:
    """Enough context for the AMD predicates, which read the model name."""

    class _HW:
        def __init__(self, model):
            self.model = model
            self.vendor = 'amd'

    class _VM:
        def __init__(self, model):
            self._hw = _FakeCtx._HW(model)

        def get_hw_descr(self):
            return self._hw

    def __init__(self, model='gfx90a'):
        self._vm = _FakeCtx._VM(model)

    def get_vm(self):
        return self._vm


def _shape(threads=32, dtype=Datatype.F32, sparse=False, explicit_simd=False):
    return ComputeShape(threads=threads, accumulator=dtype, sparse=sparse,
                        explicit_simd=explicit_simd)


# -- one question, three sets of rungs ------------------------------------- #

def test_a_target_with_no_rungs_gets_the_two_that_are_always_true():
    """What handing in nothing means.  No gap needs no instruction, and the
    trip needs none either -- those are facts about bits.  Everything between
    them is an instruction, and a target that has none is not a target the
    question cannot be asked of."""
    threads = 64
    have, want = lead_layout(threads, 4), lead_layout(threads, 1)
    indices = [(i,) for i in range(threads)]
    assert routes.reach(want, want, threads, indices) == 0
    bare = routes.reach(have, want, threads, indices)
    assert isinstance(bare, tuple) and isinstance(bare[0], staging.Transfer)


def test_the_rungs_are_what_differs_and_not_the_question():
    """AMD reaches this gap in one instruction and the others do not, and
    that is the whole of the difference: same layouts, same call, same index
    space."""
    from tensorforge.backend.instructions.compute.primitives.amd import relayout

    ext, threads = 4, 64
    have = relayout.nest_shared(ext, threads)
    want = relayout.transposed(ext, threads)
    indices = [(c, l) for c in range(ext) for l in range(threads)]
    assert routes.reach(have, want, ext, indices,
                        rungs=relayout.RUNGS) == 1
    bare = routes.reach(have, want, ext, indices)
    assert isinstance(bare, tuple) and isinstance(bare[0], staging.Transfer)


@pytest.mark.parametrize('vendor', ['amd', 'nvidia', 'intel'])
def test_every_target_answers_the_packed_operand_by_its_route(vendor):
    """No target names a width any more.  Each asks what the operand would
    need and whether it writes that, so each lifts by an emitter learning a
    route rather than by a condition being edited."""
    import inspect

    from tensorforge.backend.instructions.compute.primitives import (
        amd, intel, nvidia)
    module = {'amd': amd, 'nvidia': nvidia, 'intel': intel}[vendor]
    source = inspect.getsource(module.strategies)
    assert 'lead_width' not in source
    assert 'lead_route' in source


# -- the layout the plan derives ------------------------------------------- #

def test_the_plan_derives_the_layout_the_emitter_addresses():
    """The point of deriving it here at all.  A distribution has always been
    readable off the access -- `layout_of` does it and `LeadIndex.layout()` is
    what it reads -- but every caller of that sits inside the emission, so by
    the time a layout exists the arrangement is chosen and an operand in the
    wrong one can only be refused.

    So this checks the plan's reading against the address the generator
    writes: for each of the `threads` elements a fragment covers, the layout
    has to name the lane and the component `LeadIndex` puts it at.
    """
    ctx = Context(arch='gfx90a', backend='hip', fp_type=Datatype.F32)
    for threads in (16, 32, 64):
        for width in (1, 2, 4):
            layout = lead_layout(threads, width)
            index = LeadIndex(0, block=threads, stride=1, width=width)
            emitted = index.write(ctx)
            for tid in range(threads):
                base = eval(emitted.replace('threadIdx.x', str(tid))
                            .replace('/', '//'))
                for component in range(width):
                    if base + component >= threads:
                        continue
                    assert layout.locate(base + component) == Position(
                        lane=tid, element=component), (threads, width, tid)


def test_the_fragment_s_index_space_is_the_one_the_operand_is_read_over():
    """Worth pinning because the two candidates differ by `width`, and the
    wrong one is not obviously wrong.  A lead dimension spans `threads *
    width` elements per slot; a fragment covers `threads` of them, one per
    lane.  Describing the slot instead would compare a fragment against
    `width` times as many elements as it has lanes for."""
    packed = lead_layout(threads=32, width=4)
    assert packed.locate(31) == Position(lane=7, element=3)
    assert bitlayout.packed(packed)
    assert not bitlayout.packed(lead_layout(threads=32, width=1))


# -- shape-independent legality -------------------------------------------- #

def test_three_operands_have_no_a_and_b():
    """Every arrangement but the nest names two operands; a longer product
    has no such split to name."""
    assert not is_contraction(operands=3)
    assert is_contraction(operands=2)


def test_the_lead_width_is_not_asked_here():
    """It used to be, and refused every arrangement on every target at once.
    The reasons are not one reason, and a shared refusal cannot lift for one
    target -- so each `strategies` owns its own now, and this asks only what
    is true of the operation."""
    import inspect

    from tensorforge.backend.instructions.compute import strategy
    assert 'lead_width' not in inspect.signature(is_contraction).parameters
    assert 'lead_width' not in inspect.getsource(is_contraction)
    # Nor on the shape the arrangements read: what they ask about a packed
    # operand is "can I take this distribution", and a width is only a
    # number that stands in for one.
    assert 'lead_width' not in ComputeShape.__dataclass_fields__


@pytest.mark.parametrize('vendor', ['amd', 'nvidia', 'intel'])
@pytest.mark.parametrize('width', [2, 4])
def test_a_packed_lead_operand_reaches_only_the_lane_batched_core(
        vendor, width, monkeypatch):
    """Separately and for its own reason: no emitter writes the route from a
    packed operand to a fragment that wants the rows in lane order.  AMD's
    lane-batched MFMA does not want that order -- its lanes are independent
    rows -- and takes the packed operand one component at a time
    (`amd.componentwise`); its DPP chain takes it too, as vectors of rows and
    of contraction steps.  NVIDIA's and Intel's matrix paths want the order,
    and nothing there takes one.

    The deployment switches are turned on for this, or two of the three would
    answer nothing whatever the width and the check would read as coverage.
    """
    from tensorforge.backend.instructions.compute.strategy import Strategy
    from tensorforge.backend.instructions.compute.primitives import (
        amd, intel, nvidia)
    from tensorforge.common.context import Context

    module, arch, backend, threads = {
        'amd': (amd, 'gfx90a', 'hip', 64),
        'nvidia': (nvidia, 'sm_80', 'cuda', 32),
        'intel': (intel, 'pvc', 'esimd', 16)}[vendor]
    for switch in ('ENABLED', 'BROADCAST_ENABLED'):
        if hasattr(module, switch):
            monkeypatch.setattr(module, switch, True)
    ctx = Context(arch=arch, backend=backend, fp_type=Datatype.F32)

    def shape(width):
        return ComputeShape(threads=threads, accumulator=Datatype.F32,
                            sparse=False, explicit_simd=(vendor == 'intel'),
                            lead=threads, depth=32,
                            lead_layout=lead_layout(threads, width))

    assert shape(1) and module.strategies(shape(1), ctx) != frozenset(), (
        'the unpacked shape has to be served, or the refusal below says '
        'nothing about the width')
    packed = module.strategies(shape(width), ctx)
    assert packed == (frozenset({Strategy.MATRIX, Strategy.DPP})
                      if vendor == 'amd' else frozenset())


def test_the_nest_is_always_legal():
    """Which is why every other arrangement is free to decline."""
    assert Strategy.GENERIC in legal_strategies(frozenset())
    assert Strategy.GENERIC in legal_strategies({Strategy.MATRIX})


# -- what each target offers ----------------------------------------------- #

def test_amd_always_offers_the_dpp_chain():
    """A broadcast modifier on the multiply asks nothing of the shape, and
    where the widest form does not link `select.py` narrows rather than
    declines."""
    for dtype in (Datatype.F32, Datatype.F64):
        offered = amd.strategies(_shape(dtype=dtype), _FakeCtx())
        assert Strategy.DPP in offered


def test_amd_offers_a_matrix_core_only_where_a_tile_fits():
    """F64's MFMAs spend two lane bits on the contraction, so the lane-batched
    loop cannot feed them.  The offer follows that structural fact rather than
    a type name -- which is why the F64 answer needs no special case."""
    ctx = _FakeCtx('gfx90a')
    assert Strategy.MATRIX in amd.strategies(_shape(dtype=Datatype.F32), ctx)
    assert Strategy.MATRIX not in amd.strategies(_shape(dtype=Datatype.F64),
                                                 ctx)


def test_amd_declines_a_matrix_core_for_a_sparse_operand():
    """Read by linear index, which no fragment layout accepts.  The DPP chain
    has a branch for it and stays on offer."""
    ctx = _FakeCtx()
    offered = amd.strategies(_shape(sparse=True), ctx)
    assert offered == frozenset({Strategy.DPP})


def test_nvidia_offers_nothing_while_the_switch_is_off():
    """`ENABLED` is a deployment switch, not a capability, and it reads as one
    here: flipping it is the whole of turning the path on."""
    assert not nvidia.ENABLED
    assert nvidia.strategies(_shape(threads=32), None) == frozenset()


def test_intel_offers_the_broadcast_chain_only_under_explicit_simd():
    """A lane broadcast is an element read out of the work-item's own vector
    there and a real cross-lane instruction in SPMD.  The arrangement is built
    entirely out of that operation, so the lowering decides whether it is
    worth offering at all."""
    assert intel.BROADCAST_ENABLED
    simd = intel.strategies(_shape(threads=16, explicit_simd=True), None)
    spmd = intel.strategies(_shape(threads=16, explicit_simd=False), None)
    assert Strategy.BROADCAST in simd
    assert Strategy.BROADCAST not in spmd


def test_intel_offers_nothing_for_a_wave_that_is_not_the_execution_size():
    """`ExecutionSize` is 16 for every atom in the table, and the ESIMD
    lowering makes the vector width the thread count."""
    assert intel.strategies(_shape(threads=32, explicit_simd=True),
                            None) == frozenset()


# -- preference ------------------------------------------------------------ #

def test_a_target_with_no_row_runs_the_nest():
    """Correct and slow, which is the right default: adding a row is the whole
    of enabling a target, and forgetting to costs performance, not answers."""
    assert DEFAULT_PREFERENCE == (Strategy.GENERIC,)
    assert choose_strategy(frozenset({Strategy.MATRIX, Strategy.GENERIC}),
                           'moore') is Strategy.GENERIC


def test_amd_prefers_the_matrix_core_to_the_dpp_chain():
    legal = legal_strategies({Strategy.MATRIX, Strategy.DPP})
    assert choose_strategy(legal, 'amd') is Strategy.MATRIX


def test_amd_falls_to_dpp_when_no_tile_fits():
    legal = legal_strategies({Strategy.DPP})
    assert choose_strategy(legal, 'amd') is Strategy.DPP


def test_preference_never_invents_an_arrangement():
    """Ranking only chooses among what is already legal.  A row naming an
    arrangement the shape excluded has to be skipped, not taken -- otherwise
    the preference table can overrule a correctness condition."""
    for vendor, order in PREFERENCES.items():
        for strategy in order:
            legal = legal_strategies(frozenset())
            chosen = choose_strategy(legal, vendor)
            assert chosen is Strategy.GENERIC, (vendor, strategy)


def test_every_preferred_arrangement_can_be_emitted():
    """A row that names an arrangement no module in that vendor's package
    emits would be chosen and then have nothing to do."""
    modules = {'amd': amd, 'nvidia': nvidia, 'intel': intel}
    for vendor, order in PREFERENCES.items():
        assert Strategy.GENERIC in order, vendor
        assert vendor in modules, vendor
    assert set(PREFERENCES) == set(modules)


@pytest.mark.parametrize('vendor', sorted(PREFERENCES))
def test_the_nest_is_last_in_every_row(vendor):
    """It is legal for every shape, so anything after it is unreachable."""
    order = PREFERENCES[vendor]
    assert order[-1] is Strategy.GENERIC
    assert order.count(Strategy.GENERIC) == 1


# -- the three type roles -------------------------------------------------- #

def test_every_reachable_tile_multiplies_what_it_accumulates():
    """Why one type still reaches `matmul32` where the contract names three.

    A tile is selected by `d.dtype` and multiplies in `a.dtype` and `b.dtype`,
    and for every entry `MFMA_TILES` holds those three coincide -- so passing
    the accumulator down is right rather than lucky.  The moment an entry
    whose fragments differ becomes reachable, this fails, which is the point:
    the emitter would then need the operand types threaded through, and the
    guard in `matmul()` that declines the mismatch would start firing instead
    of standing idle.
    """
    from tensorforge.backend.instructions.compute.primitives.amd import (
        MFMA_TILES)
    mixed = [t.op.builtin for t in MFMA_TILES
             if len({t.op.a.dtype, t.op.b.dtype, t.op.d.dtype}) != 1]
    assert not mixed, mixed


def test_the_accumulator_is_what_legality_matches_on():
    """Stated by `MatrixOp.available_for` and worth pinning here: the split
    paths exist precisely because an entry can be offered for an accumulator
    it does not multiply in."""
    from tensorforge.backend.instructions.compute.primitives.amd import ops_for
    from tensorforge.common.context import Context
    ctx = Context(arch='gfx90a', backend='hip', fp_type=Datatype.F32)
    for op in ops_for(Datatype.F32, ctx):
        assert op.d.dtype is Datatype.F32, op.builtin


def test_intel_declines_operands_its_split_cannot_take():
    """The atom is picked by the accumulator, so nothing upstream has looked
    at what the operands arrive as; `splitFloatTF32` takes an F32 apart and
    would otherwise be handed something else."""
    from tensorforge.backend.instructions.compute.matmul import MatmulOperands
    ops = MatmulOperands(A=None, B=None, C=None, sparse=None,
                         lead_slots=1, lead_elements=16, n=1, k=1, kx=0,
                         threads=16, a=Datatype.F64, b=Datatype.F64,
                         accumulator=Datatype.F32)
    assert intel.matmul(None, ops, None,
                        Span(Strategy.MATRIX, 0, 1)) is False


# -- choosing among the matrix schemes ------------------------------------- #

def _hip(arch='gfx90a', dtype=Datatype.F32):
    from tensorforge.common.context import Context
    return Context(arch=arch, backend='hip', fp_type=dtype)


def test_the_three_matrix_schemes_overlap():
    """Which is why the choice is ranked rather than conditioned.  At a full
    wave on CDNA every one of them serves an F32 contraction."""
    from tensorforge.backend.instructions.compute.primitives.amd import tiling
    ctx = _hip()
    assert tiling.mfma_tile_for(64, Datatype.F32, ctx) is not None
    assert tiling.exchange_op(Datatype.F32, 64, ctx) is not None
    assert tiling.emu_tile_for(64, Datatype.F32, ctx) is not None


def test_only_the_exchange_scheme_serves_f64():
    """The lane-batched loop cannot feed an instruction that spends lane bits
    on the contraction, and there is nothing narrower to emulate F64 from."""
    from tensorforge.backend.instructions.compute.primitives.amd import tiling
    ctx = _hip(dtype=Datatype.F64)
    assert tiling.mfma_tile_for(64, Datatype.F64, ctx) is None
    assert tiling.emu_tile_for(64, Datatype.F64, ctx) is None
    assert tiling.exchange_op(Datatype.F64, 64, ctx) is not None


def test_both_gates_off_leaves_exactly_the_deployed_scheme():
    """What makes the layer snapshot-neutral: with nothing switched on, the
    only scheme offered is the one that was running."""
    from tensorforge.backend.instructions.compute.primitives.amd import tiling
    assert (tiling.EMULATION, tiling.EXCHANGE) == (False, False)
    for dtype in (Datatype.F32, Datatype.F64):
        for threads in (32, 64):
            offered = tiling.offers(threads, dtype, _hip(dtype=dtype))
            assert offered <= {tiling.Scheme.LANE_BATCHED}


def test_a_gated_scheme_outranks_the_deployed_one():
    """Otherwise the switch answers nothing: wherever the deployed scheme also
    fits, it would keep running and the gate would only affect the shapes
    nothing else served."""
    from tensorforge.backend.instructions.compute.primitives.amd import tiling
    order = tiling.ORDER
    assert order[-1] is tiling.Scheme.LANE_BATCHED
    assert set(order) == set(tiling.Scheme)


def test_only_the_lane_batched_scheme_draws_a_boundary():
    """The other two pad a partial block inside the emitter, so a plan has no
    tail to place -- and the four-wide threshold would not carry to a block of
    sixteen anyway."""
    from tensorforge.backend.instructions.compute.primitives.amd import tiling
    ctx = _hip()
    lane = tiling.Fit(tiling.Scheme.LANE_BATCHED,
                      tiling.mfma_tile_for(64, Datatype.F32, ctx).op)
    swap = tiling.Fit(tiling.Scheme.EXCHANGE,
                      tiling.exchange_op(Datatype.F32, 64, ctx))
    for n in range(1, 40):
        assert tiling.boundary(swap, n) == n
    assert tiling.boundary(lane, 9) == 8
    assert tiling.boundary(lane, 10) == 10


# -- choosing the entry within a scheme ------------------------------------ #

def test_the_entry_is_ranked_by_issues_not_by_fit():
    """Thirteen columns take four 4x4 issues or one 16x16 that wastes three
    of its sixteen.  Which is faster is a property of the two instructions
    rather than of the waste, so the count is what is stated and `CYCLES` is
    where the rest would go."""
    from tensorforge.backend.instructions.compute.primitives.amd import (
        MATRIX_OPS, tiling)
    ops = {o.builtin: o for o in MATRIX_OPS}
    narrow, wide = ops['mfma_f32_4x4x1f32'], ops['mfma_f32_16x16x1f32']
    assert tiling.issues(narrow, 13, 56, 56) == 4 * tiling.issues(
        wide, 13, 56, 56)
    from tensorforge.backend.instructions.compute import ranking
    assert ranking.CYCLES == {}, 'a guessed cycle count reads as a measurement'


def test_a_term_product_takes_the_whole_output_axis():
    """Which is why partial spare is no use: a product is a complete tile,
    not a column of one."""
    from tensorforge.backend.instructions.compute.primitives.amd import (
        MATRIX_OPS, tiling)
    ops = {o.builtin: o for o in MATRIX_OPS}
    for name, expected in (('mfma_f32_4x4x4bf16_1k', 1),
                           ('mfma_f32_16x16x4bf16_1k', 1),
                           ('mfma_f32_32x32x4bf16_1k', 3)):
        assert tiling.spare_products(ops[name], 9) == expected, name


def test_an_unknown_extent_counts_as_one_tile():
    """A shape that does not carry its leading dimension or its depth still
    ranks by what it does carry, rather than ranking everything equal."""
    from tensorforge.backend.instructions.compute.primitives.amd import (
        MATRIX_OPS, tiling)
    ops = {o.builtin: o for o in MATRIX_OPS}
    op = ops['mfma_f32_4x4x1f32']
    assert tiling.issues(op, 9) == tiling.issues(op, 9, lead=1, depth=1)


def test_the_emulated_limit_has_a_number_on_it():
    """The emitter takes only entries whose k-vector is the block width, and
    that is what keeps the 32-wide one out -- where three term products would
    share an issue instead of one."""
    from tensorforge.backend.instructions.compute.primitives.amd import (
        MATRIX_OPS, emu_tiles, tiling)
    ctx = _hip()
    offered = {tile.op.builtin for tile, _ in emu_tiles(64, Datatype.F32, ctx)}
    ops = {o.builtin: o for o in MATRIX_OPS}
    assert offered == {'mfma_f32_4x4x4bf16_1k'}
    wide = ops['mfma_f32_32x32x4bf16_1k']
    assert wide.k != wide.m
    assert tiling.spare_products(wide, 9) == 3
