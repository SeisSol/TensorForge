# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""The register-layout vocabulary, and the node that vendor intrinsics use.

They exist so that the two questions a pass has to ask -- *do these two
register images have the same distribution?* and *do these two intrinsics
touch the same accumulator?* -- have an answer that is looked up rather than
re-derived from a string.

The tests are therefore about semantics, not about output: equality,
hashing, and that an untracked layout never silently matches a tracked one.
"""

from __future__ import annotations

import pytest

from tensorforge.backend.pir import passes
from tensorforge.backend.pir.build import IRBuilder
from tensorforge.backend.pir.core import (INDEX, SCALAR_LAYOUT, LaneAxis,
                                          RegisterLayout, ScalarType,
                                          Uniformity, accesses_conflict,
                                          join_layout)
from tensorforge.backend.pir.emit import Emitter
from tensorforge.backend.symbol import LeadIndex
from tensorforge.backend.writer import Writer
from tensorforge.common.basic_types import Datatype
from tensorforge.common.context import Context

F32 = ScalarType(Datatype.F32)


# --------------------------------------------------------------------------- #
# LaneAxis / RegisterLayout
# --------------------------------------------------------------------------- #

def test_axis_rejects_degenerate_shapes():
    for bad in (0, -1):
        with pytest.raises(Exception):
            LaneAxis(bad, 1)
        with pytest.raises(Exception):
            LaneAxis(4, bad)


def test_layout_is_a_value():
    a = RegisterLayout((LaneAxis(16, 1),))
    b = RegisterLayout((LaneAxis(16, 1),))
    c = RegisterLayout((LaneAxis(16, 4),))
    assert a == b and a != c
    assert len({a, b, c}) == 2                      # hashable, and b folds into a


def test_compose_concatenates_axes():
    a = RegisterLayout((LaneAxis(16, 1),))
    c = RegisterLayout((LaneAxis(4, 16),))
    assert a.compose(c).axes == (LaneAxis(16, 1), LaneAxis(4, 16))
    assert a.compose(c).rank == 2
    # composition is not commutative: the axes are ordered
    assert a.compose(c) != c.compose(a)


def test_scalar_layout_is_not_none():
    """The empty layout is a statement; `None` is the absence of one."""
    assert RegisterLayout() is not None
    assert RegisterLayout() != None                 # noqa: E711 -- the point
    assert not RegisterLayout().is_distributed
    assert RegisterLayout((LaneAxis(16, 1),)).is_distributed
    assert not RegisterLayout((LaneAxis(1, 1),)).is_distributed


# --------------------------------------------------------------------------- #
# LeadIndex
# --------------------------------------------------------------------------- #

def test_lead_index_equality_is_structural():
    assert LeadIndex(0, 16, 1) == LeadIndex(0, 16, 1)
    assert LeadIndex(0, 16, 1) != LeadIndex(3, 16, 1)     # different slot
    assert LeadIndex(0, 16, 1) != LeadIndex(0, 32, 1)     # different block
    assert LeadIndex(0, 16, 1) != LeadIndex(0, 16, 4)     # different stride
    assert LeadIndex(0, 16, 1) != 'n0'
    assert len({LeadIndex(0, 16, 1), LeadIndex(0, 16, 1)}) == 1


def test_lead_index_layout_drops_the_slot():
    """Two indices into the same register image differ in `nonlead` only."""
    assert LeadIndex(0, 16, 1).layout() == LeadIndex(7, 16, 1).layout()
    assert LeadIndex(0, 16, 1).same_layout(LeadIndex(7, 16, 1))
    assert not LeadIndex(0, 16, 1).same_layout(LeadIndex(0, 32, 1))


def test_undistributed_lead_index_matches_a_plain_index():
    assert LeadIndex(0, 1, 1).same_layout('n0')
    assert not LeadIndex(0, 16, 1).same_layout('n0')


# --------------------------------------------------------------------------- #
# Propagation and the conservative default
# --------------------------------------------------------------------------- #

def test_values_are_untracked_by_default():
    b = IRBuilder(Datatype.F32)
    assert b.varalloc().layout is None
    assert b.op('add', F32, b.varalloc(), b.varalloc()).layout is None


def test_elementwise_ops_inherit_an_agreed_layout():
    lay = RegisterLayout((LaneAxis(16, 1),))
    b = IRBuilder(Datatype.F32)
    x = b.value(F32, layout=lay)
    y = b.value(F32, layout=lay)
    z = b.value(F32, layout=RegisterLayout((LaneAxis(32, 1),)))
    assert b.op('add', F32, x, y).layout == lay
    # disagreement is not an error, but the result stops being tracked
    assert b.op('add', F32, x, z).layout is None
    assert join_layout([x, 4]) == lay               # literals say nothing


def test_an_untracked_operand_the_lanes_differ_in_leaves_the_result_untracked():
    """`lead + 32` with the 32 a constant: the constant is replicated, and
    the lane index is untracked and differs between the lanes, so the sum
    does -- it is not replicated, nor known to be spread any one way.  An
    untracked operand the lanes agree on vetoes nothing."""
    b = IRBuilder(Datatype.F32)
    lead = b.op('rem', INDEX, b.thread_id('x'), 32)
    assert lead.layout is None and lead.uniformity <= Uniformity.LANE
    thirty_two = b.const(32, INDEX)
    assert b.op('add', INDEX, lead, thirty_two).layout is None
    assert b.op('add', F32, b.value(F32, layout=RegisterLayout(
        (LaneAxis(16, 1),))), b.op('mul', F32, lead, 2.0)).layout is None
    row = b.op('mul', INDEX, b.thread_id('y'), 4)
    assert row.layout is None and row.uniformity > Uniformity.LANE
    assert b.op('add', INDEX, row, thirty_two).layout == SCALAR_LAYOUT


def test_a_loop_whose_lanes_start_apart_counts_untracked():
    """Started at the lane's own index, the induction is no more known than
    that index: the replicated layout every other loop's induction has is a
    claim the verifier reports as wrong."""
    b = IRBuilder(Datatype.F32)
    with b.for_(b.thread_id('x'), 64, 32, uniform=Uniformity.LANE) as f:
        assert f.induction.layout is None
    with b.for_(0, 64, 32) as f:
        assert f.induction.layout == SCALAR_LAYOUT


def test_a_call_does_not_inherit_unless_asked():
    """A broadcast exists to change the distribution; inheriting would lie."""
    lay = RegisterLayout((LaneAxis(16, 1),))
    b = IRBuilder(Datatype.F32)
    x = b.value(F32, layout=lay)
    assert b.call('bcast', F32, x).layout is None
    assert b.call('negate', F32, x, keep_layout=True).layout == lay
    out = RegisterLayout((LaneAxis(64, 1),))
    assert b.call('bcast', F32, x, layout=out).layout == out


def test_cse_will_not_merge_across_layouts():
    lay = RegisterLayout((LaneAxis(16, 1),))
    b = IRBuilder(Datatype.F32)
    src = b.rawexpr('src', hint='s')
    same_a = b.call('f', F32, src, keep_layout=True)
    b.call('f', F32, src, keep_layout=True)
    body = passes.cse(b.finish())
    assert sum(1 for s in body if s.op == 'call') == 1

    b = IRBuilder(Datatype.F32)
    src = b.value(F32, layout=lay)
    b.call('f', F32, src, layout=lay)
    b.call('f', F32, src, layout=RegisterLayout((LaneAxis(64, 1),)))
    body = passes.cse(b.finish())
    assert sum(1 for s in body if s.op == 'call') == 2, \
        'two different distributions were folded into one'
    assert same_a is not None


# --------------------------------------------------------------------------- #
# The void call node
# --------------------------------------------------------------------------- #

def _acc_kernel():
    b = IRBuilder(Datatype.F32)
    c1 = b.rawexpr('0.0f', hint='acc')
    c2 = b.rawexpr('0.0f', hint='acc')
    a = b.call('tensorforge::broadcast<64, 16, 0>', F32, c1, hint='bc')
    b.call_stmt('tensorforge::fmacdpp16<0>', c1, a, c2, writes=(c1,))
    b.call_stmt('tensorforge::fmacdpp16<1>', c2, a, c1, writes=(c2,))
    return b.finish()


def test_void_call_verifies():
    passes.verify(_acc_kernel())


def test_void_call_emits_a_bare_statement():
    w = Writer()
    Emitter(w, Context(arch='gfx90a', backend='hip',
                       fp_type=Datatype.F32)).run(_acc_kernel())
    src = w.get_src()
    assert 'tensorforge::fmacdpp16<0>(v0_acc, v2_bc, v1_acc);' in src
    assert 'v3' not in src, 'a void call must not declare a result'


def test_accumulators_alias_precisely():
    """What `Effect.UNKNOWN` on a raw statement could never say."""
    calls = [s for s in _acc_kernel() if s.op == 'call' and not s.target]
    a, b = calls
    assert not any(accesses_conflict(x, y) for x in a.accesses for y in b.accesses)
    assert any(accesses_conflict(x, x) for x in a.accesses)


def test_written_operands_are_pinned():
    """A value handed to a reference parameter must not be inlined away."""
    body = _acc_kernel()
    written = {s.args[0].id for s in body if s.op == 'call' and not s.target}
    for s in body:
        if s.target and s.target[0].id in written:
            assert s.attr('escapes'), f'{s.target[0]} is written but not pinned'


def test_dce_keeps_a_pinned_accumulator():
    body = passes.dce(_acc_kernel())
    assert sum(1 for s in body if s.op == 'rawexpr') == 2


# --------------------------------------------------------------------------- #
# The lane map, against the index the generator actually emits
# --------------------------------------------------------------------------- #

def _emitted_index(tid, slot, block, stride):
    """`LeadIndex.write`, evaluated: `((tid / stride) % block) + slot * block`."""
    return ((tid // stride) % block) + slot * block


@pytest.mark.parametrize("block,stride,threads",
                         [(16, 1, 16), (16, 1, 64), (16, 4, 64),
                          (4, 4, 16), (32, 1, 32), (2, 1, 64), (1, 1, 16)])
def test_holders_matches_the_generated_index(block, stride, threads):
    """The definition and the emitted formula have to agree.

    The generator emits `((tid / stride) % block) + slot * block`, which puts
    element `s` in a run of `stride` *consecutive* threads at
    `(s % block) * stride` -- and not in lane `(s // stride) % block`: same
    symbols, different map. Every consumer of layouts inherits the map
    `LaneAxis` documents, so a wrong one would be wrong in all of them.
    """
    axis = LaneAxis(block, stride)
    for slot in range(2):
        for element in range(slot * block, (slot + 1) * block):
            expected = tuple(t for t in range(threads)
                             if _emitted_index(t, slot, block, stride) == element)
            assert axis.holders(element, threads) == expected
            assert axis.slot(element) == slot


@pytest.mark.parametrize("stride", [1, 2, 4, 8])
def test_stride_replicates_over_consecutive_threads(stride):
    """`stride` is replication, not packing.

    A lane holding several consecutive elements is a vector *type* over the
    slot dimension; it is not an axis. Conflating the two would make a
    `float4` load look like a different distribution from the scalar load of
    the same data, which it is not.
    """
    axis = LaneAxis(4, stride)
    holders = axis.holders(0, 4 * stride)
    assert holders == tuple(range(stride)), \
        "the first element should sit in the first `stride` consecutive threads"
    assert len(set(axis.holders(e, 4 * stride) for e in range(4))) == 4


def test_undistributed_axis_puts_everything_in_every_thread():
    axis = LaneAxis(1, 1)
    assert axis.holders(0, 8) == tuple(range(8))
    assert axis.slot(5) == 5


@pytest.mark.parametrize("block,stride", [(16, 1), (4, 4), (16, 4), (32, 1)])
def test_lead_index_layout_agrees_with_its_own_index(block, stride):
    """`LeadIndex.layout()` has to describe the index `LeadIndex` emits."""
    threads = 64
    axis = LeadIndex(0, block, stride).layout().axis(0)
    for element in range(block):
        assert axis.holders(element, threads) == tuple(
            t for t in range(threads)
            if _emitted_index(t, 0, block, stride) == element)


def test_a_vector_load_is_a_type_not_an_axis():
    """What carrying `float4` through the IR means, stated as a test.

    Four consecutive elements in one lane change the element *type*; the
    distribution over lanes is whatever it was. So the layout of a packed
    load equals the layout of the scalar load it replaces, and a pass that
    compares layouts will treat them as the same distribution -- which is the
    property that lets a packed load flow through operations that were
    written for scalars.
    """
    from tensorforge.backend.pir.core import ScalarType
    lay = RegisterLayout((LaneAxis(16, 1),))
    b = IRBuilder(Datatype.F32)
    scalar = b.value(ScalarType(Datatype.F32), layout=lay)
    packed = b.value(ScalarType(Datatype.F32, 4), layout=lay)
    assert scalar.layout == packed.layout
    assert scalar.type != packed.type


# --------------------------------------------------------------------------- #
# Rank 2, and what `stride` means once there is a second axis
# --------------------------------------------------------------------------- #

def test_a_fused_two_dimensional_arrangement_is_expressible():
    """Four lanes per entry of dimension 0, dimension 1 alongside.

    Lane `l` holds `(l % 4, l // 4)`. This is the shape a fused operator
    wants, and it needs no algebra: two axes and an intersection.
    """
    lay = RegisterLayout((LaneAxis(4, 1), LaneAxis(16, 4)))
    for l in range(64):
        assert lay.holders((l % 4, l // 4), 64) == (l,)


def test_the_fused_arrangement_is_a_bijection():
    lay = RegisterLayout((LaneAxis(4, 1), LaneAxis(16, 4)))
    assert lay.tiles(64)
    assert lay.replication(64) == 1


def test_stride_means_replication_alone_and_a_second_dimension_in_company():
    """The observation that makes rank 2 more than rank 1 twice over.

    `LaneAxis(16, 4)` by itself puts one element in four neighboring lanes,
    which hold copies. Beside a `LaneAxis(4, 1)` the same axis puts one
    element of dimension 1 in those lanes, and they differ in dimension 0.
    Identical field, identical number, opposite meaning -- so "is this
    replicated?" cannot be answered per axis.
    """
    alone = RegisterLayout((LaneAxis(16, 4),))
    paired = RegisterLayout((LaneAxis(4, 1), LaneAxis(16, 4)))
    assert alone.replication(64) == 4
    assert paired.replication(64) == 1
    assert alone.axes[0] == paired.axes[1]      # the very same axis


@pytest.mark.parametrize("threads", [16, 32, 64])
def test_replication_counts_every_copy(threads):
    for block in (1, 2, 4, 8, 16):
        for stride in (1, 2, 4):
            lay = RegisterLayout((LaneAxis(block, stride),))
            classes = {}
            for t in range(threads):
                classes.setdefault((t // stride) % block, 0)
                classes[(t // stride) % block] += 1
            sizes = set(classes.values())
            expected = sizes.pop() if len(sizes) == 1 else 0
            assert lay.replication(threads) == expected


def test_holders_rejects_an_index_of_the_wrong_rank():
    lay = RegisterLayout((LaneAxis(4, 1), LaneAxis(16, 4)))
    with pytest.raises(Exception):
        lay.holders((3,), 64)


# --------------------------------------------------------------------------- #
# float4 as float[4]
# --------------------------------------------------------------------------- #

def test_a_packed_value_round_trips_through_extract_and_pack():
    """`float4` is interchangeable with `float[4]`.

    `pack` and `extract` are the bridge, and both preserve the layout: the
    packing changes how many elements a lane holds, not which lane holds
    what. So a vector load can be taken apart, fed to operations written for
    scalars, and put back together, and every layout comparison along the way
    sees the same distribution.
    """
    from tensorforge.backend.pir.core import ScalarType
    lay = RegisterLayout((LaneAxis(16, 1),))
    b = IRBuilder(Datatype.F32)
    packed = b.value(ScalarType(Datatype.F32, 4), layout=lay)
    parts = [b.extract(packed, i, F32) for i in range(4)]
    assert all(p.layout == lay for p in parts)
    again = b.pack(ScalarType(Datatype.F32, 4), *parts)
    assert again.layout == lay
    assert again.type == packed.type


def test_packing_does_not_change_the_distribution():
    from tensorforge.backend.pir.core import ScalarType
    lay = RegisterLayout((LaneAxis(16, 1),))
    b = IRBuilder(Datatype.F32)
    scalar = b.value(F32, layout=lay)
    packed = b.value(ScalarType(Datatype.F32, 4), layout=lay)
    assert scalar.layout == packed.layout
    assert scalar.layout.holders((0,), 64) == packed.layout.holders((0,), 64)


def test_the_degenerate_axis_is_normalized():
    """`LaneAxis(1, s)` is not distributed whatever `s` is.

    Equality is the one thing this type exists for, so distributions that are
    the same have to compare the same. Without normalizing, `movdpp16` at 16
    threads would produce `LaneAxis(1, 16)` while the hardware simulation
    recovers `LaneAxis(1, 1)` -- the same distribution, unequal, and a
    relayout search that would not find an instruction sitting right in the
    table.
    """
    assert LaneAxis(1, 16) == LaneAxis(1, 1)
    assert LaneAxis(1, 16).stride == 1
    assert len({LaneAxis(1, s) for s in (1, 2, 4, 8, 16)}) == 1
    assert LaneAxis(1, 16).holders(0, 8) == tuple(range(8))
    # a real block keeps its stride
    assert LaneAxis(4, 16) != LaneAxis(4, 1)


# --------------------------------------------------------------------------- #
# Written arguments need an address
# --------------------------------------------------------------------------- #

def test_a_constant_cannot_be_written_through_a_reference():
    """The C++ takes its outputs by non-const reference.

    A padded MFMA tail block that handed `0.0f` to `transpose4x4b32`'s third
    and fourth parameters, which are `T &`, would be ill-formed. Snapshots,
    symbolic equivalence and the PIR verifier would all pass it -- none of
    them models C++ overload resolution -- so without this refusal only a
    compile of the generated source would see it.
    """
    from tensorforge.backend.pir.core import IRError, ScalarType
    b = IRBuilder(Datatype.F32)
    f = ScalarType(Datatype.F32)
    reg, lit = b.declare(f, hint='r'), b.const(0.0, f)
    with pytest.raises(IRError):
        b.call_stmt('tensorforge::transpose4x4b32', reg, lit, writes=(reg, lit))


def test_a_constant_is_fine_in_a_read_position():
    from tensorforge.backend.pir.core import ScalarType
    b = IRBuilder(Datatype.F32)
    f = ScalarType(Datatype.F32)
    reg, lit = b.declare(f, hint='r'), b.const(0.0, f)
    b.call_stmt('tensorforge::transpose4x4b32', reg, lit, writes=(reg,))


def test_a_non_value_cannot_be_written():
    from tensorforge.backend.pir.core import IRError, ScalarType
    b = IRBuilder(Datatype.F32)
    reg = b.declare(ScalarType(Datatype.F32), hint='r')
    with pytest.raises(IRError):
        b.call_stmt('f', reg, writes=(reg, 3.0))


# --------------------------------------------------------------------------- #
# The generated bodies
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('stem, arch, backend', [
    ('local_flux', 'sm_86', 'cuda'), ('chain_five', 'sm_86', 'cuda'),
    ('local_flux', 'gfx942', 'hip'), ('chain_five', 'pvc', 'esimd')])
def test_no_lane_varying_value_of_a_generated_body_claims_to_be_replicated(
        stem, arch, backend, monkeypatch):
    """What the verifier reports where a value's layout and its uniformity
    disagree, over bodies whose lane indices -- a lane's row of a column, a
    swizzled shared address -- meet constants in every address."""
    import contextlib
    import importlib.util
    import io
    import warnings
    from pathlib import Path

    from tensorforge.backend import pir
    from tensorforge.common.options import Options
    from tensorforge.generators.generator import Generator

    bodies = []
    emit = pir.emit

    def keep(body, writer, context=None, metrics=None):
        bodies.append(body)
        return emit(body, writer, context, metrics)

    monkeypatch.setattr(pir, 'emit', keep)
    path = Path(__file__).resolve().parent / 'cases' / f'{stem}.py'
    spec = importlib.util.spec_from_file_location('tf_layout__' + stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ctx = Context(arch=arch, backend=backend,
                  fp_type=getattr(mod, 'DTYPE', None), options=Options())
    gen = Generator(mod.descr_list(), ctx, attrs=getattr(mod, 'ATTRS', None))
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter('ignore')
        gen.generate()
    assert bodies
    found = [d for body in bodies for d in passes.verify(body, strict=False)
             if 'is lane-varying but its layout says' in d]
    assert found == []
