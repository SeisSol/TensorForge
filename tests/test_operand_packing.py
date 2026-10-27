# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Storing a batch-constant operand without its empty cells.

`common.matrix.packing`, which is not `compute.packing` -- that one counts
what an instruction axis has left over (`tests/test_packing.py`).

Three forms, one invariant: what a packing leaves out has to be what the
operand's own numbers say is not there.  The tests below check that per
form, then that the forms agree with each other where they should, then
that the generator offers one and addresses the reads through it.
"""

import re

import numpy as np
import pytest

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.context import Context
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.packing import KINDS, Packing, pack, pack_union
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.common.options import Options
from tensorforge.generators.descriptions import GemmDescr
from tensorforge.generators.generator import Generator


def _tensor(values, alias="A"):
    values = np.asarray(values, dtype=float)
    shape = list(values.shape)
    return Tensor(shape, Addressing.NONE,
                  BoundingBox([0] * len(shape), shape),
                  alias=alias, datatype=Datatype.F32, data=values)


def _band(rows=40, cols=8, runs=None, seed=3):
    rng = np.random.default_rng(seed)
    a = np.zeros((rows, cols))
    runs = runs or [(c, min(rows - 1, c + 9)) for c in range(cols)]
    for c, (lo, hi) in enumerate(runs):
        a[lo:hi + 1, c] = rng.uniform(0.5, 1.5, size=hi + 1 - lo)
    return a


# ---------------------------------------------------------------------------
# What a form may leave out
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kind", KINDS)
def test_a_packing_drops_only_cells_the_numbers_say_are_empty(kind):
    values = _band()
    t = _tensor(values)
    p = pack(t, kind)
    assert p is not None
    held = set(p.order())
    for cell in range(values.size):
        r, c = cell % values.shape[0], cell // values.shape[0]
        if values[r, c] != 0:
            assert cell in held, f"{kind} dropped {(r, c)}, which holds a value"


@pytest.mark.parametrize("kind", KINDS)
def test_the_slot_map_and_the_order_are_one_statement(kind):
    """`slot` is what the kernel addresses with and `order` what the host
    packs by; they are each other's inverse or the two disagree."""
    values = _band()
    p = pack(_tensor(values), kind)
    order = p.order()
    for c in range(p.columns):
        for r in range(p.rows):
            s = p.slot(r, c)
            if s >= 0:
                assert order[s] == r + p.rows * c


def test_the_forms_compress_in_their_stated_order():
    values = _band(runs=[(0, 30), (5, 9), (12, 12), (0, 34),
                         (31, 34), (2, 20), (7, 7), (18, 33)])
    values[15, 0] = 0.0          # a hole only the bitset leaves out
    t = _tensor(values)
    sizes = {kind: pack(t, kind).elements for kind in KINDS}
    assert sizes["rows"] > sizes["runs"] > sizes["bits"]
    assert sizes["bits"] == int(np.count_nonzero(values))


def test_a_form_that_saves_nothing_is_not_offered():
    """A dense operand packs to itself, and an order plus a predicate over
    reads that need neither is worse than no offer at all."""
    t = _tensor(np.ones((8, 4)))
    assert all(pack(t, kind) is None for kind in KINDS)


def test_a_bitset_names_the_rows_it_holds():
    values = np.zeros((12, 2))
    values[[1, 4, 5, 11], 0] = 1.0
    values[[0, 3], 1] = 2.0
    p = pack(_tensor(values), "bits")
    assert p.words == (0b100000110010, 0b001001)
    assert p.offsets == (0, 4)
    assert p.elements == 6
    assert p.slot(5, 0) == 2 and p.slot(11, 0) == 3
    assert p.slot(3, 1) == 5
    assert p.slot(2, 0) == -1


def test_a_uniform_packing_is_the_one_a_computed_column_could_address():
    """`rows` gives every column the same start and spacing, so its address
    is affine in the column; the other two are not."""
    values = _band()
    assert pack(_tensor(values), "rows").uniform
    assert not pack(_tensor(values), "runs").uniform
    assert not pack(_tensor(values), "bits").uniform


def test_an_empty_run_still_leaves_the_column_addressable():
    values = np.zeros((16, 3))
    values[2:6, 0] = 1.0
    values[9, 2] = 1.0           # column 1 holds nothing at all
    p = pack(_tensor(values), "runs")
    assert p.lengths[1] == 0
    assert p.slot(0, 1) == -1
    assert p.elements == 5


# ---------------------------------------------------------------------------
# A merged run's members share one packing
# ---------------------------------------------------------------------------

def test_the_union_holds_every_members_cells():
    a = np.zeros((20, 4)); a[0:5, :] = 1.0
    b = np.zeros((20, 4)); b[10:15, :] = 2.0
    p = pack_union([_tensor(a), _tensor(b)], "runs")
    assert p.starts == (0, 0, 0, 0) and p.lengths == (15, 15, 15, 15)
    for t, values in ((_tensor(a), a), (_tensor(b), b)):
        held = set(p.order())
        for cell in range(values.size):
            r, c = cell % 20, cell // 20
            if values[r, c] != 0:
                assert cell in held


def test_members_of_different_shapes_pack_to_nothing():
    assert pack_union([_tensor(np.ones((8, 2))),
                       _tensor(np.ones((9, 2)))], "rows") is None


def _faces(rows=24, cols=7, count=4):
    """`Q += A_f I` for operators whose numbers end at a row of their own: a
    run the merge rolls into one loop over a stand-in, and each member packed
    alone a different size."""
    i = SubTensor(Tensor([cols, 6], Addressing.STRIDED,
                         BoundingBox([0, 0], [cols, 6]), alias="I",
                         datatype=Datatype.F32))
    q = SubTensor(Tensor([rows, 6], Addressing.STRIDED,
                         BoundingBox([0, 0], [rows, 6]), alias="Q",
                         datatype=Datatype.F32))
    faces = [SubTensor(_tensor(_band(rows, cols, runs=[(0, 2 + 6 * f)] * cols,
                                     seed=f), alias=f"A{f}"))
             for f in range(count)]
    return ([GemmDescr(False, False, a, i, q, alpha=1.0, beta=1.0)
             for a in faces], [a.tensor for a in faces])


def test_a_merged_build_taken_over_stores_its_members_as_it_reads_them():
    """The merge decision builds the list merged and then written out, over
    the same tensors, and may take the merged build over after all
    (`Generator._merge_by_written_out`).  Written out, each member is packed
    by its own numbers; the merged kernel reads every one through the union,
    so taking it over states the union on them again.  Without that the
    host laid them out apiece: SeisSol's order-6 `localFluxAll` on gfx942
    (32 lanes, no matrix cores) came out wrong in every value."""
    descrs, members = _faces()
    gen = Generator(descrs, Context(arch="sm_120", backend="cuda",
                                    fp_type=Datatype.F32,
                                    options=Options(pack_operands="runs")))
    merged = gen._sibling(merge_within=Generator.EVERY_RUN)
    merged.generate()
    alike = [t.storage_elements() for t in members]
    assert len(set(alike)) == 1 and members[0].packing is not None
    gen._sibling().generate()
    assert len({t.storage_elements() for t in members}) == len(members)
    gen._take_over(merged)
    assert [t.storage_elements() for t in members] == alike
    assert all(t.packing == members[0].packing for t in members)


# ---------------------------------------------------------------------------
# A lane that holds several rows
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("granule", [2, 4])
@pytest.mark.parametrize("kind", ["rows", "runs"])
def test_a_granule_rounds_the_runs_out_to_whole_lane_vectors(kind, granule):
    """A lane's `granule` rows have to be on one side of a run's end, since
    the predicate that holds the access off cannot mask inside a lane."""
    p = pack(_tensor(_band(runs=[(1, 6), (9, 26), (31, 33), (2, 34),
                                 (5, 7), (13, 30), (7, 7), (11, 29)])),
             kind, granule)
    assert p.granule == granule
    for start, length, offset in zip(p.starts, p.lengths, p.offsets):
        assert start % granule == 0
        assert length % granule == 0
        assert offset % granule == 0


@pytest.mark.parametrize("granule", [2, 4])
def test_a_granule_still_holds_every_value(granule):
    values = _band(runs=[(1, 6), (9, 26), (31, 33), (2, 34),
                         (5, 7), (13, 30), (7, 7), (11, 29)])
    p = pack(_tensor(values), "runs", granule)
    held = set(p.order())
    for cell in range(values.size):
        r, c = cell % values.shape[0], cell // values.shape[0]
        if values[r, c] != 0:
            assert cell in held


def test_a_granule_costs_little():
    """Measured over the operators and stated here so a change shows: the
    run form gives up 40.3 % at one row a lane, 38.7 % at two, 34.1 % at
    four -- a reason to keep the wide read rather than to drop it."""
    values = _band(rows=64, cols=8,
                   runs=[(1, 6), (9, 26), (31, 33), (2, 61),
                         (45, 50), (7, 22), (37, 38), (13, 54)])
    t = _tensor(values)
    sizes = [pack(t, "runs", g).elements for g in (1, 2, 4)]
    assert sizes[0] < sizes[1] < sizes[2] < values.size


def test_a_bitset_has_no_granule_but_one():
    """Its slot is a count, so two adjacent rows are two adjacent slots only
    where both are stored -- there is no vector form, and saying so is
    better than rounding into a wrong address."""
    assert pack(_tensor(_band()), "bits", 2) is None
    with pytest.raises(ValueError):
        Packing("bits", 8, 1, [0], [8], [0], words=[0xff], granule=2)


def test_a_packing_refuses_a_granule_its_numbers_do_not_fit():
    with pytest.raises(ValueError):
        Packing("runs", 16, 2, [1, 0], [4, 4], [0, 4], granule=2)


def test_a_packing_refuses_a_kind_it_does_not_know():
    with pytest.raises(ValueError):
        pack(_tensor(_band()), "columns")


def test_a_packing_needs_one_number_per_column():
    with pytest.raises(ValueError):
        Packing("runs", 8, 2, [0], [3, 3], [0, 3])


# ---------------------------------------------------------------------------
# What the generator does with it
# ---------------------------------------------------------------------------

def _generated(values, kind, arch="sm_120", backend="cuda", **options):
    rows, cols = values.shape
    a = SubTensor(_tensor(values))
    b = SubTensor(Tensor([cols, 6], Addressing.STRIDED,
                         BoundingBox([0, 0], [cols, 6]), alias="B",
                         datatype=Datatype.F32))
    c = SubTensor(Tensor([rows, 6], Addressing.STRIDED,
                         BoundingBox([0, 0], [rows, 6]), alias="C",
                         datatype=Datatype.F32))
    descrs = [GemmDescr(False, False, a, b, c, alpha=1.0, beta=0.0)]
    gen = Generator(descrs, Context(arch=arch, backend=backend,
                                    fp_type=Datatype.F32,
                                    options=Options(pack_operands=kind,
                                                    **options)))
    gen.generate()
    return gen, a.tensor


def test_off_offers_no_packing():
    """Not "no order": `prepare_operands` may still arrange the operand for
    the instruction that reads it, which is a different question and has
    its own switch."""
    _, t = _generated(_band(), "off")
    assert t.packing is None
    assert t.storage_elements() == t.get_actual_volume() or t.simt_interleave


@pytest.mark.parametrize("kind", KINDS)
def test_the_offer_shrinks_the_buffer_and_the_order_says_how(kind):
    values = _band(rows=24, cols=7)
    _, t = _generated(values, kind)
    assert t.packing is not None and t.packing.kind == kind
    assert t.storage_elements() == t.packing.elements < values.size
    assert len(t.storage_values()) == t.packing.elements


@pytest.mark.parametrize("kind", KINDS)
def test_the_stored_values_are_the_cells_the_order_names(kind):
    values = _band(rows=24, cols=7)
    _, t = _generated(values, kind)
    stored = t.storage_values()
    for slot, cell in enumerate(t.storage_map()):
        r, c = cell % values.shape[0], cell // values.shape[0]
        assert stored[slot] == pytest.approx(values[r, c])


def test_a_read_outside_the_run_is_held_off_its_lanes():
    """The span starts at row 5, so the lanes below it must not read -- the
    element before the run is the previous column's number, not a zero."""
    values = np.zeros((32, 6))
    values[5:21, :] = 1.5
    gen, t = _generated(values, "rows")
    src = gen.get_kernel()
    assert t.packing.starts[0] == 5 and t.packing.lengths[0] == 16
    assert "_lead >= 5" in src.replace("(", "").replace(")", "")


def test_a_bitset_read_is_a_population_count():
    values = np.zeros((24, 5))
    for c in range(5):
        values[[1, 3, 4, 9, 17 + c], c] = 1.0
    gen, t = _generated(values, "bits")
    src = gen.get_kernel()
    assert t.packing.kind == "bits"
    assert "__popc(" in src


def test_a_bitset_is_not_offered_where_a_lane_slot_outruns_the_word():
    """The shift that masks the row is counted in 32 bits, so an operand
    whose lane slots reach row 32 takes a form that does not shift."""
    values = _band(rows=48, cols=6,
                   runs=[(2, 40)] * 6)
    _, t = _generated(values, "bits")
    assert t.packing is None


def test_a_wide_lead_is_packed_at_its_own_width():
    """`lead_vectorize` gives a lane four adjacent rows, and the packing
    hands it four adjacent slots by rounding its runs to four."""
    values = np.zeros((64, 8))
    rng = np.random.default_rng(31)
    for c, (lo, hi) in enumerate(((1, 6), (9, 26), (31, 33), (2, 61),
                                  (45, 50), (7, 22), (37, 38), (13, 54))):
        values[lo:hi + 1, c] = rng.uniform(0.5, 1.5, size=hi + 1 - lo)
    rows, cols = values.shape
    a = SubTensor(Tensor([rows, cols], Addressing.NONE,
                         BoundingBox([0, 0], [rows, cols]), alias="A",
                         datatype=Datatype.F32, data=values, alignment=16))
    b = SubTensor(Tensor([cols, 6], Addressing.STRIDED,
                         BoundingBox([0, 0], [cols, 6]), alias="B",
                         datatype=Datatype.F32, alignment=16))
    c = SubTensor(Tensor([rows, 6], Addressing.STRIDED,
                         BoundingBox([0, 0], [rows, 6]), alias="C",
                         datatype=Datatype.F32, alignment=16))
    gen = Generator([GemmDescr(False, False, a, b, c, alpha=1.0, beta=0.0)],
                    Context(arch="sm_120", backend="cuda",
                            fp_type=Datatype.F32,
                            options=Options(pack_operands="runs",
                                            lead_vectorize=True)))
    gen.generate()
    assert a.tensor.packing is not None
    assert a.tensor.packing.granule > 1
    assert a.tensor.storage_elements() == a.tensor.packing.elements


def test_a_rolled_reduction_takes_a_uniform_form_without_a_table():
    """The column is a loop variable then, so the slot has to be affine in
    it: `rows` is, `runs` with unequal columns is not -- and without
    `pack_tables` there is nowhere to read the per-column numbers from."""
    rng = np.random.default_rng(53)
    rows, cols = 64, 96
    even = np.zeros((rows, cols))
    even[7:46, :] = rng.uniform(0.5, 1.5, size=(39, cols))
    odd = np.zeros((rows, cols))
    for c in range(cols):
        lo = 3 + (c % 11)
        odd[lo:lo + 20, c] = rng.uniform(0.5, 1.5, size=20)

    def built(values, kind):
        a = SubTensor(Tensor([rows, cols], Addressing.NONE,
                             BoundingBox([0, 0], [rows, cols]), alias="A",
                             datatype=Datatype.F32, data=values))
        b = SubTensor(Tensor([cols, 6], Addressing.STRIDED,
                             BoundingBox([0, 0], [cols, 6]), alias="B",
                             datatype=Datatype.F32))
        c = SubTensor(Tensor([rows, 6], Addressing.STRIDED,
                             BoundingBox([0, 0], [rows, 6]), alias="C",
                             datatype=Datatype.F32))
        gen = Generator([GemmDescr(False, False, a, b, c, alpha=1.0, beta=0.0)],
                        Context(arch="sm_120", backend="cuda",
                                fp_type=Datatype.F32,
                                options=Options(pack_operands=kind, k_roll=4,
                                                pack_tables="off")))
        gen.generate()
        return gen, a.tensor

    gen, t = built(even, "rows")
    assert t.packing is not None and t.packing.uniform
    # the column term is the span, formed inside the loop
    assert f"* {t.packing.lengths[0]}" in gen.get_kernel()
    assert t.packing_table is None

    _, t = built(odd, "runs")
    assert t.packing is None


def test_a_table_makes_the_other_forms_reachable_from_a_rolled_reduction():
    """`pack_tables`: the per-column numbers go into a kernel parameter, so
    the column a loop hands over picks them instead of folding into the
    address.  Which is three loads and two comparisons a column, against
    the looser form the operand would otherwise be stored in."""
    rng = np.random.default_rng(53)
    rows, cols = 64, 96
    odd = np.zeros((rows, cols))
    for c in range(cols):
        lo = 3 + (c % 11)
        odd[lo:lo + 20 + (c % 7), c] = rng.uniform(
            0.5, 1.5, size=20 + (c % 7))

    def built(kind, tables='auto'):
        a = SubTensor(Tensor([rows, cols], Addressing.NONE,
                             BoundingBox([0, 0], [rows, cols]), alias="A",
                             datatype=Datatype.F32, data=odd.copy()))
        b = SubTensor(Tensor([cols, 6], Addressing.STRIDED,
                             BoundingBox([0, 0], [cols, 6]), alias="B",
                             datatype=Datatype.F32))
        c = SubTensor(Tensor([rows, 6], Addressing.STRIDED,
                             BoundingBox([0, 0], [rows, 6]), alias="C",
                             datatype=Datatype.F32))
        gen = Generator([GemmDescr(False, False, a, b, c, alpha=1.0, beta=0.0)],
                        Context(arch="sm_120", backend="cuda",
                                fp_type=Datatype.F32,
                                options=Options(pack_operands=kind, k_roll=4,
                                                pack_tables=tables)))
        gen.generate()
        return gen, a.tensor

    gen, t = built("runs")
    assert t.packing is not None and t.packing.kind == "runs"
    assert t.packing_table is not None
    src = gen.get_kernel()
    # Three numbers a column, read from the parameter rather than spelled.
    assert src.count(f"{t.packing_table}[") >= 3
    # In param space, and nowhere else: `__grid_constant__` on CUDA.
    assert f"__grid_constant__ const tensorforge::ValueArray<int32_t, " \
           f"{3 * t.packing.columns}> {t.packing_table}" in src
    # The caller passes nothing for it.
    assert t.packing_table not in gen.get_header()

    _, loose = built("fit", "off")
    assert loose.packing is not None and loose.packing.kind == "rows"
    _, tight = built("fit")
    assert tight.packing is not None and tight.packing.kind == "runs"
    assert tight.storage_volume() < loose.storage_volume()


def test_a_table_leaves_room_for_the_views_lower_corner():
    """`Symbol._address` subtracts the view's lower corner from every axis,
    and the three packed forms have to leave room for that in three
    different ways -- which is only visible where the corner is not zero.

    `_packed_index` puts it in the lead index's own offset,
    `_uniform_packed_index` keeps it out of the computed address and lets
    the subtraction stand, and a table has to do both: the run bounds are
    compared against the *box* row, which has the corner in it, while the
    address must not.  `bits` adds it back instead, because a population
    count owes the corner nothing.

    Nothing in the corpus reaches this: all 207940 packed reads of the
    recorded SeisSol kernels at orders 6 and 8 start their box at row zero,
    where the three agree because there is nothing to disagree about.  So
    the box is shifted here, and by 9 -- neither zero nor a multiple of the
    lane count.
    """
    rows, cols, lo, hi = 56, 96, 9, 41
    rng = np.random.default_rng(23)
    values = np.zeros((rows, cols))
    for c in range(cols):
        first = lo + (3 * c) % 7
        values[first:first + 9 + (5 * c) % 13, c] = rng.uniform(
            0.5, 1.5, size=9 + (5 * c) % 13)
    a = SubTensor(Tensor([rows, cols], Addressing.NONE,
                         BoundingBox([lo, 0], [hi, cols]), alias="A",
                         datatype=Datatype.F32, data=values))
    b = SubTensor(Tensor([cols, 5], Addressing.STRIDED,
                         BoundingBox([0, 0], [cols, 5]), alias="B",
                         datatype=Datatype.F32))
    c = SubTensor(Tensor([rows, 5], Addressing.STRIDED,
                         BoundingBox([lo, 0], [hi, 5]), alias="C",
                         datatype=Datatype.F32))
    gen = Generator([GemmDescr(False, False, a, b, c, alpha=1.0, beta=0.0)],
                    Context(arch="sm_120", backend="cuda",
                            fp_type=Datatype.F32,
                            options=Options(pack_operands="runs", k_roll=4)))
    gen.generate()
    t = a.tensor
    assert t.packing is not None and t.packing_table is not None
    src = gen.get_kernel()
    # The box row, which is the lane's tensor row less the corner, and
    # which is what the run bounds are about.
    box = re.findall(r'int32_t (v\d+_row) = v\d+_(?:lead|row) \+ -9;', src)
    assert box, src
    # The predicate compares *that* row, not the tensor row.
    # Where the predicate stands as a value of its own, and where it guards
    # the select that reads the element.
    guard = [line for line in src.splitlines()
             if '_start)' in line and '_stop)' in line]
    assert guard and all(any(name in line for name in box) for line in guard), \
        guard
    # And the address subtracts the corner exactly once -- the table's own
    # offset and start are applied to the tensor row, so the one
    # subtraction `Symbol._address` makes is the whole of it.
    slot = re.findall(r'\(\(v\d+_(?:row|lead) \+ v\d+_off\) - v\d+_start\)'
                      r'((?: - \d+)*)', src)
    assert slot, src
    assert all(tail == ' - 9' for tail in slot), slot


def test_a_table_too_big_for_the_arguments_goes_into_the_file():
    """The argument space is 4 kB on sm_86, and `module` is where a table
    that no longer fits it lives -- the same numbers, read the same way,
    out of the generated file rather than the arguments."""
    rng = np.random.default_rng(53)
    rows, cols = 64, 96
    odd = np.zeros((rows, cols))
    for c in range(cols):
        odd[3 + (c % 11):24 + (c % 7), c] = rng.uniform(
            0.5, 1.5, size=21 + (c % 7) - (c % 11))

    def built(home):
        a = SubTensor(Tensor([rows, cols], Addressing.NONE,
                             BoundingBox([0, 0], [rows, cols]), alias="A",
                             datatype=Datatype.F32, data=odd.copy()))
        b = SubTensor(Tensor([cols, 6], Addressing.STRIDED,
                             BoundingBox([0, 0], [cols, 6]), alias="B",
                             datatype=Datatype.F32))
        c = SubTensor(Tensor([rows, 6], Addressing.STRIDED,
                             BoundingBox([0, 0], [rows, 6]), alias="C",
                             datatype=Datatype.F32))
        gen = Generator([GemmDescr(False, False, a, b, c, alpha=1.0, beta=0.0)],
                        Context(arch="sm_86", backend="cuda",
                                fp_type=Datatype.F32,
                                options=Options(pack_operands="runs", k_roll=4,
                                                pack_tables=home)))
        gen.generate()
        return gen, a.tensor

    gen, t = built("module")
    assert t.packing is not None and t.packing.kind == "runs"
    assert t.packing_table is not None and not t.packing_table_param
    src = gen.get_kernel()
    assert f'__device__ const int32_t {t.packing_table}[288] = ' in src
    # Named nowhere in the signature, and read exactly as a parameter is.
    assert f'{t.packing_table},' not in src and f'{t.packing_table})' not in src
    assert src.count(f'{t.packing_table}[') >= 3
    assert t.packing_table not in gen.get_launcher()

    gen, t = built("param")
    assert t.packing_table_param
    assert '__device__ const int32_t' not in gen.get_kernel()


def test_a_bitset_word_reaches_the_table_as_a_pattern():
    """A column holding row 31 is a word no `int32_t` holds as a number,
    and the table carries `int32_t`; every read of it reads bits."""
    rows, cols = 32, 24
    values = np.zeros((rows, cols))
    rng = np.random.default_rng(11)
    for c in range(cols):
        for r in range(c % 3, rows, 1 + c % 2):
            values[r, c] = rng.uniform(0.5, 1.5)
    packing = pack(_tensor(values), "bits")
    assert packing is not None and packing.needs_table
    table = packing.columns_table()
    assert len(table) == 2 * cols and packing.table_stride == 2
    assert any(w >= 1 << 31 for w in packing.words)
    assert all(-(1 << 31) <= n < (1 << 31) for n in table)
    # The pattern survives the reinterpretation.
    assert [n & 0xFFFFFFFF for n in table[1::2]] == list(packing.words)


def test_a_named_form_takes_the_operand_from_the_fragment_order():
    """The two are not comparable -- one buys a wide load with padding, the
    other buys room -- so naming a form is an instruction and `auto` is
    not."""
    values = _band(rows=24, cols=7)
    _, named = _generated(values, "runs", prepare_operands=True)
    assert named.packing is not None and named.packing.kind == "runs"


def test_fit_stops_short_of_the_form_that_costs():
    """`fit` names a preference, not a form.  It starts at `runs`, which is
    free -- on sm_120 it timed -0.71 % against a control of -0.63 % -- and
    reaches `bits` only under room pressure, that being the one form with a
    price: +3.7 % against the same control and +15 % registers."""
    _, t = _generated(_band(rows=24, cols=7), "fit", preload_globals=True)
    assert t.packing is not None and t.packing.kind == "runs"


def test_fit_falls_through_to_a_looser_form_where_a_tighter_is_refused():
    """A form can be refused rather than merely unprofitable: a rolled
    reduction without `pack_tables` takes none whose slot is not affine in
    the column.  Offered one form only, `fit` would leave such an operand
    unpacked -- which is worse than the looser form it skipped.

    Pinned with the table off, because that is what makes this refusal a
    refusal; with it on the tighter form is reachable and `fit` takes it,
    which `test_a_table_makes_the_other_forms_reachable_from_a_rolled_reduction`
    is the other half of."""
    rows, cols = 64, 96
    rng = np.random.default_rng(83)
    values = np.zeros((rows, cols))
    for c in range(cols):                      # per-column runs: not uniform
        lo = 3 + (c % 11)
        values[lo:lo + 20, c] = rng.uniform(0.5, 1.5, size=20)
    a = SubTensor(Tensor([rows, cols], Addressing.NONE,
                         BoundingBox([0, 0], [rows, cols]), alias="A",
                         datatype=Datatype.F32, data=values))
    b = SubTensor(Tensor([cols, 6], Addressing.STRIDED,
                         BoundingBox([0, 0], [cols, 6]), alias="B",
                         datatype=Datatype.F32))
    c = SubTensor(Tensor([rows, 6], Addressing.STRIDED,
                         BoundingBox([0, 0], [rows, 6]), alias="C",
                         datatype=Datatype.F32))
    gen = Generator([GemmDescr(False, False, a, b, c, alpha=1.0, beta=0.0)],
                    Context(arch="sm_120", backend="cuda",
                            fp_type=Datatype.F32,
                            options=Options(pack_operands="fit", k_roll=4,
                                            pack_tables="off")))
    gen.generate()
    # `runs` is refused here and `rows` is not, so `rows` is what it takes
    assert a.tensor.packing is not None
    assert a.tensor.packing.kind == "rows"


def test_fit_tightens_until_the_images_fit():
    """The level the generator settles on, asked of the step that decides
    it: four operators of 24 KB against a 64 KB block need more than the
    row span gives."""
    gen, _ = _generated(_band(rows=24, cols=7), "fit", preload_globals=True)
    cap = 64 * 1024

    class _Sym:
        def __init__(self, obj):
            self.obj = obj

    big = np.zeros((32, 200))
    big[4:28, :] = 1.0
    for c in range(0, 200, 3):
        big[16, c] = 0.0          # holes the bitset leaves out
    candidates = [_Sym(_tensor(big, alias=f"A{i}")) for i in range(4)]
    gen._choose_pack_level(candidates, cap)
    assert gen._pack_level == KINDS.index("bits")
    gen._choose_pack_level(candidates, 4 * 32 * 200 * 4 + 4096)
    assert gen._pack_level == KINDS.index("rows")


def test_a_window_the_matrix_paths_decline_is_not_stored_for_them():
    """`packed_boxed_runs` on gfx942: a window from row 9 over 32 lanes, which
    the matrix paths cannot place.

    The plan said the matrix path all the same, so `settle_storage` stored
    `A` in k-quads for it, and the nest that computed the product read that
    buffer by coordinate -- the k-quads of row `r` as rows `4r..4r+3` on an
    MI300A.  The plan now says the nest, and the packing the case asks for
    takes the operand.
    """
    import contextlib
    import importlib.util
    import io
    from pathlib import Path

    from tensorforge.common.context import Context
    from tensorforge.common.options import Options
    from tensorforge.generators.generator import Generator

    path = Path(__file__).parent / 'cases' / 'packed_boxed_runs.py'
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    gen = Generator(mod.descr_list(), Context(
        arch='gfx942', backend='hip', fp_type=mod.DTYPE,
        options=Options(**mod.OPTIONS)))
    with contextlib.redirect_stdout(io.StringIO()):
        gen.generate()
    a = next(s.obj for s in gen.built._scopes.get_global_scope().values()
             if getattr(s.obj, 'alias', None) == 'A')
    assert getattr(a, 'packing', None) is not None
