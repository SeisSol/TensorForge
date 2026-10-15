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


def test_the_default_leaves_every_operand_whole():
    _, t = _generated(_band(), "off")
    assert t.packing is None and t.storage_order is None


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


def test_fit_takes_the_loosest_form_that_leaves_room():
    """`fit` names a preference, not a form: with room to spare it stops at
    `rows`, which carries no table and no population count."""
    _, t = _generated(_band(rows=24, cols=7), "fit", preload_globals=True)
    assert t.packing is not None and t.packing.kind == "rows"


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
