# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""The cost model counts what it says it counts.

Unlike the diagnostics in `test_tools.py`, these are not smoke tests.  A flop
count is a number a roofline divides by, so a model that is quietly off by the
destination's own elements produces a plot that is wrong everywhere and looks
right everywhere -- and nobody re-derives an arithmetic intensity they have no
reason to doubt.  Every convention the model chose is pinned here with the
figure it produces, so changing the convention costs a visible test edit.

Host-only throughout: no GPU, no toolchain, no code generation.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
from pathlib import Path

import pytest

from tensorforge.analysis.cost import Cost, descr_cost, list_cost
from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators.descriptions import GemmDescr, MultilinearDescr

HERE = Path(__file__).resolve().parent
CASES = HERE / "cases"


def _case(rel: str):
    path = CASES / rel
    spec = importlib.util.spec_from_file_location(f"cost_{path.stem}", path)
    mod = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(mod)
    return mod


def _cost_of(rel: str) -> Cost:
    mod = _case(rel)
    return list_cost(mod.descr_list(), batch=getattr(mod, "BATCH", 8),
                     datatype=mod.DTYPE)


def _tensor(name, shape, addressing=Addressing.STRIDED, dt=Datatype.F32,
            is_tmp=False):
    return Tensor(list(shape), addressing,
                  BoundingBox([0] * len(shape), list(shape)),
                  alias=name, datatype=dt, is_tmp=is_tmp)


# -- arithmetic ------------------------------------------------------------- #

def test_dense_gemm_is_not_two_m_n_k():
    """`2*M*N*K` is the figure everyone quotes, and it is one add too many.

    A GEMM with `beta == 0` writes its first term instead of adding it, so the
    adds are `M*N*(K-1)`, not `M*N*K`.  For 16x16x16 that is 3840 rather than
    4096 -- six percent, which is under the noise of a timing run and exactly
    large enough to make a roofline point sit somewhere it should not.

    Pinned rather than derived from a formula in the test: a test that recomputes
    the model's own arithmetic agrees with it by construction.
    """
    cost = _cost_of("square_notrans.py")          # 16x16x16, BATCH = 4
    assert cost.mults == 16 * 16 * 16 * 4 == 16384
    assert cost.adds == 16 * 16 * 15 * 4 == 15360
    assert cost.flops == 31744
    assert cost.flops < 2 * 16 * 16 * 16 * 4


def test_accumulation_pays_for_the_destination_it_reads_back():
    """`add=True` reads the destination, so every point is an add.

    The complement of the test above, and the reason the model asks the
    descriptor rather than assuming: the same shapes cost different adds
    depending on a flag, and the difference is exactly the output volume.
    """
    a = SubTensor(_tensor("A", [8, 8]))
    b = SubTensor(_tensor("B", [8, 8]))
    c = SubTensor(_tensor("C", [8, 8]))
    target, permute = [[0, -1], [-1, 1]], [[0, 1], [0, 1]]

    assign = descr_cost(MultilinearDescr(c, [a, b], target, permute, add=False))
    accum = descr_cost(MultilinearDescr(
        SubTensor(_tensor("C", [8, 8])),
        [SubTensor(_tensor("A", [8, 8])), SubTensor(_tensor("B", [8, 8]))],
        target, permute, add=True))

    assert accum.adds - assign.adds == 8 * 8
    assert accum.mults == assign.mults


def test_alpha_is_one_multiply_per_output_element():
    """A scalar operand does not multiply once per contraction point.

    `GemmDescr` injects alpha as an operand with `Addressing.SCALAR`, so a
    model that counted `len(ops) - 1` multiplies per point would charge a 9x9x9
    kernel 1458 multiplies where 810 is the work.  `alpha * sum(...)` and
    `sum(alpha * ...)` are the same value and no implementation worth measuring
    picks the second.
    """
    cost = _cost_of("csa_alpha.py")               # 9x9x9, alpha != 1, BATCH = 4
    assert cost.mults == (9 * 9 * 9 + 9 * 9) * 4 == 3240


def test_transcendentals_are_reported_and_not_summed_into_flops():
    """`sqrt` has no flop count, because its cost is the target's, not the
    operation's.  Reported separately so a caller may weight it having decided
    what the weight is."""
    cost = _cost_of("elementwise/sqrt.py")        # 16x16, BATCH = 4
    assert cost.transcendental == 16 * 16 * 4
    assert cost.flops == 0


def test_a_max_reduction_computes_no_arithmetic():
    """`max` is work and is not arithmetic.

    Bucketing it as an add would give a max-reduction the same arithmetic
    intensity as a sum-reduction over the same data, which is the one thing a
    roofline of the two is asked to distinguish.
    """
    cost = _cost_of("reduction/max_axis.py")
    assert cost.flops == 0
    assert cost.nonarith > 0
    assert _cost_of("reduction/sum_axis.py").adds > 0


# -- traffic ---------------------------------------------------------------- #

def test_batch_constant_operand_is_not_scaled_by_the_batch():
    """`Addressing.NONE` shares one storage block across the batch.

    The kernel-side pointer skips the `batchId * volume` term and the host
    allocates once (`driver_emit.py`), so scaling it by the batch would report
    a SeisSol operator matrix as the dominant traffic of every kernel that
    touches one -- and at production batch sizes that error is four orders of
    magnitude.
    """
    cost = _cost_of("addressing_none.py")         # A is NONE, B/C strided
    by_name = {t.name: t for t in cost.tensors}
    assert by_name["A"].per_batch is False
    assert by_name["B"].per_batch is True
    # 16x16 floats: one copy of A, four each of B and C.
    assert cost.bytes == 1024 + 4 * 1024 + 4 * 1024


def test_temporaries_move_no_bytes():
    """A tensor marked `is_tmp` lives in shared memory or registers for the
    life of the kernel and never reaches global memory.  Counting it would make
    the model incomparable with the DRAM counters it exists to be compared
    against."""
    mod = _case("mixed/ml_then_ew.py")
    names = {t.name for t in list_cost(mod.descr_list(), batch=mod.BATCH,
                                       datatype=mod.DTYPE).tensors}
    assert "TMP" not in names
    assert {"A", "B", "C"} <= names


def test_two_windows_of_one_tensor_unite_rather_than_take_the_larger():
    """The mutation this guards: `max(prev.read, t.read)` instead of a union.

    Two descriptors reading disjoint windows of one operand move their union.
    Taking the larger is right whenever one window contains the other, which is
    every case in the corpus -- so this is built rather than borrowed, and it is
    the one shape where the cheap answer is wrong.
    """
    big = _tensor("BIG", [16, 16])
    left = SubTensor(big, BoundingBox([0, 0], [8, 16]), [0, 0])
    right = SubTensor(big, BoundingBox([0, 0], [8, 16]), [8, 0])
    target, permute = [[0, -1], [-1, 1]], [[0, 1], [0, 1]]

    descrs = [
        MultilinearDescr(SubTensor(_tensor("O1", [8, 16])),
                         [left, SubTensor(_tensor("W", [16, 16]))],
                         target, permute),
        MultilinearDescr(SubTensor(_tensor("O2", [8, 16])),
                         [right, SubTensor(_tensor("W", [16, 16]))],
                         target, permute),
    ]
    big_traffic = {t.name: t for t in list_cost(descrs).tensors}["BIG"]
    assert big_traffic.read == 16 * 16 * 4, (
        "the two 8x16 windows united to something other than the whole 16x16 "
        "tensor")


def test_scalar_operands_move_no_bytes():
    """An alpha is a literal at the call site, not a buffer."""
    cost = _cost_of("csa_alpha.py")
    assert all(t.name in {"A", "B", "C"} for t in cost.tensors)


def test_sliced_operand_is_charged_for_its_window_not_its_storage():
    """A 16x16 window of a 32x32 operand moves 16x16.

    The bounding box is the full storage -- memory spans `upper - lower` and
    address 0 is `lower` -- while the window is a slicing offset, so a model
    reading the tensor's own extent would charge four times the traffic.
    """
    cost = _cost_of("slicing/inner_region.py")
    by_name = {t.name: t for t in cost.tensors}
    assert by_name["A"].stored == 32 * 32
    # A per-tensor entry is per batch element; `Cost.read_bytes` is where the
    # batch enters, because that is where `per_batch` is known to apply.
    assert by_name["A"].read == 16 * 16 * 4
    assert cost.read_bytes == (16 * 16 + 16 * 8) * 4 * 4


def test_sparsity_is_recorded_and_not_applied():
    """Which structurally-zero entries get elided is the backend's decision
    and the block shape's, so the density is on the record and the flop count
    stays dense.  A model that applied it would attribute a code-generation
    choice to the operation."""
    cost = _cost_of("slicing/sparsity_band.py")
    banded = {t.name: t for t in cost.tensors}["B"]
    assert banded.density < 1.0
    assert cost.flops == _cost_of("square_notrans.py").flops


# -- corpus ----------------------------------------------------------------- #

def test_the_whole_corpus_costs_without_gaps():
    """Every case that constructs gets a full answer.

    `unmodelled` is how the model reports a descriptor shape it cannot count,
    and an entry appearing there is the signal that a new descriptor kind
    landed without one -- which otherwise shows up as a roofline point silently
    sitting at zero.
    """
    costed, gaps = 0, {}
    for path in sorted(CASES.rglob("*.py")):
        if path.name.startswith("_"):
            continue
        mod = _case(str(path.relative_to(CASES)))
        if not hasattr(mod, "descr_list") or not hasattr(mod, "NAME"):
            continue
        try:
            descrs = mod.descr_list()
        except Exception:
            continue          # a case whose descriptors refuse to construct
        cost = list_cost(descrs, batch=getattr(mod, "BATCH", 8),
                         datatype=mod.DTYPE)
        costed += 1
        for reason in cost.unmodelled:
            gaps.setdefault(reason, []).append(mod.NAME)

    assert costed >= 60, f"only {costed} cases costed; the corpus is larger"
    assert not gaps, f"descriptor shapes with no cost model: {gaps}"


def test_intensity_is_none_rather_than_zero_when_nothing_moves():
    """A division that cannot be done is not a zero.  A caller plotting a
    roofline needs to leave the point out, not put it on the axis."""
    assert Cost().intensity is None


def test_a_missing_datatype_is_refused_rather_than_guessed():
    """A tensor carrying no datatype takes the context's floating-point type,
    which the model has no access to.  Guessing four bytes would halve every
    F64 kernel's traffic and never say so."""
    a = SubTensor(Tensor([8, 8], Addressing.STRIDED,
                         BoundingBox([0, 0], [8, 8]), alias="A"))
    b = SubTensor(Tensor([8, 8], Addressing.STRIDED,
                         BoundingBox([0, 0], [8, 8]), alias="B"))
    c = SubTensor(Tensor([8, 8], Addressing.STRIDED,
                         BoundingBox([0, 0], [8, 8]), alias="C"))
    descr = MultilinearDescr(c, [a, b], [[0, -1], [-1, 1]], [[0, 1], [0, 1]])
    with pytest.raises(ValueError, match="no datatype"):
        descr_cost(descr)
    assert descr_cost(descr, datatype=Datatype.F64).read_bytes == 2 * 8 * 8 * 8


def test_batch_scales_arithmetic_linearly():
    """Trivial, and the model got it wrong once by scaling the traffic and not
    the flops, which moves every arithmetic intensity by the batch."""
    one = _cost_of("square_notrans.py")
    mod = _case("square_notrans.py")
    ten = list_cost(mod.descr_list(), batch=40, datatype=mod.DTYPE)
    assert ten.flops == 10 * one.flops
    assert ten.bytes == 10 * one.bytes
    assert ten.intensity == pytest.approx(one.intensity)


def test_a_barrier_costs_nothing_and_is_not_a_gap():
    """`GridFenceDescr` computes nothing, which is different from being a
    descriptor kind the model does not know.

    Both sides are priced at the same batch on purpose.  A case's ``BATCH`` is
    a device-side knob -- the fence case runs at 96 so its batch loop takes
    more than one trip -- and reading it off each module would scale one side
    of this comparison by a ratio that says nothing about what a fence costs.
    """
    fence = _case("barrier/fence_two_gemms.py")
    one = _case("square_notrans.py")
    batch = getattr(one, "BATCH", 8)
    cost = list_cost(fence.descr_list(), batch=batch, datatype=fence.DTYPE)
    assert not cost.unmodelled
    assert cost.flops == 2 * list_cost(one.descr_list(), batch=batch,
                                       datatype=one.DTYPE).flops
