# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""A pointwise assignment defines its whole destination.

yateto narrows a result's box to where it can be non-zero, and an assignment
to the tensor itself through that window means zeros around it -- the promise
the contraction keeps in its stores (`MultilinearBuilder._promised_box`).  A
pointwise path that wrote the window in place and nothing else would turn

    M(temp) = b0 ; M = sqrt(b1) ; out = M       b1 storing [0,1) of 4

into `out = [sqrt(b1[0]), b0[1], b0[2], b0[3]]` for `[sqrt(b1[0]), 0, 0, 0]`,
and an output assigned the same way would keep whatever it held.  SeisSol's
free-surface-gravity kernel has this shape on its contraction side.

`fixtures/kernels/pointwise_assignment.json` holds the two statements as
yateto describes them, and variants of them in the same form: a slice (which
owns only its part), a destination cut into pieces (`legalize._cells`),
a first write read back wider than it wrote, and a reduction.  The generated
kernel is interpreted on the host (`kernel_eval`), so what is checked is what
the kernel computes -- the numbers below are written down from the
statements, not taken from `tensorforge.reference.descriptors`, which
evaluates contractions only.
"""

from __future__ import annotations

import contextlib
import io
import json
import pathlib
import warnings

import numpy as np
import pytest

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.context import Context
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.frontend.yateto import DescriptionReader
from tensorforge.generators import elementwise as ew
from tensorforge.generators.descriptions import (ElementwiseDescr, GemmDescr,
                                                 MultilinearDescr)
from tensorforge.generators.generator import Generator
from tensorforge.reference import kernel_eval

FIXTURE = (pathlib.Path(__file__).parent / "fixtures" / "kernels"
           / "pointwise_assignment.json")
ARCHS = ["sm_86", "sm_120"]


def recorded(name):
    description = json.loads(FIXTURE.read_text())["descriptions"][name]
    for tensor in description["tensors"]:
        # the oracle follows no pointer arrays
        if tensor["addressing"] == "n&+o&":
            tensor["addressing"] = "n*N+o&"
    return description


def run(name, arch):
    descrs, _ = DescriptionReader(None, {}).read(recorded(name))
    gen = Generator(descrs, Context(arch=arch, backend="cuda",
                                    fp_type=descrs[-1].dest.tensor.datatype))
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gen.generate()
    tensors = {}
    for d in descrs:
        # an elementwise descriptor names its operands `srcs`, the others `ops`
        operands = (getattr(d, "srcs", None) or getattr(d, "ops", None)
                    or [getattr(d, "var", None)])
        for s in [d.dest] + [s for s in operands if hasattr(s, "tensor")]:
            tensors[s.tensor.alias] = s.tensor
    lanes, mults = kernel_eval.launch_geometry(gen.get_launcher())
    mem = kernel_eval.evaluate_wave(gen.get_kernel(), lanes, seed=3,
                                    globals_only=True, mults=mults)

    def read(alias, n):
        return np.array([mem.get((tensors[alias].name, k), np.nan)
                         for k in range(n)])
    return read


@pytest.mark.parametrize("arch", ARCHS)
def test_a_temporary_assigned_from_a_narrower_operand_is_zero_around_it(arch):
    # M(temp) = b0 ; M = sqrt(b1) ; out = M
    read = run("pointwise_temporary_reassigned_from_narrower", arch)
    out = read("out", 4)
    # entry 0 is the square root of a pseudo-random, maybe negative, input
    assert np.allclose(out[1:], 0.0), out


@pytest.mark.parametrize("arch", ARCHS)
def test_an_output_assigned_from_a_narrower_operand_is_zero_around_it(arch):
    # M(global) = b0 ; M = sqrt(b1)
    read = run("pointwise_output_reassigned_from_narrower", arch)
    m = read("M", 4)
    assert np.allclose(m[1:], 0.0), m


@pytest.mark.parametrize("name,alias", [
    ("pointwise_temporary_window_past_zero", "out"),
    ("pointwise_output_window_past_zero", "M")])
@pytest.mark.parametrize("arch", ARCHS)
def test_a_window_past_the_origin_is_zero_on_both_sides(name, alias, arch):
    # M = b0 ; M = |b2| over [2,4), b2 storing [2,4) ; (out = M)
    read = run(name, arch)
    b2 = read("b2", 2)
    assert np.allclose(read(alias, 4), [0.0, 0.0, abs(b2[0]), abs(b2[1])])


@pytest.mark.parametrize("arch", ARCHS)
def test_a_temporary_read_before_it_is_assigned_anew(arch):
    # M = b0 ; out1 = M ; M = |b1| ; out = M -- the old value is read out
    # first, and the new one is all of M
    read = run("pointwise_temporary_reassigned_after_read", arch)
    b0, b1 = read("b0", 4), read("b1", 1)
    assert np.allclose(read("out1", 4), b0)
    assert np.allclose(read("out", 4), [abs(b1[0]), 0.0, 0.0, 0.0])


@pytest.mark.parametrize("arch", ARCHS)
def test_a_slice_keeps_the_rest_of_its_tensor(arch):
    # M = b0 ; M[0:1] = |b1| ; out = M -- a slice owns its part only
    read = run("pointwise_temporary_slice_reassigned", arch)
    b0, b1 = read("b0", 4), read("b1", 1)
    want = b0.copy()
    want[0] = abs(b1[0])
    assert np.allclose(read("out", 4), want)


@pytest.mark.parametrize("name,alias", [
    ("pointwise_temporary_pieces_reassigned", "out"),
    ("pointwise_output_pieces_reassigned", "M")])
@pytest.mark.parametrize("arch", ARCHS)
def test_pieces_of_one_assignment_keep_each_other(name, alias, arch):
    # M = b0 ; M = exp(b1) over [0,2), b1 storing [0,1): `exp(b1)` and
    # `exp(0)` are two pieces, and neither may zero the other
    read = run(name, arch)
    b1 = read("b1", 1)
    assert np.allclose(read(alias, 4), [np.exp(b1[0]), 1.0, 0.0, 0.0])


@pytest.mark.parametrize("name,alias", [
    ("pointwise_temporary_divided_by_wider", "out"),
    ("pointwise_output_divided_by_wider", "M")])
@pytest.mark.parametrize("arch", ARCHS)
def test_one_piece_is_the_destination_and_owes_its_zeros(name, alias, arch):
    # M = b0 ; M = b1 / b2 over [0,1), b1 storing [0,1) and b2 all of [0,4).
    # `_cells` clamps b2's box to the window and cuts nothing: its one piece
    # is the destination itself, which owes zeros over [1,4) -- as a slice
    # it would owe none, and M would keep b0 there.
    read = run(name, arch)
    b1, b2 = read("b1", 1), read("b2", 1)
    assert np.allclose(read(alias, 4), [b1[0] / b2[0], 0.0, 0.0, 0.0])


@pytest.mark.parametrize("arch", ARCHS)
def test_a_first_write_read_back_wider_is_zero_around_it(arch):
    # M = |b1| over [0,1) ; out = M over [0,4).  The first write clears the
    # buffer (`SectionPlan.zero_first`) on the pointwise path too.
    read = run("pointwise_temporary_first_write_read_wider", arch)
    b1 = read("b1", 1)
    assert np.allclose(read("out", 4), [abs(b1[0]), 0.0, 0.0, 0.0])


@pytest.mark.parametrize("name,alias", [
    ("reduction_temporary_reassigned_from_narrower", "out"),
    ("reduction_output_reassigned_from_narrower", "M")])
@pytest.mark.parametrize("arch", ARCHS)
def test_a_reduction_is_an_assignment_too(name, alias, arch):
    # M = b0 ; M = prod_j A[l, j] over the one row A stores
    read = run(name, arch)
    a = read("A", 3)       # A stores row 0 only: one entry per column
    assert np.allclose(read(alias, 4), [a.prod(), 0.0, 0.0, 0.0])


def test_the_pieces_of_a_narrow_assignment_are_assembled_aside():
    """`_cells` cuts `M = exp(b1)` into `exp(b1)` and `exp(0)`.  Each piece is
    a slice, and the assignment's zeros are kept by one multilinear that
    assigns the assembled scratch to `M`."""
    descrs, _ = DescriptionReader(None, {}).read(
        recorded("pointwise_temporary_pieces_reassigned"))
    pieces = [d for d in descrs if isinstance(d, ElementwiseDescr)]
    assert len(pieces) == 2
    assert all(p.dest.sliced for p in pieces)
    scratch = pieces[0].dest.tensor
    assert scratch.is_tmp and all(p.dest.tensor is scratch for p in pieces)
    assign = descrs[descrs.index(pieces[-1]) + 1]
    assert isinstance(assign, MultilinearDescr) and not assign.add
    assert assign.ops[0].tensor is scratch
    assert assign.dest.tensor.alias == "M" and not assign.dest.sliced


def test_pieces_that_owe_nothing_write_the_destination():
    """Where the window is the whole tensor there is nothing to keep aside:
    `exp(trace)` over eight entries, `trace` storing three."""
    gaps = (pathlib.Path(__file__).parent / "fixtures" / "kernels"
            / "yateto_codegen_gaps.json")
    description = json.loads(gaps.read_text())["descriptions"]["elementwise_14"]
    descrs, _ = DescriptionReader(None, {}).read(description)
    pieces = [d for d in descrs if isinstance(d, ElementwiseDescr)]
    assert len(pieces) == 2 and all(p.dest.sliced for p in pieces)
    assert all(p.dest.tensor.alias == "_tmp0" for p in pieces)


def test_the_reassignment_is_a_store_that_clears():
    """Through a store, not in place: the store is what liveness asks
    whether it defines the buffer, and what the barriers fence."""
    descrs, _ = DescriptionReader(None, {}).read(
        recorded("pointwise_temporary_reassigned_from_narrower"))
    gen = Generator(descrs, Context(arch="sm_86", backend="cuda",
                                    fp_type=descrs[-1].dest.tensor.datatype))
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gen.generate()
    kernel = gen.get_kernel()
    assert "= store{r>s, clear}" in kernel
    assert "// s0 = sqrt(" not in kernel


def _generate(descrs, arch):
    gen = Generator(descrs, Context(arch=arch, backend="cuda",
                                    fp_type=Datatype.F64))
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gen.generate()
    return gen


@pytest.mark.parametrize("arch", ARCHS)
def test_a_buffer_that_went_out_as_an_image_keeps_its_layout(arch):
    """`M = A @ B` declares rows 0..8 of `M`, and `A` supports rows 0..5: the
    product's image holds five rows, and it reaches the buffer that way when
    `M = |N|` over those rows settles it.  The pointwise store then keeps the
    five-row buffer -- laid over the eight rows the descriptors declare, it
    would re-stride the buffer and clear three rows past the end of what the
    first store allocated."""
    def t(shape, hi, alias, tmp=False):
        return Tensor(list(shape), Addressing.STRIDED,
                      BoundingBox([0] * len(shape), list(hi)),
                      alias=alias, datatype=Datatype.F64, is_tmp=tmp)
    rows, cols = 5, 4
    a, b = t([8, 8], [rows, 8], "A"), t([8, cols], [8, cols], "B")
    n, c = t([8, cols], [rows, cols], "N"), t([6, 8], [6, 8], "C")
    m, out = t([8, cols], [8, cols], "M", tmp=True), t([6, cols], [6, cols], "OUT")
    descrs = [
        GemmDescr(False, False, a=SubTensor(a), b=SubTensor(b),
                  c=SubTensor(m)),
        ew.abs(SubTensor(m, BoundingBox([0, 0], [rows, cols])), SubTensor(n)),
        GemmDescr(False, False, a=SubTensor(c, BoundingBox([0, 0], [6, rows])),
                  b=SubTensor(m, BoundingBox([0, 0], [rows, cols])),
                  c=SubTensor(out)),
    ]
    gen = _generate(descrs, arch)
    kernel = gen.get_kernel()
    # the rows past five are no part of the buffer, so there is nothing to
    # clear there
    assert "store{r>s, clear}" not in kernel
    lanes, mults = kernel_eval.launch_geometry(gen.get_launcher())
    mem = kernel_eval.evaluate_wave(kernel, lanes, seed=3, globals_only=True,
                                    mults=mults)

    def read(tensor, shape):
        count = int(np.prod(shape))
        return np.array([mem.get((tensor.name, k), np.nan)
                         for k in range(count)]).reshape(shape, order="F")
    want = read(c, (6, 8))[:, :rows] @ np.abs(read(n, (rows, cols)))
    assert np.allclose(read(out, (6, cols)), want)
