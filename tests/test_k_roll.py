# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""The reduction as a real loop (`Options.k_roll`).

Every loop of a multilinear product is unrolled in the generator: sparsity
and the register broadcasts need their indices at compile time.  A dense
reduction over operands in memory does not, and on a large kernel the fully
unrolled body is what no longer fits the instruction cache.  These tests hold
the three things the option promises: nothing changes without it, the count
reaches the loop with it, and a reduction it cannot roll stays as it was --
and the cap that rolls a long reduction unasked (`Options.k_unroll_max`).
"""

from __future__ import annotations

import re

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.context import Context
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators.descriptions import GemmDescr
from tensorforge.generators.generator import Generator


def _t(shape, alias, addressing=Addressing.STRIDED):
    return SubTensor(Tensor(list(shape), addressing,
                            BoundingBox([0] * len(shape), list(shape)),
                            alias=alias, datatype=Datatype.F32))


def _kernel(monkeypatch, m, n, k, a=Addressing.NONE, **options):
    """`C = A B` with a batch-constant `A`, which the product reads straight
    from global memory, the way `local_flux` reads its operators."""
    monkeypatch.delenv("TF_OPTIONS", raising=False)
    if options:
        monkeypatch.setenv("TF_OPTIONS",
                           ",".join(f"{a}={b}" for a, b in options.items()))
    gen = Generator([GemmDescr(False, False, a=_t([m, k], "A", a),
                               b=_t([k, n], "B"), c=_t([m, n], "C"))],
                    Context(arch="sm_86", backend="cuda", fp_type=Datatype.F32))
    gen.generate()
    return gen.get_kernel()


def _rolled(src):
    return re.findall(r"#pragma unroll (\d+)\s*\n\s*for \([^;]*;[^;]*< (\d+);"
                      r"\s*[^)]*?(\+\+\w+|\+= \d+)\)", src)


def test_no_rolled_loop_without_the_option(monkeypatch):
    assert _rolled(_kernel(monkeypatch, 56, 9, 56)) == []


def test_the_count_reaches_the_reduction_loop(monkeypatch):
    loops = _rolled(_kernel(monkeypatch, 56, 9, 56, k_roll=2))
    assert [(count, extent) for count, extent, _ in loops] == [("2", "56")]
    assert loops[0][2].startswith("++")


def test_a_wide_group_steps_the_rolled_loop_by_its_width(monkeypatch):
    loops = _rolled(_kernel(monkeypatch, 56, 9, 56, k_roll=4, k_width=2))
    assert [(c, e, s) for c, e, s in loops] == [("4", "56", "+= 2")]


def test_an_operand_in_registers_keeps_the_reduction_unrolled(monkeypatch):
    """A per-element `A` is staged into a register image, whose slots are
    named at compile time -- a runtime `k` would send the array to local
    memory."""
    assert _rolled(_kernel(monkeypatch, 56, 9, 56, a=Addressing.STRIDED,
                           k_roll=2)) == []


def test_a_ragged_extent_stays_unrolled(monkeypatch):
    """55 steps in groups of two leave a short last group, which only the
    unrolled loop can count."""
    assert _rolled(_kernel(monkeypatch, 56, 9, 55, k_roll=2, k_width=2)) == []


def test_a_group_folds_into_the_destination_one_product_at_a_time(monkeypatch):
    """`(c + p0) + p1`, which contracts into two fused multiply-adds, and not
    `c + (p0 + p1)`, which is a multiply, a fused multiply-add and an add."""
    src = _kernel(monkeypatch, 56, 9, 56, k_width=2)
    assert not re.search(r"\+ \(\(\w+ \* \w+\) \+ \(\w+ \* \w+\)\)", src)
    assert re.search(r"\(\(\w+ \+ \(\w+ \* \w+\)\) \+ \(\w+ \* \w+\)\)", src)


def test_a_reduction_past_the_cap_is_rolled_unasked(monkeypatch):
    """By the largest divisor of its step count that fits under the cap, so
    that the rolled loop runs whole groups."""
    loops = _rolled(_kernel(monkeypatch, 56, 9, 120))
    assert [(count, extent) for count, extent, _ in loops] == [("60", "120")]


def test_a_reduction_the_cap_covers_stays_unrolled(monkeypatch):
    assert _rolled(_kernel(monkeypatch, 56, 9, 64)) == []


def test_a_zero_cap_unrolls_every_reduction(monkeypatch):
    assert _rolled(_kernel(monkeypatch, 56, 9, 120, k_unroll_max=0)) == []
