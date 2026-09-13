# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""The kernels of yateto's own code-gen tests that did not build everywhere.

Recorded in `fixtures/kernels/yateto_codegen_gaps.json` (seissol/yateto
50200d3, interface 7).  Three gaps, each closed on this side:

* a guard on SYCL -- the barrier of a multiplication narrower than the
  sub-group is the sub-group's, and waited inside the guard for a neighbour
  that may not take the branch; `verify` refused it;
* `LogicalNot`, which had no counterpart;
* an operand whose box is narrower than the destination (`trace` stores three
  of eight entries): yateto reads zero outside it, and the elementwise
  descriptor refused an operand of another shape.
"""

from __future__ import annotations

import contextlib
import io
import json
import pathlib
import re
import warnings

import numpy as np
import pytest

import kernel_eval
from tensorforge.common.context import Context
from tensorforge.common.operation import Operation
from tensorforge.frontend.yateto import DescriptionReader
from tensorforge.generators.descriptions import ElementwiseDescr
from tensorforge.generators.generator import Generator

FIXTURE = (pathlib.Path(__file__).parent / "fixtures" / "kernels"
           / "yateto_codegen_gaps.json")


def recorded(name, strided=False):
    description = json.loads(FIXTURE.read_text())["descriptions"][name]
    if strided:     # the host oracle follows no arrays of pointers
        for tensor in description["tensors"]:
            if tensor["addressing"] == "n&+o&":
                tensor["addressing"] = "n*N+o&"
    return description


def build(description, backend, arch):
    descrs, _ = DescriptionReader(None, {}).read(description)
    generator = Generator(descrs, Context(
        arch=arch, backend=backend, fp_type=descrs[-1].dest.tensor.datatype))
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        generator.generate()
    return descrs, generator


NAMES = sorted(json.loads(FIXTURE.read_text())["descriptions"])


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("backend,arch", [("cuda", "sm_86"), ("hip", "gfx942"),
                                          ("oneapi", "pvc")])
def test_it_builds(name, backend, arch):
    build(recorded(name), backend, arch)


def _threads_per_mult(generator):
    meta = re.search(r'"threads_per_mult":(\d+)', generator.get_kernel())
    return int(meta.group(1))


def test_a_guard_on_sycl_takes_a_multiplication_the_width_of_the_sub_group():
    """Eight lanes would put two multiplications into one sub-group, whose
    barrier is the narrowest SYCL has under SPMD."""
    _, generator = build(recorded("conditional_3"), "oneapi", "pvc")
    assert _threads_per_mult(generator) == 16


def test_a_guard_leaves_the_lanes_alone_where_a_multiplication_has_a_barrier():
    _, generator = build(recorded("conditional_3"), "cuda", "sm_86")
    assert _threads_per_mult(generator) < 16


def test_a_logical_negation_is_a_comparison_with_zero():
    """`Not` is the bitwise one: `~true` is -2, and true again."""
    descrs, _ = DescriptionReader(None, {}).read(recorded("conditional_4"))
    negations = [d for d in descrs
                 if isinstance(d, ElementwiseDescr) and d.op is Operation.EQ]
    assert negations and all(d.srcs[-1] == 0 for d in negations)


def test_a_narrower_operand_cuts_the_destination_at_its_edge():
    """`exp(trace_j)` over eight entries, `trace` storing three: `exp` of the
    table where it is stored, and of zero -- one -- after."""
    descrs, _ = DescriptionReader(None, {}).read(recorded("elementwise_14"))
    exps = [d for d in descrs if getattr(d, "op", None) is Operation.EXP]
    assert [(list(d.dest.bbox.lower()), list(d.dest.bbox.upper()))
            for d in exps] == [([0], [3]), ([3], [8])]
    assert exps[1].srcs == [0]


@pytest.mark.parametrize("name", ["elementwise_10", "elementwise_14"])
@pytest.mark.parametrize("arch", ["sm_86", "sm_120"])
def test_the_cut_computes_what_yateto_means(name, arch):
    """#10 cuts along `j` (`A = v trace^T + v B`), #14 along the lead
    (`A = v exp(trace)^T`); outside its three entries `trace` is zero."""
    description = recorded(name, strided=True)
    descrs, generator = build(description, "cuda", arch)
    tensors = {}
    for d in descrs:
        # an elementwise descriptor names its operands `srcs`, a multilinear
        # one `ops`
        operands = getattr(d, "srcs", None) or getattr(d, "ops", [])
        for s in [d.dest] + [s for s in operands if hasattr(s, "tensor")]:
            tensors[s.tensor.alias] = s.tensor
    lanes, mults = kernel_eval.launch_geometry(generator.get_launcher())
    mem = kernel_eval.evaluate_wave(generator.get_kernel(), lanes, seed=4,
                                    globals_only=True, mults=mults)

    def read(alias, n):
        return np.array([mem.get((tensors[alias].name, k), np.nan)
                         for k in range(n)])

    # `trace` is constant and comes with its values: where the kernel reads it
    # from memory (#14, under `exp`) the oracle's numbers are what it used,
    # and where it is written into the code as literals (#10, a factor of a
    # multilinear) its own values are
    stated = next(t["values"]["data"] for t in description["tensors"]
                  if t["name"] == "trace")
    trace = np.zeros(8)
    trace[:3] = np.where(np.isnan(read("trace", 3)),
                         [value for _, value in stated], read("trace", 3))
    v = read("v", 8)
    got = read("A", 64).reshape(8, 8, order="F")
    if name == "elementwise_10":
        want = np.outer(v, trace) + v[:, None] * read("B", 64).reshape(
            8, 8, order="F")
    else:
        want = np.outer(v, np.exp(trace))
    assert np.allclose(got, want)
