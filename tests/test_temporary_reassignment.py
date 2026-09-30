# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""A temporary assigned anew from a narrower product defines all of it.

yateto's `=` defines its whole destination: where the operands support fewer
rows than the tensor has, the rest is zero.  A store into a shared-memory
temporary therefore writes zeros over the rows `_analyze` did not compute
(`MultilinearBuilder._assignment_zero_box`).  One that wrote only the computed
rows and left the others as the buffer held them would turn

    M(temp) = b0 ; M = b1 ; out = M          b1 stored over [0, 1) of 4

into `out = [b1[0], b0[1], b0[2], b0[3]]` instead of `[b1[0], 0, 0, 0]`.
SeisSol's free-surface-gravity kernel re-assigns `MPrev` that way.

`fixtures/kernels/temporary_reassignment.json` holds the statements as yateto
describes them (and one face of the free-surface-gravity kernel, which
`test_recorded_description` replays).  The generated kernel is interpreted on
the host (`kernel_eval`); the expected values are written down from the
statements.
"""

from __future__ import annotations

import contextlib
import io
import json
import pathlib
import warnings

import numpy as np
import pytest

from tensorforge.common.context import Context
from tensorforge.frontend.yateto import DescriptionReader
from tensorforge.generators.generator import Generator
from tensorforge.reference import kernel_eval

FIXTURE = (pathlib.Path(__file__).parent / "fixtures" / "kernels"
           / "temporary_reassignment.json")
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
        operands = getattr(d, "srcs", None) or getattr(d, "ops", [])
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
def test_a_temporary_reassigned_from_a_narrower_operand_is_zero_elsewhere(arch):
    # M = b0 ; M = b1 ; out = M
    read = run("temporary_reassigned_from_narrower", arch)
    b1 = read("b1", 1)
    assert np.allclose(read("out", 4), [b1[0], 0.0, 0.0, 0.0])


@pytest.mark.parametrize("arch", ARCHS)
def test_a_temporary_reassigned_from_itself(arch):
    # M = b0 ; M = b1 - M ; out = M
    read = run("temporary_reassigned_from_itself", arch)
    b0, b1 = read("b0", 4), read("b1", 1)
    want = -b0
    want[0] += b1[0]
    assert np.allclose(read("out", 4), want)


@pytest.mark.parametrize("arch", ARCHS)
def test_a_temporary_reassigned_after_it_was_read(arch):
    # M = b0 ; out1 = M ; M = b1 ; out = M
    read = run("temporary_reassigned_after_read", arch)
    b0, b1 = read("b0", 4), read("b1", 1)
    assert np.allclose(read("out1", 4), b0)
    assert np.allclose(read("out", 4), [b1[0], 0.0, 0.0, 0.0])
