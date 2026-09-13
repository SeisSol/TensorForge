# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""A tensor described without axes: one value per element, not a scalar.

yateto sends `rhoInv[]` -- one value per element, in memory or a temporary --
as a tensor of rank 0, and it stays one (`Tensor.rank0`).  As an operand it is
read with an empty index and needs no axis: an operation with axes broadcasts
it.  As a destination it has nothing to spread over the lanes, so a
contraction into it is the scalar branch's (`ScalarContractionInstruction`):
every lane computes the one value.

The earlier representation carried it with an axis of extent one, and which
axis that was had to be decided per operation; deciding it as the
destination's axis 0 made `t[i,j] = rhoInv * S[i,j]` write row 0 of `t`.
"""

from __future__ import annotations

import contextlib
import io

import numpy as np
import pytest

import kernel_eval
from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.context import Context
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.common.operation import Operation
from tensorforge.generators.descriptions import (ElementwiseDescr,
                                                 MultilinearDescr)
from tensorforge.generators.generator import Generator

# 16 rows: one multiplication per SYCL sub-group.  At 8, two share one, and
# the grouped batch loop that brings is refused by the verifier on its own
# account (a multgroup barrier in a mult-uniform loop), which is not the
# question here.
M, N, K = 16, 4, 6
TARGETS = [("cuda", "sm_86"), ("hip", "gfx942"), ("hip", "gfx1150"),
           ("oneapi", "pvc"), ("esimd", "pvc")]


def _t(shape, alias, addressing=Addressing.STRIDED, tmp=False):
    return Tensor(list(shape), addressing, alias=alias, is_tmp=tmp,
                  datatype=Datatype.F32)


def test_an_axisless_tensor_stays_axisless():
    t = _t([], 'rhoInv', tmp=True)
    assert t.rank0 and t.shape == ()
    assert t.bbox.rank() == 0


def test_its_data_stays_axisless_too():
    t = Tensor([], Addressing.NONE, alias='c', datatype=Datatype.F32,
               data={(): 0.5})
    assert t.data.shape == () and float(t.data) == 0.5


def test_a_scalar_is_not_one():
    assert not Tensor([], Addressing.SCALAR, datatype=Datatype.F32).rank0


def test_its_operand_takes_no_axis():
    """The descriptor states it as given: an empty target, no normalisation."""
    s = _t([], 'rhoInv', tmp=True)
    m = _t([K, N], 'S', Addressing.NONE)
    d = _t([K, N], 'T', tmp=True)
    descr = MultilinearDescr(SubTensor(d), [SubTensor(s), SubTensor(m)],
                             [[], [0, 1]], [[], [0, 1]])
    assert descr.target == [[], [0, 1]]


def _picked_then_scaled():
    """`r[] = v[z] p[z]` over a window of `z`, then `out[i,j] = r[] a[i,j]`.

    The SeisSol damage kernel's shape: a material parameter picked out of a
    vector into a temporary without axes, which then scales a matrix.
    """
    v = _t([M], 'v')
    p = _t([M], 'p', Addressing.NONE)
    r = _t([], 'r', tmp=True)
    a = _t([M, N], 'a')
    out = _t([M, N], 'out')
    window = BoundingBox([2], [5])
    descrs = [
        MultilinearDescr(SubTensor(r), [SubTensor(v, window),
                                        SubTensor(p, window)],
                         [[-1], [-1]], [[0], [0]]),
        MultilinearDescr(SubTensor(out), [SubTensor(r), SubTensor(a)],
                         [[], [0, 1]], [[], [0, 1]]),
    ]
    return descrs, (v, p, a, out)


def _generate(descrs, backend, arch):
    gen = Generator(descrs, Context(arch=arch, backend=backend,
                                    fp_type=Datatype.F32))
    with contextlib.redirect_stdout(io.StringIO()):
        gen.generate()
    return gen


@pytest.mark.parametrize("backend,arch", TARGETS,
                         ids=[f"{a}-{b}" for b, a in TARGETS])
def test_a_contraction_into_it_is_built_on_every_target(backend, arch):
    descrs, _ = _picked_then_scaled()
    assert _generate(descrs, backend, arch).get_kernel()


def test_the_picked_value_scales_every_row():
    """Against numpy, through the host oracle, over its seeded inputs."""
    descrs, (v, p, a, out) = _picked_then_scaled()
    gen = _generate(descrs, "cuda", "sm_86")
    lanes, mults = kernel_eval.launch_geometry(gen.get_launcher())
    mem = kernel_eval.evaluate_wave(gen.get_kernel(), lanes, seed=5,
                                    globals_only=True, mults=mults)
    seed = kernel_eval.Slot(5)
    vv = np.array([seed.read(v.name, i) for i in range(M)])
    pp = np.array([seed.read(p.name, i) for i in range(M)])
    aa = np.array([[seed.read(a.name, i + j * M) for j in range(N)]
                   for i in range(M)])
    want = float(vv[2:5] @ pp[2:5]) * aa
    got = np.array([[mem.get((out.name, i + j * M), np.nan) for j in range(N)]
                    for i in range(M)])
    assert np.max(np.abs(got - want)) < 1e-4 * np.max(np.abs(want))


@pytest.mark.parametrize("backend,arch", TARGETS,
                         ids=[f"{a}-{b}" for b, a in TARGETS])
def test_a_pointwise_operation_into_it_is_built(backend, arch):
    """`r[] = x[] / y[]`, then `out[i,j] = r[] a[i,j]`."""
    x, y = _t([], 'x'), _t([], 'y')
    r = _t([], 'r', tmp=True)
    a, out = _t([M, N], 'a'), _t([M, N], 'out')
    descrs = [
        ElementwiseDescr(Operation.DIV, SubTensor(r),
                         [SubTensor(x), SubTensor(y)]),
        MultilinearDescr(SubTensor(out), [SubTensor(r), SubTensor(a)],
                         [[], [0, 1]], [[], [0, 1]]),
    ]
    assert _generate(descrs, backend, arch).get_kernel()


def test_a_pointwise_operation_broadcasts_it():
    """`out[i,j] = a[i,j] / s[]`: no copy onto the destination's axes first."""
    s = _t([], 's')
    a, out = _t([M, N], 'a'), _t([M, N], 'out')
    gen = _generate([ElementwiseDescr(Operation.DIV, SubTensor(out),
                                      [SubTensor(a), SubTensor(s)])],
                    "cuda", "sm_86")
    lanes, mults = kernel_eval.launch_geometry(gen.get_launcher())
    mem = kernel_eval.evaluate_wave(gen.get_kernel(), lanes, seed=7,
                                    globals_only=True, mults=mults)
    seed = kernel_eval.Slot(7)
    aa = np.array([[seed.read(a.name, i + j * M) for j in range(N)]
                   for i in range(M)])
    want = aa / seed.read(s.name, 0)
    got = np.array([[mem.get((out.name, i + j * M), np.nan) for j in range(N)]
                    for i in range(M)])
    assert np.max(np.abs(got - want)) < 1e-4 * np.max(np.abs(want))
