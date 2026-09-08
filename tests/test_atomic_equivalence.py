# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""An atomic accumulation computes what the ordinary one computes.

The two paths differ in *how* a result reaches memory and not in what it is,
so the same descriptors generated with `atomic_accumulation` on and off have
to leave the same numbers behind.  That is a differential the host
interpreter can settle without a device: `evaluate_wave` runs every lane of
one block over a shared memory image, which is exactly the guarantee an
atomic add needs modelled -- summing the arrivals is what the hardware does,
and the ordering an atomic also promises does not change a sum.

Before this the atomic path had no numerical coverage at all on the host.
`kernel_eval` aborted on the statement, so every case that emitted one was
skipped, and the skip looked like a pass.

The lane count is read off the kernel rather than assumed.  It is
`blockDim.x`, which the launcher sizes as the section's `num_threads`, and
the lead index is `threadIdx.x % num_threads` -- so running more lanes than
that is not a wider wave, it is a second block's worth of threads addressing
one block's memory.  Doing so makes every store arrive twice, which under `=`
is the same value written twice and under `+=` is exactly double: a harness
artefact that mimics a codegen defect closely enough to be worth naming here.
On hardware the extra lanes of a warp carry a different `threadIdx.y`, hence
a different batch element and a different destination base.

NVIDIA's policy row is turned on for the duration.  That is the change this
guards -- the row is off today, and the plan is to turn it on once the store
nest is known to write each element exactly once.
"""

from __future__ import annotations

import re
from dataclasses import replace

import pytest

from tensorforge.backend import placement
from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.context import Context
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators.descriptions import GemmDescr
from tensorforge.generators.generator import Generator

from kernel_eval import evaluate_wave

K = 8
SEED = 7
#: Slots that differ by less than this are the same number; the two paths
#: sum in a different order, so bit equality is the wrong bar.
TOL = 1e-4


def _t(shape, alias, dtype):
    return Tensor(shape, Addressing.STRIDED,
                  BoundingBox([0] * len(shape), list(shape)),
                  alias=alias, datatype=dtype)


def _descrs(kind, m, n, dtype):
    """One accumulating shape, by the way its writers overlap.

    The three that are not `single` are where the interesting failures have
    been: `chain` is two writers covering the same box, `slices` two covering
    disjoint parts of it, and `mixed` an assignment followed by an
    accumulation -- the arrangement in which a deferred atomic used to
    displace the store before it.
    """
    d = _t([m, n], 'D', dtype)
    if kind == 'single':
        return [GemmDescr(False, False, a=SubTensor(_t([m, K], 'A', dtype)),
                          b=SubTensor(_t([K, n], 'B', dtype)),
                          c=SubTensor(d), alpha=1.0, beta=1.0)]
    if kind == 'chain':
        return [GemmDescr(False, False,
                          a=SubTensor(_t([m, K], f'A{i}', dtype)),
                          b=SubTensor(_t([K, n], f'B{i}', dtype)),
                          c=SubTensor(d), alpha=1.0, beta=1.0)
                for i in range(2)]
    if kind == 'slices':
        half = n // 2
        return [GemmDescr(False, False,
                          a=SubTensor(_t([m, K], f'A{i}', dtype)),
                          b=SubTensor(_t([K, half], f'B{i}', dtype)),
                          c=SubTensor(d, bbox=BoundingBox([0, 0], [m, half]),
                                      offset=[0, i * half]),
                          alpha=1.0, beta=1.0)
                for i in range(2)]
    assert kind == 'mixed'
    return [GemmDescr(False, False, a=SubTensor(_t([m, K], 'A', dtype)),
                      b=SubTensor(_t([K, n], 'B', dtype)),
                      c=SubTensor(d), alpha=1.0, beta=beta)
            for beta in (0.0, 1.0)]


def _render(descrs, dtype, atomics):
    base = placement.POLICIES['nvidia']
    placement.POLICIES['nvidia'] = replace(base, atomic_accumulation=atomics)
    try:
        ctx = Context(arch='sm_86', backend='cuda', fp_type=dtype)
        gen = Generator(descrs, ctx)
        gen.register()
        gen.generate()
        return gen.get_kernel()
    finally:
        placement.POLICIES['nvidia'] = base


def _lanes(src):
    """`blockDim.x`, read off the lead index rather than assumed."""
    mods = {int(m) for m in re.findall(r'threadIdx\.x % (\d+)', src)}
    return max(mods) if mods else 32


#: Lead extents chosen for where a guard flips: below, at and above each block
#: width, plus two basis-function counts (20 for order 3, 35 for order 4).
EXTENTS = [9, 12, 16, 17, 20, 32, 33, 35, 48, 56]


@pytest.mark.parametrize('kind', ['single', 'chain', 'slices', 'mixed'])
@pytest.mark.parametrize('m', EXTENTS)
def test_the_atomic_path_computes_what_the_plain_one_does(kind, m):
    n = 4
    descrs = _descrs(kind, m, n, Datatype.F32)
    plain = _render(descrs, Datatype.F32, False)
    atomic = _render(descrs, Datatype.F32, True)
    if 'atomicAdd' not in atomic:
        pytest.skip('no atomic emitted for this shape')
    lanes = _lanes(atomic)
    assert lanes == _lanes(plain), 'the two paths disagree on the block width'

    # The destination is zeroed so a doubled contribution shows as a factor of
    # two rather than as `D + 2AB` against `D + AB`, which is a ratio that
    # depends on the seed and hides the size of the error.
    a = evaluate_wave(plain, lanes, seed=SEED, globals_only=True,
                      preset={'m0': 0.0})
    b = evaluate_wave(atomic, lanes, seed=SEED, globals_only=True,
                      preset={'m0': 0.0})
    bad = {k for k in set(a) | set(b)
           if abs((a.get(k) or 0.0) - (b.get(k) or 0.0)) > TOL}
    assert not bad, (
        f'{len(bad)} slots differ; e.g. '
        + ', '.join(f'{k}: plain={a.get(k)!r} atomic={b.get(k)!r}'
                    for k in sorted(bad)[:3]))


def test_the_interpreter_actually_performs_the_atomic():
    """Otherwise every assertion above passes by doing nothing.

    `kernel_eval` used to abort on an `atomicAdd` statement, so a differential
    like the one above would have failed loudly -- but a version that skipped
    it instead would have compared two runs of the same non-atomic prefix and
    agreed every time.  This pins the arithmetic: two lanes adding into one
    slot leave the sum, which is also the only place the lockstep driver's
    shared memory image is asserted on directly.
    """
    # `preset` is not decoration: an unwritten slot takes a pseudo-random
    # value derived from its identity, so without zeroing it the assertion
    # would be about the seed rather than about the addition.  That is the
    # same reason the differential above presets the destination.
    src = 'void k() {\n  atomicAdd(&m0[3], 2.5);\n}'
    mem = evaluate_wave(src, 2, seed=0, globals_only=True, preset={'m0': 0.0})
    assert mem[('m0', 3)] == 5.0, 'two lanes, one slot, one arrival counted'

    hip = ('void k() {\n  __hip_atomic_fetch_add(&m0[1], 1.5, '
           '__ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);\n}')
    assert evaluate_wave(hip, 4, seed=0, globals_only=True,
                         preset={'m0': 0.0})[('m0', 1)] == 6.0
