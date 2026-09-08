# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""An atomic add needs the store nest to write each element exactly once.

A plain store needs *coverage*: every destination element written at least
once, and an element written twice with the same value costs nothing.  An
atomic add needs *exactness*: written once, no more.  The store nest was only
ever built to the first requirement, so the places where it writes an element
from more than one lane are correct under `=` and multiply the contribution
under `+=`.

There is one such place and the widened lead path is what reaches it.  A lead
extent the vector width does not divide leaves `extent % width` elements no
whole vector covers, and `LeadLoop._peel` hands each of them to the store as a
plain integer.  `_peel`'s own docstring says `Symbol.store` guards that write
to the lane that owns the element -- true for a register destination and not
for a global one, where the guard is deliberately absent because global memory
is addressed by every lane and *which* lane owns an element is not a question
with an answer.  So the whole wave stores it.

Reproduced before this was gated, on gfx90a with `TF_LEAD_VEC=1`, from a
35-element lead dimension over aligned operands with `beta=1.0`::

    #pragma unroll
    for (int32_t v561_i1 = 0; v561_i1 < 4; ++v561_i1) {
      float v564_data = r2[(v561_i1 * 2)];
      int32_t v567_a = 34 + (v561_i1 * 35);
      __builtin_amdgcn_global_atomic_fadd_f32(
          &glb_m0[v567_a], (tensorforge::broadcast<32, 1, 17>(v564_data)));
    }

No guard: 32 lanes, one address, the same value broadcast to all of them.  The
35th element of every column comes out 32 times its contribution.

The guarded block above it had a second defect of its own, which is why the
gate is on the width rather than on the tail alone: it passes
`VectorT<float, 2>` to a builtin declared `float(float address_space<1> *,
float)`.  AMD has no packed FP32 atomic add to widen to -- no builtin, no
subtarget feature -- so the width has nowhere to go here even when the tail
divides.
"""

from __future__ import annotations

import pytest

from tensorforge.backend.instructions.memory import vectorize
from tensorforge.backend.placement import (POLICIES, atomic_write_is_exact,
                                           result_is_atomic)
from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.context import Context
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators.descriptions import GemmDescr
from tensorforge.generators.generator import Generator

AMD = POLICIES['amd']

#: 35 is an order-4 basis-function count and the extent that makes the peel
#: run: at width 2 the whole vectors cover 34 and the 35th has no partner.
M, N, K = 35, 4, 8
ALIGNMENT = 16


def _t(shape, alias):
    return SubTensor(Tensor(shape, Addressing.STRIDED,
                            BoundingBox([0] * len(shape), list(shape)),
                            alias=alias, datatype=Datatype.F32,
                            alignment=ALIGNMENT))


def _accumulating_gemm():
    """`D += A B`, aligned, with a lead extent the width does not divide."""
    return [GemmDescr(trans_a=False, trans_b=False,
                      a=_t([M, K], 'A'), b=_t([K, N], 'B'), c=_t([M, N], 'D'),
                      alpha=1.0, beta=1.0)]


def _generate(arch='gfx90a', backend='hip'):
    ctx = Context(arch=arch, backend=backend, fp_type=Datatype.F32)
    gen = Generator(_accumulating_gemm(), ctx)
    gen.register()
    gen.generate()
    return gen.get_kernel()


# --------------------------------------------------------------------------- #
# The condition
# --------------------------------------------------------------------------- #

def test_a_scalar_lead_writes_exactly_once():
    assert atomic_write_is_exact(lead_width=1, lead_extent=32)


@pytest.mark.parametrize('width', [2, 4])
def test_a_widened_lead_does_not(width):
    assert not atomic_write_is_exact(lead_width=width, lead_extent=35)


def test_exactness_is_required_on_its_own():
    """Supported, accumulating, nothing pending, and still not atomic.

    Each of the four conditions has to be able to refuse by itself; folding
    this one into `supported` would tie it to the architecture, which is the
    one thing it does not depend on.
    """
    assert not result_is_atomic(accumulating=True, pending_is_atomic=True,
                                supported=True, exact=False, policy=AMD)
    assert result_is_atomic(accumulating=True, pending_is_atomic=True,
                            supported=True, exact=True, policy=AMD)


# --------------------------------------------------------------------------- #
# The kernel it protects
# --------------------------------------------------------------------------- #

def test_the_scalar_path_still_accumulates_atomically():
    """The gate must not turn the feature off where it was right.

    With the width at 1 the nest is head, whole slots and a lane-guarded tail,
    each element in exactly one of them -- so gfx90a keeps its atomic add and
    this test fails if the condition is written too broadly.
    """
    assert '__builtin_amdgcn_global_atomic_fadd_f32' in _generate()


def test_no_atomic_survives_the_widened_lead(monkeypatch):
    """The reproduction, as a test.

    `LEAD_VECTORIZE` is read at call time rather than captured at import, so
    setting it here is enough to put the generator on the widened path -- and
    leaving it to the environment variable would make this test pass by not
    running.
    """
    monkeypatch.setattr(vectorize, 'LEAD_VECTORIZE', True)
    src = _generate()
    assert 'VectorT<float, 2>' in src, (
        'the widened path did not engage, so this test proves nothing')
    assert 'atomic' not in src


def test_the_peeled_element_is_what_would_have_been_wrong(monkeypatch):
    """Names the statement, so a fix that guards it can retire this gate.

    The peel is still emitted and still unguarded; it is a plain store now,
    where writing one value from every lane is exactly the redundancy a store
    is allowed. When `Symbol.store` learns to give a global destination's
    fixed lead element an owning lane, this assertion is what stops holding
    and `atomic_write_is_exact` can narrow to the tail and the packed widths.
    """
    monkeypatch.setattr(vectorize, 'LEAD_VECTORIZE', True)
    src = _generate()
    peeled = [line for line in src.splitlines() if 'broadcast<' in line
              and 'glb_' in line]
    assert peeled, 'no peeled global write found; the shape has changed'
    assert all('=' in line and 'atomic' not in line for line in peeled)


def test_a_dividing_extent_is_refused_by_the_other_condition(monkeypatch):
    """36 elements at width 2: no peel, and still no atomic.

    The two conditions are separable now and this is what shows it.  At 35 the
    tail leaves one element over and `atomic_write_is_exact` refuses; at 36 it
    divides, the nest is exact, and what refuses instead is
    `atomics.native_add` -- AMD has no packed FP32 add for the width to go to.

    Worth pinning because the two used to be one condition.  If a later change
    lifts the peel and this still refuses, that is correct and the reason is
    the table; if it stops refusing without a packed add appearing, something
    has widened that should not have.
    """
    monkeypatch.setattr(vectorize, 'LEAD_VECTORIZE', True)
    descrs = [GemmDescr(trans_a=False, trans_b=False,
                        a=_t([36, K], 'A'), b=_t([K, N], 'B'),
                        c=_t([36, N], 'D'), alpha=1.0, beta=1.0)]
    ctx = Context(arch='gfx90a', backend='hip', fp_type=Datatype.F32)
    gen = Generator(descrs, ctx)
    gen.register()
    gen.generate()
    src = gen.get_kernel()

    assert 'VectorT<float, 2>' in src, 'the widened path did not engage'
    assert 'broadcast<' not in src.split('store{r>g}')[-1], (
        'a peel was emitted for an extent the width divides')
    assert 'atomic' not in src

    from tensorforge.backend import atomics
    from tensorforge.backend.placement import atomic_write_is_exact
    assert atomic_write_is_exact(lead_width=2, lead_extent=36), (
        'the nest is exact here; the refusal has to come from the capability')
    assert not atomics.native_add(ctx, Datatype.F32, 2)
