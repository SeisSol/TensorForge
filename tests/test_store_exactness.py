# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""An atomic add needs each element written once; now the nest does that.

A plain store needs *coverage*: every destination element written at least
once, and an element written twice with the same value costs a store and
nothing else.  An atomic add needs *exactness*: written once, no more.  The
nest met the first at every width and the second only at width 1, for one
reason -- the peeled tail.

An extent the vector width does not divide leaves `extent % width` elements no
whole vector covers, and `LeadLoop._peel` hands each to the store as a plain
integer.  For a register destination `Symbol.store` guards that write to the
lane that owns the element; for a global one it did not, so the whole wave
stored it.  Under `=` that is the same value written `threads` times and the
result is right, which is why it sat there; under `+=` the contribution is
counted `threads` times.

`StoreRegToGlb` guards it now, and guards the *load* with it rather than the
store alone -- which is what makes the cross-lane read disappear instead of
moving.  The broadcast exists so every lane has an element only one of them
holds; with the write guarded to that lane it is reading its own register, and
`readlane` is `__shfl_sync` over the full warp mask, so leaving it outside the
branch would have been a shuffle whose partners never arrive.

What is left deciding an atomic is the capability, and this file checks that
the two ends meet: no packed FP32 add on AMD, so a widened lead declines
there; `float2` from sm_90, so it does not decline on Blackwell.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from tensorforge.backend import atomics, placement
from tensorforge.backend.instructions.memory import vectorize
from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.context import Context, Options
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators.descriptions import GemmDescr
from tensorforge.generators.generator import Generator

K = 8
ALIGNMENT = 16


def _t(shape, alias):
    return SubTensor(Tensor(shape, Addressing.STRIDED,
                            BoundingBox([0] * len(shape), list(shape)),
                            alias=alias, datatype=Datatype.F32,
                            alignment=ALIGNMENT))


def _generate(monkeypatch, M, arch, backend, width=2, nvidia_atomics=False):
    """`D += A B`, aligned, at `width`."""
    monkeypatch.setattr(vectorize, 'VALIDATED_LEAD_WIDTH', width)
    if nvidia_atomics:
        monkeypatch.setitem(placement.POLICIES, 'nvidia',
                            replace(placement.POLICIES['nvidia'],
                                    atomic_accumulation=True))
    descrs = [GemmDescr(trans_a=False, trans_b=False,
                        a=_t([M, K], 'A'), b=_t([K, 3], 'B'),
                        c=_t([M, 3], 'D'), alpha=1.0, beta=1.0)]
    ctx = Context(arch=arch, backend=backend, fp_type=Datatype.F32,
                  options=Options(lead_vectorize=True))
    gen = Generator(descrs, ctx)
    gen.register()
    gen.generate()
    return gen.get_kernel()


def _peel_block(src):
    """What the store emits after its guarded main nest."""
    return src[src.rindex('store{r>g}'):]


# --------------------------------------------------------------------------- #
# The peel
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('extent', [33, 35])
def test_the_peeled_write_is_guarded_to_one_lane(monkeypatch, extent):
    """35 elements at width 2 leaves one, and one lane writes it."""
    src = _generate(monkeypatch, extent, 'gfx90a', 'hip')
    assert 'VectorT<float, 2>' in src, 'the widened path did not engage'
    peel = _peel_block(src)
    assert f'== {(extent - 1) // 2}' in peel, (
        'the peeled write names no owning lane')


@pytest.mark.parametrize('extent', [33, 35])
def test_the_shuffle_disappears_rather_than_moving(monkeypatch, extent):
    """`readlane` is `__shfl_sync` over the full warp mask.

    Guarding the store alone would have put it in a branch whose partner lanes
    never arrive.  Guarding the load with it removes it: the writing lane
    reads its own register.
    """
    src = _generate(monkeypatch, extent, 'sm_86', 'cuda')
    assert 'readlane' not in _peel_block(src)


# --------------------------------------------------------------------------- #
# What decides an atomic now
# --------------------------------------------------------------------------- #

def test_amd_declines_a_widened_lead_on_the_capability(monkeypatch):
    """Not on the nest any more.  There is no `global_atomic_pk_add_f32` --
    no builtin, and no subtarget feature to gate one on."""
    src = _generate(monkeypatch, 32, 'gfx90a', 'hip')
    assert 'VectorT<float, 2>' in src
    assert 'atomic' not in src

    ctx = Context(arch='gfx90a', backend='hip', fp_type=Datatype.F32)
    assert atomics.native_add(ctx, Datatype.F32, 1)
    assert not atomics.native_add(ctx, Datatype.F32, 2)


def test_blackwell_accumulates_a_packed_pair(monkeypatch):
    """The other end of the same question, and the reason it is a table.

    `float2` atomics arrive with sm_90, so a widened lead is not refused
    there -- and the cast to the vector overload is what makes the value,
    a GNU `VectorT<float, 2>`, reach a builtin declared over `float2`.
    """
    src = _generate(monkeypatch, 32, 'sm_90', 'cuda', nvidia_atomics=True)
    assert 'atomicAdd(reinterpret_cast<float2*>' in src


def test_ampere_declines_the_same_shape(monkeypatch):
    """Which says the refusal is the architecture and not the width."""
    src = _generate(monkeypatch, 32, 'sm_86', 'cuda', nvidia_atomics=True)
    assert 'VectorT<float, 2>' in src
    assert 'atomicAdd' not in src


def test_the_scalar_path_still_accumulates_atomically(monkeypatch):
    """The gate must not have turned the feature off where it was right."""
    src = _generate(monkeypatch, 32, 'gfx90a', 'hip', width=1)
    assert '__builtin_amdgcn_global_atomic_fadd_f32' in src
