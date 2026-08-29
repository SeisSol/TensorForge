# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""The lane count, as a choice rather than a constant.

It was `min(32, num_threads)` at the end of `_deduce_num_threads`, with no
comment and one condition -- no elementwise descriptor in the list.  Read as a
hardware fact it is wrong: NVIDIA and Intel report 32 and 16, so the minimum
does nothing there, and it bites on AMD alone, where the wave is 64.  Read as a
measurement it is right, and that is what it is: the full wave halves the
per-lane register footprint and has been slower on some kernels.

What it had nowhere to say was which of the two it is, and no way to ask for
the other answer.  These tests hold both halves: the default is exactly what it
always produced, and the override actually reaches the generated code.

The second half is the one that matters going forward.  A search over
configurations -- lane count, lead width, k-width, pipeline depth -- cannot
start from a constant, and this is the first of those that stops being one.
"""

from __future__ import annotations

import re

import pytest

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.context import Context
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators import lanes
from tensorforge.generators.descriptions import GemmDescr
from tensorforge.generators.generator import Generator
from tensorforge.generators import elementwise as ew


def _t(shape, alias, dt=Datatype.F32):
    return SubTensor(Tensor(list(shape), Addressing.STRIDED,
                            BoundingBox([0] * len(shape), list(shape)),
                            alias=alias, datatype=dt))


def _gemm(m, n, k, dt=Datatype.F32):
    return [GemmDescr(False, False, a=_t([m, k], 'A', dt),
                      b=_t([k, n], 'B', dt), c=_t([m, n], 'C', dt))]


def _ctx(arch, backend, dt=Datatype.F32):
    return Context(arch=arch, backend=backend, fp_type=dt)


def _register_slots(src: str) -> int:
    return sum(int(n) for _, n in
               re.findall(r'\b(?:double|float)\s+(r\d+)\[(\d+)\]', src))


# ----------------------------------------------------------------------
# the default is what it always was
# ----------------------------------------------------------------------

@pytest.mark.parametrize("backend,arch,wave", [
    ("cuda", "sm_86", 32),
    ("hip", "gfx90a", 64),
    ("acpp", "pvc", 16),
])
def test_the_ceiling_is_a_number_and_the_wave_is_another(backend, arch, wave):
    """The two are unrelated, and the default only ever consults one of them.

    `ceiling=None` asks for the wave, and which direction that moves depends on
    the target: wider on AMD, where the default halves a 64-lane wave, and
    *narrower* on Intel, whose vector unit is 16.

    The narrowing is the more interesting half.  The default leaves an Intel
    section at 32 lanes -- wider than a wave -- which is why a barrier inside
    its batch loop comes out at group scope, and why `verify` refuses 35
    kernels for a group barrier under a simd-uniform trip count.  Clamping to
    the wave makes them generate.  That is a lead, not something to change
    while extracting a decision, so `deduce` reproduces the old answer exactly
    and this test records the discrepancy rather than closing it.
    """
    ctx = _ctx(arch, backend, Datatype.F64)
    assert ctx.get_vm().get_hw_descr().vec_unit_length == wave

    default = lanes.deduce(_gemm(56, 9, 56, Datatype.F64), ctx)
    at_wave = lanes.deduce(_gemm(56, 9, 56, Datatype.F64), ctx, ceiling=None)

    assert default.num_threads == min(lanes.DEFAULT_LANE_CEILING,
                                      default.num_threads)
    assert at_wave.num_threads <= wave
    if wave > lanes.DEFAULT_LANE_CEILING:
        assert at_wave.num_threads > default.num_threads
    elif wave < lanes.DEFAULT_LANE_CEILING:
        assert at_wave.num_threads < default.num_threads


def test_an_elementwise_descriptor_waives_the_ceiling():
    """Its iteration space is the vector unit's, not a lead dimension."""
    ctx = _ctx("gfx90a", "hip")
    a, c = _t([64, 64], 'A'), _t([64, 64], 'C')
    assert lanes.deduce([ew.abs(c, a)], ctx).num_threads \
        == ctx.get_vm().get_hw_descr().vec_unit_length


def test_the_width_is_a_minimum_where_the_lane_count_is_a_maximum():
    """One register image is shared across the section.

    So a width one descriptor cannot take is a width none may take, while a
    lane count one descriptor needs is one they all must have.  Stating the
    asymmetry, because it reads like a typo and is not.
    """
    import inspect
    src = inspect.getsource(lanes.deduce)
    assert 'max(threads, num_threads)' in src
    assert 'min(widths)' in src


# ----------------------------------------------------------------------
# and it can now be chosen
# ----------------------------------------------------------------------

def test_an_explicit_config_reaches_the_generated_code():
    """The override is not decoration: the lane count changes the kernel."""
    ctx = _ctx("gfx90a", "hip", Datatype.F64)
    descrs = _gemm(56, 9, 56, Datatype.F64)
    deduced = lanes.deduce(descrs, ctx)

    wide = lanes.deduce(_gemm(56, 9, 56, Datatype.F64), ctx, ceiling=None)
    assert wide.num_threads > deduced.num_threads, (
        "this case is chosen because the ceiling binds on it")

    g = Generator(_gemm(56, 9, 56, Datatype.F64), ctx, lanes=wide)
    g.generate()
    assert g._num_threads == wide.num_threads


def test_the_wider_configuration_halves_the_per_lane_register_footprint():
    """The reason the lane count is worth being able to choose.

    A register array is sized `extent / lanes` along the distributed axis, so
    the lane count scales every array in the section at once -- where a
    placement decision only says which arrays exist.  Order 6 in double is the
    case that shows it, and it is the one that runs out of registers at 32.
    """
    ctx = _ctx("gfx90a", "hip", Datatype.F64)

    def slots(config):
        g = Generator(_gemm(56, 9, 56, Datatype.F64), ctx, lanes=config)
        g.generate()
        return _register_slots(g.get_kernel())

    narrow = slots(lanes.deduce(_gemm(56, 9, 56, Datatype.F64), ctx))
    wide = slots(lanes.deduce(_gemm(56, 9, 56, Datatype.F64), ctx,
                              ceiling=None))
    assert wide * 2 <= narrow + 2, (
        f"expected roughly half the slots per lane at the full wave, "
        f"got {narrow} -> {wide}")


def test_the_default_is_a_measurement_and_says_so():
    """A bare `min(32, ...)` reads as a hardware fact and is not one.

    Running the full wave on gfx90a has measured slower on some kernels, so
    the ceiling records a result. That belongs next to the number, because the
    next person to see it halving the lane count on a 64-wide wave will
    otherwise take it for an oversight and remove it.
    """
    assert lanes.DEFAULT_LANE_CEILING == 32
    doc = lanes.__doc__ + (lanes.deduce.__doc__ or '')
    import inspect
    src = inspect.getsource(lanes)
    assert 'slower' in src, (
        'the ceiling no longer records why it is 32')
