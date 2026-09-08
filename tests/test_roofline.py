# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""The roofline places a point where the point belongs.

A roofline is a picture, and a picture is believed. The arithmetic that puts a
dot somewhere is four lines long and every one of them is the kind that is
wrong in a way nobody checks: a ridge point computed the wrong way round, a
`min` that should be a `max`, a regime label that says *memory bound* for
everything because the comparison is against the wrong roof. So the arithmetic
is pinned here, along with the two things about the emitted microbenchmark that
would quietly make its numbers meaningless.

Host-only: no GPU, no toolchain, no profiler.
"""

from __future__ import annotations

import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parent.parent / "tools"
for extra in (str(TOOLS / "bench"), str(TOOLS)):
    if extra not in sys.path:
        sys.path.insert(0, extra)

import roofline as rl  # noqa: E402
from tensorforge.common.basic_types import Datatype  # noqa: E402


#: 10 TFLOP/s over 1 TB/s: a ridge at 10 flop/byte, chosen so every expected
#: figure below is one a reader can do in their head.
CEILING = rl.Ceiling(backend="cuda", arch="sm_80", datatype="F32",
                     device="test", compute_units=100,
                     flops_per_s=1e13, bytes_per_s=1e12)


def _row(**kw):
    row = {"workload": "w", "config": "baseline", "batch": 1024,
           "datatype": "F32", "flops": 0, "intensity": 1.0,
           "event_ns": 1000.0}
    row.update(kw)
    return row


# -- the roof --------------------------------------------------------------- #

def test_the_ridge_is_where_the_two_roofs_meet():
    assert CEILING.ridge == pytest.approx(10.0)
    assert CEILING.bound(CEILING.ridge) == pytest.approx(CEILING.flops_per_s)


def test_below_the_ridge_the_roof_is_bandwidth_and_above_it_is_flops():
    """The one line that is wrong if the `min` is a `max`, and the failure is
    invisible: every point still lands under a roof, just not this machine's."""
    assert CEILING.bound(1.0) == pytest.approx(1e12)      # 1 flop/byte
    assert CEILING.bound(100.0) == pytest.approx(1e13)    # past the ridge
    assert CEILING.bound(5.0) < CEILING.bound(20.0)


def test_a_machine_with_no_measured_bandwidth_has_no_ridge_rather_than_a_crash():
    dead = rl.Ceiling("cuda", "sm_80", "F32", "d", 1, 1e13, 0.0)
    assert dead.ridge == 0.0


# -- placing a point -------------------------------------------------------- #

def test_a_kernel_at_the_bandwidth_roof_is_called_memory_bound():
    """1 flop/byte puts the roof at 1 TFLOP/s; 10^9 flops in a millisecond
    reaches it."""
    point, = rl.place([_row(flops=1_000_000_000, intensity=1.0,
                            event_ns=1_000_000.0)], CEILING)
    assert point.achieved == pytest.approx(1e12)
    assert point.fraction == pytest.approx(1.0)
    assert point.regime == "memory"


def test_a_kernel_past_the_ridge_at_the_flop_roof_is_called_compute_bound():
    point, = rl.place([_row(flops=10_000_000_000, intensity=100.0,
                            event_ns=1_000_000.0)], CEILING)
    assert point.achieved == pytest.approx(1e13)
    assert point.regime == "compute"


def test_a_kernel_far_under_its_roof_is_neither():
    """The interesting case for batched small operators, and the one a
    two-label scheme gets wrong: a kernel at four percent of the bandwidth roof
    is not bandwidth bound, it is bound by something the roofline does not
    draw. Calling it *memory bound* sends a reader to optimise reuse that was
    never the constraint."""
    point, = rl.place([_row(flops=40_000_000, intensity=1.0,
                            event_ns=1_000_000.0)], CEILING)
    assert point.fraction == pytest.approx(0.04)
    assert point.regime == "neither"


def test_the_regime_boundary_is_the_declared_one():
    """Pinned against the constant rather than a literal, so moving the
    threshold is one edit and not a hunt."""
    # roof at 1 flop/byte is 1 TFLOP/s; a hair over half of it in 1 ms
    just_above = rl.place([_row(flops=int(1e9 * rl.FAR_BELOW * 1.01),
                                intensity=1.0, event_ns=1e6)], CEILING)
    assert just_above[0].regime != "neither"


def test_a_row_with_no_clock_is_left_out_rather_than_plotted_at_zero():
    """`run.py` reports `event_ns: null` on SYCL and an `error` key for a
    workload that produced nothing. Either one plotted as a zero is a dot on
    the axis that looks like a very slow kernel."""
    assert rl.place([_row(event_ns=None, wall_ns=None, flops=10)],
                    CEILING) == []
    assert rl.place([_row(error="timeout")], CEILING) == []
    assert rl.place([_row(intensity=None)], CEILING) == []


def test_the_wall_clock_is_used_when_there_is_no_device_clock():
    """SYCL has no event to query, and leaving those rows off the plot
    entirely would mean the Intel stack never gets a roofline."""
    point, = rl.place([_row(event_ns=None, wall_ns=1_000_000.0,
                            flops=1_000_000_000, intensity=1.0)], CEILING)
    assert point.achieved == pytest.approx(1e12)


# -- the picture ------------------------------------------------------------ #

def test_the_svg_is_well_formed_and_has_one_dot_per_point():
    points = rl.place([_row(flops=1_000_000_000, intensity=1.0,
                            event_ns=1e6),
                       _row(flops=10_000_000_000, intensity=100.0,
                            event_ns=1e6)], CEILING)
    tree = ET.fromstring(rl.svg(points, CEILING))
    ns = "{http://www.w3.org/2000/svg}"
    assert len(tree.findall(f"{ns}circle")) == len(points) == 2
    assert tree.findall(f"{ns}polyline"), "no roof drawn"


def test_the_svg_survives_having_nothing_to_plot():
    """A run in which every workload failed still produces a report, and a
    log10 of an empty minimum is how that turns into a traceback instead."""
    ET.fromstring(rl.svg([], CEILING))


# -- the microbenchmark source ---------------------------------------------- #

@pytest.mark.parametrize("backend", ["cuda", "hip", "oneapi", "acpp", "esimd"])
def test_the_emitted_source_has_no_leftover_placeholders(backend):
    """The template is `%`-substituted and its printf formats have to be
    doubled. A stray `%s` reaching the compiler is a build failure on the
    cluster; a stray one reaching printf is a wrong number."""
    src = rl.emit_ceiling(backend, Datatype.F32)
    assert not re.findall(r"%\([a-z]+\)s", src)
    assert 'printf("{\\"flops_per_s\\": %.6e' in src


def test_hip_gets_its_own_spelling_of_the_properties_struct():
    """CUDA calls it `cudaDeviceProp` and HIP `hipDeviceProp_t`. Prefix
    substitution does not reach the trailing `_t`, and the result compiles
    everywhere except the machine it was for."""
    assert "hipDeviceProp_t" in rl.emit_ceiling("hip", Datatype.F32)
    assert "cudaDeviceProp " in rl.emit_ceiling("cuda", Datatype.F32)


def test_the_first_repetition_is_discarded():
    """It carries the module load, and on a JIT stack the compile. Averaging it
    in makes a ceiling that is low by however long the compiler took."""
    for backend in ("cuda", "oneapi"):
        assert "if (rep > 0)" in rl.emit_ceiling(backend, Datatype.F32)


def test_repetitions_are_reduced_with_a_maximum_and_not_a_mean():
    """A ceiling is the best the machine did. A mean over a run that was
    descheduled once reports the scheduler."""
    src = rl.emit_ceiling("cuda", Datatype.F32)
    assert "fma_best > flops / s ? fma_best : flops / s" in src
    assert "bw_best > bytes / s ? bw_best : bytes / s" in src


def test_a_type_the_machine_emulates_is_refused():
    """A roof for a type the hardware does not have is a property of the
    emulation and not of the machine, and putting it on the same axes as a
    native one invites a comparison that means nothing."""
    with pytest.raises(NotImplementedError, match="ceiling microbenchmark"):
        rl.emit_ceiling("cuda", Datatype.F16)


def test_the_flop_count_matches_the_loop_it_counts():
    """Two flops per fused multiply-add, `ACCUMULATORS` of them per iteration.
    An accumulator added to the loop and not to the count is a ceiling too low
    by a factor nobody notices, because a ceiling has nothing to be checked
    against."""
    src = rl.emit_ceiling("cuda", Datatype.F32)
    assert f"for (int k = 0; k < {rl.ACCUMULATORS}; ++k) a[k] = a[k] * b + c;" \
        in src
    assert f"2.0 * {rl.ACCUMULATORS} * (double)iters" in src


def test_the_triad_counts_three_arrays():
    """STREAM's convention. On hardware that allocates on write the true
    traffic is four arrays; the convention is stated in the docstring rather
    than corrected, because a corrected number is not the one everyone else
    quotes."""
    assert "3.0 * (double)n * sizeof(T)" in rl.emit_ceiling("cuda",
                                                            Datatype.F32)
