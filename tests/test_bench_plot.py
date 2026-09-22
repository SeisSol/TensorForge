# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""The charts draw what was measured, and stop where the measurements do.

A chart is believed harder than a table, and the ways one lies are quiet: a
gap interpolated across, a ratio taken the wrong way round for a quantity
where smaller is better, a log axis handed a zero and silently clamping it to
something plottable. None of those raise, all of them produce a picture that
looks right.

So what is pinned here is the arithmetic and the omissions -- which rows are
dropped, where a line breaks, which way a ratio points -- rather than the
appearance. Two structural properties of the SVG are checked as well, because
a chart that does not parse is not a chart.

Host-only: no GPU, no toolchain, no profiler.
"""

from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parent.parent / "tools"
for extra in (str(TOOLS / "bench"), str(TOOLS)):
    if extra not in sys.path:
        sys.path.insert(0, extra)

import plot  # noqa: E402


#: 10 TFLOP/s over 1 TB/s: a ridge at 10 flop/byte, so every figure below is
#: one a reader can check in their head.
ROOF = plot.Roof(flops_per_s=1e13, bytes_per_s=1e12, source="for the test")


def row(workload="k", batch=1024, flops=2e9, byts=1e8, ns=1e6, **extra):
    r = {
        "workload": workload, "config": "default", "batch": batch,
        "flops": flops, "bytes": byts, "intensity": flops / byts,
        "event_ns": ns, "gflops": flops / ns, "gbytes": byts / ns,
        "ns_per_element": ns / batch, "arch": "pvc", "backend": "esimd",
        "datatype": "F32",
    }
    r.update(extra)
    return r


def run(rows, label="run"):
    return plot.Run(label=label, path=Path("bench.json"), rows=list(rows))


# ----------------------------------------------------------------------
# Reading
# ----------------------------------------------------------------------

def test_both_shapes_run_py_has_written_are_read(tmp_path):
    """A bare list and `{'results': [...]}` are the same file to a reader."""
    bare = tmp_path / "a.json"
    bare.write_text(json.dumps([row()]))
    wrapped = tmp_path / "b.json"
    wrapped.write_text(json.dumps({"run": {}, "results": [row()]}))
    assert len(plot.load_run(bare).rows) == 1
    assert len(plot.load_run(wrapped).rows) == 1


def test_a_row_that_failed_is_not_a_data_point(tmp_path):
    path = tmp_path / "a.json"
    path.write_text(json.dumps(
        [row(), {"workload": "bad", "error": "did not build"}]))
    assert [r["workload"] for r in plot.load_run(path).rows] == ["k"]


def test_the_stored_rate_wins_over_recomputing_it():
    """`run.py` divides by whichever clock the run was told to use.  A chart
    that recomputes from `flops` and `event_ns` disagrees with the table the
    same file prints, which is worse than either figure alone."""
    r = row(gflops=1234.0)
    assert plot.rate(r, "gflops") == 1234.0


def test_a_row_with_no_clock_is_left_out_rather_than_drawn_at_zero():
    r = row()
    del r["event_ns"], r["gflops"]
    assert plot.rate(r, "gflops") is None


def test_the_wall_clock_serves_where_there_is_no_device_clock():
    r = row(wall_ns=2e6)
    del r["event_ns"], r["ns_per_element"]
    assert plot.rate(r, "ns_per_element") == pytest.approx(2e6 / 1024)


# ----------------------------------------------------------------------
# Series
# ----------------------------------------------------------------------

def test_two_configurations_of_one_kernel_are_two_curves():
    """Averaging them is how a configuration that wins everywhere and one
    that wins nowhere come to look alike."""
    groups = plot.by_series([row(config="default"), row(config="L16")])
    assert len(groups) == 2
    assert "k [L16]" in groups


def test_a_series_is_ordered_by_launch_size_whatever_the_file_says():
    groups = plot.by_series([row(batch=4096), row(batch=64), row(batch=512)])
    assert [r["batch"] for r in groups["k"]] == [64, 512, 4096]


# ----------------------------------------------------------------------
# The roof
# ----------------------------------------------------------------------

def test_the_ridge_is_where_the_two_roofs_meet():
    assert ROOF.ridge == pytest.approx(10.0)
    assert ROOF.bound(ROOF.ridge) == pytest.approx(ROOF.flops_per_s)


def test_below_the_ridge_the_roof_is_bandwidth_and_above_it_is_flops():
    assert ROOF.bound(1.0) == pytest.approx(1e12)
    assert ROOF.bound(1000.0) == pytest.approx(1e13)


# ----------------------------------------------------------------------
# Axes
# ----------------------------------------------------------------------

def test_a_logarithmic_axis_refuses_a_non_positive_bound():
    """Clamping it silently is how a zero-time row becomes a point at the
    left edge that a reader takes for a measurement."""
    c = plot.Canvas("t", "x", "y", xlog=True)
    with pytest.raises(ValueError):
        c.fit([0.0, 1.0], [1.0, 2.0])


def test_a_gap_breaks_the_line_rather_than_being_drawn_across():
    """An interpolated point is a statement about a launch that never
    happened.  Both sides carry two points, so a line drawn across the gap
    and two lines around it are one polyline against two."""
    c = plot.Canvas("t", "x", "y")
    c.fit([0, 5], [0, 3])
    c.line([(0, 1), (1, 1), (2, None), (3, 2), (4, 2)], "#000")
    assert sum(1 for s in c.body if s.startswith("<polyline")) == 2


def test_a_line_with_no_gap_is_one_polyline():
    c = plot.Canvas("t", "x", "y")
    c.fit([0, 5], [0, 3])
    c.line([(0, 1), (1, 1), (2, 1), (3, 2), (4, 2)], "#000")
    assert sum(1 for s in c.body if s.startswith("<polyline")) == 1


def test_a_single_point_beside_a_gap_is_a_mark_and_not_a_line():
    """One measurement is a measurement; joining it to nothing is not."""
    c = plot.Canvas("t", "x", "y")
    c.fit([0, 3], [0, 3])
    c.line([(0, 1), (1, None), (2, 1), (3, 2)], "#000")
    assert sum(1 for s in c.body if s.startswith("<polyline")) == 1
    assert sum(1 for s in c.body if s.startswith("<circle")) == 3


def test_the_legend_says_how_many_it_left_out_rather_than_overflowing():
    c = plot.Canvas("t", "x", "y", height=200)
    for i in range(80):
        c.legend.append((f"kernel-{i}", "#000"))
    out = c.render()
    assert "more, hover the marks" in out
    ET.fromstring(out)


# ----------------------------------------------------------------------
# Views
# ----------------------------------------------------------------------

def test_one_launch_size_has_no_curve_to_draw():
    """And says so, rather than drawing a single point as a trend."""
    assert plot.view_scaling(run([row(batch=1024)])) is None


def test_scaling_draws_one_curve_per_series():
    svg = plot.view_scaling(run([row(batch=64), row(batch=4096),
                                 row(workload="j", batch=64),
                                 row(workload="j", batch=4096)]))
    tree = ET.fromstring(svg)
    polylines = [e for e in tree.iter()
                 if e.tag.endswith("polyline")]
    assert len(polylines) == 2


def test_the_roofline_is_well_formed_and_has_one_mark_per_row():
    svg = plot.view_roofline(run([row(), row(workload="j")]), ROOF)
    tree = ET.fromstring(svg)
    titles = [e.text for e in tree.iter() if e.tag.endswith("title")]
    assert sum(1 for t in titles if "% of roof" in (t or "")) == 2


def test_the_comparison_points_the_right_way_for_a_cost():
    """Lower ns per element is better, so the reference over the candidate;
    the other way round labels the faster build as the slower one."""
    fast = run([row(ns_per_element=1.0)], label="fast")
    slow = run([row(ns_per_element=2.0)], label="slow")
    svg = plot.view_compare([slow, fast], "ns_per_element")
    titles = [e.text for e in ET.fromstring(svg).iter()
              if e.tag.endswith("title")]
    assert any("2.000x" in (t or "") for t in titles)


def test_the_comparison_points_the_right_way_for_a_rate():
    """Higher GFLOP/s is better, so the candidate over the reference."""
    slow = run([row(gflops=1000.0)], label="slow")
    fast = run([row(gflops=2000.0)], label="fast")
    svg = plot.view_compare([slow, fast], "gflops")
    titles = [e.text for e in ET.fromstring(svg).iter()
              if e.tag.endswith("title")]
    assert any("2.000x" in (t or "") for t in titles)


def test_a_kernel_only_one_run_has_is_not_compared():
    a = run([row(workload="both"), row(workload="only-a")], label="a")
    b = run([row(workload="both")], label="b")
    svg = plot.view_compare([a, b])
    titles = [e.text for e in ET.fromstring(svg).iter()
              if e.tag.endswith("title")]
    assert not any("only-a" in (t or "") for t in titles)


def test_one_run_is_not_a_comparison():
    assert plot.view_compare([run([row()])]) is None


# ----------------------------------------------------------------------
# The profiled views
# ----------------------------------------------------------------------

def test_traffic_is_measured_over_compulsory():
    profile = {"k": [{"kernel": "k", "duration_ns": 1.0,
                      "dram_read_bytes": 1.5e8, "dram_write_bytes": 0.5e8}]}
    svg = plot.view_traffic(run([row(byts=1e8)]), profile)
    titles = [e.text for e in ET.fromstring(svg).iter()
              if e.tag.endswith("title")]
    assert any("2.00x" in (t or "") for t in titles)


def test_a_workload_the_profiler_missed_is_left_out_not_zeroed():
    assert plot.view_traffic(run([row()]), {"other": [{"kernel": "o"}]}) is None


def test_the_hottest_kernel_of_a_unit_is_the_one_the_workload_means():
    rows = [{"kernel": "setup", "duration_ns": 5.0},
            {"kernel": "the-one", "duration_ns": 500.0}]
    assert plot.hottest(rows)["kernel"] == "the-one"


def test_the_measured_roofline_moves_the_point_to_the_measured_intensity():
    """Compulsory 20 flop/byte, but 4x the traffic actually crossed, so the
    point belongs at 5 -- on the bandwidth slope rather than on the flat."""
    profile = {"k": [{"kernel": "k", "duration_ns": 1.0,
                      "dram_read_bytes": 4e8, "dram_write_bytes": 0.0}]}
    svg = plot.view_measured_roofline(
        run([row(flops=2e9, byts=1e8)]), ROOF, profile)
    titles = [e.text for e in ET.fromstring(svg).iter()
              if e.tag.endswith("title")]
    assert any("measured 5.00 flop/byte" in (t or "") for t in titles)
    assert any("compulsory 20.00 flop/byte" in (t or "") for t in titles)


def test_an_occupancy_given_as_a_fraction_and_as_a_percent_agree():
    """`profile.py` normalizes to a fraction, but not every adapter's metric
    does; a chart that plots 0.85 and 85 on one axis is unreadable."""
    svg = plot.view_occupancy(
        run([row()]), {"k": [{"kernel": "k", "duration_ns": 1.0,
                              "occupancy": 0.85}]})
    titles = [e.text for e in ET.fromstring(svg).iter()
              if e.tag.endswith("title")]
    assert any("85.0 % occupancy" in (t or "") for t in titles)


# ----------------------------------------------------------------------
# Assembly
# ----------------------------------------------------------------------

def test_a_missing_profile_skips_its_views_and_keeps_the_rest():
    charts, skipped = plot.build_charts([run([row()])], ROOF, None, None)
    drawn = {name for name, _, _ in charts}
    assert "roofline" in drawn
    assert not {"traffic", "occupancy"} & drawn
    assert any(s.startswith("traffic:") for s in skipped)


def test_no_ceilings_skips_the_roof_views_and_says_which_flags_would_help():
    _, skipped = plot.build_charts([run([row()])], None, None, None)
    assert any("--peak-flops" in s for s in skipped)


def test_every_drawn_chart_parses_and_the_page_carries_all_of_them():
    rows = [row(batch=64), row(batch=4096),
            row(workload="j", batch=64), row(workload="j", batch=4096)]
    profile = {"k": [{"kernel": "k", "duration_ns": 1.0, "occupancy": 0.5,
                      "dram_read_bytes": 2e8, "dram_write_bytes": 0.0}]}
    charts, _ = plot.build_charts(
        [run(rows, "a"), run(rows, "b")], ROOF, profile, None)
    assert len(charts) >= 8
    for name, _, svg in charts:
        ET.fromstring(svg)                      # every one is well-formed
    page = plot.index_html(charts, "t")
    for name, _, _ in charts:
        assert name in page
