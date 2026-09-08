# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""The profiler adapters build the command they meant to build.

None of this needs a GPU, a profiler or a toolchain, which is the point: the
machines that have `ncu` are not the machines anyone edits this on, and a flag
that is wrong is otherwise discovered by a queued job failing in half an hour.
What can be checked here is the part that is arithmetic and structure -- which
dispatches the sample window names, that the two vendors' ways of spelling that
window mean the same window, that the driver arguments are in the order the
driver reads them, and that a unit conversion happens where the vendor's unit
differs from the normalised one.

What cannot be checked here is whether `dram__bytes_read.sum` is still a metric
name. `--probe` asks the installed tool that, and `--dry-run` prints the
command for someone to read on a machine that has it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parent.parent / "tools"
for extra in (str(TOOLS / "bench"), str(TOOLS)):
    if extra not in sys.path:
        sys.path.insert(0, extra)

import profile as bench_profile  # noqa: E402


WARMUP, ITERS, BATCH = 5, 3, 65536
SYMBOL = "kernel_kernel_deadbeef"


def _commands(tool: str, tmp_path: Path):
    profiler = bench_profile.PROFILERS[tool]
    return profiler.commands(Path("/bin/bench"), "gemm_square_16", BATCH,
                             ITERS, WARMUP, SYMBOL, tmp_path,
                             profiler.metrics)


def _collecting(tool: str, tmp_path: Path):
    """The step that runs the program.

    Not always the only step: VTune collects and then reports, and only the
    first of those carries the driver.
    """
    for command in _commands(tool, tmp_path):
        if "/bin/bench" in command:
            return command
    raise AssertionError(f"{tool}: no step invokes the program")


# -- the driver's own arguments --------------------------------------------- #

@pytest.mark.parametrize("tool", sorted(bench_profile.PROFILERS))
def test_the_driver_is_invoked_in_profile_mode_with_the_arguments_it_reads(
        tool, tmp_path):
    """`driver_bench` reads argv as mode, workload, batch, iters, warmup,
    device. Getting the order wrong here does not fail: the driver parses
    `atoll` of a workload name as zero, runs no iterations, and the profiler
    dutifully reports on nothing."""
    cmd = _collecting(tool, tmp_path)
    tail = cmd[cmd.index("/bin/bench"):]
    assert tail == ["/bin/bench", "profile", "gemm_square_16", str(BATCH),
                    str(ITERS), str(WARMUP), "0"]


@pytest.mark.parametrize("tool", sorted(bench_profile.PROFILERS))
def test_the_tool_and_the_program_are_separated_the_way_that_tool_wants(
        tool, tmp_path):
    """`rocprofv3` and `unitrace` need `--`; `ncu` takes the program bare and
    would treat `--` as a file name."""
    cmd = _collecting(tool, tmp_path)
    if tool == "ncu":
        assert "--" not in cmd
    else:
        assert cmd[cmd.index("/bin/bench") - 1] == "--"


# -- the sample window ------------------------------------------------------ #

def test_both_vendors_name_the_same_dispatches(tmp_path):
    """The driver dispatches `warmup + iters` times and the sample is the tail.

    Nsight counts skipped launches, ROCm names a one-based inclusive range, and
    the two spellings have to pick out the same launches or a comparison
    between a CUDA and a ROCm collection is comparing a warm kernel with a cold
    one. Pinned together, in one test, because the invariant is the agreement
    and not either arithmetic on its own.
    """
    ncu = _collecting("ncu", tmp_path)
    assert ncu[ncu.index("--launch-skip") + 1] == str(WARMUP)
    assert ncu[ncu.index("--launch-count") + 1] == str(ITERS)

    roc = _collecting("rocprofv3", tmp_path)
    lo, hi = roc[roc.index("--kernel-iteration-range") + 1].split("-")
    # one-based and inclusive: dispatches WARMUP+1 .. WARMUP+ITERS
    assert int(lo) == WARMUP + 1
    assert int(hi) == WARMUP + ITERS
    assert int(hi) - int(lo) + 1 == ITERS


def test_the_warm_up_launch_is_never_in_the_sample(tmp_path):
    """The launcher does its occupancy query and `cudaFuncSetAttribute` on the
    first call only, so dispatch one is a different measurement from the rest.
    A zero warm-up is the caller's business; that the window starts after
    whatever warm-up was asked for is not."""
    for warmup in (0, 1, 17):
        roc = bench_profile.PROFILERS["rocprofv3"].commands(
            Path("/bin/bench"), "w", BATCH, ITERS, warmup, SYMBOL, tmp_path,
            ())[0]
        lo = int(roc[roc.index("--kernel-iteration-range") + 1].split("-")[0])
        assert lo == warmup + 1


@pytest.mark.parametrize("tool", sorted(bench_profile.PROFILERS))
def test_the_symbol_filter_is_used_wherever_the_tool_has_one(tool, tmp_path):
    """A binary holds every workload of its configuration.

    What keeps a collection to one kernel is the invocation -- the driver runs
    one workload per process -- and the filter is a second line, excluding the
    runtime's own dispatches. VTune has no collection-time filter and relies on
    the first line alone, which is why the capability is declared rather than
    assumed: a tool silently not filtering looks exactly like a tool filtering
    correctly until the day the binary dispatches twice.
    """
    profiler = bench_profile.PROFILERS[tool]
    cmd = _collecting(tool, tmp_path)
    if profiler.kernel_filter:
        assert any(SYMBOL in part for part in cmd)
    else:
        assert cmd[cmd.index("/bin/bench") + 1] == "profile"
        assert cmd[cmd.index("/bin/bench") + 2] == "gemm_square_16"


def test_a_command_is_printable_without_the_tool_installed(tmp_path):
    """`--dry-run` exists to be read on a machine that has the profiler, which
    is rarely the machine that has the compiler. A `None` in the argv makes the
    printing itself fail."""
    for tool in bench_profile.PROFILERS:
        for cmd in _commands(tool, tmp_path):
            assert all(isinstance(part, str) for part in cmd)
            assert " ".join(cmd)


# -- reading what comes back ------------------------------------------------ #

NCU_CSV = '''==PROF== Connected to process 4711
"ID","Kernel Name","Metric Name","Metric Unit","Metric Value"
"0","kernel_kernel_deadbeef","gpu__time_duration.sum","ns","12,345.60"
"0","kernel_kernel_deadbeef","dram__bytes_read.sum","byte","1048576"
"0","kernel_kernel_deadbeef","launch__grid_size","","108"
"0","kernel_kernel_deadbeef","sm__throughput.avg.pct_of_peak","%","n/a"
'''

ROCPROF_CSV = '''"Correlation_Id","Dispatch_Id","Kernel_Name","Grid_Size","Counter_Name","Counter_Value"
"1","6","kernel_kernel_deadbeef","65536","FETCH_SIZE","2048"
"1","6","kernel_kernel_deadbeef","65536","WRITE_SIZE","1024"
'''


def test_both_vendors_csv_is_the_same_long_shape():
    """One row per kernel per metric, with the headings spelled differently.
    Two readers would be two places for the same tolerance to be missing from
    one of them."""
    ncu, _ = bench_profile.read_long_csv(NCU_CSV)
    roc, _ = bench_profile.read_long_csv(ROCPROF_CSV)
    assert ncu["kernel_kernel_deadbeef"]["dram__bytes_read.sum"] == 1048576
    assert roc["kernel_kernel_deadbeef"]["FETCH_SIZE"] == 2048


def test_a_banner_before_the_header_is_skipped():
    """`ncu` writes progress lines above its CSV. Refusing the file over them
    throws away a collection nobody can repeat without the machine."""
    per_kernel, _ = bench_profile.read_long_csv(NCU_CSV)
    assert per_kernel


def test_thousands_separators_and_unavailable_values_survive():
    """A value of `n/a` is one metric missing, not a broken file."""
    per_kernel, note = bench_profile.read_long_csv(NCU_CSV)
    values = per_kernel["kernel_kernel_deadbeef"]
    assert values["gpu__time_duration.sum"] == pytest.approx(12345.6)
    assert "sm__throughput.avg.pct_of_peak" not in values
    assert "skipped" in note


def test_a_file_with_no_metric_column_says_so_rather_than_returning_nothing():
    rows, note = bench_profile.read_long_csv("a,b\n1,2\n")
    assert rows == {}
    assert "header" in note


# -- units ------------------------------------------------------------------ #

def test_kilobytes_become_bytes_and_percent_becomes_a_fraction():
    """ROCm reports `FETCH_SIZE` in kilobytes and Nsight `dram__bytes_read.sum`
    in bytes. Mixing them silently is a factor of 1024 in a column nobody
    re-derives."""
    roc, _ = bench_profile.read_long_csv(ROCPROF_CSV)
    rows = bench_profile.normalise(
        roc, bench_profile.PROFILERS["rocprofv3"].metrics)
    assert rows[0]["dram_read_bytes"] == 2048 * 1024
    assert rows[0]["dram_write_bytes"] == 1024 * 1024

    ncu, _ = bench_profile.read_long_csv(NCU_CSV)
    rows = bench_profile.normalise(ncu,
                                   bench_profile.PROFILERS["ncu"].metrics)
    assert rows[0]["dram_read_bytes"] == 1048576


def test_the_vendor_row_is_kept_whole():
    """Only a short set is normalised; everything else stays under `vendor` so
    a metric this does not understand is still in the report."""
    ncu, _ = bench_profile.read_long_csv(NCU_CSV)
    row = bench_profile.normalise(ncu,
                                  bench_profile.PROFILERS["ncu"].metrics)[0]
    assert "launch__grid_size" in row["vendor"]
    assert row["grid_size"] == 108


def test_every_normalised_key_a_metric_claims_is_in_the_declared_set():
    """`NORMALISED` is what a consumer may rely on. A metric normalising to a
    key outside it is a column that appears in one vendor's rows and no
    other's, which is the shape of an accidental cross-vendor comparison."""
    for name, profiler in bench_profile.PROFILERS.items():
        for metric in profiler.metrics:
            assert metric.key in bench_profile.NORMALISED, (
                f"{name}: {metric.expr} normalises to {metric.key!r}")


def test_each_vendor_has_a_profiler_and_each_profiler_a_metric_set():
    assert set(bench_profile.BY_VENDOR) == {"nvidia", "amd", "intel"}
    for vendor, tool in bench_profile.BY_VENDOR.items():
        assert tool in bench_profile.PROFILERS, vendor
    for name, profiler in bench_profile.PROFILERS.items():
        assert profiler.metrics, f"{name} normalises nothing"


def test_a_multi_step_tool_runs_the_program_once():
    """VTune collects and then reports. A second step that also carried the
    driver would run the workload twice and report on the second run's
    directory, which the first step's counters are not in."""
    steps = _commands("vtune", Path("/tmp"))
    assert len(steps) == 2
    assert sum("/bin/bench" in step for step in steps) == 1
    assert "-report" in steps[1]


def test_the_intel_default_needs_no_licence():
    """`unitrace` is free and `vtune` is not, so an unqualified Intel run
    reaches for the one that is there. The memory counters are behind
    `--tool vtune`, and that is a choice the caller makes knowingly."""
    assert bench_profile.BY_VENDOR["intel"] == "unitrace"
