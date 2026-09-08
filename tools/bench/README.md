<!--
    SPDX-FileCopyrightText: 2026 SeisSol Group

    SPDX-License-Identifier: MIT
-->

# Measurement

Unlike `tools/`'s diagnostics, these build and run things. Two tools over one
suite definition: a vendor-neutral timing run that needs only a toolchain and a
device, and a profiling run that drives the vendor's own tool and collects its
reports.

They are separate on purpose. Counter access is a privilege — NVIDIA gates it
behind `NVreg_RestrictProfilingToAdminUsers`, and on a shared cluster it is
often simply unavailable — so a benchmark that depended on the profiler would
be unusable exactly where the machines are. What they share is the workload: one
driver binary, two run modes chosen by argument, so that the thing profiled is
the thing timed.

| | |
|---|---|
| `suite.py` | what to measure: workloads, batches, configurations |
| `suites/` | suite definitions |
| `build.py` | one binary per configuration, with measurement flags |
| `run.py` | the vendor-neutral timing run |
| `profile.py` | the vendor profilers, over the same binary |
| `roofline.py` | measured machine ceilings, and the kernels under them |

The driver itself is emitted by `tests/harness/driver_bench.py`, next to the
correctness driver and sharing its operand collection and its launcher call —
two drivers deriving the launcher's parameter order independently is how they
come to disagree about a signature that changed under one of them.

## Suites

A suite is a Python module, the same shape a test case is. It defines `NAME`
and `workloads()`, and may set `BATCHES`, `CONFIGS`, `TARGETS`, `DATATYPES`.

```bash
TF_BENCH_DUMP=descriptors.json python3 tools/bench/run.py suites/seissol.py
```

Two sources of workloads ship with it. `from_cases()` takes the correctness
corpus, which is the right source of shapes and the wrong source of load —
every case declares `BATCH` two to four, and the launcher sizes its grid
`min(occupancy_gridsize, numElements0)`, so at those batches a run measures
launch overhead. The suite says what batch to run at and the case's own is
dropped. `from_dump()` takes a capture from
`tools/host/dump_descriptors.py`, which is the production corpus: real
operators, real sparsity, real chain lengths.

Captures are not vendored. One is tied to an order, an equation set and a
memory layout, and a checked-in capture would still be measured long after it
stopped describing what SeisSol generates.

## Why one binary per configuration

The emitted symbol is `kernel_kernel_<md5>`, and the hash covers the descriptor
list and the flag mode. It does not cover `Options`. Three configurations of
`gemm_square_16` — default, `wide_bodies=False`, `enable_pipeline=True` —
produce three different kernel bodies under one name:

```text
baseline  kernel_30948bd44e   38427 bytes of source
narrow    kernel_30948bd44e   38605
pipeline  kernel_30948bd44e   38764
```

A profiler keys its report on that symbol. Putting two configurations in one
binary therefore produces a report in which they cannot be told apart, and the
failure is silent — two rows, one name, plausible numbers.

So the build unit is one binary per `(target, datatype, options, lane
ceiling)`, holding every workload in the suite. Inside a binary the
configuration is fixed and the hash is unique per workload, which is what a
report needs; across binaries the configuration is the binary's identity and
goes in the manifest. The batch is a runtime argument and forces no rebuild.

## Why the manifest is not optional

`kernel_30948bd44e` is not a name anyone can read. Every run writes a manifest
mapping symbol to workload, descriptors, configuration, launch geometry and the
static figures the compiler reported, and every report is joined against it.

The hash earns something in return: it is a content key over the descriptors,
so the same symbol in two runs is the same operation, and two operations cannot
collide onto one name. Comparing runs is a join, not a guess.

## Reading the output

The launcher sizes its grid `min(occupancy_gridsize, numElements0)`, so below
saturation a kernel is launch-bound and its throughput describes the runtime
rather than the code. That is why the runner sweeps the batch and prints
nanoseconds per element beside the totals: where that column has stopped
falling, the grid is full. A single batch would produce one number and no way
to tell which regime it came from.

Two clocks per row. Wall time over back-to-back launches on one stream includes
the launch overhead, which SeisSol pays for real — it dispatches thousands of
small kernels per timestep. Device event time around one launch excludes the
queueing and is what an achieved FLOP-per-second should be divided by. SYCL has
only the first: the launcher submits internally and does not return its event,
so the device clock is reported absent rather than as a zero.

## Profiling

`profile.py` runs the same binary in its `profile` mode — a fixed number of
dispatches after the warm-up, no clock of its own — under `ncu`, `rocprofv3`,
`unitrace` or `vtune`. One workload per process, which keeps a counter
collection small and keeps `rocprof-compute`, which re-runs the application
once per counter pass, to a runtime measured in seconds.

```bash
# what is installed, and which metrics it knows
python3 tools/bench/profile.py --probe

python3 tools/bench/profile.py suites/corpus.py --dry-run
python3 tools/bench/profile.py suites/seissol.py --tool rocprofv3 --out prof/
```

Only a short set is normalised: duration, DRAM bytes each way, L2 bytes,
achieved occupancy, launch geometry. Everything else stays in the vendor's own
file, kept beside the normalised rows rather than parsed. Normalising more
would mean claiming `dram__bytes_read.sum` and `FETCH_SIZE` are the same
quantity in more places than they are, and that claim fails as a table that
looks comparable and is not. Cross-vendor comparison belongs to `run.py`: a
launch counted with a steady clock is the same measurement everywhere and a
hardware counter is not.

The column worth the trouble is `x`, the traffic amplification: measured DRAM
bytes over the compulsory bytes the cost model says the operation could not
avoid. For a batched small-operator kernel that ratio is the whole question —
an `Addressing.NONE` operator matrix is read by every block and should be an L2
hit, so near one means the cache did its job and near the block count means it
did not. No timing run can tell those apart; they differ in where the bytes
came from, not in how many arrived.

Intel has two entries. `unitrace` is the default because it needs no licence
and gives kernel timings; the memory counters are behind `--tool vtune`, which
collects and then reports, hence two commands. A roofline on that stack is
Advisor's own (`advisor --collect=roofline --profile-gpu`) and is named rather
than wrapped — it runs its own calibration, and a wrapper that got the
calibration wrong would produce a plot that looks like Advisor's and is not.

Vendor command lines move between releases, so every flag lives in one adapter,
`--dry-run` prints the exact command without running it, and `--probe` asks the
installed tool which of the default metrics it knows (`ncu --query-metrics`,
`rocprofv3-avail pmc-check`). A metric the tool rejects is dropped with a note
rather than failing the run.

Counter access is a privilege. Without `CAP_PERFMON`, or with the NVIDIA
driver's `NVreg_RestrictProfilingToAdminUsers` at its default, `ncu` collects
nothing — which is a machine configuration and not something a flag here can
work around. `run.py` never needs it.

## Rooflines

The ceilings are measured, not looked up. `hw_descr_db.yml` carries what the
code generator may emit and deliberately not what the hardware sustains — a
clock rate is not a code-generation constraint, and a database with one in it
is wrong every time a machine is power-capped. A datasheet peak is worse: it is
a number at a boost clock the machine may never hold, for an instruction mix
the vendor picked, and "we are at 12% of peak" is then mostly a statement about
the divisor.

```bash
python3 tools/bench/roofline.py --ceiling-only            # just the roof
python3 tools/bench/roofline.py out/bench.json --out roof/
```

Two microbenchmarks, built with the same recipe as everything else here: an FMA
chain with enough independent accumulators to cover the pipeline, and a STREAM
triad over a buffer far past the last level of cache, counted at three arrays
the way STREAM counts it. The first repetition of each is discarded — it
carries the module load and, on a JIT stack, the compile — and the rest are
reduced with a maximum, because a ceiling is the best the machine did and a
mean over a run that was descheduled once reports the scheduler.

Where the vendor has its own roofline it is better than this one, because it
knows its own machine: `rocprof-compute --roof-only` on MI200 and newer, Intel
Advisor on the oneAPI stack. Neither is wrapped. `--peak-flops` and
`--peak-bytes` take a figure from either, and it is recorded as supplied so a
later reader can tell it from a measurement.

Points get three labels, not two. On the memory slope the lever is reuse; on
the flat it is arithmetic; and a point below half the roof its intensity allows
is bound by something the roofline does not draw — occupancy, launch overhead,
latency — which for batched small operators is the common case. Calling that
one *memory bound* would send a reader to optimise reuse that was never the
constraint.

The plot is a hand-written SVG with no plotting dependency: this directory
exists to run on a login node somebody else administers, where `pip install
matplotlib` is not always a thing that happens, and an SVG opens in a browser
and diffs as text.

## Build flags

`tests/harness/toolchain.py` compiles with no optimisation flag at all, which is
right for a correctness harness and wrong here — under `hipcc` that is `-O0`.
The measurement path builds with `-O3 -DNDEBUG` through its own recipe.
`-DNDEBUG` also turns `CHECK_ERR` into a no-op, so correctness has to have been
established elsewhere before a number from this path means anything.

## What has to be warmed up

The launcher caches its grid size in a function-local `static` and does the
occupancy query and `cudaFuncSetAttribute` on the first call only. The first
launch is therefore not comparable with the rest: the timing mode discards
warm-up iterations, and the profiling mode skips the first dispatch
(`ncu --launch-skip`, `rocprofv3 --kernel-iteration-range`).
