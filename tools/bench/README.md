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
