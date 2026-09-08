# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Emit a driver that measures a kernel instead of checking it.

Structurally a sibling of :mod:`driver_emit` and deliberately not a mode of it:
that one exists to answer "is this right", reads its inputs from files and
writes its outputs to files, and every one of those decisions is wrong for a
measurement. What the two share is the part that must not be derived twice --
:func:`driver_emit.collect_operands` for which buffers exist and in which
order, and :func:`driver_emit.launcher_call_expr` for how the launcher wants
them spelled.

## One binary, two modes

`time` runs warm-up launches, then `iters` launches, and prints one JSON object.
`profile` runs warm-up launches and then exactly `iters` more, prints nothing,
and lets the vendor tool do the measuring. The same binary in both cases, so
the thing profiled is the thing timed -- and `rocprof-compute`, which re-runs
the whole application once per counter pass, gets an application whose runtime
is one argument away from short.

`list` prints the workloads the binary holds and the symbol each generated,
which is what a manifest is checked against.

## What the warm-up is for

The launcher caches its grid size in a function-local `static` and performs the
occupancy query and `cudaFuncSetAttribute` on the first call only. The first
launch is a different measurement from the rest, and at the batch sizes worth
running it is also the slowest by a wide margin. Discarding it is not hygiene,
it is the difference between measuring a kernel and measuring a one-off setup.

## Two clocks, because they answer different things

Wall time over `iters` back-to-back launches on one stream, divided by `iters`,
includes the launch overhead -- which is what SeisSol pays, launching thousands
of small kernels per timestep. Event time around a single launch excludes the
queueing and is what a roofline wants. Neither is "the" time and both are
reported.

SYCL gets wall time only. The launcher submits internally and does not hand its
event back, so there is nothing to query; reported as absent rather than as a
zero, so a caller can tell "no device clock here" from "took no time".

## Inputs

Filled on the host with a deterministic pattern in [0.5, 1.5) and copied once,
outside the timed region. The values do not matter for correctness -- nothing
is compared -- but they matter for speed: zeros and denormals are not
representative on any of these machines, and a buffer left uninitialised can
contain both.
"""

from __future__ import annotations

import re
from typing import List, Sequence

from tensorforge.common.basic_types import FlagMode

from .driver_emit import DriverOperand, collect_operands, launcher_call_expr

#: Bytes per element, keyed by the C++ spelling `driver_emit` produces.
_ELEM_BYTES = {"__half": 2, "float": 4, "double": 8, "__float128": 16}


def slug(name: str) -> str:
    """A C++ identifier fragment for a workload name.

    Workload names come from case modules and from captured kernel symbols, so
    they are not identifiers. Sanitised rather than hashed, because the point of
    the name in the source is that someone reading a compiler error can tell
    which workload it belongs to.
    """
    out = re.sub(r"[^0-9a-zA-Z_]", "_", name)
    return out if out[:1].isalpha() or out[:1] == "_" else f"w_{out}"


# ----------------------------------------------------------------------
# Per-workload translation unit
# ----------------------------------------------------------------------

_TU_PREAMBLE = {
    "cuda": '#include <cuda_runtime.h>\n',
    "hip": '#include <hip/hip_runtime.h>\n',
    "sycl": '#include <sycl/sycl.hpp>\n',
}

_TU_MACROS_CUDAHIP = r"""
#define DEV_MALLOC(p, n)       (void)%(p)sMalloc(&(p), (n))
#define DEV_FREE(p)            (void)%(p)sFree((p))
#define DEV_MEMCPY_H2D(d,h,n)  (void)%(p)sMemcpy((d),(h),(n),%(p)sMemcpyHostToDevice)
"""


def _macros(backend: str) -> str:
    if backend == "cuda":
        return (
            "#define DEV_MALLOC(p, n)       (void)cudaMalloc(&(p), (n))\n"
            "#define DEV_FREE(p)            (void)cudaFree((p))\n"
            "#define DEV_MEMCPY_H2D(d,h,n)  (void)cudaMemcpy((d),(h),(n),cudaMemcpyHostToDevice)\n"
            "#define DEV_STREAM_PTR(s)      ((void*)(s))\n")
    if backend == "hip":
        return (
            "#define DEV_MALLOC(p, n)       (void)hipMalloc(&(p), (n))\n"
            "#define DEV_FREE(p)            (void)hipFree((p))\n"
            "#define DEV_MEMCPY_H2D(d,h,n)  (void)hipMemcpy((d),(h),(n),hipMemcpyHostToDevice)\n"
            "#define DEV_STREAM_PTR(s)      ((void*)(s))\n")
    # SYCL: allocations need the queue, which the driver hands over before any
    # setup call. A TU-local pointer keeps the macros nullary in the queue the
    # way the CUDA/HIP ones are.
    return (
        "extern sycl::queue* tfb_queue;\n"
        "#define DEV_MALLOC(p, n) do { (p) = static_cast<decltype(p)>("
        "sycl::malloc_device((n), *tfb_queue)); } while(0)\n"
        "#define DEV_FREE(p)            sycl::free((p), *tfb_queue)\n"
        "#define DEV_MEMCPY_H2D(d,h,n)  tfb_queue->memcpy((d),(h),(n)).wait()\n"
        "#define DEV_STREAM_PTR(s)      ((void*)(s))\n")


def _fill(ctype: str, host: str, count: str) -> str:
    """Deterministic non-degenerate values, cast to the operand's own type.

    Knuth's multiplicative hash over the index rather than an RNG: the same
    buffer contents on every run and on every host, so a timing difference
    between two runs is not a difference in what was in the arrays.
    """
    return (f"        {{ {ctype}* q = ({ctype}*){host};\n"
            f"          for (size_t i = 0; i < {count}; ++i)\n"
            f"            q[i] = static_cast<{ctype}>("
            f"0.5 + (double)((i * 2654435761u) & 1023u) / 1024.0); }}")


def emit_workload_tu(generator, backend: str, name: str,
                     includes_src: str) -> str:
    """One translation unit: the kernel, its launcher, and three entry points.

    The entry points are uniform (`setup`, `launch`, `teardown`) so the driver
    can hold a table of function pointers and never see a launcher signature.
    That is what lets one binary carry every workload of a configuration: the
    driver TU is identical for all of them and does not have to be regenerated
    when a workload's operands change.
    """
    ops = collect_operands(generator)
    for op in ops:
        if op.is_scalar:
            continue
        if op.addressing not in ("strided", "none"):
            raise NotImplementedError(
                f"operand {op.kernel_name} uses {op.addressing!r} addressing; "
                f"the measurement driver handles 'strided', 'none' and "
                f"'scalar'. Pointer-based batches need a per-element pointer "
                f"table, which is a host-side arrangement a timing run would "
                f"be measuring as much as the kernel")

    tag = slug(name)
    lang = "sycl" if backend in ("oneapi", "acpp", "esimd", "sycl") else backend

    decls, allocs, fills, copies, frees = [], [], [], [], []
    for op in ops:
        if op.is_scalar:
            continue
        elem = _ELEM_BYTES[op.ctype]
        # `storage_volume` and not `volume`: the batch stride the kernel uses
        # is `Tensor.storage_volume()` (ptr_manip.py), which is the compressed
        # count for a sparse tensor. Allocating the dense volume would be
        # harmless; allocating less than the stride would not, and the two
        # names being one letter apart is reason enough to say which is meant.
        per = op.storage_volume or op.volume
        span = (f"(size_t){per}u * {elem}" if op.addressing == "none"
                else f"(size_t){per}u * batch * {elem}")
        decls.append(f"static {op.ctype}* d_{op.kernel_name} = nullptr;")
        allocs.append(f"    DEV_MALLOC(d_{op.kernel_name}, {span});")
        fills.append(
            f"    {{ void* h = std::malloc({span});\n"
            f"      if (!h) {{ std::fprintf(stderr, \"host alloc\\n\"); "
            f"std::exit(2); }}\n"
            + _fill(op.ctype, "h", f"({span}) / {elem}") + "\n"
            f"      DEV_MEMCPY_H2D(d_{op.kernel_name}, h, {span});\n"
            f"      std::free(h); }}")
        frees.append(f"    DEV_FREE(d_{op.kernel_name});"
                     f" d_{op.kernel_name} = nullptr;")

    if generator.flag_mode() is FlagMode.REQUIRED:
        decls.append("static unsigned* d_flags = nullptr;")
        allocs.append("    DEV_MALLOC(d_flags, batch * sizeof(unsigned));")
        fills.append(
            "    { unsigned* h = (unsigned*)std::malloc(batch * sizeof(unsigned));\n"
            "      if (!h) { std::fprintf(stderr, \"host alloc\\n\"); std::exit(2); }\n"
            "      for (size_t i = 0; i < batch; ++i) h[i] = 1u;\n"
            "      DEV_MEMCPY_H2D(d_flags, h, batch * sizeof(unsigned));\n"
            "      std::free(h); }")
        frees.append("    DEV_FREE(d_flags); d_flags = nullptr;")

    call = launcher_call_expr(generator, ops, batch="batch",
                              stream="DEV_STREAM_PTR(stream)", flags="d_flags")

    body = f'''
namespace tfb_{tag} {{
{chr(10).join(decls)}
}}

extern "C" void tfb_setup_{tag}(size_t batch) {{
  using namespace tfb_{tag};
{chr(10).join(allocs)}
{chr(10).join(fills)}
}}

extern "C" void tfb_launch_{tag}(void* stream, size_t batch) {{
  using namespace tfb_{tag};
  {call};
}}

extern "C" void tfb_teardown_{tag}() {{
  using namespace tfb_{tag};
{chr(10).join(frees)}
}}

extern "C" const char* tfb_symbol_{tag}() {{
  return "kernel_{generator.get_base_name()}";
}}
'''

    return (
        "// generated by tests/harness/driver_bench.py -- do not edit\n"
        "#include <cstdio>\n#include <cstdlib>\n"
        + _TU_PREAMBLE[lang]
        + _macros(lang)
        + "\n" + includes_src + "\n"
        + generator.get_header() + "\n"
        + generator.get_kernel() + "\n"
        + generator.get_launcher() + "\n"
        + body)


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------

_DRIVER_PREAMBLE_CUDA = r"""
#include <cuda_runtime.h>
#define DEV_SET_DEVICE(i)      (void)cudaSetDevice((i))
#define DEV_STREAM_T           cudaStream_t
#define DEV_STREAM_CREATE(s)   (void)cudaStreamCreate(&(s))
#define DEV_STREAM_DESTROY(s)  (void)cudaStreamDestroy((s))
#define DEV_STREAM_SYNC(s)     (void)cudaStreamSynchronize((s))
#define DEV_STREAM_PTR(s)      ((void*)(s))
#define DEV_HAS_EVENTS         1
#define DEV_EVENT_T            cudaEvent_t
#define DEV_EVENT_CREATE(e)    (void)cudaEventCreate(&(e))
#define DEV_EVENT_DESTROY(e)   (void)cudaEventDestroy((e))
#define DEV_EVENT_RECORD(e,s)  (void)cudaEventRecord((e),(s))
#define DEV_EVENT_SYNC(e)      (void)cudaEventSynchronize((e))
#define DEV_EVENT_MS(o,a,b)    (void)cudaEventElapsedTime(&(o),(a),(b))
"""

_DRIVER_PREAMBLE_HIP = r"""
#include <hip/hip_runtime.h>
#define DEV_SET_DEVICE(i)      (void)hipSetDevice((i))
#define DEV_STREAM_T           hipStream_t
#define DEV_STREAM_CREATE(s)   (void)hipStreamCreate(&(s))
#define DEV_STREAM_DESTROY(s)  (void)hipStreamDestroy((s))
#define DEV_STREAM_SYNC(s)     (void)hipStreamSynchronize((s))
#define DEV_STREAM_PTR(s)      ((void*)(s))
#define DEV_HAS_EVENTS         1
#define DEV_EVENT_T            hipEvent_t
#define DEV_EVENT_CREATE(e)    (void)hipEventCreate(&(e))
#define DEV_EVENT_DESTROY(e)   (void)hipEventDestroy((e))
#define DEV_EVENT_RECORD(e,s)  (void)hipEventRecord((e),(s))
#define DEV_EVENT_SYNC(e)      (void)hipEventSynchronize((e))
#define DEV_EVENT_MS(o,a,b)    (void)hipEventElapsedTime(&(o),(a),(b))
"""

# The queue is in-order and owned here; the workload TUs allocate through it.
# No event macros: a SYCL launcher submits internally and returns void, so
# there is no event to time. Absence is reported as absence.
_DRIVER_PREAMBLE_SYCL = r"""
#include <sycl/sycl.hpp>
sycl::queue* tfb_queue = nullptr;
#define DEV_SET_DEVICE(i)      (void)(i)
#define DEV_STREAM_T           sycl::queue*
#define DEV_STREAM_CREATE(s)   do { \
    (s) = new sycl::queue(sycl::default_selector_v, \
                          sycl::property::queue::in_order()); \
    tfb_queue = (s); } while(0)
#define DEV_STREAM_DESTROY(s)  do { delete (s); tfb_queue = nullptr; } while(0)
#define DEV_STREAM_SYNC(s)     (s)->wait()
#define DEV_STREAM_PTR(s)      ((void*)(s))
#define DEV_HAS_EVENTS         0
"""

_DRIVER_BODY = r"""
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

struct TfbWorkload {
  const char* name;
  void (*setup)(size_t);
  void (*launch)(void*, size_t);
  void (*teardown)();
  const char* (*symbol)();
};

%(decls)s

static const TfbWorkload tfb_table[] = {
%(table)s
};
static const size_t tfb_count = sizeof(tfb_table) / sizeof(tfb_table[0]);

static void usage() {
  std::fprintf(stderr,
      "usage: driver <time|profile|list> [workload|all] [batch] "
      "[iters] [warmup] [device]\n");
  std::exit(2);
}

static double percentile(std::vector<double>& xs, double q) {
  if (xs.empty()) return 0.0;
  std::sort(xs.begin(), xs.end());
  size_t i = (size_t)(q * (double)(xs.size() - 1) + 0.5);
  return xs[i];
}

static void run_one(const TfbWorkload& w, bool timing, size_t batch,
                    size_t iters, size_t warmup, DEV_STREAM_T stream) {
  w.setup(batch);

  for (size_t i = 0; i < warmup; ++i) w.launch(DEV_STREAM_PTR(stream), batch);
  DEV_STREAM_SYNC(stream);

  if (!timing) {
    /* The profiler is the clock. Exactly `iters` dispatches follow the
       warm-up, so a kernel-iteration filter has a fixed range to name. */
    for (size_t i = 0; i < iters; ++i) w.launch(DEV_STREAM_PTR(stream), batch);
    DEV_STREAM_SYNC(stream);
    w.teardown();
    return;
  }

  /* Wall: back to back on one stream, synchronised once at the end, so the
     launch overhead is inside the measurement the way it is in a solve. */
  auto t0 = std::chrono::steady_clock::now();
  for (size_t i = 0; i < iters; ++i) w.launch(DEV_STREAM_PTR(stream), batch);
  DEV_STREAM_SYNC(stream);
  auto t1 = std::chrono::steady_clock::now();
  double wall_ns =
      (double)std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0)
          .count() / (double)iters;

  std::vector<double> ev;
#if DEV_HAS_EVENTS
  DEV_EVENT_T a, b;
  DEV_EVENT_CREATE(a);
  DEV_EVENT_CREATE(b);
  for (size_t i = 0; i < iters; ++i) {
    DEV_EVENT_RECORD(a, stream);
    w.launch(DEV_STREAM_PTR(stream), batch);
    DEV_EVENT_RECORD(b, stream);
    DEV_EVENT_SYNC(b);
    float ms = 0.0f;
    DEV_EVENT_MS(ms, a, b);
    ev.push_back((double)ms * 1e6);
  }
  DEV_EVENT_DESTROY(a);
  DEV_EVENT_DESTROY(b);
#endif

  std::printf("{\"workload\": \"%%s\", \"symbol\": \"%%s\", \"batch\": %%zu, "
              "\"iters\": %%zu, \"warmup\": %%zu, \"wall_ns\": %%.3f",
              w.name, w.symbol(), batch, iters, warmup, wall_ns);
  if (ev.empty()) {
    std::printf(", \"event_ns\": null");
  } else {
    double sum = 0.0;
    for (double x : ev) sum += x;
    std::printf(", \"event_ns\": %%.3f, \"event_ns_min\": %%.3f, "
                "\"event_ns_p50\": %%.3f, \"event_ns_p90\": %%.3f",
                sum / (double)ev.size(), percentile(ev, 0.0),
                percentile(ev, 0.5), percentile(ev, 0.9));
  }
  std::printf("}\n");
  std::fflush(stdout);

  w.teardown();
}

int main(int argc, char** argv) {
  if (argc < 2) usage();
  const std::string mode = argv[1];

  if (mode == "list") {
    for (size_t i = 0; i < tfb_count; ++i)
      std::printf("{\"workload\": \"%%s\", \"symbol\": \"%%s\"}\n",
                  tfb_table[i].name, tfb_table[i].symbol());
    return 0;
  }
  if (mode != "time" && mode != "profile") usage();
  if (argc < 3) usage();

  const std::string which = argv[2];
  const size_t batch  = (argc >= 4) ? (size_t)std::atoll(argv[3]) : 1024u;
  const size_t iters  = (argc >= 5) ? (size_t)std::atoll(argv[4])
                                    : (mode == "time" ? 200u : 1u);
  const size_t warmup = (argc >= 6) ? (size_t)std::atoll(argv[5]) : 10u;
  const int device    = (argc >= 7) ? std::atoi(argv[6]) : 0;

  DEV_SET_DEVICE(device);
  DEV_STREAM_T stream;
  DEV_STREAM_CREATE(stream);

  bool ran = false;
  for (size_t i = 0; i < tfb_count; ++i) {
    if (which != "all" && which != tfb_table[i].name) continue;
    run_one(tfb_table[i], mode == "time", batch, iters, warmup, stream);
    ran = true;
  }

  DEV_STREAM_DESTROY(stream);
  if (!ran) {
    std::fprintf(stderr, "no workload named %%s in this binary\n",
                 which.c_str());
    return 3;
  }
  return 0;
}
"""


def emit_driver(names: Sequence[str], backend: str) -> str:
    """The driver TU for one binary, given the workloads it will be linked with.

    Depends on the names alone. A workload whose object failed to compile is
    left out of `names` and the binary links without it -- which is the whole
    reason the build drops rather than raises: a corpus containing cases the
    toolchain refuses is still worth the numbers for the rest.
    """
    lang = "sycl" if backend in ("oneapi", "acpp", "esimd", "sycl") else backend
    preamble = {"cuda": _DRIVER_PREAMBLE_CUDA,
                "hip": _DRIVER_PREAMBLE_HIP,
                "sycl": _DRIVER_PREAMBLE_SYCL}[lang]

    decls, table = [], []
    for name in names:
        tag = slug(name)
        decls.append(
            f'extern "C" void tfb_setup_{tag}(size_t);\n'
            f'extern "C" void tfb_launch_{tag}(void*, size_t);\n'
            f'extern "C" void tfb_teardown_{tag}();\n'
            f'extern "C" const char* tfb_symbol_{tag}();')
        escaped = name.replace('\\', '\\\\').replace('"', '\\"')
        table.append(f'  {{"{escaped}", tfb_setup_{tag}, tfb_launch_{tag}, '
                     f'tfb_teardown_{tag}, tfb_symbol_{tag}}},')

    return preamble + (_DRIVER_BODY % {
        "decls": "\n".join(decls),
        "table": "\n".join(table),
    })
