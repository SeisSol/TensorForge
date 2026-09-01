# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Emit a standalone C++ driver that exercises one generated launcher.

Design:

* Inputs are binary files that the driver ``fread``\\ s into host memory.
* Outputs are binary files the driver ``fwrite``\\ s after a D->H copy.
* All buffer sizes and argument order are resolved from the
  :class:`Generator`'s symbol table — we don't re-parse the emitted C++.
* Only STRIDED addressing is handled in the MVP. The driver emits a
  clear ``#error`` otherwise.

The same template covers CUDA and HIP because their runtime APIs agree
on ``*Malloc`` / ``*Memcpy`` / ``*Stream`` up to the prefix. A tiny
backend-selection block at the top picks between them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from tensorforge.backend.symbol import SymbolType
from tensorforge.common.basic_types import Addressing, DataFlowDirection, FlagMode

from .layout import ctype, volume


@dataclass
class DriverOperand:
    """One launcher operand, as seen from the host side.

    A launcher exposes two kinds of operands today:

    * **Batched tensors** — pointer + per-call extraOffset; allocated and
      copied per case.
    * **Scalar literals** — passed by value (e.g. ``alpha`` synthesized
      by ``GemmDescr`` for ``alpha != 1``); the runtime value is baked
      into the kernel call site at emit time.

    ``is_scalar=True`` means the operand carries a host-side literal in
    ``scalar_value``; the buffer fields (``shape``, ``volume``) are zero
    in that case.
    """
    kernel_name: str
    alias: str | None
    is_source: bool
    is_sink: bool
    shape: tuple
    volume: int
    ctype: str
    addressing: str
    is_scalar: bool = False
    scalar_value: float | None = None


def collect_operands(generator) -> List[DriverOperand]:
    """Walk the generator's global scope and materialize operand metadata.

    The global scope's insertion order is the same order the launcher
    takes its parameters in, so host-side args line up by index.
    """
    ops: List[DriverOperand] = []
    for sym in generator._scopes.get_global_scope().values():
        t = sym.obj
        if sym.stype == SymbolType.Scalar:
            # Scalar literal (e.g. alpha). Must have a baked-in value;
            # symbolic-runtime scalars (alpha='alpha') are out of MVP scope.
            if not getattr(t, 'has_values', lambda: False)() or t.get_values() is None:
                raise NotImplementedError(
                    f"scalar operand {sym.name!r} has no constant value; "
                    f"runtime-symbolic scalars are not yet supported"
                )
            value = float(t.get_values()[0])
            ops.append(DriverOperand(
                kernel_name=sym.name, alias=t.alias,
                is_source=True, is_sink=False,
                shape=(), volume=0,
                ctype=ctype(t.datatype), addressing="scalar",
                is_scalar=True, scalar_value=value,
            ))
            continue
        if sym.stype != SymbolType.Batch:
            # Other symbol kinds (Data, SharedMem, …) are kernel-internal.
            continue
        direction = t.direction
        is_src = direction in (DataFlowDirection.SOURCE,
                               DataFlowDirection.SOURCESINK)
        is_snk = direction in (DataFlowDirection.SINK,
                               DataFlowDirection.SOURCESINK)
        ops.append(DriverOperand(
            kernel_name=sym.name,
            alias=t.alias,
            is_source=is_src,
            is_sink=is_snk,
            # the kernel addresses the *stored* region: memory spans
            # upper - lower and address 0 is `lower`, so the host buffer is
            # bbox-shaped, not shape-shaped (see DataView.get_dim_strides)
            shape=tuple(t.get_actual_shape()),
            volume=int(t.get_actual_volume()),
            ctype=ctype(t.datatype),
            addressing=str(t.addressing),
        ))
    return ops


# ----------------------------------------------------------------------
# Template pieces
# ----------------------------------------------------------------------

_PREAMBLE_CUDA = r"""
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#define DEV_MALLOC(p, n)       cudaMalloc(&(p), (n))
#define DEV_FREE(p)            cudaFree((p))
#define DEV_MEMCPY_H2D(d,h,n)  cudaMemcpy((d),(h),(n),cudaMemcpyHostToDevice)
#define DEV_MEMCPY_D2H(h,d,n)  cudaMemcpy((h),(d),(n),cudaMemcpyDeviceToHost)
#define DEV_STREAM_T           cudaStream_t
#define DEV_STREAM_CREATE(s)   cudaStreamCreate(&(s))
#define DEV_STREAM_DESTROY(s)  cudaStreamDestroy((s))
#define DEV_STREAM_SYNC(s)     cudaStreamSynchronize((s))
#define DEV_SET_DEVICE(i)      cudaSetDevice((i))
#define DEV_STREAM_PTR(s)      ((void*)(s))
"""

_PREAMBLE_HIP = r"""
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <hip/hip_runtime.h>
#define DEV_MALLOC(p, n)       (void)hipMalloc(&(p), (n))
#define DEV_FREE(p)            (void)hipFree((p))
#define DEV_MEMCPY_H2D(d,h,n)  (void)hipMemcpy((d),(h),(n),hipMemcpyHostToDevice)
#define DEV_MEMCPY_D2H(h,d,n)  (void)hipMemcpy((h),(d),(n),hipMemcpyDeviceToHost)
#define DEV_STREAM_T           hipStream_t
#define DEV_STREAM_CREATE(s)   (void)hipStreamCreate(&(s))
#define DEV_STREAM_DESTROY(s)  (void)hipStreamDestroy((s))
#define DEV_STREAM_SYNC(s)     (void)hipStreamSynchronize((s))
#define DEV_SET_DEVICE(i)      (void)hipSetDevice((i))
#define DEV_STREAM_PTR(s)      ((void*)(s))
"""


# SYCL is structurally different from CUDA/HIP: there is no "stream", there
# is a queue; allocations go through USM (sycl::malloc_device), and the
# launcher itself takes a sycl::queue* rather than an opaque pointer. We
# wrap the differences in the same DEV_* macros to keep the main template
# uniform — the queue is the "stream", just spelled differently.
#
# The launcher TU already #includes <sycl/sycl.hpp>; the driver does too,
# so the macros can use sycl:: directly.
_PREAMBLE_SYCL = r"""
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <sycl/sycl.hpp>

// USM device allocations need the queue. We capture the active queue in
// a TU-local pointer set by DEV_STREAM_CREATE so the malloc/memcpy macros
// stay nullary in the queue argument. Single-threaded drivers only.
static sycl::queue* g_tf_queue = nullptr;

#define DEV_MALLOC(p, n) \
    do { (p) = static_cast<decltype(p)>(sycl::malloc_device((n), *g_tf_queue)); } while(0)
#define DEV_FREE(p)            sycl::free((p), *g_tf_queue)
#define DEV_MEMCPY_H2D(d,h,n)  g_tf_queue->memcpy((d),(h),(n)).wait()
#define DEV_MEMCPY_D2H(h,d,n)  g_tf_queue->memcpy((h),(d),(n)).wait()
#define DEV_STREAM_T           sycl::queue*
#define DEV_STREAM_CREATE(s)   do { \
    (s) = new sycl::queue(sycl::default_selector_v); \
    g_tf_queue = (s); \
} while(0)
#define DEV_STREAM_DESTROY(s)  do { delete (s); g_tf_queue = nullptr; } while(0)
#define DEV_STREAM_SYNC(s)     (s)->wait()
#define DEV_SET_DEVICE(i)      (void)(i)   /* SYCL picks device via selector */
#define DEV_STREAM_PTR(s)      ((void*)(s))
"""


# CUDA/HIP cast the stream to a void* before passing it to the launcher;
# the SYCL launcher already takes void* but expects it to point to a
# sycl::queue. Keep the call symmetric via DEV_STREAM_PTR in the main
# template.


_HELPERS = r"""
static void die(const char* msg) {
    std::fprintf(stderr, "driver error: %s\n", msg);
    std::exit(2);
}

static void read_bin(const char* path, void* dst, size_t nbytes) {
    FILE* f = std::fopen(path, "rb");
    if (!f) die("cannot open input");
    if (std::fread(dst, 1, nbytes, f) != nbytes) die("short input read");
    std::fclose(f);
}

static void write_bin(const char* path, const void* src, size_t nbytes) {
    FILE* f = std::fopen(path, "wb");
    if (!f) die("cannot open output");
    if (std::fwrite(src, 1, nbytes, f) != nbytes) die("short output write");
    std::fclose(f);
}
"""


_MAIN_TEMPLATE = r"""
int main(int argc, char** argv) {{
    if (argc < 3) die("usage: driver <input_dir> <output_dir> [batch]");
    const char* in_dir  = argv[1];
    const char* out_dir = argv[2];
    const size_t batch = (argc >= 4) ? (size_t)std::atoll(argv[3]) : {default_batch}u;
    const int device_index = (argc >= 5) ? std::atoi(argv[4]) : 0;
    DEV_SET_DEVICE(device_index);

    char path[1024];

{allocs_host}
{reads}
{allocs_dev}
{h2d}

    DEV_STREAM_T stream;
    DEV_STREAM_CREATE(stream);

    {launcher_call};

    DEV_STREAM_SYNC(stream);

{d2h}
{writes}

    DEV_STREAM_DESTROY(stream);
{frees}
    return 0;
}}
"""


def emit(generator, backend: str, default_batch: int) -> str:
    """Return full driver source text for one generated kernel."""
    if backend == "cuda":
        preamble = _PREAMBLE_CUDA
    elif backend == "hip":
        preamble = _PREAMBLE_HIP
    elif backend in ("oneapi", "acpp", "sycl"):
        preamble = _PREAMBLE_SYCL
    else:
        raise NotImplementedError(f"backend {backend!r} not supported in MVP")

    ops = collect_operands(generator)

    for op in ops:
        if op.is_scalar:
            continue
        if op.addressing not in ("strided", "none", "pointer_based"):
            raise NotImplementedError(
                f"operand {op.kernel_name} uses {op.addressing!r} addressing; "
                f"harness handles 'strided', 'none' and 'scalar'"
            )

    # --- per-operand blocks ----------------------------------------------
    allocs_host, reads, allocs_dev, h2d, d2h, writes, frees = [], [], [], [], [], [], []
    for op in ops:
        if op.is_scalar:
            continue       # scalars don't allocate, they're literals at the call site

        elem_bytes = {
            "__half": 2,
            "float": 4,
            "double": 8,
            "__float128": 16,
        }[op.ctype]

        # Batch-constant (Addressing.NONE) operands share one storage
        # block across all batch elements — see ptr_manip.py:67-71 where
        # the kernel-side pointer skips the ``batchId * volume`` term.
        # The driver therefore allocates a single ``volume`` worth of
        # bytes, independent of the batch size.
        if op.addressing == "none":
            total_expr = f"(size_t){op.volume}u * {elem_bytes}"
        else:
            total_expr = f"(size_t){op.volume}u * batch * {elem_bytes}"

        allocs_host.append(
            f"    void* h_{op.kernel_name} = std::malloc({total_expr});\n"
            f"    if (!h_{op.kernel_name}) die(\"host alloc\");"
        )
        if op.is_source:
            reads.append(
                f"    std::snprintf(path, sizeof(path), \"%s/in_{op.kernel_name}.bin\", in_dir);\n"
                f"    read_bin(path, h_{op.kernel_name}, {total_expr});"
            )
        allocs_dev.append(
            f"    {op.ctype}* d_{op.kernel_name} = nullptr;\n"
            f"    DEV_MALLOC(d_{op.kernel_name}, {total_expr});"
        )
        if op.addressing == "pointer_based":
            ptr_size = f"(batch * sizeof(void*))"

            allocs_host.append(
                f"    void** h_p_{op.kernel_name} = (void**)std::malloc({ptr_size});\n"
                f"    if (!h_p_{op.kernel_name}) die(\"host alloc\");"
            )

            base_typestr = f"{op.ctype}" if op.is_sink else f"const {op.ctype}"
            allocs_dev.append(
                f"    {base_typestr}** d_p_{op.kernel_name} = nullptr;\n"
                f"    DEV_MALLOC(d_p_{op.kernel_name}, {ptr_size});"
            )
            # TODO: move to maybe a section of its own?
            h2d.append(
                f"    for (size_t i = 0; i < batch; ++i) h_p_{op.kernel_name}[i] = d_{op.kernel_name} + i;"
            )
            h2d.append(
                f"    DEV_MEMCPY_H2D(d_p_{op.kernel_name}, h_p_{op.kernel_name}, {ptr_size});"
            )
        # SOURCE and SOURCESINK both need H2D (SINK is also copied so the
        # kernel observes a defined initial value when beta != 0).
        h2d.append(
            f"    DEV_MEMCPY_H2D(d_{op.kernel_name}, h_{op.kernel_name}, {total_expr});"
        )
        if op.is_sink:
            d2h.append(
                f"    DEV_MEMCPY_D2H(h_{op.kernel_name}, d_{op.kernel_name}, {total_expr});"
            )
            writes.append(
                f"    std::snprintf(path, sizeof(path), \"%s/out_{op.kernel_name}.bin\", out_dir);\n"
                f"    write_bin(path, h_{op.kernel_name}, {total_expr});"
            )
        frees.append(
            f"    DEV_FREE(d_{op.kernel_name});\n"
            f"    std::free(h_{op.kernel_name});"
        )

    # --- launcher call ---------------------------------------------------
    launcher_fn = f"launcher_{generator.get_base_name()}"
    call_args = []
    for op in ops:
        if op.is_scalar:
            # Bake the constant in. The literal needs the type suffix so
            # we don't accidentally widen/narrow at the call site.
            suffix = {
                "float": "f",
                "double": "",
                "__float128": "q",
            }.get(op.ctype) or ""

            call_args.append(f"static_cast<{op.ctype}>({op.scalar_value!r}{suffix})")
        else:
            if op.addressing == "pointer_based":
                call_args.append(f"d_p_{op.kernel_name}")
            else:
                call_args.append(f"d_{op.kernel_name}")
            # The launcher only takes an ``extraOffset`` for STRIDED
            # operands; NONE-addressed bindings drop it (see the
            # launcher signature in generator output).
            if op.addressing != "none":
                call_args.append("0u")
    # The launcher signature has one ``numElements{i}`` parameter per
    # section. With a single descr or only fence-less chains the section
    # count is 1 — the default. ``GridFenceDescr`` / ``GridBarrierDescr``
    # between descrs split sections, raising the count.
    num_sections = len(getattr(generator, "_sections", [None]))
    for _ in range(num_sections):
        call_args.append("batch")
    # Whether a ``flags{i}`` parameter sits beside each of them is the
    # kernel's flag mode; see ``FlagMode``.
    flag_mode = generator.flag_mode()
    if flag_mode is FlagMode.REQUIRED:
        # An all-ones mask enables every element, so the kernel computes
        # what the reference computes. The alternative — masking some
        # elements off — needs an oracle that knows which, and the point
        # here is the signature, not the masking semantics.
        allocs_host.append(
            "    unsigned* h_flags = (unsigned*)std::malloc(batch * sizeof(unsigned));\n"
            "    if (!h_flags) die(\"host alloc\");\n"
            "    for (size_t i = 0; i < batch; ++i) h_flags[i] = 1u;"
        )
        allocs_dev.append(
            "    unsigned* d_flags = nullptr;\n"
            "    DEV_MALLOC(d_flags, batch * sizeof(unsigned));"
        )
        h2d.append(
            "    DEV_MEMCPY_H2D(d_flags, h_flags, batch * sizeof(unsigned));"
        )
        frees.append("    DEV_FREE(d_flags);\n    std::free(h_flags);")
    for _ in range(num_sections):
        if flag_mode is FlagMode.REQUIRED:
            call_args.append("d_flags")
        elif flag_mode is FlagMode.OPTIONAL:
            call_args.append("nullptr")    # flags{i}: no skip-flags in tests
    call_args.append("DEV_STREAM_PTR(stream)")
    launcher_call = f"{launcher_fn}({', '.join(call_args)})"

    body = _MAIN_TEMPLATE.format(
        default_batch=default_batch,
        allocs_host="\n".join(allocs_host),
        reads="\n".join(reads) if reads else "    (void)in_dir;",
        allocs_dev="\n".join(allocs_dev),
        h2d="\n".join(h2d),
        launcher_call=launcher_call,
        d2h="\n".join(d2h),
        writes="\n".join(writes) if writes else "    (void)out_dir;",
        frees="\n".join(frees),
    )

    return preamble + "\n#include \"kernels.h\"\n" + _HELPERS + body
