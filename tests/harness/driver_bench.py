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
is compared against a reference -- but they matter for speed: zeros and
denormals are not representative on any of these machines, and a buffer left
uninitialized can contain both.

A batch-constant operand whose numbers the description gives holds those
numbers instead (`host_values`).  SeisSol's operators are those numbers, so
they are as representative as anything, and a generator that builds them into
its code computes with them whatever the buffer holds: only with them in the
buffer too does a kernel that reads it compute the same thing, which is what
`check` compares two builds by.

## A sequence of launches

A kernel cut into launches (`emit_sequence_tu`) is one workload whose
`launch` issues its launches one after the other on one stream, the
temporaries passed between them in global memory: the wall clock is then the
sequence's with every launch's overhead in it, and the device clock spans the
first launch to the last.  `tfb_launches_<tag>` says how many there are.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from tensorforge.common.basic_types import Datatype, FlagMode

from . import layout
from .driver_emit import collect_operands, launcher_call_expr

#: Bytes per element, keyed by the C++ spelling `driver_emit` produces.
_ELEM_BYTES = {"__half": 2, "float": 4, "double": 8, "__float128": 16}


def slug(name: str) -> str:
    """A C++ identifier fragment for a workload name.

    Workload names come from case modules and from captured kernel symbols, so
    they are not identifiers. Sanitized rather than hashed, because the point of
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
            "#define DEV_MEMCPY_D2H(h,d,n)  (void)cudaMemcpy((h),(d),(n),cudaMemcpyDeviceToHost)\n"
            "#define DEV_STREAM_PTR(s)      ((void*)(s))\n")
    if backend == "hip":
        return (
            "#define DEV_MALLOC(p, n)       (void)hipMalloc(&(p), (n))\n"
            "#define DEV_FREE(p)            (void)hipFree((p))\n"
            "#define DEV_MEMCPY_H2D(d,h,n)  (void)hipMemcpy((d),(h),(n),hipMemcpyHostToDevice)\n"
            "#define DEV_MEMCPY_D2H(h,d,n)  (void)hipMemcpy((h),(d),(n),hipMemcpyDeviceToHost)\n"
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
        "#define DEV_MEMCPY_D2H(h,d,n)  tfb_queue->memcpy((h),(d),(n)).wait()\n"
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


@dataclass(frozen=True)
class Region:
    """Where the checksum reads an output: the box of cells the kernel
    writes, in the storage the export lays the tensor out in -- column-major
    over `sizes`, from `starts`.

    The box and not the buffer, because generators agree on the box and not
    on the rest: gemmforge writes the rows an aligned count pads the box to,
    and zeroes a whole destination before a copy, where TensorForge leaves
    what it does not compute alone -- and a checksum over the buffer would
    compare those, which nobody reads, instead of the numbers."""
    alias: str
    starts: Tuple[int, ...]
    sizes: Tuple[int, ...]
    lo: Tuple[int, ...]
    hi: Tuple[int, ...]


@dataclass
class Buffer:
    """A device buffer the entry points own: one tensor for the whole batch,
    or one block for a batch-constant one, and how it is filled, passed and
    summed.

    `stem` names it in the source -- `d_<stem>`, and `d_p_<stem>` for the
    pointer table of a `pointer_based` one.  `written` makes the table's
    pointees mutable, which a launch writing through it needs.  `host` is
    what a batch-constant operand holds, in its storage order, where the
    description gave the numbers: SeisSol's operators are those numbers, and
    a kernel that takes them from the description and one that reads the
    buffer then agree.  `summed` puts it into the checksum, over `region`
    where there is one and over the whole buffer where not."""
    stem: str
    alias: str
    ctype: str
    per: int
    addressing: str
    written: bool = False
    host: Optional[Sequence[float]] = None
    summed: bool = False
    region: Optional[Region] = None


_DATATYPE_OF = {"float": Datatype.F32, "double": Datatype.F64}


def host_values(op) -> Optional[List[float]]:
    """What a batch-constant operand's buffer holds, from the description's
    numbers, in the storage the generator chose for it -- compressed by its
    `pack_index`, split into the parts `storage_parts` says -- the way the
    correctness harness prepares it (`runner.run_case`).  `None` where the
    description gives no numbers or the storage is one the harness cannot
    write, which leaves the buffer to the pattern."""
    if op.values is None or op.addressing != "none":
        return None
    dt = _DATATYPE_OF.get(op.ctype)
    if dt is None or op.storage_parts not in (1, 2):
        return None
    flat = np.asarray(op.values, dtype=layout.np_dtype(dt))
    if op.pack_index is not None:
        flat = layout.pack(flat[None, :], np.asarray(op.pack_index), dt)
    if op.storage_parts == 2:
        try:
            flat = layout.split_tf32(
                flat, dt, planar=(op.storage_volume // 2
                                  if op.storage_planar else 0))
        except ValueError:
            return None
    per = op.storage_volume or op.volume
    if flat.size != per:
        return None
    return [float(v) for v in flat]


def operand_buffer(op, stem: str, written: bool,
                   regions: Optional[Dict[str, Region]]) -> Buffer:
    """The buffer of one launcher operand of TensorForge's.

    `storage_volume` and not `volume`: the batch stride the kernel uses is
    `Tensor.storage_volume()` (ptr_manip.py), which is the compressed count
    for a sparse tensor.  Allocating the dense volume would be harmless;
    allocating less than the stride would not, and the two names being one
    letter apart is reason enough to say which is meant."""
    alias = op.alias or op.kernel_name
    return Buffer(
        stem=stem, alias=alias, ctype=op.ctype,
        per=op.storage_volume or op.volume, addressing=op.addressing,
        written=written, host=host_values(op),
        summed=(alias in regions) if regions is not None else op.is_sink,
        region=(regions or {}).get(alias))


def _check_addressing(ops) -> None:
    for op in ops:
        if op.is_scalar:
            continue
        if op.addressing not in ("strided", "none", "pointer_based"):
            raise NotImplementedError(
                f"operand {op.kernel_name} uses {op.addressing!r} addressing; "
                f"the measurement driver handles 'strided', 'none', "
                f"'pointer_based' and 'scalar'")
        if op.residence == "argument":
            raise NotImplementedError(
                f"operand {op.kernel_name} is passed by value; the "
                f"measurement driver keeps every operand on the device")


def _literal(value: float, ctype: str) -> str:
    text = repr(float(value))
    return text + "f" if ctype == "float" else text


def _span(b: Buffer) -> str:
    elem = _ELEM_BYTES[b.ctype]
    return (f"(size_t){b.per}u * {elem}" if b.addressing == "none"
            else f"(size_t){b.per}u * batch * {elem}")


def buffer_code(buffers: Sequence[Buffer]) -> Tuple[List[str], List[str],
                                                     List[str], List[str]]:
    """Declarations, allocations, fills and frees for `buffers`."""
    decls, allocs, fills, frees = [], [], [], []
    for b in buffers:
        elem = _ELEM_BYTES[b.ctype]
        span = _span(b)
        decls.append(f"static {b.ctype}* d_{b.stem} = nullptr;")
        allocs.append(f"    DEV_MALLOC(d_{b.stem}, {span});")
        if b.addressing == "pointer_based":
            # The pointers are *contiguous and in order*: element i lives at
            # `slab + i * per`, exactly where a STRIDED operand would put it.
            #
            # This is a choice and not the only one, which is why it is said
            # here rather than left to be inferred.  A pointer table is a
            # permutation of the batch, and the permutation is a first-order
            # term in what the kernel costs: identity keeps the coalescing a
            # STRIDED operand gets, and a scattered table would measure the
            # scatter as much as the code.  Identity is therefore the *upper
            # bound* on what pointer indirection can achieve, and a number
            # from it must not be read as what a mesh-ordered SeisSol batch
            # will see.  Measuring that needs a second layout and a statement
            # of which one the row came from.
            #
            # `i * per` and not `i`: `d_X` is a `ctype*`, so `+ i` advances by
            # one scalar where the kernel indexes by one batch element.
            ptr_bytes = "batch * sizeof(void*)"
            # `const` unless a launch writes through it.  `float**` does not
            # convert to `const float**` -- the language forbids it, because
            # through the second one could store a `const float*` into the
            # first's pointee -- so the table has the constness the writing
            # launcher's parameter has, and a launch that only reads a table
            # another one writes through is handed it cast.
            ptr_elem = b.ctype if b.written else f"const {b.ctype}"
            decls.append(f"static {ptr_elem}** d_p_{b.stem} = nullptr;")
            allocs.append(f"    DEV_MALLOC(d_p_{b.stem}, {ptr_bytes});")
            fills.append(
                f"    {{ void** h = (void**)std::malloc({ptr_bytes});\n"
                f"      if (!h) {{ std::fprintf(stderr, \"host alloc\\n\"); "
                f"std::exit(2); }}\n"
                f"      for (size_t i = 0; i < batch; ++i)\n"
                f"        h[i] = d_{b.stem} + i * (size_t){b.per}u;\n"
                f"      DEV_MEMCPY_H2D(d_p_{b.stem}, h, {ptr_bytes});\n"
                f"      std::free(h); }}")
            frees.append(f"    DEV_FREE(d_p_{b.stem}); d_p_{b.stem} = nullptr;")
        if b.host is not None:
            values = ", ".join(_literal(v, b.ctype) for v in b.host)
            decls.append(f"static const {b.ctype} h_{b.stem}[] = {{{values}}};")
            fills.append(f"    DEV_MEMCPY_H2D(d_{b.stem}, h_{b.stem}, {span});")
        else:
            fills.append(
                f"    {{ void* h = std::malloc({span});\n"
                f"      if (!h) {{ std::fprintf(stderr, \"host alloc\\n\"); "
                f"std::exit(2); }}\n"
                + _fill(b.ctype, "h", f"({span}) / {elem}") + "\n"
                f"      DEV_MEMCPY_H2D(d_{b.stem}, h, {span});\n"
                f"      std::free(h); }}")
        frees.append(f"    DEV_FREE(d_{b.stem}); d_{b.stem} = nullptr;")
    return decls, allocs, fills, frees


def flags_code() -> Tuple[List[str], List[str], List[str], List[str]]:
    """The batch flags of a kernel whose flag mode requires them: every
    element on."""
    return (["static unsigned* d_flags = nullptr;"],
            ["    DEV_MALLOC(d_flags, batch * sizeof(unsigned));"],
            ["    { unsigned* h = (unsigned*)std::malloc("
             "batch * sizeof(unsigned));\n"
             "      if (!h) { std::fprintf(stderr, \"host alloc\\n\"); "
             "std::exit(2); }\n"
             "      for (size_t i = 0; i < batch; ++i) h[i] = 1u;\n"
             "      DEV_MEMCPY_H2D(d_flags, h, batch * sizeof(unsigned));\n"
             "      std::free(h); }"],
            ["    DEV_FREE(d_flags); d_flags = nullptr;"])


def _dump_line(stem: str, span: str) -> str:
    """Where `$TFB_DUMP` names a directory, the summed buffer's bytes as the
    device left them, `<stem>.bin` in it: what a checksum that differs
    between two builds is made of, element by element."""
    return (f'    if (const char* dir = std::getenv("TFB_DUMP")) {{ '
            f'std::string path = std::string(dir) + "/{stem}.bin"; '
            f'if (FILE* f = std::fopen(path.c_str(), "wb")) {{ '
            f'std::fwrite(h, 1, {span}, f); std::fclose(f); }} }}')


def _region_sum(b: Buffer, span: str) -> str:
    """The checksum over `b.region`: every element's box, each cell weighted
    by its place in the buffer, as the whole buffer's cells are."""
    r = b.region
    strides = [1]
    for size in r.sizes[:-1]:
        strides.append(strides[-1] * size)
    count = "batch" if b.addressing != "none" else "1"
    lines = [f"  {{ void* h = std::malloc({span});",
             "    if (!h) { std::fprintf(stderr, \"host alloc\\n\"); "
             "std::exit(2); }",
             f"    DEV_MEMCPY_D2H(h, d_{b.stem}, {span});",
             _dump_line(b.stem, span),
             f"    const {b.ctype}* p = (const {b.ctype}*)h;",
             f"    for (size_t e = 0; e < {count}; ++e)"]
    indent = "    "
    for d in reversed(range(len(r.lo))):
        indent += "  "
        lines.append(f"{indent}for (size_t x{d} = {r.lo[d]}u; "
                     f"x{d} < {r.hi[d]}u; ++x{d})")
    offset = " + ".join(f"(x{d} - {r.starts[d]}u) * {strides[d]}u"
                        for d in range(len(r.lo))) or "0"
    indent += "  "
    lines.append(f"{indent}{{ const size_t i = e * (size_t){b.per}u + "
                 f"{offset};")
    lines.append(f"{indent}  const double w = (double)((i % 1021u) + 1u);")
    lines.append(f"{indent}  acc += (double)p[i] * w; "
                 f"mag += std::fabs((double)p[i]) * w; }}")
    lines.append("    std::free(h); }")
    return "\n".join(lines)


def checksum_code(buffers: Sequence[Buffer]) -> List[str]:
    """The checksum's statements over every summed buffer.  Where the
    outputs are named (`Region`), in the order of their names, so that two
    translation units holding the same outputs under other stems add them
    up in one order."""
    out = []
    summed = [b for b in buffers if b.summed]
    if any(b.region is not None for b in buffers):
        summed.sort(key=lambda b: b.alias)
    for b in summed:
        elem = _ELEM_BYTES[b.ctype]
        span = _span(b)
        if b.region is not None:
            out.append(_region_sum(b, span))
            continue
        # Weighted, so that a kernel writing the right values in the wrong
        # places is not the same number as one writing them in the right
        # ones -- which a plain sum would make it.
        out.append(
            f"  {{ const size_t n = ({span}) / {elem};\n"
            f"    void* h = std::malloc({span});\n"
            f"    if (!h) {{ std::fprintf(stderr, \"host alloc\\n\"); "
            f"std::exit(2); }}\n"
            f"    DEV_MEMCPY_D2H(h, d_{b.stem}, {span});\n"
            f"{_dump_line(b.stem, span)}\n"
            f"    const {b.ctype}* p = (const {b.ctype}*)h;\n"
            f"    for (size_t i = 0; i < n; ++i) {{\n"
            f"      const double w = (double)((i % 1021u) + 1u);\n"
            f"      acc += (double)p[i] * w; "
            f"mag += std::fabs((double)p[i]) * w; }}\n"
            f"    std::free(h); }}")
    return out


def entry_points(tag: str, decls: Sequence[str], allocs: Sequence[str],
                 fills: Sequence[str], frees: Sequence[str],
                 launch: Sequence[str], sums: Sequence[str], symbol: str,
                 launches: int, code: str = "") -> str:
    """The entry points every workload's translation unit defines, with the
    buffers in `namespace tfb_<tag>`, and beside them `code` -- what a
    sequence of launches puts there of its own."""
    nl = "\n"
    return (
        f"\nnamespace tfb_{tag} {{\n{nl.join(decls)}\n{code}\n}}\n"
        f"\nextern \"C\" void tfb_setup_{tag}(size_t batch) {{\n"
        f"  using namespace tfb_{tag};\n{nl.join(allocs)}\n{nl.join(fills)}\n}}\n"
        f"\nextern \"C\" void tfb_launch_{tag}(void* stream, size_t batch) {{\n"
        f"  using namespace tfb_{tag};\n{nl.join(launch)}\n}}\n"
        f"\nextern \"C\" void tfb_teardown_{tag}() {{\n"
        f"  using namespace tfb_{tag};\n{nl.join(frees)}\n}}\n"
        f"\nextern \"C\" double tfb_checksum_{tag}(size_t batch, "
        f"double* magnitude) {{\n"
        f"  using namespace tfb_{tag};\n  double acc = 0.0, mag = 0.0;\n"
        f"{nl.join(sums) if sums else '  (void)batch;'}\n"
        f"  if (magnitude) *magnitude = mag;\n  return acc;\n}}\n"
        f"\nextern \"C\" const char* tfb_symbol_{tag}() {{\n"
        f"  return \"{symbol}\";\n}}\n"
        f"\nextern \"C\" unsigned tfb_launches_{tag}() {{\n"
        f"  return {launches}u;\n}}\n")


def tu_head(backend: str, includes_src: str) -> str:
    """What every workload's translation unit starts with."""
    lang = "sycl" if backend in ("oneapi", "acpp", "esimd", "sycl") else backend
    return ("// generated by tests/harness/driver_bench.py -- do not edit\n"
            "#include <cmath>\n#include <cstdio>\n#include <cstdlib>\n"
            "#include <string>\n"
            + _TU_PREAMBLE[lang]
            + _macros(lang)
            + "\n" + includes_src + "\n")


def emit_workload_tu(generator, backend: str, name: str,
                     includes_src: str,
                     regions: Optional[Dict[str, Region]] = None) -> str:
    """One translation unit: the kernel, its launcher, and the entry points.

    The entry points are uniform (`setup`, `launch`, `teardown`, ...) so the
    driver can hold a table of function pointers and never see a launcher
    signature.  That is what lets one binary carry every workload of a
    configuration: the driver TU is identical for all of them and does not
    have to be regenerated when a workload's operands change.

    `regions` names the kernel's outputs and the boxes they are written over
    (`Region`); the checksum then reads those and nothing else.  Without, it
    reads every buffer the kernel writes, whole.
    """
    ops = collect_operands(generator)
    _check_addressing(ops)
    tag = slug(name)
    buffers = [operand_buffer(op, op.kernel_name, op.is_sink, regions)
               for op in ops if not op.is_scalar]
    decls, allocs, fills, frees = buffer_code(buffers)
    if generator.flag_mode() is FlagMode.REQUIRED:
        for part, extra in zip((decls, allocs, fills, frees), flags_code()):
            part.extend(extra)

    call = launcher_call_expr(generator, ops, batch="batch",
                              stream="DEV_STREAM_PTR(stream)", flags="d_flags")
    body = entry_points(tag, decls, allocs, fills, frees, [f"  {call};"],
                        checksum_code(buffers),
                        f"kernel_{generator.get_base_name()}", 1)
    return (tu_head(backend, includes_src)
            + generator.get_header() + "\n"
            + generator.get_kernel() + "\n"
            + generator.get_launcher() + "\n"
            + body)


#: What TensorForge's header and launcher open with: the launch types, under
#: a guard.  A sequence defines them once, at namespace scope, ahead of the
#: namespace its kernels go into -- defined inside it, they would make
#: `tfb_<tag>::tensorforge` the namespace every `tensorforge::` in the kernels
#: finds first, and the runtime's would be out of reach.
_LAUNCH_TYPES = re.compile(r"#ifndef TENSORFORGE_LAUNCH_TYPES\n.*?\n#endif\n",
                           re.S)


def emit_sequence_tu(generators: Sequence, backend: str, name: str,
                     includes_src: str,
                     regions: Optional[Dict[str, Region]] = None
                     ) -> Tuple[str, str]:
    """One translation unit for a kernel cut into launches: the kernel and
    launcher of every launch, and the entry points of `emit_workload_tu`,
    whose `launch` issues the launches one after the other on one stream.

    A tensor is one buffer for every launch that names it, by its name in
    the description: a temporary passed from one launch to the next is
    written by the first and read by the second, as SeisSol's scratch memory
    is.  Where two launches store a batch-constant operand differently -- a
    generator may prepare one for the instruction that reads it -- each gets
    its own; a tensor of the batch stored two ways is refused, since no one
    buffer could be both.

    The kernels go into `namespace tfb_<tag>`.  Two launches of a sequence
    can be the same kernel -- the same product of two operators on two inputs
    -- which is then defined once; two sequences can share one too, and the
    namespaces keep them apart in one binary.

    Returns the source and the symbol it names the sequence by, a digest of
    its kernels and of the order they are launched in.
    """
    tag = slug(name)
    shapes: Dict[str, Tuple] = {}
    uses: Dict[str, List] = {}
    pieces = []
    for n, gen in enumerate(generators):
        ops = collect_operands(gen)
        _check_addressing(ops)
        keys = {}
        for op in ops:
            if op.is_scalar:
                continue
            alias = op.alias or op.kernel_name
            shape = (op.addressing, op.ctype, op.storage_volume or op.volume,
                     op.storage_parts, op.pack_index)
            key = alias
            if shapes.setdefault(alias, shape) != shape:
                if op.addressing != "none":
                    raise ValueError(
                        f"{name}: the launches store {alias} two ways, "
                        f"{shapes[alias][:3]} and {shape[:3]}")
                key = f"{alias}#{n}"
            keys[op.kernel_name] = key
            uses.setdefault(key, []).append(op)
        pieces.append((gen, ops, keys))
    buffers: Dict[str, Buffer] = {}
    for i, (key, ops) in enumerate(uses.items()):
        buffers[key] = operand_buffer(ops[0], f"b{i}",
                                      any(op.is_sink for op in ops), regions)
        if regions is None:
            buffers[key].summed = buffers[key].written

    launches: List[str] = []
    kernels: Dict[str, object] = {}
    needs_flags = False
    for gen, ops, keys in pieces:
        needs_flags |= gen.flag_mode() is FlagMode.REQUIRED

        def argument(op, keys=keys):
            b = buffers[keys[op.kernel_name]]
            if op.addressing != "pointer_based":
                return f"d_{b.stem}"
            if b.written and not op.is_sink:
                return f"const_cast<const {b.ctype}**>(d_p_{b.stem})"
            return f"d_p_{b.stem}"

        call = launcher_call_expr(gen, ops, batch="batch",
                                  stream="DEV_STREAM_PTR(stream)",
                                  flags="d_flags", buffer=argument)
        launches.append(f"  {call};")
        kernels.setdefault(gen.get_base_name(), gen)

    first = next(iter(kernels.values()))
    found = _LAUNCH_TYPES.search(first.get_header())
    if found is None:
        raise ValueError("the generator's header does not open with the "
                         "launch types a sequence defines ahead of its "
                         "kernels")
    code = "\n".join(gen.get_header() + "\n" + gen.get_kernel() + "\n"
                     + gen.get_launcher() for gen in kernels.values())
    decls, allocs, fills, frees = buffer_code(list(buffers.values()))
    if needs_flags:
        for part, extra in zip((decls, allocs, fills, frees), flags_code()):
            part.extend(extra)
    symbol = "sequence_" + hashlib.sha256(
        "\n".join([code] + launches).encode()).hexdigest()[:16]
    body = entry_points(tag, decls, allocs, fills, frees, launches,
                        checksum_code(list(buffers.values())), symbol,
                        len(launches), code)
    return tu_head(backend, includes_src) + found.group(0) + body, symbol


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
#define DEV_EVENTS_LAST_ONLY   0
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
#define DEV_EVENTS_LAST_ONLY   0
"""

# The queue is in-order and owned here; the workload TUs allocate through it.
# A SYCL launcher submits internally and returns void, so there is no event of
# its own to time -- but on an in-order queue the command it submitted is the
# queue's last one (sycl_ext_oneapi_in_order_queue_events), and with profiling
# on, that event carries the device's start and end.  Recorded after the
# launch it is the kernel; the "before" record is the previous command and is
# not used.  A launcher that submitted more than one command would be timed
# by its last, and so would a workload of several launches -- for which the
# driver therefore reports no device clock (`DEV_EVENTS_LAST_ONLY`).  Without
# the extension, absence is reported as absence.
_DRIVER_PREAMBLE_SYCL = r"""
#include <sycl/sycl.hpp>
sycl::queue* tfb_queue = nullptr;
#define DEV_SET_DEVICE(i)      (void)(i)
#define DEV_STREAM_T           sycl::queue*
#define DEV_STREAM_CREATE(s)   do { \
    (s) = new sycl::queue(sycl::default_selector_v, \
                          sycl::property_list{sycl::property::queue::in_order(), \
                              sycl::property::queue::enable_profiling()}); \
    tfb_queue = (s); } while(0)
#define DEV_STREAM_DESTROY(s)  do { delete (s); tfb_queue = nullptr; } while(0)
#define DEV_STREAM_SYNC(s)     (s)->wait()
#define DEV_STREAM_PTR(s)      ((void*)(s))
#ifdef SYCL_EXT_ONEAPI_IN_ORDER_QUEUE_EVENTS
#include <optional>
/* The last event as an event: DPC++ 2025.0 returns one, 2025.2 an
   `std::optional` of one, empty before the queue's first command. */
inline sycl::event tfb_last_event(const sycl::event &e) { return e; }
inline sycl::event tfb_last_event(const std::optional<sycl::event> &e) {
  return e ? *e : sycl::event();
}
#define DEV_HAS_EVENTS         1
#define DEV_EVENT_T            sycl::event
#define DEV_EVENT_CREATE(e)    (void)(e)
#define DEV_EVENT_DESTROY(e)   (void)(e)
#define DEV_EVENT_RECORD(e,s)  ((e) = tfb_last_event((s)->ext_oneapi_get_last_event()))
#define DEV_EVENT_SYNC(e)      (e).wait()
#define DEV_EVENT_MS(o,a,b)    ((o) = (float)(1e-6 * (double)( \
    (b).get_profiling_info<sycl::info::event_profiling::command_end>() - \
    (b).get_profiling_info<sycl::info::event_profiling::command_start>())))
#define DEV_EVENTS_LAST_ONLY   1
#else
#define DEV_HAS_EVENTS         0
#endif
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
  double (*checksum)(size_t, double*);
  unsigned (*launches)();
};

%(decls)s

static const TfbWorkload tfb_table[] = {
%(table)s
};
static const size_t tfb_count = sizeof(tfb_table) / sizeof(tfb_table[0]);

static void usage() {
  std::fprintf(stderr,
      "usage: driver <time|profile|check|list> [workload|all] [batch] "
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

  /* Wall: back to back on one stream, synchronized once at the end, so the
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
  /* An event that stands for the last command of the queue times the last
     of several launches, which is no time of the workload's. */
  if (!(DEV_EVENTS_LAST_ONLY && w.launches() > 1)) {
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
  }
#endif

  std::printf("{\"workload\": \"%%s\", \"symbol\": \"%%s\", \"batch\": %%zu, "
              "\"iters\": %%zu, \"warmup\": %%zu, \"launches\": %%u, "
              "\"wall_ns\": %%.3f",
              w.name, w.symbol(), batch, iters, warmup, w.launches(),
              wall_ns);
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
      std::printf("{\"workload\": \"%%s\", \"symbol\": \"%%s\", "
                  "\"launches\": %%u}\n",
                  tfb_table[i].name, tfb_table[i].symbol(),
                  tfb_table[i].launches());
    return 0;
  }
  if (mode != "time" && mode != "profile" && mode != "check") usage();
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
    if (mode == "check") {
      /* One launch, then the sinks read back and reduced.  Not a reference
         comparison -- the driver has no oracle -- but enough to tell two
         builds of one workload apart, which is what an experiment on a
         patched kernel needs and what the timing modes cannot give.  The
         magnitude is the same sum over the values' absolute values: what a
         difference between two checksums is to be read against, since the
         signed sum can cancel to anything. */
      const TfbWorkload& w = tfb_table[i];
      w.setup(batch);
      w.launch(DEV_STREAM_PTR(stream), batch);
      DEV_STREAM_SYNC(stream);
      double magnitude = 0.0;
      const double checksum = w.checksum(batch, &magnitude);
      std::printf("{\"workload\": \"%%s\", \"batch\": %%zu, "
                  "\"checksum\": %%.17g, \"magnitude\": %%.17g}\n",
                  w.name, batch, checksum, magnitude);
      w.teardown();
      ran = true;
      continue;
    }
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
            f'extern "C" const char* tfb_symbol_{tag}();\n'
            f'extern "C" double tfb_checksum_{tag}(size_t, double*);\n'
            f'extern "C" unsigned tfb_launches_{tag}();')
        escaped = name.replace('\\', '\\\\').replace('"', '\\"')
        table.append(f'  {{"{escaped}", tfb_setup_{tag}, tfb_launch_{tag}, '
                     f'tfb_teardown_{tag}, tfb_symbol_{tag}, '
                     f'tfb_checksum_{tag}, tfb_launches_{tag}}},')

    return preamble + (_DRIVER_BODY % {
        "decls": "\n".join(decls),
        "table": "\n".join(table),
    })
