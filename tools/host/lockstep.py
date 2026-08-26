# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Execute a generated CUDA kernel on the host, all lanes, one batch element.

`kernel_eval` interprets one thread.  Shared memory is where threads meet, so
a single thread sees whatever the others have not written.  Splitting the
per-element body at its barriers and driving every lane through one phase at a
time gives the same guarantee the hardware does, on the same `Slot`.
"""
import re

from tfpaths import add_tests_to_path

add_tests_to_path()
import kernel_eval as ke                                       # noqa: E402

THREADS = 32


def extract(path, kernel):
    lines = open(path).read().split("\n")
    starts = [(i, re.match(r"\s*kernel_(kernel_\w+)\(", l).group(1))
              for i, l in enumerate(lines)
              if re.match(r"\s*kernel_kernel_\w+\(.*\{$", l)]
    ends = [i for i, l in enumerate(lines) if l.startswith("void launcher_")]
    s = [a for a, b in starts if b == kernel][0]
    e = min(x for x in ends if x > s)
    return "\n".join(lines[s - 2:e])


def desugar_async(src):
    """`cuda::memcpy_async` is the shared-memory staging; model it as a copy.

    `kernel_eval` skips anything with `::` --- for a pipeline object that is
    right, but the transfer itself carries the values every consumer reads.
    """
    return re.sub(
        r"cuda::memcpy_async\(\s*&([^,]+?),\s*&([^,]+?),[^;]*\);",
        r"\1 = \2;", src)


def flatten_batching(src):
    """One element, no extra offset: make every global pointer point at 0."""
    src = re.sub(r"&(m\d+)\[batchId0\]\[[^\]]*\]", r"&\1[0]", src)
    src = re.sub(r"&(m\d+)\[batchId0 \* \d+ \+ 0 \+ \w+\]", r"&\1[0]", src)
    src = re.sub(r"&(m\d+)\[0 \+ \w+_extraOffset\]", r"&\1[0]", src)
    return desugar_async(src)


def _walk(node, pred, out):
    if isinstance(node, tuple):
        if pred(node):
            out.append(node)
        for x in node[1:]:
            _walk(x, pred, out)
    elif isinstance(node, list):
        for x in node:
            _walk(x, pred, out)


def split_body(src):
    """(prologue statements, phases of the per-element body)."""
    nodes = ke.parse(src[src.index("{"):])
    found = []
    _walk(nodes, lambda n: n[0] == "if" and "allowed" in str(n[1]), found)
    guard = found[0]
    body = guard[2]
    stmts = body[1] if isinstance(body, tuple) and body[0] == "block" else body

    # everything the body needs -- pipeline, shared base, glb_ pointers, the
    # loop's own induction -- lives in the blocks around it
    prologue = []

    def collect(node):
        if isinstance(node, tuple):
            if node is guard:
                return
            if node[0] == "for":
                prologue.append(("expr", f"{node[1]} = {node[2]}"))
                collect(node[6])
                return
            if node[0] == "if":
                return
            if node[0] == "block":
                for c in node[1]:
                    collect(c)
                return
            prologue.append(node)
        elif isinstance(node, list):
            for c in node:
                collect(c)

    collect(nodes)

    phases, cur = [], []
    for st in stmts:
        cur.append(st)
        if (isinstance(st, tuple) and st[0] == "expr"
                and re.match(r"^__sync", str(st[1]).strip())):
            phases.append(cur)
            cur = []
    if cur:
        phases.append(cur)
    return prologue, phases


def strides(shape):
    out, cur = [], 1
    for s in shape:
        out.append(cur)
        cur *= s
    return out


def run(src, inputs, shapes, storage=None):
    mem = ke.Slot(0)
    # Every slot the kernel can touch has to be defined, or `Slot.read`
    # fabricates one and the comparison measures that instead.  Inputs get
    # their values; everything else -- outputs it accumulates onto, and the
    # shared arena -- starts at zero, which is what the reference assumes.
    import numpy as _np
    for name, shape in shapes.items():
        if not name.startswith("m"):
            continue
        ashape, lower = (storage or {}).get(name, (shape, (0,) * len(shape)))
        n = 1
        for s_ in ashape:
            n *= s_
        arr = inputs.get(name)
        if arr is None:
            for idx in range(max(n, 1)):
                mem.write(name, idx, 0.0)
            continue
        sub = arr[tuple(slice(lo, lo + sz) for lo, sz in zip(lower, ashape))] \
            if arr.ndim else arr
        flat = _np.asarray(sub).reshape(-1, order="F")
        for idx in range(max(n, 1)):
            mem.write(name, idx, float(flat[idx]) if idx < len(flat) else 0.0)
    m = re.search(r"&totalShrMem\[(\d+) \* threadIdx\.y", src)
    arena = int(m.group(1)) * 2 if m else 0
    for idx in range(arena):
        mem.write("shr", idx, 0.0)

    base = {
        "blockIdx": type("B", (), {"x": 0, "y": 0, "z": 0})(),
        "blockDim": type("D", (), {"x": THREADS, "y": 1, "z": 1})(),
        "gridDim": type("G", (), {"x": 1, "y": 1, "z": 1})(),
        "numElements0": 1,
        "flags0": None,
        "totalShrMemPtr": ke.Ptr(mem, "shr"),
    }
    # a SCALAR-addressed tensor is passed by value, not as a pointer
    k = src.index("kernel_kernel_")
    sig = src[src.index("(", k):src.index(")", k)]
    scalars = set(re.findall(r"(?<!\*)\bfloat (m\d+)\b", sig))
    scalars -= set(re.findall(r"float\s*\*+\s*(?:const\s*)?(m\d+)", sig))
    for name in re.findall(r"\b(m\d+)\b", src):
        if name in scalars:
            base.setdefault(name, float(inputs[name].reshape(-1)[0])
                            if name in inputs else 1.0)
        else:
            base.setdefault(name, ke.Ptr(mem, name))
    for name in re.findall(r"\b(\w*_extraOffset)\b", src):
        base.setdefault(name, 0)

    prologue, phases = split_body(src)
    interps = []
    for lane in range(THREADS):
        env = dict(base)
        env["threadIdx"] = type("T", (), {"x": lane, "y": 0, "z": 0})()
        it = ke.Interp(mem, env, limit=40_000_000)
        it.run(prologue)
        interps.append(it)
    for phase in phases:
        for it in interps:
            it.run(phase)
    return mem


def read(mem, name, shape, storage=None):
    import numpy as np
    ashape, lower = (storage or {}).get(name, (shape, (0,) * len(shape)))
    n = 1
    for s in ashape:
        n *= s
    flat = np.array([mem.read(name, i) for i in range(n)])
    out = np.zeros(shape)
    out[tuple(slice(lo, lo + sz) for lo, sz in zip(lower, ashape))] = \
        flat.reshape(ashape, order="F")
    return out
