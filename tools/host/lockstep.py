# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Execute a generated CUDA kernel on the host, all lanes, one batch element.

`kernel_eval.evaluate_wave` runs a kernel as the generator returns it, on
seed fill.  This runs one cut out of a generated file (`extract`), with the
batch addressing reduced to one element (`flatten_batching`), on the inputs a
capture gives it (`run`), and reads the tensors back as arrays (`read`).  The
lanes advance together, statement by statement (`kernel_eval.Lockstep`), on
one `Slot`: shared memory holds what the other lanes wrote, as it does on the
hardware.
"""
import json
import re

from tensorforge.reference import kernel_eval as ke

#: The lane count of a kernel that does not state its own.
THREADS = 32


def lanes_of(src):
    """The lanes one multiplication runs on, as the kernel's meta line states.

    Not a round number: with `threadIdx.y` held at 0, every lane past the
    kernel's own width is another copy of one of its lanes, on the same
    element.  A copy that repeats an assignment changes nothing, which is why
    a round 32 can look right; one that repeats an accumulation whose load
    and store share a phase adds its term again -- eight times over for a
    4-lane kernel.  `kernel_eval.launch_geometry` makes the same point for the
    single-wave runner, which reads the launcher this does not have.
    """
    m = re.search(r"tensorforge-meta: (\{.*\})\s*$", src, re.M)
    if m:
        try:
            return int(json.loads(m.group(1))["launch"]["threads_per_mult"])
        except (ValueError, KeyError, TypeError):
            pass
    return THREADS


def extract(path, kernel):
    lines = open(path).read().split("\n")
    starts = [(i, re.match(r"\s*kernel_(kernel_\w+)\(", l).group(1))
              for i, l in enumerate(lines)
              if re.match(r"\s*kernel_kernel_\w+\(.*\{$", l)]
    s = [a for a, b in starts if b == kernel][0]
    # up to the brace that closes the kernel: what follows it is not just
    # the launcher -- a launch configuration and a namespace sit between
    depth, e = 0, s
    for e in range(s, len(lines)):
        depth += lines[e].count("{") - lines[e].count("}")
        if depth == 0:
            break
    return "\n".join(lines[s - 2:e + 1])


def flatten_batching(src):
    """One element, no extra offset: make every global pointer point at 0.

    A pointer-based operand is indexed by the element first, and the loop
    variable carries a prefix (`&m0[v5_batchId0][0 + m0_extraOffset]`); the
    offset within the element stays, the extra offset is bound to zero.
    """
    src = re.sub(r"&(m\d+)\[batchId0\]\[[^\]]*\]", r"&\1[0]", src)
    src = re.sub(r"&(m\d+)\[\w*batchId\d+\]\[([^\]]*)\]", r"&\1[\2]", src)
    src = re.sub(r"&(m\d+)\[batchId0 \* \d+ \+ 0 \+ \w+\]", r"&\1[0]", src)
    src = re.sub(r"&(m\d+)\[0 \+ \w+_extraOffset\]", r"&\1[0]", src)
    return src


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
    if not found:
        # a kernel without flags has no guard: the element loop's body is it
        _walk(nodes, lambda n: n[0] == "for" and "batchId0" in str(n[1]), found)
    guard = found[0]
    body = guard[6] if guard[0] == "for" else guard[2]
    stmts = body[1] if isinstance(body, tuple) and body[0] == "block" else body

    # everything the body needs -- pipeline, shared base, glb_ pointers, the
    # loop's own induction -- lives in the blocks around it
    prologue = []

    def collect(node):
        if isinstance(node, tuple):
            if node is guard:
                if node[0] == "for":
                    prologue.append(("expr", f"{node[1]} = {node[2]}"))
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


def run(src, inputs, shapes, storage=None, lanes=None):
    lanes = lanes or lanes_of(src)
    mem = ke.Slot(0)
    # Every slot the kernel can touch has to be defined, or `Slot.read`
    # fabricates one and the comparison measures that instead.  Inputs get
    # their values; everything else -- outputs it accumulates onto, and the
    # shared arena -- starts at zero, which is what the reference assumes.
    import numpy as _np
    for name, shape in shapes.items():
        if not name.startswith("m"):
            continue
        ashape, lower, pack = _storage(storage, name, shape)
        n = len(pack) if pack is not None else 1
        if pack is None:
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
        if pack is not None:
            # Stored compressed: only the cells the map names are there, in
            # the order it names them.  Writing the box densely would put
            # every value at the wrong slot and shorten nothing.
            flat = flat[_np.asarray(pack)]
        for idx in range(max(n, 1)):
            mem.write(name, idx, float(flat[idx]) if idx < len(flat) else 0.0)
    mem.write("flags0", 0, 1)
    m = re.search(r"&totalShrMem\[(\d+) \* threadIdx\.y", src)
    arena = int(m.group(1)) * 2 if m else 0
    for idx in range(arena):
        mem.write("shr", idx, 0.0)

    base = {
        "blockIdx": type("B", (), {"x": 0, "y": 0, "z": 0})(),
        "blockDim": type("D", (), {"x": lanes, "y": 1, "z": 1})(),
        "gridDim": type("G", (), {"x": 1, "y": 1, "z": 1})(),
        "numElements0": 1,
        # every element allowed: a kernel that requires flags reads them
        # without asking for nullptr first
        "flags0": ke.Ptr(mem, "flags0"),
        "totalShrMemPtr": ke.Ptr(mem, "shr"),
    }
    # a SCALAR-addressed tensor is passed by value, not as a pointer -- in
    # either precision: a `double` one taken for a pointer would abort a
    # double-precision kernel at the first product with it
    k = src.index("kernel_kernel_")
    sig = src[src.index("(", k):src.index(")", k)]
    scalars = set(re.findall(r"(?<!\*)\b(?:float|double) (m\d+)\b", sig))
    scalars -= set(re.findall(r"(?:float|double)\s*\*+\s*(?:const\s*)?(m\d+)",
                              sig))
    for name in re.findall(r"\b(m\d+)\b", src):
        if name in scalars:
            base.setdefault(name, float(inputs[name].reshape(-1)[0])
                            if name in inputs else 1.0)
        else:
            base.setdefault(name, ke.Ptr(mem, name))
    for name in re.findall(r"\b(\w*_extraOffset)\b", src):
        base.setdefault(name, 0)

    # Statement by statement, not phase by phase: splitting the body at its
    # top-level barriers (`split_body`) has nothing for a barrier inside a
    # loop -- a rolled face loop has one per iteration -- nor for a lane
    # reading another's register (`readlane`), which needs the other lane at
    # the same statement.  For a body without races, any schedule the
    # barriers allow gives the same values, this one included.
    interps = []
    for lane in range(lanes):
        env = dict(base)
        env["threadIdx"] = type("T", (), {"x": lane, "y": 0, "z": 0})()
        interps.append(ke.Interp(mem, env, limit=40_000_000))
    ke.Lockstep(interps).run(ke.parse(src[src.index("{"):]))
    return mem


def _storage(storage, name, shape):
    """(actual shape, bbox lower, pack map) for one tensor.

    Tolerates the two-element form a capture can carry, which says the
    tensor is stored dense over its box.
    """
    entry = (storage or {}).get(name)
    if entry is None:
        return shape, (0,) * len(shape), None
    if len(entry) == 2:
        return entry[0], entry[1], None
    return entry


def read(mem, name, shape, storage=None):
    import numpy as np
    ashape, lower, pack = _storage(storage, name, shape)
    n = 1
    for s in ashape:
        n *= s
    out = np.zeros(shape)
    if pack is not None:
        stored = np.array([mem.read(name, i) for i in range(len(pack))])
        box = np.zeros(n)
        box[np.asarray(pack)] = stored
        flat = box
    else:
        flat = np.array([mem.read(name, i) for i in range(n)])
    out[tuple(slice(lo, lo + sz) for lo, sz in zip(lower, ashape))] = \
        flat.reshape(ashape, order="F")
    return out
