# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Evaluate a captured descriptor list in NumPy, and lay the tensors out the
way the generated kernel addresses them (first index fastest)."""
import json
import string

import numpy as np

LET = string.ascii_lowercase


def load(path, kernel=None):
    """Descriptors of one kernel out of a capture.

    `dump_descriptors.py` writes every kernel; pass which one, or leave it out
    when the file holds a single list."""
    blob = json.load(open(path))
    if "all" in blob:
        if kernel is None:
            raise SystemExit("this capture holds several kernels; name one of "
                             + ", ".join(sorted(blob["all"])))
        return blob["all"][kernel]
    return blob["descrs"]


def kernels(path):
    blob = json.load(open(path))
    return sorted(blob["all"]) if "all" in blob else []


def tensors_of(descrs):
    """name -> logical shape, plus the set of tensors the kernel writes."""
    shapes, written = {}, set()
    for d in descrs:
        if d is None:
            continue
        for x in [d["dest"]] + [o for o in d["ops"] if o]:
            shapes[x["name"]] = tuple(x["shape"])
        written.add(d["dest"]["name"])
    return shapes, written


def storage_of(descrs):
    """name -> (actual shape, bbox lower).  Storage is compacted to the
    bounding box: address 0 is `lower`, which is how the kernel addresses it."""
    out = {}
    for d in descrs:
        if d is None:
            continue
        for x in [d["dest"]] + [o for o in d["ops"] if o]:
            out[x["name"]] = (tuple(x.get("ashape") or x["shape"]),
                              tuple(x["tbbox"][0]))
    return out


def make(shapes, written, seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    for name, shape in shapes.items():
        if name in written:
            out[name] = np.zeros(shape or (1,), dtype=np.float64)
        else:
            out[name] = rng.standard_normal(shape or (1,))
    return out


def ranges_of(d):
    """Replay _analyze: intersect every index range across operands and dest."""
    rng = {}

    def narrow(t, lo, hi):
        prev = rng.get(t)
        rng[t] = (max(prev[0], lo), min(prev[1], hi)) if prev else (lo, hi)

    for op, tgt in zip(d["ops"], d["target"]):
        if op is None or not tgt:
            continue
        lo, hi = op["bbox"]
        for j, t in enumerate(tgt):
            narrow(t, lo[j], hi[j])
    lo, hi = d["dest"]["bbox"]
    for j in range(len(lo)):
        narrow(j, lo[j], hi[j])
    return rng


def apply(d, arrays):
    rng = ranges_of(d)
    out_rank = len(d["dest"]["bbox"][0])
    labels = {}
    nxt = [0]

    def label(t):
        if t not in labels:
            labels[t] = LET[nxt[0]]
            nxt[0] += 1
        return labels[t]

    for j in range(out_rank):
        label(j)                       # output axes get the first letters

    subs, operands = [], []
    scalar = 1.0
    for op, tgt in zip(d["ops"], d["target"]):
        if op is None:
            continue
        arr = arrays[op["name"]]
        if not tgt:
            scalar = scalar * np.asarray(arr).reshape(-1)[0]
            continue
        sl = tuple(slice(rng[t][0] + op["offset"][j], rng[t][1] + op["offset"][j])
                   for j, t in enumerate(tgt))
        operands.append(arr[sl])
        subs.append("".join(label(t) for t in tgt))

    outs = "".join(label(j) for j in range(out_rank))
    if operands:
        res = np.einsum(f"{','.join(subs)}->{outs}", *operands) * scalar
    else:
        res = np.zeros([rng[j][1] - rng[j][0] for j in range(out_rank)]) + scalar

    dsl = tuple(slice(rng[j][0] + d["dest"]["offset"][j],
                      rng[j][1] + d["dest"]["offset"][j])
                for j in range(out_rank))
    if d["add"]:
        arrays[d["dest"]["name"]][dsl] += res
    else:
        arrays[d["dest"]["name"]][dsl] = res


def run(path, seed=0, kernel=None):
    descrs = load(path, kernel)
    shapes, written = tensors_of(descrs)
    arrays = make(shapes, written, seed)
    inputs = {k: v.copy() for k, v in arrays.items() if k not in written}
    for d in descrs:
        if d is not None:
            apply(d, arrays)
    return arrays, inputs, shapes, written


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        raise SystemExit("usage: reference.py <descriptors.json> [kernel]")
    arrays, inputs, shapes, written = run(
        sys.argv[1], kernel=sys.argv[2] if len(sys.argv) > 2 else None)
    print("tensors:", len(shapes), " written:", len(written))
    for n in sorted(written):
        print(f"  {n:5} {shapes[n]} |x|max={np.abs(arrays[n]).max():.4g}")
