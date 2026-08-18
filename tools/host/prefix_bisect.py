#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Rebuild a captured descriptor list and find the shortest prefix that is wrong.

The kernels come out of SeisSol through yateto; reproducing one by hand is
guesswork.  Rebuilding it from what the frontend actually handed the backend
is not, and a prefix that still misbehaves is a minimal case.
"""
import json
import sys

import numpy as np

from tfpaths import add_tests_to_path

add_tests_to_path()

import lockstep as lanes                                       # noqa: E402
import reference as ref                                        # noqa: E402
from tensorforge.common.basic_types import Addressing, Datatype  # noqa: E402
from tensorforge.common.context import Context                 # noqa: E402
from tensorforge.common.matrix.boundingbox import BoundingBox   # noqa: E402
from tensorforge.common.matrix.tensor import SubTensor, Tensor  # noqa: E402
from tensorforge.generators.descriptions import MultilinearDescr  # noqa: E402
from tensorforge.generators.generator import Generator         # noqa: E402

DT = Datatype.F32
ADDR = {str(a): a for a in Addressing}


def build(descrs, upto=None):
    """(descr objects, name->Tensor).  Tensors are shared across descriptors."""
    tensors = {}

    def tensor(x):
        if x["name"] not in tensors:
            tensors[x["name"]] = Tensor(
                x["shape"], ADDR[x["addressing"]],
                BoundingBox(list(x["tbbox"][0]), list(x["tbbox"][1])),
                alias=x["name"], is_tmp=x["is_tmp"],
                data=x["data"], datatype=DT)
        return tensors[x["name"]]

    def sub(x):
        return SubTensor(tensor(x),
                         BoundingBox(list(x["bbox"][0]), list(x["bbox"][1])),
                         list(x["offset"]), sliced=x["sliced"])

    out = []
    for d in descrs[:upto]:
        if d is None:
            continue
        out.append(MultilinearDescr(
            dest=sub(d["dest"]),
            ops=[sub(o) for o in d["ops"] if o],
            target=[list(t) for t, o in zip(d["target"], d["ops"]) if o],
            permute=[list(p) for p, o in zip(d["permute"], d["ops"]) if o],
            add=d["add"]))
    return out, tensors


def prefix_descrs(descrs, upto):
    return [d for d in descrs[:upto] if d is not None]


def evaluate(descrs, upto, backend="cuda", arch="sm_86", seed=1):
    """Generate the prefix, run every lane, compare with NumPy."""
    live, tensors = build(descrs, upto)
    gen = Generator(live, Context(arch=arch, backend=backend, fp_type=DT))
    gen.generate()
    src = lanes.flatten_batching(
        "__global__ void\n" + gen.get_kernel())

    prefix = [d for d in descrs[:upto] if d is not None]
    shapes, written = ref.tensors_of(prefix)
    storage = ref.storage_of(prefix)
    arrays = ref.make(shapes, written, seed)
    # A destination that is *only* accumulated onto must carry a value on
    # entry, or a dropped bias cannot show.  One with an assignment among its
    # writers must not -- see the note in validate_dump.py.
    prefix = prefix_descrs(descrs, upto)
    assigned = {d["dest"]["name"] for d in prefix if not d["add"]}
    seeded = set()
    rng = np.random.default_rng(seed + 5)
    for d in prefix:
        n = d["dest"]["name"]
        if (d["add"] and not d["dest"]["is_tmp"]
                and n not in assigned and n not in seeded):
            arrays[n] = rng.standard_normal(shapes[n])
            seeded.add(n)
    inputs = {k: v.copy() for k, v in arrays.items() if k not in written}
    inputs.update({n: arrays[n].copy() for n in seeded})
    for d in prefix:
        ref.apply(d, arrays)

    # the generator assigns its own names; read them off the objects it saw
    rename = {orig: t.name for orig, t in tensors.items() if t.name}
    ins = {rename[k]: v for k, v in inputs.items() if k in rename}
    shp = {rename.get(k, k): v for k, v in shapes.items()}
    sto = {rename.get(k, k): v for k, v in storage.items()}

    # temporaries never reach memory the harness can read
    temps = {d["dest"]["name"] for d in prefix if d["dest"]["is_tmp"]}
    temps |= {o["name"] for d in prefix for o in d["ops"] if o and o["is_tmp"]}

    mem = lanes.run(src, ins, shp, sto)
    worst = 0.0
    for name in written:
        if name not in rename or name in temps:
            continue
        got = lanes.read(mem, rename[name], shapes[name], sto)
        exp = arrays[name]
        worst = max(worst, float((np.abs(got - exp)
                                  / np.maximum(np.abs(exp), 1e-9)).max()))
    return worst


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit(
            "usage: prefix_bisect.py <descriptors.json> <kernel>\n\n"
            "Rebuilds the descriptor list as live objects, generates a kernel\n"
            "for a prefix of it, and bisects to the shortest prefix that gets\n"
            "the wrong answer -- a minimal reproducer straight from the real\n"
            "kernel, rather than a guess at what it looked like.")
    descriptors, kernel = sys.argv[1], sys.argv[2]
    descrs = json.load(open(descriptors))["all"][kernel]
    n = len([d for d in descrs if d is not None])
    print(f"{kernel}: {n} descriptors")
    lo, hi = 1, n
    while lo < hi:
        mid = (lo + hi) // 2
        try:
            w = evaluate(descrs, mid)
        except Exception as exc:
            print(f"  prefix {mid:3}: ERROR {type(exc).__name__}: {str(exc)[:70]}")
            lo = mid + 1
            continue
        print(f"  prefix {mid:3}: worst rel = {w:.4g}")
        if w > 1e-6:
            hi = mid
        else:
            lo = mid + 1
    print(f"shortest failing prefix: {lo}")
