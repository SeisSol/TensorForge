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


def evaluable(d):
    """Whether this row is one this can evaluate.

    A product with a contraction is an einsum and is what everything here
    computes. An elementwise operation or a reduction is recorded too, but
    neither is a product, so evaluating one as if it were would be worse than
    skipping it. A row without a `kind` is a product.
    """
    return d is not None and d.get("kind", "multilinear") == "multilinear"


def tensors_of(descrs):
    """name -> logical shape, plus the set of tensors the kernel writes."""
    shapes, written = {}, set()
    for d in descrs:
        if not evaluable(d):
            continue
        for x in [d["dest"]] + [o for o in d["ops"] if o]:
            shapes[x["name"]] = tuple(x["shape"])
        written.add(d["dest"]["name"])
    return shapes, written


def storage_of(descrs):
    """name -> (actual shape, bbox lower, pack map).

    A dense tensor is compacted to its bounding box: address 0 is `lower`,
    which is how the kernel addresses it, and the pack map is `None` because
    the storage order and the box order are the same thing.

    A sparse one is stored compressed and the map says which cell of the box
    each slot holds.  Reading it as if it were dense is the mistake worth
    naming: the buffer is shorter than the box, so the values land at the
    wrong cells and the comparison measures that instead of the kernel.
    """
    out = {}
    for d in descrs:
        if not evaluable(d):
            continue
        for x in [d["dest"]] + [o for o in d["ops"] if o]:
            pack = x.get("pack")
            out[x["name"]] = (tuple(x.get("ashape") or x["shape"]),
                              tuple(x["tbbox"][0]),
                              tuple(pack) if pack is not None else None)
    return out


def constants_of(descrs):
    """name -> the values baked into the kernel.

    yateto folds constant matrices and scalars straight into the generated
    code, so seeding them randomly compares against arithmetic the kernel never
    does.  A capture carries them; use them."""
    out = {}
    for d in descrs:
        if d is None:
            continue
        for x in [d["dest"]] + [o for o in d["ops"] if o]:
            if x.get("data") is not None:
                out[x["name"]] = np.asarray(x["data"], dtype=np.float64)
    return out


def make(shapes, written, seed=0, constants=None, storage=None):
    """Inputs for one run.

    ``storage`` matters for a sparse tensor: its structural zeros are zeros,
    and the kernel never sees a value there because no slot holds one.  Left
    random, the reference would multiply them in and disagree with a kernel
    that is right -- which is a wrong answer about the kernel, not about the
    pattern.
    """
    rng = np.random.default_rng(seed)
    constants = constants or {}
    out = {}
    for name, shape in shapes.items():
        if name in constants:
            out[name] = constants[name].reshape(shape or (1,)).astype(np.float64)
        elif name in written:
            out[name] = np.zeros(shape or (1,), dtype=np.float64)
        else:
            out[name] = rng.standard_normal(shape or (1,))
    for name, entry in (storage or {}).items():
        pack = entry[2] if len(entry) > 2 else None
        if pack is None or name not in out or name in written:
            continue
        kept = np.zeros(out[name].size)
        flat = out[name].reshape(-1, order='F')
        idx = np.asarray(pack)
        kept[idx] = flat[idx]
        out[name] = kept.reshape(out[name].shape, order='F')
    return out


def seed_destinations(descrs, arrays, shapes, storage=None, seed=0):
    """Give every destination in memory a value on entry; return their names.

    Starting at zero, a destination hides whether the kernel keeps what an
    operation promises (`promised_box`): a missing zero looks like the zero
    it should have written.  A seed makes each of the three checkable -- an
    assignment has to overwrite it, zeros included; a slice has to leave it
    alone outside its box; and an accumulation has to add to it, which is
    what exposes a dropped bias.

    The seed goes where the tensor is stored, and nowhere else: the kernel
    sees nothing outside its box or its pack map, and a value there would
    stay in the reference alone.  A temporary has no entry value the harness
    could give it.
    """
    rng = np.random.default_rng(seed + 5)
    seeded = []
    for d in descrs:
        if not evaluable(d):
            continue
        dest = d["dest"]
        name = dest["name"]
        if dest["is_tmp"] or name in seeded:
            continue
        shape = shapes[name]
        entry = (storage or {}).get(name) or (shape, (0,) * len(shape))
        ashape, lower = tuple(entry[0]), tuple(entry[1])
        pack = entry[2] if len(entry) > 2 else None
        cells = np.asarray(rng.standard_normal(ashape))
        if pack is not None:
            kept = np.zeros(cells.size)
            idx = np.asarray(pack)
            kept[idx] = cells.reshape(-1, order="F")[idx]
            cells = kept.reshape(cells.shape, order="F")
        value = np.zeros(shape or (1,))      # the rank-0 form `make` gives
        value[tuple(slice(lo, lo + n) for lo, n in zip(lower, ashape))] = cells
        arrays[name] = value
        seeded.append(name)
    return set(seeded)


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


def promised_box(d):
    """What an assignment defines, as (lower, upper) in the tensor's
    coordinates; None for an accumulation.

    yateto's `=` defines its whole destination, not the part the operands
    happen to support: where `ranges_of` narrows below the destination, the
    rest is zero.  Which box that is depends on what the destination names --
    the backend's `MultilinearBuilder._promised_box` makes the same call.

    A destination that is the tensor itself promises the tensor's box
    (`tbbox`); its own, narrower box is only the window yateto knows the
    result can be nonzero in.  Nothing outside `tbbox` is stored at all.

    A slice promises its own box: the rest of the tensor belongs to other
    descriptors, and zeroing it would destroy their work.  The capture says
    which one it is (`sliced`); one without the field is taken as a slice
    exactly where it carries an offset, as `SubTensor` does.

    `+=` promises nothing: it is defined in terms of what is there.
    """
    if d["add"]:
        return None
    dest = d["dest"]
    if dest.get("sliced") or any(dest["offset"]):
        lo, hi = dest["bbox"]
        off = dest["offset"]
        return ([l + o for l, o in zip(lo, off)],
                [h + o for h, o in zip(hi, off)])
    return tuple(list(x) for x in dest["tbbox"])


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
    extent = [rng[j][1] - rng[j][0] for j in range(out_rank)]
    if operands:
        # An output index no operand carries is a *broadcast*: the same value
        # goes to every position along it.  einsum cannot state that, so
        # contract over what is carried and spread the result afterwards.
        carried = {c for sub in subs for c in sub}
        kept = "".join(c for c in outs if c in carried)
        res = np.einsum(f"{','.join(subs)}->{kept}", *operands) * scalar
        if kept != outs:
            shape, k = [], 0
            for c in outs:
                if c in carried:
                    shape.append(res.shape[k])
                    k += 1
                else:
                    shape.append(1)
            res = np.broadcast_to(res.reshape(shape), extent).copy()
    else:
        res = np.zeros(extent) + scalar

    dsl = tuple(slice(rng[j][0] + d["dest"]["offset"][j],
                      rng[j][1] + d["dest"]["offset"][j])
                for j in range(out_rank))
    dest = arrays[d["dest"]["name"]]
    if d["add"]:
        dest[dsl] += res
        return
    # An assignment defines its whole promise; assigning just the computed
    # part would leave the rest of it as whatever an earlier write put
    # there -- the very defect this is the oracle for, and one it could then
    # not see.
    box = promised_box(d)
    if out_rank and box is not None:
        dest[tuple(slice(l, h) for l, h in zip(*box))] = 0.0
    dest[dsl] = res


def run(path, seed=0, kernel=None):
    descrs = load(path, kernel)
    shapes, written = tensors_of(descrs)
    arrays = make(shapes, written, seed, constants_of(descrs))
    inputs = {k: v.copy() for k, v in arrays.items() if k not in written}
    for d in descrs:
        if evaluable(d):
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
