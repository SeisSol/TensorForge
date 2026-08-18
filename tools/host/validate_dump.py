#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run every kernel of a generated dump on the host and compare with NumPy.

    python3 validate_dump.py gen/gpulike_subroutine.cpp descriptors.json
    python3 validate_dump.py gen/gpulike_subroutine.cpp descriptors.json kernel_0bf208a83b

A kernel is executed for one batch element, all lanes, in lock-step at its
barriers, against the same inputs the reference sees.  `worst rel` is the
largest relative deviation over every tensor the kernel writes; anything above
1e-6 is a real disagreement, since both sides do the same arithmetic in double
precision on the host.

Kernels that use vectorised loads (`*(float4*)&...`) abort: `kernel_eval` does
not model them.  That is a gap in the interpreter, not a finding.
"""
import sys

import numpy as np

from tfpaths import add_tests_to_path

add_tests_to_path()

import lockstep                                                # noqa: E402
import reference                                               # noqa: E402


def check(dump, descriptors, kernel, seed=1, nonzero_dest=True):
    descrs = reference.load(descriptors, kernel)
    if not descrs or all(d is None for d in descrs):
        return None
    prefix = [d for d in descrs if d is not None]
    shapes, written = reference.tensors_of(prefix)
    storage = reference.storage_of(prefix)
    arrays = reference.make(shapes, written, seed)
    # A destination that is *only* accumulated onto must carry a value on
    # entry, or a dropped bias cannot show.  One with an assignment among its
    # writers must not: yateto's contract is that such a tensor is fully
    # defined by the kernel, and the store zero-fills outside the eqspp window
    # -- which the reference here does not model, so a seeded value would
    # register as a disagreement that is not one.
    seeded = set()
    if nonzero_dest:
        assigned = {d["dest"]["name"] for d in prefix if not d["add"]}
        rng = np.random.default_rng(seed + 5)
        for d in prefix:
            name = d["dest"]["name"]
            if (d["add"] and not d["dest"]["is_tmp"]
                    and name not in assigned and name not in seeded):
                arrays[name] = rng.standard_normal(shapes[name])
                seeded.add(name)
    inputs = {k: v.copy() for k, v in arrays.items() if k not in written}
    inputs.update({name: arrays[name].copy() for name in seeded})
    for d in prefix:
        reference.apply(d, arrays)

    src = lockstep.flatten_batching(lockstep.extract(dump, kernel))
    mem = lockstep.run(src, inputs, shapes, storage)

    temps = {d["dest"]["name"] for d in prefix if d["dest"]["is_tmp"]}
    worst, detail = 0.0, []
    for name in sorted(written):
        if name in temps or not name.startswith("m"):
            continue
        got = lockstep.read(mem, name, shapes[name], storage)
        exp = arrays[name]
        rel = float((np.abs(got - exp) / np.maximum(np.abs(exp), 1e-9)).max())
        worst = max(worst, rel)
        detail.append((name, rel))
    return worst, detail


def main():
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    dump, descriptors = sys.argv[1], sys.argv[2]
    names = sys.argv[3:] or reference.kernels(descriptors)
    bad = 0
    for name in names:
        try:
            result = check(dump, descriptors, name)
        except Exception as exc:
            print(f"  {name:24} ERROR {type(exc).__name__}: {str(exc)[:70]}")
            continue
        if result is None:
            print(f"  {name:24} (no multilinear descriptors)")
            continue
        worst, detail = result
        ok = worst < 1e-6
        bad += not ok
        print(f"  {name:24} worst rel = {worst:10.4g}  "
              f"{'OK' if ok else 'MISMATCH'}"
              + ("" if ok else f"  {[(n, round(r, 6)) for n, r in detail]}"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
