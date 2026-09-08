# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Capture, for every kernel a codegen run produces, what the frontend was
given and what it made of it.

Everything else here works off that file: it is the ground truth for what a
kernel is supposed to compute, without going through yateto a second time.

Two things are written per kernel. ``descrs`` is the descriptor list the
backend built, which is what the host-side oracle evaluates. ``description``
is the data yateto handed over, which is what a test can replay through
``DescriptionReader`` without yateto being installed at all.

Run it from SeisSol's `codegen/` directory, with `generate.py`'s own arguments:

    python3 dump_descriptors.py --out descriptors.json -- \\
        --equations poroelastic --matricesDir matrices --outputDir gen \\
        --host_arch hsw --device_backend cuda --device_arch sm_86 \\
        --device_vendor nvidia --order 4 --precision s \\
        --numberOfMechanisms 0 --memLayout config/gpu/dense.xml \\
        --multipleSimulations 1 --PlasticityMethod nb \\
        --gemm_tools tensorforge --device_codegen tensorforge \\
        --drQuadRule dunavant
"""
import json
import os
import runpy
import sys

import numpy as np

from tensorforge.frontend.yateto import YatetoFrontend
from tensorforge.generators.descriptions import (ElementwiseDescr,
                                                 MultilinearDescr,
                                                 ReductionDescr)


def _sub(x):
    t = x.tensor
    return dict(name=t.name or t.alias, alias=t.alias,
                shape=list(t.shape),
                ashape=list(t.get_actual_shape()),
                tbbox=[list(t.bbox.lower()), list(t.bbox.upper())],
                bbox=[list(x.bbox.lower()), list(x.bbox.upper())],
                offset=[int(o) for o in x.offset],
                addressing=str(t.addressing),
                is_tmp=bool(t.is_tmp),
                storage=int(t.storage_volume()),
                pack=(list(t.storage_map()) if t.storage_map() is not None
                      else None),
                sliced=bool(getattr(x, "sliced", False)),
                data=(t.data.tolist() if getattr(t.data, 'tolist', None)
                      else (list(t.data) if t.data is not None else None)))


def _multilinear(d):
    keep = [(o, t, p) for o, t, p in zip(d.ops, d.target, d.permute)
            if hasattr(o, "tensor")]
    return dict(kind="multilinear",
                dest=_sub(d.dest),
                ops=[_sub(o) for o, _, _ in keep],
                target=[list(t) for _, t, _ in keep],
                permute=[list(p) for _, _, p in keep],
                add=bool(d.add))


def _elementwise(d):
    """Every operand has the destination's shape, so every axis lines up."""
    srcs = d.tensor_srcs()
    axes = list(range(len(d.dest.bbox.sizes())))
    return dict(kind="elementwise",
                op=d.op.name,
                dest=_sub(d.dest),
                ops=[_sub(o) for o in srcs],
                target=[list(axes) for _ in srcs],
                permute=[list(axes) for _ in srcs],
                scalars=[float(v) for v in d.scalar_srcs()],
                add=False)


def _reduction(d):
    """`dims` are axes of the operand; the ones that survive keep their order,
    so the operand maps onto the destination in order with the reduced axes
    numbered negative, the way a contraction states it."""
    kept, contracted = [], -1
    for axis in range(d.var.bbox.rank()):
        if axis in d.dims:
            kept.append(contracted)
            contracted -= 1
        else:
            kept.append(len([a for a in kept if a >= 0]))
    return dict(kind="reduction",
                op=str(d.op),
                dest=_sub(d.dest),
                ops=[_sub(d.var)],
                target=[kept],
                permute=[list(range(d.var.bbox.rank()))],
                add=False)


def _row(d):
    """One descriptor, as data. `None` for a kind nothing here reads yet --
    a barrier, a region marker -- so that positions still line up."""
    if isinstance(d, MultilinearDescr):
        return _multilinear(d)
    if isinstance(d, ElementwiseDescr):
        return _elementwise(d)
    if isinstance(d, ReductionDescr):
        return _reduction(d)
    return None


def main():
    if "--" not in sys.argv:
        raise SystemExit(__doc__)
    split = sys.argv.index("--")
    mine, theirs = sys.argv[1:split], sys.argv[split + 1:]
    out = "descriptors.json"
    if "--out" in mine:
        out = mine[mine.index("--out") + 1]

    sys.path.insert(0, os.getcwd())
    captured = {}
    descriptions = {}

    recorded = []

    def record(item):
        recorded.append(item)

    sys.argv = ["generate.py"] + theirs
    with YatetoFrontend.capture(record):
        try:
            runpy.run_path("generate.py", run_name="__main__")
        except SystemExit:
            pass

    for i, item in enumerate(recorded):
        # a kernel that failed to build never got a routine to be named after
        name = item.name or f"unbuilt_{i}"
        captured[name] = [_row(d) for d in (item.descrs or [])]
        descriptions[name] = item.description

    def conv(o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        raise TypeError(str(type(o)))

    with open(out, "w") as fh:
        json.dump({"all": captured, "descriptions": descriptions}, fh,
                  default=conv)
    print(f"{len(captured)} kernels -> {out}")


if __name__ == "__main__":
    main()
