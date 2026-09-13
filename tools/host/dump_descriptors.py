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


def _row(d):
    """One descriptor, as data (`to_dict`, with the values and the storage
    order).  `None` for a kind nothing here reads yet -- a barrier, a region
    marker, a merged run -- so that positions still line up."""
    if isinstance(d, (MultilinearDescr, ElementwiseDescr, ReductionDescr)):
        return d.to_dict(data=True, pack=True)
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
        if item.description is not None:
            # a kernel that arrived as terms has none, and a null read back is
            # not a description of anything
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
