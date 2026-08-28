# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Capture, for every kernel a codegen run produces, the descriptor list the
frontend handed the backend.

Everything else here works off that file: it is the ground truth for what a
kernel is supposed to compute, without going through yateto a second time.

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

from tensorforge.generators.descriptions import MultilinearDescr
from tensorforge.generators.generator import Generator


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
                sliced=bool(getattr(x, "sliced", False)),
                data=(t.data.tolist() if getattr(t.data, 'tolist', None)
                      else (list(t.data) if t.data is not None else None)))


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
    original = Generator.generate

    def generate(self, *a, **kw):
        result = original(self, *a, **kw)
        rows = []
        for d in self.descr_list:
            if not isinstance(d, MultilinearDescr):
                rows.append(None)
                continue
            keep = [(o, t, p) for o, t, p in zip(d.ops, d.target, d.permute)
                    if hasattr(o, "tensor")]
            rows.append(dict(dest=_sub(d.dest),
                             ops=[_sub(o) for o, _, _ in keep],
                             target=[list(t) for _, t, _ in keep],
                             permute=[list(p) for _, _, p in keep],
                             add=bool(d.add)))
        captured[self.get_base_name()] = rows
        return result

    Generator.generate = generate
    sys.argv = ["generate.py"] + theirs
    try:
        runpy.run_path("generate.py", run_name="__main__")
    except SystemExit:
        pass

    def conv(o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        raise TypeError(str(type(o)))

    with open(out, "w") as fh:
        json.dump({"all": captured}, fh, default=conv)
    print(f"{len(captured)} kernels -> {out}")


if __name__ == "__main__":
    main()
