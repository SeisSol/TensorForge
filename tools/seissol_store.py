# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Pack the SeisSol exports (seissol_export.py) into a content-addressed store for the TensorForge repo.
  seissol_store.py SRC_DIR DEST_DIR
Constant tensors carry their values, and the same flux or stiffness matrix turns up in hundreds of kernels:
values are stored once, by hash, in `values.json.xz`, and a tensor names its values by `{"ref": hash}`.
Identical descriptions are stored once per equation file (`<equation>-<solver>.json.xz`), and each
configuration maps its kernel names onto description hashes."""
import collections, hashlib, json, lzma, pathlib, sys

src, dest = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
dest.mkdir(parents=True, exist_ok=True)


def digest(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':')).encode()).hexdigest()[:24]


values = {}
by_system = collections.defaultdict(lambda: {'configs': {}, 'descriptions': {}})
source = None
for path in sorted(src.glob('*.json')):
    blob = json.loads(path.read_text())
    source = source or blob['source']
    cfg = blob['config']
    # recorded before the field existed, and by runs the workaround never acted on
    cfg.setdefault('workarounds', [])
    system = by_system[f'{cfg["equation"]}-{cfg["solver"]}']
    kernels = {}
    for name, desc in blob['descriptions'].items():
        desc = json.loads(json.dumps(desc))
        for tensor in desc['tensors']:
            if tensor.get('values'):
                h = digest(tensor['values'])
                values.setdefault(h, tensor['values'])
                tensor['values'] = {'ref': h}
        h = digest(desc)
        system['descriptions'].setdefault(h, desc)
        kernels[name] = h
    system['configs'][path.stem] = {'config': cfg, 'kernels': kernels}

sizes = {}
for system, content in by_system.items():
    out = dest / f'{system}.json.xz'
    out.write_bytes(lzma.compress(json.dumps({'source': source, **content}, separators=(',', ':')).encode(),
                                  preset=9 | lzma.PRESET_EXTREME))
    sizes[out.name] = (len(content['configs']), len(content['descriptions']), out.stat().st_size)
out = dest / 'values.json.xz'
out.write_bytes(lzma.compress(json.dumps(values, separators=(',', ':')).encode(), preset=9 | lzma.PRESET_EXTREME))
sizes[out.name] = (0, len(values), out.stat().st_size)
total = 0
for name, (configs, count, size) in sorted(sizes.items()):
    total += size
    print(f'{name:40s} configs {configs:3d}  entries {count:6d}  {size / 2**20:6.2f} MiB')
print(f'total {total / 2**20:.1f} MiB')
