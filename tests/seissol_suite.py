# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""The SeisSol kernels, as SeisSol's code generator hands them to a GPU
exporter: every equation system, order 2 to 8, single and double.

`fixtures/seissol/` holds what `tools/seissol_export.py` recorded, packed by
`tools/seissol_store.py`: a constant tensor's values are stored once, by hash,
in `values.json.xz` and named by `{"ref": hash}`; each equation system's file
holds its distinct descriptions by hash and, per configuration, which kernel
is which.  This reads it back into the descriptions yateto sent.
"""

from __future__ import annotations

import functools
import json
import lzma
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

STORE = Path(__file__).resolve().parent / 'fixtures' / 'seissol'


@functools.lru_cache(maxsize=None)
def _values() -> Dict[str, object]:
    return json.loads(lzma.decompress((STORE / 'values.json.xz').read_bytes()))


@functools.lru_cache(maxsize=None)
def _system(name: str) -> dict:
    return json.loads(lzma.decompress((STORE / f'{name}.json.xz').read_bytes()))


def systems() -> List[str]:
    """`equation-solver`, one per file."""
    return sorted(p.name[:-len('.json.xz')] for p in STORE.glob('*-*.json.xz'))


def configs() -> Iterator[Tuple[str, str]]:
    """`(system, configuration)`, e.g. `('elastic-linearck',
    'elastic-linearck-o4-s')`."""
    for system in systems():
        for config in sorted(_system(system)['configs']):
            yield system, config


def config(system: str, name: str) -> dict:
    """What the configuration was generated with."""
    return _system(system)['configs'][name]['config']


def kernels(system: str, name: str) -> List[str]:
    return sorted(_system(system)['configs'][name]['kernels'])


def description(system: str, name: str, kernel: str) -> dict:
    """The description yateto sent for `kernel`, values resolved; a fresh
    copy, which a reader may change."""
    content = _system(system)
    digest = content['configs'][name]['kernels'][kernel]
    desc = json.loads(json.dumps(content['descriptions'][digest]))
    for tensor in desc['tensors']:
        ref = tensor.get('values')
        if isinstance(ref, dict) and set(ref) == {'ref'}:
            tensor['values'] = json.loads(json.dumps(_values()[ref['ref']]))
    return desc
