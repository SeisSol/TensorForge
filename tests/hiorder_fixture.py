# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""SeisSol's elastic volume at orders 10 and 12, for the high-order tile's
tests: yateto's description of order 8 (`seissol_suite`), its tensors and
operations at the order's basis, and the operators kDivM(0..2) on their
patterns at the order (tests/fixtures/seissol/hiorder-kdivm.npz, which
tools/hiorder_fixture.py packs from the modal basis's operators -- at order 8
SeisSol's own, entry for entry).

The tree keeps the patterns alone: their values are seeded here, of
SeisSol's magnitudes and none of them zero.  A job that measures accuracy
hands the real operators in (`description`'s `values`).
"""

from __future__ import annotations

import copy
from typing import Dict, Optional

import numpy as np

import seissol_suite as fx

PATTERNS = fx.STORE / 'hiorder-kdivm.npz'
ORDERS = (10, 12)
SYSTEM = 'elastic-linearck'


def basis(order: int) -> int:
    """The modal basis's functions at `order`: the tetrahedral number."""
    return order * (order + 1) * (order + 2) // 6


def pattern(order: int, name: str) -> np.ndarray:
    """The nonzero pattern of the operator `name` at `order`, a boolean
    matrix of the basis's functions square."""
    found = np.load(PATTERNS)
    rows, cols = (int(x) for x in found[f'o{order}/{name}/shape'])
    bits = np.unpackbits(found[f'o{order}/{name}/bits'])[:rows * cols]
    return bits.reshape(rows, cols).astype(bool)


def seeded(order: int, name: str, where: np.ndarray) -> np.ndarray:
    """Values on the pattern `where`: tens, either sign, none zero, seeded
    by the order and the operator."""
    rng = np.random.default_rng(1000 * order + int(name[-2]))
    out = np.zeros(where.shape)
    count = int(where.sum())
    out[where] = rng.choice((-1.0, 1.0), count) * rng.uniform(1.0, 60.0,
                                                               count)
    return out


def description(order: int, precision: str = 's',
                values: Optional[Dict[str, np.ndarray]] = None) -> dict:
    """The gpu_volume description at `order` (10 or 12) in `precision`:
    order 8's, its basis's sizes the order's -- the elements' functions
    padded to 64 bytes, the derivatives' columns the order below's -- and
    each operator's entries on its pattern, its rows of Q from the first
    that holds an entry to the last.  `values`, by operator, the real
    operators; seeded values without."""
    if order not in ORDERS:
        raise ValueError(f'order {order}: the fixture holds {ORDERS}')
    desc = copy.deepcopy(fx.description(
        SYSTEM, f'{SYSTEM}-o8-{precision}', 'gpu_volume'))
    n, m = basis(order), basis(order - 1)
    unit = 64 // (4 if precision == 's' else 8)
    npad = -(-n // unit) * unit
    mpad = -(-m // unit) * unit
    rows = {}
    for tensor in desc['tensors']:
        name = tensor['name']
        storage = tensor['storage']
        if name in ('I', 'Q'):
            storage.update(shape=[npad, 9], sizes=[npad, 9])
        elif name.startswith('_tmp'):
            storage.update(shape=[n, 9], sizes=[mpad, 9])
        elif name.startswith('kDivM'):
            where = pattern(order, name)
            if where[:, m:].any():
                raise ValueError(f'{name} at order {order}: entries past the '
                                 f'order below\'s {m} columns')
            numbers = (values[name] if values is not None
                       else seeded(order, name, where))
            found = np.nonzero(where.any(axis=1))[0]
            rows[name] = (int(found.min()), int(found.max()) + 1)
            storage.update(shape=[npad, n], sizes=[npad, m])
            tensor['values'] = {'kind': 'entries', 'data': [
                [[int(i), int(j)], float(numbers[i, j])]
                for i, j in np.argwhere(where)]}
    for op in desc['operations']:
        result, args = op['result'], op['args']
        if result['name'].startswith('_tmp'):
            result['bbox'] = [[0, 0], [m, 9]]
            args[0]['bbox'] = [[0, 0], [m, 9]]
            continue
        operator, tmp = args
        r0, r1 = rows[operator['name']]
        result['bbox'] = [[r0, 0], [r1, 9]]
        operator['bbox'] = [[r0, 0], [r1, m]]
        tmp['bbox'] = [[0, 0], [m, 9]]
    return desc
