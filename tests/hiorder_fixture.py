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
FLUX_PATTERNS = fx.STORE / 'premultiplied-flux.npz'
ORDERS = (10, 12)
FLUX_ORDERS = (6, 8, 10, 12)
SYSTEM = 'elastic-linearck'


def basis(order: int) -> int:
    """The modal basis's functions at `order`: the tetrahedral number."""
    return order * (order + 1) * (order + 2) // 6


def pattern(order: int, name: str, store=PATTERNS) -> np.ndarray:
    """The nonzero pattern of the operator `name` at `order`, a boolean
    matrix of the basis's functions square; the premultiplied fluxes' from
    their own `store` (`FLUX_PATTERNS`)."""
    found = np.load(store)
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


#: SeisSol's premultiplied flux kernels, as its GPU builds run them
#: (PREMULTIPLY_FLUX), by name: each term's matrix of the element (A) and
#: its operator.
FLUXES = {
    'gpu_localFluxAllPre': tuple((f'AplusTAll({i})', f'plusFluxMatrices({i})')
                                 for i in range(4)),
    'gpu_neighboringFluxPre[0]': (('AminusT', 'minusFluxMatrices(0)'),)}


def volume_description(order: int, precision: str = 's') -> dict:
    """The gpu_volume description at `order`: the store's to order 8, this
    fixture's at 10 and 12 (`description`)."""
    if order in ORDERS:
        return description(order, precision)
    return fx.description(SYSTEM, f'{SYSTEM}-o{order}-{precision}',
                          'gpu_volume')


def flux_description(kernel: str, order: int, precision: str = 's') -> dict:
    """The premultiplied flux kernel `kernel` (`FLUXES`) at `order` (6 to
    12): gpu_localFluxAllPre, Q += sum_{i<4} plusFluxMatrices(i) (I
    AplusTAll(i)), and gpu_neighboringFluxPre[0], Q += minusFluxMatrices(0)
    (I AminusT).  The volume's description at the order made a sum of such
    terms, each the volume's first -- I times a 9x9 matrix of the element,
    an operator times that -- its operator stored on all of the basis's
    functions, its entries on the operator's pattern at the order
    (`FLUX_PATTERNS`, which tools/hiorder_fixture.py packs from the modal
    basis's rDivM, fMrT, fP and rT), seeded as the volume's are.  The store
    holds the fluxes as yateto sends them, unpremultiplied, whose two, three
    and five asynchronous steps the high-order tile does not take."""
    if order not in FLUX_ORDERS:
        raise ValueError(f'order {order}: the fixture holds {FLUX_ORDERS}')
    base = volume_description(order, precision)
    named = {t['name']: t for t in base['tensors']}
    first, second = base['operations'][0], base['operations'][1]
    n = basis(order)
    npad = named['I']['storage']['shape'][0]
    tensors = [copy.deepcopy(named['I']), copy.deepcopy(named['Q'])]
    operations = []
    for t, (aname, fname) in enumerate(FLUXES[kernel]):
        a = copy.deepcopy(named['star(0)'])
        a['name'] = aname
        tmp = copy.deepcopy(named['_tmp0'])
        tmp['name'] = f'_tmp{t}'
        tmp['storage'].update(shape=[npad, 9], sizes=[npad, 9])
        f = copy.deepcopy(named['kDivM(0)'])
        f['name'] = fname
        f['storage'].update(shape=[npad, n], sizes=[npad, n])
        where = pattern(order, fname, FLUX_PATTERNS)
        numbers = seeded(order, fname, where)
        f['values'] = {'kind': 'entries', 'data': [
            [[int(i), int(j)], float(numbers[i, j])]
            for i, j in np.argwhere(where)]}
        s0 = copy.deepcopy(named['_scalar0'])
        s0['name'] = f'_scalar{2 * t}'
        s1 = copy.deepcopy(named['_scalar1'])
        s1['name'] = f'_scalar{2 * t + 1}'
        tensors += [a, tmp, s0, f, s1]
        mix = copy.deepcopy(first)
        mix['result']['name'] = tmp['name']
        mix['result']['bbox'] = [[0, 0], [n, 9]]
        mix['args'][0]['bbox'] = [[0, 0], [n, 9]]
        mix['args'][1]['name'] = aname
        mix['linear']['alpha']['name'] = s0['name']
        # Its rows of Q from the first that holds an entry to the last.
        found = np.nonzero(where.any(axis=1))[0]
        r0, r1 = int(found.min()), int(found.max()) + 1
        product = copy.deepcopy(second)
        product['result']['bbox'] = [[r0, 0], [r1, 9]]
        product['args'][0]['name'] = fname
        product['args'][0]['bbox'] = [[r0, 0], [r1, n]]
        product['args'][1]['name'] = tmp['name']
        product['args'][1]['bbox'] = [[0, 0], [n, 9]]
        product['linear']['alpha']['name'] = s1['name']
        operations += [mix, product]
    out = copy.deepcopy(base)
    out['tensors'], out['operations'] = tensors, operations
    return out
