# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""SeisSol's element tensors on the host oracle with their padding poisoned:
every cell of a stored box no operation's box holds -- the rows past what
yateto writes and reads -- NaN, or a number of its tensor's own.

What is checked here: no padding cell reaches a logical one -- a padding
row read into a product or a copy, NaN times the zeros of an operator's
padding the old defect -- in the slot form, in the per-element kernel and
in the matrix tiling; and the slot form leaves its outputs' padding as it
found it (it writes the rows the boxes hold, a tile of sixteen bytes it
straddles read and written back whole), where the per-element kernel
writes an assigned output's padding zero.  CPU only.
"""

from __future__ import annotations

import numpy as np
import pytest

import test_multi_element as me
import test_slot_mapping as sm

#: The kernels, their forms: the slot form, the per-element kernel, and the
#: matrix tiling (`multi_element=force`, mma_sync, its operands staged).
CASES = [('gpu_derivative', 4, 's', 'slots'),
         ('gpu_derivative', 4, 'd', 'slots'),
         ('gpu_derivative', 6, 's', 'slots'),
         ('gpu_volume', 4, 's', 'slots'),
         ('gpu_localFluxAll', 4, 's', 'slots'),
         ('gpu_derivative', 4, 's', 'element'),
         ('gpu_volume', 4, 's', 'element'),
         ('gpu_volume', 4, 's', 'tilings'),
         ('gpu_localFlux[0]', 5, 's', 'tilings')]
FORMS = {'slots': sm.SLOTS, 'element': {},
         'tilings': {'multi_element': 'force'}}


def _run(kernel, order, precision, form, padding):
    """The kernel in `form` on the oracle over two elements, each padding
    cell `padding(alias)`: the outputs' boxes after it, NumPy's, the masks."""
    descrs = sm._seissol(kernel, order, precision)
    gen = me._generate(descrs, me._context('sm_90', precision, **FORMS[form]))
    group = gen.built._element_group
    assert (group is None) == (form == 'element')
    tensors, inputs, scalars, after = me._on_the_oracle(gen, 2,
                                                        padding=padding)
    want = me._reference_all(descrs)(inputs, scalars)
    return ({a: after(tensors[a]) for a in want}, want,
            me._padding_cells(gen))


@pytest.mark.parametrize('kernel,order,precision,form', CASES,
                         ids=[f'{k}-o{o}{p}-{f}' for k, o, p, f in CASES])
def test_no_padding_cell_reaches_a_logical_one(kernel, order, precision,
                                               form):
    """NaN in every padding cell of every element tensor, inputs and
    outputs: every logical cell agrees with NumPy and holds no NaN."""
    got, want, masks = _run(kernel, order, precision, form,
                            lambda alias: float('nan'))
    tolerance = (1e-12 if precision == 'd' else
                 1e-5 if form == 'tilings' else 1e-6)
    for alias in want:
        logical = ~masks[alias]
        for e in range(2):
            mine, theirs = got[alias][e][logical], want[alias][e][logical]
            assert not np.isnan(mine).any(), (alias, e)
            scale = np.abs(theirs).max() or 1.0
            assert np.abs(mine - theirs).max() / scale <= tolerance, (alias, e)


@pytest.mark.parametrize('kernel,order,precision',
                         [c[:3] for c in CASES if c[3] == 'slots'])
def test_the_slot_form_leaves_its_outputs_padding_as_found(kernel, order,
                                                           precision):
    """Each tensor's padding a number of its own: after the slot form every
    output's padding cell holds its own still -- the per-element kernel's
    zero is not the slot form's, and no other tensor's number is copied
    in."""
    own = {}
    got, want, masks = _run(kernel, order, precision, 'slots',
                            lambda alias: own.setdefault(alias,
                                                         1000.0 + len(own)))
    for alias in want:
        for e in range(2):
            assert (got[alias][e][masks[alias]] == own[alias]).all(), alias
