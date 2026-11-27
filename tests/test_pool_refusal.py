# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""The mixer pool refused since P3 (Horizon 107909, a GB200;
`element_group.POOL_REFUSED`, `POOL_P3`): `matrix_pool=on` declined on every
target and at every order, the kernel the bands' but for its name, its
options' line and the note naming P3 -- wrong under perturbed timing at o6
and o10, A's slots numbered per trip, and no faster at o8.  The pool's
implementation stays, tested where tests/test_hiorder_pool.py lifts the
refusal; its protocol under the kernel's numbering: tests/test_pool_model.py."""

from __future__ import annotations

import re

import pytest

import test_hiorder_tile as hi
import test_multi_element as me
from tensorforge.common.options import registry
from tensorforge.generators import element_group as eg

#: The pool asked for.
POOL = {'matrix_pool': 'on'}

#: Where the pool was taken before P3, and where it was declined for
#: reasons of its own: the refusal is P3's at all of them.
ASKED = [('sm_100a', 4, {}), ('sm_100a', 6, {}), ('sm_100a', 8, {}),
         ('sm_100a', 10, {}), ('sm_100a', 12, {}),
         ('sm_100a', 8, {'matrix_input_chunks': 'whole'}),
         ('sm_100a', 8, {'matrix_bands': 3}),
         ('sm_100a', 8, {'matrix_groups': 'unrolled'}),
         ('sm_90a', 8, {})]


def _note(arch, order, **options):
    context = me._context(arch, 's', multi_element='force',
                          **{**hi.HIGH, **POOL, **options})
    group = hi._group(order, arch, **POOL, **options)
    return group, eg.tile_notes(group, context)


@pytest.mark.parametrize('arch,order,options', ASKED)
def test_the_pool_is_refused_everywhere(arch, order, options):
    """`matrix_pool=on` declined, the bands' tile, the note naming P3."""
    group, notes = _note(arch, order, **options)
    assert group.section.tile.pool == 'off'
    assert f'matrix_pool=on declined: {eg.POOL_P3}' in notes
    assert eg.POOL_P3.startswith('P3 (Horizon 107909) found it wrong under '
                                 'perturbed timing at o6 and o10')
    assert 'no faster at o8, 6.17 ns an element against the bands\' 6.01' \
        in eg.POOL_P3


@pytest.mark.parametrize('order', [6, 8])
def test_a_refused_pool_is_the_bands_kernel(order):
    """The kernel of `matrix_pool=on` the bands' but for its name, its
    options' line and the note."""
    def lines(src):
        return [line for line in re.sub(r'kernel_kernel_[0-9a-f]+', 'k',
                                        src).splitlines()
                if '// options: ' not in line
                and 'matrix_pool=on declined' not in line]
    asked = hi._kernel(order, 'sm_100a', **POOL).get_kernel()
    assert 'matrix_pool=on declined: P3 (Horizon 107909)' in asked
    assert lines(asked) == lines(hi._kernel(order, 'sm_100a').get_kernel())


def test_the_options_doc_names_p3():
    """The option's doc says it is declined, and why."""
    doc = registry()['matrix_pool'].doc
    assert 'declined everywhere since P3 (Horizon 107909' in doc
    assert 'no faster at o8' in doc
