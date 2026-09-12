# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Every lexic answers the index questions the staging loader asks it.

`GlbToShrLoader._linear_idx()` reads `thread_idx_x`, `thread_idx_y` and
`block_dim_x` off the lexic as text.  Three of the five backends had the
third, and the two that did not were not caught by anything: a blockwide
transfer is what reads it, `preload_globals` is what builds one, and its
default rule is "AMD only" -- so no SYCL or OpenCL target ever reached the
line until the option was turned on by hand.

A field that is read by name and set in five constructors is exactly the kind
of hole a test finds and a reviewer does not, which is why this checks the
*set* of names rather than any one backend's spelling.
"""

from __future__ import annotations

import pytest

from tensorforge.common.basic_types import Datatype
from tensorforge.common.vm.vm import vm_factory


#: (arch, backend) pairs, one per lexic that a target can select.
TARGETS = [
    ('sm_80', 'cuda'),
    ('gfx90a', 'hip'),
    ('pvc', 'oneapi'),
    ('pvc', 'esimd'),
]

#: What the staging loader and the batch loop read off a lexic by name.
INDICES = ('thread_idx_x', 'thread_idx_y', 'block_dim_x', 'block_dim_y',
           'block_idx_x', 'grid_dim_x')


@pytest.mark.parametrize('arch,backend', TARGETS)
@pytest.mark.parametrize('name', INDICES)
def test_the_index_is_spelled(arch, backend, name):
    lexic = vm_factory(arch, backend, Datatype.as_str(Datatype.F32)).get_lexic()
    assert getattr(lexic, name, None), (
        f'{type(lexic).__name__} has no {name}; a caller reading it gets an '
        f'AttributeError rather than a diagnostic')


def test_the_base_class_declares_them_all():
    """So that a missing one is a `None` a caller can test, not an attribute
    error from a constructor that forgot to assign it."""
    from tensorforge.common.vm.lexic.lexic import Lexic
    for name in INDICES:
        assert name in Lexic.__init__.__code__.co_names, name
