# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""`preload_partial`: the batch-constant operands that fit, staged; the rest
read from global memory.

`local_flux` stages all four of its operators on gfx942 at b = 56, so the
budget the selection sees is shrunk to make it choose -- the arithmetic that
decides what fits is the only thing that differs from a larger case.
"""
import re

import pytest

from test_amd_blgp import _kernel_at

from tensorforge.generators import generator as G

STAGED = re.compile(r'glb_(m\d+) = &totalShrMem\[')
GLOBAL = re.compile(r'GlobalMemspace> const glb_(m[0468]) =')


@pytest.fixture
def budget(monkeypatch):
    """Two of the four 12.5 KB operators under the selection's limit."""
    select = G.Generator._preload_selection
    monkeypatch.setattr(G.Generator, '_preload_selection',
                        lambda self, candidates, cap:
                        select(self, candidates, 30000))


def test_the_operators_that_fit_are_staged_and_the_rest_read_global(budget):
    src = _kernel_at('local_flux', 'gfx942', preload_partial=True)
    assert STAGED.findall(src) == ['m0', 'm4']
    assert sorted(GLOBAL.findall(src)) == ['m6', 'm8']


def test_without_it_they_are_staged_all_or_none(budget):
    src = _kernel_at('local_flux', 'gfx942')
    assert STAGED.findall(src) == ['m0', 'm4', 'm6', 'm8']
