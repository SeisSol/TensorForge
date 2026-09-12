# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""The lane-batched MFMA at lead width above one (`amd.componentwise`).

Its lanes are independent rows, so a packed lead operand needs no trip to a
flat layout: component `c` of a width-`w` operand is one element per lane, a
lane-batched problem of its own.  Nothing here runs an MFMA -- the local GPUs
have none -- so these check the arrangement the emitter writes; the numbers
come from a CDNA package.
"""
import re

import pytest

from test_amd_packed_dpp import _kernel

MFMA = re.compile(r'__builtin_amdgcn_mfma_f32_4x4x1f32\(')
#: The nest's broadcast of one lane of `B` -- what a product left to the nest
#: at lead width two reads its operand through.
NEST_BROADCAST = re.compile(r'tensorforge::broadcast<\d+, 1, \d+>')


@pytest.mark.parametrize('arch', ['gfx942', 'gfx90a'])
def test_a_packed_lead_operand_takes_the_matrix_core_componentwise(arch):
    """`local_flux` at lead width two: every product on the matrix core, the
    result stored as pairs.  All columns, the last block padded -- a ninth
    column left to the DPP chain is one it may decline (the 9x9 products'
    contraction ends mid-vector; gfx90a has no 64-bit move), and a declined
    tail sent the whole product to the nest.  So at least as many MFMAs as at
    width one, where the ninth column is the chain's."""
    one = _kernel('local_flux', arch)
    two = _kernel('local_flux', arch, width=2)
    assert len(MFMA.findall(two)) >= len(MFMA.findall(one)) > 0
    assert 'transpose4x4b32(' in two
    assert 'VectorT<float, 2>' in two
    assert not NEST_BROADCAST.search(two), 'no product was left to the nest'
