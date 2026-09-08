# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Merging a region no mask reaches.

`dppUpdate` carries a region in a row mask and a bank mask, which express a
product and stop at four lanes.  One lane out of every four -- what transposing
a 4x4 needs -- is outside them, and the merge refused it on the grounds that
the path reads no lane id.  It does: `transpose4x4b32` writes
``__lane_id() % 2 == 0 ? v1 : vv2`` four times, and the assembly it keeps in a
comment fuses that with the shuffle as `v_cndmask_b32_dpp`.
"""

from __future__ import annotations

import pathlib
import re

import pytest

from tensorforge.backend.instructions.compute.primitives.amd import (
    exchange_codegen, reorder)
from tensorforge.backend.pir.build import IRBuilder
from tensorforge.backend.pir.core import ScalarType
from tensorforge.common.basic_types import Datatype
from tensorforge.common.context import Context

HEADER = pathlib.Path(
    exchange_codegen.__file__).parents[5] / 'include' / \
    'tensorforge_device' / 'hip.h'


@pytest.fixture(scope='module')
def hip():
    return Context(arch='gfx90a', backend='hip', fp_type=Datatype.F32)


def _merge(hip, lanes, into=True):
    writer = IRBuilder(Datatype.F32, context=hip)
    ftype = ScalarType(Datatype.F32)
    value = writer.declare(ftype, hint='v')
    target = writer.declare(ftype, hint='w') if into else None
    select = reorder.Select.of(lanes, 64)
    return select, exchange_codegen._merge(writer, target, value, select,
                                           ftype)


# -- the reason the refusal was wrong -------------------------------------- #

def test_the_runtime_transpose_already_reads_the_lane_id():
    """Which is what makes the ternary a mechanism the path has rather than
    one it introduces."""
    source = HEADER.read_text()
    # The definition, not the forward declaration that precedes it.
    body = source[source.rindex('void transpose4x4b32'):]
    body = body[:body.index('\n}')]
    assert '__lane_id()' in body
    assert body.count('__lane_id()') == 8, body.count('__lane_id()')


def test_the_fused_form_is_what_the_assembly_would_have_written():
    """`v_cndmask_b32_dpp` combines the select with the shuffle; the C++ path
    keeps them apart and says so."""
    source = HEADER.read_text()
    assert 'v_cndmask_b32_dpp' in source
    assert "doesn't combine cndmask and dpp" in source


# -- and the merge that follows from it ------------------------------------ #

def test_a_region_finer_than_a_bank_is_merged(hip):
    """One lane out of every four: the region a transpose's decomposition
    produces, and the one that was refused."""
    select, merged = _merge(hip, {0, 4, 8, 12})
    assert select.kind == 'cndmask'
    assert merged is not None


def test_the_mask_is_the_region(hip):
    """`laneMerge` takes the lanes as a bitmap, so the two are one statement
    rather than a mask derived beside the region it has to reproduce."""
    select, _ = _merge(hip, {0, 4, 8, 12})
    assert select.mask == 0b0001000100010001
    assert all(select.mask >> lane & 1 for lane in select.lanes)
    assert bin(select.mask).count('1') == len(select.lanes)


def test_a_maskable_region_still_takes_the_free_path(hip):
    """The ternary costs an instruction the masks do not, so it is the
    fallback and not a replacement."""
    select, merged = _merge(hip, set(range(16)))
    assert select.free and select.kind in ('row', 'bank')
    assert merged is not None


def test_mergeable_is_not_free(hip):
    """Conflating the two is what kept the assembled exchange out of reach: a
    `cndmask` costs an instruction of its own and is still emittable."""
    fine = reorder.Select.of({0, 4, 8, 12}, 64)
    coarse = reorder.Select.of(set(range(16)), 64)
    assert fine.mergeable and not fine.free
    assert coarse.mergeable and coarse.free


def test_the_first_region_writes_rather_than_combines(hip):
    """The masks of one group partition the wave, so there is nothing to
    combine with until the second."""
    _, merged = _merge(hip, {0, 4, 8, 12}, into=False)
    assert merged is not None
