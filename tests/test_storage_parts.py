# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""An operand stored decomposed, and what that costs in address arithmetic.

`Tensor.storage_parts` says how many scalars one element occupies, so that a
value whose stored form is a decomposition -- two TF32 halves, several limbs
of a wider format -- can be prepared once by whoever fills the buffer instead
of being rebuilt in every block and every batch element.

Two things are pinned here.  That the factor reaches everything sizing or
striding a buffer, in one place and by one multiplication; and that it reaches
nothing counting *elements*, because which value sits in which slot is a
question the storage form does not change.  Those two are one word apart at
every call site, which is why they are tested rather than assumed.

Host-only: no GPU, no toolchain, no code generation.
"""

from __future__ import annotations

import numpy as np
import pytest

from tensorforge.backend.symbol import DataView
from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.spp import MaskSPP
from tensorforge.common.matrix.tensor import Tensor


def _dense(shape=(8, 4), parts=1, bbox=None):
    t = Tensor(shape=list(shape), addressing=Addressing.NONE,
               bbox=bbox, datatype=Datatype.F32)
    t.storage_parts = parts
    return t


def _banded(parts=1):
    """A 4x4 with its diagonal stored and nothing else."""
    mask = np.eye(4, dtype=bool, order="F")
    t = Tensor(shape=[4, 4], addressing=Addressing.NONE,
               spp=MaskSPP(mask), datatype=Datatype.F32)
    t.storage_parts = parts
    return t


def test_the_trivial_convention_is_the_default():
    """Nothing states `storage_parts` today, so the default is what the whole
    corpus is generated under; if it were anything but one, every existing
    kernel would change address arithmetic on the way past this."""
    assert _dense().storage_parts == 1
    assert DataView(shape=[8, 4], permute=None).elem_parts == 1
    assert DataView(shape=[8, 4], permute=None).get_dim_strides() == [1, 8]


def test_a_decomposed_element_scales_every_stride_once():
    """The parts of one element are adjacent, which is the whole convention:
    indices stay in elements, the innermost stride becomes the element size,
    and each axis above it inherits the same factor rather than a second one."""
    view = DataView(shape=[8, 4], permute=None, elem_parts=2)
    assert view.get_dim_strides() == [2, 16]
    assert view.get_volume() == 8 * 4 * 2


def test_a_masked_axis_keeps_the_factor():
    """`mask` drops an axis from the address entirely -- the axes above it
    close the gap rather than stride over it -- and the factor belongs to the
    element rather than to any one axis, so it survives the drop."""
    view = DataView(shape=[8, 4, 2], permute=None, elem_parts=3)
    assert view.get_dim_strides(mask=[1]) == [3, 24]


def test_volume_counts_scalars_and_the_map_counts_elements():
    """The two are one word apart at the call sites that matter -- an
    allocation and a batch stride want scalars, `linear_index` and the pack
    table want elements -- and a tensor stored decomposed is where they stop
    being the same number."""
    dense = _dense(shape=(8, 4), parts=2)
    assert dense.storage_elements() == 32
    assert dense.storage_volume() == 64
    assert dense.storage_map() is None

    sparse = _banded(parts=2)
    assert sparse.storage_elements() == 4
    assert sparse.storage_volume() == 8
    # one entry per stored *value*, not per scalar: the map says which cell of
    # the bounding box a slot holds, and both halves of a value hold the same
    # cell
    assert len(sparse.storage_map()) == 4
    assert sparse.storage_map() == (0, 5, 10, 15)


def test_the_box_and_not_the_shape_is_what_gets_scaled():
    """A global tensor is stored over its bounding box.  Multiplying the full
    shape instead would reserve room for elements that are never written, and
    the error would be proportional to the factor rather than visible at
    parts one."""
    boxed = _dense(shape=(8, 4), parts=2,
                   bbox=BoundingBox([0, 0], [5, 3]))
    assert boxed.storage_elements() == 15
    assert boxed.storage_volume() == 30


def test_two_tensors_differing_only_in_storage_are_not_similar():
    """Shape, addressing and box agree and the buffers still do not: one is
    read at the other's stride.  `is_similar` is what treats two operands as
    one, so it has to see the difference."""
    assert not _dense(parts=1).is_similar(_dense(parts=2))
    assert _dense(parts=2).is_similar(_dense(parts=2))


def test_the_descriptor_stays_quiet_at_parts_one():
    """The descriptor string is read by the metainfo header and the
    reproduction tools.  A field every operand carries identically is one they
    all have to parse to learn nothing, so it appears only when it says
    something."""
    plain, split = _dense(parts=1), _dense(parts=2)
    plain.name = split.name = "A"
    assert not plain.gen_descr().endswith("/1")
    assert split.gen_descr().endswith("/2")


def test_a_decomposed_operand_is_still_contiguous_on_its_first_axis():
    """Adjacency is asked in elements.  Comparing the stride against the
    literal one would call this operand strided and cost it the wide load it
    is entitled to -- the failure would be a quiet loss of a load width, not a
    wrong answer, which is why it is pinned."""
    from tensorforge.backend.instructions.compute.multilinear import (
        _contiguous_first_axis)

    class _Sym:
        def __init__(self, view):
            self.data_view = view

    assert _contiguous_first_axis(_Sym(DataView(shape=[8, 4], permute=None,
                                                elem_parts=2)))
    # a transposed view is strided under any convention
    transposed = DataView(shape=[8, 4], permute=None, elem_parts=2)
    transposed.shape = [4, 8]
    assert _contiguous_first_axis(_Sym(transposed))


def test_a_storage_convention_below_one_is_refused():
    """Zero scalars per element is not a compressed operand, it is an address
    space that collapses onto one cell."""
    with pytest.raises(Exception, match="storage_parts"):
        _dense(parts=0)
