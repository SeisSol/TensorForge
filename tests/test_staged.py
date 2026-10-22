# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""A copy a build staged, asked directly.

An image is indexed in its own coordinates and covers only what was staged,
so `holds` is what stands between an operand of the same tensor and the wrong
elements from the wrong lanes (`MultilinearBuilder._resolve_reuse`), and
`region` what a destination compares its own box against
(`_target_image_fits`).
"""

from __future__ import annotations

from tensorforge.backend.instructions.builders.staged import StagedCopy
from tensorforge.backend.symbol import Symbol, SymbolType
from tensorforge.common.matrix.boundingbox import BoundingBox


def _sym(name, stype=SymbolType.Register):
    return Symbol(obj=object(), name=name, stype=stype)


def _box(lower, upper):
    return BoundingBox(list(lower), list(upper))


def test_an_image_holds_what_was_staged():
    copy = StagedCopy(_sym("r0"), _box([0, 0], [8, 4]), [0, 0])

    assert copy.holds(_box([0, 0], [8, 4]), [0, 0])
    assert copy.holds(_box([0, 0], [4, 2]), [0, 0])
    assert not copy.holds(_box([0, 0], [8, 8]), [0, 0])


def test_the_shift_rebases_the_question_into_image_coordinates():
    """Position `r` of the image holds tensor element `r + shift`.

    An image staged for `tensor[:, 4:8]` sits at its own origin, so an
    operand asking for that same slice -- box `[0,4)` at offset 4 -- is
    asking for image positions `[0,4)` and fits.  Comparing tensor
    coordinates against image coordinates instead would reject it.
    """
    copy = StagedCopy(_sym("r0"), _box([0, 0], [8, 4]), [0, 4])

    assert copy.holds(_box([0, 0], [8, 4]), [0, 4])
    assert not copy.holds(_box([0, 0], [8, 4]), [0, 0])


def test_an_unrecorded_range_cannot_answer():
    """`holds` says yes, and leaves what that means to the caller.

    The two callers disagree: resolving an operand refuses an unrecorded range
    only when the operand carries a slicing offset, while a destination looking
    for its own image requires a recorded range and treats its absence as a
    miss.  Deciding it here would be deciding it wrongly for one of them.
    """
    copy = StagedCopy(_sym("r0"))

    assert copy.covered is None
    assert copy.holds(_box([0, 0], [99, 99]), [0, 0])
    assert copy.region() is None


def test_the_region_is_stated_in_tensor_coordinates():
    """What a destination compares its own box against."""
    copy = StagedCopy(_sym("r0"), _box([0, 0], [8, 4]), [0, 4])

    assert copy.region() == ([0, 4], [8, 8])
