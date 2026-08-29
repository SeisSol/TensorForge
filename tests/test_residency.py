# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""The residency record, asked directly.

Its whole job is a distinction -- a copy of something memory still holds
against the only copy of something it does not -- and the distinction used to
be recomputed at each site from `image is home`, alongside a lookalike test on
`home.stype` that is not the same question and misreads a writeback into a
shared-memory temporary.  Stating the kind is what makes that a typo rather
than a plausible alternative, and these are the properties that says.

The geometry half matters for the same reason: an image is indexed in its own
coordinates and covers only what its producer staged, so `holds` is what stands
between a later consumer and the wrong elements from the wrong lanes.
"""

from __future__ import annotations

import pytest

from tensorforge.backend.residency import (Residency, ResidencyEntry,
                                           ResidencyKind)
from tensorforge.backend.symbol import Symbol, SymbolType
from tensorforge.common.matrix.boundingbox import BoundingBox


def _sym(name, stype=SymbolType.Register):
    return Symbol(obj=object(), name=name, stype=stype)


def _residency():
    """A record with no working emitter.

    Recording and the geometry queries need nothing from the context; only
    `flush` does, and the cases here stop short of it -- what a flush emits is
    a statement about the store instructions, which the snapshots already make
    over the whole corpus.
    """
    return Residency(context=None, shr_mem=None, num_threads=32)


def _box(lower, upper):
    return BoundingBox(list(lower), list(upper))


# ----------------------------------------------------------------------
# the distinction
# ----------------------------------------------------------------------

def test_a_preload_is_its_own_home():
    """Nothing has to be written back, because the value is already there."""
    res = _residency()
    image = _sym("r0")
    entry = res.record_preload("m0", image, _box([0, 0], [8, 8]), [0, 0])

    assert entry.is_preload
    assert entry.home is image


def test_a_writeback_names_where_the_value_has_to_go():
    res = _residency()
    regs, home = _sym("r0"), _sym("s1", SymbolType.SharedMem)
    entry = res.record_writeback("s1", regs, home,
                                 covered=_box([0, 0], [8, 8]), shift=[0, 0])

    assert not entry.is_preload
    assert entry.image is regs and entry.home is home


def test_a_writeback_into_shared_memory_is_still_a_writeback():
    """The case the `home.stype` lookalike gets wrong.

    A temporary's writeback has a shared-memory home, so a test asking whether
    the home is global answers "no" and concludes there is nothing to write
    back -- losing the value and the symbol's data view with it.
    """
    res = _residency()
    entry = res.record_writeback("s0", _sym("r0"),
                                 _sym("s0", SymbolType.SharedMem))

    assert entry.kind is ResidencyKind.WRITEBACK
    assert entry.home.stype is not SymbolType.Global


def test_dropping_hands_the_entry_back():
    """`drop` does not act on what it removes; the caller decides.

    Emitting a store needs the instruction stream, so the record cannot do it.
    Handing the entry back is what lets the caller emit one for a writeback and
    nothing for a preload.
    """
    res = _residency()
    res.record_writeback("m0", _sym("r0"), _sym("m0", SymbolType.Global))

    entry = res.drop("m0")
    assert entry is not None and not entry.is_preload
    assert "m0" not in res
    assert res.drop("m0") is None


# ----------------------------------------------------------------------
# the geometry
# ----------------------------------------------------------------------

def test_an_image_holds_what_its_producer_staged():
    res = _residency()
    entry = res.record_preload("m0", _sym("r0"), _box([0, 0], [8, 4]), [0, 0])

    assert entry.holds(_box([0, 0], [8, 4]), [0, 0])
    assert entry.holds(_box([0, 0], [4, 2]), [0, 0])
    assert not entry.holds(_box([0, 0], [8, 8]), [0, 0])


def test_the_shift_rebases_the_question_into_image_coordinates():
    """Position `r` of the image holds tensor element `r + shift`.

    An image staged for `tensor[:, 4:8]` sits at its own origin, so a consumer
    asking for that same slice -- box `[0,4)` at offset 4 -- is asking for
    image positions `[0,4)` and fits.  Comparing tensor coordinates against
    image coordinates instead would reject it.
    """
    res = _residency()
    entry = res.record_preload("m0", _sym("r0"), _box([0, 0], [8, 4]), [0, 4])

    assert entry.holds(_box([0, 0], [8, 4]), [0, 4])
    assert not entry.holds(_box([0, 0], [8, 4]), [0, 0])


def test_an_unrecorded_range_cannot_answer():
    """`holds` says yes, and leaves what that means to the caller.

    The two callers disagree: resolving an operand refuses an unrecorded range
    only when the operand carries a slicing offset, while a destination looking
    for its own image requires a recorded range and treats its absence as a
    miss.  Deciding it here would be deciding it wrongly for one of them.
    """
    entry = ResidencyEntry(kind=ResidencyKind.PRELOAD,
                           image=_sym("r0"), home=_sym("r0"))

    assert entry.covered is None
    assert entry.holds(_box([0, 0], [99, 99]), [0, 0])
    assert entry.region() is None


def test_the_region_is_stated_in_tensor_coordinates():
    """What a destination compares its own box against."""
    res = _residency()
    entry = res.record_preload("m0", _sym("r0"), _box([0, 0], [8, 4]), [0, 4])

    assert entry.region() == ([0, 4], [8, 8])


# ----------------------------------------------------------------------
# one entry per name
# ----------------------------------------------------------------------

def test_recording_again_supersedes_the_whole_entry():
    """Symbols and geometry move together, since they are one record.

    Three parallel dicts is exactly one update away from an image whose
    recorded coverage belongs to the operation before it.
    """
    res = _residency()
    res.record_preload("m0", _sym("r0"), _box([0, 0], [8, 4]), [0, 0])
    res.record_writeback("m0", _sym("r9"), _sym("m0", SymbolType.Global),
                         covered=_box([0, 0], [8, 8]), shift=[0, 2],
                         promise=_box([0, 0], [8, 8]))

    entry = res.get("m0")
    assert entry.image.name == "r9"
    assert not entry.is_preload
    assert entry.shift == [0, 2]
    assert entry.promise is not None


def test_iteration_tolerates_dropping_during_it():
    """The epilogue walks the entries while a flush may remove them."""
    res = _residency()
    for i in range(3):
        res.record_writeback(f"m{i}", _sym(f"r{i}"),
                             _sym(f"m{i}", SymbolType.Global))

    seen = []
    for name, _ in res.items():
        seen.append(name)
        res.drop(name)
    assert seen == ["m0", "m1", "m2"]
