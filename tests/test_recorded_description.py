# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""A recorded kernel description, replayed here.

`tools/host/dump_descriptors.py` writes what yateto handed over next to what
was built from it.  The point of the description being data is that the first
half can be read back without yateto: this suite has no way to run it, so
until now the bridge could only be tested against descriptors written by hand,
which say what someone thought yateto sends rather than what it does.

The fixture is a two-statement kernel -- a contraction whose result a
reduction then sums -- captured from yateto's own `matmul`-shaped example.
Regenerate it with `dump_descriptors.py`; the `descriptions` half of that file
is exactly what is stored here.
"""

from __future__ import annotations

import json

import pytest

from tensorforge.frontend.yateto import DescriptionReader, YatetoFrontend
from tensorforge.generators.descriptions import MultilinearDescr

from fixtures.yateto_kernel import DESCRIPTION


class _Arch:
    name = "sm_86"
    backend = "cuda"
    typename = "double"
    alignment = 64


@pytest.fixture
def description():
    """Read the way the tooling reads it: as data, off the wire."""
    return json.loads(json.dumps(DESCRIPTION))


def test_the_fixture_is_the_version_this_side_reads(description):
    """A recording made against another version would be replayed wrong
    rather than refused, so the version travels with it."""
    assert description["version"] == YatetoFrontend.INTERFACE_VERSION


def test_a_recorded_description_builds_the_same_descriptors(description):
    descrs, cache = DescriptionReader(_Arch(), {}).read(description)

    # yateto states a contraction and a reduction; both arrive as
    # multilinears, because a sum over one axis is one
    assert [op["type"] for op in description["operations"]] == [
        "multilinear", "reduction"]
    assert [d.__class__ for d in descrs] == [MultilinearDescr, MultilinearDescr]

    assert {"A", "B", "C", "v"} <= set(cache)


def test_the_contraction_keeps_its_axes(description):
    descrs, _ = DescriptionReader(_Arch(), {}).read(description)
    contraction = descrs[0]
    # C_ij = A_ik B_kj: `i` and `j` are axes of the result, `k` is contracted
    assert contraction.target == [[0, -1], [-1, 1]]


def test_the_reduction_contracts_the_axis_it_sums(description):
    descrs, _ = DescriptionReader(_Arch(), {}).read(description)
    reduction = descrs[1]
    # v_i = sum_j C_ij: `i` survives, `j` is summed away
    assert reduction.target[0] == [0, -1]
    assert not reduction.add


def test_a_slice_brings_its_shift(description):
    """Nothing in this fixture is a slice, and that has to be stated rather
    than assumed: an offset silently lost reads somewhere else entirely."""
    for operation in description["operations"]:
        for ref in [operation["result"]] + operation["args"]:
            assert ref["sliced"] is False
            assert ref["offset"] in (None, [0] * len(ref["offset"] or []))


def test_replaying_it_twice_gives_the_same_thing(description):
    """The reader holds no state between kernels."""
    first, _ = DescriptionReader(_Arch(), {}).read(description)
    second, _ = DescriptionReader(_Arch(), {}).read(description)
    assert [d.target for d in first] == [d.target for d in second]
    assert [d.permute for d in first] == [d.permute for d in second]
