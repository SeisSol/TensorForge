# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""A recorded kernel description, replayed here.

`tools/host/dump_descriptors.py` writes what yateto handed over next to what
was built from it.  The point of the description being data is that the first
half can be read back without yateto: this suite has no way to run it, so
until now the bridge could only be tested against descriptors written by hand,
which say what someone thought yateto sends rather than what it does.

`tests/fixtures/kernels/` holds captures as `dump_descriptors.py` writes
them.  Everything here is parametrised over that directory, so a capture
dropped in -- a real SeisSol kernel, say -- is covered without a test being
edited.  The one the specific assertions are written against is a
two-statement kernel: a contraction whose result a reduction then sums.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from tensorforge.common.basic_types import Datatype
from tensorforge.common.context import Context
from tensorforge.frontend.yateto import DescriptionReader, YatetoFrontend
from tensorforge.generators.descriptions import MultilinearDescr
from tensorforge.generators.generator import Generator

KERNELS = pathlib.Path(__file__).parent / "fixtures" / "kernels"


def _recordings():
    """Every capture in the fixture directory, kernel by kernel.

    Parametrised over the directory rather than over a list, so that a
    capture dropped in here -- a real SeisSol kernel, say -- is covered by
    everything below without a test being edited.
    """
    out = []
    for path in sorted(KERNELS.glob("*.json")):
        blob = json.loads(path.read_text())
        for name, description in blob["descriptions"].items():
            out.append(pytest.param(description, id=f"{path.stem}:{name}"))
    return out


RECORDINGS = _recordings()


class _Arch:
    name = "sm_86"
    backend = "cuda"
    typename = "double"
    alignment = 64


@pytest.fixture
def description():
    """The kernel the specific assertions below are written against."""
    match = [p.values[0] for p in RECORDINGS
             if p.id.startswith("contract_then_reduce:")]
    assert match, "the contract_then_reduce recording is missing"
    return match[0]


def test_there_is_at_least_one_recording():
    assert RECORDINGS, f"no captures in {KERNELS}"


@pytest.mark.parametrize("recording", RECORDINGS)
def test_a_recording_states_the_version_it_was_made_against(recording):
    """One made against another version would be replayed wrong rather than
    refused, so the version travels with it."""
    assert recording["version"] == YatetoFrontend.INTERFACE_VERSION


@pytest.mark.parametrize("recording", RECORDINGS)
def test_a_recording_is_data(recording):
    """Nothing in it needs Python to be understood -- it was read out of a
    file, and it goes back into one unchanged."""
    assert json.loads(json.dumps(recording)) == recording


@pytest.mark.parametrize("recording", RECORDINGS)
def test_every_reference_in_a_recording_names_a_tensor_it_declares(recording):
    known = {tensor["name"] for tensor in recording["tensors"]}
    for operation in recording["operations"]:
        refs = [operation["result"]] + operation["args"]
        refs += [operation["linear"]["alpha"]] if "linear" in operation else []
        for ref in refs:
            assert ref["name"] in known
            assert all(isinstance(index, str) for index in ref["indices"])


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


# ----------------------------------------------------------------------
# Where each recording stops
# ----------------------------------------------------------------------

#: The captures that do not build yet, and what stops each one. Kept here
#: rather than left implicit so that fixing one of these fails this test: the
#: entry then has to go, which is the point at which the gap is closed. Each
#: capture stops at the first kernel that cannot be built, so the kernels
#: before it in the same file are ones that can.
STOPS_AT = {
    "sparse_layouts": "LeadIndex",
    "index_permutations": "not in shared mem",
    "rings": "shape-changing operation is not elementwise",
    "elementwise": "accumulates onto its destination",
    "guards": "runs under a guard",
    "plasticity": "runs under a guard",
}


def _build(recording):
    """Read a description and build what it describes, as a codegen run does."""
    descrs, _ = DescriptionReader(_Arch(), {}).read(recording)
    Generator(descrs, Context(arch="sm_86", backend="cuda",
                              fp_type=Datatype.F64), attrs={}).generate()


@pytest.mark.parametrize("path", sorted(KERNELS.glob("*.json")),
                         ids=lambda p: p.stem)
def test_a_capture_builds_or_stops_where_it_is_recorded_as_stopping(path):
    blob = json.loads(path.read_text())
    recordings = list(blob["descriptions"].values())

    if path.stem not in STOPS_AT:
        for recording in recordings:
            _build(recording)
        return

    reasons = []
    for recording in recordings:
        try:
            _build(recording)
        except Exception as caught:      # noqa: BLE001 -- the reason is the point
            reasons.append(str(caught))
    assert reasons, (
        f"{path.stem} is recorded as stopping at {STOPS_AT[path.stem]!r}, "
        f"but every kernel in it builds -- remove the entry")
    assert any(STOPS_AT[path.stem] in reason for reason in reasons), reasons
