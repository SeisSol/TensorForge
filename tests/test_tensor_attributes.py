# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""What yateto states about a tensor's data, read as metadata.

yateto exports a tensor's attributes as `attrs` on its description, and only
where the tensor states something.  Every one of them permits an optimization
and asks for none -- `temporal: 'invariant'` says the data does not change over
the time loop, so whatever depends on it alone could be computed once, by a
prologue -- and this side reads them onto the tensor and builds exactly the
kernel it built without them, since nothing reads them yet.

A key this side does not read is skipped: being among `attrs` is what lists it
as safe to ignore.  Anything that changes what the numbers mean comes as a field
of its own with an interface version, and `read` refuses a version it does not
speak.  Offerings run the other way and follow the opposite rule; the hook
where a format for a buffer the caller fills would be offered stays empty.
"""

from __future__ import annotations

import copy
import json
import pathlib
import re

import pytest

import seissol_suite
from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.context import Context
from tensorforge.common.matrix.tensor import Tensor
from tensorforge.frontend.yateto import DescriptionReader, YatetoFrontend
from tensorforge.generators.generator import Generator

KERNELS = pathlib.Path(__file__).parent / "fixtures" / "kernels"

INVARIANT = {"temporal": "invariant"}


class _Arch:
    name = "sm_86"
    backend = "cuda"
    typename = "double"
    alignment = 64


def _recording(stem):
    blob = json.loads((KERNELS / f"{stem}.json").read_text())
    return copy.deepcopy(next(iter(blob["descriptions"].values())))


def _seissol(kernel):
    return seissol_suite.description("elastic-linearck", "elastic-linearck-o4-d", kernel)


def _claiming(description, names, attrs=INVARIANT):
    """The same description, with `attrs` on the tensors named."""
    claimed = copy.deepcopy(description)
    for tensor in claimed["tensors"]:
        if tensor["name"] in names:
            tensor["attrs"] = dict(attrs)
    return claimed


def _inputs(description):
    """What the caller hands the kernel: everything but temporaries."""
    return {t["name"] for t in description["tensors"] if not t["flags"]["temporary"]}


def _tensors(description):
    _, cache = DescriptionReader(_Arch(), {}).read(description)
    return cache


def _source(description, backend, arch):
    descrs, _ = DescriptionReader(_Arch(), {}).read(description)
    generator = Generator(descrs, Context(arch=arch, backend=backend,
                                          fp_type=Datatype.F64), attrs={})
    generator.generate()
    return (generator.get_base_name(), generator.get_header(),
            generator.get_launcher(), generator.get_kernel())


class TestReadingAttributes:
    def test_a_description_without_them_claims_nothing(self):
        for tensor in _tensors(_recording("contract_then_reduce")).values():
            assert tensor.attrs == {}
            assert tensor.time_invariant is False

    def test_a_claim_reaches_the_tensor_it_is_about(self):
        description = _seissol("gpu_derivative")
        stars = {"star(0)", "star(1)", "star(2)"}
        cache = _tensors(_claiming(description, stars))
        assert {name for name, t in cache.items() if t.time_invariant} == stars
        assert cache["star(0)"].attrs == INVARIANT
        assert cache["Q"].attrs == {}

    def test_an_attribute_this_side_does_not_read_is_skipped(self):
        # Among `attrs`, it permits; skipping it builds the kernel there was.
        description = _claiming(_recording("contract_then_reduce"), {"A"},
                                {"temporal": "invariant", "reuse": "high"})
        assert _tensors(description)["A"].attrs == INVARIANT

    def test_a_value_this_side_does_not_read_is_no_claim(self):
        description = _claiming(_recording("contract_then_reduce"), {"A"},
                                {"temporal": "piecewise"})
        tensor = _tensors(description)["A"]
        assert tensor.attrs == {}
        assert tensor.time_invariant is False

    def test_attrs_that_are_not_a_mapping_are_refused_by_tensor(self):
        description = _recording("contract_then_reduce")
        description["tensors"][0]["attrs"] = ["temporal"]
        name = description["tensors"][0]["name"]
        with pytest.raises(NotImplementedError, match=rf"tensor {re.escape(name)}: .*attrs"):
            _tensors(description)

    def test_a_field_that_changes_meaning_would_come_with_a_version(self):
        # What `attrs` may not carry -- a scale, a storage precision -- is a
        # field with a version of its own, and that version is refused here.
        description = _claiming(_recording("contract_then_reduce"), {"A"})
        description["version"] = YatetoFrontend.INTERFACE_VERSION + 1
        with pytest.raises(NotImplementedError, match="interface version"):
            _tensors(description)


#: Recorded SeisSol kernels and the per-element operands they read that setup
#: writes once: the star matrices and the two flux solvers.
SEISSOL_CLAIMS = [
    ("gpu_derivative", {"star(0)", "star(1)", "star(2)"}),
    ("gpu_localFlux[0]", {"AplusT"}),
    ("gpu_neighboringFlux[0]", {"AminusT"}),
]


@pytest.mark.parametrize("backend,arch", [("cuda", "sm_86"), ("hip", "gfx942")])
class TestTheKernelIsTheSame:
    """Nothing reads the claims yet, so the code is byte for byte the same."""

    @pytest.mark.parametrize("kernel,claimed", SEISSOL_CLAIMS,
                             ids=[kernel for kernel, _ in SEISSOL_CLAIMS])
    def test_with_the_seissol_operands_claimed(self, backend, arch, kernel, claimed):
        description = _seissol(kernel)
        assert claimed <= _inputs(description)
        assert (_source(_claiming(description, claimed), backend, arch)
                == _source(description, backend, arch))

    @pytest.mark.parametrize("stem", ["contract_then_reduce", "guards", "sparse_layouts"])
    def test_with_every_input_claimed(self, backend, arch, stem):
        description = _recording(stem)
        assert (_source(_claiming(description, _inputs(description)), backend, arch)
                == _source(description, backend, arch))


class _Emitter:
    def __init__(self, tensors):
        self._tensors = tensors

    def tensors(self):
        return self._tensors


class TestNoFormatIsOfferedYet:
    """The hook where a format for a buffer the caller fills would be offered.

    Empty: the caller has no way yet to say it takes one, and without that a
    kernel reads the format the tensor gives itself.  A time-invariant operand
    is exactly the kind such an offer would be about, so it is the one to
    check.
    """

    @staticmethod
    def _frontend(tensors):
        frontend = YatetoFrontend.__new__(YatetoFrontend)
        frontend._emitter = _Emitter(tensors)
        return frontend

    def test_a_time_invariant_operand_is_offered_nothing(self):
        star = Tensor(shape=[9, 9], addressing=Addressing.PTR_BASED,
                      datatype=Datatype.F64, alias="star(0)", attrs=INVARIANT)
        frontend = self._frontend({"star(0)": star})
        assert frontend.layout_offerings() == {}
        assert frontend._rearranged() is False
