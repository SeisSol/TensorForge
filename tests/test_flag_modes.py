# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""The three shapes of the per-element flag mask.

The mask costs a parameter per section and a guard around the loop body,
and a kernel whose caller never skips an element pays for both.  Which
shape a kernel gets follows from the attributes its frontend passed at
construction, and the three cases are not symmetric:

* no attributes at all is a frontend that predates the attribute channel,
  and it has to keep getting the mask it always got -- otherwise updating
  TensorForge alone silently drops masking from a caller that relies on it;
* attributes without the mask is a frontend that had the chance to ask and
  did not, so naming ``flags`` at the call site should not compile;
* attributes with the mask is a promise to supply one, so the parameter has
  no default and the guard has nothing to check against null.

Snapshots record the whole generated text for two of these
(``cases/flags/``).  What is asserted here is the contract itself, in a
form that says which property broke when it breaks.
"""

from __future__ import annotations

import pytest

from tensorforge.common.basic_types import (Addressing, Datatype, FlagMode,
                                            GeneralLexicon)
from tensorforge.common.context import Context
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators.descriptions import GemmDescr
from tensorforge.generators.generator import Generator

DTYPE = Datatype.F32


def _descrs():
    def mat(alias):
        return SubTensor(Tensor([16, 16], Addressing.STRIDED,
                                BoundingBox([0, 0], [16, 16]),
                                alias=alias, datatype=DTYPE))
    return [GemmDescr(trans_a=False, trans_b=False,
                      a=mat("A"), b=mat("B"), c=mat("C"),
                      alpha=1.0, beta=0.0)]


def _generate(attrs):
    ctx = Context(arch="sm_86", backend="cuda", fp_type=DTYPE)
    gen = Generator(_descrs(), ctx, attrs=attrs)
    gen.generate()
    return gen


# ----------------------------------------------------------------------
# Attributes -> mode
# ----------------------------------------------------------------------

@pytest.mark.parametrize("attrs,expected", [
    (None, FlagMode.OPTIONAL),
    ({}, FlagMode.ABSENT),
    ({"flags": False}, FlagMode.ABSENT),
    ({"flags": True}, FlagMode.REQUIRED),
    ({"something_else": True}, FlagMode.ABSENT),
])
def test_mode_follows_the_attributes(attrs, expected):
    assert FlagMode.from_attrs(attrs) is expected
    assert _generate(attrs).flag_mode() is expected


def test_the_yateto_frontend_without_attributes_asks_for_nothing():
    """How a yateto that predates the attribute channel reaches this.

    It constructs the exporter with the architecture alone, so the default
    has to be the mode that keeps its kernels working: a mask on every
    kernel, checked against null.
    """
    from tensorforge.frontend.yateto import YatetoFrontend

    frontend = YatetoFrontend(object())          # positional, as yateto calls it
    assert FlagMode.from_attrs(frontend.generator._attrs) is FlagMode.OPTIONAL


def test_absent_is_not_the_same_as_no_attributes():
    """The distinction the backward compatibility rests on.

    An empty dictionary and ``None`` are both falsy, so anything that tests
    them with ``if attrs:`` collapses the two -- and the collapse is silent
    in the direction that matters, since a frontend without attributes would
    then generate kernels with no mask and the calls that pass one would
    stop compiling.
    """
    assert FlagMode.from_attrs({}) is not FlagMode.from_attrs(None)


# ----------------------------------------------------------------------
# What each mode emits
# ----------------------------------------------------------------------

def test_no_attributes_keeps_the_nullable_mask():
    gen = _generate(None)
    assert "unsigned* flags0 = nullptr" in gen.get_header()
    assert "flags0 == nullptr ? true" in gen.get_kernel()


def test_required_mask_has_no_default_and_no_null_check():
    gen = _generate({"flags": True})
    header = gen.get_header()
    assert "unsigned* flags0" in header
    assert "flags0 = nullptr" not in header
    kernel = gen.get_kernel()
    assert "const bool allowed = static_cast<bool>(flags0[batchId0]);" in kernel
    assert "flags0 == nullptr" not in kernel


def test_absent_mask_leaves_no_trace():
    gen = _generate({})
    for surface in (gen.get_header(), gen.get_launcher(), gen.get_kernel()):
        assert GeneralLexicon.FLAGS_NAME not in surface
    assert "allowed" not in gen.get_kernel()


def test_element_count_survives_the_mask_going_away():
    """The two are adjacent in the signature and only one of them is optional."""
    for attrs in (None, {}, {"flags": True}):
        assert "size_t numElements0" in _generate(attrs).get_header()


# ----------------------------------------------------------------------
# Call sites
# ----------------------------------------------------------------------

@pytest.mark.parametrize("attrs,expected", [
    (None, True),
    ({"flags": True}, True),
    ({}, False),
])
def test_call_site_passes_the_mask_exactly_when_the_signature_takes_one(
        attrs, expected):
    gen = _generate(attrs)
    call = gen.generate_call_site(mat_name_map={}, offset_name_map={})
    assert ("flags" in call) is expected
    assert "numElements" in call


# ----------------------------------------------------------------------
# Kernel identity
# ----------------------------------------------------------------------

def test_the_three_modes_are_three_kernels():
    """Two of them share a parameter list, so names must separate them.

    ``REQUIRED`` and ``OPTIONAL`` differ only inside the body.  A name
    derived from the parameters alone would be the same for both, and the
    routine cache keys on the name: one kernel would be emitted and the
    other silently dropped, with the caller of the dropped one linking
    against a body that checks a pointer it was promised is never null --
    or worse, the other way round.
    """
    names = {mode: _generate(attrs).get_base_name()
             for mode, attrs in (("optional", None),
                                 ("required", {"flags": True}),
                                 ("absent", {}))}
    assert len(set(names.values())) == 3, names
