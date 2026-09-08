# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""A guard travels with the operation it guards.

yateto states, per operation, a conjunction of literals over rank-0 condition
tensors that decides whether the operation runs.  That guard is a field on the
descriptor rather than a descriptor wrapping another one, because everything
that walks a section -- `SectionPlan`, residency, temporaries -- asks each
descriptor for its geometry through `reads`, `writes` and `effective_boxes`
and deliberately does not know what kinds of descriptor there are.  A wrapper
would have to be unwrapped by each of those walks, and a walk that forgot
would read a guarded write as an unconditional one.

Nothing lowers a guard yet.  What is asserted here is the part that has to
hold before it can be lowered: the guard reaches the descriptor, the section
sees the tensors it reads, and a descriptor that cannot be honoured stops the
generator instead of being dropped.
"""

from __future__ import annotations

import pytest

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.context import Context
from tensorforge.common.exceptions import InternalError
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators.descriptions import (ElementwiseDescr,
                                                 GuardLiteral,
                                                 OperationDescription)
from tensorforge.common.operation import Operation
from tensorforge.generators.generator import Generator

DTYPE = Datatype.F32


def _mat(alias):
    return SubTensor(Tensor([16, 16], Addressing.STRIDED,
                            BoundingBox([0, 0], [16, 16]),
                            alias=alias, datatype=DTYPE))


def _flag(alias):
    return SubTensor(Tensor([1], Addressing.STRIDED, BoundingBox([0], [1]),
                            alias=alias, datatype=Datatype.BOOL))


def _sqrt(dest="A", src="B"):
    return ElementwiseDescr(Operation.SQRT, _mat(dest), [_mat(src)])


def _generate(descrs):
    ctx = Context(arch="sm_86", backend="cuda", fp_type=DTYPE)
    gen = Generator(descrs, ctx, attrs={})
    gen.generate()
    return gen


# ----------------------------------------------------------------------
# The field
# ----------------------------------------------------------------------

def test_an_operation_without_a_guard_always_runs():
    descr = _sqrt()
    assert descr.condition is None
    assert not descr.guarded()
    assert descr.condition_reads() == []


def test_an_empty_conjunction_is_also_always():
    """yateto sends `[]` for a guard it has already decided is true.

    Not the same as `None`, which is how it says the statement is dead, but
    the two agree on whether the operation runs.
    """
    descr = _sqrt()
    descr.condition = []
    assert not descr.guarded()


def test_a_guard_is_read_by_the_operation_it_guards():
    flag = _flag("c")
    descr = _sqrt()
    descr.condition = [GuardLiteral(flag, version=0, negated=False)]
    assert descr.guarded()
    assert descr.condition_reads() == [flag]
    # and it is not one of the operands: a builder resolves those, and the
    # guard is read to decide whether to run, not to compute with
    assert flag not in descr.reads()


def test_two_versions_of_one_tensor_are_two_values():
    """Without the version a guard could be simplified against a value it
    never had: yateto may write a condition tensor and write it again."""
    flag = _flag("c")
    first = GuardLiteral(flag, version=0, negated=False)
    second = GuardLiteral(flag, version=1, negated=False)
    negated = GuardLiteral(flag, version=0, negated=True)
    assert first.key() != second.key()
    assert first.key() != negated.key()
    assert first.key() == GuardLiteral(flag, version=0, negated=False).key()


# ----------------------------------------------------------------------
# What the generator does with one
# ----------------------------------------------------------------------

def test_a_guarded_operation_stops_the_generator():
    descr = _sqrt()
    descr.condition = [GuardLiteral(_flag("c"))]
    with pytest.raises(InternalError, match="guard"):
        _generate([descr])


def test_the_section_sees_the_tensors_a_guard_reads():
    """The guard's operands are not operands of the operation, but the
    section still has to stage them, so the plan has to count them."""
    from tensorforge.backend.section_plan import SectionPlan

    flag = _flag("c")
    descr = _sqrt()
    descr.condition = [GuardLiteral(flag)]

    class _NoSymbols:
        def get_symbol(self, tensor):
            return None

    plan = SectionPlan([descr], _NoSymbols())
    assert id(flag.tensor) in plan._read_union


# ----------------------------------------------------------------------
# What used to happen instead
# ----------------------------------------------------------------------

def test_a_descriptor_nobody_builds_stops_the_generator():
    """It used to fall out of the dispatch and be dropped, which turns a
    missing builder into a wrong kernel rather than an error."""

    class Unknown(OperationDescription):
        def matrix_list(self):
            return [_mat("A")]

        def get_num_threads(self, context):
            return 32, 32

        def __str__(self):
            return "an operation nothing builds"

    with pytest.raises(InternalError, match="no builder"):
        _generate([Unknown()])
