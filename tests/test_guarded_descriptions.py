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

A guard lowers to a `GuardedRegion`: the operations under one conjunction
become one region, the condition is read once where it opens, and `verify`
recurses through the region as it does through a loop's body.  Asserted here
is that the guard reaches the descriptor, that the section sees the tensors
it reads, that neighbours under one guard share a region, and that a region
is only mult-uniform -- a block-wide barrier inside one deadlocks.
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

def _regions(gen):
    from tensorforge.backend.instructions.control.conditional import \
        GuardedRegion
    return [instr
            for section in gen._sections
            for instr in section.ir
            if isinstance(instr, GuardedRegion)]


def test_a_guarded_operation_becomes_a_region():
    descr = _sqrt()
    descr.condition = [GuardLiteral(_flag("c"))]
    regions, = _regions(_generate([descr]))
    assert len(regions.region()) > 0


def test_neighbours_under_one_guard_share_a_region():
    """The condition is read once, not once per operation."""
    flag = _flag("c")
    first, second = _sqrt(dest="A"), _sqrt(dest="C")
    first.condition = [GuardLiteral(flag)]
    second.condition = [GuardLiteral(flag)]
    assert len(_regions(_generate([first, second]))) == 1


def test_a_different_guard_is_a_different_region():
    first, second = _sqrt(dest="A"), _sqrt(dest="C")
    first.condition = [GuardLiteral(_flag("c"))]
    second.condition = [GuardLiteral(_flag("d"))]
    assert len(_regions(_generate([first, second]))) == 2


def test_an_unguarded_neighbour_stays_out_of_the_region():
    guarded, plain = _sqrt(dest="A"), _sqrt(dest="C")
    guarded.condition = [GuardLiteral(_flag("c"))]
    regions = _regions(_generate([guarded, plain]))
    assert len(regions) == 1
    assert len(regions[0].region()) > 0


def test_a_region_is_only_mult_uniform():
    """The condition is addressed per batch element, so two elements in one
    block may decide differently.  A block-wide barrier inside the region is
    then reached by some threads and not others, which deadlocks -- `verify`
    tightens its limit through `uniform_scope` on the way in."""
    from tensorforge.backend.instructions.abstract_instruction import \
        BarrierScope

    descr = _sqrt()
    descr.condition = [GuardLiteral(_flag("c"))]
    region, = _regions(_generate([descr]))
    assert region.uniform_scope() is BarrierScope.SIMD


def test_a_guard_stops_the_plan_deferring_its_operand():
    """The condition is not an operand, so no builder resolves it through the
    residency -- the region loads the symbol.  A value still in a register has
    no symbol to load, so the tensor a guard reads has to reach memory."""
    from tensorforge.backend.section_plan import SectionPlan

    flag = _flag("c")
    descr = _sqrt()
    descr.condition = [GuardLiteral(flag)]

    class _NoSymbols:
        def get_symbol(self, tensor):
            return None

    plan = SectionPlan([descr], _NoSymbols())
    assert plan.written_in_slices(flag.tensor)


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


# ----------------------------------------------------------------------
# The kernel description is data
# ----------------------------------------------------------------------

def _tensor(name, shape):
    return dict(name=name, addressing="n&+o&", datatype="f64",
                storage=dict(shape=list(shape), type="bbox",
                             start=[0] * len(shape), sizes=list(shape)),
                values=None, alignment=64,
                flags=dict(temporary=False, constant=False))


def _ref(name, indices, shape):
    return dict(name=name, indices=list(indices),
                bbox=[[0] * len(shape), list(shape)],
                offset=[0] * len(shape), sliced=False)


def _matmul_description():
    """``C_ij = A_ik B_kj``, spelled the way yateto sends it."""
    return dict(
        version=6,
        tensors=[_tensor(n, [8, 8]) for n in ("A", "B", "C")] + [
            dict(name="_scalar0", addressing="", datatype="f64",
                 storage=dict(shape=[], type="full"), alignment=0,
                 values=dict(kind="entries", data=[[[], 1.0]]),
                 flags=dict(temporary=False, constant=True))],
        operations=[dict(
            type="multilinear",
            result=_ref("C", "ij", [8, 8]),
            args=[_ref("A", "ik", [8, 8]), _ref("B", "kj", [8, 8])],
            condition=[],
            permute=[[0, 1], [0, 1]],
            target=[[0, -1], [-1, 1]],
            linear=dict(alpha=dict(name="_scalar0", indices=[], bbox=None,
                                   offset=None, sliced=False),
                        add=False))])


def test_a_kernel_description_survives_being_written_out_and_read_back():
    """Nothing in it needs Python to be understood.

    The host-side tooling records these and works off the recording rather
    than running yateto again, so a field only Python can carry is a field
    that tooling cannot.
    """
    import json

    from tensorforge.frontend.yateto import DescriptionReader, YatetoFrontend

    description = _matmul_description()
    description["version"] = YatetoFrontend.INTERFACE_VERSION
    replayed = json.loads(json.dumps(description))
    assert replayed == description

    descrs, cache = DescriptionReader(_ARCH, {}).read(replayed)
    assert len(descrs) == 1
    assert {name for name in cache} == {"A", "B", "C", "_scalar0"}


def test_a_description_from_another_interface_version_is_refused():
    from tensorforge.frontend.yateto import DescriptionReader, YatetoFrontend

    description = _matmul_description()
    description["version"] = YatetoFrontend.INTERFACE_VERSION - 1
    with pytest.raises(NotImplementedError, match="interface version"):
        DescriptionReader(_ARCH, {}).read(description)


class _Arch:
    name = "sm_86"
    backend = "cuda"
    typename = "double"
    alignment = 64

    def alignedLower(self, index):
        return index

    def alignedUpper(self, index):
        return index


_ARCH = _Arch()
