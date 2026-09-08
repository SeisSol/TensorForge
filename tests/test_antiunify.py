# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Which descriptor lists are one list apart, and where.

The cases are written as families deliberately: a family of one, a family that
differs in exactly one operand, and families that differ in each of the things
that must block a merge.  The last group is the point of the file -- a merge
that goes ahead where sparsity or addressing differ produces a kernel that
compiles, runs and is wrong for every member but one.
"""

import numpy as np
import pytest

from tensorforge.analysis.antiunify import (Generalization, Mismatch,
                                            anti_unify, instantiate,
                                            operand_key, skeleton, substitute)
from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.helper import generate_tmp_matrix
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.spp import MaskSPP
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators.descriptions import GemmDescr

DTYPE = Datatype.F32


def tensor(alias, shape, addressing=Addressing.STRIDED, spp=None,
           datatype=DTYPE, alignment=0):
    return SubTensor(Tensor(list(shape), addressing,
                            BoundingBox([0] * len(shape), list(shape)),
                            alias=alias, spp=spp, datatype=datatype,
                            alignment=alignment))


def gemm(a, b, c):
    return GemmDescr(trans_a=False, trans_b=False, a=a, b=b, c=c)


def body(operator_alias, **kwargs):
    """``Q += operator @ I`` -- the shape of one flux contribution."""
    q = tensor('Q', [56, 9])
    op = tensor(operator_alias, [56, 56], **kwargs)
    i = tensor('I', [56, 9])
    return [gemm(op, i, q)]


# --- families that merge ----------------------------------------------------


def test_single_body_generalises_to_itself():
    result = anti_unify([body('rDivM0')])
    assert isinstance(result, Generalization)
    assert result.arity == 0


def test_identical_bodies_have_no_holes():
    result = anti_unify([body('rDivM0'), body('rDivM0')])
    assert isinstance(result, Generalization)
    assert result.arity == 0


def test_one_differing_operand_is_one_hole():
    result = anti_unify([body('rDivM0'), body('rDivM1')])
    assert isinstance(result, Generalization)
    assert result.arity == 1
    assert result.binding_names() == (('rDivM0',), ('rDivM1',))


def test_a_family_of_four():
    result = anti_unify([body(f'fPrT{j}') for j in range(4)])
    assert isinstance(result, Generalization)
    assert result.arity == 1
    assert result.binding_names() == tuple((f'fPrT{j}',) for j in range(4))


def test_two_independent_holes():
    def two(op_alias, rhs_alias):
        q = tensor('Q', [56, 9])
        return [gemm(tensor(op_alias, [56, 56]), tensor(rhs_alias, [56, 9]), q)]

    result = anti_unify([two('A0', 'B0'), two('A1', 'B1')])
    assert isinstance(result, Generalization)
    assert result.arity == 2
    assert result.binding_names() == (('A0', 'B0'), ('A1', 'B1'))


def test_a_temporary_is_not_a_hole():
    """Two chains whose temporaries are distinct objects still merge.

    A temporary is created by the body that uses it, so two bodies never share
    one; identifying it by name would make every chain differ from every other
    chain at a position that carries no information.
    """
    def chain(operator_alias):
        d = tensor('D', [56, 9])
        a = tensor(operator_alias, [56, 56])
        b = tensor('B', [56, 9])
        c = tensor('C', [9, 9])
        tmp = SubTensor(generate_tmp_matrix(b, c))
        return [gemm(b, c, tmp), gemm(a, tmp, d)]

    result = anti_unify([chain('A0'), chain('A1')])
    assert isinstance(result, Generalization)
    assert result.arity == 1
    assert result.binding_names() == (('A0',), ('A1',))


def test_the_hole_covers_every_slot_that_names_it():
    """One tensor used twice is one hole, not two."""
    def twice(alias):
        q = tensor('Q', [56, 9])
        a = tensor(alias, [56, 56])
        t = SubTensor(generate_tmp_matrix(a, tensor('I', [56, 9])))
        return [gemm(a, tensor('I', [56, 9]), t), gemm(a, t, q)]

    result = anti_unify([twice('A0'), twice('A1')])
    assert isinstance(result, Generalization)
    assert result.arity == 1
    assert len(result.holes[0]) == 2


# --- families that must not merge -------------------------------------------


def test_sparsity_blocks_the_merge_and_says_so():
    """The near miss a common pattern would remove, reported as such."""
    dense = np.ones((56, 56), dtype=bool)
    sparse = np.tril(np.ones((56, 56), dtype=bool))
    result = anti_unify([body('rT0', spp=MaskSPP(dense)),
                         body('rT1', spp=MaskSPP(sparse))])
    assert isinstance(result, Mismatch)
    assert result.where == 'operand.spp'


def test_addressing_blocks_the_merge():
    result = anti_unify([body('A0', addressing=Addressing.STRIDED),
                         body('A1', addressing=Addressing.NONE)])
    assert isinstance(result, Mismatch)
    assert result.where == 'operand.addressing'


def test_alignment_blocks_the_merge():
    result = anti_unify([body('A0', alignment=0), body('A1', alignment=32)])
    assert isinstance(result, Mismatch)
    assert result.where == 'operand.alignment'


def test_datatype_blocks_the_merge():
    result = anti_unify([body('A0', datatype=Datatype.F32),
                         body('A1', datatype=Datatype.F64)])
    assert isinstance(result, Mismatch)
    assert result.where == 'operand.datatype'


def test_shape_blocks_the_merge():
    q = tensor('Q', [56, 9])
    wide = [gemm(tensor('A1', [56, 56]), tensor('I', [56, 9]), q)]
    narrow = [gemm(tensor('A2', [56, 21]), tensor('I', [21, 9]), q)]
    result = anti_unify([wide, narrow])
    assert isinstance(result, Mismatch)
    assert result.where == 'operand.shape'


def test_repeated_operand_is_not_interchangeable_with_two():
    """``C += A A`` and ``C += A B`` agree at every key and still differ."""
    q = tensor('Q', [56, 9])
    a = tensor('A', [56, 56])
    same = [gemm(a, a, tensor('Q2', [56, 56]))]
    apart = [gemm(tensor('A', [56, 56]), tensor('B', [56, 56]),
                  tensor('Q2', [56, 56]))]
    result = anti_unify([same, apart])
    assert isinstance(result, Mismatch)
    assert result.where == 'sharing'
    assert q is not None  # the fixture above is deliberately unused here


def test_different_operation_count_is_reported_by_length():
    result = anti_unify([body('A0'), body('A0') + body('A0')])
    assert isinstance(result, Mismatch)
    assert result.where == 'length'


def test_accumulation_flag_is_not_substitutable():
    q = tensor('Q', [56, 9])
    plain = [GemmDescr(trans_a=False, trans_b=False,
                       a=tensor('A0', [56, 56]), b=tensor('I', [56, 9]), c=q,
                       alpha=1.0, beta=0.0)]
    adding = [GemmDescr(trans_a=False, trans_b=False,
                        a=tensor('A1', [56, 56]), b=tensor('I', [56, 9]),
                        c=tensor('Q', [56, 9]), alpha=1.0, beta=1.0)]
    result = anti_unify([plain, adding])
    assert isinstance(result, Mismatch)
    assert result.where == 'attrs'


# --- the pieces -------------------------------------------------------------


def test_operand_key_ignores_the_alias():
    assert operand_key(tensor('A', [8, 8])) == operand_key(tensor('B', [8, 8]))


def test_skeleton_partitions_by_identity():
    a = tensor('A', [56, 56])
    q = tensor('Q', [56, 9])
    skel, views = skeleton([gemm(a, tensor('I', [56, 9]), q),
                            gemm(a, tensor('I', [56, 9]), q)])
    groups = {tuple(g) for g in skel.groups()}
    # `Q`, `A` and `I` each appear in both operations
    assert len(views) == 6
    assert (0, 3) in groups
    assert all(len(g) == 2 for g in groups)


def test_mismatch_is_falsy_and_generalization_is_truthy():
    assert not anti_unify([body('A0'), body('A0', alignment=8)])
    assert anti_unify([body('A0'), body('A1')])


@pytest.mark.parametrize('members', [2, 3, 4, 8, 16])
def test_family_size_does_not_change_the_answer(members):
    result = anti_unify([body(f'fPrT{j}') for j in range(members)])
    assert isinstance(result, Generalization)
    assert result.arity == 1
    assert len(result.bindings) == members


# --- putting a binding back in ----------------------------------------------


def test_round_trip_returns_each_member():
    """Generalise a family, bind hole by hole, get the members back.

    The property that says the generalisation kept everything it had to: if
    binding reproduces every input, nothing that distinguished them was lost
    into the common part.
    """
    bodies = [body(f'fPrT{j}') for j in range(4)]
    g = anti_unify(bodies)
    for member, original in enumerate(bodies):
        assert skeleton(instantiate(g, member))[0] == skeleton(original)[0]


def test_round_trip_over_a_chain():
    def chain(operator_alias):
        d = tensor('D', [56, 9])
        a = tensor(operator_alias, [56, 56])
        b = tensor('B', [56, 9])
        c = tensor('C', [9, 9])
        tmp = SubTensor(generate_tmp_matrix(b, c))
        return [gemm(b, c, tmp), gemm(a, tmp, d)]

    bodies = [chain('A0'), chain('A1'), chain('A2')]
    g = anti_unify(bodies)
    for member, original in enumerate(bodies):
        assert skeleton(instantiate(g, member))[0] == skeleton(original)[0]


def test_binding_a_tensor_no_member_used():
    """A hole takes any interchangeable tensor, not only the ones seen."""
    g = anti_unify([body('fPrT0'), body('fPrT1')])
    fresh = tensor('fPrT7', [56, 56])
    built = substitute(g, [fresh])
    assert skeleton(built)[0] == skeleton(body('fPrT7'))[0]


def test_substitution_is_idempotent():
    bodies = [body(f'A{j}') for j in range(3)]
    g = anti_unify(bodies)
    again = anti_unify([instantiate(g, j) for j in range(3)])
    assert isinstance(again, Generalization)
    assert again.arity == g.arity
    assert again.binding_names() == g.binding_names()


def test_one_binding_reaches_every_slot_of_its_hole():
    def twice(alias):
        q = tensor('Q', [56, 9])
        a = tensor(alias, [56, 56])
        t = SubTensor(generate_tmp_matrix(a, tensor('I', [56, 9])))
        return [gemm(a, tensor('I', [56, 9]), t), gemm(a, t, q)]

    g = anti_unify([twice('A0'), twice('A1')])
    built = substitute(g, [tensor('A9', [56, 56])])
    aliases = [v.tensor.alias for d in built for _, v in
               [('op0', d.ops[0])]]
    assert aliases == ['A9', 'A9']


def test_wrong_number_of_bindings_is_refused():
    g = anti_unify([body('A0'), body('A1')])
    with pytest.raises(ValueError):
        substitute(g, [])
    with pytest.raises(ValueError):
        substitute(g, [tensor('A2', [56, 56]), tensor('A3', [56, 56])])


def test_a_body_with_no_holes_rebuilds_unchanged():
    g = anti_unify([body('A0'), body('A0')])
    assert skeleton(substitute(g, []))[0] == skeleton(body('A0'))[0]
