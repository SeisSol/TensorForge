# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Reaching an operand that changes between iterations.

Where a run's members are already kernel parameters -- which is what a frontend
that wrote the repetition out always leaves behind, having named every one of
them -- the table is built inside the kernel and nothing reaches the interface.

Two things are checked and they matter for different reasons.  The element type
has to follow the members' addressing, since a pointer-based argument is one
indirection deeper than a batch-invariant one and the wrong choice is a type
error rather than a wrong address.  And the address arithmetic has to be the
same expression whether the base came from an argument or from a table, because
that is what makes a table-fed operand indistinguishable to everything
downstream.
"""

import pytest

from tensorforge.backend.instructions.ptr_manip import (DeclareOperandTable,
                                                        GetElementPtr)
from tensorforge.backend.symbol import Symbol, SymbolType
from tensorforge.common.basic_types import Addressing, DataFlowDirection, Datatype
from tensorforge.common.context import Context
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import Tensor


def context():
    return Context(arch='sm_86', backend='cuda', fp_type=Datatype.F32)


def tensor(name, addressing, shape=(56, 56)):
    obj = Tensor(list(shape), addressing,
                 BoundingBox([0] * len(shape), list(shape)),
                 alias=name, datatype=Datatype.F32)
    obj.name = name
    obj.set_data_flow_direction(DataFlowDirection.SOURCE)
    return obj


def symbol(name, addressing, shape=(56, 56)):
    return Symbol(name=name, stype=SymbolType.Batch,
                  obj=tensor(name, addressing, shape))


def emitted(instruction):
    lines = []
    instruction.gen_ir(lines.append)
    return '\n'.join(lines)


# --- the declaration --------------------------------------------------------


def test_a_table_over_batch_invariant_members_is_one_star():
    members = [symbol(f'm{i}', Addressing.NONE) for i in (3, 5, 7, 9)]
    text = emitted(DeclareOperandTable(context(), 'tbl', members,
                                       Addressing.NONE, Datatype.F32))
    assert 'const float *const tbl[4] = {m3, m5, m7, m9};' in text


def test_a_table_over_pointer_based_members_is_one_deeper():
    """Each member is already an array per element, so the table is `**`."""
    members = [symbol(f'm{i}', Addressing.PTR_BASED, (9, 15))
               for i in (1, 4, 6, 8)]
    text = emitted(DeclareOperandTable(context(), 'tbl', members,
                                       Addressing.PTR_BASED, Datatype.F32))
    assert 'const float **const tbl[4] = {m1, m4, m6, m8};' in text


def test_the_table_reports_its_members_as_operands():
    members = [symbol(f'm{i}', Addressing.NONE) for i in (3, 5)]
    table = DeclareOperandTable(context(), 'tbl', members, Addressing.NONE)
    assert table.get_operands() == members
    assert len(table) == 2


def test_an_empty_table_is_refused():
    from tensorforge.common.exceptions import GenerationError
    with pytest.raises(GenerationError):
        DeclareOperandTable(context(), 'tbl', [], Addressing.NONE)


# --- reading through it -----------------------------------------------------


def _binding(src, table=None, variant=None):
    dest = Symbol(name='glb_x', stype=SymbolType.Global, obj=src.obj)
    return emitted(GetElementPtr(context(), src, dest,
                                 include_extra_offset=False,
                                 table=table, variant=variant))


def test_a_table_fed_operand_offsets_the_same_way_as_an_argument_fed_one():
    """Only the base changes; the arithmetic around it is one expression."""
    src = symbol('m3', Addressing.NONE)
    table = DeclareOperandTable(context(), 'tbl',
                                [symbol(f'm{i}', Addressing.NONE)
                                 for i in (3, 5, 7, 9)],
                                Addressing.NONE)
    plain = _binding(src)
    fed = _binding(src, table=table, variant='v')
    assert plain.replace('m3[', 'tbl[v][') == fed


def test_the_variant_index_appears_where_the_argument_name_was():
    src = symbol('m1', Addressing.PTR_BASED, (9, 15))
    table = DeclareOperandTable(context(), 'tbl',
                                [symbol(f'm{i}', Addressing.PTR_BASED, (9, 15))
                                 for i in (1, 4, 6, 8)],
                                Addressing.PTR_BASED)
    text = _binding(src, table=table, variant='face')
    assert 'tbl[face][' in text
    assert 'm1[' not in text


def test_without_a_table_nothing_changes():
    src = symbol('m3', Addressing.NONE)
    assert 'm3[' in _binding(src)
    assert 'tbl' not in _binding(src)
