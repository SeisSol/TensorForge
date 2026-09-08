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


# --- the loop the tables are for --------------------------------------------


def loop_writer():
    from tensorforge.backend.writer import Writer
    return Writer()


def written(instruction):
    writer = loop_writer()
    instruction.gen_code(writer)
    return writer.get_src()


class Marker:
    """A child that writes one line, so nesting is visible in the text."""

    def __init__(self, text):
        self.text = text

    def gen_code(self, writer):
        writer(self.text)


def test_the_header_is_a_counted_loop_over_the_members():
    from tensorforge.backend.instructions.ptr_manip import VariantLoop
    loop = VariantLoop(context(), 'face', 4, [Marker('body();')])
    text = written(loop)
    assert 'for (int face = 0; face < 4; ++face)' in text
    assert 'body();' in text


def test_the_body_sits_inside_the_header():
    from tensorforge.backend.instructions.ptr_manip import VariantLoop
    lines = written(VariantLoop(context(), 'face', 2,
                                [Marker('a();'), Marker('b();')])).splitlines()
    head = next(i for i, l in enumerate(lines) if 'for (int face' in l)
    body = [i for i, l in enumerate(lines) if 'a();' in l or 'b();' in l]
    assert all(i > head for i in body)
    assert len(body) == 2


def test_the_tables_are_declared_before_the_header_not_inside_it():
    """They do not change between iterations, so they are not rebuilt in one."""
    from tensorforge.backend.instructions.ptr_manip import VariantLoop
    table = DeclareOperandTable(context(), 'tbl',
                                [symbol(f'm{i}', Addressing.NONE)
                                 for i in (3, 5)], Addressing.NONE,
                                Datatype.F32)
    lines = written(VariantLoop(context(), 'face', 2, [Marker('body();')],
                                tables=[table])).splitlines()
    decl = next(i for i, l in enumerate(lines) if 'tbl[2]' in l)
    head = next(i for i, l in enumerate(lines) if 'for (int face' in l)
    assert decl < head


def test_a_loop_reports_the_arguments_its_tables_hold():
    from tensorforge.backend.instructions.ptr_manip import VariantLoop
    members = [symbol(f'm{i}', Addressing.NONE) for i in (3, 5, 7, 9)]
    table = DeclareOperandTable(context(), 'tbl', members, Addressing.NONE)
    loop = VariantLoop(context(), 'face', 4, [], tables=[table])
    assert loop.get_operands() == members


def test_the_region_is_replaceable_like_any_other():
    """So that a pass walking regions reaches this one without knowing it."""
    from tensorforge.backend.instructions.ptr_manip import VariantLoop
    loop = VariantLoop(context(), 'face', 2, [Marker('old();')])
    assert len(loop.regions()) == 1
    loop.replace_region(0, [Marker('new();')])
    assert 'new();' in written(loop) and 'old();' not in written(loop)
    with pytest.raises(Exception):
        loop.replace_region(1, [])


def test_a_loop_that_never_runs_is_refused():
    from tensorforge.backend.instructions.ptr_manip import VariantLoop
    from tensorforge.common.exceptions import GenerationError
    with pytest.raises(GenerationError):
        VariantLoop(context(), 'face', 0, [])
