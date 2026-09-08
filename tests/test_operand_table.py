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
                                                        GetElementPtr,
                                                        TableForm, VariantLoop)
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
                                       Addressing.NONE, Datatype.F32,
                                       form=TableForm.ARRAY))
    assert 'const float *const tbl[4] = {m3, m5, m7, m9};' in text


def test_a_table_over_pointer_based_members_is_one_deeper():
    """Each member is already an array per element, so the table is `**`."""
    members = [symbol(f'm{i}', Addressing.PTR_BASED, (9, 15))
               for i in (1, 4, 6, 8)]
    text = emitted(DeclareOperandTable(context(), 'tbl', members,
                                       Addressing.PTR_BASED, Datatype.F32,
                                       form=TableForm.ARRAY))
    assert 'const float **const tbl[4] = {m1, m4, m6, m8};' in text


def test_the_table_reports_its_members_as_operands():
    members = [symbol(f'm{i}', Addressing.NONE) for i in (3, 5)]
    table = DeclareOperandTable(context(), 'tbl', members, Addressing.NONE,
                                form=TableForm.ARRAY)
    assert table.get_operands() == members
    assert len(table) == 2


def test_an_empty_table_is_refused():
    from tensorforge.common.exceptions import GenerationError
    with pytest.raises(GenerationError):
        DeclareOperandTable(context(), 'tbl', [], Addressing.NONE,
                            form=TableForm.ARRAY)


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
                                Addressing.NONE, form=TableForm.ARRAY)
    plain = _binding(src)
    fed = _binding(src, table=table, variant='v')
    assert plain.replace('m3[', 'tbl[v][') == fed


def test_the_variant_index_appears_where_the_argument_name_was():
    src = symbol('m1', Addressing.PTR_BASED, (9, 15))
    table = DeclareOperandTable(context(), 'tbl',
                                [symbol(f'm{i}', Addressing.PTR_BASED, (9, 15))
                                 for i in (1, 4, 6, 8)],
                                Addressing.PTR_BASED, form=TableForm.ARRAY)
    text = _binding(src, table=table, variant='face')
    assert 'tbl[face][' in text
    assert 'm1[' not in text


def test_without_a_table_nothing_changes():
    src = symbol('m3', Addressing.NONE)
    assert 'm3[' in _binding(src)
    assert 'tbl' not in _binding(src)


# --- the loop the tables are for --------------------------------------------


def select_table(names=(3, 5, 7, 9), addressing=Addressing.NONE,
                 variant='face', shape=(56, 56)):
    return DeclareOperandTable(
        context(), 'tbl', [symbol(f'm{i}', addressing, shape) for i in names],
        addressing, Datatype.F32, form=TableForm.SELECT, variant=variant)


# --- the form that touches no memory ----------------------------------------


def test_the_default_form_is_a_chain_and_not_an_array():
    """An array with a dynamic index cannot leave its allocation.

    It lands in the per-thread space -- `.local` on NVIDIA, scratch on AMD --
    where every thread in the block builds and holds its own copy of one set of
    pointers, and where on AMD the allocation alone can cost occupancy.  The
    counter is uniform, so a chain of selects is scalar on both vendors and
    stores nothing.
    """
    text = emitted(select_table())
    assert '[4]' not in text
    assert text.count('?') == 3
    assert 'const float *const tbl = (face == 0) ? m3 : ' in text


def test_the_chain_ends_on_the_last_member_without_a_test():
    text = emitted(select_table(names=(3, 5)))
    assert text.count('?') == 1
    assert text.rstrip().endswith(': m5;')


def test_a_chain_of_one_is_just_the_member():
    text = emitted(select_table(names=(3,)))
    assert '?' not in text
    assert 'tbl = m3;' in text


def test_a_chain_is_read_by_name_and_an_array_by_index():
    chain, array = select_table(), DeclareOperandTable(
        context(), 'tbl', [symbol(f'm{i}', Addressing.NONE) for i in (3, 5)],
        Addressing.NONE, form=TableForm.ARRAY)
    assert chain.access('face') == 'tbl'
    assert array.access('face') == 'tbl[face]'


def test_a_chain_without_a_counter_is_refused():
    from tensorforge.common.exceptions import GenerationError
    with pytest.raises(GenerationError):
        DeclareOperandTable(context(), 'tbl',
                            [symbol('m3', Addressing.NONE)], Addressing.NONE,
                            form=TableForm.SELECT)


def test_a_chain_is_emitted_inside_the_loop_and_an_array_before_it():
    """A chain is the counter's value; an array does not depend on it."""
    lines = written(VariantLoop(context(), 'face', 4, [Marker('body();')],
                                tables=[select_table()])).splitlines()
    head = next(i for i, l in enumerate(lines) if 'for (int face' in l)
    decl = next(i for i, l in enumerate(lines) if 'tbl =' in l)
    assert decl > head


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
                                Datatype.F32, form=TableForm.ARRAY)
    lines = written(VariantLoop(context(), 'face', 2, [Marker('body();')],
                                tables=[table])).splitlines()
    decl = next(i for i, l in enumerate(lines) if 'tbl[2]' in l)
    head = next(i for i, l in enumerate(lines) if 'for (int face' in l)
    assert decl < head


def test_a_loop_reports_the_arguments_its_tables_hold():
    from tensorforge.backend.instructions.ptr_manip import VariantLoop
    members = [symbol(f'm{i}', Addressing.NONE) for i in (3, 5, 7, 9)]
    table = DeclareOperandTable(context(), 'tbl', members, Addressing.NONE,
                                form=TableForm.ARRAY)
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


# --- the form the caller fills ----------------------------------------------


def param_table(names=(3, 5, 7, 9), addressing=Addressing.NONE, backend='cuda'):
    ctx = Context(arch='sm_86' if backend == 'cuda' else 'gfx90a',
                  backend=backend, fp_type=Datatype.F32)
    return DeclareOperandTable(
        ctx, 'tbl', [symbol(f'm{i}', addressing) for i in names],
        addressing, Datatype.F32, form=TableForm.PARAM)


def test_a_parameter_table_emits_nothing_in_the_body():
    """The caller filled it; there is nothing for the kernel to build."""
    assert emitted(param_table()) == ''


def test_cuda_annotates_the_parameter_and_hip_does_not():
    """What keeps a by-value parameter out of per-thread memory is the
    backend's business, and on most of them nothing is needed."""
    assert '__grid_constant__' in param_table(backend='cuda').parameter()
    assert '__grid_constant__' not in param_table(backend='hip').parameter()


def test_the_parameter_is_a_struct_and_not_an_array():
    """An array parameter decays to a pointer, and there is then no by-value
    parameter left for the annotation to keep out of per-thread memory."""
    assert param_table().parameter().endswith('const tbl_t tbl')
    assert '[' not in param_table().parameter()


def test_the_struct_carries_the_length_and_the_indirection():
    assert 'p[4]' in param_table().struct_definition()
    assert '**' in param_table(addressing=Addressing.PTR_BASED).struct_definition()


def test_the_caller_builds_it_by_value():
    assert param_table().argument() == \
        'const tbl_t tbl = {{m3, m5, m7, m9}};'


def test_a_parameter_table_is_read_through_its_member():
    assert param_table().access('face') == 'tbl.p[face]'


def test_only_a_parameter_table_has_a_parameter():
    from tensorforge.common.exceptions import GenerationError
    with pytest.raises(GenerationError):
        select_table().parameter()
    with pytest.raises(GenerationError):
        select_table().struct_definition()
    with pytest.raises(GenerationError):
        select_table().argument()


def test_a_parameter_table_is_not_emitted_inside_the_loop():
    from tensorforge.backend.instructions.ptr_manip import VariantLoop
    table = param_table()
    assert table.loop_invariant()
    text = written(VariantLoop(context(), 'face', 4, [Marker('body();')],
                               tables=[table]))
    assert 'tbl' not in text


# --- the signature ----------------------------------------------------------


def _generator_with_table():
    from tensorforge.generators.descriptions import GemmDescr
    from tensorforge.generators.generator import Generator

    pool = {}

    def view(alias, shape):
        if alias not in pool:
            pool[alias] = Tensor(
                list(shape),
                Addressing.NONE if alias.startswith('K') else Addressing.STRIDED,
                BoundingBox([0] * len(shape), list(shape)), alias=alias,
                datatype=Datatype.F32)
        from tensorforge.common.matrix.tensor import SubTensor
        return SubTensor(pool[alias])

    descrs = [GemmDescr(trans_a=False, trans_b=False, a=view(f'K{k}', [9, 9]),
                        b=view(f'i{k}', [9, 4]), c=view(f'o{k}', [9, 4]))
              for k in range(4)]
    ctx = context()
    gen = Generator(descrs, ctx)
    gen.register()
    members = [s for s in gen._scopes.get_global_scope().values()
               if s.obj.alias and s.obj.alias.startswith('K')]
    table = DeclareOperandTable(ctx, 'faceTable', members, Addressing.NONE,
                                Datatype.F32, form=TableForm.PARAM)
    gen.register_param_table(table)
    gen.generate()
    return gen, table, members


def test_the_kernel_takes_the_table_and_not_its_members():
    gen, table, members = _generator_with_table()
    signature = next(l for l in gen.get_kernel().splitlines()
                     if 'kernel_kernel' in l)
    assert '__grid_constant__ const faceTable_t faceTable' in signature
    for member in members:
        assert f' {member.name},' not in signature


def test_the_launcher_still_takes_the_members_one_by_one():
    """The substitution lives between two pieces of generated code.

    Which is the only reason it can be made without touching the caller: the
    launcher's signature is the interface, and it does not move.
    """
    gen, _, members = _generator_with_table()
    proto = gen._generate_launcher_proto(with_defaults=False)
    assert 'faceTable' not in proto
    for member in members:
        assert member.name in proto


def test_the_launcher_assembles_the_value_before_the_launch():
    gen, _, _ = _generator_with_table()
    lines = gen.get_launcher().splitlines()
    build = next(i for i, l in enumerate(lines) if 'faceTable =' in l)
    # The name also appears in the occupancy query and the attribute call, so
    # the launch is found by its argument list rather than by the name.
    call = next(i for i, l in enumerate(lines) if '<<<' in l)
    assert build < call
    assert 'faceTable' in lines[call]


def test_the_struct_type_is_reported_for_whoever_writes_the_file():
    gen, _, _ = _generator_with_table()
    assert gen.param_table_types() == [
        'struct faceTable_t { const float *const p[4]; };']


def test_a_generator_without_a_table_emits_what_it_did_before():
    from tensorforge.generators.generator import Generator
    gen = Generator(_contributions_for_signature(), context())
    gen.generate()
    assert gen.param_table_types() == []
    assert 'faceTable' not in gen.get_kernel()


def _contributions_for_signature():
    from tensorforge.common.matrix.tensor import SubTensor
    from tensorforge.generators.descriptions import GemmDescr

    def view(alias, shape):
        return SubTensor(Tensor(list(shape), Addressing.STRIDED,
                                BoundingBox([0] * len(shape), list(shape)),
                                alias=alias, datatype=Datatype.F32))
    return [GemmDescr(trans_a=False, trans_b=False, a=view('A', [9, 9]),
                      b=view('i', [9, 4]), c=view('o', [9, 4]))]
