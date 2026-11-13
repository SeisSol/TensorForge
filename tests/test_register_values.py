# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Register blocks as values (`pir.values`, `Options.register_values`).

A block is what it holds: a store defines it, a load is it.  Pinned here:
what a whole, a narrower and a predicated access become; what a branch and a
loop that stay hand out; what keeps its buffer; how both emitters spell the
parts and the constants a vector holds -- including the store of a zero
column the explicit vector's emitter wrote as `0.0f.copy_to(...)` without
any of this, which is checked with the option off -- and that off, nothing
changes.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import re
import warnings
from pathlib import Path

from tensorforge.backend.pir import (BOOL, INDEX, SIZE, IRBuilder, MemSpace,
                                     Op, emit, optimize, pipeline, verify)
from tensorforge.backend.pir.blocks import RegisterBlocks
from tensorforge.backend.pir.core import (LaneAxis, RegisterLayout,
                                          ScalarType, walk_stmts)
from tensorforge.backend.pir.emit_esimd import EsimdEmitter
from tensorforge.backend.writer import Writer
from tensorforge.common.basic_types import Datatype
from tensorforge.common.context import Context, Options
from tensorforge.generators.generator import Generator
from harness.placement import placed

F32 = ScalarType(Datatype.F32)
LANES32 = RegisterLayout((LaneAxis(32),))
LANES24 = RegisterLayout((LaneAxis(24),))
LANES16 = RegisterLayout((LaneAxis(16),))

CASES = Path(__file__).parent / 'cases'


def _builder():
    return IRBuilder(fptype=Datatype.F32, arena='shrMem')


def _ops(body, op):
    return [s for s in walk_stmts(body) if s.op == op]


def _registers(body):
    """The register buffers left, by name."""
    return {s.attr('extern') or str(s.target[0]) for s in _ops(body, Op.ALLOC)
            if s.target[0].type.space is MemSpace.REGISTER}


def _values(b, explicit_simd=True, **kw):
    return optimize(b.finish(), explicit_simd=explicit_simd,
                    register_values=True, scratch=b.scratch, **kw)


def _esimd(body) -> str:
    writer = Writer()
    ctx = Context(arch='pvc', backend='esimd', fp_type=Datatype.F32)
    EsimdEmitter(writer=writer, context=ctx, strict=False).run(placed(body))
    return writer.get_src()


def _spmd(body, arch='sm_86', backend='cuda') -> str:
    writer = Writer()
    emit(body, writer, Context(arch=arch, backend=backend,
                               fp_type=Datatype.F32))
    return writer.get_src()


# --------------------------------------------------------------------------
# straight-line
# --------------------------------------------------------------------------

def test_a_load_is_what_the_store_before_it_stored():
    """The column a load reads is the value the store wrote, and a column
    nothing wrote is the zeros the image started from."""
    b = _builder()
    img = b.alloc(Datatype.F32, (64,), MemSpace.REGISTER, extern='r1',
                  init='{}')
    glb = b.alloc(Datatype.F32, (64,), MemSpace.GLOBAL, hint='g')
    out = b.alloc(Datatype.F32, (64,), MemSpace.GLOBAL, hint='o')
    a = b.load(glb, 0, hint='a', layout=LANES32)
    b.store(img, a, 0)
    b.store(out, b.load(img, 0, hint='x', layout=LANES32), 0)
    b.store(out, b.load(img, 32, hint='y', layout=LANES32), 32)
    body = _values(b)
    assert _registers(body) == set()
    stores = [s for s in _ops(body, Op.STORE)]
    assert stores[0].args[1] is a
    zero = stores[1].args[1]
    const = next(s for s in _ops(body, Op.CONST) if s.target[0] is zero)
    assert const.attr('value') == 0.0 and zero.distributed
    src = _esimd(body)
    assert 'r1' not in src, src
    assert 'simd<float, 32>(0.0f).copy_to(' in src, src
    assert verify(body, strict=False) == []


def test_a_narrower_access_is_a_part_inserted_or_sliced():
    """A full-lane tail: 24 of a column's 32 lanes written, and read back as
    24 and as the whole column -- an insert into the zeros and a slice."""
    b = _builder()
    img = b.alloc(Datatype.F32, (64,), MemSpace.REGISTER, extern='r1',
                  init='{}')
    glb = b.alloc(Datatype.F32, (64,), MemSpace.GLOBAL, hint='g')
    out = b.alloc(Datatype.F32, (64,), MemSpace.GLOBAL, hint='o')
    b.store(img, b.load(glb, 0, hint='a', layout=LANES32), 0)
    b.store(img, b.load(glb, 32, hint='t', layout=LANES24), 32)
    b.store(out, b.load(img, 32, hint='y', layout=LANES24), 32)
    b.store(out, b.load(img, 32, hint='z', layout=LANES32), 0)
    body = _values(b)
    assert _registers(body) == set()
    (insert,) = _ops(body, Op.INSERT)
    (part,) = _ops(body, Op.SLICE)
    assert insert.args[2] == 0 and part.args == (insert.target[0], 0)
    src = _esimd(body)
    assert re.search(r'\w+\.template select<24, 1>\(0\) = v\d+_t;', src), src
    assert ('simd<float, 24>(' in src
            and '.template select<24, 1>(0))' in src), src


def test_a_tail_read_is_its_lanes_in_zeros():
    """A load whose lanes past `valid` are not the block's reads zeros
    there, as the explicit vector's emitter held it to them."""
    b = _builder()
    img = b.alloc(Datatype.F32, (32,), MemSpace.REGISTER, extern='r1',
                  init='{}')
    glb = b.alloc(Datatype.F32, (64,), MemSpace.GLOBAL, hint='g')
    out = b.alloc(Datatype.F32, (64,), MemSpace.GLOBAL, hint='o')
    b.store(img, b.load(glb, 0, hint='a', layout=LANES32), 0)
    b.store(out, b.load(img, 0, hint='x', layout=LANES32, valid=24), 0)
    body = _values(b)
    (insert,) = _ops(body, Op.INSERT)
    (part,) = _ops(body, Op.SLICE)
    assert insert.args[1] is part.target[0]
    zero = insert.args[0]
    assert next(s for s in _ops(body, Op.CONST)
                if s.target[0] is zero).attr('value') == 0.0


def test_spmd_spells_a_part_of_a_vector_by_its_elements():
    """Under SPMD a block is a lane's slots: a pair, read as its second
    element, and written there."""
    pair = ScalarType(Datatype.F32, 2)
    b = _builder()
    img = b.alloc(Datatype.F32, (4,), MemSpace.REGISTER, extern='r1',
                  init='{}', align=8)
    glb = b.alloc(Datatype.F32, (8,), MemSpace.GLOBAL, hint='g')
    out = b.alloc(Datatype.F32, (8,), MemSpace.GLOBAL, hint='o')
    b.store(img, b.load(glb, 0, type_=pair, hint='a', align=8), 0, align=8)
    b.store(img, b.load(glb, 4, hint='s'), 3)
    b.store(out, b.load(img, 1, hint='y'), 0)
    b.store(out, b.load(img, 2, type_=pair, hint='z', align=8), 2, align=8)
    # The passes alone: `fold` takes the read of slot 1 for a zero, its
    # bookkeeping keyed on where the pair's store starts.
    blocks = RegisterBlocks(b.scratch, values=True)
    body = blocks.as_values(blocks.split(b.finish()))
    assert _registers(body) == set()
    src = _spmd(body)
    assert re.search(r'= \(?v\d+_a\[1\]\)?;', src), src
    assert re.search(r'(\w+)\[1\] = v\d+_s;', src), src
    assert 'tensorforge::VectorT<float, 2>{}' in src, src


# --------------------------------------------------------------------------
# across control flow that stays
# --------------------------------------------------------------------------

def _branch_and_loop(b):
    img = b.alloc(Datatype.F32, (4,), MemSpace.REGISTER, extern='r1',
                  init='{}')
    glb = b.alloc(Datatype.F32, (64,), MemSpace.GLOBAL, hint='g')
    out = b.alloc(Datatype.F32, (64,), MemSpace.GLOBAL, hint='o')
    count = b.extern_value('count', SIZE, hint='count')
    with b.if_(b.op('lt', BOOL, count, 5, hint='few')):
        b.store(img, b.load(glb, 1, hint='b'), 1)
    with b.for_(0, count, 1, hint='t', index_type=SIZE) as loop:
        acc = b.load(img, 2, hint='acc')
        b.store(img, b.op('add', F32, acc, b.load(glb, loop.induction,
                                                  hint='x'), hint='sum'), 2)
    for k in range(4):
        b.store(out, b.load(img, k, hint=f'y{k}'), k)


def test_a_branch_hands_out_what_it_stored_and_a_loop_carries_it():
    b = _builder()
    _branch_and_loop(b)
    report: list = []
    body = optimize(b.finish(), register_values=True, scratch=b.scratch)
    assert _registers(body) == set()
    assert verify(body, strict=False) == []
    (branch,) = [s for s in body if s.op == Op.IF]
    (loop,) = [s for s in body if s.op == Op.FOR]
    assert len(branch.target) == 1 and len(branch.regions) == 2
    # the arm that was not there hands on the zeros
    (handed,) = branch.regions[1].body[-1].args
    assert next(s for s in _ops(body, Op.CONST)
                if s.target[0] is handed).attr('value') == 0.0
    assert len(loop.target) == 1 and len(loop.iter_args) == 1
    assert loop.regions[0].body[-1].op == Op.YIELD
    src = _spmd(body)
    assert re.search(r'for \(size_t v\d+_t = 0; v\d+_t < count; \+\+v\d+_t\) '
                     r'\{\s*float v\d+_x = \w+\[v\d+_t\];\s*(v\d+_r1_b2) = '
                     r'\(\1 \+ v\d+_x\);', src), src
    del report


def test_a_store_in_a_queried_loop_or_raw_text_keeps_its_block():
    b = _builder()
    img = b.alloc(Datatype.F32, (2,), MemSpace.REGISTER, extern='r1',
                  init='{}')
    out = b.alloc(Datatype.F32, (8,), MemSpace.GLOBAL, hint='o')
    with b.while_(0, hint='q') as loop:
        b.store(img, b.const(1.0), 0)
        nxt = b.op('add', INDEX, loop.induction, 1, hint='n')
        loop.exit_when(b.op('lt', BOOL, 3, nxt, hint='done'))
        loop.yield_(nxt)
    with b.For('int k = 0; k < 2; ++k'):
        b.store(img, b.const(2.0), 1)
    b.store(out, b.load(img, 0, hint='x'), 0)
    b.store(out, b.load(img, 1, hint='y'), 1)
    report: list = []
    blocks = RegisterBlocks(b.scratch, values=True, report=report)
    body = blocks.as_values(blocks.split(b.finish()))
    assert _registers(body) == {'r1_b0', 'r1_b1'}
    assert sorted(line for line in report if line.startswith('values: -')) == [
        'values: - r1_b0: a store in a loop whose trips are queried',
        'values: - r1_b1: a store in a block of raw text']


# --------------------------------------------------------------------------
# under a predicate
# --------------------------------------------------------------------------

def test_a_predicated_store_is_a_select_and_a_lane_mask_a_merge():
    """Under the explicit vector a lane guard is a mask, and the store it
    holds keeps the column where the mask is off."""
    b = _builder()
    img = b.alloc(Datatype.F32, (32,), MemSpace.REGISTER, extern='r1',
                  init='{}')
    glb = b.alloc(Datatype.F32, (64,), MemSpace.GLOBAL, hint='g')
    out = b.alloc(Datatype.F32, (64,), MemSpace.GLOBAL, hint='o')
    lead = b.rawexpr('tensorforge::intel_esimd::simd<int32_t, 32>(0, 1)',
                     type_=INDEX, layout=LANES32, pure=True, hint='lead')
    inside = b.op('lt', BOOL, lead, 24, hint='inside')
    b.store(img, b.load(glb, 0, hint='a', layout=LANES32), 0,
            predicate=inside)
    b.store(out, b.load(img, 0, hint='x', layout=LANES32), 0)
    body = _values(b)
    assert _registers(body) == set()
    (pick,) = _ops(body, 'select')
    assert pick.args[0] is inside
    assert '.merge(' in _esimd(body)


def test_the_images_a_predicate_kept_whole_split_where_values_follow():
    def build(b):
        img = b.alloc(Datatype.F32, (4,), MemSpace.REGISTER, extern='r1',
                      init='{}')
        out = b.alloc(Datatype.F32, (8,), MemSpace.GLOBAL, hint='o')
        lane = b.rawexpr('threadIdx.x', type_=INDEX, hint='tid')
        inside = b.op('lt', BOOL, lane, 9, hint='inside')
        b.store(img, b.const(1.0), 1, predicate=inside)
        b.store(out, b.load(img, 1, hint='x'), 0)
    for values, left in ((False, {'r1'}), (True, set())):
        b = _builder()
        build(b)
        blocks = RegisterBlocks(b.scratch, values=values)
        body = blocks.split(b.finish())
        if values:
            body = blocks.as_values(body)
        assert _registers(body) == left


# --------------------------------------------------------------------------
# constants a vector holds
# --------------------------------------------------------------------------

def test_a_zero_column_is_stored_as_a_vector_with_the_option_off():
    """`fold` takes a read of a pre-zeroed image where nothing wrote for the
    zero it holds, typed as the column it reads.  The explicit vector's
    emitter wrote that back as `0.0f.copy_to(...)`, which is not C++."""
    b = _builder()
    img = b.alloc(Datatype.F32, (32,), MemSpace.REGISTER, extern='r1',
                  init='{}')
    glb = b.alloc(Datatype.F32, (32,), MemSpace.GLOBAL, hint='g')
    b.store(img, b.load(glb, 0, layout=LANES16), 0)
    b.store(glb, b.load(img, 16, layout=LANES16), 16)
    src = _esimd(optimize(b.finish(), explicit_simd=True))
    assert '0.0f.copy_to' not in src, src
    assert 'simd<float, 16>(0.0f).copy_to(' in src, src


def test_a_vector_constant_is_spelled_as_its_vector_under_spmd():
    b = _builder()
    glb = b.alloc(Datatype.F32, (8,), MemSpace.GLOBAL, hint='g')
    pair = ScalarType(Datatype.F32, 2)
    b.store(glb, b.splat(0.0, pair), 0, align=8)
    b.store(glb, b.splat(2.0, pair), 2, align=8)
    src = _spmd(b.finish())
    assert 'tensorforge::VectorT<float, 2>{}' in src, src
    assert 'tensorforge::VectorT<float, 2>{2.0f, 2.0f}' in src, src


# --------------------------------------------------------------------------
# behind the option
# --------------------------------------------------------------------------

def test_values_run_where_the_body_asks_and_take_the_blocks_along():
    b = _builder()
    pm = pipeline.standard_pipeline('', None, None, None, None, None, None,
                                    None, None, None, None, None, None,
                                    RegisterBlocks(b.scratch))
    steps = {p.name: p for p in pm.passes
             if p.name.startswith(('blocks-', 'values'))}
    assert list(steps) == ['blocks-unroll', 'blocks-split', 'blocks-dce',
                           'values', 'values-dce']
    for values, blocks in ((False, False), (True, True)):
        pc = pipeline.BodyContext((), register_values=values)
        assert steps['values'].enabled(pc) is values
        assert steps['blocks-split'].enabled(pc) is blocks


def _kernel(name, backend, **options):
    path = next(CASES.rglob(f'{name}.py'))
    spec = importlib.util.spec_from_file_location(f'tf_values__{name}', path)
    case = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(case)
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter('ignore')
        gen = Generator(case.descr_list(),
                        Context(arch='pvc', backend=backend,
                                fp_type=getattr(case, 'DTYPE', Datatype.F32),
                                options=Options(**options)))
        gen.generate()
    return gen.get_kernel()


def test_off_nothing_changes_and_on_no_block_is_memory():
    for backend in ('esimd', 'oneapi'):
        default = _kernel('chain', backend)
        assert _kernel('chain', backend, register_values=False) == default
        values = _kernel('chain', backend, register_values=True)
        assert values != default
        assert not re.search(r'\w+_b\d+\[\d+\]', values), values
        assert not re.search(r'simd<\w+, \d+> r\d+_b\d+\(', values), values
