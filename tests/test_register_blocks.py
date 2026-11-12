# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Register images as blocks (`pir.blocks`, `Options.register_blocks`).

Two passes: one writes out the loops an image is addressed through, so that
every offset is a constant, and one replaces an image every access of which
lands inside one block by its blocks.  What is pinned here is what each may
do and what it has to leave: an access whose offset nobody can know, one that
reaches across two blocks, one under a predicate, and an image raw text names
keep their image; a loop that carries a value stays a loop.  And with the
option off, nothing changes at all.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import warnings
from pathlib import Path

from tensorforge.backend.pir import (BOOL, INDEX, TOKEN, IRBuilder,
                                     MemSpace, Op, emit, optimize, pipeline,
                                     verify)
from tensorforge.backend.pir.blocks import (UNROLL_MAX, RegisterBlocks,
                                            split_register_images,
                                            unroll_register_loops)
from tensorforge.backend.pir.core import (LaneAxis, RegisterLayout,
                                          ScalarType, walk_stmts)
from tensorforge.backend.pir.emit_esimd import EsimdEmitter
from tensorforge.backend.writer import Writer
from tensorforge.common.basic_types import Datatype
from tensorforge.common.context import Context, Options
from tensorforge.generators.generator import Generator

F32 = ScalarType(Datatype.F32)
#: A value spread over sixteen lanes: under the explicit vector, an access
#: of one spans sixteen elements of an image.
SPREAD16 = RegisterLayout((LaneAxis(16),))

CASES = Path(__file__).parent / 'cases'


def _image(b, volume=64, init='{}'):
    return b.alloc(Datatype.F32, (volume,), MemSpace.REGISTER, extern='r1',
                   init=init)


def _columns(b, trips=4, width=16):
    """An image filled a column a trip, as a loader fills one, and read
    back a column at a time into global memory."""
    img = _image(b, volume=trips * width)
    src = b.alloc(Datatype.F32, (trips * width,), MemSpace.GLOBAL, hint='g')
    out = b.alloc(Datatype.F32, (trips * width,), MemSpace.GLOBAL, hint='o')
    with b.for_(0, trips, hint='i') as loop:
        at = b.op('mul', INDEX, loop.induction, width, hint='at')
        b.store(img, b.load(src, at, hint='x', layout=SPREAD16), at)
    for k in range(trips):
        b.store(out, b.load(img, k * width, hint='y', layout=SPREAD16),
                k * width)
    return img


def _builder():
    return IRBuilder(fptype=Datatype.F32, arena='shrMem')


def _ops(body, op):
    return [s for s in walk_stmts(body) if s.op == op]


def _allocs(body):
    return {s.attr('extern'): s.target[0] for s in _ops(body, Op.ALLOC)
            if s.attr('extern')}


def _register_accesses(body):
    return [s for s in walk_stmts(body) if s.op in (Op.LOAD, Op.STORE)
            and s.accesses and s.accesses[0].space is MemSpace.REGISTER]


def _blocks(build, explicit_simd=True):
    """What `build(builder)` builds, through both passes."""
    b = _builder()
    build(b)
    body = unroll_register_loops(b.finish(), b.scratch,
                                 explicit_simd=explicit_simd)
    report: list = []
    body = split_register_images(body, b.scratch, explicit_simd=explicit_simd,
                                 report=report)
    return body, report


# --------------------------------------------------------------------------
# writing loops out
# --------------------------------------------------------------------------

def test_a_loop_over_an_image_is_written_out_with_constant_offsets():
    b = _builder()
    _columns(b)
    body = unroll_register_loops(b.finish(), b.scratch, explicit_simd=True)
    assert not _ops(body, Op.FOR)
    consts = {s.target[0].id: s.attr('value') for s in _ops(body, Op.CONST)}
    stores = [s for s in _register_accesses(body) if s.op == Op.STORE]
    assert len(stores) == 4
    assert sorted(consts[s.args[2].id] for s in stores) == [0, 16, 32, 48]


def test_a_loop_that_carries_a_value_stays():
    """A trip hands the next one a sum, and that is left to the loop: only
    a token, which is no variable of the source, is handed on from copy to
    copy.  The image the loop reads is then addressed through a loop that
    stands, and keeps its array."""
    b = _builder()
    img = _image(b)
    with b.for_(0, 4, inits=(b.const(0.0),), types=(F32,),
                hint='i') as loop:
        at = b.op('mul', INDEX, loop.induction, 16, hint='at')
        x = b.load(img, at, hint='x', layout=SPREAD16)
        loop.yield_(b.op('add', F32, loop.iter_args[0], x, hint='sum'))
    before = b.finish()
    assert unroll_register_loops(before, b.scratch,
                                 explicit_simd=True) is before
    report: list = []
    assert split_register_images(before, b.scratch, explicit_simd=True,
                                 report=report) is before
    assert report == ['blocks: - r1: it is addressed through a loop: it '
                      'carries a value']


def test_a_loop_that_carries_a_token_hands_it_from_copy_to_copy():
    """A load issued for the next trip, waited for on it: the copies hand
    the token on, and what waits behind the loop waits for the last."""
    b = _builder()
    img = _image(b, volume=4)
    src = b.alloc(Datatype.F32, (8,), MemSpace.GLOBAL, hint='g')
    first = b.load_async(src, 0, hint='x')
    with b.for_(0, 4, inits=(first,), types=(TOKEN,), hint='i') as loop:
        b.store(img, b.wait(loop.iter_args[0]), loop.induction)
        loop.yield_(b.load_async(src, b.op('add', INDEX, loop.induction, 1,
                                           hint='next'), hint='x'))
    last = b.wait(loop.result)
    body = unroll_register_loops(b.finish(), b.scratch)
    assert not _ops(body, Op.FOR)
    assert verify(body, strict=False) == []
    issued = [s.target[0] for s in _ops(body, Op.LOAD_ASYNC)]
    waits = _ops(body, Op.WAIT)
    assert len(issued) == len(waits) == 5
    assert [w.args[0] for w in waits] == issued
    assert waits[-1].target[0] is last


def test_a_merged_run_stays():
    """A merged run is a loop because the code written out would not fit
    the instruction cache (`merge_variants`): the blocks do not undo that."""
    b = _builder()
    img = _image(b)
    with b.for_(0, 4, hint='i', rolled=True) as loop:
        at = b.op('mul', INDEX, loop.induction, 16, hint='at')
        b.store(img, b.load(img, at, hint='x', layout=SPREAD16), at)
    before = b.finish()
    assert unroll_register_loops(before, b.scratch,
                                 explicit_simd=True) is before


def test_a_loop_whose_text_names_its_induction_stays():
    """Raw text spelling the induction would spell a name no copy declares."""
    b = _builder()
    img = _image(b)
    with b.for_(0, 4, hint='i') as loop:
        at = b.op('mul', INDEX, loop.induction, 16, hint='at')
        b.store(img, b.const(1.0), at)
        b(f'log({loop.induction});', loop.induction, accesses=())
    before = b.finish()
    assert unroll_register_loops(before, b.scratch,
                                 explicit_simd=True) is before


def test_no_access_is_written_out_more_than_unroll_max_times():
    """65 trips of one loop are past `UNROLL_MAX`, and so are eight of one
    loop around nine of another written out for another image: a statement
    of the inner one would be copied 72 times."""
    def long(b):
        img = _image(b, volume=128)
        with b.for_(0, UNROLL_MAX + 1, hint='i') as loop:
            b.store(img, b.const(1.0), loop.induction)
    assert _refused(long, explicit_simd=False) == [
        'blocks: - r1: 65 trips of the loops it is addressed through, more '
        'than 64']

    def nested(b):
        img = _image(b, volume=16)
        other = b.alloc(Datatype.F32, (16,), MemSpace.REGISTER,
                        extern='r2', init='{}')
        with b.for_(0, 8, hint='i') as outer:
            b.store(img, b.const(1.0), outer.induction)
            with b.for_(0, 9, hint='j') as inner:
                b.store(other, b.const(2.0), inner.induction)
    why = 'more than 64 trips of loops written out around one another'
    assert _refused(nested, explicit_simd=False) == [
        f'blocks: - r1: {why}', f'blocks: - r2: {why}']


def test_a_fence_stands_between_trips_under_the_explicit_vector_only():
    """vISA's software fence between the copies, where `Options.
    esimd_unroll_fence` asks for one: three for four trips, pinned, and
    touching no memory a pass models.  SPMD has no such fence to write."""
    for simd, fences in ((True, 3), (False, 0)):
        b = _builder()
        _columns(b)
        body = unroll_register_loops(b.finish(), b.scratch,
                                     explicit_simd=simd, fence=128)
        found = [s for s in _ops(body, Op.RAWSTMT) if 'fence' in s.text]
        assert len(found) == fences
        assert all(s.text == 'tensorforge::intel_esimd::fence<128>();'
                   and not s.accesses and not s.movable for s in found)


# --------------------------------------------------------------------------
# splitting images
# --------------------------------------------------------------------------

def test_an_image_whose_accesses_land_in_blocks_is_its_blocks():
    body, report = _blocks(_columns)
    allocs = _allocs(body)
    assert sorted(allocs) == ['r1_b0', 'r1_b1', 'r1_b2', 'r1_b3']
    for k in range(4):
        assert allocs[f'r1_b{k}'].type.shape == (16,)
    for s in _register_accesses(body):
        assert s.args[-1] == 0
        assert s.accesses[0].base is s.args[0]
    assert report == ['blocks: r1: 64 elements as 4 of 4 blocks of 16']


def test_under_spmd_an_access_is_an_element_and_a_block_is_one():
    """The same body per lane: an access is one element of the lane's array,
    and a block of one is what the emitter writes as a one-element array."""
    def build(b):
        img = _image(b, volume=8)
        for k in (0, 5):
            b.store(img, b.const(float(k)), k)
        b.store(b.alloc(Datatype.F32, (8,), MemSpace.GLOBAL, hint='o'),
                b.load(img, 5, hint='y'), 0)
    body, report = _blocks(build, explicit_simd=False)
    assert {'r1_b0', 'r1_b5'} <= set(_allocs(body))
    assert report == ['blocks: r1: 8 elements as 2 of 8 blocks of 1']
    writer = Writer()
    emit(body, writer)
    src = writer.get_src()
    assert 'float r1_b5[1]{};' in src and 'r1_b5[0]' in src, src
    assert 'r1[' not in src, src


def test_the_explicit_vector_declares_each_block_as_a_simd():
    body, _ = _blocks(_columns)
    writer = Writer()
    ctx = Context(arch='pvc', backend='esimd', fp_type=Datatype.F32)
    EsimdEmitter(writer=writer, context=ctx, strict=False).run(body)
    src = writer.get_src()
    for k in range(4):
        assert f'simd<float, 16> r1_b{k}(' in src, src
    assert '> r1(' not in src, src


def _refused(build, explicit_simd=True):
    b = _builder()
    build(b)
    before = b.finish()
    report: list = []
    after = split_register_images(unroll_register_loops(
        before, b.scratch, explicit_simd=explicit_simd), b.scratch,
        explicit_simd=explicit_simd, report=report)
    assert 'r1' in _allocs(after)
    return report


def test_an_offset_nobody_knows_keeps_the_image():
    def build(b):
        img = _image(b)
        lane = b.rawexpr('threadIdx.x', type_=INDEX, hint='tid')
        b.store(img, b.const(1.0), lane)
        b.store(img, b.const(1.0), 16)
    assert _refused(build) == ['blocks: - r1: an offset that is not known']


def test_an_access_across_two_blocks_keeps_the_image():
    def build(b):
        img = _image(b)
        b.store(img, b.load(img, 0, hint='x', layout=SPREAD16), 8)
        b.store(img, b.load(img, 32, hint='y', layout=SPREAD16), 48)
    assert _refused(build) == ['blocks: - r1: an access across two blocks']


def test_an_access_under_a_predicate_keeps_the_image():
    def build(b):
        img = _image(b)
        lane = b.rawexpr('threadIdx.x', type_=INDEX, hint='tid')
        inside = b.op('lt', BOOL, lane, 9, hint='inside')
        b.store(img, b.const(1.0), 16, predicate=inside)
    assert _refused(build) == ['blocks: - r1: an access under a predicate']


def test_an_image_raw_text_names_keeps_the_image():
    def build(b):
        img = _image(b)
        b.store(img, b.const(1.0), 16)
        b('consume(r1);', accesses=())
    assert _refused(build) == ['blocks: - r1: raw text names it']


def test_a_comment_naming_the_image_is_no_use_of_it():
    def build(b):
        img = _image(b)
        b.Comment('r1 = load{g>r}(glb_m0);')
        b.store(img, b.load(img, 0, hint='x', layout=SPREAD16), 16)
    body, report = _blocks(build)
    assert report == ['blocks: r1: 64 elements as 2 of 4 blocks of 16']


def test_an_image_that_starts_from_something_else_keeps_its_array():
    def build(b):
        img = _image(b, init='{1.0f}')
        b.store(img, b.const(1.0), 16)
    assert _refused(build) == ['blocks: - r1: it starts from {1.0f}']


# --------------------------------------------------------------------------
# behind the option
# --------------------------------------------------------------------------

def _enabled(register_blocks: bool) -> bool:
    b = _builder()
    pm = pipeline.standard_pipeline('', None, None, None, None, None, None,
                                    None, None, None, None, None, None,
                                    RegisterBlocks(b.scratch))
    pc = pipeline.BodyContext((), register_blocks=register_blocks)
    steps = [p for p in pm.passes if p.name.startswith('blocks-')]
    assert [p.name for p in steps] == ['blocks-unroll', 'blocks-split',
                                       'blocks-dce']
    states = {p.enabled(pc) for p in steps}
    assert len(states) == 1
    return states.pop()


def test_the_passes_run_where_the_body_asks_for_them():
    assert not _enabled(False)
    assert _enabled(True)


def test_optimize_leaves_an_image_whole_unless_asked():
    for asked, image in ((False, True), (True, False)):
        b = _builder()
        _columns(b)
        body = optimize(b.finish(), explicit_simd=True, register_blocks=asked,
                        scratch=b.scratch)
        assert ('r1' in _allocs(body)) is image
        assert bool(_ops(body, Op.FOR)) is image


def _kernel(name, backend, **options):
    path = next(CASES.rglob(f'{name}.py'))
    spec = importlib.util.spec_from_file_location(f'tf_blocks__{name}', path)
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


def test_off_the_source_is_the_default_one():
    """Asked off, the kernel is byte for byte the one nobody asked anything
    for -- and asked on, it is not, so that the comparison says something."""
    for backend in ('esimd', 'oneapi'):
        default = _kernel('chain', backend)
        assert _kernel('chain', backend, register_blocks=False) == default
        blocks = _kernel('chain', backend, register_blocks=True)
        assert blocks != default
        assert '_b0' in blocks and '_b0' not in default
