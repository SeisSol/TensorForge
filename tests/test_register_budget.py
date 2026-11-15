# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""The register budget (`pir.budget`, `Options.register_spill`).

Pinned here: the model the pass decides over is `passes.pressure`'s, value
for value; a value that is a load of memory nothing writes goes where too
much is live and comes back before its next read, the one read again
furthest ahead first, and nothing else goes -- not a load of memory the body
writes; a constant goes too; a body within the budget stays as it is; a
register buffer nothing touches is not held against the budget.  The stop:
a body within the margin stays; a reload that costs more than the spill it
saves is not taken; where values that cannot go hold the excess, only
constants go; and where no load goes, the images stay whole.  The budget a
target gives, from the calibrations there are; and with the option off,
nothing changes.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import warnings
from pathlib import Path

from tensorforge.backend.instructions.abstract_instruction import (
    register_budget)
from tensorforge.backend.pir import (IRBuilder, MemSpace, Op, passes, pipeline,
                                     verify)
from tensorforge.backend.pir.blocks import RegisterBlocks
from tensorforge.backend.pir.budget import (RELOAD_PRICE, Budget, Liveness,
                                            rematerialize)
from tensorforge.backend.pir.core import (INDEX, LaneAxis, RegisterLayout,
                                          ScalarType, walk_stmts)
from tensorforge.common.basic_types import Datatype
from tensorforge.common.context import Context, Options
from tensorforge.generators.generator import Generator

F32 = ScalarType(Datatype.F32)
LANES16 = RegisterLayout((LaneAxis(16),))
CASES = Path(__file__).parent / 'cases'


def _loads(body, buf=None):
    return [s for s in walk_stmts(body) if s.op == Op.LOAD
            and (buf is None or s.args[0] is buf)]


def _gap(b, written=False, operators=4, between=12):
    """Operators read once early and once late, a stretch in between that
    holds `between` values of its own at once, and -- `written` -- the
    operators' memory written at the end."""
    op = b.alloc(Datatype.F32, (operators,), MemSpace.GLOBAL, hint='op')
    out = b.alloc(Datatype.F32, (64,), MemSpace.GLOBAL, hint='out')
    xs = [b.load(op, k, hint=f'a{k}') for k in range(operators)]
    s = xs[0]
    for x in xs[1:]:
        s = b.op('add', F32, s, x, hint='early')
    b.store(out, s, 0)
    mid = [b.load(out, 8 + k, hint=f'm{k}') for k in range(between)]
    m = mid[0]
    for v in mid[1:]:
        m = b.op('add', F32, m, v, hint='mid')
    b.store(out, m, 1)
    u = xs[-1]
    for x in reversed(xs[:-1]):
        u = b.op('mul', F32, u, x, hint='late')
    b.store(out, u, 2)
    if written:
        b.store(op, u, 0)
    return op, xs


def test_the_model_is_pressures_value_for_value():
    for simd in (False, True):
        b = IRBuilder(fptype=Datatype.F32)
        glb = b.alloc(Datatype.F32, (64,), MemSpace.GLOBAL, hint='g')
        layout = LANES16 if simd else None
        xs = [b.load(glb, 16 * k, hint=f'x{k}', layout=layout)
              for k in range(4)]
        acc = xs[0]
        for x in xs[1:]:
            acc = b.op('add', F32, acc, x, hint='acc')
        b.store(glb, acc, 0)
        body = b.finish()
        assert Liveness(body, simd).peak == passes.pressure(
            body, in_bytes=True, explicit_simd=simd)


def test_an_operator_is_loaded_again_across_a_stretch_too_full_for_it():
    b = IRBuilder(fptype=Datatype.F32)
    op, xs = _gap(b)
    body = b.finish()
    peak = Liveness(body, False).peak
    report: list = []
    out = rematerialize(body, b.scratch, budget=peak - 4, report=report)
    assert verify(out, strict=False) == []
    assert Liveness(out, False).peak <= peak - 4
    again = _loads(out, op)[len(xs):]
    assert again, report
    # Loaded again from where it was, and read from there on.
    reloaded = {s.args[1] for s in again}
    assert reloaded <= {s.args[1] for s in _loads(out, op)[:len(xs)]}
    late = [s for s in walk_stmts(out) if s.op == 'mul']
    read = {v.id for s in late for v in s.operands()}
    assert {s.target[0].id for s in again} <= read


def test_furthest_ahead_goes_first():
    """Two operators held across a stretch with room for one: the one read
    again last goes, the other stays."""
    b = IRBuilder(fptype=Datatype.F32)
    op = b.alloc(Datatype.F32, (2,), MemSpace.GLOBAL, hint='op')
    out = b.alloc(Datatype.F32, (64,), MemSpace.GLOBAL, hint='out')
    x0, x1 = (b.load(op, k, hint=f'a{k}') for k in range(2))
    b.store(out, b.op('add', F32, x0, x1, hint='early'), 0)
    mid = [b.load(out, 8 + k, hint=f'm{k}') for k in range(6)]
    m = mid[0]
    for v in mid[1:]:
        m = b.op('add', F32, m, v, hint='mid')
    b.store(out, m, 1)
    b.store(out, x1, 2)
    b.store(out, x0, 3)
    body = b.finish()
    peak = Liveness(body, False).peak
    after = rematerialize(body, b.scratch, budget=peak - 4)
    (again,) = _loads(after, op)[2:]
    assert again.args[1] == 0


def test_a_load_of_memory_the_body_writes_stays():
    b = IRBuilder(fptype=Datatype.F32)
    op, xs = _gap(b, written=True)
    body = b.finish()
    out = rematerialize(body, b.scratch,
                        budget=Liveness(body, False).peak - 4)
    assert len(_loads(out, op)) == len(xs)


def test_a_constant_is_put_back_as_itself():
    b = IRBuilder(fptype=Datatype.F32)
    out = b.alloc(Datatype.F32, (64,), MemSpace.GLOBAL, hint='out')
    zero = b.splat(0.0, F32, layout=LANES16)
    x = b.load(out, 0, hint='x', layout=LANES16)
    b.store(out, b.op('add', F32, x, zero, hint='early'), 0)
    mid = [b.load(out, 16 * (k + 1), hint=f'm{k}', layout=LANES16)
           for k in range(3)]
    m = mid[0]
    for v in mid[1:]:
        m = b.op('add', F32, m, v, hint='mid')
    b.store(out, m, 16)
    b.store(out, b.op('add', F32, m, zero, hint='late'), 32)
    body = b.finish()
    peak = Liveness(body, True).peak
    after = rematerialize(body, b.scratch, budget=peak - 64,
                          explicit_simd=True)
    consts = [s for s in walk_stmts(after) if s.op == Op.CONST]
    assert len(consts) == 2


def test_a_body_within_the_budget_stays():
    b = IRBuilder(fptype=Datatype.F32)
    _gap(b)
    body = b.finish()
    assert rematerialize(body, b.scratch,
                         budget=Liveness(body, False).peak) is body


# --------------------------------------------------------------------------
# where it stops
# --------------------------------------------------------------------------

def _crowded(b, guarded=False, between=12, constant=False):
    """`_gap`'s shape: four operators read early and late -- under a
    predicate where `guarded` -- and a stretch between that holds `between`
    loads of memory the body writes, which cannot go; with `constant`, a
    constant read early and late as well."""
    op = b.alloc(Datatype.F32, (4,), MemSpace.GLOBAL, hint='op')
    out = b.alloc(Datatype.F32, (64,), MemSpace.GLOBAL, hint='out')
    g = b.op('lt', ScalarType(Datatype.I32), b.load(out, 63, hint='n'), 4,
             hint='g') if guarded else None
    xs = [b.load(op, k, hint=f'a{k}', predicate=g, other=0.0 if g else None)
          for k in range(4)]
    c = b.const(2.0) if constant else None
    s = xs[0]
    for x in xs[1:] + ([c] if c is not None else []):
        s = b.op('add', F32, s, x, hint='early')
    b.store(out, s, 0)
    mid = [b.load(out, 8 + k, hint=f'm{k}') for k in range(between)]
    m = mid[0]
    for v in mid[1:]:
        m = b.op('add', F32, m, v, hint='mid')
    b.store(out, m, 1)
    u = xs[-1]
    for x in reversed(xs[:-1]):
        u = b.op('mul', F32, u, x, hint='late')
    if c is not None:
        u = b.op('mul', F32, u, c, hint='late')
    b.store(out, u, 2)
    return op


def test_a_body_within_the_margin_stays():
    b = IRBuilder(fptype=Datatype.F32)
    _crowded(b)
    body = b.finish()
    peak = Liveness(body, False).peak
    # Over the capacity by one value, within a fifth of it.
    budget = Budget(peak - 4, margin=0.2)
    assert budget.trigger >= peak
    assert rematerialize(body, b.scratch, budget=budget) is body


def test_a_reload_dearer_than_the_spill_it_saves_is_not_taken():
    """The operators under a predicate: a guarded reload costs four, which a
    spill of two does not pay for and one of four does."""
    for price, back in ((2, False), (RELOAD_PRICE['guarded'], True)):
        b = IRBuilder(fptype=Datatype.F32)
        op = _crowded(b, guarded=True)
        body = b.finish()
        peak = Liveness(body, False).peak
        taken: dict = {}
        out = rematerialize(body, b.scratch,
                            budget=Budget(peak - 8, margin=0.0,
                                          spill_price=price), taken=taken)
        assert (len(_loads(out, op)) > 4) is back, (price, taken)
        assert bool(taken['guarded']) is back


def test_where_what_cannot_go_holds_the_excess_only_constants_go():
    """A stretch whose own loads, of memory the body writes, leave it over
    the margin whatever goes: the operators stay, the constant goes."""
    b = IRBuilder(fptype=Datatype.F32)
    op = _crowded(b, between=40, constant=True)
    body = b.finish()
    peak = Liveness(body, False).peak
    taken: dict = {}
    out = rematerialize(body, b.scratch,
                        budget=Budget(peak // 2, margin=0.1), taken=taken)
    assert len(_loads(out, op)) == 4
    assert taken == {'const': 1, 'load': 0, 'guarded': 0}
    consts = [s for s in walk_stmts(out) if s.op == Op.CONST]
    assert len(consts) > len([s for s in walk_stmts(body)
                              if s.op == Op.CONST])
    # Where the stretch can be brought within the margin, the operators go.
    b = IRBuilder(fptype=Datatype.F32)
    op = _crowded(b, between=12)
    body = b.finish()
    peak = Liveness(body, False).peak
    out = rematerialize(body, b.scratch,
                        budget=Budget(peak - 12, margin=0.0))
    assert len(_loads(out, op)) > 4


def test_where_no_load_goes_the_images_stay_whole():
    """`RegisterBlocks.remat` hands back the body as it came to `unroll`
    where it would put back nothing but constants, and the body it made
    where a load goes."""
    for between, kept in ((40, True), (12, False)):
        b = IRBuilder(fptype=Datatype.F32)
        _crowded(b, between=between, constant=True)
        body = b.finish()
        peak = Liveness(body, False).peak
        budget = (Budget(peak // 2, margin=0.1) if kept
                  else Budget(peak - 12, margin=0.0, left_over=1.0))
        blocks = RegisterBlocks(b.scratch, values=True, budget=budget,
                                report=[])
        out = blocks.remat(blocks.as_values(blocks.split(
            blocks.unroll(body))))
        assert (out is body) is kept, blocks.report


def test_judged_against_the_body_as_written():
    """Where the body as written is known (`as_written`), loads go only if
    its excess is large against the capacity (`EXCESS_MIN`) and what pays
    closes most of it (`CLOSED_MIN`)."""
    from tensorforge.backend.pir.budget import CLOSED_MIN, EXCESS_MIN
    def run(between, capacity_of, written_of):
        b = IRBuilder(fptype=Datatype.F32)
        op = _crowded(b, between=between)
        body = b.finish()
        capacity = capacity_of(Liveness(body, False).peak)
        out = rematerialize(body, b.scratch,
                            budget=Budget(capacity, margin=0.0),
                            as_written=written_of(capacity))
        return len(_loads(out, op)) > 4
    # Closable, and the body as written twice the capacity: loads go.
    assert run(12, lambda p: p - 12, lambda c: 2 * c)
    # Closable, but the excess as written too small to be worth it.
    assert not run(12, lambda p: p - 12,
                   lambda c: int(c * (1 + EXCESS_MIN / 2)))
    # What cannot go holds most of it: the share closed stays under the bar.
    assert not run(40, lambda p: p // 2, lambda c: 2 * c)
    assert 0 < CLOSED_MIN < 1


def test_a_register_buffer_nothing_touches_is_not_counted():
    """Under the explicit vector the model holds a register buffer for the
    whole body: one a comment names and nothing reads goes before the budget
    is asked, one raw text names and one a load reads stay."""
    b = IRBuilder(fptype=Datatype.F32)
    out = b.alloc(Datatype.F32, (64,), MemSpace.GLOBAL, hint='out')
    names = ('r1', 'r2', 'r3')
    regs = [b.alloc(Datatype.F32, (64,), MemSpace.REGISTER, extern=n)
            for n in names]
    b.Comment('r1 = load{g>r}(glb_m0);')
    b('consume(r2);', accesses=())
    b.store(out, b.load(regs[2], 0, hint='x', layout=LANES16), 0)
    body = b.finish()
    report: list = []
    after = rematerialize(body, b.scratch, budget=10 ** 6,
                          explicit_simd=True, report=report)
    kept = {s.attr('extern') for s in walk_stmts(after) if s.op == Op.ALLOC
            and s.target[0].type.space is MemSpace.REGISTER}
    assert kept == {'r2', 'r3'}
    assert report[0] == ('budget: 1 register buffer(s) nothing touches '
                         'dropped, 256 B the model held')
    assert Liveness(after, True).peak == Liveness(body, True).peak - 256


# --------------------------------------------------------------------------
# the budget a target gives
# --------------------------------------------------------------------------

def _budget(arch, backend, lanes=None, fp=Datatype.F64, **options):
    return register_budget(Context(arch=arch, backend=backend, fp_type=fp,
                                   options=Options(**options)), lanes)


def test_the_budget_reads_the_file_back_through_the_calibrations():
    # ptxas: 66 + 0.83 a modeled register, at 255.
    assert _budget('sm_90', 'cuda').capacity == int((255 - 66) / 0.83 * 4)
    # hipcc: 0.95 a modeled register in double, the tuner's 1.26 in single,
    # of 512 and of 256.
    assert _budget('gfx942', 'hip').capacity == int(512 / 0.95 * 4)
    assert _budget('gfx1150', 'hip').capacity == int(256 / 0.95 * 4)
    assert _budget('gfx942', 'hip', fp=Datatype.F32).capacity == int(
        512 / 1.26 * 4)
    # The explicit vector: 256 registers of 64 B, less the reserve.
    assert _budget('pvc', 'esimd').capacity == (256 - 16) * 64
    assert _budget('pvc', 'esimd', sycl_grf='128').capacity == (
        (128 - 16) * 64)
    # SPMD on Intel: the thread's file over the sub-group, through IGC's
    # need of 0.8.
    assert _budget('pvc', 'oneapi', lanes=16).capacity == 640
    assert _budget('sm_90', 'cuda', register_budget=640).capacity == 640
    # A reload pays on IGC; a guarded one does not on ptxas and hipcc.
    assert _budget('pvc', 'oneapi', lanes=16).spill_price >= (
        RELOAD_PRICE['guarded'])
    for arch, backend in (('sm_90', 'cuda'), ('gfx1150', 'hip')):
        assert _budget(arch, backend).spill_price < RELOAD_PRICE['guarded']


# --------------------------------------------------------------------------
# behind the option
# --------------------------------------------------------------------------

def test_remat_runs_where_asked_and_takes_the_values_along():
    b = IRBuilder(fptype=Datatype.F32)
    pm = pipeline.standard_pipeline('', None, None, None, None, None, None,
                                    None, None, None, None, None, None,
                                    RegisterBlocks(b.scratch,
                                                   budget=Budget(100)))
    remat = next(p for p in pm.passes if p.name == 'remat')
    values = next(p for p in pm.passes if p.name == 'values')
    for spill, on in (('off', False), ('remat', True)):
        pc = pipeline.BodyContext((), register_spill=spill)
        assert remat.enabled(pc) is on
        assert values.enabled(pc) is on


def _kernel(name, backend, arch='pvc', **options):
    path = next(CASES.rglob(f'{name}.py'))
    spec = importlib.util.spec_from_file_location(f'tf_budget__{name}', path)
    case = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(case)
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter('ignore')
        gen = Generator(case.descr_list(),
                        Context(arch=arch, backend=backend,
                                fp_type=getattr(case, 'DTYPE', Datatype.F32),
                                options=Options(**options)))
        gen.generate()
    return gen.get_kernel()


def test_off_nothing_changes():
    for backend, arch in (('esimd', 'pvc'), ('cuda', 'sm_90')):
        default = _kernel('chain', backend, arch)
        assert _kernel('chain', backend, arch,
                       register_spill='off') == default
        assert _kernel('chain', backend, arch, register_spill='remat',
                       register_budget=64) != default


# --------------------------------------------------------------------------
# the next tier: shared memory
# --------------------------------------------------------------------------

def _computed_gap(b, held=4, between=12):
    """`held` values computed early -- sums nothing can load again -- read
    once more late, and a stretch in between that holds `between` values of
    its own at once."""
    src = b.alloc(Datatype.F32, (64,), MemSpace.GLOBAL, hint='src')
    out = b.alloc(Datatype.F32, (64,), MemSpace.GLOBAL, hint='out')
    lane = b.thread_id('x')
    xs = [b.op('mul', F32, b.load(src, b.op('add', INDEX, lane, 16 * k)),
               2.0, hint=f'c{k}') for k in range(held)]
    mid = [b.load(out, b.op('add', INDEX, lane, 16 * k), hint=f'm{k}')
           for k in range(between)]
    m = mid[0]
    for v in mid[1:]:
        m = b.op('add', F32, m, v, hint='mid')
    b.store(out, m, lane)
    u = xs[-1]
    for x in reversed(xs[:-1]):
        u = b.op('add', F32, u, x, hint='late')
    b.store(out, u, b.op('add', INDEX, lane, 16))
    return xs


def test_a_computed_value_goes_to_a_slot_of_its_own_lane_and_comes_back():
    """`spill_shared`: what `rematerialize` cannot load again -- a sum -- is
    stored behind its definition into its lane's slot of shared memory and
    loaded before its next read; the slots are one buffer of the
    multiplication's arena, its own lane's to every lane, for the whole
    body; within the slots, and the body fits where enough of them are."""
    from tensorforge.backend.pir.budget import spill_shared
    b = IRBuilder(fptype=Datatype.F32, arena='shrMem')
    xs = _computed_gap(b)
    body = b.finish()
    verify(body)
    peak = Liveness(body, False).peak
    report = []
    out = spill_shared(body, b.scratch, capacity=peak - 8, shared=4 * 16 * 4,
                       threads=16, arena='shrMem', report=report)
    verify(out)
    allocs = [s for s in walk_stmts(out) if s.op == Op.ALLOC
              and s.target[0].type.space is MemSpace.SHARED]
    assert len(allocs) == 1
    assert allocs[0].attr('own') and allocs[0].attr('lifetime') == 'body'
    buf = allocs[0].target[0]
    stores = [s for s in walk_stmts(out) if s.op == Op.STORE
              and s.args[0] is buf]
    reloads = _loads(out, buf)
    assert stores and len(reloads) == len(stores)
    assert {s.args[1].id for s in stores} <= {x.id for x in xs}
    # Made where they stand: no compiler takes a load back from the store.
    assert all(s.attr('volatile') for s in stores + reloads)
    assert Liveness(out, False).peak < peak
    assert report and 'spilled' in report[-1], report
    # No slot, no spill: within a budget, or without room.
    assert spill_shared(body, b.scratch, capacity=peak, shared=4096,
                        threads=16, arena='shrMem') is body
    assert spill_shared(body, b.scratch, capacity=peak - 8, shared=32,
                        threads=16, arena='shrMem') is body


def test_a_volatile_access_is_spelled_through_a_volatile_pointer():
    """`Emitter.elem_access`: what `spill_shared` marks `volatile` the
    compiler has to make as often as it stands."""
    from dataclasses import replace as _replace
    from tensorforge.backend.pir import emit
    from tensorforge.backend.writer import Writer
    b = IRBuilder(fptype=Datatype.F32)
    glb = b.alloc(Datatype.F32, (64,), MemSpace.GLOBAL, hint='g')
    b.store(glb, b.op('mul', F32, b.load(glb, 0), 2.0), 1)
    body = tuple(_replace(s, attrs=s.attrs + (('volatile', True),))
                 if s.op in (Op.LOAD, Op.STORE) else s for s in b.finish())
    writer = Writer()
    emit(body, writer, Context(arch='sm_90', backend='cuda',
                               fp_type=Datatype.F32))
    src = writer.get_src()
    assert src.count('*(volatile float*)&') == 2, src


def test_with_too_little_budget_computed_values_go_to_shared_memory():
    """`register_spill=shared` on a kernel: under SPMD on Intel and on
    NVIDIA, with a budget far too small, values go to their lanes' slots of
    shared memory, accessed volatile; off it changes nothing, and the
    explicit vector has no such tier."""
    for backend, arch in (('oneapi', 'pvc'), ('cuda', 'sm_90')):
        src = _kernel('chain', backend, arch, register_spill='shared',
                      register_budget=64)
        assert '*(volatile float*)&' in src and '_spill[' in src, backend
        assert _kernel('chain', backend, arch, register_spill='shared',
                       register_budget=64, register_shared=0) == src
    assert '_spill[' not in _kernel('chain', 'esimd', 'pvc',
                                    register_spill='shared',
                                    register_budget=64)


def test_what_came_of_a_broadcast_is_pinned_through_its_slot():
    """`values._Guards` follows a value through its slot of shared memory
    (`spilled`): what came of a cross-lane read, spilled and loaded back
    into a lane guard, is pinned as it is without the slot -- IGC 2.41.5
    computed the order-6 dense `localFluxAll` wrong with the slots and 27 of
    its 45 pins gone."""
    from tensorforge.backend.pir.budget import spill_shared
    from tensorforge.backend.pir.core import BOOL
    from tensorforge.backend.pir.values import pin_crosslane_sources
    b = IRBuilder(fptype=Datatype.F32, arena='shrMem')
    src = b.alloc(Datatype.F32, (64,), MemSpace.GLOBAL, hint='src')
    out = b.alloc(Datatype.F32, (64,), MemSpace.GLOBAL, hint='out')
    lane = b.thread_id('x')
    inside = b.op('lt', BOOL, lane, 3, hint='inside')
    v = b.op('mul', F32, b.rawexpr(
        'sycl::group_broadcast(item.get_sub_group(), {0}, 1)',
        b.load(src, lane), type_=F32, hint='bc', pure=True, crosslane=True),
        2.0, hint='v')
    mid = [b.load(out, b.op('add', INDEX, lane, 16 * k), hint=f'm{k}')
           for k in range(12)]
    m = mid[0]
    for x in mid[1:]:
        m = b.op('add', F32, m, x, hint='mid')
    b.store(out, m, lane)
    b.store(out, v, b.op('add', INDEX, lane, 16), predicate=inside)
    body = b.finish()
    spilled = spill_shared(body, b.scratch,
                           capacity=Liveness(body, False).peak - 4,
                           shared=4 * 16 * 4, threads=16, arena='shrMem')
    assert spilled is not body
    pins = [s for s in walk_stmts(pin_crosslane_sources(spilled))
            if s.op == Op.RAWSTMT and 'asm' in (s.text or '')]
    assert pins
