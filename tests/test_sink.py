# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""A store of a number nothing reads goes to the end of its list, past what
touches other elements (`pir.sink`), asked of bodies built by hand.

The padding an operation stores past its rows stands behind the operation,
and with the operations of a list handing their results through memory that
is between a product and what reads it; `plConvertToNodal` of SeisSol's
elastic order 4 takes 255 registers on sm_86 with it there, 138 with it at
the end.  Here: what goes, how far, and what stops it.
"""

from __future__ import annotations

from tensorforge.backend.pir import IRBuilder, verify
from tensorforge.backend.pir.core import (BOOL, INDEX, SIZE, Access,
                                          BufferType, Effect, MemSpace, Op,
                                          ScalarType, Uniformity, walk_stmts)
from tensorforge.backend.pir.sink import sink_numbers
from tensorforge.common.basic_types import Datatype

F32 = Datatype.F32
REAL = ScalarType(F32)
LANES = 32
ROWS = 24
HELD = 20
COLS = 4


class Body:
    """`q`, an element's `ROWS x COLS` in global memory, of which the lanes
    below `HELD` hold rows; `p`, one a lane."""

    def __init__(self):
        b = self.b = IRBuilder(fptype=F32, arena='shrMem')
        k = b.extern_value('element', SIZE, uniform=Uniformity.MULT, hint='k')
        self.q = b.decl_expr(
            'float *const __restrict__ glb_m0', '&m0[{0} * 96]',
            BufferType(F32, (ROWS * COLS,), MemSpace.GLOBAL), 'm0', args=(k,),
            kind=Effect.READ, space=MemSpace.GLOBAL, hint='glb_m0',
            extern='glb_m0')
        self.p = b.decl_expr(
            'float *const __restrict__ glb_m1', '&m1[{0} * 32]',
            BufferType(F32, (LANES,), MemSpace.GLOBAL), 'm1', args=(k,),
            kind=Effect.READ, space=MemSpace.GLOBAL, hint='glb_m1',
            extern='glb_m1')
        self.lane = b.op('rem', INDEX, b.thread_id('x'), LANES, hint='lead')
        self.rows = b.op('lt', BOOL, self.lane, HELD, hint='g')
        self.pads = b.op('and', BOOL, b.op('ge', BOOL, self.lane, HELD),
                         b.op('lt', BOOL, self.lane, ROWS), hint='z')

    def element(self, j):
        b = self.b
        return b.op('add', INDEX, self.lane, b.op('mul', INDEX, j, ROWS),
                    hint='a')

    def pad(self, value=0.0):
        """`q` past the rows the lanes hold: `value` -- the padding."""
        b = self.b
        with b.if_(self.pads):
            with b.for_(0, COLS, 1, unroll=True) as hop:
                b.store(self.q, value, self.element(hop.induction))

    def product(self):
        """A number no pass knows, a lane."""
        b = self.b
        return b.op('mul', REAL, b.load(self.p, self.lane), 2.0, hint='v')

    def rows_of_q(self, v):
        """`q` in the rows the lanes hold: `v`."""
        b = self.b
        with b.if_(self.rows):
            with b.for_(0, COLS, 1, unroll=True) as hop:
                b.store(self.q, v, self.element(hop.induction))

    def run(self):
        body = self.b.finish()
        verify(body)
        out = sink_numbers(body, threads=LANES, arena='shrMem')
        verify(out)
        return out


def _order(body, q):
    """The statements of the list that store or load `q`, or are a barrier,
    as `pad` (a store of a number), `load`, `rows` (a store of a value) or
    `barrier`."""
    out = []
    for s in body:
        inner = list(walk_stmts((s,)))
        if any(x.op == Op.STORE and x.args[0] is q
               and not hasattr(x.args[1], 'id') for x in inner):
            out.append('pad')
        elif any(x.op == Op.LOAD and x.args[0] is q for x in inner):
            out.append('load')
        elif any(x.op == Op.STORE and x.args[0] is q for x in inner):
            out.append('rows')
        elif any(x.op == Op.BARRIER for x in inner):
            out.append('barrier')
    return out


def test_the_padding_goes_behind_the_rows():
    """The padding, stored behind a product and ahead of what reads the
    product, goes behind the rows: they are other elements of `q`."""
    t = Body()
    v = t.product()
    t.pad()
    t.rows_of_q(t.b.op('add', REAL, v, 1.0, hint='w'))
    out = t.run()
    assert _order(out, t.q) == ['rows', 'pad']


def test_the_padding_stays_ahead_of_a_load_of_its_elements():
    t = Body()
    b = t.b
    t.pad()
    with b.if_(t.pads):
        b.store(t.p, b.load(t.q, t.element(0)), t.lane)
    out = t.run()
    assert _order(out, t.q) == ['pad', 'load']


def test_the_padding_stays_ahead_of_a_store_of_its_elements():
    """Behind it, the store would leave its number where the other store
    left a value."""
    t = Body()
    b = t.b
    t.pad()
    with b.if_(t.pads):
        b.store(t.q, t.product(), t.element(1))
    out = t.run()
    assert _order(out, t.q) == ['pad', 'rows']


def test_the_padding_stays_ahead_of_a_barrier():
    t = Body()
    b = t.b
    t.pad()
    b.barrier()
    t.rows_of_q(t.product())
    out = t.run()
    assert _order(out, t.q)[:2] == ['pad', 'barrier']


def test_a_store_of_a_value_stays():
    """A value stored is live where it is stored; behind, it would be live
    longer."""
    t = Body()
    b = t.b
    v = t.product()
    with b.if_(t.pads):
        b.store(t.q, v, t.element(0))
    t.rows_of_q(b.op('add', REAL, v, 1.0, hint='w'))
    out = t.run()
    assert _order(out, t.q) == ['rows', 'rows']
    first = next(s for s in out if any(x.op == Op.STORE
                                       for x in walk_stmts((s,))))
    assert any(x.op == Op.STORE and x.args[1] is v
               for x in walk_stmts((first,)))


def test_a_padding_something_reads_stays():
    """A number another statement reads is where it is read from."""
    t = Body()
    b = t.b
    t.pad()
    t.rows_of_q(t.product())
    with b.if_(t.pads):
        b.store(t.p, b.load(t.q, t.element(2)), t.lane)
    out = t.run()
    assert _order(out, t.q)[0] == 'pad'


def test_two_paddings_go_down_in_their_order():
    """Two numbers into the same elements: the second is what memory ends
    with, behind as ahead."""
    t = Body()
    t.pad(0.0)
    t.pad(1.0)
    t.rows_of_q(t.product())
    out = t.run()
    numbers = [x.args[1] for s in out for x in walk_stmts((s,))
               if x.op == Op.STORE and not hasattr(x.args[1], 'id')]
    assert numbers == [0.0, 1.0]
    assert _order(out, t.q) == ['rows', 'pad', 'pad']


def _pads(s, q):
    return any(x.op == Op.STORE and x.args[0] is q
               and not hasattr(x.args[1], 'id') for x in walk_stmts((s,)))


def test_the_padding_stays_ahead_of_the_yield_of_its_loop():
    """A yield ends the loop's body: nothing goes behind it."""
    t = Body()
    b = t.b
    with b.for_(0, 2, 1, inits=(0.0,), types=(REAL,), hint='t') as loop:
        t.pad()
        loop.yield_(b.op('add', REAL, loop.iter_args[0], t.product(),
                         hint='s'))
    out = t.run()
    body = next(s for s in out if s.op == Op.FOR and s.target).regions[0].body
    assert body[-1].op == Op.YIELD
    assert any(_pads(s, t.q) for s in body)


def test_the_padding_stays_ahead_of_an_exit():
    """Behind the exit, the run that leaves the loop would not pad."""
    t = Body()
    b = t.b
    with b.while_(0, hint='t') as loop:
        t.pad()
        step = b.op('add', INDEX, loop.induction, 1, hint='n')
        loop.exit_when(b.op('ge', BOOL, step, 2, hint='e'))
        loop.yield_(step)
    out = t.run()
    body = next(s for s in out if s.op == Op.WHILE).regions[0].body
    ops = [('pad' if _pads(s, t.q) else s.op) for s in body]
    assert ops.index('pad') < ops.index(Op.EXIT)


def test_the_padding_stays_ahead_of_text_that_says_nothing_of_memory():
    """An expression of raw text, impure, with no access declared, may do
    anything."""
    t = Body()
    b = t.b
    t.pad()
    c = b.rawexpr('atomicAdd(counter, 1u)', type_=INDEX, hint='c')
    t.rows_of_q(t.product())
    out = t.run()
    ops = [('pad' if _pads(s, t.q) else 'text' if c in s.target else s.op)
           for s in out]
    assert ops.index('pad') < ops.index('text')


def test_the_padding_stays_ahead_of_what_a_statement_says_it_reads():
    """`touches` says what statements around it reach through a name the
    IR does not follow; it emits nothing, and is no comment."""
    t = Body()
    b = t.b
    t.pad()
    said = b.touches((Access(Effect.READ, MemSpace.GLOBAL, t.q),))
    t.rows_of_q(t.product())
    out = t.run()
    ops = [('pad' if _pads(s, t.q) else 'said' if s is said else s.op)
           for s in out]
    assert ops.index('pad') < ops.index('said')


def test_the_padding_goes_past_a_comment():
    t = Body()
    b = t.b
    v = t.product()
    t.pad()
    b.Comment('the rows')
    t.rows_of_q(b.op('add', REAL, v, 1.0, hint='w'))
    out = t.run()
    assert _order(out, t.q) == ['rows', 'pad']


def test_the_pipeline_asks_the_sink_of_a_body(monkeypatch):
    """Built for sm_86, every body goes through the sink."""
    import contextlib
    import importlib.util
    import io
    import pathlib
    from tensorforge.backend.pir import pipeline
    from tensorforge.common.context import Context
    from tensorforge.generators.generator import Generator
    asked = []
    real = pipeline.sink_numbers

    def spy(body, **kwargs):
        asked.append(kwargs.get('threads'))
        return real(body, **kwargs)
    monkeypatch.setattr(pipeline, 'sink_numbers', spy)
    path = pathlib.Path(__file__).parent / 'cases' / 'mixed' / 'ml_then_ew.py'
    spec = importlib.util.spec_from_file_location('ml_then_ew', path)
    case = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(case)
    gen = Generator(case.descr_list(), Context(arch='sm_86', backend='cuda',
                                               fp_type=case.DTYPE))
    with contextlib.redirect_stdout(io.StringIO()):
        gen.generate()
    assert asked and all(asked)
