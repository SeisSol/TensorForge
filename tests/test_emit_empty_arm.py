# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""An `if` whose first arm writes nothing.

The writer leaves out a block nothing was written into, its head with it
(`Writer.speculate`).  Behind a first arm left out that way the second was
written as `else { ... }`, an `else` of nothing, which does not compile.
The emitter writes it under the condition negated instead (`Emitter.
_emit_if`), and a block says whether it was left out (`Block.elided`).
"""

from __future__ import annotations

from tensorforge.backend.pir.build import IRBuilder
from tensorforge.backend.pir.core import BOOL, MemSpace
from tensorforge.backend.pir.emit import Emitter
from tensorforge.backend.writer import Writer
from tensorforge.common.basic_types import Datatype
from tensorforge.common.context import Context


def _source(then_writes: bool, else_writes: bool) -> str:
    ctx = Context(arch='sm_86', backend='cuda', fp_type=Datatype.F32)
    b = IRBuilder(fptype=Datatype.F32, context=ctx)
    out = b.alloc(Datatype.F32, (4,), MemSpace.GLOBAL, hint='o', extern='m0')
    lane = b.thread_id('x')
    cond = b.op('lt', BOOL, lane, 3, hint='c')
    guard = b.if_else(cond, types=())
    with guard.then():
        if then_writes:
            b.store(out, 1.0, 0)
    with guard.otherwise():
        if else_writes:
            b.store(out, 2.0, 1)
    writer = Writer()
    Emitter(writer, ctx).run(b.finish())
    return writer.get_src()


def _lines(src):
    return [line.strip() for line in src.splitlines() if line.strip()]


def test_a_first_arm_that_writes_nothing_leaves_the_second_under_the_negation():
    lines = _lines(_source(False, True))
    assert 'else {' not in lines and 'else' not in lines
    at = lines.index('if (!(threadIdx.x < 3)) {')
    assert lines[at + 1] == 'm0[1] = 2.0;' and lines[at + 2] == '}'


def test_both_arms_are_written_as_they_were():
    lines = _lines(_source(True, True))
    at = lines.index('if (threadIdx.x < 3) {')
    assert lines[at + 1:at + 6] == ['m0[0] = 1.0;', '}', 'else {',
                                    'm0[1] = 2.0;', '}']


def test_a_second_arm_that_writes_nothing_is_left_out():
    lines = _lines(_source(True, False))
    at = lines.index('if (threadIdx.x < 3) {')
    assert lines[at + 1:at + 3] == ['m0[0] = 1.0;', '}']
    assert not any(line.startswith('else') for line in lines)


def test_two_arms_that_write_nothing_leave_nothing():
    lines = _lines(_source(False, False))
    assert not any(line.startswith(('if', 'else')) for line in lines)


def test_a_block_says_whether_it_was_left_out():
    writer = Writer()
    empty = writer.Block('if (a)')
    with empty:
        pass
    full = writer.Block('if (b)')
    with full:
        writer('x = 1;')
    assert empty.elided and not full.elided
    assert writer.get_src() == 'if (b) {\n  x = 1;\n}\n'
