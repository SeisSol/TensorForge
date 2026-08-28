# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Inline assembly whose operands are values, not baked-in names.

`mma.sync` has no intrinsic, so the NVIDIA path emits PTX directly.  That does
not make it opaque.  What the statement reads and writes is exactly as
knowable as for any other vendor primitive --- the operands go in as values,
the ones with a read-write or write-only constraint go in as declared register
accesses, and two `mma.sync` calls on different accumulators then provably do
not conflict.

Before, the whole block was raw text with the operand names interpolated.
That had two consequences beyond the opacity count: the values it named had no
use edge, so whatever produced them was reachable by nothing and removable,
and the operand *numbering* was maintained by hand in two places at once.

Numbering is the part worth checking mechanically.  Outputs and inputs share
one sequence, `%0` onward, so folding the accumulator into a single `"+f"`
operand shifts every input down by the number of accumulator registers.  Get
that wrong and the assembler reads different registers than intended and the
kernel still compiles --- no warning, no crash, wrong numbers.
"""

from __future__ import annotations

import pytest

from tensorforge.backend.pir.build import IRBuilder
from tensorforge.backend.pir.core import (Effect, IRError, MemSpace, Op,
                                          ScalarType, walk)
from tensorforge.backend.pir.emit import Emitter
from tensorforge.common.basic_types import Datatype


def builder():
    return IRBuilder(fptype=Datatype.F32)


def emitted(body):
    lines = []
    Emitter(lines.append).run(body)
    return "\n".join(lines)


def test_the_operands_are_rendered_from_values():
    b = builder()
    c, x, y = (b.declare(hint=h) for h in 'cxy')
    b.asm_stmt('"mma "\n"{%0}, {%1}, {%2}, {%0};"',
               [('+f', c), ('r', x), ('r', y)])
    text = emitted(b.finish())
    assert f'"+f"({c})' in text
    assert f'"r"({x}), "r"({y})' in text


def test_a_written_operand_becomes_a_register_access():
    """The point of not being raw text.  `Effect.UNKNOWN` can never say that
    two of these do not conflict; a declared access on the accumulator can."""
    b = builder()
    c, x = b.declare(hint='c'), b.declare(hint='x')
    stmt = b.asm_stmt('"mma "\n"{%0}, {%1};"', [('+f', c), ('r', x)])
    assert stmt.effect == Effect.WRITE
    bases = {a.base for a in stmt.accesses}
    assert bases == {c}
    assert all(a.space == MemSpace.REGISTER for a in stmt.accesses)


def test_the_read_operands_are_uses():
    """Without a use edge the statement that produced a fragment is reachable
    by nothing, and `dce` is right to remove it."""
    b = builder()
    c, x = b.declare(hint='c'), b.declare(hint='x')
    stmt = b.asm_stmt('"mma "\n"{%0}, {%1};"', [('+f', c), ('r', x)])
    assert x in stmt.args and c in stmt.args


def test_a_numbering_mismatch_is_refused():
    """The failure that compiles cleanly and computes the wrong thing."""
    b = builder()
    c, x = b.declare(hint='c'), b.declare(hint='x')
    with pytest.raises(IRError, match="operands were given"):
        b.asm_stmt('"mma "\n"{%0}, {%1}, {%2};"', [('+f', c), ('r', x)])


def test_a_gap_in_the_numbering_is_refused():
    b = builder()
    c, x = b.declare(hint='c'), b.declare(hint='x')
    with pytest.raises(IRError, match="operands were given"):
        b.asm_stmt('"mma "\n"{%0}, {%2};"', [('+f', c), ('r', x)])


def test_an_output_after_an_input_is_refused():
    """Not a style rule: the assembler numbers outputs first, so an output
    listed later is numbered as though it were earlier."""
    b = builder()
    c, x = b.declare(hint='c'), b.declare(hint='x')
    with pytest.raises(IRError, match="outputs have to come first"):
        b.asm_stmt('"mma "\n"{%0}, {%1};"', [('r', x), ('+f', c)])


def test_a_written_operand_must_have_an_address():
    """A constraint that writes needs an lvalue; a literal or an expression
    would be written nowhere."""
    b = builder()
    with pytest.raises(IRError, match="not a value|address"):
        b.asm_stmt('"mma "\n"{%0};"', [('+f', 'v58[0][0]')])


def test_write_only_and_read_write_both_count_as_outputs():
    b = builder()
    d, c, x = (b.declare(hint=h) for h in 'dcx')
    stmt = b.asm_stmt('"mma "\n"{%0}, {%1}, {%2};"',
                      [('=f', d), ('+f', c), ('r', x)])
    assert {a.base for a in stmt.accesses} == {d, c}


def test_the_statement_is_pinned():
    """It is still assembly.  Nothing here can reason about the text, so it
    must not be hoisted or folded even though its memory effects are known."""
    b = builder()
    c = b.declare(hint='c')
    stmt = b.asm_stmt('"mma "\n"{%0};"', [('+f', c)])
    assert not stmt.movable
    assert not stmt.pure


def test_it_is_not_a_second_op():
    """`Op.CALL`, so the passes have one set of rules to know rather than two
    that must agree."""
    b = builder()
    c = b.declare(hint='c')
    b.asm_stmt('"mma "\n"{%0};"', [('+f', c)])
    ops = {s.op for s, _ in walk(b.finish())}
    assert Op.CALL in ops


# --------------------------------------------------------------------------- #
# The type the vendor signature forced
# --------------------------------------------------------------------------- #

def test_u32_exists_because_a_reference_parameter_demands_it():
    """`splitFloatTF32(uint32_t &, uint32_t &, float)`.

    An `I32` there does not bind to the reference and the kernel does not
    compile.  Nothing in the generator computes with this type.
    """
    assert Datatype.U32.ctype() == 'uint32_t'
    assert Datatype.U32.size() == Datatype.I32.size()
    assert Datatype.str2enum('uint32_t') is Datatype.U32
    assert Datatype.U32.literal(7) == '7u'


def test_a_u32_value_declares_as_uint32():
    b = builder()
    v = b.declare(ScalarType(Datatype.U32), hint='u')
    assert 'uint32_t' in emitted(b.finish())
