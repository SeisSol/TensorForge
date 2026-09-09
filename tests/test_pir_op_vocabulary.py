# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""An op name pir does not know cannot pass for one it does.

Three places have to answer "what is this statement?" and each of them can
answer from a default: the fields on `Stmt`, which describe scalar arithmetic;
`IRBuilder.op`, which supplies those fields without being asked; and the
emitter, which can print any name as a call.  The three defaults agree with
each other and with nothing else, so a name outside the arithmetic vocabulary
gets treated as pure, movable, memory-free and callable --- four claims, none
of them examined, all of them acted on by the passes.

`Op.ARITH` is the vocabulary, and it is closed by what the emitter can spell
without a call.  That makes the question decidable at each of the three
points, which is what these tests hold.
"""

import pytest

from tensorforge.backend.pir.core import (Access, Effect, IRError, MemSpace,
                                          Op, ScalarType, Stmt, Value)
from tensorforge.common.basic_types import Datatype

F32 = ScalarType(Datatype.F32)


def _value(i=0):
    return Value(id=i, type=F32)


# -- the vocabulary itself -------------------------------------------------- #


def test_the_arithmetic_names_are_disjoint_from_the_structural_ones():
    """Two answers to "is this op known?" would be two answers to what it is.

    `Op.KNOWN` is derived from the constants on the class, so a name that
    appeared in both sets would be one whose meaning depends on which check
    ran first.
    """
    assert not (Op.ARITH & Op.KNOWN), (
        f'names in both vocabularies: {sorted(Op.ARITH & Op.KNOWN)}')


def test_every_arithmetic_name_has_a_spelling():
    """A name pir will build and the emitter cannot print is a late failure.

    The emitter's tables are the definition of what can be spelled without a
    function call, so `Op.ARITH` is exactly their union.  Read from the
    emitter rather than restated here: a copy would agree at the moment it was
    written and never afterwards.
    """
    from tensorforge.backend.pir.emit import _INFIX, _LEXIC_BINOP

    spellable = set(_INFIX) | set(_LEXIC_BINOP) | {'fma', 'select', 'neg'}
    assert set(Op.ARITH) == spellable, (
        f'buildable but not spellable: {sorted(set(Op.ARITH) - spellable)}; '
        f'spellable but not buildable: {sorted(spellable - set(Op.ARITH))}')


# -- Stmt ------------------------------------------------------------------- #


def test_an_unknown_op_may_not_take_the_permissive_defaults():
    with pytest.raises(IRError, match='must say what it does'):
        Stmt(op='spec.multilinear', target=(_value(),))


def test_an_unknown_op_may_not_look_harmless_by_being_impure_alone():
    """`pure=False` alone still leaves the statement free to be reordered.

    Purity governs CSE and DCE; movability with no declared accesses governs
    whether a pass may lift the statement past a store.  A high-level op that
    writes a buffer needs the second answer as much as the first, so denying
    only the first is not a declaration.
    """
    with pytest.raises(IRError, match='must say what it does'):
        Stmt(op='spec.multilinear', target=(_value(),), pure=False)


def test_an_unknown_op_that_declares_itself_is_accepted():
    """The check asks for a declaration, not for a particular one.

    An op set that could not grow would push every new operation into
    `rawstmt`, which is the opacity this vocabulary exists to reduce.
    """
    s = Stmt(op='spec.multilinear', target=(_value(),), pure=False,
             movable=False)
    assert s.op == 'spec.multilinear'

    acc = (Access(Effect.WRITE, MemSpace.SHARED, None),)
    s = Stmt(op='spec.multilinear', target=(_value(),), pure=False,
             effect=Effect.WRITE, accesses=acc)
    assert s.accesses == acc


def test_replacing_a_field_re_checks_the_statement():
    """`dataclasses.replace` re-runs `__post_init__`, so passes are covered.

    A pass that rewrites a declared statement back into a harmless one would
    otherwise slip past a check that only ran at construction.
    """
    from dataclasses import replace

    s = Stmt(op='spec.multilinear', target=(_value(),), pure=False,
             movable=False)
    with pytest.raises(IRError, match='must say what it does'):
        replace(s, pure=True)


def test_a_known_op_keeps_the_permissive_defaults():
    assert Stmt(op='add', target=(_value(),)).pure
    assert Stmt(op=Op.CONST, target=(_value(),)).pure


# -- IRBuilder.op ----------------------------------------------------------- #


def _builder():
    from tensorforge.backend.pir.build import IRBuilder

    return IRBuilder()


def test_the_builder_refuses_a_name_outside_the_arithmetic_vocabulary():
    b = _builder()
    x = b.const(1.0)
    with pytest.raises(IRError, match='not scalar arithmetic'):
        b.op('sqrtf', F32, x)


def test_the_builder_accepts_the_arithmetic_vocabulary():
    b = _builder()
    x = b.const(1.0)
    for name in sorted(Op.ARITH):
        assert b.op(name, F32, x, x) is not None


# -- the emitter ------------------------------------------------------------ #


def test_the_emitter_refuses_to_invent_a_callee():
    """An unspellable name used to become `name(args)`.

    C++ resolves that against whatever the translation unit has pulled in, so
    the failure is a link error at best and a silently different overload at
    worst --- neither of which names the statement that caused it.
    """
    from tensorforge.backend.pir.emit import Emitter

    b = _builder()
    x = b.const(1.0)
    body = b.finish()
    body = body + (Stmt(op='spec.multilinear', target=(_value(9_000),),
                        args=(x,), pure=False, movable=False),)
    with pytest.raises(IRError, match='no spelling for op'):
        Emitter([].append).run(body)


def test_a_split_names_its_callee_in_an_attribute():
    """The op name says what pir thinks it is; the attribute says what C++
    calls it.  Keeping them apart is what leaves `Op.SPLIT` decidable."""
    b = _builder()
    x = b.const(1.0)
    u32 = ScalarType(Datatype.U32)
    b.split_op('tensorforge::splitFloatTF32', (u32, u32), x, hints=('u', 'l'))
    stmt = b.finish()[-1]
    assert stmt.op == Op.SPLIT
    assert stmt.attr('callee') == 'tensorforge::splitFloatTF32'
