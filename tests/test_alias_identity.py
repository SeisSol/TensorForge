# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""One buffer, one alias identity.

`may_alias` compares bases with `is`, so a buffer that reaches the access
model as two objects is two buffers as far as every pass is concerned: a
write through one is invisible to a read through the other.

`Symbol` has exactly that shape.  The structured path records against the
value (`load`/`store` -> `alias_root(buf)`), the text path against the symbol
(`load_expr`/`access_stmt`).  Nothing in the corpus takes both paths for one
symbol in one body today; the gate that keeps it that way is `vec == 1` in
`Symbol.load_linear`, and the vector widths in `memory/load.py` are commented
out rather than absent.  This pins the invariant before that changes.
"""
from tensorforge.backend.pir import (Effect, IRBuilder, MemSpace, ScalarType,
                                     load_cse)
from tensorforge.backend.symbol import Symbol, SymbolType
from tensorforge.common.basic_types import Datatype

F32 = ScalarType(Datatype.F32)


def _body(text_base_is_symbol: bool):
    b = IRBuilder(fptype=Datatype.F32, scratch=('tempShrMem', 64))
    buf = b.alloc(Datatype.F32, (64,), MemSpace.SHARED, hint='A')

    sym = Symbol('A', SymbolType.SharedMem, None)
    sym.set_pir_buffer(b, buf)          # what AllocateInstruction does

    base = sym if text_base_is_symbol else buf
    r1 = b.load_expr('A[0]', F32, base, hint='r1')   # the vec > 1 path
    b.store(buf, b.const(1.0), 0)                    # the vec == 1 path
    r2 = b.load_expr('A[0]', F32, base, hint='r2')
    b.op('add', F32, r1, r2, hint='use')
    return b.finish()


def _raw_reads(body):
    return sum(1 for s in body if str(s.op) == 'rawexpr')


def test_a_store_kills_a_raw_read_of_the_same_buffer():
    for through_symbol in (False, True):
        body = _body(through_symbol)
        assert _raw_reads(body) == 2
        assert _raw_reads(load_cse(body)) == 2, (
            'the second read was folded onto the first across a store to the '
            'same buffer; the symbol and its value are not one identity')


def test_symbol_and_its_value_resolve_to_one_base():
    b = IRBuilder(fptype=Datatype.F32, scratch=('tempShrMem', 64))
    buf = b.alloc(Datatype.F32, (64,), MemSpace.SHARED, hint='A')
    sym = Symbol('A', SymbolType.SharedMem, None)
    sym.set_pir_buffer(b, buf)
    assert b.alias_root(sym) is b.alias_root(buf)


def test_a_value_from_another_body_leaves_the_symbol_alone():
    """`pir_buffer` is body-scoped, and that is the wanted answer: in a body
    the value does not belong to, the symbol is the only identity there is."""
    b1 = IRBuilder(fptype=Datatype.F32, scratch=('tempShrMem', 64))
    buf = b1.alloc(Datatype.F32, (64,), MemSpace.SHARED, hint='A')
    sym = Symbol('A', SymbolType.SharedMem, None)
    sym.set_pir_buffer(b1, buf)

    b2 = IRBuilder(fptype=Datatype.F32, scratch=('tempShrMem', 64))
    assert b2.alias_root(sym) is sym
