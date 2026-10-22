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
from tensorforge.backend.pir import IRBuilder, MemSpace, ScalarType, load_cse
from tensorforge.backend.symbol import Symbol, SymbolType
from tensorforge.common.basic_types import Datatype

F32 = ScalarType(Datatype.F32)


def _body(text_base_is_symbol: bool):
    b = IRBuilder(fptype=Datatype.F32, arena='shrMem')
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
    b = IRBuilder(fptype=Datatype.F32, arena='shrMem')
    buf = b.alloc(Datatype.F32, (64,), MemSpace.SHARED, hint='A')
    sym = Symbol('A', SymbolType.SharedMem, None)
    sym.set_pir_buffer(b, buf)
    assert b.alias_root(sym) is b.alias_root(buf)


def test_a_value_from_another_body_leaves_the_symbol_alone():
    """`pir_buffer` is body-scoped, and that is the wanted answer: in a body
    the value does not belong to, the symbol is the only identity there is."""
    b1 = IRBuilder(fptype=Datatype.F32, arena='shrMem')
    buf = b1.alloc(Datatype.F32, (64,), MemSpace.SHARED, hint='A')
    sym = Symbol('A', SymbolType.SharedMem, None)
    sym.set_pir_buffer(b1, buf)

    b2 = IRBuilder(fptype=Datatype.F32, arena='shrMem')
    assert b2.alias_root(sym) is sym


def _register_reads(through_symbol, through_value, *, store=None):
    """A read of register `r5` through its symbol, as the builder spells
    one, then a read through the value that declares it, as the forwarding
    spells one (`forward._Forward._register`) -- with a store between, where
    asked."""
    b = IRBuilder(fptype=Datatype.F32)
    buf = b.alloc(Datatype.F32, (9,), MemSpace.REGISTER, hint='r5')
    sym = Symbol('r5', SymbolType.Register, None)
    sym.set_pir_buffer(b, buf)
    r1 = b.load(sym, *through_symbol, hint='r1')
    if store is not None:
        b.store(buf, b.const(1.0), store)
    r2 = b.load(buf, *through_value, hint='r2')
    b.op('add', F32, r1, r2, hint='use')
    return b.finish()


def _loads(body):
    return sum(1 for s in body if str(s.op) == 'load')


def test_a_register_read_through_its_symbol_or_its_value_is_one_read():
    """Both are `r5[3]`: the second read is the first."""
    assert _loads(_register_reads((3,), (3,))) == 2
    assert _loads(load_cse(_register_reads((3,), (3,)))) == 1


def test_the_reads_of_a_register_stay_two_where_they_are_two():
    """Another element, a store to the element between, or an index of more
    than one term -- which the symbol's view and the buffer's shape need not
    linearize alike (`Emitter.address`)."""
    assert _loads(load_cse(_register_reads((3,), (4,)))) == 2
    assert _loads(load_cse(_register_reads((3,), (3,), store=3))) == 2
    assert _loads(load_cse(_register_reads((1, 1), (1, 1)))) == 2
