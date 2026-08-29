# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""`flatten_scopes` keeps the braces a raw declaration needs.

The pass splices away anonymous `{ }` regions that cannot cause a
redeclaration, which is worth doing: an opaque block head makes the async
scheduler give up its state and nothing reorders across one.  Whether a region
*can* cause one is decided by `_CDECL`, a regex over the raw text.

It required `=`, `;` or `[` immediately after the declared name, so it did not
recognise brace initialisation (`float x{};`) or a second declarator
(`uint32_t a, b;`).  Ten percent of the declarations the corpus emits take one
of those two forms.

Nothing had broken, because the misses were masked.  The NVIDIA accumulator
was declared as `float v58[4][2]{};`, which matches on the `[`, so the region
holding it kept its braces and the `float v58_0{};` beside it survived by
association.  Making the accumulator a structured value removed the match, the
braces went with it, and six declarations of one name ended up in one block --
in a kernel that had compiled the day before.

That is the shape to guard: not "does the pass work", but "does its predicate
see every form the generator actually emits".  Over-matching costs a pair of
braces that could have been removed; under-matching costs a kernel that does
not compile.
"""

from __future__ import annotations

import pytest

from tensorforge.backend.pir import passes
from tensorforge.backend.pir.build import IRBuilder
from tensorforge.backend.pir.core import Op, walk
from tensorforge.backend.pir.emit import Emitter
from tensorforge.common.basic_types import Datatype


@pytest.mark.parametrize("text", [
    "float x = y;",
    "float x;",
    "float x[4];",
    "float x[4][2]{};",
    "float x{};",                       # brace initialisation
    "uint32_t a, b;",                   # two declarators
    "const float x = y;",
    "int32_t v11_a = -4_i32 + 0;",
    "auto x = y;",
])
def test_a_declaration_is_recognised(text):
    assert passes._CDECL.search(text), (
        f"{text!r} declares a name; a region containing it must keep its "
        f"braces")


@pytest.mark.parametrize("text", [
    "return floaty;",
    "x = y;",
    "__syncwarp();",
    "tensorforge::splitFloatTF32(a, b, c);",
    "s0[v295_a] = v286_data;",
])
def test_a_non_declaration_is_not(text):
    assert not passes._CDECL.search(text), (
        f"{text!r} declares nothing; treating it as a declaration keeps "
        f"braces that cost the async scheduler its state")


def _region_with(*texts):
    b = IRBuilder(fptype=Datatype.F32)
    with b.AnonymousScope():
        for t in texts:
            b(t)
    return b.finish()


def _has_scope(body):
    return any(s.op == Op.RAWBLOCK for s, _ in walk(body))


def test_a_region_that_declares_keeps_its_braces():
    for text in ("float x{};", "uint32_t a, b;", "float x[4][2]{};"):
        body = passes.flatten_scopes(_region_with(text))
        assert _has_scope(body), f"braces dropped around {text!r}"


def test_a_region_that_declares_nothing_is_spliced():
    body = passes.flatten_scopes(_region_with("__syncwarp();"))
    assert not _has_scope(body)


def test_two_sibling_regions_declaring_one_name_stay_separate():
    """The failure this prevents, end to end.

    Two blocks each declaring `v0_c`.  Splice both and the two declarations
    land at one level, which is the redeclaration the braces were preventing.
    """
    b = IRBuilder(fptype=Datatype.F32)
    for _ in range(2):
        with b.AnonymousScope():
            b("float v0_c{};")
    body = passes.flatten_scopes(b.finish())
    blocks = [s for s, _ in walk(body) if s.op == Op.RAWBLOCK]
    assert len(blocks) == 2, (
        f"{len(blocks)} of the two regions kept their braces; both "
        f"declarations would be emitted at one level")


def test_the_masking_that_hid_it():
    """A region declaring an array *and* a brace-initialised name.

    The array matched on `[` and carried the region; the other declaration was
    never seen.  Removing the array is what exposed the miss, so a region with
    only the second form has to keep its braces on its own account.
    """
    with_array = passes.flatten_scopes(
        _region_with("float v0[4][2]{};", "float v0_c{};"))
    without = passes.flatten_scopes(_region_with("float v0_c{};"))
    assert _has_scope(with_array)
    assert _has_scope(without), (
        "the region survived only because of the array beside it")


def test_a_transfer_opens_no_scope_unless_its_buffer_rotates():
    """The braces exist for one thing, so they are opened for one thing.

    `MemoryInstruction.gen_ir` wrapped every transfer body in `{ }` so that a
    rotating buffer's write-side alias could not clash with the consumer's
    pointer of the same name.  Nothing else it emits can clash -- the
    temporaries are values the shared allocator numbers -- and the brace is not
    free: an opaque block head is a wall the async scheduler gives up its state
    at and nothing reorders across, sitting in exactly the stretch `WrapLoads`
    wants to move a transfer along.

    `flatten_scopes` removed the ones that declared nothing, so this changed no
    emitted source.  It changed 594 blocking nodes into none, at build time,
    where the passes that matter run.
    """
    import inspect

    from tensorforge.backend.instructions import memory

    src = inspect.getsource(memory.MemoryInstruction.gen_ir)
    assert 'rotates' in src, (
        'the scope is opened unconditionally again; it is a wall for every '
        'transfer, not only the rotating ones')
    assert src.count('sink.Scope()') == 1
