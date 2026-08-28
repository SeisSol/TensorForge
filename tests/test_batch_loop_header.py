# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Can `Op.FOR` render the loop the macro layer owns?

Making `BatchLoop` a PIR loop is the step that would let a transfer move to the
previous iteration -- `tools/macro_surface.py` measures why: the loop, its
induction variable and its lookahead bindings are the 2% of a kernel that no
body contains, so no pass can move anything across the back edge.

Before restructuring the generator around that, the cheap half is worth
checking on its own: does a PIR loop *emit* the header that is there today,
character for character?  Two things had to give.  The induction variable is
called `batchId0`, spelled out by the lookahead bindings, the flag guard and
every `access_address` in the body, so the IR cannot pick its own name.  And
its type is `size_t`, because it is compared against `numElements0`, where
`INDEX` renders to `int32_t`.

The expected string below is copied from a recorded snapshot rather than
written by hand, so it fails if either side moves.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tensorforge.backend.pir import emit, optimize, verify
from tensorforge.backend.pir.build import IRBuilder
from tensorforge.backend.pir.core import MemSpace
from tensorforge.backend.writer import Writer
from tensorforge.common.basic_types import Datatype
from tensorforge.common.vm.vm import vm_factory

SNAPSHOTS = Path(__file__).resolve().parent / "snapshots"

#: The header as the generator emits it today, for the persistent loop mode.
EXPECTED = ("for (size_t batchId0 = threadIdx.y + blockDim.y * (blockIdx.x); "
            "batchId0 < numElements0; "
            "batchId0 += (gridDim.x * blockDim.y)) {")


def test_the_expected_header_is_the_one_in_the_corpus():
    """Guard against the two sides drifting apart quietly."""
    hits = [line.strip()
            for path in SNAPSHOTS.glob("*.cuda.cpp")
            for line in path.read_text().splitlines()
            if "for (size_t batchId0" in line]
    assert hits, "no persistent batch loop in the recorded snapshots"
    assert EXPECTED in hits, (
        f"the generator's header changed; update EXPECTED.\n"
        f"  expected: {EXPECTED}\n"
        f"  found:    {sorted(set(hits))[:2]}")


def test_a_pir_loop_renders_that_header():
    b = IRBuilder(fptype=Datatype.F32)
    g = b.alloc(Datatype.F32, (16,), MemSpace.GLOBAL, hint="g")
    with b.for_("threadIdx.y + blockDim.y * (blockIdx.x)", "numElements0",
                "(gridDim.x * blockDim.y)",
                extern="batchId0", ctype="size_t"):
        b.store(g, 1.0, 0)
    body = b.finish()
    verify(body)

    w = Writer()
    emit(optimize(body), w, vm_factory("sm_86", "cuda", "float"))
    lines = [l.strip() for l in w.get_src().splitlines()]
    assert EXPECTED in lines, (
        "a PIR loop must be able to spell the header the macro layer emits, "
        f"or the migration changes generated code for no reason:\n"
        + "\n".join(lines))


def test_without_the_overrides_it_cannot():
    """The two overrides are load-bearing, not cosmetic."""
    b = IRBuilder(fptype=Datatype.F32)
    g = b.alloc(Datatype.F32, (16,), MemSpace.GLOBAL, hint="g")
    with b.for_("threadIdx.y + blockDim.y * (blockIdx.x)", "numElements0",
                "(gridDim.x * blockDim.y)"):
        b.store(g, 1.0, 0)
    body = b.finish()
    w = Writer()
    emit(optimize(body), w, vm_factory("sm_86", "cuda", "float"))
    src = w.get_src()
    assert "batchId0" not in src, src
    assert "size_t" not in src, src
