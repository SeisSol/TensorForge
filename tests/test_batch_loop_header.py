# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""`Op.FOR` renders the batch loop's header.

The batch loop is a loop of the section's body -- which is what lets a pass
move a transfer across its back edge (`pir/wrap.py`) -- and its header has to
come out the way the recorded kernels spell it.  Two things have to give.
The induction variable carries the name `batchId0` the rest of the kernel is
read by, so the IR cannot pick one unrelated to it -- it names the value from
that hint and its own number, `v8_batchId0`, which is the one difference
between the two sides and the one `_same_but_for_the_number` takes out.  And
its type is `size_t`, because it is compared against `numElements0`, where
`INDEX` renders to `int32_t`.

The expected string below is copied from a recorded snapshot rather than
written by hand, so it fails if either side moves.
"""

from __future__ import annotations

import re
from pathlib import Path

from tensorforge.backend.pir import emit, optimize, verify
from tensorforge.backend.pir.build import IRBuilder
from tensorforge.backend.pir.core import SIZE, MemSpace
from tensorforge.backend.writer import Writer
from tensorforge.common.basic_types import Datatype
from tensorforge.common.target import Target

SNAPSHOTS = Path(__file__).resolve().parent / "snapshots"

#: The header the generator emits for the persistent loop mode, with the
#: value's number taken out of the induction variable's name.
EXPECTED = ("for (size_t batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); "
            "batchId0 < numElements0; "
            "batchId0 += (gridDim.x * blockDim.y)) {")


def _same_but_for_the_number(line: str) -> str:
    """`v8_batchId0` and `v1329_batchId0` are the same variable.

    The number is the value's, and a snapshot of another kernel has another
    one; everything else in the header is what this test is about.
    """
    return re.sub(r"\bv\d+_batchId0\b", "batchId0", line.strip())


def test_the_expected_header_is_the_one_in_the_corpus():
    """Guard against the two sides drifting apart quietly."""
    hits = [_same_but_for_the_number(line)
            for path in SNAPSHOTS.glob("*.cuda.cpp")
            for line in path.read_text().splitlines()
            if re.search(r"for \(size_t v?\d*_?batchId0", line)]
    assert hits, "no persistent batch loop in the recorded snapshots"
    assert EXPECTED in hits, (
        f"the generator's header changed; update EXPECTED.\n"
        f"  expected: {EXPECTED}\n"
        f"  found:    {sorted(set(hits))[:2]}")


def test_a_pir_loop_renders_that_header():
    b = IRBuilder(fptype=Datatype.F32)
    g = b.alloc(Datatype.F32, (16,), MemSpace.GLOBAL, hint="g")
    # Parenthesised as the generator writes it, which is the other half of
    # emitting the same header: the expression is the caller's text, and so
    # is its precedence.
    with b.for_("(threadIdx.y + blockDim.y * (blockIdx.x))", "numElements0",
                "(gridDim.x * blockDim.y)",
                extern="batchId0", index_type=SIZE):
        b.store(g, 1.0, 0)
    body = b.finish()
    verify(body)

    w = Writer()
    emit(optimize(body), w, Target("sm_86", "cuda"))
    lines = [_same_but_for_the_number(l) for l in w.get_src().splitlines()]
    assert EXPECTED in lines, (
        "the PIR loop no longer spells the header the snapshots record:\n"
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
    emit(optimize(body), w, Target("sm_86", "cuda"))
    src = w.get_src()
    assert "batchId0" not in src, src
    assert "size_t" not in src, src
