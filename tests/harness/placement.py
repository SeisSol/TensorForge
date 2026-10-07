# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""A hand-built body laid out the way the generator lays out a section's.

A shared buffer has no offset until the allocator gives it one, and the
emitter refuses one without (`pir.allocate`).  A test that builds a body by
hand and emits it goes through the same step, over the one arena its
builder allocates in.
"""

from tensorforge.backend.pir.allocate import MULT, allocate

#: The arena the tests' builders allocate in.
ARENA = 'shrMem'


def placed(body, arena: str = ARENA, align: int = 4):
    """`body` with every shared buffer of `arena` at an offset."""
    out, _ = allocate(tuple(body), arenas={arena: MULT}, align=align)
    return out
