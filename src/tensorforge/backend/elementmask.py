# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""The element mask a global write runs under.

Its own module because neither of the two parties owns it: the batch loop
decides that a mask is needed and what it says, and `Symbol.store` is where
every global write passes through.  A stack rather than a field, for the same
reason `AbstractInstruction._induction_value` is one -- the region it covers is
a scope in the emission, not a property of any one object in it.
"""
from contextlib import contextmanager
from typing import Any, Optional, Tuple

#: Innermost last.  Each entry is `(value, name)`: a structured write takes the
#: mask as a predicate operand, a text write as the condition of a guard block,
#: and only one of the two spellings exists for any given store.
_STACK: list = []


@contextmanager
def element_mask(value: Any, name: str):
    """Run a region with every global write predicated on `name`.

    Set where several multiplications share a wave and therefore a barrier: the
    barrier has to be reached by every one of them, so the row whose element is
    out of range cannot be branched around.  It runs the body and writes
    nothing.

    Reads are deliberately untouched.  Whoever sets the mask clamps the row's
    index into range, so a masked-off row reads a valid element and computes an
    answer nobody keeps.  That costs the arithmetic and saves predicating a
    chain of loads, each of which would then have to answer what an
    unpredicated lane holds.

    Global memory only.  Shared memory and registers are the multiplication's
    own scratch, and what a row writes there is read back by that row alone.
    """
    _STACK.append((value, name))
    try:
        yield
    finally:
        _STACK.pop()


def active() -> Tuple[Optional[Any], Optional[str]]:
    """The innermost mask, or `(None, None)` outside any."""
    return _STACK[-1] if _STACK else (None, None)
