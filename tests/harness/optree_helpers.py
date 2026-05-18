# SPDX-License-Identifier: MIT
"""Minimal helpers for constructing optree :class:`TensorVar`\\ s in tests.

The yateto frontend builds :class:`tensorforge.generators.optree.TensorVar`
objects via an *assigner* callback (see ``yateto.py:add_scalar``): each
``TensorVar`` is first constructed with only a ``pretensor`` placeholder,
then ``Assignment.assignTensor(assigner)`` is called, which resolves the
placeholder to an actual ``(SubTensor, indices)`` pair.

In the test harness we already *have* the :class:`SubTensor`\\ s in hand —
each case constructs them directly. There is no intermediate yateto-side
representation to look up. So we bypass the assigner indirection entirely
and build TensorVars whose ``.tensor``, ``.indices``, and ``.offset``
fields are populated up front. ``ElementwiseInstruction`` only ever calls
``assignSymbols`` / ``getRanges`` afterward, both of which read these
fields directly (cf.\\ ``optree.py:TensorVar.assignSymbols`` and
``TensorVar.getRanges``).

This module also picks a sensible index convention. ``ElementwiseInstruction``
expects ranges keyed by negative integers ``-1, -2, …`` (see
``elementwise.py:_assignment_loop`` which iterates ``-i-1``), so for a
rank-``r`` tensor we hand it indices ``[-1, -2, …, -r]``.
"""

from __future__ import annotations

from typing import List

from tensorforge.common.matrix.tensor import SubTensor
from tensorforge.generators import optree


def make_tvar(subtensor: SubTensor, ndims: int) -> optree.TensorVar:
    """Build a fully-resolved :class:`TensorVar` over ``subtensor``.

    ``ndims`` is the rank of the tensor as seen by the elementwise loop
    nest. It must match ``subtensor.bbox.rank()``; this is asserted
    rather than inferred so a mismatch shows up as a clear test-side
    error instead of as a confusing ``IndexError`` deep in the writer.

    The returned :class:`TensorVar` is in the state the rest of the
    pipeline expects right after ``Assignment.assignTensor`` would have
    fired in the yateto frontend.
    """
    assert ndims == subtensor.bbox.rank(), (
        f"ndims={ndims} does not match subtensor.bbox.rank()={subtensor.bbox.rank()}"
    )
    tv = optree.TensorVar(subtensor, slicing=None, pretensor=None)
    # Negative indices: ElementwiseInstruction's range map is keyed by
    # ``-i-1`` (see _assignment_loop). The same convention shows up in
    # TensorVar.write for index emission.
    tv.indices = [-(i + 1) for i in range(ndims)]
    tv.offset = list(subtensor.bbox.lower())
    return tv


def make_tvars(subtensors: List[SubTensor]) -> List[optree.TensorVar]:
    """Convenience: build TensorVars for a list of SubTensors (rank inferred)."""
    return [make_tvar(st, st.bbox.rank()) for st in subtensors]
