# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""NumPy reference implementations for :class:`MultilinearDescr`.

The generator's ``target`` convention: per operand, a list of ints where
a non-negative entry names an output axis and a negative entry names a
contraction axis shared between operands (same negative number = same
axis). ``permute`` is an intra-operand axis permutation applied before
contraction. ``GemmDescr`` with ``alpha != 1`` synthesizes an extra
scalar operand with ``target=[]``.

This maps cleanly onto :func:`numpy.einsum`: each operand gets a
subscript string whose letters are drawn from ``out_labels`` (for
non-negative targets) or ``contraction_labels`` (for negatives), with a
leading batch label ``'B'``.
"""

from __future__ import annotations

import string
from typing import Dict, List, Sequence, Tuple

import numpy as np


_OUT_LETTERS = string.ascii_lowercase[:20]      # a..t
_CON_LETTERS = string.ascii_lowercase[20:]      # u..z, extended below
_BATCH = "B"


def _labels_for(target_list: Sequence[Sequence[int]]) -> Tuple[List[str], int]:
    """Return a subscript per operand, plus the number of output axes."""
    out_rank = 0
    for t in target_list:
        for x in t:
            if x >= 0:
                out_rank = max(out_rank, x + 1)

    con_map: Dict[int, str] = {}
    def con_label(n: int) -> str:
        if n not in con_map:
            idx = len(con_map)
            if idx >= len(_CON_LETTERS):
                raise NotImplementedError("too many contraction axes for MVP")
            con_map[n] = _CON_LETTERS[idx]
        return con_map[n]

    subs: List[str] = []
    for t in target_list:
        parts = []
        for x in t:
            if x >= 0:
                parts.append(_OUT_LETTERS[x])
            else:
                parts.append(con_label(x))
        subs.append("".join(parts))
    return subs, out_rank


def multilinear_reference(
    target: Sequence[Sequence[int]],
    permute: Sequence[Sequence[int]],
    add: bool,
    operands: Sequence[np.ndarray],
    dest_in: np.ndarray,
) -> np.ndarray:
    """Evaluate a multilinear contraction.

    ``operands`` and ``dest_in`` carry a leading batch axis; each operand
    is ``(batch, *permuted_shape)`` when we enter. ``permute`` is applied
    here so callers can pass the raw per-element arrays.
    """
    assert len(operands) == len(target) == len(permute)

    # Apply permute (skip the batch axis).
    permuted = []
    for arr, p in zip(operands, permute):
        if len(p) == 0:                      # scalar operand
            permuted.append(arr)
        else:
            full = (0, *(int(x) + 1 for x in p))
            permuted.append(np.transpose(arr, full))

    subs, out_rank = _labels_for(target)

    lhs = ",".join(
        (_BATCH if s else "") + s for s in subs
    )
    out_sub = _BATCH + _OUT_LETTERS[:out_rank]

    # einsum broadcasts scalars automatically when their subscript is empty.
    result = np.einsum(f"{lhs}->{out_sub}", *permuted)

    if add:
        result = result + dest_in
    return result
