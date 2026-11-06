# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""The operator product ahead of the small product it feeds
(`Options.reassociate_products`).

yateto writes SeisSol's kernels with the per-element product first:

    T = X S          X 56x9 per element, S 9x9 per element (a star matrix)
    Q (+)= A T       A the operator: 56x56, 24x56 for a face

which is ``Q (+)= (A X) S`` as well.  In the second order:

* the small product runs over the rows the operator has rather than the rows
  of X -- 24 behind a face matrix instead of 56, and behind the derivative's
  ``kDivMT`` only the rows of its box;
* the operator products of one step all read the same X, the element's data,
  where in the first each reads a T of its own -- which is what lets a kernel
  share the operand between them.

The number of products is the same and so is the operator's work: ``A`` is
multiplied with a matrix of as many columns either way.

Taken where it is a plain matrix product on both sides -- two operands, the
contraction over the inner axis, no permutation -- T a temporary that one
later product reads and nothing else touches, and only where the small
product does not grow: the operator has no more rows in its box than X has.
That last condition also keeps a chain from being taken twice the wrong way
round: behind a face matrix, ``rDivM (U S)`` would become ``(rDivM U) S``,
the small product back on 56 rows.
"""

from typing import List, Optional, Sequence

from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor

from .descriptions import MultilinearDescr

#: ``C[i, j] = sum_k L[i, k] R[k, j]`` in the descriptions' notation.
MATMUL = [[0, -1], [-1, 1]]
IDENTITY = [[0, 1], [0, 1]]


def _is_matmul(d) -> bool:
    return (isinstance(d, MultilinearDescr) and len(d.ops) == 2
            and [list(t) for t in d.target] == MATMUL
            and [list(p) for p in d.permute] == IDENTITY
            and d.add_dims in (None, [0, 1])
            and len(d.dest.tensor.shape) == 2
            and all(len(o.tensor.shape) == 2 for o in d.ops)
            and not d.dest.sliced and not any(o.sliced for o in d.ops))


def _touches(d, tensor) -> bool:
    reads = getattr(d, 'reads', None)
    writes = getattr(d, 'writes', None)
    if reads is None or writes is None:
        # Something that does not say what it touches touches everything.
        return True
    w = writes()
    return (any(getattr(r, 'tensor', r) is tensor for r in (reads() or ()))
            or (w is not None and getattr(w, 'tensor', w) is tensor))


def _writes(d, tensor) -> bool:
    writes = getattr(d, 'writes', None)
    if writes is None:
        return True
    w = writes()
    return w is not None and getattr(w, 'tensor', w) is tensor


def _rows(view: SubTensor):
    box = view.bbox
    return box.lower()[0], box.upper()[0]


def _pair(descrs: Sequence, i: int) -> Optional[int]:
    """The index of the product the product at `i` feeds, where the two may
    be reassociated."""
    first = descrs[i]
    if not _is_matmul(first) or first.add or not first.dest.tensor.is_tmp:
        return None
    t = first.dest.tensor
    users = [j for j in range(i + 1, len(descrs)) if _touches(descrs[j], t)]
    if len(users) != 1:
        return None
    j = users[0]
    second = descrs[j]
    if (not _is_matmul(second) or second.ops[1].tensor is not t
            or second.ops[0].tensor is t or second.dest.tensor is t):
        return None
    x, s = first.ops
    a = second.ops[0]
    if any(_writes(descrs[k], v.tensor) for k in range(i + 1, j)
           for v in (x, s, a)):
        return None
    # The small product must not grow: rows of A's box against rows of X's.
    a_lo, a_hi = _rows(a)
    x_lo, x_hi = _rows(x)
    if a_hi - a_lo > x_hi - x_lo:
        return None
    return j


def _temporary(like: Tensor, rows: int, cols: int, box: BoundingBox,
               alias: str) -> Tensor:
    return Tensor([rows, cols], like.addressing, box, alias=alias,
                  is_tmp=True, datatype=like.datatype,
                  alignment=like.alignment, residence=like.residence)


def reassociate(descrs: Sequence) -> List:
    """`descrs` with every reassociable pair rewritten, in place of the two
    it replaces; the rest as it was."""
    out = list(descrs)
    for i in range(len(out)):
        j = _pair(out, i)
        if j is None:
            continue
        first, second = out[i], out[j]
        x, s = first.ops
        a = second.ops[0]
        t = first.dest.tensor
        a_lo, a_hi = _rows(a)
        x_box = x.bbox
        box = BoundingBox([a_lo, x_box.lower()[1]], [a_hi, x_box.upper()[1]])
        u = _temporary(t, a.tensor.shape[0], x.tensor.shape[1], box,
                       f'{t.alias}_ax')
        out[i] = MultilinearDescr(dest=SubTensor(u),
                                  ops=[SubTensor(a.tensor, a.bbox, a.offset),
                                       SubTensor(x.tensor, x.bbox, x.offset)],
                                  target=[list(r) for r in MATMUL],
                                  permute=[list(r) for r in IDENTITY],
                                  add=False)
        out[j] = MultilinearDescr(dest=second.dest,
                                  ops=[SubTensor(u),
                                       SubTensor(s.tensor, s.bbox, s.offset)],
                                  target=[list(r) for r in MATMUL],
                                  permute=[list(r) for r in IDENTITY],
                                  add=(list(second.add_dims)
                                       if second.add_dims is not None
                                       else second.add))
    return out
