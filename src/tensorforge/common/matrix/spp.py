# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
from functools import reduce
import numpy as np

class SparsityPattern:
    """Which entries of a tensor are stored: `is_nz`, `count_nz`, and the
    `linear_index` that numbers them."""

class FullSPP(SparsityPattern):
    def __init__(self, shape, permute=None):
        self.shape = shape

    def is_nz(self, index):
        return all(i < s for i, s in zip(index, self.shape))

    def count_nz(self):
        return reduce(lambda x,y: x*y, self.shape, 1)

    def linear_index(self, tupleindex):
        stride = 1
        index = 0
        for x,s in zip(tupleindex, self.shape):
            index += x * stride
            stride *= s
        return index


class BoundingBoxSPP(SparsityPattern):
    def __init__(self, bbox):
        self.bbox = bbox

    def is_nz(self, index):
        return self.bbox.contains(index)

    def count_nz(self):
        return reduce(lambda x,y: x*y, self.bbox.sizes(), 1)

    def linear_index(self, tupleindex):
        stride = 1
        index = 0
        for x,l,u in zip(tupleindex, self.bbox.lower(), self.bbox.upper()):
            assert x >= l and x < u
            index += (x - l) * stride
            stride *= (u - l)
        return index


class MaskSPP(SparsityPattern):
    def __init__(self, mask):
        self.mask = mask

        self.indexmask = np.zeros(self.mask.shape, dtype=np.int64, order='F')

        # cf. e.g. https://numpy.org/doc/2.1/reference/arrays.nditer.html
        i = 0
        with np.nditer(self.indexmask, op_flags=['readwrite']) as it:
            for x,y in zip(np.nditer(self.mask), it):
                if x:
                    i += 1
                    y[...] = i
        #print(self.mask)
        #print(self.indexmask)

    def is_nz(self, index):
        return all(i < s for i, s in zip(index, self.mask.shape)) and self.mask[tuple(index)]

    def count_nz(self):
        return np.count_nonzero(self.mask)

    def linear_index(self, tupleindex):
        if all(ti < ims for ti, ims in zip(tupleindex, self.indexmask.shape)):
            preindex = self.indexmask[tupleindex]
            return preindex - 1 if preindex > 0 else None
        else:
            return None


class ListSPP(SparsityPattern):
    def __init__(self, lst, shape):
        self.list = lst

        self.indexmask = np.zeros(shape, dtype=np.int64, order='F')

        i = 0
        for entry in self.list:
            i += 1
            self.indexmask[tuple(entry)] = i

    def is_nz(self, index):
        return all(i < s for i, s in zip(index, self.indexmask.shape)) and self.indexmask[tuple(index)] > 0

    def count_nz(self):
        return np.count_nonzero(self.indexmask)

    def linear_index(self, tupleindex):
        if all(ti < ims for ti, ims in zip(tupleindex, self.indexmask.shape)):
            preindex = self.indexmask[tupleindex]
            return preindex - 1 if preindex > 0 else None
        else:
            return None


class NonzeroSPP(SparsityPattern):
    """Which entries of a tensor can be non-zero, as against which are stored.

    The other patterns in this file answer the *storage* question: an entry
    they admit occupies a slot, and `linear_index` says which.  This one
    answers the arithmetic question instead -- an operand densely stored over
    its bounding box still has entries that are zero in every batch element,
    and a product reading one of them contributes nothing.

    The two are separate because a tensor has both at once and they disagree:
    SeisSol's `kDivMT` is stored dense over a 40x53 box and 14 % of it is
    non-zero.  Keeping them in one field would force a choice between
    addressing the operand wrongly and counting it wrongly.

    `linear_index` is refused rather than answered, because every sensible
    answer is a trap: numbering the non-zeros would address a densely stored
    operand as though the zeros had been squeezed out, and deferring to the
    storage pattern would hand out an address this object has no business
    knowing.
    """

    def __init__(self, mask):
        #: Truth per entry, in the tensor's own index space.
        self.mask = np.asarray(mask, dtype=bool)

    def is_nz(self, index):
        index = tuple(index)
        return bool(all(i < s for i, s in zip(index, self.mask.shape))
                    and self.mask[index])

    def count_nz(self):
        return int(np.count_nonzero(self.mask))

    def linear_index(self, tupleindex):
        raise NotImplementedError(
            'a non-zero pattern says which entries matter, not where they '
            'live; ask the tensor\'s storage pattern for an address')
