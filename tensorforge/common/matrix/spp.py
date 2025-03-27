from functools import reduce
import numpy as np

class SparsityPattern:
    @classmethod
    def from_values(cls, data, threshold=1):
        pass

class FullSPP(SparsityPattern):
    def __init__(self, shape, permute=None):
        self.shape = shape

    def is_nz(self, index):
        return True
    
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
    pass

class MaskSPP(SparsityPattern):
    def __init__(self, mask):
        self.mask = mask

        self.indexmask = np.zeros(self.mask.shape, dtype=np.int64)

        # cf. e.g. https://numpy.org/doc/2.1/reference/arrays.nditer.html
        i = 0
        with np.nditer(self.indexmask, op_flags=['readwrite']) as it:
            for x,y in zip(np.nditer(self.mask), it):
                if x:
                    i += 1
                    y[...] = i
    
    def is_nz(self, index):
        return self.mask[tuple(index)]
    
    def count_nz(self):
        return np.count_nonzero(self.mask)
    
    def linear_index(self, tupleindex):
        preindex = self.indexmask[tupleindex]
        return preindex - 1 if preindex > 0 else None
