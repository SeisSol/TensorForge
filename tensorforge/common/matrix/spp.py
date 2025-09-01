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
        return self.mask[tuple(index)]
    
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
        return self.indexmask[tuple(index)] > 0
    
    def count_nz(self):
        return np.count_nonzero(self.indexmask)
    
    def linear_index(self, tupleindex):
        if all(ti < ims for ti, ims in zip(tupleindex, self.indexmask.shape)):
            preindex = self.indexmask[tupleindex]
            return preindex - 1 if preindex > 0 else None
        else:
            return None
