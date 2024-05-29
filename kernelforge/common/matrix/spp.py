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

class BoundingBoxSPP(SparsityPattern):
    pass

class IndexedSPP(SparsityPattern):
    def __init__(self, indices):
        self.indices = indices
    
    def is_nz(self, index):
        return tuple(index) in self.indices
    
    def count_nz(self):
        return len(self.indices)

class MaskSPP(SparsityPattern):
    def __init__(self, mask):
        self.mask = mask
    
    def is_nz(self, index):
        return self.mask[tuple(index)]
    
    def count_nz(self):
        return np.count_nonzero(self.mask)
