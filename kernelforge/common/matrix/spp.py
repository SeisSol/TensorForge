from functools import reduce

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

class IndexedSPP(SparsityPattern):
    def __init__(self, indices):
        self.indices = indices
    
    def is_nz(self, index):
        return tuple(index) in self.indices
    
    def count_nz(self):
        return len(self.indices)
