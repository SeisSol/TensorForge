from typing import Union, List

class BoundingBox:
    def __init__(self, lower: List[int], upper: List[int]):
        assert len(lower) == len(upper)
        self._lower = lower
        self._upper = upper
    
    def __eq__(self, other):
        return self._lower == other._lower and self._upper == other._upper
    
    def rank(self):
        return len(self._lower)

    def contains(self, index):
        return all(i >= l and i < u for i,l,u in zip(index, self._lower, self_upper))
    
    def empty(self):
        return all(u >= l for l,u in zip(self._lower, self._upper))

    def lower(self):
        return self._lower

    def upper(self):
        return self._upper
    
    def size(self, dim):
        return self._upper[dim] - self._lower[dim]
    
    def sizes(self):
        return [u - l for l,u in zip(self._lower, self._upper)]

    def __str__(self):
        return '×'.join(f'{{{l}..{u}}}' for l,u in zip(self._lower, self._upper))
