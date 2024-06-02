from typing import Union, List

class BoundingBox:
    def __init__(self, lower: List[int], upper: List[int]):
        assert len(lower) == len(upper)
        self._lower = tuple(lower)
        self._upper = tuple(upper)
    
    def intersect(self, other):
        return BoundingBox([max(sl, ol) for sl, ol in zip(self.lower, other.lower)], [min(sl, ol) for sl, ol in zip(self.upper, other.upper)])

    def unite(self, other):
        return BoundingBox([min(sl, ol) for sl, ol in zip(self.lower, other.lower)], [max(sl, ol) for sl, ol in zip(self.upper, other.upper)])

    def __rand__(self, other):
        return self.intersect(other)

    def __ror__(self, other):
        return self.unite(other)

    def __iter__(self):
        for l,u in zip(self._lower, self._upper):
            yield l,u

    def __eq__(self, other):
        return self._lower == other._lower and self._upper == other._upper
    
    def rank(self):
        return len(self._lower)

    def contains(self, index):
        return all(i >= l and i < u for i,l,u in zip(index, self._lower, self._upper))
    
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
