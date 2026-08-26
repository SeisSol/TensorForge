# SPDX-FileCopyrightText: 2015 SeisSol Group
#
# SPDX-License-Identifier: MIT

"""Insertion-ordered set.

Why this exists: several IR analyses accumulate sets of ``Symbol`` /
``Vertex`` objects and then *iterate* them to derive code -- register
allocation, memory-region colouring, barrier placement.  Neither class
overrides ``__hash__``, so a builtin ``set`` orders them by ``id()``,
i.e. by heap address.  Iteration order then varies between runs and the
emitted source is not reproducible.

Observed on ``tests/cases/chain_five.py`` before this was introduced: two
consecutive generations in one process assigned the two live shared-memory
buffers *swapped* offsets (0 / 608) **and** placed the ``__syncwarp()``
at different points, because barrier insertion keys on region membership.
A synchronisation decision that depends on heap addresses is a race that
appears and disappears between builds.

Ordering by first insertion makes every such decision a function of
program order alone.  ``dict`` already provides that, so this is a thin
facade over ``dict`` keys rather than a real data structure.
"""

from __future__ import annotations

from typing import Dict, Generic, Iterable, Iterator, Optional, TypeVar

T = TypeVar('T')


class OrderedSet(Generic[T]):
    """Set semantics, insertion-ordered iteration.  ``O(1)`` membership."""

    __slots__ = ('_d',)

    def __init__(self, items: Optional[Iterable[T]] = None):
        self._d: Dict[T, None] = {}
        if items is not None:
            for item in items:
                self._d[item] = None

    # -- set protocol ------------------------------------------------------ #

    def add(self, item: T) -> None:
        self._d[item] = None

    def discard(self, item: T) -> None:
        self._d.pop(item, None)

    def remove(self, item: T) -> None:
        del self._d[item]

    def update(self, items: Iterable[T]) -> None:
        for item in items:
            self._d[item] = None

    def __contains__(self, item) -> bool:
        return item in self._d

    def __iter__(self) -> Iterator[T]:
        return iter(self._d)

    def __len__(self) -> int:
        return len(self._d)

    def __bool__(self) -> bool:
        return bool(self._d)

    # -- algebra ----------------------------------------------------------- #

    def copy(self) -> 'OrderedSet[T]':
        new = OrderedSet.__new__(OrderedSet)
        new._d = dict(self._d)
        return new

    __copy__ = copy

    def intersection(self, other) -> 'OrderedSet[T]':
        """Keeps *self*'s order, which is program order for liveness maps."""
        new = OrderedSet.__new__(OrderedSet)
        new._d = {k: None for k in self._d if k in other}
        return new

    def union(self, other) -> 'OrderedSet[T]':
        new = self.copy()
        new.update(other)
        return new

    def difference(self, other) -> 'OrderedSet[T]':
        new = OrderedSet.__new__(OrderedSet)
        new._d = {k: None for k in self._d if k not in other}
        return new

    __and__ = intersection
    __or__ = union
    __sub__ = difference

    # -- misc -------------------------------------------------------------- #

    def __eq__(self, other) -> bool:
        if isinstance(other, OrderedSet):
            return self._d.keys() == other._d.keys()
        if isinstance(other, (set, frozenset)):
            return set(self._d.keys()) == other
        return NotImplemented

    def __repr__(self) -> str:
        inner = ', '.join(getattr(k, 'name', None) or repr(k) for k in self._d)
        return f'{{{inner}}}'
