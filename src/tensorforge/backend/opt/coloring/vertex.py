# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
from typing import Set, TypeVar, Generic
from tensorforge.common.ordered import OrderedSet
VertexType = TypeVar('VertexType')


class Vertex(Generic[VertexType]):
  def __init__(self, vid: int):
    self._id: int = vid
    # ordered: the coloring walks neighbors, so iteration order leaks
    # into the emitted shared-memory offsets
    self._neighbors: OrderedSet = OrderedSet()

  def add_neighbor(self, vertex: VertexType) -> None:
    if not (vertex == self):
      self._neighbors.add(vertex)

  def get_neighbors(self) -> OrderedSet:
    return self._neighbors

  def has_neighbors(self) -> bool:
    return bool(self._neighbors)

  def remove_neighbor(self, vertex: VertexType) -> None:
    self._neighbors.remove(vertex)

  def get_id(self) -> int:
    return self._id

  def get_num_neighbors(self) -> int:
    return len(self._neighbors)

  def __eq__(self, other: VertexType) -> bool:
    return True if self._id == other.get_id() else False

  def __ne__(self, other: VertexType) -> bool:
    return not (self == other)

  def __str__(self) -> str:
    neighbors_str = [str(vertex.get_id()) for vertex in self._neighbors]
    neighbors_str = ', '.join(neighbors_str)
    return f'{self._id} -> {neighbors_str}'

  def __hash__(self) -> int:
    return hash(self._id)
