# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
from typing import Set, TypeVar, Generic
from tensorforge.common.ordered import OrderedSet
VertexType = TypeVar('VertexType')


class Vertex(Generic[VertexType]):
  def __init__(self, vid: int):
    self._id: int = vid
    # ordered: the colouring walks neighbours, so iteration order leaks
    # into the emitted shared-memory offsets
    self._neighbours: OrderedSet = OrderedSet()

  def add_neighbor(self, vertex: VertexType) -> None:
    if not (vertex == self):
      self._neighbours.add(vertex)

  def get_neighbors(self) -> OrderedSet:
    return self._neighbours

  def has_neighbours(self) -> bool:
    return bool(self._neighbours)

  def remove_neighbour(self, vertex: VertexType) -> None:
    self._neighbours.remove(vertex)

  def get_id(self) -> int:
    return self._id

  def get_num_neighbours(self) -> int:
    return len(self._neighbours)

  def __eq__(self, other: VertexType) -> bool:
    return True if self._id == other.get_id() else False

  def __ne__(self, other: VertexType) -> bool:
    return not (self == other)

  def __str__(self) -> str:
    neighbours_str = [str(vertex.get_id()) for vertex in self._neighbours]
    neighbours_str = ', '.join(neighbours_str)
    return f'{self._id} -> {neighbours_str}'

  def __hash__(self) -> int:
    return hash(self._id)
