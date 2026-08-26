# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
from copy import copy
from typing import List, Dict, Set, Union
from .vertex import Vertex


class VertexStack:
  def __init__(self):
    self._vertices: List[Vertex] = []

  def add_edges(self, vertex: Vertex):
    self._vertices.append(copy(vertex))

  def pop_edges(self) -> Vertex:
    return self._vertices.pop()

  def print_stack(self) -> None:
    for edge in self._vertices:
      print(edge)

  def empty(self) -> bool:
    return len(self._vertices) == 0


class GraphColoring:
  def __init__(self, graph: List[Vertex], user_objects: List[object]):
    self._graph: List[Vertex] = graph
    self._max_num_colors: int = len(user_objects)
    self._colors: List[int] = [i for i in range(self._max_num_colors)]
    self._allowed_color_set: Set[int] = set(self._colors)
    self._color2object_map: Dict[int, object] = {color: user_object
                                                 for color, user_object
                                                 in zip(self._colors, user_objects)}
    self._stack: VertexStack = VertexStack()
    self._vertex2color_map: Dict[Vertex, Union[int, None]] = {v: None for v in self._graph}

  def apply(self) -> Dict[Vertex, object]:
    self._graph = sorted(self._graph, key=lambda x: x.get_num_neighbours(), reverse=True)

    # run the first part of the algorithm
    while self._coarse_graph():
      pass

    # Chaitin simplify is done.  It stops either because the residual graph
    # has no edges (the intended bottom case) *or* because no vertex has
    # degree < k -- the spill case.  Colouring the latter with a single
    # colour silently aliases two simultaneously-live buffers, so assert
    # instead.  For a straight-line instruction list the interference graph
    # is an interval graph, hence chordal, hence k = max|liveset| always
    # suffices; this fires only if that premise breaks (e.g. once loops
    # carry live ranges).
    residual = [v for v in self._graph if v.get_neighbors()]
    if residual:
      raise RuntimeError(
          f'graph colouring failed to simplify: {len(residual)} vertices of '
          f'degree >= {self._max_num_colors} remain '
          f'({", ".join(str(v.get_id()) for v in residual)}). Colouring them '
          f'alike would alias live shared-memory buffers.')

    # it is the bottom case i.e., a graph consists of only nodes without edges
    for vertex in self._graph:
      free_color = self._colors[0]
      self._vertex2color_map[vertex] = free_color

    # run the second part of the algorithm
    while not self._stack.empty():
      self._restore_graph_and_color()

    # map Dict[vertex, color] to Dict[vertex, object] as it was required by the user
    vertex2object = {}
    for vertex, color in self._vertex2color_map.items():
      vertex2object[vertex] = self._color2object_map[color]
    return vertex2object

  def print_graph(self) -> None:
    print('~' * 80)
    for vertex in self._graph:
      print(vertex)

  def _coarse_graph(self) -> bool:
    for index, vertex in enumerate(self._graph):
      if vertex.get_neighbors():
        if self._max_num_colors > vertex.get_num_neighbours():
          candidate = self._graph.pop(index)
          self._stack.add_edges(candidate)
          self._remove_edges(candidate)
          return True
    return False

  def _remove_edges(self, vertex) -> None:
    for neighbour in vertex.get_neighbors():
      for item in self._graph:
        if neighbour == item:
          item.remove_neighbour(vertex)

  def _restore_graph_and_color(self) -> None:
    vertex = self._stack.pop_edges()
    self._assign_color(vertex)
    self._add_edges_to_graph(vertex)

  def _assign_color(self, vertex) -> None:
    occupied_colors = set()
    for neighbour in vertex.get_neighbors():
      assigned_color = self._vertex2color_map[neighbour]
      occupied_colors.add(assigned_color)
    free_colors = self._allowed_color_set - occupied_colors
    if not free_colors:
      raise RuntimeError(f'no free colour for vertex {vertex.get_id()}')
    # min(), not set.pop(): make the choice explicit rather than relying on
    # CPython's small-int set ordering
    self._vertex2color_map[vertex] = min(free_colors)

  def _add_edges_to_graph(self, vertex) -> None:
    for neighbour in vertex.get_neighbors():
      for item in self._graph:
        if item == neighbour:
          item.add_neighbor(vertex)
