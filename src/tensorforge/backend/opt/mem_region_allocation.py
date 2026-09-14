# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
from bisect import bisect_left
from collections import OrderedDict
from copy import copy
from tensorforge.common.ordered import OrderedSet
from typing import Dict, Optional, Set, Union, List, Tuple
from tensorforge.backend.symbol import Symbol
from .abstract import AbstractOptStage, Context
from .coloring import Vertex
from .coloring import GraphColoring


class Region:
  """Buffers that may share memory.

  Either a color -- every buffer in it starts at one offset, which `ShrMemOpt`
  sizes by the largest of them -- or, where the allocation placed each buffer
  itself (`MemoryRegionAllocation._pack`), one stretch of the arena, `size`
  elements from `offset`, holding every buffer that covers it.  A buffer is
  then in every region it spans.  Either way two buffers share a region
  exactly when they may share memory, which is what the barrier pass asks.
  """

  def __init__(self, offset: Optional[int] = None, size: Optional[int] = None):
    self._items: List[Symbol] = []
    self._counter: int = 0
    self.offset = offset
    self.size = size

  def add_item(self, item: Symbol) -> None:
    self._items.append(item)

  def __iter__(self):
    return self._items.__iter__()

  def __contains__(self, item):
    return item in self._items

  def print(self) -> None:
    for item in self._items:
      print(item.name)


class MemoryRegionAllocation(AbstractOptStage):
  def __init__(self, context: Context, live_map):
    super(MemoryRegionAllocation, self).__init__(context)

    self._live_map: Dict[int, OrderedSet] = live_map
    self._vertex_counter: int = 0
    self._adj_list: List[Vertex] = []
    self._objects2vertices_map: Union[Dict[Symbol, Vertex], None] = None
    self._regions: Union[List[Region], None] = None

  def apply(self) -> None:
    num_regions = MemoryRegionAllocation.compute_num_regions(self._live_map)
    variable_set = self._get_variable_set()
    self._adj_list, self._objects2vertices_map = self._generate_vertices(variable_set)
    self._assign_neighbors()

    self._regions: List[Region] = [Region() for i in range(num_regions)]
    gc = GraphColoring(graph=copy(self._adj_list), user_objects=self._regions)
    coloring_map: Dict[Vertex, object] = gc.apply()

    vertices2objects = {vertex: name for name, vertex in self._objects2vertices_map.items()}
    for vertex in self._adj_list:
      mem_region = coloring_map[vertex]
      mem_region.add_item(vertices2objects[vertex])

    if getattr(self._context.get_user_options(), 'shared_packing', True):
      packed = self._pack()
      if packed is not None:
        self._regions = packed

  def get_regions(self) -> List[Region]:
    return self._regions

  def _pack(self) -> Optional[List[Region]]:
    """Each buffer placed on its own, or None where that is no smaller.

    The coloring counts colors, not bytes: it needs as many as there are
    buffers live at once, and each color is as large as the largest buffer it
    was given.  SeisSol's damage step (order 6, double) has 908 buffers, never
    more than 22 of them live and never more than 65 KB; colored, the arena
    was 145 KB per multiplication and no block held one.  Placing the largest
    first, each at the lowest offset clear of every buffer it is live with,
    gives the 65 KB.

    Against the same interference the coloring uses, so what may share memory
    has not changed -- only where it lands.  Where the coloring is as small,
    it stays, and so does every kernel it already laid out well.
    """
    from .shr_mem_analyzer import SHR_ALIGN_BYTES
    symbols = list(self._objects2vertices_map)
    if not symbols:
      return None
    sizes: Dict[int, int] = {}
    for sym in symbols:
      size = _size(sym)
      if size is None:
        return None
      sizes[id(sym)] = size
    fp = self._context.fp_type.size()
    align = max(1, SHR_ALIGN_BYTES // fp) if fp else 1

    def aligned(count: int) -> int:
      return -(-count // align) * align

    # the coloring's arena as `ShrMemOpt` lays it out: each color from the
    # next aligned offset, as large as its largest buffer
    colored = 0
    for region in self._regions:
      colored = aligned(colored) + max((sizes[id(s)] for s in region), default=0)

    neighbors: Dict[int, Set[int]] = {id(s): set() for s in symbols}
    for live_vars in self._live_map.values():
      ids = [id(s) for s in live_vars]
      for i in ids:
        neighbors[i].update(ids)

    placed: Dict[int, int] = {}
    # largest first; `sorted` is stable, so equal sizes keep program order
    for sym in sorted(symbols, key=lambda s: -sizes[id(s)]):
      size = sizes[id(sym)]
      taken = sorted((placed[n], placed[n] + sizes[n])
                     for n in neighbors[id(sym)] if n in placed and n != id(sym))
      at = 0
      for lo, hi in taken:
        if at + size <= lo:
          break
        at = max(at, aligned(hi))
      placed[id(sym)] = at
    if max(placed[id(s)] + sizes[id(s)] for s in symbols) >= colored:
      return None

    edges = sorted({p for s in symbols
                    for p in (placed[id(s)], placed[id(s)] + sizes[id(s)])})
    stretches = [Region(offset=lo, size=hi - lo) for lo, hi in zip(edges, edges[1:])]
    empty: List[Region] = []
    for sym in symbols:
      start, end = placed[id(sym)], placed[id(sym)] + sizes[id(sym)]
      first, last = bisect_left(edges, start), bisect_left(edges, end)
      if first == last:
        # nothing to share, but `ShrMemOpt` still has to find its offset
        empty.append(Region(offset=start, size=0))
        empty[-1].add_item(sym)
      for index in range(first, last):
        stretches[index].add_item(sym)
    return [r for r in stretches if list(r)] + empty

  def _get_variable_set(self) -> Dict[Symbol, None]:
    ordered_variable_set = OrderedDict()
    for live_vars in self._live_map.values():
      for var in live_vars:
        ordered_variable_set[var] = None
    return ordered_variable_set

  def _generate_vertices(self, variable_set: Dict[Symbol, None]) -> Tuple[List[Vertex],
                                                                          Dict[Symbol, Vertex]]:
    objects2vertices_map: Dict[Symbol, Vertex] = {}
    vertices: List[Vertex] = []
    for variable in variable_set.keys():
      vertex = Vertex(self._gen_new_vertex_id())
      vertices.append(vertex)
      objects2vertices_map[variable] = vertex
    return vertices, objects2vertices_map

  def _assign_neighbors(self) -> None:
    for live_vars in self._live_map.values():
      if live_vars:
        # the loop over both variables will lead to a symmetric graph
        for var1 in live_vars:
          for var2 in live_vars:
            vertex1 = self._objects2vertices_map[var1]
            vertex2 = self._objects2vertices_map[var2]
            vertex1.add_neighbor(vertex2)

  def _gen_new_vertex_id(self) -> int:
    id = self._vertex_counter
    self._vertex_counter += 1
    return id

  @classmethod
  def compute_num_regions(self, live_map: Dict[int, Set[Symbol]]) -> int:
    num_regions = 0
    for prog_point in live_map.values():
      num_regions = max(num_regions, len(prog_point))
    return num_regions


def _size(symbol: Symbol) -> Optional[int]:
  """Elements, as `ShrMemOpt` sizes a buffer: by its first user."""
  user = symbol.get_first_user()
  size_fn = getattr(user, 'compute_shared_mem_size', None)
  if not callable(size_fn):
    return None
  return size_fn()
