from kernelforge.common.exceptions import GenerationError
from kernelforge.common.basic_types import Addressing, DataFlowDirection
from typing import Union, List
from .matrix import Matrix
import json


# Cordinate object form needs be a dictionary of the following entries:
# rows - number of rows (int) e.g. "rows" : 2
# cols - number of columns (int) e.g. "cols": 2
# entries - and array of coordinate arrays of row,col, 0-indexed e.g. "entries" : [[0,0],[1,1]] (given as the storage order)
# Optionally values = [0.2f, 9.4f]... (it is assumed that non of the values here are 0)
class SparseMatrix(Matrix):
  def __init__(self, num_rows: int, num_cols: int, addressing: Addressing, coordinates,
               values: Union[List[float], None]=None,
               bbox: Union[List[int], None]=None,
               alias: Union[str, None]=None,
               is_tmp: bool = False):
    Matrix.__init__(self, num_rows, num_cols, addressing, bbox, alias, is_tmp)
    self.elcount = 0

    self.values = values

    self.dense_representation = [[0] * num_cols for _ in range(num_rows)]
    i = 0
    for coordinate in coordinates:
      val = "X"
      if values != None:
        assert i < len(values), f"i < len(values) : {i} < {len}"
        val = values[i]
      self.dense_representation[int(coordinate[0])][int(coordinate[1])] = val
      i += 1
    self.coo = coordinates

    self.coo_per_row = [[] for _ in range(num_rows)]
    self.coo_per_col = [[] for _ in range(num_cols)]

    i = 0
    for coordinate in coordinates:
      val = "X"
      if values != None:
        val = values[i]
      self.coo_per_row[int(coordinate[0])].append(int(coordinate[1]))
      self.coo_per_col[int(coordinate[1])].append(int(coordinate[0]))
      self.elcount += 1
      i += 1

    non_zero_cols = 0
    non_zero_rows = 1
    for i in self.coo_per_col:
      if len(i) > 0:
        non_zero_cols += 1
    for i in self.coo_per_row:
      if len(i) > 0:
        non_zero_rows += 1
    self.num_max_non_zero_cols = non_zero_cols
    self.num_max_non_zero_rows = non_zero_rows

  def get_actual_num_rows(self):
    return self.num_rows

  def get_actual_num_cols(self):
    return self.num_cols

  def get_actual_num_max_nonzero_cols(self):
    return self.num_max_non_zero_cols

  def get_actual_num_max_nonzero_rows(self):
    return self.num_max_non_zero_cols

  def get_actual_volume(self):
    return self.num_rows * self.num_cols

  def get_real_volume(self):
    return self.get_el_count()

  def get_offset_to_first_element(self):
    return 0

  def __str__(self):
    string = super().__str__()
    return f'[sparse] {string}'

  def get_coo_per_row(self):
    return self.coo_per_row

  def get_coo_per_col(self):
    return self.coo_per_col

  def get_coordinates(self):
    return self.coo

  def get_values(self):
    return self.values

  def get_el_count(self):
    return self.elcount

  def find_1d_offset(self, row, col):
    assert (row < self.get_actual_num_rows())
    assert (col < self.get_actual_num_cols())
    coordinates = self.get_coordinates()
    i = 0
    for (_row, _col) in coordinates:
      if row == _row and col == _col:
        break
      i += 1
    assert (i < len(coordinates))
    return i

  def sparsity(self):
    size = self.get_actual_num_cols() * self.get_actual_num_rows()
    el_count = self.get_el_count()
    return 1.0 - float(el_count / size)
