from gemmforge.exceptions import GenerationError
from gemmforge.matrix.matrix import Matrix
import json


# Cordinate object form needs be a dictionary of the following entries:
# rows - number of rows (int) e.g. "rows" : 2
# cols - number of columns (int) e.g. "cols": 2
# entries - and array of coordinate arrayys of row,col,value, 0-indexed e.g. "entries" : [[0,0,"1.0"],[1,1,"1.0"]]
# Optionally a name can be provided e.g. "name": "simple_identity"
# Final example : {"name": "simple_identity", "rows": 2, "cols": 2, "coordinates": [[0,0],[1,1], ...]

# 1-input 1 coordinate list (sbp)
# 2-input 2 optional list of values (valeus)
class SparseMatrix(Matrix):
  def __init__(self, num_rows, num_cols, addressing, coordinates, values=None):
    Matrix.__init__(self, num_rows, num_cols, addressing)
    self.elcount = 0

    self.values = values

    self.dense_representation = [[0] * num_cols for _ in range(num_rows)]
    iter = 0
    for coordinate in coordinates:
      val = "X"
      if values != None:
        assert iter < len(values), f"iter < len(values) : {iter} < {len}"
        val = values[iter]
      self.dense_representation[int(coordinate[0])][int(coordinate[1])] = val
      iter += 1
    self.coo = coordinates

    self.coo_row_major = [[] for _ in range(num_rows)]
    self.coo_col_major = [[] for _ in range(num_cols)]

    iter = 0
    for coordinate in coordinates:
      val = "X"
      if values != None:
        val = values[iter]
      self.coo_row_major[int(coordinate[0])].append(int(coordinate[1]))
      self.coo_col_major[int(coordinate[1])].append(int(coordinate[0]))
      self.elcount += 1
      iter += 1

    non_zero_cols = []
    for i in self.coo_row_major:
      non_zero_cols.append(len(i))
    self.num_max_non_zero_cols = max(non_zero_cols)

    # If the coordinates are not sorted, we need to generated iteration orders, during the generation we need
    # to find the offsets of the elements
    for row in self.coo_row_major:
      row.sort()
    for col in self.coo_col_major:
      col.sort()

  def get_actual_num_rows(self):
    return self.num_rows

  def get_actual_num_cols(self):
    return self.num_cols

  def get_actual_num_max_nonzero_cols(self):
    return self.num_max_non_zero_cols

  def get_actual_volume(self):
    return self.num_rows * self.num_cols

  def get_real_volume(self):
    return self.get_el_count()

  def get_offset_to_first_element(self):
    return 0

  def __str__(self):
    string = super().__str__()
    string += str(self.dense_representation)
    return string

  def get_coo_row_major(self):
    return self.coo_row_major

  def get_coo_col_major(self):
    return self.coo_col_major

  def get_coordinates(self):
    return self.coo

  def get_values(self):
    return self.values

  def get_el_count(self):
    return self.elcount

  def find_1d_offset(self, row, col, transpose=False):
    assert (row < self.get_actual_num_rows())
    assert (col < self.get_actual_num_cols())
    coordinates = self.get_coordinates()
    iter = 0
    for (_row, _col) in coordinates:
      if row == _row and col == _col:
        break
      iter += 1
    assert (iter < len(coordinates))
    return iter
