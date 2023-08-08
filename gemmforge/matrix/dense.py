from gemmforge.exceptions import GenerationError
from gemmforge.basic_types import DataFlowDirection
from typing import Union


class DenseMatrix:
  ADDRESSIGN = ["none", "strided", "pointer_based"]
  PTR_TYPES = {"none": "*",
               "strided": "*",
               "pointer_based": "**"}

  def __init__(self, num_rows, num_cols, addressing, bbox=None):
    self.name = None
    self.num_rows = num_rows
    self.num_cols = num_cols
    self.direction: Union[DataFlowDirection, None] = None

    if bbox is not None:
      self.bbox = bbox

      # check whether bbox were given correctly
      coords = [coord for coord in self.bbox]
      if (self.num_rows < self.get_actual_num_rows()) or (
          self.num_cols < self.get_actual_num_cols()):
        raise GenerationError('Matrix size {}x{} is '
                              'smaller than bbox {}'.format(self.num_rows,
                                                            self.num_cols,
                                                            coords))
      if (self.num_rows < self.bbox[2]) or (self.num_cols < self.bbox[3]):
        raise GenerationError('Bbox {} is '
                              'outside of Matrix {}x{}'.format(coords,
                                                               self.num_rows,
                                                               self.num_cols))
    else:
      self.bbox = (0, 0, num_rows, num_cols)

    if addressing in DenseMatrix.ADDRESSIGN:
      self.addressing = addressing
      self.ptr_type = DenseMatrix.PTR_TYPES[self.addressing]
    else:
      raise ValueError('Invalid matrix addressing. '
                       'Valid types: {}'.format(", ".join(DenseMatrix.ADDRESSIGN)))

  def set_data_flow_direction(self, direction: DataFlowDirection):
    self.direction = direction

  def get_actual_num_rows(self):
    return self.bbox[2] - self.bbox[0]

  def get_actual_num_cols(self):
    return self.bbox[3] - self.bbox[1]

  def get_actual_volume(self):
    return self.get_actual_num_rows() * self.get_actual_num_cols()

  def get_real_volume(self):
    return self.num_rows * self.num_cols

  def get_offset_to_first_element(self):
    return self.num_rows * self.bbox[1] + self.bbox[0]

  def set_name(self, name):
    self.name = name

  def __str__(self):
    string = "num. rows = {}\n".format(self.num_rows)
    string += "num. columns = {}\n".format(self.num_cols)
    string += "bounding box = {}\n".format(self.bbox)
    string += "addressing = {}\n".format(self.addressing)
    string += "num. actual rows = {}\n".format(self.get_actual_num_rows())
    string += "num. actual cols = {}\n".format(self.get_actual_num_cols())
    return string
