from abc import ABC, abstractmethod

from gemmforge.exceptions import GenerationError
from gemmforge.basic_types import DataFlowDirection
from typing import Union


class Matrix(ABC):
  ADDRESSIGN = ["none", "strided", "pointer_based"]
  PTR_TYPES = {"none": "*",
               "strided": "*",
               "pointer_based": "**"}

  def __init__(self, num_rows, num_cols, addressing):
    self.name = None
    self.num_rows = num_rows
    self.num_cols = num_cols
    self.direction: Union[DataFlowDirection, None] = None

    if addressing in Matrix.ADDRESSIGN:
      self.addressing = addressing
      self.ptr_type = Matrix.PTR_TYPES[self.addressing]
    else:
      raise ValueError('Invalid matrix addressing. '
                       'Valid types: {}'.format(", ".join(Matrix.ADDRESSIGN)))

  def set_data_flow_direction(self, direction: DataFlowDirection):
    self.direction = direction

  @abstractmethod
  def get_actual_num_rows(self):
    return self.num_rows

  @abstractmethod
  def get_actual_num_cols(self):
    return self.num_cols

  @abstractmethod
  def get_actual_volume(self):
    return self.num_rows * self.num_cols

  @abstractmethod
  def get_real_volume(self):
    return self.num_rows * self.num_cols

  @abstractmethod
  def get_offset_to_first_element(self):
    return 0

  def set_name(self, name):
    self.name = name

  @abstractmethod
  def __str__(self):
    string = "num. rows = {}\n".format(self.num_rows)
    string += "num. columns = {}\n".format(self.num_cols)
    string += "addressing = {}\n".format(self.addressing)
    return string
