from kernelforge.common.exceptions import GenerationError
from kernelforge.common.basic_types import Addressing, DataFlowDirection
from typing import Union, List
from .matrix import Matrix

class DenseMatrix(Matrix):
  def __init__(self,
               num_rows,
               num_cols,
               addressing,
               bbox=None,
               alias=None,
               is_tmp=False):
    super(DenseMatrix, self).__init__(num_rows,
                                      num_cols,
                                      addressing,
                                      bbox,
                                      alias,
                                      is_tmp)
  
  def __str__(self):
    string = super().__str__()
    return f'[dense] {string}'
