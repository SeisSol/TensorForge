from kernelforge.common import Addressing
from kernelforge.common.matrix.tensor import Tensor


class VarFactory:
  @classmethod
  def produce_matrix(cls, text_descr):
    addrs = Addressing.str2addr(text_descr['addressing'])

    descr = Tensor(shape=[text_descr['num_rows'], text_descr['num_cols']],
                        addressing=addrs,
                        bbox=text_descr['bbox'],
                        alias=text_descr['name'],
                        is_tmp=False)
    return text_descr['name'], descr

  @classmethod
  def produce_scalar(cls, text_descr):
    return text_descr['name'], text_descr['value']
