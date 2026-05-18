from .lexic import Lexic, lexic_factory
from .hw_descr import HwDecription, hw_descr_factory
from math import ceil
from typing import Type


class VM:
  def __init__(self,
               hw_descr: HwDecription,
               lexic: Type[Lexic],
               fp_type: str):
    self._hw_descr = hw_descr
    self._lexic = lexic
    self._fp_type = None
    self.set_fp_type(fp_type)

  def get_hw_descr(self):
    return self._hw_descr

  def get_lexic(self):
    return self._lexic

  def set_fp_type(self, fp_type: str):
    if VM._is_valid_type(fp_type):
      self._fp_type = fp_type

  def fp_as_str(self):
    return self._fp_type

  def align(self, num):
    return ceil(num / self._hw_descr.vec_unit_length) * self._hw_descr.vec_unit_length

  def get_headers(self):
    return ['tensorforge_aux.h'] + self._lexic.get_headers()

  @classmethod
  def _is_valid_type(self, fp_type: str):
    allowed = ['__float128', 'double', 'float']
    if fp_type not in allowed:
      raise RuntimeError(f'unknown fp_type. Allowed {", ".join(allowed)}, given {fp_type}')
    return True


def vm_factory(arch: str,
               backend: str,
               fp_type: str):
  descr = hw_descr_factory(arch, backend)
  lexic = lexic_factory(backend=backend, underlying_hardware=descr.vendor)
  return VM(hw_descr=descr,
            lexic=lexic,
            fp_type=fp_type)
