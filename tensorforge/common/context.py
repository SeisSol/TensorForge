# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
from math import ceil
from tensorforge.common.vm.vm import VM, vm_factory
from tensorforge.common.basic_types import Datatype


class Options:
  def __init__(self,
               exact_contraction_length=False,
               align_shr_mem=True,
               enable_sync_block_opt=True,
               enable_pipeline=False,
               enable_multibuffer=False,
               pipeline_depth=2):
    self.exact_contraction_length: bool = exact_contraction_length
    self.align_shr_mem: bool = align_shr_mem
    self.enable_sync_block_opt = enable_sync_block_opt
    # Software pipelining. `enable_pipeline` advances the address computation;
    # `enable_multibuffer` additionally rotates the shared-memory buffers, which
    # needs `enable_pipeline` (rotation reads the advanced pointer) and is
    # implemented for `pipeline_depth == 2` only -- see backend/opt/pipeline.py.
    #
    # Both stay off by default pending hardware numbers; correctness no longer
    # blocks them.
    self.enable_pipeline = enable_pipeline
    self.enable_multibuffer = enable_multibuffer
    self.pipeline_depth = pipeline_depth


class Context:
  def __init__(self,
               arch: str,
               backend: str,
               fp_type: Datatype,
               options: Options = Options()):
    self._vm: VM = vm_factory(arch, backend, Datatype.as_str(fp_type))
    self.fp_type = fp_type
    self._options = options

  def set_fp_type(self, fp_type: Datatype):
    self.fp_type = fp_type

  def fp_as_str(self):
    return Datatype.as_str(self.fp_type)

  def get_vm(self):
    return self._vm

  def get_user_options(self):
    return self._options

  def align(self, num):
    fp_size = self.fp_type.size()
    hw_fp_word_size = self._vm.get_hw_descr().hw_fp_word_size
    vec_unit_length = self._vm.get_hw_descr().vec_unit_length

    align_length = (vec_unit_length * hw_fp_word_size) / fp_size
    return int(ceil(num / align_length) * align_length)

  def align_range(self, begin, end):
    assert end > begin
    fp_size = self.fp_type.size()
    mem_access_align_size = self._vm.get_hw_descr().mem_access_align_size
    align_factor =  mem_access_align_size / fp_size

    aligned_begin = begin - begin % align_factor
    aligned_end = end + (align_factor - end % align_factor) % align_factor
    return int(aligned_begin), int(aligned_end)
