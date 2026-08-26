# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
class ShrMemObject:
  def __init__(self, name, size=None, mults_per_block=None):
    self.name = name
    self._size_per_mult = size
    self._mults_per_block = mults_per_block
    self._global_size = 0
    self._temp_offset = 0

  def alloc_global(self, size):
    startpoint = self._global_size
    self._global_size += size
    return startpoint

  def set_size_per_mult(self, size):
    self._size_per_mult = size

  def set_mults_per_block(self, num_mults):
    self._mults_per_block = num_mults

  def get_size_per_mult(self):
    return self._size_per_mult

  def get_mults_per_block(self):
    return self._mults_per_block

  def get_global_size(self):
    return self._global_size

  def get_total_size(self):
    # ShrMemAlloc.is_ready() calls this and tests the result against None, so
    # it must not raise before the thread-block policy has set the factors.
    # get_total_size_as_str() already had this guard; this one did not, and
    # any caller reaching it early died with a TypeError instead.
    if self._size_per_mult is None or self._mults_per_block is None:
      return None
    return self._size_per_mult * self._mults_per_block + self._global_size

  def get_total_size_as_str(self):
    if self._size_per_mult and self._mults_per_block:
      return self.get_total_size()
    else:
      return 'not yet defined'

  def set_temp_offset(self, offset):
    self._temp_offset = offset

  def get_temp_offset(self):
    return self._temp_offset

  def is_dense(self):
    return True

  def __str__(self) -> str:
    total_size = self.get_total_size_as_str()
    return f'name {self.name}: total size = {total_size}'

class RegMemObject:
  def __init__(self, name, size=None, datatype=None, spp=None):
    self.name = name
    self.size = size
    self.datatype = datatype
    self.spp = spp

  def __str__(self):
    return f'name: {self.name}; size = {self.size}'

  def is_dense(self):
    return self.spp is None

  def linear_index(self, index):
    # TODO: remove?
    realindex = tuple(index)
    return self.spp.linear_index(realindex)
