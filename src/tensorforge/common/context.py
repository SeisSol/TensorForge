# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
from math import ceil
from typing import Optional

from tensorforge.common.vm.vm import VM, vm_factory
from tensorforge.common.basic_types import Datatype
from tensorforge.common.options import Options, ResolvedOptions

__all__ = ['Context', 'Options', 'ResolvedOptions']


class Context:
  def __init__(self,
               arch: str,
               backend: str,
               fp_type: Datatype,
               options: Optional[Options] = None):
    self._vm: VM = vm_factory(arch, backend, Datatype.as_str(fp_type))
    self.fp_type = fp_type
    #: What the caller asked for, kept apart from what it resolved to: the two
    #: are the same statement about different things, and a report wants to
    #: name the first while generation reads the second.
    self._asked_options: Options = Options() if options is None else options
    self._options: ResolvedOptions = self._asked_options.resolve(
        self._vm.get_hw_descr())

    #: Whether every emitted body should report its peak register footprint.
    #:
    #: Off by default: the figure costs a liveness walk per body, and only a
    #: caller searching over configurations reads it.  `lanes.search` turns it
    #: on for the builds it does and puts it back afterwards.
    #:
    #: Per context and not per class, because two contexts in one process are
    #: two searches and neither is entitled to the other's answer.
    self.measure_pressure: bool = False

    #: The largest figure reported so far, in bytes per lane, or None.
    #:
    #: Here rather than on the instruction that measured it, because the body
    #: that matters has no instruction: with wide bodies a whole section is one
    #: body, emitted by `AbstractInstruction.shared_body`, which is a
    #: classmethod and has only the context to hand.  A maximum, since a
    #: register budget is per kernel and the widest body is what has to fit.
    self.peak_pressure: Optional[int] = None

    #: Arithmetic operations written out so far, or None (`record_work`).
    self.emitted_work: Optional[int] = None

  def record_pressure(self, value: int) -> None:
    if self.peak_pressure is None or value > self.peak_pressure:
      self.peak_pressure = value

  def record_work(self, value: int = 1) -> None:
    """Count arithmetic the emitter wrote out.

    What a lane geometry changes and neither the register model nor blocks per
    SM can see: a packed FMA does two elements in one operation and a matrix
    instruction does a whole tile, so a geometry that reaches either issues
    fewer of these.  Counted per kernel, so a *rolled* reduction
    (`Options.k_roll`) counts its body once however many times it runs --
    the same caveat the line estimate carries.
    """
    self.emitted_work = (self.emitted_work or 0) + value

  def set_fp_type(self, fp_type: Datatype):
    self.fp_type = fp_type

  def fp_as_str(self):
    return Datatype.as_str(self.fp_type)

  def get_vm(self):
    return self._vm

  def get_user_options(self) -> ResolvedOptions:
    """The settled options: one value per declared name, for this hardware."""
    return self._options

  def get_asked_options(self) -> Options:
    """What the caller passed, unresolved."""
    return self._asked_options

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
