# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
from math import ceil
from typing import Optional

from tensorforge.common.target import Target
from tensorforge.common.basic_types import Datatype
from tensorforge.common.options import Options, ResolvedOptions

__all__ = ['Context', 'Options', 'ResolvedOptions']


class Context:
  def __init__(self,
               arch: str,
               backend: str,
               fp_type: Datatype,
               options: Optional[Options] = None):
    #: The device and the lowering (`common.target`).
    self.target: Target = Target(arch, backend)
    allowed = ('float', 'double', '__float128')
    if Datatype.as_str(fp_type) not in allowed:
      raise RuntimeError(f'unknown fp_type. Allowed {", ".join(allowed)}, '
                         f'given {Datatype.as_str(fp_type)}')
    self.fp_type = fp_type
    #: What the caller asked for, kept apart from what it resolved to: the two
    #: are the same statement about different things, and a report wants to
    #: name the first while generation reads the second.
    self._asked_options: Options = Options() if options is None else options
    self._options: ResolvedOptions = self._asked_options.resolve(self.target)

  def set_fp_type(self, fp_type: Datatype):
    self.fp_type = fp_type

  def fp_as_str(self):
    return Datatype.as_str(self.fp_type)

  def get_user_options(self) -> ResolvedOptions:
    """The settled options: one value per declared name, for this hardware."""
    return self._options

  def get_asked_options(self) -> Options:
    """What the caller passed, unresolved."""
    return self._asked_options

  def align(self, num):
    fp_size = self.fp_type.size()
    hw_fp_word_size = self.target.hw.hw_fp_word_size
    vec_unit_length = self.target.hw.vec_unit_length

    align_length = (vec_unit_length * hw_fp_word_size) / fp_size
    return int(ceil(num / align_length) * align_length)
