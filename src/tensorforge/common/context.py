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
               pipeline_depth=2,
               enable_wrap_loads=False,
               wrap_distance=1,
               wide_bodies=True):
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
    # Slot-granular prefetch: move a register transfer `wrap_distance` compute
    # slots ahead of its consumer, wrapping to the previous iteration when that
    # runs off the front of the body.  One buffer copy for any distance up to
    # n - 1; see backend/opt/wrap.py.
    self.enable_wrap_loads = enable_wrap_loads
    self.wrap_distance = wrap_distance
    # One PIR body per loop body, rather than one per macro instruction.
    #
    # A pass sees a body.  Per macro instruction that means `RegisterAlloc`,
    # the loader that fills the buffer and the multilinear that reads it are
    # three separate bodies, and the only thing connecting them is the C++
    # name -- 60.7% of buffers in the corpus are named for that reason alone,
    # against 10.3% per loop body (tools/buffer_spans.py).  Everything still
    # needing a name here outlives one loop body: the shared arena, its
    # scratch tail, and the tiles of the two cases that have two batch loops.
    #
    # So this is not primarily a code-quality switch -- the cross-instruction
    # CSE win is 0.2% -- it is what makes the naming go away, and with it the
    # reason `symbol.py` builds addresses as text.
    #
    # `TF_IR_WIDE=0` forces it off without touching a call site, because a
    # change that moves 71 of 108 generated outputs has to stay bisectable.
    self.wide_bodies = wide_bodies


class Context:
  #: Whether every emitted body should report its peak register footprint.
  #:
  #: Off by default: the figure costs a liveness walk per body, and only a
  #: caller searching over configurations reads it.  `lanes.search` turns it
  #: on for the builds it does and puts it back afterwards.
  measure_pressure: bool = False

  #: The largest figure reported so far, in bytes per lane, or None.
  #:
  #: Here rather than on the instruction that measured it, because the body
  #: that matters has no instruction: with wide bodies a whole section is one
  #: body, emitted by `AbstractInstruction.shared_body`, which is a
  #: classmethod and has only the context to hand.  A maximum, since a
  #: register budget is per kernel and the widest body is what has to fit.
  peak_pressure = None

  def record_pressure(self, value: int) -> None:
    if self.peak_pressure is None or value > self.peak_pressure:
      self.peak_pressure = value

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
