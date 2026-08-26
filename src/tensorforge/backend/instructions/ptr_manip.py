# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
from .abstract_instruction import AbstractInstruction
from tensorforge.common.context import Context
from tensorforge.common.helper import get_extra_offset_name, Addressing
from tensorforge.common.basic_types import GeneralLexicon, DataFlowDirection, StridedAddressing
from tensorforge.common.exceptions import GenerationError
from tensorforge.backend.pir.core import Effect

class GetElementPtr(AbstractInstruction):
  def __init__(self,
               context: Context,
               src,
               dest,
               include_extra_offset=True,
               batch_offset=0,
               update_dest=None,
               pipeline = False):
    super(GetElementPtr, self).__init__(context)
    self._src = src
    self._dest = dest
    self._include_extra_offset = include_extra_offset
    self._is_ready = True
    # int -> `batchId{n}`, the n-th lookahead index bound by the loop.
    # str -> used verbatim, which is how a peeled iteration names an index that
    # exists *outside* the loop: `batchId0` is the loop variable and does not
    # exist in the prologue, and the pre-loop bindings of batchId1/batchId2 mean
    # something different from the in-loop ones (clamped from batchId_start
    # rather than from batchId0).
    self._batch_offset = batch_offset
    self._update_dest = update_dest
    self._pipeline = pipeline

  def batch_index(self) -> str:
    if isinstance(self._batch_offset, str):
      return self._batch_offset
    return f'{GeneralLexicon.BATCH_ID_NAME}{self._batch_offset}'

  def gen_ir(self, writer):

    batch_obj = self._src.obj
    batch_addressing = batch_obj.addressing

    if self._include_extra_offset:
      extra_offset = f' + {get_extra_offset_name(self._src)}'
    else:
      extra_offset = ''

    datatype = self._vm._fp_type if self._src.obj.datatype is None else self._src.obj.datatype

    const_mod = '' if self._pipeline else 'const'

    address = ''
    if isinstance(batch_addressing, StridedAddressing):
      main_offset = f'{self.batch_index()} * {batch_addressing.stride}'
      sub_offset = f'{batch_obj.get_offset_to_first_element()}'
      address = f'{main_offset} + {batch_addressing.offset} + {sub_offset}{extra_offset}'
      rhs = f'&{self._src.name}[{address}]'
      lhs = 'const ' if self._src.obj.direction == DataFlowDirection.SOURCE else ''
      lhs += f'{datatype} *{const_mod} {self._vm.get_lexic().restrict_kw} {self._dest.name}'
    if batch_addressing == Addressing.STRIDED:
      # distance between batch elements is the *stored* volume, i.e.
      # prod(upper - lower), not prod(shape)
      main_offset = f'{self.batch_index()} * {batch_obj.get_actual_volume()}'
      sub_offset = f'{batch_obj.get_offset_to_first_element()}'
      address = f'{main_offset} + {sub_offset}{extra_offset}'
      rhs = f'&{self._src.name}[{address}]'
      lhs = 'const ' if self._src.obj.direction == DataFlowDirection.SOURCE else ''
      lhs += f'{datatype} *{const_mod} {self._vm.get_lexic().restrict_kw} {self._dest.name}'
    elif batch_addressing == Addressing.PTR_BASED:
      main_offset = f'{self.batch_index()}'
      sub_offset = f'{batch_obj.get_offset_to_first_element()}'
      address = f'{main_offset}][{sub_offset}{extra_offset}'
      src_suffix = '_ptr' if self._vm.get_lexic()._backend == 'targetdart' else ''
      rhs = f'&{self._src.name}{src_suffix}[{address}]'
      lhs = 'const ' if self._src.obj.direction == DataFlowDirection.SOURCE else ''
      if self._context.get_vm().get_hw_descr().vendor == 'amd':
        lhs += f'{datatype}'
        rhs = f'(tensorforge::SpacePtrRestrict<{lhs}, tensorforge::GlobalMemspace>){rhs}'
        lhs = f'auto {self._dest.name}'
      else:
        lhs += f'{datatype} *{const_mod} {self._vm.get_lexic().restrict_kw} {self._dest.name}'
    elif batch_addressing == Addressing.NONE:
      address = f'{batch_obj.get_offset_to_first_element()}'
      rhs = f'&{self._src.name}[{address}]'
      lhs = 'const ' if self._src.obj.direction == DataFlowDirection.SOURCE else ''
      lhs += f'{datatype} *{const_mod} {self._vm.get_lexic().restrict_kw} {self._dest.name}'
    elif batch_addressing == Addressing.SCALAR:
      rhs = f'{self._src.name}'
      lhs = f'{datatype} {self._dest.name}'
    else:
      GenerationError(f'unknown addressing of {self._src.name}, given {batch_addressing}')

    if self._update_dest:
      writer(f'const auto {self._update_dest.name} = {self._dest.name};')
      writer(f'{self._dest.name} = {rhs};')
    else:
      self._emit_binding(writer, f'{lhs} = {rhs};')

  def _emit_binding(self, writer, text: str) -> None:
    """The binding, saying what it touches.

    It was a bare statement, so `Effect.UNKNOWN`, so it conflicted with every
    access in the body and pinned everything on both sides of it -- 362 nodes
    corpus-wide, the largest blocking site after `allocate.py`.  That matters
    here rather than in the abstract: `WrapLoads` moves a transfer past the
    instructions between it and its consumer, and a binding that conflicts
    with everything is a wall in the middle of exactly that stretch.

    What it actually touches is the batch handle it reads.  Strided
    addressing does not even do that -- the right-hand side is address
    arithmetic and touches no memory -- but `Addressing.PTR_BASED` reads
    `m1[batchId]` out of the pointer array, so declaring the read covers both
    and is conservative for the one that does less.

    `movable=False`, and not out of caution: the pointer it defines is still
    a C++ name, so a consumer reading `glb_m1` does so through text the IR
    cannot see.  Letting the binding sink below such a consumer would compile
    to a use before its definition.  Movability comes back when the
    definition becomes a value -- the same trade as `extern` on the register
    tiles, and it ends the same way.
    """
    if hasattr(writer, 'access_stmt'):
      writer.access_stmt(text, self._src, Effect.READ, movable=False)
    else:
      writer(text)

  def defs(self):
    return (self._dest,) if self._update_dest is None else (self._dest, self._update_dest)

  def uses(self):
    return (self._src,)

  def __str__(self) -> str:
    return (f'{self._dest.name} = getelementptr_b2g {self._src.name} '
            f'[{self.batch_index()}];')
