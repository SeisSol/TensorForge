# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
from tensorforge.common.context import Context
from tensorforge.backend.scopes import Scopes, Symbol
from tensorforge.backend.symbol import SymbolType, DataView
from tensorforge.backend.instructions.ptr_manip import GetElementPtr
from tensorforge.common.basic_types import GeneralLexicon
from .abstract_builder import AbstractBuilder


class GetElementPtrBuilder(AbstractBuilder):
  def __init__(self, context: Context, scopes: Scopes):
    super(GetElementPtrBuilder, self).__init__(context, scopes)

  def build(self, src: Symbol, include_extra_offset: bool = True,
            table=None, variant=None, name=None, obj=None):
    """Name a pointer for `src`, and register it under what it stands for.

    `obj` is that thing where it is not `src`'s own: a merged run's rotated
    stand-in reads a *slot*, and the slot is a buffer several members share
    (`Generator._rotate_members`), so `src.obj` is whichever member happened
    to open it.  Registered under that member, the binding would do two
    wrong things at once -- shadow the member, so a second table over the
    same family selects this pass's pointer in its place, and leave the
    stand-in resolving to the bare slot, so the body reads one buffer every
    iteration and never the table.  Both are one mistake: the binding
    belongs to the hole, not to the buffer behind it.
    """
    self._reset()

    dstype = src.stype

    # A binding is global because what it names is: `&m2[batchId][offset]`.
    # A rotated stand-in's table holds *shared* buffers -- the slots the
    # family turns through -- and a pointer into shared memory is not a
    # global one however it was obtained.  Typed global it would be read in
    # the wrong address space, which `copy_async` reports as a base that is
    # shared under an access claiming global.
    if dstype not in (SymbolType.Scalar, SymbolType.Data) and not (
            dstype is SymbolType.SharedMem and obj is not None):
      dstype = SymbolType.Global

    dest = Symbol(name=f'{GeneralLexicon.GLOBAL_MEM_PREFIX}{name or src.name}',
                    stype=dstype,
                    obj=src.obj if obj is None else obj)

    if dstype == SymbolType.SharedMem:
      # The window is the slot's, and every slot of a family has the same
      # one (`_rotate_members` refuses a family that disagrees on a shape),
      # so the pointer the table hands over addresses what the slot did.
      dest.data_view = src.data_view
      # And none of its own: this names storage the slots hold, so the pass
      # that gives every shared symbol a window (`_declare_buffers`) would
      # ask the first thing that reads it to declare one -- a multiplication,
      # which declares nothing -- and the allocator would hold a second
      # buffer for a pointer.
      dest.borrowed = True
    elif src.stype not in (SymbolType.Scalar, SymbolType.Data):
      # TODO: remove this code path
      dest.data_view = DataView(shape=src.obj.get_actual_shape(), permute=None,
                                bbox=src.obj.get_bbox(),
                                elem_parts=src.obj.storage_parts,
                                owner=src.obj)
    else:
      dest.data_view = DataView(shape=src.obj.shape, permute=None)
    self._scopes.add_symbol(dest)

    if src.stype != SymbolType.Data:
      self._instructions.append(GetElementPtr(self._context, src, dest,
                                             include_extra_offset,
                                             table=table, variant=variant))

    src.add_user(self)
