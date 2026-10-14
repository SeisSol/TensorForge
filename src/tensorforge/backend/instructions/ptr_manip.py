# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
import enum
from typing import List, Tuple
from contextlib import contextmanager

from .abstract_instruction import AbstractInstruction
from tensorforge.common.context import Context
from tensorforge.common.helper import get_extra_offset_name, Addressing
from tensorforge.common.basic_types import GeneralLexicon, DataFlowDirection, StridedAddressing
from tensorforge.common.exceptions import (GenerationError,
                                           InternalError)
from tensorforge.backend.pir.core import Access, Effect, Qual
from tensorforge.backend.pir.core import MemSpace
from tensorforge.backend.symbol import SymbolType, passed_by_value
from tensorforge.generators import stacking

class GetElementPtr(AbstractInstruction):
  def __init__(self,
               context: Context,
               src,
               dest,
               include_extra_offset=True,
               table=None,
               variant=None):
    super(GetElementPtr, self).__init__(context)
    self._src = src
    # Where the base pointer is read from, when it is not the argument itself.
    # A run whose operand changes between iterations reaches it through a table
    # the kernel builds from arguments it already has, so only the *name* on
    # the right-hand side changes; the offsetting below is the same arithmetic
    # either way, and keeping it one expression is what makes a table-fed
    # operand indistinguishable downstream from an argument-fed one.
    self._table = table
    self._variant = variant
    self._dest = dest
    self._include_extra_offset = include_extra_offset
    self._is_ready = True

  #: Stands in for the element index while an address is being assembled.
  #:
  #: The index has to end up as an operand and the address around it is text,
  #: so the two are spliced somewhere.  Splicing on the *name* -- searching the
  #: finished address for `batchId0` -- would not do: a search cannot tell the
  #: index this binding was given from a longer name that happens to start
  #: alike, and it finds nothing at all for the offsets whose name the
  #: enclosing body does not bind.  A token no address can otherwise contain is
  #: the same splice without the search.
  _INDEX_HOLE = '\x00batchIndex\x00'

  #: Stands in for a merged run's counter while a table access is assembled.
  #:
  #: Same reason as the index's hole, and a sharper one: with the counter only
  #: in the text, a table access has no operand that varies with the loop, so
  #: `licm` would hoist it out of the loop that binds the counter -- to a place
  #: where the name is not in scope at all.
  _VARIANT_HOLE = '\x00variantIndex\x00'

  #: Stands in for a select chain the body made a value (`DeclareOperandTable`
  #: `.value`): the binding reads it as an operand, so it is ordered after the
  #: chain by the def-use edge, and a copy of the two for another iteration
  #: reads the copy of the chain.
  _TABLE_HOLE = '\x00operandTable\x00'

  def batch_index(self) -> str:
    """The name this binding's element index goes by: the loop's own.

    A key rather than a spelling: the loop's bindings carry no `extern`, so
    what the emitted code calls the index is the IR's business, and the name
    is what a pass and a loop agree to call the same element.  A binding for
    another element is a pass's copy of this one (`pir.ahead.Clone`), given
    that element's index as an operand.
    """
    return f'{GeneralLexicon.BATCH_ID_NAME}0'

  def batch_value(self, writer):
    """The element index, as something this body can name as an operand.

    Two answers, and which one applies is decided by whether the loop is in
    this body -- not by which element is wanted.

    Where it is, the loop's own binding: the value carries a def-use edge back
    to the statement computing the index, which is what stops `licm` hoisting
    an address out of the loop that defines what it reads, and what gives
    `substitute` something to rewrite when a pass moves this binding to
    another element.  An index the loop does *not* bind is an error here
    rather than a name resolved against whatever is in scope -- `batchId1`
    outside a loop that binds no lookahead is the prologue's binding, a
    different element, and reading it would address the wrong one and
    compile.

    Where the loop is elsewhere, the seam: a value standing for a name bound
    outside this body.  No edge -- there is nothing here to have an edge to --
    but the index is an operand rather than text, so the address stops having
    an element spelled into it.  This is what every binding gets when a body
    is one macro instruction wide.

    `None` where the index is not named `batchId<something>`, which nothing
    produces today; the address then keeps the name as text rather than the
    emitter being asked to bind one it knows nothing about.
    """
    from .batch_loop import BatchLoop
    if not hasattr(writer, 'batch_id'):
      return None
    name = self.batch_index()
    bound = BatchLoop.indices_in(writer)
    if bound is not None:
      value = bound.get(name)
      if value is None:
        raise InternalError(
            f'{self._dest.name} addresses element `{name}`, which the loop in '
            f'this body does not bind (it binds '
            f'{", ".join(sorted(bound)) or "nothing"})')
      return value
    prefix = GeneralLexicon.BATCH_ID_NAME
    if not name.startswith(prefix):
      return None
    return writer.batch_id(name[len(prefix):])

  def _splice_index(self, writer, text: str):
    """`(text, args)`, with the hole turned into an operand where there is one.

    The escaping and the placeholder happen in one pass because they cannot be
    ordered: escaping afterwards turns the `{0}` this writes into `{{0}}`, and
    escaping before means the hole has to survive it, which it only does by
    containing no braces -- true of this hole and not something a caller should
    have to know.
    """
    escaped = text.replace('{', '{{').replace('}', '}}')
    args = []
    for hole, value, name in ((self._INDEX_HOLE, self.batch_value(writer)
                               if self._INDEX_HOLE in escaped else None,
                               self.batch_index),
                              (self._VARIANT_HOLE, self.variant_value(writer)
                               if self._VARIANT_HOLE in escaped else None,
                               lambda: str(self._variant)),
                              (self._TABLE_HOLE, self._table.value(writer)
                               if self._TABLE_HOLE in escaped else None,
                               lambda: self._table.name)):
      if hole not in escaped:
        continue
      if value is None:
        escaped = escaped.replace(hole, name())
      else:
        escaped = escaped.replace(hole, f'{{{len(args)}}}')
        args.append(value)
    return escaped, tuple(args)

  def _as_text(self, rhs: str) -> str:
    """The right-hand side with its holes spelled out, for an emitter that
    takes no operands."""
    out = (rhs.replace(self._INDEX_HOLE, self.batch_index())
           .replace(self._VARIANT_HOLE, str(self._variant)))
    if self._table is not None:
      out = out.replace(self._TABLE_HOLE, self._table.name)
    return out

  def variant_value(self, writer):
    """The counter of the merged run this binding belongs to, as an operand.

    `None` where the run's loop is not the one open here, or where it does not
    publish a value -- an unrolled emission, say. The access then keeps the
    name as text.
    """
    if self._variant is None:
      return None
    return VariantLoop.counter_value(writer, self._variant)

  def dereferences_the_batch(self) -> bool:
    """Does computing this address read memory indexed by the element?

    `Addressing.PTR_BASED` does: the source is an array of pointers and the
    address is `&m[batchId][off]`, which loads `m[batchId]` before offsetting
    it.  Strided addressing and the batch-invariant modes do not -- the
    right-hand side is arithmetic on a base the caller passed in.

    The distinction decides whether this binding may be emitted outside the
    per-element flag guard.  A masked element is one the caller told us not to
    process, and nothing in the interface promises its pointer is valid to
    dereference; an element whose *index* is merely multiplied by a stride
    carries no such promise to break.
    """
    return getattr(self._src.obj, 'addressing', None) == Addressing.PTR_BASED

  def source_name(self) -> str:
    """What the right-hand side reads the base pointer out of."""
    if self._table is None:
      return self._src.name
    if self._table.form is TableForm.SELECT:
      return self._TABLE_HOLE
    return self._table.access(self._VARIANT_HOLE)

  def _pointer_type(self, datatype, const_mod: str,
                    restrict: bool = True) -> str:
    """The type of this binding, as the backend spells a pointer into global.

    One place for all four addressing modes, and the reason is not tidiness:
    on AMD the address space sits in the pointer's type, so four spellings
    could declare a generic pointer in three modes where the fourth casts to a
    space-qualified one -- the same binding, two types, decided by nothing but
    which mode the operand happens to use.  Asking the lexic makes them agree,
    and makes the space something a pass can reproduce when it declares a copy
    of the value instead of losing it to `auto`.

    A batch-constant operand the kernel only reads is spelled in the constant
    space (`KernelParam.of_symbol` says why).  Only the spelling: where a
    backend names no spaces it is the same text, and the IR keeps calling it
    global, which is what the loads and their ordering care about.

    One passed by value is spelled in no space at all.  Its elements are
    where the argument is, and that is not always the argument segment: a
    read at an index the compiler cannot resolve takes the struct's address,
    and clang then copies it into private memory first.  Cast into the
    constant space, that private address faults on gfx1150; generic, the
    compiler infers whichever space it is.
    """
    readonly = self._src.obj.direction == DataFlowDirection.SOURCE
    if self._table is not None:
      # Whatever the stand-in's direction says, the table is what it is read
      # out of: a writable binding over a `const` table does not compile.
      readonly = not self._table.writable
    if passed_by_value(self._src):
      space = None
    elif readonly and self._src.obj.addressing == Addressing.NONE:
      space = MemSpace.CONSTANT
    else:
      space = MemSpace.GLOBAL
    return self._context.target.lexic.pointer_type(
        f'{datatype}', space, readonly=readonly,
        restrict=restrict, const=bool(const_mod))

  def _declarator(self, datatype, const_mod: str) -> str:
    return f'{self._pointer_type(datatype, const_mod)} {self._dest.name}'

  def _coerce(self, datatype, const_mod: str, rhs: str) -> str:
    """Bring an address into the space the declaration claims for it.

    The right-hand side is arithmetic on a kernel argument, which is generic;
    a space-qualified pointer is reached from one only through a cast, and the
    conversion the other way is the implicit one.  So the cast follows the
    declared type rather than a branch on the vendor: where the backend spells
    no space this is the identity.

    Note for AMD that this applies to every addressing mode, not to
    `PTR_BASED` alone.  That is the consistent answer -- all of these point
    into global memory -- but claiming the space tells the compiler something,
    beyond how the pointer is spelled, and it wants a measurement before it is
    taken as settled.
    """
    cast = self._pointer_type(datatype, '')
    if '<' not in cast:
      return rhs
    return f'({cast}){rhs}'

  def gen_ir(self, writer):

    if self._table is not None and stacking.factor(self._src.obj) > 1:
      self._gen_stacked_table(writer)
      return
    if self._table is not None:
      # The table's members are the bindings the prologue already made, so
      # they carry the element offset.  Offsetting again would apply it twice;
      # what varies between iterations is which of them to take, and that is
      # the whole of it.
      datatype = self._fp_as_str if self._src.obj.datatype is None else self._src.obj.datatype
      if self._src.obj.addressing == Addressing.SCALAR:
        # A table over scalars selects a value, and the binding is that value.
        self._emit_binding(writer, f'{datatype} {self._dest.name}',
                           self.source_name(), scalar=True)
        return
      lhs = self._declarator(datatype, 'const')
      # The table is declared generic -- its members may come from either
      # space and a select over them decays -- so the binding casts back into
      # the space it claims, as every other addressing mode does (`_coerce`).
      self._emit_binding(writer, lhs,
                         self._coerce(datatype, 'const', self.source_name()))
      return

    batch_obj = self._src.obj
    batch_addressing = batch_obj.addressing

    if stacking.factor(batch_obj) > 1:
      self._gen_stacked(writer)
      return

    if self._include_extra_offset:
      extra_offset = f' + {get_extra_offset_name(self._src)}'
    else:
      extra_offset = ''

    datatype = self._fp_as_str if self._src.obj.datatype is None else self._src.obj.datatype

    const_mod = 'const'

    address = ''
    if isinstance(batch_addressing, StridedAddressing):
      main_offset = f'{self._INDEX_HOLE} * {batch_addressing.stride}'
      sub_offset = f'{batch_obj.get_offset_to_first_element()}'
      address = f'{main_offset} + {batch_addressing.offset} + {sub_offset}{extra_offset}'
      rhs = self._coerce(datatype, const_mod, f'&{self.source_name()}[{address}]')
      lhs = self._declarator(datatype, const_mod)
    if batch_addressing == Addressing.STRIDED:
      # distance between batch elements is the *stored* volume, i.e.
      # prod(upper - lower), not prod(shape)
      main_offset = f'{self._INDEX_HOLE} * {batch_obj.storage_volume()}'
      sub_offset = f'{batch_obj.get_offset_to_first_element()}'
      address = f'{main_offset} + {sub_offset}{extra_offset}'
      rhs = self._coerce(datatype, const_mod, f'&{self.source_name()}[{address}]')
      lhs = self._declarator(datatype, const_mod)
    elif batch_addressing == Addressing.PTR_BASED:
      main_offset = f'{self._INDEX_HOLE}'
      sub_offset = f'{batch_obj.get_offset_to_first_element()}'
      address = f'{main_offset}][{sub_offset}{extra_offset}'
      source = self._context.target.lexic.batch_source(self.source_name())
      rhs = self._coerce(datatype, const_mod, f'&{source}[{address}]')
      lhs = self._declarator(datatype, const_mod)
    elif batch_addressing == Addressing.NONE:
      address = f'{batch_obj.get_offset_to_first_element()}'
      rhs = self._coerce(datatype, const_mod, f'&{self.source_name()}[{address}]')
      lhs = self._declarator(datatype, const_mod)
    elif batch_addressing == Addressing.SCALAR:
      rhs = f'{self.source_name()}'
      lhs = f'{datatype} {self._dest.name}'
    else:
      GenerationError(f'unknown addressing of {self._src.name}, given {batch_addressing}')

    self._emit_binding(writer, lhs, rhs,
                       scalar=batch_addressing == Addressing.SCALAR)

  def _gen_stacked_table(self, writer) -> None:
    """The stand-in of a merged run over stacked members: a pointer per
    element of the stack, each the table's chain for that element
    (`DeclareOperandTable._declare_chains`)."""
    chains = self._table.value(writer)
    if not isinstance(chains, tuple):
      raise InternalError(f'{self._dest.name}: a stacked stand-in over a '
                          f'table that holds no chain per element')
    from tensorforge.backend.pir.core import BufferType
    obj = self._src.obj
    datatype = self._fp_as_str if obj.datatype is None else obj.datatype
    readonly = not self._table.writable
    values = []
    for e, chain in enumerate(chains):
      name = f'{self._dest.name}_s{e}'
      values.append(writer.decl_expr(
          f'{self._pointer_type(datatype, "const", readonly)} {name}',
          self._coerce(datatype, 'const', '{0}'),
          BufferType(self._dest.get_fptype(), chain.type.shape, MemSpace.GLOBAL,
                     readonly=readonly),
          self._src, args=(chain,), kind=Effect.READ, hint=name, extern=name,
          alias_root=self._src))
    self._dest.set_pir_stack(writer, values,
                             self._pointer_type(datatype, 'const', readonly),
                             [f'{self._dest.name}_s{e}'
                              for e in range(len(values))])

  def _gen_stacked(self, writer) -> None:
    """A pointer per element of the stack (`Options.stack_elements`).

    An element owns its memory, and under `Addressing.PTR_BASED` two
    elements' memory is two allocations: there is no stride from one to the
    other, so the stack's last axis picks a pointer rather than adding to an
    address (`Symbol.stacked_binding`).  Each pointer is the element's own
    where the element is one to write (`BatchLoop._stack_actives`), and
    otherwise the first such element's of the stack -- an element past the
    end of the batch has no pointer to read, and one its flag masks a
    pointer nothing promises is valid -- or the stack's first element's where
    there is none, which is in the batch.  Strided addressing reads the same
    elements, for uniformity: the stride would reach an element past the end.

    The stack's index and the masks are operands, so a pass that moves a
    transfer to another stack (`pir.ahead.Clone`) computes these again for
    it as it does a single element's pointer.
    """
    from .batch_loop import BatchLoop
    obj = self._src.obj
    k = stacking.factor(obj)
    if not hasattr(writer, 'decl_expr'):
      raise InternalError(f'{self._dest.name}: a stack of elements is bound in '
                          f'a body built as IR')
    stack = self.batch_value(writer)
    bound = BatchLoop.indices_in(writer) or {}
    actives = [bound.get(stacking.mask_key(e)) for e in range(k)]
    if stack is None or any(a is None for a in actives):
      raise InternalError(f'{self._dest.name}: a stack of elements is bound '
                          f'where its loop binds no masks for it')
    datatype = self._fp_as_str if obj.datatype is None else obj.datatype
    extra = (f' + {get_extra_offset_name(self._src)}'
             if self._include_extra_offset else '')
    # `{0}` the stack's index, `{1 + e}` element `e`'s mask.
    first = ' : '.join([f'{{{1 + e}}} ? {k} * {{0}} + {e}' for e in range(k)]
                       + [f'{k} * {{0}}'])
    volume = obj.storage_volume() // k
    from tensorforge.backend.pir.core import BufferType
    values = []
    for e in range(k):
      element = f'({{{1 + e}}} ? {k} * {{0}} + {e} : ({first}))'
      if obj.addressing == Addressing.PTR_BASED:
        source = self._context.target.lexic.batch_source(self._src.name)
        rhs = f'&{source}[{element}][{obj.get_offset_to_first_element()}{extra}]'
      elif obj.addressing == Addressing.STRIDED:
        rhs = (f'&{self._src.name}[{element} * {volume} + '
               f'{obj.get_offset_to_first_element()}{extra}]')
      else:
        raise InternalError(f'{self._src.name}: stacked, and the batch shares '
                            f'it ({obj.addressing})')
      name = f'{self._dest.name}_s{e}'
      readonly = obj.direction == DataFlowDirection.SOURCE
      lhs = f'{self._pointer_type(datatype, "const", readonly)} {name}'
      rhs = self._coerce(datatype, 'const', rhs)
      values.append(writer.decl_expr(
          lhs, rhs, BufferType(self._dest.get_fptype(),
                               (self._extent() // k,), MemSpace.GLOBAL,
                               readonly=readonly),
          self._src, args=(stack, *actives), kind=Effect.READ, hint=name,
          extern=name, alias_root=self._src,
          attrs=(('element_pointer', True),)
          if self.dereferences_the_batch() else (),
          quals=(Qual.RESTRICT,) if readonly else ()))
    # Not `restrict` where the operand is written: the pointers of an element
    # the stack lacks are its first element's, so two of them reach the same
    # memory, and one writes through it (as `_emit_binding` says of a clone).
    self._dest.set_pir_stack(
        writer, values,
        self._pointer_type(datatype, 'const',
                           obj.direction == DataFlowDirection.SOURCE),
        [f'{self._dest.name}_s{e}' for e in range(k)])

  def _emit_binding(self, writer, lhs: str, rhs: str,
                    scalar: bool = False) -> None:
    """The binding, as a definition rather than a statement.

    A bare statement is `Effect.UNKNOWN`, so it conflicts with every access
    in the body and pins everything on both sides of it.  That matters here
    rather than in the abstract: a pass moving a transfer moves it past the
    statements between it and its consumer, and a binding that conflicts
    with everything is a wall in the middle of exactly that stretch.

    Declaring only its accesses would make it reorderable but leave it
    nameless, so it would have to stay pinned anyway: a consumer reading
    `glb_m1` would do so through text the IR cannot see, and letting the
    binding sink below one would compile to a use before its definition.
    Producing a *value* is what removes that reason -- the def-use edge
    exists, so the scheduler knows the distance it may not close.

    The declarator stays text.  `const float *const __restrict__ p` and the
    AMD `auto p` with its type inside a cast are not renderable from a type,
    and teaching the emitter a declarator grammar for one caller would buy
    nothing the value does not already buy.

    What it touches is the batch handle it reads.  Strided addressing does not
    even do that -- the right-hand side is address arithmetic and reaches no
    memory -- but `Addressing.PTR_BASED` reads `m1[batchId]` out of the
    pointer array, so declaring the read covers both.  The accesses are
    recorded against `self._src`, not against the value: `may_alias` treats
    distinct bases as never aliasing, and a window that claimed its own
    identity would let a write through the underlying buffer reorder past a
    read through the window.

    `scalar` says the binding *is* the number rather than addressing one --
    `Addressing.SCALAR`, whose pointer type is the empty string.  Then the
    value is typed as a scalar and published on the symbol's scalar field, so
    a reader takes a number where a number is what the text holds.  Only that
    addressing: a scalar that varies per element is a buffer of one number,
    read through an address like any other operand, and typing it here as a
    value would say it is the same for every element.
    """
    if hasattr(writer, 'decl_expr'):
      from tensorforge.backend.pir.core import BufferType, ScalarType
      # Name the element index as an operand.  The address is
      # `&m2[batchId0 * 324 + ...]`, and with the element only in the text a
      # pass that moves this binding to another one would have nothing to
      # substitute -- `wrap_loads` could not advance a transfer that reads
      # through it.
      text, args = self._splice_index(writer, rhs)
      type_ = (ScalarType(self._dest.get_fptype()) if scalar else
               BufferType(self._dest.get_fptype(), (self._extent(),),
                          MemSpace.GLOBAL,
                          readonly=self._src.obj.direction
                          == DataFlowDirection.SOURCE))
      value = writer.decl_expr(
          lhs, text, type_,
          self._src, args=args, kind=Effect.READ, hint=self._dest.name,
          extern=self._dest.name, alias_root=self._src,
          # What the declared read cannot say: strided addressing reads
          # nothing and declares the read all the same, so only this tells a
          # pass that the address is a pointer of the element's own, valid to
          # follow for an element the caller did not mask.
          attrs=(('element_pointer', True),)
          if self.dereferences_the_batch() else (),
          # Only where nothing is written through it.  `restrict` promises
          # that what this pointer reaches is reached through no other in
          # scope, and a pass may clone this computation for the next element
          # while the original is still live -- two pointers into one buffer,
          # which is a promise kept as long as both only read and broken the
          # moment one writes.
          quals=(() if scalar else
                 (Qual.RESTRICT,)
                 if self._src.obj.direction == DataFlowDirection.SOURCE
                 else ()))
      if scalar:
        self._dest.set_pir_scalar(writer, value)
      else:
        self._dest.set_pir_buffer(writer, value)
    else:
      writer(f'{lhs} = {self._as_text(rhs)};')

  def _extent(self) -> int:
    """How many elements the reads through this binding reach: one batch
    element of the operand, as it is stored.

    The reads name the binding as their buffer (`IRBuilder.load`), so its
    type is what a lowering asks how far a run of them may go -- the explicit
    vector merges consecutive reads into one message only inside the buffer
    -- and how far a hint for the element reaches (`pir.prefetch`).  The
    extent is the one the operand would have answered: its storage where it
    is stored in an order of its own, compressed where it is sparse, its view
    otherwise.
    """
    obj = self._dest.obj
    if (getattr(obj, 'storage_order', None) is not None
        or not getattr(obj, 'is_dense', lambda: True)()):
      return int(obj.storage_volume())
    view = self._dest.data_view
    shape = getattr(view, 'shape', None) if view is not None else None
    extent = 1
    for dim in shape or ():
      extent *= int(dim)
    return extent

  def defs(self):
    return (self._dest,)

  def uses(self):
    return (self._src,)

  def __str__(self) -> str:
    return (f'{self._dest.name} = getelementptr_b2g {self._src.name} '
            f'[{self.batch_index()}];')


class TableForm(enum.Enum):
  """How a run reaches the member its counter names.

  `SELECT` is a chain of conditionals collapsing to one pointer.  The counter
  is uniform, so each step is a scalar select on both vendors and nothing is
  stored anywhere; the chain is `n - 1` selects long, which is why it is the
  choice for a handful of members and not for many.

  `ARRAY` is an initialized array indexed by the counter.  Constant in the
  length of the run, and it pays for that by being *memory*: an array with a
  dynamic index cannot be promoted out of its allocation, so it lands in the
  per-thread space -- `.local` on NVIDIA, scratch on AMD -- where every thread
  in the block builds and holds its own copy of one set of pointers, and where
  on AMD the allocation alone can cost occupancy.

  `PARAM` is neither: the table is a kernel parameter the caller fills, so the
  members never appear as separate arguments and the read comes from the space
  kernel arguments already live in -- broadcast, uniform and cached.  The best
  of the three wherever it applies, and the one that costs an interface, since
  the launcher has to assemble the array.  It is also the only one whose cost
  does not grow with the length of the run, so it is what a run of sixteen
  wants where a run of four is better off with a chain.

  So `SELECT` is the default and `ARRAY` earns its place only once the chain
  is longer than the memory is worth.  A third form is better than both where
  it applies and is not available here: members that are one contiguous blob
  need neither, since the address is the base plus the counter times a stride.
  """

  SELECT = 'select'
  ARRAY = 'array'
  PARAM = 'param'


class DeclareOperandTable(AbstractInstruction):
  """How the kernel's own arguments are reached by a counter.

  A run whose operand changes between iterations needs that operand reachable
  by index.  Where its members are already parameters -- which is the case a
  frontend that wrote the repetition out always produces, since it named every
  one of them -- nothing has to reach the interface; it is assembled inside the
  kernel from what is already there.

  Assembled in one of two shapes, and `TableForm` says why the default is the
  one that touches no memory.

  The element type follows the members' addressing rather than being chosen
  here.  A pointer-based argument is already an array per element, so a table
  over such members is one indirection deeper than a table over batch-invariant
  ones, and getting that wrong is a type error at compile time rather than a
  wrong address at run time.
  """

  #: Beyond this many members the chain is longer than the memory is worth.
  #: A guide for whoever constructs one; nothing here enforces it.
  SELECT_LIMIT = 8

  def __init__(self, context: Context, name: str, members, addressing,
               datatype=None, form: 'TableForm' = None, variant: str = None,
               writable: bool = False, first: int = 0):
    super(DeclareOperandTable, self).__init__(context)
    #: The first member the loop selects.  Those before it are the
    #: iterations that ran before the loop (`VariantLoop.start`): a select
    #: chain leaves them out, and nothing reaches them through the table.
    self._first = first
    #: Whether the loop writes through the members: the table then holds
    #: pointers to mutable data.  Not `const` regardless: a written stand-in's
    #: binding (`float *const glb_v0`) cannot be initialized from a
    #: `const float *` -- SeisSol's merged `gpu_derivative` has one.
    self._writable = writable
    if not members:
      raise GenerationError('an operand table has at least one member')
    passed = [m.name for m in members if passed_by_value(m)]
    if passed:
      # A table holds addresses, and these are values: the struct a member is
      # passed as is not a pointer to select or to store.
      raise GenerationError(
          f'operand table {name}: {", ".join(passed)} are passed by value, '
          f'and a table reaches its members through their addresses')
    # Numbers the kernel carries in its code: a scalar is a literal, which a
    # table can hold (`_member`); an array has no address to select.
    inlined = [m.name for m in members if m.stype == SymbolType.Data
               and m.obj.addressing != Addressing.SCALAR]
    if inlined:
      raise GenerationError(
          f'operand table {name}: {", ".join(inlined)} are in the code as '
          f'literals, and a table of addresses has nothing to point at')
    self._name = name
    self._members = list(members)
    self._addressing = addressing
    self._datatype = datatype
    self._form = form or TableForm.SELECT
    self._variant = variant
    if self._form is TableForm.SELECT and variant is None:
      raise GenerationError('a select chain has to name the counter it reads')
    #: `(builder uid, value)`: the chain, where the body being built made it
    #: a value (`gen_ir`).
    self._value = None
    self._is_ready = True

  @property
  def form(self) -> 'TableForm':
    return self._form

  def value(self, writer):
    """The pointer the chain selects, as a value of the body `writer` builds,
    or None where it is text -- an array, a parameter, or a chain built where
    the counter is no value."""
    if self._value is None:
      return None
    owner, value = self._value
    uid = getattr(writer, 'uid', None)
    return value if uid is not None and owner == uid else None

  def loop_invariant(self) -> bool:
    """Whether this may be emitted once, ahead of the loop.

    An array does not depend on the counter and is built before the header; a
    select chain is the counter's value and belongs at the top of the body.  A
    parameter is built by the caller and is emitted nowhere at all.
    """
    return self._form is not TableForm.SELECT

  def access(self, variant: str) -> str:
    """The expression that yields the member `variant` names."""
    if self._form is TableForm.PARAM:
      return f'{self._name}.p[{variant}]'
    if self._form is TableForm.ARRAY:
      return f'{self._name}[{variant}]'
    return self._name

  @property
  def name(self) -> str:
    return self._name

  def __len__(self) -> int:
    return len(self._members)

  def _require_param(self) -> None:
    if self._form is not TableForm.PARAM:
      raise GenerationError(f'{self._form.value} is not a parameter')

  def struct_name(self) -> str:
    return f'{self._name}_t'

  def struct_definition(self) -> str:
    """The by-value type the table is passed as.

    A struct and not an array, because an array parameter decays to a pointer:
    `const float* const t[4]` in a signature *is* `const float* const* t`, so
    the caller would have to put the four pointers somewhere the device can
    read them and pass the address of that.  Which is a global load through an
    extra indirection -- the opposite of the point, and the thing the
    annotation cannot fix, since there is no by-value parameter left to
    annotate.  Wrapping the array in a struct is what keeps it a value.
    """
    self._require_param()
    datatype = self._datatype or self._fp_as_str
    stars = Addressing.addr2ptr_type(self._addressing)
    return (f'struct {self.struct_name()} {{ {self._qual()}{datatype} '
            f'{self._inner(stars)}p[{len(self._members)}]; }};')

  @property
  def writable(self) -> bool:
    return self._writable

  def _qual(self) -> str:
    return '' if self._writable else 'const '

  def _inner(self, stars: str) -> str:
    """The pointer's own qualifier; a table of values has no pointer, and a
    second `const` on the value is a duplicate nvcc rejects."""
    return f'{stars}const ' if stars else ''

  def parameter(self) -> str:
    """How this table is declared in the kernel's signature, for `PARAM`.

    The annotation is the lexic's, not this site's: what keeps a by-value
    parameter out of per-thread memory is a property of the backend, and on
    most of them the answer is that nothing is needed.

    Asked as a question about *residency* rather than about a keyword.  The
    table lives in param space -- one copy for the whole grid, read-only,
    broadcast -- and `__grid_constant__` is CUDA's way of being told so, not a
    separate fact about the declaration.
    """
    self._require_param()
    annotation = self._context.target.lexic.storage_class(MemSpace.PARAM)
    annotation = f'{annotation} ' if annotation else ''
    return f'{annotation}const {self.struct_name()} {self._name}'

  def argument(self) -> str:
    """How the caller builds it, one launch before the kernel reads it."""
    self._require_param()
    entries = ', '.join(m.name for m in self._members)
    return f'const {self.struct_name()} {self._name} = {{{{{entries}}}}};'

  def _member(self, member, datatype, stars, name=None) -> str:
    """A member as the table holds it: generic.  Spelled `name` where the
    caller has a spelling of its own for it -- an operand's placeholder.

    Where the backend spells the address space in the pointer's type (AMD),
    a member is a space-qualified binding and the table is a plain pointer,
    and the conversion between the two is explicit there.  Elsewhere the
    cast would be the identity, so it is left out.

    A constant scalar has no name at all: the kernel reads it as a literal,
    and its binding is never declared.  So the table holds its number --
    SeisSol's damage step selects between such constants in its merged
    loops, and a table naming them would not compile.
    """
    if member.stype == SymbolType.Data:
      spelling = member.obj.datatype or datatype
      values = member.obj.get_values()
      # A scalar's numbers are a zero-dimensional array: the value itself.
      value = values.item() if getattr(values, 'ndim', 1) == 0 else values[0]
      return spelling.literal(value)
    name = member.name if name is None else name
    spaced = '<' in self._context.target.lexic.pointer_type(
        f'{datatype}', MemSpace.GLOBAL, readonly=True, restrict=True, const=True)
    return (f'({self._qual()}{datatype} {stars}){name}'
            if spaced and stars else name)

  def _accesses(self, writer):
    """What the loop touches through the table, said where it is built.

    Its members: every one is read through the binding the table feeds, and
    written where the loop writes them -- by accesses that name the binding,
    so an analysis keyed by buffer would not see the member in them.  Here it
    does, once an iteration, which is all a buffer occupied throughout the
    loop needs: the allocator keeps a member staged in shared memory clear of
    what the loop allocates, and the barrier placement orders it after what
    wrote it.  A member that is a number touches no memory.
    """
    from tensorforge.backend.pir.build import access_of
    from tensorforge.backend.pir.core import Access
    kind = Effect.READ | (Effect.WRITE if self._writable else Effect.NONE)
    out = []
    for member in self._members[self._first:]:
      if member.stype == SymbolType.Data:
        continue
      window = (member.pir_buffer(writer)
                if member.stype == SymbolType.SharedMem else None)
      out.append(Access(kind, MemSpace.SHARED, window) if window is not None
                 else access_of(member, kind))
    return tuple(out)

  def gen_ir(self, writer):
    if self._form is TableForm.PARAM:
      # Nothing to emit: the caller filled it and the signature names it.
      return
    datatype = self._datatype or self._fp_as_str
    stars = Addressing.addr2ptr_type(self._addressing)
    if self._form is not TableForm.SELECT and any(m.stack() > 1
                                                  for m in self._members):
      raise GenerationError(
          f'operand table {self._name}: a stack of elements is reached through '
          f'a select chain per element, not an array')
    if self._form is TableForm.ARRAY:
      entries = ', '.join(self._member(m, datatype, stars) for m in self._members)
      self._declare(writer, f'{self._qual()}{datatype} '
                            f'{self._inner(stars)}{self._name}'
                            f'[{len(self._members)}] = {{{entries}}};')
      return
    decl = f'{self._qual()}{datatype} {self._inner(stars)}{self._name}'
    counter = (VariantLoop.counter_value(writer, self._variant)
               if hasattr(writer, 'decl_expr') else None)
    if counter is None:
      self._declare(writer, f'{decl} = ' + self._chain(
          lambda m: self._member(m, datatype, stars), self._variant) + ';')
      return
    self._value = (writer.uid, self._declare_chain(writer, decl, counter,
                                                   datatype, stars))

  def _chain(self, spell, counter: str) -> str:
    """The conditionals that collapse to the member the counter names, with
    the counter spelled `counter` and each member as `spell` has it."""
    chain = spell(self._members[-1])
    for index in range(len(self._members) - 2, self._first - 1, -1):
      chain = f'({counter} == {index}) ? {spell(self._members[index])} : {chain}'
    return chain

  def _declare_chain(self, writer, decl: str, counter, datatype, stars):
    """The chain as a definition: a pointer -- a number, over scalars --
    computed from the counter, which the bindings reading it take as an
    operand; and, beside it, what the run touches through the members.

    Text alone would be a statement that defines a name nothing else in the
    IR can see, and so a wall (`Effect.UNKNOWN`): the bindings reading the
    name could not be ordered after it otherwise.  As a value the order is
    the def-use edge, and the chain is arithmetic: it selects an address
    and reads nothing, so a pass may compute it again -- for the next trip,
    with the counter it substitutes, which is why the counter is an operand
    and not text.  So are the members, where the body has them as values:
    computed again for the next element, the chain selects among that
    element's bindings, and a member read through text alone would be the
    one of the element it was built for.

    What it does not say is what the reads and writes through the pointer
    reach, since they name the stand-in: that is a statement of its own
    (`IRBuilder.touches`, `_accesses`), which stays in the loop.  On the
    chain, it would be lost to the analyses that ask -- they take a
    declaration of a pointer for the address it is (`allocate._binds`) --
    and it would make the chain something a pass cannot compute twice.
    """
    from tensorforge.backend.pir.core import BufferType, ScalarType
    touched = self._accesses(writer)
    if touched:
      writer.touches(touched)
    stack = max((m.stack() for m in self._members), default=1)
    if stack > 1:
      return self._declare_chains(writer, decl, counter, datatype, stars, stack)
    dtype = self._datatype or self._context.fp_type
    if self._addressing == Addressing.SCALAR:
      type_ = ScalarType(dtype)
    else:
      type_ = BufferType(dtype, (self._extent(),), MemSpace.GLOBAL,
                         readonly=not self._writable)
    args = [counter] if len(self._members) - self._first > 1 else []

    def spell(member):
      value = self._value_of(writer, member)
      if value is None:
        text = self._member(member, datatype, stars)
        return text.replace('{', '{{').replace('}', '}}')
      args.append(value)
      return self._member(member, datatype, stars, f'{{{len(args) - 1}}}')
    text = self._chain(spell, '{0}')
    return writer.decl_expr(decl, text, type_, None, args=tuple(args),
                            hint=self._name, extern=self._name)

  def _declare_chains(self, writer, decl: str, counter, datatype, stars,
                      stack: int):
    """A chain per element of a stack (`Options.stack_elements`): over the
    members' pointers for that element, `_declare_chain` each.  The tuple of
    them is the table's value, and the stand-in's binding takes one each
    (`GetElementPtr._gen_stacked_table`)."""
    from tensorforge.backend.pir.core import BufferType
    if any(m.stack() != stack for m in self._members):
      raise GenerationError(
          f'operand table {self._name}: members stacked differently, or not '
          f'at all, beside stacked ones')
    dtype = self._datatype or self._context.fp_type
    values = []
    for e in range(stack):
      args = [counter] if len(self._members) - self._first > 1 else []

      def spell(member, _e=e):
        pointers = member.pir_stack(writer)
        if pointers is None:
          raise InternalError(
              f'operand table {self._name}: {member.name} has no pointers for '
              f'its stack in this body')
        args.append(pointers[_e])
        return self._member(member, datatype, stars, f'{{{len(args) - 1}}}')
      text = self._chain(spell, '{0}')
      name = f'{self._name}_s{e}'
      type_ = BufferType(dtype, (self._extent() // stack,), MemSpace.GLOBAL,
                         readonly=not self._writable)
      values.append(writer.decl_expr(decl[:-len(self._name)] + name, text,
                                     type_, None, args=tuple(args), hint=name,
                                     extern=name))
    return tuple(values)

  @staticmethod
  def _value_of(writer, member):
    """`member` as a value of the body `writer` builds -- its binding, its
    window, its number -- or None where it is text there."""
    if member.stype == SymbolType.Data:
      return None
    if getattr(member.obj, 'addressing', None) == Addressing.SCALAR:
      return member.pir_scalar(writer)
    return member.pir_buffer(writer)

  def _extent(self) -> int:
    """How many elements a read through the selected pointer reaches: the
    largest member's, as `GetElementPtr` types a binding."""
    out = 1
    for member in self._members:
      obj = member.obj
      if obj is None or member.stype == SymbolType.Data:
        continue
      if (getattr(obj, 'storage_order', None) is not None
          or not getattr(obj, 'is_dense', lambda: True)()):
        extent = int(obj.storage_volume())
      else:
        view = member.data_view
        extent = 1
        for dim in getattr(view, 'shape', None) or ():
          extent *= int(dim)
      out = max(out, extent)
    return out

  def _declare(self, writer, line: str) -> None:
    if hasattr(writer, 'decl_expr'):
      writer(line, accesses=self._accesses(writer))
    else:
      writer(line)

  def get_operands(self):
    return list(self._members)

  def __str__(self):
    return f'{self._name} = table[{len(self._members)}]'


class VariantLoop(AbstractInstruction):
  """A run stated once, with a counter its varying operands are read by.

  The body is a region like `BatchLoop`'s, so the passes that walk regions
  reach it without knowing what kind of loop this is.  What differs is the
  trip count: it is a constant known at generation, not a bound the caller
  supplies, so the header is a plain counted loop and the counter is uniform
  across the block by construction.

  The tables are declared ahead of the header rather than inside it.  They are
  arrays of the kernel's own arguments and do not change between iterations,
  and a declaration inside the body would rebuild them on every one -- which a
  compiler would very likely hoist, and which there is no reason to write down
  and hope for.

  Like `BatchLoop`, this drives child instructions that route themselves
  through the pseudo-IR, so it overrides `gen_code` rather than `gen_ir`.
  """

  def __init__(self, context: Context, counter: str, count: int,
               region, tables=(), unroll: bool = False, start: int = 0,
               carried=()):
    super(VariantLoop, self).__init__(context)
    if count < 1:
      raise GenerationError(f'a loop runs at least once, given {count}')
    if not 0 <= start < count:
      raise GenerationError(f'a loop starting at {start} of {count} runs '
                            f'no iterations')
    self._counter = counter
    self._start = start
    self._count = count
    self._region = list(region)
    self._tables = list(tables)
    self._unroll = unroll
    #: `(init, result)` per value the body threads through itself.
    #:
    #: `init` is the link the body reads, `result` the link it writes.  The
    #: two being different registers is the whole of what a back edge has to
    #: fix, and an empty tuple here means a body that carries nothing -- not a
    #: body whose carried value went unnoticed.
    self._carried = tuple(carried)
    self._is_ready = True

  @property
  def carried(self):
    return self._carried

  # -- structure ----------------------------------------------------------- #

  @property
  def counter(self) -> str:
    return self._counter

  @property
  def count(self) -> int:
    return self._count

  @property
  def region(self):
    return self._region

  @property
  def tables(self):
    return self._tables

  def regions(self):
    return (tuple(self._region),)

  def barrier_scope(self):
    """A loop containing a barrier synchronizes, seen from outside -- as
    `BatchLoop` says of its own body, which asks this of its instructions.
    Silent, it would leave a staged member's block barriers
    (`Generator._stage_member`) invisible to the block sizing that has to
    allow them."""
    inner = [i.barrier_scope() for i in self._region]
    return max((s for s in inner if s is not None), default=None)

  def replace_region(self, index: int, instrs) -> None:
    if index != 0:
      raise GenerationError(f'a VariantLoop has one region, not {index + 1}')
    self._region = list(instrs)

  def get_operands(self):
    out = []
    for table in self._tables:
      out.extend(table.get_operands())
    return out

  def uses(self) -> Tuple:
    """The members its tables select from, and what its body reads before
    it writes it -- as `BatchLoop` says of its own body."""
    out, seen, defined = [], set(), set()
    for sym in self.get_operands():
      if id(sym) not in seen:
        seen.add(id(sym))
        out.append(sym)
    for instr in self._region:
      for sym in instr.uses():
        if id(sym) not in defined and id(sym) not in seen:
          seen.add(id(sym))
          out.append(sym)
      for sym in instr.defs():
        defined.add(id(sym))
    return tuple(out)

  def defs(self) -> Tuple:
    """What its body defines, and is defined behind the loop as well.

    Which is everything but the bindings its counter selects, which are
    declared in the body and scoped to it.  The loop runs at least once
    (`__init__`), registers are allocated ahead of it
    (`Generator._emit_variant_loop`) and shared windows declared ahead of it
    (`_declare_windows_early`), so an operand the first iteration stages is
    there for whatever reads it after the loop -- a later operation the
    residency or the scopes hand the staged copy.
    """
    scoped = {id(sym) for instr in self._region
              if isinstance(instr, GetElementPtr) for sym in instr.defs()}
    out, seen = [], set(scoped)
    for instr in self._region:
      for sym in instr.defs():
        if id(sym) not in seen:
          seen.add(id(sym))
          out.append(sym)
    return tuple(out)

  def accesses(self) -> Tuple:
    """What its body touches, and the members it reaches through its tables:
    the body's accesses name the stand-ins."""
    out = []
    for table in self._tables:
      for sym in table.get_operands():
        space = MemSpace.from_symbol_type(getattr(sym, 'stype', None))
        if space is not MemSpace.NONE:
          kind = Effect.READ | (Effect.WRITE if table.writable
                                else Effect.NONE)
          out.append(Access(kind, space, sym))
    for instr in self._region:
      out.extend(instr.accesses())
    return tuple(out)

  # -- the counter, as a value ---------------------------------------------- #

  #: `(builder, {name: value})`, innermost last: what the runs open here bind.
  _counters: List[Tuple] = []

  @classmethod
  @contextmanager
  def counter_bound(cls, builder, name, value):
    """Publish this run's counter while its body is open.

    Kept apart from the element indices `BatchLoop` publishes, although a
    reader reaches both the same way. A frame there is the answer to *every*
    name a body asks for, so a second one would leave `batchId0` unbound;
    and a counter is not an element index, which is the distinction that
    registry is careful about.
    """
    frame = (builder, {name: value})
    cls._counters.append(frame)
    try:
      yield
    finally:
      cls._counters.pop()

  @classmethod
  def counter_value(cls, builder, name):
    """The value of `name` where `builder`'s body is inside the run that
    binds it, else `None`."""
    for owner, bound in reversed(cls._counters):
      if owner is builder and name in bound:
        return bound[name]
    return None

  # -- emission ------------------------------------------------------------ #

  @property
  def start(self) -> int:
    return self._start

  def header(self) -> str:
    # `int32_t` and not `int`: the same type, and the spelling the IR renders
    # from `INDEX`, so the two ways this loop can be emitted agree.
    return (f'int32_t {self._counter} = {self._start}; '
            f'{self._counter} < {self._count}; ++{self._counter}')

  def _emit_region(self, writer, per_iteration) -> None:
    for table in per_iteration:
      table.gen_code(writer)
    for instruction in self._region:
      instruction.gen_code(writer)

  def _declare_windows_early(self, writer, instrs, cleared) -> None:
    """Declare the shared windows the body fills ahead of the header.

    The same hoist `BatchLoop` and `Conditional` do out of their own
    regions, for the same reason: a window is declared by the first
    instruction that fills it, and one inside the body would scope the name
    to the body, where everything after the loop that reads the buffer --
    the residency hands a later operation the copy the loop staged, and a
    temporary the loop assigns is read behind it -- names something never
    declared where it stands.  The offset is the allocator's and the same on
    every trip, so the declaration does not depend on the counter.  Nested
    regions too: a guard in the body would declare its windows ahead of the
    guard, which is still inside the loop.
    """
    for instr in instrs:
      declare = getattr(instr, 'gen_code_declare', None)
      if declare is not None and getattr(instr, '_declare', False):
        declare(writer)
        instr._declare = False
        cleared.append(instr)
      for region in instr.regions():
        self._declare_windows_early(writer, region, cleared)

  def gen_code(self, writer) -> None:
    # Restored afterwards: clearing the flag is a statement about this build,
    # and a body built twice would otherwise declare nothing at all.
    cleared = []
    try:
      self._gen_code(writer, cleared)
    finally:
      for instr in cleared:
        instr._declare = True

  def _gen_code(self, writer, cleared) -> None:
    invariant = [t for t in self._tables if t.loop_invariant()]
    per_iteration = [t for t in self._tables if not t.loop_invariant()]
    for table in invariant:
      table.gen_code(writer)
    self._declare_windows_early(writer, self._region, cleared)

    if hasattr(writer, 'for_'):
      # `extern` because the counter's name is the macro layer's: a table
      # built where the counter is no value (`For`, below) spells it as text.
      with writer.for_(self._start, self._count, 1, extern=self._counter,
                       unroll=self._unroll, rolled=True) as loop:
        # The counter as a value, which the select chains and the array
        # accesses read as an operand: an edge back to the loop, so nothing
        # takes them ahead of the header, and something to substitute for a
        # pass that builds a trip's statements for another one.  Named in the
        # text alone it would read as invariant, and `licm` would move the
        # access ahead of the header.
        with self.counter_bound(writer, self._counter, loop.induction):
          self._emit_region(writer, per_iteration)
      return

    with writer.For(self.header(), unroll=self._unroll):
      self._emit_region(writer, per_iteration)

  def __str__(self):
    return (f'for {self._counter} in [{self._start},{self._count}): '
            f'{len(self._region)} instruction(s)')
