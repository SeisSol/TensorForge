# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
from .abstract_instruction import AbstractInstruction
from tensorforge.common.context import Context
from tensorforge.common.helper import get_extra_offset_name, Addressing
from tensorforge.common.basic_types import GeneralLexicon, DataFlowDirection, StridedAddressing
from tensorforge.common.exceptions import GenerationError
from tensorforge.backend.pir.core import Effect
from tensorforge.backend.pir.core import MemSpace

class GetElementPtr(AbstractInstruction):
  def __init__(self,
               context: Context,
               src,
               dest,
               include_extra_offset=True,
               batch_offset=0,
               update_dest=None,
               pipeline = False,
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
    return f'{self._table.name}[{self._variant}]'

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
      rhs = f'&{self.source_name()}[{address}]'
      lhs = 'const ' if self._src.obj.direction == DataFlowDirection.SOURCE else ''
      lhs += f'{datatype} *{const_mod} {self._vm.get_lexic().restrict_kw} {self._dest.name}'
    if batch_addressing == Addressing.STRIDED:
      # distance between batch elements is the *stored* volume, i.e.
      # prod(upper - lower), not prod(shape)
      main_offset = f'{self.batch_index()} * {batch_obj.storage_volume()}'
      sub_offset = f'{batch_obj.get_offset_to_first_element()}'
      address = f'{main_offset} + {sub_offset}{extra_offset}'
      rhs = f'&{self.source_name()}[{address}]'
      lhs = 'const ' if self._src.obj.direction == DataFlowDirection.SOURCE else ''
      lhs += f'{datatype} *{const_mod} {self._vm.get_lexic().restrict_kw} {self._dest.name}'
    elif batch_addressing == Addressing.PTR_BASED:
      main_offset = f'{self.batch_index()}'
      sub_offset = f'{batch_obj.get_offset_to_first_element()}'
      address = f'{main_offset}][{sub_offset}{extra_offset}'
      src_suffix = '_ptr' if self._vm.get_lexic()._backend == 'targetdart' else ''
      rhs = f'&{self.source_name()}{src_suffix}[{address}]'
      lhs = 'const ' if self._src.obj.direction == DataFlowDirection.SOURCE else ''
      if self._context.get_vm().get_hw_descr().vendor == 'amd':
        lhs += f'{datatype}'
        rhs = f'(tensorforge::SpacePtrRestrict<{lhs}, tensorforge::GlobalMemspace>){rhs}'
        lhs = f'auto {self._dest.name}'
      else:
        lhs += f'{datatype} *{const_mod} {self._vm.get_lexic().restrict_kw} {self._dest.name}'
    elif batch_addressing == Addressing.NONE:
      address = f'{batch_obj.get_offset_to_first_element()}'
      rhs = f'&{self.source_name()}[{address}]'
      lhs = 'const ' if self._src.obj.direction == DataFlowDirection.SOURCE else ''
      lhs += f'{datatype} *{const_mod} {self._vm.get_lexic().restrict_kw} {self._dest.name}'
    elif batch_addressing == Addressing.SCALAR:
      rhs = f'{self.source_name()}'
      lhs = f'{datatype} {self._dest.name}'
    else:
      GenerationError(f'unknown addressing of {self._src.name}, given {batch_addressing}')

    if self._update_dest:
      writer(f'const auto {self._update_dest.name} = {self._dest.name};')
      writer(f'{self._dest.name} = {rhs};')
    else:
      self._emit_binding(writer, lhs, rhs)

  def _emit_binding(self, writer, lhs: str, rhs: str) -> None:
    """The binding, as a definition rather than a statement.

    It was a bare statement, so `Effect.UNKNOWN`, so it conflicted with every
    access in the body and pinned everything on both sides of it -- the
    largest blocking site after `allocate.py`.  That matters here rather than
    in the abstract: `WrapLoads` moves a transfer past the instructions
    between it and its consumer, and a binding that conflicts with everything
    is a wall in the middle of exactly that stretch.

    Declaring only its accesses made it reorderable but still nameless, so it
    had to stay pinned anyway: a consumer reading `glb_m1` did so through text
    the IR could not see, and letting the binding sink below one would compile
    to a use before its definition.  Producing a *value* is what removes that
    reason -- the def-use edge exists, so the scheduler knows the distance it
    may not close.

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
    """
    if hasattr(writer, 'decl_expr'):
      from tensorforge.backend.pir.core import BufferType
      from .abstract_instruction import AbstractInstruction
      # Name the element index as an operand where there is one to name.  The
      # address is `&m2[batchId0 * 324 + ...]`, and with `batchId0` only in
      # the text a pass that moves this binding to another element has nothing
      # to substitute -- which is exactly why `wrap_prefetch` could not
      # advance a transfer that reads through it.
      text = rhs.replace('{', '{{').replace('}', '}}')
      args = ()
      ind = (AbstractInstruction._induction_value[-1]
             if AbstractInstruction._induction_value else None)
      # Only for the loop's *own* index.  `batch_index()` is `batchId1` for a
      # lookahead binding -- the peeled prologue's -- and substituting the
      # induction there rewrites element k+1 back to element k, silently.
      if (ind is not None and self._batch_offset == 0
              and self.batch_index() in rhs):
        text = text.replace(self.batch_index(), '{0}')
        args = (ind,)
      value = writer.decl_expr(
          lhs, text,
          BufferType(self._dest.get_fptype(), (1,), MemSpace.GLOBAL),
          self._src, args=args, kind=Effect.READ, hint=self._dest.name,
          extern=self._dest.name, alias_root=self._src)
      self._dest.set_pir_buffer(writer, value)
    else:
      writer(f'{lhs} = {rhs};')

  def defs(self):
    return (self._dest,) if self._update_dest is None else (self._dest, self._update_dest)

  def uses(self):
    return (self._src,)

  def __str__(self) -> str:
    return (f'{self._dest.name} = getelementptr_b2g {self._src.name} '
            f'[{self.batch_index()}];')


class DeclareOperandTable(AbstractInstruction):
  """An array of the kernel's own arguments, indexed by a loop counter.

  A run whose operand changes between iterations needs that operand reachable
  by index.  Where its members are already parameters -- which is the case a
  frontend that wrote the repetition out always produces, since it named every
  one of them -- nothing has to reach the interface: the table is built inside
  the kernel from what is already there, costs one initialised array, and the
  index into it is uniform, so the load stays on the scalar path.

  The element type follows the members' addressing rather than being chosen
  here.  A pointer-based argument is already an array per element, so a table
  over such members is one indirection deeper than a table over batch-invariant
  ones, and getting that wrong is a type error at compile time rather than a
  wrong address at run time.
  """

  def __init__(self, context: Context, name: str, members, addressing,
               datatype=None):
    super(DeclareOperandTable, self).__init__(context)
    if not members:
      raise GenerationError('an operand table has at least one member')
    self._name = name
    self._members = list(members)
    self._addressing = addressing
    self._datatype = datatype
    self._is_ready = True

  @property
  def name(self) -> str:
    return self._name

  def __len__(self) -> int:
    return len(self._members)

  def gen_ir(self, writer):
    datatype = self._datatype or self._vm._fp_type
    stars = Addressing.addr2ptr_type(self._addressing)
    entries = ', '.join(m.name for m in self._members)
    writer(f'const {datatype} {stars}const {self._name}[{len(self._members)}]'
           f' = {{{entries}}};')

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
               region, tables=(), unroll: bool = False):
    super(VariantLoop, self).__init__(context)
    if count < 1:
      raise GenerationError(f'a loop runs at least once, given {count}')
    self._counter = counter
    self._count = count
    self._region = list(region)
    self._tables = list(tables)
    self._unroll = unroll
    self._is_ready = True

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

  def replace_region(self, index: int, instrs) -> None:
    if index != 0:
      raise GenerationError(f'a VariantLoop has one region, not {index + 1}')
    self._region = list(instrs)

  def get_operands(self):
    out = []
    for table in self._tables:
      out.extend(table.get_operands())
    return out

  # -- emission ------------------------------------------------------------ #

  def header(self) -> str:
    return (f'int {self._counter} = 0; {self._counter} < {self._count}; '
            f'++{self._counter}')

  def gen_code(self, writer) -> None:
    for table in self._tables:
      table.gen_code(writer)
    with writer.For(self.header(), unroll=self._unroll):
      for instruction in self._region:
        instruction.gen_code(writer)

  def __str__(self):
    return (f'for {self._counter} in [0,{self._count}): '
            f'{len(self._region)} instruction(s)')
