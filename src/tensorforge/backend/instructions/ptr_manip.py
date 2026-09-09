# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
import enum

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

  def reads_the_pointer_array(self) -> bool:
    """Does this binding load `m[batchId0]` out of an array of pointers?

    Narrower than `dereferences_the_batch`, which asks about the addressing
    mode alone. Two bindings carry that mode and still read no pointer array
    by the loop's own index: a table-fed one takes its base from a table the
    prologue already built, so the element index does not enter the address at
    all, and a lookahead one names an element the loop has not reached, whose
    pointer is therefore already being asked for.
    """
    return (self.dereferences_the_batch()
            and self._table is None
            and self._batch_offset == 0)

  def source_name(self) -> str:
    """What the right-hand side reads the base pointer out of."""
    if self._table is None:
      return self._src.name
    return self._table.access(self._variant)

  def gen_ir(self, writer):

    if self._table is not None:
      # The table's members are the bindings the prologue already made, so
      # they carry the element offset.  Offsetting again would apply it twice;
      # what varies between iterations is which of them to take, and that is
      # the whole of it.
      datatype = self._vm._fp_type if self._src.obj.datatype is None else self._src.obj.datatype
      lhs = 'const ' if self._src.obj.direction == DataFlowDirection.SOURCE else ''
      lhs += f'{datatype} *const {self._vm.get_lexic().restrict_kw} {self._dest.name}'
      self._emit_binding(writer, lhs, self._table.access(self._variant))
      return

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
          BufferType(self._dest.get_fptype(), (1,), MemSpace.GLOBAL,
                     readonly=self._src.obj.direction
                     == DataFlowDirection.SOURCE),
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


class TableForm(enum.Enum):
  """How a run reaches the member its counter names.

  `SELECT` is a chain of conditionals collapsing to one pointer.  The counter
  is uniform, so each step is a scalar select on both vendors and nothing is
  stored anywhere; the chain is `n - 1` selects long, which is why it is the
  choice for a handful of members and not for many.

  `ARRAY` is an initialised array indexed by the counter.  Constant in the
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
               datatype=None, form: 'TableForm' = None, variant: str = None):
    super(DeclareOperandTable, self).__init__(context)
    if not members:
      raise GenerationError('an operand table has at least one member')
    self._name = name
    self._members = list(members)
    self._addressing = addressing
    self._datatype = datatype
    self._form = form or TableForm.SELECT
    self._variant = variant
    if self._form is TableForm.SELECT and variant is None:
      raise GenerationError('a select chain has to name the counter it reads')
    self._is_ready = True

  @property
  def form(self) -> 'TableForm':
    return self._form

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
    datatype = self._datatype or self._vm._fp_type
    stars = Addressing.addr2ptr_type(self._addressing)
    return (f'struct {self.struct_name()} {{ const {datatype} {stars}const '
            f'p[{len(self._members)}]; }};')

  def parameter(self) -> str:
    """How this table is declared in the kernel's signature, for `PARAM`.

    The annotation is the lexic's, not this site's: what keeps a by-value
    parameter out of per-thread memory is a property of the backend, and on
    most of them the answer is that nothing is needed.
    """
    self._require_param()
    annotation = self._vm.get_lexic().grid_constant_kw
    annotation = f'{annotation} ' if annotation else ''
    return f'{annotation}const {self.struct_name()} {self._name}'

  def argument(self) -> str:
    """How the caller builds it, one launch before the kernel reads it."""
    self._require_param()
    entries = ', '.join(m.name for m in self._members)
    return f'const {self.struct_name()} {self._name} = {{{{{entries}}}}};'

  def gen_ir(self, writer):
    if self._form is TableForm.PARAM:
      # Nothing to emit: the caller filled it and the signature names it.
      return
    datatype = self._datatype or self._vm._fp_type
    stars = Addressing.addr2ptr_type(self._addressing)
    if self._form is TableForm.ARRAY:
      entries = ', '.join(m.name for m in self._members)
      writer(f'const {datatype} {stars}const {self._name}'
             f'[{len(self._members)}] = {{{entries}}};')
      return
    chain = self._members[-1].name
    for index in range(len(self._members) - 2, -1, -1):
      chain = (f'({self._variant} == {index}) ? {self._members[index].name} '
               f': {chain}')
    writer(f'const {datatype} {stars}const {self._name} = {chain};')

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

  @property
  def start(self) -> int:
    return self._start

  def header(self) -> str:
    return (f'int {self._counter} = {self._start}; '
            f'{self._counter} < {self._count}; ++{self._counter}')

  def _emit_region(self, writer, per_iteration) -> None:
    for table in per_iteration:
      table.gen_code(writer)
    for instruction in self._region:
      instruction.gen_code(writer)

  def gen_code(self, writer) -> None:
    invariant = [t for t in self._tables if t.loop_invariant()]
    per_iteration = [t for t in self._tables if not t.loop_invariant()]
    for table in invariant:
      table.gen_code(writer)

    if hasattr(writer, 'for_'):
      # `extern` and `ctype` because the counter's name and type are the macro
      # layer's: the select chains and every table access spell it out as text.
      #
      # And the induction *value* is pushed, not merely its name.  Anything
      # inside that mentions the counter has to say so as an operand, or the
      # IR sees a computation with no inputs -- a select chain over four
      # pointers is exactly that -- and is free to hoist it out of the loop
      # that defines the thing it reads.  Silently, and only in the text.
      with writer.for_(self._start, self._count, 1, extern=self._counter,
                       ctype='int', unroll=self._unroll) as loop:
        AbstractInstruction._induction_value.append(loop.induction)
        try:
          self._emit_region(writer, per_iteration)
        finally:
          AbstractInstruction._induction_value.pop()
      return

    with writer.For(self.header(), unroll=self._unroll):
      self._emit_region(writer, per_iteration)

  def __str__(self):
    return (f'for {self._counter} in [{self._start},{self._count}): '
            f'{len(self._region)} instruction(s)')
