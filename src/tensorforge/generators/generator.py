# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
from typing import List, Optional, Union, Type
from copy import deepcopy
import hashlib
from tensorforge.generators.descriptions import ForDescr, OperationDescription, MultilinearDescr, ElementwiseDescr, RegionDescription, ReductionDescr
from tensorforge.common.context import Context
from tensorforge.common.basic_types import Addressing, FlagMode, GeneralLexicon, DataFlowDirection
from tensorforge.common.helper import get_extra_offset_name
from tensorforge.backend.data_types import ShrMemObject, RegMemObject
from tensorforge.backend import pir
from tensorforge.backend.opt import OptimizationStage
from tensorforge.backend.opt.inspect import async_depth, format_diagnostics, verify
from tensorforge.backend.scopes import Scopes
from tensorforge.backend.residency import Residency
from tensorforge.backend.section_plan import SectionPlan
from tensorforge.generators import lanes as lane_config
from tensorforge.generators.lanes import LaneConfig
from tensorforge.backend.temporaries import Temporaries
from tensorforge.backend.symbol import Symbol, SymbolType
from tensorforge.backend.instructions.abstract_instruction import AbstractInstruction
from tensorforge.backend.instructions.builders.loader_builder import GlobalLoaderBuilder
from tensorforge.backend.instructions.builders.multilinear_builder import MultilinearBuilder
from tensorforge.backend.instructions.builders.pointwise_builders import (
    ElementwiseBuilder, ReductionBuilder)
from tensorforge.backend.instructions.builders.ptr_manip_builder import GetElementPtrBuilder
from tensorforge.backend.instructions.builders.allocator_builder import ShrMemAllocBuilder
from tensorforge.backend.instructions.control.conditional import GuardedRegion
from tensorforge.backend.instructions.sync_block import SyncThreads, SyncBlock, SyncGrid
from tensorforge.backend.instructions.batch_loop import BatchLoop, LoopMode
from tensorforge.backend.writer import Writer
from tensorforge.common.exceptions import GenerationError, InternalError
from tensorforge.common.threads import mults_per_group
from tensorforge.generators.identity import registry

import tensorforge.interop as interop

class AbstractThreadBlockPolicy:
  def __init__(self, context: Context, global_mem: int, mem_per_mult: int, num_threads: int):
    self._context: Context = context
    self._mem_per_mult: int = mem_per_mult
    self._global_mem: int = global_mem
    self._num_threads: int = num_threads
    #: Whether the section this sizes a block for contains a barrier at all.
    self._has_barrier: bool = False

    vm = self._context.get_vm()
    self._max_blocks = vm.get_hw_descr().max_block_per_sm
    self._max_allowed_mem = vm.get_hw_descr().max_local_mem_size_per_block
    self._max_threads = vm.get_hw_descr().max_threads_per_block

  def get_num_mults_per_block(self):
    pass

  def _barrier_cap(self):
    """How many multiplications a block may hold and still separate them.

    The cap is about a barrier, so it is asked only of a section that has one
    -- `set_has_barrier`.  A section whose multiplications never rendezvous is
    packed by memory and threads alone, which is most of the register-resident
    paths.

    Where there is a barrier, the block is sized so that the strongest barrier
    the target can spell is exactly the set that has to meet.  A target with a
    sub-block rendezvous can separate one multiplication from the next, so
    nothing is capped; without one, the smallest separable set is the group of
    multiplications that fills a whole number of waves, and the block is sized
    to hold one of those.

    `None` is no cap, which is different from a cap of one.
    """
    if not self._has_barrier:
      return None
    vm = self._context.get_vm()
    wave = vm.get_hw_descr().vec_unit_length
    if self._num_threads == wave:
      return None
    if vm.get_lexic().has_sync_mult(self._num_threads, vm.get_hw_descr()):
      return None
    return mults_per_group(self._num_threads, wave)

  def set_has_barrier(self, has_barrier: bool) -> None:
    self._has_barrier = bool(has_barrier)


class RegmaxBlockPolicy(AbstractThreadBlockPolicy):
  def __init__(self, context, global_mem, mem_size_per_mult, num_threads,
               lead_width=1):
    super().__init__(context, global_mem, mem_size_per_mult, num_threads)
    #: Lanes times width is what the lane count *was* before the lead
    #: dimension was vectorised, and it is the right divisor here.
    #:
    #: This is the whole occupancy story of the vectorisation, so it is worth
    #: stating: `256 // num_threads` binds in every case in the corpus -- the
    #: memory bound never does -- so halving the lane count would otherwise
    #: double the mults, double the shared memory per block and halve the
    #: occupancy above roughly 256 elements per mult.  Holding the mults
    #: instead makes the block smaller: shared memory per block unchanged,
    #: blocks per SM unchanged or better, and the same work in flight with
    #: half the instructions issued to do it.
    #: Width times blocking: how many lead-dimension elements one lane now
    #: covers where it used to cover one.  `num_threads * this` is the lane
    #: count the operators started with, which is what `mults_per_block` has
    #: to be sized from -- sizing it from the *reduced* count would double
    #: the mults, double the shared memory per block and halve the occupancy,
    #: spending the whole win on memory.
    self._lane_factor = max(1, lead_width)

  def get_num_mults_per_block(self):
    # the //2 is a heuristic
    # self._max_threads // self._num_threads // 2
    max_thread_mults = 256 // (self._num_threads * self._lane_factor)
    if self._mem_per_mult == 0:
      mults = max_thread_mults
    else:
      max_mem_mults = (self._max_allowed_mem - self._global_mem * self._context.fp_type.size()) // (self._mem_per_mult * self._context.fp_type.size())
      mults = min(max_mem_mults, max_thread_mults)
    cap = self._barrier_cap()
    return mults if cap is None else min(mults, cap)

class Section:
  def __init__(self):
    # `global_ir` and `ir` are what the builders author: the section prologue
    # and the per-element body.  `stream` is the optimised result -- prologue
    # followed by one BatchLoop carrying the body as its region -- and is what
    # gets emitted.
    self.ir: List[AbstractInstruction] = []
    self.global_ir: List[AbstractInstruction] = []
    self.stream: List[AbstractInstruction] = []
    self.shr_mem_obj: Union[ShrMemObject, None] = None
    self.scopes: Scopes = Scopes()
    self.barrier = False

class _GuardGrouping:
  """Collects the instructions of neighbouring operations under one guard.

  An operation's guard is a conjunction of literals, and two operations that
  state the same conjunction run under the same condition -- so their
  instructions go into one region and the condition is evaluated once. The
  run ends where the conjunction changes, which keeps the emitted order the
  order the descriptor list states.
  """

  def __init__(self, context, scopes, residency, out):
    self._context = context
    self._scopes = scopes
    self._residency = residency
    self._out = out
    self._key = None
    self._literals = []
    self._pending = []

  @staticmethod
  def _key_of(descr):
    """What makes two guards the same one.

    `None` for an operation that always runs, so an unguarded stretch never
    joins a guarded one. The literals are sorted: a conjunction is a set, and
    two operations may state the same one in different orders.
    """
    condition = getattr(descr, 'condition', None)
    if not condition:
      return None
    return tuple(sorted(literal.key() for literal in condition))

  def _resolve(self, descr):
    return [(self._scopes.get_symbol(literal.tensor.tensor), literal.negated)
            for literal in descr.condition]

  def add(self, descr, instrs) -> None:
    key = self._key_of(descr)
    if key != self._key:
      self.flush()
      self._key = key
      self._literals = self._resolve(descr) if key is not None else []
      # The condition is read where the region opens, which is before its
      # body runs. A value another operation left in registers has to reach
      # its buffer *here* -- `prepare_operands` flushes an operation's own
      # operands, and a guard's operands are not among them.
      for symbol, _ in self._literals:
        self._out.extend(self._residency.flush(symbol.name))
    self._pending.extend(instrs)

  def flush(self) -> None:
    if self._pending:
      if self._key is None:
        self._out.extend(self._pending)
      else:
        self._out.append(GuardedRegion(self._context, self._literals,
                                       self._pending))
    self._key = None
    self._literals = []
    self._pending = []


def _supports_launch_control(context) -> bool:
  """Does this target have `clusterlaunchcontrol.try_cancel`?

  PTX ISA 8.6, and the CCCL wrappers gate the instruction on
  `__CUDA_ARCH__ >= 1000`, so a lower target does not fail to compile -- it
  fails to *link*, against a stub named
  `__cuda_ptx_clusterlaunchcontrol_try_cancel_is_not_supported_before_SM_100__`.
  Which is a decent error to read and a bad one to reach from a switch, so the
  question is answered here.

  The arch names are `sm_NN` throughout `hw_descr_db.yml`, two or three digits
  and no `a`/`f` suffix, so the numeric part orders them; `sm_120` answers yes
  and `sm_90` no, which is the split the instruction actually has.  Verified on
  an sm_120 part: the plain target assembles it, no `sm_120a` needed.
  """
  hw = context.get_vm().get_hw_descr()
  if hw.vendor != 'nvidia':
    return False
  model = hw.model
  if not model.startswith('sm_') or not model[3:].isdigit():
    return False
  return int(model[3:]) >= 100


class Generator:
  #: Hex characters of the digest that end up in the symbol.  Sixty-four bits
  #: rather than forty: the digest is the whole of the name's discriminating
  #: power, and a corpus of 10^5 kernels has a percent-level chance of a
  #: birthday collision at forty.  A collision is caught (`identity.registry`)
  #: rather than silent, but being caught means a build that stops.
  NAME_ENCODING_LENGTH = 16

  #: What stands in for the kernel name while the source that determines it is
  #: being written.  A C++ identifier, so that the intermediate text is still
  #: the text it will be -- the substitution has to leave everything except
  #: this token where it was.
  NAME_PLACEHOLDER = 'TENSORFORGE_UNNAMED_KERNEL'

  def __init__(self,
               gemm_list: List[OperationDescription],
               context: Context,
               thread_block_policy_type: Type[AbstractThreadBlockPolicy] = RegmaxBlockPolicy,
               lanes: Optional[LaneConfig] = None,
               attrs: Optional[dict] = None):
    self.descr_list: List[OperationDescription] = gemm_list
    self._context: Context = context
    #: Destination names whose transfer should get two stages, or None to
    #: work that out.  Set to a concrete set on the throwaway generator that
    #: works it out, which is what stops it recursing.
    self._rotate: Optional[set] = None
    #: Switches the frontend's caller set on this kernel, or None from a
    #: frontend that has no attribute channel.  Only the flag mask reads
    #: these; the distinction between None and {} is what keeps a frontend
    #: without one generating the same kernels it did before.
    self._attrs: Optional[dict] = attrs
    self._flags: FlagMode = FlagMode.from_attrs(attrs)
    self._thread_block_policy_type: Type[AbstractThreadBlockPolicy] = thread_block_policy_type
    self._base_kernel_name: Union[str, None] = None
    #: Whether a completed generation announces its name to the process-wide
    #: registry.  Off for the generators that only exist to be asked a
    #: question -- their bodies are built under settings nothing will emit, so
    #: recording them fills the registry with kernels that reach no file.
    self._announce_identity: bool = True

    self._kernel = None
    self._launcher = None
    self._header = None

    self._matrix_list = None
    self._tmp_list = None
    self._scopes: Scopes = Scopes()
    self._is_registerd: bool = False
    #: Tables substituted into the kernel's signature in place of their
    #: members.  Empty unless something registers one, so a generator that
    #: never sees a repeated run emits exactly what it did before.
    self._param_tables = []
    self._table_member = {}
    #: Whether a `ForDescr` becomes a loop or is expanded into its iterations.
    #:
    #: Off, because the loop does not verify yet: a body built once reads its
    #: accumulator at the top of the first iteration, and the definition that
    #: reaches it is the one the *previous* iteration made -- which is not a
    #: definition the verifier can see, since nothing carries a value across
    #: the back edge.  Expansion is meanwhile exact, so leaving it on would
    #: trade working code for a diagnostic.
    self._emit_loops = False

    self._num_threads: int = 0
    self._num_active_threads: int = 0
    self._lead_width: int = 1
    #: An explicit lane geometry, or None to take the one the descriptors ask
    #: for.  Overriding it is how a caller says "build this with 64 lanes
    #: instead" -- the thing a search over configurations needs and a constant
    #: cannot offer.
    self._lanes: Optional[LaneConfig] = lanes
    #: Peak register footprint over this kernel's bodies, in bytes per lane,
    #: or None when the context did not ask for it.  A maximum and not a sum,
    #: because the budget is per kernel and the widest body is what has to
    #: fit.
    self.peak_pressure: Optional[int] = None
    #: Blocks resident per SM under the resources that are known exactly --
    #: shared memory and threads.  Not the register limit; see
    #: `_resident_blocks`.
    self.resident_blocks: Optional[int] = None

    self._section: Section = Section()
    self._sections: List[Section] = []

    if context.get_user_options().merge_variants:
      # Before the operands are named, which is the first thing that reads the
      # list -- a stand-in arriving after it has no name and then no symbol,
      # and the failure is a table built from `None` several phases later.
      from tensorforge.generators.rolling import roll
      options = context.get_user_options()
      self.descr_list = roll(self.descr_list,
                             min_count=options.merge_min_count,
                             max_arity=options.merge_max_arity)
      self._emit_loops = True

    self._name_operands(self.descr_list)

    # Asked for and unavailable is an error, not a fallback.  A caller who
    # switched the traversal and silently got the other one would attribute
    # the grid-stride loop's numbers to the queue.
    if context.get_user_options().launch_control and not _supports_launch_control(context):
      hw = context.get_vm().get_hw_descr()
      raise GenerationError(
          f'launch_control needs `clusterlaunchcontrol`, which is sm_100 and '
          f'above; this target is {hw.vendor} {hw.model}')
    prefer_launchcontrol = context.get_user_options().launch_control
    prefer_persistent = not prefer_launchcontrol
    # The vendor rule is the default and not the decision; it is carried by the
    # option's declaration, and a caller asking either way overrides it there,
    # so that a sweep can price both.
    prefer_preload = (context.get_user_options().preload_globals
                      and not prefer_launchcontrol)

    self._persistent_threading = prefer_persistent
    self._preload_globals = prefer_preload

    self._clusterlaunchcontrol = prefer_launchcontrol
    self._launch_control_depth = context.get_user_options().launch_control_depth

    if prefer_launchcontrol:
      # The queue answers with a CTA id nothing can predict, so `batchId1`,
      # which every prefetch pass reads, is `batchId_start + stride` and names
      # an element this block will never be handed.  Nothing crashes: the
      # transfer lands in the buffer the next iteration reads, and every
      # element after the first is computed from another element's operands.
      # Refused rather than silently wrong, until the lookahead index comes
      # out of the queue instead of out of the stride.
      options = context.get_user_options()
      for name in ('enable_wrap_loads', 'enable_pipeline', 'enable_multibuffer'):
        if getattr(options, name):
          raise GenerationError(
              f'{name} prefetches element `batchId1 = batchId_start + stride`, '
              f'which under launch_control is not the element the queue hands '
              f'out next; the two cannot be combined until the lookahead index '
              f'is read from the queue (needs launch_control_depth >= 2)')

  def set_kernel_name(self, name):
    self._base_kernel_name = name

  def flag_mode(self) -> FlagMode:
    """Whether this kernel takes a per-element flag mask, and how.

    The launcher signature follows from it, so a caller that emits the call
    site (a test driver, a frontend) has to be able to ask.
    """
    return self._flags

  def register(self):
    self._collect_tmp_matrices()
    self._populate_global_scope()

  def _set_threadconfig(self):
    # Top level only, which is the prologue plus the loop itself.  The default
    # is a no-op; only a blockwide GlbToShrLoader overrides it, and those live
    # in the prologue.
    mults = self._section.shr_mem_obj.get_mults_per_block()
    for instr in self._section.stream:
      instr.set_threadconfig_pre(self._num_threads, mults)

  def _rotation_targets(self) -> set:
    """Which transfers should get a second buffer, asked of the pass itself.

    `ShrMemOpt` sizes the arena before a body exists, so the decision has to
    be made in advance -- and the only exact answer comes from
    `wrap_prefetch`, which needs the body.  So the section is built once to
    ask and once to use the answer.

    `tools/rotation_cost.py` is why it is this way round rather than giving
    every async transfer two stages: that costs 9% of arena on average and
    25-29% on the kernels with several transfers, which are the ones a
    pipeline is for, and shared memory is paid per launch where a second
    build is paid once.

    The cheap part is knowing when not to ask.  A description list with no
    shared async transfer cannot benefit, and being wrong about *that* costs
    a needless query rather than a buffer nobody uses.
    """
    from tensorforge.backend.pir import wrap as _wrap
    from tensorforge.backend.instructions.memory.load import GlbToShrLoader

    names: set = set()
    original = _wrap.wrap_prefetch

    def asking(body, make_value, next_index=None, report=None,
               assume_rotated=False):
      before = original(body, make_value, next_index, [], assume_rotated=True)
      for stmt, _ in pir.walk(before):
        if stmt.op is pir.Op.FOR and stmt.target:
          for x, _ in pir.walk((stmt,)):
            if x.op in (pir.Op.COPY_ASYNC, pir.Op.LOAD_ASYNC) and x.args:
              base = getattr(x.args[0], 'hint', None)
              if base:
                names.add(base)
      return original(body, make_value, next_index, report, assume_rotated)

    probe = Generator(self.descr_list, self._context, attrs=self._attrs)
    probe._rotate = set()
    probe._announce_identity = False
    _wrap.wrap_prefetch = asking
    try:
      probe.generate()
    except Exception:
      return set()
    finally:
      _wrap.wrap_prefetch = original
    if not names:
      return names

    # Ask again, this time of the body the answer *produces*.  The first probe
    # runs unrotated, so the windows are static and declared ahead of the loop;
    # granting the rotation then declares the write window *inside* it, because
    # its offset moves with the stage counter -- and that is one of the pass's
    # refusal conditions.  So a transfer could be accepted while unrotated,
    # rotated on the strength of that, and then declined for a reason the
    # rotation itself created.
    #
    # Rotated-and-not-wrapped is not a missed optimisation, it is wrong code:
    # the compute reads stage `pipeStage % 2` and the transfer fills the other
    # one, so no iteration ever fills the stage it reads and the first element
    # computes from whatever the arena held.  `trans_a` did exactly that.
    #
    # Hence the invariant this restores: rotated if and only if wrapped.
    confirmed: set = set()
    original2 = _wrap.wrap_prefetch

    def confirming(body, make_value, next_index=None, report=None,
                   assume_rotated=False):
      # `assume_rotated=True`: the buffers really are rotated in this build, so
      # the refusal that exists only for a single copy does not apply.
      after = original2(body, make_value, next_index, report, True)
      for stmt, _ in pir.walk(after):
        if stmt.op is pir.Op.FOR and stmt.target:
          for x, _ in pir.walk((stmt,)):
            if x.op in (pir.Op.COPY_ASYNC, pir.Op.LOAD_ASYNC) and x.args:
              base = getattr(x.args[0], 'hint', None)
              if base:
                confirmed.add(base)
      return after

    check = Generator(self.descr_list, self._context, attrs=self._attrs)
    check._rotate = set(names)
    check._announce_identity = False
    _wrap.wrap_prefetch = confirming
    try:
      check.generate()
    except Exception:
      return set()
    finally:
      _wrap.wrap_prefetch = original2
    return names & confirmed

  def _apply_rotation(self, loop) -> None:
    """Give the chosen transfers two stages, before anything is allocated."""
    if self._rotate is None:
      return
    if not self._rotate:
      return
    from tensorforge.backend.instructions.memory.load import GlbToShrLoader
    if not hasattr(loop, 'request_stage_counter'):
      return
    # Request it, not merely name it.  `stage_counter_name()` answers what the
    # counter is called; `_declare_stage_counter` only emits one when a depth
    # has been requested.  Naming it without requesting it produced kernels
    # that read `pipeStage0` and never declared it -- which renders, and which
    # nothing in the suite compiles, because the syntax check runs on
    # snapshots taken with this flag off.
    stage = loop.request_stage_counter(2)
    for instr in getattr(loop, 'region', []) or []:
      if not isinstance(instr, GlbToShrLoader):
        continue
      if instr._dest.name not in self._rotate:
        continue
      instr.set_stages(2, f'{stage} % 2', f'({stage} + 1) % 2')

  def generate(self):
    if (self._rotate is None
        and self._context.get_user_options().enable_wrap_loads):
      self._rotate = self._rotation_targets()
    # Reset rather than only read at the end: a context outlives one generator
    # -- a search builds several against the same one -- so a figure left over
    # from a previous build would be attributed to this one, and a maximum
    # never falls back on its own.
    self._context.peak_pressure = None

    self.register()

    self._deduce_num_threads()

    descrlist = []
    currlist = []
    barrier = []
    for descr in self.descr_list:
      if descr.barrier():
        # avoid empty sections
        if len(currlist) > 0:
          descrlist += [currlist]
          barrier += [descr.trueBarrier()]
        elif len(barrier) > 0:
          barrier[-1] = barrier[-1] or descr.trueBarrier()
        currlist = []
      else:
        currlist += [descr]
    if len(currlist) > 0:
      descrlist += [currlist]
      barrier += [False]

    for codesection, lastbarrier in zip(descrlist, barrier):
      scopecnt = self._scopes.get_num_scopes()
      self._scopes.add_scope()
      self._section = Section()

      self._emit_global_ir()
      self._emit_ir(codesection)

      # Build the loop *before* optimising, so that the passes see one stream
      # with the body as a region.  This is what removes the
      # `_global_instrs` side channel: a pipelining pass that wants a prologue
      # now peels an iteration into this same list, ahead of the loop, instead
      # of publishing it through a second list nothing else indexed.
      index = len(self._sections)
      start, stride = self._section_traversal(index)
      loop = BatchLoop(context=self._context,
                       section_index=index,
                       mode=self._batch_loop_mode(),
                       start=start,
                       stride=stride,
                       region=self._section.ir,
                       flags=self._flags,
                       queue_depth=self._launch_control_depth,
                       group_size=self._group_size(index, start))

      # The prologue stays *out* of the rewritable stream.  Its shared-memory
      # symbols are allocated by ShrMemObject.alloc_global, a separate bump
      # allocator in a separate arena, so letting them reach the region
      # allocator gives them a second, conflicting offset -- observable as the
      # preloaded operators moving from totalShrMem into localShrMem0.  The
      # optimiser reads the prologue (for symbols live on entry) but never
      # rewrites it.
      #
      # A peeled prologue from a pipelining pass belongs *here*, ahead of the
      # loop in `instructions`, not in the section prologue.
      self._apply_rotation(loop)
      opt = OptimizationStage(context=self._context,
                              shr_mem=self._section.shr_mem_obj,
                              instructions=[loop],
                              num_threads=self._num_threads,
                              scopes = self._scopes,
                              global_ir = self._section.global_ir)
      opt.optimize()
      self._section.stream = list(self._section.global_ir) + opt.get_instructions()

      # Final sync for persistent threads, appended *after* optimisation on
      # purpose: SyncThreadsOpt drops barriers it considers redundant, and this
      # one guards the next iteration's writes against the previous
      # iteration's reads -- a dependency across the back edge that the pass
      # does not model.  Adding it before optimisation removes it again.
      #
      # `LAUNCHCTRL` is excluded, and not because it needs the separation less.
      # It gets it from the hand-off, which carries a block barrier outside the
      # size guard between one element's body and the next.  Appending a second
      # one puts it *inside* that guard, where the rows of a block decide the
      # predicate differently and a barrier is reached by some of them only.
      if self._persistent_threading:
        loop.append(SyncThreads(self._context, self._num_threads))

      self._deduce_mults_per_block()
      self._set_threadconfig()

      if lastbarrier:
        self._section.barrier = True

      while scopecnt < self._scopes.get_num_scopes():
        self._scopes.remove_scope()
      self._sections += [self._section]

    # Write the source before naming it: the name is the digest of what comes
    # out, so it cannot be known until it has.  Everything the three emitters
    # would spell as the name spells the placeholder instead, and
    # `_resolve_identity` puts the real one in afterwards.
    pinned = self._base_kernel_name is not None
    if not pinned:
      self._base_kernel_name = Generator.NAME_PLACEHOLDER

    self._generate_kernel()
    self._generate_launcher()
    self._generate_header()

    self._resolve_identity(pinned)

  def _verify_section(self, stream, index: int) -> None:
    """Structural check over one section, immediately before emitting it.

    One call over one stream: the per-element loop is a ``BatchLoop``
    instruction, so ``verify`` recurses into its region and derives the legal
    barrier scope from ``uniform_scope`` rather than the caller passing a flag.
    """
    # `check_ready` needs the thread-block policy, which has run by now: this
    # is the emit-time call the pass manager's comment defers to, and which
    # nothing was actually making.
    diags = verify(stream,
                   predefined=list(self._scopes.get_global_scope().values()),
                   backend=self._context.get_vm().get_lexic()._backend,
                   check_ready=True)
    errors = [d for d in diags if d.severity == 'error']
    if errors:
      raise GenerationError(
          f'section {index} cannot be emitted:\n{format_diagnostics(errors)}')

  def _section_traversal(self, index: int):
    """``(start, stride)`` for one section's element traversal.

    Sections after a barrier restart at the block id; sections that follow
    without one are offset by the preceding element counts so that consecutive
    sections do not all hammer the same elements.
    """
    vm = self._context.get_vm()
    offset = []
    idx = index - 1
    for ssection in reversed(self._sections[:index]):
      if ssection.barrier:
        break
      offset += [f'{GeneralLexicon.NUM_ELEMENTS}{idx}']
      idx -= 1

    stride = f'({vm.get_lexic().grid_dim_x} * {vm.get_lexic().block_dim_y})'
    if len(offset) == 0:
      start = self._get_2d_block_id()
    else:
      start = f'({self._get_2d_block_id()} + {" + ".join(offset)}) % {stride}'
    return start, stride

  def _group_size(self, index: int, start: str) -> int:
    """How many rows this section's traversal drives together.

    `1` where the multiplication divides the wave evenly -- each one is then
    its own group and the traversal stays per row -- and `1` for a rotated
    start, which is the restriction worth stating.  A rotated start is taken
    modulo the stride, and the modulo does not distribute over the lane offset:
    the leader's rotated start plus a lane is not the row's own rotated start
    once the wrap falls between them, so rows would collide on one element and
    miss another.  Lifting it needs the rotation to move whole groups, which is
    a change to what the offset means rather than to how it is spelled.
    """
    wave = self._context.get_vm().get_hw_descr().vec_unit_length
    if self._num_threads <= wave:
      return 1
    if start != self._get_2d_block_id():
      return 1
    return mults_per_group(self._num_threads, wave)

  def _batch_loop_mode(self) -> LoopMode:
    if self._persistent_threading:
      return LoopMode.PERSISTENT
    if self._clusterlaunchcontrol:
      return LoopMode.LAUNCHCTRL
    return LoopMode.SINGLE

  def _generate_kernel(self):
    vm = self._context.get_vm()

    writer = Writer()
    # Ahead of the signature that names them, and ahead of the launcher that
    # builds one: both sit in this translation unit, and the kernel comes
    # first in it.
    for definition in self.param_table_types():
      writer(definition)
    if self._param_tables:
      writer.new_line()
    with self._generate_kernel_proto(writer):
      self._write_kernel_meta_data(writer)

      for i,section in enumerate(self._sections):
        with writer.AnonymousScope():
          if self._context.get_vm().get_hw_descr().has_cuda_pipeline():
            # Size the pipeline to the transfers that may be outstanding at
            # once.  cuda::make_pipeline() yields a single stage, so a second
            # producer_acquire() before the matching consumer_wait() blocks on a
            # slot that never frees -- a hang rather than a wrong answer.  Two
            # independent things reach that state: software pipelining commits a
            # peeled transfer before the loop, and a body holding two
            # shared-memory loads commits twice before waiting.  The latter is
            # not new; trans_a already did it.
            # Declared whether or not anything drives it.  Whether a transfer
            # takes the structured path is decided per body, from whether its
            # buffers are values there, and this runs before any body exists --
            # so the choice cannot be made here without building the section
            # twice.  An unused local is the price, and it is the safe
            # direction: under-declaring is a compile error, over-declaring is
            # a line nvcc drops.
            # The condition is the target's, not the vendor's, because that
            # last sentence has a floor: below sm_70 the type does not exist
            # and an unused declaration is a compile error too.
            depth = async_depth(section.stream)
            if depth > 1 and False: # disabled for now (not needed for thread_scope_thread)
              # NOT __shared__: the scope is thread, so the state is private and
              # each thread needs its own.  Putting a thread-scope state in
              # shared memory would have every thread of the block driving one
              # FIFO.
              writer(f'cuda::pipeline_shared_state<cuda::thread_scope_thread, {depth}> pipelineState;')
              writer(f'cuda::pipeline<cuda::thread_scope_thread> pipeline = cuda::make_pipeline(cooperative_groups::this_thread(), &pipelineState);')
            else:
              writer(f'cuda::pipeline<cuda::thread_scope_thread> pipeline = cuda::make_pipeline();')

          start, stride = self._section_traversal(i)

          writer(f'const auto {GeneralLexicon.BATCH_ID_NAME}_start = {start};')
          writer(f'const auto {GeneralLexicon.BATCH_ID_NAME}1 = {GeneralLexicon.BATCH_ID_NAME}_start < {GeneralLexicon.NUM_ELEMENTS}{i} ? {GeneralLexicon.BATCH_ID_NAME}_start : 0;')
          writer(f'const auto {GeneralLexicon.BATCH_ID_NAME}2 = {GeneralLexicon.BATCH_ID_NAME}1 + {stride} < {GeneralLexicon.NUM_ELEMENTS}{i} ? {GeneralLexicon.BATCH_ID_NAME}1 + {stride} : {GeneralLexicon.BATCH_ID_NAME}1;')

          # Everything is in place now (offsets from ShrMemOpt, arena size from
          # the thread-block policy), so this is the point where the full check
          # is meaningful.  Previously each instruction was tested one at a time
          # and the first unprepared one aborted, hiding every other problem
          # behind it.
          self._verify_section(section.stream, i)

          for instruction in section.stream:
            instruction.gen_code(writer)

    self._kernel = writer.get_src()
    self.peak_pressure = self._context.peak_pressure
    self.resident_blocks = self._resident_blocks()

  def _resident_blocks(self) -> Optional[int]:
    """How many of these blocks fit on one SM, counting what is known exactly.

    Shared memory per block and threads per block are not estimates: the first
    is what `ShrMemOpt` allocated and the second is the launch geometry, and
    both budgets are in the hardware description.  So this half of occupancy
    can be computed rather than modelled -- unlike the register half, where
    the figure is bytes of live values and the hardware counts registers after
    allocation, a mapping that spreads over a factor of seventy across the
    corpus.

    Which makes the two worth keeping apart rather than adding up.  A caller
    comparing configurations can let this decide where it speaks and fall back
    on the model where it does not, instead of folding a fact and a guess into
    one number that is neither.

    The register limit is *not* applied here even though `max_reg_per_block`
    exists: applying it would need the register count, which is the thing that
    is not known.
    """
    if self._section is None or self._section.shr_mem_obj is None:
      return None
    hw = self._context.get_vm().get_hw_descr()
    shr = self._section.shr_mem_obj
    per_block = shr.get_total_size() * self._context.fp_type.size()
    threads = self._num_threads * shr.get_mults_per_block()
    limits = [hw.max_block_per_sm]
    if per_block:
      limits.append(hw.max_local_mem_size_per_block // per_block)
    if threads:
      limits.append(hw.max_threads_per_sm // threads)
    return max(0, min(limits))

  def _generate_launcher(self):
    writer = Writer()
    proto = self._generate_launcher_proto(with_defaults=False)
    mults_per_block = self._section.shr_mem_obj.get_mults_per_block()
    lexic = self._context.get_vm().get_lexic()
    with writer.Block(f'{proto}'):
      kernel_name = f'kernel_{self._base_kernel_name}'

      shmemsize = f'{self._section.shr_mem_obj.get_total_size()} * sizeof({self._context.fp_as_str()})'

      # TODO: allow multi-kernel approach instead
      coop = any(section.barrier for section in self._sections)

      writer(f'{lexic.kernel_range_object("block", f"{self._num_threads}, {mults_per_block}, 1")};')
      if self._clusterlaunchcontrol:
        # Stated, not checked.  The queue is the one traversal with a ceiling
        # of its own: the grid is sized by the batch rather than by occupancy,
        # `gridDim.x` stops at 2^31-1, and `query_cancel_get_first_ctaid_x`
        # answers in 32 bits.  So the batch cannot exceed that many blocks --
        # here, {ceiling} elements.
        #
        # No runtime guard, because the bound is not reachable: at this
        # kernel's footprint the device would need more memory than exists to
        # hold that batch, and a branch on every launch is not free when the
        # caller dispatches thousands of them per step.  The grid-stride loop
        # has no such ceiling; `batchId0` is 64-bit there and the grid is
        # occupancy-sized.
        writer(f'// launch_control: at most {2**31 - 1} blocks, i.e. '
               f'{(2**31 - 1) * mults_per_block} elements')
      if not self._persistent_threading:
        if coop:
          # Both remaining traversals launch one block per element rather than
          # one per worker, which is what a grid barrier cannot have: a
          # cooperative launch requires every block of the grid to be resident
          # at once, and this grid is sized by the batch.
          #
          # Under `launch_control` it is not a sizing question but a
          # structural one.  The queue *works* by the launcher never starting
          # most of the grid -- a resident block cancels the CTAs it then
          # processes itself -- so the blocks a grid barrier would wait for
          # are precisely the ones that will not exist.  Nor do the surviving
          # blocks agree on a count: each runs until its own cancel request
          # comes back empty.
          how = ('launch_control cancels most of the grid before it is '
                 'launched, so the blocks a grid barrier waits for never '
                 'run' if self._clusterlaunchcontrol else
                 'one block per element does not fit on the device at once')
          raise GenerationError(
              f'this kernel has a grid-wide barrier, which needs a '
              f'cooperative launch over resident workers; {how}. Use the '
              f'grid-stride traversal for a section that synchronises '
              f'across the grid.')
        num_blocks = f'({GeneralLexicon.NUM_ELEMENTS}0 + {mults_per_block} - 1) / {mults_per_block}'
      else:
        writer(f'{lexic.get_launch_size(kernel_name, "block", shmemsize)}')
        if coop:
          num_blocks = 'gridsize'
        else:
          num_blocks = f'std::min(gridsize, {GeneralLexicon.NUM_ELEMENTS}0)'
      writer(f'{lexic.kernel_range_object("grid", f"{num_blocks}, 1, 1")};')

      writer(lexic.set_shmem_size(kernel_name, shmemsize))

      lexic.get_stream_via_pointer(writer, 'stream', GeneralLexicon.STREAM_PTR_STR)

      for table in self._param_tables:
        writer(table.argument())

      args = self._generate_kernel_base_args()
      args = ', '.join(args)
      call_site = lexic.get_launch_code(func_name=kernel_name,
                                        grid='grid',
                                        block='block',
                                        stream='stream',
                                        func_params=args,
                                        shmem=shmemsize,
                                        coop=coop)
      writer(f'{call_site};')
      writer('CHECK_ERR;')
    self._launcher = writer.get_src()

  def _generate_header(self):
    self._header = f'{self._generate_launcher_proto(with_defaults=True)};\n'

  def _deduce_num_threads(self):
    """Adopt the section's lane geometry: the caller's, or the deduced one."""
    # Over the expansion: lane geometry follows from the operations, and a
    # descriptor that stands for several is not one of them.
    flat = [op for descr in self.descr_list for op in descr.operations()]
    config = self._lanes or lane_config.deduce(flat, self._context)
    self._num_threads = config.num_threads
    self._num_active_threads = config.num_active_threads
    self._lead_width = config.lead_width

  def _emit_global_ir(self):
    nonfirst_block = len(self._sections) > 0
    last_barrier = len(self._sections) > 0 and self._sections[-1].barrier

    shmbuilder = ShrMemAllocBuilder(self._context, self._scopes)

    self._scopes.add_scope()
    # allocate shared memory
    shmbuilder.build(size=None)
    self._section.shr_mem_obj = shmbuilder.get_resultant_obj()
    self._section.global_ir.extend(shmbuilder.get_instructions())

    # load globals to shared memory (if requested)
    if self._preload_globals:
      load_ir = []
      shmem_load = 0

      if nonfirst_block:
        load_ir.append(SyncBlock(self._context))

      self._scopes.add_scope()

      builder = GlobalLoaderBuilder(self._context, self._scopes, self._section.shr_mem_obj, self._num_threads)
      for symbol in self._scopes.get_global_scope().values():
        if symbol.obj.addressing == Addressing.NONE and symbol.stype != SymbolType.Data:
          shmem_load += builder.build(symbol)
          load_ir.extend(builder.get_instructions())

      vm = self._context.get_vm()
      shmem_cap = vm.get_hw_descr().max_local_mem_size_per_block

      if shmem_load < shmem_cap:
        self._section.global_ir += load_ir
        if last_barrier:
          self._section.global_ir.append(SyncGrid(self._context))
        else:
          self._section.global_ir.append(SyncBlock(self._context))
        return True
      else:
        # make sure to clean up all new symbols that didn't get added
        self._scopes.remove_scope()
        self._preload_globals = False

    builder = GetElementPtrBuilder(self._context, self._scopes)
    for symbol in self._scopes.get_global_scope().values():
      if symbol.obj.addressing == Addressing.SCALAR or (symbol.obj.addressing == Addressing.NONE and (symbol.stype == SymbolType.Data or not self._preload_globals)):
        builder.build(symbol)
        self._section.global_ir.extend(builder.get_instructions())

    # pipelines
    for symbol in self._scopes.get_global_scope().values():
      if symbol.obj.addressing in [Addressing.STRIDED, Addressing.PTR_BASED]:
        pass

    if not self._preload_globals:
      if last_barrier:
        self._section.global_ir.append(SyncGrid(self._context))
      elif nonfirst_block:
        self._section.global_ir.append(SyncBlock(self._context))

    return False

  def _emit_ir(self, descr_list):
    # find local data from batches
    builder = GetElementPtrBuilder(self._context, self._scopes)
    self._scopes.add_scope()
    for symbol in self._scopes.get_global_scope().values():
      if getattr(symbol.obj, 'is_variant', False):
        # Bound inside the loop, from the table, once per iteration.  A
        # binding here would name one member for the whole run.
        continue
      firstptr = symbol.obj.addressing == Addressing.SCALAR or symbol.obj.addressing == Addressing.NONE
      if not firstptr:
        builder.build(symbol)
        self._section.ir.extend(builder.get_instructions())

    self._scopes.add_scope()
    # Both of these belong to the section rather than to any one builder.
    #
    # The plan is the section's read/write geometry, computed once from the
    # descriptor list; the residency is where the section's values currently
    # are, which every operation reads and writes as it goes.
    #
    # The residency's lifetime is the section's, deliberately.  Carrying one
    # across a barrier would push its writebacks past the barrier that was
    # supposed to publish them.
    plan = SectionPlan(descr_list, self._scopes)
    residency = self._residency = Residency(self._context,
                          self._scopes.get_symbol(self._section.shr_mem_obj),
                          self._num_threads,
                          self._lead_width)
    temporaries = Temporaries(self._context, self._scopes, self._num_threads)
    # Every register image this section allocates is blocked the same way, so
    # the factory carries it rather than each caller passing it along.
    temporaries._lead_width = self._lead_width

    # One builder per kind of operation, all sharing the section's plan,
    # residency and temporaries.  The list is ordered: `GemmDescr` is a
    # `MultilinearDescr`, so the first match wins rather than the exact type.
    common = (self._context, self._scopes,
              self._scopes.get_symbol(self._section.shr_mem_obj),
              self._num_threads, plan, residency, temporaries,
              self._lead_width)
    builders = [
        (MultilinearDescr, MultilinearBuilder(*common)),
        (ElementwiseDescr, ElementwiseBuilder(*common)),
        (ReductionDescr, ReductionBuilder(*common)),
    ]


    # Expanded, like the section's plan above: a descriptor that stands for
    # several operations is built as those operations.  While that is all a
    # loop lowers to, a rolled list and the same list written out generate the
    # same body, which is the state the loop's own lowering has to be measured
    # against before it replaces this.
    #
    # Neighbours under one guard become one region rather than one region
    # each: the condition is then read once, and a body that is skipped is
    # skipped as a whole.
    guard = _GuardGrouping(self._context, self._scopes, residency,
                           self._section.ir)
    for outer in descr_list:
      if isinstance(outer, ForDescr) and self._emit_loops:
        guard.flush()
        self._emit_variant_loop(outer, builders)
        continue
      for descr in outer.operations():
        for kind, builder in builders:
          if isinstance(descr, kind):
            builder.build(descr)
            guard.add(descr, builder.get_instructions())
            break
        else:
          raise InternalError(f'{type(descr)} has no registered builder.')

    guard.flush()

    # Anything the section still holds only in registers has to reach memory
    # before the section ends.
    self._section.ir.extend(residency.flush_all())

  def _emit_variant_loop(self, loop, builders) -> None:
    """One body, one counter, and one binding per varying operand.

    The body is built with the same builders as anything else -- it is an
    ordinary descriptor list over the stand-ins -- and the only thing this adds
    is where a stand-in resolves: a table over the members, and a binding
    inside the loop that reads it at the counter.  Which is why the loop is
    assembled here and not inside a builder: no operation in the body knows it
    is in a loop, and none of them has to.
    """
    from tensorforge.backend.instructions.ptr_manip import (
        DeclareOperandTable, TableForm, VariantLoop)
    from tensorforge.backend.instructions.builders.ptr_manip_builder import \
        GetElementPtrBuilder

    # The first iteration is peeled, and it is not an optimisation.
    #
    # A body built cold does what the *first* of the expanded descriptors did:
    # it loads the destination and computes a result from it.  Repeating that
    # recomputes `Q + contribution` from the stored `Q` every time and keeps
    # only the last one.  What the loop wants is what descriptors two onwards
    # did -- accumulate into a destination that is already resident -- and the
    # way to build that body is to let one iteration establish the residency
    # first.
    #
    # It also settles the invariants without a pass: the shared staging of an
    # operand every iteration shares, and the destination's load, both happen
    # in the peeled copy and the residency stops the body from repeating them.
    # The cost is one body written out beside the loop, so four iterations
    # cost two copies rather than four.
    # ...unless something before the loop already did.  A run that follows a
    # descriptor writing the same destination -- which is what a frontend
    # produces when the first contribution assigns and the rest add -- has its
    # residency established already, and peeling again writes a second copy of
    # the body for nothing.
    accumulated_keys = [
        f'{GeneralLexicon.GLOBAL_MEM_PREFIX}{d.writes().tensor.name}'
        for d in loop.body(0)
        if getattr(d, 'add', False) and d.writes() is not None]
    resident = bool(accumulated_keys) and all(
        self._residency.get(key) is not None for key in accumulated_keys)

    if not resident:
      for descr in loop.body(0):
        for kind, builder in builders:
          if isinstance(descr, kind):
            builder.build(descr)
            self._section.ir.extend(builder.get_instructions())
            break

    body, variants = loop.decompose()
    counter = f'{GeneralLexicon.BATCH_ID_NAME}v{len(self._section.ir)}'

    tables, region = [], []
    pointers = GetElementPtrBuilder(self._context, self._scopes)
    for variant in variants:
      members = [self._scopes.get_symbol(view.tensor) for view in variant.members]
      stand_in = self._scopes.get_symbol(variant.stand_in.tensor)
      table = DeclareOperandTable(
          self._context, f'{stand_in.name}Table', members,
          stand_in.obj.addressing, stand_in.obj.datatype,
          form=(TableForm.SELECT
                if len(members) <= DeclareOperandTable.SELECT_LIMIT
                else TableForm.ARRAY),
          variant=counter)
      tables.append(table)
      pointers.build(stand_in, table=table, variant=counter)
      region.extend(pointers.get_instructions())

    # What the body threads through itself.  A destination is not accumulated
    # in place: each build takes a fresh register and reads the last one, so
    # the expanded form is a chain and one turn of it is what the body holds.
    #
    # Read off either side of the build, because that is the only moment both
    # links exist.  Keyed by the *binding*, which is what the residency is
    # keyed by -- the tensor is `m0`, its entry is `glb_m0`, and asking for the
    # tensor finds nothing and looks exactly like nothing to carry.
    accumulated = [descr.writes() for descr in body
                   if getattr(descr, 'add', False) and descr.writes() is not None]
    keys = [f'{GeneralLexicon.GLOBAL_MEM_PREFIX}{v.tensor.name}'
            for v in accumulated]
    before = {}
    for key in keys:
      entry = self._residency.get(key)
      if entry is not None:
        before[key] = entry.image

    for descr in body:
      for kind, builder in builders:
        if isinstance(descr, kind):
          builder.build(descr)
          region.extend(builder.get_instructions())
          break
      else:
        # A descriptor nobody recognises used to fall out of the loop and be
        # dropped, which turns a missing builder into a wrong kernel rather
        # than an error.
        raise InternalError(
            f'no builder for {descr.__class__.__name__}: {descr}')

    # An allocation is not a per-iteration act.  Left in the region it would
    # give the enclosing stream no definition for a register the body fills
    # and something after the loop reads -- the accumulator's writeback is
    # exactly that -- and in the emitted text it would put the declaration
    # inside the braces its users sit outside of.
    from tensorforge.backend.instructions.allocate import RegisterAlloc
    allocations = [i for i in region if isinstance(i, RegisterAlloc)]
    region = [i for i in region if not isinstance(i, RegisterAlloc)]
    carried = []
    for key in keys:
      entry = self._residency.get(key)
      was = before.get(key)
      if was is not None and entry is not None and entry.image is not was:
        carried.append((was, entry.image))

    # Close the chain: the body reads one register and writes another, and a
    # loop needs the two to be one.  Substituted on the built region rather
    # than arranged during the build, because what the residency hands out is
    # its business and the fact that a repeated body must land where it
    # started is not something it can know.
    for key, (init, result) in zip(
            [k for k in keys if k in before], carried):
      for instr in region:
        instr.substitute(result, init)
      # And tell the residency where the value now lives, because the
      # writeback is emitted after this returns and would otherwise store a
      # register the substitution has just made unreachable.
      entry = self._residency.get(key)
      self._residency.record_writeback(key, init, entry.home)
    carried = tuple((init, init) for init, _ in carried)

    # A destination that *varies* has to be stored inside the loop, and one
    # that does not must not be.
    #
    # The residency flushes once, at the end of the section, against whatever
    # address the entry holds by then.  For a destination that is the same
    # tensor every iteration -- an accumulator -- that is right and is the
    # whole point: the sum is written once.  For one that is a different
    # tensor every iteration the address is the last iteration's, so every
    # iteration computes and only the last is kept.
    #
    # Which it is, is whether the destination is a stand-in.  Not whether it
    # escapes: an accumulator escapes too, and asking that stores the sum on
    # every pass.
    for descr in body:
      dest = descr.writes()
      if dest is None or not getattr(dest.tensor, 'is_variant', False):
        continue
      key = f'{GeneralLexicon.GLOBAL_MEM_PREFIX}{dest.tensor.name}'
      if self._residency.get(key) is not None:
        region.extend(self._residency.flush(key))

    self._section.ir.extend(allocations)
    self._section.ir.append(
        VariantLoop(self._context, counter, loop.iterations, region, tables,
                    start=0 if resident else 1, carried=tuple(carried)))

  def _deduce_mults_per_block(self):
    policy = self._thread_block_policy_type(self._context,
                                            self._section.shr_mem_obj.get_global_size(),
                                            self._section.shr_mem_obj.get_size_per_mult(),
                                            self._num_threads,
                                            self._lead_width
                                            * self._context.get_user_options().lead_blocking)
    policy.set_has_barrier(
        any(instr.barrier_scope() is not None for instr in self._section.stream))
    num_mults_per_block = policy.get_num_mults_per_block()
    # A block holds whole groups or the group is not a unit.  Rounding down
    # rather than up, because up would exceed whatever bound produced the
    # number -- shared memory, threads, or the barrier cap itself.
    group = self._group_size(len(self._sections), self._section_traversal(
        len(self._sections))[0])
    if group > 1:
      num_mults_per_block = max(group, num_mults_per_block
                                - num_mults_per_block % group)
    self._section.shr_mem_obj.set_mults_per_block(num_mults_per_block)
    # The loop reads it to answer how far its body is uniform, and the answer
    # is what `verify` weighs a barrier against.  Over the optimised stream
    # rather than the loop this method's caller built: a pass may have
    # replaced it.
    for instr in self._section.stream:
      if isinstance(instr, BatchLoop):
        instr.set_mults_per_block(num_mults_per_block)

  def get_kernel(self):
    return self._kernel

  def get_launcher(self):
    return self._launcher

  def get_header(self):
    return self._header

  def _name_operands(self, gemm_list: List[OperationDescription]):
    tmp_counter = 0
    op_counter = 0

    pre_matrix_list = {}
    for gemm in gemm_list:
      local_list = gemm.matrix_list()

      # gather all matrices
      for matrix in local_list:
        # dict preserves ordering starting with 3.7
        pre_matrix_list[matrix.tensor] = None

      # A guard's operands are not operands of the operation -- no builder
      # resolves them, and `matrix_list` deliberately does not name them --
      # but the kernel still reads them, so they are parameters like any
      # other and need a name and a symbol.
      for view in gemm.condition_reads():
        pre_matrix_list[view.tensor] = None

    self._matrix_list = list(pre_matrix_list.keys())

    variant_counter = 0
    for matrix in self._matrix_list:
      if getattr(matrix, 'is_variant', False):
        # Its own series: a stand-in is not a parameter, so letting it take an
        # `m` number would move every later parameter along by one and change
        # the signature of a kernel that has nothing to do with it.
        matrix.name = f'v{variant_counter}'
        variant_counter += 1
      elif matrix.is_tmp:
        matrix.name = f't{tmp_counter}'
        tmp_counter += 1
      else:
        matrix.name = f'm{op_counter}'
        op_counter += 1

  def _collect_tmp_matrices(self):
    self._tmp_list = []
    for matrix in self._matrix_list:
      if matrix.is_tmp and matrix not in self._tmp_list:
        self._tmp_list.append(matrix)

  def _populate_global_scope(self):
    """
    Add non-tmp matrices to the global scope
    :return:
    """
    for matrix in self._matrix_list:
      if matrix not in self._tmp_list:
        if matrix.addressing == Addressing.SCALAR:
          # known scalars will always be inlined
          if matrix.has_values():
            stype = SymbolType.Data
          else:
            stype = SymbolType.Scalar
        else:
          stype = SymbolType.Batch
        # If the tensor was constructed without an explicit datatype,
        # inherit the context's. This makes Symbol.get_fptype()'s
        # resolution path well-defined for the common case where a
        # user-facing description (csa.py, four_matrices.py, …) only
        # specifies dtype on the Context.
        if getattr(matrix, 'datatype', None) is None:
          matrix.datatype = self._context.fp_type
        symbol = Symbol(obj=matrix,
                      name=matrix.name,
                      stype=stype)
        self._scopes.add_to_global(symbol)

  def unnamed_source(self) -> str:
    """Everything this kernel contributes to a file, with the name left out.

    The three surfaces in the order a translation unit sees them, plus the
    includes they need.  What identifies a kernel is what a compiler is given
    for it, so the question the name has to answer -- are these two kernels
    the same program -- is answered here by comparing exactly that, with the
    name itself replaced by a fixed token so that it does not answer its own
    question.

    Two kernels agreeing on this string differ in nothing a compiler reads,
    which is what makes one symbol for both correct rather than merely
    convenient: the routine cache emitting one and discarding the other loses
    nothing.
    """
    parts = [
        '\n'.join(self.get_helper_headers()),
        self._header or '',
        self._launcher or '',
        self._kernel or '',
    ]
    source = '\n'.join(parts)
    if self._base_kernel_name:
      source = source.replace(self._base_kernel_name,
                              Generator.NAME_PLACEHOLDER)
    return source

  def _resolve_identity(self, pinned: bool) -> None:
    """Name the kernel after its source, then put that name into the source.

    Where the caller pinned a name, keep it: `set_kernel_name` exists so that
    a frontend can address a kernel by a name of its own, and a caller doing
    that is answering for its uniqueness.  The registry holds it to that.
    """
    if not pinned:
      sha = hashlib.new('md5', usedforsecurity=False)
      sha.update(self.unnamed_source().encode())
      digest = sha.hexdigest()[:Generator.NAME_ENCODING_LENGTH]
      name = f'kernel_{digest}'

      for attr in ('_kernel', '_launcher', '_header'):
        text = getattr(self, attr)
        if text is not None:
          setattr(self, attr, text.replace(Generator.NAME_PLACEHOLDER, name))
      self._base_kernel_name = name

    if self._announce_identity:
      registry().register(self._base_kernel_name,
                          self.unnamed_source(),
                          self.descr_list)

  def get_base_name(self):
    return self._base_kernel_name

  def param_table_types(self) -> List[str]:
    """The by-value types the signature names, for whoever writes the file."""
    return [table.struct_definition() for table in self._param_tables]

  def _write_kernel_meta_data(self, writer):
    writer(f'// generated with TensorForge. Version: {interop.get_version()}')
    # What was asked for, so that a file found on its own says which of several
    # configurations of one workload it is.
    writer(f'// options: {self._context.get_user_options().describe()}')
    writer('// meta data:')
    glb_matrices = self._scopes.get_global_scope().values()
    for matrix in glb_matrices:
      writer(f'// {matrix.obj.gen_descr()}')

    writer.new_line()
    for item in self.descr_list:
      writer(f'// {item}')
    writer.new_line()

  def register_param_table(self, table) -> None:
    """Take a table into the kernel's signature in place of its members.

    Only the kernel's.  The launcher keeps taking the members one by one and
    assembles the value itself, so the interface a caller sees does not move --
    the substitution lives entirely between two pieces of generated code, which
    is the only reason it can be made at all without touching SeisSol.
    """
    self._param_tables.append(table)
    for member in table.get_operands():
      self._table_member[member.name] = table

  def _generate_base_params_list(self, symbol_list, with_types=True,
                                 with_defaults=False, substitute_tables=False):
    params = []
    emitted_tables = set()
    for symbol in symbol_list:
      if getattr(symbol.obj, 'is_variant', False):
        continue
      table = self._table_member.get(symbol.name) if substitute_tables else None
      if table is not None:
        # In place of the first member, once; the rest of them vanish.
        if id(table) not in emitted_tables:
          emitted_tables.add(id(table))
          params.append(table.parameter() if with_types else table.name)
        continue
      datatype = self._context.fp_type if symbol.obj.datatype is None else symbol.obj.datatype
      if symbol.obj.addressing == Addressing.SCALAR:
        if not symbol.stype == SymbolType.Data:
          params.extend([f'{datatype} {symbol.name}' if with_types else f'{symbol.name}'])
      else:
        ptr_type = symbol.obj.addressing.to_pointer()
        const_modifier = 'const ' if symbol.obj.direction == DataFlowDirection.SOURCE else ''
        batch_type = f'{const_modifier}{datatype}{ptr_type}' if with_types else ''
        # `size_t`, and the same `size_t` the element count uses.  This is an
        # *element* offset added to `batchId0 * stride`, so 32 bits caps what a
        # caller can express at 2^32-1 elements -- 17.2 GB into an f32 buffer,
        # 34.4 GB into an f64 one.  Both are reachable on a current card, and a
        # caller past them loses the high bits silently at the call site, which
        # is a wrong answer rather than a diagnostic.
        #
        # The arithmetic was never the problem: `batchId0 * stride` is already
        # 64-bit and the unsigned offset promoted into it.  What was capped is
        # what the *signature* can carry.
        offset_type = 'size_t' if with_types else ''
        params.extend([f'{batch_type} {symbol.name}'])
        if symbol.obj.addressing != Addressing.NONE:
          params.extend([f'{offset_type} {get_extra_offset_name(symbol)}'])

    batch_size_type = 'size_t' if with_types else ''

    for i, section in enumerate(self._sections):
      params.append(f'{batch_size_type} {GeneralLexicon.NUM_ELEMENTS}{i}')

    if self._flags is not FlagMode.ABSENT:
      flags_type = 'unsigned*' if with_types else ''
      # A mask the kernel dereferences unconditionally has no default: the
      # signature is where "you have to pass one" is stated.
      defaulted = with_defaults and self._flags is FlagMode.OPTIONAL
      default_flags_value = '= nullptr' if defaulted else ''

      for i, section in enumerate(self._sections):
        params.append(f'{flags_type} {GeneralLexicon.FLAGS_NAME}{i} {default_flags_value}')

    return params

  def _generate_kernel_base_args(self):
    global_symbols = self._scopes.get_global_scope().values()
    args = self._generate_base_params_list(global_symbols, with_types=False,
                                           substitute_tables=True)
    return args

  def _generate_kernel_proto(self, writer):
    global_symbols = self._scopes.get_global_scope().values()

    params = self._generate_base_params_list(symbol_list=global_symbols,
                                             with_types=True,
                                             substitute_tables=True)
    str_params = ', '.join(params)

    mults_per_block = min(section.shr_mem_obj.get_mults_per_block() for section in self._sections)
    shr_total_size = max(section.shr_mem_obj.get_total_size() for section in self._sections)

    total_num_threads_per_block = self._num_threads * mults_per_block

    lexic = self._context.get_vm().get_lexic()

    launch_bounds = (total_num_threads_per_block,)

    return lexic.kernel_definition(writer, launch_bounds, self._base_kernel_name, str_params, self._context.fp_as_str(),
                                         shr_total_size, global_symbols)

  def _generate_launcher_proto(self, with_defaults=True):
    global_symbols = self._scopes.get_global_scope().values()

    params = self._generate_base_params_list(symbol_list=global_symbols,
                                                  with_types=True,
                                                  with_defaults=with_defaults)

    default_value = ' = nullptr' if with_defaults else ''
    params.append(f'void* {GeneralLexicon.STREAM_PTR_STR}{default_value}')
    str_params = ', '.join(params)
    return f'void launcher_{self._base_kernel_name}({str_params})'

  def default_generate_call_site(self):
    if not self._is_registerd:
      raise RuntimeError('generator is not registered. Call register first.')
    symbols = deepcopy(list(self._scopes.get_global_scope().values()))
    for item in symbols:
      if item.obj.alias:
        item.name = item.obj.alias

    args = self._generate_base_params_list(symbol_list=symbols,
                                                with_types=False)

    if self._flags is not FlagMode.ABSENT:
      args.append(f'{GeneralLexicon.FLAGS_NAME}')
    args.append(f'{GeneralLexicon.STREAM_PTR_STR}')
    str_args = ', '.join(args)
    return f'launcher_{self._base_kernel_name}({str_args});'

  def get_helper_headers(self):
    headerset = set()
    for section in self._sections:
      for irinst in section.global_ir:
        for header in irinst.get_headers():
          headerset.add(header)
      for irinst in section.ir:
        for header in irinst.get_headers():
          headerset.add(header)
    # Sorted, not set order.  Iteration over a set of strings depends on
    # `PYTHONHASHSEED`, so the include list this produces varies between
    # processes -- which puts a run-to-run difference into the emitted file,
    # and into the name derived from it.
    return sorted(headerset)

  def generate_call_site(self,
                         mat_name_map,
                         offset_name_map):
    args = []

    # add tensors
    symbols = list(self._scopes.get_global_scope().values())
    for symbol in symbols:
      if symbol.obj.alias in mat_name_map:
        args.append(mat_name_map[symbol.obj.alias])
        if symbol.obj.addressing not in [Addressing.SCALAR, Addressing.NONE]:
          args.append(offset_name_map[symbol.obj.alias])

    flags = []
    regions = 0
    for desc in self.descr_list:
      if isinstance(desc, RegionDescription):
        regions += 1
        args.append(f'{desc.name}.numElements')
        flags.append(f'{desc.name}.flags')
    if regions == 0:
      args.append(f'numElements')
      flags.append(f'flags')

    # The element counts are always arguments; the masks beside them are only
    # arguments when the signature has parameters for them.
    if self._flags is not FlagMode.ABSENT:
      args += flags

    args.append('streamPtr')

    args = ', '.join(args)
    return f'launcher_{self._base_kernel_name}({args});'

  def _get_2d_block_id(self, block=None):
    lexic = self._context.get_vm().get_lexic()
    if block is None:
      block = lexic.block_idx_x
    return f'{lexic.thread_idx_y} + {lexic.block_dim_y} * ({block})'

  # NOTE: _get_element_size_guard and _get_flag_guard moved onto BatchLoop,
  # which is the only thing that needed them.
