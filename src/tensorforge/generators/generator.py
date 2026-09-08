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
from tensorforge.backend.instructions.sync_block import SyncThreads, SyncBlock, SyncGrid
from tensorforge.backend.instructions.batch_loop import BatchLoop, LoopMode
from tensorforge.backend.writer import Writer
from tensorforge.common.exceptions import GenerationError, InternalError

import tensorforge.interop as interop

class AbstractThreadBlockPolicy:
  def __init__(self, context: Context, global_mem: int, mem_per_mult: int, num_threads: int):
    self._context: Context = context
    self._mem_per_mult: int = mem_per_mult
    self._global_mem: int = global_mem
    self._num_threads: int = num_threads

    vm = self._context.get_vm()
    self._max_blocks = vm.get_hw_descr().max_block_per_sm
    self._max_allowed_mem = vm.get_hw_descr().max_local_mem_size_per_block
    self._max_threads = vm.get_hw_descr().max_threads_per_block

  def get_num_mults_per_block(self):
    pass


def _lead_blocking() -> int:
  from tensorforge.backend.instructions.memory import vectorize
  return vectorize.LEAD_BLOCKING


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
      return max_thread_mults
    else:
      max_mem_mults = (self._max_allowed_mem - self._global_mem * self._context.fp_type.size()) // (self._mem_per_mult * self._context.fp_type.size())
      return min(max_mem_mults, max_thread_mults)

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

class Generator:
  NAME_ENCODING_LENGTH = 10

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

    self._name_operands(self.descr_list)

    # launch control is (still) broken
    prefer_launchcontrol = False # context.get_vm().get_hw_descr().vendor == 'nvidia' and int(context.get_vm().get_hw_descr().model[3:]) >= 100
    prefer_persistent = context.get_vm().get_hw_descr().vendor in ['amd', 'nvidia'] and not prefer_launchcontrol
    prefer_preload = context.get_vm().get_hw_descr().vendor in ['amd'] and not prefer_launchcontrol

    self._persistent_threading = prefer_persistent
    self._preload_globals = prefer_preload

    self._clusterlaunchcontrol = prefer_launchcontrol

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
                       flags=self._flags)

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
      if self._persistent_threading or self._clusterlaunchcontrol:
        loop.append(SyncThreads(self._context, self._num_threads))

      self._deduce_mults_per_block()
      self._set_threadconfig()

      if lastbarrier:
        self._section.barrier = True

      while scopecnt < self._scopes.get_num_scopes():
        self._scopes.remove_scope()
      self._sections += [self._section]

    if not self._base_kernel_name:
      self._generate_kernel_name()

    self._generate_kernel()
    self._generate_launcher()
    self._generate_header()

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
          if self._context.get_vm().get_hw_descr().vendor == 'nvidia':
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
      if not self._persistent_threading:
        assert not coop
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
    residency = Residency(self._context,
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


    for descr in descr_list:
      if getattr(descr, 'guarded', lambda: False)():
        raise InternalError(
            f'{descr} runs under a guard, and nothing lowers one yet. What is '
            f'missing is the region: evaluate the conjunction, open a '
            f'`writer.if_` around the operation\'s body, and stop the plan '
            f'from counting a guarded write as covering its destination.')

    # Expanded, like the section's plan above: a descriptor that stands for
    # several operations is built as those operations.  While that is all a
    # loop lowers to, a rolled list and the same list written out generate the
    # same body, which is the state the loop's own lowering has to be measured
    # against before it replaces this.
    for outer in descr_list:
      if isinstance(outer, ForDescr) and self._emit_loops:
        self._emit_variant_loop(outer, builders)
        continue
      for descr in outer.operations():
        for kind, builder in builders:
          if isinstance(descr, kind):
            builder.build(descr)
            self._section.ir.extend(builder.get_instructions())
            break

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
    self._section.ir.extend(allocations)
    self._section.ir.append(
        VariantLoop(self._context, counter, loop.iterations, region, tables))

  def _deduce_mults_per_block(self):
    policy = self._thread_block_policy_type(self._context,
                                            self._section.shr_mem_obj.get_global_size(),
                                            self._section.shr_mem_obj.get_size_per_mult(),
                                            self._num_threads,
                                            self._lead_width
                                            * _lead_blocking())
    num_mults_per_block = policy.get_num_mults_per_block()
    self._section.shr_mem_obj.set_mults_per_block(num_mults_per_block)

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

  def _generate_kernel_name(self):
    global_symbols = self._scopes.get_global_scope().values()
    long_name = []
    for item in global_symbols:
      long_name.append(item.obj.gen_descr())

    for descr in self.descr_list:
      long_name.extend([
        str(descr)
      ])

    # needed for type differences (but same names)
    global_symbols = self._scopes.get_global_scope().values()
    params = self._generate_base_params_list(symbol_list=global_symbols, with_types=True)
    long_name.extend(params)

    descrs = '\n'.join(f'{descr}' for descr in self.descr_list)

    sha = hashlib.new('md5', usedforsecurity=False)
    sha.update(', '.join(long_name).encode())
    sha.update(descrs.encode())
    # `REQUIRED` and `OPTIONAL` have the same parameter list and different
    # bodies, so parameters alone do not identify the kernel: two kernels
    # differing only in the mask shape would collide on one name, and the
    # routine cache keeps whichever it saw first.  `OPTIONAL` stays out of the
    # hash so that names a caller without attributes gets do not depend on
    # this at all.
    if self._flags is not FlagMode.OPTIONAL:
      sha.update(self._flags.value.encode())
    md5encoding = sha.hexdigest()
    self._base_kernel_name = f'kernel_{md5encoding[:Generator.NAME_ENCODING_LENGTH]}'

  def get_base_name(self):
    return self._base_kernel_name

  def param_table_types(self) -> List[str]:
    """The by-value types the signature names, for whoever writes the file."""
    return [table.struct_definition() for table in self._param_tables]

  def _write_kernel_meta_data(self, writer):
    writer(f'// generated with TensorForge. Version: {interop.get_version()}')
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
        offset_type = 'unsigned' if with_types else ''
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
    return list(headerset)

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
