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
from tensorforge.generators.kernel_params import KernelParam
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
from contextlib import contextmanager

#: The kernel-side names for a multiplication that spans waves
#: (`Generator._lane_mapping`).  Declared inside each section's scope, so they
#: may repeat between sections; bound as the lexic's spelling of the lane and
#: the multiplication, so that everything which asks for either -- addressing,
#: guards, barriers, the batch traversal -- reads the derived value.
UNIT_NAME = 'tfUnit'
MULT_NAME = 'tfMult'
LANE_NAME = 'tfLane'
from tensorforge.common.threads import MultLayout, mults_per_group
from tensorforge.generators.identity import registry
from tensorforge.generators.launch import (LaunchConfig, SectionLaunch,
                                           launch_info_initializer,
                                           launch_types)

import tensorforge.interop as interop

class AbstractThreadBlockPolicy:
  def __init__(self, context: Context, global_mem: int, mem_per_mult: int, num_threads: int):
    self._context: Context = context
    self._mem_per_mult: int = mem_per_mult
    self._global_mem: int = global_mem
    self._num_threads: int = num_threads
    #: Whether the section this sizes a block for contains a barrier at all.
    self._has_barrier: bool = False
    #: The multiplications a block barrier has to meet, where the section
    #: states it rather than the lane layout implying it (`stage_members`).
    self._barrier_group = None

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
    if self._barrier_group:
      return self._barrier_group
    vm = self._context.get_vm()
    wave = vm.get_hw_descr().vec_unit_length
    if self._num_threads == wave:
      return None
    if vm.get_lexic().has_sync_mult(self._num_threads, vm.get_hw_descr()):
      return None
    return mults_per_group(self._num_threads, wave)

  def set_has_barrier(self, has_barrier: bool) -> None:
    self._has_barrier = bool(has_barrier)

  def set_barrier_group(self, group) -> None:
    self._barrier_group = group


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
    #:
    #: Not on AMD, where the smaller block is measured to lose: `local_flux`
    #: at lead width two on gfx1150 took 242 ns an element at 128 threads
    #: (four mults) against 153 at 256 (eight), and 256 was the fastest
    #: arrangement of the kernel at either width.  The same direction as
    #: the halved NVIDIA block in `get_num_mults_per_block`, which lost there
    #: too.
    vendor = context.get_vm().get_hw_descr().vendor
    self._lane_factor = 1 if vendor == 'amd' else max(1, lead_width)

  def get_num_mults_per_block(self):
    # 128 threads on NVIDIA, four warps: one per scheduler of an SM (four
    # since Volta), so that an SM holds several independent blocks rather
    # than one large one.  Through the generator on sm_120: `local_flux`
    # -6.0 %, `chain_three`, `square_notrans` and `wide_cascade` within the
    # noise (+0.1 to +1.5 %).  Not on AMD: on gfx1150 the halved block made
    # `chain_three` 3.6 % and `wide_cascade` 4.1 % slower, so it stays at
    # 256 there and on the other vendors until something says otherwise.
    #
    # And not where the block preloads operators into shared memory
    # (`global_mem`): its multiplications share that one copy, and halving
    # the block doubles the copies and halves the blocks that fit.  A
    # multiplication wider than 128 threads keeps the old bound.
    lanes = self._num_threads * self._lane_factor
    vendor = self._context.get_vm().get_hw_descr().vendor
    threads = 128 if vendor == 'nvidia' and self._global_mem == 0 else 256
    max_thread_mults = threads // lanes or 256 // lanes
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
    #: The operators preloaded into shared memory, with their waits and the
    #: barrier after them: a run of `global_ir` that is emitted as one body.
    self.preload: List[AbstractInstruction] = []
    #: Where the preloaded images start in the prologue's arena, so that they
    #: can be laid out again once their operands' storage is settled.
    self.preload_mark: int = 0
    #: The block-wide copies of merged runs' members (`stage_members`), which
    #: take their buffers from the same arena, after the images.
    self.stage_loaders: List[AbstractInstruction] = []

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
    #: The list as the caller handed it, before merging rewrites it -- what
    #: `Options.autotune` builds its candidates from and rebuilds the pick on.
    self._given: List[OperationDescription] = gemm_list
    #: The configuration `Options.autotune` chose, or None where it did not run.
    self.tuned = None
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
    #: Arithmetic operations this build wrote out (`Context.record_work`).
    self.emitted_work: Optional[int] = None
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
    # `preload_partial`: the operands that fit rather than all or none.  How
    # many of those taken a retry has dropped again, which were staged, and
    # which were candidates and are read from global memory.
    self._preload_partial = (prefer_preload
                             and context.get_user_options().preload_partial)
    self._preload_drop = 0
    self._preloaded = set()
    self._preload_left = set()
    #: The launch every section runs under, once the sections are built.
    self._launch = None

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
    # Into the regions as well: the default is a no-op, and only a blockwide
    # GlbToShrLoader overrides it -- the prologue's, and a staged member's
    # inside a merged run (`_stage_member`).
    mults = self._section.shr_mem_obj.get_mults_per_block()

    def walk(instrs):
      for instr in instrs:
        instr.set_threadconfig_pre(self._num_threads, mults)
        for region in instr.regions():
          walk(region)
    walk(self._section.stream)

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
    self._context.emitted_work = None

    self._autotune()

    self.register()

    self._deduce_num_threads()

    with self._lane_mapping():
      return self._generate_bound()

  def _autotune(self) -> None:
    """Rebuild this generator at the configuration `Options.autotune` picks.

    Only where nobody fixed the geometry: an explicit `lanes` is a caller's
    decision, and it is also what every candidate build carries, so a
    candidate never tunes itself.  The candidates are built from deep copies,
    because preparing and rolling leave their marks on the tensors; the pick
    is then built here, on the caller's own, which is where the host reads the
    storage from.
    """
    opts = self._context.get_user_options()
    if opts.autotune in ('', 'off') or self._lanes is not None:
      return
    if opts.lanes_per_mult:
      return
    import copy
    from tensorforge.generators import tuning
    given = self._given
    pick = tuning.autotune(lambda: copy.deepcopy(given), self._context,
                           mode=opts.autotune, budget=opts.autotune_budget,
                           cache=opts.autotune_cache or None)
    if pick is None:
      return
    announce = self._announce_identity
    rotate = self._rotate
    self.__init__(given, pick.context(self._context),
                  self._thread_block_policy_type, lanes=pick.lanes,
                  attrs=self._attrs)
    self._announce_identity = announce
    self._rotate = rotate
    self.tuned = pick
    self._context.peak_pressure = None
    self._context.emitted_work = None

  def _generate_bound(self):
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
      self._preload_drop = 0
      # Two attempts, with the operators preloaded and without -- or, under
      # `preload_partial`, one more per operator dropped.  Each failed attempt
      # either drops one or gives up preloading, and an attempt without it
      # ends the loop, so it ends.
      while True:
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
                         group_size=self._group_size(index, start),
                         narrow_group=self._num_threads
                         < self._context.get_vm().get_hw_descr().vec_unit_length)

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

        settled = self._settle_storage()
        fits = self._deduce_mults_per_block() and settled
        if fits or not self._section.preload:
          self._set_threadconfig()
          break
        # The preloaded operators left no room for one multiplication: the
        # check that admitted them compares against the block's limit before
        # anything per multiplication is known.  So the section is built again
        # without them, which is what the check would have decided had it
        # known -- and the operators are read from global memory, as they are
        # wherever they do not fit.
        while scopecnt < self._scopes.get_num_scopes():
          self._scopes.remove_scope()
        if self._preload_partial and len(self._preloaded) > 1:
          # One operator fewer rather than none.
          self._preload_drop += 1
        else:
          self._preload_globals = False

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

    self._launch = self._make_launch_config()
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
    if self._section.stage_loaders:
      # A staged member is shared by every multiplication of the block and
      # fenced by block barriers, so the block is one group on the same trips.
      if start != self._get_2d_block_id():
        return 1
      return self._stage_group()
    split = MultLayout(self._num_threads, wave)
    if not (split.contiguous or split.whole_waves):
      # The lanes of one multiplication sit in several waves, so its barrier
      # reaches the whole group (`Lexic.sync_mult`) and every multiplication in
      # the group has to take the same trips through the loop.
      if start != self._get_2d_block_id():
        return 1
      return mults_per_group(self._num_threads, wave)
    if self._num_threads < wave and self._needs_wave_group():
      # The multiplications of one wave, driven together, because something
      # in the body is issued by the whole wave at once.  A rotated start has
      # the same trouble as below, and leaves the verifier to refuse.
      if start != self._get_2d_block_id():
        return 1
      return wave // self._num_threads
    if self._num_threads <= wave:
      return 1
    if start != self._get_2d_block_id():
      return 1
    return mults_per_group(self._num_threads, wave)

  def _needs_wave_group(self) -> bool:
    """Whether the section holds an instruction the whole wave issues together.

    A matrix fragment product is one (`convergence_scope`).  Where a
    multiplication is narrower than the wave, its neighbours in the wave have
    to take the same trips through the batch loop, so the loop is driven a
    wave of rows at a time.
    """
    def walk(instrs):
      return any(instr.convergence_scope() is not None
                 or any(walk(region) for region in instr.regions())
                 for instr in instrs)
    return walk(self._section.stream or self._section.ir)

  def _settle_storage(self) -> bool:
    """Settle how every operand is stored, and size the preloaded copies by it.

    An operand's storage order is offered where the matrix path is emitted,
    and the section prologue's copies of it were sized when they were built,
    before any body existed.  An order with padding slots -- a fragment image
    is tiled, 3584 slots for a 56x56 -- then outgrew its copy.  So the orders
    are settled here, on the final stream, and the images laid out again from
    the start of the prologue's arena in the order they were built.

    Whether the images still fit the block, against the same limit the
    prologue checked them against when they were smaller.
    """
    def walk(instrs):
      for instr in instrs:
        if hasattr(instr, 'settle_storage'):
          instr.settle_storage()
        for region in instr.regions():
          walk(region)
    walk(self._section.stream)
    images = [instr for instr in self._section.preload
              if getattr(instr, '_verbatim', False)]
    if not images:
      return True
    obj = self._section.shr_mem_obj
    obj.release_global(self._section.preload_mark)
    for image in images:
      image._get_bounding_box_dense()
      image.set_shr_mem_offset(obj.alloc_global(image.compute_shared_mem_size()),
                               True, True)
    # The staged members' buffers came after the images, and go after them
    # again rather than under an image that grew.
    for loader in self._section.stage_loaders:
      loader._get_bounding_box_dense()
      loader.set_shr_mem_offset(
          obj.alloc_global(loader.compute_shared_mem_size()), True, True)
    cap = self._context.get_vm().get_hw_descr().max_local_mem_size_per_block
    return obj.get_global_size() * self._context.fp_type.size() < cap

  @staticmethod
  def _set_mult_stride(section) -> None:
    """Tell every instruction how much shared memory one multiplication owns.

    Known only now: `ShrMemOpt` has sized the arena.  A matrix path whose warp
    holds several multiplications reads its neighbours' tiles at that
    distance (`nvidia._warp_group`).
    """
    obj = section.shr_mem_obj
    if obj is None or obj.get_size_per_mult() is None:
      return
    stride = obj.get_size_per_mult()

    def walk(instrs):
      for instr in instrs:
        if hasattr(instr, 'set_mult_stride'):
          instr.set_mult_stride(stride)
        for region in instr.regions():
          walk(region)
    walk(section.stream)

  def _batch_loop_mode(self) -> LoopMode:
    if self._persistent_threading:
      return LoopMode.PERSISTENT
    if self._clusterlaunchcontrol:
      return LoopMode.LAUNCHCTRL
    return LoopMode.SINGLE

  def _declare_lane_mapping(self, writer) -> None:
    """Emit the three values `_lane_mapping` binds the spellings to.

    Inside the section's scope, so the names may repeat from section to
    section; what must not repeat is the arithmetic, which is the layout's
    (`MultLayout`).
    """
    lexic = self._context.get_vm().get_lexic()
    wave = self._context.get_vm().get_hw_descr().vec_unit_length
    layout = MultLayout(self._num_threads, wave)
    if layout.contiguous or layout.whole_waves:
      return
    tid_x = getattr(lexic, 'raw_thread_idx_x', lexic.thread_idx_x)
    tid_y = getattr(lexic, 'raw_thread_idx_y', lexic.thread_idx_y)
    upg, mpg, unit = layout.units_per_group, layout.mults_per_group, layout.unit
    writer(f'const auto {UNIT_NAME} = {tid_y};')
    group = f'({UNIT_NAME} / {upg})' if upg > 1 else '0'
    within = f'({UNIT_NAME} % {upg})' if upg > 1 else UNIT_NAME
    writer(f'const auto {MULT_NAME} = {group} * {mpg} + {within} % {mpg};'
           if mpg > 1 else f'const auto {MULT_NAME} = {group};')
    writer(f'const auto {LANE_NAME} = {within} / {mpg} * {unit} + {tid_x};')

  @contextmanager
  def _lane_mapping(self):
    """Bind the lane and multiplication spellings for the whole build.

    Where a multiplication is a run of lanes inside one wave, `threadIdx.x`
    *is* the lane and `threadIdx.y` the multiplication, and this binds nothing.

    Where it is not -- 48 lanes over a 32-wide wave -- the two cannot both be
    an axis of the launch geometry.  Laying the multiplication along `x` would
    put 32 of its lanes in one wave and 16 in the next, so the three waves of
    a group would hold three different shapes (32-0, 16-16, 0-32) and the body
    would have to branch on which one it is in.  So the launch is in units of
    `gcd` lanes instead (`MultLayout`), dealt out to the multiplications of a
    group in turn, and the lane and the multiplication are *derived* from the
    unit: every wave then holds the same shape.

    Named rather than substituted at each use: the two indices are read by the
    addressing, the guards, the barriers and the batch loop, and every one of
    them asks the lexic for the spelling.  Rebinding the spelling for the
    length of the section reaches all of them at once, and `block_dim_y` goes
    with them -- the batch stride is multiplications per block, which is no
    longer the `y` extent.
    """
    lexic = self._context.get_vm().get_lexic()
    wave = self._context.get_vm().get_hw_descr().vec_unit_length
    layout = MultLayout(self._num_threads, wave)
    # The hardware spellings, kept aside for the few places that mean the
    # thread and not the lane: a block-wide loader numbering its threads
    # (`AbstractShrMemLoader._linear_idx`) asks for a unique index over the
    # block, which the derived lane is not.
    lexic.raw_thread_idx_x = getattr(lexic, 'thread_idx_x', None)
    lexic.raw_thread_idx_y = getattr(lexic, 'thread_idx_y', None)
    # Not every lexic spells a block extent along `x`: SYCL names the axes
    # differently, and the mapping below is not reached on those targets.
    lexic.raw_block_dim_x = getattr(lexic, 'block_dim_x', None)
    if layout.contiguous or layout.whole_waves:
        # A run of lanes inside one wave, or a whole number of waves: either
        # way every wave holds one shape already, and `threadIdx.x` is the
        # lane it always was.
        yield layout
        return
    saved = (lexic.thread_idx_x, lexic.thread_idx_y, lexic.block_dim_y)
    lexic.thread_idx_x, lexic.thread_idx_y = LANE_NAME, MULT_NAME
    # Multiplications per block, which is what the batch traversal steps by.
    # As an expression over the `y` extent rather than the number itself:
    # `mults_per_block` is decided by `ShrMemOpt`, long after the loop that
    # reads this was built, and a name declared in the kernel would not reach
    # the launcher.
    lexic.block_dim_y = (f'({lexic.block_dim_y} / {layout.units_per_mult})'
                         if layout.units_per_mult > 1 else lexic.block_dim_y)
    try:
        yield layout
    finally:
        lexic.thread_idx_x, lexic.thread_idx_y, lexic.block_dim_y = saved

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
          self._declare_lane_mapping(writer)
          start, stride = self._section_traversal(i)

          writer(f'const auto {GeneralLexicon.BATCH_ID_NAME}_start = {start};')
          writer(f'const auto {GeneralLexicon.BATCH_ID_NAME}1 = {GeneralLexicon.BATCH_ID_NAME}_start < {GeneralLexicon.NUM_ELEMENTS}{i} ? {GeneralLexicon.BATCH_ID_NAME}_start : 0;')
          writer(f'const auto {GeneralLexicon.BATCH_ID_NAME}2 = {GeneralLexicon.BATCH_ID_NAME}1 + {stride} < {GeneralLexicon.NUM_ELEMENTS}{i} ? {GeneralLexicon.BATCH_ID_NAME}1 + {stride} : {GeneralLexicon.BATCH_ID_NAME}1;')

          # Everything is in place now (offsets from ShrMemOpt, arena size from
          # the thread-block policy), so this is the point where the full check
          # is meaningful.  Previously each instruction was tested one at a time
          # and the first unprepared one aborted, hiding every other problem
          # behind it.
          self._set_mult_stride(section)
          self._verify_section(section.stream, i)

          preload = {id(instr) for instr in section.preload}
          for instruction in section.stream:
            if id(instruction) not in preload:
              instruction.gen_code(writer)
            elif instruction is section.preload[0]:
              with AbstractInstruction.shared_body(self._context, writer):
                for member in section.preload:
                  member.gen_code(writer)

    self._kernel = writer.get_src()
    self.peak_pressure = self._context.peak_pressure
    self.emitted_work = self._context.emitted_work
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
    if self._launch is None:
      return None
    hw = self._context.get_vm().get_hw_descr()
    per_block = self._launch.shared_bytes
    threads = self._num_threads * self._launch.mults_per_block
    limits = [hw.max_block_per_sm]
    if per_block:
      limits.append(hw.max_local_mem_size_per_block // per_block)
    if threads:
      limits.append(hw.max_threads_per_sm // threads)
    return max(0, min(limits))

  def _make_launch_config(self) -> LaunchConfig:
    """The launch every section runs under (`generators.launch`).

    One launch serves all sections, so they have to agree on how many
    multiplications a block holds: a section planned for fewer would find
    per-multiplication windows and groups it never laid out.  None of the
    corpus's multi-section kernels disagree; the launcher took the last
    section's figure and the launch bounds the smallest, so a disagreement
    would have been a wrong launch rather than an error.  The shared memory
    is the largest any section needs.
    """
    sections = tuple(
        SectionLaunch(section.shr_mem_obj.get_mults_per_block(),
                      section.shr_mem_obj.get_total_size(),
                      bool(section.barrier))
        for section in self._sections)
    mults = {s.mults_per_block for s in sections}
    if len(mults) != 1:
      raise InternalError(
          f'the sections of one kernel planned {sorted(mults)} '
          f'multiplications per block, and one launch serves them all')
    mults_per_block = mults.pop()
    shared = max(s.shared_elements for s in sections)
    lexic = self._context.get_vm().get_lexic()
    wave = self._context.get_vm().get_hw_descr().vec_unit_length
    layout = MultLayout(self._num_threads, wave)
    plain = layout.contiguous or layout.whole_waves
    block_x = self._num_threads if plain else layout.unit
    block_y = (mults_per_block if plain
               else mults_per_block * layout.units_per_mult)
    if getattr(lexic, 'simd_mode', False):
      # One work-item per multiplication: the lanes are the vector, and the
      # block counts multiplications (`EsimdLexic`).
      block_x, block_y = 1, mults_per_block
    return LaunchConfig(
        threads_per_mult=self._num_threads,
        active_threads=self._num_active_threads,
        lead_width=self._lead_width,
        mults_per_block=mults_per_block,
        block=(block_x, block_y, 1),
        shared_elements=shared,
        shared_bytes=shared * self._context.fp_type.size(),
        cooperative=any(s.barrier for s in sections),
        persistent=bool(self._persistent_threading),
        sections=sections)

  def launch_config(self) -> LaunchConfig:
    """The launch this kernel runs under; `None` before `generate`."""
    return self._launch

  def _launch_config_proto(self, with_defaults=True):
    """`launch_config_<kernel>(numElements..., streamPtr)`, the function that
    decides the launch at run time -- declared in the header, defined next to
    the launcher, which calls it."""
    params = [KernelParam.size(f'{GeneralLexicon.NUM_ELEMENTS}{i}')
              for i in range(len(self._sections))]
    params.append(KernelParam.opaque(
        'void*', GeneralLexicon.STREAM_PTR_STR,
        ' = nullptr' if with_defaults else ''))
    str_params = ', '.join(self._declare(params, with_defaults=with_defaults,
                                         host=True))
    return (f'tensorforge::LaunchConfig '
            f'launch_config_{self._base_kernel_name}({str_params})')

  def _generate_launch_config(self, writer, lexic, kernel_name, shmemsize):
    """The body of `launch_config_<kernel>`: the block and the shared memory
    as generated, the grid as the device and the element count allow."""
    launch = self._launch
    mults_per_block = launch.mults_per_block
    coop = launch.cooperative
    block_x, block_y, block_z = launch.block
    with writer.Block(self._launch_config_proto(with_defaults=False)):
      for i in range(len(self._sections)):
        writer(f'(void){GeneralLexicon.NUM_ELEMENTS}{i};')
      writer(f'(void){GeneralLexicon.STREAM_PTR_STR};')
      writer(f'{lexic.kernel_range_object("block", f"{block_x}, {block_y}, {block_z}")};')
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
        writer(f'{lexic.get_launch_size(kernel_name, "block", shmemsize, resident=coop)}')
        if coop:
          num_blocks = 'gridsize'
        else:
          num_blocks = f'std::min(gridsize, {GeneralLexicon.NUM_ELEMENTS}0)'
      writer('tensorforge::LaunchConfig config{};')
      writer(f'config.grid[0] = {num_blocks};')
      writer('config.grid[1] = 1;')
      writer('config.grid[2] = 1;')
      writer(f'config.block[0] = {block_x};')
      writer(f'config.block[1] = {block_y};')
      writer(f'config.block[2] = {block_z};')
      writer(f'config.sharedMemBytes = {shmemsize};')
      writer(f'config.cooperative = {"true" if coop else "false"};')
      writer('return config;')

  def _generate_launcher(self):
    """`launch_config_<kernel>`, then the launcher, which launches what it
    decides -- so the launch host code can ask about and the launch that
    happens are one computation (`generators.launch`)."""
    writer = Writer()
    lexic = self._context.get_vm().get_lexic()
    kernel_name = f'kernel_{self._base_kernel_name}'
    shmemsize = (f'{self._launch.shared_elements} * '
                 f'sizeof({self._context.fp_as_str()})')
    coop = self._launch.cooperative
    # Guarded, and here as well as in the header: a translation unit holding
    # the launcher need not include the header.
    writer(launch_types())
    self._generate_launch_config(writer, lexic, kernel_name, shmemsize)
    with writer.Block(self._generate_launcher_proto(with_defaults=False)):
      counts = [f'{GeneralLexicon.NUM_ELEMENTS}{i}'
                for i in range(len(self._sections))]
      writer(f'const tensorforge::LaunchConfig config = '
             f'launch_config_{self._base_kernel_name}'
             f'({", ".join(counts + [GeneralLexicon.STREAM_PTR_STR])});')
      writer(f'{lexic.kernel_range_object("block", "config.block[0], config.block[1], config.block[2]")};')
      writer(f'{lexic.kernel_range_object("grid", "config.grid[0], config.grid[1], config.grid[2]")};')

      writer(lexic.set_shmem_size(kernel_name, 'config.sharedMemBytes'))

      lexic.get_stream_via_pointer(writer, 'stream', GeneralLexicon.STREAM_PTR_STR)

      for table in self._param_tables:
        writer(table.argument())

      args = self._generate_kernel_base_args(writer)
      args = ', '.join(args)
      call_site = lexic.get_launch_code(func_name=kernel_name,
                                        grid='grid',
                                        block='block',
                                        stream='stream',
                                        func_params=args,
                                        shmem='config.sharedMemBytes',
                                        coop=coop)
      writer(f'{call_site};')
      writer('CHECK_ERR;')
    self._launcher = writer.get_src()

  def _generate_header(self):
    """The launcher's prototype, and what host code may ask about the launch
    without calling it: `launch_info_<kernel>` for what generation fixed, and
    `launch_config_<kernel>` for the grid as well (`generators.launch`)."""
    self._header = (
        f'{launch_types()}'
        f'inline constexpr tensorforge::LaunchInfo '
        f'launch_info_{self._base_kernel_name} = '
        f'{launch_info_initializer(self._launch)};\n'
        f'{self._launch_config_proto(with_defaults=True)};\n'
        f'{self._generate_launcher_proto(with_defaults=True)};\n')

  def _deduce_num_threads(self):
    """Adopt the section's lane geometry: the caller's, or the deduced one."""
    # Over the expansion: lane geometry follows from the operations, and a
    # descriptor that stands for several is not one of them.
    flat = [op for descr in self.descr_list for op in descr.operations()]
    config = (self._lanes or lane_config.requested(flat, self._context)
              or lane_config.deduce(flat, self._context))
    self._num_threads = config.num_threads
    self._num_active_threads = config.num_active_threads
    self._lead_width = config.lead_width

  def _preload_selection(self, candidates, cap):
    """The operands `preload_partial` stages: first fit, in the order the
    kernel declares them, under the block's shared memory in bytes -- less
    the last `_preload_drop` of those, which a retry has dropped again."""
    size = self._context.fp_type.size()
    chosen, used = [], 0
    for symbol in candidates:
      need = int(symbol.obj.storage_volume()) * size
      if used + need < cap:
        chosen.append(symbol)
        used += need
    return chosen[:max(0, len(chosen) - self._preload_drop)]

  def _emit_global_ir(self):
    nonfirst_block = len(self._sections) > 0
    last_barrier = len(self._sections) > 0 and self._sections[-1].barrier
    self._preloaded = set()
    self._preload_left = set()

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

      mark = self._section.shr_mem_obj.get_global_size()
      self._section.preload_mark = mark
      builder = GlobalLoaderBuilder(self._context, self._scopes, self._section.shr_mem_obj, self._num_threads)
      # A stand-in of a merged run is not an argument: which member it is
      # changes per iteration, through a table over the members.  Preloading
      # the stand-in took the address of a name no kernel declares; preloading
      # its members would make that table select between shared copies while
      # its binding claims global memory -- a wrong address space that only a
      # target spelling the space in the type (AMD) notices.  So a merged run
      # reads its members from global, as it does where nothing is preloaded.
      scope = self._scopes.get_global_scope()
      merged = {id(member) for symbol in scope.values()
                if getattr(symbol.obj, 'is_variant', False)
                for member in getattr(symbol.obj, 'variant_members', ())}
      vm = self._context.get_vm()
      shmem_cap = vm.get_hw_descr().max_local_mem_size_per_block
      candidates = [symbol for symbol in scope.values()
                    if not getattr(symbol.obj, 'is_variant', False)
                    and id(symbol.obj) not in merged
                    and symbol.obj.addressing == Addressing.NONE
                    and symbol.stype != SymbolType.Data]
      chosen = (self._preload_selection(candidates, shmem_cap)
                if self._preload_partial else candidates)
      for symbol in chosen:
        shmem_load += builder.build(symbol)
        load_ir.extend(builder.get_instructions())
        self._preloaded.add(id(symbol.obj))
      self._preload_left = ({id(s.obj) for s in candidates}
                            - {id(s.obj) for s in chosen})

      # Bytes against bytes: `shmem_load` counts elements, the cap is the
      # hardware's figure in bytes.  Compared as they were, 57600 floats of
      # preloaded operators (local_flux at b = 120, 225 KB) passed a 64 KB
      # cap on gfx942 -- and FP64 at b = 56 (98 KB) with it -- and the launch
      # asked for more LDS than the device has.
      if chosen and shmem_load * self._context.fp_type.size() < shmem_cap:
        # Waited for before the barrier that publishes them.  A barrier orders
        # the threads, not the copies they issued: without the waits the
        # block went past `__syncthreads()` with the transfers still in flight
        # wherever they were asynchronous -- on NVIDIA, every one of them.
        from tensorforge.backend.instructions.memory.load import (
            GlbToShrLoader, LoadWait)
        load_ir += [LoadWait(instr) for instr in load_ir
                    if isinstance(instr, GlbToShrLoader)]
        if last_barrier:
          load_ir.append(SyncGrid(self._context))
        else:
          load_ir.append(SyncBlock(self._context))
        self._section.global_ir += load_ir
        # The operands `preload_partial` left out are read from global memory,
        # bound as they are where nothing is staged -- here, since this branch
        # returns before that binding below.
        ptrs = GetElementPtrBuilder(self._context, self._scopes)
        for symbol in candidates:
          if id(symbol.obj) in self._preload_left:
            ptrs.build(symbol)
            self._section.global_ir.extend(ptrs.get_instructions())
        # One body for the lot, so that a transfer, the pointer it reads and
        # the wait that retires it are values of the same body -- the
        # condition for the structured `copy.async`.  Each in a body of its
        # own, the transfer found no source value and fell back to driving a
        # `cuda::pipeline` object as text, one no kernel declares: nvcc
        # refused every NVIDIA kernel with `preload_globals`.
        self._section.preload = load_ir
        return True
      else:
        # make sure to clean up all new symbols that didn't get added -- and
        # the shared memory their loaders reserved while being built, which
        # otherwise stays in the launch's request (225 KB at b = 120 on a
        # 64 KB gfx942, with nothing preloaded)
        self._scopes.remove_scope()
        self._section.shr_mem_obj.release_global(mark)
        self._preload_globals = False
        self._preloaded = set()
        self._preload_left = set()

    builder = GetElementPtrBuilder(self._context, self._scopes)
    for symbol in self._scopes.get_global_scope().values():
      if getattr(symbol.obj, 'is_variant', False):
        # A stand-in is not a parameter: it is bound inside the merged loop,
        # from the table, and `_emit_ir` skips it for the same reason.  Bound
        # here as well it came out as `glb_v0 = &v0[0]` over a name nothing
        # declares -- and `get_symbol` then found that binding for the table,
        # so the loop's own one was named `glb_glb_v0`.
        continue
      if symbol.obj.addressing == Addressing.SCALAR or (symbol.obj.addressing == Addressing.NONE and (symbol.stype == SymbolType.Data or not self._preload_globals or id(symbol.obj) in self._preload_left)):
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
    written = {id(d.writes().tensor) for d in body if d.writes() is not None}
    counter = f'{GeneralLexicon.BATCH_ID_NAME}v{len(self._section.ir)}'

    tables, region = [], []
    pointers = GetElementPtrBuilder(self._context, self._scopes)
    for variant in variants:
      members = [self._scopes.get_symbol(view.tensor) for view in variant.members]
      stand_in = self._scopes.get_symbol(variant.stand_in.tensor)
      # The members are the bindings, not the parameters: `glb_m5` is
      # already `&m5[batchId][offset]`, one element's data, whatever the
      # parameter's addressing was -- and the binding after the table reads
      # it as such.  So the table holds plain data pointers, which is what a
      # batch-invariant operand's are.  Typed by the stand-in's addressing it
      # declared `const float **` over `const float *` members.
      resolved = all(m.name.startswith(GeneralLexicon.GLOBAL_MEM_PREFIX)
                     for m in members)
      table = DeclareOperandTable(
          self._context, f'{stand_in.name}Table', members,
          Addressing.NONE if resolved else stand_in.obj.addressing,
          stand_in.obj.datatype,
          form=(TableForm.SELECT
                if len(members) <= DeclareOperandTable.SELECT_LIMIT
                else TableForm.ARRAY),
          variant=counter)
      tables.append(table)
      if self._stages(stand_in, written):
        region.extend(self._stage_member(stand_in, table, counter))
        continue
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

  def _stages(self, stand_in, written) -> bool:
    """Whether a merged run's hole is staged per iteration (`stage_members`):
    a batch-constant, dense operand the body only reads."""
    obj = stand_in.obj
    return bool(self._context.get_user_options().stage_members
                and obj is not None
                and obj.addressing == Addressing.NONE
                and obj.is_dense()
                and id(obj) not in written
                and stand_in.stype != SymbolType.Data)

  def _stage_member(self, stand_in, table, counter):
    """The current member of a merged run's batch-constant hole, copied by the
    whole block into a buffer its multiplications share.

    The table still selects between the members' global bindings, now under a
    name of their own (`glb_v0g`); the shared copy takes the stand-in's name
    and, added after it, is what the body's builders find -- the arrangement
    `GlobalLoaderBuilder` uses for the prologue's images, which is also why
    the members stay out of the preload: a table over shared copies would
    claim global memory.  One barrier in front, since the previous iteration's
    readers -- or the previous element's last -- may still be in the buffer,
    and one behind, since every multiplication reads what the whole block
    wrote.  The buffer is in the prologue's arena and out of the liveness
    that places the per-multiplication windows (`block_shared`).
    """
    from tensorforge.backend.scopes import Symbol
    from tensorforge.backend.instructions.memory.load import (GlbToShrLoader,
                                                              LoadWait)
    pointers = GetElementPtrBuilder(self._context, self._scopes)
    pointers.build(stand_in, table=table, variant=counter,
                   name=f'{stand_in.name}g')
    binding = list(pointers.get_instructions())
    src = self._scopes.get_symbol(stand_in.obj)
    dest = Symbol(name=f'{GeneralLexicon.GLOBAL_MEM_PREFIX}{stand_in.name}',
                  stype=SymbolType.SharedMem, obj=stand_in.obj)
    dest.block_shared = True
    self._scopes.add_symbol(dest)
    loader = GlbToShrLoader(context=self._context, src=src, dest=dest,
                            shr_mem=self._scopes.get_symbol(
                                self._section.shr_mem_obj),
                            num_threads=self._num_threads, permute=None,
                            blockwide=True, max_load_offset=0, verbatim=True)
    obj = self._section.shr_mem_obj
    loader.set_shr_mem_offset(obj.alloc_global(loader.compute_shared_mem_size()),
                              True, True)
    self._section.stage_loaders.append(loader)
    # The wait goes where `MoveLoads` puts every load's: at the loader's place,
    # in front of the second barrier, with the transfer hoisted up to the
    # first.  Written here as well it would be a second wait the pass takes
    # for a load.  Without the pass, an asynchronous copy still has to be
    # retired before the barrier publishes it.
    wait = ([] if self._context.get_user_options().enable_move_loads
            else [LoadWait(loader)])
    return binding + [SyncBlock(self._context), loader, *wait,
                      SyncBlock(self._context)]

  def _stage_group(self) -> int:
    """Multiplications per block under `stage_members`: `stage_group`,
    rounded up to whole waves."""
    wave = self._context.get_vm().get_hw_descr().vec_unit_length
    base = (wave // self._num_threads if self._num_threads < wave
            else mults_per_group(self._num_threads, wave))
    base = max(1, base)
    want = max(1, int(self._context.get_user_options().stage_group))
    return -(-want // base) * base

  def _deduce_mults_per_block(self):
    policy = self._thread_block_policy_type(self._context,
                                            self._section.shr_mem_obj.get_global_size(),
                                            self._section.shr_mem_obj.get_size_per_mult(),
                                            self._num_threads,
                                            self._lead_width
                                            * self._context.get_user_options().lead_blocking)
    policy.set_has_barrier(
        any(instr.barrier_scope() is not None for instr in self._section.stream))
    if self._section.stage_loaders:
      policy.set_barrier_group(self._stage_group())
    num_mults_per_block = policy.get_num_mults_per_block()
    fits = num_mults_per_block >= 1
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
    return fits

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
    if self.tuned is not None:
      writer(f'// tuned: {self.tuned.label()}')
    if self._launch is not None:
      writer(f'// launch: {self._launch.describe()}')
    writer('// operands:')
    for matrix in self._scopes.get_global_scope().values():
      writer(f'//   {matrix.obj.gen_descr()}')
    writer('// operations:')
    for item in self.descr_list:
      text = item.summary() if hasattr(item, 'summary') else str(item)
      for line in text.splitlines():
        writer(f'//   {line}')
    # The same, as data, on one line: what `kernel_info` returns, for a tool
    # that reads a kernel back without the generator (`parse_generated`).
    import json
    writer('// tensorforge-meta: '
           + json.dumps(self.kernel_info(), sort_keys=True,
                        separators=(',', ':'), ensure_ascii=False))
    writer.new_line()

  def kernel_info(self) -> dict:
    """What this kernel is, as data: the operands, every operation (a merged
    run's written out, its loop kept beside) and the launch.  The
    `tensorforge-meta` line of the kernel's comment block carries it.
    Deterministic, since it is part of the source the kernel is named after.

    Not the options or the tuning label: those have lines of their own, and a
    comparison of two kernels that differ only in how they were asked for
    (`test_wrap_loads`, `test_full_lane_tails`) strips exactly those.  Nor
    the kernel's name or the target: the name is the prototype's, and a
    kernel whose source named its target would be a different kernel on a
    target it is the same program for (`test_syntax`'s f128 check)."""

    def operand(obj):
      box = obj.get_bbox()
      return dict(name=obj.name, alias=obj.alias,
                  shape=[int(d) for d in obj.shape],
                  bbox=[[int(v) for v in box.lower()],
                        [int(v) for v in box.upper()]],
                  addressing=str(obj.addressing),
                  parts=int(obj.storage_parts),
                  ordered=obj.storage_order is not None,
                  variant=bool(getattr(obj, 'is_variant', False)))

    # An operation names its operands rather than restating them: the
    # tensors are under `operands`, and a view is which one and which part.
    def ref(view):
      return {k: view[k] for k in ('name', 'shape', 'bbox', 'offset',
                                   'addressing', 'is_tmp')}

    def compact(row):
      row = dict(row)
      if 'dest' in row:
        row['dest'] = ref(row['dest'])
      if 'ops' in row:
        row['ops'] = [ref(o) for o in row['ops']]
      if 'body' in row:
        row['body'] = [compact(b) for b in row['body']]
      return row

    operations, loops = [], []
    for descr in self.descr_list:
      if hasattr(descr, 'decompose') and hasattr(descr, 'to_dict'):
        loops.append(compact(descr.to_dict()))
      for op in descr.operations():
        operations.append(compact(op.to_dict()) if hasattr(op, 'to_dict')
                          else dict(kind=str(op)))
    return dict(version=interop.get_version(),
                fp=self._context.fp_as_str(),
                launch=self._launch.to_dict() if self._launch else None,
                operands=[operand(s.obj)
                          for s in self._scopes.get_global_scope().values()],
                operations=operations,
                loops=loops)

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

  def _base_params(self, symbol_list, substitute_tables=False):
    """The kernel's parameters, once, as parameters.

    Four callers used to walk this list building four different strings from
    it, and nothing but a shared loop body kept the four in step.  They now
    read one list and ask each entry what it looks like on their surface.
    """
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
          params.append(KernelParam.table(table))
        continue
      datatype = self._context.fp_type if symbol.obj.datatype is None else symbol.obj.datatype
      if symbol.obj.addressing == Addressing.SCALAR:
        if not symbol.stype == SymbolType.Data:
          params.append(KernelParam.of_symbol(symbol, datatype))
      else:
        params.append(KernelParam.of_symbol(symbol, datatype))
        if symbol.obj.addressing != Addressing.NONE:
          params.append(KernelParam.size(get_extra_offset_name(symbol)))

    for i, section in enumerate(self._sections):
      params.append(KernelParam.size(f'{GeneralLexicon.NUM_ELEMENTS}{i}'))

    if self._flags is not FlagMode.ABSENT:
      # A mask the kernel dereferences unconditionally has no default: the
      # signature is where "you have to pass one" is stated.
      default = ' = nullptr' if self._flags is FlagMode.OPTIONAL else ''
      for i, section in enumerate(self._sections):
        params.append(KernelParam.flags(f'{GeneralLexicon.FLAGS_NAME}{i}',
                                        default))

    return params

  def _declare(self, params, with_defaults=False, host=False):
    lexic = self._context.get_vm().get_lexic()
    return [p.declaration(lexic, with_default=with_defaults, host=host)
            for p in params]

  def _generate_kernel_base_args(self, writer=None):
    """The arguments of the launcher's call into the kernel.

    With `writer`, the locals they need first (`KernelParam.binding`): a
    pointer the kernel declares in a space the launcher's does not carry.
    """
    global_symbols = self._scopes.get_global_scope().values()
    lexic = self._context.get_vm().get_lexic()
    params = self._base_params(global_symbols, substitute_tables=True)
    if writer is not None:
      for p in params:
        bound = p.binding(lexic)
        if bound is not None:
          writer(bound)
    return [p.argument(lexic) for p in params]

  def _generate_kernel_proto(self, writer):
    global_symbols = self._scopes.get_global_scope().values()

    str_params = ', '.join(self._declare(
        self._base_params(global_symbols, substitute_tables=True)))

    mults_per_block = self._launch.mults_per_block
    shr_total_size = self._launch.shared_elements

    total_num_threads_per_block = self._num_threads * mults_per_block

    lexic = self._context.get_vm().get_lexic()

    launch_bounds = (total_num_threads_per_block,)

    return lexic.kernel_definition(writer, launch_bounds, self._base_kernel_name, str_params, self._context.fp_as_str(),
                                         shr_total_size, global_symbols)

  def _generate_launcher_proto(self, with_defaults=True):
    global_symbols = self._scopes.get_global_scope().values()

    params = self._base_params(global_symbols)
    params.append(KernelParam.opaque(
        'void*', GeneralLexicon.STREAM_PTR_STR,
        ' = nullptr' if with_defaults else ''))
    str_params = ', '.join(self._declare(params, with_defaults=with_defaults,
                                         host=True))
    return f'void launcher_{self._base_kernel_name}({str_params})'

  def default_generate_call_site(self):
    if not self._is_registerd:
      raise RuntimeError('generator is not registered. Call register first.')
    symbols = deepcopy(list(self._scopes.get_global_scope().values()))
    for item in symbols:
      if item.obj.alias:
        item.name = item.obj.alias

    args = [p.argument() for p in self._base_params(symbol_list=symbols)]
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
    """Where a thread's traversal starts, as text the IR carries as an operand.

    Parenthesised, like the stride beside it, because it is a *sum* and the
    reader decides the precedence.  It reaches the loop as `lo`, and
    `wrap_prefetch` puts `lo` in the induction's place when it peels an
    iteration: the peeled address then reads `lo * stride`, which without the
    parentheses parsed as `threadIdx.y + blockDim.y * blockIdx.x * stride` --
    the right element for row 0 and the wrong one for every other row.
    """
    lexic = self._context.get_vm().get_lexic()
    if block is None:
      block = lexic.block_idx_x
    return f'({lexic.thread_idx_y} + {lexic.block_dim_y} * ({block}))'

  # NOTE: _get_element_size_guard and _get_flag_guard moved onto BatchLoop,
  # which is the only thing that needed them.
