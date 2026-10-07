# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
import copy
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
from tensorforge.common.context import Context
from tensorforge.backend.writer import Writer
from tensorforge.common.exceptions import InternalError
import warnings
from contextlib import contextmanager

from tensorforge.backend import pir
from tensorforge.backend.pir.core import (Access, Effect, MemSpace,
                                          Uniformity)


def _record_pressure(context, body, simd: bool,
                     num_threads: Optional[int] = None) -> None:
  """Measure `body` and hand the figure to the context, split by file.

  What separates the two files is whether a whole wave agrees on the value,
  and that is a fact about the geometry rather than about the value: a
  multiplication narrower than a wave holds several per wave, so mult-uniform
  is then not wave-uniform.  `Participants.WAVE.arrival` answers exactly that
  question for barriers, and the answer is the same one here.

  The thread count comes from the instruction where there is one and from the
  context otherwise -- the shared body is built by a classmethod, which has no
  instruction to ask.
  """
  from tensorforge.backend.pir.core import Participants
  wave = context.target.hw.vec_unit_length
  threads = num_threads or getattr(context, 'lane_threads', None) or wave
  split: List[int] = []
  total = pir.pressure(body, in_bytes=True, explicit_simd=simd,
                       by_file=split,
                       wave_uniform=Participants.WAVE.arrival(threads, wave),
                       folded_crosslane=context.target.folds_broadcast())
  context.record_pressure(total, *split)


def _explicit_simd(context) -> bool:
  """Whether this kernel is lowered with the lane in the type.

  Asked of the target rather than passed down, for the same reason `emit()`
  asks it: the two lowerings are already distinguished there, and a second
  place to decide it is a second place for the two to disagree.
  """
  try:
    return bool(context.target.explicit_simd)
  except AttributeError:
    return False


def _check_register_budget(body, simd: bool, context, where: str) -> None:
  """Warn when a body asks for more register file than a thread has.

  Only an explicit vector comes close: under SPMD a value is one register per
  thread and a register-resident tile is split across the threads that hold
  it.  Under an explicit vector the work-item holds the *whole* tile --
  `align(lead, threads) x nonlead` elements -- and 11 of the corpus's 46 ESIMD
  kernels are over the 8 kB a PVC thread gets, the worst by a factor of five.

  A warning and not an error, deliberately.  Spilling to scratch is slow and
  correct, the compiler already does it silently, and refusing to generate
  would take a working kernel away over a budget this generator cannot
  enforce anyway -- `[[intel::grf_size(256)]]` doubles it at the cost of
  halving the threads in flight, which is a decision and not a fallback.  What
  the warning adds is that somebody is told.

  Narrowing the vector barely helps, and it is worth saying why: `lanes *
  slots` is the lead dimension rounded up to a multiple of the thread count,
  so a narrower vector trims the rounding waste (56 rows: 504 elements at 8
  threads against 576 at 16) and leaves `lead x nonlead` alone.  The lever is
  how much of the operator one work-item owns, not how wide its registers are.
  """
  budget = getattr(context.target.hw, 'max_reg_per_thread', None)
  if not simd or budget is None:
    return
  used = pir.pressure(body, in_bytes=True, explicit_simd=simd)
  if used > budget:
    warnings.warn(
        f'{where}: {used} B of register file per work-item against a budget '
        f'of {budget} B -- this body will spill to scratch. The tile is '
        f'align(lead, threads) x nonlead elements, so a narrower vector does '
        f'not shrink it; what changes it is how much of the operator one '
        f'work-item owns.',
        RegisterBudgetWarning, stacklevel=2)


#: How much of `max_reg_per_thread` a body may take and still be expected not
#: to spill.  From sm_120, where every lane geometry of the corpus that spilled
#: more than a few registers came out above it under the slot-level
#: `pir.pressure`, and none that fitted did.
FIT_FRACTION = 0.85


def _fused_if_over_budget(context, attempt, finish):
  """`attempt()`, finished, and once more with fused broadcasts if the first
  one does not fit.

  A materialized broadcast -- one DPP move whose result several plain FMAs
  read, the arrangement VOPD and packed math want -- keeps every moved value
  in a register of its own until the last product has read it.  Where that
  fits it lets the FMAs pair; where it does not, the compiler spills: local_flux
  at 16 lanes on gfx1150 went to 5.6 KB of scratch and ran 46 times slower,
  and the fused form, one `v_fmac_f32_dpp` per product, ran 8 % *faster* than
  the default.  `select_broadcast_form` cannot see the body it is part of, so
  the question is asked of the body instead: built, measured, and built again
  with the broadcast fused if a materialized one took it over the budget.

  `finish(builder, body)` is what the caller does to a built body before
  emitting it -- the pipeline -- and the measurement is taken of its result:
  that is the body emitted where it fits, so nothing is finished twice.

  Not under an explicit vector, where `pir.pressure` still counts arrays whole
  and so would push every body back; and nothing happens where the target
  states no budget or no broadcast was materialized.
  """
  context.materialized_broadcast = False
  builder, body = attempt()
  body = finish(builder, body)
  if not getattr(context, 'materialized_broadcast', False):
    return builder, body
  simd = _explicit_simd(context)
  budget = getattr(context.target.hw, 'max_reg_per_thread', None)
  if simd or budget is None:
    return builder, body
  used = pir.pressure(body, in_bytes=True, explicit_simd=simd)
  if used <= FIT_FRACTION * budget:
    return builder, body
  context.force_fused_broadcast = True
  try:
    builder, body = attempt()
    return builder, finish(builder, body)
  finally:
    context.force_fused_broadcast = False
    context.materialized_broadcast = False


class RegisterBudgetWarning(UserWarning):
  """A body needs more register file per thread than the target provides."""


def _as_tuple(x) -> Tuple:
  if x is None:
    return ()
  if isinstance(x, (list, tuple, set, frozenset)):
    return tuple(v for v in x if v is not None)
  return (x,)


class AbstractInstruction(ABC):
  def __init__(self, context: Context):
    if not isinstance(context, Context):
      raise RuntimeError(f'received wrong type, expected Context, given {type(context)}')

    self._context = context
    self._fp_as_str = context.fp_as_str()
    self._is_ready = False

  # ----------------------------------------------------------------- #
  # Data-flow interface
  #
  # These three methods are the single interface, so that no pass has to
  # discriminate on ``isinstance`` against a concrete class and then reach
  # for whichever of ``get_dest`` / ``get_src`` / ``get_operands`` /
  # ``._dest`` / ``._src`` that class happens to expose.  The defaults
  # below adapt those accessors, so a subclass that has them needs nothing
  # more.
  #
  # Contract: a subclass that cannot describe itself must *not* look
  # pure.  The default ``accesses()`` returns an UNKNOWN-space access in
  # that case, so a pass that reorders on the basis of accesses stays
  # conservative rather than silently gaining permission.
  # ----------------------------------------------------------------- #

  def substitute(self, old, new) -> bool:
    """Make this instruction name `new` wherever it named `old`.

    Reflective over the same attributes `defs` and `uses` read, and for the
    same reason: what an instruction calls its destination and its operands is
    a convention, not an interface, and a substitution that knew each kind
    would have to be extended for every kind that is ever added.

    Returns whether anything changed, so a caller can tell a substitution that
    did nothing from one it never reached.
    """
    changed = False
    for attr in ('_dest', '_src'):
      if getattr(self, attr, None) is old:
        setattr(self, attr, new)
        changed = True
    for attr in ('_ops', '_srcs', '_operands'):
      held = getattr(self, attr, None)
      if isinstance(held, list) and any(x is old for x in held):
        setattr(self, attr, [new if x is old else x for x in held])
        changed = True
      # An operand held as a view (`SymbolView`: the multilinear's `_ops`,
      # the pointwise `_srcs`) names its symbol one level down.  Missed, a
      # merged run closing its chain (`Generator`, `carried`) would rename the
      # epilogue writing the image and leave every reader of it in the body
      # on the register the substitution retired -- never written inside the
      # loop.  A copy, not the view itself: a view may be shared with
      # instructions outside the region.
      if isinstance(held, list) and any(
          getattr(x, 'symbol', None) is old for x in held):
        setattr(self, attr, [_renamed(x, new)
                             if getattr(x, 'symbol', None) is old else x
                             for x in held])
        changed = True
    view = getattr(self, '_dest', None)
    if getattr(view, 'symbol', None) is old:
      self._dest = _renamed(view, new)
      changed = True
    return changed

  def defs(self) -> Tuple:
    """Symbols written by this instruction."""
    get_dest = getattr(self, 'get_dest', None)
    if callable(get_dest):
      return _as_tuple(get_dest())
    return _as_tuple(getattr(self, '_dest', None))

  def partial_defs(self) -> Tuple:
    """Symbols among `defs()` that this instruction writes only in part.

    A write that covers part of a buffer defines it without ending what was
    there before: the rest still holds what an earlier write put in, and a
    later read may want both.  Liveness is the one analysis that has to tell
    the two apart -- it kills a symbol at its definition, and killing a buffer
    at its second slice makes the first slice dead in between, so the region
    allocator hands that stretch to another buffer.  Everything else keeps
    treating a partial write as a write, which it is.
    """
    return ()

  def uses(self) -> Tuple:
    """Symbols read by this instruction."""
    out = []
    get_operands = getattr(self, 'get_operands', None)
    if callable(get_operands):
      out += list(_as_tuple(get_operands()))
    get_src = getattr(self, 'get_src', None)
    if callable(get_src):
      out += list(_as_tuple(get_src()))
    elif not out:
      out += list(_as_tuple(getattr(self, '_src', None)))
    # de-duplicate, keep order
    seen, uniq = set(), []
    for sym in out:
      if id(sym) not in seen:
        seen.add(id(sym))
        uniq.append(sym)
    return tuple(uniq)

  def describes_dataflow(self) -> bool:
    """Whether ``defs()``/``uses()`` are trustworthy for this instruction."""
    return bool(self.defs() or self.uses())

  def barrier_scope(self) -> Optional[Uniformity]:
    """Who has to arrive here, or `None` where this is not a barrier.

    `None` rather than a rung of its own: the ladder is ordered by "the same
    across more threads", and "not a barrier" is not a point on it.  A rung
    would compare against the others and every comparison would then have to
    exclude it by hand.
    """
    return None

  def convergence_scope(self) -> Optional[Uniformity]:
    """How far threads have to execute this in step, or `None`.

    Not a barrier -- nothing waits here and no memory is ordered -- but the
    same demand on the region: a wave-collective instruction (a matrix
    fragment product) is issued by every lane of the wave together, so where a
    wave holds several multiplications, all of them have to reach it the same
    number of times.  Separate from `barrier_scope` because passes read that
    one as "shared memory is ordered here", which this does not promise.
    """
    return None

  def regions(self) -> Tuple[Tuple['AbstractInstruction', ...], ...]:
    """Nested instruction streams, e.g. a loop body.

    Empty for everything except control constructs.  Passes and verify walk
    these, so an instruction that carries a region must report it or its body
    becomes invisible.
    """
    return ()

  def replace_region(self, index: int,
                     instrs: List['AbstractInstruction']) -> None:
    """Swap out one region's body.

    Needed by per-region passes: the manager rewrites a body and hands the
    result back.  Anything that reports a region must accept a replacement,
    otherwise a pass can read it but not transform it.
    """
    raise InternalError(
        f'{type(self).__name__} reports a region but cannot replace it')

  def uniform_scope(self) -> Uniformity:
    """The strongest barrier that may legally appear inside this instruction's
    regions.

    ``GRID`` means no restriction.  A loop whose trip count differs between
    blocks returns ``BLOCK``: a grid barrier in its body would deadlock, since
    blocks with fewer iterations exit without arriving.
    """
    return Uniformity.GRID

  def accesses(self) -> Tuple[Access, ...]:
    """Localized memory effects, in ``pir``'s vocabulary."""
    if not self.describes_dataflow():
      # opaque: conflicts with everything
      return (Access(Effect.READ | Effect.WRITE, MemSpace.UNKNOWN, None),)
    out = []
    for sym in self.uses():
      space = MemSpace.from_symbol_type(getattr(sym, 'stype', None))
      if space is not MemSpace.NONE:
        out.append(Access(Effect.READ, space, sym))
    for sym in self.defs():
      space = MemSpace.from_symbol_type(getattr(sym, 'stype', None))
      if space is not MemSpace.NONE:
        out.append(Access(Effect.WRITE, space, sym))
    return tuple(out)

  def effect(self) -> Effect:
    eff = Effect.NONE
    for acc in self.accesses():
      eff |= acc.kind
      if acc.space is MemSpace.UNKNOWN:
        eff |= Effect.UNKNOWN
    if self.barrier_scope() is not None:
      eff |= Effect.BARRIER
    return eff

  def gen_code(self, writer: Writer) -> None:
    """Route this instruction's body through the pseudo-IR.

    Concrete, not abstract: `gen_ir` is the single hook an instruction
    overrides, and routing is the same for all of them.  `BatchLoop` does
    override this, because it drives child instructions that route
    themselves.
    """
    self.through_pir(writer, self.gen_ir)

  # ---- pseudo-IR routing ------------------------------------------------ #
  #
  # An instruction builds its body into an `IRBuilder` instead of writing text
  # straight into the `Writer`.  Because `IRBuilder` is call-compatible with
  # `Writer`, an instruction that writes text into it produces opaque `raw*`
  # nodes and comes out byte-identical to the direct path; one that overrides
  # `gen_ir` and uses the structured constructors gives the passes something
  # to work with.  What they cannot see is countable: the `raw*` nodes.
  #
  # Set False on a subclass to bypass the IR entirely -- useful for bisecting a
  # suspected emitter difference.
  _use_pir: bool = True

  def gen_ir(self, builder) -> None:
    """Build this instruction's body.  Overriding this is the only hook an
    instruction needs; `gen_code` routes it through the IR."""
    inner = getattr(self, 'gen_code_inner', None)
    if inner is not None:
      inner(builder)

  # One shared body may span several instructions.  While such a scope is
  # open, an instruction builds into it instead of opening its own builder,
  # which is what lets CSE see across instruction boundaries and lets a
  # `copy.async` and its `wait` --- issued by two different instructions ---
  # end up in the same body.
  _shared_body: List = []

  @classmethod
  @contextmanager
  def shared_body(cls, context, writer: Writer):
    """Open one PIR body that several instructions build into.

    A body opened here has no arena: shared memory is placed by the section's
    allocator (`pir.allocate`), so a shared buffer is allocated in a section
    body (`optimized_body`) and nowhere else.
    """
    builder = cls._body_builder(context, getattr(writer, 'alloc', None))
    cls._shared_body.append(builder)
    try:
      yield builder
    finally:
      cls._shared_body.pop()
    cls._emit_shared_body(context, writer, cls._optimize_shared_body(
        context, builder, builder.finish()))

  @classmethod
  def build_shared_body(cls, context, writer, fill) -> None:
    """`shared_body` for a caller that states its contents as `fill(builder)`.

    The one thing a `with` block cannot do is run twice, and that is what this
    is for: a body that chose a materialized broadcast and came out over the
    register budget is built again with the broadcast fused into its FMAs
    (`_fused_if_over_budget`).  Everything after the build is the same as for
    the context manager.
    """
    cls._emit_shared_body(context, writer, cls.optimized_body(
        context, getattr(writer, 'alloc', None), fill))

  @classmethod
  def optimized_body(cls, context, names, fill, arena=None, place=None,
                     barriers=None):
    """`fill(builder)` as one body, through the pipeline, ready to emit.

    The half of `build_shared_body` that needs no writer: `names` is the
    allocator the values are named from, which has to be the one the body is
    emitted with, since a name is unique per file and not per body.  Built
    ahead of the writer, a body is a fact the generator can decide the launch
    from before anything is written.

    For a body that holds a whole section and so everything its shared
    memory and its barriers depend on: `arena` is the multiplication's arena,
    where a shared buffer that names no other one goes; `place` lays the
    buffers out (`pir.PlaceBuffers`) and `barriers` places the barriers
    (`pir.PlaceBarriers`), both behind everything that moves a statement.
    """
    def attempt():
      builder = cls._body_builder(context, names, arena)
      cls._shared_body.append(builder)
      try:
        fill(builder)
      finally:
        cls._shared_body.pop()
      return builder, builder.finish()
    _, body = _fused_if_over_budget(
        context, attempt,
        lambda builder, body: cls._optimize_shared_body(context, builder, body,
                                                        place, barriers))
    return body

  @staticmethod
  def _body_builder(context, names, arena=None):
    return pir.IRBuilder(fptype=context.fp_type, context=context,
                         alloc=names, arena=arena)

  @staticmethod
  def _optimize_shared_body(context, builder, body, place=None, barriers=None):
    """A shared body through the pipeline, with the transfers issued ahead
    where that is asked for: within their statement list, and across the
    batch loop's back edge.

    Both go behind the cleanup and ahead of the allocator
    (`pir.standard_pipeline`), and the statements the second adds are built
    by a builder numbering its values from the one that built this body.
    """
    options = context.get_user_options()
    move = wrap = prefetch = None
    report: list = []
    hints: list = []
    if getattr(options, 'enable_move_loads', False):
      move = pir.MoveLoads(distance=options.move_distance)
    if getattr(options, 'enable_wrap_loads', False):
      wrap = pir.WrapLoads(builder.scratch, distance=options.move_distance,
                           stages=2 if options.enable_multibuffer else 1,
                           report=report)
    if options.enable_prefetch or options.prefetch_data:
      prefetch = pir.Prefetch(builder.scratch,
                              pointers=options.enable_prefetch,
                              data=options.prefetch_data,
                              level=options.prefetch_level,
                              line_bytes=context.target.prefetch_line_bytes(),
                              report=hints)
    body = pir.optimize(body, explicit_simd=_explicit_simd(context),
                        debug=options.ir_debug, move=move, wrap=wrap,
                        place=place, barriers=barriers, prefetch=prefetch,
                        where='shared body')
    if wrap is not None:
      record = getattr(context, 'record_wrap', None)
      if record is not None:
        record(report)
      if options.ir_debug:
        for line in report:
          print(f'wrap: {line}')
    if prefetch is not None and options.ir_debug:
      for line in hints:
        print(f'prefetch: {line}')
    return body

  @staticmethod
  def _emit_shared_body(context, writer, body) -> None:
    _check_register_budget(body, _explicit_simd(context), context,
                           'shared body')
    if getattr(context, 'measure_pressure', False):
      _record_pressure(context, body, _explicit_simd(context))
    pir.emit(body, writer, context)

  def through_pir(self, writer: Writer, build) -> None:
    """Route ``build(sink)`` through the pseudo-IR into ``writer``.

    ``build`` takes the emission sink so that the same closure serves both
    paths; nothing about it knows which one it got.
    """
    if not self._use_pir:
      build(writer)
      return

    if self._shared_body:
      # Join the enclosing body.  Where this instruction defines a buffer
      # whole, it says so ahead of its first write (`_kills`): the buffer
      # holds nothing anybody wants up to there.  A window this instruction
      # declares itself has no value yet, and its `alloc` says the same.
      builder = self._shared_body[-1]
      for sym in getattr(self, '_kills', ()):
        buf = sym.pir_buffer(builder)
        if buf is not None:
          builder.mark('defines', buf)
      build(builder)
      return

    simd = _explicit_simd(self._context)
    debug = self._context.get_user_options().ir_debug
    def attempt():
      builder = pir.IRBuilder(fptype=self._context.fp_type,
                              context=self._context,
                              alloc=getattr(writer, 'alloc', None))
      build(builder)
      return builder, builder.finish()
    _, body = _fused_if_over_budget(
        self._context, attempt,
        lambda _builder, body: pir.optimize(body, explicit_simd=simd,
                                            debug=debug,
                                            where=type(self).__name__))

    self._check_register_budget(body, simd)
    # Reported here because this is where the body exists: it is discarded
    # after `emit`, so anything wanting a number about it has to take it now.
    # Computing it always would put a liveness walk into every generation for
    # the sake of the callers that search over configurations, and they are
    # the only ones that read it.
    if getattr(self._context, 'measure_pressure', False):
      _record_pressure(self._context, body, simd,
                       getattr(self, '_num_threads', None))
    if self._context.get_user_options().ir_stats:
      print(f'{type(self).__name__}: {sum(1 for _ in pir.walk(body))} nodes, '
            f'register pressure {pir.pressure(body)} values, '
            f'{pir.pressure(body, in_bytes=True, explicit_simd=simd)} B '
            f'({"per work-item" if simd else "per lane"})')
    pir.emit(body, writer, self._context)

  def _check_register_budget(self, body, simd: bool) -> None:
    """See the module-level `_check_register_budget`."""
    _check_register_budget(body, simd, self._context, type(self).__name__)

  def get_headers(self) -> List[str]:
    return []

  def is_ready(self) -> bool:
    return self._is_ready

  @abstractmethod
  def __str__(self) -> str:
    pass

  def set_threadconfig_pre(self, num_threads, mults):
    pass


def _renamed(view, symbol):
  """`view` over `symbol` instead, the rest of it unchanged."""
  out = copy.copy(view)
  out.symbol = symbol
  return out
