from abc import ABC, abstractmethod
from enum import IntEnum
from typing import List, Tuple
from tensorforge.common.context import Context, VM
from tensorforge.backend.writer import Writer
from tensorforge.common.exceptions import InternalError
import os

from tensorforge.backend import pir
from tensorforge.backend.pir.core import Access, Effect, MemSpace


class BarrierScope(IntEnum):
  """How far a synchronisation reaches.

  The scope used to be *derived* inside ``SyncThreads.__str__`` from a thread
  count, which made it invisible to any pass.  It is a property of the
  instruction, so it lives here: barrier legality inside a loop depends on
  the trip count being uniform *across the scope*, and only ``GRID`` can
  deadlock a kernel outright.
  """

  NONE = 0
  SIMD = 1     # wave / warp
  GROUP = 2    # thread block
  GRID = 3     # whole grid (cooperative launch)


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
    self._vm: VM = context.get_vm()
    self._fp_as_str = context.fp_as_str()
    self._is_ready = False

  # ----------------------------------------------------------------- #
  # Data-flow interface
  #
  # Until now every pass had to discriminate on ``isinstance`` against a
  # concrete class and then reach for whichever of ``get_dest`` /
  # ``get_src`` / ``get_operands`` / ``._dest`` / ``._src`` that class
  # happened to expose.  These three methods are the single interface;
  # the defaults below adapt the existing accessors so that no subclass
  # has to change at once.
  #
  # Contract: a subclass that cannot describe itself must *not* look
  # pure.  The default ``accesses()`` returns an UNKNOWN-space access in
  # that case, so a pass that reorders on the basis of accesses stays
  # conservative rather than silently gaining permission.
  # ----------------------------------------------------------------- #

  def defs(self) -> Tuple:
    """Symbols written by this instruction."""
    get_dest = getattr(self, 'get_dest', None)
    if callable(get_dest):
      return _as_tuple(get_dest())
    return _as_tuple(getattr(self, '_dest', None))

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

  def barrier_scope(self) -> 'BarrierScope':
    return BarrierScope.NONE

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

  def uniform_scope(self) -> 'BarrierScope':
    """The strongest barrier that may legally appear inside this instruction's
    regions.

    ``GRID`` means no restriction.  A loop whose trip count differs between
    blocks returns ``GROUP``: a grid barrier in its body would deadlock, since
    blocks with fewer iterations exit without arriving.
    """
    return BarrierScope.GRID

  def accesses(self) -> Tuple[Access, ...]:
    """Localised memory effects, in ``pir``'s vocabulary."""
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
    if self.barrier_scope() is not BarrierScope.NONE:
      eff |= Effect.BARRIER
    return eff

  def gen_code(self, writer: Writer) -> None:
    """Route this instruction's body through the pseudo-IR.

    Concrete now, not abstract: `gen_ir` is the single hook an instruction
    overrides, and routing is the same for all of them.  `BatchLoop` still
    overrides this, because it drives child instructions that route
    themselves.
    """
    self.through_pir(writer, self.gen_ir)

  # ---- pseudo-IR routing ------------------------------------------------ #
  #
  # An instruction builds its body into an `IRBuilder` instead of writing text
  # straight into the `Writer`.  Because `IRBuilder` is call-compatible with
  # `Writer`, an un-migrated instruction produces opaque `raw*` nodes and comes
  # out byte-identical; a migrated one overrides `gen_ir` and uses the
  # structured constructors, at which point the passes have something to work
  # with.  Progress is countable: the number of `raw*` nodes left.
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

  def through_pir(self, writer: Writer, build) -> None:
    """Route ``build(sink)`` through the pseudo-IR into ``writer``.

    ``build`` takes the emission sink so that the same closure serves both
    paths; nothing about it knows which one it got.
    """
    if not self._use_pir:
      build(writer)
      return

    builder = pir.IRBuilder(fptype=self._context.fp_type, context=self._context,
                            alloc=getattr(writer, 'alloc', None))
    build(builder)
    body = builder.finish()

    if os.environ.get('TF_IR_DEBUG'):
      diag = pir.verify(body, strict=False)
      if diag:
        print(f'pir diagnostics in {type(self).__name__}:')
        for d in diag:
          print(f'  {d}')

    body = pir.optimize(body)
    if os.environ.get('TF_IR_STATS'):
      print(f'{type(self).__name__}: {sum(1 for _ in pir.walk(body))} Knoten, '
            f'Registerdruck {pir.pressure(body)}')
    pir.emit(body, writer, self._context)

  def get_headers(self) -> List[str]:
    return []

  def is_ready(self) -> bool:
    return self._is_ready

  @abstractmethod
  def __str__(self) -> str:
    pass

  def set_threadconfig_pre(self, num_threads, mults):
    pass

  def gen_mask_threads(self, num_threads) -> str:
    return f'{self._vm.get_lexic().thread_idx_x} < {num_threads}'

  def gen_range_mask_threads(self, begin, end) -> str:
    assert begin < end
    tid = self._vm.get_lexic().thread_idx_x
    if begin == 0:
      return f'{tid} < {end}'
    else:
      return f'({tid} >= {begin}) && ({tid} < {end})'

  # @abstractmethod
  def get_perfdata(self):
    pass

  def temp_shmem(self):
    return 0
