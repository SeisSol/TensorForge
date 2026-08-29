# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
from ..abstract_instruction import AbstractInstruction
from abc import abstractmethod
from tensorforge.backend.symbol import DataView, SymbolType
from tensorforge.backend.writer import Writer
from tensorforge.common.exceptions import InternalError
from tensorforge.common.matrix.tensor import Tensor

class ComputeInstruction(AbstractInstruction):
  @staticmethod
  def lead_dim(view) -> int:
    """Which axis of `view` is spread across the lanes.

    Read off the symbol, because that is where every *reader* takes it from:
    `Symbol.load` and `Symbol.store` index `lead_dims[0]`, `load.py` writes a
    register image with the lane on `self._dest.lead_dims[0]`, and
    `multilinear_builder` sets it to something other than 0 whenever a
    transposed operand carries the lead index elsewhere.

    An instruction that keeps its own copy is stating the same fact a second
    time, and the two can disagree -- which is the failure `load.py` already
    warns about: an image written with the lane on axis 0 while every reader
    addresses it on axis 1 hands each lane the wrong element, with no shape
    check anywhere to notice.

    Takes a `SymbolView` or a bare `Symbol`: the compute instructions carry
    both -- elementwise and the reduction hold views, multilinear holds the
    destination symbol itself -- and the fact being read belongs to the
    symbol either way.
    """
    symbol = getattr(view, 'symbol', view)
    lead = symbol.lead_dims
    if len(lead) != 1:
      raise InternalError(
          f'{symbol.name} declares {len(lead)} lead dimensions; '
          f'exactly one is supported')
    return lead[0]

  def shared_lead_dim(self, views, what: str) -> int:
    """The lane axis of an iteration space whose operands share their shape.

    Only for instructions where iteration axis `i` is axis `i` of every
    operand -- elementwise is the case, and the reduction is not, since
    dropping the contracted axes renumbers what is left.
    """
    leads = {self.lead_dim(v) for v in views}
    if len(leads) > 1:
      raise InternalError(
          f'{what}: operands disagree about the lane axis '
          f'({sorted(leads)}); the loop can only distribute one of them, and '
          f'the others would be addressed on an axis they do not spread')
    return leads.pop() if leads else 0

  @abstractmethod
  def get_operands(self):
    return []

  def claim_destination(self, view, bbox=None) -> None:
    """Give a fresh register array the shape this operation writes into it.

    A register array is allocated with a slot count and nothing else; what
    those slots *mean* is decided by whoever writes them, so the array is not
    addressable until an instruction says so.  The multilinear says it inside
    `_analyze`, from the intersected range it ends up iterating, which is why
    it was the only operation that could write one -- and therefore the only
    one that could produce a temporary at all.

    `bbox` is for an operation whose destination is indexed in a different
    origin than it is declared in.  Everything but the contraction indexes at
    the box it declares and leaves it alone.

    Silent on a destination that already has a view: a global or shared symbol
    gets one from its tensor, and an array being accumulated into across
    several operations keeps the one the first of them established.
    """
    symbol = view.symbol
    if symbol.data_view is not None:
      return
    if symbol.stype not in (SymbolType.Register, SymbolType.Scratch):
      raise InternalError(
          f'{symbol.name} is in {symbol.stype} and has no data view; only a '
          f'register array is shaped by the operation that writes it')
    box = view.bbox if bbox is None else bbox
    symbol.data_view = DataView(shape=box.sizes(),
                                permute=list(range(box.rank())),
                                bbox=box)

  @staticmethod
  def check_addressable(views, what: str) -> None:
    """Every operand has to be something an address can be computed for.

    A tensor-backed symbol carries its shape; a register array carries one
    once the operation that writes it has said so.  Anything else -- a shared
    memory window, a scalar -- has no element to index.
    """
    for view in views:
      if isinstance(view.symbol.obj, Tensor):
        continue
      if view.symbol.stype in (SymbolType.Register, SymbolType.Scratch) \
          and view.symbol.data_view is not None:
        continue
      raise InternalError(
          f'{what}: {view.symbol.name} is neither a tensor nor a shaped '
          f'register array, so it cannot be indexed')

  @abstractmethod
  def gen_code_inner(self, writer: Writer):
    pass

  def gen_ir(self, sink):
    with sink.Scope():
      sink.Comment(self.__str__())
      self.gen_code_inner(sink)
