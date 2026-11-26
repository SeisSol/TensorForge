# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""An element's operands by 2D block messages (`Options.sycl_element_blocks`).

Under SPMD for Intel a sub-group of 16 is one element's multiplication, its
lanes the rows of the operands' register images and their columns in each
lane's registers.  A transfer between an element's operand in global memory
and such an image reads or writes it a (lane slot, column) at a time, one
message each.  A 2D block message moves eight columns of a slot at once: the
element's storage is the surface, its columns the surface's rows -- `pitch`
bytes, the leading dimension -- and lane `l` of a block at column `x` gets
row `x + l` of the operand (`pir.block_reads.load_block2d`,
`store_block2d`, `tensorforge::elementBlock2d`).

What is decided here, and nowhere else:

* which operands have such a surface (`surface`): an operand of one
  element, in global memory, F32, stored dense in its box with its rows
  along the leading dimension, a pitch the rules of the messages allow;
* which transfers take it (`GlbToRegLoader`, `StoreRegToGlb`): those into or
  out of an image of sixteen lanes a slot, the multiplication's;
* what the device has to check (`needs_check`): the messages want a 64-byte
  base, the element needs only the alignment its pitch and 64 share (`grain`)
  -- 32 bytes for SeisSol's padding to 32 on PVC -- and the caller may have
  promised that (`Options.element_base_align`);
* where the blocks are on (`active`): the transfers of the body built behind
  the check that `BatchLoop` chose, which builds the body once more behind
  its failure, as today.  A read of an operand the body wrote before it
  stays as it is, and so does a write the body reads back after it: the
  pseudo-IR takes such a read from the register the lane stored
  (`pir.forward`), and a block -- a vector, a call -- is no store or read
  it can match.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterable, NamedTuple, Optional

from tensorforge.common.basic_types import Datatype

#: The lanes of a 2D block message, and so of the slot it moves.
LANES = 16

#: The base alignment the messages want (`block_reads.SURFACE_BASE_ALIGN`).
BASE_ALIGN = 64

#: The columns one message moves; the ninth of SeisSol's quantities is a
#: message of its own.
ROWS = 8


class Surface(NamedTuple):
  """An operand's storage as the surface of an element: `columns` rows of
  `pitch` bytes, `rows` elements each (the leading dimension)."""
  pitch: int
  columns: int
  rows: int

  @property
  def grain(self) -> int:
    """The alignment an element needs for its surface: the largest power of
    two dividing both the pitch and 64 (`detail::ElementSurface`)."""
    return min(self.pitch & -self.pitch, BASE_ALIGN)


def asked(context) -> bool:
  """Whether the option asks for blocks: `True` only -- `auto` is off until
  the blocks are measured."""
  return getattr(context.get_user_options(), 'sycl_element_blocks',
                 False) is True


def target_refusal(context) -> Optional[str]:
  """Why the target has no element blocks, or None: SPMD SYCL on an Intel
  device under oneAPI, which states its sub-group of 16."""
  target = context.target
  if not getattr(target, 'sycl', False) or getattr(target, 'explicit_simd',
                                                   False):
    return 'not SPMD SYCL'
  if target.hw.vendor != 'intel' or target.backend != 'oneapi':
    return 'not oneAPI on Intel'
  sizes = target.pinned_sub_groups
  if sizes is None or LANES not in sizes:
    return f'no sub-group of {LANES} stated'
  if context.fp_type is not Datatype.F32:
    return 'not single precision'
  return None


def surface_refusal(symbol) -> Optional[str]:
  """Why `symbol` has no element surface, or None."""
  from tensorforge.backend.symbol import SymbolType
  from tensorforge.common.matrix.tensor import Tensor
  if symbol.stype is not SymbolType.Global:
    return 'not in global memory'
  tensor = symbol.obj
  if not isinstance(tensor, Tensor):
    return 'not a tensor'
  kind = getattr(getattr(tensor, 'addressing', None), 'kind', None)
  if kind is None or not kind.per_element():
    return 'no data of an element\'s own'
  if symbol.get_fptype() is not Datatype.F32:
    return 'not F32'
  for field in ('storage_order', 'simt_interleave', 'slot_major',
                'storage_surface', 'packing'):
    if getattr(tensor, field, None) is not None:
      return f'stored in another order ({field})'
  if getattr(tensor, 'storage_planar', False):
    return 'stored in planes'
  if getattr(tensor, 'storage_parts', 1) != 1:
    return 'stored in parts'
  if getattr(symbol, 'stack', lambda: 1)() > 1:
    return 'a stack of elements'
  shape = tensor.get_actual_shape()
  if len(shape) != 2:
    return f'of rank {len(shape)}, not 2'
  rows, columns = int(shape[0]), int(shape[1])
  pitch = rows * 4
  if pitch < BASE_ALIGN or pitch % 16:
    return (f'a column of {pitch} bytes: the messages want at least '
            f'{BASE_ALIGN} and a whole number of 16')
  if not 1 <= columns <= (1 << 24):
    return f'{columns} columns'
  return None


def surface(symbol) -> Optional[Surface]:
  """`symbol`'s element surface, or None where it has none."""
  if surface_refusal(symbol) is not None:
    return None
  rows, columns = (int(n) for n in symbol.obj.get_actual_shape())
  return Surface(pitch=rows * 4, columns=columns, rows=rows)


def needs_check(context, found: Surface) -> bool:
  """Whether the device has to check an element for `found`'s surface:
  unless the caller promised an alignment the surface needs no more of
  (`Options.element_base_align`)."""
  promise = int(getattr(context.get_user_options(), 'element_base_align',
                        0) or 0)
  return not (promise > 0 and promise % found.grain == 0)


def active(context, transfer) -> Optional[Surface]:
  """The surface `transfer` reads or writes by blocks in the body being
  built (`BatchLoop._element_plan`), or None."""
  chosen = getattr(context, 'element_blocks', None)
  if not chosen:
    return None
  return chosen.get(id(transfer))


@contextmanager
def building(context, chosen):
  """The blocks on for the transfers `chosen` -- by `id`, to their surfaces
  -- while a body is built."""
  before = getattr(context, 'element_blocks', None)
  context.element_blocks = dict(chosen)
  try:
    yield
  finally:
    context.element_blocks = before


def slots_inside(found: Surface, first_row: int, slot: int) -> bool:
  """Whether the lane slot `slot` of an image whose rows start at
  `first_row` of the operand lies inside its columns: the block reads or
  writes its 16 rows whole."""
  start = first_row + slot * LANES
  return start >= 0 and start + LANES <= found.rows


def chunks(columns_lo: int, columns_hi: int):
  """The messages a slot's columns `lo` to `hi` take: `(first, rows)` of
  eight columns while eight are left, then the rest one 4, 2, 1 rows each
  -- 8 + 1 for SeisSol's nine quantities."""
  out, q = [], columns_lo
  for rows in (8, 4, 2, 1):
    while columns_hi - q >= rows:
      out.append((q, rows))
      q += rows
  return out


__all__ = ['BASE_ALIGN', 'LANES', 'ROWS', 'Surface', 'active', 'asked',
           'building', 'chunks', 'needs_check', 'slots_inside', 'surface',
           'surface_refusal', 'target_refusal']
