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
from typing import NamedTuple, Optional

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
  """Whether the option asks for blocks outright: `True` -- `auto` decides
  kernel by kernel (`auto_refusal`)."""
  return getattr(context.get_user_options(), 'sycl_element_blocks',
                 False) is True


def auto(context) -> bool:
  """Whether the option leaves the blocks to J6b's rule (`auto_refusal`)."""
  return getattr(context.get_user_options(), 'sycl_element_blocks',
                 False) == 'auto'


def wanted(context) -> bool:
  """Whether a kernel may take blocks: asked for, or left to the rule."""
  return asked(context) or auto(context)


#: The lane slots a column an operand the blocks take may have where `auto`
#: builds them (`auto_refusal`, rule 4).
AUTO_RAGGED_SLOTS = 6

#: The times a body may write an element operand where `auto` builds the
#: blocks under the operators' 2D blocks (`auto_refusal`, rule 6).
AUTO_BLOCK2D_WRITES = 3


def auto_refusal(context, transfers, surfaces) -> Optional[str]:
  """Why `sycl_element_blocks=auto` keeps a kernel's element blocks off, or
  None where it builds them -- the rule pvcport J6b's, J7's and J9's
  timings support (the Max 1550; SeisSol's elastic kernels at orders 5 to 8 on
  NVIDIA's padding, SeisSol's own PVC layout and the layouts step-2 rule;
  the blocks under `sycl_dpas` against `sycl_dpas` alone, the batch as it
  is and every element 16 bytes off its alignment):

  1. the DPAS path (`Options.sycl_dpas` on): under FMA the blocks lost on
     localFluxAll (x1.11, J6b);
  2. whole lane slots for every operand -- moot since the last slot of a
     leading dimension that is no multiple of 16 is read and written as the
     block that ends with the column (`overlap_shift`);
  3. no element operand the body writes and then reads for another one's
     product -- the derivative's chain, each derivative read back for the
     next and for the integral, x0.99-1.73 at orders 5 to 8 -- an
     accumulator read back to be added to is no chain; across the sections
     of a kernel fission cut, the kernel's (`chain_refusal`,
     `context.element_chain`);
  4. at most `AUTO_RAGGED_SLOTS` lane slots a column for every operand the
     blocks take: at order 7, six slots (leading dimensions 84, 88, 96),
     the volume paid (x0.90-0.94) and localFluxAll (x0.70-0.74); at order 8,
     eight, the volume lost (x1.11-1.22) and so did localFluxAll on
     NVIDIA's padding and SeisSol's PVC layout (x1.03-1.04);
  5. an element operand the body writes more than once -- the volume's Q,
     written once a product, three times, localFluxAll's four.  A
     calibration, not a mechanism: it tells apart the three kernels J7
     timed.  The neighbor flux writes its Q once, and lost with the blocks
     on SeisSol's PVC layout and the step-2 rule (x1.10-1.35 at orders 6 to
     8, its blocks' arm itself, for a reason J7 does not show) and gained at
     most 7 % on NVIDIA's padding, where a batch off its alignment cost 23 %
     at order 8;
  6. under the operators' 2D blocks (`Options.sycl_dpas_block2d`, on or
     `auto`), no element operand written more than `AUTO_BLOCK2D_WRITES`
     times: there localFluxAll's four writes lost with the blocks at every
     order J9 timed -- block2d alone x1.03-1.14 at orders 5 and 7, with 32
     threads x1.05-1.60 at order 5 and seven of nine layouts and orders 5
     to 7 (J7 and J8 alike) -- where the volume's three paid (x0.43-0.94
     with 32 threads, x0.48 and x0.83 alone).  Under 32 threads without
     block2d the blocks paid for localFluxAll at order 5 (x0.71-0.76) and
     lost at order 7 (x1.13-1.24): the order, not the base, and no rule
     here.  `auto` block2d counts as on; J9 timed it on.  A calibration
     like rule 5.

  `transfers` are the body's `(kind, name, first_row)` in order, for every
  operand of the element -- a load or a store of `name`, the operand row
  it starts at -- a product's loads before the store of its result;
  `surfaces` the surfaces of the operands the blocks take, by name.  With
  the rule, J7's step (derivative, volume, localFluxAll, four neighbor
  fluxes) is 1.6-2.7 % faster at order 5, 4.1-4.8 % at order 6 and 7.8-9.4 %
  at order 7, order 8 as without, and a batch off its alignment is never
  more than 0.1 % slower than without.  Off by default."""
  if getattr(context.get_user_options(), 'sycl_dpas', False) is not True:
    return 'not the DPAS path'
  chain = getattr(context, 'element_chain', None)
  if chain is not None:
    return chain
  stored, read_back, writes = set(), set(), {}
  for kind, name, _ in transfers:
    if kind != 'store':
      if name in stored:
        read_back.add(name)
      continue
    chained = sorted(read_back - {name})
    if chained:
      return f'{", ".join(chained)} read back for {name}'
    read_back.clear()
    stored.add(name)
    writes[name] = writes.get(name, 0) + 1
  for name in sorted(surfaces):
    slots = -(-surfaces[name].rows // LANES)
    if slots > AUTO_RAGGED_SLOTS:
      return f'{name}: {slots} lane slots a column'
  if not any(count > 1 for count in writes.values()):
    return 'no element operand written more than once'
  if getattr(context.get_user_options(), 'sycl_dpas_block2d', False):
    name, count = min(writes.items(), key=lambda item: (-item[1], item[0]))
    if count > AUTO_BLOCK2D_WRITES:
      return f'{name} written {count} times under sycl_dpas_block2d'
  return None


def chain_refusal(descrs) -> Optional[str]:
  """Rule 3 of `auto_refusal` asked of a kernel's operations in their
  order, whatever sections fission cuts them into -- `auto_refusal` sees a
  section's transfers alone, and a derivative read back in the section
  after the one that wrote it is a load of nothing the section stored: a
  tensor of the element's own, not a temporary, that one operation writes
  and a later one reads to write another.  The derivative writes each
  derivative and reads it back for the next and for the integral; an
  accumulator read to be added to is its own operation's destination, and
  a temporary lives in the body.  None where there is no such tensor."""
  written = set()
  for operation in (o for d in descrs for o in d.operations()):
    dest = operation.writes()
    target = getattr(dest, 'tensor', None)
    if target is None:
      continue
    for view in operation.reads():
      tensor = getattr(view, 'tensor', None)
      if (tensor is not None and tensor is not target
          and id(tensor) in written):
        return (f'{tensor.alias or tensor.name} read back for '
                f'{target.alias or target.name}')
    addressing = getattr(target, 'addressing', None)
    if (not getattr(target, 'is_tmp', False) and addressing is not None
        and addressing.per_element()):
      written.add(id(target))
  return None


def target_refusal(context) -> Optional[str]:
  """Why the target has no element blocks, or None: SPMD SYCL on an Intel
  device under oneAPI, which states its sub-group of 16 -- the explicit
  vector form has a backend of its own (`EXPLICIT_SIMD_BACKENDS`)."""
  target = context.target
  if not getattr(target, 'sycl', False):
    return 'not SYCL'
  if target.hw.vendor != 'intel' or target.backend != 'oneapi':
    return 'not SPMD SYCL under oneAPI on Intel'
  sizes = target.pinned_sub_groups
  if sizes is None or LANES not in sizes:
    return f'no sub-group of {LANES} stated'
  if context.fp_type is not Datatype.F32:
    return 'not single precision'
  return None


def cacheopts_refusal(context, option: str) -> None:
  """Refuse `option` -- `sycl_element_streaming` or `sycl_element_prefetch`,
  which go through IGC's own cache-opts and prefetch forms of the 2D block
  messages -- where the `ocloc` on PATH does not resolve them
  (`toolchain.intel_block2d_cacheopts`), naming the reason: never a quiet
  return to messages without the cache option asked for.  Where there is no
  `ocloc` to ask, the build is the caller's to make."""
  from tensorforge import toolchain
  from tensorforge.common.exceptions import GenerationError
  device = getattr(context.target.hw, 'model', None) or 'pvc'
  if toolchain.intel_block2d_cacheopts(device) is False:
    raise GenerationError(
        f'{option}: the IGC behind the ocloc on PATH does not resolve its '
        f'cache-opts and prefetch forms of the 2D block messages '
        f'({", ".join(toolchain.BLOCK2D_CACHEOPTS_BUILTINS)}); IGC 2.41.5 '
        f'and 2.42.6 do.  Turn {option} off or put such an ocloc on PATH.')


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


def overlap_shift(found: Surface, first_row: int, slot: int) -> Optional[int]:
  """For the lane slot `slot` of an image whose rows start at `first_row`
  that leaves the operand's columns at their end -- the last slot of a
  leading dimension that is no multiple of 16, rows 48-55 of SeisSol's 56 --
  the lanes it lies past the block that ends with the column: read as the
  16 rows before the end (`x = rows - 16`, inside the column), the slot's
  lane `l` is that block's lane `l + shift`, which a rotation of the
  sub-group puts in place (`Lexic.sub_group_shift`); the lanes past the
  column are the image's padding.  None for a slot inside the columns, one
  before them, one with none of its rows in them, or a column of fewer than
  16 rows."""
  start = first_row + slot * LANES
  if (start < 0 or start + LANES <= found.rows or start >= found.rows
      or found.rows < LANES):
    return None
  return start - (found.rows - LANES)


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


__all__ = ['AUTO_BLOCK2D_WRITES', 'AUTO_RAGGED_SLOTS', 'BASE_ALIGN', 'LANES',
           'ROWS', 'Surface',
           'active', 'asked', 'auto', 'auto_refusal', 'building',
           'chain_refusal', 'chunks',
           'needs_check', 'overlap_shift', 'slots_inside', 'surface',
           'surface_refusal', 'target_refusal', 'wanted']
