# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Several elements of the batch in one block, side by side: the
multi-element layout (`Options.multi_element`).

Every kernel TensorForge writes otherwise gives each element a
multiplication of its own: lanes over the element's rows, a block holding
as many multiplications as fit, each of them reading the batch-constant
operators by itself.  A matrix engine wants the other arrangement -- many
elements in one instruction's rows, the operator read once for all of
them -- and so does an operator too large to be read again for every
element.  The multi-element layout gives a block a *group* of E elements:
E warps, taking E consecutive elements on every trip of the batch loop
together (the loop's `Trip.ELEMENT_GROUP`), their operands side by side in
shared memory, the operators once for the block.

The layout takes a whole section or nothing, and nothing in it is a kernel
of its own: `classify` sorts the section's operands and products, and the
lowering (`compute.multi_element`) has one form per kind of product.

*Operands*, by what an element of the batch has of them (`Role`):

* `OPERATOR`: batch-constant (`Addressing.NONE`), staged once a block.
* `GROUP`: an element's own matrix over the basis (axis 0) and the
  quantities (axis 1) -- SeisSol's degrees of freedom.  Read ones are
  staged a trip; written ones are the section's outputs, written back per
  element, under its mask, read-modify-write where the description
  accumulates.
* `SMALL`: an element's own small matrix over quantities -- a star --
  staged a trip.
* `INTERMEDIATE`: a temporary between products, in a shared buffer of the
  group, element-outer, in the layout its consumer reads: plain rows for
  the FMA forms; the matrix engines' A ring follows.

*Products*, by their operands (`Kind`):

* `OPERATOR`: an operator times a group operand, `C = K x G`, contracting
  the basis: the matrix tiling on `MATRIX_MAC` (`mma_sync`, 3xTF32, for
  F32; `dmma` for F64), the group's rows -- an element's quantities,
  element after element -- along M, the operator's rows along N, the
  contraction along K; or the FMA form, a lane an output, where the matrix
  engines are off (`matrix_enabled`) -- over the operator's nonzeros alone
  where it carries its numbers (`Terms`).
* `MIXING`: a group operand times a small one, `C = G x S`, contracting
  the quantities: each element's rows mixed by its own operand, on the FMA
  units, a lane an output.

Products into one destination with nothing between them that reads it
are one `Step`, lowered together: the matrix tiling concatenates them
along K and keeps D in registers across them, the FMA forms sum them per
output.  The matrix tiling reads each product's operator as a `Stage` --
its window, zero elsewhere, padded to the instruction's tiles -- and
hands D to its consumer's buffer, or adds it into the output straight
from the accumulators: a tile's rows are several elements', so each warp
shares its element's pointer to the output with the block
(`pointer_tables`), and a lane adds each of its cells through its row's.
In F32 both operands can come in the TF32 halves its 3xTF32 products
multiply, split once rather than at every read where the group has the
room: B as staged (`Stage.split`), A as the star step writes it
(`Buffer.halves`) where the layout's switch takes that (`HALVES`, off).

The *lanes form* (`Options.multi_element_fma` `lanes`, `Lanes`) is the
other arrangement of the FMA units: a lane an element rather than an
output, a warp's lanes the group, and the block's warps each a share of
the one output -- its columns of a class, at its range of rows.  Nothing
is staged: each lane reads its own element through its own pointers, a
mixing step's entries are computed where a product reads them, an
operator step's column is held in registers while a product reads it, and
only the operators' nonzeros are multiplied by -- literals of the code, or
read from the operator's memory at addresses every lane shares
(`lane_literals`).  Every step's lowering is `lane`.

The layout is written for CUDA on NVIDIA, HIP on AMD and DPC++ under SPMD
on Intel (`TARGETS`), and the forms say "warp" for what the target has:
the warp form's warp is 32 lanes everywhere -- half a wave on CDNA, a
sub-group of 32 on Intel, as the kernel states it -- and the lanes form's
is the target's own, a wave or a sub-group of its vector unit's width
(`lane_elements`).  Only the block's barriers reach across warps.  The
matrix tiling is NVIDIA's (`lower`).

`plan` decides whether and how a section takes the layout: every product
classified or none, E for the section, and each step's lowering -- the
fill rule (`fill`), static so far, each refusal with its reason.

`multi_element` is off by default.  `force` takes the layout wherever the
rule finds a plan; `auto` where it finds one measured to pay
(`auto_takes`): so far the lanes form on Intel at F32, where its plan holds
the output in one row range (`AUTO_LANES`).
"""

from __future__ import annotations

import dataclasses
import enum
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

from tensorforge.backend.pir import terms
from tensorforge.common.basic_types import Addressing, Datatype

#: The largest extent of a `SMALL` operand along either axis: an element's
#: own matrix wider than this is no operator over its quantities.
SMALL_EXTENT = 32

#: The group's size by the extent of the basis the section's group operands
#: run over and the precision: what the design settles on for the matrix
#: engines (`~/tf-probe/design/multi-element-volume.md`) -- for the volume,
#: M = 9E rows of the element-outer product.  A section takes the entry of
#: the smallest basis that holds its own, `DEFAULT_ELEMENTS` past the table.
ELEMENTS = {
    (20, Datatype.F32): 32, (20, Datatype.F64): 16,
    (56, Datatype.F32): 16, (56, Datatype.F64): 16,
    (120, Datatype.F32): 14, (120, Datatype.F64): 8,
}
DEFAULT_ELEMENTS = 16

#: The smallest group worth a block: below it the operator is staged for
#: too few elements to pay for the staging.
MIN_ELEMENTS = 4

#: A warp's lanes in the warp form (`ElementGroup.threads`): the `mix` form
#: takes an element's columns a thread each, at most as many.
WARP_LANES = 32

#: Bytes of a block's shared memory left to the rest of the kernel -- the
#: arena's own windows -- when the group's staging is sized.
SHARED_RESERVE = 1024

#: Bytes of an element's entry in a pointer table (`pointer_tables`): a
#: global address.
POINTER_BYTES = 8

#: The layout's deployment switch for the matrix engines: an operator step
#: is tiled on them where `Options.tensor_cores` asks for it, or where the
#: option is not set and this is on.  On: the layout exists for them, and is
#: off by default itself (`Options.multi_element`).
MATRIX = True

#: The layout's switch for A in TF32 halves (`Buffer.halves`): the fill rule
#: takes them, where the group has the room, only where this is on.  Off: on
#: GH200 they cost the 3xTF32 tiling 12.6 % at o6 F32 and 3.8 % at o8 against
#: the split at every read (11-21).  The walk reads A again for every N tile it
#: keeps, so the halves read it twice as often from shared memory, the HMMAs
#: waiting on ldmatrix where a split hid its latency, and their room shrinks
#: L1 to 22 kB; the FP32 split they spare is cheap there.  On sm_120 they were
#: even.  Kept for what reads A split from shared memory itself (wgmma,
#: tcgen05) and for A held in registers across a warp's N tiles, read once.
HALVES = False

#: The targets the layout is written for: a backend, and the vendor it runs
#: on there.  SYCL is DPC++ under SPMD on Intel, whose kernels state their
#: sub-group (`Target.pinned_sub_groups`): the lanes form's warp is one, and
#: the fences in a range's code are sub-group barriers under a condition
#: only whole sub-groups take.  The explicit vector, and AdaptiveCpp, whose
#: device picks the sub-group, are not.
TARGETS = (('cuda', 'nvidia'), ('hip', 'amd'), ('oneapi', 'intel'))


class Role(enum.Enum):
    """What an element of the batch has of an operand (see the module)."""

    OPERATOR = 'operator'
    GROUP = 'group'
    SMALL = 'small'
    INTERMEDIATE = 'intermediate'
    #: A scalar factor of a scaled copy, passed by value (`Kind.SCALE`).
    SCALAR = 'scalar'


class Kind(enum.Enum):
    """A product, by its operands (see the module)."""

    OPERATOR = 'operator'
    MIXING = 'mixing'
    #: `dest (+)= first`, or `dest (+)= first * s` with `s` a scalar factor
    #: (`second`, None for a plain copy): elementwise, on the window of
    #: `first`.
    SCALE = 'scale'


Range = Tuple[int, int]


def _range(sub, axis: int) -> Range:
    return int(sub.bbox.lower()[axis]), int(sub.bbox.upper()[axis])


def _union(ranges: Sequence[Range]) -> Range:
    return min(r[0] for r in ranges), max(r[1] for r in ranges)


def _covers(outer: Range, inner: Range) -> bool:
    return outer[0] <= inner[0] and inner[1] <= outer[1]


@dataclass(frozen=True)
class Product:
    """One product of the section, `dest (+)= first x second` with
    `dest(i, j) = sum_k first(i, k) second(k, j)`, and its kind.

    The views are the description's (`SubTensor`); `rows`, `cols` and
    `depth` are the ranges of `i`, `j` and `k` it runs over."""

    descr: object
    kind: Kind
    first: object
    second: object
    dest: object
    add: bool

    @property
    def rows(self) -> Range:
        return _range(self.dest, 0)

    @property
    def cols(self) -> Range:
        return _range(self.dest, 1)

    @property
    def depth(self) -> Range:
        return _range(self.first, 1)

    @property
    def group(self):
        """The group operand: what the product's rows run over."""
        return self.second if self.kind is Kind.OPERATOR else self.first

    @property
    def operands(self) -> Tuple:
        """The views the product reads: both, or a plain copy's one."""
        return tuple(sub for sub in (self.first, self.second)
                     if sub is not None)

    @property
    def other(self):
        """The operator, or the small operand."""
        return self.first if self.kind is Kind.OPERATOR else self.second


@dataclass(frozen=True)
class Terms:
    """An operator step's products by their operators' nonzeros: what the
    sparse `fma` form walks (`compute.multi_element`) in place of the
    operators' boxes.

    `order` is the destination's rows in the order the lanes take them: by
    how many nonzeros the step's products have in a row, the most first,
    each row's columns side by side -- so that the lanes of a warp, about
    four rows of their element's columns, walk rows of about one length.
    An accumulating step leaves the rows without any out; an assigning one
    keeps them, to write their zeros.  Per product, in the step's order:
    `counts[p][s]` nonzeros at row `order[s]`, and over all of them, row
    after row of `order`, their contraction indices `cols[p]` and their
    cells in the operator's storage `cells[p]`."""

    order: Tuple[int, ...]
    counts: Tuple[Tuple[int, ...], ...]
    cols: Tuple[Tuple[int, ...], ...]
    cells: Tuple[Tuple[int, ...], ...]

    @property
    def entries(self) -> int:
        """The nonzeros of all products: the tables' length."""
        return sum(len(c) for c in self.cols)

    def base(self, product: int) -> int:
        """Where a product's nonzeros start in the tables."""
        return sum(len(c) for c in self.cols[:product])

    @property
    def stride(self) -> int:
        """Entries of a row's data: the row, then each product's first
        nonzero and count."""
        return 1 + 2 * len(self.counts)

    def rows_table(self) -> Tuple[int, ...]:
        """The row data, `stride` entries a row of `order`."""
        firsts = []
        for p, counts in enumerate(self.counts):
            at, mine = self.base(p), []
            for n in counts:
                mine.append(at)
                at += n
            firsts.append(mine)
        return tuple(x for s, row in enumerate(self.order)
                     for x in (row, *(v for p in range(len(self.counts))
                                      for v in (firsts[p][s],
                                                self.counts[p][s]))))

    @property
    def volume(self) -> int:
        """Elements of the block's arena the tables take: a value and an
        index a nonzero, and the row data -- an index takes an element of
        the arena's type whatever its own width (`pir.allocate`)."""
        return 2 * self.entries + len(self.order) * self.stride


@dataclass(frozen=True)
class Step:
    """Products into one destination, lowered together (see the module).

    The first may assign; every other accumulates.  `rows` and `cols` are
    the union of the products' windows of the destination, each product
    adding only into its own.  `lowering` is the fill rule's: `fma` for an
    operator product so far, `rows` for a mixing one.  `terms` are the
    operators' nonzeros where an `fma` step reads only those (`lower`)."""

    products: Tuple[Product, ...]
    lowering: str
    terms: Optional[Terms] = None
    #: How a matrix step adds D into its output where not by a reduction a
    #: cell: `bulk`, a bulk reduction a column of the element's output;
    #: `box`, one over its whole box (`Options.matrix_output`,
    #: `output_refusal`); '' by a reduction a cell.
    output: str = ''

    @property
    def kind(self) -> Kind:
        return self.products[0].kind

    @property
    def dest(self):
        return self.products[0].dest.tensor

    @property
    def assigns(self) -> bool:
        return not self.products[0].add

    @property
    def rows(self) -> Range:
        return _union([p.rows for p in self.products])

    @property
    def cols(self) -> Range:
        return _union([p.cols for p in self.products])

    def reads(self) -> List:
        return [sub.tensor for p in self.products for sub in p.operands]


@dataclass(frozen=True)
class Buffer:
    """The group's copy of one operand in shared memory: per element, the
    box `rows` x `cols` of it, column by column -- an element's rows of a
    group operand side by side, its basis contiguous -- and the elements
    one after another.

    `stride` is how far one column starts from the next where a reader
    wants more than the box's rows -- the matrix tiling's ldmatrix, whose
    rows are 16 bytes, and the `mix` form, which reads a column 16 bytes at
    a time (`_pitched`); `pitch` is the distance either way.
    The elements past the rows are padding, zero.

    `halves`: the buffer holds each value as the two TF32 halves the 3xTF32
    tiling multiplies, split once by the step that writes it rather than by
    the tiling at every read (`_halved`).  The upper halves are laid out as
    the values would be, the lower ones the same way after the whole
    group's upper ones (`half_offset`): each half's rows of a group lie on
    the banks as the values' would, across elements too -- a block per
    element instead would set the next element's rows a block further
    along, which in a fragment's eight rows over an element's end meets a
    bank group twice."""

    tensor: object
    role: Role
    rows: Range
    cols: Range
    stride: int = 0
    halves: bool = False
    #: An intermediate the star step folded into its reader writes where
    #: the engine reads A, never into shared memory as values
    #: (`GroupSection.star`): no room of the group's.
    held: bool = False

    @property
    def height(self) -> int:
        return self.rows[1] - self.rows[0]

    @property
    def pitch(self) -> int:
        return self.stride or self.height

    @property
    def per_element(self) -> int:
        """How far one element's box starts from the next's: its columns
        over the pitch -- of the upper halves, where the buffer holds
        `halves`."""
        return self.pitch * (self.cols[1] - self.cols[0])

    @property
    def room(self) -> int:
        """What an element takes of the group's buffer: its box, and as
        much again for the lower halves where the buffer holds them --
        nothing where the buffer is held by the engine (`held`)."""
        if self.held:
            return 0
        return self.per_element * (2 if self.halves else 1)

    def half_offset(self, elements: int) -> int:
        """Where a cell's lower TF32 half is from its upper one, in a group
        of `elements`: past the upper halves of the whole group."""
        return elements * self.per_element


@dataclass(frozen=True)
class Stage:
    """A batch-constant operator in the block's shared memory, staged once a
    block: as stored (`window` None), for the FMA forms, or as one matrix
    product's B.

    B is the product's window of the operator, transposed onto the
    instruction's axes -- `B(k, n) = K(n0 + n, k0 + k)`, `k0` the window's
    first column, `n0` the step's first row -- zero wherever the product
    has no entry, the contraction padded to `depth` (a multiple of the
    instruction's `k`) and the step's rows to `width` (of its n, 8).  The
    zeros past the window's contraction are load-bearing: the tilings'
    padded steps of K multiply them by the group's rows past the product's
    depth, and an input's padding there holds its column's first row, not
    zero (`compute.multi_element.MultiElementSection._stage`) -- so every
    engine's B has them, staged or streamed through a ring.  It is
    stored in the order the instruction's fragments are read: tile after
    tile of `k` x 8, and in a tile lane after lane, each lane's registers
    of B side by side (`regs`) -- one load a lane and fragment.  `split`
    stores each value as its two TF32 halves, the upper ones first, which
    the 3xTF32 products then read prepared (`fill`).

    The tiles are those of `tiles`, `(kt, nt)` on the stage's grid -- K
    from the window's first column in steps of `k`, N from `n0` in steps
    of 8 -- in the order stored: only those where the operator has an entry
    in the window of a product the stage serves (`kept_tiles`), so that a
    tile of zeros is neither stored nor multiplied.  `n0` is the grid's
    origin: the step's first row, or that rounded down to a tile
    (`_staged`'s `aligned`), where a consumer's tiles have to line up with
    another step's.

    An asynchronous engine reads B through a descriptor instead (`order`
    `canonical`, `ASYNC_LOWERINGS`): the canonical K-major layout of core
    matrices, eight rows of N by four TF32 of K, 128 contiguous bytes each,
    the ones along K 128 bytes apart and those along N `depth` x 32
    (`canonical_offset`); split, the upper halves' layout whole, then the
    lower halves'.  Whole: its descriptors walk every tile, and its N grid
    is the step's first row on, D's columns the step's rows.

    `prepared`: such a B, split, made at generation from the description's
    numbers (`Options.matrix_ring` `prepared`) -- the operator carries them
    -- in the order the engine reads it, a table of the generated file
    (`prepared_table`) the block copies by bulk copies: once, where the
    stage is resident; a chunk at a time through the step's prepared ring
    (`ring_chunks`), where it is streamed.  The operator's own memory is
    not read for it."""

    tensor: object
    window: Optional[Tuple[Range, Range]] = None
    n0: int = 0
    depth: int = 0
    width: int = 0
    k: int = 0
    split: bool = False
    tiles: Tuple[Tuple[int, int], ...] = ()
    #: `fragments` (`mma_sync`, `dmma`) or `canonical` (`ASYNC_LOWERINGS`).
    order: str = 'fragments'
    #: Streamed through the step's ring rather than staged whole ahead of
    #: the loop (`RING_SLOTS`, `ring_volume`): nothing of it is in shared
    #: memory between the trips.
    streamed: bool = False
    #: Prepared at generation and bulk-copied (see above).
    prepared: bool = False

    @property
    def regs(self) -> int:
        """A lane's registers of B a fragment, both halves where split."""
        return (self.k // 4) * (2 if self.split else 1)

    @property
    def volume(self) -> int:
        if self.window is None:
            return int(self.tensor.storage_volume())
        if self.streamed:
            return 0
        if self.order == 'canonical':
            return self.depth * self.width * (2 if self.split else 1)
        return len(self.tiles) * self.k * 8 * (2 if self.split else 1)

    def tile(self, kt: int, nt: int) -> int:
        """Where B's tile `(kt, nt)` is stored, in tiles, or -1 where the
        stage leaves it out: a tile of zeros."""
        try:
            return self.tiles.index((kt, nt))
        except ValueError:
            return -1

    def canonical_offset(self, n: int, k: int) -> int:
        """Where `B(k, n)` is in the canonical K-major layout (`order`
        `canonical`): its core matrix's eight rows of N, its four of K,
        then its place in it, row by row -- one half's layout."""
        return (n // 8) * self.depth * 8 + (k // 4) * 32 + (n % 8) * 4 + k % 4

    @property
    def leading(self) -> int:
        """Bytes between core matrices next to each other along K, the
        descriptor's leading byte offset: one core matrix, 128."""
        return 128

    @property
    def stride(self) -> int:
        """Bytes between core matrices next to each other along N, the
        descriptor's stride byte offset: eight rows of the whole depth."""
        return self.depth * 32


@dataclass(frozen=True)
class Phase:
    """A phase of a chained section (`GroupSection.phases`): its mixing
    and operator steps into one output of the element (`main`, none in a
    phase of copies alone), and the scaled copies (`scales`) that read that
    output -- or, without `main`, what no step writes."""

    main: Tuple['Step', ...]
    scales: Tuple['Step', ...]


@dataclass(frozen=True)
class GroupSection:
    """A section as the multi-element layout runs it (`classify`): its
    steps in order, the operators staged once, the element's operands
    staged a trip (`inputs`), the intermediates, and the outputs."""

    steps: Tuple[Step, ...]
    operators: Tuple[object, ...]
    inputs: Tuple[Buffer, ...]
    intermediates: Tuple[Buffer, ...]
    outputs: Tuple[object, ...]
    dtype: Datatype
    #: The operators as the block stages them (`Stage`, `lower`).
    stages: Tuple[Stage, ...] = ()
    #: How many of the inputs, in the order a trip stages them, take a
    #: second stage: those `pir.wrap` may copy a trip ahead (`lower`).
    doubled: int = 0
    #: The scaled copies' factors, scalars passed by value (`Kind.SCALE`).
    scalars: Tuple[object, ...] = ()
    #: The slots of the ring a producer warp fills the inputs through, its
    #: trips ahead of the element warps' (`producer_refusal`); 0 where each
    #: element's warp stages its own element.
    ring: int = 0
    #: The registers a thread of the producer keeps where its warpgroup
    #: gives the rest to the element warps (`register_budgets`), 0 where
    #: every warp keeps what the kernel's entry gives it.
    registers: int = 0
    #: The prepared operator ring of the steps whose B streams prepared
    #: (`Stage.prepared`, `Options.matrix_ring_slots`, `matrix_ring_rows`):
    #: its slots and the rows of K a piece of a product holds; 0 where no
    #: stage streams prepared.
    ring_slots: int = 0
    ring_rows: int = 0
    #: How many chunks the products hold a slot of the prepared ring for
    #: (`Options.matrix_ring_hold`, `ring_hold`); 0 where each engine
    #: holds its own (`RING_HOLDS`).
    ring_hold: int = 0
    #: `engine` where the star step is folded into the asynchronous steps
    #: that read its intermediates as A (`Options.matrix_star`, `fuse`):
    #: those steps compute A themselves, a block of K of every product at a
    #: time, into the engine's operand memory (`Buffer.held`).
    star: str = ''
    #: Why the fill rule took mma_sync over the target's asynchronous
    #: engine, for the kernel's comment (`precision_notes`); '' where it
    #: did not.
    engine_note: str = ''
    #: `three` where the block runs the section in three roles of its warps
    #: (`Options.matrix_roles`, `roles_refusal`): the element warps
    #: staging and computing A, an operator producer behind them, and the
    #: warpgroup that issues and writes D behind that; '' where the element
    #: warps do everything.
    roles: str = ''

    @property
    def chained(self) -> bool:
        """Whether the section reads back an output it wrote, or copies one
        scaled: phases one after another (`phases`), each reading what the
        ones before it wrote -- SeisSol's derivative, an order a phase."""
        written = {id(s.dest) for s in self.steps if not s.dest.is_tmp}
        return any(s.kind is Kind.SCALE for s in self.steps) or any(
            id(t) in written for s in self.steps for t in s.reads())

    def phases(self) -> Tuple['Phase', ...]:
        """The section as its phases run (`Phase`): a phase of the scaled
        copies that come after no other step's write of what they read or
        write, first, where there are such; then each run of steps into an
        output of the element -- the mixing steps and the operator steps
        that read them -- with the scaled copies after it that read or add
        to what it, or a copy after it, wrote: SeisSol's Taylor sum of an
        order, a kernel's `Q += P` after the product into `Q`.  An output is
        whole when its phase ends: a later phase reads it from where it was
        written."""
        first: List['Step'] = []
        runs: List[List] = []
        run: List['Step'] = []
        # The phase that last wrote each tensor of the element: -1 the
        # copies' own, else a run's (`runs`).
        writer: Dict[int, int] = {}
        for step in self.steps:
            if step.kind is Kind.SCALE:
                at = max((writer.get(id(t), -1) for t in (
                    step.products[0].first.tensor, step.dest)), default=-1)
                (first if at < 0 else runs[at][1]).append(step)
                writer[id(step.dest)] = at
                continue
            run.append(step)
            if not step.dest.is_tmp:
                writer[id(step.dest)] = len(runs)
                runs.append([tuple(run), []])
                run = []
        if run:
            runs.append([tuple(run), []])
        phases = [Phase((), tuple(first))] if first else []
        return tuple(phases + [Phase(main, tuple(scales))
                               for main, scales in runs])

    def phase_section(self, phase: 'Phase') -> 'GroupSection':
        """The section of one phase's mixing and operator steps, into its
        one output."""
        return dataclasses.replace(self, steps=phase.main,
                                   outputs=(phase.main[-1].dest,))

    @property
    def products(self) -> List[Product]:
        return [p for s in self.steps for p in s.products]

    @property
    def pointer_tables(self) -> Tuple[object, ...]:
        """The outputs a matrix step adds into: the block keeps a table of
        the group's pointers to each, every warp its element's -- a tile's
        rows are several elements', and the lane that holds a cell of D adds
        it through its row's element's pointer."""
        return tuple(t for t in self.outputs
                     if any(s.dest is t and s.lowering in MATRIX_LOWERINGS
                            for s in self.steps))

    @property
    def basis(self) -> int:
        """The largest extent of the basis the group operands run over: what
        sizes the group (`ELEMENTS`)."""
        return max([s.rows[1] for s in self.steps]
                   + [b.rows[1] for b in self.inputs + self.intermediates
                      if b.role is not Role.SMALL])

    def buffer(self, tensor) -> Buffer:
        """The group's buffer of `tensor`, an input or an intermediate."""
        for b in self.inputs + self.intermediates:
            if b.tensor is tensor:
                return b
        raise KeyError(tensor.alias)

    def stage(self, tensor, window=None) -> Stage:
        """The staged form of an operator: as stored, or one window's B."""
        for st in self.stages:
            if st.tensor is tensor and st.window == window:
                return st
        raise KeyError((tensor.alias, window))

    def role(self, tensor) -> Role:
        if any(t is tensor for t in self.operators):
            return Role.OPERATOR
        if any(t is tensor for t in self.outputs):
            return Role.GROUP
        return self.buffer(tensor).role

    def shape(self) -> Tuple:
        """What the section's lowering depends on, in the description's
        terms: two classifications of one section agree on it."""
        return tuple((s.kind, s.lowering, s.dest.alias, s.assigns,
                      tuple((p.first.tensor.alias,
                             None if p.second is None
                             else p.second.tensor.alias,
                             p.rows, p.cols, p.depth) for p in s.products),
                      s.terms, s.output)
                     for s in self.steps)


@dataclass(frozen=True)
class ElementGroup:
    """How a section takes the multi-element layout: E elements a block,
    one warp each, and the section as it runs (`GroupSection`, each step
    with its lowering) -- or, in the lanes form (`lanes`, a `Lanes`), E
    elements on the lanes of a warp and `lanes.warps` warps that all take
    them, each a share of the output."""

    elements: int
    section: GroupSection
    shared_bytes: int
    lanes: Optional['Lanes'] = None

    @property
    def threads(self) -> int:
        """A warp's lanes: `WARP_LANES`, or in the lanes form the group's
        elements, a lane each (`lane_elements`)."""
        return self.elements if self.lanes is not None else WARP_LANES

    @property
    def producers(self) -> int:
        """The block's producer warps: one where the section's inputs come
        through a ring (`GroupSection.ring`), behind the element warps -- a
        warpgroup where it gives its registers to them, as `setmaxnreg`
        moves a warpgroup's (`GroupSection.registers`)."""
        if self.section.roles:
            # The operator producer's warp and the mma role's warpgroup
            # behind the element warps (`roles_refusal`).
            return OPERATOR_WARPS + MMA_WARPS
        if not self.section.ring:
            return 0
        return WARPGROUP_WARPS if self.section.registers else PRODUCER_WARPS

    @property
    def warps(self) -> int:
        """The block's warps: one an element and the producer's, or the lanes
        form's shares."""
        if self.lanes is not None:
            return self.lanes.warps
        return self.elements + self.producers


def _operations(descrs: Sequence) -> list:
    out = []
    for descr in descrs:
        ops = getattr(descr, 'operations', None)
        out.extend(ops() if ops is not None else [descr])
    return out


def _addressed(tensor) -> Optional[Role]:
    """The role a tensor's kind gives it before the products say which an
    element's own one plays: None for an element's own."""
    if tensor.is_tmp:
        return Role.INTERMEDIATE
    if tensor.addressing.kind is Addressing.NONE:
        return Role.OPERATOR
    return None


def _whole(sub) -> bool:
    """Whether a view is its tensor's whole box."""
    box = sub.tensor.bbox
    return (tuple(int(x) for x in sub.bbox.lower())
            == tuple(int(x) for x in box.lower())
            and tuple(int(x) for x in sub.bbox.upper())
            == tuple(int(x) for x in box.upper()))


def _scaled(op) -> Optional[Tuple[object, object]]:
    """A scaled copy -- `dest (+)= src`, or `dest (+)= src * s` with `s` a
    scalar -- as `(src, s)`, `s` None for a plain copy; else None."""
    target = [list(t) for t in op.target]
    permute = [list(p) for p in op.permute]
    if len(op.ops) == 1 and target == [[0, 1]] and permute == [[0, 1]]:
        return op.ops[0], None
    if (len(op.ops) == 2 and target == [[0, 1], []]
            and permute == [[0, 1], []]
            and len(op.ops[1].bbox.lower()) == 0):
        return op.ops[0], op.ops[1]
    return None


def classify(descrs: Sequence) -> Tuple[Optional[GroupSection], str]:
    """The section of `descrs` as the multi-element layout runs it, or None
    and why not -- every product classified, or none."""
    from tensorforge.generators.descriptions import MultilinearDescr
    ops = _operations(descrs)
    if not ops:
        return None, 'the section has no operations'
    dtype = None
    roles: Dict[int, Role] = {}

    def claim(tensor, role):
        known = roles.setdefault(id(tensor), role)
        if known is not role:
            return (f'{tensor.alias} is both {known.value} and '
                    f'{role.value}')
        return None

    products: List[Product] = []
    # Assignments of a window of an element's tensor: a chained section's
    # (SeisSol's derivatives, an order's rows of its buffer) -- what the
    # rest holds nothing reads -- and refused in any other.
    windows: List[str] = []
    for n, op in enumerate(ops):
        if not isinstance(op, MultilinearDescr):
            return None, (f'operation {n} is a {type(op).__name__}, which '
                          f'the layout does not lower yet')
        if op.guarded():
            return None, f'operation {n} is guarded'
        scaled = _scaled(op)
        if scaled is not None:
            first, scalar = scaled
            for sub in (first, op.dest):
                t = sub.tensor
                if len(sub.bbox.lower()) != 2:
                    return None, f'{t.alias} is no matrix'
                if t.is_tmp or _addressed(t) is not None:
                    return None, (f'operation {n} copies {first.tensor.alias}'
                                  f' into {op.dest.tensor.alias}: a scaled '
                                  f'copy is of an element\'s own matrices')
                if not t.is_dense():
                    return None, f'{t.alias} is stored sparse'
                if getattr(t, 'storage_order', None) is not None:
                    return None, f'{t.alias} is stored in an order of its own'
                if getattr(sub, 'sliced', False):
                    return None, f'{t.alias} is read or written as a slice'
                if dtype is None:
                    dtype = t.datatype
                elif t.datatype is not dtype:
                    return None, 'the operands are of more than one type'
            if scalar is not None and (
                    scalar.tensor.addressing is not Addressing.SCALAR
                    and not scalar.tensor.has_values()):
                return None, (f'operation {n} scales by '
                              f'{scalar.tensor.alias}, which is no scalar '
                              f'the kernel is handed or holds')
            rows, cols = _range(first, 0), _range(first, 1)
            dest_rows, dest_cols = _range(op.dest, 0), _range(op.dest, 1)
            if cols != dest_cols or not _covers(dest_rows, rows) or (
                    not op.add and rows != dest_rows):
                return None, (f'operation {n} copies a window of '
                              f'{first.tensor.alias} onto another of '
                              f'{op.dest.tensor.alias}')
            why = (claim(first.tensor, Role.GROUP)
                   or claim(op.dest.tensor, Role.GROUP)
                   or (scalar is not None
                       and claim(scalar.tensor, Role.SCALAR)))
            if why:
                return None, why
            if not op.add and not _whole(op.dest):
                windows.append(f'operation {n} assigns a window of '
                               f'{op.dest.tensor.alias}, whose rest it would '
                               f'have to zero')
            products.append(Product(op, Kind.SCALE, first, scalar, op.dest,
                                    op.add))
            continue
        if len(op.ops) != 2:
            return None, (f'operation {n} is not a product of two matrices '
                          f'(a scaled product, or another arity)')
        if ([list(t) for t in op.target] != [[0, -1], [-1, 1]]
                or [list(p) for p in op.permute] != [[0, 1], [0, 1]]):
            return None, (f'operation {n} is not a product of two matrices '
                          f'into a matrix in their order')
        first, second = op.ops
        for sub in (first, second, op.dest):
            t = sub.tensor
            if len(sub.bbox.lower()) != 2:
                return None, f'{t.alias} is no matrix'
            if getattr(t, 'storage_order', None) is not None:
                return None, f'{t.alias} is stored in an order of its own'
            if getattr(sub, 'sliced', False):
                return None, f'{t.alias} is read or written as a slice'
            if dtype is None:
                dtype = t.datatype
            elif t.datatype is not dtype:
                return None, 'the operands are of more than one type'
        if dtype not in (Datatype.F32, Datatype.F64):
            return None, f'{dtype} is not a floating type the layout takes'
        if _range(first, 1) != _range(second, 0):
            return None, f'operation {n} contracts over two extents'
        if (_range(first, 0) != _range(op.dest, 0)
                or _range(second, 1) != _range(op.dest, 1)):
            return None, (f'operation {n} writes another window than its '
                          f'operands span')
        a, b = _addressed(first.tensor), _addressed(second.tensor)
        if a is Role.OPERATOR and b is not Role.OPERATOR:
            kind = Kind.OPERATOR
            why = claim(first.tensor, Role.OPERATOR)
            if b is None:
                why = why or claim(second.tensor, Role.GROUP)
        elif a is not Role.OPERATOR and b is None:
            kind = Kind.MIXING
            extents = [hi - lo for lo, hi in (_range(second, 0),
                                              _range(second, 1))]
            if max(extents) > SMALL_EXTENT:
                return None, (f'{second.tensor.alias} is an element\'s own '
                              f'and too large for a small operand')
            why = claim(second.tensor, Role.SMALL)
            if a is None:
                why = why or claim(first.tensor, Role.GROUP)
        else:
            return None, (f'operation {n} multiplies {first.tensor.alias} '
                          f'and {second.tensor.alias}: neither an operator '
                          f'times a group operand nor a group operand times '
                          f'an element\'s small one')
        # Stored sparse, only an element's small operand: its stored cells
        # are read where they are, the rest are zeros (SeisSol's `star`
        # under its GPU memory layout, 24 of 81).  An operator or a group
        # operand stored so the layout does not read.
        for sub in (first, second, op.dest):
            if (not sub.tensor.is_dense()
                    and not (kind is Kind.MIXING and sub is second)):
                return None, f'{sub.tensor.alias} is stored sparse'
        if why:
            return None, why
        dest = op.dest.tensor
        if _addressed(dest) is Role.OPERATOR:
            return None, f'operation {n} writes the batch\'s {dest.alias}'
        if _addressed(dest) is None:
            why = claim(dest, Role.GROUP)
            if why:
                return None, why
            if not op.add and not _whole(op.dest):
                windows.append(f'operation {n} assigns a window of '
                               f'{dest.alias}, whose rest it would have to '
                               f'zero')
        products.append(Product(op, kind, first, second, op.dest, op.add))

    # An element's own tensor the section writes and reads: every write
    # ahead of every read, and read only where written -- an output read back
    # whole by a later phase (`GroupSection.phases`), the order before as
    # SeisSol's derivative reads it.
    writes: Dict[int, List[int]] = {}
    spans_of: Dict[int, List[Tuple[Range, Range]]] = {}
    for n, p in enumerate(products):
        if not p.dest.tensor.is_tmp:
            writes.setdefault(id(p.dest.tensor), []).append(n)
            spans_of.setdefault(id(p.dest.tensor), []).append(
                (_range(p.dest, 0), _range(p.dest, 1)))
    rereads = False
    for n, p in enumerate(products):
        for sub in p.operands:
            t = sub.tensor
            if t.is_tmp or id(t) not in writes:
                continue
            if max(writes[id(t)]) > n:
                return None, (f'{t.alias} is written and read in the '
                              f'section, read before its last write')
            rows, cols = _range(sub, 0), _range(sub, 1)
            if not any(_covers(r, rows) and _covers(c, cols)
                       for r, c in spans_of[id(t)]):
                return None, (f'operation {n} reads {t.alias} where '
                              f'nothing wrote it')
            rereads = True
    chained = rereads or any(p.kind is Kind.SCALE for p in products)
    if windows and not chained:
        return None, windows[0]
    # Intermediates: assigned before anything reads or adds to them, and
    # read only where written.
    assigned: Dict[int, List[Tuple[Range, Range]]] = {}
    for n, p in enumerate(products):
        for sub in p.operands:
            t = sub.tensor
            if not t.is_tmp:
                continue
            spans = assigned.get(id(t))
            if spans is None:
                return None, (f'operation {n} reads {t.alias} before '
                              f'anything writes it')
            rows, cols = _range(sub, 0), _range(sub, 1)
            if not any(_covers(r, rows) and _covers(c, cols)
                       for r, c in spans):
                return None, (f'operation {n} reads {t.alias} where '
                              f'nothing wrote it')
        t = p.dest.tensor
        if t.is_tmp:
            if p.add and id(t) not in assigned:
                return None, (f'operation {n} adds into {t.alias}, which '
                              f'nothing assigned')
            assigned.setdefault(id(t), []).append(
                (_range(p.dest, 0), _range(p.dest, 1)))
    return _section(_steps(products), dtype), ''


#: Each kind's lowering, so far.
LOWERING = {Kind.OPERATOR: 'fma', Kind.MIXING: 'rows', Kind.SCALE: 'scale'}


def _steps(products: Sequence[Product]) -> Tuple[Step, ...]:
    """The products as steps, in the order they run: a product joins the
    open step into its destination where it accumulates, is of the step's
    kind, and nothing since that step opened read its destination or
    wrote what it reads.  A step runs where it closes."""
    out: List[Step] = []
    open_steps: List[List[Product]] = []

    def close(members):
        open_steps.remove(members)
        out.append(Step(tuple(members), LOWERING[members[0].kind]))

    for p in products:
        reads = {id(sub.tensor) for sub in p.operands}
        dest = id(p.dest.tensor)
        for members in list(open_steps):
            read_by = {id(sub.tensor) for m in members for sub in m.operands}
            if id(members[0].dest.tensor) in reads or dest in read_by:
                close(members)
        into = [m for m in open_steps if id(m[0].dest.tensor) == dest]
        # A scaled copy is a step of its own: a chained section's phases take
        # each where what it copies is whole (`GroupSection.phases`).
        if (into and p.add and p.kind is into[0][0].kind
                and p.kind is not Kind.SCALE):
            into[0].append(p)
            continue
        for members in into:
            close(members)
        open_steps.append([p])
    for members in list(open_steps):
        close(members)
    return tuple(out)


def _section(steps: Tuple[Step, ...], dtype) -> GroupSection:
    operators, outputs, scalars = [], [], []
    boxes: Dict[int, Tuple[object, Role, List[Range], List[Range]]] = {}

    def keep(tensor, role, sub):
        entry = boxes.setdefault(id(tensor), (tensor, role, [], []))
        entry[2].append(_range(sub, 0))
        entry[3].append(_range(sub, 1))

    for p in (p for s in steps for p in s.products):
        if p.kind is Kind.SCALE:
            keep(p.first.tensor, Role.GROUP, p.first)
            if p.second is not None and not any(
                    t is p.second.tensor for t in scalars):
                scalars.append(p.second.tensor)
            if not any(o is p.dest.tensor for o in outputs):
                outputs.append(p.dest.tensor)
            continue
        for sub in (p.first, p.second):
            t = sub.tensor
            if t.is_tmp:
                keep(t, Role.INTERMEDIATE, sub)
            elif t.addressing.kind is Addressing.NONE:
                if not any(o is t for o in operators):
                    operators.append(t)
            else:
                small = p.kind is Kind.MIXING and sub is p.second
                keep(t, Role.SMALL if small else Role.GROUP, sub)
        t = p.dest.tensor
        if t.is_tmp:
            keep(t, Role.INTERMEDIATE, p.dest)
        elif not any(o is t for o in outputs):
            outputs.append(t)
    inputs, intermediates = [], []
    for tensor, role, rows, cols in boxes.values():
        buffer = Buffer(tensor, role, _union(rows), _union(cols))
        (intermediates if role is Role.INTERMEDIATE else inputs).append(
            buffer)
    section = GroupSection(steps, tuple(operators), tuple(inputs),
                           tuple(intermediates), tuple(outputs), dtype,
                           scalars=tuple(scalars))
    return _staged(section, 0)


#: The asynchronous matrix engines' lowerings, F32 as 3xTF32 (TF32 halves
#: of A and B, `TF32_TERMS`): wgmma, a warpgroup's on sm_90a, A from
#: registers by ldmatrix and D in them; tcgen05, one thread's on sm_100a, A
#: and D in tensor memory.  B is read through descriptors, in the canonical
#: K-major order (`Stage.order`).
ASYNC_LOWERINGS = ('wgmma', 'tcgen05')

MATRIX_LOWERINGS = ('mma_sync', 'dmma') + ASYNC_LOWERINGS

#: A warpgroup's warps: a wgmma step's group is whole warpgroups of them,
#: a warp an element (`engine_elements`).
WARPGROUP_WARPS = 4

#: tcgen05's rows, its M at one CTA: the group's rows of a step, elements
#: by the destination's columns, at most these (`engine_elements`).
TCGEN05_ROWS = 128

#: An instruction's N at most, for either engine: a step's rows past it
#: stay on mma_sync (`_async_engine`).
ASYNC_N = 256

#: The ring an asynchronous step streams B through where its stages do not
#: fit a block whole (`Stage.streamed`): slots of `RING_CHUNK` rows of K
#: each -- a chunk the block converts into the canonical order while the
#: engine multiplies the last -- `RING_SLOTS` of them.
RING_SLOTS = 2
RING_CHUNK = 32


def matrix_shape(lowering: str, sm: int) -> Tuple[int, int, int]:
    """`(m, n, k)` of the instruction a matrix step issues: `m16n8k8` over
    TF32 halves for F32, and for F64 `m16n8k4` from sm_90 on, `m8n8k4`
    before; wgmma `m64nNk8` and tcgen05 `m128nNk8`, N the step's rows
    rounded to the `n` given here, 8 and 16."""
    if lowering == 'mma_sync':
        return 16, 8, 8
    if lowering == 'wgmma':
        return 64, 8, 8
    if lowering == 'tcgen05':
        return TCGEN05_ROWS, 16, 8
    return (16, 8, 4) if sm >= 90 else (8, 8, 4)


def _round(value: int, step: int) -> int:
    return -(-value // step) * step


def _staged(section: GroupSection, sm: int, split: bool = False,
            sparse: bool = True, aligned: bool = False,
            streamed: bool = False, prepared: bool = False) -> GroupSection:
    """The section with its operators' stages, for the steps' lowerings:
    an operator as stored where an FMA form reads it, a B a window where a
    matrix step does -- in TF32 halves where `split` asks, its tiles of
    zeros left out where `sparse` lets the operator's pattern say which
    (`kept_tiles`), its N grid from the step's first row or, `aligned`,
    from that rounded down to a tile; an asynchronous step's halves
    prepared at generation where `prepared` asks and every operator of the
    step carries its numbers (`Stage.prepared`, `prepared_refusal`)."""
    stages: List[Stage] = []

    def add(stage):
        # By the tensor's identity: two operators alike in shape compare
        # equal as tensors.
        key = (id(stage.tensor), stage.window, stage.n0)
        if not any((id(s.tensor), s.window, s.n0) == key for s in stages):
            stages.append(stage)

    for step in section.steps:
        matrix = step.lowering in MATRIX_LOWERINGS
        if step.lowering == 'lane':
            # The lanes form stages nothing: its operators are literals or
            # read where they are stored, and each lane reads its own
            # element where it is stored.
            continue
        # A step's B prepared where every operator of it carries its
        # numbers: one ring a step, of prepared chunks or converted ones.
        ready = (prepared and split and step.lowering in ASYNC_LOWERINGS
                 and step_refusal(step) is None)
        for p in step.products:
            t = p.first.tensor if p.kind is Kind.OPERATOR else None
            if t is None:
                continue
            if not matrix:
                # A step that walks its operators' nonzeros reads those,
                # packed by the block (`Terms`), and not the operators.
                if step.terms is None:
                    add(Stage(t))
                continue
            _, n, k = matrix_shape(step.lowering, sm)
            asynchronous = step.lowering in ASYNC_LOWERINGS
            # An asynchronous engine's B is whole, from the step's first
            # row: its descriptors walk every tile (`Stage`).
            n0 = step.rows[0] - (step.rows[0] % n
                                 if aligned and not asynchronous else 0)
            depth = _round(p.depth[1] - p.depth[0], k)
            width = _round(step.rows[1] - n0, n)
            add(Stage(t, (p.rows, p.depth), n0, depth, width, k,
                      split and step.lowering in ('mma_sync',)
                      + ASYNC_LOWERINGS,
                      tuple(sorted(kept_tiles(
                          p, n0, k, depth, width,
                          sparse and not asynchronous))),
                      'canonical' if asynchronous else 'fragments',
                      streamed and asynchronous, ready))
    return dataclasses.replace(section, stages=tuple(stages))


def step_refusal(step: Step) -> Optional[str]:
    """Why an asynchronous step's B cannot be prepared at generation
    (`Stage.prepared`), or None: an operator of it that carries no numbers
    -- the halves are made from the description's, not read from the
    operator's memory."""
    for p in step.products:
        tensor = p.first.tensor
        if not tensor.has_values():
            return f'{tensor.alias} carries no numbers'
    return None


def kept_tiles(product: Product, n0: int, k: int, depth: int, width: int,
               sparse: bool = True) -> Set[Tuple[int, int]]:
    """The tiles of a matrix product's B, `(kt, nt)` -- `k` x 8, on the
    grid from the window's first column and from `n0` -- where its
    operator has an entry inside the product's window
    (`Tensor.nonzero_pattern`): all of them where `sparse` is off
    (`Options.skip_known_zeros`) or the pattern says nothing."""
    tiles = {(kt, nt) for kt in range(depth // k) for nt in range(width // 8)}
    pattern = getattr(product.first.tensor, 'nonzero_pattern', None)
    if not sparse or pattern is None:
        return tiles
    (r0, r1), (k0, k1) = product.rows, product.depth
    return {((kk - k0) // k, (i - n0) // 8) for i in range(r0, r1)
            for kk in range(k0, k1) if pattern.is_nz((i, kk))}


#: What the walk's balance weighs a unit by (`tile_units`): an MMA over TF32
#: halves is three, an A tile read for a k-block and split two, a B tile one.
TILE_COSTS = {'mma_sync': (3, 2, 1), 'dmma': (1, 1, 1)}

#: The accumulator registers a lane gives a group's tiles of D at most
#: (`tile_units`): the slots times the M tiles walked together, times a
#: tile's registers.
TILE_REGISTERS = 32

#: What a unit costs a warp besides its tiles (`tile_units`), in the units
#: of `TILE_COSTS`: reading it from the tables and starting its walk.
TILE_UNIT = 4

#: The most N tiles a warp holds for a unit (`tile_units`), or None for as
#: many as pay.  A second slot keeps A over the slots, but each slot's
#: tiles then stand under a test of their own, which on sm_120 cost more
#: than the A reads they save (o6 F32: 6.55 ms with three slots, 4.50 ms
#: with one).
TILE_SLOTS: Optional[int] = 1

#: The least an entry of the walk takes (`tile_units`), in the units of
#: `TILE_COSTS`: its A and B in flight and a tile's three products one
#: after another, which the M tiles walked together hide.
TILE_LATENCY = 12

#: Into how many parts a set's entries may go (`tile_units`), each its own
#: unit, where the step adds into its output: the heaviest N tiles' walks
#: shared by two or more warps.  Each part adds its D on its own, a
#: reduction a cell, which on sm_120 cost more than the balance gained
#: (o6 F32: 4.98 ms split four ways, 4.47 ms whole).
TILE_SPLIT = 1


@dataclass(frozen=True)
class TileUnits:
    """A matrix step's tiles of D over the block's warps (`tile_units`).

    A unit is a range of M tiles, `(first, last)`, and the N tiles of the
    `slots` a warp holds for it, -1 in a slot left empty.  `warps[w]` are
    the units warp `w` takes, in turn, and a warp walks a unit's M tiles
    `group` at a time: D of the group's M tiles and the slots' N tiles in
    registers, A read once an M tile and k-block and kept over the slots,
    B once a slot and k-block and kept over the M tiles.  `entries`, by
    unit and product, are what the walk reads: a k-block of the product's
    B and, a slot each, its stored tile there (`Stage.tile`) -- -1 for a
    tile of zeros -- only the k-blocks where a slot has one."""

    slots: int
    group: int
    units: Tuple[Tuple[int, int, Tuple[int, ...]], ...]
    warps: Tuple[Tuple[int, ...], ...]
    entries: Tuple[Tuple[Tuple[Tuple[int, Tuple[int, ...]], ...], ...], ...]


def tile_units(section: GroupSection, step: Step, warps: int,
               sm: int) -> TileUnits:
    """How a matrix step's tiles of D go to the block's `warps`, fixed at
    generation (`TileUnits`): the N tiles in sets of `slots`, each set a
    warp's, the heaviest tiles spread first; the M tiles in parts; where
    the step adds into its output, a set's entries -- its kept k-blocks,
    product after product -- in parts as well, each part's D added on its
    own (split K); the units -- a set, an M part and a part of its entries
    -- to the warps the costliest first (LPT), by their MMAs, the A tiles
    they read and the B tiles (`TILE_COSTS`), an entry at least its
    latency (`TILE_LATENCY`).  Of every form, the one whose busiest warp
    walks the fewest entries, then costs least."""
    m_, n_, _ = matrix_shape(step.lowering, sm)
    mtiles = -(-(warps * (step.cols[1] - step.cols[0])) // m_)
    stages = [section.stage(p.first.tensor, (p.rows, p.depth))
              for p in step.products]
    ntiles = stages[0].width // n_
    blocks = [range(st.depth // st.k) for st in stages]
    mma, a_read, b_read = TILE_COSTS[step.lowering]
    regs = m_ * n_ // 32 * (2 if section.dtype is Datatype.F64 else 1)
    # Two warps may hold one tile of D only where each adds its share into
    # the output (`red.global.add`): a store or an intermediate's
    # read-modify-write would race.
    split = not step.dest.is_tmp and not step.assigns

    def walk(nset):
        """A set's entries, `(product, kt)`, where a slot keeps a tile."""
        return [(p, kt) for p, (st, ks) in enumerate(zip(stages, blocks))
                for kt in ks if any(st.tile(kt, nt) >= 0 for nt in nset)]

    def tiles(nset, part):
        return sum(stages[p].tile(kt, nt) >= 0 for p, kt in part
                   for nt in nset)

    def cost(lo, hi, nset, part, group):
        # Each M group walks the part's entries: at least the latency of
        # one, which the group's M tiles share, else its MMAs; its A tiles
        # and its B tiles besides.
        held, total = tiles(nset, part), 0
        for first in range(lo, hi, group):
            m = min(group, hi - first)
            total += (max(TILE_LATENCY * len(part), mma * m * held)
                      + a_read * m * len(part) + b_read * held)
        return total

    best = None
    for slots in range(1, min(ntiles, TILE_SLOTS or ntiles) + 1):
        sets: List[List[int]] = [[] for _ in range(-(-ntiles // slots))]
        weight = [len(walk([nt])) for nt in range(ntiles)]
        for nt in sorted(range(ntiles), key=lambda nt: (-weight[nt], nt)):
            min((s for s in sets if len(s) < slots),
                key=lambda s: (sum(weight[x] for x in s), len(s))).append(nt)
        # As many M parts as the warps take whole sets of, and finer, down
        # to an M tile a part: more units than warps let the largest-first
        # assignment even the warps out.
        for count in range(max(1, min(mtiles, warps // len(sets))),
                           mtiles + 1):
            bounds = [mtiles * i // count for i in range(count + 1)]
            parts = [(lo, hi) for lo, hi in zip(bounds, bounds[1:])
                     if lo < hi]
            group = max(1, min(max(hi - lo for lo, hi in parts),
                               TILE_REGISTERS // (slots * regs)))
            for pieces in range(1, (TILE_SPLIT if split else 1) + 1):
                units = []
                for nset in sets:
                    entries = walk(nset)
                    cuts = min(pieces, len(entries)) or 1
                    for c in range(cuts):
                        part = entries[len(entries) * c // cuts:
                                       len(entries) * (c + 1) // cuts]
                        units += [(lo, hi, tuple(sorted(nset)) + (-1,) * (
                            slots - len(nset)), tuple(part))
                            for lo, hi in parts]
                costs = [cost(lo, hi, [nt for nt in nts if nt >= 0], part,
                              group) + TILE_UNIT
                         for lo, hi, nts, part in units]
                load, assign = [0] * warps, [[] for _ in range(warps)]
                for u in sorted(range(len(units)),
                                key=lambda u: (-costs[u], u)):
                    w = min(range(warps), key=lambda w: (load[w], w))
                    load[w] += costs[u]
                    assign[w].append(u)
                # The walks a warp takes -- an entry for each M group of
                # its units -- first: each waits on its own loads and
                # products, where the group's M tiles share the wait.
                walks = max(sum(-(-(units[u][1] - units[u][0]) // group)
                                * len(units[u][3]) for u in taken)
                            for taken in assign)
                if best is None or (walks, max(load)) < best[0]:
                    best = ((walks, max(load)), slots, group, units, assign)
    _, slots, group, units, assign = best
    entries = tuple(
        tuple(tuple((kt, tuple(st.tile(kt, nt) if nt >= 0 else -1
                               for nt in nset))
                    for q, kt in part if q == p)
              for p, st in enumerate(stages))
        for _, _, nset, part in units)
    return TileUnits(slots, group,
                     tuple((lo, hi, nset) for lo, hi, nset, _ in units),
                     tuple(tuple(a) for a in assign), entries)


def _pitched(section: GroupSection) -> GroupSection:
    """The section with a pitch for every buffer an F32 matrix step reads A
    from: ldmatrix takes a fragment's rows 16 bytes at a time
    (`compute.multi_element`), so a column of the buffer -- a row of A --
    starts on a multiple of four floats, and an odd number of 16-byte units
    from the next (four more than a multiple of eight floats), which puts a
    fragment's eight rows on eight different bank groups.  Long enough for
    every step of K a product reads, the padding zero, so that the padded
    end of the contraction multiplies zeros.  The same for the group operand
    a `mix` step reads 16 bytes of a column at a time, its rows in whole
    units of them."""
    need: Dict[int, int] = {}
    unit = 16 // section.dtype.size()
    for step in section.steps:
        if step.lowering in ('mix', 'fused'):
            buffer = section.buffer(step.products[0].first.tensor)
            need[id(buffer.tensor)] = max(
                need.get(id(buffer.tensor), buffer.height),
                _round(buffer.height, unit))
            continue
        # wgmma's A is mma_sync's, by ldmatrix; tcgen05's is read a row a
        # thread, and wants the padding zero all the same.
        if step.lowering not in ('mma_sync',) + ASYNC_LOWERINGS:
            continue
        _, _, k = matrix_shape(step.lowering, 0)
        for p in step.products:
            buffer = section.buffer(p.second.tensor)
            span = (p.depth[0] - buffer.rows[0]) + _round(
                p.depth[1] - p.depth[0], k)
            need[id(buffer.tensor)] = max(
                need.get(id(buffer.tensor), buffer.height), span)

    def pitched(buffer):
        n = need.get(id(buffer.tensor))
        if n is None:
            return buffer
        # The least pitch from `n` on that is an odd number of 16 bytes.
        n = _round(n, unit)
        return dataclasses.replace(
            buffer, stride=n + (unit if (n // unit) % 2 == 0 else 0))

    return dataclasses.replace(
        section, inputs=tuple(map(pitched, section.inputs)),
        intermediates=tuple(map(pitched, section.intermediates)))


#: The lowerings that read A in TF32 halves where the star step writes
#: them (`halvable`): mma_sync's tiling and wgmma by ldmatrix, tcgen05's
#: store of A into tensor memory a row a thread.
HALF_READERS = ('mma_sync',) + ('wgmma', 'tcgen05')


def halvable(section: GroupSection) -> List[Buffer]:
    """The buffers the 3xTF32 tiling can read A from in TF32 halves, as the
    step that writes them splits it (`Buffer.halves`): F32 intermediates
    that only `mix` steps write and only matrix steps read, as A
    (`HALF_READERS`: by ldmatrix, or a row a thread into tcgen05's tensor
    memory; pitched, `_pitched`).  An input keeps the split at its use:
    its copies cannot split.  So do F64, the FMA forms and the lanes form,
    which read no halves.  The buffers whose readers keep the most tiles of
    B first: a reader reads A once an M tile and k-block it keeps a tile
    at, and each such read is a split the halves spare."""
    if section.dtype is not Datatype.F32:
        return []
    reads: Dict[int, int] = {}
    for step in section.steps:
        for p in step.products:
            for sub in p.operands:
                t = sub.tensor
                if not t.is_tmp:
                    continue
                if (step.lowering not in HALF_READERS or sub is not p.second
                        or p.first.tensor is t):
                    reads[id(t)] = -1
                elif reads.get(id(t), 0) >= 0:
                    reads[id(t)] = reads.get(id(t), 0) + len(
                        section.stage(p.first.tensor,
                                      (p.rows, p.depth)).tiles)
    writers: Dict[int, Set[str]] = {}
    for step in section.steps:
        writers.setdefault(id(step.dest), set()).add(step.lowering)
    found = [(-reads[id(b.tensor)], n, b)
             for n, b in enumerate(section.intermediates)
             if b.stride and reads.get(id(b.tensor), -1) > 0
             and writers.get(id(b.tensor)) == {'mix'}]
    return [b for _, _, b in sorted(found, key=lambda x: x[:2])]


def _halved(section: GroupSection, count: int) -> GroupSection:
    """The section with A in TF32 halves in the first `count` buffers that
    can hold it so (`halvable`): their `mix` steps write both halves of
    every value, and the tiling reads both and splits nothing.  The halves
    take twice the buffer's room, which `fill` weighs."""
    chosen = {id(b.tensor) for b in halvable(section)[:count]}
    if not chosen:
        return section
    return dataclasses.replace(section, intermediates=tuple(
        dataclasses.replace(b, halves=True) if id(b.tensor) in chosen else b
        for b in section.intermediates))


def _halves_order(lowerings: Sequence[GroupSection]) -> List[GroupSection]:
    """`fill`'s lowerings again with A in TF32 halves (`_halved`), in its
    order of preference: the inputs' second stage first, and within each
    stage, a run of lowerings at a time, as many of A's buffers in halves as
    the run's sections hold first, each count over the run in its order --
    a section whose star step is folded (`fuse`) holds A in no buffer and
    keeps its place between the runs.  So where the fold is refused for one
    of B's tilings (`fuse_refusal`: B streamed through the converting
    ring), that tiling takes the halves in its place, behind the folded ones
    ahead of it and ahead of those behind it.  On the asynchronous engines
    a run's tilings of B in halves come before those of B whole, whatever
    the count: B whole is B in TF32 there (`multi_element._terms`), and A's
    halves spare work, not precision."""
    out: List[GroupSection] = []
    for doubled in sorted({t.doubled for t in lowerings}, reverse=True):
        run: List[GroupSection] = []

        def flush():
            parts = ([[t for t in run if split(t)],
                      [t for t in run if not split(t)]]
                     if run and asynchronous(run[0]) else [list(run)])
            for part in parts:
                most = max((len(halvable(t)) for t in part), default=0)
                out.extend(_halved(t, n) for n in range(most, -1, -1)
                           for t in part)
            run.clear()
        for t in lowerings:
            if t.doubled != doubled:
                continue
            if t.star:
                flush()
                out.append(t)
            else:
                run.append(t)
        flush()
    return out


#: A's two halves by B whole, B read in TF32 alone: what an asynchronous
#: engine multiplies where B is staged unsplit (`async_plan`), which only
#: `Options.matrix_tf32_operands` lets a plan do (`tf32_refusal`).
TF32_HALVES_BY_WHOLE = terms.plan('2xtf32', terms.Form(Datatype.TF32, 2),
                                  terms.Form(Datatype.TF32, 1),
                                  order='largest')


def async_plan(stage):
    """The emulated product a step of K of an asynchronous engine (wgmma,
    tcgen05) issues for an F32 product (`terms.TermPlan`): 3xTF32
    (`nvidia.TF32_PLAN`) where B is staged in its halves; A's two halves by
    B's one where B is staged whole (`TF32_HALVES_BY_WHOLE`) -- B read in
    TF32 alone, which only `Options.matrix_tf32_operands` lets a plan do
    (`tf32_refusal`).  A is always in both halves: split where it is read,
    or written so.  One accumulator, D."""
    from tensorforge.backend.instructions.compute.primitives import nvidia
    return nvidia.TF32_PLAN if stage.split else TF32_HALVES_BY_WHOLE


def async_terms(stage) -> Tuple[Tuple[int, int], ...]:
    """The products of `async_plan`, as `(A's half, B's half)`."""
    return async_plan(stage).products


def tf32_operands(context) -> bool:
    """Whether the options let an F32 plan read an operand in TF32 alone
    (`Options.matrix_tf32_operands`)."""
    return bool(getattr(context.get_user_options(), 'matrix_tf32_operands',
                        False))


def tf32_refusal(section: GroupSection, context) -> Optional[str]:
    """Why a plan of an F32 section is refused for reading an operand in
    TF32 alone -- an asynchronous step whose products are not 3xTF32's
    three (`async_terms`), B staged whole -- or None: the plan keeps F32's
    accuracy, the section is not F32, or the options let it trade
    (`Options.matrix_tf32_operands`).  The fill rule takes no plan this
    refuses, and the generator checks the plan it lowers again: a change
    of the fill cannot bring the trade back without the option."""
    from tensorforge.backend.instructions.compute.primitives import nvidia
    if section.dtype is not Datatype.F32 or tf32_operands(context):
        return None
    full = set(nvidia.TF32_TERMS)
    for step in section.steps:
        if step.lowering not in ASYNC_LOWERINGS:
            continue
        for p in step.products:
            if p.kind is not Kind.OPERATOR:
                continue
            stage = section.stage(p.first.tensor, (p.rows, p.depth))
            if set(async_terms(stage)) != full:
                return (f'{step.lowering} would read {p.first.tensor.alias} '
                        f'in TF32 alone, two of 3xTF32\'s three products '
                        f'(about 2^-11 relative a product against 2^-22): '
                        f'it needs Options.matrix_tf32_operands')
    return None


def prefer_mma_sync(asynchronous_elements: int, mma_elements: int,
                    cap: int) -> bool:
    """Whether the fill rule takes mma_sync's group over the asynchronous
    engine's, both keeping F32's accuracy: where mma_sync holds the larger
    group -- counted to at most `cap`, the most the asynchronous engine
    takes of the section at all (`engine_elements`: tcgen05's 128 lanes,
    wgmma's whole warpgroups), so that its lane limit alone does not give
    a section to mma_sync.  A rule of thumb standing in for the cost
    model, which is to replace this function: what the asynchronous
    engine at a smaller group costs against mma_sync at a larger one is
    measured nowhere yet (11-21)."""
    return min(mma_elements, cap) > asynchronous_elements


def matrix_enabled(context) -> bool:
    """Whether operator steps are tiled on the matrix engines:
    `Options.tensor_cores` where it is set, `MATRIX` where it is not."""
    asked = context.get_user_options().tensor_cores
    return MATRIX if asked is None else bool(asked)


def _cell(tensor, row: int, col: int) -> int:
    """The storage index of `(row, col)` of a dense operator stored column
    by column over its bounding box -- what `compute.multi_element` reads
    it at."""
    lower = [int(x) for x in tensor.bbox.lower()]
    return (row - lower[0]) + (col - lower[1]) * int(tensor.bbox.sizes()[0])


def nonzero_terms(step: Step) -> Optional[Terms]:
    """An operator step's products by their operators' nonzeros (`Terms`),
    or None where an operator does not carry its numbers.

    The pattern is the operator's own (`Tensor.nonzero_pattern`: its
    numbers, intersected with what it stores) within the product's window;
    the numbers a kernel multiplies by stay the ones in the operator's
    memory.  The rows go in the order of their nonzeros over all the step's
    products, the most first, ties by the row."""
    per = []
    for p in step.products:
        tensor = p.first.tensor
        if not tensor.has_values():
            return None
        pattern = tensor.nonzero_pattern
        (r0, r1), (k0, k1) = p.rows, p.depth
        per.append({i: [k for k in range(k0, k1) if pattern.is_nz((i, k))]
                    for i in range(r0, r1)})
    lo, hi = step.rows
    count = {i: sum(len(rows.get(i, ())) for rows in per)
             for i in range(lo, hi)}
    order = tuple(sorted((i for i in range(lo, hi)
                          if step.assigns or count[i]),
                         key=lambda i: (-count[i], i)))
    return Terms(
        order,
        tuple(tuple(len(rows.get(i, ())) for i in order) for rows in per),
        tuple(tuple(k for i in order for k in rows.get(i, ()))
              for rows in per),
        tuple(tuple(_cell(p.first.tensor, i, k) for i in order
                    for k in rows.get(i, ()))
              for p, rows in zip(step.products, per)))


def _async_engine(step: Step, context, engine: Optional[str]) -> Optional[str]:
    """The asynchronous matrix engine an F32 operator step takes on the
    context's target, or None: where the choice is the target's
    (`Options.matrix_engine` `auto`, unless `engine` holds it), wgmma on
    sm_90a and tcgen05 on sm_100a (`Target.wgmma`, `Target.tcgen05`).  An
    instruction's N is at most `ASYNC_N`, so the step's rows are; and
    tcgen05's rows hold a group's columns of the step, so that the
    smallest group, `MIN_ELEMENTS` of them, has to fit `TCGEN05_ROWS`.
    The group's own size is the fill rule's (`engine_elements`)."""
    choice = engine or context.get_user_options().matrix_engine
    if choice != 'auto':
        return None
    target = context.target
    if step.rows[1] - step.rows[0] > ASYNC_N:
        return None
    if target.wgmma():
        return 'wgmma'
    width = step.cols[1] - step.cols[0]
    if target.tcgen05() and MIN_ELEMENTS * width <= TCGEN05_ROWS:
        return 'tcgen05'
    return None


def _lowering(step: Step, dtype, context, sm: int, matrix: bool,
              engine: Optional[str] = None) -> str:
    """A step's lowering on the context's target: an operator step on the
    matrix engines where they are on (`matrix`, `matrix_enabled`) and the
    target has them (sm_80 on) and its products share the step's columns --
    an element's rows, which the tiling takes whole -- its kind's FMA form
    elsewhere.  F32 goes to the target's asynchronous engine where it has
    one and the step fits it (`_async_engine`), to mma_sync elsewhere and
    where `engine` holds it there."""
    if step.kind is not Kind.OPERATOR:
        return LOWERING[step.kind]
    if (not matrix or not matrix_enabled(context) or sm < 80
            or any(p.cols != step.cols for p in step.products)):
        return LOWERING[step.kind]
    if dtype is not Datatype.F32:
        return 'dmma'
    return _async_engine(step, context, engine) or 'mma_sync'


def engine_elements(section: GroupSection, elements: int) -> Optional[int]:
    """The largest group of at most `elements` the section's asynchronous
    steps take, or None where none of at least `MIN_ELEMENTS` does: wgmma
    a whole number of warpgroups -- a warp an element, `WARPGROUP_WARPS` of
    them a warpgroup -- since its tiles are a warpgroup's, every warpgroup
    taking them in the same trips; tcgen05 the group's rows of each step,
    its elements by the destination's columns, on its `TCGEN05_ROWS` lanes,
    which four warps hold.  A section without them takes any group."""
    for step in section.steps:
        if step.lowering == 'wgmma':
            elements -= elements % WARPGROUP_WARPS
        elif step.lowering == 'tcgen05':
            elements = min(elements,
                           TCGEN05_ROWS // (step.cols[1] - step.cols[0]))
    return elements if elements >= MIN_ELEMENTS else None


def lower(section: GroupSection, context, matrix: bool = True,
          split: bool = False, lanes: bool = False,
          engine: Optional[str] = None, streamed: bool = False,
          double: bool = True, ring: int = 0,
          registers: int = 0, halves: int = 0,
          prepared: bool = False, star: str = '',
          output: str = 'red', roles: str = '') -> GroupSection:
    """The section with each step's lowering for the context's target --
    the FMA forms only, where `matrix` is off; the lanes form's `lane` for
    every step, where `lanes` is on -- and the stages those lowerings read,
    B in TF32 halves where `split` asks, B streamed through a ring where
    `streamed` does, the inputs a trip ahead in a second stage where the
    options take them there (`second_stages`) and `double` does not
    decline it, and A in TF32 halves in as many as `halves` of the
    buffers that can hold it so (`_halved`).  An `fma` step reads only its
    operators' nonzeros (`nonzero_terms`) where they carry their numbers
    and `Options.skip_known_zeros` is on, as the per-element products do.
    `engine` holds F32 to mma_sync where it says so (`_lowering`).  `ring`
    slots of a ring a producer warp fills the inputs through, in place of
    the second stage, where the section takes one (`producer_refusal`).
    `prepared`: an asynchronous step's B halves prepared at generation
    where its operators carry their numbers (`Stage.prepared`), streamed
    through the prepared ring of the options' slots and rows where
    `streamed` asks.  `star` `engine`: the star step folded into each
    asynchronous step that can take it (`fuse`).  `output` `bulk` or `box`:
    D added into the output by bulk reductions where a matrix step can
    (`output_refusal`).  `roles` `three`: the block in the three roles
    where the section takes them (`roles_refusal`).
    The matrix tiling is NVIDIA's: elsewhere every step takes an FMA form,
    as below sm_80."""
    from tensorforge.backend.instructions.compute.primitives import nvidia
    sm = (nvidia.sm_of(context)
          if getattr(context.target.hw, 'vendor', None) == 'nvidia' else 0)
    sparse = context.get_user_options().skip_known_zeros
    steps = []
    for s in section.steps:
        lowering = ('lane' if lanes
                    else _lowering(s, section.dtype, context, sm, matrix,
                                   engine))
        steps.append(dataclasses.replace(
            s, lowering=lowering,
            terms=nonzero_terms(s) if sparse and lowering == 'fma'
            else None))
    if any(s.lowering in MATRIX_LOWERINGS for s in steps):
        # The matrix tiling's mixing steps block-wide, a thread a row of A.
        steps = [dataclasses.replace(s, lowering='mix')
                 if _mixes(s, section) else s for s in steps]
    staged = _staged(dataclasses.replace(section, steps=tuple(steps)), sm,
                     split, sparse, streamed=streamed, prepared=prepared)
    if star == 'engine':
        staged = fuse(staged)
    if output in ('bulk', 'box'):
        staged = dataclasses.replace(staged, steps=tuple(
            dataclasses.replace(s, output=output)
            if output_refusal(staged, s, context) is None else s
            for s in staged.steps))
    if any(st.prepared and st.streamed for st in staged.stages):
        options = context.get_user_options()
        staged = dataclasses.replace(
            staged, ring_slots=int(options.matrix_ring_slots),
            ring_rows=int(options.matrix_ring_rows))
        staged = _held(staged, context)
    pitched = _pitched(staged)
    if roles and not lanes:
        roled = dataclasses.replace(pitched, roles=roles, doubled=0)
        if roles_refusal(roled, context) is None:
            return _held(roled, context)
    if ring and not lanes:
        ringed = _bulk_pitched(pitched)
        if producer_refusal(ringed, context) is None:
            return dataclasses.replace(
                ringed, ring=int(ring), doubled=0,
                registers=int(registers)
                if context.target.register_budgets() else 0)
    return _halved(dataclasses.replace(
        pitched, doubled=second_stages(staged, context)
        if double and not lanes else 0), halves)


def async_staging(context) -> bool:
    """Whether a trip stages its dense operands by asynchronous copies,
    which `pir.wrap` can take a trip ahead: NVIDIA's `cp.async`
    (`Target.async_copy_path`).  AMD's LDS-DMA writes a wave's lanes past
    one base, which a box over its pitch is not to the emitter; there, as
    where there is no such path, a trip stages by loads and stores."""
    return (getattr(context.target.hw, 'vendor', None) == 'nvidia'
            and context.target.async_copy_path())


#: The warps a block's producer takes: one, whose first lane issues and
#: whose lanes take an element each (`compute.multi_element`).
PRODUCER_WARPS = 1

#: The slots of a producer's ring the fill rule tries, the deepest first.
PRODUCER_SLOTS = (3, 2)

#: Registers of the SM a block holds where one block is resident, which a
#: kernel's warpgroups share (`register_budgets`).
SM_REGISTERS = 65536


def register_budgets(group: 'ElementGroup') -> Optional[Tuple[int, int,
                                                               int]]:
    """`(entry, producer, elements)`: the registers a thread holds at the
    kernel's entry -- the block's warps sharing the SM's alike -- and from
    the moves on, a producer's and an element warp's, where the section
    moves them (`GroupSection.registers`): the producer's warpgroup gives
    what it does not keep to the element warps' (`setmaxnreg`, a multiple
    of eight from 24 to 256).  None where nothing moves, or where the
    element warps would gain nothing.

    What the moves draw on is the block's own allocation, `entry`
    registers a thread of it, and not the SM's: the pool `setmaxnreg`
    moves registers within is the block's and zero-sum, and an `inc` past
    it waits forever.  So the element warps share what the block holds at
    its entry less what the producer keeps -- the roles' rule
    (`pir.roles.register_budgets`), the producer and the element warps
    two roles of the block."""
    from tensorforge.backend.pir.roles import register_budgets as moves
    keep = group.section.registers
    if not keep or not group.section.ring:
        return None
    found = moves((('producer', 32 * group.producers),
                   ('elements', 32 * group.elements)), {'producer': keep},
                  registers=SM_REGISTERS)
    if found is None:
        return None
    entry, budgets = found
    return entry, budgets['producer'], budgets['elements']


#: The warps of the three roles' operator producer, and of their mma role
#: on tcgen05: a warpgroup, whose four warps reach the four quarters of
#: tensor memory's lanes (`Options.matrix_roles`).
OPERATOR_WARPS = 1
MMA_WARPS = 4


def roles_asked(context) -> str:
    """The roles the options ask the block in (`Options.matrix_roles`)."""
    return context.get_user_options().matrix_roles


def roles_refusal(section: GroupSection, context) -> Optional[str]:
    """Why the section cannot run in the three roles (`Options.
    matrix_roles`), or None where it can: one asynchronous step, on
    tcgen05 so far, with the star step folded into it (`matrix_star`
    `engine`) -- the element warps compute A into its ring -- and B
    streamed prepared through the operator ring (`matrix_ring`
    `prepared`), which the operator role fills; D added a cell at a time
    by the mma role, not by bulk reductions."""
    if roles_asked(context) == 'off':
        return 'the option is off'
    steps = [s for s in section.steps if s.lowering in ASYNC_LOWERINGS]
    if len(steps) != 1:
        return f'{len(steps)} asynchronous steps; the roles take one'
    step = steps[0]
    if step.lowering != 'tcgen05':
        return (f'the step is {step.lowering}\'s; three roles are tcgen05\'s '
                f'(sm_100a) so far')
    if not folded(section, step):
        return ('the star step is not folded into the operator step '
                '(`matrix_star` `engine`): the element warps compute A '
                'into its ring')
    others = sorted({s.lowering for s in section.steps
                     if s.lowering not in ASYNC_LOWERINGS + ('fused',)})
    if others:
        return (f'steps of {", ".join(others)} beside the asynchronous one; '
                f'the roles take its folded star and it alone')
    if not all(st.prepared and st.streamed
               for st in _step_stages(section, step)):
        return ('B does not stream prepared (`matrix_ring` `prepared`): the '
                'operator role fills that ring')
    if step.output:
        return ('D is added by bulk reductions; the mma role adds it a cell '
                'at a time')
    return None


def roles_notes(group: 'ElementGroup', context) -> List[str]:
    """What the kernel's comment says of the block's roles where the options
    ask for them (`Options.matrix_roles`): the warps each role takes and
    what it does, or why the block runs in none -- the section's refusal,
    or, where its own plan would take them, the fill rule's choice of
    another plan (`fill`)."""
    asked = roles_asked(context)
    if asked == 'off':
        return []
    section = group.section
    if section.roles:
        e = group.elements
        mma = e + OPERATOR_WARPS
        return [f'the block in three roles, meeting on the rings and on a '
                f'named barrier each: warps 0 to {e - 1} stage their '
                f'elements and compute A into the folded star\'s ring '
                f'(star), warp {e} fills B\'s prepared ring by bulk copies '
                f'(operator), warps {mma} to {mma + MMA_WARPS - 1} issue the '
                f'products and add D into the outputs (mma)']
    why = roles_refusal(dataclasses.replace(section, roles=asked), context)
    if why is None:
        why = ('the roles\' plan does not fit the group the fill rule '
               'chose, which this one does')
    return [f'matrix_roles={asked} declined: {why}']


def producer_refusal(section: GroupSection, context) -> Optional[str]:
    """Why a producer warp cannot fill the section's inputs through a ring,
    or None where it can (`Options.element_producer`).

    The producer copies an element's box of an input a column at a time,
    one bulk copy a column (`IRBuilder.copy_bulk` into a `RingSlot`), where
    the copy takes it (`bulk_input`), and a cell an asynchronous copy of
    its first warp otherwise -- of an input stored sparse, SeisSol's `star`
    under its GPU memory layout, the cells it stores, each from its place
    (`compute.multi_element._stored_cells`): the target has the bulk copy
    (sm_90 on), and at least one input takes it -- the producer is there
    for the large ones.  F32 only, as the matrix tiling's group is."""
    options = context.get_user_options()
    if not getattr(options, 'element_producer', False):
        return 'the option is off'
    if not context.target.bulk_copy_path():
        return 'no bulk copy on the target'
    if section.dtype is not Datatype.F32:
        return f'{section.dtype} inputs'
    if not section.inputs:
        return 'no inputs'
    if not any(bulk_input(b) for b in section.inputs):
        return 'no input a bulk copy takes'
    return None


def _bulk_pitched(section: GroupSection) -> GroupSection:
    """The section with each input a bulk copy may take (`bulk_input`)
    pitched for it: its columns on four floats in the group's buffer, the
    rows a copy moves (`bulk_rows`) to each -- a pitch the matrix tiling
    already gave it stays where it holds them."""
    inputs = []
    for buffer in section.inputs:
        rows = bulk_rows(buffer)
        if _bulk_source(buffer) and (buffer.pitch % 4 or buffer.pitch < rows):
            buffer = dataclasses.replace(buffer, stride=_round_up4(max(
                buffer.pitch, rows)))
        inputs.append(buffer)
    return dataclasses.replace(section, inputs=tuple(inputs))


def _round_up4(n: int) -> int:
    return -(-n // 4) * 4


def _bulk_source(buffer: Buffer) -> bool:
    """Whether the input's storage lets a bulk copy take a column of its
    box (`bulk_input`), whatever the group's buffer is pitched to."""
    tensor = buffer.tensor
    if not tensor.is_dense():
        return False
    lower = int(tensor.bbox.lower()[0])
    ld = int(tensor.bbox.sizes()[0])
    first = buffer.rows[0] - lower
    return ((getattr(tensor, 'alignment', 0) or 0) >= 16
            and first % 4 == 0 and ld % 4 == 0
            and first + bulk_rows(buffer) <= ld)


def bulk_input(buffer: Buffer) -> bool:
    """Whether a producer copies an input's box by bulk copies, a column
    each (`producer_refusal`), rather than a cell a copy: the operand
    promises 16 bytes (`alignment`, as `Symbol.linear_align_bytes` reads
    it), a column's first row and the columns' stride in its storage -- the
    bounding box's rows -- are on four floats, the box's rows rounded to
    four stay inside the column, and the pitched buffer's columns are on
    four floats and hold them."""
    rows = bulk_rows(buffer)
    return (_bulk_source(buffer) and buffer.pitch % 4 == 0
            and buffer.pitch >= rows)


def bulk_rows(buffer: Buffer) -> int:
    """The rows a bulk copy of one of the box's columns moves: its rows
    rounded to four floats, 16 bytes (`bulk_input`)."""
    return -(-buffer.height // 4) * 4


def second_stages(section: GroupSection, context) -> int:
    """How many of the section's inputs, in the order a trip stages them,
    `pir.wrap` may copy a trip ahead into a second stage, which the group
    then holds twice: the first `Options.move_distance` -- the transfers it
    moves -- where the options ask for the move and for its second stage
    (`enable_wrap_loads`, `enable_multibuffer`), the copies are the
    hardware's (`async_staging`), and the barrier placement runs
    (`enable_sync_block_opt`): an instruction's own barrier ahead of the
    staging stops the move.  At most that many: an input the pass leaves
    where it is keeps its room unused."""
    options = context.get_user_options()
    if not (options.enable_wrap_loads and options.enable_multibuffer
            and options.enable_sync_block_opt and async_staging(context)):
        return 0
    return min(int(options.move_distance), len(section.inputs))


def _mixes(step: Step, section: GroupSection) -> bool:
    """Whether a mixing step takes the `mix` form (`compute.multi_element.
    _mix`): one product, over the step's window, of a group operand's
    buffer and an element's small operand's, assigned to an intermediate --
    the star step the matrix tiling reads its A from -- whose columns a
    block's warp holds a thread each of."""
    if step.kind is not Kind.MIXING or len(step.products) != 1:
        return False
    p = step.products[0]
    if ((p.rows, p.cols) != (step.rows, step.cols) or not step.assigns
            or not step.dest.is_tmp):
        return False
    try:
        small = section.buffer(p.second.tensor)
        section.buffer(p.first.tensor)
    except KeyError:
        return False
    return small.role is Role.SMALL and p.cols[1] - p.cols[0] <= WARP_LANES


def tiled(section: GroupSection) -> bool:
    """Whether a step of the section is lowered on the matrix engines."""
    return any(s.lowering in MATRIX_LOWERINGS for s in section.steps)


def asynchronous(section: GroupSection) -> bool:
    """Whether a step of the section is lowered on an asynchronous matrix
    engine (`ASYNC_LOWERINGS`)."""
    return any(s.lowering in ASYNC_LOWERINGS for s in section.steps)


def streamed(section: GroupSection) -> bool:
    """Whether the section streams B through its steps' rings."""
    return any(st.streamed for st in section.stages)


#: The rows of K a block of the folded star step holds of every product
#: (`GroupSection.star`): one step of the engines' K -- what a slot of A's
#: ring holds, and a chunk of B's prepared ring.
STAR_ROWS = 8

#: The slots of A's ring of the folded star step, by engine: tcgen05's in
#: tensor memory, a block of K of every product each, both halves -- the
#: star step a few blocks ahead of the products; wgmma's in shared memory,
#: a warpgroup's tile of rows of a block each, the next block written while
#: the engine multiplies this one.
STAR_SLOTS = {'tcgen05': 4, 'wgmma': 2}


def star_mixes(section: GroupSection, step: Step) -> Optional[List[Step]]:
    """The star steps whose intermediates the products of `step` read as A,
    a step a product, in the products' order -- or None where some
    product's A is no intermediate one `mix` step alone writes."""
    out = []
    for p in step.products:
        t = p.second.tensor
        if not t.is_tmp:
            return None
        writers = [s for s in section.steps if s.dest is t]
        if len(writers) != 1 or writers[0].lowering not in ('mix', 'fused'):
            return None
        out.append(writers[0])
    return out


def fuse_refusal(section: GroupSection, step: Step) -> Optional[str]:
    """Why the star step cannot be folded into `step` (`fuse`), or None:
    an asynchronous step whose products read as A the intermediates of one
    run of star steps -- one group operand over one window, each
    intermediate read by this step alone, as A -- over one contraction, the
    products' columns their star steps' own; B resident or streamed
    prepared (the converting ring's chunks are a product's, and the folded
    star walks K a block of every product at a time); on tcgen05, the
    tensor memory for D and A's ring."""
    if step.lowering not in ASYNC_LOWERINGS:
        return f'{step.lowering} takes no folded star step'
    mixes = star_mixes(section, step)
    if mixes is None:
        return (f'an A of {step.dest.alias} is no intermediate a star step '
                f'alone writes')
    for p in step.products:
        t = p.second.tensor
        if any(sub.tensor is t for other in section.steps
               for q in other.products for sub in q.operands
               if not (other is step and q is p and sub is p.second)):
            return f'{t.alias} is read besides as the A of {step.dest.alias}'
    if len({(id(m.products[0].first.tensor), m.products[0].rows,
             m.products[0].depth) for m in mixes}) != 1:
        return 'the star steps read different group operands or windows'
    if len({p.depth for p in step.products}) != 1:
        return f'the contractions of {step.dest.alias}\'s products differ'
    for p, mix in zip(step.products, mixes):
        m = mix.products[0]
        if not (m.rows[0] <= p.depth[0] and p.depth[1] <= m.rows[1]
                and p.cols == m.cols):
            return (f'{p.second.tensor.alias} is read past what its star '
                    f'step writes')
    stages = _step_stages(section, step)
    if not all(st.split for st in stages):
        return ('B is read whole, in TF32, where its halves leave the group '
                'no room: the folded star takes B\'s halves')
    if any(st.streamed and not st.prepared for st in stages):
        return ('B streams through the converting ring, a product\'s '
                'chunks at a time, and the folded star walks K a block of '
                'every product at a time: it needs B resident or prepared '
                '(matrix_ring=prepared)')
    if step.lowering == 'tcgen05':
        _, n_, _ = matrix_shape('tcgen05', 100)
        n = _round(step.rows[1] - step.rows[0], n_)
        columns = n + STAR_SLOTS['tcgen05'] * 2 * STAR_ROWS * len(stages)
        if columns > 512:
            return (f'D and A\'s ring take {columns} columns of tensor '
                    f'memory, past its 512')
    return None


def fuse(section: GroupSection) -> GroupSection:
    """The section with its star steps folded into the asynchronous steps
    that can take them (`fuse_refusal`): those star steps lowered `fused`
    -- their work the operator step's -- and their intermediates held
    (`Buffer.held`), `star` `engine`; unchanged where none can."""
    folded: Set[int] = set()
    held: Set[int] = set()
    for step in section.steps:
        if fuse_refusal(section, step) is not None:
            continue
        for mix in star_mixes(section, step):
            folded.add(id(mix))
            held.add(id(mix.dest))
    if not folded:
        return section
    steps = tuple(dataclasses.replace(s, lowering='fused')
                  if id(s) in folded else s for s in section.steps)
    return dataclasses.replace(
        section, steps=steps, star='engine',
        intermediates=tuple(dataclasses.replace(b, held=True)
                            if id(b.tensor) in held else b
                            for b in section.intermediates))


def folded(section: GroupSection, step: Step) -> bool:
    """Whether `step` computes its A itself, the star step folded into it
    (`GroupSection.star`)."""
    return (section.star == 'engine' and step.lowering in ASYNC_LOWERINGS
            and bool(star_mixes(section, step))
            and all(m.lowering == 'fused' for m in star_mixes(section, step)))


def star_volume(section: GroupSection, elements: int) -> int:
    """Elements of shared memory the folded star's A takes in a group of
    `elements`: wgmma's ring, `STAR_SLOTS` slots a warpgroup -- its tile of
    64 rows of a block of every product, both halves; tcgen05's is in
    tensor memory, and only its mbarriers take shared memory
    (`shared_bytes`)."""
    total = 0
    for step in section.steps:
        if step.lowering != 'wgmma' or not folded(section, step):
            continue
        groups = max(1, elements // WARPGROUP_WARPS)
        total += (groups * STAR_SLOTS['wgmma'] * 64 * STAR_ROWS * 2
                  * len(step.products))
    return total


def precision_notes(section: GroupSection, context) -> List[str]:
    """What the kernel's comment says of the matrix engine the fill rule
    chose for an F32 section, where it is not the target's asynchronous one
    for the accuracy it keeps (`GroupSection.engine_note`), or of the
    reduced precision the options let it take (`tf32_operands`)."""
    notes = [section.engine_note] if section.engine_note else []
    if section.dtype is Datatype.F32 and tf32_operands(context):
        whole = sorted({p.first.tensor.alias for step in section.steps
                        if step.lowering in ASYNC_LOWERINGS
                        for p in step.products if p.kind is Kind.OPERATOR
                        and not section.stage(p.first.tensor,
                                              (p.rows, p.depth)).split})
        if whole:
            notes.append(f'{", ".join(whole)} read in TF32 alone, two of '
                         f'3xTF32\'s three products (about 2^-11 relative a '
                         f'product): Options.matrix_tf32_operands')
    return notes


def star_notes(section: GroupSection, context) -> List[str]:
    """What the kernel's comment says of its star step where the options
    ask for it folded (`Options.matrix_star` `engine`): which steps compute
    their A themselves, where it goes, or why a step does not (and takes
    the halves where they fit)."""
    if star_asked(context) != 'engine':
        return []
    notes = []
    for step in section.steps:
        if step.lowering not in ASYNC_LOWERINGS:
            continue
        if folded(section, step):
            where = ('tensor memory' if step.lowering == 'tcgen05'
                     else 'shared memory, the canonical layout')
            notes.append(f'the star step folded into {step.dest.alias}: A\'s '
                         f'halves into {where}, a block of {STAR_ROWS} rows '
                         f'of every product at a time through a ring of '
                         f'{STAR_SLOTS[step.lowering]} slots')
            continue
        unfolded = dataclasses.replace(section, star='', steps=tuple(
            dataclasses.replace(s, lowering='mix') if s.lowering == 'fused'
            else s for s in section.steps))
        why = fuse_refusal(unfolded, step)
        if why is None:
            # A step that takes the fold, the fill rule's choice: the fold
            # does not leave the group room that the unfolded tiling does.
            why = ('the fold does not fit the group the fill rule chose, '
                   'which the unfolded tiling does')
        notes.append(f'matrix_star=engine declined for {step.dest.alias}: '
                     f'{why}; A in TF32 halves where they fit')
    return notes


def output_asked(context) -> str:
    """How the options ask a matrix step to add D into its output
    (`Options.matrix_output`): `red`, `bulk` or `box`."""
    return context.get_user_options().matrix_output


def output_refusal(section: GroupSection, step: Step, context) -> Optional[str]:
    """Why a matrix step cannot add D into its output by bulk reductions
    (`Step.output`), or None: it adds -- a store would be a bulk copy, not
    built -- into an output of the element that no other step of the
    section writes (two steps' reductions into one region would land in an
    order of their own, and a check bit for bit would see it) nor reads (the
    kernel waits for the reductions to have read the staged columns, not
    for their writes to land), stored dense,
    on a target with the bulk copy (sm_90 on), its columns on sixteen bytes
    -- the operand promising them (`alignment`) and its column's rows a
    multiple of sixteen bytes -- and the step's columns at most a warp's
    lanes, a lane issuing each."""
    if step.lowering not in MATRIX_LOWERINGS:
        return f'{step.lowering} writes no D'
    tensor = step.dest
    if tensor.is_tmp:
        return f'D goes to the intermediate {tensor.alias}'
    if step.assigns:
        return f'the step assigns {tensor.alias}'
    writers = sum(1 for s in section.steps if s.dest is tensor)
    if writers > 1:
        return (f'{writers} steps add into {tensor.alias}: their reductions '
                f'would land in an order of their own')
    if any(sub.tensor is tensor for s in section.steps for p in s.products
           for sub in p.operands):
        return (f'a step reads {tensor.alias}: the reductions\' writes land '
                f'behind the wait for their reads only')
    if not context.target.bulk_copy_path():
        return 'no bulk copy on the target'
    if not tensor.is_dense():
        return f'{tensor.alias} is stored sparse'
    size = section.dtype.size()
    align = int(getattr(tensor, 'alignment', 0) or 0)
    if align < 16:
        return (f'{tensor.alias} promises {align} bytes of alignment, the '
                f'bulk reduction 16')
    ld = int(tensor.bbox.sizes()[0])
    if (ld * size) % 16:
        return (f'a column of {tensor.alias} is {ld * size} bytes, not a '
                f'multiple of 16')
    if step.cols[1] - step.cols[0] > WARP_LANES:
        return (f'{step.cols[1] - step.cols[0]} columns, more than a warp\'s '
                f'lanes issue')
    return None


def output_geometry(section: GroupSection, step: Step) -> Tuple[int, int, int]:
    """`(lo, hi, stride)` of a step's D staged for its bulk reductions
    (`Step.output`): the rows of the output's column a reduction adds, from
    `lo` to `hi` in its storage (from the bounding box's first row), and the
    elements between two columns staged -- `bulk`: the step's rows rounded
    out to sixteen bytes, the stride an odd number of sixteen bytes, so that
    a warp's lanes, a column each, store on distinct banks; `box`: the whole
    column, the stride the output's own, so that an element's columns are
    one contiguous box."""
    tensor = step.dest
    unit = 16 // section.dtype.size()
    ld = int(tensor.bbox.sizes()[0])
    first = int(tensor.bbox.lower()[0])
    if step.output == 'box':
        return 0, ld, ld
    lo = (step.rows[0] - first) // unit * unit
    hi = min(ld, _round(step.rows[1] - first, unit))
    stride = _round(hi - lo, unit)
    if (stride // unit) % 2 == 0:
        stride += unit
    return lo, hi, stride


def output_volume(section: GroupSection, elements: int) -> int:
    """Elements of shared memory the steps' staged D takes for their bulk
    reductions in a group of `elements` (`Step.output`), past what tcgen05
    stages anyway (`shared_bytes`)."""
    total = 0
    for step in section.steps:
        if not step.output:
            continue
        _, _, stride = output_geometry(section, step)
        total += elements * (step.cols[1] - step.cols[0]) * stride
    return total


def output_of(section: GroupSection) -> str:
    """How the section's matrix steps add D into their output: `bulk` or
    `box` where one takes the bulk reductions (`Step.output`), `red`
    where none does."""
    return next((s.output for s in section.steps if s.output), 'red')


def _output_twins(section: GroupSection) -> Tuple[GroupSection, ...]:
    """The section, and where a step of it adds D by bulk reductions the
    same section adding it a cell at a time (`fill`)."""
    if not any(s.output for s in section.steps):
        return (section,)
    return (section, dataclasses.replace(section, steps=tuple(
        dataclasses.replace(s, output='') for s in section.steps)))


def output_notes(section: GroupSection, context) -> List[str]:
    """What the kernel's comment says of its matrix steps' output where
    the options ask for bulk reductions (`Options.matrix_output`): the rows
    each reduction adds and the +0.0 they add outside the step's windows, or
    why a step keeps its reduction a cell."""
    asked = output_asked(context)
    if asked == 'red':
        return []
    notes = []
    for step in section.steps:
        if step.lowering not in MATRIX_LOWERINGS or step.dest.is_tmp:
            continue
        if not step.output:
            why = ('the mma role adds D a cell at a time (`matrix_roles`)'
                   if section.roles else output_refusal(section, step,
                                                        context))
            if why is None:
                # A step that takes them, the fill rule's choice: the
                # group's size, or another of its lowering's choices.
                why = ('its staged columns do not fit beside the group '
                       'the fill rule chose -- its size and its lowering\'s '
                       'other choices kept')
            notes.append(f'matrix_output={asked} declined for '
                         f'{step.dest.alias}: {why}; a reduction a cell')
            continue
        lo, hi, _ = output_geometry(section, step)
        first = int(step.dest.bbox.lower()[0])
        what = ('a bulk reduction a column of the element' if step.output ==
                'bulk' else 'one bulk reduction over the element\'s box')
        notes.append(f'{step.dest.alias} += D by {what}, rows {first + lo} '
                     f'to {first + hi}: the rows the products\' windows leave '
                     f'out there receive exactly +0.0 (a -0.0 becomes +0.0)')
    return notes


def _step_stages(section: GroupSection, step: Step) -> List[Stage]:
    """The stages a matrix step reads, a product each, in its order."""
    return [section.stage(p.first.tensor, (p.rows, p.depth))
            for p in step.products]


def ring_volume(section: GroupSection) -> int:
    """Elements of the rings the section's streamed steps take: for each,
    `RING_SLOTS` slots of a chunk of its B, both halves where split -- or,
    where it streams prepared, the prepared ring's slots of its largest
    chunk (`ring_chunks`) and their pair of mbarriers each, eight bytes
    apiece in elements of the section's type."""
    total = 0
    for step in section.steps:
        if step.lowering not in MATRIX_LOWERINGS:
            continue
        stages = [st for st in _step_stages(section, step) if st.streamed]
        if not stages:
            continue
        if stages[0].prepared:
            total += section.ring_slots * max(
                chunk_volume(section, step, chunk)
                for chunk in ring_chunks(section, step))
            total += 2 * section.ring_slots * 8 // section.dtype.size()
            continue
        width = max(st.width for st in stages)
        halves = 2 if stages[0].split else 1
        total += RING_SLOTS * RING_CHUNK * width * halves
    return total


#: How many chunks of the prepared ring an engine's products hold a slot
#: for where the options leave it to them (`Options.matrix_ring_hold`,
#: `engines.RingSpec.hold`): wgmma's warps give a chunk's slot back behind
#: the wait that leaves the next chunk's products in flight and retires its
#: own (`multi_element._wgmma_ring`, `wait_group 1`); tcgen05 commits a
#: chunk's products onto its slot's empty side, which they arrive at once
#: complete.
RING_HOLDS = {'wgmma': 2, 'tcgen05': 1}


def ring_hold(section: GroupSection, step: Step) -> int:
    """How many chunks the products of `step` hold a slot of its prepared
    ring for (`engines.RingSpec.hold`): the section's, where the options
    asked and every ring takes it (`hold_refusal`), else the engine's
    own."""
    return section.ring_hold or RING_HOLDS[step.lowering]


def hold_refusal(section: GroupSection, hold: int) -> Optional[str]:
    """Why the prepared rings of the section's streamed steps cannot hold
    their slots for `hold` chunks (`Options.matrix_ring_hold`), or None:
    tcgen05 gives a slot back by the commit of its chunk's products, which
    arrive at once complete -- 1 alone, but in the three roles
    (`GroupSection.roles`), whose mma role may commit a chunk's slot behind
    later chunks' products (`multi_element._tcgen05_roles`); at most the
    ring's slots, a hold past them waiting for a round no slot is free for
    -- and on wgmma 2 where the star step is folded, whose A has two slots
    a warpgroup that the same wait gives back (`STAR_SLOTS`)."""
    for step in section.steps:
        if step.lowering not in ASYNC_LOWERINGS:
            continue
        stages = _step_stages(section, step)
        if not (stages[0].prepared and stages[0].streamed):
            continue
        if step.lowering == 'tcgen05' and hold != 1 and not section.roles:
            return ('tcgen05 commits a chunk\'s products onto its slot\'s '
                    'empty side, which they arrive at once complete: a '
                    'hold of 1 alone, but in the three roles '
                    '(`matrix_roles`)')
        if hold > section.ring_slots:
            return (f'a hold of {hold} chunks is past the ring\'s '
                    f'{section.ring_slots} slots')
        if folded(section, step) and hold > STAR_SLOTS[step.lowering]:
            return (f'the folded star\'s A has {STAR_SLOTS[step.lowering]} '
                    f'slots a warpgroup, which the products\' wait gives '
                    f'back with B\'s')
    return None


def _held(section: GroupSection, context) -> GroupSection:
    """The section with the hold the options ask of its prepared rings
    (`Options.matrix_ring_hold`) where they take it (`hold_refusal`), each
    engine's own otherwise (`GroupSection.ring_hold` 0)."""
    hold = context.get_user_options().matrix_ring_hold
    if hold is None or not section.ring_slots:
        return section
    taken = int(hold) if hold_refusal(section, int(hold)) is None else 0
    return dataclasses.replace(section, ring_hold=taken)


def ring_chunks(section: GroupSection, step: Step) -> Tuple[
        Tuple[Tuple[int, int, int], ...], ...]:
    """The chunks a step streams its prepared B through the ring in, in the
    order its products read them: each a slot's worth, pieces `(product,
    k0, rows)` -- `rows` rows of K of the product's stage from `k0` on,
    each piece both halves, the upper ones' canonical layout over its rows
    and then the lower ones' (`_Piece`).  A product's depth in pieces of
    the ring's rows (`GroupSection.ring_rows`), a piece a chunk."""
    stages = _step_stages(section, step)
    if folded(section, step):
        # The folded star's order: a block of K of every product a chunk,
        # what a slot of A's ring holds too.
        depth = max(st.depth for st in stages)
        return tuple(tuple((p, k0, min(STAR_ROWS, st.depth - k0))
                           for p, st in enumerate(stages) if k0 < st.depth)
                     for k0 in range(0, depth, STAR_ROWS))
    rows = section.ring_rows or RING_CHUNK
    return tuple(((p, k0, min(rows, st.depth - k0)),)
                 for p, st in enumerate(stages)
                 for k0 in range(0, st.depth, rows))


def chunk_volume(section: GroupSection, step: Step, chunk) -> int:
    """Elements a chunk of a step's prepared ring holds: each piece's rows
    of its stage's width, both halves."""
    stages = _step_stages(section, step)
    return sum(rows * stages[p].width * 2 for p, _, rows in chunk)


def prepared(section: GroupSection) -> bool:
    """Whether a stage of the section is prepared at generation
    (`Stage.prepared`)."""
    return any(st.prepared for st in section.stages)


def star_asked(context) -> str:
    """Where the options ask the star step to put A for the asynchronous
    engines (`Options.matrix_star`): `buffer`, `halves` or `engine`."""
    return context.get_user_options().matrix_star


def ring_asked(context) -> bool:
    """Whether the options ask for the asynchronous engines' B prepared
    (`Options.matrix_ring` `prepared`)."""
    return context.get_user_options().matrix_ring == 'prepared'


def ring_notes(section: GroupSection, context) -> List[str]:
    """What the kernel's comment says of its asynchronous steps' B where the
    options ask for it prepared (`Options.matrix_ring`): the table each
    step's B comes from -- the description's numbers, the operators'
    pointers not read for it -- and its ring; or why a step keeps B
    converted by the element warps."""
    if not ring_asked(context):
        return []
    from tensorforge.backend.instructions.compute.primitives import nvidia
    sm = nvidia.sm_of(context)
    notes = []
    for step in section.steps:
        if step.lowering not in ASYNC_LOWERINGS:
            continue
        stages = _step_stages(section, step)
        operators = ', '.join(st.tensor.alias for st in stages)
        if not stages[0].prepared:
            why = step_refusal(step) or ('its B is not split into TF32 '
                                         'halves')
            notes.append(f'matrix_ring=prepared declined for {operators} '
                         f'into {step.dest.alias}: {why}; B converted by '
                         f'the element warps')
            continue
        name, numbers, _ = prepared_table(section, step, sm)
        where = (f'streamed through a ring of {section.ring_slots} slots, '
                 f'{len(ring_chunks(section, step))} chunks a trip'
                 if stages[0].streamed else 'copied once a block')
        if stages[0].streamed and section.ring_hold:
            where += (f', a slot held for {section.ring_hold} '
                      f'chunk{"s" if section.ring_hold > 1 else ""}')
        notes.append(f'B of {operators} into {step.dest.alias} prepared from '
                     f'the description\'s numbers in {name} ({len(numbers)} '
                     f'floats), {where}; the operators\' pointers are not '
                     f'read for it')
    hold = context.get_user_options().matrix_ring_hold
    why = (hold_refusal(section, int(hold))
           if hold is not None and section.ring_slots else None)
    if why is not None:
        notes.append(f'matrix_ring_hold={hold} declined: {why}; each ring '
                     f'holds its engine\'s own (wgmma {RING_HOLDS["wgmma"]}, '
                     f'tcgen05 {RING_HOLDS["tcgen05"]})')
    return notes


def prepared_table(section: GroupSection, step: Step, sm: int, form=None):
    """`(name, numbers, offsets)`: a step's B prepared at generation
    (`Stage.prepared`) as the table of the generated file its bulk copies
    read -- the description's numbers in the derived form `form`, by
    default the TF32 halves as `splitFloatTF32` splits them on the target
    (`forms.tf32_device_for`) -- and where each piece starts in it, by
    `(product, k0)`.

    Streamed, the ring's chunks one after another (`ring_chunks`); resident,
    each product's stage whole, as `MultiElementOperators._canonical` lays
    it out.  A piece (`forms.lay`): the first part of its rows of K in the
    canonical K-major layout over those rows (`forms.canonical_k_major`) --
    core matrices of eight rows of N by four of K, 128 bytes apart along K
    and `rows` x 32 along N -- then each further part the same way;
    `B(k, n)` the operator's `(n0 + n, k0' + k)` inside the product's
    window and zero outside it, past its contraction too (`Stage`).  Named
    after its numbers (`multilinear._module_table_name`): kernels of one
    file with the same B share it."""
    import numpy as np
    from tensorforge.backend.instructions.compute.multilinear import \
        _module_table_name
    from tensorforge.common.matrix import forms
    form = form if form is not None else forms.tf32_device_for(sm)
    stages = _step_stages(section, step)
    if not all(st.prepared for st in stages):
        raise ValueError(f'{step.dest.alias}: a step whose B is not prepared')
    if stages[0].streamed:
        pieces = [piece for chunk in ring_chunks(section, step)
                  for piece in chunk]
    else:
        pieces = [(p, 0, st.depth) for p, st in enumerate(stages)]
    parts, offsets, at = [], {}, 0
    for p, k0, rows in pieces:
        stage = stages[p]
        values = np.asarray(stage.tensor.get_values(), dtype=np.float64)
        (r0, r1), (d0, d1) = stage.window
        block = np.zeros((rows, stage.width))
        for k in range(rows):
            col = d0 + k0 + k
            if col >= d1:
                continue
            for n in range(stage.width):
                row = stage.n0 + n
                if r0 <= row < r1:
                    block[k, n] = values[row, col]
        laid = forms.lay(form, block,
                         forms.canonical_k_major(rows, stage.width))
        offsets[(p, k0)] = at
        at += laid.size
        parts.append(laid)
    numbers = tuple(float(v) for v in np.concatenate(parts))
    return _module_table_name('bk', form.stored, numbers), numbers, offsets


def split(section: GroupSection) -> bool:
    """Whether the section's B stages hold TF32 halves."""
    return any(st.split for st in section.stages)


def halves(section: GroupSection) -> int:
    """How many of the section's buffers hold A in TF32 halves
    (`Buffer.halves`, `_halved`)."""
    return sum(b.halves for b in section.intermediates)


def shared_bytes(section: GroupSection, elements: int) -> int:
    """Shared memory a group of `elements` takes: the operators as staged
    -- their stages, the rings B streams through, and the terms of the
    steps that walk them -- each element's copy of every input and
    intermediate, both halves where one holds them (`Buffer.room`), its
    second copy of every input staged a trip ahead
    (`GroupSection.doubled`), and its entry in every pointer table."""
    size = section.dtype.size()
    operators = (sum(st.volume for st in section.stages)
                 + sum(s.terms.volume for s in section.steps
                       if s.terms is not None)
                 + ring_volume(section))
    per_element = sum(b.room for b in section.inputs
                      + section.intermediates
                      + section.inputs[:section.doubled])
    if section.ring:
        # Every slot holds the group's inputs; and a pair of mbarriers a
        # slot.
        per_element += (section.ring - 1) * sum(b.per_element
                                                for b in section.inputs)
        operators += 2 * section.ring * 8 // size
    # The folded star's A: wgmma's ring in shared memory; tcgen05's in
    # tensor memory, a pair of mbarriers a slot here.
    operators += star_volume(section, elements)
    operators += sum(2 * STAR_SLOTS['tcgen05'] * 8 // size
                     for s in section.steps
                     if s.lowering == 'tcgen05' and folded(section, s))
    # tcgen05's D staged in shared memory for every element warp to write
    # (`compute.multi_element._tcgen05`): the widest step's rows of the
    # element, an odd number of floats apart -- for bulk reductions, the
    # columns of the output they add (`output_geometry`), past the others.
    per_element += max(((s.cols[1] - s.cols[0]) * ((s.rows[1] - s.rows[0]) | 1)
                        for s in section.steps if s.lowering == 'tcgen05'
                        and not s.output),
                       default=0)
    operators += output_volume(section, elements)
    return (size * (operators + elements * per_element)
            + POINTER_BYTES * elements * len(section.pointer_tables))


def table_elements(section: GroupSection) -> int:
    """The table's size for the section: the entry of the smallest basis
    that holds the section's, of its precision."""
    fitting = [b for b, t in ELEMENTS
               if t is section.dtype and b >= section.basis]
    if not fitting:
        return DEFAULT_ELEMENTS
    return ELEMENTS[(min(fitting), section.dtype)]


# -- the lanes form ---------------------------------------------------------- #

def lane_elements(context) -> int:
    """The lanes form's group on the context's target: a warp's lanes, an
    element each -- the width of the target's vector unit, a CUDA warp of
    32, a wave of 64 on CDNA and of 32 on RDNA, a sub-group of 16 on
    Xe-HPC, which the kernel states as its multiplication's width
    (`reqd_sub_group_size`).  Not wider: two of the block's warps on one
    wave would be two shares of the output, whose ranges' guards diverge."""
    return int(context.target.hw.vec_unit_length)


def lane_grf(context) -> Optional[str]:
    """The register file the lanes form's kernel asks IGC for on Intel, as
    `Options.sycl_grf` spells it: the option's where it is set, and where it
    is not the large file, 256 registers a thread -- the plan holds a
    lane's column in registers and is sized for it, as the explicit vector
    is (and the PVC prototype asked for it explicitly: IGC's own choice ran
    its chained products twice as long).  None off Intel."""
    if getattr(context.target.hw, 'vendor', None) != 'intel':
        return None
    grf = str(context.get_user_options().sycl_grf).strip()
    return '256' if grf == 'default' else grf


def lane_registers(context, elements: int) -> int:
    """Registers, in 32-bit words, the lanes form's plan lets a lane hold
    live on the context's target: `LANE_REGISTERS` of the 255 a CUDA thread
    addresses, and the same share of what a lane addresses elsewhere -- 256
    VGPRs on AMD, the ones an FMA takes as operands (CDNA's AGPRs, which
    share the file from gfx90a on, are the matrix cores' and the spills');
    on Intel the lane's share of its hardware thread's file, which the
    sub-group's `elements` lanes split: 128 words a lane at SIMD16 in the
    small file, 256 in the large one (`lane_grf`)."""
    hw = context.target.hw
    file = getattr(hw, 'max_reg_per_thread', None) or 1020
    if getattr(hw, 'vendor', None) == 'amd':
        file = min(file, 1024)
    if getattr(hw, 'vendor', None) == 'intel':
        # `max_reg_per_thread` is the small file, 128 registers of 64 bytes.
        if lane_grf(context) in ('256', 'auto'):
            file *= 2
        file //= elements
    return LANE_REGISTERS * (file // 4) // 255


#: Where the lanes form takes its operators' nonzeros from by vendor and
#: element type, where `Options.multi_element_values` is `auto`
#: (`lane_values`).  NVIDIA: literals of the code -- an FFMA takes its
#: immediate and a DFMA reads its number from a constant bank as an operand,
#: at no instruction (CUDA Binary Utilities, "Instruction Set Reference":
#: FFMA32I, constant-bank operands).  AMD: literals at F32, which a VOP2
#: `v_fmac_f32` takes as a dword of the instruction, and so does a VOP3 from
#: gfx11 on; the operator's own memory at F64, scalar loads into SGPRs --
#: gfx9 has no VOP3 literal, so a `v_fma_f64` cannot take its number from the
#: code (AMD "CDNA3 ISA", VOP3 encoding, literal constants).  Read from
#: memory at F32, the numbers were what the AMDGPU backend held: the o6 volume
#: spilled 1070 SGPRs on gfx942 and 64 on gfx1150, as literals none.  Intel:
#: an Xe `mad` takes no 32-bit immediate, and a uniform read is a message a
#: number; a table of the numbers in the order they are multiplied by is read
#: a sub-group's width at a time and each handed to the lanes from the lane
#: that holds it, which IGC makes a register region of the `mad` -- the PVC
#: prototype's operands.
LANE_VALUES: Dict[str, Dict[Datatype, str]] = {
    'nvidia': {Datatype.F32: 'literal', Datatype.F64: 'literal'},
    'amd': {Datatype.F32: 'literal', Datatype.F64: 'memory'},
    'intel': {Datatype.F32: 'table', Datatype.F64: 'table'}}

#: The vendors whose lanes form takes every nonzero of an operator as a
#: literal where its numbers are literals (`lane_literals`), and not at most
#: `Options.inline_constants`: that cap is the per-element kernels' -- 64 on
#: AMD, whose literals there are scalar registers -- and the lanes form's
#: literal on AMD is a dword of its FMA.  Elsewhere the option caps them.
LANE_LITERALS = ('amd',)

#: Target features the lanes form's kernel turns off, by vendor
#: (`lane_features`).  AMD at F32, where the target has packed FP32
#: arithmetic (`packed-fp32-ops`, CDNA2 and later): LLVM's SLP vectorizer
#: pairs the column's FMAs into `v_pk_fma_f32`, whose numbers are registers
#: -- pairs of SGPRs, gathered ahead of the pairs -- and on gfx942 the o6
#: volume's literals then took 852 SGPRs of spill; none without the pairs.
LANE_FEATURES = {'amd': 'no-packed-fp32-ops'}

#: The vendors whose lanes form anchors what it reads to the column it reads
#: it for (`lane_anchors`): an address that does not change from column to
#: column -- an element's own rows the mixing product reads, an operator's
#: number -- is otherwise taken out of the column loop by the compiler, and
#: every one of them held across it.  On PVC the per-lane addresses of the
#: element's rows of the o6 volume were 200 registers of 64 bytes, spilled;
#: an anchor of the column -- a zero the compiler cannot see -- keeps them in
#: the loop.  And an anchor of each walk over the rows: the four faces of
#: `localFluxAll` read the same rows of the element, which LLVM read once
#: for all four under one anchor and held -- 16 to 60 kB of spill a thread at
#: o4 and o6 on PVC; none with an anchor a walk.  ptxas forms the addresses
#: where they are read.
LANE_ANCHORS = ('amd', 'intel')


def lane_values(context) -> str:
    """Where the lanes form's operator nonzeros come from on the context's
    target at its element type: `literal`, `memory` or `table`
    (`Options.multi_element_values`, `LANE_VALUES`)."""
    asked = context.get_user_options().multi_element_values
    if asked != 'auto':
        return asked
    return LANE_VALUES.get(getattr(context.target.hw, 'vendor', None),
                           {}).get(context.fp_type, 'memory')


def lane_literal_cap(context) -> Optional[int]:
    """The most nonzeros of an operator the lanes form takes as literals on
    the context's target, None for all of them (`LANE_LITERALS`,
    `Options.inline_constants`)."""
    if getattr(context.target.hw, 'vendor', None) in LANE_LITERALS:
        return None
    return context.get_user_options().inline_constants


def lane_features(context) -> Optional[str]:
    """The target features the lanes form's kernel turns off on the
    context's target, as the compiler spells them, or None
    (`LANE_FEATURES`): AMD's packed FP32 at F32, where the target has it."""
    hw = context.target.hw
    vendor = getattr(hw, 'vendor', None)
    if (vendor not in LANE_FEATURES or context.fp_type is not Datatype.F32
            or context.target.packed_fma_width(Datatype.F32) < 2):
        return None
    from tensorforge.backend.instructions.compute.primitives.amd.features \
        import carries
    level = hw.gfx_level()
    return (LANE_FEATURES[vendor]
            if level is not None and carries(level, 'packed-fp32-ops')
            else None)


def lane_literals(context) -> bool:
    """Whether the lanes form's operator nonzeros are literals of the code
    on the context's target (`lane_values`)."""
    return lane_values(context) == 'literal'


def lane_anchors(context) -> bool:
    """Whether the lanes form anchors its reads to the column and the walk
    (`LANE_ANCHORS`)."""
    return getattr(context.target.hw, 'vendor', None) in LANE_ANCHORS


#: The FMA form `Options.multi_element_fma` `auto` takes, by vendor: the
#: warp form everywhere, until the measurements say otherwise.
FMA_FORMS: Dict[str, str] = {}


def fma_form(context) -> str:
    """The FMA form a section takes on the context's target, `warp` or
    `lanes` (`Options.multi_element_fma`)."""
    asked = context.get_user_options().multi_element_fma
    if asked != 'auto':
        return asked
    return FMA_FORMS.get(getattr(context.target.hw, 'vendor', None), 'warp')


#: The lanes form's column classes: warp `w` of a group takes the output's
#: columns `c0 + w % classes + classes * i`, one after another.
LANE_CLASSES = 3

#: Registers, in 32-bit words, the lanes form's plan lets a lane hold live
#: by its estimate (`Lanes.registers`) on CUDA: the output is cut into more
#: row ranges until each fits.  Short of the 255 a CUDA thread addresses,
#: for the addresses and the compiler's own temporaries; the same share of
#: a lane's registers elsewhere (`lane_registers`).
LANE_REGISTERS = 168

#: The most row ranges the lanes form cuts an output into.
LANE_RANGES = 8

#: The most warps a block of the lanes form has: a block of more caps what
#: a thread may hold below the 255 registers on CUDA (`__launch_bounds__`
#: sizes it by the block), and the cut into ranges is exactly what was to
#: keep it there.  The column classes give way first: they add warps, no
#: room.  Eight waves of 64 are two a SIMD on CDNA, and 256 VGPRs each;
#: eight sub-groups are eight of an Xe core's 32 threads in the large file.
LANE_WARPS = 8

#: The most of a class's columns the lanes form computes in one pass
#: (`Lanes.columns`), by vendor, where the plan's registers allow it: one
#: everywhere so far.  A pass of several reads each number of the operators
#: once for all of them, and each row of the element's operands once.
LANE_COLUMNS: Dict[str, int] = {}


def lane_columns(context) -> int:
    """The most columns a pass of the lanes form takes on the context's
    target (`LANE_COLUMNS`)."""
    return LANE_COLUMNS.get(getattr(context.target.hw, 'vendor', None), 1)


@dataclass(frozen=True)
class Lanes:
    """A section with its group's elements on the lanes of a warp
    (`Options.multi_element_fma` `lanes`): E elements a group, a lane each,
    and `warps` warps that all take the same E elements, each a share of
    the one output, the last step's destination -- the columns of its
    class (`classes`), `columns` of them a pass, one pass after another,
    over its row range (`ranges`).

    What a warp computes for a column is its cone (`cone`): the output's
    rows of its range, and of every tensor upstream the rows those read.
    A mixing step's intermediate is not stored: where a product reads an
    entry of it, the entry is computed from the element's own operands.
    An operator step's intermediate is a column of registers, its needed
    rows, for as long as the product that reads it runs.  Only the
    operators' nonzeros are multiplied by (`operator_nonzeros`): literals
    of the code, the description's numbers, or read from the operator's
    memory at addresses every lane shares (`lane_literals`)."""

    classes: int
    ranges: Tuple[Range, ...]
    #: The plan's estimate of the registers a lane holds live at the
    #: costliest range, in 32-bit words.
    registers: int
    #: The class's columns a pass computes together: they share the
    #: operators' numbers and the element's rows each pass reads.  More
    #: than one only where they divide the class's columns, and every class
    #: has as many (`lanes_plan`).
    columns: int = 1
    #: A chained section's plans, a phase each (`GroupSection.phases`), None
    #: for a phase of copies alone: one warp takes every phase, all of its
    #: columns, so that a lane reads back what it wrote -- the output of the
    #: phase before, whose every column the next phase's columns read.
    phases: Tuple[Optional['Lanes'], ...] = ()

    @property
    def warps(self) -> int:
        if self.phases:
            return 1
        return self.classes * len(self.ranges)


def operator_nonzeros(product: Product) -> Dict[int, List[Tuple[int, float]]]:
    """An operator product's nonzeros in its window by row, `(k, value)`:
    the pattern the operator's own (`Tensor.nonzero_pattern`), the values
    the description's."""
    import numpy as np
    tensor = product.first.tensor
    data = np.asarray(tensor.get_values())
    pattern = tensor.nonzero_pattern
    (r0, r1), (k0, k1) = product.rows, product.depth
    return {i: [(k, float(data[i, k])) for k in range(k0, k1)
                if pattern.is_nz((i, k))]
            for i in range(r0, r1)}


class Cones:
    """What the lanes form asks of a section's output rows (`Lanes`): the
    rows of every value they need, the work and the registers a column of
    them takes -- over the operators' nonzeros, found once.

    A value is what one step writes: an intermediate written again -- a
    merged run's temporary, reused for every member -- is another value
    each time, and a product reads the one the last step ahead of it wrote
    (`source`).  So the values are keyed by the step that writes them."""

    def __init__(self, section: GroupSection):
        self.section = section
        self.nonzeros = {id(p): operator_nonzeros(p)
                         for s in section.steps for p in s.products
                         if p.kind is Kind.OPERATOR}
        #: The step whose destination a product reads (`operand`), by the
        #: product's `id`: the last one ahead of it to write the tensor, or
        #: None for an element's own operand read from memory.
        self.sources: Dict[int, Optional[Step]] = {}
        last: Dict[int, Step] = {}
        for step in section.steps:
            for p in step.products:
                self.sources[id(p)] = last.get(id(self.operand(p).tensor))
            last[id(step.dest)] = step
        #: `need` of each output row alone, and the works of the ranges a
        #: lane holds (`costs`), found once.
        self._each: Optional[Dict[int, Dict[int, Set[int]]]] = None
        self._costs: Dict[tuple, Dict[Range, int]] = {}

    @staticmethod
    def operand(product: Product):
        """What a product reads at a row of its contraction: an operator
        product's group operand, a mixing product's mixed one."""
        return (product.second if product.kind is Kind.OPERATOR
                else product.first)

    def source(self, product: Product) -> Optional[Step]:
        return self.sources[id(product)]

    @property
    def outputs(self) -> List[Step]:
        """The steps that write the section's output, in order."""
        output = self.section.steps[-1].dest
        return [step for step in self.section.steps if step.dest is output]

    @property
    def rows(self) -> Range:
        """The output's rows its steps write."""
        return _union([step.rows for step in self.outputs])

    @property
    def cols(self) -> Range:
        """The output's columns its steps write."""
        return _union([step.cols for step in self.outputs])

    def need(self, rows) -> Dict[int, Set[int]]:
        """The rows of every value, by the `id` of the step that writes it,
        the output needs at `rows`: an operator product's operand at the
        contraction indices of its nonzeros in the rows it adds to, a mixing
        product's at those rows."""
        steps = self.section.steps
        need: Dict[int, Set[int]] = {id(step): set(rows)
                                     for step in self.outputs}
        for step in reversed(steps):
            mine = need.get(id(step), set())
            for p in step.products:
                source = self.source(p)
                if source is None:
                    continue
                here = {i for i in mine if p.rows[0] <= i < p.rows[1]}
                if p.kind is Kind.OPERATOR:
                    nonzeros = self.nonzeros[id(p)]
                    here = {k for i in here for k, _ in nonzeros.get(i, ())}
                need.setdefault(id(source), set()).update(here)
        return need

    def work(self, rows) -> int:
        """Multiply-adds a column of the output's `rows` takes a lane: each
        operator product's nonzeros at the rows the cone needs, each mixing
        product's contraction there."""
        need = self.need(rows)
        total = 0
        for step in self.section.steps:
            mine = need.get(id(step), set())
            for p in step.products:
                here = [i for i in mine if p.rows[0] <= i < p.rows[1]]
                if p.kind is Kind.OPERATOR:
                    nonzeros = self.nonzeros[id(p)]
                    total += sum(len(nonzeros.get(i, ())) for i in here)
                else:
                    total += len(here) * (p.depth[1] - p.depth[0])
        return total

    def live(self, rows, tile: int = 1, columns: int = 1,
             need: Optional[Dict[int, Set[int]]] = None) -> int:
        """The values a lane holds live for `columns` columns of the
        output's `rows`, in elements: the rows it adds into, and at the
        costliest point either a column some product reads materialized --
        with what its own products hold -- or what the walk over the
        computed entries holds: each mixing product's column of its small
        operand and the rows of the operand it mixes one read brings in
        (`tile`), which the products and the columns share; and a few
        temporaries.  `need`: the rows' cone where it is found already."""
        need = self.need(rows) if need is None else need

        def held(step, own):
            materialized, small, row = 0, 0, 0
            for p in step.products:
                if p.kind is Kind.MIXING:
                    small += p.depth[1] - p.depth[0]
                    row = max(row, p.depth[1] - p.depth[0])
                    continue
                source = self.source(p)
                if source is None:
                    continue
                if source.kind is Kind.OPERATOR:
                    materialized = max(materialized, held(
                        source, len(need.get(id(source), ()))))
                    continue
                small += sum(q.depth[1] - q.depth[0]
                             for q in source.products)
                row = max([row] + [q.depth[1] - q.depth[0]
                                   for q in source.products])
            return (own * columns
                    + max(materialized, small * columns + row * tile))

        return max(held(step, len(rows)) for step in self.outputs) + 4

    def costs(self, limit: Optional[int] = None, tile: int = 1,
              columns: int = 1) -> Dict[Range, int]:
        """The work (`work`) of every range of the output's rows a lane
        holds within `limit` elements (`live`; every range where None):
        each range grown from its first row a row at a time, its cone the
        union of its rows' -- `need` maps row by row -- until a lane no
        longer holds it, as no longer range does."""
        key = (limit, tile, columns)
        if key in self._costs:
            return self._costs[key]
        lo, hi = self.rows
        if self._each is None:
            self._each = {r: self.need((r,)) for r in range(lo, hi)}
        # What a row of each value costs: `work`, row by row.
        weights: Dict[int, Dict[int, int]] = {}
        for step in self.section.steps:
            weight = weights.setdefault(id(step), {})
            for p in step.products:
                for i in range(*p.rows):
                    weight[i] = weight.get(i, 0) + (
                        len(self.nonzeros[id(p)].get(i, ()))
                        if p.kind is Kind.OPERATOR
                        else p.depth[1] - p.depth[0])
        costs: Dict[Range, int] = {}
        for a in range(lo, hi):
            need: Dict[int, Set[int]] = {}
            work = 0
            for b in range(a + 1, hi + 1):
                for value, rows in self._each[b - 1].items():
                    mine = need.setdefault(value, set())
                    weight = weights.get(value, {})
                    work += sum(weight.get(i, 0) for i in rows - mine)
                    mine |= rows
                if (limit is not None and self.live(
                        range(a, b), tile, columns, need) > limit):
                    break
                costs[(a, b)] = work
        self._costs[key] = costs
        return costs

    def ranges(self, count: int, limit: Optional[int] = None,
               tile: int = 1, columns: int = 1
               ) -> Optional[Tuple[Range, ...]]:
        """The output's rows in `count` contiguous ranges a lane holds each
        (`costs`), the ones whose work, all of it and the costliest range's
        added, is least, then the earliest cuts: the block's warps compute
        all of it -- rows of a value two ranges need, each computes, so a
        cut costs least where the cones it parts share least -- and the
        block is done when its costliest range is.  None where no `count`
        ranges are held."""
        costs = self.costs(limit, tile, columns)
        lo, hi = self.rows
        # The rows up to each row in as many ranges as cut so far: every
        # (total, costliest, cuts) no other is below in both.
        best: Dict[int, List[tuple]] = {lo: [(0, 0, ())]}
        for _ in range(count):
            reach: Dict[int, List[tuple]] = {}
            for (a, b), work in costs.items():
                for total, top, cuts in best.get(a, ()):
                    reach.setdefault(b, []).append(
                        (total + work, max(top, work), cuts + (b,)))
            best = {}
            for b, taken in reach.items():
                front, least = [], None
                for t in sorted(taken):
                    if least is None or t[1] < least:
                        front.append(t)
                        least = t[1]
                best[b] = front
        if hi not in best:
            return None
        cuts = (lo,) + min(best[hi], key=lambda t: (t[0] + t[1], t[2]))[2]
        return tuple(zip(cuts, cuts[1:]))


def lanes_plan(section: GroupSection,
               context) -> Tuple[Optional[Lanes], str]:
    """The lanes form's plan for `section` (`Lanes`), or None and why not:
    a chained section's a phase at a time (`_chained_plan`), any other's
    whole (`_lanes_plan`)."""
    if section.chained:
        return _chained_plan(section, context)
    return _lanes_plan(section, context)


def _chained_plan(section: GroupSection,
                  context) -> Tuple[Optional[Lanes], str]:
    """The lanes form's plan of a chained section: every phase's own
    (`_lanes_plan`), each in one warp -- one class of all the output's
    columns, one range of its rows -- so that a column of a phase reads what
    the same lane wrote in the phases before, with no barrier between.  A
    phase that holds no warp's share alone the form does not take yet."""
    plans: List[Optional[Lanes]] = []
    for phase in section.phases():
        if not phase.main:
            plans.append(None)
            continue
        plan, why = _lanes_plan(section.phase_section(phase), context,
                                one_warp=True)
        if plan is None:
            return None, (f'the phase into {phase.main[-1].dest.alias}: '
                          f'{why}')
        plans.append(plan)
    taken = [p for p in plans if p is not None]
    if not taken:
        return None, 'a chained section of copies alone'
    return Lanes(1, taken[0].ranges, max(p.registers for p in taken), 1,
                 tuple(plans)), ''


def _lanes_plan(section: GroupSection, context,
                one_warp: bool = False) -> Tuple[Optional[Lanes], str]:
    """The lanes form's plan of a section of one output (`Lanes`), or
    None and why not; where `one_warp`, in one class and one range or none.

    It takes a section with one output, the last step's destination, which
    no other step reads; whose mixing products mix the element's own
    operands and not an intermediate -- which a lane would have to hold
    whole, every column of it -- and whose operators carry numbers, their
    pattern, at most `Options.inline_constants` nonzeros each where those
    are literals (`lane_literals`) and the target caps them
    (`lane_literal_cap`).  An intermediate
    written again is another value each time (`Cones`), and the steps
    into the output -- a merged run's, one a member -- add into one column
    of registers in turn.  The output is cut into as few row ranges as
    keep the estimate of a lane's registers within the target's budget
    (`lane_registers`), where their work adds up to the least
    (`Cones.ranges`), and has as many column classes as `LANE_CLASSES`
    allows within `LANE_WARPS` a block."""
    if len(section.outputs) != 1:
        return None, (f'the lanes form writes one output, not '
                      f'{len(section.outputs)}')
    for step in section.steps:
        for p in step.products:
            if p.kind is Kind.MIXING:
                if p.first.tensor.is_tmp:
                    return None, (f'a mixing product of {p.first.tensor.alias}'
                                  f', which a lane would hold whole')
                continue
            tensor = p.first.tensor
            if not tensor.has_values():
                return None, (f'{tensor.alias} carries no numbers, whose '
                              f'nonzeros the lanes form would walk')
    cones = Cones(section)
    most = lane_literal_cap(context)
    for step in section.steps:
        for p in step.products:
            if (p.kind is Kind.MIXING or not lane_literals(context)
                    or most is None):
                continue
            tensor = p.first.tensor
            count = sum(len(v) for v in cones.nonzeros[id(p)].values())
            if count > most:
                return None, (f'{tensor.alias} has {count} nonzeros, more '
                              f'than the {most} literals an operator may '
                              f'take (`inline_constants`)')
    size = 2 if section.dtype is Datatype.F64 else 1
    # A lane reads its operands' rows 16 bytes at a time
    # (`compute.multi_element._LaneTrip.own`).
    tile = max(1, 16 // section.dtype.size())
    lo, hi = cones.rows
    width = cones.cols[1] - cones.cols[0]
    classes = max(1, min(LANE_CLASSES, width))
    budget = lane_registers(context, lane_elements(context))
    if one_warp:
        classes = 1
    # The most columns a pass takes first, each at as few row ranges as
    # hold it; a pass of several only where they divide the class's
    # columns and every class has as many.
    for columns in range(1 if one_warp else lane_columns(context), 0, -1):
        for count in range(1, 2 if one_warp else
                           min(LANE_RANGES, hi - lo) + 1):
            kinds = max(1, min(classes, LANE_WARPS // count))
            trips = -(-width // kinds)
            if columns > 1 and (trips % columns or width % kinds):
                continue
            ranges = cones.ranges(count, budget // size, tile, columns)
            if ranges is not None:
                live = max(cones.live(range(*r), tile, columns)
                           for r in ranges) * size
                return Lanes(kinds, ranges, live, columns), ''
    count = 1 if one_warp else min(LANE_RANGES, hi - lo)
    live = max(cones.live(range(*r), tile) for r in cones.ranges(
        count, None, tile)) * size
    return None, (f'a lane would hold {live} registers at {count} row '
                  f'ranges, more than the {budget} the lanes form takes '
                  f'here')


def fill(section: GroupSection, context,
         form: Optional[str] = None) -> Tuple[Optional[ElementGroup], str]:
    """The fill rule: the group's size for `section` on the context's
    target, or None and why not.

    Only on the targets the layout is written for (`TARGETS`).  The lanes
    form where it is asked for -- `form`, else the option's (`fma_form`) --
    and the section fits it (`lanes_plan`): a group of a warp's lanes
    (`lane_elements`), an element each.
    Otherwise the size is the table's (`table_elements`) where the block
    holds it, halved while it does not -- in shared memory, in threads --
    and `MIN_ELEMENTS` the last size tried.  Each step's lowering is the
    target's (`lower`): the matrix tiling, its inputs in a second stage, A
    in TF32 halves in as many of its buffers as fit (where `HALVES` is
    on), and B in TF32 halves, at the sizes where those fit, in that order
    of preference; where no group of the tiling fits -- its padded stages
    take room the FMA forms do not -- the FMA forms'.  The group carries
    the section lowered."""
    target = context.target
    hw = target.hw
    if (target.backend, getattr(hw, 'vendor', None)) not in TARGETS:
        return None, (f'the multi-element layout is written for CUDA on '
                      f'NVIDIA, HIP on AMD and DPC++ under SPMD on Intel, '
                      f'not {target.backend} on {hw.vendor}')
    options = context.get_user_options()
    # The batch loop's `Trip.ELEMENT_GROUP` takes neither.
    if options.launch_control:
        return None, ('the launch-control work queue hands out elements, '
                      'not groups')
    if int(options.stack_elements) > 1:
        return None, (f'a trip of {options.stack_elements} stacked elements '
                      f'is no element group')
    if (form or fma_form(context)) == 'lanes':
        lanes, why = lanes_plan(section, context)
        if lanes is not None:
            return ElementGroup(lane_elements(context),
                                lower(section, context, lanes=True), 0,
                                lanes), ''
        if section.chained:
            return None, why
    if section.chained:
        # The warp form stages an element's operands ahead of its steps and
        # writes an output's tiles whole: neither reads back an output, nor
        # leaves the rest of an assigned window as it is.
        return None, ('a chained section is the lanes form\'s alone (its '
                      'phases one after another, `GroupSection.phases`)')
    budget = hw.max_local_mem_size_per_block - SHARED_RESERVE
    # The matrix tiling, with B in TF32 halves where that costs no group
    # size -- the halves take twice B's room -- then the FMA forms.  An
    # asynchronous engine's tiling is followed by mma_sync's, for a group
    # it does not take.  Its halves prepared at generation where the
    # options ask (`Options.matrix_ring`) and the operators carry their
    # numbers: resident ones copied once, streamed ones through the
    # prepared ring in place of the converting one.
    asked = ring_asked(context)
    star = star_asked(context)
    output = output_asked(context)
    tilings = [lower(section, context, split=True, prepared=asked,
                     star=star, output=output)]
    if not tiled(tilings[0]):
        tilings = []
    elif split(tilings[0]):
        if asynchronous(tilings[0]):
            # B's halves streamed through a ring ahead of B whole.
            tilings.append(lower(section, context, split=True,
                                 streamed=True, prepared=asked, star=star,
                                 output=output))
            # Where the star step folded, the same tilings without the fold
            # behind them: B's halves before the fold, where only the
            # unfolded ones leave the group room -- B whole is B in TF32
            # on these engines (`multi_element._terms`), which the fold
            # does not take (`fuse_refusal`).
            tilings += [lower(section, context, split=True,
                              streamed=streamed(t), prepared=asked,
                              output=output)
                        for t in list(tilings) if t.star]
        if not asynchronous(tilings[0]) or tf32_operands(context):
            # B whole: mma_sync splits it where it reads it; the
            # asynchronous engines read it in TF32 alone, which only the
            # option lets a plan do (`tf32_refusal`).
            tilings.append(lower(section, context, star=star,
                                 output=output))
    candidates = [(tilings, None)] if tilings else []
    if tilings and asynchronous(tilings[0]):
        held = [lower(section, context, split=True, engine='mma_sync',
                      output=output)]
        if split(held[0]):
            held.append(lower(section, context, engine='mma_sync',
                              output=output))
        candidates.append((held, 'mma_sync'))
    candidates.append(([lower(section, context, matrix=False)], None))
    # Each again with the inputs in one stage, where the options ask for a
    # second: a group the second stage would halve is worth more than it.
    for lowerings, engine in candidates:
        lowerings += [lower(section, context, matrix=tiled(t),
                            split=split(t), engine=engine,
                            streamed=streamed(t), double=False,
                            prepared=prepared(t), star=t.star,
                            output=output)
                      for t in lowerings if t.doubled]
        # Each tiling on mma_sync again with A in TF32 halves (`_halved`)
        # where the layout's switch takes them (`HALVES`), in as many of its
        # buffers as fit first -- the walk reads an A tile for every tile of
        # B it keeps an M group over, so a buffer's halves spare many more
        # splits than B's -- and at each count B's halves first; the second
        # stage, where the options ask for it, before either.  An
        # asynchronous engine's where `Options.matrix_star` asks for them
        # (`_halves_order`: a folded star keeps its place, and a tiling the
        # fold was refused for takes the halves there).
        if tiled(lowerings[0]) and (
                star_asked(context) in ('halves', 'engine')
                if asynchronous(lowerings[0]) else HALVES):
            lowerings[:] = _halves_order(lowerings)
        # Ahead of all of them, the first through a producer's ring, the
        # deepest that fits first: the producer's trips run ahead of the
        # element warps' (`producer_refusal`).
        first = lowerings[0]
        keep = int(getattr(context.get_user_options(), 'element_registers',
                           0) or 0)
        rings = [lower(section, context, matrix=tiled(first),
                       split=split(first), engine=engine,
                       streamed=streamed(first), ring=slots, registers=keep,
                       prepared=prepared(first), star=first.star,
                       output=output)
                 for slots in PRODUCER_SLOTS]
        lowerings[:0] = [r for r in rings if r.ring]
        # Ahead of them, the block in its three roles where the options ask
        # and the section takes them (`roles_refusal`): B streamed through
        # the prepared ring, which the operator role fills.
        if roles_asked(context) != 'off' and asynchronous(first):
            roled = lower(section, context, split=True, streamed=True,
                          prepared=ring_asked(context), star=first.star,
                          roles=roles_asked(context))
            lowerings[:0] = [roled] if roled.roles else []
        # Each with D added by bulk reductions where the options ask
        # (`Options.matrix_output`), then as it was: the staged columns
        # take room the reductions a cell do not, and where they would cost
        # the group its size -- or another of its lowering's choices --
        # the group keeps them, and its comment says why (`output_notes`).
        lowerings[:] = [twin for t in lowerings
                        for twin in _output_twins(t)]
        # No plan that reads an operand of an F32 section in TF32 alone,
        # unless the options let it (`tf32_refusal`).
        lowerings[:] = [t for t in lowerings
                        if tf32_refusal(t, context) is None]
    candidates = [(lowerings, engine) for lowerings, engine in candidates
                  if lowerings]

    def first_fit(lowerings):
        elements = table_elements(lowerings[0])
        while True:
            for lowered in lowerings:
                # The asynchronous engines' own sizes (`engine_elements`).
                taken = engine_elements(lowered, elements)
                if taken is None:
                    continue
                if lowered.registers:
                    # Whole warpgroups of element warps, as `setmaxnreg`
                    # moves a warpgroup's registers.
                    taken -= taken % WARPGROUP_WARPS
                    if not taken:
                        continue
                need = shared_bytes(lowered, taken)
                warps = taken + (0 if not lowered.ring else WARPGROUP_WARPS
                                 if lowered.registers else PRODUCER_WARPS)
                if lowered.roles:
                    warps += OPERATOR_WARPS + MMA_WARPS
                if (need <= budget
                        and 32 * warps <= hw.max_threads_per_block):
                    return ElementGroup(taken, lowered, need)
            if elements <= MIN_ELEMENTS:
                return None
            elements = max(MIN_ELEMENTS, elements // 2)

    for i, (lowerings, engine) in enumerate(candidates):
        group = first_fit(lowerings)
        if group is None:
            continue
        held = (candidates[i + 1] if i + 1 < len(candidates)
                and candidates[i + 1][1] == 'mma_sync' else None)
        if engine is None and held is not None \
                and asynchronous(group.section):
            # The target's asynchronous engine with every operand in both
            # halves, against mma_sync, which splits B where it reads it:
            # the larger group (`prefer_mma_sync`, the cost model's to
            # replace), up to the engine's own lane limit.
            other = first_fit(held[0])
            # The most the engine takes of the section in the lowering the
            # fill rule chose -- its lanes, and whole warpgroups where a
            # producer moves registers (`Options.element_registers`).
            chosen = group.section
            cap = engine_elements(chosen, table_elements(chosen)) or 0
            if chosen.registers:
                cap -= cap % WARPGROUP_WARPS
            if other is not None and prefer_mma_sync(
                    group.elements, other.elements, cap):
                engines = sorted({s.lowering for s in group.section.steps
                                  if s.lowering in ASYNC_LOWERINGS})
                note = (f'{"/".join(engines)} declined: with every operand '
                        f'in both TF32 halves it holds a group of '
                        f'{group.elements}, mma_sync -- B split where it '
                        f'reads it -- {other.elements} (prefer_mma_sync)')
                if not tf32_operands(context):
                    whole = first_fit([lower(section, context, star=star,
                                             output=output)])
                    if whole is not None and whole.elements > \
                            group.elements:
                        note += (f'; B read whole, in TF32 alone, would '
                                 f'hold {whole.elements} '
                                 f'(Options.matrix_tf32_operands)')
                return dataclasses.replace(other, section=dataclasses.replace(
                    other.section, engine_note=note)), ''
        return group, ''
    candidates = [lowerings for lowerings, _ in candidates]
    return None, (f'the operators and a group of {MIN_ELEMENTS} take '
                  f'{shared_bytes(candidates[-1][0], MIN_ELEMENTS)} B of '
                  f'shared memory, more than the {budget} B a block has: the '
                  f'operator needs to be streamed')


#: Where `Options.multi_element` `auto` takes the layout, by vendor: the
#: element types of the sections it takes, in the lanes form, where its plan
#: holds the output in one row range (`auto_takes`).  Measured on PVC (Max
#: 1550, SNG2 563442 and 563448; the lanes form against the per-element
#: kernel, the time an element): at F32 the volume -41 % at o4 and o6 (-46
#: and -35 % with the stars stored sparse, as SeisSol's GPU builds store
#: them), localFlux[0] -4 and -21 %, neighboringFlux[0] -1 and -16 %,
#: localFluxAll -45 % at o4 -- and +281 % at o6, where it spilled before
#: each walk took an anchor of its own; at F64 the o6 volume +44 % and the
#: fluxes +12 to +20 % under the plans before the ranges were cut for the
#: least work.  Elsewhere nothing, until the measurements say otherwise.
AUTO_LANES: Dict[str, Tuple[Datatype, ...]] = {'intel': (Datatype.F32,)}


def auto_takes(section: GroupSection,
               context) -> Tuple[Optional[Lanes], str]:
    """The lanes form's plan of `section` where `Options.multi_element`
    `auto` takes it on the context's target, or None and why not: where
    the target's vendor takes the section's element type (`AUTO_LANES`),
    the lanes form is not declined (`Options.multi_element_fma` `warp`),
    and the plan holds the output in one row range -- where it cuts more,
    each range computes again the rows of a value the others need too,
    which nothing has measured to pay yet."""
    vendor = getattr(context.target.hw, 'vendor', None)
    if section.chained:
        return None, ('`auto` takes no chained section: none is measured '
                      'yet')
    if section.dtype not in AUTO_LANES.get(vendor, ()):
        return None, (f'`auto` takes no {section.dtype.name} section on '
                      f'{vendor} (`AUTO_LANES`)')
    if context.get_user_options().multi_element_fma == 'warp':
        return None, ('`auto` takes the lanes form, and the warp form is '
                      'asked for')
    lanes, why = lanes_plan(section, context)
    if lanes is None:
        return None, why
    if len(lanes.ranges) > 1:
        return None, (f'the lanes form cuts the output into '
                      f'{len(lanes.ranges)} row ranges; `auto` takes one')
    return lanes, ''


def plan(descrs: Sequence, context) -> Optional[ElementGroup]:
    """The section's multi-element plan, or None: where
    `Options.multi_element` is `force`, every product classifies and the
    fill rule finds a group; where `auto`, the lanes form's group where
    `auto_takes` takes the section."""
    mode = context.get_user_options().multi_element
    if mode not in ('force', 'auto'):
        return None
    section, _ = classify(descrs)
    if section is None:
        return None
    if mode == 'auto':
        if auto_takes(section, context)[0] is None:
            return None
        group, _ = fill(section, context, form='lanes')
        return group
    group, _ = fill(section, context)
    return group


__all__ = ['ASYNC_LOWERINGS', 'ASYNC_N', 'AUTO_LANES', 'Buffer', 'Cones',
           'DEFAULT_ELEMENTS', 'ELEMENTS', 'ElementGroup', 'FMA_FORMS',
           'GroupSection', 'HALVES', 'Kind', 'LANE_ANCHORS',
           'LANE_CLASSES',
           'LANE_COLUMNS', 'LANE_FEATURES', 'LANE_LITERALS', 'LANE_RANGES',
           'LANE_REGISTERS', 'LANE_VALUES', 'LANE_WARPS', 'LOWERING', 'Lanes',
           'MATRIX', 'MATRIX_LOWERINGS', 'MIN_ELEMENTS', 'POINTER_BYTES',
           'HALF_READERS', 'Product', 'RING_CHUNK', 'RING_SLOTS', 'Role',
           'STAR_ROWS', 'STAR_SLOTS', 'folded', 'fuse', 'fuse_refusal',
           'TF32_HALVES_BY_WHOLE', 'async_plan', 'async_terms',
           'precision_notes',
           'prefer_mma_sync',
           'tf32_operands', 'tf32_refusal',
           'output_asked', 'output_geometry', 'output_notes', 'output_of',
           'output_refusal', 'output_volume',
           'star_mixes', 'star_notes', 'star_volume',
           'SHARED_RESERVE',
           'SMALL_EXTENT', 'Stage', 'Step', 'TARGETS', 'TCGEN05_ROWS',
           'TILE_COSTS', 'TILE_LATENCY', 'TILE_REGISTERS', 'TILE_SLOTS',
           'TILE_SPLIT', 'TILE_UNIT', 'Terms', 'TileUnits', 'WARPGROUP_WARPS',
           'async_staging', 'asynchronous', 'auto_takes', 'classify',
           'engine_elements', 'fill', 'fma_form', 'halvable', 'halves',
           'kept_tiles',
           'lane_anchors', 'lane_columns', 'lane_elements', 'lane_features',
           'lane_grf', 'lane_literal_cap', 'lane_literals', 'lane_registers',
           'lane_values', 'lanes_plan', 'lower', 'matrix_enabled',
           'matrix_shape', 'nonzero_terms', 'operator_nonzeros', 'plan',
           'PRODUCER_SLOTS', 'PRODUCER_WARPS', 'bulk_input', 'bulk_rows',
           'chunk_volume', 'prepared', 'prepared_table',
           'producer_refusal', 'register_budgets', 'ring_asked', 'star_asked',
           'ring_chunks', 'ring_notes', 'ring_volume',
           'second_stages', 'shared_bytes', 'SM_REGISTERS', 'split',
           'step_refusal', 'streamed', 'table_elements', 'tile_units',
           'tiled']
