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
    rows are 16 bytes (`_pitched`); `pitch` is the distance either way.
    The elements past the rows are padding, zero."""

    tensor: object
    role: Role
    rows: Range
    cols: Range
    stride: int = 0

    @property
    def height(self) -> int:
        return self.rows[1] - self.rows[0]

    @property
    def pitch(self) -> int:
        return self.stride or self.height

    @property
    def per_element(self) -> int:
        return self.pitch * (self.cols[1] - self.cols[0])


@dataclass(frozen=True)
class Stage:
    """A batch-constant operator in the block's shared memory, staged once a
    block: as stored (`window` None), for the FMA forms, or as one matrix
    product's B.

    B is the product's window of the operator, transposed onto the
    instruction's axes -- `B(k, n) = K(n0 + n, k0 + k)`, `k0` the window's
    first column, `n0` the step's first row -- zero wherever the product
    has no entry, the contraction padded to `depth` (a multiple of the
    instruction's `k`) and the step's rows to `width` (of its n, 8).  It is
    stored in the order the instruction's fragments are read: tile after
    tile of `k` x 8, K outer, and in a tile lane after lane, each lane's
    registers of B side by side (`regs`) -- one load a lane and fragment.
    `split` stores each value as its two TF32 halves, the upper ones first,
    which the 3xTF32 products then read prepared (`fill`)."""

    tensor: object
    window: Optional[Tuple[Range, Range]] = None
    n0: int = 0
    depth: int = 0
    width: int = 0
    k: int = 0
    split: bool = False

    @property
    def regs(self) -> int:
        """A lane's registers of B a fragment, both halves where split."""
        return (self.k // 4) * (2 if self.split else 1)

    @property
    def volume(self) -> int:
        if self.window is None:
            return int(self.tensor.storage_volume())
        return self.depth * self.width * (2 if self.split else 1)


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
        copies of what no step writes first, where there are such; then each
        run of steps into an output of the element -- the mixing steps and
        the operator steps that read them -- with the scaled copies of that
        output after it.  An output is whole when its phase ends: a later
        phase reads it from where it was written."""
        first, runs, run = [], [], []
        outputs: Dict[int, int] = {}
        for step in self.steps:
            if step.kind is Kind.SCALE:
                continue
            run.append(step)
            if not step.dest.is_tmp:
                outputs[id(step.dest)] = len(runs)
                runs.append([tuple(run), []])
                run = []
        if run:
            runs.append([tuple(run), []])
        for step in self.steps:
            if step.kind is not Kind.SCALE:
                continue
            source = id(step.products[0].first.tensor)
            if source in outputs:
                runs[outputs[source]][1].append(step)
            else:
                first.append(step)
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
                      s.terms)
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
        """A warp's lanes: 32, or in the lanes form the group's elements, a
        lane each (`lane_elements`)."""
        return self.elements if self.lanes is not None else 32

    @property
    def warps(self) -> int:
        """The block's warps: one an element, or the lanes form's shares."""
        return self.elements if self.lanes is None else self.lanes.warps


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


MATRIX_LOWERINGS = ('mma_sync', 'dmma')


def matrix_shape(lowering: str, sm: int) -> Tuple[int, int, int]:
    """`(m, n, k)` of the instruction a matrix step issues: `m16n8k8` over
    TF32 halves for F32, and for F64 `m16n8k4` from sm_90 on, `m8n8k4`
    before."""
    if lowering == 'mma_sync':
        return 16, 8, 8
    return (16, 8, 4) if sm >= 90 else (8, 8, 4)


def _round(value: int, step: int) -> int:
    return -(-value // step) * step


def _staged(section: GroupSection, sm: int,
            split: bool = False) -> GroupSection:
    """The section with its operators' stages, for the steps' lowerings:
    an operator as stored where an FMA form reads it, a B a window where a
    matrix step does -- in TF32 halves where `split` asks."""
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
            add(Stage(t, (p.rows, p.depth), step.rows[0],
                      _round(p.depth[1] - p.depth[0], k),
                      _round(step.rows[1] - step.rows[0], n), k,
                      split and step.lowering == 'mma_sync'))
    return dataclasses.replace(section, stages=tuple(stages))


def _pitched(section: GroupSection) -> GroupSection:
    """The section with a pitch for every buffer an F32 matrix step reads A
    from: ldmatrix takes a fragment's rows 16 bytes at a time
    (`compute.multi_element`), so a column of the buffer -- a row of A --
    starts on a multiple of four floats, and an odd number of 16-byte units
    from the next (four more than a multiple of eight floats), which puts a
    fragment's eight rows on eight different bank groups.  Long enough for
    every step of K a product reads, the padding zero, so that the padded
    end of the contraction multiplies zeros."""
    need: Dict[int, int] = {}
    for step in section.steps:
        if step.lowering != 'mma_sync':
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
        return dataclasses.replace(buffer, stride=n + (4 - n % 8) % 8)

    return dataclasses.replace(
        section, inputs=tuple(map(pitched, section.inputs)),
        intermediates=tuple(map(pitched, section.intermediates)))


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


def _lowering(step: Step, dtype, context, sm: int, matrix: bool) -> str:
    """A step's lowering on the context's target: an operator step on the
    matrix engines where they are on (`matrix`, `matrix_enabled`) and the
    target has them (sm_80 on) and its products share the step's columns --
    an element's rows, which the tiling takes whole -- its kind's FMA form
    elsewhere."""
    if step.kind is not Kind.OPERATOR:
        return LOWERING[step.kind]
    if (not matrix or not matrix_enabled(context) or sm < 80
            or any(p.cols != step.cols for p in step.products)):
        return LOWERING[step.kind]
    return 'mma_sync' if dtype is Datatype.F32 else 'dmma'


def lower(section: GroupSection, context, matrix: bool = True,
          split: bool = False, lanes: bool = False,
          double: bool = True) -> GroupSection:
    """The section with each step's lowering for the context's target --
    the FMA forms only, where `matrix` is off; the lanes form's `lane` for
    every step, where `lanes` is on -- and the stages those lowerings read,
    B in TF32 halves where `split` asks, the inputs a trip ahead in a
    second stage where the options take them there (`second_stages`) and
    `double` does not decline it.  An `fma` step reads only its operators'
    nonzeros (`nonzero_terms`) where they carry their numbers and
    `Options.skip_known_zeros` is on, as the per-element products do.  The
    matrix tiling is NVIDIA's: elsewhere every step takes an FMA form, as
    below sm_80."""
    from tensorforge.backend.instructions.compute.primitives import nvidia
    sm = (nvidia.sm_of(context)
          if getattr(context.target.hw, 'vendor', None) == 'nvidia' else 0)
    sparse = context.get_user_options().skip_known_zeros
    steps = []
    for s in section.steps:
        lowering = ('lane' if lanes
                    else _lowering(s, section.dtype, context, sm, matrix))
        steps.append(dataclasses.replace(
            s, lowering=lowering,
            terms=nonzero_terms(s) if sparse and lowering == 'fma'
            else None))
    staged = _staged(dataclasses.replace(section, steps=tuple(steps)), sm,
                     split)
    return _pitched(dataclasses.replace(
        staged, doubled=second_stages(staged, context)
        if double and not lanes else 0))


def async_staging(context) -> bool:
    """Whether a trip stages its dense operands by asynchronous copies,
    which `pir.wrap` can take a trip ahead: NVIDIA's `cp.async`
    (`Target.async_copy_path`).  AMD's LDS-DMA writes a wave's lanes past
    one base, which a box over its pitch is not to the emitter; there, as
    where there is no such path, a trip stages by loads and stores."""
    return (getattr(context.target.hw, 'vendor', None) == 'nvidia'
            and context.target.async_copy_path())


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


def tiled(section: GroupSection) -> bool:
    """Whether a step of the section is lowered on the matrix engines."""
    return any(s.lowering in MATRIX_LOWERINGS for s in section.steps)


def split(section: GroupSection) -> bool:
    """Whether the section's B stages hold TF32 halves."""
    return any(st.split for st in section.stages)


def shared_bytes(section: GroupSection, elements: int) -> int:
    """Shared memory a group of `elements` takes: the operators as staged
    -- their stages, and the terms of the steps that walk them -- each
    element's copy of every input and intermediate, its second copy of
    every input staged a trip ahead (`GroupSection.doubled`), and its entry
    in every pointer table."""
    size = section.dtype.size()
    operators = (sum(st.volume for st in section.stages)
                 + sum(s.terms.volume for s in section.steps
                       if s.terms is not None))
    per_element = sum(b.per_element for b in section.inputs
                      + section.intermediates
                      + section.inputs[:section.doubled])
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
    target's (`lower`): the matrix tiling, its inputs in a second stage and
    B in TF32 halves at the sizes where those fit, the second stage first;
    where no group of the tiling fits -- its padded stages take room the
    FMA forms do not -- the FMA forms'.  The group carries the section
    lowered."""
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
    # size -- the halves take twice B's room -- then the FMA forms.
    tilings = [lower(section, context, split=True)]
    if not tiled(tilings[0]):
        tilings = []
    elif split(tilings[0]):
        tilings.append(lower(section, context))
    forms = [lower(section, context, matrix=False)]
    # Each again with the inputs in one stage, where the options ask for a
    # second: a group the second stage would halve is worth more than it.
    for lowerings in (tilings, forms):
        lowerings += [lower(section, context, matrix=tiled(t),
                            split=split(t), double=False)
                      for t in lowerings if t.doubled]
    candidates = [tilings] if tilings else []
    candidates.append(forms)
    for lowerings in candidates:
        elements = table_elements(lowerings[0])
        while True:
            for lowered in lowerings:
                need = shared_bytes(lowered, elements)
                if (need <= budget
                        and 32 * elements <= hw.max_threads_per_block):
                    return ElementGroup(elements, lowered, need), ''
            if elements <= MIN_ELEMENTS:
                break
            elements = max(MIN_ELEMENTS, elements // 2)
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


__all__ = ['AUTO_LANES', 'Buffer', 'Cones', 'DEFAULT_ELEMENTS', 'ELEMENTS',
           'ElementGroup', 'FMA_FORMS', 'GroupSection', 'Kind',
           'LANE_ANCHORS', 'LANE_CLASSES', 'LANE_COLUMNS', 'LANE_FEATURES',
           'LANE_LITERALS', 'LANE_RANGES', 'LANE_REGISTERS', 'LANE_VALUES',
           'LANE_WARPS', 'LOWERING', 'Lanes', 'MATRIX', 'MATRIX_LOWERINGS',
           'MIN_ELEMENTS', 'POINTER_BYTES', 'Product', 'Role',
           'SHARED_RESERVE', 'SMALL_EXTENT', 'Stage', 'Step', 'TARGETS',
           'Terms', 'async_staging', 'auto_takes', 'classify', 'fill',
           'fma_form', 'lane_anchors', 'lane_columns', 'lane_elements',
           'lane_features', 'lane_grf', 'lane_literal_cap', 'lane_literals',
           'lane_registers', 'lane_values', 'lanes_plan', 'lower',
           'matrix_enabled', 'matrix_shape', 'nonzero_terms',
           'operator_nonzeros', 'plan', 'second_stages', 'shared_bytes',
           'split', 'table_elements', 'tiled']
