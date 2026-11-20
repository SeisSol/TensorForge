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
  engines are off (`matrix_enabled`).
* `MIXING`: a group operand times a small one, `C = G x S`, contracting
  the quantities: each element's rows mixed by its own operand, on the FMA
  units, a lane an output.

Products into one destination with nothing between them that reads it
are one `Step`, lowered together: the matrix tiling concatenates them
along K and keeps D in registers across them, the FMA forms sum them per
output.  The matrix tiling reads each product's operator as a `Stage` --
its window, zero elsewhere, padded to the instruction's tiles -- and
hands D to its consumer's buffer, or to the output's (`results`), which
each warp then adds into its element's output.

`plan` decides whether and how a section takes the layout: every product
classified or none, E for the section, and each step's lowering -- the
fill rule (`fill`), static so far, each refusal with its reason.

`multi_element` is off by default.  `force` takes the layout wherever the
rule finds a plan; `auto` where it finds one measured to pay, which nothing
is yet -- so `auto` is `off` until the measurements say otherwise.
"""

from __future__ import annotations

import dataclasses
import enum
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

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

#: The layout's deployment switch for the matrix engines: an operator step
#: is tiled on them where `Options.tensor_cores` asks for it, or where the
#: option is not set and this is on.  On: the layout exists for them, and is
#: off by default itself (`Options.multi_element`).
MATRIX = True


class Role(enum.Enum):
    """What an element of the batch has of an operand (see the module)."""

    OPERATOR = 'operator'
    GROUP = 'group'
    SMALL = 'small'
    INTERMEDIATE = 'intermediate'


class Kind(enum.Enum):
    """A product, by its operands (see the module)."""

    OPERATOR = 'operator'
    MIXING = 'mixing'


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
    def other(self):
        """The operator, or the small operand."""
        return self.first if self.kind is Kind.OPERATOR else self.second


@dataclass(frozen=True)
class Step:
    """Products into one destination, lowered together (see the module).

    The first may assign; every other accumulates.  `rows` and `cols` are
    the union of the products' windows of the destination, each product
    adding only into its own.  `lowering` is the fill rule's: `fma` for an
    operator product so far, `rows` for a mixing one."""

    products: Tuple[Product, ...]
    lowering: str

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
        return [t for p in self.products
                for t in (p.first.tensor, p.second.tensor)]


@dataclass(frozen=True)
class Buffer:
    """The group's copy of one operand in shared memory: per element, the
    box `rows` x `cols` of it, column by column -- an element's rows of a
    group operand side by side, its basis contiguous -- and the elements
    one after another."""

    tensor: object
    role: Role
    rows: Range
    cols: Range

    @property
    def per_element(self) -> int:
        return ((self.rows[1] - self.rows[0])
                * (self.cols[1] - self.cols[0]))


@dataclass(frozen=True)
class Stage:
    """A batch-constant operator in the block's shared memory, staged once a
    block: as stored (`window` None), for the FMA forms, or as one matrix
    product's B.

    B is the product's window of the operator, transposed onto the
    instruction's axes -- `B(k, n) = K(n0 + n, k0 + k)`, `k0` the window's
    first column, `n0` the step's first row -- zero wherever the product
    has no entry, `depth` rows of `ld` columns: the contraction padded to
    the instruction's depth, the step's rows to its width and the row length
    to a stride whose fragment reads meet no bank twice."""

    tensor: object
    window: Optional[Tuple[Range, Range]] = None
    n0: int = 0
    depth: int = 0
    ld: int = 0

    @property
    def volume(self) -> int:
        if self.window is None:
            return int(self.tensor.storage_volume())
        return self.depth * self.ld


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
    #: The operators as the block stages them (`Stage`), and the buffers the
    #: matrix steps hand an output's D to (`lower`).
    stages: Tuple[Stage, ...] = ()
    results: Tuple[Buffer, ...] = ()

    @property
    def products(self) -> List[Product]:
        return [p for s in self.steps for p in s.products]

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

    def result(self, tensor) -> Buffer:
        """The buffer a matrix step hands the output `tensor`'s D to."""
        for b in self.results:
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
                      tuple((p.first.tensor.alias, p.second.tensor.alias,
                             p.rows, p.cols, p.depth) for p in s.products))
                     for s in self.steps)


@dataclass(frozen=True)
class ElementGroup:
    """How a section takes the multi-element layout: E elements a block,
    one warp each, and the section as it runs (`GroupSection`, each step
    with its lowering)."""

    elements: int
    section: GroupSection
    shared_bytes: int

    @property
    def threads(self) -> int:
        """Lanes of one multiplication: a warp."""
        return 32


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
    for n, op in enumerate(ops):
        if not isinstance(op, MultilinearDescr):
            return None, (f'operation {n} is a {type(op).__name__}, which '
                          f'the layout does not lower yet')
        if op.guarded():
            return None, f'operation {n} is guarded'
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
                return None, (f'operation {n} assigns a window of '
                              f'{dest.alias}, whose rest it would have to '
                              f'zero')
        products.append(Product(op, kind, first, second, op.dest, op.add))

    # An element's own tensor is read or written in the section, not both.
    written = {id(p.dest.tensor) for p in products}
    for p in products:
        for sub in (p.first, p.second):
            t = sub.tensor
            if not t.is_tmp and id(t) in written:
                return None, (f'{t.alias} is written and read in the '
                              f'section')
    # Intermediates: assigned before anything reads or adds to them, and
    # read only where written.
    assigned: Dict[int, List[Tuple[Range, Range]]] = {}
    for n, p in enumerate(products):
        for sub in (p.first, p.second):
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
LOWERING = {Kind.OPERATOR: 'fma', Kind.MIXING: 'rows'}


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
        reads = {id(p.first.tensor), id(p.second.tensor)}
        dest = id(p.dest.tensor)
        for members in list(open_steps):
            read_by = {id(t) for m in members
                       for t in (m.first.tensor, m.second.tensor)}
            if id(members[0].dest.tensor) in reads or dest in read_by:
                close(members)
        into = [m for m in open_steps if id(m[0].dest.tensor) == dest]
        if into and p.add and p.kind is into[0][0].kind:
            into[0].append(p)
            continue
        for members in into:
            close(members)
        open_steps.append([p])
    for members in list(open_steps):
        close(members)
    return tuple(out)


def _section(steps: Tuple[Step, ...], dtype) -> GroupSection:
    operators, outputs = [], []
    boxes: Dict[int, Tuple[object, Role, List[Range], List[Range]]] = {}

    def keep(tensor, role, sub):
        entry = boxes.setdefault(id(tensor), (tensor, role, [], []))
        entry[2].append(_range(sub, 0))
        entry[3].append(_range(sub, 1))

    for p in (p for s in steps for p in s.products):
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
                           tuple(intermediates), tuple(outputs), dtype)
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


def _staged(section: GroupSection, sm: int) -> GroupSection:
    """The section with its operators' stages and its outputs' result
    buffers, for the steps' lowerings: an operator as stored where an FMA
    form reads it, a B a window where a matrix step does."""
    stages: List[Stage] = []
    results: Dict[int, Tuple[object, List[Range], List[Range]]] = {}

    def add(stage):
        # By the tensor's identity: two operators alike in shape compare
        # equal as tensors.
        key = (id(stage.tensor), stage.window, stage.n0)
        if not any((id(s.tensor), s.window, s.n0) == key for s in stages):
            stages.append(stage)

    for step in section.steps:
        matrix = step.lowering in MATRIX_LOWERINGS
        for p in step.products:
            t = p.first.tensor if p.kind is Kind.OPERATOR else None
            if t is None:
                continue
            if not matrix:
                add(Stage(t))
                continue
            _, n, k = matrix_shape(step.lowering, sm)
            width = _round(step.rows[1] - step.rows[0], n)
            if section.dtype is Datatype.F64:
                ld = width + 4
            else:
                ld = width + (8 if width % 32 in (0, 16) else 0)
            add(Stage(t, (p.rows, p.depth), step.rows[0],
                      _round(p.depth[1] - p.depth[0], k), ld))
        if matrix and not step.dest.is_tmp:
            entry = results.setdefault(id(step.dest), (step.dest, [], []))
            entry[1].append(step.rows)
            entry[2].append(step.cols)
    return dataclasses.replace(
        section, stages=tuple(stages),
        results=tuple(Buffer(t, Role.GROUP, _union(r), _union(c))
                      for t, r, c in results.values()))


def matrix_enabled(context) -> bool:
    """Whether operator steps are tiled on the matrix engines:
    `Options.tensor_cores` where it is set, `MATRIX` where it is not."""
    asked = context.get_user_options().tensor_cores
    return MATRIX if asked is None else bool(asked)


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


def lower(section: GroupSection, context,
          matrix: bool = True) -> GroupSection:
    """The section with each step's lowering for the context's target --
    the FMA forms only, where `matrix` is off -- and the stages and result
    buffers those lowerings read and write."""
    from tensorforge.backend.instructions.compute.primitives import nvidia
    sm = nvidia.sm_of(context)
    steps = tuple(dataclasses.replace(
        s, lowering=_lowering(s, section.dtype, context, sm, matrix))
        for s in section.steps)
    return _staged(dataclasses.replace(section, steps=steps), sm)


def tiled(section: GroupSection) -> bool:
    """Whether a step of the section is lowered on the matrix engines."""
    return any(s.lowering in MATRIX_LOWERINGS for s in section.steps)


def shared_bytes(section: GroupSection, elements: int) -> int:
    """Shared memory a group of `elements` takes: the operators as staged,
    and each element's copy of every input, intermediate and result."""
    size = section.dtype.size()
    operators = sum(st.volume for st in section.stages)
    per_element = sum(b.per_element for b in section.inputs
                      + section.intermediates + section.results)
    return size * (operators + elements * per_element)


def table_elements(section: GroupSection) -> int:
    """The table's size for the section: the entry of the smallest basis
    that holds the section's, of its precision."""
    fitting = [b for b, t in ELEMENTS
               if t is section.dtype and b >= section.basis]
    if not fitting:
        return DEFAULT_ELEMENTS
    return ELEMENTS[(min(fitting), section.dtype)]


def fill(section: GroupSection,
         context) -> Tuple[Optional[ElementGroup], str]:
    """The fill rule: the group's size for `section` on the context's
    target, or None and why not.

    The size is the table's (`table_elements`) where the block holds it,
    halved while it does not -- in shared memory, in threads -- and
    `MIN_ELEMENTS` the last size tried.  Each step's lowering is the target's (`lower`), and
    where no group of the matrix tiling fits -- its padded stages and its
    outputs' result buffers take room the FMA forms do not -- the FMA forms'
    (o6 F64 on sm_120's 99 kB).  The group carries the section lowered."""
    target = context.target
    hw = target.hw
    if getattr(hw, 'vendor', None) != 'nvidia' or target.backend != 'cuda':
        return None, (f'the multi-element layout is written for CUDA on '
                      f'NVIDIA, not {target.backend} on {hw.vendor}')
    options = context.get_user_options()
    # The batch loop's `Trip.ELEMENT_GROUP` takes neither.
    if options.launch_control:
        return None, ('the launch-control work queue hands out elements, '
                      'not groups')
    if int(options.stack_elements) > 1:
        return None, (f'a trip of {options.stack_elements} stacked elements '
                      f'is no element group')
    budget = hw.max_local_mem_size_per_block - SHARED_RESERVE
    candidates = [lower(section, context)]
    if tiled(candidates[0]):
        candidates.append(lower(section, context, matrix=False))
    for lowered in candidates:
        elements = table_elements(lowered)
        while True:
            need = shared_bytes(lowered, elements)
            if need <= budget and 32 * elements <= hw.max_threads_per_block:
                return ElementGroup(elements, lowered, need), ''
            if elements <= MIN_ELEMENTS:
                break
            elements = max(MIN_ELEMENTS, elements // 2)
    return None, (f'the operators and a group of {MIN_ELEMENTS} take '
                  f'{shared_bytes(candidates[-1], MIN_ELEMENTS)} B of shared '
                  f'memory, more than the {budget} B a block has: the '
                  f'operator needs to be streamed')


def plan(descrs: Sequence, context) -> Optional[ElementGroup]:
    """The section's multi-element plan, or None: where
    `Options.multi_element` is `force`, every product classifies and the
    fill rule finds a group.  `auto` takes none yet."""
    if context.get_user_options().multi_element != 'force':
        return None
    section, _ = classify(descrs)
    if section is None:
        return None
    group, _ = fill(section, context)
    return group


__all__ = ['Buffer', 'DEFAULT_ELEMENTS', 'ELEMENTS', 'ElementGroup',
           'GroupSection', 'Kind', 'LOWERING', 'MATRIX', 'MATRIX_LOWERINGS',
           'MIN_ELEMENTS', 'Product', 'Role', 'SHARED_RESERVE',
           'SMALL_EXTENT', 'Stage', 'Step', 'classify', 'fill', 'lower',
           'matrix_enabled', 'matrix_shape', 'plan', 'shared_bytes',
           'table_elements', 'tiled']
