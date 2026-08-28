# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
# SPDX-FileContributor: David Schneller

"""Pseudo-IR: core data model.

This is the *micro*-IR living inside a single ``AbstractInstruction`` --- the
layer that today is plain string emission through ``backend.writer.Writer``.
It deliberately knows nothing about C++, CUDA, HIP or SYCL; rendering happens
in ``pir.emit``, construction in ``pir.build``, analysis in ``pir.passes``.

Design decisions (see the region-vs-CFG discussion):

  * Structured, region-based control flow (MLIR ``scf`` style), *not* a flat
    CFG with phi nodes.  Control flow is a ``Stmt`` that *owns* ``Region`` s.
  * SSA across region boundaries via explicit ``yield`` plus region arguments
    (induction variable + iter_args).  No phi nodes, no mutation.
  * ``Value`` has identity semantics (``eq=False``) but a *deterministic* hash
    derived from its integer id --- never hash by ``id()``.
  * Memory effects are *localized*: an :class:`Access` carries (kind, space,
    base).  Two accesses to distinct bases never conflict, which is what makes
    reordering possible at all in a codegen where every buffer is known
    statically.  Allocation (offsets, lifetime, reuse) stays one layer up in
    ``opt.mem_region_allocation`` / ``opt.shr_mem_analyzer``; this layer only
    ever names a buffer, never places it.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import IntEnum, IntFlag, auto
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

from tensorforge.common.basic_types import Datatype

# NOTE: `from tensorforge.common import Datatype` does not work --
# tensorforge/common/ has no __init__.py and re-exports nothing.


class IRError(Exception):
    """Raised by the verifier and by malformed IR construction."""


# --------------------------------------------------------------------------- #
# Types
# --------------------------------------------------------------------------- #

class MemSpace(IntEnum):
    """Address space of a buffer.  Mirrors ``backend.symbol.SymbolType``."""

    NONE = 0        # not a memory operand at all (pure SSA value)
    REGISTER = 1    # thread-private registers / local array
    SHARED = 2      # __shared__ / LDS / local_accessor
    GLOBAL = 3      # device global memory
    CONSTANT = 4    # __constant__ / read-only
    SCRATCH = 5     # driver-provided scratchpad
    UNKNOWN = 6     # opaque (raw text): conflicts with everything

    @classmethod
    def from_symbol_type(cls, stype: Any) -> 'MemSpace':
        """Map a ``SymbolType`` onto a space.

        Takes the enum by *name* on purpose, so that ``pir`` does not have to
        import ``backend.symbol`` (which imports the ``Writer`` that ``pir``
        is meant to replace).
        """
        name = getattr(stype, 'name', str(stype))
        return {
            'Batch': cls.GLOBAL,
            'Global': cls.GLOBAL,
            'Data': cls.GLOBAL,
            'SharedMem': cls.SHARED,
            'Register': cls.REGISTER,
            'WarpwideSource': cls.REGISTER,
            'WarpwideAccumulator': cls.REGISTER,
            'Scratch': cls.SCRATCH,
            'Scalar': cls.NONE,
        }.get(name, cls.UNKNOWN)


@dataclass(frozen=True)
class ScalarType:
    """Type of an SSA value: a scalar, or a fixed-length vector of scalars."""

    base: Datatype
    length: Optional[int] = None    # None: scalar.  n >= 1: vector of n.

    def __post_init__(self):
        if self.length is not None and self.length < 1:
            raise IRError(f'vector length must be >= 1, got {self.length}')

    @property
    def is_vector(self) -> bool:
        # `length is not None`, NOT `if self.length` -- length 0 is invalid,
        # but a hypothetical 1-vector must not silently degrade to a scalar.
        return self.length is not None

    def __repr__(self):
        return f'{self.base}' if self.length is None else f'{self.base}x{self.length}'


@dataclass(frozen=True)
class BufferType:
    """Type of a memory buffer (the result of an ``alloc``).

    Carries shape and space but deliberately *no* offset: the byte offset is
    handed out by the whole-kernel allocator one layer up.
    """

    elem: Datatype
    shape: Tuple[int, ...]
    space: MemSpace

    @property
    def volume(self) -> int:
        v = 1
        for s in self.shape:
            v *= s
        return v

    def __repr__(self):
        dims = 'x'.join(str(s) for s in self.shape)
        return f'buffer<{dims}x{self.elem}, {self.space.name.lower()}>'


@dataclass(frozen=True)
class TokenType:
    """Completion token of an asynchronous memory operation.

    A token is an ordinary SSA value, so "this wait belongs to that copy" is a
    def-use edge rather than a side channel.  Carrying a token through a
    ``for`` loop's iter_args is exactly how a double-buffered pipeline is
    expressed: the copy issued in iteration k is waited in iteration k+1.
    """

    def __repr__(self):
        return 'token'


TOKEN = TokenType()


IRType = Union[ScalarType, BufferType, TokenType]

BOOL = ScalarType(Datatype.BOOL)
INDEX = ScalarType(Datatype.I32)


# --------------------------------------------------------------------------- #
# Effects
# --------------------------------------------------------------------------- #

class Effect(IntFlag):
    NONE = 0
    READ = auto()       # reads memory
    WRITE = auto()      # writes memory
    ATOMIC = auto()     # read-modify-write
    BARRIER = auto()    # block-wide / wave-wide synchronisation
    ASYNC = auto()      # issued here, completes only at a matching `wait`
    UNKNOWN = auto()    # opaque raw text: assume it does everything

    @property
    def is_opaque(self) -> bool:
        return bool(self & Effect.UNKNOWN)


ANY_EFFECT = (Effect.READ | Effect.WRITE | Effect.ATOMIC | Effect.BARRIER |
              Effect.ASYNC | Effect.UNKNOWN)

# Raw int masks.  `Effect` is an `IntFlag`, and `a | b` / `a & b` on one of
# those goes through `enum.Flag.__or__` -> `_get_value` -> `Flag.__call__` ->
# `Flag.__new__`, which is roughly two orders of magnitude slower than an int
# operation.  The analysis passes evaluate these predicates millions of times
# per kernel, so the masks are folded once here and the tests are done on the
# underlying int.
_M_WRITES = int(Effect.WRITE | Effect.ATOMIC)
_M_SIDE = int(Effect.WRITE | Effect.ATOMIC | Effect.BARRIER | Effect.ASYNC |
              Effect.UNKNOWN)
_M_SYNC = int(Effect.BARRIER | Effect.ASYNC)
_M_CLOBBER = int(Effect.WRITE | Effect.ATOMIC | Effect.UNKNOWN)


@dataclass(frozen=True)
class Access:
    """A *localized* memory effect: what kind, in which space, on which base.

    ``base`` is the *identity* of a buffer (a ``Symbol``, an alloc ``Value``,
    or any hashable token).  ``None`` means "somewhere in this space, don't
    know where" and therefore conflicts with everything in that space.
    """

    kind: Effect                # READ | WRITE | ATOMIC (never BARRIER)
    space: MemSpace
    base: Optional[Any] = None

    @property
    def writes(self) -> bool:
        return int(self.kind) & _M_WRITES != 0

    def __repr__(self):
        k = '+'.join(f.name.lower() for f in Effect if f and (self.kind & f))
        b = 'ยง' if self.base is None else getattr(self.base, 'name', str(self.base))
        return f'{k} {self.space.name.lower()}:{b}'


def may_alias(a: Access, b: Access) -> bool:
    """Three cheap levels; level 1 already covers most of a static codegen."""
    if MemSpace.UNKNOWN in (a.space, b.space):
        return True
    if a.space != b.space:
        return False                    # different address spaces never alias
    if a.base is not None and b.base is not None and a.base is not b.base:
        return False                    # distinct named buffers never alias
    return True                         # same base (or unknown): assume alias


def accesses_conflict(a: Access, b: Access) -> bool:
    if not (a.writes or b.writes):
        return False                    # read-after-read is free
    return may_alias(a, b)


# --------------------------------------------------------------------------- #
# Values
# --------------------------------------------------------------------------- #

class Uniformity(IntEnum):
    """How wide a value is the same across.

    The flag used to be a bool: uniform, or "thread-dependent (derived from lane
    id)".  Two levels cannot express the batch id, which is

        batchId0 = threadIdx.y + blockDim.y * blockIdx.x

    -- the same for every thread working on one multiplication, and different
    between the multiplications packed into a block.  Calling that uniform lets
    a pass hoist something out of a per-multiplication scope where it is not
    invariant; calling it lane-dependent forbids every hoist that would be
    legal.  So it is a lattice, ordered by "same across more threads".

    Propagation takes the ``min``: a value is only as uniform as its least
    uniform operand.
    """

    LANE = 0     # differs per thread -- derived from the lane id
    MULT = 1     # same within one multiplication, differs between them
    BLOCK = 2    # same within a thread block
    GRID = 3     # same everywhere


@dataclass(frozen=True)
class LaneAxis:
    """One tensor dimension, spread over the lanes of a wave.

    The map is the one the index machinery generates, read backwards::

        idx = ((tid / stride) % block) + slot * block

    so element ``s`` lives in slot ``s // block``, held by every thread ``t``
    with ``(t // stride) % block == s % block`` --- that is, by a run of
    ``stride`` *consecutive* threads starting at ``(s % block) * stride``, and
    again every ``stride * block`` threads after that.

    So ``block`` is how many distinct elements the dimension is spread over
    before it wraps to the next slot, and ``stride`` is how many neighbouring
    threads hold a *copy* of the same element.  ``stride`` is replication, not
    packing: it does not mean a lane holds several elements.  A lane holding
    four consecutive elements is a vector *type* (``ScalarType(base, 4)``)
    over the slot dimension, which is a different thing and does not belong on
    this axis.

    ``block == 1`` is the degenerate case: not distributed, every lane holds
    the whole extent.
    """

    block: int
    stride: int = 1

    def __post_init__(self):
        if self.block < 1:
            raise IRError(f'lane block must be >= 1, got {self.block}')
        if self.stride < 1:
            raise IRError(f'lane stride must be >= 1, got {self.stride}')
        if self.block == 1:
            # Not distributed: every thread holds the whole extent, and no
            # stride changes that.  Normalised because equality is the one
            # thing this type is *for* --- two axes describing the same
            # distribution have to compare equal, or a pass refuses a merge
            # that was legitimate and a relayout search fails to find an
            # instruction that was already there.
            object.__setattr__(self, 'stride', 1)

    @property
    def is_distributed(self) -> bool:
        return self.block > 1

    def holders(self, element: int, threads: int) -> Tuple[int, ...]:
        """Which threads hold `element`.  The definition, executable.

        Exists so that the mapping can be *checked* against the index the
        generator emits rather than restated in prose next to it --- the two
        had already drifted once.
        """
        want = element % self.block
        return tuple(t for t in range(threads)
                     if (t // self.stride) % self.block == want)

    def slot(self, element: int) -> int:
        return element // self.block

    def __repr__(self):
        return (f'{self.block}' if self.stride == 1
                else f'{self.block}@{self.stride}')


@dataclass(frozen=True)
class RegisterLayout:
    """How a register-resident value is distributed over a wave.

    A tuple of :class:`LaneAxis`, one per tensor dimension, outermost first.
    This is *not* a layout algebra: it cannot be inverted, and composing two
    layouts concatenates their axes rather than solving for a shared lane map.
    That is on purpose.  For the operator shapes this generator targets the
    set of layouts in play is small and enumerable, and the property a pass
    actually needs is only ``==``: may these two register images be treated as
    the same distribution, or does moving between them take a shuffle?

    ``None`` -- the default on :class:`Value` -- means *not tracked*, not
    *scalar*.  Every conservative check therefore has to treat an untracked
    value as distinct from every tracked one.
    """

    axes: Tuple[LaneAxis, ...] = ()

    @property
    def rank(self) -> int:
        return len(self.axes)

    @property
    def is_distributed(self) -> bool:
        return any(a.is_distributed for a in self.axes)

    def holders(self, index: Tuple[int, ...], threads: int) -> Tuple[int, ...]:
        """Which threads hold the element at multi-index `index`.

        The intersection of the per-axis answers.  For a rank-2 layout of a
        fused operator -- four lanes per entry of dimension 0, dimension 1
        alongside, so lane ``l`` holds ``(l % 4, l // 4)`` -- this comes out as
        a single lane, which is what makes the layout a bijection rather than
        a replication.
        """
        if len(index) != self.rank:
            raise IRError(f'index of rank {len(index)} for a rank-{self.rank} '
                          f'layout')
        out = set(range(threads))
        for axis, i in zip(self.axes, index):
            out &= set(axis.holders(i, threads))
        return tuple(sorted(out))

    def tiles(self, threads: int) -> bool:
        """Do the axes partition the lanes exactly, one lane per multi-index?

        This is the question a single axis cannot answer, and the reason
        ``stride`` reads differently at rank 1 and rank 2.  ``LaneAxis(16, 4)``
        on its own puts one element in four neighbouring lanes: they hold a
        *copy*, and the value is replicated fourfold.  The same axis beside a
        ``LaneAxis(4, 1)`` puts one element of *dimension 1* in those four
        lanes, and they differ in dimension 0 -- no copy anywhere.

        Same field, same number, opposite meaning, decided by the rest of the
        layout.  Which is why this defers to :meth:`replication` rather than
        carrying a second rule of its own: two rules for one fact drift, and
        the structural one had already disagreed with the count.
        """
        return self.replication(threads) == 1

    def replication(self, threads: int) -> int:
        """How many lanes hold a copy of the same element.

        Counted, not derived.  A structural formula has to reason about which
        thread-id bits the axes leave uncovered, and gets `LaneAxis(16, 4)`
        alone wrong -- the four lanes below the stride are covered by no axis,
        so they are copies, and a formula keyed on the strides tiling reports
        "does not tile" instead of "replicates by four".  At wave sizes an
        enumeration is exact and costs nothing.

        Returns 0 if the replication is not uniform across lanes.
        """
        classes = {}
        for t in range(threads):
            key = tuple((t // a.stride) % a.block for a in self.axes)
            classes.setdefault(key, 0)
            classes[key] += 1
        sizes = set(classes.values())
        return sizes.pop() if len(sizes) == 1 else 0

    def compose(self, other: 'RegisterLayout') -> 'RegisterLayout':
        """Append `other`'s axes.  Used to build a multi-dimensional layout
        out of the per-dimension descriptions the index machinery produces."""
        return RegisterLayout(self.axes + other.axes)

    def axis(self, dim: int) -> LaneAxis:
        return self.axes[dim]

    def __repr__(self):
        return f'layout<{",".join(repr(a) for a in self.axes)}>'


SCALAR_LAYOUT = RegisterLayout()


def join_layout(operands) -> Optional[RegisterLayout]:
    """The layout an elementwise result inherits from its operands.

    A replicated operand does not veto.  ``alpha * A`` where ``alpha`` is a
    scalar broadcast to every lane and ``A`` is spread across them produces a
    value spread exactly like ``A``; the old rule saw two distinct layouts,
    called it a disagreement, and returned ``None``.  That is not
    conservative, it is a loss: ``None`` means *unknown*, so every consumer
    downstream of a single scaling had to fail closed, and a scaling is on
    almost every operator SeisSol generates.

    Genuine disagreement -- two *different distributions* -- still gives
    ``None``.  A vendor intrinsic may legitimately consume two of those, so it
    is not an error here, but nothing may be concluded from it either.
    """
    values = [x for x in operands if isinstance(x, Value)]
    if operands and not values:
        # Every operand is a literal.  A literal is the same in every lane by
        # definition, so anything computed from literals alone is too -- the
        # same argument that gives `const` its layout, one step further along.
        #
        # This is the biggest single hole it closes: `LeadIndex.build` emits
        # `mul(nonlead, block)` for the slot offset, and both operands are
        # plain integers, so the result was *unknown* rather than *replicated*
        # -- and every address derived from it inherited the unknown.
        #
        # Zero operands is deliberately not covered.  A `rawexpr` with no
        # arguments is text the IR cannot read, and `threadIdx.x` is exactly
        # such a text; calling it replicated would be a guess, and the wrong
        # one.
        return SCALAR_LAYOUT
    seen = {x.layout for x in values if x.layout is not None}
    if not seen:
        return None
    spread = {lay for lay in seen if lay.is_distributed}
    if len(spread) == 1:
        return next(iter(spread))
    if not spread and len(seen) == 1:
        # every operand replicated, and they agree on the rank
        return next(iter(seen))
    return None


@dataclass(frozen=True, eq=False)
class Value:
    """An SSA value.

    ``eq=False`` -> identity comparison.  ``__hash__`` is overridden to hash
    the *id* rather than the object address, so that any accidental ``set``
    iteration stays reproducible across runs.
    """

    id: int
    type: IRType
    # NOTE: `uniform` below is a *derived* property, kept so that every existing
    # check kept its exact meaning when the lattice was introduced.  A pass that
    # wants the extra precision reads `uniformity` instead.
    uniformity: Uniformity = Uniformity.GRID
    hint: str = ''          # debug-only name fragment, e.g. 'acc' or 'data0'
    # How the value is spread over the lanes, when that is known.  `None` is
    # *untracked*, and untracked is not a layout: two untracked values say
    # nothing about each other, so every check that compares layouts has to
    # fail closed.  Nothing attaches one yet -- the field exists so that the
    # loaders, the vendor intrinsics and the passes can start agreeing on a
    # vocabulary one at a time instead of all at once.
    layout: Optional[RegisterLayout] = None

    def __post_init__(self):
        """One fact, one place: a distributed value is lane-varying.

        ``uniformity`` and ``layout.is_distributed`` are two statements about
        the same thing, and they disagreed on 71443 of the 93837 values in the
        corpus that carried a layout -- always in the unsafe direction, with
        ``uniformity`` reading ``GRID`` ("the same everywhere") for a value
        spread across the lanes.

        Not an oversight at those call sites.  ``op()`` and ``load()`` join the
        uniformity of their *operands*, and for a register-resident tile the
        operand is the slot index, which genuinely is the same in every lane --
        the lane is implicit in "each thread has its own array".  The address
        is uniform; the value it names is not.  Joining the operands answers
        the address's question and writes the answer on the value.

        Deriving it here rather than fixing six call sites is deliberate: a
        rule that lives on the type cannot drift from itself, and the next
        helper that forwards a layout gets it right without knowing this
        existed.  Only tightening is applied -- a caller that already said
        ``LANE`` is not overruled.
        """
        if (self.layout is not None and self.layout.is_distributed
                and self.uniformity > Uniformity.LANE):
            object.__setattr__(self, 'uniformity', Uniformity.LANE)

    @property
    def distributed(self) -> bool:
        """Does the wave hold a tensor here, rather than one value per lane?

        The question the ESIMD emitter asks to pick between ``T`` and
        ``simd<T, N>``.  ``False`` for an untracked value is *not* an answer --
        see :meth:`lane_span`, which refuses instead of guessing.
        """
        return self.layout is not None and self.layout.is_distributed

    def lane_span(self) -> int:
        """How many lanes this value is spread over.

        ``1`` means replicated: every lane holds the whole thing.  An
        untracked layout raises rather than returning ``1``, because those two
        are the same number and opposite facts -- and an emitter that cannot
        tell them apart writes a scalar where a vector belongs.
        """
        if self.layout is None:
            raise IRError(f'{self!r} has no tracked distribution; its lane '
                          f'span is unknown, not 1')
        span = 1
        for axis in self.layout.axes:
            span *= axis.block
        return span

    @property
    def uniform(self) -> bool:
        """Block-uniform, the level the original boolean meant.

        Anything narrower -- a lane index, or a per-multiplication value like
        the batch id -- reads False here, which is what every existing check
        expects.  New checks should compare ``uniformity`` against the level
        they actually need.
        """
        return self.uniformity >= Uniformity.BLOCK

    def __hash__(self):
        return hash(self.id)

    def __str__(self):
        # Kept identical in spirit to Writer.varalloc(): legacy f-strings that
        # interpolate a value still produce a valid C++ identifier.
        return f'v{self.id}_{self.hint}' if self.hint else f'v{self.id}'

    def __repr__(self):
        marks = {Uniformity.LANE: '~', Uniformity.MULT: '^',
                 Uniformity.BLOCK: '', Uniformity.GRID: ''}
        lay = '' if self.layout is None else f'/{self.layout!r}'
        return f'%{marks[self.uniformity]}{self}:{self.type!r}{lay}'


# An operand is either an SSA value or an inline literal.  Literals are allowed
# because in a codegen almost all loop bounds are compile-time constants and
# forcing them through `const` ops buys nothing.
Operand = Union[Value, int, float, str]


def value_operands(xs) -> Tuple[Value, ...]:
    return tuple(x for x in xs if isinstance(x, Value))


# --------------------------------------------------------------------------- #
# Statements & regions
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Region:
    """A single-entry/single-exit block owned by a statement."""

    args: Tuple[Value, ...] = ()
    body: Tuple['Stmt', ...] = ()

    @property
    def terminator(self) -> Optional['Stmt']:
        return self.body[-1] if self.body and self.body[-1].op == Op.YIELD else None

    @property
    def yielded(self) -> Tuple[Operand, ...]:
        t = self.terminator
        return t.args if t is not None else ()

    def __repr__(self):
        return f'Region(args={len(self.args)}, body={len(self.body)})'


class Op:
    """Canonical op names.  Anything not listed here is treated as a generic
    pure op by the passes, which is what makes the set extensible."""

    CONST = 'const'
    YIELD = 'yield'
    IF = 'if'
    FOR = 'for'
    ALLOC = 'alloc'
    LOAD = 'load'
    STORE = 'store'
    COPY_ASYNC = 'copy.async'   # global -> shared, completes at its wait
    LOAD_ASYNC = 'load.async'   # global -> register, ditto
    WAIT = 'wait'
    BARRIER = 'barrier'
    CALL = 'call'
    DECLARE = 'declare'     # `Ty name{};` -- a definition with no initialiser
    ACCUM = 'accum'         # `target += value;` -- in-place, no result
    PACK = 'pack'           # `VecTy v{a, b};`  -- aggregate initialisation
    EXTRACT = 'extract'     # `v[i]`            -- element of a packed vector
    # legacy escape hatches
    RAWEXPR = 'rawexpr'     # exactly one target; `text` is an *expression*
    RAWSTMT = 'rawstmt'     # no target;          `text` is a *statement*
    RAWBLOCK = 'rawblock'   # one region;         `text` is the block *head*

    CONTROL = frozenset({IF, FOR, RAWBLOCK})
    RAW = frozenset({RAWEXPR, RAWSTMT, RAWBLOCK})
    ASYNC = frozenset({COPY_ASYNC, LOAD_ASYNC})
    # statements that lower to a C++ declaration and therefore handle a
    # predicate themselves (as a select) rather than through a guard block
    DECLARING = frozenset({RAWEXPR, LOAD, LOAD_ASYNC, CALL, PACK, EXTRACT})


@dataclass(frozen=True)
class Stmt:
    op: str
    target: Tuple[Value, ...] = ()
    args: Tuple[Operand, ...] = ()
    regions: Tuple[Region, ...] = ()

    # optional per-statement predicate (SASS-style predication / if-conversion)
    predicate: Optional[Value] = None

    pure: bool = True           # CSE-able: same operands => same result
    movable: bool = True        # may be reordered subject to its accesses
    effect: Effect = Effect.NONE
    accesses: Tuple[Access, ...] = ()

    text: Optional[str] = None                      # raw ops only
    attrs: Tuple[Tuple[str, Any], ...] = ()         # small, hashable side data

    # -- convenience ------------------------------------------------------- #

    @property
    def is_block(self) -> bool:
        return len(self.regions) > 0

    @property
    def is_terminator(self) -> bool:
        return self.op == Op.YIELD

    def attr(self, key: str, default=None):
        for k, v in self.attrs:
            if k == key:
                return v
        return default

    def with_attr(self, key: str, value) -> 'Stmt':
        rest = tuple((k, v) for k, v in self.attrs if k != key)
        return replace(self, attrs=rest + ((key, value),))

    def operands(self) -> Tuple[Value, ...]:
        vs = list(value_operands(self.args))
        if self.predicate is not None:
            vs.append(self.predicate)
        return tuple(vs)

    @property
    def has_side_effects(self) -> bool:
        return int(self.effect) & _M_SIDE != 0

    def writes(self) -> Tuple[Access, ...]:
        return tuple(a for a in self.accesses if a.writes)

    # -- structured accessors (thin, so the positional convention stays local) #

    @property
    def cond(self) -> Operand:
        assert self.op == Op.IF
        return self.args[0]

    @property
    def loop_bounds(self) -> Tuple[Operand, Operand, Operand]:
        assert self.op == Op.FOR
        return self.args[0], self.args[1], self.args[2]

    @property
    def loop_inits(self) -> Tuple[Operand, ...]:
        assert self.op == Op.FOR
        return self.args[3:]

    @property
    def induction(self) -> Value:
        assert self.op == Op.FOR
        return self.regions[0].args[0]

    @property
    def iter_args(self) -> Tuple[Value, ...]:
        assert self.op == Op.FOR
        return self.regions[0].args[1:]

    # copy.async: args = (dst, src, *dst_index, *src_index); the split is in
    # the `ndst` attribute so that the positional convention stays local.

    @property
    def copy_dst(self) -> Operand:
        assert self.op == Op.COPY_ASYNC
        return self.args[0]

    @property
    def copy_src(self) -> Operand:
        assert self.op == Op.COPY_ASYNC
        return self.args[1]

    @property
    def copy_dst_index(self) -> Tuple[Operand, ...]:
        assert self.op == Op.COPY_ASYNC
        return self.args[2:2 + self.attr('ndst', 0)]

    @property
    def copy_src_index(self) -> Tuple[Operand, ...]:
        assert self.op == Op.COPY_ASYNC
        return self.args[2 + self.attr('ndst', 0):]

    @property
    def load_base(self) -> Operand:
        assert self.op == Op.LOAD_ASYNC
        return self.args[0]

    @property
    def load_index(self) -> Tuple[Operand, ...]:
        assert self.op == Op.LOAD_ASYNC
        return self.args[1:]

    @property
    def counter(self) -> str:
        """Which completion counter this async statement belongs to.

        AMD tracks global->LDS and global->VGPR in the same `vmcnt`; NVIDIA
        has a group counter for `cp.async` and hardware scoreboarding for
        register loads.  Keeping the classes apart lets the emitter decide.
        """
        return self.attr('counter', 'copy')

    @property
    def waited(self) -> Optional[Value]:
        """The token this `wait` consumes; None means "drain everything"."""
        assert self.op == Op.WAIT
        return self.args[0] if self.args else None


# --------------------------------------------------------------------------- #
# Traversal & def-use
# --------------------------------------------------------------------------- #

def walk(body: Tuple[Stmt, ...],
         parents: Tuple[Stmt, ...] = ()) -> Iterator[Tuple[Stmt, Tuple[Stmt, ...]]]:
    """Pre-order traversal.  Yields ``(stmt, enclosing statements)``."""
    for s in body:
        yield s, parents
        for r in s.regions:
            yield from walk(r.body, parents + (s,))


def defined_here(body: Tuple[Stmt, ...]) -> Dict[int, Value]:
    """Values defined *directly* in this body (not inside nested regions)."""
    out: Dict[int, Value] = {}
    for s in body:
        for t in s.target:
            out[t.id] = t
    return out


def defined_within(body: Tuple[Stmt, ...]) -> Dict[int, Value]:
    """Values defined anywhere in this body, including nested regions."""
    out: Dict[int, Value] = {}
    for s, _ in walk(body):
        for r in s.regions:
            for a in r.args:
                out[a.id] = a
        for t in s.target:
            out[t.id] = t
    return out


def def_use(body: Tuple[Stmt, ...]):
    """Return ``(defs, uses)``.

    ``defs``: value id -> defining statement (region args map to their owner).
    ``uses``: value id -> list of using statements, in traversal order.

    Both are plain dicts keyed by ``int``, so iteration order is deterministic
    and independent of object addresses.
    """
    defs: Dict[int, Stmt] = {}
    uses: Dict[int, List[Stmt]] = {}

    for s, _ in walk(body):
        for r in s.regions:
            for a in r.args:
                if a.id in defs:
                    raise IRError(f'{a!r} bound more than once')
                defs[a.id] = s
                uses.setdefault(a.id, [])
        for t in s.target:
            if t.id in defs:
                raise IRError(f'{t!r} defined more than once (not SSA)')
            defs[t.id] = s
            uses.setdefault(t.id, [])
        for v in s.operands():
            uses.setdefault(v.id, []).append(s)

    return defs, uses


def free_values(body: Tuple[Stmt, ...]) -> Dict[int, Value]:
    """Values used inside ``body`` but defined outside of it."""
    inside = defined_within(body)
    out: Dict[int, Value] = {}
    for s, _ in walk(body):
        for v in s.operands():
            if v.id not in inside:
                out[v.id] = v
    return out


def collect_effect(body: Tuple[Stmt, ...]) -> Effect:
    acc = 0
    for s, _ in walk(body):
        acc |= int(s.effect)
    return Effect(acc)


def collect_accesses(body: Tuple[Stmt, ...]) -> Tuple[Access, ...]:
    out: List[Access] = []
    seen = set()
    for s, _ in walk(body):
        for a in s.accesses:
            if a not in seen:
                seen.add(a)
                out.append(a)
    return tuple(out)


# --------------------------------------------------------------------------- #
# Textual form (dump between passes: TF_DUMP_IR=after-licm ...)
# --------------------------------------------------------------------------- #

def _fmt_operand(x: Operand) -> str:
    if isinstance(x, Value):
        return f'%{x}'
    if isinstance(x, float):
        return repr(x)
    return str(x)


def dump(body: Tuple[Stmt, ...], indent: int = 0) -> str:
    lines: List[str] = []
    pad = '  ' * indent
    for s in body:
        head = ''
        if s.target:
            head = ', '.join(f'%{t}:{t.type!r}' for t in s.target) + ' = '
        pred = f'@%{s.predicate} ' if s.predicate is not None else ''
        args = ' '.join(_fmt_operand(a) for a in s.args)

        notes = []
        for k, v in s.attrs:
            notes.append(f'{k}={v}')
        if not s.pure:
            notes.append('impure')
        if not s.movable:
            notes.append('pinned')
        for a in s.accesses:
            notes.append(repr(a))
        if s.effect & Effect.BARRIER:
            notes.append('barrier')
        if s.effect & Effect.UNKNOWN:
            notes.append('opaque')
        note = ('   { ' + ', '.join(notes) + ' }') if notes else ''

        body_txt = ''
        if s.text is not None:
            body_txt = f' "{s.text}"'

        if s.op == Op.FOR:
            lo, hi, st = s.loop_bounds
            it = ''
            if s.iter_args:
                it = ' iter(' + ', '.join(
                    f'%{a} = {_fmt_operand(i)}'
                    for a, i in zip(s.iter_args, s.loop_inits)) + ')'
            lines.append(f'{pad}{head}for %{s.induction} = {_fmt_operand(lo)} '
                         f'to {_fmt_operand(hi)} step {_fmt_operand(st)}{it} {{{note}')
            lines.append(dump(s.regions[0].body, indent + 1))
            lines.append(f'{pad}}}')
        elif s.op == Op.IF:
            lines.append(f'{pad}{head}if {_fmt_operand(s.cond)} {{{note}')
            lines.append(dump(s.regions[0].body, indent + 1))
            if len(s.regions) > 1:
                lines.append(f'{pad}}} else {{')
                lines.append(dump(s.regions[1].body, indent + 1))
            lines.append(f'{pad}}}')
        elif s.regions:
            lines.append(f'{pad}{head}{pred}{s.op}{body_txt} {{{note}')
            for r in s.regions:
                lines.append(dump(r.body, indent + 1))
            lines.append(f'{pad}}}')
        else:
            lines.append(f'{pad}{head}{pred}{s.op} {args}{body_txt}{note}'.rstrip())

    return '\n'.join(l for l in lines if l != '')
