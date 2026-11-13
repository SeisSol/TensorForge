# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Pseudo-IR: register images as blocks.

A register image (`MemSpace.REGISTER`) is one variable to the compiler: an
array per lane under SPMD, and under the explicit vector one `simd` of the
whole tile, since one work-item there is the wave
(`EsimdEmitter._plan_register_buffers`).  What the body does with it is
parts: under the explicit vector a product writes a column at a time and the
next operation reads it back a column at a time.  As one variable it is live
as a whole -- VC keeps `simd<double, 576>` in the register file or spills and
fills it in pieces around every `mad`, and each column read is a copy out of
it.

Two passes make the columns the variables:

`unroll_register_loops` writes out the short loops an image is addressed
through -- a loader's, a storer's -- each trip with its induction a constant,
so that every access to the image has an offset that is known.

`split_register_images` replaces an image every access of which lands inside
one block -- the widest access, a column -- by a buffer per block it touches,
and each access by the same access to its block, at the offset inside it.  An
image a pass cannot see all of keeps its array: one that lies in an arena,
starts from anything but zeros, is named in raw text, or is touched by
anything but an unpredicated load or store of its own -- the restrictions
under which the explicit-vector emitter promotes an image to a `simd` at all.

What the emitter makes of a block is what it makes of any image: one `simd`
of a column under the explicit vector, and under SPMD, where an access is one
element of a lane's array, a one-element array.

Both came out of an emitter-level prototype (`EsimdEmitter`, the explicit
vector only), measured on PVC: `volume` spilled 60 to 95 % fewer scratch
messages, and `localFluxAll` got worse.  Which column to keep and which to
spill is a question of liveness, and liveness is a question about the body --
so the structure is here, where a following pass can ask it, rather than in
one emitter.
"""

from __future__ import annotations

import itertools
import math
import re
from dataclasses import replace
from typing import Callable, Dict, FrozenSet, List, Optional, Sequence, Tuple

from tensorforge.common.basic_types import Datatype

from .ahead import Clone
from .core import (Access, BufferType, MemSpace, Op, ScalarType, Stmt,
                   TokenType, Value, walk_stmts)
from .passes import substitute

#: The most trips written out around one access: the trips of the loops its
#: offset reads, multiplied, and of the loops written out around one another.
#: SeisSol's images at order 6 are addressed through loops of nine trips, a
#: column each, and at most 27 nested: three slots by nine columns under SPMD.
UNROLL_MAX = 64

#: The index arithmetic an offset is followed through.
_ARITH = frozenset({'add', 'sub', 'mul'})

_IDENTIFIER = re.compile(r'\b[A-Za-z_]\w*\b')

#: What raw text may say that would make the trips of a loop more than copies
#: of its body.
_JUMPS = re.compile(r'\b(?:break|continue|return|goto)\b')

#: vISA's software fence, with the mask `Options.esimd_unroll_fence` gives:
#: 128, bit 7, is `FENCE_SW` -- no instruction, a barrier to the scheduler
#: only, so that the loads of the next trip are not hoisted over this one's
#: uses.
_FENCE = 'tensorforge::intel_esimd::fence<{}>();'


def _comment(s: Stmt) -> bool:
    """A raw statement that is only a comment: the macro layer writes the
    instruction it lowers above it (`// r1 = +(r0 * s0) + None`), which names
    every buffer and touches none."""
    text = (s.text or '').strip()
    return s.op == Op.RAWSTMT and bool(text) and all(
        line.strip().startswith('//') for line in text.splitlines())


def _is_buffer(x) -> bool:
    """A buffer as an operand: the value an allocation or a binding defines,
    or the macro layer's `Symbol`."""
    return ((isinstance(x, Value) and isinstance(x.type, BufferType))
            or hasattr(x, 'stype'))


def _width(v, explicit_simd: bool) -> int:
    """Elements of an image an access to it spans: the value's slots, times
    its lanes under the explicit vector, where one work-item holds them all
    (`EsimdEmitter._width`).  Under SPMD the lane is in the image -- an array
    per lane -- and a value is its own slots."""
    if not isinstance(v, Value) or not isinstance(v.type, ScalarType):
        return 1
    n = v.type.length or 1
    if explicit_simd and v.distributed:
        n *= v.lane_span()
    return n


def _flat(base, nums: Sequence[int]) -> int:
    """The element an access at the indices `nums` reaches, as the emitter
    computes its address (`Emitter.address`): leading dimension first where
    there is an index per dimension, their sum where there is not."""
    if not nums:
        return 0
    if isinstance(base, Value) and isinstance(base.type, BufferType):
        shape = base.type.shape
    else:
        view = getattr(base, 'data_view', None)
        shape = tuple(view.shape) if view is not None else None
    if shape is None or len(shape) != len(nums):
        return sum(nums)
    flat = nums[-1]
    for k in reversed(range(len(nums) - 1)):
        flat = nums[k] + shape[k] * flat
    return flat


class _Image:
    """A register image taken apart: its allocation, its block, every access
    with the elements it starts at over the trips of the loops its offset
    reads, and those loops' inductions."""

    __slots__ = ('name', 'alloc', 'block', 'accesses', 'loops')

    def __init__(self, name: str, alloc: Stmt, block: int,
                 accesses: List[Tuple[Stmt, List[int]]],
                 loops: FrozenSet[int]):
        self.name = name
        self.alloc = alloc
        self.block = block
        self.accesses = accesses
        self.loops = loops

    @property
    def value(self) -> Value:
        return self.alloc.target[0]


class _Body:
    """What both passes ask of a body: what an index is computed from, which
    loop an induction counts and whether it can be written out, and the
    register images with everything that touches them."""

    def __init__(self, body: Tuple[Stmt, ...], explicit_simd: bool):
        self.explicit_simd = explicit_simd
        self.walked = walk_stmts(body)
        self.defs: Dict[int, Stmt] = {}
        self.consts: Dict[int, int] = {}
        #: The `for` an induction counts, by the induction's id.
        self.loops: Dict[int, Stmt] = {}
        #: The inductions of the loops around a loop, outermost first.
        self.outer: Dict[int, Tuple[int, ...]] = {}
        #: The statements that read a value, by its id.
        self.uses: Dict[int, List[Stmt]] = {}
        #: The name a buffer is spelled by, by the id of the value that is
        #: it.  Accesses reach an image as the value its allocation defines or
        #: as the macro layer's `Symbol`, and only the name says that the two
        #: are one buffer (`EsimdEmitter._buf_name`).
        self.names: Dict[int, str] = {}
        for s in self.walked:
            for t in s.target:
                self.defs[t.id] = s
            for v in s.operands():
                self.uses.setdefault(v.id, []).append(s)
            if s.op == Op.CONST and s.target:
                c = s.attr('value')
                if isinstance(c, int) and not isinstance(c, bool):
                    self.consts[s.target[0].id] = c
            elif s.op == Op.FOR and s.regions:
                self.loops[s.induction.id] = s
            if s.target and isinstance(s.target[0].type, BufferType):
                if s.op == Op.ALLOC:
                    self.names[s.target[0].id] = (s.attr('extern')
                                                  or str(s.target[0]))
                elif s.attr('extern'):
                    self.names[s.target[0].id] = s.attr('extern')
        self._nest(body, ())
        self._reads: Dict[int, Optional[FrozenSet[int]]] = {}
        self._trips: Dict[int, Tuple[Optional[range], str]] = {}

    def _nest(self, stmts: Sequence[Stmt], around: Tuple[int, ...]) -> None:
        for s in stmts:
            inner = around
            if s.op == Op.FOR and s.regions:
                self.outer[s.induction.id] = around
                inner = around + (s.induction.id,)
            for r in s.regions:
                self._nest(r.body, inner)

    def name_of(self, x) -> str:
        if isinstance(x, Value):
            return self.names.get(x.id, str(x))
        return getattr(x, 'name', str(x))

    # -- index arithmetic ---------------------------------------------------- #

    def reads(self, x) -> Optional[FrozenSet[int]]:
        """The inductions the index `x` is computed from, through additions,
        subtractions and multiplications of those and of constants; None where
        it reads anything else."""
        if isinstance(x, bool):
            return None
        if isinstance(x, int):
            return frozenset()
        if not isinstance(x, Value):
            return None
        if x.id in self._reads:
            return self._reads[x.id]
        if x.id in self.consts:
            got = frozenset()
        elif x.id in self.loops:
            got = frozenset((x.id,))
        else:
            d = self.defs.get(x.id)
            got = None
            if (d is not None and d.op in _ARITH and len(d.args) == 2
                    and d.predicate is None):
                a, b = self.reads(d.args[0]), self.reads(d.args[1])
                if a is not None and b is not None:
                    got = a | b
        self._reads[x.id] = got
        return got

    def value(self, x, trip: Dict[int, int]) -> int:
        """The index `x` on the trip that binds each induction it reads as
        `trip` says.  Only for an index `reads` knows."""
        if isinstance(x, int):
            return x
        if x.id in trip:
            return trip[x.id]
        if x.id in self.consts:
            return self.consts[x.id]
        d = self.defs[x.id]
        a, b = self.value(d.args[0], trip), self.value(d.args[1], trip)
        return a + b if d.op == 'add' else a - b if d.op == 'sub' else a * b

    # -- loops --------------------------------------------------------------- #

    def trips(self, induction: int) -> Tuple[Optional[range], str]:
        """The trips of the loop `induction` counts, where it can be written
        out; None and the reason where it cannot."""
        got = self._trips.get(induction)
        if got is None:
            got = self._trips[induction] = self._trips_of(self.loops[induction])
        return got

    def _trips_of(self, loop: Stmt) -> Tuple[Optional[range], str]:
        """A loop is written out as copies of its body, one a trip, with the
        induction a constant in each: that takes bounds that are constants,
        nothing carried but tokens -- whose last values the results become --
        and a body every copy of which stands on its own.

        Not a merged run (`rolled`), which is a loop because the generator
        decided the code written out would not fit (`merge_variants`).  Not
        where raw text names the induction, which no copy declares, or
        may leave a trip; not where a trip allocates memory other than
        registers, whose copies would be buffers the allocator places apart;
        not where a commit names copies by their ids; and not where anything
        a trip defines is read behind the loop -- a copy issued in the loop
        whose wait stands behind it (`passes._risen`).  A declaration with a
        name of its own gets one per copy (`Clone.stmts`), and the text of the
        copy is respelled with it; a `Symbol` is not text, and one spelling a
        name a trip declares would go on spelling the old one."""
        lo, hi, step = loop.loop_bounds
        if not step or not all(isinstance(x, int) and not isinstance(x, bool)
                               for x in (lo, hi, step)):
            return None, 'its bounds are not constants'
        if any(not isinstance(a.type, TokenType) for a in loop.iter_args):
            return None, 'it carries a value'
        if loop.attr('rolled'):
            return None, 'it is a merged run'
        region = loop.regions[0]
        induction = {str(loop.induction)}
        if loop.attr('extern'):
            induction.add(str(loop.attr('extern')))
        declared = {str(x.attr('extern')) for x in region.body
                    if x.attr('extern')}
        inner = walk_stmts(region.body)
        inside = {id(x) for x in inner}
        defined = {a.id for a in region.args}
        for x in inner:
            if x.op == Op.COMMIT_ASYNC:
                return None, 'it closes a group of copies'
            if (x.op == Op.ALLOC and x.target
                    and getattr(x.target[0].type, 'space',
                                None) is not MemSpace.REGISTER):
                return None, 'a trip allocates memory other than registers'
            text = ' '.join(t for t in (x.text, x.attr('decl'))
                            if isinstance(t, str))
            if text and not _comment(x):
                if induction & set(_IDENTIFIER.findall(text)):
                    return None, 'raw text names its induction'
                if _JUMPS.search(text):
                    return None, 'raw text may leave a trip'
            if declared and any(
                    _is_buffer(a) and not isinstance(a, Value)
                    and getattr(a, 'name', None) in declared for a in x.args):
                return None, 'a symbol names what a trip declares'
            defined.update(t.id for t in x.target)
            for r in x.regions:
                defined.update(a.id for a in r.args)
        for vid in defined:
            for user in self.uses.get(vid, ()):
                if id(user) not in inside:
                    return None, 'what a trip defines is read behind the loop'
        return range(lo, hi, step), ''

    # -- images -------------------------------------------------------------- #

    def images(self, written_out: bool = False,
               refused: Optional[Dict[str, str]] = None,
               predicated: bool = False,
               whole: bool = False) -> List[_Image]:
        """The register images that are blocks once the loops they are
        addressed through are written out -- or, `written_out`, as they
        stand.  `refused` collects why the others are not, by name: an image
        a loop that stands addresses says why the loop could not be written
        out, where it could not.

        `predicated` takes an access under a predicate as one, where what
        follows makes it a select (`values.block_values`); `whole` takes an
        image that is a block already: one a split made (`block_of`), whose
        block is all of it whatever its widest access, or one whose widest
        access spans it."""
        refused = {} if refused is None else refused

        def refuse(name: str, why: str) -> None:
            refused.setdefault(name, why)

        allocs: Dict[str, List[Stmt]] = {}
        for s in self.walked:
            if (s.op == Op.ALLOC and s.target
                    and isinstance(s.target[0].type, BufferType)
                    and s.target[0].type.space is MemSpace.REGISTER):
                allocs.setdefault(self.names[s.target[0].id], []).append(s)
        held: Dict[str, Stmt] = {}
        for name, found in allocs.items():
            a = found[0]
            if len(found) > 1:
                refuse(name, 'more than one allocation by its name')
            elif a.attr('arena') is not None:
                refuse(name, 'it is a window into an arena')
            elif a.attr('init') not in (None, '', '{}'):
                refuse(name, f'it starts from {a.attr("init")}')
            elif a.target[0].type.swizzle is not None:
                refuse(name, 'it is permuted')
            else:
                held[name] = a
        if not held:
            return []
        reached: Dict[str, List[Stmt]] = {name: [] for name in held}
        for s in self.walked:
            if s.text and s.op in Op.RAW and not _comment(s):
                for word in held.keys() & set(_IDENTIFIER.findall(s.text)):
                    refuse(word, 'raw text names it')
            mine = None
            for i, arg in enumerate(s.args):
                if not _is_buffer(arg):
                    continue
                name = self.name_of(arg)
                if name not in held:
                    continue
                if (i == 0 and s.op in (Op.LOAD, Op.STORE)
                        and len(s.accesses) == 1
                        and (s.op == Op.STORE
                             or isinstance(s.target[0].type, ScalarType))):
                    if s.predicate is not None and not predicated:
                        refuse(name, 'an access under a predicate')
                    else:
                        reached[name].append(s)
                        mine = name
                else:
                    refuse(name, f'a `{s.op}` takes it')
            for a in s.accesses:
                if _is_buffer(a.base):
                    name = self.name_of(a.base)
                    if name in held and name != mine:
                        refuse(name, f'a `{s.op}` touches it')
        out = []
        for name, alloc in held.items():
            if name in refused or not reached[name]:
                continue
            image = self._image(name, alloc, reached[name], whole)
            if isinstance(image, str):
                refuse(name, image)
            else:
                out.append(image)
        out = self._within_unroll(out, refused)
        if written_out:
            for im in out:
                if im.loops:
                    refuse(im.name, 'an offset through a loop that stands')
            out = [im for im in out if not im.loops]
        return out

    def _image(self, name: str, alloc: Stmt, reached: List[Stmt],
               whole: bool = False):
        """`name` as blocks, or why it cannot be."""
        volume = alloc.target[0].type.volume
        seen: List[Tuple[Stmt, List[int], int]] = []
        loops: set = set()
        for s in reached:
            index = s.args[1:] if s.op == Op.LOAD else s.args[2:]
            reads: FrozenSet[int] = frozenset()
            for i in index:
                r = self.reads(i)
                if r is None:
                    return 'an offset that is not known'
                reads |= r
            order = sorted(reads)
            trips = []
            for induction in order:
                t, why = self.trips(induction)
                if t is None:
                    return f'it is addressed through a loop: {why}'
                trips.append(t)
            count = math.prod(len(t) for t in trips)
            if count > UNROLL_MAX:
                return (f'{count} trips of the loops it is addressed '
                        f'through, more than {UNROLL_MAX}')
            flats = [_flat(s.args[0], [self.value(i, dict(zip(order, ks)))
                                       for i in index])
                     for ks in itertools.product(*trips)]
            width = _width(s.target[0] if s.op == Op.LOAD else s.args[1],
                           self.explicit_simd)
            seen.append((s, flats, width))
            loops |= reads
        block = max(w for _, _, w in seen)
        if whole and alloc.attr('block_of') is not None and block <= volume:
            block = volume
        if block > volume or (block == volume and not whole):
            return 'its widest access spans it whole'
        if volume % block:
            return f'blocks of {block} do not tile {volume} elements'
        for s, flats, w in seen:
            for f in flats:
                if f < 0 or f + w > volume:
                    return 'an access past its end'
                if f // block != (f + w - 1) // block:
                    return 'an access across two blocks'
                if not self.explicit_simd and block % w:
                    # A wide access is a reinterpret cast of the address,
                    # which has to stay aligned in the block as it was in
                    # the image (`Emitter.elem_access`).
                    return 'a wide access its blocks would misalign'
        return _Image(name, alloc, block,
                      [(s, flats) for s, flats, _ in seen], frozenset(loops))

    def _within_unroll(self, images: List[_Image],
                       refused: Dict[str, str]) -> List[_Image]:
        """`images` without those whose loops, written out, would multiply
        past `UNROLL_MAX` copies of a statement: an access counts the loops
        its own offset reads, and a statement in a loop written out for one
        image inside a loop written out for another is copied for both."""
        while True:
            plan = set().union(*(im.loops for im in images))
            over = set()
            for induction in plan:
                chain = [x for x in self.outer[induction] + (induction,)
                         if x in plan]
                copies = math.prod(len(self._trips[x][0]) for x in chain)
                if copies > UNROLL_MAX:
                    over.update(chain)
            if not over:
                return images
            for im in images:
                if im.loops & over:
                    refused.setdefault(
                        im.name, f'more than {UNROLL_MAX} trips of loops '
                                 f'written out around one another')
            images = [im for im in images if not im.loops & over]


# --------------------------------------------------------------------------- #
# Writing loops out
# --------------------------------------------------------------------------- #

def unroll_register_loops(body: Tuple[Stmt, ...], scratch, *,
                          explicit_simd: bool = False, fence: int = 0,
                          predicated: bool = False,
                          report: Optional[List[str]] = None
                          ) -> Tuple[Stmt, ...]:
    """`body` with every loop written out that an image is addressed through
    which is blocks once it is (`split_register_images`).

    A loop goes whole, as one copy of its body a trip -- inner loops first,
    so that a copy of the outer body is a copy of the written-out inner one
    -- each with a constant for the induction, and the tokens it carries
    handed from one copy to the next; what used its results uses the last
    copy's.  A loop written out for no image stays: a copy of it is code and
    nothing else.

    What a copy computes from its constant -- its offsets -- is a constant
    too, and is written as one (`_fold_offsets`).

    `scratch` makes the builder the copies' values are numbered by; `fence`,
    under the explicit vector (`explicit_simd`), is the mask of a fence
    between the copies (`Options.esimd_unroll_fence`); `predicated` is
    `split_register_images`'s; `report` collects the loops written out.
    """
    seen = _Body(body, explicit_simd)
    images = seen.images(predicated=predicated)
    plan = set().union(*(im.loops for im in images))
    if not plan:
        return body
    if report is not None:
        for induction in sorted(plan):
            loop = seen.loops[induction]
            names = sorted(im.name for im in images if induction in im.loops)
            trips = len(seen.trips(induction)[0])
            report.append(f'blocks: wrote out the {trips} trips of '
                          f'%{loop.induction} for {", ".join(names)}')
    b = scratch()

    def fresh(v: Value) -> Value:
        return b.value(v.type, hint=v.hint, uniform=v.uniformity,
                       layout=v.layout, quals=v.quals)

    taken = {str(s.attr('extern')) for s in seen.walked if s.attr('extern')}
    text = _FENCE.format(int(fence)) if explicit_simd and fence else None
    known = dict(seen.consts)

    def holds(s: Stmt) -> bool:
        return any(x.op == Op.FOR and x.regions and x.induction.id in plan
                   for x in walk_stmts((s,)))

    def rewrite(stmts: Tuple[Stmt, ...]) -> Tuple[Stmt, ...]:
        out: List[Stmt] = []
        results: Dict[int, Value] = {}
        changed = False
        for s in stmts:
            if results:
                t = substitute((s,), results)[0]
                changed = changed or t is not s
                s = t
            if not s.regions or not holds(s):
                out.append(s)
                continue
            changed = True
            s = replace(s, regions=tuple(replace(r, body=rewrite(r.body))
                                         for r in s.regions))
            if s.op == Op.FOR and s.induction.id in plan:
                copies, last = _write_out(s, fresh, taken, scratch, text,
                                          known)
                out.extend(copies)
                results.update(last)
            else:
                out.append(s)
        return tuple(out) if changed else stmts

    return rewrite(body)


def _write_out(loop: Stmt, fresh: Callable[[Value], Value], taken: set,
               scratch, fence: Optional[str], known: Dict[int, int]
               ) -> Tuple[List[Stmt], Dict[int, Value]]:
    """The copies `loop` is written out as, and its results by what the last
    copy yields -- what it starts from where it has no trip.  `known` is
    every integer constant of the body, by value id, which the copies'
    offsets are folded with."""
    lo, hi, step = loop.loop_bounds
    induction = loop.induction
    region = loop.regions[0]
    last = region.terminator
    trip_body = region.body[:-1] if last is not None else region.body
    carried = list(loop.loop_inits)
    out: List[Stmt] = []
    for n, k in enumerate(range(lo, hi, step)):
        if n and fence is not None:
            f = scratch()
            # Touches no memory a pass models: it orders nothing but the
            # scheduler, and is pinned where it stands.
            f(fence, accesses=())
            out.extend(f.finish())
        c = scratch()
        # A name of its own for each declaration of the trip that has one,
        # unique in the file: the induction is.
        clone = Clone(fresh, f'trip{induction.id}_{n}', taken)
        clone.given(induction, c.const(k, induction.type))
        for arg, value in zip(region.args[1:], carried):
            clone.given(arg, value)
        out.extend(_fold_offsets(c.finish() + tuple(
            clone.stmts(trip_body, named=True)), known))
        if last is not None:
            carried = [clone.mapping.get(y.id, y) if isinstance(y, Value) else y
                       for y in last.args]
    return out, {r.id: v for r, v in zip(loop.target, carried)
                 if isinstance(v, Value)}


def _integer(t) -> bool:
    return (isinstance(t, ScalarType) and not t.is_vector
            and (t.base.name.startswith(('I', 'U'))
                 or t.base is Datatype.SIZE))


def _fold_offsets(stmts: Tuple[Stmt, ...], known: Dict[int, int],
                  spelled: Optional[set] = None) -> Tuple[Stmt, ...]:
    """`stmts` with every addition, subtraction and multiplication of
    integers that are known -- constants, by value id in `known`, which this
    adds to -- written as the constant it comes to: the offsets of a
    written-out copy, computed from its trip.  Not a value raw text spells
    by its name, which a constant does not declare.

    Not `passes.fold`, for what else it knows: a read of a pre-zeroed image
    at an offset nothing wrote is the zero it holds.  So it is of a block,
    and the zero is a scalar where the explicit vector holds a column, which
    its emitter stores as `0.0f.copy_to(...)` -- not C++.  Such a read
    reaches the emitter without the blocks as well, wherever a pipeline
    leaves one outside a loop; written out, a loop that reads an image where
    nothing wrote it leaves one."""
    if spelled is None:
        spelled = {word for x in walk_stmts(stmts) if x.text
                   for word in _IDENTIFIER.findall(x.text)}
    out: List[Stmt] = []
    changed = False
    for s in stmts:
        if s.regions:
            regions = tuple(replace(r, body=_fold_offsets(r.body, known,
                                                          spelled))
                            for r in s.regions)
            if any(r.body is not q.body for r, q in zip(regions, s.regions)):
                s = replace(s, regions=regions)
                changed = True
        elif s.op == Op.CONST and s.target:
            c = s.attr('value')
            if isinstance(c, int) and not isinstance(c, bool):
                known[s.target[0].id] = c
        elif (s.op in _ARITH and len(s.args) == 2 and len(s.target) == 1
              and s.predicate is None and not s.attr('escapes')
              and _integer(s.target[0].type)
              and str(s.target[0]) not in spelled):
            a, b = (x if isinstance(x, int) and not isinstance(x, bool)
                    else known.get(x.id) if isinstance(x, Value) else None
                    for x in s.args)
            if a is not None and b is not None:
                c = (a + b if s.op == 'add' else a - b if s.op == 'sub'
                     else a * b)
                known[s.target[0].id] = c
                s = replace(s, op=Op.CONST, args=(), attrs=(('value', c),))
                changed = True
        out.append(s)
    return tuple(out) if changed else stmts


# --------------------------------------------------------------------------- #
# Splitting images
# --------------------------------------------------------------------------- #

def split_register_images(body: Tuple[Stmt, ...], scratch, *,
                          explicit_simd: bool = False,
                          predicated: bool = False,
                          report: Optional[List[str]] = None
                          ) -> Tuple[Stmt, ...]:
    """`body` with every register image whose accesses all land inside one
    block of it at offsets that are known replaced by its blocks.

    The block is the widest access -- a column under the explicit vector,
    whose accesses span their lanes (`explicit_simd`), an element under SPMD
    -- where the blocks tile the image.  The image's allocation becomes one
    per block an access touches, named after it -- `r1_b3` for block 3 of
    `r1` where it has a name of its own -- and starting from what it started
    from; a block nothing touches holds nothing anybody reads, and is not
    declared.  Each access becomes the same access to its block, at the
    offset inside it.

    An access under a predicate keeps an image whole, but where `predicated`
    says that the blocks become values next (`values.block_values`), which
    make the access a select: the block is the access's own then.

    `scratch` makes the builder the blocks are numbered by; `report`
    collects what was split and why the rest was not.
    """
    refused: Dict[str, str] = {}
    seen = _Body(body, explicit_simd)
    images = seen.images(written_out=True, refused=refused,
                         predicated=predicated)
    replaced: Dict[int, Tuple[Stmt, ...]] = {}
    b = scratch() if images else None
    for image in images:
        v = image.value
        blocks: Dict[int, Value] = {}
        for s, flats in image.accesses:
            k, offset = divmod(flats[0], image.block)
            if k not in blocks:
                blocks[k] = b.value(
                    BufferType(v.type.elem, (image.block,), MemSpace.REGISTER,
                               readonly=v.type.readonly),
                    hint=f'{v.hint}_b{k}', uniform=v.uniformity,
                    layout=v.layout, quals=v.quals)
            replaced[id(s)] = (_moved(s, blocks[k], offset),)
        replaced[id(image.alloc)] = tuple(
            _declared(image.alloc, blocks[k], k, image.name)
            for k in sorted(blocks))
        if report is not None:
            report.append(f'blocks: {image.name}: {v.type.volume} elements as '
                          f'{len(blocks)} of {v.type.volume // image.block} '
                          f'blocks of {image.block}')
    if report is not None:
        for name, why in refused.items():
            report.append(f'blocks: - {name}: {why}')
    return _rebuild(body, replaced) if replaced else body


def _moved(s: Stmt, block: Value, offset: int) -> Stmt:
    """The load or store `s` of an image, as the same access to `block`."""
    kept = s.args[1:2] if s.op == Op.STORE else ()
    return replace(s, args=(block,) + kept + (offset,),
                   accesses=(Access(s.accesses[0].kind, MemSpace.REGISTER,
                                    block),))


def _declared(alloc: Stmt, block: Value, k: int, image: str) -> Stmt:
    """The allocation of an image, as that of its block `k`, which says
    whose block it is (`block_of`): a block is all of itself, whatever its
    widest access (`_Body.images`)."""
    return replace(alloc, target=(block,),
                   attrs=tuple((key, f'{val}_b{k}' if key == 'extern' else val)
                               for key, val in alloc.attrs)
                   + (('block_of', image),))


def _rebuild(stmts: Tuple[Stmt, ...],
             replaced: Dict[int, Tuple[Stmt, ...]]) -> Tuple[Stmt, ...]:
    out: List[Stmt] = []
    changed = False
    for s in stmts:
        hit = replaced.get(id(s))
        if hit is not None:
            out.extend(hit)
            changed = True
            continue
        if s.regions:
            regions = tuple(replace(r, body=_rebuild(r.body, replaced))
                            if any(id(x) in replaced
                                   for x in walk_stmts(r.body)) else r
                            for r in s.regions)
            if any(r is not q for r, q in zip(regions, s.regions)):
                s = replace(s, regions=regions)
                changed = True
        out.append(s)
    return tuple(out) if changed else stmts


class RegisterBlocks:
    """The passes as the pipeline runs them (`pipeline.standard_pipeline`):
    with the builder of what they add, whether the body is under the explicit
    vector, the fence between written-out trips, and whether the blocks
    become values (`values.block_values`), which takes an access under a
    predicate into a block too."""

    def __init__(self, scratch, explicit_simd: bool = False, fence: int = 0,
                 values: bool = False,
                 report: Optional[List[str]] = None):
        self.scratch = scratch
        self.explicit_simd = explicit_simd
        self.fence = fence
        self.values = values
        self.report = report

    def unroll(self, body: Tuple[Stmt, ...]) -> Tuple[Stmt, ...]:
        return unroll_register_loops(body, self.scratch,
                                     explicit_simd=self.explicit_simd,
                                     fence=self.fence,
                                     predicated=self.values,
                                     report=self.report)

    def split(self, body: Tuple[Stmt, ...]) -> Tuple[Stmt, ...]:
        return split_register_images(body, self.scratch,
                                     explicit_simd=self.explicit_simd,
                                     predicated=self.values,
                                     report=self.report)

    def as_values(self, body: Tuple[Stmt, ...]) -> Tuple[Stmt, ...]:
        from .values import block_values
        return block_values(body, self.scratch,
                            explicit_simd=self.explicit_simd,
                            report=self.report)
