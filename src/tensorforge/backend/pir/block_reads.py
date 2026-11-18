# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Pseudo-IR: reads served from one block message.

Two groupings of reads the explicitly vectorized lowering makes, written as
statements so that what they are is in the body rather than in a table of
the emitter's:

* a **window**: scalar reads of shared memory at constant addresses, with
  nothing between them that could change the buffer, read as one `slmLoad`
  into a vector whose elements they become.  A broadcast operand is read one
  element at a time, and where one work-item is the whole vector each of
  those reads is a message of its own -- 1836 SLM reads in `local_flux`,
  next to 727 global ones.
* a **run**: reads of one buffer at constant, consecutive offsets, each one
  element per lane, read as one block message where that is fewer messages
  and stays inside the buffer's own extent.  A lead dimension of 56 on 32
  lanes is two reads per column, one of 32 and a tail -- three messages
  where one of 64 would do.

Each becomes one `load` with `engine='block'` and the `shape` it reads --
`(W,)` for a window of W elements, `(w, n)` for a run of n reads of w lanes
-- defining a vector, in front of the first read it serves; and the reads
become pure extracts of that vector where they stood: a window's read the
`extract` of its element, a run's the `slice` of its w elements.

Run on the body the emitter writes, behind every pass: which reads go
together is decided by the final order, and a window held as one vector is
not what the analyses that ran before it measured -- the pressure and the
code they counted, and the decisions taken on those counts, stay what they
were.  The vectors it defines are numbered apart from every value (`_ids`),
so that nothing built after it is renumbered either.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

from tensorforge.common.basic_types import Datatype

from .core import (BufferType, MemSpace, Op, RegisterLayout, ScalarType, Stmt,
                   Uniformity, Value, walk_stmts)

#: The widest window, in elements: 64 registers of 64 bytes.
WINDOW_MAX = 1024

#: What one block message moves: LSC's transposed read takes at most 64
#: dwords (`d32x64t`) per address.
RUN_BYTES = 256

#: What a vector holds under the explicit vector.  `__float128` is not a
#: device type on pvc at all, and its literal needs a GNU suffix; its arrays
#: stay as they were.
VECTOR_ELEMS = (Datatype.F32, Datatype.F64)

#: The spaces a symbol's type names.
_SPACES = {'Global': MemSpace.GLOBAL, 'SharedMem': MemSpace.SHARED,
           'Register': MemSpace.REGISTER}

#: Where the vectors a window and a run define are numbered: below every
#: object (`IRBuilder.new_object`) and apart from each other, the first read
#: they serve taken as the offset -- negative, and drawn from no counter.
_WINDOW_IDS = -(1 << 40)
_RUN_IDS = -(1 << 41)


# --------------------------------------------------------------------------- #
# What a read is a read of
# --------------------------------------------------------------------------- #

def is_buffer(x: Any) -> bool:
    """A buffer is reached as the `Value` its `Op.ALLOC` defined or as the
    macro layer's `Symbol`."""
    return ((isinstance(x, Value) and isinstance(x.type, BufferType))
            or hasattr(x, 'stype'))


def space_of(x: Any) -> MemSpace:
    """Where a buffer lives, read off either form of it."""
    if isinstance(x, Value) and isinstance(x.type, BufferType):
        return x.type.space
    stype = getattr(x, 'stype', None)
    return _SPACES.get(getattr(stype, 'name', None), MemSpace.UNKNOWN)


def is_comment(x: Stmt) -> bool:
    """A raw statement that is only a comment: the macro layer writes the
    instruction it lowers above it (`// r1 = +(r0 * s0) + None`), which
    names every buffer and touches none."""
    text = (x.text or '').strip()
    return x.op == Op.RAWSTMT and bool(text) and all(
        line.strip().startswith('//') for line in text.splitlines())


def allocations(stmts) -> Tuple[Dict[str, Tuple[Stmt, Value]],
                                Dict[int, str]]:
    """`(allocs, names)`: name -> the `Op.ALLOC` and the `Value` it defines,
    and value id -> the name a buffer is spelled by.

    A pointer binding into global memory is a buffer the accesses name as
    well, by the name it binds -- not an allocation, so it is named and not
    offered as one."""
    allocs: Dict[str, Tuple[Stmt, Value]] = {}
    names: Dict[int, str] = {}
    for a in stmts:
        if not (a.target and isinstance(a.target[0].type, BufferType)):
            continue
        v = a.target[0]
        if a.op == Op.ALLOC:
            name = a.attr('extern') or str(v)
            names[v.id] = name
            allocs[name] = (a, v)
        elif a.attr('extern'):
            names[v.id] = a.attr('extern')
    return allocs, names


def buffer_name(x: Any, names: Dict[int, str]) -> str:
    """The name a buffer is spelled by.  Accesses reach one either as the
    `Value` its `Op.ALLOC` defined or as the macro layer's `Symbol`, and only
    the name says they are the same buffer."""
    if isinstance(x, Value):
        return names.get(x.id, str(x))
    return getattr(x, 'name', str(x))


def _shape(buf: Any) -> Optional[Tuple[int, ...]]:
    if isinstance(buf, Value) and isinstance(buf.type, BufferType):
        return buf.type.shape
    view = getattr(buf, 'data_view', None)
    return tuple(view.shape) if view is not None else None


def const_flat(buf: Any, indices, consts: Dict[int, Any]) -> Optional[int]:
    """The element an access of `buf` at `indices` reads, if every index is
    known: an integer, or a value `consts` gives one for."""
    nums = []
    for i in indices:
        n = consts.get(i.id) if isinstance(i, Value) else i
        if isinstance(n, bool) or not isinstance(n, int):
            return None
        nums.append(n)
    if not nums:
        return 0
    shape = _shape(buf)
    if shape is None or len(shape) != len(nums):
        return sum(nums)
    flat = nums[-1]
    for k in reversed(range(len(nums) - 1)):
        flat = nums[k] + shape[k] * flat
    return flat


def _indices_of(buf: Any, flat: int, rank: int) -> Tuple[int, ...]:
    """The indices of rank `rank` that `const_flat` reads as `flat`."""
    shape = _shape(buf)
    if rank == 0:
        return ()
    if shape is None or len(shape) != rank:
        return (flat,) + (0,) * (rank - 1)
    out = []
    for extent in shape[:-1]:
        out.append(flat % extent)
        flat //= extent
    return tuple(out) + (flat,)


def vector_width(v: Value) -> int:
    """Elements a distributed value spans: its lanes times its slots."""
    return v.lane_span() * (v.type.length or 1)


def messages(width: int, elem_bytes: int) -> int:
    """Block messages a `width`-element read is issued as: powers of two,
    at most `RUN_BYTES` each -- how `copy_from` and `slmLoad` split it."""
    cap = max(1, RUN_BYTES // elem_bytes)
    count, left, size = 0, width, cap
    while left:
        while size > left:
            size //= 2
        count += 1
        left -= size
    return count


def _linear(body) -> Tuple[Dict[int, Any], List[Tuple[Stmt, int]]]:
    """The constants of `body`, and its statements in order with the depth
    of the region each stands in."""
    consts: Dict[int, Any] = {}
    seq: List[Tuple[Stmt, int]] = []

    def lin(stmts, depth):
        for x in stmts:
            if x.op == Op.CONST and x.target:
                consts[x.target[0].id] = x.attr('value')
            seq.append((x, depth))
            for r in x.regions:
                lin(r.body, depth + 1)
    lin(body, 0)
    return consts, seq


def _single_read(x: Stmt) -> bool:
    """An unpredicated load of a buffer, defining its value."""
    return (x.op == Op.LOAD and x.predicate is None and bool(x.target)
            and bool(x.args) and is_buffer(x.args[0]))


# --------------------------------------------------------------------------- #
# Windows
# --------------------------------------------------------------------------- #

class Window(NamedTuple):
    """Scalar reads of one shared buffer, served by one read of `width`
    elements from `lo`: `reads` are each read and the element it is."""

    name: str
    buf: Any
    lo: int
    width: int
    elem: Datatype
    reads: Tuple[Tuple[Stmt, int], ...]


def plan_windows(body, allocs, names) -> List[Window]:
    """Scalar reads of shared memory, served from one block read.

    A run of them with constant addresses and nothing in between that could
    change the buffer is one `slmLoad` into a vector, and the reads become
    its elements.

    What may sit between two reads of a window: anything that does not
    write shared memory or synchronize.  Any shared store ends every window,
    not just the stored buffer's -- the arena reuses offsets, and a store
    through another name can land on the same bytes.  A window reaches into
    a nested region only if nothing in that region ends it, so a loop cannot
    read a window that its own last iteration made stale.

    Named `{buffer}_w{k}`, numbered in the order the windows close.
    """
    shared = {n: v for n, (a, v) in allocs.items()
              if v.type.space == MemSpace.SHARED
              and v.type.elem in VECTOR_ELEMS
              and getattr(v.type, 'swizzle', None) is None}
    consts, seq = _linear(body)

    def ends(x) -> bool:
        if is_comment(x):
            return False
        if x.op in Op.RAW or x.op in (Op.BARRIER, Op.WAIT, Op.CALL,
                                      Op.COMMIT_ASYNC, Op.COPY_ASYNC,
                                      Op.ACCUM):
            return True
        if x.op == Op.STORE:
            b = x.args[0] if x.args else None
            return space_of(b) not in (MemSpace.REGISTER, MemSpace.GLOBAL)
        return (not x.pure and not x.regions and x.op not in (
            Op.LOAD, Op.LOAD_ASYNC, Op.PREFETCH, Op.ALLOC, Op.DECLARE,
            Op.YIELD, Op.EXIT, Op.CONST))

    memo: Dict[int, bool] = {}

    def ends_inside(x) -> bool:
        if id(x) not in memo:
            memo[id(x)] = any(ends(y) for y in walk_stmts(
                tuple(z for r in x.regions for z in r.body)))
        return memo[id(x)]

    open_: Dict[str, dict] = {}
    out: List[Window] = []

    def close(name):
        st = open_.pop(name, None)
        if st is None or len(st['loads']) < 2:
            return
        vol = shared[name].type.volume
        flats = [f for _, f in st['loads']]
        lo, hi = min(flats), max(flats)
        width = hi - lo + 1
        if width > WINDOW_MAX or width > 8 * len(flats):
            return
        width = min(-(-width // 16) * 16, vol - lo)
        first = st['loads'][0][0]
        out.append(Window(f'{name}_w{len(out)}', first.args[0], lo, width,
                          shared[name].type.elem,
                          tuple((x, f - lo) for x, f in st['loads'])))

    for x, depth in seq:
        for n in [n for n, st in open_.items() if depth < st['depth']]:
            close(n)
        if ends(x) or (x.regions and ends_inside(x)):
            for n in list(open_):
                close(n)
            continue
        if not _single_read(x):
            continue
        buf, v = x.args[0], x.target[0]
        name = buffer_name(buf, names)
        if (name not in shared or x.attr('nontemporal')
                or not isinstance(v.type, ScalarType) or v.type.length
                or v.layout is None or v.distributed):
            continue
        flat = const_flat(buf, x.args[1:], consts)
        if flat is None or not 0 <= flat < shared[name].type.volume:
            continue
        open_.setdefault(name, {'depth': depth, 'loads': []})['loads'].append(
            (x, flat))
    for n in list(open_):
        close(n)
    return out


# --------------------------------------------------------------------------- #
# Runs
# --------------------------------------------------------------------------- #

class Run(NamedTuple):
    """Reads of `w` lanes each at consecutive offsets of one buffer, served
    by one read of all of them: `reads` are each read and the element of
    the run it starts at -- under SPMD, the element of the buffer
    (`plan_spmd_runs`)."""

    name: str
    elem: Datatype
    w: int
    reads: Tuple[Tuple[Stmt, int], ...]

    @property
    def total(self) -> int:
        return self.w * len(self.reads)


def _extent(buf: Any, name: str, allocs) -> Optional[int]:
    """Elements `buf` holds, where that is known."""
    if name in allocs:
        return allocs[name][1].type.volume
    if isinstance(buf, Value) and isinstance(buf.type, BufferType):
        return buf.type.volume
    # A prepared operand is longer than its matrix: its order pads
    # (`Tensor.storage_order`), and the padding is stored -- which is what
    # lets a run of a slot-major operand read to the end of its last slot.
    obj = getattr(buf, 'obj', None)
    if obj is not None and getattr(obj, 'storage_order', None) is not None:
        return int(obj.storage_volume())
    view = getattr(buf, 'data_view', None)
    shape = getattr(view, 'shape', None) if view is not None else None
    if not shape:
        return None
    n = 1
    for d in shape:
        n *= int(d)
    return n


def _run_ends(x: Stmt, space: MemSpace) -> bool:
    """Whether `x` ends a run of reads of a buffer in `space`."""
    if is_comment(x):
        return False
    # A fence ends a run as the raw text it replaces did: a run read as one
    # message would carry the loads behind it across it.
    if x.op in Op.RAW or x.op in (Op.CALL, Op.COPY_ASYNC, Op.ACCUM,
                                  Op.FENCE, Op.WAIT_COUNTER,
                                  Op.SET_PRIORITY):
        return True
    if space is MemSpace.SHARED and x.op in (Op.BARRIER, Op.WAIT,
                                              Op.COMMIT_ASYNC, Op.SYNC_INIT,
                                              Op.ARRIVE, Op.WAIT_PHASE):
        return True
    return False


def plan_runs(body, allocs, names) -> List[Run]:
    """Adjacent vector reads of one buffer, as one block message.

    Every message costs about five instructions besides the read: a 64-bit
    add for the address, a copy of it into the payload register, two
    `mov null` IGC puts around a send on this part, and a scoreboard wait.
    Read as one, `local_flux` on pvc takes 6450 instructions instead of
    9386, with the tail at full width (`Options.full_lane_tails`).

    A run is reads of the same buffer at constant, consecutive offsets,
    each one element per lane, taken together only where that is fewer
    messages and stays inside the buffer's own extent -- so the elements
    past a tail's `valid` lanes are the buffer's next ones, in range, and
    land only in lanes whose results are never stored.  The later reads of
    a run have to be dominated by the first and nothing may write the
    buffer in between; for shared memory nothing may synchronize either.

    Named `{buffer}_run{k}`, numbered in the order the runs close.
    """
    consts, seq = _linear(body)
    ends = _run_ends
    chains: Dict[str, dict] = {}
    out: List[Run] = []

    def close(name):
        ch = chains.pop(name, None)
        if ch is None or len(ch['loads']) < 2:
            return
        w, eb = ch['width'], ch['elem_bytes']
        total = w * len(ch['loads'])
        before = len(ch['loads']) * messages(w, eb)
        if messages(total, eb) >= before:
            return
        vol = ch['extent']
        if vol is None or ch['flat'] + total > vol:
            return
        out.append(Run(f'{name}_run{len(out)}', ch['elem'], w,
                       tuple((x, k * w) for k, x in enumerate(ch['loads']))))

    for x, depth in seq:
        for n in [n for n, ch in chains.items() if depth < ch['depth']]:
            close(n)
        if x.op == Op.STORE and x.args and is_buffer(x.args[0]):
            close(buffer_name(x.args[0], names))
            if space_of(x.args[0]) is MemSpace.SHARED:
                for n in [n for n, ch in chains.items()
                          if ch['space'] is MemSpace.SHARED]:
                    close(n)
            continue
        if any(ends(x, ch['space']) for ch in chains.values()):
            for n in [n for n, ch in chains.items() if ends(x, ch['space'])]:
                close(n)
        if x.regions:
            continue
        if not _single_read(x):
            continue
        buf, v = x.args[0], x.target[0]
        space = space_of(buf)
        if space not in (MemSpace.GLOBAL, MemSpace.SHARED):
            continue
        if (not isinstance(v.type, ScalarType) or v.type.length
                or v.layout is None or not v.distributed
                or v.type.base not in VECTOR_ELEMS
                or x.attr('nontemporal') or x.attr('head')):
            # A head keeps its narrower read (`_declare_unpredicated`); a run
            # reads the whole vector.
            continue
        flat = const_flat(buf, x.args[1:], consts)
        if flat is None:
            continue
        name = buffer_name(buf, names)
        w = vector_width(v)
        ch = chains.get(name)
        eb = v.type.base.size()
        if (ch is not None and ch['width'] == w and ch['next'] == flat
                and (len(ch['loads']) + 1) * w * eb <= RUN_BYTES):
            ch['loads'].append(x)
            ch['next'] = flat + w
            continue
        close(name)
        chains[name] = {'loads': [x], 'width': w, 'flat': flat,
                        'next': flat + w, 'depth': depth, 'space': space,
                        'elem': v.type.base, 'elem_bytes': eb,
                        'extent': _extent(buf, name, allocs)}
    for n in list(chains):
        close(n)
    return out


# --------------------------------------------------------------------------- #
# Runs under SPMD
# --------------------------------------------------------------------------- #

#: The most reads a sub-group block read serves: `sub_group::load` takes
#: vectors of 1, 2, 4 and 8.
SPMD_RUN_MAX = 8

#: The calls that read the lane along `x` (`IRBuilder.thread_id`).
_LANE_CALLEES = ('thread_idx_x', 'raw_thread_idx_x')


def _lane_offset(index: Any, defs: Dict[int, Stmt], consts: Dict[int, Any],
                 lanes: int) -> Optional[int]:
    """`c` where `index` is the work-item's lane in a block of `lanes` plus
    the constant `c`; None for anything else.

    The lane is `threadIdx.x % lanes`, as `IRBuilder.lane_index` writes it --
    or `threadIdx.x` itself, which the caller only asks of a multiplication
    of `lanes` lanes, where it never reaches `lanes`."""

    def walk(v, depth=0) -> Optional[Tuple[int, int]]:
        if isinstance(v, bool):
            return None
        if isinstance(v, int):
            return 0, v
        if not isinstance(v, Value) or depth > 32:
            return None
        known = consts.get(v.id)
        if isinstance(known, int) and not isinstance(known, bool):
            return 0, known
        d = defs.get(v.id)
        if d is None or d.predicate is not None:
            return None
        if d.op == Op.CALL and d.attr('callee') in _LANE_CALLEES:
            return 1, 0
        if d.op == 'add' and len(d.args) == 2:
            a, b = walk(d.args[0], depth + 1), walk(d.args[1], depth + 1)
            if a is None or b is None:
                return None
            return a[0] + b[0], a[1] + b[1]
        if d.op == 'rem' and len(d.args) == 2:
            tid = defs.get(getattr(d.args[0], 'id', None))
            extent = walk(d.args[1], depth + 1)
            if (tid is not None and tid.op == Op.CALL
                    and tid.attr('callee') in _LANE_CALLEES
                    and extent == (0, lanes)):
                return 1, 0
        return None

    found = walk(index)
    return found[1] if found is not None and found[0] == 1 else None


def plan_spmd_runs(body, allocs, names, lanes: int) -> List[Run]:
    """Runs where the lane is in the address: reads of one global buffer at
    `lane + c`, `lane + c + lanes`, ... -- each the multiplication's lanes,
    one element a lane -- served by one sub-group block read, which hands
    work-item `l` the element `c + j lanes + l` of read `j`.

    Taken together where they stand in one list of statements, in whatever
    order: an image filled a row block at a time reads a column's blocks
    far apart, and the first of them in the list is where the block read
    goes, ahead of the rest.  The same end rules as under the explicit
    vector, at most `SPMD_RUN_MAX` reads and a power of two of them, and the
    same extent: the reads stay inside the buffer.

    Only for a multiplication of `lanes` lanes laid out against a sub-group
    of as many (the caller's to know), and only over global memory: a block
    read of the local memory asks more of the address than an element's
    alignment.  `reads` are each read and the element of the buffer it
    starts at, in the order of the list.
    """
    consts, seq = _linear(body)
    defs = {v.id: x for x, _ in seq for v in x.target}
    # buffer name -> (the list the reads stand in, flat -> read, in order)
    groups: Dict[str, Tuple[int, Dict[int, Stmt]]] = {}
    extents: Dict[str, Optional[int]] = {}
    elems: Dict[str, Datatype] = {}
    out: List[Run] = []

    def close(name):
        found = groups.pop(name, None)
        if found is None:
            return
        reads = found[1]
        order = {id(x): i for i, x in enumerate(reads.values())}
        taken: set = set()
        for flat in sorted(reads):
            if flat in taken:
                continue
            count = 0
            while (count < SPMD_RUN_MAX
                   and flat + count * lanes in reads
                   and flat + count * lanes not in taken):
                count += 1
            size = 1
            while size * 2 <= count:
                size *= 2
            extent = extents[name]
            if size < 2 or extent is None or flat + size * lanes > extent:
                continue
            members = [(reads[flat + k * lanes], flat + k * lanes)
                       for k in range(size)]
            taken.update(f for _, f in members)
            members.sort(key=lambda m: order[id(m[0])])
            out.append(Run(f'{name}_run{len(out)}', elems[name], lanes,
                           tuple(members)))

    # The list each statement stands in.  A statement with regions ends
    # every group, so a group's reads are a stretch of one list with nothing
    # between them that branches, and the first of them dominates the rest.
    parents: Dict[int, int] = {}

    def mark(stmts):
        for x in stmts:
            parents[id(x)] = id(stmts)
            for r in x.regions:
                mark(r.body)
    mark(body)

    for x, _ in seq:
        here = parents.get(id(x), 0)
        if x.op == Op.STORE and x.args and is_buffer(x.args[0]):
            close(buffer_name(x.args[0], names))
            continue
        if _run_ends(x, MemSpace.GLOBAL) or x.regions:
            for n in list(groups):
                close(n)
            continue
        if not _single_read(x) or len(x.args) != 2:
            continue
        buf, v = x.args[0], x.target[0]
        if space_of(buf) is not MemSpace.GLOBAL:
            continue
        # A cache hint is no reason to keep a read apart: the SPMD lowering
        # spells none for one (`Lexic.glb_load`).
        if (not isinstance(v.type, ScalarType) or v.type.length
                or v.layout is None or not v.distributed
                or v.lane_span() != lanes
                or v.type.base not in VECTOR_ELEMS or x.attr('head')):
            continue
        flat = _lane_offset(x.args[1], defs, consts, lanes)
        if flat is None:
            continue
        name = buffer_name(buf, names)
        group = groups.get(name)
        if group is not None and (group[0] != here or flat in group[1]):
            close(name)
            group = None
        if group is None:
            group = groups[name] = (here, {})
            extents[name] = _extent(buf, name, allocs)
            elems[name] = v.type.base
        group[1][flat] = x
    for n in list(groups):
        close(n)
    return out


# --------------------------------------------------------------------------- #
# The statements
# --------------------------------------------------------------------------- #

_REPLICATED = RegisterLayout(())


def _window_stmts(win: Window) -> Tuple[Stmt, Dict[int, Stmt]]:
    """The window's read and the extract of each element in place of the
    read it serves."""
    first = win.reads[0][0]
    readers = [x.target[0] for x, _ in win.reads]
    vec = Value(id=_WINDOW_IDS - first.target[0].id,
                type=ScalarType(win.elem, win.width),
                uniformity=min(v.uniformity for v in readers),
                hint=win.name, layout=_REPLICATED)
    rank = len(first.args) - 1
    head = replace(first, target=(vec,),
                   args=(win.buf,) + _indices_of(win.buf, win.lo, rank),
                   attrs=(('engine', 'block'), ('shape', (win.width,)),
                          ('extern', win.name)))
    members = {}
    for x, k in win.reads:
        members[id(x)] = Stmt(
            op=Op.EXTRACT, target=x.target, args=(vec,), pure=True,
            movable=True, attrs=(('lane', k), ('no_inline', True))
            + ((('extern', x.attr('extern')),) if x.attr('extern') else ()))
    return head, members


def _run_stmts(run: Run) -> Tuple[Stmt, Dict[int, Stmt]]:
    """The run's read and the slice of each read's lanes in place of it."""
    first = run.reads[0][0]
    vec = Value(id=_RUN_IDS - first.target[0].id,
                type=ScalarType(run.elem, run.total),
                uniformity=Uniformity.LANE, hint=run.name,
                layout=_REPLICATED)
    head = replace(first, target=(vec,),
                   attrs=(('engine', 'block'),
                          ('shape', (run.w, len(run.reads))),
                          ('extern', run.name)))
    members = {}
    for x, off in run.reads:
        members[id(x)] = Stmt(
            op=Op.SLICE, target=x.target, args=(vec, off), pure=True,
            movable=True, attrs=(('no_inline', True),)
            + ((('extern', x.attr('extern')),) if x.attr('extern') else ()))
    return head, members


def _spmd_run_stmts(run: Run, lanes: int) -> Tuple[Stmt, Dict[int, Stmt]]:
    """The sub-group block read of a run, in front of the first read it
    serves in the list, and the element of it each read becomes: the read
    of the `j`-th block of lanes is element `j` of every work-item's
    vector.  The block read is addressed at the run's first element, the
    lane taken out -- it puts the lane back."""
    first = run.reads[0][0]
    v0 = first.target[0]
    vec = Value(id=_RUN_IDS - v0.id,
                type=ScalarType(run.elem, len(run.reads)),
                uniformity=v0.uniformity, hint=run.name, layout=v0.layout)
    start = min(flat for _, flat in run.reads)
    head = replace(first, target=(vec,), args=(first.args[0], start),
                   attrs=(('engine', 'block'),
                          ('shape', (lanes, len(run.reads))),
                          ('extern', run.name)))
    members = {}
    for x, flat in run.reads:
        members[id(x)] = Stmt(
            op=Op.EXTRACT, target=x.target, args=(vec,), pure=True,
            movable=True,
            attrs=(('lane', (flat - start) // lanes), ('no_inline', True))
            + ((('extern', x.attr('extern')),) if x.attr('extern') else ()))
    return head, members


def block_reads(body: Tuple[Stmt, ...], *, windows: bool = True,
                runs: bool = True,
                spmd_lanes: Optional[int] = None) -> Tuple[Stmt, ...]:
    """`body` with its windows and runs written as block reads: under the
    explicit vector `windows` and `runs`, under SPMD the runs of a
    multiplication of `spmd_lanes` lanes, one sub-group's worth
    (`plan_spmd_runs`).

    The read of a window or a run goes in front of the first read it serves,
    which dominates the rest (every planner closes a grouping when its
    region is left), and each read it serves becomes the extract of what it
    read, where it stood -- so the text keeps every declaration in its
    place.
    """
    stmts = walk_stmts(body)
    allocs, names = allocations(stmts)
    heads: Dict[int, Stmt] = {}
    members: Dict[int, Stmt] = {}
    if windows:
        for win in plan_windows(body, allocs, names):
            head, parts = _window_stmts(win)
            heads[id(win.reads[0][0])] = head
            members.update(parts)
    if runs:
        for run in plan_runs(body, allocs, names):
            head, parts = _run_stmts(run)
            heads[id(run.reads[0][0])] = head
            members.update(parts)
    if spmd_lanes:
        for run in plan_spmd_runs(body, allocs, names, spmd_lanes):
            head, parts = _spmd_run_stmts(run, spmd_lanes)
            heads[id(run.reads[0][0])] = head
            members.update(parts)
    if not members:
        return body

    def visit(stmts):
        out, changed = [], False
        for s in stmts:
            head = heads.get(id(s))
            if head is not None:
                out.append(head)
            part = members.get(id(s))
            if part is not None:
                out.append(part)
                changed = True
                continue
            if s.regions:
                regions = []
                inner = False
                for r in s.regions:
                    body_, did = visit(r.body)
                    inner = inner or did
                    regions.append(replace(r, body=body_) if did else r)
                if inner:
                    s = replace(s, regions=tuple(regions))
                    changed = True
            out.append(s)
        return tuple(out), changed

    return visit(body)[0]


def is_block(s: Stmt) -> bool:
    """The read of a window or a run (`block_reads`)."""
    return s.op == Op.LOAD and s.attr('engine') == 'block'


def is_part(s: Stmt) -> bool:
    """A read a window or a run serves, as the extract of it it became:
    told by the vector it extracts from, whose number is one `block_reads`
    gives and no builder does."""
    return (s.op in (Op.EXTRACT, Op.SLICE) and bool(s.args)
            and isinstance(s.args[0], Value) and s.args[0].id <= _WINDOW_IDS)


__all__ = ['RUN_BYTES', 'Run', 'SPMD_RUN_MAX', 'VECTOR_ELEMS', 'WINDOW_MAX',
           'Window', 'plan_spmd_runs',
           'allocations', 'block_reads', 'buffer_name', 'const_flat',
           'is_block', 'is_buffer', 'is_comment', 'is_part', 'messages',
           'plan_runs', 'plan_windows', 'space_of', 'vector_width']
