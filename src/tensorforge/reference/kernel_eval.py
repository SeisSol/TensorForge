# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
# SPDX-FileContributor: David Schneller

"""A generated CUDA kernel, executed on the host.

An interpreter for the subset of CUDA the generator emits: declarations,
assignments, `if`/`else`, counted `for`, ternaries, array and pointer
indexing, short vectors (`Vec`), the `cp.async` transfer and the bulk copy,
atomic accumulation, the cross-lane reads and the fold over the lanes
(`tensorforge::reduction`), and the warp's matrix instruction, `mma.sync`,
with the TF32 split around it.  `evaluate_wave` runs the lanes of one
multiplication together over one memory, statement by statement
(`Lockstep`), which is what a cooperatively staged operand, an atomic and a
`readlane` need; `evaluate` runs one lane on its own.

A read of a slot nothing wrote yields a pseudo-random value derived from the
slot's identity (`Slot`), so two kernels, or a kernel and a NumPy evaluation
of its descriptors, see the same inputs without anyone supplying them.

Limits, stated rather than hidden: one block runs, for one batch element.  A
barrier is a no-op -- `evaluate_wave` advances its lanes together anyway, and
`evaluate` has no other lanes, so what they would have written to shared
memory reads as seed fill there.  A construct outside the subset raises
`Abort` rather than being guessed at.
"""

from __future__ import annotations

import ast
import functools
import hashlib
import re
import struct
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple

_DECL = re.compile(
    # `alignas(N)` is a declaration specifier, not a statement of its own: a
    # register staging array carries one so that a wide access may be cast
    # onto it, and the oracle has to read past it to find the array.
    r'^(?:alignas\s*\(\s*\d+\s*\)\s+)?'
    r'(?:const\s+)?(?:__restrict__\s+)?'
    # `uint32_t` is how a kernel built with `enable_wrap_loads` holds the flag
    # words it carries across the loop (`pir/wrap.py`), and without a match
    # the whole configuration would abort ahead of the loop -- the one path
    # that most needs an oracle would have none.
    r'(?:tensorforge::Vector(?:Relaxed)?T\s*<[^>]*>|float[234]|double[234]|'
    r'u?int(?:8|16|32|64)_t|float|double|int|unsigned|size_t|bool|auto|'
    # A TF32 half (`nvidia.TF32_HALF`): held here as the float it stands
    # for, which is what `mma.sync` multiplies (`Interp._mma`).
    r'__float128|char|tensorforge::tf32)'
    r'(?P<ptr>\s*\*(?:\s*const)?(?:\s*__restrict__)?)?\s+'
    r'(?P<name>\w+)\s*(?P<arr>\[\s*(?P<dim>\d+)\s*\])?\s*'
    r'(?:\{\s*\}|=\s*(?P<init>.+))?$')
_FOR = re.compile(r'^for\s*\((?:const\s+)?\w+\s+(?P<v>\w+)\s*=\s*(?P<a>.+?);'
                  r'\s*\w+\s*(?P<cmp><=?|>=?)\s*(?P<b>.+?);\s*(?P<step>.+?)\)$')
_IF = re.compile(r'^if\s*\((?P<c>.*)\)$')
_WHILE = re.compile(r'^while\s*\((?P<c>.*)\)$')
#: The launch-control queue and the per-thread cursor through it
#: (`ClusterLaunchQueue`, `ClusterLaunchCursor` in `cuda.h`).
_LAUNCH_QUEUE = re.compile(r'^__shared__\s+tensorforge::ClusterLaunchQueue\s*<[^>]*>'
                           r'\s*(?P<name>\w+)$')
_LAUNCH_CURSOR = re.compile(r'^tensorforge::ClusterLaunchCursor\s*<[^>]*>'
                            r'\s*(?P<name>\w+)$')
_LAUNCH_START = re.compile(r'^(?P<name>\w+)\.start\s*\((?P<queue>\w+)\)$')
#: A declaration whose type is a namespaced vector type.
_VEC_DECL = re.compile(r'^tensorforge::Vector(?:Relaxed)?T\s*<[^>]*>\s+\w+')

#: The spellings an atomic accumulation reaches memory through.  All of them
#: are `dest[i] += value` here; what differs between them is which
#: instruction the vendor's compiler picks, which is not a question the host
#: can answer or needs to.
_ATOMIC_ADD = re.compile(
    r'^(?:atomicAdd|atomicAdd_block|__builtin_amdgcn_global_atomic_fadd_f(?:32|64)'
    r'|__hip_atomic_fetch_add)\s*\(\s*&(?P<base>\w+)\s*\[(?P<idx>.*)\]\s*,'
    r'\s*(?P<val>.*?)\s*(?:,\s*__ATOMIC_\w+\s*,\s*__HIP_MEMORY_SCOPE_\w+\s*)?\)$')

#: And the fourth, which is not a call: CUDA has no intrinsic that names the
#: global space, so `cuda_lexic.atomic_store` restates it in PTX at the
#: access (`red.global.add`) to keep the compiler off the generic decision
#: tree.  Without this the oracle sees no write at all and reports an empty
#: result rather than a wrong one -- which is how it was found.
_ATOMIC_ADD_ASM = re.compile(
    r'^asm\s+volatile\s*\(\s*"red\.global\.add\.\w+\s*\[%0\],\s*%1;"\s*'
    r'::\s*"l"\s*\(\s*__cvta_generic_to_global\s*\(\s*&\s*(?P<base>\w+)\s*'
    r'\[(?P<idx>.*?)\]\s*\)\s*\)\s*,\s*"[df]"\s*\(\s*(?P<val>.*?)\s*\)'
    r'\s*:\s*"memory"\s*\)$')
#: `*(SomeVecType*)&name[expr] = value`, captured for the interpreter
_VEC_STORE = re.compile(
    r'^\*\s*\(([^()]*?)\s*\*\)\s*&\s*(\w+)\s*\[(.+?)\]\s*=\s*(.+?);?$')
#: `*(SomeVecType*)&name[expr] = value`
_VEC_ASSIGN = re.compile(r'^\*\s*\([^)]*\)\s*&\s*\w+\s*\[.*\]\s*=')

#: `__pipeline_memcpy_async(&dst[i], &src[j], N)`.  The transfer half of
#: CUDA's pipeline primitives; the commit and the wait are the other half and
#: move nothing.  With a fourth argument, `Z`, the zero-filling form: `N - Z`
#: bytes read, the last `Z` written as zeros.
_ASYNC_COPY = re.compile(
    r'^__pipeline_memcpy_async\s*\(\s*&\s*(?P<dst>\w+)\s*\[(?P<di>[^\]]*)\]\s*,'
    r'\s*&\s*(?P<src>\w+)\s*\[(?P<si>[^\]]*)\]\s*,\s*(?P<bytes>\d+)\s*'
    r'(?:,\s*(?P<zfill>.+?)\s*)?\)$')

#: `tensorforge::copyAsyncFill<N>(&dst[i], &src[j], S)` (`cuda.h`): a copy
#: that fills, `S` bytes read -- an expression, zero where its condition
#: fails -- and the rest of the `N` written as zeros.
_ASYNC_FILL = re.compile(
    r'^tensorforge::copyAsyncFill\s*<\s*(?P<bytes>\d+)\s*>\s*\(\s*&\s*'
    r'(?P<dst>\w+)\s*\[(?P<di>[^\]]*)\]\s*,\s*&\s*(?P<src>\w+)\s*'
    r'\[(?P<si>[^\]]*)\]\s*,\s*(?P<read>.+)\)$')

#: AMD's LDS-DMA as `HipLexic.copy_async` writes it, `global_load_lds` from
#: `&src[i]` in global memory to `&dst[j]` in the LDS, `N` bytes a lane.
_LDS_DMA = re.compile(
    r'^__builtin_amdgcn_global_load_lds\s*\(\s*'
    r'\(\s*__attribute__\s*\(\(\s*address_space\s*\(\s*1\s*\)\s*\)\)\s*\w+'
    r'\s*\*\s*\)\s*\(\s*&\s*(?P<src>\w+)\s*\[(?P<si>[^\]]*)\]\s*\)\s*,\s*'
    r'\(\s*__attribute__\s*\(\(\s*address_space\s*\(\s*3\s*\)\s*\)\)\s*\w+'
    r'\s*\*\s*\)\s*\(\s*&\s*(?P<dst>\w+)\s*\[(?P<di>[^\]]*)\]\s*\)\s*,\s*'
    r'(?P<bytes>\d+)\s*,\s*0\s*,\s*0\s*\)$')

#: gfx1250's copy into the LDS as `HipLexic.copy_async` writes it,
#: `global_load_async_to_lds_bN` from `&src[i]` in global memory to `&dst[j]`
#: in the LDS -- each lane to its own address -- `N` bits a lane.
_ASYNC_LDS = re.compile(
    r'^__builtin_amdgcn_global_load_async_to_lds_b(?P<bits>\d+)\s*\(\s*'
    r'\(\s*__attribute__\s*\(\(\s*address_space\s*\(\s*1\s*\)\s*\)\)'
    r'\s*[\w:<>, ]+?\s*\*\s*\)\s*\(\s*&\s*(?P<src>\w+)\s*\[(?P<si>[^\]]*)\]'
    r'\s*\)\s*,\s*'
    r'\(\s*__attribute__\s*\(\(\s*address_space\s*\(\s*3\s*\)\s*\)\)'
    r'\s*[\w:<>, ]+?\s*\*\s*\)\s*\(\s*&\s*(?P<dst>\w+)\s*\[(?P<di>[^\]]*)\]'
    r'\s*\)\s*,\s*0\s*,\s*0\s*\)$')

#: gfx1250's tensor copy as `HipLexic` writes it: the descriptor of a tile of
#: `rows` rows of `cols` elements of `T` from `&src[i]` in global memory, its
#: rows `pitch` apart, to `&dst[j]` in the LDS, row after row; and the copy
#: through one, which one wave issues.
_TENSOR_DESC = re.compile(
    r'^(?:const\s+)?tensorforge::TensorDescriptor\s+(?P<name>\w+)\s*=\s*'
    r'tensorforge::tensorDescriptor\s*<\s*(?P<type>[\w:]+)\s*>\s*\(\s*'
    r'&\s*(?P<dst>\w+)\s*\[(?P<di>[^\]]*)\]\s*,\s*'
    r'&\s*(?P<src>\w+)\s*\[(?P<si>[^\]]*)\]\s*,\s*'
    r'(?P<rows>[^,]+),\s*(?P<cols>[^,]+),\s*(?P<pitch>[^,()]+)\)$')
_TENSOR_LOAD = re.compile(
    r'^tensorforge::tensorLoad\s*\(\s*(?P<desc>\w+)\s*\)$')

#: Bytes of the elements a tensor descriptor names.
_TENSOR_ELEMENT = {'float': 4, 'double': 8, 'int': 4, 'unsigned': 4,
                   'int32_t': 4, 'uint32_t': 4, 'int64_t': 8, 'uint64_t': 8}


class _TensorTile(NamedTuple):
    """A tensor descriptor: the tile it moves, `rows` rows of `cols`
    elements of `width` bytes, from `src` with its rows `pitch` apart, to
    `dst` row after row."""
    dst: Any
    src: Any
    rows: int
    cols: int
    pitch: int
    width: int


#: The buffer engine's helpers (`hip.h`, `pir.buffers`): a resource over
#: `bytes` bytes from `base`; a write through one, `N` elements of `T` at an
#: element index with an optional guard last (a read is an expression,
#: `BUFLOAD`); and the LDS-DMA through one, its LDS form.
_BUFFER_RESOURCE = re.compile(
    r'^(?:const\s+)?tensorforge::BufferResource\s+(?P<name>\w+)\s*=\s*'
    r'tensorforge::bufferResource\s*\(\s*(?P<base>.+?)\s*,'
    r'\s*(?P<bytes>[^,()]+)\)$')
_BUFFER_STORE = re.compile(
    r'^tensorforge::bufferStore\s*<\s*(?P<type>[\w:]+)\s*'
    r'(?:,\s*(?P<n>\d+)\s*)?>\s*\((?P<args>.*)\)$')
_BUFFER_LDS = re.compile(
    r'^tensorforge::bufferLoadLds\s*<\s*(?P<type>[\w:]+)\s*>\s*\(\s*'
    r'(?P<rsrc>\w+)\s*,\s*&\s*(?P<dst>\w+)\s*\[(?P<di>[^\]]*)\]\s*,'
    r'\s*(?P<index>.+)\)$')

#: Bytes of the elements the buffer helpers name.
_BUFFER_ELEMENT = {'float': 4, 'double': 8, 'int': 4, 'unsigned': 4,
                   'int32_t': 4, 'uint32_t': 4, 'int64_t': 8, 'uint64_t': 8,
                   'size_t': 8, 'std::int32_t': 4, 'std::uint32_t': 4,
                   'std::int64_t': 8, 'std::uint64_t': 8, 'std::size_t': 8}


class _Resource(NamedTuple):
    """A buffer resource: `nbytes` bytes from `base`.  An access is inside
    it where all its bytes are; one past it reads zero and writes nothing --
    what a guard makes of an access, its offset moved 2^31 past it."""
    base: Any
    nbytes: int

    def inside(self, index: int, count: int, size: int, keep=True) -> bool:
        return bool(keep) and index >= 0 and (index + count) * size <= self.nbytes

#: A bulk copy's cursor, armed by the thread the condition names
#: (`tensorforge::bulk_arm`); the copy through it, which that thread issues for
#: the whole multiplication; and the wait on it, after which the bytes are
#: visible to the thread that waited.
_BULK_ARM = re.compile(
    r'^tensorforge::BulkCursor\s+(?P<name>\w+)\s*=\s*tensorforge::bulk_arm'
    r'\s*\(\s*(?P<bar>\w+)\s*,\s*(?P<leader>.+)\)$')
_BULK_COPY = re.compile(
    r'^tensorforge::bulk_copy\s*\(\s*(?P<cursor>\w+)\s*,'
    r'\s*&\s*(?P<dst>\w+)\s*\[(?P<di>[^\]]*)\]\s*,'
    r'\s*&\s*(?P<src>\w+)\s*\[(?P<si>[^\]]*)\]\s*,\s*(?P<bytes>\d+)\s*\)$')
_BULK_WAIT = re.compile(r'^tensorforge::bulk_wait\s*\(\s*(?P<cursor>\w+)\s*\)$')

#: `tensorforge::splitFloatTF32(upper, lower, value)` (`cuda.h`): the two
#: TF32 halves a 3xTF32 product multiplies, written through references.
_SPLIT_TF32 = re.compile(
    r'^tensorforge::splitFloatTF32\s*\(\s*(?P<upper>\w+)\s*,\s*(?P<lower>\w+)'
    r'\s*,\s*(?P<value>.+)\)$')

#: `tensorforge::ldmatrixX{n}(r0, ..., base, index)` (`cuda.h`): `n` (1, 2
#: or 4) 8x8 matrices of a warp's rows in shared memory, written through
#: references; `tensorforge::stmatrixX{n}(base, index, r0, ...)`, the
#: registers into the rows (`engines.matrix_load`, `matrix_store`).
_LDMATRIX = re.compile(
    r'^tensorforge::ldmatrixX(?P<count>[124])\s*\((?P<args>.+)\)$')
_STMATRIX = re.compile(
    r'^tensorforge::stmatrixX(?P<count>[124])\s*\((?P<args>.+)\)$')

#: The name of `mma.sync` as NVIDIA's path issues it (`MMAInstr`): the
#: shape, and the types of D, A, B and C.
_MMA_NAME = re.compile(
    r'^mma\.sync\.aligned\.m(?P<m>\d+)n(?P<n>\d+)k(?P<k>\d+)\.row\.col'
    r'\.(?P<d>\w+)\.(?P<a>\w+)\.(?P<b>\w+)\.(?P<c>\w+)\s')


class _Mma(NamedTuple):
    """One `mma.sync` statement: the shape, A's type, and the operand
    expressions of its four register groups, in the template's order."""
    m: int
    n: int
    k: int
    atype: str
    d: Tuple[str, ...]
    a: Tuple[str, ...]
    b: Tuple[str, ...]
    c: Tuple[str, ...]


def _top_level(text: str, sep: str) -> List[str]:
    """`text` split at every `sep` outside parentheses and string literals."""
    out, depth, quoted, start = [], 0, False, 0
    for i, ch in enumerate(text):
        if ch == '"':
            quoted = not quoted
        elif not quoted and ch == '(':
            depth += 1
        elif not quoted and ch == ')':
            depth -= 1
        elif not quoted and depth == 0 and ch == sep:
            out.append(text[start:i])
            start = i + 1
    out.append(text[start:])
    return out


def _mma_statement(stmt: str) -> Optional[_Mma]:
    """`stmt` as an `mma.sync`, or None where it is something else.

    The inline-asm form `MMAInstr.asmcall` writes: a template of adjacent
    string literals naming the instruction and four groups of `%i`
    placeholders -- D, A, B, C -- then the outputs and the inputs, each a
    constraint and an expression.  A read-write operand (`"+f"`) stands for
    D and C at once, which is how the accumulator is passed.
    """
    head = re.match(r'^asm\s*(?:volatile\s*)?\(', stmt)
    if head is None or 'mma.sync' not in stmt or not stmt.endswith(')'):
        return None
    parts = _top_level(stmt[head.end():-1], ':')
    if len(parts) != 3:
        raise Abort(f'mma.sync without outputs and inputs: {stmt!r}')
    template = ''.join(re.findall(r'"([^"]*)"', parts[0]))
    name = _MMA_NAME.match(template)
    groups = re.findall(r'\{([^}]*)\}', template)
    if name is None or len(groups) != 4:
        raise Abort(f'mma.sync of a form not modeled: {template!r}')
    operands = []
    for text in parts[1:]:
        for operand in _top_level(text, ','):
            if not operand.strip():
                continue
            om = re.match(r'^\s*"(?P<con>[^"]*)"\s*\((?P<expr>.*)\)\s*$',
                          operand, re.S)
            if om is None:
                raise Abort(f'mma.sync operand {operand!r}')
            operands.append(om.group('expr').strip())

    def group(text):
        return tuple(operands[int(p)] for p in
                     re.findall(r'%(\d+)', text))
    return _Mma(int(name.group('m')), int(name.group('n')),
                int(name.group('k')), name.group('a'),
                *(group(g) for g in groups))


def _to_float32(value: float) -> float:
    """`value` rounded to the nearest float."""
    try:
        return struct.unpack('<f', struct.pack('<f', value))[0]
    except OverflowError:
        return float('inf') if value > 0 else float('-inf')


def _tf32(value: float) -> float:
    """`value` as the TF32 operand of an `mma.sync` reads it: the upper 19
    bits of the float -- sign, exponent and ten mantissa bits.  An operand
    is a register, and the instruction ignores what the conversion that
    filled it did not clear; the lower half of a split is the raw remainder
    (`splitFloatTF32`), so it loses its last thirteen bits here, as it does
    on the device."""
    bits = struct.unpack('<I', struct.pack('<f', _to_float32(value)))[0]
    return struct.unpack('<f', struct.pack('<I', bits & 0xffffe000))[0]


def _round_tf32(value: float) -> float:
    """`cvt.rna.tf32.f32`: `value` rounded to ten mantissa bits, half away
    from zero.  The device rounds half to even from sm_90 on
    (`splitFloatTF32`); either is a decomposition, and what is checked here
    is that the lower half carries what the upper one dropped."""
    bits = struct.unpack('<I', struct.pack('<f', _to_float32(value)))[0]
    if (bits >> 23) & 0xff == 0xff:
        return value
    bits = (bits + 0x1000) & 0xffffe000
    return struct.unpack('<f', struct.pack('<I', bits & 0xffffffff))[0]


#: `uint64_t d = tensorforge::matrixDescriptor(&buf[i], 0x...ull);` (`cuda.h`):
#: a shared-memory matrix descriptor, the operand's address and the bits of
#: every other field (`engines.matrix_descriptor_bits`).
_MATRIX_DESC = re.compile(
    r'^(?:const\s+)?(?:std::)?uint64_t\s+(?P<name>\w+)\s*=\s*'
    r'tensorforge::matrixDescriptor\s*\(\s*&\s*(?P<buf>\w+)\s*\[(?P<idx>.*)\]'
    r'\s*,\s*(?P<layout>0x[0-9a-fA-F]+)ull\s*\)$')
_WGMMA_FENCE = re.compile(r'^tensorforge::wgmmaFence\s*\(\s*\)$')
_WGMMA_COMMIT = re.compile(r'^tensorforge::wgmmaCommit\s*\(\s*\)$')
_WGMMA_WAIT = re.compile(
    r'^tensorforge::wgmmaWait\s*<\s*(?P<pending>\d+)\s*>\s*\(\s*\)$')
_PROXY_FENCE = re.compile(r'^tensorforge::fenceProxyAsync\s*\(\s*\)$')
#: An operand fence (`engines.operand_fence`): an empty volatile asm that
#: only tells the compiler what it may not move, nothing to execute.
_OPERAND_FENCE = re.compile(r'^asm\s+volatile\s*\(\s*""\s*:')
#: The instruction of a `wgmma.mma_async`: the shape, the types of D, A, B.
_WGMMA_NAME = re.compile(
    r'wgmma\.mma_async\.sync\.aligned\.m(?P<m>\d+)n(?P<n>\d+)k(?P<k>\d+)'
    r'\.(?P<d>\w+)\.(?P<a>\w+)\.(?P<b>\w+)\s+(?P<ops>[^;]*);')


class _MatrixDesc(NamedTuple):
    """A shared-memory matrix descriptor as a lane holds it: where its
    operand starts, and the bits of its other fields."""
    start: 'Ptr'
    layout: int


class _Wgmma(NamedTuple):
    """One `wgmma.mma_async` statement: the shape, A's type, the operand
    expressions of D, of A (one descriptor, or registers), of B's
    descriptor, and `scale-d` -- the literal, or the expression of the
    register the predicate is set from."""
    m: int
    n: int
    k: int
    atype: str
    d: Tuple[str, ...]
    a: Tuple[str, ...]
    a_desc: bool
    b: str
    scale: str


class _InFlight:
    """What a register holds while a wgmma that writes it is in flight:
    nothing a thread may read.  The product's value lands in it at the wait
    that retires its group (`Interp._wgmma_wait`); before that the PTX ISA
    leaves an access undefined ("wgmma.mma_async": accessing the
    accumulator registers ... without first performing a wgmma.wait_group
    ... is undefined behavior), and every use of this raises."""

    __slots__ = ('name',)

    def __init__(self, name: str):
        self.name = name

    def _refuse(self, *args):
        raise Abort(f'{self.name} read while a wgmma has it in flight, ahead '
                    f'of the wait that retires it')

    __add__ = __radd__ = __sub__ = __rsub__ = __mul__ = __rmul__ = _refuse
    __truediv__ = __rtruediv__ = __neg__ = __pos__ = __abs__ = _refuse
    __float__ = __int__ = __index__ = __bool__ = _refuse
    __lt__ = __le__ = __gt__ = __ge__ = _refuse

    def __repr__(self):
        return f'<{self.name} in flight>'


def _wgmma_statement(stmt: str) -> Optional[_Wgmma]:
    """`stmt` as a `wgmma.mma_async`, or None where it is something else.

    The inline-asm form `engines._lower_wgmma` writes: a template naming
    the instruction with D's registers in braces, then A -- one placeholder
    for a descriptor, braces for registers --, B's descriptor, and
    `scale-d`, the literal or the predicate a `setp` ahead of it sets from
    an operand; then the outputs and the inputs, each a constraint and an
    expression.  The immediate scales of A and B are 1: TF32 is not negated."""
    head = re.match(r'^asm\s*(?:volatile\s*)?\(', stmt)
    if head is None or 'wgmma.mma_async' not in stmt or not stmt.endswith(')'):
        return None
    parts = _top_level(stmt[head.end():-1], ':')
    if len(parts) not in (3, 4):
        raise Abort(f'wgmma without outputs and inputs: {stmt!r}')
    template = ''.join(re.findall(r'"([^"]*)"', parts[0])).replace('\\n', '\n')
    name = _WGMMA_NAME.search(template)
    if name is None:
        raise Abort(f'wgmma of a form not modeled: {template!r}')
    operands = []
    for text in parts[1:3]:
        for operand in _top_level(text, ','):
            if not operand.strip():
                continue
            om = re.match(r'^\s*"(?P<con>[^"]*)"\s*\((?P<expr>.*)\)\s*$',
                          operand, re.S)
            if om is None:
                raise Abort(f'wgmma operand {operand!r}')
            operands.append(om.group('expr').strip())
    fields = [f.strip() for f in _top_level(
        re.sub(r'\{([^}]*)\}', lambda mo: '{' + mo.group(1).replace(',', ' ')
               + '}', name.group('ops')), ',')]
    if len(fields) != 6 or fields[4:] != ['1', '1']:
        raise Abort(f'wgmma operands not modeled: {name.group("ops")!r}')

    def group(text):
        return tuple(operands[int(p)] for p in re.findall(r'%(\d+)', text))
    scale = fields[3]
    if scale == 'p':
        setp = re.search(r'setp\.ne\.b32\s+p\s*,\s*%(\d+)\s*,\s*0\s*;',
                         template)
        if setp is None:
            raise Abort(f'wgmma predicate set by nothing modeled: {template!r}')
        scale = operands[int(setp.group(1))]
    elif scale not in ('0', '1'):
        raise Abort(f'wgmma scale-d {scale!r}')
    a_desc = not fields[1].startswith('{')
    a = group(fields[1])
    b = group(fields[2])
    if len(b) != 1:
        raise Abort(f'wgmma B of {len(b)} operands: {stmt!r}')
    return _Wgmma(int(name.group('m')), int(name.group('n')),
                  int(name.group('k')), name.group('a'), group(fields[0]),
                  a, a_desc, b[0], scale)


#: The swizzle a wgmma descriptor's two top bits state, in bytes (PTX ISA,
#: wgmma's "Matrix Descriptor Format": 0 none, 1 128-byte, 2 64-byte,
#: 3 32-byte).
_WGMMA_SWIZZLE = {0: 0, 1: 128, 2: 64, 3: 32}

#: The swizzle tcgen05's descriptor states in its three top bits (PTX ISA,
#: "tcgen05 Shared Memory Descriptor": 0 none, 2 128-byte, 4 64-byte, 6
#: 32-byte); 1, the 128-byte one with 32-byte atoms, is not modeled.
_TC_SWIZZLE = {0: 0, 2: 128, 4: 64, 6: 32}

#: Tensor memory (`MemSpace.TMEM`): `uint32_t t = tensorforge::tmemAlloc<C>
#: (&slot[0]);`, the free, the waits and the fences of the engine (`cuda.h`).
_TMEM_ALLOC = re.compile(
    r'^(?:const\s+)?uint32_t\s+(?P<name>\w+)\s*=\s*tensorforge::tmemAlloc\s*<'
    r'\s*(?P<cols>\d+)\s*>\s*\(\s*&\s*(?P<slot>\w+)\s*\[\s*0\s*\]\s*\)$')
_TMEM_DEALLOC = re.compile(
    r'^tensorforge::tmemDealloc\s*<\s*(?P<cols>\d+)\s*>\s*\((?P<tmem>.+)\)$')
_TC_QUIET = re.compile(r'^tensorforge::(?:tmemWaitLd|tmemWaitSt|'
                       r'tcgen05FenceBefore|tcgen05FenceAfter)\s*\(\s*\)$')
_TC_COMMIT = re.compile(
    r'^tensorforge::tcgen05Commit\s*\(\s*(?P<cursor>\w+)\s*,(?P<issue>.+)\)$')
#: The product and the loads and stores of tensor memory, as the engines'
#: lowerings write them inside their asm.
_TC_MMA = re.compile(
    r'@e\s+tcgen05\.mma\.cta_group::1\.kind::tf32\s+\[%(?P<d>\d+)\]\s*,\s*'
    r'(?P<a>\[%\d+\]|%\d+)\s*,\s*%(?P<b>\d+)\s*,\s*%(?P<i>\d+)\s*,\s*'
    r'(?P<en>[01p])\s*;')
_TMEM_LDST = re.compile(
    r'tcgen05\.(?P<op>ld|st)\.sync\.aligned\.32x32b\.x(?P<n>\d+)\.b32\s+'
    r'(?P<ops>[^;]*);')

#: Columns of a lane of tensor memory, which addresses it here as
#: `lane * TMEM_ROW + column`.
_TMEM_ROW = 512


def _asm_operands(stmt: str):
    """An inline asm statement's template, with its escapes as the
    assembler reads them, and the expressions of its outputs and of its
    inputs; None where `stmt` is no asm."""
    head = re.match(r'^asm\s*(?:volatile\s*)?\(', stmt)
    if head is None or not stmt.endswith(')'):
        return None
    parts = _top_level(stmt[head.end():-1], ':')
    if len(parts) not in (3, 4):
        raise Abort(f'asm without outputs and inputs: {stmt!r}')
    template = ''.join(re.findall(r'"([^"]*)"', parts[0])).replace('\\n', '\n')
    lists = []
    for text in parts[1:3]:
        exprs = []
        for operand in _top_level(text, ','):
            if not operand.strip():
                continue
            om = re.match(r'^\s*"(?P<con>[^"]*)"\s*\((?P<expr>.*)\)\s*$',
                          operand, re.S)
            if om is None:
                raise Abort(f'asm operand {operand!r}')
            exprs.append(om.group('expr').strip())
        lists.append(tuple(exprs))
    return template, lists[0], lists[1]


class _Cursor(NamedTuple):
    """A lane's `tensorforge::BulkCursor`: which one, whether this lane
    issues the copies through it, and the barrier they complete on.

    The barrier is memory as far as `Races` goes in one respect: arming it
    writes it, and every wait reads it -- so a thread that waits before a
    barrier made the leader's arming visible to it is a race, as it is on the
    hardware.  The phases a copy completes are synchronization, and touch no
    slot here."""
    name: str
    leader: bool
    barrier: Optional['Ptr'] = None


class TensorMapArg(NamedTuple):
    """A tensor map the kernel is handed (`engines.TensorMap`): what a copy
    of the tensor memory accelerator reads a box through, passed as a
    parameter by value (`evaluate_wave`'s `scalars`, under the map's name).

    The tensor is the global array `buffer` from element `offset` on, of
    elements of `size` bytes; `dims` its extents and `strides` the elements
    from one index of each dimension to the next, outermost first, the
    innermost's 1 included; `box` the extents of a box; `swizzle` the bytes
    of NVIDIA's swizzle the box lies in in shared memory, 0 for none; `nan`
    whether what a box reads past the tensor's ends is NaN rather than
    zero.  `of` builds one from the map's static half."""
    buffer: str
    dims: Tuple[int, ...]
    strides: Tuple[int, ...]
    box: Tuple[int, ...]
    swizzle: int = 0
    size: int = 4
    offset: int = 0
    nan: bool = False

    @classmethod
    def of(cls, tmap: Any, buffer: str, offset: int = 0,
           dims: Optional[Sequence[int]] = None,
           strides: Optional[Sequence[int]] = None) -> 'TensorMapArg':
        """The map `tmap` (`engines.TensorMap`) over `buffer`, the extents
        and strides the launch binds given where the map leaves them."""
        dims = tuple(tmap.dims if dims is None else dims)
        strides = tuple(tmap.element_strides() if strides is None
                        else strides)
        if any(n is None for n in dims + strides):
            raise ValueError(f'{tmap.name}: an extent or a stride the launch '
                             f'binds is not given')
        return cls(buffer, dims, strides, tuple(tmap.box), tmap.swizzle,
                   tmap.elem.size(), offset, tmap.fill == 'nan')


#: A box copied through a tensor map (`tensorforge::tmaLoad`, `cuda_tma.h`):
#: the cursor whose leader issues it, the destination's element, the map,
#: the bytes announced, the coordinates innermost first.  A cache policy
#: (`tmaLoad<L2Evict::...>`) tells L2 what to keep, and moves the same box.
_TMA_LOAD = re.compile(
    r'^tensorforge::tmaLoad\s*(?:<\s*[\w:]+\s*>\s*)?\(\s*(?P<cursor>\w+)\s*,'
    r'\s*&\s*(?P<dst>\w+)\s*\[(?P<di>[^\]]*)\]\s*,'
    r'\s*&\s*(?P<map>\w+)\s*,\s*(?P<bytes>\d+)\s*,(?P<coords>.+)\)$')


#: A store under an L2 policy (`tensorforge::stL2<P>(&x, v)`, `cuda.h`):
#: the element after the `&`, then the value.
_STORE_L2 = re.compile(
    r'^tensorforge::stL2\s*<\s*[\w:]+\s*>\s*\(\s*&(?P<rest>.*)\)$')


def _first_operand(text: str) -> Tuple[str, str]:
    """`text` split at its first comma outside brackets: a call's first
    operand and the rest."""
    depth = 0
    for at, ch in enumerate(text):
        if ch in '([{':
            depth += 1
        elif ch in ')]}':
            depth -= 1
        elif ch == ',' and depth == 0:
            return text[:at].strip(), text[at + 1:].strip()
    raise Abort(f'a store under a cache policy without its value: {text!r}')


#: A box of shared memory added into the tensor of a map
#: (`tensorforge::tmaReduceAdd`): the map, the box's first element, the
#: coordinates innermost first; and the issuing thread's wait on its bulk
#: groups (`tensorforge::bulkWaitRead`).
_TMA_REDUCE = re.compile(
    r'^tensorforge::tmaReduceAdd\s*\(\s*&\s*(?P<map>\w+)\s*,'
    r'\s*&\s*(?P<src>\w+)\s*\[(?P<si>[^\]]*)\]\s*,(?P<coords>.+)\)$')
_BULK_WAIT_READ = re.compile(
    r'^tensorforge::bulkWaitRead\s*<\s*(?P<pending>\d+)\s*>\s*\(\s*\)$')


#: Scalar type -> bytes, for turning a copy's byte count into element slots.
#: The interpreter's memory is untyped, so the width has to come from the
#: declaration of the pointer being written.
_SCALAR_BYTES = {
    'float': 4, 'double': 8, 'half': 2, 'bfloat16': 2, '__float128': 16,
    'int8_t': 1, 'uint8_t': 1, 'int16_t': 2, 'uint16_t': 2,
    'int32_t': 4, 'uint32_t': 4, 'int64_t': 8, 'uint64_t': 8,
    'int': 4, 'char': 1, 'bool': 1,
}


def _scalar_bytes(stmt: str) -> Optional[int]:
    """The element size of a pointer declaration, from its type."""
    for word in re.findall(r'\w+', stmt):
        if word in _SCALAR_BYTES:
            return _SCALAR_BYTES[word]
    return None

# How many slots `evaluate(preset=...)` fills per array.  Larger than any
# operand a test case declares; the interpreter reads only what the kernel
# addresses, so the surplus is inert.
_PRESET_SLOTS = 4096


def _fill(mem: 'Slot', preset: Dict) -> None:
    """Write `preset` into `mem`: a value fills every slot of its array, a
    sequence the first slots one by one -- a flag per element, say."""
    for name, value in preset.items():
        if isinstance(value, (list, tuple)):
            for idx, one in enumerate(value):
                mem.write(name, idx, one)
            continue
        # generously wide: the interpreter only ever reads slots the
        # kernel addresses, so over-filling costs nothing
        for idx in range(_PRESET_SLOTS):
            mem.write(name, idx, value)


class Slot:
    """Flat memory, addressed by ``(base, index)``."""

    def __init__(self, seed: int = 0):
        self.data: Dict[Tuple[str, int], float] = {}
        self.seed = seed
        #: Watches the accesses to shared memory, where one is asked for
        #: (`evaluate_wave`, `races`).
        self.tracker: Optional['Races'] = None

    def read(self, base: str, idx: int) -> float:
        key = (base, int(idx))
        if self.tracker is not None and base == Races.SHARED:
            self.tracker.read(key[1])
        if key not in self.data:
            h = hashlib.blake2b(f'{self.seed}:{base}:{idx}'.encode(),
                                digest_size=8).digest()
            self.data[key] = (struct.unpack('<Q', h)[0] % 20011) / 1009.0 - 9.9
        return self.data[key]

    def write(self, base: str, idx: int, value) -> None:
        if isinstance(value, _InFlight):
            # A store reads the register it stores (`_InFlight`).
            value._refuse()
        if self.tracker is not None and base == Races.SHARED:
            self.tracker.write(int(idx))
        self.data[(base, int(idx))] = value


class Race(NamedTuple):
    """Two accesses to one slot of shared memory with nothing ordering them,
    or a barrier that not everyone it waits for reaches.

    `kind` is what the later access does against the earlier one: `RAW`, a
    read of what another lane wrote; `WAR`, a write over what another lane
    read; `WAW`, two lanes writing one slot; `FLIGHT`, an access to a slot an
    asynchronous copy is still landing in, by any lane, the issuing one
    included -- the copy is only done at the wait that retires it.

    `ARRIVAL` and `COUNT` are a barrier's, and `slot` is -1 for both.
    `ARRIVAL`: `lane` is one the barrier waits for and that does not execute
    it there, `other` how many lanes do.  On the hardware that is a hang, or
    a rendezvous paired with the wrong one -- the lanes that skipped it meet
    the others at their next barrier, and from there on every barrier orders
    the wrong accesses.  `COUNT`: a counted barrier whose count, `other`, is
    not the threads of the warps its lanes are in, `lane` of them.
    """
    kind: str
    slot: int
    lane: int
    other: int
    statement: str

    def __str__(self) -> str:
        if self.kind == 'ARRIVAL':
            return (f'ARRIVAL: lane {self.lane} does not reach '
                    f'`{self.statement}`, which {self.other} lanes wait at')
        if self.kind == 'COUNT':
            return (f'COUNT: `{self.statement}` counts {self.other} threads, '
                    f'and the warps of its lanes hold {self.lane}')
        return (f'{self.kind} on shared slot {self.slot}: lane {self.lane} '
                f'against lane {self.other} at `{self.statement}`')


class Races:
    """Which accesses to shared memory a barrier separates, and which it
    does not.

    Barrier intervals: two accesses of one slot by two lanes, at least one of
    them a write, are ordered exactly when both lanes took part in a barrier
    between them.  The interpreter runs its lanes in lockstep, so a missing
    barrier never changes a value here -- the reader always finds the write
    done.  What this does is say where the hardware would not have.

    An asynchronous copy writes its destination when the wait that retires it
    runs, not when it is issued: `__pipeline_commit` closes the lane's open
    copies into a group, `__pipeline_wait_prior(n)` completes all but its `n`
    youngest.  Until then the slots are in flight, and touching them is a race
    for every lane.

    AMD's LDS-DMA (`global_load_lds`) is counted per wave instead, one copy a
    statement, in `vmcnt`: `__builtin_amdgcn_s_waitcnt` completes all but the
    wave's `n` youngest, and so does a `__syncthreads`, whose fence drains the
    counter -- the only barrier that does.

    Only the lanes the run drives are seen -- one multiplication, on one
    block -- so what is checked is the multiplication's own synchronization,
    which is all a barrier this generator places has to provide.

    A barrier orders the lanes it meets, and which those are is in its
    spelling (`reach`): `__syncwarp` meets the lanes of one warp its mask
    names, `barrier.sync id, count` the warps that arrive at barrier `id` --
    one multiplication's, the id being the multiplication's own -- and
    `__syncthreads` the block.  So two lanes of a multiplication wider than a
    warp are not ordered by a `__syncwarp`, however many of them execute it
    together here.  And every lane the barrier meets has to execute it: the
    lockstep run goes on without the ones a branch or a loop left behind,
    the hardware waits for them (`ARRIVAL`).
    """

    #: The base the shared arena is modeled under (`_base_env`).
    SHARED = 'shr'

    #: A barrier: everything the target spells as one for the lanes of a
    #: multiplication or the block -- the launch-control hand-off among
    #: them, whose `next` lets a response slot go behind a `__syncthreads`.
    _BARRIER = re.compile(
        r'^(?:__syncthreads|__syncwarp|__builtin_amdgcn_s_barrier)\s*\('
        r'|^asm\s+volatile\s*\(\s*"(?:barrier|bar)\.sync'
        r'|^tensorforge::(?:sync|join)NamedBarrier\b'
        r'|^cooperative_groups::this_grid\(\)\.sync\('
        r'|.*\blaunchCursor\d+\.next\s*\(')
    #: The first half of a rendezvous: an arrival at a named barrier that
    #: waits for nobody (`engines.ARRIVE`) -- `bar.arrive` on NVIDIA, the
    #: signal of a gfx1250 object.  What the arriving lane wrote ahead of it
    #: is what the lanes waiting at the same barrier read behind their wait.
    _ARRIVE = re.compile(
        r'^asm\s+volatile\s*\(\s*"(?:barrier|bar)\.arrive'
        r'|^tensorforge::arriveNamedBarrier\b')
    _COMMIT = re.compile(r'^__pipeline_commit\s*\(\s*\)$')
    _WAIT = re.compile(r'^__pipeline_wait_prior\s*\(\s*(\d+)\s*\)$')
    #: A wait on gfx9's counters, the immediate `s_waitcnt` takes.
    _WAITCNT = re.compile(r'^__builtin_amdgcn_s_waitcnt\s*\(\s*(\d+)\s*\)$')
    #: The barrier whose fence drains `vmcnt` on AMD: HIP's `__syncthreads`
    #: is a release fence of the work-group, a barrier and an acquire.
    _DRAINS = re.compile(r'^__syncthreads\s*\(')
    #: Lanes per wave on the devices that have the LDS-DMA (CDNA2/3).
    DMA_WAVE = 64
    #: A wait on gfx1250's counter of its copies into the LDS.  Not drained
    #: by a block barrier: `__syncthreads` waits for no `asynccnt`.
    _WAIT_ASYNC = re.compile(
        r'^__builtin_amdgcn_s_wait_asynccnt\s*\(\s*(\d+)\s*\)$')
    #: Lanes per wave on gfx1250 and gfx1251, whose copies into the LDS are
    #: counted a wave at a time as well.
    ASYNC_WAVE = 32
    #: A wait on gfx1250's counter of its tensor copies, a wave's; not
    #: drained by a block barrier either.
    _WAIT_TENSOR = re.compile(
        r'^__builtin_amdgcn_s_wait_tensorcnt\s*\(\s*(\d+)\s*\)$')
    #: Lanes per wave whose tensor copies one counter counts.
    TENSOR_WAVE = 32
    #: The two barriers narrower than the block: a warp's, with its mask,
    #: and a counted one, with its operands -- `%n` naming the n-th input.
    _WARP_BARRIER = re.compile(r'^__syncwarp\s*\((?P<mask>.*)\)$')
    _NAMED_BARRIER = re.compile(
        r'^asm\s+volatile\s*\(\s*"(?:barrier|bar)\.(?:sync|arrive)'
        r'(?:\.aligned)?\s+'
        r'(?P<ops>[^";]*);?"\s*(?::\s*:\s*(?P<inputs>.*?))?\s*'
        r'(?::\s*"memory"\s*)?\)$')
    #: A gfx1250 object of the multiplication's, the element of the
    #: kernel's array its row picks (`HipLexic.sync_mult`): the row is the
    #: barrier's id, and the count the setup gave it is the multiplication's.
    _OBJECT_BARRIER = re.compile(
        r'^tensorforge::(?:sync|arrive)NamedBarrier\s*\(\s*&\s*\w+\s*'
        r'\[(?P<row>.*)\]\s*\)$')
    _INPUT = re.compile(r'"\w+"\s*\(')
    #: Lanes per warp, which a warp barrier's mask counts in.
    WARP = 32

    @classmethod
    def _inputs(cls, text: str) -> List[str]:
        """The expressions of an asm statement's input operands, in order."""
        out = []
        for m in cls._INPUT.finditer(text or ''):
            depth, k = 1, m.end()
            while depth and k < len(text):
                depth += {'(': 1, ')': -1}.get(text[k], 0)
                k += 1
            out.append(text[m.end():k - 1].strip())
        return out

    def reach(self, stmt: str) -> Tuple:
        """Whom the barrier `stmt` meets: `('warp', mask)`, the lanes of the
        warp whose bits `mask` sets, all of them for no mask; `('named', id,
        count)`, the lanes whose barrier `id` is this one, `count` of them;
        or `('block',)`.  The operands are expressions, each lane's own."""
        m = self._WARP_BARRIER.match(stmt)
        if m:
            return ('warp', m.group('mask').strip() or None)
        m = self._OBJECT_BARRIER.match(stmt)
        if m:
            return ('named', m.group('row').strip(), None)
        m = self._NAMED_BARRIER.match(stmt)
        if m:
            inputs = self._inputs(m.group('inputs'))
            ops = [o.strip() for o in m.group('ops').split(',')]
            resolved = [inputs[int(o[1:])] if re.fullmatch(r'%\d+', o) else o
                        for o in ops]
            return ('named', resolved[0],
                    resolved[1] if len(resolved) > 1 else None)
        return ('block',)

    def arrival(self, lane: int, present: int) -> None:
        """`lane` is waited for at the barrier being executed and does not
        execute it; `present` lanes do."""
        self._report('ARRIVAL', -1, present, lane=lane)

    def count(self, threads: int, counted: int) -> None:
        """The barrier being executed counts `counted` threads, where the
        warps of the lanes it meets hold `threads`."""
        self._report('COUNT', -1, counted, lane=threads)

    def __init__(self, limit: int = 64):
        self.time = 0
        self.lane = 0
        self.statement = ''
        self.limit = limit
        self.found: List[Race] = []
        self._seen = set()
        self._writes: Dict[int, Tuple[int, int]] = {}
        self._reads: Dict[int, Dict[int, int]] = {}
        #: (lane, other) -> the time of the last barrier both took part in
        self._met: Dict[Tuple[int, int], int] = {}
        self._flight: Dict[int, int] = {}
        self._open: Dict[int, set] = {}
        self._groups: Dict[int, List[set]] = {}
        #: Set while an asynchronous copy writes, so its slots go in flight.
        self.issuing = False
        #: The cursor a bulk copy writing right now goes through, if one is.
        self.bulk: Optional[str] = None
        #: slot -> (cursor, when) of the bulk copy in flight into it, and
        #: (cursor, lane) -> when that lane last waited on the cursor.
        self._bulk_flight: Dict[int, Tuple[str, int]] = {}
        self._landed: Dict[Tuple[str, int], int] = {}
        #: slot -> the lane whose reduction of a box still reads it, and
        #: lane -> the slots each of its bulk groups reads, oldest first
        #: (`reduce_issue`, `reduce_wait`).
        self._reduce_flight: Dict[int, int] = {}
        self._bulk_groups: Dict[int, List[set]] = {}
        #: named barrier -> lane -> when it arrived without waiting.
        self._arrived: Dict[int, Dict[int, int]] = {}
        #: Set while an LDS-DMA copy writes: the wave it counts in -- for
        #: gfx1250's copies into the LDS, `('async', wave)`, and for its
        #: tensor copies `('tensor', wave)`.
        self.dma: Optional[int] = None
        #: wave -> its LDS-DMA copies in flight, oldest first, each the time
        #: it was issued and the slots it writes; slot -> (wave, time, lane)
        #: of the copy landing in it.
        self._dma: Dict[int, List[Tuple[int, set]]] = {}
        self._dma_flight: Dict[int, Tuple[int, int, int]] = {}
        #: slot -> the wave whose wait landed the copy last written to it:
        #: what every lane of that wave reads behind the wait, which is the
        #: wave's and not a lane's.
        self._dma_wave: Dict[int, int] = {}

    def statement_kind(self, stmt: str) -> Optional[str]:
        """`'barrier'`, `'arrive'`, `'commit'`, `'wait'`, `'vmcnt'`,
        `'asynccnt'`, `'tensorcnt'` or None."""
        if self._WAIT_TENSOR.match(stmt):
            return 'tensorcnt'
        if self._ARRIVE.match(stmt):
            return 'arrive'
        if self._BARRIER.match(stmt):
            return 'barrier'
        if self._COMMIT.match(stmt):
            return 'commit'
        if self._WAIT.match(stmt):
            return 'wait'
        if self._WAITCNT.match(stmt):
            return 'vmcnt'
        if self._WAIT_ASYNC.match(stmt):
            return 'asynccnt'
        return None

    @classmethod
    def asynccnt_of(cls, stmt: str) -> int:
        """The count an `s_wait_asynccnt` waits down to."""
        return int(cls._WAIT_ASYNC.match(stmt).group(1))

    def asynccnt(self, wave: int, keep: int) -> int:
        """gfx1250's copies into the LDS of `wave` land but its `keep`
        youngest -- as `vmcnt` lands the LDS-DMA's, on a counter of their
        own (`_dma`, under `('async', wave)`)."""
        return self.vmcnt(('async', wave), keep)

    @classmethod
    def tensorcnt_of(cls, stmt: str) -> int:
        """The count an `s_wait_tensorcnt` waits down to."""
        return int(cls._WAIT_TENSOR.match(stmt).group(1))

    def tensorcnt(self, wave: int, keep: int) -> int:
        """gfx1250's tensor copies of `wave` land but its `keep` youngest,
        on their own counter (`_dma`, under `('tensor', wave)`)."""
        return self.vmcnt(('tensor', wave), keep)

    @classmethod
    def vmcnt_of(cls, stmt: str) -> int:
        """The `vmcnt` a gfx9 `s_waitcnt` immediate waits down to: its low
        four bits and bits 14 and 15 above them; 63 waits for nothing."""
        imm = int(cls._WAITCNT.match(stmt).group(1))
        return (imm & 0xF) | ((imm >> 14) & 0x3) << 4

    def vmcnt(self, wave: int, keep: int) -> int:
        """The wave's LDS-DMA copies land but its `keep` youngest: now, each
        as a write of the lane whose bytes it moved.  How many slots
        landed."""
        copies = self._dma.get(wave, [])
        cut = max(0, len(copies) - keep)
        done, self._dma[wave] = copies[:cut], copies[cut:]
        landed = 0
        for when, slots in done:
            for slot in slots:
                found = self._dma_flight.get(slot)
                if found is None or found[:2] != (wave, when):
                    continue
                del self._dma_flight[slot]
                self._writes[slot] = (found[2], self.time)
                self._dma_wave[slot] = wave
                self._reads.pop(slot, None)
                landed += 1
        return landed

    def _own_wave(self, slot: int) -> bool:
        """Whether the last write of `slot` is a copy the reading lane's own
        wave waited for: the wait is the wave's, and its lanes go on in
        lockstep behind it."""
        wave = self._dma_wave.get(slot)
        if isinstance(wave, tuple) and wave[0] == 'tensor':
            return wave == ('tensor', self.lane // self.TENSOR_WAVE)
        if isinstance(wave, tuple):
            return wave == ('async', self.lane // self.ASYNC_WAVE)
        return wave == self.lane // self.DMA_WAVE

    def drains(self, stmt: str) -> bool:
        """Whether the barrier `stmt` waits for every LDS-DMA copy of the
        waves that reach it (`_DRAINS`)."""
        return bool(self._DRAINS.match(stmt))

    def tick(self, stmt: str) -> None:
        self.time += 1
        self.statement = stmt

    def barrier(self, lanes: List[int],
                arrived: Optional[Dict[int, int]] = None) -> None:
        """`lanes` meet: each is ordered behind what every other did so far
        -- and behind what a lane that only arrived (`arrived`, lane -> when)
        did up to its arrival, which waits for none of them."""
        for a in lanes:
            for b in lanes:
                self._met[(a, b)] = self.time
            for b, when in (arrived or {}).items():
                self._met[(a, b)] = max(self._met.get((a, b), -1), when)

    def arrive(self, key: int, lane: int) -> None:
        """`lane` arrived at the named barrier `key` without waiting: the
        lanes that wait there next meet what it did up to now."""
        self._arrived.setdefault(key, {})[lane] = self.time

    def arrivals(self, key: int) -> Dict[int, int]:
        """The lanes that arrived at `key` since its last rendezvous, which
        that rendezvous completes."""
        return self._arrived.pop(key, {})

    def commit(self, lane: int) -> None:
        self._groups.setdefault(lane, []).append(self._open.pop(lane, set()))

    def wait(self, lane: int, stmt: str) -> None:
        keep = int(self._WAIT.match(stmt).group(1))
        groups = self._groups.get(lane, [])
        cut = max(0, len(groups) - keep)
        done, self._groups[lane] = groups[:cut], groups[cut:]
        for group in done:
            for slot in group:
                if self._flight.get(slot) == lane:
                    del self._flight[slot]
                    self._writes[slot] = (lane, self.time)
                    self._reads.pop(slot, None)

    def reduce_issue(self, lane: int, slots) -> None:
        """`lane` issued a reduction of the box in `slots`, closed into a
        bulk group of its own: the accelerator reads them until the lane
        waits for the group (`reduce_wait`), and a write before is a race,
        whatever barrier stands between."""
        group = set(slots)
        self._bulk_groups.setdefault(lane, []).append(group)
        for slot in group:
            self._reduce_flight[slot] = lane

    def reduce_wait(self, lane: int, pending: int) -> None:
        """`lane` waited until at most `pending` of its bulk groups read on:
        the others' boxes are read, as if by the lane now -- so a write by
        another lane is ordered behind it by a barrier, as on the hardware."""
        groups = self._bulk_groups.get(lane, [])
        cut = max(0, len(groups) - pending)
        for group in groups[:cut]:
            for slot in group:
                if self._reduce_flight.get(slot) == lane:
                    del self._reduce_flight[slot]
                self._reads.setdefault(slot, {})[lane] = self.time
        self._bulk_groups[lane] = groups[cut:]

    def bulk_wait(self, lane: int, cursor: str) -> None:
        """`lane` waited on `cursor`: what the bulk copies through it wrote
        so far is visible to it."""
        self._landed[(cursor, lane)] = self.time

    def _bulk_read(self, slot: int) -> bool:
        """A read of a slot a bulk copy wrote: in flight for a lane that has
        not waited on the copy's cursor since, landed for one that has --
        every lane waits for itself, and no barrier is part of it."""
        found = self._bulk_flight.get(slot)
        if found is None:
            return False
        cursor, when = found
        if self._landed.get((cursor, self.lane), -1) <= when:
            self._report('FLIGHT', slot, -1)
        self._reads.setdefault(slot, {})[self.lane] = self.time
        return True

    def _ordered(self, lane: int, other: int, when: int) -> bool:
        return lane == other or self._met.get((lane, other), -1) > when

    def _report(self, kind: str, slot: int, other: int,
                lane: Optional[int] = None) -> None:
        lane = self.lane if lane is None else lane
        key = (kind, self.statement, lane, other)
        if key in self._seen or len(self.found) >= self.limit:
            return
        self._seen.add(key)
        self.found.append(Race(kind, slot, lane, other, self.statement))

    def read(self, slot: int) -> None:
        if self._bulk_read(slot):
            return
        if slot in self._flight:
            self._report('FLIGHT', slot, self._flight[slot])
        if slot in self._dma_flight:
            self._report('FLIGHT', slot, self._dma_flight[slot][2])
        last = self._writes.get(slot)
        if (last is not None and not self._ordered(self.lane, last[0], last[1])
                and not self._own_wave(slot)):
            self._report('RAW', slot, last[0])
        self._reads.setdefault(slot, {})[self.lane] = self.time

    def write(self, slot: int) -> None:
        if slot in self._reduce_flight:
            # Over a box the accelerator may still be reading.
            self._report('FLIGHT', slot, self._reduce_flight[slot])
        landing = self._bulk_flight.pop(slot, None)
        if landing is not None and self._landed.get(
                (landing[0], self.lane), -1) <= landing[1]:
            self._report('FLIGHT', slot, -1)
        if slot in self._flight and (not self.issuing
                                     or self._flight[slot] != self.lane):
            self._report('FLIGHT', slot, self._flight[slot])
        if slot in self._dma_flight:
            # A write over a copy still landing: which lands last is not said.
            self._report('FLIGHT', slot, self._dma_flight[slot][2])
        last = self._writes.get(slot)
        if (last is not None and not self._ordered(self.lane, last[0], last[1])
                and not self._own_wave(slot)):
            self._report('WAW', slot, last[0])
        for other, when in self._reads.get(slot, {}).items():
            if not self._ordered(self.lane, other, when):
                self._report('WAR', slot, other)
        if self.issuing:
            self._flight[slot] = self.lane
            self._open.setdefault(self.lane, set()).add(slot)
            return
        self._dma_wave.pop(slot, None)
        if self.dma is not None:
            # In flight until a wait of the wave's retires the copy.
            copies = self._dma.setdefault(self.dma, [])
            if not copies or copies[-1][0] != self.time:
                copies.append((self.time, set()))
            copies[-1][1].add(slot)
            self._dma_flight[slot] = (self.dma, self.time, self.lane)
            return
        if self.bulk is not None:
            # In flight until each lane waits on the cursor for itself.
            self._bulk_flight[slot] = (self.bulk, self.time)
            self._writes.pop(slot, None)
            self._reads.pop(slot, None)
            return
        self._writes[slot] = (self.lane, self.time)
        self._reads.pop(slot, None)


class Ptr:
    """A pointer into :class:`Slot`, i.e. a base name plus an offset."""

    __slots__ = ('mem', 'base', 'off')

    def __init__(self, mem: Slot, base: str, off: int = 0):
        self.mem, self.base, self.off = mem, base, int(off)

    def __getitem__(self, i):
        return self.mem.read(self.base, self.off + int(i))

    def __setitem__(self, i, v):
        self.mem.write(self.base, self.off + int(i), v)

    def __add__(self, n):
        return Ptr(self.mem, self.base, self.off + int(n))


class PtrTable:
    """An array of pointers, one per element -- an operand addressed through
    the elements' own pointers (`Addressing.PTR_BASED`): entry `e` points at
    an array of its own, the slots of `base@e`."""

    __slots__ = ('mem', 'base')

    def __init__(self, mem: Slot, base: str):
        self.mem, self.base = mem, base

    def __getitem__(self, e):
        return Ptr(self.mem, f'{self.base}@{int(e)}')


class Indices:
    """The launch's index list (`Addressing.INDEXED`) where a run is given
    none: element `n` is cell `n`, which reads what a strided operand would.
    A preset `elementIndices` replaces it (`evaluate_wave`)."""

    __slots__ = ()

    def __getitem__(self, n):
        return int(n)


class Vec(tuple):
    """A short vector value: `VectorT<T, N>`, `floatN`, or a `{a, b}` list.

    Arithmetic is elementwise, which is what the generated code means by it --
    GNU vector types on HIP, and on CUDA a struct whose operators are
    elementwise as well.  Mixing with a scalar
    is deliberately *not* supported: the generator splats a broadcast operand
    into a full vector before the product, so a scalar meeting a vector here
    means the splat went missing and silently broadcasting it would hide
    exactly that.
    """

    def _zip(self, other, op):
        if not isinstance(other, Vec):
            raise Abort(f'vector op with a scalar {other!r}: the splat is missing')
        if len(self) != len(other):
            raise Abort(f'vector width mismatch: {len(self)} vs {len(other)}')
        return Vec(op(a, b) for a, b in zip(self, other))

    def __add__(self, o): return self._zip(o, lambda a, b: a + b)
    def __sub__(self, o): return self._zip(o, lambda a, b: a - b)
    def __mul__(self, o): return self._zip(o, lambda a, b: a * b)
    def __truediv__(self, o): return self._zip(o, lambda a, b: a / b)
    def __neg__(self): return Vec(-a for a in self)


class _Break(Exception):
    """A `break`, carried to the loop it leaves; `lanes` are the lanes that
    take it, where lanes run together."""

    def __init__(self, lanes=None):
        super().__init__('break')
        self.lanes = lanes


class _LaunchCursor:
    """`ClusterLaunchCursor` on the host: the CTA ids the queue hands this
    block, in order (`launches`), then -1.  Per thread, as on the device:
    every lane advances its own, and they agree because they read one
    script."""

    def __init__(self, launches):
        self.launches = tuple(launches)
        self.at = 0

    def start(self, queue) -> None:
        self.at = 0

    def peek(self, queue) -> int:
        """What `next` hands out, without taking it."""
        return self.launches[self.at] if self.at < len(self.launches) else -1

    def next(self, queue) -> int:
        cta = self.peek(queue)
        if cta >= 0:
            self.at += 1
        return cta


class Abort(Exception):
    """Raised when the interpreter meets something outside its subset."""


#: The texts `wave_uniform` reads as the wave's, by the key `_wave_reads`
#: hands `Interp._wave_uniform`; and the key of each.
_WAVE_TEXTS: List[str] = []
_WAVE_KEYS: Dict[str, int] = {}

_WAVE_UNIFORM = re.compile(
    r'\btensorforge::wave_uniform\s*<\s*(\d+)\s*(?:,[^<>()]*)?>\s*\(')


def _wave_reads(e: str) -> str:
    """Every `tensorforge::wave_uniform<Lanes>(v)` of `e` as
    `WAVE_UNIFORM(Lanes, key)`, with `key` naming the text of `v`."""
    out, at = [], 0
    while True:
        m = _WAVE_UNIFORM.search(e, at)
        if m is None:
            out.append(e[at:])
            return ''.join(out)
        depth, j = 1, m.end()
        while j < len(e) and depth:
            depth += (e[j] == '(') - (e[j] == ')')
            j += 1
        text = e[m.end():j - 1]
        key = _WAVE_KEYS.get(text)
        if key is None:
            key = _WAVE_KEYS[text] = len(_WAVE_TEXTS)
            _WAVE_TEXTS.append(text)
        out.append(f'{e[at:m.start()]}WAVE_UNIFORM({m.group(1)}, {key})')
        at = j


_REDUCTION = re.compile(
    r'\btensorforge::reduction\s*<\s*tensorforge::ReductionOperation\s*<'
    r'[^<>]*,\s*tensorforge::Operation::(\w+)\s*>\s*,\s*(\d+)\s*,\s*(\d+)'
    r'\s*(?:,[^<>()]*)?>\s*\(')


def _fold_reads(e: str) -> str:
    """Every `tensorforge::reduction<ReductionOperation<T, Op>, Block,
    Subblock, T>(v)` of `e` as `REDUCE("Op", Block, Subblock, key)`, with
    `key` naming the text of `v` as `_wave_reads` does."""
    out, at = [], 0
    while True:
        m = _REDUCTION.search(e, at)
        if m is None:
            out.append(e[at:])
            return ''.join(out)
        depth, j = 1, m.end()
        while j < len(e) and depth:
            depth += (e[j] == '(') - (e[j] == ')')
            j += 1
        text = e[m.end():j - 1]
        key = _WAVE_KEYS.get(text)
        if key is None:
            key = _WAVE_KEYS[text] = len(_WAVE_TEXTS)
            _WAVE_TEXTS.append(text)
        out.append(f'{e[at:m.start()]}REDUCE("{m.group(1)}", {m.group(2)}, '
                   f'{m.group(3)}, {key})')
        at = j


_LANE_READ = re.compile(
    r'\btensorforge::(?:readlane|broadcast\s*<\s*(\d+)\s*,\s*(\d+)\s*,'
    r'\s*(\d+)\s*>)\s*\(')


def _lane_reads(e: str) -> str:
    """Every `tensorforge::readlane(v, L)` of `e` as `READLANE(v, L)`, and
    every `tensorforge::broadcast<Block, Subblock, Lane>(v)` as
    `BROADCAST(v, Block, Subblock, Lane)`: `v` as the name it is, in
    quotes, or as the key of its text where it is an expression, as
    `_wave_reads` hands one over.  Either way not as a value: by the time
    Python evaluated it, it would hold this lane's copy, which is the one
    thing the call is not asking for."""
    out, at = [], 0
    while True:
        m = _LANE_READ.search(e, at)
        if m is None:
            out.append(e[at:])
            return ''.join(out)
        depth, j, cut = 1, m.end(), None
        while j < len(e) and depth:
            depth += (e[j] == '(') - (e[j] == ')')
            if e[j] == ',' and depth == 1:
                cut = j
            j += 1
        if m.group(1) is None:
            text, lane = e[m.end():cut].strip(), e[cut + 1:j - 1].strip()
        else:
            text = e[m.end():j - 1].strip()
        if re.fullmatch(r'\w+', text):
            ref = f'"{text}"'
        else:
            key = _WAVE_KEYS.get(text)
            if key is None:
                key = _WAVE_KEYS[text] = len(_WAVE_TEXTS)
                _WAVE_TEXTS.append(text)
            ref = str(key)
        if m.group(1) is None:
            call = f'READLANE({ref}, {lane})'
        else:
            call = (f'BROADCAST({ref}, {m.group(1)}, {m.group(2)}, '
                    f'{m.group(3)})')
        out.append(e[at:m.start()] + call)
        at = j


#: What `tensorforge::ReductionOperation<T, Op>::applyOperation` does, by
#: the operation's name (`include/tensorforge_device/base.h`).
_FOLDS = {
    'Add': lambda a, b: a + b,
    'Mul': lambda a, b: a * b,
    'Min': lambda a, b: b if b < a else a,
    'Max': lambda a, b: b if a < b else a,
    'And': lambda a, b: a & b,
    'Or': lambda a, b: a | b,
    'Xor': lambda a, b: a ^ b,
}


def _pick(count, mult, *values):
    """`tensorforge::pick<count>(mult, values...)`."""
    if len(values) != int(count) or not 0 <= int(mult) < len(values):
        raise Abort(f'pick<{count}>({mult}, ...) among {len(values)} values')
    return values[int(mult)]


def _buffer_load(count, size, resource, index, keep=True):
    """`tensorforge::bufferLoad<T, count>(resource, index, keep)`: `count`
    elements of `size` bytes from element `index` of the resource, zero
    where `keep` is false or any of them lies past it (`_Resource`)."""
    if not isinstance(resource, _Resource) or not size:
        raise Abort(f'a buffer read through {resource!r} of {size} B')
    at = int(index)
    if resource.inside(at, int(count), int(size), keep):
        values = [resource.base[at + k] for k in range(int(count))]
    else:
        values = [0.0] * int(count)
    return values[0] if int(count) == 1 else Vec(values)


def _same(a, b) -> bool:
    """Whether two lanes hold the same value: a pointer by where it points."""
    if isinstance(a, Ptr) and isinstance(b, Ptr):
        return (a.base, a.off) == (b.base, b.off)
    return a == b


def _py(expr: str) -> str:
    """Translate a C expression into a Python one."""
    e = expr
    # `base.h`'s integer literals: `0_i64` is the null a pointer table holds
    # for a masked element (`compute.multi_element`).
    e = re.sub(r'\b(\d+)_i(?:32|64)\b', r'\1', e)
    # A float literal's suffix, and only a literal's: a name may end in a
    # digit and an `f` as well -- a table named by its numbers' digest,
    # `tf_ellw_..c0f` -- which a rule on the last digit alone cut short.
    e = re.sub(r'(?<![\w.])(\d+(?:\.\d*)?(?:[eE][-+]?\d+)?|\.\d+(?:[eE][-+]?\d+)?)'
               r'[fF]\b', r'\1', e)
    # An integer suffix says which type the literal has, which Python's
    # integers do not care about: `0u` is the `0` a `size_t` select holds,
    # and `0x0000ffffu` the mask of a warp barrier.
    e = re.sub(r'\b(0[xX][0-9a-fA-F]+|\d+)(?:[uU][lL]{0,2}|[lL]{1,2}[uU]?)\b',
               r'\1', e)
    e = re.sub(r'\b(?:static_cast|reinterpret_cast|const_cast)\s*<[^>]*>\s*', '', e)
    # the C-style cast to a memory-space pointer a global window is bound
    # through: `(tensorforge::SpacePtrRestrict<double, ...>)&m0[...]`
    e = re.sub(r'\(\s*tensorforge::\w+<[^()]*>\s*(?:const\s*)?\)\s*', '', e)
    # `wave_uniform<Lanes>(v)` reads the first lane's copy of a value every
    # lane of the wave holds alike.  Its text is handed over by a key, since
    # the rewrites below would reach into a string, and evaluated by
    # `Interp._wave_uniform` in every lane of the wave: they have to agree.
    e = _wave_reads(e)
    # `reduction<..., Block, Subblock, T>(v)`, the all-reduce over the lanes
    # of a block (`Interp._reduce`): its text by key, as above.
    e = _fold_reads(e)
    # `pick<G>(m, a, b, ...)`: multiplication `m`'s value among `G`.
    e = re.sub(r'\btensorforge::pick\s*<\s*(\d+)\s*>\s*\(', r'PICK(\1, ', e)
    # `bufferLoad<T, N>(r, i, keep)`: `N` elements of `T` through the
    # resource `r` (`Interp._buffer_load`), the element's size said here.
    e = re.sub(r'\btensorforge::bufferLoad\s*<\s*([\w:]+)\s*(?:,\s*(\d+)\s*)?'
               r'>\s*\(',
               lambda m: (f'BUFLOAD({m.group(2) or 1}, '
                          f'{_BUFFER_ELEMENT.get(m.group(1), 0)}, '), e)
    e = re.sub(r'\b__ldcg\s*\(\s*&', 'DEREF(', e)
    # A load under an L2 policy (`tensorforge::ldL2<P>`): a policy, never a
    # value.
    e = re.sub(r'\btensorforge::ldL2\s*<\s*[\w:]+\s*>\s*\(\s*&', 'DEREF(', e)
    e = re.sub(r'\b__ldg\s*\(\s*&', 'DEREF(', e)
    e = re.sub(r'\b__builtin_nontemporal_load\s*\(\s*&', 'DEREF(', e)
    # `readlane(v, L)` reads lane `L`'s copy of `v`, and `broadcast<Block,
    # Subblock, Lane>(v)`, the template form the operand broadcast takes on
    # a narrow extent, the copy of a lane `_broadcast` resolves, since which
    # lane it reads depends on the reading lane's own index.  `v` a name, or
    # an expression the emitter inlined (`_lane_reads`).
    e = _lane_reads(e)
    # `gather<Block>(v, L)`, the same with `L` a value of each lane's own
    # (`MultilinearInstruction._ell_plan`, a shuffle's `engines.
    # RELAYOUT_STRATEGIES`).  `L` may hold two levels of parentheses -- a
    # table read the emitter inlined, a lane computed in place.
    e = re.sub(r'\btensorforge::gather\s*<\s*(\d+)\s*>\s*\(\s*(\w+)\s*,'
               r'\s*((?:[^()]|\((?:[^()]|\([^()]*\))*\))*)\)',
               r'GATHER("\2", \1, \3)', e)
    # `tensorforge::fma(a, b, c)`, the vector FMA CUDA's lexic names so that
    # sm_100 can pair it.  The same `a * b + c` the infix spelling computes.
    e = re.sub(r'\btensorforge::fma\s*\(', 'FMA(', e)
    # The address of a warp's quarter of tensor memory's lanes (`cuda.h`).
    e = re.sub(r'\btensorforge::tmemWarpAddress\s*\(', 'TMEMADDR(', e)
    # The standard library's functions, which the environment holds by their
    # own names.
    e = re.sub(r'\bstd::(?=\w+\s*\()', '', e)
    # The reinterpretations between a float and a TF32 register: the oracle
    # holds a TF32 half as the float it stands for (`Interp._mma`), so both
    # are the value itself.
    e = re.sub(r'\b__(?:float_as_uint|uint_as_float)\s*\(', '(', e)
    e = e.replace('&&', ' and ').replace('||', ' or ')
    e = re.sub(r'(?<![=!<>&|])!(?!=)', ' not ', e)
    # `*(SomeVecType*)&p[i]` -> `VLOAD(ADDR(p, i), N)`.  Done before the
    # `&p[i]` rewrite below so the address form is still intact to match on.
    def _vec_width(ty: str) -> int:
        m = re.search(r'<[^,>]*,\s*(\d+)\s*>', ty) or re.search(r'(\d)\s*$', ty.strip())
        return int(m.group(1)) if m else 1
    pat = re.compile(r'\*\s*\(([^()]*?)\s*\*\)\s*&(\w+)\[')
    out, i = [], 0
    while i < len(e):
        m = pat.match(e, i)
        if not m:
            out.append(e[i]); i += 1; continue
        depth, j = 1, m.end()
        while j < len(e) and depth:
            depth += (e[j] == '[') - (e[j] == ']')
            j += 1
        out.append(f'VLOAD({_vec_width(m.group(1))}, {m.group(2)}, '
                   + e[m.end():j - 1] + ')')
        i = j
    e = ''.join(out)
    # A vector value-initialized inside an expression -- the zero a guarded
    # wide read selects where its guard fails, `g ? *(VectorT<float, 4>*)&p[i]
    # : tensorforge::VectorT<float, 4>{}` -- is that many zeros; with numbers
    # in the braces it is the literal below, its type said once more.
    e = re.sub(r'\btensorforge::Vector(?:Relaxed)?T\s*<[^<>]*?,\s*(\d+)\s*>'
               r'\s*\{\s*\}',
               lambda m: 'VEC(' + ', '.join(['0.0'] * int(m.group(1))) + ')',
               e)
    e = re.sub(r'\btensorforge::Vector(?:Relaxed)?T\s*<[^<>]*>\s*(?=\{)', '',
               e)
    # `{a, b}` in an expression is a vector literal.  A bare `{}` is
    # zero-initialization and is handled by `_DECL`, so it never reaches here.
    e = re.sub(r'\{\s*([^{}]+?)\s*\}', r'VEC(\1)', e)

    def _subscript(at):
        """The end of the subscript whose `[` stands just ahead of `at`."""
        depth, j = 1, at
        while j < len(e) and depth:
            depth += (e[j] == '[') - (e[j] == ']')
            j += 1
        return j

    out, i = [], 0                                    # &p[i] -> ADDR(p, i)
    while i < len(e):
        m = re.compile(r'&(\w+)\[').match(e, i)
        if not m:
            out.append(e[i]); i += 1; continue
        j = _subscript(m.end())
        first = e[m.end():j - 1]
        if j < len(e) and e[j] == '[':
            # `&p[e][i]`: entry `e` of a table of pointers, and slot `i` of
            # the array it points at.
            k = _subscript(j + 1)
            out.append(f'ADDR({m.group(1)}[{first}], ' + e[j + 1:k - 1] + ')')
            i = k
            continue
        out.append(f'ADDR({m.group(1)}, ' + first + ')')
        i = j
    e = ''.join(out)
    e = e.replace('true', 'True').replace('false', 'False')
    e = e.replace('nullptr', 'None')
    # ternary: rightmost ? : first, and only the group it sits in
    #
    # Taking everything before the `?` as the condition works for an
    # expression that *is* a ternary and for nothing else.  A select emitted
    # inside an fma reads `x + ((c) ? a : b) * y`, where the prefix is
    # `x + ((c` -- unbalanced, and the whole configuration aborts on an
    # expression that is perfectly well formed.  So the group is found from
    # the `?` outwards: back to the paren that opens it, forward to the one
    # that closes it.
    while '?' in e:
        q = e.rindex('?')

        depth = 0
        start = None
        inner = False          # this ternary is the true arm of another
        for i in range(q - 1, -1, -1):
            if e[i] == ')':
                depth += 1
            elif e[i] == '(':
                if depth == 0:
                    start = i
                    break
                depth -= 1
            elif depth == 0 and (e[i] == '?' or (
                    e[i] == ':' and e[i - 1:i + 1] != '::'
                    and e[i:i + 2] != '::')):
                # `a ? b : c ? d : e` is `a ? b : (c ? d : e)`
                start = i
                inner = e[i] == '?'
                break
        if start is None:
            start = -1

        depth = 0
        colon = None
        stop = None
        for i in range(q + 1, len(e)):
            if e[i] == '(':
                depth += 1
            elif e[i] == ')':
                if depth == 0:
                    stop = i
                    break
                depth -= 1
            elif e[i] == ':' and depth == 0 and colon is None:
                colon = i
            elif e[i] == ':' and depth == 0 and inner:
                stop = i                # the outer ternary's own `:`
                break
        if colon is None:
            raise Abort(f'unbalanced ternary in {expr!r}')
        if stop is None:
            stop = len(e)

        cond, a, b = e[start + 1:q], e[q + 1:colon], e[colon + 1:stop]
        e = e[:start + 1] + f'(({a}) if ({cond}) else ({b}))' + e[stop:]
    return e


def _cdiv(a, b):
    """`a / b` as C has it: between integers the quotient truncated toward
    zero, where Python's `/` is not an integer at all."""
    if isinstance(a, int) and isinstance(b, int):
        q = abs(a) // abs(b)
        return q if (a >= 0) == (b >= 0) else -q
    return a / b


def _cmod(a, b):
    """`a % b` as C has it between integers: what is left of `_cdiv`'s
    quotient, signed like `a`, where Python's takes the sign of `b`."""
    if isinstance(a, int) and isinstance(b, int):
        return a - b * _cdiv(a, b)
    return a % b


class _CArithmetic(ast.NodeTransformer):
    """`/` and `%` of a translated expression through `_cdiv` and
    `_cmod`.  Which of the two meanings applies is the operands' type, and
    that is known when the expression is evaluated, not in its text."""

    def visit_BinOp(self, node):
        self.generic_visit(node)
        name = {ast.Div: 'CDIV', ast.Mod: 'CMOD'}.get(type(node.op))
        if name is None:
            return node
        return ast.copy_location(
            ast.Call(ast.Name(name, ast.Load()), [node.left, node.right], []),
            node)


#: What a compiled expression finds besides a lane's own names.
_GLOBALS = {'__builtins__': {}, 'CDIV': _cdiv, 'CMOD': _cmod}


@functools.lru_cache(maxsize=None)
def _compiled(expr: str):
    """`expr` translated (`_py`) and compiled, once for every lane and
    every trip that evaluates it."""
    tree = _CArithmetic().visit(ast.parse(_py(expr).strip(), mode='eval'))
    return compile(ast.fix_missing_locations(tree), '<kernel>', 'eval')


class Interp:
    def __init__(self, mem: Slot, env: Dict[str, object], limit: int = 400000):
        self.mem = mem
        self.env = dict(env)
        self.budget = limit
        #: Which lane this is, for `Races`.
        self.lane = getattr(self.env.get('threadIdx'), 'x', 0)
        #: The CTA ids a launch-control queue hands this block (`_LaunchCursor`).
        self.launches: Tuple[int, ...] = ()
        #: wgmma products this lane issued and no commit closed yet, the
        #: groups committed and not retired (oldest first), and the value
        #: each register written by one in flight will hold, with how many
        #: products in flight write it (`_wgmma`, `_wgmma_wait`).
        self.wg_open: List[Tuple] = []
        self.wg_groups: List[List[Tuple]] = []
        self.wg_pending: Dict[Any, float] = {}
        self.wg_writers: Dict[Any, int] = {}
        #: Element size per pointer name, from the type in its declaration.
        #: An async copy is measured in bytes and the memory here is untyped,
        #: so this is the only place the two can be reconciled.
        self.widths: Dict[str, int] = {}
        self.env['DEREF'] = lambda p, i=0: p[i] if isinstance(p, Ptr) else p
        self.env['ADDR'] = lambda p, i: (p + i) if isinstance(p, Ptr) else p
        #: The other lanes of the wave, set by `Lockstep`.  Empty for a
        #: single-lane run, where a cross-lane read has no defined answer and
        #: says so instead of quietly returning this lane's copy.
        self.peers: List['Interp'] = []
        #: The run this lane takes part in, set by `Lockstep`.
        self.lockstep: Optional['Lockstep'] = None
        self.env['READLANE'] = self._readlane
        self.env['BROADCAST'] = self._broadcast
        self.env['GATHER'] = self._gather
        self.env['WAVE_UNIFORM'] = self._wave_uniform
        self.env['REDUCE'] = self._reduce
        self.env['PICK'] = _pick
        self.env['BUFLOAD'] = _buffer_load
        self.env['VEC'] = lambda *xs: Vec(xs)
        self.env['FMA'] = lambda a, b, c: a * b + c
        # `tensorforge::tmemWarpAddress`: the lane 32 (w % 4) of the warp w
        # this thread is in, in the address's upper half.
        self.env['TMEMADDR'] = lambda base, column: (
            int(base) + ((32 * ((self.lane // 32) % 4)) << 16) + int(column))
        self.env['VLOAD'] = lambda n, p, i: Vec(p[i + k] for k in range(n))
        for fn in ('min', 'max', 'abs'):
            self.env[fn] = __builtins__[fn] if isinstance(__builtins__, dict) \
                else getattr(__builtins__, fn)
        self.env['fabs'] = abs
        # The libraries spell an elementwise `max`/`min` of floating-point
        # numbers this way.
        self.env['fmax'] = self.env['max']
        self.env['fmin'] = self.env['min']
        import math
        def exp(x):
            # Past the largest double, what `std::exp` gives: infinity.
            try:
                return math.exp(x)
            except OverflowError:
                return math.inf
        for name, fn in (('sqrt', lambda x: abs(x) ** 0.5), ('exp', exp),
                         ('log', lambda x: math.log(abs(x) + 1e-9)),
                         ('pow', lambda a, b: abs(a) ** b),
                         ('tanh', math.tanh), ('sin', math.sin),
                         ('cos', math.cos), ('erf', math.erf)):
            self.env[name] = fn
        # The neutral element a max- or min-fold starts from (`<cmath>`'s
        # macro).
        self.env['INFINITY'] = math.inf

    def _readlane(self, name, lane):
        """Lane `lane`'s copy of `name` -- or, where `name` is the key of a
        text (`_lane_reads`), what that text comes to in lane `lane`.

        Requires a lockstep run: the lanes have to have reached this point
        together, or the peer's binding is from a different statement.  In a
        single-lane run there are no peers and this aborts rather than
        returning the local copy -- a cross-lane read that silently reads
        across nothing is how a broadcast disappears from the model without
        anything noticing.
        """
        lane = int(lane)
        if not self.peers:
            raise Abort(f'readlane({name!r}, {lane}) outside a lockstep run')
        if not 0 <= lane < len(self.peers):
            raise Abort(f'readlane({name!r}, {lane}): no such lane')
        peer = self.peers[lane]
        if not isinstance(name, str):
            return peer.ev(_WAVE_TEXTS[int(name)])
        if name not in peer.env:
            # The lane was masked off where this value was defined.  On the
            # hardware the register holds whatever it held before, which is
            # not something to invent a number for.
            raise Abort(f'readlane({name!r}, {lane}): lane {lane} never '
                        f'defined it -- it was masked off there')
        return peer.env[name]

    def _reduce(self, op: str, block: int, subblock: int, key: int):
        """`tensorforge::reduction<ReductionOperation<T, op>, block,
        subblock, T>(text)`, the text `key` names, for this lane: the fold
        of every `subblock`-th lane of this lane's run of `block` lanes, as
        `include/tensorforge_device/cuda.h` computes it -- a butterfly that
        combines with the lane at distance `block / 2` first, down to
        `subblock`, in that order, which a floating-point sum depends on;
        a vote for a flag.  Every lane of the run has to take part: the
        hardware's exchange reads a register of each.  Once per statement
        and run of lanes (`Lockstep.wave_cache`)."""
        fold = _FOLDS.get(op)
        run = self.lockstep
        if fold is None:
            raise Abort(f'reduction over {op!r}')
        if run is None:
            raise Abort(f'reduction<{op}> outside a lockstep run')
        block, subblock = int(block), int(subblock)
        me = self.env['threadIdx']
        group = (me.y, me.z, me.x // block)
        cached = run.wave_cache.get(('reduce', group, block, subblock, key))
        if cached is None:
            text = _WAVE_TEXTS[int(key)]
            values = {}
            for it in run.current:
                t = it.env['threadIdx']
                if (t.y, t.z, t.x // block) == group:
                    values[t.x] = it.ev(text)
            if len(values) != block:
                raise Abort(f'reduction<{op}, {block}> over {len(values)} '
                            f'lanes taking part')
            if all(isinstance(v, bool) for v in values.values()):
                if op not in ('And', 'Or'):
                    raise Abort(f'reduction<{op}> of flags')
                cached = {}
                for x in values:
                    mine = [values[y] for y in values
                            if y % subblock == x % subblock]
                    cached[x] = all(mine) if op == 'And' else any(mine)
            else:
                cached = dict(values)
                step = block // 2
                while step >= subblock:
                    cached = {x: fold(v, cached[x ^ step])
                              for x, v in cached.items()}
                    step //= 2
            run.wave_cache[('reduce', group, block, subblock, key)] = cached
        return cached[me.x]

    def _wave_uniform(self, lanes: int, key: int):
        """`tensorforge::wave_uniform<lanes>(text)`, the text `key` names:
        its value in the first lane of this lane's `lanes`, which every lane
        of them that executes it holds alike.

        Checked rather than taken on trust: the lanes of the run that take
        part are asked for their copies, and a value they do not agree on is
        an `Abort`, since the hardware would hand every lane the first one's.
        Once per statement and group of lanes (`Lockstep.wave_cache`).  A
        lane run on its own has nothing to ask, and takes its own.
        """
        text = _WAVE_TEXTS[int(key)]
        run = self.lockstep
        if run is None:
            return self.ev(text)
        lanes = int(lanes)
        group = self.lane // lanes
        cached = run.wave_cache.get((group, lanes, key), self)
        if cached is not self:
            return cached
        values = [it.ev(text) for it in run.current
                  if it.lane // lanes == group]
        if any(not _same(v, values[0]) for v in values[1:]):
            raise Abort(f'wave_uniform<{lanes}>({text}): the lanes of the wave '
                        f'hold different values')
        run.wave_cache[(group, lanes, key)] = values[0]
        return values[0]

    def _broadcast(self, name: str, block: int, subblock: int, lane: int):
        """`tensorforge::broadcast<Block, Subblock, Lane>(name)` for this lane.

        Modeled from the definition in `include/tensorforge_device/cuda.h`
        rather than from what it is used for.  The degenerate case returns the
        lane's own copy; otherwise it is a `__shfl_sync` with a *width*, which
        splits the warp into segments of `Block` lanes and resolves the source
        within the caller's own segment:

            src = (L // Block) * Block
                  + ((Subblock * Lane + L % Subblock) % Block)

        Without this the narrow extents -- the ones whose operand broadcast
        takes the template form -- would have no numerical coverage at all:
        every such case would abort, and an abort that a caller turns into a
        skip looks exactly like a pass.
        """
        block, subblock, lane = int(block), int(subblock), int(lane)
        if block == 1 or block == subblock:
            if not isinstance(name, str):
                return self.ev(_WAVE_TEXTS[int(name)])
            if name not in self.env:
                raise Abort(f'broadcast({name!r}): not defined in this lane')
            return self.env[name]
        me = self.env['threadIdx'].x
        src = (me // block) * block + ((subblock * lane + me % subblock)
                                       % block)
        return self._readlane(name, src)

    def _gather(self, name: str, block: int, lane):
        """`tensorforge::gather<Block>(name, lane)` for this lane: lane
        `lane` of this lane's own run of `Block` lanes, as `__shfl_sync`
        with a width resolves it (`include/tensorforge_device/cuda.h`)."""
        block, me = int(block), self.env['threadIdx'].x
        return self._readlane(name, (me // block) * block + int(lane) % block)

    def _mma(self, mma: _Mma, stmt: str) -> None:
        """One `mma.sync`, for the warp this lane is in.

        The instruction is the warp's: every lane hands in its fragments
        and every lane gets its part of `D` back, so it is evaluated once
        per statement and warp, from the operands every lane holds before
        any of them has written its result (`Lockstep.wave_cache`).  The
        warp is the hardware's -- 32 consecutive threads of the block -- so
        the lanes of several multiplications that share one take part
        together, and a warp that is not whole at the statement aborts:
        the instruction is `.aligned`.

        The fragment layouts are PTX's, stated here from the ISA's tables
        rather than taken from the emitter, which is what makes this a check
        of it: lane `l` of the warp is `g = l / 4` and `t = l % 4`; register
        `i` of A holds row `g + 8 (i % (m / 8))`, column `t + 4 (i / (m /
        8))`; register `i` of B row `t + 4 i`, column `g`; register `i` of C
        and D row `g + 8 (i / 2)`, column `2 t + i % 2`.  That covers every
        shape the path issues: `m16n8k4` and `m16n8k8` in TF32, `m8n8k4`,
        `m16n8k4`, `m16n8k8` and `m16n8k16` in FP64.

        A TF32 operand is read as the instruction reads its register
        (`_tf32`); the products are summed in double precision, as every
        other sum of the oracle is.
        """
        run = self.lockstep
        if run is None:
            raise Abort(f'mma.sync outside a lockstep run: {stmt!r}')
        warp = self.lane // 32
        key = ('mma', warp, stmt)
        result = run.wave_cache.get(key)
        if result is None:
            lanes = {it.lane % 32: it for it in run.current
                     if it.lane // 32 == warp}
            if len(lanes) != 32:
                raise Abort(f'mma.sync with {len(lanes)} lanes of the warp '
                            f'taking part')
            m, n, k = mma.m, mma.n, mma.k
            mregs = m // 8
            if (m % 8 or n != 8 or len(mma.a) != m * k // 32
                    or len(mma.b) != k * n // 32 or len(mma.c) != m * n // 32
                    or len(mma.d) != len(mma.c)):
                raise Abort(f'mma.sync of a shape not modeled: {stmt!r}')
            read = _tf32 if mma.atype == 'tf32' else (lambda v: v)
            A = [[0.0] * k for _ in range(m)]
            B = [[0.0] * n for _ in range(k)]
            C = [[0.0] * n for _ in range(m)]
            for l, it in lanes.items():
                g, t = l // 4, l % 4
                for i, text in enumerate(mma.a):
                    A[g + 8 * (i % mregs)][t + 4 * (i // mregs)] = \
                        read(it.ev(text))
                for i, text in enumerate(mma.b):
                    B[t + 4 * i][g] = read(it.ev(text))
                for i, text in enumerate(mma.c):
                    C[g + 8 * (i // 2)][2 * t + i % 2] = it.ev(text)
            result = {}
            for l in lanes:
                g, t = l // 4, l % 4
                out = []
                for i in range(len(mma.d)):
                    r, c = g + 8 * (i // 2), 2 * t + i % 2
                    out.append(C[r][c] + sum(A[r][x] * B[x][c]
                                             for x in range(k)))
                result[l] = out
            run.wave_cache[key] = result
        for text, value in zip(mma.d, result[self.lane % 32]):
            am = re.match(r'^(\w+)\s*\[(.*)\]$', text)
            if am:
                self.env[am.group(1)][self.ev(am.group(2))] = value
            elif re.match(r'^\w+$', text):
                self.env[text] = value
            else:
                raise Abort(f'mma.sync writes {text!r}')

    def _register(self, text: str):
        """The register an operand expression names: a variable, or an
        element of an array -- as a key, and how to write it."""
        am = re.match(r'^(\w+)\s*\[(.*)\]$', text)
        if am:
            return (am.group(1), int(self.ev(am.group(2))))
        if re.match(r'^\w+$', text):
            return text
        raise Abort(f'wgmma writes {text!r}')

    def _store_register(self, key, value) -> None:
        if isinstance(key, tuple):
            self.env[key[0]][key[1]] = value
        else:
            self.env[key] = value

    def _wgmma_current(self, text: str):
        """What a product reads of the register `text` as C: the value a
        product in flight will leave there, where one writes it -- products
        of one shape accumulate in their order -- and its own value
        otherwise."""
        key = self._register(text)
        if key in self.wg_pending:
            return self.wg_pending[key]
        return self.ev(text)

    @staticmethod
    def _descriptor_operand(lanes, text: str, rows: int, k: int,
                            what: str, stmt: str, form: str = 'wgmma'):
        """The `rows` x `k` operand the descriptor `text` names, as the
        warpgroup's lanes hold it -- alike, or the instruction has none to
        read -- from shared memory, by the canonical K-major layout its bits
        state (PTX ISA, "Asynchronous Warpgroup Level Matrix Shared Memory
        Layout" and "Matrix Descriptor Format"): core matrices of eight rows
        of 16 bytes, the leading byte offset (bits 16 to 29, in 16-byte
        units) between those next to each other along K and the stride byte
        offset (bits 32 to 45) between those next to each other along the
        rows; swizzled (bits 62 and 63), atoms of eight rows of 32, 64 or
        128 bytes, K along a row and the stride byte offset between atoms,
        each address's 16-byte chunk XORed with its row in the atom.  An
        element is four bytes, a TF32 operand read as the instruction reads
        it (`_tf32`), at the address its slot has with the buffer at 0 --
        the operand is placed on its atom.  Returns the matrix and the
        slots read, for the check where the product is retired.

        tcgen05's format (`form`) the same but for its fixed bits -- 0b001
        from bit 46, the leading offset relative (bit 52 clear) -- and its
        swizzle, three bits from bit 61 ("tcgen05 Shared Memory
        Descriptor")."""
        descs = [it.ev(text) for it in lanes.values()]
        first = descs[0]
        if not isinstance(first, _MatrixDesc):
            raise Abort(f'wgmma {what} through no descriptor: {stmt!r}')
        if any(not isinstance(d, _MatrixDesc) or d.layout != first.layout
               or d.start.base != first.start.base
               or d.start.off != first.start.off for d in descs):
            raise Abort(f'the warpgroup\'s lanes hold different descriptors '
                        f'of {what}: {stmt!r}')
        layout = first.layout
        lbo = ((layout >> 16) & 0x3FFF) << 4
        sbo = ((layout >> 32) & 0x3FFF) << 4
        if form == 'tcgen05':
            swizzle = _TC_SWIZZLE.get((layout >> 61) & 7)
            if swizzle is None or (layout >> 46) & 7 != 1 \
                    or (layout >> 52) & 1:
                raise Abort(f'tcgen05 {what}: a descriptor of a form not '
                            f'modeled: {stmt!r}')
        else:
            swizzle = _WGMMA_SWIZZLE[(layout >> 62) & 3]
            if (layout >> 46) & 7:
                raise Abort(f'wgmma {what}: a descriptor of tcgen05\'s '
                            f'format: {stmt!r}')
        if (layout >> 49) & 7 or layout & 0x3FFF:
            raise Abort(f'{form} {what}: a descriptor with a base offset or '
                        f'an address in its layout: {stmt!r}')
        bits = {0: 0, 32: 1, 64: 2, 128: 3}[swizzle]
        size = 4
        start = first.start.off * size
        matrix = [[0.0] * k for _ in range(rows)]
        reads = []
        for r in range(rows):
            for x in range(k):
                kb = x * size
                if swizzle:
                    if kb >= swizzle:
                        raise Abort(f'wgmma {what}: K of {k} past a row of '
                                    f'{swizzle} bytes')
                    off = (r // 8) * sbo + (r % 8) * swizzle + kb
                else:
                    off = (r // 8) * sbo + (kb // 16) * lbo + (r % 8) * 16 \
                        + kb % 16
                address = start + off
                address ^= ((address >> 7) & ((1 << bits) - 1)) << 4
                slot = address // size
                value = first.start.mem.read(first.start.base, slot)
                reads.append((first.start.base, slot, value))
                matrix[r][x] = _tf32(value)
        return matrix, reads

    def _wgmma(self, wg: _Wgmma, stmt: str) -> None:
        """One `wgmma.mma_async`, for the warpgroup this lane is in: the
        four warps of 128 consecutive threads of the block.

        Evaluated once per statement and warpgroup, from the operands every
        lane holds before any of them has executed it
        (`Lockstep.wave_cache`); a warpgroup that is not whole at the
        statement aborts: the instruction is `.aligned`.  The layouts are
        PTX's, stated here from the ISA rather than taken from the emitter
        (PTX ISA, "wgmma ... Register Fragments"): lane `l` of warp `w` of
        the warpgroup is `g = l / 4` and `t = l % 4`; register `i` of D
        holds row `16 w + g + 8 ((i % 4) / 2)`, column `8 (i / 4) + 2 t +
        i % 2`; register `i` of A, where A is in registers, row `16 w + g +
        8 (i % 2)`, column `t + 4 (i / 2)`.  A and B in shared memory are
        read through their descriptors (`_descriptor_operand`), B as its N
        rows of K.  `scale-d` 0 makes D = A * B, 1 makes D = A * B + D.

        Nothing is written here: the products of each lane are noted as in
        flight, its D registers then hold nothing readable (`_InFlight`),
        and the values land at the wait that retires the group the commit
        closed them into (`_wgmma_wait`).  What they read is checked
        unchanged there."""
        run = self.lockstep
        if run is None:
            raise Abort(f'wgmma outside a lockstep run: {stmt!r}')
        team = self.lane // 128
        key = ('wgmma', team, stmt)
        cached = run.wave_cache.get(key)
        if cached is None:
            lanes = {it.lane % 128: it for it in run.current
                     if it.lane // 128 == team}
            if len(lanes) != 128:
                raise Abort(f'wgmma with {len(lanes)} lanes of the warpgroup '
                            f'taking part')
            m, n, k = wg.m, wg.n, wg.k
            if ((m, k) != (64, 8) or n % 8 or not 8 <= n <= 256
                    or wg.atype != 'tf32' or len(wg.d) != n // 2
                    or (not wg.a_desc and len(wg.a) != 4)):
                raise Abort(f'wgmma of a shape not modeled: {stmt!r}')
            reads = []
            if wg.a_desc:
                A, got = self._descriptor_operand(lanes, wg.a[0], m, k, 'A',
                                                  stmt)
                reads += got
            else:
                A = [[0.0] * k for _ in range(m)]
                for l, it in lanes.items():
                    w, g, t = l // 32, (l % 32) // 4, l % 4
                    for i, text in enumerate(wg.a):
                        A[16 * w + g + 8 * (i % 2)][t + 4 * (i // 2)] = \
                            _tf32(it.ev(text))
            B, got = self._descriptor_operand(lanes, wg.b, n, k, 'B', stmt)
            reads += got
            scales = {bool(it.ev(wg.scale)) for it in lanes.values()}
            if len(scales) != 1:
                raise Abort(f'the warpgroup\'s lanes disagree on scale-d: '
                            f'{stmt!r}')
            accumulate = scales.pop()
            result = {}
            for l, it in lanes.items():
                w, g, t = l // 32, (l % 32) // 4, l % 4
                out = []
                for i, text in enumerate(wg.d):
                    r = 16 * w + g + 8 * ((i % 4) // 2)
                    c = 8 * (i // 4) + 2 * t + i % 2
                    value = sum(A[r][x] * B[c][x] for x in range(k))
                    if accumulate:
                        value = it._wgmma_current(text) + value
                    out.append(value)
                a_regs = () if wg.a_desc else tuple(
                    (text, it.ev(text)) for text in wg.a)
                result[l] = (out, a_regs)
            cached = (result, tuple(reads))
            run.wave_cache[key] = cached
        result, reads = cached
        values, a_regs = result[self.lane % 128]
        written = []
        for text, value in zip(wg.d, values):
            reg = self._register(text)
            self.wg_pending[reg] = value
            self.wg_writers[reg] = self.wg_writers.get(reg, 0) + 1
            self._store_register(reg, _InFlight(text))
            written.append(reg)
        self.wg_open.append((tuple(written), a_regs, reads, stmt))

    def _tc_state(self) -> Dict:
        """What tcgen05 has in flight in the block: the products issued and
        not committed (`open`), those committed onto each cursor's mbarrier
        and not landed (`bound`), and the value each cell of tensor memory a
        product in flight writes will hold, with how many write it."""
        state = getattr(self.mem, 'tc_state', None)
        if state is None:
            state = {'open': [], 'bound': {}, 'overlay': {}, 'writers': {}}
            self.mem.tc_state = state
        return state

    def _tmem_cell(self, lane: int, column: int, stmt: str, what: str):
        columns = getattr(self.mem, 'tmem_columns', None)
        if columns is None or not 0 <= lane < 128 or not 0 <= column < columns:
            raise Abort(f'{what} of tensor memory at lane {lane}, column '
                        f'{column}, outside what the block allocated '
                        f'({columns} columns): {stmt!r}')
        return lane * _TMEM_ROW + column

    def _tcgen05_asm(self, stmt: str) -> None:
        """A tcgen05 instruction written as inline asm: the product, or a
        warp's load or store of tensor memory."""
        template, outs, ins = _asm_operands(stmt)
        mm = _TC_MMA.search(template)
        if mm:
            self._tcgen05_mma(mm, template, ins, stmt)
            return
        ls = _TMEM_LDST.search(template)
        if ls:
            self._tmem_ldst(ls, outs, ins, stmt)
            return
        raise Abort(f'tcgen05 instruction not modeled: {stmt!r}')

    def _tcgen05_mma(self, mm, template: str, ins, stmt: str) -> None:
        """One `tcgen05.mma.cta_group::1.kind::tf32`, issued by the thread
        whose predicate `e` holds, for the block.

        The instruction descriptor as PTX states it ("tcgen05 Instruction
        descriptor"): D's format FP32 (bits 4-5: 1), A's and B's TF32 (bits
        7-9 and 10-12: 2), both K-major (bits 15 and 16 clear), N >> 3
        from bit 17 and M >> 4 from bit 24 -- M = 128 modeled, row r of D
        lane r of tensor memory, its column j the column `d + j` ("Data path
        layout": M = 128, one CTA).  A through its descriptor
        (`_descriptor_operand`, tcgen05's format) or from tensor memory,
        lane r its row, K along the columns; B through its descriptor, its
        N rows of K.  `enable-input-d` 0 makes D = A * B.

        Nothing is written here: the product is noted as in flight, and its
        cells of tensor memory hold its values from the wait for the phase
        of the mbarrier its commit named on (`_tc_land`)."""
        setp = re.search(r'setp\.ne\.b32\s+e\s*,\s*%(\d+)\s*,\s*0\s*;',
                         template)
        if setp is None:
            raise Abort(f'tcgen05.mma issued by no predicate modeled: '
                        f'{stmt!r}')
        if not bool(self.ev(ins[int(setp.group(1))])):
            return
        idesc = int(self.ev(ins[int(mm.group('i'))]))
        n = ((idesc >> 17) & 0x3F) << 3
        m = ((idesc >> 24) & 0x1F) << 4
        if ((idesc >> 4) & 3, (idesc >> 7) & 7, (idesc >> 10) & 7,
                (idesc >> 15) & 3, m) != (1, 2, 2, 0, 128) or n % 16:
            raise Abort(f'tcgen05.mma of an instruction descriptor not '
                        f'modeled ({idesc:#x}): {stmt!r}')
        k = 8
        d_address = int(self.ev(ins[int(mm.group('d'))]))
        if d_address >> 16:
            raise Abort(f'tcgen05.mma with D from lane {d_address >> 16}, '
                        f'where an M = 128 product starts at lane 0: '
                        f'{stmt!r}')
        d_column = d_address & 0xFFFF
        state = self._tc_state()
        reads = []
        a_text = mm.group('a')
        if a_text.startswith('['):
            a_address = int(self.ev(ins[int(a_text[2:-1])]))
            A = [[0.0] * k for _ in range(m)]
            for r in range(m):
                for x in range(k):
                    cell = self._tmem_cell(r + (a_address >> 16),
                                           (a_address & 0xFFFF) + x, stmt,
                                           'A')
                    if cell in state['overlay']:
                        raise Abort(f'tcgen05.mma reads A from tensor memory '
                                    f'a product in flight writes: {stmt!r}')
                    value = self.mem.read('tmem', cell)
                    reads.append(('tmem', cell, value))
                    A[r][x] = _tf32(value)
        else:
            A, got = self._descriptor_operand(
                {0: self}, ins[int(a_text[1:])], m, k, 'A', stmt, 'tcgen05')
            reads += got
        B, got = self._descriptor_operand(
            {0: self}, ins[int(mm.group('b'))], n, k, 'B', stmt, 'tcgen05')
        reads += got
        en = mm.group('en')
        if en == 'p':
            pset = re.search(r'setp\.ne\.b32\s+p\s*,\s*%(\d+)\s*,\s*0\s*;',
                             template)
            if pset is None:
                raise Abort(f'enable-input-d set by nothing modeled: '
                            f'{stmt!r}')
            enable = bool(self.ev(ins[int(pset.group(1))]))
        else:
            enable = en == '1'
        writes = []
        for r in range(m):
            for c in range(n):
                cell = self._tmem_cell(r, d_column + c, stmt, 'D')
                value = sum(A[r][x] * B[c][x] for x in range(k))
                if enable:
                    # What D holds: the value of a product in flight that
                    # writes it -- products accumulate in their order --,
                    # tensor memory's otherwise.
                    value += (state['overlay'][cell]
                              if cell in state['overlay']
                              else self.mem.read('tmem', cell))
                state['overlay'][cell] = value
                state['writers'][cell] = state['writers'].get(cell, 0) + 1
                writes.append(cell)
        state['open'].append((tuple(writes), tuple(reads), stmt))

    def _tc_land(self, cursor: str) -> None:
        """The products committed onto `cursor`'s mbarrier, landed: the
        values of the cells no product still in flight writes go to tensor
        memory, and what each read is checked unchanged -- a write of an
        operand ahead of the wait is a race with the engine."""
        state = getattr(self.mem, 'tc_state', None)
        if state is None:
            return
        for writes, reads, issued in state['bound'].pop(cursor, []):
            for base, slot, value in reads:
                if not _same(self.mem.read(base, slot), value):
                    raise Abort(f'{base}[{slot}] written while the tcgen05 '
                                f'product reading it is in flight: {issued!r}')
            for cell in writes:
                state['writers'][cell] -= 1
                if not state['writers'][cell]:
                    del state['writers'][cell]
                    self.mem.write('tmem', cell, state['overlay'].pop(cell))

    def _tmem_ldst(self, ls, outs, ins, stmt: str) -> None:
        """A warp's `tcgen05.ld` or `tcgen05.st` of `.32x32b.x{N}`: thread t
        of warp w the lane 32 (w % 4) + t, N columns from the address's on
        -- the address's lane the warp's quarter, which is all it reaches
        ("Tensor Memory", "Access restrictions").  A cell a product in
        flight writes is neither read nor written."""
        count = int(ls.group('n'))
        load = ls.group('op') == 'ld'
        address = int(self.ev(ins[-1] if load else ins[0]))
        quarter = 32 * ((self.lane // 32) % 4)
        if address >> 16 != quarter:
            raise Abort(f'warp {self.lane // 32} reaches the lanes '
                        f'{quarter} on, not {address >> 16}: {stmt!r}')
        state = self._tc_state()
        lane = quarter + self.lane % 32
        exprs = outs if load else ins[1:]
        if len(exprs) != count:
            raise Abort(f'tcgen05.{ls.group("op")} of {count} columns with '
                        f'{len(exprs)} operands: {stmt!r}')
        for i, text in enumerate(exprs):
            cell = self._tmem_cell(lane, (address & 0xFFFF) + i, stmt,
                                   ls.group('op'))
            if cell in state['overlay']:
                raise Abort(f'tensor memory at lane {lane}, column '
                            f'{(address & 0xFFFF) + i} reached while a '
                            f'product writes it: {stmt!r}')
            if load:
                self._store_register(self._register(text),
                                     self.mem.read('tmem', cell))
            else:
                self.mem.write('tmem', cell, self.ev(text))

    def _wgmma_wait(self, pending: int, stmt: str) -> None:
        """`wgmma.wait_group N` for this lane: every group but the `N` most
        recent ones retired, oldest first -- the values their products
        leave land in the registers no product still in flight writes, and
        what each read is checked unchanged, its A registers and the slots
        of shared memory it read through its descriptors: a write of them
        ahead of the wait is a race with the engine."""
        mem = self.mem
        while len(self.wg_groups) > pending:
            for written, a_regs, reads, issued in self.wg_groups.pop(0):
                for base, slot, value in reads:
                    if not _same(mem.read(base, slot), value):
                        raise Abort(f'{base}[{slot}] written while the wgmma '
                                    f'reading it is in flight: {issued!r}')
                for text, value in a_regs:
                    if not _same(self.ev(text), value):
                        raise Abort(f'{text} written while the wgmma reading '
                                    f'it as A is in flight: {issued!r}')
                for reg in written:
                    self.wg_writers[reg] -= 1
                    if not self.wg_writers[reg]:
                        del self.wg_writers[reg]
                        self._store_register(reg, self.wg_pending.pop(reg))

    def _matrix_operands(self, m, stmt: str, what: str):
        """`(count, parts, lanes)` of a matrix transfer: how many matrices,
        its operands split at the top level, and the lanes of this lane's
        warp by their place in it -- all 32, as the instruction is
        `.aligned`: a warp that is not whole at it aborts."""
        run = self.lockstep
        if run is None:
            raise Abort(f'{what} outside a lockstep run: {stmt!r}')
        count = int(m.group('count'))
        parts = [p.strip() for p in _top_level(m.group('args'), ',')]
        if len(parts) != count + 2:
            raise Abort(f'{what} of {count} matrices with {len(parts)} '
                        f'operands: {stmt!r}')
        warp = self.lane // 32
        lanes = {it.lane % 32: it for it in run.current
                 if it.lane // 32 == warp}
        if len(lanes) != 32:
            raise Abort(f'{what} with {len(lanes)} lanes of the warp taking '
                        f'part')
        return count, parts, lanes

    def _ldmatrix(self, lm, stmt: str) -> None:
        """One `ldmatrixX{n}`, for the warp this lane is in: once per
        statement and warp, from the rows its lanes name.

        Modeled from PTX's statement of the instruction ("Warp-level matrix
        load instruction: ldmatrix") rather than from what it is used for:
        lane `l` names the 16-byte row `l % 8` of matrix `l / 8`, four
        floats from `base + index` -- the first `8 n` lanes, whose rows the
        `n` matrices are -- and register `i` of lane `t` is the 32-bit word
        `t % 4` of row `t / 4` of matrix `i`."""
        count, parts, lanes = self._matrix_operands(lm, stmt, 'ldmatrix')
        run = self.lockstep
        key = ('ldmatrix', self.lane // 32, stmt)
        rows = run.wave_cache.get(key)
        if rows is None:
            rows = {}
            for l, it in lanes.items():
                if l >= 8 * count:
                    continue
                base = it.ev(parts[count])
                at = int(it.ev(parts[count + 1]))
                rows[l] = [base[at + w] for w in range(4)]
            run.wave_cache[key] = rows
        t = self.lane % 32
        for i in range(count):
            self.env[parts[i]] = rows[8 * i + t // 4][t % 4]

    def _stmatrix(self, sm, stmt: str) -> None:
        """One `stmatrixX{n}`: the row this lane names, of the matrices the
        first `8 n` lanes name the rows of -- lane `l` row `l % 8` of matrix
        `l / 8`, at `base + index` -- from the registers of the lanes that
        hold it: word `w` of row `r` of matrix `i` is register `i` of lane
        `4 r + w`.

        `_ldmatrix` the other way round, from PTX's statement ("Warp-level
        matrix store instruction: stmatrix").  Each lane writes the row it
        names, so that the race check sees every write as the lane's whose
        address it went to."""
        count, parts, lanes = self._matrix_operands(sm, stmt, 'stmatrix')
        l = self.lane % 32
        if l >= 8 * count:
            return
        i, r = divmod(l, 8)
        base = self.ev(parts[0])
        at = int(self.ev(parts[1]))
        for w in range(4):
            base[at + w] = lanes[4 * r + w].ev(parts[2 + i])

    def ev(self, expr: str):
        self.budget -= 1
        if self.budget < 0:
            raise Abort('budget exhausted')
        try:
            return eval(_compiled(expr), _GLOBALS, self.env)
        except Abort:
            raise
        except Exception as exc:
            raise Abort(f'{expr!r}: {exc}')

    def run(self, block: List) -> None:
        for node in block:
            kind = node[0]
            if kind == 'expr':
                self.assign(node[1])
            elif kind == 'if':
                if self.ev(node[1]):
                    self.run(node[2])
                elif node[3] is not None:
                    self.run(node[3])
            elif kind == 'for':
                v, a, cmp_, b, step = node[1:6]
                self.env[v] = self.ev(a)
                inc = 1 if '++' in step or '+=' in step else -1
                if '+=' in step or '-=' in step:
                    inc = self.ev(step.split('=')[1]) * (1 if '+=' in step else -1)
                while (self.env[v] < self.ev(b) if '<' in cmp_
                       else self.env[v] > self.ev(b)):
                    self.run(node[6])
                    self.env[v] += inc
                    self.budget -= 1
                    if self.budget < 0:
                        raise Abort('budget exhausted')
            elif kind == 'while':
                while self.ev(node[1]):
                    try:
                        self.run(node[2])
                    except _Break:
                        break
                    self.budget -= 1
                    if self.budget < 0:
                        raise Abort('budget exhausted')
            elif kind == 'break':
                raise _Break()
            elif kind == 'block':
                self.run(node[1])

    def _async_copy(self, cp, stmt: str) -> None:
        """Move `bytes` worth of elements from one buffer to another: those
        the copy reads, and zeros for the rest -- the last `zfill` bytes of
        the pipeline's four-argument form, everything past the `read` bytes
        of a copy that fills (`tensorforge::copyAsyncFill`), which at zero
        reads nothing of its source, wherever it points."""
        dst = self.env.get(cp.group('dst'))
        src = self.env.get(cp.group('src'))
        if not isinstance(dst, Ptr) or not isinstance(src, Ptr):
            raise Abort(f'async copy between unknown buffers: {stmt!r}')
        width = self.widths.get(cp.group('dst'))
        nbytes = int(cp.group('bytes'))
        if width is None or nbytes % width:
            raise Abort(f'async copy of {nbytes} B into {cp.group("dst")!r}, '
                        f'whose element size is {width!r}: {stmt!r}')
        groups = cp.groupdict()
        if groups.get('read') is not None:
            read = int(self.ev(groups['read']))
        elif groups.get('zfill') is not None:
            read = nbytes - int(self.ev(groups['zfill']))
        else:
            read = nbytes
        if not 0 <= read <= nbytes or read % width:
            raise Abort(f'async copy reading {read} of {nbytes} B, in '
                        f'elements of {width}: {stmt!r}')
        di = int(self.ev(cp.group('di')))
        si = int(self.ev(cp.group('si')))
        # In flight until the wait that retires it, for `Races`; the values
        # land at once, which no race-free reader can tell apart.
        tracker = self.mem.tracker
        if tracker is not None:
            tracker.issuing = True
        try:
            for k in range(nbytes // width):
                dst[di + k] = src[si + k] if k < read // width else 0.0
        finally:
            if tracker is not None:
                tracker.issuing = False

    def _buffer_store(self, bs, stmt: str) -> None:
        """`tensorforge::bufferStore<T, N>(resource, index, value, keep)`:
        written where `keep` holds and all of it lies inside the resource,
        nowhere else (`_Resource`)."""
        args = _top_level(bs.group('args'), ',')
        size = _BUFFER_ELEMENT.get(bs.group('type'))
        if len(args) not in (3, 4) or size is None:
            raise Abort(f'buffer write not modeled: {stmt!r}')
        resource = self.ev(args[0])
        if not isinstance(resource, _Resource):
            raise Abort(f'a buffer write through {resource!r}: {stmt!r}')
        at, value = int(self.ev(args[1])), self.ev(args[2])
        keep = self.ev(args[3]) if len(args) == 4 else True
        count = int(bs.group('n') or 1)
        if not resource.inside(at, count, size, keep):
            return
        values = value if isinstance(value, Vec) else Vec([value])
        for k in range(count):
            resource.base[at + k] = values[k]

    def _buffer_lds(self, bl, stmt: str) -> None:
        """The LDS-DMA through a resource, its LDS form: the LDS-DMA's one
        destination a wave (`_lds_dma`), each lane's element read through
        the resource -- zero past it, and written all the same."""
        resource = self.env.get(bl.group('rsrc'))
        dst = self.env.get(bl.group('dst'))
        if not isinstance(resource, _Resource) or not isinstance(dst, Ptr):
            raise Abort(f'buffer LDS-DMA between unknown buffers: {stmt!r}')
        size = _BUFFER_ELEMENT.get(bl.group('type'))
        width = self.widths.get(bl.group('dst'))
        if size != 4 or (width is not None and width != size):
            raise Abort(f'buffer LDS-DMA of {bl.group("type")!r} into '
                        f'{bl.group("dst")!r}, whose element size is '
                        f'{width!r}: {stmt!r}')
        wave = self.lane // Races.DMA_WAVE
        first = self
        if self.lockstep is not None:
            first = min((it for it in self.lockstep.current
                         if it.lane // Races.DMA_WAVE == wave),
                        key=lambda it: it.lane, default=self)
        base = first.env.get(bl.group('dst'))
        if not isinstance(base, Ptr):
            raise Abort(f'buffer LDS-DMA into an unknown buffer: {stmt!r}')
        base = base + int(first.ev(bl.group('di')))
        value = _buffer_load(1, size, resource, self.ev(bl.group('index')))
        tracker = self.mem.tracker
        if tracker is not None:
            tracker.dma = wave
        try:
            base[self.lane % Races.DMA_WAVE] = value
        finally:
            if tracker is not None:
                tracker.dma = None

    def _lds_dma(self, dm, stmt: str) -> None:
        """AMD's LDS-DMA: what the instruction writes, not what its operands
        name.

        The destination is one address a wave -- the address of the first of
        its lanes executing the copy, which the compiler reads into M0 -- and
        lane t of the wave writes t elements past it.  A lane whose own address is
        another is written where the hardware puts it, and what it was meant
        to read there is not: the copy is checked by the values it leaves.
        In flight until a wait of the wave's retires it (`Races.vmcnt`).
        """
        src = self.env.get(dm.group('src'))
        dst = self.env.get(dm.group('dst'))
        if not isinstance(dst, Ptr) or not isinstance(src, Ptr):
            raise Abort(f'LDS-DMA between unknown buffers: {stmt!r}')
        width = self.widths.get(dm.group('dst'))
        nbytes = int(dm.group('bytes'))
        if width is None or nbytes != width:
            raise Abort(f'LDS-DMA of {nbytes} B into {dm.group("dst")!r}, '
                        f'whose element size is {width!r}: {stmt!r}')
        wave = self.lane // Races.DMA_WAVE
        first = self
        if self.lockstep is not None:
            first = min((it for it in self.lockstep.current
                         if it.lane // Races.DMA_WAVE == wave),
                        key=lambda it: it.lane, default=self)
        base = first.env.get(dm.group('dst'))
        if not isinstance(base, Ptr):
            raise Abort(f'LDS-DMA into an unknown buffer: {stmt!r}')
        base = base + int(first.ev(dm.group('di')))
        value = src[int(self.ev(dm.group('si')))]
        tracker = self.mem.tracker
        if tracker is not None:
            tracker.dma = wave
        try:
            base[self.lane % Races.DMA_WAVE] = value
        finally:
            if tracker is not None:
                tracker.dma = None

    def _async_lds(self, am, stmt: str) -> None:
        """gfx1250's copy into the LDS: every lane writes where its own
        address says -- one element, or a vector of them -- and the copy is
        in flight until a wait of the lane's wave on `asynccnt` retires it
        (`Races.asynccnt`).  A block barrier does not: its fence waits for
        no `asynccnt`."""
        src = self.env.get(am.group('src'))
        dst = self.env.get(am.group('dst'))
        if not isinstance(dst, Ptr) or not isinstance(src, Ptr):
            raise Abort(f'copy into the LDS between unknown buffers: {stmt!r}')
        width = self.widths.get(am.group('dst'))
        nbytes = int(am.group('bits')) // 8
        if width is None or nbytes % width:
            raise Abort(f'copy of {nbytes} B into {am.group("dst")!r}, whose '
                        f'element size is {width!r}: {stmt!r}')
        base = dst + int(self.ev(am.group('di')))
        start = int(self.ev(am.group('si')))
        values = [src[start + k] for k in range(nbytes // width)]
        tracker = self.mem.tracker
        if tracker is not None:
            tracker.dma = ('async', self.lane // Races.ASYNC_WAVE)
        try:
            for k, value in enumerate(values):
                base[k] = value
        finally:
            if tracker is not None:
                tracker.dma = None

    def _tensor_desc(self, td, stmt: str) -> None:
        """A tensor descriptor: the tile it names, from the source's element
        at `si`, its rows `pitch` apart, to the destination's at `di`."""
        src = self.env.get(td.group('src'))
        dst = self.env.get(td.group('dst'))
        if not isinstance(dst, Ptr) or not isinstance(src, Ptr):
            raise Abort(f'tensor descriptor between unknown buffers: {stmt!r}')
        width = _TENSOR_ELEMENT.get(td.group('type'))
        known = self.widths.get(td.group('dst'))
        if width is None or (known is not None and known != width):
            raise Abort(f'tensor descriptor of {td.group("type")!r} into '
                        f'{td.group("dst")!r}, whose element size is '
                        f'{known!r}: {stmt!r}')
        self.env[td.group('name')] = _TensorTile(
            dst + int(self.ev(td.group('di'))),
            src + int(self.ev(td.group('si'))),
            int(self.ev(td.group('rows'))), int(self.ev(td.group('cols'))),
            int(self.ev(td.group('pitch'))), width)

    def _tensor_load(self, tl, stmt: str) -> None:
        """gfx1250's tensor copy: one instruction of the wave that reaches
        it, so the copy is made once -- by the first of the wave's lanes
        executing it -- row after row into the LDS, and in flight until a
        wait of that wave's on `tensorcnt` retires it (`Races.tensorcnt`).
        A block barrier does not: its fence waits for no `tensorcnt`."""
        tile = self.env.get(tl.group('desc'))
        if not isinstance(tile, _TensorTile):
            raise Abort(f'tensor copy through an unknown descriptor: {stmt!r}')
        wave = self.lane // Races.TENSOR_WAVE
        if self.lockstep is not None:
            first = min((it.lane for it in self.lockstep.current
                         if it.lane // Races.TENSOR_WAVE == wave),
                        default=self.lane)
            if first != self.lane:
                return
        values = [tile.src[r * tile.pitch + c]
                  for r in range(tile.rows) for c in range(tile.cols)]
        tracker = self.mem.tracker
        if tracker is not None:
            tracker.dma = ('tensor', wave)
        try:
            for k, value in enumerate(values):
                tile.dst[k] = value
        finally:
            if tracker is not None:
                tracker.dma = None

    def _box(self, tmap: 'TensorMapArg', coords: List[int], start: int,
             width: int, stmt: str):
        """The elements of a box at `coords` (outermost first) whose shared
        bytes start at `start`: for each, where it is in the tensor -- None
        past an end -- and its slot in shared memory, the arena counted in
        elements of `width` bytes (`_tma_load`)."""
        rank = len(tmap.box)
        if len(coords) != rank:
            raise Abort(f'{len(coords)} coordinates into a map of {rank} '
                        f'dimensions: {stmt!r}')
        if (coords[-1] * tmap.size) % 16:
            raise Abort(f'a box starting {coords[-1] * tmap.size} bytes into '
                        f'a row, off 16: an illegal instruction: {stmt!r}')
        if start % 128:
            raise Abort(f'a box starting {start} bytes into shared memory, '
                        f'off 128: {stmt!r}')
        bits = {0: 0, 32: 1, 64: 2, 128: 3}[tmap.swizzle]
        pitch = tmap.swizzle or tmap.box[-1] * width
        volume = 1
        for n in tmap.box:
            volume *= n
        for k in range(volume):
            at, rest = [], k
            for n in reversed(tmap.box):
                at.append(rest % n)
                rest //= n
            at.reverse()
            where = [c + i for c, i in zip(coords, at)]
            inside = all(0 <= w < n for w, n in zip(where, tmap.dims))
            row = k // tmap.box[-1]
            address = start + row * pitch + at[-1] * width
            address ^= ((address >> 7) & ((1 << bits) - 1)) << 4
            yield (sum(w * t for w, t in zip(where, tmap.strides))
                   if inside else None), address // width

    def _tma_reduce(self, tr, stmt: str) -> None:
        """A box of shared memory added into the tensor of a map (PTX ISA,
        "cp.reduce.async.bulk.tensor", `.add`), by the lane that reaches it
        -- the one its guard picks -- in the layout a copy through the map
        leaves (`_box`), what lies past the tensor's ends left out.  Added
        here; the box is read until that lane's wait on its bulk groups
        retires the reduction (`Races.reduce_issue`)."""
        tmap = self.env.get(tr.group('map'))
        if not isinstance(tmap, TensorMapArg):
            raise Abort(f'a reduction through {tr.group("map")!r}, which is '
                        f'no tensor map the kernel is handed: {stmt!r}')
        src = self.env.get(tr.group('src'))
        if not isinstance(src, Ptr):
            raise Abort(f'tensor reduction of an unknown buffer: {stmt!r}')
        width = self.widths.get(tr.group('src'))
        if width != tmap.size:
            raise Abort(f'a box of {tmap.size}-byte elements in '
                        f'{tr.group("src")!r}, whose element size is '
                        f'{width!r}: {stmt!r}')
        coords = [int(self.ev(c)) for c in _top_level(tr.group('coords'),
                                                      ',')][::-1]
        start = (src.off + int(self.ev(tr.group('si')))) * width
        dst = Ptr(self.mem, tmap.buffer, tmap.offset)
        arena = Ptr(self.mem, src.base)
        slots = []
        for at, slot in list(self._box(tmap, coords, start, width, stmt)):
            value = arena[slot]
            slots.append(slot)
            if at is not None:
                dst[at] = dst[at] + value
        tracker = self.mem.tracker
        if tracker is not None:
            tracker.reduce_issue(tracker.lane, slots)

    def _tma_load(self, tl, stmt: str) -> None:
        """A box through a tensor map, by the leader of the cursor for the
        lanes it is for, as a bulk copy (PTX ISA, "cp.async.bulk.tensor",
        `.tile`): the box's element `(i...)` -- outermost first -- is the
        tensor's at the coordinates plus `i`, or the fill where that lies
        past an end; in shared memory the box's rows lie one after the
        other, a swizzled box's each the swizzle's span long, and each
        address's 16-byte chunk is XORed with its 128-byte line, one, two or
        three bits for a swizzle of 32, 64 or 128 bytes -- the address in
        shared memory, with the arena at 0.  So a box starts on 128 bytes,
        which is checked, and a box placed off the swizzle's period lands
        where the hardware puts it, which is not where a buffer of the
        swizzle reads it.  In flight until the wait for the cursor's phase
        (`Races.bulk_wait`); the bytes announced are the whole box's, or
        the phase would never complete; and the box starts on 16 bytes
        along its rows, which the hardware otherwise refuses."""
        cursor = self.env.get(tl.group('cursor'))
        if not isinstance(cursor, _Cursor):
            raise Abort(f'tensor map copy through an unarmed cursor: '
                        f'{stmt!r}')
        if not cursor.leader:
            return
        tmap = self.env.get(tl.group('map'))
        if not isinstance(tmap, TensorMapArg):
            raise Abort(f'a copy through {tl.group("map")!r}, which is no '
                        f'tensor map the kernel is handed: {stmt!r}')
        dst = self.env.get(tl.group('dst'))
        if not isinstance(dst, Ptr):
            raise Abort(f'tensor map copy into an unknown buffer: {stmt!r}')
        width = self.widths.get(tl.group('dst'))
        if width != tmap.size:
            raise Abort(f'a box of {tmap.size}-byte elements into '
                        f'{tl.group("dst")!r}, whose element size is '
                        f'{width!r}: {stmt!r}')
        coords = [int(self.ev(c)) for c in _top_level(tl.group('coords'),
                                                      ',')][::-1]
        volume = 1
        for n in tmap.box:
            volume *= n
        if int(tl.group('bytes')) != volume * tmap.size:
            raise Abort(f'{tl.group("bytes")} bytes announced for a box of '
                        f'{volume * tmap.size}: the phase would not '
                        f'complete: {stmt!r}')
        start = (dst.off + int(self.ev(tl.group('di')))) * width
        placed = list(self._box(tmap, coords, start, width, stmt))
        src = Ptr(self.mem, tmap.buffer, tmap.offset)
        arena = Ptr(self.mem, dst.base)
        fill = float('nan') if tmap.nan else 0.0
        tracker = self.mem.tracker
        if tracker is not None:
            tracker.bulk = cursor.name
        try:
            for at, slot in placed:
                arena[slot] = fill if at is None else src[at]
        finally:
            if tracker is not None:
                tracker.bulk = None

    def _bulk_copy(self, bc, stmt: str) -> None:
        """The copy through a bulk cursor: the leader moves the bytes for the
        whole multiplication, every other lane only notes it."""
        cursor = self.env.get(bc.group('cursor'))
        if not isinstance(cursor, _Cursor):
            raise Abort(f'bulk copy through an unarmed cursor: {stmt!r}')
        if not cursor.leader:
            return
        dst = self.env.get(bc.group('dst'))
        src = self.env.get(bc.group('src'))
        if not isinstance(dst, Ptr) or not isinstance(src, Ptr):
            raise Abort(f'bulk copy between unknown buffers: {stmt!r}')
        width = self.widths.get(bc.group('dst'))
        nbytes = int(bc.group('bytes'))
        if width is None or nbytes % width or nbytes % 16:
            raise Abort(f'bulk copy of {nbytes} B into {bc.group("dst")!r}, '
                        f'whose element size is {width!r}: {stmt!r}')
        di = int(self.ev(bc.group('di')))
        si = int(self.ev(bc.group('si')))
        tracker = self.mem.tracker
        if tracker is not None:
            tracker.bulk = cursor.name
        try:
            for k in range(nbytes // width):
                dst[di + k] = src[si + k]
        finally:
            if tracker is not None:
                tracker.bulk = None

    def assign(self, stmt: str) -> None:
        sl = _STORE_L2.match(stmt)
        if sl:
            # A store under an L2 policy is the assignment it hints.
            lhs, rhs = _first_operand(sl.group('rest'))
            self.assign(f'{lhs} = {rhs}')
            return
        mma = _mma_statement(stmt)
        if mma is not None:
            self._mma(mma, stmt)
            return
        wg = _wgmma_statement(stmt)
        if wg is not None:
            self._wgmma(wg, stmt)
            return
        if re.match(r'^asm\b', stmt) and 'tcgen05.' in stmt:
            self._tcgen05_asm(stmt)
            return
        ta = _TMEM_ALLOC.match(stmt)
        if ta:
            # The columns from lane 0 and column 0 on: an address is the
            # lane in its upper half and the column in its lower.
            slot = self.ev(ta.group('slot'))
            if isinstance(slot, Ptr):
                slot[0] = 0
            self.env[ta.group('name')] = 0
            self.mem.tmem_columns = int(ta.group('cols'))
            return
        td = _TMEM_DEALLOC.match(stmt)
        if td:
            state = self._tc_state()
            if state['open'] or any(state['bound'].values()) \
                    or state['overlay']:
                raise Abort(f'tensor memory freed with products in flight: '
                            f'{stmt!r}')
            return
        if _TC_QUIET.match(stmt):
            return
        tc = _TC_COMMIT.match(stmt)
        if tc:
            # The products the issuing thread issued, onto the cursor's
            # mbarrier: they land at the wait for its phase (`bulk_wait`).
            if bool(self.ev(tc.group('issue'))):
                state = self._tc_state()
                state['bound'].setdefault(tc.group('cursor'), []).extend(
                    state['open'])
                state['open'] = []
            return
        md = _MATRIX_DESC.match(stmt)
        if md:
            start = self.ev(md.group('buf'))
            if not isinstance(start, Ptr):
                raise Abort(f'a matrix descriptor of no buffer: {stmt!r}')
            self.env[md.group('name')] = _MatrixDesc(
                start + int(self.ev(md.group('idx'))),
                int(md.group('layout'), 16))
            return
        if _WGMMA_FENCE.match(stmt) or _PROXY_FENCE.match(stmt) or \
                _OPERAND_FENCE.match(stmt):
            # Orderings the lanes here keep anyway: they run statement by
            # statement, and what a product reads is read where it is
            # issued -- and checked unchanged where it is retired.
            return
        if _WGMMA_COMMIT.match(stmt):
            self.wg_groups.append(self.wg_open)
            self.wg_open = []
            return
        ww = _WGMMA_WAIT.match(stmt)
        if ww:
            self._wgmma_wait(int(ww.group('pending')), stmt)
            return
        lm = _LDMATRIX.match(stmt)
        if lm:
            self._ldmatrix(lm, stmt)
            return
        sm = _STMATRIX.match(stmt)
        if sm:
            self._stmatrix(sm, stmt)
            return
        sp = _SPLIT_TF32.match(stmt)
        if sp:
            # The upper half rounded, the lower the remainder as it is: what
            # the device keeps of it is the instruction's to decide (`_tf32`).
            value = self.ev(sp.group('value'))
            upper = _round_tf32(value)
            self.env[sp.group('upper')] = upper
            self.env[sp.group('lower')] = value - upper
            return
        lq = _LAUNCH_QUEUE.match(stmt)
        if lq:
            # The response slots in shared memory, which only the cursor reads.
            self.env[lq.group('name')] = lq.group('name')
            return
        lc = _LAUNCH_CURSOR.match(stmt)
        if lc:
            self.env[lc.group('name')] = _LaunchCursor(self.launches)
            return
        ls = _LAUNCH_START.match(stmt)
        if ls and isinstance(self.env.get(ls.group('name')), _LaunchCursor):
            self.env[ls.group('name')].start(self.ev(ls.group('queue')))
            return
        bm = _BULK_ARM.match(stmt)
        if bm:
            barrier = self.env.get(bm.group('bar'))
            if not isinstance(barrier, Ptr):
                raise Abort(f'bulk cursor on an unknown barrier: {stmt!r}')
            cursor = _Cursor(bm.group('name'), bool(self.ev(bm.group('leader'))),
                             barrier)
            if cursor.leader:
                barrier[0] = 1.0
            self.env[cursor.name] = cursor
            return
        tm = _TMA_LOAD.match(stmt)
        if tm:
            self._tma_load(tm, stmt)
            return
        tr = _TMA_REDUCE.match(stmt)
        if tr:
            self._tma_reduce(tr, stmt)
            return
        bw = _BULK_WAIT_READ.match(stmt)
        if bw:
            tracker = self.mem.tracker
            if tracker is not None:
                tracker.reduce_wait(tracker.lane, int(bw.group('pending')))
            return
        bc = _BULK_COPY.match(stmt)
        if bc:
            self._bulk_copy(bc, stmt)
            return
        bw = _BULK_WAIT.match(stmt)
        if bw:
            cursor = self.env.get(bw.group('cursor'))
            if not isinstance(cursor, _Cursor):
                raise Abort(f'bulk wait on an unarmed cursor: {stmt!r}')
            self._tc_land(bw.group('cursor'))
            cursor.barrier[0]
            tracker = self.mem.tracker
            if tracker is not None:
                tracker.bulk_wait(tracker.lane, cursor.name)
            return
        td = _TENSOR_DESC.match(stmt)
        if td:
            self._tensor_desc(td, stmt)
            return
        tl = _TENSOR_LOAD.match(stmt)
        if tl:
            self._tensor_load(tl, stmt)
            return
        if re.search(r'tensor_load_to_lds|tensorforge::tensor(?:Load|'
                     r'Descriptor)\b', stmt):
            raise Abort(f'tensor copy not modeled: {stmt!r}')
        br = _BUFFER_RESOURCE.match(stmt)
        if br:
            base = self.ev(br.group('base'))
            if not isinstance(base, Ptr):
                raise Abort(f'a buffer resource over no buffer: {stmt!r}')
            self.env[br.group('name')] = _Resource(
                base, int(self.ev(br.group('bytes'))))
            return
        bs = _BUFFER_STORE.match(stmt)
        if bs:
            self._buffer_store(bs, stmt)
            return
        bl = _BUFFER_LDS.match(stmt)
        if bl:
            self._buffer_lds(bl, stmt)
            return
        if re.match(r'^(?:const\s+)?tensorforge::(?:BufferResource|buffer'
                    r'(?:Store|LoadLds))\b', stmt):
            raise Abort(f'buffer access not modeled: {stmt!r}')
        # A global window bound through a memory-space pointer,
        # `tensorforge::SpacePtrRestrict<double, tensorforge::GlobalMemspace>
        # const glb_m0 = ...`, is a pointer to its element type here: the
        # comma inside the template arguments would keep `_DECL` from matching
        # it.
        stmt = re.sub(r'^tensorforge::SpacePtr\w*\s*<\s*((?:const\s+)?[\w:]+)\s*,[^>]*>'
                      r'\s*(?:const\s+)?(\w+\s*=)', r'\1 * const \2', stmt)
        m = _DECL.match(stmt)
        am = re.match(r'^(?:const\s+)?auto\s*&\s*(\w+)\s*=\s*(.+)$', stmt)
        if am:
            self.env[am.group(1)] = self.ev(am.group(2))   # alias: same object
            return
        vm = _VEC_STORE.match(stmt)
        if vm:
            # `*(SomeVecType*)&name[expr] = value;`  The components go to
            # consecutive slots, which is the whole content of a wide store --
            # and the reason this has to be modeled rather than skipped: a
            # cyclic reader of a blocked image produces plausible code and
            # wrong numbers, and nothing else in the harness looks at numbers.
            _ty, name, idx, rhs = vm.groups()
            base = self.env.get(name)
            if base is None:
                raise Abort(f'store into unknown {name!r}')
            val = self.ev(rhs)
            start = int(self.ev(idx))
            if not isinstance(val, Vec):
                val = Vec([val])
            for k, comp in enumerate(val):
                base[start + k] = comp
            return
        if '::' in stmt and _VEC_ASSIGN.match(stmt):
            # A widened store in a form `_VEC_STORE` does not parse.  Refused
            # rather than skipped: a skipped store leaves its destination at
            # zero, and a kernel evaluated that way still agrees with itself.
            raise Abort(f'vector assignment not modeled: {stmt!r}')
        dm = _LDS_DMA.match(stmt)
        if dm:
            self._lds_dma(dm, stmt)
            return
        if 'global_load_lds' in stmt:
            raise Abort(f'LDS-DMA not modeled: {stmt!r}')
        am = _ASYNC_LDS.match(stmt)
        if am:
            self._async_lds(am, stmt)
            return
        if 'global_load_async_to_lds' in stmt:
            raise Abort(f'copy into the LDS not modeled: {stmt!r}')
        cp = _ASYNC_COPY.match(stmt) or _ASYNC_FILL.match(stmt)
        if cp:
            # The transfer half of the pipeline primitives.  It moves data, so
            # it is carried out here rather than left to the catch-all below
            # with the commit and the wait, which do not: `Slot` fills an
            # unwritten address from its seed, so a skipped transfer would
            # still produce a full destination of plausible numbers, none of
            # which had been through it.
            #
            # Completion is not modeled and does not need to be.  The wait is
            # the only point at which the copy is guaranteed done, this
            # interpreter is sequential, and finishing early is the one
            # rounding of `cp.async` that no reader here can observe.
            self._async_copy(cp, stmt)
            return
        if 'memcpy_async' in stmt or 'copyAsyncFill' in stmt:
            # Another spelling.  Refused rather than swallowed: a transfer
            # that quietly does nothing produces plausible numbers, as
            # above.
            raise Abort(f'async copy not modeled: {stmt!r}')
        if (('pipeline' in stmt or '::' in stmt)
                and not _VEC_DECL.match(stmt) and 'tensorforge::' not in stmt
                and 'std::' not in stmt
                # `::` is also what separates an inline asm's empty output
                # list from its inputs, and `red.global.add` is written that
                # way.  Swallowed here, the accumulation it performs would
                # simply not happen on the host: the oracle reported no
                # write to the destination at all rather than a wrong one.
                and not _ATOMIC_ADD_ASM.match(stmt)):
            # The commit and the wait of the pipeline primitives, the grid
            # barrier: statements with no effect on the values compared here.
            # A declaration whose *type* is namespaced is not one of those --
            # swallowing `tensorforge::VectorT<float,2> v = ...` would leave
            # the name unbound -- and neither is a call into `tensorforge::`
            # or the standard library.
            return
        if m and m.group('name') not in ('return',):
            name = m.group('name')
            if m.group('ptr') or m.group('arr'):
                self.widths[name] = _scalar_bytes(stmt)
            if m.group('arr'):
                self.env[name] = Ptr(self.mem, f'{name}#{id(self.env)}')
                for i in range(int(m.group('dim'))):
                    self.env[name][i] = 0.0
                return
            init = m.group('init')
            self.env[name] = self.ev(init) if init else 0.0
            return
        cm = re.match(r'^(\w+)\s*(\[.*\])?\s*\+=(.*)$', stmt)
        if cm:
            # `target += value`, an accumulator updated in place
            # (`IRBuilder.accumulate`).
            name, at, rhs = cm.groups()
            val = self.ev(rhs.strip())
            if at:
                base, idx = self.env[name], self.ev(at.strip()[1:-1])
                base[idx] = base[idx] + val
            elif name in self.env:
                self.env[name] = self.env[name] + val
            else:
                raise Abort(f'{stmt!r}: {name} is not defined')
            return
        if '=' in stmt and not re.search(r'[=!<>]=', stmt.split('=')[0] + '='):
            lhs, rhs = stmt.split('=', 1)
            lhs, rhs = lhs.strip(), rhs.strip()
            val = self.ev(rhs)
            am = re.match(r'^(\w+)\s*\[(.*)\]$', lhs)
            if am:
                self.env[am.group(1)][self.ev(am.group(2))] = val
            elif re.match(r'^\w+$', lhs):
                self.env[lhs] = val
            else:
                raise Abort(f'unsupported lhs {lhs!r}')
            return
        if re.match(r'^(__syncthreads|__syncwarp|__threadfence)\s*\(', stmt):
            return
        if re.match(r'^__builtin_amdgcn_sched_barrier\s*\(', stmt):
            return                      # a fence for the compiler's schedule
        if _NO_VALUE.match(stmt):
            # The engines' orderings and hints (`pir.engines`): a priority, a
            # scheduling group, a memory fence, a counter wait, a gfx1250
            # object's setup and rendezvous, a cache hint for the next
            # element's data (`pir.prefetch`) -- nothing a value changes by.
            # The rendezvous themselves are the race tracker's.
            return
        if stmt.startswith('using namespace '):
            return
        am = _ATOMIC_ADD.match(stmt) or _ATOMIC_ADD_ASM.match(stmt)
        if am:
            # `dest[i] += value`, and every lane that reaches this statement
            # performs one.  That is the whole content of an atomic here: the
            # interpreter runs the lanes in lockstep over one shared `Slot`,
            # so summing the arrivals *is* the hardware's guarantee, and the
            # ordering an atomic also promises does not change a sum.
            #
            # Modeled rather than skipped, and rather than treated as a
            # store.  Skipping would leave the accumulation out of the
            # comparison entirely, and the atomic path without numerical
            # coverage on the host; treating it as `=` would agree with the
            # correct answer whenever exactly one lane arrives, which is
            # precisely the case that is never in doubt.
            base, index, value = am.group('base'), am.group('idx'), am.group('val')
            ptr = self.env[base]
            slot = self.ev(index)
            ptr[slot] = (ptr[slot] or 0.0) + self.ev(value)
            return
        if stmt.startswith('extern ') or stmt.startswith('__shared__'):
            return                      # the shared arena, modeled as a base
        if re.match(r'^(?:const\s+)?auto\s*\*?\s*\w+', stmt) and '=' not in stmt:
            return
        raise Abort(f'unsupported statement {stmt!r}')


#: Statements of the engines with no effect on a value (`Interp.assign`).
_NO_VALUE = re.compile(
    r'^(?:__builtin_amdgcn_(?:s_setprio|sched_group_barrier|fence|s_waitcnt'
    r'|s_wait_asynccnt|s_wait_tensorcnt)|__threadfence_block'
    r'|__atomic_signal_fence'
    r'|tensorforge::(?:sync|arrive|join)NamedBarrier(?:<[^>]*>)?'
    r'|tensorforge::prefetch(?:Runs)?(?:L1|L2|Hinted)(?:<[^>]*>)?)\s*\(')


def _parenthesized(text: str) -> bool:
    """Whether every parenthesis of `text` outside its string literals is
    closed."""
    depth, quoted = 0, False
    for ch in text:
        if ch == '"':
            quoted = not quoted
        elif not quoted:
            depth += (ch == '(') - (ch == ')')
    return depth <= 0


def parse(src: str) -> List:
    """Line-based parse into nested statement nodes."""
    lines = []
    #: An inline asm spread over several lines, `mma.sync`'s, as one
    #: statement: its template and operand lists are one call.
    held = None
    for raw in src.splitlines():
        t = raw.split('//')[0].strip()
        if held is not None:
            held = f'{held} {t}'
            if not _parenthesized(held):
                continue
            t, held = held, None
        elif re.match(r'^asm\s*(?:volatile\s*)?\(', t) and not _parenthesized(t):
            held = t
            continue
        if not t or t.startswith('#'):
            continue
        while t.endswith('{') and len(t) > 1:     # `for (...) {` on one line
            lines.append(t[:-1].strip())
            t = '{'
        lines.append(t)

    pos = 0

    def block() -> List:
        nonlocal pos
        out = []
        while pos < len(lines):
            t = lines[pos]
            if t == '}':
                pos += 1
                return out
            pos += 1
            if t == '{':
                out.append(('block', block()))
                continue
            m = _IF.match(t)
            if m:
                if pos < len(lines) and lines[pos] == '{':
                    pos += 1
                    body = block()
                else:
                    body = [('expr', lines[pos].rstrip(';'))]
                    pos += 1
                els = None
                if pos < len(lines) and lines[pos].startswith('else'):
                    pos += 1
                    if pos < len(lines) and lines[pos] == '{':
                        pos += 1
                        els = block()
                out.append(('if', m.group('c'), body, els))
                continue
            m = _FOR.match(t)
            if m:
                if pos < len(lines) and lines[pos] == '{':
                    pos += 1
                    body = block()
                else:
                    body = [('expr', lines[pos].rstrip(';'))]
                    pos += 1
                out.append(('for', m.group('v'), m.group('a'), m.group('cmp'),
                            m.group('b'), m.group('step'), body))
                continue
            m = _WHILE.match(t)
            if m:
                if pos < len(lines) and lines[pos] == '{':
                    pos += 1
                    body = block()
                else:
                    body = [('expr', lines[pos].rstrip(';'))]
                    pos += 1
                out.append(('while', m.group('c'), body))
                continue
            if t.rstrip(';').strip() == 'break':
                out.append(('break',))
                continue
            out.append(('expr', t.rstrip(';')))
        return out

    return block()


def _kernel_start(src: str) -> int:
    """Where the kernel's body opens: the first brace after `__global__`.

    Not the file's first brace -- the generator writes what the kernel
    reads by name ahead of it, a parameter table's struct or a table of
    numbers at namespace scope, and either has braces of its own."""
    at = src.find('__global__')
    return src.index('{', max(at, 0))


#: A table of numbers at namespace scope, as `Lexic.module_constant` spells
#: one on CUDA and HIP or anywhere else.
_MODULE_ARRAY = re.compile(
    r'(?:__device__\s+)?(?:const|constexpr)\s+[\w:]+\s+(?P<name>\w+)\s*'
    r'\[\s*\d+\s*\]\s*=\s*\{(?P<values>[^}]*)\}\s*;')


def _number(text: str):
    """A literal of such a table: an integer, or a float with its suffix."""
    text = text.strip()
    try:
        return int(text)
    except ValueError:
        return float(text.rstrip('fF'))


def _base_env(src: str, mem: Slot, tid: int, lanes: int = 256,
              mults: int = 1, mult: int = 0, block: int = 0,
              grid: int = 1) -> Dict[str, object]:
    """The names a kernel body starts with, for one lane.

    `blockDim` follows the launcher.  `blockDim.x` appears in no generated
    kernel today, but `blockDim.y` is the stride of the batch loop in most of
    them, and a fixed 1 there is a guess that happens to agree only while
    `numElements` is 1.  `mult` is the multiplication of the block the lane
    belongs to, `threadIdx.y`; `block` the block of a grid of `grid`.
    """
    env = {
        'threadIdx': type('T', (), {'x': tid, 'y': mult, 'z': 0})(),
        'blockIdx': type('B', (), {'x': block, 'y': 0, 'z': 0})(),
        'blockDim': type('D', (), {'x': lanes, 'y': mults, 'z': 1})(),
        'gridDim': type('G', (), {'x': grid, 'y': 1, 'z': 1})(),
    }
    # An operand declared a pointer to pointers is a table of them, one per
    # element.
    for name in re.findall(r'\*\s*(?:const\s*)?\*\s*(?:__restrict__\s*)?(m\d+)\b',
                           src):
        env.setdefault(name, PtrTable(mem, name))
    # An operand passed by value -- a scalar the whole batch shares -- is the
    # number its one slot holds.
    head = src[:_kernel_start(src)] if '{' in src else ''
    for name in re.findall(r'[(,]\s*(?:const\s+)?(?:float|double|__float128)'
                           r'\s+(m\d+)\s*(?=[,)])', head):
        env.setdefault(name, mem.read(name, 0))
    # Numbers the file defines ahead of the kernel, at namespace scope
    # (`Generator.module_table_definitions`): a list the lanes index.
    for m in _MODULE_ARRAY.finditer(src[:_kernel_start(src)]):
        env.setdefault(m.group('name'), [_number(v) for v in
                                         m.group('values').split(',')])
    for name in re.findall(r'\b(m\d+)\b', src):
        env.setdefault(name, Ptr(mem, name))
    for name in re.findall(r'\b(\w*_extraOffset)\b', src):
        env.setdefault(name, 0)
    # The address parameters bound at run time (`ElementAddressing`): a slot
    # of 0, and a stride that keeps the elements of every operand a test
    # declares apart, as the preset's width does.  `scalars` sets others.
    for name in re.findall(r'\b(m\d+_stride)\b', src):
        env.setdefault(name, _PRESET_SLOTS)
    for name in re.findall(r'\b(m\d+_slot)\b', src):
        env.setdefault(name, 0)
    if re.search(r'\belementIndices\b', src):
        env.setdefault('elementIndices', Indices())
    for name in re.findall(r'\b(numElements\d+)\b', src):
        env.setdefault(name, 1)
    for name in re.findall(r'\b(flags\d+)\b', src):
        env.setdefault(name, None)
    env['totalShrMemPtr'] = Ptr(mem, 'shr')
    return env


class Lockstep:
    """Drives a set of lanes through one body together.

    A cross-lane read is only answerable if the lanes have reached the same
    statement, so they are advanced statement by statement rather than lane by
    lane. Divergence is handled by partitioning: an `if` runs its arms on the
    lanes that take them, a `for` iterates while any lane is still in range,
    and a lane that is masked off simply does not execute the statements
    inside -- which is why a `readlane` of a value it never defined aborts
    rather than inventing one.

    `groups` are the lanes that read across each other, each a
    multiplication of the block; all of them together unless given.
    """

    def __init__(self, interps: List['Interp'],
                 groups: Optional[List[List['Interp']]] = None):
        self.interps = interps
        for group in groups if groups is not None else [interps]:
            for it in group:
                it.peers = group
        for it in interps:
            it.lockstep = self
        #: The lanes executing the statement at hand, and what of it was read
        #: as the wave's (`Interp._wave_uniform`).
        self.current: List['Interp'] = interps
        self.wave_cache: Dict = {}
        self.tracker: Optional[Races] = interps[0].mem.tracker if interps \
            else None

    def _tracked(self, stmt: str, lanes: List['Interp']) -> None:
        """One statement under `Races`: the synchronization it is, and which
        lane each access belongs to."""
        tracker = self.tracker
        tracker.tick(stmt)
        kind = tracker.statement_kind(stmt)
        if kind == 'arrive':
            _, ident, _ = tracker.reach(stmt)
            for it in lanes:
                tracker.arrive(int(it.ev(ident)), it.lane)
        if kind == 'barrier' and tracker.drains(stmt):
            # The fence ahead of the barrier: the copies of every wave that
            # reaches it land first, and the barrier orders them.
            waves = sorted({it.lane // Races.DMA_WAVE for it in lanes})
            if sum(tracker.vmcnt(w, 0) for w in waves):
                tracker.time += 1
        if kind == 'vmcnt':
            keep = tracker.vmcnt_of(stmt)
            for wave in sorted({it.lane // Races.DMA_WAVE for it in lanes}):
                tracker.vmcnt(wave, keep)
        if kind == 'asynccnt':
            keep = tracker.asynccnt_of(stmt)
            for wave in sorted({it.lane // Races.ASYNC_WAVE for it in lanes}):
                tracker.asynccnt(wave, keep)
        if kind == 'tensorcnt':
            keep = tracker.tensorcnt_of(stmt)
            for wave in sorted({it.lane // Races.TENSOR_WAVE for it in lanes}):
                tracker.tensorcnt(wave, keep)
        if kind == 'barrier':
            for present, waited, arrived in self._meetings(stmt, lanes):
                tracker.barrier([it.lane for it in present], arrived)
                missing = [it.lane for it in waited
                           if it not in present and it.lane not in arrived]
                if missing:
                    tracker.arrival(missing[0], len(present))
        for it in lanes:
            tracker.lane = it.lane
            if kind == 'commit':
                tracker.commit(it.lane)
            elif kind == 'wait':
                tracker.wait(it.lane, stmt)
            it.assign(stmt)

    def _meetings(self, stmt: str, lanes: List['Interp']):
        """The rendezvous one barrier statement is, as `(present, waited,
        arrived)`: the lanes executing it that meet one another there, every
        lane of the run that has to be among them (`Races.reach`), and of a
        named barrier the lanes that arrived at it ahead without waiting.

        A lane's warp is its thread index in the block over the warp size,
        which is where the run places it (`evaluate_wave`): the lanes of one
        multiplication, and the multiplications one after another.
        """
        reach = self.tracker.reach(stmt)
        warp = Races.WARP
        if reach[0] == 'block':
            return [(lanes, self.interps, {})]
        out = []
        if reach[0] == 'warp':
            mask_of = (lambda it: int(it.ev(reach[1])) & 0xffffffff) \
                if reach[1] else (lambda it: 0xffffffff)
            keys = {}
            for it in lanes:
                keys.setdefault((it.lane // warp, mask_of(it)), []).append(it)
            for (w, mask), present in keys.items():
                waited = [it for it in self.interps if it.lane // warp == w
                          and mask >> (it.lane % warp) & 1]
                out.append((present, waited, {}))
            return out
        _, ident, count = reach
        keys = {}
        for it in lanes:
            keys.setdefault(int(it.ev(ident)), []).append(it)
        for key, present in keys.items():
            waited = [it for it in self.interps if int(it.ev(ident)) == key]
            if count is not None:
                # The barrier counts warps as they arrive, a warp's worth of
                # threads each, so a count the lanes of the id do not fill
                # completes early -- ordering only the warps it counted -- or
                # never.
                warps = {it.lane // warp for it in waited}
                counted = int(present[0].ev(count))
                if counted != len(warps) * warp:
                    self.tracker.count(len(warps) * warp, counted)
            out.append((present, waited, self.tracker.arrivals(key)))
        return out

    def run(self, block: List, lanes: Optional[List['Interp']] = None) -> None:
        lanes = self.interps if lanes is None else lanes
        if not lanes:
            return
        for node in block:
            kind = node[0]
            self.current = lanes
            self.wave_cache = {}
            if kind == 'expr':
                if self.tracker is not None:
                    self._tracked(node[1], lanes)
                    continue
                for it in lanes:
                    it.assign(node[1])
            elif kind == 'if':
                taken = [it for it in lanes if it.ev(node[1])]
                other = [it for it in lanes if it not in taken]
                self.run(node[2], taken)
                if node[3] is not None:
                    self.run(node[3], other)
            elif kind == 'for':
                v, a, cmp_, b, step = node[1:6]
                for it in lanes:
                    it.env[v] = it.ev(a)
                inc = 1 if '++' in step or '+=' in step else -1
                while True:
                    self.current, self.wave_cache = lanes, {}
                    active = [it for it in lanes
                              if (it.env[v] < it.ev(b) if '<' in cmp_
                                  else it.env[v] > it.ev(b))]
                    if not active:
                        break
                    self.run(node[6], active)
                    self.current, self.wave_cache = active, {}
                    for it in active:
                        it.env[v] += (it.ev(step.split('=')[1]) *
                                      (1 if '+=' in step else -1)
                                      if ('+=' in step or '-=' in step) else inc)
                        it.budget -= 1
                        if it.budget < 0:
                            raise Abort('budget exhausted')
            elif kind == 'while':
                # Left by a `break` every lane takes at once: the loops that
                # break here -- the launch-control traversal -- leave on a
                # condition every thread of the block reads alike.
                while True:
                    self.current, self.wave_cache = lanes, {}
                    active = [it for it in lanes if it.ev(node[1])]
                    if not active:
                        break
                    if len(active) != len(lanes):
                        raise Abort('lanes disagree on whether to enter a '
                                    'while loop')
                    try:
                        self.run(node[2], active)
                    except _Break as left:
                        if len(left.lanes) != len(active):
                            raise Abort('a break some lanes of a loop take '
                                        'and others do not')
                        break
                    for it in active:
                        it.budget -= 1
                        if it.budget < 0:
                            raise Abort('budget exhausted')
            elif kind == 'break':
                raise _Break(lanes)
            elif kind == 'block':
                self.run(node[1], lanes)


#: `dim3 block(threads, mults, 1)` in the generated launcher.  The lane count
#: belongs to the kernel, not to the test that runs it, and this is where the
#: kernel states it.
_BLOCK_DIM = re.compile(r'\bblock\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*\d+\s*\)')


def launch_geometry(launcher: str) -> Tuple[int, int]:
    """``(lanes per multiplication, multiplications per block)``.

    Running more lanes than the launcher asks for is not a harmless surplus.
    A hop loop is only guarded where the extent does not divide evenly, so the
    invented lanes execute unguarded copies at their own offsets and read past
    the end of the operand -- which reads as a generator overrun and is not
    one.  It would stay invisible if the async copy were swallowed, because a
    lane that copies nothing reads nothing either.
    """
    m = _BLOCK_DIM.search(launcher)
    if not m:
        raise Abort('launcher states no block dimensions')
    return int(m.group(1)), int(m.group(2))


def evaluate_wave(src: str, lanes: int, seed: int = 0,
                  globals_only: bool = False,
                  preset: Optional[Dict[str, object]] = None,
                  mults: int = 1,
                  races: Optional[List[Race]] = None,
                  elements: int = 1, block: bool = False,
                  launches: Sequence[int] = (), grid: int = 1,
                  scalars: Optional[Dict[str, object]] = None) -> Dict:
    """Run one kernel body for `lanes` lanes together; return the memory.

    The difference from calling `evaluate` per lane is that the lanes share
    one memory *and* can see each other's registers, which is what a
    `readlane` needs. Everything a single-lane run could check, this checks
    too; what it adds is the cross-lane traffic -- the broadcast of an operand
    and the peeled tail of a widened lead dimension, neither of which a
    single-lane run can evaluate.

    `lanes` is the kernel's own, from `launch_geometry`, not a round number:
    see there for what a surplus lane does with the copies modeled.

    `races`, a list, collects the accesses to shared memory that no barrier
    orders (`Races`).  `elements` is the element count the kernel is handed:
    with one block of one multiplication the batch loop then visits each of
    them in turn, which is what carries a write across its back edge.
    `launches` are the CTA ids a launch-control queue hands the block after
    its own, in order (`_LaunchCursor`).

    `grid` runs every block of a grid that many blocks wide, one after the
    other on the one memory: a block's traversal of the batch is then its
    share of it, as the launch hands it out.  Blocks take no part in each
    other's work, so in turn is as good an order as any.

    `scalars` are kernel parameters passed by value, by name -- an operand's
    stride or slot bound at run time (`m3_stride`), its extra offset.  A
    preset `elementIndices` is the launch's index list, read as memory; the
    identity without one (`Indices`).

    The lanes run are those of the block's first multiplication, unless
    `block` asks for all `mults` of them.  What the block does together
    before its batch loop -- copying an operand all elements read into
    shared memory, each thread its share -- is only whole with every thread
    of the block taking part: run with the first multiplication alone, the
    other shares read as uninitialized shared memory.  The lanes of each
    multiplication read across their own multiplication only.
    """
    body = src[_kernel_start(src):]
    mem = Slot(seed)
    if preset:
        _fill(mem, preset)
    if races is not None:
        mem.tracker = Races()
    for cta in range(grid):
        groups = []
        for mult in range(mults if block else 1):
            group = []
            for tid in range(lanes):
                env = _base_env(src, mem, tid, lanes=lanes, mults=mults,
                                mult=mult, block=cta, grid=grid)
                for name in re.findall(r'\b(numElements\d+)\b', src):
                    env[name] = elements
                # A mask the preset gives is one the kernel is handed; without
                # one, the pointer is null and every element counts.
                for name in preset or ():
                    if (re.fullmatch(r'flags\d+', name)
                            or name == 'elementIndices'):
                        env[name] = Ptr(mem, name)
                env.update(scalars or {})
                interp = Interp(mem, env)
                interp.lane = tid + mult * lanes
                interp.launches = tuple(launches)
                group.append(interp)
            groups.append(group)
        interps = [it for group in groups for it in group]
        Lockstep(interps, groups).run(parse(body))
    if races is not None:
        races.extend(mem.tracker.found)
        mem.tracker = None
    if globals_only:
        return {k: v for k, v in mem.data.items() if k[0].startswith('m')}
    return {(k[0].split('#')[0], k[1]): v for k, v in mem.data.items()}


def evaluate(src: str, tid: int = 0, seed: int = 0,
             globals_only: bool = False,
             preset: Optional[Dict[str, object]] = None,
             scalars: Optional[Dict[str, object]] = None) -> Dict:
    """Run one kernel body for one thread; return the resulting memory.

    ``preset`` maps a global array name to a value every one of its slots is
    filled with before the run, or to a sequence of values for its first
    slots, one each.  Unwritten slots otherwise take a
    pseudo-random value derived from their identity, which is what makes two
    runs comparable --- but it also means an input cannot be *chosen*.  Being
    able to zero one operand turns this into a reference-free sensitivity
    test: if the result does not move when a term's operand is zeroed, the
    kernel is not using that term.
    """
    body = src[_kernel_start(src):]
    mem = Slot(seed)
    if preset:
        _fill(mem, preset)
    env = _base_env(src, mem, tid)
    if preset and 'elementIndices' in preset:
        env['elementIndices'] = Ptr(mem, 'elementIndices')
    env.update(scalars or {})
    interp = Interp(mem, env)
    interp.run(parse(body))
    if globals_only:
        return {k: v for k, v in mem.data.items() if k[0].startswith('m')}
    # scratch arrays are keyed by env identity, which no other run shares;
    # key them by name and index, so that two runs compare
    return {(k[0].split('#')[0], k[1]): v for k, v in mem.data.items()}
