# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
# SPDX-FileContributor: David Schneller

"""Pseudo-IR: lowering to the existing ``backend.writer.Writer``.

This is the *only* file in ``pir`` that knows C++ exists.  Everything
vendor-specific goes through the context's lexic, so a second backend is a
handful of overrides here rather than a fork of the builders.

Declaration placement: the IR is SSA, C++ is not.  Values that a region yields
out are declared in the *parent* scope and assigned at the ``yield``; every
other value is declared at its definition.  That keeps the IR pure while the
generated code still looks like something a human would write.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from tensorforge.common.basic_types import Addressing, Datatype

from tensorforge.common.basic_types import GeneralLexicon
from tensorforge.common.operation import Operation
from .core import (Access, BufferType, Effect, IRError, MemSpace, Op, Operand,
                   Qual,
                   Region, ScalarType, Stmt, TokenType, Uniformity, Value, def_use,
                   walk, walk_stmts)

_ATOM = __import__('re').compile(r'^(?:[A-Za-z_][A-Za-z0-9_.:]*|\d[\w.]*)$')


# A predicate becomes a select only where suppressing the statement is not the
# point.  Reads -- synchronous or asynchronous -- may be evaluated under a
# ternary; anything that writes, is atomic, synchronizes or is opaque has to
# keep a real branch, or the effect would happen when it must not.  This used
# to be a test on the *shape* of the statement (does it have a target?), which
# would silently fold a value-returning atomic into a ternary.
_MUST_BRANCH = Effect.WRITE | Effect.ATOMIC | Effect.BARRIER | Effect.UNKNOWN


def _folds_predicate(s: Stmt) -> bool:
    if s.effect & _MUST_BRANCH or s.regions:
        return False
    return bool(s.target) or s.op in Op.DECLARING


def _unwrap(expr: str) -> str:
    """Drop one redundant paren level: `Writer.If` adds its own."""
    if not (expr.startswith('(') and expr.endswith(')')):
        return expr
    depth = 0
    for i, c in enumerate(expr):
        depth += (c == '(') - (c == ')')
        if depth == 0 and i < len(expr) - 1:
            return expr                 # e.g. "(a) && (b)" -- not a wrapper
    return expr[1:-1]


def _atomic(expr: str) -> bool:
    """An identifier or literal needs no parentheses when inlined."""
    return bool(_ATOM.match(expr))


def _sm_at_least(model: str, minimum: int) -> bool:
    digits = ''.join(c for c in str(model)[3:] if c.isdigit())
    return str(model).startswith('sm_') and bool(digits) and int(digits) >= minimum


# Which architectures actually have the asynchronous global -> shared path.
# The lexic knows how the call *looks*; this table knows whether it *exists*.
_ASYNC_ARCH = {
    'nvidia': lambda m: _sm_at_least(m, 80),                    # cp.async
    'amd': lambda m: str(m) in ('gfx90a', 'gfx940', 'gfx941',   # global_load_lds
                                'gfx942', 'gfx950'),
}


# generic pure ops -> infix C++ operators
_INFIX = {
    'add': '+', 'sub': '-', 'mul': '*', 'div': '/',
    'rem': '%', 'bitand': '&', 'bitor': '|', 'bitxor': '^', 'shl': '<<', 'shr': '>>',
    'lt': '<', 'le': '<=', 'gt': '>', 'ge': '>=', 'eq': '==', 'ne': '!=',
    'and': '&&', 'or': '||',
}

# Ops with no infix form that the lexic already spells correctly.  Without
# this they fell through to the generic `f'{op}({args})'`, i.e. unqualified
# `min(a, b)` -- which happens to resolve in CUDA and HIP device code through
# the vendor headers' global-namespace overloads, and so worked by accident
# while silently depending on which headers a translation unit had pulled in.
# `get_operation` gives `fminf`/`fmin` by dtype, which is what the elementwise
# path has always emitted for the same operator.
_LEXIC_BINOP = {'min': Operation.MIN, 'max': Operation.MAX}


#: The operations `Context.record_work` counts, where the result is a floating
#: point value: the arithmetic a lane geometry changes the amount of.  Index
#: arithmetic is deliberately out -- it is addressing, it scales with the
#: geometry for its own reasons, and counting it would drown the term it is
#: being weighed against.
_WORK_OPS = frozenset({'add', 'sub', 'mul', 'div', 'fma', 'neg'})


#: The floating-point members of `Datatype`, which is what makes an operation
#: arithmetic here rather than addressing.
_WORK_TYPES = frozenset({Datatype.F16, Datatype.BF16, Datatype.F32,
                         Datatype.F64, Datatype.F128})


def _counts_as_work(op: str, value) -> bool:
    if op not in _WORK_OPS:
        return False
    return getattr(getattr(value, 'type', None), 'base', None) in _WORK_TYPES


#: Statements that put nothing into the instruction stream: a constant is an
#: immediate of whatever reads it, a declaration or an extraction names a
#: register, a yield is the variable the loop already shares.
_NO_CODE = frozenset({Op.CONST, Op.YIELD, Op.DECLARE, Op.ALLOC, Op.EXTRACT,
                      Op.SPLIT, Op.PACK})

#: A rolled loop's own instructions per copy of its body: the counter, the
#: test and the branch.
_LOOP_OVERHEAD = 3

#: The memory spaces a load or store occupies a pipe for, by the name
#: `analysis.pipeline` counts them under.  A register access is a name.
_MIX_SPACES = {MemSpace.GLOBAL: 'global', MemSpace.PARAM: 'global',
               MemSpace.CONSTANT: 'constant', MemSpace.SHARED: 'shared',
               MemSpace.SCRATCH: 'local'}

#: Calls that go to the transcendental unit (MUFU, `v_exp`, ...), as a whole
#: name: `sin` is not `single`.
_SFU_CALL = re.compile(r'(?:^|[^a-z0-9])_*(exp2?|log2?|sqrt|rsqrt|rcp|sin|cos'
                       r'|tan|pow|erf|cbrt|tanh|sinh|cosh)f?(?:$|[^a-z0-9])',
                       re.IGNORECASE)

#: Matrix instructions reached as builtins (`__builtin_amdgcn_mfma_*`, WMMA);
#: NVIDIA's arrive as inline assembly.
_MATRIX_CALL = re.compile(r'(mfma|wmma|smfmac)', re.IGNORECASE)

#: An FMA whose operand comes from another lane through a DPP modifier: one
#: VALU instruction, the move is free.  So it is arithmetic, not a move.
_FMA_CALL = re.compile(r'(fmac?dpp|fmadpp|fma_dpp)', re.IGNORECASE)

#: Calls that move a value between lanes (`__shfl_sync`, a DPP move,
#: `readlane`, a sub-group broadcast, a register transpose).
_CROSSLANE_CALL = re.compile(r'(shfl|dpp|readlane|readfirstlane|permute'
                             r'|swizzle|broadcast|select_from_group'
                             r'|group_broadcast|transpose)', re.IGNORECASE)

#: Calls that constrain the compiler and lay nothing down (`pin` is an empty
#: `asm volatile` holding a value in a register).
_NO_CODE_CALL = re.compile(r'(^|::)(pin|keep|opaque)$')

#: Raw text that stores into memory: an atomic, or an assignment to a global
#: window (`glb_m1[...] = ...`) or a shared one.
_ATOMIC_TEXT = re.compile(r'atomic|fetch_add|atomicAdd', re.IGNORECASE)
_GLOBAL_STORE_TEXT = re.compile(r'^\s*glb_\w+\s*\[[^\]]*\]\s*=[^=]')
_SHARED_STORE_TEXT = re.compile(r'^\s*s\d+\s*\[[^\]]*\]\s*=[^=]')


def _text_category(text: str, fallback: str) -> str:
    """The category of a raw expression or statement, read off its text --
    the one thing the IR cannot see into.  The math library's calls go to the
    transcendental unit, a reduction or broadcast moves values between lanes,
    and an atomic or an assignment to a window is a store."""
    if _ATOMIC_TEXT.search(text) or _GLOBAL_STORE_TEXT.match(text):
        return 'global.store'
    if _SHARED_STORE_TEXT.match(text):
        return 'shared.store'
    if _CROSSLANE_CALL.search(text) or 'reduction<' in text:
        return 'xlane'
    if _SFU_CALL.search(text) or re.match(r'^\(\s*1\s*/', text):
        return 'sfu'
    if re.search(r'\bfabsf?\b|\babs\b|\bfmaf?\b|\bfminf?\b|\bfmaxf?\b', text):
        return 'fp'
    if re.search(r'(^|::)swap\b', text):
        return 'int'
    return fallback


def _value_bytes(type_) -> int:
    """Bytes a value of `type_` occupies: element size times vector width."""
    base = getattr(type_, 'base', None)
    try:
        size = base.size()
    except (AttributeError, TypeError):
        return 0
    return size * (getattr(type_, 'length', None) or 1)


def _code_copies(unroll, trips: Optional[int]) -> int:
    """How many copies of a `for` body the compiler lays down.

    `#pragma unroll` over a count it knows unrolls the loop whole; `#pragma
    unroll N` lays down N (and a remainder, not counted); anything else stays a
    loop, one copy.  `trips` is None where the bounds are not constants -- the
    batch loop -- and then even the bare pragma leaves a loop.
    """
    if unroll is True:
        return trips or 1
    if isinstance(unroll, int) and not isinstance(unroll, bool) and unroll > 1:
        return min(unroll, trips) if trips else unroll
    return 1


class Emitter:
    def __init__(self, writer, context: Any = None):
        self.writer = writer
        self.context = context
        #: How many times the statement being written runs per element: the
        #: product of the trip counts of the loops around it that have
        #: constant bounds (`_emit_for`).
        self._work_scale = 1
        #: How many copies of the statement being written the compiler lays
        #: down: the product of the unroll factors of the loops around it
        #: (`_code_copies`).  Not `_work_scale`: a loop rolled by `k_roll` runs
        #: its count and is written once.
        self._code_scale = 1
        self._names: Dict[int, str] = {}
        self._consts: Dict[int, str] = {}
        self._async_lex = None
        self._async_note = ''
        self._prefetch_lex = None
        self._prefetch_note = ''
        self._inline: set = set()
        self._pending: Dict[int, str] = {}   # load.async token id -> C++ name

    def _record_work(self) -> None:
        # The emitter is handed a context or, from older call sites, the VM;
        # only the former counts (`Context.record_work`).
        # Times the trip counts of the loops around it: a loop the compiler
        # unrolls, or one rolled by `k_roll`, is written once and runs its
        # count.  Counted once, rolling a reduction by ten made a kernel look
        # like a tenth of the arithmetic, and a search ranking by the count
        # rolled everything it could.  A loop without constant bounds -- the
        # batch loop -- counts once, so the figure stays per element.
        record = getattr(self.context, 'record_work', None)
        if record is not None:
            record(self._work_scale)

    def _record_code(self, op) -> None:
        """What a statement puts into the instruction stream
        (`Context.record_code`), once per copy of it (`_code_scale`).  One unit
        a statement, two for a branch; a loop's counter, test and branch are
        counted where the loop is (`_emit_for`).  Units, not instructions:
        `analysis.icache` converts, with a factor fitted against the compilers
        (`tools/calibrate_icache.py`)."""
        if op in _NO_CODE or op == Op.FOR:
            return
        self._record_code_units(2 if op == Op.IF
                                else _LOOP_OVERHEAD if op == Op.WHILE else 1)

    def _record_code_units(self, units: int) -> None:
        record = getattr(self.context, 'record_code', None)
        if record is not None:
            record(units * self._code_scale)

    def _record_mix(self, s: Stmt) -> None:
        """What a statement occupies, and what it moves (`analysis.pipeline`).

        Beside `_record_code`, and counted the same way: per statement, times
        the trip counts around it for what is issued (`_work_scale`) and times
        the copies laid down for the code (`_code_scale`).  A register access
        is a name and occupies nothing; everything else is one instruction of
        its category, whatever its width -- a packed FMA or a 16-byte load is
        one issue, which is the point of both.
        """
        mix = getattr(self.context, 'record_mix', None)
        if mix is None:
            return
        category, moved = self._mix_category(s)
        if category is None:
            return
        mix(category, self._work_scale, self._code_scale)
        record = getattr(self.context, 'record_bytes', None)
        if record is not None:
            for key, nbytes in moved:
                record(key, nbytes * self._work_scale)

    def _mix_category(self, s: Stmt):
        """`(category, [(space.direction, bytes per lane)])` of a statement,
        or `(None, [])` where it puts nothing into the stream."""
        op = s.op
        if op in _NO_CODE or op in (Op.FOR, Op.YIELD):
            return None, []
        space = s.accesses[0].space if s.accesses else None
        if op in (Op.LOAD, Op.STORE):
            level = _MIX_SPACES.get(space)
            if level is None:
                return None, []
            value = s.target[0] if op == Op.LOAD else s.args[1]
            nbytes = _value_bytes(getattr(value, 'type', None))
            if op == Op.LOAD:
                key = f'{level}.read'
                # Read by every lane at one address: one broadcast per warp,
                # not a word per lane -- counted per lane it made a shared
                # operand's bandwidth bind `gpu_volume` above its measured
                # time.  And a batch-invariant operand comes out of a cache,
                # not out of what each element streams in.
                if getattr(value, 'uniformity', None) not in (None,
                                                              Uniformity.LANE):
                    key += '.bcast'
                elif (level == 'global' and getattr(getattr(
                        s.args[0], 'obj', None), 'addressing', None)
                      == Addressing.NONE):
                    key += '.const'
                return f'{level}.load', [(key, nbytes)]
            return f'{level}.store', [(f'{level}.write', nbytes)]
        if op == Op.LOAD_ASYNC:
            types = s.attr('types', ())
            return 'global.load', [('global.read',
                                    _value_bytes(types[0] if types else None))]
        if op == Op.COPY_ASYNC:
            nbytes = s.attr('elems', 1) * self.elem_size(s.copy_dst)
            return 'async.copy', [('global.read', nbytes),
                                  ('shared.write', nbytes)]
        if op == Op.PREFETCH:
            return 'global.prefetch', []
        if op in Op.ARITH or op == Op.ACCUM:
            value = s.target[0] if s.target else (s.args[1] if len(s.args) > 1
                                                  else None)
            base = getattr(getattr(value, 'type', None), 'base', None)
            if base == Datatype.F64:
                return 'fp64', []
            return ('fp' if base in _WORK_TYPES else 'int'), []
        if op == Op.CALL:
            if s.attr('asm') is not None:
                return 'matrix', []
            if s.attr('assign'):
                return None, []
            callee = s.attr('callee') or ''
            if callee.startswith(('thread_idx_', 'extern_')):
                return None, []
            if _NO_CODE_CALL.search(callee):
                return None, []
            if _MATRIX_CALL.search(callee):
                return 'matrix', []
            if _FMA_CALL.search(callee):
                wide = any(getattr(getattr(a, 'type', None), 'base', None)
                           == Datatype.F64 for a in s.args)
                return ('fp64' if wide else 'fp'), []
            if _SFU_CALL.search(callee):
                return 'sfu', []
            if _CROSSLANE_CALL.search(callee):
                return 'xlane', []
            return _text_category(callee, 'other'), []
        if op == Op.RAWEXPR:
            if s.attr('crosslane'):
                return 'xlane', []
            # a pointer binding computes an address once
            if s.attr('decl') is not None:
                return 'int', []
            return _text_category(s.text or '', 'other'), []
        if op == Op.RAWSTMT:
            text = (s.text or '').lstrip()
            if s.attr('bare_newline') or not text or text.startswith('//'):
                return None, []
            return _text_category(text, 'other'), []
        if op == Op.RAWBLOCK:
            return 'branch', []
        if op == Op.BARRIER:
            return 'barrier', []
        if op in (Op.WAIT, Op.COMMIT_ASYNC):
            return 'sync', []
        if op in (Op.IF, Op.WHILE, Op.EXIT):
            return 'branch', []
        return 'other', []

    # -- naming ------------------------------------------------------------ #

    def name(self, v: Value) -> str:
        return self._names.setdefault(v.id, str(v))

    def bind(self, v: Value, text: str) -> None:
        self._names[v.id] = text

    def operand(self, x: Operand, type_=None) -> str:
        """A value's name, or an immediate spelled for `type_`.

        `str(x)` is fine for the index arithmetic that supplies most
        immediates, and wrong for a float one: Python prints an infinity as
        `inf`, which is not C++. That only became reachable once a loop
        carried an operator's neutral element -- `MaxOperator`'s is
        `-math.inf` -- as its initial value, since `Op.CONST` spells its
        value through `Datatype.literal` and a loop init did not.
        """
        if isinstance(x, Value):
            if x.id in self._consts:
                return self._consts[x.id]
            return self.name(x)
        if isinstance(x, bool):
            return 'true' if x else 'false'
        if isinstance(x, (int, float)) and isinstance(type_, ScalarType):
            return type_.base.literal(x)
        if x is None:
            # `str(None)` is `None`, which is a perfectly good C++ identifier
            # and a perfectly bad one to emit.  It arrives when a producer
            # answered with nothing and the consumer used the answer anyway --
            # `Symbol.load` returns None for every structured load under
            # `simd_mode`, and the value flowed into an arithmetic op, which
            # came out as `sycl::max(float(acc), float(None))`.  Loud here,
            # because the alternative is a compiler error pointing at the
            # arithmetic rather than at the load that had no value.
            raise IRError(
                'a None operand reached the emitter; some producer returned '
                'no value and its consumer used the result anyway')
        return str(x)

    # -- types ------------------------------------------------------------- #

    def ctype(self, t, value: Optional[Value] = None) -> str:
        """The C++ spelling of a value's type.

        ``value`` is the value being declared, when there is one.  This
        emitter does not need it -- in SPMD the lane is in the *address*, so a
        value's type says nothing about how it is spread across the wave.  An
        explicitly vectorized emitter needs exactly that, and it needs it at
        every declaration; passing the value here rather than threading a
        second parameter through six call sites is what keeps the two
        emitters one class apart instead of one file apart.
        """
        if isinstance(t, TokenType):
            raise IRError('a completion token has no C++ representation; it '
                          'must not escape into generated code')
        if isinstance(t, ScalarType):
            if t.length is None:
                return t.base.ctype()
            lex = self._lexic()
            if lex is not None:
                return lex.get_fptype(t.base.ctype(), t.length)
            return f'{t.base.ctype()}{t.length}'
        if isinstance(t, BufferType):
            # A buffer *declaration* renders its element type and puts the
            # extent in the declarator -- `float r0[36]`, which `Op.ALLOC`
            # spells itself.  A buffer in a *value* position is a pointer to
            # that element type, and there are such positions now: a rolling
            # pointer carried across a loop's back edge is an `iter_args`
            # entry, and the loop declares it from its type alone.
            #
            # Rendering the element type there produced `float v4 = p0;` for a
            # carried pointer, which is a narrowing conversion the compiler
            # rejects rather than a wrong answer -- but only because the
            # element type happened to be arithmetic.
            # The backend's spelling, not one assembled here.  A pointer's
            # address space is part of its type wherever the target has spaces
            # in its type system, and dropping it produced a declaration that
            # converts the space away -- silently, and only where a pass
            # declares a copy of a pointer rather than the site that bound it.
            lex = self._lexic()
            if lex is not None:
                quals = () if value is None else value.quals
                return lex.pointer_type(
                    t.elem.ctype(), t.space,
                    readonly=getattr(t, 'readonly', False),
                    restrict=Qual.RESTRICT in quals)
            const = 'const ' if getattr(t, 'readonly', False) else ''
            return f'{const}{t.elem.ctype()}*'
        raise IRError(f'cannot render type {t!r}')

    # -- lexic ------------------------------------------------------------- #

    def _vm(self):
        """Accepts either a ``Context`` or a ``VM`` as ``context``."""
        if self.context is None:
            return None
        if hasattr(self.context, 'get_vm'):
            return self.context.get_vm()
        if hasattr(self.context, 'get_lexic'):
            return self.context
        return None

    def _lexic(self):
        vm = self._vm()
        return None if vm is None else vm.get_lexic()

    def _hw(self):
        vm = self._vm()
        return None if vm is None else vm.get_hw_descr()

    def _infix(self, op: str, v: Value, args: Sequence[str]) -> str:
        """`a op b`.  A hook: the explicit-vector emitter spells a comparison
        by the type its result is declared as."""
        return f'{args[0]} {_INFIX[op]} {args[1]}'

    def _lexic_binop(self, op: str, v: Value, args: Sequence[str]) -> str:
        """`min`/`max` as the lexic spells them, by result dtype.

        Falls back to the bare call when there is no lexic -- the IR-level
        tests build bodies without a context, and an emitter that raised there
        would make them require one for an op that is not what they test.
        """
        lex = self._lexic()
        if lex is None:
            return f'{op}({", ".join(args)})'
        base = getattr(v.type, 'base', None)
        return lex.get_operation(_LEXIC_BINOP[op], base, args[0], args[1])

    def _fma(self, v: Value, args: Sequence[str]) -> str:
        """`a * b + c`, unless the lexic spells a vector one itself.

        The infix form is exact for scalars, and for GNU vectors, which
        contract it.  CUDA's vector structs contract it too, through their
        componentwise operators -- but into scalar FMAs, and the paired FMA of
        sm_100 is only reached through a call the lexic names.
        """
        lex = self._lexic()
        if lex is not None and getattr(v.type, 'length', None) is not None:
            spelled = lex.vector_fma(*args)
            if spelled is not None:
                return spelled
        return f'{args[0]} * {args[1]} + {args[2]}'

    def _sync(self, participants=None, threads=None, wave=None) -> str:
        """The instruction for what the barrier says it covers.

        Read off `participants` alone, never off the arrival level: the two
        are different questions and a wave barrier answers them differently.
        `Lexic.sync_mult` is asked only for `MULT`, and only after
        `has_sync_mult` agreed at the same width -- so a target with no
        sub-block rendezvous is never handed a request it has to approximate.

        Falls back to the bare call when there is no lexic: the IR-level tests
        build bodies without a context, and an emitter that raised there would
        make them require one for an op that is not what they test.
        """
        lex = self._lexic()
        if lex is None:
            return '__syncthreads();'
        who = getattr(participants, 'value', 'block')
        if who == 'wave':
            return lex.sync_simd()
        if who == 'grid':
            return lex.sync_grid()
        if who == 'mult' and threads is not None and self._hw() is not None:
            return lex.sync_mult(threads, self._hw())
        # `multgroup` and `block` alike: the block is sized to hold one group
        # wherever a group is what has to meet, so the block barrier is the
        # group's own and there is nothing narrower to reach for.
        return lex.sync_block()

    def _thread_idx(self, axis: str) -> str:
        lex = self._lexic()
        if lex is not None:
            attr = getattr(lex, f'thread_idx_{axis}', None)
            if attr is not None:
                return str(attr)
        return f'threadIdx.{axis}'

    # -- addressing -------------------------------------------------------- #

    def address(self, base: Operand, indices: Sequence[Operand]) -> str:
        idx = [self.operand(i) for i in indices]
        if not idx:
            return '0'
        shape: Optional[Tuple[int, ...]] = None
        if isinstance(base, Value) and isinstance(base.type, BufferType):
            shape = base.type.shape
        else:
            view = getattr(base, 'data_view', None)
            if view is not None:
                shape = tuple(view.shape)
        if shape is None or len(shape) != len(idx):
            return ' + '.join(idx)
        # leading dimension first, matching DataView.get_address
        addr = idx[-1]
        for i in reversed(range(len(idx) - 1)):
            addr = f'{idx[i]} + {shape[i]} * ({addr})'
        return addr

    def elem_size(self, base: Operand) -> int:
        if isinstance(base, Value) and isinstance(base.type, BufferType):
            return base.type.elem.size()
        dt = getattr(base, 'datatype', None)
        return dt.size() if dt is not None else 4

    def elem_type(self, base: Operand):
        """The `Datatype` a buffer holds, or None where the base does not say.

        The same two places `elem_size` reads, returning the type rather than
        its width, because the lexic asks which overload exists for it and a
        size does not answer that.
        """
        if isinstance(base, Value) and isinstance(base.type, BufferType):
            return base.type.elem
        return getattr(base, 'datatype', None)

    def access_type(self, t, base: Operand):
        """The (datatype, width) an access to `base` carries a value of.

        The value's own type where it has one -- a vector-typed value reads
        several elements through one cast, and the width is what decides
        whether a target can hint at that access.  Where it has none (a stored
        literal is not a `Value`), the buffer's element type stands in, which
        is what the access is spelled through anyway.
        """
        if isinstance(t, ScalarType):
            return t.base, (t.length or 1)
        return self.elem_type(base), 1

    def _decide_async(self, body: Tuple[Stmt, ...]) -> None:
        """One decision per kernel body, not per copy.

        The wait counters are a single hardware resource: if even one copy has
        to take the synchronous fallback, the counting no longer describes what
        is in flight, so the whole body goes synchronous.  Mixed mode would be
        silently wrong rather than merely slow.
        """
        self._async_lex = None
        copies = [s for s in walk_stmts(body) if s.op == Op.COPY_ASYNC]
        if not copies:
            return

        lex, hw = self._lexic(), self._hw()
        if lex is None or hw is None:
            self._async_note = 'no hardware description available'
            return
        supported = _ASYNC_ARCH.get(getattr(hw, 'vendor', None))
        if supported is None or not supported(getattr(hw, 'model', '')):
            self._async_note = f'{getattr(hw, "model", "?")} has no async copy path'
            return

        sizes = lex.copy_async_sizes()
        for c in copies:
            nbytes = c.attr('elems', 1) * self.elem_size(c.copy_dst)
            if nbytes not in sizes:
                self._async_note = (f'{nbytes} B per thread is not one of '
                                    f'{sizes} on {hw.model}')
                return
        self._async_lex = lex

    def _decide_prefetch(self, body: Tuple[Stmt, ...]) -> None:
        """One decision per body, and a lighter one than the copies above.

        A hint a target cannot spell is dropped, and dropping it is safe by
        construction: the body computes the same thing, more slowly.  What
        would not be safe is dropping it *quietly*, because "this part has no
        data prefetch" and "nothing asked for one" then read identically in
        the output, and only one of them is worth acting on.

        The question goes to the lexic rather than to a table keyed on the
        vendor.  A SYCL target's answer is about the library it compiles
        against and not about the part it runs on, and HIP compiles for NVIDIA
        as well -- a vendor string answers neither.
        """
        self._prefetch_lex = None
        if not any(s.op == Op.PREFETCH for s in walk_stmts(body)):
            return

        lex, hw = self._lexic(), self._hw()
        if lex is None or hw is None:
            self._prefetch_note = 'no hardware description available'
            return
        if not lex.has_prefetch(hw):
            self._prefetch_note = (f'{getattr(hw, "model", "?")} has no data '
                                   f'prefetch')
            return
        self._prefetch_lex = lex

    def zero(self, t, value: Optional[Value] = None) -> str:
        """What a predicated statement produces where the predicate is false.

        A vector-typed one needs a vector.  `pred ? vec : 0.0f` does not
        compile: `VectorStruct` is an aggregate and a scalar does not convert
        to one, so the ternary has no common type -- which is how `k_width` 2
        turned four corpus cases from slow into unbuildable, all of them a
        wide load under a lane guard.  Value-initialized rather than filled,
        because the guard is the statement's own and the elements it covers
        are the ones the false branch stands for: all of them.
        """
        if isinstance(t, ScalarType) and t.is_vector:
            return f'{self.ctype(t, value)}{{}}'
        return t.base.literal(0)

    def declare(self, v: Value, expr: str, s: Stmt, name: str = None) -> None:
        """Emit `Ty name = expr;`, folding a predicate into a select.

        A predicated statement that produces a value must *not* be wrapped in a
        guard block --- the declaration would be scoped inside it and the value
        would be unusable afterwards.  Lowering to a select also keeps the
        statement hoistable, which a guard region never is.
        """
        if s.predicate is not None and _folds_predicate(s):
            other = s.attr('other')
            other = (self.operand(other) if other is not None
                     else self.zero(v.type, v))
            expr = f'{self.operand(s.predicate)} ? ({expr}) : ({other})'
        if name is None and v.id in self._inline:
            self.bind(v, _atomic(expr) and expr or f'({expr})')
            return
        self.writer(f'{self.ctype(v.type, v)} {name or self.name(v)} = {expr};')

    def window_expr(self, arena: str, offset) -> str:
        """A window `offset` elements into the shared arena.

        The backend's spelling, for the same reason the *type* of the window
        is: on a target where a shared address is an offset rather than a
        pointer, `&arena[off]` takes the address of what `operator[]`
        returned, and the declaration on the left of it names a type that
        expression does not produce.
        """
        lex = self._lexic()
        if lex is None:
            return f'&{arena}[{offset}]'
        return lex.shared_window_expr(arena, offset)

    def base_name(self, base: Operand) -> str:
        if isinstance(base, Value):
            return self.name(base)
        return getattr(base, 'name', str(base))

    def elem_access(self, base: Operand, addr: str, t,
                    relaxed: bool = False, pointer: str = None) -> str:
        """``base[addr]``, reinterpreted when the value is wider than one element.

        A buffer is typed by its element, so a vector-typed access reads or
        writes several of them at once and has to be spelled through a pointer
        of the wider type.  That cast is not new -- ``load_linear`` and
        ``store_linear`` formatted the same one into a raw string.  Putting it
        *here* is what lets a vectorized access stay an ``Op.LOAD``/``Op.STORE``
        with the buffer as an operand: the string form had to leave the
        structured path (``pir_buffer`` was consulted only for ``vec == 1``),
        which cost every pass its view of which buffer the access touches.

        The cast is only defined when ``addr`` is aligned to the wider type.
        Nothing here checks that, exactly as nothing checked it before; the
        legality belongs with whoever chooses the width, not with the spelling.
        """
        # `pointer` overrides the *name* written through, never the base the
        # accesses are attributed to: a rotating buffer's stages are one
        # buffer, and telling a pass otherwise would let it reorder a fill
        # past a read of the stage being filled.
        access = f'{pointer or self.base_name(base)}[{addr}]'
        if isinstance(t, ScalarType) and t.length is not None:
            return f'*({self._vector_ctype(t, relaxed)}*)&{access}'
        return access

    def initializer(self, v: Value, name: str, expr: str) -> str:
        """The declaration that starts a loop-carried value at `expr`.

        Its own method because the two lowerings spell it differently, and the
        difference is not cosmetic: see `EsimdEmitter.initializer`.
        """
        return f'{self.ctype(v.type, v)} {name} = {expr};'

    def _vector_ctype(self, t, relaxed: bool) -> str:
        lex = self._lexic()
        if lex is None:
            return self.ctype(t)
        try:
            return lex.get_fptype(t.base.ctype(), t.length, relaxed=relaxed)
        except TypeError:
            # A lexic that predates the flag spells one type for both.
            return lex.get_fptype(t.base.ctype(), t.length)

    # -- driver ------------------------------------------------------------ #

    def _plan_inlining(self, body: Tuple[Stmt, ...]) -> set:
        """Values that should become expressions rather than declarations.

        A pure single-use value is written straight into its consumer, so a
        migrated construct emits as compactly as the string it replaces.
        Without this every structured op leaves a named temporary behind, and
        the generated source grows with the migration instead of staying
        comparable to it.

        The use has to sit *directly* in the same region: pushing a
        computation into a nested loop would change how often it runs.
        """
        _, uses = def_use(body)
        inline: set = set()

        def scan(stmts: Tuple[Stmt, ...]) -> None:
            here: Dict[int, int] = {}
            for s in stmts:
                for v in s.operands():
                    here[v.id] = here.get(v.id, 0) + 1

            pending: set = set()
            for s in stmts:
                for v in s.operands():
                    if v.id in pending:
                        inline.add(v.id)
                        pending.discard(v.id)
                if s.has_side_effects or s.regions:
                    # Everything still pending has to materialize here.  The
                    # arithmetic is pure, so moving it past a wait or a store
                    # would not change the result -- but it would undo a
                    # deliberately interleaved schedule, which is the whole
                    # point of having placed the statement where it is.
                    pending.clear()
                if (s.pure and not s.regions and not s.has_side_effects
                        and s.effect == Effect.NONE and len(s.target) == 1
                        and s.op not in (Op.CONST, Op.PACK)
                        and not s.attr('escapes')
                        and not s.attr('no_inline')):
                    t = s.target[0]
                    if len(uses.get(t.id, ())) == 1 and here.get(t.id, 0) == 1:
                        pending.add(t.id)
                for r in s.regions:
                    scan(r.body)

        scan(body)
        return inline

    def run(self, body: Tuple[Stmt, ...]) -> None:
        self._inline = self._plan_inlining(body)
        self._decide_async(body)
        if self._async_lex is None and self._async_note:
            self.writer.Comment(f'async copies lowered synchronously: '
                                f'{self._async_note}')
        self._decide_prefetch(body)
        if self._prefetch_lex is None and self._prefetch_note:
            self.writer.Comment(f'prefetch hints dropped: '
                                f'{self._prefetch_note}')
        self._emit_body(body, ())

    def _emit_body(self, body: Tuple[Stmt, ...],
                   yield_to: Tuple[Optional[str], ...]) -> None:
        for s in body:
            if (s.predicate is not None and s.op != Op.YIELD
                    and not _folds_predicate(s)):
                with self.writer.If(self.operand(s.predicate)):
                    self._emit_stmt(s, yield_to)
            else:
                self._emit_stmt(s, yield_to)

    def _emit_stmt(self, s: Stmt, yield_to: Tuple[Optional[str], ...]) -> None:
        w = self.writer
        op = s.op
        self._record_code(op)
        self._record_mix(s)

        if op == Op.CONST:
            v = s.target[0]
            lit = v.type.base.literal(s.attr('value'))
            self._consts[v.id] = lit
            return

        if op == Op.YIELD:
            for target, val in zip(yield_to, s.args):
                if target is None:      # token: lives only in the IR
                    continue
                src = self.operand(val)
                if src != target:
                    w(f'{target} = {src};')
            return

        if op == Op.RAWSTMT:
            if s.attr('bare_newline'):
                self.writer.Emptyline()
            elif s.attr('fmt'):
                w(s.text.format(*[self.operand(a) for a in s.args]))
            else:
                w(s.text)
            return

        if op == Op.RAWEXPR:
            v = s.target[0]
            decl = s.attr('decl')
            if decl is not None:
                # The declarator is the caller's text, not something rendered
                # from `v.type`: a pointer binding reads
                # `const float *const __restrict__ p`, and on the AMD
                # pointer-based path `auto p` with the type inside a cast on
                # the right.  The value still exists, so consumers address the
                # buffer rather than spelling its name.
                extern = s.attr('extern')
                if extern is not None:
                    self.bind(v, extern)
                w(f'{decl} = '
                  f'{s.text.format(*[self.operand(a) for a in s.args])};')
                return
            text = s.text.format(*[self.operand(a) for a in s.args])
            self.declare(v, text, s)
            return

        if op == Op.RAWBLOCK:
            if s.attr('pragma'):
                w(f'#pragma {s.attr("pragma")}')
            for t in s.target:      # a value-producing block declares it first
                w(f'{self.ctype(t.type, t)} {self.name(t)};')
            with w.Block(s.text):
                self._emit_body(s.regions[0].body, yield_to)
            return

        if op == Op.BARRIER:
            sync_instr = self._sync(s.attr('participants'), s.attr('threads'),
                                    s.attr('wave'))
            if not sync_instr and s.attr('handoff') and self._lexic() is not None:
                sync_instr = self._lexic().handoff_fence()
            if sync_instr is not None:
                w(sync_instr)
            return

        if op == Op.ALLOC:
            v = s.target[0]
            t = v.type
            arena = s.attr('arena')
            if arena is not None:
                # A shared buffer is a window into the kernel's one arena, at
                # the offset the builder bumped out of this instruction's
                # declared scratch tail.  Declaring `__shared__` here instead
                # would allocate outside the size ShrMemOpt computed, which is
                # what the occupancy calculation and the barrier placement both
                # read.
                off = s.attr('offset', 0)
                extern = s.attr('extern')
                if extern is not None:
                    self.bind(v, extern)
                window = self.window_expr(arena, off)
                # The arena is an array of the kernel's floating-point type; a
                # buffer of another element -- the boolean a comparison
                # writes, an integer -- is a window of that type into it, and
                # `&arena[off]` is still a pointer to the arena's.  Where the
                # window is an offset rather than a pointer (explicit SIMD), it
                # is the same byte address counted in the other element: it
                # used to be left alone, and `SlmPtr<bool> = SlmPtr<float> +
                # off` does not compile (every SeisSol `damageStep`).  The room
                # reserved is in arena elements, so one no larger than those
                # fits.
                fp = getattr(self.context, 'fp_type', None)
                elem = getattr(t.elem, 'base', t.elem)
                if fp is not None and elem != fp:
                    lex = self._lexic()
                    window = (lex.shared_window_retype(window, elem.ctype())
                              if lex is not None
                              else f'reinterpret_cast<{elem.ctype()}*>({window})')
                w(f'{self.ctype(t, v)} {self.name(v)} = {window};')
                return
            qual = {MemSpace.CONSTANT: 'const '}.get(t.space, '')
            extern = s.attr('extern')
            if extern is not None:
                self.bind(v, extern)
            align = s.attr('align')
            spec = f'alignas({align}) ' if align else ''
            w(f'{spec}{qual}{t.elem.ctype()} {self.name(v)}[{t.volume}]'
              f'{s.attr("init", "")};')
            return

        if op == Op.LOAD:
            lex = self._lexic()

            v = s.target[0]
            addr = self.address(s.args[0], s.args[1:])
            nontemporal = s.attr('nontemporal')
            access = self.elem_access(s.args[0], addr, v.type,
                                      s.attr('align') == 'relaxed')
            # `extern` is the name the macro layer already gave this value.
            # Emitting under it is what lets a named load take the structured
            # path: the consumer still spells `v58_data`, while the address is
            # an operand rather than a string, so aliasing, liveness and the
            # swizzle all see the access.
            named = s.attr('extern')
            if nontemporal:
                # The attribute is the kind of hint (`hints.cache_hint`), and
                # the lexic spells whichever it has.
                dt, width = self.access_type(v.type, s.args[0])
                self.declare(v, f'{lex.glb_load(access, datatype=dt, length=width, nontemporal=nontemporal)}',
                             s, name=named)
            else:
                self.declare(v, access, s, name=named)
            if named:
                self.bind(v, named)
            return

        if op == Op.LOAD_ASYNC:
            tok = s.target[0]
            t = s.attr('types', ())[0]
            name = f'v{tok.id}_{s.attr("hint", "ld")}'
            self._pending[tok.id] = name
            addr = self.address(s.load_base, s.load_index)
            # On AMD this is an ordinary global_load whose s_waitcnt we place
            # ourselves; on NVIDIA the scoreboard stalls at the first use and
            # the matching wait lowers to nothing.
            self.declare(Value(tok.id, t), f'{self.base_name(s.load_base)}[{addr}]',
                         s, name=name)
            return

        if op == Op.STORE:
            addr = self.address(s.args[0], s.args[2:])
            val = s.args[1]
            vt = val.type if isinstance(val, Value) else None
            access = self.elem_access(s.args[0], addr, vt,
                                      s.attr('align') == 'relaxed',
                                      pointer=s.attr('pointer'))
            # A store to global memory goes through the lexic, the same way
            # `Op.LOAD` above goes through `glb_load`: the nontemporal hint is
            # `__stcg` on NVIDIA and `__builtin_nontemporal_store` on AMD, and
            # neither is expressible as an assignment.  Without this the hint
            # would be silently dropped for every store that reaches here --
            # which is why global stores had to stay on the text path.
            space = s.accesses[0].space if s.accesses else None
            lex = self._lexic()
            if space is MemSpace.GLOBAL and lex is not None:
                dt, width = self.access_type(vt, s.args[0])
                w(lex.glb_store(access, self.operand(val),
                                datatype=dt, length=width,
                                nontemporal=s.attr('nontemporal') or False))
                return
            w(f'{access} = {self.operand(val)};')
            return

        if op == Op.PREFETCH:
            # Nothing where the target has no instruction: `run` has said so
            # once for the whole body, so this is not a silent drop.
            if self._prefetch_lex is None:
                return
            addr = self.address(s.prefetch_base, s.prefetch_index)
            text = self._prefetch_lex.prefetch(
                f'&{self.base_name(s.prefetch_base)}[{addr}]',
                datatype=self.elem_type(s.prefetch_base),
                elems=s.attr('elems', 1), level=s.attr('level', 'l2'))
            if text:
                w(text)
            return

        if op == Op.COPY_ASYNC:
            dst_b = self.base_name(s.copy_dst)
            src_b = self.base_name(s.copy_src)
            dst_a = self.address(s.copy_dst, s.copy_dst_index)
            src_a = self.address(s.copy_src, s.copy_src_index)
            elems = s.attr('elems', 1)
            zfill = s.attr('zfill', 0)
            text = None
            if self._async_lex is not None:
                elem = self.elem_size(s.copy_dst)
                text = self._async_lex.copy_async(
                    f'&{dst_b}[{dst_a}]', f'&{src_b}[{src_a}]', elems * elem,
                    zfill * elem)
            if text is not None:
                w(text)
            elif zfill:
                # No zero-filling copy here, and the fallbacks below would read
                # the elements the fill stands for -- which are past the
                # source.  The caller asks the lexic first and narrows; getting
                # here means it did not.
                raise IRError(
                    f'a copy of {elems} elements with {zfill} zero-filled is '
                    f'not available on this target')
            elif elems == 1:
                w(f'{dst_b}[{dst_a}] = {src_b}[{src_a}];')
            else:
                c = f'c{s.target[0].id}'
                w(f'for (int {c} = 0; {c} < {elems}; ++{c}) '
                  f'{{ {dst_b}[({dst_a}) + {c}] = {src_b}[({src_a}) + {c}]; }}')
            return

        if op == Op.COMMIT_ASYNC:
            # Nothing when the copies took the synchronous fallback: there is
            # no group to close, and `_decide_async` has already made that
            # decision once for the whole body.
            if self._async_lex is not None:
                txt = self._async_lex.commit_async()
                if txt:
                    w(txt)
            return

        if op == Op.WAIT:
            # released values simply alias the variable the issue declared
            tok = s.waited
            if tok is not None and tok.id in self._pending:
                for v in s.target:
                    self.bind(v, self._pending[tok.id])
            lex = self._lexic()
            if lex is None:
                return
            cls = s.attr('counter', 'copy')
            # AMD counts both classes in one vmcnt -- but only while the copy
            # path is actually the hardware one; if copies fell back to plain
            # assignments they are not in flight and must not be counted.
            unified = (self._async_lex is not None and
                       getattr(self._hw(), 'vendor', None) == 'amd')
            n = s.attr('prior_unified' if unified else 'prior', 0)
            texts = []
            if cls in ('load', 'all'):
                texts.append(lex.wait_async_regs(n))
            if cls in ('copy', 'all') and self._async_lex is not None:
                texts.append(lex.wait_async(n))
            for txt in texts:
                if txt:
                    w(txt)
            return

        if op == Op.CALL and s.attr('assign'):
            target, value = s.args
            w(f'{self.operand(target)} = {self.operand(value)};')
            return

        if op == Op.CALL and s.attr('asm') is not None:
            # A matrix instruction is one operation over a whole tile.
            self._record_work()
            # The operands are rendered here rather than baked into the
            # template, which is the whole point: the constraint list names
            # values the IR knows, so the split that produced a fragment has a
            # real use and cannot be removed underneath the asm.
            constraints = s.attr('constraints')
            pairs = list(zip(constraints, s.args))
            outs = ', '.join(f'"{c}"({self.operand(v)})'
                             for c, v in pairs if c.startswith(('=', '+')))
            ins = ', '.join(f'"{c}"({self.operand(v)})'
                            for c, v in pairs if not c.startswith(('=', '+')))
            w(f'asm({s.attr("asm")}\n: {outs}\n: {ins}\n);')
            return

        if op == Op.CALL:
            callee = s.attr('callee')
            if callee is not None and callee.startswith('thread_idx_'):
                v = s.target[0]
                self.bind(v, self._thread_idx(callee[-1]))
                return
            if callee is not None and callee.startswith('extern_'):
                # Bound, not declared: the macro layer already emits the name
                # around this body, so the value is a name that exists rather
                # than a call to make.  This is the seam -- the micro IR
                # reasons about a typed, uniformity-carrying value, the macro
                # IR owns what it is called and how it comes to exist.
                v = s.target[0]
                self.bind(v, callee[len('extern_'):])
                return
            args = ', '.join(self.operand(a) for a in s.args)
            if not s.target:
                # A void intrinsic: invoked for what it does to a register it
                # takes by reference, so there is nothing to declare.  The
                # operands still went through `operand()`, which is the point
                # -- the arguments are values the IR knows, not baked-in names.
                w(f'{callee}({args});')
                return
            v = s.target[0]
            self.declare(v, f'{callee}({args})', s)
            return

        if op == Op.DECLARE:
            v = s.target[0]
            # No initializer to inline, so `declare()`'s folding machinery does
            # not apply -- this is the plain declaration the raw text used to
            # emit, byte for byte.
            w(f'{self.ctype(v.type, v)} {self.name(v)}{s.attr("init", "{}")};')
            return

        if op == Op.PACK:
            # Never inlined -- see `_plan_inlining`.  `{a, b}` is an
            # *initializer*, not an expression: `x * {a, b}` is not C++, and
            # inlining a pack into its consumer produced exactly that as soon
            # as a splatted operand met a multiply.  It has to keep its own
            # declaration, which is also how the vendor path has always used
            # it.
            v = s.target[0]
            parts = ', '.join(self.operand(a) for a in s.args)
            self.declare(v, f'{{{parts}}}', s)
            return

        if op == Op.EXTRACT:
            v = s.target[0]
            self.declare(v, f'{self.operand(s.args[0])}[{s.attr("lane")}]', s)
            return

        if op == Op.ACCUM:
            target, value = s.args
            w(f'{self.operand(target)} += {self.operand(value)};')
            return

        if op == Op.FOR:
            self._emit_for(s)
            return

        if op == Op.WHILE:
            self._emit_while(s)
            return

        if op == Op.EXIT:
            with w.If(_unwrap(self.operand(s.exit_cond))):
                w('break;')
            return

        if op == Op.IF:
            self._emit_if(s)
            return

        # A pure operation with several results.  The vendor spells it as a
        # call writing through references, which is a property of the
        # signature and not of the operation -- keeping that spelling out of
        # the IR is what lets CSE hash-cons it.
        if op == Op.SPLIT:
            callee = s.attr('callee')
            if callee is None:
                raise IRError(f'{Op.SPLIT} without a callee attribute')
            for t in s.target:
                w(f'{self.ctype(t.type, t)} {self.name(t)}{{}};')
            outs = ', '.join(self.name(t) for t in s.target)
            args = ', '.join(self.operand(a) for a in s.args)
            w(f'{callee}({outs}, {args});')
            return

        # generic pure op
        if s.target:
            v = s.target[0]
            if _counts_as_work(op, v):
                # One operation, whatever it holds: at a lead width of two the
                # value is a pair and this is one packed FMA for two elements,
                # which is the difference the count exists to show.
                self._record_work()
            args = [self.operand(a) for a in s.args]
            if op in _INFIX and len(args) == 2:
                expr = self._infix(op, v, args)
            elif op == 'fma' and len(args) == 3:
                expr = self._fma(v, args)
            elif op == 'select' and len(args) == 3:
                # The arms in the result's type: an immediate spelled bare is
                # a `double` in C++, and `p ? x : 0.0` turns a float select
                # into two conversions.
                expr = (f'{args[0]} ? {self.operand(s.args[1], v.type)} : '
                        f'{self.operand(s.args[2], v.type)}')
            elif op == 'neg' and len(args) == 1:
                expr = f'-{args[0]}'
            elif op in _LEXIC_BINOP and len(args) == 2:
                expr = self._lexic_binop(op, v, args)
            else:
                # Spelling an unrecognized name as a call is a guess at what
                # the op meant, and one that C++ resolves against whichever
                # headers the translation unit happens to have pulled in.  A
                # call the IR intends is an `Op.CALL` and says so.
                raise IRError(
                    f'no spelling for op {op!r} with {len(args)} operand(s); '
                    f'a function call belongs in `IRBuilder.call`')
            self.declare(v, expr, s)
            return

        raise IRError(f'no lowering for op {op!r}')

    # -- control flow ------------------------------------------------------ #

    def _emit_for(self, s: Stmt) -> None:
        w = self.writer
        lo, hi, step = s.loop_bounds
        ind = s.induction

        # iter_args and results share one C++ variable: no copy at the latch,
        # and the loop reads like an ordinary accumulator loop.
        targets: List[Optional[str]] = []
        for arg, init, res in zip(s.iter_args, s.loop_inits, s.target):
            if isinstance(arg.type, TokenType):
                # a carried token is pure bookkeeping; the hardware counter is
                # what actually crosses the back edge
                targets.append(None)
                continue
            nm = self.name(arg)
            self.bind(res, nm)
            w(self.initializer(arg, nm, self.operand(init, arg.type)))
            targets.append(nm)

        extern = s.attr('extern')
        if extern is not None:
            self.bind(ind, extern)
        i = self.name(ind)
        cmp_ = '<' if not isinstance(step, int) or step > 0 else '>'
        if step == 1:
            advance = f'++{i}'
        elif step == -1:
            advance = f'--{i}'
        else:
            advance = f'{i} += {self.operand(step)}'
        # `ctype` is an override for a loop the macro layer owns: the batch
        # loop compares its induction variable against `numElements`, so it is
        # `size_t` and not the `int32_t` that `INDEX` renders to.
        ind_ctype = self.ctype(ind.type, ind)
        head = (f'{ind_ctype} {i} = {self.operand(lo)}; '
                f'{i} {cmp_} {self.operand(hi)}; {advance}')
        # unroll goes through Writer.For, which folds the pragma into the block
        # head; a separate statement would flush the enclosing speculation and
        # defeat empty-block elision.
        constant = all(isinstance(x, int) for x in (lo, hi, step)) and step
        trips = len(range(lo, hi, step)) if constant else 1
        scale, self._work_scale = self._work_scale, self._work_scale * trips
        copies = _code_copies(s.attr('unroll'), trips if constant else None)
        if not constant or copies < trips:
            self._record_code_units(_LOOP_OVERHEAD)
            # the counter and the test, then the branch, once per iteration
            # that is not unrolled away
            mix = getattr(self.context, 'record_mix', None)
            if mix is not None:
                runs = scale * trips // max(copies, 1)
                mix('int', 2 * runs, 2 * self._code_scale)
                mix('branch', runs, self._code_scale)
        code_scale, self._code_scale = self._code_scale, self._code_scale * copies
        try:
            with w.For(head, unroll=s.attr('unroll') or False):
                self._emit_body(s.regions[0].body, tuple(targets))
        finally:
            self._work_scale = scale
            self._code_scale = code_scale

    def _emit_while(self, s: Stmt) -> None:
        """`Ty i = init; while (true) { ... i = next; }`.

        The induction and its successor share one C++ variable, as a `for`'s
        iter_arg and result do: the `yield` at the bottom is the assignment
        that closes the back edge, so nothing has to be copied at the latch.
        The head carries no condition -- every exit is a statement in the body,
        emitted where the answer it tests becomes available.
        """
        ind = s.induction
        extern = s.attr('extern')
        if extern is not None:
            self.bind(ind, extern)
        name = self.name(ind)
        ind_ctype = self.ctype(ind.type, ind)
        self.writer(f'{ind_ctype} {name} = '
                    f'{self.operand(s.loop_init, ind.type)};')
        with self.writer.While('true'):
            self._emit_body(s.regions[0].body, (name,))

    def _emit_if(self, s: Stmt) -> None:
        w = self.writer
        targets: List[Optional[str]] = []
        for res in s.target:
            if isinstance(res.type, TokenType):
                targets.append(None)
                continue
            nm = self.name(res)
            w(f'{self.ctype(res.type, res)} {nm};')
            targets.append(nm)

        with w.If(_unwrap(self.operand(s.cond))):
            self._emit_body(s.regions[0].body, tuple(targets))
        if len(s.regions) > 1:
            with w.Block('else'):
                self._emit_body(s.regions[1].body, tuple(targets))


def emit(body: Tuple[Stmt, ...], writer, context: Any = None) -> None:
    """Lower ``body`` into ``writer``, in whichever model the lexic asks for.

    The choice is the lexic's because the lexic is where the rest of the
    model already lives -- the kernel attributes, the broadcast spelling, the
    wave barrier.  Splitting the decision between here and there is how the
    old arrangement ended up with an ESIMD kernel attribute on an SPMD body.
    """
    lex = getattr(context, 'get_vm', None)
    simd = False
    if lex is not None:
        try:
            simd = bool(getattr(context.get_vm().get_lexic(), 'simd_mode', False))
        except Exception:
            simd = False
    if simd:
        from .emit_esimd import EsimdEmitter
        EsimdEmitter(writer, context).run(body)
    else:
        Emitter(writer, context).run(body)
