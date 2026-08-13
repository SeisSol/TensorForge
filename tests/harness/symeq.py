# SPDX-License-Identifier: MIT
"""Symbolic equivalence of two generated kernels' compute bodies.

The AMD conversion changes *how* the arithmetic is written down, not what it
computes: the same MFMA / DPP / transpose instructions, in the same order, on
the same operands.  That claim is worth checking rather than asserting,
because the conversion also let CSE remove more than half the register reads,
and a wrongly reused read is exactly the failure this would produce.

The check treats every vendor intrinsic as an *uninterpreted function*: the
value stored to an output slot becomes a symbolic expression tree over the
input slots, and two kernels are equivalent when those trees match after
renaming the temporaries.  Uninterpreted is the point --- nothing needs to
model what an MFMA does, only that both versions apply it to the same things.

In-place mutation is modelled explicitly: `transpose4x4b32(a,b,c,d, a,b,c,d)`
and `fmacdpp16<r>(c, a, b)` rebind their written arguments, which is how a
wrongly reused pre-transpose value would show up as a mismatch.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# --- statement forms actually produced by the generator --------------------- #

DECL = re.compile(
    r'^\s*(?:const\s+)?(?:float|double|__half|__float128|int32_t|int64_t|'
    r'unsigned|size_t|auto|tensorforge::VectorT<[^>]*>)\s+'
    r'(\w+)\s*=\s*(.+);\s*$')
DECL_EMPTY = re.compile(
    r'^\s*(?:const\s+)?(?:float|double|__half|__float128|'
    r'tensorforge::VectorT<[^>]*>)\s+(\w+)\{\}\s*;\s*$')
ASSIGN = re.compile(r'^\s*([A-Za-z_]\w*)\s*\[([^\]]*)\]\s*=\s*(.+);\s*$')
# `v15 = mfma(...);` -- the pre-SSA accumulator update.  Matching this is not
# optional: without it the whole accumulation chain of the old kernel is
# invisible and every output slot compares as an untouched zero.
REASSIGN = re.compile(r'^\s*([A-Za-z_]\w*)\s*=\s*(.+);\s*$')
VOIDCALL = re.compile(r'^\s*((?:tensorforge::)?\w+(?:<[^>]*>)?)\s*\((.*)\)\s*;\s*$')
REF = re.compile(r'^\s*auto&\s*(\w+)\s*=\s*(\w+)\s*;\s*$')

# For each void intrinsic: which argument positions it *writes*, and which
# ones contribute to the value it computes.  The two differ, and conflating
# them is what makes an output parameter look like an input.
#
# `transpose4x4b32(w1..w4, v1..v4)` writes the first four and reads the last
# four.  Emitted in place the two sets are the same names, and the value is
# still built from what they held *before* the call, so one rule covers both
# forms -- which is what lets an in-place kernel be compared against the SSA
# one that replaced it.
#
# `fmacdpp{n}(c, a, b)` accumulates, so `c` is written *and* read.
SIGNATURES = {
    'transpose4x4b32':
        lambda n: (range(4), range(4, n) if n > 4 else range(4)),
    'transpose16x16b32': lambda n: (range(n), range(n)),
    'transpose32x32b32': lambda n: (range(n), range(n)),
    'fmacdpp4': lambda n: (range(1), range(n)),
    'fmacdpp8': lambda n: (range(1), range(n)),
    'fmacdpp16': lambda n: (range(1), range(n)),
}


def split_args(s: str):
    """Top-level comma split, respecting (), [] and <>."""
    out, depth, cur = [], 0, ''
    for ch in s:
        if ch in '([<':
            depth += 1
        elif ch in ')]>':
            depth -= 1
        if ch == ',' and depth == 0:
            out.append(cur.strip())
            cur = ''
        else:
            cur += ch
    if cur.strip():
        out.append(cur.strip())
    return out


class Env:
    """Symbolic store.  Values are nested tuples; leaves are strings."""

    def __init__(self):
        self.var = {}       # name -> expr
        self.mem = {}       # (array, index-expr) -> expr
        self.alias = {}     # reference name -> array name

    def base(self, name):
        return self.alias.get(name, name)

    def read_slot(self, arr, idx):
        key = (self.base(arr), idx)
        if key not in self.mem:
            # never written: a stable symbol, identical in both kernels
            self.mem[key] = ('in', key[0], idx)
        return self.mem[key]

    def eval(self, expr: str):
        expr = expr.strip()
        while expr.startswith('(') and _balanced(expr[1:-1]):
            expr = expr[1:-1].strip()

        m = re.fullmatch(r'([A-Za-z_]\w*(?:::\w+)*(?:<[^>]*>)?)\s*\((.*)\)', expr,
                         re.S)
        if m:
            fn = m.group(1)
            return (_norm_fn(fn),) + tuple(self.eval(a)
                                           for a in split_args(m.group(2)))

        m = re.fullmatch(r'([A-Za-z_]\w*)\s*\[([^\]]*)\]', expr, re.S)
        if m:
            arr, idx = m.group(1), m.group(2).strip()
            if arr in self.var:                    # vector element
                return ('elem', self.var[arr], self.eval_index(idx))
            return self.read_slot(arr, self.eval_index(idx))

        for op in ('*', '+', '-', '/'):
            parts = _top_split(expr, op)
            if len(parts) > 1:
                return (op,) + tuple(self.eval(p) for p in parts)

        if expr in self.var:
            return self.var[expr]
        # A leaf that still carries text -- an address expression, say --  may
        # name generated temporaries.  Those names are allocated per kernel and
        # carry no meaning, so normalise them: the comparison is equivalence
        # *up to renaming*, not textual identity.
        return ('lit', _rename_free(expr))

    def eval_index(self, idx: str):
        """Indices are concrete in the unrolled body; keep them textual but
        normalised, so `8` and `8 ` compare equal."""
        idx = idx.strip()
        try:
            return str(eval(idx, {'__builtins__': {}}, {}))   # noqa: S307
        except Exception:
            return _rename_free(idx)


def _norm_fn(fn: str) -> str:
    return fn.replace('tensorforge::', '').replace('__builtin_amdgcn_', '')


def _rename_free(s: str) -> str:
    return re.sub(r'\bv\d+\w*\b', 'V', s)


def _balanced(s: str) -> bool:
    d = 0
    for ch in s:
        if ch in '([':
            d += 1
        elif ch in ')]':
            d -= 1
            if d < 0:
                return False
    return d == 0


def _top_split(s: str, op: str):
    out, depth, cur = [], 0, ''
    i = 0
    while i < len(s):
        ch = s[i]
        if ch in '([<':
            depth += 1
        elif ch in ')]>':
            depth -= 1
        if depth == 0 and ch == op and cur.strip() and i + 1 < len(s):
            out.append(cur)
            cur = ''
        else:
            cur += ch
        i += 1
    out.append(cur)
    return [p for p in out if p.strip()] if len(out) > 1 else [s]


def interpret(path: Path):
    env = Env()
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith('//') or line.startswith('#'):
            continue
        if line in ('{', '}', ';', '} else {'):
            continue
        if line.startswith(('if', 'for', 'while', 'return', 'else')):
            # control flow appears only around the load/store loops, whose
            # effect the compute body sees through `mem`
            continue

        m = REF.match(line)
        if m:
            env.alias[m.group(1)] = env.base(m.group(2))
            continue
        m = DECL_EMPTY.match(line)
        if m:
            env.var[m.group(1)] = ('zero',)
            continue
        m = DECL.match(line)
        if m:
            env.var[m.group(1)] = env.eval(m.group(2))
            continue
        m = ASSIGN.match(line)
        if m:
            arr, idx, rhs = m.group(1), m.group(2), m.group(3)
            env.mem[(env.base(arr), env.eval_index(idx))] = env.eval(rhs)
            continue
        m = REASSIGN.match(line)
        if m:
            env.var[m.group(1)] = env.eval(m.group(2))
            continue
        m = VOIDCALL.match(line)
        if m:
            fn = _norm_fn(m.group(1))
            stem = re.sub(r'<.*', '', fn)
            args = split_args(m.group(2))
            if stem in SIGNATURES:
                writes, reads = SIGNATURES[stem](len(args))
                # evaluate every read *before* any write lands, so an in-place
                # call still refers to the values its arguments held on entry
                result = (fn,) + tuple(env.eval(args[i]) for i in reads)
                for pos, i in enumerate(writes):
                    name = args[i]
                    if re.fullmatch(r'\w+', name):
                        env.var[name] = ('out', pos, result)
            continue
    return env


def outputs(env):
    """Slots the kernel wrote, keyed by array and index."""
    return {k: v for k, v in env.mem.items() if not _is_pure_input(v)}


def _is_pure_input(v):
    return isinstance(v, tuple) and v and v[0] == 'in'


def compare(old: Path, new: Path):
    a, b = interpret(old), interpret(new)
    oa, ob = outputs(a), outputs(b)
    if set(oa) != set(ob):
        return (f'different output slots: only-old={sorted(set(oa)-set(ob))[:4]} '
                f'only-new={sorted(set(ob)-set(oa))[:4]}')
    for k in sorted(oa):
        if oa[k] != ob[k]:
            return f'slot {k} differs'
    return None
