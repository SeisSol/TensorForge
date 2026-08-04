# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
# SPDX-FileContributor: David Schneller

"""Numeric equivalence of two generated kernels.

Every migration step so far could be checked structurally --- byte-identical,
canonically identical, or the same statement sequence --- because nothing ever
moved between regions.  If-conversion does move statements between regions, and
then only one question matters: does the kernel still compute the same thing?

This interprets the generated CUDA directly for a small subset (declarations,
assignments, `if`/`else`, counted `for`, ternaries, array and pointer
indexing) and compares the resulting memory state.  Reads of never-written
slots yield a deterministic pseudo-random value derived from the slot's
identity, so both versions see the same inputs without anyone having to supply
them.

Limits, stated rather than hidden: one thread at a time, so `__syncthreads()`
is a no-op and genuinely racy code would not be caught.  What it does catch is
a transform that changes a value, drops a store, or reorders two accesses to
the same location --- which is the failure mode of an if-conversion pass.
"""

from __future__ import annotations

import hashlib
import re
import struct
from typing import Dict, List, Optional, Tuple

_DECL = re.compile(
    r'^(?:const\s+)?(?:__restrict__\s+)?'
    r'(?:float|double|int32_t|int|unsigned|size_t|bool|auto|__float128|char)'
    r'(?P<ptr>\s*\*(?:\s*const)?(?:\s*__restrict__)?)?\s+'
    r'(?P<name>\w+)\s*(?P<arr>\[\s*(?P<dim>\d+)\s*\])?\s*'
    r'(?:\{\s*\}|=\s*(?P<init>.+))?$')
_FOR = re.compile(r'^for\s*\((?:const\s+)?\w+\s+(?P<v>\w+)\s*=\s*(?P<a>.+?);'
                  r'\s*\w+\s*(?P<cmp><=?|>=?)\s*(?P<b>.+?);\s*(?P<step>.+?)\)$')
_IF = re.compile(r'^if\s*\((?P<c>.*)\)$')

# How many slots `evaluate(preset=...)` fills per array.  Larger than any
# operand a test case declares; the interpreter reads only what the kernel
# addresses, so the surplus is inert.
_PRESET_SLOTS = 4096


class Slot:
    """Flat memory, addressed by ``(base, index)``."""

    def __init__(self, seed: int = 0):
        self.data: Dict[Tuple[str, int], float] = {}
        self.seed = seed

    def read(self, base: str, idx: int) -> float:
        key = (base, int(idx))
        if key not in self.data:
            h = hashlib.blake2b(f'{self.seed}:{base}:{idx}'.encode(),
                                digest_size=8).digest()
            self.data[key] = (struct.unpack('<Q', h)[0] % 20011) / 1009.0 - 9.9
        return self.data[key]

    def write(self, base: str, idx: int, value) -> None:
        self.data[(base, int(idx))] = value


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


class Abort(Exception):
    """Raised when the interpreter meets something outside its subset."""


def _py(expr: str) -> str:
    """Translate a C expression into a Python one."""
    e = expr
    e = re.sub(r'\b(\d+)_i32\b', r'\1', e)
    e = re.sub(r'(\d)[fF]\b', r'\1', e)
    e = re.sub(r'\b(?:static_cast|reinterpret_cast|const_cast)\s*<[^>]*>\s*', '', e)
    e = re.sub(r'\b__ldcg\s*\(\s*&', 'DEREF(', e)
    e = re.sub(r'\b__ldg\s*\(\s*&', 'DEREF(', e)
    e = e.replace('&&', ' and ').replace('||', ' or ')
    e = re.sub(r'(?<![=!<>&|])!(?!=)', ' not ', e)
    out, i = [], 0                                    # &p[i] -> ADDR(p, i)
    while i < len(e):
        m = re.compile(r'&(\w+)\[').match(e, i)
        if not m:
            out.append(e[i]); i += 1; continue
        depth, j = 1, m.end()
        while j < len(e) and depth:
            depth += (e[j] == '[') - (e[j] == ']')
            j += 1
        out.append(f'ADDR({m.group(1)}, ' + e[m.end():j - 1] + ')')
        i = j
    e = ''.join(out)
    e = e.replace('true', 'True').replace('false', 'False')
    e = e.replace('nullptr', 'None')
    # ternary: rightmost ? : first
    while '?' in e:
        q = e.rindex('?')
        depth = 0
        for i in range(q + 1, len(e)):
            if e[i] == '(':
                depth += 1
            elif e[i] == ')':
                if depth == 0:
                    break
                depth -= 1
            elif e[i] == ':' and depth == 0:
                cond, a, b = e[:q], e[q + 1:i], e[i + 1:]
                e = f'(({a}) if ({cond}) else ({b}))'
                break
        else:
            raise Abort(f'unbalanced ternary in {expr!r}')
    return e


class Interp:
    def __init__(self, mem: Slot, env: Dict[str, object], limit: int = 400000):
        self.mem = mem
        self.env = dict(env)
        self.budget = limit
        self.env['DEREF'] = lambda p, i=0: p[i] if isinstance(p, Ptr) else p
        self.env['ADDR'] = lambda p, i: (p + i) if isinstance(p, Ptr) else p
        for fn in ('min', 'max', 'abs'):
            self.env[fn] = __builtins__[fn] if isinstance(__builtins__, dict) \
                else getattr(__builtins__, fn)
        self.env['fabsf'] = self.env['fabs'] = abs
        import math
        for name, fn in (('sqrt', lambda x: abs(x) ** 0.5), ('exp', math.exp),
                         ('log', lambda x: math.log(abs(x) + 1e-9)),
                         ('pow', lambda a, b: abs(a) ** b),
                         ('tanh', math.tanh), ('sin', math.sin),
                         ('cos', math.cos), ('erf', math.erf)):
            self.env[name] = self.env[name + 'f'] = fn

    def ev(self, expr: str):
        self.budget -= 1
        if self.budget < 0:
            raise Abort('budget exhausted')
        try:
            return eval(_py(expr), {'__builtins__': {}}, self.env)
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
            elif kind == 'block':
                self.run(node[1])

    def assign(self, stmt: str) -> None:
        m = _DECL.match(stmt)
        am = re.match(r'^(?:const\s+)?auto\s*&\s*(\w+)\s*=\s*(.+)$', stmt)
        if am:
            self.env[am.group(1)] = self.ev(am.group(2))   # alias: same object
            return
        if 'pipeline' in stmt or '::' in stmt:
            return
        if m and m.group('name') not in ('return',):
            name = m.group('name')
            if m.group('arr'):
                self.env[name] = Ptr(self.mem, f'{name}#{id(self.env)}')
                for i in range(int(m.group('dim'))):
                    self.env[name][i] = 0.0
                return
            init = m.group('init')
            self.env[name] = self.ev(init) if init else 0.0
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
        if stmt.startswith('extern ') or stmt.startswith('__shared__'):
            return                      # the shared arena, modelled as a base
        if re.match(r'^(?:const\s+)?auto\s*\*?\s*\w+', stmt) and '=' not in stmt:
            return
        if 'pipeline' in stmt or '::' in stmt:
            return                      # cuda::pipeline and friends: no effect
                                        # on the values we compare
        raise Abort(f'unsupported statement {stmt!r}')


def parse(src: str) -> List:
    """Line-based parse into nested statement nodes."""
    lines = []
    for raw in src.splitlines():
        t = raw.split('//')[0].strip()
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
            out.append(('expr', t.rstrip(';')))
        return out

    return block()


def evaluate(src: str, tid: int = 0, seed: int = 0,
             globals_only: bool = False,
             preset: Optional[Dict[str, float]] = None) -> Dict:
    """Run one kernel body for one thread; return the resulting memory.

    ``preset`` maps a global array name to a value every one of its slots is
    filled with before the run.  Unwritten slots otherwise take a
    pseudo-random value derived from their identity, which is what makes two
    runs comparable --- but it also means an input cannot be *chosen*.  Being
    able to zero one operand turns this into a reference-free sensitivity
    test: if the result does not move when a term's operand is zeroed, the
    kernel is not using that term.
    """
    body = src[src.index('{'):]
    mem = Slot(seed)
    if preset:
        for name, value in preset.items():
            # generously wide: the interpreter only ever reads slots the
            # kernel addresses, so over-filling costs nothing
            for idx in range(_PRESET_SLOTS):
                mem.write(name, idx, value)
    env = {
        'threadIdx': type('T', (), {'x': tid, 'y': 0, 'z': 0})(),
        'blockIdx': type('B', (), {'x': 0, 'y': 0, 'z': 0})(),
        'blockDim': type('D', (), {'x': 256, 'y': 1, 'z': 1})(),
        'gridDim': type('G', (), {'x': 1, 'y': 1, 'z': 1})(),

    }
    for name in re.findall(r'\b(m\d+)\b', src):
        env.setdefault(name, Ptr(mem, name))
    for name in re.findall(r'\b(\w*_extraOffset)\b', src):
        env.setdefault(name, 0)
    for name in re.findall(r'\b(numElements\d+)\b', src):
        env.setdefault(name, 1)
    for name in re.findall(r'\b(flags\d+)\b', src):
        env.setdefault(name, None)
    env['totalShrMemPtr'] = Ptr(mem, 'shr')
    interp = Interp(mem, env)
    interp.run(parse(body))
    if globals_only:
        return {k: v for k, v in mem.data.items() if k[0].startswith('m')}
    # scratch arrays are keyed by env identity, which differs between the two
    # runs; compare them by name and index instead
    return {(k[0].split('#')[0], k[1]): v for k, v in mem.data.items()}


# Lanes worth probing: the edges of the usual block widths, where a guard
# flips, plus a few in the interior.
DEFAULT_TIDS = (0, 1, 3, 7, 8, 9, 15, 16, 17, 23, 31, 32, 33, 47, 63)
DEFAULT_SEEDS = (7, 101, 4242)


def compare(a: str, b: str, tids=DEFAULT_TIDS, seeds=DEFAULT_SEEDS,
            globals_only: bool = False):
    """Return ``(equal, message)``.

    Compares *all* of memory by default, not just the global arrays: an
    intermediate a transform got wrong often never reaches a global store for
    the lanes being probed, and restricting the comparison to globals was the
    main reason a changed constant went unnoticed.  Several seeds matter for
    the same reason as several lanes -- a wrong expression can agree with the
    right one at one particular input.
    """
    for seed in seeds:
        for tid in tids:
            try:
                ma = evaluate(a, tid, seed, globals_only)
                mb = evaluate(b, tid, seed, globals_only)
            except Abort as exc:
                return None, f'nicht auswertbar: {exc}'
            if ma != mb:
                diff = [k for k in set(ma) | set(mb) if ma.get(k) != mb.get(k)]
                k = sorted(diff)[0]
                return False, (f'tid={tid} seed={seed}: {k} = {ma.get(k)!r} '
                               f'vs {mb.get(k)!r} ({len(diff)} Unterschiede)')
    return True, 'gleich'
