# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""What every shared-memory access costs in bank cycles.

Shared memory is 32 banks of four bytes.  A warp's access is served in as many
cycles as the largest number of *distinct* addresses that land in one bank;
several lanes reading one address broadcast for free.  A tile written a row at
a time and read a column at a time is the usual way to get that wrong, and the
usual first fix -- padding the row -- does not always work: eight rows of nine
wrap past 64 and collide again.

This was done by hand for the NVIDIA B tile, found a 2-way conflict, and led
to `XorSwizzle`.  Doing it by hand does not scale to four backends and 58
cases, and more to the point it does not notice when a conflict *appears*.

Read off the generated source rather than the IR, because the address the
hardware sees is the one that was emitted: a swizzle folded into a constant, a
strength reduction, an index the IR carries as opaque text -- all of them
change the answer, and only the source has them all.

    python3 tools/bank_conflicts.py           # the whole corpus
    python3 tools/bank_conflicts.py --verbose # every conflicting site

What it cannot answer is whether a conflict costs anything end to end.  A
2-way access takes two cycles instead of one; whether that shows up depends on
what else the kernel is waiting for, and that needs hardware.  This counts
cycles the memory system will spend, not time the kernel will take.
"""
import ast
import importlib.util
import re
import sys
from collections import Counter
from typing import List, Tuple
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from tensorforge.common.basic_types import Datatype                # noqa: E402
from tensorforge.common.context import Context                     # noqa: E402
from tensorforge.generators.generator import Generator             # noqa: E402

BANKS = 32
BANK_BYTES = 4

#: 32 lanes, on both vendors.  A CDNA wave64 access to LDS is processed in two
#: halves of 32, and an RDNA wave32 in one, so the unit that contends for banks
#: is 32 lanes either way.
LANES = 32

TARGETS = [('sm_86', 'cuda'), ('gfx90a', 'hip')]

CASES = Path(__file__).resolve().parent.parent / 'tests' / 'cases'

#: `float* name = &arena[...]` -- how a shared window is spelled.  Matching the
#: declaration rather than the name is what keeps this from guessing: `s0` and
#: `v60_btile` have nothing in common except where they come from.
_WINDOW = re.compile(
    r'^\s*(?:const\s+)?(\w+)\s*\*\s*(?:__restrict__\s+)?(\w+)\s*=\s*&\s*(\w+)\[')
_ASSIGN = re.compile(r'^\s*(?:const\s+)?[\w:<>,\s*]+?\s(\w+)\s*=\s*([^;]+);\s*$')

#: `for (int32_t v18_i1 = 0; ...)`.  A loop variable is never declared by a
#: statement the assignment pattern can see, so 77 accesses in the corpus came
#: back unresolved -- and an unresolved access is one the census does not
#: count, which is a blind spot rather than a caveat.
#:
#: Substituting the initialiser is sound because these are loop bounds over
#: tensor dimensions: every lane in the wave is on the same iteration, so the
#: value shifts every lane's address by the same amount and leaves the bank
#: pattern exactly as it was.  Anything lane-dependent reaches the address
#: through `threadIdx`, which is substituted separately.
_FOR_INIT = re.compile(r'\bfor\s*\(\s*(?:const\s+)?[\w:<>,\s*]+?\s(\w+)\s*=\s*'
                       r'([^;]+);')

_ELEM_BYTES = {'float': 4, 'double': 8, 'int32_t': 4, 'uint32_t': 4,
               'int64_t': 8, '__half': 2}


#: `*(tensorforge::VectorT<float, 4>*)&tile[i] = v;` -- a wide access is
#: spelled as a reinterpret cast, and the width is inside the type.  Matching
#: only `\w+` there missed every one of them, which put a `float4` store in the
#: 4-byte column and reported it as conflicting when the hardware serves it in
#: phases that are not.
_CAST = re.compile(r'\*\(\s*([^)]*?)\s*\*\s*\)\s*&\s*\w+\[')
_VECTOR_WIDTH = {}
for _t in ('float', 'double', 'int32_t', '_Float16'):
    for _n in (2, 4, 8, 16):
        _VECTOR_WIDTH[f'tensorforge::VectorT<{_t}, {_n}>'] = _n
        _VECTOR_WIDTH[f'{_t}{_n}'] = _n

#: `if (threadIdx.x < 16)`, `if (threadIdx.x >= 8 && threadIdx.x < 16)`.
_GUARD = re.compile(r'\bif\s*\(([^)]*threadIdx\.x[^)]*)\)')
_BOUND = re.compile(r'threadIdx\.x\s*(<=|>=|<|>|==)\s*(\d+)')


def _lanes_under(conditions):
    """Which lanes reach an access under these guards.

    Without this the tool measures every access as though the whole warp took
    it, and the staging steps here are guarded to a quarter or a half of the
    wave.  Counting inactive lanes into a bank is how a diagnostic earns a
    reputation for crying wolf.

    Only the forms the generator emits are understood; a guard this cannot read
    leaves the lane set full, which over-reports rather than under-reports.
    """
    lanes = set(range(LANES))
    for cond in conditions:
        for op, num in _BOUND.findall(cond):
            n = int(num)
            test = {'<': lambda t: t < n, '<=': lambda t: t <= n,
                    '>': lambda t: t > n, '>=': lambda t: t >= n,
                    '==': lambda t: t == n}[op]
            lanes = {t for t in lanes if test(t)}
    return sorted(lanes) or list(range(LANES))


def _subscripts(line, windows):
    for m in re.finditer(r'\b(\w+)\[([^\[\]]+)\]', line):
        if m.group(1) in windows:
            yield m.group(1), m.group(2)


class Unresolved(Exception):
    pass


def _to_python(expr: str) -> str:
    """The C++ index expression as Python, with integer division.

    Index arithmetic is integral throughout, so `/` is `//`.  Nothing else in
    the subset that reaches here -- `+ - * % ^ & | << >>`, parens, integer
    literals, `threadIdx.x` -- differs between the two languages.
    """
    expr = expr.replace('threadIdx.x', 'tid')
    # Uniform within a warp, so they shift every lane's address equally and
    # leave the bank pattern alone.  `threadIdx.y` indexes the warp inside the
    # block and `blockDim`/`blockIdx` are the same for all of them; refusing
    # the access instead left three of them uncounted.
    expr = expr.replace('threadIdx.y', '0').replace('threadIdx.z', '0')
    expr = re.sub(r'\bblockDim\.[xyz]\b', str(LANES), expr)
    expr = re.sub(r'\bblockIdx\.[xyz]\b', '0', expr)
    expr = re.sub(r'\b(\d+)_i(?:8|16|32|64)\b', r'\1', expr)   # typed literals
    expr = re.sub(r'(?<![/])/(?![/])', '//', expr)
    return expr


class _Lane(ast.NodeVisitor):
    """Evaluate an index expression for one lane, or refuse."""

    def __init__(self, tid: int):
        self.tid = tid

    def visit_Expression(self, node):
        return self.visit(node.body)

    def visit_Constant(self, node):
        if not isinstance(node.value, int):
            raise Unresolved(repr(node.value))
        return node.value

    def visit_Name(self, node):
        if node.id == 'tid':
            return self.tid
        raise Unresolved(node.id)

    def visit_UnaryOp(self, node):
        v = self.visit(node.operand)
        if isinstance(node.op, ast.USub):
            return -v
        if isinstance(node.op, ast.UAdd):
            return v
        raise Unresolved(type(node.op).__name__)

    _BIN = {ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b,
            ast.Mult: lambda a, b: a * b, ast.FloorDiv: lambda a, b: a // b,
            ast.Mod: lambda a, b: a % b, ast.BitXor: lambda a, b: a ^ b,
            ast.BitAnd: lambda a, b: a & b, ast.BitOr: lambda a, b: a | b,
            ast.LShift: lambda a, b: a << b, ast.RShift: lambda a, b: a >> b}

    def visit_BinOp(self, node):
        f = self._BIN.get(type(node.op))
        if f is None:
            raise Unresolved(type(node.op).__name__)
        return f(self.visit(node.left), self.visit(node.right))

    def visit_IfExp(self, node):
        raise Unresolved('conditional')

    def generic_visit(self, node):
        raise Unresolved(type(node).__name__)


def _resolve(expr: str, defs: dict, depth: int = 24) -> str:
    """Substitute local definitions until only `tid` and literals are left.

    Any identifier that has a definition, not only the generator's own
    `v{n}` names: a loop variable is spelled `i`, and matching the allocator's
    naming convention meant 38 accesses in `accumulate_chain` stayed
    unresolved while their definition sat in the table.
    """
    for _ in range(depth):
        # Not after a dot: `threadIdx.x` ends in an identifier that a plain
        # word boundary happily matches, and substituting there rewrites the
        # thread index into whatever `x` happened to be.
        grown = re.sub(r'(?<![.\w])([A-Za-z_]\w*)\b',
                       lambda m: f'({defs[m.group(1)]})' if m.group(1) in defs
                       else m.group(0), expr)
        if grown == expr:
            break
        expr = grown
    return expr


def ways(expr: str, base_bytes: int, width: int = 1, lanes=None):
    """Bank cycles this access costs.

    `base_bytes` is the buffer's element size and `width` how many of them the
    access covers -- the two are separate because the *index* is in base
    elements even when the access is a vector: `*(VectorT<float,4>*)&tile[i]`
    reads floats `i .. i+3`, so the byte address is `i * 4` and the span is
    16.  Folding them into one number said the lane stride was 64 bytes when
    it is 16, and turned a conflict-free store into a reported 4-way.

    A wide access is served in phases of `128 // span` lanes, which is what
    makes that store conflict-free where a whole-warp model calls it 2-way:
    eight lanes of sixteen bytes cover the bank width exactly.
    """
    if lanes is None:
        lanes = list(range(LANES))
    tree = ast.parse(_to_python(expr), mode='eval')
    addrs = [_Lane(t).visit(tree) for t in lanes]
    span = base_bytes * width
    per_phase = max(1, (BANKS * BANK_BYTES) // span)
    worst = 0
    for start in range(0, len(lanes), per_phase):
        seen = {}
        for a in addrs[start:start + per_phase]:
            byte = a * base_bytes
            for off in range(0, span, BANK_BYTES):
                bank = ((byte + off) // BANK_BYTES) % BANKS
                seen.setdefault(bank, set()).add(byte)
        worst = max(worst, max((len(v) for v in seen.values()), default=0))
    return worst


def accesses(source: str):
    """Every shared-memory subscript, with its element size and direction."""
    windows = {}
    defs = {}
    for line in source.splitlines():
        m = _WINDOW.match(line)
        if m:
            ctype, name, _arena = m.groups()
            windows[name] = _ELEM_BYTES.get(ctype, 4)
            continue
        m = _FOR_INIT.search(line)
        if m and 'threadIdx' not in m.group(2):
            defs.setdefault(m.group(1), m.group(2))
        m = _ASSIGN.match(line)
        # The right-hand side has to be an expression, not a memory read: a
        # loaded value is not a function of the lane, and substituting one
        # into an address produces something that is neither.
        if m and '[' not in m.group(1) and '[' not in m.group(2):
            defs[m.group(1)] = m.group(2)

    depth = 0
    guard: List[Tuple[int, str]] = []      # (brace depth, condition)
    for line in source.splitlines():
        cond = _GUARD.search(line)
        for name, index in _subscripts(line, windows):
            wide = _CAST.search(line)
            width = _VECTOR_WIDTH.get(wide.group(1).strip()) if wide else None
            # A cast store writes through the cast, so the line does not begin
            # with the buffer name.  Deciding direction on that alone put every
            # vectorised store in the load column.
            written = (line.strip().startswith(f'{name}[')
                       or (wide is not None
                           and re.search(re.escape(name) + r'\[[^\]]*\]\s*=',
                                         line) is not None))
            lanes = _lanes_under([c for _, c in guard])
            yield (name, index, windows[name], width or 1,
                   'store' if written else 'load', defs, lanes)

        if cond:
            guard.append((depth, cond.group(1)))
        depth += line.count('{') - line.count('}')
        while guard and guard[-1][0] >= depth:
            guard.pop()


def _cases():
    return sorted(p for p in CASES.rglob('*.py') if not p.name.startswith('_'))


def _load(path):
    spec = importlib.util.spec_from_file_location('case', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main(argv):
    verbose = '--verbose' in argv
    from tensorforge.backend.instructions.compute.primitives import nvidia
    nvidia.ENABLED = True          # measure the path even while it is parked

    totals = Counter()
    sites = Counter()
    worst_case = {}

    for path in _cases():
        for arch, backend in TARGETS:
            try:
                mod = _load(path)
                ctx = Context(arch=arch, backend=backend,
                              fp_type=getattr(mod, 'DTYPE', None)
                              or Datatype.F32)
                gen = Generator(mod.descr_list(), ctx)
                gen.generate()
                source = gen.get_kernel()
            except Exception:
                totals['did not generate'] += 1
                continue

            for name, index, base, width, kind, defs, lanes in accesses(source):
                try:
                    w = ways(_resolve(index, defs), base, width, lanes)
                except Unresolved:
                    totals['address not static'] += 1
                    continue
                totals[f'{w}-way'] += 1
                if w > 1:
                    key = (path.stem, backend, name, kind, w)
                    sites[key] += 1
                    prev = worst_case.get((path.stem, backend), 0)
                    worst_case[(path.stem, backend)] = max(prev, w)

    conflicting = sum(v for k, v in totals.items()
                      if k.endswith('-way') and not k.startswith('1-'))
    clean = totals.get('1-way', 0)

    if verbose and sites:
        print('conflicting accesses, by site:')
        for (case, backend, buf, kind, w), n in sites.most_common():
            print(f'  {n:5d}  {w}-way  {case}/{backend}  {buf} {kind}')
        print()

    print(f'{clean + conflicting} static shared accesses over '
          f'{len(_cases())} cases x {len(TARGETS)} targets')
    for k in sorted(k for k in totals if k.endswith('-way')):
        print(f'  {totals[k]:6d}  {k}')
    for k in ('address not static', 'did not generate'):
        if totals[k]:
            print(f'  {totals[k]:6d}  {k}')

    if conflicting:
        print(f'\nworst per case:')
        for (case, backend), w in sorted(worst_case.items(),
                                         key=lambda kv: -kv[1])[:10]:
            print(f'  {w}-way  {case}/{backend}')
    raise SystemExit(1 if conflicting else 0)


if __name__ == '__main__':
    main(sys.argv[1:])
