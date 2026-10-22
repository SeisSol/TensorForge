# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Do the guards actually catch anything?

Every check added here is only worth its runtime if it can fail.  A test that
passes because the property is trivially true, or because the test and the
code under test share a mistake, is worse than no test: it reads as coverage.

So each guard has a matching mutation --- the defect it exists to catch, put
in --- and this runs them all and reports which are caught.  The mutations
are defects this code can actually have, not invented ones:

* `fmacdpp4` emitted for gfx900, where the specializations are switched off
* `fmacdpp8` selected, which the runtime declares nowhere
* the MFMA tail recomputed, so two paths write the same columns
* the `LaneAxis` lane map with `stride` read as packing
* the broadcast layout with the right numbers in the wrong roles
* `0.0f` handed to a `T &` parameter of `transpose4x4b32`
* a shared buffer declared inside a body, outside the sized arena
* two buffers occupied together placed on the same bytes

Source files are edited in place and restored in a `finally`.  Run it on a
clean tree.

A `finally` does not survive SIGKILL, and a timeout is SIGKILL.  When that
happens the tree is left mutated, and the next thing anyone runs is usually
`pytest --snapshot-update`, which records the mutated generator's output as
the new baseline across dozens of files -- during a migration the only symptom
is a large diff, which there would be anyway.  So the mutated paths are also
written to a lock file, checked on startup and restored from git before
anything else runs.

    python3 tools/mutation_check.py            # all groups
    python3 tools/mutation_check.py layout     # one group
"""

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

PKG = Path('src/tensorforge/backend/instructions/compute/primitives/amd')
CORE = Path('src/tensorforge/backend/pir/core.py')
BUILD = Path('src/tensorforge/backend/pir/build.py')
SYM = Path('src/tensorforge/backend/symbol.py')
HIP = Path('src/tensorforge/include/tensorforge_device/hip.h')
EMIT = Path('src/tensorforge/backend/pir/emit.py')
ABSTR = Path('src/tensorforge/backend/instructions/abstract_instruction.py')
EQUIV = Path('tools/access_equiv.py')
ALLOC = Path('src/tensorforge/backend/pir/allocate.py')
LAYOUT = Path('src/tensorforge/backend/pir/layout_check.py')
BARRIERS = Path('src/tensorforge/backend/pir/barriers.py')
WRAP = Path('src/tensorforge/backend/pir/wrap.py')
AHEAD = Path('src/tensorforge/backend/pir/ahead.py')
PREFETCH = Path('src/tensorforge/backend/pir/prefetch.py')
MOVE = Path('src/tensorforge/backend/pir/move.py')
TRANSFERS = Path('src/tensorforge/backend/pir/transfers.py')
TARGET = Path('src/tensorforge/common/target.py')
LEXIC = Path('src/tensorforge/common/vm/lexic/lexic.py')
HIP_LEXIC = Path('src/tensorforge/common/vm/lexic/hip_lexic.py')
HIP_WAVE = Path('src/tensorforge/include/tensorforge_device/hip_wave.h')
SYCL_LEXIC = Path('src/tensorforge/common/vm/lexic/sycl_lexic.py')
LEGALIZE = Path('src/tensorforge/generators/legalize.py')
ORACLE = Path('src/tensorforge/reference/kernel_eval.py')
SHARDS = Path('src/tensorforge/backend/pir/shards.py')
LOAD = Path('src/tensorforge/backend/instructions/memory/load.py')
PASSES = Path('src/tensorforge/backend/pir/passes.py')
GEN = Path('src/tensorforge/generators/generator.py')
KPARAMS = Path('src/tensorforge/generators/kernel_params.py')
OPTIONS = Path('src/tensorforge/common/options.py')
PTR = Path('src/tensorforge/backend/instructions/ptr_manip.py')
ASYNCMEM = Path('src/tensorforge/backend/pir/asyncmem.py')
MLB = Path('src/tensorforge/backend/instructions/builders/multilinear_builder.py')
TRIPS = Path('src/tensorforge/backend/pir/trips.py')
ASYNCMEM = Path('src/tensorforge/backend/pir/asyncmem.py')
PIPE = Path('src/tensorforge/backend/pir/pipeline.py')
ROWS = Path('src/tensorforge/backend/pir/rows.py')
STACKING = Path('src/tensorforge/generators/stacking.py')
BATCHLOOP = Path('src/tensorforge/backend/instructions/batch_loop.py')
TUNING = Path('src/tensorforge/generators/tuning.py')
BOUND = Path('src/tensorforge/analysis/pipeline.py')
RESIDENCY = Path('src/tensorforge/backend/residency.py')
CARRY = Path('src/tensorforge/backend/pir/carry.py')
ATOMICS = Path('src/tensorforge/backend/pir/atomics.py')
FORWARD = Path('src/tensorforge/backend/pir/forward.py')
FLOW = Path('src/tensorforge/backend/pir/flow.py')
TILING = PKG / 'tiling.py'


def _run_tests(target):
    """Run pytest on a mutated tree, with the bytecode cache out of the way.

    Source files here are rewritten several times a second, and CPython
    invalidates a `.pyc` by comparing the source mtime at one-second
    resolution.  Two writes inside the same second therefore leave the stale
    cache in place, and the subprocess imports the *unmutated* module ---
    which reports the guard as working when it was never exercised.  This
    harness exists to catch exactly that class of false confidence, so it had
    better not produce it.
    """
    for cache in Path('src/tensorforge').rglob('__pycache__'):
        shutil.rmtree(cache, ignore_errors=True)
    # This tree's package ahead of any installed one: the mutations are made
    # here, and a run that imported another copy would report every one of
    # them as uncaught.
    path = [str(Path('src').resolve())] + [
        p for p in [os.environ.get('PYTHONPATH')] if p]
    # The lock is this run's own while the tests run: `tests/conftest.py`
    # refuses a tree a killed run left broken, and tells this run from one by
    # `TF_MUTATION_RUN`.
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1',
               PYTHONPATH=os.pathsep.join(path), TF_MUTATION_RUN='1')
    return subprocess.run(
        [sys.executable, '-B', '-m', 'pytest', *target.split(), '-q', '-x',
         '--no-header'],
        capture_output=True, text=True, env=env)


#: What pytest's exit status says about a mutation: tests ran and one failed
#: (1), or the mutated tree did not even collect (2) -- either way the suite
#: notices.  Everything else ran no test that could: a usage error (4), no
#: test collected (5), an internal error (3).  Those are reported, and not
#: counted as caught: a refusal exits nonzero exactly as a failure does.
CAUGHT = (1, 2)


def verdict(returncode: int) -> str:
    """`caught`, `missed` (the tests passed), or what kept them from running."""
    if returncode in CAUGHT:
        return 'caught'
    if returncode == 0:
        return 'missed'
    return f'not run (pytest exit {returncode})'


def sub(path, old, new, count=0):
    """A mutation as (path, replacement-text), failing loudly if it no-ops.

    A mutation that does not apply silently reports its guard as working,
    which is the failure mode this whole file exists to avoid.
    """
    def make():
        if not path.exists():
            # Same failure as a mutation that no longer applies, and reported
            # as one: a `FileNotFoundError` here would abort the whole sweep,
            # so one stale path would hide every group after it.
            raise AssertionError(
                f'{path} does not exist: the file has moved, so this check is '
                f'no longer testing anything')
        text = path.read_text()
        out = text.replace(old, new) if not count else text.replace(old, new, count)
        if out == text:
            raise AssertionError(
                f'mutation did not apply to {path}: the code has moved, so '
                f'this check is no longer testing anything')
        return path, out
    return make



def resub(path, pattern, repl, count=1):
    """`sub` with a pattern, for text whose names are not stable.

    A snapshot names its values by SSA number, and every regeneration
    renumbers them: an anchor spelled with `v41_data` stops matching the
    first time the corpus is rewritten, for no reason the check cares about.
    The pattern names the construct instead -- a call, its arity, a type --
    and the same "did not apply" rule holds.
    """
    compiled = re.compile(pattern)

    def make():
        if not path.exists():
            raise AssertionError(
                f'{path} does not exist: the file has moved, so this check is '
                f'no longer testing anything')
        text = path.read_text()
        out = compiled.sub(repl, text, count=count)
        if out == text:
            raise AssertionError(
                f'mutation did not apply to {path}: the code has moved, so '
                f'this check is no longer testing anything')
        return path, out
    return make


def subs(path, *pairs):
    """Several `sub`s of one file as one mutation, each applied once and in
    order: a defect that is two edits apart."""
    def make():
        if not path.exists():
            raise AssertionError(
                f'{path} does not exist: the file has moved, so this check is '
                f'no longer testing anything')
        text = path.read_text()
        for old, new in pairs:
            out = text.replace(old, new, 1)
            if out == text:
                raise AssertionError(
                    f'mutation did not apply to {path}: the code has moved, '
                    f'so this check is no longer testing anything')
            text = out
        return path, text
    return make

GROUPS = {
    # The IR-level bank analysis.  A mistake in it that over-reports is the
    # kind that gets a check ignored.
    'irbanks': ('tests/test_pir_banks.py', [
        ('the volume rule picks a width the pattern does not want',
         sub(Path('src/tensorforge/backend/instructions/memory/__init__.py'),
             '    while width * 2 <= self._BANKS and volume % (width * granule * 2) == 0:',
             '    while width * 2 <= 8 and volume % (width * granule * 2) == 0:', 1)),
        ('the recommender scores every candidate the same',
         sub(Path('src/tensorforge/backend/pir/banks.py'),
             '                if candidate > 1:',
             '                if False:', 1)),
        ('a dead access counted as one the hardware makes',
         sub(Path('src/tensorforge/backend/pir/pipeline.py'),
             'def optimize(body: Tuple[Stmt, ...], *, explicit_simd: bool = False,',
             'def optimize(body, *_a, **_k):\n    return body\n\n\n'
             'def _optimize(body: Tuple[Stmt, ...], *, explicit_simd: bool = False,', 1)),
        ('guards ignored, so inactive lanes count',
         sub(Path('src/tensorforge/backend/pir/banks.py'),
             '        if parent.op is not Op.IF:\n            continue',
             '        if True:\n            continue', 1)),
        ('the width read from the target for a store too',
         sub(Path('src/tensorforge/backend/pir/banks.py'),
             "            carrier = (stmt.target[0] if stmt.op == Op.LOAD and stmt.target",
             "            carrier = (stmt.target[0] if stmt.target", 1)),
        ('a numpy integer is not an integer again',
         sub(Path('src/tensorforge/backend/pir/banks.py'), '        return operator.index(operand)',
             '        return operand if isinstance(operand, int) else 1 / 0', 1)),
        ('a loop variable no longer resolves to its bound',
         sub(Path('src/tensorforge/backend/pir/banks.py'),
             '        if (stmt.op in (Op.FOR, Op.WHILE) and stmt.regions\n'
             '                and stmt.regions[0].args):',
             '        if False:', 1)),
        ('an unreadable address counted rather than refused',
         sub(Path('src/tensorforge/backend/pir/banks.py'), '                unresolved += 1\n                continue',
             '                continue', 1)),
    ]),

    # The passes over a body, as a pipeline the pass manager runs.
    'pipeline': ('tests/test_pass_pipeline.py', [
        ('a requirement nobody provides, registered anyway',
         sub(Path('src/tensorforge/backend/passmanager.py'),
             '        missing = [r for r in p.requires if r not in available]\n'
             '        if missing:',
             '        missing = [r for r in p.requires if r not in available]\n'
             '        if False:')),
        ('a transform that keeps every fact',
         sub(Path('src/tensorforge/backend/passmanager.py'),
             '            if p.is_transform:\n',
             '            if False:\n')),
        ('a pipeline whose result is not checked for what it delivers',
         sub(Path('src/tensorforge/backend/passmanager.py'),
             '        missing = [r for r in self._delivers if not pc.has(r)]',
             '        missing = []')),
        ('a finding reported without the pass that introduced it',
         sub(Path('src/tensorforge/backend/pir/pipeline.py'),
             "else f'after {stage}: {finding}'",
             "else finding")),
        ('a body finished again to be measured',
         sub(ABSTR,
             '  used = pir.pressure(body, in_bytes=True, explicit_simd=simd)\n'
             '  if used <= FIT_FRACTION * budget:',
             '  used = pir.pressure(finish(builder, body), in_bytes=True,\n'
             '                      explicit_simd=simd)\n'
             '  if used <= FIT_FRACTION * budget:')),
    ]),

    'dryrun': ('tests/test_tools.py::test_every_mutation_still_applies', [
        ('a stale anchor stops being reported',
         sub(Path('tools/mutation_check.py'),
             "                print(f'  {group}: {name}: {exc}')\n                stale += 1",
             "                pass", 1)),
    ]),

    # The bank model.  A mistake in it that over-reports is the kind that gets
    # a check ignored.
    'banks': ('tests/test_bank_conflicts.py', [
        ('the arena counted as a window again',
         sub(Path('tools/bank_conflicts.py'),
             '    for name in arenas:\n        windows.pop(name, None)',
             '    pass', 1)),
        ('loop variables go unresolved again',
         sub(Path('tools/bank_conflicts.py'),
             "        m = _FOR_INIT.search(line)", '        m = None', 1)),
        ('the resolver substitutes after a dot',
         sub(Path('tools/bank_conflicts.py'),
             r"        grown = re.sub(r'(?<![.\w])([A-Za-z_]\w*)\b',",
             r"        grown = re.sub(r'\b([A-Za-z_]\w*)\b',", 1)),
        ('a loaded value taken as an address expression',
         sub(Path('tools/bank_conflicts.py'),
             "if m and '[' not in m.group(1) and '[' not in m.group(2):",
             "if m and '[' not in m.group(1):", 1)),
        ('the vector width is read from a single identifier again',
         sub(Path('tools/bank_conflicts.py'),
             "([^)]*?)\\s*\\*\\s*\\)\\s*&", "(\\w+)\\s*\\*\\s*\\)\\s*&", 1)),
        ('the element size and the access span folded back together',
         sub(Path('tools/bank_conflicts.py'), '            byte = a * base_bytes',
             '            byte = a * base_bytes * width', 1)),
        ('a cast store counted as a load',
         sub(Path('tools/bank_conflicts.py'),
             "            written = (line.strip().startswith(f'{name}[')",
             "            written = (False and line.strip().startswith(f'{name}[')", 1)),
        ('guards ignored, so inactive lanes count',
         sub(Path('tools/bank_conflicts.py'), '            lanes = _lanes_under([c for _, c in guard])',
             '            lanes = None', 1)),
        ('an unreadable guard narrows the lanes to nothing',
         sub(Path('tools/bank_conflicts.py'), '    return sorted(lanes) or list(range(LANES))',
             '    return sorted(lanes)', 1)),
        ('the phase model dropped, so a wide access reads as conflicting',
         sub(Path('tools/bank_conflicts.py'), '    per_phase = max(1, (BANKS * BANK_BYTES) // span)',
             '    per_phase = len(lanes)', 1)),
    ]),

    'ceiling': ('tests/test_tools.py::test_no_shared_access_costs_more_than_four_bank_cycles', [
        ('the window stops being permuted',
         sub(Path('src/tensorforge/backend/instructions/memory/__init__.py'),
             '      if not self._structured_copy(writer):\n        return None',
             '      if True:\n        return None', 1)),
        ('the loader question replaced by the proxy that broke',
         sub(Path('src/tensorforge/backend/instructions/memory/__init__.py'),
             "    if writer is not None and hasattr(self, '_structured_copy'):\n"
             "      if not self._structured_copy(writer):",
             "    if writer is not None and hasattr(self, '_src'):\n"
             "      if self._src.pir_buffer(writer) is None:", 1)),
    ]),

    # A tile's permutation lives on the buffer so no access can forget it.
    'swizzle': ('tests/test_pir_swizzle.py', [
        ('esimd allowed to swizzle, so a vector read reorders itself',
         sub(Path('src/tensorforge/backend/instructions/memory/__init__.py'),
             '    if _explicit_simd(self._context):\n      return None',
             '    if False:\n      return None', 1)),
        ('a window written outside `store` swizzled anyway',
         sub(Path('src/tensorforge/backend/instructions/memory/__init__.py'),
             '      if not self._structured_copy(writer):\n        return None',
             '      if False:\n        return None', 1)),
        ('the vector cast survives the pointer rewrite',
         sub(Path('src/tensorforge/backend/pir/emit_esimd.py'),
             '        m = _VECTOR_ACCESS.match(access.strip())',
             '        m = None', 1)),
        ('the C tile loses its swizzle',
         sub(Path('src/tensorforge/backend/instructions/compute/primitives/nvidia.py'),
             "hint='ctile',\n                        swizzle=XorSwizzle(wave))",
             "hint='ctile')", 1)),
        ('the C tile takes the B tile\'s width',
         sub(Path('src/tensorforge/backend/instructions/compute/primitives/nvidia.py'),
             "hint='ctile',\n                        swizzle=XorSwizzle(wave))",
             "hint='ctile',\n                        swizzle=XorSwizzle(atom.k))", 1)),
        ('a named load falls back to text',
         sub(Path('src/tensorforge/backend/pir/build.py'),
             "            attrs += [('extern', extern)]",
             '            pass', 1)),
        ('a raw access to a swizzled buffer goes unnoticed',
         sub(Path('src/tensorforge/backend/pir/build.py'),
             '        self._check_swizzles_are_total(body)', '        pass', 1)),
        ('the symbol is not resolved to its buffer, so nothing is swizzled',
         sub(Path('src/tensorforge/backend/pir/build.py'),
             "        buf = base.pir_buffer(self) if hasattr(base, 'pir_buffer') else base",
             '        buf = base', 1)),
        ('the window width stops dividing the volume',
         sub(Path('src/tensorforge/backend/instructions/memory/__init__.py'),
             '    while width * 2 <= self._BANKS and volume % (width * granule * 2) == 0:',
             '    while width * 2 <= self._BANKS:', 1)),
        ('an odd volume swizzled anyway',
         sub(Path('src/tensorforge/backend/instructions/memory/__init__.py'),
             '    if width < 2:\n      return None',
             '    if False:\n      return None', 1)),
        ('the volume is taken as the row width again',
         sub(Path('src/tensorforge/backend/instructions/memory/__init__.py'),
             '    volume = 1\n    for n in view.shape:\n      volume *= n',
             '    volume = view.shape[0]', 1)),
        ('the load stops applying it',
         sub(Path('src/tensorforge/backend/pir/build.py'),
             '        indices = self._shifted(tuple(self._swizzled(base, i) for i in indices),\n'
             '                                shift)\n'
             '        if uniform is None:',
             '        indices = self._shifted(tuple(indices), shift)\n'
             '        if uniform is None:', 1)),
        ('the store stops applying it',
         sub(Path('src/tensorforge/backend/pir/build.py'),
             '        indices = self._shifted(tuple(self._swizzled(base, i) for i in indices),\n'
             '                                shift)\n'
             '        kind = Effect.ATOMIC if atomic else Effect.WRITE',
             '        indices = self._shifted(tuple(indices), shift)\n'
             '        kind = Effect.ATOMIC if atomic else Effect.WRITE', 1)),
        ('the permutation stops being a bijection',
         sub(Path('src/tensorforge/backend/pir/core.py'),
             'return (g ^ ((g // self.width) % self.width)) * self.granule + r',
             'return (g ^ (g % self.width)) * self.granule + r', 1)),
        ('a non-power-of-two width accepted',
         sub(Path('src/tensorforge/backend/pir/core.py'),
             'if self.width < 1 or self.width & (self.width - 1):',
             'if False:', 1)),
        ('the divide is by the wrong amount',
         sub(Path('src/tensorforge/backend/pir/build.py'),
             "        bits = swz.width.bit_length() - 1",
             "        bits = swz.width.bit_length()", 1)),
        ('the B tile loses its swizzle',
         sub(Path('src/tensorforge/backend/instructions/compute/primitives/nvidia.py'),
             "hint='btile',\n                        swizzle=XorSwizzle(atom.k))",
             "hint='btile')", 1)),
    ]),

    'op_vocabulary': ('tests/test_pir_op_vocabulary.py', [
        ('an unknown op keeps the permissive defaults',
         sub(CORE,
             "        if self.op in Op.KNOWN or self.op in Op.ARITH:\n"
             "            return",
             "        if True:\n"
             "            return", 1)),
        ('only purity is denied, not free reordering',
         sub(CORE,
             "        harmless = self.pure or (self.movable and not self.accesses\n"
             "                                 and self.effect == Effect.NONE)",
             "        harmless = self.pure", 1)),
        ('the builder takes any name as arithmetic',
         sub(BUILD,
             "        if name not in Op.ARITH:",
             "        if False:", 1)),
        ('the emitter invents a callee again',
         sub(EMIT,
             "                raise IRError(\n"
             "                    f'no spelling for op {op!r} with {len(args)} operand(s); '\n"
             "                    f'a function call belongs in `IRBuilder.call`')",
             "                expr = f'{op}({\", \".join(args)})'", 1)),
    ]),

    # The PTX node.  Its numbering check is the one that matters: a mismatch
    # reads different registers and still compiles.
    'asm': ('tests/test_pir_asm.py', [
        ('the split goes back to a side-effecting call',
         sub(Path('src/tensorforge/backend/pir/build.py'),
             "        self._emit_op(Op.SPLIT, vs, tuple(args), pure=True,",
             "        self._emit_op(Op.SPLIT, vs, tuple(args), pure=False,", 1)),
        ('only the first result is declared',
         sub(Path('src/tensorforge/backend/pir/emit.py'),
             "            for t in s.target:\n                w(f'{self.ctype(t.type, t)} {self.name(t)}{{}};')",
             "            t = s.target[0]\n            w(f'{self.ctype(t.type, t)} {self.name(t)}{{}};')", 1)),
        ('assign does not declare an access on its target',
         sub(Path('src/tensorforge/backend/pir/build.py'),
             '            accesses=(Access(Effect.WRITE, MemSpace.REGISTER, base=target),),',
             '            accesses=(),', 1)),
        ('assign accepts a target with no address',
         sub(Path('src/tensorforge/backend/pir/build.py'),
             "        self._require_addressable(target, 'assign')",
             '        pass', 1)),
        ('assign made movable',
         sub(Path('src/tensorforge/backend/pir/build.py'),
             '            Op.CALL, (), (target, value), pure=False, movable=False,',
             '            Op.CALL, (), (target, value), pure=False, movable=True,', 1)),
        ('the numbering check dropped',
         sub(Path('src/tensorforge/backend/pir/build.py'),
             '        if found != wanted:', '        if False:', 1)),
        ('outputs allowed to follow inputs',
         sub(Path('src/tensorforge/backend/pir/build.py'),
             '            if is_out and seen_input:', '            if False:', 1)),
        ('written operands not declared as accesses',
         sub(Path('src/tensorforge/backend/pir/build.py'),
             "        writes = [v for c, v in operands if c.startswith(('=', '+'))]",
             '        writes = []', 1)),
        ('a write-only constraint not counted as an output',
         sub(Path('src/tensorforge/backend/pir/build.py'),
             "            is_out = constraint.startswith(('=', '+'))",
             "            is_out = constraint.startswith('+')", 1)),
        ('the asm made movable',
         sub(Path('src/tensorforge/backend/pir/build.py'),
             '    def asm_stmt(self, template: str, operands: Sequence[Tuple[str, Operand]],\n'
             '                 *, movable: bool = False) -> Stmt:',
             '    def asm_stmt(self, template: str, operands: Sequence[Tuple[str, Operand]],\n'
             '                 *, movable: bool = True) -> Stmt:', 1)),
        ('uint32_t collapsed onto int32_t',
         sub(Path('src/tensorforge/common/basic_types.py'),
             "           Datatype.U32: 'uint32_t',",
             "           Datatype.U32: 'int32_t',", 1)),
    ]),

    # `flatten_scopes` decides which braces are load-bearing, and it decides
    # it with a regex over raw text.
    'cdecl': ('tests/test_flatten_scopes.py', [
        ('brace initialization not seen as a declaration',
         sub(Path('src/tensorforge/backend/pir/passes.py'),
             r"\w+_t)\s+(\w+)\s*[=;\[{,]')",
             r"\w+_t)\s+(\w+)\s*[=;\[,]')", 1)),
        ('a second declarator not seen',
         sub(Path('src/tensorforge/backend/pir/passes.py'),
             r"\w+_t)\s+(\w+)\s*[=;\[{,]')",
             r"\w+_t)\s+(\w+)\s*[=;\[{]')", 1)),
        ('every region spliced regardless',
         sub(Path('src/tensorforge/backend/pir/passes.py'),
             '                and not _declares(s.regions[0].body)):',
             '                and True):', 1)),
    ]),

    # The diagnostics themselves.  A broken `ir_opacity` reports the whole
    # corpus as failing to generate for as long as nobody re-derives its
    # number.
    'tools': ('tests/test_tools.py::test_ir_opacity_still_generates_the_corpus', [
        ('the counting wrapper stops accepting what it wraps',
         sub(Path('tools/ir_opacity.py'),
             'def _counting_optimize(body, *args, **kwargs):',
             'def _counting_optimize(body):', 1)),
    ]),

    'tools_sites': ('tests/test_tools.py::test_no_site_label_is_ambiguous', [
        ('the site label stops naming a file',
         sub(Path('tools/ir_opacity.py'),
             "            path = Path(name)",
             "            return f'{Path(name).name}:{f.f_code.co_name}'\n            path = Path(name)", 1)),
    ]),

    'caps': ('tests/test_amd_caps.py', [
        ('gfx900 guard removed (the original bug)',
         sub(PKG / 'caps.py', '    return amdarch(ctx) != 0x900',
             '    return True')),
        ('fmacdpp8 offered below gfx10, where the runtime has none',
         sub(PKG / 'caps.py',
             '    return amdarch(ctx) >= 0x1000\n',
             '    return amdarch(ctx) >= 0x900\n', 1)),
        ('codegen widened past the header (gfx908)',
         sub(PKG / 'caps.py',
             'return arch in (0x90a, 0x940, 0x941, 0x942, 0x950) or arch >= 0x1000',
             'return arch in (0x908, 0x90a, 0x940, 0x941, 0x942, 0x950) or arch >= 0x1000')),
        ('header narrowed, codegen not updated',
         sub(HIP, '#if defined(__gfx90a__) || defined(__gfx940__)',
             '#if defined(__gfx940__)', 1)),
    ]),

    # The boundary between the two spans, which `plan` states once.  Stated
    # twice, it can let the spans overlap, so the first two mutations put the
    # boundary where each half of such a mistake would have it.
    'tiling': ('tests/test_amd_tiling.py', [
        ('the boundary ignores the padding decision',
         sub(PKG / 'tiling.py', '    return ((n // fit.width) * fit.width) if empty in (0, fit.width - 1) else n',
             '    return (n // fit.width) * fit.width')),
        ('over-corrected: the tail dropped entirely',
         sub(PKG / 'tiling.py', '    return ((n // fit.width) * fit.width) if empty in (0, fit.width - 1) else n', '    return n')),
        ('padding policy inverted',
         sub(PKG / 'tiling.py', 'if empty in (0, fit.width - 1) else n',
             'if empty not in (0, fit.width - 1) else n')),
        ('a block of one real column padded anyway',
         sub(PKG / 'tiling.py', 'if empty in (0, fit.width - 1) else n',
             'if empty == 0 else n')),
        ('a scheme that pads its own tail asked for a boundary anyway',
         sub(PKG / 'tiling.py',
             '    if fit.scheme is not Scheme.LANE_BATCHED:\n        return n',
             '    if False:\n        return n')),
        ('the empty span is planned rather than dropped',
         sub(PKG / '__init__.py', '    if edge <= 0:', '    if False:')),
    ]),

    'packing': ('tests/test_packing.py', [
        ('waste counted from the wrong end',
         sub(Path('src/tensorforge/backend/instructions/compute/packing.py'),
             '    return (-max(demand, 0)) % capacity',
             '    return max(demand, 0) % capacity')),
        ('packing restarts per product like the unpacked layout',
         sub(Path('src/tensorforge/backend/instructions/compute/packing.py'),
             '    pairs = [(product, step) for product in products '
             'for step in range(steps)]',
             '    pairs = [(product, 0) for product in products '
             'for step in range(steps)]')),
    ]),

    'leadwidth': ('tests/test_compute_strategy.py tests/test_staging.py '
                  'tests/test_bitlayout.py', [
        ('the packing dropped from the layout the plan derives',
         sub(Path('src/tensorforge/backend/instructions/compute/strategy.py'),
             'index.layout(), (threads,), (width,))',
             'index.layout(), (threads,))')),
        ("the operand's slot read where a fragment's lanes were meant",
         sub(Path('src/tensorforge/backend/instructions/compute/strategy.py'),
             'index.layout(), (threads,), (width,))',
             'index.layout(), (threads * width,), (width,))')),
        ('an unstated distribution treated as a packed one',
         sub(Path('src/tensorforge/backend/instructions/compute/bitlayout.py'),
             '    return layout is not None and any(bit.place is Place.VECTOR',
             '    return layout is None or any(bit.place is Place.VECTOR')),
        ('a packed lead operand offered the AMD arrangements anyway',
         sub(Path('src/tensorforge/backend/instructions/compute/primitives/amd/__init__.py'),
             '    if bitlayout.packed(shape.lead_layout):\n'
             '        offered = set()',
             '    if False:\n'
             '        offered = set()')),
        ('a packed lead operand offered the NVIDIA fragments',
         sub(Path('src/tensorforge/backend/instructions/compute/primitives/nvidia.py'),
             '    if not takes(lead_route(shape)):\n        return frozenset()',
             '    if False:\n        return frozenset()')),
        ('a packed lead operand offered DPAS and the Intel chain',
         sub(Path('src/tensorforge/backend/instructions/compute/primitives/intel.py'),
             '    if lead_route(shape) != 0:\n        return frozenset()',
             '    if False:\n        return frozenset()')),
        ('the trip offered as a route every target writes',
         sub(Path('src/tensorforge/backend/instructions/compute/primitives/nvidia.py'),
             '    return route == 0',
             '    return True')),
        ("a target's rungs used where it handed in none",
         sub(Path('src/tensorforge/backend/instructions/compute/routes.py'),
             '    if wave is None or rungs is None or rungs.assembled is None:',
             '    if wave is None:')),
        ('the staged trip counted as a route an emitter writes',
         sub(Path('src/tensorforge/backend/instructions/compute/primitives/amd/relayout.py'),
             '    if route == 0 or route == 1:\n        return True',
             '    if True:\n        return True')),
    ]),

    'regimage': ('tests/test_distribution.py', [
        ('the packing counted as part of the distribution',
         sub(Path('src/tensorforge/backend/symbol.py'),
             '      coords.append(int(idx) // self.lead_width '
             'if position == 0 else int(idx))',
             '      coords.append(int(idx))')),
        ('a replicating pair of axes taken as a bijection',
         sub(Path('src/tensorforge/backend/symbol.py'),
             '  return layout if layout.tiles(num_threads) else None',
             '  return layout')),
        ('the broadcast index divided by the wave on every axis',
         sub(Path('src/tensorforge/backend/symbol.py'),
             '              block = self.lead_block(dim)\n'
             '              bc_index[dim] = LeadIndex(',
             '              block = self.num_threads\n'
             '              bc_index[dim] = LeadIndex(')),
        ('an image with no axes read as though it had them',
         sub(Path('src/tensorforge/backend/symbol.py'),
             '        if len(self.lead_dims) > 1 and '
             'self.register_layout() is None:',
             '        if False:')),
        ('every axis given stride one',
         sub(Path('src/tensorforge/backend/temporaries.py'),
             '            out.append(stride)\n            stride *= block',
             '            out.append(1)\n            stride *= block')),
        ('the slot run counted once per axis',
         sub(Path('src/tensorforge/backend/temporaries.py'),
             "                    block, getattr(self, '_lead_width', 1))",
             "                    block, getattr(self, '_lead_width', 1)) "
             "* DataView.lead_lanes(\n"
             "                    None, _explicit_simd(self._context), "
             "self._num_threads)")),
        ('lane blocks that do not divide the wave admitted',
         sub(Path('src/tensorforge/backend/temporaries.py'),
             '        if self._num_threads and self._num_threads % product != 0:',
             '        if False:')),
        ('a packed image on two axes given an owner anyway',
         sub(Path('src/tensorforge/backend/symbol.py'),
             '    if self.lead_width > 1 and len(self.lead_dims) > 1:\n'
             '      return ()',
             '    if False:\n      return ()')),
        ('an undeclared lead index read as unspread',
         sub(Path('src/tensorforge/backend/symbol.py'),
             '    if dim not in self.lead_dims:\n      return self.num_threads',
             '    if dim not in self.lead_dims:\n      return 1')),
        ("the axis's replication read where its block was meant",
         sub(Path('src/tensorforge/backend/symbol.py'),
             '    return layout.axis(self.lead_dims.index(dim)).block',
             '    return layout.axis(self.lead_dims.index(dim)).stride')),
        ('slots counted in elements rather than in blocks',
         sub(Path('src/tensorforge/backend/symbol.py'),
             '    span = block * lead_width',
             '    span = lead_width')),
        ('positions read as axes where none were stated',
         sub(Path('src/tensorforge/backend/symbol.py'),
             '      if len(self.lead_dims) != 1:\n        return None\n'
             '      axes = (LaneAxis(self.num_threads, 1),)',
             '      axes = tuple(LaneAxis(self.num_threads, 1)\n'
             '                   for _ in self.lead_dims)')),
    ]),

    # Which dimension of an operand's register image is on the lanes, and the
    # read that gives a wrong choice away.
    'laneaxis': ('tests/test_lead_dim.py', [
        ('an operand without the lead index keeps dimension 0 on the lanes',
         sub(Path('src/tensorforge/backend/instructions/builders/'
                  'multilinear_builder.py'),
             '    elif -1 in target and dense:',
             '    elif False:')),
        ('a read one broadcast cannot serve, broadcast anyway',
         sub(Path('src/tensorforge/backend/symbol.py'),
             '          if spread:\n            raise InternalError(',
             '          if False:\n            raise InternalError(')),
    ]),

    # The merge decision, from the bounds the merged probe puts on the size
    # written out.
    'mergebounds': ('tests/test_rolling.py', [
        ('the lower bound taken for the upper',
         sub(Path('src/tensorforge/generators/generator.py'),
             'below is None or above <= budget\n',
             'below is None or below <= budget\n')),
        ('the band decided without building the list written out',
         sub(Path('src/tensorforge/generators/generator.py'),
             '    return self._merge_by_written_out(merged=probe)',
             '    return False')),
        ('a merged run counted once for the size written out',
         sub(EMIT,
             'self._written_scale * (trips if rolled else copies))',
             'self._written_scale * copies)')),
        ('the list built after a probe keeps the probe\'s names',
         sub(Path('src/tensorforge/generators/generator.py'),
             '    self._name_operands(self._descr_list)\n    return False',
             '    return False')),
    ]),

    # A generator that hands its build to another answers from it.
    'delegation': ('tests/test_autotune.py tests/test_rolling.py '
                   'tests/test_icache.py', [
        ('the kernel asked of the generator that handed it on',
         sub(Path('src/tensorforge/generators/generator.py'),
             '    while built._built is not None:\n', '    while False:\n', 1)),
        ('a build taken over with the operands as the last probe named them',
         sub(Path('src/tensorforge/generators/generator.py'),
             '    other._name_operands(other._descr_list)\n', '', 1)),
        ('the configuration the tuner picked tuned again',
         sub(Path('src/tensorforge/generators/generator.py'),
             '    tuned._may_tune = False\n', '', 1)),
        ('a build handed on with this generator\'s own left to read',
         sub(Path('src/tensorforge/generators/generator.py'),
             '    self._sections = self._section = self._scopes = None\n', '',
             1)),
    ]),

    # The searches build with what they are handed, and the generator's
    # modules import one another one way.
    'drivers': ('tests/test_tuning.py', [
        ('the tuner naming the generator',
         sub(Path('src/tensorforge/generators/tuning.py'),
             "    ctx = candidate.context(base, autotune='off')\n",
             "    from tensorforge.generators.generator import Generator  # noqa\n"
             "    ctx = candidate.context(base, autotune='off')\n", 1)),
        ('the lane search naming the generator',
         sub(Path('src/tensorforge/generators/lanes.py'),
             '    scores = {}\n    blocks = {}\n',
             '    from tensorforge.generators.generator import Generator  # noqa\n'
             '    scores = {}\n    blocks = {}\n', 1)),
        ('the preferences reading the tuner',
         sub(Path('src/tensorforge/generators/preferences.py'),
             '    flat = [op for d in descrs for op in d.operations()]\n',
             '    from tensorforge.generators.tuning import _flat\n'
             '    flat = _flat(descrs)\n', 1)),
    ]),

    # What a build measured of what it laid down, as that build's.
    'metrics': ('tests/test_icache.py tests/test_lanes.py', [
        ('the bodies written without counting',
         sub(ABSTR, '    pir.emit(body, writer, context, metrics)',
             '    pir.emit(body, writer, context, None)', 1)),
        ('the register footprint measured unasked',
         sub(ABSTR, '    if metrics is not None and metrics.pressure:',
             '    if metrics is not None:', 1)),
        ('a probe counting into its caller\'s figures',
         sub(Path('src/tensorforge/generators/generator.py'),
             '    other._announce_identity = False\n',
             '    other._announce_identity = False\n'
             '    other._metrics = self._metrics\n', 1)),
        ('the scorer reading a size the build does not keep',
         sub(Path('src/tensorforge/generators/tuning.py'),
             'icache_excess(result.generator.metrics.code_units, hw)',
             "icache_excess(getattr(result.generator, 'code_units', None), hw)",
             1)),
    ]),

    # The shards of the batch-constant operands: held where the loads read
    # them, copied whole by the whole block ahead of a barrier, as many as
    # the budget admits -- least cover first -- and that budget one that
    # keeps the blocks an SM holds.
    'shards': ('tests/test_shard_loads.py', [
        ('a scalar load held',
         sub(SHARDS, "    if 'x' not in leaves:\n        return None\n", '', 1)),
        ('an operand the kernel writes held',
         sub(SHARDS, '        if any(accesses_conflict(w, a) for a in stmt.accesses for w in writes):\n',
             '        if False:\n', 1)),
        ('a load over a loop\'s trips taking the lane blocks with it',
         sub(SHARDS, 'by_buffer.setdefault((load.buffer.id, load.trips), [])',
             'by_buffer.setdefault((load.buffer.id, False), [])', 1)),
        ('a load moved by the shard\'s start the wrong way',
         sub(SHARDS, 'base - shard.lo, scratch)', 'shard.lo - base, scratch)', 1)),
        ('a wide load counted as one element',
         sub(SHARDS, "    width = getattr(stmt.target[0].type, 'length', None) or 1\n",
             '    width = 1\n', 1)),
        ('the distance between runs left out of a copy',
         sub(SHARDS, "b.op('mul', INDEX, k, step - copy.count)", "b.op('mul', INDEX, k, step)", 1)),
        ('a copy larger than the block taken in one pass',
         sub(SHARDS, '    if copy.total <= block:\n', '    if True:\n', 1)),
        ('the copy not ordered ahead of the reads',
         sub(SHARDS, '    b.barrier(Participants.BLOCK)\n', '', 1)),
        ('a shard held past the budget',
         sub(SHARDS, '        if used + need > free:\n            continue\n', '', 1)),
        ('the longest cover first',
         sub(SHARDS, 'key=lambda s: (s.cover, s.position)', 'key=lambda s: (-s.cover, s.position)', 1)),
        ('a shard off the alignment its operand gives it',
         sub(SHARDS, '            end += (shard.lo - end) % align\n', '', 1)),
        ('the copy stepping by the block the body was first built for',
         sub(GEN, '                or section.shards) else None,', '                or False) else None,', 1)),
        ('the budget taking blocks off the SM',
         sub(GEN, '      allowed = cap // max(1, min(limits)) - taken', '      allowed = cap - taken', 1)),
    ]),

    # The host oracle runs the block: every multiplication's threads take
    # their share of what the block copies, and an integer quotient is one.
    'oracleblock': ('tests/test_oracle_block.py', [
        ('every multiplication run as the first',
         sub(ORACLE, '                            mult=mult)', '                            mult=0)', 1)),
        ('an integer quotient left a fraction',
         sub(ORACLE, '        q = abs(a) // abs(b)\n        return q if (a >= 0) == (b >= 0) else -q\n',
             '        return a / b\n', 1)),
        ('a remainder signed like the divisor',
         sub(ORACLE, '        return a - b * _cdiv(a, b)\n', '        return a % b\n', 1)),
    ]),

    # A lane-varying value claims no layout it does not have: an untracked
    # operand the lanes differ in leaves the join untracked, and so does a
    # loop the lanes count from starts of their own.
    'layoutjoin': ('tests/test_layout.py', [
        ('an untracked lane-varying operand passed over in the join',
         sub(CORE, '    if any(x.layout is None and x.uniformity <= '
             'Uniformity.LANE\n           for x in values):\n'
             '        return None\n', '', 1)),
        ('a lane-varying loop counting as a replicated index',
         sub(BUILD, '        if _as_uniformity(uniform) <= Uniformity.LANE:\n',
             '        if False:\n', 1)),
    ]),

    # The verifier has a copy's token in scope behind the region that issued
    # it, where its wait stands, and a load's token only where it was made.
    'risen': ('tests/test_async_commit.py', [
        ("a copy's token kept inside the region that issued it",
         sub(PASSES, '            live.update(_risen(s))\n', '', 1)),
        ("a load's token rising with the copies'",
         sub(PASSES, '            if x.op == Op.COPY_ASYNC for t in x.target}',
             '            if x.op in (Op.COPY_ASYNC, Op.LOAD_ASYNC)'
             ' for t in x.target}', 1)),
    ]),

    # Under an explicit vector a broadcast is an element read out of the
    # vector the lanes make up; an element that keeps its lanes needs a slot
    # axis to come from.
    'lanebroadcast': ('tests/test_layout.py', [
        ('the broadcast read out of no vector',
         sub(PASSES, '                    n = vec.lane_span()\n',
             '                    n = None\n', 1)),
        ('the broadcast read past the lanes the value spans',
         sub(PASSES, '                    n = vec.lane_span()\n',
             '                    n = vec.lane_span() + 1\n', 1)),
        ('an element kept on its lanes taken for a broadcast',
         sub(PASSES, '                if (n is None and vec.distributed and out.layout is not None\n'
             '                        and not out.distributed):\n',
             '                if (n is None and vec.distributed):\n', 1)),
    ]),

    # A synchronous copy's wide accesses state what they prove about their
    # addresses: bytes where the promise covers the access, the relaxed
    # spelling where it does not.
    'hopclaims': ('tests/test_hop_claims.py', [
        ("the copy's load without its claim",
         sub(LOAD, 'nontemporal=_nt, layout=_l, align=_sa)',
             'nontemporal=_nt, layout=_l)', 1)),
        ("the copy's store without its claim",
         sub(LOAD, 'writer.store(_d, value, lhs, align=_da)',
             'writer.store(_d, value, lhs)', 1)),
        ('an operand without a promise claimed at the width',
         sub(LOAD, '    if claim == RELAXED or claim < width * symbol.get_fptype().size():\n'
             '      return RELAXED\n    return claim\n',
             '    if claim == RELAXED:\n      return RELAXED\n'
             '    return max(claim, width * symbol.get_fptype().size())\n', 1)),
        ('a promised operand read through the relaxed type',
         sub(LOAD, '    if claim == RELAXED or claim < width * symbol.get_fptype().size():\n'
             '      return RELAXED\n    return claim\n',
             '    return RELAXED\n', 1)),
        ('a lowering without the relaxed type copying past its promise',
         sub(LOAD, '    proven = (self._use_cuda_memcpy\n'
             '              or not self._context.target.relaxed_vectors)\n',
             '    proven = self._use_cuda_memcpy\n', 1)),
        ('the explicit vector taken for a lowering without the relaxed type',
         sub(TARGET, '        return not self.sycl or self.explicit_simd\n',
             '        return not self.sycl\n', 1)),
    ]),

    # A copy into an image whose rows are padded keeps every row whole: no
    # access crosses the end of a row, and under an explicit vector, where
    # every transfer would, the image keeps the operand's rows.
    'stagerows': ('tests/test_stage_rows.py', [
        ('a stepped access across the end of a row',
         sub(LOAD, '          and _keeps_rows(vecsize, linscale)):',
             '          and True):', 1)),
        ('a bypassing access across the end of a row',
         sub(LOAD, '    if not _keeps_rows(width, linscale):\n      return 0\n',
             '', 1)),
        ('an explicit vector copying into padded rows',
         sub(LOAD, '    if self._explicit_simd():\n      # The image keeps',
             '    if False:\n      # The image keeps', 1)),
    ]),

    # A section is one body, and its traversal's indices are values of it.
    'sectionbody': ('tests/test_section_body.py', [
        ('the traversal indices left unpublished',
         sub(Path('src/tensorforge/generators/generator.py'),
             '      bound.update(self._traversal_indices(body, index))',
             '      pass')),
    ]),

    'sectionbody_text': ('tests/test_mult_groups.py', [
        ('a traversal spelled as text publishes nothing',
         sub(Path('src/tensorforge/backend/instructions/batch_loop.py'),
             "        if not hasattr(writer, 'batch_id'):\n            yield",
             "        if True:\n            yield")),
    ]),

    'staging': ('tests/test_staging.py', [
        ('a vector bit left where it is',
         sub(Path('src/tensorforge/backend/instructions/compute/bitlayout.py'),
             '            if bit.place is Place.VECTOR:',
             '            if False:')),
        ('the freed bits counted but not moved',
         sub(Path('src/tensorforge/backend/instructions/compute/bitlayout.py'),
             '                out.append(Bit(Place.SLOT, 1 << slot))',
             '                out.append(bit)')),
        ('the unpack skipped before the route is asked',
         sub(Path('src/tensorforge/backend/instructions/compute/routes.py'),
             '    have, _ = bitlayout.unpacked(have)',
             '    have = have')),
        ('the assembled exchange preferred over the builtin',
         sub(Path('src/tensorforge/backend/instructions/compute/routes.py'),
             '            and rungs.single(have, want, ext)):\n        return 1',
             '            and False):\n        return 1')),
        ('a route offered that no merge mask expresses',
         sub(Path('src/tensorforge/backend/instructions/compute/primitives/amd/relayout.py'),
             '    return composed if composed is not None and '
             'emittable(composed) else None',
             '    return composed')),
        ('emittable answering the free question again',
         sub(Path('src/tensorforge/backend/instructions/compute/primitives/amd/reorder.py'),
             '    return all(move.select.mergeable for group in groups '
             'for move in group)',
             '    return all(move.select.free for group in groups '
             'for move in group)')),
        ('the two counts compared the wrong way round',
         sub(Path('src/tensorforge/backend/instructions/compute/routes.py'),
             'return composed if rungs.cost(composed) <= stores + loads else trip',
             'return trip if rungs.cost(composed) <= stores + loads else composed')),
        ('two elements sharing an address',
         sub(Path('src/tensorforge/backend/instructions/compute/staging.py'),
             '    for address, index in enumerate(indices):',
             '    for address, index in [(0, i) for i in indices]:')),
        ('the nothing-to-do answer skipped',
         sub(Path('src/tensorforge/backend/instructions/compute/routes.py'),
             '    if gap == ():\n        return 0',
             '    if False:\n        return 0')),
    ]),

    'bitlayout': ('tests/test_bitlayout.py', [
        ('a packed element read as a register of its own',
         sub(Path('src/tensorforge/backend/instructions/compute/bitlayout.py'),
             '            out.append(Bit(Place.VECTOR, 1 << position))',
             '            out.append(Bit(Place.SLOT, 1 << position))')),
        ('a width that is not a power of two taken as unpacked',
         sub(Path('src/tensorforge/backend/instructions/compute/bitlayout.py'),
             '    if low is None or base is None or packed is None:',
             '    packed = packed or 0\n'
             '    if low is None or base is None:')),
        ("the value's type read for everything but its length",
         sub(Path('src/tensorforge/backend/instructions/compute/bitlayout.py'),
             '    width = 1 if length is None else length',
             '    width = 1')),
        ('a replicated fragment planned from its lowest lane',
         sub(Path('src/tensorforge/backend/instructions/compute/primitives/amd/reorder.py'),
             "    if fragment is None or op.replication(which.lower()) != 1:",
             "    if fragment is None:")),
        ('the plan kept for every slot, not the one asked for',
         sub(Path('src/tensorforge/backend/instructions/compute/primitives/amd/reorder.py'),
             '                 for move in found if move.target == slot)',
             '                 for move in found)')),
        ('an unpaired bit move accepted as an exchange',
         sub(Path('src/tensorforge/backend/instructions/compute/bitlayout.py'),
             '    if set(forward) != set(backward):\n        return None',
             '    if False:\n        return None')),
        ('a bit staying on its own side counted as moving',
         sub(Path('src/tensorforge/backend/instructions/compute/bitlayout.py'),
             '        if source.place is target.place:\n            return None',
             '        if False:\n            return None')),
        ('the transpose narrowed by one bit',
         sub(Path('src/tensorforge/backend/instructions/compute/primitives/amd/relayout.py'),
             "    return tuple((1 << bit, 1 << bit) for bit in range((ext - 1).bit_length()))",
             "    return tuple((1 << bit, 1 << bit) for bit in range((ext - 1).bit_length() - 1))")),
        ('a region with two toggles accepted as one',
         sub(Path('src/tensorforge/backend/instructions/compute/bitlayout.py'),
             '        if len(toggles) != 1:\n            return None',
             '        if False:\n            return None')),
        ('the regions keyed by the target slot alone',
         sub(Path('src/tensorforge/backend/instructions/compute/bitlayout.py'),
             '        regions.setdefault((here.slot, there.slot), []).append(',
             '        regions.setdefault((0, there.slot), []).append(')),
        ('the group offset dropped from the destination',
         sub(Path('src/tensorforge/backend/instructions/compute/bitlayout.py'),
             '        there = base_want + want.locate(*index)',
             '        there = want.locate(*index)')),
        ('the destination lanes reported as the source lanes',
         sub(Path('src/tensorforge/backend/instructions/compute/bitlayout.py'),
             'tuple(sorted({there for _, there in pairs}))',
             'tuple(sorted({here for here, _ in pairs}))')),
        ('a slot weight read as a lane weight',
         sub(Path('src/tensorforge/backend/instructions/compute/bitlayout.py'),
             'Bit(Place.LANE if weight > 0 else Place.SLOT, abs(weight))',
             'Bit(Place.LANE, abs(weight))')),
        ('the cut point put on the wrong side of the axis',
         sub(Path('src/tensorforge/backend/instructions/compute/bitlayout.py'),
             '        elif position < packed + low:',
             '        elif position >= packed + low:')),
        ('a stride that is not a power of two admitted anyway',
         sub(Path('src/tensorforge/backend/instructions/compute/bitlayout.py'),
             '    if low is None or base is None or packed is None:\n'
             '        return None',
             '    if low is None or packed is None:\n        return None\n'
             '    base = 0 if base is None else base')),
    ]),

    'lanemerge': ('tests/test_lane_merge.py', [
        ('a region finer than a bank refused again',
         sub(Path('src/tensorforge/backend/instructions/compute/primitives/amd/exchange_codegen.py'),
             "        callee = f'tensorforge::laneMerge<{select.mask}ULL>'",
             '        return None')),
        ('the mask built from the wrong side of the region',
         sub(Path('src/tensorforge/backend/instructions/compute/primitives/amd/reorder.py'),
             '        return sum(1 << lane for lane in self.lanes)',
             '        return sum(1 << lane for lane in self.lanes) >> 1')),
    ]),

    'catalog': ('tests/test_amd_catalog.py', [
        ('tile claims a transpose that does not exist',
         sub(PKG / 'catalog.py', "'tensorforge::transpose32x32b32'",
             "'tensorforge::transpose4x4b32'")),
        ('the unusable tile force-enabled',
         sub(PKG / 'catalog.py',
             '        return self.transpose is None or self.transpose in DEFINED_TRANSPOSES',
             '        return True')),
        ('scale formula off by one',
         sub(PKG / 'catalog.py',
             'return min(self.blocks, threads // self.n).bit_length() - 1',
             'return min(self.blocks, threads // self.n).bit_length()')),
        ('the fits() check dropped',
         sub(PKG / 'catalog.py',
             '        if not self.fits(threads):\n            return False\n',
             '')),
    ]),

    'layout': ('tests/test_layout.py', [
        ('the lane map as first documented',
         sub(CORE,
             '        want = element % self.block\n        return tuple(t for t in range(threads)\n                     if (t // self.stride) % self.block == want)',
             '        want = (element // self.stride) % self.block\n        return tuple(t for t in range(threads)\n                     if t % self.block == want)')),
        ('stride read as packing',
         sub(CORE, '        want = element % self.block',
             '        want = (element // self.stride) % self.block')),
        ('slot formula wrong',
         sub(CORE, '        return element // self.block',
             '        return element // (self.block * self.stride)')),
        ('LeadIndex.layout drops the stride',
         sub(SYM, 'return RegisterLayout((LaneAxis(self._block, self._stride),))',
             'return RegisterLayout((LaneAxis(self._block, 1),))')),
        ('holders unions instead of intersecting',
         sub(CORE, '            out &= set(axis.holders(i, threads))',
             '            out |= set(axis.holders(i, threads))')),
        ('replication ignores the stride',
         sub(CORE,
             '        key = tuple((t // a.stride) % a.block for a in axes)',
             '        key = tuple(t % a.block for a in axes)')),
        ('tiles reintroduced as a second rule',
         sub(CORE, '        return self.replication(threads) == 1',
             '        return all(a.stride == 1 for a in self.axes)')),
        ('extract drops the layout',
         sub(BUILD,
             '        v = self.value(type_, hint=hint, uniform=_join((vec,)),\n                       layout=vec.layout)',
             '        v = self.value(type_, hint=hint, uniform=_join((vec,)))')),
    ]),

    'relayout': ('tests/test_amd_relayout.py', [
        ('the broadcast layout as first annotated',
         sub(PKG / 'relayout.py',
             '    return RegisterLayout((LaneAxis(step, 1),))\n',
             '    return RegisterLayout((LaneAxis(threads // step, step),))\n', 1)),
        ('movdpp16 claims the wrong period',
         sub(PKG / 'relayout.py',
             '    return RegisterLayout((LaneAxis(max(threads // 16, 1), 16),))',
             '    return RegisterLayout((LaneAxis(16, 1),))')),
        ('the transpose row silently drops to rank 1',
         sub(PKG / 'relayout.py',
             '    return RegisterLayout((LaneAxis(4, 1), LaneAxis(max(threads // 4, 1), 4)))',
             '    return RegisterLayout((LaneAxis(4, 1),))')),
        ('a lossy row claims to be lossless',
         sub(PKG / 'relayout.py',
             "    lossy=True,\n    selects_data=('lane',),",
             "    lossy=False,\n    selects_data=('lane',),")),
        ('the simulator models an unknown dpp control',
         sub(Path('tests/harness/wavesim.py'),
             '        raise NotImplementedError(',
             '        return list(vals)\n        raise NotImplementedError(')),
    ]),

    # Where each shared buffer goes, from the lifetimes in the final body.
    'allocate': ('tests/test_pir_allocate.py tests/test_shared_clobber.py', [
        ('a shared alloc declares its own array again',
         sub(EMIT,
             "                w(f'{self.ctype(t, v)} {self.name(v)} = {window};')",
             "                w(f'__shared__ {t.elem.ctype()} {self.name(v)}[{t.volume}];')")),
        ('a window nobody placed emitted at offset zero',
         sub(EMIT, "                if off is None:\n                    raise IRError(",
             "                if off is None and False:\n                    raise IRError(", 1)),
        ('every buffer on the same bytes: occupancy ignored',
         sub(ALLOC, '        taken = sorted((placed[n], placed[n] + by_bit[n].size)',
             '        taken = sorted((placed[n], placed[n]) ', 1)),
        ('a mark no longer ends a value',
         sub(ALLOC, "            if s.attr('mark') == 'defines':",
             "            if False:", 1)),
        ('every write ends a value, the slices of one buffer included',
         sub(ALLOC, '        return uses, defs, 0\n',
             '        return uses, defs, defs & ~uses\n', 1)),
        ('a back edge not followed',
         sub(ALLOC, '            if entry == head:\n                break',
             '            if True:\n                break', 1)),
        ('a statement that says nothing read as touching nothing',
         sub(ALLOC, '                uses |= self.everything\n',
             '                pass\n', 1)),
        ('the stages of a rotating buffer not reserved',
         sub(ALLOC, '        b.size = max(b.size, v.type.volume * stages)',
             '        b.size = max(b.size, v.type.volume)', 1)),
        ('alignment ignored',
         sub(ALLOC, '            at = max(at, _aligned(hi, b.align))',
             '            at = max(at, hi)', 1)),
        ('the block\'s buffers laid over each other',
         sub(ALLOC, '        end_block = at + b.size', '        end_block = at', 1)),
    ]),

    # A transfer issued ahead of where its builder wrote it.
    'move': ('tests/test_pir_move.py', [
        ('a transfer moved past whatever stands above it',
         sub(MOVE, '        if not (register and x.op is Op.BARRIER and not x.regions) and not (\n'
                   '                crosses(moving, footprint(x))):',
             '        if False:', 1)),
        ('a shared transfer moved across a barrier',
         sub(MOVE, '        if not (register and x.op is Op.BARRIER',
             '        if not (x.op is Op.BARRIER', 1)),
        ('the distance ignored',
         sub(MOVE, '            if len(passed) + 1 >= distance:',
             '            if len(passed) + 1 >= 1:', 1)),
        ('a pointer binding crossed',
         sub(MOVE, '        if _binding(x):\n            break',
             '        if False:\n            break', 1)),
        ('moved from the top down',
         sub(MOVE, '    for i in reversed(range(len(found))):',
             '    for i in range(len(found)):', 1)),
        ('its declaration left behind',
         sub(MOVE, '        if (x.op is Op.ALLOC and x.target\n'
                   '                and x.target[0].id == t.dest.id) or (',
             '        if False or (', 1)),
        ('the wait moved with the transfer',
         sub(TRANSFERS, 'Op.WAIT, Op.COMMIT_ASYNC', 'Op.COMMIT_ASYNC')),
        ('the reads a merged run shares lifted out of it',
         sub(PASSES, "            elif loop.attr('rolled'):\n                # A merged run's loop keeps its reads.",
             "            elif False:\n                # A merged run's loop keeps its reads.", 1)),
        ('a transfer through what the trip binds taken out of the loop',
         sub(MOVE, '        if vid not in inside:\n            return True',
             '        if True:\n            return True', 1)),
        ('a transfer from what the loop writes taken out of it',
         sub(MOVE, '                    if writes(a) and any(accesses_conflict(r, a) for r in reads):',
             '                    if False:', 1)),
        ('a buffer something else in the loop writes filled once',
         sub(MOVE, '                        if writes(a) or order[id(x)] < start:',
             '                        if False:', 1)),
        ('a loop that may not run left with its transfers ahead of it',
         sub(AHEAD, '    return (all(isinstance(x, int) for x in (lo, hi, step)) and step > 0\n'
                   '            and lo < hi)',
             '    return (all(isinstance(x, int) for x in (lo, hi, step)) and step > 0)', 1)),
        ('a shared copy taken across the barriers of its loop',
         sub(MOVE, '        if t.shared and barrier:\n            continue',
             '        if False:\n            continue', 1)),
        ('the copy out of the loop and its wait left in it',
         sub(MOVE, '        waits = [x for x in body if x.op is Op.WAIT and x.waited is not None\n'
                   '                 and x.waited.id in tokens]',
             '        waits = []', 1)),
        ('the comment saying what a wait retires left behind',
         sub(MOVE, '            while j >= 0 and neutral(body[j]) and id(body[j]) not in mine:',
             '            while False:', 1)),
    ]),

    # A merged run's loop: the selection its counter makes, a value the
    # bindings read; the destination made resident ahead of the loop rather
    # than an iteration built beside it, what the loop shares taken out of
    # it, and the names its body declares where whatever reads them sees
    # them.
    'runs': ('tests/test_rolling.py::test_the_counter_selects_through_a_value '
             'tests/test_rolling.py::test_the_loop_runs_every_iteration '
             'tests/test_rolling.py::test_the_destination_is_loaded_ahead_of_the_loop '
             'tests/test_rolling.py::test_an_atomic_destination_starts_from_zeros '
             'tests/test_rolling.py::test_what_every_iteration_shares_leaves_the_loop '
             'tests/test_rolling.py::test_where_every_result_goes_out_to_memory_the_run_is_peeled '
             'tests/test_rolling.py::test_a_run_that_reads_its_destination_first_is_peeled '
             'tests/test_rolling.py::test_a_run_that_adds_onto_what_it_reads_is_peeled '
             'tests/test_rolling.py::test_a_destination_assembled_from_several_writes_is_peeled '
             'tests/test_rolling.py::test_a_window_the_loop_fills_is_declared_ahead_of_it '
             'tests/test_rolling.py::test_a_merged_run_computes_what_the_list_written_out_does '
             'tests/test_rolling.py::test_a_merged_accumulation_is_read_back_after_the_loop '
             'tests/test_rolling.py::test_a_run_reads_its_table_and_its_factor_inside_the_loop', [
        ('the select chain left a wall',
         sub(PTR, '    self._value = (writer.uid, self._declare_chain(writer, decl, counter,\n'
                  '                                                   datatype, stars))',
             "    self._declare(writer, f'{decl} = ' + self._chain(\n"
             "        lambda m: self._member(m, datatype, stars), self._variant) + ';')", 1)),
        ('every run peeled as before',
         sub(GEN, '    resident = not missing or self._establish(body, builders, missing)',
             '    resident = not missing', 1)),
        ('a destination written in slices made resident ahead',
         sub(MLB, '        or dest_symbol.stype != SymbolType.Global\n'
                  '        or self._plan.written_in_slices(self._dest_obj.tensor)):',
             '        or dest_symbol.stype != SymbolType.Global):', 1)),
        ('a policy that stores every result taken for one that keeps it',
         sub(MLB, '    if not self._policy.keep_results_in_registers:\n      return False\n',
             '', 1)),
        ('a chain that starts with a read of its destination made resident',
         sub(GEN, '          or any(self._residency_key(v) == key for v in first.reads())):',
             '          or False):', 1)),
        ('the windows a run fills declared in its body',
         sub(PTR, '    self._declare_windows_early(writer, self._region, cleared)\n', '', 1)),
        ('what a run defines not defined behind it',
         sub(PTR, '    out, seen = [], set(scoped)\n', '    return ()\n', 1)),
    ]),

    # A pipeline for every memory path.
    'pipelines': ('tests/test_pir_move.py tests/test_pir_wrap.py '
                  'tests/test_wrap_loads.py::test_a_copy_wraps_in_a_pipeline_of_its_own '
                  'tests/test_wrap_loads.py::test_an_unknown_pipeline_is_refused', [
        ('move counting every transfer, whatever its pipeline',
         sub(MOVE, '                and (not keys or keys[other] == keys[i])):',
             '                ):', 1)),
        ('wrap taking the first transfers of the body, whatever their pipeline',
         sub(WRAP, '            key = pipeline(t)\n', '            key = None\n', 1)),
        ('every path in one pipeline',
         sub(TRANSFERS, '    return t.dest.type.space\n', '    return None\n', 1)),
        ('the option not reaching the passes',
         sub(ABSTR, '    pipeline = pipelines(options.transfer_pipelines)\n',
             "    pipeline = pipelines('path')\n", 1)),
        ('an unknown pipeline taken for the default',
         sub(TRANSFERS, '    found = PIPELINES.get(name)\n',
             '    found = PIPELINES.get(name, memory_path)\n', 1)),
    ]),

    # A transfer for the next element issued right behind the last access to
    # its buffer: behind the last access and not the first, behind the head
    # it reads its pointer from, inside the guard with the masked element's
    # copy in the other arm, and where both paths must agree, at the tail.
    'early': ('tests/test_pir_wrap.py '
              'tests/test_wrap_loads.py::test_wrapped_write_lands_after_the_last_read '
              'tests/test_wrap_loads.py::test_the_next_elements_registers_are_loaded_inside_the_guard '
              'tests/test_wrap_loads.py::test_the_next_elements_registers_are_loaded_at_the_tail_unless_asked '
              'tests/test_wrap_loads.py::test_issued_early_a_load_keeps_no_more_cover_than_it_needs', [
        ('issued early behind the first access to the buffer',
         sub(WRAP, '            last = _last_access(rest, dest)\n',
             '            last = _first_access(rest, dest)\n', 1)),
        ('issued early behind the first access inside the guard',
         sub(WRAP, '                last = _last_access(then, dest)\n',
             '                last = _first_access(then, dest)\n', 1)),
        ('issued early ahead of the head',
         sub(WRAP, '            at = self._spot(rest, max(last, floor),',
             '            at = self._spot(rest, last,', 1)),
        ('a buffer read behind the guard filled inside it',
         sub(WRAP, '                if last is None or _last_access(later, dest) is not None:',
             '                if last is None:', 1)),
        ('the masked element issuing nothing for its successor',
         sub(WRAP, '                rest_else += masked_arm\n', '                pass\n', 1)),
        ('a copy into shared memory issued in one arm of the guard',
         sub(WRAP, '            if (l.early and not p.rotate and l.guard_at is not None\n'
                   '                    and not p.transfer.shared):',
             '            if (l.early and not p.rotate and l.guard_at is not None):', 1)),
        ('a copy issued early ahead of the barrier closing the iteration',
         sub(WRAP, '            if last is None or (p.transfer.shared and closing is not None):',
             '            if last is None:', 1)),
        ('issued early whatever the cover',
         sub(WRAP, '        if self.l.cover <= 0:\n            return at\n',
             '        return at\n', 1)),
        ('issued early whatever the option',
         sub(ABSTR, '                           early=options.wrap_early,\n',
             '                           early=True,\n', 1)),
        ('the cover not reaching the wrap',
         sub(ABSTR, '                           early=options.wrap_early,\n'
                    '                           cover=options.issue_cover,\n',
             '                           early=options.wrap_early,\n'
             '                           cover=0,\n', 1)),
    ]),

    # The cover a transfer keeps: down to it and no further, up to the wait
    # for a copy, past only what it may cross and above a transfer of its
    # own pipeline; and the option reaching `move` through the pipeline.
    'cover': ('tests/test_pir_move.py tests/test_issue_cover.py', [
        ('a capped transfer taken past one of its own pipeline',
         sub(MOVE, '        if other is not None and other != k and keys[other] == keys[k]:\n'
                   '            break\n', '', 1)),
        ('a capped transfer taken past what it may not cross',
         sub(MOVE, '        if not (register and x.op is Op.BARRIER and not x.regions) and not (\n'
                   '                crosses(moving, footprint(x))):\n'
                   '            break\n', '', 1)),
        ('a capped transfer taken below its cover',
         sub(MOVE, '        if ahead - units[n] < cover:\n            break\n', '', 1)),
        ('a copy capped against its first read rather than its wait',
         sub(MOVE, '    copied = bool(t.tokens())\n', '    copied = False\n', 1)),
        ('the cap left out of the pipeline',
         sub(Path('src/tensorforge/backend/pir/pipeline.py'),
             '    if move is not None and move.cover > 0:\n', '    if False:\n', 1)),
        ('the cover not reaching move',
         sub(ABSTR, '                           cover=options.issue_cover)\n'
                    '    if getattr(options, \'enable_wrap_loads\', False):',
             '                           cover=0)\n'
             '    if getattr(options, \'enable_wrap_loads\', False):', 1)),
    ]),

    # A transfer is one tag: a loader out of global memory tags what it
    # writes and nothing else does, two in a row stay two, and the passes
    # ahead of the ones that move a transfer neither lose the tag nor stop at
    # it.
    'tags': ('tests/test_pir_transfers.py', [
        ('two transfers into one buffer read as one',
         sub(TRANSFERS, '                and current.tag == tag(s)\n', '', 1)),
        ('a transfer opened inside a transfer',
         sub(BUILD, "        if self._in_transfer:\n"
                    "            raise IRError('a transfer opened inside a transfer')\n",
             '', 1)),
        ('the tag part of what cse compares',
         sub(PASSES, "    attrs = tuple(a for a in s.attrs if a[0] != 'transfer')\n",
             '    attrs = s.attrs\n', 1)),
        ('a scope a transfer opened kept by flatten',
         sub(PASSES, "                and all(k == 'transfer' for k, _ in s.attrs)\n",
             '                and not s.attrs\n', 1)),
        ('the tag of a scope flatten takes away dropped',
         sub(PASSES, '            out.extend(s.regions[0].body if tag is None\n'
                     '                       else (tagged(x, tag) for x in s.regions[0].body))\n',
             '            out.extend(s.regions[0].body)\n', 1)),
        ('a loader into registers writing no tag',
         sub(LOAD, 'class GlbToRegLoader(MemoryInstruction, LoadInstruction):\n'
                   '  _transfer = True\n',
             'class GlbToRegLoader(MemoryInstruction, LoadInstruction):\n'
             '  _transfer = False\n', 1)),
        ('a loader into shared memory writing no tag',
         sub(LOAD, 'class GlbToShrLoader(AbstractShrMemWrite, LoadInstruction):\n'
                   '  _transfer = True\n',
             'class GlbToShrLoader(AbstractShrMemWrite, LoadInstruction):\n'
             '  _transfer = False\n', 1)),
        ('the wait for a copy tagged as the copy',
         sub(Path('src/tensorforge/backend/instructions/memory/__init__.py'),
             '  _transfer = False\n', '  _transfer = True\n', 1)),
    ]),

    # A transfer into registers without a branch: its ragged tail read on
    # every lane, and the next element's flag in the predicate of its loads
    # -- ahead of the predicate they have, into registers only, and not under
    # the explicit vector.
    'ternary': ('tests/test_pir_wrap.py '
                'tests/test_full_lane_tails.py::test_a_transfer_into_registers_reads_its_tail_on_every_lane '
                'tests/test_full_lane_tails.py::test_a_lead_origin_shift_keeps_the_tail_narrow '
                'tests/test_wrap_loads.py::test_the_next_elements_pointer_is_followed_where_its_flag_holds', [
        ('the tail of a transfer into registers in a branch',
         sub(LOAD, '                             full_lane=self._full_lane_tail(i))]',
             '                             full_lane=False)]', 1)),
        ('the loader reading its tail on every lane whatever the option',
         sub(LOAD, "    if not getattr(opts, 'full_lane_tails', False) or self._lead_width != 1:\n",
             '    if self._lead_width != 1:\n', 1)),
        ('a window of the box read on every lane',
         sub(LOAD, '    return window == (0, box.upper()[i]) and box.lower()[i] == 0\n',
             '    return True\n', 1)),
        ('the flag in a block around a transfer into registers',
         sub(WRAP, '            predicated = (None if self.l.explicit_simd\n'
                   '                          else self._predicated(stmts, cond))\n',
             '            predicated = None\n', 1)),
        ('the flag in the predicate under the explicit vector',
         sub(WRAP, '            predicated = (None if self.l.explicit_simd\n'
                   '                          else self._predicated(stmts, cond))\n',
             '            predicated = self._predicated(stmts, cond)\n', 1)),
        ('the explicit vector not reaching wrap',
         sub(Path('src/tensorforge/backend/pir/pipeline.py'),
             '                             cover=self._cover,\n'
             '                             explicit_simd=pc.explicit_simd,\n',
             '                             cover=self._cover,\n', 1)),
        ("a load's own predicate dropped for the flag",
         sub(WRAP, '            if held is None:\n                return cond\n',
             '            return cond\n', 1)),
        ('a copy into shared memory predicated load by load',
         sub(WRAP, '                if writes(a) and a.space is not MemSpace.REGISTER:\n'
                   '                    return None\n', '', 1)),
    ]),

    # A window copied by the tensor memory accelerator: only where the target
    # has one and the operand proves the alignment; the cursor armed ahead of
    # everything behind a barrier, its barrier kept apart; no group, no
    # barrier behind its wait; one stage; and every wait -- the drains and
    # the one behind the loop included -- a wait on the cursor.
    'bulk': ('tests/test_bulk_copies.py', [
        ('a bulk copy on a part without one',
         sub(TARGET, '        return level is not None and level >= 90\n',
             '        return level is not None and level >= 80\n', 1)),
        ('a bulk copy from a source not proved aligned',
         sub(LOAD, '    if (self._loadsize * elem) % 16 or self._src.linear_align_bytes() < 16:\n',
             '    if (self._loadsize * elem) % 16:\n', 1)),
        ('a bulk copy whatever the option',
         sub(LOAD, "    if not getattr(self._context.get_user_options(), 'bulk_copies', False):\n"
                   '      return False\n', '', 1)),
        ('the cursor armed where it is asked for',
         sub(BUILD, '        body = tuple(self._prologue) + tuple(self._stack[0].body)',
             '        body = tuple(self._stack[0].body)', 1)),
        ('no barrier behind the arming',
         sub(BUILD, '        b.barrier(participants, threads=threads)\n'
                    '        self._prologue += list(b.finish())',
             '        self._prologue += list(b.finish())', 1)),
        ('the barrier sharing bytes with what the body holds after',
         sub(ALLOC, "        if b.arena == MULT and any(w.attr('lifetime') == 'body'",
             "        if False and any(w.attr('lifetime') == 'body'", 1)),
        ('a bulk copy in a group',
         sub(Path('src/tensorforge/backend/pir/asyncmem.py'),
             "    return s.op == Op.COPY_ASYNC and s.counter == 'copy'\n",
             '    return s.op == Op.COPY_ASYNC\n', 1)),
        ('a barrier behind the wait for a bulk copy',
         sub(BARRIERS, "        if s.attr('bulk') is not None:\n",
             '        if False:\n', 1)),
        ('a second stage for a bulk copy',
         sub(WRAP, "        if any(x.op is Op.COPY_ASYNC and x.counter == 'bulk'\n"
                   '               for x in walk_stmts(tuple(t.pieces))):\n',
             '        if False:\n', 1)),
        ('the wait behind the loop without its cursor',
         sub(WRAP, '                effect=p.wait.effect, accesses=accesses,\n'
                   '                attrs=p.wait.attrs))',
             '                effect=p.wait.effect, accesses=accesses))', 1)),
        ('a wait for a bulk copy without its cursor',
         sub(BUILD, "                             attrs=(('bulk', cursor),) if cursor else ())",
             '                             )', 1)),
        ('a bulk copy in the pipeline of cp.async',
         sub(TRANSFERS, "    if any(x.op is Op.COPY_ASYNC and x.counter == 'bulk'\n"
                        '           for x in walk_stmts(tuple(t.pieces))):\n'
                        '        return BULK\n', '', 1)),
        ('a drain waiting on no cursor',
         sub(EMIT, "                texts += [f'tensorforge::bulk_wait({c});'\n"
                   '                          for c in self._bulk_cursors]\n',
             '                pass\n', 1)),
    ]),

    # What a target is, can and prefers.
    'target': ('tests/test_target.py tests/test_nontemporal.py '
               'tests/test_prefetch.py tests/test_packed_fma.py '
               'tests/test_atomics.py tests/test_sycl_sub_group.py '
               'tests/test_placement.py', [
        ('a hint on a wide CUDA access',
         sub(TARGET, 'return length == 1 and datatype in CUDA_CACHE_HINT_TYPES',
             'return datatype in CUDA_CACHE_HINT_TYPES', 1)),
        ('a HIP rendezvous across waves',
         sub(TARGET, '                return wave % num_threads == 0\n'
                     '            return False\n',
             '                return wave % num_threads == 0\n'
                     '            return True\n', 1)),
        ('the sub-group stated where the device picks it',
         sub(TARGET, "if (self.hw.vendor == 'intel' and self.backend == 'oneapi'):",
             "if self.hw.vendor == 'intel':", 1)),
        ('a prefetch below gfx12',
         sub(TARGET, 'level >= 0x1200', 'level >= 0x1000', 1)),
        ('the paired FFMA on sm_120',
         sub(TARGET, 'and 100 <= level < 120) else 1',
             'and 100 <= level) else 1', 1)),
        ('an atomic under the explicit vector',
         sub(TARGET, '        if self.explicit_simd:\n            return False\n'
                     '        from tensorforge.backend import atomics\n',
             '        from tensorforge.backend import atomics\n', 1)),
        ('an async copy where the device has no path',
         sub(TARGET, '        if not self.async_copy_path():\n            return ()\n',
             '', 1)),
        ('the broadcast in place kept under the explicit vector',
         sub(TARGET, 'broadcast_without_staging=False),',
             'broadcast_without_staging=True),', 1)),
        ('a grid barrier under SYCL',
         sub(TARGET, "SYCL, whose lexic has no `sync_grid`.\"\"\"\n"
                     "        return not self.sycl",
             "SYCL, whose lexic has no `sync_grid`.\"\"\"\n"
                     "        return True", 1)),
    ]),

    # What a descriptor states beyond what the builders take, rewritten.
    'legalize': ('tests/test_legalize.py', [
        ('an accumulation written in place',
         sub(LEGALIZE, '    if not d.add:\n      return d.dest, lambda: []',
             '    if True:\n      return d.dest, lambda: []', 1)),
        ('an operand on other axes read as it is',
         sub(LEGALIZE, '      if axes == list(range(rank)) or',
             '      if True or', 1)),
        ('the destination not cut at an operand\'s edge',
         sub(LEGALIZE, '  if all(list(box.lower()) == lower and list(box.upper()) == upper\n'
                       '         for box in boxes):',
             '  if True:', 1)),
        ('pieces written in place where the destination owes zeros',
         sub(LEGALIZE, '    if dest.owed_zeros() is None:\n      return dest, cells,',
             '    if True:\n      return dest, cells,', 1)),
        ('the factor dropped',
         sub(LEGALIZE, '    if alpha is None:\n      return []',
             '    if True:\n      return []', 1)),
        ('two hoists under one guard version',
         sub(LEGALIZE, '    version = -self._hoisted', '    version = -1', 1)),
        ('a ternary hoisted over the condition it writes',
         sub(LEGALIZE, ' or cond.tensor is d.dest.tensor:', ':', 1)),
        ('a root standing in for a power',
         sub(LEGALIZE, '    if minus_one(y):\n      return _rewritten(d, Operation.RCP, [0])',
             '    if _is_num(y) and y == 0.5:\n'
             '      return _rewritten(d, Operation.SQRT, [0])\n'
             '    if minus_one(y):\n      return _rewritten(d, Operation.RCP, [0])', 1)),
        ('a square left to the power',
         sub(LEGALIZE, '    if _is_num(y) and y == 2:', '    if False:', 1)),
        ('a product by one kept',
         sub(LEGALIZE, '      if one(d.srcs[other]):\n        return _rewritten(d, Operation.COPY',
             '      if False:\n        return _rewritten(d, Operation.COPY', 1)),
        ('an identity that drops the accumulation',
         sub(LEGALIZE, '                         add=d.add_mask(), alpha=d.alpha)',
             '                         alpha=d.alpha)', 1)),
    ]),

    # An elementwise operation as one typed statement, spelled per target.
    'math': ('tests/test_math_op.py', [
        ('a misspelt function in one library',
         sub(LEXIC, "(Operation.LOG1P, 'log1p')",
             "(Operation.LOG1P, 'logp1')", 1)),
        ('a minimum that is another function of a NaN in one library',
         sub(SYCL_LEXIC, "Operation.MIN: 'sycl::fmin(",
             "Operation.MIN: 'sycl::min(", 1)),
        ('an integer through the floating-point function',
         sub(LEXIC, '    if fptype in _INTEGERS:\n', '    if False:\n', 1)),
        ('the absolute value of an unsigned integer asked of `abs`',
         sub(LEXIC, '    if op == Operation.ABS and fptype in _UNSIGNED:\n',
             '    if False:\n', 1)),
        ('spelled for another type than the result',
         sub(EMIT, 'return lex.get_operation(fn, s.target[0].type.base, *args,',
             'return lex.get_operation(fn, Datatype.F64, *args,', 1)),
        ('the operation left off the statement',
         sub(BUILD, "self._emit_op(Op.MATH, (v,), args, attrs=(('fn', fn),))",
             "self._emit_op(Op.MATH, (v,), args)", 1)),
        ('two operations on one operand taken for one value',
         sub(BUILD, 'key = _cons_key(Op.MATH, type_, args, fn)',
             'key = _cons_key(Op.MATH, type_, args)', 1)),
    ]),

    # The transfer for the next element, issued across the back edge.
    'wrap': ('tests/test_pir_wrap.py tests/test_pass_pipeline.py', [
        ('the tail fetches the element the iteration computes',
         sub(WRAP, '        tail_copy.given(l.k, l.next)',
             '        tail_copy.given(l.k, l.k)', 1)),
        ('the distance ignored',
         sub(WRAP, '            if count.get(key, 0) < distance:',
             '            if count.get(key, 0) < 1:', 1)),
        ('a buffer another statement writes moved all the same',
         sub(WRAP, "            if any(writes(a) and same(a.base, d) for a in x.accesses):\n"
                   "                raise Refusal('the buffer is written by more than this '",
             "            if False:\n"
             "                raise Refusal('the buffer is written by more than this '", 1)),
        ('a shared transfer moved across a barrier ahead of it',
         sub(WRAP, '            if t.shared and (x.op is Op.BARRIER or x.effect & Effect.BARRIER):',
             '            if False:', 1)),
        ('a shared tail ahead of the barrier that fences it',
         sub(WRAP, '            rest[closing + 1:closing + 1] = tails_shared',
             '            rest[closing:closing] = tails_shared', 1)),
        ('the masked path leaves the copy it skipped the wait of in flight',
         sub(WRAP, '            if waits and not lift:\n                # A masked element skips',
             '            if False:\n                # A masked element skips', 1)),
        ('an element\'s own pointer followed under no flag',
         sub(WRAP, '            if p.owns_pointer and peel_flag is not None:',
             '            if False:', 1)),
        ('a loop rewritten without a cloned dependency handed back unchanged',
         sub(WRAP, '    if not rewritten:\n        return given',
             '    if not copied:\n        return given', 1)),
        ('a second stage taken from the element',
         sub(WRAP, "            following = b.op('bitxor', INDEX, stage, 1, hint='stageNext')",
             "            following = b.op('bitxor', INDEX, l.k, 1, hint='stageNext')", 1)),
        ('the stage never advanced',
         sub(WRAP, '            yielded.append(following)',
             '            yielded.append(carried[-1])', 1)),
        ('a copy into the second stage left at the tail',
         sub(WRAP, '                heads += tail\n            elif l.early and l.guard_at is None:',
             '                tails_shared.extend(tail)\n            elif l.early and l.guard_at is None:', 1)),
        ('the masked path drains behind a copy at the head',
         sub(WRAP, '            if waits and not lift:',
             '            if waits:', 1)),
        ('the stage read declared dead where the other is filled',
         sub(WRAP, "                tail = [x for x in tail if not (\n"
                   "                    x.op is Op.MARK and x.attr('mark') == 'defines')]",
             "                tail = list(tail)", 1)),
        ('loads and stores given a second stage',
         sub(WRAP, "        if not tokens:\n            return ('its copies are loads and stores",
             "        if False:\n            return ('its copies are loads and stores", 1)),
        ('the stage of a window not spelled',
         sub(EMIT, '                    off = off.format(*(self.operand(a) for a in s.args))',
             '                    pass', 1)),
        ('a counted loop left as it is',
         sub(WRAP, "        if counted(s) and (kind == 'run' or transfers(s.regions[0].body)):",
             "        if False:", 1)),
        ('the trip after the last one fetched',
         sub(WRAP, "    following = b.op('select', counter.type, inside, ahead, counter,\n"
                   "                     hint='trip')",
             "    following = ahead", 1)),
        ('a counted loop wrapped again as a batch loop',
         sub(WRAP, "                or s.attr('first') is None or id(s) in done):",
             "                or s.attr('first') is None):", 1)),
        ('a value its consumer spells twice inlined',
         sub(EMIT, '        out = [(a, times.get(i, 1)) for i, a in enumerate(s.args)',
             '        out = [(a, 1) for i, a in enumerate(s.args)', 1)),
    ]),

    # Any loop whose trips are counted, and not only a merged run's: moved in
    # by `move` and `wrap` alike, the innermost first, a transfer's own hops
    # left alone.
    'counted': ('tests/test_counted_loops.py '
                'tests/test_pir_move.py::test_any_counted_loop_loses_what_every_trip_does_alike '
                'tests/test_pir_move.py::test_a_loop_that_may_not_run_keeps_its_transfers', [
        ('a counted loop no run was rolled into passed over',
         sub(AHEAD, "    if s.op is not Op.FOR or not s.regions or s.attr('next') is not None:",
             "    if s.op is not Op.FOR or not s.regions or s.attr('next') is not None or not s.attr('rolled'):", 1)),
        ('a loop that may not run taken for a counted one',
         sub(AHEAD, '    return (all(isinstance(x, int) for x in (lo, hi, step)) and step > 0\n'
                    '            and lo < hi)',
             '    return (all(isinstance(x, int) for x in (lo, hi, step)) and step > 0)', 1)),
        ('a loop over bounds known at run time taken for a counted one',
         sub(AHEAD, '    return (all(isinstance(x, int) for x in (lo, hi, step)) and step > 0\n'
                    '            and lo < hi)',
             '    return (isinstance(step, int) and step > 0\n'
             '            and (not all(isinstance(x, int) for x in (lo, hi)) or lo < hi))', 1)),
        ("a transfer's own hops wrapped",
         sub(WRAP, "            if id(s) in pieces:\n                # A transfer's own loop",
             "            if False:\n                # A transfer's own loop", 1)),
        ('the outer loop first',
         sub(WRAP, "        regions = []\n        for r in s.regions:\n            inner, changed = _runs(",
             "        regions = []\n        for r in (() if counted(s) else s.regions):\n            inner, changed = _runs(", 1)),
        ('a reduction without a transfer reported',
         sub(WRAP, "        if counted(s) and (kind == 'run' or transfers(s.regions[0].body)):",
             "        if counted(s):", 1)),
        ('the loop itself taken for what stands around it',
         sub(WRAP, '        around = tuple(x for x in body if x is not s)\n        regions = []',
             '        around = tuple(body)\n        regions = []', 1)),
    ]),

    # A loop over groups of rows, one group ahead: from the group's index
    # one stride on, each row's element bound again from it, the first group
    # named by the loop and clamped into the batch.
    'grouped': ('tests/test_group_wrap.py '
                'tests/test_wrap_loads.py::test_a_loop_over_groups_of_rows_is_wrapped_a_group_ahead', [
        ("the element one stride on taken for the group's index",
         sub(WRAP, "            self.next_stmts, self.next = self.successor(self.k, 'groupNext')",
             "            self.next = loop.attr('next')", 1)),
        ('the element the group binds kept for the next group',
         sub(WRAP, '            self.element = self.k\n            self.flag_word = None\n',
             '            self.flag_word = None\n', 1)),
        ("a group's pointer followed under the flag of the group's index",
         sub(WRAP, '            self.element = self.k\n            self.flag_word = None\n',
             '            self.element = self.k\n', 1)),
        ('a loop over groups of rows naming no first group',
         sub(Path('src/tensorforge/backend/instructions/batch_loop.py'),
             '                         peel_index=self._first_group(writer, start),\n',
             '', 1)),
        ('the first group taken where it is past the end',
         sub(Path('src/tensorforge/backend/instructions/batch_loop.py'),
             "        first = writer.op('select', SIZE, inside, begin, 0,\n"
             "                          hint=f'{self._group_batch()}First')",
             '        first = begin', 1)),
    ]),

    # An element's own pointer a copy at the head follows, read an iteration
    # earlier and carried: the loop starting with the second element's, the
    # head reading the one after next, the copy following the carried one --
    # and only there.
    'pointerstage': ('tests/test_pointer_stage.py '
                     'tests/test_pir_wrap.py::test_a_pointer_the_copy_at_the_head_follows_rides_the_loop '
                     'tests/test_pir_wrap.py::test_a_pointer_nothing_follows_at_the_head_has_no_stage', [
        ("the loop started with the first element's pointer",
         sub(WRAP, '        second_copy.given(l.k, second)',
             '        second_copy.given(l.k, l.first)', 1)),
        ("the next element's pointer read for the one after it",
         sub(WRAP, '        ahead_copy.given(l.k, ahead)',
             '        ahead_copy.given(l.k, l.next)', 1)),
        ('the copy following the pointer read for this element',
         sub(WRAP, '            tail_copy.given(old, held)\n', '', 1)),
        ('a pointer staged for a transfer behind the head',
         sub(WRAP, '                if p.rotate and p.owns_pointer:\n                    bindings',
             '                if p.owns_pointer:\n                    bindings', 1)),
        ('a table of pointers read as one array',
         sub(ORACLE, "        env.setdefault(name, PtrTable(mem, name))\n", '', 1)),
        ('the option not reaching the wrap',
         sub(ABSTR, '                           pointer_stage=options.pointer_stage,',
             '                           pointer_stage=False,', 1)),
    ]),

    # The launch-control traversal reading the next element off the queue
    # at the top: the transfer for it, the peel ahead of the loop, one guard
    # for the size and the mask, the copies it cannot carry left where they
    # are -- and the host oracle's queue and hand-off.
    'lookahead': ('tests/test_launch_control.py', [
        ('the transfer moved ahead for this element again',
         sub(Path('src/tensorforge/backend/instructions/batch_loop.py'),
             "        loop._next_index = builder.op('select', SIZE, valid, element, held,",
             "        loop._next_index = builder.op('select', SIZE, valid, held, held,", 1)),
        # Not the first element bound inside the loop's frame instead: under
        # the generator the section binds it ahead of the loop, and both
        # frames answer with that binding (`BatchLoop._prologue_element`).
        ('the queue loop naming no first element for the peel',
         sub(Path('src/tensorforge/backend/instructions/batch_loop.py'),
             '        peel = self._peel_index(builder) if lookahead else None\n',
             '        peel = None\n', 1)),
        ('the size left out of the element guard',
         sub(Path('src/tensorforge/backend/instructions/batch_loop.py'),
             "                'const bool allowed', f'{{0}} < {self._num_elements()} && '\n"
             "                f'{read}', BOOL, None, args=(loop.induction,),",
             "                'const bool allowed', f'{read}', BOOL, None,\n"
             "                args=(loop.induction,),", 1)),
        ('a copy the queried loop cannot carry moved ahead',
         sub(WRAP, '        if tokens and not self.strided:\n',
             '        if False:\n', 1)),
        ('the flags made to ride a queried loop',
         sub(WRAP, '        carry = (l.flag_word is not None and l.guard_at is not None\n'
                   '                 and l.strided)',
             '        carry = (l.flag_word is not None and l.guard_at is not None)', 1)),
        ('a queried loop with the wrap refused at depth two',
         sub(GEN, '          and self._launch_control_depth < 2):',
             '          and self._launch_control_depth < 3):', 1)),
        ("the oracle's hand-off no barrier",
         sub(ORACLE, "        r'|^cooperative_groups::this_grid\\(\\)\\.sync\\('\n"
                     "        r'|.*\\blaunchCursor\\d+\\.next\\s*\\(')",
             "        r'|^cooperative_groups::this_grid\\(\\)\\.sync\\(')", 1)),
        ("the oracle's queue answering the element after next",
         sub(ORACLE, '        return self.launches[self.at] if self.at < len(self.launches) else -1',
             '        return (self.launches[self.at + 1]\n'
             '                if self.at + 1 < len(self.launches) else -1)', 1)),
    ]),

    # Whom a barrier meets, in the host oracle: a warp's lanes, the warps of
    # one named barrier, or the block -- and everyone it waits for arriving.
    'barrierreach': ('tests/test_races.py tests/test_rank0_operands.py', [
        ('a warp barrier meeting every lane that executes it',
         sub(ORACLE, "        if reach[0] == 'block':\n"
                     "            return [(lanes, self.interps)]\n",
             "        if reach[0] in ('block', 'warp'):\n"
             "            return [(lanes, self.interps)]\n", 1)),
        ('a lane that skips a barrier never missed',
         sub(ORACLE, '                if missing:\n'
                     '                    tracker.arrival(missing[0], len(present))',
             '                if False:\n'
             '                    tracker.arrival(missing[0], len(present))', 1)),
        ('a count the warps do not fill never noticed',
         sub(ORACLE, '                if counted != len(warps) * warp:',
             '                if False:', 1)),
        ("a warp barrier's mask read as the whole warp",
         sub(ORACLE, "            mask_of = (lambda it: int(it.ev(reach[1])) & 0xffffffff) \\\n"
                     "                if reach[1] else (lambda it: 0xffffffff)",
             "            mask_of = (lambda it: 0xffffffff)", 1)),
        ('an operand passed by value read as a pointer',
         sub(ORACLE, '        env.setdefault(name, mem.read(name, 0))',
             '        pass', 1)),
    ]),

    # A block holds no more multiplications than it has barriers for, and
    # the barrier behind a bulk copy's arming is the multiplication's own.
    'barrierbudget': ('tests/test_barrier_scope.py tests/test_stage_members.py', [
        ('no limit on the named barriers',
         sub(TARGET, '            return self.lexic.NAMED_BARRIERS - 1\n',
             '            return None\n', 1)),
        ('the stage group past the named barriers',
         sub(GEN, '    if limit is not None and group > limit:\n',
             '    if False:\n', 1)),
        ("the bulk copy's barrier a multiplication's at any width",
         sub(LOAD, '        participants=SyncThreads(self._context,\n'
                   '                                 self._num_threads).participants(),\n',
             '', 1)),
        ('a barrier of one multiplication spelled where it has none',
         sub(EMIT, '            if sync_mult is not None and not sync_mult(threads):',
             '            if False:', 1)),
        ('the finished body not weighed against the block',
         sub(GEN, '    self._check_barriers(body, index)\n', '', 1)),
        ('a block of any size taken for one the target meets apart',
         sub(PASSES, '    if separable is None or mults_per_block <= separable:\n'
                     '        return []\n',
             '    return []\n', 1)),
        ("a block barrier taken for a multiplication's",
         sub(PASSES, "        if s.op == Op.BARRIER and s.attr('participants') == Participants.MULT:",
             '        if s.op == Op.BARRIER:', 1)),
        ('a barrier inside a loop not seen',
         sub(PASSES, '    for s in walk_stmts(body):\n'
                     "        if s.op == Op.BARRIER and s.attr('participants')",
             '    for s in body:\n'
             "        if s.op == Op.BARRIER and s.attr('participants')", 1)),
    ]),

    # Groups that change places across a loop's back edge are no steady
    # state.
    'swap': ('tests/test_loop_carried.py', [
        ('a steady state told by how many groups are in flight',
         sub(ASYNCMEM, '                and _carried(trial, s).same(entry))',
             '                and len(trial.outstanding) == len(entry.outstanding))', 1)),
    ]),

    # A carried copy retired early by a younger wait leaves the loop's
    # pipeline steady.
    'steady': ('tests/test_loop_carried.py', [
        ('an iteration that leaves a carried group retired out of balance',
         sub(ASYNCMEM, '               and _settles(trial, entry, s))',
             '               and False)', 1)),
        ('the shorter state carried on behind the loop',
         sub(ASYNCMEM, '    if settled:\n        # The longer of the two states',
             '    if False:\n        # The longer of the two states', 1)),
    ]),

    # What a trip of a counted loop reads in place, read into registers a
    # trip ahead.
    'trips': ('tests/test_trip_reads.py', [
        ('a loop unrolled whole gathered',
         sub(TRIPS, "        if counted(s) and s.attr('unroll') is not True:",
             '        if counted(s):', 1)),
        ('a load all lanes read alike taken into a register array',
         sub(TRIPS, '        if value.uniformity is not Uniformity.LANE:',
             '        if False:', 1)),
        ('a load from a buffer the loop writes taken',
         sub(TRIPS, '        if unknown or any(_key(a.base) in written for a in x.accesses):',
             '        if False:', 1)),
        ('an index the loop loads computed again for the next trip',
         sub(TRIPS, '        if d is None or not glue(d):\n            return None',
             '        if d is None:\n            return None', 1)),
        ('the slots read where the loads stood',
         subs(TRIPS,
              ('    declare: List[Stmt] = []\n',
               '    declare: List[Stmt] = []\n    placed = {}\n'),
              ('            head.extend(built[:-1])\n'
               '            head.append(replace(built[-1], target=(x.target[0],)))\n',
               '            placed[id(x)] = tuple(built[:-1]) + (\n'
               '                replace(built[-1], target=(x.target[0],)),)\n'),
              ('    rest = [x for x in stmts if id(x) not in moved]\n',
               '    rest = [y for x in stmts for y in placed.get(id(x), ())\n'
               '            or ((x,) if id(x) not in moved else ())]\n'))),
        ('the reads gathered without the wrap that fills them',
         sub(PIPE, '        if self._trip_reads and not pc.explicit_simd:',
             '        if False:', 1)),
        ('a trip gathered however many loads it reads',
         sub(TRIPS, '    if len(found) > TRIP_SLOTS:', '    if False:', 1)),
    ]),

    # What a trip of a counted loop reads in place, copied into the
    # multiplication's shared memory a shard a trip.
    'tripshards': ('tests/test_trip_shards.py', [
        # Not the target's gate in `Generator._trip_spec` (asynchronous
        # staging and a copy path): where it says no, a later check does as
        # well -- sm_70 has no copy size, and gfx942 builds no shard with the
        # gate or without, at 32 and 64 lanes and with the operators read in
        # place -- so taking it out changes no build there is to test.
        ('a predicate the lanes decide left out of the span',
         sub(TRIPS, "            if reach <= {'x', counter.id}:",
             '            if False:', 1)),
        ('a run that does not move by a step taken',
         sub(TRIPS, '        if lo != first + k * step or hi - lo + 1 != count:',
             '        if False:', 1)),
        ('the lanes past the run copying past it',
         sub(TRIPS, "            within = (copy.op('min', INDEX, at, count - 1, hint='hop')\n"
                    "                      if (hop + 1) * threads > count else at)",
             '            within = at', 1)),
        ('the shards taken behind the register array',
         subs(PIPE,
              ('        if self._trip_shards is not None and not pc.explicit_simd:\n'
               '            pc.body = trip_shards(pc.body, self._scratch, report=self._report,\n'
               '                                  **self._trip_shards)\n', ''),
              ('            pc.body = trip_reads(pc.body, self._scratch, report=self._report)\n',
               '            pc.body = trip_reads(pc.body, self._scratch, report=self._report)\n'
               '        if self._trip_shards is not None and not pc.explicit_simd:\n'
               '            pc.body = trip_shards(pc.body, self._scratch, report=self._report,\n'
               '                                  **self._trip_shards)\n'))),
    ]),

    # Several consecutive elements of the batch in one multiplication.
    'stack': ('tests/test_stack_elements.py', [
        ('the stacks counted rounding down',
         sub(STACKING, "  return f'(({name} + {k - 1}) / {k})'",
             "  return f'({name} / {k})'", 1)),
        ('an element one past the batch taken for one in it',
         sub(BATCHLOOP, "            cond = f'{element} < {n}'",
             "            cond = f'{element} <= {n}'", 1)),
        ('the flag of an element of the stack left unread',
         sub(BATCHLOOP, "                cond = f'{cond} && {read}'\n"
                        "                kind, space = Effect.READ, MemSpace.GLOBAL\n"
                        "            out.append(",
             "                kind, space = Effect.READ, MemSpace.GLOBAL\n"
             "            out.append(", 1)),
        ('a stack visited for its first element alone',
         sub(BATCHLOOP, "            either = ' || '.join(f'{{{e}}}' for e in range(self._stack))",
             "            either = '{0}'", 1)),
        ('every element of the stack through the first one\'s pointer',
         sub(PTR, "      element = f'({{{1 + e}}} ? {k} * {{0}} + {e} : ({first}))'",
             "      element = f'({k} * {{0}})'", 1)),
        ('the stack\'s index added to the address instead of picking the pointer',
         sub(SYM, "      if self.stack() > 1:\n        # The stack's index picks",
             "      if False:\n        # The stack's index picks", 1)),
        ('every write held to the first element\'s mask',
         sub(SYM, "    if isinstance(var, (int, np.integer)):\n"
                  "      return names[int(var) - lower]",
             "    if True:\n      return names[0]", 1)),
        ('a stacked operand copied into shared memory as one run',
         sub(LOAD, "      k = self._src.stack()\n      if k > 1:",
             "      k = self._src.stack()\n      if False:", 1)),
        ('a merged run\'s members read through one element\'s pointers',
         sub(PTR, "        args.append(pointers[_e])", "        args.append(pointers[0])", 1)),
        ('the grid sized by the elements',
         sub(GEN, "    stacks = stacking.count(0, self._stack)\n",
             "    stacks = stacking.count(0, 1)\n", 1)),
        ('the first stack clamped by the element count',
         sub(GEN, "    count = body.extern_value(stacking.count(index, self._stack), SIZE,",
             "    count = body.extern_value(stacking.count(index, 1), SIZE,", 1)),
        ('an operand the batch shares read by a pointwise stack as if stacked',
         sub(STACKING, "        target.append(list(range(rank)))\n        broadcast = True",
             "        target.append(list(range(rank + 1)))\n        broadcast = True", 1)),
        ('the stack reduced with the axes a reduction names',
         sub(STACKING, "    return ReductionDescr(self.view(d.dest), self.view(d.var), list(d.dims),",
             "    return ReductionDescr(self.view(d.dest), self.view(d.var),\n"
             "                          list(d.dims) + [d.var.bbox.rank()],", 1)),
    ]),

    # The stack and the width of the matrix instruction as the tuner's knobs,
    # and the bound per element they are weighed by.
    'tune': ('tests/test_autotune.py tests/test_matrix_columns.py '
             'tests/test_stack_elements.py', [
        ('a stacked build\'s bound left per stack',
         sub(BOUND, "    if b is None or stack == 1:\n        return b",
             "    if True:\n        return b", 1)),
        ('the rounds of a persistent grid counted in elements',
         sub(TUNING, "            stacks = -(-self.batch // _stack_of(gen))",
             "            stacks = self.batch", 1)),
        ('an asked width left to the count',
         sub(TILING, "        if asked:\n            fits = (",
             "        if False:\n            fits = (", 1)),
        ('the stack not offered',
         sub(TUNING, "            knobs.append(Knob('stack_elements', lambda c: (2,)))",
             "            pass", 1)),
        ('the width not offered',
         sub(TUNING, "            knobs.append(Knob('matrix_columns', widths))",
             "            pass", 1)),
        ('the stack and the width offered to the static scorer',
         sub(TUNING, "    if mode == 'compiled':\n        if context.target.prefs.tune_stack",
             "    if True:\n        if context.target.prefs.tune_stack", 1)),
        ('the space built for the scorer asked for, not the one that ranks',
         sub(TUNING, "    knobs = simple_space(descrs, context, mode=mode)",
             "    knobs = simple_space(descrs, context)", 1)),
        ('the residency counted in multiplications',
         sub(TUNING, "    return (over > 0, issue, -resident * _stack_of(gen),",
             "    return (over > 0, issue, -resident,", 1)),
        ('a list that does not stack offered the stack',
         sub(TUNING, "    except GenerationError:\n        return False\n    return True",
             "    except GenerationError:\n        return True\n    return True", 1)),
        ('the candidates built from the stacked list',
         sub(GEN, "    given = self._unstacked\n", "    given = self._given\n", 1)),
        ('an asked stack turned anyway',
         sub(GEN, "        fixed[name] = getattr(opts, name)", "        pass", 1)),
        ('a knob without values ending the enumeration',
         sub(TUNING, "        for v in knobs[i].values(cand) or (None,):",
             "        for v in knobs[i].values(cand):", 1)),
    ]),

    # Barrier placement keeps what it found for a statement and a state; a
    # memo keyed by less than both answers for another one.
    'barrierwalks': ('tests/test_pir_barriers.py tests/test_ternary.py', [
        ('a construct placed once whatever the state it arrives with',
         sub(BARRIERS, '        key = (id(s), state.frozen(), level)\n',
             '        key = (id(s), level)\n', 1)),
        ('the hazards on the way into a construct found once whatever the state',
         sub(BARRIERS, '        key = (id(s), state.frozen())\n',
             '        key = id(s)\n', 1)),
        ("one statement's accesses taken for every other's",
         sub(BARRIERS, '        hit = self._accesses.get(id(s))\n',
             '        hit = next(iter(self._accesses.values()), None)\n', 1)),
        ('an access no barrier reaches counted at every pass',
         sub(BARRIERS, '                        unfenced.add(id(inner))\n',
             '                        unfenced.add(len(unfenced))\n', 1)),
    ]),

    # The row of the block read once as the wave's, where every
    # multiplication is a whole number of waves, and the element pointers
    # with it; the compile reads the first lane only of its own whole waves.
    'waveuniform': ('tests/test_wave_uniform.py', [
        ('the row read as the wave\'s where multiplications share it',
         sub(GEN, '    return (layout.whole_waves\n',
             '    return ((layout.whole_waves or layout.contiguous)\n', 1)),
        ('a target without readfirstlane asked for it',
         sub(TARGET, "        return self.hw.vendor == 'amd' and self.backend == 'hip'",
             '        return True', 1)),
        ('the switch ignored',
         sub(GEN, '            and self._context.get_user_options().wave_uniform_rows)',
             '            and True)', 1)),
        ('the row declared behind the arena that addresses by it',
         subs(GEN, ('    self._declare_wave_row(builder)\n', ''),
              ('    return tuple(builder.finish()) + tuple(body)',
               '    self._declare_wave_row(builder)\n'
               '    return tuple(builder.finish()) + tuple(body)'))),
        ('the row declared with the wave\'s lanes',
         sub(GEN, "lexic.first_lane(self._row_source, self._num_threads, 'unsigned')",
             "lexic.first_lane(self._row_source, 64, 'unsigned')", 1)),
        ('the block-wide loader numbering threads by the hardware row',
         sub(GEN, '        lexic.thread_idx_y = lexic.raw_thread_idx_y = MULT_NAME\n',
             '        lexic.thread_idx_y = MULT_NAME\n', 1)),
        ('the element pointers left per lane',
         sub(PTR, '    if not lanes:\n      return address\n',
             '    return address\n', 1)),
        ('the row spelled without its type',
         sub(HIP_LEXIC, "    targs = f'{lanes}, {ctype}' if ctype else f'{lanes}'",
             "    targs = f'{lanes}'", 1)),
        ('the first lane read whatever the wave',
         sub(HIP_WAVE, '  if constexpr (Lanes % WaveSize != 0) {',
             '  if constexpr (false) {', 1)),
        ('gfx10 and later taken for wave64',
         sub(HIP_WAVE, 'inline constexpr int WaveSize = 32;',
             'inline constexpr int WaveSize = 64;', 1)),
        ('a compile for another width let through',
         sub(HIP_WAVE, 'static_assert(__AMDGCN_WAVEFRONT_SIZE == WaveSize,',
             'static_assert(true || __AMDGCN_WAVEFRONT_SIZE == WaveSize,', 1)),
        ('the oracle reading the call as an unknown one',
         sub(ORACLE, '    e = _wave_reads(e)\n', '', 1)),
    ]),

    # The collectives of `hip.h` by the width of the wave.  The tests compile
    # the header for the device and need a HIP installation's headers
    # (`TF_HIP_INCLUDE`); without them they skip, and every mutation here
    # reads as missed.
    'wavewidth': ('tests/test_wave_width.py', [
        ('readlane for 64 lanes whatever the wave',
         sub(HIP, '  } else if constexpr (Block == WaveSize && Subblock == 1) {\n'
                  '    // One lane for the whole wave',
             '  } else if constexpr (Block == 64 && Subblock == 1) {\n'
             '    // One lane for the whole wave', 1)),
        ('readlane for a block narrower than the wave',
         sub(HIP, '  } else if constexpr (Block == WaveSize && Subblock == 1) {\n'
                  '    // One lane for the whole wave',
             '  } else if constexpr (Subblock == 1 && Block >= 32) {\n'
             '    // One lane for the whole wave', 1)),
        ('a broadcast past the wave let through',
         sub(HIP, 'static_assert(Block <= WaveSize, "a broadcast reaches',
             'static_assert(true || Block <= WaveSize, "a broadcast reaches', 1)),
        ('a swap past the wave let through',
         sub(HIP, 'static_assert(Block <= WaveSize, "a swap reaches',
             'static_assert(true || Block <= WaveSize, "a swap reaches', 1)),
        ('the ballot bit counted from threadIdx.x',
         sub(HIP, '  const auto lane = static_cast<unsigned>(__lane_id());',
             '  const auto lane = static_cast<unsigned>(threadIdx.x);', 1)),
        ('the butterfly reached by a boolean too',
         sub(HIP, '    return ballotReduction<Op, Block, Subblock>(value);\n'
                  '  } else {\n',
             '    return ballotReduction<Op, Block, Subblock>(value);\n'
             '  }\n  {\n', 1)),
    ]),

    # The arrays of pointers and the flags promised unwritten and unaliased
    # in the kernel's prototype, the launcher's left as the interface has it.
    'restrictarrays': ('tests/test_restrict_arrays.py', [
        ('the arrays of pointers left unpromised',
         sub(GEN, '      if (symbol.obj.addressing == Addressing.PTR_BASED\n',
             '      if (False\n', 1)),
        ('every pointer promised, the batch\'s own operand too',
         sub(GEN, '      if (symbol.obj.addressing == Addressing.PTR_BASED\n',
             '      if (symbol.obj.addressing != Addressing.SCALAR\n', 1)),
        ('the flags left unpromised',
         sub(GEN, '            promise=self._promise_arrays()))',
             '            promise=False))', 1)),
        ('SYCL promised as well',
         sub(GEN, "            and self._context.target.backend in ('cuda', 'hip'))",
             '            and True)', 1)),
        ('the launcher declaring the kernel\'s types',
         sub(KPARAMS, '        elif host:\n', '        elif False:\n', 1)),
        ('the kernel called with the caller\'s pointers',
         sub(KPARAMS, "        return (self.name if self._kernel_type(lexic) is None\n"
             "                else f'{self.name}Arg')",
             '        return self.name', 1)),
        ('the levels between left writable',
         sub(LEXIC, "    stars = ('*const ' * (depth - 1) + '*') if inner_const else '*' * depth",
             "    stars = '*' * depth", 1)),
        ('the promise without restrict',
         sub(KPARAMS, '    def _kernel_pointer(self, lexic, restrict: bool = True)',
             '    def _kernel_pointer(self, lexic, restrict: bool = False)', 1)),
        ('the promise asked for rather than made',
         sub(OPTIONS, "declare('restrict_batch_arrays',\n        default=True,",
             "declare('restrict_batch_arrays',\n        default=False,", 1)),
    ]),

    # The multiplications of a wave driven through the batch loop together
    # where only the caller asks for it.
    'wavegroups': ('tests/test_wave_group_traversal.py', [
        ('the option read and ignored',
         sub(GEN, '                                     or not self._mult_met_alone() or asked):',
             '                                     or not self._mult_met_alone()):', 1)),
        ('every shared wave grouped, asked or not',
         sub(GEN, '    asked = self._context.get_user_options().wave_group_traversal\n',
             '    asked = True\n', 1)),
        ('a rotated start grouped as well',
         sub(GEN, '      # per row.\n      if start != self._get_2d_block_id():\n        return 1\n',
             '      # per row.\n', 1)),
        ('the group asked for by default',
         sub(OPTIONS, "declare('wave_group_traversal',\n        default=False,",
             "declare('wave_group_traversal',\n        default=True,", 1)),
        ('a held back row writing its group\'s element',
         sub(BATCHLOOP, '                    with elementmask.element_mask(active, self._active()), \\\n',
             '                    with elementmask.element_mask(None, None), \\\n', 1)),
        ('the oracle stopped by the schedule fence',
         sub(ORACLE, "        if re.match(r'^__builtin_amdgcn_sched_barrier\\s*\\(', stmt):\n"
                     "            return                      # a fence for the compiler's schedule\n",
             '', 1)),
    ]),

    # Each multiplication's row of a group read as the wave's, with its flag
    # word and its element pointers, and a lane's own picked.
    'grouprows': ('tests/test_wave_uniform_group_rows.py tests/test_pir_rows.py', [
        ('the rows the lanes\' own though asked for',
         sub(GEN, '    if group < 2 or group * self._num_threads != wave:\n'
                  '      return 0\n    return wave',
             '    return 0', 1)),
        ('a stack of elements read per multiplication',
         sub(BATCHLOOP, '        wave = self._wave_rows if self._stack == 1 else 0\n',
             '        wave = self._wave_rows\n', 1)),
        ('the place left unmarked',
         sub(BUILD, "        attrs = ((('escapes', True),) if escapes else ()) + tuple(attrs)\n",
             "        attrs = (('escapes', True),) if escapes else ()\n", 1)),
        ('the loop counted per lane',
         sub(BATCHLOOP, "            start = lexic.first_lane(start, wave, 'unsigned')\n",
             '            pass\n', 1)),
        ('the flag word read per lane',
         sub(BATCHLOOP, '                                             if wave and kind is Effect.READ\n',
             '                                             if False\n', 1)),
        ('the element pointers read per lane',
         sub(PTR, "      per_mult = (owns and getattr(self._context.target.lexic,\n"
                  "                                   'rows_per_mult', False))\n",
             '      per_mult = False\n', 1)),
        ('a copy that keeps the lane\'s place',
         sub(ROWS, '        clone.given(value, mult)\n', '', 1)),
        ('a row through what cannot be computed again copied all the same',
         sub(ROWS, '        if len(clean) != len(reached):\n            continue\n',
             '', 1)),
        ('every lane taking the first multiplication\'s copy',
         sub(ROWS, "                text=pick('{0}', [f'{{{m + 1}}}' for m in range(count)]),\n",
             "                text='{1}',\n", 1)),
        ('what a row reads copied again for each binding',
         sub(ROWS, '                todo = [x for x in reads + [s]\n'
                   '                        if x.target[0].id not in clone.mapping]\n',
             '                todo = reads + [s]\n', 1)),
        ('the first multiplication\'s row left at its place',
         sub(PIPE, "        for name, fn in (('rows-fold', fold), ('rows-dce', dce)):\n",
             "        for name, fn in (('rows-dce', dce),):\n", 1)),
        ('the lane\'s element left behind',
         sub(PIPE, "        for name, fn in (('rows-fold', fold), ('rows-dce', dce)):\n",
             "        for name, fn in (('rows-fold', fold),):\n", 1)),
        ('the oracle taking a lane\'s own copy',
         sub(ORACLE, '        run = self.lockstep\n        if run is None:\n',
             '        run = self.lockstep\n        if True:\n', 1)),
        ('the oracle picking the first value',
         sub(ORACLE, '    return values[int(mult)]\n', '    return values[0]\n', 1)),
        ('the header picking the value after',
         sub(HIP_WAVE, '((out = (mult == ++g) ? static_cast<T>(rest) : out), ...);',
             '((out = (mult == g++) ? static_cast<T>(rest) : out), ...);', 1)),
        ('a pick of any length',
         sub(HIP_WAVE, 'static_assert(sizeof...(rest) + 1 == G, "one value per multiplication");',
             'static_assert(true, "one value per multiplication");', 1)),
        ('the pick spelled for two',
         sub(HIP_LEXIC, "return f'tensorforge::pick<{len(values)}>(",
             "return f'tensorforge::pick<2>(", 1)),
        ('the rows read per multiplication by default',
         sub(OPTIONS, "declare('wave_uniform_group_rows',\n        default=False,",
             "declare('wave_uniform_group_rows',\n        default=True,", 1)),
    ]),

    # The cache hints for the element the loop reaches next.
    'prefetch': ('tests/test_pir_prefetch.py tests/test_prefetch_batch.py '
                 'tests/test_prefetch_data.py', [
        ('the entry asked for at the element\'s own index',
         sub(PREFETCH, '            b.prefetch(base, self.l.next, level=self.level)',
             '            b.prefetch(base, self.l.element, level=self.level)', 1)),
        ('an entry asked for once per binding reading it',
         sub(PREFETCH, '            if id(base) in seen:\n                continue\n',
             '', 1)),
        ('the data asked for of the element itself',
         sub(PREFETCH, '        clone.given(self.l.element, self.l.next)',
             '        clone.given(self.l.element, self.l.element)', 1)),
        ('what binds the element copied for its successor',
         sub(AHEAD, '        inner = {self.element.id}', '        inner = set()', 1)),
        ('a hint per element rather than per line',
         sub(PREFETCH, '        per = max(1, self.line_bytes // elem)',
             '        per = 1', 1)),
        ('an element\'s own pointer followed under no flag',
         sub(PREFETCH, '        if tails and owns and l.flag_word is not None:',
             '        if False:', 1)),
        ('the flag named as the wrap names its own',
         sub(PREFETCH, "'allowed_hint'", "'allowed_next'", 1)),
        ('the tail behind the barrier closing the iteration',
         sub(PREFETCH, '            body[closing:closing] = tails',
             '            body[closing + 1:closing + 1] = tails', 1)),
        ('the head ahead of what it reads',
         sub(PREFETCH, '        at = behind_defs(body, head)', '        at = 0', 1)),
        ('a compressed operand asked for over its view',
         sub(Path('src/tensorforge/backend/instructions/ptr_manip.py'),
             "        or not getattr(obj, 'is_dense', lambda: True)()):",
             "        or False):", 1)),
    ]),

    'equiv': ('tests/test_access_equiv.py', [
        ('renaming collapses distinct names onto one',
         sub(EQUIV, "renames.setdefault(m.group(0), f'N{len(renames)}_{m.group(1)}')",
             "renames.setdefault(m.group(0), 'N')")),
        ('the base name is not compared',
         sub(EQUIV, "out[(canon_names(base),", "out[(('any'),")),
        ('multiplicity dropped: a set, not a multiset',
         sub(EQUIV, "                 canon_names(_canon_expr(_expand(index, defs))))] += 1",
             "                 canon_names(_canon_expr(_expand(index, defs))))] = 1")),
        # A loop header is kept out of `defs` by two independent things: it
        # does not start with a type, and it does not end in `;`.  Breaking
        # either alone changes nothing, which is the point -- so the mutation
        # that shows the guard is real has to break both.
        ('the definition pattern loosened at both ends, so a loop header is a definition',
         sub(EQUIV,
             "DEF = re.compile(r'^\\s*(?:const\\s+)?'\n"
             "                 r'(?:int32_t|unsigned|float|double|auto|size_t)\\s+'\n"
             "                 r'(v\\d+_\\w+)\\s*=\\s*(.+?);\\s*$')",
             "DEF = re.compile(r'.*?(?:const\\s+)?'\n"
             "                 r'(?:int32_t|unsigned|float|double|auto|size_t)\\s+'\n"
             "                 r'(v\\d+_\\w+)\\s*=\\s*(.+?);')")),
        ('a typed literal falls back to flat text',
         sub(EQUIV, r"_SUFFIX = re.compile(r'\b(\d+)_[iu]\d+\b')",
             r"_SUFFIX = re.compile(r'\b(\d+)_NOMATCH\b')")),
        ('an unparseable subscript silently downgrades to text',
         sub(EQUIV, '        raise ValueError(',
             "        return re.sub(r'\\s+', '', text)\n        raise ValueError(")),
        ('identity folding dropped',
         sub(EQUIV, '        if isinstance(node.op, ast.Add):', '        if False:')),
        ('definitions never expand, so inlining reads as a change',
         sub(EQUIV, "    if depth > 64:              # a cycle would mean the source is not SSA",
             "    if True:")),
    ]),

    'access': ('tests/test_pir_access.py', [
        ('the structured store falls back to text',
         sub(SYM, '    structured = (isinstance(variable, _Value)',
             '    structured = (False and isinstance(variable, _Value)')),
        ('the address pinned again, so nothing folds or shares',
         sub(SYM, '    return self.build_address(writer, context, index, shift=shift)\n\n'
                  '  def access_address',
             '    return writer.pin(self.build_address(writer, context, index, shift=shift))\n\n'
             '  def access_address')),
        ('the innermost loop body wrapped in a scope again',
         sub(SYM, '    if len(loops) == 0:\n      inner(varlist)',
             '    if len(loops) == 0:\n      with writer.Scope():\n        inner(varlist)')),
        ('a scalar routed through Op.LOAD, inventing a subscript',
         sub(SYM, '''      if (bc_lane is None and self.stype in (
              SymbolType.Register, SymbolType.Scratch, SymbolType.SharedMem,
              SymbolType.Batch, SymbolType.Global)):''',
             '      if bc_lane is None:')),
    ]),

    'reachability': ('tests/test_amd_reachability.py', [
        ('an unreachable function added',
         lambda: (PKG / 'arch.py',
                  (PKG / 'arch.py').read_text() + '\n\ndef orphan(x):\n    return x + 1\n')),
        ('a name defined in two modules',
         lambda: (PKG / 'caps.py',
                  (PKG / 'caps.py').read_text() + '\n\ndef rdna(ctx):\n    return True\n')),
        ('an empty stub added',
         lambda: (PKG / 'codegen.py',
                  (PKG / 'codegen.py').read_text() + '\n\ndef hook(writer):\n    pass\n')),
    ]),

    # `test_syntax.py` reads the committed snapshots, so a mutation to the
    # generator would not reach it.  The snapshots are the input here.
    #
    # The last three are what a check of the reference-taking calls alone
    # would miss: lifting those out with a regex leaves a defect anywhere else
    # in the kernel --- or in an argument the regex cannot type --- outside
    # what it can see.
    'syntax': ('tests/test_syntax.py', [
        ('a literal handed to a reference parameter',
         resub(Path('tests/snapshots/gemm_56x18_x_18x18.hip.cpp'),
               r'tensorforge::transpose4x4b32\((\w+), (\w+), \w+, \w+,',
               r'tensorforge::transpose4x4b32(\1, \2, 0.0f, 0.0f,')),
        ('an argument dropped from a transpose',
         resub(Path('tests/snapshots/gemm_square_16.hip.cpp'),
               r'(tensorforge::transpose16x16b32\((?:\w+, ){14}\w+), \w+\);',
               r'\1);')),
        ('an operand that is never declared',
         resub(Path('tests/snapshots/gemm_square_16.hip.cpp'),
               r'(tensorforge::transpose16x16b32\((?:\w+, ){15})\w+\);',
               r'\1undeclared_operand);')),
        ('an MFMA accumulator of the wrong width',
         resub(Path('tests/snapshots/gemm_56x18_x_18x18.hip.cpp'),
               r'tensorforge::VectorT<float, 4> (\w+_acc) =',
               r'tensorforge::VectorT<float, 2> \1 =')),
        ('a store past the end of a shared-memory declaration',
         sub(Path('tests/snapshots/gemm_square_16.hip.cpp'),
             'float* localShrMem0 = &totalShrMem[16 * threadIdx.y + 0];',
             'float* localShrMem0 = &totalShrMem[16 * threadIdx.y + 0]; '
             'const auto _unused = undeclared_symbol;',
             1)),
    ]),

    # One list, two readers.  A tool that reports permanent failures for
    # cases the suite already tracks is how a check stops being read.
    'knownbad': ('tests/test_tools.py::test_the_runner_agrees_with_the_suite', [
        ('the tool stops recognizing a tracked failure',
         sub(Path('tools/syntax_check.py'),
             '    reason = syntax.known_bad(r.path)',
             "    reason = ''", 1)),
        ('a tracked entry counted as well-formed',
         sub(Path('tools/syntax_check.py'),
             "        kinds['known'] += 1",
             "        kinds['ok'] += 1", 1)),
    ]),

    # The shim is a copy of a C++ fact; the check that it stays one has to
    # fail when the copy drifts, in either direction.
    'shim': ('tests/test_syntax.py::test_shim_matches_the_device_headers', [
        ('an overload dropped from the shim',
         sub(Path('tests/shim/tensorforge_host.h'),
             'template <int Row> void fmacdpp16(double &c, double a, double b);\n',
             '', 1)),
        ('the shim made more permissive than the header',
         sub(Path('tests/shim/tensorforge_host.h'),
             'template <int Row, typename T> T movdpp16(T a);',
             'template <int Row, typename T> T movdpp16(T a);\n'
             'template <int Row, typename T> T movdpp16(T a, T b);', 1)),
        ('a parameter that should be a reference passed by value',
         sub(Path('tests/shim/tensorforge_host.h'),
             'void transpose16x2(T &w1, T &w2, T v1, T v2);',
             'void transpose16x2(T w1, T &w2, T v1, T v2);', 1)),
    ]),

    # The sparse loader's layout is a claim about a *write*, recorded where the
    # write happens and read back somewhere else.  Both ends have to fail.
    'sparse': ('tests/test_sparse_layout.py', [
        ('the shared image stops being described',
         sub(Path('src/tensorforge/backend/symbol.py'),
             '    if self.stype not in (SymbolType.Register, SymbolType.SharedMem):',
             '    if self.stype is not SymbolType.Register:', 1)),
        ('a global image described as though it were staged',
         sub(Path('src/tensorforge/backend/symbol.py'),
             '    if self.stype not in (SymbolType.Register, SymbolType.SharedMem):',
             '    if False:', 1)),
        ('the fill records nothing',
         sub(Path('src/tensorforge/backend/symbol.py'),
             '    self._record_linear_layout(index, vec, threads, writer)\n', '', 1)),
        ('the read drops what the fill recorded',
         sub(Path('src/tensorforge/backend/symbol.py'),
             "        return writer.load(buf, addr, type_=ltype, hint='lin',\n"
             "                           layout=self._linear_read_layout(index, vec,\n"
             "                                                           threads),",
             "        return writer.load(buf, addr, type_=ltype, hint='lin',\n"
             "                           layout=None,", 1)),
        ('the wave width taken as the block instead of the thread count',
         sub(Path('src/tensorforge/backend/symbol.py'),
             'layout = RegisterLayout((LaneAxis(threads, 1),))',
             'layout = RegisterLayout((LaneAxis(threads, 2),))', 1)),
        ('a mid-slot fill claimed anyway',
         sub(Path('src/tensorforge/backend/symbol.py'),
             'if not isinstance(index, int) or index % (threads * vec) != 0:',
             'if not isinstance(index, int):', 1)),
        ('two disagreeing fills, last one wins',
         sub(Path('src/tensorforge/backend/symbol.py'),
             '      self.layout = None\n      return\n    self.layout = layout',
             '      pass\n    self.layout = layout', 1)),
        ('the layout lost on clone',
         sub(Path('src/tensorforge/backend/symbol.py'),
             '    cloned.layout = self.layout\n', '', 1)),
    ]),

    # The contract on `Tensor.data`, and two test patterns a refactor can make
    # stale without anything failing loudly enough.
    'data': ('tests/test_regressions.py', [
        ('a list handed to Tensor.data',
         sub(Path('src/tensorforge/generators/descriptions.py'),
             'data=(np.array(alpha, dtype=float)',
             'data=([alpha]', 1)),
        ('the shape check dropped',
         sub(Path('src/tensorforge/common/matrix/tensor.py'),
             '            if self.data.shape != self.shape:',
             '            if False:', 1)),
        ('the type check dropped',
         sub(Path('src/tensorforge/common/matrix/tensor.py'),
             '            if not isinstance(self.data, np.ndarray):',
             '            if False:', 1)),
    ]),

    # The guard that keeps the nvidia path from silently going dead, and the
    # gate that keeps it from aborting cases it cannot take.
    'nvidia': ('tests/test_nvidia_reachability.py', [
        ('a second definition of matmul',
         sub(Path('src/tensorforge/backend/instructions/compute/primitives/nvidia.py'),
             'def matmul(writer, ops, ctx, span):',
             'def matmul(*args, **kwargs):\n    pass\n\n'
             'def matmul(writer, ops, ctx, span):', 1)),
        ('an unreachable helper reintroduced',
         sub(Path('src/tensorforge/backend/instructions/compute/primitives/nvidia.py'),
             'def tfconvert(writer: Writer, variables):',
             'def shuffle_swap(writer, v):\n'
             '    return f"__shfl_xor_sync(0xffffffff, {v}, 1)"\n\n'
             'def tfconvert(writer: Writer, variables):', 1)),
    ]),

    'fragmentbits': ('tests/test_fragment_order.py tests/test_nvidia_gate.py', [
        ("the row group's place value in the operand list halved",
         sub(Path('src/tensorforge/backend/instructions/compute/primitives/nvidia.py'),
             '                 + [Bit(Place.SLOT, 2 << b)\n'
             '                    for b in range((mregs - 1).bit_length())])\n'
             '    cols = tuple([Bit(Place.SLOT, 1)]',
             '                 + [Bit(Place.SLOT, 1 << b)\n'
             '                    for b in range((mregs - 1).bit_length())])\n'
             '    cols = tuple([Bit(Place.SLOT, 1)]')),
        ("the accumulator's column pair read above its lane bits",
         sub(Path('src/tensorforge/backend/instructions/compute/primitives/nvidia.py'),
             '    cols = tuple([Bit(Place.SLOT, 1)]\n'
             '                 + [Bit(Place.LANE, 1 << b) for b in range(2)])',
             '    cols = tuple([Bit(Place.LANE, 1 << b) for b in range(2)]\n'
             '                 + [Bit(Place.SLOT, 1)])')),
        ('the slot offsets read off a lane that is not the first',
         sub(Path('src/tensorforge/backend/instructions/compute/primitives/nvidia.py'),
             '            if at.lane == 0:\n'
             '                cells[at.slot] = row * atom.n + col',
             '            if at.lane == 1:\n'
             '                cells[at.slot] = row * atom.n + col')),
        ("the row's lane bits read one place too low",
         sub(Path('src/tensorforge/backend/instructions/compute/primitives/nvidia.py'),
             'rows = tuple([Bit(Place.LANE, 1 << (2 + b)) for b in range(3)]',
             'rows = tuple([Bit(Place.LANE, 1 << (1 + b)) for b in range(3)]')),
        ("the column's slot digit given the row's place value",
         sub(Path('src/tensorforge/backend/instructions/compute/primitives/nvidia.py'),
             '+ [Bit(Place.SLOT, mregs << b)',
             '+ [Bit(Place.SLOT, 1 << b)')),
        ('two cells allowed onto one slot and lane',
         sub(Path('src/tensorforge/backend/instructions/compute/primitives/nvidia.py'),
             '            if (at.slot, at.lane) in cells:',
             '            if False:')),
    ]),

    'gate': ('tests/test_nvidia_gate.py', [
        ('the i8 entries let into the candidates',
         sub(Path('src/tensorforge/backend/instructions/compute/primitives/nvidia.py'),
             'EMITTED_MODES = (MMAMode.TF32, MMAMode.DIRECT)',
             'EMITTED_MODES = (MMAMode.TF32, MMAMode.DIRECT, MMAMode.I8)', 1)),
        ('the capability floor raised so a contextless caller is credited',
         sub(Path('src/tensorforge/backend/instructions/compute/primitives/nvidia.py'),
             'BASELINE_SM = 75', 'BASELINE_SM = 90', 1)),
        ('the target dropped, so every arch selects against the floor',
         sub(Path('src/tensorforge/backend/instructions/compute/primitives/nvidia.py'),
             '    return int(digits) if digits else BASELINE_SM',
             '    return BASELINE_SM', 1)),
        ('the accumulator row stride pinned to m16n8k8',
         sub(Path('src/tensorforge/backend/instructions/compute/primitives/nvidia.py'),
             '                cells[at.slot] = row * atom.n + col',
             '                cells[at.slot] = row * 8 + col', 1)),
        ('the accumulator column pair dropped',
         sub(Path('src/tensorforge/backend/instructions/compute/primitives/nvidia.py'),
             '            if at.lane == 0:\n'
             '                cells[at.slot] = row * atom.n + col',
             '            if at.lane == 0 and at.slot % 2 == 0:\n'
             '                cells[at.slot] = row * atom.n + col', 1)),
        ('an address goes back to raw text',
         sub(Path('src/tensorforge/backend/instructions/compute/primitives/nvidia.py'),
             "    v = writer.thread_id('x')",
             "    return writer.rawexpr('threadIdx.x', type_=INDEX, hint='a')\n"
             "    v = writer.thread_id('x')", 1)),
        ('the wrap is dropped from the index',
         sub(Path('src/tensorforge/backend/instructions/compute/primitives/nvidia.py'),
             '    if mod is not None:', '    if False:', 1)),
        ('the stride is dropped from the index',
         sub(Path('src/tensorforge/backend/instructions/compute/primitives/nvidia.py'),
             '    if scale != 1:', '    if False:', 1)),
        ('a staged fragment goes back to a varalloc name',
         sub(Path('src/tensorforge/backend/instructions/compute/primitives/nvidia.py'),
             '                                    got = A(writer, None, slot,\n'
             '                                            k + kk + kkk, parts=aparts)',
             "                                    got = writer.varalloc()\n"
             "                                    A(writer, f'{got}', slot,\n"
             "                                      k + kk + kkk, parts=aparts)", 1)),
        ('a padding fragment declared as text again',
         sub(Path('src/tensorforge/backend/instructions/compute/primitives/nvidia.py'),
             "                                        AregParts[pt][q, kkk] = writer.declare(\n"
             "                                            ScalarType(atom.d), hint='as')",
             "                                        AregParts[pt][q, kkk] = writer.varalloc()\n"
             "                                        writer(f'float {AregParts[pt][q, kkk]}{{}};',\n"
             "                                               accesses=())", 1)),
        ('the wave width no longer checked',
         sub(Path('src/tensorforge/backend/instructions/compute/primitives/nvidia.py'),
             '    return (threads <= WAVE and WAVE % threads == 0\n'
             '            and dtype in (Datatype.F32, Datatype.F64)',
             '    return (dtype in (Datatype.F32, Datatype.F64)', 1)),
        ('the operand type no longer checked',
         sub(Path('src/tensorforge/backend/instructions/compute/primitives/nvidia.py'),
             '            and dtype in (Datatype.F32, Datatype.F64)\n'
             '            and not sparse',
             '            and not sparse', 1)),
        ('the gate bypassed entirely',
         sub(Path('src/tensorforge/backend/instructions/compute/primitives/nvidia.py'),
             '    if (enabled(ctx)\n'
             '            and supports(shape.threads, shape.accumulator, '
             'shape.sparse,\n'
             '                         shape.depth)\n'
             '            and instrs_for(shape.accumulator, sm_of(ctx))):',
             '    if enabled(ctx):', 1)),
        ('the arch dropped from the offer, so a target with no entry is offered one',
         sub(Path('src/tensorforge/backend/instructions/compute/primitives/nvidia.py'),
             '            and instrs_for(shape.accumulator, sm_of(ctx))):',
             '            and True):', 1)),
        ('the deployment switch flipped without re-recording',
         sub(Path('src/tensorforge/backend/instructions/compute/primitives/nvidia.py'),
             'ENABLED = False', 'ENABLED = True', 1)),
    ]),

    # A raw statement may narrow what it touches, and must then be complete
    # about what it uses.  Both halves, plus the arena a shared buffer needs.
    'rawaccess': ('tests/test_pir_raw_accesses.py', [
        ('a body with no arena allocates shared memory anyway',
         sub(Path('src/tensorforge/backend/pir/build.py'),
             "            if arena is None:\n                raise GenerationError(",
             "            if False:\n                raise GenerationError(", 1)),
        ('a narrowed access set not checked at all',
         sub(Path('src/tensorforge/backend/pir/build.py'),
             '            self._check_declared_accesses(code, accesses, args, defines)',
             '            pass', 1)),
        ('the operand requirement dropped',
         sub(Path('src/tensorforge/backend/pir/build.py'),
             '            if id(v) not in named:',
             '            if False:', 1)),
        ('the declared-access requirement dropped',
         sub(Path('src/tensorforge/backend/pir/build.py'),
             '            if id(v) in shared and not conservative and v not in covered:',
             '            if False:', 1)),
        ('operands accepted but not recorded, so no use edge',
         sub(Path('src/tensorforge/backend/pir/build.py'),
             'return self._emit_op(Op.RAWSTMT, tuple(defines), tuple(args),\n'
             '                             pure=False, movable=False,',
             'return self._emit_op(Op.RAWSTMT, tuple(defines), (),\n'
             '                             pure=False, movable=False,', 1)),
        ('a raw statement made movable by declaring no accesses',
         sub(Path('src/tensorforge/backend/pir/build.py'),
             '                             pure=False, movable=False,',
             '                             pure=False, movable=(accesses == ()),', 1)),
        ('a varalloc name asked to claim a use it cannot have',
         sub(Path('src/tensorforge/backend/pir/build.py'),
             '        self._by_name.pop(str(v), None)\n', '', 1)),
        ('defines accepted but discarded',
         sub(Path('src/tensorforge/backend/pir/build.py'),
             'return self._emit_op(Op.RAWSTMT, tuple(defines), tuple(args),',
             'return self._emit_op(Op.RAWSTMT, (), tuple(args),', 1)),
    ]),

    # Where the threads meet, from the order of the final body.
    'barriers': ('tests/test_pir_barriers.py tests/test_ternary.py', [
        ('a read of what another lane wrote not met',
         sub(BARRIERS, '            for hit in hits:\n                needed = True',
             '            for hit in ():\n                needed = True', 1)),
        ('a write over bytes read since the last barrier not met',
         sub(BARRIERS, '                if _overlap(ext, rext):\n                    needed = True',
             '                if False:\n                    needed = True', 1)),
        ('extents of other buffers taken for disjoint',
         sub(BARRIERS, '    return la < hb and lb < ha', '    return False', 1)),
        ('a store after a clearing store not met',
         sub(BARRIERS, '            for where in cleared:\n                needed = True',
             '            for where in ():\n                needed = True', 1)),
        ('a clearing store after a store not met',
         sub(BARRIERS, "        if s.attr('mark') == 'clears':",
             "        if False:", 1)),
        ('a one-lane store not handed off',
         sub(BARRIERS, '            out.writes[key] = (key in self.handoff, root)',
             '            out.writes[key] = (False, root)', 1)),
        ('a one-lane store into global memory not recognized',
         sub(Path('src/tensorforge/generators/generator.py'),
             '  return tuple(roots.get(id(sym), sym) for sym in out)',
             '  return tuple(out)', 1)),
        ('the back edge not followed',
         sub(BARRIERS, '            for _ in range(8):', '            for _ in range(1):', 1)),
        ('a barrier already there not counted',
         sub(BARRIERS, '        if int(s.effect) & _M_BARRIER:\n            return state.fence(_scope_of(s))',
             '        if int(s.effect) & _M_BARRIER:\n            return state', 1)),
        ('a barrier put inside a block some lanes skip',
         sub(BARRIERS, '            if inner < self.arrival:\n                if needed:',
             '            if False:\n                if needed:', 1)),
        ('a statement that says nothing taken to touch nothing',
         sub(BARRIERS, "                reads.append((_OPAQUE, _ANYWHERE, MULT))\n"
                       "                writes.append((_OPAQUE, _ANYWHERE, MULT))\n",
             "", 1)),
    ]),

    # The allocator derives a layout; this is the check that it holds.
    'layoutcheck': ('tests/test_layout_check.py', [
        ('a read across another buffer\'s write no longer reported',
         sub(LAYOUT, '                if t.reads and last_write[other] is not None:',
             '                if False:', 1)),
        ('a rewrite between the clobber and the read not noticed',
         sub(LAYOUT, '                    if mine is None or mine < last_write[other]:',
             '                    if True:', 1)),
        ('windows compared without checking that they overlap',
         sub(LAYOUT, '            if not win[a].overlaps(win[b]):\n                continue',
             '            if False:\n                continue', 1)),
        ('an undeclared statement passed over in silence',
         sub(LAYOUT, '                opaque.append(here)', '                pass', 1)),
        ('a mark counted as a rewrite',
         sub(LAYOUT, '        if stmt.op in (Op.ALLOC, Op.MARK):\n            continue\n        for a in stmt.accesses:',
             '        if stmt.op == Op.ALLOC:\n            continue\n        for a in stmt.accesses:', 1)),
        ('two windows of one buffer taken for two buffers',
         sub(LAYOUT, "    return ('o', id(identity)) if identity is not None \\\n        else ('v', stmt.target[0].id)",
             "    return ('v', stmt.target[0].id)", 1)),
    ]),

    'operands': ('tests/test_snapshots.py', [
        ('hfma asks for the wrong distribution',
         sub(PKG / 'relayout.py',
             '    return RegisterLayout((LaneAxis(step, 1),))\n',
             '    return RegisterLayout((LaneAxis(max(step // 2, 1), 1),))\n', 1)),
        ('the transpose skipped, MFMA fed a raw load',
         sub(PKG / 'codegen.py',
             '                        reached = transpose(regs)',
             '                        reached = list(regs)')),
        ('the gap answered as already closed',
         sub(PKG / 'relayout.py',
             '    if gap == ():\n        return 0',
             '    return 0\n    if gap == ():\n        return 0')),
        ('a gap this instruction does not close emitted anyway',
         sub(PKG / 'relayout.py',
             '    return 1 if gap == transpose_exchange(ext) else None',
             '    return 1')),
        ('the broadcast lane taken from the table, not the algorithm',
         sub(PKG / 'codegen.py',
             '                    params = dict(params, lane=i // step)',
             '                    params = dict(params, lane=0)')),
    ]),

    # The holes of a merged run that share members: one buffer to the passes,
    # no `restrict`, and the transfer for the next trip behind this trip's
    # write of what it reads.
    'tablealias': ('tests/test_rolling.py::test_holes_that_share_members_are_one_buffer_to_the_passes '
                   'tests/test_rolling.py::test_holes_that_only_read_shared_members_stay_apart '
                   'tests/test_rolling.py::test_the_next_trips_level_is_fetched_behind_this_trips_write_of_it '
                   'tests/test_rolling.py::test_a_merged_run_computes_what_the_list_written_out_does', [
        ('the tables of a run never joined',
         sub(PTR, '    self._join_tables()\n', '', 1)),
        ('tables of reads alone joined as well',
         sub(PTR, '      if len(group) < 2 or not any(t.writable for t in group):',
             '      if len(group) < 2:', 1)),
        ('a pointer of the family still promising restrict',
         sub(PTR, '    restrict = (self._space() is not MemSpace.SHARED\n'
                  '                and self._alias_root() is self._src)',
             '    restrict = self._space() is not MemSpace.SHARED', 1)),
        ('the transfer for the next trip left where its buffer frees up',
         sub(WRAP, '        if l.kind is not None:\n            self._behind_writers(rest)\n',
             '', 1)),
        ('the writes of what it reads not looked for in a counted loop',
         sub(WRAP, '        if self.kind is not None:\n            writers = tuple(',
             '        if False:\n            writers = tuple(', 1)),
    ]),
    'closedtrips': ('tests/test_rolling.py::test_a_merged_run_computes_what_the_list_written_out_does '
                    'tests/test_rolling.py::test_a_closed_trip_stores_a_temporary_the_next_trip_reads '
                    'tests/test_rolling.py::test_a_closed_trip_stores_what_outlives_it_and_nothing_it_used_up '
                    'tests/test_residency.py', [
        ('a closed trip opened on a result still in registers',
         sub(GEN, '      self._section.ir.extend(self._residency.flush(name))\n'
                  '      if name not in writes:',
             '      if name not in writes:', 1)),
        ('a closed trip that stores nothing at its end',
         sub(RESIDENCY, '        out = [self._store(entry) for entry in self._entries.values()\n'
                        '               if not entry.is_preload and outlives(entry)]',
             '        out = []', 1)),
        ('a trip that forgets what the next one reads of it',
         sub(GEN, '      carried.update(id(t) for t in reads if id(t) not in seen)\n', '', 1)),
        ('a trip that stores the temporaries it used up',
         sub(GEN, '      return id(tensor) in needed\n', '      return True\n', 1)),
        ('a trip lets go of a result held around it',
         sub(RESIDENCY, '                if not entry.is_preload:\n'
                        '                    raise InternalError(',
             '                if False:\n'
                        '                    raise InternalError(', 1)),
    ]),
    'carry': ('tests/test_carry.py '
              'tests/test_rolling.py::test_the_carry_holds_an_accumulator_in_registers_over_the_trips '
              'tests/test_rolling.py::test_an_atomic_accumulator_is_summed_in_registers_and_added_once '
              'tests/test_accumulate_atomically.py '
              'tests/test_rolling.py::test_an_accumulator_begun_ahead_of_the_run_stays_in_its_registers '
              'tests/test_rolling.py::test_a_temporary_each_lane_adds_onto_is_held_in_its_registers '
              'tests/test_rolling.py::test_a_merged_run_computes_what_the_list_written_out_does', [
        ('a store of other elements taken for the load\'s',
         sub(CARRY, '                    if held.get(lane, {}).get(slot, _ZERO) != element:',
             '                    if False:', 1)),
        ('a binding handed on goes unnoticed',
         sub(CARRY, '        if bindings & self._addresses:', '        if False:', 1)),
        ('the image the load fills written by the trip',
         sub(CARRY, '                if a.writes:\n'
                    '                    raise _Refused(\'the trip writes the image the load fills\')',
             '                if False:\n'
                    '                    raise _Refused(\'the trip writes the image the load fills\')', 1)),
        ('the copy fused without asking the slots',
         sub(CARRY, "                    if what == 'read head' and slot in first \\\n"
                    "                            and at > first[slot]:\n"
                    "                        return body\n", '', 1)),
        ('the image computed in place for rows nothing hands on',
         sub(CARRY, '                if set(first) != handed.get(lane, set()):',
             '                if False:', 1)),
        ('a load ahead of the loop taken from a store that left rows out',
         sub(CARRY, '                if written.get(lane, {}).get(element) != slot:',
             '                if False:', 1)),
        ('a load ahead of the loop looked for where an earlier one rebuilt it',
         sub(CARRY, '                for s in (rewrite((x,)) if id(x) in load else (x,))]',
             '                for s in rewrite((x,))]', 1)),
        ('the store ahead dropped where the one behind writes less',
         sub(CARRY, '        dead = all(set(w) <= c.stored.get(lane, set())\n'
                    '                   for lane, w in written.items())',
             '        dead = True', 1)),
        ('a load under a guard the store is not under carried',
         sub(CARRY, '        if any(not spelled(e) <= guards for u in loads for e in u.events):',
             '        if False:', 1)),
        ('a guard that names what the trip computes carried',
         sub(CARRY, "                if re.search(rf'\\b{re.escape(name)}\\b', text):",
             '                if False:', 1)),
        ('the trips summed in place under a guard where another output reads them',
         sub(CARRY, '        if closed or not guards:', '        if True:', 1)),
        ('a read of the tail besides the copy taken for closed',
         sub(CARRY, '                if _key(a.base) == _key(tail.image) and not a.writes:\n'
                    '                    return False\n', '', 1)),
        ('a store ahead under a guard dropped where the trips read more',
         sub(CARRY, '        if guards != c.guards or (guards and not c.closed):',
             '        if guards != c.guards:', 1)),
        ('a load behind the loop outside the guard taken from registers',
         sub(CARRY, '            if not (c.guards <= under or (not c.fused and c.loaded <= under)):',
             '            if False:', 1)),
        ('a load behind the loop taken from what was not stored',
         sub(CARRY, '                if (element not in c.stored.get(lane, set())\n'
                    '                        or c.held.get(lane, {}).get(slot) != element):',
             '                if False:', 1)),
        ('an element another lane writes taken for a lane\'s own',
         sub(CARRY, '        if any(len(touched[e]) > 1 for e in written):',
             '        if False:', 1)),
        ('a register given two elements of a lane',
         sub(CARRY, '                if held.setdefault(find(node), element) != element:',
             '                if False:', 1)),
        ('a lane\'s first read taken from what nothing stored',
         sub(CARRY, '            if not set(first.values()) <= seeded.get(lane, set()):',
             '            if False:', 1)),
        ('the store ahead of a lane\'s own buffer always dropped',
         sub(CARRY, '        elif (again or not read_behind) and self._movable(out, at):',
             '        elif self._movable(out, at):', 1)),
        ('the back edge of a loop around the run not followed',
         sub(CARRY, '        if _repeats(holder) and _again(path, level, key):',
             '        if False:', 1)),
        ('the carry not run at all',
         sub(CARRY, "            if counted(s) and s.attr('rolled'):\n",
             "            if False:\n", 1)),
    ]),
    # An accumulation added atomically where nothing reads the sum, and the
    # per-lane account of which store every load reads that it asks.
    'atomics': ('tests/test_accumulate_atomically.py tests/test_flow.py '
                'tests/test_rolling.py::test_an_atomic_destination_starts_from_zeros '
                'tests/test_rolling.py::test_an_atomic_accumulator_is_summed_in_registers_and_added_once', [
        ('a read of what the store wrote ignored',
         sub(ATOMICS, '        if m & c.mask and reader not in heads:', '        if False:', 1)),
        ('a register on the chain read elsewhere taken for a link',
         sub(ATOMICS, '            if m & l.mask and reader != l.load:', '            if False:', 1)),
        ('a write between the load and the store ignored',
         sub(ATOMICS, '            if r is None or before.get((lane, cell)) != (r.writer, True):',
             '            if r is None:', 1)),
        ('another element taken for the one the load read',
         sub(ATOMICS, '        if loaded.get(lane) != stored.get(lane):', '        if False:', 1)),
        ('lanes the load leaves to an else-value added onto',
         sub(ATOMICS, '    if c.mask & ~bias.mask:\n', '    if False:\n', 1)),
        ('a store where its load did not run',
         sub(ATOMICS, '    if not bias.conds <= store.conds:', '    if False:', 1)),
        ('a statement on a chain changed for some of its runs',
         sub(ATOMICS, '            if set(flow.by_stmt.get(stmt, ())) - seen:', '            if False:', 1)),
        ('a type the target cannot add made atomic',
         sub(ATOMICS, '        if not supported(s.args[1]):', '        if False:', 1)),
        ('the image carried over a loop left with its bias',
         sub(ATOMICS, '                self.zeroed.add(id(bias))\n', '', 1)),
        ('a guard spelled in text dropped from what a statement stands under',
         sub(FLOW, '                inner = conds if s.text in self.masks \\\n'
                   '                    else conds | {(s.text, True)}',
             '                inner = conds', 1)),
        ('a loop that runs again read as running once',
         sub(FLOW, '        self._wrap(s, first)\n', '', 1)),
        ('a write under a guard taken for one that replaces what was there',
         sub(FLOW, '                    if layer.conds <= conds:\n'
                   '                        prior = layer.writer\n'
                   '                        break\n'
                   '                    certain = False\n',
             '                    prior = layer.writer\n'
                   '                    break\n', 1)),
        ('what text reads not taken as read',
         sub(FLOW, '                            for layer in table.get(cell, ()):\n'
                   '                                self.readers[layer.writer].append(\n'
                   '                                    (n, self.full))\n'
                   '                                self.escaped.add(layer.writer)\n',
             '                            for layer in table.get(cell, ()):\n'
                   '                                pass\n', 1)),
        ('a binding a loop computes taken for one buffer every trip',
         sub(FLOW, '            hops = tuple(env.get(l.induction.id)\n'
                   '                         for l in self.loops.get(id(d), ()))',
             '            hops = ()', 1)),
        ('a buffer defined anew taken for what it held',
         sub(FLOW, "        if s.op == Op.MARK and s.attr('mark') == 'defines':",
             '        if False:', 1)),
        ('an element mask taken for a guard of unknown value',
         sub(ATOMICS, '    flow = Flow(body, threads, arena, masks=True)',
             '    flow = Flow(body, threads, arena)', 1)),
        ('a body past the budget asked all the same',
         sub(FLOW, '        if self._budget < 0:\n', '        if False:\n', 1)),
        ('a store the flow cannot place taken for one it can',
         sub(ATOMICS, '    if id(s) in flow.opaque:\n', '    if False:\n', 1)),
    ]),
    # A load taken from where its own lane's store took the value, a store
    # nothing reads dropped, and the masks taken to hold for it.
    'forward': ('tests/test_forward.py', [
        ('a register written since taken for what it held',
         sub(FLOW, '            if not layer.conds <= conds or layer.writer != writer:\n'
                   '                return False\n', '', 1)),
        ('another lane\'s store taken for the lane\'s own',
         sub(FLOW, '            if not r.certain or r.writer is None or r.writer < 0 or not r.own:',
             '            if not r.certain or r.writer is None or r.writer < 0:', 1)),
        ('a load taken where some of its lanes read something else',
         sub(FORWARD, '            if covered & inst.mask != inst.mask:\n'
                      '                return False\n', '', 1)),
        ('a value taken where it does not stand',
         sub(FORWARD, '        if v.type != s.target[0].type or not self._stands(v, s):',
             '        if v.type != s.target[0].type:', 1)),
        ('a register a rolled loop picks taken',
         sub(FORWARD, '        if fit is None or set(fit[1]) - unrolled:',
             '        if fit is None:', 1)),
        ('a store the body ends with taken for dead',
         sub(FORWARD, '        if all(n not in flow.escaped and n not in flow.final and n not in kept',
             '        if all(n not in flow.escaped and n not in kept', 1)),
        ('a store a forwarded load reads from now on dropped',
         sub(FORWARD, '        if all(n not in flow.escaped and n not in flow.final and n not in kept',
             '        if all(n not in flow.escaped and n not in flow.final', 1)),
        ('a named load replaced by a number',
         sub(FORWARD, "        if kind != 'reg' and s.attr('extern') is not None:",
             '        if False:', 1)),
        ('a guard spelled in text taken to hold',
         sub(FLOW, '                inner = conds if s.text in self.masks \\\n'
                   '                    else conds | {(s.text, True)}',
             '                inner = conds', 1)),
        ('masks taken to hold where a global write stands outside them',
         sub(FLOW, '                    if a.space in (MemSpace.GLOBAL, MemSpace.UNKNOWN):\n'
                   '                        if not under:\n'
                   '                            return False\n',
             '                    if a.space in (MemSpace.GLOBAL, MemSpace.UNKNOWN):\n'
                   '                        pass\n', 1)),
        ('masks taken to hold where a branch turns on memory',
         sub(FLOW, '                    if loads.reads(cond):\n'
                   '                        return False\n', '', 1)),
        ('a statement that stands in for others asked as a write of its own',
         sub(FLOW, '                if _stands_in(x):\n'
                   '                    continue\n', '', 1)),
        ('text with a known effect taken to stand in for others',
         sub(FLOW, '    return (x.op == Op.RAWSTMT and not x.text and bool(x.accesses)',
             '    return (x.op == Op.RAWSTMT and bool(x.accesses)', 1)),
        ('a zeroed declaration taken to hold nothing known',
         sub(FLOW, "                if zeros and self._values and space is MemSpace.REGISTER:\n"
                   "                    self.origin[n] = [(mask, ('const', 0))]\n", '', 1)),
        ('a buffer nothing reaches any more left declared',
         sub(FORWARD, '        body = _strip(body, _unreached(body))\n'
                      '        return substitute(body, self.mapped) if self.mapped else body\n',
             '        return substitute(body, self.mapped) if self.mapped else body\n', 1)),
        ('a buffer something still reaches taken for unreached',
         sub(FORWARD, '            reached |= hit\n', '            pass\n', 1)),
        ('a body past the budget forwarded all the same',
         sub(FLOW, '        if self._budget < 0:\n', '        if False:\n', 1)),
    ]),
}


LOCK = Path('.mutation-in-progress')


def _recover():
    """Undo a run that was killed before its `finally` could run.

    `git checkout --` rather than the recorded text: the lock survives because
    the process did not, so whatever it was holding in memory is gone.  Only
    the paths named in the lock are touched, so unrelated edits in the working
    tree are left alone.
    """
    if not LOCK.exists():
        return
    paths = [p for p in LOCK.read_text().split('\n') if p]
    print(f'a previous run was killed with {len(paths)} file(s) mutated; '
          f'restoring from git:')
    for p in paths:
        print(f'  {p}')
    if paths:
        subprocess.run(['git', 'checkout', '--', *paths], check=False)
    LOCK.unlink()
    print()


def run(group, target, mutations):
    print(f'\n=== {group}  ({target})')
    originals = {}
    caught = 0
    try:
        for name, make in mutations:
            try:
                path, text = make()
            except AssertionError as exc:
                print(f'  {name:56s} SKIPPED: {exc}')
                continue
            originals.setdefault(path, path.read_text())
            LOCK.write_text('\n'.join(str(p) for p in originals))
            path.write_text(text)
            r = _run_tests(target)
            said = verdict(r.returncode)
            caught += said == 'caught'
            print(f'  {name:56s} {said if said == "caught" else "*** " + said.upper() + " ***"}')
            for p, t in originals.items():
                p.write_text(t)
            originals.clear()
            LOCK.unlink(missing_ok=True)
    finally:
        for p, t in originals.items():
            p.write_text(t)
        LOCK.unlink(missing_ok=True)
    return caught, len(mutations)


def _dry_run(wanted):
    """Report anchors that no longer match, without running a test.

    Applying a mutation costs a full test run; checking that its anchor is
    still findable costs a string search.  The second is the part that rots,
    and separating them is what lets a test assert freshness without taking
    minutes.
    """
    stale = 0
    for group in wanted:
        target, mutations = GROUPS[group]
        for name, mutation in mutations:
            try:
                # `make()` computes the mutated text and writes nothing, so
                # asking it is enough and there is nothing to undo.
                mutation()
            except AssertionError as exc:
                print(f'  {group}: {name}: {exc}')
                stale += 1
    print(f'\n{stale} anchor(s) no longer match')
    return 1 if stale else 0


def main():
    _recover()
    argv = sys.argv[1:]
    dry = '--dry-run' in argv
    if dry:
        argv = [a for a in argv if a != '--dry-run']
    wanted = argv or list(GROUPS)
    for group in wanted:
        if group not in GROUPS:
            print(f'unknown group {group!r}; have: {", ".join(GROUPS)}')
            return 2
    if dry:
        return _dry_run(wanted)
    total = hit = 0
    for group in wanted:
        target, mutations = GROUPS[group]
        c, n = run(group, target, mutations)
        hit += c
        total += n
    print(f'\n{hit}/{total} mutations caught')
    return 0 if hit == total else 1


if __name__ == '__main__':
    sys.exit(main())
