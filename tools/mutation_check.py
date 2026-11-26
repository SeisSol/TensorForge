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
BLOCKS = Path('src/tensorforge/backend/pir/block_reads.py')
SPMD_H = Path('src/tensorforge/include/tensorforge_device/sycl_spmd.h')
EBLOCKS = Path('src/tensorforge/backend/instructions/memory/element_blocks.py')
STORE = Path('src/tensorforge/backend/instructions/memory/store.py')
ABSTR = Path('src/tensorforge/backend/instructions/abstract_instruction.py')
EQUIV = Path('tools/access_equiv.py')
ALLOC = Path('src/tensorforge/backend/pir/allocate.py')
LAYOUT = Path('src/tensorforge/backend/pir/layout_check.py')
BARRIERS = Path('src/tensorforge/backend/pir/barriers.py')
ENGINES = Path('src/tensorforge/backend/pir/engines.py')
WRAP = Path('src/tensorforge/backend/pir/wrap.py')
AHEAD = Path('src/tensorforge/backend/pir/ahead.py')
PREFETCH = Path('src/tensorforge/backend/pir/prefetch.py')
MOVE = Path('src/tensorforge/backend/pir/move.py')
TRANSFERS = Path('src/tensorforge/backend/pir/transfers.py')
TARGET = Path('src/tensorforge/common/target.py')
LEXIC = Path('src/tensorforge/common/vm/lexic/lexic.py')
HIP_LEXIC = Path('src/tensorforge/common/vm/lexic/hip_lexic.py')
CUDA_LEXIC = Path('src/tensorforge/common/vm/lexic/cuda_lexic.py')
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
OPB = Path('src/tensorforge/backend/instructions/builders/operation_builder.py')
TRIPS = Path('src/tensorforge/backend/pir/trips.py')
ASYNCMEM = Path('src/tensorforge/backend/pir/asyncmem.py')
PIPE = Path('src/tensorforge/backend/pir/pipeline.py')
ROWS = Path('src/tensorforge/backend/pir/rows.py')
STACKING = Path('src/tensorforge/generators/stacking.py')
ME_SECTION = Path('src/tensorforge/backend/instructions/compute/multi_element.py')
ME_GROUP = Path('src/tensorforge/generators/element_group.py')
GRIDDEP = Path('src/tensorforge/backend/pir/griddep.py')
CUDA_LEXIC = Path('src/tensorforge/common/vm/lexic/cuda_lexic.py')
CUDA_H = Path('src/tensorforge/include/tensorforge_device/cuda.h')
BATCHLOOP = Path('src/tensorforge/backend/instructions/batch_loop.py')
TUNING = Path('src/tensorforge/generators/tuning.py')
BOUND = Path('src/tensorforge/analysis/pipeline.py')
CARRY = Path('src/tensorforge/backend/pir/carry.py')
ATOMICS = Path('src/tensorforge/backend/pir/atomics.py')
FORWARD = Path('src/tensorforge/backend/pir/forward.py')
FLOW = Path('src/tensorforge/backend/pir/flow.py')
SINK = Path('src/tensorforge/backend/pir/sink.py')
DEMAND = Path('src/tensorforge/backend/pir/demand.py')
EPILOGUE = Path('src/tensorforge/backend/instructions/compute/epilogue.py')
PROMISE = Path('src/tensorforge/backend/instructions/compute/promise.py')
MULTILINEAR = Path('src/tensorforge/backend/instructions/compute/multilinear.py')
CROSSLANE = Path('src/tensorforge/backend/instructions/compute/crosslane.py')
TILING = PKG / 'tiling.py'
BUFFERS = Path('src/tensorforge/backend/pir/buffers.py')
TENSORS = Path('src/tensorforge/backend/pir/tensors.py')
ENTRIES = Path('src/tensorforge/backend/pir/entries.py')


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
        ('the C tile swizzled over the wave whatever the element size',
         sub(Path('src/tensorforge/backend/instructions/compute/primitives/nvidia.py'),
             "hint='ctile',\n                         swizzle=XorSwizzle(BANK_ROW_BYTES // atom.d.size())))",
             "hint='ctile',\n                         swizzle=XorSwizzle(WAVE)))", 1)),
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
        ('bounds that exclude every lane leave no lane at all',
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
             "hint='ctile',\n                         swizzle=XorSwizzle(BANK_ROW_BYTES // atom.d.size())))",
             "hint='ctile'))", 1)),
        ('the C tile takes the B tile\'s width',
         sub(Path('src/tensorforge/backend/instructions/compute/primitives/nvidia.py'),
             "hint='ctile',\n                         swizzle=XorSwizzle(BANK_ROW_BYTES // atom.d.size())))",
             "hint='ctile',\n                         swizzle=XorSwizzle(atom.k)))", 1)),
        ('the stacked C tile swizzled over the wave whatever the element size',
         sub(Path('src/tensorforge/backend/instructions/compute/primitives/nvidia.py'),
             "hint='ctile',\n                      swizzle=XorSwizzle(BANK_ROW_BYTES // atom.d.size()))",
             "hint='ctile',\n                      swizzle=XorSwizzle(WAVE))", 1)),
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
             '    volume = 1\n    for n in self._dest.data_view.shape:\n      volume *= n',
             '    volume = self._dest.data_view.shape[0]', 1)),
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
             'return (g ^ ((g // self.span) % self.width)) * self.granule + r',
             'return (g ^ (g % self.width)) * self.granule + r', 1)),
        ('a non-power-of-two width accepted',
         sub(Path('src/tensorforge/backend/pir/core.py'),
             'if self.width < 1 or self.width & (self.width - 1):',
             'if False:', 1)),
        ('the divide is by the wrong amount',
         sub(Path('src/tensorforge/backend/pir/build.py'),
             "        bits = swz.span.bit_length() - 1",
             "        bits = swz.span.bit_length()", 1)),
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
             "    ctx = candidate.context(base, **override)\n",
             "    from tensorforge.generators.generator import Generator  # noqa\n"
             "    ctx = candidate.context(base, **override)\n", 1)),
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
         sub(SHARDS, '    if units <= block:\n', '    if True:\n', 1)),
        ('the copy not ordered ahead of the reads',
         sub(SHARDS, '    if copied:\n        b.barrier(Participants.BLOCK)\n', '', 1)),
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
        ('a value begins at its mark, not at its first write behind it',
         sub(ALLOC, '                    self.begins[id(s)] = hit',
             '                    pass', 1)),
        ('a value begins at a write that reads it',
         sub(ALLOC, '                hit = pending & defs & ~uses',
             '                hit = pending & defs', 1)),
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
        ('the block\'s buffers share bytes without being asked',
         sub(ALLOC, '    tracked = (MULT, BLOCK) if share_block else (MULT,)\n',
             '    share_block = True\n    tracked = (MULT, BLOCK)\n', 1)),
        ('the block\'s buffers shared with no occupancy of their own',
         sub(ALLOC, '    live = _Liveness(buffers, value_key, single, tracked)',
             '    live = _Liveness(buffers, value_key, single)', 1)),
        ('a block buffer held for the body shares bytes',
         sub(ALLOC, "        if b.arena in tracked and any(w.attr('lifetime') == 'body'",
             "        if b.arena == MULT and any(w.attr('lifetime') == 'body'", 1)),
        ('the block\'s arena ends where nothing shared it',
         sub(ALLOC, '        end_block = _by_occupancy(',
             '        _by_occupancy(', 1)),
    ]),

    # The block's buffers sharing bytes by occupancy (`share_block_buffers`):
    # asked for, only with the barrier placement behind the allocator, which
    # fences a write over the bytes a dead buffer was read from.
    'blockshare': ('tests/test_block_buffer_sharing.py tests/test_pir_allocate.py', [
        ('the option never reaches the allocator',
         sub(GEN, '        share_block=self._share_block_buffers())',
             '        share_block=False)', 1)),
        ('the pass drops what it was asked',
         sub(PIPE, '                                        share_block=self._share_block)',
             '                                        share_block=False)', 1)),
        ('shared without the barrier placement that fences the bytes',
         sub(GEN, '    return bool(options.share_block_buffers\n                and options.enable_sync_block_opt)',
             '    return bool(options.share_block_buffers)', 1)),
        ('a write over bytes another buffer was read from left unfenced',
         sub(BARRIERS, '                if _overlap(ext, rext):',
             '                if ext == rext:', 1)),
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
         subs(TRANSFERS, ('retires(s) or s.op in', 's.op in'),
              ('retires(x) or x.op in', 'x.op in'))),
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
         sub(MOVE, '        waits = [x for x in body if retires(x) and x.waited is not None\n'
                   '                 and x.waited.id in tokens]',
             '        waits = []', 1)),
        ('the comment saying what a wait retires left behind',
         sub(MOVE, '            while j >= 0 and neutral(body[j]) and id(body[j]) not in mine:',
             '            while False:', 1)),
    ]),

    # A merged run's loop: the selection its counter makes, a value the
    # bindings read; what the loop shares taken out of it, and the names its
    # body declares where whatever reads them sees them.
    'runs': ('tests/test_rolling.py::test_the_counter_selects_through_a_value '
             'tests/test_rolling.py::test_the_loop_runs_every_iteration '
             'tests/test_rolling.py::test_an_atomic_destination_starts_from_zeros '
             'tests/test_rolling.py::test_what_every_iteration_shares_leaves_the_loop '
             'tests/test_rolling.py::test_a_window_the_loop_fills_is_declared_ahead_of_it '
             'tests/test_rolling.py::test_a_merged_run_computes_what_the_list_written_out_does '
             'tests/test_rolling.py::test_a_merged_accumulation_is_read_back_after_the_loop '
             'tests/test_rolling.py::test_a_run_reads_its_table_and_its_factor_inside_the_loop', [
        ('the select chain left a wall',
         sub(PTR, '    self._value = (writer.uid, self._declare_chain(writer, decl, counter,\n'
                  '                                                   datatype, stars))',
             "    self._declare(writer, f'{decl} = ' + self._chain(\n"
             "        lambda m: self._member(m, datatype, stars), self._variant) + ';')", 1)),
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
    # Windows and runs as block reads (`pir.block_reads`), and runs under SPMD.
    'blockreads': ('tests/test_block_reads.py tests/test_esimd_emitter.py', [
        ('the read of a window behind its first element',
         sub(BLOCKS, '            if head is not None:\n'
                     '                out.append(head)\n'
                     '            part = members.get(id(s))\n'
                     '            if part is not None:\n'
                     '                out.append(part)\n',
             '            part = members.get(id(s))\n'
             '            if part is not None:\n'
             '                out.append(part)\n'
             '            if head is not None:\n'
             '                out.append(head)\n', 1)),
        ('a window\'s elements counted from zero',
         sub(BLOCKS, '                          tuple((x, f - lo) for x, f in st[\'loads\'])))',
             '                          tuple((x, f) for x, f in st[\'loads\'])))', 1)),
        ('a run sliced a lane apart',
         sub(BLOCKS, "                       tuple((x, k * w) for k, x in enumerate(ch['loads']))))",
             "                       tuple((x, k) for k, x in enumerate(ch['loads']))))", 1)),
        ('an SPMD run of three reads',
         sub(BLOCKS, '            size = 1\n'
                     '            while size * 2 <= count:\n'
                     '                size *= 2\n',
             '            size = count\n', 1)),
        ('a predicated read in an SPMD run',
         sub(BLOCKS, '        if not _single_read(x) or len(x.args) != 2:\n',
             '        if x.op != Op.LOAD or len(x.args) != 2:\n', 1)),
        ('a store to its buffer not parting an SPMD run',
         sub(BLOCKS, '        if x.op == Op.STORE and x.args and is_buffer(x.args[0]):\n'
                     '            close(buffer_name(x.args[0], names))\n'
                     '            continue\n'
                     '        if _run_ends(x, MemSpace.GLOBAL) or x.regions:',
             '        if x.op == Op.STORE and x.args and is_buffer(x.args[0]):\n'
             '            continue\n'
             '        if _run_ends(x, MemSpace.GLOBAL) or x.regions:', 1)),
        ('the SPMD run read at a lane\'s own address',
         sub(BLOCKS, '    head = replace(first, target=(vec,), args=(first.args[0], start),',
             '    head = replace(first, target=(vec,),', 1)),
        ('an SPMD run where the sub-group is not the multiplication',
         sub(EMIT, '                or target.sub_group_width(lanes) != lanes):\n',
             '                ):\n', 1)),
    ]),

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
         sub(BUILD, '        b.barrier(Participants.BLOCK if who is Participants.ROLE\n'
                    '                  else participants, threads=threads)\n'
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
         sub(BARRIERS, "        if completes_on_barrier(s):\n",
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
         sub(BUILD, '        if barrier is not None:\n'
                    '            return self.emit(engines.wait_phase(barrier, tokens, accesses))\n',
             '', 1)),
        ('a bulk copy in the pipeline of cp.async',
         sub(TRANSFERS, "    if any(x.op is Op.COPY_ASYNC and x.counter == 'bulk'\n"
                        '           for x in walk_stmts(tuple(t.pieces))):\n'
                        '        return BULK\n', '', 1)),
        ('a drain waiting on no cursor',
         sub(EMIT, "                texts += [f'tensorforge::bulk_wait({c});'\n"
                   '                          for c in self._bulk_cursors]\n',
             '                pass\n', 1)),
        ('a channel kept by an attempt that was discarded',
         sub(BUILD, '        self.on_rollback(undo)\n        return obj\n',
             '        return obj\n', 1)),
    ]),

    # AMD's LDS-DMA: its waits written on vmcnt, the copies it can write.
    'ldsdma': ('tests/test_pir_engines.py', [
        ('a wait for the DMA counting the loads into registers too',
         sub(ASYNCMEM, "        return s.with_attr('count', pinned)\n",
             "        return s.with_attr('count', unified)\n", 1)),
        ('a predicated copy counted behind a wait',
         sub(ASYNCMEM, "    return s.predicate is None and engines.one_to_one(s.attr('engine'))\n",
             "    return engines.one_to_one(s.attr('engine'))\n", 1)),
        ('any destination taken for the lane past a base',
         sub(ENGINES, '    if len(dst) != 1 or not lane_linear(dst[0], defs):\n',
             '    if len(dst) != 1:\n', 1)),
        ('a guard that leaves out the first lane taken',
         sub(ENGINES, '        if not isinstance(cond, tuple) and not lane_prefix(cond, defs):\n',
             '        if False:\n', 1)),
        ('a copy of a narrow multiplication taken',
         sub(ENGINES, '    if threads % wave:\n',
             '    if False:\n', 1)),
        ('a drain in a body of DMA copies written as nothing',
         sub(EMIT, "            if cls == 'all' and self._copy_counter is not None:\n",
             '            if False:\n', 1)),
        ('AMD staging through the DMA whatever the option',
         sub(TARGET, "                or not getattr(options, 'amd_async_copies', False)\n",
             '', 1)),
        ('the interpreter landing every copy at any wait',
         sub(ORACLE, '            keep = tracker.vmcnt_of(stmt)\n',
             '            keep = 0\n', 1)),
        ('the interpreter writing each lane at its own address',
         sub(ORACLE, "        base = base + int(first.ev(dm.group('di')))\n",
             "        base = base + int(self.ev(dm.group('di'))) - self.lane % Races.DMA_WAVE\n", 1)),
        ('the interpreter landing no copy at a block barrier',
         sub(ORACLE, '        if kind == \'barrier\' and tracker.drains(stmt):\n',
             '        if False:\n', 1)),
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
        ('a binding a kept binding reads dropped',
         sub(WRAP, '    pending = list(read)\n', '    pending = []\n', 1)),
        ('the tail fetches the element the iteration computes',
         sub(WRAP, '        tail_copy.given(l.k, l.next)',
             '        tail_copy.given(l.k, l.k)', 1)),
        ('the distance ignored',
         sub(WRAP, '            if count.get(key, 0) < (distance if asked is None else asked[1]):',
             '            if count.get(key, 0) < (1 if asked is None else asked[1]):', 1)),
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

    # An element's entry of the index list read once and carried across the
    # batch loop: the next element's entry carried, the peel's starting the
    # loop, nothing carried that nothing reads, a stack's elements each their
    # own, the entry after next read once for the staged pointers -- and the
    # switch that turns all of it off.
    'carryindex': ('tests/test_carry_element_index.py -n 6', [
        ("the element's own entry carried instead of the next one's",
         sub(WRAP, '            yields.append(ahead)\n',
             '            yields.append(held)\n', 1)),
        ("the loop started with the next element's entry, not the peel's",
         sub(WRAP, '            inits.append(first)\n',
             '            inits.append(ahead)\n', 1)),
        ('an entry carried where nothing left in the body reads it',
         sub(WRAP, '            if (not plain or id(s) not in alive or first is None\n',
             '            if (not plain or first is None\n', 1)),
        ("a stack's elements reading its first element's entry",
         sub(BATCHLOOP, 'GetElementPtr.list_entry(stacking.element(k, e)),',
             'GetElementPtr.list_entry(stacking.element(k, 0)),', 1)),
        ('the entry after next read again for every pointer staged',
         sub(WRAP, '            fetch = ahead_copy.stmts(_unseen(reads, ahead_copy), named=True)\n',
             '            fetch = ahead_copy.stmts(reads, named=True)\n', 1)),
        ('the switch ignored',
         sub(BATCHLOOP, "                              .carry_element_index) == 'never'\n",
             "                              .carry_element_index) == 'off'\n", 1)),
    ]),

    # Whether the loop carries an entry (`pir.entries`): the rule kept, the
    # peak a stretch and not a statement, the reader looked for through the
    # bindings and through a store's symbol.
    'carrydistance': ('tests/test_carry_element_index.py -n 6', [
        ('every entry the body reads carried, the rule ignored',
         sub(WRAP, "            if l.carry == 'auto':\n",
             "            if False:\n", 1)),
        ('the peak taken for the one statement at the maximum',
         sub(ENTRIES, 'SCARCE = 0.9\n', 'SCARCE = 1.0\n', 1)),
        ('a binding taken for the reader of the entry',
         sub(ENTRIES, '        if _binds_address(x):\n', '        if False:\n', 1)),
        ("a store's destination not read through its symbol",
         sub(ENTRIES, '    if names and any(not isinstance(a, Value)\n',
             '    if False and any(not isinstance(a, Value)\n', 1)),
        ('the stretch judged by the pressure at its reader alone',
         sub(ENTRIES, '        return max(self._curve[:j + 1]) < self._scarce\n',
             '        return self._curve[j] < self._scarce\n', 1)),
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
                     "            return [(lanes, self.interps, {})]\n",
             "        if reach[0] in ('block', 'warp'):\n"
             "            return [(lanes, self.interps, {})]\n", 1)),
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
         sub(ENGINES, '        if sync_mult is not None and not sync_mult(obj.threads):',
             '        if False:', 1)),
        ('the finished body not weighed against the block',
         sub(GEN, '    self._check_barriers(body, index)\n', '', 1)),
        ('a block of any size taken for one the target meets apart',
         sub(PASSES, '    if separable is None or mults_per_block <= separable:\n'
                     '        return []\n',
             '    return []\n', 1)),
        ("a block barrier taken for a multiplication's",
         sub(ENGINES, '    if any(obj.per_mult for obj in objects(body)):',
             '    if any(True for obj in objects(body)):', 1)),
        ('a barrier inside a loop not seen',
         sub(ENGINES, '    for s in walk_stmts(tuple(body)):\n'
                      '        if s.op in REGISTRY:',
             '    for s in tuple(body):\n'
             '        if s.op in REGISTRY:', 1)),
        ('a fold over every lane taken for one over some',
         sub(CROSSLANE, '        if width >= self._num_threads and not writer._explicit_simd():',
             '        if False:', 1)),
        ('a fold over some lanes taken for one over every lane',
         sub(CROSSLANE, '        if width >= self._num_threads and not writer._explicit_simd():',
             '        if not writer._explicit_simd():', 1)),
        ('a fold over every lane as lane-varying as what it folds',
         sub(BUILD, '                       uniform=max(Uniformity.MULT, _join((value,))),\n',
             '                       uniform=_join((value,)),\n', 1)),
    ]),

    # A merged run's members streamed through the shards: the copy written
    # out pass by pass, the lanes past the end of its last pass back on their
    # first, and the shards as long as the members are stored, not as their
    # box.
    'members': ('tests/test_stage_members.py', [
        ("a member's storage taken for its box",
         sub(GEN, '                extents=self._operand_extents(),\n', '', 1)),
        ('the last pass reaching past the end of the copy',
         sub(SHARDS, "            u = b.op('select', INDEX, b.op('lt', BOOL, u, units, hint='s'),\n"
                     "                     u, flat, hint='s')\n", '            pass\n', 1)),
        ('the copy looped by lane, a round trip a pass',
         sub(SHARDS, 'UNROLL = 16\n', 'UNROLL = 0\n', 1)),
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
         sub(STACKING, "  return f'({{{1 + e}}} ? {k} * {{0}} + {e} : ({first}))'",
             "  return f'({k} * {{0}})'", 1)),
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
         sub(GEN, "    stacks = stacking.count(head, self._stack)\n",
             "    stacks = stacking.count(head, 1)\n", 1)),
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
         sub(TUNING, "    return (cliff, issue, -resident * _stack_of(gen),",
             "    return (cliff, issue, -resident,", 1)),
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
         sub(GEN, '      if (symbol.obj.addressing.kind in (Addressing.PTR_BASED,\n'
                  '                                         Addressing.INDEXED_POINTER)\n',
             '      if (False\n', 1)),
        ('every pointer promised, the batch\'s own operand too',
         sub(GEN, '      if (symbol.obj.addressing.kind in (Addressing.PTR_BASED,\n'
                  '                                         Addressing.INDEXED_POINTER)\n',
             '      if (symbol.obj.addressing.kind != Addressing.SCALAR\n', 1)),
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
         sub(GEN, '    if self._wave_in_step() or (self._num_threads < wave and asked):',
             '    if self._wave_in_step():', 1)),
        ('every shared wave grouped, asked or not',
         sub(GEN, '    asked = self._context.get_user_options().wave_group_traversal\n',
             '    asked = True\n', 1)),
        ('a rotated start grouped as well',
         sub(GEN, '      # stays per row.\n      if start != self._get_2d_block_id():\n        return 1\n',
             '      # stays per row.\n', 1)),
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
        ('what the element stores ahead of reading it back asked for',
         sub(PREFETCH, '                    written.get(_buffer(a.base), at) < at\n',
             '                    False\n', 1)),
        ('what the element stores asked for until it is stored',
         sub(PREFETCH, '                    written.get(_buffer(a.base), at) < at\n',
             '                    _buffer(a.base) in written\n', 1)),
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

    # `verify` bounds an access by the array it reaches, and a name only by
    # the largest array of it.
    'bounds': ('tests/test_pir_bounds.py', [
        ('an array bounded by the largest array of its name',
         sub(PASSES, '            name, size = str(base), sizes.get(base.id)\n',
             '            name, size = str(base), named.get(base.hint)\n', 1)),
        ('a name bounded by the last array of it',
         sub(PASSES, '                named[array.hint] = max(named.get(array.hint, 0), count)\n',
             '                named[array.hint] = count\n', 1)),
        ('an access through a symbol left unbounded',
         sub(PASSES, '            size = named.get(name)\n',
             '            size = None\n', 1)),
        ('a store bounded by its value rather than its index',
         sub(PASSES, '            base, index = (s.args + (None,) * 3)[0], (s.args + (None,) * 3)[2]\n',
             '            base, index = (s.args + (None,) * 3)[0], (s.args + (None,) * 3)[1]\n', 1)),
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
        ('a value without axes written to shared memory as text',
         sub(SYM, '    axisless = (not structured and not named',
             '    axisless = (False and not structured and not named', 1)),
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
        ('a presplit order asks for one part',
         sub(Path('src/tensorforge/backend/instructions/compute/primitives/nvidia.py'),
             '        return SplitOrder(order, parts=2)', '        return SplitOrder(order, parts=1)', 1)),
        ('a narrow multiplication with its own A offered the matrix path again',
         sub(Path('src/tensorforge/backend/instructions/compute/primitives/nvidia.py'),
             '    if shape.threads < WAVE and not shape.a_uniform:\n        return frozenset()\n',
             '', 1)),
        ('the depth floor read from the module whatever the build asks',
         sub(Path('src/tensorforge/backend/instructions/compute/primitives/nvidia.py'),
             '                         shape.depth, min_depth(ctx))',
             '                         shape.depth)', 1)),
        ('an operator read over a window staged per element again',
         sub(Path('src/tensorforge/backend/instructions/compute/primitives/nvidia.py'),
             '    8.8; in double the viscoelastic flux 40.5 -> 37.2, the rest within 1 %.\n'
             '    """\n    return True',
             '    8.8; in double the viscoelastic flux 40.5 -> 37.2, the rest within 1 %.\n'
             '    """\n    return False', 1)),
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
             '                         shape.depth, min_depth(ctx))\n'
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

    # Which register arrays raw text names, to the pressure: the words of the
    # text outside its comments.
    'rawnames': ('tests/test_pressure_offsets.py::'
                 'test_raw_text_names_an_array_by_its_whole_name', [
        ('a name found inside a longer word',
         sub(PASSES, '                if name in words:',
             '                if name in code:', 1)),
        ('a comment naming an array',
         sub(PASSES,
             r"            code = re.sub(r'//[^\n]*|/\*.*?\*/', '', st.text, flags=re.S)",
             '            code = st.text', 1)),
        ('a word of letters only',
         sub(PASSES, r"_WORDS = re.compile(r'\w+')",
             r"_WORDS = re.compile(r'[A-Za-z_]+')", 1)),
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
         sub(BARRIERS, '        if level is not None:\n            return state.fence(level)',
             '        if level is not None:\n            return state', 1)),
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
         sub(CARRY, '        if closed or not {g for g in guards\n'
                    '                          if not (g[1] and g[0] in self._masks)}:',
             '        if True:', 1)),
        ('a guard spelled in text taken for an element\'s mask in the trips',
         sub(CARRY, '                          if not (g[1] and g[0] in self._masks)}:',
             '                          if not g[1]}:', 1)),
        ('a guard spelled in text taken for an element\'s mask behind the loop',
         sub(CARRY, '        masked = {g for g in c.guards if g[1] and g[0] in self._masks}',
             '        masked = {g for g in c.guards if g[1]}', 1)),
        ('an element\'s mask taken to hold where the body does not keep it',
         sub(FLOW, '    return frozenset(masks) if kept(body, False) else frozenset()',
             '    return frozenset(masks)', 1)),
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
        ('an element seeded by the first store ahead of the loop, not the last',
         sub(CARRY, '                    last[(lane, element)] = k\n',
             '                    last.setdefault((lane, element), k)\n', 1)),
        ('every store ahead of the loop seeding',
         sub(CARRY, '        seeding = sorted(set(last.values()))\n',
             '        seeding = list(range(len(traced)))\n', 1)),
        ('a store ahead seeding what the trips do not reach',
         sub(CARRY, "                if r is None:\n"
                    "                    raise _Refused('the code ahead of the loop stores what '\n",
             "                if r is None:\n"
             "                    continue\n"
             "                    raise _Refused('the code ahead of the loop stores what '\n", 1)),
        ('the store ahead of a lane\'s own buffer always dropped',
         sub(CARRY, '        elif (again or not read_behind) and self._movable(out, at):',
             '        elif self._movable(out, at):', 1)),
        ('the back edge of a loop around the run not followed',
         sub(CARRY, '        if _repeats(holder) and _again(path, level, key):',
             '        if False:', 1)),
        ('the carry not run at all',
         sub(CARRY, "            if counted(s) and s.attr('rolled'):\n",
             "            if False:\n", 1)),
        ('a statement of a space nobody knows taken to reach no buffer',
         sub(CARRY, '        found = set(self._anywhere)\n',
             '        found = set()\n', 1)),
        ('a statement of no buffer of the space taken to reach none of it',
         sub(CARRY, '        found.update(self._unnamed.get(access.space, ()))\n',
             '', 1)),
        ('a buffer no register image is stored into asked all the same',
         sub(CARRY, '            if registers:\n                self._from_registers |= written\n',
             '            self._from_registers |= written\n', 1)),
        ('the statements of a transfer taken one by one',
         sub(CARRY, '                self._transfers.setdefault(t, []).append(i)\n',
             '                pass\n', 1)),
        ('a declaration in the trip unseen',
         sub(CARRY, '                    self.declared.add(id(x))\n',
             '                    pass\n', 1)),
        ('a statement in the trip that says nothing of what it touches unseen',
         sub(CARRY, '                    self.unsaid = True\n',
             '                    pass\n', 1)),
        ('the forward of memory behind the carry',
         sub(PIPE, "    if early_forward is not None:\n"
                   "        pm.add(early_forward)\n"
                   "        for name, fn in (('forward-memory-fold', fold),\n"
                   "                         ('forward-memory-dce', dce)):\n"
                   "            pm.add(Rewrite(name, fn, preserves=('flat',)))\n"
                   "    if carry is not None:\n"
                   "        pm.add(carry)\n"
                   "        pm.add(Rewrite('carry-dce', dce, preserves=('flat',)))\n",
             "    if carry is not None:\n"
             "        pm.add(carry)\n"
             "        pm.add(Rewrite('carry-dce', dce, preserves=('flat',)))\n"
             "    if early_forward is not None:\n"
             "        pm.add(early_forward)\n"
             "        for name, fn in (('forward-memory-fold', fold),\n"
             "                         ('forward-memory-dce', dce)):\n"
             "            pm.add(Rewrite(name, fn, preserves=('flat',)))\n", 1)),
    ]),
    # A buffer of the arena every lane reads only what it wrote itself, in
    # registers over the statements that touch it.
    'keepown': ('tests/test_keep_own.py', [
        ('what a lane reads before it writes it taken for its own',
         sub(CARRY, "                if exposed:\n"
                    "                    raise _Refused('a lane reads what it did not write there '\n",
             "                if False:\n"
             "                    raise _Refused('a lane reads what it did not write there '\n", 1)),
        ('a statement that says nothing of what it touches taken for none',
         sub(CARRY, '        found = (any(a.space is MemSpace.UNKNOWN or (\n',
             '        found = (False and any(a.space is MemSpace.UNKNOWN or (\n', 1)),
        ('text that touches it taken for a load or a store',
         sub(CARRY, "                if x.op not in (Op.LOAD, Op.STORE):\n"
                    "                    raise _Refused(f'a {x.op} touches it')\n"
                    "                if any(a.base is not base for a in mine):\n",
             "                if any(a.base is not base for a in mine):\n", 1)),
        ("an element another lane writes taken for a lane's own",
         sub(CARRY, '        if any(len(touched[e]) > 1 for e in written):',
             '        if False:', 1)),
        ('the trips of a loop around it not asked',
         sub(CARRY, "                if (owner is None or owner.op is not Op.FOR\n"
                    "                        or not counted(owner)):\n",
             "                if True:\n", 1)),
        ('a loop that does not hold it asked all the same',
         sub(CARRY, '        for x in ((xs[i] for i in where(xs)) if where is not None else xs):\n'
                    '            if x.regions:\n'
                    '                if where is None and not any(\n',
             '        for x in xs:\n'
             '            if x.regions:\n'
             '                if where is None and not any(\n', 1)),
        ('the declaration of the buffer left',
         sub(CARRY, '                     marks | {id(alloc)})\n', '                     marks)\n', 1)),
        ('keeping a buffer no lane hands on off on AMD',
         sub(TARGET, '                               read_across=True,\n'
                     '                               keep_own=True),',
             '                               read_across=True),', 1)),
        ('keeping a buffer no lane hands on off on NVIDIA',
         sub(TARGET, '        placement=VendorPolicy(preload_operands_into_registers=True,\n'
                     '                               keep_own=True),',
             '        placement=VendorPolicy(preload_operands_into_registers=True),', 1)),
        ('keeping a buffer no lane hands on asked of no vendor',
         sub(OPTIONS, "        rule=lambda target: target.prefs.placement.keep_own,",
             "        default=False,", 1)),
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
         sub(FLOW, '                    if piece.conds <= conds:\n'
                   '                        prior = piece.writer\n'
                   '                        break\n'
                   '                    certain = False\n',
             '                    prior = piece.writer\n'
                   '                    break\n', 1)),
        ('a write of a register under a guard taken for one that replaces what was there',
         sub(FLOW, '                    if layer.conds <= conds:\n'
                   '                        prior = layer.writer\n'
                   '                        break\n'
                   '                    certain = False\n',
             '                    prior = layer.writer\n'
                   '                    break\n', 1)),
        ('what text reads not taken as read',
         sub(FLOW, '                            for layer in table.get(where, ()):\n'
                   '                                self.readers[layer.writer].append(\n'
                   '                                    (n, self.full))\n'
                   '                                self.escaped.add(layer.writer)\n',
             '                            for layer in table.get(where, ()):\n'
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
        ('the loads of memory not taken ahead of the atomics',
         sub(PIPE, '    if early_forward is not None:\n', '    if False:\n', 1)),
    ]),
    # Every register image of a tensor in the lanes of its rows: the lead
    # loop's origin, the stagings' shifts, and the copies that hold their
    # rows elsewhere staged again.
    'lanes': ('tests/test_canonical_lanes.py', [
        ('the lead loop\'s origin left at 0',
         sub(MLB, '    return (self._dest_obj.offset[0] if self._dest_obj.offset else 0) % span\n',
             '    return 0\n', 1)),
        ('a staging that absorbs the whole lower bound',
         sub(MLB, '    shift[lead_pos] = lower - (lower - residue) % span\n',
             '    shift[lead_pos] = lower\n', 1)),
        ('an image of other rows read where it is',
         sub(MLB, '    return self._in_loop_lanes(i, shift)\n', '    return True\n', 1)),
        ('a bias staged in other lanes added onto',
         sub(MLB, '    return (offset - shift - theta) % span == 0\n',
             '    return True\n', 1)),
        ('a transposed image of other rows staged into registers',
         sub(MLB, '    if (placement is Placement.REGISTER and has_lead_dim and transpose\n'
                  '            and not self._in_loop_lanes(i, 0)):\n'
                  '      placement = Placement.SHARED\n', '', 1)),
        ('the accumulator read where its range holds the block\'s first row alone',
         sub(EPILOGUE, '                return start < hi_i and lo_i < start + var._block * var.width\n',
             '                return lo_i <= start < hi_i\n', 1)),
    ]),
    # What a multiplication's register image holds outside its rows, said
    # to the passes, and gone before the emitter.
    'promises': ('tests/test_promised_zeros.py tests/test_forward.py', [
        ('no promise behind a multiplication',
         sub(MLB, '    promise = self._promised_zeros(compute)\n',
             '    promise = None\n', 1)),
        ('a promise over the rows the multiplication computed',
         sub(PROMISE, "            past = writer.op('ge', BOOL, lane, valid, hint='o')\n",
             '            past = None\n', 1)),
        ('a promise left to the emitter',
         sub(PIPE, "    pm.add(Rewrite('promises', drop_promises, preserves=('flat',)))\n",
             '', 1)),
        ('a promise taken for a write nobody can place',
         sub(FLOW, '        elif s.op in (Op.STORE, Op.MARK) and len(s.accesses) == 1:\n',
             '        elif s.op == Op.STORE and len(s.accesses) == 1:\n', 1)),
    ]),
    # What a window computed on every lane leaves beside it stays out of the
    # destination's box: before the window always, past it where an operand
    # reads no zero there.
    'beside': ('tests/test_full_lane_tails.py::test_what_a_window_leaves_out_adds_nothing_to_the_destination '
               'tests/test_full_lane_tails.py::test_an_operand_reads_zero_past_its_window_where_nothing_lies_there '
               'tests/test_full_lane_tails.py::test_the_lanes_beside_a_window_computed_on_every_lane_are_told', [
        ('a register image taken to hold zeros past rows it does not end with',
         sub(MULTILINEAR, '        return sym.data_view.get_bbox().upper()[axis] <= end\n',
             '        return True\n', 1)),
        ('an image nobody vouched for taken to hold zeros past its rows',
         sub(MULTILINEAR, "        if not getattr(sym, 'zero_past', False):\n"
                          '            return False\n', '', 1)),
        ('the accumulator read whole beside its range',
         sub(EPILOGUE, '                valvar = held_to_range(valvar, varlist, btype)\n',
             '', 1)),
        ('the lanes before the window taken for zeros',
         sub(MULTILINEAR, '            out[i] = (True, past)\n',
             '            out[i] = (False, past)\n', 1)),
        ('the lanes past the window taken for zeros',
         sub(MULTILINEAR, '            out[i] = (True, past)\n',
             '            out[i] = (True, False)\n', 1)),
        ('the lane before the window\'s first row taken into it',
         sub(EPILOGUE, "                    bounds.append(('ge', acc.lower()[i] - start))\n",
             "                    bounds.append(('ge', acc.lower()[i] - start - 1))\n", 1)),
    ]),
    # One register, one read: the forwarding spells a register by the value
    # that declares it, the builder by its symbol, and `load_cse` keys both
    # by the buffer where one index is the whole address.
    'loadkey': ('tests/test_alias_identity.py', [
        ('a register read through its value taken for another than through its symbol',
         sub(PASSES, "        key = (op, (('buf', id(s.accesses[0].base)),) + args[1:]) + key[2:]\n",
             '', 1)),
        ('a register read at more than one index taken by its buffer',
         sub(PASSES, '    if (s.op == Op.LOAD and len(s.args) == 2 and len(s.accesses) == 1\n',
             '    if (s.op == Op.LOAD and len(s.accesses) == 1\n', 1)),
    ]),
    # Results handed between the operations of a list through memory: each
    # stored behind the operation that computes it, the cover counted behind
    # the forward that takes the round trips back, and the bias loaded
    # behind the product.
    'memsem': ('tests/test_memory_semantics.py tests/test_memory_ordering.py '
               'tests/test_canonical_lanes.py::test_the_shape_computes_what_numpy_does', [
        ('a product\'s result stored nowhere',
         sub(MLB, '      elif dest_symbol.stype == SymbolType.Global:\n'
                  '        self._instructions.append(StoreRegToGlb(',
             '      elif dest_symbol.stype == SymbolType.Global:\n'
             '        (StoreRegToGlb(', 1)),
        ('a product\'s temporary stored nowhere',
         sub(MLB, '\n      self._instructions.append(self._store_whole(dest_symbol))\n',
             '\n      pass\n', 1)),
        ('a pointwise result stored nowhere',
         sub(OPB, '        self._home_store = lambda: StoreRegToShr(\n',
             '        lambda: StoreRegToShr(\n', 1)),
        ('a result laid over its own box',
         sub(MLB, '    view = dest_symbol.data_view\n    if view is None:\n',
             '    view = None\n    if view is None:\n', 1)),
        ('the cover counted ahead of the forward of memory',
         subs(PIPE, ("    pm.add(Rewrite('flatten', flatten_scopes, provides=('flat',)))\n",
                     "    pm.add(Rewrite('flatten', flatten_scopes, provides=('flat',)))\n"
                     "    if move is not None and move.cover > 0:\n"
                     "        pm.add(Rewrite('cap', move.cap, requires=('flat',),\n"
                     "                       preserves=('flat',)))\n"),
              ("    if move is not None and move.cover > 0:\n"
               "        pm.add(Rewrite('cap', move.cap, requires=('flat',),\n"
               "                       preserves=('flat',)))\n"
               "    if atomics is not None:\n",
               "    if atomics is not None:\n"))),
        ('the bias loaded ahead of the product',
         sub(MLB, "      if (any(isinstance(x, GlbToRegLoader) for x in staged)\n",
             "      if (False and any(isinstance(x, GlbToRegLoader) for x in staged)\n", 1)),
        ('the bias staged and never loaded',
         sub(MLB, "    self._instructions.extend(getattr(self, '_bias_loads', ()))\n"
                  "    self._bias_loads = []\n", "", 1)),
        ('no forward of memory ahead of the cover without atomics',
         sub(ABSTR, '      early = pir.Forward(builder.scratch, report=carries,\n'
                    '                          registers=False, fetches=True, **carry)\n',
             '      early = pir.Forward(builder.scratch, report=carries,\n'
             '                          registers=False, fetches=True,\n'
             '                          **carry) if added else None\n', 1)),
        ('an accumulation says its destination holds nothing ahead of it',
         sub(GEN, '                  and id(sym) not in read\n', '', 1)),
    ]),
    # Which lanes of a value anything reads, and the loads taken where the
    # lanes they would get wrong are none of those.
    'demand': ('tests/test_demand.py', [
        ('a broadcast taken to read the lane it gives',
         sub(DEMAND, '     lambda block, sub, lane: (block, lambda p: lane * sub + p % sub)),',
             '     lambda block, sub, lane: (block, lambda p: p)),', 1)),
        ('the operand an FMA adds taken for the one it moves',
         sub(DEMAND, "    (re.compile(r'tensorforge::fmacdpp(4|8|16)<\\s*(\\d+)\\s*>$'), 1, False,",
             "    (re.compile(r'tensorforge::fmacdpp(4|8|16)<\\s*(\\d+)\\s*>$'), 2, False,", 1)),
        ('the lanes a broadcast is read in not handed on',
         sub(DEMAND, '            wanted = self.lanes(u.target[0], env, depth + 1)\n',
             '            wanted = 0\n', 1)),
        ('a call it does not know taken to read no lane',
         sub(DEMAND, '        if known is None:\n            return full\n',
             '        if known is None:\n            return 0\n', 1)),
        ('a text it does not know taken to read no lane',
         sub(DEMAND, '                if m is None:\n                    return full\n',
             '                if m is None:\n                    continue\n', 1)),
        ('a value stored to memory taken for one stored to registers',
         sub(DEMAND, '                or u.accesses[0].space is not MemSpace.REGISTER\n'
                     '                or len(u.args) < 2', '                or len(u.args) < 2', 1)),
        ('a register an access nothing can place reads taken for one read back',
         sub(DEMAND, '                    # Anything but a load back: an access the flow cannot\n'
                     '                    # place, which reads every element it may reach.\n'
                     '                    return full\n', '                    continue\n', 1)),
        ('a register read back taken to read nothing',
         sub(DEMAND, "                out |= self.lanes(x.target[0], flow.insts[r].env(),\n"
                     "                                  depth + 1) & m\n",
             "                pass\n", 1)),
        ('the multiplicand of an FMA taken to read no lane',
         sub(DEMAND, '            return self._accumulated(u, env, depth)\n'
                     '        if len(u.args) <= operand or u.args[operand] is not v:\n',
             '            return 0\n'
             '        if len(u.args) <= operand or u.args[operand] is not v:\n', 1)),
        ('any call on an accumulator taken for an FMA that adds into it',
         sub(DEMAND, '        known = lane_map(u.attr(\'callee\'))\n'
                     '        return (known is not None and not known[1] and len(u.args) == 3\n',
             '        return True\n'
             '        return (known is not None and not known[1] and len(u.args) == 3\n', 1)),
        ('the lanes a predicate leaves out taken for unread unasked',
         sub(FORWARD, '        if s.predicate is None:\n            return True\n        execs = flow.executions(s)\n',
             '        return True\n        execs = flow.executions(s)\n', 1)),
        ('no lane of a load taken to be read',
         sub(FORWARD, '        return self.demand.lanes(s.target[0], self.flow.insts[n].env())\n',
             '        return 0\n', 1)),
    ]),
    # A load taken from where its own lane's store took the value, a store
    # nothing reads dropped, and the masks taken to hold for it.
    'forward': ('tests/test_forward.py', [
        ('an operand every product loads, loaded again behind the wrap',
         sub(ABSTR, '                          registers=False, fetches=True, **carry)\n',
             '                          registers=False, fetches=False, **carry)\n', 1)),
        ('a select over the lanes told apart under the explicit vector alone',
         sub(FLOW, "        if (d is not None and d.op == 'select'\n",
             "        if (self.simd and d is not None and d.op == 'select'\n", 1)),
        ('a register written since taken for what it held',
         sub(FLOW, '            if not layer.conds <= conds or layer.writer != writer:\n'
                   '                return False\n', '', 1)),
        ('another lane\'s store taken for the lane\'s own',
         sub(FLOW, '                if not r.own and not self._same_on_every_lane(o):\n'
                   '                    continue\n', '', 1)),
        ('a value that differs between the lanes taken for one the same on all',
         sub(FLOW, '        if v.uniformity < Uniformity.MULT:\n'
                   '            return False\n', '', 1)),
        ('a read of registers spread nobody knows how taken for the same on all',
         sub(FLOW, '            elif v.layout is None and any(\n',
             '            elif False and any(\n', 1)),
        ('what a loop carries taken for the same on all whatever a trip hands on',
         sub(FLOW, '        out = (all(self(x) for x in d.operands())\n'
                   '               and all(self(y) for y in d.regions[0].yielded))\n',
             '        out = all(self(x) for x in d.operands())\n', 1)),
        ('what a loop carries never taken for the same on all',
         sub(FLOW, '            if d.op == Op.FOR:\n                out = self._loop(d)\n',
             '            if False:\n                out = self._loop(d)\n', 1)),
        ('a number read as a truth value taken for the number',
         sub(FORWARD, "            v = b.op('ne', BOOL, v, b.const(_zero_like(v.type), v.type),\n"
                      "                     hint='fwd')\n",
             "            pass\n", 1)),
        ('a number never read as a truth value',
         sub(FORWARD, '        truth = v.type != s.target[0].type and _truth_of(s, v.type)\n',
             '        truth = False\n', 1)),
        ('a fold every lane holds alike asked of what it folds',
         sub(FLOW, "            elif d.attr('alike'):\n",
             '            elif False:\n', 1)),
        ('a register of numbers never read as truth values',
         sub(FORWARD, '        read = self._register_read(s, base, points, truth=True)\n',
             '        read = self._register_read(s, base, points)\n', 1)),
        ('what every lane stores alike read from memory all the same',
         sub(FLOW, '                if not r.own and not self._same_on_every_lane(o):\n',
             '                if not r.own:\n', 1)),
        ('a load taken where some of its lanes read something else',
         sub(FORWARD, '            if covered & need != need:\n'
                      '                return False\n', '', 1)),
        ('a value taken where it does not stand',
         sub(FORWARD, '        if (v.type != s.target[0].type and not truth) or not self._stands(v, s):',
             '        if (v.type != s.target[0].type and not truth):', 1)),
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
         sub(FLOW, '                if a.space in (MemSpace.GLOBAL, MemSpace.UNKNOWN):\n'
                   '                    if not under:\n'
                   '                        return False\n',
             '                if a.space in (MemSpace.GLOBAL, MemSpace.UNKNOWN):\n'
                   '                    pass\n', 1)),
        ('masks taken to hold where a branch turns on memory',
         sub(FLOW, '                if loads.reads(cond):\n'
                   '                    return False\n', '', 1)),
        ('a statement that stands in for others asked as a write of its own',
         sub(FLOW, '            if _stands_in(x):\n'
                   '                continue\n', '', 1)),
        ('text with a known effect taken to stand in for others',
         sub(FLOW, '    return (x.op == Op.RAWSTMT and not x.text and bool(x.accesses)',
             '    return (x.op == Op.RAWSTMT and bool(x.accesses)', 1)),
        ('a zeroed declaration taken to hold nothing known',
         sub(FLOW, "                if zeros and self._values and space is MemSpace.REGISTER:\n"
                   "                    self.origin[n] = [(mask, ('const', 0))]\n", '', 1)),
        ('a buffer nothing reaches any more left declared',
         sub(FORWARD, '        body = _strip(body, _unreached(body))\n'
                      '        mapped = self.mapped\n',
             '        mapped = self.mapped\n', 1)),
        ('a buffer something still reaches taken for unreached',
         sub(FORWARD, '            reached |= hit\n', '            pass\n', 1)),
        ('a body past the budget forwarded all the same',
         sub(FLOW, '        if self._budget < 0:\n', '        if False:\n', 1)),
        ('a number the lanes take read in a register that need not hold it',
         sub(FORWARD, '            if want & ~got:\n'
                      '                return False\n', '', 1)),
        ('a register never read for the number its lanes take',
         sub(FORWARD, '                and self._holding(next(iter(consts)), taken, lambda n, h:',
             '                and False and self._holding(next(iter(consts)), taken, lambda n, h:', 1)),
        ('a predicate dropped where the register holds something else',
         sub(FORWARD, '        if predicate is not None and self._idle(s, taken):',
             '        if predicate is not None:', 1)),
        ('a promise of zeros taken for nothing',
         sub(FLOW, "        if s.op == Op.MARK and s.attr('mark') != 'zeros':\n",
             "        if s.op == Op.MARK:\n", 1)),
        ('a mark around a clearing store taken for a write nobody can place',
         sub(FLOW, "        if s.op == Op.MARK and s.attr('mark') != 'zeros':\n"
                   "            # What a later pass is told about the order of the writes\n"
                   "            # (`clears`, `cleared`): no element changes.\n"
                   "            return\n", '', 1)),
        ('a load of memory taken from no earlier load',
         sub(FLOW, '                found += self._fetches(n, conds, taken)\n', '', 1)),
        ('an earlier load taken whatever write it found',
         sub(FLOW, '                    if (found.writer != r.writer\n'
                   '                            or not self._part(found, lane).certain):\n',
             '                    if not self._part(found, lane).certain:\n', 1)),
        ('an earlier load taken where the lane\'s own store answers',
         sub(FLOW, '            left = r.mask & ~taken\n', '            left = r.mask\n', 1)),
        ('the register of an earlier load taken whether it holds it or not',
         sub(FLOW, '                                   if m & bit\n'
                   '                                   and self._holds(cell, store, bit, conds)),\n',
             '                                   if m & bit),\n', 1)),
        ('two rounds where a chain of copies takes more',
         sub(FORWARD, 'ROUNDS = 4\n', 'ROUNDS = 2\n', 1)),
        ('the last load of an element asked before the first',
         sub(FLOW, '                    self._fetched[(r.cell_at(lane), lane)].append((load, r))\n',
             '                    self._fetched[(r.cell_at(lane), lane)].insert(0, (load, r))\n', 1)),
        ('a load taken from a load the same round takes read as that one\'s value',
         sub(FORWARD, '            while v.id in mapped:\n'
                      '                v = mapped[v.id]\n', '', 1)),
        ('a value one trip stored read in another',
         sub(FLOW, "                if o[0] == 'value' and not self._one_trip(o[1], r.writer, n):",
             '                if False:', 1)),
        ('a value asked of every loop around the load',
         sub(FLOW, '        loops = self.loops.get(id(d), ())\n',
             '        loops = self.loops.get(id(self.insts[n].stmt), ())\n', 1)),
        ('a value a loop hands on taken for one computed ahead of it',
         sub(FLOW, '        if d.op == Op.FOR and not any(t is v for t in d.target):\n',
             '        if False:\n', 1)),
        ('another lane\'s register read whether it holds the store still',
         sub(FLOW, "                    if o[0] == 'value' or self._holds(o[1], o[2], 1 << r.by,",
             "                    if True or self._holds(o[1], o[2], 1 << r.by,", 1)),
        ('lanes past one exchange read across',
         sub(FORWARD, '                        exchange=exchange is not None and not simd\n'
                      '                        and threads <= exchange\n',
             '                        exchange=exchange is not None and not simd\n', 1)),
        ('a multiplication of lanes no block of a power of two holds read across',
         sub(FORWARD, '                        and threads <= exchange\n'
                      '                        and not threads & (threads - 1), simd=simd)',
             '                        and threads <= exchange, simd=simd)', 1)),
        ('a load under a branch the lanes take apart read across',
         sub(FORWARD, '                if not flow.alike(p.cond):\n'
                      '                    return False\n', '                pass\n', 1)),
        ('a load in a loop of a count the lanes may not share read across',
         sub(FORWARD, '                if not all(flow.alike(x) for x in p.loop_bounds):\n'
                      '                    return False\n', '                pass\n', 1)),
        ('a block whose head is text taken for one every lane runs',
         sub(FORWARD, '            elif p.op != Op.RAWBLOCK or p.text:',
             '            elif p.op != Op.RAWBLOCK:', 1)),
        ('a lane reading its own store beside one lane\'s read across',
         sub(FORWARD, '        if own & ~(1 << lane):', '        if False:', 1)),
        ('the stores of several lanes read across as one lane\'s',
         sub(FORWARD, '        if len(sources) != 1 or not self._converged(s):\n'
                      '            return False\n'
                      '        lane, = sources\n',
             '        if not sources or not self._converged(s):\n'
                      '            return False\n'
                      '        lane = min(sources)\n', 1)),
        ('one lane\'s stores from two registers read across from the first',
         sub(FORWARD, "            if any(o[0] != 'reg' for _, o in origins) or len(\n"
                      "                    {o[1][0][1] for _, o in origins}) != 1:",
             "            if any(o[0] != 'reg' for _, o in origins):", 1)),
        ('one lane\'s stores of two values read across as the first',
         sub(FORWARD, '        if len({o[1].id for _, o in origins}) != 1:\n'
                      '            return False\n', '', 1)),
        ('a register every lane holds alike read across',
         sub(FORWARD, '            held = self._held_alike(origins)\n',
             '            held = None\n', 1)),
        ('a register every lane holds alike read only where the lanes go together',
         sub(FORWARD, "        if all(o[0] == 'reg' for _, o in origins):\n"
                      "            held = self._held_alike(origins)\n",
             "        if all(o[0] == 'reg' for _, o in origins) and self._converged(s):\n"
                      "            held = self._held_alike(origins)\n", 1)),
        ('a register of a lane\'s own taken for one every lane holds alike',
         sub(FORWARD, "                if (o[0] != 'value' or not flow.alike(o[1])\n",
             "                if (o[0] != 'value'\n", 1)),
        ('a register a trip before wrote taken for this trip\'s value',
         sub(FORWARD, '                        or not flow._one_trip(o[1], writer, n)):\n',
             '                        or False):\n', 1)),
        ('a named load read across',
         sub(FORWARD, "        if s.attr('extern') is not None:\n"
                      "            # Text names the load's value, and a broadcast does not keep it.\n"
                      "            return False\n", '', 1)),
        ('a value read across where it does not stand',
         sub(FORWARD, '            if value.type != s.target[0].type or not self._stands(value, s):',
             '            if value.type != s.target[0].type:', 1)),
        ('a broadcast of the multiplication\'s lanes taken for one of a lane',
         sub(FORWARD, "hint='fwd', whole=True)", "hint='fwd')", 1)),
        ('the read across the lanes off on AMD',
         sub(TARGET, '                               read_across=True,\n'
                     '                               keep_own=True),',
             '                               keep_own=True),', 1)),
        ('the read across the lanes asked of no vendor',
         sub(OPTIONS, "        rule=lambda target: target.prefs.placement.read_across,",
             "        default=False,", 1)),
        ('a broadcast of every lane not said to be alike',
         sub(BUILD, "attrs=(('crosslane', True), ('alike', True)))",
             "attrs=(('crosslane', True),))", 1)),
    ]),

    # The forward under the explicit vector: an access placed an element a
    # lane, and a load taking only the same vector.
    'vectors': ('tests/test_forward.py tests/test_esimd_emitter.py::'
                'test_a_tail_read_of_a_register_array_is_zero_past_its_lanes', [
        ('the vector placed as under SPMD',
         sub(FLOW, '        if self.simd:\n'
                   '            return self._place_vector(s, env, mask)\n', '', 1)),
        ('a vector in memory placed at one element for every lane',
         sub(FLOW, '                starts = (tuple(addr[-1] + lane for lane in lanes) if spread\n'
                   '                          else (addr[-1],) * len(lanes))\n',
             '                starts = (addr[-1],) * len(lanes)\n', 1)),
        ('a register vector placed from between two',
         sub(FLOW, '            if not spread or addr[-1] % self.threads:\n',
             '            if not spread:\n', 1)),
        ('an element every lane holds placed in a register vector',
         sub(FLOW, '            if not spread or addr[-1] % self.threads:\n',
             '            if addr[-1] % self.threads:\n', 1)),
        ('a vector of more lanes than the multiplication placed',
         sub(FLOW, '            if span > self.threads:\n'
                   '                return None\n', '', 1)),
        ('a vector placed in lanes past its own',
         sub(FLOW, '            mask &= (1 << span) - 1\n', '', 1)),
        ('a window\'s head placed whole',
         sub(FLOW, "                or s.attr('head') is not None\n", '', 1)),
        ('a tail placed past its lanes',
         sub(FLOW, '            mask &= (1 << valid) - 1\n', '', 1)),
        ('a tail stored into a register array placed for its lanes alone',
         sub(FLOW, '        if valid is not None and not (\n'
                   "                a.space is MemSpace.REGISTER and kind != 'read'):\n",
             '        if valid is not None:\n', 1)),
        ('a vector of several elements a lane placed',
         sub(FLOW, '                not isinstance(value, Value) or value.layout is None\n'
                   "                or getattr(value.type, 'length', None) is not None):",
             '                not isinstance(value, Value) or value.layout is None):', 1)),
        ('a select over the lanes not told apart',
         sub(FLOW, "        if (d is not None and d.op == 'select'\n",
             "        if (False and d is not None and d.op == 'select'\n", 1)),
        ("the lanes' own index not read as the lane",
         sub(SHARDS, '        if not self._simd or s.op != Op.RAWEXPR or s.args:\n',
             '        if True:\n', 1)),
        ("the lanes' own index not said",
         sub(BUILD, "            attrs=(('lanes', block),))", ')', 1)),
        ('a load of a vector under a mask taken',
         sub(FORWARD, '        if flow.simd and s.predicate is not None:\n',
             '        if False:\n', 1)),
        ('a number taken for a vector',
         sub(FORWARD, "        if flow.simd and 'const' in kinds:\n",
             '        if False:\n', 1)),
        ('a value of other lanes taken for a vector',
         sub(FORWARD, '        if self.flow.simd and (truth or v.layout != s.target[0].layout\n'
                      "                               or s.attr('valid') is not None):\n",
             '        if self.flow.simd and (truth\n'
             "                               or s.attr('valid') is not None):\n", 1)),
        ('a value taken for a tail',
         sub(FORWARD, "                               or s.attr('valid') is not None):\n",
             '                               ):\n', 1)),
        ('every lane of a vector asked of a vector of fewer',
         sub(FORWARD, '                lanes = (1 << v.lane_span()) - 1\n',
             '                lanes = self.flow.full\n', 1)),
        ('a tail asked of every lane of its vector',
         sub(FORWARD, '                return lanes if valid is None else lanes & ((1 << valid) - 1)\n',
             '                return lanes\n', 1)),
        ('the register read in a tail\'s place giving what it holds past it',
         sub(FORWARD, "                      or (k == 'valid' and self.flow.simd))\n",
             "                      or False)\n", 1)),
        ('a tail of a register array read whole',
         sub(Path('src/tensorforge/backend/pir/emit_esimd.py'), '        part = self._valid_width(s, v) if w > 1 else None\n',
             '        part = None\n', 1)),
        ('another lane read across under the explicit vector',
         sub(FORWARD, '                        exchange=exchange is not None and not simd\n',
             '                        exchange=exchange is not None\n', 1)),
        ('the forward under the explicit vector asked as under SPMD',
         sub(PIPE, '                                 simd=pc.explicit_simd,\n',
             '                                 simd=False,\n', 1)),
    ]),

    # The flow's broadcasts and what no lane changes, each placed once.
    'broadcasts': ('tests/test_flow.py', [
        ('a broadcast whose writing lane reads another lane\'s store',
         sub(FLOW, '                    own = lanes & (1 << lane) if lane >= 0 else 0\n',
             '                    own = 0\n', 1)),
        ('every lane of a broadcast reading its own store',
         sub(FLOW, '                    own = lanes & (1 << lane) if lane >= 0 else 0\n',
             '                    own = lanes\n', 1)),
        ('an access placed for nothing against the budget',
         sub(FLOW, '                                kind))\n'
                   '        self._spend(len(out))\n',
             '                                kind))\n'
                   '        self._spend(0)\n', 1)),
        ('the loads fetched before kept by their lane group',
         sub(FLOW, '                    self._fetched[(r.cell_at(lane), lane)].append((load, r))\n',
             '                    self._fetched[(r.cell_at(lane), r.mask)].append((load, r))\n', 1)),
        ('what a lane changes kept for the next lane',
         sub(SHARDS, "        if 'x' not in self._leaves[operand.id]:\n"
                     '            return self.evaluate(operand, env, fixed)\n',
             '        if True:\n'
             '            return self.evaluate(operand, env, fixed)\n', 1)),
        ('every lane taken for the first',
         sub(SHARDS, '            # The thread index along `x`, or the lanes\' own (`_lanes`).\n'
                     '            out = lanes\n',
             '            out = (lanes[0],) * len(lanes)\n', 1)),
        ('a select over the lanes taking one arm for all of them',
         sub(SHARDS, '            if not isinstance(c, tuple):\n'
                     '                out = self.evaluate_lanes(a if c else b, env, lanes, memo,\n',
             '            if True:\n'
             '                out = self.evaluate_lanes(a if c else b, env, lanes, memo,\n', 1)),
    ]),

    # Memory kept in blocks, an access's lanes as one (`flow._Cells`).
    'cells': ('tests/test_flow.py', [
        ('a write taken for one of each element whoever wrote it after',
         sub(FLOW, '                    piece.bits &= ~bits\n'
                   '                    if not piece.bits:\n'
                   '                        continue\n',
             '                    pass\n', 1)),
        ('a write of other elements taken for the lanes\' own',
         sub(FLOW, '                and self.starts == other.starts and self.alone())\n',
             '                and self.alone())\n', 1)),
        ('what several lanes of a write store taken for the first lane\'s',
         sub(FLOW, '            last = self._last = {e2: lane for lane, _, e2 in self.each()}\n',
             '            last = self._last = {e2: lane for lane, _, e2 in\n'
             '                                 reversed(list(self.each()))}\n', 1)),
        ('a write of uncertain standing taken for certain',
         sub(FLOW, '                        if piece.conds <= conds:\n'
                   '                            found.append((block, hit, piece, maybe))\n',
             '                        if True:\n'
             '                            found.append((block, hit, piece, maybe))\n', 1)),
        ('an element keeping one write of uncertain standing more',
         sub(FLOW, '                if seen > _DEPTH:\n', '                if seen > _DEPTH + 1:\n', 1)),
        ('a lane of a write taken to replace what the lane before stored',
         sub(FLOW, '            if e in done:\n'
                   '                prior, certain = n, True\n'
                   '            else:\n',
             '            if False:\n'
             '                prior, certain = n, True\n'
             '            else:\n', 1)),
        ('a reach a loop tells apart left whole',
         sub(FLOW, '                reach.mask = lanes\n', '                pass\n', 1)),
        ('a forward dropped for every lane where some lanes lose it',
         sub(FLOW, '                    for part in self._parts(r):\n'
                   '                        if not part.certain:\n'
                   '                            m &= ~part.mask\n',
             '                    if not r.certain:\n'
             '                        m = 0\n', 1)),
        ('the run\'s writes of other elements of the block handed on',
         sub(FLOW, '                            hit = bits & wrote\n',
             '                            hit = wrote\n', 1)),
        # s7: the arms of the element check.
        ('the arms of any branch taken to exclude each other',
         sub(FLOW, "                if s.attr('guard') == 'element_blocks':\n"
                   "                    # This run's arms, apart from another run's.\n",
             '                if True:\n'
             "                    # This run's arms, apart from another run's.\n", 1)),
        ('the arms of every run of the branch taken for one',
         sub(FLOW, "                    arm = ('arms', len(self._exclusive))\n",
             "                    arm = ('arms', 0)\n", 1)),
        ('a write taken to exclude its own arm',
         sub(FLOW, '            if name in self._exclusive and (name, not value) in read:\n',
             '            if name in self._exclusive and (name, value) in read:\n', 1)),
    ]),

    # keep_own asks the body once, and rewrites it once.
    'ownonce': ('tests/test_keep_own.py', [
        ('what is nested left out of what a statement accesses',
         sub(CARRY, '                _, a, b = self._get(y)\n'
                    '                accessed |= a\n', '                _, a, b = self._get(y)\n', 1)),
        ('an access of a kept buffer left as it was',
         sub(CARRY, '            if fit is not None:\n'
                    '                instead[id(x)] = keeper._privatized_one(\n',
             '            if False:\n'
             '                instead[id(x)] = keeper._privatized_one(\n', 1)),
        ('a list rewritten without the images declared at its head',
         sub(CARRY, '        held[:0] = b.finish()\n', '        b.finish()\n', 1)),
    ]),

    # A store of a number nothing reads, taken behind the work it stands in.
    'sink': ('tests/test_sink.py', [
        ('a store of a value taken for a number',
         sub(SINK, '            return v.id in self.consts\n',
             '            return True\n', 1)),
        ('a number something reads taken down',
         sub(SINK, '                if (self.flow.readers.get(n) or n in self.flow.escaped\n',
             '                if (n in self.flow.escaped\n', 1)),
        ('the stores going down out of their order',
         sub(SINK, '                going.append((s, cells))\n',
             '                going.insert(0, (s, cells))\n', 1)),
        ('an element both touch not asked',
         sub(SINK, '        return any(mine & cells for _, cells in going)\n',
             '        return False\n', 1)),
        ('a barrier passed',
         sub(SINK, '            if x.op == Op.BARRIER or x.effect & Effect.BARRIER:\n',
             '            if False:\n', 1)),
        ('a yield passed',
         sub(SINK, '        if s.op == Op.YIELD:\n', '        if False:\n', 1)),
        ('an exit passed',
         sub(SINK, '            if x.op == Op.EXIT:\n', '            if False:\n', 1)),
        ('text of an effect the IR does not say passed',
         sub(SINK, '            if (x.effect & Effect.UNKNOWN and not x.accesses\n',
             '            if (False\n', 1)),
        ('an access the flow could not place passed',
         sub(SINK, '            if id(x) in flow.opaque and any(\n',
             '            if False and any(\n', 1)),
        ('the pass not run',
         sub(PIPE, '        pm.add(SinkNumbers(forward._threads, forward._arena,\n'
                   '                           report=forward._report))\n', '', 1)),
    ]),

    # The host oracle's reads across the lanes of an expression the emitter
    # inlined.
    'lanereads': ('tests/test_vector_oracle.py -k expression', [
        ('an expression read across taken in the reading lane',
         sub(ORACLE, '            return peer.ev(_WAVE_TEXTS[int(name)])\n',
             '            return self.ev(_WAVE_TEXTS[int(name)])\n', 1)),
        ('an expression of the template form not read at all',
         sub(ORACLE, '            if not isinstance(name, str):\n'
                     '                return self.ev(_WAVE_TEXTS[int(name)])\n', '', 1)),
        ('an expression read across taken for a name',
         sub(ORACLE, "        if re.fullmatch(r'\\w+', text):", '        if True:', 1)),
    ]),

    # The host oracle's `mma.sync`, and what it found in NVIDIA's path.
    'mmaoracle': ('tests/test_mma_oracle.py', [
        ("the oracle's accumulator columns handed back swapped",
         sub(ORACLE, '                    r, c = g + 8 * (i // 2), 2 * t + i % 2\n',
             '                    r, c = g + 8 * (i // 2), 2 * t + 1 - i % 2\n', 1)),
        ("the oracle's B read along the row group",
         sub(ORACLE, '                    B[t + 4 * i][g] = read(it.ev(text))\n',
             '                    B[(t + 4 * i) % k][(g + 1) % n] = read(it.ev(text))\n',
             1)),
        ("the oracle's lower TF32 half dropped",
         sub(ORACLE, "            self.env[sp.group('lower')] = value - upper\n",
             "            self.env[sp.group('lower')] = 0.0\n", 1)),
        ('the accumulators declared the same on every lane again',
         sub(Path('src/tensorforge/backend/instructions/compute/primitives/nvidia.py'),
             '_PER_LANE = Uniformity.LANE', '_PER_LANE = Uniformity.GRID', 1)),
        ('a window inside its first slot walked for its extent alone',
         sub(Path('src/tensorforge/backend/instructions/compute/primitives/nvidia.py'),
             '    M = ops.lead_first + ops.lead_elements\n',
             '    M = ops.lead_elements\n', 1)),
        ('a B fragment read where it lies past the rows of the operand',
         sub(Path('src/tensorforge/backend/instructions/compute/primitives/nvidia.py'),
             '    rows = min(ktile, depth - kbase)\n', '    rows = ktile\n', 1)),
    ]),

    # Products that add into one output, summed in the D layout.
    'dchain': ('tests/test_d_chain.py', [
        ('the target keeps no chains',
         sub(Path('src/tensorforge/backend/instructions/compute/primitives/nvidia.py'),
             'D_CHAINS = True\n', 'D_CHAINS = False\n', 1)),
        ("the old value not read into the chain's accumulators",
         sub(Path('src/tensorforge/backend/instructions/compute/primitives/nvidia.py'),
             '                    writer.assign(accs[j, i, ii, 0, slot],\n',
             '                    if False: writer.assign(accs[j, i, ii, 0, slot],\n', 1)),
        ("the chain's window taken from its first member",
         sub(Path('src/tensorforge/backend/instructions/compute/dchain.py'),
             '                       rows=_union([m.d_chain_rows() for m in members]),\n',
             '                       rows=first.d_chain_rows(),\n', 1)),
        ('a member let in that adds numbers into rows it does not own',
         sub(Path('src/tensorforge/backend/instructions/compute/dchain.py'),
             '        if foreign and not member._known_zero_rows(foreign):\n',
             '        if False:\n', 1)),
        ('the carried column taken from one lane of the quad',
         sub(Path('src/tensorforge/backend/instructions/compute/primitives/nvidia.py'),
             "    fold = ctx.target.lexic.reduction('{0}', Operation.ADD, atom.d, 4,\n",
             "    fold = ctx.target.lexic.reduction('{0}', Operation.ADD, atom.d, 1,\n", 1)),
        ('every lane of a quad starting the carried column with the old value',
         sub(Path('src/tensorforge/backend/instructions/compute/primitives/nvidia.py'),
             "    return writer.op('eq', BOOL, _index(writer, mod=4), 0, hint='g')\n",
             "    return writer.op('ge', BOOL, _index(writer, mod=4), 0, hint='g')\n", 1)),
        ('a read of the output between two members not seen',
         sub(Path('src/tensorforge/backend/instructions/compute/dchain.py'),
             '            if any(_reads_or_writes(x, (dest,), dest.obj)\n',
             '            if False and any(_reads_or_writes(x, (dest,), dest.obj)\n', 1)),
    ]),
    # gfx1250's copy into the LDS: through its engine where it exists, each
    # wait counted on asynccnt, and the interpreter's model of it.
    'asynclds': ('tests/test_async_lds.py', [
        ("gfx1250's copies taken for the LDS-DMA",
         sub(ENGINES, "        return 'async_lds' if target.async_lds() else 'lds_dma'\n",
             "        return 'lds_dma'\n", 1)),
        ('a wait on asynccnt counting what another counter counts',
         sub(ASYNCMEM, '    pinned = sum(1 for u in after if u.pinned and u.waited == counter)\n',
             '    pinned = sum(1 for u in after if u.pinned)\n', 1)),
        ("gfx1250's copy sized as the LDS-DMA's",
         sub(TARGET, '            if self.async_lds():\n                return (1, 4, 8, 16)\n',
             '', 1)),
        ('a multiplication of whole waves asked of the copy',
         sub(TARGET, '        if self.async_lds():\n            return elem_bytes in (4, 8)\n',
             '', 1)),
        ('loads into registers counted with the copies',
         sub(TARGET, "        return self.hw.vendor == 'amd' and engine != 'async_lds'\n",
             "        return self.hw.vendor == 'amd'\n", 1)),
        ('the copy spelled as the LDS-DMA',
         sub(HIP_LEXIC, "    if engine == 'async_lds':\n      return _async_to_lds(dst, src, nbytes)\n",
             '', 1)),
        ('eight bytes moved as a long',
         sub(HIP_LEXIC, "_ASYNC_LDS_ELEMENT = {1: 'char', 4: 'int', 8: 'tensorforge::VectorT<int, 2>',\n",
             "_ASYNC_LDS_ELEMENT = {1: 'char', 4: 'int', 8: 'long',\n", 1)),
        ('the emitter not saying which engine a copy is',
         sub(EMIT, "                    self._copy_bytes(s), zfill * elem,\n"
                   "                    engine=s.attr('engine'), **fills)\n",
             "                    self._copy_bytes(s), zfill * elem, **fills)\n", 1)),
        ('the interpreter landing every copy at any wait',
         sub(ORACLE, '        return int(cls._WAIT_ASYNC.match(stmt).group(1))\n',
             '        return 0\n', 1)),
        ('the interpreter landing a copy as it is issued',
         sub(ORACLE, "            tracker.dma = ('async', self.lane // Races.ASYNC_WAVE)\n",
             '            tracker.dma = None\n', 1)),
        ('the interpreter writing a copy where its source is',
         sub(ORACLE, "        base = dst + int(self.ev(am.group('di')))\n",
             "        base = dst + int(self.ev(am.group('si')))\n", 1)),
        ('no fence ahead of the copy',
         sub(HIP_LEXIC, "  return (f'{COMPILER_FENCE}\\n'\n", "  return (''\n", 1)),
        ('no fence behind the wait',
         sub(ENGINES, "        return Lowered((f'__builtin_amdgcn_s_wait_asynccnt({count});',\n"
                      '                        COMPILER_FENCE))\n',
             "        return Lowered((f'__builtin_amdgcn_s_wait_asynccnt({count});',))\n", 1)),
        ('the interpreter not knowing the fence',
         sub(ORACLE, "    r'|__atomic_signal_fence'\n", '', 1)),
    ]),
    # AMD's buffer engine: what it takes, where the resource goes, and the
    # guard in the offset.
    'buffers': ('tests/test_buffer_transfers.py', [
        ('a binding the wave does not hold alike taken',
         sub(BUFFERS, '    if threads is not None and threads >= wave and threads % wave == 0:\n',
             '    if True:\n', 1)),
        ('a binding taken at the level its declaration states',
         sub(BUFFERS, '        for y in self._reads(x):\n',
             '        for y in ():\n', 1)),
        ('a hinted access taken',
         sub(BUFFERS, "REFUSED = ('nontemporal', 'cache', 'other', 'valid', 'head', 'volatile')\n",
             "REFUSED = ('other', 'valid', 'head', 'volatile')\n", 1)),
        ('a load with a value of its own under its guard taken',
         sub(BUFFERS, "REFUSED = ('nontemporal', 'cache', 'other', 'valid', 'head', 'volatile')\n",
             "REFUSED = ('nontemporal', 'cache', 'valid', 'head', 'volatile')\n", 1)),
        ('an operand passed by value taken',
         sub(BUFFERS, '            or _by_value(s.accesses[0].base)):\n',
             '            or False):\n', 1)),
        ('an access through the symbol left global',
         sub(BUFFERS, "        base = (named or {}).get(getattr(base, 'name', None))\n",
             '        base = None\n', 1)),
        ('a name declared in a region taken as declared behind it',
         sub(BUFFERS, '        named = dict(named)\n        out = []\n',
             '        out = []\n', 1)),
        ('the resources at the top of the body',
         sub(BUFFERS, '            out.extend(behind(s.target))\n',
             '            pass\n', 1)),
        ('a resource sized in elements',
         sub(BUFFERS, '                         args=(base, base.type.volume * base.type.elem.size()),\n',
             '                         args=(base, base.type.volume),\n', 1)),
        ('a guarded store through the engine under a branch as well',
         sub(EMIT, '                    and not _folds_predicate(s) and not _through_buffer(s)):\n',
             '                    and not _folds_predicate(s)):\n', 1)),
        ('a guarded load through the engine selected as well',
         sub(EMIT, '        if (s.predicate is not None and _folds_predicate(s)\n'
                   '                and not _through_buffer(s)):\n',
             '        if (s.predicate is not None and _folds_predicate(s)):\n', 1)),
        ('the guard dropped from an access through the engine',
         sub(EMIT, '        keep = None if s.predicate is None else self.operand(s.predicate)\n'
                   '        text = None\n',
             '        keep = None\n        text = None\n', 1)),
        ('a vector through the engine spelled as one element',
         sub(HIP_LEXIC, "  return elem if length == 1 else f'{elem}, {length}'\n",
             '  return elem\n', 1)),
        ('the engine whatever the option',
         sub(ABSTR, '    if options.buffer_transfers and context.target.buffer_resources():\n',
             '    if context.target.buffer_resources():\n', 1)),
        ('buffer resources on every target',
         sub(TARGET, "        return self.backend == 'hip' and self.hw.vendor == 'amd'\n\n"
                     '    def native_atomic(',
             '        return True\n\n    def native_atomic(', 1)),
        ('a guard that keeps every lane',
         sub(HIP, '         (keep ? 0u : BufferOutOfRange);\n',
             '         0u;\n', 1)),
        ('the words of a load converted rather than reinterpreted',
         sub(HIP, '    return __builtin_bit_cast(\n'
                  '        V, __builtin_amdgcn_raw_buffer_load_b32(resource, offset, 0, 0));\n',
             '    return static_cast<V>(\n'
             '        __builtin_amdgcn_raw_buffer_load_b32(resource, offset, 0, 0));\n', 1)),
        ('gfx11 resources without the raw range check',
         sub(HIP, '#elif defined(__GFX11__)\n  constexpr std::int32_t Flags = 0x31004000;\n',
             '#elif defined(__GFX11__)\n  constexpr std::int32_t Flags = 0x01004000;\n', 1)),
    ]),
    # Intel's 2D block reads (`block_reads.load_block2d`): each limit its own
    # answer, the lanes' share, and both lowerings of the one statement.
    # An element's operands by 2D blocks (`Options.sycl_element_blocks`): the
    # check, the fallback, the reads the IR forwards, the slots a guard holds,
    # the promise and the surfaces.
    'elementblocks': ('tests/test_sycl_element_blocks.py', [
        ('the blocks without the element\'s check',
         sub(BATCHLOOP, '        cond = self._element_check(writer, plan.bindings, plan.checked)\n'
                    '        if cond is None:\n'
                    '            with element_blocks.building(self._context, plan.unchecked):\n',
             '        cond = None\n'
             '        if cond is None:\n'
             '            with element_blocks.building(self._context, plan.chosen):\n', 1)),
        ('the fallback built with the blocks',
         sub(BATCHLOOP, '        with branch.otherwise():\n'
                    '            with element_blocks.building(self._context, plan.unchecked):\n',
             '        with branch.otherwise():\n'
             '            with element_blocks.building(self._context, plan.chosen):\n', 1)),
        ('a read after a write by a block',
         sub(BATCHLOOP, "            if kind == 'load' and at > first_store.get(symbol.name, at):\n"
                    '                continue\n', '', 1)),
        ('a write a read follows by a block',
         sub(BATCHLOOP, "            if kind == 'store' and at < last_load.get(symbol.name, at):\n"
                    '                continue\n', '', 1)),
        ('a transfer of a loop inside the body by blocks',
         sub(BATCHLOOP, '            if not top or symbol.name not in bound:\n',
             '            if symbol.name not in bound:\n', 1)),
        ('the promise ignored',
         sub(EBLOCKS, '  return not (promise > 0 and promise % found.grain == 0)\n',
             '  return True\n', 1)),
        ('the promise taken for any alignment',
         sub(EBLOCKS, '  return not (promise > 0 and promise % found.grain == 0)\n',
             '  return not promise > 0\n', 1)),
        ('a surface on a column of whole words',
         sub(EBLOCKS, '  if pitch < BASE_ALIGN or pitch % 16:\n', '  if pitch % 4:\n', 1)),
        ('the grain of a pitch on whole cache lines only',
         sub(EBLOCKS, '    return min(self.pitch & -self.pitch, BASE_ALIGN)\n',
             '    return BASE_ALIGN\n', 1)),
        ('a slot sticking out of its columns',
         sub(EBLOCKS, '  return start >= 0 and start + LANES <= found.rows\n',
             '  return start >= 0 and start < found.rows\n', 1)),
        ('a slot of a guard written whole',
         sub(STORE, '      if isinstance(slot, int):\n'
                    '        return start <= slot * span and (slot + 1) * span <= end\n',
             '      if isinstance(slot, int):\n'
             '        return True\n', 1)),
        ('auto taken for on',
         sub(EBLOCKS, "                 False) is True\n", "                 False) in (True, 'auto')\n", 1)),
        ('the explicit vector given blocks',
         sub(EBLOCKS, "  if target.hw.vendor != 'intel' or target.backend != 'oneapi':\n",
             "  if target.hw.vendor != 'intel' or target.backend not in ('oneapi', 'esimd'):\n", 1)),
        ('no header for an element block',
         sub(Path('src/tensorforge/common/vm/lexic/sycl_lexic.py'),
             '           or self.block2d or self.element_blocks)\n',
             '           or self.block2d)\n', 1)),
        ('the check spelled at another pitch',
         sub(Path('src/tensorforge/common/vm/lexic/sycl_lexic.py'),
             "    return (f'tensorforge::elementBlocks<{pitch}>(item.get_sub_group(), '\n",
             "    return (f'tensorforge::elementBlocks<64>(item.get_sub_group(), '\n", 1)),
        ('the blocks streamed without the cache option',
         sub(Path('src/tensorforge/common/context.py'),
             "          2 if getattr(self._options, 'sycl_element_streaming', False) else 0)\n",
             '          0)\n', 1)),
        ('an IGC without the cache-opts forms taken for one with them',
         sub(EBLOCKS, '  if toolchain.intel_block2d_cacheopts(device) is False:\n',
             '  if False:\n', 1)),
        ('streaming blocks without asking the IGC',
         sub(BATCHLOOP, '            element_blocks.cacheopts_refusal(self._context,\n'
                        "                                             'sycl_element_streaming')\n",
             '            pass\n', 1)),
        ('2D prefetches without asking the IGC',
         sub(Path('src/tensorforge/backend/instructions/abstract_instruction.py'),
             "        element_blocks.cacheopts_refusal(context, 'sycl_element_prefetch')\n",
             '        pass\n', 1)),
        ('the IGC asked for 2D prefetches without element blocks',
         sub(Path('src/tensorforge/backend/instructions/abstract_instruction.py'),
             '                      and element_blocks.asked(context)\n', '', 1)),
        ('the next element\'s I hinted a lane under the 2D prefetch',
         sub(Path('src/tensorforge/backend/instructions/ptr_manip.py'),
             "    return (('prefetch2d', True),)\n", '    return ()\n', 1)),
        ('the hints of a versioned body read off its block body',
         sub(Path('src/tensorforge/backend/pir/prefetch.py'),
             '            out.extend(s.regions[1].body)\n', '            out.append(s)\n', 1)),
        ('a 2D prefetch of one element\'s worth',
         sub(Path('src/tensorforge/backend/pir/emit.py'),
             "                nbytes = (int(s.attr('elems', 1))\n", '                nbytes = (1\n', 1)),
        ('the cache-opts probe answering yes to an undefined form',
         sub(Path('src/tensorforge/toolchain.py'),
             '                    answer = bool(tables) and not any(\n'
             '                        builtin in name\n',
             '                    answer = bool(tables) or not any(\n'
             '                        builtin in name\n', 1)),
        # s7: the fallback as today's body, the ragged last slot, auto.
        ('the arms of the element check taken for any two',
         sub(FLOW, "                if s.attr('guard') == 'element_blocks':\n"
                   "                    # This run's arms, apart from another run's.\n",
             '                if False:\n'
             "                    # This run's arms, apart from another run's.\n", 1)),
        ('the block body\'s unplaced writes met by the fallback',
         sub(FLOW, '        if self._exclusive:\n'
                   '            out = [w for w in out\n',
             '        if False:\n'
             '            out = [w for w in out\n', 1)),
        ('both arms weighed',
         sub(EMIT, '            record(-min(middle[0] - start[0], end[0] - middle[0]))\n',
             '            record(0)\n', 1)),
        ('both arms weighed as written',
         sub(EMIT, '            record(-min(middle[1] - start[1], end[1] - middle[1]))\n',
             '            record(0)\n', 1)),
        ('no overlapping read of a ragged slot',
         sub(LOAD, "      if shift is not None and lexic.sub_group_shift('{0}', shift) is not None:\n",
             '      if False:\n', 1)),
        ('the overlapping read a row past the column',
         sub(LOAD, '        up_front[s] = read(found.rows - span)\n',
             '        up_front[s] = read(found.rows - span + 1)\n', 1)),
        ('the overlapping read rotated up',
         sub(LOAD, "          value = writer.rawexpr(lexic.sub_group_shift('{0}', shift), value,\n",
             "          value = writer.rawexpr(lexic.sub_group_shift('{0}', shift, down=False), value,\n", 1)),
        ('no overlapping write of a ragged slot',
         sub(STORE, '    tail = self._overlap_tail_plan(found, lead, loop_lo, loop_hi,\n',
             '    tail = None and self._overlap_tail_plan(found, lead, loop_lo, loop_hi,\n', 1)),
        ('the ragged slot written lane by lane as well',
         sub(STORE, '      if static and tail is not None and slot == tail[0]:\n',
             '      if False:\n', 1)),
        ('a padded tail written as the overlap',
         sub(STORE, '    if (not end % span or lead.pad\n', '    if (not end % span\n', 1)),
        ('the merge\'s halves swapped',
         sub(STORE, "        merged[q] = writer.op('select', ScalarType(elem), low, down, up,\n",
             "        merged[q] = writer.op('select', ScalarType(elem), low, up, down,\n", 1)),
        ('the overlapping write a row off',
         sub(STORE, '      x = slot * span + first_row + valid - span\n',
             '      x = slot * span + first_row + valid - span + 1\n', 1)),
        ('a slot inside its columns given a rotation',
         sub(EBLOCKS, '  if (start < 0 or start + LANES <= found.rows or start >= found.rows\n',
             '  if (start < 0 or start >= found.rows\n', 1)),
        ('the rotation a lane short',
         sub(EBLOCKS, '  return start - (found.rows - LANES)\n',
             '  return start - (found.rows - LANES) - 1\n', 1)),
        ('no rotation spelled',
         sub(SYCL_LEXIC, '  def sub_group_shift(self, variable, delta, down=True):\n'
                         '    if self._explicit_simd or self._sub_groups is None:\n',
             '  def sub_group_shift(self, variable, delta, down=True):\n'
             '    if True:\n', 1)),
        ('the rotation spelled the other way',
         sub(SYCL_LEXIC, "    which = 'shift_group_left' if down else 'shift_group_right'\n",
             "    which = 'shift_group_right' if down else 'shift_group_left'\n", 1)),
        ('auto taken for off',
         sub(EBLOCKS, '  return asked(context) or auto(context)\n', '  return asked(context)\n', 1)),
        ('the auto rule not asked',
         sub(BATCHLOOP, '                surfaces)\n'
                        '            if why is not None:\n'
                        '                return None\n',
             '                surfaces)\n'
             '            if False:\n'
             '                return None\n', 1)),
        ('auto under FMA',
         sub(EBLOCKS, "  if getattr(context.get_user_options(), 'sycl_dpas', False) is not True:\n"
                      "    return 'not the DPAS path'\n",
             '  if False:\n'
             "    return 'not the DPAS path'\n", 1)),
        ('the derivative\'s chain taken',
         sub(EBLOCKS, '    chained = sorted(read_back - {name})\n', '    chained = []\n', 1)),
        ('an accumulator taken for a chain',
         sub(EBLOCKS, '    chained = sorted(read_back - {name})\n',
             '    chained = sorted(read_back)\n', 1)),
        ('a ragged accumulation over any number of slots',
         sub(EBLOCKS, '    if name in loaded and slots > AUTO_RAGGED_SLOTS:\n', '    if False:\n', 1)),
        ('a store alone taken for an accumulation',
         sub(EBLOCKS, '    if name in loaded and slots > AUTO_RAGGED_SLOTS:\n',
             '    if slots > AUTO_RAGGED_SLOTS:\n', 1)),
        ('an accumulation from row 0 held to the slots',
         sub(EBLOCKS, '    if name in surfaces and first_row > 0:\n', '    if name in surfaces:\n', 1)),
        ('the kernel\'s chain not asked',
         sub(GEN, '      context.element_chain = element_blocks.chain_refusal(self._descr_list)\n',
             '      context.element_chain = None\n', 1)),
        ('the kernel\'s chain not heeded',
         sub(EBLOCKS, '  if chain is not None:\n    return chain\n', '  if False:\n    return chain\n', 1)),
        ('a temporary taken for a chain',
         sub(EBLOCKS, "    if (not getattr(target, 'is_tmp', False) and addressing is not None\n",
             '    if (addressing is not None\n', 1)),
    ]),

    'block2d': ('tests/test_block2d.py', [
        ('a transposed store taken for a read',
         sub(BLOCKS, "    if store and transpose:\n"
                     "        return 'the hardware has no transposed 2D block store'\n",
             '', 1)),
        ('a store of 16 rows taken',
         sub(BLOCKS, 'BLOCK2D_STORE_ROWS = (1, 2, 4, 8)\n',
             'BLOCK2D_STORE_ROWS = (1, 2, 4, 8, 16)\n', 1)),
        ('a store of another width than the lanes',
         sub(BLOCKS, '    if store and (cols != BLOCK2D_LANES or rows not in BLOCK2D_STORE_ROWS):\n',
             '    if store and rows not in BLOCK2D_STORE_ROWS:\n', 1)),
        ('an element surface on the caller\'s alignment',
         sub(BLOCKS, '    return SURFACE_BASE_ALIGN if element else align\n',
             '    return align\n', 1)),
        ('an element surface on any pitch',
         sub(BLOCKS, '    why = block2d_refusal(elem, rows, cols, transpose=transpose, vnni=vnni,\n',
             '    why = None if element else block2d_refusal(elem, rows, cols, transpose=transpose, vnni=vnni,\n',
             1)),
        ('the write recorded against no memory',
         sub(BLOCKS, '        accesses=(Access(Effect.WRITE, MemSpace.GLOBAL,\n'
                     '                         builder.alias_root(base)),),\n'
                     "        attrs=(('block2d_store', True), ('shape', (rows, cols)),\n",
             "        accesses=(),\n"
             "        attrs=(('block2d_store', True), ('shape', (rows, cols)),\n", 1)),
        ('the write moved by the passes',
         sub(BLOCKS, '        op=Op.CALL, target=(), args=(base, x, y, value), pure=False,\n'
                     '        movable=False, effect=Effect.WRITE,\n',
             '        op=Op.CALL, target=(), args=(base, x, y, value), pure=False,\n'
                     '        movable=True, effect=Effect.WRITE,\n', 1)),
        ('a write of fewer elements than rows',
         sub(BLOCKS, '    if length != rows:\n', '    if False:\n', 1)),
        ('the write told apart from no call',
         sub(BLOCKS, "    return s.op == Op.CALL and bool(s.attr('block2d_store'))\n",
             "    return s.op == Op.CALL\n", 1)),
        ('the element read spelled as a surface read',
         sub(EMIT, "        if s.attr('element'):\n"
                   "            # One element's own storage (`block_reads.load_block2d`): the\n",
             "        if False:\n"
             "            # One element's own storage (`block_reads.load_block2d`): the\n", 1)),
        ('the element write spelled as a surface write',
         sub(EMIT, "        if s.attr('element'):\n"
                   "            write = (getattr(lex, 'element_block2d_store', None)\n",
             "        if False:\n"
             "            write = (getattr(lex, 'element_block2d_store', None)\n", 1)),
        ('SPMD writing a block with a sub-group of 32',
         sub(EMIT, "        if stated != lanes or lanes != getattr(self.context, 'mult_lanes',\n",
             "        if stated != lanes or False and lanes != getattr(self.context, 'mult_lanes',\n", 1)),
        ('the element surface\'s inverse taken as one',
         sub(SPMD_H, '  static constexpr int inverse = (Pitch / grain) % period;\n',
             '  static constexpr int inverse = 1;\n', 1)),
        ('the element surface on whole cache lines only',
         sub(SPMD_H, '  static constexpr int grain = (Pitch & -Pitch) < 64 ? (Pitch & -Pitch) : 64;\n',
             '  static constexpr int grain = 64;\n', 1)),
        ('the element surface shifted by the pitch\'s grains',
         sub(SPMD_H, '    return static_cast<int>((((address & 63) / grain) * inverse) &\n',
             '    return static_cast<int>((((address & 63) / grain)) &\n', 1)),
        ('elements through a table taken',
         sub(BLOCKS, '    table = elements and kind is not None and kind.follows_pointer()\n',
             '    table = False\n', 1)),
        ('a block within one element refused for its table',
         sub(BLOCKS, '    table = elements and kind is not None and kind.follows_pointer()\n',
             '    table = kind is not None and kind.follows_pointer()\n', 1)),
        ('VNNI taken',
         sub(BLOCKS, '    if vnni:\n        return (f\'the VNNI',
             '    if False:\n        return (f\'the VNNI', 1)),
        ('a sub-group of another width taken',
         sub(BLOCKS, '    if lanes != BLOCK2D_LANES:\n', '    if False:\n', 1)),
        ('a 64-bit transpose of 16 rows taken',
         sub(BLOCKS, '    if size == 8:\n        if transpose:\n',
             '    if size == 8 and not transpose:\n        if transpose:\n', 1)),
        ('any number of rows untransposed',
         sub(BLOCKS, "    if not transpose and (cols != BLOCK2D_LANES or rows not in BLOCK2D_ROWS):\n",
             "    if not transpose and cols != BLOCK2D_LANES:\n", 1)),
        ('sixteen columns transposed',
         sub(BLOCKS, 'BLOCK2D_COLS = (1, 2, 4, 8)\n', 'BLOCK2D_COLS = (1, 2, 4, 8, 16)\n', 1)),
        ('a base of unknown alignment taken',
         sub(BLOCKS, '    if align is None or align % SURFACE_BASE_ALIGN:\n',
             '    if align is not None and align % SURFACE_BASE_ALIGN:\n', 1)),
        ('a surface narrower than 64 bytes',
         sub(BLOCKS, 'SURFACE_MIN_WIDTH = 64\n', 'SURFACE_MIN_WIDTH = 32\n', 1)),
        ('a width of part elements',
         sub(BLOCKS, '    if width % size:\n', '    if False:\n', 1)),
        ('a surface of no rows',
         sub(BLOCKS, '    if height is None or not 0 < height <= SURFACE_MAX:\n',
             '    if height is None or not 0 <= height <= SURFACE_MAX:\n', 1)),
        ('a pitch under the width',
         sub(BLOCKS, '    if (pitch is None or pitch < width or pitch % SURFACE_PITCH_ALIGN\n',
             '    if (pitch is None or pitch % SURFACE_PITCH_ALIGN\n', 1)),
        ('a pitch of eight-byte steps',
         sub(BLOCKS, 'SURFACE_PITCH_ALIGN = 16\n', 'SURFACE_PITCH_ALIGN = 8\n', 1)),
        ('a surface in shared memory',
         sub(BLOCKS, '    if space is not MemSpace.GLOBAL:\n        raise IRError(f\'a 2D block {what} of',
             '    if False:\n        raise IRError(f\'a 2D block {what} of', 1)),
        ('the transposed share counted in rows',
         sub(BLOCKS, '    count = cols if transpose else rows\n    v = builder.value(',
             '    count = rows\n    v = builder.value(', 1)),
        ('the share over 32 lanes',
         sub(BLOCKS, '                      layout=RegisterLayout((LaneAxis(BLOCK2D_LANES),)))\n',
             '                      layout=RegisterLayout((LaneAxis(32),)))\n', 1)),
        ('the read recorded against no memory',
         sub(BLOCKS, '        accesses=(Access(Effect.READ, MemSpace.GLOBAL,\n'
                     '                         builder.alias_root(base)),),\n',
             '        accesses=(),\n', 1)),
        ('every block read a 2D block',
         sub(BLOCKS, "    return s.op == Op.LOAD and s.attr('engine') == 'block2d'\n",
             '    return is_block(s)\n', 1)),
        ('the block\'s width and height swapped',
         sub(BLOCKS, "    return cols, rows, (cols if s.attr('transpose') else rows)\n",
             "    return rows, cols, (cols if s.attr('transpose') else rows)\n", 1)),
        ('the explicit vector copying the block in',
         sub(Path('src/tensorforge/backend/pir/emit_esimd.py'),
             "        self.writer(f'{self.ctype(v.type, v)} {self.name(v)} = {text};')\n",
             '        self.declare(v, text, s)\n', 1)),
        ('the explicit vector reading a 2D block as a run',
         sub(Path('src/tensorforge/backend/pir/emit_esimd.py'),
             '        if op == Op.LOAD and block_reads.is_block2d(s):\n',
             '        if False:\n', 1)),
        ('a predicated block under the explicit vector',
         sub(Path('src/tensorforge/backend/pir/emit_esimd.py'),
             "            raise IRError(f'{v}: a 2D block read under a predicate, which '\n"
             "                          f'one message for the block whole has none of')\n",
             '            pass\n', 1)),
        ('SPMD reading a block with a sub-group of 32',
         sub(EMIT, "                or getattr(self.context, 'mult_lanes', None) != lanes\n",
             '', 1)),
        ('a predicated block under SPMD',
         sub(EMIT, "            raise IRError(f'{v}: a 2D block read under a predicate, which '\n"
                   "                          f'one message for the sub-group has none of')\n",
             '            pass\n', 1)),
        ('a share of 32 under SPMD',
         sub(EMIT, '        if count > block_reads.BLOCK2D_SPMD_MAX:\n', '        if False:\n', 1)),
        ('SPMD reading a 2D block as a run',
         sub(EMIT, '        if block_reads.is_block2d(s):\n            self._emit_block2d(s)\n',
             '        if False:\n            self._emit_block2d(s)\n', 1)),
        ('a share of one a vector of one',
         sub(SYCL_LEXIC, "    kind = elem if count == 1 else f'sycl::vec<{elem}, {count}>'\n",
             "    kind = f'sycl::vec<{elem}, {count}>'\n", 1)),
        ('the explicit vector never transposing',
         sub(SYCL_LEXIC, "    flag = 'true' if transpose else 'false'\n"
                         "    return (f'tensorforge::blockLoad2d",
             "    flag = 'false'\n    return (f'tensorforge::blockLoad2d", 1)),
        ('no header for a 2D block',
         sub(SYCL_LEXIC, '           or self.block2d or self.element_blocks)\n',
             '           or self.element_blocks)\n', 1)),
        ('a builtin left undeclared',
         sub(Path('src/tensorforge/include/tensorforge_device/sycl_spmd.h'),
             'TENSORFORGE_BLOCK2D_READ(intel_sub_group_2d_block_read_transpose_32b_16r4x1c)\n',
             '', 1)),
        ('the shim\'s 2D load without the pitch',
         sub(Path('tests/shim/tensorforge_sycl.h'),
             'intel_esimd::simd<T, Width * Height> blockLoad2d(const T *, unsigned, unsigned,\n'
             '                                                 unsigned, int, int) {\n',
             'intel_esimd::simd<T, Width * Height> blockLoad2d(const T *, unsigned, unsigned,\n'
             '                                                 int, int) {\n', 1)),
    ]),
    # gfx1250's tensor data mover: the engine and its counter, the copy by
    # the first wave, its descriptor's words, its waits, and the
    # interpreter's model of it.
    'tdm': ('tests/test_tdm.py tests/test_pir_engines.py', [
        ('the tiles counted on asynccnt',
         sub(ENGINES, "                   waited_on='tensorcnt', one_to_one=True),\n",
             "                   waited_on='asynccnt', one_to_one=True),\n", 1)),
        ('a tile under its first wave left out of the count behind a wait',
         sub(ASYNCMEM, "    if s.attr('leader') is True and s.attr('engine') == 'tdm':\n"
                       '        return True\n',
             "    if s.attr('leader') is True and s.attr('engine') == 'tdm':\n"
             '        return False\n', 1)),
        ('the copy issued by every wave',
         sub(BUILD, '                      pure=False, movable=True, predicate=lead,\n',
             '                      pure=False, movable=True, predicate=None,\n', 1)),
        ('lanes of no whole number of waves taken',
         sub(ENGINES, '    if threads % wave:\n'
                      "        return f'{threads} lanes are no whole number",
             '    if False:\n'
             "        return f'{threads} lanes are no whole number", 1)),
        ('a guard the lanes do not share taken',
         sub(ENGINES, '    if level < team:\n', '    if False:\n', 1)),
        ('a predicate of another wave taken for the first',
         sub(ENGINES, '            and _lane_terms(d.args[0], defs) == 1 and d.args[1] == wave):\n',
             '            and _lane_terms(d.args[0], defs) == 1):\n', 1)),
        ('the tensor data mover on every AMD part',
         sub(TARGET, '        return (self.backend == \'hip\' and self.hw.vendor == \'amd\'\n'
                     '                and str(self.hw.model) in AMD_TENSOR_DATA_MOVER)\n',
             '        return (self.backend == \'hip\' and self.hw.vendor == \'amd\')\n', 1)),
        ('the tile without the option',
         sub(LOAD, "    if not getattr(self._context.get_user_options(), 'amd_tensor_copies',\n"
                   '                   False):\n      return False\n',
             '', 1)),
        ('no descriptor ahead of the copy',
         sub(TENSORS, '                out.append(made)\n', '', 1)),
        ('the waits dropped where the copies a lane are loads',
         sub(EMIT, '        if (op == Op.WAIT_COUNTER and s.args and self._async_lex is None\n'
                   "                and s.attr('engine') not in ('tdm',) + engines.REDUCE_ENGINES\n"
                   '                and retires(s)):\n',
             '        if (op == Op.WAIT_COUNTER and s.args and self._async_lex is None\n'
             '                and retires(s)):\n', 1)),
        ('a drain that leaves the tiles in flight',
         sub(EMIT, "            if cls == 'all' and self._tensor_copies:\n",
             '            if False:\n', 1)),
        ('no fence behind the wait',
         sub(ENGINES, "        if counter == 'tensorcnt':\n            lines += (TENSOR_FENCE,)\n",
             '', 1)),
        ('the tile\'s extents swapped in the descriptor',
         sub(HIP, '       (rows >> 16) | ((cols & 0xFFFFu) << 16), rows & 0xFFFFu,\n',
             '       (rows >> 16) | ((cols & 0xFFFFu) << 16), cols & 0xFFFFu,\n', 1)),
        ('the descriptor of another type',
         sub(HIP, '       (static_cast<std::uint32_t>(global >> 32) & 0x1FFFFFFu) | (2u << 30)},\n',
             '       (static_cast<std::uint32_t>(global >> 32) & 0x1FFFFFFu) | (1u << 30)},\n', 1)),
        ('the high bits of the pitch dropped',
         sub(HIP, '       static_cast<std::uint32_t>(pitch >> 32) & 0xFFFFu, 0u}};\n',
             '       0u, 0u}};\n', 1)),
        ('the interpreter copying the tile for every lane',
         sub(ORACLE, '            if first != self.lane:\n                return\n',
             '', 1)),
        ('the interpreter landing a tile as it is issued',
         sub(ORACLE, "            tracker.dma = ('tensor', wave)\n",
             '            tracker.dma = None\n', 1)),
        ('the interpreter reading the rows a row apart',
         sub(ORACLE, '        values = [tile.src[r * tile.pitch + c]\n',
             '        values = [tile.src[r * tile.cols + c]\n', 1)),
        ('the interpreter landing every tile at any wait',
         sub(ORACLE, '        return int(cls._WAIT_TENSOR.match(stmt).group(1))\n',
             '        return 0\n', 1)),
        ('copies for the lanes of two teams taken',
         sub(EMIT, '        if len(teams) > 1:\n', '        if False:\n', 1)),
    ]),
    # The buffer engine's LDS form: an LDS-DMA copy through its source's
    # resource, the copies it leaves alone, and the interpreter's model.
    'buflds': ('tests/test_buffer_lds.py', [
        ('a predicated copy taken, whose held-off lanes write zero',
         sub(BUFFERS, '            or s.regions or s.predicate is not None or s.attr(\'zfill\')):\n',
             '            or s.regions or s.attr(\'zfill\')):\n', 1)),
        ('a copy of another width taken',
         sub(BUFFERS, "    if s.attr('width') not in LDS_WIDTHS or s.attr('elems', 1) != 1:\n",
             '    if False:\n', 1)),
        ('a copy of another engine taken',
         sub(BUFFERS, "    if (s.attr('engine') != 'lds_dma' or s.attr('form') is not None\n",
             "    if (s.attr('form') is not None\n", 1)),
        ('the copy turned into the engine\'s load',
         sub(BUFFERS, '                if base is not None and s.op == Op.COPY_ASYNC:\n',
             '                if False:\n', 1)),
        ('the copy written through the global pointer',
         sub(EMIT, "            if self._async_lex is not None and s.attr('form') == 'buffer':\n",
             '            if False:\n', 1)),
        ('every lane reading the resource\'s first element',
         sub(HIP_LEXIC, "            f'{lds}, {index});')\n",
             "            f'{lds}, 0);')\n", 1)),
        ('the verifier taking a predicated copy through a resource',
         sub(BUFFERS, '        if s.predicate is not None:\n'
                      "            diag.append(f'{s.op}: a predicated copy through a resource, whose '\n",
             '        if False:\n'
             "            diag.append(f'{s.op}: a predicated copy through a resource, whose '\n", 1)),
        ('the interpreter reading past the resource',
         sub(ORACLE, '    if resource.inside(at, int(count), int(size), keep):\n',
             '    if True:\n', 1)),
        ('the interpreter writing past the resource or a guard',
         sub(ORACLE, '        if not resource.inside(at, count, size, keep):\n            return\n',
             '', 1)),
        ('the interpreter writing each lane at its own address',
         sub(ORACLE, "        base = base + int(first.ev(bl.group('di')))\n",
             "        base = base + int(self.ev(bl.group('di'))) - self.lane % Races.DMA_WAVE\n", 1)),
        ('the interpreter landing the copy as it is issued',
         subs(ORACLE, ("        value = _buffer_load(1, size, resource, self.ev(bl.group('index')))\n"
                       '        tracker = self.mem.tracker\n'
                       '        if tracker is not None:\n'
                       '            tracker.dma = wave\n',
                       "        value = _buffer_load(1, size, resource, self.ev(bl.group('index')))\n"
                       '        tracker = self.mem.tracker\n'
                       '        if tracker is not None:\n'
                       '            tracker.dma = None\n'))),
    ]),
    # The matrix op (`Op.MATRIX_MAC`) on NVIDIA's engines: what it states,
    # the inline PTX the registry writes it as, and the figures read off it.
    'matrixnv': ('tests/test_matrix_mac.py', [
        ('the instruction named with D and A swapped',
         sub(ENGINES, "    return (f'mma.sync.aligned.m{m}n{n}k{k}.row.col.{_PTX_TYPES[d]}.'\n"
                      "            f'{_PTX_TYPES[a]}.{_PTX_TYPES[b]}.{_PTX_TYPES[c]}')\n",
             "    return (f'mma.sync.aligned.m{m}n{n}k{k}.row.col.{_PTX_TYPES[a]}.'\n"
             "            f'{_PTX_TYPES[d]}.{_PTX_TYPES[b]}.{_PTX_TYPES[c]}')\n", 1)),
        ('C written as an output alone',
         sub(ENGINES, "    operands = ([(f'+{acc}', v) for v in c] + [(reg, v) for v in a]\n",
             "    operands = ([(f'={acc}', v) for v in c] + [(reg, v) for v in a]\n", 1)),
        ('D numbered from A',
         sub(ENGINES, '    d = group(len(c), 0)\n', '    d = group(len(c), len(a))\n', 1)),
        ('FP64 operands in 32-bit registers',
         sub(ENGINES, "    acc, reg = ('d', 'd') if wide else ('f', 'r')\n",
             "    acc, reg = ('d', 'r') if wide else ('f', 'r')\n", 1)),
        ('an FP64 instruction one issue on sm_100',
         sub(EMIT, '        return max(1, (m * n * k) // (8 * 8 * 4))\n',
             '        return 1\n', 1)),
        ('sm_90 counting DMMA as eight',
         sub(EMIT, "        if (hw is None or getattr(hw, 'family', None) != 'nvidia'\n"
                   "                or (hw.sm_level() or 0) < 100):\n",
             "        if (hw is None or getattr(hw, 'family', None) != 'nvidia'\n"
             "                or (hw.sm_level() or 0) < 90):\n", 1)),
        ('the matrix op no work',
         sub(EMIT, '            if op == Op.MATRIX_MAC and engines.matrix_engine(s).work:\n',
             '            if False:\n', 1)),
        ('the issues past the first left out of the code',
         sub(EMIT, '        if op == Op.MATRIX_MAC:\n'
                   '            self._record_code_units(self._issues(s) - 1)\n',
             '        if False:\n'
             '            self._record_code_units(self._issues(s) - 1)\n', 1)),
        ('the issue category lost',
         sub(ENGINES, "def _matrix(s: Stmt) -> str:\n    return 'matrix'\n",
             "def _matrix(s: Stmt) -> str:\n    return 'other'\n", 1)),
        ('C not declared as read and written',
         sub(BUILD, '            accesses = tuple(Access(Effect.READ | Effect.WRITE,\n'
                    '                                    MemSpace.REGISTER, base=w) for w in c)\n',
             '            accesses = ()\n', 1)),
        ('the accumulators left unpinned',
         sub(BUILD, "                self._require_addressable(w, 'matrix.mac')\n"
                    '                self.pin(w)\n',
             "                self._require_addressable(w, 'matrix.mac')\n", 1)),
        ('the op made movable',
         sub(BUILD, '        movable = engines.movable_of(Op.MATRIX_MAC, attrs)\n',
             '        movable = True\n', 1)),
        ('the term left off the op',
         sub(BUILD, '        if term is not None:\n'
                    "            attrs += (('term', tuple(term)),)\n",
             '        if False:\n'
             "            attrs += (('term', tuple(term)),)\n", 1)),
        ('the collective left off the op',
         sub(BUILD, '        if found.collective is not None:\n'
                    "            attrs += (('collective', found.collective),)\n",
             '        if False:\n'
             "            attrs += (('collective', found.collective),)\n", 1)),
        ('the op checked by its slots alone',
         sub(ENGINES, '    if spec.check is not None:\n        diag.extend(spec.check(s))\n',
             '    if False:\n        diag.extend(spec.check(s))\n', 1)),
        ('written in place and declared as writing nothing',
         sub(ENGINES, "    if op == Op.MATRIX_MAC and dict(attrs).get('in_place'):\n"
                      '        return Effect.WRITE\n',
             "    if op == Op.MATRIX_MAC and dict(attrs).get('in_place'):\n"
             '        return Effect.NONE\n', 1)),
        ('the layouts of A and D swapped',
         sub(Path('src/tensorforge/backend/instructions/compute/primitives/nvidia.py'),
             '            self._layouts = (a_fragment_bits(self), None,\n'
             '                             d_fragment_bits(self))\n',
             '            self._layouts = (d_fragment_bits(self), None,\n'
             '                             a_fragment_bits(self))\n', 1)),
        ('TF32 issued as DMMA',
         sub(Path('src/tensorforge/backend/instructions/compute/primitives/nvidia.py'),
             "_ENGINES = {MMAMode.TF32: ('mma_sync', (Datatype.TF32, Datatype.TF32,\n",
             "_ENGINES = {MMAMode.TF32: ('dmma', (Datatype.TF32, Datatype.TF32,\n", 1)),
    ]),
    # AMD's MFMA on the matrix op: D declared as the emitter declares a
    # value, the builtin's operands and modifiers, the category priced by
    # shape, and the wait a DPP block owes the instruction.
    'matrixamd': ('tests/test_matrix_mac.py', [
        ('the modifiers out of order',
         sub(ENGINES, "MFMA_MODIFIERS = ('cbsz', 'abid', 'blgp')\n",
             "MFMA_MODIFIERS = ('abid', 'cbsz', 'blgp')\n", 1)),
        ('the operands out of order',
         sub(ENGINES, '    args = [env.name(v) for v in (*a, *b, *c)]\n',
             '    args = [env.name(v) for v in (*b, *a, *c)]\n', 1)),
        ('D declared in a type of its own',
         sub(EMIT, '                self.declare(s.target[0], lowered.result, s)\n',
             "                w(f'auto {self.name(s.target[0])} = {lowered.result};')\n", 1)),
        ('D left undeclared',
         sub(EMIT, '            if lowered.result is not None:\n',
             '            if False:\n', 1)),
        ('the category without the shape',
         sub(ENGINES, "        return f'mfma.{instruction[len(\"mfma_\"):]}'\n",
             "        return 'mfma'\n", 1)),
        ('an MFMA no work',
         sub(ENGINES, "    MatrixEngine('mfma', Participants.WAVE, True, _mfma_pipe, _lower_mfma,\n",
             "    MatrixEngine('mfma', Participants.WAVE, False, _mfma_pipe, _lower_mfma,\n", 1)),
        ('D inlined into its reader',
         sub(BUILD, "                      attrs=attrs + ((('no_inline', True),) if materialize\n"
                    '                                     else ()))\n',
             '                      attrs=attrs)\n', 1)),
        ('an MFMA declared impure',
         sub(BUILD, '        self._emit_op(Op.MATRIX_MAC, (v,), a + b + c, pure=True,\n',
             '        self._emit_op(Op.MATRIX_MAC, (v,), a + b + c, pure=False,\n', 1)),
        ('D uniform where its fragments are not',
         sub(BUILD, '        v = self.value(result, hint=hint, uniform=_join(a + b + c),\n',
             '        v = self.value(result, hint=hint, uniform=Uniformity.GRID,\n', 1)),
        ('the instruction left off the op',
         sub(BUILD, '        if instruction is not None:\n'
                    "            attrs += (('instruction', instruction),)\n",
             '        if False:\n'
             "            attrs += (('instruction', instruction),)\n", 1)),
        ('the vendor\'s layouts left off',
         sub(Path('src/tensorforge/backend/instructions/compute/primitives/amd/catalog.py'), '                layouts=vendor_layouts(self), term=term, plan=plan)\n',
             '                layouts=None, term=term, plan=plan)\n', 1)),
        ('a derived layout taken for the vendor\'s',
         sub(Path('src/tensorforge/backend/instructions/compute/primitives/amd/catalog.py'), '    stated = (Provenance.MEASURED, Provenance.DOCUMENTED)\n',
             '    stated = (Provenance.MEASURED, Provenance.DOCUMENTED,\n'
             '              Provenance.DERIVED)\n', 1)),
        ('the emulated operands taken as D\'s type',
         sub(Path('src/tensorforge/backend/instructions/compute/primitives/amd/catalog.py'), '        types = (self.a.dtype, self.b.dtype, self.d.dtype, self.d.dtype)\n',
             '        types = (self.d.dtype, self.d.dtype, self.d.dtype, self.d.dtype)\n', 1)),
        ('a kernel of MFMAs not waiting behind its blocks',
         sub(Path('src/tensorforge/backend/instructions/compute/primitives/amd/dpp_block.py'), '            if (stmt.op == Op.MATRIX_MAC\n'
                  '                    and stmt.attr(\'engine\') == HAZARD.then):\n',
             '            if False:\n', 1)),
        ('any matrix op taken for an MFMA',
         sub(Path('src/tensorforge/backend/instructions/compute/primitives/amd/dpp_block.py'), '            if (stmt.op == Op.MATRIX_MAC\n'
                  '                    and stmt.attr(\'engine\') == HAZARD.then):\n',
             '            if stmt.op == Op.MATRIX_MAC:\n', 1)),
        ('the wait behind a block one state short',
         sub(Path('src/tensorforge/backend/instructions/compute/primitives/amd/dpp_block.py'), "HAZARD = Hazard(then='mfma', states=2)\n",
             "HAZARD = Hazard(then='mfma', states=1)\n", 1)),
        ('raw text spelling an MFMA not read',
         sub(Path('src/tensorforge/backend/instructions/compute/primitives/amd/dpp_block.py'), "            if any(_MATRIX.search(text) for text in (stmt.attr('asm') or '',\n"
                  "                                                     stmt.text or '')):\n",
             '            if False:\n', 1)),
    ]),
    # Intel's DPAS on the matrix op: the SPMD call over the sub-group, the
    # explicit vector's written into the assignment to its accumulator, and
    # the figures it was counted under.
    'matrixintel': ('tests/test_matrix_mac.py', [
        ('the sub-group left out of the SPMD call',
         sub(ENGINES, '    args = [group] + [env.name(v) for v in (*a, *b, *c)]\n',
             '    args = [env.name(v) for v in (*a, *b, *c)]\n', 1)),
        ('XMX\'s operands in the op\'s order',
         sub(ENGINES, '    args = [env.name(v) for v in (*c, *b, *a)]\n',
             '    args = [env.name(v) for v in (*a, *b, *c)]\n', 1)),
        ('the systolic depth read off another modifier',
         sub(ENGINES, "    depth = dict(s.attr('modifiers', ())).get('depth')\n",
             "    depth = dict(s.attr('modifiers', ())).get('repeat')\n", 1)),
        ('the element of A for the accumulator\'s',
         sub(ENGINES, "                          f'{types[3].ctype()}>({\", \".join(args)})')\n",
             "                          f'{types[0].ctype()}>({\", \".join(args)})')\n", 1)),
        ('the explicit vector\'s DPAS in a variable of its own',
         sub(Path('src/tensorforge/backend/instructions/compute/primitives/intel.py'), '                        layout=join_layout((acc, bf, af)), materialize=False,\n',
             '                        layout=join_layout((acc, bf, af)), materialize=True,\n', 1)),
        ('M and N swapped in the explicit vector\'s shape',
         sub(Path('src/tensorforge/backend/instructions/compute/primitives/intel.py'), "                        'dpas_esimd', (atom.m, atom.n, atom.k),\n",
             "                        'dpas_esimd', (atom.n, atom.m, atom.k),\n", 1)),
        ('DPAS counted as matrix work',
         sub(ENGINES, "    MatrixEngine('dpas_spmd', Participants.WAVE, False, _other,\n",
             "    MatrixEngine('dpas_spmd', Participants.WAVE, True, _other,\n", 1)),
        ('DPAS under the matrix category',
         sub(ENGINES, "def _other(s: Stmt) -> str:\n    return 'other'\n",
             "def _other(s: Stmt) -> str:\n    return 'matrix'\n", 1)),
        ('the explicit vector\'s DPAS a collective of the wave',
         sub(ENGINES, "    MatrixEngine('dpas_esimd', None, False, _other, _lower_dpas_esimd,\n",
             "    MatrixEngine('dpas_esimd', Participants.WAVE, False, _other,\n"
             '                 _lower_dpas_esimd,\n', 1)),
        ('the SPMD product\'s term left off',
         sub(Path('src/tensorforge/backend/instructions/compute/primitives/intel_spmd.py'), '        term=term, plan=plan)\n', '        term=None, plan=plan)\n', 1)),
        ('the measured layouts left off',
         sub(Path('src/tensorforge/backend/instructions/compute/primitives/intel_spmd.py'), '        layouts=(a_fragment_bits(), b_fragment_bits(), d_fragment_bits()),\n',
             '        layouts=None,\n', 1)),
        ('a matrix op not ending a run',
         sub(BLOCKS, '                                  Op.SET_PRIORITY, Op.REGISTER_BUDGET,\n'
                     '                                  Op.MATRIX_MAC):\n',
             '                                  Op.SET_PRIORITY, Op.REGISTER_BUDGET):\n', 1)),
    ]),
    # wgmma on the matrix op (engines phase 3b, part 1): the asynchronous
    # product of a warpgroup on sm_90a -- its descriptors, the text it is
    # written as inside its guard, the fence, the commit and the wait, what
    # it declares, the targets that have it, and the host oracle's model of
    # it (PTX's layouts, D landing at the wait, the operands checked there).
    # The compile checks need a local nvcc; the GPU cases an sm_90 part.
    'wgmma': ('tests/test_wgmma.py', [
        ('the instruction named with A\'s type for D\'s',
         sub(ENGINES, "            f'{_WGMMA_TYPES[d]}.{_WGMMA_TYPES[a]}.{_WGMMA_TYPES[b]}')\n",
             "            f'{_WGMMA_TYPES[a]}.{_WGMMA_TYPES[d]}.{_WGMMA_TYPES[b]}')\n", 1)),
        ('D numbered from A',
         sub(ENGINES, "    d = group(len(c), 0)\n    a_text = f'%{len(c)}' if shared_a else group(len(a), len(c))\n",
             "    d = group(len(c), len(a))\n    a_text = f'%{len(c)}' if shared_a else group(len(a), len(c))\n", 1)),
        ('a descriptor of A passed in a 32-bit register',
         sub(ENGINES, "                + [('l' if shared_a else 'r', v) for v in a]\n",
             "                + [('r', v) for v in a]\n", 1)),
        ('scale-d always 1',
         sub(ENGINES, "                    f'\"{d}, {a_text}, {b_text}, {int(scale)}, 1, 1;\"')\n",
             "                    f'\"{d}, {a_text}, {b_text}, 1, 1, 1;\"')\n", 1)),
        ('a flag passed for scale-d as the byte it is',
         sub(ENGINES, "        return (f'static_cast<uint32_t>({text})'\n",
             "        return (text\n", 1)),
        ('the product not volatile',
         sub(ENGINES, '                    asm_text(template, operands, spell, volatile=True),\n',
             '                    asm_text(template, operands, spell),\n', 1)),
        ('the product outside its guard',
         sub(ENGINES, "    return Lowered((f'#if defined({WGMMA_FEATURE})',\n",
             "    return Lowered((f'// {WGMMA_FEATURE}',\n", 1)),
        ('the guard any target\'s',
         sub(ENGINES, "WGMMA_FEATURE = '__CUDA_ARCH_FEAT_SM90_ALL'\n",
             "WGMMA_FEATURE = '__CUDA_ARCH__'\n", 1)),
        ('a product read as synchronous',
         sub(ENGINES, '    if asynchronous:\n        effect |= Effect.ASYNC\n',
             '    if False:\n        effect |= Effect.ASYNC\n', 1)),
        ('a descriptor of each lane taken for the warpgroup\'s',
         sub(ENGINES, '        if isinstance(v, Value) and v.uniformity <= Uniformity.LANE:\n',
             '        if False:\n', 1)),
        ("D's layout with the row's slot bit for the column's",
         sub(Path('src/tensorforge/backend/instructions/compute/primitives/nvidia.py'),
             '            + (Bit(Place.SLOT, 2),)\n'
             '            + tuple(Bit(Place.LANE, 1 << (5 + b)) for b in range(2)))\n'
             '    cols = ((Bit(Place.SLOT, 1),)\n',
             '            + (Bit(Place.SLOT, 1),)\n'
             '            + tuple(Bit(Place.LANE, 1 << (5 + b)) for b in range(2)))\n'
             '    cols = ((Bit(Place.SLOT, 2),)\n', 1)),
        ("A's layout with its warps as one",
         sub(Path('src/tensorforge/backend/instructions/compute/primitives/nvidia.py'),
             '            + (Bit(Place.SLOT, 1),)\n'
             '            + tuple(Bit(Place.LANE, 1 << (5 + b)) for b in range(2)))\n'
             '    cols = (tuple(Bit(Place.LANE, 1 << b) for b in range(2))\n',
             '            + (Bit(Place.SLOT, 1),))\n'
             '    cols = (tuple(Bit(Place.LANE, 1 << b) for b in range(2))\n', 1)),
        ('D declared the same in every lane taken',
         sub(ENGINES, '        if not isinstance(v, Value) or v.uniformity > Uniformity.LANE:\n',
             '        if not isinstance(v, Value):\n', 1)),
        ('a product the lowering refuses built',
         sub(BUILD, '        why = engines.wgmma_refusal(s)\n        if why is not None:\n            raise IRError(why)\n',
             '', 1)),
        ('the wait without its operand fence',
         sub(ENGINES, '    if registers:\n        lines += (operand_fence(registers, env.name),)\n',
             '    if False:\n        lines += (operand_fence(registers, env.name),)\n', 1)),
        ('the fence of registers without the compiler\'s half',
         sub(ENGINES, '    lines = (operand_fence(registers, env.name),) if registers else ()\n',
             '    lines = ()\n', 1)),
        ('the wait of wgmma not written',
         sub(ENGINES, "    if counter == 'wgmma':\n        return _wait_wgmma(s, env)\n",
             '', 1)),
        ('a commit of an engine without groups',
         sub(ENGINES, "    if engine not in COMMIT_ENGINES:\n"
                      "        return [f'{s.op}: {engine!r} commits no groups; '\n",
             "    if False:\n"
             "        return [f'{s.op}: {engine!r} commits no groups; '\n", 1)),
        ('the offsets of a descriptor swapped',
         sub(ENGINES, '    bits = (((desc.lbo >> 4) & 0x3FFF) << 16\n'
                      '            | ((desc.sbo >> 4) & 0x3FFF) << 32\n',
             '    bits = (((desc.sbo >> 4) & 0x3FFF) << 16\n'
             '            | ((desc.lbo >> 4) & 0x3FFF) << 32\n', 1)),
        ('wgmma\'s swizzle numbered as tcgen05\'s',
         sub(ENGINES, "_SWIZZLE_FIELD = {'wgmma': ({0: 0, 128: 1, 64: 2, 32: 3}, 62),\n",
             "_SWIZZLE_FIELD = {'wgmma': ({0: 0, 128: 2, 64: 1, 32: 3}, 62),\n", 1)),
        ('the swizzle without the row',
         sub(ENGINES, '    return address ^ (((address >> 7) & ((1 << bits) - 1)) << 4)\n',
             '    return address ^ (((address >> 4) & ((1 << bits) - 1)) << 4)\n', 1)),
        ('a swizzled row as wide as a core matrix\'s',
         sub(ENGINES, '    return (row // 8) * desc.sbo + (row % 8) * desc.swizzle + kb\n',
             '    return (row // 8) * desc.sbo + (row % 8) * 16 + kb\n', 1)),
        ('the reads through the descriptors not declared',
         sub(BUILD, '            accesses += self._descriptor_reads(\n',
             '            accesses += () and self._descriptor_reads(\n', 1)),
        ('a descriptor inlined into its product',
         sub(BUILD, "                      attrs=(('no_inline', True),))\n        self._desc_buffers[v.id]",
             "                      attrs=())\n        self._desc_buffers[v.id]", 1)),
        ('a swizzled descriptor over a buffer off its atom',
         sub(BUILD, '            if not placed or placed % align:\n',
             '            if False:\n', 1)),
        ('the wait of a body without copies dropped',
         sub(EMIT, "                and s.attr('engine') not in ('tdm',) + engines.REDUCE_ENGINES\n                and retires(s)):\n",
             "                and s.attr('engine') not in ('tdm',) + engines.REDUCE_ENGINES):\n", 1)),
        ('plain sm_90 taken for sm_90a',
         sub(TARGET, "        return (self.backend == 'cuda' and self.arch_specific()\n",
             "        return (self.backend == 'cuda'\n", 1)),
        ('an arch-specific target built with -arch',
         sub(Path('src/tensorforge/toolchain.py'),
             "            return [f'-gencode=arch=compute_{level}a,code=sm_{level}a']\n",
             "            return [f'-arch={arch}']\n", 1)),
        ('a warpgroup taken for whole where the multiplication is wider',
         sub(CORE, '                    if num_threads % (WARPGROUP_WAVES * wave) == 0\n',
             '                    if num_threads >= WARPGROUP_WAVES * wave\n', 1)),
        ('the interpreter writing D at the issue',
         sub(ORACLE, '            self._store_register(reg, _InFlight(text))\n',
             '            self._store_register(reg, value)\n', 1)),
        ('the interpreter not checking the operands at the wait',
         sub(ORACLE, '                for base, slot, value in reads:\n',
             '                for base, slot, value in ():\n', 1)),
        ('a wait retiring every group',
         sub(ORACLE, '        while len(self.wg_groups) > pending:\n',
             '        while len(self.wg_groups) > 0:\n', 1)),
        ('the rows of D laid out as A\'s',
         sub(ORACLE, '                    r = 16 * w + g + 8 * ((i % 4) // 2)\n',
             '                    r = 16 * w + g + 8 * (i % 2)\n', 1)),
        ('the columns of A laid out as D\'s',
         sub(ORACLE, '                        A[16 * w + g + 8 * (i % 2)][t + 4 * (i // 2)] = \\\n',
             '                        A[16 * w + g + 8 * (i % 2)][2 * t + i % 2] = \\\n', 1)),
        ('the oracle reading the offsets swapped',
         sub(ORACLE, '        lbo = ((layout >> 16) & 0x3FFF) << 4\n        sbo = ((layout >> 32) & 0x3FFF) << 4\n',
             '        sbo = ((layout >> 16) & 0x3FFF) << 4\n        lbo = ((layout >> 32) & 0x3FFF) << 4\n', 1)),
        ('the oracle reading without the swizzle',
         sub(ORACLE, '                address ^= ((address >> 7) & ((1 << bits) - 1)) << 4\n',
             '', 1)),
        ('the fence helper fencing nothing',
         sub(Path('src/tensorforge/include/tensorforge_device/cuda.h'),
             '  asm volatile("wgmma.fence.sync.aligned;" ::: "memory");\n',
             '', 1)),
        ('the wait helper leaving one group more in flight',
         sub(Path('src/tensorforge/include/tensorforge_device/cuda.h'),
             'wgmma.wait_group.sync.aligned %0;" ::"n"(Pending) : "memory");\n',
             'wgmma.wait_group.sync.aligned %0;" ::"n"(Pending + 1) : "memory");\n', 1)),
    ]),
    # tcgen05 on the matrix op (engines phase 3b, part 2): the product of the
    # 5th generation tensor core on sm_100a, D in tensor memory -- the
    # instruction descriptor, the descriptor of tcgen05's format, the text
    # inside its guard, the tensor memory ops, the commit onto an mbarrier
    # and the wait for its phase, what each declares, the targets, and the
    # host oracle's model (the product in flight until the phase, the
    # warps' quarters of the lanes).  The compile checks need a local nvcc.
    'tcgen05': ('tests/test_tcgen05.py', [
        ('M and N swapped in the instruction descriptor',
         sub(ENGINES, '    return (1 << 4) | (2 << 7) | (2 << 10) | ((n >> 3) << 17) | ((m >> 4) << 24)\n',
             '    return (1 << 4) | (2 << 7) | (2 << 10) | ((m >> 3) << 17) | ((n >> 4) << 24)\n', 1)),
        ('D in FP16 by the instruction descriptor',
         sub(ENGINES, '    return (1 << 4) | (2 << 7) | (2 << 10) | ((n >> 3) << 17) | ((m >> 4) << 24)\n',
             '    return (0 << 4) | (2 << 7) | (2 << 10) | ((n >> 3) << 17) | ((m >> 4) << 24)\n', 1)),
        ("tcgen05's descriptor without its version",
         sub(ENGINES, "    if desc.form == 'tcgen05':\n        bits |= 1 << 46\n",
             "    if False:\n        bits |= 1 << 46\n", 1)),
        ("tcgen05's swizzle numbered as wgmma's",
         sub(ENGINES, "                  'tcgen05': ({0: 0, 128: 2, 64: 4, 32: 6}, 61)}\n",
             "                  'tcgen05': ({0: 0, 128: 1, 64: 2, 32: 3}, 61)}\n", 1)),
        ('A in tensor memory passed as a descriptor',
         sub(ENGINES, "                ('r', at(a, 'a_column')) if tmem_a else ('l', env.name(a)),\n",
             "                ('l', env.name(a)),\n", 1)),
        ('A in tensor memory read at no address',
         sub(ENGINES, "    a_text = '[%1]' if tmem_a else '%1'\n", "    a_text = '%1'\n", 1)),
        ('every thread issuing the product',
         sub(ENGINES, "                f'\"@e tcgen05.mma.cta_group::1.kind::tf32 [%0], {a_text}, %2, '\n",
             "                f'\"tcgen05.mma.cta_group::1.kind::tf32 [%0], {a_text}, %2, '\n", 1)),
        ('enable-input-d inverted',
         sub(ENGINES, "        head, predicate = '', str(int(enable))\n",
             "        head, predicate = '', str(1 - int(enable))\n", 1)),
        ('the product clobbering no memory',
         sub(ENGINES, "        template, operands, lambda x: x, volatile=True,\n        clobbers=('memory',)),)))\n",
             "        template, operands, lambda x: x, volatile=True,\n        clobbers=()),)))\n", 1)),
        ('the guard any target\'s',
         sub(ENGINES, "TCGEN05_FEATURE = 'TENSORFORGE_TCGEN05'\n", "TCGEN05_FEATURE = '__CUDA_ARCH__'\n", 1)),
        ('a load of a width the instruction lacks written',
         sub(ENGINES, "    count = len(names)\n    if count not in TMEM_COUNTS:\n",
             "    count = len(names)\n    if False:\n", 1)),
        ('the width of a load not checked',
         sub(ENGINES, '        if count not in TMEM_COUNTS:\n            diag.append(',
             '        if False:\n            diag.append(', 1)),
        ('the sides of the fence swapped',
         sub(ENGINES, "    side = {'before': 'Before', 'after': 'After'}.get(s.attr('side'))\n",
             "    side = {'before': 'After', 'after': 'Before'}.get(s.attr('side'))\n", 1)),
        ('a target without tcgen05 written for',
         sub(ENGINES, "    has = getattr(env.target, 'tcgen05', None)\n    if callable(has) and not has():\n",
             "    has = getattr(env.target, 'tcgen05', None)\n    if False:\n", 1)),
        ('A in tensor memory read from shared memory by the check',
         sub(ENGINES, '        spaces = [MemSpace.SHARED if src is Source.SHARED else MemSpace.TMEM\n',
             '        spaces = [MemSpace.SHARED if src is not Source.REGISTER else MemSpace.TMEM\n', 1)),
        ('the allocation slot taken for a window',
         sub(ALLOC, '               and t.type.space is not MemSpace.TMEM for t in s.target)\n',
             '               for t in s.target)\n', 1)),
        ('the wait making nothing in tensor memory valid',
         sub(BUILD, '            accesses = (tuple(Access(Effect.READ | Effect.WRITE,\n                                     MemSpace.TMEM, v) for v in tmem)\n',
             '            accesses = (()\n', 1)),
        ('sm_90a taken for a target with tcgen05',
         sub(TARGET, "                and getattr(self.hw, 'tmem_columns', None) is not None)\n",
             "                and (self.hw.sm_level() or 0) >= 90)\n", 1)),
        ('the interpreter not checking the operands at the phase',
         sub(ORACLE, "        for writes, reads, issued in landing:\n            for base, slot, value in reads:\n",
             "        for writes, reads, issued in landing:\n            for base, slot, value in ():\n", 1)),
        ('the interpreter reading a cell a product writes',
         sub(ORACLE, "            if cell in state['overlay']:\n                raise Abort(f'tensor memory at lane {lane}, column '\n",
             "            if False:\n                raise Abort(f'tensor memory at lane {lane}, column '\n", 1)),
        ('the rows of D along the columns',
         sub(ORACLE, "                cell = self._tmem_cell(r, d_column + c, stmt, 'D')\n",
             "                cell = self._tmem_cell(c, d_column + r, stmt, 'D')\n", 1)),
        ('a warp reaching every quarter',
         sub(ORACLE, '        if address >> 16 != quarter:\n', '        if False:\n', 1)),
        ('A in tensor memory read along the lanes',
         sub(ORACLE, "                    cell = self._tmem_cell(r + (a_address >> 16),\n                                           (a_address & 0xFFFF) + x, stmt,\n",
             "                    cell = self._tmem_cell(x + (a_address >> 16),\n                                           (a_address & 0xFFFF) + r, stmt,\n", 1)),
        ('tensor memory freed in flight taken',
         sub(ORACLE, "            if state['open'] or any(records for _, records\n                                    in state['bound']) \\\n                    or state['overlay']:\n",
             "            if False:\n", 1)),
        ('the commit helper committing nothing',
         sub(Path('src/tensorforge/include/tensorforge_device/cuda.h'), '    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::"\n'
                      '                 "cluster.b64 [%0];" ::"r"(bulk_shared_address(cursor.barrier))\n'
                      '                 : "memory");\n', '', 1)),
        ('the wait for a load waiting for nothing',
         sub(Path('src/tensorforge/include/tensorforge_device/cuda.h'), '  asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");\n', '', 1)),
        ('no target taken for one with tcgen05',
         sub(Path('src/tensorforge/include/tensorforge_device/cuda.h'), '#define TENSORFORGE_TCGEN05 1\n', '', 1)),
    ]),
    # The Blackwell data-center family in the descriptions -- sm_100, sm_103,
    # sm_107 and sm_110 with their `a` and `f` targets, and sm_121: what the
    # rows state (tensor memory, shared memory and its opt-in, the
    # descriptor's address bits), the targets' tcgen05, register budgets and
    # columns, sm_101's refusal, nvcc's flags and the tensor memory checks,
    # the harness's cache key, the builder's columns, and what `cuda.h` makes
    # of each target.  The `cuda.h` checks need a local nvcc.
    'blackwell': ('tests/test_blackwell_family.py', [
        ('an f target taken for no family',
         sub(TARGET, "                and model.endswith('f'))\n",
             "                and model.endswith('F'))\n", 1)),
        ('tcgen05 on a part without tensor memory',
         sub(TARGET, "                and getattr(self.hw, 'tmem_columns', None) is not None)\n",
             "                and True)\n", 1)),
        ('tcgen05 on the plain targets',
         sub(TARGET, "        return (self.backend == 'cuda'\n                and (self.arch_specific() or self.family_specific())\n                and getattr(",
             "        return (self.backend == 'cuda'\n                and getattr(", 1)),
        ('no register budgets on the family targets',
         sub(TARGET, "                and (self.arch_specific() or self.family_specific()))\n",
             "                and self.arch_specific())\n", 1)),
        ('the exclusive columns never answered',
         sub(TARGET, '        if exclusive and wider is not None:\n', '        if False:\n', 1)),
        ('the exclusive columns answered for a plain allocation',
         sub(TARGET, '        if exclusive and wider is not None:\n',
             '        if wider is not None:\n', 1)),
        ('the columns read from the lanes',
         sub(Path('src/tensorforge/common/vm/hw_descr.py'),
             "    self.tmem_columns = _int_or_none(param_table.get('tmem_columns'))\n",
             "    self.tmem_columns = _int_or_none(param_table.get('tmem_lanes'))\n", 1)),
        ('the opt-in read as the planning limit',
         sub(Path('src/tensorforge/common/vm/hw_descr.py'),
             "        parseBytes(param_table['max_local_mem_size_per_block_oversized'])\n",
             "        parseBytes(param_table['max_local_mem_size_per_block'])\n", 1)),
        ('sm_101 refused without its new name',
         sub(Path('src/tensorforge/common/vm/hw_descr.py'), '  if arch in RENAMED:\n',
             '  if False:\n', 1)),
        ("sm_107's exclusive columns a plain allocation's",
         sub(Path('src/tensorforge/common/vm/hw_descr_db.yml'), '  tmem_exclusive_columns: 576\n',
             '  tmem_exclusive_columns: 512\n', 1)),
        ("sm_107's descriptor at 14 bits",
         sub(Path('src/tensorforge/common/vm/hw_descr_db.yml'), '  matrix_descriptor_address_bits: 15\n',
             '  matrix_descriptor_address_bits: 14\n', 1)),
        ('sm_107f described as sm_100',
         sub(Path('src/tensorforge/common/vm/hw_descr_db.yml'), '- arch: sm_107f\n  base: sm_107\n',
             '- arch: sm_107f\n  base: sm_100\n', 1)),
        ('Thor without tensor memory',
         sub(Path('src/tensorforge/common/vm/hw_descr_db.yml'),
             "  # sm_100's, presumably: tcgen05 is there, `.kind::i8` included.\n"
             '  tmem_lanes: 128\n  tmem_columns: 512\n', '', 1)),
        ('a family target built as an arch-specific one',
         sub(Path('src/tensorforge/toolchain.py'),
             "            return [f'-gencode=arch=compute_{level}f,code=sm_{level}f']\n",
             "            return [f'-gencode=arch=compute_{level}a,code=sm_{level}a']\n", 1)),
        ('a family target built as a part',
         sub(Path('src/tensorforge/toolchain.py'),
             "    return (arch.startswith('sm_') and arch.endswith('f')\n",
             "    return (arch.startswith('sm_') and arch.endswith('F')\n", 1)),
        ('the tensor memory checks on a target without tcgen05',
         sub(Path('src/tensorforge/toolchain.py'),
             '        return [self.TMEM_ACCESS_CHECK] if tcgen05 else []\n',
             '        return [self.TMEM_ACCESS_CHECK]\n', 1)),
        ('the tensor memory checks never built in',
         sub(Path('src/tensorforge/toolchain.py'),
             '        return self._target(arch) + self.check_flags(arch)\n',
             '        return self._target(arch)\n', 1)),
        ('the tensor memory checks on by default',
         sub(Path('src/tensorforge/toolchain.py'),
             "    return os.environ.get(name, '') in ('1', 'true', 'yes', 'on')\n",
             "    return os.environ.get(name, '1') in ('1', 'true', 'yes', 'on')\n", 1)),
        ("sm_101's report build back for Thor's",
         sub(Path('src/tensorforge/toolchain.py'),
             "    ARCH_SPECIFIC = ('sm_90', 'sm_100', 'sm_103', 'sm_107', 'sm_110',\n",
             "    ARCH_SPECIFIC = ('sm_90', 'sm_100', 'sm_103', 'sm_107', 'sm_101',\n", 1)),
        ('the harness keying a build with the checks as one without',
         sub(Path('tests/harness/toolchain.py'),
             '        if checks:\n            h.update(" ".join(checks).encode())\n',
             '        if False:\n            h.update(" ".join(checks).encode())\n', 1)),
        ('the builder allocating past its target',
         sub(BUILD, '        if reach is not None and columns > reach:\n', '        if False:\n', 1)),
        ("sm_107's family without tcgen05",
         sub(CUDA_H, '      __CUDA_ARCH_FAMILY_SPECIFIC__ == 1070 ||                                 \\\n',
             '', 1)),
        ("sm_110's family without tcgen05",
         sub(CUDA_H, '      __CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))\n',
             '      __CUDA_ARCH_FAMILY_SPECIFIC__ == 1110))\n', 1)),
        ("sm_107's descriptor field cut to 14 bits",
         sub(CUDA_H, '#define TENSORFORGE_MATRIX_ADDRESS_BITS 15\n',
             '#define TENSORFORGE_MATRIX_ADDRESS_BITS 14\n', 1)),
        ("the 15 bits on sm_107a alone",
         sub(CUDA_H, '     __CUDA_ARCH_FAMILY_SPECIFIC__ == 1070)\n#define TENSORFORGE_MATRIX_ADDRESS_BITS 15\n',
             '     __CUDA_ARCH_FAMILY_SPECIFIC__ == 1071)\n#define TENSORFORGE_MATRIX_ADDRESS_BITS 15\n', 1)),
        ('the start address masked to 14 bits whatever the target',
         sub(CUDA_H, '          ((1ull << TENSORFORGE_MATRIX_ADDRESS_BITS) - 1ull));\n',
             '          0x3FFFull);\n', 1)),
    ]),
    # The tensor memory accelerator's copy of a box through a tensor map
    # (engine `tma`, `engines.TensorMap`) and of its reduction into the
    # tensor (`tma_reduce`): the swizzle forms, the map's rules, the copy's,
    # the stride check of a batch and its fallback, the text, the
    # interpreter's model, the launcher's parameter and the encoding in
    # `cuda_tma.h` -- the last caught on the local GPU (sm_90 or newer),
    # where the boxes run.
    'tma': ('tests/test_tma.py', [
        ('the 64-byte swizzle XORing with the row of 64 bytes',
         sub(ENGINES, '    return XorSwizzle(swizzle // 16, 16 // size, line=8)\n',
             '    return XorSwizzle(swizzle // 16, 16 // size)\n', 1)),
        ('the swizzle permuting elements rather than chunks',
         sub(ENGINES, '    return XorSwizzle(swizzle // 16, 16 // size, line=8)\n',
             '    return XorSwizzle(swizzle // 16, 1, line=8)\n', 1)),
        ('a buffer of another layout taking the box',
         sub(ENGINES, '    if t.swizzle != form:\n',
             '    if False:\n', 1)),
        ('a box off 16 bytes along its rows taken',
         sub(ENGINES, '            and (inner * tmap.elem.size()) % TMA_GLOBAL_ALIGN):\n',
             '            and False):\n', 1)),
        ('a box off 128 bytes in the buffer taken',
         sub(ENGINES, '        if (start * elem.size()) % TMA_SHARED_ALIGN:\n',
             '        if False:\n', 1)),
        ('a box past the buffer taken',
         sub(ENGINES, '        if start + need > t.volume:\n',
             '        if False:\n', 1)),
        ("a swizzled box's rows dense in shared memory",
         sub(ENGINES, '        return self.swizzle or self.box[-1] * self.elem.size()\n',
             '        return self.box[-1] * self.elem.size()\n', 1)),
        ('a box row wider than its swizzle taken',
         sub(ENGINES, '        if self.swizzle and inner > self.swizzle:\n',
             '        if False:\n', 1)),
        ('a stride off 16 bytes taken',
         sub(ENGINES, '            if (n * size) % TMA_GLOBAL_ALIGN or n * size >= TMA_STRIDE_LIMIT:\n',
             '            if n * size >= TMA_STRIDE_LIMIT:\n', 1)),
        ('the dense strides one element each',
         sub(ENGINES, '            step = None if step is None or n is None else step * n\n',
             '            step = None if step is None or n is None else step\n', 1)),
        ('the box announced in elements',
         sub(ENGINES, '        return volume * self.elem.size()\n',
             '        return volume\n', 1)),
        ('an array of pointers through a map',
         sub(ENGINES, '    if kind in (Addressing.PTR_BASED, Addressing.INDEXED_POINTER):\n        return ("an element',
             '    if False:\n        return ("an element', 1)),
        ('a stride bound at run time through a map',
         sub(ENGINES, "    if getattr(addressing, 'stride', None) is RUNTIME:\n",
             '    if False:\n', 1)),
        ('a window of several runs as one bulk copy',
         sub(ENGINES, '    elif not contiguous:\n',
             '    elif False:\n', 1)),
        ('the copy counted in commit groups',
         sub(ENGINES, "    TransferEngine('tma', (Op.COPY_ASYNC,), Completion.BARRIER,\n",
             "    TransferEngine('tma', (Op.COPY_ASYNC,), Completion.GROUP,\n", 1)),
        ('the swizzle row its width again',
         sub(CORE, '        return (g ^ ((g // self.span) % self.width)) * self.granule + r\n',
             '        return (g ^ ((g // self.width) % self.width)) * self.granule + r\n', 1)),
        ('a row as long as the width kept apart',
         sub(CORE, '            if self.line == self.width:\n',
             '            if False:\n', 1)),
        ('the builder selecting by the width',
         sub(BUILD, '        bits = swz.span.bit_length() - 1\n',
             '        bits = swz.width.bit_length() - 1\n', 1)),
        ('a copy into a buffer off the period taken',
         sub(BUILD, "        if not placed or placed % align:\n            raise IRError(f'a tensor copy into {dst}, which is not placed on '\n",
             "        if False:\n            raise IRError(f'a tensor copy into {dst}, which is not placed on '\n", 1)),
        ('the copy announcing the padded box',
         sub(BUILD, "                 ('width', tmap.box_bytes()),\n                 ('shape', tuple(tmap.box)), ('map', tmap))\n",
             "                 ('width', tmap.shared_bytes()),\n                 ('shape', tuple(tmap.box)), ('map', tmap))\n", 1)),
        ('a descriptor over a buffer of another swizzle',
         sub(BUILD, '        if form is not None and form != engines.swizzle_form(desc.swizzle,\n',
             '        if False and form != engines.swizzle_form(desc.swizzle,\n', 1)),
        ('the coordinates outermost first in the text',
         sub(EMIT, '                               for c in reversed(s.copy_src_index))\n',
             '                               for c in s.copy_src_index)\n', 1)),
        ('a copy through a map on a target without one',
         sub(EMIT, '            if target is None or not target.tensor_maps():\n',
             '            if False:\n', 1)),
        ('sm_80 taken for one with tensor maps',
         sub(TARGET, '        return self.bulk_copy_path()\n\n    def arch_specific(self) -> bool:\n',
             "        return self.backend == 'cuda'\n\n    def arch_specific(self) -> bool:\n", 1)),
        ('the interpreter not swizzling the box',
         sub(ORACLE, '            address ^= ((address >> 7) & ((1 << bits) - 1)) << 4\n            yield (',
             '            yield (', 1)),
        ('the interpreter swizzling relative to the box',
         sub(ORACLE, '            address = start + row * pitch + at[-1] * width\n            address ^= ((address >> 7) & ((1 << bits) - 1)) << 4\n',
             '            address = row * pitch + at[-1] * width\n            address ^= ((address >> 7) & ((1 << bits) - 1)) << 4\n            address += start\n', 1)),
        ('the interpreter filling zeros for NaN',
         sub(ORACLE, "        fill = float('nan') if tmap.nan else 0.0\n"
             '        tracker = self.mem.tracker\n',
             '        fill = 0.0\n        tracker = self.mem.tracker\n', 1)),
        ('the interpreter taking other bytes than the box',
         sub(ORACLE, "        if int(tl.group('bytes')) != volume * tmap.size:\n",
             '        if False:\n', 1)),
        ('the interpreter taking a box off 128 bytes',
         sub(ORACLE, '        if start % 128:\n',
             '        if False:\n', 1)),
        ('the interpreter taking a box off 16 bytes along its rows',
         sub(ORACLE, '        if (coords[-1] * tmap.size) % 16:\n',
             '        if False:\n', 1)),
        ('the interpreter reading the coordinates outermost first',
         sub(ORACLE, "        coords = [int(self.ev(c)) for c in _top_level(tl.group('coords'),\n                                                      ',')][::-1]\n",
             "        coords = [int(self.ev(c)) for c in _top_level(tl.group('coords'),\n                                                      ',')]\n", 1)),
        ('the launcher dropping the NaN fill',
         sub(KPARAMS, "        nan = ', true' if tmap.fill == 'nan' else ''\n",
             "        nan = ''\n", 1)),
        ('the map a by-value copy in the kernel',
         sub(KPARAMS, "        return f'{storage}const CUtensorMap {self.name}'\n",
             "        return f'const CUtensorMap {self.name}'\n", 1)),
        ("the extents in TensorForge's order to the driver",
         sub(Path('src/tensorforge/include/tensorforge_device/cuda_tma.h'), '    globalDim[d] = dims[Rank - 1 - d];\n',
             '    globalDim[d] = dims[d];\n', 1)),
        ("the strides in TensorForge's order to the driver",
         sub(Path('src/tensorforge/include/tensorforge_device/cuda_tma.h'), '    globalStrides[d] = strides[Rank - 2 - d] * sizeof(T);\n',
             '    globalStrides[d] = strides[d] * sizeof(T);\n', 1)),
        ('the 128-byte swizzle encoded as the 64-byte one',
         sub(Path('src/tensorforge/include/tensorforge_device/cuda_tma.h'), 'swizzle == 128  ? CU_TENSOR_MAP_SWIZZLE_128B\n',
             'swizzle == 128  ? CU_TENSOR_MAP_SWIZZLE_64B\n', 1)),
        ('the NaN fill encoded as zeros',
         sub(Path('src/tensorforge/include/tensorforge_device/cuda_tma.h'), '                   nanFill ? CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA\n',
             '                   nanFill ? CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE\n', 1)),
        ('the third coordinate the second in the copy',
         sub(Path('src/tensorforge/include/tensorforge_device/cuda_tma.h'), '        "l"(reinterpret_cast<uint64_t>(map)), "r"(c0), "r"(c1), "r"(c2),\n        "r"(bulk_shared_address(cursor.barrier))\n',
             '        "l"(reinterpret_cast<uint64_t>(map)), "r"(c0), "r"(c2), "r"(c1),\n        "r"(bulk_shared_address(cursor.barrier))\n', 1)),
        ('a reduction every thread issues taken',
         sub(ENGINES, "        if copy.predicate is None:\n            return 'every thread would add the box: no thread issues it alone'\n",
             "        if False:\n            return 'every thread would add the box: no thread issues it alone'\n", 1)),
        ('a reduction of doubles taken',
         sub(ENGINES, 'TMA_REDUCE_TYPES = (Datatype.F32, Datatype.I32, Datatype.U32)\n',
             'TMA_REDUCE_TYPES = (Datatype.F32, Datatype.F64, Datatype.I32, Datatype.U32)\n', 1)),
        ('a reduction read for coordinates where its box is',
         sub(ENGINES, '    coords = copy.copy_dst_index if reduce else copy.copy_src_index\n',
             '    coords = copy.copy_src_index\n', 1)),
        ('the wait leaving a bulk group reading',
         sub(ENGINES, "        return Lowered((f'tensorforge::bulkWaitRead<{count}>();',))\n",
             "        return Lowered((f'tensorforge::bulkWaitRead<{count + 1}>();',))\n", 1)),
        ('the reduction issued by every thread',
         sub(BUILD, '                          pure=False, movable=True, predicate=issue,\n',
             '                          pure=False, movable=True, predicate=None,\n', 1)),
        ('the wait for a reduction dropped with the copies',
         sub(EMIT, "                and s.attr('engine') not in ('tdm',) + engines.REDUCE_ENGINES\n",
             "                and s.attr('engine') not in ('tdm', 'bulk_reduce')\n", 1)),
        ("the reduction's coordinates outermost first",
         sub(EMIT, '                               for c in reversed(s.copy_dst_index))\n',
             '                               for c in s.copy_dst_index)\n', 1)),
        ('a reduction deciding the copies of cp.async',
         sub(EMIT, "        copies = [c for c in copies if c.counter != 'bulk'\n                  and c.attr('engine') not in engines.REDUCE_ENGINES]\n",
             "        copies = [c for c in copies if c.counter != 'bulk']\n", 1)),
        ('a drain leaving the reductions reading',
         sub(EMIT, "            if cls == 'all' and self._reductions:\n",
             '            if False:\n', 1)),
        ('the interpreter writing the box over the tensor',
         sub(ORACLE, '                dst[at] = dst[at] + value\n',
             '                dst[at] = value\n', 1)),
        ('the interpreter letting the box go at the issue',
         sub(ORACLE, '                dst[at] = dst[at] + value\n        tracker = self.mem.tracker\n'
                     '        if tracker is not None:\n            tracker.reduce_issue(tracker.lane, slots)\n',
             '                dst[at] = dst[at] + value\n        tracker = self.mem.tracker\n'
             '        if tracker is not None:\n            pass\n', 1)),
        ('the interpreter taking a write over a box read',
         sub(ORACLE, '        if slot in self._reduce_flight:\n',
             '        if False:\n', 1)),
        ('the reduction closed into no group',
         sub(Path('src/tensorforge/include/tensorforge_device/cuda_tma.h'), '      "r"(c0), "r"(c1), "r"(bulk_shared_address(src))\n      : "memory");\n  asm volatile("cp.async.bulk.commit_group;" ::: "memory");\n',
             '      "r"(c0), "r"(c1), "r"(bulk_shared_address(src))\n      : "memory");\n', 1)),
    ]),
    # The multi-element layout's operator steps on the asynchronous matrix
    # engines (part 4): the engine's choice and the group's size, B's
    # canonical stage and its ring, the wgmma and tcgen05 lowerings and
    # their D, the fallback to mma_sync's tiling.  The numbers are the
    # host oracle's.
    'measync': ('tests/test_multi_element_async.py', [
        ('wgmma taken on a target without it',
         sub(Path('src/tensorforge/generators/element_group.py'), "    if target.wgmma():\n        return 'wgmma'\n",
             "    if True:\n        return 'wgmma'\n", 1)),
        ('tcgen05 for a step four elements overfill',
         sub(Path('src/tensorforge/generators/element_group.py'), '    if target.tcgen05() and MIN_ELEMENTS * width <= TCGEN05_ROWS:\n',
             '    if target.tcgen05():\n', 1)),
        ('a step wider than the instruction on it',
         sub(Path('src/tensorforge/generators/element_group.py'), '    if step.rows[1] - step.rows[0] > ASYNC_N:\n        return None\n',
             '    if False:\n        return None\n', 1)),
        ('the engine held for nothing',
         sub(Path('src/tensorforge/generators/element_group.py'), '    choice = engine or context.get_user_options().matrix_engine\n',
             "    choice = 'auto'\n", 1)),
        ('a group of parts of warpgroups',
         sub(Path('src/tensorforge/generators/element_group.py'), '            elements -= elements % WARPGROUP_WARPS\n',
             '            elements -= 0\n', 1)),
        ("tcgen05's lanes overfilled",
         sub(Path('src/tensorforge/generators/element_group.py'), '            elements = min(elements,\n                           TCGEN05_ROWS // (step.cols[1] - step.cols[0]))\n',
             '            elements = min(elements, elements)\n', 1)),
        ('the core matrices along K a row apart',
         sub(Path('src/tensorforge/generators/element_group.py'), '        return (n // 8) * self.depth * 8 + (k // 4) * 32 + (n % 8) * 4 + k % 4\n',
             '        return (n // 8) * self.depth * 8 + (k // 4) * 4 + (n % 8) * 4 + k % 4\n', 1)),
        ('the stride byte offset one core matrix',
         sub(Path('src/tensorforge/generators/element_group.py'), '        return self.depth * 32\n',
             '        return 128\n', 1)),
        ('B in fragments for an asynchronous engine',
         sub(Path('src/tensorforge/generators/element_group.py'), "                      'canonical' if asynchronous else 'fragments',\n",
             "                      'fragments',\n", 1)),
        ('B unsplit on the asynchronous engines',
         sub(Path('src/tensorforge/generators/element_group.py'), "                      split and step.lowering in ('mma_sync',)\n                      + ASYNC_LOWERINGS,\n",
             "                      split and step.lowering == 'mma_sync',\n", 1)),
        ('A unpitched for the asynchronous engines',
         sub(Path('src/tensorforge/generators/element_group.py'), "        if step.lowering not in ('mma_sync',) + ASYNC_LOWERINGS:\n",
             "        if step.lowering != 'mma_sync':\n", 1)),
        ("the fill without mma_sync's tiling behind",
         sub(Path('src/tensorforge/generators/element_group.py'), '    if tilings and asynchronous(tilings[0]):\n',
             '    if False:\n', 1)),
        ("the warpgroup's warps on the same rows",
         sub(Path('src/tensorforge/backend/instructions/compute/multi_element.py'), "                   b.op('mul', INDEX, warp, 16, hint='m'), hint='m')\n",
             "                   b.op('mul', INDEX, warp, 0, hint='m'), hint='m')\n", 1)),
        ('D written over by every wgmma',
         sub(Path('src/tensorforge/backend/instructions/compute/multi_element.py'), "                b.fence(FenceKind.REGISTERS, engine='wgmma', registers=own)\n                for ia, ib in plan.products:\n                    b.wgmma(n, halves[ia], bs[ib], accs,\n                            scale_d=0 if first else 1, layouts=layouts,\n",
             "                b.fence(FenceKind.REGISTERS, engine='wgmma', registers=own)\n                for ia, ib in plan.products:\n                    b.wgmma(n, halves[ia], bs[ib], accs,\n                            scale_d=0, layouts=layouts,\n", 1)),
        ("the products without A's lower half",
         sub(Path('src/tensorforge/generators/element_group.py'), '    return nvidia.TF32_PLAN if stage.split else TF32_HALVES_BY_WHOLE\n',
             "    return (nvidia.terms.TermPlan('3xtf32', nvidia.TF32_FORM, nvidia.TF32_FORM, ((0, 0), (0, 1)), (0, 0))\n"
             '            if stage.split else TF32_HALVES_BY_WHOLE)\n', 1)),
        ('no wait for the products',
         sub(Path('src/tensorforge/backend/instructions/compute/multi_element.py'), "    b.matrix_wait('wgmma', keep, registers=registers,\n                  reads=[x for _, reads in done for x in reads])\n",
             '', 1)),
        ('D read with its last products in flight',
         sub(Path('src/tensorforge/backend/instructions/compute/multi_element.py'), '            _wgmma_retire(b, flight, 0, accs)\n',
             '            _wgmma_retire(b, flight, 1, accs)\n', 1)),
        ('no fence of the registers',
         sub(Path('src/tensorforge/backend/instructions/compute/multi_element.py'), "                b.fence(FenceKind.REGISTERS, engine='wgmma', registers=own)\n",
             '', 1)),
        ("the warp's tile rather than its warpgroup's",
         sub(Path('src/tensorforge/backend/instructions/compute/multi_element.py'), "        tile = b.op('add', INDEX, wg, b.op('mul', INDEX, loop.induction,\n",
             "        tile = b.op('add', INDEX, warp, b.op('mul', INDEX, loop.induction,\n", 1)),
        # The converting ring's issue (`_tcgen05_resident`), on lines the
        # prepared ring's (`_tcgen05_ring`, ahead of it) does not share: a
        # count-1 anchor on a shared line mutates the copy these tests never
        # build.  opring mutates the prepared ring's.
        ("A's lower half at the upper's columns",
         sub(Path('src/tensorforge/backend/instructions/compute/multi_element.py'), '        upper_at = columns[step.products.index(product)]\n        lower_at = upper_at + stage.depth\n',
             '        upper_at = columns[step.products.index(product)]\n        lower_at = upper_at\n', 1)),
        ('D written over by every tcgen05',
         sub(Path('src/tensorforge/backend/instructions/compute/multi_element.py'), '                          enable_input_d=0 if first else 1,\n                          a_column=(upper_at, lower_at)[ia] + piece.k0 + kt,\n',
             '                          enable_input_d=0,\n                          a_column=(upper_at, lower_at)[ia] + piece.k0 + kt,\n', 1)),
        ('the products committed by nobody',
         sub(Path('src/tensorforge/backend/instructions/compute/multi_element.py'), "            continue\n        b.commit_group('tcgen05', object=cursor, issue=issue)\n        if copy is not None and i + 1 < len(pieces):\n",
             "            continue\n        if copy is not None and i + 1 < len(pieces):\n", 1)),
        ('D read from its stage before the lane-holding warps wrote it',
         sub(ME_SECTION, "    b.fence(FenceKind.THREAD_SYNC, engine='tcgen05', side='before')\n    b.barrier(trip.meet)\n    threads = trip.group.threads * trip.group.elements\n",
             "    b.fence(FenceKind.THREAD_SYNC, engine='tcgen05', side='before')\n    threads = trip.group.threads * trip.group.elements\n", 1)),
        ('a band of warps the group does not hold whole storing A',
         sub(ME_SECTION, '    k_lo, k_hi = product.depth\n    bands = max(1, trip.group.elements // 4)\n',
             '    k_lo, k_hi = product.depth\n    bands = trip.group.elements // 4 + 1\n', 1)),
        ('every product\'s A at D\'s end in one round',
         sub(ME_SECTION, '            at.append(column)\n            column += 2 * depth\n',
             '            at.append(n)\n            column += 2 * depth\n', 1)),
        ('the chunk read where it is put',
         sub(ME_SECTION, '        if i + 1 < len(pieces):\n            ahead[i + 1] = fetch(i + 1)\n',
             '        if False:\n            ahead[i + 1] = fetch(i + 1)\n', 1)),
        ('a streamed step of one product taken apart',
         sub(ME_SECTION, '    return n + 2 * max(depths), tuple(n for _ in depths), False\n',
             '    return n + 2 * max(depths), tuple(n for _ in depths), len(depths) == 1\n', 1)),
        ('the tensor memory allocated again by a trip',
         sub(Path('src/tensorforge/backend/instructions/compute/multi_element.py'), '    padded = M < TCGEN05_ROWS\n    tmem = trip.tmem\n',
             '    padded = M < TCGEN05_ROWS\n    tmem = b.tmem_alloc(256, arena=GeneralLexicon.TOTAL_SHR_MEM)\n', 1)),
        ('a D stage a tcgen05 step',
         sub(Path('src/tensorforge/backend/instructions/compute/multi_element.py'), '    if trip.dstage is None:\n',
             '    if True:\n', 1)),
        ("the tensor memory's slot a warp's own",
         sub(Path('src/tensorforge/backend/instructions/compute/multi_element.py'), '            columns, arena=GeneralLexicon.TOTAL_SHR_MEM) if columns else None\n',
             '            columns) if columns else None\n', 1)),
        ("the products' mbarrier a warp's own",
         sub(Path('src/tensorforge/backend/instructions/compute/multi_element.py'), '                            participants=trip.meet,\n                            arena=GeneralLexicon.TOTAL_SHR_MEM)\n',
             '                            participants=trip.meet)\n', 1)),
        ('the tensor memory never freed',
         sub(Path('src/tensorforge/generators/generator.py'), '        self._section.post_ir.append(MultiElementRelease(self._context,\n                                                         operators))\n',
             '        pass\n', 1)),
        ('D\'s rows past the first two warps never loaded',
         sub(Path('src/tensorforge/backend/instructions/compute/multi_element.py'), "    holds = b.op('lt', BOOL, trip.slot, 4, hint='lanes')\n",
             "    holds = b.op('lt', BOOL, trip.slot, 2, hint='lanes')\n", 1)),
        ('an accumulating step storing',
         sub(Path('src/tensorforge/backend/instructions/compute/multi_element.py'), '                predicate=keep, atomic=not step.assigns)\n',
             '                predicate=keep, atomic=False)\n', 1)),
        ("B's lower half over its upper",
         sub(Path('src/tensorforge/backend/instructions/compute/multi_element.py'), "                        b.op('add', INDEX, slot, cells, hint='s'))\n",
             '                        slot)\n', 1)),
        ('B staged without the proxy fence',
         sub(Path('src/tensorforge/backend/instructions/compute/multi_element.py'), '            builder.fence(FenceKind.PROXY)\n        builder.barrier(Participants.BLOCK)\n',
             '        builder.barrier(Participants.BLOCK)\n', 1)),
        # B's zeros past the window's contraction: an input's padding holds
        # its column's first row, which the padded steps of K multiply by
        # them (`element_group.Stage`).  wgmma reads that padding through
        # ldmatrix (the chain's I); tcgen05 stores A's padding as zeros.
        ("B's padding ones",
         sub(Path('src/tensorforge/backend/instructions/compute/multi_element.py'),
             "                         b.const(0.0, fp), hint='k')\n"
             "            # `(n / 8) depth 8 + (k / 4) 32 + (n % 8) 4 + k % 4`\n",
             "                         b.const(1.0, fp), hint='k')\n"
             "            # `(n / 8) depth 8 + (k / 4) 32 + (n % 8) 4 + k % 4`\n", 1)),
        ("a chunk's padding ones",
         sub(Path('src/tensorforge/backend/instructions/compute/multi_element.py'),
             "        value = b.op('select', fp, inside, value, b.const(0.0, fp),\n"
             "                     hint='k')\n",
             "        value = b.op('select', fp, inside, value, b.const(1.0, fp),\n"
             "                     hint='k')\n", 1)),
        ("a held plan built on the target's engine",
         sub(Path('src/tensorforge/generators/generator.py'), "      engine = (None if element_group.asynchronous(planned)\n                else 'mma_sync')\n",
             '      engine = None\n', 1)),
        ("the ring left out of the block's room",
         sub(Path('src/tensorforge/generators/element_group.py'), '                       if s.terms is not None)\n                 + ring_volume(section))\n',
             '                       if s.terms is not None))\n', 1)),
        ('B whole tried before its stream',
         sub(Path('src/tensorforge/generators/element_group.py'), '            tilings.append(lower(section, context, split=True,\n                                 streamed=True, prepared=asked, star=star,\n                                 output=output))\n',
             '', 1)),
        ('a streamed stage staged whole besides',
         sub(Path('src/tensorforge/generators/element_group.py'), '        if self.streamed:\n            return 0\n',
             '', 1)),
        ("the plan's stream not built",
         sub(Path('src/tensorforge/generators/generator.py'), '          streamed=element_group.streamed(planned),\n',
             '          streamed=False,\n', 1)),
        ("a chunk's lower halves over its upper",
         sub(Path('src/tensorforge/backend/instructions/compute/multi_element.py'), "            b.store(ring, _as_float(b, lower),\n                    b.op('add', INDEX, place, cells, hint='s'),\n                    predicate=live)\n",
             '            b.store(ring, _as_float(b, lower), place, predicate=live)\n', 1)),
        ("a chunk's rows of K read from the window's first",
         sub(Path('src/tensorforge/backend/instructions/compute/multi_element.py'), "        col = _index(b, 'add', k, d0 + k0, 'c')\n",
             "        col = _index(b, 'add', k, d0, 'c')\n", 1)),
        ("a chunk's core matrices the depth apart",
         sub(Path('src/tensorforge/backend/instructions/compute/multi_element.py'), '                              sbo=self.rows * 32, form=form)\n',
             '                              sbo=self.stage.stride, form=form)\n', 1)),
        ('every chunk in the first slot',
         sub(Path('src/tensorforge/backend/instructions/compute/multi_element.py'), '            at = (len(pieces) % RING_SLOTS) * slot\n',
             '            at = 0\n', 1)),
        ("wgmma's A at the product's first rows a chunk",
         sub(Path('src/tensorforge/backend/instructions/compute/multi_element.py'), '            plan = _plan(stage)\n            k_lo = product.depth[0] + piece.k0\n',
             '            plan = _plan(stage)\n            k_lo = product.depth[0]\n', 1)),
        ("tcgen05's A at the product's first columns a chunk",
         sub(Path('src/tensorforge/backend/instructions/compute/multi_element.py'), '                          a_column=(upper_at, lower_at)[ia] + piece.k0 + kt,\n',
             '                          a_column=(upper_at, lower_at)[ia] + kt,\n', 1)),
        ("no barrier behind wgmma's chunk",
         sub(Path('src/tensorforge/backend/instructions/compute/multi_element.py'), '            b.fence(FenceKind.PROXY)\n            b.barrier(trip.meet)\n        if flight:',
             '            b.fence(FenceKind.PROXY)\n        if flight:', 1)),
    ]),
    # The multi-element layout (`generators.element_group`, `compute.
    # multi_element`, `Trip.ELEMENT_GROUP`): the classification and its
    # steps, the fill rule, the option, the loop's group, and the lowerings
    # -- the FMA forms and the matrix tiling: the staging and the operators'
    # stages, the instructions' own barriers (a build without the barrier
    # optimization), the masks, each product's window, the contractions, D's
    # way to its buffer and to the output.  Three shapes: the volume, a chain
    # with the star at the end, and a reused intermediate.  The numbers are
    # the host oracle's and the local GPU's.
    'multielement': ('tests/test_multi_element.py', [
        ('a product adding outside its window',
         sub(ME_SECTION, '            if (product.rows, product.cols) == (rows, cols):\n',
             '            if True:\n', 1)),
        ('the element\'s write without its mask',
         sub(ME_SECTION, '            predicate = self.mask\n',
             '            predicate = None\n', 1)),
        ('every warp staging into the first slot',
         sub(ME_SECTION, "                                group.elements, hint='slot'))\n",
             "                                1, hint='slot'))\n", 1)),
        ('a step reading what the block has not finished writing',
         sub(ME_SECTION, '            if written & reads or dests & (written | read):\n',
             '            if dests & (written | read):\n', 1)),
        ('a step writing what the block has not finished reading',
         sub(ME_SECTION, '            if written & reads or dests & (written | read):\n',
             '            if written & reads:\n', 1)),
        ('the operators read before the block has copied them',
         sub(ME_SECTION, '        builder.barrier(Participants.BLOCK)\n        self._budgets(builder)\n',
             '        self._budgets(builder)\n', 1)),
        ('a trip\'s staging written over the last trip\'s reads',
         sub(ME_SECTION, '            builder.barrier(Participants.BLOCK)\n'
                         '        tokens = []\n',
             '        tokens = []\n', 1)),
        ('an accumulating step assigning',
         sub(ME_SECTION, '        if not step.assigns:\n', '        if False:\n', 1)),
        ('a buffer read at another height than staged',
         sub(ME_SECTION, "            b, row, col, (buffer.rows[0], buffer.cols[0]), buffer.pitch),\n",
             "            b, row, col, (buffer.rows[0], buffer.cols[0]), buffer.pitch + 1),\n", 1)),
        ('a short contraction cut short',
         sub(ME_SECTION, '        for k in range(lo, hi):\n',
             '        for k in range(lo, hi - 1):\n', 1)),
        ('a long contraction cut short',
         sub(ME_SECTION, '    with b.for_(lo, hi, 1, inits=(0.0,), types=(trip.fp,),\n',
             '    with b.for_(lo, hi - 1, 1, inits=(0.0,), types=(trip.fp,),\n', 1)),
        # The FMA form over the operators' nonzeros (`element_group.Terms`,
        # `multi_element._sparse`): the terms, their tables and their room,
        # the walk and its guards, and the emitter's count of it.
        ('the terms\' rows in their own order',
         sub(ME_GROUP, '                         key=lambda i: (-count[i], i)))\n',
             '                         key=lambda i: i))\n', 1)),
        ('an accumulating step walking rows without terms',
         sub(ME_GROUP, '                          if step.assigns or count[i]),\n',
             '                          if True),\n', 1)),
        ('an assigning step leaving its rows without terms unwritten',
         sub(ME_GROUP, '                          if step.assigns or count[i]),\n',
             '                          if count[i]),\n', 1)),
        ('a product\'s last column of nonzeros left out',
         sub(ME_GROUP, '        per.append({i: [k for k in range(k0, k1) if pattern.is_nz((i, k))]\n',
             '        per.append({i: [k for k in range(k0, k1 - 1) if pattern.is_nz((i, k))]\n', 1)),
        ('an operator\'s cells read transposed',
         sub(ME_GROUP, '    return (row - lower[0]) + (col - lower[1]) * int(tensor.bbox.sizes()[0])\n',
             '    return (col - lower[1]) + (row - lower[0]) * int(tensor.bbox.sizes()[0])\n', 1)),
        ('the terms taken where the zeros are not to be skipped',
         sub(ME_GROUP, "            terms=nonzero_terms(s) if sparse and lowering == 'fma'\n",
             "            terms=nonzero_terms(s) if lowering == 'fma'\n", 1)),
        ('the operators staged as stored beside their terms',
         sub(ME_GROUP, '                if step.terms is None:\n                    add(Stage(t))\n',
             '                if True:\n                    add(Stage(t))\n', 1)),
        ('the terms\' room left out of the group\'s',
         sub(ME_GROUP, '                 + sum(s.terms.volume for s in section.steps\n',
             '                 + 0 * sum(s.terms.volume for s in section.steps\n', 1)),
        ('a row\'s count where its first nonzero belongs',
         sub(ME_GROUP, '                                      for v in (firsts[p][s],\n'
                       '                                                self.counts[p][s]))))\n',
             '                                      for v in (self.counts[p][s],\n'
             '                                                firsts[p][s]))))\n', 1)),
        ('a lane walking one term past its row',
         sub(ME_SECTION, "                    held_here = b.op('lt', BOOL, t, terms_here, hint='in')\n",
             "                    held_here = b.op('le', BOOL, t, terms_here, hint='in')\n", 1)),
        ('the walk\'s guard claiming every lane of every step',
         sub(ME_SECTION, '                                      attrs=_lanes_active(sum(held),\n',
             '                                      attrs=_lanes_active(steps * live,\n', 1)),
        ('the last chunk\'s lanes past the outputs not held off',
         sub(ME_SECTION, "        if live < lanes:\n            with b.if_(b.op('lt', BOOL, o, count, hint='in'),\n",
             "        if False:\n            with b.if_(b.op('lt', BOOL, o, count, hint='in'),\n", 1)),
        ('a term reading the group operand a row off',
         sub(ME_SECTION, "                              group.rows[0], 'b')\n",
             "                              group.rows[0] - 1, 'b')\n", 1)),
        ('every product\'s values packed from the tables\' start',
         sub(ME_SECTION, "                        _index(b, 'add', e, base, 'e'))\n",
             '                        e)\n', 1)),
        ('every product walking from the first\'s nonzeros',
         sub(ME_SECTION, "                first_term = b.load(info, _index(b, 'add', at, 1 + 2 * p,\n",
             "                first_term = b.load(info, _index(b, 'add', at, 1,\n", 1)),
        ('the windows\' walk over the outputs past the last',
         sub(ME_SECTION, '        if count % lanes == 0:\n', '        if True:\n', 1)),
        ('a product\'s window guard claiming the whole step',
         sub(ME_SECTION, '                                  attrs=_lanes_active(window, count))\n',
             '                                  attrs=_lanes_active(count, count))\n', 1)),
        ('the tail of the outputs counted whole',
         sub(ME_SECTION, '                   attrs=_lanes_active(count, chunks * lanes)):\n',
             '                   attrs=_lanes_active(chunks * lanes, chunks * lanes)):\n', 1)),
        ('a step with terms read over its windows',
         sub(ME_SECTION, '    (_sparse if step.terms is not None else _per_output)(trip, step)\n',
             '    _per_output(trip, step)\n', 1)),
        ('an if with results losing its statement\'s attributes',
         sub(BUILD, '                               regions=regions, pure=False, movable=False,\n'
                    '                               attrs=self._attrs))\n',
             '                               regions=regions, pure=False, movable=False))\n', 1)),
        # The lanes form (`element_group.Lanes`, `multi_element._lanes`,
        # `Trip.LANE_GROUP`): its plan and cones, the walk, the row tiles,
        # the column's write-back, the traversal and the count.
        ('the lanes form taking a mixing product of an intermediate',
         sub(ME_GROUP, '                if p.first.tensor.is_tmp:\n',
             '                if False:\n', 1)),
        ('`auto` the lanes form',
         sub(ME_GROUP, "    return FMA_FORMS.get(getattr(context.target.hw, 'vendor', None), 'warp')\n",
             "    return FMA_FORMS.get(getattr(context.target.hw, 'vendor', None), 'lanes')\n", 1)),
        # `multi_element` `auto` (`element_group.auto_takes`, `AUTO_LANES`)
        # and the generator's gates it passes as `force` does.
        ('`auto` taking F64 on Intel',
         sub(ME_GROUP, "AUTO_LANES: Dict[str, Tuple[Datatype, ...]] = {'intel': (Datatype.F32,)}\n",
             "AUTO_LANES: Dict[str, Tuple[Datatype, ...]] = {'intel': (Datatype.F32, Datatype.F64)}\n", 1)),
        ('`auto` taking NVIDIA\'s F32',
         sub(ME_GROUP, "AUTO_LANES: Dict[str, Tuple[Datatype, ...]] = {'intel': (Datatype.F32,)}\n",
             "AUTO_LANES: Dict[str, Tuple[Datatype, ...]] = {'intel': (Datatype.F32,),\n"
             "                                               'nvidia': (Datatype.F32,)}\n", 1)),
        ('`auto` taking several row ranges',
         sub(ME_GROUP, '    if len(lanes.ranges) > 1:\n        return None, (f\'the lanes form cuts the output into \'\n',
             '    if False:\n        return None, (f\'the lanes form cuts the output into \'\n', 1)),
        ('`auto` taking the lanes form where the warp form is asked for',
         sub(ME_GROUP, "    if context.get_user_options().multi_element_fma == 'warp':\n",
             '    if False:\n', 1)),
        ('`auto` in the option\'s FMA form',
         sub(ME_GROUP, "        group, _ = fill(section, context, form=(\n"
                       "            'slots' if taken.mapping is not None else 'lanes'))\n",
             '        group, _ = fill(section, context)\n', 1)),
        ('`force` under `auto`\'s rule',
         sub(ME_GROUP, "    if mode == 'auto':\n        taken = auto_takes(section, context)[0]\n",
             "    if mode in ('auto', 'force'):\n        taken = auto_takes(section, context)[0]\n", 1)),
        ('a kernel `auto` takes cut by fission',
         sub(GEN, "    if opts.multi_element in ('force', 'auto'):\n",
             "    if opts.multi_element == 'force':\n", 1)),
        ('a description `auto` takes reassociated',
         sub(GEN, "    elif options.multi_element != 'auto':\n      return False\n",
             '    else:\n      return False\n', 1)),
        ('a cone needing the rows it writes, not the ones it reads',
         sub(ME_GROUP, '                    here = {k for i in here for k, _ in nonzeros.get(i, ())}\n',
             '                    here = set(here)\n', 1)),
        ('a product reading the first write of a reused intermediate',
         sub(ME_GROUP, '            last[id(step.dest)] = step\n',
             '            last.setdefault(id(step.dest), step)\n', 1)),
        ('the plan\'s estimate without the row tile',
         sub(ME_GROUP, '                    + max(materialized, small * columns + row * tile))\n',
             '                    + max(materialized, small * columns + row))\n', 1)),
        ('the column classes past the block\'s warps',
         sub(ME_GROUP, '            kinds = max(1, min(classes, LANE_WARPS // count))\n',
             '            kinds = classes\n', 1)),
        ('a range cut by its work alone',
         sub(ME_GROUP, '        cuts = (lo,) + min(best[hi], key=lambda t: (t[0] + t[1], t[2]))[2]\n',
             '        cuts = (lo,) + min(best[hi], key=lambda t: (t[0], t[2]))[2]\n', 1)),
        ('a range cut where its two parts are closest',
         sub(ME_GROUP, '        cuts = (lo,) + min(best[hi], key=lambda t: (t[0] + t[1], t[2]))[2]\n',
             '        cuts = (lo,) + min(best[hi], key=lambda t: (t[1], t[2]))[2]\n', 1)),
        ('a range a lane does not hold',
         sub(ME_GROUP, '                        range(a, b), tile, columns, need) > limit):\n'
                       '                    break\n',
             '                        range(a, b), tile, columns, need) > limit):\n'
                       '                    pass\n', 1)),
        ('a range\'s work counting the cone its rows share again',
         sub(ME_GROUP, '                    work += sum(weight.get(i, 0) for i in rows - mine)\n'
                       '                    mine |= rows\n',
             '                    work += sum(weight.get(i, 0) for i in rows)\n'
                       '                    mine |= rows\n', 1)),
        ('a row of a tile read at its neighbour',
         sub(ME_SECTION, "        return b.extract(self.tiles[key], row - first, hint='e')\n",
             "        return b.extract(self.tiles[key], (row - first + 1) % self.width, hint='e')\n", 1)),
        ('a product of another column window counted in',
         sub(ME_SECTION, '        if product.cols == step.cols:\n            return value\n',
             '        if True:\n            return value\n', 1)),
        ('the output\'s column not read where its steps accumulate',
         sub(ME_SECTION, '            trip.acc[(i, j)] = (None if steps[0].assigns\n',
             '            trip.acc[(i, j)] = (None if True\n', 1)),
        ('a tile written where a row of it is not',
         sub(ME_SECTION, '            if claim is not None and all((r, j) in values for r in tile):\n',
             '            if claim is not None:\n', 1)),
        ('a lane\'s write without its mask',
         sub(ME_SECTION, '            b.store(target, held(value, fp), _address(b, tensor, i, col),\n'
                         '                    predicate=predicate)\n',
             '            b.store(target, held(value, fp), _address(b, tensor, i, col))\n', 1)),
        # Not 'a walk sharing the last walk's reads' (`begin_operation`'s
        # `force` dropped, both of a walk's boundaries -- at its start, and
        # after a product of a held column -- with it): every lanes kernel
        # the tests build comes out the same but for its values' numbers on
        # sm_90, gfx942 and PVC -- a walk's reads are its own already
        # (`_LaneTrip.own`'s tiles, the walk's anchor) -- so no test can tell
        # it; it was missed since milestone 2.
        ('the columns\' loop not rolled',
         sub(ME_SECTION, "        with b.for_(0, passes, 1, hint='c', rolled=True, unroll=1) as loop:\n",
             "        with b.for_(0, passes, 1, hint='c', unroll=1) as loop:\n", 1)),
        ('the lane group starting at the warps\' rows',
         sub(BATCHLOOP, "            return f'{self._group_size} * ({lexic.block_idx_x})'\n",
             "            return f'{lexic.block_dim_y} * ({lexic.block_idx_x})'\n", 1)),
        ('the lane group\'s element the warp\'s',
         sub(BATCHLOOP, "            writer(f'const auto {self._lane()} = {lexic.thread_idx_x};')\n",
             "            writer(f'const auto {self._lane()} = {lexic.thread_idx_y};')\n", 1)),
        ('the lane group\'s trips not the block\'s',
         sub(BATCHLOOP, '        if self._mults_per_block == 1 or self._trip is Trip.LANE_GROUP or (\n',
             '        if self._mults_per_block == 1 or (\n', 1)),
        ('the lane groups a stride of warps apart',
         sub(GEN, "      stride = f'({lexic.grid_dim_x} * {self._element_group.elements})'\n",
             "      stride = f'({lexic.grid_dim_x} * {lexic.block_dim_y})'\n", 1)),
        ('the lanes form\'s flops a warp\'s for an element',
         sub(BOUND, '        lanes = group.warps\n', '        lanes = lanes\n', 1)),
        ('a step over its first product\'s rows only',
         sub(ME_GROUP, '        return _union([p.rows for p in self.products])\n',
             '        return self.products[0].rows\n', 1)),
        ('products into one destination never one step',
         sub(ME_GROUP, '        if (into and p.add and p.kind is into[0][0].kind\n'
                       '                and p.kind is not Kind.SCALE):\n',
             '        if False:\n', 1)),
        ('a step left open past a read of its destination',
         sub(ME_GROUP, '            if id(members[0].dest.tensor) in reads or dest in read_by:\n',
             '            if dest in read_by:\n', 1)),
        ('a step left open past a write of what it reads',
         sub(ME_GROUP, '            if id(members[0].dest.tensor) in reads or dest in read_by:\n',
             '            if id(members[0].dest.tensor) in reads:\n', 1)),
        ('an element\'s own operator taken for the batch\'s',
         sub(ME_GROUP, '    if tensor.addressing.kind is Addressing.NONE:\n',
             '    if False:\n', 1)),
        ('a small operand of any size',
         sub(ME_GROUP, '            if max(extents) > SMALL_EXTENT:\n',
             '            if False:\n', 1)),
        ('a group larger than the block\'s shared memory',
         sub(ME_GROUP, '                if (need <= budget\n',
             '                if (True\n', 1)),
        ('the work queue\'s loop taking a group',
         sub(ME_GROUP, '    if options.launch_control:\n', '    if False:\n', 1)),
        ('`auto` taking the layout as `force` does',
         sub(ME_GROUP, "    if mode == 'auto':\n        taken = auto_takes(section, context)[0]\n",
             "    if mode == 'never':\n        taken = auto_takes(section, context)[0]\n", 1)),
        ('the layout on by default',
         sub(OPTIONS, "declare('multi_element',\n        default='off',\n",
             "declare('multi_element',\n        default='force',\n", 1)),
        ('a block of more than the group',
         sub(BATCHLOOP, '        if (self._trip is Trip.ELEMENT_GROUP\n                and mults != self._group_size + self._producer_rows):\n',
             '        if False:\n', 1)),
        ('the group\'s traversal one element a trip',
         sub(GEN, '      return self._element_group.elements\n', '      return 1\n', 1)),
        # The matrix tiling of operator steps.
        ('a product\'s B reaching past its window',
         sub(ME_SECTION, "                b.op('lt', BOOL, row, r1, hint='in'), hint='in'),\n",
             "                b.op('lt', BOOL, row, r1 + 2, hint='in'), hint='in'),\n", 1)),
        ('B read from another lane\'s fragment',
         sub(ME_SECTION, "            'mul', INDEX, place, 32, hint='f'), trip.lane, hint='f'),\n",
             "            'mul', INDEX, place, 32, hint='f'), quad, hint='f'),\n", 1)),
        ('one product of a step tiled',
         sub(ME_SECTION, '            for p, product in enumerate(step.products):\n'
                         '                buffer = section.buffer(product.second.tensor)\n',
             '            for p, product in enumerate(step.products[:1]):\n'
             '                buffer = section.buffer(product.second.tensor)\n', 1)),
        ('a TF32 product left out',
         sub(ME_SECTION, '                for term, (fa, fb) in enumerate(products):\n',
             '                for term, (fa, fb) in enumerate(products[:1]):\n', 1)),
        ('D written past the step\'s rows',
         sub(ME_SECTION, "            live = b.op('lt', BOOL, n, height, hint='in')\n",
             "            live = b.op('lt', BOOL, n, height + 8, hint='in')\n", 1)),
        ('D added into an intermediate the step assigns',
         sub(ME_SECTION, '            if step.dest.is_tmp and not step.assigns:\n',
             '            if step.dest.is_tmp:\n', 1)),
        ('an F64 step at sm_80\'s shape on sm_90',
         sub(ME_GROUP, '    return (16, 8, 4) if sm >= 90 else (8, 8, 4)\n',
             '    return (8, 8, 4)\n', 1)),
        ('the matrix switch ignored',
         sub(ME_GROUP, '    return MATRIX if asked is None else bool(asked)\n',
             '    return MATRIX\n', 1)),
        ('the FMA forms not tried where no tiling fits',
         sub(ME_GROUP, "    candidates.append(([lower(section, context, matrix=False)], None))\n", '', 1)),
        ('the halving stepping over the smallest group',
         sub(ME_GROUP, '            elements = max(MIN_ELEMENTS, elements // 2)\n',
             '            elements = elements // 2\n', 1)),
        # ldmatrix for A, B in fragment order.
        ('ldmatrix naming another column',
         sub(ME_SECTION, "        lcol = b.op('mul', INDEX, b.op('div', INDEX, trip.lane, 16,\n",
             "        lcol = b.op('mul', INDEX, b.op('div', INDEX, trip.lane, 8,\n", 1)),
        ('ldmatrix naming the rows of one half',
         sub(ME_SECTION, "        lrow = b.op('add', INDEX, b.op('rem', INDEX, trip.lane, 8, hint='l'),\n",
             "        lrow = b.op('add', INDEX, b.op('rem', INDEX, trip.lane, 4, hint='l'),\n", 1)),
        ('A read by lanes where ldmatrix could',
         sub(ME_SECTION, "    ldmatrix = step.lowering == 'mma_sync' and all(\n",
             "    ldmatrix = False and all(\n", 1)),
        ('ldmatrix stating another placement of its matrices',
         sub(ME_SECTION, "                                      origins=[a_cells[slot] for slot\n"
                         "                                               in sorted(a_cells)],\n",
             "                                      origins=[a_cells[slot] for slot\n"
             "                                               in (0, 2, 1, 3)],\n", 1)),
        ('the padding left as it was',
         sub(ME_SECTION, '            if buffer.pitch > buffer.height and not buffer.held:\n',
             '            if False:\n', 1)),
        ('a pitch not of 16-byte rows',
         sub(ME_GROUP, '            buffer, stride=n + (unit if (n // unit) % 2 == 0 else 0))\n',
             '            buffer, stride=n + 1)\n', 1)),
        ('the TF32 halves of B paired wrong',
         sub(ME_SECTION, '            return None, [(nvidia._as_tf32(b, regs[j]),\n'
                         '                           nvidia._as_tf32(b, regs[kregs + j]))\n',
             '            return None, [(nvidia._as_tf32(b, regs[kregs + j]),\n'
             '                           nvidia._as_tf32(b, regs[j]))\n', 1)),
        ('the unsplit B not tried where the halves cost a group',
         sub(ME_GROUP, '    elif split(tilings[0]):\n', '    elif False:\n', 1)),
        # D into the output from the accumulators, through the pointer tables.
        ('a masked element\'s pointer published',
         sub(ME_SECTION, "            word = b.op('select', WORD, self.mask, word, 0, hint='p')\n",
             '            pass\n', 1)),
        ('the pointers never published',
         sub(ME_SECTION, '        for tensor in section.pointer_tables:\n'
                         '            trip.publish(tensor)\n', '', 1)),
        ('a cell added where its element has no pointer',
         sub(ME_SECTION, "                    predicate=b.op('and', BOOL, live, held, hint='in'),\n",
             '                    predicate=live,\n', 1)),
        ('a cell added through its tile\'s first row\'s pointer',
         sub(ME_SECTION, '                if r0 not in pointers:\n',
             '                if not pointers:\n', 1)),
        ('the cells added into the warp\'s own element',
         sub(ME_SECTION, '                    pointers[r0] = trip.pointer(step.dest, e) + (col,)\n',
             '                    pointers[r0] = trip.pointer(step.dest, trip.slot) + (col,)\n', 1)),
        ('an accumulating step\'s cells stored over the output',
         sub(ME_SECTION, '                    atomic=not step.assigns)\n',
             '                    atomic=False)\n', 1)),
        ('an assigning step\'s cells added into the output',
         sub(ME_SECTION, '                    atomic=not step.assigns)\n',
             '                    atomic=True)\n', 1)),
        # The trip's operands by cp.async, a trip ahead (`pir.wrap`), in a
        # second stage where the options ask and the group has the room.
        ('the element group traversed as text',
         sub(BATCHLOOP, '        if ((self._narrow_group or (self._trip is Trip.ELEMENT_GROUP\n',
             '        if ((self._narrow_group or (False\n', 1)),
        ('the staging\'s ragged hop unguarded',
         sub(ME_SECTION, '                guard = (contextlib.nullcontext() if hop + lanes <= total\n',
             '                guard = (contextlib.nullcontext() if True\n', 1)),
        ('a pitched input copied over its box\'s rows',
         sub(ME_SECTION, '            total = pitch * width\n', '            total = height * width\n', 1)),
        ('the staged operands read before their copies land',
         sub(ME_SECTION, '        if tokens:\n            builder.wait(*tokens)\n',
             '        if False:\n            builder.wait(*tokens)\n', 1)),
        ('the second stage not counted',
         sub(ME_GROUP, '                      + section.inputs[:section.doubled])\n',
             '                      )\n', 1)),
        ('the second stage taken where it halves the group',
         sub(ME_GROUP, '                      for t in lowerings if t.doubled]\n',
             '                      for t in lowerings if False]\n', 1)),
        ('the copies a trip ahead where the plan declined their stage',
         sub(ME_SECTION, '                or second_stages(section, self._context) > section.doubled):\n',
             '                or False):\n', 1)),
        ('the plan\'s stage not the one the section is lowered with',
         sub(GEN, '          double=planned.doubled > 0, ring=planned.ring,\n',
             '          double=True, ring=planned.ring,\n', 1)),
        # Milestone 3, SYCL (DPC++ SPMD) and HIP: the targets, the matrix
        # tiling NVIDIA's, the lanes form's warp, budget and register file, its
        # numbers' source and its anchors, and what the generator keeps from
        # the layout's kernels -- the preload, the reassociation, the wave's
        # rows, the narrow group, fission.
        ('the layout under the explicit vector and AdaptiveCpp',
         sub(ME_GROUP, "TARGETS = (('cuda', 'nvidia'), ('hip', 'amd'), ('oneapi', 'intel'))\n",
             "TARGETS = (('cuda', 'nvidia'), ('hip', 'amd'), ('oneapi', 'intel'),\n"
             "           ('esimd', 'intel'), ('acpp', 'intel'))\n", 1)),
        # Not `if True`: `nvidia.sm_of` floors a model it does not know at
        # sm_75, short of the tiling's sm_80, so the guard's condition alone
        # changes no kernel; what it keeps off another vendor is a number.
        ('the matrix tiling off NVIDIA',
         sub(ME_GROUP, "          if getattr(context.target.hw, 'vendor', None) == 'nvidia' else 0)\n",
             "          if getattr(context.target.hw, 'vendor', None) == 'nvidia' else 80)\n", 1)),
        ('the lanes form\'s group a CUDA warp everywhere',
         sub(ME_GROUP, '    return int(context.target.hw.vec_unit_length)\n', '    return 32\n', 1)),
        ('the lanes form\'s warp of 32 lanes whatever its group',
         sub(ME_GROUP, '        return self.elements if self.lanes is not None else WARP_LANES\n',
             '        return WARP_LANES\n', 1)),
        ('Intel\'s lanes budgeted for the large file in the small one',
         sub(ME_GROUP, "        if lane_grf(context) in ('256', 'auto'):\n",
             "        if True:\n", 1)),
        ('Intel\'s lanes form in the small file where the option is open',
         sub(ME_GROUP, "    return '256' if grf == 'default' else grf\n",
             "    return grf\n", 1)),
        ('AMD\'s lanes budgeted for the AGPRs',
         sub(ME_GROUP, '        file = min(file, 1024)\n', '        pass\n', 1)),
        ('the lanes form\'s numbers literals on Intel',
         sub(ME_GROUP, "    'intel': {Datatype.F32: 'table', Datatype.F64: 'table'}}\n",
             "    'intel': {Datatype.F32: 'literal', Datatype.F64: 'literal'}}\n", 1)),
        # AMD's lanes form: F32 literals uncapped, F64 from memory, the
        # written values held where a walk holds a column, packed FP32 off.
        ('AMD\'s F32 numbers read from memory',
         sub(ME_GROUP, "    'amd': {Datatype.F32: 'literal', Datatype.F64: 'memory'},\n",
             "    'amd': {Datatype.F32: 'memory', Datatype.F64: 'memory'},\n", 1)),
        ('AMD\'s F64 numbers literals',
         sub(ME_GROUP, "    'amd': {Datatype.F32: 'literal', Datatype.F64: 'memory'},\n",
             "    'amd': {Datatype.F32: 'literal', Datatype.F64: 'literal'},\n", 1)),
        ('AMD\'s lanes literals capped by `inline_constants`',
         sub(ME_GROUP, "LANE_LITERALS = ('amd',)\n", 'LANE_LITERALS = ()\n', 1)),
        ('AMD\'s written values not held',
         sub(HIP_LEXIC, "    return f'tensorforge::opaque({value})'\n",
             '    return super().opaque_value(value)\n', 1)),
        ('a written value held where no walk holds a column',
         sub(ME_SECTION, '               if (trip.mask is not None and held_columns\n',
             '               if (trip.mask is not None and True\n', 1)),
        ('a written value held under numbers read from memory',
         sub(ME_SECTION, "                   and trip.values == 'literal') else None)\n",
             '                   and True) else None)\n', 1)),
        ('packed FP32 left on for AMD\'s lanes form',
         sub(ME_GROUP, "LANE_FEATURES = {'amd': 'no-packed-fp32-ops'}\n", 'LANE_FEATURES = {}\n', 1)),
        ('packed FP32 turned off at F64',
         sub(ME_GROUP, '    if (vendor not in LANE_FEATURES or context.fp_type is not Datatype.F32\n',
             '    if (vendor not in LANE_FEATURES or False\n', 1)),
        ('a kernel\'s target features not spelled',
         sub(CUDA_LEXIC, '    target = (f\' __attribute__((target("{features}")))\' if features\n',
             '    target = (\'\' if features\n', 1)),
        ('AMD\'s tiles read without the arithmetic before them',
         sub(HIP_LEXIC, "    return f'tensorforge::opaqueZeroAfter({value})'\n",
             '    return super().chained_zero(value)\n', 1)),
        ('the volume\'s tiles chained',
         sub(ME_SECTION, "        self.chain = (context.target.lexic.chained_zero('{0}') if held\n",
             "        self.chain = (context.target.lexic.chained_zero('{0}') if True\n", 1)),
        ('a tile chained to no product',
         sub(ME_SECTION, '            self.last = acc[(row, j)]\n', '            pass\n', 1)),
        ('the oracle reading `opaqueZeroAfter` as a call',
         sub(ORACLE, "    e = re.sub(r'\\btensorforge::opaqueZeroAfter\\s*\\(\\s*\\w+\\s*\\)', '0', e)\n", '', 1)),
        ('the oracle reading `opaque` as a call',
         sub(ORACLE, "    e = re.sub(r'\\btensorforge::opaque\\s*\\(', '(', e)\n", '', 1)),
        ('the operator\'s memory read as the description\'s numbers',
         sub(ME_SECTION, "        if self.values == 'memory':\n",
             "        if self.values == 'memory' and False:\n", 1)),
        ('a table\'s number broadcast from its block\'s first lane',
         sub(ME_SECTION, "        return b.lane_broadcast(self.block[1], at - first, lanes, hint='k')\n",
             "        return b.lane_broadcast(self.block[1], 0, lanes, hint='k')\n", 1)),
        ('a table read one block for all its numbers',
         sub(ME_SECTION, '        if self.block is None or self.block[0] != first:\n',
             '        if self.block is None:\n', 1)),
        ('no anchors off NVIDIA',
         sub(ME_GROUP, "LANE_ANCHORS = ('amd', 'intel')\n", "LANE_ANCHORS = ()\n", 1)),
        ('an element\'s rows read without the column\'s anchor',
         sub(ME_SECTION, '                source, self.anchored(_address(b, tensor, first, col)),\n',
             '                source, _address(b, tensor, first, col),\n', 1)),
        ('every walk at its pass\'s anchor',
         sub(ME_SECTION, '        if self.anchor_seed is None:\n            return\n        b = self.builder\n',
             '        if self.anchor_seed is None or self.anchor is not None:\n            return\n'
             '        b = self.builder\n', 1)),
        ('AMD\'s anchor a plain zero',
         sub(HIP_LEXIC, "    return 'tensorforge::opaqueZero()'\n", "    return '0'\n", 1)),
        ('the operators preloaded for the layout\'s kernels',
         sub(GEN, '      # one nothing reads.\n      self._preload_globals = False\n',
             '      # one nothing reads.\n', 1)),
        ('the lanes form\'s description reassociated on AMD',
         sub(GEN, '    if options.reassociate_products and not self._lanes_take(context):\n',
             '    if options.reassociate_products:\n', 1)),
        ('the lanes form\'s pointers read as the wave\'s',
         sub(GEN, '    if self._element_group is not None and self._element_group.lanes:\n'
                  '      return False\n    return (layout.whole_waves\n',
             '    return (layout.whole_waves\n', 1)),
        ('an element group asking for a group narrower than the block',
         sub(GEN, '                         narrow_group=self._element_group is None\n'
                  '                         and self._num_threads\n',
             '                         narrow_group=self._num_threads\n', 1)),
        ('a kernel the layout takes cut by fission',
         sub(GEN, '    if self._element_group is not None:\n'
                  '      # The multi-element layout\'s section is the kernel\'s whole list, which\n',
             '    if False:\n'
                  '      # The multi-element layout\'s section is the kernel\'s whole list, which\n', 1)),
        ('the lanes form\'s register file left to the option',
         sub(GEN, '      grf = element_group.lane_grf(self._context) or grf\n',
             '      pass\n', 1)),
        # Milestone 3b: the lanes form's FMAs as the one operation each is,
        # and its passes of several columns.
        ('the lanes form\'s FMAs infix where the target spells them fused',
         sub(ME_SECTION, "        self.fma_attrs = ((('fused', True), ('no_inline', True))\n",
             "        self.fma_attrs = (()\n", 1)),
        ('the lanes form\'s fused FMAs inlined into one expression',
         sub(ME_SECTION, "        self.fma_attrs = ((('fused', True), ('no_inline', True))\n",
             "        self.fma_attrs = ((('fused', True),)\n", 1)),
        ('a fused FMA spelled infix',
         sub(EMIT, '        if fused and lex is not None:\n', '        if False:\n', 1)),
        ('HIP\'s fused FMA not spelled',
         sub(HIP_LEXIC, "    return f'fma({a}, {b}, {c})'\n", '    return None\n', 1)),
        ('a pass\'s columns all its first',
         sub(ME_SECTION, "                           c0 + j * classes, 'p') for j in range(columns)]\n",
             "                           c0, 'p') for j in range(columns)]\n", 1)),
        ('a pass\'s columns sharing their small operands',
         sub(ME_SECTION, '                key = (id(p.second.tensor), q, self.slot)\n',
             '                key = (id(p.second.tensor), q)\n', 1)),
        ('a pass\'s columns sharing their own operands',
         sub(ME_SECTION, '            key = (id(product.second.tensor), row, self.slot)\n',
             '            key = (id(product.second.tensor), row)\n', 1)),
        ('a pass\'s output written to its first column',
         sub(ME_SECTION, '        col = trip.cols[j]\n', '        col = trip.cols[0]\n', 1)),
        # The sparse star: an element's small operand stored sparse, read
        # through a table of its cells' places.
        ('an operator stored sparse taken',
         sub(ME_GROUP, '                    and not (kind is Kind.MIXING and sub is second)):\n',
             '                    and False):\n', 1)),
        ('a small operand stored sparse refused',
         sub(ME_GROUP, '                    and not (kind is Kind.MIXING and sub is second)):\n',
             '                    and True):\n', 1)),
        ('a sparse operand staged as a box',
         sub(ME_SECTION, '            if buffer.tensor.is_dense():\n', '            if True:\n', 1)),
        ('a lane reading a sparse operand as a box',
         sub(ME_SECTION, '        if not tensor.is_dense():\n            return _sparse_read(',
             '        if False:\n            return _sparse_read(', 1)),
        ('a cell a sparse operand does not store read as stored',
         sub(ME_SECTION, "    return b.op('select', fp, stored, value, b.const(0.0, fp), hint=hint)\n",
             '    return value\n', 1)),
        ('a sparse operand\'s cells one place off',
         sub(ME_SECTION, '            out.append(int(tensor.linear_index(cell))\n',
             '            out.append(int(tensor.linear_index(cell)) + 1\n', 1)),
        # Milestone 3h: a chained section -- SeisSol's derivative -- in the
        # lanes form, its phases one after another in one warp.
        ('a scaled copy merged into the one before',
         sub(ME_GROUP, '                and p.kind is not Kind.SCALE):\n', '                ):\n', 1)),
        ('every scaled copy in the first phase',
         sub(ME_GROUP, '                (first if at < 0 else runs[at][1]).append(step)\n',
             '                first.append(step)\n', 1)),
        ('a scaled copy placed by what it reads alone',
         sub(ME_GROUP, '                    step.products[0].first.tensor, step.dest)), default=-1)\n',
             '                    step.products[0].first.tensor,)), default=-1)\n', 1)),
        ('an output read before its last write taken',
         sub(ME_GROUP, '            if max(writes[id(t)]) > n:\n', '            if False:\n', 1)),
        ('an output read where nothing wrote it taken',
         sub(ME_GROUP, '            if not any(_covers(r, rows) and _covers(c, cols)\n'
                       '                       for r, c in spans_of[id(t)]):\n',
             '            if False:\n', 1)),
        ('a window assigned outside a chain taken',
         sub(ME_GROUP, '    if windows and not chained:\n', '    if False:\n', 1)),
        ('a chained section in the warp form',
         sub(ME_GROUP, '    if section.chained:\n        # The warp form stages',
             '    if False:\n        # The warp form stages', 1)),
        ('`auto` taking a chained section',
         sub(ME_GROUP, "    if section.chained:\n        return None, ('`auto` takes no chained section",
             "    if False:\n        return None, ('`auto` takes no chained section", 1)),
        ('a phase in several classes',
         sub(ME_GROUP, '    if one_warp:\n        classes = 1\n', '    if one_warp:\n        pass\n', 1)),
        ('the factors not the kernel\'s symbols',
         sub(GEN, '                         + [t for t in section.scalars\n',
             '                         + [t for t in ()\n', 1)),
        ('a pass of copies alone computing',
         sub(ME_SECTION, '        self.computes = any(s.kind is not self.kinds.SCALE\n',
             '        self.computes = True or any(s.kind is not self.kinds.SCALE\n', 1)),
        ('the phase of copies alone dropped',
         sub(ME_SECTION, '        trip.fused = phase.scales\n',
             '        trip.fused = phase.scales if phase.main else ()\n', 1)),
        ('the integral\'s sums dropped',
         sub(ME_SECTION, '        trip.fused = phase.scales\n',
             '        trip.fused = () if phase.main else phase.scales\n', 1)),
        ('the integral\'s sum of an order read back from memory',
         sub(ME_SECTION, '        held = trip.holds is not None and trip.holds is source\n',
             '        held = False\n', 1)),
        ('a scaled copy without its factor',
         sub(ME_SECTION, '        scale = None if p.second is None else _scalar(trip, p.second.tensor)\n',
             '        scale = None\n', 1)),
        ('a scaled sum assigning',
         sub(ME_SECTION, '                if p.add:\n                    old = trip.own(dest, i, trip.col)\n',
             '                if False:\n                    old = trip.own(dest, i, trip.col)\n', 1)),
        ('a scaled copy\'s products counted for the table',
         sub(ME_SECTION, '            if p.kind is self.kinds.OPERATOR)\n',
             '            if p.kind is not self.kinds.MIXING)\n', 1)),
        ('a table for a pass that reads no number',
         sub(ME_SECTION, '        if not self.sequence:\n', '        if False:\n', 1)),
        # The star step beside the matrix tiling, block-wide (`mix`).
        ('the star step a warp an element beside the tiling',
         sub(ME_GROUP, '    if any(s.lowering in MATRIX_LOWERINGS for s in steps):\n'
                       '        # The matrix tiling\'s mixing steps block-wide',
             '    if False:\n        # The matrix tiling\'s mixing steps block-wide', 1)),
        ('the three directions a pass each',
         sub(ME_SECTION, "        if (step.lowering == 'mix' and run and run[0].lowering == 'mix'\n",
             "        if (False and run and run[0].lowering == 'mix'\n", 1)),
        ('the group operand read an element at a time',
         sub(ME_SECTION, '        loads, stores = fits(source, rows[0]), [\n', '        loads, stores = False, [\n', 1)),
        ('the consumer\'s buffers written an element at a time',
         sub(ME_SECTION, '            fits(section.buffer(step.dest), rows[0]) for step in steps]\n',
             '            False for step in steps]\n', 1)),
        ('a row of A a thread, the block\'s other threads idle',
         sub(ME_SECTION, '    sets = threads // count\n', '    sets = 1\n', 1)),
        ('a row\'s element taken as the warp\'s',
         sub(ME_SECTION, "    e = b.op('div', INDEX, m, width, hint='e')\n    col = _index(",
             "    e = trip.slot\n    col = _index(", 1)),
        ('a block of rows a row off',
         sub(ME_SECTION, "            first = b.op('mul', INDEX, g, vec, hint='l')\n",
             "            first = b.op('add', INDEX, b.op('mul', INDEX, g, vec, hint='l'), 1, hint='l')\n", 1)),
        ('a block\'s rows held off a block early',
         sub(ME_SECTION, "                'lt', BOOL, first, height - v, hint='in')\n",
             "                'lt', BOOL, first, height - v - vec, hint='in')\n", 1)),
        # The copies of a trip, where pitched inputs go: an input has a
        # pitch only where the matrix tiling reads it, on NVIDIA from sm_80
        # on, where every dense input is copied by cp.async, and a group
        # operand is dense -- the loads and stores never stage one.
        ('a pitched input staged over its box\'s rows',
         sub(ME_SECTION, '            total = pitch * width\n',
             '            total = height * width\n', 1)),
        # The matrix tiling's kept tiles of B and its walk over units: a
        # warp's N tiles in slots, its M tiles in groups (`TileUnits`).
        ('a tile of zeros kept',
         sub(ME_GROUP, '    if not sparse or pattern is None:\n', '    if True:\n', 1)),
        ('a stored tile\'s B written to the next N tile',
         sub(ME_SECTION, "        nts = self._table('mbn', [nt for _, nt in stage.tiles])\n",
             "        nts = self._table('mbn', [nt + 1 for _, nt in stage.tiles])\n", 1)),
        ('a slot\'s tile read where it holds zeros',
         sub(ME_SECTION, "                              b.if_(b.op('ge', BOOL, place, 0, hint='in'))):\n",
             "                              b.if_(b.op('ge', BOOL, place, -1, hint='in'))):\n", 1)),
        ('a group\'s M tile past its unit kept',
         sub(ME_SECTION, "                            'lt', BOOL, mq, hi, hint='in'), b.op(\n",
             "                            'lt', BOOL, mq, b.op('add', INDEX, hi, 1, hint='g'), hint='in'), b.op(\n", 1)),
        # Not 'A split again for every slot' (`halves` None, so that each
        # slot's products split A themselves): the same values, and with one
        # slot a warp, the default, the same code -- a cost no test of the
        # numbers can see.  The split's reuse that changes values instead: an
        # M tile of the group multiplied with another's A.
        ('every M tile of a group multiplied with the first one\'s A',
         sub(ME_SECTION, '                                    else halves[q])\n',
             '                                    else halves[0])\n', 1)),
        ('a warp walking no unit',
         sub(ME_SECTION, "    with b.for_(read(t_warp, w, 'u'), read(t_warp, _index(b, 'add', w, 1,\n",
             "    with b.for_(read(t_warp, _index(b, 'add', w, 1, 'u'), 'u'), read(t_warp, _index(b, 'add', w, 1,\n", 1)),
        ('every unit to the first warp',
         sub(ME_GROUP, '            w = min(range(warps), key=lambda w: (load[w], w))\n',
             '            w = 0\n', 1)),
        ('the grid\'s rows below the step written',
         sub(ME_SECTION, '            if shift:\n', '            if False:\n', 1)),
        # A in TF32 halves (`element_group.Buffer.halves`): the star step
        # writes both halves of each value, the tiling reads both; the rule
        # that picks the buffers, the fill rule's order, the room.
        ('A in halves read as values and split again',
         sub(ME_SECTION, '                    if buffer.halves:\n'
                         '                        # A as its producer split it',
             '                    if False:\n'
             '                        # A as its producer split it', 1)),
        ('A\'s lower halves read from the upper block',
         sub(ME_SECTION, '                        lower = buffer.half_offset(trip.group.elements)\n',
             '                        lower = 0\n', 1)),
        ('the star step\'s lower halves stored over the upper',
         sub(ME_SECTION, "                              (_index(b, 'add', at,\n"
                         "                                      dest.half_offset(group.elements), 'a'),\n",
             '                              (at,\n', 1)),
        ('the upper halves stored as the values',
         sub(ME_SECTION, '                    writes = [(at, [_as_float(b, u) for u, _ in split]),\n',
             '                    writes = [(at, values),\n', 1)),
        ('the lower halves\' padding left unwritten',
         sub(ME_SECTION, '            if buffer.halves:\n'
                         '                b.store(target, zero, _index(\n',
             '            if False:\n'
             '                b.store(target, zero, _index(\n', 1)),
        ('A\'s halves taken where they cost the group',
         sub(ME_GROUP, '                out.extend(_halved(t, n) for n in range(most, -1, -1)\n',
             '                out.extend(_halved(t, n) for n in [most]\n', 1)),
        ('B\'s halves preferred to A\'s',
         sub(ME_GROUP, '                out.extend(_halved(t, n) for n in range(most, -1, -1)\n'
                       '                           for t in part)\n',
             '                out.extend(_halved(t, n) for t in part\n'
             '                           for n in range(most, -1, -1))\n', 1)),
        ('the room without the lower halves',
         sub(ME_GROUP, '    per_element = sum(b.room for b in section.inputs\n',
             '    per_element = sum(b.per_element for b in section.inputs\n', 1)),
        ('the lower halves an element\'s box past the upper',
         sub(ME_GROUP, '        return elements * self.per_element\n',
             '        return self.per_element\n', 1)),
        ('A in halves where the tiling writes it',
         sub(ME_GROUP, "             and writers.get(id(b.tensor)) == {'mix'}]\n",
             '             and bool(writers.get(id(b.tensor)))]\n', 1)),
        ('the buffers in halves the least read first',
         sub(ME_GROUP, '    found = [(-reads[id(b.tensor)], n, b)\n',
             '    found = [(reads[id(b.tensor)], n, b)\n', 1)),
        ('the section lowered without the plan\'s halves',
         sub(GEN, '          halves=element_group.halves(planned),\n',
             '          halves=0,\n', 1)),
        # The layout's switch for them (`HALVES`), off by default.
        ('A\'s halves on by default',
         sub(ME_GROUP, 'HALVES = False\n', 'HALVES = True\n', 1)),
        ('A\'s halves taken with the switch off',
         sub(ME_GROUP, '                if asynchronous(lowerings[0]) else HALVES):\n',
             '                if asynchronous(lowerings[0]) else True):\n', 1)),
    ]),
    # The consolidation of the engines layer (`pir/engines.py`).  Matrix
    # transfers: ldmatrix and stmatrix as transfer engines (`Op.MATRIX_LOAD`,
    # `Op.MATRIX_STORE`) -- the fragment layout they carry, what the
    # verifier holds them to, the text and the targets, the figures, the
    # builder, the oracle's model, and the helpers in `cuda.h`, the last
    # two caught on the local GPU (sm_90 or newer).  Copies that fill
    # (`copy_async`'s `valid`) and copies under a predicate of their own:
    # the engines that fill, the operand, the builder, the text and the
    # synchronous form, the oracle's model of both zero-fills, `pir.wrap`
    # taking both, and the helper (on the local GPU, sm_80 or newer).  A
    # buffer's own request of the transfer into it (`alloc`'s `ahead`):
    # what the builder takes, and which transfers wrap picks, at what stages.
    # RELAYOUT: AMD's relayouts as the calls they were, the relabel and its
    # solver, the shuffle and the oracle's reading of it.
    'pircons': ('tests/test_matrix_transfers.py tests/test_copy_fill.py '
                'tests/test_wrap_ahead.py tests/test_relayout.py '
                'tests/test_transfer_engines.py tests/test_block2d.py '
                'tests/test_wmma.py tests/test_cache_policies.py '
                'tests/test_multi_element.py::test_ldmatrix_is_a_declared_read_of_its_buffer', [
        ('the rows of a matrix on the lane bits of its words',
         sub(ENGINES, '    rows = (tuple(Bit(Place.LANE, 4 << k) for k in range(3))\n',
             '    rows = (tuple(Bit(Place.LANE, 1 << k) for k in range(3))\n', 1)),
        ('a placement off the sums of its steps taken',
         sub(ENGINES, '        if at != want:\n            return None\n',
             '        if False:\n            return None\n', 1)),
        ('a gap of rows between the matrices taken',
         sub(ENGINES, '    if any(r != MATRIX_ROWS << k for k, (r, _) in enumerate(down)) or any(\n',
             '    if any(r % MATRIX_ROWS for k, (r, _) in enumerate(down)) or any(\n', 1)),
        ('a count the instruction lacks accepted',
         sub(ENGINES, "    if count not in MATRIX_TRANSFER_COUNTS:\n        return (f'{count!r} matrices;",
             "    if False:\n        return (f'{count!r} matrices;", 1)),
        ('a predicated transfer accepted',
         sub(ENGINES, "    if s.predicate is not None:\n        return ('every lane of the warp takes part",
             "    if False:\n        return ('every lane of the warp takes part", 1)),
        ('a register of 64 bits accepted',
         sub(ENGINES, '                and callable(size) and size() == 4):\n',
             '                and callable(size)):\n', 1)),
        ('a store held to the read of a load',
         sub(ENGINES, '    kind = Effect.READ if s.op == Op.MATRIX_LOAD else Effect.WRITE\n',
             '    kind = Effect.READ\n', 1)),
        ('stmatrix on the architecture of ldmatrix',
         sub(ENGINES, "MATRIX_TRANSFER_SM = {'ldmatrix': 75, 'stmatrix': 90}\n",
             "MATRIX_TRANSFER_SM = {'ldmatrix': 75, 'stmatrix': 75}\n", 1)),
        ('ldmatrix written for every backend',
         sub(ENGINES, "                  '': _lower_matrix_transfer, '*': _no_matrix_transfer},\n",
             "                  '': _lower_matrix_transfer, '*': _lower_matrix_transfer},\n", 1)),
        ('the registers of a load not declared',
         sub(ENGINES, "        return Lowered(tuple(f'{v.type.base.ctype()} {n}{{}};'\n"
                      "                             for v, n in zip(registers, names))\n"
                      "                       + (f'tensorforge::ldmatrixX{count}('\n",
             "        return Lowered((f'tensorforge::ldmatrixX{count}('\n", 1)),
        ('the registers of a store ahead of its row',
         sub(ENGINES, "    return Lowered((f'tensorforge::stmatrixX{count}({where}, '\n"
                      "                    f'{\", \".join(names)});',))\n",
             "    return Lowered((f'tensorforge::stmatrixX{count}({\", \".join(names)}, '\n"
             "                    f'{where});',))\n", 1)),
        ('a load of integers written for the float helpers',
         sub(ENGINES, "        if v.type.base != Datatype.F32:\n            raise IRError(f'{s.attr(\"engine\")} of",
             "        if False:\n            raise IRError(f'{s.attr(\"engine\")} of", 1)),
        ('ldmatrix counted as code',
         sub(ENGINES, '           units=0, pipe=_matrix_transfer_pipe,\n',
             '           units=1, pipe=_matrix_transfer_pipe,\n', 1)),
        ('stmatrix counted as no issue',
         sub(ENGINES, "    return 'shared.store' if s.op == Op.MATRIX_STORE else None\n",
             '    return None\n', 1)),
        ('the cover counting every engine op as one',
         sub(TRANSFERS, '            n += units_of(s)\n', '            n += 1\n', 1)),
        ('the read of ldmatrix declared as a write',
         sub(BUILD, '        access = Access(Effect.READ, self._space_of(base),\n'
                    '                        self.alias_root(base))\n'
                    '        self._emit_transfer(engines.matrix_load(',
             '        access = Access(Effect.WRITE, self._space_of(base),\n'
             '                        self.alias_root(base))\n'
             '        self._emit_transfer(engines.matrix_load(', 1)),
        ('a transfer the instruction is not built unrefused',
         sub(BUILD, '        why = engines.matrix_transfer_refusal(s)\n        if why is not None:\n',
             '        why = engines.matrix_transfer_refusal(s)\n        if False:\n', 1)),
        ('the fragment layout of a placement dropped',
         sub(BUILD, '        return self._swizzled(base, index), layout\n',
             '        return self._swizzled(base, index), None\n', 1)),
        ('a row of a swizzled buffer not permuted',
         sub(BUILD, '        return self._swizzled(base, index), layout\n',
             '        return index, layout\n', 1)),
        ('a permutation that splits a row taken',
         sub(BUILD, '        self._check_width(base, engines.MATRIX_ROW_BYTES // elem.size())\n',
             '', 1)),
        ('the oracle loading a row of the wrong lanes',
         sub(ORACLE, '            self.env[parts[i]] = rows[8 * i + t // 4][t % 4]\n',
             '            self.env[parts[i]] = rows[8 * i + t % 8][t % 4]\n', 1)),
        ('the oracle storing a word of the wrong lane',
         sub(ORACLE, '            base[at + w] = lanes[4 * r + w].ev(parts[2 + i])\n',
             '            base[at + w] = lanes[4 * w + r].ev(parts[2 + i])\n', 1)),
        ('the oracle taking a warp that is not whole',
         sub(ORACLE, "        if len(lanes) != 32:\n            raise Abort(f'{what} with",
             "        if False:\n            raise Abort(f'{what} with", 1)),
        ('the helper of two matrices loading one',
         sub(Path('src/tensorforge/include/tensorforge_device/cuda.h'),
             '  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\\n"\n',
             '  asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0,%1}, [%2];\\n"\n', 1)),
        ('the helper of four stores storing a register twice',
         sub(Path('src/tensorforge/include/tensorforge_device/cuda.h'),
             '               "r"(__float_as_uint(r2)), "r"(__float_as_uint(r3))\n'
             '               : "memory");\n',
             '               "r"(__float_as_uint(r1)), "r"(__float_as_uint(r3))\n'
             '               : "memory");\n', 1)),
        ('the store helper trapping on sm_90',
         sub(Path('src/tensorforge/include/tensorforge_device/cuda.h'),
             '__device__ __forceinline__ void stmatrixX1(float *base, int index, float r0) {\n'
             '#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900\n',
             '__device__ __forceinline__ void stmatrixX1(float *base, int index, float r0) {\n'
             '#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000\n', 1)),
        # Copies that fill, and copies under a predicate of their own.
        ('cp.async filling with nothing',
         sub(ENGINES, "                   fill='zero'),\n", "                   fill=None),\n", 1)),
        ('a fill of another value than its engine writes accepted',
         sub(ENGINES, '    if fill != fill_of(engine):\n', '    if False:\n', 1)),
        ('a fill under a condition that is no boolean accepted',
         sub(ENGINES, "            and getattr(getattr(t, 'base', None), 'name', None) == 'BOOL'):\n",
             '            ):\n', 1)),
        ('the condition of a fill taken for an index',
         sub(CORE, "        end = (len(self.args) - (self.attr('fill') is not None)\n",
             "        end = (len(self.args)\n", 1)),
        ('a copy that fills read as one that does not',
         sub(CORE, "        return self.args[-1] if self.attr('fill') is not None else None\n",
             '        return None\n', 1)),
        ('a fill through an engine without one built',
         sub(BUILD, '            why = engines.fill_refusal(s)\n            if why is not None:\n'
                    "                raise IRError(f'a copy that fills: {why}')\n",
             '            why = engines.fill_refusal(s)\n            if False:\n'
             "                raise IRError(f'a copy that fills: {why}')\n", 1)),
        ('the zero-filled tail of a copy that fills read',
         sub(Path('src/tensorforge/common/vm/lexic/cuda_lexic.py'),
             "              f'({valid}) ? {nbytes - zfill}u : 0u);')\n",
             "              f'({valid}) ? {nbytes}u : 0u);')\n", 1)),
        ('a copy that fills reading where its condition fails',
         sub(Path('src/tensorforge/common/vm/lexic/cuda_lexic.py'),
             "              f'({valid}) ? {nbytes - zfill}u : 0u);')\n",
             "              f'({valid}) ? 0u : {nbytes - zfill}u);')\n", 1)),
        ('the copies of AMD filling',
         sub(Path('src/tensorforge/common/vm/lexic/hip_lexic.py'),
             '    if zfill or valid is not None:\n', '    if zfill:\n', 1)),
        ('a synchronous copy that fills reading everywhere',
         sub(EMIT, "                    w(f'{dst_b}[{dst_a}] = ({valid}) ? {src_b}[{src_a}] : '\n"
                   "                      f'{zero};')\n",
             "                    w(f'{dst_b}[{dst_a}] = {src_b}[{src_a}];')\n", 1)),
        ('the oracle copying what a copy fills',
         sub(ORACLE, '                dst[di + k] = src[si + k] if k < read // width else 0.0\n',
             '                dst[di + k] = src[si + k]\n', 1)),
        ("the oracle reading the pipeline's zero-filled tail",
         sub(ORACLE, "            read = nbytes - int(self.ev(groups['zfill']))\n",
             '            read = nbytes\n', 1)),
        ("wrap dropping a copy's own predicate behind the flag",
         sub(WRAP, "                both = b.op('and', BOOL, cond, s.predicate, hint='g')\n"
                   '                return list(b.finish()) + [replace(s, predicate=both)]\n',
             "                both = b.op('and', BOOL, cond, s.predicate, hint='g')\n"
             '                return list(b.finish()) + [replace(s, predicate=cond)]\n', 1)),
        ('wrap refusing a predicated copy again',
         sub(WRAP, "        # guard where there is one (`_Rewrite._guarded`).\n\n",
             "        # guard where there is one (`_Rewrite._guarded`).\n"
             "        if any(x.predicate is not None for x in walk_stmts(tuple(t.pieces))\n"
             "               if x.op is Op.COPY_ASYNC):\n"
             "            raise Refusal('a copy of the transfer is predicated already')\n\n", 1)),
        ('the helper of a copy that fills reading every byte',
         sub(Path('src/tensorforge/include/tensorforge_device/cuda.h'),
             '                 "l"(from), "r"(srcBytes)\n', '                 "l"(from), "r"(16u)\n', 1)),
        # A buffer's own request of the transfer into it (`alloc`'s `ahead`).
        ('a ring of three stages asked for and taken',
         sub(BUILD, '    if stages not in AHEAD_STAGES:\n', '    if False:\n', 1)),
        ('a second stage asked for a register buffer and taken',
         sub(BUILD, '    if stages > 1 and space != MemSpace.SHARED:\n', '    if False:\n', 1)),
        ('a distance that is no count of transfers taken',
         sub(BUILD, '    if not isinstance(distance, int) or isinstance(distance, bool) \\\n',
             '    if False and not isinstance(distance, int) or isinstance(distance, bool) \\\n', 1)),
        ("a buffer's request not said",
         sub(BUILD, "            attrs = attrs + (('ahead', _ahead(ahead, space)),)\n",
             "            _ahead(ahead, space)\n", 1)),
        ("wrap counting every transfer at the pass's distance",
         sub(WRAP, '            if count.get(key, 0) < (distance if asked is None else asked[1]):\n',
             '            if count.get(key, 0) < distance:\n', 1)),
        ('wrap counting the places of the transfers it takes alone',
         sub(WRAP, '            if count.get(key, 0) < (distance if asked is None else asked[1]):\n'
                   '                picked.append(t)\n'
                   '            count[key] = count.get(key, 0) + 1\n',
             '            if count.get(key, 0) < (distance if asked is None else asked[1]):\n'
             '                picked.append(t)\n'
             '                count[key] = count.get(key, 0) + 1\n', 1)),
        ("wrap giving every buffer the pass's stages",
         sub(WRAP, '        stages = self.stages if asked is None else asked[0]\n',
             '        stages = self.stages\n', 1)),
        ('the windows of a second stage asking again',
         sub(WRAP, "                     if k not in ('identity', 'stages', 'extern', 'ahead')]\n",
             "                     if k not in ('identity', 'stages', 'extern')]\n", 1)),
        # RELAYOUT (`engines.RELAYOUT_STRATEGIES`): AMD's relayouts as the
        # calls they were, the relabel, the shuffle.
        ('a broadcast within a row counted as crossing the rows',
         sub(ENGINES, "    return 'xlane.wide' if int(params['threads']) > 16 else 'xlane'\n",
             "    return 'xlane.wide' if int(params['threads']) >= 16 else 'xlane'\n", 1)),
        ('a relabel counted as an instruction',
         sub(ENGINES, '        pipe=_no_issue, units=0,\n', '        pipe=_no_issue, units=1,\n', 1)),
        ('a relabel moving elements between lanes accepted',
         sub(ENGINES, '        if at_s.lane != at_t.lane or at_s.element != at_t.element:\n',
             '        if at_s.element != at_t.element:\n', 1)),
        ('a lane bit relabeled onto another lane bit',
         sub(ENGINES, '                   if c.place is b.place and c.weight == b.weight]\n',
             '                   if c.place is b.place][:1]\n', 1)),
        ('a relabel written in the order of its source',
         sub(ENGINES, "        for v, at in zip(s.target, s.attr('order')):\n",
             '        for at, v in enumerate(s.target):\n', 1)),
        ('a transpose without its writes declared accepted',
         sub(ENGINES, '        regs = list(s.args[:written])\n',
             '        regs = [a.base for a in s.accesses]\n', 1)),
        ('a transpose declared as writing nothing',
         sub(ENGINES, "    if op == Op.RELAYOUT and dict(attrs).get('written'):\n",
             "    if False:\n", 1)),
        ('two broadcasts of other lanes shared',
         sub(BUILD, '                         extra=(strategy, tuple(sorted(params.items())),\n',
             '                         extra=(strategy, (),\n', 1)),
        ("a transpose's outputs not pinned",
         sub(BUILD, '        for w in written:\n            self.pin(w)\n        return self.emit(engines.relayout(\n',
             '        return self.emit(engines.relayout(\n', 1)),
        ("demand reading every lane of a relayout's source",
         sub(DEMAND, '            elif u.op == Op.RELAYOUT:\n', '            elif False:\n', 1)),
        ('the code of a relayout counted without the statement',
         sub(EMIT, '        self._record_code(op, s)\n', '        self._record_code(op)\n', 1)),
        ("the oracle's gather taking one level of parentheses",
         sub(ORACLE, r"               r'\s*((?:[^()]|\((?:[^()]|\([^()]*\))*\))*)\)'," + "\n",
             r"               r'\s*([^()]*(?:\([^()]*\)[^()]*)*)\)'," + "\n", 1)),
        # The block reads in the transfer registry (`block`, `block2d`), and
        # every transfer held to a registered engine of its op.
        ('a transfer through no engine accepted',
         sub(ENGINES, '    found = TRANSFER_ENGINES.get(engine)\n    if found is None:\n',
             '    found = TRANSFER_ENGINES.get(engine)\n    if False:\n', 1)),
        ('a transfer through an engine of another op accepted',
         sub(ENGINES, '    if s.op not in found.ops:\n        return f', '    if False:\n        return f', 1)),
        ('the verifier not asking the engine of a transfer',
         sub(PASSES, '        why = engines.transfer_engine_refusal(s)\n        if why is not None:\n            diag.append(why)\n',
             '', 1)),
        ('a 2D block read through the block messages',
         sub(BLOCKS, "        attrs=(('engine', 'block2d'), ('shape', (rows, cols)),\n",
             "        attrs=(('engine', 'block'), ('shape', (rows, cols)),\n", 1)),
        # AMD's WMMA on the matrix op (`wmma`, `wmma_gfx1250`), issued from
        # the catalog's rows (`MatrixOp.mac`).
        ("a WMMA of RDNA through the MFMA's engine",
         sub(PKG / 'catalog.py', "            engine, modifiers = 'wmma', ()\n",
             "            engine, modifiers = 'mfma', ()\n", 1)),
        ("gfx1250's WMMA through RDNA's engine",
         sub(PKG / 'catalog.py', "            engine = 'wmma_gfx1250'\n",
             "            engine = 'wmma'\n", 1)),
        ("a WMMA's modifier stated as zero whatever was asked",
         sub(PKG / 'catalog.py', "            modifiers = tuple((name, int(wmma.get(name, 0)))\n",
             "            modifiers = tuple((name, 0)\n", 1)),
        ('modifiers on a WMMA of RDNA let through',
         sub(PKG / 'catalog.py', "            engine, modifiers = 'wmma', ()\n            if wmma:\n",
             "            engine, modifiers = 'wmma', ()\n            if False:\n", 1)),
        ("a modifier gfx1250's WMMA lacks let through",
         sub(PKG / 'catalog.py', "            if unknown:\n                raise ValueError(f'{self.builtin} has no modifier",
             "            if False:\n                raise ValueError(f'{self.builtin} has no modifier", 1)),
        ("a WMMA's layouts left behind",
         sub(PKG / 'catalog.py', "            modifiers=modifiers, layouts=vendor_layouts(self), term=term,\n",
             "            modifiers=modifiers, layouts=None, term=term,\n", 1)),
        ("gfx1250's modifiers out of their places among the operands",
         sub(ENGINES, "        args = [flag('a_neg'), a, flag('b_neg'), b,\n",
             "        args = [a, flag('a_neg'), flag('b_neg'), b,\n", 1)),
        ("a WMMA's flags written as numbers",
         sub(ENGINES, "            return 'true' if modifiers.get(name, 0) else 'false'\n",
             "            return '1' if modifiers.get(name, 0) else '0'\n", 1)),
        ('a WMMA lowered off HIP',
         sub(ENGINES, '    if env.family and env.family not in AMD_FAMILIES:\n'
             "        raise IRError(f'a WMMA on",
             "    if False:\n        raise IRError(f'a WMMA on", 1)),
        ("RDNA's WMMA spelled as gfx1250's",
         sub(ENGINES, "def _lower_wmma(s: Stmt, env: Spelling) -> Lowered:\n    return _wmma_call(s, env, gfx1250=False)\n",
             "def _lower_wmma(s: Stmt, env: Spelling) -> Lowered:\n    return _wmma_call(s, env, gfx1250=True)\n", 1)),
        # Cache policies on transfers (`engines.CACHE_POLICIES`): the kinds of
        # hint as they were, L2's through a policy, a box under one.
        ('an L2 policy spelled as the L1 bypass',
         sub(Path('src/tensorforge/common/vm/lexic/cuda_lexic.py'),
             "      evict = self._l2_evict(nontemporal)\n      if evict is not None:\n        return f'tensorforge::ldL2",
             "      evict = self._l2_evict(nontemporal)\n      if False:\n        return f'tensorforge::ldL2", 1)),
        ('an L2 policy before sm_80',
         sub(ENGINES, '                and level is not None and level >= self.sm\n',
             '                and level is not None\n', 1)),
        ('an L2 policy on a wide value',
         sub(ENGINES, '                and length == 1 and datatype in L2_HINT_TYPES)\n',
             '                and datatype in L2_HINT_TYPES)\n', 1)),
        ('a bare request taken as streaming',
         sub(ENGINES, "    return 'cg' if request is True else request\n",
             "    return 'cs' if request is True else request\n", 1)),
        ('a request for no policy of a load taken',
         sub(ENGINES, "    if policy is None or not {'scalar', 'vector'} & set(policy.engines):\n",
             '    if False:\n', 1)),
        ("the verifier not asking a transfer's cache policy",
         sub(PASSES, '        why = engines.cache_refusal(s)\n        if why is not None:\n            diag.append(why)\n',
             '', 1)),
        ('a policy of another engine taken',
         sub(ENGINES, "    if engine not in policy.engines:\n        return (f'{s.op} through {engine} with the cache policy",
             "    if False:\n        return (f'{s.op} through {engine} with the cache policy", 1)),
        ("a box's policy left off its copy",
         sub(EMIT, "            evict = engines.l2_evict_of(s.attr('cache'))\n",
             '            evict = None\n', 1)),
        ("the oracle's box without its policy's template",
         sub(ORACLE, r"    r'^tensorforge::tmaLoad\s*(?:<\s*[\w:]+\s*>\s*)?\(\s*(?P<cursor>\w+)\s*,'" + "\n",
             r"    r'^tensorforge::tmaLoad\s*\(\s*(?P<cursor>\w+)\s*,'" + "\n", 1)),
        ("createpolicy's priorities swapped",
         sub(Path('src/tensorforge/include/tensorforge_device/cuda.h'),
             '  if constexpr (P == L2Evict::Last) {\n',
             '  if constexpr (P == L2Evict::First) {\n', 1)),
        ('a store under a policy passed over by the oracle',
         sub(ORACLE, '        sl = _STORE_L2.match(stmt)\n', '        sl = None\n', 1)),
    ]),
    # An `if` whose first arm writes nothing: the writer leaves that arm out,
    # its head with it, and the second is written under the negated
    # condition rather than as an `else` of nothing (`Emitter._emit_if`).
    'emitarms': ('tests/test_emit_empty_arm.py', [
        ('the second arm an else of nothing',
         sub(EMIT, "            other = (w.If(f'!({cond})') if getattr(then, 'elided', False)\n",
             "            other = (w.If(f'!({cond})') if False\n", 1)),
        ('the second arm under the condition itself',
         sub(EMIT, "            other = (w.If(f'!({cond})') if getattr(then, 'elided', False)\n",
             "            other = (w.If(cond) if getattr(then, 'elided', False)\n", 1)),
        ('a block not saying it was left out',
         sub(Path('src/tensorforge/backend/writer.py'),
             "    self.elided = self.writer.speculateClear('}' + self.foot)\n",
             "    self.writer.speculateClear('}' + self.foot)\n", 1)),
    ]),
    # Programmatic Dependent Launch (`Options.programmatic_launch`): the wait
    # for the grid launched ahead, where it stands (`pir.griddep`), the
    # launch, and the targets and launches it is for.
    'pdl': ('tests/test_programmatic_launch.py', [
        ('the wait behind the first access it is for',
         sub(GRIDDEP, '            return body[:i] + (engines.grid_wait(),) + body[i:], True\n',
             '            return body[:i + 1] + (engines.grid_wait(),) + body[i + 1:], True\n', 1)),
        ('an operand the kernel writes read ahead of the wait',
         sub(GRIDDEP, '    if id(tensor) in writes:\n        return False\n', '', 1)),
        ('every operand read ahead of the wait',
         sub(GRIDDEP, "    return bool(getattr(tensor, 'time_invariant', False))\n",
             '    return True\n', 1)),
        ('the caller\'s statement of an invariant operand ignored',
         sub(GRIDDEP, "    return bool(getattr(tensor, 'time_invariant', False))\n",
             '    return False\n', 1)),
        ('opaque text read as touching nothing',
         sub(GRIDDEP, '    if collect_effect((s,)) & Effect.UNKNOWN:\n        return True\n', '', 1)),
        ('an access of no operand read as free',
         sub(GRIDDEP, '        if not tensors or not all(free(t, writes) for t in tensors):\n',
             '        if tensors and not all(free(t, writes) for t in tensors):\n', 1)),
        ('an access in an unknown space read as free',
         sub(GRIDDEP, '_SPACES = (MemSpace.GLOBAL, MemSpace.UNKNOWN)\n',
             '_SPACES = (MemSpace.GLOBAL,)\n', 1)),
        ('shared memory read as another kernel\'s',
         sub(GRIDDEP, '        if a.space not in _SPACES:\n            continue\n', '', 1)),
        ('the wait written as nothing on CUDA',
         sub(ENGINES, "    return Lowered(('tensorforge::gridDependencyWait();',))\n",
             '    return Lowered()\n', 1)),
        ('the oracle not reading the wait',
         sub(ORACLE, "    r'|tensorforge::gridDependencyWait'\n", '', 1)),
        ('no wait placed',
         sub(GEN, '    if not self._programmatic():\n      return\n    from tensorforge.backend.pir import griddep\n',
             '    return\n    from tensorforge.backend.pir import griddep\n', 1)),
        ('the ordinary launch for a programmatic kernel',
         sub(GEN, '        if self._programmatic():\n          return lexic.get_programmatic_launch_code(\n',
             '        if False:\n          return lexic.get_programmatic_launch_code(\n', 1)),
        ('a programmatic launch below sm_90',
         sub(GEN, '    if nvidia.sm_of(self._context) < 90:\n      return False\n    return not self._launch.cooperative\n',
             '    return not self._launch.cooperative\n', 1)),
        ('a programmatic launch off CUDA',
         sub(GEN, "    if (target.backend != 'cuda'\n        or getattr(target.hw, 'vendor', None) != 'nvidia'):\n      return False\n",
             '', 1)),
        ('a cooperative kernel launched programmatically',
         sub(GEN, '    return not self._launch.cooperative\n', '    return True\n', 1)),
        ('the launch spelled without its function',
         sub(CUDA_LEXIC, '    return (f"tensorforge::launchProgrammatic({func_name}, {grid}, {block}, "\n',
             '    return (f"{func_name}<<<{grid}, {block}, "\n', 1)),
        ('the wait compiled out on sm_90',
         sub(CUDA_H, '#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900\n  asm volatile("griddepcontrol.wait;"',
             '#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000\n  asm volatile("griddepcontrol.wait;"', 1)),
        ('a flag word read ahead of the wait',
         sub(GRIDDEP, "    if any(x.attr('text_reads') in _SPACES for x in walk_stmts((s,))):\n        return True\n",
             '', 1)),
        ('the read a text states left unrecorded',
         sub(BUILD, "            stated = ((('text_reads', space),)\n"
                    "                      if space is not None and kind & Effect.READ else ())\n",
             '            stated = ()\n', 1)),
        ('the runtime\'s launch helper launching plainly',
         sub(CUDA_H, '  return cudaLaunchKernelEx(&config, kernel, std::forward<Args>(args)...);\n',
             '  kernel<<<grid, block, shared, stream>>>(std::forward<Args>(args)...);\n'
             '  return cudaGetLastError();\n', 1)),
    ]),
    # A grid barrier as a launch boundary (`GridBarrierDescr` without
    # `Options.cooperative_grid`): the cut, each launch's kernel, grid and
    # wait, and the cooperative launch behind the option.
    'launchcut': ('tests/test_launch_boundary.py '
                  '--deselect tests/test_launch_boundary.py::test_gpu_two_launches_compute_what_the_cooperative_one_does '
                  'tests/test_kernels.py::test_barriers_cases_construct_and_generate '
                  'tests/test_launch_control.py::test_a_grid_barrier_is_a_launch_boundary_of_two_queues', [
        ('a grid barrier kept in one cooperative launch',
         sub(GEN, "    if getattr(self._context.get_user_options(), 'cooperative_grid', False):\n      return False\n    return self._context.target.backend in ('cuda', 'hip')\n",
             '    return False\n', 1)),
        ('the cooperative launch where it is not asked for',
         sub(GEN, "    if getattr(self._context.get_user_options(), 'cooperative_grid', False):\n      return False\n",
             '', 1)),
        ('the launch after the cut started rotated',
         sub(GEN, '      if ssection.barrier or ssection.block_barrier or ssection.launch_cut:\n',
             '      if ssection.barrier or ssection.block_barrier:\n', 1)),
        ('every launch sized by the first count',
         sub(GEN, '    head = self._sections[self._launch_groups()[launch_index][0]].count_index\n',
             '    head = 0\n', 1)),
        ('the launch after the cut never issued',
         sub(GEN, '      for launch in range(1, len(groups)):\n        suffix = self._launch_suffix(launch)\n',
             '      for launch in range(1, 1):\n        suffix = self._launch_suffix(launch)\n', 1)),
        ('one wait for all the launches',
         sub(GEN, '        if placed:\n          section.body = body\n          break\n',
             '        if placed:\n          section.body = body\n          return\n', 1)),
        ('what another launch writes read as free',
         sub(GEN, '    writes = griddep.written(section.body for section in self._sections)\n    for group in self._launch_groups():\n',
             '    for group in self._launch_groups():\n      writes = griddep.written(self._sections[i].body for i in group)\n', 1)),
        ('the sections after the cut in the first kernel',
         sub(GEN, '      if section.launch_cut and i + 1 < len(self._sections):\n        groups.append([])\n',
             '', 1)),
    ]),
    # The multi-element layout's producer (`Options.element_producer`): the
    # ring's protocol as the section writes it, the roles, the producer's
    # copies and the traversal it shares with the element warps.  The host
    # oracle's numbers, races and waits.
    'producer': ('tests/test_element_producer.py', [
        ('the producer waiting for its consumers\' parity',
         sub(BUILD, "            'sub', INDEX, 1, parity, hint='free'), writes=writes, reads=reads)\n",
             "            'add', INDEX, 0, parity, hint='free'), writes=writes, reads=reads)\n", 1)),
        ('the slot\'s bytes announced short',
         sub(ME_SECTION, '            nbytes += elements * width * rows * section.dtype.size()\n',
             '            nbytes += elements * width * (rows - 4) * section.dtype.size()\n', 1)),
        ('a slot given back by the first lane alone',
         sub(ME_SECTION, "        b.barrier(Participants.WAVE)\n        b.ring_release(ring, slot,",
             "        b.ring_release(ring, slot,", 1)),
        ('the bytes announced ahead of the lanes\' stores',
         sub(ME_SECTION, "        b.barrier(Participants.WAVE)\n        b.ring_publish(ring, slot,",
             "        b.ring_publish(ring, slot,", 1)),
        ('a slot free once one element warp is done',
         sub(ME_SECTION, "            'me', section.ring, full=1, empty=elements,\n",
             "            'me', section.ring, full=1, empty=1,\n", 1)),
        ('every trip in the first slot',
         sub(ME_SECTION, "                hint=_hint('r', buffer.tensor), stages=section.ring,\n                stage=slot, identity",
             "                hint=_hint('r', buffer.tensor), stages=section.ring,\n                stage=0, identity", 1)),
        ('the producer\'s rows starting a group of their own',
         sub(Path('src/tensorforge/backend/instructions/batch_loop.py'),
             "            start = f'{self._group_size} * ({lexic.block_idx_x})'\n",
             "            pass\n", 1)),
        ('the traversal striding the block\'s rows',
         sub(GEN, "      stride = f'({lexic.grid_dim_x} * {self._element_group.elements})'\n    if not offset:",
             "      stride = f'({lexic.grid_dim_x} * {lexic.block_dim_y})'\n    if not offset:", 1)),
        ('a column copied without its rounding',
         sub(ME_GROUP, '    return -(-buffer.height // 4) * 4\n',
             '    return buffer.height\n', 1)),
        ('the element warps meeting the block',
         sub(ME_SECTION, '        trip.meet = Participants.ROLE\n',
             '        trip.meet = Participants.BLOCK\n', 1)),
        ('the slot zeroed nowhere',
         sub(ME_SECTION, "                    builder.store(window, builder.const(0.0, fp),\n                                  loop.induction)\n",
             "                    pass\n", 1)),
        ('the copies\' column a pitch apart in the source too',
         sub(ME_SECTION, "                    dst_index=(_index(b, 'add', base, c * buffer.pitch,\n",
             "                    dst_index=(_index(b, 'add', base, c * (buffer.pitch + 4),\n", 1)),
        ('the intermediates\' padding left as the kernel finds it',
         sub(ME_SECTION, '        for buffer in padded:\n            self._pad(trip, buffer)\n',
             '', 1)),
        ('the producer\'s copies not waited for',
         sub(ME_SECTION, '            b.wait(*copies)\n', '', 1)),
        ('every element\'s cells into the first box',
         sub(ME_SECTION, "                base = b.op('mul', INDEX, mine, buffer.per_element, hint='b')\n",
             "                base = 0\n", 1)),
        ('a hop of the producer\'s lanes skipped',
         sub(ME_SECTION, "                for hop in range(0, total, per):\n",
             "                for hop in range(0, total, 2 * per):\n", 1)),
        # A sparse star's stored cells (`_stored_cells`): each from its place
        # among the stored entries to where the element's box holds it.
        ('a sparse star\'s cells read at their place in the box',
         sub(ME_SECTION, "            place = places[(row - lower[0]) + (col - lower[1]) * height]\n",
             "            place = ((row - lower[0]) + (col - lower[1]) * height\n"
             "                     if places[(row - lower[0]) + (col - lower[1]) * height] >= 0\n"
             "                     else -1)\n", 1)),
        ('a sparse star\'s cells written at their place among the stored',
         sub(ME_SECTION, "                out.append(((row - buffer.rows[0])\n"
                         "                            + (col - buffer.cols[0]) * buffer.pitch, place))\n",
             "                out.append((place, place))\n", 1)),
    ]),
    # The ring of mbarriers and the roles in PIR (`IRBuilder.mbarrier_ring`,
    # `roles`), the register budget, and the host oracle's ring model, as
    # tests/test_mbarrier_ring.py builds them without the multi-element
    # layout.
    'ring': ('tests/test_mbarrier_ring.py', [
        ('the empty side read as the full one',
         sub(ENGINES, "    offset = obj.slots if s.attr('side') == 'empty' else 0\n",
             '    offset = 0\n', 1)),
        ("a round's parity the trip's own",
         sub(BUILD, "        parity = self.op('rem', INDEX, self.op(\n"
                    "            'div', INDEX, trip, ring.slots, hint=f'{hint}p'), 2,\n"
                    "            hint=f'{hint}p')\n",
             "        parity = self.op('rem', INDEX, trip, 2, hint=f'{hint}p')\n", 1)),
        ("the empty barriers counting the full side's arrivals",
         sub(ENGINES, '            f\'{s.attr("empty")});\',\n',
             '            f\'{s.attr("full")});\',\n', 1)),
        ("a role's barrier outside the roles let through",
         sub(PASSES, '            if role_level is None:\n',
             '            if False:\n', 1)),
        ("the slot's bytes announced as a plain arrival",
         sub(ENGINES, "        if tx:\n            return Lowered((f'tensorforge::mbarrierExpect('\n",
             "        if False:\n            return Lowered((f'tensorforge::mbarrierExpect('\n", 1)),
        ('an arrival past the count let through',
         sub(ORACLE, "        if state['pending'] <= 0:\n",
             '        if False:\n', 1)),
        ("no barrier behind the ring's arming",
         sub(BUILD, "                                 leader=leader, extern=f'{name}_ring'))\n"
                    "        b.barrier(Participants.BLOCK)\n",
             "                                 leader=leader, extern=f'{name}_ring'))\n", 1)),
        ('a register budget where the target moves none',
         sub(ENGINES, "    if env.family != 'cuda' or target is None or not getattr(\n",
             "    if False and not getattr(\n", 1)),
    ]),
    # The asynchronous engines' B prepared at generation and streamed
    # through the operator ring (`Options.matrix_ring` `prepared`,
    # `multi_element._OperatorRing`): the producer's waits, the consumers'
    # releases, the bytes, the table's numbers and layout, the trip's head.
    'opring': ('tests/test_operator_ring.py::test_every_element_of_two_trips_on_the_oracle '
               'tests/test_operator_ring.py::test_every_element_with_own_barriers_on_the_oracle '
               'tests/test_operator_ring.py::test_a_slot_is_read_only_where_its_chunk_landed '
               'tests/test_operator_ring.py::test_the_ring_over_several_rounds_of_trips_on_the_oracle '
               'tests/test_operator_ring.py::test_the_table_holds_what_the_converting_ring_puts '
               'tests/test_operator_ring.py::test_the_table_is_zero_outside_the_windows '
               'tests/test_operator_ring.py::test_the_oracle_refuses_a_copy_past_a_table '
               'tests/test_operator_ring.py::test_a_resident_table_is_each_stage_in_its_canonical_layout '
               'tests/test_operator_ring.py::test_the_kernel_copies_b_and_converts_none_of_it '
               'tests/test_operator_ring.py::test_the_commit_onto_a_rings_slot_is_its_empty_side', [
        ('the producer copying into a slot it did not wait for',
         sub(ME_SECTION, "        b.ring_acquire(self.ring, slot, parity, writes=[self.storage])\n", '', 1)),
        ("tcgen05's products never giving their slot back",
         sub(ME_SECTION, "            b.ring_release(self.ring, slot, issue, engine='tcgen05')\n", '', 1)),
        # measync's mutations of the converting ring's issue, on the
        # prepared ring's own copies of its lines.
        ("the prepared ring's A lower half at the upper's columns",
         sub(ME_SECTION, '            upper_at = columns[step.products.index(piece.product)]\n'
                         '            lower_at = upper_at + stage.depth\n',
             '            upper_at = columns[step.products.index(piece.product)]\n'
             '            lower_at = upper_at\n', 1)),
        ('D written over by every tcgen05 of the prepared ring',
         sub(ME_SECTION, '                              enable_input_d=0 if first else 1,\n'
                         '                              a_column=(upper_at, lower_at)[ia]\n',
             '                              enable_input_d=0,\n'
             '                              a_column=(upper_at, lower_at)[ia]\n', 1)),
        ("the prepared ring's products committed by nobody",
         sub(ME_SECTION, '            continue\n'
                         "        b.commit_group('tcgen05', object=cursor, issue=issue)\n"
                         "        b.matrix_wait('tcgen05', object=cursor, reads=reads, tmem=[tmem])\n",
             '            continue\n'
             "        b.matrix_wait('tcgen05', object=cursor, reads=reads, tmem=[tmem])\n", 1)),
        ('D written over by every wgmma of the prepared ring',
         sub(ME_SECTION, '                plan = _plan(stage)\n'
                         '                for ia, ib in plan.products:\n'
                         '                    b.wgmma(n, halves[ia], bs[ib], accs,\n'
                         '                            scale_d=0 if first else 1, layouts=layouts,\n',
             '                plan = _plan(stage)\n'
             '                for ia, ib in plan.products:\n'
             '                    b.wgmma(n, halves[ia], bs[ib], accs,\n'
             '                            scale_d=0, layouts=layouts,\n', 1)),
        ("wgmma's A at the product's first rows of a prepared chunk",
         sub(ME_SECTION, '            source = trip.buffers[id(product.second.tensor)]\n'
                         '            k_lo = product.depth[0] + piece.k0\n',
             '            source = trip.buffers[id(product.second.tensor)]\n'
             '            k_lo = product.depth[0]\n', 1)),
        ("a chunk's bytes announced short",
         sub(ME_SECTION, '                      tx=self.volumes[j] * Datatype.F32.size(),\n',
             '                      tx=(self.volumes[j] - 4) * Datatype.F32.size(),\n', 1)),
        ("the table's halves swapped",
         sub(Path('src/tensorforge/common/matrix/forms.py'), '        laid[p * size + slots] = parts[p]\n',
             '        laid[p * size + slots] = parts[form.parts - 1 - p]\n', 1)),
        ("the table's core matrices along K packed",
         sub(Path('src/tensorforge/common/matrix/forms.py'), '    return (n // 8) * rows * 8 + (k // 4) * 32 + (n % 8) * 4 + k % 4\n',
             '    return (n // 8) * rows * 8 + (k // 4) * 4 + (n % 8) * rows + k % 4\n', 1)),
        ("the table's numbers past the window's contraction",
         sub(ME_GROUP, '            if col >= d1:\n                continue\n', '', 1)),
        ("wgmma's slot given back ahead of the wait for its products",
         sub(ME_SECTION, '            flight = _wgmma_retire(b, flight, ring.hold - 1, accs)\n'
                 '            ring.release(trip, i - ring.hold + 1, base)\n',
             '            ring.release(trip, i - ring.hold + 1, base)\n'
             '            flight = _wgmma_retire(b, flight, ring.hold - 1, accs)\n', 1)),
        ("wgmma's slot given back by its first lane alone",
         sub(ME_SECTION, '        b.barrier(Participants.WAVE)\n'
                 "        b.ring_release(self.ring, slot,\n",
             "        b.ring_release(self.ring, slot,\n", 1)),
        ("the trip's head filling every slot",
         sub(ME_SECTION, '            for j in range(min(self.lead, len(self.chunks))):\n',
             '            for j in range(min(self.lead + 1, len(self.chunks))):\n', 1)),
        ("the trip's first chunks never filled",
         sub(ME_SECTION, '        for ring in trip.rings.values():\n            ring.head(trip)\n', '', 1)),
        ("a product's A written over its products in flight",
         sub(ME_SECTION, "        turns = not apart and not last and any(\n"
                 "            k0 == 0 for _, k0, _ in ring.chunks[i + 1])\n",
             '        turns = False\n', 1)),
        ("the rounds of a trip counted from the block's first",
         sub(ME_SECTION, "        return _index(b, 'mul', _index(b, 'add', _index(\n"
                 "            b, 'mul', trips, self.passes, 'g'), pass_, 'g'), count, 'g')\n",
             "        return _index(b, 'mul', pass_, count, 'g')\n", 1)),
        ('a resident stage read before its copy landed',
         sub(ME_SECTION, '                    builder.wait(builder.copy_bulk(\n', '                    (builder.copy_bulk(\n', 1)),
        ('the oracle landing no product a ring commit bound',
         sub(ORACLE, "            self._tc_land(f'mbar:{key[0]}:{key[1]}')\n            return\n",
             '            return\n', 1)),
        ('the oracle taking a table copy past its end',
         sub(ORACLE, "    if isinstance(src, list) and not (0 <= at and at + count <= len(src)):\n",
             '    if False:\n', 1)),
    ]),
    # A for the asynchronous engines written once by the star step
    # (`Options.matrix_star`): the halves read unsplit, and the star step
    # folded into the engine's operand memory (`element_group.fuse`,
    # `multi_element._tcgen05_star`, `_wgmma_star`).
    'opstar': ('tests/test_folded_star.py::test_the_folded_star_frees_the_buffers_of_a '
               'tests/test_folded_star.py::test_the_folded_star_needs_b_resident_or_prepared '
               'tests/test_folded_star.py::test_b_keeps_its_halves_ahead_of_a '
               'tests/test_folded_star.py::test_the_folded_kernel_stores_a_where_the_engine_reads_it '
               'tests/test_folded_star.py::test_the_folded_star_over_two_trips_on_the_oracle '
               'tests/test_folded_star.py::test_the_folded_star_reads_only_what_was_written '
               'tests/test_folded_star.py::test_the_halves_over_two_trips_on_the_oracle '
               'tests/test_folded_star.py::test_the_halves_give_d_bit_for_bit '
               'tests/test_folded_star.py::test_several_folded_steps_build', [
        ("every folded step's A ring under one name",
         sub(ME_SECTION, "    areas = b.mbarrier_ring(f'ta{section.steps.index(step)}', slots,\n",
             "    areas = b.mbarrier_ring('ta', slots,\n", 1)),
        ("a band's A handed over short",
         sub(ME_SECTION, "    areas = b.mbarrier_ring(f'ta{section.steps.index(step)}', slots,\n"
                         "                            full=128, empty=1,\n",
             "    areas = b.mbarrier_ring(f'ta{section.steps.index(step)}', slots,\n"
             "                            full=96, empty=1,\n", 1)),
        ('a band writing a slot the products still read',
         sub(ME_SECTION, "            b.ring_acquire(self.areas, slot, parity)\n", '', 1)),
        # measync's mutations of `_tcgen05_a` and `_tcgen05`, on the fold's
        # own copies of their lines.
        ('a band of warps the group does not hold whole storing the folded A',
         sub(ME_SECTION, '        bands = max(1, trip.group.elements // 4)\n'
                         "        band = b.op('div', INDEX, trip.slot, 4, hint='band')\n",
             '        bands = trip.group.elements // 4 + 1\n'
             "        band = b.op('div', INDEX, trip.slot, 4, hint='band')\n", 1)),
        ('the tensor memory allocated again by a folded trip',
         sub(ME_SECTION, '        self.tmem = trip.tmem\n',
             '        self.tmem = b.tmem_alloc(256, arena=GeneralLexicon.TOTAL_SHR_MEM)\n', 1)),
        ("a block's lower halves over its upper ones",
         sub(ME_SECTION, '                b.tmem_st(tmem, at + 2 * STAR_ROWS * p + STAR_ROWS,\n',
             '                b.tmem_st(tmem, at + 2 * STAR_ROWS * p,\n', 1)),
        ("every block's products reading the first slot's A",
         sub(ME_SECTION, '                              a_column=at + 2 * STAR_ROWS * p\n',
             '                              a_column=columns[0] + 2 * STAR_ROWS * p\n', 1)),
        ("wgmma's slot of A written while its products are in flight",
         sub(ME_SECTION, '            flight = _wgmma_retire(b, flight, hold - 1, accs)\n'
                         '            if ring is not None:\n'
                         '                ring.release(trip, j - hold + 1, base)\n',
             '            flight = _wgmma_retire(b, flight, hold, accs)\n'
             '            if ring is not None:\n'
             '                ring.release(trip, j - hold + 1, base)\n', 1)),
        ("wgmma's stores of A not fenced for the async proxy",
         sub(ME_SECTION, '                                     \'as\'), mine, \'as\'), align=16)\n'
                         '        b.fence(FenceKind.PROXY)\n',
             '                                     \'as\'), mine, \'as\'), align=16)\n', 1)),
        ('the held intermediates counted in the room',
         sub(ME_GROUP, '        if self.held:\n            return 0\n', '', 1)),
        ('the folded star taking B through the converting ring',
         sub(ME_GROUP, '    if any(st.streamed and not st.prepared for st in stages):\n',
             '    if False:\n', 1)),
        ("tcgen05's A halves split again",
         sub(ME_SECTION, '            if lower is not None:\n'
                         '                # Both halves as the star step split them\n',
             '            if False:\n'
             '                # Both halves as the star step split them\n', 1)),
        ("wgmma's lower halves read where the upper ones are",
         sub(ME_SECTION, "            b, 'add', at, buffer.half_offset(trip.group.elements), 'a'),\n",
             "            b, 'add', at, 0, 'a'),\n", 1)),
        ("A's halves ahead of B's on the asynchronous engines",
         sub(ME_GROUP, '                     if run and asynchronous(run[0]) else [list(run)])\n',
             '                     if False else [list(run)])\n', 1)),
        ('the folded star over B read whole',
         sub(ME_GROUP, "    if not all(st.split for st in stages):\n        return ('B is read whole",
             "    if False:\n        return ('B is read whole", 1)),
        ('no unfolded tiling behind the fold',
         sub(ME_GROUP, '            tilings += [lower(section, context, split=True,\n'
                       '                              streamed=streamed(t), prepared=asked,\n'
                       '                              output=output)\n'
                       '                        for t in list(tilings) if t.star]\n', '', 1)),
        ("the star's numbers of the next column",
         sub(ME_SECTION, "        column = _index(b, 'add', b.op('rem', INDEX, row, self.width,\n",
             "        column = _index(b, 'add', b.op('rem', INDEX, _index(b, 'add', row, 1, 'p'), self.width,\n", 1)),
    ]),
    # A matrix step's D added into the element's output by bulk reductions
    # (`Options.matrix_output`, `element_group.output_refusal`,
    # `multi_element._bulk_output`).
    'opbulk': ('tests/test_bulk_output.py::test_every_matrix_lowering_takes_the_bulk_output '
               'tests/test_bulk_output.py::test_a_target_without_the_bulk_copy_keeps_the_reductions '
               'tests/test_bulk_output.py::test_the_refusals_say_why '
               'tests/test_bulk_output.py::test_the_bulk_output_costs_no_group_size '
               'tests/test_bulk_output.py::test_the_staged_columns '
               'tests/test_bulk_output.py::test_f64_columns_are_sixteen_bytes_too '
               'tests/test_bulk_output.py::test_the_kernel_adds_by_bulk_reductions '
               'tests/test_bulk_output.py::test_every_element_of_two_trips_on_the_oracle '
               'tests/test_bulk_output.py::test_the_reductions_own_wait_and_barrier_on_the_oracle '
               'tests/test_bulk_output.py::test_a_staged_column_written_again_before_the_wait_is_a_race '
               'tests/test_bulk_output.py::test_a_body_whose_one_copy_is_a_bulk_reduction_keeps_its_wait '
               'tests/test_bulk_output.py::test_the_window_is_bit_for_bit_and_the_rest_gets_plus_zero', [
        ('the staged columns not fenced for the async proxy',
         sub(ME_SECTION, '    b.fence(FenceKind.PROXY)\n'
                         '    b.barrier(trip.meet)\n'
                         '    pointer = _bound(b, trip.symbols[id(tensor)])\n',
             '    b.barrier(trip.meet)\n'
             '    pointer = _bound(b, trip.symbols[id(tensor)])\n', 1)),
        ('the reductions issued before the element warps meet',
         sub(ME_SECTION, '    b.fence(FenceKind.PROXY)\n'
                         '    b.barrier(trip.meet)\n'
                         '    pointer = _bound(b, trip.symbols[id(tensor)])\n',
             '    b.fence(FenceKind.PROXY)\n'
             '    pointer = _bound(b, trip.symbols[id(tensor)])\n', 1)),
        ('the staged columns written again before the reductions read them',
         sub(ME_SECTION, '                          dst_index=(dst,), src_index=(src,))\n'
                         '    b.wait(token)\n',
             '                          dst_index=(dst,), src_index=(src,))\n', 1)),
        # tma's mutations of the interpreter's and the emitter's reductions,
        # on the bulk reduction's own lines.
        ("the interpreter letting a bulk reduction's run go at the issue",
         sub(ORACLE, '            dst[di + k] = dst[di + k] + value\n        tracker = self.mem.tracker\n'
                     '        if tracker is not None:\n            tracker.reduce_issue(tracker.lane, slots)\n',
             '            dst[di + k] = dst[di + k] + value\n        tracker = self.mem.tracker\n'
             '        if tracker is not None:\n            pass\n', 1)),
        ('the wait for a bulk reduction dropped with the copies',
         sub(EMIT, "                and s.attr('engine') not in ('tdm',) + engines.REDUCE_ENGINES\n",
             "                and s.attr('engine') not in ('tdm', 'tma_reduce')\n", 1)),
        ('the rows outside the windows left as they were',
         sub(ME_SECTION, '        for r in list(range(0, begin)) + list(range(end, hi - lo)):\n',
             '        for r in []:\n', 1)),
        ("D's cells a row down in the staged columns",
         sub(ME_SECTION, "    at = _index(b, 'add', b.op('mul', INDEX, m, stride, hint='d'),\n"
                         "                _index(b, 'add', n, shift, 'd'), 'd')\n",
             "    at = _index(b, 'add', b.op('mul', INDEX, m, stride, hint='d'),\n"
             "                _index(b, 'add', n, shift + 1, 'd'), 'd')\n", 1)),
        ("tcgen05's rows of D from the span's first row",
         sub(ME_SECTION, '                    staged[shift + c0 + j] = value\n',
             '                    staged[c0 + j] = value\n', 1)),
        ("the box's last column left out",
         sub(ME_SECTION, '        elems, src = width * stride, ',
             '        elems, src = (width - 1) * stride, ', 1)),
        ('a masked element adding into the output',
         sub(ME_SECTION, '    if trip.mask is not None:\n'
                         "        issue = b.op('and', BOOL, issue, trip.mask, hint='issue')\n"
                         '    token = b.reduce_bulk(',
             '    token = b.reduce_bulk(', 1)),
        ("the staged columns an even number of sixteen bytes apart",
         sub(ME_GROUP, '    stride = _round(hi - lo, unit)\n    if (stride // unit) % 2 == 0:\n        stride += unit\n'
                       '    return lo, hi, stride\n',
             '    stride = _round(hi - lo, unit)\n    return lo, hi, stride\n', 1)),
        ("two steps' reductions into one output",
         sub(ME_GROUP, '    if writers > 1:\n', '    if writers > 2:\n', 1)),
        ('the bulk output where a step reads the output',
         sub(ME_GROUP, '    if any(sub.tensor is tensor for s in section.steps for p in s.products\n',
             '    if False and any(sub.tensor is tensor for s in section.steps for p in s.products\n', 1)),
        ('an output on eight bytes',
         sub(ME_GROUP, '    if align < 16:\n', '    if align < 8:\n', 1)),
        ('the staged columns left out of the room',
         sub(ME_GROUP, '    operators += output_volume(section, elements)\n', '', 1)),
        ('the bulk output at the cost of the group',
         sub(ME_GROUP, '        lowerings[:] = [twin for t in lowerings\n'
                       '                        for twin in _output_twins(t)]\n', '', 1)),
    ]),
    # F32's accuracy first in the multi-element plans: no asynchronous
    # engine reads an operand in TF32 alone unless the options let it
    # (`element_group.tf32_refusal`, `prefer_mma_sync`, `async_terms`).
    'opprec': ('tests/test_precision_first.py', [
        ('the check passing every plan',
         sub(ME_GROUP, '    full = set(nvidia.TF32_TERMS)\n    for step in section.steps:\n',
             '    return None\n    full = set(nvidia.TF32_TERMS)\n    for step in section.steps:\n', 1)),
        ("a whole B's two terms taken for 3xTF32's three",
         sub(ME_GROUP, '    return nvidia.TF32_PLAN if stage.split else TF32_HALVES_BY_WHOLE\n',
             '    return nvidia.TF32_PLAN\n', 1)),
        ('the option ignored',
         sub(ME_GROUP, "    return bool(getattr(context.get_user_options(), 'matrix_tf32_operands',\n",
             "    return False and bool(getattr(context.get_user_options(), 'matrix_tf32_operands',\n", 1)),
        ('the asynchronous engine whatever group mma_sync holds',
         sub(ME_GROUP, '    return min(mma_elements, cap) > asynchronous_elements\n',
             '    return False\n', 1)),
        ("mma_sync's group not held to the engine's lane limit",
         sub(ME_GROUP, '    return min(mma_elements, cap) > asynchronous_elements\n',
             '    return mma_elements > asynchronous_elements\n', 1)),
        ("the generator's check of the plan dropped",
         sub(GEN, '    if refused is not None:\n      raise InternalError(',
             '    if False:\n      raise InternalError(', 1)),
        ("the kernel's comment silent on the engine declined",
         sub(ME_GROUP, '    notes = [section.engine_note] if section.engine_note else []\n',
             '    notes = []\n', 1)),
    ]),
    # The term plan (`pir.terms`): an emulated product's products, their
    # order, partials and epilogue stated once, the paths issuing a plan's
    # products and `check_plans` reading them back off the kernels; the
    # derived forms the tables are made in (`common.matrix.forms`).
    'terms': ('tests/test_term_plan.py tests/test_derived_forms.py', [
        ('a plan short of its last product',
         sub(Path('src/tensorforge/backend/pir/terms.py'),
             '    return TermPlan(name, a, b, pairs, (0,) * len(pairs), Epilogue.NONE,\n',
             '    pairs = pairs[:-1]\n'
             '    return TermPlan(name, a, b, pairs, (0,) * len(pairs), Epilogue.NONE,\n', 1)),
        ('the epilogue without the leading partial',
         sub(Path('src/tensorforge/backend/pir/terms.py'),
             '        return add(partials[0], correction)\n',
             '        return correction\n', 1)),
        ('check_plans silent on a run out of its order',
         sub(Path('src/tensorforge/backend/pir/terms.py'),
             '            if (len(terms) % len(want) or tuple(terms) != want * whole):\n',
             '            if False:\n', 1)),
        ('the order split.products is asked for ignored',
         sub(Path('src/tensorforge/backend/instructions/compute/split.py'),
             "    sign = -1 if order == 'smallest' else 1\n",
             '    sign = -1\n', 1)),
        ('the stacked tile issuing its products backwards',
         sub(PKG / 'stacked.py',
             '                    for i, j in plan.products:\n',
             '                    for i, j in reversed(plan.products):\n', 1)),
        ("Intel's halves named the other way round",
         sub(Path('src/tensorforge/backend/instructions/compute/primitives/intel.py'),
             '                for q, p in TF32_PLAN.products:\n',
             '                for p, q in TF32_PLAN.products:\n', 1)),
        ("NVIDIA's products into the leading partial whatever the plan",
         sub(Path('src/tensorforge/backend/instructions/compute/primitives/nvidia.py'),
             '                    partial = plan.partial(term) if plan is not None else 0\n',
             '                    partial = 0\n', 1)),
        ("the device pair's upper half rounded away on sm_90",
         sub(Path('src/tensorforge/common/matrix/forms.py'),
             '        make=lambda x: np.stack(tf32_halves(x, ties)))\n',
             "        make=lambda x: np.stack(tf32_halves(x, 'away')))\n", 1)),
        ('three parts taken for the TF32 pair',
         sub(Path('src/tensorforge/common/matrix/forms.py'),
             '            BF16_TRIPLE.parts: BF16_TRIPLE}.get(int(parts))\n',
             '            BF16_TRIPLE.parts: TF32_STORED}.get(int(parts))\n', 1)),
    ]),
    # The stacked emulated table (`stacked.EMULATED_TILES`): the 32-wide
    # tiles' own M order of D, gfx950's two quads a lane, their `__bf16`
    # vectors, the asked tile, the padded ninth column.
    'bf16tiles': ('tests/test_amd_stacked.py -k tile', [
        ("the 32-wide tile's lanes read in the F32 tile's M order",
         sub(PKG / 'stacked.py',
             '    if stacking(op) == TABLE:\n        if H != 1:\n',
             '    if False:\n        if H != 1:\n', 1)),
        ("a gfx950 lane's second quad of steps at its first's",
         sub(PKG / 'stacked.py',
             '    bases = [QUAD * count * (i // QUAD) for i in range(vector)]\n',
             '    bases = [0 for i in range(vector)]\n', 1)),
        ("the host-split operator's second quad read where the first is",
         sub(PKG / 'stacked.py',
             '        first = r0 // QUAD + count * g\n',
             '        first = r0 // QUAD\n', 1)),
        ('eight steps split as four',
         sub(PKG / 'stacked.py',
             '    splitter = SPLIT_BF16 if vector == QUAD else SPLIT_BF16_X8\n',
             '    splitter = SPLIT_BF16\n', 1)),
        ("gfx950's vectors in the generic name, which HIP does not know",
         sub(HIP_LEXIC, "HIP_TYPE_NAMES = {'bfloat16': '__bf16'}\n",
             'HIP_TYPE_NAMES = {}\n', 1)),
        ('the asked tile at any lanes',
         sub(PKG / 'stacked.py',
             '        return op.builtin if op is not None and op.n == threads else None\n',
             '        return op.builtin if op is not None else None\n', 1)),
        ("the 32-wide tile's ninth column never padded",
         sub(PKG / 'stacked.py',
             '        if (padded(ctx) and (quarters(op, shape.threads) > 1 or width > 4)\n',
             '        if (padded(ctx) and quarters(op, shape.threads) > 1\n', 1)),
        ("gfx950's operands as short vectors",
         sub(PKG / 'catalog.py',
             "BF16_REGISTER_FEATURES = frozenset({'gfx950-insts', 'wmma-n16-insts'})\n",
             "BF16_REGISTER_FEATURES = frozenset({'wmma-n16-insts'})\n", 1)),
    ]),
    # The host oracle's arms of an `if` as coroutines that hand over at a
    # pending ring wait (`kernel_eval.Lockstep._arms`, `COROUTINE_ARMS`).
    'oraclearms': ('tests/test_oracle_roles.py tests/test_mbarrier_ring.py', [
        ('a pending ring wait not seen',
         sub(ORACLE, "        return state['phases'] % 2 == int(self.ev(m.group('parity'))) & 1\n",
             '        return False\n', 1)),
        ("a pending wait's parity the other way round",
         sub(ORACLE, "        return state['phases'] % 2 == int(self.ev(m.group('parity'))) & 1\n",
             "        return state['phases'] % 2 != int(self.ev(m.group('parity'))) & 1\n", 1)),
        ('the arms of an if whole one after the other',
         sub(ORACLE, '                if COROUTINE_ARMS and len(arms) == 2 and taken and other:\n',
             '                if False:\n', 1)),
        ('arms left waiting where none went on',
         sub(ORACLE, '            if running and self.steps == before:\n'
                     '                yield waits[id(running[0])]\n',
             '            if running and self.steps == before:\n'
             '                break\n', 1)),
    ]),
    # The lanes form's slot form (`multi_element_fma` `slots`,
    # `element_group.SlotMapping`, `plan_exchange`, `slots_plan`,
    # `multi_element._SlotTrip`, the loop's lane map): its plan, its
    # exchange, the lane's coefficients, the column's write-back, the
    # binding of a lane's element.
    'slots': ('tests/test_slot_mapping.py tests/test_batch_parts.py -n 8', [
        ('the exchange taking its needs unrotated',
         sub(ME_GROUP, '                       key=lambda x: ((x - quantity) % quantities, x))\n',
             '                       key=lambda x: x)\n', 1)),
        ("a shuffle reading the first element's lane",
         sub(ME_GROUP, '            sources[k][lane] = mapping.place(element, x)[0]\n',
             '            sources[k][lane] = mapping.place(0, x)[0]\n', 1)),
        ('a shuffle fewer than a lane needs',
         sub(ME_GROUP, '    count = max((len(n) for n in needs.values()), default=0)\n',
             '    count = max((len(n) for n in needs.values()), default=1) - 1\n', 1)),
        ("a sparse operand's cell taken as stored",
         sub(ME_GROUP, '    return tensor.is_dense() or bool(tensor.spp.is_nz((row, col)))\n',
             '    return True\n', 1)),
        ('a lane needing its own column from another',
         sub(ME_GROUP, '                    if k != col and _possible(p.second.tensor, k, col):\n',
             '                    if _possible(p.second.tensor, k, col):\n', 1)),
        ("a product's own term left out",
         sub(ME_GROUP, '        terms.append(-1)\n', '        pass\n', 1)),
        ('every shuffle taken to bring every lane something',
         sub(ME_GROUP, '        return all(self.received[shuffle][lane] >= 0\n',
             '        return True or all(self.received[shuffle][lane] >= 0\n', 1)),
        ('the slot form taken on HIP',
         sub(ME_GROUP, "SLOT_TARGETS = (('cuda', 'nvidia'),)\n",
             "SLOT_TARGETS = (('cuda', 'nvidia'), ('hip', 'amd'))\n", 1)),
        ('a chained section planned whole',
         sub(ME_GROUP, '    if section.chained:\n        return _chained_slots(section, context)\n',
             '    if False:\n        return _chained_slots(section, context)\n', 1)),
        ("a phase's copies left out",
         sub(ME_SECTION, '        trip.fused = tuple(phase.scales)\n        _slots(trip)\n',
             '        trip.fused = ()\n        _slots(trip)\n', 1)),
        ("a held output's copy reading the phase's sums alone",
         sub(ME_SECTION, '    trip.acc = values\n', '    pass\n', 1)),
        ('copies alone computed as a column',
         sub(ME_SECTION, '    if trip.computes:\n        _slot_column(trip)\n',
             '    if True:\n        _slot_column(trip)\n', 1)),
        ('an intermediate mixed in slots',
         sub(ME_GROUP, '                if mixed.is_tmp:\n', '                if False:\n', 1)),
        ("the table read by the slots",
         sub(ME_GROUP, "    if lane_values(context) == 'table':\n", '    if False:\n', 1)),
        ("the slot form's warps one a block",
         sub(ME_GROUP, "            # The slot form's: a warp is a group of its own elements.\n"
                       "            return self.units\n",
             '            return 1\n', 1)),
        ("the slot form's warp as wide as its group",
         sub(ME_GROUP, '            return self.lanes.mapping.width\n',
             '            return self.elements\n', 1)),
        ("a lane's coefficients all at its own quantity",
         sub(ME_SECTION, "                row = _index(b, 'add', got, self.origin, 'g')\n",
             '                row = self.column\n', 1)),
        ('a shuffle reading the lane itself',
         sub(ME_SECTION, "                    self.source(term), type_=self.fp, hint='r')\n",
             "                    self.lane, type_=self.fp, hint='r')\n", 1)),
        ('the zero where a lane receives nothing the other way round',
         sub(ME_SECTION, "                    valid = b.op('ne', BOOL, source, self.lane, hint='v')\n",
             "                    valid = b.op('eq', BOOL, source, self.lane, hint='v')\n", 1)),
        ("a row's accumulations inlined into the write",
         sub(ME_SECTION, '        self.fma_attrs = tuple(dict(self.fma_attrs, no_inline=True).items())\n',
             '        self.fma_attrs = self.fma_attrs\n', 1)),
        ('the sums started at the output read ahead of the walk',
         sub(ME_SECTION, '    trip.acc = {(i, 0): None for i in write}\n',
             '    trip.acc = {(i, 0): trip.own(tensor, i, trip.col) for i in write}\n', 1)),
        ('a tile the column writes in part written row by row',
         sub(ME_SECTION, '            kept |= {r for r in range(first, first + trip.width)\n'
                         '                     if r not in write}\n',
             '            kept |= set()\n', 1)),
        ('the idle lanes holding an element',
         sub(BATCHLOOP, "            cond = f'{lexic.thread_idx_x} < {elements * lanes} && {cond}'\n",
             '            cond = cond\n', 1)),
        ("every warp the block's first elements",
         sub(BATCHLOOP, "            writer(f'const auto {self._lane()} = {elements} * '\n"
                        "                   f'{lexic.thread_idx_y} + {lexic.thread_idx_x} / {lanes};')\n",
             "            writer(f'const auto {self._lane()} = '\n"
             "                   f'{lexic.thread_idx_x} / {lanes};')\n", 1)),
        ('the loop without its lane map',
         sub(GEN, '                         lane_map=self._lane_map())\n',
             '                         lane_map=None)\n', 1)),
        ('a description the slot form takes reassociated',
         sub(GEN, "      if element_group.fma_form(context) not in ('lanes', 'slots'):\n",
             "      if element_group.fma_form(context) != 'lanes':\n", 1)),
        ("the slot form's register caps not tried",
         sub(TUNING, '    if _slot_form(descrs, context):\n', '    if False:\n', 1)),
        ("the slot form's caps short of the one that binds",
         sub(TUNING, 'SLOT_MIN_BLOCKS = (1, 2, 3)\n', 'SLOT_MIN_BLOCKS = (1, 2)\n', 1)),
        # The dense exchange as broadcasts (`element_group.BROADCASTS`): the
        # planner's rule, the own term, the computed source, the coefficient's
        # row.
        ('the dense exchange planned by needs',
         sub(ME_GROUP, '    if BROADCASTS and quantities > 1 and all(\n',
             '    if False and quantities > 1 and all(\n', 1)),
        ('every exchange planned as broadcasts',
         sub(ME_GROUP, '            len(needs.get(lane, ())) == quantities - 1\n',
             '            True\n', 1)),
        ('a broadcast keeping its own term',
         sub(ME_GROUP, '    if not exchange.broadcast and any(\n', '    if any(\n', 1)),
        ('a broadcast reading the lane after',
         sub(ME_SECTION, "            self.sources[shuffle] = _index(self.builder, 'add', self.first,\n"
                         "                                           shuffle, 'src')\n",
             "            self.sources[shuffle] = _index(self.builder, 'add', self.first,\n"
             "                                           shuffle + 1, 'src')\n", 1)),
        ("a broadcast's coefficient at the lane's own quantity",
         sub(ME_SECTION, '                row = term + self.origin\n', '                row = self.column\n', 1)),
        ('a broadcast read from a table of source lanes',
         sub(ME_SECTION, '        if shuffle not in self.sources and self.exchange.broadcast:\n',
             '        if False:\n', 1)),
        # Where the output is read (`Options.slot_output_read`) and the grid
        # that covers the batch in one round (`Options.one_round_grid`).
        ('the output read behind the walk with nothing around it',
         subs(ME_SECTION, ('    old = {} if masked else read()\n', '    old = {}\n'),
              ('    values = added(old)\n', '    values = added(read())\n'))),
        ('the masked block never taken',
         sub(ME_SECTION, "    masked = (slot_read(trip.section, trip.context) == 'masked'\n",
             "    masked = (False and slot_read(trip.section, trip.context) == 'masked'\n", 1)),
        ("the masked block's stores guarded again",
         sub(ME_SECTION, '            _write(trip, tensor, rows, added(read()), masked=False)\n',
             '            _write(trip, tensor, rows, added(read()))\n', 1)),
        ("a phase whose copies read its column masked",
         sub(ME_SECTION, '              and trip.mask is not None and not trip.fused)\n',
             '              and trip.mask is not None)\n', 1)),
        ('the masked block reading the output twice',
         sub(ME_SECTION, '            _write(trip, tensor, rows, added(read()), masked=False)\n',
             '            _write(trip, tensor, rows, added({i: b.op(\'add\', fp, v, v) for i, v in read().items()}), masked=False)\n', 1)),
        ('one round asked and the occupancy grid taken',
         sub(GEN, '                    and not self._grid_one_round()))\n', '                    ))\n', 1)),
        ('one round asked together with a resident grid',
         sub(GEN, '    if options.one_round_grid is True and options.resident_grid > 0:\n',
             '    if False:\n', 1)),
        ('one round under `auto` refusing a resident grid',
         sub(GEN, '    if options.one_round_grid is True and options.resident_grid > 0:\n',
             '    if options.one_round_grid and options.resident_grid > 0:\n', 1)),
        ("one round's launch said to loop",
         sub(GEN, '            self._grid_one_round()\n            and self._context.get_user_options().resident_grid <= 0\n',
             '            False\n            and self._context.get_user_options().resident_grid <= 0\n', 1)),
        # `multi_element=auto` on NVIDIA (`element_group.AUTO_SLOTS`,
        # `SLOT_DECLINES`), the slot form's defaults (`slot_read`,
        # `slot_one_round`) and its knobs (`tuning.SLOT_OUTPUT_READS`).
        ('`auto` not taking the slot form on NVIDIA',
         sub(ME_GROUP, '    if (context.target.backend, vendor) in SLOT_TARGETS:\n'
                       '        if section.dtype not in AUTO_SLOTS.get(vendor, ()):\n',
             '    if False:\n'
             '        if section.dtype not in AUTO_SLOTS.get(vendor, ()):\n', 1)),
        ('`auto` taking no F64 on NVIDIA',
         sub(ME_GROUP, "    'nvidia': (Datatype.F32, Datatype.F64)}\n", "    'nvidia': (Datatype.F32,)}\n", 1)),
        ('`auto` taking the slot form where another form is asked for',
         sub(ME_GROUP, "        if asked not in ('auto', 'slots'):\n", '        if False:\n', 1)),
        ('a measured loss not declined',
         sub(ME_GROUP, '        if decline is not None:\n            return None, decline.why\n',
             '        if False:\n            return None, decline.why\n', 1)),
        ("one target's decline on every target",
         sub(ME_GROUP, '                and (entry.archs is None\n', '                and (True\n', 1)),
        ('a decline at every element type',
         sub(ME_GROUP, '        if (entry.signature == signature and section.dtype in entry.dtypes\n',
             '        if (entry.signature == signature\n', 1)),
        ("the stars' storage left out of the signature",
         sub(ME_GROUP, '            kinds.count(Kind.OPERATOR), Cones(section).rows[1], bool(sparse))\n',
             '            kinds.count(Kind.OPERATOR), Cones(section).rows[1], False)\n', 1)),
        ('a held F32 section read masked',
         sub(ME_GROUP, "        return 'masked' if section.dtype is Datatype.F64 else 'ahead'\n",
             "        return 'masked'\n", 1)),
        ("GH200's short outputs read masked",
         sub(ME_GROUP, "    if _gh200_rules(context) and not long:\n        return 'ahead'\n",
             "    if False:\n        return 'ahead'\n", 1)),
        ('every target reading short outputs ahead',
         sub(ME_GROUP, "    if _gh200_rules(context) and not long:\n        return 'ahead'\n",
             "    if not long:\n        return 'ahead'\n", 1)),
        ('GH200\'s rules on every target',
         sub(ME_GROUP, '    return any(arch.startswith(a) for a in SLOT_GH200_RULES)\n', '    return True\n', 1)),
        ('a section holding columns not seen',
         sub(ME_GROUP, '    held = any((cones.source(p) is not None\n', '    held = False and any((cones.source(p) is not None\n', 1)),
        ('the placement asked for not taken',
         sub(ME_GROUP, "    asked = context.get_user_options().slot_output_read\n    if asked != 'auto':\n        return asked\n",
             "    asked = context.get_user_options().slot_output_read\n    if False:\n        return asked\n", 1)),
        ("the slot form's grid by occupancy unless asked",
         sub(ME_GROUP, '    if _gh200_rules(context):\n        return True\n    held, long = _slot_shape(group.section)\n'
                       '    return held or long\n',
             '    return False\n', 1)),
        ("GB200's grid in one round everywhere",
         sub(ME_GROUP, '    return held or long\n', '    return True\n', 1)),
        ("GH200's grid by GB200's rule",
         sub(ME_GROUP, '    if _gh200_rules(context):\n        return True\n', '    if False:\n        return True\n', 1)),
        ('a kernel without the slot form taken for one round',
         sub(ME_GROUP, '    if group is None or group.lanes is None or group.lanes.mapping is None:\n        return False\n',
             '    if group is None:\n        return False\n', 1)),
        ("the slot form's placements not tried",
         sub(TUNING, "        knobs.append(Knob('slot_output_read', lambda c: SLOT_OUTPUT_READS))\n", '', 1)),
        ("the slot form's grids not tried",
         sub(TUNING, "        knobs.append(Knob('one_round_grid', lambda c: (False, True)))\n", '', 1)),
    ]),
    # --- warp roles (tensorforge-roles) ------------------------------------- #
    # A block in N roles (`pir.roles`, `IRBuilder.split_roles`): the chain of
    # links, each role's barrier id, the verifier's and the placement's
    # reading of the chain, and the roles' register budgets.
    'roles': ('tests/test_roles.py '
              'tests/test_element_producer.py::test_the_producers_registers_go_to_the_element_warps '
              'tests/test_element_producer.py::test_the_moves_stay_in_the_blocks_registers '
              'tests/test_element_producer.py::test_the_kernel_moves_the_registers_once '
              'tests/test_mbarrier_ring.py -n 6', [
        ("a role's id stepping past its range's first warp",
         sub(Path('src/tensorforge/backend/pir/roles.py'),
             "        steps = ''.join(f' + ({self.warp} >= {lo})' for lo in self.starts)\n",
             "        steps = ''.join(f' + ({self.warp} > {lo})' for lo in self.starts)\n", 1)),
        ("a middle role's condition its range's start alone",
         sub(Path('src/tensorforge/backend/pir/roles.py'),
             "        return b.op('and', BOOL, b.op('ge', BOOL, warp, lo, hint=role.name),\n",
             "        return b.op('ge', BOOL, warp, lo, hint=role.name)\n"
             "        return b.op('and', BOOL, b.op('ge', BOOL, warp, lo, hint=role.name),\n", 1)),
        ("every link of the chain carrying the whole split",
         sub(Path('src/tensorforge/backend/pir/roles.py'),
             "                    movable=False, attrs=(('roles', tuple(roles[k:])),))\n",
             "                    movable=False, attrs=(('roles', tuple(roles)),))\n", 1)),
        ("a chain a statement broke let through",
         sub(Path('src/tensorforge/backend/pir/roles.py'),
             '    if (len(rest) != 1 or not is_split(rest[0])\n',
             '    if False and (len(rest) != 1 or not is_split(rest[0])\n', 1)),
        ("a region built with the next role current",
         sub(Path('src/tensorforge/backend/pir/roles.py'),
             '        b._roles.append(split.roles[i])\n',
             '        b._roles.append(split.roles[(i + 1) % len(split.roles)])\n', 1)),
        ("the moves past the block's pool let through",
         sub(Path('src/tensorforge/backend/pir/roles.py'),
             '    if held > entry * split.warps * wave:\n',
             '    if False:\n', 1)),
        ("a moving role of no whole warpgroups let through",
         sub(Path('src/tensorforge/backend/pir/roles.py'),
             '        if count != entry and (lo % WARPGROUP_WARPS\n',
             '        if False and (lo % WARPGROUP_WARPS\n', 1)),
        ("the takers' share of the SM's registers, not the block's",
         sub(Path('src/tensorforge/backend/pir/roles.py'),
             '    gained = (entry * total - given) // sum(t for _, t in takers) // 8 * 8\n',
             '    gained = (registers - given) // sum(t for _, t in takers) // 8 * 8\n', 1)),
        ("the registers raised before the block met",
         sub(Path('src/tensorforge/backend/pir/roles.py'),
             '    b.barrier(Participants.BLOCK)\n    for role in split.roles:\n',
             '    for role in split.roles:\n', 1)),
        ("the conditions of every role but the first",
         sub(BUILD, '                 for role in split.roles[:-1]]\n',
             '                 for role in split.roles[1:]]\n', 1)),
        ("another role's barrier in a role let through",
         sub(PASSES, '            elif (obj.ident, obj.threads) != (role.ident, role.threads):\n',
             '            elif False:\n', 1)),
        ("the verifier reading no chain's shape",
         sub(PASSES, '                why = role_split.chain_refusal(s)\n',
             '                why = None\n', 1)),
        ("a later link's roles reached at its condition's level",
         sub(PASSES, '                inner_role = reachable_at if role_level is None \\\n'
                     '                    else role_level\n',
             '                inner_role = reachable_at\n', 1)),
        ("every role's hazards met on the first role's barrier",
         sub(BARRIERS, '            def make(handoff, block, role=role):\n',
             '            def make(handoff, block, role=s.attr(\'roles\')[0]):\n', 1)),
        ("the accesses across the roles of the first role's writes alone",
         sub(BARRIERS, '        for i, (_, writes_a) in enumerate(roles):\n',
             '        for i, (_, writes_a) in enumerate(roles[:1]):\n', 1)),
        ("the state past a split of its first two roles' ends",
         sub(BARRIERS, '        for end in ends[1:]:\n',
             '        for end in ends[1:2]:\n', 1)),
    ]),
    # The ring as one object (`engines.RingSpec`, `IRBuilder.ring_acquire`
    # and its kin, `pir.rings`): its description, the scope every statement
    # on it states, its acts held to its roles, and the hand-offs a pass
    # reads.
    'ringobj': ('tests/test_ring_object.py tests/test_mbarrier_ring.py -n 6', [
        ("a ring's description not carried by its object",
         sub(ENGINES, "    return SyncObjectType(scope, kind, Participants.BLOCK, slots=int(slots),\n"
                      "                          ring=spec)\n",
             "    return SyncObjectType(scope, kind, Participants.BLOCK, slots=int(slots))\n", 1)),
        ("a ring's waits stating no scope",
         sub(ENGINES, "    return _stmt(Op.WAIT_PHASE, (('object', ring), ('side', side),\n"
                      "                                 ('scope', _ring_scope(ring))),\n",
             "    return _stmt(Op.WAIT_PHASE, (('object', ring), ('side', side)),\n", 1)),
        ("a cluster's scope spelled as the block's",
         sub(ENGINES, '    if scope is not None and scope is not SyncScope.BLOCK:\n',
             '    if False:\n', 1)),
        ("a role's acts read the other way round",
         sub(ENGINES, "        producer = (side == 'empty') == waits\n",
             "        producer = (side == 'full') == waits\n", 1)),
        ('a producer among its own consumers let through',
         sub(ENGINES, '        if self.producer is not None and self.producer in consumers:\n',
             '        if False:\n', 1)),
        ("a commit onto a slot read as a publish",
         sub(Path('src/tensorforge/backend/pir/rings.py'), "        return 'release', 'empty'\n",
             "        return 'publish', 'full'\n", 1)),
        ("a split's later roles' hand-offs given to its first",
         sub(Path('src/tensorforge/backend/pir/rings.py'),
             '                    for r, region in regions:\n'
             '                        visit(region.body, r.name)\n',
             '                    for r, region in regions:\n'
             '                        visit(region.body, regions[0][0].name)\n', 1)),
        ("a publish's bytes read as none",
         sub(Path('src/tensorforge/backend/pir/rings.py'), "        return int(s.attr('tx') or 0)\n",
             '        return 0\n', 1)),
        ("a publish of other bytes than its round's let through",
         sub(Path('src/tensorforge/backend/pir/rings.py'),
             "    if act == 'publish' and spec.tx and int(s.attr('tx') or 0) != spec.tx:\n",
             '    if False:\n', 1)),
        ("the verifier asking no ring's roles",
         sub(PASSES, '        why = ring_acts.refusal(s, None if role is None else role.name)\n',
             '        why = None\n', 1)),
        ('a ring set up without the roles it names',
         sub(BUILD, '                                producer, tuple(consumers), int(tx),\n',
             '                                None, (), int(tx),\n', 1)),
        ("an engine's release written as an arrival",
         sub(BUILD, '        if engine is not None:\n'
                    '            return self.commit_group(engine, object=ring, issue=issue,\n',
             '        if False:\n'
             '            return self.commit_group(engine, object=ring, issue=issue,\n', 1)),
        ("the producer's ring naming no roles",
         sub(ME_SECTION, "            arena=GeneralLexicon.TOTAL_SHR_MEM, producer='producer',\n"
                         "            consumers=('elements',))\n",
             '            arena=GeneralLexicon.TOTAL_SHR_MEM)\n', 1)),
        # The release point (`RingSpec.hold`, `Options.matrix_ring_hold`).
        ("a ring held past its slots",
         sub(ENGINES, '        if not 1 <= int(self.hold) <= int(self.slots):\n',
             '        if not 1 <= int(self.hold):\n', 1)),
        ("a ring's hold not carried to its description",
         sub(BUILD, '                                kind, scope, int(hold), tuple(waiters),\n',
             '                                kind, scope, 1, tuple(waiters),\n', 1)),
        ("wgmma's slot given back a chunk past the one its wait retired",
         sub(ME_SECTION, '            ring.release(trip, i - ring.hold + 1, base)\n',
             '            ring.release(trip, i - ring.hold + 2, base)\n', 1)),
        ("the wait behind a wgmma chunk leaving its hold in flight",
         sub(ME_SECTION, '            flight = _wgmma_retire(b, flight, ring.hold - 1, accs)\n',
             '            flight = _wgmma_retire(b, flight, ring.hold, accs)\n', 1)),
        ("the slots still held behind the last chunk never given back",
         sub(ME_SECTION, '    for i in range(max(0, len(ring.chunks) - ring.hold + 1),\n',
             '    for i in range(max(0, len(ring.chunks) - 1),\n', 1)),
        ("the first thread filling past the slot given back",
         sub(ME_SECTION, "        return self.slots - self.hold + (self.step.lowering != 'tcgen05')\n",
             "        return self.slots - self.hold + 1 + (self.step.lowering != 'tcgen05')\n", 1)),
        ("the first thread's lead blind to the hold",
         sub(ME_SECTION, "        return self.slots - self.hold + (self.step.lowering != 'tcgen05')\n",
             "        return self.slots - 2 + (self.step.lowering != 'tcgen05')\n", 1)),
        ("wgmma's folded star waiting as before whatever the hold",
         sub(ME_SECTION, '    hold = slots if ring is None else ring.hold\n',
             '    hold = slots\n', 1)),
        ("a hold its rings refuse taken by the plan",
         sub(ME_GROUP, '    taken = int(hold) if hold_refusal(section, int(hold)) is None else 0\n',
             '    taken = int(hold)\n', 1)),
        ("wgmma's folded star holding B past A's slots",
         sub(ME_GROUP, '        if folded(section, step) and hold > STAR_SLOTS[step.lowering]:\n',
             '        if False:\n', 1)),
    ]),
    # The roles and rings on AMD and Intel (`pir.roles.roles_refusal`,
    # `engines.ring_kind`): gfx1250's role objects and the rings of LDS
    # counters on the host oracle, CDNA's and SPMD's refusals, PVC's named
    # barriers in the text.
    'roleslower': ('tests/test_roles_lowerings.py -n 6', [
        ("every role of gfx1250 meeting on the first object",
         sub(ENGINES, "            return (f'tensorforge::syncNamedBarrier(&{ROLE_OBJECTS}'\n"
                      "                    f'[{obj.ident}]);')\n",
             "            return (f'tensorforge::syncNamedBarrier(&{ROLE_OBJECTS}'\n"
             "                    f'[0]);')\n", 1)),
        ("an LDS ring's empty side counting the full side's arrivals",
         sub(ENGINES, "        f'    tensorforge::ldsRingInit(&{ring}[{obj.slots} + r], '\n"
                      "        f'{s.attr(\"empty\")});',\n",
             "        f'    tensorforge::ldsRingInit(&{ring}[{obj.slots} + r], '\n"
             "        f'{s.attr(\"full\")});',\n", 1)),
        ("an LDS ring's wait for the first phase's parity always",
         sub(ENGINES, "        return Lowered((f'tensorforge::ldsRingWait({_ring_barrier(s, env)}, '\n"
                      "                        f'{env.name(s.args[1])});',))\n",
             "        return Lowered((f'tensorforge::ldsRingWait({_ring_barrier(s, env)}, '\n"
             "                        f'0);',))\n", 1)),
        ("an LDS ring's arrival with bytes let through",
         sub(ENGINES, "        if s.attr('tx'):\n"
                      "            raise IRError(f'{s.op} of {s.attr(\"tx\")} bytes on a ring of LDS '\n",
             "        if False:\n"
             "            raise IRError(f'{s.op} of {s.attr(\"tx\")} bytes on a ring of LDS '\n", 1)),
        ("the roles' objects counted in the split's order, not their ids'",
         sub(Path('src/tensorforge/backend/pir/roles.py'),
             '    ordered = sorted(split.roles, key=lambda r: r.warps[0])\n',
             '    ordered = list(split.roles)\n', 1)),
        ("a second split's objects let through on gfx1250",
         sub(BUILD, '        elif self._role_objects != split.ident:\n',
             '        elif False:\n', 1)),
        ('CDNA refusing roles for want of objects alone',
         sub(Path('src/tensorforge/backend/pir/roles.py'),
             "        if str(hw.model).startswith('gfx9'):\n"
             "            return CDNA_REFUSAL\n", '', 1)),
        ("a PVC ring's empty sides on its full sides' ids",
         sub(ENGINES, "    return obj.ring.ident + (obj.slots if side == 'empty' else 0)\n",
             '    return obj.ring.ident\n', 1)),
        ("the consumers' first release on the full sides",
         sub(ENGINES, "    return _stmt(Op.ARRIVE, (('object', ring), ('side', 'empty'),\n",
             "    return _stmt(Op.ARRIVE, (('object', ring), ('side', 'full'),\n", 1)),
        ("PVC's roles named by the split's id of each thread, a value",
         sub(BUILD, '                split, first=self._named_take(len(split.roles)),\n'
                    '                constant=True)\n',
             '                split, first=self._named_take(len(split.roles)))\n', 1)),
        ("every role's constant id the first role's",
         sub(Path('src/tensorforge/backend/pir/roles.py'),
             '            r._replace(ident=str(self.id_of(r.name)) if self.constant\n',
             '            r._replace(ident=str(self.first) if self.constant\n', 1)),
        ("PVC's named barriers let through whatever the IGC",
         sub(ENGINES, 'PVC_NAMED_BARRIERS: Optional[bool] = None\n',
             'PVC_NAMED_BARRIERS: Optional[bool] = True\n', 1)),
        ("a PVC role split let through where its barriers hang",
         sub(Path('src/tensorforge/backend/pir/roles.py'),
             '        why = engines.pvc_named_refusal()\n        if why is not None:\n            return why\n',
             '', 1)),
        ("no IGC to ask taken for one that writes the gateway message",
         sub(ENGINES, '    return PVC_UNPROBED if form is None else PVC_REFUSAL\n',
             '    return None if form is None else PVC_REFUSAL\n', 1)),
        ("IGC 1.0.15468's signal taken for the gateway message",
         sub(Path('src/tensorforge/toolchain.py'), "        return 'flag-producers'\n",
             "        return 'gateway'\n", 1)),
        ("a PVC ring's helper told one slot more",
         sub(ENGINES, "    at = (f'{named_ring_id(obj, side)}, {obj.slots}, {counts[0]}, '\n",
             "    at = (f'{named_ring_id(obj, side)}, {obj.slots + 1}, {counts[0]}, '\n", 1)),
        ("a PVC ring's side counted as the other side",
         sub(ENGINES, "    counts = ((spec.full, full) if side == 'full' else (spec.empty, empty))\n",
             "    counts = ((spec.empty, empty) if side == 'full' else (spec.full, full))\n", 1)),
        ("the kernel's named barriers never counted",
         sub(BUILD, '        if self._named:\n', '        if False:\n', 1)),
        ("SPMD refused for no reason of its own",
         sub(Path('src/tensorforge/backend/pir/roles.py'),
             "    if fam == 'sycl':\n        return SPMD_REFUSAL\n", '', 1)),
        ('a ring of mbarriers below sm_90 let through',
         sub(ENGINES, '        if level is None or level < 90:\n', '        if False:\n', 1)),
        ("the oracle's arms never handing over at an LDS ring's wait",
         sub(ORACLE, "        if not stmt.startswith(('tensorforge::mbarrierWait',\n"
                     "                                'tensorforge::ldsRingWait')):\n",
             "        if not stmt.startswith('tensorforge::mbarrierWait'):\n", 1)),
    ]),
    # The three-role demonstrator (`Options.matrix_roles`,
    # `multi_element._tcgen05_roles`): each role's acts, the rings named to
    # their roles, the mma role's D and output, the plan's refusals and the
    # oracle's landing of products across roles.
    'threeroles': ('tests/test_three_roles.py -n 6', [
        ("the operator role filling no chunk",
         sub(ME_SECTION, '                ring.fill(trip, j, base, issue)\n',
             '                pass\n', 1)),
        ("the mma role giving no slot of B back behind its blocks",
         sub(ME_SECTION, '            if j >= hold - 1:\n'
                         '                ring.release(trip, j - hold + 1, base, issue)\n',
             '            if False:\n'
             '                ring.release(trip, j - hold + 1, base, issue)\n', 1)),
        ("the mma role's last held slots of B never given back",
         sub(ME_SECTION, '        for j in range(max(0, fold.blocks - hold + 1), fold.blocks):\n'
                         '            ring.release(trip, j, base, issue)\n', '', 1)),
        ("A's ring named to no roles",
         sub(ME_SECTION, "    fold = _Fold(trip, step, M, producer='star', consumers=('mma',))\n",
             '    fold = _Fold(trip, step, M)\n', 1)),
        ("B's ring named to no roles in the three roles",
         sub(ME_SECTION, "            named = (dict(producer='operator', consumers=('mma',))\n",
             '            named = (dict()\n', 1)),
        ("every row's output through the group's first element",
         sub(ME_SECTION, "    source = b.op('select', SIZE, held, e, group, hint='e')\n",
             '    source = group\n', 1)),
        ("the epilogue's rows by the warp, not its quarter",
         sub(ME_SECTION, "        'rem', INDEX, b.thread_id('y'), 4, hint='q'), 32, hint='m'),\n",
             "        'rem', INDEX, b.thread_id('y'), 8, hint='q'), 32, hint='m'),\n", 1)),
        ("the star warps' staging not met behind the last trip's bands",
         sub(ME_SECTION, "            # every element's rows.\n            b.barrier(Participants.ROLE)\n",
             "            # every element's rows.\n", 1)),
        ("the mma role's barrier ahead of the next trip's products dropped",
         sub(ME_SECTION, "        b.fence(FenceKind.THREAD_SYNC, engine='tcgen05', side='before')\n"
                         '        b.barrier(Participants.ROLE)\n',
             "        b.fence(FenceKind.THREAD_SYNC, engine='tcgen05', side='before')\n", 1)),
        ("D read ahead of the wait for its products",
         sub(ME_SECTION, "        b.commit_group('tcgen05', object=cursor, issue=issue)\n"
                         "        b.matrix_wait('tcgen05', object=cursor, tmem=[trip.tmem])\n"
                         "        b.fence(FenceKind.THREAD_SYNC, engine='tcgen05', side='after')\n"
                         '        _tcgen05_role_epilogue(trip, step, M)\n',
             "        b.commit_group('tcgen05', object=cursor, issue=issue)\n"
             "        b.fence(FenceKind.THREAD_SYNC, engine='tcgen05', side='after')\n"
             '        _tcgen05_role_epilogue(trip, step, M)\n', 1)),
        ("roles where the star step is not folded",
         sub(ME_GROUP, "    if not folded(section, step):\n"
                       "        return ('the star step is not folded into the operator step '\n",
             '    if False:\n'
             "        return ('the star step is not folded into the operator step '\n", 1)),
        ("roles on wgmma",
         sub(ME_GROUP, "    if step.lowering != 'tcgen05':\n"
                       "        return (f'the step is {step.lowering}\\'s; three roles are tcgen05\\'s '\n",
             '    if False:\n'
             "        return (f'the step is {step.lowering}\\'s; three roles are tcgen05\\'s '\n", 1)),
        ("the comment's mma warps from the operator's",
         sub(ME_GROUP, '        mma = e + OPERATOR_WARPS\n', '        mma = e\n', 1)),
        ("the generator lowering the section again without its roles",
         sub(GEN, '          output=element_group.output_of(planned), roles=planned.roles)\n',
             '          output=element_group.output_of(planned))\n', 1)),
        ("the oracle's check of a landing product counted as an access",
         sub(ORACLE, '                if not _same(self.mem.peek(base, slot), value):\n',
             '                if not _same(self.mem.read(base, slot), value):\n', 1)),
    ]),
    # --- the high-order tile, H1: dense rows, consumer warpgroups (tensorforge-hotile)
    'hirows': ('tests/test_hiorder_tile.py -n 6', [
        ("a tile's rows of A from the instruction's 64, not its elements'",
         sub(ME_SECTION, "    m = _index(b, 'add', r, tile.used * t, 'm')\n",
             "    m = _index(b, 'add', r, tile.m * t, 'm')\n", 1)),
        ("a tile's rows of D from the instruction's 64, not its elements'",
         sub(ME_SECTION, "                      _index(b, 'add', local, tile.used * t, 'm'), M,\n",
             "                      _index(b, 'add', local, tile.m * t, 'm'), M,\n", 1)),
        ("the tile's rings counting a warpgroup's arrival, not a warp's",
         sub(ME_SECTION, '    return (WARPGROUP_WARPS * tiles,\n', '    return (tiles,\n', 1)),
        ("the inputs staged by the operator ring's warp",
         sub(ME_SECTION, '                              bound[MASK_KEY], warp=first)\n',
             '                              bound[MASK_KEY], warp=first + 1)\n', 1)),
        ("a tile's role counting half its threads",
         sub(ME_SECTION, "    roles = tuple(Role(f'tile{t}', lanes * WARPGROUP_WARPS,\n",
             "    roles = tuple(Role(f'tile{t}', lanes * 2,\n", 1)),
        ("both tiles' A in one buffer",
         sub(ME_SECTION, '                _wgmma_tile(trip, step, t, areas[t], done)\n',
             '                _wgmma_tile(trip, step, t, areas[0], done)\n', 1)),
        ("the input slot given back behind the first block of A",
         sub(ME_SECTION, '            if last and j + 1 == end:\n'
                         "                # Every read of the trip's inputs is behind: the slot back.\n",
             '            if j == first:\n'
             "                # Every read of the trip's inputs is behind: the slot back.\n", 1)),
        ("a tile's rows past its elements taken as theirs",
         sub(ME_GROUP, '        return self.per * self.width\n',
             '        return self.per * self.width + 1\n', 1)),
        ("the block's rows an element's in the tile's layout",
         sub(ME_GROUP, '        return 0 if rowless(self.section) else self.elements\n',
             '        return self.elements\n', 1)),
        ("one consumer warpgroup fewer than the options give",
         sub(ME_GROUP, '        for tiles in range(consumers, 0, -1):\n',
             '        for tiles in range(consumers - 1, 0, -1):\n', 1)),
        ("the generator lowering the section again without its tile",
         sub(GEN, '      if planned.tile is not None:\n'
                  "        # The high-order tile's own lowering (`element_group.tile_plan`).\n",
             '      if False:\n'
             "        # The high-order tile's own lowering (`element_group.tile_plan`).\n", 1)),
        ("the batch loop's rows of a block all an element's",
         sub(BATCHLOOP, '            if mults != self._element_rows + self._producer_rows:\n',
             '            if mults != self._group_size + self._producer_rows:\n', 1)),
    ]),
    # --- the high-order tile, H2: B in kept spans (tensorforge-hotile)
    'hispans': ('tests/test_hiorder_tile.py -n 6', [
        ("a span starting where its first entry is, off the engine's granule",
         sub(ME_GROUP, '    lo = found[0] - found[0] % granule\n', '    lo = found[0]\n', 1)),
        ("a span's last granule dropped",
         sub(ME_GROUP, '    hi = min(stage.width, -(-(found[-1] + 1) // granule) * granule)\n',
             '    hi = min(stage.width, (found[-1] + 1) // granule * granule)\n', 1)),
        ("the first block's first piece not whole",
         sub(ME_GROUP, '            if not out and not pieces:\n                span = (0, stages[p].width)\n',
             '', 1)),
        ("every block's first piece whole",
         sub(ME_GROUP, '            if not out and not pieces:\n                span = (0, stages[p].width)\n',
             '            if not pieces:\n                span = (0, stages[p].width)\n', 1)),
        ("the spans the options ask for not the tile's on wgmma",
         sub(ME_GROUP, "            tile = TileRows('wgmma', width, max(1, per), tiles,\n"
                       "                            spans_asked(context),\n",
             "            tile = TileRows('wgmma', width, max(1, per), tiles,\n"
                       "                            'dense',\n", 1)),
        ("the spans kept where known zeros are not skipped",
         sub(ME_GROUP, "    if not getattr(options, 'skip_known_zeros', True):\n        return 'dense'\n", '', 1)),
        ("an inactive group taken",
         sub(ME_GROUP, '        if empty:\n            return (f\'block{"s" if len(empty) > 1 else ""} \'\n',
             '        if False:\n            return (f\'block{"s" if len(empty) > 1 else ""} \'\n', 1)),
        ("the table's span read from the stage's first row",
         sub(ME_GROUP, '            for n in range(lo, hi):\n', '            for n in range(0, hi - lo):\n', 1)),
        ("a chunk counted at its stages' widths",
         sub(ME_GROUP, '    return sum(piece[2] * ((piece[4] - piece[3]) if len(piece) > 3\n',
             '    return sum(piece[2] * ((piece[4] - piece[3]) if False\n', 1)),
        ("a piece's lower halves past the stage's width, not its span's",
         sub(ME_SECTION, '        half = self.rows * self.width\n',
             '        half = self.rows * self.stage.width\n', 1)),
        ("a chunk's pieces the stage's width apart in its slot",
         sub(ME_SECTION, '            offset += 2 * rows * (hi - lo)\n',
             '            offset += 2 * rows * stage.width\n', 1)),
        ("a wgmma span's products into D's registers of its columns, not from the first",
         sub(ME_SECTION, '                span = accs[:piece.width // 2]\n',
             '                span = accs[piece.lo // 2:(piece.lo + piece.width) // 2]\n', 1)),
        ("a wgmma span short of its stage's last row",
         sub(ME_GROUP, '            if tile is not None and tile.descending:\n'
                       '                # On to the stage\'s last row, the first of the descending\n'
                       '                # order (`TileRows.descending`).\n'
                       '                span = (span[0], stages[p].width)\n', '', 1)),
        ("the wgmma spans' tiles laid first first in the table",
         sub(ME_GROUP, '        if section.tile is not None and section.tile.descending:\n'
                       '            # The span\'s tiles of N last first (`TileRows.descending`).\n',
             '        if False:\n'
             '            # The span\'s tiles of N last first (`TileRows.descending`).\n', 1)),
        ("D's columns read back in the ascending order on wgmma",
         sub(ME_SECTION, "                       phi - 8 * (i // 4 + 1) + i % 2, hint='n')\n",
             "                       phi - (phi - plo) + 8 * (i // 4) + i % 2, hint='n')\n", 1)),
        ("a span's products into D's slot's first columns on tcgen05",
         sub(ME_SECTION, "                                  'add', INDEX, d_base, piece.lo, hint='dc'),\n",
             "                                  'add', INDEX, d_base, 0, hint='dc'),\n", 1)),
        ("a span multiplied at the step's N on tcgen05",
         sub(ME_SECTION, '                    b.tcgen05(n if piece.width == piece.stage.width\n'
                         '                              and not piece.lo else piece.width, tmem,\n',
             '                    b.tcgen05(n, tmem,\n', 1)),
        ("the spans' option defaulting to dense",
         sub(OPTIONS, "declare('matrix_spans',\n        default='kept',\n",
             "declare('matrix_spans',\n        default='dense',\n", 1)),
    ]),
    # --- the high-order tile, H3: the group engine on wgmma (tensorforge-hotile)
    'higroups': ('tests/test_hiorder_tile.py tests/test_tcgen05.py -n 6', [
        ("every block's A into the first slot",
         sub(ME_SECTION, "        at = b.op('mul', INDEX, b.op('rem', INDEX, j, slots, hint='gs'),\n",
             "        at = b.op('mul', INDEX, b.op('rem', INDEX, j, 1, hint='gs'),\n", 1)),
        ("every block mixing the first rows of K",
         sub(ME_SECTION, "                'mul', INDEX, block, STAR_ROWS, hint='k'), hint='k'), live,\n",
             "                'mul', INDEX, block, 0, hint='k'), hint='k'), live,\n", 1)),
        ("every block dispatched at the first block's width",
         sub(ME_SECTION, "            f'{table}[{{0}}]', j, type_=INDEX, pure=True, movable=True,\n",
             "            f'{table}[{{0}}]', 0, type_=INDEX, pure=True, movable=True,\n", 1)),
        ("the dispatch on a width ptxas cannot tell is the warp's alike",
         sub(ME_SECTION, "        eighths = b.lane_broadcast(b.rawexpr(\n"
                         "            f'{table}[{{0}}]', j, type_=INDEX, pure=True, movable=True,\n"
                         "            hint='gw'), 0, 32, hint='gw')\n",
             "        eighths = b.rawexpr(\n"
             "            f'{table}[{{0}}]', j, type_=INDEX, pure=True, movable=True,\n"
             "            hint='gw')\n", 1)),
        ("a product's lower halves of B read where its upper ones are",
         sub(ME_SECTION, "                'mul', INDEX, part, 2 * p + k, hint='gb'), hint='gb'),\n"
                         "                                desc=bdesc, hint='b')\n",
             "                'mul', INDEX, part, 2 * p, hint='gb'), hint='gb'),\n"
             "                                desc=bdesc, hint='b')\n", 1)),
        ("the loop's wait leaving every block in flight",
         sub(ME_SECTION, "        b.matrix_wait('wgmma', hold - 1, reads=[\n",
             "        b.matrix_wait('wgmma', hold, reads=[\n", 1)),
        ("the chunk in hand given back behind the wait",
         sub(ME_SECTION, "                ring.release(trip, b.op('sub', INDEX, j, hold - 1,\n"
                         "                                        hint='gr'), base)\n",
             "                ring.release(trip, b.op('sub', INDEX, j, 0,\n"
             "                                        hint='gr'), base)\n", 1)),
        ("the dispatch's leaves the narrowest width's alone",
         sub(ME_SECTION, '        for width in sorted(set(widths)):\n',
             '        for width in sorted(set(widths))[:1]:\n', 1)),
        ("each of the loop's products writing D fresh",
         sub(ME_SECTION, '                        b.wgmma(width, [a[ia]], bs[ib], accs[:width // 2],\n'
                         '                                scale_d=1, layouts=shapes, term=(ia, ib),\n',
             '                        b.wgmma(width, [a[ia]], bs[ib], accs[:width // 2],\n'
             '                                scale_d=0, layouts=shapes, term=(ia, ib),\n', 1)),
        ("the group loop left to the compiler to unroll",
         sub(ME_SECTION, "    with b.for_(first, first + taken, unroll=1, hint='g') as loop:\n",
             "    with b.for_(first, first + taken, unroll=True, hint='g') as loop:\n", 1)),
        # tcgen05's group engine
        ("tcgen05's block products into D's first columns",
         sub(ME_SECTION, '                          d_column=lo, term=(ia, ib), plan=plan,\n',
             '                          d_column=0, term=(ia, ib), plan=plan,\n', 1)),
        ("tcgen05's blocks at the widest N",
         sub(ME_SECTION, '                b.tcgen05(n, tmem, bs[ib], tmem, issue=issue,\n'
                         '                          enable_input_d=fresh if first else 1,\n',
             '                b.tcgen05(widths[-1], tmem, bs[ib], tmem, issue=issue,\n'
             '                          enable_input_d=fresh if first else 1,\n', 1)),
        ("the star role's loop taking a slot of A not given back",
         sub(ME_SECTION, '        b.ring_acquire(self.areas, slot, parity=parity)\n', '', 1)),
        ("tcgen05's every block writing D fresh",
         sub(ME_SECTION, '                          enable_input_d=fresh if first else 1,\n',
             '                          enable_input_d=0 if first else 1,\n', 1)),
        ("tcgen05's lower halves of A read where the upper ones are",
         sub(ME_SECTION, '                                        2 * STAR_ROWS * p + STAR_ROWS * ia,\n',
             '                                        2 * STAR_ROWS * p,\n', 1)),
        ("tcgen05's lower halves of B read where the upper ones are",
         sub(ME_SECTION, "                'mul', INDEX, part, 2 * p + k, hint='gb'), hint='gb'),\n"
                         "                                desc=desc, hint='b')\n",
             "                'mul', INDEX, part, 2 * p, hint='gb'), hint='gb'),\n"
             "                                desc=desc, hint='b')\n", 1)),
        ("the star role's loop mixing the first rows of K",
         sub(ME_SECTION, "        k0 = b.op('mul', INDEX, j, STAR_ROWS, hint='k')\n",
             "        k0 = b.op('mul', INDEX, j, 0, hint='k')\n", 1)),
        ("a slot of A's ring used once a trip in the loop's parity",
         sub(ME_SECTION, "            'mul', INDEX, trips, uses, hint='ta'), b.op(\n",
             "            'mul', INDEX, trips, 1, hint='ta'), b.op(\n", 1)),
        # the PIR's values of N and of the columns (tcgen05, tmem_st)
        ("a value of N put into the descriptor unshifted",
         sub(ENGINES, "                      f'>> 3) << 17)')\n",
             "                      f') << 17)')\n", 1)),
        ("a column that is a value left out of its address",
         sub(ENGINES, "            text += f' + static_cast<uint32_t>({env.name(runtime[key])})'\n",
             '            pass\n', 1)),
        ("a store's column that is a value stored as a column's value",
         sub(ENGINES, "    values = s.args[2:] if s.attr('runtime_column') else s.args[1:]\n",
             '    values = s.args[1:]\n', 1)),
        ("the widths of a value of N unchecked",
         sub(ENGINES, '        if not widths or any(w not in TCGEN05_N for w in widths) \\\n',
             '        if not widths or False \\\n', 1)),
        ("a value of N's statement as wide as its narrowest width",
         sub(BUILD, '            widest = max(int(w) for w in widths)\n',
             '            widest = min(int(w) for w in widths)\n', 1)),
        ("the groups the options ask for not the tile's on wgmma",
         sub(ME_GROUP, "                            tile_groups('wgmma', context, tiles,\n"
                       "                                        step.rows[1] - step.rows[0]),\n",
             "                            'loop',\n", 1)),
        ("a block's products at their own spans in the loop",
         sub(ME_GROUP, "        if pieces and tile is not None and tile.groups == 'loop':\n",
             '        if False:\n', 1)),
        ("the groups' option defaulting to written out",
         sub(OPTIONS, "declare('matrix_groups',\n        default='auto',\n",
             "declare('matrix_groups',\n        default='unrolled',\n", 1)),
    ]),
    # --- the high-order tile, H3c: tcgen05's own layout, D in two slots (tensorforge-hotile)
    'hioverlap': ('tests/test_hiorder_tile.py -n 6', [
        ("every trip's D in the first slot",
         sub(ME_SECTION, "        dbase = b.op('mul', INDEX, dslot, n, hint='db')\n",
             "        dbase = b.op('mul', INDEX, dslot, 0, hint='db')\n", 1)),
        ("the mma warp not waiting for D's slot back",
         sub(ME_SECTION, '                b.ring_acquire(dring, dslot, dparity)\n', '', 1)),
        ("the epilogue reading D ahead of its products",
         sub(ME_SECTION, '            b.ring_take(dring, dslot, dparity)\n', '', 1)),
        ("the products committed onto D's empty side",
         sub(ME_SECTION, "            b.commit_group('tcgen05', object=dring, issue=lane0, slot=dslot,\n"
                         "                           side='full')\n",
             "            b.commit_group('tcgen05', object=dring, issue=lane0, slot=dslot,\n"
             "                           side='empty')\n", 1)),
        ("the epilogue reading the first slot of D",
         sub(ME_SECTION, '            _tcgen05_role_epilogue(trip, step, M, base=dbase)\n',
             '            _tcgen05_role_epilogue(trip, step, M, base=None)\n', 1)),
        ("A's ring behind one slot of D",
         sub(ME_SECTION, '        if section.tile is not None:\n            n *= section.tile.dslots\n', '', 1)),
        ("D's slots one where two fit",
         sub(ME_GROUP, '    return 2 if 2 * tile_n + ring <= budget else 1\n', '    return 1\n', 1)),
        ("the budget of tensor memory past a block's",
         sub(ME_GROUP, 'TMEM_BUDGET = 512\n', 'TMEM_BUDGET = 1024\n', 1)),
        ("the bands the options ask for not the tile's",
         sub(ME_GROUP, "        bands = min(int(getattr(context.get_user_options(), 'matrix_bands',\n"
                       "                                TILE_BANDS)), aslots)\n",
             '        bands = min(TILE_BANDS, aslots)\n', 1)),
        ("the star bands from the element count",
         sub(ME_SECTION, '            # The high-order tile\'s star bands (`TileRows.bands`).\n'
                         '            bands = trip.section.tile.bands\n',
             '            # The high-order tile\'s star bands (`TileRows.bands`).\n'
             '            bands = bands\n', 1)),
        ("the inputs' ring filled by wgmma's producer role",
         sub(ME_SECTION, "        producer = 'input' if section.tile.engine == 'tcgen05' else 'producer'\n",
             "        producer = 'producer'\n", 1)),
        ("a commit onto a ring's full side taken for a release",
         sub(Path('src/tensorforge/backend/pir/rings.py'),
             "            return 'publish', 'full'\n",
             "            return 'release', 'empty'\n", 1)),
        # the epilogue by bulk reductions
        ("the tile's warps taken by their element, past the elements",
         sub(ME_SECTION, "        self.slot = (builder.thread_id('y') if _rowless(self.section) else\n",
             "        self.slot = (builder.thread_id('y') if False else\n", 1)),
        ("the tile's bulk epilogue asked and a reduction a cell taken",
         sub(ME_GROUP, "    output = 'red' if tile is None else tile.output\n",
             "    output = 'red'\n", 1)),
        ("a row of D staged one column of the box past its own",
         sub(ME_SECTION, "    at = b.op('mul', INDEX, row, stride, hint='d')\n    staged = {}\n",
             "    at = b.op('mul', INDEX, row + 1, stride, hint='d')\n    staged = {}\n", 1)),
        ("D's slot given back ahead of its last loads",
         subs(ME_SECTION,
              ("        column = b.op('add', INDEX, base, c0, hint='dc')\n",
               "        if not c0:\n            release()\n"
               "        column = b.op('add', INDEX, base, c0, hint='dc')\n"),
              ('        if c0 >= end:\n            # The row of D in registers: the slot back to the mma warp.\n'
               '            release()\n', ''))),
        ("the box's stores not fenced for the async proxy",
         sub(ME_SECTION, "    b.fence(FenceKind.PROXY)\n    b.barrier(Participants.ROLE)\n"
                         "    element = b.op('div', INDEX, row, width, hint='e')\n",
             "    b.barrier(Participants.ROLE)\n"
             "    element = b.op('div', INDEX, row, width, hint='e')\n", 1)),
        ("a masked element's columns added too",
         sub(ME_SECTION, "    issue = b.op('and', BOOL, live, held, hint='issue')\n",
             '    issue = live\n', 1)),
        ("an element's box added by each of its columns' threads",
         sub(ME_SECTION, "        issue = b.op('and', BOOL, issue, b.op('eq', BOOL, quantity, 0,\n",
             "        issue = b.op('or', BOOL, issue, b.op('eq', BOOL, quantity, 0,\n", 1)),
        ("the tile's reductions not waited for",
         sub(ME_SECTION, "    # The reductions' reads of the box done, in the thread that issued them.\n"
                         "    b.wait(token)\n",
             '', 1)),
        ("a thread's lane of tensor memory its warp's, not its quarter's",
         sub(ME_SECTION, "    quarter = b.op('rem', INDEX, b.thread_id('y'), 4, hint='q')\n",
             "    quarter = b.op('rem', INDEX, b.thread_id('y'), 8, hint='q')\n", 1)),
        ("the box written again ahead of the reductions' reads",
         sub(ME_SECTION, "    # The box's next writes, the next trip's, behind every reduction's reads\n"
                         "    # of it: an element's box is read by one thread and written by all of\n"
                         "    # its columns'.\n    b.barrier(Participants.ROLE)\n",
             '', 1)),
        ("a load of tensor memory at a value column at its first",
         sub(ENGINES, "        return _stmt(Op.TMEM_LD, (('column', 0), ('runtime_column', True)),\n"
                      "                     args=(tmem, column), target=tuple(targets),\n",
             "        return _stmt(Op.TMEM_LD, (('column', 0),),\n"
             "                     args=(tmem,), target=tuple(targets),\n", 1)),
    ]),
    # --- the high-order tile, H3d: the tiles by cluster launch control (tensorforge-hotile)
    'hisched': ('tests/test_hiorder_tile.py -n 6', [
        ("the fetcher publishing the static stride where its launch claims",
         sub(Path('src/tensorforge/backend/instructions/batch_loop.py'),
             '                         self.claiming, tile, self.grid, self.tiles,\n',
             '                         False, tile, self.grid, self.tiles,\n', 1)),
        ("every warp reading the round after the one published",
         sub(Path('src/tensorforge/backend/instructions/batch_loop.py'),
             "        slot, parity = b.ring_round(self.ring, trips, hint='tq')\n"
             "        b.ring_take(self.ring, slot, parity, reads=[self.ids])\n",
             "        slot, parity = b.ring_round(self.ring, trips + 1, hint='tq')\n"
             "        b.ring_take(self.ring, slot, parity, reads=[self.ids])\n", 1)),
        ("a claimed tile ending the loop",
         sub(Path('src/tensorforge/backend/instructions/batch_loop.py'),
             "            loop.exit_when(writer.op('lt', BOOL, nxt, 0, hint='drained'))\n",
             "            loop.exit_when(writer.op('ge', BOOL, nxt, 0, hint='drained'))\n", 1)),
        ("the carried trip count left where it was",
         sub(Path('src/tensorforge/backend/instructions/batch_loop.py'),
             "                'shl', SIZE, writer.op('add', SIZE, trips, 1, hint='trips'),\n",
             "                'shl', SIZE, writer.op('add', SIZE, trips, 0, hint='trips'),\n", 1)),
        ("the rings' trips the static grid's in the scheduled loop",
         sub(ME_SECTION, '    if TRIPS_KEY in bound:\n        trip.trips = b.op(',
             '    if False:\n        trip.trips = b.op(', 1)),
        ("the fetcher never claiming",
         sub(ME_SECTION, '                scheduled[0].claim(b, lane0)\n', '', 1)),
        ("the next tile never published",
         sub(ME_SECTION, '                scheduled[0].publish(b, trips, scheduled[1], lane0)\n', '', 1)),
        ("a block a tile under clc not launched",
         sub(GEN, '          num_blocks = tiles\n          if self._tile_schedule() == \'auto\' and bounded:\n',
             "          num_blocks = f'std::min(gridsize, {stacks})'\n"
             "          if self._tile_schedule() == 'auto' and bounded:\n", 1)),
        ("auto's blocks a tile from one wave on",
         sub(GEN, "            num_blocks = (f'({tiles} >= 2 * gridsize ? {tiles} : '\n",
             "            num_blocks = (f'({tiles} >= gridsize ? {tiles} : '\n", 1)),
        ("the schedule taken where the target has no launch control",
         sub(ME_GROUP, "    if asked == 'static' or engine != 'tcgen05' \\\n"
                       "            or not context.target.launch_control():\n",
             "    if asked == 'static' or engine != 'tcgen05':\n", 1)),
        ("the schedule the options ask for not the tile's",
         sub(ME_GROUP, "                        tile_schedule('tcgen05', context),\n",
             "                        'static',\n", 1)),
        ("the schedule's shared memory not counted",
         sub(ME_GROUP, '        operators += SCHEDULE_BYTES // size\n', '', 1)),
        ("a claim granted twice on the oracle",
         sub(ORACLE, '            if cta in cancelled or cta == block:\n',
             '            if cta == block:\n', 1)),
        ("a cancelled block run on the oracle",
         sub(ORACLE, "            if cta in getattr(mem, 'tile_cancelled', ()):\n",
             '            if False:\n', 1)),
    ]),
    # --- the high-order tile, H4: orders 10 and 12 -- wgmma's passes over N, tcgen05's products cut at 256
    # (tensorforge-hotile)
    'hipasses': ('tests/test_hiorder_tile.py -n 6', [
        ("every pass from the stage's first row",
         sub(ME_GROUP, '    return tuple((min(width, units * p // passes * granule),\n',
             '    return tuple((min(width, 0),\n', 1)),
        ("a pass's pieces not cut to its rows",
         sub(ME_GROUP, '            span = (max(span[0], plo), min(span[1], phi))\n',
             '            span = span\n', 1)),
        ("a pass's first piece not whole over the pass",
         sub(ME_GROUP, '            if not out and not pieces:\n                span = (plo, phi)\n',
             '            if False:\n                span = (plo, phi)\n', 1)),
        ("a pass's spans on wgmma short of its last row",
         sub(ME_GROUP, '            if tile.descending:\n                span = (span[0], phi)\n',
             '            if tile.descending:\n                span = (span[0], span[1])\n', 1)),
        ("one pass where D does not fit the registers",
         sub(ME_GROUP, '    return max(-(-n // (2 * TILE_ACCUMULATORS)), -(-n // ASYNC_N))\n',
             '    return 1\n', 1)),
        ("the passes the options force ignored",
         sub(ME_GROUP, '    if asked:\n        return asked\n', '', 1)),
        ("wgmma's steps wider than an instruction refused",
         sub(ME_GROUP, "        # Its passes over N (`tile_passes`).\n        return 'wgmma'\n",
             "        # Its passes over N (`tile_passes`).\n        return None\n", 1)),
        ("a pass's D not zeroed in the loop",
         sub(ME_SECTION, '            for acc in accs:\n                b.assign(acc, b.const(0.0, fp))\n', '', 1)),
        ("the input slot given back behind every pass",
         sub(ME_SECTION, '                          done if last else None, accs,\n',
             '                          done, accs,\n', 1)),
        ("a pass's blocks read at the chunks' index",
         sub(ME_SECTION, '    stride = blocks != list(range(len(chunks)))\n', '    stride = False\n', 1)),
        ("a pass's releases counted from the trip's first chunk",
         sub(ME_SECTION, "            with b.if_(b.op('ge', BOOL, j, first + hold - 1, hint='gr')):\n",
             "            with b.if_(b.op('ge', BOOL, j, hold - 1, hint='gr')):\n", 1)),
        ("a chunk's B at its block's first piece's offset",
         sub(ME_SECTION, '                starts = [offsets[tuple(chunk[0])] for chunk in chunks]\n',
             '                starts = starts\n', 1)),
        ("a pass's D read past its rows",
         sub(ME_SECTION, '        for i in range(_round_up(phi - plo, n_) // 2):\n',
             '        for i in range(widest // 2):\n', 1)),
        ("a pass's D read with its products in flight",
         sub(ME_SECTION, '            # The pass\'s every product retired: its D whole.\n'
                         '            flight = _wgmma_retire(b, flight, 0, accs)\n',
             '            # The pass\'s every product retired: its D whole.\n'
             '            flight = flight\n', 1)),
        ("tcgen05's wide products uncut",
         sub(ME_SECTION, '                if wide > TCGEN05_N_MAX:\n', '                if False:\n', 1)),
        ("tcgen05's cut products at the piece's first columns",
         sub(ME_SECTION, "                        cut = piece.descriptors(b, k0 + kt, 'tcgen05', c0)\n",
             "                        cut = piece.descriptors(b, k0 + kt, 'tcgen05', 0)\n", 1)),
        ("A's two slots where four do not fit beside D",
         sub(ME_GROUP, "    for dslots, aslots in ((2, STAR_SLOTS['tcgen05']),\n"
                       "                           (1, STAR_SLOTS['tcgen05']), (1, 2)):\n",
             "    for dslots, aslots in ((2, STAR_SLOTS['tcgen05']),\n"
             "                           (1, STAR_SLOTS['tcgen05'])):\n", 1)),
        ("B's ring's slots the options' alone",
         sub(ME_GROUP, '    return tuple(range(asked, 1, -1)) or (asked,)\n', '    return (asked,)\n', 1)),
    ]),
    'hichunks': ('tests/test_hiorder_tile.py -n 6', [
        ('I whole where it has more rows than a chunk',
         sub(ME_GROUP, '    return asked or TILE_CHUNK\n',
             '    return 0\n', 1)),
        ("a chunk's rows past I's box",
         sub(ME_GROUP, '    return tuple((lo, min(lo + rows, buffer.rows[1]))\n',
             '    return tuple((lo, lo + rows)\n', 1)),
        ('the chunks not counted in shared memory',
         sub(ME_GROUP, '        per_element += tile.cslots * tile.chunk * (chunked.cols[1]\n',
             '        per_element += 0 * tile.cslots * tile.chunk * (chunked.cols[1]\n', 1)),
        ("the chunk ring's mbarriers not counted",
         sub(ME_GROUP, '        operators += 2 * tile.cslots * 8 // size\n',
             '        operators += 0\n', 1)),
        ('I in every slot of the input ring beside its chunks',
         sub(ME_GROUP, '        per_element -= chunked.room + (section.ring - 1) * chunked.per_element\n',
             '', 1)),
        ("a trip's chunks only its first pass's",
         sub(ME_GROUP, '    for first, count in tile_pass_chunks(section, step):\n        seen: List[int] = []\n',
             '    for first, count in tile_pass_chunks(section, step)[:1]:\n        seen: List[int] = []\n', 1)),
        ("every block its pass's first to read its chunk",
         sub(ME_GROUP, '            out.append((h, at + order.index(h), i == 0 or hs[i - 1] != h,\n',
             '            out.append((h, at + order.index(h), True,\n', 1)),
        ("every block its pass's last to read its chunk",
         sub(ME_GROUP, '                        i + 1 == len(hs) or hs[i + 1] != h))\n',
             '                        True))\n', 1)),
        ("a block's chunk counted in blocks",
         sub(ME_GROUP, '    per = section.tile.chunk // STAR_ROWS\n    chunks = tile_chunks(section, step)\n    out: List[Tuple[int, int, bool, bool]] = []\n',
             '    per = 1\n    chunks = tile_chunks(section, step)\n    out: List[Tuple[int, int, bool, bool]] = []\n', 1)),
        ('no input streamed in chunks',
         sub(ME_GROUP, '    if section.tile is None or not section.tile.chunk:\n        return None\n    return star_operand(section)\n',
             '    if True:\n        return None\n    return star_operand(section)\n', 1)),
        ("an element's chunk at its whole I's offset",
         sub(ME_SECTION, "        'mul', INDEX, element, chunks.per_element, hint='b'), hint='b'), row,\n",
             "        'mul', INDEX, element, group.per_element, hint='b'), hint='b'), row,\n", 1)),
        ("a chunk's rows read from I's first row",
         sub(ME_SECTION, "        'mul', INDEX, element, chunks.per_element, hint='b'), hint='b'), row,\n        chunks.rows)\n",
             "        'mul', INDEX, element, chunks.per_element, hint='b'), hint='b'),\n        group.rows[0], chunks.rows)\n", 1)),
        ("a chunk's columns at I's pitch",
         sub(ME_SECTION, "        'mul', INDEX, element, chunks.per_element, hint='b'), hint='b'), row,\n        chunks.rows)\n",
             "        'mul', INDEX, element, chunks.per_element, hint='b'), hint='b'), row,\n        group.pitch)\n", 1)),
        ("every chunk copied from I's first row",
         sub(ME_SECTION, '                src_index=(_address(b, buffer.tensor, lo,\n',
             '                src_index=(_address(b, buffer.tensor, buffer.rows[0],\n', 1)),
        ("a chunk's columns copied into one",
         sub(ME_SECTION, "                dst_index=(_index(b, 'add', start, c * chunks.rows, 'd'),),\n",
             '                dst_index=(start,),\n', 1)),
        ("the producer not waiting for a chunk's slot back",
         sub(ME_SECTION, '            b.ring_acquire(chunks.ring, slot, parity,\n                           writes=[chunks.storage])\n',
             '', 1)),
        ("every chunk's bytes announced as a whole chunk's",
         sub(ME_SECTION, '                           tx=elements * chunks.width * rows\n',
             '                           tx=elements * chunks.width * chunks.rows\n', 1)),
        ("a trip's rounds of chunks counted from the ring's first",
         sub(ME_SECTION, "        return _index(trip.builder, 'mul', _trip_count(trip),\n                      len(self.order), 'ic')\n",
             '        return 0\n', 1)),
        ("wgmma's blocks written out reading a chunk before it lands",
         sub(ME_SECTION, '                if take:\n                    chunks.take(trip, cslot, cparity)\n',
             '                if False:\n                    chunks.take(trip, cslot, cparity)\n', 1)),
        ("wgmma's blocks written out giving a chunk back at its first",
         sub(ME_SECTION, '            if chunks is not None and give:\n',
             '            if chunks is not None and take:\n', 1)),
        ("wgmma's group loop reading a chunk before it lands",
         sub(ME_SECTION, '            ichunks.take(trip, cslot, cparity)\n',
             '', 1)),
        ("wgmma's group loop a round a block",
         sub(ME_SECTION, "        rounds = trip.table('tq', [r[1] for r in reads_of])\n",
             "        rounds = trip.table('tq', list(range(len(reads_of))))\n", 1)),
        ("wgmma's group loop giving a chunk back every block",
         sub(ME_SECTION, "        gives = trip.table('tr', [int(r[3]) for r in reads_of])\n",
             "        gives = trip.table('tr', [1 for r in reads_of])\n", 1)),
        ("wgmma's input slot back behind the blocks with I in chunks",
         sub(ME_SECTION, '    if trip.chunks is not None:\n        # I streams in chunks through a ring of its own: the input slot\n',
             '    if False:\n        # I streams in chunks through a ring of its own: the input slot\n', 1)),
        ("tcgen05's bands keeping every chunk",
         sub(ME_SECTION, '                                   (cat, chunks.row(b, h)))\n                chunks.release(trip, cslot)\n',
             '                                   (cat, chunks.row(b, h)))\n', 1)),
        ("tcgen05's bands from the chunk's first block",
         sub(ME_SECTION, "                start = b.op('add', INDEX, lo, b.op('rem', INDEX, b.op(\n",
             "                start = b.op('add', INDEX, band, b.op('rem', INDEX, b.op(\n", 1)),
        ("tcgen05's bands on past the chunk",
         sub(ME_SECTION, "                end = b.op('min', INDEX, b.op('add', INDEX, lo, per,\n                                               hint='he'), self.blocks,\n                           hint='he')\n",
             '                end = self.blocks\n', 1)),
        ("tcgen05's bands reading a chunk before it lands",
         sub(ME_SECTION, "                chunks.take(trip, cslot, cparity)\n                lo = b.op('mul', INDEX, h, per, hint='hl')\n",
             "                lo = b.op('mul', INDEX, h, per, hint='hl')\n", 1)),
        ("tcgen05's written-out bands giving a chunk back at its first",
         sub(ME_SECTION, '                    if chunks is not None and (j % per == per - 1\n',
             '                    if chunks is not None and (j % per == 0\n', 1)),
        ("tcgen05's input slot back behind the blocks with I in chunks",
         sub(ME_SECTION, '            held = fold.rows(trip, M)\n            if trip.chunks is not None:\n',
             '            held = fold.rows(trip, M)\n            if False:\n', 1)),
    ]),
    'foldslots': ('tests/test_three_roles.py tests/test_hiorder_tile.py -n 6', [
        ("the bands' slots of A in turn between roles",
         sub(ME_SECTION, "        if named:\n            # Between roles, each slot one band's (`_fold_bands`): a slot's\n",
             "        if False:\n            # Between roles, each slot one band's (`_fold_bands`): a slot's\n", 1)),
        ("a slot of A more than the bands own alike",
         sub(ME_SECTION, '            self.slots = bands * (self.slots // bands)\n',
             '            self.slots = bands * (self.slots // bands) + 1\n', 1)),
        ("the bands counted in eights of the element warps",
         sub(ME_SECTION, '    return max(1, trip.group.elements // 4)\n',
             '    return max(1, trip.group.elements // 8)\n', 1)),
        ("the tile's bands past A's slots",
         sub(ME_GROUP, "                                TILE_BANDS)), aslots)\n",
             "                                TILE_BANDS)), 4)\n", 1)),
        ("the tile's slots of A past the bands' own",
         sub(ME_GROUP, '                        bands * (aslots // bands)',
             '                        aslots', 1)),
    ]),
    'hiout': ('tests/test_hiorder_tile.py -n 6', [
        ("wgmma's staged rows at the step's first row of the output",
         sub(ME_SECTION, '    shift = step.rows[0] - first - lo\n    count = _round_up(phi - plo, 8) // 2\n',
             '    shift = 0\n    count = _round_up(phi - plo, 8) // 2\n', 1)),
        ("wgmma's staged rows no cell of the pass is left as they were",
         sub(ME_SECTION, '                if blanks:\n                    with b.if_(lead):\n',
             '                if False:\n                    with b.if_(lead):\n', 1)),
        ("wgmma's staged rows not fenced for the async proxy",
         sub(ME_SECTION, "            b.fence(FenceKind.PROXY)\n            b.barrier(Participants.WAVE)\n"
                         "            local = _index(b, 'add', b.op('add', INDEX, b.op(\n",
             "            b.barrier(Participants.WAVE)\n"
             "            local = _index(b, 'add', b.op('add', INDEX, b.op(\n", 1)),
        ("a round's reductions not waited for on wgmma",
         sub(ME_SECTION, "            # it; the warp's barrier hands them to its next round's stores.\n            b.wait(token)\n",
             "            # it; the warp's barrier hands them to its next round's stores.\n", 1)),
        # Not "the warpgroup not meeting behind wgmma's last round": without
        # that barrier PIR's barrier pass places the warpgroup's own ahead of
        # the next blocks' writes of A's slots -- an equivalent mutant.
        ("a masked element's row added on wgmma",
         sub(ME_SECTION, "                'and', BOOL, inside, held, hint='issue'), hint='issue')\n",
             "                'and', BOOL, inside, inside, hint='issue'), hint='issue')\n", 1)),
        ("every quad staging its rows in every round on wgmma",
         sub(ME_SECTION, "            mine = b.op('and', BOOL, b.op('ge', BOOL, quad, g0, hint='qr'),\n",
             "            mine = b.op('and', BOOL, b.op('ge', BOOL, quad, 0, hint='qr'),\n", 1)),
        ("wgmma's staged rows an even number of sixteen bytes apart",
         sub(ME_GROUP, '    stride = _round(max(hi - lo, unit), unit)\n    if (stride // unit) % 2 == 0:\n'
                       '        stride += unit\n',
             '    stride = _round(max(hi - lo, unit), unit)\n', 1)),
        ("a pass's rows added at the output's first on wgmma",
         sub(ME_SECTION, '                                  dst_index=(_address(b, tensor, first + lo,\n',
             '                                  dst_index=(_address(b, tensor, first,\n', 1)),
        ("wgmma's tile declining the bulk epilogue",
         sub(ME_GROUP, "        return 'bulk' if output_asked(context) == 'bulk' else 'red'\n",
             "        return 'red'\n", 1)),
        ("wgmma's staged rows counted in shared memory",
         sub(ME_GROUP, "        if section.tile is not None and section.tile.engine == 'wgmma':\n"
                       "            # Staged in the warpgroups' slots of A (`wgmma_bulk_geometry`).\n"
                       "            continue\n", '', 1)),
    ]),
    # The high-order tile's defaults on tcgen05, as a GB200 measured them
    # (`Options.matrix_bands`, `Options.matrix_output` `auto`,
    # `element_group.tile_output`, `tile_plan`'s weighing of the bulk
    # reductions against a reduction a cell).
    'hiauto': ('tests/test_hiorder_tile.py -n 6', [
        ("three star bands by default",
         sub(OPTIONS, "declare('matrix_bands',\n        default=2,\n",
             "declare('matrix_bands',\n        default=3,\n", 1)),
        ("the plan's own bands three",
         sub(ME_GROUP, 'TILE_BANDS = 2\n', 'TILE_BANDS = 3\n', 1)),
        ("auto's box from half the rows",
         sub(ME_GROUP, "    return 'bulk' if rows < TILE_BOX_ROWS else 'box'\n",
             "    return 'bulk' if rows < TILE_BOX_ROWS // 2 else 'box'\n", 1)),
        ("auto's bulk and box the other way round",
         sub(ME_GROUP, "    return 'bulk' if rows < TILE_BOX_ROWS else 'box'\n",
             "    return 'box' if rows < TILE_BOX_ROWS else 'bulk'\n", 1)),
        ("auto a reduction a cell on the tile too",
         sub(ME_GROUP, "    asked = output_asked(context)\n    if asked != 'auto':\n        return asked\n",
             "    asked = output_asked(context)\n    if asked != 'auto':\n        return asked\n    return 'red'\n", 1)),
        ("a bulk or box asked taken as auto",
         sub(ME_GROUP, "    asked = output_asked(context)\n    if asked != 'auto':\n        return asked\n",
             "    asked = output_asked(context)\n    if asked == 'red':\n        return asked\n", 1)),
        ("wgmma's tile given auto's bulk reductions",
         sub(ME_GROUP, "        return 'bulk' if output_asked(context) == 'bulk' else 'red'\n",
             "        return 'bulk' if output_asked(context) in ('bulk', 'auto') else 'red'\n", 1)),
        ("auto's bulk reductions kept where they take elements",
         sub(ME_GROUP, '        if red[0] is not None and (\n                chosen[0] is None or\n',
             '        if False and (\n                chosen[0] is None or\n', 1)),
        ("auto's bulk reductions kept at any loss of elements",
         sub(ME_GROUP, 'TILE_OUTPUT_KEEP = 0.75\n', 'TILE_OUTPUT_KEEP = 0.0\n', 1)),
        ("auto's bulk reductions only where they keep every element",
         sub(ME_GROUP, 'TILE_OUTPUT_KEEP = 0.75\n', 'TILE_OUTPUT_KEEP = 1.0\n', 1)),
        ("the tile's epilogue from the options, not its plan",
         sub(ME_GROUP, "    output = 'red' if tile is None else tile.output\n",
             "    output = 'red' if tile is None else output_asked(context)\n", 1)),
        ("auto's notes on a kernel without the tile",
         sub(ME_GROUP, "        asked = section.tile.output if section.tile is not None else 'red'\n",
             "        asked = section.tile.output if section.tile is not None else 'bulk'\n", 1)),
        ("the loop by default",
         sub(OPTIONS, "declare('matrix_groups',\n        default='auto',\n",
             "declare('matrix_groups',\n        default='loop',\n", 1)),
        ("auto's groups one loop on wgmma too",
         sub(ME_GROUP, "    return 'unrolled' if tiles == 1 or rows <= TILE_UNROLLED_ROWS else 'loop'\n",
             "    return 'loop'\n", 1)),
        ("auto's groups written out on tcgen05 too",
         sub(ME_GROUP, "    if engine != 'wgmma':\n        return 'loop'\n",
             "    if engine != 'wgmma':\n        return 'unrolled'\n", 1)),
        ("two warpgroups' blocks written out past o8",
         sub(ME_GROUP, "    return 'unrolled' if tiles == 1 or rows <= TILE_UNROLLED_ROWS else 'loop'\n",
             "    return 'unrolled'\n", 1)),
        ("one warpgroup's blocks in the loop past o8",
         sub(ME_GROUP, "    return 'unrolled' if tiles == 1 or rows <= TILE_UNROLLED_ROWS else 'loop'\n",
             "    return 'unrolled' if rows <= TILE_UNROLLED_ROWS else 'loop'\n", 1)),
        ("the written-out rows' bound past o10",
         sub(ME_GROUP, 'TILE_UNROLLED_ROWS = 128\n', 'TILE_UNROLLED_ROWS = 256\n', 1)),
        ("a loop or written-out groups asked taken as auto",
         sub(ME_GROUP, "    asked = groups_asked(context)\n    if asked != 'auto':\n        return asked\n",
             "    asked = groups_asked(context)\n    if asked == 'never':\n        return asked\n", 1)),
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
