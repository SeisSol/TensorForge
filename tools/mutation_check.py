# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Do the guards actually catch anything?

Every check added here is only worth its runtime if it can fail.  A test that
passes because the property is trivially true, or because the test and the
code under test share a mistake, is worse than no test: it reads as coverage.

So each guard has a matching mutation --- the defect it was written for, put
back --- and this runs them all and reports which are caught.  The mutations
are the real ones from the session's history, not invented ones:

* `fmacdpp4` emitted for gfx900, where the specialisations are switched off
* `fmacdpp8` selected, which the runtime declares nowhere
* the MFMA tail recomputed, so two paths wrote the same columns
* the `LaneAxis` lane map as first documented, with `stride` read as packing
* the broadcast layout as first annotated: right numbers, wrong roles
* `0.0f` handed to a `T &` parameter of `transpose4x4b32`
* a shared buffer declared inside a body, outside the sized arena
* two scratch windows handed out at the same offset

Source files are edited in place and restored in a `finally`.  Run it on a
clean tree.

A `finally` does not survive SIGKILL, and a timeout is SIGKILL.  When that
happens the tree is left mutated, and the next thing anyone runs is usually
`pytest --snapshot-update`, which records the mutated generator's output as
the new baseline across dozens of files -- during a migration the only symptom
is a large diff, which there would be anyway.  That has happened.  So the
mutated paths are also written to a lock file, checked on startup and restored
from git before anything else runs.

    python3 tools/mutation_check.py            # all groups
    python3 tools/mutation_check.py layout     # one group
"""

import os
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
SYM = Path('src/tensorforge/backend/symbol.py')


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
    for cache in Path('tensorforge').rglob('__pycache__'):
        shutil.rmtree(cache, ignore_errors=True)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    return subprocess.run(
        [sys.executable, '-B', '-m', 'pytest', *target.split(), '-q', '-x',
         '--no-header'],
        capture_output=True, text=True, env=env)


def sub(path, old, new, count=0):
    """A mutation as (path, replacement-text), failing loudly if it no-ops.

    A mutation that does not apply silently reports its guard as working,
    which is the failure mode this whole file exists to avoid.
    """
    def make():
        if not path.exists():
            # Same failure as a mutation that no longer applies, and it used to
            # be worse: a `FileNotFoundError` here aborts the whole sweep, so
            # one stale path hides every group after it.  The move to `src/`
            # left 38 of them and the harness stopped running entirely.
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


GROUPS = {
    # The diagnostics themselves.  `ir_opacity` reported the whole corpus as
    # failing to generate for as long as nobody re-derived its number.
    # `flatten_scopes` decides which braces are load-bearing, and it decides
    # it with a regex over raw text.
    # The PTX node.  Its numbering check is the one that matters: a mismatch
    # reads different registers and still compiles.
    # A tile's permutation lives on the buffer so no access can forget it.
    # The bank model.  Every mistake it made over-reported, which is the
    # direction that gets a check ignored.
    # The IR-level bank analysis.  Every mistake it made over-reported, which
    # is the direction that gets a check ignored.
    'irbanks': ('tests/test_pir_banks.py', [
        ('the volume rule picks a width the pattern does not want',
         sub(Path('src/tensorforge/backend/instructions/memory/__init__.py'),
             '    while width * 2 <= self._BANKS and volume % (width * 2) == 0:',
             '    while width * 2 <= 8 and volume % (width * 2) == 0:', 1)),
        ('the recommender scores every candidate the same',
         sub(Path('src/tensorforge/backend/pir/banks.py'),
             '                if candidate > 1:',
             '                if False:', 1)),
        ('a dead access counted as one the hardware makes',
         sub(Path('src/tensorforge/backend/pir/passes.py'),
             'def optimize(body: Tuple[Stmt, ...], dump_hook=None,',
             'def optimize(body, *_a, **_k):\n    return body\n\n\n'
             'def _optimize(body: Tuple[Stmt, ...], dump_hook=None,', 1)),
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

    'dryrun': ('tests/test_tools.py::test_every_mutation_still_applies', [
        ('a stale anchor stops being reported',
         sub(Path('tools/mutation_check.py'),
             "                print(f'  {group}: {name}: {exc}')\n                stale += 1",
             "                pass", 1)),
    ]),

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
             "hint='ctile', swizzle=XorSwizzle(threads))", "hint='ctile')", 1)),
        ('the C tile takes the B tile\'s width',
         sub(Path('src/tensorforge/backend/instructions/compute/primitives/nvidia.py'),
             "hint='ctile', swizzle=XorSwizzle(threads))",
             "hint='ctile', swizzle=XorSwizzle(atom.k))", 1)),
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
             '    while width * 2 <= self._BANKS and volume % (width * 2) == 0:',
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
             '        indices = tuple(self._swizzled(base, i) for i in indices)\n'
             '        if uniform is None:',
             '        if uniform is None:', 1)),
        ('the store stops applying it',
         sub(Path('src/tensorforge/backend/pir/build.py'),
             '        indices = tuple(self._swizzled(base, i) for i in indices)\n'
             '        kind = Effect.ATOMIC if atomic else Effect.WRITE',
             '        kind = Effect.ATOMIC if atomic else Effect.WRITE', 1)),
        ('the permutation stops being a bijection',
         sub(Path('src/tensorforge/backend/pir/core.py'),
             'return index ^ ((index // self.width) % self.width)',
             'return index ^ (index % self.width)', 1)),
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
             "hint='btile', swizzle=XorSwizzle(atom.k))",
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

    'cdecl': ('tests/test_flatten_scopes.py', [
        ('brace initialisation not seen as a declaration',
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
        ('fmacdpp8 re-enabled without a runtime',
         sub(PKG / 'caps.py',
             '    return False\n\n\ndef has_fmacdpp16',
             '    return True\n\n\ndef has_fmacdpp16')),
        ('codegen widened past the header (gfx908)',
         sub(PKG / 'caps.py',
             'return arch in (0x90a, 0x940, 0x941, 0x942, 0x950) or arch >= 0x1000',
             'return arch in (0x908, 0x90a, 0x940, 0x941, 0x942, 0x950) or arch >= 0x1000')),
        ('header narrowed, codegen not updated',
         sub(HIP, '#if defined(__gfx90a__) || defined(__gfx940__)',
             '#if defined(__gfx940__)', 1)),
    ]),

    # The boundary between the two spans, which `plan` states once.  Stating
    # it twice is what let the spans overlap, so the first two mutations put
    # the boundary back where each half of that mistake had it.
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
             '        return (frozenset({Strategy.MATRIX})',
             '    if False:\n'
             '        return (frozenset({Strategy.MATRIX})')),
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
             '    return layout if layout.tiles(self.num_threads) else None',
             '    return layout')),
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
        ('blocks that leave the lanes holding copies admitted',
         sub(Path('src/tensorforge/backend/temporaries.py'),
             '        if self._num_threads and product != self._num_threads:',
             '        if False:')),
        ('a packed image on two axes given an owner anyway',
         sub(Path('src/tensorforge/backend/symbol.py'),
             '    if self.lead_width > 1 and len(self.lead_dims) > 1:\n'
             '      # One width',
             '    if False:\n      # One width')),
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
        ('the buffer sized by something other than the plan',
         sub(Path('src/tensorforge/backend/instructions/compute/staging.py'),
             '    addresses = {transfer.address for transfer in plan}\n'
             '    return len(addresses)',
             '    return 0')),
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
             '            key = tuple((t // a.stride) % a.block for a in self.axes)',
             '            key = tuple(t % a.block for a in self.axes)')),
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

    'scratch': ('tests/test_pir_scratch.py', [
        ('a shared alloc declares its own array again',
         sub(EMIT,
             "                w(f'{t.elem.ctype()}* {qual}{self.name(v)} = &{arena}[{off}];')",
             "                w(f'__shared__ {t.elem.ctype()} {self.name(v)}[{t.volume}];')")),
        ('the budget check dropped',
         sub(BUILD, '        if max(end, self._scratch_peak) > budget:',
             '        if False:')),
        ('windows overlap: the cursor never advances',
         sub(BUILD, '        self._scratch_used = end', '        pass')),
        ('alignment ignored',
         sub(BUILD, '        align = max(1, 16 // elem.size())', '        align = 1')),
        ('no budget read as unlimited',
         sub(BUILD, '        if self._scratch is None:', '        if False:')),
        ('the instruction hands over a budget it did not declare',
         sub(ABSTR, "scratch=(('tempShrMem', budget) if budget else None))",
             "scratch=('tempShrMem', 1 << 20))")),
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
         sub(SYM, '    structured = (not atomic and isinstance(variable, _Value)',
             '    structured = (False and isinstance(variable, _Value)')),
        ('an atomic routed through Op.STORE, losing the atomicity',
         sub(SYM, '    structured = (not atomic and isinstance(variable, _Value)',
             '    structured = (isinstance(variable, _Value)')),
        ('the address pinned again, so nothing folds or shares',
         sub(SYM, '    return self.build_address(writer, context, index)\n\n  def access_address',
             '    return writer.pin(self.build_address(writer, context, index))\n\n  def access_address')),
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
    # The last three could not be expressed against the old
    # `test_signatures.py`: it lifted reference-taking calls out with a regex
    # and looked at those alone, so a defect anywhere else in the kernel --- or
    # in an argument it could not type --- was outside what it could see.
    'syntax': ('tests/test_syntax.py', [
        ('a literal handed to a reference parameter',
         sub(Path('tests/snapshots/gemm_56x18_x_18x18.hip.cpp'),
             'tensorforge::transpose4x4b32(v55_tp, v56_tp, v57_tp, v58_tp,',
             'tensorforge::transpose4x4b32(v55_tp, v56_tp, 0.0f, 0.0f,', 1)),
        ('an argument dropped from a transpose',
         sub(Path('tests/snapshots/gemm_square_16.hip.cpp'),
             ', v41_data, v42_data, v43_data, v44_data);',
             ', v41_data, v42_data, v43_data);', 1)),
        ('an operand that is never declared',
         sub(Path('tests/snapshots/gemm_square_16.hip.cpp'),
             ', v41_data, v42_data, v43_data, v44_data);',
             ', v41_data, v42_data, v43_data, v44_undeclared);', 1)),
        ('an MFMA accumulator of the wrong width',
         sub(Path('tests/snapshots/gemm_square_16.hip.cpp'),
             'tensorforge::VectorT<float, 4>',
             'tensorforge::VectorT<float, 2>', 1)),
        ('a store past the end of a shared-memory declaration',
         sub(Path('tests/snapshots/gemm_square_16.hip.cpp'),
             'const auto batchId_start',
             'const auto batchId_start = undeclared_symbol; const auto _unused',
             1)),
    ]),

    # The shim is a copy of a C++ fact; the check that it stays one has to
    # fail when the copy drifts, in either direction.
    # One list, two readers.  The tool reported three permanent failures for
    # cases the suite already tracked, which is how a check stops being read.
    'knownbad': ('tests/test_tools.py::test_the_runner_agrees_with_the_suite', [
        ('the tool stops recognising a tracked failure',
         sub(Path('tools/syntax_check.py'),
             '    reason = syntax.known_bad(r.path)',
             "    reason = ''", 1)),
        ('a tracked entry counted as well-formed',
         sub(Path('tools/syntax_check.py'),
             "        kinds['known'] += 1",
             "        kinds['ok'] += 1", 1)),
    ]),

    'shim': ('tests/test_syntax.py::test_shim_matches_the_device_headers', [
        ('an overload dropped from the shim',
         sub(Path('tests/shim/tensorforge_host.h'),
             'template <int Row> void fmacdpp16(double &c, double a, double b);\n',
             '', 1)),
        ('the shim made more permissive than the header',
         sub(Path('tests/shim/tensorforge_host.h'),
             'template <int Row> float movdpp16(float a);',
             'template <int Row, typename T> T movdpp16(T a);', 1)),
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
             "                           layout=self.layout,",
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

    # The contract on `Tensor.data`, and the two test patterns that the PIR
    # refactor made stale without making anything fail loudly enough.
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

    # The nvidia path is live now; the guard that keeps it from silently
    # going dead again, and the gate that keeps it from aborting cases it
    # cannot take.
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
        ('the reservation sized for one candidate instead of all of them',
         sub(Path('src/tensorforge/backend/instructions/compute/primitives/nvidia.py'),
             '    return max((size(atom) for atom in instrs_for(dtype, sm)), '
             'default=0)',
             '    return size(instrs_for(dtype, sm)[0])', 1)),
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
             '                                            got = A(writer, None, i // threads,\n'
             '                                                    k + kk + kkk, parts=aparts)',
             "                                            got = writer.varalloc()\n"
             "                                            A(writer, f'{got}', i // threads,\n"
             "                                              k + kk + kkk, parts=aparts)", 1)),
        ('a padding fragment declared as text again',
         sub(Path('src/tensorforge/backend/instructions/compute/primitives/nvidia.py'),
             "                                                AregParts[pt][kkk] = writer.declare(\n"
             "                                                    ScalarType(atom.d), hint='as')",
             "                                                AregParts[pt][kkk] = writer.varalloc()\n"
             "                                                writer(f'float {AregParts[pt][kkk]}{{}};',\n"
             "                                                       accesses=())", 1)),
        ('the wave width no longer checked',
         sub(Path('src/tensorforge/backend/instructions/compute/primitives/nvidia.py'),
             '    return (threads == 32 and dtype in (Datatype.F32, '
             'Datatype.F64)',
             '    return (dtype in (Datatype.F32, Datatype.F64)', 1)),
        ('the operand type no longer checked',
         sub(Path('src/tensorforge/backend/instructions/compute/primitives/nvidia.py'),
             '    return (threads == 32 and dtype in (Datatype.F32, '
             'Datatype.F64)\n            and not sparse',
             '    return (threads == 32\n            and not sparse', 1)),
        ('the gate bypassed entirely',
         sub(Path('src/tensorforge/backend/instructions/compute/primitives/nvidia.py'),
             '    if (ENABLED\n'
             '            and supports(shape.threads, shape.accumulator, '
             'shape.sparse,\n'
             '                         shape.depth)\n'
             '            and instrs_for(shape.accumulator, sm_of(ctx))):',
             '    if ENABLED:', 1)),
        ('the arch dropped from the offer, so a target with no entry is offered one',
         sub(Path('src/tensorforge/backend/instructions/compute/primitives/nvidia.py'),
             '            and instrs_for(shape.accumulator, sm_of(ctx))):',
             '            and True):', 1)),
        ('the deployment switch flipped without re-recording',
         sub(Path('src/tensorforge/backend/instructions/compute/primitives/nvidia.py'),
             'ENABLED = False', 'ENABLED = True', 1)),
    ]),

    # A raw statement may narrow what it touches, and must then be complete
    # about what it uses.  Both halves, plus the scope that carries the
    # lifetime a liveness analysis cannot yet see.
    'rawaccess': ('tests/test_pir_raw_accesses.py', [
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
        ('the scope no longer releases',
         sub(Path('src/tensorforge/backend/pir/build.py'),
             '            self._scratch_used = mark',
             '            pass', 1)),
        ('a varalloc name asked to claim a use it cannot have',
         sub(Path('src/tensorforge/backend/pir/build.py'),
             '        self._by_name.pop(str(v), None)\n', '', 1)),
        ('defines accepted but discarded',
         sub(Path('src/tensorforge/backend/pir/build.py'),
             'return self._emit_op(Op.RAWSTMT, tuple(defines), tuple(args),',
             'return self._emit_op(Op.RAWSTMT, (), tuple(args),', 1)),
        ('the budget checked against the mark, not the peak',
         sub(Path('src/tensorforge/backend/pir/build.py'),
             'if max(end, self._scratch_peak) > budget:',
             'if end > budget:', 1)),
    ]),

    # `scratch_scope` declares a packing; this is the check that it holds.
    'scratchcheck': ('tests/test_scratch_check.py', [
        ('a read across a reused window no longer reported',
         sub(Path('src/tensorforge/backend/pir/scratch_check.py'),
             '                if t.reads and last_write[other] is not None:',
             '                if False:', 1)),
        ('a rewrite between the clobber and the read not noticed',
         sub(Path('src/tensorforge/backend/pir/scratch_check.py'),
             '                    if mine is None or mine < last_write[other]:',
             '                    if True:', 1)),
        ('windows compared without checking that they overlap',
         sub(Path('src/tensorforge/backend/pir/scratch_check.py'),
             '            if not win[a].overlaps(win[b]):\n                continue',
             '            if False:\n                continue', 1)),
        ('an undeclared statement passed over in silence',
         sub(Path('src/tensorforge/backend/pir/scratch_check.py'),
             '                opaque.append(here)', '                pass', 1)),
        ('the allocation counted as a use of its own buffer',
         sub(Path('src/tensorforge/backend/pir/scratch_check.py'),
             '        if stmt.op == Op.ALLOC:\n            continue',
             '        if False:\n            continue', 1)),
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
            ok = r.returncode != 0
            caught += ok
            print(f'  {name:56s} {"caught" if ok else "*** MISSED ***"}')
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
    still findable costs a string search.  The second is the part that rots --
    five anchors had gone stale before anyone counted -- and separating them
    is what lets a test assert freshness without taking minutes.
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
