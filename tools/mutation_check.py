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

Source files are edited in place and restored in a `finally`.  Run it on a
clean tree.

    python3 tools/mutation_check.py            # all groups
    python3 tools/mutation_check.py layout     # one group
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

PKG = Path('tensorforge/backend/instructions/compute/primitives/amd')
CORE = Path('tensorforge/backend/pir/core.py')
BUILD = Path('tensorforge/backend/pir/build.py')
SYM = Path('tensorforge/backend/symbol.py')
HIP = Path('tensorforge/include/tensorforge_device/hip.h')


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
        text = path.read_text()
        out = text.replace(old, new) if not count else text.replace(old, new, count)
        if out == text:
            raise AssertionError(
                f'mutation did not apply to {path}: the code has moved, so '
                f'this check is no longer testing anything')
        return path, out
    return make


GROUPS = {
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

    'tiling': ('tests/test_amd_tiling.py', [
        ('the original tail overlap',
         sub(PKG / 'codegen.py',
             '        tail = ((N // tile.block) * tile.block) if cap else N',
             '        tail = (N // tile.block) * tile.block')),
        ('over-corrected: the tail dropped entirely',
         sub(PKG / 'codegen.py',
             '        tail = ((N // tile.block) * tile.block) if cap else N',
             '        tail = N')),
        ('cap policy inverted',
         sub(PKG / 'codegen.py', '        cap = N % tile.block < 2',
             '        cap = N % tile.block >= 2')),
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
             'return (threads // self.block).bit_length() - 1',
             'return (threads // self.block).bit_length()')),
        ('the fits() check dropped',
         sub(PKG / 'catalog.py',
             '        if not self.fits(threads):\n            return False\n        return self.transpose is None',
             '        return self.transpose is None')),
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
             'tensorforge::transpose4x4b32(v31_tp, v32_tp, v33_tp, v34_tp,',
             'tensorforge::transpose4x4b32(v31_tp, v32_tp, 0.0f, 0.0f,', 1)),
        ('an argument dropped from a transpose',
         sub(Path('tests/snapshots/gemm_square_16.hip.cpp'),
             ', v14_data, v15_data, v16_data, v17_data);',
             ', v14_data, v15_data, v16_data);', 1)),
        ('an operand that is never declared',
         sub(Path('tests/snapshots/gemm_square_16.hip.cpp'),
             ', v14_data, v15_data, v16_data, v17_data);',
             ', v14_data, v15_data, v16_data, v17_undeclared);', 1)),
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
        ('the fill records nothing',
         sub(Path('tensorforge/backend/symbol.py'),
             '    self._record_linear_layout(index, vec)\n', '', 1)),
        ('the read drops what the fill recorded',
         sub(Path('tensorforge/backend/symbol.py'),
             "        return writer.load_expr(text, type_, self, hint='lin',\n"
             "                                layout=self.layout)",
             "        return writer.load_expr(text, type_, self, hint='lin')", 1)),
        ('the wave width taken as the block instead of the thread count',
         sub(Path('tensorforge/backend/symbol.py'),
             'layout = RegisterLayout((LaneAxis(self.num_threads, 1),))',
             'layout = RegisterLayout((LaneAxis(self.num_threads, 2),))', 1)),
        ('a mid-slot fill claimed anyway',
         sub(Path('tensorforge/backend/symbol.py'),
             'if not isinstance(index, int) or index % (self.num_threads * vec) != 0:',
             'if not isinstance(index, int):', 1)),
        ('two disagreeing fills, last one wins',
         sub(Path('tensorforge/backend/symbol.py'),
             '      self.layout = None\n      return\n    self.layout = layout',
             '      pass\n    self.layout = layout', 1)),
        ('the layout lost on clone',
         sub(Path('tensorforge/backend/symbol.py'),
             '    cloned.layout = self.layout\n', '', 1)),
    ]),

    # The contract on `Tensor.data`, and the two test patterns that the PIR
    # refactor made stale without making anything fail loudly enough.
    'data': ('tests/test_regressions.py', [
        ('a list handed to Tensor.data',
         sub(Path('tensorforge/generators/descriptions.py'),
             'data=(np.array(alpha, dtype=float)',
             'data=([alpha]', 1)),
        ('the shape check dropped',
         sub(Path('tensorforge/common/matrix/tensor.py'),
             '            if self.data.shape != self.shape:',
             '            if False:', 1)),
        ('the type check dropped',
         sub(Path('tensorforge/common/matrix/tensor.py'),
             '            if not isinstance(self.data, np.ndarray):',
             '            if False:', 1)),
    ]),

    # The nvidia path is live now; the guard that keeps it from silently
    # going dead again, and the gate that keeps it from aborting cases it
    # cannot take.
    'nvidia': ('tests/test_nvidia_reachability.py', [
        ('a second definition of matmul',
         sub(Path('tensorforge/backend/instructions/compute/primitives/nvidia.py'),
             'def matmul(writer, C, A, B, M, N, K, kx, threads, dtype, sparse, ctx, shmptr, shmsize):',
             'def matmul(*args, **kwargs):\n    pass\n\n'
             'def matmul(writer, C, A, B, M, N, K, kx, threads, dtype, sparse, ctx, shmptr, shmsize):',
             1)),
        ('an unreachable helper reintroduced',
         sub(Path('tensorforge/backend/instructions/compute/primitives/nvidia.py'),
             'def tfconvert(writer: Writer, variables):',
             'def shuffle_swap(writer, v):\n'
             '    return f"__shfl_xor_sync(0xffffffff, {v}, 1)"\n\n'
             'def tfconvert(writer: Writer, variables):', 1)),
        ('the atom spelled out a second time',
         sub(Path('tensorforge/backend/instructions/compute/primitives/nvidia.py'),
             'def shmsize(stages):\n    atom = ATOM',
             'def shmsize(stages):\n    atom = INSTRS[1]', 1)),
    ]),

    'gate': ('tests/test_nvidia_gate.py', [
        ('the wave width no longer checked',
         sub(Path('tensorforge/backend/instructions/compute/primitives/nvidia.py'),
             'return threads == 32 and dtype == ATOM.d and not sparse',
             'return dtype == ATOM.d and not sparse', 1)),
        ('the operand type no longer checked',
         sub(Path('tensorforge/backend/instructions/compute/primitives/nvidia.py'),
             'return threads == 32 and dtype == ATOM.d and not sparse',
             'return threads == 32 and not sparse', 1)),
        ('the gate bypassed entirely',
         sub(Path('tensorforge/backend/instructions/compute/multilinear.py'),
             '            return nvidia.supports(self._num_threads, self._idest.datatype,\n'
             '                                   self._second_operand_is_sparse())',
             '            return True', 1)),
    ]),

    'operands': ('tests/test_snapshots.py', [
        ('hfma asks for the wrong distribution',
         sub(PKG / 'relayout.py',
             '    return RegisterLayout((LaneAxis(step, 1),))\n',
             '    return RegisterLayout((LaneAxis(max(step // 2, 1), 1),))\n', 1)),
        ('the transpose skipped, MFMA fed a raw load',
         sub(PKG / 'codegen.py',
             '                        tA[k // threads] = transpose(regs)',
             '                        tA[k // threads] = list(regs)')),
        ('the broadcast lane taken from the table, not the algorithm',
         sub(PKG / 'codegen.py',
             '                    params = dict(params, lane=i // step)',
             '                    params = dict(params, lane=0)')),
    ]),
}


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
            path.write_text(text)
            r = _run_tests(target)
            ok = r.returncode != 0
            caught += ok
            print(f'  {name:56s} {"caught" if ok else "*** MISSED ***"}')
            for p, t in originals.items():
                p.write_text(t)
            originals.clear()
    finally:
        for p, t in originals.items():
            p.write_text(t)
    return caught, len(mutations)


def main():
    wanted = sys.argv[1:] or list(GROUPS)
    total = hit = 0
    for group in wanted:
        if group not in GROUPS:
            print(f'unknown group {group!r}; have: {", ".join(GROUPS)}')
            return 2
        target, mutations = GROUPS[group]
        c, n = run(group, target, mutations)
        hit += c
        total += n
    print(f'\n{hit}/{total} mutations caught')
    return 0 if hit == total else 1


if __name__ == '__main__':
    sys.exit(main())
