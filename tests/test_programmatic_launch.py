# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Programmatic Dependent Launch (`Options.programmatic_launch`).

A kernel launched as the programmatic dependent of the work ahead of it on the
stream may start while that work's last blocks still run, and every thread
waits for it in `griddepcontrol.wait` (`tensorforge::gridDependencyWait`).
Ahead of the wait a kernel may read only what no kernel of the run writes: the
batch-constant operators -- and the multi-element layout's stage of them --
and the operands its caller states unchanged over the run (yateto's
`temporal: invariant`).  Anything else comes behind it, and the wait stands at
a body's level, so that every thread meets it (`pir.griddep`).

Off by default, and off it changes nothing; on, it changes nothing below
sm_90, off CUDA, or for a kernel launched cooperatively.  The host oracle runs
one grid, where the wait has nothing to wait for.
"""

from __future__ import annotations

import contextlib
import copy
import importlib.util
import io
import json
import os
import re
import shutil
import subprocess
import sys
import warnings
from pathlib import Path

import pytest

import seissol_suite
from tensorforge import toolchain
from tensorforge.backend.pir import engines, griddep
from tensorforge.backend.pir.core import Access, Effect, MemSpace, Op, Stmt
from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.context import Context
from tensorforge.common.options import Options
from tensorforge.frontend.yateto import DescriptionReader
from tensorforge.generators.generator import Generator
from tensorforge.reference import kernel_eval as ke

WAIT = 'tensorforge::gridDependencyWait();'
LAUNCH = 'tensorforge::launchProgrammatic('
CASES = Path(__file__).resolve().parent / 'cases'


def _description(kernel, invariant=()):
    d = copy.deepcopy(seissol_suite.description(
        'elastic-linearck', 'elastic-linearck-o4-d', kernel))
    for tensor in d['tensors']:
        if tensor['name'] in invariant:
            tensor['attrs'] = {'temporal': 'invariant'}
    return d


def _generate(gen):
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter('ignore')
        gen.generate()
    return gen


def _seissol(kernel, arch='sm_90', backend='cuda', invariant=(), **options):
    descrs, _ = DescriptionReader(None, {}).read(_description(kernel, invariant))
    ctx = Context(arch=arch, backend=backend, fp_type=Datatype.F64,
                  options=Options(**options))
    return _generate(Generator(descrs, ctx, attrs={}))


def _case(stem, arch='sm_90', **options):
    path = next(CASES.rglob(f'{stem}.py'))
    spec = importlib.util.spec_from_file_location('tf_pdl__' + stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ctx = Context(arch=arch, backend='cuda',
                  fp_type=getattr(mod, 'DTYPE', None),
                  options=Options(**options))
    return _generate(Generator(mod.descr_list(), ctx,
                               attrs=getattr(mod, 'ATTRS', None)))


def _lines(gen):
    return gen.get_kernel().split('\n')


def _unnamed(text):
    """`text` without what a build with another option set differs in
    whatever it does: the options line, and the names digested from the
    source."""
    text = '\n'.join(line for line in text.split('\n') if '// options:' not in line)
    return re.sub(r'kernel_[0-9a-f]{16}', 'kernel_', text)


def _wait_line(gen) -> int:
    lines = _lines(gen)
    waits = [i for i, line in enumerate(lines) if WAIT in line]
    assert len(waits) == 1, f'{len(waits)} waits in the kernel'
    return waits[0]


def _operands(gen):
    """The kernel's operands by name: alias and addressing, from its meta
    line."""
    meta = next(line for line in _lines(gen) if 'tensorforge-meta:' in line)
    blob = json.loads(meta.split('tensorforge-meta:', 1)[1])
    return {o['name']: (o['alias'], o['addressing']) for o in blob['operands']}


def _first_use(gen, names) -> int:
    """The first line of the kernel's body that names one of the operands
    `names` as an array or pointer (`m3[`, `glb_m3[`, `peel_glb_m3`, ...)."""
    lines = _lines(gen)
    start = next(i for i, line in enumerate(lines) if 'tensorforge-meta:' in line)
    pattern = re.compile(r'\b(?:\w+_)?(' + '|'.join(map(re.escape, names))
                         + r')\b(?:\[|\s*\+|\))')
    for i in range(start + 1, len(lines)):
        if pattern.search(lines[i]) and not lines[i].lstrip().startswith('//'):
            return i
    return len(lines)


def _per_element(gen, alive=()):
    """The per-element operands, but for those named in `alive`."""
    return [name for name, (alias, addressing) in _operands(gen).items()
            if addressing not in ('none', 'scalar') and alias not in alive]


class TestTheOption:
    def test_off_builds_what_it_built(self):
        for kernel in ('gpu_localFlux[0]', 'gpu_volume'):
            plain = _seissol(kernel)
            off = _seissol(kernel, programmatic_launch=False)
            assert plain.get_kernel() == off.get_kernel()
            assert plain.get_launcher() == off.get_launcher()
            assert WAIT not in plain.get_kernel()
            assert LAUNCH not in plain.get_launcher()

    def test_on_launches_programmatically_and_waits_once(self):
        gen = _seissol('gpu_localFlux[0]', programmatic_launch=True)
        assert LAUNCH in gen.get_launcher()
        assert '<<<' not in gen.get_launcher()
        _wait_line(gen)

    def test_not_below_sm90(self):
        gen = _seissol('gpu_localFlux[0]', arch='sm_80', programmatic_launch=True)
        assert WAIT not in gen.get_kernel()
        assert LAUNCH not in gen.get_launcher()
        plain = _seissol('gpu_localFlux[0]', arch='sm_80')
        assert _unnamed(gen.get_kernel()) == _unnamed(plain.get_kernel())
        assert _unnamed(gen.get_launcher()) == _unnamed(plain.get_launcher())

    def test_not_off_cuda(self):
        on = _seissol('gpu_localFlux[0]', arch='gfx942', backend='hip',
                      programmatic_launch=True)
        off = _seissol('gpu_localFlux[0]', arch='gfx942', backend='hip')
        assert WAIT not in on.get_kernel()
        assert _unnamed(on.get_kernel()) == _unnamed(off.get_kernel())
        assert _unnamed(on.get_launcher()) == _unnamed(off.get_launcher())

    def test_not_off_cuda_on_an_nvidia_part_either(self):
        # HIP and SYCL build for NVIDIA parts as well; neither launches so.
        for backend in ('hip', 'acpp'):
            on = _seissol('gpu_localFlux[0]', arch='sm_90', backend=backend,
                          programmatic_launch=True)
            off = _seissol('gpu_localFlux[0]', arch='sm_90', backend=backend)
            assert WAIT not in on.get_kernel()
            assert _unnamed(on.get_kernel()) == _unnamed(off.get_kernel())
            assert _unnamed(on.get_launcher()) == _unnamed(off.get_launcher())

    def test_a_cooperative_launch_stays_one(self):
        # A grid barrier needs every block resident at once, which the
        # cooperative launch is for, and nothing would gain by the other.
        gen = _case('barrier_two_gemms', programmatic_launch=True,
                    cooperative_grid=True)
        assert 'cudaLaunchCooperativeKernel' in gen.get_launcher()
        assert LAUNCH not in gen.get_launcher()
        assert WAIT not in gen.get_kernel()


class TestWhereTheKernelWaits:
    def test_ahead_of_every_per_element_operand(self):
        for kernel in ('gpu_localFlux[0]', 'gpu_volume', 'gpu_neighboringFlux[0]',
                       'gpu_derivative'):
            gen = _seissol(kernel, programmatic_launch=True)
            assert _wait_line(gen) < _first_use(gen, _per_element(gen)), kernel

    def test_the_operators_staged_ahead_of_it(self):
        gen = _seissol('gpu_volume', programmatic_launch=True,
                       multi_element='force')
        lines = _lines(gen)
        wait = _wait_line(gen)
        stage = [i for i, line in enumerate(lines) if re.search(r'bkDivM\d+\[', line)
                 and '=' in line.split('[', 1)[0] + '=']
        assert stage and max(i for i in stage if i < wait) < wait
        loop = next(i for i, line in enumerate(lines) if 'for (size_t batchIdGroup' in line)
        assert wait < loop
        assert wait < _first_use(gen, _per_element(gen))

    def test_invariant_operands_ahead_of_it(self):
        stated = _seissol('gpu_localFlux[0]', invariant=('I', 'AplusT'),
                          programmatic_launch=True)
        plain = _seissol('gpu_localFlux[0]', programmatic_launch=True)
        operands = _operands(stated)
        invariant = [n for n, (alias, _) in operands.items() if alias in ('I', 'AplusT')]
        # Stated unchanged, the first element's I and AplusT are read ahead of
        # the wait; Q still behind it.
        assert _first_use(stated, invariant) < _wait_line(stated)
        assert _wait_line(stated) < _first_use(stated, _per_element(stated, ('I', 'AplusT')))
        assert _wait_line(stated) > _wait_line(plain)

    def test_a_stated_output_is_not_free(self):
        # The kernel writes Q; whatever its caller states, it is not read
        # ahead of the wait.
        gen = _seissol('gpu_localFlux[0]', invariant=('I', 'AplusT', 'Q'),
                       programmatic_launch=True)
        q = [n for n, (alias, _) in _operands(gen).items() if alias == 'Q']
        assert _wait_line(gen) < _first_use(gen, q)

    def test_ahead_of_the_flag_words(self):
        # A frontend without attributes leaves the kernel a flag mask it may
        # pass (`FlagMode.OPTIONAL`), and the wrap pass reads the first
        # element's flag word and its successor's ahead of the batch loop --
        # as text, no access of the body's.  An earlier kernel may have
        # computed the mask.
        descrs, _ = DescriptionReader(None, {}).read(_description('gpu_localFlux[0]'))
        ctx = Context(arch='sm_90', backend='cuda', fp_type=Datatype.F64,
                      options=Options(programmatic_launch=True))
        gen = _generate(Generator(descrs, ctx, attrs=None))
        lines = _lines(gen)
        reads = [i for i, line in enumerate(lines)
                 if re.search(r'\bflags0\[', line) and 'launcher' not in line]
        assert any('flagWordFirst' in line for line in lines)
        assert reads and _wait_line(gen) < min(reads)

    def test_the_oracle_computes_the_same_behind_it(self):
        for stem in ('addressing_none', 'chain'):
            plain = _case(stem)
            waits = _case(stem, programmatic_launch=True)
            assert WAIT in waits.get_kernel()
            geometry = ke.launch_geometry(plain.get_launcher())
            assert ke.launch_geometry(waits.get_launcher()) == geometry
            lanes, mults = geometry
            got = ke.evaluate_wave(waits.get_kernel(), lanes, seed=5,
                                   globals_only=True, mults=mults, block=True)
            want = ke.evaluate_wave(plain.get_kernel(), lanes, seed=5,
                                    globals_only=True, mults=mults, block=True)
            assert got == want


# --------------------------------------------------------------------------- #
# The pass on bodies built by hand
# --------------------------------------------------------------------------- #

class _Tensor:
    def __init__(self, addressing, time_invariant=False, members=()):
        self.addressing = addressing
        self.time_invariant = time_invariant
        self.variant_members = tuple(members)


class _Base:
    def __init__(self, obj):
        self.obj = obj


def _touch(base, kind=Effect.READ, space=MemSpace.GLOBAL, op=Op.RAWSTMT):
    return Stmt(op=op, text='touch', pure=False, movable=False, effect=kind,
                accesses=(Access(kind=kind, space=space, base=base),))


def _quiet():
    return Stmt(op=Op.RAWSTMT, text='nothing', pure=False, movable=False,
                effect=Effect.NONE)


OPERATOR = _Base(_Tensor(Addressing.NONE))
ELEMENT = _Base(_Tensor(Addressing.PTR_BASED))
STATED = _Base(_Tensor(Addressing.PTR_BASED, time_invariant=True))


class TestThePass:
    def _placed(self, body, writes=frozenset()):
        out, placed = griddep.place_wait(body, writes)
        return [i for i, s in enumerate(out) if s.op == Op.GRID_WAIT], placed

    def test_behind_what_is_free_ahead_of_the_rest(self):
        body = (_quiet(), _touch(OPERATOR), _touch(STATED), _touch(ELEMENT), _quiet())
        assert self._placed(body) == ([3], True)

    def test_a_written_operator_is_not_free(self):
        body = (_touch(OPERATOR), _touch(ELEMENT))
        writes = griddep.written([(_touch(OPERATOR, kind=Effect.WRITE),)])
        assert self._placed(body, writes) == ([0], True)

    def test_a_stand_in_is_free_where_its_members_are(self):
        mixed = _Base(_Tensor(Addressing.NONE, members=(ELEMENT.obj,)))
        constant = _Base(_Tensor(Addressing.NONE, members=(OPERATOR.obj,)))
        assert self._placed((_touch(constant), _touch(mixed))) == ([1], True)

    def test_an_access_of_no_operand_or_no_space_is_not_free(self):
        assert self._placed((_touch(OPERATOR), _touch(None))) == ([1], True)
        assert self._placed((_touch(OPERATOR), _touch(ELEMENT, space=MemSpace.UNKNOWN),
                             _touch(ELEMENT))) == ([1], True)

    def test_opaque_text_is_not_free(self):
        opaque = Stmt(op=Op.RAWSTMT, text='?', pure=False, movable=False,
                      effect=Effect.UNKNOWN)
        assert self._placed((_touch(OPERATOR), opaque, _touch(ELEMENT))) == ([1], True)

    def test_text_stated_to_read_memory_is_not_free(self):
        # A flag word: pure text with no access, its read stated beside it
        # (`IRBuilder.decl_expr`, `text_reads`).
        flag = Stmt(op=Op.RAWSTMT, text='flags0[0]', pure=True, movable=True,
                    effect=Effect.NONE, attrs=(('text_reads', MemSpace.GLOBAL),))
        assert self._placed((_touch(OPERATOR), flag, _touch(ELEMENT))) == ([1], True)
        local = Stmt(op=Op.RAWSTMT, text='s[0]', pure=True, movable=True,
                     effect=Effect.NONE, attrs=(('text_reads', MemSpace.SHARED),))
        assert self._placed((local, _touch(ELEMENT))) == ([1], True)

    def test_shared_memory_is_not_another_kernel_s(self):
        body = (_touch(None, space=MemSpace.SHARED), _touch(ELEMENT))
        assert self._placed(body) == ([1], True)

    def test_none_where_nothing_touches_another_kernel_s_memory(self):
        out, placed = griddep.place_wait((_quiet(), _touch(OPERATOR)), frozenset())
        assert not placed and all(s.op != Op.GRID_WAIT for s in out)


class TestTheOp:
    def test_written_on_cuda_and_nowhere_else(self):
        wait = engines.grid_wait()
        assert engines.lower(wait, None).lines == (WAIT,)
        assert wait.effect & Effect.UNKNOWN and not wait.movable
        assert not engines.verify_engine(wait)


# --------------------------------------------------------------------------- #
# Compiled, and on the hardware
# --------------------------------------------------------------------------- #

INCLUDE = Path(toolchain.include_dir())
BENCH = Path(__file__).resolve().parents[1] / 'tools' / 'bench'

#: Runs the sequence of `_sequence` `argv[2]` times over, on a stream of its
#: own or captured into a graph and launched as one (`argv[3]`), then sums and
#: dumps its outputs (`$TFB_DUMP`, `driver_bench`).  For another workload's
#: entry points: `main_source`.
MAIN = r"""
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
extern "C" void tfb_setup_pdl(size_t);
extern "C" void tfb_launch_pdl(void*, size_t);
extern "C" void tfb_teardown_pdl();
extern "C" double tfb_checksum_pdl(size_t, double*);
#define CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  std::fprintf(stderr, "%s: %s\n", #x, cudaGetErrorString(e_)); return 1; } } while (0)
int main(int argc, char** argv) {
  if (argc < 4) return 2;
  const size_t batch = std::strtoull(argv[1], nullptr, 10);
  const int rounds = std::atoi(argv[2]);
  const bool graph = std::strcmp(argv[3], "graph") == 0;
  cudaStream_t s;
  CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
  tfb_setup_pdl(batch);
  CHECK(cudaDeviceSynchronize());
  if (graph) {
    cudaGraph_t g;
    cudaGraphExec_t e;
    CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
    for (int r = 0; r < rounds; ++r) tfb_launch_pdl((void*)s, batch);
    CHECK(cudaStreamEndCapture(s, &g));
    CHECK(cudaGraphInstantiate(&e, g, 0));
    CHECK(cudaGraphLaunch(e, s));
    CHECK(cudaStreamSynchronize(s));
    CHECK(cudaGraphExecDestroy(e));
    CHECK(cudaGraphDestroy(g));
  } else {
    for (int r = 0; r < rounds; ++r) tfb_launch_pdl((void*)s, batch);
    CHECK(cudaStreamSynchronize(s));
  }
  CHECK(cudaGetLastError());
  double magnitude = 0.0;
  const double sum = tfb_checksum_pdl(batch, &magnitude);
  std::printf("%.17g %.17g\n", sum, magnitude);
  tfb_teardown_pdl();
  CHECK(cudaStreamDestroy(s));
  return 0;
}
"""


def main_source(tag: str) -> str:
    """`MAIN` for the entry points of the bench workload `tag`
    (`driver_bench.entry_points`)."""
    return MAIN.replace('_pdl(', f'_{tag}(')


def _nvcc():
    found = toolchain.COMPILERS['cuda'].find()
    if not found or not shutil.which(found):
        pytest.skip('nvcc not found')
    return found


def _gpu():
    """The first GPU's `sm_XY`, or None."""
    tool = shutil.which('nvidia-smi')
    if tool is None:
        return None
    try:
        run = subprocess.run([tool, '--query-gpu=compute_cap',
                              '--format=csv,noheader'], capture_output=True,
                             text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    first = run.stdout.strip().splitlines()[:1]
    if run.returncode or not first:
        return None
    return 'sm_' + first[0].strip().replace('.', '')


def _sequence(arch, programmatic):
    """SeisSol's local flux through face 0 (elastic, order 3, F32) cut into
    its three multiplications, one launch each, the temporaries passed from
    one to the next through memory (`tools/bench`, partition `ops`): one
    translation unit with the bench's entry points
    (`driver_bench.emit_sequence_tu`), and its generators."""
    if str(BENCH) not in sys.path:
        sys.path.insert(0, str(BENCH))
    import build as bench_build
    import suite
    from harness import driver_bench
    workload, = [w for w in suite.from_dump(BENCH / 'data' / 'elastic-3-s-s1.json')
                 if w.name == 'gpu_localFlux[0]']
    options = Options(programmatic_launch=programmatic)
    ctx = Context(arch=arch, backend='cuda', fp_type=Datatype.F32, options=options)
    config = suite.Config('pdl', options, partition='ops')
    gens = [_generate(Generator(bench_build._read(piece), ctx, attrs=workload.attrs))
            for _, piece in bench_build.pieces(workload, config)]
    headers = list(ctx.target.headers())
    for gen in gens:
        headers += [h for h in gen.get_helper_headers() if h not in headers]
    src, _ = driver_bench.emit_sequence_tu(
        gens, 'cuda', 'pdl', '\n'.join(f'#include "{h}"' for h in headers),
        bench_build.regions_of(workload.description))
    return gens, src


def _compile(nvcc, arch, sources, out, *flags):
    cuda = toolchain.COMPILERS['cuda']
    cmd = [nvcc, *cuda.language_flags(), '-O3', '-DNDEBUG', *cuda.target_flags(arch),
           *flags, '-I', str(INCLUDE), *map(str, sources), '-o', str(out)]
    run = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    assert run.returncode == 0, run.stderr[-4000:]
    return out


class TestCompiled:
    def test_each_kernel_waits_once_and_each_launcher_launches_so(self, tmp_path):
        """nvcc for sm_90: one `griddepcontrol.wait` in each kernel's PTX, and
        the launchers call `cudaLaunchKernelEx`."""
        nvcc = _nvcc()
        gens, src = _sequence('sm_90', True)
        (tmp_path / 'seq.cu').write_text(src)
        obj = _compile(nvcc, 'sm_90', [tmp_path / 'seq.cu'], tmp_path / 'seq.o', '-c')
        tool = shutil.which('cuobjdump') or str(Path(nvcc).parent / 'cuobjdump')
        ptx = subprocess.run([tool, '-ptx', str(obj)], capture_output=True, text=True,
                             timeout=600).stdout
        entries = re.split(r'\.entry\s', ptx)[1:]
        assert entries and len(entries) == len({g.get_base_name() for g in gens})
        assert [e.count('griddepcontrol.wait') for e in entries] == [1] * len(entries)
        assert b'cudaLaunchKernelExC' in obj.read_bytes()

    def test_on_the_gpu_as_plain_launches_on_a_stream_and_in_a_graph(self, tmp_path):
        """On the GPU present, of sm_90 or newer: the sequence three times
        over -- nine launches, each reading what the one before it wrote, the
        next round's first overwriting a temporary the round before read --
        on a stream and captured into a graph, launched so and plainly: every
        output the same, to the bit."""
        arch = _gpu()
        if arch is None or int(arch[3:]) < 90:
            pytest.skip('no GPU of sm_90 or newer')
        nvcc = _nvcc()
        (tmp_path / 'main.cu').write_text(main_source('pdl'))
        aux = INCLUDE / toolchain.COMPILERS['cuda'].aux
        outputs = {}
        for programmatic in (False, True):
            gens, src = _sequence(arch, programmatic)
            assert (WAIT in gens[0].get_kernel()) == programmatic
            where = tmp_path / ('on' if programmatic else 'off')
            where.mkdir()
            (where / 'seq.cu').write_text(src)
            exe = _compile(nvcc, arch, [where / 'seq.cu', tmp_path / 'main.cu', aux],
                           where / 'seq')
            for how in ('stream', 'graph'):
                dump = where / how
                dump.mkdir()
                run = subprocess.run([str(exe), '30011', '3', how], capture_output=True,
                                     text=True, timeout=300,
                                     env=dict(os.environ, TFB_DUMP=str(dump)))
                assert run.returncode == 0, run.stderr[-2000:]
                outputs[programmatic, how] = (
                    run.stdout, {p.name: p.read_bytes() for p in dump.glob('*.bin')})
        want = outputs[False, 'stream']
        assert want[1], 'no outputs dumped'
        for key, got in outputs.items():
            assert got[1].keys() == want[1].keys(), key
            assert [n for n in want[1] if got[1][n] != want[1][n]] == [], key
            assert got[0] == want[0], key
