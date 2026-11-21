# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""A grid barrier (`GridBarrierDescr`) as a launch boundary.

Without `Options.cooperative_grid` the kernel is cut at a grid barrier into
launches: one `__global__` function each, issued by the launcher one after the
other on the stream with the same arguments, each with the grid its own element
count asks for.  The end of a launch orders its writes before the next launch
reads, which no grid has to be resident for.  With `programmatic_launch` every
launch waits for the one before ahead of its first access to anything a launch
of the kernel writes (`pir.griddep`).  The cooperative launch, one kernel with
a barrier of the whole grid between the sections, stays behind the option --
`cases/barrier/barrier_two_gemms.py` is its case.  SYCL has neither, as before.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import os
import re
import subprocess
import warnings
from pathlib import Path

import pytest

from tensorforge import toolchain
from tensorforge.common.context import Context
from tensorforge.common.options import Options
from tensorforge.generators.generator import Generator

from test_programmatic_launch import (INCLUDE, LAUNCH, WAIT, _compile, _gpu,
                                      _nvcc, main_source)

CASE = Path(__file__).resolve().parent / 'cases' / 'barrier' / 'barrier_two_gemms.py'


def _module():
    spec = importlib.util.spec_from_file_location('tf_launch_boundary_case', CASE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _barrier(arch='sm_90', backend='cuda', invariant=(), attrs=None,
             **options):
    """`barrier_two_gemms` -- D = A B, a grid barrier, E = D C -- built with
    `options` and not the case's own; `attrs` as a frontend's (None: a flag
    mask the caller may pass, `{}`: none)."""
    mod = _module()
    descrs = mod.descr_list()
    for d in descrs:
        for view in [getattr(d, 'dest', None)] + list(getattr(d, 'ops', []) or []):
            tensor = getattr(view, 'tensor', None)
            if tensor is not None and tensor.alias in invariant:
                tensor.attrs = dict(tensor.attrs or {}, temporal='invariant')
    ctx = Context(arch=arch, backend=backend, fp_type=mod.DTYPE,
                  options=Options(**options))
    gen = Generator(descrs, ctx, attrs=attrs)
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter('ignore')
        gen.generate()
    return gen


def _function(text, head):
    """The body of the function whose definition starts with `head`."""
    start = text.index(head)
    open_ = text.index('{', start)
    depth = 0
    for i in range(open_, len(text)):
        depth += {'{': 1, '}': -1}.get(text[i], 0)
        if depth == 0:
            return text[open_:i + 1]
    raise AssertionError(f'{head}: no end')


def _unnamed(text):
    return re.sub(r'kernel_[0-9a-f]{16}', 'kernel_', text)


class TestTheCut:
    def test_two_kernels_launched_one_after_the_other(self):
        gen = _barrier()
        name = gen.get_base_name()
        kernel, launcher = gen.get_kernel(), gen.get_launcher()
        assert kernel.count('__global__') == 2
        assert f'kernel_{name}(' in kernel and f'kernel_{name}_launch1(' in kernel
        assert 'this_grid' not in kernel
        assert 'Cooperative' not in launcher
        first = launcher.index(f'kernel_{name}<<<')
        second = launcher.index(f'kernel_{name}_launch1<<<')
        assert first < second
        assert not gen.launch_config().cooperative
        assert gen.kernel_info()['launches'] == [[0], [1]]

    def test_each_launch_takes_the_grid_of_its_own_count(self):
        gen = _barrier()
        name = gen.get_base_name()
        launcher = gen.get_launcher()
        grids = []
        for suffix in ('', '_launch1'):
            body = _function(launcher,
                             f'tensorforge::LaunchConfig launch_config_{name}{suffix}(')
            grids.append(next(line for line in body.split('\n')
                              if 'config.grid[0] =' in line))
        assert 'numElements0' in grids[0] and 'numElements1' not in grids[0]
        assert 'numElements1' in grids[1] and 'numElements0' not in grids[1]

    def test_the_caller_sees_the_cooperative_kernel_s_interface(self):
        cut, coop = _barrier(), _barrier(cooperative_grid=True)
        proto = lambda g: _unnamed(g.get_header().split('void launcher_', 1)[1])
        assert proto(cut) == proto(coop)

    def test_the_sections_after_the_cut_start_unrotated(self):
        kernel = _barrier().get_kernel()
        second = kernel[kernel.index('_launch1('):]
        assert 'numElements0' not in re.sub(r'size_t numElements0', '', second.split('{', 1)[1])

    def test_hip_cuts_as_well(self):
        gen = _barrier(arch='gfx942', backend='hip')
        assert gen.get_kernel().count('__global__') == 2
        assert 'Cooperative' not in gen.get_launcher()

    def test_one_section_one_launch_as_before(self):
        # A kernel without a grid barrier carries no launch of its own.
        mod = _module()
        descrs = mod.descr_list()[:1]
        ctx = Context(arch='sm_90', backend='cuda', fp_type=mod.DTYPE,
                      options=Options())
        gen = Generator(descrs, ctx)
        with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
            warnings.simplefilter('ignore')
            gen.generate()
        assert gen.get_kernel().count('__global__') == 1
        assert '_launch1' not in gen.get_kernel() + gen.get_launcher()
        assert 'launches' not in gen.kernel_info()


class TestTheCooperativeLaunch:
    def test_behind_the_option(self):
        gen = _barrier(cooperative_grid=True)
        kernel, launcher = gen.get_kernel(), gen.get_launcher()
        assert kernel.count('__global__') == 1
        assert 'cooperative_groups::this_grid().sync();' in kernel
        assert 'cudaLaunchCooperativeKernel' in launcher
        assert gen.launch_config().cooperative
        assert 'launches' not in gen.kernel_info()

    def test_the_case_asks_for_it(self):
        assert _module().OPTIONS.get('cooperative_grid') is True


class TestProgrammatic:
    def test_each_launch_waits_and_launches_so(self):
        gen = _barrier(programmatic_launch=True)
        kernel, launcher = gen.get_kernel(), gen.get_launcher()
        name = gen.get_base_name()
        first, second = kernel.split(f'kernel_{name}_launch1(')
        assert first.count(WAIT) == 1 and second.count(WAIT) == 1
        assert launcher.count(LAUNCH) == 2

    def test_what_a_launch_writes_is_not_free_in_the_next(self):
        # D is written by the first launch and read by the second: stated
        # unchanged over the run, it is still read behind the second's wait.
        # C, the second launch's other input, stated so as well, is free,
        # and without a flag mask nothing else is read ahead: the wait then
        # stands where the first read of D needs it.
        gen = _barrier(programmatic_launch=True, invariant=('D', 'C'), attrs={})
        operands = {o['alias']: o['name'] for o in gen.kernel_info()['operands']}
        kernel = gen.get_kernel()
        second = kernel.split(f'kernel_{gen.get_base_name()}_launch1(', 1)[1]
        body = second.split('{', 1)[1]
        wait = body.index(WAIT)
        read = re.search(rf'\b{operands["D"]}\[', body)
        assert read is not None and wait < read.start()

    def test_the_cooperative_launch_stays_one(self):
        gen = _barrier(programmatic_launch=True, cooperative_grid=True)
        assert WAIT not in gen.get_kernel()
        assert LAUNCH not in gen.get_launcher()


# --------------------------------------------------------------------------- #
# On the hardware
# --------------------------------------------------------------------------- #

def _source(gen, tag):
    from harness import driver_bench
    ctx = gen._context
    headers = list(ctx.target.headers()) + list(gen.get_helper_headers())
    return driver_bench.emit_workload_tu(
        gen, 'cuda', tag, '\n'.join(f'#include "{h}"' for h in headers))


def test_gpu_two_launches_compute_what_the_cooperative_one_does(tmp_path):
    """On the GPU present: the barrier case at 50000 elements -- more than the
    grid holds, so that a block makes several trips -- three times over, as
    two launches and as one cooperative launch, on a stream; the two launches
    captured into a graph as well, and with programmatic launches from sm_90:
    D and E the same, to the bit."""
    arch = _gpu()
    if arch is None:
        pytest.skip('no GPU')
    nvcc = _nvcc()
    builds = {'cooperative': dict(cooperative_grid=True), 'launches': {}}
    if int(arch[3:]) >= 90:
        builds['programmatic'] = dict(programmatic_launch=True)
    aux = INCLUDE / toolchain.COMPILERS['cuda'].aux
    (tmp_path / 'main.cu').write_text(main_source('b2g'))
    outputs = {}
    for label, options in builds.items():
        gen = _barrier(arch=arch, **options)
        assert (gen.get_kernel().count('__global__') == 1) == (label == 'cooperative')
        where = tmp_path / label
        where.mkdir()
        (where / 'b2g.cu').write_text(_source(gen, 'b2g'))
        exe = _compile(nvcc, arch, [where / 'b2g.cu', tmp_path / 'main.cu', aux],
                       where / 'b2g')
        for how in ('stream',) if label == 'cooperative' else ('stream', 'graph'):
            dump = where / how
            dump.mkdir()
            run = subprocess.run([str(exe), '50000', '3', how], capture_output=True,
                                 text=True, timeout=300,
                                 env=dict(os.environ, TFB_DUMP=str(dump)))
            assert run.returncode == 0, run.stderr[-2000:]
            outputs[label, how] = (run.stdout,
                                   {p.name: p.read_bytes() for p in dump.glob('*.bin')})
    want = outputs['cooperative', 'stream']
    assert len(want[1]) == 2, sorted(want[1])
    for key, got in outputs.items():
        assert got[1].keys() == want[1].keys(), key
        assert [n for n in want[1] if got[1][n] != want[1][n]] == [], key
        assert got[0] == want[0], key
