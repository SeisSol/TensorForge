# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""What `tools/bench/build.py` hands back for workloads it does not build.

No toolchain needed: the compiler lookup is stubbed out, or the binary is
a cache hit that is never run.  What is checked is bookkeeping -- a workload
that is not in the binary comes back with a reason, because `run.py` takes a
record without one for a measurement.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parent.parent / "tools"
for extra in (str(TOOLS / "bench"), str(TOOLS)):
    if extra not in sys.path:
        sys.path.insert(0, extra)

import build as bench_build  # noqa: E402
from suite import BuildUnit, Config, TargetSpec, Workload  # noqa: E402
from tensorforge.common.basic_types import Datatype  # noqa: E402


def _case(name):
    path = Path(__file__).parent / 'cases' / f'{name}.py'
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.descr_list


def _unit(*workloads):
    return BuildUnit(suite='t', target=TargetSpec('cuda', 'sm_90'),
                     datatype=Datatype.F32, config=Config('default'),
                     workloads=tuple(Workload(n, _case(c), Datatype.F32)
                                     for n, c in workloads),
                     batches=(4,))


def test_the_same_kernel_twice_is_built_once(monkeypatch):
    """Two workloads, one kernel: the same symbol, which one binary cannot
    define twice -- the second is refused and says whose number it has."""
    monkeypatch.setattr(bench_build, 'compiler_binary', lambda compiler: None)
    built = bench_build.build(_unit(('a', 'square_notrans'),
                                    ('b', 'square_notrans'),
                                    ('c', 'csa_alpha')))
    a, b, c = built.workloads
    assert a.symbol == b.symbol != c.symbol
    assert not a.error and not c.error
    assert b.error == 'same kernel as a'


def test_a_cache_hit_refuses_what_did_not_compile(monkeypatch, tmp_path):
    """A cached binary without one workload's remarks did not compile that
    workload, and the record says so instead of coming back empty."""
    monkeypatch.setattr(bench_build, 'compiler_binary', lambda compiler: 'cc')
    monkeypatch.setattr(bench_build, '_digest', lambda *args: 'hit')
    unit = _unit(('a', 'square_notrans'), ('c', 'csa_alpha'))
    out = tmp_path / unit.label / 'hit'
    out.mkdir(parents=True)
    (out / 'bench').write_text('')
    (out / 'a.remarks').write_text('')

    built = bench_build.build(unit, cache=tmp_path)
    a, c = built.workloads
    assert built.exe == out / 'bench'
    assert a.ok
    assert not c.ok and c.error.startswith('did not compile; see ')


def test_what_a_workload_cost_to_make_is_recorded(monkeypatch, tmp_path):
    """Generation is timed and its code sized against the target's
    instruction cache; compiling is timed per translation unit, and a cache
    hit, which compiles nothing, reads back what the compilation cost."""
    import subprocess

    monkeypatch.setattr(bench_build, 'compiler_binary', lambda compiler: 'cc')
    monkeypatch.setattr(bench_build, '_digest', lambda *args: 'fresh')

    def compile_one(cc, compiler, arch, src, obj):
        obj.write_bytes(b'x' * 1234)
        return True, ''

    def link(cmd, **kwargs):
        Path(cmd[-1]).write_text('')
        return subprocess.CompletedProcess(cmd, 0, '', '')

    monkeypatch.setattr(bench_build, '_compile_one', compile_one)
    monkeypatch.setattr(bench_build.subprocess, 'run', link)
    unit = _unit(('a', 'square_notrans'))
    built = bench_build.build(unit, cache=tmp_path)
    a, = built.workloads
    assert a.ok and a.generate_s > 0 and a.compile_s >= 0
    assert a.object_bytes == 1234
    assert a.code_bytes > 0 and a.icache_bytes == 128 * 1024
    assert set(a.cost_fields()) == {'generate_s', 'compile_s',
                                    'object_bytes', 'code_bytes',
                                    'icache_bytes'}

    again, = bench_build.build(unit, cache=tmp_path).workloads
    assert again.compile_s == pytest.approx(a.compile_s)
    assert again.object_bytes == 1234


def test_a_workload_that_does_not_generate_has_no_cost(monkeypatch):
    monkeypatch.setattr(bench_build, 'compiler_binary', lambda compiler: None)
    broken = Workload('broken', lambda: 1 / 0, Datatype.F32)
    unit = BuildUnit(suite='t', target=TargetSpec('cuda', 'sm_90'),
                     datatype=Datatype.F32, config=Config('default'),
                     workloads=(broken,), batches=(4,))
    record, = bench_build.build(unit).workloads
    assert record.error.startswith('ZeroDivisionError')
    assert record.code_bytes is None and record.compile_s is None
