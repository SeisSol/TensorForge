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
