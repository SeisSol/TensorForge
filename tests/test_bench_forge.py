# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""The generators SeisSol used before TensorForge, building the launches
gemmforge-chainforge makes of a kernel (`forge.py`).

The comparison with gemmforge and chainforge is only a comparison if they
are handed what yateto handed them.  So every matrix and factor is checked
against a recording of yateto doing it (`tests/fixtures/forge`), and the
translation unit they are timed in on SeisSol's exports.  No GPU and no
compiler; skipped where the generators are not installed.
"""

from __future__ import annotations

import contextlib
import io
import json
import re
import sys
import warnings
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / 'tools' / 'bench'
if str(BENCH) not in sys.path:
    sys.path.insert(0, str(BENCH))

import build as bench_build  # noqa: E402
import suite  # noqa: E402
from tensorforge.common.basic_types import Datatype  # noqa: E402

DATA = BENCH / 'data'
FIXTURES = ROOT / 'tests' / 'fixtures' / 'forge'
RECORDED = sorted(FIXTURES.glob('yateto-*.json'))


def _workload(export: str, kernel: str) -> suite.Workload:
    workload, = [w for w in suite.from_dump(DATA / export)
                 if w.name == kernel]
    return workload


def _generate(config: str, workload, backend='cuda', arch='sm_80'):
    unit = suite.BuildUnit('t', suite.TargetSpec(backend, arch), Datatype.F64,
                           suite.CONFIGS[config], (workload,), (256,))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return bench_build.generate(workload, unit)


def _forge():
    pytest.importorskip('gemmforge')
    pytest.importorskip('chainforge')
    import forge
    return forge


def _norm(text) -> str:
    return re.sub(r'np\.(?:int|float)\d+\(([^)]*)\)', r'\1', str(text))


def _inputs(routine) -> dict:
    """What a generator was handed, as the recording prints it."""
    g = routine.generator
    if routine.kind == 'csa':
        return {'kind': 'csa', 'alpha': ('<scalar>' if isinstance(
                    g._alpha, str) else repr(g._alpha)),
                'beta': repr(g._beta), 'a': _norm(g._mat_a),
                'b': _norm(g._mat_b)}
    if routine.kind == 'gemm':
        return {'kind': 'gemm', 'trans': [False, False],
                'alpha': repr(g._alpha), 'beta': repr(g._beta),
                'a': _norm(g._mat_a), 'b': _norm(g._mat_b),
                'c': _norm(g._mat_c)}
    return {'kind': 'chain', 'gemms': [
        {'alpha': repr(x.alpha), 'beta': repr(x.beta),
         'prefer_align': str(bool(x.prefer_align)),
         'a': _norm(x.mat_a.gen_descr()), 'b': _norm(x.mat_b.gen_descr()),
         'c': _norm(x.mat_c.gen_descr())} for x in g.gemm_list]}


@pytest.mark.parametrize('recorded', RECORDED, ids=lambda p: p.stem)
def test_what_yateto_hands_the_generators(recorded):
    """Every launch of every kernel, with chainforge and without: the
    matrices and the factors yateto handed gemmforge and chainforge."""
    forge = _forge()
    blob = json.loads(recorded.read_text())
    backend, arch = blob['target']
    exported = suite.from_dump(DATA / blob['export'])
    datatype = exported[0].datatype.name
    for partition in ('chains', 'ops'):
        config = suite.CONFIGS['gemmforge-chainforge' if partition == 'chains'
                               else 'gemmforge']
        for kernel, routines in blob[partition].items():
            workload, = [w for w in exported if w.name == kernel]
            with contextlib.redirect_stdout(io.StringIO()):
                lowered = [forge.lower(piece, launch, backend, arch, datatype)
                           for launch, piece in
                           bench_build.pieces(workload, config)]
            assert [_inputs(r) for r in lowered] == routines, (
                partition, kernel)


@pytest.mark.parametrize('backend,arch', [('cuda', 'sm_80'),
                                          ('hip', 'gfx90a')])
def test_the_old_generators_build_a_translation_unit(backend, arch):
    _forge()
    workload = _workload('elastic-4-d-s1.json', 'gpu_derivative')
    src, record = _generate('gemmforge-chainforge', workload,
                            backend=backend, arch=arch)
    assert src is not None, record.error
    assert record.launches == 8 and record.symbol.startswith('forge_')
    assert record.kernels[2].startswith('kernel_cf_gemms_')
    assert '#define CHECK_ERR' in src
    launch = re.search(r'tfb_launch_gpu_derivative\(void\* stream, size_t '
                       r'batch\) \{\n  using namespace tfb_gpu_derivative;\n'
                       r'(.*?)\n\}', src, re.S).group(1)
    assert len(launch.splitlines()) == 8
    # The integral's copies take the factor the TensorForge driver passes.
    assert launch.count('static_cast<double>(0.5)') == 4
    # Every routine once, however often it is launched.
    for kernel in set(record.kernels):
        # Defined once; the launcher names it without a parenthesis after.
        assert src.count(f'{kernel}(') == 1
    assert src.count('for (size_t x0 = ') == 5


def test_the_old_generators_refuse_what_they_do_not_emit():
    _forge()
    src, record = _generate('gemmforge-chainforge',
                            _workload('elastic-4-d-s1.json', 'gpu_volume'),
                            backend='oneapi', arch='pvc')
    assert src is None and 'CUDA and HIP' in record.error


def test_without_them_the_configurations_say_why(monkeypatch):
    import forge
    monkeypatch.setattr(forge.importlib.util, 'find_spec', lambda name: None)
    src, record = _generate('gemmforge',
                            _workload('elastic-4-d-s1.json', 'gpu_volume'))
    assert src is None and 'requirements-forge.txt' in record.error
