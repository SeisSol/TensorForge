# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""SeisSol's kernels, every equation system, order and precision.

Recorded from SeisSol's code generator (`tools/seissol_export.py`) and packed
into `fixtures/seissol/` (`seissol_suite`).  Everything is read on every run;
a sample is built; `--seissol` builds all of it, on every target -- which is
hours, and which is what the analysis passes are checked against.
"""

from __future__ import annotations

import contextlib
import io
import warnings

import pytest

import seissol_suite as suite
from tensorforge.common.basic_types import Datatype
from tensorforge.common.context import Context
from tensorforge.frontend.yateto import DescriptionReader
from tensorforge.generators.generator import Generator

SYSTEMS = {'acoustic-linearck', 'anisotropic-linearck', 'elastic-linearck',
           'poroelastic-stp', 'viscoelastic-linearck',
           'viscoelastic-linearckanelastic', 'viscoacoustic-linearck',
           'viscoacoustic-linearckanelastic', 'damage-nonlinearck'}
TARGETS = [('cuda', 'sm_86'), ('hip', 'gfx942'), ('oneapi', 'pvc')]
CONFIGS = list(suite.configs())


def _read(system, config, kernel):
    return DescriptionReader(None, {}).read(
        suite.description(system, config, kernel))[0]


def _build(descrs, backend, arch):
    fp = next((d.dest.tensor.datatype for d in reversed(descrs)
               if getattr(getattr(d, 'dest', None), 'tensor', None) is not None
               and d.dest.tensor.datatype in (Datatype.F32, Datatype.F64)),
              Datatype.F32)
    gen = Generator(descrs, Context(arch=arch, backend=backend, fp_type=fp))
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter('ignore')
        gen.generate()
    return gen


def test_every_system_order_and_precision_is_recorded():
    assert set(suite.systems()) == SYSTEMS
    for system in SYSTEMS:
        seen = {(suite.config(system, c)['order'],
                 suite.config(system, c)['precision'])
                for s, c in CONFIGS if s == system}
        assert seen == {(o, p) for o in range(2, 9) for p in ('s', 'd')}


def test_a_configuration_says_what_it_was_generated_with():
    system, name = CONFIGS[0]
    cfg = suite.config(system, name)
    assert {'equation', 'solver', 'order', 'precision', 'mechanisms',
            'workarounds'} <= set(cfg)
    assert suite.kernels(system, name)


def test_the_recorded_workarounds_are_the_known_poroelastic_alignment():
    """yateto describes the space-time predictor twice in one poroelastic
    kernel, with alignment 64 and then 0; the recording keeps the lesser.
    Nothing else was worked around."""
    for system, name in CONFIGS:
        notes = suite.config(system, name)['workarounds']
        if system == 'poroelastic-stp':
            assert all('alignment described twice' in n for n in notes)
        else:
            assert notes == []


@pytest.mark.slow
@pytest.mark.parametrize('system,config', CONFIGS,
                         ids=[c for _, c in CONFIGS])
def test_every_kernel_is_read(system, config):
    for kernel in suite.kernels(system, config):
        assert _read(system, config, kernel), kernel


#: A kernel of each kind, at order 4 in single precision: the volume and the
#: face integrals, the time kernel, and whatever the system adds.
SAMPLE = [(system, f'{system}-o4-s') for system in sorted(SYSTEMS)]


@pytest.mark.parametrize('system,config', SAMPLE, ids=[c for _, c in SAMPLE])
def test_a_sample_builds(system, config):
    names = suite.kernels(system, config)
    picked = [n for n in names if any(k in n for k in
                                      ('volume', 'localFlux', 'neighborFlux',
                                       'derivative'))][:4] or names[:4]
    for kernel in picked:
        _build(_read(system, config, kernel), 'cuda', 'sm_86')


@pytest.mark.parametrize('backend,arch', TARGETS)
@pytest.mark.parametrize('system,config', CONFIGS,
                         ids=[c for _, c in CONFIGS])
def test_every_kernel_builds(request, system, config, backend, arch):
    if not request.config.getoption('--seissol'):
        pytest.skip('builds every SeisSol kernel; pass --seissol')
    failed = []
    for kernel in suite.kernels(system, config):
        try:
            _build(_read(system, config, kernel), backend, arch)
        except Exception as error:     # noqa: BLE001 -- collected, all named
            failed.append(f'{kernel}: {type(error).__name__}: {error}')
    assert not failed, '\n'.join(failed[:20])
