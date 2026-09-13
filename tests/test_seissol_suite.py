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


def test_a_merged_derivative_stores_its_members_before_the_loop():
    """The anelastic time derivative merges its levels into one loop, which
    reads `dQ(k)` through a table -- memory, not the registers.  Three things
    went wrong there, silently: the three contractions with `dQ(k)` read it
    through the first one's window (`antiunify.substitute`); with that fixed,
    `dQ(k+1) = dQext(k+1)` asked for rows the register image did not hold and
    the merge fell back; and the peeled level kept `dQ(1)` in registers,
    stored after the loop, so the loop read the buffer as the launch found it.
    """
    import re

    from tensorforge.generators.generator import MergeFallbackWarning

    system = 'viscoelastic-linearckanelastic'
    descrs = _read(system, f'{system}-o4-d', 'gpu_derivative')
    gen = Generator(descrs, Context(arch='sm_86', backend='cuda',
                                    fp_type=Datatype.F64))
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter('ignore')
        warnings.simplefilter('error', MergeFallbackWarning)
        gen.generate()
    src = gen.get_kernel()
    loop = src.index('for (int32_t batchIdv')
    members = {m for line in src[loop:].splitlines() if 'Table = ' in line
               for m in re.findall(r'\bglb_m\d+\b', line)}
    assert members
    for member in members:
        for store in re.finditer(rf'// {member} = store', src):
            assert store.start() < loop, member


def test_a_merged_run_stores_where_the_written_out_one_does():
    """The poroelastic derivative's loop reads `dQ(k)` over columns 10..13
    and then 0..13.  Its stand-in answered no question of the section plan,
    so the staging was sized for the first reader; the second re-staged it
    from row 1, the lead origin was pinned at 31, and three temporaries went
    into shared memory at `lead - 32` -- the previous multiplication's window,
    a race with more than one multiplication per block.  Merged, every store
    sits where the written-out build puts it."""
    import re

    from tensorforge.common.context import Options

    system = 'poroelastic-stp'
    offsets = {}
    for merge in (False, True):
        gen = Generator(_read(system, f'{system}-o4-s', 'gpu_derivative'),
                        Context(arch='sm_86', backend='cuda',
                                fp_type=Datatype.F32,
                                options=Options(merge_variants=merge)))
        with contextlib.redirect_stdout(io.StringIO()), \
                warnings.catch_warnings():
            warnings.simplefilter('ignore')
            gen.generate()
        src = gen.get_kernel()
        assert merge == ('for (int32_t batchIdv' in src)
        offsets[merge] = set(re.findall(r'_lead - (\d+);', src))
    assert offsets[True] == offsets[False]


def _carried(system, config, kernel):
    from tensorforge.backend.instructions.ptr_manip import VariantLoop
    from tensorforge.generators.generator import MergeFallbackWarning

    descrs = _read(system, config, kernel)
    fp = Datatype.F64 if config.endswith('-d') else Datatype.F32
    gen = Generator(descrs, Context(arch='sm_86', backend='cuda', fp_type=fp))
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter('ignore')
        warnings.simplefilter('error', MergeFallbackWarning)
        gen.generate()
    loops = [i for s in gen._sections for i in s.ir if isinstance(i, VariantLoop)]
    assert loops, kernel
    return [len(loop._carried) for loop in loops]


def test_only_a_value_read_before_it_is_assigned_rides_the_back_edge():
    """The damage model projects four times, each computing `I` afresh
    (`I = dQ(0) ...; I += ...`) and reading it back: nothing flows from one
    projection to the next, and chaining `I` anyway renamed its reload but
    not the product reading it, so `QDR(1..3)` came out zero.  The anelastic
    derivative adds each level onto `I` and `Iane`: those two do flow."""
    assert _carried('damage-nonlinearck', 'damage-nonlinearck-o4-s',
                    'gpu_projectToDR[0]') == [0]
    system = 'viscoelastic-linearckanelastic'
    assert _carried(system, f'{system}-o4-d', 'gpu_derivative') == [2]


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
