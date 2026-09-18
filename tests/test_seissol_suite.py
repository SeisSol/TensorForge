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


def test_a_table_over_constants_holds_their_numbers():
    """The damage step merges runs that differ only in scalar constants.
    The kernel reads those as literals and declares no name for them, so a
    table naming them (`glb_m116`) did not compile; the literal was then read
    off a zero-dimensional array with `[0]` and the merge fell back."""
    import re

    from tensorforge.generators.generator import MergeFallbackWarning

    system = 'damage-nonlinearck'
    gen = Generator(_read(system, f'{system}-o4-s', 'gpu_damageStep'),
                    Context(arch='sm_86', backend='cuda', fp_type=Datatype.F32))
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter('ignore')
        warnings.simplefilter('error', MergeFallbackWarning)
        gen.generate()
    src = gen.get_kernel()
    tables = [line for line in src.splitlines()
              if re.search(r'const float v\d+Table = ', line)]
    assert tables, 'no table over scalars'
    declared = set(re.findall(r'\b(glb_m\d+) = ', src))
    for line in tables:
        named = set(re.findall(r'\bglb_m\d+\b', line))
        assert named <= declared, sorted(named - declared)


def test_a_multiplication_that_does_not_fit_is_refused():
    """One multiplication of the damage step needs more shared memory than a
    block has.  The block was then sized to hold no multiplication at all --
    height 0, the window never declared -- and the source went out as if
    nothing had happened.

    Three things have to be off for the damage step to reach that size at all,
    and each of them is why it no longer does by default: `shared_packing`
    (the coloring made every color its largest buffer), `merge_variants`, and
    `register_temporaries` (a temporary read out of its producer's image never
    reaches a buffer).  Order 6 stood here, then order 7; with images the
    whole family fits, so what the test pins is the guard, not the order."""
    from tensorforge.common.context import Options
    from tensorforge.common.exceptions import GenerationError

    system = 'damage-nonlinearck'
    gen = Generator(_read(system, f'{system}-o7-d', 'gpu_damageStep'),
                    Context(arch='sm_86', backend='cuda', fp_type=Datatype.F64,
                            options=Options(merge_variants=False,
                                            shared_packing=False,
                                            register_temporaries='none')))
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with pytest.raises(GenerationError, match='shared memory'):
            gen.generate()


def test_a_multiplication_wider_than_a_block_says_so():
    """The other way to get no multiplication per block, and it used to be
    reported as the first: `RegmaxBlockPolicy` caps a block at 128 threads on
    NVIDIA, so a multiplication 512 lanes wide gets none -- while its shared
    memory, half a kilobyte, fits a hundred times over."""
    from tensorforge.common.exceptions import GenerationError
    from tensorforge.generators.lanes import LaneConfig, deduce

    descrs = _read('elastic-linearck', 'elastic-linearck-o2-s', 'gpu_volume')
    context = Context(arch='sm_86', backend='cuda', fp_type=Datatype.F32)
    rows = deduce([op for d in descrs for op in d.operations()],
                  context).num_active_threads
    gen = Generator(descrs, context,
                    lanes=LaneConfig(num_threads=512, num_active_threads=rows,
                                     lead_width=1))
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with pytest.raises(GenerationError, match='no block holds one'):
            gen.generate()


def test_packed_buffers_take_what_is_live_and_no_more():
    """The same damage step has 908 shared buffers, never more than 22 of
    them live at once and never more than 65 KB.  The coloring made 22
    colors, each as large as its largest buffer: 145 KB per multiplication,
    and no block held one.  Packed, the arena is what is live, and no two
    buffers live at the same time share a byte of it."""
    from tensorforge.backend.opt.inspect import _check_shared_aliasing

    system = 'damage-nonlinearck'
    gen = Generator(_read(system, f'{system}-o6-d', 'gpu_damageStep'),
                    Context(arch='sm_86', backend='cuda', fp_type=Datatype.F64))
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter('ignore')
        gen.generate()
    for section in gen._sections:
        assert section.shr_mem_obj.get_mults_per_block() >= 1
        assert section.shr_mem_obj.get_size_per_mult() * 8 < 70_000
        prologue = {id(i) for i in section.global_ir}
        stream = [i for i in section.stream if id(i) not in prologue]
        assert not [d for d in _check_shared_aliasing(stream)
                    if d.severity == 'error']


def test_a_result_is_stored_once_nothing_reads_it_again():
    """The elastic derivative computes `dQ(0)` first and reads it once, at the
    start of the first level; held to the section's end, it and every later
    `dQ(k)` waited in registers for their stores -- `dQ(0)` for all but the
    first few hundred of 19,500 lines, 72 registers of 255 by the end."""
    import re

    from tensorforge.common.context import Options

    system = 'elastic-linearck'
    where = {}
    for early in (False, True):
        gen = Generator(_read(system, f'{system}-o6-s', 'gpu_derivative'),
                        Context(arch='sm_86', backend='cuda',
                                fp_type=Datatype.F32,
                                options=Options(early_writebacks=early)))
        with contextlib.redirect_stdout(io.StringIO()), \
                warnings.catch_warnings():
            warnings.simplefilter('ignore')
            gen.generate()
        lines = gen.get_kernel().splitlines()
        dq0 = next(s.name for s in gen._scopes.get_global_scope().values()
                   if getattr(s.obj, 'alias', None) == 'dQ(0)')
        # the store names the binding, `glb_m3`, and not the parameter
        store = next(i for i, l in enumerate(lines)
                     if re.search(rf'// (?:glb_)?{dq0} = store', l))
        where[early] = store / len(lines)
    assert where[False] > 0.9
    assert where[True] < 0.1


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


def test_a_merged_run_carries_the_temporaries_it_reads_back():
    """What the body reads before it assigns rides the back edge -- including
    the values that are not `+=` and the ones that are temporaries.

    The residency keys its entries by *symbol* name, and the loop spelled its
    keys from the tensor instead: `glb_` and the tensor's name, which is the
    symbol for a kernel parameter and for nothing else.  A temporary is `s233`
    where its tensor is `t232`, so the damage step's seven integrals were
    never found -- the peeled copy left each as a register image, the chain
    was never closed, and `I` and `sourceI` came out at three fifths of what
    the written-out kernel computes (A100, every input scale) while
    `transportDer(0..3)`, which are parameters, were right.
    """
    from tensorforge.backend.instructions.ptr_manip import VariantLoop

    system = 'damage-nonlinearck'
    gen = Generator(_read(system, f'{system}-o4-s', 'gpu_damageStep'),
                    Context(arch='sm_86', backend='cuda', fp_type=Datatype.F32))
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter('ignore')
        gen.generate()

    def loops(instructions):
        for instr in instructions:
            if isinstance(instr, VariantLoop):
                yield instr
            for region in instr.regions() or ():
                yield from loops(region)

    found = [loop for section in gen._sections
             for loop in loops(section.stream)]
    assert found, 'the damage step merged nothing'
    biggest = max(found, key=lambda loop: len(loop.region))
    assert biggest.carried, ('the longest run carries nothing round its back '
                             'edge, though its body reads its integrals back')
