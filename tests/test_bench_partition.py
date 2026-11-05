# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""A kernel cut into launches (`suite.chain_partition`), timed as one
sequence of them and checked against the whole (`build.py`, `run.py`).

The cut is the one gemmforge-chainforge makes only if it is the one yateto
makes.  So it is checked against a recording of yateto doing it
(`tests/fixtures/forge`), and the rest -- that a sequence of launches shares
one buffer per tensor, that it costs what its launches move, that a build
whose outputs differ from the reference's is kept out of the timing -- on
SeisSol's exports.  No GPU and no compiler.
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
import run as bench_run  # noqa: E402
import suite  # noqa: E402
from harness import driver_bench  # noqa: E402
from tensorforge.analysis.cost import Cost, sequence_cost  # noqa: E402
from tensorforge.common.basic_types import Datatype  # noqa: E402

DATA = BENCH / 'data'
FIXTURES = ROOT / 'tests' / 'fixtures' / 'forge'
RECORDED = sorted(FIXTURES.glob('yateto-*.json'))


def _workload(export: str, kernel: str) -> suite.Workload:
    # By name and not by pattern: `gpu_neighboringFlux[0]` is a character
    # class to `fnmatch`.
    workload, = [w for w in suite.from_dump(DATA / export)
                 if w.name == kernel]
    return workload


def _unit(config: str, *workloads, backend='cuda', arch='sm_80',
          datatype=Datatype.F64) -> suite.BuildUnit:
    return suite.BuildUnit('t', suite.TargetSpec(backend, arch), datatype,
                           suite.CONFIGS[config], tuple(workloads), (256,))


# ----------------------------------------------------------------------
# The cut
# ----------------------------------------------------------------------

@pytest.mark.parametrize('recorded', RECORDED, ids=lambda p: p.stem)
def test_the_cut_is_the_one_yateto_makes(recorded):
    """With chainforge and without: the same launches, of the same kinds,
    as yateto's code makes of each kernel."""
    blob = json.loads(recorded.read_text())
    descriptions = json.loads((DATA / blob['export']).read_text())[
        'descriptions']
    for partition, cut in (('chains', suite.chain_partition),
                           ('ops', suite.ops_partition)):
        for kernel, routines in blob[partition].items():
            launches = cut(descriptions[kernel])
            assert [launch.kind for launch in launches] == [
                r['kind'] for r in routines], (partition, kernel)


def test_a_derivative_step_is_a_chain_between_copies():
    """`csa csa (chain6 csa) x (O - 1)`: the time derivatives a chain each,
    the integral accumulated between them by gemmforge."""
    for order in (3, 4, 5, 6):
        description = _workload(f'elastic-{order}-d-s1.json',
                                'gpu_derivative').description
        kinds = [launch.kind for launch in
                 suite.chain_partition(description)]
        assert kinds == ['csa', 'csa'] + ['chain', 'csa'] * (order - 1)
        assert all(len(launch.operations) == 6 for launch in
                   suite.chain_partition(description)
                   if launch.kind == 'chain')


def test_a_chain_keeps_its_temporaries_and_passes_on_no_other():
    description = _workload('elastic-4-d-s1.json',
                            'gpu_derivative').description
    launches = suite.chain_partition(description)
    pieces = suite.split_description(description, launches)
    for launch, piece in zip(launches, pieces):
        temporary = {t['name'] for t in piece['tensors']
                     if t['flags']['temporary']}
        if launch.kind == 'chain':
            assert temporary and all(n.startswith('_tmp') for n in temporary)
        else:
            assert not temporary
    # Every operation once, in order, and the original untouched.
    assert [op for piece in pieces for op in piece['operations']] == \
        description['operations']
    with pytest.raises(ValueError, match='once and in order'):
        suite.split_description(description, list(reversed(launches)))


def test_a_kernel_of_one_chain_is_one_launch_and_costs_the_same():
    for kernel in ('gpu_volume', 'gpu_localFluxAll', 'gpu_neighboringFlux[0]'):
        workload = _workload('elastic-4-d-s1.json', kernel)
        launch, = suite.chain_partition(workload.description)
        assert launch.kind == 'chain'
        whole = bench_build.workload_cost(workload, suite.CONFIGS['baseline'],
                                          1, Datatype.F64)
        chains = bench_build.workload_cost(workload, suite.CONFIGS['chains'],
                                           1, Datatype.F64)
        assert (chains.flops, chains.bytes) == (whole.flops, whole.bytes)


def test_the_launches_cost_what_they_move():
    """The kernel's arithmetic in every cut, and more traffic the finer it
    is cut: what a temporary between two launches and an input every launch
    reads add to the fused kernel's."""
    workload = _workload('elastic-4-d-s1.json', 'gpu_derivative')
    costs = {label: bench_build.workload_cost(workload, suite.CONFIGS[label],
                                              1, Datatype.F64)
             for label in ('baseline', 'chains', 'ops')}
    assert len({(c.flops, c.nonzero_flops) for c in costs.values()}) == 1
    assert costs['baseline'].bytes < costs['chains'].bytes < \
        costs['ops'].bytes


def test_a_sequence_adds_its_launches_up():
    one = Cost(mults=2, adds=1, read_bytes=8, write_bytes=4, batch=3)
    two = Cost(mults=5, adds=5, read_bytes=16, write_bytes=8, batch=3,
               nonzero_exact=False)
    both = sequence_cost([one, two])
    assert (both.flops, both.bytes, both.batch) == (13, 36, 3)
    assert not both.nonzero_exact
    with pytest.raises(ValueError):
        sequence_cost([one, Cost(batch=4)])


def test_the_outputs_and_the_boxes_written():
    regions = bench_build.regions_of(
        _workload('elastic-4-d-s1.json', 'gpu_derivative').description)
    assert sorted(regions) == ['I', 'dQ(0)', 'dQ(1)', 'dQ(2)', 'dQ(3)']
    assert (regions['I'].lo, regions['I'].hi) == ((0, 0), (20, 9))
    assert (regions['dQ(1)'].lo, regions['dQ(1)'].hi,
            regions['dQ(1)'].sizes) == ((0, 0), (10, 9), (16, 9))
    assert bench_build.regions_of(None) is None


# ----------------------------------------------------------------------
# A sequence of TensorForge launches
# ----------------------------------------------------------------------

def _generate(config: str, workload, **unit):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return bench_build.generate(workload, _unit(config, workload, **unit))


def test_a_sequence_is_one_buffer_per_tensor_and_one_definition_per_kernel():
    workload = _workload('elastic-4-d-s1.json', 'gpu_derivative')
    src, record = _generate('chains', workload)
    assert src is not None, record.error
    assert record.launches == 8 and len(record.kernels) == 8
    assert record.symbol.startswith('sequence_')
    launch = re.search(r'tfb_launch_gpu_derivative\(void\* stream, size_t '
                       r'batch\) \{\n  using namespace tfb_gpu_derivative;\n'
                       r'(.*?)\n\}', src, re.S).group(1)
    assert launch.count('launcher_kernel_') == 8
    # The launch types at namespace scope, ahead of the kernels: inside the
    # namespace the guard keeps every launch's copy of them out.
    assert src.index('struct LaunchInfo {') < src.index(
        'namespace tfb_gpu_derivative')
    # Q is read by the first launches and dQ(1) written by one, read by the
    # next: one buffer each, whoever names it.
    assert len(re.findall(r'static double\* d_b\d+ = nullptr;', src)) == len(
        set(re.findall(r'static double\* (d_b\d+) = nullptr;', src)))
    assert 'const_cast<const double**>' in src
    # The checksum reads the kernel's outputs, by name.
    assert src.count('for (size_t x0 = ') == 5
    assert record.cost.flops == bench_build.workload_cost(
        workload, suite.CONFIGS['baseline'], 1, Datatype.F64).flops


def test_the_same_kernel_twice_in_a_sequence_is_defined_once():
    """One kernel launched twice -- the same product on two pairs of
    operands -- is defined once, and two sequences sharing it keep their
    definitions apart in their namespaces."""
    from tensorforge.common.context import Context, Options
    from tensorforge.generators.generator import Generator
    workload = _workload('elastic-4-d-s1.json', 'gpu_volume')
    piece = suite.split_description(workload.description)[0]
    gen = Generator(bench_build._read(piece),
                    Context(arch='sm_80', backend='cuda',
                            fp_type=Datatype.F64, options=Options()))
    with contextlib.redirect_stdout(io.StringIO()), \
            warnings.catch_warnings():
        warnings.simplefilter('ignore')
        gen.generate()
    src, symbol = driver_bench.emit_sequence_tu([gen, gen], 'cuda', 'twice',
                                                '')
    launcher = f'launcher_{gen.get_base_name()}'
    # Declared by its header, defined by its launcher: once each.
    assert src.count(f'void {launcher}(') == 2
    assert src.count(f'  {launcher}(') == 2
    assert 'namespace tfb_twice {' in src and symbol.startswith('sequence_')


def test_the_whole_kernel_reads_its_constants_numbers():
    """A batch-constant operator holds the description's numbers, which a
    generator that builds them into its code computes with either way."""
    src, _ = _generate('baseline', _workload('elastic-4-d-s1.json',
                                             'gpu_volume'))
    assert re.search(r'static const double h_\w+\[\] = \{', src)
    assert 'tfb_launches_gpu_volume() {\n  return 1u;' in src


def test_only_an_export_is_cut():
    workload = suite.Workload('case', lambda: [], Datatype.F32)
    src, record = _generate('chains', workload, datatype=Datatype.F32)
    assert src is None and 'only an exported kernel' in record.error


# ----------------------------------------------------------------------
# The check before a configuration is timed
# ----------------------------------------------------------------------

def test_a_build_that_computes_something_else_is_not_timed(monkeypatch):
    workload = _workload('elastic-4-d-s1.json', 'gpu_volume')
    checksums = {'baseline': 1.0, 'chains': 1.0 + 1e-13, 'ops': 1.5}
    built = []
    for label in checksums:
        unit = _unit(label, workload)
        record = bench_build.WorkloadBuild(name=workload.name, origin='',
                                           obj=Path('x.o'))
        built.append((unit, bench_build.UnitBuild(unit, Path(label),
                                                  [record])))

    def check(exe, name, batch, device, timeout):
        return {'checksum': checksums[str(exe)], 'magnitude': 2.0}

    monkeypatch.setattr(bench_run, 'check_binary', check)
    args = type('Args', (), {'check_batch': 64, 'timeout': 1.0})()
    device = bench_run.Device(suite.TargetSpec('cuda', 'sm_80'), 'nvidia', 0,
                              'device')
    verdicts = bench_run.check_units(built, device, args)
    configs = {unit.label: unit.config.label for unit, _ in built}
    by_config = {configs[key[0]]: v for key, v in verdicts.items()}
    assert by_config['baseline']['check'] == 'reference'
    assert not by_config['chains']['failed']
    assert by_config['ops']['failed']
    assert 'differs from baseline' in by_config['ops']['check']
    unit, unit_build = built[-1]
    rows = bench_run.measure(unit_build, device, [workload], args, verdicts)
    assert len(rows) == 1 and 'differs from baseline' in rows[0]['error']
