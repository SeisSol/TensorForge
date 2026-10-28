# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""The whole evaluation, in one command.

Four tools and an exporter have to agree on where the files are: an export
names a configuration, a run names a device and a configuration, a profile
names a workload inside a run, and a plot joins all three. Driving them by
hand is how a plot comes to be drawn over last week's run -- the file is
there, it parses, and nothing says it is the wrong one.

    python3 tools/bench/evaluate.py --out eval/
    python3 tools/bench/evaluate.py --out eval/ --export --profile
    python3 tools/bench/evaluate.py --out eval/ --exports tools/bench/data
    python3 tools/bench/evaluate.py --out eval/ --no-split --no-buildcost

What it does, skipping anything the machine cannot do and saying so:

1. **export** (`--export`): `tools/seissol_export.py` over the configuration
   matrix, into `<out>/exports`. Needs SeisSol's codegen tree; without it the
   step is skipped and existing exports are used.
2. **roofline**: once per device and precision, into `<out>/ceiling/<F32|F64>`,
   and joined into every plot of that precision. The ceilings are a property
   of the machine, not of the workload, so measuring them per export would be
   the same measurement a dozen times -- but not of the machine alone: the FMA
   chain is built for one type, and a double-precision export drawn under the
   single-precision roof sits as far below it as the part's two rates differ.
3. **run**: `run.py` per export, into `<out>/runs/<export>`.  An export of one
   simulation over the ramp-up sweep (`publish.RAMPUP_BATCHES`), which the
   launch cost and a time step's quarter launches need; one of several
   simulations at the suite's batches, of which only the largest is read.
   Where the matrix units change what is built -- NVIDIA, Intel's ESIMD
   (`matrix_paths`) -- in `tensor-cores` as well, in the same run, since
   two runs differ by the device's clock as much as by the code.
4. **split** (`--split`, on by default): each kernel of a one-simulation
   export whole and as one launch per operation (`suites/seissolsplit.py`),
   into `<out>/split/<export>` -- what fusion is worth.
5. **buildcost** (`--buildcost`, on by default): every kernel of the export
   built once, nothing run (`suites/seissol.py`), into `<out>/build/<export>`
   -- what generating and compiling a configuration costs, not only the few
   kernels the run times.
6. **profile** (`--profile`): `profile.py` per export, into
   `<out>/profiles/<export>`.
7. **plot**: `plot.py` per export with whatever of the above exists.
8. **compare** and **publish** (`--publish`, on by default): across
   everything that ran, the comparison PDF and the paper's figures and tables
   in `<out>/publish`, compiled where pdflatex is at hand.  `--seissol` and
   `--clustering` go to `publish.py`.
9. **manifest**: `<out>/evaluation.json` -- what ran, what was skipped and
   why, and the version of everything involved.

The manifest is the point of doing this in one command rather than nine. A
directory of SVGs says what a kernel did; it does not say which generator
built it, which device ran it, or whether the ceiling beside it came from the
same machine. A reader a month later cannot reconstruct that, and an
evaluation nobody can reconstruct is a picture rather than a measurement.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for extra in (str(HERE), str(ROOT / 'src'), str(ROOT / 'tests')):
    if extra not in sys.path:
        sys.path.insert(0, extra)

from capture import Capture                                     # noqa: E402
from publish import RAMPUP_BATCHES                              # noqa: E402

#: The configurations `tools/seissol_export_all.sh` produces, as the data they
#: are rather than as a shell loop: the same list drives the export and names
#: the output, so the two cannot drift.
EXPORT_MATRIX = [
    # (equation, solver, order, precision, mechanisms, simcount)
    ('elastic', 'linearck', 3, 's', 0, 1),
    ('elastic', 'linearck', 3, 'd', 0, 1),
    ('elastic', 'linearck', 4, 's', 0, 1),
    ('elastic', 'linearck', 4, 'd', 0, 1),
    ('elastic', 'linearck', 6, 's', 0, 1),
    ('elastic', 'linearck', 6, 'd', 0, 1),
    ('viscoelastic', 'linearckanelastic', 3, 's', 3, 1),
    ('viscoelastic', 'linearckanelastic', 3, 'd', 3, 1),
    ('viscoelastic', 'linearckanelastic', 4, 's', 3, 1),
    ('viscoelastic', 'linearckanelastic', 4, 'd', 3, 1),
    ('viscoelastic', 'linearckanelastic', 6, 's', 3, 1),
    ('viscoelastic', 'linearckanelastic', 6, 'd', 3, 1),
    ('elastic', 'linearck', 3, 's', 0, 8),
    ('elastic', 'linearck', 3, 'd', 0, 8),
    ('elastic', 'linearck', 4, 's', 0, 8),
    ('elastic', 'linearck', 4, 'd', 0, 8),
]


def export_name(equation, solver, order, precision, mechanisms, simcount):
    """The capture's tag: the readers place a run by its directory where the
    rows do not say, so the two spellings must be one (`capture.py`)."""
    return Capture(equation, order, precision, simcount, mechanisms,
                   solver).tag


@dataclass
class Step:
    """One tool invocation, and what became of it."""
    name: str
    command: List[str] = field(default_factory=list)
    status: str = 'pending'
    seconds: float = 0.0
    note: str = ''

    def to_dict(self):
        return {'step': self.name, 'status': self.status,
                'seconds': round(self.seconds, 1), 'note': self.note,
                'command': ' '.join(self.command)}


def run_step(name: str, command: List[str], log: Path,
             timeout: float) -> Step:
    """One subprocess, its output kept, its failure recorded and not raised.

    A step that fails is a finding: the evaluation goes on and the manifest
    says which part of it is missing. Stopping would throw away every
    measurement after the first machine that lacks a profiler.
    """
    step = Step(name, command)
    started = time.monotonic()
    log.parent.mkdir(parents=True, exist_ok=True)
    try:
        proc = subprocess.run(command, capture_output=True, text=True,
                              timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        step.status, step.note = 'timeout', f'after {timeout:.0f}s'
        step.seconds = time.monotonic() - started
        return step
    except OSError as exc:
        step.status, step.note = 'failed', str(exc)
        step.seconds = time.monotonic() - started
        return step
    step.seconds = time.monotonic() - started
    log.write_text(' '.join(command) + '\n\n' + proc.stdout + '\n' +
                   proc.stderr)
    if proc.returncode != 0:
        step.status = 'failed'
        tail = [l for l in (proc.stderr or proc.stdout).splitlines() if l.strip()]
        step.note = (tail[-1][:160] if tail else f'exit {proc.returncode}') \
            + f' (see {log})'
    else:
        step.status = 'ok'
    return step


def codegen_available() -> Optional[Path]:
    """SeisSol's codegen tree, where `seissol_export.py` would find it."""
    path = Path(os.environ.get(
        'SEISSOL_CODEGEN', ROOT.parent / 'seissol21' / 'codegen'))
    return path if (path / 'generate.py').exists() else None


def do_exports(out: Path, timeout: float, steps: List[Step]) -> None:
    """The configuration matrix, each into its own JSON."""
    codegen = codegen_available()
    if codegen is None:
        steps.append(Step('export', [], 'skipped',
                          note='no SeisSol codegen tree; set $SEISSOL_CODEGEN'))
        return
    out.mkdir(parents=True, exist_ok=True)
    for entry in EXPORT_MATRIX:
        equation, solver, order, precision, mechanisms, simcount = entry
        target = out / f'{export_name(*entry)}.json'
        if target.exists():
            steps.append(Step(f'export {target.name}', [], 'cached'))
            continue
        command = [sys.executable, str(ROOT / 'tools' / 'seissol_export.py'),
                   equation, solver, str(order), precision, str(target),
                   '--mechanisms', str(mechanisms),
                   '--simcount', str(simcount)]
        steps.append(run_step(f'export {target.name}', command,
                              out / f'{target.stem}.log', timeout))


def matrix_paths(backend: Optional[str], arch: Optional[str]) -> bool:
    """Whether `tensor-cores` changes what a target builds, and so is worth
    a second configuration in the timing run: on NVIDIA and in Intel's ESIMD
    builds it does, for some kernels (`suite.CONFIGS`); AMD's ranking chooses
    its matrix path whatever the option says, and over the exports it
    changed no SPMD build for PVC -- there the configuration would time
    `baseline`'s code again.  The kernels it leaves alone where it changes
    some are timed twice, which is what `control` is for anyway."""
    if backend:
        return backend in ('cuda', 'esimd')
    if arch:
        return not arch.startswith('gfx')
    try:
        from harness.gpu_detect import detect_all
    except ImportError:
        return False
    return any(gpu.vendor in ('nvidia', 'intel') for gpu in detect_all())


def bench_command(suite: Path, out: Path, args, *extra: str) -> List[str]:
    """`run.py` over `suite` into `out`, on the target asked for."""
    command = [sys.executable, str(HERE / 'run.py'), str(suite),
               '--out', str(out), '--repeat', str(args.repeat),
               '--iters', str(args.iters), *extra]
    for flag, value in (('--backend', args.backend), ('--arch', args.arch)):
        if value:
            command += [flag, value]
    return command


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--out', type=Path, required=True,
                    help='the evaluation tree to write')
    ap.add_argument('--exports', type=Path, default=None,
                    help='directory of descriptor exports to measure '
                         '(default: <out>/exports, or tools/bench/data)')
    ap.add_argument('--export', action='store_true',
                    help='produce the exports first (needs SeisSol codegen)')
    ap.add_argument('--profile', action='store_true',
                    help='also run the vendor profiler per export')
    ap.add_argument('--suite', type=Path,
                    default=HERE / 'suites' / 'seissolbench.py')
    ap.add_argument('--backend', default=None)
    ap.add_argument('--arch', default=None)
    ap.add_argument('--configs', default=None,
                    help='the configurations of the timing run, as run.py '
                         'takes them (default: the suite\'s, and '
                         '`tensor-cores` where it changes the build)')
    ap.add_argument('--batches', default=None,
                    help='the batches of the timing run, every export '
                         '(default: the ramp-up sweep for one simulation, '
                         'the suite\'s for several)')
    ap.add_argument('--split', action=argparse.BooleanOptionalAction,
                    default=True,
                    help='time each kernel whole and in pieces too, for '
                         'what fusion is worth')
    ap.add_argument('--buildcost', action=argparse.BooleanOptionalAction,
                    default=True,
                    help='build every kernel of each export once, for what '
                         'making a configuration costs')
    ap.add_argument('--publish', action=argparse.BooleanOptionalAction,
                    default=True,
                    help='write the paper\'s figures and tables at the end')
    ap.add_argument('--seissol', type=Path, action='append', default=[],
                    help='SeisSol run output for publish.py (repeatable)')
    ap.add_argument('--clustering', action='append', default=[],
                    help='a clustering for publish.py, CSV[@TAG] '
                         '(repeatable)')
    ap.add_argument('--repeat', type=int, default=3)
    ap.add_argument('--iters', type=int, default=200)
    ap.add_argument('--timeout', type=float, default=7200.0)
    ap.add_argument('--archive', action='store_true',
                    help='pack the tree into <out>.tar.xz when done')
    args = ap.parse_args()

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    steps: List[Step] = []
    started = time.time()

    # 1. exports
    exports = args.exports or (out / 'exports')
    if args.export:
        do_exports(out / 'exports', args.timeout, steps)
        exports = out / 'exports'
    if not exports.exists() or not sorted(exports.glob('*.json')):
        fallback = HERE / 'data'
        if sorted(fallback.glob('*.json')):
            exports = fallback
        else:
            print(f'no exports in {exports} and none in {fallback}; '
                  f'pass --export or --exports', file=sys.stderr)
            return 2
    captures = sorted(exports.glob('*.json'))
    print(f'{len(captures)} exports from {exports}')

    # 2. the ceilings, once for this machine and each precision the exports
    #    are in.  An export whose configuration does not say is measured
    #    against both rather than against a guess.
    ceiling = out / 'ceiling'
    datatypes = sorted({c.datatype for c in map(_capture_of, captures) if c}
                       or {'F32', 'F64'})
    for datatype in datatypes:
        command = [sys.executable, str(HERE / 'roofline.py'), '--ceiling-only',
                   '--datatype', datatype, '--out', str(ceiling / datatype)]
        if args.backend:
            command += ['--backend', args.backend]
        if args.arch:
            command += ['--arch', args.arch]
        steps.append(run_step(f'roofline {datatype}', command,
                              out / 'logs' / f'roofline-{datatype}.log',
                              args.timeout))

    # What the timing run is measured in: the matrix units beside the suite's
    # own where they change the build, in the same run.
    configs = args.configs
    if configs is None and matrix_paths(args.backend, args.arch):
        configs = '+tensor-cores'
    sweep = ','.join(map(str, RAMPUP_BATCHES))

    # 3..7. per export: run, split, buildcost, profile, plot
    for capture in captures:
        tag = capture.stem
        captured = _capture_of(capture)
        single = captured is None or captured.simulations == 1
        env_dump = {'TF_BENCH_DUMP': str(capture)}
        run_dir = out / 'runs' / tag
        batches = args.batches or (sweep if single else None)
        extra: List[str] = []
        for flag, value in (('--configs', configs), ('--batches', batches)):
            if value:
                extra += [flag, value]
        step = run_with_env(f'run {tag}',
                            bench_command(args.suite, run_dir, args, *extra),
                            env_dump, out / 'logs' / f'run-{tag}.log',
                            args.timeout)
        steps.append(step)
        if step.status != 'ok':
            continue

        if args.split and single:
            steps.append(run_with_env(
                f'split {tag}',
                bench_command(HERE / 'suites' / 'seissolsplit.py',
                              out / 'split' / tag, args),
                env_dump, out / 'logs' / f'split-{tag}.log', args.timeout))
        elif args.split:
            steps.append(Step(f'split {tag}', [], 'skipped',
                              note='several simulations: the fusion figure '
                                   'reads one'))

        if args.buildcost:
            steps.append(run_with_env(
                f'buildcost {tag}',
                bench_command(HERE / 'suites' / 'seissol.py',
                              out / 'build' / tag, args, '--build-only',
                              '--configs', 'baseline'),
                env_dump, out / 'logs' / f'buildcost-{tag}.log',
                args.timeout))

        profile_dir = None
        if args.profile:
            profile_dir = out / 'profiles' / tag
            command = [sys.executable, str(HERE / 'profile.py'),
                       str(args.suite), '--out', str(profile_dir)]
            for flag, value in (('--backend', args.backend),
                                ('--arch', args.arch),
                                ('--configs', args.configs)):
                if value:
                    command += [flag, value]
            p = run_with_env(f'profile {tag}', command, env_dump,
                             out / 'logs' / f'profile-{tag}.log', args.timeout)
            steps.append(p)
            if p.status != 'ok':
                profile_dir = None

        command = [sys.executable, str(HERE / 'plot.py'),
                   str(run_dir / 'bench.json'), '--out', str(out / 'plots' / tag)]
        # Every precision's roof; `plot.py` draws each row under its own.
        for roofline_json in sorted(ceiling.glob('*/roofline.json')):
            command += ['--roofline-json', str(roofline_json)]
        if profile_dir is not None:
            command += ['--profile', str(profile_dir)]
        command.append('--pdf')
        steps.append(run_step(f'plot {tag}', command,
                              out / 'logs' / f'plot-{tag}.log', args.timeout))

    # 8. across everything that ran: the configurations against each other,
    #    the orders against each other, and the machines against each other.
    #    Last, because it reads what the steps above wrote and nothing else
    #    -- a campaign that half failed still gets the comparison of the
    #    half that did not.  The split runs stay out of the comparison, whose
    #    views know nothing of pieces; the publication reads the whole tree.
    steps.append(run_step(
        'compare',
        [sys.executable, str(HERE / 'compare.py'), str(out / 'runs'),
         str(out / 'ceiling'), '--out', str(out / 'compare'), '--pdf'],
        out / 'logs' / 'compare.log', args.timeout))
    if args.publish:
        command = [sys.executable, str(HERE / 'publish.py'), str(out),
                   '--out', str(out / 'publish'),
                   '--title', f'TensorForge evaluation {out.name}']
        for path in args.seissol:
            command += ['--seissol', str(path)]
        for text in args.clustering:
            command += ['--clustering', text]
        if shutil.which('pdflatex'):
            command.append('--compile')
        steps.append(run_step('publish', command,
                              out / 'logs' / 'publish.log', args.timeout))

    # 9. the manifest
    from tensorforge import interop                                 # noqa
    manifest = {
        'evaluation': {
            'started': started,
            'elapsed_s': round(time.time() - started, 1),
            'tensorforge': interop.get_version(),
            'exports_from': str(exports),
            'suite': str(args.suite),
            'repeat': args.repeat, 'iters': args.iters,
            'configs': configs,
            'batches': args.batches or {'one simulation': sweep,
                                        'several': 'the suite\'s'},
            'split': bool(args.split), 'buildcost': bool(args.buildcost),
            'profiled': bool(args.profile),
            'published': bool(args.publish),
        },
        'steps': [s.to_dict() for s in steps],
    }
    (out / 'evaluation.json').write_text(json.dumps(manifest, indent=2))

    ok = sum(1 for s in steps if s.status == 'ok')
    bad = [s for s in steps if s.status in ('failed', 'timeout')]
    skipped = [s for s in steps if s.status in ('skipped', 'cached')]
    print(f'\n{ok} steps ok, {len(bad)} failed, {len(skipped)} skipped or '
          f'cached; {out / "evaluation.json"}')
    for s in bad:
        print(f'  {s.name}: {s.note}')

    if args.archive:
        base = str(out)
        shutil.make_archive(base, 'xztar', root_dir=out.parent,
                            base_dir=out.name)
        print(f'archive: {base}.tar.xz')
    return 1 if bad else 0


def _capture_of(path: Path) -> Optional[Capture]:
    """The configuration of an export: the one its file name spells, which
    is how this script names them, else the one it says it was generated
    at -- read only then, an export being up to twenty megabytes."""
    named = Capture.from_tag(path.stem)
    if named is not None:
        return named
    try:
        blob = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return Capture.from_config(blob.get('config'))


def run_with_env(name: str, command: List[str], extra: Dict[str, str],
                 log: Path, timeout: float) -> Step:
    """`run_step` with environment on top, for the steps that take their
    capture through `TF_BENCH_DUMP` rather than through an argument."""
    old = {k: os.environ.get(k) for k in extra}
    os.environ.update(extra)
    try:
        return run_step(name, command, log, timeout)
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


if __name__ == '__main__':
    raise SystemExit(main())
