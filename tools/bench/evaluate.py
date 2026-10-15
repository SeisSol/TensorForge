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

What it does, skipping anything the machine cannot do and saying so:

1. **export** (`--export`): `tools/seissol_export.py` over the configuration
   matrix, into `<out>/exports`. Needs SeisSol's codegen tree; without it the
   step is skipped and existing exports are used.
2. **run**: `run.py` per export, into `<out>/runs/<export>`.
3. **roofline**: once per device, into `<out>/ceiling`, and joined into every
   plot. The ceilings are a property of the machine, not of the workload, so
   measuring them per export would be the same measurement a dozen times.
4. **profile** (`--profile`): `profile.py` per export, into
   `<out>/profiles/<export>`.
5. **plot**: `plot.py` per export with whatever of the above exists.
6. **manifest**: `<out>/evaluation.json` -- what ran, what was skipped and
   why, and the version of everything involved.

The manifest is the point of doing this in one command rather than four. A
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
    return f'{equation}-{order}-{precision}-s{simcount}'


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
    ap.add_argument('--configs', default=None)
    ap.add_argument('--batches', default=None)
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

    # 2. the ceilings, once for this machine
    ceiling = out / 'ceiling'
    command = [sys.executable, str(HERE / 'roofline.py'), '--ceiling-only',
               '--out', str(ceiling)]
    if args.backend:
        command += ['--backend', args.backend]
    if args.arch:
        command += ['--arch', args.arch]
    steps.append(run_step('roofline', command, out / 'logs' / 'roofline.log',
                          args.timeout))
    roofline_json = ceiling / 'roofline.json'

    # 3..5. per export: run, profile, plot
    for capture in captures:
        tag = capture.stem
        env_dump = {'TF_BENCH_DUMP': str(capture)}
        run_dir = out / 'runs' / tag
        command = [sys.executable, str(HERE / 'run.py'), str(args.suite),
                   '--out', str(run_dir), '--repeat', str(args.repeat),
                   '--iters', str(args.iters)]
        for flag, value in (('--backend', args.backend), ('--arch', args.arch),
                            ('--configs', args.configs),
                            ('--batches', args.batches)):
            if value:
                command += [flag, value]
        step = run_with_env(f'run {tag}', command, env_dump,
                            out / 'logs' / f'run-{tag}.log', args.timeout)
        steps.append(step)
        if step.status != 'ok':
            continue

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
        if roofline_json.exists():
            command += ['--roofline-json', str(roofline_json)]
        if profile_dir is not None:
            command += ['--profile', str(profile_dir)]
        steps.append(run_step(f'plot {tag}', command,
                              out / 'logs' / f'plot-{tag}.log', args.timeout))

    # 6. the manifest
    from tensorforge import interop                                 # noqa
    manifest = {
        'evaluation': {
            'started': started,
            'elapsed_s': round(time.time() - started, 1),
            'tensorforge': interop.get_version(),
            'exports_from': str(exports),
            'suite': str(args.suite),
            'repeat': args.repeat, 'iters': args.iters,
            'profiled': bool(args.profile),
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
