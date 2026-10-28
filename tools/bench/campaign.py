# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""One evaluation on several machines, driven from here.

    python3 tools/bench/campaign.py submit machines.yaml --campaign paper/
    python3 tools/bench/campaign.py status paper/
    python3 tools/bench/campaign.py fetch paper/
    python3 tools/bench/campaign.py publish paper/ -- --clustering c.csv

The figures of a paper are across machines, and every machine has its own
login, scheduler options, modules and place for files.  Running `runall.sh`
on each by hand is how a campaign comes to measure two commits, or to lose a
machine's tree.  This keeps the books and does nothing clever:

- **the code** is a commit of this checkout (`git archive`), shipped once
  per host and commit, so every machine measures the same thing.  A tree
  with changes is refused unless `--dirty`, which ships it as
  `git stash create` records it.
- **the job** is a script written from the machine's entry -- scheduler
  options, setup lines (modules, an environment), a launch prefix and the
  arguments of `evaluate.py` -- and kept beside the campaign, here and there,
  so that what ran can be read.  `--dry-run` prints it and touches nothing.
- **status** asks the scheduler and the job's own status file; **fetch**
  copies a finished tree back into `<campaign>/<machine>/eval`; **publish**
  sets every fetched tree into one publication, `<campaign>/publish`.

What it does not do is log in.  A remote call is `ssh -o BatchMode=yes
<host>` under the user's own configuration -- keys, `ProxyJump`, and for a
site behind a second factor a `ControlMaster` connection opened once by hand
-- so a call that would prompt fails instead of hanging; `$TF_CAMPAIGN_SSH`
replaces the command.  Nor does it wait for queues, which take as long as
they take: `submit` returns, and `fetch` takes what is done whenever asked.

## The machines file

YAML (or JSON): `machines`, a mapping from a name to an entry, and
`defaults`, laid under every entry -- `sbatch` key by key.

| key | |
|---|---|
| `host` | an ssh alias; none for this machine |
| `root` | where the code, the jobs and the trees live there |
| `scheduler` | `slurm` (default) or `shell`, run in the background -- never on a login node |
| `sbatch` | `#SBATCH` options, `{option: value}`; `true` for a bare flag |
| `setup` | shell lines before the evaluation: modules, an environment |
| `prepare` | shell lines `prepare` runs once, after `setup`, where the job cannot: a venv with `pip`, on a login node that reaches the internet |
| `launch` | what the evaluation is started under, e.g. `srun -n 1` |
| `python` | the interpreter (default `python3`) |
| `args` | arguments to `evaluate.py`: `--profile`, `--backend`, `--arch` |

`$TF_ROOT`, `$TF_CODE` (the shipped commit) and `$TF_OUT` (the tree) are set
for `setup`, `prepare` and the evaluation.  `machines.example.yaml` has one
of each kind.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

#: What an entry may say.  Anything else is a typo, and a typo in a scheduler
#: option is a job that runs somewhere it should not.
KEYS = ('host', 'root', 'scheduler', 'sbatch', 'setup', 'prepare', 'launch',
        'python', 'args')
SCHEDULERS = ('slurm', 'shell')
#: A machine's name is a directory here and there.
NAME = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.-]*$')
#: A value nobody filled in, as the example writes one: `<project>`.  Not a
#: shell redirection, which has a space after its `<`.
PLACEHOLDER = re.compile(r'<[^<>\s](?:[^<>]*[^<>\s])?>')


class CampaignError(RuntimeError):
    """A machine that could not be reached or would not take the job."""


# ----------------------------------------------------------------------
# Machines
# ----------------------------------------------------------------------

@dataclass
class Machine:
    name: str
    root: str
    host: Optional[str] = None
    scheduler: str = 'slurm'
    sbatch: Dict[str, object] = field(default_factory=dict)
    setup: List[str] = field(default_factory=list)
    prepare: List[str] = field(default_factory=list)
    launch: str = ''
    python: str = 'python3'
    args: List[str] = field(default_factory=list)

    def placeholders(self) -> List[str]:
        """What the entry has left as a placeholder, `<project>` and the
        like: a job submitted with one would be refused at best."""
        values = [self.host or '', self.root, self.launch, self.python,
                  *self.setup, *self.prepare, *self.args,
                  *(str(v) for v in self.sbatch.values())]
        return sorted({m for v in values for m in PLACEHOLDER.findall(v)})

    @classmethod
    def from_entry(cls, name: str, entry: Dict,
                   defaults: Optional[Dict] = None) -> 'Machine':
        defaults = defaults or {}
        for where, mapping in (('defaults', defaults), (name, entry)):
            unknown = sorted(set(mapping) - set(KEYS))
            if unknown:
                raise ValueError(f'{where}: unknown key(s) {unknown}; an '
                                 f'entry may say {list(KEYS)}')
        if not NAME.match(name):
            raise ValueError(f'{name!r}: a machine name is a directory '
                             'name -- letters, digits, _ . -')
        merged = {**defaults, **entry,
                  'sbatch': {**(defaults.get('sbatch') or {}),
                             **(entry.get('sbatch') or {})}}
        if not merged.get('root'):
            raise ValueError(f'{name}: no root, where the code and the '
                             'trees go')
        if any(c.isspace() for c in str(merged['root'])):
            raise ValueError(f'{name}: a root with a space in it cannot be '
                             'an #SBATCH --output path')
        scheduler = merged.get('scheduler') or 'slurm'
        if scheduler not in SCHEDULERS:
            raise ValueError(f'{name}: scheduler {scheduler!r} is none of '
                             f'{list(SCHEDULERS)}')
        lines = {}
        for key in ('setup', 'prepare', 'args'):
            value = merged.get(key) or []
            lines[key] = [str(v) for v in ([value] if isinstance(value, str)
                                           else value)]
        return cls(name=name, root=str(merged['root']).rstrip('/'),
                   host=merged.get('host') or None, scheduler=scheduler,
                   sbatch=merged['sbatch'], launch=merged.get('launch') or '',
                   python=merged.get('python') or 'python3', **lines)


def load_machines(path: Path) -> List[Machine]:
    """The entries of a machines file, in its order."""
    text = Path(path).read_text()
    if Path(path).suffix == '.json':
        blob = json.loads(text)
    else:
        try:
            import yaml
        except ImportError as exc:
            raise SystemExit('a YAML machines file needs PyYAML '
                             '(pip install pyyaml), or write it as JSON'
                             ) from exc
        blob = yaml.safe_load(text)
    if not isinstance(blob, dict) or not isinstance(blob.get('machines'),
                                                    dict):
        raise ValueError(f'{path}: no `machines` mapping')
    defaults = blob.get('defaults') or {}
    return [Machine.from_entry(name, entry or {}, defaults)
            for name, entry in blob['machines'].items()]


# ----------------------------------------------------------------------
# Reaching a machine
# ----------------------------------------------------------------------

def ssh_command() -> List[str]:
    return shlex.split(os.environ.get('TF_CAMPAIGN_SSH',
                                      'ssh -o BatchMode=yes'))


def shell_command(machine: Machine, script: str) -> List[str]:
    """`script` in a login shell on the machine, where `module` and the
    scheduler's commands are found."""
    command = ['bash', '-lc', script]
    if machine.host is None:
        return command
    return ssh_command() + [machine.host, shlex.join(command)]


def remote(machine: Machine, script: str, stdin: Optional[bytes] = None,
           check: bool = True) -> subprocess.CompletedProcess:
    proc = subprocess.run(shell_command(machine, script), input=stdin,
                          capture_output=True, check=False)
    if check and proc.returncode != 0:
        tail = (proc.stderr or proc.stdout).decode(errors='replace')
        tail = [line for line in tail.splitlines() if line.strip()]
        raise CampaignError(f'{machine.name}: exit {proc.returncode}'
                            + (f': {tail[-1]}' if tail else ''))
    return proc


def _q(text: str) -> str:
    return shlex.quote(text)


# ----------------------------------------------------------------------
# The code
# ----------------------------------------------------------------------

def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(['git', '-C', str(repo), *args],
                          capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise SystemExit(f'git {" ".join(args)}: {proc.stderr.strip()}')
    return proc.stdout


def snapshot(repo: Path, dirty: bool) -> str:
    """The commit every machine measures: HEAD, or with `dirty` the
    tracked state of the work tree as `git stash create` records it --
    which changes nothing here.  Untracked files are not shipped either
    way."""
    if _git(repo, 'status', '--porcelain', '--untracked-files=no').strip():
        if not dirty:
            raise SystemExit('the work tree has changes: commit them, or '
                             'pass --dirty to measure them as they are')
        stashed = _git(repo, 'stash', 'create').strip()
        if stashed:
            return stashed
    return _git(repo, 'rev-parse', 'HEAD').strip()


def code_dir(machine: Machine, sha: str) -> str:
    return f'{machine.root}/code/{sha}'


def ship(machine: Machine, repo: Path, sha: str) -> bool:
    """The commit onto the machine, unless it is there already; whether
    anything was sent.  Unpacked beside its place and moved in, so that a
    transfer cut short leaves no half a tree that looks whole."""
    code = code_dir(machine, sha)
    if remote(machine, f'test -e {_q(code)}/.complete',
              check=False).returncode == 0:
        return False
    archive = subprocess.run(['git', '-C', str(repo), 'archive',
                              '--format=tar.gz', sha], capture_output=True,
                             check=True).stdout
    part = f'{code}.part'
    remote(machine, f'set -e; rm -rf {_q(part)}; mkdir -p {_q(part)}; '
                    f'tar -xzf - -C {_q(part)}; rm -rf {_q(code)}; '
                    f'mv {_q(part)} {_q(code)}; touch {_q(code)}/.complete',
           stdin=archive)
    return True


# ----------------------------------------------------------------------
# The job
# ----------------------------------------------------------------------

def campaign_dir(machine: Machine, campaign: str) -> str:
    return f'{machine.root}/runs/{campaign}'


def out_dir(machine: Machine, campaign: str) -> str:
    return f'{campaign_dir(machine, campaign)}/{machine.name}'


def _environment(machine: Machine, sha: str,
                 campaign: Optional[str] = None) -> List[str]:
    return ([f'export TF_ROOT={_q(machine.root)}',
             f'export TF_CODE={_q(code_dir(machine, sha))}']
            + ([f'export TF_OUT={_q(out_dir(machine, campaign))}']
               if campaign else []))


def job_script(machine: Machine, campaign: str, sha: str) -> str:
    """The script the machine runs: the scheduler's header, the
    environment, the setup lines and the evaluation, framed by a status
    file the job writes itself -- which outlives the scheduler's memory of
    the job."""
    out = out_dir(machine, campaign)
    lines = ['#!/bin/bash -l']
    if machine.scheduler == 'slurm':
        lines += [f'#SBATCH --job-name=tf-{campaign}-{machine.name}'[:72],
                  f'#SBATCH --output={out}.log']
        for key, value in machine.sbatch.items():
            if value is True:
                lines.append(f'#SBATCH --{key}')
            elif value is not None and value is not False:
                lines.append(f'#SBATCH --{key}={value}')
    lines += [f'# campaign {campaign}, machine {machine.name}, commit {sha}',
              *_environment(machine, sha, campaign),
              'mkdir -p "$TF_OUT"',
              'echo "running $(date -u +%FT%TZ) $(hostname)" > '
              '"$TF_OUT/STATUS"',
              *machine.setup,
              'cd "$TF_CODE" || exit 1',
              ' '.join(filter(None, [
                  machine.launch, machine.python,
                  'tools/bench/evaluate.py --out "$TF_OUT"',
                  shlex.join(machine.args)])),
              'code=$?',
              'echo "done $code $(date -u +%FT%TZ)" > "$TF_OUT/STATUS"',
              'exit $code']
    return '\n'.join(lines) + '\n'


def submit_job(machine: Machine, campaign: str, script: str) -> str:
    """The script onto the machine and into its queue -- or, without a
    scheduler, into the background; the job's id or the process's."""
    base = campaign_dir(machine, campaign)
    path = f'{base}/{machine.name}.sh'
    remote(machine, f'mkdir -p {_q(base)} && cat > {_q(path)} && '
                    f'chmod +x {_q(path)}', stdin=script.encode())
    if machine.scheduler == 'slurm':
        proc = remote(machine, f'cd {_q(base)} && sbatch --parsable '
                               f'{_q(path)}')
        # `--parsable` is `<id>` or `<id>;<cluster>` -- after whatever a
        # site's wrapper around sbatch says first.
        for line in reversed(proc.stdout.decode().splitlines()):
            m = re.match(r'^\s*(\d+)(;\S+)?\s*$', line)
            if m:
                return m.group(1)
        raise CampaignError(f'{machine.name}: sbatch named no job: '
                            + proc.stdout.decode().strip()[-200:])
    log = f'{base}/{machine.name}.log'
    proc = remote(machine, f'cd {_q(base)} && nohup bash {_q(path)} > '
                           f'{_q(log)} 2>&1 < /dev/null & echo $!')
    return proc.stdout.decode().strip().splitlines()[-1]


def job_state(machine: Machine, record: Dict) -> str:
    """What became of the job: `done <exit code>` once the job said so,
    else the scheduler's word for it, else `gone` -- killed, or never
    started, before it could say anything."""
    out, job = record['out'], record['job']
    said = remote(machine, f'cat {_q(out)}/STATUS 2>/dev/null',
                  check=False).stdout.decode().strip()
    if said.startswith('done'):
        return said
    if machine.scheduler == 'slurm':
        state = remote(machine, f'squeue -h -j {_q(job)} -o %T 2>/dev/null',
                       check=False).stdout.decode().strip()
        if not state:
            state = remote(machine, f'sacct -n -X -j {_q(job)} -o State '
                                    '2>/dev/null', check=False
                           ).stdout.decode().strip()
        state = state.split()[0].lower() if state else ''
    else:
        alive = remote(machine, f'kill -0 {_q(job)} 2>/dev/null',
                       check=False).returncode == 0
        state = 'running' if alive else ''
    return state or ('gone' + (f' ({said})' if said else ''))


def fetch_tree(machine: Machine, record: Dict, into: Path) -> None:
    """The machine's tree, whole, into `into`: packed there, unpacked here
    beside its place and moved in."""
    part = into.with_name(into.name + '.part')
    shutil.rmtree(part, ignore_errors=True)
    part.mkdir(parents=True)
    # The far side's complaints to a file, not a pipe: a tree still being
    # written can make tar say a great deal, and a full pipe stops it.
    with tempfile.TemporaryFile() as complaints:
        pack = subprocess.Popen(shell_command(
            machine, f'tar -czf - -C {_q(record["out"])} .'),
            stdout=subprocess.PIPE, stderr=complaints)
        unpack = subprocess.run(['tar', '-xzf', '-', '-C', str(part)],
                                stdin=pack.stdout, capture_output=True)
        pack.stdout.close()
        status = pack.wait()
        complaints.seek(0)
        err = complaints.read().decode(errors='replace').strip()
    if status != 0 or unpack.returncode != 0:
        shutil.rmtree(part, ignore_errors=True)
        raise CampaignError(f'{machine.name}: fetch failed: '
                            + (err or unpack.stderr.decode().strip()))
    shutil.rmtree(into, ignore_errors=True)
    part.rename(into)


# ----------------------------------------------------------------------
# The campaign, here
# ----------------------------------------------------------------------

def read_campaign(path: Path) -> Dict:
    record = Path(path) / 'campaign.json'
    if not record.exists():
        raise SystemExit(f'{path}: no campaign here (campaign.json)')
    return json.loads(record.read_text())


def write_campaign(path: Path, blob: Dict) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)
    (Path(path) / 'campaign.json').write_text(json.dumps(blob, indent=2))


def _only(machines: Sequence[Machine], only: Optional[str],
          where: str) -> List[Machine]:
    """The machines `--only` names, or all of them."""
    if not only:
        return list(machines)
    wanted = {n.strip() for n in only.split(',') if n.strip()}
    missing = sorted(wanted - {m.name for m in machines})
    if missing:
        raise SystemExit(f'not in {where}: {missing}')
    return [m for m in machines if m.name in wanted]


def _machines_of(blob: Dict, only: Optional[str]) -> List[Machine]:
    return _only([Machine(**entry['machine'])
                  for entry in blob['machines'].values()], only,
                 'the campaign')


def cmd_submit(args) -> int:
    machines = _only(load_machines(args.machines), args.only,
                     str(args.machines))
    here = Path(args.campaign)
    name = here.resolve().name
    if not NAME.match(name):
        raise SystemExit(f'{name!r}: a campaign\'s name is a directory '
                         'name there too -- letters, digits, _ . -')
    sha = snapshot(args.repo, args.dirty)
    blob = (json.loads((here / 'campaign.json').read_text())
            if (here / 'campaign.json').exists()
            else {'name': name, 'commit': sha, 'machines': {}})
    if blob['commit'] != sha:
        raise SystemExit(f'{here} measures {blob["commit"]}, the checkout '
                         f'is at {sha}: one campaign, one commit -- start '
                         'another')
    failed = 0
    for machine in machines:
        script = job_script(machine, name, sha)
        if args.dry_run:
            print(f'== {machine.name} ({machine.host or "this machine"})\n'
                  f'{script}')
            continue
        unfilled = machine.placeholders()
        if unfilled:
            print(f'{machine.name}: fill in {", ".join(unfilled)} first',
                  file=sys.stderr)
            failed += 1
            continue
        if machine.name in blob['machines'] and not args.again:
            print(f'{machine.name}: submitted already (job '
                  f'{blob["machines"][machine.name]["job"]}); --again to '
                  'submit it anew')
            continue
        try:
            sent = ship(machine, args.repo, sha)
            job = submit_job(machine, name, script)
        except CampaignError as exc:
            print(exc, file=sys.stderr)
            failed += 1
            continue
        (here / machine.name).mkdir(parents=True, exist_ok=True)
        (here / machine.name / 'job.sh').write_text(script)
        blob['machines'][machine.name] = {
            'machine': asdict(machine), 'job': job,
            'out': out_dir(machine, name), 'submitted': time.time()}
        write_campaign(here, blob)
        print(f'{machine.name}: job {job}'
              + (f', commit {sha[:12]} shipped' if sent else ''))
    return 1 if failed else 0


def cmd_prepare(args) -> int:
    failed = 0
    sha = snapshot(args.repo, args.dirty)
    for machine in _only(load_machines(args.machines), args.only,
                         str(args.machines)):
        script = '\n'.join(_environment(machine, sha) + machine.setup
                           + ['set -e'] + machine.prepare)
        if args.dry_run:
            print(f'== {machine.name}\n{script}\n')
            continue
        unfilled = machine.placeholders()
        if unfilled:
            print(f'{machine.name}: fill in {", ".join(unfilled)} first',
                  file=sys.stderr)
            failed += 1
            continue
        try:
            ship(machine, args.repo, sha)
            if machine.prepare:
                remote(machine, script)
            print(f'{machine.name}: prepared at {sha[:12]}')
        except CampaignError as exc:
            print(exc, file=sys.stderr)
            failed += 1
    return 1 if failed else 0


def cmd_status(args) -> int:
    blob = read_campaign(args.campaign)
    for machine in _machines_of(blob, args.only):
        record = blob['machines'][machine.name]
        try:
            state = job_state(machine, record)
        except CampaignError as exc:
            state = f'unreachable -- {exc}'
        fetched = (Path(args.campaign) / machine.name / 'eval').exists()
        print(f'{machine.name:16s} job {record["job"]:>10s}  {state}'
              + ('  (fetched)' if fetched else ''))
    return 0


def cmd_fetch(args) -> int:
    blob = read_campaign(args.campaign)
    failed = 0
    for machine in _machines_of(blob, args.only):
        record = blob['machines'][machine.name]
        try:
            state = job_state(machine, record)
            if not state.startswith('done') and not args.partial:
                print(f'{machine.name}: {state}; not fetched (--partial '
                      'takes it as it is)')
                continue
            fetch_tree(machine, record,
                       Path(args.campaign) / machine.name / 'eval')
            print(f'{machine.name}: fetched ({state})')
        except CampaignError as exc:
            print(exc, file=sys.stderr)
            failed += 1
    return 1 if failed else 0


def publish_command(campaign: Path, extra: Sequence[str]) -> List[str]:
    trees = sorted(Path(campaign).glob('*/eval'))
    return [sys.executable, str(HERE / 'publish.py'),
            *(str(t) for t in trees), '--out',
            str(Path(campaign) / 'publish'), *extra]


def cmd_publish(args) -> int:
    read_campaign(args.campaign)
    if not sorted(Path(args.campaign).glob('*/eval')):
        raise SystemExit(f'{args.campaign}: nothing fetched yet')
    return subprocess.run(publish_command(Path(args.campaign), args.extra),
                          check=False).returncode


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest='command', required=True)

    def with_code(p):
        p.add_argument('machines', type=Path, help='the machines file')
        p.add_argument('--only', help='comma-separated machine names')
        p.add_argument('--dirty', action='store_true',
                       help='ship the work tree\'s changes as they are')
        p.add_argument('--repo', type=Path, default=ROOT,
                       help=argparse.SUPPRESS)
        p.add_argument('--dry-run', action='store_true',
                       help='print what would run, and touch nothing')

    p = sub.add_parser('submit', help='ship the commit and queue the '
                                      'evaluation on every machine')
    with_code(p)
    p.add_argument('--campaign', type=Path, required=True,
                   help='the campaign\'s directory here; its name is its '
                        'name there')
    p.add_argument('--again', action='store_true',
                   help='submit a machine the campaign has submitted')
    p.set_defaults(run=cmd_submit)

    p = sub.add_parser('prepare', help='ship the commit and run each '
                                       'machine\'s prepare lines once')
    with_code(p)
    p.set_defaults(run=cmd_prepare)

    for name, run, text in (('status', cmd_status, 'where every job is'),
                            ('fetch', cmd_fetch, 'copy finished trees back')):
        p = sub.add_parser(name, help=text)
        p.add_argument('campaign', type=Path)
        p.add_argument('--only', help='comma-separated machine names')
        if name == 'fetch':
            p.add_argument('--partial', action='store_true',
                           help='fetch a tree whose job has not finished')
        p.set_defaults(run=run)

    p = sub.add_parser('publish', help='publish.py over every fetched tree; '
                                       'arguments after -- go to it')
    p.add_argument('campaign', type=Path)
    p.add_argument('extra', nargs=argparse.REMAINDER)
    p.set_defaults(run=cmd_publish)

    args = ap.parse_args(argv)
    if getattr(args, 'extra', None) and args.extra[0] == '--':
        args.extra = args.extra[1:]
    try:
        return args.run(args)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
