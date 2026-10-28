# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""A campaign over several machines (`tools/bench/campaign.py`), run here.

No cluster is at hand, so the far side is stood in for, and only that: `ssh`
by a script that runs the command on this machine, `sbatch` by one that runs
the job in the background, the evaluation by one that writes a tree.
Everything between -- the commit shipped, the job script, the status file,
the tree fetched back -- is the real thing, through `bash -l` as on a
cluster.
"""

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

BENCH = Path(__file__).resolve().parent.parent / 'tools' / 'bench'
if str(BENCH) not in sys.path:
    sys.path.insert(0, str(BENCH))

import campaign  # noqa: E402

IDENTITY = {'GIT_AUTHOR_NAME': 'test', 'GIT_AUTHOR_EMAIL': 't@example.org',
            'GIT_COMMITTER_NAME': 'test',
            'GIT_COMMITTER_EMAIL': 't@example.org'}


def _script(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(0o755)
    return path


@pytest.fixture
def world(tmp_path, monkeypatch):
    """A repository to ship, a stand-in for ssh, sbatch and squeue, and an
    evaluation that writes a tree where it is told and says where it ran."""
    for key, value in IDENTITY.items():
        monkeypatch.setenv(key, value)
    repo = tmp_path / 'repo'
    (repo / 'tools' / 'bench').mkdir(parents=True)
    (repo / 'tools' / 'bench' / 'evaluate.py').write_text('# stand-in\n')
    subprocess.run(['git', 'init', '-q', str(repo)], check=True)
    subprocess.run(['git', '-C', str(repo), 'add', '-A'], check=True)
    subprocess.run(['git', '-C', str(repo), 'commit', '-q', '-m', 'x'],
                   check=True)
    bin_dir = tmp_path / 'bin'
    _script(bin_dir / 'fake-ssh', '#!/bin/bash\n'
            'while [ $# -gt 0 ]; do case "$1" in -o) shift 2;; '
            '-*) shift;; *) break;; esac; done\n'
            'shift\nexec bash -c "$*"\n')
    # Some sites wrap sbatch in a script that has its say first.
    _script(bin_dir / 'sbatch', '#!/bin/bash\n'
            '[ "$1" = --parsable ] && shift\n'
            'echo "--> Verifying valid submit host...OK"\n'
            'nohup bash "$1" > "$1.out" 2>&1 < /dev/null &\n'
            'echo "4242;testcluster"\n')
    _script(bin_dir / 'squeue', '#!/bin/bash\nexit 0\n')
    _script(bin_dir / 'sacct', '#!/bin/bash\nexit 0\n')
    evaluation = _script(tmp_path / 'evaluation', '#!/bin/bash\n'
                         'while [ $# -gt 0 ]; do [ "$1" = --out ] && '
                         'out="$2"; shift; done\n'
                         'mkdir -p "$out/runs/x"\n'
                         'echo "{}" > "$out/runs/x/bench.json"\n'
                         'pwd > "$out/where"\n')
    monkeypatch.setenv('PATH', f'{bin_dir}{os.pathsep}{os.environ["PATH"]}')
    monkeypatch.setenv('TF_CAMPAIGN_SSH', str(bin_dir / 'fake-ssh'))
    return {'repo': repo, 'root': tmp_path / 'far', 'evaluation': evaluation,
            'here': tmp_path / 'paper'}


def _machines(world, path, **entries):
    blob = {'defaults': {'sbatch': {'nodes': 1, 'time': '01:00:00'}},
            'machines': entries}
    path.write_text(json.dumps(blob))
    return path


def _wait_done(world, name, timeout=30.0):
    status = world['root'] / name / 'runs' / 'paper' / name / 'STATUS'
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if status.exists() and status.read_text().startswith('done'):
            return status.read_text()
        time.sleep(0.1)
    raise AssertionError(f'{name}: the job did not finish')


# ----------------------------------------------------------------------
# The machines file
# ----------------------------------------------------------------------

def test_an_entry_lies_over_the_defaults_and_a_typo_is_refused():
    defaults = {'sbatch': {'nodes': 1, 'time': '01:00:00'}, 'args': ['-x']}
    machine = campaign.Machine.from_entry(
        'a', {'root': '/r/', 'sbatch': {'time': '02:00:00'},
              'setup': 'module load x'}, defaults)
    assert machine.sbatch == {'nodes': 1, 'time': '02:00:00'}
    assert machine.root == '/r' and machine.setup == ['module load x']
    assert machine.args == ['-x'] and machine.scheduler == 'slurm'
    with pytest.raises(ValueError, match='partiton'):
        campaign.Machine.from_entry('a', {'root': '/r', 'partiton': 'x'})
    with pytest.raises(ValueError, match='scheduler'):
        campaign.Machine.from_entry('a', {'root': '/r', 'scheduler': 'pbs'})
    with pytest.raises(ValueError, match='no root'):
        campaign.Machine.from_entry('a', {})
    with pytest.raises(ValueError, match='directory'):
        campaign.Machine.from_entry('../a', {'root': '/r'})
    with pytest.raises(ValueError, match='space'):
        campaign.Machine.from_entry('a', {'root': '/my files'})


def test_a_placeholder_left_in_is_not_submitted(world, tmp_path, capsys):
    machines = _machines(world, tmp_path / 'm.json', cluster={
        'host': 'login', 'root': str(world['root'] / 'cluster'),
        'sbatch': {'account': '<project>'},
        'setup': ['module load <cuda module>', 'cat < in > out']})
    machine = campaign.load_machines(machines)[0]
    assert machine.placeholders() == ['<cuda module>', '<project>']
    assert campaign.main(['submit', str(machines), '--campaign',
                          str(world['here']), '--repo',
                          str(world['repo'])]) == 1
    assert 'fill in <cuda module>, <project>' in capsys.readouterr().err
    assert not world['root'].exists()


def test_the_example_is_a_machines_file():
    machines = campaign.load_machines(BENCH / 'machines.example.yaml')
    assert [m.name for m in machines] == ['nvidia-cluster', 'amd-cluster',
                                          'workstation']
    nvidia, amd, workstation = machines
    assert nvidia.sbatch['time'] == '08:00:00'
    assert amd.sbatch['exclusive'] is True and amd.launch.startswith('srun')
    assert workstation.scheduler == 'shell'


def test_a_slurm_job_says_what_it_runs():
    machine = campaign.Machine.from_entry('lumi', {
        'root': '/scratch/x', 'sbatch': {'partition': 'gpu', 'exclusive': True,
                                         'qos': None},
        'setup': ['module load rocm'], 'launch': 'srun -n 1',
        'args': ['--clustering', 'mesh clustering.csv@elastic-4-d']})
    script = campaign.job_script(machine, 'paper', 'abc123')
    lines = script.splitlines()
    assert lines[0] == '#!/bin/bash -l'
    assert '#SBATCH --partition=gpu' in lines
    assert '#SBATCH --exclusive' in lines
    assert not any('qos' in line for line in lines)
    assert '#SBATCH --output=/scratch/x/runs/paper/lumi.log' in lines
    # Every #SBATCH line before the first command, as sbatch reads them.
    first = next(i for i, line in enumerate(lines[1:], 1)
                 if not line.startswith('#'))
    assert all(not line.startswith('#SBATCH') for line in lines[first:])
    assert 'export TF_CODE=/scratch/x/code/abc123' in lines
    assert lines.index('module load rocm') < lines.index(
        'cd "$TF_CODE" || exit 1')
    assert ("srun -n 1 python3 tools/bench/evaluate.py --out \"$TF_OUT\" "
            "--clustering 'mesh clustering.csv@elastic-4-d'") in lines
    assert lines[-2].startswith('echo "done $code')


# ----------------------------------------------------------------------
# A campaign, end to end
# ----------------------------------------------------------------------

def test_slurm_over_ssh_submit_status_fetch_publish(world, tmp_path):
    machines = _machines(world, tmp_path / 'm.json', cluster={
        'host': 'login', 'root': str(world['root'] / 'cluster'),
        'sbatch': {'partition': 'gpu'}, 'python': str(world['evaluation']),
        'args': ['--profile']})
    argv = ['submit', str(machines), '--campaign', str(world['here']),
            '--repo', str(world['repo'])]
    assert campaign.main(argv) == 0
    record = json.loads((world['here'] / 'campaign.json').read_text())
    entry = record['machines']['cluster']
    assert entry['job'] == '4242'
    assert (world['here'] / 'cluster' / 'job.sh').read_text().startswith(
        '#!/bin/bash -l')
    assert _wait_done(world, 'cluster').startswith('done 0')
    sha = record['commit']
    shipped = world['root'] / 'cluster' / 'code' / sha
    assert (shipped / '.complete').exists()
    assert (shipped / 'tools' / 'bench' / 'evaluate.py').exists()

    machine = campaign.Machine(**entry['machine'])
    assert campaign.job_state(machine, entry).startswith('done 0')
    assert campaign.main(['fetch', str(world['here'])]) == 0
    tree = world['here'] / 'cluster' / 'eval'
    assert json.loads((tree / 'runs' / 'x' / 'bench.json').read_text()) == {}
    assert (tree / 'where').read_text().strip() == str(shipped)
    assert (tree / 'STATUS').read_text().startswith('done 0')

    command = campaign.publish_command(world['here'], ['--clustering', 's'])
    assert command[2:] == [str(tree), '--out',
                           str(world['here'] / 'publish'),
                           '--clustering', 's']

    # A second submission of the same machine is refused, and so is another
    # commit in the same campaign.
    assert campaign.main(argv) == 0
    assert json.loads((world['here'] / 'campaign.json').read_text()
                      ) == record
    (world['repo'] / 'new').write_text('x')
    subprocess.run(['git', '-C', str(world['repo']), 'add', '-A'], check=True)
    subprocess.run(['git', '-C', str(world['repo']), 'commit', '-q', '-m',
                    'y'], check=True)
    with pytest.raises(SystemExit, match='one commit'):
        campaign.main(argv)


def test_a_machine_without_a_scheduler_runs_it_in_the_background(world,
                                                                  tmp_path):
    machines = _machines(world, tmp_path / 'm.json', box={
        'scheduler': 'shell', 'root': str(world['root'] / 'box'),
        'python': str(world['evaluation'])})
    assert campaign.main(['submit', str(machines), '--campaign',
                          str(world['here']), '--repo',
                          str(world['repo'])]) == 0
    assert _wait_done(world, 'box').startswith('done 0')
    entry = json.loads((world['here'] / 'campaign.json').read_text()
                       )['machines']['box']
    assert entry['machine']['host'] is None and entry['job'].isdigit()
    assert campaign.main(['fetch', str(world['here']), '--only', 'box']) == 0
    assert (world['here'] / 'box' / 'eval' / 'runs').is_dir()


def test_an_unfinished_job_is_not_fetched_unless_asked(world, tmp_path,
                                                       capsys):
    machine = campaign.Machine.from_entry('slow', {
        'scheduler': 'shell', 'root': str(world['root'] / 'slow')})
    out = world['root'] / 'slow' / 'runs' / 'paper' / 'slow'
    out.mkdir(parents=True)
    (out / 'STATUS').write_text('running')
    campaign.write_campaign(world['here'], {
        'name': 'paper', 'commit': 'x', 'machines': {'slow': {
            'machine': dataclasses.asdict(machine), 'job': '999999',
            'out': str(out), 'submitted': 0}}})
    assert campaign.main(['fetch', str(world['here'])]) == 0
    assert 'not fetched' in capsys.readouterr().out
    assert not (world['here'] / 'slow' / 'eval').exists()
    assert campaign.main(['fetch', str(world['here']), '--partial']) == 0
    assert (world['here'] / 'slow' / 'eval' / 'STATUS').exists()


def test_changes_are_shipped_only_when_asked(world, tmp_path):
    (world['repo'] / 'tools' / 'bench' / 'evaluate.py').write_text('# new\n')
    with pytest.raises(SystemExit, match='--dirty'):
        campaign.snapshot(world['repo'], dirty=False)
    head = subprocess.run(['git', '-C', str(world['repo']), 'rev-parse',
                           'HEAD'], capture_output=True, text=True,
                          check=True).stdout.strip()
    sha = campaign.snapshot(world['repo'], dirty=True)
    assert sha != head
    shown = subprocess.run(['git', '-C', str(world['repo']), 'show',
                            f'{sha}:tools/bench/evaluate.py'],
                           capture_output=True, text=True, check=True)
    assert shown.stdout == '# new\n'


def test_a_dry_run_prints_and_touches_nothing(world, tmp_path, capsys):
    machines = _machines(world, tmp_path / 'm.json', cluster={
        'host': 'login', 'root': str(world['root'] / 'cluster')})
    assert campaign.main(['submit', str(machines), '--campaign',
                          str(world['here']), '--repo', str(world['repo']),
                          '--dry-run']) == 0
    assert '#SBATCH --time=01:00:00' in capsys.readouterr().out
    assert not world['root'].exists() and not world['here'].exists()
