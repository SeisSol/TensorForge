# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Which configurations a command line asks a suite to be measured in
(`tools/bench/suite.py`, `select_configs`).

`run.py` and `profile.py` both take `--configs`, and a campaign uses it to
add a configuration the suite does not list -- so the rule is checked once,
here: the suite's own and the named ones of `CONFIGS`, in the order given or
added to the suite's, and an unknown label refused rather than measured as
nothing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

BENCH = Path(__file__).resolve().parent.parent / 'tools' / 'bench'
if str(BENCH) not in sys.path:
    sys.path.insert(0, str(BENCH))

import suite  # noqa: E402
from tensorforge.common.basic_types import Datatype  # noqa: E402
from tensorforge.common.context import Options  # noqa: E402


def _suite(*configs):
    return suite.Suite(name='t', workloads=(), batches=(4,),
                       configs=tuple(configs), targets=None,
                       datatypes=(Datatype.F64,))


def test_no_labels_keep_the_suites_configurations():
    spec = _suite(suite.CONFIGS['baseline'])
    assert suite.select_configs(spec, None) is spec
    assert suite.select_configs(spec, '') is spec


def test_a_label_names_the_suites_own_or_a_known_one():
    own = suite.Config('mine', Options(tensor_cores=False))
    spec = _suite(suite.CONFIGS['baseline'], own)
    chosen = suite.select_configs(spec, 'tensor-cores, mine,,tensor-cores')
    assert [c.label for c in chosen.configs] == ['tensor-cores', 'mine']
    assert chosen.configs[1] is own
    assert chosen.configs[0] is suite.CONFIGS['tensor-cores']


def test_a_plus_adds_to_the_suites_configurations():
    spec = _suite(suite.CONFIGS['baseline'])
    chosen = suite.select_configs(spec, '+tensor-cores,baseline')
    assert [c.label for c in chosen.configs] == ['baseline', 'tensor-cores']


def test_an_unknown_label_is_refused():
    with pytest.raises(ValueError, match="'nonsense'"):
        suite.select_configs(_suite(suite.CONFIGS['baseline']),
                             'baseline,nonsense')
