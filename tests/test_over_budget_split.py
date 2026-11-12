# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""The register budget of a split sub-group (`tuning._over_budget`).

Under SPMD on Intel one hardware thread holds a whole sub-group, so what has
to fit its register file is a lane's footprint times the sub-group.  Eight
lanes on PVC (`Options.lanes_split_floor`) are two multiplications in one
16-wide sub-group: the thread holds both, and the budget has to count sixteen
lanes, not eight.
"""

from __future__ import annotations

from types import SimpleNamespace

from tensorforge.common.basic_types import Datatype
from tensorforge.common.context import Context
from tensorforge.generators import tuning


def _result(lanes: int, peak: float, arch: str = 'pvc',
            backend: str = 'oneapi'):
    ctx = Context(arch=arch, backend=backend, fp_type=Datatype.F32)
    metrics = SimpleNamespace(peak_pressure=peak, peak_uniform_pressure=0)
    gen = SimpleNamespace(lanes=SimpleNamespace(num_threads=lanes),
                          metrics=metrics)
    return SimpleNamespace(context=ctx, generator=gen, ok=True)


def test_a_split_sub_group_counts_both_multiplications():
    file = Context(arch='pvc', backend='oneapi',
                   fp_type=Datatype.F32).target.hw.max_reg_per_thread
    assert tuning._over_budget(_result(16, 600)) == 600 * 16 - file
    assert tuning._over_budget(_result(8, 600)) == 600 * 16 - file


def test_a_multiplication_with_a_sub_group_of_its_own_counts_its_lanes():
    file = Context(arch='pvc', backend='oneapi',
                   fp_type=Datatype.F32).target.hw.max_reg_per_thread
    assert tuning._over_budget(_result(32, 300)) == 300 * 32 - file
    assert tuning._over_budget(_result(16, 300)) == 0
