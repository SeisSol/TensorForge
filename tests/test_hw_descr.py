# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""What `HwDecription` reads off a model name.

`sm_level` is what every NVIDIA architecture floor in the generator compares
against -- the instruction size, packed FP32 FMA, the prefetch helpers -- so a
misread model moves all of them at once.  `sm_100` sorts below `sm_60` under a
two-character read of the suffix, and that read is the plausible way to get it
wrong.
"""

from __future__ import annotations

import pytest

from tensorforge.common.vm.hw_descr import hw_descr_factory


@pytest.mark.parametrize("arch,expected", [
    ("sm_60", 60), ("sm_86", 86), ("sm_100", 100), ("sm_120", 120)])
def test_sm_level_reads_every_digit(arch, expected):
    assert hw_descr_factory(arch, "cuda").sm_level() == expected


def test_amd_model_has_no_compute_capability():
    assert hw_descr_factory("gfx90a", "hip").sm_level() is None
