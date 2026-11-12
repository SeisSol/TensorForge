# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""The probe of `fission='auto'` leaves the operands stored as it found them
(`Generator._decide_fission`).

The probe is a sibling built whole, on the caller's tensors, and a build
states its storage on them: orders, packings, and through a merged run's
stand-in its members'.  Left as the probe stored them, the build that
follows decided nothing of its own: on PVC, where `fission='auto'` is the
default, the lanes rule's picks with the face variants of SeisSol's
`localFluxAll` merged came out NaN at orders 6 to 8 (sng2 560690), and the
same kernel built with `fission=0` or `fission=2` was right.  A probe that
decides on one piece must leave the build exactly as without it.
"""
from __future__ import annotations

import re

import pytest

from tensorforge.common.basic_types import Datatype
from tensorforge.common.context import Context
from tensorforge.common.options import Options
from tensorforge.generators import generator as generator_mod
from tensorforge.generators.generator import Generator


def _build(fission, merge=True, order=4):
    import seissol_suite as fx
    from tensorforge.frontend.yateto import DescriptionReader
    descrs = DescriptionReader(None, {}).read(fx.description(
        'elastic-linearck', f'elastic-linearck-o{order}-d',
        'gpu_localFluxAll'))[0]
    ctx = Context(arch='pvc', backend='oneapi', fp_type=Datatype.F64,
                  options=Options(merge_variants=merge, fission=fission))
    gen = Generator(descrs, ctx)
    gen.generate()
    stored = {t.name: repr(state) for t, state in gen._storage.values()}
    # The name hashes the options asked for, `fission` among them, and a
    # comment lists them.
    text = re.sub(r'kernel_kernel_[0-9a-f]+', 'K', gen.get_kernel())
    text = re.sub(r'\n *// options:[^\n]*', '', text)
    return text, stored


@pytest.mark.parametrize('merge', [True, False])
def test_a_probe_that_keeps_one_piece_changes_nothing(monkeypatch, merge):
    # Probed whatever its size, and kept whole whatever its code.
    monkeypatch.setattr(generator_mod, 'FISSION_PROBE_FLOPS', 0)
    monkeypatch.setitem(generator_mod.FISSION_CODE, 'intel', 1 << 40)
    probed, probed_storage = _build('auto', merge)
    plain, plain_storage = _build(0, merge)
    assert probed_storage == plain_storage
    assert probed == plain
