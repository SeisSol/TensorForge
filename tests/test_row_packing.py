# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""`Options.row_packing`: what a matrix product does with the rows an
element brings that do not fill the instruction's tile (`compute.rowpack`).

`auto` is each target's own rule, the behavior before the option; the SPMD
lowering's DPAS lowers `fma` (the remainder carried on FMAs) and `pad` (a
tile of its own), the explicit vector's DPAS `pad`; elsewhere only `auto`
is taken, and every other packing is refused by name.
"""

import contextlib
import io
import re
import warnings

import pytest

import seissol_suite as suite
from tensorforge.backend.instructions.compute import rowpack
from tensorforge.backend.instructions.compute.primitives import intel_spmd
from tensorforge.backend.instructions.compute.rowpack import (Exchange,
                                                              Packing, RowCaps)
from tensorforge.common.basic_types import Datatype
from tensorforge.common.context import Context, Options
from tensorforge.common.exceptions import GenerationError
from tensorforge.frontend.yateto import DescriptionReader
from tensorforge.generators.generator import Generator


@pytest.mark.parametrize('quantities, height, elements', [
    (9, 8, 8),      # SeisSol's elastic ninth quantity on DPAS: 8 elements
    (21, 8, 8),     # viscoelastic, two mechanisms: 5 left, 8 elements
    (27, 8, 8),     # three mechanisms: 3 left, 8 elements, 3 tiles
    (9, 16, 16),    # nine on MFMA's 16: 16 elements, 9 tiles
    (12, 8, 2),     # 4 left: two elements fill a tile
    (10, 8, 4),     # 2 left: four elements
    (16, 8, 1),     # nothing left
])
def test_gather_fills_whole_tiles(quantities, height, elements):
    assert rowpack.gather_elements(quantities, height) == elements
    r = quantities % height
    if r:
        assert elements * r % height == 0


def _caps(auto=Packing.FMA, exchanges=frozenset()):
    return RowCaps(8, frozenset({Packing.FMA, Packing.PAD}), auto, exchanges)


def _ctx(backend='oneapi', arch='pvc', **options):
    return Context(arch=arch, backend=backend, fp_type=Datatype.F32,
                   options=Options(**options))


def test_auto_is_the_targets_own():
    assert rowpack.plan_rows(9, _caps(), _ctx()).packing is Packing.FMA
    assert rowpack.plan_rows(9, _caps(Packing.PAD), _ctx()).packing \
        is Packing.PAD


def test_a_packing_the_target_does_not_lower_is_refused():
    with pytest.raises(GenerationError, match='lowers fma, pad'):
        rowpack.plan_rows(9, _caps(), _ctx(row_packing='gather'))
    with pytest.raises(GenerationError, match='only auto'):
        rowpack.plan_rows(9, None, _ctx(row_packing='pad'))
    with pytest.raises(GenerationError, match='one of'):
        rowpack.plan_rows(9, _caps(), _ctx(row_packing='diagonal'))


def test_no_remainder_asks_nothing():
    plan = rowpack.plan_rows(16, _caps(), _ctx(row_packing='gather'))
    assert plan.remainder == 0 and plan.why == 'no remainder'


def test_gather_takes_its_elements_and_an_exchange_the_target_has():
    caps = RowCaps(8, frozenset({Packing.GATHER}), Packing.GATHER,
                   frozenset({Exchange.SHARED}))
    plan = rowpack.plan_rows(9, caps, _ctx())
    assert (plan.elements, plan.exchange) == (8, Exchange.SHARED)
    plan = rowpack.plan_rows(9, caps, _ctx(row_packing_elements=16))
    assert plan.elements == 16
    with pytest.raises(GenerationError, match='through'):
        rowpack.plan_rows(9, caps, _ctx(row_packing_exchange='item'))


def test_the_spmd_caps_are_todays_rule():
    """Up to `TAIL_MAX` columns past the last tile carried, more padded, one
    tile or less nothing to carry."""
    assert intel_spmd.row_caps(9).auto is Packing.FMA
    assert intel_spmd.row_caps(10).auto is Packing.FMA
    assert intel_spmd.row_caps(11).auto is Packing.PAD
    assert intel_spmd.row_caps(8).packings == frozenset({Packing.PAD})
    assert intel_spmd.row_caps(9).packings == frozenset({Packing.FMA,
                                                         Packing.PAD})


def _volume(backend, order=6, **options):
    descrs = DescriptionReader(None, {}).read(suite.description(
        'elastic-linearck', f'elastic-linearck-o{order}-s', 'gpu_volume'))[0]
    gen = Generator(descrs, _ctx(backend, **options))
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter('ignore')
        gen.generate()
    return gen.get_kernel()


def _same(a, b):
    """Two kernels alike but for their names and the options comment,
    which follow the options asked."""
    def norm(text):
        text = re.sub(r'kernel_[0-9a-f]{16}', 'K', text)
        return re.sub(r'// options: [^\n]*', '', text)
    return norm(a) == norm(b)


def test_spmd_dpas_fma_is_auto_and_pad_is_a_tile():
    """SeisSol's order-6 volume under `sycl_dpas`: `fma` is the default's
    kernel; `pad` runs the ninth quantity as a tile of its own -- twice the
    DPAS of the operator products, and no broadcasts for a carried column."""
    auto = _volume('oneapi', sycl_dpas=True)
    assert _same(_volume('oneapi', sycl_dpas=True, row_packing='fma'), auto)
    pad = _volume('oneapi', sycl_dpas=True, row_packing='pad')
    assert pad.count('dpasTF32(') == 2 * auto.count('dpasTF32(')
    assert '_dbc = sycl::group_broadcast' in auto
    assert '_dbc = sycl::group_broadcast' not in pad


def test_the_explicit_vector_pads():
    options = dict(tensor_cores=True, lanes_per_mult=16,
                   enable_move_loads=False)
    auto = _volume('esimd', **options)
    assert _same(_volume('esimd', row_packing='pad', **options), auto)
    with pytest.raises(GenerationError, match='lowers pad'):
        _volume('esimd', row_packing='fma', **options)


def test_targets_without_a_rule_take_only_auto():
    with pytest.raises(GenerationError, match='states no packing'):
        _volume('cuda', order=4, arch='sm_86', row_packing='gather')
