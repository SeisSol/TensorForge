# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""The operator product ahead of the small product it feeds
(`generators.reassociate`, `Options.reassociate_products`).

Checked on SeisSol's recorded kernels, where the pattern comes from: which
pairs are taken, what the taken ones become, and which are left alone because
the small product would grow.  That the rewritten lists compute the same is
`tools/host/corpus_check.py`'s to say, against NumPy on the original list.
"""

from __future__ import annotations

import contextlib
import io

import pytest

import seissol_suite as suite
from tensorforge.common.basic_types import Datatype
from tensorforge.common.context import Context, Options
from tensorforge.frontend.yateto import DescriptionReader
from tensorforge.generators.generator import Generator
from tensorforge.generators.reassociate import reassociate

SYSTEM = 'elastic-linearck'


def _descrs(config, kernel):
    return DescriptionReader(None, {}).read(
        suite.description(SYSTEM, config, kernel))[0]


def _shapes(d):
    def name(view):
        return f'{view.tensor.alias}{list(view.tensor.shape)}'
    return name(d.dest), [name(o) for o in d.ops]


def test_a_face_product_comes_before_the_flux_matrix():
    """``T0 = I A+; T1 = fMrT T0`` becomes ``U = fMrT I; T1 = U A+``: the
    small product runs over the face's 24 rows instead of the element's
    56, and the four faces' first products all read I."""
    before = _descrs('elastic-linearck-o6-d', 'gpu_localFluxAll')
    after = reassociate(before)
    assert _shapes(after[0]) == ('_tmp0_ax[24, 9]',
                                 ['fMrT(0)[24, 56]', 'I[56, 9]'])
    assert _shapes(after[1]) == ('_tmp1[24, 9]',
                                 ['_tmp0_ax[24, 9]', 'AplusTAll(0)[9, 9]'])
    # The flux matrix's product stays: rDivM has 56 rows, U 24.
    assert after[2] is before[2]
    firsts = [after[i].ops[1].tensor.alias for i in (0, 3, 6, 9)]
    assert firsts == ['I'] * 4


def test_the_volume_is_left_alone_where_the_small_product_would_grow():
    """yateto cuts I to the 35 rows kDivM contracts over, and kDivM has 53
    in its box: the small product would run over more rows."""
    before = _descrs('elastic-linearck-o6-d', 'gpu_volume')
    assert all(a is b for a, b in zip(before, reassociate(before)))


@pytest.mark.parametrize('config,taken', [('elastic-linearck-o4-d', 18),
                                          ('elastic-linearck-o6-d', 30),
                                          ('elastic-linearck-o8-d', 42)])
def test_the_derivative_takes_the_steps_that_shrink(config, taken):
    before = _descrs(config, 'gpu_derivative')
    after = reassociate(before)
    assert sum(1 for a, b in zip(before, after) if a is not b) == taken


def test_without_the_option_nothing_is_rewritten():
    """The rewrite is a gate: off, the generator builds the list as given."""
    import re

    def build(**options):
        ctx = Context(arch='sm_90', backend='cuda', fp_type=Datatype.F64,
                      options=Options(**options))
        gen = Generator(_descrs('elastic-linearck-o4-d',
                                'gpu_neighboringFlux[0]'), ctx)
        with contextlib.redirect_stdout(io.StringIO()):
            gen.generate()
        src = re.sub(r'(?m)^\s*// options:.*$', '', gen.get_kernel())
        return re.sub(r'kernel_kernel_[0-9a-f]+', 'K', src)
    assert build(reassociate_products=False) == build()
    assert build(reassociate_products=True) != build()


def test_a_temporary_written_narrower_than_it_is_read_is_left_alone():
    """`temp_partial_write`: ``tmp[:, 0:6] = B C; D = A tmp`` reads twelve
    columns of a temporary its one writer fills six of -- the rest zero.
    ``(A B) C`` would put products there; on gfx1150 that was 37.7 off."""
    import importlib.util
    import pathlib
    path = (pathlib.Path(__file__).parent / 'cases' / 'slicing'
            / 'temp_partial_write.py')
    spec = importlib.util.spec_from_file_location('temp_partial_write', path)
    case = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(case)
    from tensorforge.generators.descriptions import MultilinearDescr

    def flat(descrs):
        return [op for d in descrs for op in d.operations()]

    descrs = flat(case.descr_list())
    out = reassociate(descrs)
    assert all(a is b for a, b in zip(out, descrs)) and len(out) == len(descrs)
    assert all(isinstance(d, MultilinearDescr) for d in out)
