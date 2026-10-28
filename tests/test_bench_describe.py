# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""The formula read off an export is the export's.

A table of kernels in a paper is read as a statement about the code, and a
formula in it that the code does not compute is worse than none.  So what is
pinned here is that the reconstruction loses nothing -- every tensor the
export names appears in its formula -- that it folds what it claims to fold
and nothing else, and that the hand-written part (`ROLES`) names only kernels
an export actually has.

Host-only: no GPU, no toolchain.  The figures need TensorForge's cost model,
which the tests have.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / 'tools' / 'bench'
if str(BENCH) not in sys.path:
    sys.path.insert(0, str(BENCH))

import describe  # noqa: E402

DATA = BENCH / 'data'


def _one(export, kernel, costs=False):
    got = describe.describe_all([DATA / f'{export}.json'], [kernel],
                                costs=costs)
    assert len(got) == 1, kernel
    return got[0]


def _text(d):
    return describe.lines(d.formula, describe.to_text)


# ----------------------------------------------------------------------
# What the reconstruction says
# ----------------------------------------------------------------------

@pytest.mark.parametrize('kernel,expected', [
    ('gpu_volume',
     ['Q[kp] += sum_i kDivM(i)[kl] I[lq] star(i)[qp]  (i in 0..2)']),
    ('gpu_localFluxAll',
     ['Q[kp] += sum_i rDivM(i)[km] fMrT(i)[ml] I[lq] AplusTAll(i)[qp]'
      '  (i in 0..3)']),
    ('gpu_neighboringFlux[0]',
     ['Q[kp] += rDivM(0)[km] fPrT(0)[ml] I[lq] AminusT[qp]']),
    ('gpu_derivativeTaylorExpansion',
     ['I[ab] = sum_i power(i) dQ(i)[ab]  (i in 0..3)']),
])
def test_a_chain_and_its_faces_read_as_one_sum(kernel, expected):
    """Three and twelve operations in the export, one statement here: the
    temporaries put back, the faces or directions one sum."""
    assert _text(_one('elastic-4-d-s1', kernel)) == expected


def test_the_derivatives_read_as_a_recursion():
    """Each derivative is computed from the one before and the time
    integral is accumulated between them; the formula says the rule once
    and the integral once."""
    assert _text(_one('elastic-4-d-s1', 'gpu_derivative')) == [
        'dQ(0)[ab] = Q[ab]',
        'dQ(j+1)[kp] = sum_i kDivMT(i)[kl] dQ(j)[lq] star(i)[qp]'
        '  (i in 0..2)  for j in 0..2',
        'I[ab] = sum_i power(i) dQ(i)[ab]  (i in 0..3)',
    ]


def test_a_recursion_of_several_statements_is_one_loop():
    d = _one('viscoelastic-6-s-s1', 'gpu_derivative')
    loops = [x for x in d.formula if isinstance(x, describe.Loop)]
    assert len(loops) == 1
    assert loops[0].count == 5 and len(loops[0].body) == 5


def test_the_yateto_spelling_is_yateto():
    d = _one('elastic-4-d-s1', 'gpu_localFluxAll')
    assert describe.lines(d.formula, describe.to_yateto) == [
        "Q['kp'] <= Q['kp'] + sum(rDivM[i]['km'] * fMrT[i]['ml'] * I['lq']"
        " * AplusTAll[i]['qp'] for i in range(4))"]


def test_the_paper_spelling_uses_the_notation_and_balances():
    d = _one('elastic-4-d-s1', 'gpu_derivative')
    tex = describe.lines(d.formula,
                         lambda s: describe.to_latex(s, describe.NOTATION))
    assert r'\partial_t^{j+1} Q_{kp}' in tex[1]
    for line in tex:
        assert line.count('{') == line.count('}'), line


# ----------------------------------------------------------------------
# What it must not lose
# ----------------------------------------------------------------------

def _names(formula):
    out = set()
    for item in formula:
        for st in (item.body if isinstance(item, describe.Loop) else [item]):
            for name in [st.dest.name, *(f.name for f in st.factors)]:
                out.add(describe._ANY_FAMILY.sub(
                    lambda m: m['base'], name))
    return out


@pytest.mark.parametrize('export', ['elastic-4-d-s1', 'viscoelastic-4-d-s1',
                                    'elastic-3-s-s8'])
def test_every_tensor_of_the_export_is_in_its_formula(export):
    """A tensor the kernel reads or writes and the formula does not name is
    a formula of another kernel.  Temporaries may vanish -- that is the
    point -- and so may a member 0 the recursion's copy stands for."""
    blob = json.loads((DATA / f'{export}.json').read_text())
    for name, description in blob['descriptions'].items():
        d = describe.describe(name, description, costs=False)
        used = {op_ref['name'] for op in description['operations']
                for op_ref in [op['result'], *op['args']]}
        temps = set(describe.temporaries(description))
        bases = {describe._ANY_FAMILY.sub(lambda m: m['base'], n)
                 for n in used - temps}
        missing = bases - _names(d.formula)
        assert not missing, (name, missing)


def test_every_role_names_a_kernel_some_export_has():
    """The hand-written part is the part that goes stale."""
    families = set()
    for path in sorted(DATA.glob('*.json')):
        families |= {describe.family_of(n) for n in
                     json.loads(path.read_text())['descriptions']}
    assert set(describe.ROLES) <= families, set(describe.ROLES) - families


# ----------------------------------------------------------------------
# The figures beside it
# ----------------------------------------------------------------------

def test_the_counts_are_the_cost_model_s():
    """The same 40 680 dense and 25 092 needed flops the README quotes for
    `gpu_localFluxAll`, elastic order 4."""
    d = _one('elastic-4-d-s1', 'gpu_localFluxAll', costs=True)
    assert (d.flops, d.nonzero_flops) == (40680, 25092)


def test_an_operator_every_element_shares_is_not_an_element_s_byte():
    """Read once per launch, it is no term at the batches worth measuring;
    counting it per element would make the kernel look memory-bound by a
    constant that is the launch's."""
    from tensorforge.analysis.cost import list_cost
    from tensorforge.common.basic_types import Datatype
    from tensorforge.frontend.yateto import DescriptionReader
    d = _one('elastic-4-d-s1', 'gpu_localFluxAll', costs=True)
    blob = json.loads((DATA / 'elastic-4-d-s1.json').read_text())
    descrs = DescriptionReader(arch=None).read(
        blob['descriptions']['gpu_localFluxAll'])[0]
    at = list_cost(descrs, batch=1000, datatype=Datatype.F64)
    assert d.shared_bytes > 0
    assert at.bytes == 1000 * d.bytes + d.shared_bytes
    assert d.intensity == pytest.approx(d.flops / d.bytes)
