# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""The term plan (`pir.terms`): an emulated product as the matrix ops of its
terms, stated once and read back off a body.

Three kinds of check:

* the arithmetic -- `split.products` gives today's lists where it is asked
  as it was, and the orders the paths issue in where it is asked for them;
* the plan -- which products, into which accumulator, and the epilogue,
  through `issue` and `close` on a recorder;
* the reading back (`check_plans`) -- a body whose accumulators take their
  plans' products whole and in order passes, and one that drops, repeats or
  reorders a product, mixes partials or multiplies a term in another
  format is a finding; the plan on a statement writes no other text.
"""

from __future__ import annotations

import pytest

from tensorforge.backend.instructions.compute import split
from tensorforge.backend.instructions.compute.primitives import nvidia
from tensorforge.backend.pir import engines, terms
from tensorforge.backend.pir.build import IRBuilder
from tensorforge.backend.pir.core import Op, ScalarType, walk
from tensorforge.backend.pir.emit import Emitter
from tensorforge.common.basic_types import Datatype

TF32 = terms.Form(Datatype.TF32, 2)
BF16 = terms.Form(Datatype.BF16, 3)
WHOLE = terms.Form(Datatype.TF32, 1)

ATOM = next(i for i in nvidia.INSTRS if i.name.startswith(
    'mma.sync.aligned.m16n8k8.row.col.f32'))


# -- the arithmetic ------------------------------------------------------------ #

def _before(count, keep=None):
    """`split.products` as it was written before the orders were stated."""
    keep = count if keep is None else keep
    pairs = [(i, j) for i in range(count) for j in range(count)
             if i + j < keep]
    return tuple(sorted(pairs, key=lambda p: (-(p[0] + p[1]), p)))


@pytest.mark.parametrize('count', [1, 2, 3, 4])
@pytest.mark.parametrize('keep', [None, 1, 2, 3])
def test_products_asked_as_before_are_what_they_were(count, keep):
    assert split.products(count, keep) == _before(count, keep)


def test_the_three_orders_the_paths_issue_in():
    """NVIDIA's 3xTF32 largest first, the AMD paths' and the default
    smallest first, Intel's smallest first with B's term breaking a tie --
    `(A's term, B's term)` in each."""
    assert split.products(2, order='largest') == ((0, 0), (0, 1), (1, 0))
    assert split.products(2) == ((0, 1), (1, 0), (0, 0))
    assert split.products(2, major='b') == ((1, 0), (0, 1), (0, 0))
    assert [(q, p) for p, q in split.products(2)] == list(
        split.products(2, major='b'))


def test_sides_of_different_counts():
    """A in two halves against B whole: two products, `keep` the larger."""
    assert split.products(2, counts=(2, 1), order='largest') == ((0, 0),
                                                                (1, 0))
    assert split.products(3, counts=(3, 2)) == ((1, 1), (2, 0), (0, 1),
                                                (1, 0), (0, 0))


@pytest.mark.parametrize('bad', [{'order': 'random'}, {'major': 'c'}])
def test_an_order_it_does_not_know_is_refused(bad):
    with pytest.raises(ValueError):
        split.products(2, **bad)


# -- the plan ------------------------------------------------------------------ #

def test_a_plan_is_split_products_into_one_accumulator():
    p = terms.plan('bf16x6', BF16, BF16)
    assert p.products == split.products(3)
    assert p.partials == (0,) * 6 and p.count == 1
    assert p.epilogue is terms.Epilogue.NONE
    assert p.a.covered >= 24 and p.a.significand == 8


def test_each_product_into_an_accumulator_of_its_own():
    p = terms.plan('tf32x3', TF32, TF32, order='largest').each()
    assert p.partials == (0, 1, 2) and p.count == 3
    assert p.epilogue is terms.Epilogue.SUM
    assert p.sequence(1) == ((0, 1),)
    assert p.each() is p


@pytest.mark.parametrize('bad', [
    dict(products=((0, 0), (0, 0)), partials=(0, 0)),
    dict(products=((0, 2),), partials=(0,)),
    dict(products=((0, 0), (0, 1)), partials=(0,)),
    dict(products=((0, 0), (0, 1)), partials=(0, 2),
         epilogue=terms.Epilogue.SUM),
    dict(products=((0, 0), (0, 1)), partials=(0, 1)),
    dict(products=((0, 0),), partials=(0,), epilogue=terms.Epilogue.SUM),
])
def test_a_plan_that_says_something_impossible_is_refused(bad):
    with pytest.raises(ValueError):
        terms.TermPlan('bad', TF32, TF32, **bad)


def test_a_plan_is_a_hashable_attribute():
    """A statement's attributes are part of its identity to the passes."""
    p = terms.plan('tf32x3', TF32, TF32, order='largest')
    assert hash(p) == hash(terms.plan('tf32x3', TF32, TF32, order='largest'))
    assert p != p.each()


def test_issue_takes_every_product_in_order_into_its_partial():
    p = terms.plan('tf32x3', TF32, TF32, order='largest').each()
    seen = []

    def mac(a, b, c, term):
        seen.append((a, b, c, term))
        return c + (term,)
    accs = p.issue(mac, ['a0', 'a1'], ['b0', 'b1'], [(), (), ()])
    assert seen == [('a0', 'b0', (), (0, 0)), ('a0', 'b1', (), (0, 1)),
                    ('a1', 'b0', (), (1, 0))]
    assert accs == [((0, 0),), ((0, 1),), ((1, 0),)]


def test_issue_in_place_keeps_the_accumulator():
    p = terms.plan('bf16x6', BF16, BF16)
    calls = []
    accs = p.issue(lambda a, b, c, t: calls.append(t), 'abc', 'xyz', ['c'])
    assert accs == ['c'] and calls == list(split.products(3))


def test_issue_wants_an_accumulator_a_partial():
    p = terms.plan('tf32x3', TF32, TF32).each()
    with pytest.raises(ValueError):
        p.issue(lambda *x: None, 'ab', 'ab', ['c'])


def test_close_adds_the_corrections_first():
    """`p0 + (p1 + p2)`: what NVIDIA's chains sum before their epilogue."""
    p = terms.plan('tf32x3', TF32, TF32, order='largest')
    assert p.close(lambda x, y: (x, y), ['d']) == 'd'
    assert p.each().close(lambda x, y: f'({x} + {y})',
                          ['p0', 'p1', 'p2']) == '(p0 + (p1 + p2))'


# -- reading it back off a body -------------------------------------------------- #

PLAN = terms.plan('tf32x3', TF32, TF32, order='largest')


def _mma(steps=1, plan=PLAN, terms_=None, accumulators=1):
    """`steps` steps of the contraction of NVIDIA's `mma.sync`, in place,
    each issuing `terms_` (the plan's products by default) into
    `accumulators` sets of C, as the multilinear path issues them."""
    b = IRBuilder(fptype=Datatype.F32)
    na, nb, nc = (ATOM.m * ATOM.k // 32, ATOM.k * ATOM.n // 32,
                  ATOM.m * ATOM.n // 32)
    halves = [[b.declare(hint='a') for _ in range(na)] for _ in range(2)]
    bh = [[b.declare(hint='b') for _ in range(nb)] for _ in range(2)]
    cs = [[b.declare(hint='c') for _ in range(nc)]
          for _ in range(accumulators)]
    terms_ = plan.products if terms_ is None else terms_
    for _ in range(steps):
        for n, (i, j) in enumerate(terms_):
            c = cs[n % accumulators]
            ATOM.asmcall(b, c, halves[i], bh[j], c, term=(i, j), plan=plan)
    return b.finish()


def _mfma(chain=None, plan=None, types=None):
    """An MFMA chain (`MatrixOp.mac`'s matrix op) over BF16 terms: D of one
    the C of the next."""
    plan = plan or terms.plan('bf16x6', BF16, BF16)
    b = IRBuilder(fptype=Datatype.F32)
    acc = ScalarType(Datatype.F32, 4)
    terms4 = ScalarType(Datatype.I16, 4)
    at = [b.declare(terms4, hint='at') for _ in range(3)]
    bt = [b.declare(terms4, hint='bt') for _ in range(3)]
    c = b.declare(acc, hint='acc')
    for i, j in (plan.products if chain is None else chain):
        c = b.matrix_mac('mfma', (16, 16, 16), types or (
            Datatype.BF16, Datatype.BF16, Datatype.F32, Datatype.F32),
            (at[i],), (bt[j],), (c,), result=acc, instruction=(
                'mfma_f32_16x16x16bf16_1k'), term=(i, j), plan=plan)
    return b.finish()


def test_whole_runs_in_order_pass():
    assert terms.check_plans(_mma(steps=3)) == []
    assert terms.check_plans(_mfma()) == []


def test_partials_each_take_their_own_products():
    each = PLAN.each()
    assert terms.check_plans(_mma(steps=2, plan=each, accumulators=3)) == []


@pytest.mark.parametrize('chain,what', [
    (PLAN.products[:2], 'dropped'),
    (PLAN.products + PLAN.products[:1], 'a run cut short'),
    ((PLAN.products[1], PLAN.products[0], PLAN.products[2]), 'reordered'),
    (PLAN.products[:1] + PLAN.products, 'repeated'),
])
def test_a_product_dropped_repeated_or_reordered_is_a_finding(chain, what):
    found = terms.check_plans(_mma(terms_=chain))
    assert found and 'its runs are' in found[0], what


def test_a_product_into_another_partials_accumulator_is_a_finding():
    """Each product of the per-product plan into one accumulator."""
    found = terms.check_plans(_mma(plan=PLAN.each(), accumulators=1))
    assert found and 'partial 0' in found[0]


def test_a_term_the_plan_does_not_have_is_a_finding():
    found = terms.check_plans(_mma(terms_=((1, 1),)))
    assert found and 'does not have' in found[0]


def test_the_engine_multiplying_another_format_is_a_finding():
    found = terms.check_plans(_mfma(types=(Datatype.F16, Datatype.F16,
                                           Datatype.F32, Datatype.F32)))
    assert found and 'multiplies F16 by F16' in found[0]


def test_a_chain_of_results_is_followed_through_its_ds():
    bad = terms.plan('bf16x6', BF16, BF16).products
    found = terms.check_plans(_mfma(chain=bad[1:]))
    assert found and 'its runs are' in found[0]


def test_two_plans_into_one_accumulator_is_a_finding():
    b = IRBuilder(fptype=Datatype.F32)
    na, nb, nc = 4, 2, 4
    a = [b.declare(hint='a') for _ in range(na)]
    bb = [b.declare(hint='b') for _ in range(nb)]
    c = [b.declare(hint='c') for _ in range(nc)]
    ATOM.asmcall(b, c, a, bb, c, term=(0, 0), plan=PLAN)
    ATOM.asmcall(b, c, a, bb, c, term=(0, 0), plan=PLAN.each())
    found = terms.check_plans(b.finish())
    assert any('takes the products of' in f for f in found)


def test_a_variable_written_back_is_one_accumulator():
    """Intel's explicit vector writes each D into its accumulator
    (`IRBuilder.assign`), and the next product reads the variable."""
    p = terms.plan('tf32x3/b', TF32, TF32, major='b')
    b = IRBuilder(fptype=Datatype.F32)
    acc = b.declare(ScalarType(Datatype.F32, 8), hint='dacc')
    af = [b.declare(ScalarType(Datatype.TF32, 4), hint='a') for _ in range(2)]
    bf = [b.declare(ScalarType(Datatype.TF32, 8), hint='b') for _ in range(2)]
    for _ in range(2):
        for i, j in p.products:
            b.assign(acc, b.matrix_mac(
                'dpas_esimd', (8, 16, 8), (Datatype.TF32, Datatype.TF32,
                                           Datatype.F32, Datatype.F32),
                (af[i],), (bf[j],), (acc,), result=acc.type, hint='dp',
                materialize=False, term=(i, j), plan=p,
                modifiers=(('depth', 8),)))
    assert terms.check_plans(b.finish()) == []


def test_lists_are_read_apart():
    """A run of a K loop's body is checked in that body."""
    b = IRBuilder(fptype=Datatype.F32)
    a = [[b.declare(hint='a') for _ in range(4)] for _ in range(2)]
    bb = [[b.declare(hint='b') for _ in range(2)] for _ in range(2)]
    c = [b.declare(hint='c') for _ in range(4)]
    from tensorforge.backend.pir.core import Uniformity
    with b.for_(0, 4, 1, hint='k', uniform=Uniformity.MULT):
        for i, j in PLAN.products[:2]:
            ATOM.asmcall(b, c, a[i], bb[j], c, term=(i, j), plan=PLAN)
    found = terms.check_plans(b.finish())
    assert found and 'its runs are' in found[0]


def test_a_product_without_a_plan_is_none_of_its_business():
    b = IRBuilder(fptype=Datatype.F32)
    a = [b.declare(hint='a') for _ in range(4)]
    bb = [b.declare(hint='b') for _ in range(2)]
    c = [b.declare(hint='c') for _ in range(4)]
    ATOM.asmcall(b, c, a, bb, c, term=(1, 1))
    assert terms.check_plans(b.finish()) == []


# -- the plan on the statement ----------------------------------------------------- #

def test_the_plan_is_a_slot_of_the_matrix_op():
    s = next(s for s, _ in walk(_mma()) if s.op == Op.MATRIX_MAC)
    assert s.attr('plan') is PLAN and s.attr('term') == (0, 0)
    assert engines.verify_engine(s) == []
    bad = s.with_attr('plan', 'tf32x3')
    assert any('plan' in d for d in engines.verify_engine(bad))


def test_the_plan_writes_no_text():
    """A term of a plan is the instruction it was: the same text."""
    def body(plan):
        b = IRBuilder(fptype=Datatype.F32)
        a = [b.declare(hint='a') for _ in range(4)]
        bb = [b.declare(hint='b') for _ in range(2)]
        c = [b.declare(hint='c') for _ in range(4)]
        ATOM.asmcall(b, c, a, bb, c, term=(0, 1), plan=plan)
        lines = []
        Emitter(lines.append, None, None).run(b.finish())
        return '\n'.join(lines)
    assert body(PLAN) == body(None)


def test_the_asynchronous_engines_carry_it_through():
    """wgmma and tcgen05 hand `term` and `plan` to the matrix op."""
    from tensorforge.backend.pir.core import MemSpace
    b = IRBuilder(fptype=Datatype.F32)
    buf = b.alloc(Datatype.F32, (64 * 8,), MemSpace.SHARED, hint='bs',
                  extern='sb', arena='blk', offset=0)
    desc = b.matrix_desc(buf, 0, desc=engines.MatrixDescType(Datatype.TF32,
                                                             sbo=128))
    from tensorforge.backend.pir.core import Uniformity
    a = [b.declare(ScalarType(Datatype.TF32), hint='a',
                   uniform=Uniformity.LANE) for _ in range(4)]
    d = [b.declare(hint='d', uniform=Uniformity.LANE) for _ in range(4)]
    b.wgmma(8, a, desc, d, term=(1, 0), plan=PLAN)
    s = next(s for s, _ in walk(b.finish()) if s.op == Op.MATRIX_MAC)
    assert s.attr('plan') is PLAN and s.attr('term') == (1, 0)
