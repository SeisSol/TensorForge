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


# -- the paths' plans ------------------------------------------------------------ #

def test_the_paths_state_their_products_as_plans():
    """Each path's products are one plan's (`pir.terms`), in the order it
    issued them before: NVIDIA's largest first, the AMD paths' smallest
    first, Intel's smallest first with B's half breaking a tie."""
    from tensorforge.backend.instructions.compute.primitives import (
        intel, intel_spmd)
    from tensorforge.backend.instructions.compute.primitives.amd import \
        stacked
    from tensorforge.generators import element_group
    assert nvidia.TF32_PLAN.products == ((0, 0), (0, 1), (1, 0))
    assert nvidia.TF32_TERMS == nvidia.TF32_PLAN.products
    assert element_group.TF32_HALVES_BY_WHOLE.products == ((0, 0), (1, 0))
    assert stacked.BF16X6.products == split.products(3)
    assert intel.TF32_PLAN.products == ((1, 0), (0, 1), (0, 0))
    assert intel_spmd.PLAN == intel.TF32_PLAN
    assert [(q, p) for p, q in intel_spmd.PRODUCTS] == list(
        intel_spmd.PLAN.products)
    for plan in (nvidia.TF32_PLAN, element_group.TF32_HALVES_BY_WHOLE,
                 stacked.BF16X6, intel.TF32_PLAN):
        assert plan.count == 1 and plan.epilogue is terms.Epilogue.NONE

    class _Stage:
        def __init__(self, split):
            self.split = split
    assert element_group.async_plan(_Stage(True)) is nvidia.TF32_PLAN
    assert element_group.async_plan(_Stage(False)) is \
        element_group.TF32_HALVES_BY_WHOLE
    assert element_group.async_terms(_Stage(False)) == ((0, 0), (1, 0))


def test_nvidia_gives_each_product_its_accumulator_below_chains_min():
    """`MMAInstr.plan_for`: below `CHAINS_MIN` independent chains a step's
    products each take an accumulator of their own, summed behind them --
    corrections first (`TermPlan.close`)."""
    assert ATOM.plan is nvidia.TF32_PLAN and ATOM.terms == 3
    assert ATOM.plan_for(nvidia.CHAINS_MIN - 1) == nvidia.TF32_PLAN.each()
    assert ATOM.plan_for(nvidia.CHAINS_MIN) is nvidia.TF32_PLAN
    dmma = next(i for i in nvidia.INSTRS if i.mode == nvidia.MMAMode.DIRECT)
    assert dmma.plan is None and dmma.plan_for(1) is None and dmma.terms == 1


# -- the generated kernels, read back -------------------------------------------- #

def _case(name):
    import importlib.util
    from pathlib import Path
    path = Path(__file__).parent / 'cases' / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.descr_list(), getattr(mod, 'OPTIONS', {})


def _seissol(kernel, order=6, precision='s'):
    import seissol_suite as suite
    from tensorforge.frontend.yateto import DescriptionReader
    system = 'elastic-linearck'
    return DescriptionReader(None, {}).read(suite.description(
        system, f'{system}-o{order}-{precision}', kernel))[0], {}


def _me(shape, order=6):
    import test_multi_element as me
    return me._descrs(shape, order, 's'), {}


def _stacked_gemm(rows=16, depth=16, columns=4):
    """`C = A B` with `A` batch-constant and its numbers known, `B` and `C`
    an element's: at sixteen lanes a multiplication and with
    `Options.mma_stack_mults`, NVIDIA's `mma.sync` stacks the warp's two
    multiplications along N (`nvidia._stacked`) -- one tile of rows, eight
    columns, one chain: each product into an accumulator of its own
    (`MMAInstr.plan_for`)."""
    import numpy as np
    from tensorforge.common.basic_types import Addressing
    from tensorforge.common.matrix.boundingbox import BoundingBox
    from tensorforge.common.matrix.tensor import SubTensor, Tensor
    from tensorforge.generators.descriptions import GemmDescr
    f32 = Datatype.F32
    numbers = np.random.default_rng(1).uniform(-1, 1, (rows, depth))
    a = SubTensor(Tensor([rows, depth], Addressing.NONE,
                         BoundingBox([0, 0], [rows, depth]), alias='A',
                         datatype=f32, data=numbers))
    b = SubTensor(Tensor([depth, columns], Addressing.STRIDED,
                         BoundingBox([0, 0], [depth, columns]), alias='B',
                         datatype=f32))
    c = SubTensor(Tensor([rows, columns], Addressing.STRIDED,
                         BoundingBox([0, 0], [rows, columns]), alias='C',
                         datatype=f32))
    return [GemmDescr(trans_a=False, trans_b=False, a=a, b=b, c=c,
                      alpha=1.0, beta=0.0)], {}


#: The multiplications of a warp stacked along N (`nvidia._stacked`).
STACKED = {'tensor_cores': True, 'mma_stack_mults': True,
           'lanes_per_mult': 16}


def _generated(descrs, arch, backend, options, emulation=False):
    import contextlib
    import io
    import warnings
    from tensorforge.common.context import Context, Options
    from tensorforge.generators.generator import Generator
    from tensorforge.backend.instructions.compute.primitives.amd import \
        tiling
    ctx = Context(arch=arch, backend=backend, fp_type=Datatype.F32,
                  options=Options(**options))
    before = tiling.EMULATION
    tiling.EMULATION = emulation or before
    try:
        gen = Generator(descrs, ctx)
        with warnings.catch_warnings(), \
                contextlib.redirect_stdout(io.StringIO()):
            warnings.simplefilter('ignore')
            gen.generate()
    finally:
        tiling.EMULATION = before
    return [section.body for section in gen.built._sections]


#: A configuration of every path that issues an emulated product: what it
#: generates, where, and the plan its products carry.
PATHS = {
    'mma.sync': (lambda: _seissol('gpu_volume'), 'sm_86', 'cuda',
                 {'tensor_cores': True}, '3xtf32'),
    'mma.sync, a chain': (lambda: _seissol('gpu_localFluxAll'), 'sm_86',
                          'cuda', {'tensor_cores': True,
                                   'mma_d_layout': True}, '3xtf32'),
    'mma.sync, a product an accumulator': (
        lambda: _case('accumulate_then_read.py'), 'sm_86', 'cuda',
        {'tensor_cores': True}, '3xtf32/each'),
    'mma.sync, both': (lambda: _case('chain_five.py'), 'sm_90a', 'cuda',
                       {'tensor_cores': True}, '3xtf32'),
    'mma.sync, multiplications stacked along N': (
        _stacked_gemm, 'sm_86', 'cuda', STACKED, '3xtf32/each'),
    'multi-element mma_sync': (lambda: _me('volume'), 'sm_120', 'cuda',
                               {'multi_element': 'force'}, '3xtf32'),
    'wgmma': (lambda: _me('volume'), 'sm_90a', 'cuda',
              {'multi_element': 'force', 'matrix_engine': 'auto'},
              '3xtf32'),
    'tcgen05': (lambda: _me('volume'), 'sm_100a', 'cuda',
                {'multi_element': 'force', 'matrix_engine': 'auto'},
                '3xtf32'),
    'tcgen05, prepared ring': (lambda: _me('volume', 8), 'sm_100a', 'cuda',
                               {'multi_element': 'force',
                                'matrix_engine': 'auto',
                                'matrix_ring': 'prepared'}, '3xtf32'),
    'tcgen05, folded star': (lambda: _me('volume', 8), 'sm_100a', 'cuda',
                             {'multi_element': 'force',
                              'matrix_engine': 'auto',
                              'matrix_ring': 'prepared',
                              'matrix_star': 'engine'}, '3xtf32'),
    'wgmma, B whole': (lambda: _me('volume'), 'sm_90a', 'cuda',
                       {'multi_element': 'force', 'matrix_engine': 'auto',
                        'matrix_tf32_operands': True}, None),
    'AMD stacked, emulated': (lambda: _case('stacked_volume_f32.py'),
                              'gfx942', 'hip',
                              {'amd_matrix_k4_emulate': True}, 'bf16x6'),
    'AMD stacked, emulated, chained': (
        lambda: _seissol('gpu_volume'), 'gfx942', 'hip',
        {'amd_matrix_k4': True, 'amd_matrix_k4_emulate': True,
         'mma_d_layout': True, 'lanes_per_mult': 16}, 'bf16x6'),
    'AMD lane-batched, emulated': (lambda: _case('local_flux.py'), 'gfx942',
                                   'hip', {}, 'bf16x6'),
    'Intel SPMD DPAS': (lambda: _seissol('gpu_volume'), 'pvc', 'oneapi',
                        {'sycl_dpas': True}, '3xtf32/b'),
    # DPAS under the explicit vector only with the transfers in place
    # (`intel.moves_refused`).
    'Intel ESIMD DPAS': (lambda: _case('local_flux.py'), 'pvc', 'esimd',
                         {'tensor_cores': True, 'lanes_per_mult': 16,
                          'prepare_operands': False,
                          'enable_move_loads': False}, '3xtf32/b'),
    'Intel ESIMD DPAS, prepared': (lambda: _case('local_flux.py'), 'pvc',
                                   'esimd', {'tensor_cores': True,
                                             'lanes_per_mult': 16,
                                             'enable_move_loads': False},
                                   '3xtf32/b'),
}


@pytest.mark.parametrize('path', sorted(PATHS))
def test_every_emulated_product_a_kernel_issues_is_its_plans(path):
    """Generated over the path, every matrix op of an emulated product
    carries its plan, and every accumulator takes the plan's products whole
    and in order (`check_plans`)."""
    make, arch, backend, options, name = PATHS[path]
    descrs, case_options = make()
    bodies = _generated(descrs, arch, backend,
                        dict(case_options, **options),
                        emulation=path == 'AMD lane-batched, emulated')
    from tensorforge.backend.pir.core import walk_stmts
    planned = [s for body in bodies for s in walk_stmts(body)
               if s.op == Op.MATRIX_MAC and s.attr('plan') is not None]
    assert planned, f'{path}: no emulated product generated'
    if name is not None:
        assert name in {s.attr('plan').name for s in planned}
    assert [f for body in bodies for f in terms.check_plans(body)] == []


@pytest.mark.parametrize('arch', ['sm_86', 'sm_90a'])
def test_each_product_of_a_lone_chain_takes_its_own_accumulator(arch):
    """Where a step has a single chain -- the warp's two multiplications
    stacked along N into one tile (`nvidia._stacked`) -- each of 3xTF32's
    products adds into an accumulator of its own (`TermPlan.each`): the
    registers of C a product writes are its term's, no other term's, and
    every term has a set.  Summed behind the products, accumulators shared
    give the same numbers within the tolerance, so it is the registers that
    tell."""
    descrs, _ = _stacked_gemm()
    bodies = _generated(descrs, arch, 'cuda', STACKED)
    from tensorforge.backend.pir.core import walk_stmts
    from tensorforge.backend.pir.engines import fragments
    written = {}
    for body in bodies:
        for s in walk_stmts(body):
            if s.op != Op.MATRIX_MAC or s.attr('plan') is None:
                continue
            assert s.attr('plan') == nvidia.TF32_PLAN.each()
            regs = frozenset(v.id for v in fragments(s)[2])
            written.setdefault(s.attr('term'), set()).add(regs)
    assert set(written) == set(nvidia.TF32_PLAN.products)
    for term, regs in written.items():
        for other, theirs in written.items():
            if other != term:
                assert not (regs & theirs), (term, other)
