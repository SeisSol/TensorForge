# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""The multi-element layout's asynchronous tiling in three roles of the
block's warps (`Options.matrix_roles` `three`, `pir.roles`), a demonstrator
of the roles and rings: the element warps stage their elements and compute
A into the folded star's ring (`star`); a warp behind them fills the
prepared operator ring by bulk copies (`operator`); a warpgroup behind that
issues the products and writes D (`mma`) -- meeting on the rings alone, and
on the barriers of their own roles (`multi_element._tcgen05_roles`).

What is checked here: the plan -- the option off changes nothing, the roles
where the section folds its star step into one tcgen05 step that streams B
prepared, the refusals and the kernel's note of them; the kernel's text --
the split, each role's acts and no block barrier inside it; the PIR -- the
rings' hand-offs by role, verified through every pass; the host oracle over
two trips of a group with a masked warp, against NumPy and without a race;
the build for sm_100a read back; and on a GB200 the volume against NumPy.
"""

from __future__ import annotations

import dataclasses
import re
import types

import pytest

import test_folded_star as tfs
import test_multi_element as me
import test_multi_element_async as tma
import test_ring_object as tro
from tensorforge.backend.pir.passes import verify
from tensorforge.common.basic_types import Datatype
from tensorforge.generators import element_group as eg

#: The three roles asked for, where they take: tcgen05's folded star, B
#: streamed prepared.
THREE = {'matrix_engine': 'auto', 'matrix_ring': 'prepared',
         'matrix_star': 'engine', 'matrix_roles': 'three'}


def _group(order=6, arch='sm_100a', shape='volume', **options):
    context = me._context(arch, 's', multi_element='force',
                          **{**THREE, **options})
    section, why = eg.classify(me._descrs(shape, order, 's'))
    assert section is not None, why
    group, why = eg.fill(section, context)
    assert group is not None, why
    return group, context


def _kernel(order=6, **options):
    return tma._kernel('sm_100a', 'volume', order, 's',
                       **{**THREE, **options}).get_kernel()


# -- the plan ------------------------------------------------------------------- #

def test_the_option_off_is_the_kernel_without_it():
    """`off`, the default: no roles, the kernel byte for byte the one the
    other options give."""
    group, _ = _group(matrix_roles='off')
    assert group.section.roles == '' and group.producers == 0
    plain = {k: v for k, v in THREE.items() if k != 'matrix_roles'}
    assert _kernel(**{**plain, 'matrix_roles': 'off'}) == tma._kernel(
        'sm_100a', 'volume', 6, 's', **plain).get_kernel()


@pytest.mark.parametrize('order', [4, 6, 8])
def test_the_roles_take_the_folded_tcgen05_step(order):
    """The volume's one tcgen05 step, its star folded and B streamed
    prepared, in the three roles: fourteen element warps -- the 126 rows of
    tensor memory's 128 nine columns each -- the operator's warp and the mma
    role's warpgroup behind them, and the kernel's comment says which warps
    each role takes."""
    group, _ = _group(order)
    section = group.section
    (step,) = [s for s in section.steps if s.lowering in eg.ASYNC_LOWERINGS]
    assert section.roles == 'three' and eg.folded(section, step)
    assert eg.streamed(section) and section.ring_slots == 4
    assert group.elements == 14
    assert group.producers == eg.OPERATOR_WARPS + eg.MMA_WARPS == 5
    assert group.warps == 19
    assert ('// roles: the block in three roles, meeting on the rings and on '
            'a named barrier each: warps 0 to 13 stage their elements and '
            'compute A into the folded star\'s ring (star), warp 14 fills '
            'B\'s prepared ring by bulk copies (operator), warps 15 to 18 '
            'issue the products and add D into the outputs (mma)'
            ) in _kernel(order)


@pytest.mark.parametrize('arch,shape,options,why', [
    ('sm_90a', 'volume', {}, "the step is wgmma's; three roles are "
     "tcgen05's (sm_100a) so far"),
    ('sm_100a', 'volume', {'matrix_star': 'buffer'},
     'the star step is not folded into the operator step'),
    ('sm_100a', 'volume', {'matrix_ring': 'convert'},
     'B does not stream prepared'),
    ('sm_100a', 'localFluxAll', {}, '5 asynchronous steps; the roles take '
     'one')], ids=['wgmma', 'buffer', 'convert', 'localFluxAll'])
def test_where_the_roles_do_not_take_the_kernel_says_why(arch, shape,
                                                         options, why):
    group, _ = _group(6, arch, shape, **options)
    assert group.section.roles == '' and group.producers == 0
    src = tma._kernel(arch, shape, 6, 's',
                      **{**THREE, **options}).get_kernel()
    assert f'// roles: matrix_roles=three declined: {why}' in src


def test_the_roles_add_d_a_cell_at_a_time():
    """Bulk reductions asked for: the mma role adds D a cell at a time, and
    the output's comment says so."""
    src = _kernel(matrix_output='bulk')
    assert '// roles: the block in three roles' in src
    assert ('// output: matrix_output=bulk declined for Q: the mma role adds '
            'D a cell at a time (`matrix_roles`)') in src
    assert 'cp.reduce.async.bulk' not in src


# -- the kernel ----------------------------------------------------------------- #

#: The id of a thread's role's barrier, NVIDIA's top three
#: (`roles.first_id`): 13 for the star warps, 14 the operator's, 15 the mma
#: role's.
IDENT = r'\(13 \+ \(threadIdx\.y >= 14\) \+ \(threadIdx\.y >= 15\)\)'


def _regions(src):
    """The trip's three regions of the kernel's text: the star warps', the
    operator's and the mma role's (the split's chain, `roles.chain`)."""
    loop = src[src.index('for (size_t'):src.index('tmemDealloc')]
    star = loop.index('if (v')
    operator = re.search(r'\n\s*if \(v\d+_operator\) \{', loop)
    mma = re.search(r'\n\s*bool v\d+_mma = ', loop)
    assert operator and mma
    return (loop[star:operator.start()], loop[operator.start():mma.start()],
            loop[mma.start():])


def test_the_kernel_splits_the_block_into_its_roles():
    """The split ahead of the trips, a warp's role by its index; a role's
    barrier its own, its id by the role -- the star warps' 448 threads around
    their staging, the mma role's 128 ahead of the next trip's products; no
    barrier of the block inside the trips: the roles meet on the rings."""
    src = _kernel()
    assert re.search(r'bool v\d+_star = threadIdx\.y < 14;', src)
    assert re.search(r'bool v\d+_operator = \(threadIdx\.y >= 14\) && '
                     r'\(threadIdx\.y < 15\);', src)
    counts = re.findall(rf'barrier\.sync %0, %1;" :: "r"\({IDENT}\), '
                        rf'"r"\((\d+)\)', src)
    assert counts == ['448', '448', '128']
    loop = src[src.index('for (size_t'):src.index('tmemDealloc')]
    assert '__syncthreads' not in loop


def test_each_role_does_its_own():
    """The star warps: A's blocks, each behind its slot's empty side, stored
    into tensor memory and published -- and nothing of B; the operator's
    first lane: the trip's five chunks of B, each behind its slot's empty
    side, a bulk copy and its bytes; the mma role's first thread: each
    block's A and B taken, its products, their commit onto both slots'
    empty sides, then D's commit, the role's wait for it and D added into
    the outputs."""
    star, operator, mma = _regions(_kernel())
    assert star.count('tensorforge::mbarrierArrive(&ta3_ring[') == 5
    assert 'tcgen05.st' in star and 'bk3_ring' not in star
    assert 'tcgen05.mma' not in star
    assert operator.count('tensorforge::bulkCopyOn(&bk3_ring[') == 5
    assert operator.count('tensorforge::mbarrierExpect(&bk3_ring[') == 5
    assert len(re.findall(r'mbarrierWait\(&bk3_ring\[4 \+ ', operator)) == 5
    assert 'ta3_ring' not in operator and 'tcgen05' not in operator
    assert len(re.findall(r'mbarrierWait\(&ta3_ring\[\d+_i32\]',
                          mma)) == 5
    assert len(re.findall(r'mbarrierWait\(&bk3_ring\[v\d+_bks\]', mma)) == 5
    # A's ring three slots, a band each (`multi_element._fold_bands`): their
    # empty sides the three mbarriers past the full ones.
    assert mma.count('tensorforge::tcgen05CommitOn(&ta3_ring[3 + ') == 5
    assert mma.count('tensorforge::tcgen05CommitOn(&bk3_ring[4 + ') == 5
    assert mma.index('tensorforge::tcgen05Commit(tcmma_bulk') < mma.index(
        'tensorforge::bulk_wait(tcmma_bulk)') < mma.index('tcgen05.ld')
    assert 'red.global.add.f32' in mma and 'bulkCopyOn' not in mma


def _b_acts(src):
    """The mma role's acts on B's ring in the text's order: `('take', j)`
    and `('release', j)`, the chunk a commit onto a slot's empty side gives
    back -- named by the take's slot."""
    taken, out = {}, []
    for line in _regions(src)[2].splitlines():
        take = re.search(r'mbarrierWait\(&bk3_ring\[(\w+)\],', line)
        release = re.search(r'tcgen05CommitOn\(&bk3_ring\[4 \+ \((\w+)\)\]',
                            line)
        if take:
            taken[take.group(1)] = len(taken)
            out.append(('take', taken[take.group(1)]))
        elif release:
            out.append(('release', taken[release.group(1)]))
    return out


@pytest.mark.parametrize('hold', [None, 2])
def test_the_products_give_b_back_as_the_ring_holds_it(hold):
    """The ring's hold (`Options.matrix_ring_hold`, `RingSpec.hold`): by
    default the products of a block commit its chunk's slot behind them,
    which they arrive at once complete; held for two, the slot of the chunk
    before, behind the next block's products too -- the last ones behind
    the last block.  The comment says the hold taken."""
    options = {} if hold is None else {'matrix_ring_hold': hold}
    src = _kernel(**options)
    held = hold or 1
    expected = []
    for j in range(5):
        expected.append(('take', j))
        if j >= held - 1:
            expected.append(('release', j - held + 1))
    expected += [('release', j) for j in range(5 - held + 1, 5)]
    assert _b_acts(src) == expected
    assert ('5 chunks a trip, a slot held for 2 chunks;' in src) is (
        hold == 2)


# -- the PIR -------------------------------------------------------------------- #

def _bodies():
    return tro._bodies(order=6, arch='sm_100a', **{
        k: v for k, v in THREE.items() if k != 'matrix_engine'})


def test_the_rings_hand_over_between_the_roles():
    """The rings' acts by role (`pir.rings.handoffs`): B's ring filled by the
    operator and drained by the mma role, A's by the star warps and the
    mma role -- as their descriptions name them."""
    uses = tro._uses(_bodies())
    (b,) = [u for name, u in uses.items() if name.startswith('bk')]
    (a,) = [u for name, u in uses.items() if name.startswith('ta')]
    assert (b.spec.producer, b.spec.consumers) == ('operator', ('mma',))
    assert b.roles('acquire') == b.roles('copy') == b.roles('publish') == (
        'operator',)
    assert b.roles('take') == b.roles('release') == ('mma',)
    assert (a.spec.producer, a.spec.consumers) == ('star', ('mma',))
    assert a.roles('acquire') == a.roles('publish') == ('star',)
    assert a.roles('take') == a.roles('release') == ('mma',)


def test_the_roles_verify():
    """Through every pass: no act of a ring out of its roles, no role's
    barrier out of its role, the chain of the split whole."""
    found = []
    for body in _bodies():
        found += [d for d in verify(body, strict=False)
                  if "slot's" in d or 'role' in d or 'chain' in d]
    assert not found, found[:3]


# -- the oracle ------------------------------------------------------------------ #

def _two_trips(monkeypatch, **options):
    """`test_folded_star._two_trips_at` at o6 on sm_100a in the three roles:
    a group of four over seven elements.  The roles' plan streams B already
    (`element_group.roles_refusal`), ahead of the resident one, so the
    room is left as it is (`test_multi_element_async._streaming` would cut
    it below the roles' plan)."""
    monkeypatch.setattr(tma, '_streaming', lambda *args, **kwargs: None)
    tfs._two_trips_at(6, 'volume', 'sm_100a', monkeypatch, True,
                      **{**THREE, **options})


@pytest.mark.parametrize('slots', [2, 4])
def test_the_three_roles_over_two_trips_on_the_oracle(slots, monkeypatch):
    """Every element of two trips of a group of four over seven elements,
    the last warp masked, against NumPy, and no access that a ring, a
    role's barrier or a wait does not order (`kernel_eval.Races`): the
    star warps' A read by the products only behind its slot's full side,
    B's slots only behind the operator warp's bytes, either written again
    only behind the products' commit, D read only behind theirs -- and a
    wait for a phase nobody completes refused as the hang it would be.
    B's ring of two slots goes round twice and more within a trip."""
    _two_trips(monkeypatch, matrix_ring_slots=slots)


def test_the_roles_own_barriers_carry_them_without_the_placement(
        monkeypatch):
    """Without the barrier placement (`enable_sync_block_opt` off): the
    roles' own barriers are all there is -- the star warps' around their
    staging, the mma role's ahead of the next trip's products, each in the
    text -- and they are enough: every element of two trips against NumPy,
    without a race."""
    src = _kernel(enable_sync_block_opt=False)
    counts = re.findall(rf'barrier\.sync %0, %1;" :: "r"\({IDENT}\), '
                        rf'"r"\((\d+)\)', src)
    assert counts == ['448', '448', '128']
    _two_trips(monkeypatch, enable_sync_block_opt=False)


@pytest.mark.parametrize('slots', [2, 4])
def test_b_held_for_two_blocks_on_the_oracle(slots, monkeypatch):
    """B's slots held for two blocks (`matrix_ring_hold` 2): the operator's
    copy into a slot waits for the products of the block after its chunk's
    as well -- with two slots, every block's copy behind the one before's
    products."""
    _two_trips(monkeypatch, matrix_ring_slots=slots, matrix_ring_hold=2)


# -- the build and the hardware ------------------------------------------------- #

def test_the_build_reads_back(tmp_path):
    """Built for sm_100a: the products, the operator's bulk copies and the
    roles' barriers in the SASS, as many as the text has."""
    src = _kernel()
    sass = tma._sass('sm_100a', tmp_path, 'volume', 6, **THREE)
    assert tma._count(sass, 'UTCHMMA') == src.count('tcgen05.mma.cta_group')
    assert tma._count(sass, 'UBLKCP') == src.count('bulkCopyOn(') == 5
    assert len(re.findall(r'\bBAR\.SYNC\b', sass)) >= 3


# -- A's slots, each band's own ------------------------------------------------- #

def _a_slots(src):
    """The slots of the kernel's ring of A in tensor memory: its mbarriers
    set up, a full side of a band's 128 threads each."""
    (count,) = re.findall(r'for \(int r = 0; r < (\d+); \+\+r\) \{\s*'
                          r'tensorforge::mbarrierInit\(&ta\d+_ring\[r\], 128\)',
                          src)
    return int(count)


@pytest.mark.parametrize('order', [4, 6, 8])
def test_a_s_slots_are_each_band_s_own(order):
    """The fourteen element warps' three bands fill A's ring of tensor
    memory between roles: three of its four slots, so that block `j`'s
    slot `j % 3` is band `j % 3`'s -- every round of a slot one band's, in
    turn.  Without the roles every warp meets the products at the trip's
    end, and the four slots stay."""
    group, _ = _group(order)
    assert group.elements == 14
    assert _a_slots(_kernel(order)) == 3
    plain = {k: v for k, v in THREE.items() if k != 'matrix_roles'}
    assert _a_slots(tma._kernel('sm_100a', 'volume', order, 's',
                                **plain).get_kernel()) == 4


@pytest.mark.parametrize('blocks', [2, 5, 11])
def test_the_ring_model_sees_the_bands_race(blocks):
    """The protocol model of the ring (`ring_model`), each warp its own
    agent under seeded random schedules: three bands on four slots hand
    the products a slot another band wrote for a later block -- a band
    waiting for a round two phases ahead of the slot's empty side passes
    the wait, its parity alike -- as soon as a trip has a fifth block;
    three bands on three slots, and two, one or four bands on four, never.
    The bands meeting at each trip's head, as the three roles' staging
    makes them."""
    import ring_model
    racy = ring_model.violations(3, 4, blocks, sync='trip')
    assert (racy > 0) == (blocks > 4), racy
    for bands, slots in ((3, 3), (2, 4), (1, 4), (4, 4)):
        assert ring_model.violations(bands, slots, blocks, sync='trip') == 0


@pytest.mark.parametrize('order', [4, 6, 8])
def test_the_roles_ring_of_a_holds_in_the_model(order):
    """The kernel's own ring -- its slots read off its text, its bands the
    element warps' whole fours, its blocks the folded step's -- in the
    protocol model: no schedule hands the products another block's A."""
    import ring_model
    group, _ = _group(order)
    section = group.section
    (step,) = [s for s in section.steps if s.lowering == 'tcgen05']
    depth = max(st.depth for st in eg._step_stages(section, step))
    blocks = -(-depth // eg.STAR_ROWS)
    assert ring_model.violations(group.elements // 4, _a_slots(
        _kernel(order)), blocks, sync='trip') == 0


def test_the_three_roles_compute_on_the_gpu(target, cache_root,
                                            tensorforge_include):
    """On a GB200: the o6 volume in the three roles against NumPy over a
    batch ending in a partial group, several trips a block, the operators
    their description's numbers (`test_multi_element._described`)."""
    if target.backend != 'cuda' or target.arch != 'sm_100':
        pytest.skip('no GPU with tcgen05')
    from harness.runner import run_case
    special = dataclasses.replace(target, arch='sm_100a')
    tma._nvcc_builds(special.arch)
    group = eg.plan(me._descrs('volume', 6, 's'), me._context(
        special.arch, 's', multi_element='force', **THREE))
    assert group is not None and group.section.roles == 'three'
    case = types.SimpleNamespace(
        NAME='three_roles_volume_o6s', DTYPE=Datatype.F32, BATCH=4099,
        TOL=me._normwise(), OPTIONS={'multi_element': 'force', **THREE},
        INPUT_TRANSFORM=me._described(me._descrs('volume', 6, 's')),
        descr_list=lambda: me._descrs('volume', 6, 's'),
        reference=me._reference(me._descrs('volume', 6, 's')))
    result = run_case(case, special, cache_root, tensorforge_include)
    assert result.passed, (result.max_abs_err, result.max_rel_err,
                           result.detail)


@pytest.mark.parametrize('order', [6, 8])
def test_many_trips_of_the_three_roles_on_the_gpu(order, target, cache_root,
                                                  tensorforge_include):
    """On a GB200, where the bands' race on A's ring showed: the o6 and o8
    volume over 65537 elements on a twentieth of the grid the device holds
    -- some 650 trips a block (`Options.resident_grid`) -- in the three
    roles, bit for bit what the folded star without them gives over the
    same elements (the same products in the same order)."""
    if target.backend != 'cuda' or target.arch != 'sm_100':
        pytest.skip('no GPU with tcgen05')
    import numpy as np
    from harness import driver_emit
    from harness.runner import execute
    special = dataclasses.replace(target, arch='sm_100a')
    tma._nvcc_builds(special.arch)
    plain = {k: v for k, v in THREE.items() if k != 'matrix_roles'}
    builds = [me._generate(me._descrs('volume', order, 's'), me._context(
        special.arch, 's', multi_element='force', resident_grid=0.05,
        **options)) for options in (THREE, plain)]
    assert builds[0].built._element_group.section.roles == 'three'
    batch = 65537
    rng = np.random.default_rng(17)
    inputs = {}
    for op in driver_emit.collect_operands(builds[0]):
        if op.is_scalar or not op.is_source:
            continue
        key = op.alias or op.kernel_name
        if op.values is not None:
            inputs[key] = np.asarray(op.values, dtype=np.float32).reshape(
                op.shape, order='F')[None]
            continue
        rows = 1 if op.addressing == 'none' else batch
        inputs[key] = rng.standard_normal((rows, *op.shape)).astype(
            np.float32)
    got, want = (execute(gen, special, cache_root, tensorforge_include,
                         batch, inputs, runtime_timeout=300.0)
                 for gen in builds)
    assert set(got) == set(want)
    for name in want:
        assert np.array_equal(got[name], want[name]), name
