# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""The high-order tile of the multi-element layout's asynchronous tiling
(`Options.matrix_tile` `high`, `element_group.TileRows`,
`compute.multi_element._tile`): the group's rows dense on the engine's --
on wgmma (sm_90a) a tile of whole elements a consumer warpgroup, each a
role of its own, sharing the prepared operator ring that a producer's warp
fills while another stages the inputs; on tcgen05 (sm_100a) the three roles
at as many elements as tensor memory's lanes hold.

What is checked here: the plan -- the options' defaults, the rows' map, the
group and the block the fill rule takes on each engine, the refusals and
the kernel's note of them; the kernel's text -- the roles, their barriers
and the rings' counts; the PIR -- the rings' hand-offs by role, verified
through every pass; the host oracle over two trips, the second's elements
masked past the batch, against NumPy and without a race; the build for
sm_90a and sm_100a read back; and on a GH200 / GB200 the volume against
NumPy.
"""

from __future__ import annotations

import dataclasses
import re
import shutil
import subprocess
import types
import warnings

import pytest

import test_multi_element as me
import test_multi_element_async as tma
import test_ring_object as tro
from tensorforge.backend.pir.passes import verify
from tensorforge.common.basic_types import Datatype
from tensorforge.generators import element_group as eg

#: The high-order tile asked for.
HIGH = {'matrix_engine': 'auto', 'matrix_tile': 'high'}


def _group(order=8, arch='sm_90a', shape='volume', precision='s',
           **options):
    group = eg.plan(me._descrs(shape, order, precision), me._context(
        arch, precision, multi_element='force', **{**HIGH, **options}))
    assert group is not None
    return group


def _kernel(order=8, arch='sm_90a', **options):
    return tma._kernel(arch, 'volume', order, 's', **{**HIGH, **options})


# -- the plan ------------------------------------------------------------------- #

def test_the_options_default_to_the_tilings_as_before():
    from tensorforge.common.options import registry
    assert registry()['matrix_tile'].default == 'off'
    assert registry()['matrix_consumers'].default == 2


def test_the_options_parse():
    from tensorforge.common.options import (parse_matrix_consumers,
                                            parse_matrix_tile)
    assert [parse_matrix_tile(t) for t in ('off', 'HIGH')] == ['off', 'high']
    assert [parse_matrix_consumers(t) for t in '1234'] == [1, 2, 3, 4]
    for text in ('0', '5'):
        with pytest.raises(ValueError):
            parse_matrix_consumers(text)
    with pytest.raises(ValueError):
        parse_matrix_tile('low')


def test_the_rows_are_whole_elements_a_tile():
    """Seven elements of nine quantities a wgmma tile of 64 rows, the 64th
    padding; fourteen on tcgen05's 128 lanes, two rows of padding."""
    wg = eg.TileRows('wgmma', 9, 7, 2)
    assert (wg.m, wg.used, wg.elements) == (64, 63, 14)
    assert wg.place(0, 0) == (0, 0) and wg.place(6, 8) == (0, 62)
    assert wg.place(7, 0) == (1, 0) and wg.place(13, 8) == (1, 62)
    tc = eg.TileRows('tcgen05', 9, 14, 1)
    assert (tc.m, tc.used, tc.elements) == (128, 126, 14)
    assert tc.place(13, 8) == (0, 125)


@pytest.mark.parametrize('order', [4, 6, 8])
def test_on_wgmma_two_consumer_warpgroups_of_seven_elements(order):
    """sm_90a: fourteen elements, each consumer warpgroup a tile of seven
    on 63 of its 64 rows -- not E/4 warpgroups each issuing 64 rows over
    9E -- and two producer warps behind them; no warp an element's, the
    batch loop's rows all a producer's; the star folded, B streamed
    prepared through its ring, the inputs through the producer's."""
    group = _group(order)
    section = group.section
    assert section.tile == eg.TileRows('wgmma', 9, 7, 2)
    assert group.elements == 14 and group.warps == 10
    assert group.rows == 0 and group.producers == 10
    assert eg.rowless(section) and not section.roles
    assert section.star == 'engine' and section.ring >= 1
    (step,) = [s for s in section.steps if s.lowering == 'wgmma']
    assert all(st.prepared and st.streamed
               for st in eg._step_stages(section, step))


def test_one_consumer_warpgroup_where_the_options_say_so():
    group = _group(8, matrix_consumers=1)
    assert group.section.tile == eg.TileRows('wgmma', 9, 7, 1)
    assert group.elements == 7 and group.warps == 6


@pytest.mark.parametrize('order', [4, 6, 8])
def test_on_tcgen05_the_three_roles_at_fourteen_elements(order):
    """sm_100a: the three roles, fourteen elements on 126 of tensor
    memory's 128 lanes, whatever the table's group or `matrix_roles`."""
    group = _group(order, 'sm_100a')
    section = group.section
    assert section.tile == eg.TileRows('tcgen05', 9, 14, 1)
    assert section.roles == 'three' and not eg.rowless(section)
    assert group.elements == 14 and group.rows == 14
    assert group.warps == 14 + eg.OPERATOR_WARPS + eg.MMA_WARPS


def test_the_option_off_is_the_kernel_without_it():
    for arch in ('sm_90a', 'sm_100a'):
        off = tma._kernel(arch, 'volume', 6, 's', matrix_tile='off')
        none = tma._kernel(arch, 'volume', 6, 's')
        assert off.get_kernel() == none.get_kernel()
        assert '// tile:' not in off.get_kernel()


@pytest.mark.parametrize('arch,precision,options,why', [
    ('sm_90a', 'd', {}, 'steps on the asynchronous engines'),
    ('sm_80', 's', {}, 'steps on the asynchronous engines'),
])
def test_where_the_tile_does_not_take_the_kernel_says_why(arch, precision,
                                                          options, why):
    group = _group(6, arch, precision=precision, **options)
    assert group.section.tile is None
    assert why in group.tile_note
    context = me._context(arch, precision, multi_element='force',
                          **{**HIGH, **options})
    (note,) = eg.tile_notes(group, context)
    assert note.startswith('matrix_tile=high declined: ') and why in note


def test_the_kernel_says_what_the_tile_took():
    src = _kernel(8).get_kernel()
    (note,) = [line.strip() for line in src.splitlines()
               if line.strip().startswith('// tile:')]
    assert '14 elements, 2 tiles of 7 on 63 of 64 rows' in note
    src = _kernel(8, 'sm_100a').get_kernel()
    (note,) = [line.strip() for line in src.splitlines()
               if line.strip().startswith('// tile:')]
    assert 'in the three roles' in note


# -- the kernel ------------------------------------------------------------------ #

#: A role's barrier id as the split spells it: the top three of NVIDIA's
#: sixteen, by the warp (`roles.first_id`).
IDENT = r'\(13 \+ \(threadIdx\.y >= 4\) \+ \(threadIdx\.y >= 8\)\)'


def test_the_kernel_splits_the_block_into_its_roles():
    """The two tiles' roles and the producer's, each meeting on its own
    named barrier of its threads -- a warpgroup's 128 -- and no block
    barrier in the trip; both rings' empty sides counting a consumer
    warp's arrival each."""
    src = _kernel(8).get_kernel()
    assert re.search(r'bool v\d+_tile0 = threadIdx\.y < 4;', src)
    assert re.search(r'bool v\d+_tile1 = \(threadIdx\.y >= 4\) && '
                     r'\(threadIdx\.y < 8\);', src)
    counts = set(re.findall(rf'barrier\.sync %0, %1;" :: "r"\({IDENT}\), '
                            rf'"r"\((\d+)\)', src))
    assert counts == {'128'}
    loop = src[src.index('for (size_t'):]
    assert '__syncthreads()' not in loop
    assert re.search(r'mbarrierInit\(&me_ring\[\d+ \+ r\], 8\)', src)
    assert re.search(r'mbarrierInit\(&bk\d+_ring\[\d+ \+ r\], 8\)', src)
    # Each tile's 99 products of o8: three of 3xTF32 a block of eight rows
    # of K of every product, eleven blocks.
    assert src.count('wgmma.mma_async') == 2 * 99


def _bodies(**options):
    return tro._bodies(order=6, arch='sm_90a', **{
        k: v for k, v in {**HIGH, **options}.items()
        if k != 'matrix_engine'})


def test_the_rings_hand_over_between_the_roles():
    """The rings' acts by role (`pir.rings.handoffs`): the producer fills
    both, the inputs' and B's, and the tiles drain both."""
    uses = tro._uses(_bodies())
    inputs = uses['me_ring']
    (b,) = [u for name, u in uses.items() if name.startswith('bk')]
    for use in (inputs, b):
        assert (use.spec.producer, use.spec.consumers) == (
            'producer', ('tile0', 'tile1'))
        assert use.roles('acquire') == use.roles('copy') == use.roles(
            'publish') == ('producer',)
        assert use.roles('take') == use.roles('release') == ('tile0',
                                                             'tile1')


def test_the_roles_verify():
    """Through every pass: no act of a ring out of its roles, no role's
    barrier out of its role, the chain of the split whole."""
    found = []
    for body in _bodies():
        found += [d for d in verify(body, strict=False)
                  if "slot's" in d or 'role' in d or 'chain' in d]
    assert not found, found[:3]


# -- the oracle ------------------------------------------------------------------ #

def _on_the_oracle(arch, order, count, **options):
    """Every element of `count` against NumPy on the host oracle, the
    group's trips over them -- the last one's elements past the batch
    masked -- and no access a ring, a role's barrier or a wait does not
    order (`kernel_eval.Races`)."""
    import numpy as np
    from tensorforge.common.basic_types import Addressing
    from tensorforge.reference import kernel_eval
    gen = tma._kernel(arch, 'volume', order, 's', **{**HIGH, **options})
    tensors = {}
    for outer in gen.descr_list:
        for d in outer.operations():
            for m in d.matrix_list():
                t = getattr(m, 'tensor', m)
                tensors.setdefault(t.alias, t)

    def box(t):
        return [(int(lo), int(hi)) for lo, hi in zip(t.bbox.lower(),
                                                      t.bbox.upper())]

    def memory(t, read):
        (r0, r1), (c0, c1) = box(t)
        out = np.zeros((count, r1 - r0, c1 - c0))
        for e in range(count):
            for c in range(c1 - c0):
                for r in range(r1 - r0):
                    out[e, r, c] = read(e, r + c * (r1 - r0))
        return out

    fill = kernel_eval.Slot(me.SEED)
    preset, inputs = {}, {}
    for alias, t in tensors.items():
        if t.is_tmp:
            continue
        if t.addressing.kind is Addressing.NONE:
            preset[t.name] = [float(v) for v in t.storage_values()]
            (r0, r1), (c0, c1) = box(t)
            inputs[alias] = np.asarray(t.get_values())[None, r0:r1, c0:c1]
        else:
            inputs[alias] = memory(t, lambda e, s, t=t: fill.read(
                f'{t.name}@{e}', s))
    output = next(t for t in tensors.values() if not t.is_tmp
                  and t.addressing.kind is not Addressing.NONE
                  and any(d.writes().tensor is t for outer in gen.descr_list
                          for d in outer.operations()))
    lanes, mults = kernel_eval.launch_geometry(gen.get_launcher())
    races = []
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        mem = kernel_eval.evaluate_wave(
            gen.get_kernel(), lanes, seed=me.SEED, globals_only=True,
            mults=mults, block=True, preset=preset, elements=count,
            races=races)
    assert not races, races[:3]
    got = memory(output, lambda e, s: mem.get((f'{output.name}@{e}', s),
                                              fill.read(f'{output.name}@{e}',
                                                        s)))
    want = me._reference(me._descrs('volume', order, 's'))(
        inputs, inputs[output.alias])
    for e in range(count):
        scale = np.abs(want[e]).max() or 1.0
        error = np.abs(got[e] - want[e]).max() / scale
        assert error <= 1e-6, (e, error)
    return gen


@pytest.mark.parametrize('arch', ['sm_90a', 'sm_100a'])
def test_two_trips_of_the_tile_on_the_oracle(arch):
    """Seventeen elements: a trip of fourteen and one of three, the eleven
    past the batch masked -- on wgmma the second tile's rows all of masked
    elements, its products run, its writes none."""
    gen = _on_the_oracle(arch, 4, 17)
    assert gen.built._element_group.elements == 14


def test_one_consumer_warpgroup_on_the_oracle():
    gen = _on_the_oracle('sm_90a', 4, 9, matrix_consumers=1)
    assert gen.built._element_group.elements == 7


def test_one_slot_of_the_input_ring_on_the_oracle(monkeypatch):
    """The input ring of one slot, the plan's last choice (`tile_plan`):
    the second trip's inputs go into the slot the first trip's mixing
    read, the producer behind every consumer warp's last block of A
    (`_wgmma_tile`'s `done`) -- which the oracle's ring order checks."""
    monkeypatch.setattr(eg, 'PRODUCER_SLOTS', ())
    gen = _on_the_oracle('sm_90a', 4, 17)
    assert gen.built._element_group.section.ring == 1


# -- the build and the hardware ------------------------------------------------- #

def _ptxas(arch, tmp_path, order=8, **options):
    """The kernel built for `arch` with ptxas's report: `(sass, report)`."""
    from tensorforge import toolchain
    nvcc = tma._nvcc_builds(arch)
    gen = _kernel(order, arch, **options)
    context = me._context(arch, 's', multi_element='force',
                          **{**HIGH, **options})
    headers = list(context.target.headers()) + list(gen.get_helper_headers())
    src = tmp_path / 'kernel.cu'
    src.write_text(''.join(f'#include "{h}"\n' for h in headers)
                   + '#include "tensorforge_aux.h"\n' + gen.get_kernel()
                   + '\n' + gen.get_launcher())
    entry = toolchain.COMPILERS['cuda']
    obj = tmp_path / 'kernel.o'
    run = subprocess.run([nvcc, *entry.language_flags(),
                          *entry.target_flags(arch), '-Xptxas', '-v',
                          '-I', str(tma.INCLUDE), '-c', str(src), '-o',
                          str(obj)], capture_output=True, text=True,
                         timeout=1800)
    assert run.returncode == 0, run.stderr[-4000:]
    tool = shutil.which('cuobjdump')
    if tool is None:
        pytest.skip('cuobjdump not found')
    sass = subprocess.run([tool, '-sass', str(obj)], capture_output=True,
                          text=True, timeout=600).stdout
    return sass, run.stderr


@pytest.mark.parametrize('arch,engine,products', [
    ('sm_90a', 'HGMMA', 2 * 99), ('sm_100a', 'UTCHMMA', 99)])
def test_the_build_reads_back(arch, engine, products, tmp_path):
    """o8 built: every product an instruction of the engine's, as many as
    the text has, and none of them serialized by ptxas: no C7510 (a wgmma
    serialized for a register's use), C7511, C7514 or C7517 in its
    report."""
    sass, report = _ptxas(arch, tmp_path)
    assert tma._count(sass, engine) == products
    for code in ('C7510', 'C7511', 'C7514', 'C7517'):
        assert code not in report, report[-2000:]


@pytest.mark.parametrize('order', [6, 8])
def test_the_tile_computes_on_the_gpu(order, target, cache_root,
                                      tensorforge_include):
    """On a GH200 (wgmma) or GB200 (tcgen05): the volume in the high-order
    tile against NumPy over a batch ending in a partial group, several
    trips a block, the operators their description's numbers
    (`test_multi_element._described`)."""
    if target.backend != 'cuda' or target.arch not in ('sm_90', 'sm_100'):
        pytest.skip('no GPU with wgmma or tcgen05')
    from harness.runner import run_case
    special = dataclasses.replace(target, arch=target.arch + 'a')
    tma._nvcc_builds(special.arch)
    group = eg.plan(me._descrs('volume', order, 's'), me._context(
        special.arch, 's', multi_element='force', **HIGH))
    assert group is not None and group.section.tile is not None
    case = types.SimpleNamespace(
        NAME=f'hiorder_tile_volume_o{order}s', DTYPE=Datatype.F32,
        BATCH=4099, TOL=me._normwise(),
        OPTIONS={'multi_element': 'force', **HIGH},
        INPUT_TRANSFORM=me._described(me._descrs('volume', order, 's')),
        descr_list=lambda: me._descrs('volume', order, 's'),
        reference=me._reference(me._descrs('volume', order, 's')))
    result = run_case(case, special, cache_root, tensorforge_include)
    assert result.passed, (result.max_abs_err, result.max_rel_err,
                           result.detail)
