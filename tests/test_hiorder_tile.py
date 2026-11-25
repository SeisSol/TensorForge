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

B streams in kept spans (`Options.matrix_spans`, `element_group.
tile_chunks`): of each product's block of K only its tiles of N from the
first that holds an entry to the last, in the engine's granules, each
multiplied at its own N into D's columns of it.  On wgmma the blocks are one
loop (`Options.matrix_groups`, `multi_element._wgmma_groups`): its body
written once, one dispatch a block on the block's width.

What is checked here: the plan -- the options' defaults, the rows' map, the
group and the block the fill rule takes on each engine, the refusals and
the kernel's note of them; the kept spans against the operators' entries,
and the prepared table as the descriptors read it; the kernel's text -- the
roles, their barriers, the rings' counts, the spans' widths and the group
loop; the PIR -- the rings' hand-offs by role, verified
through every pass; the host oracle over two trips, the second's elements
masked past the batch, against NumPy and without a race, and with dense
spans bit for bit what the kept ones give; the build for sm_90a and
sm_100a read back; and on a GH200 / GB200 the volume against NumPy.
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
    assert section.tile == eg.TileRows('wgmma', 9, 7, 2, groups='loop')
    assert group.elements == 14 and group.warps == 10
    assert group.rows == 0 and group.producers == 10
    assert eg.rowless(section) and not section.roles
    assert section.star == 'engine' and section.ring >= 1
    (step,) = [s for s in section.steps if s.lowering == 'wgmma']
    assert all(st.prepared and st.streamed
               for st in eg._step_stages(section, step))


def test_one_consumer_warpgroup_where_the_options_say_so():
    group = _group(8, matrix_consumers=1)
    assert group.section.tile == eg.TileRows('wgmma', 9, 7, 1, groups='loop')
    assert group.elements == 7 and group.warps == 6


@pytest.mark.parametrize('order', [4, 6, 8])
def test_on_tcgen05_the_three_roles_at_fourteen_elements(order):
    """sm_100a: the three roles, fourteen elements on 126 of tensor
    memory's 128 lanes, whatever the table's group or `matrix_roles`."""
    group = _group(order, 'sm_100a')
    section = group.section
    assert section.tile == eg.TileRows('tcgen05', 9, 14, 1, groups='loop')
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
    (note, _) = [line.strip() for line in src.splitlines()
                 if line.strip().startswith('// tile:')]
    assert '14 elements, 2 tiles of 7 on 63 of 64 rows' in note
    src = _kernel(8, 'sm_100a').get_kernel()
    (note, _) = [line.strip() for line in src.splitlines()
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
    # Each tile's group loop at o8: a block's nine products -- three of
    # 3xTF32 of each of its three -- once for each of the blocks' six
    # widths; written out, 99: nine of each of the eleven blocks.
    assert src.count('wgmma.mma_async') == 2 * 6 * 9
    unrolled = _kernel(8, matrix_groups='unrolled').get_kernel()
    assert unrolled.count('wgmma.mma_async') == 2 * 99


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
    return gen, got


@pytest.mark.parametrize('arch', ['sm_90a', 'sm_100a'])
def test_two_trips_of_the_tile_on_the_oracle(arch):
    """Seventeen elements: a trip of fourteen and one of three, the eleven
    past the batch masked -- on wgmma the second tile's rows all of masked
    elements, its products run, its writes none."""
    gen, _ = _on_the_oracle(arch, 4, 17)
    assert gen.built._element_group.elements == 14


def test_one_consumer_warpgroup_on_the_oracle():
    gen, _ = _on_the_oracle('sm_90a', 4, 9, matrix_consumers=1)
    assert gen.built._element_group.elements == 7


@pytest.mark.parametrize('groups', ['loop', 'unrolled'])
def test_one_slot_of_the_input_ring_on_the_oracle(monkeypatch, groups):
    """The input ring of one slot, the plan's last choice (`tile_plan`):
    the second trip's inputs go into the slot the first trip's mixing
    read, the producer behind every consumer warp's last block of A
    (`_wgmma_tile`'s `done`) -- which the oracle's ring order checks."""
    monkeypatch.setattr(eg, 'PRODUCER_SLOTS', ())
    gen, _ = _on_the_oracle('sm_90a', 4, 17, matrix_groups=groups)
    assert gen.built._element_group.section.ring == 1


@pytest.mark.parametrize('arch', ['sm_90a', 'sm_100a'])
@pytest.mark.parametrize('groups', ['loop', 'unrolled'])
def test_two_slots_of_b_s_ring_on_the_oracle(arch, groups):
    """B's prepared ring at two slots: o4's two trips' four chunks through
    them, each slot taken again behind the wait that retired the products
    that read it and the warps' arrivals that gave it back -- which the
    oracle's ring order checks."""
    gen, _ = _on_the_oracle(arch, 4, 17, matrix_ring_slots=2,
                            matrix_groups=groups)
    assert gen.built._element_group.section.ring_slots == 2


@pytest.mark.parametrize('arch', ['sm_90a', 'sm_100a'])
def test_the_group_loop_computes_what_the_unrolled_groups_do(arch):
    """The group loop against the groups written out, both against NumPy
    on the oracle: bit for bit -- the same products in the same order; on
    wgmma the loop's D started at the accumulators' zeros where the
    unrolled groups' first product writes it fresh, on tcgen05 N and D's
    columns values."""
    _, loop = _on_the_oracle(arch, 4, 17)
    _, unrolled = _on_the_oracle(arch, 4, 17, matrix_groups='unrolled')
    assert (loop == unrolled).all()


# -- the kept spans ------------------------------------------------------------- #

def _chunks(order=8, arch='sm_90a', **options):
    section = _group(order, arch, **options).section
    (step,) = [s for s in section.steps if s.lowering in eg.ASYNC_LOWERINGS]
    return section, step, eg.tile_chunks(section, step)


def test_the_spans_option_defaults_to_kept():
    from tensorforge.common.options import parse_matrix_spans, registry
    assert registry()['matrix_spans'].default == 'kept'
    assert [parse_matrix_spans(t) for t in ('kept', 'Dense')] == ['kept',
                                                                 'dense']
    with pytest.raises(ValueError):
        parse_matrix_spans('sparse')


@pytest.mark.parametrize('arch', ['sm_90a', 'sm_100a'])
@pytest.mark.parametrize('order', [4, 6, 8])
def test_every_entry_of_b_is_in_exactly_one_kept_piece(arch, order):
    """The kept spans against the operators' entries: every entry inside a
    product's window in exactly one piece of its product -- the block of K
    it is in, the rows of N of its span; the spans whole granules of the
    engine's N (8 on wgmma, 16 on tcgen05) inside the stage, on wgmma each
    on to the stage's last row (`TileRows.descending`), the first block's
    first piece whole; at o6 and o8 fewer rows of N than the dense pieces'
    (o8: 62 % of them on wgmma, 61 % on tcgen05, its group loop's union of
    a block's spans)."""
    section, step, chunks = _chunks(order, arch)
    granule = section.tile.granule
    assert granule == (16 if arch == 'sm_100a' else 8)
    assert section.tile.descending == (arch == 'sm_90a')
    stages = eg._step_stages(section, step)
    first = chunks[0][0]
    assert first[3:] == (0, stages[first[0]].width)
    for chunk in chunks:
        for p, k0, rows, lo, hi in chunk:
            assert 0 <= lo < hi <= stages[p].width
            assert lo % granule == 0 and hi % granule == 0
            assert hi == stages[p].width or not section.tile.descending
    for p, product in enumerate(step.products):
        stage = stages[p]
        (r0, r1), (d0, d1) = stage.window
        pattern = product.first.tensor.nonzero_pattern
        for kk in range(d0, d1):
            for row in range(r0, r1):
                if not pattern.is_nz((row, kk)):
                    continue
                n, k = row - stage.n0, kk - d0
                held = [None for chunk in chunks
                        for q, k0, rows, lo, hi in chunk
                        if q == p and k0 <= k < k0 + rows and lo <= n < hi]
                assert len(held) == 1, (p, row, kk)
    kept = sum(rows * (hi - lo) for chunk in chunks
               for _, _, rows, lo, hi in chunk)
    dense = sum(rows * stages[p].width for chunk in chunks
                for p, _, rows, _, _ in chunk)
    assert kept <= dense
    if order == 8:
        # A block's products one span in the group loop, their union: 61 %
        # on tcgen05 where each product's own would keep 59 %.
        assert round(kept / dense, 2) == (0.61 if arch == 'sm_100a'
                                          else 0.62)


def test_dense_spans_are_whole_widths():
    section, step, chunks = _chunks(8, matrix_spans='dense')
    stages = eg._step_stages(section, step)
    assert section.tile.spans == 'dense'
    assert all((lo, hi) == (0, stages[p].width) for chunk in chunks
               for p, _, _, lo, hi in chunk)


def test_the_spans_are_dense_where_known_zeros_are_not_skipped():
    section, _, _ = _chunks(8, skip_known_zeros=False)
    assert section.tile.spans == 'dense'


def test_a_block_of_no_entry_is_refused(monkeypatch):
    """A block of K that no product has an entry in -- an inactive group,
    which the passes bring -- is refused with its reason, not taken."""
    kept = eg.kept_span

    def none_in_the_second_block(section, step, product, k0, rows, granule):
        return None if k0 == 8 else kept(section, step, product, k0, rows,
                                          granule)
    monkeypatch.setattr(eg, 'kept_span', none_in_the_second_block)
    group = _group(6)
    assert group.section.tile is None
    assert 'block 1 of K hold' in group.tile_note
    assert 'inactive group' in group.tile_note


@pytest.mark.parametrize('arch,sm', [('sm_90a', 90), ('sm_100a', 100)])
def test_the_table_holds_each_kept_piece_as_its_descriptors_read_it(arch, sm):
    """The prepared table walked as the descriptors read it (`_Piece`): a
    chunk's pieces one after another, each its rows of K by its span's rows
    of N in the canonical K-major layout over them -- core matrices of
    eight rows of N by four of K, on wgmma the span's tiles of N last first
    (`TileRows.descending`) -- the upper TF32 halves, then the lower; each
    entry of the operator's window split (`tf32_halves`), zero outside it;
    a chunk's elements what the ring copies (`chunk_volume`)."""
    import numpy as np
    from tensorforge.common.matrix.prepare import tf32_halves
    section, step, chunks = _chunks(6, arch)
    _, numbers, offsets = eg.prepared_table(section, step, sm)
    numbers = np.asarray(numbers, dtype=np.float32)
    stages = eg._step_stages(section, step)
    at = 0
    for chunk in chunks:
        size = 0
        for p, k0, rows, lo, hi in chunk:
            stage = stages[p]
            values = np.asarray(stage.tensor.get_values(), dtype=np.float64)
            (r0, r1), (d0, d1) = stage.window
            width = hi - lo
            assert offsets[(p, k0)] == at
            for k in range(rows):
                for n in range(width):
                    place = ((n // 8) * rows * 8 + (k // 4) * 32
                             + (n % 8) * 4 + k % 4)
                    # The laid row `n`: the span's `n` on tcgen05; on wgmma
                    # its tile `n / 8` counted from the span's last.
                    row = stage.n0 + (hi - 8 * (n // 8 + 1) + n % 8
                                      if section.tile.descending else lo + n)
                    col = d0 + k0 + k
                    value = (values[row, col] if r0 <= row < r1 and col < d1
                             else 0.0)
                    upper, lower = tf32_halves(np.array([value],
                                                        dtype=np.float32))
                    assert numbers[at + place] == upper[0], (p, k0, n, k)
                    assert numbers[at + rows * width + place] == lower[0]
            at += 2 * rows * width
            size += 2 * rows * width
        assert eg.chunk_volume(section, step, chunk) == size
    assert at == len(numbers)


@pytest.mark.parametrize('groups', ['loop', 'unrolled'])
def test_the_kernel_multiplies_each_span_at_its_width(groups):
    """o8 on wgmma: a product a piece, its term and its tile -- three of
    3xTF32 a piece, each tile its own -- at the piece's width: several N;
    written out, their sum the spans' rows of N; in the group loop, a
    block's products once for each width the blocks have.  Every one of
    them into the accumulators from their first register
    (`TileRows.descending`)."""
    section, step, chunks = _chunks(8, matrix_groups=groups)
    src = _kernel(8, matrix_groups=groups).get_kernel()
    widths = [int(n) for n in re.findall(r'wgmma\.mma_async\.sync\.aligned\.'
                                         r'm64n(\d+)k8', src)]
    pieces = [hi - lo for chunk in chunks for *_, lo, hi in chunk]
    if groups == 'loop':
        assert sorted(widths) == sorted(2 * 9 * sorted(set(pieces)))
    else:
        assert len(widths) == 2 * 3 * len(pieces)
        assert sorted(widths) == sorted(2 * 3 * pieces)
    assert len(set(widths)) > 1
    firsts = re.findall(r'wgmma\.mma_async\.sync\.aligned\.m64n\d+k8[^;]*;"'
                        r'\s*:\s*"\+f"\((\w+)\)', src)
    assert len(firsts) == len(widths) and len(set(firsts)) == 2


def test_the_kernel_says_what_the_spans_keep():
    src = _kernel(8).get_kernel()
    notes = [line.strip() for line in src.splitlines()
             if line.strip().startswith('// tile:')]
    assert any('B in kept spans: 62 % of the dense rows of N, 6 widths '
               'from 40 to 120, each to the stage\'s last row' in note
               for note in notes), notes
    src = _kernel(8, matrix_spans='dense').get_kernel()
    assert any('B dense' in line for line in src.splitlines()
               if line.strip().startswith('// tile:'))


def test_the_groups_option_defaults_to_the_loop():
    from tensorforge.common.options import parse_matrix_groups, registry
    assert registry()['matrix_groups'].default == 'loop'
    assert [parse_matrix_groups(t) for t in ('loop', 'Unrolled')] == [
        'loop', 'unrolled']
    with pytest.raises(ValueError):
        parse_matrix_groups('rolled')
    assert _group(8, 'sm_100a').section.tile.groups == 'loop'


@pytest.mark.parametrize('arch', ['sm_90a', 'sm_100a'])
@pytest.mark.parametrize('order', [4, 6, 8])
def test_the_group_loop_takes_one_width_a_block(arch, order):
    """In the group loop a block's products are one width, its key in the
    dispatch: the union of their kept spans -- every entry still in its
    product's piece of its block -- the first block whole."""
    section, step, chunks = _chunks(order, arch)
    assert section.tile.groups == 'loop'
    stages = eg._step_stages(section, step)
    for chunk in chunks:
        assert len(chunk) == len(step.products)
        assert len({piece[3:] for piece in chunk}) == 1
    assert chunks[0][0][3:] == (0, stages[0].width)
    _, _, apart = _chunks(order, arch, matrix_groups='unrolled')
    for loop, unrolled in zip(chunks, apart):
        lo = min(piece[3] for piece in unrolled)
        assert all(piece[3] == lo for piece in loop)


def test_the_kernel_walks_the_blocks_in_one_loop():
    """o8: each tile's eleven blocks one rolled loop, their widths a table
    of the generated file read once a block, one dispatch on them: six
    widths, a branch each -- the widths 40 to 120 as the spans give them."""
    section, step, chunks = _chunks(8)
    src = _kernel(8).get_kernel()
    assert len(re.findall(r'#pragma unroll 1\s*for \(int32_t \w+ = 0; '
                          r'\w+ < 11; \+\+\w+\)', src)) == 2
    (table,) = re.findall(r'const int32_t (tf_tw_\w+)\[11\] = \{([^}]*)\}',
                          src)
    eighths = [int(x) for x in table[1].split(',')]
    assert eighths == [(chunk[0][4] - chunk[0][3]) // 8 for chunk in chunks]
    assert sorted(set(eighths)) == [5, 9, 11, 13, 14, 15]
    assert src.count(f'{table[0]}[') == 2 + 1
    # tcgen05: the star role's band loop and the mma role's, one block's
    # nine products, N and D's first column read off two tables.
    section, step, chunks = _chunks(8, 'sm_100a')
    src = _kernel(8, 'sm_100a').get_kernel()
    assert src.count('#pragma unroll 1') == 2
    assert src.count('tcgen05.mma.cta_group') == 9
    spans = [chunk[0][3:] for chunk in chunks]
    for prefix, numbers in (('tl', [lo for lo, _ in spans]),
                            ('tn', [hi - lo for lo, hi in spans])):
        (found,) = re.findall(rf'const int32_t tf_{prefix}_\w+\[11\] = '
                              rf'\{{([^}}]*)\}}', src)
        assert [int(x) for x in found.split(',')] == numbers


def test_two_trips_of_kept_spans_on_tcgen05_on_the_oracle():
    """o6 on tcgen05, seventeen elements: two trips, the second's D written
    fresh over the first's in tensor memory -- which, unlike wgmma's
    accumulators declared zero each trip, holds the last trip's D."""
    _on_the_oracle('sm_100a', 6, 17)


@pytest.mark.parametrize('arch', ['sm_90a', 'sm_100a'])
def test_the_first_block_s_first_piece_is_whole(monkeypatch, arch):
    """The first block's first piece whole whatever its product's own span
    -- its product writes D first, every column fresh, which on tcgen05
    tensor memory holds the last trip's D in otherwise -- here with every
    span of the first block narrowed to a granule: the volume's operators
    have entries on every row of N there, so that only this shows the rule
    (`tile_chunks`)."""
    kept = eg.kept_span

    def narrowed(section, step, product, k0, rows, granule):
        span = kept(section, step, product, k0, rows, granule)
        return (0, granule) if k0 == 0 and span is not None else span
    monkeypatch.setattr(eg, 'kept_span', narrowed)
    section, step, chunks = _chunks(6, arch)
    stages = eg._step_stages(section, step)
    assert chunks[0][0][3:] == (0, stages[chunks[0][0][0]].width)


@pytest.mark.parametrize('arch,order,count,groups', [
    ('sm_90a', 4, 17, 'loop'), ('sm_100a', 6, 14, 'loop'),
    ('sm_100a', 6, 14, 'unrolled')])
def test_kept_and_dense_spans_compute_alike_on_the_oracle(arch, order,
                                                          count, groups):
    """The kept spans leave out products of zeros alone: on the oracle every
    output bit for bit what dense spans give, both against NumPy -- where
    the spans are narrower than the stage: o4 on wgmma (83 % of the dense
    rows of N), two trips, the second's elements masked past the batch; o6
    on tcgen05 (73 %; at o4 its granule of 16 keeps every row), a trip, its
    blocks in the group loop and written out -- there each product's own
    span, at its own N and D's columns from its first."""
    _, kept = _on_the_oracle(arch, order, count, matrix_groups=groups)
    _, dense = _on_the_oracle(arch, order, count, matrix_spans='dense',
                              matrix_groups=groups)
    assert (kept == dense).all()


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


@pytest.mark.parametrize('arch,engine,products,options', [
    ('sm_90a', 'HGMMA', 2 * 6 * 9, {}),
    ('sm_90a', 'HGMMA', 2 * 99, {'matrix_groups': 'unrolled'}),
    ('sm_100a', 'UTCHMMA', 9, {}),
    ('sm_100a', 'UTCHMMA', 99, {'matrix_groups': 'unrolled'})])
def test_the_build_reads_back(arch, engine, products, options, tmp_path):
    """o8 built: every product an instruction of the engine's, as many as
    the text has -- wgmma's group loop's six widths' nine each, tcgen05's
    nine, its N a value, or written out the 33 kept pieces' three each,
    every tile's -- and none of them
    serialized by ptxas: no C7510 (a wgmma serialized for a register's
    use), C7511, C7514, C7517 or C7519 (for an arrive in a divergent path:
    the group loop's dispatch on a width ptxas cannot tell is the warp's
    alike) in its report."""
    sass, report = _ptxas(arch, tmp_path, **options)
    if engine == 'HGMMA':
        assert len(re.findall(r'\bHGMMA\.64x\d+x8\b', sass)) == products
    else:
        assert tma._count(sass, engine) == products
    for code in ('C7510', 'C7511', 'C7514', 'C7517', 'C7519'):
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
