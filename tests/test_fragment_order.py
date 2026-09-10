# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""The pre-ordered storage layout, against the staging it replaces.

`nvidia.fragment_order` says where each element of a prepared `A` operand
lives.  The claim it makes is not "some permutation" but a specific one: the
order the emitter's *own* shared tile put the elements in, so that a read that
used to hit `Ashm[lane + threads * f]` hits memory at the same offset instead.

That claim is exactly checkable and nothing else checks it.  The emitter and
the layout function are two derivations of one layout, and two derivations are
how a layout comes to disagree with itself -- silently, since a wrong
permutation still runs, still fills every slot, and returns numbers.  So the
staging is replayed here as plain integer arithmetic, transcribed from the
store and the load, and the two are compared slot for slot.

The hardware's half of the layout -- which `(m, k)` of a tile a lane holds in
fragment `f` -- was verified on the device three independent ways (the `wmma`
loader, a raw `mma.sync` against a host reference, and an impulse response).
That is not repeated here, because it needs a GPU and this does not; what is
repeated is everything that could drift when this repository changes.
"""

from __future__ import annotations

import pytest

from tensorforge.backend.instructions.compute.primitives import nvidia
from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.exceptions import GenerationError
from tensorforge.common.matrix.tensor import Tensor

THREADS = 32
#: The emitter's own fragment tiling, from `matmul`.
MTILE, KTILE = 8, 4


def _staged_tile(atom):
    """`Ashm` after the store, as `slot -> (m_local, k_local)`.

    Transcribed from the store site: a lane in `threadrange(ii, atom.m)` holds
    row `t - ii` of the tile and writes its `atom.k` depth values at
    `(t - ii) * ktile + kkk * atom.m + n`, with `kkk` stepping by `ktile`.
    """
    tile = {}
    for row in range(atom.m):
        for kkk in range(0, atom.k, KTILE):
            for n in range(KTILE):
                tile[row * KTILE + kkk * atom.m + n] = (row, kkk + n)
    assert len(tile) == atom.m * atom.k
    return tile


def _staged_fragments(atom):
    """What the fragment read takes out of it: `(lane, f) -> (m, k)`.

    The load is `Afrag[iii + kf * mregs] = Ashm[lane + 32 * f]`, so this is
    the tile indexed at that offset and nothing else.
    """
    tile = _staged_tile(atom)
    aregs = (atom.m * atom.k) // THREADS
    return {(lane, f): tile[lane + THREADS * f]
            for f in range(aregs) for lane in range(THREADS)}


def _atoms():
    """Every entry the emitter can issue, over the architectures it gates on."""
    seen, out = set(), []
    for dtype in (Datatype.F32, Datatype.F64):
        for sm in (75, 80, 86, 90, 120):
            for atom in nvidia.instrs_for(dtype, sm):
                key = (atom.m, atom.n, atom.k, atom.d)
                if key not in seen:
                    seen.add(key)
                    out.append(atom)
    return out


@pytest.mark.parametrize('atom', _atoms(),
                         ids=lambda a: f'm{a.m}n{a.n}k{a.k}_{a.d.name}')
def test_the_a_layout_factors_into_bits(atom):
    """The PTX map, against the arithmetic it is stated as.

    Written out here rather than imported, because a table checked against
    its own generator checks nothing.  Every term of the map is a shift --
    `t / ktile`, `t % ktile`, `iii * mtile`, `kf * ktile` -- which is why it
    factors at all, and the point of saying so in bits is that a distribution
    stated as bits can be held against what an operand actually holds.
    """
    bits = nvidia.a_fragment_bits(atom, THREADS)
    mregs = atom.m // MTILE
    for f in range(atom.m * atom.k // THREADS):
        iii, kf = f % mregs, f // mregs
        for lane in range(THREADS):
            row = lane // KTILE + iii * MTILE
            col = lane % KTILE + kf * KTILE
            at = bits.locate(row, col)
            assert (at.slot, at.lane) == (f, lane), (f, lane, at)
            assert at.element == 0, (
                'a fragment holds one element per slot per lane; an index bit '
                'inside the register is what `strategies` refuses an operand '
                'for')


@pytest.mark.parametrize('atom', _atoms(),
                         ids=lambda a: f'm{a.m}n{a.n}k{a.k}_{a.d.name}')
def test_a_tile_fits_its_own_fragments_exactly(atom):
    """Cells and places are equal in number by construction, so a table that
    sends two cells to one place has silently dropped a third -- and the
    operand packed from it would be missing an element the kernel then reads
    as whatever was there before."""
    cells = nvidia._fragment_cells(atom, THREADS)
    assert len(cells) == atom.m * atom.k
    assert set(cells.values()) == {(row, col) for row in range(atom.m)
                                   for col in range(atom.k)}


def test_a_table_that_is_not_a_bijection_is_refused():
    """Exercised with a broken table, because the real one never trips it and
    a guard nobody can make fire is a guard nobody has read.  Giving the
    column axis the row axis's slot weights folds two cells onto one place."""
    from tensorforge.backend.instructions.compute.bitlayout import (
        Bit, BitLayout, Place)
    atom = next(a for a in _atoms() if a.m // MTILE > 1 and a.k // KTILE > 1)
    broken = BitLayout((
        tuple([Bit(Place.LANE, 1 << (2 + b)) for b in range(3)]
              + [Bit(Place.SLOT, 1)]),
        tuple([Bit(Place.LANE, 1 << b) for b in range(2)]
              + [Bit(Place.SLOT, 1 << b)
                 for b in range((atom.k // KTILE - 1).bit_length())])))
    with pytest.raises(GenerationError, match='two cells'):
        nvidia._fragment_cells(atom, THREADS, broken)


@pytest.mark.parametrize('atom', _atoms(),
                         ids=lambda a: f'm{a.m}n{a.n}k{a.k}_{a.d.name}')
def test_the_order_is_the_one_the_staging_produced(atom):
    """One tile, read back where the staged read would have looked."""
    rows, cols = atom.m, atom.k
    order = nvidia.fragment_order((rows, cols), atom, THREADS)
    assert len(order) == (atom.m * atom.k), 'one tile, no padding'

    want = _staged_fragments(atom)
    aregs = (atom.m * atom.k) // THREADS
    for f in range(aregs):
        for lane in range(THREADS):
            m_local, k_local = want[(lane, f)]
            # F-order over the bounding box: the first axis is the fast one.
            assert order[THREADS * f + lane] == m_local + rows * k_local, (
                f'fragment {f} of lane {lane}')


def test_a_tiled_operand_pads_rather_than_wrapping():
    """A matrix the tiling does not divide gets slots with no cell.

    56 rows against a 16-row tile is the shape the corpus actually has, and
    the answer that would be wrong quietly is a wrap: every slot filled, every
    cell stored, and eight rows of the operand multiplied twice.
    """
    atom = nvidia.instr_for(Datatype.F32, columns=9, lead=56, depth=56, sm=120)
    order = nvidia.fragment_order((56, 56), atom, THREADS)

    starts = nvidia.tile_starts(56, THREADS, atom.m)
    assert starts == (0, 16, 32, 48), 'the nest tiles rows in four'
    assert len(order) == 64 * 56, 'the image spans the tiling, not the matrix'

    stored = [c for c in order if c != -1]
    assert len(stored) == 56 * 56, 'every cell exactly once'
    assert len(set(stored)) == len(stored), 'and no cell twice'
    assert order.count(-1) == (64 - 56) * 56


def test_the_order_and_the_count_reach_the_tensor():
    t = Tensor([56, 56], Addressing.NONE, datatype=Datatype.F32)
    assert t.storage_map() is None and t.storage_elements() == 56 * 56

    atom = nvidia.instr_for(Datatype.F32, columns=9, lead=56, depth=56, sm=120)
    t.storage_order = nvidia.fragment_order((56, 56), atom, THREADS)
    assert t.storage_elements() == 64 * 56
    assert t.storage_volume() == 64 * 56, 'one scalar per element, so far'
    assert t.storage_map() == t.storage_order

    # The two conventions compose: the order says which element, the parts say
    # how much room it takes.
    t.storage_parts = 2
    assert t.storage_elements() == 64 * 56
    assert t.storage_volume() == 2 * 64 * 56


@pytest.mark.parametrize('order,why', [
    ((0, 1, 2), 'drops'),
    ((0, 1, 2, 3, 3), 'holds a cell twice'),
    ((0, 1, 2, 99), 'outside'),
])
def test_an_order_that_is_not_one_is_refused(order, why):
    """Every way of being wrong that a later reader could not detect.

    A short order under-sizes the buffer, a repeated cell drops an element,
    and a cell outside the box reads past it.  None of the three fails where
    it is spent -- they size an allocation -- so they are refused where they
    are stated.
    """
    t = Tensor([2, 2], Addressing.NONE, datatype=Datatype.F32)
    with pytest.raises(GenerationError, match=why):
        t.storage_order = order


# -- the host half ---------------------------------------------------------- #

def test_the_packer_zeroes_a_slot_with_no_cell():
    """`layout.pack` is a gather, and a padding slot gathers from nowhere.

    The bug this forecloses is quiet and specific: `dense[:, -1]` is a legal
    read of the *last* cell, so an unhandled `-1` does not raise -- it stores
    the wrong element, in the slots a tile hangs off the end of the matrix
    into, where a wrong value is multiplied by whatever the other operand has
    there.  The kernel's own padding registers read zero, so these must too.
    """
    import numpy as np

    from harness import layout

    # A 2x2 matrix in an order with one padding slot, cells in F-order.
    view = np.arange(2 * 2 * 2, dtype=np.float32).reshape(2, 2, 2)
    order = np.array([2, 0, -1, 3, 1], dtype=np.int64)
    out = layout.pack(view, order, Datatype.F32).reshape(2, 5)

    for b in range(2):
        cells = np.asarray(view[b]).ravel(order='F')
        assert list(out[b]) == [cells[2], cells[0], 0.0, cells[3], cells[1]]


# -- B, read where it lies -------------------------------------------------- #

def _multilinear(a_shape, b_shape, *, kslice=None, nslice=None, target=None):
    """One `X = A @ B` instruction, built the way a frontend would."""
    from tensorforge.common.matrix.boundingbox import BoundingBox
    from tensorforge.common.matrix.tensor import SubTensor
    from tensorforge.common.context import Context
    from tensorforge.generators.descriptions import MultilinearDescr

    def t(shape, alias, addressing, bbox=None, is_tmp=False):
        return Tensor(list(shape), addressing,
                      BoundingBox([0] * len(shape), list(bbox or shape)),
                      alias=alias, datatype=Datatype.F32, is_tmp=is_tmp)

    m, k = a_shape
    _, n = b_shape
    a = t(a_shape, 'A', Addressing.NONE)
    b = SubTensor(t(b_shape, 'B', Addressing.STRIDED),
                  bbox=BoundingBox([kslice or 0, nslice or 0], [k, n])
                  if (kslice or nslice) else None)
    x = t([m, n], 'X', Addressing.STRIDED, is_tmp=True)
    descr = MultilinearDescr(dest=SubTensor(x), ops=[SubTensor(a), b],
                             target=target or [[0, -1], [-1, 1]],
                             permute=[[0, 1], [0, 1]])
    return descr, Context(arch='sm_120', backend='cuda',
                          fp_type=Datatype.F32)


def _b_direct(**kw):
    """`B_direct` as the emitter would ask it, for one built instruction."""
    from tensorforge.generators.generator import Generator

    descr, ctx = _multilinear((56, 56), kw.pop('b_shape', (56, 9)), **kw)
    seen = []
    from tensorforge.backend.instructions.compute.primitives import nvidia as nv
    orig = nv.matmul

    def spy(writer, ops, context, span):
        seen.append(ops.B_direct(4, 8) if ops.B_direct else None)
        return orig(writer, ops, context, span)

    nv.matmul, was = spy, nv.ENABLED
    nv.ENABLED = True
    try:
        Generator([descr], ctx).generate()
    finally:
        nv.matmul, nv.ENABLED = orig, was
    return seen


def test_b_is_read_where_it_lies_when_the_address_says_so():
    """The plain case: a whole operand, no slice, no permutation."""
    assert _b_direct() == [True]


@pytest.mark.parametrize('kw,why', [
    # `target` is what names the axes, not `permute`: B's axis 0 becomes the
    # output index and axis 1 the contraction, so the lane bits that meant `k`
    # now mean `n` and the two `LeadIndex` would address the transpose.
    ({'target': [[0, -1], [1, -1]], 'b_shape': (9, 56)},
     'a transposed operand'),
    ({'nslice': 1}, 'an output slice off its block'),
])
def test_what_the_predicate_refuses(kw, why):
    """Every refusal is a coordinate that stops being `base + lane`.

    Not a performance question: the staging path is correct for all of these,
    and reading them directly would address the wrong element rather than the
    same one more slowly.  A refusal that turns into a `True` is the failure
    this guards, so it is checked from the outside -- by building the
    instruction and asking what the emitter would have been told.
    """
    answers = _b_direct(**kw)
    assert answers and not any(answers), why


def _merged_local_flux(monkeypatch, parts):
    """`local_flux` on the matrix path, faces 1-3 merged over a stand-in, `A`
    prepared and stored as `parts` scalars per element."""
    import contextlib
    import importlib.util
    import io
    from pathlib import Path

    from tensorforge.common.context import Context
    from tensorforge.generators.generator import Generator

    monkeypatch.setenv('TF_OPTIONS', 'prepare_operands=1,merge_variants=1')
    monkeypatch.setattr(nvidia, 'ENABLED', True)
    case = Path(__file__).parent / 'cases' / 'local_flux.py'
    spec = importlib.util.spec_from_file_location('lf_fragment_merged', case)
    mod = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(mod)
    descrs = mod.descr_list()
    for descr in descrs:
        for op in getattr(descr, 'ops', ()):
            if op.tensor.addressing is Addressing.NONE:
                op.tensor.storage_parts = parts
    gen = Generator(descrs, Context(arch='sm_120', backend='cuda',
                                    fp_type=mod.DTYPE))
    with contextlib.redirect_stdout(io.StringIO()):
        gen.generate()
    operands = [s.obj for s in gen._scopes.get_global_scope().values()
                if getattr(s.obj, 'addressing', None) is Addressing.NONE]
    return gen.get_kernel(), operands


@pytest.mark.parametrize('parts', [1, 2])
def test_a_merged_run_stores_every_member_in_the_stand_ins_order(monkeypatch,
                                                                 parts):
    """Faces 1-3 of `local_flux` merge into one body that reads whichever
    member the counter selects, in the stand-in's order and at its part
    count.  Only the stand-in used to be marked: the peeled first face was
    stored in fragment order and the other three were not, which moved the
    checksum by 0.8 % -- and by 38 % with the split on top, where the
    stand-in also read one scalar where two were stored."""
    src, operands = _merged_local_flux(monkeypatch, parts)
    members = [o for o in operands if not getattr(o, 'is_variant', False)]
    stand_ins = [o for o in operands if getattr(o, 'is_variant', False)]
    assert len(members) == 4 and len(stand_ins) == 1
    assert all(o.storage_order is not None for o in members + stand_ins)
    assert len({tuple(o.storage_order) for o in members + stand_ins}) == 1
    assert all(o.storage_parts == parts for o in members + stand_ins)
    if parts == 2:
        # Every A half is read and none is computed: the splits left are B's.
        assert src.count('splitFloatTF32') < src.count('mma.sync')
