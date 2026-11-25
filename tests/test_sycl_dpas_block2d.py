# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""`Options.sycl_dpas_block2d`: the SPMD DPAS path's operator copies laid out
as 2D surfaces, each tile read by one 2D block message of the sub-group
(`intel_spmd.surface_order`, `Tensor.storage_surface`,
`MatmulOperands.A_block2d`).

What a wrong answer would come from: the surface's layout -- a block of a
tile's rows has to hand every lane the share the tiles' two vectors hand it
-- and the reads the kernel makes of it.  The first is held here against the
tiles in NumPy, the second against the copies and the kernel the generator
writes.  The device is tfemu's (`~/tf-probe/pvcport/s4/emu_s4.sh`) and the
Max 1550's.
"""

import contextlib
import io
import re
import warnings

import numpy as np
import pytest

import seissol_suite as suite
from tensorforge import toolchain
from tensorforge.backend.instructions.compute.primitives import intel
from tensorforge.backend.instructions.compute.primitives import \
    intel_spmd as ds
from tensorforge.common.basic_types import Datatype
from tensorforge.common.context import Context, Options
from tensorforge.common.matrix import forms
from tensorforge.frontend.yateto import DescriptionReader
from tensorforge.generators.generator import Generator


@pytest.fixture(autouse=True)
def _no_ocloc(monkeypatch):
    """Every build as on a machine with no `ocloc` to ask, as in
    `test_sycl_dpas`: what is written does not depend on the test machine's
    IGC."""
    monkeypatch.setattr(toolchain, 'intel_dpas_tf32', lambda *a, **k: None)


def _ctx(backend='oneapi', **options):
    return Context(arch='pvc', backend=backend, fp_type=Datatype.F32,
                   options=Options(**options))


ON = dict(sycl_dpas=True, sycl_dpas_block2d=True)

#: The rows of a tile on the surface: its two halves of `DEPTH` steps.
ROWS = ds.SPLIT_TERMS * ds.DEPTH


def _block(stored, tile):
    """What one 2D block of a tile's rows hands the lanes, as
    `tensorforge::dpasBlock2d` reads it from the surface `stored`:
    `(lane, element)`, lane `l` column `l` of the rows in order."""
    surface = np.asarray(stored, np.float32).reshape(-1, ds.SUB_GROUP)
    return surface[ROWS * tile:ROWS * (tile + 1)].T


def _tile(planar, length, tile):
    """The tiles' share of tile `tile` as a lane reads its two vectors:
    `(lane, element)` of either part, the parts `length` apart."""
    planar = np.asarray(planar, np.float32)
    part = slice(tile * ds.TILE, (tile + 1) * ds.TILE)
    return tuple(planar[p * length:(p + 1) * length][part].reshape(
        ds.SUB_GROUP, ds.DEPTH) for p in range(ds.SPLIT_TERMS))


def test_the_surface_cells_are_the_tile_cells_transposed():
    tile, surface = ds.tile_cells(), ds.surface_cells()
    assert len(set(surface)) == ds.TILE and set(surface) == set(tile)
    for element in range(ds.DEPTH):
        for lane in range(ds.SUB_GROUP):
            assert (surface[element * ds.SUB_GROUP + lane]
                    == tile[lane * ds.DEPTH + element])


@pytest.mark.parametrize('window, head', [
    (((0, 64), (0, 56)), 0),     # whole tiles
    (((16, 48), (0, 35)), 0),    # from the second slot, a padded chunk
    (((0, 48), (1, 53)), 1),     # a contraction from a column inside a block
    (((0, 16), (3, 17)), 7),     # across blocks of sixteen steps
])
def test_a_block_of_the_surface_is_a_tile_as_the_lanes_read_it(window,
                                                               head):
    """Every tile's block: lane `l`'s first eight elements are the upper
    halves the tiles' first part hands it, its next eight the lower ones of
    the second part -- the numbers the host writes, as it writes them."""
    shape = (64, 60)
    flat = np.random.default_rng(7).uniform(-1, 1, shape).astype(
        np.float32).flatten(order='F')
    tiles = ds.SplitTiles(ds.fragment_order(shape, window, head))
    surfaced = ds.surface_order(tiles, _ctx(**ON))
    assert surfaced.surface == (ds.SUB_GROUP, ds.TILE)
    assert surfaced.parts == ds.SPLIT_TERMS

    def values(order):
        return np.array([flat[c] if c >= 0 else 0.0 for c in order],
                        np.float32)
    form = forms.stored(ds.SPLIT_TERMS)
    planar = form.store(values(tiles), planar=len(tiles))
    stored = form.store(values(surfaced), planar=surfaced.surface[1])
    for t in range(len(tiles) // ds.TILE):
        got = _block(stored, t)
        hi, lo = _tile(planar, len(tiles), t)
        assert np.array_equal(got[:, :ds.DEPTH], hi)
        assert np.array_equal(got[:, ds.DEPTH:], lo)


def test_only_the_option_on_the_path_lays_a_surface():
    tiles = ds.SplitTiles(ds.fragment_order((64, 56), ((0, 64), (0, 56)),
                                            0))
    assert ds.surface_order(tiles, _ctx(sycl_dpas=True)) is None
    assert ds.surface_order(tiles, _ctx(sycl_dpas_block2d=True)) is None
    assert ds.surface_order(tuple(tiles), _ctx(**ON)) is None
    assert intel.surface_order(tiles, _ctx(**ON)) is not None
    assert intel.surface_order(
        tiles, _ctx('esimd', tensor_cores=True, sycl_dpas_block2d=True,
                    enable_move_loads=False)) is None


def _volume(order=6, **options):
    descrs = DescriptionReader(None, {}).read(suite.description(
        'elastic-linearck', f'elastic-linearck-o{order}-s', 'gpu_volume'))[0]
    gen = Generator(descrs, _ctx(**options))
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter('ignore')
        gen.generate()
    return gen


#: A read of a tile as the tiles' build makes it: one vector of eight a part.
VECTOR = re.compile(r'\*\(tensorforge::NativeFloat8\*\)&glb_'
                    r'TENSORFORGE_OPERATOR_COPY_\d+\[')

#: One 2D block of a tile's rows, in OpenCL's own vector: the surface's
#: height and the row, the anchor's value or one past it by a constant.
BLOCK = re.compile(r'tensorforge::NativeFloat16 \w+ = tensorforge::dpasBlock2d'
                   r'\(item\.get_sub_group\(\), glb_TENSORFORGE_OPERATOR_'
                   r'COPY_\d+, 64, (\d+), 64, 0, (\w+)\);')

#: A row past the anchor (`_Product._anchor`).
PAST = re.compile(r'int32_t (\w+_slot) = \w+_dz \+ (\d+);')


def _rows(kernel):
    """`(height, row)` of every block the kernel reads."""
    past = dict(PAST.findall(kernel))
    return [(int(height), int(past.get(row, 0)))
            for height, row in BLOCK.findall(kernel)]


def _copies(gen):
    return {t.alias: t for t in gen._module_operands}


def test_the_volume_reads_a_block_a_tile():
    """SeisSol's order-6 volume: each copy of `kDivM` a surface sixteen
    floats wide, the halves planar a tile at a time, holding the tiles'
    numbers as a block hands them out; one block read where the tiles'
    build makes two vector reads, on rows inside the surface, and the same
    DPAS."""
    tiles = _volume(sycl_dpas=True)
    blocks = _volume(**ON)
    flat, surface = tiles.get_kernel(), blocks.get_kernel()
    reads = _rows(surface)
    assert 2 * len(reads) == len(VECTOR.findall(flat)) > 0
    assert surface.count('dpasBlock2d(') == len(reads)
    assert not VECTOR.search(surface)
    assert surface.count(f'{ds.DPAS}(') == flat.count(f'{ds.DPAS}(')
    for height, row in reads:
        assert row % ROWS == 0 and row + ROWS <= height
    assert len({row for _, row in reads}) > 1
    before, after = _copies(tiles), _copies(blocks)
    assert set(before) == set(after) and len(after) == 3
    for alias, copy in after.items():
        assert copy.storage_surface == (ds.SUB_GROUP, ds.TILE)
        assert before[alias].storage_surface is None
        length = len(before[alias].storage_order)
        planar, stored = (before[alias].storage_values(),
                          copy.storage_values())
        assert len(stored) == ds.SPLIT_TERMS * length
        for t in range(length // ds.TILE):
            got = _block(stored, t)
            hi, lo = _tile(planar, length, t)
            assert np.array_equal(got[:, :ds.DEPTH], hi)
            assert np.array_equal(got[:, ds.DEPTH:], lo)


def test_an_operator_the_caller_prepares_keeps_its_tiles():
    """Without the copies the operators are the caller's buffers, laid out
    by the host, and not known to start on a cache line: tiles, as without
    the option."""
    options = dict(mma_operator_copies=False, prepare_operands=True)
    kernel = _volume(**ON, **options).get_kernel()
    assert 'dpasBlock2d' not in kernel and 'subGroupBlock2d' not in kernel
    assert (kernel.count(f'{ds.DPAS}(')
            == _volume(sycl_dpas=True, **options).get_kernel().count(
                f'{ds.DPAS}('))


def test_a_block_kernel_parses():
    from harness import syntax
    if syntax.compiler() is None:
        pytest.skip('no host C++ compiler')
    result = syntax.check_source(_volume(**ON).get_kernel(),
                                 shim=syntax.SYCL_SHIM)
    assert result.ok, result.errors()


def _norm(text):
    """A kernel but for its name and the options comment, which follow the
    options asked."""
    text = re.sub(r'kernel_[0-9a-f]{16}', 'K', text)
    return re.sub(r'// options: [^\n]*', '', text)


def test_the_option_alone_changes_nothing():
    assert (_norm(_volume(sycl_dpas_block2d=True).get_kernel())
            == _norm(_volume().get_kernel()))


def test_auto_reads_blocks_up_to_four_slots():
    """`auto` (`intel_spmd.AUTO_SLOTS`): the order-6 volume's products span
    four lane slots and read blocks; the order-8 volume's span eight and
    keep their tiles -- the kernel the tiles' build writes."""
    six = _volume(sycl_dpas=True, sycl_dpas_block2d='auto').get_kernel()
    assert _rows(six) and not VECTOR.search(six)
    eight = _volume(order=8, sycl_dpas=True,
                    sycl_dpas_block2d='auto').get_kernel()
    assert 'dpasBlock2d' not in eight
    assert _norm(eight) == _norm(_volume(order=8, sycl_dpas=True).get_kernel())
