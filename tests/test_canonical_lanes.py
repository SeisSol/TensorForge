# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Where a multiplication computes its destination's rows, and where its
operands' rows are staged to meet them.

The origin of the lead loop is what a pending result the operation reads in
place says -- the destination's own first -- and the destination's offset
modulo a round of the lanes where nothing does, so that row `r` of a tensor
sits in lane `(r div w) mod T` (`MultilinearBuilder._lead_origin_shift`).  An
operand is staged in the lanes the loop reads it in (`_stage_shift`), and an
image that holds its rows elsewhere is not read there (`_image_serves`,
`_target_image_fits`): a copy is staged again, a pending result goes through
memory, which is the one thing that moves a value between lanes.  So two
register images cannot pin the loop in two places, and each shape below
builds -- the first three were refused as two pins disagreeing modulo the
lane count, where every image and every staging set the origin -- and
computes what NumPy does, on the host oracle.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
from pathlib import Path

import numpy as np
import pytest

from tensorforge.backend.instructions.builders import multilinear_builder
from tensorforge.backend.residency import Residency
from tensorforge.backend.symbol import SymbolType
from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.context import Context, Options
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators.descriptions import MultilinearDescr
from tensorforge.generators.generator import Generator
from tensorforge.reference import descriptors as reference
from tensorforge.reference import kernel_eval

ROOT = Path(__file__).resolve().parent.parent
F64 = Datatype.F64
GEMM = dict(target=[[0, -1], [-1, 1]], permute=[[0, 1], [0, 1]])
#: Where the shapes are built; the oracle runs the CUDA one.
TARGETS = [('cuda', 'sm_86'), ('hip', 'gfx942'), ('hip', 'gfx1250'),
           ('hip', 'gfx1251')]
SEED = 7


def _host():
    path = ROOT / 'tools' / 'host' / 'corpus_check.py'
    spec = importlib.util.spec_from_file_location('canonical_corpus_check',
                                                  path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


HOST = _host()


def _t(shape, alias, box=None, is_tmp=False):
    lower, upper = box or ([0] * len(shape), list(shape))
    return Tensor(shape, Addressing.STRIDED, BoundingBox(lower, upper),
                  alias=alias, is_tmp=is_tmp, datatype=F64)


def _rows(tensor, lower, upper):
    """Rows `lower..upper` of `tensor`, all of its columns."""
    cols = tensor.shape[1]
    return SubTensor(tensor, BoundingBox([0, 0], [upper - lower, cols]),
                     [lower, 0], sliced=True)


def _built(descrs, backend='cuda', arch='sm_86'):
    gen = Generator(descrs, Context(arch=arch, backend=backend, fp_type=F64,
                                    options=Options()))
    with contextlib.redirect_stdout(io.StringIO()):
        gen.generate()
    return gen


def _worst(descrs):
    """The largest deviation of the kernel's outputs from NumPy's, over every
    slot they store, on the host oracle (sm_86)."""
    gen = _built(descrs)
    lanes, mults = kernel_eval.launch_geometry(gen.get_launcher())
    mem = kernel_eval.evaluate_wave(gen.get_kernel(), lanes, seed=SEED,
                                    globals_only=True, mults=mults,
                                    block=True)
    fill = kernel_eval.Slot(SEED)
    tensors = {}
    for d in descrs:
        for view in [d.dest] + list(d.ops):
            tensors.setdefault(view.tensor.alias, view.tensor)
    arrays = {}
    for alias, t in tensors.items():
        arrays[alias] = np.zeros(tuple(int(s) for s in t.shape))
        if not t.is_tmp:
            for slot, cell in HOST._slots(t).items():
                arrays[alias][cell] = fill.read(t.name, slot)
    for d in descrs:
        reference.apply({'dest': HOST._row(d.dest),
                         'ops': [HOST._row(o) for o in d.ops],
                         'target': [list(t) for t in d.target],
                         'add': bool(d.add)}, arrays)
    worst = 0.0
    for alias in {d.dest.tensor.alias for d in descrs}:
        t = tensors[alias]
        if t.is_tmp:
            continue
        for slot, cell in HOST._slots(t).items():
            got = mem.get((t.name, slot))
            if got is None:
                got = fill.read(t.name, slot)
            worst = max(worst, abs(got - arrays[alias][cell]))
    return worst


# --------------------------------------------------------------------------- #
# the shapes
# --------------------------------------------------------------------------- #

M, K, N = 20, 9, 6


def window_from_row_one():
    """`D = X Y`, then `D += A B` with `A` stored from row 1 on: the eqspp
    window of an operator whose first row is zero, as SeisSol's `kDivMT` has
    it.  The image of `A` starts inside a round of the lanes, the image of
    `D` at its start."""
    d = _t([M, N], 'D')
    a = _t([M, K], 'A', ([1, 0], [M - 1, K]))
    return [
        MultilinearDescr(dest=SubTensor(d),
                         ops=[SubTensor(_t([M, K], 'X')),
                              SubTensor(_t([K, N], 'Y'))], **GEMM),
        MultilinearDescr(dest=SubTensor(d),
                         ops=[SubTensor(a), SubTensor(_t([K, N], 'B'))],
                         add=True, **GEMM),
    ]


def image_read_for_other_rows():
    """`T = A B` kept in registers, then `D += T[8:16] C` onto `D`'s image:
    the rows `T` holds in lanes 8..15 are the ones `D` computes in lanes
    0..7."""
    t = _t([16, K], 'T', is_tmp=True)
    d = _t([8, N], 'D')
    return [
        MultilinearDescr(dest=SubTensor(t),
                         ops=[SubTensor(_t([16, K], 'A')),
                              SubTensor(_t([K, K], 'B'))], **GEMM),
        MultilinearDescr(dest=SubTensor(d),
                         ops=[SubTensor(_t([8, K], 'P')),
                              SubTensor(_t([K, N], 'Q'))], **GEMM),
        MultilinearDescr(dest=SubTensor(d),
                         ops=[_rows(t, 8, 16), SubTensor(_t([K, N], 'C'))],
                         add=True, **GEMM),
    ]


def bias_staged_for_other_rows():
    """`D = X[8:16] B` stages rows 8..15 of `X` into lanes 0..7, and then
    `X[8:16] += Y Z` adds onto those rows, which it computes in lanes
    8..15: the staged rows are the bias's box, in other lanes.  `E = W V`
    makes the kernel 32 lanes wide."""
    x = _t([32, K], 'X')
    return [
        MultilinearDescr(dest=SubTensor(_t([32, N], 'E')),
                         ops=[SubTensor(_t([32, K], 'W')),
                              SubTensor(_t([K, N], 'V'))], **GEMM),
        MultilinearDescr(dest=SubTensor(_t([8, N], 'D')),
                         ops=[_rows(x, 8, 16), SubTensor(_t([K, N], 'B'))],
                         **GEMM),
        MultilinearDescr(dest=_rows(x, 8, 16),
                         ops=[SubTensor(_t([8, K], 'Y')),
                              SubTensor(_t([K, K], 'Z'))],
                         add=True, **GEMM),
    ]


def slice_of_a_temporary():
    """`T = A B`, then `Q = T[8:16] C` into an output: the shape of
    `tests/cases/slicing/register_operand_lead.py`."""
    t = _t([16, K], 'T', is_tmp=True)
    return [
        MultilinearDescr(dest=SubTensor(t),
                         ops=[SubTensor(_t([16, K], 'A')),
                              SubTensor(_t([K, K], 'B'))], **GEMM),
        MultilinearDescr(dest=SubTensor(_t([8, N], 'Q')),
                         ops=[_rows(t, 8, 16), SubTensor(_t([K, N], 'C'))],
                         **GEMM),
    ]


def transposed_rows():
    """`D = A[8:16]^T B`, the operand stored the other way round, where a
    vendor reads it into registers as it is stored: the image keeps its
    tensor's rows, rows 8..15 in lanes 8..15, which compute nothing of
    `D`.  `E = W V` makes the kernel 32 lanes wide."""
    a = _t([32, K], 'A')
    return [
        MultilinearDescr(dest=SubTensor(_t([32, N], 'E')),
                         ops=[SubTensor(_t([32, K], 'W')),
                              SubTensor(_t([K, N], 'V'))], **GEMM),
        MultilinearDescr(dest=SubTensor(_t([8, N], 'D')),
                         ops=[_rows(a, 8, 16), SubTensor(_t([K, N], 'B'))],
                         target=[[0, -1], [-1, 1]],
                         permute=[[1, 0], [0, 1]]),
    ]


SHAPES = [window_from_row_one, image_read_for_other_rows,
          bias_staged_for_other_rows, slice_of_a_temporary, transposed_rows]


@pytest.mark.parametrize('shape', SHAPES, ids=lambda f: f.__name__)
@pytest.mark.parametrize('backend,arch', TARGETS)
def test_the_shape_builds(shape, backend, arch):
    _built(shape(), backend, arch)


@pytest.mark.parametrize('shape', SHAPES, ids=lambda f: f.__name__)
def test_the_shape_computes_what_numpy_does(shape):
    assert _worst(shape()) < 1e-9


# --------------------------------------------------------------------------- #
# the invariant
# --------------------------------------------------------------------------- #

#: The shapes where no pending result is read in place for other rows.
UNPINNED = [window_from_row_one, image_read_for_other_rows,
            bias_staged_for_other_rows, transposed_rows]


@pytest.mark.parametrize('shape', UNPINNED, ids=lambda f: f.__name__)
@pytest.mark.parametrize('backend,arch', TARGETS)
def test_a_result_nothing_pins_holds_its_rows_in_their_lanes(
        monkeypatch, shape, backend, arch):
    """What a multiplication leaves in registers for the next one is in the
    lanes of its tensor's rows: position `r` of the image holds element `r +
    shift`, and the shift is a whole number of rounds.  A copy staged for
    other rows decides nothing (`bias_staged_for_other_rows`), and neither
    does an operand's pending result the destination's own outranks
    (`image_read_for_other_rows`)."""
    held = []
    real = Residency.record_writeback

    def record(self, name, image, home, covered=None, shift=None,
               promise=None):
        held.append((name, image, list(shift or [])))
        return real(self, name, image, home, covered=covered, shift=shift,
                    promise=promise)
    monkeypatch.setattr(Residency, 'record_writeback', record)
    _built(shape(), backend, arch)
    assert held
    for name, image, shift in held:
        if (image.stype is not SymbolType.Register or not shift
                or image.lead_dims != [0] or not image.num_threads):
            continue
        span = image.num_threads * getattr(image, 'lead_width', 1)
        assert shift[0] % span == 0, (name, shift, span)


@pytest.mark.parametrize('backend,arch', TARGETS)
def test_a_pending_result_read_in_place_keeps_its_lanes(backend, arch):
    """`slice_of_a_temporary`: rows 8..15 of `T` are in lanes 8..15, which is
    where `Q`'s rows 0..7 are computed, and `T` is read there -- no store
    into its shared buffer, no barrier, no load back.  `Q` holds its rows
    where `T` did, which nothing after it reads in registers."""
    src = _built(slice_of_a_temporary(), backend, arch).get_kernel()
    assert 'store{r>s}' not in src


@pytest.mark.parametrize('backend,arch', TARGETS)
def test_an_operand_with_the_destinations_rows_is_staged_by_whole_rounds(
        monkeypatch, backend, arch):
    """`A` from row 1 on, read for the rows of `D` it has: its image starts
    at row 0, a round below the window, and not at the window's first row --
    where the loop's origin would follow the image."""
    shifts = []
    real = multilinear_builder.MultilinearBuilder._stage_shift

    def spy(self, i, absorb_lead):
        out = real(self, i, absorb_lead)
        if absorb_lead:
            shifts.append((self._ops[i].symbol.name, out))
        return out
    monkeypatch.setattr(multilinear_builder.MultilinearBuilder,
                        '_stage_shift', spy)
    _built(window_from_row_one(), backend, arch)
    assert shifts
    assert all(shift[0] == 0 for _, shift in shifts), shifts
