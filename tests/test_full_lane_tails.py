# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""`Options.full_lane_tails`: the ragged end of a lead dimension on every lane.

On by default.  On, the tail block of a multiplication computes on the whole
wave and only its memory accesses are held to the lanes that hold data
(`LeadIndex.valid`) -- which under ESIMD turns a 24-wide vector, issued as
16 + 8, into a 32-wide one issued once.  Held here: that it is on unless
turned off, what it changes where it applies, that it does not apply where
the lanes past the window are another slice's rows (a lead origin shift, or
an accumulator image larger than the window), and that what it computes
beside the window stays out of a destination larger than the window.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import re
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from harness import UNSUPPORTED
from tensorforge.backend.instructions.compute.multilinear import (
    MultilinearInstruction)
from tensorforge.backend.symbol import SymbolType
from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.context import Context, Options
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.common.options import registry
from tensorforge.generators.descriptions import MultilinearDescr
from tensorforge.generators.generator import Generator

CASES = Path(__file__).resolve().parent / "cases"


def _module(name):
    path = next(CASES.rglob(f'{name}.py'))
    spec = importlib.util.spec_from_file_location('tf_flt__' + name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _kernel(descrs, backend='esimd', arch='pvc', dtype=Datatype.F32, **opts):
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter('ignore')
        gen = Generator(descrs, Context(arch=arch, backend=backend, fp_type=dtype,
                                        options=Options(**opts)))
        gen.generate()
    # the kernel's name is a hash of its source, the options line names what
    # was asked: neither is what these tests compare
    return re.sub(r'kernel_[0-9a-f]{16}|// options:.*', '', gen.get_kernel())


def test_on_by_default():
    assert registry()['full_lane_tails'].default is True


def test_the_tail_computes_on_the_whole_wave_under_esimd():
    """`local_flux`: 56 rows on 32 lanes, a tail of 24.  Off, the accumulator
    is written as a `select<24>` of its register array -- an update, or the
    sum the forward pass leaves of the updates; on, none is, and the
    operator's rows are read 24 wide into a zeroed 32-wide vector."""
    mod = _module('local_flux')
    off = _kernel(mod.descr_list(), full_lane_tails=False)
    on = _kernel(mod.descr_list(), full_lane_tails=True)
    narrow_update = re.compile(r'\bi?r\d+\.template select<24, 1>\(\d+\) =')
    assert narrow_update.search(off)
    assert not narrow_update.search(on), 'the accumulator tail is 32 wide'
    assert 'template select<24, 1>(0) = ' in on, 'the read stays 24 wide'


def test_a_lead_origin_shift_keeps_the_tail_narrow():
    """Theta: the window starts inside a register block, and the lanes past
    its end in the next block are the tensor's next rows.  The code is the
    same; the listing is not quite: with the tail on every lane the product
    promises zeros past its rows (`PromisedZeros`), an instruction that
    emits nothing but the line naming it."""
    mod = _module('lead_window_spans_two_blocks')

    def code(kernel):
        return re.sub(r'\n[ \t]*//[^\n]*', '', kernel)
    assert (code(_kernel(mod.descr_list(), full_lane_tails=True))
            == code(_kernel(mod.descr_list(), full_lane_tails=False)))


def test_spmd_keeps_only_the_memory_to_the_tail():
    """The same tail under SPMD: no branch around the block, the operator read
    folded to `lane < 24 ? p[i] : 0`, and the updates of the accumulator's
    padding lanes unguarded -- so the scheduler can look across them."""
    mod = _module('local_flux')
    # Unprepared, because the predicate below is what a tail does to a read
    # of the operand *as the frontend described it*.  Stored in the SIMT
    # interleave the tail needs none: the lanes past the data reach the
    # order's own padding, which is stored and is zero
    # (`Symbol._interleaved_load`).
    off = _kernel(mod.descr_list(), backend='cuda', arch='sm_100',
                  full_lane_tails=False, prepare_operands=False)
    on = _kernel(mod.descr_list(), backend='cuda', arch='sm_100',
                 full_lane_tails=True, prepare_operands=False)
    guarded = re.compile(r'if \(v\d+_g\)')
    assert len(guarded.findall(on)) < len(guarded.findall(off)) / 10
    assert re.search(r'v\d+_g \? \(glb_m0\[', on), 'the read keeps its lanes'
    assert not re.search(r'v\d+_g \? \(glb_m0\[', off)


def _guarded_reads(kernel, operand):
    """Branches on a lane guard with a read of `operand` right inside."""
    lines = kernel.splitlines()
    return sum(1 for i, line in enumerate(lines)
               if re.search(r'if \(v\d+_g\) \{', line)
               and any(f'{operand}[' in x for x in lines[i + 1:i + 5]))


def test_a_transfer_into_registers_reads_its_tail_on_every_lane():
    """`chain_five`: a 56-row operand into registers, on 32 lanes.  Off, the
    tail block of the transfer is a loop in a branch on the lane; on, the loop
    stands alone and reads `lane < 24 ? p[i] : 0`, zeros into the image's
    padding."""
    mod = _module('chain_five')
    off = _kernel(mod.descr_list(), backend='cuda', arch='sm_86',
                  full_lane_tails=False)
    on = _kernel(mod.descr_list(), backend='cuda', arch='sm_86',
                 full_lane_tails=True)
    folded = re.compile(r'v\d+_g \? \(__ldcg\(&peel_glb_m0\[')
    assert folded.search(on) and not folded.search(off)
    assert _guarded_reads(off, 'peel_glb_m0') and not _guarded_reads(
        on, 'peel_glb_m0')


def _slice_accumulation():
    """`t = A v` over 64 rows, then `t[0:40] += B w` into the image that
    already holds `t`, then `D = t`: the accumulator of the second product is
    a 40-row window of a 64-row image, so lanes 8..31 of its last block are
    `t`'s rows 40..63."""
    def tensor(shape, alias, tmp=False, addressing=Addressing.PTR_BASED):
        return Tensor(shape, addressing, BoundingBox([0] * len(shape), list(shape)),
                      alias=alias, is_tmp=tmp, datatype=Datatype.F32)
    t = tensor([64, 9], 't', tmp=True, addressing=Addressing.STRIDED)
    window = SubTensor(t, BoundingBox([0, 0], [40, 9]))
    return [
        MultilinearDescr(dest=SubTensor(t),
                         ops=[SubTensor(tensor([64, 64], 'A', addressing=Addressing.NONE)),
                              SubTensor(tensor([64, 9], 'v'))],
                         target=[[0, -1], [-1, 1]], permute=[[0, 1], [0, 1]]),
        MultilinearDescr(dest=window,
                         ops=[SubTensor(tensor([40, 40], 'B', addressing=Addressing.NONE)),
                              SubTensor(tensor([40, 9], 'w'))],
                         target=[[0, -1], [-1, 1]], permute=[[0, 1], [0, 1]],
                         add=True),
        MultilinearDescr(dest=SubTensor(tensor([64, 9], 'D')), ops=[SubTensor(t)],
                         target=[[0, 1]], permute=[[0, 1]]),
    ]


def test_an_accumulation_into_a_slice_of_an_image_keeps_the_tail_narrow():
    """The space-time predictor's shape without the shift: the window's end
    is not the image's, so a full-lane tail would overwrite `t[40:64]`."""
    try:
        off = _kernel(_slice_accumulation(), full_lane_tails=False)
    except UNSUPPORTED as exc:  # pragma: no cover - shape not lowerable
        import pytest
        pytest.skip(f'the slice accumulation does not lower: {exc}')
    on = _kernel(_slice_accumulation(), full_lane_tails=True)
    # the 40-row window's tail is 8 lanes; its update stays 8 wide
    assert re.search(r'select<8, 1>', off)
    assert re.search(r'select<8, 1>', on), on


def _padded_local_flux():
    """`local_flux` with its result stored as the whole `64 x 9`: the box is
    the tensor, 56 rows are computed, and the store promises zeros beyond."""
    mod = _module('local_flux')
    made = mod._tensor

    def tensor(shape, alias, *args, **kwargs):
        if alias == 'R':
            kwargs['bbox'] = None
        return made(shape, alias, *args, **kwargs)
    mod._tensor = tensor
    try:
        return mod.descr_list()
    finally:
        mod._tensor = made


def test_a_zero_filled_tail_is_written_whole():
    """The rows after the 56 computed ones are zero-filled by the store
    anyway: the tail writes them in its own write, padding lanes zeroed --
    under ESIMD a merge and a 32-wide write where off has 16 + 8 and a fill
    nest -- and the fill nest has nothing left to do.  Under SPMD the run's
    trips keep `R` in a register and store it whole behind the last: its rows
    past the 56 are the zeros it holds there, the tail on every lane or not.
    What the first face writes ahead of the run, the rows a lane holds and
    the zeros past them, the first trip reads from the registers it came
    from, and that write goes."""
    off = _kernel(_padded_local_flux(), full_lane_tails=False)
    on = _kernel(_padded_local_flux(), full_lane_tails=True)
    assert 'v' not in re.findall(r'v\d+_pad', off) and not re.search(r'v\d+_pad', off)
    assert re.search(r'v\d+_pad', on), on
    assert not re.search(r'simd<float, 24>\([^;]*\)\.copy_to\(glb_m2', on)
    cuda_off = _kernel(_padded_local_flux(), backend='cuda', arch='sm_100',
                       full_lane_tails=False)
    cuda = _kernel(_padded_local_flux(), backend='cuda', arch='sm_100',
                   full_lane_tails=True)
    zeros = re.compile(r'glb_m2\[[^\]]*\] = 0\.0f;')
    assert not zeros.search(cuda_off) and not zeros.search(cuda), 'no fill nest left'
    assert len(re.findall(r'glb_m2\[[^\]]*\] = ', cuda_off)) == 1, cuda_off
    assert len(re.findall(r'glb_m2\[[^\]]*\] = ', cuda)) == 1, cuda
    assert not re.search(r'= glb_m2\[', cuda), cuda


def test_a_tensor_stored_as_its_box_keeps_its_narrow_write():
    """Plain `local_flux`: 56 of 64 rows in the box, and stored as the box --
    there is no row after the tail to write."""
    mod = _module('local_flux')
    assert '_pad' not in _kernel(mod.descr_list(), full_lane_tails=True)


def test_a_window_ending_inside_the_data_keeps_its_narrow_write():
    """Theta: `D[20:35] +=` of a 64-row tensor.  An accumulation promises no
    zeros, and past row 35 are the tensor's own rows."""
    mod = _module('lead_window_spans_two_blocks')
    on = _kernel(mod.descr_list(), full_lane_tails=True)
    assert '_pad' not in on


# --------------------------------------------------------------------------- #
# what a window leaves out
# --------------------------------------------------------------------------- #


def _host():
    path = CASES.parent.parent / 'tools' / 'host' / 'corpus_check.py'
    spec = importlib.util.spec_from_file_location('tf_flt_corpus_check', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


HOST = _host()
GEMM = dict(target=[[0, -1], [-1, 1]], permute=[[0, 1], [0, 1]])


def _beside(shape):
    """Four faces into `R`, `64 x 9` and stored as its whole box, of which 56
    rows are computed.  `past`: `R (+)= T[0:56] C_f`, `T = A_f B` of 64 rows,
    whose image holds rows 56..63 of `T` in the lanes past the window.
    `before`: `R (+)= T[8:56] C_f`, `T` of 56 rows, rows 0..7 in the lanes
    before it.  `operator`: `R (+)= A_f[8:56] B`, the operator read from
    global memory, which the lanes before the window read whole.  The
    operators carry numbers, so that every build reads the same matrices
    whatever order it stores them in."""
    rng = np.random.default_rng(5)

    def tensor(dims, alias, addressing=Addressing.STRIDED, box=None,
               is_tmp=False):
        data = (rng.uniform(-1.0, 1.0, size=tuple(dims))
                if addressing == Addressing.NONE else None)
        return Tensor(list(dims), addressing,
                      BoundingBox([0, 0], list(box or dims)), alias=alias,
                      is_tmp=is_tmp, datatype=Datatype.F64, data=data)

    head = 0 if shape == 'past' else 8
    b = tensor([64, 9], 'B', Addressing.PTR_BASED, box=[56, 9])
    r = tensor([64, 9], 'R', Addressing.PTR_BASED)
    out = []
    if shape == 'operator':
        for f in range(4):
            a = tensor([56, 56], f'A{f}', Addressing.NONE)
            out.append(MultilinearDescr(
                dest=SubTensor(r),
                ops=[SubTensor(a, BoundingBox([head, 0], [56, 56])),
                     SubTensor(b)], add=f > 0, **GEMM))
        return out
    rows = 64 if shape == 'past' else 56
    t = tensor([rows, 9], 'T', is_tmp=True)
    for f in range(4):
        a = tensor([rows, 56], f'A{f}', Addressing.NONE)
        c = tensor([9, 9], f'C{f}', Addressing.PTR_BASED)
        out.append(MultilinearDescr(dest=SubTensor(t),
                                    ops=[SubTensor(a), SubTensor(b)], **GEMM))
        out.append(MultilinearDescr(
            dest=SubTensor(r),
            ops=[SubTensor(t, BoundingBox([head, 0], [56, 9])), SubTensor(c)],
            add=f > 0, **GEMM))
    return out


class _Beside(HOST.Kernel):
    def __init__(self, shape):
        self._shape = shape
        super().__init__(None, Datatype.F64, 'sm_86')

    def descrs(self):
        return _beside(self._shape)


BUILDS = {'written out': dict(merge_variants=False),
          'merged': dict(merge_variants=True)}


@pytest.mark.parametrize('build', list(BUILDS))
@pytest.mark.parametrize('shape', ['past', 'before', 'operator'])
def test_what_a_window_leaves_out_adds_nothing_to_the_destination(shape,
                                                                   build):
    """`R`'s rows outside the 56 its products compute are zeros, on the host
    oracle against NumPy, which zero-fills an assignment's whole box.  The
    window computes on every lane, and the lanes beside it hold what the
    products there come to -- rows of `T`, or of the operator read whole --
    and not the zeros the accumulator is declared with; the epilogue that
    writes `R`'s box keeps them out (`MultilinearEpilogue`,
    `MultilinearInstruction._beside`)."""
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter('ignore')
        kernel = _Beside(shape)
        got, runs = kernel.run(BUILDS[build])
        want = kernel.expected()
    assert runs == (0 if build == 'written out' else 1)
    wrong = sorted(key for key, value in want.items()
                   if abs(got[key] - value) > 1e-9 * max(1.0, abs(value)))
    assert not wrong, wrong[:8]


def _operand(stype, image, window, zero_past=True):
    """An operand read over rows `window` of a symbol of `stype` whose image
    holds rows `image`, nine columns each."""
    view = SimpleNamespace(get_bbox=lambda: BoundingBox([image[0], 0],
                                                        [image[1], 9]))
    symbol = SimpleNamespace(stype=stype, zero_past=zero_past, data_view=view)
    return SimpleNamespace(symbol=symbol, offset=[0, 0],
                           bbox=BoundingBox([window[0], 0], [window[1], 9]))


@pytest.mark.parametrize('stype,image,zero_past,reads_zero', [
    (SymbolType.SharedMem, (0, 64), None, True),
    (SymbolType.Register, (0, 56), True, True),
    (SymbolType.Register, (0, 64), True, False),
    (SymbolType.Register, (0, 56), False, False),
], ids=['memory', 'an image ending with the window',
        'an image with rows past it', 'an image nobody vouched for'])
def test_an_operand_reads_zero_past_its_window_where_nothing_lies_there(
        stype, image, zero_past, reads_zero):
    """Rows 0..55 read: memory holds the read to them, an image of the 56 rows
    holds the zeros its producer said it does past them, and one of 64 holds
    rows 56..63 there (`MultilinearInstruction._zero_past`)."""
    region = _operand(stype, image, (0, 56), zero_past)
    assert MultilinearInstruction._zero_past(region, 0) is reads_zero


def test_the_lanes_beside_a_window_computed_on_every_lane_are_told():
    """`_beside`: nothing where the window keeps to its rows; where it
    computes on every lane, the lanes before it, and those past it where an
    operand holds rows there."""
    def instruction(every_lane, image):
        return SimpleNamespace(
            _lead_dims=[0], _ns=[(0, 56), (0, 9)],
            _ops=[_operand(SymbolType.Register, image, (0, 56))],
            _target=[[0, -1]],
            _full_lane_tail=lambda i, lo, hi: every_lane,
            _zero_past=MultilinearInstruction._zero_past)
    beside = MultilinearInstruction._beside
    assert beside(instruction(False, (0, 64))) == {}
    assert beside(instruction(True, (0, 56))) == {0: (True, False)}
    assert beside(instruction(True, (0, 64))) == {0: (True, True)}
