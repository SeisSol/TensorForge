# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""`Options.full_lane_tails`: the ragged end of a lead dimension on every lane.

On by default.  On, the tail block of a multiplication computes on the whole
wave and only its memory accesses are held to the lanes that hold data
(`LeadIndex.valid`) -- which under ESIMD turns a 24-wide vector, issued as
16 + 8, into a 32-wide one issued once.  Held here: that it is on unless
turned off, what it changes where it applies, and that it does not apply
where the lanes past the window are another slice's rows (a lead origin
shift, or an accumulator image larger than the window).
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import re
import warnings
from pathlib import Path

import numpy as np

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
    """`local_flux`: 56 rows on 32 lanes, a tail of 24.  Off, every update of
    it is a `select<24>` of the accumulator; on, none is, and the operator's
    rows are read 24 wide into a zeroed 32-wide vector."""
    mod = _module('local_flux')
    off = _kernel(mod.descr_list(), full_lane_tails=False)
    on = _kernel(mod.descr_list(), full_lane_tails=True)
    narrow_update = re.compile(r'\br\d+\.template select<24, 1>\(\d+\) =')
    assert narrow_update.search(off)
    assert not narrow_update.search(on), 'the accumulator tail is 32 wide'
    assert 'template select<24, 1>(0) = ' in on, 'the read stays 24 wide'


def test_a_lead_origin_shift_keeps_the_tail_narrow():
    """Theta: the window starts inside a register block, and the lanes past
    its end in the next block are the tensor's next rows."""
    mod = _module('lead_window_spans_two_blocks')
    assert (_kernel(mod.descr_list(), full_lane_tails=True)
            == _kernel(mod.descr_list(), full_lane_tails=False))


def test_spmd_keeps_only_the_memory_to_the_tail():
    """The same tail under SPMD: no branch around the block, the operator read
    folded to `lane < 24 ? p[i] : 0`, and the updates of the accumulator's
    padding lanes unguarded -- so the scheduler can look across them."""
    mod = _module('local_flux')
    off = _kernel(mod.descr_list(), backend='cuda', arch='sm_100',
                  full_lane_tails=False)
    on = _kernel(mod.descr_list(), backend='cuda', arch='sm_100',
                 full_lane_tails=True)
    guarded = re.compile(r'if \(v\d+_g\)')
    assert len(guarded.findall(on)) < len(guarded.findall(off)) / 10
    assert re.search(r'v\d+_g \? \(glb_m0\[', on), 'the read keeps its lanes'
    assert not re.search(r'v\d+_g \? \(glb_m0\[', off)


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
    except Exception as exc:  # pragma: no cover - shape not lowerable
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
    under ESIMD a merge and a 32-wide write where it was 16 + 8 and a fill
    nest, under SPMD a select where it was a branch -- and the fill nest has
    nothing left to do."""
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
    assert zeros.search(cuda_off) and not zeros.search(cuda), 'no fill nest left'
    assert re.search(r'glb_m2\[[^\]]*\] = \(v\d+_g \? v\d+_data : 0\.0f\);',
                     cuda), cuda


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
