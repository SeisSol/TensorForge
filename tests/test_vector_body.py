# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""A lead-vectorised body, spelled in each target's vector types.

With a lead width above one a lane holds adjacent rows, and the multilinear
body loads, multiplies and stores them as one vector.  Two places did not
follow: the pass that folds an accumulator into a destination already holding
values typed its sum as a scalar, so the store wrote a whole vector into one
register slot; and the fused multiply-add had only the infix spelling, which
CUDA's vector structs contract into scalar FMAs where sm_100 has a paired one.
A small version of `local_flux` -- two faces, the second adding to the first
-- reaches both.
"""

from __future__ import annotations

import re

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.context import Context
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators.descriptions import MultilinearDescr
from tensorforge.generators.generator import Generator
from tensorforge.generators.lanes import LaneConfig

_M, _N = 16, 4


def _t(shape, alias, addressing, is_tmp=False):
    return Tensor(list(shape), addressing, BoundingBox([0, 0], list(shape)),
                  alias=alias, datatype=Datatype.F32, is_tmp=is_tmp)


def _faces():
    b = _t([_M, _N], "B", Addressing.PTR_BASED)
    r = _t([_M, _N], "R", Addressing.PTR_BASED)
    x = _t([_M, _N], "X", Addressing.STRIDED, is_tmp=True)
    descrs = []
    for f in range(2):
        a = _t([_M, _M], f"A{f}", Addressing.NONE)
        c = _t([_N, _N], f"C{f}", Addressing.PTR_BASED)
        descrs.append(MultilinearDescr(
            dest=SubTensor(x), ops=[SubTensor(a), SubTensor(b)],
            target=[[0, -1], [-1, 1]], permute=[[0, 1], [0, 1]]))
        descrs.append(MultilinearDescr(
            dest=SubTensor(r), ops=[SubTensor(x), SubTensor(c)],
            target=[[0, -1], [-1, 1]], permute=[[0, 1], [0, 1]],
            add=(f > 0)))
    return descrs


def _kernel(arch, backend, lanes=None):
    gen = Generator(_faces(), Context(arch=arch, backend=backend,
                                      fp_type=Datatype.F32), lanes=lanes)
    gen.generate()
    return gen.get_kernel()


#: Four lanes of four adjacent rows each.
_WIDE = LaneConfig(num_threads=4, num_active_threads=_M, lead_width=4)


def _narrowing_stores(src):
    """`r[i] = (a + b)` where `a` or `b` is a vector: a store of a vector
    value into a single register slot."""
    vectors = set(re.findall(r"VectorT<[^>]*> (v\d+_\w+) =", src))
    return [m.group(0).strip() for m in re.finditer(
        r"^\s*(?:i?r\d+)\[[^\]]+\] = \((v\d+_\w+) [-+*] (v\d+_\w+)\);",
        src, re.M)
        if m.group(1) in vectors or m.group(2) in vectors]


def test_the_accumulating_face_stores_a_whole_vector():
    src = _kernel("sm_86", "cuda", _WIDE)
    assert "VectorT<float, 4>" in src, "the body did not vectorise at all"
    assert _narrowing_stores(src) == []


def test_cuda_spells_the_vector_fma_as_a_call():
    assert "tensorforge::fma(" in _kernel("sm_86", "cuda", _WIDE)


def test_hip_keeps_the_infix_fma():
    """A GNU vector contracts `a * b + c` itself, and `hip.h` has no call
    to make."""
    src = _kernel("gfx942", "hip", _WIDE)
    assert "VectorT<float, 4>" in src
    assert "tensorforge::fma(" not in src
    assert _narrowing_stores(src) == []


def test_a_scalar_body_is_left_alone():
    assert "tensorforge::fma(" not in _kernel("sm_86", "cuda")
