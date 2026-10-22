# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""What a multiplication's register image holds past its rows, said to the
passes (`compute.promise.PromisedZeros`, `IRBuilder.promise_zero`).

A result of ten rows on sixteen lanes holds lanes 10..15 of its slot all the
same.  A multiplication without an epilogue computes them, from the zeros its
operands read there, and says so: a statement per slot that emits nothing,
under the lanes past the rows.  The forward pass reads it as a store of the
zero (`test_forward.py`); here: that it is said, only where it holds, and
that it is gone by the time the body reaches the emitter.
"""

from __future__ import annotations

import contextlib
import io

import pytest

from tensorforge.backend import pir
from tensorforge.backend.instructions import abstract_instruction
from tensorforge.backend.pir.core import Op, walk_stmts
from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.context import Context, Options
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators.descriptions import MultilinearDescr
from tensorforge.generators.generator import Generator

F32 = Datatype.F32
GEMM = dict(target=[[0, -1], [-1, 1]], permute=[[0, 1], [0, 1]])
TARGETS = [('cuda', 'sm_86'), ('hip', 'gfx942'), ('hip', 'gfx1250'),
           ('hip', 'gfx1251')]


def _t(shape, alias, is_tmp=False):
    return Tensor(list(shape), Addressing.STRIDED,
                  BoundingBox([0] * len(shape), list(shape)), alias=alias,
                  is_tmp=is_tmp, datatype=F32)


def _chain():
    """`T = A B` with ten rows, read as the operand of `D = T C`: `T` is
    left in registers."""
    t = _t([10, 6], 'T', is_tmp=True)
    return [
        MultilinearDescr(dest=SubTensor(t),
                         ops=[SubTensor(_t([10, 5], 'A')),
                              SubTensor(_t([5, 6], 'B'))], **GEMM),
        MultilinearDescr(dest=SubTensor(_t([10, 4], 'D')),
                         ops=[SubTensor(t), SubTensor(_t([6, 4], 'C'))],
                         **GEMM),
    ]


def _slices():
    """`T = A B` with twenty rows, `U = T[0:10] C`, which reads ten of them,
    then `V = U E` and `W = V F`."""
    t = _t([20, 6], 'T', is_tmp=True)
    u = _t([10, 4], 'U', is_tmp=True)
    v = _t([10, 3], 'V', is_tmp=True)
    return [
        MultilinearDescr(dest=SubTensor(t),
                         ops=[SubTensor(_t([20, 5], 'A')),
                              SubTensor(_t([5, 6], 'B'))], **GEMM),
        MultilinearDescr(dest=SubTensor(u),
                         ops=[SubTensor(t, BoundingBox([0, 0], [10, 6])),
                              SubTensor(_t([6, 4], 'C'))], **GEMM),
        MultilinearDescr(dest=SubTensor(v),
                         ops=[SubTensor(u), SubTensor(_t([4, 3], 'E'))],
                         **GEMM),
        MultilinearDescr(dest=SubTensor(_t([10, 2], 'W')),
                         ops=[SubTensor(v), SubTensor(_t([3, 2], 'F'))],
                         **GEMM),
    ]


def _bodies(monkeypatch, backend, arch, descrs=None):
    """The bodies the pipeline was handed and what it made of them."""
    seen = []
    real = pir.optimize

    def spy(body, *args, **kwargs):
        out = real(body, *args, **kwargs)
        seen.append((body, out))
        return out
    monkeypatch.setattr(abstract_instruction.pir, 'optimize', spy)
    # Without shards: a section that may hold them is placed and built a
    # second time with them (`Generator._place_section`), and every body
    # would be seen twice.
    gen = Generator(descrs or _chain(),
                    Context(arch=arch, backend=backend, fp_type=F32,
                            options=Options(preload_shards=False)))
    with contextlib.redirect_stdout(io.StringIO()):
        gen.generate()
    return seen, gen.get_kernel()


def _promises(body):
    return [s for s in walk_stmts(body)
            if s.op == Op.MARK and s.attr('mark') == 'zeros']


def _said(seen):
    """The statements said of each image, by image."""
    images = {}
    for body, _ in seen:
        for s in _promises(body):
            images.setdefault(id(s.args[0]), []).append(s)
    return images


@pytest.mark.parametrize('backend,arch', TARGETS)
def test_a_result_of_fewer_rows_than_lanes_says_the_rest_are_zero(
        monkeypatch, backend, arch):
    """The slot of `T`'s image says that its lanes past row 10 hold a zero
    -- one statement a column, under the lanes' condition.  `D`'s is
    written by an epilogue, which keeps to the rows: past them the image
    holds the zeros it is declared with, and nothing needs saying."""
    seen, _ = _bodies(monkeypatch, backend, arch)
    images = _said(seen)
    assert all(s.predicate is not None for v in images.values() for s in v)
    assert sorted(len(v) for v in images.values()) == [6]


@pytest.mark.parametrize('backend,arch', TARGETS)
def test_an_operand_read_through_a_window_reads_zero_past_it(
        monkeypatch, backend, arch):
    """`U` reads rows 0..9 of `T` where `T`'s multiplication stored it, a
    read held to the window (`LeadIndex.valid`) though rows 10..19 follow in
    the buffer: what `U` computes past the rows is a zero, and `U` says so.
    So does `V`, which reads `U` where it was stored, and `T`, which reads
    `A` from memory."""
    seen, _ = _bodies(monkeypatch, backend, arch, _slices())
    assert sorted(len(v) for v in _said(seen).values()) == [3, 4, 6]


@pytest.mark.parametrize('backend,arch', TARGETS)
def test_the_promise_is_gone_before_the_emitter(monkeypatch, backend, arch):
    seen, kernel = _bodies(monkeypatch, backend, arch)
    assert not [s for _, out in seen for s in _promises(out)]
    assert '0 past' not in kernel.split('// operations:')[0]
