# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""What a descriptor states beyond what the builders take, rewritten
(`generators.legalize`).

An operand on other axes than its destination, an accumulation, a factor, an
operand narrower than the destination, a ternary on one condition per
element: yateto states each of them, and so may the Python API, and both get
the same descriptors from one place.  What is pinned here is what each
becomes, that the names of the scratch tensors depend on the list alone, that
the generator takes a stated descriptor, and -- interpreted on the host --
what such a kernel computes.
"""

from __future__ import annotations

import contextlib
import io
import math
import warnings

import numpy as np
import pytest

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.context import Context
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.common.operation import AddOperator, Operation
from tensorforge.generators import elementwise as ew
from tensorforge.generators.descriptions import (ElementwiseDescr,
                                                 MultilinearDescr,
                                                 ReductionDescr)
from tensorforge.generators.generator import Generator
from tensorforge.generators.legalize import legalize
from tensorforge.reference import kernel_eval

F32 = Datatype.F32


def _view(alias, shape, box=None, datatype=F32):
    tensor = Tensor(list(shape), Addressing.STRIDED,
                    BoundingBox([0] * len(shape), list(shape)), alias=alias,
                    datatype=datatype)
    return SubTensor(tensor, None if box is None else BoundingBox(*box))


def _scalar(alias):
    return SubTensor(Tensor([], Addressing.SCALAR, alias=alias, datatype=F32))


def _box(view):
    return list(view.bbox.lower()), list(view.bbox.upper())


# --------------------------------------------------------------------------- #
# What each statement becomes
# --------------------------------------------------------------------------- #

def test_what_the_builders_take_comes_back_as_it_is():
    d = ElementwiseDescr(Operation.ADD, _view('C', [8, 8]),
                         [_view('A', [8, 8]), 1.0])
    out = legalize([d])
    assert len(out) == 1 and out[0] is d


def test_an_accumulation_writes_a_scratch_that_is_added_on():
    c = _view('C', [8, 8])
    d = ElementwiseDescr(Operation.EXP, c, [_view('A', [8, 8])], add=True)
    d.condition = []
    computed, put = legalize([d])
    assert isinstance(computed, ElementwiseDescr)
    assert computed.dest.tensor.alias == '_accum0'
    assert computed.dest.tensor.is_tmp
    assert isinstance(put, MultilinearDescr) and put.add
    assert put.dest.tensor is c.tensor
    assert put.ops[0].tensor is computed.dest.tensor
    # under the statement's guard, both of them
    assert computed.condition is d.condition and put.condition is d.condition


def test_a_factor_scales_what_was_written():
    c, w = _view('C', [8, 4]), _scalar('w')
    d = ElementwiseDescr(Operation.EXP, c, [_view('A', [8, 4])], alpha=w)
    computed, scaled = legalize([d])
    assert computed.dest is c
    assert scaled.dest is c and scaled.ops[0] is c
    assert scaled.ops[1].tensor is w.tensor
    assert scaled.target == [[0, 1], []]
    assert not scaled.add


def test_a_factor_on_an_accumulation_scales_the_scratch():
    """It multiplies what was computed, not what it is added to."""
    d = ElementwiseDescr(Operation.EXP, _view('C', [8]), [_view('A', [8])],
                         add=True, alpha=_scalar('w'))
    computed, scaled, put = legalize([d])
    assert scaled.dest is computed.dest
    assert put.ops[0].tensor is computed.dest.tensor


@pytest.mark.parametrize('axes,shape', [([1, 0], [4, 8]), ([1], [4])])
def test_an_operand_on_other_axes_is_copied_onto_the_destinations(axes,
                                                                    shape):
    """`C[i,j] = A[i,j] + B[j,i]`, and `C[i,j] = A[i,j] + b[j]`."""
    c, a, b = _view('C', [8, 4]), _view('A', [8, 4]), _view('B', shape)
    d = ElementwiseDescr(Operation.ADD, c, [a, b], target=[[0, 1], axes])
    copy, computed = legalize([d])
    assert isinstance(copy, MultilinearDescr)
    assert copy.dest.tensor.alias == '_conform0'
    assert copy.ops == [b] and copy.target == [axes]
    assert _box(copy.dest) == _box(c)
    assert computed.srcs == [a, copy.dest]


def test_an_axis_the_destination_does_not_carry_is_refused():
    d = ElementwiseDescr(Operation.EXP, _view('C', [8]), [_view('A', [8, 4])],
                         target=[[0, -1]])
    with pytest.raises(NotImplementedError, match='does not carry'):
        legalize([d])


def test_an_operand_narrower_than_its_destination_is_zero_outside():
    """Three of eight entries stored: the destination is cut at the edge,
    and the operand is the number 0 beyond it."""
    c, a = _view('C', [8]), _view('A', [8], box=([0], [3]))
    d = ElementwiseDescr(Operation.EXP, c, [a], target=[[0]])
    inside, outside = legalize([d])
    assert _box(inside.dest) == ([0], [3]) and _box(inside.srcs[0]) == ([0], [3])
    assert _box(outside.dest) == ([3], [8]) and outside.srcs == [0]
    # each piece owns its part; the destination owes nothing beyond it
    assert inside.dest.sliced and outside.dest.sliced


def test_pieces_of_a_window_that_owes_zeros_are_assembled_aside():
    """A destination written through a window narrower than the tensor
    defines the whole tensor, so its pieces are assembled in a scratch,
    which one multilinear assigns, zeros included."""
    c = _view('C', [8], box=([0], [6]))
    assert c.owed_zeros() is not None
    a = _view('A', [8], box=([0], [3]))
    d = ElementwiseDescr(Operation.EXP, c, [a], target=[[0]])
    *pieces, assign = legalize([d])
    assert [p.dest.tensor.alias for p in pieces] == ['_pieces0'] * 2
    assert isinstance(assign, MultilinearDescr) and not assign.add
    assert assign.dest.tensor is c.tensor and _box(assign.dest) == _box(c)


def test_a_ternary_on_one_condition_per_element_is_two_guarded_statements():
    c = _view('C', [8])
    yes, no = _view('Y', [8]), _view('N', [8])
    cond = SubTensor(Tensor([], Addressing.STRIDED, alias='q',
                            datatype=Datatype.BOOL))
    first = ElementwiseDescr(Operation.SELECT, c, [yes, no, cond])
    second = ElementwiseDescr(Operation.SELECT, c, [no, yes, cond])
    first.condition = second.condition = []
    out = legalize([first, second])
    assert all(isinstance(x, MultilinearDescr) for x in out)
    assert [x.ops[0] for x in out] == [yes, no, no, yes]
    guards = [[(g.tensor.tensor.alias, g.version, g.negated)
               for g in x.condition] for x in out]
    # a version per hoist, so that the two do not merge into one region
    assert guards == [[('q', -1, False)], [('q', -1, True)],
                      [('q', -2, False)], [('q', -2, True)]]


def test_a_ternary_writing_its_own_condition_stays_a_select():
    """The second half would read the condition the first half wrote."""
    cond = SubTensor(Tensor([], Addressing.STRIDED, alias='q',
                            datatype=Datatype.BOOL))
    d = ElementwiseDescr(Operation.SELECT, cond, [1.0, 0.0, cond])
    assert legalize([d]) == [d]


def test_a_reduction_that_accumulates_and_is_scaled():
    c, a, w = _view('C', [8]), _view('A', [8, 4]), _scalar('w')
    d = ReductionDescr(c, a, [1], AddOperator(), add=True, alpha=w)
    reduced, scaled, put = legalize([d])
    assert isinstance(reduced, ReductionDescr) and reduced.legal()
    assert reduced.dest.tensor.alias == '_accum0'
    assert scaled.dest is reduced.dest
    assert put.dest.tensor is c.tensor and put.add


def test_the_scratch_names_count_across_the_list():
    """And depend on nothing else: the same list legalized twice is named
    the same twice."""
    def stated():
        return [ElementwiseDescr(Operation.EXP, _view(f'C{k}', [8]),
                                 [_view(f'A{k}', [8])], add=True)
                for k in range(2)]
    for _ in range(2):
        out = legalize(stated())
        assert [x.dest.tensor.alias for x in out[0::2]] == ['_accum0',
                                                             '_accum1']


# --------------------------------------------------------------------------- #
# Identities
# --------------------------------------------------------------------------- #

def _legalized(d):
    out, = legalize([d])
    return out


@pytest.mark.parametrize('exponent', [0.5, -0.5, 1 / 3, -1 / 3, 3.0])
def test_a_power_is_the_power_where_a_root_is_another_function(exponent):
    """`sqrt(-0.0)` is -0 where `pow(-0.0, 0.5)` is +0, and `cbrt(-8.0)` is
    -2 where `pow(-8.0, 1/3)` is a NaN: a kernel that means the root asks for
    it by name."""
    d = ew.pow(_view('B', [16]), _view('A', [16]), exponent)
    assert _legalized(d).op == Operation.POW


def test_a_base_of_e_is_the_power():
    """`math.e` is not e, and `pow` with it is not `exp`."""
    d = ew.pow(_view('B', [16]), math.e, _view('A', [16]))
    assert _legalized(d).op == Operation.POW


@pytest.mark.parametrize('build,x,y,op,srcs', [
    (ew.pow, 'A', 2, Operation.MUL, ('A', 'A')),
    (ew.pow, 'A', 2.0, Operation.MUL, ('A', 'A')),
    (ew.pow, 'A', 1, Operation.COPY, ('A',)),
    (ew.pow, 'A', -1.0, Operation.RCP, ('A',)),
    (ew.pow, 1.0, 'A', Operation.COPY, (1.0,)),
    (ew.mul, 1, 'A', Operation.COPY, ('A',)),
    (ew.mul, 'A', -1.0, Operation.NEG, ('A',)),
    (ew.div, 1.0, 'A', Operation.RCP, ('A',)),
    (ew.div, 'A', 1, Operation.COPY, ('A',)),
    (ew.div, 'A', -1, Operation.NEG, ('A',)),
])
def test_an_operation_that_is_the_same_function_stands_in(build, x, y, op,
                                                          srcs):
    """Signed zeros, infinities and NaN included."""
    tensors = {'A': _view('A', [16])}
    d = _legalized(build(_view('B', [16]), tensors.get(x, x),
                         tensors.get(y, y)))
    assert d.op == op
    assert [s.tensor.alias if hasattr(s, 'tensor') else s
            for s in d.srcs] == list(srcs)


def test_an_identity_keeps_the_rest_of_the_statement():
    """The guard, the accumulation and the axes the operand is stated on."""
    c, a = _view('C', [8, 4]), _view('A', [4, 8])
    d = ElementwiseDescr(Operation.MUL, c, [a, 1.0], target=[[1, 0], None],
                         add=True)
    d.condition = []
    copy, computed, put = legalize([d])
    assert copy.ops == [a] and copy.target == [[1, 0]]
    assert computed.op == Operation.COPY and computed.srcs == [copy.dest]
    assert put.add and put.dest.tensor is c.tensor
    assert all(x.condition is d.condition for x in (copy, computed, put))


# --------------------------------------------------------------------------- #
# Through the generator
# --------------------------------------------------------------------------- #

def _computed(descrs, arch='sm_86'):
    gen = Generator(descrs, Context(arch=arch, backend='cuda', fp_type=F32))
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter('ignore')
        gen.generate()
    lanes, mults = kernel_eval.launch_geometry(gen.get_launcher())
    mem = kernel_eval.evaluate_wave(gen.get_kernel(), lanes, seed=3,
                                    globals_only=True, mults=mults)

    def read(view, n):
        return np.array([mem.get((view.tensor.name, k), np.nan)
                         for k in range(n)])
    return read


def test_the_generator_takes_a_transposed_operand():
    """`C[i,j] = A[i,j] + B[j,i]`, as the kernel computes it."""
    c, a, b = _view('C', [8, 4]), _view('A', [8, 4]), _view('B', [4, 8])
    read = _computed([ElementwiseDescr(Operation.ADD, c, [a, b],
                                       target=[[0, 1], [1, 0]])])
    out, x, y = read(c, 32), read(a, 32), read(b, 32)
    want = [x[i + 8 * j] + y[j + 4 * i] for j in range(4) for i in range(8)]
    assert np.allclose(out, want), out


def test_the_generator_takes_a_narrower_operand():
    """Zero beyond the three entries stored: `exp(0)` is one."""
    c, a = _view('C', [8]), _view('A', [8], box=([0], [3]))
    read = _computed([ElementwiseDescr(Operation.EXP, c, [a], target=[[0]])])
    out, x = read(c, 8), read(a, 3)
    assert np.allclose(out, list(np.exp(x)) + [1.0] * 5, rtol=1e-6), out
