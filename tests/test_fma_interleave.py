# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""The nest's accumulation in the order the nest writes it
(`Options.fma_interleave`), and in partial sums (`Options.fma_partial_sums`).

The nest runs the reduction outermost and every output a lane holds inside
it, and the emitter used to undo that: each step's sum is the next step's
only operand, so `inline_pir_values` folded an output's whole reduction into
one expression at its last step.  IGC keeps the order of that expression, and
on PVC most accumulating `mad`s then wait for the one right before them.

`C = A B` on PVC in double: sixteen rows on sixteen lanes, so a lane holds
one slot of each of the four columns, and eight steps.
"""

from __future__ import annotations

import contextlib
import io
import re

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.context import Context
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.common.options import Options
from tensorforge.generators.descriptions import GemmDescr
from tensorforge.generators.generator import Generator

#: Outputs a lane holds: one slot of each of the four columns.
OUTPUTS = 4
STEPS = 8

#: A step folded into an output, `vN_p = <sum so far> + (a * b);`.
STEP = re.compile(r'double (v\d+_p) = (v\d+_p|\(v\d+_data \* v\d+_bc\)) '
                  r'\+ \(v\d+_data \* v\d+_bc\);')
#: Two partial sums added up.
ADDED = re.compile(r'double (v\d+_p) = (v\d+_p) \+ (v\d+_p);')


def _t(shape, alias, addressing=Addressing.STRIDED):
    return SubTensor(Tensor(list(shape), addressing,
                            BoundingBox([0] * len(shape), list(shape)),
                            alias=alias, datatype=Datatype.F64))


def _kernel(b=Addressing.STRIDED, **options):
    gen = Generator([GemmDescr(False, False,
                               a=_t([16, STEPS], 'A', Addressing.NONE),
                               b=_t([STEPS, OUTPUTS], 'B', b),
                               c=_t([16, OUTPUTS], 'C'))],
                    Context(arch='pvc', backend='oneapi', fp_type=Datatype.F64,
                            options=Options(**options)))
    with contextlib.redirect_stdout(io.StringIO()):
        gen.generate()
    return gen.get_kernel()


def test_asked_off_it_is_the_default():
    assert _kernel(fma_interleave=False) == _kernel()


def test_without_it_an_output_is_one_expression():
    """What the option is for: the steps of one output nested into a single
    expression, the reduction innermost."""
    src = _kernel()
    assert not STEP.search(src)
    assert any(line.count(' * ') >= STEPS - 1 for line in src.splitlines())


def test_two_steps_in_a_row_write_different_outputs():
    """Each step reads the sum its output had one step back, which is every
    other output's step further up: the reduction outermost."""
    steps = STEP.findall(_kernel(fma_interleave=True))
    # The first step of each output is a product of its own, folded into
    # the second (`fold` on the zeroed accumulator).
    assert len(steps) == OUTPUTS * (STEPS - 1)
    names = [name for name, _ in steps]
    for at, (_, addend) in enumerate(steps):
        if addend.startswith('v'):
            assert names.index(addend) == at - OUTPUTS


def test_the_product_stays_in_its_step():
    """A compiler that contracts only within one expression (`-ffp-contract
    =on`) still fuses the multiply into the sum: every step is one
    statement."""
    src = _kernel(fma_interleave=True)
    for line in src.splitlines():
        if '_p = ' in line:
            assert line.count(' + ') == 1 and line.count(' * ') <= 2, line


def test_partial_sums_take_the_steps_in_turn():
    """Two partial sums an output, each every other step, added behind the
    reduction: a step reads its sum from two rounds of the outputs back."""
    src = _kernel(fma_partial_sums=2)
    steps = STEP.findall(src)
    assert len(steps) == OUTPUTS * (STEPS - 2)
    names = [name for name, _ in steps]
    for at, (_, addend) in enumerate(steps):
        if addend.startswith('v'):
            assert names.index(addend) == at - 2 * OUTPUTS
    added = ADDED.findall(src)
    assert len(added) == OUTPUTS
    last = set(names[-2 * OUTPUTS:])
    assert all({a, b} <= last for _, a, b in added)


def test_a_rolled_reduction_keeps_one_sum():
    """A rolled loop's step is a run-time value, and the body one step for
    all of them: nothing to deal, and no partial sum.  `B` batch-constant as
    well, since the reduction is rolled only over operands in memory, and
    under SPMD a per-element one is a register image."""
    src = _kernel(b=Addressing.NONE, fma_partial_sums=2, k_roll=2)
    assert re.search(r'#pragma unroll 2', src)
    assert not ADDED.search(src)
    assert '_s1' not in src
