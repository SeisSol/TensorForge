# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""The nest's accumulation in the order the nest writes it
(`Options.fma_interleave`), in groups of outputs (`Options.
fma_interleave_group`), and in partial sums (`Options.fma_partial_sums`).

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


# -- groups of outputs ------------------------------------------------------- #

#: A step of the nest's accumulation, whatever its first factor is called (a
#: lane's component of a vector read is a `_p` too).
ANY_STEP = re.compile(r'double (v\d+_p) = (v\d+_p|\(v\d+_\w+ \* v\d+_bc\)) '
                      r'\+ \(v\d+_\w+ \* v\d+_bc\);')
#: An output's sum written into the accumulator's image.
STORED = re.compile(r'ir\d+\[(\d+)\] = (v\d+_p);')


def _grouped(rows, **options):
    """`C = A B` over `rows` rows on sixteen lanes, four columns and eight
    steps, interleaved with `options`."""
    gen = Generator([GemmDescr(False, False,
                               a=_t([rows, STEPS], 'A', Addressing.NONE),
                               b=_t([STEPS, OUTPUTS], 'B'),
                               c=_t([rows, OUTPUTS], 'C'))],
                    Context(arch='pvc', backend='oneapi', fp_type=Datatype.F64,
                            options=Options(fma_interleave=True, **options)))
    with contextlib.redirect_stdout(io.StringIO()):
        gen.generate()
    return gen.get_kernel()


def _groups(src, slots):
    """The outputs whose reductions overlap, `[{(slot, column), ...}, ...]`
    in the order they run: each output's chain of sums from its first step
    to its last, named by the image entry it ends in."""
    chain_of, spans, ends = {}, [], {}
    for at, line in enumerate(src.splitlines()):
        step = ANY_STEP.search(line)
        if step:
            name, addend = step.groups()
            chain = chain_of.get(addend)
            if chain is None:
                chain = len(spans)
                spans.append([at, at])
            spans[chain][1] = at
            chain_of[name] = chain
            continue
        stored = STORED.search(line)
        if stored and stored.group(2) in chain_of:
            entry = int(stored.group(1))
            ends[chain_of[stored.group(2)]] = (entry % slots, entry // slots)
    assert len(ends) == len(spans) == slots * OUTPUTS
    groups, last = [], -1
    for chain in sorted(range(len(spans)), key=lambda c: spans[c][0]):
        if spans[chain][0] > last:
            groups.append(set())
        groups[-1].add(ends[chain])
        last = max(last, spans[chain][1])
    return groups


def test_without_a_group_every_output_is_in_flight():
    assert _groups(_grouped(32), 2) == [{(s, c) for s in range(2)
                                         for c in range(OUTPUTS)}]


def test_a_group_of_columns_takes_every_slot_of_them():
    """Four outputs at two slots a lane: two columns, both slots, and the
    group done before the next one starts."""
    groups = _groups(_grouped(32, fma_interleave_group=4,
                              fma_interleave_cut='columns'), 2)
    assert groups == [{(s, c) for s in range(2) for c in (0, 1)},
                      {(s, c) for s in range(2) for c in (2, 3)}]


def test_a_group_of_slots_takes_every_column_of_them():
    groups = _groups(_grouped(32, fma_interleave_group=4,
                              fma_interleave_cut='slots'), 2)
    assert groups == [{(s, c) for c in range(OUTPUTS)} for s in range(2)]


def test_a_column_or_a_slot_is_the_smallest_group():
    assert _groups(_grouped(32, fma_interleave_group=1,
                            fma_interleave_cut='columns'), 2) == [
        {(s, c) for s in range(2)} for c in range(OUTPUTS)]
    assert _groups(_grouped(32, fma_interleave_group=1,
                            fma_interleave_cut='slots'), 2) == [
        {(s, c) for c in range(OUTPUTS)} for s in range(2)]


def test_auto_cuts_between_columns_below_four_slots():
    """`auto`: between columns at two slots a lane; at four, every output in
    flight -- the text but for the kernel's name and options."""
    two = _groups(_grouped(32, fma_interleave_group=4), 2)
    assert two == _groups(_grouped(32, fma_interleave_group=4,
                                   fma_interleave_cut='columns'), 2)
    assert _unnamed(_grouped(64, fma_interleave_group=4)) == _unnamed(
        _grouped(64))


def _unnamed(src):
    return [line for line in re.sub(r'kernel_kernel_[0-9a-f]+', 'k',
                                    src).splitlines()
            if '// options:' not in line]


def test_a_group_as_large_as_the_outputs_is_no_group():
    """Eight outputs a lane in a group of eight: the order of every output in
    flight, and the text but for the kernel's name and options."""
    assert _unnamed(_grouped(32, fma_interleave_group=8)) == _unnamed(
        _grouped(32))
