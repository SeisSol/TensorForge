# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""The pointer hint: which operands get one, and what it costs when they do.

`PrefetchBatch` is the first thing that builds an `Op.PREFETCH`, so what it
has to answer for is narrower than the op's own tests: not what a hint means,
but which address is worth one and which loops can name it.

The address is the pointer array under `Addressing.PTR_BASED`, because that is
the one place a batched kernel pays a dependent load -- `m[batchId0]` has to
arrive before any address in the iteration exists. Strided addressing pays no
such load, and a hint for it would need the element-offset formula written a
second time, which is why the pass leaves it alone rather than covering it
approximately.

The other half is that turning the switch off leaves the output exactly as it
was. A pass that is off has to be invisible, and this one inserts into the
head of a region several other passes also rewrite.
"""

from __future__ import annotations

import pytest

from tensorforge.backend.instructions.batch_loop import BatchLoop, LoopMode
from tensorforge.backend.instructions.prefetch import PrefetchBatchPointer
from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.context import Context
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.common.options import Options
from tensorforge.generators.descriptions import GemmDescr
from tensorforge.generators.generator import Generator


def _gemm(addressing):
    def operand(alias):
        return SubTensor(Tensor([16, 16], addressing,
                                BoundingBox([0, 0], [16, 16]),
                                alias=alias, datatype=Datatype.F32))
    return [GemmDescr(False, False, operand('A'), operand('B'), operand('C'),
                      alpha=1.0, beta=0.0)]


def _generate(addressing, arch='sm_86', backend='cuda', **opts):
    ctx = Context(arch=arch, backend=backend, fp_type=Datatype.F32,
                  options=Options(**opts) if opts else None)
    gen = Generator(_gemm(addressing), ctx)
    gen.generate()
    return gen, gen.get_kernel()


def _kernel(addressing, arch='sm_86', backend='cuda', **opts):
    return _generate(addressing, arch=arch, backend=backend, **opts)[1]


def _needs_a_batch_loop(gen):
    """Skip where the traversal gives the body no next element to ask for.

    `prefer_persistent` in `generator.py` decides between a grid-stride loop
    and one element per work-item, and it is a vendor list under revision. So
    the generator is asked rather than the answer assumed: a target that grows
    a batch loop turns these green without anything here changing.
    """
    if gen._batch_loop_mode() is LoopMode.SINGLE:
        pytest.skip('one element per work-item on this target, so there is no '
                    'next element to hint for')


# --------------------------------------------------------------------------- #
# The option
# --------------------------------------------------------------------------- #

def test_the_switch_is_off_by_default():
    """Off pending numbers, like every other switch that trades bandwidth."""
    assert Options().resolve(
        Context(arch='sm_86', backend='cuda',
                fp_type=Datatype.F32)._vm.get_hw_descr()).enable_prefetch is False


def test_off_leaves_the_kernel_byte_identical():
    """The property that lets this be reviewed as an addition and nothing else.

    It inserts at the head of a region that `MoveLoads`, `WrapLoads` and
    `Pipeline` also rewrite, so "does nothing when disabled" is a claim about
    ordering as much as about the guard on the pass.
    """
    assert (_kernel(Addressing.PTR_BASED)
            == _kernel(Addressing.PTR_BASED, enable_prefetch=False))


def test_the_level_reaches_the_instruction():
    l1 = _kernel(Addressing.PTR_BASED, enable_prefetch=True,
                 prefetch_level='l1')
    l2 = _kernel(Addressing.PTR_BASED, enable_prefetch=True,
                 prefetch_level='l2')
    assert 'prefetchL1' in l1 and 'prefetchL2' not in l1
    assert 'prefetchL2' in l2 and 'prefetchL1' not in l2


# --------------------------------------------------------------------------- #
# Which operands
# --------------------------------------------------------------------------- #

def test_a_pointer_based_operand_is_hinted():
    src = _kernel(Addressing.PTR_BASED, enable_prefetch=True)
    assert 'tensorforge::prefetchL2(&' in src, src


def test_the_hint_names_the_next_element_and_not_this_one():
    """`batchId1`, the clamped lookahead -- `batchId0` would be the line the
    iteration is about to read anyway, which is a request for nothing."""
    src = _kernel(Addressing.PTR_BASED, enable_prefetch=True)
    hints = [ln for ln in src.splitlines() if 'prefetchL2' in ln]
    assert hints, src
    for line in hints:
        assert 'batchId1' in line, line


def test_strided_addressing_is_left_alone():
    """No dependent load to shadow, and the address formula lives elsewhere.

    Matched on the call and not on the word: the resolved options are written
    into the kernel as a comment, so `enable_prefetch=1` is in the text of
    every kernel this switch is on for.
    """
    src = _kernel(Addressing.STRIDED, enable_prefetch=True)
    assert 'tensorforge::prefetch' not in src


def test_each_operand_is_hinted_once():
    """Three operands, three hints -- not one per binding that reads them."""
    src = _kernel(Addressing.PTR_BASED, enable_prefetch=True)
    assert sum(1 for ln in src.splitlines() if 'prefetchL2' in ln) == 3, src


# --------------------------------------------------------------------------- #
# Which loops
# --------------------------------------------------------------------------- #

def _loops(instrs):
    for instr in instrs:
        if isinstance(instr, BatchLoop):
            yield instr
        for region in instr.regions():
            yield from _loops(region)


def _hints_in(loop):
    return [i for i in loop.region if isinstance(i, PrefetchBatchPointer)]


def test_a_target_without_a_prefetch_drops_it_and_says_so():
    """The pass still runs: what a target can spell is the emitter's question.

    Worth having as its own test, because the alternative design -- refusing
    to insert on such a target -- would put the same fact in two places and
    make the IR depend on the vendor.
    """
    src = _kernel(Addressing.PTR_BASED, arch='gfx90a', backend='hip',
                  enable_prefetch=True)
    assert '__builtin_prefetch' not in src
    assert 'prefetch hints dropped' in src, src


def test_gfx12_gets_the_builtin():
    src = _kernel(Addressing.PTR_BASED, arch='gfx1200', backend='hip',
                  enable_prefetch=True)
    assert '__builtin_prefetch(&' in src, src


def test_sycl_spells_the_core_2020_form():
    gen, src = _generate(Addressing.PTR_BASED, arch='pvc', backend='acpp',
                         enable_prefetch=True)
    _needs_a_batch_loop(gen)
    assert 'address_space_cast' in src and '.prefetch(1);' in src, src


def test_esimd_spells_it_through_the_helper():
    """The cache hints are mandatory on this path, so a helper carries them.

    `check_cache_hints` refuses a prefetch whose property list is empty, and
    refuses several of the combinations that are not, so the legal pairs live
    in `isycl.h` and the generated line names one of two helpers.
    """
    gen, src = _generate(Addressing.PTR_BASED, arch='pvc', backend='esimd',
                         enable_prefetch=True)
    _needs_a_batch_loop(gen)
    # one hint alone, or several side by side as one gather
    assert ('tensorforge::prefetchL2(&' in src
            or 'tensorforge::prefetchRunsL2<' in src), src


def test_the_hint_sits_at_the_head_of_the_body():
    """Ahead of the first binding, so the whole iteration is cover.

    And outside the flag guard, ahead of it.  It used to sit inside, which was
    pinned here as a limitation to be lifted by decision -- a masked element
    issued no hint.  The decision came with `enable_wrap_loads`: the wrapped
    transfer's pointer is an unguarded prefix of the body, a hint inside the
    guard ahead of it made the prefix non-contiguous, and the combination did
    not generate.  Outside is safe: the address is a clamped index into the
    pointer array, and nothing dereferences what it asks for.
    """
    src = _kernel(Addressing.PTR_BASED, enable_prefetch=True).splitlines()
    first_hint = next(n for n, ln in enumerate(src)
                      if 'tensorforge::prefetch' in ln)
    first_binding = next(n for n, ln in enumerate(src) if 'glb_m' in ln)
    guard = next(n for n, ln in enumerate(src) if 'if (allowed)' in ln)
    assert first_hint < guard and first_hint < first_binding
