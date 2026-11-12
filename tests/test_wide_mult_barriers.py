# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""A multiplication wider than any sub-group meets at the block barrier
around what its sub-groups exchange through shared memory.

64 lanes on PVC are two 32-wide sub-groups, and nothing narrower than the
block meets both (`Target.sync_mult`), so the block holds that one
multiplication and its barrier is the block's.  The batch traversal then
runs alike for the whole block (`BatchLoop.uniform_scope`), and so do the
size and flag guards on the element index -- which the index has to say
(`BatchLoop._index_uniformity`): stated as the multiplication's own, the
guarded body was a region `pir.barriers` may put no block barrier into, it
reported the accesses it could not fence, and SeisSol's order-5 F64
`neighboringFlux` and `volume` raced on the Max 1550 (the lanes rule's
picks there).  Run on the CPU (tfemu), the unfenced kernel reads the other
sub-group's rows before they are written.
"""

from __future__ import annotations

import re

import pytest

from tensorforge.common.basic_types import Datatype
from tensorforge.common.context import Context
from tensorforge.common.options import Options


def _source(kernel: str, lanes: int) -> str:
    import seissol_suite as fx
    from tensorforge.frontend.yateto import DescriptionReader
    from tensorforge.generators.generator import Generator
    descrs = DescriptionReader(None, {}).read(fx.description(
        'elastic-linearck', 'elastic-linearck-o5-d', kernel))[0]
    ctx = Context(arch='pvc', backend='oneapi', fp_type=Datatype.F64,
                  options=Options(lanes_per_mult=lanes))
    gen = Generator(descrs, ctx)
    gen.generate()
    return gen.get_kernel()


def _staged_then_read(src: str):
    """For every staging into shared memory (`// sK = store{r>s}`), the text
    between it and the first statement that reads the buffer back -- an
    element of it on the right of an assignment."""
    out = []
    for m in re.finditer(r'// (s\d+) = store\{r>s\}', src):
        read = re.compile(rf'= [^;]*\b{m.group(1)}\[')
        hit = read.search(src, m.end())
        if hit is not None:
            out.append((m.group(1), src[m.end():hit.start()]))
    return out


@pytest.mark.parametrize('kernel', ['gpu_neighboringFlux[0]', 'gpu_volume'])
def test_sixty_four_lanes_meet_between_a_staging_and_its_reads(kernel):
    src = _source(kernel, 64)
    assert 'reqd_sub_group_size(32)' in src
    pairs = _staged_then_read(src)
    assert pairs
    for name, between in pairs:
        assert 'item.barrier();' in between, name


def test_the_index_is_the_blocks_at_one_multiplication_a_block():
    from tensorforge.backend.instructions.batch_loop import BatchLoop
    from tensorforge.backend.pir.core import Uniformity

    class One(BatchLoop):
        def __init__(self, mults, grouped=False):
            self._mults_per_block = mults
            self._group_size = 1
            self._grouped_flag = grouped

        def _grouped(self):
            return self._grouped_flag

    assert One(1)._index_uniformity() is Uniformity.BLOCK
    assert One(4)._index_uniformity() is Uniformity.MULT


def test_an_unfenced_exchange_is_refused_rather_than_emitted(monkeypatch):
    """Where `pir.barriers` meets shared accesses it cannot fence, the
    section is not emitted: the old, multiplication-wide index makes the
    guarded body a region no block barrier may enter, and the kernel it
    would have produced raced."""
    from tensorforge.backend.instructions.batch_loop import BatchLoop
    from tensorforge.backend.pir.core import Uniformity
    from tensorforge.common.exceptions import GenerationError
    monkeypatch.setattr(BatchLoop, '_index_uniformity',
                        lambda self: Uniformity.MULT)
    with pytest.raises(GenerationError, match='fences'):
        _source('gpu_neighboringFlux[0]', 64)
