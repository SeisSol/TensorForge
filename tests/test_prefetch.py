# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""The data prefetch: a statement whose whole content is where it sits.

`Op.PREFETCH` has no result, so three of the mechanisms the IR runs on had an
answer for it before it existed, and two of those answers were wrong. Dead
code elimination deletes statements that produce nothing; the emitter prints
what a lexic hands it and nothing where it hands back None; and the reordering
machinery decides from effects and accesses whether a statement may move.

What these pin is that the three agree on what a hint *is*: something that may
move, may not be deleted, and may be missing from a target's output without
the body meaning anything different. The spelling per backend is checked too,
but it is the smaller half -- an instruction that reaches no hardware is a
missed optimisation, while a hint that survives DCE only on the targets that
can spell it is a body that optimises differently depending on who reads it.
"""

from __future__ import annotations

import pytest

from tensorforge.backend.pir import emit, optimize, verify, walk
from tensorforge.backend.pir.build import IRBuilder
from tensorforge.backend.pir.core import Datatype, MemSpace, Op
from tensorforge.backend.pir.passes import dce
from tensorforge.backend.pir.schedule import can_reorder
from tensorforge.backend.writer import Writer
from tensorforge.common.vm.lexic.cuda_lexic import CudaLexic
from tensorforge.common.vm.lexic.hip_lexic import HipLexic
from tensorforge.common.vm.lexic.ocl_lexic import OpenCLLexic
from tensorforge.common.vm.lexic.sycl_lexic import SyclLexic
from tensorforge.common.vm.vm import vm_factory


class _Hw:
    """Just the two members the availability question reads.

    A real `HwDecription` comes out of the hardware table, which has a row
    only for the parts the generator is built for -- and the interesting cases
    here are the ones on the far side of a threshold, which is exactly where
    that table has no row.
    """

    def __init__(self, model):
        self.model = model

    def sm_level(self):
        text = str(self.model)
        if not text.startswith('sm_'):
            return None
        digits = ''.join(c for c in text[3:] if c.isdigit())
        return int(digits) if digits else None


def _body(level='l2', elems=1):
    b = IRBuilder(fptype=Datatype.F32)
    g = b.alloc(Datatype.F32, (256,), MemSpace.GLOBAL, hint='g')
    b.prefetch(g, 64, level=level, elems=elems)
    return b.finish()


def _emit(body, arch='sm_86', backend='cuda'):
    w = Writer()
    emit(body, w, vm_factory(arch, backend, 'float'))
    return w.get_src()


# --------------------------------------------------------------------------- #
# What a hint is
# --------------------------------------------------------------------------- #

def test_a_hint_survives_dce():
    """The rule it would otherwise fall to is "produces nothing, does nothing".

    Half right, and that is the trap: a prefetch does produce nothing, and
    deleting it never changes a result. It would have been deleted from every
    body, on every target, and the only symptom would have been that the op
    bought nothing.
    """
    body = _body()
    assert [s.op for s, _ in walk(dce(body))] == [Op.ALLOC, Op.PREFETCH]


def test_a_hint_may_move_but_not_past_a_write_to_what_it_names():
    """Movable, because a hint issued late is a hint wasted.

    Not past a store to the same buffer, though. That is not about
    correctness -- nothing reads what a prefetch brings in -- it is about the
    hint staying a hint for the line the load will want, rather than for one
    the store is on its way to dirtying.
    """
    b = IRBuilder(fptype=Datatype.F32)
    g = b.alloc(Datatype.F32, (256,), MemSpace.GLOBAL, hint='g')
    other = b.alloc(Datatype.F32, (256,), MemSpace.GLOBAL, hint='o')
    pf = b.prefetch(g, 64)
    to_other = b.store(other, 1.0, 0)
    to_same = b.store(g, 1.0, 0)

    assert can_reorder(to_other, pf), (
        'a store to a different buffer blocked the hint; distinct bases do '
        'not alias, and a hint that cannot cross one cannot be hoisted at all')
    assert not can_reorder(to_same, pf)


def test_a_hint_verifies_and_needs_no_wait():
    """No token, so nothing pairs with it and `verify` must not look for one."""
    assert verify(_body()) == []


def test_the_optimiser_leaves_one_hint_standing():
    """End to end through the default pipeline, which is where DCE runs."""
    body = optimize(_body())
    assert sum(1 for s, _ in walk(body) if s.op == Op.PREFETCH) == 1


# --------------------------------------------------------------------------- #
# The gate
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('model', ['sm_50', 'sm_80', 'sm_90', 'sm_120'])
def test_cuda_has_a_prefetch_from_sm_50(model):
    assert CudaLexic('cuda', 'nvidia').has_prefetch(_Hw(model))


def test_cuda_declines_below_the_ptx_instruction():
    """The inline PTX carries no guard, so the refusal has to happen here."""
    assert not CudaLexic('cuda', 'nvidia').has_prefetch(_Hw('sm_35'))


@pytest.mark.parametrize('model', ['gfx1200', 'gfx1250', 'gfx1251'])
def test_amd_has_a_prefetch_from_gfx12(model):
    assert HipLexic('hip', 'amd').has_prefetch(_Hw(model))


@pytest.mark.parametrize('model', ['gfx900', 'gfx90a', 'gfx942', 'gfx950',
                                   'gfx1030', 'gfx1100'])
def test_amd_has_none_below_gfx12(model):
    """`__builtin_prefetch` compiles here and selects nothing.

    Which is why this is asked at all: the failure it prevents is not a build
    error but a statement the IR goes on carrying to no effect, and a body
    that cannot be hinted saying nothing about it.
    """
    assert not HipLexic('hip', 'amd').has_prefetch(_Hw(model))


def test_the_gfx_number_is_read_as_hexadecimal():
    """gfx90a has a letter in it, and gfx940 is below gfx1030, not above."""
    lex = HipLexic('hip', 'amd')
    assert not lex.has_prefetch(_Hw('gfx90a'))
    assert not lex.has_prefetch(_Hw('gfx940'))


def test_hip_on_nvidia_declines():
    """The same condition `glb_store` and `atomic_store` already carry."""
    assert not HipLexic('hip', 'nvidia').has_prefetch(_Hw('sm_80'))


def test_sycl_answers_from_the_library_and_not_the_part():
    lex = SyclLexic('oneapi', 'intel')
    assert lex.has_prefetch(_Hw('pvc'))
    assert SyclLexic('acpp', 'nvidia').has_prefetch(_Hw('sm_80'))


def test_esimd_has_one_where_the_lsc_does():
    """`prefetch(const T*, props)` is the block form: one address, no mask.

    The gather forms take a vector of byte offsets and are what a scattered
    access wants; this hook hands out a single address, and the API has an
    overload for exactly that.
    """
    lex = SyclLexic('oneapi', 'intel', explicit_simd=True)
    assert lex.has_prefetch(_Hw('pvc'))
    assert lex.prefetch('&g[0]', datatype=Datatype.F32) == (
        'tensorforge::prefetchL2(&g[0]);')


def test_esimd_declines_where_the_lsc_does_not():
    """DG2 and PVC, per the API's own documentation; dg1 is Xe-LP."""
    lex = SyclLexic('oneapi', 'intel', explicit_simd=True)
    assert not lex.has_prefetch(_Hw('dg1'))


def test_the_spmd_answer_does_not_depend_on_the_part():
    """A core SYCL call an implementation has to accept, instruction or not."""
    lex = SyclLexic('oneapi', 'intel')
    assert lex.has_prefetch(_Hw('dg1')) and lex.has_prefetch(_Hw('pvc'))


# --------------------------------------------------------------------------- #
# The spelling
# --------------------------------------------------------------------------- #

def test_cuda_spells_the_level():
    lex = CudaLexic('cuda', 'nvidia')
    assert lex.prefetch('&g[i]', datatype=Datatype.F32,
                        level='l1') == 'tensorforge::prefetchL1(&g[i]);'
    assert lex.prefetch('&g[i]', datatype=Datatype.F32,
                        level='l2') == 'tensorforge::prefetchL2(&g[i]);'


@pytest.mark.parametrize('level', ['l1', 'l2'])
def test_amd_has_one_instruction_for_both_levels(level):
    """The locality argument is a scope on this target, not a cache level.

    Stated as a test because the level is accepted here and dropped, which is
    the kind of silent non-honouring worth pinning: a caller that starts
    depending on the distinction should find out from this, not from a
    profile.
    """
    lex = HipLexic('hip', 'amd')
    assert lex.prefetch('&g[i]', datatype=Datatype.F32,
                        level=level) == '__builtin_prefetch(&g[i], 0, 3);'


def test_sycl_spells_the_count_and_not_the_level():
    lex = SyclLexic('oneapi', 'intel')
    text = lex.prefetch('&g[i]', datatype=Datatype.F32, elems=4, level='l1')
    assert text.endswith('.prefetch(4);')
    assert 'address_space_cast' in text, (
        'a multi_ptr built from a raw pointer is the deprecated legacy '
        'spelling')


def test_opencl_passes_the_extent_through():
    """The one builtin so far whose instruction takes a count.

    Unbound, because `OpenCLLexic` cannot be constructed: six of `Lexic`'s
    abstract methods have no implementation there, so the backend is a stub
    and instantiating it raises. That is worth knowing rather than working
    around -- the spelling is checked here so that it is right on the day the
    rest of the class arrives, and this line is where a reader learns that day
    has not come.
    """
    assert OpenCLLexic.prefetch(None, '&g[i]', datatype=Datatype.F32,
                                elems=8) == 'prefetch(&g[i], 8);'


# --------------------------------------------------------------------------- #
# Emission
# --------------------------------------------------------------------------- #

def test_the_hint_reaches_the_generated_source():
    src = _emit(_body())
    assert 'tensorforge::prefetchL2(&v0_g[64]);' in src, src


def test_a_target_without_one_drops_it_and_says_so():
    """Dropping is safe; dropping quietly is not.

    Without the note, a body on gfx1030 and a body that was never hinted read
    the same, and only one of the two is worth doing something about.
    """
    src = _emit(_body(), arch='gfx1030', backend='hip')
    assert '__builtin_prefetch' not in src
    assert 'prefetch hints dropped' in src, src
    assert 'gfx1030' in src


def test_a_body_with_no_hints_says_nothing_about_prefetching():
    """The note is per body and conditional on there being something to note."""
    b = IRBuilder(fptype=Datatype.F32)
    g = b.alloc(Datatype.F32, (256,), MemSpace.GLOBAL, hint='g')
    b.store(g, 1.0, 0)
    assert 'prefetch' not in _emit(b.finish(), arch='gfx1030', backend='hip')


def test_the_address_is_an_operand_and_not_text():
    """The index goes through the same address arithmetic a load does.

    Which is the reason this is an op rather than a `rawstmt` carrying a
    pointer expression: the alias analysis, the swizzle and the liveness all
    read the base and the indices, and none of them can read a string.
    """
    b = IRBuilder(fptype=Datatype.F32)
    g = b.alloc(Datatype.F32, (16, 32), MemSpace.GLOBAL, hint='g')
    b.prefetch(g, 3, 5)
    stmt = next(s for s, _ in walk(b.finish()) if s.op == Op.PREFETCH)
    assert stmt.prefetch_index == (3, 5)
    assert '3 + 16 * (5)' in _emit(b.finish())
