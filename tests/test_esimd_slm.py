# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Shared memory under ESIMD is an offset, and every access has to agree.

SLM is a separate address space on this hardware.  The ESIMD accessors for it
take a byte offset into a chunk `slm_init` reserved; a raw `T*` into a
`local_accessor` is not an address into it, and `copy_from` on one compiles
and reads global memory.  So the whole shared path had to stop being pointers
-- the arena, the windows into it, and every read and write through them.

Two things are worth testing, and neither is "does it look right".  The first
is that *no* pointer-shaped use of an SLM offset survives anywhere in the
corpus, because one that does is the silent failure the change exists to
remove.  The second is the shape of the scalar accesses: a subscript works
only because `SlmRef` converts to `T` and assigns from one, and a conversion
operator does not participate in template argument deduction -- so
`s0[i] * someSimd` would not compile even though `float x = s0[i]` does.  The
corpus produces only the two shapes the proxy covers, and that is a fact with
a shelf life, so it is checked rather than remembered.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import re
import warnings
from pathlib import Path

import pytest

from tensorforge.common.basic_types import Datatype
from tensorforge.common.context import Context, Options
from tensorforge.common.vm.vm import vm_factory
from tensorforge.generators.generator import Generator

CASES = Path(__file__).parent / 'cases'
ALL_CASES = sorted({p.stem for p in CASES.rglob('*.py')
                    if '__pycache__' not in str(p)})

#: A binding of a name to a position in SLM.
DECL = re.compile(r'tensorforge::SlmPtr<[^>]*>\s+(\w+)')


def _kernel(name, backend='esimd', **options):
    path = next(CASES.rglob(f'{name}.py'))
    spec = importlib.util.spec_from_file_location(name, path)
    case = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(case)
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter('ignore')
        gen = Generator(case.descr_list(),
                        Context(arch='pvc', backend=backend,
                                fp_type=getattr(case, 'DTYPE', Datatype.F32),
                                options=Options(**options)))
        gen.generate()
    return gen.get_kernel()


# --------------------------------------------------------------------------
# the arena
# --------------------------------------------------------------------------

def test_the_arena_is_a_reserved_chunk_and_not_an_accessor():
    src = _kernel('square_notrans')
    assert 'local_accessor' not in src, (
        'an accessor alongside `slm_init` reserves the space twice, and the '
        'accesses address the chunk')
    assert re.search(r'tensorforge::slmReserve<\d+ \* sizeof\(float\)>\(\);', src)
    assert 'tensorforge::SlmPtr<float> totalShrMem = tensorforge::SlmPtr<float>(0)' in src


def test_a_kernel_of_several_sections_reserves_once():
    """`slm_init` may be called once per kernel, and every section binds its
    own arena: `fence_two_gemms` declared it twice and IGC refused the kernel
    ("slm_init is called more than once").  The reservation is the kernel's,
    at the largest section's size; the binding stays the section's."""
    src = _kernel('fence_two_gemms')
    assert src.count('slmReserve<') == 1, src
    assert src.count('SlmPtr<float> totalShrMem = ') == 2, src


def test_the_spmd_arena_is_still_an_accessor():
    """The same kernel on the lowering where local memory does have pointers."""
    src = _kernel('square_notrans', backend='oneapi')
    assert 'sycl::local_accessor<float, 1> totalShrMem' in src
    assert 'slmArena' not in src


def test_the_size_has_to_be_passed():
    """`slm_init` takes it as a template argument, so a caller that does not
    know it cannot declare the arena -- and silently declaring a zero-sized
    one would put every access out of bounds."""
    lexic = vm_factory('pvc', 'esimd', 'float').get_lexic()
    with pytest.raises(ValueError, match='size'):
        lexic.declare_shared_memory('totalShrMem', 'float')


# --------------------------------------------------------------------------
# nothing is addressed as a pointer
# --------------------------------------------------------------------------

#: Uses that would compile against a `float*` and do not against an offset --
#: or, worse, would compile against a raw pointer into the arena and read the
#: wrong memory.
POINTERISH = (
    (r'copy_from\(\s*{}\b', 'copy_from on an SLM offset'),
    (r'copy_to\(\s*&?\s*{}\b', 'copy_to on an SLM offset'),
    (r'&\s*{}\s*\[', 'address of an SLM element'),
    (r'\*\s*\([^)]*\)\s*&\s*{}\b', 'reinterpret cast of an SLM address'),
)


@pytest.mark.slow
@pytest.mark.parametrize('name', ALL_CASES)
@pytest.mark.parametrize('preload', [False, True])
def test_no_slm_offset_is_used_as_a_pointer(name, preload):
    try:
        src = _kernel(name, preload_globals=preload)
    except Exception as exc:  # noqa: BLE001 -- cases that do not lower at all
        pytest.skip(f'{name} does not lower to ESIMD: {type(exc).__name__}')
    names = set(DECL.findall(src))
    for line in src.splitlines():
        text = line.strip()
        if text.startswith('//') or DECL.search(text):
            continue
        for shared in names:
            for pattern, what in POINTERISH:
                assert not re.search(pattern.format(shared), text), (
                    f'{what}: {text}')


def test_a_vector_access_goes_through_the_slm_instruction():
    src = _kernel('square_notrans')
    assert 'tensorforge::slmLoad<float, 16>(s0 + (' in src
    assert 'tensorforge::slmStore<float, 64>(s0 + (' in src


def test_the_unstructured_prologue_fill_does_too():
    """The preload runs where there is no body, so it never passes the
    emitter -- and wrote its destination as an assignment for that reason."""
    src = _kernel('addressing_none', preload_globals=True)
    assert re.search(r'tensorforge::slmStore<float, \d+>\(glb_m1 \+ \(', src)


# --------------------------------------------------------------------------
# the scalar accesses, whose shape the proxy has to cover
# --------------------------------------------------------------------------

_SUBSCRIPT_READ = re.compile(r'^\w[\w:<>, ]* \w+ = {}\[[^\]]*\];$')
_SUBSCRIPT_WRITE = re.compile(r'^{}\[[^\]]*\] = [\w.]+;$')


@pytest.mark.slow
@pytest.mark.parametrize('name', ALL_CASES)
def test_a_subscripted_slm_access_is_only_ever_a_whole_statement(name):
    """`SlmRef` converts to `T` and assigns from one, and that is all it can
    do: a conversion operator is invisible to template argument deduction, so
    `s0[i] * v` would not deduce even though `float x = s0[i]` converts.

    Every scalar access in the corpus is one of those two statements today.
    If a pass starts inlining one into an expression, this is where that shows
    up -- as a failure here rather than as a wall of deduction errors from a
    header nobody wants to read.
    """
    try:
        src = _kernel(name)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f'{name} does not lower to ESIMD: {type(exc).__name__}')
    names = set(DECL.findall(src))
    for line in src.splitlines():
        text = line.strip()
        if text.startswith('//') or DECL.search(text):
            continue
        for shared in names:
            if not re.search(rf'\b{shared}\s*\[', text):
                continue
            assert (re.match(_SUBSCRIPT_READ.pattern.format(shared), text)
                    or re.match(_SUBSCRIPT_WRITE.pattern.format(shared), text)), (
                f'a subscripted SLM access that is not a whole statement, so '
                f'the proxy cannot carry it: {text}')


# --------------------------------------------------------------------------
# the other backends are untouched
# --------------------------------------------------------------------------

@pytest.mark.parametrize('backend,arch', [('cuda', 'sm_80'), ('hip', 'gfx90a'),
                                          ('oneapi', 'pvc'), ('acpp', 'pvc')])
def test_a_shared_window_is_still_a_pointer_elsewhere(backend, arch):
    """The hook is a question, and four of five backends answer it as before
    -- down to the whitespace, so that no snapshot moves for a refactor."""
    lexic = vm_factory(arch, backend, 'float').get_lexic()
    assert lexic.shared_pointer_type('float') == 'float*'
    assert lexic.shared_pointer_type('float', restrict=True).startswith('float* ')
    assert lexic.shared_window_expr('totalShrMem', 256) == '&totalShrMem[256]'
    assert lexic.get_slm_load('float', 16, 'p') is None
    assert lexic.get_slm_store('float', 16, 'p', 'v') is None
