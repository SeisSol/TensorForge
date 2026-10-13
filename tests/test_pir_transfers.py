# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""What a transfer is, as the passes that move one find it.

A loader tags the statements of its transfer (`IRBuilder.transfer`), and
`pir.transfers.transfers` tells one transfer from the next by the tag.  The
tag is no region: what the passes between the builder and `wrap` do across
statements -- `flatten`, `cse`, `licm` -- they do across a transfer's as well,
and the tests here are mostly about that holding.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from tensorforge.backend import pir
from tensorforge.backend.pir import walk
from tensorforge.backend.pir.build import IRBuilder
from tensorforge.backend.pir.core import (INDEX, SIZE, BufferType, Effect,
                                          IRError, MemSpace, Op, Uniformity)
from tensorforge.backend.pir.passes import cse, flatten_scopes
from tensorforge.backend.pir.transfers import tag, transfers
from tensorforge.common.basic_types import Datatype
from tensorforge.common.context import Context, Options
from tensorforge.generators.generator import Generator

F32 = Datatype.F32
CASES = Path(__file__).parent / "cases"


def _binding(b, k, base):
    return b.decl_expr(
        f'const float *const __restrict__ glb_{base}', f'&{base}[{{0}} * 64]',
        BufferType(F32, (64,), MemSpace.GLOBAL, readonly=True), base,
        args=(k,), kind=Effect.READ, space=MemSpace.GLOBAL, hint=f'glb_{base}',
        extern=f'glb_{base}')


def _fill(b, src, dest, at=0):
    """Eight elements of `src` into `dest` from `at` on, the way a loader
    writes a transfer into registers."""
    b.Comment('load{g>r}')
    with b.for_(0, 8, 1, unroll=True) as i:
        to = b.op('add', INDEX, i.induction, at, hint='to')
        b.store(dest, b.load(src, i.induction, hint='g'), to)


def test_the_statements_a_transfer_adds_carry_its_tag():
    """The statements of the list the transfer was opened in, and not what
    stands in their regions: a list is what a pass finds transfers in."""
    b = IRBuilder(fptype=F32)
    b.Comment('before')
    with b.transfer() as mine:
        b.Comment('a transfer')
        with b.for_(0, 4, 1, unroll=True):
            b.Comment('inside its loop')
    b.Comment('after')
    body = b.finish()
    assert [tag(s) for s in body] == [None, mine, mine, None]
    assert all(tag(s) is None for s in body[2].regions[0].body)


def test_two_transfers_get_two_tags():
    b = IRBuilder(fptype=F32)
    with b.transfer() as one:
        pass
    with b.transfer() as other:
        pass
    assert one != other


def test_a_transfer_inside_a_transfer_is_refused():
    b = IRBuilder(fptype=F32)
    with b.transfer():
        with pytest.raises(IRError, match='inside a transfer'):
            with b.transfer():
                pass


def test_two_transfers_into_one_buffer_in_a_row_stay_two():
    """Read off the accesses alone, two fills of one buffer in a row are one
    transfer; written as two, they are two."""
    def body(tagged):
        b = IRBuilder(fptype=F32)
        k = b.extern_value('element', SIZE, uniform=Uniformity.MULT, hint='k')
        m0, m1 = _binding(b, k, 'm0'), _binding(b, k, 'm1')
        dest = b.alloc(F32, (16,), MemSpace.REGISTER, hint='r', extern='r0')
        for src, at in ((m0, 0), (m1, 8)):
            if tagged:
                with b.transfer():
                    _fill(b, src, dest, at)
            else:
                _fill(b, src, dest, at)
        return b.finish()
    assert len(transfers(body(tagged=False))) == 1
    two = transfers(body(tagged=True))
    assert len(two) == 2 and two[0].tag != two[1].tag
    assert all(t.tag == tag(t.pieces[0]) for t in two)


def test_flatten_hands_a_scope_its_tag_to_what_it_held():
    """A scope a transfer opened goes as any anonymous scope does, and what
    it held is the transfer's from then on."""
    b = IRBuilder(fptype=F32)
    with b.transfer() as mine:
        with b.Scope():
            b.Comment('scoped')
            b.Comment('as well')
    flat = flatten_scopes(b.finish())
    assert [s.text for s in flat] == ['// scoped', '// as well']
    assert all(tag(s) == mine for s in flat)


def test_what_two_transfers_compute_alike_is_computed_once():
    """Each loader computes its lane from the thread index; the tags tell
    the transfers apart, and the lane is one value all the same."""
    def lane(b):
        with b.transfer():
            return b.op('rem', INDEX, b.thread_id('x'), 32, hint='lead')
    b = IRBuilder(fptype=F32)
    lane(b)
    other = b.scratch()
    lane(other)
    body = b.finish() + other.finish()
    assert len([s for s in body if s.op == 'rem']) == 2
    assert len([s for s in cse(body) if s.op == 'rem']) == 1


def test_a_loader_writes_its_transfer_under_a_tag(monkeypatch):
    """Both loaders out of global memory tag what they write, one tag each;
    the wait for a copy is no part of the copy."""
    bodies = []
    real = pir.optimize

    def spy(body, **kwargs):
        bodies.append(body)
        return real(body, **kwargs)
    monkeypatch.setattr(pir, 'optimize', spy)
    path = CASES / 'chain_five.py'
    spec = importlib.util.spec_from_file_location('tf_tags__' + path.stem,
                                                  path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    Generator(mod.descr_list(),
              Context(arch='sm_86', backend='cuda', options=Options(),
                      fp_type=getattr(mod, 'DTYPE', None))).generate()
    comments = [s for body in bodies for s, _ in walk(body)
                if s.op == Op.RAWSTMT and (s.text or '').startswith('// ')]
    waits = [s for s in comments if s.text.startswith('// wait(')]
    loads = [s for s in comments if 'load{g>' in s.text
             and not s.text.startswith('// wait(')]
    assert {'load{g>r}', 'load{g>s}'} <= {
        part for s in loads for part in ('load{g>r}', 'load{g>s}')
        if part in s.text}
    assert all(tag(s) is not None for s in loads)
    assert len({tag(s) for s in loads}) == len(loads)
    assert waits and all(tag(s) is None for s in waits)
