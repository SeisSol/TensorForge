# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Cache hints for the element the batch loop reaches next.

`pir.prefetch.prefetch_hints` on the smallest bodies that show each thing it
does: which pointers and which operands get a hint, where the hints stand
relative to the element guard and the barrier closing the iteration, which
flag a pointer of the element's own is followed under, and what a loop over
groups of rows names as the next element.
"""

from __future__ import annotations

from contextlib import nullcontext

from tensorforge.backend.pir import emit, verify, walk
from tensorforge.backend.pir.build import IRBuilder
from tensorforge.backend.pir.core import (BOOL, SIZE, BufferType, Effect,
                                          MemSpace, Op, Participants,
                                          Uniformity)
from tensorforge.backend.pir.prefetch import prefetch_hints
from tensorforge.backend.writer import Writer
from tensorforge.common.basic_types import Datatype
from tensorforge.common.target import Target
from harness.placement import placed

F32 = Datatype.F32

#: The operands, as the bases their accesses are recorded against.
SRC, DST, PTRS = 'm0', 'm1', 'm2'


def _binding(b, index, *, base=SRC, name='glb_m0', element_pointer=False,
             writable=False):
    """A pointer to `index`'s element, the way `GetElementPtr` binds one."""
    return b.decl_expr(
        f'const float *const __restrict__ {name}',
        f'&{base}[{{0}} * 64 + 0]' if not element_pointer
        else f'&{base}[{{0}}][0]',
        BufferType(F32, (64,), MemSpace.GLOBAL, readonly=not writable), base,
        args=(index,), kind=Effect.READ, space=MemSpace.GLOBAL, hint=name,
        extern=name, attrs=(('element_pointer', True),) if element_pointer
        else ())


class Section:
    """A section body around one batch loop, and handles to its parts."""

    def __init__(self, *, flags=True, element_pointer=False, bindings=1,
                 grouped=False, successor=True, closing_barrier=True,
                 guard=None):
        # A loop over groups of rows has a mask and no guard: the mask holds
        # the global writes alone.
        guard = flags and not grouped if guard is None else guard
        b = self.b = IRBuilder(fptype=F32, arena='shrMem')
        count = b.extern_value('numElements0', SIZE, hint='count')
        with b.for_('start', count, 'stride', hint='batchId0',
                    index_type=SIZE, uniform=Uniformity.MULT,
                    flag_word='flags0[{0}]' if flags else None) as f:
            k = self.k = f.induction
            element = k
            if grouped:
                # The row's element: the group's index and the row in it,
                # the group's own where the row is past the end.
                lane = b.extern_value('lane', SIZE, uniform=Uniformity.MULT,
                                      hint='lane')
                row = b.op('add', SIZE, k, lane, hint='row')
                active = b.op('lt', BOOL, row, count, hint='active')
                element = b.op('select', SIZE, active, row, k, hint='batchId0')
                f._element_index = element
            self.element = element
            ahead = b.op('add', SIZE, element, 'stride', hint='ahead1')
            fits = b.op('lt', BOOL, ahead, count, hint='inbatch1')
            self.next = b.op('select', SIZE, fits, ahead, element,
                             hint='batchId1', escapes=True)
            if successor:
                f._next_index = self.next
            out = _binding(b, element, base=DST, name='glb_m1', writable=True)
            if guard:
                allowed = b.decl_expr(
                    'const bool allowed', 'static_cast<bool>(flags0[{0}])',
                    BOOL, None, args=(element,), hint='allowed',
                    extern='allowed')
                guard = b.if_(allowed, attrs=(('guard', 'element'),))
            else:
                guard = nullcontext()
            with guard:
                for j in range(bindings):
                    name = f'glb_m0_{j}' if j else 'glb_m0'
                    src = _binding(b, element,
                                   base=PTRS if element_pointer else SRC,
                                   name=name, element_pointer=element_pointer)
                    dest = b.alloc(F32, (8,), MemSpace.REGISTER, hint='r',
                                   extern=f'r{j}')
                    b.Comment(f'r{j} = load{{g>r}}({name})')
                    with b.for_(0, 8, 1, unroll=True) as i:
                        b.store(dest, b.load(src, i.induction, hint='g'),
                                i.induction)
                    b.store(out, b.load(dest, 3, hint='u'), j)
            if closing_barrier:
                b.barrier(Participants.MULT, threads=32)
        self.body = b.finish()
        verify(self.body)

    def hints(self, **kw):
        self.report = []
        out = prefetch_hints(self.body, self.b.scratch, report=self.report,
                             **kw)
        verify(out)
        return out


def _loop(body):
    return next(s for s in body if s.op is Op.FOR and s.attr('next'))


def _hints(stmts):
    return [s for s, _ in walk(tuple(stmts)) if s.op is Op.PREFETCH]


def _defs(body):
    return {t.id: s for s, _ in walk(body) for t in s.target}


def _guard_at(region):
    return next(i for i, s in enumerate(region)
                if s.op is Op.IF and s.attr('guard') == 'element')


# --------------------------------------------------------------------------- #
# when nothing changes
# --------------------------------------------------------------------------- #

def test_with_neither_kind_asked_for_the_body_is_returned_as_it_is():
    sec = Section(element_pointer=True)
    assert sec.hints() is sec.body


def test_a_loop_that_names_no_successor_is_left_alone():
    sec = Section(element_pointer=True, successor=False)
    assert sec.hints(pointers=True, data=True) is sec.body
    assert sec.report == []


def test_strided_addressing_has_no_pointer_to_ask_for():
    """No pointer array, so no dependent load for a hint to shadow; the loop
    stays as it is, and the report says why."""
    sec = Section()
    assert sec.hints(pointers=True) is sec.body
    assert sec.report == ['- loop: nothing to hint']


# --------------------------------------------------------------------------- #
# the pointers
# --------------------------------------------------------------------------- #

def test_the_entry_of_a_pointer_array_is_asked_for_once_ahead_of_the_guard():
    """Two bindings read their element's pointer out of one array: one hint,
    for the successor's entry, outside the guard -- a masked element still
    has a successor."""
    sec = Section(element_pointer=True, bindings=2)
    region = _loop(sec.hints(pointers=True)).regions[0].body
    hints = _hints(region)
    assert len(hints) == 1
    assert hints[0].args == (PTRS, sec.next)
    assert region.index(hints[0]) < _guard_at(region)
    assert sec.report == ['+ glb_m0 [pointer]']


# --------------------------------------------------------------------------- #
# the operands
# --------------------------------------------------------------------------- #

def test_the_data_is_asked_for_a_line_at_a_time_behind_the_guard():
    """64 floats a line of 128 bytes apart: two hints, through a pointer to
    the successor bound at the head.  Behind the guard, and ahead of the
    barrier that closes the iteration."""
    sec = Section()
    out = sec.hints(data=True, line_bytes=128)
    region = _loop(out).regions[0].body
    hints = _hints(region)
    assert [(h.args[1], h.attr('elems')) for h in hints] == [(0, 32), (32, 32)]
    at = [region.index(h) for h in hints]
    barrier = max(i for i, s in enumerate(region) if s.op is Op.BARRIER)
    assert all(_guard_at(region) < i < barrier for i in at)
    pointer = _defs(out)[hints[0].args[0].id]
    assert pointer.attr('extern') == 'pf_glb_m0'
    assert pointer.args == (sec.next,)
    assert region.index(pointer) < _guard_at(region)
    assert sec.report == ['+ glb_m0 [data]']


def test_a_pointer_of_the_elements_own_is_followed_under_its_flag():
    """The successor's entry is in range, the loop clamps its index; the
    pointer in it is the caller's to promise, for an element it did not
    mask.  So the hints run under the successor's flag, read the way the
    loop reads one, by a name of their own."""
    sec = Section(element_pointer=True)
    out = sec.hints(data=True)
    region = _loop(out).regions[0].body
    guards = [s for s in region[_guard_at(region) + 1:] if s.op is Op.IF]
    assert len(guards) == 1 and _hints(guards[0].regions[0].body)
    defs = _defs(out)
    flag = defs[guards[0].args[0].id]
    assert flag.attr('extern') == 'allowed_hint'
    word = defs[flag.args[0].id]
    assert word.attr('extern') == 'flagWordHint'
    assert word.args == (sec.next,)


def test_without_a_mask_the_pointer_is_followed_unguarded():
    sec = Section(element_pointer=True, flags=False)
    region = _loop(sec.hints(data=True)).regions[0].body
    assert _hints(region)
    assert not any(s.op is Op.IF for s in region)


def test_a_group_of_rows_follows_the_successors_pointer_under_its_flag():
    """No guard, the mask folded into the writes: the hints still run
    under the successor's flag."""
    sec = Section(grouped=True, element_pointer=True)
    region = _loop(sec.hints(data=True)).regions[0].body
    guards = [s for s in region if s.op is Op.IF]
    assert len(guards) == 1 and _hints(guards[0].regions[0].body)
    assert not _hints(s for s in region if s.op is not Op.IF)


def test_a_group_of_rows_asks_for_the_rows_successor():
    """Where the body binds the element -- a row of a group, offset from the
    group's index -- the hints are for that element's successor, and what
    binds the element is not computed again: for the successor, it would
    offset the group's index by the row a second time."""
    sec = Section(grouped=True)
    out = sec.hints(data=True)
    defs = _defs(out)
    pointer = defs[_hints(_loop(out).regions[0].body)[0].args[0].id]
    assert pointer.args == (sec.next,)
    rows = [s for s, _ in walk(out) if s.op == 'select'
            and s.target[0].hint == 'batchId0']
    assert len(rows) == 1


def test_the_hints_render():
    w = Writer()
    emit(placed(Section().hints(data=True)), w, Target('sm_86', 'cuda'))
    src = w.get_src()
    assert 'tensorforge::prefetchL2(&pf_glb_m0[0]);' in src, src
    assert 'tensorforge::prefetchL2(&pf_glb_m0[32]);' in src, src
