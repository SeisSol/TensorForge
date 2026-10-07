# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""The transfer for the next element, issued across the batch loop's back edge.

`pir.wrap.wrap_loads` on the smallest bodies that show each thing it does:
where a moved transfer lands, what it reads for the element it is for, what a
shared transfer's loop carries, the flags a masked traversal reads -- and,
mostly, the stretches it may not cross.  A transformation that changes what a
loop reads and when is only as good as the cases where it declines, and
"nothing moved" reads the same whether the pass was right or merely broken.
"""

from __future__ import annotations

import re

import pytest

from tensorforge.backend.pir import emit, verify, walk
from tensorforge.backend.pir.asyncmem import schedule_async
from tensorforge.backend.pir.build import IRBuilder
from tensorforge.backend.pir.core import (BOOL, SIZE, BufferType, Effect,
                                          MemSpace, Op, Participants,
                                          Uniformity)
from tensorforge.backend.pir.wrap import wrap_loads
from tensorforge.backend.writer import Writer
from tensorforge.common.basic_types import Datatype
from tensorforge.common.vm.vm import vm_factory
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


def _register_transfer(b, src, dest, name, n=8):
    # The comment the macro layer writes, in its names.
    b.Comment(f'{name} = load{{g>r}}(glb_m0)')
    with b.for_(0, n, 1, unroll=True) as i:
        b.store(dest, b.load(src, i.induction, hint='g'), i.induction)


class Section:
    """A section body around one batch loop, and handles to its parts."""

    def __init__(self, *, flags=True, element_pointer=False, transfers=1,
                 shared=False, ahead=None, init='', second_writer=False,
                 closing_barrier=True, first=True, reread=False,
                 read_after=False):
        b = self.b = IRBuilder(fptype=F32, arena='shrMem')
        count = b.extern_value('numElements0', SIZE, hint='count')
        start = b.extern_value('start', SIZE, uniform=Uniformity.MULT,
                               hint='start')
        inside = b.op('lt', BOOL, start, count, hint='inside')
        self.first = b.op('select', SIZE, inside, start, 0, hint='batchId1',
                          escapes=True)
        self.window = (b.alloc(F32, (64,), MemSpace.SHARED, hint='s',
                               extern='s0') if shared else None)
        with b.for_('start', count, 'stride', hint='batchId0',
                    index_type=SIZE, peel_index=self.first if first else None,
                    uniform=Uniformity.MULT,
                    flag_word='flags0[{0}]' if flags else None) as f:
            k = self.k = f.induction
            ahead_k = b.op('add', SIZE, k, 'stride', hint='ahead1')
            fits = b.op('lt', BOOL, ahead_k, count, hint='inbatch1')
            self.next = b.op('select', SIZE, fits, ahead_k, k, hint='batchId1',
                             escapes=True)
            f._next_index = self.next
            srcs = [_binding(b, k, base=PTRS if element_pointer else SRC,
                             name=f'glb_m0_{j}' if j else 'glb_m0',
                             element_pointer=element_pointer)
                    for j in range(transfers)]
            out = _binding(b, k, base=DST, name='glb_m1', writable=True)
            if flags:
                allowed = b.decl_expr(
                    'const bool allowed', 'static_cast<bool>(flags0[{0}])',
                    BOOL, None, args=(k,), hint='allowed', extern='allowed')
                guard = b.if_(allowed, attrs=(('guard', 'element'),))
            else:
                from contextlib import nullcontext
                guard = nullcontext()
            with guard:
                if ahead == 'store':
                    b.store(srcs[0], 1.0, 0)
                if ahead == 'barrier':
                    b.barrier(Participants.MULT, threads=32)
                self.bufs = []
                for j, src in enumerate(srcs):
                    if shared == 'sync':
                        # Loads and stores, the way a target without
                        # asynchronous copies moves the bytes.
                        dest = self.window
                        b.mark('defines', dest)
                        lane = b.thread_id('x')
                        with b.for_(0, 8, 1, unroll=True) as i:
                            b.store(dest, b.load(src, i.induction, hint='g'),
                                    i.induction)
                        self.bufs.append(dest)
                        b.store(out, b.load(dest, lane, hint='u'), lane)
                    elif shared:
                        dest = self.window
                        if ahead == 'read':
                            b.store(out, b.load(dest, 0, hint='early'), 0)
                        b.mark('defines', dest)
                        lane = b.thread_id('x')
                        tok = b.copy_async(dest, src, dst_index=(lane,),
                                           src_index=(lane,))
                        self.bufs.append(dest)
                        b.store(out, 0.0, 1)     # something between
                        b.wait(tok)
                        b.store(out, b.load(dest, lane, hint='u'), lane)
                    else:
                        dest = b.alloc(F32, (8,), MemSpace.REGISTER,
                                       hint='r', extern=f'r{j}', init=init)
                        if ahead == 'read':
                            b.store(out, b.load(dest, 0, hint='early'), 0)
                        self.bufs.append(dest)
                        _register_transfer(b, src, dest, f'r{j}')
                        if second_writer:
                            b.store(dest, 2.0, 0)
                        b.store(out, b.load(dest, 3, hint='u'), j)
                        if reread:
                            b.store(out, b.load(dest, 4, hint='v'), j + 8)
            if closing_barrier:
                b.barrier(Participants.MULT, threads=32)
        if read_after:
            b.load(self.window, 0, hint='late')
        self.body = b.finish()
        verify(self.body)

    def wrap(self, distance=1, stages=1):
        self.report = []
        out = wrap_loads(self.body, self.b.scratch, distance=distance,
                         stages=stages, report=self.report)
        verify(out)
        return out


def _loop(body):
    return next(s for s in body if s.op is Op.FOR and s.attr('next'))


def _moved(body) -> bool:
    return any(s.op is Op.ALLOC and s.target[0].type.space is
               MemSpace.REGISTER for s in body) or any(
        s.op is Op.FOR and s.target for s in body)


def _ops_before_loop(body):
    i = next(i for i, s in enumerate(body) if s.op is Op.FOR and s.attr('next'))
    return body[:i]


def _reads_through(stmts):
    """The pointer bindings the global loads in `stmts` read through."""
    return [s.args[0] for s, _ in walk(tuple(stmts))
            if s.op is Op.LOAD and s.accesses
            and s.accesses[0].space is MemSpace.GLOBAL]


def _defs(body):
    return {t.id: s for s, _ in walk(body) for t in s.target}


# --------------------------------------------------------------------------- #
# where a moved transfer goes
# --------------------------------------------------------------------------- #

def test_a_register_transfer_moves_behind_the_guard():
    sec = Section()
    out = sec.wrap()
    assert sec.report == ['+ r0 [reg]']
    loop = _loop(out)
    region = loop.regions[0].body
    guard = next(s for s in region if s.op is Op.IF)
    inside = guard.regions[0].body
    # Out of the guard: nothing inside writes the register any more ...
    assert not any(a.writes and a.base is sec.bufs[0]
                   for s, _ in walk(inside) for a in s.accesses)
    # ... and the tail behind it fills it, for the next element.
    after = region[region.index(guard) + 1:]
    tail = [s for s in after if s.op is Op.FOR]
    assert tail, 'no transfer at the tail'
    defs = _defs(out)
    bound = _reads_through(tail)
    assert bound and all(defs[v.id].args == (sec.next,) for v in bound)
    assert all(defs[v.id].attr('extern') == 'wrap_glb_m0' for v in bound)


def test_the_tail_comes_ahead_of_the_closing_barrier_for_a_register():
    out = Section().wrap()
    region = _loop(out).regions[0].body
    barrier = max(i for i, s in enumerate(region) if s.op is Op.BARRIER)
    tail = [i for i, s in enumerate(region) if s.op is Op.FOR]
    assert tail and max(tail) < barrier


def test_the_declaration_leaves_the_loop_and_the_peel_fills_it():
    sec = Section()
    out = sec.wrap()
    head = _ops_before_loop(out)
    assert any(s.op is Op.ALLOC and s.target[0] is sec.bufs[0] for s in head)
    assert not any(s.op is Op.ALLOC and s.target[0] is sec.bufs[0]
                   for s, _ in walk((_loop(out),)))
    defs = _defs(out)
    bound = _reads_through(head)
    assert bound and all(defs[v.id].args == (sec.first,) for v in bound), (
        'the peel reads the first element, the one the loop names')
    assert all(defs[v.id].attr('extern') == 'peel_glb_m0' for v in bound)


def test_the_binding_nothing_reads_any_more_goes():
    out = Section().wrap()
    names = [s.attr('extern') for s, _ in walk(out) if s.attr('extern')]
    assert 'glb_m0' not in names
    assert {'peel_glb_m0', 'wrap_glb_m0', 'glb_m1'} <= set(names)


def test_the_moved_body_renders():
    out = Section().wrap()
    w = Writer()
    emit(placed(out), w, vm_factory('sm_86', 'cuda', 'float'))
    src = w.get_src()
    assert 'peel_glb_m0' in src.partition('for (')[0]
    assert 'wrap_glb_m0' in src.partition('for (')[2]


# --------------------------------------------------------------------------- #
# which transfers
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('distance,moved', [(1, ['r0']), (2, ['r0', 'r1']),
                                            (5, ['r0', 'r1'])])
def test_the_distance_is_how_many_transfers_wrap(distance, moved):
    sec = Section(transfers=2)
    sec.wrap(distance)
    assert [l[2:].split()[0] for l in sec.report if l.startswith('+')] == moved


def test_a_loop_that_names_no_first_element_is_left_alone():
    """A traversal with nothing to peel for -- a group of rows driven in
    lockstep -- names no first element, and the loop stays as it is."""
    sec = Section(first=False)
    assert sec.wrap() is sec.body


# --------------------------------------------------------------------------- #
# what it may not cross
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('kwargs,why', [
    (dict(ahead='read'), 'access to its buffer'),
    (dict(ahead='store'), 'write of what it reads'),
    (dict(second_writer=True), 'written by more than this transfer'),
    (dict(init='{1.0f}'), 'initializer'),
    (dict(shared=True, ahead='barrier'), 'barrier'),
    (dict(shared=True, ahead='read'), 'access to its buffer'),
])
def test_it_refuses(kwargs, why):
    sec = Section(**kwargs)
    out = sec.wrap()
    assert out is sec.body
    assert any(why in line for line in sec.report), sec.report


def test_a_barrier_ahead_of_a_register_transfer_is_no_obstacle():
    """A barrier orders shared memory, and a register is the thread's own."""
    sec = Section(ahead='barrier')
    sec.wrap()
    assert sec.report == ['+ r0 [reg]']


def test_a_buffer_read_twice_still_moves_behind_both_reads():
    sec = Section(reread=True)
    out = sec.wrap()
    region = _loop(out).regions[0].body
    guard = next(i for i, s in enumerate(region) if s.op is Op.IF)
    tail = next(i for i, s in enumerate(region) if s.op is Op.FOR)
    assert tail > guard


# --------------------------------------------------------------------------- #
# a shared buffer
# --------------------------------------------------------------------------- #

def test_a_shared_transfer_carries_its_tokens_and_is_drained():
    sec = Section(shared=True, flags=False)
    out = sec.wrap()
    assert sec.report == ['+ s0 [shr]']
    loop = _loop(out)
    tokens = [v for v in loop.regions[0].args[1:]]
    assert len(tokens) == 1 and len(loop.target) == 1
    # The wait in the body names the carried token, the drain the result.
    waits = [s for s, _ in walk(loop.regions[0].body) if s.op is Op.WAIT]
    assert [w.args for w in waits] == [(tokens[0],)]
    after = out[out.index(loop) + 1:]
    assert after and after[0].op is Op.WAIT and after[0].args == loop.target
    scheduled, diag = schedule_async(out)
    assert not diag, diag
    loop = _loop(scheduled)
    prior = [s.attr('prior') for s, _ in walk(loop.regions[0].body)
             if s.op is Op.WAIT]
    assert prior == [0], prior


def test_a_masked_element_retires_the_copy_it_skips_the_wait_of():
    """Element k masked, its consumer's wait is skipped and the copy issued
    for it is still in flight when the tail issues k + 1's into the same
    buffer -- two copies of one thread to one place, in no promised order.
    So the other path of the guard retires it."""
    sec = Section(shared=True)
    out = sec.wrap()
    region = _loop(out).regions[0].body
    guard = next(s for s in region if s.op is Op.IF
                 and s.attr('guard') == 'element')
    assert len(guard.regions) == 2
    assert [s.op for s in guard.regions[1].body] == [Op.WAIT]
    assert guard.regions[1].body[0].args == ()
    scheduled, diag = schedule_async(out)
    assert not diag, diag


def test_a_shared_tail_comes_behind_the_closing_barrier():
    sec = Section(shared=True, flags=False)
    out = sec.wrap()
    region = _loop(out).regions[0].body
    barrier = max(i for i, s in enumerate(region) if s.op is Op.BARRIER)
    copies = [i for i, s in enumerate(region) if s.op is Op.COPY_ASYNC]
    assert copies and min(copies) > barrier


def test_a_two_hop_peel_and_tail_issue_the_same_copies():
    sec = Section(shared=True, flags=False)
    out = sec.wrap()
    peel = [s for s in _ops_before_loop(out) if s.op is Op.COPY_ASYNC]
    tail = [s for s in _loop(out).regions[0].body if s.op is Op.COPY_ASYNC]
    assert len(peel) == len(tail) == 1


# --------------------------------------------------------------------------- #
# a second stage
# --------------------------------------------------------------------------- #

def _allocs(stmts):
    return [s for s in stmts if s.op is Op.ALLOC]


def _staged(out):
    """The peel's window, and the loop's two: the one it reads and the one
    it fills."""
    loop = _loop(out)
    region = loop.regions[0].body
    peel = [s for s in _allocs(_ops_before_loop(out))
            if s.target[0].type.space is MemSpace.SHARED]
    read, fill = [s for s in _allocs(region)
                  if s.target[0].type.space is MemSpace.SHARED]
    return loop, peel, read, fill


def test_a_second_stage_puts_the_copy_at_the_head():
    sec = Section(shared=True, flags=False)
    out = sec.wrap(stages=2)
    assert sec.report == ['+ s0 [shr, 2 stages]']
    loop, _, read, fill = _staged(out)
    region = loop.regions[0].body
    copies = [i for i, s in enumerate(region) if s.op is Op.COPY_ASYNC]
    waits = [i for i, s in enumerate(region) if s.op is Op.WAIT]
    assert copies and waits and max(copies) < min(waits), (
        'the copy for the next element is issued ahead of the wait for this '
        'one')
    copy = region[copies[0]]
    assert copy.args[0] is fill.target[0], 'it fills the stage not read'
    # The window the iteration reads is the buffer the body names.
    assert read.target[0] is sec.window
    scheduled, diag = schedule_async(out)
    assert not diag, diag
    prior = [s.attr('prior') for s, _ in walk(_loop(scheduled).regions[0].body)
             if s.op is Op.WAIT]
    assert prior == [1], 'the copy just issued stays in flight'


def test_the_stage_is_carried_and_not_taken_from_the_element():
    """One thread's elements are a stride apart: a stage taken from the
    element would not alternate where two divides the stride, and the peel
    could fill only one of them for the first."""
    sec = Section(shared=True, flags=False)
    out = sec.wrap(stages=2)
    loop, peel, read, fill = _staged(out)
    stage = read.args[0]
    assert stage in loop.regions[0].args[1:], 'the stage rides the loop'
    defs = _defs(out)
    following = defs[fill.args[0].id]
    assert following.op == 'bitxor' and following.args == (stage, 1)
    init = loop.args[3 + loop.regions[0].args.index(stage) - 1]
    assert defs[init.id].attr('value') == 0, 'the first iteration reads 0'
    assert not peel[0].args, 'and the peel fills it'
    assert loop.regions[0].body[-1].args[-1] is fill.args[0]


def test_the_windows_are_one_buffer_twice_over():
    sec = Section(shared=True, flags=False)
    out = sec.wrap(stages=2)
    _, peel, read, fill = _staged(out)
    windows = peel + [read, fill]
    assert len({id(s.attr('identity')) for s in windows}) == 1
    assert {s.attr('stages') for s in windows} == {2}
    assert [s.attr('extern') for s in windows] == ['peel_s0', 's0', 'wrap_s0']
    laid = placed(out)
    offsets = {s.attr('extern'): s.attr('offset') for s, _ in walk(laid)
               if s.op is Op.ALLOC and s.attr('extern')}
    assert offsets['peel_s0'] == 0
    assert offsets['s0'] == offsets['wrap_s0'] == '0 + ({0}) * 64'
    w = Writer()
    emit(laid, w, vm_factory('sm_86', 'cuda', 'float'))
    src = w.get_src()
    assert re.search(r'float \* s0 = &shrMem\[0 \+ \(v\d+_stage\) \* 64\];',
                     src), src
    assert re.search(r'float \* wrap_s0 = &shrMem\[0 \+ \(v\d+_stageNext\) '
                     r'\* 64\];', src), src
    assert 'float * peel_s0 = &shrMem[0];' in src


def test_the_stage_read_is_not_declared_dead_where_the_other_is_filled():
    """`mark defines` ends what a buffer holds, and the windows are one
    buffer to the allocator: at the head of the body the stage the
    iteration reads is still wanted."""
    out = Section(shared=True, flags=False).wrap(stages=2)
    assert not any(s.op is Op.MARK and s.attr('mark') == 'defines'
                   for s, _ in walk(_loop(out).regions[0].body))


def test_with_a_mask_the_wait_goes_ahead_of_the_guard():
    """A masked element cannot drain behind a copy issued at the head: it
    would retire that one as well, and leave the next iteration a different
    count in flight on each path.  Ahead of the guard both paths wait."""
    sec = Section(shared=True)
    out = sec.wrap(stages=2)
    region = _loop(out).regions[0].body
    guard = next(i for i, s in enumerate(region) if s.op is Op.IF)
    waits = [i for i, s in enumerate(region) if s.op is Op.WAIT]
    assert waits and max(waits) < guard
    assert len(region[guard].regions) == 1, 'no drain on the masked path'
    scheduled, diag = schedule_async(out)
    assert not diag, diag
    prior = [s.attr('prior') for s, _ in walk(_loop(scheduled).regions[0].body)
             if s.op is Op.WAIT]
    assert prior == [1]


def test_the_next_pointer_is_followed_under_its_flag_at_the_head():
    sec = Section(shared=True, element_pointer=True)
    out = sec.wrap(stages=2)
    region = _loop(out).regions[0].body
    guard = next(i for i, s in enumerate(region) if s.op is Op.IF
                 and s.attr('guard') == 'element')
    copies = [s for s in region[:guard] if s.op is Op.COPY_ASYNC]
    assert copies and all(c.predicate is not None for c in copies)
    defs = _defs(out)
    assert defs[copies[0].predicate.id].attr('extern') == 'allowed_next'


@pytest.mark.parametrize('kwargs,why', [
    (dict(shared='sync'), 'loads and stores'),
    (dict(shared=True, read_after=True), 'used outside the loop'),
])
def test_one_stage_stays_where_two_would_not_help(kwargs, why):
    sec = Section(flags=False, **kwargs)
    out = sec.wrap(stages=2)
    assert len(sec.report) == 1 and sec.report[0].startswith('+ s0 [shr] '
                                                             'one stage:')
    assert why in sec.report[0], sec.report
    assert not any(s.attr('stages') for s, _ in walk(out)
                   if s.op is Op.ALLOC)


def test_a_register_transfer_keeps_its_place_beside_a_second_stage():
    sec = Section(flags=False)
    out = sec.wrap(stages=2)
    assert sec.report == ['+ r0 [reg]']
    region = _loop(out).regions[0].body
    barrier = max(i for i, s in enumerate(region) if s.op is Op.BARRIER)
    tail = [i for i, s in enumerate(region) if s.op is Op.FOR]
    assert tail and max(tail) < barrier


# --------------------------------------------------------------------------- #
# the mask
# --------------------------------------------------------------------------- #

def test_the_flags_ride_the_loop():
    sec = Section()
    out = sec.wrap()
    loop = _loop(out)
    carried = loop.regions[0].args[1:]
    assert [v.type for v in carried] == [carried[0].type] * 2
    region = loop.regions[0].body
    guard = next(s for s in region if s.op is Op.IF)
    defs = _defs(out)
    assert defs[guard.cond.id].args == (carried[0],), (
        'the guard reads the word the iteration came in with')
    assert not any(s.attr('extern') == 'allowed' and s.args == (sec.k,)
                   for s, _ in walk(out)), 'the read at the head went'
    yielded = region[-1].args
    assert yielded[0] is carried[1], 'the next word becomes the own one'


def test_a_pointer_of_the_element_is_followed_under_its_flag():
    sec = Section(element_pointer=True)
    out = sec.wrap()
    region = _loop(out).regions[0].body
    guard = next(i for i, s in enumerate(region) if s.op is Op.IF
                 and s.attr('guard') == 'element')
    tails = [s for s in region[guard + 1:] if s.op is Op.IF]
    assert tails, 'the tail is not under a flag'
    defs = _defs(out)
    assert defs[tails[0].cond.id].attr('extern') == 'allowed_next'
    peels = [s for s in _ops_before_loop(out) if s.op is Op.IF]
    assert peels and defs[peels[0].cond.id].attr('extern') == 'allowed_peel'
    # The array entry itself is read at the head, unconditionally.
    bound = [s for s in region[:guard] if s.attr('extern') == 'wrap_glb_m0']
    assert bound


def test_without_a_mask_a_pointer_is_followed_unguarded():
    sec = Section(element_pointer=True, flags=False)
    out = sec.wrap()
    region = _loop(out).regions[0].body
    assert not any(s.op is Op.IF for s in region)


# --------------------------------------------------------------------------- #
# text
# --------------------------------------------------------------------------- #

def test_a_name_in_text_is_spelled_for_the_copy():
    """Text names what it is about -- the comment over a transfer names the
    pointer it reads -- and a copy for another element reads another one, so
    the copy's text names that."""
    out = Section().wrap()
    comments = [s.text for s, _ in walk(out) if s.op is Op.RAWSTMT
                and (s.text or '').startswith('// r0')]
    assert '// r0 = load{g>r}(peel_glb_m0)' in comments
    assert '// r0 = load{g>r}(wrap_glb_m0)' in comments
