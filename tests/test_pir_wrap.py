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
from tensorforge.backend.pir.transfers import one_pipeline
from tensorforge.backend.pir.wrap import wrap_loads
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


def _register_transfer(b, src, dest, name, n=8, held=None):
    # The comment the macro layer writes, in its names.  `held` is a predicate
    # its loads keep to, as a ragged lane tail's do.
    b.Comment(f'{name} = load{{g>r}}(glb_m0)')
    with b.for_(0, n, 1, unroll=True) as i:
        b.store(dest, b.load(src, i.induction, hint='g', predicate=held),
                i.induction)


class Section:
    """A section body around one batch loop, and handles to its parts."""

    def __init__(self, *, flags=True, element_pointer=False, transfers=1,
                 shared=False, ahead=None, init='', second_writer=False,
                 closing_barrier=True, first=True, reread=False,
                 read_after=False, trailing=0, reg_read_after=False,
                 late_next=False, held=False):
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

            def successor():
                ahead_k = b.op('add', SIZE, k, 'stride', hint='ahead1')
                fits = b.op('lt', BOOL, ahead_k, count, hint='inbatch1')
                self.next = b.op('select', SIZE, fits, ahead_k, k,
                                 hint='batchId1', escapes=True)
                f._next_index = self.next
            if not late_next:
                successor()
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
            # Declared ahead of the guard where it is read behind it as well.
            early_alloc = (b.alloc(F32, (8,), MemSpace.REGISTER, hint='r',
                                   extern='r0', init=init)
                           if reg_read_after else None)
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
                        dest = (early_alloc if early_alloc is not None
                                and j == 0 else
                                b.alloc(F32, (8,), MemSpace.REGISTER,
                                        hint='r', extern=f'r{j}', init=init))
                        if ahead == 'read':
                            b.store(out, b.load(dest, 0, hint='early'), 0)
                        self.bufs.append(dest)
                        lanes = (b.op('lt', BOOL, b.thread_id('x'), 4,
                                      hint='g') if held else None)
                        _register_transfer(b, src, dest, f'r{j}', held=lanes)
                        if second_writer:
                            b.store(dest, 2.0, 0)
                        b.store(out, b.load(dest, 3, hint='u'), j)
                        if reread:
                            b.store(out, b.load(dest, 4, hint='v'), j + 8)
                # Work behind the last read of the buffers, which a transfer
                # issued early has as cover.
                for n in range(trailing):
                    b.store(out, float(n), 16 + n)
            if reg_read_after:
                b.store(out, b.load(self.bufs[0], 5, hint='w'), 30)
            if late_next:
                # The next element's index computed at the end of the body,
                # behind every read of the buffers.
                successor()
            if closing_barrier:
                b.barrier(Participants.MULT, threads=32)
        if read_after:
            b.load(self.window, 0, hint='late')
        self.body = b.finish()
        verify(self.body)

    def wrap(self, distance=1, stages=1, **kwargs):
        self.report = []
        out = wrap_loads(self.body, self.b.scratch, distance=distance,
                         stages=stages, report=self.report, **kwargs)
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
    emit(placed(out), w, Target('sm_86', 'cuda'))
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
    """A traversal that names no first element has nothing to peel for, and
    the loop stays as it is."""
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
    emit(laid, w, Target('sm_86', 'cuda'))
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


def _global_loads(stmts):
    return [x for x, _ in walk(tuple(stmts)) if x.op is Op.LOAD
            and any(a.space is MemSpace.GLOBAL for a in x.accesses)]


def test_a_pointer_of_the_element_is_followed_under_its_flag():
    """A transfer into registers takes the flag into the predicate of its
    loads: no block around it, and zeros where the element is masked."""
    sec = Section(element_pointer=True)
    out = sec.wrap()
    region = _loop(out).regions[0].body
    guard = next(i for i, s in enumerate(region) if s.op is Op.IF
                 and s.attr('guard') == 'element')
    defs = _defs(out)
    tail = _global_loads(region[guard + 1:])
    assert tail and all(
        defs[x.predicate.id].attr('extern') == 'allowed_next' for x in tail)
    assert not any(s.op is Op.IF for s in region[guard + 1:])
    peel = _global_loads(_ops_before_loop(out))
    assert peel and all(
        defs[x.predicate.id].attr('extern') == 'allowed_peel' for x in peel)
    # The array entry itself is read at the head, unconditionally.
    bound = [s for s in region[:guard] if s.attr('extern') == 'wrap_glb_m0']
    assert bound


def test_a_load_keeps_its_own_predicate_behind_the_flag():
    """A load already held to some of the lanes is held to them and to the
    flag."""
    sec = Section(element_pointer=True, held=True)
    out = sec.wrap()
    region = _loop(out).regions[0].body
    guard = next(i for i, s in enumerate(region) if s.op is Op.IF
                 and s.attr('guard') == 'element')
    defs = _defs(out)
    tail = _global_loads(region[guard + 1:])
    assert tail
    for x in tail:
        both = defs[x.predicate.id]
        assert both.op == 'and'
        flag, lanes = both.args
        assert defs[flag.id].attr('extern') == 'allowed_next'
        assert defs[lanes.id].op == 'lt'


def test_a_copy_written_as_loads_and_stores_is_followed_in_a_block():
    """Into shared memory, zeros would land in a window other threads read:
    the transfer stays in a block under the flag."""
    sec = Section(shared='sync', element_pointer=True)
    out = sec.wrap()
    region = _loop(out).regions[0].body
    guard = next(i for i, s in enumerate(region) if s.op is Op.IF
                 and s.attr('guard') == 'element')
    tails = [s for s in region[guard + 1:] if s.op is Op.IF]
    assert tails and all(x.predicate is None for x in _global_loads(tails))


def test_under_the_explicit_vector_a_pointer_is_followed_in_a_block():
    """Its loads fill a vector, which a uniform predicate does not mask."""
    sec = Section(element_pointer=True)
    out = sec.wrap(explicit_simd=True)
    region = _loop(out).regions[0].body
    guard = next(i for i, s in enumerate(region) if s.op is Op.IF
                 and s.attr('guard') == 'element')
    tails = [s for s in region[guard + 1:] if s.op is Op.IF]
    assert tails, 'the tail is not under a flag'
    defs = _defs(out)
    assert defs[tails[0].cond.id].attr('extern') == 'allowed_next'
    assert all(x.predicate is None for x in _global_loads(tails))
    peels = [s for s in _ops_before_loop(out) if s.op is Op.IF]
    assert peels and defs[peels[0].cond.id].attr('extern') == 'allowed_peel'


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


# --------------------------------------------------------------------------- #
# a merged run's loop
# --------------------------------------------------------------------------- #

class Run:
    """A merged run's loop, the way `VariantLoop` writes one: a counter, the
    chain selecting a member by it, the binding reading the chain, and the
    transfer through that binding."""

    def __init__(self, *, shared=False, transfer=True, trips=4, trailing=0):
        b = self.b = IRBuilder(fptype=F32, arena='shrMem')
        k = b.extern_value('element', SIZE, uniform=Uniformity.MULT, hint='k')
        members = [_binding(b, k, base=f'm{3 + j}', name=f'glb_m{3 + j}')
                   for j in range(trips)]
        out = _binding(b, k, base=DST, name='glb_m1', writable=True)
        self.dest = (b.alloc(F32, (64,), MemSpace.SHARED, hint='s',
                             extern='s0') if shared
                     else b.alloc(F32, (8,), MemSpace.REGISTER, hint='r',
                                  extern='r0'))
        with b.for_(0, trips, 1, extern='batchIdv0', rolled=True) as run:
            self.counter = c = run.induction
            text = f'{{{trips}}}'
            for j in reversed(range(trips - 1)):
                text = f'({{0}} == {j}) ? {{{j + 1}}} : {text}'
            pointer = BufferType(F32, (64,), MemSpace.GLOBAL, readonly=True)
            self.chain = b.decl_expr('const float *const v0Table', text,
                                     pointer, None, args=(c, *members),
                                     hint='v0Table', extern='v0Table')
            binding = b.decl_expr(
                'const float *const __restrict__ glb_v0', '{0}', pointer,
                'v0', args=(self.chain,), kind=Effect.READ,
                space=MemSpace.GLOBAL, hint='glb_v0', extern='glb_v0')
            lane = b.thread_id('x')
            if transfer and shared:
                b.mark('defines', self.dest)
                tok = b.copy_async(self.dest, binding, dst_index=(lane,),
                                   src_index=(lane,))
                b.store(out, 0.0, 1)
                b.wait(tok)
            elif transfer:
                _register_transfer(b, binding, self.dest, 'r0')
            b.store(out, b.load(self.dest, lane, hint='u'), c)
            for n in range(trailing):
                b.store(out, float(n), 16 + n)
        self.body = b.finish()
        verify(self.body)

    def wrap(self, **kwargs):
        self.report = []
        out = wrap_loads(self.body, self.b.scratch, report=self.report,
                         **kwargs)
        verify(out)
        return out


def _run_loop(body):
    return next(s for s, _ in walk(body) if s.op is Op.FOR
                and s.attr('rolled'))


def _chains(stmts):
    return {s.attr('extern'): s for s, _ in walk(tuple(stmts))
            if s.op is Op.RAWEXPR and s.attr('extern', '').endswith('Table')}


def test_a_run_fetches_the_next_trips_member_at_its_tail():
    """The operand a run's counter selects is a different member every
    trip: fetched for the trip after once this one is done with its buffer,
    from the chain computed again for that trip -- and the first trip's
    ahead of the loop, from the chain at the loop's first trip."""
    run = Run()
    out = run.wrap()
    assert run.report == ['+ r0 [reg] in a run']
    loop = _run_loop(out)
    body = loop.regions[0].body
    following = loop.attr('next')
    ahead = _chains(body)
    assert set(ahead) == {'wrap_v0Table'}
    assert ahead['wrap_v0Table'].args[0] is following
    # the trip after, held at the last one
    defs = _defs(out)
    select = defs[following.id]
    assert select.op == 'select' and select.args[2] is run.counter
    assert defs[select.args[1].id].args[:2] == (run.counter, 1)
    tail = [s for s in body if s.op is Op.FOR]
    assert tail and body.index(tail[-1]) > max(
        i for i, s in enumerate(body) if s.op is Op.LOAD
        and s.accesses[0].base is run.dest)
    peel = _chains(out[:out.index(loop)])
    assert set(peel) == {'peel_v0Table'}
    assert peel['peel_v0Table'].args[0] == 0


def test_a_run_reads_no_member_through_its_own_chain_any_more():
    """What the trip itself selected was only there to be fetched, and the
    fetch is now the trip before's."""
    out = Run().wrap()
    names = {s.attr('extern') for s, _ in walk(out) if s.attr('extern')}
    assert not names & {'v0Table', 'glb_v0'}
    assert {'peel_glb_v0', 'wrap_glb_v0'} <= names


def test_a_runs_shared_copy_rides_the_loop_and_is_drained():
    run = Run(shared=True)
    out = run.wrap()
    assert run.report and run.report[0].startswith('+ s0 [shr]')
    loop = _run_loop(out)
    assert loop.target, 'the loop carries the tail\'s copies'
    behind = out[out.index(loop) + 1:]
    assert any(s.op is Op.WAIT for s in behind)


def test_a_run_without_a_transfer_is_left_alone():
    run = Run(transfer=False)
    assert run.wrap() is run.body
    assert run.report == ['- run: no transfer in the body']


def test_the_trip_a_run_fetches_for_is_named_once():
    """The chain spells its counter once per member: the trip after is a
    value of its own there, and not computed again at every spelling."""
    out = Run().wrap()
    w = Writer()
    emit(placed(out), w, Target('sm_86', 'cuda'))
    src = w.get_src()
    line = next(l for l in src.splitlines() if 'wrap_v0Table =' in l)
    name = re.search(r'\((\w+) == 0\)', line).group(1)
    assert line.count(f'({name} == ') == 3
    assert '?' not in line.split('=', 1)[1].split('== 0')[0]


# --------------------------------------------------------------------------- #
# pipelines
# --------------------------------------------------------------------------- #

def _two_paths():
    """A batch loop whose body loads into registers and then copies into
    shared memory: a transfer of each memory path."""
    b = IRBuilder(fptype=F32, arena='shrMem')
    count = b.extern_value('numElements0', SIZE, hint='count')
    start = b.extern_value('start', SIZE, uniform=Uniformity.MULT,
                           hint='start')
    inside = b.op('lt', BOOL, start, count, hint='inside')
    first = b.op('select', SIZE, inside, start, 0, hint='batchId1',
                 escapes=True)
    window = b.alloc(F32, (64,), MemSpace.SHARED, hint='s', extern='s0')
    with b.for_('start', count, 'stride', hint='batchId0', index_type=SIZE,
                peel_index=first, uniform=Uniformity.MULT) as f:
        k = f.induction
        ahead_k = b.op('add', SIZE, k, 'stride', hint='ahead1')
        fits = b.op('lt', BOOL, ahead_k, count, hint='inbatch1')
        f._next_index = b.op('select', SIZE, fits, ahead_k, k,
                             hint='batchId1', escapes=True)
        src = _binding(b, k, name='glb_m0')
        staged = _binding(b, k, base='m3', name='glb_m3')
        out = _binding(b, k, base=DST, name='glb_m1', writable=True)
        reg = b.alloc(F32, (8,), MemSpace.REGISTER, hint='r', extern='r0')
        _register_transfer(b, src, reg, 'r0')
        b.store(out, b.load(reg, 3, hint='u'), 0)
        b.mark('defines', window)
        lane = b.thread_id('x')
        tok = b.copy_async(window, staged, dst_index=(lane,),
                           src_index=(lane,))
        b.store(out, 0.0, 1)
        b.wait(tok)
        b.store(out, b.load(window, lane, hint='v'), lane)
        b.barrier(Participants.MULT, threads=32)
    body = b.finish()
    verify(body)
    return b, body


def test_the_first_transfer_of_each_pipeline_wraps():
    """A pipeline for every memory path (the default): at distance 1 the
    load into registers and the copy into shared memory are each the first
    of theirs, and both go one element ahead."""
    b, body = _two_paths()
    report = []
    out = wrap_loads(body, b.scratch, report=report)
    verify(out)
    assert report == ['+ r0 [reg]', '+ s0 [shr]']


def test_with_one_pipeline_the_first_transfer_alone_wraps():
    b, body = _two_paths()
    report = []
    out = wrap_loads(body, b.scratch, report=report, pipeline=one_pipeline)
    verify(out)
    assert report == ['+ r0 [reg]']


# --------------------------------------------------------------------------- #
# issued early
# --------------------------------------------------------------------------- #

def _last_read(stmts, buf):
    return max(i for i, s in enumerate(stmts)
               if any(a.base is buf and not a.writes
                      for x, _ in walk((s,)) for a in x.accesses))


def _filling(body, buf):
    """The statements of `body` writing `buf`, the waits that retire a copy
    into it apart."""
    return [i for i, s in enumerate(body)
            if any(a.writes and a.base is buf
                   for x, _ in walk((s,)) if x.op is not Op.WAIT
                   for a in x.accesses)]


def _only_comments_between(body, lo, hi):
    return all(s.op is Op.RAWSTMT and not s.accesses and not s.target
               or s.op is Op.MARK for s in body[lo + 1:hi])


@pytest.mark.parametrize('shared', [False, True], ids=['reg', 'shr'])
def test_a_run_issues_the_next_trip_behind_the_last_read(shared):
    """Issued early, the transfer for the next trip goes right behind the
    trip's last read of its buffer, with the rest of the trip as cover."""
    run = Run(shared=shared, trailing=6)
    out = run.wrap(early=True)
    body = _run_loop(out).regions[0].body
    fills = _filling(body, run.dest)
    assert fills, 'nothing fills the buffer in the loop'
    last = _last_read(body, run.dest)
    assert min(fills) > last and _only_comments_between(body, last,
                                                         min(fills))
    assert len(body) - max(fills) > 6, 'the trailing work stands behind it'


@pytest.mark.parametrize('shared', [False, True], ids=['reg', 'shr'])
def test_a_run_issues_at_the_tail_unless_asked(shared):
    run = Run(shared=shared, trailing=6)
    out = run.wrap()
    body = _run_loop(out).regions[0].body
    fills = _filling(body, run.dest)
    trailing = [i for i, s in enumerate(body) if s.op is Op.STORE
                and isinstance(s.args[1], float)]
    assert min(fills) > max(trailing)


def test_the_cover_cap_holds_an_early_transfer_back():
    """`cover`: no more ahead of the first read than the cap -- here two of
    the six statements behind the last read, and the stretch at the head."""
    run = Run(trailing=6)
    early = _run_loop(run.wrap(early=True)).regions[0].body
    capped = _run_loop(Run(trailing=6).wrap(early=True, cover=3)).regions[0].body
    assert min(_filling(capped, _run_dest(capped))) > min(
        _filling(early, _run_dest(early)))


def _run_dest(body):
    return next(a.base for s in body for x, _ in walk((s,)) for a in x.accesses
                if a.writes and getattr(a.base, 'type', None) is not None
                and a.base.type.space in (MemSpace.REGISTER, MemSpace.SHARED))


@pytest.mark.parametrize('reread', [False, True], ids=['once', 'twice'])
def test_a_guarded_register_transfer_goes_into_the_guard(reread):
    """In a guarded batch loop the transfer into registers for the next
    element goes behind the last read inside the guard -- the last of two,
    not the first, which would refill the buffer the arm still reads -- and a
    masked element issues it in the guard's other arm."""
    sec = Section(trailing=6, reread=reread)
    out = sec.wrap(early=True)
    region = _loop(out).regions[0].body
    guard = next(s for s in region if s.op is Op.IF
                 and s.attr('guard') == 'element')
    inside = guard.regions[0].body
    buf = sec.bufs[0]
    fills = _filling(inside, buf)
    last = _last_read(inside, buf)
    assert fills and min(fills) > last
    assert _only_comments_between(inside, last, min(fills))
    assert len(guard.regions) == 2, 'the masked arm'
    other = guard.regions[1].body
    assert any(a.writes and a.base is buf
               for x, _ in walk(other) for a in x.accesses)
    after = region[region.index(guard) + 1:]
    assert not any(a.writes and a.base is buf
                   for x, _ in walk(tuple(after)) for a in x.accesses), (
        'and nothing behind the guard fills it any more')


def test_a_register_read_behind_the_guard_keeps_the_tail():
    sec = Section(trailing=6, reg_read_after=True, closing_barrier=False)
    out = sec.wrap(early=True)
    region = _loop(out).regions[0].body
    guard = next(s for s in region if s.op is Op.IF
                 and s.attr('guard') == 'element')
    buf = sec.bufs[0]
    assert not any(a.writes and a.base is buf
                   for x, _ in walk(guard.regions[0].body) for a in x.accesses)
    after = region[region.index(guard) + 1:]
    assert any(a.writes and a.base is buf
               for x, _ in walk(tuple(after)) for a in x.accesses)


def test_a_guarded_copy_into_shared_memory_keeps_the_tail():
    """A copy issued in one arm would leave the paths with different counts
    of groups in flight: it stays behind the guard."""
    sec = Section(shared=True, trailing=6)
    out = sec.wrap(early=True)
    region = _loop(out).regions[0].body
    guard = next(s for s in region if s.op is Op.IF
                 and s.attr('guard') == 'element')
    assert not any(x.op is Op.COPY_ASYNC for x, _ in walk(guard.regions[0].body))
    after = region[region.index(guard) + 1:]
    assert any(x.op is Op.COPY_ASYNC for x, _ in walk(tuple(after)))


def test_an_early_body_renders():
    sec = Section(trailing=6)
    out = sec.wrap(early=True)
    w = Writer()
    emit(placed(out), w, Target('sm_86', 'cuda'))
    src = w.get_src()
    assert 'wrap_glb_m0' in src and 'else' in src


def test_an_early_copy_stays_behind_the_barrier_that_closes_the_iteration():
    """Where a barrier closes the iteration it fences every thread's reads
    of the window against the write for the next element: issued early or
    not, a copy into shared memory stays behind it."""
    sec = Section(flags=False, shared=True, trailing=6)
    region = _loop(sec.wrap(early=True)).regions[0].body
    closing = max(i for i, s in enumerate(region) if s.op is Op.BARRIER)
    copies = [i for i, s in enumerate(region)
              if any(x.op is Op.COPY_ASYNC for x, _ in walk((s,)))]
    assert copies and min(copies) > closing


def test_an_early_copy_goes_behind_the_last_read_where_no_barrier_closes():
    sec = Section(flags=False, shared=True, trailing=6, closing_barrier=False)
    region = _loop(sec.wrap(early=True)).regions[0].body
    last = _last_read(region, sec.window)
    copies = [i for i, s in enumerate(region)
              if any(x.op is Op.COPY_ASYNC for x, _ in walk((s,)))]
    assert copies and min(copies) > last
    assert _only_comments_between(region, last, min(copies))


def test_an_early_transfer_stands_behind_the_head():
    """The transfer for the next element reads the pointer the head computes
    for it, and goes behind the head even where the buffer's last read comes
    first -- here the body computes the next element's index at its end."""
    sec = Section(flags=False, closing_barrier=False, trailing=6,
                  late_next=True)
    region = _loop(sec.wrap(early=True)).regions[0].body
    pointer = next(i for i, s in enumerate(region)
                   if str(s.attr('extern') or '').startswith('wrap_'))
    fills = _filling(region, sec.bufs[0])
    assert pointer > _last_read(region, sec.bufs[0])
    assert fills and min(fills) > pointer
