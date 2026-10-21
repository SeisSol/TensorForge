# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""A copy per multiplication of what a row of a group reads, and a pick
(`pir.rows.group_rows`).

The pass takes on trust only what the builder marks: the row's place in its
group (`mult_lane`) and a read through it that each multiplication is to
have a copy of (`per_mult`).  Everything else stays the lane's: a read
nobody marked, and one whose row passes through something that cannot be
computed again.
"""

from __future__ import annotations

from tensorforge.backend.pir import BOOL, SIZE, Effect, IRBuilder, Op, emit
from tensorforge.backend.pir.rows import group_rows
from tensorforge.backend.pir.core import walk_stmts
from tensorforge.backend.writer import Writer
from tensorforge.common.basic_types import Datatype


def _pick(index, values):
    return f'pick({index}, {", ".join(values)})'


def _place(builder, group=2):
    return builder.op('rem', SIZE, builder.thread_id('y'), group, hint='place',
                      attrs=(('mult_lane', group),))


def _flag(builder, row, name='flag'):
    """A read through the row, marked for a copy per multiplication."""
    return builder.decl_expr(f'const bool {name}', 'flags[{0}] != 0', BOOL,
                             None, args=(row,), extern=name,
                             attrs=(('per_mult', True),))


def _source(builder, body) -> str:
    writer = Writer()
    emit(body, writer)
    return writer.get_src()


def _by_name(body):
    return {s.attr('extern'): s for s in walk_stmts(body) if s.attr('extern')}


def test_a_read_through_the_place_is_a_pick_among_copies():
    builder = IRBuilder(fptype=Datatype.F32)
    place = _place(builder)
    group = builder.extern_value('group', SIZE)
    row = builder.op('add', SIZE, group, place, hint='row')
    _flag(builder, row)
    builder.decl_expr('const bool other', 'other[{0}] != 0', BOOL, None,
                      args=(row,), extern='other')
    body = group_rows(builder.finish(), builder.scratch, _pick)
    named = _by_name(body)
    pick = named['flag']
    assert pick.text == 'pick({0}, {1}, {2})'
    assert pick.args[0] is place
    assert [a for a in pick.args[1:]] == [named['mult0_flag'].target[0],
                                         named['mult1_flag'].target[0]]
    assert pick.attr('per_mult') is None
    for mult in (0, 1):
        copy = named[f'mult{mult}_flag']
        assert copy.text == 'flags[{0}] != 0'
        assert copy.attr('per_mult') is None
    # The place is the multiplication's own in its copy.
    src = _source(builder, body)
    assert 'group + 0' in src and 'group + 1' in src, src
    # Unmarked, the read stays the lane's.
    assert named['other'].text == 'other[{0}] != 0'
    assert named['other'].args == (row,)


def test_what_a_row_reads_is_copied_once_per_multiplication():
    """Two reads through one flag: the flag is copied once per
    multiplication, and the second read's copies take those."""
    builder = IRBuilder(fptype=Datatype.F32)
    place = _place(builder)
    group = builder.extern_value('group', SIZE)
    row = builder.op('add', SIZE, group, place, hint='row')
    flag = _flag(builder, row)
    element = builder.op('select', SIZE, flag, row, group, hint='element')
    builder.decl_expr('const int *const ptr', '&table[{0}][0]', SIZE, None,
                      args=(element,), extern='ptr',
                      attrs=(('per_mult', True),))
    body = group_rows(builder.finish(), builder.scratch, _pick)
    named = _by_name(body)
    assert sorted(n for n in named if n.startswith('mult')) == [
        'mult0_flag', 'mult0_ptr', 'mult1_flag', 'mult1_ptr']
    for mult in (0, 1):
        flag_copy = named[f'mult{mult}_flag'].target[0]
        selects = [s for s in walk_stmts(body)
                   if s.op == 'select' and s.args[0] is flag_copy]
        assert len(selects) == 1
        assert named[f'mult{mult}_ptr'].args == selects[0].target


def test_a_row_through_what_cannot_be_copied_stays_the_lanes():
    """A read through the row and through a call with an effect: a copy
    would do the call again, or read the lane's own result of it, so the
    read is left as it is."""
    builder = IRBuilder(fptype=Datatype.F32)
    place = _place(builder)
    group = builder.extern_value('group', SIZE)
    row = builder.op('add', SIZE, group, place, hint='row')
    other = builder.call('next_row', SIZE, place, pure=False, movable=False,
                         effect=Effect.UNKNOWN, hint='other')
    builder.decl_expr('const bool flag', 'flags[{0}] && flags[{1}]', BOOL,
                      None, args=(row, other), extern='flag',
                      attrs=(('per_mult', True),))
    before = builder.finish()
    after = group_rows(before, builder.scratch, _pick)
    assert after == before


def test_a_read_in_a_nested_scope_takes_the_copies_made_outside():
    builder = IRBuilder(fptype=Datatype.F32)
    place = _place(builder)
    group = builder.extern_value('group', SIZE)
    row = builder.op('add', SIZE, group, place, hint='row')
    flag = _flag(builder, row)
    with builder.if_(builder.op('lt', BOOL, group, 8, hint='inside')):
        builder.decl_expr('const bool again', '{0} && more', BOOL, None,
                          args=(flag,), extern='again',
                          attrs=(('per_mult', True),))
    body = group_rows(builder.finish(), builder.scratch, _pick)
    named = _by_name(body)
    copies = [n for n in named if n.startswith('mult')]
    assert sorted(copies) == ['mult0_again', 'mult0_flag', 'mult1_again',
                              'mult1_flag']
    for mult in (0, 1):
        assert named[f'mult{mult}_again'].args == (
            named[f'mult{mult}_flag'].target[0],)
    guard = next(s for s in body if s.op == Op.IF)
    inside = {s.attr('extern') for s in walk_stmts(guard.regions[0].body)}
    assert {'mult0_again', 'mult1_again', 'again'} <= inside
    assert 'mult0_flag' not in inside


def test_a_body_without_a_marked_place_is_left_alone():
    builder = IRBuilder(fptype=Datatype.F32)
    place = builder.op('rem', SIZE, builder.thread_id('y'), 2, hint='place')
    _flag(builder, builder.op('add', SIZE, 4, place, hint='row'))
    before = builder.finish()
    assert group_rows(before, builder.scratch, _pick) is before
