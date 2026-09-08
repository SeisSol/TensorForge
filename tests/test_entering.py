# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""What a repeated region needs that it does not start with.

Two kinds, and they want opposite treatments: one is a loop-carried argument
initialised before the header, the other is a read nobody produces and is owed
a definition by whoever assembled the region.  One answer for both would send
half of them to the wrong fix.
"""

from tensorforge.backend.opt.inspect import entering


class Sym:
    def __init__(self, name):
        self.name = name
        self.stype = None

    def __repr__(self):
        return self.name


class Instr:
    """Just the two questions `verify` asks of an instruction."""

    def __init__(self, writes=(), reads=()):
        self._writes, self._reads = tuple(writes), tuple(reads)

    def defs(self):
        return self._writes

    def uses(self):
        return self._reads


def names(syms):
    return [s.name for s in syms]


def test_a_region_that_defines_before_it_reads_needs_nothing():
    r = Sym('r0')
    carried, missing = entering([Instr(writes=[r]), Instr(reads=[r])])
    assert (carried, missing) == ((), ())


def test_a_value_read_then_written_is_carried():
    """What the first iteration reads is what the previous one wrote."""
    acc = Sym('acc')
    carried, missing = entering([Instr(reads=[acc]), Instr(writes=[acc])])
    assert names(carried) == ['acc']
    assert missing == ()


def test_a_value_never_written_is_missing_and_not_carried():
    x = Sym('x')
    carried, missing = entering([Instr(reads=[x])])
    assert carried == ()
    assert names(missing) == ['x']


def test_what_is_live_on_entry_is_neither():
    x = Sym('x')
    carried, missing = entering([Instr(reads=[x])], predefined=[x])
    assert (carried, missing) == ((), ())


def test_each_is_reported_once():
    acc = Sym('acc')
    region = [Instr(reads=[acc]), Instr(writes=[acc]), Instr(reads=[acc])]
    carried, _ = entering(region)
    assert names(carried) == ['acc']


def test_the_two_kinds_are_told_apart_in_one_region():
    acc, x = Sym('acc'), Sym('x')
    carried, missing = entering([Instr(reads=[acc, x]), Instr(writes=[acc])])
    assert names(carried) == ['acc']
    assert names(missing) == ['x']


def test_an_empty_region_needs_nothing():
    assert entering([]) == ((), ())
