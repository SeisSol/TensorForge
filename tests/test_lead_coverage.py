# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""The lead loop covers each element of its range exactly once.

Not read off one generated kernel.  Which elements a nest touches is fully
decided at generation time -- the slot bounds, the guards and the peel are all
arithmetic in `LeadLoop.write` -- so the property can be *proved* over a
matrix of shapes rather than measured on a device or eyeballed in one file.
Running `write` against a recording writer and counting how many (lane, block)
pairs land on each element is the whole of it.

Two different requirements are checked, because the store nest serves two
callers with different needs:

* **coverage** -- every element in `[start, end)` touched at least once.  What
  a plain store needs.  An element written twice with the same value costs a
  store and nothing else.
* **exactness** -- every element touched exactly once.  What an atomic
  accumulation needs, and what `placement.atomic_write_is_exact` gates on.

The distinction is why a defect here can sit in the tree for a long time
looking like a performance question.

`_narrow` is deliberately not exercised: it only fires under the explicit-SIMD
lowering, which reports through `writer._explicit_simd`, and the recording
writer says no.  A separate matrix for the narrowed shapes is worth having and
is a different fixture, since the emitted object is a vector extent rather
than a lane range.
"""

from __future__ import annotations

from collections import Counter

import pytest

from tensorforge.backend.symbol import LeadIndex, LeadLoop


class _Guard:
    """A recorded `lead >= lo && lead < hi`, as a context manager."""

    def __init__(self, rec, lo, hi):
        self.rec, self.lo, self.hi = rec, lo, hi

    def __enter__(self):
        self.rec._guards.append((self.lo, self.hi))
        return self

    def __exit__(self, *exc):
        self.rec._guards.pop()
        return False


class _Loop:
    """A recorded `for` over slots, whose induction variable is a marker."""

    class _Induction:
        def __init__(self, lo, hi):
            self.lo, self.hi = lo, hi

        def __str__(self):
            return f'slots[{self.lo},{self.hi})'

    def __init__(self, lo, hi):
        self.induction = self._Induction(lo, hi)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class Recorder:
    """A writer that records guards and slots instead of emitting anything.

    The surface is exactly what `LeadLoop.write` touches, which is small and
    is the point: the loop asks the writer for a lane index, a comparison, a
    branch and a counted loop, and nothing it asks for needs a real IR to
    answer.
    """

    def __init__(self, threads):
        self.threads = threads
        self._guards = []
        #: `(slot, lo, hi, width, offset)` for each block, and `('peel', e)`
        #: for each scalar leftover.
        self.blocks = []

    # -- the writer surface -------------------------------------------------
    def lane_index(self, threads, stride, hint=None):
        return 'lane'

    def op(self, name, type_, *args, hint=None):
        return (name, args)

    def if_(self, cond, attrs=()):
        # `cond` is the tree `_guard` built; the bounds are read back off it
        # rather than passed, so a change to how the guard is composed shows
        # up here as a failure instead of being silently mis-recorded.
        lo, hi = _bounds(cond)
        return _Guard(self, lo, hi)

    def for_(self, lo, hi, step, unroll=False, hint=None):
        return _Loop(lo, hi)

    def const(self, value):
        return value

    # -- what the nest reported --------------------------------------------
    def inner(self, indices):
        idx = indices[0]
        lo, hi = _active(self._guards)
        if isinstance(idx, LeadIndex):
            # `_nonlead` is the *rendered* slot -- `write` passes
            # `str(loop.induction)` for a counted loop and keeps the object in
            # `_value`.  Recording the string would make every multi-slot nest
            # look like one slot at an unusable index, so the object wins
            # where there is one.
            slot = (idx._value if isinstance(idx._value, _Loop._Induction)
                    else idx._nonlead)
            self.blocks.append((slot, lo, hi, idx._width, idx._offset))
        else:
            self.blocks.append(('peel', idx))

    def elements(self):
        """Every element the nest writes, with its multiplicity.

        A lane outside a guard writes nothing; a lane inside one writes
        `width` adjacent elements at its slot.  A peeled leftover is a fixed
        element and one lane holds it, so it is written once -- `StoreRegToGlb`
        guards the whole body to that lane, which is also what removes the
        cross-lane read of a value the writing lane already has.

        This counted `threads` before, and that was not a modelling choice: it
        was the behaviour.  The peeled write had no guard at all, so the wave
        stored the element -- right under `=`, and `threads` times the
        contribution under `+=`.
        """
        seen = Counter()
        for block in self.blocks:
            if block[0] == 'peel':
                seen[block[1]] += 1
                continue
            slot, lo, hi, width, offset = block
            slots = (range(slot.lo, slot.hi)
                     if isinstance(slot, _Loop._Induction) else [slot])
            for s in slots:
                for lane in range(lo, hi):
                    base = width * (lane + s * self.threads) + offset
                    for c in range(width):
                        seen[base + c] += 1
        return seen


def _bounds(cond):
    """`(lo, hi)` from the comparison tree `LeadLoop._guard` builds."""
    lo, hi = None, None
    stack = [cond]
    while stack:
        node = stack.pop()
        if not isinstance(node, tuple):
            continue
        name, args = node
        if name == 'and':
            stack.extend(args)
        elif name == 'ge':
            lo = args[1]
        elif name == 'lt':
            hi = args[1]
    return lo, hi


def _active(guards):
    """The lane range the innermost guards leave open."""
    lo, hi = 0, None
    for g_lo, g_hi in guards:
        if g_lo is not None:
            lo = max(lo, g_lo)
        if g_hi is not None:
            hi = g_hi if hi is None else min(hi, g_hi)
    return lo, hi


def cover(start, end, threads, width=1):
    """What `LeadLoop(start, end, threads, width)` writes, per element."""
    rec = Recorder(threads)
    LeadLoop('i0', start, end, threads, 1, width=width).write(None, rec,
                                                              rec.inner)
    # An open upper guard means every remaining lane, which is the wave.
    rec.blocks = [b if b[0] == 'peel' or b[2] is not None
                  else (b[0], b[1], rec.threads, b[3], b[4])
                  for b in rec.blocks]
    return rec.elements(), rec


#: Extents chosen for the shapes that break, not for roundness.  35 is an
#: order-4 basis-function count; 48 is 32 plus a half-wave tail, which is the
#: alternating full/half pattern a store nest shows in the generated source;
#: 20 is an order-3 count; 56 and 72 straddle two slots with a ragged end.
EXTENTS = [1, 3, 8, 12, 16, 17, 20, 31, 32, 33, 34, 35, 47, 48, 49, 56, 64,
           65, 72, 96]
THREADS = [8, 16, 32, 64]


@pytest.mark.parametrize('width', [1, 2, 4])
@pytest.mark.parametrize('threads', THREADS)
@pytest.mark.parametrize('end', EXTENTS)
def test_the_nest_covers_its_whole_range(end, threads, width):
    """Coverage: nothing in `[0, end)` is left undefined.

    The weaker of the two properties and the one a plain store rests on.  A
    gap here would be a wrong kernel on every backend and for every operator,
    so it is checked first and separately -- a failure of exactness below is a
    much narrower claim than a failure here.
    """
    seen, _ = cover(0, end, threads, width)
    missing = [e for e in range(end) if seen.get(e, 0) == 0]
    assert not missing, f'never written: {missing[:8]}'


@pytest.mark.parametrize('threads', THREADS)
@pytest.mark.parametrize('end', EXTENTS)
def test_a_scalar_nest_writes_each_element_exactly_once(end, threads):
    """Exactness at `width == 1`, which is what atomics are gated to.

    Every block is a whole slot or a lane-guarded partial one, and the guards
    partition the lanes rather than overlapping them -- so the alternating
    full-slot and half-slot writes a 48-element extent produces are 32 lanes
    and then 16 different lanes, not 32 and then 32 again.
    """
    seen, _ = cover(0, end, threads, 1)
    twice = {e: n for e, n in seen.items() if n > 1}
    assert not twice, f'written more than once: {sorted(twice.items())[:8]}'
    assert not [e for e in seen if e >= end], 'wrote past the end'


@pytest.mark.parametrize('threads', THREADS)
@pytest.mark.parametrize('end', EXTENTS)
def test_a_scalar_nest_stays_inside_its_range(end, threads):
    seen, _ = cover(0, end, threads, 1)
    outside = [e for e in seen if e < 0 or e >= end]
    assert not outside, f'outside [0, {end}): {sorted(outside)[:8]}'


@pytest.mark.parametrize('width', [1, 2, 4])
@pytest.mark.parametrize('threads', THREADS)
@pytest.mark.parametrize('end', EXTENTS)
def test_every_width_writes_each_element_exactly_once(end, threads, width):
    """Exactness at every width, which is new.

    It held at width 1 and failed at 2 and 4 for one reason: the peeled tail
    was written by the whole wave.  With that write guarded to the lane that
    owns the element, the nest partitions its range at every width -- so the
    condition `placement.atomic_write_is_exact` was carrying is no longer
    about the nest.
    """
    seen, _ = cover(0, end, threads, width)
    twice = {e: n for e, n in seen.items() if n > 1}
    assert not twice, f'written more than once: {sorted(twice.items())[:8]}'
    assert not [e for e in seen if e >= end], 'wrote past the end'
