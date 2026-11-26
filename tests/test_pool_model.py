# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""The mixer pool's protocol (`pool_model`), every interleaving at a small
size: as built, no deadlock and nothing out of order; each of the rules it
lives by dropped, what goes wrong.  A size that runs in a second here, the
larger ones marked slow."""

from __future__ import annotations

import pytest

import pool_model

#: The size the suite runs: two quarters of a band warp and an epilogue
#: warp each, three trips of four blocks, A's ring of two slots.
SMALL = dict(quarters=2, bands=1, trips=3, blocks=4, aslots=2)


def _check(**kw):
    return pool_model.violations(pool_model.Model(**{**SMALL, **kw}))


def test_the_pool_as_built():
    """Every state the pool reaches goes on or is done; no wait passes on
    the parity of a phase two short; no draw before its counter's reset and
    no reset racing a draw -- some thirteen thousand states."""
    found = _check()
    assert found['deadlocks'] == 0 and found['errors'] == {}
    assert found['states'] > 10_000


def test_no_reset():
    """The counters never reset: the trip two on draws past the blocks at
    once, its blocks never filled -- a draw before the reset, a deadlock."""
    found = _check(reset='none')
    assert found['deadlocks'] > 0
    assert found['errors'].get('draw before reset', 0) > 0


def test_the_reset_ahead_of_the_barrier():
    """The counters reset by the first mixer at its arrival, not behind the
    barrier: draws of the trip still to come race it."""
    found = _check(reset='early')
    assert found['deadlocks'] > 0
    assert found['errors'].get('reset races a draw', 0) > 0


def test_no_drain():
    """The epilogue warps never drain D: the trip two on waits for D's slot
    forever."""
    found = _check(drain='none')
    assert found['deadlocks'] > 0


def test_the_drain_ahead_of_the_barrier_is_safe():
    """D(t) drained behind an epilogue warp's last item, ahead of the
    barrier: no deadlock -- it only holds the other mixers at trip t's
    barrier, where behind it they would mix trip t + 1, which is why the
    drain sits behind the barrier."""
    found = _check(drain='ahead')
    assert found['deadlocks'] == 0 and found['errors'] == {}


def test_fewer_a_slots_than_a_quarter_s_mixers():
    """A's ring of one slot for a quarter's two mixers: a mixer's wait for
    a slot passes on the parity of a phase two short -- the refusal's
    reason (`element_group.pool_refusal`)."""
    found = _check(aslots=1)
    assert found['errors'].get('A empty alias', 0) > 0
    assert found['deadlocks'] > 0


@pytest.mark.slow
@pytest.mark.parametrize('size', [
    dict(quarters=2, bands=1, trips=4, blocks=3, aslots=2),
    dict(quarters=2, bands=2, trips=3, blocks=3, aslots=3),
    dict(quarters=2, bands=2, trips=2, blocks=5, aslots=3),
    dict(quarters=2, bands=1, trips=5, blocks=2, aslots=2)])
def test_the_pool_as_built_at_larger_sizes(size):
    """The pool as built at sizes up to some two hundred thousand states: two
    band warps a quarter, five blocks, five trips."""
    found = pool_model.violations(pool_model.Model(**size))
    assert found['deadlocks'] == 0 and found['errors'] == {}


@pytest.mark.slow
def test_fewer_a_slots_at_two_bands():
    """Two band warps and an epilogue warp a quarter on A's two slots, five
    blocks a trip: aliasing waits and deadlocks."""
    found = pool_model.violations(pool_model.Model(
        quarters=2, bands=2, trips=2, blocks=5, aslots=2))
    assert found['errors'].get('A empty alias', 0) > 0
    assert found['deadlocks'] > 0
