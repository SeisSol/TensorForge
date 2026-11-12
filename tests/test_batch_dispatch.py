# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""`Options.batch_dispatch`: a second kernel for the batches the device holds
at once, picked by the shortest chain among the lanes rule's candidates
(`tuning.small_variant`), and a launcher that hands those batches to it.

The compiler and the counts are stood in as in `test_lanes_rule`; the chain of
a candidate is read from a table the same way.  The operands are not prepared
(`Options.prepare_operands`) but where the storage is the point: prepared, the
operators of `_local_flux` are interleaved for the lanes that read them, and no
two lane counts store them alike.
"""

from __future__ import annotations

import re

from tensorforge.generators import tuning
from tensorforge.generators.generator import Generator

from test_lanes_rule import (TableScorer, _compiler_asked_throughout,  # noqa: F401
                             _context, _counting, _forget_picks, _key,
                             _local_flux)

#: The large batches' ranking: sixteen lanes unmerged, as in
#: `test_lanes_rule.test_the_generator_builds_the_pick_and_remembers_it`.
COUNTS = {(32, None): 1000, (32, True): 1000, (32, False): 900,
          (16, True): 950, (16, False): 700, (8, True): 800,
          (8, False): 500, (64, True): 1500, (64, False): 1400}


def _stand_in(monkeypatch, chains):
    """The counts above, the chains from `chains` by (lanes, merge), and a
    compiler that spills 300 bytes at eight lanes unmerged."""
    monkeypatch.setattr(tuning, 'executed_instructions', _counting(COUNTS))
    monkeypatch.setattr(tuning, 'chain_cycles',
                        lambda result: chains.get(_key(result.candidate)))
    scorer = TableScorer({(8, False): 300})
    monkeypatch.setattr(tuning.CompiledScore, 'available',
                        lambda self, context: True)
    monkeypatch.setattr(tuning.CompiledScore, 'resources',
                        lambda self, result: scorer.resources(result))
    return scorer


def test_small_batches_take_the_shortest_chain(monkeypatch):
    """Whatever the large batches' pick: the count ranks sixteen lanes
    first, the chain 64."""
    chains = {key: 1000.0 for key in COUNTS}
    chains[(32, False)] = 600.0
    chains[(64, False)] = 400.0
    _stand_in(monkeypatch, chains)
    found = {}
    gen = Generator(_local_flux()(), _context(autotune='lanes',
                                              prepare_operands=False))
    pick = tuning.autotune(lambda: _local_flux()(), gen._context, mode='lanes',
                           generate=Generator.trial, dispatch=found)
    assert _key(pick) == (16, False)
    assert _key(found['small']) == (64, False)


def test_the_small_kernel_stores_the_operands_as_the_pick_does(monkeypatch):
    """Prepared, the operators are interleaved for sixteen lanes: 64 would
    read them in another order than the host lays them out."""
    chains = {key: 1000.0 for key in COUNTS}
    chains[(64, False)] = 100.0
    _stand_in(monkeypatch, chains)
    found = {}
    gen = Generator(_local_flux()(), _context(autotune='lanes'))
    tuning.autotune(lambda: _local_flux()(), gen._context, mode='lanes',
                    generate=Generator.trial, dispatch=found)
    assert found['small'] is None or found['small'].lanes.num_threads == 16


def test_a_chain_within_the_margin_builds_one_kernel(monkeypatch):
    chains = {key: 1000.0 for key in COUNTS}
    chains[(64, False)] = 0.97 * 1000.0
    _stand_in(monkeypatch, chains)
    found = {}
    gen = Generator(_local_flux()(), _context(autotune='lanes',
                                              prepare_operands=False))
    tuning.autotune(lambda: _local_flux()(), gen._context, mode='lanes',
                    generate=Generator.trial, dispatch=found)
    assert found['small'] is None


def _dispatching(monkeypatch, **opts):
    chains = {key: 1000.0 for key in COUNTS}
    chains[(64, False)] = 100.0
    scorer = _stand_in(monkeypatch, chains)
    gen = Generator(_local_flux()(),
                    _context(autotune='lanes', batch_dispatch=1.0,
                             prepare_operands=False, **opts))
    gen.generate()
    return gen, scorer


def test_the_launcher_hands_small_batches_to_the_second_kernel(monkeypatch):
    gen, _ = _dispatching(monkeypatch)
    built = gen.built
    assert built._small is not None and _key(built._small_pick[0]) == (64, False)
    name = gen.get_base_name()
    assert Generator.NAME_PLACEHOLDER not in name
    kernel, launcher, header = (gen.get_kernel(), gen.get_launcher(),
                                gen.get_header())
    # Both kernels, the small one under the large one's name.
    assert f'kernel_{name}(' in kernel and f'kernel_{name}_small(' in kernel
    # The launcher and the launch it reports both hand small batches over, at
    # what the first kernel holds resident.
    below = f'dispatch_below_{name}()'
    assert f'static std::size_t {below}' in launcher
    assert re.search(rf'if \(numElements0 < {re.escape(below)}\) \{{\s*'
                     rf'launcher_{name}_small\(', launcher)
    assert re.search(rf'if \(numElements0 < {re.escape(below)}\) \{{\s*'
                     rf'return launch_config_{name}_small\(', launcher)
    # The first literal block is the large kernel's: sixteen lanes.
    first = re.search(r'\bblock\s*\(\s*(\d+)\s*,', launcher)
    assert first is not None and int(first.group(1)) == 16
    # The header is the large kernel's alone.
    assert '_small' not in header
    assert gen.kernel_info()['dispatch']['small'].startswith('64w1')


def test_a_second_generation_asks_nothing_and_dispatches_again(monkeypatch):
    first, scorer = _dispatching(monkeypatch)
    asked = len(scorer.asked)
    again = Generator(_local_flux()(),
                      _context(autotune='lanes', batch_dispatch=1.0,
                               prepare_operands=False))
    again.generate()
    assert len(scorer.asked) == asked
    assert again.built._small is not None
    assert again.get_kernel() == first.get_kernel()


def test_without_the_option_one_kernel(monkeypatch):
    _stand_in(monkeypatch, {key: 100.0 for key in COUNTS})
    gen = Generator(_local_flux()(), _context(autotune='lanes'))
    gen.generate()
    assert gen.built._small is None
    assert '_small' not in gen.get_kernel() + gen.get_launcher()


def test_the_small_kernel_issues_nothing_ahead(monkeypatch):
    """Below the switch a lane group has one element at most: nothing to move
    ahead or fetch for the next one (`tuning.SMALL_BATCH_OPTIONS`)."""
    gen, _ = _dispatching(monkeypatch)
    large = gen.built._context.get_user_options()
    small = gen.built._small._context.get_user_options()
    assert large.enable_wrap_loads and large.enable_move_loads
    assert not small.enable_wrap_loads and not small.enable_move_loads


def test_what_the_caller_asked_stands_in_the_small_kernel(monkeypatch):
    gen, _ = _dispatching(monkeypatch, enable_wrap_loads=True)
    small = gen.built._small._context.get_user_options()
    assert small.enable_wrap_loads and not small.enable_move_loads
