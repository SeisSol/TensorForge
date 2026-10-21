# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""A loop over groups of rows, issuing its transfers one group ahead.

Where the whole wave issues something in the body -- a matrix fragment
product -- and a multiplication is narrower than the wave, the batch loop is
driven a group of rows at a time (`BatchLoop._gen_grouped_ir`): each row's
element is bound from the group's index, the row where it has one and the
group's own element where the mask holds it back.  `wrap` takes the loop as
it takes a batch loop: the transfer for the next group from the group's index
one stride on, the peel from the first group the loop names, each row's
element computed from it the way the body computes it.

The host oracle checks that the buffer holds the right element: the kernel
computes what it computes without the transform, under a mask that holds a
row back on one trip and not the next -- the row for which the element one
stride on is another row's.  The oracle runs no matrix fragment product, so
the group is forced on a kernel of plain products.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import re
import warnings
from pathlib import Path

import pytest

from tensorforge.common.context import Context, Options
from tensorforge.generators.generator import Generator
from tensorforge.reference import kernel_eval as ke

CASES = Path(__file__).parent / 'cases'


def _build(stem: str, monkeypatch, **options):
    """`stem` on sm_86 at sixteen lanes, half a wave, driven two rows at a
    time as a wave-wide instruction in its body would have it."""
    monkeypatch.setattr(Generator, '_needs_wave_group', lambda self: True)
    path = next(CASES.rglob(f'{stem}.py'))
    spec = importlib.util.spec_from_file_location('tf_group__' + stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ctx = Context(arch='sm_86', backend='cuda',
                  fp_type=getattr(mod, 'DTYPE', None),
                  options=Options(lanes_per_mult=16, **options))
    gen = Generator(mod.descr_list(), ctx, attrs=getattr(mod, 'ATTRS', None))
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter('ignore')
        gen.generate()
    src = gen.get_kernel()
    assert re.search(r'for \(size_t \w*batchIdGroup0\b', src), (
        'expected a loop over groups of rows')
    return src, ke.launch_geometry(gen.get_launcher())


def _mask(mults: int):
    """A flag per element of three trips: row 1 held back on its first trip
    and not on its second, row 2 -- the leader of its group -- the other way
    round."""
    flags = [1] * (3 * mults)
    flags[1] = 0
    flags[2 + mults] = 0
    return flags


def _written(src: str, geometry, races=None, masked=True) -> dict:
    """What the kernel leaves in the operands it writes, over three trips of
    one block."""
    lanes, mults = geometry
    signature = re.search(r'\bkernel_\w+\(([^)]*)\)', src).group(1)
    outputs = {p.split()[-1] for p in signature.split(',')
               if '*' in p and not p.strip().startswith('const')
               and '<const ' not in p}
    mem = ke.evaluate_wave(src, lanes, seed=5, globals_only=True,
                           mults=mults, block=True, races=races,
                           elements=3 * mults,
                           preset={'flags0': _mask(mults)} if masked else None)
    written = {k: v for k, v in mem.items() if k[0] in outputs}
    assert written, 'nothing written to compare'
    return written


@pytest.mark.parametrize('stem', ['square_notrans', 'aligned_operands'])
def test_a_group_computes_what_it_computes_unwrapped(stem, monkeypatch):
    """A load into registers and a copy into shared memory, each one group
    ahead -- and no access the hardware would not have ordered."""
    plain, at = _build(stem, monkeypatch, enable_wrap_loads=False)
    wrapped, wrapped_at = _build(stem, monkeypatch)
    assert 'peel_' not in plain
    assert 'peel_glb_' in wrapped and 'wrap_glb_' in wrapped
    assert re.search(r'__pipeline_memcpy_async\([^;]*\bwrap_glb_', wrapped)
    assert at == wrapped_at
    races = []
    assert _written(wrapped, at, races) == _written(plain, at)
    assert not races, races


def test_the_mask_holds_rows_back(monkeypatch):
    """The oracle hands the kernel the mask: without it every element is
    written, with it two fewer."""
    plain, at = _build('square_notrans', monkeypatch, enable_wrap_loads=False)
    whole = _written(plain, at, masked=False)
    held = _written(plain, at)
    assert set(held) < set(whole)
    per_element = len(whole) // (3 * at[1])
    assert len(whole) - len(held) == 2 * per_element


def test_the_first_group_is_clamped_into_the_batch(monkeypatch):
    """The peel runs where the loop does not, for a group whose start is
    past the end: it fetches for group 0 there, as a row's peel fetches for
    element 0."""
    src, _ = _build('square_notrans', monkeypatch)
    first = re.search(r'(\w+_batchIdGroup0First) = \((.+) < numElements0\) '
                      r'\? (.+) : 0u?;', src)
    assert first and first.group(2) == first.group(3)
    assert re.search(rf'\w+_row = {first.group(1)} \+ \w+_batchIdLane0;',
                     src)


def test_the_next_group_is_held_at_the_last(monkeypatch):
    src, _ = _build('square_notrans', monkeypatch)
    assert re.search(r'\w+_groupNext = \((\w+_groupNextAhead) < numElements0\)'
                     r' \? \1 : \w+_batchIdGroup0;', src)


def test_a_row_of_the_next_group_is_read_under_its_own_flag(monkeypatch):
    """The element the next group's row is on, bound as the body binds it:
    the row where its flag holds, the group's own element where not."""
    src, _ = _build('square_notrans', monkeypatch)
    active = re.search(r'const bool wrap_batchIdActive0 = (\w+_row) < '
                       r'numElements0 && \(flags0 == nullptr \|\| '
                       r'static_cast<bool>\(flags0\[\1\]\)\);', src)
    assert active
    # The element, named where more than one binding reads it.
    select = rf'wrap_batchIdActive0 \? {active.group(1)} : \w+_groupNext'
    named = re.search(rf'(\w+_batchId0) = {select};', src)
    element = named.group(1) if named else rf'\(?\({select}\)'
    assert re.search(rf'wrap_glb_\w+ = &\w+\[\(?{element}', src)


def test_a_group_follows_no_pointer_under_a_flag_of_its_own(monkeypatch):
    """A row loop follows the next element's own pointer under that
    element's flag.  A group's rows take the group's element where theirs is
    held back, and the body follows that pointer unguarded: the moved
    transfer follows the pointer the body would, and no flag of the group's
    index decides it."""
    src, _ = _build('addressing_ptr_based', monkeypatch)
    assert 'peel_glb_' in src and 'wrap_glb_' in src
    assert 'allowed_next' not in src and 'allowed_peel' not in src
