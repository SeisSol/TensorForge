# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""A staged transfer under the explicit vector, checked as coverage.

The staging loader builds its addresses as *text*, in the macro layer, and so
never passes the emitter that refuses a lane index.  Everything the emitter
guarantees about the explicit-vector lowering therefore stops at this one
path, and what came out of it was an SPMD address: `4 * item.get_local_id(0)`,
which is a work-group coordinate used as a lane number.

Two properties are worth testing rather than one, because only the second
catches what actually went wrong.  That no lane index survives is a syntactic
check and finds the term.  That the tile is covered exactly once is the
semantic one -- with sixteen work-items in the x extent, each writing a full
vector at `4 * its own id`, every element of the tile was written by several
parties and most of it by none.  A kernel like that compiles, runs, and is
wrong, which is the failure mode this file exists for.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import re
import warnings
from pathlib import Path

import pytest

from tensorforge.common.basic_types import Datatype
from tensorforge.common.context import Context, Options
from tensorforge.common.vm.vm import vm_factory
from tensorforge.generators.generator import Generator

CASES = Path(__file__).parent / 'cases'


def _generate(name, backend='esimd', **options):
    path = next(CASES.rglob(f'{name}.py'))
    spec = importlib.util.spec_from_file_location(name, path)
    case = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(case)
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        # The register-budget warning is about the operator's size, not about
        # the transfer, and it fires for most of the corpus at order 6.
        warnings.simplefilter('ignore')
        gen = Generator(case.descr_list(),
                        Context(arch='pvc', backend=backend,
                                fp_type=getattr(case, 'DTYPE', Datatype.F32),
                                options=Options(**options)))
        gen.generate()
    return gen


#: Every case in the corpus, by name, paired with both answers to the option
#: this file is about.  Cases the ESIMD lowering cannot take at all are
#: skipped where they raise -- they fail for reasons of their own and are not
#: this file's subject.
ALL_CASES = sorted({p.stem for p in CASES.rglob('*.py')
                    if '__pycache__' not in str(p)})


# --------------------------------------------------------------------------
# there is no lane index
# --------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.parametrize('name', ALL_CASES)
@pytest.mark.parametrize('preload', [False, True])
def test_no_work_group_coordinate_is_used_as_a_lane(name, preload):
    """`item.get_local_id(0)` must not appear in an ESIMD kernel at all.

    Not "should mostly not": under this lowering one work-item *is* the
    vector, so the x coordinate is not a lane number and there is no correct
    use for it.  `EsimdEmitter._thread_idx('x')` says exactly that and raises;
    this is the same rule applied to the paths that reach the output without
    passing the emitter -- the staging loader, and `Symbol.load_linear` /
    `store_linear`, all three of which formatted the index into a string.
    """
    try:
        src = _generate(name, preload_globals=preload).get_kernel()
    except Exception as exc:  # noqa: BLE001 -- see the docstring above
        pytest.skip(f'{name} does not lower to ESIMD: {type(exc).__name__}')
    assert 'get_local_id(0)' not in src


# --------------------------------------------------------------------------
# the block holds one work-item per wave, not one per lane
# --------------------------------------------------------------------------

def test_the_x_extent_is_one_work_item():
    """The lane count moved into the type, so it must leave the launch.

    A block of `(num_threads, mults, 1)` launches the whole multiplication
    `num_threads` times over under this lowering, each copy writing the full
    width of every store.
    """
    block = re.search(r'sycl::range<3> block \(([^)]*)\)',
                      _generate('square_notrans').get_launcher()).group(1)
    assert [t.strip() for t in block.split(',')][0] == '1'


def test_the_spmd_sycl_block_is_unchanged():
    """The same question, asked of a lowering where a work-item is a lane."""
    block = re.search(r'sycl::range<3> block \(([^)]*)\)',
                      _generate('square_notrans',
                                backend='oneapi').get_launcher()).group(1)
    assert [t.strip() for t in block.split(',')][0] == '16'


def test_a_wave_spanning_multiplication_is_still_one_work_item():
    """32 lanes on a 16-wide unit is two registers, not two work-items.

    The wave-spanning split exists to make several work-items cooperate on
    the lanes of one multiplication.  Under this lowering there is only ever
    the one, and the lane count is the width of the values' type -- so a
    multiplication that spans waves must not widen the x extent, and must not
    be compensated for by multiplying the y extent either.
    """
    gen = _generate('local_flux')
    block = re.search(r'sycl::range<3> block \(([^)]*)\)',
                      gen.get_launcher()).group(1)
    extents = [t.strip() for t in block.split(',')]
    assert gen._num_threads > vm_factory(
        'pvc', 'esimd', 'float').get_hw_descr().vec_unit_length, (
        'this case no longer spans waves, so it no longer tests the split')
    assert extents[0] == '1'
    assert {int(w) for w in re.findall(r'simd<\w+, (\d+)>', gen.get_kernel())} \
        & {gen._num_threads}, 'the lane count should appear as a vector width'


def test_the_spmd_split_is_untouched():
    """The same case on the lowering the split was written for."""
    block = re.search(r'sycl::range<3> block \(([^)]*)\)',
                      _generate('local_flux',
                                backend='oneapi').get_launcher()).group(1)
    assert [t.strip() for t in block.split(',')][0] != '1'


# --------------------------------------------------------------------------
# the tile is covered exactly once
# --------------------------------------------------------------------------

#: A staged write, with its width in the call.  The preload fill moved into
#: the PIR upstream, so the whole transfer goes out as one of these rather
#: than as a hand-built temporary -- the width is the vector's, which is what
#: this needs, and there is nothing to correlate across lines.
_SLM_STORE = re.compile(
    r'tensorforge::slmStore<\w+, (\d+)>\(\s*(\w+) \+ \((.*?)\),')
_GUARD = re.compile(r'if \(item\.get_local_id\(1\) == (\d+)\) \{')


#: Where the section prologue ends and the batch loop begins.
_BATCH_LOOP = re.compile(r'for \(size_t \w+_batchId0')


def _fill_writes(src, mults):
    """Which elements of each staged tile the prologue writes, and by whom.

    Reads the emitted fill back as one `(tile, element)` pair per element per
    work-item, so that a double write is a duplicate in this list rather than
    something a reader has to spot.

    The prologue only.  Inside the batch loop every work-item stages its own
    multiplication into its own window, so the same offsets appear once per
    work-item there and are meant to -- counting those as writes to one tile
    is counting `mults` private tiles as if they were shared.
    """
    lines = src.splitlines()
    for i, raw in enumerate(lines):
        if _BATCH_LOOP.search(raw):
            lines = lines[:i]
            break
    guard, writes = [], []
    for raw in lines:
        line = raw.strip()
        if _GUARD.match(line):
            guard.append(int(_GUARD.match(line).group(1)))
            continue
        if line == '}' and guard:
            guard.pop()
            continue
        store = _SLM_STORE.search(line)
        if store:
            width, name = int(store.group(1)), store.group(2)
            expr = store.group(3).replace('item.get_local_id(1)', 'y')
            for y in (guard[-1:] or range(mults)):
                base = eval(expr, {'__builtins__': {}}, {'y': y})  # noqa: S307
                writes += [(name, base + k) for k in range(width)]
    return writes


@pytest.mark.parametrize('name,tiles', [
    ('addressing_none', {'glb_m1': 256}),
    ('local_flux', None),
])
def test_the_preloaded_tile_is_written_exactly_once(name, tiles):
    """Every element of a staged operand, by exactly one work-item.

    A gap leaves the tile holding whatever was there before, and an overlap
    means two work-items wrote the same word -- with the same value, here, so
    the result is not even reliably wrong.  Both are invisible in the emitted
    text and both were present.
    """
    gen = _generate(name, preload_globals=True)
    mults = int(re.search(r'sycl::range<3> block \(\s*\d+\s*,\s*(\d+)',
                          gen.get_launcher()).group(1))
    writes = _fill_writes(gen.get_kernel(), mults)
    assert writes, 'no staged fill found; the case no longer preloads'

    by_tile = {}
    for tile, element in writes:
        by_tile.setdefault(tile, []).append(element)

    for tile, elements in by_tile.items():
        assert len(elements) == len(set(elements)), (
            f'{tile}: {len(elements) - len(set(elements))} elements written '
            f'more than once')
        assert sorted(elements) == list(range(len(elements))), (
            f'{tile}: the written elements are not the tile -- a gap, or a '
            f'run that does not start at zero')
        if tiles is not None:
            assert len(elements) == tiles[tile]


def test_the_tail_is_a_width_and_not_a_lane_guard():
    """`if (linear_idx < rest)` is a statement about lanes, and there are none.

    Under this lowering the condition is a scalar, so the branch is taken
    whole and the transfer inside moves the register's full width -- past the
    end of the tile.  What the guard meant is that the transfer is `rest`
    elements wide, which is a compile-time width here.
    """
    src = _generate('local_flux', preload_globals=True).get_kernel()
    assert ' < 64)' not in src and ' < 32)' not in src, (
        'a lane-count comparison survived into the fill')
    widths = {int(m.group(1)) for m in _SLM_STORE.finditer(src)}
    assert len(widths) > 1, 'the tail should be narrower than the body hops'
