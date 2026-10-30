# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Products that add into one output, summed in the matrix instruction's
accumulator layout (`Options.mma_d_layout`, `compute.dchain`).

The output is read into the accumulators of the D layout once, before the
first product, and written once, after the last, each lane at the cells
its registers hold; no epilogue tile.  On the host oracle, which runs
`mma.sync` from PTX's tables (`test_mma_oracle.py`): SeisSol's volume at
order 6, and the same output summed from two products that read their
second operand from memory -- with windows of different lengths, with one
product whose operator is not zero where its window ends, and with the
output read between the two.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import re
import warnings
from pathlib import Path

import pytest

from tensorforge.common.basic_types import Datatype

ROOT = Path(__file__).resolve().parent.parent

#: Normwise, by precision: 3xTF32 and FP64 against NumPy; and a chain
#: against the same build without it, where only the order of the sums
#: differs.
TOLERANCE = {'s': 1e-6, 'd': 1e-12}
SAME = 1e-12

CHAIN = {'tensor_cores': True, 'mma_d_layout': True}
PLAIN = {'tensor_cores': True}


def _host():
    path = ROOT / 'tools' / 'host' / 'corpus_check.py'
    spec = importlib.util.spec_from_file_location('dchain_corpus_check', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


HOST = _host()


def _volume(precision):
    import seissol_suite as fx
    system = 'elastic-linearck'
    return fx.description(system, f'{system}-o6-{precision}', 'gpu_volume')


def _two_products(precision, windows, between=False):
    """`Q += kDivM(0) I` and `Q += kDivM(1) I`, the volume's operators read
    over row windows `windows` against the element's `I`: two products of a
    batch-constant operator into one output, nine columns, the second
    operand from memory.  `between` reads `Q` between the two -- the second
    product multiplies `Q star(0)` instead of `I`."""
    d = json.loads(json.dumps(_volume(precision)))
    ops = [op for op in d['operations'] if op['result']['name'] == 'Q'][:2]
    reads_q = {'type': 'multilinear',
               'result': {'name': '_tmp0', 'indices': ['l', 'p'],
                          'bbox': [[0, 0], [35, 9]], 'offset': [0, 0],
                          'offset_from': None, 'sliced': False},
               'args': [{'name': 'Q', 'indices': ['l', 'q'],
                         'bbox': [[0, 0], [35, 9]], 'offset': [0, 0],
                         'offset_from': None, 'sliced': False},
                        {'name': 'star(0)', 'indices': ['q', 'p'],
                         'bbox': [[0, 0], [9, 9]], 'offset': [0, 0],
                         'offset_from': None, 'sliced': False}],
               'condition': [], 'permute': [[0, 1], [0, 1]],
               'target': [[0, -1], [-1, 1]],
               'linear': {'alpha': ops[0]['linear']['alpha'], 'add': False}}
    for n, (op, (lo, hi)) in enumerate(zip(ops, windows)):
        op['result']['bbox'][0][0], op['result']['bbox'][1][0] = lo, hi
        op['args'][0]['bbox'][0][0], op['args'][0]['bbox'][1][0] = lo, hi
        op['args'][1]['name'] = '_tmp0' if between and n == 1 else 'I'
    d['operations'] = [ops[0]] + ([reads_q] if between else []) + [ops[1]]
    return d


def _kernel(description, precision, arch='sm_90'):
    dtype = Datatype.F64 if precision == 'd' else Datatype.F32
    return HOST.Kernel(description, dtype, arch)


def _normwise(got, want):
    """The largest deviation per output, against its largest magnitude --
    infinite where a cell is not a number."""
    out = {}
    for alias in {alias for alias, _ in want}:
        keys = [key for key in want if key[0] == alias]
        scale = max(abs(want[key]) for key in keys) or 1.0
        if any(got[key] != got[key] for key in keys):
            out[alias] = float('inf')
            continue
        out[alias] = max(abs(got[key] - want[key]) for key in keys) / scale
    return out


def _run(kernel, options):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        got, _ = kernel.run(options)
    return got


def _source(description, precision, arch='sm_90', chains=None, **options):
    """The kernel a build emits, and the chains `dchain.link` made, as
    lists of member counts."""
    return _build(description, precision, arch, chains, **options).get_kernel()


def _build(description, precision, arch='sm_90', chains=None, **options):
    """The generator of a build (`_source`)."""
    from tensorforge.backend.instructions.compute import dchain
    from tensorforge.common.context import Context, Options
    from tensorforge.generators.generator import Generator
    kernel = _kernel(description, precision, arch)
    made = [] if chains is None else chains
    real = dchain.link

    def spy(context, instrs):
        out = real(context, instrs)
        seen = {id(x._dchain): x._dchain for x in instrs
                if getattr(x, '_dchain', None) is not None}
        made[:] = [len(c.members) for c in seen.values()]
        return out
    dchain.link = spy
    try:
        gen = Generator(kernel.descrs(), Context(
            arch=arch, backend='cuda', fp_type=kernel._dtype,
            options=Options(**options)))
        with warnings.catch_warnings(), \
                contextlib.redirect_stdout(io.StringIO()):
            warnings.simplefilter('ignore')
            gen.generate()
    finally:
        dchain.link = real
    return gen


def _output(src, alias):
    """The kernel's name for the tensor `alias` (`m<i>`), from its banner."""
    m = re.search(r'"alias":"' + re.escape(alias) + r'"[^}]*"name":"(m\d+)"',
                  src)
    assert m, alias
    return m.group(1)


@pytest.mark.parametrize('precision', ['s', 'd'])
def test_the_volume_reads_and_writes_its_output_once(precision):
    """No epilogue tile, and every access of `Q` outside the products:
    the old value read ahead of the first `mma.sync`, the sum written
    behind the last."""
    chains = []
    src = _source(_volume(precision), precision, chains=chains, **CHAIN)
    assert chains == [3]
    assert 'ctile' not in src
    q = _output(src, 'Q')
    body = src[src.index('mma.sync'):src.rindex('mma.sync')]
    assert not re.search(r'\bglb_' + q + r'\[', body)
    plain = _source(_volume(precision), precision, **PLAIN)
    assert 'ctile' in plain


@pytest.mark.parametrize('arch', ['sm_86', 'sm_90'])
@pytest.mark.parametrize('precision', ['s', 'd'])
def test_the_chained_volume_computes_on_the_oracle(arch, precision):
    kernel = _kernel(_volume(precision), precision, arch)
    want = kernel.expected()
    chained = _run(kernel, CHAIN)
    for alias, error in _normwise(chained, want).items():
        assert error <= TOLERANCE[precision], (alias, error)
    for alias, error in _normwise(chained, _run(kernel, PLAIN)).items():
        assert error <= SAME, (alias, error)


@pytest.mark.parametrize('precision', ['s', 'd'])
@pytest.mark.parametrize('windows', [((1, 54), (1, 55)), ((1, 56), (1, 55)),
                                     ((1, 55), (1, 56))])
def test_two_products_of_different_windows_chain(precision, windows):
    """One chain over the union of the windows: each product adds the
    zeros its operator holds in the rows the other owns.  `kDivM(0)` is
    zero from row 54 on, `kDivM(1)` from row 55 (the volume's windows)."""
    description = _two_products(precision, windows)
    chains = []
    _source(description, precision, chains=chains, **CHAIN)
    assert chains == [2]
    kernel = _kernel(description, precision)
    got = _run(kernel, CHAIN)
    for alias, error in _normwise(got, kernel.expected()).items():
        assert error <= TOLERANCE[precision], (alias, error)


@pytest.mark.parametrize('first', [(1, 40), (2, 49)])
def test_a_window_narrower_than_its_operator_does_not_join(first):
    """`kDivM(0)` read over fewer rows than it has numbers in: the rows of
    the second window it does not own would take what it holds there.  The
    products stay apart, each a chain of its own."""
    description = _two_products('d', (first, (1, 56)))
    chains = []
    _source(description, 'd', chains=chains, **CHAIN)
    assert sorted(chains) == [1, 1]
    kernel = _kernel(description, 'd')
    got = _run(kernel, CHAIN)
    for alias, error in _normwise(got, kernel.expected()).items():
        assert error <= TOLERANCE['d'], (alias, error)


def test_a_read_of_the_output_between_ends_the_chain():
    """`Q` read by a product between the two: the first writes it before
    the read, and the second starts a chain of its own after it."""
    description = _two_products('d', ((1, 54), (1, 56)), between=True)
    chains = []
    _source(description, 'd', chains=chains, **CHAIN)
    assert sorted(chains) == [1, 1]
    kernel = _kernel(description, 'd')
    got = _run(kernel, CHAIN)
    for alias, error in _normwise(got, kernel.expected()).items():
        assert error <= TOLERANCE['d'], (alias, error)


@pytest.mark.parametrize('precision', ['s', 'd'])
def test_the_chained_volume_orders_its_shared_memory(precision):
    """Over two elements, so that the next element's transfers are in flight
    while this one reads: no access to shared memory that no barrier or wait
    orders (`kernel_eval.Races`).  The arena is laid out anew without the
    epilogue tile, and an operator's transfer lands right behind `B`."""
    from tensorforge.reference import kernel_eval as ke
    gen = _build(_volume(precision), precision, **CHAIN)
    lanes, mults = ke.launch_geometry(gen.get_launcher())
    races = []
    ke.evaluate_wave(gen.get_kernel(), lanes, seed=3, globals_only=True,
                     mults=mults, block=True, races=races, elements=2)
    assert not races, [str(r) for r in races[:3]]


def test_no_chain_without_the_option():
    chains = []
    src = _source(_volume('s'), 's', chains=chains, **PLAIN)
    assert chains == [] and 'ctile' in src
