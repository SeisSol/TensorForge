# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""The block's shared buffers sharing bytes by occupancy
(`Options.share_block_buffers`, `pir.allocate`).

A block that runs phases one after another holds buffers dead before others
are written: the multi-element layout's chained sections, an order's buffer
dead once the next order read it.  Laid out back to back, as the block's
arena always was, they add up; by occupancy, as the multiplication's arena
is, they share bytes.  The block's threads reach a statement at different
times, so the barrier placement behind the allocator fences a write over the
bytes another buffer was read from -- which is why the option is taken only
where that placement runs.

The shape here is a relay of products whose intermediates live one step
each: ``T1 = kDivM(0) I``, ``T2 = T1 star(0)``, ``T3 = kDivM(1) T2``,
``Q += T3 star(1)`` -- ``T1`` dead before ``T3`` is written.  The
multi-element layout's group loop is IR once the asynchronous staging lands
(its `BatchLoop._gen_grouped`); until then it is text, whose statements say
nothing of what they touch and so keep every buffer of the trip occupied, and
the tests that need the trip's buffers to share make the loop IR as that
change does (`_ir_group_loop`).
"""

from __future__ import annotations

import contextlib
import copy
import io
import json
import re
import warnings

import pytest

from tensorforge.common.basic_types import Datatype
from tensorforge.common.context import Context
from tensorforge.common.options import Options, registry
from tensorforge.generators import element_group as eg

SYSTEM = 'elastic-linearck'

#: The oracle's seed for its fill of memory (`kernel_eval.Slot`).
SEED = 29


def _relay(order=6, precision='s'):
    """The relay: the volume's tensors, three temporaries as tall as `Q`."""
    import seissol_suite as fx
    d = json.loads(json.dumps(fx.description(
        SYSTEM, f'{SYSTEM}-o{order}-{precision}', 'gpu_volume')))
    tensors = {t['name']: t for t in d['tensors']}
    ops = [op for op in d['operations'] if op['result']['name'] == 'Q']
    one = ops[0]['linear']['alpha']
    r0a, r1a = ops[0]['result']['bbox'][0][0], ops[0]['result']['bbox'][1][0]
    r0b, r1b = ops[1]['result']['bbox'][0][0], ops[1]['result']['bbox'][1][0]
    depth = ops[0]['args'][0]['bbox'][1][1]
    # The second operator reads T2's rows from 1, which the first wrote.
    inner = min(depth, r1a)
    for name in ('_relay1', '_relay2', '_relay3'):
        tmp = copy.deepcopy(tensors['_tmp0'])
        tmp['name'] = name
        tmp['storage'] = copy.deepcopy(tensors['Q']['storage'])
        d['tensors'].append(tmp)

    def view(name, indices, lo, hi):
        return {'name': name, 'indices': indices, 'bbox': [lo, hi],
                'offset': [0, 0], 'offset_from': None, 'sliced': False}

    def product(result, first, second, add):
        return {'type': 'multilinear', 'result': result,
                'args': [first, second], 'condition': [],
                'permute': [[0, 1], [0, 1]], 'target': [[0, -1], [-1, 1]],
                'linear': {'alpha': one, 'add': [0, 1] if add else False}}
    d['operations'] = [
        product(view('_relay1', ['k', 'q'], [r0a, 0], [r1a, 9]),
                view('kDivM(0)', ['k', 'l'], [r0a, 0], [r1a, depth]),
                view('I', ['l', 'q'], [0, 0], [depth, 9]), False),
        product(view('_relay2', ['k', 'p'], [r0a, 0], [r1a, 9]),
                view('_relay1', ['k', 'q'], [r0a, 0], [r1a, 9]),
                view('star(0)', ['q', 'p'], [0, 0], [9, 9]), False),
        product(view('_relay3', ['k', 'q'], [r0b, 0], [r1b, 9]),
                view('kDivM(1)', ['k', 'l'], [r0b, 1], [r1b, inner]),
                view('_relay2', ['l', 'q'], [1, 0], [inner, 9]), False),
        product(view('Q', ['k', 'p'], [r0b, 0], [r1b, 9]),
                view('_relay3', ['k', 'q'], [r0b, 0], [r1b, 9]),
                view('star(1)', ['q', 'p'], [0, 0], [9, 9]), True)]
    return d


def _descrs(precision='s'):
    from tensorforge.frontend.yateto import DescriptionReader
    return DescriptionReader(None, {}).read(_relay(6, precision))[0]


def _generate(precision='s', **options):
    from tensorforge.generators.generator import Generator
    gen = Generator(_descrs(precision), Context(
        arch='sm_90', backend='cuda',
        fp_type=Datatype.F64 if precision == 'd' else Datatype.F32,
        options=Options(multi_element='force', **options)))
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter('ignore')
        gen.generate()
    return gen


def _windows(kernel):
    """The group's buffers' offsets in the block's arena, by hint."""
    return {name: int(at) for name, at in re.findall(
        r'\w+?_(g_?(?:relay\d|I|star\d))\s*=\s*&totalShrMem\[(\d+)\]',
        kernel)}


@pytest.fixture
def _ir_group_loop(monkeypatch):
    """The multi-element layout's group loop as IR, as the asynchronous
    staging makes it (`BatchLoop._gen_grouped`)."""
    from tensorforge.backend.instructions import batch_loop as bl
    text = bl.BatchLoop._gen_grouped

    def grouped(self, writer):
        if (self._trip is bl.Trip.ELEMENT_GROUP
                and self._mode is bl.LoopMode.PERSISTENT
                and hasattr(writer, 'for_')):
            return self._gen_grouped_ir(writer)
        return text(self, writer)
    monkeypatch.setattr(bl.BatchLoop, '_gen_grouped', grouped)


def test_the_option_is_off_by_default():
    assert registry()['share_block_buffers'].default is False


def test_off_the_block_s_buffers_are_back_to_back(_ir_group_loop):
    """Off, and on without the barrier placement that would fence the shared
    bytes: every buffer of the group its own bytes, in allocation order."""
    off = _windows(_generate(tensor_cores=False).get_kernel())
    assert len(set(off.values())) == len(off) == 6, off
    unfenced = _windows(_generate(tensor_cores=False, share_block_buffers=True,
                                  enable_sync_block_opt=False).get_kernel())
    assert unfenced == off


@pytest.mark.parametrize('precision', ['s', 'd'])
def test_a_buffer_dead_before_another_is_written_shares_its_bytes(
        precision, _ir_group_loop):
    """`T1` is dead once `T2` is written from it, before `T3` is written:
    the two share bytes, and so does the staged `I` with `T2`."""
    on = _windows(_generate(precision, tensor_cores=False,
                            share_block_buffers=True).get_kernel())
    assert on['g_relay1'] == on['g_relay3'], on
    off = _windows(_generate(precision, tensor_cores=False).get_kernel())
    assert max(on.values()) < max(off.values())


@pytest.mark.parametrize('precision,options', [
    ('s', {'tensor_cores': False}), ('d', {'tensor_cores': False}),
    ('s', {}), ('d', {})],
    ids=['fma-f32', 'fma-f64', 'mma_sync', 'dmma'])
def test_shared_bytes_compute_every_element_on_the_oracle(
        precision, options, monkeypatch, _ir_group_loop):
    """Two trips of a group of four over seven elements on the host oracle,
    the second partial, with the block's buffers sharing bytes: every
    element's output against NumPy, and no access to shared memory that no
    barrier orders (`kernel_eval.Races`) -- a write over the bytes a dead
    buffer was read from among them."""
    import numpy as np
    from tensorforge.common.basic_types import Addressing
    from tensorforge.reference import kernel_eval
    group, count = 4, 7
    monkeypatch.setattr(eg, 'ELEMENTS', {key: group for key in eg.ELEMENTS})
    monkeypatch.setattr(eg, 'DEFAULT_ELEMENTS', group)
    gen = _generate(precision, share_block_buffers=True, **options)
    assert gen.built._element_group.elements == group
    kernel = gen.get_kernel()
    windows = _windows(kernel)
    assert len(set(windows.values())) < len(windows), windows
    tensors = {}
    for outer in gen.descr_list:
        for d in outer.operations():
            for m in d.matrix_list():
                t = getattr(m, 'tensor', m)
                tensors.setdefault(t.alias, t)

    def box(t):
        return [(int(lo), int(hi)) for lo, hi in zip(t.bbox.lower(),
                                                      t.bbox.upper())]

    def memory(t, read):
        (r0, r1), (c0, c1) = box(t)
        out = np.zeros((count, r1 - r0, c1 - c0))
        for e in range(count):
            for c in range(c1 - c0):
                for r in range(r1 - r0):
                    out[e, r, c] = read(e, r + c * (r1 - r0))
        return out

    fill = kernel_eval.Slot(SEED)
    preset, inputs = {}, {}
    for alias, t in tensors.items():
        if t.is_tmp:
            continue
        if t.addressing.kind is Addressing.NONE:
            preset[t.name] = [float(v) for v in t.storage_values()]
            (r0, r1), (c0, c1) = box(t)
            inputs[alias] = np.asarray(t.get_values())[None, r0:r1, c0:c1]
        else:
            inputs[alias] = memory(t, lambda e, s, t=t: fill.read(
                f'{t.name}@{e}', s))
    output = tensors['Q']
    lanes, mults = kernel_eval.launch_geometry(gen.get_launcher())
    races = []
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        mem = kernel_eval.evaluate_wave(
            kernel, lanes, seed=SEED, globals_only=True, mults=mults,
            block=True, preset=preset, elements=count, races=races)
    assert not races, races[:3]
    got = memory(output, lambda e, s: mem.get((f'{output.name}@{e}', s),
                                              fill.read(f'{output.name}@{e}',
                                                        s)))
    want = _reference(gen.descr_list, inputs, output)
    tolerance = 1e-12 if precision == 'd' else 1e-6
    for e in range(count):
        scale = np.abs(want[e]).max() or 1.0
        error = np.abs(got[e] - want[e]).max() / scale
        assert error <= tolerance, (e, error)


def _reference(descrs, inputs, output):
    """NumPy over the description, every element at once, in double
    precision: each product into its window, an intermediate zero outside
    what is written; the arrays are the tensors' boxes."""
    import numpy as np
    full = {}

    def whole(tensor, data):
        lo = tuple(int(x) for x in tensor.bbox.lower())
        hi = tuple(int(x) for x in tensor.bbox.upper())
        data = np.asarray(data, dtype=np.float64)
        out = np.zeros((data.shape[0],) + tuple(hi))
        out[:, lo[0]:hi[0], lo[1]:hi[1]] = data
        return out

    def array(tensor):
        if tensor.alias not in full:
            full[tensor.alias] = whole(tensor, inputs[tensor.alias])
        return full[tensor.alias]

    full[output.alias] = whole(output, inputs[output.alias])
    for descr in descrs:
        for op in descr.operations():
            a, b = op.ops
            (r0, k0), (r1, k1) = a.bbox.lower(), a.bbox.upper()
            c0, c1 = b.bbox.lower()[1], b.bbox.upper()[1]
            value = np.einsum('...ik,...kj->...ij',
                              array(a.tensor)[:, r0:r1, k0:k1],
                              array(b.tensor)[:, k0:k1, c0:c1])
            dest = op.dest.tensor
            if dest.is_tmp and not op.add:
                full[dest.alias] = np.zeros(
                    (value.shape[0],) + tuple(int(x) for x in
                                              dest.bbox.upper()))
            if op.add:
                full[dest.alias][:, r0:r1, c0:c1] += value
            else:
                full[dest.alias][:, r0:r1, c0:c1] = value
    lo = tuple(int(x) for x in output.bbox.lower())
    hi = tuple(int(x) for x in output.bbox.upper())
    return full[output.alias][:, lo[0]:hi[0], lo[1]:hi[1]]
