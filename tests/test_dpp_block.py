# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Runs of fused DPP FMAs, written as one block of inline assembly on gfx9
(`amd.dpp_block`).

`fmacdpp{4,16}<row>` is inline assembly the hazard recognizer does not look
into, and on gfx9 its broadcast source wants two wait states after the VALU
write of it -- a reload from an AGPR right in front made SeisSol's F64
derivative wrong on gfx942.  A block waits once in front: nothing can be
scheduled into it.  And once behind: an MFMA reads what the block wrote two
wait states after the write at the earliest, which the recognizer does not
see either -- SeisSol's elastic F32 derivative with the products reassociated
came out wrong on MI300A and MI210 without it.  Pinned here: what is a block and what is not, how its
operands are numbered, where a run has to break, that the kernel's own
figures count the products as before, and that targets without the wait keep
their calls.
"""
import contextlib
import io
import json
import re
import warnings

import pytest

from tensorforge.backend.instructions.compute.primitives.amd import dpp_block
from tensorforge.backend.pir.build import IRBuilder
from tensorforge.backend.pir.core import ScalarType
from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.context import Context
from tensorforge.common.helper import generate_tmp_matrix
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.common.options import Options
from tensorforge.generators.descriptions import GemmDescr
from tensorforge.generators.generator import Generator

BLOCK = re.compile(r'asm\((.*?)\n\s*: (.*?)\n\s*: (.*?)\n\s*\);', re.S)


def _volume(arch, **options):
    """`Q += kDivM_d (I star_d)`, dense, in double precision: the fused
    chain over a 56 x 56 operator, three times."""
    def t(shape, alias, addressing=Addressing.STRIDED):
        return SubTensor(Tensor(list(shape), addressing,
                                BoundingBox([0, 0], list(shape)),
                                alias=alias, datatype=Datatype.F64))
    i, q = t((56, 9), 'I'), t((56, 9), 'Q')
    descrs = []
    for d in range(3):
        star = t((9, 9), f'star{d}')
        tmp = SubTensor(generate_tmp_matrix(i, star))
        descrs.append(GemmDescr(False, False, i, star, tmp))
        descrs.append(GemmDescr(False, False,
                                t((56, 56), f'kDivM{d}', Addressing.NONE),
                                tmp, q, alpha=1.0, beta=1.0))
    gen = Generator(descrs, Context(arch=arch, backend='hip',
                                    fp_type=Datatype.F64,
                                    options=Options(**options)))
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter('ignore')
        gen.generate()
    return gen


def test_a_run_waits_once_and_holds_no_call():
    """On gfx942 every fused product is in a block, each block waits once in
    front and takes at most `MAX_RUN` products."""
    src = _volume('gfx942').get_kernel()
    blocks = BLOCK.findall(src)
    assert blocks
    assert 'tensorforge::fmacdpp' not in src
    for template, _, _ in blocks:
        lines = re.findall(r'"([^"]*)\\n\\t"', template)
        assert lines[0] == dpp_block.WAIT
        assert lines[-1] == dpp_block.WAIT_AFTER
        assert 0 < len(lines) - 2 <= dpp_block.MAX_RUN
        assert all(l.startswith('v_fmac_f64_dpp ') for l in lines[1:-1])


def test_without_the_wait_to_keep_the_calls_stay():
    """gfx1150 keeps no such wait (LLVM: `hasNoDataDepHazard`), and the
    compiler may schedule its calls one by one."""
    assert not BLOCK.search(_volume('gfx1150').get_kernel())


def test_the_figures_count_the_products_as_calls(monkeypatch):
    """What the kernel issues is what it issued as calls: the block changes
    the spelling, not the products -- and adds one unit of code a block."""
    blocked = _volume('gfx942').metrics
    monkeypatch.setattr(dpp_block, 'ENABLED', False)
    calls = _volume('gfx942').metrics
    assert blocked.issue_mix == calls.issue_mix
    assert blocked.emitted_work == calls.emitted_work
    assert blocked.code_units > calls.code_units


def _fma(b, acc, src, mul, row, step=16):
    return b.call_stmt(f'tensorforge::fmacdpp{step}<{row}>', acc, src, mul,
                       writes=(acc,))


def test_operands_are_numbered_outputs_first_and_once():
    """Each accumulator is one read-write operand however often the run adds
    into it, each source one input, and outputs come first: the assembler
    numbers both in one sequence."""
    b = IRBuilder(Datatype.F64)
    f64 = ScalarType(Datatype.F64)
    c0, c1 = b.declare(f64, hint='c'), b.declare(f64, hint='c')
    a, x, y = (b.declare(f64, hint=h) for h in 'axy')
    run = [_fma(b, c0, a, x, 0), _fma(b, c1, a, y, 0), _fma(b, c0, a, y, 5)]
    out = dpp_block.render(run, lambda v: v.hint if hasattr(v, 'hint')
                           else str(v))
    lines = re.findall(r'"([^"]*)\\n\\t"', out)
    assert lines == [dpp_block.WAIT,
                     'v_fmac_f64_dpp %0, %2, %3 row_newbcast:0 row_mask:0xf '
                     'bank_mask:0xf bound_ctrl:1',
                     'v_fmac_f64_dpp %1, %2, %4 row_newbcast:0 row_mask:0xf '
                     'bank_mask:0xf bound_ctrl:1',
                     'v_fmac_f64_dpp %0, %2, %4 row_newbcast:5 row_mask:0xf '
                     'bank_mask:0xf bound_ctrl:1',
                     dpp_block.WAIT_AFTER]
    assert out.count('"+&v"(') == 2 and out.count('"v"(') == 3


def test_accumulators_are_early_clobbered():
    """A block writes an accumulator before its later products read their
    inputs, so no input may share an accumulator's register: without `&` a
    guarded load and an accumulator just zeroed shared one on gfx942, and
    `localFluxAll` came out wrong."""
    for template, outs, _ in BLOCK.findall(_volume('gfx942').get_kernel()):
        assert outs.count('"+&v"(') == outs.count('(')


def test_a_broadcast_of_what_the_run_wrote_starts_the_next_block():
    """A product that broadcasts an accumulator the run has written would
    read it right behind the write, inside the block: it starts the next
    one, behind a wait of its own."""
    b = IRBuilder(Datatype.F64)
    f64 = ScalarType(Datatype.F64)
    c, d, a, x = (b.declare(f64, hint=h) for h in 'cdax')
    body = (_fma(b, c, a, x, 0), _fma(b, c, a, x, 1), _fma(b, d, c, x, 2))
    assert dpp_block.runs(body, 0) == 2
    assert dpp_block.runs(body, 2) == 3


@pytest.mark.parametrize('model, blocks', [('gfx942', True), ('gfx90a', True),
                                           ('gfx950', True), ('gfx908', False),
                                           ('gfx1150', False),
                                           ('gfx1250', False)])
def test_blocks_are_written_from_cdna2_and_on_gfx9_only(model, blocks):
    assert dpp_block.blocks_for(model) is blocks


SNIPPET = r'''
#include "tensorforge_device/hip.h"
__global__ void k(float *o, double *p) {
  float c = o[threadIdx.x], a = o[threadIdx.x + 64], b = o[threadIdx.x + 128];
  tensorforge::fmacdpp16<3>(c, a, b);
  tensorforge::fmacdpp4<1>(c, a, b);
  double d = p[threadIdx.x];
  tensorforge::fmacdpp16<2>(d, p[threadIdx.x + 64], p[threadIdx.x + 128]);
  float w1, w2, w3, w4;
  tensorforge::transpose4x4b32Fused(w1, w2, w3, w4, c, a, b, c * a);
  o[threadIdx.x] = c + w1 + w2 * w3 * w4;
  p[threadIdx.x] = d;
}
'''


@pytest.mark.parametrize('arch', ['gfx90a', 'gfx942'])
def test_on_gfx9_every_dpp_asm_waits_behind_itself(arch, tmp_path):
    """`hip.h`'s own DPP assembly -- the single fused FMAs and the fused quad
    transpose -- ends in a wait on gfx9, as a block does: hipcc's listing has
    an `s_nop` right behind the last DPP instruction of each."""
    import subprocess
    from harness import syntax
    from tensorforge import toolchain
    hipcc = syntax.device_compiler('hip')
    if hipcc is None:
        pytest.skip('hipcc not found')
    src = tmp_path / 'dpp.hip'
    src.write_text(SNIPPET)
    run = subprocess.run([hipcc, '-O3', '-std=c++17', f'--offload-arch={arch}',
                          '-I', toolchain.include_dir(), '--cuda-device-only',
                          '-S', '-o', '-', str(src)],
                         capture_output=True, text=True, timeout=600)
    if run.returncode and 'offload-arch' in run.stderr:
        pytest.skip(f'hipcc does not know {arch}')
    assert run.returncode == 0, run.stderr[-2000:]
    body = [l.strip() for l in run.stdout.splitlines()
            if l.startswith('\t') and not l.strip().startswith(';')]
    # Wherever a run of DPP instructions ends -- the next one neither DPP nor
    # the mask move between the transpose's pairs -- a wait follows.
    ends = [i for i, line in enumerate(body[:-1]) if '_dpp' in line
            and '_dpp' not in body[i + 1]
            and not body[i + 1].startswith('s_mov_b64 vcc')]
    assert len(ends) >= 3
    for i in ends:
        assert body[i + 1].startswith('s_nop'), body[i - 2:i + 3]
