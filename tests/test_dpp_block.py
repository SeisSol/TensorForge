# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Runs of fused DPP FMAs, written as one block of inline assembly on gfx9
(`amd.dpp_block`).

`fmacdpp{4,16}<row>` is inline assembly the hazard recognizer does not look
into, and on gfx9 its broadcast source wants two wait states after the VALU
write of it -- a reload from an AGPR right in front made SeisSol's F64
derivative wrong on gfx942.  A block waits once in front: nothing can be
scheduled into it.  And behind itself where an MFMA can read what it wrote:
the MFMA wants two wait states after the write, which the recognizer does not
see either -- SeisSol's elastic F32 derivative with the products reassociated
came out wrong on MI300A and MI210 without it.  Pinned here: what is a block
and what is not, how its operands are numbered, where a run has to break and
where it goes on, where a block waits (`Options.amd_dpp_waits`), that the
kernel's own figures count the products as before, and that targets without
the wait keep their calls.
"""
import contextlib
import io
import json
import re
import warnings
from pathlib import Path

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


def _volume(arch, datatype=Datatype.F64, **options):
    """`Q += kDivM_d (I star_d)`, dense, in double precision unless said
    otherwise: the fused chain over a 56 x 56 operator, three times.  In
    single precision on CDNA the operator products take the lane-batched
    MFMA tile, with its tails in the chain."""
    def t(shape, alias, addressing=Addressing.STRIDED):
        return SubTensor(Tensor(list(shape), addressing,
                                BoundingBox([0, 0], list(shape)),
                                alias=alias, datatype=datatype))
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
                                    fp_type=datatype,
                                    options=Options(**options)))
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter('ignore')
        gen.generate()
    return gen


def _lines(template):
    return re.findall(r'"([^"]*)\\n\\t"', template)


def test_a_block_counts_the_flops_of_its_calls(monkeypatch):
    """A run written as one block does the arithmetic its calls do, and the
    executed flop count (`analysis.pipeline.executed_flops`) says so: the
    same with blocks as with the calls one by one.  A block once counted its
    code and mix but not its flops, which left SeisSol's F64 kernels on CDNA
    at a two-hundredth of their arithmetic in paper-1106's rates.  F64:
    in F32 the operator products take the MFMA tile, which counts none."""
    from tensorforge.analysis.pipeline import executed_flops
    blocks = _volume('gfx942')
    assert BLOCK.findall(blocks.get_kernel())
    monkeypatch.setattr(dpp_block, 'ENABLED', False)
    calls = _volume('gfx942')
    assert not BLOCK.findall(calls.get_kernel())
    assert executed_flops(calls) is not None
    assert executed_flops(blocks) == executed_flops(calls)


def test_always_waits_on_both_sides_of_every_block():
    """`always`, as before: on gfx942 every fused product is in a block, each
    block waits once in front and once behind and takes at most `MAX_RUN`
    products."""
    src = _volume('gfx942', amd_dpp_waits='always').get_kernel()
    blocks = BLOCK.findall(src)
    assert blocks
    assert 'tensorforge::fmacdpp' not in src
    for template, _, _ in blocks:
        lines = _lines(template)
        assert lines[0] == dpp_block.WAIT
        assert lines[-1] == dpp_block.WAIT_AFTER
        assert 0 < len(lines) - 2 <= dpp_block.MAX_RUN
        assert all(l.startswith('v_fmac_f64_dpp ') for l in lines[1:-1])


def test_without_a_matrix_instruction_a_block_waits_in_front_only():
    """`needed`, the default: the kernel issues no MFMA, and every other
    reader of what a block wrote the compiler waits for by itself -- so the
    blocks wait in front and nowhere else, and still hold every product."""
    src = _volume('gfx942').get_kernel()
    assert '__builtin_amdgcn_mfma' not in src
    blocks = BLOCK.findall(src)
    assert blocks
    assert 'tensorforge::fmacdpp' not in src
    for template, _, _ in blocks:
        lines = _lines(template)
        assert lines[0] == dpp_block.WAIT
        assert all(l.startswith('v_fmac_f64_dpp ') for l in lines[1:])


def test_with_a_matrix_instruction_every_block_waits_behind_itself():
    """In single precision the operator products are MFMAs, and an MFMA may
    come to stand right behind any block of the kernel: each waits behind
    itself, as under `always`."""
    src = _volume('gfx942', Datatype.F32).get_kernel()
    assert '__builtin_amdgcn_mfma' in src
    blocks = BLOCK.findall(src)
    assert blocks
    for template, _, _ in blocks:
        lines = _lines(template)
        assert lines[0] == dpp_block.WAIT
        assert lines[-1] == dpp_block.WAIT_AFTER


def test_a_body_written_alone_waits_behind_its_blocks():
    """Whether a block waits behind itself is asked of the whole kernel; an
    emitter that is not told the kernel's bodies does not know, and waits."""
    from tensorforge.backend.pir.emit import Emitter
    context = Context(arch='gfx942', backend='hip', fp_type=Datatype.F64)
    emitter = Emitter(None, context)
    emitter._decide_dpp_waits()
    assert (emitter._dpp_waits, emitter._dpp_after) == ('needed', True)
    context.kernel_bodies = ()
    emitter._decide_dpp_waits()
    assert (emitter._dpp_waits, emitter._dpp_after) == ('needed', False)


def test_without_the_wait_to_keep_the_calls_stay():
    """gfx1150 keeps no such wait (LLVM: `hasNoDataDepHazard`), and the
    compiler may schedule its calls one by one."""
    assert not BLOCK.search(_volume('gfx1150').get_kernel())


@pytest.mark.parametrize('waits', dpp_block.WAITS)
def test_the_figures_count_the_products_as_calls(monkeypatch, waits):
    """What the kernel issues is what it issued as calls: the block changes
    the spelling, not the products -- and adds one unit of code a wait in
    front of a product."""
    blocked = _volume('gfx942', amd_dpp_waits=waits).metrics
    monkeypatch.setattr(dpp_block, 'ENABLED', False)
    calls = _volume('gfx942', amd_dpp_waits=waits).metrics
    assert blocked.issue_mix == calls.issue_mix
    assert blocked.emitted_work == calls.emitted_work
    assert blocked.code_units > calls.code_units


def _fma(b, acc, src, mul, row, step=16):
    return b.call_stmt(f'tensorforge::fmacdpp{step}<{row}>', acc, src, mul,
                       writes=(acc,))


def _hint(v):
    return v.hint if hasattr(v, 'hint') else str(v)


def test_operands_are_numbered_outputs_first_and_once():
    """Each accumulator is one read-write operand however often the run adds
    into it, each source one input, and outputs come first: the assembler
    numbers both in one sequence."""
    b = IRBuilder(Datatype.F64)
    f64 = ScalarType(Datatype.F64)
    c0, c1 = b.declare(f64, hint='c'), b.declare(f64, hint='c')
    a, x, y = (b.declare(f64, hint=h) for h in 'axy')
    run = [_fma(b, c0, a, x, 0), _fma(b, c1, a, y, 0), _fma(b, c0, a, y, 5)]
    out = dpp_block.render(run, _hint)
    lines = _lines(out)
    assert lines == [dpp_block.WAIT,
                     'v_fmac_f64_dpp %0, %2, %3 row_newbcast:0 row_mask:0xf '
                     'bank_mask:0xf bound_ctrl:1',
                     'v_fmac_f64_dpp %1, %2, %4 row_newbcast:0 row_mask:0xf '
                     'bank_mask:0xf bound_ctrl:1',
                     'v_fmac_f64_dpp %0, %2, %4 row_newbcast:5 row_mask:0xf '
                     'bank_mask:0xf bound_ctrl:1',
                     dpp_block.WAIT_AFTER]
    assert out.count('"+&v"(') == 2 and out.count('"v"(') == 3
    assert _lines(dpp_block.render(run, _hint, after=False)) == lines[:-1]


def test_accumulators_are_early_clobbered():
    """A block writes an accumulator before its later products read their
    inputs, so no input may share an accumulator's register: without `&` a
    guarded load and an accumulator just zeroed shared one on gfx942, and
    `localFluxAll` came out wrong."""
    for waits in dpp_block.WAITS:
        src = _volume('gfx942', amd_dpp_waits=waits).get_kernel()
        for template, outs, _ in BLOCK.findall(src):
            assert outs.count('"+&v"(') == outs.count('(')


def test_a_broadcast_of_what_the_run_wrote_starts_the_next_block():
    """`always`: a product that broadcasts an accumulator the run has written
    would read it right behind the write, inside the block: it starts the
    next one, behind a wait of its own."""
    b = IRBuilder(Datatype.F64)
    f64 = ScalarType(Datatype.F64)
    c, d, a, x = (b.declare(f64, hint=h) for h in 'cdax')
    body = (_fma(b, c, a, x, 0), _fma(b, c, a, x, 1), _fma(b, d, c, x, 2))
    assert dpp_block.runs(body, 0) == 2
    assert dpp_block.runs(body, 0, 'always') == 2
    assert dpp_block.runs(body, 2) == 3


@pytest.mark.parametrize('between, wait', [(0, 's_nop 1'), (1, 's_nop 0'),
                                           (2, None)])
def test_inside_a_block_a_broadcast_of_its_write_waits_what_it_needs(
        between, wait):
    """`needed`: the broadcast of an accumulator the run wrote stays in the
    block, and waits there -- two wait states after the write, counted from
    the products between -- only as long as the write is too close."""
    b = IRBuilder(Datatype.F64)
    f64 = ScalarType(Datatype.F64)
    c, d, e, a, x = (b.declare(f64, hint=h) for h in 'cdeax')
    body = ((_fma(b, c, a, x, 0),)
            + tuple(_fma(b, e, a, x, 1 + k) for k in range(between))
            + (_fma(b, d, c, x, 7),))
    assert dpp_block.runs(body, 0, 'needed') == len(body)
    lines = _lines(dpp_block.render(body, _hint, after=False))
    assert lines[0] == dpp_block.WAIT
    expect = 1 + len(body) + (wait is not None)
    assert len(lines) == expect
    if wait is not None:
        assert lines[-2] == wait
    assert dpp_block.waits_in(body) == 1 + (wait is not None)


def test_a_run_goes_on_past_max_run_while_its_operands_fit():
    """`needed` with `amd_dpp_block_operands`: a run that adds into the same
    accumulators from the same sources is one block however long, while one
    whose sources keep coming stops at the first product past `MAX_RUN` that
    takes it over the cap.  Without a cap a run stops at `MAX_RUN`, as
    under `always`."""
    b = IRBuilder(Datatype.F64)
    f64 = ScalarType(Datatype.F64)
    c0, c1, a = (b.declare(f64, hint=h) for h in ('c0', 'c1', 'a'))
    xs = [b.declare(f64, hint=f'x{k}') for k in range(16)]
    shared = tuple(_fma(b, (c0, c1)[k % 2], a, xs[k % 16], k % 16)
                   for k in range(3 * dpp_block.MAX_RUN))
    assert dpp_block.runs(shared, 0) == dpp_block.MAX_RUN
    assert dpp_block.runs(shared, 0, 'needed') == dpp_block.MAX_RUN
    assert dpp_block.runs(shared, 0, 'needed', 19) == len(shared)
    assert dpp_block.runs(shared, 0, 'needed', 18) == dpp_block.MAX_RUN
    fresh = [b.declare(f64, hint=f'y{k}') for k in range(64)]
    growing = tuple(_fma(b, c0, a, fresh[k], k % 16) for k in range(64))
    # The accumulator, the broadcast source, and one more source a product.
    assert dpp_block.runs(growing, 0, 'needed', 44) == 42


def test_the_operand_cap_takes_runs_past_max_run_in_a_kernel():
    """In the kernel, a cap lets the runs of the 56 x 56 operator go on: at
    16 lanes a row of it is longer than `MAX_RUN`.  Fewer blocks, the same
    products."""
    def blocks(**options):
        src = _volume('gfx942', lanes_per_mult=16, **options).get_kernel()
        return [_lines(t) for t, _, _ in BLOCK.findall(src)]
    cut, merged = blocks(), blocks(amd_dpp_block_operands=44)
    assert len(merged) < len(cut)
    assert sum(map(len, merged)) - len(merged) == sum(map(len, cut)) - len(cut)
    assert max(len(l) - 1 for l in merged) > dpp_block.MAX_RUN


@pytest.mark.parametrize('model, blocks', [('gfx942', True), ('gfx90a', True),
                                           ('gfx950', True), ('gfx908', False),
                                           ('gfx1150', False),
                                           ('gfx1250', False)])
def test_blocks_are_written_from_cdna2_and_on_gfx9_only(model, blocks):
    assert dpp_block.blocks_for(model) is blocks


def test_the_device_headers_issue_no_matrix_instruction_of_their_own():
    """A block waits behind itself where the kernel's bodies issue a matrix
    instruction (`dpp_block.waits_behind`), which asks the bodies: an MFMA
    inside a helper of the device headers would be one they do not show."""
    from tensorforge import toolchain
    include = Path(toolchain.include_dir())
    for header in include.rglob('*.h'):
        code = re.sub(r'//[^\n]*|/\*.*?\*/', '', header.read_text(), flags=re.S)
        assert not re.search(r'__builtin_amdgcn_(mfma|smfmac|wmma)|v_mfma'
                             r'|v_smfmac|v_wmma', code), header


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
    transpose -- ends in a wait on gfx9, as a block where the kernel issues a
    matrix instruction does: `hip.h` cannot tell whether it does.  hipcc's
    listing has an `s_nop` right behind the last DPP instruction of each."""
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


HAZARD_SNIPPET = r"""
#include <hip/hip_runtime.h>
#define WRITE(y, x) asm("v_mul_f32 %0, 0x40400000, %1" : "=v"(y) : "v"(x))
extern "C" __global__ void k_dpp(float *o) {
  float x = o[threadIdx.x], y;
  WRITE(y, x);
  o[threadIdx.x] = __builtin_bit_cast(float, __builtin_amdgcn_update_dpp(
      0, __builtin_bit_cast(int, y), 0x153, 0xf, 0xf, true));
}
extern "C" __global__ void k_readlane(float *o, int *r) {
  float x = o[threadIdx.x], y;
  WRITE(y, x);
  r[threadIdx.x] = __builtin_amdgcn_readlane(__builtin_bit_cast(int, y), 5)
                   + threadIdx.x;
}
"""


@pytest.mark.parametrize('arch, lane_read', [('gfx90a', 0), ('gfx942', 1),
                                             ('gfx950', 1)])
def test_the_compiler_waits_behind_assembly_for_what_reads_it(arch, lane_read,
                                                             tmp_path):
    """What `needed` rests on, asked of the hipcc at hand: behind a VGPR
    written in inline assembly, the hazard recognizer waits in front of a DPP
    instruction that reads it (two wait states) and of a lane read (one, on
    gfx94x and later) as it does behind a VALU instruction -- it counts any
    writer there.  Only an MFMA it waits for behind a VALU instruction alone,
    which is why a block waits behind itself in a kernel that issues one."""
    import subprocess
    from harness import syntax
    hipcc = syntax.device_compiler('hip')
    if hipcc is None:
        pytest.skip('hipcc not found')
    src = tmp_path / 'hazard.hip'
    src.write_text(HAZARD_SNIPPET)
    run = subprocess.run([hipcc, '-O3', '-std=c++17', f'--offload-arch={arch}',
                          '--cuda-device-only', '-S', '-o', '-', str(src)],
                         capture_output=True, text=True, timeout=600)
    if run.returncode and 'offload-arch' in run.stderr:
        pytest.skip(f'hipcc does not know {arch}')
    assert run.returncode == 0, run.stderr[-2000:]

    def behind_write(kernel):
        """The wait states between the assembly's write and its reader."""
        text = run.stdout.split(f'{kernel}:', 1)[1].split('s_endpgm', 1)[0]
        body = [l.strip() for l in text.splitlines()
                if l.startswith('\t') and not l.strip().startswith(('.', ';'))]
        start = next(i for i, l in enumerate(body) if l.startswith('v_mul_f32'))
        states = 0
        for line in body[start + 1:]:
            if '_dpp' in line or line.startswith('v_readlane'):
                return states
            states += (int(line.split()[1], 0) + 1
                       if line.startswith('s_nop') else 1)
        raise AssertionError(f'{kernel}: no reader')

    assert behind_write('k_dpp') >= 2
    assert behind_write('k_readlane') >= lane_read
