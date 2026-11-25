# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""NVIDIA's Blackwell data-center family in the descriptions: sm_100 (B200,
GB200), sm_103 (B300, GB300), sm_107 (Rubin) and sm_110 (Thor), each with
its arch-specific target (`sm_100a`) and its family-specific one
(`sm_100f`), and sm_121 (GB10) beside sm_120.

What is checked here: what each row of `hw_descr_db.yml` states -- tensor
memory, shared memory and sm_107's oversized opt-in, the bits of a matrix
descriptor's start address -- and that every row loads; that an `f` row is
its family's least member; that sm_101 is refused by its new name; which
targets have tcgen05 and register budgets, and how many columns a block
allocates; how nvcc is told each target, with ptxas's checks of tensor
memory switched off and on, and the test harness's cache keyed on them;
that the builder allocates no more columns than its target has; and, with
a local nvcc, that `cuda.h` takes each target for what its description says
and that ptxas builds the o4 volume tile for the family's new members as
for sm_100a.
"""

from __future__ import annotations

import functools
import re
import shutil
import subprocess
import types
from pathlib import Path

import pytest
import yaml

import test_multi_element_async as tma
from tensorforge import toolchain
from tensorforge.backend.pir.build import IRBuilder
from tensorforge.backend.pir.core import IRError
from tensorforge.common.basic_types import Datatype
from tensorforge.common.context import Context
from tensorforge.common.target import Target
from tensorforge.common.vm import hw_descr
from tensorforge.common.vm.hw_descr import hw_descr_factory

INCLUDE = Path(__file__).resolve().parents[1] / 'src' / 'tensorforge' / 'include'
KB = 1024

#: Every target of the family, its arch- and family-specific ones.
SPECIFIC = ('sm_100a', 'sm_100f', 'sm_103a', 'sm_103f', 'sm_107a',
            'sm_107f', 'sm_110a', 'sm_110f')

#: What each row states: the lanes and the columns of tensor memory, the
#: columns an exclusive allocation reaches, shared memory a block plans
#: against and what it may opt into, in kB, and the bits of a descriptor's
#: start address.
ROWS = {
    'sm_90': (None, None, None, 227, None, 14),
    'sm_90a': (None, None, None, 227, None, 14),
    'sm_100': (128, 512, None, 227, None, 14),
    'sm_100a': (128, 512, None, 227, None, 14),
    'sm_100f': (128, 512, None, 227, None, 14),
    'sm_103': (128, 512, None, 227, None, 14),
    'sm_103a': (128, 512, None, 227, None, 14),
    'sm_103f': (128, 512, None, 227, None, 14),
    'sm_107': (128, 512, 576, 227, 327, 15),
    'sm_107a': (128, 512, 576, 227, 327, 15),
    'sm_107f': (128, 512, 576, 227, 327, 15),
    'sm_110': (128, 512, None, 227, None, 14),
    'sm_110a': (128, 512, None, 227, None, 14),
    'sm_110f': (128, 512, None, 227, None, 14),
    'sm_120': (None, None, None, 99, None, None),
    'sm_121': (None, None, None, 99, None, None),
}


def _kb(value):
    return None if value is None else value // KB


# -- the descriptions -------------------------------------------------------- #

@pytest.mark.parametrize('arch', sorted(ROWS))
def test_each_row_states_its_memories(arch):
    hw = hw_descr_factory(arch, 'cuda')
    assert (hw.tmem_lanes, hw.tmem_columns, hw.tmem_exclusive_columns,
            _kb(hw.max_local_mem_size_per_block),
            _kb(hw.max_local_mem_size_per_block_oversized),
            hw.matrix_descriptor_address_bits) == ROWS[arch]


def _rows():
    path = Path(hw_descr.__file__).with_name('hw_descr_db.yml')
    return yaml.safe_load(path.read_text())


_BACKEND = {'nvidia': 'cuda', 'amd': 'hip', 'intel': 'oneapi',
            'generic': 'acpp'}


def test_every_row_of_the_table_loads():
    """A row's base read ahead of it, and every field a description needs
    there: a row whose base the table states later would take the table for
    its base (`process_arch`) and lose the fields."""
    rows = _rows()
    named = {row['arch']: row for row in rows}
    for row in rows:
        vendor = row.get('name')
        base = row
        while vendor is None:
            base = named[base['base']]
            vendor = base.get('name')
        hw = hw_descr_factory(row['arch'], _BACKEND[vendor])
        assert isinstance(hw.vec_unit_length, int), row['arch']
        assert hw.max_local_mem_size_per_block > 0, row['arch']


def _fields(arch):
    hw = hw_descr_factory(arch, 'cuda')
    return {k: v for k, v in vars(hw).items() if k not in ('model',)}


@pytest.mark.parametrize('model,part', [
    ('sm_100a', 'sm_100'), ('sm_100f', 'sm_100'), ('sm_103', 'sm_100'),
    ('sm_103a', 'sm_103'), ('sm_103f', 'sm_103'), ('sm_107a', 'sm_107'),
    ('sm_107f', 'sm_107'), ('sm_110a', 'sm_110'), ('sm_110f', 'sm_110'),
    ('sm_121', 'sm_120')])
def test_a_target_is_described_as_its_part(model, part):
    """An `a` row is its part.  An `f` row is what its family's least member
    has -- `compute_100f` runs on 10.0, 10.3 and 10.7, so sm_100f is sm_100,
    without sm_107's exclusive columns, oversized shared memory and wider
    descriptor.  B300 is B200 in everything described, and GB10 is
    sm_120."""
    assert _fields(model) == _fields(part)


@pytest.mark.parametrize('old,new', [('sm_101', 'sm_110'),
                                     ('sm_101a', 'sm_110a')])
def test_sm101_is_refused_by_its_new_name(old, new):
    """Thor is sm_110 from CUDA 13.0 on, and nvcc 13 builds no sm_101."""
    with pytest.raises(ValueError, match=rf'{old} is {new}\b'):
        hw_descr_factory(old, 'cuda')
    with pytest.raises(ValueError, match=rf'ask for {new}$'):
        Target(old, 'cuda')
    assert 'sm_101' not in toolchain.Nvcc.ARCH_SPECIFIC


def test_a_descriptor_holds_every_address_a_block_can_have():
    """The start address in 16-byte units, so `16 << bits` bytes: past the
    largest shared memory a block of each target with a matrix engine may
    be given -- the oversized opt-in where there is one -- and the 1 kB the
    system reserves ahead of it.  sm_107's opt-in needs its 15 bits."""
    for arch in ('sm_90a',) + SPECIFIC:
        hw = Target(arch, 'cuda').hw
        largest = (hw.max_local_mem_size_per_block_oversized
                   or hw.max_local_mem_size_per_block)
        assert largest + KB <= 16 << hw.matrix_descriptor_address_bits, arch
    oversized = hw_descr_factory('sm_107', 'cuda')
    assert oversized.max_local_mem_size_per_block_oversized > 16 << 14


# -- the targets ------------------------------------------------------------- #

NOT_TCGEN05 = [('sm_80', 'cuda'), ('sm_90', 'cuda'), ('sm_90a', 'cuda'),
               ('sm_100', 'cuda'), ('sm_103', 'cuda'), ('sm_107', 'cuda'),
               ('sm_110', 'cuda'), ('sm_120', 'cuda'), ('sm_121', 'cuda'),
               ('sm_100a', 'hip'), ('gfx942', 'hip'), ('pvc', 'oneapi')]


@pytest.mark.parametrize('arch', SPECIFIC)
def test_the_familys_specific_targets_have_tcgen05(arch):
    target = Target(arch, 'cuda')
    assert target.tcgen05() and target.register_budgets()
    assert target.arch_specific() != target.family_specific()
    assert not target.wgmma()


@pytest.mark.parametrize('arch,backend', NOT_TCGEN05)
def test_the_others_have_none(arch, backend):
    """A plain target's PTX has no tcgen05, sm_90a has wgmma in its place,
    the consumer parts have no tensor memory, and HIP spells none."""
    target = Target(arch, backend)
    assert not target.tcgen05()
    assert target.tmem_columns() is None
    assert target.tmem_columns(exclusive=True) is None


@pytest.mark.parametrize('arch,backend,budgets', [
    ('sm_90a', 'cuda', True), ('sm_100f', 'cuda', True),
    ('sm_107f', 'cuda', True), ('sm_90', 'cuda', False),
    ('sm_107', 'cuda', False), ('sm_100f', 'hip', False)])
def test_register_budgets_on_the_arch_and_family_specific_targets(
        arch, backend, budgets):
    """`setmaxnreg`: sm_90a's, and every `a` and `f` target of Blackwell's
    (CCCL's list; ptxas 13.5 takes it for sm_100f to sm_110f)."""
    assert Target(arch, backend).register_budgets() == budgets


@pytest.mark.parametrize('arch,plain,exclusive', [
    ('sm_100a', 512, 512), ('sm_100f', 512, 512), ('sm_103a', 512, 512),
    ('sm_103f', 512, 512), ('sm_107a', 512, 576), ('sm_107f', 512, 576),
    ('sm_110a', 512, 512), ('sm_110f', 512, 512)])
def test_the_columns_a_block_allocates(arch, plain, exclusive):
    """512 in a plain allocation everywhere; 576 in an exclusive one on
    sm_107 alone -- sm_100f's family has sm_100 in it."""
    target = Target(arch, 'cuda')
    assert target.tmem_columns() == plain
    assert target.tmem_columns(exclusive=True) == exclusive


def _builder(target):
    return IRBuilder(fptype=Datatype.F32, arena='shrMem',
                     context=types.SimpleNamespace(target=target))


def test_the_builder_allocates_no_more_columns_than_the_target_has():
    """The target's columns where the context names one, and the powers of
    two up to 512 where it names none."""
    context = Context(arch='sm_100a', backend='cuda', fp_type=Datatype.F32)
    b = IRBuilder(fptype=Datatype.F32, arena='shrMem', context=context)
    assert b.tmem_alloc(512).type.shape == (128, 512)
    smaller = types.SimpleNamespace(tmem_columns=lambda: 256,
                                    hw=types.SimpleNamespace(model='sm_x'))
    with pytest.raises(IRError, match='at most 256'):
        _builder(smaller).tmem_alloc(512)
    assert _builder(smaller).tmem_alloc(256).type.shape == (128, 256)
    with pytest.raises(IRError):
        _builder(None).tmem_alloc(1024)


# -- the flags --------------------------------------------------------------- #

@pytest.fixture
def no_checks(monkeypatch):
    monkeypatch.delenv('TF_TMEM_ACCESS_CHECK', raising=False)


@pytest.mark.parametrize('arch,flags', [
    ('sm_100f', ['-gencode=arch=compute_100f,code=sm_100f']),
    ('sm_103a', ['-gencode=arch=compute_103a,code=sm_103a']),
    ('sm_103f', ['-gencode=arch=compute_103f,code=sm_103f']),
    ('sm_107a', ['-gencode=arch=compute_107a,code=sm_107a']),
    ('sm_107f', ['-gencode=arch=compute_107f,code=sm_107f']),
    ('sm_110a', ['-gencode=arch=compute_110a,code=sm_110a']),
    ('sm_110f', ['-gencode=arch=compute_110f,code=sm_110f']),
    ('sm_107', ['-arch=sm_107']), ('sm_121', ['-arch=sm_121'])])
def test_each_target_is_built_for_itself(arch, flags, no_checks):
    """An `a` target alone, an `f` target for its family, a plain part with
    `-arch`."""
    assert toolchain.COMPILERS['cuda'].target_flags(arch) == flags


@pytest.mark.parametrize('arch', ['sm_103', 'sm_107', 'sm_110', 'sm_121'])
def test_a_plain_parts_report_is_its_arch_specific_build(arch, no_checks):
    cmd, _ = toolchain.COMPILERS['cuda'].report_build(
        'nvcc', 'k.cu', '/tmp', arch, '/i')
    assert f'-arch={arch}a' in cmd


def test_the_report_of_a_family_target_is_the_familys_build(no_checks):
    cmd, _ = toolchain.COMPILERS['cuda'].report_build(
        'nvcc', 'k.cu', '/tmp', 'sm_100f', '/i')
    assert '-gencode=arch=compute_100f,code=sm_100f' in cmd


CHECK = '-Xptxas=-g-tmem-access-check'


def test_the_tensor_memory_checks_are_off_by_default(no_checks):
    for arch in SPECIFIC:
        assert CHECK not in toolchain.COMPILERS['cuda'].target_flags(arch)


def test_the_switch_builds_them_into_every_target_with_tcgen05(monkeypatch):
    """`TF_TMEM_ACCESS_CHECK=1`: ptxas's guardrails on every `a` and `f`
    target of the family, and on no other, where ptxas would only note that
    it ignores them."""
    monkeypatch.setenv('TF_TMEM_ACCESS_CHECK', '1')
    nvcc = toolchain.COMPILERS['cuda']
    for arch in SPECIFIC:
        assert nvcc.target_flags(arch)[-1] == CHECK, arch
    for arch, backend in NOT_TCGEN05:
        if backend == 'cuda':
            assert CHECK not in nvcc.target_flags(arch), arch
    monkeypatch.setenv('TF_TMEM_ACCESS_CHECK', '0')
    assert CHECK not in nvcc.target_flags('sm_100a')


def _inputs(arch):
    from harness import toolchain as htc
    target = htc.Target('cuda', arch, 'nvidia', 0, 'test')
    return htc, htc.BuildInputs(Path('.'), 'i', 'k', 'h', 'l', 'd', target,
                                INCLUDE)


def test_the_harness_keys_a_build_with_the_checks_apart(monkeypatch):
    """A binary built without the checks is not handed back for a run with
    them, and the ones built without keep their keys."""
    monkeypatch.delenv('TF_TMEM_ACCESS_CHECK', raising=False)
    htc, tcgen05 = _inputs('sm_100a')
    _, wgmma = _inputs('sm_90a')
    off = htc._cache_hash(tcgen05), htc._cache_hash(wgmma)
    monkeypatch.setenv('TF_TMEM_ACCESS_CHECK', '1')
    on = htc._cache_hash(tcgen05), htc._cache_hash(wgmma)
    assert on[0] != off[0] and on[1] == off[1]


# -- with nvcc --------------------------------------------------------------- #

#: The release that brought the family-specific targets, of parts an older
#: nvcc already lists (CUDA 12.8 has sm_100 and no sm_100f).
FAMILY_RELEASE = (12, 9)


@functools.lru_cache(maxsize=None)
def _toolkit():
    found = toolchain.COMPILERS['cuda'].find()
    if not found or not shutil.which(found):
        return None, (), (0, 0)
    listed = subprocess.run([found, '--list-gpu-code'], capture_output=True,
                            text=True, timeout=60).stdout.split()
    version = re.search(r'release (\d+)\.(\d+)', subprocess.run(
        [found, '--version'], capture_output=True, text=True,
        timeout=60).stdout)
    release = (int(version.group(1)), int(version.group(2))) if version \
        else (0, 0)
    return found, tuple(listed), release


def _nvcc_for(arch):
    """The toolkit's nvcc, or a skip where it builds no `arch`: the part not
    in `nvcc --list-gpu-code`, or a family target older than its release."""
    nvcc, listed, release = _toolkit()
    if nvcc is None:
        pytest.skip('nvcc not found')
    if (arch.rstrip('af') not in listed
            or (arch.endswith('f') and release < FAMILY_RELEASE)):
        pytest.skip(f'nvcc {release[0]}.{release[1]} builds no {arch}')
    return nvcc


#: What the device pass of `cuda.h` decides for a target, checked against
#: what its description says, by the preprocessor alone.
_GUARD = '''#include "tensorforge_device/cuda.h"
#if defined(__CUDA_ARCH__)
#if defined(TENSORFORGE_TCGEN05) != WANT_TCGEN05
#error "TENSORFORGE_TCGEN05 is not as the description says"
#endif
#if WANT_BITS >= 0 && TENSORFORGE_MATRIX_ADDRESS_BITS != WANT_BITS
#error "the descriptor's address bits are not as the description says"
#endif
#endif
'''


@pytest.mark.parametrize('arch', SPECIFIC + ('sm_90a', 'sm_100', 'sm_107',
                                             'sm_120'))
def test_cuda_h_takes_each_target_as_described(arch, tmp_path, no_checks):
    """`TENSORFORGE_TCGEN05` where the target has tcgen05 and nowhere else
    -- nvcc names an `f` target by `__CUDA_ARCH_FAMILY_SPECIFIC__` alone --
    and, where a matrix engine reads descriptors, their start address in as
    many bits as the description states: 15 on sm_107a and sm_107f."""
    nvcc = _nvcc_for(arch)
    target = Target(arch, 'cuda')
    engine = target.tcgen05() or target.wgmma()
    bits = target.hw.matrix_descriptor_address_bits if engine else None
    src = tmp_path / 'guard.cu'
    src.write_text(_GUARD)
    entry = toolchain.COMPILERS['cuda']
    run = subprocess.run(
        [nvcc, '-E', *entry.language_flags(), *entry.target_flags(arch),
         f'-DWANT_TCGEN05={int(target.tcgen05())}',
         f'-DWANT_BITS={-1 if bits is None else bits}', '-I', str(INCLUDE),
         str(src), '-o', str(tmp_path / 'guard.ii')],
        capture_output=True, text=True, timeout=300)
    assert run.returncode == 0, run.stderr[-2000:]


_DESCRIPTOR = '''#include "tensorforge_device/cuda.h"
__global__ void descriptor(unsigned long long *out, int i) {
  extern __shared__ float window[];
  out[threadIdx.x] = tensorforge::matrixDescriptor(&window[i], 0ull);
}
'''


@pytest.mark.parametrize('arch', SPECIFIC + ('sm_90a',))
def test_the_descriptor_fills_the_field_the_description_states(arch,
                                                               tmp_path,
                                                               no_checks):
    """The start address in 16-byte units, masked to the description's
    bits in the PTX of `tensorforge::matrixDescriptor` -- 15 on sm_107a and
    sm_107f, where an opted-into block reaches past 256 kB -- whether nvcc
    writes the mask or a field extraction."""
    nvcc = _nvcc_for(arch)
    bits = Target(arch, 'cuda').hw.matrix_descriptor_address_bits
    src = tmp_path / 'descriptor.cu'
    src.write_text(_DESCRIPTOR)
    entry = toolchain.COMPILERS['cuda']
    out = tmp_path / 'descriptor.ptx'
    run = subprocess.run([nvcc, '-ptx', *entry.language_flags(),
                          *entry.target_flags(arch), '-I', str(INCLUDE),
                          str(src), '-o', str(out)],
                         capture_output=True, text=True, timeout=300)
    assert run.returncode == 0, run.stderr[-2000:]
    ptx = out.read_text()
    mask = re.findall(r'\band\.b(?:32|64)\s+%\w+, %\w+, (\d+);', ptx)
    field = re.findall(r'\bbfe\.u(?:32|64)\s+%\w+, %\w+, 4, (\d+);', ptx)
    assert [int(m) for m in mask] == [(1 << bits) - 1] or \
        [int(f) for f in field] == [bits], ptx[-1500:]


def _text(arch):
    gen = tma._kernel(arch, 'volume', 4, 's')
    return gen.get_kernel(), gen.get_launcher(), gen.get_header()


@pytest.mark.parametrize('arch', [a for a in SPECIFIC if a != 'sm_100a'])
def test_the_family_writes_sm100as_tile(arch):
    """The o4 volume tile on tcgen05, byte for byte sm_100a's -- kernel,
    launcher and header: the targets differ in what `cuda.h` makes of
    them, and in nothing a planner reads yet.  A planner that takes what
    sm_107 adds (`Target.tmem_columns(exclusive=True)`) changes this."""
    assert _text(arch) == _text('sm_100a')


@pytest.mark.parametrize('arch', ['sm_103a', 'sm_107a', 'sm_110a',
                                  'sm_100f'])
def test_ptxas_builds_the_o4_volume_tile(arch, tmp_path):
    """Compiled for each: one UTCHMMA a product of the source -- the guard
    holds, so no product is left out -- and the tensor memory allocated and
    freed."""
    _nvcc_for(arch)
    src = tma._kernel(arch, 'volume', 4, 's').get_kernel()
    sass = tma._sass(arch, tmp_path, 'volume', 4)
    products = src.count('tcgen05.mma.cta_group')
    assert products > 0
    assert tma._count(sass, 'UTCHMMA') == products
    assert len(re.findall(r'UTCATOMSWS\.FIND_AND_SET', sass)) >= 1
