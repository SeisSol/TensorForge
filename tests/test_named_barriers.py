# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""A multiplication wider than the wave meets on a barrier object of its own
on gfx1250 and gfx1251.

Before gfx12.5 the block barrier is the only one an AMD shader sees, so a
multiplication spanning waves got a block to itself.  gfx12.5 gives a block
sixteen objects it may assign to subsets of its waves (CDNA 5 ISA, 5.6),
counted in waves; a wave waits on the one it joined.  A kernel whose
multiplications meet apart opens with a prologue -- every wave sets its
multiplication's count, the block meets once at its own barrier, every wave
joins -- and meets each multiplication on its object after that, so a block
holds as many multiplications as it has objects (`Target.separable_mults`).

No gfx1250 runs here.  The kernels are compiled for gfx1250 and gfx1251 and
read as instructions, which is what can be checked without one: the counts in
`M0`, the order of the prologue, a signal and a wait per rendezvous, and the
objects the kernel descriptor asks the dispatch for.
"""

from __future__ import annotations

import re
import subprocess

import pytest

from tensorforge.common.basic_types import Datatype
from tensorforge.common.context import Context


def _fold(arch, threads, rows=128):
    """`out[j] = sum_i a[i, j]` at `threads` lanes: past a wave, the waves'
    partials meet in the multiplication's shared memory between two
    rendezvous of the multiplication (`_fold_across_waves`)."""
    from tensorforge.common.basic_types import Addressing
    from tensorforge.common.matrix.boundingbox import BoundingBox
    from tensorforge.common.matrix.tensor import SubTensor, Tensor
    from tensorforge.common.operation import AddOperator
    from tensorforge.generators.descriptions import ReductionDescr
    from tensorforge.generators.generator import Generator
    from tensorforge.generators.lanes import LaneConfig

    source = SubTensor(Tensor([rows, 4], Addressing.STRIDED,
                              BoundingBox([0, 0], [rows, 4]), alias="A",
                              datatype=Datatype.F32))
    dest = SubTensor(Tensor([4], Addressing.STRIDED, BoundingBox([0], [4]),
                            alias="OUT", datatype=Datatype.F32))
    gen = Generator([ReductionDescr(dest, source, [0], AddOperator())],
                    Context(arch=arch, backend="hip", fp_type=Datatype.F32),
                    lanes=LaneConfig(num_threads=threads,
                                     num_active_threads=min(rows, threads),
                                     lead_width=1))
    gen.generate()
    return gen


def _sync_text(arch, participants, threads):
    from tensorforge.backend.pir.emit import Emitter

    class _Sink:
        def __call__(self, *args, **kwargs):
            pass

        def __getattr__(self, _):
            return lambda *a, **k: None

    ctx = Context(arch=arch, backend="hip", fp_type=Datatype.F32)
    return Emitter(_Sink(), ctx)._sync(participants, threads)


# -- what the target answers ------------------------------------------------- #

def test_the_barrier_of_a_wide_multiplication_is_its_object():
    """`SyncThreads` asks for the multiplication itself, and the emitter
    spells its object -- the row's element of the kernel's array."""
    from tensorforge.backend.instructions.sync_block import SyncThreads
    from tensorforge.backend.pir.core import Participants, Uniformity

    ctx = Context(arch="gfx1250", backend="hip", fp_type=Datatype.F32)
    sync = SyncThreads(ctx, 64)
    assert sync.participants() is Participants.MULT
    assert sync.barrier_scope() is Uniformity.MULT
    text = _sync_text("gfx1250", Participants.MULT, 64)
    assert text == "tensorforge::syncNamedBarrier(&tfBarriers[threadIdx.y]);"
    # Within a wave the lanes are in lockstep, and nothing is spelled.
    assert _sync_text("gfx1250", Participants.MULT, 16) is None
    # A width with a partial wave has no count of waves: the group's.
    assert SyncThreads(ctx, 48).participants() is Participants.MULTGROUP


@pytest.mark.parametrize("arch", ["gfx942", "gfx1150", "gfx1200"])
def test_without_objects_a_wide_multiplication_has_no_barrier_of_its_own(arch):
    """RDNA 4 included: clang emits the instructions for gfx1200, and the
    dispatch allocates no object there."""
    from tensorforge.backend.pir.core import IRError, Participants

    ctx = Context(arch=arch, backend="hip", fp_type=Datatype.F32)
    wide = 2 * ctx.target.hw.vec_unit_length
    assert not ctx.target.sync_mult(wide)
    assert ctx.target.lexic.named_barrier_prologue(
        wide, 2, ctx.target.hw) == ()
    with pytest.raises(IRError):
        _sync_text(arch, Participants.MULT, wide)


def test_the_prologue_counts_in_waves_and_holds_an_object_a_multiplication():
    ctx = Context(arch="gfx1251", backend="hip", fp_type=Datatype.F32)
    lexic, hw = ctx.target.lexic, ctx.target.hw
    assert lexic.named_barrier_prologue(128, 3, hw) == (
        "__shared__ __amdgpu_named_workgroup_barrier_t tfBarriers[3];",
        "tensorforge::joinNamedBarrier<4>(&tfBarriers[threadIdx.y]);")
    assert lexic.named_barrier_prologue(32, 3, hw) == (), (
        "a multiplication that is the wave needs no object")
    assert lexic.named_barrier_prologue(48, 3, hw) == ()
    with pytest.raises(ValueError):
        lexic.named_barrier_prologue(64, 17, hw)


def test_a_block_holds_no_more_multiplications_than_objects(monkeypatch):
    """Sixteen objects, all the multiplications': however many threads the
    block may have, a seventeenth multiplication would have none."""
    import dataclasses
    from tensorforge.common import target as target_module
    from tensorforge.generators.generator import RegmaxBlockPolicy

    ctx = Context(arch="gfx1250", backend="hip", fp_type=Datatype.F32)
    wide = dataclasses.replace(target_module.PREFERENCES['amd'],
                               unstaged_block_threads=4096)
    monkeypatch.setitem(target_module.PREFERENCES, 'amd', wide)
    policy = RegmaxBlockPolicy(ctx, global_mem=0, mem_size_per_mult=0,
                               num_threads=64)
    assert policy.get_num_mults_per_block() > 16, (
        "without a barrier nothing caps the block, and the test needs that")
    policy.set_has_barrier(True)
    assert policy.get_num_mults_per_block() == 16

    rdna = Context(arch="gfx1150", backend="hip", fp_type=Datatype.F32)
    policy = RegmaxBlockPolicy(rdna, global_mem=0, mem_size_per_mult=0,
                               num_threads=64)
    policy.set_has_barrier(True)
    assert policy.get_num_mults_per_block() == 1, (
        "no object: the block barrier is the multiplication's own")


# -- what the generator writes ----------------------------------------------- #

def test_the_kernel_opens_with_the_prologue_and_meets_each_multiplication():
    """A fold at 64 lanes, which a gfx1250 block could not hold more than
    one of before: now several, each meeting on its own object, and the
    block's barrier once, in the prologue, ahead of every section."""
    gen = _fold("gfx1250", 64)
    src = gen.get_kernel()
    mults = gen._launch.mults_per_block
    assert mults > 1
    lines = [l.strip() for l in src.splitlines()]
    declare = lines.index(
        f"__shared__ __amdgpu_named_workgroup_barrier_t tfBarriers[{mults}];")
    assert lines[declare + 1] == (
        "tensorforge::joinNamedBarrier<2>(&tfBarriers[threadIdx.y]);")
    assert lines[declare + 2] == "{", "ahead of the first section"
    meets = [l for l in lines if "syncNamedBarrier" in l]
    assert len(meets) >= 2
    assert all(l == "tensorforge::syncNamedBarrier(&tfBarriers[tfMult]);"
               for l in meets), meets
    assert "__syncthreads" not in src


def test_a_kernel_whose_multiplications_never_meet_asks_for_no_object():
    """Objects come out of a pool of 64 a WGP; a kernel without a barrier of
    one multiplication declares none."""
    assert "tfBarriers" not in _fold("gfx1250", 32).get_kernel()


def test_a_block_of_one_multiplication_meets_at_the_block_barrier():
    """At 256 lanes the block holds one multiplication: its barrier is the
    block's, as on every AMD target before, with no prologue and no object."""
    gen = _fold("gfx1251", 256, rows=256)
    src = gen.get_kernel()
    assert gen._launch.mults_per_block == 1
    assert "tfBarriers" not in src and "joinNamedBarrier" not in src
    assert src.count("__syncthreads();") >= 2


# -- what the device compiler makes of it ------------------------------------ #

def _assemble(source: str, arch: str, tmp_path):
    from harness import syntax
    from tensorforge import toolchain

    hipcc = syntax.device_compiler('hip')
    if hipcc is None:
        pytest.skip('hipcc not found')
    path = tmp_path / f'{arch}.hip'
    path.write_text(source)
    run = subprocess.run([hipcc, '-O3', '-std=c++17', f'--offload-arch={arch}',
                          '-I', toolchain.include_dir(), '--cuda-device-only',
                          '-S', '-o', '-', str(path)],
                         capture_output=True, text=True, timeout=600)
    if run.returncode and re.search(r'unsupported .*architecture|'
                                    r'invalid offload arch', run.stderr):
        pytest.skip(f'hipcc does not know {arch}')
    return run


@pytest.mark.parametrize('arch', ['gfx1250', 'gfx1251'])
def test_the_compiled_kernel_counts_joins_and_meets(arch, tmp_path):
    """Read off the instructions: the count in `M0[22:16]`, the block's own
    barrier between the counts and the joins and nowhere else, a signal of
    an object followed by a wait on the joined one per rendezvous, and the
    objects asked of the dispatch -- in fours, so four of them are one."""
    gen = _fold(arch, 64)
    run = _assemble('#include "tensorforge_device/hip.h"\n'
                    + gen.get_kernel(), arch, tmp_path)
    assert run.returncode == 0, run.stderr[-3000:]
    body = [line.strip() for line in run.stdout.splitlines()
            if line.startswith('\t') and not line.strip().startswith(';')]
    init = body.index('s_barrier_init m0')
    counts = [ins for ins in body[:init] if ins.startswith('s_or_b32 m0')]
    assert counts and counts[-1].startswith('s_or_b32 m0, 0x20000'), (
        'two waves, in bits 22:16 of M0')
    join = body.index('s_barrier_join m0')
    assert init < body.index('s_barrier_signal -1') < body.index(
        's_barrier_wait -1') < join
    assert body.count('s_barrier_signal -1') == 1
    signals = [i for i, ins in enumerate(body) if ins == 's_barrier_signal m0']
    assert signals and all(i > join for i in signals)
    assert all(body[i + 1] == 's_barrier_wait 1' for i in signals)
    assert re.search(r'\.amdhsa_named_barrier_count\s+1\b', run.stdout)
    assert re.search(r'\.num_named_barrier,\s*4\b', run.stdout), (
        'one object a multiplication of the block')


@pytest.mark.parametrize('arch', ['gfx942', 'gfx1200'])
def test_the_helpers_refuse_a_target_without_objects(arch, tmp_path):
    """Where the dispatch allocates no object the instructions are no-ops:
    the header stops the compile instead of letting everyone through."""
    run = _assemble('#include "tensorforge_device/hip.h"\n'
                    '__global__ void k(float *x) {\n'
                    '  __shared__ __amdgpu_named_workgroup_barrier_t b[2];\n'
                    '  tensorforge::joinNamedBarrier<2>(&b[threadIdx.y]);\n'
                    '  tensorforge::syncNamedBarrier(&b[threadIdx.y]);\n'
                    '  x[threadIdx.x] = 1.f;\n'
                    '}\n', arch, tmp_path)
    assert run.returncode != 0
    assert 'named barriers exist on gfx1250 and gfx1251 only' in run.stderr
