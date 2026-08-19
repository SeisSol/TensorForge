# SPDX-License-Identifier: MIT
"""The NVIDIA matmul path is asked whether it can emit, not told to try.

`primitives/nvidia.py` was unreachable: `_is_matmul` asked
`vendor in ['amd']`, so the `elif vendor == 'nvidia'` branch under it could
never run.  Turning it on is one word, and one word is exactly the wrong size
for this change -- the emitter's preconditions were `assert` statements, which
were harmless only while nothing reached them.

With the path live an assertion is not a rejection, it is an abort: a case
with a 16-wide wave would stop generating altogether, when the generic path
handles it perfectly well.  So the preconditions became `nvidia.supports()`, a
question the caller asks first, and `_is_matmul` consults it.  This file
checks both halves -- that the gate turns the right cases away, and that a
case it turns away still comes out of the generator.

Measured while enabling it: 9 of the corpus's CUDA cases take the path, and
the same 9 snapshots changed.  No HIP snapshot moved.

The path is parked (`nvidia.ENABLED`) pending a run on real hardware: `"+f"`
versus `"=f"`/`"f"` on the accumulator is a register-allocation difference no
front end can see.  The tests below that need the emitter's output turn it on
for themselves.  A parked path whose tests skip is a path that quietly rots --
that is how `nvidia.py` accumulated 23 unreachable definitions in the first
place -- so what is checked here is the emitter, which is worth checking
whether or not it is deployed.  `test_the_switch_is_off` is the separate,
one-line statement of the deployment decision.
"""

from __future__ import annotations

import pytest

from tensorforge.backend.instructions.compute.primitives import nvidia
from tensorforge.common.basic_types import Datatype

ATOM_TYPE = nvidia.ATOM.d


# --------------------------------------------------------------------------- #
# What the gate admits
# --------------------------------------------------------------------------- #

def test_a_warp_wide_dense_case_of_the_atoms_type_is_admitted():
    assert nvidia.supports(32, ATOM_TYPE, sparse=None)


@pytest.mark.parametrize("threads", [1, 2, 4, 8, 16, 64])
def test_any_other_wave_width_is_turned_away(threads):
    """The emitter is warp-level throughout -- it stages operands through
    `__syncwarp` and indexes shared memory by `threadIdx.x` modulo the atom's
    `k`.  Narrower waves need a warp-level broadcast and a way back; wider
    ones are a different instruction."""
    assert not nvidia.supports(threads, ATOM_TYPE, sparse=None)


def test_a_different_operand_type_is_turned_away():
    """`ATOM` is a TF32 instruction and nothing downstream compares the
    operand type against it, so an f64 case would emit
    `mma.sync...f32.tf32.tf32.f32` over doubles.  Quietly wrong is worse than
    loudly unsupported."""
    other = Datatype.F64 if ATOM_TYPE != Datatype.F64 else Datatype.F32
    assert not nvidia.supports(32, other, sparse=None)


def test_a_sparse_operand_is_turned_away():
    """`matmul` already declines these by returning `False`.  The gate has to
    agree, because `temp_shmem` reserves shared memory off the same
    predicate -- disagreement means a reservation for a kernel that never
    uses it."""
    assert not nvidia.supports(32, ATOM_TYPE, sparse=lambda k, j: True)


# --------------------------------------------------------------------------- #
# A rejected case still generates
# --------------------------------------------------------------------------- #

def _generate(case, backend="cuda", arch="sm_86"):
    import importlib.util
    from pathlib import Path

    from tensorforge.common.context import Context
    from tensorforge.generators.generator import Generator

    path = Path(__file__).parent / "cases" / case
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ctx = Context(arch=arch, backend=backend,
                  fp_type=getattr(mod, "DTYPE", None) or Datatype.F32)
    gen = Generator(mod.descr_list(), ctx)
    gen.generate()
    return gen.get_kernel()


#: `NAME` is `gemm_56x18_x_18x18`; the snapshot goes by that.
CASE_THAT_TAKES_THE_PATH = "rectangular.py"


@pytest.fixture
def enabled(monkeypatch):
    """The emitter, independent of whether the path is deployed."""
    monkeypatch.setattr(nvidia, "ENABLED", True)


def test_the_switch_is_off():
    """Not an opinion about whether it should be -- a place where the
    deployment decision is written down once, so flipping it is a diff."""
    assert nvidia.ENABLED is False, (
        "the path is live now; drop this test and re-record the CUDA "
        "snapshots, which move for 9 cases")


def test_a_case_that_takes_the_path_emits_the_instruction(enabled):
    source = _generate(CASE_THAT_TAKES_THE_PATH)
    assert nvidia.ATOM.name in source, (
        "the case no longer reaches the MMA path; pick another one")


def test_the_same_case_still_generates_when_the_gate_says_no(enabled,
                                                            monkeypatch):
    monkeypatch.setattr(nvidia, "supports", lambda *a, **k: False)
    source = _generate(CASE_THAT_TAKES_THE_PATH)
    assert source, "generation produced nothing"
    assert nvidia.ATOM.name not in source, "the gate was not consulted"


# --------------------------------------------------------------------------- #
# The inline asm the path emits
# --------------------------------------------------------------------------- #

def test_the_accumulator_is_one_read_write_operand(enabled):
    """`D` and `C` are the same accumulator at every call site.

    Listing it as `"=f"` under outputs and again as `"f"` under inputs states
    two unrelated operands that happen to name one C++ lvalue, and nothing
    then requires the compiler to give them the same register: it may read the
    accumulator into one and write the result into another, dropping the
    accumulation.  `"+f"` is the constraint that says read-and-write.
    """
    source = _generate(CASE_THAT_TAKES_THE_PATH)
    assert '"+f"' in source, "the accumulator is not a read-write operand"
    assert '"=f"' not in source, (
        "an output-only constraint on an operand that is also read")


def test_the_operand_numbering_survives_the_fold(enabled):
    """PTX numbers outputs and inputs in one sequence, so folding C into D
    shifts A and B down by `len(C)`.  Getting that wrong reads the wrong
    registers and still compiles."""
    import re

    source = _generate(CASE_THAT_TAKES_THE_PATH)
    m = re.search(r'"\{([%\d,]+)\}, \{([%\d,]+)\}, \{([%\d,]+)\}, '
                  r'\{([%\d,]+)\};"', source)
    assert m, "no mma operand groups found"
    d, a, b, c = (g.split(",") for g in m.groups())
    assert d == c, "D and C must name the same operands once folded"
    numbers = [int(x.lstrip("%")) for x in d + a + b]
    assert numbers == list(range(len(numbers))), (
        f"operand numbering is not contiguous from 0: {numbers}")
