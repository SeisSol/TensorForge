# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""A destination preloaded as the accumulation bias sits in the loop's frame.

``D[35:56, :] += S[35:56, :] Z`` --- an accumulation onto a slice whose lead
offset is not a multiple of the thread count. The lead loop runs in the
theta-shifted origin, theta being the part of the offset below a slot
boundary, and the store adds the slot-aligned part back on the way to memory.
The preloaded image has to be staged in that same frame: staged in the
tensor's, its origin is one whole slot away from the loop's, and the bias is
read one element before the array it was loaded into. The first element of the
accumulation then picks up whatever register precedes it and the last one is
dropped.

Nothing about it shows up in the numbers the host oracle produces, because the
interpreter keeps registers in an unbounded dict and happily serves index -1.
So the assertion is on the generated source, through the same bounds check
that runs over a whole dump.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.context import Context
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators.descriptions import MultilinearDescr
from tensorforge.generators.generator import Generator

ROOT = Path(__file__).resolve().parent.parent
CHECKER = ROOT / "tools" / "host" / "check_structure.py"

DTYPE = Datatype.F32
LEAD, WIDTH, ROWS, OFFSET = 64, 6, 21, 35

TARGETS = [("cuda", "sm_86"), ("hip", "gfx90a")]


def _tensor(shape, alias, addressing=Addressing.PTR_BASED, is_tmp=False):
    return Tensor(shape, addressing, BoundingBox([0] * len(shape), list(shape)),
                  alias=alias, is_tmp=is_tmp, datatype=DTYPE)


def _descr_list():
    staged = _tensor([LEAD, WIDTH], "S", Addressing.STRIDED, is_tmp=True)
    destination = _tensor([LEAD, WIDTH], "D")
    window = BoundingBox([0, 0], [ROWS, WIDTH])
    return [
        MultilinearDescr(dest=SubTensor(staged),
                         ops=[SubTensor(_tensor([LEAD, WIDTH], "A"))],
                         target=[[0, 1]], permute=[[0, 1]]),
        MultilinearDescr(
            dest=SubTensor(destination, bbox=window, offset=[OFFSET, 0]),
            ops=[SubTensor(staged, bbox=window, offset=[OFFSET, 0]),
                 SubTensor(_tensor([WIDTH, WIDTH], "Z"))],
            target=[[0, -1], [-1, 1]], permute=[[0, 1], [0, 1]], add=True),
    ]


def _generate(backend, arch):
    gen = Generator(_descr_list(), Context(arch=arch, backend=backend, fp_type=DTYPE))
    gen.generate()
    return gen.get_kernel() + "\n" + gen.get_launcher()


@pytest.mark.parametrize("backend,arch", TARGETS)
def test_the_bias_is_read_from_inside_its_array(tmp_path, backend, arch):
    dump = tmp_path / "gpulike_subroutine.cpp"
    dump.write_text(_generate(backend, arch))
    finished = subprocess.run([sys.executable, str(CHECKER), str(dump)],
                              capture_output=True, text=True, timeout=300)
    assert finished.returncode == 0, finished.stderr
    assert "REG OOB" not in finished.stdout, finished.stdout


@pytest.mark.parametrize("backend,arch", TARGETS)
def test_the_bias_is_read_at_all(backend, arch):
    """The image is staged once and used, rather than staged twice.

    Moving the image into the loop's frame changes the box it covers, and the
    residency is keyed on that box: recorded in one frame and looked up in the
    other, the entry misses, the operation stages a second copy of the same
    region, and the first is dead weight in a kernel that already spills.
    """
    source = _generate(backend, arch)
    staged = re.findall(r"^\s*// (r\d+) = load\{g>r\}\(glb_\w+\);", source,
                        re.MULTILINE)
    assert len(staged) == 2, staged
