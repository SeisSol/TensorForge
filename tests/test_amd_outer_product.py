# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""An outer product on AMD: one contraction row, nothing to spread over lanes.

`matmuldpp` read `B(0, j)` at the address every lane reads, and then repeated
it across each 16-lane row with a DPP row share.  On gfx1150 that came out
wrong for `lead_window_spans_two_blocks` -- 6.7e-5 off its numpy checksum with
the operand in global memory, and 18 % off once it was in the constant space:
the DPP in the inline assembly reads lanes that LLVM had no reason to write
the uniform value to, since it only copies it into a VGPR under the lane mask
that uses it.  The nest multiplies by the scalar and matches numpy exactly.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import pathlib

import pytest

from tensorforge.common.context import Context
from tensorforge.generators.generator import Generator

CASES = pathlib.Path(__file__).parent / 'cases'


def kernel(case, arch):
    spec = importlib.util.spec_from_file_location(case, CASES / f'{case}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    generator = Generator(module.descr_list(),
                          Context(arch=arch, backend='hip',
                                  fp_type=module.DTYPE))
    with contextlib.redirect_stdout(io.StringIO()):
        generator.generate()
    return generator.get_kernel()


@pytest.mark.parametrize('arch', ['gfx1150', 'gfx942', 'gfx90a', 'gfx1250'])
def test_an_outer_product_takes_no_row_share(arch):
    """`t[i,j,l] = A[i,j] v[l]` and nothing else that multiplies."""
    text = kernel('lead_window_spans_two_blocks', arch)
    assert 'fmacdpp' not in text
    assert 'tensorforge::broadcast<' not in text
