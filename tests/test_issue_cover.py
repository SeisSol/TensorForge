# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""The cover a transfer is issued with, checked on the generated text.

`move` takes a transfer ahead over whole instructions, and one that passes an
instruction passes all of it -- `chain_five` has transfers it takes to the head
of the element, thousands of lines ahead of what reads them.  `issue_cover`
caps the cover they keep: behind `flatten` a transfer issued further ahead
than that goes back down among the statements (`pir.move.cap_cover`), and its
buffer is free for the stretch it no longer covers.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

from tensorforge.common.context import Context, Options
from tensorforge.generators.generator import Generator

CASES = Path(__file__).parent / "cases"


def _generate(case_file: str, backend: str, arch: str, **opts) -> str:
    path = CASES / case_file
    spec = importlib.util.spec_from_file_location("tf_cover__" + path.stem,
                                                  path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ctx = Context(arch=arch, backend=backend,
                  fp_type=getattr(mod, "DTYPE", None),
                  options=Options(**opts))
    gen = Generator(mod.descr_list(), ctx)
    gen.generate()
    return gen.get_kernel()


def _gaps(kernel):
    """For every transfer of the element's own -- not one for the next
    element, nor the peel -- the lines from its comment to the first read of
    what it fills: the wait, for a copy into shared memory."""
    lines = kernel.splitlines()
    out = {}
    for i, line in enumerate(lines):
        m = re.search(r"// (\w+) = load\{g>[rs]\}\(glb_", line)
        if not m:
            continue
        name = m.group(1)
        out[name] = next(
            j for j in range(i + 1, len(lines))
            if re.search(rf"=\s*[^=;]*\b{name}\[", lines[j])
            or f"wait({name} " in lines[j]) - i
    return out


def test_a_transfer_issued_far_ahead_goes_back_down():
    free = _gaps(_generate("chain_five.py", "cuda", "sm_86", issue_cover=0))
    capped = _gaps(_generate("chain_five.py", "cuda", "sm_86"))
    far = [name for name, gap in free.items() if gap > 1000]
    assert far, "expected move to take a transfer of chain_five far ahead"
    assert all(capped[name] * 4 < free[name] for name in far), (free, capped)
