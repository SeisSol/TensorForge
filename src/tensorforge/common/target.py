# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
# SPDX-FileContributor: David Schneller

"""The target a kernel is generated for: a device, and the language it is
written in for that device.

Two parts, and they answer different questions:

* `hw`, the device's row of `hw_descr_db.yml`: the wave, the register files,
  shared memory, the instruction cache.  What the silicon has.
* `lexic`, the spelling: how a statement the generator has decided on is
  written in CUDA, HIP, SYCL, ESIMD or OpenMP.

The device alone does not name a target -- HIP compiles for NVIDIA as well,
and one Intel device runs two lowerings -- and neither does the language, so
a context holds the pair.
"""

from typing import List

from tensorforge.common.vm.hw_descr import HwDecription, hw_descr_factory
from tensorforge.common.vm.lexic import Lexic, lexic_factory


class Target:
    """One device and one lowering for it."""

    def __init__(self, arch: str, backend: str):
        #: What the device has.
        self.hw: HwDecription = hw_descr_factory(arch, backend)
        #: How a decided statement is written.
        self.lexic: Lexic = lexic_factory(backend=backend,
                                          underlying_hardware=self.hw.vendor)

    def headers(self) -> List[str]:
        """The headers a translation unit with this target's kernels
        includes."""
        return ['tensorforge_aux.h'] + self.lexic.get_headers()
