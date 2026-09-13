# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""What a launch of one generated kernel uses, decided once.

The block, the multiplications a block holds, the shared memory and whether
the launch is cooperative were decided in the generator and then existed only
as literals in the launcher text -- which the launcher, the kernel's launch
bounds, the occupancy estimate, the tuning and half a dozen host tools each
re-derived or read back with a regex.  This is the one place they are stated,
and what the header publishes to host code (`LaunchInfo`, and the
`launch_config_<kernel>` function that adds the grid, which depends on the
device and the element count and so is decided at run time).
"""

from dataclasses import asdict, dataclass
from typing import Tuple

#: The macro guarding the two structs, so that every header and every launcher
#: translation unit can carry them and a file holding several kernels
#: defines them once.
TYPES_GUARD = 'TENSORFORGE_LAUNCH_TYPES'


@dataclass(frozen=True)
class SectionLaunch:
    """What one section of the kernel planned for."""

    mults_per_block: int
    #: Elements of shared memory, the prologue's arena included.
    shared_elements: int
    #: Whether the section ends in a grid-wide barrier.
    barrier: bool


@dataclass(frozen=True)
class LaunchConfig:
    """The launch every section of one kernel runs under."""

    #: Lanes one multiplication is spread over, and how many of them hold data.
    threads_per_mult: int
    active_threads: int
    #: Adjacent lead elements a lane holds.
    lead_width: int
    mults_per_block: int
    block: Tuple[int, int, int]
    shared_elements: int
    shared_bytes: int
    #: A grid-wide barrier needs every block resident at once.
    cooperative: bool
    #: Whether the grid is sized by occupancy and walks the batch (the
    #: default), or holds one block per `mults_per_block` elements.
    persistent: bool
    sections: Tuple[SectionLaunch, ...]

    @property
    def threads_per_block(self) -> int:
        x, y, z = self.block
        return x * y * z

    def to_dict(self) -> dict:
        out = asdict(self)
        out['block'] = list(self.block)
        out['sections'] = [asdict(s) for s in self.sections]
        return out

    def describe(self) -> str:
        """One line for the kernel's comment block."""
        x, y, z = self.block
        active = ('' if self.active_threads == self.threads_per_mult
                  else f' ({self.active_threads} active)')
        width = '' if self.lead_width == 1 else f', lead width {self.lead_width}'
        grid = 'occupancy grid' if self.persistent else 'one block per element group'
        coop = ', cooperative' if self.cooperative else ''
        return (f'{self.threads_per_mult} lanes{active}{width} x '
                f'{self.mults_per_block} per block = block {x}x{y}x{z}, '
                f'{self.shared_bytes} B shared, {grid}{coop}')


def launch_types() -> str:
    """The two structs host code sees, behind `TYPES_GUARD`.

    Plain C++ and nothing of a device runtime: a header holding them is
    included by host code that knows no `dim3` and no `sycl::range`.
    """
    return (f'#ifndef {TYPES_GUARD}\n'
            f'#define {TYPES_GUARD}\n'
            '#include <cstddef>\n'
            'namespace tensorforge {\n'
            '// Fixed when the kernel is generated: `launch_info_<kernel>`.\n'
            'struct LaunchInfo {\n'
            '  unsigned block[3];\n'
            '  unsigned threadsPerMult;\n'
            '  unsigned activeThreads;\n'
            '  unsigned leadWidth;\n'
            '  unsigned multsPerBlock;\n'
            '  std::size_t sharedMemBytes;\n'
            '  bool cooperative;\n'
            '  bool persistent;\n'
            '  unsigned sections;\n'
            '};\n'
            '// What one launch uses, the grid included: `launch_config_<kernel>`.\n'
            'struct LaunchConfig {\n'
            '  std::size_t grid[3];\n'
            '  std::size_t block[3];\n'
            '  std::size_t sharedMemBytes;\n'
            '  bool cooperative;\n'
            '};\n'
            '} // namespace tensorforge\n'
            '#endif\n')


def launch_info_initializer(config: LaunchConfig) -> str:
    """`LaunchInfo`'s fields for `config`, in declaration order."""
    x, y, z = config.block
    flag = lambda b: 'true' if b else 'false'
    return (f'{{{{{x}, {y}, {z}}}, {config.threads_per_mult}, '
            f'{config.active_threads}, {config.lead_width}, '
            f'{config.mults_per_block}, {config.shared_bytes}, '
            f'{flag(config.cooperative)}, {flag(config.persistent)}, '
            f'{len(config.sections)}}}')
