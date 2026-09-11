# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""How multiplications sit in waves."""
from dataclasses import dataclass
from math import gcd


def mults_per_group(num_threads: int, wave: int) -> int:
    """How many multiplications share the smallest whole number of waves.

    A barrier reaches waves, never parts of one, so this is the smallest set of
    multiplications a hardware barrier can separate from the rest of a block.
    One where the width divides the wave evenly; two where it does not, which
    in double precision is every other width the alignment produces --
    ``align_length`` is ``vec_unit_length * hw_fp_word_size / fp_size``, so at
    eight bytes per element it is half a wave and half the widths above the
    wave are odd multiples of it.
    """
    if num_threads <= 0 or wave <= 0:
        return 1
    return (num_threads * wave // gcd(num_threads, wave)) // num_threads


@dataclass(frozen=True)
class MultLayout:
    """Where the lanes of one multiplication sit, across waves.

    The unit is ``gcd(threads, wave)``: the longest run of lanes that neither
    straddles a wave nor splits a multiplication.  Every wave then holds a
    whole number of units and so does every multiplication, which is the only
    arrangement in which all the waves of a group look alike -- and looking
    alike is what lets one kernel body serve them.

    48 lanes on a 32-wide wave is the case that shows the difference.  The unit
    is 16, three waves hold two multiplications, and each wave holds 16 lanes
    of one and 16 of the other::

        wave 0: mult 0 lanes  0..15 | mult 1 lanes  0..15
        wave 1: mult 0 lanes 16..31 | mult 1 lanes 16..31
        wave 2: mult 0 lanes 32..47 | mult 1 lanes 32..47

    Laying a multiplication out contiguously gives the same three waves a
    different and worse split -- 32-0, 16-16, 0-32 -- where the first wave
    holds one multiplication, the last holds the other, and only the middle one
    is shared.  Three shapes instead of one, so the body would have to branch
    on which wave it is in.

    A *group* is ``lcm(threads, wave)`` threads: the smallest whole number of
    waves holding whole multiplications, and therefore the smallest set a
    hardware barrier can separate.  Everything a multiplication does that
    reaches past its own lanes -- a barrier above all -- is a property of its
    group, because a barrier reaches waves and never parts of one.
    """

    threads: int
    wave: int

    def __post_init__(self):
        if self.threads <= 0 or self.wave <= 0:
            raise ValueError(f'{self.threads} lanes over a {self.wave}-wide '
                             f'wave is not a layout')

    @property
    def unit(self) -> int:
        """Lanes that always stay together, in one wave and one mult."""
        return gcd(self.threads, self.wave)

    @property
    def units_per_mult(self) -> int:
        return self.threads // self.unit

    @property
    def units_per_wave(self) -> int:
        return self.wave // self.unit

    @property
    def group_threads(self) -> int:
        """The smallest whole number of waves holding whole multiplications."""
        return self.threads * self.wave // self.unit

    @property
    def mults_per_group(self) -> int:
        return self.group_threads // self.threads

    @property
    def waves_per_group(self) -> int:
        return self.group_threads // self.wave

    @property
    def units_per_group(self) -> int:
        return self.group_threads // self.unit

    @property
    def contiguous(self) -> bool:
        """Whether a multiplication is a run of lanes in one wave.

        Then the unit is the multiplication itself or the wave, the group is
        one wave, and the layout is what it always was: `threadIdx.x` is the
        lane and `threadIdx.y` the multiplication.
        """
        return self.threads <= self.wave and self.wave % self.threads == 0

    @property
    def whole_waves(self) -> bool:
        """Whether a multiplication is a whole number of waves."""
        return self.threads % self.wave == 0

    def mult_of_unit(self, index: int) -> int:
        """Which multiplication unit `index` of a block belongs to.

        Units are dealt out to the multiplications of a group in turn, so the
        neighbours in a wave belong to *different* multiplications; the wave
        then holds the same shape wherever it sits in the group.
        """
        group, within = divmod(index, self.units_per_group)
        return group * self.mults_per_group + within % self.mults_per_group

    def part_of_unit(self, index: int) -> int:
        """Which part of its multiplication unit `index` carries."""
        within = index % self.units_per_group
        return within // self.mults_per_group

    def lane_of(self, index: int, lane_in_unit: int) -> int:
        """The lane within its multiplication, for one thread."""
        return self.part_of_unit(index) * self.unit + lane_in_unit
