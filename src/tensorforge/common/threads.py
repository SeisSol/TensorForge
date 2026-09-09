# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""How multiplications sit in waves."""
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
