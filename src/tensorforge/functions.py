# SPDX-FileCopyrightText: 2015 SeisSol Group
#
# SPDX-License-Identifier: MIT

"""Public elementwise builders: ``tensorforge.functions.tanh(dest, src)`` etc.

The helpers build :class:`ElementwiseDescr` directly.  The calling convention
is destination first: ``f(dest, *srcs)``.
"""

from tensorforge.generators.elementwise import *   # noqa: F401,F403
from tensorforge.generators.elementwise import __all__   # noqa: F401
