# SPDX-FileCopyrightText: 2015 SeisSol Group
#
# SPDX-License-Identifier: MIT

"""Public elementwise builders: ``tensorforge.functions.tanh(dest, src)`` etc.

This used to re-export ``generators.optree``, whose helpers wrapped an
``Operation`` member in an expression-tree node.  The tree is gone; the helpers
now build :class:`ElementwiseDescr` directly.  Note the changed calling
convention: destination first, ``f(dest, *srcs)``, rather than a nested
expression that had to be wrapped in an ``Assignment``.
"""

from tensorforge.generators.elementwise import *   # noqa: F401,F403
from tensorforge.generators.elementwise import __all__   # noqa: F401
