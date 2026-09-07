# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Static analyses over descriptor lists that no code generation depends on.

Everything here answers a question *about* a kernel rather than contributing
to building one: how much arithmetic it contains, how many bytes it has to
move, what its arithmetic intensity is.  Kept out of
:mod:`tensorforge.generators.descriptions` deliberately -- a descriptor
describes what to compute, and a cost is a claim about the machine, which is a
different kind of statement and changes for different reasons.
"""

from .cost import (Cost, TensorTraffic, descr_cost, list_cost)

__all__ = ['Cost', 'TensorTraffic', 'descr_cost', 'list_cost']
