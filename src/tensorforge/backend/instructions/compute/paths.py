# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Which module emits a target's contractions.

Asked here rather than of the target: the modules are this package's, and
they ask the target what it can do -- a target that knew its modules would
be imported by what it imports.
"""

from typing import Optional

from .strategy import MatrixPaths


def matrix_paths(target) -> Optional[MatrixPaths]:
    """The module that emits `target`'s contractions, one per vendor, or None
    where only the generic nest does."""
    from .primitives import amd, intel, nvidia
    return {'amd': amd, 'nvidia': nvidia, 'intel': intel}.get(
        target.hw.vendor)
