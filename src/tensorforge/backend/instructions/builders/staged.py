# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""A copy one operation's build staged of a tensor it reads.

An operand loaded into a register array or a shared buffer, and the bias an
accumulation adds onto, are copies of what memory holds.  A build that reads
one tensor twice -- the destination an operand as well, or the bias staged
ahead of the product and asked for again behind it -- finds the first copy
and reads it where it serves (`MultilinearBuilder._staged`).  Nothing of it
outlives the build: the next operation stages from memory anew, and what the
round trip costs, the pseudo-IR takes back (`pir.forward`).

`covered` and `shift` are not decoration.  An image is indexed in its own
coordinates -- position `r` holds tensor element `r + shift` -- and covers
only what was staged, so an operand asking for a different slice of the same
tensor has to be told no rather than handed the wrong elements.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from tensorforge.backend.symbol import Symbol
from tensorforge.common.matrix.boundingbox import BoundingBox


@dataclass
class StagedCopy:
    """One tensor's copy, and what of the tensor it holds."""

    #: Where the copy is.
    image: Symbol
    #: What the image holds, in the image's own coordinates, or None when
    #: the staging did not record it.
    covered: Optional[BoundingBox] = None
    #: Position `r` of the image holds tensor element `r + shift`.
    shift: List[int] = field(default_factory=list)

    def holds(self, bbox: BoundingBox, offset: List[int]) -> bool:
        """Does the image cover the box `bbox + offset` asks for?

        Answered in image coordinates, which is what `shift` is for.  A copy
        with no recorded `covered` cannot answer and says so by returning
        True: the caller then has to decide what an unrecorded range means.
        """
        if self.covered is None:
            return True
        for j in range(bbox.rank()):
            lo = bbox.lower()[j] + offset[j] - self.shift[j]
            hi = bbox.upper()[j] + offset[j] - self.shift[j]
            if lo < self.covered.lower()[j] or hi > self.covered.upper()[j]:
                return False
        return True

    def region(self) -> Optional[Tuple[List[int], List[int]]]:
        """What the image holds, in *tensor* coordinates."""
        if self.covered is None:
            return None
        return ([l + s for l, s in zip(self.covered.lower(), self.shift)],
                [u + s for u, s in zip(self.covered.upper(), self.shift)])
