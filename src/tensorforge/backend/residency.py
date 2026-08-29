# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Where the newest copy of a tensor currently lives.

A value does not have to be in the place its symbol names.  An operand loaded
into a register array is a copy of something global memory still holds; an
accumulator holding a result that has not been stored yet is the *only* copy of
it.  Both are recorded here, and telling them apart is the whole point: the
first may be dropped at will, the second has to reach memory before anything
can be allowed to overwrite or bypass it.

Entries are keyed by symbol name and live for the whole kernel, which is why
`covered` and `shift` are not decoration.  An image is indexed in its own
coordinates -- position `r` holds tensor element `r + shift` -- and covers only
what its producer happened to stage, so a later consumer asking for a different
slice has to be told no rather than handed the wrong elements.

`kind` is stated rather than derived.  It used to be read back off `image is
home`, which is true but reads as an accident at each of the three sites that
did it, and the neighbouring test on `home.stype` -- which looks equivalent and
is not, since a writeback into a shared-memory temporary has a non-global home
-- is a step away.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterator, List, Optional, Tuple

from tensorforge.backend.instructions.memory.store import (StoreRegToGlb,
                                                           StoreRegToShr)
from tensorforge.backend.symbol import Symbol, SymbolType
from tensorforge.common.matrix.boundingbox import BoundingBox


class ResidencyKind(Enum):
    #: A copy of something a slower memory still holds.  Dropping it loses
    #: nothing; the ordinary path re-stages whatever is wanted.
    PRELOAD = 'preload'
    #: A result that exists only in `image`.  Dropping it loses the value, so
    #: it has to be written to `home` first.
    WRITEBACK = 'writeback'


@dataclass
class ResidencyEntry:
    """One tensor's newest copy, and what may be done with it."""

    kind: ResidencyKind
    #: Where the value is now.
    image: Symbol
    #: Where it belongs.  The same symbol as `image` for a preload, since a
    #: preload's value is already where it belongs.
    home: Symbol
    #: What the image actually holds, in the image's own coordinates, or None
    #: when the producer did not record it.
    covered: Optional[BoundingBox] = None
    #: Position `r` of the image holds tensor element `r + shift`.
    shift: List[int] = field(default_factory=list)
    #: Whether the pending store is an atomic update.  Only meaningful for a
    #: writeback.
    atomic: Optional[bool] = None
    #: The box the producing operation undertook to define, so the store can
    #: zero-fill exactly what the range narrowing left out and nothing else.
    promise: Optional[BoundingBox] = None

    @property
    def is_preload(self) -> bool:
        return self.kind is ResidencyKind.PRELOAD

    def holds(self, bbox: BoundingBox, offset: List[int]) -> bool:
        """Does the image cover the box `bbox + offset` asks for?

        Answered in image coordinates, which is what `shift` is for.  An entry
        with no recorded `covered` cannot answer and says so by returning True:
        the caller then has to decide what an unrecorded range means, and the
        two callers here disagree, so it is not decided for them.
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


class Residency:
    """The set of tensors whose newest copy is not where their symbol says.

    One of these belongs to one section.  The lifetime matters: an entry that
    outlived its section would have its writeback emitted after the barrier
    that was supposed to make the value visible, which is later than "never" is
    wrong in a more confusing way.

    Flushing lives here rather than in whoever happens to be building, because
    everything a store needs -- the context, the section's shared memory, the
    lane count -- is a fact about the section and not about the operation that
    left the value behind.  The instructions are returned rather than appended,
    so a caller splices them into its own stream at the point it chose.
    """

    def __init__(self, context, shr_mem: Symbol, num_threads: int,
                 lead_width: int = 1):
        self._entries: Dict[str, ResidencyEntry] = {}
        self._context = context
        self._shr_mem = shr_mem
        self._num_threads = num_threads
        #: How many lead-dimension elements one lane covers.  A property of
        #: the section like the other three, and it has to reach the store:
        #: a writeback emitted here is the same store the builder would have
        #: emitted inline, and it addresses the same way.
        self._lead_width = lead_width

    def __contains__(self, name: str) -> bool:
        return name in self._entries

    def get(self, name: str) -> Optional[ResidencyEntry]:
        return self._entries.get(name)

    def items(self) -> Iterator[Tuple[str, ResidencyEntry]]:
        return iter(list(self._entries.items()))

    def record_preload(self, name: str, image: Symbol,
                       covered: Optional[BoundingBox] = None,
                       shift: Optional[List[int]] = None) -> ResidencyEntry:
        """`image` is a copy of what `name` already holds elsewhere."""
        return self._put(ResidencyEntry(kind=ResidencyKind.PRELOAD,
                                        image=image, home=image,
                                        covered=covered,
                                        shift=list(shift or []),
                                        atomic=None, promise=None), name)

    def record_writeback(self, name: str, image: Symbol, home: Symbol,
                         covered: Optional[BoundingBox] = None,
                         shift: Optional[List[int]] = None,
                         atomic: Optional[bool] = None,
                         promise: Optional[BoundingBox] = None
                         ) -> ResidencyEntry:
        """`image` is the only copy of a result destined for `home`."""
        return self._put(ResidencyEntry(kind=ResidencyKind.WRITEBACK,
                                        image=image, home=home,
                                        covered=covered,
                                        shift=list(shift or []),
                                        atomic=atomic, promise=promise), name)

    def drop(self, name: str) -> Optional[ResidencyEntry]:
        """Forget `name`, and hand back what was forgotten.

        A preload can be discarded on the spot.  A writeback cannot: the caller
        gets the entry back precisely so it can emit the store first, and
        `is_preload` is what it tests to decide.
        """
        return self._entries.pop(name, None)

    def flush(self, name: str) -> List:
        """Make `name`'s newest copy reach the place its symbol names.

        A preload needs nothing: the value is already there, so the entry is
        simply forgotten.  A writeback is emitted, to global or to shared
        memory according to where its home is.

        Anything about to overwrite or bypass the image has to call this
        first, which is the whole reason the record exists.
        """
        entry = self.drop(name)
        if entry is None or entry.is_preload:
            return []
        return [self._store(entry)]

    def flush_all(self) -> List:
        """Empty the record at the end of a section.

        Only writebacks with a *global* home are emitted.  A shared-memory
        temporary still holding its result in registers is dropped, on the
        grounds that a section's temporaries have no readers left once the
        section is over -- true only while nothing after the last contraction
        reads one, which is exactly what a pointwise consumer would do.
        """
        out = []
        for name, entry in self.items():
            if entry.is_preload or entry.home.stype != SymbolType.Global:
                self.drop(name)
                continue
            out.append(self._store(entry))
            self.drop(name)
        return out

    def _store(self, entry: ResidencyEntry):
        if entry.home.stype == SymbolType.Global:
            shift = entry.shift or [0] * entry.home.data_view.rank()
            return StoreRegToGlb(context=self._context,
                                 src=entry.image,
                                 dest=entry.home,
                                 num_threads=self._num_threads,
                                 lead_width=self._lead_width,
                                 atomic=entry.atomic,
                                 dest_offset=shift,
                                 dest_bbox=entry.promise,
                                 zero_fill=entry.promise is not None)
        return StoreRegToShr(context=self._context,
                             src=entry.image,
                             dest=entry.home,
                             shr_mem=self._shr_mem,
                             num_threads=self._num_threads,
                             lead_width=self._lead_width)

    def _put(self, entry: ResidencyEntry, name: str) -> ResidencyEntry:
        # One entry per name, and replacing is the normal case: a later
        # operation on the same tensor supersedes what the earlier one left.
        # Keeping the geometry in the same record as the symbols is what stops
        # a replacement from updating one and not the other.
        self._entries[name] = entry
        return entry
