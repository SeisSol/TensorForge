# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Which parts of one descriptor list are the same part, repeated.

A body that states the same operation once per step -- a recursion unrolled by
the frontend, a face loop written out -- carries its repetition only in the
fact that the descriptors look alike.  Finding it is the search half of the
question :mod:`tensorforge.analysis.antiunify` answers: that module says
whether given bodies are one body, this one says which slices of a list to ask
about.

The search is cheap for a reason worth stating.  Two bodies generalize exactly
when their skeletons are equal -- the comparison has no third outcome -- so a
run of chunks can be found by computing one skeleton per chunk and testing
equality, and the generalization itself is built once, for the run that wins.

A chunk is a contiguous slice, and everything in it takes part: a fence sitting
between two operations is a descriptor like any other, with no slots and an
attribute that has to match.  So a fence inside a repeating chunk is carried
into the repetition, and a fence that falls in different places in different
chunks breaks the run rather than being quietly dropped.  That is the intended
reading -- what a fence orders is not something a substitution may change.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from tensorforge.analysis.antiunify import (Generalization, Skeleton,
                                            _attrs, _identity, _slots,
                                            anti_unify, operand_key,
                                            skeleton)
from tensorforge.generators.descriptions import OperationDescription


@dataclass(frozen=True)
class Repeat:
    """A maximal run of consecutive chunks that are one chunk apart.

    ``period`` is in descriptors and ``count`` in chunks, so a run covers
    ``period * count`` descriptors starting at ``start``.  ``general`` holds
    the common chunk and, per repetition, what fills its holes -- which is
    exactly what a loop over the run would need: one body and a table indexed
    by the counter.
    """

    start: int
    period: int
    count: int
    general: Generalization

    @property
    def span(self) -> int:
        return self.period * self.count

    @property
    def stop(self) -> int:
        return self.start + self.span

    @property
    def arity(self) -> int:
        """How many operands the run varies over.

        Zero is the interesting degenerate case: a run that repeats with
        nothing changing is the same computation done several times, which is
        a question for the frontend and not something to roll into a loop.
        """
        return self.general.arity


def _chunk_skeletons(descrs: Sequence[OperationDescription],
                     period: int) -> Dict[int, Skeleton]:
    """The skeleton of every chunk of the given period, keyed by start."""
    out: Dict[int, Skeleton] = {}
    limit = len(descrs) - period
    for start in range(limit + 1):
        out[start] = skeleton(descrs[start:start + period])[0]
    return out


_MOD = (1 << 61) - 1
_BASE = 1_000_003


class _Chunks:
    """Chunk comparisons for one descriptor list, at every period.

    Equal skeletons are the test, and building one per chunk and period was
    the search's whole cost: every period up to half the list, every start,
    every operand key again -- quadratic in the period, and over an hour for
    SeisSol's damage step (1787 operations) before anything was generated.

    A skeleton's shape and keys come one descriptor at a time, so two chunks
    can only be equal where their strings of per-descriptor signatures are.
    Those are compared in constant time with prefix hashes; the skeleton,
    which adds how the slots share tensors, is built only for chunks that
    pass, once each.  Equal is still exactly equal skeletons: a hash
    collision is caught by the skeletons, and equal strings hash alike.
    """

    def __init__(self, descrs: Sequence[OperationDescription]):
        self._descrs = descrs
        ids: Dict[Tuple, int] = {}
        n = len(descrs)
        self._prefix = [0] * (n + 1)
        self._power = [1] * (n + 1)
        for i, descr in enumerate(descrs):
            slots = _slots(descr)
            signature = (_attrs(descr), tuple(role for role, _ in slots),
                         tuple(operand_key(view) for _, view in slots))
            code = ids.setdefault(signature, len(ids) + 1)
            self._prefix[i + 1] = (self._prefix[i]
                                   + code * self._power[i]) % _MOD
            self._power[i + 1] = self._power[i] * _BASE % _MOD
        self._skeletons: Dict[Tuple[int, int], Skeleton] = {}

    def skeleton(self, start: int, period: int) -> Skeleton:
        key = (start, period)
        if key not in self._skeletons:
            self._skeletons[key] = skeleton(
                self._descrs[start:start + period])[0]
        return self._skeletons[key]

    def same(self, a: int, b: int, period: int) -> bool:
        # Both hashes brought to the same power of the base: no inverse needed.
        ha = (self._prefix[a + period] - self._prefix[a]) * self._power[b]
        hb = (self._prefix[b + period] - self._prefix[b]) * self._power[a]
        if ha % _MOD != hb % _MOD:
            return False
        return self.skeleton(a, period) == self.skeleton(b, period)


def _identities(descrs: Sequence[OperationDescription],
                start: int, period: int, groups: List[List[int]]) -> Tuple:
    """What each slot group of one chunk names, in the chunk's own terms."""
    views = [view for descr in descrs[start:start + period]
             for _, view in _slots(descr)]
    return tuple(_identity(views[group[0]], group[0]) for group in groups)


def _runs_at(descrs: Sequence[OperationDescription], period: int,
             min_count: int, max_arity: Optional[int],
             chunks: Optional[_Chunks] = None) -> List[Tuple[int, int]]:
    """Maximal ``(start, count)`` runs of equal chunks at one period.

    Maximal in both directions: a run is reported from where it begins, and a
    start that merely sits inside a longer run is not reported again.  Without
    that a run of six would also be found as five, four and three, and the
    caller would have to undo it.

    Equal skeletons are the whole legality test, and it is a weaker condition
    than it first looks: two contractions of the same shapes over entirely
    unrelated tensors have equal skeletons, because the only thing that
    distinguishes them is which tensors they name, which is what a hole is
    for.  So a run found this way can be a genuine repetition or an accident
    of shape, and the two are told apart by what the run *costs* -- one hole
    or six.  ``max_arity`` is where a caller states how much it is willing to
    pay; without it every operation of one shape joins one run.
    """
    if chunks is None:
        chunks = _Chunks(descrs)
    runs: List[Tuple[int, int]] = []
    start = 0
    while start + period * min_count <= len(descrs):
        count = 1
        if max_arity is None:
            while (start + period * (count + 1) <= len(descrs)
                   and chunks.same(start + period * count, start, period)):
                count += 1
        else:
            groups = seen = None
            while start + period * (count + 1) <= len(descrs):
                nxt = start + period * count
                if not chunks.same(nxt, start, period):
                    break
                if groups is None:
                    groups = chunks.skeleton(start, period).groups()
                    seen = [{ident} for ident in
                            _identities(descrs, start, period, groups)]
                widened = [s | {ident} for s, ident in
                           zip(seen, _identities(descrs, nxt, period, groups))]
                if sum(1 for s in widened if len(s) > 1) > max_arity:
                    break
                seen = widened
                count += 1
        if count >= min_count:
            runs.append((start, count))
            start += period * count
        else:
            start += 1
    return runs


def find_repeats(descrs: Sequence[OperationDescription],
                 min_count: int = 2,
                 max_period: Optional[int] = None,
                 max_arity: Optional[int] = None) -> List[Repeat]:
    """The repeated runs of a descriptor list, longest first, non-overlapping.

    Candidates are ranked by how much of the list they cover, then by how few
    holes that costs, then by the finer period, then by position.  Covering
    more is worth more because it is what a loop saves, and among equal
    coverage the run that varies in less is the one that shares more.
    Position breaks the remaining ties and is what makes the result of this
    function depend on nothing but its input -- a run picked by iteration
    order would move the generated code between two runs of the generator.
    """
    if min_count < 2:
        raise ValueError('a run repeats at least twice')

    ceiling = len(descrs) // min_count
    if max_period is not None:
        ceiling = min(ceiling, max_period)

    candidates: List[Repeat] = []
    table = _Chunks(descrs)
    for period in range(1, ceiling + 1):
        for start, count in _runs_at(descrs, period, min_count, max_arity,
                                     table):
            chunks = [list(descrs[start + period * i:start + period * (i + 1)])
                      for i in range(count)]
            general = anti_unify(chunks)
            if not general:
                continue
            candidates.append(Repeat(start, period, count, general))

    candidates.sort(key=lambda r: (-r.span, r.arity, r.period, r.start))

    taken: List[Repeat] = []
    covered: List[Tuple[int, int]] = []
    for run in candidates:
        if any(run.start < other_stop and other_start < run.stop
               for other_start, other_stop in covered):
            continue
        covered.append((run.start, run.stop))
        taken.append(run)

    taken.sort(key=lambda r: r.start)
    return taken
