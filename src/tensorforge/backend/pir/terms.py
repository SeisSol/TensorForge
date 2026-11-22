# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Pseudo-IR: an emulated product as the matrix ops of its terms.

A matrix engine that multiplies in TF32 or BF16 still reaches F32 -- and,
with the integer kinds to come, F64 -- where the operands are taken apart
into terms whose sum is the operand (`split`), the terms are multiplied
pairwise on the engine, and the products are added up.  Which terms, which
of their products, in which order and into which accumulator is arithmetic
and not hardware, and until here every path wrote its own: NVIDIA's three
3xTF32 products as a literal, largest first; the AMD paths' six BF16 ones
from `split.products`, smallest first; Intel's from the same function
iterated the other way round; the asynchronous engines' from a function of
the stage.  Three orders, each stated where it was issued and nowhere else.

A `TermPlan` is that statement, once.  How each side is taken apart
(`Form`); the products, as `(A's term, B's term)` in the order they are
issued; which accumulator each adds into (`partials`), and how the partials
become D (`Epilogue`) -- one accumulator that is D, or several summed
behind the products (NVIDIA's chains below `CHAINS_MIN`, where each product
gets an accumulator of its own so that the next does not wait for it).  The
scaled sum of Ozaki's slices and the CRT of its residues are further
epilogues of the same shape, and further term kinds (`Kind`).

The products are `MATRIX_MAC`s of whatever engine the path issues on, each
carrying its plan and its term (`IRBuilder.matrix_mac`'s `plan`, `term`),
and an engine lowers them as it lowers any product: nothing here spells an
instruction.  Where the products interleave across accumulators -- term by
term over a step's tiles, which hides the latency of one HMMA behind the
others -- is the path's schedule and stays the path's (`TermPlan.steps`);
what each accumulator receives, and in what order, is the plan's, and
`check_plans` reads it back off a body: an accumulator that misses a
product of its plan, takes one twice or in another order, or multiplies a
term in another format than the plan's, is a finding.

What is not here is how a term comes about: the split at run time is each
target's routine, written as its registers want it, and the derived form an
operator is prepared in at generation is `common.matrix.forms`'.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import (Any, Callable, Dict, Iterator, List, Optional, Sequence,
                    Tuple)

from tensorforge.common.basic_types import Datatype

from .core import Op, Stmt, Value


class Kind(Enum):
    """What a side's terms are.

    `FLOAT`: values of a narrower float format whose sum is the operand --
    TF32 halves, BF16 terms -- multiplied by the engine as such and added in
    the accumulator's format.  The integer slices of Ozaki's first scheme
    (a power-of-two scale a row or column) and the residues of its second
    (the CRT behind the products) are kinds of their own, and come with the
    epilogues that undo them."""

    FLOAT = 'float'


@dataclass(frozen=True)
class Form:
    """How one side of an emulated product reaches the engine: `count`
    terms of `kind`, each multiplied in `element` -- the engine's operand
    format, `Datatype.TF32` for a TF32 half although the register holds a
    float (`split.MANTISSA` keeps the difference)."""

    element: Datatype
    count: int = 1
    kind: Kind = Kind.FLOAT

    def __post_init__(self):
        if self.count < 1:
            raise ValueError(f'a side of {self.count} terms')

    @property
    def significand(self) -> int:
        """Significand bits a term carries (`split.MANTISSA`)."""
        from tensorforge.backend.instructions.compute import split
        return split.MANTISSA[self.element]

    @property
    def covered(self) -> int:
        """Significand bits the terms carry together (`split.covered`)."""
        from tensorforge.backend.instructions.compute import split
        return split.covered(self.significand, self.count)

    def __repr__(self):
        return f'{self.count}x{self.element.name}'


class Epilogue(Enum):
    """How a plan's partials become D.

    `NONE`: one partial, which is D.  `SUM`: the partials added behind the
    products -- the corrections first, the partials past the leading one in
    their order, then that sum onto the leading partial (`TermPlan.close`):
    the small ones meet before the large one rounds them off, which is the
    order NVIDIA's chains below `CHAINS_MIN` sum them in."""

    NONE = 'none'
    SUM = 'sum'


@dataclass(frozen=True)
class TermPlan:
    """An emulated product: D = A B as the products of A's and B's terms.

    `products` are `(A's term, B's term)`, in the order they are issued;
    `partials` the accumulator each adds into, 0 the leading one, one entry
    a product; `epilogue` how the partials become D; `accumulator` the
    format they hold.  `name` says which plan it is where a body is read.

    Frozen and hashable: a `MATRIX_MAC` carries it as an attribute
    (`IRBuilder.matrix_mac`'s `plan`), and the attributes of a statement are
    part of its identity to the passes."""

    name: str
    a: Form
    b: Form
    products: Tuple[Tuple[int, int], ...]
    partials: Tuple[int, ...]
    epilogue: Epilogue = Epilogue.NONE
    accumulator: Datatype = Datatype.F32

    def __post_init__(self):
        if not self.products:
            raise ValueError(f'{self.name}: a plan of no products')
        if len(set(self.products)) != len(self.products):
            raise ValueError(f'{self.name}: a product issued twice')
        for i, j in self.products:
            if not (0 <= i < self.a.count and 0 <= j < self.b.count):
                raise ValueError(f'{self.name}: the product {(i, j)} of '
                                 f'{self.a!r} by {self.b!r}')
        if len(self.partials) != len(self.products):
            raise ValueError(f'{self.name}: {len(self.partials)} partials '
                             f'for {len(self.products)} products')
        if sorted(set(self.partials)) != list(range(self.count)):
            raise ValueError(f'{self.name}: partials {self.partials} are '
                             f'not 0 to {self.count - 1}, each taken')
        if (self.count == 1) != (self.epilogue is Epilogue.NONE):
            raise ValueError(f'{self.name}: {self.count} partial(s) under '
                             f'the epilogue {self.epilogue.value}')

    @property
    def count(self) -> int:
        """The partial accumulators the plan adds into."""
        return max(self.partials) + 1

    def term(self, index: int) -> Tuple[int, int]:
        """`(A's term, B's term)` of product `index`."""
        return self.products[index]

    def partial(self, index: int) -> int:
        """The accumulator product `index` adds into."""
        return self.partials[index]

    def sequence(self, partial: int) -> Tuple[Tuple[int, int], ...]:
        """The products accumulator `partial` receives, in their order: what
        `check_plans` holds each accumulator of a body to."""
        return tuple(t for t, p in zip(self.products, self.partials)
                     if p == partial)

    def steps(self) -> Iterator[Tuple[int, Tuple[int, int], int]]:
        """`(index, term, partial)` of every product, in the plan's order:
        for a path that interleaves the products of several accumulators,
        term by term over them, and issues each itself."""
        for index, (term, partial) in enumerate(zip(self.products,
                                                     self.partials)):
            yield index, term, partial

    def each(self) -> 'TermPlan':
        """The same products, each into an accumulator of its own and
        summed behind them (`Epilogue.SUM`): what NVIDIA's `mma.sync`
        issues where a step has too few independent chains to hide an
        HMMA's latency (`nvidia.CHAINS_MIN`)."""
        if self.count == len(self.products):
            return self
        return TermPlan(f'{self.name}/each', self.a, self.b, self.products,
                        tuple(range(len(self.products))), Epilogue.SUM,
                        self.accumulator)

    def issue(self, mac: Callable, a_terms: Sequence, b_terms: Sequence,
              accs: Sequence) -> List:
        """Every product of the plan, once, in its order, through `mac(a, b,
        c, term)`: A's term `a_terms[i]`, B's `b_terms[j]`, `c` the partial
        the product adds into (`accs[partial]`) and `term` its `(i, j)`.
        `mac` returns the new partial -- D, where the engine returns one --
        or None where it writes C in place.  Returns the partials."""
        accs = list(accs)
        if len(accs) != self.count:
            raise ValueError(f'{self.name}: {len(accs)} accumulators for '
                             f'{self.count} partials')
        for _, (i, j), p in self.steps():
            out = mac(a_terms[i], b_terms[j], accs[p], (i, j))
            if out is not None:
                accs[p] = out
        return accs

    def close(self, add: Callable, partials: Sequence):
        """D out of the partials, by the epilogue: the one partial under
        `NONE`; under `SUM` the corrections added in their order and that
        sum onto the leading partial, through `add(x, y)`."""
        partials = list(partials)
        if len(partials) != self.count:
            raise ValueError(f'{self.name}: {len(partials)} partials for '
                             f'{self.count}')
        if self.epilogue is Epilogue.NONE:
            return partials[0]
        correction = partials[1]
        for more in partials[2:]:
            correction = add(correction, more)
        return add(partials[0], correction)

    def __repr__(self):
        terms = ' '.join(f'({i},{j})' for i, j in self.products)
        if self.count == 1:
            where = 'one accumulator'
        else:
            where = (f'partials {"".join(str(p) for p in self.partials)}, '
                     f'{self.epilogue.value}')
        return (f'TermPlan({self.name}: {self.a!r} by {self.b!r} into '
                f'{self.accumulator.name}: {terms}; {where})')


def plan(name: str, a: Form, b: Form, *, keep: Optional[int] = None,
         order: str = 'smallest', major: str = 'a',
         accumulator: Datatype = Datatype.F32) -> TermPlan:
    """The plan of `a` by `b` whose products `split.products` prescribes --
    those above the accumulator's rounding (`keep`), in `order`, a tie
    broken by `major`'s term -- all into one accumulator.  `TermPlan.each`
    gives each its own."""
    from tensorforge.backend.instructions.compute import split
    pairs = split.products(max(a.count, b.count), keep,
                           counts=(a.count, b.count), order=order,
                           major=major)
    return TermPlan(name, a, b, pairs, (0,) * len(pairs), Epilogue.NONE,
                    accumulator)


# --------------------------------------------------------------------------- #
# Reading the plans back off a body
# --------------------------------------------------------------------------- #

def _lists(body: Sequence[Stmt]) -> Iterator[Sequence[Stmt]]:
    """Every statement list of `body`: itself, and each region's body, at
    every depth."""
    yield body
    for s in body:
        for region in s.regions:
            yield from _lists(region.body)


def _ids(values) -> Tuple:
    return tuple(v.id if isinstance(v, Value) else ('c', repr(v))
                 for v in values)


def _chain(s: Stmt, produced: Dict[int, Tuple]) -> Tuple:
    """The accumulator `s` adds into: its C where the engine writes it in
    place -- with the columns of tensor memory D starts at -- or, where it
    returns D, the chain C belongs to: D of a product before it in the same
    list, or C itself -- a value, or a variable each product's D is written
    back into."""
    from .engines import fragments
    c = fragments(s)[2]
    if s.attr('in_place'):
        return ('in place', _ids(c), s.attr('d_column'))
    found = (produced.get(c[0].id) if len(c) == 1 and isinstance(c[0], Value)
             else None)
    key = found if found is not None else ('from', _ids(c))
    for d in s.target:
        produced[d.id] = key
    return key


def check_plans(body: Sequence[Stmt]) -> List[str]:
    """What a body's emulated products do that their plans do not say.

    Every `MATRIX_MAC` that carries a plan (`TermPlan`) is a term of it and
    adds into an accumulator (`_chain`); each accumulator of a statement
    list has to receive its partial's products (`TermPlan.sequence`) in the
    plan's order, in whole runs -- a step of the contraction after another
    -- each of one plan, and multiply its terms in the plan's formats.  A
    product dropped, taken twice or out of its order, a product into
    another partial's accumulator, or a term the engine reads in another
    format than the plan states, is a finding.  Per statement list: a
    product of a run is never apart from the others of its run in a body,
    the term-major paths interleaving accumulators but not lists."""
    findings: List[str] = []
    for stmts in _lists(body):
        produced: Dict[int, Tuple] = {}
        chains: Dict[Tuple, List] = {}
        order: List[Tuple] = []
        for s in stmts:
            if s.op != Op.MATRIX_MAC:
                continue
            plan_ = s.attr('plan')
            if plan_ is None:
                continue
            term = s.attr('term')
            key = _chain(s, produced)
            if not isinstance(plan_, TermPlan):
                findings.append(f'a matrix op carries {plan_!r} as its plan')
                continue
            if term not in plan_.products:
                findings.append(f'{plan_.name}: a product of the term '
                                f'{term!r}, which it does not have')
                continue
            types = s.attr('types') or ()
            if (len(types) == 4 and (types[0], types[1], types[3]) != (
                    plan_.a.element, plan_.b.element, plan_.accumulator)):
                findings.append(
                    f'{plan_.name}: {s.attr("engine")} multiplies '
                    f'{types[0].name} by {types[1].name} into '
                    f'{types[3].name}, where the plan has '
                    f'{plan_.a.element.name} by {plan_.b.element.name} into '
                    f'{plan_.accumulator.name}')
            if key not in chains:
                chains[key] = [plan_, []]
                order.append(key)
            elif chains[key][0] != plan_:
                findings.append(f'one accumulator takes the products of '
                                f'{chains[key][0].name} and {plan_.name}')
                continue
            chains[key][1].append(term)
        for key in order:
            plan_, terms = chains[key]
            partial = plan_.partial(plan_.products.index(terms[0]))
            want = plan_.sequence(partial)
            whole = len(terms) // len(want)
            if (len(terms) % len(want) or tuple(terms) != want * whole):
                got = ' '.join(f'({i},{j})' for i, j in terms)
                findings.append(
                    f'{plan_.name}: an accumulator of partial {partial} '
                    f'takes {got}, where its runs are '
                    + ' '.join(f'({i},{j})' for i, j in want))
    return findings


__all__ = ['Epilogue', 'Form', 'Kind', 'TermPlan', 'check_plans', 'plan']
