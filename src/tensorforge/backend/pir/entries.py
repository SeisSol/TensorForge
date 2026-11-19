# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT

"""Pseudo-IR: whether the batch loop carries an element's entry of the
launch's index list from one iteration to the next.

The loop reads the entry once, at the top of the element, and every address
through the list takes it (`BatchLoop._bind_entries`); where the wrap reads
the next element's entry ahead, the loop may carry that read into the next
iteration in place of the one at the top (`wrap._Rewrite._carry`).  Carried,
the entry is held from the top of the body to the first statement that reads
it.  The rule (`cheap_to_hold`): carried where that stretch stays clear of the
body's pressure peak, and otherwise read at the top of the element, where the
addresses through the list always stood.

* Why carry: through a pointer per cell the addresses of an element wait for
  the entry, the pointer and the data in turn, and carried, the first of the
  three is there when the iteration starts.  SeisSol's `volume` at order 4 in
  single precision and its `neighboringFlux[0]` in double, both through a
  pointer per cell, read their entries at 55 and 70 % of their bodies' peaks
  and, carried, fit under GB200's occupancy cliff again: 167 and 166
  registers against 175 and 170 read in each address, and 29 and 25 % less
  time there.
* Why not everywhere: where the first reader lies past the peak -- the
  element's own output, loaded behind the products that fill it -- carrying
  bought nothing on GH200 and lost.  `localFlux[0]` at order 6 in single
  precision was 4.7 % slower carried, and 4.3 % slower with the read moved
  down to its reader, where the compiler hoisted it back up but formed the
  address late.  Read at the top of the element it compiles to the machine
  code of an entry read in each address, and runs as fast.

What the losers lose is not registers -- the same count, the same
instructions -- but time waiting on memory.  On sm_90, ptxas puts the head's
read of the next element's entry (or the pointer through it) on the same
dependency barrier as the element's output loads, so the first use of the
next entry -- the address of the next element's prefetch, in the middle of
the body -- waits for the output as well, and the prefetch issues a memory
round trip late.  An entry read at the top of each element is used there,
which settles that barrier before the output loads join it.  The pressure
profile does not see this: `neighboringFlux[0]` at order 6 in single
precision, with I and dQ through the cell's pointer, reads its entry at 73 %
of its peak, is carried, and loses 4.8 % on GH200 the same way.

*The peak* is the stretch where the body's pressure (`passes.pressure_profile`)
comes within a tenth of its maximum (`SCARCE`), not the one statement at the
maximum: `localFlux[0]` at order 6 loads its output three statements into a
climb to 97 % of its peak, which it reaches thirty statements later.  Where
the body is lowered as an explicit vector, the profile counts each array
whole from its declaration on, and so stays within a tenth of its maximum
from the top of the body: there the rule rarely carries an entry.

*The first reader* is what reads the entry other than to bind an address from
it.  The bindings that take it (`GetElementPtr`) are where an address is
spelled and not where it is followed, so they are looked through, as is
anything that binds an address from them in turn -- the select a merged run
picks a member with.  A binding named in the text of a statement that is no
comment is read there, and a store names its destination by the binding's
symbol.
"""

from __future__ import annotations

from typing import Optional, Sequence

from .build import _names_in
from .core import BufferType, Stmt, Value, walk_stmts
from .passes import pressure_profile

#: The share of a body's peak pressure from which on its registers are
#: scarce: a value held through a stretch that comes this close to the peak is
#: held where every register counts.  The kernels that decided it are named
#: in the module's note: entries read at 55 and 70 % of their peaks, worth
#: carrying, and one read at 97 %, which is not.
SCARCE = 0.9


def is_entry(s: Stmt) -> bool:
    """Whether `s` reads an element's entry of the index list
    (`BatchLoop._bind_entries`), or a copy of such a read for another element
    (`pir.ahead.Clone`)."""
    return s.attr('element_index') is not None


def _binds_address(s: Stmt) -> bool:
    """A declaration of an address: a binding, or a select among bindings."""
    return (bool(s.attr('decl')) and bool(s.target)
            and all(isinstance(v.type, BufferType) for v in s.target))


def _code(s: Stmt) -> str:
    """The text of `s` that can name a value, which a comment cannot."""
    text = s.text or ''
    return '' if text.lstrip().startswith('//') else text


def _reads(s: Stmt, values: set, names: set) -> bool:
    """Whether `s` reads one of `values`: as an operand, as the base of an
    access, or by one of `names` -- in its text, or as an argument that is the
    symbol a binding is declared as, which is how a store names its
    destination (`_drop_unread` reads it the same way)."""
    if any(v.id in values for v in s.operands()):
        return True
    if any(isinstance(a.base, Value) and a.base.id in values
           for a in s.accesses):
        return True
    if names and any(not isinstance(a, Value)
                     and str(getattr(a, 'name', '')) in names
                     for a in s.args):
        return True
    text = _code(s)
    return bool(text) and any(_names_in(text, n) for n in names)


def first_reader(stmts: Sequence[Stmt], entry: Stmt) -> Optional[Stmt]:
    """The first of `stmts`, in the order they run, that reads `entry`'s
    value other than to bind an address from it -- looking through the
    bindings, and what binds from those; None where nothing does."""
    values = {v.id for v in entry.target}
    names = {str(entry.attr('extern'))} if entry.attr('extern') else set()
    for x in walk_stmts(tuple(stmts)):
        if x is entry or not _reads(x, values, names):
            continue
        if _binds_address(x):
            values.update(v.id for v in x.target)
            if x.attr('extern'):
                names.add(str(x.attr('extern')))
            continue
        return x
    return None


class Pressure:
    """A body's pressure at each of its statements (`passes.pressure_profile`),
    and where on it the registers are scarce (`SCARCE`): worked out once for
    every entry of a loop."""

    def __init__(self, body: Sequence[Stmt], explicit_simd: bool = False):
        profile = pressure_profile(tuple(body), in_bytes=True,
                                   explicit_simd=explicit_simd)
        self._at = {id(s): i for i, (s, _) in enumerate(profile)}
        self._curve = [b for _, b in profile]
        self._scarce = SCARCE * max(self._curve, default=0)

    def clear(self, end: Stmt) -> bool:
        """Whether the pressure stays short of the scarce stretch from the
        top of the body through `end`."""
        if self._scarce <= 0:
            return True
        j = self._at.get(id(end))
        if j is None:
            return False
        return max(self._curve[:j + 1]) < self._scarce


def cheap_to_hold(body: Sequence[Stmt], entry: Stmt,
                  explicit_simd: bool = False,
                  pressure: Optional[Pressure] = None) -> bool:
    """Whether holding `entry` from the top of `body` to its first reader in
    `body` keeps it clear of the body's peak.  Not where nothing in `body`
    reads it: a value nobody reads is held for nothing at all."""
    reader = first_reader(body, entry)
    if reader is None:
        return False
    if pressure is None:
        pressure = Pressure(body, explicit_simd)
    return pressure.clear(reader)
