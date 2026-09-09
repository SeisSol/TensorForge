# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""One symbol, one program.

A kernel name is a C++ identifier that has to keep two kernels apart whenever
their bodies differ.  Deriving it from a hand-picked list of properties makes
that a maintenance obligation: any property that reaches the generated code
and is not on the list produces two different programs under one name, and
what happens next depends on the linker.  Where duplicate symbols are refused
the build stops; where they are tolerated -- separate shared objects,
``-z muldefs``, translation units that never meet the same invocation -- one
kernel's launcher binds the other's body, and the numbers are quietly wrong.

:class:`~tensorforge.generators.generator.Generator` therefore derives the
name from the generated source itself, which is the one input that cannot be
incomplete.  This module is the check on that: every generation registers its
name together with the source behind it, and a name arriving twice with two
different sources raises.  It is cheap, and it is what turns any future gap --
a truncated digest colliding, a name pinned by hand for two kernels, a naming
scheme that goes back to deriving names from properties -- into a precise
error naming both kernels, at generation time, instead of a link error or a
wrong result.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, List, Optional

from tensorforge.common.exceptions import GenerationError


class KernelNameCollision(GenerationError):
  """Two different kernels reached one symbol name."""


@dataclass(frozen=True)
class _Entry:
  digest: str
  length: int
  descriptors: str


def _digest(source: str) -> str:
  return hashlib.sha256(source.encode()).hexdigest()


class KernelRegistry:
  """Every name generated in this process, with the source behind it.

  Only a digest of the source is kept, plus the descriptor list for the error
  message: a build generating thousands of kernels should not hold every
  emitted body until it exits.
  """

  def __init__(self):
    self._seen: Dict[str, _Entry] = {}

  def register(self, name: str, source: str,
               descriptors: Optional[List] = None) -> None:
    """Record ``name``; raise if it is taken by a different source."""
    entry = _Entry(digest=_digest(source),
                   length=len(source),
                   descriptors='\n'.join(f'{d}' for d in (descriptors or [])))
    previous = self._seen.get(name)
    if previous is None:
      self._seen[name] = entry
      return
    if previous.digest == entry.digest:
      return
    raise KernelNameCollision(
        f'two different kernels are called `{name}`.\n'
        f'  first:  {previous.length} chars, sha256 {previous.digest[:12]}\n'
        f'{_indent(previous.descriptors)}'
        f'  second: {entry.length} chars, sha256 {entry.digest[:12]}\n'
        f'{_indent(entry.descriptors)}'
        f'Whatever separates these two kernels does not reach the name. '
        f'Both bodies cannot carry this symbol: one of them would be dropped '
        f'by the routine cache, or bound to the other\'s launcher.')

  def names(self) -> List[str]:
    return sorted(self._seen)

  def clear(self) -> None:
    self._seen.clear()

  def __len__(self) -> int:
    return len(self._seen)


def _indent(text: str) -> str:
  if not text:
    return ''
  return ''.join(f'            {line}\n' for line in text.splitlines())


_REGISTRY = KernelRegistry()


def registry() -> KernelRegistry:
  """The process-wide registry every generation reports to."""
  return _REGISTRY
