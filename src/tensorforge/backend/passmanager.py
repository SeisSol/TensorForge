# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT

"""Running a list of passes over an IR, for both levels.

Three things are explicit here:

*Analyses vs. transforms.*  An analysis derives a fact and stores it under a
name; a transform rewrites the IR.  A transform invalidates every analysis
unless it declares otherwise, so a stale ``live_map`` cannot be read by
accident -- which matters because ``live_map`` is keyed by *instruction
index*, and any transform that inserts or removes an instruction silently
reinterprets every key.

*Declared dependencies.*  A pass names the facts it consumes; the manager
checks that an earlier pass provides each of them, and that none of them was
invalidated on the way.  A missing dependency is an error at registration,
not an ``AttributeError`` halfway through code generation.

*Verification between passes.*  Under ``ir_debug`` the context checks the IR
after every pass, so a diagnostic names the pass that introduced it.

What an IR is and how it is checked is the context's business: the macro
stream (`opt.manager.StreamContext`) and the pseudo-IR body
(`pir.pipeline.BodyContext`) answer that differently, and the manager asks
the same question of both.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence, Set

from tensorforge.common.exceptions import GenerationError


class PassContext:
    """The IR a pipeline rewrites, plus what is known about it.

    Passes reach for named facts instead of being handed positional
    arguments, which would let ``live_map`` and ``regions`` drift apart from
    the stream they describe.  A fact is an analysis result or a property of
    the IR a transform established (`flat`, `scheduled`); both are invalidated
    alike by the next transform that does not say it preserves them.
    """

    def __init__(self, extra: Dict[str, Any] = None):
        self._analyses: Dict[str, Any] = {}
        self.extra: Dict[str, Any] = dict(extra or {})

    # -- analysis cache ---------------------------------------------------- #

    def get(self, name: str) -> Any:
        if name not in self._analyses:
            raise GenerationError(
                f'analysis {name!r} requested but not available; the pass '
                f'should declare it in `requires`')
        return self._analyses[name]

    def has(self, name: str) -> bool:
        return name in self._analyses

    def put(self, name: str, value: Any) -> None:
        self._analyses[name] = value

    def invalidate(self, keep: Iterable[str] = ()) -> None:
        keep = set(keep)
        for name in list(self._analyses):
            if name not in keep:
                del self._analyses[name]

    # -- what the IR answers ----------------------------------------------- #

    def check(self, stage: str, debug: str) -> None:
        """Check the IR after `stage` -- `'build'` before the first pass -- and
        dump it where `debug` asks for that.  Called only under `ir_debug`."""

class Pass:
    """Base class.

    ``name``       identifier used in dependency lists and logs
    ``requires``   facts this pass reads
    ``provides``   facts this pass establishes: an analysis result, or a
                   property of the IR a transform leaves behind
    ``preserves``  facts that survive this pass (transforms only)
    """

    name: str = ''
    requires: Sequence[str] = ()
    provides: Sequence[str] = ()
    preserves: Sequence[str] = ()
    is_transform: bool = False

    def enabled(self, pc: PassContext) -> bool:
        return True

    def run(self, pc: PassContext) -> None:
        raise NotImplementedError


class PassManager:
    def __init__(self, debug: str = '', given: Sequence[str] = (),
                 delivers: Sequence[str] = ()):
        self._passes: List[Pass] = []
        #: What the `ir_debug` option said, passed in rather than read here:
        #: a pipeline is built per generation and the options belong to the
        #: context that asked for it.
        self._debug = debug
        #: Facts the IR has on entry, which the passes may require without an
        #: earlier pass providing them -- the output of another pipeline.
        self._given = tuple(given)
        #: Facts whoever consumes the result relies on.  Checked once the last
        #: pass has run: a pass appended behind the one that provides such a
        #: fact invalidates it, and nothing downstream would notice.
        self._delivers = tuple(delivers)

    def add(self, p: Pass) -> 'PassManager':
        available: Set[str] = set(self._given)
        for earlier in self._passes:
            available.update(earlier.provides)
        missing = [r for r in p.requires if r not in available]
        if missing:
            raise GenerationError(
                f'pass {p.name!r} requires {missing} but no earlier pass '
                f'provides it; registered so far: '
                f'{[q.name for q in self._passes]}')
        self._passes.append(p)
        return self

    @property
    def passes(self) -> List[Pass]:
        return list(self._passes)

    def run(self, pc: PassContext) -> None:
        missing = [r for r in self._given if not pc.has(r)]
        if missing:
            raise GenerationError(
                f'the pipeline expects {missing} of its input, and the input '
                f'does not have it')
        if self._debug:
            pc.check('build', self._debug)
        for p in self._passes:
            if not p.enabled(pc):
                continue
            missing = [r for r in p.requires if not pc.has(r)]
            if missing:
                raise GenerationError(
                    f'pass {p.name!r} needs {missing}, invalidated by an '
                    f'earlier transform and not recomputed')
            p.run(pc)
            if p.is_transform:
                pc.invalidate(keep=tuple(p.preserves) + tuple(p.provides))
            if self._debug:
                pc.check(p.name, self._debug)
        missing = [r for r in self._delivers if not pc.has(r)]
        if missing:
            raise GenerationError(
                f'the pipeline ends without {missing}, which its result is '
                f'expected to have: a pass after the one providing it '
                f'invalidated it')
