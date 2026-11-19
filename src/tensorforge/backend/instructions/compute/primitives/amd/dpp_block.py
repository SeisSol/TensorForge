# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Runs of fused DPP FMAs, written as one block of inline assembly on gfx9.

`fmacdpp{4,16}<row>` is inline assembly (`hip.h` says why), and the hazard
recognizer does not look into assembly.  On gfx9 a DPP instruction reads its
first source two wait states after a VALU write of it at the earliest, and
nothing keeps the compiler from writing it right in front: a reload of the
value from the AGPR it was spilled to.  SeisSol's F64 `gpu_derivative` at
order 6 on gfx942 had 36 of them, and 315 of its 1872 values came out wrong.

Each call waits by itself in `hip.h` now, which cost that derivative 38 % and
the order-8 `volume` 36 %.  A run of them written as one block waits once.
What the block reads is in its registers before it starts, and nothing can be
scheduled into it, so its first instruction is the only one that can follow
a write too closely.  Over the order of the products the emitter decides
nothing: the run is written as the body has it.

The same blindness has a second side: what the block writes, the hazard
recognizer does not see written by a VALU instruction either, and an MFMA
reads a VGPR two wait states after a VALU write of it at the earliest (LLVM:
`LegacyVALUNotDotWritesVGPRWaitStates`).  With the operator products
reassociated (`Options.reassociate_products`), SeisSol's elastic
`derivative` at order 5 in single precision fed an accumulator of a block
straight into the MFMA behind it, and came out 7 % off on MI300A and 15 %
on MI210 in its largest number -- every run alike; 1.4e-7 with a wait behind
the block, nothing changed with more in front.  So a block waits behind
itself too (`WAIT_AFTER`).

Which of these waits a hazard can need is a question of what the generator
knows, and it knows the inside of a block and the kernel as a whole, not the
order the compiler puts the rest in (`Options.amd_dpp_waits`):

- The wait in front stays.  Anything may stand in front of a block -- the
  producer of its first broadcast source, a copy, a reload -- and nothing tells
  the generator what.  It covers more than that source besides: an F64 FMA
  wants two wait states behind a double-precision MFMA (LLVM:
  `DMFMAToFMA64WaitStates`), which the recognizer does not see in assembly
  either.
- The wait behind a block is for a reader of what it wrote that the recognizer
  misses, and on CDNA 2 and 3 that takes a matrix instruction.  The readers
  that want waits behind a VALU write are a DPP instruction's first source
  (two wait states), a lane read (`v_readlane`, one, gfx94x),
  `v_permlane*_swap` (two, gfx950) and an MFMA (two); hipcc (ROCm 7.2 and
  7.14) waits behind a block in front of the first three as it does behind a
  VALU instruction -- it counts any writer there, assembly included -- and in
  front of an MFMA not, as it asks whether the writer is a VALU instruction.
  It asks the same of the write in front of a double-precision MFMA that
  stands between it and a memory instruction reading it on gfx90a
  (`DMFMABetweenVALUWriteVMEMRead`).  So a block waits behind itself in a
  kernel that issues a matrix instruction somewhere (`waits_behind`), and
  nowhere else.  Somewhere: the order of the kernel's code is the compiler's,
  and an MFMA of one section may come to stand behind a block of another.
- A broadcast of an accumulator the run wrote stays in the run: inside one
  block every instruction is the emitter's, and it waits in front of that
  broadcast only where the write is closer than two wait states
  (`_schedule`).  A run may go on past its 32nd product too, one wait in front
  instead of two (`Options.amd_dpp_block_operands`) -- but every operand of a
  block is a register at once.

Over SeisSol's elastic kernels at orders 4, 6 and 8 and viscoelastic ones at 4
and 6, in both precisions at 16 and 32 lanes, that leaves 10048 of the 16770
waits in blocks on gfx90a, gfx942 and gfx950 alike: half of those of the
double-precision kernels, none of the single-precision ones, which issue
MFMAs.  Registers, scratch and occupancy stay what they were, and hipcc's
listing has no hazard left, inline assembly included.
"""

import re
from typing import (Callable, Dict, Iterable, Iterator, List, Optional,
                    Sequence, Tuple)

from tensorforge.backend.pir.engines import Hazard
from tensorforge.common.basic_types import Datatype
from tensorforge.common.options import DPP_WAITS

#: Whether the emitter writes runs as blocks at all.  Off, each call is
#: `hip.h`'s, which waits by itself.
ENABLED = True

#: How a block waits (`Options.amd_dpp_waits`): `needed`, where a hazard can
#: be; `always`, in front of and behind every block, with a run cut at
#: `MAX_RUN` products and in front of a broadcast of what it wrote.  Anything
#: but `needed` is `always`.
WAITS = DPP_WAITS

#: The calls a block takes, with the step and the row of each.
_CALL = re.compile(r'^tensorforge::fmacdpp(4|16)<(\d+)>$')

#: Products in one block at the most, unless `Options.amd_dpp_block_operands`
#: lets a run go on.  Every operand of a block is a register at once: 36
#: double-precision products with their accumulators and sources assemble, 64
#: find no registers.  A block of 32 pays a sixteenth of the waits one per
#: call does.
MAX_RUN = 32

#: Wait states from a VALU write of a VGPR to a DPP instruction that reads it
#: as its first source.
WAIT_STATES = 2

#: The wait in front of a block: two wait states.
WAIT = 's_nop 1'

#: The wait a VALU write owes a matrix instruction that reads it as its A or
#: B, in the engines' terms: two wait states between the write and an MFMA
#: (LLVM's `LegacyVALUNotDotWritesVGPRWaitStates`; measured on gfx942,
#: 2026-11-15).  The compiler keeps it behind a VALU instruction and not
#: behind assembly, so a block keeps it itself, where a kernel issues one
#: (`waits_behind`).  The other way the hardware interlocks.
HAZARD = Hazard(then='mfma', states=2)

#: The wait behind a block, for a VALU write of its an MFMA reads next.
WAIT_AFTER = f's_nop {HAZARD.states - 1}'

_INSTRUCTION = {Datatype.F32: 'v_fmac_f32_dpp', Datatype.F64: 'v_fmac_f64_dpp'}

#: A matrix instruction in text the IR does not see into, assembly or raw:
#: `mfma`, `smfmac`, WMMA.
_MATRIX = re.compile(r'mfma|smfmac|wmma', re.IGNORECASE)


def blocks_for(model: Optional[str]) -> bool:
    """Whether runs are written as blocks for the architecture `model`.

    CDNA2 on (gfx90a, gfx94x, gfx950): gfx9 has the wait to keep, and these
    spell the row broadcast `row_newbcast`, as `hip.h` does.  From gfx10 on
    the hardware keeps it (LLVM: `hasNoDataDepHazard`), and the calls stay
    calls the compiler may schedule one by one.
    """
    if not ENABLED or not model or not model.startswith('gfx9'):
        return False
    try:
        number = int(model[3:], 16)
    except ValueError:
        return False
    return number >= 0x90a


def member(stmt) -> Optional[Tuple[int, int]]:
    """`(step, row)` where `stmt` is a fused DPP FMA a block can take: one of
    `hip.h`'s calls on scalars of a type it has a fused form for."""
    callee = stmt.attr('callee') if hasattr(stmt, 'attr') else None
    found = _CALL.match(callee or '')
    if found is None or stmt.target or len(stmt.args) != 3:
        return None
    step, row = int(found.group(1)), int(found.group(2))
    base = _base(stmt.args[0])
    if base not in _INSTRUCTION or (step == 4 and base != Datatype.F32):
        return None
    if any(getattr(getattr(a, 'type', None), 'length', 1) not in (None, 1)
           for a in stmt.args):
        return None
    return step, row


def _base(value) -> Optional[Datatype]:
    return getattr(getattr(value, 'type', None), 'base', None)


def _key(value):
    """What tells two operands apart: the value's id, or a literal's text."""
    ident = getattr(value, 'id', None)
    return ('value', ident) if ident is not None else ('literal', repr(value))


def runs(body: Sequence, start: int, waits: str = 'always',
         operands: int = 0) -> int:
    """Where the run of members starting at `body[start]` ends: unguarded
    calls a block can take.

    `always`: at most `MAX_RUN` of them, and none that broadcasts an
    accumulator the run has written, which would read it right behind the
    write inside the block; it starts the next one, behind a wait of its own.
    `needed`: a broadcast of what the run wrote stays in it -- `render` waits
    in front of it where the write is too close -- and the run goes on past
    `MAX_RUN` while it has no more distinct operands than `operands`.
    """
    end = start
    written = set()
    seen = set()
    while (end < len(body) and body[end].predicate is None
           and member(body[end]) is not None):
        stmt = body[end]
        if waits == 'needed':
            keys = {_key(value) for value in stmt.args}
            if end - start >= MAX_RUN and len(seen | keys) > operands:
                break
            seen |= keys
        elif end - start >= MAX_RUN or _key(stmt.args[1]) in written:
            break
        written.add(_key(stmt.args[0]))
        end += 1
    return end


def _schedule(run: Sequence) -> Iterator[Tuple[str, object]]:
    """The block in order: `('wait', n)` for an `s_nop n`, `('fma', stmt)`
    for a product.

    The wait in front, and one in front of a product that broadcasts an
    accumulator an earlier product of the block wrote fewer than
    `WAIT_STATES` before -- counted in wait states, an `s_nop n` being
    `n + 1`.  Its other operands want none: they are read where any VALU
    operand is, and a block adds into one accumulator back to back.
    """
    yield 'wait', 1
    at = 2
    wrote: Dict = {}
    for stmt in run:
        source = _key(stmt.args[1])
        if source in wrote and at - wrote[source] - 1 < WAIT_STATES:
            states = WAIT_STATES - (at - wrote[source] - 1)
            yield 'wait', states - 1
            at += states
        yield 'fma', stmt
        wrote[_key(stmt.args[0])] = at
        at += 1


def waits_in(run: Sequence) -> int:
    """`s_nop`s `render` writes into `run` in front of a product: the one in
    front of the block and any between its products."""
    return sum(1 for kind, _ in _schedule(run) if kind == 'wait')


def waits_behind(bodies: Iterable[Sequence]) -> bool:
    """Whether a block of a kernel made of `bodies` waits behind itself: where
    any statement of them is the matrix instruction its write owes a wait
    (`HAZARD`) -- a matrix op of that engine -- or assembly or raw text
    spells a matrix instruction."""
    from tensorforge.backend.pir.core import Op, walk_stmts
    for body in bodies:
        for stmt in walk_stmts(body):
            if (stmt.op == Op.MATRIX_MAC
                    and stmt.attr('engine') == HAZARD.then):
                return True
            if any(_MATRIX.search(text) for text in (stmt.attr('asm') or '',
                                                     stmt.text or '')):
                return True
    return False


def render(run: Sequence, operand: Callable, after: bool = True) -> str:
    """`run` as one `asm` statement, `operand` spelling a value, `after`
    whether it waits behind itself.

    Each accumulator is a read-write operand, listed once however often the
    run adds into it, and each source an input, once; outputs first, since
    the assembler numbers both in one sequence.
    """
    outputs: Dict = {}
    order: List = []
    for stmt in run:
        acc = stmt.args[0]
        if _key(acc) not in outputs:
            outputs[_key(acc)] = len(order)
            order.append(acc)
    inputs: Dict = {}
    sources: List = []
    for stmt in run:
        for value in stmt.args[1:]:
            if _key(value) in outputs or _key(value) in inputs:
                continue
            inputs[_key(value)] = len(order) + len(sources)
            sources.append(value)

    def number(value) -> int:
        return outputs.get(_key(value), inputs.get(_key(value)))

    lines = []
    for kind, item in _schedule(run):
        if kind == 'wait':
            lines.append(WAIT if not lines else f's_nop {item}')
            continue
        step, row = member(item)
        acc, src, mul = item.args
        control = (f'row_newbcast:{row}' if step == 16
                   else f'quad_perm:[{row},{row},{row},{row}]')
        lines.append(f'{_INSTRUCTION[_base(acc)]} %{number(acc)}, '
                     f'%{number(src)}, %{number(mul)} {control} '
                     f'row_mask:0xf bank_mask:0xf bound_ctrl:1')
    if after:
        lines.append(WAIT_AFTER)
    template = '\n'.join(f'"{line}\\n\\t"' for line in lines)
    # Early-clobbered: a block writes an accumulator before its last product
    # reads its inputs, and an output without `&` may share its register with
    # an input the compiler takes to hold the same value at the start.  On
    # gfx942 a guarded load (`g ? x : 0.0`) shared one with an accumulator
    # just zeroed, and SeisSol's order-6 `localFluxAll` read the partial sum
    # as that multiplicand in 192 of its 504 values.  A single call reads all
    # three before it writes, which is why `hip.h` needs none.
    outs = ', '.join(f'"+&v"({operand(v)})' for v in order)
    ins = ', '.join(f'"v"({operand(v)})' for v in sources)
    return f'asm({template}\n: {outs}\n: {ins}\n);'
