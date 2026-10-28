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
"""

import re
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from tensorforge.common.basic_types import Datatype

#: Whether the emitter writes runs as blocks at all.  Off, each call is
#: `hip.h`'s, which waits by itself.
ENABLED = True

#: The calls a block takes, with the step and the row of each.
_CALL = re.compile(r'^tensorforge::fmacdpp(4|16)<(\d+)>$')

#: Products in one block at the most.  Every operand of a block is a register
#: at once: 36 double-precision products with their accumulators and sources
#: assemble, 64 find no registers.  A block of 32 pays a sixteenth of the
#: waits one per call does.
MAX_RUN = 32

#: The wait in front of a block: two wait states.
WAIT = 's_nop 1'

_INSTRUCTION = {Datatype.F32: 'v_fmac_f32_dpp', Datatype.F64: 'v_fmac_f64_dpp'}


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


def runs(body: Sequence, start: int) -> int:
    """Where the run of members starting at `body[start]` ends: unguarded
    calls a block can take, at most `MAX_RUN` of them -- and none that
    broadcasts an accumulator the run has written, which would read it right
    behind the write inside the block; it starts the next one, behind a wait
    of its own."""
    end = start
    written = set()
    while (end < len(body) and end - start < MAX_RUN
           and body[end].predicate is None and member(body[end]) is not None
           and _key(body[end].args[1]) not in written):
        written.add(_key(body[end].args[0]))
        end += 1
    return end


def render(run: Sequence, operand: Callable) -> str:
    """`run` as one `asm` statement, `operand` spelling a value.

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

    lines = [WAIT]
    for stmt in run:
        step, row = member(stmt)
        acc, src, mul = stmt.args
        control = (f'row_newbcast:{row}' if step == 16
                   else f'quad_perm:[{row},{row},{row},{row}]')
        lines.append(f'{_INSTRUCTION[_base(acc)]} %{number(acc)}, '
                     f'%{number(src)}, %{number(mul)} {control} '
                     f'row_mask:0xf bank_mask:0xf bound_ctrl:1')
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
