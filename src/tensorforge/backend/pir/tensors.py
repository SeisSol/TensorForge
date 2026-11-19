# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT

"""Pseudo-IR: the descriptors of gfx1250's tensor copies (`engines`, `tdm`).

A tensor copy (`IRBuilder.copy_tensor`) moves a tile through a descriptor:
the tile's LDS and global addresses, its rows and the elements of a row, the
pitch of its rows in global memory and the size of its element -- words the
instruction reads from scalar registers (`hip.h`, `tensorDescriptorWords`).
The descriptor is a value of its own (`Op.TENSOR_DESC`), pure, which the
copy names (`desc`) rather than takes as an operand: a copy's operands are
its two buffers and their indices, as every copy's are, and the passes that
move, clone or redirect a copy read them so.

So it is made here, behind all of them (`abstract_instruction.
_optimize_shared_body`), right ahead of its copy and in the same statement
list, over what the copy names by then -- the window a stage of it is, the
indices a pass rewrote.  The emitter refuses a tensor copy that names no
descriptor.  The descriptor is not predicated: its words are the wave's,
computed by every wave and taken by the one that issues the copy.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Callable, List, Optional, Tuple

from .core import Effect, Op, Stmt, Uniformity, Value, walk_stmts
from . import engines


def tensor_copy(s: Stmt) -> bool:
    """Is `s` a copy through gfx1250's tensor data mover?"""
    return s.op == Op.COPY_ASYNC and s.attr('engine') == 'tdm'


def _descriptor(b, copy: Stmt) -> Stmt:
    """The descriptor `copy` goes through: over the buffers and indices it
    names, and its tile's extents and pitch."""
    elem = engines.element_of(None, copy.copy_dst)
    level = engines.TDM_TEAMS.get(copy.attr('team'), Uniformity.MULT)
    desc = b.value(engines.TensorDescType(elem), hint='tdesc', uniform=level)
    return Stmt(op=Op.TENSOR_DESC, target=(desc,), args=copy.args,
                pure=True, movable=True, effect=Effect.NONE,
                attrs=(('ndst', copy.attr('ndst', 0)),
                       ('shape', copy.attr('shape')),
                       ('pitch', copy.attr('pitch'))))


def tensor_descriptors(body: Tuple[Stmt, ...],
                       scratch: Callable[[], object],
                       report: Optional[list] = None) -> Tuple[Stmt, ...]:
    """`body` with a descriptor right ahead of every tensor copy that names
    none, which the copy then names (`desc`); `scratch` builds the values,
    numbered from the body's.  A body without such copies is returned as it
    is."""
    if not any(tensor_copy(s) and s.attr('desc') is None
               for s in walk_stmts(body)):
        return body
    b = scratch()

    def rewrite(stmts: Tuple[Stmt, ...]) -> Tuple[Stmt, ...]:
        out = []
        for s in stmts:
            if s.regions:
                s = replace(s, regions=tuple(replace(r, body=rewrite(r.body))
                                             for r in s.regions))
            if tensor_copy(s) and s.attr('desc') is None:
                made = _descriptor(b, s)
                out.append(made)
                s = replace(s, attrs=s.attrs + (('desc', made.target[0]),))
                if report is not None:
                    report.append(f'tdm: {s.target[0]} through '
                                  f'{made.target[0]}, a tile of '
                                  f'{s.attr("shape")} a pitch of '
                                  f'{s.attr("pitch")}')
            out.append(s)
        return tuple(out)

    return rewrite(body)


def _tile(s: Stmt) -> List[str]:
    """What is wrong with the tile a descriptor or a copy states."""
    shape, pitch = s.attr('shape'), s.attr('pitch')
    if not (isinstance(shape, tuple) and len(shape) == 2
            and all(isinstance(n, int) and 1 <= n <= engines.TDM_TILE_LIMIT
                    for n in shape)):
        return [f'{s.op}: a tile of {shape!r}, which is not two extents of '
                f'1 to {engines.TDM_TILE_LIMIT}']
    if not (isinstance(pitch, int)
            and shape[1] <= pitch <= engines.TDM_PITCH_LIMIT):
        return [f'{s.op}: a pitch of {pitch!r} for rows of {shape[1]}']
    return []


def verify_tensor(s: Stmt) -> List[str]:
    """What is wrong with `s`, a descriptor or a copy through the tensor
    data mover: a descriptor is one value over a destination, a source and
    a tile, reads no memory and is not predicated; a copy states its tile,
    is predicated on its first wave (`leader`), and names a descriptor where
    it names one."""
    diag: List[str] = []
    if s.op == Op.TENSOR_DESC:
        if len(s.target) != 1 or not isinstance(s.target[0].type,
                                                engines.TensorDescType):
            diag.append('tensor.desc: produces exactly one descriptor')
        if len(s.args) < 2:
            diag.append('tensor.desc: takes a destination and a source')
        if s.predicate is not None or s.accesses or s.effect != Effect.NONE:
            diag.append('tensor.desc: reads no memory and is not predicated')
        return diag + _tile(s)
    desc = s.attr('desc')
    if desc is not None and not (isinstance(desc, Value) and isinstance(
            desc.type, engines.TensorDescType)):
        diag.append(f'{s.op}: names {desc!r} for its descriptor')
    if s.predicate is None or s.attr('leader') is not True:
        diag.append(f'{s.op}: a tensor copy not issued by the first wave of '
                    f'its lanes')
    if s.attr('team') not in engines.TDM_TEAMS:
        diag.append(f'{s.op}: a tensor copy for the lanes of '
                    f'{s.attr("team")!r}')
    return diag + _tile(s)


__all__ = ['tensor_copy', 'tensor_descriptors', 'verify_tensor']
