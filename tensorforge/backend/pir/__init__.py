# SPDX-FileCopyrightText: 2015 SeisSol Group
#
# SPDX-License-Identifier: MIT
# SPDX-FileContributor: David Schneller

"""TensorForge pseudo-IR --- the micro-IR inside a single instruction.

    from tensorforge.backend import pir

    b = pir.IRBuilder(fptype=Datatype.F32, context=ctx)
    ...                                   # build with b
    body = b.finish()
    pir.verify(body)
    body = pir.optimize(body)
    pir.emit(body, writer, ctx)

Layering, for the shared-memory question: this layer *names* buffers and
reasons about aliasing on them (``Access``/``MemSpace``); it never assigns an
offset.  Placement, lifetime and reuse of shared memory stay in
``backend.opt.mem_region_allocation`` / ``backend.opt.shr_mem_analyzer``, which
have the whole-kernel view that those decisions need.  ``Op.ALLOC`` is the
seam between the two.
"""

from .core import (ANY_EFFECT, BOOL, INDEX, TOKEN, Access, BufferType, Effect,
                   IRError, MemSpace, Op, Operand, Region, ScalarType, Stmt,
                   TokenType, Value, accesses_conflict, collect_accesses,
                   collect_effect, def_use, defined_within, dump, free_values,
                   may_alias, walk)
from .asyncmem import check_tokens, schedule_async
from .build import IRBuilder, access_of
from .passes import cse, dce, fold, licm, optimize, substitute, verify
from .emit import Emitter, emit

__all__ = [
    'ANY_EFFECT', 'BOOL', 'INDEX', 'TOKEN', 'Access', 'BufferType', 'Effect',
    'Emitter', 'IRBuilder', 'IRError', 'MemSpace', 'Op', 'Operand', 'Region',
    'ScalarType', 'Stmt', 'TokenType', 'Value', 'access_of',
    'accesses_conflict', 'check_tokens', 'collect_accesses', 'collect_effect',
    'cse', 'dce', 'def_use', 'defined_within', 'dump', 'emit', 'fold',
    'free_values', 'licm', 'may_alias', 'optimize', 'schedule_async',
    'substitute', 'verify',
    'walk',
]
