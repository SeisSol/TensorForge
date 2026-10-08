# SPDX-FileCopyrightText: 2015 SeisSol Group
#
# SPDX-License-Identifier: MIT
# SPDX-FileContributor: David Schneller

"""TensorForge pseudo-IR --- the IR a section of a kernel is built in.

    from tensorforge.backend import pir

    b = pir.IRBuilder(fptype=Datatype.F32, context=ctx)
    ...                                   # build with b
    body = b.finish()
    pir.verify(body)
    body = pir.optimize(body)
    pir.emit(body, writer, ctx)

For the shared-memory question: a body *names* its buffers and reasons about
aliasing on them (``Access``/``MemSpace``), and decides at the end of its
pipeline where each one goes (``allocate``) and where the threads meet around
them (``barriers``) -- once the order of the accesses is final, which is what
both decisions depend on.  ``Op.ALLOC`` carries the answer: a shared buffer
has no offset until the allocator gives it one.
"""

from .core import (ANY_EFFECT, BOOL, INDEX, SIZE, TOKEN, Access, BufferType, Effect,
                   IRError, MemSpace, Op, Operand, Region, ScalarType, Stmt,
                   TokenType, Value, accesses_conflict, collect_accesses,
                   collect_effect, def_use, defined_within, dump, free_values,
                   may_alias, walk)
from .asyncmem import (check_tokens, place_commits, schedule_async,
                       strip_commits)
from .build import IRBuilder, access_of
from .passes import cluster_loads, flatten_scopes, if_convert, pressure, cse, dce, fold, licm, load_cse, substitute, verify
from .barriers import Arena
from .pipeline import (BodyContext, MoveLoads, PlaceBarriers, PlaceBuffers,
                       Prefetch, ShardLoads, WrapLoads, optimize,
                       standard_pipeline)
from .emit import Emitter, emit

__all__ = [
    'ANY_EFFECT', 'BOOL', 'INDEX', 'SIZE', 'TOKEN', 'Access', 'Arena', 'BodyContext', 'BufferType', 'Effect',
    'Emitter', 'IRBuilder', 'IRError', 'MemSpace', 'Op', 'Operand', 'Region',
    'ScalarType', 'Stmt', 'TokenType', 'Value', 'access_of',
    'accesses_conflict', 'check_tokens', 'collect_accesses', 'collect_effect',
    'cse', 'dce', 'cluster_loads', 'flatten_scopes', 'if_convert', 'pressure', 'def_use', 'defined_within', 'dump', 'emit', 'fold',
    'free_values', 'licm', 'load_cse', 'may_alias', 'optimize', 'place_commits',
    'schedule_async', 'standard_pipeline', 'strip_commits',
    'substitute', 'verify', 'MoveLoads', 'PlaceBarriers', 'PlaceBuffers',
    'Prefetch', 'ShardLoads', 'WrapLoads',
    'walk',
]
