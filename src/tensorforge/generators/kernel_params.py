# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""One kernel parameter, and the surfaces it has to appear on.

The signature was assembled four times: with types for the kernel prototype,
without them for the call into it, with types and defaults for the launcher,
and without them for the call site.  Four passes over the same symbols,
building four strings, and the only thing keeping them in step was that they
read the same loop.  A parameter that renders itself is the same list read four
ways instead.

What that buys beyond the duplication is a place to ask the question.  The
declaration used to be concatenated here from `addr2ptr_type`, a `const`, and a
literal `'size_t'`, which is why `ocl_lexic.kernel_definition` is `pass`: an
address space has to go *inside* the declaration, and by the time the backend
saw the parameter list it was a finished string.  A parameter carrying its
space can be spelled by whoever knows how.
"""

from dataclasses import dataclass
from typing import Optional, Union

from tensorforge.backend.pir.core import MemSpace
from tensorforge.common.basic_types import Addressing, Datatype, DataFlowDirection


@dataclass(frozen=True)
class KernelParam:
    """A single parameter of the generated kernel and of its launcher.

    `depth` is the indirection: 0 passes the thing itself, 1 a pointer to it, 2
    a pointer to an array of pointers -- which is `Addressing.PTR_BASED`, where
    the space belongs to the pointer that is *loaded* rather than to the one
    passed in, and where `ptr_manip` puts it.

    `decl` is an escape for a parameter that already knows how to declare
    itself.  `DeclareOperandTable` is the one: its by-value struct, that
    struct's definition and this declaration are three renderings of a type
    only it has, and pulling them apart is its own step rather than a detour
    in this one.
    """

    name: str
    #: What spells the element type: a `Datatype`, or the spelling itself
    #: where the interface has one of its own.
    datatype: Union[Datatype, str, None] = None
    space: MemSpace = MemSpace.GLOBAL
    readonly: bool = False
    depth: int = 0
    default: str = ''
    decl: Optional[str] = None

    # -- the surfaces ----------------------------------------------------- #

    def declaration(self, lexic, with_default: bool = False,
                    host: bool = False) -> str:
        """With types, for a prototype.

        `host` is the launcher's prototype, which host code calls with the
        pointers it has.  Those are generic, and on HIP a space-qualified
        pointer is a type of its own that a generic one reaches only through a
        cast: `float *` does not initialise a `SpacePtr<float, 1>` parameter.
        So the launcher declares the generic spelling, and casts into a local
        of the kernel's type before it calls the kernel (`binding`).
        """
        tail = self.default if with_default else ''
        if self.decl is not None:
            return f'{self.decl}{tail}'
        if self.depth == 0:
            const = 'const ' if self.readonly else ''
            body = f'{const}{self.datatype}'
        else:
            body = lexic.pointer_type(f'{self.datatype}',
                                      None if host else self.space,
                                      readonly=self.readonly,
                                      depth=self.depth)
        storage = '' if host else lexic.storage_class(self.space)
        storage = f'{storage} ' if storage else ''
        return f'{storage}{body} {self.name}{tail}'

    def _kernel_type(self, lexic) -> Optional[str]:
        """The kernel's pointer spelling, where the launcher's differs."""
        if lexic is None or self.decl is not None or self.depth == 0:
            return None
        kernel = lexic.pointer_type(f'{self.datatype}', self.space,
                                    readonly=self.readonly, depth=self.depth)
        host = lexic.pointer_type(f'{self.datatype}', None,
                                  readonly=self.readonly, depth=self.depth)
        return None if kernel == host else kernel

    def binding(self, lexic) -> Optional[str]:
        """The launcher's local of the kernel's type, where the two differ.

        A named local rather than a cast in the argument list: a cooperative
        launch hands the arguments over by address (`argsPtrs(Args &...)`),
        and a cast has none -- `barrier_two_gemms_16x16` stopped compiling.
        """
        kernel = self._kernel_type(lexic)
        if kernel is None:
            return None
        return f'{kernel} {self.name}Arg = ({kernel}){self.name};'

    def argument(self, lexic=None) -> str:
        """Without types, for a call.

        With `lexic`, for the launcher's call into the kernel, which names
        the local `binding` declares where there is one.
        """
        return (self.name if self._kernel_type(lexic) is None
                else f'{self.name}Arg')

    # -- construction ----------------------------------------------------- #

    @classmethod
    def of_symbol(cls, symbol, datatype) -> 'KernelParam':
        """The parameter a data operand is passed as."""
        addressing = symbol.obj.addressing
        readonly = symbol.obj.direction == DataFlowDirection.SOURCE
        if addressing == Addressing.SCALAR:
            return cls(symbol.name, datatype, MemSpace.NONE, readonly, depth=0)
        return cls(symbol.name, datatype, MemSpace.GLOBAL, readonly,
                   depth=len(Addressing.addr2ptr_type(addressing)))

    @classmethod
    def size(cls, name: str) -> 'KernelParam':
        """An element count or an element offset.

        `Datatype.SIZE`, and the same one for both.  An extra offset is added to
        `batchId0 * stride`, so a narrower type caps what a *caller* can
        express -- 2^32-1 elements is 17.2 GB into an f32 buffer and 34.4 GB
        into an f64 one, both reachable on a current card, and a caller past
        them loses the high bits at the call site, which is a wrong answer
        rather than a diagnostic.  The arithmetic was never the problem: the
        product is already 64-bit and the unsigned offset promotes into it.
        What was capped is what the signature can carry.
        """
        return cls(name, Datatype.SIZE, MemSpace.NONE, depth=0)

    @classmethod
    def flags(cls, name: str, default: str = '') -> 'KernelParam':
        """The per-element mask.

        Spelled rather than typed, and the spelling is the interface's: callers
        pass `unsigned*`, and `Datatype.U32` would render the same type under a
        different name in a header they already include.  It is still a global
        pointer here, which is the part a backend with address spaces needs and
        a bare string would not have carried.
        """
        return cls(name, 'unsigned', MemSpace.GLOBAL, readonly=False,
                   depth=1, default=default)

    @classmethod
    def table(cls, table) -> 'KernelParam':
        return cls(table.name, space=MemSpace.PARAM, decl=table.parameter())

    @classmethod
    def opaque(cls, decl: str, name: str, default: str = '') -> 'KernelParam':
        """A parameter whose type this module has nothing to say about."""
        return cls(name, decl=f'{decl} {name}', default=default)
