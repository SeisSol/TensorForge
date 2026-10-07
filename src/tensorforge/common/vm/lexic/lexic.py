# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
from abc import ABC, abstractmethod
from typing import Dict

from tensorforge.common.basic_types import Datatype
from tensorforge.common.operation import Operation

#: The operations every C-like spelling writes with an operator.
INFIX = {
  Operation.ADD: '+', Operation.SUB: '-', Operation.MUL: '*',
  Operation.DIV: '/', Operation.XOR: '^',
  Operation.LT: '<', Operation.LE: '<=', Operation.GT: '>',
  Operation.GE: '>=', Operation.EQ: '==', Operation.NEQ: '!=',
}

#: The C++ standard library's mathematical functions, one name for every
#: floating-point type.  CUDA and HIP declare them for device code as well.
#: `fmin` and `fmax` are the IEEE minimum and maximum, which return the other
#: operand where one is a NaN.
STD_MATH = {
  Operation.ABS: 'std::fabs({0})',
  Operation.MIN: 'std::fmin({0}, {1})',
  Operation.MAX: 'std::fmax({0}, {1})',
  Operation.POW: 'std::pow({0}, {1})',
  Operation.GAMMA: 'std::tgamma({0})',
  **{op: f'std::{name}({{0}})' for op, name in (
    (Operation.ERF, 'erf'), (Operation.EXP, 'exp'), (Operation.LOG, 'log'),
    (Operation.EXPM1, 'expm1'), (Operation.LOG1P, 'log1p'),
    (Operation.SQRT, 'sqrt'), (Operation.CBRT, 'cbrt'),
    (Operation.SIN, 'sin'), (Operation.COS, 'cos'), (Operation.TAN, 'tan'),
    (Operation.ASIN, 'asin'), (Operation.ACOS, 'acos'),
    (Operation.ATAN, 'atan'), (Operation.SINH, 'sinh'),
    (Operation.COSH, 'cosh'), (Operation.TANH, 'tanh'),
    (Operation.ASINH, 'asinh'), (Operation.ACOSH, 'acosh'),
    (Operation.ATANH, 'atanh'))},
}

#: The types `Lexic.INTEGER_MATH` is asked for.
_INTEGERS = frozenset({Datatype.BOOL, Datatype.I8, Datatype.I16, Datatype.I32,
                       Datatype.I64, Datatype.U32, Datatype.SIZE})
_UNSIGNED = frozenset({Datatype.U32, Datatype.SIZE})

class Lexic(ABC):
  """How a statement the generator has decided on is written in one language.

  Spelling only: whether a target has an instruction, how wide it is or how
  far it reaches is `common.target.Target`'s question, and a method here is
  reached only once that was answered.
  """

  def __init__(self, underlying_hardware):
    self._underlying_hardware = underlying_hardware
    self.thread_idx_x = None
    self.thread_idx_y = None
    self.thread_idx_z = None
    self.block_dim_x = None
    self.block_dim_y = None
    self.block_dim_z = None
    self.block_idx_x = None
    self.block_idx_z = None
    self.grid_dim_x = None
    self.stream_type = None
    self.restrict_kw = None

  def storage_class(self, space) -> str:
    """How a declaration says where its object resides, where it has to.

    Empty for almost everything, and for two different reasons that are worth
    keeping apart.  Most spaces need no annotation on a declaration because the
    declaration's *form* already fixes them -- a kernel argument is a kernel
    argument.  `MemSpace.PARAM` is the one that does not: CUDA otherwise copies
    a by-value parameter into per-thread memory as soon as its address is taken
    or a runtime value indexes it, and `__grid_constant__` is how that copy is
    refused.  Backends whose kernel arguments already live in a broadcast space
    -- which is most of them -- need nothing and say so by returning nothing.

    A space and not a keyword the caller looks up, so that the one site asking
    the question asks it about residency rather than about a vendor.
    """
    return ''

  def pointer_type(self, elem: str, space=None, readonly: bool = False,
                   restrict: bool = False, const: bool = False,
                   depth: int = 1) -> str:
    """How a pointer into `space` is spelled, for a declaration of one.

    Asked of the backend rather than assembled from `restrict_kw` at the call
    site, because on one of them the space and the restrict promise are not
    independently spellable: HIP puts the space in an attribute on the pointee
    and ships `SpacePtrRestrict` as the fused alias, so a caller concatenating
    two strings gets something that does not compile.  Both facts arrive here
    together and the backend decides.

    The default is the generic pointer, which is what every backend without
    address spaces in its type system wants.

    `readonly` is about the pointee and `const` about the pointer: a binding
    the pipelined form advances is `T *` and the ordinary one `T *const`, and
    both may point at something nothing writes.

    `depth` is the indirection.  Above one the space stops applying: an array
    of pointers in global memory holds pointers whose *pointee* is what the
    space describes, and the one being declared here only holds them.

    `space` is a `pir.MemSpace`, taken structurally so this module does not
    have to import the IR.
    """
    lhs = 'const ' if readonly else ''
    ptr = 'const' if const else ''
    qual = f' {self.restrict_kw}' if restrict and self.restrict_kw else ''
    return f'{lhs}{elem} {"*" * depth}{ptr}{qual}'

  def shared_pointer_type(self, elem: str, restrict: bool = False) -> str:
    """The declarator for a window into the shared arena, without the name.

    Separate from `pointer_type` and not a call into it, because the two
    spell it differently -- `float*` here where `pointer_type` writes
    `float *` -- and a whitespace change here is a change to every snapshot on
    every backend for no reason anybody reading the diff could recover.

    What the hook buys is the question, not the answer: a target where a
    shared address is not a pointer can say so in one place rather than
    having four call sites format `{elem}*` at it.
    """
    tail = f' {self.restrict_kw}' if restrict and self.restrict_kw else ''
    return f'{elem}*{tail}'

  def shared_window_expr(self, arena: str, offset) -> str:
    """How a window `offset` elements into the shared arena is spelled.

    `&arena[offset]` wherever a shared address is a pointer, which is every
    backend but one.  Asked rather than formatted because the exception does
    not have pointers into that space at all: an ESIMD address is an offset,
    and taking the address of a subscript of one is neither meaningful nor
    ill-formed enough to be caught -- `&s0[i]` on a proxy is a diagnostic on
    a good day and a pointer to a temporary otherwise.
    """
    return f'&{arena}[{offset}]'

  def shared_window_retype(self, window: str, elem: str) -> str:
    """`window` as a window of `elem` at the same address.

    For a buffer of another element than the arena's -- the boolean a
    comparison writes.  A pointer cast wherever a window is a pointer.
    """
    return f'reinterpret_cast<{elem}*>({window})'

  def batch_source(self, name: str) -> str:
    """The array a pointer-based operand's elements are read through in the
    kernel body: the operand itself, wherever the body can dereference what
    the caller passed."""
    return name

  def get_slm_load(self, elem: str, width: int, address: str) -> str:
    """A vector read of shared memory, where that is its own instruction.

    None where a shared address is a pointer and the ordinary load spelling
    already covers it, so a caller tests rather than always substituting.
    """
    return None

  def get_slm_store(self, elem: str, width: int, address: str,
                    value: str) -> str:
    return None

  @abstractmethod
  def get_launch_code(self, func_name, grid, block, stream, func_params):
    pass

  @abstractmethod
  def set_shmem_size(self, func_name, shmem):
    pass

  @abstractmethod
  def declare_shared_memory(self, name, precision, size=None):
    pass

  @abstractmethod
  def kernel_definition(self, file, kernel_bounds, base_name, params, precision=None,
                        total_shared_mem_size=None, global_symbols=None,
                        lanes=None):
    pass

  @abstractmethod
  def sync_block(self):
    pass

  @abstractmethod
  def sync_simd(self):
    pass

  def loop_body_fence(self) -> str:
    """A statement for the head of a loop body that loads cannot be hoisted
    across, or empty where the backend's compiler needs none.

    For the batch loop without a mask.  With one, the body sits under
    `if (allowed)`, and a load there runs only conditionally, so LICM may not
    speculate it into the preheader.  Without one the body runs every
    iteration and LICM is free to -- and it takes loads that are invariant by
    design, the operators staged into shared memory once per block, and holds
    them in registers across the whole loop.
    """
    return ''

  def handoff_fence(self):
    """What orders one lane's store before the other lanes' load, where the
    rendezvous itself is spelled as nothing -- or None where it is not.

    A barrier that costs nothing because the lanes are in lockstep is still no
    barrier to the compiler.  One lane storing a value and every lane reading
    it back from the same address is, seen by the compiler, one thread storing
    under a condition and then loading: it may forward the stored value where
    the store ran and hoist the load above it for the rest, which then read
    what was there before.  Where `sync_simd` already is an instruction
    (`__syncwarp`, a sub-group barrier), it orders memory as well, and nothing
    more is needed.
    """
    return None

  def sync_mult(self, num_threads: int, hw):
    """Rendezvous exactly the ``num_threads`` threads of one multiplication.

    Only reached where the multiplication is *wider* than a wave -- narrower
    than that and `SyncThreads` asks for `sync_simd` directly, because the
    threads are in lockstep anyway.  So the default is the honest one: a whole
    block, which over-synchronizes but never deadlocks, and which is exact
    where the target has no narrower rendezvous (`Target.sync_mult`), because
    the policy then puts one multiplication in a block.

    The opportunity a vendor can take here is a sub-block rendezvous, which
    lets several wide multiplications share a block:

    * NVIDIA, `bar.sync id, count` -- meets exactly `count` threads at named
      barrier `id`.
    * AMD from gfx12.5, `s_barrier_init` / `s_barrier_join` on barrier objects
      1..16, whose expected count is in waves.
    * Intel under ESIMD, `named_barrier_signal` / `named_barrier_wait`, whose
      counts are in sub-groups.

    The catch is the same on all three: every multiplication in the block
    needs an `id` of its own, so it holds only while `mults_per_block` stays
    under the number of barrier resources.  Two multiplications sharing an
    `id` rendezvous with each other, which is a deadlock the moment they run
    the body a different number of times.
    """
    return self.sync_block()

  def exchange_xor(self, variable, mask):
    """`variable` as held by the lane whose index differs from this one's in
    the bits of `mask` -- one step of a butterfly -- or None where the target
    has an all-reduce of its own to call instead (`reduction`)."""
    return None

  @abstractmethod
  def kernel_range_object(self, name, values):
    pass

  @abstractmethod
  def get_stream_via_pointer(self, file, stream_name, pointer_name):
    pass

  @abstractmethod
  def get_headers(self):
    pass

  #: The functions this spelling calls, by the operation they compute: a
  #: template over the operands `{0}` and `{1}`, the suffix `{f}` the C math
  #: library gives a `float` function (`f`, and nothing for any other type),
  #: and the type `{t}`.  What has neither an entry nor an operator is refused
  #: (`get_operation`), and not substituted: a function that exists under one
  #: library's name and not another's is a numerics question, not a spelling.
  MATH: Dict[Operation, str] = {}
  #: Where an integer takes another function than a floating-point number.
  #: `fmin`, `fmax` and `fabs` take an integer converted to `double`, which a
  #: 64-bit one does not survive.
  INTEGER_MATH: Dict[Operation, str] = {}

  def get_operation(self, op: Operation, fptype, value1, value2):
    """`op` over `value1` and `value2` as an expression in `fptype`.

    `value2` is `''` for a unary operation.  The operators are the same in
    every C-like spelling and are written here once; the functions differ by
    library and come from `MATH`, or from `INTEGER_MATH` for an integer.
    """
    if op == Operation.COPY:
      return value1
    if op == Operation.NEG:
      return f'(-{value1})'
    if op == Operation.RCP:
      return f'(1 / {value1})'
    if op == Operation.ABS and fptype in _UNSIGNED:
      # Its own absolute value, where `abs` is ambiguous between the
      # overloads for the signed types.
      return value1
    template = None
    if fptype in _INTEGERS:
      template = self.INTEGER_MATH.get(op)
    if template is None:
      template = self.MATH.get(op)
    if template is not None:
      return template.format(value1, value2,
                             f='f' if fptype == Datatype.F32 else '',
                             t=fptype)
    if op in INFIX:
      return f'({value1} {INFIX[op]} {value2})'
    boolean = fptype == Datatype.BOOL
    if op == Operation.NOT:
      return f'(!{value1})' if boolean else f'(~{value1})'
    if op == Operation.AND:
      return f'({value1} {"&&" if boolean else "&"} {value2})'
    if op == Operation.OR:
      return f'({value1} {"||" if boolean else "|"} {value2})'
    raise NotImplementedError(f'{op}')

  def vector_fma(self, a, b, c):
    """`a * b + c` over the target's vector type, spelled as a call, or
    None where the infix form already says everything.

    It does for scalars everywhere and for GNU vectors, which contract it.
    A target whose vector type has no fused operation of its own -- CUDA's
    structs, whose paired FMA needs an intrinsic -- names its function here.
    """
    return None

  def reduction(self, variable, optype, fptype, block, subblock=1):
    """An all-reduce of `variable` across `block` lanes, in groups of
    `subblock`.

    Declared here so a backend that has no answer says so.  `CudaLexic`
    implements it and `HipLexic` inherits that; a backend without it would
    meet a cross-lane reduction as an `AttributeError` --- a missing attribute
    reads as a typo, and this is a missing feature.

    Not implemented for SYCL because the signature is the open question, not
    the body.  `sycl::reduce_over_group` takes a whole group and has no
    `subblock`, so it answers only for `subblock == 1` and
    `block == sub_group_size`; anything else is a hand-built exchange over
    `permute_group_by_xor`.  Under ESIMD the model is different again --- the
    vector is explicit and the reduction is an operation on `simd<T, N>`
    rather than a cross-lane construct.  Committing to a spelling before that
    is decided would fix the wrong shape in place.
    """
    raise NotImplementedError(
        f'{type(self).__name__} has no cross-lane reduction; see '
        f'Lexic.reduction')

  # --- global accesses ------------------------------------------------------
  # `datatype` is keyword-only and has no default, so a call site cannot omit
  # it and a positional `nontemporal` cannot land in it.  The hint is spelled
  # by an overload set on at least one target, which makes the type part of
  # the question rather than decoration on it: what an unanswered type buys is
  # not a plainer access but a kernel that does not compile.  So `nontemporal`
  # is passed only where `Target.nontemporal` agreed for the type.

  def glb_store(self, lhs, rhs, *, datatype, length=1, nontemporal=False):
    return f'{lhs} = {rhs};'

  def glb_load(self, rhs, *, datatype, length=1, nontemporal=False):
    return f'{rhs}'

  # --- atomic accumulation --------------------------------------------------
  # Declared here so that every backend answers through one interface, the one
  # the single call site uses.  Declared per backend, the interfaces drift
  # apart, and a target whose override takes other arguments, or that has
  # none, reaches this path with a TypeError or an AttributeError.
  #
  # `ctx` is a parameter and not a field because the lexic is constructed with
  # the vendor alone (`Target` passes `hw.vendor`), while the spelling turns
  # on the architecture: which builtin names the instruction.

  def atomic_store(self, ctx, access, variable, op, datatype, length=1):
    """One atomic update, as a statement.

    Only reached when `Target.native_atomic` agreed, so a backend overriding
    this does not have to answer for the cases that one turns away.
    """
    raise NotImplementedError(
        f'{type(self).__name__} has no atomic store; see Lexic.atomic_store')

  # --- asynchronous global -> shared copies --------------------------------
  # A backend without a hardware path returns None; the caller then emits a
  # synchronous fallback, so correctness never depends on these being present.
  # All three are *per thread*: every lane copies `nbytes` bytes, one of
  # `Target.copy_async_sizes`.

  def copy_async(self, dst, src, nbytes, zfill: int = 0):
    """`zfill` trailing bytes are written as zero instead of copied.

    The hardware reads `nbytes - zfill` from `src` and fills the rest, so a run
    that does not end on a whole access is still moved by one and the source is
    never read past its end.  A backend without such an operand returns None
    for a non-zero `zfill`, and the caller narrows rather than silently copying
    whatever follows the source.
    """
    return None

  def commit_async(self):
    return None

  def wait_async(self, prior):
    """Wait until at most `prior` issued copies are still in flight."""
    return None

  def wait_async_regs(self, prior):
    """Same, for global -> register loads.

    Separate from `wait_async` because the two are not the same counter
    everywhere: AMD tracks both in `vmcnt`, while NVIDIA scoreboards register
    loads in hardware and needs no instruction at all (hence None).
    """
    return None

  # --- data prefetch --------------------------------------------------------
  # A hint and nothing else: it moves no value, releases no token, and the
  # emitter drops it on a target that has none (`Target.prefetch`).

  def prefetch_runs(self, addresses, byte_counts, level='l2'):
    """Several hints as one statement, or None where each is its own.

    `addresses[i]` is a pointer expression and `byte_counts[i]` how far past
    it the hint reaches.  Only a target whose one instruction can name
    several addresses answers.
    """
    return None

  def prefetch(self, address, *, datatype, elems=1, level='l2'):
    """One prefetch, as a statement.  `address` is a pointer expression.

    `level` is `'l1'` or `'l2'`, and it is a request rather than an
    instruction: a target with one prefetch for both honors neither, and
    says so here instead of pretending the distinction survived.  `elems` is
    likewise honored only where the instruction takes a count.

    Only reached when `Target.prefetch` agreed, so an implementation does not
    have to answer for the targets that one turns away.
    """
    return None
