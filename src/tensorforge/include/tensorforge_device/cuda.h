// SPDX-FileCopyrightText: 2026 SeisSol Group
//
// SPDX-License-Identifier: MIT
#ifndef SEISSOL_TENSORFORGE_INCLUDE_TENSORFORGE_DEVICE_CUDA_H_
#define SEISSOL_TENSORFORGE_INCLUDE_TENSORFORGE_DEVICE_CUDA_H_

#include <type_traits>
#include <utility>

#include "base.h"

#include <cooperative_groups.h>

#include <cuda/ptx>

namespace tensorforge {

// The participation mask for the `_sync` intrinsics: every lane.  `warpSize`
// is the *width* of a warp -- 32, i.e. the single bit 0x20 -- so as a mask it
// would name lane 5 alone, and no other lane would take part in the exchange.
inline constexpr unsigned FullWarpMask = 0xffffffffu;

// CUDA's own `float2`/`float4` are `__align__(8)`/`__align__(16)`, which is
// right for a global access and wrong for a private one: a register staging
// array is 4-byte aligned by every rule that applies to it, and casting an
// over-aligned type onto it is undefined however reliably the compiler has
// been getting away with it.  This is the same pair `hip.h` declares -- the
// natural width for a base that proves it, and an element-aligned twin for a
// base that does not.
/// A vector as a *struct*, not as a GNU `vector_size` typedef.
///
/// nvcc declines `vector_size` in device code outright --- as the type of a
/// value, and as the target of a cast, with "is a vector, which is not
/// supported in device code".  So on this target there is no spelling of the
/// typedef that reaches a kernel, and nothing here emits one.
///
/// The subscript is what makes this a drop-in: everything the generator emits
/// for a vector value --- `v[i]` to take a component, an assignment through a
/// cast to move one --- is spelled the same way for both, so only this
/// definition changes and no call site does.  `__align__` is what makes it a
/// *wide* access rather than N narrow ones: the natural pair is eight bytes
/// and the relaxed twin is element-aligned, exactly as the typedefs of
/// `hip.h` are.
///
/// Alignment is a *parameter*, not a second struct.  That is forced: under
/// `vector_size` the natural and the relaxed spelling are the same type
/// carrying different alignment attributes, so `*(VectorRelaxedT*)&buf[i] = v`
/// assigns a `VectorT` straight through; written as two unrelated structs the
/// same line stops compiling, which is precisely the trap `cuda_lexic` warns
/// about for `float2`.  One template with the alignment in the type keeps them
/// related, and the conversion below restores the assignment in both
/// directions.  It is a member-wise copy of a trivially copyable POD of
/// identical layout, so it costs nothing the `vector_size` spelling does not.
///
/// Elementwise arithmetic is not built in, as it is for a GNU vector, and is
/// supplied below: the lead-vectorized multilinear body spells its products as
/// `c + (a * b)` over these types, the same text it emits for HIP.  Lengths
/// CUDA's own structs do not have are not a special case here --- any N works
/// --- but a length the hardware has no wide access for simply compiles to N
/// narrow ones.
template <typename T, std::size_t N, std::size_t Align>
struct alignas(Align) VectorStruct {
  T data[N];
  __device__ __forceinline__ T &operator[](std::size_t i) { return data[i]; }
  __device__ __forceinline__ const T &operator[](std::size_t i) const {
    return data[i];
  }
  /// The same width at another alignment.  Never selected for `A == Align`:
  /// a conversion function to its own class type is never used.
  template <std::size_t A>
  __device__ __forceinline__ operator VectorStruct<T, N, A>() const {
    VectorStruct<T, N, A> out{};
#pragma unroll
    for (std::size_t i = 0; i < N; ++i) {
      out.data[i] = data[i];
    }
    return out;
  }
};

template <typename T, std::size_t N> struct VectorOf {
  typedef VectorStruct<T, N, N * sizeof(T)> type;
  typedef VectorStruct<T, N, sizeof(T)> relaxed;
};

template <typename T, std::size_t N>
using VectorT = typename VectorOf<T, N>::type;

/// An operand passed by value (`Residence.ARGUMENT`): its numbers, not their
/// address.  A struct so that the array stays a value -- an array parameter
/// decays to a pointer -- and so that it lives where kernel arguments live,
/// the constant bank; `__grid_constant__` on the parameter keeps it there when
/// the body takes an element's address.  It indexes like the pointer a
/// batch-constant operand in memory is, which is all the body asks of it.
/// `from` is the launcher's: it copies the host array the caller passed.
template <typename T, std::size_t N> struct ValueArray {
  T v[N];
  __device__ __forceinline__ const T &operator[](std::size_t i) const {
    return v[i];
  }
  static ValueArray from(const T *values) {
    ValueArray out{};
    for (std::size_t i = 0; i < N; ++i) {
      out.v[i] = values[i];
    }
    return out;
  }
};

template <typename T, std::size_t N>
using VectorRelaxedT = typename VectorOf<T, N>::relaxed;

/// Componentwise, as a GNU vector computes: the result takes the left
/// operand's alignment, and either side may be the relaxed twin.  Per
/// component and inlined, so `c + (a * b)` reaches the compiler as N scalar
/// multiplies feeding N scalar adds, which it contracts into N fused
/// multiply-adds exactly as it does for the scalar body.
#define TENSORFORGE_VECTOR_OP(OP)                                              \
  template <typename T, std::size_t N, std::size_t A, std::size_t B>           \
  __device__ __forceinline__ VectorStruct<T, N, A> operator OP(                \
      const VectorStruct<T, N, A> &x, const VectorStruct<T, N, B> &y) {        \
    VectorStruct<T, N, A> out;                                                 \
    _Pragma("unroll") for (std::size_t i = 0; i < N; ++i) {                    \
      out.data[i] = x.data[i] OP y.data[i];                                    \
    }                                                                          \
    return out;                                                                \
  }
TENSORFORGE_VECTOR_OP(+)
TENSORFORGE_VECTOR_OP(-)
TENSORFORGE_VECTOR_OP(*)
TENSORFORGE_VECTOR_OP(/)
#undef TENSORFORGE_VECTOR_OP

/// `x * y + z` in one rounding per component: what an `fma` op on a vector
/// spells to.  Written out rather than left to contraction for one reason,
/// the paired FMA of sm_100 and sm_103 (`FFMA2`): nvcc reaches it through
/// `__ffma2_rn` and does not pair two scalar FMAs by itself, so a vector body
/// that wants it has to say so here.  The pairs are adjacent components, which
/// is what a lead-vectorized accumulator holds.
///
/// sm_120 declares `__ffma2_rn` as well, but has no paired unit: it lowers to
/// two FFMA.  Not left to do so, because packing the pairs still changes the
/// code around them -- the same FFMA count, but 8 and 16 more instructions
/// (moves, NOP padding, address arithmetic) in the two kernels compared.  So
/// the pairs are formed on 10.x and 11.x only, and everywhere else the plain
/// expression contracts into N scalar FMAs.
template <typename T, std::size_t N, std::size_t A, std::size_t B,
          std::size_t C>
__device__ __forceinline__ VectorStruct<T, N, C>
fma(const VectorStruct<T, N, A> &x, const VectorStruct<T, N, B> &y,
    const VectorStruct<T, N, C> &z) {
  VectorStruct<T, N, C> out;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000 && __CUDA_ARCH__ < 1200
  if constexpr (std::is_same<T, float>::value && N % 2 == 0) {
#pragma unroll
    for (std::size_t i = 0; i < N; i += 2) {
      const float2 r = __ffma2_rn(make_float2(x.data[i], x.data[i + 1]),
                                  make_float2(y.data[i], y.data[i + 1]),
                                  make_float2(z.data[i], z.data[i + 1]));
      out.data[i] = r.x;
      out.data[i + 1] = r.y;
    }
  } else
#endif
  {
#pragma unroll
    for (std::size_t i = 0; i < N; ++i) {
      out.data[i] = x.data[i] * y.data[i] + z.data[i];
    }
  }
  return out;
}

} // namespace tensorforge

// The two properties the struct exists for, asserted rather than assumed.
// Width first: a member array is only as wide as its element count if nothing
// pads it, and a padded `VectorT` would make every wide transfer copy a
// fraction of what it names -- silently, since the subscripts still compile.
// Alignment second: `VectorT` is wide *because* it is over-aligned, and its
// relaxed twin is castable *because* it is not; swap either and the code is
// still valid C++ that does the wrong thing.  A GNU `vector_size` typedef
// carries both, and can get it wrong: in an alias template with a dependent
// element type GCC drops the attribute with a warning and turns
// `VectorT<float, 4>` into plain `float` (see `hip.h`).
static_assert(sizeof(tensorforge::VectorT<float, 4>) == 4 * sizeof(float),
              "VectorT is padded: it is not as wide as it claims");
static_assert(alignof(tensorforge::VectorT<float, 4>) == 4 * sizeof(float),
              "VectorT is not naturally aligned");
static_assert(alignof(tensorforge::VectorRelaxedT<float, 4>) == alignof(float),
              "VectorRelaxedT is over-aligned: it exists to be castable onto "
              "an array that only has element alignment");

namespace tensorforge {

__device__ __forceinline__ int lane_id() {
  int lane;
  asm("mov.u32 %0, %%laneid" : "=r"(lane)::);
  return lane;
}

/// Ask a cache for the line at `ptr` and carry on without it.
///
/// `prefetch.global.L1` and `.L2` are PTX ISA 2.0 and sm_50, and there is no
/// `__CUDA_ARCH__` guard here on purpose: which architectures may reach these
/// is decided in `Target.prefetch`, and a guard would turn a target
/// below the line into a call that compiles and does nothing rather than into
/// the question it is.
///
/// No `"memory"` clobber. The hint reads nothing the surrounding code can
/// observe, and declaring otherwise would pin every access around it in
/// place -- which is the scheduling freedom the prefetch was issued to use.
/// `volatile` is what keeps the statement at all, since it has no result that
/// anything could be waiting on.
template <typename T> __device__ __forceinline__ void prefetchL1(const T *ptr) {
  asm volatile("prefetch.global.L1 [%0];" ::"l"(ptr));
}

/// The same one level out, which is where a hint issued far ahead of its use
/// belongs: L1 is small enough that the line is likely evicted before the
/// load arrives.
template <typename T> __device__ __forceinline__ void prefetchL2(const T *ptr) {
  asm volatile("prefetch.global.L2 [%0];" ::"l"(ptr));
}

// Bits at 0, Subblock, 2*Subblock, ... below Block: the lanes that share a
// reduction with lane 0 of a block.
template <std::size_t Block, std::size_t Subblock>
constexpr unsigned groupMask() {
  unsigned mask = 0;
  for (std::size_t k = 0; k < Block; k += Subblock) {
    mask |= 1u << k;
  }
  return mask;
}

template <typename Op, std::size_t Block, std::size_t Subblock>
__device__ __forceinline__ bool ballotReduction(bool value) {
  const auto ballot = __ballot_sync(FullWarpMask, value ? 1 : 0);
  const auto thread = (threadIdx.x / Block) * Block;
  const auto subthread = Subblock == 1 ? 0 : (threadIdx.x % Subblock);

  // The mask has to name all Block/Subblock lanes that participate: a single
  // bit, `(1 << subthread) << thread`, would make `(mask & ballot) == mask`
  // re-read only this lane's own contribution, and every reduction would
  // return it unchanged.
  const auto mask = groupMask<Block, Subblock>() << (thread + subthread);

  if constexpr (Op::Op == Operation::And) {
    return (mask & ballot) == mask;
  }
  if constexpr (Op::Op == Operation::Or) {
    return (mask & ballot) != 0;
  }
  if constexpr (Op::Op == Operation::Xor) {
    return (__popc((mask & ballot)) & 1) == 0;
  }
}

// A butterfly all-reduce: after it, every lane of a Block-sized group holds
// the same result.
//
// The shuffle mask is `FullWarpMask` -- `warpSize` as a mask is lane 5 only --
// and the XOR distance is the single bit `i`; `i - 1` would be a mask of low
// bits and pair lanes with the wrong partners.
template <typename Op, typename T, std::size_t Block, std::size_t Subblock>
__device__ __forceinline__ T fullReduction(T value) {
  T result = value;
#pragma unroll
  for (std::size_t i = Block >> 1; i >= Subblock; i >>= 1) {
    const auto other = __shfl_xor_sync(FullWarpMask, result, i);
    result = Op::applyOperation(result, other);
  }
  return result;
}

template <typename Op, std::size_t Block, std::size_t Subblock, typename T>
__device__ __forceinline__ T reduction(const T &value) {
  if constexpr (Block == Subblock) {
    return value;
  } else if constexpr (std::is_same_v<T, bool> && Op::Op == Operation::And &&
                       Block == 32 && Subblock == 1) {
    return __all_sync(FullWarpMask, value ? 1 : 0) != 0;
  } else if constexpr (std::is_same_v<T, bool> && Op::Op == Operation::Or &&
                       Block == 32 && Subblock == 1) {
    return __any_sync(FullWarpMask, value ? 1 : 0) != 0;
  } else if constexpr (std::is_same_v<T, bool>) {
    return ballotReduction<Op, Block, Subblock>(value);
  } else {
    return fullReduction<Op, T, Block, Subblock>(value);
  }
}

template <typename T> __device__ __forceinline__ T readlane(T value, int lane) {
  return __shfl_sync(FullWarpMask, value, lane);
}

template <std::size_t Block, std::size_t Subblock, std::size_t Lane, typename T>
__device__ __forceinline__ T broadcast(T value) {
  if constexpr (Block == 1 || Block == Subblock) {
    return value;
  } else {
    const auto subblockvar = lane_id() % Subblock;
    return __shfl_sync(FullWarpMask, value, Subblock * Lane + subblockvar,
                       Block);
  }
}

/// Lane `lane` of every `Block` consecutive lanes, read by each lane of them
/// with a `lane` of its own: a gather, where `broadcast` hands one lane's
/// value to all.  The same instruction either way, `SHFL.IDX`.
template <std::size_t Block, typename T>
__device__ __forceinline__ T gather(T value, int lane) {
  return __shfl_sync(FullWarpMask, value, lane, Block);
}

// `cuda::ptx` spells the cluster-launch-control wrappers from CCCL 2.8 on, and
// only for PTX ISA 8.6 (CUDA 12.8) or newer; the mbarrier ones used alongside
// are missing from CCCL 2.3 as well.  Unguarded, every translation unit that
// includes this header would need that toolkit -- sm_80 and sm_90 included,
// where nothing can ask for `launch_control` -- and CUDA 12.4 would refuse all
// of them.
#if defined(CCCL_VERSION) && CCCL_VERSION >= 2008000 &&                        \
    defined(__cccl_ptx_isa) && __cccl_ptx_isa >= 860

/// Blackwell's hardware work queue, as a ring of `Depth` outstanding requests.
///
/// `clusterlaunchcontrol.try_cancel` asks the grid launcher to *not* launch a
/// CTA that has not started yet, and hands the caller its id.  A block that
/// keeps canceling therefore drains the grid without the launcher ever
/// putting those blocks on an SM, which is a persistent kernel whose work
/// queue is the grid itself -- and, unlike a grid-stride loop, one that needs
/// no occupancy query to size the launch.
///
/// The request is asynchronous: it writes 16 bytes into shared memory and
/// signals an mbarrier.  So the response for element k + 1 is asked for
/// before element k is computed and collected after, and the queue latency
/// disappears behind the body.  `Depth` > 1 keeps that many requests in
/// flight, which is what a *data* prefetch needs: at depth 1 the next index
/// is known only at the bottom of the iteration, with no body left to overlap
/// the transfer with.
///
/// Split in two on purpose.  `ClusterLaunchQueue` is the shared state -- the
/// response slots and their barriers -- and `ClusterLaunchCursor` is the
/// bookkeeping, which is per *thread*.  Putting the cursor in shared memory
/// alongside the queue reads naturally and is a data race: every thread of
/// the block advances it, so `parity ^= 1` from 128 threads leaves a parity
/// nobody agrees on and the next wait blocks forever.  Each thread holding
/// its own copy costs a few registers and is correct, because every value it
/// derives comes from a response all of them read.
template <int Depth = 1> struct alignas(16) ClusterLaunchQueue {
  // `try_cancel` writes a 16-byte response, so the slot has to be 16-byte
  // aligned; `alignas` on the struct states it rather than inheriting it from
  // `uint4` by luck.
  uint4 response[Depth];
  uint64_t barrier[Depth];

  __device__ __forceinline__ void arm() {
    namespace cg = cooperative_groups;
    namespace ptx = cuda::ptx;
    if (cg::thread_block::thread_rank() == 0) {
      for (int i = 0; i < Depth; ++i) {
        ptx::mbarrier_init(&barrier[i], 1);
      }
      // `try_cancel` completes the barrier through the *async* proxy, and an
      // initialization written through the generic one is not ordered against
      // it without this.  It works without the fence on the part this was
      // measured on, which is not the same as being allowed to omit it.
      ptx::fence_proxy_async(ptx::space_shared);
    }
    __syncthreads();
  }

  __device__ __forceinline__ void post(int slot) {
    namespace cg = cooperative_groups;
    namespace ptx = cuda::ptx;
    if (cg::thread_block::thread_rank() == 0) {
      // expect-tx *before* the operation that completes it.  The other order
      // races: the response can land before the transaction count is
      // registered, and the barrier then never completes the phase.
      ptx::mbarrier_arrive_expect_tx(ptx::sem_relaxed, ptx::scope_cta,
                                     ptx::space_shared, &barrier[slot],
                                     sizeof(uint4));
      ptx::clusterlaunchcontrol_try_cancel(&response[slot], &barrier[slot]);
    }
  }
};

/// The per-thread half.  Every thread of the block runs it and they stay in
/// lockstep, because each decision is read from a response in shared memory
/// behind a barrier.
template <int Depth = 1> struct ClusterLaunchCursor {
  // The mbarrier phase, which alternates.  This is what a caller cannot be
  // trusted with: passed as a plain `int` argument it is a copy, the caller's
  // parity never flips, and from the second iteration on the wait returns
  // immediately on a phase that already completed -- so the block reads a
  // stale response, believes it holds work that another block also holds, and
  // the loop never terminates.  Keeping it here is the reason the type exists.
  uint32_t phase[Depth];
  int head;
  int outstanding;
  bool refill;

  __device__ __forceinline__ void start(ClusterLaunchQueue<Depth> &queue) {
    for (int i = 0; i < Depth; ++i) {
      phase[i] = 0;
    }
    head = 0;
    outstanding = Depth;
    refill = true;
    queue.arm();
    for (int i = 0; i < Depth; ++i) {
      queue.post(i);
    }
  }

  /// The CTA id `next` hands out next, or -1 when the grid is exhausted --
  /// without taking it, so that the element after this one is known at the
  /// top of the iteration and its operands can be fetched while this one is
  /// computed.  At `Depth` 2 the response asked for an iteration ago is read.
  ///
  /// Every thread waits for the response itself, and no rendezvous is needed
  /// to read it: the slot is let go in `next`, behind its barrier.  A
  /// response that comes back empty is passed over here, as `next` passes
  /// it: the grid is drained, nothing is asked again in its slot, and a slot
  /// nobody reposts needs no rendezvous to be passed.  So what this answers
  /// is what the following `next` answers.
  __device__ __forceinline__ int peek(ClusterLaunchQueue<Depth> &queue) {
    namespace ptx = cuda::ptx;
    while (outstanding > 0) {
      const int slot = head;
      while (!ptx::mbarrier_try_wait_parity(ptx::sem_acquire, ptx::scope_cta,
                                            &queue.barrier[slot],
                                            phase[slot])) {
      }
      if (ptx::clusterlaunchcontrol_query_cancel_is_canceled(
              queue.response[slot])) {
        return ptx::clusterlaunchcontrol_query_cancel_get_first_ctaid_x<int>(
            queue.response[slot]);
      }
      phase[slot] ^= 1u;
      head = (head + 1) % Depth;
      --outstanding;
      refill = false;
    }
    return -1;
  }

  /// The next CTA id, or -1 when the grid is exhausted.
  ///
  /// Block-uniform, which is the property the enclosing loop rests on: every
  /// thread reads the same response, so all of them leave the loop on the
  /// same iteration and a block-wide barrier in the body is reached by all or
  /// by none.  A grid-stride loop is only warp-uniform in the same place.
  __device__ __forceinline__ int next(ClusterLaunchQueue<Depth> &queue) {
    namespace ptx = cuda::ptx;
    while (outstanding > 0) {
      const int slot = head;
      while (!ptx::mbarrier_try_wait_parity(ptx::sem_acquire, ptx::scope_cta,
                                            &queue.barrier[slot],
                                            phase[slot])) {
      }
      const bool granted = ptx::clusterlaunchcontrol_query_cancel_is_canceled(
          queue.response[slot]);
      int cta = -1;
      if (granted) {
        // Only ctaid.x: the launcher is one-dimensional here, and the y/z
        // queries would answer for a geometry nothing emits.
        cta = ptx::clusterlaunchcontrol_query_cancel_get_first_ctaid_x<int>(
            queue.response[slot]);
      }
      // Everyone has read the response; only now may the slot be reposted.
      // Reposting earlier flips the barrier a second time under a thread that
      // is still waiting on the first, and that thread then waits for a phase
      // that has already gone past.
      __syncthreads();
      phase[slot] ^= 1u;
      head = (head + 1) % Depth;
      --outstanding;
      if (granted) {
        // While `refill` holds, `outstanding` is still Depth, so the slot
        // just freed is exactly the tail of the ring and reposting it keeps
        // the order.  Once a request comes back empty the grid is drained and
        // reposting would only ask again for nothing.
        if (refill) {
          queue.post(slot);
          ++outstanding;
        }
        return cta;
      }
      refill = false;
    }
    return -1;
  }
};

#else

// Declared regardless, so that a kernel asking for `launch_control` on an
// older toolkit fails with the reason instead of an undeclared identifier.
// The assertion depends on `Depth` and fires only on instantiation.
template <int Depth = 1> struct ClusterLaunchQueue {
  static_assert(Depth < 0,
                "launch_control needs cuda::ptx::clusterlaunchcontrol_*: "
                "CCCL >= 2.8 and PTX ISA >= 8.6 (CUDA 12.8 or newer)");
};
template <int Depth = 1> struct ClusterLaunchCursor {
  static_assert(Depth < 0,
                "launch_control needs cuda::ptx::clusterlaunchcontrol_*: "
                "CCCL >= 2.8 and PTX ISA >= 8.6 (CUDA 12.8 or newer)");
};

#endif

/// `Bytes` bytes -- 4, 8 or 16 -- into shared memory at `dst` by `cp.async`,
/// of which the first `srcBytes` are read from global memory at `src` and the
/// rest written as zeros: the instruction's source size (PTX ISA, "cp.async",
/// `src-size`), held in a register, so one copy is a read where its condition
/// holds and a fill where not (`pir.engines`, a copy that fills).  At zero it
/// reads nothing, and `src` need be the address of nothing.  Sixteen bytes go
/// past L1 (`.cg`), four and eight through it (`.ca`), as
/// `__pipeline_memcpy_async` takes them, and the copy joins the thread's open
/// group, which `__pipeline_commit` closes.  A device without the instruction
/// traps.
template <int Bytes>
__device__ __forceinline__ void copyAsyncFill(void *dst, const void *src,
                                              uint32_t srcBytes) {
  static_assert(Bytes == 4 || Bytes == 8 || Bytes == 16,
                "cp.async moves 4, 8 or 16 bytes");
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  const uint32_t at = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
  const size_t from = __cvta_generic_to_global(src);
  if constexpr (Bytes == 16) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(at),
                 "l"(from), "r"(srcBytes)
                 : "memory");
  } else {
    asm volatile("cp.async.ca.shared.global [%0], [%1], %2, %3;\n" ::"r"(at),
                 "l"(from), "n"(Bytes), "r"(srcBytes)
                 : "memory");
  }
#elif defined(__CUDA_ARCH__)
  __trap();
#endif
}

/// The priority L2 gives the lines an access touches through a cache policy
/// (`engines.CACHE_POLICIES`, `evict_first` and `evict_last`).
enum class L2Evict { First, Last };

/// A cache policy of priority `P` for every line an access touches: the
/// 64-bit operand `.L2::cache_hint` takes (PTX ISA, "createpolicy":
/// `.fractional`, a fraction of 1.0; sm_80 and newer).  Not volatile: two
/// of the same policy are one value, which the compiler keeps once.
template <L2Evict P> __device__ __forceinline__ uint64_t l2Policy() {
  uint64_t policy = 0;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  if constexpr (P == L2Evict::Last) {
    asm("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(policy));
  } else {
    asm("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(policy));
  }
#elif defined(__CUDA_ARCH__)
  __trap();
#endif
  return policy;
}

/// A load from global memory under the cache policy `P` (PTX ISA, "ld":
/// `.L2::cache_hint`; sm_80 and newer).  The "memory" clobber keeps it in
/// its place among the other accesses, as a load the compiler knows is.
template <L2Evict P> __device__ __forceinline__ float ldL2(const float *at) {
  float value = 0;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm("ld.global.L2::cache_hint.f32 %0, [%1], %2;"
      : "=f"(value)
      : "l"(__cvta_generic_to_global(at)), "l"(l2Policy<P>())
      : "memory");
#elif defined(__CUDA_ARCH__)
  __trap();
#endif
  return value;
}

template <L2Evict P> __device__ __forceinline__ double ldL2(const double *at) {
  double value = 0;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm("ld.global.L2::cache_hint.f64 %0, [%1], %2;"
      : "=d"(value)
      : "l"(__cvta_generic_to_global(at)), "l"(l2Policy<P>())
      : "memory");
#elif defined(__CUDA_ARCH__)
  __trap();
#endif
  return value;
}

/// A store to global memory under the cache policy `P` (PTX ISA, "st":
/// `.L2::cache_hint`; sm_80 and newer).
template <L2Evict P>
__device__ __forceinline__ void stL2(float *at, float value) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("st.global.L2::cache_hint.f32 [%0], %1, %2;" ::"l"(
                   __cvta_generic_to_global(at)),
               "f"(value), "l"(l2Policy<P>())
               : "memory");
#elif defined(__CUDA_ARCH__)
  __trap();
#endif
}

template <L2Evict P>
__device__ __forceinline__ void stL2(double *at, double value) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("st.global.L2::cache_hint.f64 [%0], %1, %2;" ::"l"(
                   __cvta_generic_to_global(at)),
               "d"(value), "l"(l2Policy<P>())
               : "memory");
#elif defined(__CUDA_ARCH__)
  __trap();
#endif
}

/// A window copied into shared memory by the tensor memory accelerator: one
/// `cp.async.bulk` for the whole window, issued by one thread of the
/// multiplication and completing on an mbarrier in shared memory, where
/// `cp.async` has every thread copy sixteen bytes at a time and count groups.
///
/// The same split as `ClusterLaunchQueue` and `ClusterLaunchCursor`: the
/// barrier is shared, the cursor is per thread.  Every thread keeps the phase
/// it waits for, and whether a copy is in flight at all -- which is what lets
/// a wait that may come with nothing in flight (a masked element's drain, the
/// drain behind a loop) be the same call as one behind a copy: with nothing
/// pending it returns at once, where a wait on the barrier would wait for a
/// phase no copy completes.  One copy in flight a barrier, so a cursor is the
/// whole of a window's pipeline.
///
/// Inline PTX rather than `cuda::ptx`: the wrappers for the bulk copy and the
/// parity wait arrive in different CCCL releases, and the instructions are
/// the same on every toolkit that has sm_90 at all.
struct BulkCursor {
  uint64_t *barrier;
  uint32_t phase;
  bool pending;
  bool leader;
};

__device__ __forceinline__ uint32_t bulk_shared_address(const void *p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

/// The cursor for the barrier at `barrier`, which `leader` arms: one arrival
/// a phase, the leader's, which also announces the bytes in flight.  No
/// thread may wait on it before a barrier behind this made the arming
/// visible; the generator places one.
__device__ __forceinline__ BulkCursor bulk_arm(void *barrier, bool leader) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  if (leader) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(
                     bulk_shared_address(barrier))
                 : "memory");
    // The copy completes the barrier through the async proxy, and the
    // initialization went through the generic one.
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  }
#elif defined(__CUDA_ARCH__)
  __trap();
#endif
  return BulkCursor{static_cast<uint64_t *>(barrier), 0u, false, leader};
}

/// `bytes` from `src` to `dst`, both on sixteen bytes and `bytes` a multiple
/// of sixteen: the leader announces them on the barrier and issues the copy,
/// and every thread notes that one is in flight.
__device__ __forceinline__ void bulk_copy(BulkCursor &cursor, void *dst,
                                          const void *src, uint32_t bytes) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  if (cursor.leader) {
    // What the threads read of the window before the barrier in front of
    // this, they read through the generic proxy; the copy writes through the
    // async one.
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    // expect-tx before the copy that completes it, as for `post` above.
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(
                     bulk_shared_address(cursor.barrier)),
                 "r"(bytes)
                 : "memory");
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::"
                 "bytes [%0], [%1], %2, [%3];" ::"r"(bulk_shared_address(dst)),
                 "l"(src), "r"(bytes), "r"(bulk_shared_address(cursor.barrier))
                 : "memory");
  }
#elif defined(__CUDA_ARCH__)
  __trap();
#endif
  cursor.pending = true;
}

/// The copy in flight landed, if there is one: the barrier's phase has gone
/// past the one this thread waits for, which makes the bytes visible to it.
__device__ __forceinline__ void bulk_wait(BulkCursor &cursor) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  if (cursor.pending) {
    uint32_t done = 0;
    do {
      asm volatile("{\n"
                   ".reg .pred p;\n"
                   "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
                   "selp.u32 %0, 1, 0, p;\n"
                   "}"
                   : "=r"(done)
                   : "r"(bulk_shared_address(cursor.barrier)), "r"(cursor.phase)
                   : "memory");
    } while (!done);
    cursor.phase ^= 1u;
    cursor.pending = false;
  }
#elif defined(__CUDA_ARCH__)
  __trap();
#endif
}

// A ring of mbarriers (`IRBuilder.mbarrier_ring`): slots a producer fills and
// consumers drain, each slot a pair -- the full barrier the producer's bytes
// complete on, the empty one the consumers arrive at once done with the slot.
// No cursor: the phase a thread waits for is the parity of the round of the
// ring it is in, which every role counts from its trips (PTX ISA, "mbarrier":
// "Phase completion", "mbarrier.try_wait.parity").

/// The mbarrier at `barrier`, for `count` arrivals a phase (PTX ISA,
/// "mbarrier.init"); by the one thread that sets the ring up, which fences the
/// initialization for the async proxy the copies complete it through, and a
/// block barrier behind shows it to every thread.
__device__ __forceinline__ void mbarrierInit(void *barrier, uint32_t count) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(
                   bulk_shared_address(barrier)),
               "r"(count)
               : "memory");
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
#elif defined(__CUDA_ARCH__)
  __trap();
#endif
}

/// Wait until the phase of parity `parity` of the mbarrier at `barrier` has
/// completed: what was written and announced on it is visible behind this
/// (`mbarrier.try_wait.parity`, acquire at the CTA's scope by default).  A
/// waiter one round ahead of the barrier waits for the parity of a phase the
/// barrier has not reached; one a whole round behind returns at once.
__device__ __forceinline__ void mbarrierWait(void *barrier, uint32_t parity) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  uint32_t done = 0;
  do {
    asm volatile("{\n"
                 ".reg .pred p;\n"
                 "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
                 "selp.u32 %0, 1, 0, p;\n"
                 "}"
                 : "=r"(done)
                 : "r"(bulk_shared_address(barrier)), "r"(parity)
                 : "memory");
  } while (!done);
#elif defined(__CUDA_ARCH__)
  __trap();
#endif
}

/// One arrival at the mbarrier at `barrier`, by the thread `issue` picks
/// (`mbarrier.arrive`, release at the CTA's scope by default): what it read of
/// the slot before, it is done with.
__device__ __forceinline__ void mbarrierArrive(void *barrier, bool issue) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  if (issue) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(
                     bulk_shared_address(barrier))
                 : "memory");
  }
#elif defined(__CUDA_ARCH__)
  __trap();
#endif
}

/// One arrival at the mbarrier at `barrier` that announces `bytes` for the
/// copies through it to complete (`mbarrier.arrive.expect_tx`), by the thread
/// `issue` picks: the phase completes once they have landed.
__device__ __forceinline__ void mbarrierExpect(void *barrier, uint32_t bytes,
                                               bool issue) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  if (issue) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(
                     bulk_shared_address(barrier)),
                 "r"(bytes)
                 : "memory");
  }
#elif defined(__CUDA_ARCH__)
  __trap();
#endif
}

/// The high-order tile's claim of a tile (`batch_loop.TileSchedule`), by the
/// thread `issue` picks: the arrival on the claims' mbarrier -- the full
/// barrier of the 1-slot ring `barriers` -- that announces the 16-byte answer,
/// then `clusterlaunchcontrol.try_cancel` (PTX ISA 8.6, sm_100 on), which
/// cancels a CTA of the launch that has not started and writes its id into
/// `response`, or writes that none was left, completing the barrier's phase.
/// The expectation ahead of the request: the answer may land before an
/// expectation behind it was registered, and the phase would never complete.
__device__ __forceinline__ void tileClaim(void *response, uint64_t *barriers,
                                          bool issue) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  if (issue) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], 16;" ::"r"(
                     bulk_shared_address(&barriers[0]))
                 : "memory");
    asm volatile("clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::"
                 "complete_tx::bytes.b128 [%0], [%1];" ::"r"(
                     bulk_shared_address(response)),
                 "r"(bulk_shared_address(&barriers[0]))
                 : "memory");
  }
#elif defined(__CUDA_ARCH__)
  __trap();
#endif
}

/// The tile after `tile`: where the launch is a CTA a tile (`claiming`), the
/// answer of the claim the phase of parity `parity` of the claims' barrier
/// completes -- the first CTA id of the launch it cancelled, or -1 where
/// none was left -- and else the static stride's, `grid` on, or -1 past the
/// `tiles`.
__device__ __forceinline__ int tileNext(const void *response,
                                        uint64_t *barriers, uint32_t parity,
                                        bool claiming, size_t tile, size_t grid,
                                        size_t tiles) {
  if (!claiming) {
    return tile + grid < tiles ? static_cast<int>(tile + grid) : -1;
  }
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  mbarrierWait(&barriers[0], parity);
  const unsigned long long lo =
      reinterpret_cast<const volatile unsigned long long *>(response)[0];
  const unsigned long long hi =
      reinterpret_cast<const volatile unsigned long long *>(response)[1];
  uint32_t canceled = 0, first = 0;
  asm volatile(
      "{\n\t.reg .b128 r;\n\t.reg .pred p;\n\t"
      "mov.b128 r, {%2, %3};\n\t"
      "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p, r;\n\t"
      "selp.b32 %0, 1, 0, p;\n\t"
      "clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 "
      "%1, r;\n\t}"
      : "=r"(canceled), "=r"(first)
      : "l"(lo), "l"(hi)
      : "memory");
  return canceled ? static_cast<int>(first) : -1;
#else
#if defined(__CUDA_ARCH__)
  __trap();
#endif
  return -1;
#endif
}

/// `bytes` from `src` to `dst`, both on sixteen bytes and `bytes` a multiple of
/// sixteen, completing on the mbarrier at `barrier` without arriving at it
/// (`cp.async.bulk`, `.mbarrier::complete_tx::bytes`): what the slot's
/// expected bytes count down.  By the thread `issue` picks.
__device__ __forceinline__ void bulkCopyOn(void *barrier, void *dst,
                                           const void *src, uint32_t bytes,
                                           bool issue) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  if (issue) {
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::"
                 "bytes [%0], [%1], %2, [%3];" ::"r"(bulk_shared_address(dst)),
                 "l"(src), "r"(bytes), "r"(bulk_shared_address(barrier))
                 : "memory");
  }
#elif defined(__CUDA_ARCH__)
  __trap();
#endif
}

/// A shared-memory matrix descriptor (`Op.MATRIX_DESC`): what wgmma reads an
/// operand in shared memory through.  `layout` holds every field but the
/// start address -- the leading and the stride byte offset, the swizzle, the
/// fixed bits of a format (`engines.matrix_descriptor_bits`) -- and the
/// address goes in here, in 16-byte units, bits 0 to 13 (PTX ISA, wgmma's
/// "Matrix Descriptor Format").  It is the operand's address in the shared
/// window, so a buffer is described wherever the allocator placed it.
__device__ __forceinline__ uint64_t matrixDescriptor(const void *operand,
                                                     uint64_t layout) {
  return layout |
         ((static_cast<uint64_t>(__cvta_generic_to_shared(operand)) >> 4) &
          0x3FFFull);
}

/// The fence, the commit and the wait of wgmma (PTX ISA, "wgmma.fence",
/// "wgmma.commit_group", "wgmma.wait_group"; sm_90a).  The products
/// themselves are written where they are issued, inside the guard nvcc
/// defines for sm_90a alone (`engines.WGMMA_FEATURE`): `-arch=sm_90a` builds
/// the source for `compute_90` as well, where ptxas refuses them.  These
/// three trap there instead, as the bulk copy does below sm_90, so that a
/// kernel whose products the guard left out never runs to the end.
///
/// The fence orders what the warpgroup wrote to the registers a product
/// behind it reads -- its accumulator, its A in registers -- before that
/// product; the commit closes the products issued since the last one into a
/// group; the wait lets at most `Pending` groups stay in flight, and the
/// registers of the others are valid behind it.  Each is executed by every
/// thread of the warpgroup alike.
__device__ __forceinline__ void wgmmaFence() {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
  asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
#elif defined(__CUDA_ARCH__)
  __trap();
#endif
}

__device__ __forceinline__ void wgmmaCommit() {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
  asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
#elif defined(__CUDA_ARCH__)
  __trap();
#endif
}

template <int Pending> __device__ __forceinline__ void wgmmaWait() {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
  asm volatile("wgmma.wait_group.sync.aligned %0;" ::"n"(Pending) : "memory");
#elif defined(__CUDA_ARCH__)
  __trap();
#endif
}

/// What the threads wrote to shared memory as ordinary stores, visible to
/// what reads it through the async proxy behind this -- a product reading
/// an operand through its descriptor, a bulk or a tensor copy -- and what
/// they read, ordered before what that proxy writes there
/// (`fence.proxy.async.shared::cta`, PTX ISA 8.0, sm_90).
__device__ __forceinline__ void fenceProxyAsync() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
#elif defined(__CUDA_ARCH__)
  __trap();
#endif
}

/// Programmatic Dependent Launch (`Options.programmatic_launch`): wait until
/// the grid this one was launched as the programmatic dependent of has
/// completed and its memory is visible (`griddepcontrol.wait`, PTX ISA 7.8,
/// sm_90).  Returns at once where the launch named no such dependency, and is
/// nothing below sm_90, where no launch can name one.
__device__ __forceinline__ void gridDependencyWait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}

/// The launch of `kernel` as the programmatic dependent of the work ahead of
/// it on `stream` (`cudaLaunchAttributeProgrammaticStreamSerialization`): its
/// blocks may be scheduled as the blocks of the grid before exit, and every
/// thread waits for that grid in `gridDependencyWait`.  A stream capture keeps
/// the dependency as a programmatic edge of the graph.  Without the attribute
/// in the runtime, the ordinary launch.
template <typename... Params, typename... Args>
inline cudaError_t launchProgrammatic(void (*kernel)(Params...), dim3 grid,
                                      dim3 block, std::size_t shared,
                                      cudaStream_t stream, Args &&...args) {
#if defined(CUDART_VERSION) && CUDART_VERSION >= 12000
  cudaLaunchConfig_t config = {};
  config.gridDim = grid;
  config.blockDim = block;
  config.dynamicSmemBytes = shared;
  config.stream = stream;
  cudaLaunchAttribute attribute[1];
  attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attribute[0].val.programmaticStreamSerializationAllowed = 1;
  config.attrs = attribute;
  config.numAttrs = 1;
  return cudaLaunchKernelEx(&config, kernel, std::forward<Args>(args)...);
#else
  kernel<<<grid, block, shared, stream>>>(std::forward<Args>(args)...);
  return cudaGetLastError();
#endif
}

/// The targets that have tcgen05 and tensor memory (sm_100a, sm_103a, and
/// their families): the source of a product writes its instructions inside
/// this guard (`engines.TCGEN05_FEATURE`), since `-arch=sm_100a` builds the
/// source for `compute_100` as well, where ptxas refuses them; the helpers
/// below trap outside it.
#if defined(__CUDA_ARCH_FEAT_SM100_ALL) ||                                     \
    defined(__CUDA_ARCH_FEAT_SM103_ALL) ||                                     \
    (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) &&                                 \
     (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000 ||                                 \
      __CUDA_ARCH_FAMILY_SPECIFIC__ == 1030))
#define TENSORFORGE_TCGEN05 1
#endif

/// The warp of the block a thread is in, by its linear index: what the
/// allocation of tensor memory and a warp's quarter of its lanes go by.
__device__ __forceinline__ uint32_t tmemWarp() {
  return (threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z)) /
         32u;
}

/// `Columns` columns of tensor memory for the block (PTX ISA,
/// "tcgen05.alloc", "tcgen05.relinquish_alloc_permit"; sm_100a): allocated by
/// the first warp into the shared slot, the permit given up, and the address
/// read back from the slot by every thread behind a block barrier between the
/// engine's fences -- the allocation writes the slot through the engine.
/// `Columns` is a power of two from 32 to 512.
template <int Columns>
__device__ __forceinline__ uint32_t tmemAlloc(void *slot) {
#if defined(TENSORFORGE_TCGEN05)
  if (tmemWarp() == 0) {
    asm volatile(
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::
            "r"(bulk_shared_address(slot)),
        "n"(Columns)
        : "memory");
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::
                     : "memory");
  }
  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
  __syncthreads();
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
  return *static_cast<volatile uint32_t *>(slot);
#elif defined(__CUDA_ARCH__)
  __trap();
  return 0;
#else
  return 0;
#endif
}

/// The columns freed again by the first warp, once every thread of the block
/// is done with them: a block barrier between the engine's fences, then
/// `tcgen05.dealloc` (PTX ISA, "tcgen05.dealloc").  A block frees what it
/// allocated before it exits.
template <int Columns>
__device__ __forceinline__ void tmemDealloc(uint32_t tmem) {
#if defined(TENSORFORGE_TCGEN05)
  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
  __syncthreads();
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
  if (tmemWarp() == 0) {
    asm volatile(
        "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(tmem),
        "n"(Columns)
        : "memory");
  }
#elif defined(__CUDA_ARCH__)
  __trap();
#endif
}

/// The address of the warp's quarter of the lanes of `tmem`, from `column`
/// on: warp w of its warpgroup reaches the lanes 32 (w % 4) to 32 (w % 4) + 31
/// alone, the lane in the upper half of the address (PTX ISA, "Tensor
/// Memory").
__device__ __forceinline__ uint32_t tmemWarpAddress(uint32_t tmem,
                                                    uint32_t column) {
  return tmem + ((32u * (tmemWarp() % 4u)) << 16) + column;
}

/// The waits for a warp's loads from tensor memory and its stores to it
/// (PTX ISA, "tcgen05.wait"): what was loaded is valid behind the first, what
/// was stored is in tensor memory behind the second.
__device__ __forceinline__ void tmemWaitLd() {
#if defined(TENSORFORGE_TCGEN05)
  asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
#elif defined(__CUDA_ARCH__)
  __trap();
#endif
}

__device__ __forceinline__ void tmemWaitSt() {
#if defined(TENSORFORGE_TCGEN05)
  asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
#elif defined(__CUDA_ARCH__)
  __trap();
#endif
}

/// The engine's operations ahead of this before the barrier or wait behind
/// it, and those behind it after the barrier or wait ahead of it (PTX ISA,
/// "tcgen05.fence").
__device__ __forceinline__ void tcgen05FenceBefore() {
#if defined(TENSORFORGE_TCGEN05)
  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
#elif defined(__CUDA_ARCH__)
  __trap();
#endif
}

__device__ __forceinline__ void tcgen05FenceAfter() {
#if defined(TENSORFORGE_TCGEN05)
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
#elif defined(__CUDA_ARCH__)
  __trap();
#endif
}

/// The products the thread `issue` picks issued since its last commit,
/// tracked by the cursor's mbarrier, which they arrive at once complete (PTX
/// ISA, "tcgen05.commit"): issued by that thread alone -- a commit tracks the
/// operations its own thread issued -- and noted by every thread on its
/// cursor, so that `bulk_wait` waits for the phase.
__device__ __forceinline__ void tcgen05Commit(BulkCursor &cursor, bool issue) {
#if defined(TENSORFORGE_TCGEN05)
  if (issue) {
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::"
                 "cluster.b64 [%0];" ::"r"(bulk_shared_address(cursor.barrier))
                 : "memory");
  }
#elif defined(__CUDA_ARCH__)
  __trap();
#endif
  cursor.pending = true;
}

/// The products the thread `issue` picks issued since its last commit,
/// tracked by the mbarrier at `barrier` -- the empty side of a ring's slot
/// (`mbarrierInit`) -- which they arrive at once complete: the slot's
/// operand read, its producer may fill it again (PTX ISA,
/// "tcgen05.commit").  No cursor: the producer waits for the round's parity
/// (`mbarrierWait`).
__device__ __forceinline__ void tcgen05CommitOn(void *barrier, bool issue) {
#if defined(TENSORFORGE_TCGEN05)
  if (issue) {
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::"
                 "cluster.b64 [%0];" ::"r"(bulk_shared_address(barrier))
                 : "memory");
  }
#elif defined(__CUDA_ARCH__)
  __trap();
#endif
}

/// `bytes` of shared memory from `src` added into global memory at `dst`,
/// both on sixteen bytes and `bytes` a multiple of sixteen (PTX ISA,
/// "cp.reduce.async.bulk": `.add`, completing by `.bulk_group`; sm_90 and
/// newer), closed into a bulk async-group of the issuing thread's own, which
/// `bulkWaitRead` counts.  Issued by one thread: what the others wrote to
/// `src` through the generic proxy, they fenced for the async one ahead of
/// the barrier that shows it to this thread.  No tensor map: a run of
/// contiguous elements.  Every element of the run is added, an element the
/// writers left zero adding exactly +0.0.
__device__ __forceinline__ void bulkReduceAdd(float *dst, const float *src,
                                              uint32_t bytes) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32 "
               "[%0], [%1], %2;" ::"l"(dst),
               "r"(bulk_shared_address(src)), "r"(bytes)
               : "memory");
  asm volatile("cp.async.bulk.commit_group;" ::: "memory");
#elif defined(__CUDA_ARCH__)
  __trap();
#endif
}

/// The same over doubles.
__device__ __forceinline__ void bulkReduceAdd(double *dst, const double *src,
                                              uint32_t bytes) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f64 "
               "[%0], [%1], %2;" ::"l"(dst),
               "r"(bulk_shared_address(src)), "r"(bytes)
               : "memory");
  asm volatile("cp.async.bulk.commit_group;" ::: "memory");
#elif defined(__CUDA_ARCH__)
  __trap();
#endif
}

/// At most `Pending` of this thread's bulk groups still reading their
/// sources (PTX ISA, "cp.async.bulk.wait_group": `.read`): the boxes of the
/// others may be written again.  A thread that issued none waits for
/// nothing.  Here rather than in `cuda_tma.h`, which includes this: the
/// bulk reductions without a map (`bulkReduceAdd`) wait on it too.
template <int Pending> __device__ __forceinline__ void bulkWaitRead() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("cp.async.bulk.wait_group.read %0;" ::"n"(Pending) : "memory");
#elif defined(__CUDA_ARCH__)
  __trap();
#endif
}

/// The 19-bit E8M10 the tensor cores multiply, as its bit pattern.
///
/// A typedef and not a wrapper struct, and the reason is the constraint
/// letter: the halves go straight into `mma.sync` as `"r"` operands, which
/// binds a 32-bit *register* and not a class type -- so a struct around the
/// same four bytes would not compile, however much better it would document
/// itself.
///
/// The distinction is therefore not enforced here.  It is enforced where it
/// can be and where it matters: on the Intel side `tf32` is
/// `esimd::tfloat32`, a real class, and `simd<float, N>` will not pass for
/// `simd<tf32, N>` there.  Both spellings are well-formed C++ and only one is
/// the instruction's operand, so the check belongs on the side that can make
/// it.
using tf32 = uint32_t;

/// The two TF32 halves 3xTF32 multiplies: `upper` rounded to TF32, `lower`
/// what that left over.
///
/// `lower` goes in unrounded.  The instruction reads a TF32 operand by
/// ignoring its low 13 bits, so rounding it first changes the result by less
/// than the truncation that follows anyway -- measured over 4096 m16n8k8
/// 3xTF32 tiles (K = 56) against an FP64 reference, the error is the same to
/// two digits (max 18.3 against 18.0, mean 2.43 against 2.42, in units of
/// 2^-24 of the sum of |a b|).  `upper` takes `cvt.rn` from sm_90 on, which
/// is one instruction there (`F2FP.TF32.F32.PACK_B`), where `cvt.rna` is
/// emulated by a compare, a select and two integer operations.  Together two
/// instructions per value instead of eight, for the same accuracy.
///
/// Not the split the host applies to a prepared operand
/// (`tests/harness/layout.split_tf32`): each is a valid decomposition, and
/// nothing requires the kernel's own to match it bit for bit.
__device__ __forceinline__ void splitFloatTF32(tf32 &upper, tf32 &lower,
                                               float value) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm("cvt.rn.tf32.f32 %0, %1;\n" : "=r"(upper) : "f"(value));
#else
  asm("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(upper) : "f"(value));
#endif
  lower = __float_as_uint(value - __uint_as_float(upper));
}

/// Four 8x8 matrices of 16-bit elements from shared memory, as `ldmatrix
/// .x4` loads them (PTX ISA, "Warp-level matrix load instruction:
/// ldmatrix"): lane `l` names the 16-byte row `l % 8` of matrix `l / 8`, and
/// register `i` of lane `t` receives the 32-bit word `t % 4` of row `t / 4`
/// of matrix `i`.  Over 32-bit elements the two halves of a word stay
/// together, so rows of four floats land as the TF32 `m16n8k8` A fragment
/// reads them where the four matrices are the tile's rows 0-7 and 8-15 at
/// columns 0-3, then at 4-7 -- the multi-element section's matrix tiling
/// (`compute.multi_element`).  `base + index` is the lane's row in shared
/// memory, 16-byte aligned.
///
/// `volatile` and no memory clobber: the read must not move across the
/// block's barriers, and nothing else is ordered against it.  A device
/// without the instruction traps (`engines.MATRIX_TRANSFER_SM`).
__device__ __forceinline__ void ldmatrixX4(float &r0, float &r1, float &r2,
                                           float &r3, const float *base,
                                           int index) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
  uint32_t x0, x1, x2, x3;
  const uint32_t at =
      static_cast<uint32_t>(__cvta_generic_to_shared(base + index));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, "
               "[%4];\n"
               : "=r"(x0), "=r"(x1), "=r"(x2), "=r"(x3)
               : "r"(at));
  r0 = __uint_as_float(x0);
  r1 = __uint_as_float(x1);
  r2 = __uint_as_float(x2);
  r3 = __uint_as_float(x3);
#elif defined(__CUDA_ARCH__)
  __trap();
#endif
}

/// One 8x8 matrix, as `ldmatrixX4` loads four (`.x1`): lanes 0-7 name its
/// rows -- the others' addresses are not read -- and lane `t` receives the
/// word `t % 4` of row `t / 4`.
__device__ __forceinline__ void ldmatrixX1(float &r0, const float *base,
                                           int index) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
  uint32_t x0;
  const uint32_t at =
      static_cast<uint32_t>(__cvta_generic_to_shared(base + index));
  asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];\n"
               : "=r"(x0)
               : "r"(at));
  r0 = __uint_as_float(x0);
#elif defined(__CUDA_ARCH__)
  __trap();
#endif
}

/// Two 8x8 matrices (`.x2`): lanes 0-15 name their rows.
__device__ __forceinline__ void ldmatrixX2(float &r0, float &r1,
                                           const float *base, int index) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
  uint32_t x0, x1;
  const uint32_t at =
      static_cast<uint32_t>(__cvta_generic_to_shared(base + index));
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
               : "=r"(x0), "=r"(x1)
               : "r"(at));
  r0 = __uint_as_float(x0);
  r1 = __uint_as_float(x1);
#elif defined(__CUDA_ARCH__)
  __trap();
#endif
}

/// The registers of a warp into 8x8 matrices of 16-bit elements in shared
/// memory, as `stmatrix` stores them (PTX ISA, "Warp-level matrix store
/// instruction: stmatrix", sm_90): `ldmatrixX4` the other way round -- lane
/// `l` names the 16-byte row `l % 8` of matrix `l / 8`, and register `i` of
/// lane `t` is the 32-bit word `t % 4` of row `t / 4` of matrix `i`; `.x1`
/// and `.x2` take the rows lanes 0-7 and 0-15 name.
///
/// `volatile` and a memory clobber: the store writes memory no operand
/// names, so a read of it behind the store must not be answered from before
/// it, nor a write ahead of it moved behind.  A device without the
/// instruction traps.
__device__ __forceinline__ void stmatrixX1(float *base, int index, float r0) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  const uint32_t at =
      static_cast<uint32_t>(__cvta_generic_to_shared(base + index));
  asm volatile(
      "stmatrix.sync.aligned.m8n8.x1.shared.b16 [%0], {%1};\n" ::"r"(at),
      "r"(__float_as_uint(r0))
      : "memory");
#elif defined(__CUDA_ARCH__)
  __trap();
#endif
}

__device__ __forceinline__ void stmatrixX2(float *base, int index, float r0,
                                           float r1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  const uint32_t at =
      static_cast<uint32_t>(__cvta_generic_to_shared(base + index));
  asm volatile(
      "stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1,%2};\n" ::"r"(at),
      "r"(__float_as_uint(r0)), "r"(__float_as_uint(r1))
      : "memory");
#elif defined(__CUDA_ARCH__)
  __trap();
#endif
}

__device__ __forceinline__ void stmatrixX4(float *base, int index, float r0,
                                           float r1, float r2, float r3) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  const uint32_t at =
      static_cast<uint32_t>(__cvta_generic_to_shared(base + index));
  asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], "
               "{%1,%2,%3,%4};\n" ::"r"(at),
               "r"(__float_as_uint(r0)), "r"(__float_as_uint(r1)),
               "r"(__float_as_uint(r2)), "r"(__float_as_uint(r3))
               : "memory");
#elif defined(__CUDA_ARCH__)
  __trap();
#endif
}

} // namespace tensorforge
#endif // SEISSOL_TENSORFORGE_INCLUDE_TENSORFORGE_DEVICE_CUDA_H_
