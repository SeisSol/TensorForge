// SPDX-FileCopyrightText: 2026 SeisSol Group
//
// SPDX-License-Identifier: MIT
#ifndef SEISSOL_TENSORFORGE_INCLUDE_TENSORFORGE_DEVICE_CUDA_H_
#define SEISSOL_TENSORFORGE_INCLUDE_TENSORFORGE_DEVICE_CUDA_H_

#include <optional>
#include <type_traits>

#include "base.h"

#include <cooperative_groups.h>

#include <cuda/pipeline>
#include <cuda/ptx>

namespace tensorforge {

// The participation mask for the `_sync` intrinsics.  Every call site here
// passed `warpSize`, which is the *width* of a warp -- 32, i.e. the single bit
// 0x20 -- so the mask named lane 5 alone and no other lane took part in the
// exchange.
inline constexpr unsigned FullWarpMask = 0xffffffffu;

// CUDA's own `float2`/`float4` are `__align__(8)`/`__align__(16)`, which is
// right for a global access and wrong for a private one: a register staging
// array is 4-byte aligned by every rule that applies to it, and casting an
// over-aligned type onto it is undefined however reliably the compiler has
// been getting away with it.  This is the same pair `hip.h` declares -- the
// natural width for a base that proves it, and an element-aligned twin for a
// base that does not.
template <typename T, std::size_t N> struct VectorOf {
  typedef T type __attribute__((__vector_size__(N * sizeof(T))));
  typedef T relaxed
      __attribute__((__vector_size__(N * sizeof(T)), __aligned__(sizeof(T))));
};

template <typename T, std::size_t N>
using VectorRelaxedT = typename VectorOf<T, N>::relaxed;

__device__ __forceinline__ int lane_id() {
  int lane;
  asm("mov.u32 %0, %%laneid" : "=r"(lane)::);
  return lane;
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

  // `(1 << subthread) << thread` was a single bit, so `(mask & ballot) == mask`
  // only ever re-read this lane's own contribution and every reduction
  // returned it unchanged.  The mask has to name all Block/Subblock lanes that
  // participate.
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
// Four things were wrong here at once, and the first two cancel any effect the
// others might have had:
//
//   - the return type was `bool`, so every reduction of a numeric type came
//     back as 0 or 1;
//   - `value` was never read.  `result` started at the neutral element and
//     nothing else fed the loop, so the answer was the neutral element;
//   - the shuffle mask was `warpSize`, i.e. lane 5 only (see FullWarpMask);
//   - the XOR distance was `i - 1`, a mask of low bits rather than the single
//     bit `i`, so lanes paired with the wrong partners.
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

// #if __CUDA_ARCH__ >= 1000
// cf. the new CUDA programming guide 4.12
// declare this one as __shared__
struct ClusterLaunchCtrl {
private:
  uint4 result_;
  uint64_t barrier_;

public:
  __device__ __forceinline__ void init() {
    namespace cg = cooperative_groups;
    namespace ptx = cuda::ptx;

    if (cg::thread_block::thread_rank() == 0) {
      result_ = {};
      barrier_ = {};
      ptx::mbarrier_init(&barrier_, 1);
    }
  }

  __device__ __forceinline__ void setupNext() {
    namespace cg = cooperative_groups;
    namespace ptx = cuda::ptx;

    __syncthreads();

    if (cg::thread_block::thread_rank() == 0) {
      ptx::fence_proxy_async_generic_sync_restrict(
          ptx::sem_acquire, ptx::space_cluster, ptx::scope_cluster);

      cg::invoke_one(cg::coalesced_threads(), [&]() {
        ptx::clusterlaunchcontrol_try_cancel(&result_, &barrier_);
      });

      ptx::mbarrier_arrive_expect_tx(ptx::sem_relaxed, ptx::scope_cta,
                                     ptx::space_shared, &barrier_,
                                     sizeof(uint4));
    }
  }

  __device__ __forceinline__ std::optional<int> queryNext(int phase) {
    namespace cg = cooperative_groups;
    namespace ptx = cuda::ptx;

    while (!ptx::mbarrier_try_wait_parity(ptx::sem_acquire, ptx::scope_cta,
                                          &barrier_, phase)) {
    }
    phase ^= 1;

    const bool success =
        ptx::clusterlaunchcontrol_query_cancel_is_canceled(result_);
    if (!success) {
      return {};
    }

    // we only use blockIdx.x
    const auto nextBlock =
        ptx::clusterlaunchcontrol_query_cancel_get_first_ctaid_x<int>(result_);

    ptx::fence_proxy_async_generic_sync_restrict(
        ptx::sem_release, ptx::space_shared, ptx::scope_cluster);

    return nextBlock;
  }
};
// #endif

__device__ __forceinline__ void splitFloatTF32(uint32_t &upper, uint32_t &lower,
                                               float value) {
  asm("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(upper) : "f"(value));
  const auto upperF = *reinterpret_cast<float *>(&upper);
  asm("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(lower) : "f"(value - upperF));
}

} // namespace tensorforge
#endif // SEISSOL_TENSORFORGE_INCLUDE_TENSORFORGE_DEVICE_CUDA_H_
