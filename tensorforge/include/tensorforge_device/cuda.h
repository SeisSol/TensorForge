#pragma once

#include <optional>
#include <type_traits>

#include "base.h"

#include <cooperative_groups.h>

#include <cuda/pipeline>
#include <cuda/ptx>

namespace tensorforge {

__device__ __forceinline__ int lane_id() {
  int lane;
  asm("mov.u32 %0, %%laneid" : "=r"(lane)::);
  return lane;
}

template <typename Op, std::size_t Block, std::size_t Subblock>
__device__ __forceinline__ bool ballotReduction(bool value) {
  const auto ballot = __ballot_sync(warpSize, value ? 1 : 0);
  const auto thread = (threadIdx.x / Block) * Block;
  const auto subthread = Subblock == 1 ? 0 : (threadIdx.x % Subblock);
  constexpr auto basemask = 1;

  const auto mask = (basemask << subthread) << thread;

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

template <typename Op, typename T, std::size_t Block, std::size_t Subblock>
__device__ __forceinline__ bool fullReduction(T value) {
  T result = Op::neutral();
#pragma unroll
  for (std::size_t i = Block >> 1; i >= Subblock; i >>= 1) {
    const auto other = __shfl_xor_sync(warpSize, result, i - 1);
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
    return __all_sync(warpSize, value ? 1 : 0) != 0;
  } else if constexpr (std::is_same_v<T, bool> && Op::Op == Operation::Or &&
                       Block == 32 && Subblock == 1) {
    return __any_sync(warpSize, value ? 1 : 0) != 0;
  } else if constexpr (std::is_same_v<T, bool>) {
    return ballotReduction<Op, Block, Subblock>(value);
  } else {
    return fullReduction<Op, T, Block, Subblock>(value);
  }
}

template <typename T> __device__ __forceinline__ T readlane(T value, int lane) {
  return __shfl_sync(warpSize, value, lane);
}

template <std::size_t Block, std::size_t Subblock, std::size_t Lane, typename T>
__device__ __forceinline__ T broadcast(T value) {
  if constexpr (Block == 1 || Block == Subblock) {
    return value;
  } else {
    const auto subblockvar = lane_id() % Subblock;
    return __shfl_sync(warpSize, value, Subblock * Lane + subblockvar, Block);
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
