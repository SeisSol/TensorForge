// SPDX-FileCopyrightText: 2026 SeisSol Group
//
// SPDX-License-Identifier: MIT
#ifndef SEISSOL_TENSORFORGE_INCLUDE_TENSORFORGE_DEVICE_CUDA_TMA_H_
#define SEISSOL_TENSORFORGE_INCLUDE_TENSORFORGE_DEVICE_CUDA_TMA_H_

// Tensor maps: the boxes the tensor memory accelerator copies into shared
// memory (engine `tma`, `engines.TensorMap`), the copy on the device and the
// map's encoding on the host.  A header of its own, which a translation unit
// includes where it copies through a map: the driver's header it needs for
// the map's type is no part of the others.

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include <cuda.h>
#include <cuda_runtime.h>

#include "cuda.h"

namespace tensorforge {

/// A box of a tensor through its tensor map, into shared memory at `dst`
/// (PTX ISA, "cp.async.bulk.tensor": `.tile`, completing on an mbarrier by
/// `mbarrier::complete_tx::bytes`; sm_90 and newer).  The bulk copy's
/// protocol (`bulk_copy`): the cursor's leader announces the box's bytes on
/// the barrier and issues the copy, every thread notes that one is in flight,
/// and `bulk_wait` waits for the phase.  `map` is the generic address of a
/// tensor map in param space -- a `__grid_constant__` kernel parameter the
/// launcher encoded (`tensorMapTiled`) -- and the coordinates the box's first
/// element, the innermost dimension's first, as PTX counts them.  `bytes`
/// are the whole box's: what lies past the tensor's ends is filled, and
/// counted.  `dst` is on 128 bytes, and on the swizzle's period for a map
/// that swizzles: the hardware takes the permutation from the address.
__device__ __forceinline__ void tmaLoad(BulkCursor &cursor, void *dst,
                                        const CUtensorMap *map, uint32_t bytes,
                                        int c0, int c1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  if (cursor.leader) {
    // What the threads read of the window before the barrier in front of
    // this, they read through the generic proxy; the copy writes through the
    // async one.
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(
                     bulk_shared_address(cursor.barrier)),
                 "r"(bytes)
                 : "memory");
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.tile."
        "mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];" ::"r"(
            bulk_shared_address(dst)),
        "l"(reinterpret_cast<uint64_t>(map)), "r"(c0), "r"(c1),
        "r"(bulk_shared_address(cursor.barrier))
        : "memory");
  }
#elif defined(__CUDA_ARCH__)
  __trap();
#endif
  cursor.pending = true;
}

/// The same for a box of three dimensions.
__device__ __forceinline__ void tmaLoad(BulkCursor &cursor, void *dst,
                                        const CUtensorMap *map, uint32_t bytes,
                                        int c0, int c1, int c2) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  if (cursor.leader) {
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(
                     bulk_shared_address(cursor.barrier)),
                 "r"(bytes)
                 : "memory");
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global.tile."
        "mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4}], [%5];" ::"r"(
            bulk_shared_address(dst)),
        "l"(reinterpret_cast<uint64_t>(map)), "r"(c0), "r"(c1), "r"(c2),
        "r"(bulk_shared_address(cursor.barrier))
        : "memory");
  }
#elif defined(__CUDA_ARCH__)
  __trap();
#endif
  cursor.pending = true;
}

/// The same under the cache policy `P` for the box's lines in L2
/// (`engines.CACHE_POLICIES`): the copy's `.L2::cache_hint`, its operand
/// `l2Policy<P>` (PTX ISA, "cp.async.bulk.tensor"; "createpolicy").
template <L2Evict P>
__device__ __forceinline__ void tmaLoad(BulkCursor &cursor, void *dst,
                                        const CUtensorMap *map, uint32_t bytes,
                                        int c0, int c1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  if (cursor.leader) {
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(
                     bulk_shared_address(cursor.barrier)),
                 "r"(bytes)
                 : "memory");
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile."
                 "mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%2, "
                 "%3}], [%4], %5;" ::"r"(bulk_shared_address(dst)),
                 "l"(reinterpret_cast<uint64_t>(map)), "r"(c0), "r"(c1),
                 "r"(bulk_shared_address(cursor.barrier)), "l"(l2Policy<P>())
                 : "memory");
  }
#elif defined(__CUDA_ARCH__)
  __trap();
#endif
  cursor.pending = true;
}

/// The same for a box of three dimensions.
template <L2Evict P>
__device__ __forceinline__ void tmaLoad(BulkCursor &cursor, void *dst,
                                        const CUtensorMap *map, uint32_t bytes,
                                        int c0, int c1, int c2) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  if (cursor.leader) {
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(
                     bulk_shared_address(cursor.barrier)),
                 "r"(bytes)
                 : "memory");
    asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.tile."
                 "mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%2, "
                 "%3, %4}], [%5], %6;" ::"r"(bulk_shared_address(dst)),
                 "l"(reinterpret_cast<uint64_t>(map)), "r"(c0), "r"(c1),
                 "r"(c2), "r"(bulk_shared_address(cursor.barrier)),
                 "l"(l2Policy<P>())
                 : "memory");
  }
#elif defined(__CUDA_ARCH__)
  __trap();
#endif
  cursor.pending = true;
}

/// The box of shared memory at `src` added into the tensor of `map` from
/// the coordinates on, innermost first, what lies past the tensor's ends left
/// out (PTX ISA, "cp.reduce.async.bulk.tensor": `.add`, `.tile`, completing
/// by `.bulk_group`; sm_90 and newer), and closed into a bulk async-group of
/// the issuing thread's own ("cp.async.bulk.commit_group"), so that
/// `bulkWaitRead` counts it.  Issued by one thread: what the others wrote to
/// `src` through the generic proxy, they fenced for the async one
/// (`fenceProxyAsync`) ahead of the barrier that shows it to this thread.
/// `src` is placed as `tmaLoad`'s destination is.
__device__ __forceinline__ void tmaReduceAdd(const CUtensorMap *map,
                                             const void *src, int c0, int c1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile(
      "cp.reduce.async.bulk.tensor.2d.global.shared::cta.add.tile."
      "bulk_group [%0, {%1, %2}], [%3];" ::"l"(reinterpret_cast<uint64_t>(map)),
      "r"(c0), "r"(c1), "r"(bulk_shared_address(src))
      : "memory");
  asm volatile("cp.async.bulk.commit_group;" ::: "memory");
#elif defined(__CUDA_ARCH__)
  __trap();
#endif
}

/// The same for a box of three dimensions.
__device__ __forceinline__ void
tmaReduceAdd(const CUtensorMap *map, const void *src, int c0, int c1, int c2) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("cp.reduce.async.bulk.tensor.3d.global.shared::cta.add.tile."
               "bulk_group [%0, {%1, %2, %3}], [%4];" ::"l"(
                   reinterpret_cast<uint64_t>(map)),
               "r"(c0), "r"(c1), "r"(c2), "r"(bulk_shared_address(src))
               : "memory");
  asm volatile("cp.async.bulk.commit_group;" ::: "memory");
#elif defined(__CUDA_ARCH__)
  __trap();
#endif
}

/// At most `Pending` of this thread's bulk groups still reading their
/// sources (PTX ISA, "cp.async.bulk.wait_group": `.read`): the boxes of the
/// others may be written again.  A thread that issued none waits for
/// nothing.
template <int Pending> __device__ __forceinline__ void bulkWaitRead() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("cp.async.bulk.wait_group.read %0;" ::"n"(Pending) : "memory");
#elif defined(__CUDA_ARCH__)
  __trap();
#endif
}

/// The driver's `cuTensorMapEncodeTiled`, as the runtime hands it out.
using TensorMapEncoder = CUresult (*)(CUtensorMap *, CUtensorMapDataType,
                                      cuuint32_t, void *, const cuuint64_t *,
                                      const cuuint64_t *, const cuuint32_t *,
                                      const cuuint32_t *, CUtensorMapInterleave,
                                      CUtensorMapSwizzle,
                                      CUtensorMapL2promotion,
                                      CUtensorMapFloatOOBfill);

/// The encoder, looked up once through the runtime -- so that a launcher
/// needs no link against the driver's library -- or null where the driver
/// has none.  `cudaGetDriverEntryPointByVersion` from CUDA 12.5 on, which
/// names the version of the entry point (12.0, the first with tensor maps);
/// before it the unversioned call, which CUDA 13 deprecates.
inline TensorMapEncoder tensorMapEncoder() {
  static const TensorMapEncoder encoder = [] {
    void *fn = nullptr;
    cudaDriverEntryPointQueryResult found = cudaDriverEntryPointSymbolNotFound;
#if CUDART_VERSION >= 12050
    const cudaError_t status = cudaGetDriverEntryPointByVersion(
        "cuTensorMapEncodeTiled", &fn, 12000, cudaEnableDefault, &found);
#else
    const cudaError_t status = cudaGetDriverEntryPoint(
        "cuTensorMapEncodeTiled", &fn, cudaEnableDefault, &found);
#endif
    return status == cudaSuccess && found == cudaDriverEntryPointSuccess
               ? reinterpret_cast<TensorMapEncoder>(fn)
               : nullptr;
  }();
  return encoder;
}

/// The driver's name of an element a map holds (`engines.TMA_TYPES`).
template <typename T> struct TensorMapElement;
template <> struct TensorMapElement<float> {
  static constexpr CUtensorMapDataType value = CU_TENSOR_MAP_DATA_TYPE_FLOAT32;
};
template <> struct TensorMapElement<double> {
  static constexpr CUtensorMapDataType value = CU_TENSOR_MAP_DATA_TYPE_FLOAT64;
};
template <> struct TensorMapElement<int32_t> {
  static constexpr CUtensorMapDataType value = CU_TENSOR_MAP_DATA_TYPE_INT32;
};
template <> struct TensorMapElement<uint32_t> {
  static constexpr CUtensorMapDataType value = CU_TENSOR_MAP_DATA_TYPE_UINT32;
};
template <> struct TensorMapElement<int64_t> {
  static constexpr CUtensorMapDataType value = CU_TENSOR_MAP_DATA_TYPE_INT64;
};

/// The tensor map of the `Rank`-dimensional tensor of `T` at `base`, for the
/// copies of `tmaLoad` (CUDA's `cuTensorMapEncodeTiled`, tile mode): its
/// extents (`dims`), the elements from one index of each dimension to the
/// next but the innermost's (`strides`), the box a copy moves (`box`), the
/// swizzle of the box in shared memory (0, or 32, 64 or 128 bytes), and
/// whether what a box reads past the tensor's ends is NaN rather than zero.
/// Outermost dimension first, as TensorForge states them
/// (`engines.TensorMap`); the driver's order is the other one, and this
/// turns it around.
///
/// A map the driver refuses -- a base or a stride off 16 bytes, an extent
/// past its limits -- would have the copies read what it did not describe,
/// so the refusal ends the program, with what was asked.
template <typename T, int Rank>
inline CUtensorMap
tensorMapTiled(const T *base, const unsigned long long (&dims)[Rank],
               const unsigned long long (&strides)[Rank - 1],
               const unsigned (&box)[Rank], int swizzle, bool nanFill = false) {
  static_assert(Rank >= 2 && Rank <= 5, "a tensor map has 2 to 5 dimensions");
  cuuint64_t globalDim[Rank];
  cuuint64_t globalStrides[Rank - 1];
  cuuint32_t boxDim[Rank];
  cuuint32_t elementStrides[Rank];
  for (int d = 0; d < Rank; ++d) {
    globalDim[d] = dims[Rank - 1 - d];
    boxDim[d] = box[Rank - 1 - d];
    elementStrides[d] = 1;
  }
  for (int d = 0; d + 1 < Rank; ++d) {
    globalStrides[d] = strides[Rank - 2 - d] * sizeof(T);
  }
  const CUtensorMapSwizzle mode = swizzle == 128  ? CU_TENSOR_MAP_SWIZZLE_128B
                                  : swizzle == 64 ? CU_TENSOR_MAP_SWIZZLE_64B
                                  : swizzle == 32 ? CU_TENSOR_MAP_SWIZZLE_32B
                                                  : CU_TENSOR_MAP_SWIZZLE_NONE;
  CUtensorMap map{};
  const TensorMapEncoder encode = tensorMapEncoder();
  const CUresult status =
      encode == nullptr
          ? CUDA_ERROR_NOT_FOUND
          : encode(&map, TensorMapElement<T>::value, Rank,
                   const_cast<T *>(base), globalDim, globalStrides, boxDim,
                   elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE, mode,
                   CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                   nanFill ? CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA
                           : CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (status != CUDA_SUCCESS ||
      (swizzle != 0 && mode == CU_TENSOR_MAP_SWIZZLE_NONE)) {
    std::fprintf(stderr,
                 "tensorforge: no tensor map of %d dimensions over %p "
                 "(swizzle %d): the driver answered %d\n",
                 Rank, static_cast<const void *>(base), swizzle,
                 static_cast<int>(status));
    std::abort();
  }
  return map;
}

} // namespace tensorforge

#endif // SEISSOL_TENSORFORGE_INCLUDE_TENSORFORGE_DEVICE_CUDA_TMA_H_
