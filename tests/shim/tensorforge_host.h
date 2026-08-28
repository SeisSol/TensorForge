// SPDX-FileCopyrightText: 2026 SeisSol Group
//
// SPDX-License-Identifier: MIT
#ifndef SEISSOL_TESTS_SHIM_TENSORFORGE_HOST_H_
#define SEISSOL_TESTS_SHIM_TENSORFORGE_HOST_H_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <tuple>

// --------------------------------------------------------------------------
// Execution-space keywords and the thread geometry.
// --------------------------------------------------------------------------

#define __global__
#define __device__
#define __host__
#define __forceinline__ inline
#define __shared__
#define __launch_bounds__(...)

struct dim3 {
  unsigned x, y, z;
  dim3(unsigned a = 1, unsigned b = 1, unsigned c = 1) : x(a), y(b), z(c) {}
};
extern dim3 threadIdx, blockIdx, blockDim, gridDim;

inline void __syncthreads() {}
inline void __threadfence() {}
inline void __threadfence_block() {}
inline void __syncwarp(unsigned = 0xffffffffu) {}

// from include/tensorforge_device/base.h
constexpr std::int32_t operator"" _i32(unsigned long long value) {
  return static_cast<std::int32_t>(value);
}
constexpr std::int64_t operator"" _i64(unsigned long long value) {
  return static_cast<std::int64_t>(value);
}

// --------------------------------------------------------------------------
// Vector types.
//
// GCC silently drops a `vector_size` attribute written on an alias template,
// which degrades `VectorT<float, 4>` to plain `float` -- and then every MFMA
// call type-checks for the wrong reason and the whole check is worthless
// while looking green.  The attribute therefore goes on a member typedef, and
// the property is asserted below rather than assumed.
// --------------------------------------------------------------------------

namespace tfshim {
template <typename T, std::size_t N> struct Vec {
  typedef T type __attribute__((__vector_size__(N * sizeof(T))));
};
} // namespace tfshim

// --------------------------------------------------------------------------
// Compiler builtins.  These have no declaration in any header -- they come
// from clang -- so the conformance test cannot check them and the signatures
// here are taken from the LLVM definitions.
// --------------------------------------------------------------------------

typedef tfshim::Vec<float, 4>::type __tfshim_v4f;

// (a, b, acc, cbsz, abid, blgp) -> acc
__tfshim_v4f __builtin_amdgcn_mfma_f32_4x4x1f32(float, float, __tfshim_v4f, int,
                                                int, int);

template <typename T> T __builtin_amdgcn_global_atomic_fadd_f32(T *, T);
template <typename T> T __builtin_amdgcn_global_atomic_fadd_f64(T *, T);
template <typename T> T __builtin_nontemporal_load(const T *);
template <typename T> void __builtin_nontemporal_store(T, T *);

// NVIDIA cache-control loads and the warp collectives.
template <typename T> T __ldg(const T *);
template <typename T> T __ldca(const T *);
template <typename T> T __ldcg(const T *);
template <typename T> T __ldcs(const T *);
template <typename T> void __stcg(T *, T);
template <typename T> void __stwt(T *, T);
template <typename T> T __shfl_sync(unsigned, T, int, int = 32);
template <typename T> T __shfl_xor_sync(unsigned, T, int, int = 32);
inline unsigned __ballot_sync(unsigned, int) { return 0; }
inline int __popc(unsigned) { return 0; }
inline float __uint_as_float(unsigned) { return 0.f; }
inline unsigned __float_as_uint(float) { return 0u; }
template <typename T> T atomicAdd(T *, T);

struct float2 {
  float x, y;
};
struct float4 {
  float x, y, z, w;
};
inline float2 make_float2(float a, float b) { return {a, b}; }
inline float4 make_float4(float a, float b, float c, float d) {
  return {a, b, c, d};
}

// --------------------------------------------------------------------------
// libcu++ pipeline surface, as used by the async-copy path.
// --------------------------------------------------------------------------

namespace cuda {
enum thread_scope {
  thread_scope_thread,
  thread_scope_block,
  thread_scope_device,
  thread_scope_system
};
template <thread_scope S> struct pipeline {
  void producer_acquire() {}
  void producer_commit() {}
  void consumer_wait() {}
  void consumer_release() {}
};
template <std::size_t A> struct aligned_size_t {
  std::size_t v;
  aligned_size_t(std::size_t n) : v(n) {}
  operator std::size_t() const { return v; }
};
inline pipeline<thread_scope_thread> make_pipeline() { return {}; }
template <typename D, typename S, typename Sz, thread_scope Sc>
void memcpy_async(D *, const S *, Sz, pipeline<Sc> &);
} // namespace cuda

// --------------------------------------------------------------------------
// The `__pipeline_*` primitives from <cuda_pipeline.h>.
//
// `cuda::pipeline` above is a wrapper over exactly these, and the structured
// copy path lowers to the primitives directly: no stage count fixed at
// compile time, no acquire/release bookkeeping, and `__pipeline_wait_prior`
// takes the number of outstanding groups to leave in flight rather than a
// number of stages.  Both surfaces are declared here because both are
// reachable -- a transfer that migrated uses the primitives, one that did not
// still drives the object.
// --------------------------------------------------------------------------

void __pipeline_memcpy_async(void *, const void *, std::size_t);
void __pipeline_commit();
void __pipeline_wait_prior(std::size_t);

namespace cooperative_groups {
struct grid_group {
  void sync() const {}
};
inline grid_group this_grid() { return {}; }
} // namespace cooperative_groups

// --------------------------------------------------------------------------
// include/tensorforge_device/{hip,cuda}.h
//
// Signatures only.  A body would drag in the DPP builtins that only an
// AMDGPU compiler has, and would be a second implementation of something
// that already exists.
// --------------------------------------------------------------------------

namespace tensorforge {

template <typename T, std::size_t N>
using VectorT = typename tfshim::Vec<T, N>::type;

constexpr std::size_t GlobalMemspace = 1;
constexpr std::size_t ConstantMemspace = 4;

// `address_space` is an AMDGPU attribute with no host equivalent.  Dropping
// it changes nothing this check can see -- the generated code only ever
// forms, indexes and passes these pointers.
template <typename T, std::size_t Space> using SpacePtr = T *;
template <typename T, std::size_t Space> using SpacePtrRestrict = T *__restrict;

template <typename T>
void transpose4x4b32(T &w1, T &w2, T &w3, T &w4, T v1, T v2, T v3, T v4);

template <typename T>
void transpose16x16b32(T &w1, T &w2, T &w3, T &w4, T &w5, T &w6, T &w7, T &w8,
                       T &w9, T &w10, T &w11, T &w12, T &w13, T &w14, T &w15,
                       T &w16);

template <typename T> void transpose16x2(T &w1, T &w2, T v1, T v2);

template <typename T>
void transpose16x4(T &w1, T &w2, T &w3, T &w4, T v1, T v2, T v3, T v4);

template <int Row> void fmacdpp4(float &c, float a, float b);
template <int Row> void fmacdpp16(float &c, float a, float b);
template <int Row> void fmacdpp16(double &c, double a, double b);
// The packed pair.  `codegen.py` reaches it only through the `bcst` path,
// which is switched off today -- so nothing in the corpus calls it, and a
// shim written from the corpus alone would not have it.  The header is the
// reference here, not the corpus: the moment `bcst` comes back the missing
// overload would be reported as a generator defect.
template <int Row> void fmacdpp16(float2 &c, float2 a, float2 b);

template <std::size_t Block, std::size_t Subblock, std::size_t Lane, typename T>
T broadcast(T value);

// The operator tag is a type, so `Operation` and `ReductionOperation` have to
// be here too -- a `reduction` declared over an opaque `typename Op` would
// accept `ReductionOperation<float, Operation::Nonsense>` and report nothing.
enum class Operation { Add, Mul, And, Or, Xor, Min, Max };

template <typename T, Operation OpT> struct ReductionOperation {
  static constexpr Operation Op = OpT;
  static T applyOperation(const T &a1, const T &a2);
  static T neutral();
};

template <typename Op, std::size_t Block, std::size_t Subblock, typename T>
T reduction(const T &value);

// Two overloads, not one template.  A single `template <int Row, typename T>
// T movdpp16(T)` would accept calls the runtime rejects -- `movdpp16<0>` on a
// `double` has no definition there -- and a shim that is more permissive than
// the header cannot report the failure it exists to report.
template <int Row> float2 movdpp16(float2 a);
template <int Row> float movdpp16(float a);

// cuda.h
void splitFloatTF32(std::uint32_t &upper, std::uint32_t &lower, float value);

} // namespace tensorforge

// The one property this file has already got wrong once.
static_assert(sizeof(tensorforge::VectorT<float, 4>) == 4 * sizeof(float),
              "VectorT lost its vector_size attribute: every check that "
              "passes through it is now meaningless");
static_assert(sizeof(tensorforge::VectorT<double, 2>) == 2 * sizeof(double),
              "VectorT lost its vector_size attribute");
#endif // SEISSOL_TESTS_SHIM_TENSORFORGE_HOST_H_
