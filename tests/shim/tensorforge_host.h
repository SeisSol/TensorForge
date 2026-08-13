// SPDX-License-Identifier: MIT
//
// A declaration-only stand-in for the device runtime, so that a plain host
// compiler can parse a generated kernel.
//
// Why this is here at all: nothing in this repository compiled, and that was a
// hole with a shape.  A padded MFMA tail block handed `0.0f` to a `T &`
// parameter -- ill-formed C++ that the snapshot corpus, the symbolic
// equivalence checker and the PIR verifier all passed, because none of them
// models overload resolution.  A real compiler does, and the intrinsic surface
// the generator actually emits is small enough to declare:
//
//     3248 __builtin_amdgcn_mfma_f32_4x4x1f32     4185 tensorforge::fmacdpp16
//        9 __builtin_amdgcn_global_atomic_fadd     200 tensorforge::broadcast
//                                                  126
//                                                  tensorforge::transpose4x4b32
//
// So `g++ -fsyntax-only` over the corpus is affordable, and it decides name
// lookup, overload resolution, reference binding and argument counts for
// *every* line the generator wrote --- not for the subset a regex could lift
// out.
//
// What this is not
// ----------------
// It is not a HIP or CUDA implementation and must never be included by
// anything that runs.  Nothing here has a body, the DPP controls are absent,
// and no statement about *semantics* follows from a file that compiles
// against it.  It lives under `tests/` rather than beside `hip.h` for exactly
// that reason: a shim in the shipped include directory is a shim that
// eventually gets included by accident.
//
// It also covers the `// === kernel ===` section only.  The launcher needs
// the host-side runtime API and, on CUDA, the `<<<>>>` launch syntax, which
// is not C++; stubbing that far would mean writing a fake runtime, and a fake
// runtime is a second place for a fact to live.  The launcher is therefore a
// known gap, named here rather than left to be discovered.
//
// Every declaration below that has a counterpart in
// `include/tensorforge_device/{hip,cuda}.h` is checked against it by
// `tests/test_syntax.py::test_shim_matches_the_device_headers`.  A copy of a
// C++ fact is a real cost; a checked copy is a seam.
#pragma once

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
