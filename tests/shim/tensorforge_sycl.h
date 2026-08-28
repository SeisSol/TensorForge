// SPDX-FileCopyrightText: 2026 SeisSol Group
//
// SPDX-License-Identifier: MIT
#ifndef SEISSOL_TESTS_SHIM_TENSORFORGE_SYCL_H_
#define SEISSOL_TESTS_SHIM_TENSORFORGE_SYCL_H_

// Declaration-only SYCL, for the same reason `tensorforge_host.h` is
// declaration-only CUDA/HIP: deciding whether the generated source is
// *well-formed* needs a C++ front end, not a device toolchain, and requiring
// oneAPI to answer that question would mean the answer is never asked in CI.
//
// The surface below is not "SYCL"; it is exactly the names the generator
// emits, which `tools/sycl_surface.py` enumerates from the corpus.  Anything
// outside that set is deliberately absent -- a shim that accepts more than
// the real header would make this check green for the wrong reason, which is
// the failure mode `tensorforge_host.h` already ran into once with
// `vector_size` on an alias template.
//
// This is a copy of a fact that lives in someone else's specification, and
// copies drift.  `tests/test_syntax.py` therefore treats a real front end
// (`icpx -fsycl -fsyntax-only`, when `TF_SYCL_CXX` names one) as the
// authority and this file as the always-available approximation.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>

// from include/tensorforge_device/base.h
constexpr std::int32_t operator"" _i32(unsigned long long value) {
  return static_cast<std::int32_t>(value);
}
constexpr std::int64_t operator"" _i64(unsigned long long value) {
  return static_cast<std::int64_t>(value);
}

namespace sycl {

// -- geometry --------------------------------------------------------------

template <int Dim> class range {
public:
  template <typename... Ts>
  range(Ts... vs) : _v{static_cast<std::size_t>(vs)...} {}
  std::size_t get(int i) const { return _v[i]; }
  std::size_t operator[](int i) const { return _v[i]; }

private:
  std::size_t _v[Dim > 0 ? Dim : 1]{};
};

template <int Dim> class id {
public:
  std::size_t operator[](int i) const { return _v[i]; }
  operator std::size_t() const { return _v[0]; }

private:
  std::size_t _v[Dim > 0 ? Dim : 1]{};
};

template <int Dim> class nd_range {
public:
  nd_range(range<Dim> global, range<Dim> local) : _g(global), _l(local) {}

private:
  range<Dim> _g, _l;
};

template <int Dim> class group {
public:
  std::size_t get_group_id(int) const { return 0; }
  std::size_t get_local_range(int) const { return 0; }
};

class sub_group {
public:
  id<1> get_local_id() const { return {}; }
};

template <int Dim> class nd_item {
public:
  std::size_t get_local_id(int) const { return 0; }
  std::size_t get_global_id(int) const { return 0; }
  std::size_t get_global_range(int) const { return 0; }
  group<Dim> get_group() const { return {}; }
  sub_group get_sub_group() const { return {}; }
  void barrier() const {}
};

// -- vectors ---------------------------------------------------------------
//
// A plain aggregate: the generator only ever bit-casts through these
// (`*(sycl::vec<float,4>*)&p[i] = *(sycl::vec<float,4>*)&q[j]`), so element
// access is not part of the contract and is deliberately not provided.

template <typename T, int N> struct vec {
  T data[N > 0 ? N : 1];
};

// -- accessors -------------------------------------------------------------

namespace access {
enum class mode { read, write, read_write };
enum class target { local, global_buffer };
} // namespace access

template <typename T, int Dim, access::mode Mode, access::target Target>
class accessor {
public:
  template <typename H> accessor(std::size_t, H &) {}
  T &operator[](std::size_t) const { return *_p; }

private:
  T *_p = nullptr;
};

template <typename T, int Dim> class local_accessor {
public:
  template <typename H> local_accessor(std::size_t, H &) {}
  T &operator[](std::size_t) const { return *_p; }

private:
  T *_p = nullptr;
};

// -- queue / handler -------------------------------------------------------

class handler {
public:
  template <int Dim, typename F> void parallel_for(nd_range<Dim>, F &&f) {
    (void)sizeof(f);
  }
};

class event {};

class queue {
public:
  template <typename F> event submit(F &&f) {
    (void)sizeof(f);
    return {};
  }
  void wait() {}
};

// -- collectives -----------------------------------------------------------

template <typename G, typename T> T group_broadcast(G, T x, std::size_t) {
  return x;
}

template <typename G> void group_barrier(G) {}

// -- math ------------------------------------------------------------------
//
// Templates rather than overloads: the generator instantiates these at
// float, double and __float128, and a missing overload set would report a
// generator defect that is not there.

template <typename T> T fabs(T x) { return x; }
template <typename T> T abs(T x) { return x; }
template <typename T> T sqrt(T x) { return x; }
template <typename T> T cbrt(T x) { return x; }
template <typename T> T exp(T x) { return x; }
template <typename T> T log(T x) { return x; }
template <typename T> T expm1(T x) { return x; }
template <typename T> T logp1(T x) { return x; }
template <typename T> T sin(T x) { return x; }
template <typename T> T cos(T x) { return x; }
template <typename T> T tan(T x) { return x; }
template <typename T> T asin(T x) { return x; }
template <typename T> T acos(T x) { return x; }
template <typename T> T atan(T x) { return x; }
template <typename T> T sinh(T x) { return x; }
template <typename T> T cosh(T x) { return x; }
template <typename T> T tanh(T x) { return x; }
template <typename T> T asinh(T x) { return x; }
template <typename T> T acosh(T x) { return x; }
template <typename T> T atanh(T x) { return x; }
template <typename T> T pow(T x, T) { return x; }
template <typename T> T min(T x, T y) { return x < y ? x : y; }
template <typename T> T max(T x, T y) { return x < y ? y : x; }

} // namespace sycl

// -- ESIMD -----------------------------------------------------------------
//
// `simd<T, N>` is a value type over N elements; `select<Size, Stride>(offset)`
// is a *reference* into it, which is why the generator can assign through it.
// Both properties are load-bearing for the ESIMD path and both are asserted
// in `tests/cpp/sycl_shim.cpp` rather than assumed here.

namespace tensorforge {

namespace intel_esimd {

template <typename T, int N> class simd;
template <int N> class simd_mask;

template <typename T, int N, int Size, int Stride> class simd_view {
public:
  explicit simd_view(T *base) : _base(base) {}
  simd_view &operator=(const simd_view &o) {
    _base[0] = o._base[0];
    return *this;
  }
  template <int S2, int St2>
  simd_view &operator=(const simd_view<T, N, S2, St2> &) {
    return *this;
  }
  simd_view &operator=(const simd<T, Size> &) { return *this; }
  simd_view &operator=(T v) {
    _base[0] = v;
    return *this;
  }
  simd_view &operator+=(const simd_view &) { return *this; }
  simd_view &operator+=(T) { return *this; }
  operator T() const { return _base[0]; }

private:
  T *_base;
};

template <typename T, int N> class simd {
public:
  simd() = default;
  explicit simd(T v) {
    for (int i = 0; i < N; ++i)
      _v[i] = v;
  }
  /// Linear progression `base, base+step, ...` -- `simd_obj_impl(Ty, Ty)` in
  /// the real header.  This is how a lane index is built when there is no
  /// thread id to ask.
  simd(T base, T step) {
    for (int i = 0; i < N; ++i)
      _v[i] = static_cast<T>(base + T(i) * step);
  }
  template <typename U> explicit simd(U *p) { (void)p; }

  template <int Size, int Stride = 1>
  simd_view<T, N, Size, Stride> select(int offset = 0) {
    return simd_view<T, N, Size, Stride>(&_v[offset % N]);
  }

  T &operator[](int i) { return _v[i]; }
  const T &operator[](int i) const { return _v[i]; }

  /// `merge(Val, Mask)`: take `Val`'s elements where the mask is set, keep
  /// this object's where it is not.  The vector form of a select, and the
  /// reason a predicated declaration is two statements rather than a ternary.
  template <int M> void merge(const simd &, const simd_mask<M> &) {}

  void copy_to(T *p) const { (void)p; }
  void copy_from(const T *p) { (void)p; }

  // Arithmetic takes a scalar on either side as well as another vector --
  // `v * 2.0f` is ordinary ESIMD and the generator emits it.  Templated on
  // the scalar type because the generator mixes `float` literals into
  // `simd<double, N>` expressions and the real API converts.
  simd operator+(const simd &) const { return *this; }
  simd operator-(const simd &) const { return *this; }
  simd operator*(const simd &) const { return *this; }
  simd operator/(const simd &) const { return *this; }
  template <typename U> simd operator+(U) const { return *this; }
  template <typename U> simd operator-(U) const { return *this; }
  template <typename U> simd operator*(U) const { return *this; }
  template <typename U> simd operator/(U) const { return *this; }
  simd operator-() const { return *this; }
  simd &operator+=(const simd &) { return *this; }

private:
  T _v[N > 0 ? N : 1]{};
};

// -- math ------------------------------------------------------------------
//
// Only what `sycl/ext/intel/esimd/math.hpp` actually declares.  A shim with a
// wider surface than the real header would make `test_syntax.py` accept a
// call that does not exist -- which is the whole class of defect this check
// is for, and the reason `sycl::tanh` is deliberately absent here too.

template <typename T, int N> simd<T, N> abs(simd<T, N> x) { return x; }
template <typename T, int N> simd<T, N> sqrt(simd<T, N> x) { return x; }
template <typename T, int N> simd<T, N> rsqrt(simd<T, N> x) { return x; }
template <typename T, int N> simd<T, N> inv(simd<T, N> x) { return x; }
template <typename T, int N> simd<T, N> exp(simd<T, N> x) { return x; }
template <typename T, int N> simd<T, N> log(simd<T, N> x) { return x; }
template <typename T, int N> simd<T, N> sin(simd<T, N> x) { return x; }
template <typename T, int N> simd<T, N> cos(simd<T, N> x) { return x; }
template <typename T, int N> simd<T, N> trunc(simd<T, N> x) { return x; }
template <typename T, int N> simd<T, N> min(simd<T, N> a, simd<T, N>) {
  return a;
}
template <typename T, int N> simd<T, N> max(simd<T, N> a, simd<T, N>) {
  return a;
}
template <typename T, int N> simd<T, N> pow(simd<T, N> a, simd<T, N>) {
  return a;
}
template <typename T, int N, typename U> simd<T, N> pow(simd<T, N> a, U) {
  return a;
}

/// `simd_mask<N>`: what a comparison over a `simd` yields, and the only thing
/// a predicated operation accepts.  A distinct family, as in the real header
/// -- it deliberately does *not* convert to `bool`, so that a mask reaching a
/// branch condition is a compile error here as well as there.
template <int N> class simd_mask {
public:
  simd_mask() = default;
  simd_mask operator&(const simd_mask &) const { return *this; }
  simd_mask operator|(const simd_mask &) const { return *this; }
  simd_mask operator!() const { return *this; }
};

template <typename T, int N, typename U>
simd_mask<N> operator<(const simd<T, N> &, U) {
  return {};
}
template <typename T, int N, typename U>
simd_mask<N> operator<=(const simd<T, N> &, U) {
  return {};
}
template <typename T, int N, typename U>
simd_mask<N> operator>(const simd<T, N> &, U) {
  return {};
}
template <typename T, int N, typename U>
simd_mask<N> operator>=(const simd<T, N> &, U) {
  return {};
}
template <typename T, int N, typename U>
simd_mask<N> operator==(const simd<T, N> &, U) {
  return {};
}
template <typename T, int N, typename U>
simd_mask<N> operator!=(const simd<T, N> &, U) {
  return {};
}

} // namespace intel_esimd

namespace intel_xmx {

/// `Result = C + B x A`, with the shape fixed by (SystolicDepth, RepeatCount)
/// and the element types.  Declaration only: what it computes needs a device,
/// what it *accepts* does not, and the second is what this check is for.
template <int SystolicDepth, int RepeatCount, typename T, typename CT,
          typename BT, typename AT, int N, int BN, int AN>
intel_esimd::simd<T, N> dpas(intel_esimd::simd<CT, N> C,
                             intel_esimd::simd<BT, BN> B,
                             intel_esimd::simd<AT, AN> A);

} // namespace intel_xmx

/// Stand-in for `sycl::ext::intel::experimental::esimd::tfloat32`: a distinct
/// type that converts from and to float, which is all the generator needs it
/// to be here.
class TF32 {
public:
  TF32() = default;
  TF32(float v) : _v(v) {}
  operator float() const { return _v; }

private:
  float _v = 0.0f;
};

} // namespace tensorforge

#endif // SEISSOL_TESTS_SHIM_TENSORFORGE_SYCL_H_
