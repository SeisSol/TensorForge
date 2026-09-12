// SPDX-FileCopyrightText: 2026 SeisSol Group
//
// SPDX-License-Identifier: MIT
#ifndef SEISSOL_TENSORFORGE_INCLUDE_TENSORFORGE_DEVICE_ISYCL_H_
#define SEISSOL_TENSORFORGE_INCLUDE_TENSORFORGE_DEVICE_ISYCL_H_

#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/experimental/esimd/tfloat32.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <sycl/sycl.hpp>

#include <cstdint>

#include "base.h"

namespace tensorforge {
namespace intel_esimd = sycl::ext::intel::esimd;
namespace intel_xmx = intel_esimd::xmx;

/// The same 19-bit E8M10, and here it is a real type rather than a bit
/// pattern: `simd<float, N>` does not convert to `simd<tf32, N>` implicitly,
/// so a fragment staged with the wrong precision is a compile error.  On CUDA
/// the constraint letter forces a typedef; see `cuda.h`.
using tf32 = sycl::ext::intel::experimental::esimd::tfloat32;
/// Kept for the existing spelling in `isycl.h`'s own helpers.
using TF32 = tf32;

/// Ask a cache for the word at `ptr`, under the hints this API requires.
///
/// The hints are not optional here, which is the whole reason a helper exists.
/// `check_cache_hints` static-asserts that a prefetch names an L1 hint from
/// {cached, uncached, streaming} and an L2 hint from {cached, uncached}, and
/// refuses both uncached -- so the empty property list every other backend
/// gets away with is a compile error on this path, and which combinations are
/// legal is a rule worth stating once instead of at every call site.
///
/// One element, and one address. `prefetch(const T*, props)` is the block form
/// of the API; the gather forms take a vector of byte offsets with a mask
/// beside it, and neither is what a pointer chase wants -- the address is a
/// slot in an array of pointers and the next slot belongs to another element.
///
/// DG2 and PVC only, which the generator gates on rather than this: an earlier
/// Xe part has no LSC prefetch for the API to lower to.
///
/// `N` elements from that address, for a hint that covers an operand rather
/// than a pointer.  Up to 256 bytes that is the transposed form, one address
/// and a run of up to 64 dwords -- rounded up to a length the message takes.
/// Beyond, it is the gather form with one lane per 64-byte line: a line is
/// what a prefetch brings in whatever it names, so 32 lanes ask for 2 kB in
/// one message where the block form would take eight.  The offsets stop at
/// the run's last element -- the lanes past it ask for that line again, and
/// nothing past the run is touched -- and one lane more than whole lines
/// covers a run that does not start on one.
template <intel_esimd::cache_hint L1H, intel_esimd::cache_hint L2H, int N = 1,
          typename T>
ESIMD_INLINE void prefetchHinted(const T *ptr) {
  constexpr auto props = intel_esimd::properties{
      intel_esimd::cache_hint_L1<L1H>, intel_esimd::cache_hint_L2<L2H>};
  constexpr int bytes = N * static_cast<int>(sizeof(T));
  if constexpr (bytes <= 256) {
    constexpr int run = N <= 1    ? 1
                        : N <= 2  ? 2
                        : N <= 4  ? 4
                        : N <= 8  ? 8
                        : N <= 16 ? 16
                        : N <= 32 ? 32
                                  : 64;
    intel_esimd::prefetch<T, run>(ptr, props);
  } else {
    constexpr int lines = bytes / 64 + 1;
    static_assert(lines <= 32,
                  "one gather message holds 32 lines; split the run");
    constexpr int lanes = lines <= 8 ? 8 : lines <= 16 ? 16 : 32;
    intel_esimd::simd<std::uint32_t, lanes> offsets(0, 64);
    constexpr std::uint32_t last = bytes - static_cast<int>(sizeof(T));
    offsets.merge(intel_esimd::simd<std::uint32_t, lanes>(last),
                  offsets > last);
    intel_esimd::prefetch<T, lanes>(ptr, offsets, props);
  }
}

namespace detail {
template <typename P, typename... Ps>
ESIMD_INLINE const P *firstRun(const P *p, const Ps *...) {
  return p;
}

template <int Lane, int Lanes>
ESIMD_INLINE void fillRuns(intel_esimd::simd<std::uint64_t, Lanes> &,
                           std::uint64_t) {}

/// The lanes for one run and then the rest: a lane per 64-byte line of the
/// `B` bytes at `p`, the last clamped onto the run's last dword, as byte
/// offsets from `base`.  Unsigned, so a run below the base wraps and the sum
/// comes back to its address.
template <int Lane, int Lanes, int B, int... Rest, typename P, typename... Ps>
ESIMD_INLINE void fillRuns(intel_esimd::simd<std::uint64_t, Lanes> &offsets,
                           std::uint64_t base, const P *p, const Ps *...rest) {
  constexpr int lines = B / 64 + 1;
  constexpr std::uint32_t last = static_cast<std::uint32_t>((B - 1) & ~3);
  intel_esimd::simd<std::uint32_t, lines> within(0, 64);
  within.merge(intel_esimd::simd<std::uint32_t, lines>(last), within > last);
  offsets.template select<lines, 1>(Lane) =
      intel_esimd::simd<std::uint64_t, lines>(within) +
      (reinterpret_cast<std::uint64_t>(p) - base);
  fillRuns<Lane + lines, Lanes, Rest...>(offsets, base, rest...);
}
} // namespace detail

/// Several runs asked for in one gather: `Bytes[i]` from `ptrs[i]`, a lane per
/// line of each, as `prefetchHinted` does for one run.  Hints for different
/// operands are different addresses, which a block message cannot take and a
/// gather can -- so a body's hints for the next element are one message where
/// they were one per operand.  The lanes left over ask for the first run's
/// first line again.
template <intel_esimd::cache_hint L1H, intel_esimd::cache_hint L2H,
          int... Bytes, typename... P>
ESIMD_INLINE void prefetchRunsHinted(const P *...ptrs) {
  static_assert(sizeof...(Bytes) == sizeof...(P), "one length per run");
  constexpr int total = (0 + ... + (Bytes / 64 + 1));
  static_assert(total <= 32,
                "one gather message holds 32 lines; split the runs");
  constexpr int lanes = total <= 8 ? 8 : total <= 16 ? 16 : 32;
  const auto *first = detail::firstRun(ptrs...);
  const std::uint64_t base = reinterpret_cast<std::uint64_t>(first);
  intel_esimd::simd<std::uint64_t, lanes> offsets(0);
  detail::fillRuns<0, lanes, Bytes...>(offsets, base, ptrs...);
  intel_esimd::prefetch<std::uint32_t, lanes>(
      reinterpret_cast<const std::uint32_t *>(first), offsets,
      intel_esimd::properties{intel_esimd::cache_hint_L1<L1H>,
                              intel_esimd::cache_hint_L2<L2H>});
}

template <int... Bytes, typename... P>
ESIMD_INLINE void prefetchRunsL1(const P *...ptrs) {
  prefetchRunsHinted<intel_esimd::cache_hint::cached,
                     intel_esimd::cache_hint::cached, Bytes...>(ptrs...);
}

template <int... Bytes, typename... P>
ESIMD_INLINE void prefetchRunsL2(const P *...ptrs) {
  prefetchRunsHinted<intel_esimd::cache_hint::uncached,
                     intel_esimd::cache_hint::cached, Bytes...>(ptrs...);
}

/// Keep it near: cached at both levels.
template <int N = 1, typename T> ESIMD_INLINE void prefetchL1(const T *ptr) {
  prefetchHinted<intel_esimd::cache_hint::cached,
                 intel_esimd::cache_hint::cached, N>(ptr);
}

/// Keep it out of L1. A hint issued a whole loop body ahead of its use lands
/// in L1 long before anything wants it, and displaces what the current
/// iteration is reading to no purpose.
template <int N = 1, typename T> ESIMD_INLINE void prefetchL2(const T *ptr) {
  prefetchHinted<intel_esimd::cache_hint::uncached,
                 intel_esimd::cache_hint::cached, N>(ptr);
}

/// Split a vector of floats into the two TF32 halves a DPAS multiplies.
///
/// The same arrangement as `splitFloatTF32` in `cuda.h`, and it has to be:
/// both feed a three-term product whose error analysis is the split's and not
/// the instruction's.  TF32 keeps 11 mantissa bits against FP32's 24, so
/// `upper` holds the top 11 and `lower` the next 11 of what is left; the
/// remaining two fall below what the accumulator distinguishes.
///
/// A whole vector at a time, and the destinations are templates, because a
/// DPAS fragment is filled by *runs*: `Src1[k * N + n]` is `A(n, k)`, so the
/// sixteen lanes an operand load already returns land in sixteen consecutive
/// slots.  What arrives here is therefore `frag.select<16, 1>(k * 16)` -- a
/// view into a vector -- and a signature taking `simd<tf32, N> &` cannot bind
/// one.  Views are proxies and go by value.
template <int N, typename UpperT, typename LowerT, typename ValueT>
ESIMD_INLINE void splitFloatTF32(UpperT upper, LowerT lower, ValueT value) {
  // `N` is explicit at the call site rather than deduced: all three operands
  // can be views into larger fragments -- `Src2` is filled a repeat row at a
  // time out of an operand that is itself a run -- and a view carries the
  // *parent's* length in its type, so nothing here could deduce the run
  // width from it.
  const intel_esimd::simd<float, N> v(value);
  const intel_esimd::simd<tf32, N> hi(v);
  const intel_esimd::simd<float, N> hiF(hi);
  upper = hi;
  lower = intel_esimd::simd<tf32, N>(v - hiF);
}

/// A segmented all-reduce over a vector: each group of `Subblock` lanes ends
/// holding the reduction over the lanes that share its position in the group.
///
/// The same statement as `tensorforge::reduction` in `cuda.h`, whose body is
/// `for (i = Block/2; i >= Subblock; i >>= 1) x = Op(x, shfl_xor(x, i))` --
/// and it has to be the same, because a kernel that reduces differently on
/// two backends is two kernels.
///
/// The shuffle is a two-dimensional region here.  `shfl_xor(x, i)` pairs each
/// lane with the one `i` away, which within every block of `2i` swaps the two
/// halves: `select<Block / (2 * i), 2 * i, i, 1>(0)` is all the low halves and
/// `(i)` all the high ones.  Combining the two views and writing the result
/// back into both is one butterfly step over the whole vector at once.
///
/// `Subblock == Block` is the identity: each group is one lane wide, so there
/// is nothing to combine.  `Subblock == 1` collapses to a single value, which
/// the ESIMD `reduce`/`hmax`/`hmin` intrinsics do in one call -- the lexic
/// prefers those and only reaches here when a group is to be kept.
template <typename Op, int Block, int Subblock, typename T>
ESIMD_INLINE intel_esimd::simd<T, Block>
segmentedReduction(intel_esimd::simd<T, Block> v) {
  static_assert(Subblock >= 1 && Subblock <= Block,
                "the kept group cannot be wider than the reduced one");
  if constexpr (Block > Subblock) {
    constexpr int I = Block / 2;
    auto lo = v.template select<Block / (2 * I), 2 * I, I, 1>(0);
    auto hi = v.template select<Block / (2 * I), 2 * I, I, 1>(I);
    const intel_esimd::simd<T, Block / 2> combined =
        Op::applyOperation(lo.read(), hi.read());
    lo = combined;
    hi = combined;
    // Halving the *block* halves the stride: after this step lanes that
    // differ only in bit `I` agree, so the next pairing is over `Block / 2`.
    return segmentedReduction<Op, Block / 2, Subblock, T>(v);
  } else {
    return v;
  }
}

/// A position in the work-group's shared local memory, counted in elements.
///
/// Not a pointer, and it cannot be one.  SLM is a separate address space on
/// this hardware: the ESIMD block and scalar accessors take a *byte offset*
/// into the chunk `slm_init` reserved, and the stateless block access a raw
/// `T*` lowers to reads global memory.  So `s0 + i` and `s0[i]` -- which is
/// how every consumer in the generator addresses a staged tile -- have to
/// keep meaning what they meant while ceasing to be pointer arithmetic.
///
/// Elements rather than bytes, because that is the unit every address in the
/// generator is already in.  Converting at the access is one multiplication
/// stated once; converting at the binding would put `sizeof(T)` into every
/// offset the macro layer computes, where a single missed site is an address
/// that is wrong by a factor of four and compiles.
template <typename T> class SlmRef;

template <typename T> class SlmPtr {
public:
  SlmPtr() = default;
  explicit constexpr SlmPtr(std::uint32_t elements) : elements_(elements) {}

  constexpr std::uint32_t elements() const { return elements_; }
  constexpr std::uint32_t bytes() const {
    return elements_ * static_cast<std::uint32_t>(sizeof(T));
  }

  // Templated on the index type, because the addresses the generator builds
  // are whatever the expression that produced them was -- `size_t` out of
  // `get_local_id`, `int32_t` out of a literal.  A fixed `uint32_t` parameter
  // makes every one of them a narrowing conversion at the call, which is a
  // warning per access and, under `-Werror`, a build.
  template <typename I> constexpr SlmPtr operator+(I n) const {
    return SlmPtr(elements_ + static_cast<std::uint32_t>(n));
  }
  template <typename I> constexpr SlmRef<T> operator[](I n) const;

private:
  std::uint32_t elements_{0};
};

/// One element of SLM, as something an assignment and a read both work on.
///
/// The proxy exists so that `x = s0[i]` and `s0[i] = x` -- the two commonest
/// statements in a generated kernel by a wide margin -- need no special case
/// in the emitter.  A scalar access is the one shape where the pointer
/// spelling and the offset spelling can be made to coincide, and coinciding
/// is worth a proxy: the alternative is an emitter branch on the address
/// space at every element read.
template <typename T> class SlmRef {
public:
  explicit constexpr SlmRef(SlmPtr<T> at) : at_(at) {}

  operator T() const { return intel_esimd::slm_scalar_load<T>(at_.bytes()); }

  // `const`, so that the assignment binds to the prvalue `s0[i]` produces.
  const SlmRef &operator=(T value) const {
    intel_esimd::slm_scalar_store<T>(at_.bytes(), value);
    return *this;
  }
  const SlmRef &operator=(const SlmRef &other) const {
    return *this = static_cast<T>(other);
  }

private:
  SlmPtr<T> at_;
};

template <typename T>
template <typename I>
constexpr SlmRef<T> SlmPtr<T>::operator[](I n) const {
  return SlmRef<T>(*this + n);
}

/// Can `N` elements of `T` be one SLM block message?
///
/// A block access takes a power-of-two run within one message, and anything
/// else has to go as a gather -- which is why this is asked rather than
/// assumed: a staging tail is `length % num_threads` wide and owes nothing to
/// a power of two.  Getting it wrong is not a slower kernel, it is a
/// `static_assert` inside the API or, worse, a message that moves a different
/// number of elements than the caller believes.
template <typename T, int N> constexpr bool slmBlockable() {
  constexpr std::size_t bytes = N * sizeof(T);
  return bytes >= 4 && bytes <= 512 && (N & (N - 1)) == 0;
}

/// The alignment an SLM access may assume, stated once.
///
/// The tiles this addresses start wherever `ShrMemOpt` placed them --
/// `272 * threadIdx.y` is a real offset out of the allocator -- so the only
/// promise that holds is the element's own. The default an ESIMD block access
/// takes is the *vector's* alignment, which those offsets do not meet, and
/// the violation is a runtime one because the offset is a runtime value.
template <typename T>
inline constexpr auto slmAligned =
    intel_esimd::properties{intel_esimd::alignment<sizeof(T)>};

/// The run a block message can take first out of `N`: the largest power of
/// two that is a whole message (`slmBlockable`), or 0 where not even one
/// element is -- a sub-dword type, which only a gather moves.
///
/// Largest first, so that a run is as few messages as its binary digits: 24
/// floats are 16 + 8, 504 are four of 128 and then 64 + 32 + 16 + 8.  A
/// gather of the same run is one address per element, and a wide one is a
/// register of offsets on top -- which is what every staging tail and every
/// run past 512 bytes used to be.
template <typename T, int N> constexpr int slmChunk() {
  // `slmBlockable`'s rule, spelled on values: a template argument cannot be
  // the loop variable.
  int c = 1;
  while (2 * c <= N && 2 * c * sizeof(T) <= 512)
    c *= 2;
  return c * sizeof(T) >= 4 ? c : 0;
}

/// `N` consecutive elements out of SLM, as a vector.
template <typename T, int N>
ESIMD_INLINE intel_esimd::simd<T, N> slmLoad(SlmPtr<T> at) {
  constexpr int C = slmChunk<T, N>();
  if constexpr (C == N) {
    return intel_esimd::slm_block_load<T, N>(at.bytes(), slmAligned<T>);
  } else if constexpr (C == 0) {
    const intel_esimd::simd<std::uint32_t, N> offsets(
        at.bytes(), static_cast<std::uint32_t>(sizeof(T)));
    return intel_esimd::slm_gather<T, N>(offsets);
  } else {
    intel_esimd::simd<T, N> out;
    out.template select<C, 1>(0) =
        intel_esimd::slm_block_load<T, C>(at.bytes(), slmAligned<T>);
    out.template select<N - C, 1>(C) = slmLoad<T, N - C>(at + C);
    return out;
  }
}

/// The same run, written.
template <typename T, int N>
ESIMD_INLINE void slmStore(SlmPtr<T> at, intel_esimd::simd<T, N> value) {
  constexpr int C = slmChunk<T, N>();
  if constexpr (C == N) {
    intel_esimd::slm_block_store<T, N>(at.bytes(), value, slmAligned<T>);
  } else if constexpr (C == 0) {
    const intel_esimd::simd<std::uint32_t, N> offsets(
        at.bytes(), static_cast<std::uint32_t>(sizeof(T)));
    intel_esimd::slm_scatter<T, N>(offsets, value);
  } else {
    intel_esimd::slm_block_store<T, C>(
        at.bytes(), intel_esimd::simd<T, C>(value.template select<C, 1>(0)),
        slmAligned<T>);
    slmStore<T, N - C>(at + C, intel_esimd::simd<T, N - C>(
                                   value.template select<N - C, 1>(C)));
  }
}

/// Reserve the work-group's SLM chunk and hand back its base.
///
/// `Bytes` is a template argument because the API requires a compile-time
/// size, which is what made this the arena of choice over a `local_accessor`:
/// the accessor would have to be threaded to every access site as a second
/// operand, while a reserved chunk is addressed by offset alone.  The size is
/// a constant in the generated kernel -- `ShrMemOpt` fixes it before any body
/// is built -- so the requirement costs nothing here.
/// Reserve the work-group's SLM chunk, once per kernel.  `slm_init` may not
/// be called twice, and a kernel of several sections binds its arena once per
/// section -- so the reservation is the kernel's and the binding the
/// section's (`SlmPtr<T>(0)`).
template <std::size_t Bytes> ESIMD_INLINE void slmReserve() {
  intel_esimd::slm_init<static_cast<std::uint32_t>(Bytes)>();
}

template <std::size_t Bytes, typename T> ESIMD_INLINE SlmPtr<T> slmArena() {
  intel_esimd::slm_init<static_cast<std::uint32_t>(Bytes)>();
  return SlmPtr<T>(0);
}

} // namespace tensorforge
#endif // SEISSOL_TENSORFORGE_INCLUDE_TENSORFORGE_DEVICE_ISYCL_H_
