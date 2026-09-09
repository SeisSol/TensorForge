// SPDX-FileCopyrightText: 2026 SeisSol Group
//
// SPDX-License-Identifier: MIT
#ifndef SEISSOL_TENSORFORGE_INCLUDE_TENSORFORGE_DEVICE_ISYCL_H_
#define SEISSOL_TENSORFORGE_INCLUDE_TENSORFORGE_DEVICE_ISYCL_H_

#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/experimental/esimd/tfloat32.hpp>
#include <sycl/sycl.hpp>

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
template <intel_esimd::cache_hint L1H, intel_esimd::cache_hint L2H, typename T>
ESIMD_INLINE void prefetchHinted(const T *ptr) {
  intel_esimd::prefetch<T>(
      ptr, intel_esimd::properties{intel_esimd::cache_hint_L1<L1H>,
                                   intel_esimd::cache_hint_L2<L2H>});
}

/// Keep it near: cached at both levels.
template <typename T> ESIMD_INLINE void prefetchL1(const T *ptr) {
  prefetchHinted<intel_esimd::cache_hint::cached,
                 intel_esimd::cache_hint::cached>(ptr);
}

/// Keep it out of L1. A hint issued a whole loop body ahead of its use lands
/// in L1 long before anything wants it, and displaces what the current
/// iteration is reading to no purpose.
template <typename T> ESIMD_INLINE void prefetchL2(const T *ptr) {
  prefetchHinted<intel_esimd::cache_hint::uncached,
                 intel_esimd::cache_hint::cached>(ptr);
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

} // namespace tensorforge
#endif // SEISSOL_TENSORFORGE_INCLUDE_TENSORFORGE_DEVICE_ISYCL_H_
